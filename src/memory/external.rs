//! Producer-owned external memory pinned into the pipeline (#194).
//!
//! An [`ExternalSlot`] wraps memory the pipeline does not own — typically a
//! codec's own frame allocation, like a refcounted `dav1d::Picture` — so a
//! decoder can emit it directly instead of de-stride-copying every frame
//! into an arena slot. The slot keeps the producer's allocation alive by
//! owning an opaque handle to it; dropping the last `Arc<ExternalSlot>`
//! clone drops that handle (releasing the producer's refcount) and fires
//! the optional release hook exactly once.
//!
//! External memory is read-only, never IPC-shareable, and its byte layout
//! is producer-defined: strided video planes are described by the buffer's
//! `Metadata` plane layout, not by the raw byte span. That is why
//! `MemoryType::External` requires explicit consumer opt-in in caps
//! negotiation — a byte-reading consumer would silently misinterpret it.

use std::any::Any;

/// Release hook: fired exactly once when the last reference to an
/// [`ExternalSlot`] drops, *after* the owner handle itself is still alive
/// (the hook runs from `Drop`, the owner is dropped with the struct).
/// Producers use it for pool recycling; tests to observe release.
pub type ExternalReleaseHook = Box<dyn Fn() + Send + Sync>;

/// A refcounted slice of producer-owned memory — the External counterpart
/// of [`DmaBufSlot`](crate::memory::DmaBufSlot)'s discipline: cloning a
/// buffer clones an `Arc<ExternalSlot>`, and the last drop releases the
/// producer's allocation by dropping the owner handle.
pub struct ExternalSlot {
    ptr: *const u8,
    len: usize,
    /// Whatever keeps `ptr..ptr+len` alive — e.g. a cloned
    /// `dav1d::Picture`. Dropping it IS the release.
    _owner: Box<dyn Any + Send + Sync>,
    /// Fired exactly once, from `Drop`.
    release: Option<ExternalReleaseHook>,
    /// Producer tag for `Debug` ("dav1d-picture", "external-test-src").
    label: &'static str,
}

// SAFETY: the raw pointer targets memory kept alive by `_owner`, which is
// itself `Send + Sync`; the slot only ever reads through it.
unsafe impl Send for ExternalSlot {}
unsafe impl Sync for ExternalSlot {}

impl ExternalSlot {
    /// Wrap producer-owned memory.
    ///
    /// # Safety
    ///
    /// `ptr..ptr + len` must remain valid, initialized, immutable, and
    /// unaliased-for-write for as long as `owner` is alive. The owner must
    /// really be what keeps that span alive (a refcount clone, a pool
    /// guard) — not a copy of the data.
    pub unsafe fn new(
        ptr: *const u8,
        len: usize,
        owner: impl Any + Send + Sync,
        label: &'static str,
    ) -> Self {
        Self {
            ptr,
            len,
            _owner: Box::new(owner),
            release: None,
            label,
        }
    }

    /// Like [`new`](Self::new), plus a hook fired exactly once on last
    /// drop (pool recycling).
    ///
    /// # Safety
    ///
    /// Same contract as [`new`](Self::new).
    pub unsafe fn with_release(
        ptr: *const u8,
        len: usize,
        owner: impl Any + Send + Sync,
        label: &'static str,
        hook: ExternalReleaseHook,
    ) -> Self {
        // SAFETY: forwarded contract.
        let mut slot = unsafe { Self::new(ptr, len, owner, label) };
        slot.release = Some(hook);
        slot
    }

    /// The full external span.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `new`'s contract — valid + initialized while `_owner`
        // lives, and `_owner` lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Base pointer of the span.
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Length of the span in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the span is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for ExternalSlot {
    fn drop(&mut self) {
        if let Some(hook) = self.release.take() {
            hook();
        }
    }
}

impl std::fmt::Debug for ExternalSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ExternalSlot")
            .field("ptr", &self.ptr)
            .field("len", &self.len)
            .field("label", &self.label)
            .field("has_release", &self.release.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Owner whose drop is observable — stands in for a `dav1d::Picture`.
    struct CountingOwner(Arc<AtomicUsize>);
    impl Drop for CountingOwner {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn owner_and_hook_release_exactly_once_on_last_drop() {
        let owner_drops = Arc::new(AtomicUsize::new(0));
        let hook_fires = Arc::new(AtomicUsize::new(0));
        let data: Box<[u8]> = Box::new([7u8; 64]);

        let slot = {
            let hook_fires = hook_fires.clone();
            // SAFETY: `data` outlives the slot in this test (owner is a
            // stand-in; the span is `data`'s).
            unsafe {
                ExternalSlot::with_release(
                    data.as_ptr(),
                    data.len(),
                    CountingOwner(owner_drops.clone()),
                    "test",
                    Box::new(move || {
                        hook_fires.fetch_add(1, Ordering::SeqCst);
                    }),
                )
            }
        };
        let shared = Arc::new(slot);
        let clone_a = shared.clone();
        let clone_b = shared.clone();

        assert_eq!(shared.as_slice()[0], 7);
        drop(shared);
        drop(clone_a);
        assert_eq!(owner_drops.load(Ordering::SeqCst), 0, "still referenced");
        assert_eq!(hook_fires.load(Ordering::SeqCst), 0);

        drop(clone_b);
        assert_eq!(owner_drops.load(Ordering::SeqCst), 1, "owner released once");
        assert_eq!(hook_fires.load(Ordering::SeqCst), 1, "hook fired once");
    }

    #[test]
    fn slice_reads_the_producer_bytes() {
        let data: Vec<u8> = (0..32).collect();
        // SAFETY: `data` outlives `slot`.
        let slot = unsafe { ExternalSlot::new(data.as_ptr(), data.len(), (), "test") };
        assert_eq!(slot.len(), 32);
        assert!(!slot.is_empty());
        assert_eq!(slot.as_slice(), &data[..]);
        let dbg = format!("{slot:?}");
        assert!(dbg.contains("test"), "{dbg}");
        drop(slot);
        drop(data);
    }
}
