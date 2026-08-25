//! Caller-supplied frame buffers for libvpx's VP9 decoder (#198).
//!
//! By default libvpx decodes into its own allocations, and
//! [`VpxDecoder`](super::VpxDecoder) copies each plane out into an arena slot.
//! `vpx_codec_set_frame_buffer_functions` lets the caller supply those buffers
//! instead, so a decoded frame can ride the pipeline as
//! [`MemoryType::External`](crate::memory::MemoryType) with no copy at all —
//! the same shape `Dav1dDecoder` uses for its pictures.
//!
//! # VP9 only, and libvpx says so
//!
//! `vpx_decoder.h`: *"Currently this only works with VP9."* A VP8 context
//! answers `VPX_CODEC_INCAPABLE`, so the decoder keeps the packed copy there
//! and declares CPU-only caps rather than promising something it cannot do.
//!
//! # Two holders, one buffer
//!
//! The hard part is not the callbacks, it is the lifetime. libvpx releases a
//! buffer when *it* stops referencing it — but the pipeline may still be
//! holding the frame, and a `Buffer` outlives the `decode()` call that produced
//! it by design. So each pooled buffer carries **two** independent flags, and
//! is reusable only when both are clear:
//!
//! - `in_libvpx`, set by the get callback and cleared by the release callback;
//! - `in_pipeline`, set when a frame is emitted and cleared when the last
//!   `Buffer` referencing it drops.
//!
//! Handing a buffer back to libvpx while the pipeline still reads it would let
//! the decoder draw the next frame over one already on screen — the same class
//! of bug the dma-buf import path solves with `GPU_IN_FLIGHT`.
//!
//! # Why raw allocations rather than `Box<[u8]>`
//!
//! libvpx writes through the pointer while Rust holds shared references to the
//! owning struct. Deriving that pointer from a `&mut [u8]` would be an aliasing
//! violation; allocating with [`std::alloc`] and never forming a reference to
//! the payload keeps it clean.

use std::alloc::{Layout, alloc_zeroed, dealloc, realloc};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// libvpx's own floor: `VP9_MAXIMUM_REF_BUFFERS` + `VPX_MAXIMUM_WORK_BUFFERS`.
///
/// `vpx_decoder.h` says the application "may be required to pass in at least"
/// this many. We grow past it on demand — the pipeline's own retention is what
/// pushes it higher — but starting below would guarantee a stall.
pub(super) const MIN_FRAME_BUFFERS: usize = 8 + 8;

/// A ceiling that should never be reached, so that a leak shows up as an error
/// rather than as unbounded memory growth. Pipeline retention is structurally
/// bounded by link capacity plus `retained_buffers()`, so real graphs sit far
/// below this.
const MAX_FRAME_BUFFERS: usize = 128;

/// Alignment libvpx is happy with. It aligns internally and the header says
/// "The application does not need to align the allocated data", but matching a
/// cache line costs nothing and keeps plane starts tidy.
const ALIGN: usize = 64;

/// One frame buffer handed to libvpx.
pub(super) struct VpxFrameBuffer {
    ptr: *mut u8,
    len: usize,
    /// libvpx currently references this buffer.
    in_libvpx: AtomicBool,
    /// A pipeline `Buffer` currently references this buffer.
    in_pipeline: AtomicBool,
}

// SAFETY: the payload is only ever written by libvpx, and only while
// `in_libvpx` is set and `in_pipeline` is not — the pool refuses to hand a
// buffer out until both are clear, so a write and a read never overlap. The
// flags themselves are atomic, and every field is otherwise immutable.
unsafe impl Send for VpxFrameBuffer {}
unsafe impl Sync for VpxFrameBuffer {}

impl VpxFrameBuffer {
    fn layout(len: usize) -> Layout {
        Layout::from_size_align(len.max(1), ALIGN).expect("frame buffer layout")
    }

    fn new(len: usize) -> Option<Self> {
        // libvpx requires freshly allocated data to be zeroed.
        // SAFETY: a non-zero, correctly aligned layout.
        let ptr = unsafe { alloc_zeroed(Self::layout(len)) };
        (!ptr.is_null()).then_some(Self {
            ptr,
            len,
            in_libvpx: AtomicBool::new(false),
            in_pipeline: AtomicBool::new(false),
        })
    }

    /// Whether neither holder references this buffer.
    fn is_free(&self) -> bool {
        !self.in_libvpx.load(Ordering::Acquire) && !self.in_pipeline.load(Ordering::Acquire)
    }

    /// Grow in place to at least `len`, zeroing the new tail.
    ///
    /// Only ever called on a free buffer, so nothing can be reading it.
    fn grow(&mut self, len: usize) -> bool {
        if self.len >= len {
            return true;
        }
        // SAFETY: `ptr` came from `alloc_zeroed` with this layout, and the
        // buffer is free, so no other holder can observe the move.
        let grown = unsafe { realloc(self.ptr, Self::layout(self.len), len) };
        if grown.is_null() {
            return false;
        }
        // SAFETY: `realloc` leaves the tail uninitialized; libvpx wants zeros.
        unsafe { std::ptr::write_bytes(grown.add(self.len), 0, len - self.len) };
        self.ptr = grown;
        self.len = len;
        true
    }

    /// The span this buffer covers, for bounds-checking a decoded frame.
    pub(super) fn span(&self) -> (*const u8, usize) {
        (self.ptr as *const u8, self.len)
    }
}

impl Drop for VpxFrameBuffer {
    fn drop(&mut self) {
        // SAFETY: `ptr` came from `alloc_zeroed`/`realloc` with this layout and
        // is freed exactly once.
        unsafe { dealloc(self.ptr, Self::layout(self.len)) };
    }
}

/// Keeps a pooled buffer alive for as long as a pipeline `Buffer` references
/// it, and frees the pool slot when that reference drops.
///
/// This is the `ExternalSlot` owner: it is what makes the memory outlive the
/// decoder, and what tells the pool the frame is no longer on screen.
pub(super) struct PipelineHold(Arc<VpxFrameBuffer>);

impl Drop for PipelineHold {
    fn drop(&mut self) {
        self.0.in_pipeline.store(false, Ordering::Release);
    }
}

/// The buffers libvpx decodes into.
///
/// Held behind an `Arc` because the callbacks reach it through a raw
/// `cb_priv` pointer, and because an emitted frame's `PipelineHold` keeps its
/// own buffer alive independently of the pool's lifetime.
pub(super) struct VpxFramePool {
    /// Only mutated from the decode thread, inside libvpx's get callback.
    buffers: std::sync::Mutex<Vec<Arc<VpxFrameBuffer>>>,
    /// High-water mark, for the debug log on drop.
    peak: AtomicUsize,
}

impl VpxFramePool {
    pub(super) fn new() -> Arc<Self> {
        Arc::new(Self {
            buffers: std::sync::Mutex::new(Vec::with_capacity(MIN_FRAME_BUFFERS)),
            peak: AtomicUsize::new(0),
        })
    }

    /// How many buffers the pool holds.
    pub(super) fn len(&self) -> usize {
        self.buffers.lock().map(|b| b.len()).unwrap_or(0)
    }

    /// The largest the pool ever grew.
    pub(super) fn peak(&self) -> usize {
        self.peak.load(Ordering::Relaxed)
    }

    /// Claim a free buffer of at least `min_size`, growing the pool if needed.
    fn acquire(&self, min_size: usize) -> Option<Arc<VpxFrameBuffer>> {
        let mut buffers = self.buffers.lock().ok()?;

        // Prefer a free buffer that is already big enough: no allocation, and
        // no re-zeroing — libvpx's own reference implementation only zeroes
        // when it (re)allocates, and a recycled buffer is fully overwritten by
        // the decode that claims it.
        if let Some(buf) = buffers
            .iter()
            .find(|b| b.is_free() && b.len >= min_size)
            .cloned()
        {
            buf.in_libvpx.store(true, Ordering::Release);
            return Some(buf);
        }

        // Otherwise grow a free-but-too-small one in place. `get_mut` succeeds
        // only when this is the pool's sole reference, which is exactly the
        // case where no emitted frame can still be reading it.
        if let Some(index) = buffers
            .iter()
            .position(|b| b.is_free() && Arc::strong_count(b) == 1)
            && let Some(buf) = Arc::get_mut(&mut buffers[index])
            && buf.grow(min_size)
        {
            let buf = buffers[index].clone();
            buf.in_libvpx.store(true, Ordering::Release);
            return Some(buf);
        }

        if buffers.len() >= MAX_FRAME_BUFFERS {
            return None;
        }
        let buf = Arc::new(VpxFrameBuffer::new(min_size)?);
        buf.in_libvpx.store(true, Ordering::Release);
        buffers.push(buf.clone());
        self.peak.fetch_max(buffers.len(), Ordering::Relaxed);
        Some(buf)
    }

    /// Mark a buffer as riding the pipeline, and take the owner that will
    /// release it. `None` if the pointer is not one of ours.
    pub(super) fn claim_for_pipeline(
        &self,
        fb_priv: *mut std::ffi::c_void,
    ) -> Option<PipelineHold> {
        let buffers = self.buffers.lock().ok()?;
        let buf = buffers
            .iter()
            .find(|b| Arc::as_ptr(b) as *const std::ffi::c_void == fb_priv.cast_const())?
            .clone();
        buf.in_pipeline.store(true, Ordering::Release);
        Some(PipelineHold(buf))
    }

    /// The buffer a decoded image lives in, for bounds-checking its planes.
    pub(super) fn find(&self, fb_priv: *mut std::ffi::c_void) -> Option<Arc<VpxFrameBuffer>> {
        let buffers = self.buffers.lock().ok()?;
        buffers
            .iter()
            .find(|b| Arc::as_ptr(b) as *const std::ffi::c_void == fb_priv.cast_const())
            .cloned()
    }

    fn release(&self, fb_priv: *mut std::ffi::c_void) {
        let Ok(buffers) = self.buffers.lock() else {
            return;
        };
        if let Some(buf) = buffers
            .iter()
            .find(|b| Arc::as_ptr(b) as *const std::ffi::c_void == fb_priv.cast_const())
        {
            buf.in_libvpx.store(false, Ordering::Release);
        }
    }
}

impl Drop for VpxFramePool {
    fn drop(&mut self) {
        // The pool's size is the one number that says whether the frame-buffer
        // path behaved: libvpx's own floor is 16, and anything far above that
        // means the pipeline was holding frames longer than its budget said.
        tracing::debug!(
            "vpxdecoder: frame-buffer pool held {} buffers (peak {})",
            self.len(),
            self.peak()
        );
    }
}

/// libvpx's get-frame-buffer callback.
///
/// # Safety
///
/// `cb_priv` must be a live `*const VpxFramePool` (the decoder holds the `Arc`
/// and destroys its codec context before dropping it), and `fb` a valid
/// `vpx_codec_frame_buffer_t` — both guaranteed by libvpx's contract.
pub(super) unsafe extern "C" fn get_frame_buffer(
    cb_priv: *mut std::ffi::c_void,
    min_size: usize,
    fb: *mut vpx_sys::vpx_codec_frame_buffer_t,
) -> std::os::raw::c_int {
    if cb_priv.is_null() || fb.is_null() {
        return -1;
    }
    // SAFETY: see the function contract.
    let pool = unsafe { &*(cb_priv as *const VpxFramePool) };
    let Some(buf) = pool.acquire(min_size) else {
        // libvpx turns this into a decode error, which the element reports.
        // Better than a silent stall.
        return -1;
    };
    let (ptr, len) = buf.span();
    // SAFETY: `fb` is a valid out-parameter for the duration of this call.
    unsafe {
        (*fb).data = ptr as *mut u8;
        (*fb).size = len;
        // Not an owning handle: the pool's Vec keeps the Arc alive, and this
        // identifies the buffer in the release callback and in `img.fb_priv`.
        (*fb).priv_ = Arc::as_ptr(&buf) as *mut std::ffi::c_void;
    }
    0
}

/// libvpx's release-frame-buffer callback.
///
/// # Safety
///
/// Same contract as [`get_frame_buffer`].
pub(super) unsafe extern "C" fn release_frame_buffer(
    cb_priv: *mut std::ffi::c_void,
    fb: *mut vpx_sys::vpx_codec_frame_buffer_t,
) -> std::os::raw::c_int {
    if cb_priv.is_null() || fb.is_null() {
        return -1;
    }
    // SAFETY: see the function contract.
    let pool = unsafe { &*(cb_priv as *const VpxFramePool) };
    // SAFETY: `fb` is valid for this call.
    let fb_priv = unsafe { (*fb).priv_ };
    pool.release(fb_priv);
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn priv_of(buf: &Arc<VpxFrameBuffer>) -> *mut std::ffi::c_void {
        Arc::as_ptr(buf) as *mut std::ffi::c_void
    }

    #[test]
    fn a_fresh_buffer_is_zeroed_because_libvpx_requires_it() {
        let pool = VpxFramePool::new();
        let buf = pool.acquire(4096).expect("acquire");
        let (ptr, len) = buf.span();
        // SAFETY: the buffer is ours and alive.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn a_buffer_libvpx_still_holds_is_never_handed_out_again() {
        let pool = VpxFramePool::new();
        let first = pool.acquire(1024).expect("first");
        let second = pool.acquire(1024).expect("second");
        assert_ne!(
            first.span().0,
            second.span().0,
            "two live buffers, two allocations"
        );
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn releasing_from_libvpx_alone_does_not_free_a_frame_the_pipeline_holds() {
        let pool = VpxFramePool::new();
        let buf = pool.acquire(1024).expect("acquire");
        let ptr = buf.span().0;

        let hold = pool.claim_for_pipeline(priv_of(&buf)).expect("claim");
        pool.release(priv_of(&buf)); // libvpx is done...

        // ...but the pipeline is not, so the pool must not reuse it.
        let next = pool.acquire(1024).expect("acquire again");
        assert_ne!(
            next.span().0,
            ptr,
            "a frame still on screen must not be redecoded into"
        );
        assert_eq!(pool.len(), 2);

        drop(hold);
    }

    #[test]
    fn a_buffer_comes_back_only_when_both_holders_are_done() {
        let pool = VpxFramePool::new();
        let buf = pool.acquire(1024).expect("acquire");
        let ptr = buf.span().0;
        let hold = pool.claim_for_pipeline(priv_of(&buf)).expect("claim");
        let fb_priv = priv_of(&buf);
        drop(buf);

        pool.release(fb_priv);
        drop(hold);

        let reused = pool.acquire(1024).expect("reuse");
        assert_eq!(
            reused.span().0,
            ptr,
            "both holders done, so the slot recycles"
        );
        assert_eq!(pool.len(), 1, "and no second allocation was needed");
    }

    #[test]
    fn the_pipeline_alone_also_pins_a_buffer() {
        let pool = VpxFramePool::new();
        let buf = pool.acquire(1024).expect("acquire");
        let fb_priv = priv_of(&buf);
        let hold = pool.claim_for_pipeline(fb_priv).expect("claim");
        pool.release(fb_priv);
        drop(buf);

        // in_libvpx is clear, in_pipeline is not.
        assert_ne!(pool.acquire(1024).expect("acquire").span().0, {
            let _ = &hold;
            pool.find(fb_priv).expect("still pooled").span().0
        });
    }

    #[test]
    fn a_larger_frame_grows_a_free_buffer_rather_than_allocating_beside_it() {
        let pool = VpxFramePool::new();
        let small = pool.acquire(1024).expect("small");
        let fb_priv = priv_of(&small);
        assert_eq!(small.span().1, 1024);
        drop(small);
        pool.release(fb_priv);

        let big = pool.acquire(64 * 1024).expect("big");
        assert!(big.span().1 >= 64 * 1024);
        assert_eq!(
            pool.len(),
            1,
            "a resolution change must not double the pool"
        );

        // And the grown tail is zeroed, as libvpx requires of new memory.
        let (ptr, len) = big.span();
        // SAFETY: the buffer is ours and alive.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, len) };
        assert!(bytes[1024..].iter().all(|&b| b == 0));
    }

    #[test]
    fn an_unknown_pointer_claims_nothing_rather_than_panicking() {
        let pool = VpxFramePool::new();
        let _ = pool.acquire(1024).expect("acquire");
        let stray = 0xdead_beefusize as *mut std::ffi::c_void;
        assert!(pool.claim_for_pipeline(stray).is_none());
        assert!(pool.find(stray).is_none());
        pool.release(stray); // must not panic
    }

    #[test]
    fn a_runaway_pool_stops_growing_instead_of_eating_memory() {
        let pool = VpxFramePool::new();
        let held: Vec<_> = (0..MAX_FRAME_BUFFERS)
            .map(|_| pool.acquire(64).expect("within the cap"))
            .collect();
        assert_eq!(pool.len(), MAX_FRAME_BUFFERS);
        assert!(
            pool.acquire(64).is_none(),
            "past the cap the callback must fail the decode, not allocate forever"
        );
        assert_eq!(pool.peak(), MAX_FRAME_BUFFERS);
        drop(held);
    }

    #[test]
    fn the_callbacks_round_trip_through_libvpx_s_own_struct() {
        let pool = VpxFramePool::new();
        let raw = Arc::as_ptr(&pool) as *mut std::ffi::c_void;
        let mut fb: vpx_sys::vpx_codec_frame_buffer_t = unsafe { std::mem::zeroed() };

        assert_eq!(unsafe { get_frame_buffer(raw, 2048, &mut fb) }, 0);
        assert!(!fb.data.is_null());
        assert!(fb.size >= 2048);
        assert_eq!(pool.len(), 1);

        // A second get while the first is held allocates a second buffer.
        let mut fb2: vpx_sys::vpx_codec_frame_buffer_t = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { get_frame_buffer(raw, 2048, &mut fb2) }, 0);
        assert_ne!(fb.data, fb2.data);

        // Releasing the first lets it come back.
        assert_eq!(unsafe { release_frame_buffer(raw, &mut fb) }, 0);
        let mut fb3: vpx_sys::vpx_codec_frame_buffer_t = unsafe { std::mem::zeroed() };
        assert_eq!(unsafe { get_frame_buffer(raw, 2048, &mut fb3) }, 0);
        assert_eq!(fb3.data, fb.data);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn a_null_argument_is_refused_rather_than_dereferenced() {
        let pool = VpxFramePool::new();
        let raw = Arc::as_ptr(&pool) as *mut std::ffi::c_void;
        let mut fb: vpx_sys::vpx_codec_frame_buffer_t = unsafe { std::mem::zeroed() };
        assert_eq!(
            unsafe { get_frame_buffer(std::ptr::null_mut(), 16, &mut fb) },
            -1
        );
        assert_eq!(
            unsafe { get_frame_buffer(raw, 16, std::ptr::null_mut()) },
            -1
        );
        assert_eq!(
            unsafe { release_frame_buffer(std::ptr::null_mut(), &mut fb) },
            -1
        );
    }
}
