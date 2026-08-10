//! Pad probes for intercepting buffers and events in the pipeline.
//!
//! Pad probes allow applications to:
//! - Inspect or modify buffers at any point in the pipeline
//! - Intercept events (downstream and upstream)
//! - Block data flow for safe dynamic reconfiguration
//! - Drop specific buffers or events
//!
//! Modeled after GStreamer's pad probe system.
//!
//! # Important: Callback Requirements
//!
//! Probe callbacks run **synchronously on the data path** — they execute
//! inline during buffer processing. Callbacks must be:
//!
//! - **Non-blocking**: No I/O, no sleeps, no mutex waits
//! - **Fast**: Keep processing time minimal (microseconds, not milliseconds)
//!
//! For expensive operations, send data to a separate task via a channel.
//!
//! A callback that panics is caught, logged at `error!`, and **removed** — an
//! observer does not get to end the run it is watching. The rest of the
//! pipeline carries on as if the probe had never been attached.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::probe::{ProbeType, ProbeReturn, ProbeData};
//!
//! // Add a buffer inspection probe
//! let probe_id = pipeline.add_probe(
//!     pad_ref,
//!     ProbeType::BUFFER,
//!     |data| {
//!         if let ProbeData::Buffer(buf) = data {
//!             println!("Buffer: {} bytes", buf.len());
//!         }
//!         ProbeReturn::Ok
//!     },
//! )?;
//!
//! // Remove probe when done
//! pipeline.remove_probe(probe_id)?;
//! ```

use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::buffer::Buffer;
use crate::event::Event;
use crate::pipeline::NodeId;

// ============================================================================
// Probe Types
// ============================================================================

/// What to intercept at a pad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProbeType(u32);

impl ProbeType {
    /// Intercept buffers.
    pub const BUFFER: Self = Self(0b0000_0001);
    /// Intercept downstream events (EOS, segment, tags).
    pub const EVENT_DOWN: Self = Self(0b0000_0010);
    /// Intercept upstream events (seek, reconfigure).
    pub const EVENT_UP: Self = Self(0b0000_0100);
    /// Intercept flush events specifically.
    pub const EVENT_FLUSH: Self = Self(0b0000_1000);
    /// Block data flow at this point (triggers callback once, then blocks).
    pub const BLOCK: Self = Self(0b0001_0000);
    /// Trigger callback when no data is flowing (idle).
    pub const IDLE: Self = Self(0b0010_0000);

    /// Create empty probe type (matches nothing).
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Check if this probe type includes another.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Union of probe types.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

impl std::ops::BitOr for ProbeType {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

/// Probe callback return value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeReturn {
    /// Pass the data through normally.
    Ok,
    /// Drop this buffer/event (don't forward downstream).
    Drop,
    /// Remove this probe after this invocation (one-shot).
    Remove,
    /// Data was handled by the probe (don't forward).
    Handled,
}

/// Data passed to a probe callback.
#[derive(Debug)]
pub enum ProbeData<'a> {
    /// A buffer flowing through the pad.
    Buffer(&'a Buffer),
    /// A mutable buffer (for modification probes).
    BufferMut(&'a mut Buffer),
    /// An event flowing through the pad.
    Event(&'a Event),
    /// The pad is idle (no data flowing).
    Idle,
}

/// Unique probe identifier for registration and removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProbeId(u64);

impl ProbeId {
    /// Generate a new unique probe ID.
    fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }
}

impl std::fmt::Display for ProbeId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "probe-{}", self.0)
    }
}

// ============================================================================
// PadRef
// ============================================================================

/// Reference to a specific pad on a node for probe attachment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PadRef {
    /// The node this pad belongs to.
    pub node_id: NodeId,
    /// The direction of the pad.
    pub direction: PadDirection,
    /// Pad index (0 for default pad).
    pub index: usize,
}

/// Direction of a pad.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadDirection {
    /// Output pad (data flows out).
    Source,
    /// Input pad (data flows in).
    Sink,
}

impl PadRef {
    /// Create a reference to the default source (output) pad of a node.
    pub fn src(node_id: NodeId) -> Self {
        Self {
            node_id,
            direction: PadDirection::Source,
            index: 0,
        }
    }

    /// Create a reference to the default sink (input) pad of a node.
    pub fn sink(node_id: NodeId) -> Self {
        Self {
            node_id,
            direction: PadDirection::Sink,
            index: 0,
        }
    }
}

// ============================================================================
// Probe Entry
// ============================================================================

/// A registered probe with its callback.
pub(crate) struct ProbeEntry {
    /// Unique probe identifier.
    pub id: ProbeId,
    /// What to intercept.
    pub probe_type: ProbeType,
    /// Callback function.
    pub callback: Box<dyn FnMut(ProbeData<'_>) -> ProbeReturn + Send>,
}

impl std::fmt::Debug for ProbeEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProbeEntry")
            .field("id", &self.id)
            .field("probe_type", &self.probe_type)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Probe Registry
// ============================================================================

/// Thread-safe registry of pad probes.
///
/// Probes are stored per-pad and invoked during data flow by the executor.
#[derive(Clone, Default)]
pub struct ProbeRegistry {
    inner: Arc<Mutex<ProbeRegistryInner>>,
}

#[derive(Default)]
struct ProbeRegistryInner {
    /// Probes indexed by pad reference.
    probes: HashMap<PadRef, Vec<ProbeEntry>>,
}

/// Run one probe callback, converting a panic into a request to remove it.
///
/// A probe is an observer. It has no business ending the run it is watching —
/// before this, a panicking callback unwound its way out through the element
/// task that happened to invoke it, killing that element and every sink below
/// it, and attributing the failure to an element that had done nothing wrong.
///
/// Removing the offender rather than retrying it puts the pipeline back the way
/// it was before the aid was attached, which is the least surprising thing a
/// broken observer can do. It is loud, not silent: the panic message carries the
/// callback's own source location.
///
/// `AssertUnwindSafe` is sound for the same reason it is in the executor's
/// `guard`: the callback is dropped rather than called again, so there is no
/// half-updated state left for anyone to observe.
fn guarded(pad: &PadRef, data: ProbeData<'_>, probe: &mut ProbeEntry) -> ProbeReturn {
    match std::panic::catch_unwind(AssertUnwindSafe(|| (probe.callback)(data))) {
        Ok(ret) => ret,
        Err(payload) => {
            tracing::error!(
                probe = ?probe.id,
                pad = ?pad,
                "probe callback panicked and has been removed: {}",
                crate::error::panic_message(payload.as_ref())
            );
            ProbeReturn::Remove
        }
    }
}

impl ProbeRegistry {
    /// Create a new empty probe registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Lock the registry, ignoring poisoning.
    ///
    /// A panic elsewhere must not turn this registry into a second failure that
    /// kills every element task that touches it. Callback panics are caught in
    /// [`guarded`] before they can poison anything, so a poisoned lock here only
    /// ever means someone else died holding it — and the probe map is a plain
    /// `HashMap` with no invariant that a panic could have left half-applied.
    fn lock(&self) -> std::sync::MutexGuard<'_, ProbeRegistryInner> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Add a probe to a pad.
    ///
    /// Returns the probe ID for later removal.
    #[must_use]
    pub fn add<F>(&self, pad: PadRef, probe_type: ProbeType, callback: F) -> ProbeId
    where
        F: FnMut(ProbeData<'_>) -> ProbeReturn + Send + 'static,
    {
        let id = ProbeId::next();
        let entry = ProbeEntry {
            id,
            probe_type,
            callback: Box::new(callback),
        };
        let mut inner = self.lock();
        inner.probes.entry(pad).or_default().push(entry);
        id
    }

    /// Remove a probe by ID.
    ///
    /// Returns `true` if the probe was found and removed.
    pub fn remove(&self, id: ProbeId) -> bool {
        let mut inner = self.lock();
        for probes in inner.probes.values_mut() {
            if let Some(pos) = probes.iter().position(|p| p.id == id) {
                probes.remove(pos);
                return true;
            }
        }
        false
    }

    /// Check if any probes are registered for a pad.
    pub fn has_probes(&self, pad: &PadRef) -> bool {
        let inner = self.lock();
        inner
            .probes
            .get(pad)
            .is_some_and(|probes| !probes.is_empty())
    }

    /// Invoke buffer probes for a pad.
    ///
    /// Returns the probe result that determines what happens to the buffer.
    pub fn invoke_buffer(&self, pad: &PadRef, buffer: &Buffer) -> ProbeReturn {
        let mut inner = self.lock();
        let Some(probes) = inner.probes.get_mut(pad) else {
            return ProbeReturn::Ok;
        };

        let mut to_remove = Vec::new();
        let mut result = ProbeReturn::Ok;

        for (i, probe) in probes.iter_mut().enumerate() {
            if probe.probe_type.contains(ProbeType::BUFFER) {
                match guarded(pad, ProbeData::Buffer(buffer), probe) {
                    ProbeReturn::Ok => continue,
                    ProbeReturn::Remove => {
                        to_remove.push(i);
                        continue;
                    }
                    ret @ (ProbeReturn::Drop | ProbeReturn::Handled) => {
                        result = ret;
                        break;
                    }
                }
            }
        }

        // Remove one-shot probes (reverse order to preserve indices)
        for i in to_remove.into_iter().rev() {
            probes.remove(i);
        }

        result
    }

    /// Invoke event probes for a pad.
    ///
    /// Returns the probe result that determines what happens to the event.
    pub fn invoke_event(&self, pad: &PadRef, event: &Event, downstream: bool) -> ProbeReturn {
        let probe_type = if event.is_serialized() {
            if downstream {
                ProbeType::EVENT_DOWN
            } else {
                ProbeType::EVENT_UP
            }
        } else {
            ProbeType::EVENT_FLUSH
        };

        let mut inner = self.lock();
        let Some(probes) = inner.probes.get_mut(pad) else {
            return ProbeReturn::Ok;
        };

        let mut to_remove = Vec::new();
        let mut result = ProbeReturn::Ok;

        for (i, probe) in probes.iter_mut().enumerate() {
            if probe.probe_type.contains(probe_type) {
                match guarded(pad, ProbeData::Event(event), probe) {
                    ProbeReturn::Ok => continue,
                    ProbeReturn::Remove => {
                        to_remove.push(i);
                        continue;
                    }
                    ret @ (ProbeReturn::Drop | ProbeReturn::Handled) => {
                        result = ret;
                        break;
                    }
                }
            }
        }

        for i in to_remove.into_iter().rev() {
            probes.remove(i);
        }

        result
    }

    /// Check if a pad has a BLOCK probe.
    ///
    /// When a pad has a BLOCK probe, data flow should be paused.
    pub fn is_blocked(&self, pad: &PadRef) -> bool {
        let inner = self.lock();
        inner.probes.get(pad).is_some_and(|probes| {
            probes
                .iter()
                .any(|p| p.probe_type.contains(ProbeType::BLOCK))
        })
    }

    /// Invoke IDLE probes for a pad (called when no data is flowing).
    pub fn invoke_idle(&self, pad: &PadRef) {
        let mut inner = self.lock();
        let Some(probes) = inner.probes.get_mut(pad) else {
            return;
        };

        let mut to_remove = Vec::new();

        for (i, probe) in probes.iter_mut().enumerate() {
            if probe.probe_type.contains(ProbeType::IDLE)
                && guarded(pad, ProbeData::Idle, probe) == ProbeReturn::Remove
            {
                to_remove.push(i);
            }
        }

        for i in to_remove.into_iter().rev() {
            probes.remove(i);
        }
    }

    /// Get the number of registered probes.
    pub fn len(&self) -> usize {
        let inner = self.lock();
        inner.probes.values().map(|v| v.len()).sum()
    }

    /// Check if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl std::fmt::Debug for ProbeRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let inner = self.lock();
        f.debug_struct("ProbeRegistry")
            .field(
                "total_probes",
                &inner.probes.values().map(|v| v.len()).sum::<usize>(),
            )
            .field("pads_with_probes", &inner.probes.len())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(64, 16).unwrap())
    }

    fn make_test_buffer() -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 16);
        Buffer::new(handle, Metadata::from_sequence(0))
    }

    fn make_pad_ref() -> PadRef {
        PadRef::src(NodeId(daggy::NodeIndex::new(0)))
    }

    #[test]
    fn test_probe_type_operations() {
        let buf_event = ProbeType::BUFFER | ProbeType::EVENT_DOWN;
        assert!(buf_event.contains(ProbeType::BUFFER));
        assert!(buf_event.contains(ProbeType::EVENT_DOWN));
        assert!(!buf_event.contains(ProbeType::BLOCK));
    }

    #[test]
    fn test_probe_registry_add_remove() {
        let registry = ProbeRegistry::new();
        let pad = make_pad_ref();

        let id = registry.add(pad, ProbeType::BUFFER, |_| ProbeReturn::Ok);
        assert!(registry.has_probes(&pad));
        assert_eq!(registry.len(), 1);

        assert!(registry.remove(id));
        assert!(!registry.has_probes(&pad));
        assert_eq!(registry.len(), 0);
    }

    #[test]
    fn test_probe_invoke_buffer_pass() {
        let registry = ProbeRegistry::new();
        let pad = make_pad_ref();
        let buffer = make_test_buffer();

        let _ = registry.add(pad, ProbeType::BUFFER, |_| ProbeReturn::Ok);

        let result = registry.invoke_buffer(&pad, &buffer);
        assert_eq!(result, ProbeReturn::Ok);
    }

    #[test]
    fn test_probe_invoke_buffer_drop() {
        let registry = ProbeRegistry::new();
        let pad = make_pad_ref();
        let buffer = make_test_buffer();

        let _ = registry.add(pad, ProbeType::BUFFER, |_| ProbeReturn::Drop);

        let result = registry.invoke_buffer(&pad, &buffer);
        assert_eq!(result, ProbeReturn::Drop);
    }

    #[test]
    fn test_probe_invoke_buffer_one_shot() {
        let registry = ProbeRegistry::new();
        let pad = make_pad_ref();
        let buffer = make_test_buffer();

        let _ = registry.add(pad, ProbeType::BUFFER, |_| ProbeReturn::Remove);

        // First invocation triggers and removes
        let result = registry.invoke_buffer(&pad, &buffer);
        assert_eq!(result, ProbeReturn::Ok); // Remove means pass but remove probe

        // Second invocation: probe is gone
        assert!(!registry.has_probes(&pad));
    }

    #[test]
    fn test_probe_buffer_inspection() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let registry = ProbeRegistry::new();
        let pad = make_pad_ref();

        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let _ = registry.add(pad, ProbeType::BUFFER, move |data| {
            if let ProbeData::Buffer(buf) = data {
                count_clone.fetch_add(buf.len(), Ordering::Relaxed);
            }
            ProbeReturn::Ok
        });

        let buffer = make_test_buffer();
        registry.invoke_buffer(&pad, &buffer);
        registry.invoke_buffer(&pad, &buffer);

        assert_eq!(count.load(Ordering::Relaxed), 32); // 16 * 2
    }

    #[test]
    fn test_probe_block_detection() {
        let registry = ProbeRegistry::new();
        let pad = make_pad_ref();

        assert!(!registry.is_blocked(&pad));

        let _ = registry.add(pad, ProbeType::BLOCK, |_| ProbeReturn::Ok);
        assert!(registry.is_blocked(&pad));
    }

    #[test]
    fn test_probe_no_probes_for_pad() {
        let registry = ProbeRegistry::new();
        let pad = make_pad_ref();
        let buffer = make_test_buffer();

        // No probes registered - should pass through
        let result = registry.invoke_buffer(&pad, &buffer);
        assert_eq!(result, ProbeReturn::Ok);
    }

    #[test]
    fn test_multiple_probes_on_same_pad() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let registry = ProbeRegistry::new();
        let pad = make_pad_ref();

        let count = Arc::new(AtomicUsize::new(0));
        let c1 = count.clone();
        let c2 = count.clone();

        let _ = registry.add(pad, ProbeType::BUFFER, move |_| {
            c1.fetch_add(1, Ordering::Relaxed);
            ProbeReturn::Ok
        });
        let _ = registry.add(pad, ProbeType::BUFFER, move |_| {
            c2.fetch_add(1, Ordering::Relaxed);
            ProbeReturn::Ok
        });

        let buffer = make_test_buffer();
        registry.invoke_buffer(&pad, &buffer);

        assert_eq!(count.load(Ordering::Relaxed), 2); // Both probes called
    }
}
