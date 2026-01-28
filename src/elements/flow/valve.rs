//! Valve element for controlling buffer flow.
//!
//! A simple on/off switch for buffer flow through a pipeline.

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// A valve element that can drop or pass buffers.
///
/// When the valve is open, buffers pass through unchanged.
/// When closed, buffers are dropped.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Valve;
///
/// let valve = Valve::new();
/// let control = valve.control();
///
/// // Later, close the valve to drop buffers
/// control.close();
///
/// // Re-open to pass buffers again
/// control.open();
/// ```
pub struct Valve {
    name: String,
    inner: Arc<ValveInner>,
}

struct ValveInner {
    open: AtomicBool,
    passed: AtomicU64,
    dropped: AtomicU64,
}

/// Control handle for a Valve.
///
/// Can be cloned and sent to other threads.
#[derive(Clone)]
pub struct ValveControl {
    inner: Arc<ValveInner>,
}

impl Valve {
    /// Create a new valve (open by default).
    pub fn new() -> Self {
        Self {
            name: "valve".to_string(),
            inner: Arc::new(ValveInner {
                open: AtomicBool::new(true),
                passed: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
            }),
        }
    }

    /// Create a new valve with initial state.
    pub fn with_state(open: bool) -> Self {
        Self {
            name: "valve".to_string(),
            inner: Arc::new(ValveInner {
                open: AtomicBool::new(open),
                passed: AtomicU64::new(0),
                dropped: AtomicU64::new(0),
            }),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get a control handle for this valve.
    pub fn control(&self) -> ValveControl {
        ValveControl {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Check if the valve is open.
    pub fn is_open(&self) -> bool {
        self.inner.open.load(Ordering::SeqCst)
    }

    /// Open the valve.
    pub fn open(&self) {
        self.inner.open.store(true, Ordering::SeqCst);
    }

    /// Close the valve.
    pub fn close(&self) {
        self.inner.open.store(false, Ordering::SeqCst);
    }

    /// Get statistics.
    pub fn stats(&self) -> ValveStats {
        ValveStats {
            is_open: self.inner.open.load(Ordering::SeqCst),
            passed: self.inner.passed.load(Ordering::SeqCst),
            dropped: self.inner.dropped.load(Ordering::SeqCst),
        }
    }
}

impl Default for Valve {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Valve {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        if self.inner.open.load(Ordering::SeqCst) {
            self.inner.passed.fetch_add(1, Ordering::SeqCst);
            Ok(Some(buffer))
        } else {
            self.inner.dropped.fetch_add(1, Ordering::SeqCst);
            Ok(None)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl ValveControl {
    /// Check if the valve is open.
    pub fn is_open(&self) -> bool {
        self.inner.open.load(Ordering::SeqCst)
    }

    /// Open the valve.
    pub fn open(&self) {
        self.inner.open.store(true, Ordering::SeqCst);
    }

    /// Close the valve.
    pub fn close(&self) {
        self.inner.open.store(false, Ordering::SeqCst);
    }

    /// Toggle the valve state.
    pub fn toggle(&self) -> bool {
        let was_open = self.inner.open.fetch_xor(true, Ordering::SeqCst);
        !was_open // Return new state
    }

    /// Get statistics.
    pub fn stats(&self) -> ValveStats {
        ValveStats {
            is_open: self.inner.open.load(Ordering::SeqCst),
            passed: self.inner.passed.load(Ordering::SeqCst),
            dropped: self.inner.dropped.load(Ordering::SeqCst),
        }
    }
}

/// Statistics about valve operation.
#[derive(Debug, Clone, Copy)]
pub struct ValveStats {
    /// Whether the valve is currently open.
    pub is_open: bool,
    /// Total buffers passed through.
    pub passed: u64,
    /// Total buffers dropped.
    pub dropped: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(128, 64).unwrap())
    }

    fn create_test_buffer(seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 100);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn test_valve_default_open() {
        let valve = Valve::new();
        assert!(valve.is_open());
    }

    #[test]
    fn test_valve_pass_when_open() {
        let mut valve = Valve::new();

        let result = valve.process(create_test_buffer(0)).unwrap();
        assert!(result.is_some());
        assert_eq!(valve.stats().passed, 1);
        assert_eq!(valve.stats().dropped, 0);
    }

    #[test]
    fn test_valve_drop_when_closed() {
        let mut valve = Valve::with_state(false);

        let result = valve.process(create_test_buffer(0)).unwrap();
        assert!(result.is_none());
        assert_eq!(valve.stats().passed, 0);
        assert_eq!(valve.stats().dropped, 1);
    }

    #[test]
    fn test_valve_control() {
        let mut valve = Valve::new();
        let control = valve.control();

        // Pass when open
        let result = valve.process(create_test_buffer(0)).unwrap();
        assert!(result.is_some());

        // Close via control
        control.close();
        assert!(!valve.is_open());

        // Drop when closed
        let result = valve.process(create_test_buffer(1)).unwrap();
        assert!(result.is_none());

        // Re-open via control
        control.open();
        assert!(valve.is_open());

        // Pass again
        let result = valve.process(create_test_buffer(2)).unwrap();
        assert!(result.is_some());

        let stats = valve.stats();
        assert_eq!(stats.passed, 2);
        assert_eq!(stats.dropped, 1);
    }

    #[test]
    fn test_valve_toggle() {
        let valve = Valve::new();
        let control = valve.control();

        assert!(control.is_open());

        let new_state = control.toggle();
        assert!(!new_state);
        assert!(!control.is_open());

        let new_state = control.toggle();
        assert!(new_state);
        assert!(control.is_open());
    }

    #[test]
    fn test_valve_with_name() {
        let valve = Valve::new().with_name("my-valve");
        assert_eq!(valve.name(), "my-valve");
    }
}
