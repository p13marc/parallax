//! Queue element for asynchronous buffering.
//!
//! Provides a buffer queue between pipeline elements, enabling:
//! - Decoupling of producer and consumer rates
//! - Backpressure handling via flow signals
//! - Thread boundary crossing
//!
//! # Flow Control
//!
//! The queue supports flow control via water marks:
//! - When fill level reaches **high water mark** (default 80%), emits `FlowSignal::Busy`
//! - When fill level drops to **low water mark** (default 20%), emits `FlowSignal::Ready`
//!
//! This hysteresis prevents oscillation between busy and ready states.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::Queue;
//! use parallax::pipeline::flow::{FlowSignal, WaterMarks};
//!
//! let queue = Queue::new(100)
//!     .with_water_marks(WaterMarks::from_capacity(100));
//!
//! // Check flow state
//! let signal = queue.flow_signal();
//! if signal == FlowSignal::Busy {
//!     // Upstream should slow down or drop frames
//! }
//! ```

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::pipeline::flow::{FlowSignal, FlowStateHandle, SharedFlowState, WaterMarks};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A queue element that buffers data between pipeline stages.
///
/// The queue provides asynchronous decoupling between upstream and downstream
/// elements, allowing them to operate at different rates while handling
/// backpressure.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Queue;
///
/// // Create a queue with max 100 buffers
/// let queue = Queue::new(100);
///
/// // Or with byte limit
/// let queue = Queue::with_limits(100, 1024 * 1024); // 100 buffers or 1MB
/// ```
pub struct Queue {
    name: String,
    inner: Arc<QueueInner>,
    leaky: LeakyMode,
    water_marks: Option<WaterMarks>,
}

struct QueueInner {
    state: Mutex<QueueState>,
    not_empty: Condvar,
    not_full: Condvar,
    /// Current flow signal (atomic for lock-free reads)
    flow_signal: AtomicU8,
    /// Shared flow state for external consumers (sources)
    shared_flow_state: FlowStateHandle,
}

struct QueueState {
    buffers: VecDeque<Buffer>,
    max_buffers: usize,
    max_bytes: Option<usize>,
    current_bytes: usize,
    total_pushed: u64,
    total_popped: u64,
    total_dropped: u64,
    flushing: bool,
    /// Number of times high water mark was reached
    high_water_events: u64,
}

/// Leaky mode determines what happens when the queue is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LeakyMode {
    /// Block until space is available (default).
    #[default]
    None,
    /// Drop new buffers when full (upstream leaky).
    Upstream,
    /// Drop old buffers when full (downstream leaky).
    Downstream,
}

impl Queue {
    /// Create a new queue with a maximum buffer count.
    pub fn new(max_buffers: usize) -> Self {
        Self {
            name: format!("queue-{}", max_buffers),
            inner: Arc::new(QueueInner {
                state: Mutex::new(QueueState {
                    buffers: VecDeque::with_capacity(max_buffers.min(1024)),
                    max_buffers,
                    max_bytes: None,
                    current_bytes: 0,
                    total_pushed: 0,
                    total_popped: 0,
                    total_dropped: 0,
                    flushing: false,
                    high_water_events: 0,
                }),
                not_empty: Condvar::new(),
                not_full: Condvar::new(),
                flow_signal: AtomicU8::new(FlowSignal::Ready as u8),
                shared_flow_state: Arc::new(SharedFlowState::new()),
            }),
            leaky: LeakyMode::None,
            water_marks: None,
        }
    }

    /// Create a queue with both buffer count and byte limits.
    pub fn with_limits(max_buffers: usize, max_bytes: usize) -> Self {
        Self {
            name: format!("queue-{}buf-{}B", max_buffers, max_bytes),
            inner: Arc::new(QueueInner {
                state: Mutex::new(QueueState {
                    buffers: VecDeque::with_capacity(max_buffers.min(1024)),
                    max_buffers,
                    max_bytes: Some(max_bytes),
                    current_bytes: 0,
                    total_pushed: 0,
                    total_popped: 0,
                    total_dropped: 0,
                    flushing: false,
                    high_water_events: 0,
                }),
                not_empty: Condvar::new(),
                not_full: Condvar::new(),
                flow_signal: AtomicU8::new(FlowSignal::Ready as u8),
                shared_flow_state: Arc::new(SharedFlowState::new()),
            }),
            leaky: LeakyMode::None,
            water_marks: None,
        }
    }

    /// Set water marks for flow control.
    ///
    /// When the queue level reaches the high water mark, `flow_signal()` returns `Busy`.
    /// When it drops to the low water mark, `flow_signal()` returns `Ready`.
    pub fn with_water_marks(mut self, water_marks: WaterMarks) -> Self {
        self.water_marks = Some(water_marks);
        self
    }

    /// Enable default water marks (80% high, 20% low).
    pub fn with_flow_control(mut self) -> Self {
        let state = self.inner.state.lock().unwrap();
        let capacity = state.max_buffers;
        drop(state);
        self.water_marks = Some(WaterMarks::from_capacity(capacity));
        self
    }

    /// Set the leaky mode.
    pub fn leaky(mut self, mode: LeakyMode) -> Self {
        self.leaky = mode;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the current number of buffers in the queue.
    pub fn len(&self) -> usize {
        self.inner.state.lock().unwrap().buffers.len()
    }

    /// Check if the queue is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the current byte count in the queue.
    pub fn current_bytes(&self) -> usize {
        self.inner.state.lock().unwrap().current_bytes
    }

    /// Get statistics about the queue.
    pub fn stats(&self) -> QueueStats {
        let state = self.inner.state.lock().unwrap();
        QueueStats {
            current_buffers: state.buffers.len(),
            current_bytes: state.current_bytes,
            max_buffers: state.max_buffers,
            total_pushed: state.total_pushed,
            total_popped: state.total_popped,
            total_dropped: state.total_dropped,
            high_water_events: state.high_water_events,
        }
    }

    /// Get the current flow signal.
    ///
    /// This is a lock-free read suitable for frequent polling.
    /// Returns `Ready` if flow control is disabled.
    #[inline]
    pub fn flow_signal(&self) -> FlowSignal {
        FlowSignal::from(self.inner.flow_signal.load(Ordering::Acquire) as u8)
    }

    /// Check if the queue is signaling backpressure.
    #[inline]
    pub fn is_busy(&self) -> bool {
        self.flow_signal() == FlowSignal::Busy
    }

    /// Get the current fill level as a percentage (0.0 to 100.0).
    pub fn fill_level(&self) -> f64 {
        let state = self.inner.state.lock().unwrap();
        if state.max_buffers == 0 {
            return 0.0;
        }
        (state.buffers.len() as f64 / state.max_buffers as f64) * 100.0
    }

    /// Get a flow state handle that stays synchronized with queue state.
    ///
    /// This handle can be shared with upstream sources to enable them to
    /// respond to backpressure. When the queue reaches high water mark,
    /// the handle will signal `Busy`; when it drops to low water mark,
    /// it signals `Ready`.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let queue = Queue::new(100).with_flow_control();
    /// let flow_state = queue.flow_state_handle();
    ///
    /// // Share with source
    /// source.set_flow_state(flow_state.clone());
    ///
    /// // Source can now check: if !flow_state.should_produce() { drop_frame(); }
    /// ```
    pub fn flow_state_handle(&self) -> FlowStateHandle {
        Arc::clone(&self.inner.shared_flow_state)
    }

    /// Update flow signal based on current fill level.
    fn update_flow_signal(&self, level: usize) {
        let Some(wm) = &self.water_marks else {
            return; // Flow control disabled
        };

        let current = self.inner.flow_signal.load(Ordering::Acquire);
        let current_signal = FlowSignal::from(current as u8);

        let new_signal = if wm.is_high(level) && current_signal == FlowSignal::Ready {
            // Transition to Busy
            let mut state = self.inner.state.lock().unwrap();
            state.high_water_events += 1;
            drop(state);
            FlowSignal::Busy
        } else if wm.is_low(level) && current_signal == FlowSignal::Busy {
            // Transition to Ready
            FlowSignal::Ready
        } else {
            return; // No change
        };

        // Update both internal and shared state
        self.inner
            .flow_signal
            .store(new_signal as u8, Ordering::Release);
        self.inner.shared_flow_state.set_signal(new_signal);
    }

    /// Flush all buffers from the queue.
    pub fn flush(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.buffers.clear();
        state.current_bytes = 0;
        self.inner.not_full.notify_all();
    }

    /// Set flushing mode (causes blocked operations to return).
    pub fn set_flushing(&self, flushing: bool) {
        let mut state = self.inner.state.lock().unwrap();
        state.flushing = flushing;
        if flushing {
            self.inner.not_empty.notify_all();
            self.inner.not_full.notify_all();
        }
    }

    /// Push a buffer into the queue.
    ///
    /// Behavior depends on leaky mode:
    /// - `None`: Blocks until space is available
    /// - `Upstream`: Drops the new buffer if full
    /// - `Downstream`: Drops the oldest buffer if full
    pub fn push(&self, buffer: Buffer) -> Result<()> {
        self.push_timeout(buffer, None)
    }

    /// Push a buffer with a timeout.
    pub fn push_timeout(&self, buffer: Buffer, timeout: Option<Duration>) -> Result<()> {
        let buffer_len = buffer.len();
        let mut state = self.inner.state.lock().unwrap();

        // Check if full
        while self.is_full_locked(&state) && !state.flushing {
            match self.leaky {
                LeakyMode::None => {
                    // Block until space available
                    state = if let Some(t) = timeout {
                        let (s, result) = self.inner.not_full.wait_timeout(state, t).unwrap();
                        if result.timed_out() {
                            return Err(Error::Element("queue push timeout".into()));
                        }
                        s
                    } else {
                        self.inner.not_full.wait(state).unwrap()
                    };
                }
                LeakyMode::Upstream => {
                    // Drop the incoming buffer
                    state.total_dropped += 1;
                    return Ok(());
                }
                LeakyMode::Downstream => {
                    // Drop the oldest buffer
                    if let Some(old) = state.buffers.pop_front() {
                        state.current_bytes = state.current_bytes.saturating_sub(old.len());
                        state.total_dropped += 1;
                    }
                    break;
                }
            }
        }

        if state.flushing {
            return Err(Error::Element("queue is flushing".into()));
        }

        state.buffers.push_back(buffer);
        state.current_bytes += buffer_len;
        state.total_pushed += 1;
        let level = state.buffers.len();
        drop(state);

        self.update_flow_signal(level);
        self.inner.not_empty.notify_one();
        Ok(())
    }

    /// Pop a buffer from the queue.
    ///
    /// Blocks until a buffer is available or the queue is flushing.
    pub fn pop(&self) -> Result<Option<Buffer>> {
        self.pop_timeout(None)
    }

    /// Pop a buffer with a timeout.
    pub fn pop_timeout(&self, timeout: Option<Duration>) -> Result<Option<Buffer>> {
        let mut state = self.inner.state.lock().unwrap();

        while state.buffers.is_empty() && !state.flushing {
            state = if let Some(t) = timeout {
                let (s, result) = self.inner.not_empty.wait_timeout(state, t).unwrap();
                if result.timed_out() {
                    return Ok(None);
                }
                s
            } else {
                self.inner.not_empty.wait(state).unwrap()
            };
        }

        if state.flushing && state.buffers.is_empty() {
            return Ok(None);
        }

        if let Some(buffer) = state.buffers.pop_front() {
            state.current_bytes = state.current_bytes.saturating_sub(buffer.len());
            state.total_popped += 1;
            let level = state.buffers.len();
            drop(state);

            self.update_flow_signal(level);
            self.inner.not_full.notify_one();
            Ok(Some(buffer))
        } else {
            Ok(None)
        }
    }

    fn is_full_locked(&self, state: &QueueState) -> bool {
        if state.buffers.len() >= state.max_buffers {
            return true;
        }
        if let Some(max_bytes) = state.max_bytes {
            if state.current_bytes >= max_bytes {
                return true;
            }
        }
        false
    }
}

/// Statistics about queue operation.
#[derive(Debug, Clone, Copy)]
pub struct QueueStats {
    /// Current number of buffers in the queue.
    pub current_buffers: usize,
    /// Current total bytes in the queue.
    pub current_bytes: usize,
    /// Maximum buffer capacity.
    pub max_buffers: usize,
    /// Total buffers pushed to the queue.
    pub total_pushed: u64,
    /// Total buffers popped from the queue.
    pub total_popped: u64,
    /// Total buffers dropped (due to leaky mode).
    pub total_dropped: u64,
    /// Number of times high water mark was reached.
    pub high_water_events: u64,
}

impl Element for Queue {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // In element mode, queue acts as pass-through with internal buffering
        // For full async operation, use push/pop directly
        self.push(buffer)?;
        self.pop()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Clone for Queue {
    fn clone(&self) -> Self {
        Self {
            name: self.name.clone(),
            inner: Arc::clone(&self.inner),
            leaky: self.leaky,
            water_marks: self.water_marks.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;
    use std::thread;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(256, 256).unwrap())
    }

    fn create_test_buffer(size: usize, seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, size);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn test_queue_creation() {
        let queue = Queue::new(10);
        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);
    }

    #[test]
    fn test_queue_push_pop() {
        let queue = Queue::new(10);

        queue.push(create_test_buffer(100, 0)).unwrap();
        queue.push(create_test_buffer(100, 1)).unwrap();

        assert_eq!(queue.len(), 2);

        let buf = queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 0);

        let buf = queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 1);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_queue_leaky_upstream() {
        let queue = Queue::new(2).leaky(LeakyMode::Upstream);

        queue.push(create_test_buffer(100, 0)).unwrap();
        queue.push(create_test_buffer(100, 1)).unwrap();
        queue.push(create_test_buffer(100, 2)).unwrap(); // Should be dropped

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.stats().total_dropped, 1);

        // First buffer should still be 0
        let buf = queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(buf.unwrap().metadata().sequence, 0);
    }

    #[test]
    fn test_queue_leaky_downstream() {
        let queue = Queue::new(2).leaky(LeakyMode::Downstream);

        queue.push(create_test_buffer(100, 0)).unwrap();
        queue.push(create_test_buffer(100, 1)).unwrap();
        queue.push(create_test_buffer(100, 2)).unwrap(); // Should drop oldest

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.stats().total_dropped, 1);

        // First buffer should now be 1 (0 was dropped)
        let buf = queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(buf.unwrap().metadata().sequence, 1);
    }

    #[test]
    fn test_queue_byte_limit() {
        let queue = Queue::with_limits(100, 200).leaky(LeakyMode::Upstream);

        queue.push(create_test_buffer(100, 0)).unwrap();
        queue.push(create_test_buffer(100, 1)).unwrap();
        queue.push(create_test_buffer(100, 2)).unwrap(); // Should be dropped (at 200 byte limit)

        assert_eq!(queue.len(), 2);
        assert_eq!(queue.current_bytes(), 200);
    }

    #[test]
    fn test_queue_flush() {
        let queue = Queue::new(10);

        queue.push(create_test_buffer(100, 0)).unwrap();
        queue.push(create_test_buffer(100, 1)).unwrap();

        queue.flush();

        assert!(queue.is_empty());
        assert_eq!(queue.current_bytes(), 0);
    }

    #[test]
    fn test_queue_multithreaded() {
        let queue = Queue::new(100);
        let queue_clone = queue.clone();

        let producer = thread::spawn(move || {
            for i in 0..50 {
                queue_clone.push(create_test_buffer(10, i)).unwrap();
            }
        });

        let consumer = thread::spawn(move || {
            let mut count = 0;
            while count < 50 {
                if queue
                    .pop_timeout(Some(Duration::from_millis(100)))
                    .unwrap()
                    .is_some()
                {
                    count += 1;
                }
            }
            count
        });

        producer.join().unwrap();
        let count = consumer.join().unwrap();
        assert_eq!(count, 50);
    }

    #[test]
    fn test_queue_flushing() {
        let queue = Queue::new(10);
        let queue_clone = queue.clone();

        let consumer = thread::spawn(move || {
            // This will block waiting for data
            queue_clone.pop()
        });

        // Give consumer time to start waiting
        thread::sleep(Duration::from_millis(50));

        // Set flushing to unblock
        queue.set_flushing(true);

        let result = consumer.join().unwrap();
        assert!(result.unwrap().is_none());
    }

    #[test]
    fn test_queue_stats() {
        let queue = Queue::new(10);

        queue.push(create_test_buffer(100, 0)).unwrap();
        queue.push(create_test_buffer(100, 1)).unwrap();
        queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();

        let stats = queue.stats();
        assert_eq!(stats.total_pushed, 2);
        assert_eq!(stats.total_popped, 1);
        assert_eq!(stats.current_buffers, 1);
        assert_eq!(stats.max_buffers, 10);
    }

    #[test]
    fn test_queue_flow_control_disabled() {
        // Without water marks, flow signal is always Ready
        let queue = Queue::new(10);

        for i in 0..10 {
            queue.push(create_test_buffer(10, i)).unwrap();
        }

        // Even at full capacity, signal should be Ready (flow control disabled)
        assert_eq!(queue.flow_signal(), FlowSignal::Ready);
        assert!(!queue.is_busy());
    }

    #[test]
    fn test_queue_flow_control_high_water() {
        // With water marks (80% high, 20% low), 10 buffers means:
        // - High water at 8 buffers
        // - Low water at 2 buffers
        let queue = Queue::new(10).with_flow_control();

        // Push 7 buffers - still below high water
        for i in 0..7 {
            queue.push(create_test_buffer(10, i)).unwrap();
        }
        assert_eq!(queue.flow_signal(), FlowSignal::Ready);
        assert!(!queue.is_busy());

        // Push 8th buffer - reaches high water mark
        queue.push(create_test_buffer(10, 7)).unwrap();
        assert_eq!(queue.flow_signal(), FlowSignal::Busy);
        assert!(queue.is_busy());

        // Stats should show high water event
        assert_eq!(queue.stats().high_water_events, 1);
    }

    #[test]
    fn test_queue_flow_control_low_water() {
        let queue = Queue::new(10).with_flow_control();

        // Fill to high water
        for i in 0..8 {
            queue.push(create_test_buffer(10, i)).unwrap();
        }
        assert_eq!(queue.flow_signal(), FlowSignal::Busy);

        // Pop down to 3 - still above low water (2), should stay Busy
        for _ in 0..5 {
            queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        }
        assert_eq!(queue.len(), 3);
        assert_eq!(queue.flow_signal(), FlowSignal::Busy);

        // Pop one more to reach 2 - at low water, should become Ready
        queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(queue.len(), 2);
        assert_eq!(queue.flow_signal(), FlowSignal::Ready);
    }

    #[test]
    fn test_queue_flow_control_hysteresis() {
        // Test that hysteresis prevents oscillation
        let queue = Queue::new(10).with_flow_control();

        // Fill to high water (8)
        for i in 0..8 {
            queue.push(create_test_buffer(10, i)).unwrap();
        }
        assert_eq!(queue.flow_signal(), FlowSignal::Busy);

        // Pop one to 7 - still Busy (not at low water yet)
        queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(queue.flow_signal(), FlowSignal::Busy);

        // Push back to 8 - still Busy (already was Busy)
        queue.push(create_test_buffer(10, 8)).unwrap();
        assert_eq!(queue.flow_signal(), FlowSignal::Busy);

        // Should only have one high water event (first time reaching 8)
        assert_eq!(queue.stats().high_water_events, 1);
    }

    #[test]
    fn test_queue_fill_level() {
        let queue = Queue::new(10);

        assert_eq!(queue.fill_level(), 0.0);

        queue.push(create_test_buffer(10, 0)).unwrap();
        assert!((queue.fill_level() - 10.0).abs() < 0.01);

        for i in 1..5 {
            queue.push(create_test_buffer(10, i)).unwrap();
        }
        assert!((queue.fill_level() - 50.0).abs() < 0.01);
    }

    #[test]
    fn test_queue_custom_water_marks() {
        // Custom water marks: 50% high, 10% low
        let wm = WaterMarks::new(5, 1); // 5 high, 1 low for 10-buffer queue
        let queue = Queue::new(10).with_water_marks(wm);

        // Push 4 - below high water
        for i in 0..4 {
            queue.push(create_test_buffer(10, i)).unwrap();
        }
        assert_eq!(queue.flow_signal(), FlowSignal::Ready);

        // Push 5th - at high water
        queue.push(create_test_buffer(10, 4)).unwrap();
        assert_eq!(queue.flow_signal(), FlowSignal::Busy);

        // Pop to 2 - still above low water (1)
        for _ in 0..3 {
            queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        }
        assert_eq!(queue.flow_signal(), FlowSignal::Busy);

        // Pop to 1 - at low water
        queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(queue.flow_signal(), FlowSignal::Ready);
    }

    #[test]
    fn test_queue_flow_state_handle_sync() {
        let queue = Queue::new(10).with_flow_control();
        let flow_state = queue.flow_state_handle();

        // Initial state should be Ready
        assert_eq!(flow_state.signal(), FlowSignal::Ready);
        assert!(flow_state.should_produce());

        // Fill to high water (8)
        for i in 0..8 {
            queue.push(create_test_buffer(10, i)).unwrap();
        }

        // Handle should now show Busy
        assert_eq!(flow_state.signal(), FlowSignal::Busy);
        assert!(!flow_state.should_produce());

        // Drain to low water (2)
        for _ in 0..6 {
            queue.pop_timeout(Some(Duration::from_millis(100))).unwrap();
        }

        // Handle should now show Ready again
        assert_eq!(flow_state.signal(), FlowSignal::Ready);
        assert!(flow_state.should_produce());
    }

    #[test]
    fn test_queue_flow_state_handle_shared() {
        let queue = Queue::new(10).with_flow_control();
        let handle1 = queue.flow_state_handle();
        let handle2 = queue.flow_state_handle();

        // Both handles point to the same state
        for i in 0..8 {
            queue.push(create_test_buffer(10, i)).unwrap();
        }

        assert_eq!(handle1.signal(), FlowSignal::Busy);
        assert_eq!(handle2.signal(), FlowSignal::Busy);

        // Both see the same backpressure event count
        assert!(handle1.backpressure_events() >= 1);
        assert_eq!(handle1.backpressure_events(), handle2.backpressure_events());
    }
}
