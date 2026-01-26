//! Queue element for asynchronous buffering.
//!
//! Provides a buffer queue between pipeline elements, enabling:
//! - Decoupling of producer and consumer rates
//! - Backpressure handling
//! - Thread boundary crossing

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::{Error, Result};
use std::collections::VecDeque;
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
}

struct QueueInner {
    state: Mutex<QueueState>,
    not_empty: Condvar,
    not_full: Condvar,
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
                }),
                not_empty: Condvar::new(),
                not_full: Condvar::new(),
            }),
            leaky: LeakyMode::None,
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
                }),
                not_empty: Condvar::new(),
                not_full: Condvar::new(),
            }),
            leaky: LeakyMode::None,
        }
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
            total_pushed: state.total_pushed,
            total_popped: state.total_popped,
            total_dropped: state.total_dropped,
        }
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
    /// Total buffers pushed to the queue.
    pub total_pushed: u64,
    /// Total buffers popped from the queue.
    pub total_popped: u64,
    /// Total buffers dropped (due to leaky mode).
    pub total_dropped: u64,
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
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;
    use std::thread;

    fn create_test_buffer(size: usize, seq: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(size).unwrap());
        let handle = MemoryHandle::from_segment(segment);
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
    }
}
