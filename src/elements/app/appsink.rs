//! AppSink element for extracting data to application code.
//!
//! Allows applications to pull buffers from a pipeline programmatically.

use crate::buffer::Buffer;
use crate::element::Sink;
use crate::error::{Error, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A sink element that allows applications to extract buffers from a pipeline.
///
/// AppSink provides a way for application code to pull data from a pipeline.
/// Buffers are queued internally and can be retrieved via a handle.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::AppSink;
///
/// let app_sink = AppSink::new();
/// let handle = app_sink.handle();
///
/// // Pipeline pushes data to sink...
///
/// // In application code:
/// while let Some(buffer) = handle.pull_buffer()? {
///     // Process buffer
/// }
/// ```
pub struct AppSink {
    name: String,
    inner: Arc<AppSinkInner>,
}

struct AppSinkInner {
    state: Mutex<AppSinkState>,
    data_available: Condvar,
    space_available: Condvar,
}

struct AppSinkState {
    queue: VecDeque<Buffer>,
    max_buffers: usize,
    eos: bool,
    flushing: bool,
    drop_on_full: bool,
    total_received: u64,
    total_pulled: u64,
    total_dropped: u64,
}

/// Handle for pulling data from an AppSink.
///
/// This handle can be cloned and sent to other threads.
#[derive(Clone)]
pub struct AppSinkHandle {
    inner: Arc<AppSinkInner>,
}

impl AppSink {
    /// Create a new AppSink with default settings.
    pub fn new() -> Self {
        Self::with_max_buffers(64)
    }

    /// Create a new AppSink with a specific queue size.
    pub fn with_max_buffers(max_buffers: usize) -> Self {
        Self {
            name: "appsink".to_string(),
            inner: Arc::new(AppSinkInner {
                state: Mutex::new(AppSinkState {
                    queue: VecDeque::with_capacity(max_buffers.min(256)),
                    max_buffers,
                    eos: false,
                    flushing: false,
                    drop_on_full: false,
                    total_received: 0,
                    total_pulled: 0,
                    total_dropped: 0,
                }),
                data_available: Condvar::new(),
                space_available: Condvar::new(),
            }),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set whether to drop buffers when the queue is full.
    ///
    /// If false (default), the sink will block when full.
    pub fn drop_on_full(self, drop: bool) -> Self {
        self.inner.state.lock().unwrap().drop_on_full = drop;
        self
    }

    /// Get a handle for pulling data from this sink.
    pub fn handle(&self) -> AppSinkHandle {
        AppSinkHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Get the current queue length.
    pub fn queue_len(&self) -> usize {
        self.inner.state.lock().unwrap().queue.len()
    }

    /// Check if end-of-stream has been received.
    pub fn is_eos(&self) -> bool {
        self.inner.state.lock().unwrap().eos
    }

    /// Get statistics.
    pub fn stats(&self) -> AppSinkStats {
        let state = self.inner.state.lock().unwrap();
        AppSinkStats {
            queued_buffers: state.queue.len(),
            total_received: state.total_received,
            total_pulled: state.total_pulled,
            total_dropped: state.total_dropped,
            eos: state.eos,
        }
    }

    /// Signal end of stream from the pipeline side.
    pub fn send_eos(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.eos = true;
        self.inner.data_available.notify_all();
    }
}

impl Default for AppSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for AppSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let mut state = self.inner.state.lock().unwrap();

        if state.flushing {
            return Err(Error::Element("appsink is flushing".into()));
        }

        // Handle full queue
        while state.queue.len() >= state.max_buffers && !state.flushing {
            if state.drop_on_full {
                state.total_dropped += 1;
                return Ok(());
            }
            state = self.inner.space_available.wait(state).unwrap();
        }

        if state.flushing {
            return Err(Error::Element("appsink is flushing".into()));
        }

        state.queue.push_back(buffer);
        state.total_received += 1;

        self.inner.data_available.notify_one();
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl AppSinkHandle {
    /// Pull a buffer from the sink.
    ///
    /// Returns `Ok(None)` when EOS is reached and no more buffers are available.
    pub fn pull_buffer(&self) -> Result<Option<Buffer>> {
        self.pull_buffer_timeout(None)
    }

    /// Pull a buffer with a timeout.
    ///
    /// Returns `Ok(None)` on timeout or EOS.
    pub fn pull_buffer_timeout(&self, timeout: Option<Duration>) -> Result<Option<Buffer>> {
        let mut state = self.inner.state.lock().unwrap();

        // Wait for data
        while state.queue.is_empty() && !state.eos && !state.flushing {
            state = if let Some(t) = timeout {
                let (s, result) = self.inner.data_available.wait_timeout(state, t).unwrap();
                if result.timed_out() {
                    return Ok(None);
                }
                s
            } else {
                self.inner.data_available.wait(state).unwrap()
            };
        }

        if state.flushing {
            return Err(Error::Element("appsink is flushing".into()));
        }

        if let Some(buffer) = state.queue.pop_front() {
            state.total_pulled += 1;
            self.inner.space_available.notify_one();
            Ok(Some(buffer))
        } else if state.eos {
            Ok(None)
        } else {
            Ok(None)
        }
    }

    /// Try to pull a buffer without blocking.
    pub fn try_pull_buffer(&self) -> Option<Buffer> {
        let mut state = self.inner.state.lock().unwrap();

        if let Some(buffer) = state.queue.pop_front() {
            state.total_pulled += 1;
            self.inner.space_available.notify_one();
            Some(buffer)
        } else {
            None
        }
    }

    /// Set flushing mode.
    pub fn set_flushing(&self, flushing: bool) {
        let mut state = self.inner.state.lock().unwrap();
        state.flushing = flushing;
        if flushing {
            self.inner.data_available.notify_all();
            self.inner.space_available.notify_all();
        }
    }

    /// Clear the queue.
    pub fn clear(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.queue.clear();
        self.inner.space_available.notify_all();
    }

    /// Get the current queue length.
    pub fn queue_len(&self) -> usize {
        self.inner.state.lock().unwrap().queue.len()
    }

    /// Check if EOS has been reached.
    pub fn is_eos(&self) -> bool {
        self.inner.state.lock().unwrap().eos
    }

    /// Check if there are buffers available.
    pub fn has_buffer(&self) -> bool {
        !self.inner.state.lock().unwrap().queue.is_empty()
    }
}

/// Statistics about AppSink operation.
#[derive(Debug, Clone, Copy)]
pub struct AppSinkStats {
    /// Number of buffers currently queued.
    pub queued_buffers: usize,
    /// Total buffers received from the pipeline.
    pub total_received: u64,
    /// Total buffers pulled by the application.
    pub total_pulled: u64,
    /// Total buffers dropped (when drop_on_full is enabled).
    pub total_dropped: u64,
    /// Whether EOS has been received.
    pub eos: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;
    use std::thread;

    fn create_test_buffer(seq: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(100).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn test_appsink_creation() {
        let sink = AppSink::new();
        assert_eq!(sink.queue_len(), 0);
        assert!(!sink.is_eos());
    }

    #[test]
    fn test_appsink_consume_pull() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        sink.consume(create_test_buffer(0)).unwrap();
        sink.consume(create_test_buffer(1)).unwrap();

        assert_eq!(handle.queue_len(), 2);

        let buf = handle
            .pull_buffer_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 0);

        let buf = handle
            .pull_buffer_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 1);
    }

    #[test]
    fn test_appsink_eos() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        sink.consume(create_test_buffer(0)).unwrap();
        sink.send_eos();

        assert!(sink.is_eos());

        // Should still get the buffered data
        let buf = handle.pull_buffer().unwrap();
        assert!(buf.is_some());

        // Now should get None for EOS
        let buf = handle.pull_buffer().unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn test_appsink_try_pull() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        // No data - should return None immediately
        assert!(handle.try_pull_buffer().is_none());

        sink.consume(create_test_buffer(0)).unwrap();

        // Now should get data
        let buf = handle.try_pull_buffer();
        assert!(buf.is_some());
    }

    #[test]
    fn test_appsink_drop_on_full() {
        let mut sink = AppSink::with_max_buffers(2).drop_on_full(true);

        sink.consume(create_test_buffer(0)).unwrap();
        sink.consume(create_test_buffer(1)).unwrap();
        sink.consume(create_test_buffer(2)).unwrap(); // Should be dropped

        assert_eq!(sink.queue_len(), 2);
        assert_eq!(sink.stats().total_dropped, 1);
    }

    #[test]
    fn test_appsink_multithreaded() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        let producer = thread::spawn(move || {
            for i in 0..10 {
                sink.consume(create_test_buffer(i)).unwrap();
            }
            sink.send_eos();
        });

        let mut received = Vec::new();
        while let Ok(Some(buf)) = handle.pull_buffer() {
            received.push(buf.metadata().sequence);
        }

        producer.join().unwrap();
        assert_eq!(received.len(), 10);
    }

    #[test]
    fn test_appsink_clear() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        sink.consume(create_test_buffer(0)).unwrap();
        sink.consume(create_test_buffer(1)).unwrap();

        handle.clear();

        assert_eq!(handle.queue_len(), 0);
    }

    #[test]
    fn test_appsink_stats() {
        let mut sink = AppSink::new();
        let handle = sink.handle();

        sink.consume(create_test_buffer(0)).unwrap();
        sink.consume(create_test_buffer(1)).unwrap();
        handle.try_pull_buffer();

        let stats = sink.stats();
        assert_eq!(stats.total_received, 2);
        assert_eq!(stats.total_pulled, 1);
        assert_eq!(stats.queued_buffers, 1);
    }
}
