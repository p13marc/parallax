//! AppSrc element for injecting data from application code.
//!
//! Allows applications to push buffers into a pipeline programmatically.

use crate::buffer::Buffer;
use crate::element::{ProduceContext, ProduceResult, Source};
use crate::error::{Error, Result};
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A source element that allows applications to inject buffers into a pipeline.
///
/// AppSrc provides a way for application code to push data into a GStreamer-style
/// pipeline. It supports both push (application-driven) and pull (pipeline-driven)
/// modes of operation.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::AppSrc;
/// use parallax::buffer::Buffer;
///
/// let app_src = AppSrc::new();
/// let handle = app_src.handle();
///
/// // In another thread or async task:
/// handle.push_buffer(buffer)?;
///
/// // Signal end of stream when done:
/// handle.end_stream();
/// ```
pub struct AppSrc {
    name: String,
    inner: Arc<AppSrcInner>,
}

struct AppSrcInner {
    state: Mutex<AppSrcState>,
    data_available: Condvar,
}

struct AppSrcState {
    queue: VecDeque<Buffer>,
    max_buffers: usize,
    eos: bool,
    flushing: bool,
    total_pushed: u64,
    total_produced: u64,
}

/// Handle for pushing data into an AppSrc.
///
/// This handle can be cloned and sent to other threads.
#[derive(Clone)]
pub struct AppSrcHandle {
    inner: Arc<AppSrcInner>,
}

impl AppSrc {
    /// Create a new AppSrc with default settings.
    pub fn new() -> Self {
        Self::with_max_buffers(64)
    }

    /// Create a new AppSrc with a specific queue size.
    pub fn with_max_buffers(max_buffers: usize) -> Self {
        Self {
            name: "appsrc".to_string(),
            inner: Arc::new(AppSrcInner {
                state: Mutex::new(AppSrcState {
                    queue: VecDeque::with_capacity(max_buffers.min(256)),
                    max_buffers,
                    eos: false,
                    flushing: false,
                    total_pushed: 0,
                    total_produced: 0,
                }),
                data_available: Condvar::new(),
            }),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get a handle for pushing data into this source.
    pub fn handle(&self) -> AppSrcHandle {
        AppSrcHandle {
            inner: Arc::clone(&self.inner),
        }
    }

    /// Get the current queue length.
    pub fn queue_len(&self) -> usize {
        self.inner.state.lock().unwrap().queue.len()
    }

    /// Check if end-of-stream has been signaled.
    pub fn is_eos(&self) -> bool {
        self.inner.state.lock().unwrap().eos
    }

    /// Get statistics.
    pub fn stats(&self) -> AppSrcStats {
        let state = self.inner.state.lock().unwrap();
        AppSrcStats {
            queued_buffers: state.queue.len(),
            total_pushed: state.total_pushed,
            total_produced: state.total_produced,
            eos: state.eos,
        }
    }
}

impl Default for AppSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for AppSrc {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        let mut state = self.inner.state.lock().unwrap();

        // Wait for data or EOS
        while state.queue.is_empty() && !state.eos && !state.flushing {
            state = self.inner.data_available.wait(state).unwrap();
        }

        if state.flushing {
            return Err(Error::Element("appsrc is flushing".into()));
        }

        if let Some(buffer) = state.queue.pop_front() {
            state.total_produced += 1;
            Ok(ProduceResult::OwnBuffer(buffer))
        } else if state.eos {
            Ok(ProduceResult::Eos)
        } else {
            Ok(ProduceResult::WouldBlock)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl AppSrcHandle {
    /// Push a buffer into the source.
    ///
    /// This may block if the internal queue is full.
    pub fn push_buffer(&self, buffer: Buffer) -> Result<()> {
        self.push_buffer_timeout(buffer, None)
    }

    /// Push a buffer with a timeout.
    pub fn push_buffer_timeout(&self, buffer: Buffer, timeout: Option<Duration>) -> Result<()> {
        let mut state = self.inner.state.lock().unwrap();

        if state.eos {
            return Err(Error::Element("appsrc is at EOS".into()));
        }

        if state.flushing {
            return Err(Error::Element("appsrc is flushing".into()));
        }

        // Wait if queue is full
        while state.queue.len() >= state.max_buffers && !state.flushing {
            state = if let Some(t) = timeout {
                let (s, result) = self.inner.data_available.wait_timeout(state, t).unwrap();
                if result.timed_out() {
                    return Err(Error::Element("appsrc push timeout".into()));
                }
                s
            } else {
                self.inner.data_available.wait(state).unwrap()
            };
        }

        if state.flushing {
            return Err(Error::Element("appsrc is flushing".into()));
        }

        state.queue.push_back(buffer);
        state.total_pushed += 1;

        self.inner.data_available.notify_one();
        Ok(())
    }

    /// Signal end of stream.
    ///
    /// After calling this, no more buffers can be pushed.
    pub fn end_stream(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.eos = true;
        self.inner.data_available.notify_all();
    }

    /// Set flushing mode.
    pub fn set_flushing(&self, flushing: bool) {
        let mut state = self.inner.state.lock().unwrap();
        state.flushing = flushing;
        if flushing {
            self.inner.data_available.notify_all();
        }
    }

    /// Clear the queue and reset EOS state.
    pub fn reset(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.queue.clear();
        state.eos = false;
        state.flushing = false;
    }

    /// Get the current queue length.
    pub fn queue_len(&self) -> usize {
        self.inner.state.lock().unwrap().queue.len()
    }

    /// Check if the queue is full.
    pub fn is_full(&self) -> bool {
        let state = self.inner.state.lock().unwrap();
        state.queue.len() >= state.max_buffers
    }
}

/// Statistics about AppSrc operation.
#[derive(Debug, Clone, Copy)]
pub struct AppSrcStats {
    /// Number of buffers currently queued.
    pub queued_buffers: usize,
    /// Total buffers pushed by the application.
    pub total_pushed: u64,
    /// Total buffers produced to the pipeline.
    pub total_produced: u64,
    /// Whether EOS has been signaled.
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

    /// Helper to call produce with a dummy context (AppSrc provides its own buffer)
    fn produce_buffer(src: &mut AppSrc) -> Result<ProduceResult> {
        let mut ctx = ProduceContext::without_buffer();
        src.produce(&mut ctx)
    }

    #[test]
    fn test_appsrc_creation() {
        let src = AppSrc::new();
        assert_eq!(src.queue_len(), 0);
        assert!(!src.is_eos());
    }

    #[test]
    fn test_appsrc_push_produce() {
        let mut src = AppSrc::new();
        let handle = src.handle();

        handle.push_buffer(create_test_buffer(0)).unwrap();
        handle.push_buffer(create_test_buffer(1)).unwrap();

        assert_eq!(src.queue_len(), 2);

        let result = produce_buffer(&mut src).unwrap();
        let buf = match result {
            ProduceResult::OwnBuffer(b) => b,
            _ => panic!("Expected OwnBuffer"),
        };
        assert_eq!(buf.metadata().sequence, 0);

        let result = produce_buffer(&mut src).unwrap();
        let buf = match result {
            ProduceResult::OwnBuffer(b) => b,
            _ => panic!("Expected OwnBuffer"),
        };
        assert_eq!(buf.metadata().sequence, 1);
    }

    #[test]
    fn test_appsrc_eos() {
        let mut src = AppSrc::new();
        let handle = src.handle();

        handle.push_buffer(create_test_buffer(0)).unwrap();
        handle.end_stream();

        assert!(src.is_eos());

        // Should still get the buffered data
        let result = produce_buffer(&mut src).unwrap();
        assert!(matches!(result, ProduceResult::OwnBuffer(_)));

        // Now should get Eos
        let result = produce_buffer(&mut src).unwrap();
        assert!(result.is_eos());
    }

    #[test]
    fn test_appsrc_push_after_eos() {
        let src = AppSrc::new();
        let handle = src.handle();

        handle.end_stream();

        let result = handle.push_buffer(create_test_buffer(0));
        assert!(result.is_err());
    }

    #[test]
    fn test_appsrc_multithreaded() {
        let mut src = AppSrc::new();
        let handle = src.handle();

        let producer = thread::spawn(move || {
            for i in 0..10 {
                handle.push_buffer(create_test_buffer(i)).unwrap();
            }
            handle.end_stream();
        });

        let mut received = Vec::new();
        loop {
            match produce_buffer(&mut src).unwrap() {
                ProduceResult::OwnBuffer(buf) => {
                    received.push(buf.metadata().sequence);
                }
                ProduceResult::Eos => break,
                ProduceResult::WouldBlock | ProduceResult::Produced(_) => {
                    // Shouldn't happen in this test, but handle gracefully
                }
            }
        }

        producer.join().unwrap();
        assert_eq!(received.len(), 10);
    }

    #[test]
    fn test_appsrc_reset() {
        let src = AppSrc::new();
        let handle = src.handle();

        handle.push_buffer(create_test_buffer(0)).unwrap();
        handle.end_stream();

        assert!(src.is_eos());
        assert_eq!(src.queue_len(), 1);

        handle.reset();

        assert!(!src.is_eos());
        assert_eq!(src.queue_len(), 0);
    }

    #[test]
    fn test_appsrc_stats() {
        let mut src = AppSrc::new();
        let handle = src.handle();

        handle.push_buffer(create_test_buffer(0)).unwrap();
        handle.push_buffer(create_test_buffer(1)).unwrap();
        let _ = produce_buffer(&mut src).unwrap();

        let stats = src.stats();
        assert_eq!(stats.total_pushed, 2);
        assert_eq!(stats.total_produced, 1);
        assert_eq!(stats.queued_buffers, 1);
    }
}
