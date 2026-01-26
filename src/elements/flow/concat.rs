//! Concat element for sequential stream concatenation.
//!
//! Concatenates multiple streams one after another.

use crate::buffer::Buffer;
use crate::element::{ProduceContext, ProduceResult, Source};
use crate::error::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A concat element that plays streams sequentially.
///
/// Unlike `Funnel` which interleaves buffers from all inputs,
/// `Concat` plays each input stream to completion before moving to the next.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Concat;
///
/// let concat = Concat::new();
///
/// // Add streams in order
/// let stream1 = concat.add_stream();
/// let stream2 = concat.add_stream();
///
/// // stream1 will play completely, then stream2
/// ```
pub struct Concat {
    name: String,
    inner: Arc<ConcatInner>,
}

struct ConcatInner {
    state: Mutex<ConcatState>,
    data_available: Condvar,
}

struct ConcatState {
    streams: Vec<StreamData>,
    current_stream: usize,
    max_buffers_per_stream: usize,
    total_produced: u64,
    flushing: bool,
}

struct StreamData {
    queue: VecDeque<Buffer>,
    eos: bool,
}

/// Handle for adding data to a concat stream.
#[derive(Clone)]
pub struct ConcatStream {
    inner: Arc<ConcatInner>,
    id: usize,
}

impl Concat {
    /// Create a new concat element.
    pub fn new() -> Self {
        Self::with_max_buffers_per_stream(64)
    }

    /// Create a new concat with a specific buffer limit per stream.
    pub fn with_max_buffers_per_stream(max_buffers: usize) -> Self {
        Self {
            name: "concat".to_string(),
            inner: Arc::new(ConcatInner {
                state: Mutex::new(ConcatState {
                    streams: Vec::new(),
                    current_stream: 0,
                    max_buffers_per_stream: max_buffers,
                    total_produced: 0,
                    flushing: false,
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

    /// Add a new stream to the concat.
    ///
    /// Streams are played in the order they are added.
    pub fn add_stream(&self) -> ConcatStream {
        let mut state = self.inner.state.lock().unwrap();
        let id = state.streams.len();
        let capacity = state.max_buffers_per_stream.min(64);
        state.streams.push(StreamData {
            queue: VecDeque::with_capacity(capacity),
            eos: false,
        });

        ConcatStream {
            inner: Arc::clone(&self.inner),
            id,
        }
    }

    /// Get the number of streams.
    pub fn stream_count(&self) -> usize {
        self.inner.state.lock().unwrap().streams.len()
    }

    /// Get the currently playing stream index.
    pub fn current_stream(&self) -> usize {
        self.inner.state.lock().unwrap().current_stream
    }

    /// Check if all streams have finished.
    pub fn is_eos(&self) -> bool {
        let state = self.inner.state.lock().unwrap();
        state.current_stream >= state.streams.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> ConcatStats {
        let state = self.inner.state.lock().unwrap();
        ConcatStats {
            stream_count: state.streams.len(),
            current_stream: state.current_stream,
            total_produced: state.total_produced,
            is_eos: state.current_stream >= state.streams.len(),
        }
    }

    /// Set flushing mode.
    pub fn set_flushing(&self, flushing: bool) {
        let mut state = self.inner.state.lock().unwrap();
        state.flushing = flushing;
        if flushing {
            self.inner.data_available.notify_all();
        }
    }

    /// Pull a buffer with timeout.
    pub fn pull_timeout(&self, timeout: Option<Duration>) -> Result<Option<Buffer>> {
        let mut state = self.inner.state.lock().unwrap();

        loop {
            if state.flushing {
                return Ok(None);
            }

            // Check if we're past all streams
            if state.current_stream >= state.streams.len() {
                return Ok(None);
            }

            let current = state.current_stream;

            // Try to get a buffer from current stream
            if let Some(buffer) = state.streams[current].queue.pop_front() {
                state.total_produced += 1;
                return Ok(Some(buffer));
            }

            // Current stream is empty
            if state.streams[current].eos {
                // Move to next stream
                state.current_stream += 1;
                continue;
            }

            // Wait for data
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
    }

    /// Skip to a specific stream.
    pub fn skip_to(&self, stream_index: usize) {
        let mut state = self.inner.state.lock().unwrap();
        if stream_index < state.streams.len() {
            state.current_stream = stream_index;
            self.inner.data_available.notify_all();
        }
    }

    /// Skip to the next stream.
    pub fn skip_next(&self) {
        let mut state = self.inner.state.lock().unwrap();
        if state.current_stream < state.streams.len() {
            state.current_stream += 1;
            self.inner.data_available.notify_all();
        }
    }
}

impl Default for Concat {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for Concat {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        match self.pull_timeout(None)? {
            Some(buffer) => Ok(ProduceResult::OwnBuffer(buffer)),
            None => Ok(ProduceResult::Eos),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl ConcatStream {
    /// Push a buffer to this stream.
    pub fn push(&self, buffer: Buffer) -> Result<()> {
        let mut state = self.inner.state.lock().unwrap();

        if state.flushing || self.id >= state.streams.len() {
            return Ok(());
        }

        if state.streams[self.id].eos {
            return Ok(());
        }

        let max_buffers = state.max_buffers_per_stream;
        let current = state.current_stream;

        if state.streams[self.id].queue.len() < max_buffers {
            state.streams[self.id].queue.push_back(buffer);

            // Only notify if this is the current stream
            if current == self.id {
                self.inner.data_available.notify_one();
            }
        }

        Ok(())
    }

    /// Signal end of stream for this stream.
    pub fn end_stream(&self) {
        let mut state = self.inner.state.lock().unwrap();
        if self.id < state.streams.len() {
            state.streams[self.id].eos = true;

            // Notify if this is the current stream
            if state.current_stream == self.id {
                self.inner.data_available.notify_all();
            }
        }
    }

    /// Get the stream ID.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Check if this stream has ended.
    pub fn is_eos(&self) -> bool {
        let state = self.inner.state.lock().unwrap();
        self.id < state.streams.len() && state.streams[self.id].eos
    }
}

/// Statistics about concat operation.
#[derive(Debug, Clone, Copy)]
pub struct ConcatStats {
    /// Total number of streams.
    pub stream_count: usize,
    /// Currently playing stream index.
    pub current_stream: usize,
    /// Total buffers produced.
    pub total_produced: u64,
    /// Whether all streams have finished.
    pub is_eos: bool,
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
    fn test_concat_creation() {
        let concat = Concat::new();
        assert_eq!(concat.stream_count(), 0);
        assert!(concat.is_eos()); // No streams = EOS
    }

    #[test]
    fn test_concat_add_stream() {
        let concat = Concat::new();

        let stream0 = concat.add_stream();
        let stream1 = concat.add_stream();

        assert_eq!(concat.stream_count(), 2);
        assert_eq!(stream0.id(), 0);
        assert_eq!(stream1.id(), 1);
    }

    #[test]
    fn test_concat_sequential_playback() {
        let concat = Concat::new();
        let stream0 = concat.add_stream();
        let stream1 = concat.add_stream();

        // Add data to both streams
        stream0.push(create_test_buffer(0)).unwrap();
        stream0.push(create_test_buffer(1)).unwrap();
        stream0.end_stream();

        stream1.push(create_test_buffer(10)).unwrap();
        stream1.push(create_test_buffer(11)).unwrap();
        stream1.end_stream();

        // Should get stream0 buffers first
        let buf = concat
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(buf.unwrap().metadata().sequence, 0);

        let buf = concat
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(buf.unwrap().metadata().sequence, 1);

        // Now stream1 buffers
        let buf = concat
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(buf.unwrap().metadata().sequence, 10);

        let buf = concat
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(buf.unwrap().metadata().sequence, 11);

        // Should be EOS
        let buf = concat
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_none());
        assert!(concat.is_eos());
    }

    #[test]
    fn test_concat_skip_to() {
        let concat = Concat::new();
        let stream0 = concat.add_stream();
        let stream1 = concat.add_stream();

        stream0.push(create_test_buffer(0)).unwrap();
        stream0.end_stream();

        stream1.push(create_test_buffer(10)).unwrap();
        stream1.end_stream();

        // Skip to stream 1
        concat.skip_to(1);
        assert_eq!(concat.current_stream(), 1);

        // Should get stream1 buffer
        let buf = concat
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(buf.unwrap().metadata().sequence, 10);
    }

    #[test]
    fn test_concat_skip_next() {
        let concat = Concat::new();
        let stream0 = concat.add_stream();
        let stream1 = concat.add_stream();

        stream0.push(create_test_buffer(0)).unwrap();
        stream1.push(create_test_buffer(10)).unwrap();
        stream1.end_stream();

        // Skip to next stream
        concat.skip_next();
        assert_eq!(concat.current_stream(), 1);

        // Should get stream1 buffer
        let buf = concat
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(buf.unwrap().metadata().sequence, 10);
    }

    #[test]
    fn test_concat_multithreaded() {
        let mut concat = Concat::new();
        let stream0 = concat.add_stream();
        let stream1 = concat.add_stream();

        let producer0 = thread::spawn(move || {
            for i in 0..5 {
                stream0.push(create_test_buffer(i)).unwrap();
            }
            stream0.end_stream();
        });

        let producer1 = thread::spawn(move || {
            for i in 10..15 {
                stream1.push(create_test_buffer(i)).unwrap();
            }
            stream1.end_stream();
        });

        let mut received = Vec::new();
        let mut ctx = ProduceContext::without_buffer();
        loop {
            match concat.produce(&mut ctx).unwrap() {
                ProduceResult::OwnBuffer(buf) => {
                    received.push(buf.metadata().sequence);
                }
                ProduceResult::Eos => break,
                _ => {}
            }
        }

        producer0.join().unwrap();
        producer1.join().unwrap();

        // Should have all 10 buffers in order (0-4, then 10-14)
        assert_eq!(received.len(), 10);

        // First 5 should be from stream0
        for i in 0..5 {
            assert_eq!(received[i], i as u64);
        }

        // Next 5 should be from stream1
        for i in 0..5 {
            assert_eq!(received[5 + i], 10 + i as u64);
        }
    }

    #[test]
    fn test_concat_stats() {
        let concat = Concat::new();
        let stream0 = concat.add_stream();

        stream0.push(create_test_buffer(0)).unwrap();
        stream0.push(create_test_buffer(1)).unwrap();
        stream0.end_stream();

        concat
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();

        let stats = concat.stats();
        assert_eq!(stats.stream_count, 1);
        assert_eq!(stats.current_stream, 0);
        assert_eq!(stats.total_produced, 1);
    }
}
