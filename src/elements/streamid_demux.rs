//! StreamIdDemux element for demultiplexing by stream ID.
//!
//! Routes buffers to different outputs based on a stream identifier.

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// Function type for extracting stream ID from a buffer.
pub type StreamIdExtractor = Box<dyn Fn(&Buffer) -> u64 + Send + Sync>;

/// A demuxer that routes buffers to outputs based on stream ID.
///
/// Each unique stream ID gets its own output. New outputs are created
/// automatically when new stream IDs are encountered.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::StreamIdDemux;
///
/// // Demux by buffer sequence number modulo 2 (even/odd)
/// let demux = StreamIdDemux::new(|buf| buf.metadata().sequence % 2);
///
/// // Or use metadata extra field as stream ID
/// let demux = StreamIdDemux::new(|buf| {
///     buf.metadata().extra.get("stream_id")
///         .and_then(|v| v.parse().ok())
///         .unwrap_or(0)
/// });
/// ```
pub struct StreamIdDemux {
    name: String,
    inner: Arc<StreamIdDemuxInner>,
    extractor: StreamIdExtractor,
}

struct StreamIdDemuxInner {
    state: Mutex<StreamIdDemuxState>,
}

struct StreamIdDemuxState {
    outputs: HashMap<u64, Arc<DemuxOutput>>,
    max_buffers_per_output: usize,
    total_received: u64,
    total_routed: u64,
    eos: bool,
}

struct DemuxOutput {
    state: Mutex<DemuxOutputState>,
    data_available: Condvar,
}

struct DemuxOutputState {
    queue: VecDeque<Buffer>,
    max_buffers: usize,
    eos: bool,
}

/// Handle for pulling buffers from a specific stream.
pub struct StreamOutput {
    stream_id: u64,
    output: Arc<DemuxOutput>,
}

impl StreamIdDemux {
    /// Create a new stream ID demuxer with an extractor function.
    pub fn new<F>(extractor: F) -> Self
    where
        F: Fn(&Buffer) -> u64 + Send + Sync + 'static,
    {
        Self::with_max_buffers(extractor, 64)
    }

    /// Create a new stream ID demuxer with a specific buffer limit per output.
    pub fn with_max_buffers<F>(extractor: F, max_buffers: usize) -> Self
    where
        F: Fn(&Buffer) -> u64 + Send + Sync + 'static,
    {
        Self {
            name: "streamid-demux".to_string(),
            inner: Arc::new(StreamIdDemuxInner {
                state: Mutex::new(StreamIdDemuxState {
                    outputs: HashMap::new(),
                    max_buffers_per_output: max_buffers,
                    total_received: 0,
                    total_routed: 0,
                    eos: false,
                }),
            }),
            extractor: Box::new(extractor),
        }
    }

    /// Create a demuxer that uses the buffer sequence number as stream ID.
    pub fn by_sequence() -> Self {
        Self::new(|buf| buf.metadata().sequence)
    }

    /// Create a demuxer that routes based on sequence modulo N.
    pub fn by_sequence_mod(n: u64) -> Self {
        Self::new(move |buf| buf.metadata().sequence % n)
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the output for a specific stream ID.
    ///
    /// Creates the output if it doesn't exist.
    pub fn get_output(&self, stream_id: u64) -> StreamOutput {
        let mut state = self.inner.state.lock().unwrap();
        let max_buffers = state.max_buffers_per_output;

        let output = state
            .outputs
            .entry(stream_id)
            .or_insert_with(|| {
                Arc::new(DemuxOutput {
                    state: Mutex::new(DemuxOutputState {
                        queue: VecDeque::with_capacity(max_buffers.min(64)),
                        max_buffers,
                        eos: false,
                    }),
                    data_available: Condvar::new(),
                })
            })
            .clone();

        StreamOutput { stream_id, output }
    }

    /// Get all known stream IDs.
    pub fn stream_ids(&self) -> Vec<u64> {
        self.inner
            .state
            .lock()
            .unwrap()
            .outputs
            .keys()
            .copied()
            .collect()
    }

    /// Get the number of streams.
    pub fn stream_count(&self) -> usize {
        self.inner.state.lock().unwrap().outputs.len()
    }

    /// Signal end of stream to all outputs.
    pub fn send_eos(&self) {
        let mut state = self.inner.state.lock().unwrap();
        state.eos = true;

        for output in state.outputs.values() {
            let mut ostate = output.state.lock().unwrap();
            ostate.eos = true;
            output.data_available.notify_all();
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> StreamIdDemuxStats {
        let state = self.inner.state.lock().unwrap();
        StreamIdDemuxStats {
            stream_count: state.outputs.len(),
            total_received: state.total_received,
            total_routed: state.total_routed,
        }
    }
}

impl Element for StreamIdDemux {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let stream_id = (self.extractor)(&buffer);

        let mut state = self.inner.state.lock().unwrap();
        state.total_received += 1;
        let max_buffers = state.max_buffers_per_output;

        // Get or create output for this stream ID
        let output = state
            .outputs
            .entry(stream_id)
            .or_insert_with(|| {
                Arc::new(DemuxOutput {
                    state: Mutex::new(DemuxOutputState {
                        queue: VecDeque::with_capacity(max_buffers.min(64)),
                        max_buffers,
                        eos: false,
                    }),
                    data_available: Condvar::new(),
                })
            })
            .clone();

        // Push to output queue
        let mut ostate = output.state.lock().unwrap();
        if ostate.queue.len() < ostate.max_buffers {
            ostate.queue.push_back(buffer);
            state.total_routed += 1;
            drop(ostate);
            output.data_available.notify_one();
        }

        Ok(None) // Demux consumes buffers
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl StreamOutput {
    /// Get the stream ID for this output.
    pub fn stream_id(&self) -> u64 {
        self.stream_id
    }

    /// Pull a buffer from this stream.
    pub fn pull(&self) -> Result<Option<Buffer>> {
        self.pull_timeout(None)
    }

    /// Pull a buffer with timeout.
    pub fn pull_timeout(&self, timeout: Option<Duration>) -> Result<Option<Buffer>> {
        let mut state = self.output.state.lock().unwrap();

        while state.queue.is_empty() && !state.eos {
            state = if let Some(t) = timeout {
                let (s, result) = self.output.data_available.wait_timeout(state, t).unwrap();
                if result.timed_out() {
                    return Ok(None);
                }
                s
            } else {
                self.output.data_available.wait(state).unwrap()
            };
        }

        if let Some(buffer) = state.queue.pop_front() {
            Ok(Some(buffer))
        } else {
            Ok(None)
        }
    }

    /// Try to pull without blocking.
    pub fn try_pull(&self) -> Option<Buffer> {
        self.output.state.lock().unwrap().queue.pop_front()
    }

    /// Get the queue length.
    pub fn queue_len(&self) -> usize {
        self.output.state.lock().unwrap().queue.len()
    }

    /// Check if EOS has been received.
    pub fn is_eos(&self) -> bool {
        self.output.state.lock().unwrap().eos
    }
}

/// Statistics for StreamIdDemux.
#[derive(Debug, Clone, Copy)]
pub struct StreamIdDemuxStats {
    /// Number of unique streams.
    pub stream_count: usize,
    /// Total buffers received.
    pub total_received: u64,
    /// Total buffers routed to outputs.
    pub total_routed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    fn create_test_buffer(seq: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(100).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::with_sequence(seq))
    }

    #[test]
    fn test_streamid_demux_creation() {
        let demux = StreamIdDemux::new(|buf| buf.metadata().sequence);
        assert_eq!(demux.stream_count(), 0);
    }

    #[test]
    fn test_streamid_demux_routing() {
        let mut demux = StreamIdDemux::by_sequence_mod(2);

        // Get outputs for even and odd
        let even_output = demux.get_output(0);
        let odd_output = demux.get_output(1);

        // Process buffers
        demux.process(create_test_buffer(0)).unwrap(); // -> even
        demux.process(create_test_buffer(1)).unwrap(); // -> odd
        demux.process(create_test_buffer(2)).unwrap(); // -> even
        demux.process(create_test_buffer(3)).unwrap(); // -> odd

        assert_eq!(even_output.queue_len(), 2);
        assert_eq!(odd_output.queue_len(), 2);

        // Pull and verify
        let buf = even_output.try_pull().unwrap();
        assert_eq!(buf.metadata().sequence, 0);

        let buf = even_output.try_pull().unwrap();
        assert_eq!(buf.metadata().sequence, 2);

        let buf = odd_output.try_pull().unwrap();
        assert_eq!(buf.metadata().sequence, 1);

        let buf = odd_output.try_pull().unwrap();
        assert_eq!(buf.metadata().sequence, 3);
    }

    #[test]
    fn test_streamid_demux_auto_create() {
        let mut demux = StreamIdDemux::by_sequence();

        // Process buffers - outputs created automatically
        demux.process(create_test_buffer(0)).unwrap();
        demux.process(create_test_buffer(1)).unwrap();
        demux.process(create_test_buffer(2)).unwrap();

        assert_eq!(demux.stream_count(), 3);

        let ids = demux.stream_ids();
        assert!(ids.contains(&0));
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn test_streamid_demux_eos() {
        let mut demux = StreamIdDemux::by_sequence_mod(2);

        let output = demux.get_output(0);
        demux.process(create_test_buffer(0)).unwrap();
        demux.send_eos();

        assert!(output.is_eos());

        // Should still get buffered data
        let buf = output
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_some());

        // Should get None (EOS)
        let buf = output
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn test_streamid_demux_stats() {
        let mut demux = StreamIdDemux::by_sequence_mod(2);

        demux.process(create_test_buffer(0)).unwrap();
        demux.process(create_test_buffer(1)).unwrap();
        demux.process(create_test_buffer(2)).unwrap();

        let stats = demux.stats();
        assert_eq!(stats.stream_count, 2);
        assert_eq!(stats.total_received, 3);
        assert_eq!(stats.total_routed, 3);
    }

    #[test]
    fn test_streamid_demux_custom_extractor() {
        // Route based on buffer length
        let mut demux = StreamIdDemux::new(|buf| buf.len() as u64);

        let output_100 = demux.get_output(100);

        demux.process(create_test_buffer(0)).unwrap();

        assert_eq!(output_100.queue_len(), 1);
    }

    #[test]
    fn test_streamid_demux_with_name() {
        let demux = StreamIdDemux::by_sequence().with_name("my-demux");
        assert_eq!(demux.name(), "my-demux");
    }
}
