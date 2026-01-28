//! Funnel element for merging multiple inputs into one output.
//!
//! N-to-1 pipe fitting that combines buffers from multiple sources.

use crate::buffer::Buffer;
use crate::element::{ProduceContext, ProduceResult, Source};
use crate::error::Result;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A funnel element that merges multiple inputs into a single output.
///
/// Buffers from all inputs are interleaved in arrival order.
/// This is the complement to `Tee` which splits one output to many.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Funnel;
///
/// let funnel = Funnel::new();
///
/// // Get input handles for different sources
/// let input1 = funnel.new_input();
/// let input2 = funnel.new_input();
///
/// // Push from different sources
/// input1.push(buffer1)?;
/// input2.push(buffer2)?;
///
/// // Pull merged output
/// let buffer = funnel.produce()?;
/// ```
pub struct Funnel {
    name: String,
    inner: Arc<FunnelInner>,
}

struct FunnelInner {
    state: Mutex<FunnelState>,
    data_available: Condvar,
}

struct FunnelState {
    queue: VecDeque<Buffer>,
    max_buffers: usize,
    input_count: usize,
    active_inputs: usize,
    eos_received: usize,
    total_received: u64,
    total_produced: u64,
    flushing: bool,
}

/// Input handle for pushing data into a Funnel.
#[derive(Clone)]
pub struct FunnelInput {
    inner: Arc<FunnelInner>,
    id: usize,
    eos_sent: Arc<Mutex<bool>>,
}

impl Funnel {
    /// Create a new funnel with default settings.
    pub fn new() -> Self {
        Self::with_max_buffers(256)
    }

    /// Create a new funnel with a specific queue size.
    pub fn with_max_buffers(max_buffers: usize) -> Self {
        Self {
            name: "funnel".to_string(),
            inner: Arc::new(FunnelInner {
                state: Mutex::new(FunnelState {
                    queue: VecDeque::with_capacity(max_buffers.min(256)),
                    max_buffers,
                    input_count: 0,
                    active_inputs: 0,
                    eos_received: 0,
                    total_received: 0,
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

    /// Create a new input handle.
    pub fn new_input(&self) -> FunnelInput {
        let mut state = self.inner.state.lock().unwrap();
        let id = state.input_count;
        state.input_count += 1;
        state.active_inputs += 1;

        FunnelInput {
            inner: Arc::clone(&self.inner),
            id,
            eos_sent: Arc::new(Mutex::new(false)),
        }
    }

    /// Get the number of inputs.
    pub fn input_count(&self) -> usize {
        self.inner.state.lock().unwrap().input_count
    }

    /// Get the number of active (non-EOS) inputs.
    pub fn active_inputs(&self) -> usize {
        self.inner.state.lock().unwrap().active_inputs
    }

    /// Get the current queue length.
    pub fn queue_len(&self) -> usize {
        self.inner.state.lock().unwrap().queue.len()
    }

    /// Check if all inputs have sent EOS.
    pub fn is_eos(&self) -> bool {
        let state = self.inner.state.lock().unwrap();
        state.active_inputs == 0 && state.queue.is_empty()
    }

    /// Get statistics.
    pub fn stats(&self) -> FunnelStats {
        let state = self.inner.state.lock().unwrap();
        FunnelStats {
            input_count: state.input_count,
            active_inputs: state.active_inputs,
            queued_buffers: state.queue.len(),
            total_received: state.total_received,
            total_produced: state.total_produced,
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

        // Wait for data or all inputs EOS
        while state.queue.is_empty() && state.active_inputs > 0 && !state.flushing {
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
            return Ok(None);
        }

        if let Some(buffer) = state.queue.pop_front() {
            state.total_produced += 1;
            Ok(Some(buffer))
        } else {
            // All inputs EOS and queue empty
            Ok(None)
        }
    }
}

impl Default for Funnel {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for Funnel {
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

impl FunnelInput {
    /// Push a buffer into the funnel.
    pub fn push(&self, buffer: Buffer) -> Result<()> {
        let mut state = self.inner.state.lock().unwrap();

        if state.flushing {
            return Ok(());
        }

        // Simple drop policy if full (could be made configurable)
        if state.queue.len() < state.max_buffers {
            state.queue.push_back(buffer);
            state.total_received += 1;
            self.inner.data_available.notify_one();
        }

        Ok(())
    }

    /// Signal end of stream for this input.
    pub fn end_stream(&self) {
        let mut eos_sent = self.eos_sent.lock().unwrap();
        if *eos_sent {
            return; // Already sent EOS
        }
        *eos_sent = true;

        let mut state = self.inner.state.lock().unwrap();
        state.active_inputs = state.active_inputs.saturating_sub(1);
        state.eos_received += 1;

        // Notify in case we're waiting and this was the last input
        self.inner.data_available.notify_all();
    }

    /// Get the input ID.
    pub fn id(&self) -> usize {
        self.id
    }
}

impl Drop for FunnelInput {
    fn drop(&mut self) {
        // Ensure EOS is sent when the input handle is dropped
        self.end_stream();
    }
}

/// Statistics about funnel operation.
#[derive(Debug, Clone, Copy)]
pub struct FunnelStats {
    /// Total number of inputs created.
    pub input_count: usize,
    /// Number of active (non-EOS) inputs.
    pub active_inputs: usize,
    /// Buffers currently in the queue.
    pub queued_buffers: usize,
    /// Total buffers received from all inputs.
    pub total_received: u64,
    /// Total buffers produced to output.
    pub total_produced: u64,
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
        ARENA.get_or_init(|| SharedArena::new(128, 256).unwrap())
    }

    fn create_test_buffer(seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 100);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn test_funnel_creation() {
        let funnel = Funnel::new();
        assert_eq!(funnel.input_count(), 0);
        assert_eq!(funnel.queue_len(), 0);
    }

    #[test]
    fn test_funnel_new_input() {
        let funnel = Funnel::new();

        let input1 = funnel.new_input();
        let input2 = funnel.new_input();

        assert_eq!(funnel.input_count(), 2);
        assert_eq!(funnel.active_inputs(), 2);
        assert_eq!(input1.id(), 0);
        assert_eq!(input2.id(), 1);
    }

    #[test]
    fn test_funnel_push_pull() {
        let funnel = Funnel::new();
        let input = funnel.new_input();

        input.push(create_test_buffer(0)).unwrap();
        input.push(create_test_buffer(1)).unwrap();

        let buf = funnel
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 0);

        let buf = funnel
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 1);
    }

    #[test]
    fn test_funnel_multiple_inputs() {
        let funnel = Funnel::new();
        let input1 = funnel.new_input();
        let input2 = funnel.new_input();

        input1.push(create_test_buffer(0)).unwrap();
        input2.push(create_test_buffer(1)).unwrap();
        input1.push(create_test_buffer(2)).unwrap();

        assert_eq!(funnel.queue_len(), 3);

        let mut sequences = Vec::new();
        for _ in 0..3 {
            let buf = funnel
                .pull_timeout(Some(Duration::from_millis(100)))
                .unwrap();
            sequences.push(buf.unwrap().metadata().sequence);
        }

        // All buffers should be received
        assert_eq!(sequences.len(), 3);
        assert!(sequences.contains(&0));
        assert!(sequences.contains(&1));
        assert!(sequences.contains(&2));
    }

    #[test]
    fn test_funnel_eos() {
        let funnel = Funnel::new();
        let input1 = funnel.new_input();
        let input2 = funnel.new_input();

        input1.push(create_test_buffer(0)).unwrap();
        input1.end_stream();

        assert_eq!(funnel.active_inputs(), 1);

        input2.end_stream();

        assert_eq!(funnel.active_inputs(), 0);

        // Should still get buffered data
        let buf = funnel
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_some());

        // Now should get None (all inputs EOS)
        let buf = funnel
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn test_funnel_input_drop() {
        let funnel = Funnel::new();

        {
            let _input1 = funnel.new_input();
            let _input2 = funnel.new_input();
            assert_eq!(funnel.active_inputs(), 2);
        }

        // Inputs dropped, should auto-EOS
        assert_eq!(funnel.active_inputs(), 0);
    }

    #[test]
    fn test_funnel_multithreaded() {
        let mut funnel = Funnel::new();
        let input1 = funnel.new_input();
        let input2 = funnel.new_input();

        let producer1 = thread::spawn(move || {
            for i in 0..5 {
                input1.push(create_test_buffer(i)).unwrap();
            }
            input1.end_stream();
        });

        let producer2 = thread::spawn(move || {
            for i in 10..15 {
                input2.push(create_test_buffer(i)).unwrap();
            }
            input2.end_stream();
        });

        let mut received = Vec::new();
        let mut ctx = ProduceContext::without_buffer();
        loop {
            match funnel.produce(&mut ctx) {
                Ok(ProduceResult::OwnBuffer(buf)) => {
                    received.push(buf.metadata().sequence);
                }
                Ok(ProduceResult::Eos) => break,
                Ok(_) => break,
                Err(_) => break,
            }
        }

        producer1.join().unwrap();
        producer2.join().unwrap();

        assert_eq!(received.len(), 10);
    }

    #[test]
    fn test_funnel_stats() {
        let funnel = Funnel::new();
        let input = funnel.new_input();

        input.push(create_test_buffer(0)).unwrap();
        input.push(create_test_buffer(1)).unwrap();
        funnel
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();

        let stats = funnel.stats();
        assert_eq!(stats.input_count, 1);
        assert_eq!(stats.total_received, 2);
        assert_eq!(stats.total_produced, 1);
    }
}
