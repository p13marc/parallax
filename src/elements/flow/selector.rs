//! Selector elements for stream routing.
//!
//! - `InputSelector`: N-to-1 stream selection
//! - `OutputSelector`: 1-to-N stream routing

use crate::buffer::Buffer;
use crate::element::{Element, ProduceContext, ProduceResult, Source};
use crate::error::Result;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

// ============================================================================
// InputSelector - N-to-1 selection
// ============================================================================

/// An input selector that routes one of N inputs to a single output.
///
/// Only buffers from the currently selected input are passed through.
/// Buffers from non-selected inputs are dropped.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::InputSelector;
///
/// let selector = InputSelector::new();
///
/// let input0 = selector.new_input();
/// let input1 = selector.new_input();
///
/// // Select input 1
/// selector.select(1);
///
/// // Only buffers from input1 will be passed through
/// ```
pub struct InputSelector {
    name: String,
    inner: Arc<InputSelectorInner>,
}

struct InputSelectorInner {
    state: Mutex<InputSelectorState>,
    data_available: Condvar,
    selected: AtomicUsize,
}

struct InputSelectorState {
    queues: Vec<VecDeque<Buffer>>,
    max_buffers: usize,
    active_inputs: Vec<bool>,
    total_received: u64,
    total_produced: u64,
    total_dropped: u64,
    flushing: bool,
}

/// Input handle for InputSelector.
#[derive(Clone)]
pub struct SelectorInput {
    inner: Arc<InputSelectorInner>,
    id: usize,
}

impl InputSelector {
    /// Create a new input selector.
    pub fn new() -> Self {
        Self::with_max_buffers(64)
    }

    /// Create a new input selector with a specific buffer limit per input.
    pub fn with_max_buffers(max_buffers: usize) -> Self {
        Self {
            name: "input-selector".to_string(),
            inner: Arc::new(InputSelectorInner {
                state: Mutex::new(InputSelectorState {
                    queues: Vec::new(),
                    max_buffers,
                    active_inputs: Vec::new(),
                    total_received: 0,
                    total_produced: 0,
                    total_dropped: 0,
                    flushing: false,
                }),
                data_available: Condvar::new(),
                selected: AtomicUsize::new(0),
            }),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Create a new input and return its handle.
    pub fn new_input(&self) -> SelectorInput {
        let mut state = self.inner.state.lock().unwrap();
        let id = state.queues.len();
        let capacity = state.max_buffers.min(64);
        state.queues.push(VecDeque::with_capacity(capacity));
        state.active_inputs.push(true);

        SelectorInput {
            inner: Arc::clone(&self.inner),
            id,
        }
    }

    /// Select which input to use (0-indexed).
    pub fn select(&self, input: usize) {
        self.inner.selected.store(input, Ordering::SeqCst);
        self.inner.data_available.notify_all();
    }

    /// Get the currently selected input.
    pub fn selected(&self) -> usize {
        self.inner.selected.load(Ordering::SeqCst)
    }

    /// Get the number of inputs.
    pub fn input_count(&self) -> usize {
        self.inner.state.lock().unwrap().queues.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> InputSelectorStats {
        let state = self.inner.state.lock().unwrap();
        InputSelectorStats {
            input_count: state.queues.len(),
            selected: self.inner.selected.load(Ordering::SeqCst),
            total_received: state.total_received,
            total_produced: state.total_produced,
            total_dropped: state.total_dropped,
        }
    }

    /// Pull from the selected input with timeout.
    pub fn pull_timeout(&self, timeout: Option<Duration>) -> Result<Option<Buffer>> {
        let mut state = self.inner.state.lock().unwrap();

        loop {
            let selected = self.inner.selected.load(Ordering::SeqCst);

            if state.flushing {
                return Ok(None);
            }

            // Check if selected input exists and has data
            if selected < state.queues.len() {
                if let Some(buffer) = state.queues[selected].pop_front() {
                    state.total_produced += 1;
                    return Ok(Some(buffer));
                }

                // Check if selected input is still active
                if !state.active_inputs[selected] {
                    return Ok(None); // EOS on selected input
                }
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
}

impl Default for InputSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for InputSelector {
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

impl SelectorInput {
    /// Push a buffer to this input.
    pub fn push(&self, buffer: Buffer) -> Result<()> {
        let mut state = self.inner.state.lock().unwrap();

        if state.flushing || self.id >= state.queues.len() {
            return Ok(());
        }

        state.total_received += 1;

        let selected = self.inner.selected.load(Ordering::SeqCst);

        // Only queue if this is the selected input
        if self.id == selected {
            if state.queues[self.id].len() < state.max_buffers {
                state.queues[self.id].push_back(buffer);
                self.inner.data_available.notify_one();
            } else {
                state.total_dropped += 1;
            }
        } else {
            // Drop buffers from non-selected inputs
            state.total_dropped += 1;
        }

        Ok(())
    }

    /// Signal end of stream for this input.
    pub fn end_stream(&self) {
        let mut state = self.inner.state.lock().unwrap();
        if self.id < state.active_inputs.len() {
            state.active_inputs[self.id] = false;
            self.inner.data_available.notify_all();
        }
    }

    /// Get the input ID.
    pub fn id(&self) -> usize {
        self.id
    }
}

/// Statistics for InputSelector.
#[derive(Debug, Clone, Copy)]
pub struct InputSelectorStats {
    /// Number of inputs.
    pub input_count: usize,
    /// Currently selected input.
    pub selected: usize,
    /// Total buffers received.
    pub total_received: u64,
    /// Total buffers produced.
    pub total_produced: u64,
    /// Total buffers dropped.
    pub total_dropped: u64,
}

// ============================================================================
// OutputSelector - 1-to-N routing
// ============================================================================

/// An output selector that routes a single input to one of N outputs.
///
/// Buffers are sent to the currently selected output only.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::OutputSelector;
///
/// let selector = OutputSelector::new();
///
/// let output0 = selector.new_output();
/// let output1 = selector.new_output();
///
/// // Select output 1
/// selector.select(1);
///
/// // Buffers will be routed to output1
/// ```
pub struct OutputSelector {
    name: String,
    inner: Arc<OutputSelectorInner>,
}

struct OutputSelectorInner {
    state: Mutex<OutputSelectorState>,
    selected: AtomicUsize,
}

struct OutputSelectorState {
    outputs: Vec<Arc<OutputQueue>>,
    total_received: u64,
    total_routed: u64,
}

struct OutputQueue {
    state: Mutex<OutputQueueState>,
    data_available: Condvar,
}

struct OutputQueueState {
    queue: VecDeque<Buffer>,
    max_buffers: usize,
    eos: bool,
}

/// Output handle for OutputSelector.
pub struct SelectorOutput {
    queue: Arc<OutputQueue>,
    id: usize,
}

impl OutputSelector {
    /// Create a new output selector.
    pub fn new() -> Self {
        Self {
            name: "output-selector".to_string(),
            inner: Arc::new(OutputSelectorInner {
                state: Mutex::new(OutputSelectorState {
                    outputs: Vec::new(),
                    total_received: 0,
                    total_routed: 0,
                }),
                selected: AtomicUsize::new(0),
            }),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Create a new output and return its handle.
    pub fn new_output(&self) -> SelectorOutput {
        self.new_output_with_max_buffers(64)
    }

    /// Create a new output with a specific buffer limit.
    pub fn new_output_with_max_buffers(&self, max_buffers: usize) -> SelectorOutput {
        let mut state = self.inner.state.lock().unwrap();
        let id = state.outputs.len();

        let queue = Arc::new(OutputQueue {
            state: Mutex::new(OutputQueueState {
                queue: VecDeque::with_capacity(max_buffers.min(64)),
                max_buffers,
                eos: false,
            }),
            data_available: Condvar::new(),
        });

        state.outputs.push(Arc::clone(&queue));

        SelectorOutput { queue, id }
    }

    /// Select which output to use (0-indexed).
    pub fn select(&self, output: usize) {
        self.inner.selected.store(output, Ordering::SeqCst);
    }

    /// Get the currently selected output.
    pub fn selected(&self) -> usize {
        self.inner.selected.load(Ordering::SeqCst)
    }

    /// Get the number of outputs.
    pub fn output_count(&self) -> usize {
        self.inner.state.lock().unwrap().outputs.len()
    }

    /// Signal EOS to all outputs.
    pub fn send_eos(&self) {
        let state = self.inner.state.lock().unwrap();
        for output in &state.outputs {
            let mut ostate = output.state.lock().unwrap();
            ostate.eos = true;
            output.data_available.notify_all();
        }
    }

    /// Get statistics.
    pub fn stats(&self) -> OutputSelectorStats {
        let state = self.inner.state.lock().unwrap();
        OutputSelectorStats {
            output_count: state.outputs.len(),
            selected: self.inner.selected.load(Ordering::SeqCst),
            total_received: state.total_received,
            total_routed: state.total_routed,
        }
    }
}

impl Default for OutputSelector {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for OutputSelector {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let selected = self.inner.selected.load(Ordering::SeqCst);
        let mut state = self.inner.state.lock().unwrap();

        state.total_received += 1;

        if selected < state.outputs.len() {
            let output = Arc::clone(&state.outputs[selected]);
            let mut ostate = output.state.lock().unwrap();

            if ostate.queue.len() < ostate.max_buffers {
                ostate.queue.push_back(buffer);
                state.total_routed += 1;
                drop(ostate);
                output.data_available.notify_one();
            }
        }

        Ok(None) // OutputSelector consumes buffers
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl SelectorOutput {
    /// Pull a buffer from this output.
    pub fn pull(&self) -> Result<Option<Buffer>> {
        self.pull_timeout(None)
    }

    /// Pull a buffer with timeout.
    pub fn pull_timeout(&self, timeout: Option<Duration>) -> Result<Option<Buffer>> {
        let mut state = self.queue.state.lock().unwrap();

        while state.queue.is_empty() && !state.eos {
            state = if let Some(t) = timeout {
                let (s, result) = self.queue.data_available.wait_timeout(state, t).unwrap();
                if result.timed_out() {
                    return Ok(None);
                }
                s
            } else {
                self.queue.data_available.wait(state).unwrap()
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
        self.queue.state.lock().unwrap().queue.pop_front()
    }

    /// Get the output ID.
    pub fn id(&self) -> usize {
        self.id
    }

    /// Get the queue length.
    pub fn queue_len(&self) -> usize {
        self.queue.state.lock().unwrap().queue.len()
    }
}

/// Statistics for OutputSelector.
#[derive(Debug, Clone, Copy)]
pub struct OutputSelectorStats {
    /// Number of outputs.
    pub output_count: usize,
    /// Currently selected output.
    pub selected: usize,
    /// Total buffers received.
    pub total_received: u64,
    /// Total buffers routed to outputs.
    pub total_routed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::CpuSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    fn create_test_buffer(seq: u64) -> Buffer {
        let segment = Arc::new(CpuSegment::new(100).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    // InputSelector tests

    #[test]
    fn test_input_selector_creation() {
        let selector = InputSelector::new();
        assert_eq!(selector.input_count(), 0);
        assert_eq!(selector.selected(), 0);
    }

    #[test]
    fn test_input_selector_new_input() {
        let selector = InputSelector::new();

        let input0 = selector.new_input();
        let input1 = selector.new_input();

        assert_eq!(selector.input_count(), 2);
        assert_eq!(input0.id(), 0);
        assert_eq!(input1.id(), 1);
    }

    #[test]
    fn test_input_selector_routing() {
        let selector = InputSelector::new();
        let input0 = selector.new_input();
        let input1 = selector.new_input();

        // Select input 0 (default)
        input0.push(create_test_buffer(0)).unwrap();
        input1.push(create_test_buffer(10)).unwrap(); // Should be dropped

        let buf = selector
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 0);

        // Switch to input 1
        selector.select(1);

        input0.push(create_test_buffer(1)).unwrap(); // Should be dropped
        input1.push(create_test_buffer(11)).unwrap();

        let buf = selector
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 11);
    }

    #[test]
    fn test_input_selector_stats() {
        let selector = InputSelector::new();
        let input0 = selector.new_input();
        let input1 = selector.new_input();

        input0.push(create_test_buffer(0)).unwrap();
        input1.push(create_test_buffer(1)).unwrap(); // Dropped
        selector
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();

        let stats = selector.stats();
        assert_eq!(stats.total_received, 2);
        assert_eq!(stats.total_produced, 1);
        assert_eq!(stats.total_dropped, 1);
    }

    // OutputSelector tests

    #[test]
    fn test_output_selector_creation() {
        let selector = OutputSelector::new();
        assert_eq!(selector.output_count(), 0);
        assert_eq!(selector.selected(), 0);
    }

    #[test]
    fn test_output_selector_new_output() {
        let selector = OutputSelector::new();

        let output0 = selector.new_output();
        let output1 = selector.new_output();

        assert_eq!(selector.output_count(), 2);
        assert_eq!(output0.id(), 0);
        assert_eq!(output1.id(), 1);
    }

    #[test]
    fn test_output_selector_routing() {
        let mut selector = OutputSelector::new();
        let output0 = selector.new_output();
        let output1 = selector.new_output();

        // Select output 0 (default)
        selector.process(create_test_buffer(0)).unwrap();

        assert_eq!(output0.queue_len(), 1);
        assert_eq!(output1.queue_len(), 0);

        // Switch to output 1
        selector.select(1);
        selector.process(create_test_buffer(1)).unwrap();

        assert_eq!(output0.queue_len(), 1);
        assert_eq!(output1.queue_len(), 1);

        // Pull from outputs
        let buf = output0.try_pull().unwrap();
        assert_eq!(buf.metadata().sequence, 0);

        let buf = output1.try_pull().unwrap();
        assert_eq!(buf.metadata().sequence, 1);
    }

    #[test]
    fn test_output_selector_eos() {
        let mut selector = OutputSelector::new();
        let output0 = selector.new_output();

        selector.process(create_test_buffer(0)).unwrap();
        selector.send_eos();

        // Should get the buffer
        let buf = output0
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_some());

        // Should get None (EOS)
        let buf = output0
            .pull_timeout(Some(Duration::from_millis(100)))
            .unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn test_output_selector_stats() {
        let mut selector = OutputSelector::new();
        let _output0 = selector.new_output();

        selector.process(create_test_buffer(0)).unwrap();
        selector.process(create_test_buffer(1)).unwrap();

        let stats = selector.stats();
        assert_eq!(stats.total_received, 2);
        assert_eq!(stats.total_routed, 2);
    }
}
