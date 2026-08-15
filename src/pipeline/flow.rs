//! Flow control for pipeline backpressure management.
//!
//! In a task-per-element engine the link channel is the queue, so flow
//! signals are produced by the **executor** from link-channel occupancy
//! (see [`Pipeline::monitor_link`](crate::pipeline::Pipeline::monitor_link))
//! and consumed by live sources polling a shared [`FlowStateHandle`]:
//!
//! ```text
//! Producer ──link channel──> Consumer
//!     ^            │occupancy sampled each send/recv
//!     │            v
//!     └── FlowStateHandle (Busy/Ready, watermark hysteresis)
//! ```
//!
//! A source that received a handle via `set_flow_state` checks
//! `should_produce()` before doing capture work and skips the frame while
//! the signal is [`FlowSignal::Busy`] — cheaper than `LinkPolicy::DropNewest`
//! alone, which can only discard the frame *after* it was captured and
//! copied. At the low watermark the signal returns to
//! [`FlowSignal::Ready`] and production resumes.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Flow control signal from downstream to upstream.
///
/// These signals propagate backward through the pipeline to inform
/// producers about the state of downstream consumers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(u8)]
pub enum FlowSignal {
    /// Normal operation - continue producing data.
    #[default]
    Ready = 0,

    /// Downstream is busy - skip or defer production until Ready.
    Busy = 1,
}

impl FlowSignal {
    /// Check if production should continue.
    #[inline]
    pub fn should_produce(&self) -> bool {
        matches!(self, FlowSignal::Ready)
    }

    /// Check if the signal indicates backpressure.
    #[inline]
    pub fn is_backpressure(&self) -> bool {
        matches!(self, FlowSignal::Busy)
    }
}

impl From<u8> for FlowSignal {
    fn from(value: u8) -> Self {
        match value {
            1 => FlowSignal::Busy,
            _ => FlowSignal::Ready, // Safe default
        }
    }
}

impl From<FlowSignal> for u8 {
    fn from(signal: FlowSignal) -> Self {
        signal as u8
    }
}

/// Shared flow state for thread-safe signal passing.
///
/// This is used when flow signals need to be passed between threads
/// (e.g., from executor to source task).
#[derive(Debug)]
pub struct SharedFlowState {
    /// Current flow signal (atomic for lock-free access).
    signal: AtomicU32,
    /// Frames dropped counter.
    frames_dropped: AtomicU64,
    /// Backpressure events counter.
    backpressure_events: AtomicU64,
}

impl Default for SharedFlowState {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedFlowState {
    /// Create new shared flow state.
    pub fn new() -> Self {
        Self {
            signal: AtomicU32::new(FlowSignal::Ready as u32),
            frames_dropped: AtomicU64::new(0),
            backpressure_events: AtomicU64::new(0),
        }
    }

    /// Get the current flow signal.
    #[inline]
    pub fn signal(&self) -> FlowSignal {
        FlowSignal::from(self.signal.load(Ordering::Acquire) as u8)
    }

    /// Set the flow signal.
    #[inline]
    pub fn set_signal(&self, signal: FlowSignal) {
        let old = self.signal.swap(signal as u32, Ordering::Release);
        // Track backpressure events
        if signal == FlowSignal::Busy && FlowSignal::from(old as u8) != FlowSignal::Busy {
            self.backpressure_events.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Check if production should continue.
    #[inline]
    pub fn should_produce(&self) -> bool {
        self.signal().should_produce()
    }

    /// Record a dropped frame.
    pub fn record_drop(&self) {
        self.frames_dropped.fetch_add(1, Ordering::Relaxed);
    }

    /// Get the number of dropped frames.
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped.load(Ordering::Relaxed)
    }

    /// Get the number of backpressure events.
    pub fn backpressure_events(&self) -> u64 {
        self.backpressure_events.load(Ordering::Relaxed)
    }
}

/// Handle to shared flow state.
pub type FlowStateHandle = Arc<SharedFlowState>;

/// Create a new flow state handle.
pub fn new_flow_state() -> FlowStateHandle {
    Arc::new(SharedFlowState::new())
}

/// Watermark-driven [`FlowSignal`] production for one monitored link.
///
/// The executor samples the link channel's occupancy on **both** sides —
/// after every send and after every receive — and folds it through the
/// hysteresis below into the shared [`FlowStateHandle`] that
/// [`Pipeline::monitor_link`](crate::pipeline::Pipeline::monitor_link)
/// returned. Sender-only sampling would deadlock the Ready transition: a
/// gated source stops sending, so occupancy would never be re-sampled.
#[derive(Debug)]
pub(crate) struct LinkFlowMonitor {
    marks: WaterMarks,
    state: FlowStateHandle,
}

impl LinkFlowMonitor {
    pub(crate) fn new(marks: WaterMarks, state: FlowStateHandle) -> Self {
        Self { marks, state }
    }

    /// Fold one occupancy sample into the signal: Ready → Busy at the high
    /// mark, Busy → Ready at the low mark, hysteresis in between.
    pub(crate) fn update(&self, occupancy: usize) {
        match self.state.signal() {
            FlowSignal::Ready if self.marks.is_high(occupancy) => {
                self.state.set_signal(FlowSignal::Busy);
            }
            FlowSignal::Busy if self.marks.is_low(occupancy) => {
                self.state.set_signal(FlowSignal::Ready);
            }
            _ => {}
        }
    }
}

/// Water mark configuration for queue-based flow control.
#[derive(Debug, Clone, Copy)]
pub struct WaterMarks {
    /// High water mark - trigger backpressure when reached.
    pub high: usize,
    /// Low water mark - release backpressure when reached.
    pub low: usize,
}

impl WaterMarks {
    /// Create water marks with explicit high and low values.
    pub fn new(high: usize, low: usize) -> Self {
        Self { high, low }
    }

    /// Create water marks from a capacity.
    ///
    /// High = 80% of capacity, Low = 20% of capacity.
    pub fn from_capacity(capacity: usize) -> Self {
        Self {
            high: (capacity * 80) / 100,
            low: (capacity * 20) / 100,
        }
    }

    /// Create water marks with custom percentages.
    pub fn with_percentages(capacity: usize, high_percent: usize, low_percent: usize) -> Self {
        Self {
            high: (capacity * high_percent) / 100,
            low: (capacity * low_percent) / 100,
        }
    }

    /// Check if level is at or above high water mark.
    #[inline]
    pub fn is_high(&self, level: usize) -> bool {
        level >= self.high
    }

    /// Check if level is at or below low water mark.
    #[inline]
    pub fn is_low(&self, level: usize) -> bool {
        level <= self.low
    }
}

impl Default for WaterMarks {
    fn default() -> Self {
        Self::from_capacity(32) // Sensible default
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flow_signal_conversions() {
        assert_eq!(FlowSignal::from(0u8), FlowSignal::Ready);
        assert_eq!(FlowSignal::from(1u8), FlowSignal::Busy);
        assert_eq!(FlowSignal::from(255u8), FlowSignal::Ready); // Invalid -> default
    }

    #[test]
    fn test_flow_signal_properties() {
        assert!(FlowSignal::Ready.should_produce());
        assert!(!FlowSignal::Busy.should_produce());

        assert!(!FlowSignal::Ready.is_backpressure());
        assert!(FlowSignal::Busy.is_backpressure());
    }

    #[test]
    fn test_shared_flow_state() {
        let state = SharedFlowState::new();

        assert_eq!(state.signal(), FlowSignal::Ready);
        assert!(state.should_produce());

        state.set_signal(FlowSignal::Busy);
        assert_eq!(state.signal(), FlowSignal::Busy);
        assert!(!state.should_produce());
        assert_eq!(state.backpressure_events(), 1);

        // Setting Busy again shouldn't increment
        state.set_signal(FlowSignal::Busy);
        assert_eq!(state.backpressure_events(), 1);

        state.set_signal(FlowSignal::Ready);
        state.set_signal(FlowSignal::Busy);
        assert_eq!(state.backpressure_events(), 2);

        state.record_drop();
        state.record_drop();
        assert_eq!(state.frames_dropped(), 2);
    }

    #[test]
    fn test_water_marks() {
        let wm = WaterMarks::from_capacity(100);
        assert_eq!(wm.high, 80);
        assert_eq!(wm.low, 20);

        assert!(!wm.is_high(79));
        assert!(wm.is_high(80));
        assert!(wm.is_high(100));

        assert!(wm.is_low(20));
        assert!(wm.is_low(0));
        assert!(!wm.is_low(21));

        let custom = WaterMarks::with_percentages(100, 90, 10);
        assert_eq!(custom.high, 90);
        assert_eq!(custom.low, 10);
    }

    #[test]
    fn link_flow_monitor_hysteresis() {
        let state = new_flow_state();
        let monitor = LinkFlowMonitor::new(WaterMarks::new(8, 2), state.clone());

        // Climbing through the band keeps Ready until the high mark.
        monitor.update(5);
        assert_eq!(state.signal(), FlowSignal::Ready);
        monitor.update(8);
        assert_eq!(state.signal(), FlowSignal::Busy);

        // Draining through the band keeps Busy until the low mark.
        monitor.update(5);
        assert_eq!(state.signal(), FlowSignal::Busy);
        monitor.update(2);
        assert_eq!(state.signal(), FlowSignal::Ready);

        // One backpressure event counted for the single Ready→Busy edge.
        assert_eq!(state.backpressure_events(), 1);
    }

    #[test]
    fn test_flow_state_handle() {
        let handle1 = new_flow_state();
        let handle2 = Arc::clone(&handle1);

        handle1.set_signal(FlowSignal::Busy);
        assert_eq!(handle2.signal(), FlowSignal::Busy);

        handle2.set_signal(FlowSignal::Ready);
        assert_eq!(handle1.signal(), FlowSignal::Ready);
    }
}
