//! Flow control for pipeline backpressure management.
//!
//! This module provides types for managing data flow between elements in a pipeline.
//! When downstream elements (consumers) are slower than upstream elements (producers),
//! backpressure signals propagate upstream to prevent memory exhaustion.
//!
//! # Design
//!
//! The flow control system uses a hybrid pull/push model:
//!
//! ```text
//! Producer ──data──> Queue ──data──> Consumer
//!     ^                                  │
//!     └────── FlowSignal::Busy ──────────┘
//! ```
//!
//! When the downstream queue fills to its high-water mark:
//! 1. Queue sends `FlowSignal::Busy` upstream
//! 2. Source responds according to its `FlowPolicy`:
//!    - `Block`: Wait until downstream is ready
//!    - `Drop`: Drop frames and continue
//!    - `RingBuffer`: Buffer frames, drop oldest when full
//! 3. When queue drains to low-water mark, sends `FlowSignal::Ready`
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::flow::{FlowSignal, FlowPolicy};
//!
//! // Source with drop policy for live capture
//! impl Source for CameraSrc {
//!     fn flow_policy(&self) -> FlowPolicy {
//!         FlowPolicy::Drop {
//!             log_drops: true,
//!             max_consecutive: Some(30), // Error after 1s at 30fps
//!         }
//!     }
//!
//!     fn handle_flow_signal(&mut self, signal: FlowSignal) {
//!         self.flow_state = signal;
//!     }
//! }
//! ```

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

    /// Downstream is busy - reduce production rate or buffer.
    ///
    /// Sources should respond according to their `FlowPolicy`.
    Busy = 1,

    /// Downstream requests frame dropping until further notice.
    ///
    /// Used when downstream is severely overloaded and needs to
    /// catch up. Sources should skip producing data.
    Drop = 2,

    /// End of stream has been acknowledged by downstream.
    ///
    /// The source can clean up resources.
    EosAck = 3,

    /// Pipeline is pausing - stop producing but retain state.
    Pausing = 4,

    /// Pipeline is stopping - stop producing and clean up.
    Stopping = 5,
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
        matches!(self, FlowSignal::Busy | FlowSignal::Drop)
    }

    /// Check if the signal indicates the pipeline is ending.
    #[inline]
    pub fn is_ending(&self) -> bool {
        matches!(
            self,
            FlowSignal::EosAck | FlowSignal::Pausing | FlowSignal::Stopping
        )
    }
}

impl From<u8> for FlowSignal {
    fn from(value: u8) -> Self {
        match value {
            0 => FlowSignal::Ready,
            1 => FlowSignal::Busy,
            2 => FlowSignal::Drop,
            3 => FlowSignal::EosAck,
            4 => FlowSignal::Pausing,
            5 => FlowSignal::Stopping,
            _ => FlowSignal::Ready, // Safe default
        }
    }
}

impl From<FlowSignal> for u8 {
    fn from(signal: FlowSignal) -> Self {
        signal as u8
    }
}

/// Flow control policy for sources.
///
/// Determines how a source responds to backpressure signals.
#[derive(Debug, Clone, PartialEq)]
pub enum FlowPolicy {
    /// Block production when downstream is busy.
    ///
    /// This is the safest policy - no data is lost, but the source
    /// may fall behind real-time. Best for file sources.
    Block,

    /// Drop frames when downstream is busy.
    ///
    /// Best for live sources (cameras, screen capture) where
    /// real-time is more important than completeness.
    Drop {
        /// Whether to log when frames are dropped.
        log_drops: bool,
        /// Maximum consecutive drops before returning an error.
        /// `None` means unlimited drops are allowed.
        max_consecutive: Option<u32>,
    },

    /// Buffer frames in a ring buffer, dropping oldest when full.
    ///
    /// Provides a middle ground - some buffering before drops.
    RingBuffer {
        /// Maximum number of frames to buffer.
        capacity: usize,
    },

    /// Adaptive policy that adjusts based on conditions.
    ///
    /// Starts with blocking, switches to dropping if backpressure
    /// persists beyond the threshold.
    Adaptive {
        /// How long to block before switching to drop mode.
        block_timeout_ms: u64,
        /// Whether to log mode switches.
        log_switches: bool,
    },
}

impl Default for FlowPolicy {
    fn default() -> Self {
        FlowPolicy::Block
    }
}

impl FlowPolicy {
    /// Create a drop policy with default settings.
    pub fn drop_with_logging() -> Self {
        FlowPolicy::Drop {
            log_drops: true,
            max_consecutive: None,
        }
    }

    /// Create a drop policy with a maximum consecutive drop limit.
    pub fn drop_with_limit(max_consecutive: u32) -> Self {
        FlowPolicy::Drop {
            log_drops: true,
            max_consecutive: Some(max_consecutive),
        }
    }

    /// Create a ring buffer policy.
    pub fn ring_buffer(capacity: usize) -> Self {
        FlowPolicy::RingBuffer { capacity }
    }

    /// Check if this policy allows dropping frames.
    pub fn allows_dropping(&self) -> bool {
        matches!(
            self,
            FlowPolicy::Drop { .. } | FlowPolicy::RingBuffer { .. } | FlowPolicy::Adaptive { .. }
        )
    }
}

/// Statistics about flow control behavior.
#[derive(Debug, Clone, Default)]
pub struct FlowStats {
    /// Total frames produced.
    pub frames_produced: u64,
    /// Total frames dropped due to backpressure.
    pub frames_dropped: u64,
    /// Current consecutive drop count.
    pub consecutive_drops: u32,
    /// Maximum consecutive drops seen.
    pub max_consecutive_drops: u32,
    /// Total time spent in busy state (microseconds).
    pub busy_time_us: u64,
    /// Number of times backpressure was triggered.
    pub backpressure_events: u64,
}

impl FlowStats {
    /// Record a produced frame.
    pub fn record_produced(&mut self) {
        self.frames_produced += 1;
        self.consecutive_drops = 0;
    }

    /// Record a dropped frame.
    pub fn record_dropped(&mut self) {
        self.frames_dropped += 1;
        self.consecutive_drops += 1;
        if self.consecutive_drops > self.max_consecutive_drops {
            self.max_consecutive_drops = self.consecutive_drops;
        }
    }

    /// Record backpressure event.
    pub fn record_backpressure(&mut self) {
        self.backpressure_events += 1;
    }

    /// Get the drop rate as a percentage.
    pub fn drop_rate(&self) -> f64 {
        let total = self.frames_produced + self.frames_dropped;
        if total == 0 {
            0.0
        } else {
            (self.frames_dropped as f64 / total as f64) * 100.0
        }
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
        assert_eq!(FlowSignal::from(2u8), FlowSignal::Drop);
        assert_eq!(FlowSignal::from(255u8), FlowSignal::Ready); // Invalid -> default
    }

    #[test]
    fn test_flow_signal_properties() {
        assert!(FlowSignal::Ready.should_produce());
        assert!(!FlowSignal::Busy.should_produce());
        assert!(!FlowSignal::Drop.should_produce());

        assert!(!FlowSignal::Ready.is_backpressure());
        assert!(FlowSignal::Busy.is_backpressure());
        assert!(FlowSignal::Drop.is_backpressure());

        assert!(!FlowSignal::Ready.is_ending());
        assert!(FlowSignal::EosAck.is_ending());
        assert!(FlowSignal::Stopping.is_ending());
    }

    #[test]
    fn test_flow_policy_defaults() {
        let policy = FlowPolicy::default();
        assert_eq!(policy, FlowPolicy::Block);
        assert!(!policy.allows_dropping());

        let drop_policy = FlowPolicy::drop_with_logging();
        assert!(drop_policy.allows_dropping());
    }

    #[test]
    fn test_flow_stats() {
        let mut stats = FlowStats::default();

        stats.record_produced();
        stats.record_produced();
        stats.record_dropped();
        stats.record_dropped();
        stats.record_dropped();

        assert_eq!(stats.frames_produced, 2);
        assert_eq!(stats.frames_dropped, 3);
        assert_eq!(stats.consecutive_drops, 3);
        assert_eq!(stats.max_consecutive_drops, 3);
        assert!((stats.drop_rate() - 60.0).abs() < 0.01);

        // Producing resets consecutive drops
        stats.record_produced();
        assert_eq!(stats.consecutive_drops, 0);
        assert_eq!(stats.max_consecutive_drops, 3); // Max unchanged
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
    fn test_flow_state_handle() {
        let handle1 = new_flow_state();
        let handle2 = Arc::clone(&handle1);

        handle1.set_signal(FlowSignal::Busy);
        assert_eq!(handle2.signal(), FlowSignal::Busy);

        handle2.set_signal(FlowSignal::Ready);
        assert_eq!(handle1.signal(), FlowSignal::Ready);
    }
}
