//! Timeout and debounce elements for timing control.
//!
//! Elements that interact with timing constraints.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Element;
use crate::error::Result;
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Slot size for fallback buffers when nothing larger is asked for.
///
/// Matches the 4 KiB slots the old process-wide arena used, so a `Timeout` with
/// a small fallback still builds its arena exactly once.
const FALLBACK_SLOT_SIZE: usize = 4096;

/// A timeout element that produces a fallback buffer if no input arrives in time.
///
/// This is useful for heartbeat/keepalive scenarios where you need to ensure
/// some output even when input stops flowing.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Timeout;
/// use std::time::Duration;
///
/// // Produce empty fallback if no input for 1 second
/// let timeout = Timeout::new(Duration::from_secs(1));
///
/// // Or with custom fallback data
/// let timeout = Timeout::new(Duration::from_secs(1))
///     .with_fallback(b"timeout".to_vec());
/// ```
pub struct Timeout {
    name: String,
    timeout: Duration,
    last_buffer: Option<Instant>,
    fallback_data: Vec<u8>,
    timeouts_triggered: AtomicU64,
    buffers_passed: AtomicU64,
    /// Per-instance output arena for fallback buffers.
    ///
    /// This used to be a process-wide `static`, so every `Timeout` in the
    /// process drew from the same 32 slots and no budget could mean anything
    /// (#95).
    output: OutputArena,
}

impl Timeout {
    /// Create a new timeout element.
    pub fn new(timeout: Duration) -> Self {
        Self {
            name: "timeout".to_string(),
            timeout,
            last_buffer: None,
            fallback_data: Vec::new(),
            timeouts_triggered: AtomicU64::new(0),
            buffers_passed: AtomicU64::new(0),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(FALLBACK_SLOT_SIZE)
                .grow_to_fit(),
        }
    }

    /// Create from milliseconds.
    pub fn from_millis(millis: u64) -> Self {
        Self::new(Duration::from_millis(millis))
    }

    /// Set custom fallback data.
    pub fn with_fallback(mut self, data: Vec<u8>) -> Self {
        self.fallback_data = data;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Check if timeout has occurred and return fallback if so.
    pub fn check_timeout(&mut self) -> Result<Option<Buffer>> {
        if let Some(last) = self.last_buffer
            && last.elapsed() >= self.timeout
        {
            self.timeouts_triggered.fetch_add(1, Ordering::Relaxed);
            self.last_buffer = Some(Instant::now());
            return self.create_fallback();
        }
        Ok(None)
    }

    fn create_fallback(&mut self) -> Result<Option<Buffer>> {
        let len = self.fallback_data.len();
        let mut slot = self.output.acquire(len, "timeout")?;

        if !self.fallback_data.is_empty() {
            slot.data_mut()[..len].copy_from_slice(&self.fallback_data);
        }

        let handle = MemoryHandle::with_len(slot, len);
        let mut metadata = Metadata::new();
        metadata.flags = metadata.flags.insert(crate::metadata::BufferFlags::TIMEOUT);

        Ok(Some(Buffer::new(handle, metadata)))
    }

    /// Get statistics.
    pub fn stats(&self) -> TimeoutStats {
        TimeoutStats {
            buffers_passed: self.buffers_passed.load(Ordering::Relaxed),
            timeouts_triggered: self.timeouts_triggered.load(Ordering::Relaxed),
        }
    }
}

impl Element for Timeout {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.last_buffer = Some(Instant::now());
        self.buffers_passed.fetch_add(1, Ordering::Relaxed);
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Timeout element.
#[derive(Debug, Clone, Copy)]
pub struct TimeoutStats {
    /// Buffers that passed through normally.
    pub buffers_passed: u64,
    /// Number of timeout events triggered.
    pub timeouts_triggered: u64,
}

/// A debounce element that suppresses rapid buffer sequences.
///
/// Only passes buffers that arrive after a quiet period. Useful for
/// rate limiting or suppressing bursts.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Debounce;
/// use std::time::Duration;
///
/// // Only pass buffers after 100ms of quiet
/// let debounce = Debounce::new(Duration::from_millis(100));
/// ```
pub struct Debounce {
    name: String,
    quiet_period: Duration,
    last_buffer_time: Option<Instant>,
    last_buffer: Option<Buffer>,
    passed: AtomicU64,
    suppressed: AtomicU64,
}

impl Debounce {
    /// Create a new debounce element.
    pub fn new(quiet_period: Duration) -> Self {
        Self {
            name: "debounce".to_string(),
            quiet_period,
            last_buffer_time: None,
            last_buffer: None,
            passed: AtomicU64::new(0),
            suppressed: AtomicU64::new(0),
        }
    }

    /// Create from milliseconds.
    pub fn from_millis(millis: u64) -> Self {
        Self::new(Duration::from_millis(millis))
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Check if quiet period has passed and flush held buffer.
    pub fn check_quiet(&mut self) -> Option<Buffer> {
        if let Some(time) = self.last_buffer_time
            && time.elapsed() >= self.quiet_period
        {
            self.passed.fetch_add(1, Ordering::Relaxed);
            self.last_buffer_time = None;
            return self.last_buffer.take();
        }
        None
    }

    /// Flush any held buffer immediately.
    pub fn flush(&mut self) -> Option<Buffer> {
        self.last_buffer_time = None;
        let buf = self.last_buffer.take();
        if buf.is_some() {
            self.passed.fetch_add(1, Ordering::Relaxed);
        }
        buf
    }

    /// Get statistics.
    pub fn stats(&self) -> DebounceStats {
        DebounceStats {
            passed: self.passed.load(Ordering::Relaxed),
            suppressed: self.suppressed.load(Ordering::Relaxed),
        }
    }
}

impl Element for Debounce {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // If we have a held buffer, it gets suppressed
        if self.last_buffer.is_some() {
            self.suppressed.fetch_add(1, Ordering::Relaxed);
        }

        self.last_buffer = Some(buffer);
        self.last_buffer_time = Some(Instant::now());

        // Never return immediately - caller should use check_quiet()
        Ok(None)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Debounce element.
#[derive(Debug, Clone, Copy)]
pub struct DebounceStats {
    /// Buffers that were passed after quiet period.
    pub passed: u64,
    /// Buffers that were suppressed (replaced before quiet period).
    pub suppressed: u64,
}

/// Cloneable handle to change a running [`Throttle`]'s rate.
///
/// Framerate is the third bandwidth lever (after bitrate and resolution), and
/// the one to combine with a bitrate cut: fewer frames sharing the same budget
/// keeps each of them watchable.
///
/// Like the other runtime control handles, clone it from the element **before**
/// `executor.start()` — elements are moved into their executor tasks at start.
///
/// # Example
///
/// ```rust,ignore
/// let throttle = Throttle::rate(30.0);
/// let rate = throttle.control();      // BEFORE start
/// pipeline.add_filter("rate", throttle);
/// let handle = executor.start(&mut pipeline)?;
///
/// rate.set_rate(10.0);   // drop to 10 fps
/// ```
#[derive(Clone, Debug)]
pub struct ThrottleControl(Arc<AtomicU64>);

/// A rate of zero means "pass nothing": an interval no clock will ever reach.
const DROP_EVERYTHING_NS: u64 = u64::MAX;

impl ThrottleControl {
    /// Limit to `buffers_per_second`.
    ///
    /// A rate of zero (or negative) drops every buffer — the throttle becomes a
    /// closed valve rather than dividing by zero.
    pub fn set_rate(&self, buffers_per_second: f64) {
        let nanos = if buffers_per_second > 0.0 {
            (1e9 / buffers_per_second).round() as u64
        } else {
            DROP_EVERYTHING_NS
        };
        self.0.store(nanos, Ordering::Release);
    }

    /// Set the minimum interval between passed buffers directly.
    pub fn set_min_interval(&self, interval: Duration) {
        self.0.store(
            interval.as_nanos().min(u64::MAX as u128) as u64,
            Ordering::Release,
        );
    }

    /// The current minimum interval between passed buffers.
    pub fn min_interval(&self) -> Duration {
        Duration::from_nanos(self.0.load(Ordering::Acquire))
    }

    /// The current rate in buffers per second (0.0 when dropping everything).
    pub fn rate(&self) -> f64 {
        match self.0.load(Ordering::Acquire) {
            0 => f64::INFINITY,
            DROP_EVERYTHING_NS => 0.0,
            nanos => 1e9 / nanos as f64,
        }
    }
}

/// A throttle element that limits the rate of buffer flow.
///
/// Drops buffers if they arrive too quickly. Different from RateLimiter
/// which delays rather than drops — dropping is what you want ahead of a live
/// source, where delaying would back-pressure the camera.
///
/// The rate can be changed on a running pipeline through
/// [`control`](Self::control).
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Throttle;
/// use std::time::Duration;
///
/// // Allow at most 1 buffer per 100ms
/// let throttle = Throttle::new(Duration::from_millis(100));
/// ```
pub struct Throttle {
    name: String,
    /// Minimum interval between passed buffers, in nanoseconds. Shared with
    /// [`ThrottleControl`] handles, so it can change while the pipeline runs.
    min_interval_ns: Arc<AtomicU64>,
    last_passed: Option<Instant>,
    passed: AtomicU64,
    dropped: AtomicU64,
}

impl Throttle {
    /// Create a new throttle element.
    pub fn new(min_interval: Duration) -> Self {
        let throttle = Self {
            name: "throttle".to_string(),
            min_interval_ns: Arc::new(AtomicU64::new(0)),
            last_passed: None,
            passed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        };
        throttle.control().set_min_interval(min_interval);
        throttle
    }

    /// Create from milliseconds.
    pub fn from_millis(millis: u64) -> Self {
        Self::new(Duration::from_millis(millis))
    }

    /// Create limiting to a specific rate (buffers per second).
    ///
    /// A rate of zero drops everything (see [`ThrottleControl::set_rate`]).
    pub fn rate(buffers_per_second: f64) -> Self {
        let throttle = Self::new(Duration::ZERO);
        throttle.control().set_rate(buffers_per_second);
        throttle
    }

    /// Get a cloneable handle for changing the rate at runtime.
    ///
    /// Clone it *before* the pipeline starts — see [`ThrottleControl`].
    pub fn control(&self) -> ThrottleControl {
        ThrottleControl(Arc::clone(&self.min_interval_ns))
    }

    /// The current minimum interval between passed buffers.
    pub fn min_interval(&self) -> Duration {
        Duration::from_nanos(self.min_interval_ns.load(Ordering::Relaxed))
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> ThrottleStats {
        ThrottleStats {
            passed: self.passed.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}

impl crate::control::Controllable for Throttle {
    type Control = ThrottleControl;

    fn control(&self) -> ThrottleControl {
        ThrottleControl(Arc::clone(&self.min_interval_ns))
    }
}

impl Element for Throttle {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let now = Instant::now();
        // One relaxed load per buffer; the rate may have been changed by a
        // ThrottleControl handle since the last one.
        let min_interval_ns = self.min_interval_ns.load(Ordering::Relaxed);

        let should_pass = if min_interval_ns == DROP_EVERYTHING_NS {
            false
        } else {
            match self.last_passed {
                None => true,
                Some(last) => now.duration_since(last).as_nanos() >= min_interval_ns as u128,
            }
        };

        if should_pass {
            self.last_passed = Some(now);
            self.passed.fetch_add(1, Ordering::Relaxed);
            Ok(Some(buffer))
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Throttle element.
#[derive(Debug, Clone, Copy)]
pub struct ThrottleStats {
    /// Buffers that passed through.
    pub passed: u64,
    /// Buffers that were dropped due to rate limiting.
    pub dropped: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SharedArena;
    use std::sync::OnceLock;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(64, 64).unwrap())
    }

    fn create_test_buffer(seq: u64) -> Buffer {
        let arena = test_arena();
        // Reclaim first: the arena is shared across the module's tests and has
        // 64 slots, so a test that runs more buffers than that through a
        // dropping element would otherwise exhaust it.
        arena.reclaim();
        let slot = arena.acquire().expect("test arena slot");
        let handle = MemoryHandle::new(slot);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    // Timeout tests

    #[test]
    fn test_timeout_passthrough() {
        let mut timeout = Timeout::from_millis(100);

        let buffer = create_test_buffer(42);
        let result = timeout.process(buffer).unwrap();

        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    #[test]
    fn test_timeout_triggers() {
        let mut timeout = Timeout::from_millis(50);

        // Process a buffer
        timeout.process(create_test_buffer(0)).unwrap();

        // Wait for timeout
        std::thread::sleep(Duration::from_millis(60));

        // Check should trigger
        let fallback = timeout.check_timeout().unwrap();
        assert!(fallback.is_some());
        assert!(fallback.unwrap().metadata().flags.is_timeout());
    }

    #[test]
    fn test_timeout_no_trigger_when_active() {
        let mut timeout = Timeout::from_millis(100);

        timeout.process(create_test_buffer(0)).unwrap();

        // Check immediately - should not trigger
        let result = timeout.check_timeout().unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_timeout_with_fallback_data() {
        let mut timeout = Timeout::from_millis(10).with_fallback(b"fallback".to_vec());

        timeout.process(create_test_buffer(0)).unwrap();
        std::thread::sleep(Duration::from_millis(20));

        let fallback = timeout.check_timeout().unwrap().unwrap();
        assert_eq!(fallback.as_bytes(), b"fallback");
    }

    #[test]
    fn each_timeout_owns_its_arena() {
        // The arena used to be a process-wide static, so two Timeouts drew
        // from the same 32 slots (#95). They must not any more — and neither
        // should exist before the first fallback is built.
        let mut a = Timeout::from_millis(1).with_fallback(b"a".to_vec());
        let mut b = Timeout::from_millis(1).with_fallback(b"b".to_vec());
        assert!(!a.output.is_built());
        assert!(!b.output.is_built());

        a.process(create_test_buffer(0)).unwrap();
        b.process(create_test_buffer(0)).unwrap();
        std::thread::sleep(Duration::from_millis(5));

        let from_a = a.check_timeout().unwrap().unwrap();
        let from_b = b.check_timeout().unwrap().unwrap();

        assert_eq!(from_a.as_bytes(), b"a");
        assert_eq!(from_b.as_bytes(), b"b");
        // Distinct arenas: two separate mappings, not one shared static.
        assert!(!std::ptr::eq(
            a.output.arena().unwrap(),
            b.output.arena().unwrap()
        ));
    }

    #[test]
    fn a_budget_sizes_the_fallback_arena() {
        let mut timeout = Timeout::from_millis(1).with_fallback(vec![0u8; 8]);
        timeout.set_output_budget(OutputBudget::new(64, 4));

        timeout.process(create_test_buffer(0)).unwrap();
        std::thread::sleep(Duration::from_millis(5));
        timeout.check_timeout().unwrap().unwrap();

        assert_eq!(timeout.output.arena().unwrap().slot_count(), 68);
    }

    // Debounce tests

    #[test]
    fn test_debounce_holds_buffer() {
        let mut debounce = Debounce::from_millis(50);

        // Process should never return immediately
        let result = debounce.process(create_test_buffer(0)).unwrap();
        assert!(result.is_none());

        // Check immediately - not enough quiet time
        assert!(debounce.check_quiet().is_none());
    }

    #[test]
    fn test_debounce_releases_after_quiet() {
        let mut debounce = Debounce::from_millis(30);

        debounce.process(create_test_buffer(42)).unwrap();

        // Wait for quiet period
        std::thread::sleep(Duration::from_millis(40));

        let result = debounce.check_quiet();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    #[test]
    fn test_debounce_suppresses() {
        let mut debounce = Debounce::from_millis(100);

        debounce.process(create_test_buffer(0)).unwrap();
        debounce.process(create_test_buffer(1)).unwrap();
        debounce.process(create_test_buffer(2)).unwrap();

        let stats = debounce.stats();
        assert_eq!(stats.suppressed, 2); // First two were suppressed
    }

    #[test]
    fn test_debounce_flush() {
        let mut debounce = Debounce::from_millis(1000);

        debounce.process(create_test_buffer(42)).unwrap();

        // Flush immediately without waiting
        let result = debounce.flush();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    // Throttle tests

    #[test]
    fn test_throttle_first_passes() {
        let mut throttle = Throttle::from_millis(100);

        let result = throttle.process(create_test_buffer(0)).unwrap();
        assert!(result.is_some());
    }

    #[test]
    fn test_throttle_drops_rapid() {
        let mut throttle = Throttle::from_millis(100);

        // First passes
        let r1 = throttle.process(create_test_buffer(0)).unwrap();
        assert!(r1.is_some());

        // Immediate second should be dropped
        let r2 = throttle.process(create_test_buffer(1)).unwrap();
        assert!(r2.is_none());

        let stats = throttle.stats();
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.dropped, 1);
    }

    #[test]
    fn test_throttle_passes_after_interval() {
        let mut throttle = Throttle::from_millis(30);

        throttle.process(create_test_buffer(0)).unwrap();

        std::thread::sleep(Duration::from_millis(40));

        let result = throttle.process(create_test_buffer(1)).unwrap();
        assert!(result.is_some());

        assert_eq!(throttle.stats().passed, 2);
    }

    #[test]
    fn test_throttle_rate() {
        let throttle = Throttle::rate(10.0); // 10 per second = 100ms interval
        assert!(throttle.min_interval() >= Duration::from_millis(99));
        assert!(throttle.min_interval() <= Duration::from_millis(101));
    }

    // ========================================================================
    // Runtime rate control (#30)
    // ========================================================================

    #[test]
    fn rate_change_takes_effect_immediately() {
        let mut throttle = Throttle::from_millis(100);
        let control = throttle.control();

        // First passes, second is too soon at 100ms.
        assert!(throttle.process(create_test_buffer(0)).unwrap().is_some());
        assert!(throttle.process(create_test_buffer(1)).unwrap().is_none());

        // Raising the rate applies to the very next buffer.
        control.set_min_interval(Duration::ZERO);
        assert!(
            throttle.process(create_test_buffer(2)).unwrap().is_some(),
            "a raised rate must apply to the next buffer, not the next interval"
        );
    }

    #[test]
    fn lowering_the_rate_does_not_stall() {
        let mut throttle = Throttle::from_millis(1);
        let control = throttle.control();

        throttle.process(create_test_buffer(0)).unwrap();
        control.set_rate(1.0); // one per second

        // The next buffer is dropped (too soon)...
        assert!(throttle.process(create_test_buffer(1)).unwrap().is_none());
        // ...but the throttle recovers once the new interval elapses, rather
        // than wedging itself shut.
        control.set_min_interval(Duration::from_millis(5));
        std::thread::sleep(Duration::from_millis(10));
        assert!(throttle.process(create_test_buffer(2)).unwrap().is_some());
    }

    #[test]
    fn zero_rate_drops_everything() {
        // Previously this panicked: 1.0/0.0 is infinite and
        // Duration::from_secs_f64(inf) is not representable.
        let mut throttle = Throttle::rate(0.0);
        let control = throttle.control();
        assert_eq!(control.rate(), 0.0);

        for i in 0..5 {
            assert!(
                throttle.process(create_test_buffer(i)).unwrap().is_none(),
                "a rate of zero passes nothing"
            );
        }
        assert_eq!(throttle.stats().passed, 0);
        assert_eq!(throttle.stats().dropped, 5);

        // ...and it reopens.
        control.set_rate(1000.0);
        assert!(throttle.process(create_test_buffer(9)).unwrap().is_some());
    }

    #[test]
    fn control_reports_the_current_rate() {
        let throttle = Throttle::rate(25.0);
        let control = throttle.control();
        assert!((control.rate() - 25.0).abs() < 0.01);

        control.set_rate(5.0);
        assert!((control.rate() - 5.0).abs() < 0.01);
        assert_eq!(control.min_interval(), Duration::from_millis(200));
    }

    #[test]
    fn halving_the_rate_halves_the_output() {
        // The bandwidth claim: half the framerate, half the frames on the wire.
        let count_passed = |fps: f64| -> u64 {
            let mut throttle = Throttle::rate(fps);
            // Feed for ~200ms at a much higher rate than the throttle allows.
            let deadline = Instant::now() + Duration::from_millis(200);
            let mut seq = 0;
            while Instant::now() < deadline {
                throttle.process(create_test_buffer(seq)).unwrap();
                seq += 1;
                std::thread::sleep(Duration::from_millis(2));
            }
            throttle.stats().passed
        };

        let fast = count_passed(50.0); // ~10 in 200ms
        let slow = count_passed(25.0); // ~5 in 200ms

        assert!(
            slow < fast,
            "halving the rate must halve the frames passed (got {slow} vs {fast})"
        );
    }
}
