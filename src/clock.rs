//! Clock and time types for pipeline synchronization.
//!
//! This module provides:
//! - [`ClockTime`]: A nanosecond timestamp type (8 bytes, Copy)
//! - [`Clock`]: Trait for time sources
//! - [`SystemClock`]: Monotonic system clock
//! - [`PipelineClock`]: Pipeline timing context with base time

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// ============================================================================
// ClockTime
// ============================================================================

/// Time in nanoseconds (8 bytes, Copy).
///
/// This is the fundamental time type in Parallax. It represents time as
/// nanoseconds since an arbitrary epoch (usually pipeline start).
///
/// # Special Values
///
/// - `ClockTime::ZERO`: Zero time
/// - `ClockTime::NONE`: Invalid/unset time (sentinel value)
/// - `ClockTime::MAX`: Maximum representable time
///
/// # Examples
///
/// ```rust
/// use parallax::clock::ClockTime;
///
/// let t1 = ClockTime::from_secs(1);
/// let t2 = ClockTime::from_millis(500);
/// let t3 = t1 + t2;
///
/// assert_eq!(t3.millis(), 1500);
/// assert_eq!(format!("{}", t3), "1.500s");
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ClockTime(u64);

impl ClockTime {
    /// Zero time.
    pub const ZERO: Self = Self(0);

    /// Maximum representable time (one less than NONE sentinel).
    pub const MAX: Self = Self(u64::MAX - 1);

    /// Invalid/unset time (sentinel value).
    pub const NONE: Self = Self(u64::MAX);

    /// Create from nanoseconds.
    #[inline]
    pub const fn from_nanos(ns: u64) -> Self {
        Self(ns)
    }

    /// Create from microseconds.
    #[inline]
    pub const fn from_micros(us: u64) -> Self {
        Self(us.saturating_mul(1_000))
    }

    /// Create from milliseconds.
    #[inline]
    pub const fn from_millis(ms: u64) -> Self {
        Self(ms.saturating_mul(1_000_000))
    }

    /// Create from seconds.
    #[inline]
    pub const fn from_secs(s: u64) -> Self {
        Self(s.saturating_mul(1_000_000_000))
    }

    /// Create from seconds and nanoseconds.
    #[inline]
    pub const fn from_secs_nanos(secs: u64, nanos: u32) -> Self {
        Self(
            secs.saturating_mul(1_000_000_000)
                .saturating_add(nanos as u64),
        )
    }

    /// Get as nanoseconds.
    #[inline]
    pub const fn nanos(self) -> u64 {
        self.0
    }

    /// Get as microseconds (truncated).
    #[inline]
    pub const fn micros(self) -> u64 {
        self.0 / 1_000
    }

    /// Get as milliseconds (truncated).
    #[inline]
    pub const fn millis(self) -> u64 {
        self.0 / 1_000_000
    }

    /// Get as seconds (truncated).
    #[inline]
    pub const fn secs(self) -> u64 {
        self.0 / 1_000_000_000
    }

    /// Get the nanoseconds part (0..999_999_999).
    #[inline]
    pub const fn subsec_nanos(self) -> u32 {
        (self.0 % 1_000_000_000) as u32
    }

    /// Check if this is the NONE sentinel value.
    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u64::MAX
    }

    /// Check if this is a valid time (not NONE).
    #[inline]
    pub const fn is_some(self) -> bool {
        self.0 != u64::MAX
    }

    /// Convert to Option, returning None for the NONE sentinel.
    #[inline]
    pub const fn to_option(self) -> Option<Self> {
        if self.is_none() { None } else { Some(self) }
    }

    /// Saturating addition. Returns NONE if either operand is NONE.
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        if self.is_none() || rhs.is_none() {
            return Self::NONE;
        }
        let result = self.0.saturating_add(rhs.0);
        // Don't overflow into NONE
        if result == u64::MAX {
            Self::MAX
        } else {
            Self(result)
        }
    }

    /// Saturating subtraction. Returns NONE if either operand is NONE.
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        if self.is_none() || rhs.is_none() {
            return Self::NONE;
        }
        Self(self.0.saturating_sub(rhs.0))
    }

    /// Checked subtraction. Returns None if either operand is NONE or underflow.
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        if self.is_none() || rhs.is_none() {
            return None;
        }
        match self.0.checked_sub(rhs.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }

    /// Checked addition. Returns None if either operand is NONE or overflow.
    #[inline]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        if self.is_none() || rhs.is_none() {
            return None;
        }
        match self.0.checked_add(rhs.0) {
            Some(v) if v != u64::MAX => Some(Self(v)),
            _ => None,
        }
    }

    /// Calculate absolute difference between two times.
    #[inline]
    pub const fn abs_diff(self, other: Self) -> Self {
        if self.is_none() || other.is_none() {
            return Self::NONE;
        }
        Self(self.0.abs_diff(other.0))
    }

    /// Multiply by a scalar.
    #[inline]
    pub const fn saturating_mul(self, rhs: u64) -> Self {
        if self.is_none() {
            return Self::NONE;
        }
        let result = self.0.saturating_mul(rhs);
        if result == u64::MAX {
            Self::MAX
        } else {
            Self(result)
        }
    }

    /// Divide by a scalar.
    #[inline]
    pub const fn checked_div(self, rhs: u64) -> Option<Self> {
        if self.is_none() || rhs == 0 {
            return None;
        }
        Some(Self(self.0 / rhs))
    }
}

impl std::ops::Add for ClockTime {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self {
        self.saturating_add(rhs)
    }
}

impl std::ops::AddAssign for ClockTime {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = self.saturating_add(rhs);
    }
}

impl std::ops::Sub for ClockTime {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self {
        self.saturating_sub(rhs)
    }
}

impl std::ops::SubAssign for ClockTime {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = self.saturating_sub(rhs);
    }
}

impl From<Duration> for ClockTime {
    #[inline]
    fn from(d: Duration) -> Self {
        Self(d.as_nanos() as u64)
    }
}

impl From<ClockTime> for Duration {
    #[inline]
    fn from(t: ClockTime) -> Self {
        if t.is_none() {
            Duration::ZERO
        } else {
            Duration::from_nanos(t.0)
        }
    }
}

impl std::fmt::Display for ClockTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "NONE")
        } else {
            let secs = self.secs();
            let ms = (self.0 / 1_000_000) % 1000;
            write!(f, "{}.{:03}s", secs, ms)
        }
    }
}

// ============================================================================
// Clock Capabilities
// ============================================================================

/// Capabilities and characteristics of a clock.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct ClockFlags(u32);

impl ClockFlags {
    /// No flags set.
    pub const NONE: Self = Self(0);
    /// Clock can be used as pipeline master (provides timing reference).
    pub const CAN_BE_MASTER: Self = Self(1 << 0);
    /// Clock can slave to another clock (adjust its rate).
    pub const CAN_SET_MASTER: Self = Self(1 << 1);
    /// Clock provides hardware timestamps (more accurate than software).
    pub const HARDWARE: Self = Self(1 << 2);
    /// Clock is from a network source (PTP, NTP).
    pub const NETWORK: Self = Self(1 << 3);
    /// Clock is real-time (audio device, maintains constant rate).
    pub const REALTIME: Self = Self(1 << 4);

    /// Check if a flag is set.
    #[inline]
    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    /// Set a flag.
    #[inline]
    pub const fn insert(self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }

    /// Clear a flag.
    #[inline]
    pub const fn remove(self, flag: Self) -> Self {
        Self(self.0 & !flag.0)
    }

    /// Combine flags using bitwise OR.
    #[inline]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// ============================================================================
// Clock Trait
// ============================================================================

/// A clock that provides the current time.
///
/// Implementations should provide monotonic time (never goes backwards).
/// Clocks are used to synchronize media playback and capture.
///
/// # Clock Selection Priority
///
/// When multiple clocks are available, the pipeline selects one based on
/// priority (see `ClockProvider::clock_priority()`):
/// - 0-99: Software clocks (system monotonic)
/// - 100-199: Hardware clocks (audio devices)
/// - 200-299: Network clocks (NTP)
/// - 300+: Precision clocks (PTP)
pub trait Clock: Send + Sync {
    /// Get the current time.
    fn now(&self) -> ClockTime;

    /// Get clock capabilities.
    fn flags(&self) -> ClockFlags {
        ClockFlags::CAN_BE_MASTER
    }

    /// Get clock resolution in nanoseconds.
    ///
    /// Returns 0 if resolution is unknown.
    fn resolution(&self) -> u64 {
        0
    }

    /// Get a human-readable name for the clock.
    fn name(&self) -> &str {
        "unknown"
    }
}

// ============================================================================
// Clock Provider Trait
// ============================================================================

/// Elements that can provide a clock for pipeline synchronization.
///
/// Some elements (particularly audio sinks) can provide a clock that
/// represents the rate at which media is being consumed. This clock
/// can be used as the pipeline's master clock for synchronization.
///
/// # Example
///
/// ```rust,ignore
/// impl ClockProvider for AlsaSink {
///     fn provide_clock(&self) -> Option<Arc<dyn Clock>> {
///         Some(Arc::new(AlsaClock::new(&self.pcm, self.sample_rate)))
///     }
///
///     fn clock_priority(&self) -> u32 {
///         100 // Hardware audio clock
///     }
/// }
/// ```
pub trait ClockProvider: Send + Sync {
    /// Return a clock if this element can provide one.
    ///
    /// Returns `None` if this element cannot provide a clock.
    fn provide_clock(&self) -> Option<Arc<dyn Clock>>;

    /// Priority for clock selection (higher = preferred).
    ///
    /// Priority ranges:
    /// - 0-99: Software clocks (system monotonic) - default
    /// - 100-199: Hardware clocks (audio devices)
    /// - 200-299: Network clocks (NTP)
    /// - 300+: Precision clocks (PTP)
    fn clock_priority(&self) -> u32 {
        0
    }
}

// ============================================================================
// SystemClock
// ============================================================================

/// System monotonic clock.
///
/// Uses `std::time::Instant` for monotonic time measurement.
/// Time is relative to when the clock was created.
pub struct SystemClock {
    epoch: Instant,
    name: String,
}

impl SystemClock {
    /// Create a new system clock with the current instant as epoch.
    pub fn new() -> Self {
        Self {
            epoch: Instant::now(),
            name: "system-monotonic".to_string(),
        }
    }

    /// Create a system clock with a custom name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            epoch: Instant::now(),
            name: name.into(),
        }
    }
}

impl Default for SystemClock {
    fn default() -> Self {
        Self::new()
    }
}

impl Clock for SystemClock {
    #[inline]
    fn now(&self) -> ClockTime {
        ClockTime::from_nanos(self.epoch.elapsed().as_nanos() as u64)
    }

    fn flags(&self) -> ClockFlags {
        ClockFlags::CAN_BE_MASTER
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// PipelineClock
// ============================================================================

/// Pipeline timing context.
///
/// Manages pipeline time with a base time (when the pipeline started).
/// Provides running time calculation and synchronization helpers.
///
/// # Running Time vs Clock Time
///
/// - **Clock time**: Absolute time from the clock
/// - **Base time**: Clock time when the pipeline started
/// - **Running time**: Clock time - Base time (time since pipeline start)
///
/// # Example
///
/// ```rust
/// use parallax::clock::{PipelineClock, ClockTime};
///
/// let clock = PipelineClock::system();
/// clock.start();
///
/// // ... later ...
/// let running = clock.running_time();
/// println!("Pipeline has been running for {}", running);
/// ```
pub struct PipelineClock {
    /// The underlying clock.
    clock: Arc<dyn Clock>,
    /// Base time (clock time when pipeline started).
    /// u64::MAX means not started.
    base_time: AtomicU64,
}

impl PipelineClock {
    /// Create a new pipeline clock with the given clock source.
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self {
            clock,
            base_time: AtomicU64::new(u64::MAX),
        }
    }

    /// Create a pipeline clock using the system monotonic clock.
    pub fn system() -> Self {
        Self::new(Arc::new(SystemClock::new()))
    }

    /// Start the pipeline clock (set base time to now).
    pub fn start(&self) {
        self.base_time.store(self.clock.now().0, Ordering::Release);
    }

    /// Reset the pipeline clock (clear base time).
    pub fn reset(&self) {
        self.base_time.store(u64::MAX, Ordering::Release);
    }

    /// Check if the pipeline clock has been started.
    #[inline]
    pub fn is_started(&self) -> bool {
        self.base_time.load(Ordering::Acquire) != u64::MAX
    }

    /// Get the base time (when pipeline started).
    ///
    /// Returns `ClockTime::NONE` if not started.
    #[inline]
    pub fn base_time(&self) -> ClockTime {
        ClockTime(self.base_time.load(Ordering::Acquire))
    }

    /// Get a reference to the underlying clock.
    #[inline]
    pub fn clock(&self) -> Arc<dyn Clock> {
        self.clock.clone()
    }

    /// Get the current clock time.
    #[inline]
    pub fn clock_time(&self) -> ClockTime {
        self.clock.now()
    }

    /// Get the running time (time since pipeline start).
    ///
    /// Returns `ClockTime::NONE` if not started.
    #[inline]
    pub fn running_time(&self) -> ClockTime {
        let base = self.base_time.load(Ordering::Acquire);
        if base == u64::MAX {
            return ClockTime::NONE;
        }
        self.clock.now().saturating_sub(ClockTime(base))
    }

    /// Convert a running time to a clock time.
    ///
    /// Returns `ClockTime::NONE` if not started.
    #[inline]
    pub fn to_clock_time(&self, running: ClockTime) -> ClockTime {
        let base = self.base_time.load(Ordering::Acquire);
        if base == u64::MAX || running.is_none() {
            return ClockTime::NONE;
        }
        ClockTime(base).saturating_add(running)
    }

    /// Convert a clock time to a running time.
    ///
    /// Returns `ClockTime::NONE` if not started or if clock_time < base_time.
    #[inline]
    pub fn to_running_time(&self, clock_time: ClockTime) -> ClockTime {
        let base = self.base_time.load(Ordering::Acquire);
        if base == u64::MAX || clock_time.is_none() {
            return ClockTime::NONE;
        }
        clock_time.saturating_sub(ClockTime(base))
    }

    /// Wait until the running time reaches the target.
    ///
    /// Returns immediately if:
    /// - The pipeline is not started
    /// - The target is NONE
    /// - The target has already passed
    pub async fn wait_until(&self, target: ClockTime) {
        if target.is_none() {
            return;
        }

        loop {
            let now = self.running_time();
            if now.is_none() || now >= target {
                break;
            }

            let wait = target.saturating_sub(now);
            if wait.is_none() || wait == ClockTime::ZERO {
                break;
            }

            // Sleep in small increments to handle clock adjustments
            let sleep_duration = Duration::from(wait).min(Duration::from_millis(10));
            tokio::time::sleep(sleep_duration).await;
        }
    }

    /// Wait for a duration from now.
    pub async fn wait_for(&self, duration: ClockTime) {
        if duration.is_none() || duration == ClockTime::ZERO {
            return;
        }

        let target = self.running_time().saturating_add(duration);
        self.wait_until(target).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_time_creation() {
        assert_eq!(ClockTime::from_nanos(1_000).nanos(), 1_000);
        assert_eq!(ClockTime::from_micros(1_000).nanos(), 1_000_000);
        assert_eq!(ClockTime::from_millis(1_000).nanos(), 1_000_000_000);
        assert_eq!(ClockTime::from_secs(1).nanos(), 1_000_000_000);
    }

    #[test]
    fn test_clock_time_conversions() {
        let t = ClockTime::from_secs(2);
        assert_eq!(t.secs(), 2);
        assert_eq!(t.millis(), 2_000);
        assert_eq!(t.micros(), 2_000_000);
        assert_eq!(t.nanos(), 2_000_000_000);
    }

    #[test]
    fn test_clock_time_subsec() {
        let t = ClockTime::from_secs_nanos(1, 500_000_000);
        assert_eq!(t.secs(), 1);
        assert_eq!(t.subsec_nanos(), 500_000_000);
        assert_eq!(t.millis(), 1500);
    }

    #[test]
    fn test_clock_time_none() {
        assert!(ClockTime::NONE.is_none());
        assert!(!ClockTime::NONE.is_some());
        assert!(!ClockTime::ZERO.is_none());
        assert!(ClockTime::ZERO.is_some());
    }

    #[test]
    fn test_clock_time_arithmetic() {
        let t1 = ClockTime::from_secs(1);
        let t2 = ClockTime::from_millis(500);

        assert_eq!((t1 + t2).millis(), 1500);
        assert_eq!((t1 - t2).millis(), 500);
    }

    #[test]
    fn test_clock_time_none_arithmetic() {
        let t = ClockTime::from_secs(1);
        let none = ClockTime::NONE;

        assert!((t + none).is_none());
        assert!((none + t).is_none());
        assert!((t - none).is_none());
        assert!((none - t).is_none());
    }

    #[test]
    fn test_clock_time_saturating() {
        let t = ClockTime::from_secs(1);

        // Subtraction saturates to zero
        let result = ClockTime::from_millis(100) - t;
        assert_eq!(result, ClockTime::ZERO);

        // Addition saturates to MAX
        let big = ClockTime::MAX;
        let result = big + t;
        assert_eq!(result, ClockTime::MAX);
    }

    #[test]
    fn test_clock_time_display() {
        assert_eq!(format!("{}", ClockTime::from_millis(1500)), "1.500s");
        assert_eq!(format!("{}", ClockTime::from_secs(0)), "0.000s");
        assert_eq!(format!("{}", ClockTime::NONE), "NONE");
    }

    #[test]
    fn test_clock_time_duration_conversion() {
        let t = ClockTime::from_millis(1500);
        let d: Duration = t.into();
        assert_eq!(d, Duration::from_millis(1500));

        let d2 = Duration::from_secs(2);
        let t2: ClockTime = d2.into();
        assert_eq!(t2.secs(), 2);
    }

    #[test]
    fn test_system_clock() {
        let clock = SystemClock::new();
        let t1 = clock.now();
        std::thread::sleep(Duration::from_millis(10));
        let t2 = clock.now();
        assert!(t2 > t1);
    }

    #[test]
    fn test_pipeline_clock_not_started() {
        let clock = PipelineClock::system();
        assert!(!clock.is_started());
        assert!(clock.running_time().is_none());
        assert!(clock.base_time().is_none());
    }

    #[test]
    fn test_pipeline_clock_started() {
        let clock = PipelineClock::system();
        clock.start();

        assert!(clock.is_started());
        assert!(clock.running_time().is_some());
        assert!(clock.base_time().is_some());

        // Running time should be very small
        let running = clock.running_time();
        assert!(running.millis() < 100);
    }

    #[test]
    fn test_pipeline_clock_reset() {
        let clock = PipelineClock::system();
        clock.start();
        assert!(clock.is_started());

        clock.reset();
        assert!(!clock.is_started());
        assert!(clock.running_time().is_none());
    }

    #[test]
    fn test_pipeline_clock_time_conversion() {
        let clock = PipelineClock::system();
        clock.start();

        let running = ClockTime::from_millis(100);
        let clock_time = clock.to_clock_time(running);
        assert!(clock_time.is_some());

        let back = clock.to_running_time(clock_time);
        assert_eq!(back, running);
    }
}
