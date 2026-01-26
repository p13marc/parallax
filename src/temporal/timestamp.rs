//! Timestamp and time range types.

use rkyv::{Archive, Deserialize, Serialize};
use std::cmp::Ordering;
use std::ops::{Add, Sub};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

/// A high-precision timestamp in nanoseconds since an epoch.
///
/// Timestamps are used for temporal ordering and alignment of buffers.
/// They can represent either:
/// - Wall-clock time (from system clock)
/// - Stream time (relative to stream start)
/// - Monotonic time (from monotonic clock)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
pub struct Timestamp {
    /// Nanoseconds since epoch.
    nanos: u64,
    /// The clock source this timestamp was derived from.
    source: ClockSource,
}

/// The source of a timestamp.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Archive, Serialize, Deserialize)]
pub enum ClockSource {
    /// Unknown or unspecified clock source.
    #[default]
    Unknown,
    /// Wall-clock time (system time since Unix epoch).
    WallClock,
    /// Monotonic clock (guaranteed to not go backwards).
    Monotonic,
    /// Stream-relative time (time since stream start).
    StreamTime,
    /// External clock (e.g., PTP, GPS).
    External,
}

impl Timestamp {
    /// Create a timestamp from nanoseconds with unknown source.
    pub const fn from_nanos(nanos: u64) -> Self {
        Self {
            nanos,
            source: ClockSource::Unknown,
        }
    }

    /// Create a timestamp from nanoseconds with a specific source.
    pub const fn from_nanos_with_source(nanos: u64, source: ClockSource) -> Self {
        Self { nanos, source }
    }

    /// Create a timestamp from milliseconds.
    pub const fn from_millis(millis: u64) -> Self {
        Self::from_nanos(millis * 1_000_000)
    }

    /// Create a timestamp from seconds.
    pub const fn from_secs(secs: u64) -> Self {
        Self::from_nanos(secs * 1_000_000_000)
    }

    /// Create a timestamp from a Duration.
    pub fn from_duration(duration: Duration) -> Self {
        Self::from_nanos(duration.as_nanos() as u64)
    }

    /// Create a timestamp from a Duration with a specific source.
    pub fn from_duration_with_source(duration: Duration, source: ClockSource) -> Self {
        Self::from_nanos_with_source(duration.as_nanos() as u64, source)
    }

    /// Get the current wall-clock time.
    pub fn now() -> Self {
        let duration = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO);
        Self::from_nanos_with_source(duration.as_nanos() as u64, ClockSource::WallClock)
    }

    /// Create a zero timestamp (epoch).
    pub const fn zero() -> Self {
        Self::from_nanos(0)
    }

    /// Get the raw nanoseconds value.
    pub const fn as_nanos(&self) -> u64 {
        self.nanos
    }

    /// Get the timestamp in milliseconds.
    pub const fn as_millis(&self) -> u64 {
        self.nanos / 1_000_000
    }

    /// Get the timestamp in seconds.
    pub const fn as_secs(&self) -> u64 {
        self.nanos / 1_000_000_000
    }

    /// Get the timestamp in seconds with fractional part.
    pub fn as_secs_f64(&self) -> f64 {
        self.nanos as f64 / 1_000_000_000.0
    }

    /// Convert to a Duration.
    pub fn as_duration(&self) -> Duration {
        Duration::from_nanos(self.nanos)
    }

    /// Get the clock source.
    pub const fn source(&self) -> ClockSource {
        self.source
    }

    /// Create a new timestamp with a different source.
    pub const fn with_source(self, source: ClockSource) -> Self {
        Self {
            nanos: self.nanos,
            source,
        }
    }

    /// Calculate the absolute difference between two timestamps.
    pub fn abs_diff(&self, other: &Self) -> Duration {
        let diff = self.nanos.abs_diff(other.nanos);
        Duration::from_nanos(diff)
    }

    /// Check if this timestamp is within a tolerance of another.
    pub fn within_tolerance(&self, other: &Self, tolerance: Duration) -> bool {
        self.abs_diff(other) <= tolerance
    }

    /// Saturating subtraction - returns zero if result would be negative.
    pub fn saturating_sub(&self, duration: Duration) -> Self {
        let nanos = self.nanos.saturating_sub(duration.as_nanos() as u64);
        Self {
            nanos,
            source: self.source,
        }
    }
}

impl Default for Timestamp {
    fn default() -> Self {
        Self::zero()
    }
}

impl PartialOrd for Timestamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Timestamp {
    fn cmp(&self, other: &Self) -> Ordering {
        self.nanos.cmp(&other.nanos)
    }
}

impl Add<Duration> for Timestamp {
    type Output = Self;

    fn add(self, rhs: Duration) -> Self::Output {
        Self {
            nanos: self.nanos + rhs.as_nanos() as u64,
            source: self.source,
        }
    }
}

impl Sub<Duration> for Timestamp {
    type Output = Self;

    fn sub(self, rhs: Duration) -> Self::Output {
        Self {
            nanos: self.nanos - rhs.as_nanos() as u64,
            source: self.source,
        }
    }
}

impl Sub<Timestamp> for Timestamp {
    type Output = Duration;

    fn sub(self, rhs: Timestamp) -> Self::Output {
        Duration::from_nanos(self.nanos.saturating_sub(rhs.nanos))
    }
}

impl From<Duration> for Timestamp {
    fn from(d: Duration) -> Self {
        Self::from_duration(d)
    }
}

impl From<Timestamp> for Duration {
    fn from(ts: Timestamp) -> Self {
        ts.as_duration()
    }
}

impl std::fmt::Display for Timestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secs = self.as_secs();
        let nanos = self.nanos % 1_000_000_000;
        write!(f, "{}.{:09}s", secs, nanos)
    }
}

/// A time range representing an interval [start, end).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Archive, Serialize, Deserialize)]
pub struct TimeRange {
    /// Start of the range (inclusive).
    pub start: Timestamp,
    /// End of the range (exclusive).
    pub end: Timestamp,
}

impl TimeRange {
    /// Create a new time range.
    pub fn new(start: Timestamp, end: Timestamp) -> Self {
        debug_assert!(start <= end, "start must be <= end");
        Self { start, end }
    }

    /// Create a time range from a start timestamp and duration.
    pub fn from_start_duration(start: Timestamp, duration: Duration) -> Self {
        Self {
            start,
            end: start + duration,
        }
    }

    /// Create a time range centered on a timestamp with a given tolerance.
    pub fn centered(center: Timestamp, tolerance: Duration) -> Self {
        Self {
            start: center.saturating_sub(tolerance),
            end: center + tolerance,
        }
    }

    /// Get the duration of this range.
    pub fn duration(&self) -> Duration {
        self.end - self.start
    }

    /// Check if a timestamp falls within this range.
    pub fn contains(&self, ts: Timestamp) -> bool {
        ts >= self.start && ts < self.end
    }

    /// Check if this range overlaps with another.
    pub fn overlaps(&self, other: &TimeRange) -> bool {
        self.start < other.end && other.start < self.end
    }

    /// Get the intersection of two ranges, if any.
    pub fn intersection(&self, other: &TimeRange) -> Option<TimeRange> {
        if !self.overlaps(other) {
            return None;
        }
        Some(TimeRange {
            start: std::cmp::max(self.start, other.start),
            end: std::cmp::min(self.end, other.end),
        })
    }

    /// Extend the range to include a timestamp.
    pub fn extend_to(&mut self, ts: Timestamp) {
        if ts < self.start {
            self.start = ts;
        }
        if ts >= self.end {
            self.end = ts + Duration::from_nanos(1);
        }
    }

    /// Check if this range is empty (zero duration).
    pub fn is_empty(&self) -> bool {
        self.start >= self.end
    }
}

impl std::fmt::Display for TimeRange {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {})", self.start, self.end)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_creation() {
        let ts = Timestamp::from_nanos(1_000_000_000);
        assert_eq!(ts.as_secs(), 1);
        assert_eq!(ts.as_millis(), 1000);
        assert_eq!(ts.as_nanos(), 1_000_000_000);
    }

    #[test]
    fn test_timestamp_from_millis() {
        let ts = Timestamp::from_millis(1500);
        assert_eq!(ts.as_millis(), 1500);
        assert_eq!(ts.as_secs(), 1);
    }

    #[test]
    fn test_timestamp_ordering() {
        let t1 = Timestamp::from_millis(100);
        let t2 = Timestamp::from_millis(200);
        let t3 = Timestamp::from_millis(100);

        assert!(t1 < t2);
        assert!(t2 > t1);
        assert_eq!(t1, t3);
    }

    #[test]
    fn test_timestamp_arithmetic() {
        let ts = Timestamp::from_secs(10);
        let added = ts + Duration::from_secs(5);
        let subtracted = ts - Duration::from_secs(3);

        assert_eq!(added.as_secs(), 15);
        assert_eq!(subtracted.as_secs(), 7);
    }

    #[test]
    fn test_timestamp_diff() {
        let t1 = Timestamp::from_secs(10);
        let t2 = Timestamp::from_secs(15);

        let diff: Duration = t2 - t1;
        assert_eq!(diff, Duration::from_secs(5));
    }

    #[test]
    fn test_timestamp_abs_diff() {
        let t1 = Timestamp::from_secs(10);
        let t2 = Timestamp::from_secs(15);

        assert_eq!(t1.abs_diff(&t2), Duration::from_secs(5));
        assert_eq!(t2.abs_diff(&t1), Duration::from_secs(5));
    }

    #[test]
    fn test_timestamp_within_tolerance() {
        let t1 = Timestamp::from_millis(100);
        let t2 = Timestamp::from_millis(105);

        assert!(t1.within_tolerance(&t2, Duration::from_millis(10)));
        assert!(!t1.within_tolerance(&t2, Duration::from_millis(3)));
    }

    #[test]
    fn test_time_range_contains() {
        let range = TimeRange::new(Timestamp::from_secs(10), Timestamp::from_secs(20));

        assert!(!range.contains(Timestamp::from_secs(5)));
        assert!(range.contains(Timestamp::from_secs(10)));
        assert!(range.contains(Timestamp::from_secs(15)));
        assert!(!range.contains(Timestamp::from_secs(20))); // exclusive end
    }

    #[test]
    fn test_time_range_overlaps() {
        let r1 = TimeRange::new(Timestamp::from_secs(10), Timestamp::from_secs(20));
        let r2 = TimeRange::new(Timestamp::from_secs(15), Timestamp::from_secs(25));
        let r3 = TimeRange::new(Timestamp::from_secs(25), Timestamp::from_secs(30));

        assert!(r1.overlaps(&r2));
        assert!(r2.overlaps(&r1));
        assert!(!r1.overlaps(&r3));
        assert!(!r3.overlaps(&r1));
    }

    #[test]
    fn test_time_range_intersection() {
        let r1 = TimeRange::new(Timestamp::from_secs(10), Timestamp::from_secs(20));
        let r2 = TimeRange::new(Timestamp::from_secs(15), Timestamp::from_secs(25));

        let intersection = r1.intersection(&r2).unwrap();
        assert_eq!(intersection.start, Timestamp::from_secs(15));
        assert_eq!(intersection.end, Timestamp::from_secs(20));
    }

    #[test]
    fn test_time_range_centered() {
        let center = Timestamp::from_secs(100);
        let range = TimeRange::centered(center, Duration::from_secs(10));

        assert_eq!(range.start.as_secs(), 90);
        assert_eq!(range.end.as_secs(), 110);
    }

    #[test]
    fn test_clock_source() {
        let ts = Timestamp::from_nanos_with_source(1000, ClockSource::WallClock);
        assert_eq!(ts.source(), ClockSource::WallClock);

        let ts2 = ts.with_source(ClockSource::Monotonic);
        assert_eq!(ts2.source(), ClockSource::Monotonic);
        assert_eq!(ts2.as_nanos(), 1000);
    }

    #[test]
    fn test_timestamp_display() {
        let ts = Timestamp::from_nanos(1_500_000_000);
        assert_eq!(format!("{}", ts), "1.500000000s");
    }
}
