//! Metadata operation elements.
//!
//! Elements for manipulating buffer metadata: timestamps, sequence numbers, etc.

use crate::buffer::Buffer;
use crate::clock::ClockTime;
use crate::element::Element;
use crate::error::Result;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Adds or updates sequence numbers on buffers.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::SequenceNumber;
///
/// // Start from sequence 0
/// let seq = SequenceNumber::new();
///
/// // Start from a specific sequence
/// let seq = SequenceNumber::starting_at(1000);
/// ```
pub struct SequenceNumber {
    name: String,
    counter: AtomicU64,
    increment: u64,
}

impl SequenceNumber {
    /// Create a new sequence number element starting at 0.
    pub fn new() -> Self {
        Self::starting_at(0)
    }

    /// Create starting at a specific sequence number.
    pub fn starting_at(start: u64) -> Self {
        Self {
            name: "sequence-number".to_string(),
            counter: AtomicU64::new(start),
            increment: 1,
        }
    }

    /// Set the increment value.
    pub fn with_increment(mut self, increment: u64) -> Self {
        self.increment = increment;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the current sequence number (next to be assigned).
    pub fn current(&self) -> u64 {
        self.counter.load(Ordering::SeqCst)
    }

    /// Reset to starting value.
    pub fn reset(&self, start: u64) {
        self.counter.store(start, Ordering::SeqCst);
    }
}

impl Default for SequenceNumber {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for SequenceNumber {
    fn process(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
        let seq = self.counter.fetch_add(self.increment, Ordering::SeqCst);
        buffer.metadata_mut().sequence = seq;
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Adds or updates timestamps on buffers.
///
/// Supports different timestamp modes: system time, monotonic time,
/// or relative to pipeline start.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::{Timestamper, TimestampMode};
///
/// // Use system time (wall clock)
/// let ts = Timestamper::new(TimestampMode::SystemTime);
///
/// // Use monotonic time since element creation
/// let ts = Timestamper::new(TimestampMode::Monotonic);
/// ```
pub struct Timestamper {
    name: String,
    mode: TimestampMode,
    start_instant: Instant,
    start_system: SystemTime,
    count: AtomicU64,
}

/// Timestamp mode for the Timestamper element.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimestampMode {
    /// System time (nanoseconds since Unix epoch).
    #[default]
    SystemTime,
    /// Monotonic time since element creation (nanoseconds).
    Monotonic,
    /// Preserve existing timestamps (pass through).
    Preserve,
    /// Set PTS (presentation timestamp) only.
    PtsOnly,
    /// Set DTS (decode timestamp) only.
    DtsOnly,
}

impl Timestamper {
    /// Create a new timestamper with the specified mode.
    pub fn new(mode: TimestampMode) -> Self {
        Self {
            name: "timestamper".to_string(),
            mode,
            start_instant: Instant::now(),
            start_system: SystemTime::now(),
            count: AtomicU64::new(0),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the timestamp mode.
    pub fn mode(&self) -> TimestampMode {
        self.mode
    }

    /// Set the timestamp mode.
    pub fn set_mode(&mut self, mode: TimestampMode) {
        self.mode = mode;
    }

    /// Get the number of buffers processed.
    pub fn buffer_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Reset the start time.
    pub fn reset(&mut self) {
        self.start_instant = Instant::now();
        self.start_system = SystemTime::now();
        self.count.store(0, Ordering::Relaxed);
    }

    /// Get current timestamp based on mode.
    fn get_timestamp(&self) -> Option<ClockTime> {
        match self.mode {
            TimestampMode::SystemTime => SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .ok()
                .map(ClockTime::from),
            TimestampMode::Monotonic => Some(ClockTime::from(self.start_instant.elapsed())),
            TimestampMode::Preserve | TimestampMode::PtsOnly | TimestampMode::DtsOnly => None,
        }
    }
}

impl Element for Timestamper {
    fn process(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);

        match self.mode {
            TimestampMode::SystemTime | TimestampMode::Monotonic => {
                if let Some(ts) = self.get_timestamp() {
                    let meta = buffer.metadata_mut();
                    meta.pts = ts;
                    meta.dts = ts;
                }
            }
            TimestampMode::PtsOnly => {
                if let Some(ts) = self.get_timestamp() {
                    buffer.metadata_mut().pts = ts;
                }
            }
            TimestampMode::DtsOnly => {
                if let Some(ts) = self.get_timestamp() {
                    buffer.metadata_mut().dts = ts;
                }
            }
            TimestampMode::Preserve => {
                // Do nothing
            }
        }

        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Injects or modifies arbitrary metadata on buffers.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::MetadataInject;
///
/// let inject = MetadataInject::new()
///     .with_stream_id(42)
///     .with_duration(Duration::from_millis(100));
/// ```
pub struct MetadataInject {
    name: String,
    stream_id: Option<u32>,
    duration: Option<ClockTime>,
    offset: Option<u64>,
    count: AtomicU64,
}

impl MetadataInject {
    /// Create a new metadata inject element.
    pub fn new() -> Self {
        Self {
            name: "metadata-inject".to_string(),
            stream_id: None,
            duration: None,
            offset: None,
            count: AtomicU64::new(0),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the stream ID to inject.
    pub fn with_stream_id(mut self, id: u32) -> Self {
        self.stream_id = Some(id);
        self
    }

    /// Set the duration to inject.
    pub fn with_duration(mut self, duration: ClockTime) -> Self {
        self.duration = Some(duration);
        self
    }

    /// Set the duration to inject (from std Duration).
    pub fn with_duration_from(mut self, duration: Duration) -> Self {
        self.duration = Some(ClockTime::from(duration));
        self
    }

    /// Set the offset to inject.
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Get the buffer count.
    pub fn buffer_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl Default for MetadataInject {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for MetadataInject {
    fn process(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);

        let meta = buffer.metadata_mut();

        if let Some(id) = self.stream_id {
            meta.stream_id = id;
        }
        if let Some(dur) = self.duration {
            meta.duration = dur;
        }
        if let Some(off) = self.offset {
            meta.offset = Some(off);
        }

        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Debug element that logs timestamp information flowing through the pipeline.
///
/// This is useful for debugging timing issues, verifying PTS/DTS propagation,
/// and understanding pipeline timing characteristics.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::TimestampDebug;
///
/// // Basic logging to stderr
/// let debug = TimestampDebug::new();
///
/// // Custom prefix and format
/// let debug = TimestampDebug::new()
///     .with_prefix("video:")
///     .with_format(TimestampFormat::Detailed);
///
/// // Verbose mode shows all fields
/// let debug = TimestampDebug::verbose();
///
/// // Get statistics
/// println!("Stats: {:?}", debug.stats());
/// ```
pub struct TimestampDebug {
    name: String,
    prefix: String,
    format: TimestampFormat,
    log_level: TimestampDebugLevel,
    stats: TimestampDebugStats,
    start_instant: Instant,
}

/// Output format for timestamp debug information.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimestampFormat {
    /// Just PTS value: `[00:00:01.234]`
    #[default]
    Simple,
    /// PTS and DTS: `PTS=00:00:01.234 DTS=00:00:01.200`
    Standard,
    /// Full timing info including duration and sequence
    Detailed,
    /// Machine-readable nanosecond values
    Raw,
}

/// Log level for timestamp debug output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TimestampDebugLevel {
    /// Log every buffer
    #[default]
    All,
    /// Log every Nth buffer
    Sample(u64),
    /// Only log warnings/anomalies (gaps, backwards PTS)
    Warnings,
    /// Silent (stats only)
    Silent,
}

/// Statistics collected by TimestampDebug.
#[derive(Debug, Clone, Default)]
pub struct TimestampDebugStats {
    /// Number of buffers processed
    pub buffer_count: u64,
    /// Number of buffers with no PTS
    pub missing_pts_count: u64,
    /// Number of buffers with backwards PTS (PTS decreased)
    pub backwards_pts_count: u64,
    /// Number of PTS discontinuities (gaps larger than expected)
    pub discontinuity_count: u64,
    /// Minimum PTS seen
    pub min_pts: Option<ClockTime>,
    /// Maximum PTS seen
    pub max_pts: Option<ClockTime>,
    /// Last PTS seen
    pub last_pts: Option<ClockTime>,
    /// Average inter-frame interval
    pub avg_interval_ns: Option<f64>,
}

impl TimestampDebug {
    /// Create a new timestamp debug element with default settings.
    pub fn new() -> Self {
        Self {
            name: "timestamp-debug".to_string(),
            prefix: String::new(),
            format: TimestampFormat::default(),
            log_level: TimestampDebugLevel::default(),
            stats: TimestampDebugStats::default(),
            start_instant: Instant::now(),
        }
    }

    /// Create a verbose timestamp debug element that logs all fields.
    pub fn verbose() -> Self {
        Self {
            format: TimestampFormat::Detailed,
            ..Self::new()
        }
    }

    /// Create a silent timestamp debug element that only collects stats.
    pub fn silent() -> Self {
        Self {
            log_level: TimestampDebugLevel::Silent,
            ..Self::new()
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set a prefix for log messages.
    pub fn with_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.prefix = prefix.into();
        self
    }

    /// Set the output format.
    pub fn with_format(mut self, format: TimestampFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the log level.
    pub fn with_log_level(mut self, level: TimestampDebugLevel) -> Self {
        self.log_level = level;
        self
    }

    /// Get the collected statistics.
    pub fn stats(&self) -> &TimestampDebugStats {
        &self.stats
    }

    /// Reset the statistics.
    pub fn reset_stats(&mut self) {
        self.stats = TimestampDebugStats::default();
        self.start_instant = Instant::now();
    }

    /// Format a ClockTime for display.
    fn format_time(&self, time: ClockTime) -> String {
        if time == ClockTime::NONE {
            return "NONE".to_string();
        }

        match self.format {
            TimestampFormat::Raw => format!("{}ns", time.nanos()),
            TimestampFormat::Simple | TimestampFormat::Standard | TimestampFormat::Detailed => {
                let nanos = time.nanos();
                let hours = nanos / 3_600_000_000_000;
                let minutes = (nanos % 3_600_000_000_000) / 60_000_000_000;
                let seconds = (nanos % 60_000_000_000) / 1_000_000_000;
                let millis = (nanos % 1_000_000_000) / 1_000_000;
                format!("{:02}:{:02}:{:02}.{:03}", hours, minutes, seconds, millis)
            }
        }
    }

    /// Log a buffer's timestamp information.
    fn log_buffer(&self, buffer: &Buffer, warning: Option<&str>) {
        let meta = buffer.metadata();
        let prefix = if self.prefix.is_empty() {
            String::new()
        } else {
            format!("{} ", self.prefix)
        };

        let wall_time = self.format_time(ClockTime::from(self.start_instant.elapsed()));

        match self.format {
            TimestampFormat::Simple => {
                let warning_str = warning.map(|w| format!(" [{}]", w)).unwrap_or_default();
                eprintln!(
                    "{}[{}] seq={}{warning_str}",
                    prefix,
                    self.format_time(meta.pts),
                    meta.sequence
                );
            }
            TimestampFormat::Standard => {
                let warning_str = warning.map(|w| format!(" [{}]", w)).unwrap_or_default();
                eprintln!(
                    "{}PTS={} DTS={} seq={}{warning_str}",
                    prefix,
                    self.format_time(meta.pts),
                    self.format_time(meta.dts),
                    meta.sequence
                );
            }
            TimestampFormat::Detailed => {
                let warning_str = warning.map(|w| format!(" [{}]", w)).unwrap_or_default();
                eprintln!(
                    "{}@{wall_time} seq={} PTS={} DTS={} dur={} len={} stream={} flags={:?}{warning_str}",
                    prefix,
                    meta.sequence,
                    self.format_time(meta.pts),
                    self.format_time(meta.dts),
                    self.format_time(meta.duration),
                    buffer.len(),
                    meta.stream_id,
                    meta.flags
                );
            }
            TimestampFormat::Raw => {
                let warning_str = warning
                    .map(|w| format!(" warning={}", w))
                    .unwrap_or_default();
                eprintln!(
                    "{}seq={} pts={} dts={} dur={} len={} stream={}{warning_str}",
                    prefix,
                    meta.sequence,
                    meta.pts.nanos(),
                    meta.dts.nanos(),
                    meta.duration.nanos(),
                    buffer.len(),
                    meta.stream_id
                );
            }
        }
    }

    /// Update statistics with a buffer's timing info.
    fn update_stats(&mut self, buffer: &Buffer) -> Option<&'static str> {
        let pts = buffer.metadata().pts;
        self.stats.buffer_count += 1;

        // Track missing PTS
        if pts == ClockTime::NONE {
            self.stats.missing_pts_count += 1;
            return Some("MISSING_PTS");
        }

        let mut warning = None;

        // Track backwards PTS
        if let Some(last) = self.stats.last_pts {
            if last != ClockTime::NONE && pts < last {
                self.stats.backwards_pts_count += 1;
                warning = Some("BACKWARDS_PTS");
            }

            // Track discontinuities (more than 2x expected interval)
            if let Some(avg) = self.stats.avg_interval_ns {
                if last != ClockTime::NONE {
                    let gap = pts.nanos().saturating_sub(last.nanos()) as f64;
                    if gap > avg * 2.0 {
                        self.stats.discontinuity_count += 1;
                        warning = Some("DISCONTINUITY");
                    }
                }
            }

            // Update average interval
            if last != ClockTime::NONE {
                let interval = pts.nanos().saturating_sub(last.nanos()) as f64;
                if let Some(avg) = self.stats.avg_interval_ns {
                    // Running average
                    self.stats.avg_interval_ns =
                        Some(avg + (interval - avg) / self.stats.buffer_count as f64);
                } else {
                    self.stats.avg_interval_ns = Some(interval);
                }
            }
        }

        // Update min/max
        match self.stats.min_pts {
            None => self.stats.min_pts = Some(pts),
            Some(min) if pts < min => self.stats.min_pts = Some(pts),
            _ => {}
        }
        match self.stats.max_pts {
            None => self.stats.max_pts = Some(pts),
            Some(max) if pts > max => self.stats.max_pts = Some(pts),
            _ => {}
        }

        // Update last PTS
        self.stats.last_pts = Some(pts);

        warning
    }
}

impl Default for TimestampDebug {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for TimestampDebug {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Update stats and check for warnings
        let warning = self.update_stats(&buffer);

        // Determine if we should log
        let should_log = match self.log_level {
            TimestampDebugLevel::All => true,
            TimestampDebugLevel::Sample(n) => self.stats.buffer_count % n == 0,
            TimestampDebugLevel::Warnings => warning.is_some(),
            TimestampDebugLevel::Silent => false,
        };

        if should_log {
            self.log_buffer(&buffer, warning);
        }

        // Pass through unchanged
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(128, 128).unwrap())
    }

    fn create_test_buffer(seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 64);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    // SequenceNumber tests

    #[test]
    fn test_sequence_number_basic() {
        let mut seq = SequenceNumber::new();

        let buf1 = seq.process(create_test_buffer(999)).unwrap().unwrap();
        let buf2 = seq.process(create_test_buffer(888)).unwrap().unwrap();
        let buf3 = seq.process(create_test_buffer(777)).unwrap().unwrap();

        assert_eq!(buf1.metadata().sequence, 0);
        assert_eq!(buf2.metadata().sequence, 1);
        assert_eq!(buf3.metadata().sequence, 2);
    }

    #[test]
    fn test_sequence_number_starting_at() {
        let mut seq = SequenceNumber::starting_at(100);

        let buf = seq.process(create_test_buffer(0)).unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 100);
    }

    #[test]
    fn test_sequence_number_increment() {
        let mut seq = SequenceNumber::starting_at(0).with_increment(10);

        let buf1 = seq.process(create_test_buffer(0)).unwrap().unwrap();
        let buf2 = seq.process(create_test_buffer(0)).unwrap().unwrap();

        assert_eq!(buf1.metadata().sequence, 0);
        assert_eq!(buf2.metadata().sequence, 10);
    }

    #[test]
    fn test_sequence_number_reset() {
        let mut seq = SequenceNumber::new();

        seq.process(create_test_buffer(0)).unwrap();
        seq.process(create_test_buffer(0)).unwrap();
        assert_eq!(seq.current(), 2);

        seq.reset(50);
        assert_eq!(seq.current(), 50);
    }

    // Timestamper tests

    #[test]
    fn test_timestamper_system_time() {
        let mut ts = Timestamper::new(TimestampMode::SystemTime);

        let buffer = create_test_buffer(0);
        // Default pts is ZERO
        assert_eq!(buffer.metadata().pts, crate::clock::ClockTime::ZERO);

        let result = ts.process(buffer).unwrap().unwrap();
        // After timestamping, pts should be non-zero (system time since epoch)
        assert!(result.metadata().pts.nanos() > 0);
        assert!(result.metadata().dts.nanos() > 0);
    }

    #[test]
    fn test_timestamper_monotonic() {
        let mut ts = Timestamper::new(TimestampMode::Monotonic);

        std::thread::sleep(Duration::from_millis(10));

        let result = ts.process(create_test_buffer(0)).unwrap().unwrap();
        let pts = result.metadata().pts;

        // Should be at least 10ms
        assert!(pts.millis() >= 10);
    }

    #[test]
    fn test_timestamper_preserve() {
        let mut ts = Timestamper::new(TimestampMode::Preserve);

        let buffer = create_test_buffer(0);
        let result = ts.process(buffer).unwrap().unwrap();

        // Should remain at default (ZERO, which is_none() == false, but value is 0)
        assert_eq!(result.metadata().pts, crate::clock::ClockTime::default());
    }

    #[test]
    fn test_timestamper_buffer_count() {
        let mut ts = Timestamper::new(TimestampMode::SystemTime);

        ts.process(create_test_buffer(0)).unwrap();
        ts.process(create_test_buffer(1)).unwrap();

        assert_eq!(ts.buffer_count(), 2);
    }

    // MetadataInject tests

    #[test]
    fn test_metadata_inject_stream_id() {
        let mut inject = MetadataInject::new().with_stream_id(42);

        let buffer = create_test_buffer(0);
        assert_eq!(buffer.metadata().stream_id, 0); // Default is 0

        let result = inject.process(buffer).unwrap().unwrap();
        assert_eq!(result.metadata().stream_id, 42);
    }

    #[test]
    fn test_metadata_inject_duration() {
        let mut inject = MetadataInject::new().with_duration(crate::clock::ClockTime::from_secs(5));

        let result = inject.process(create_test_buffer(0)).unwrap().unwrap();
        assert_eq!(
            result.metadata().duration,
            crate::clock::ClockTime::from_secs(5)
        );
    }

    #[test]
    fn test_metadata_inject_offset() {
        let mut inject = MetadataInject::new().with_offset(1000);

        let result = inject.process(create_test_buffer(0)).unwrap().unwrap();
        assert_eq!(result.metadata().offset, Some(1000));
    }

    #[test]
    fn test_metadata_inject_combined() {
        let mut inject = MetadataInject::new()
            .with_stream_id(1)
            .with_duration(crate::clock::ClockTime::from_millis(100))
            .with_offset(0);

        let result = inject.process(create_test_buffer(0)).unwrap().unwrap();
        assert_eq!(result.metadata().stream_id, 1);
        assert_eq!(
            result.metadata().duration,
            crate::clock::ClockTime::from_millis(100)
        );
        assert_eq!(result.metadata().offset, Some(0));
    }

    #[test]
    fn test_metadata_inject_count() {
        let mut inject = MetadataInject::new();

        inject.process(create_test_buffer(0)).unwrap();
        inject.process(create_test_buffer(1)).unwrap();

        assert_eq!(inject.buffer_count(), 2);
    }

    // TimestampDebug tests

    fn create_timestamped_buffer(seq: u64, pts_nanos: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 64);
        let mut meta = Metadata::from_sequence(seq);
        meta.pts = ClockTime::from_nanos(pts_nanos);
        Buffer::new(handle, meta)
    }

    #[test]
    fn test_timestamp_debug_basic() {
        let mut debug = TimestampDebug::silent(); // Silent to avoid noise in tests

        let buf = create_timestamped_buffer(0, 1_000_000_000); // 1 second
        let result = debug.process(buf).unwrap();

        assert!(result.is_some()); // Should pass through
        assert_eq!(debug.stats().buffer_count, 1);
    }

    #[test]
    fn test_timestamp_debug_stats_min_max() {
        let mut debug = TimestampDebug::silent();

        debug
            .process(create_timestamped_buffer(0, 100_000_000))
            .unwrap(); // 100ms
        debug
            .process(create_timestamped_buffer(1, 200_000_000))
            .unwrap(); // 200ms
        debug
            .process(create_timestamped_buffer(2, 150_000_000))
            .unwrap(); // 150ms

        let stats = debug.stats();
        assert_eq!(stats.buffer_count, 3);
        assert_eq!(stats.min_pts, Some(ClockTime::from_nanos(100_000_000)));
        assert_eq!(stats.max_pts, Some(ClockTime::from_nanos(200_000_000)));
        assert_eq!(stats.last_pts, Some(ClockTime::from_nanos(150_000_000)));
    }

    #[test]
    fn test_timestamp_debug_backwards_pts() {
        let mut debug = TimestampDebug::silent();

        debug
            .process(create_timestamped_buffer(0, 200_000_000))
            .unwrap(); // 200ms
        debug
            .process(create_timestamped_buffer(1, 100_000_000))
            .unwrap(); // 100ms (backwards!)

        let stats = debug.stats();
        assert_eq!(stats.backwards_pts_count, 1);
    }

    #[test]
    fn test_timestamp_debug_missing_pts() {
        let mut debug = TimestampDebug::silent();

        // Buffer with NONE PTS
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 64);
        let mut meta = Metadata::from_sequence(0);
        meta.pts = ClockTime::NONE;
        let buf = Buffer::new(handle, meta);

        debug.process(buf).unwrap();

        assert_eq!(debug.stats().missing_pts_count, 1);
    }

    #[test]
    fn test_timestamp_debug_average_interval() {
        let mut debug = TimestampDebug::silent();

        // 33ms intervals (30fps)
        debug.process(create_timestamped_buffer(0, 0)).unwrap();
        debug
            .process(create_timestamped_buffer(1, 33_000_000))
            .unwrap();
        debug
            .process(create_timestamped_buffer(2, 66_000_000))
            .unwrap();
        debug
            .process(create_timestamped_buffer(3, 99_000_000))
            .unwrap();

        let stats = debug.stats();
        // Average should be around 33ms
        let avg = stats.avg_interval_ns.unwrap();
        assert!(avg > 30_000_000.0 && avg < 36_000_000.0);
    }

    #[test]
    fn test_timestamp_debug_reset() {
        let mut debug = TimestampDebug::silent();

        debug
            .process(create_timestamped_buffer(0, 100_000_000))
            .unwrap();
        debug
            .process(create_timestamped_buffer(1, 200_000_000))
            .unwrap();

        assert_eq!(debug.stats().buffer_count, 2);

        debug.reset_stats();

        assert_eq!(debug.stats().buffer_count, 0);
        assert!(debug.stats().min_pts.is_none());
        assert!(debug.stats().max_pts.is_none());
    }

    #[test]
    fn test_timestamp_debug_format_time() {
        let debug = TimestampDebug::new();

        // 1h 23m 45s 678ms
        let time = ClockTime::from_nanos(
            1 * 3_600_000_000_000 + 23 * 60_000_000_000 + 45 * 1_000_000_000 + 678_000_000,
        );
        let formatted = debug.format_time(time);
        assert_eq!(formatted, "01:23:45.678");

        // NONE
        let formatted_none = debug.format_time(ClockTime::NONE);
        assert_eq!(formatted_none, "NONE");
    }

    #[test]
    fn test_timestamp_debug_format_raw() {
        let debug = TimestampDebug::new().with_format(TimestampFormat::Raw);

        let time = ClockTime::from_nanos(1_234_567_890);
        let formatted = debug.format_time(time);
        assert_eq!(formatted, "1234567890ns");
    }

    #[test]
    fn test_timestamp_debug_verbose() {
        let debug = TimestampDebug::verbose();
        assert_eq!(debug.format, TimestampFormat::Detailed);
    }

    #[test]
    fn test_timestamp_debug_builder() {
        let debug = TimestampDebug::new()
            .with_name("video-debug")
            .with_prefix("video:")
            .with_format(TimestampFormat::Standard)
            .with_log_level(TimestampDebugLevel::Sample(10));

        assert_eq!(debug.name(), "video-debug");
        assert_eq!(debug.prefix, "video:");
        assert_eq!(debug.format, TimestampFormat::Standard);
        assert!(matches!(debug.log_level, TimestampDebugLevel::Sample(10)));
    }
}
