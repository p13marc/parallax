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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::CpuSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    fn create_test_buffer(seq: u64) -> Buffer {
        let segment = Arc::new(CpuSegment::new(64).unwrap());
        let handle = MemoryHandle::from_segment(segment);
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
}
