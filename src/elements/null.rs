//! Null elements - NullSink and NullSource.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Sink, Source};
use crate::error::Result;
use crate::memory::HeapSegment;
use crate::metadata::Metadata;
use std::sync::Arc;

/// A sink that discards all buffers.
///
/// This is useful for:
/// - Benchmarking pipeline throughput
/// - Testing source elements
/// - Draining a pipeline without side effects
///
/// # Example
///
/// ```rust
/// use parallax::elements::NullSink;
/// use parallax::element::Sink;
/// # use parallax::buffer::{Buffer, MemoryHandle};
/// # use parallax::memory::HeapSegment;
/// # use parallax::metadata::Metadata;
/// # use std::sync::Arc;
///
/// let mut sink = NullSink::new();
///
/// // Create a test buffer
/// # let segment = Arc::new(HeapSegment::new(8).unwrap());
/// # let handle = MemoryHandle::from_segment(segment);
/// # let buffer = Buffer::new(handle, Metadata::with_sequence(0));
///
/// // NullSink just discards the buffer
/// sink.consume(buffer).unwrap();
///
/// // Check how many buffers were consumed
/// assert_eq!(sink.count(), 1);
/// ```
pub struct NullSink {
    name: String,
    count: u64,
}

impl NullSink {
    /// Create a new NullSink.
    pub fn new() -> Self {
        Self {
            name: "nullsink".to_string(),
            count: 0,
        }
    }

    /// Create a new NullSink with a custom name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            count: 0,
        }
    }

    /// Get the number of buffers consumed.
    pub fn count(&self) -> u64 {
        self.count
    }
}

impl Default for NullSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for NullSink {
    fn consume(&mut self, _buffer: Buffer) -> Result<()> {
        self.count += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A source that produces a fixed number of empty buffers.
///
/// This is useful for:
/// - Testing pipeline throughput
/// - Benchmarking element processing
/// - Testing sink elements
///
/// # Example
///
/// ```rust
/// use parallax::elements::NullSource;
/// use parallax::element::Source;
///
/// let mut source = NullSource::new(5);
///
/// // Produces 5 buffers, then EOS
/// for i in 0..5 {
///     let buffer = source.produce().unwrap();
///     assert!(buffer.is_some());
/// }
///
/// // After 5 buffers, returns None (EOS)
/// let buffer = source.produce().unwrap();
/// assert!(buffer.is_none());
/// ```
pub struct NullSource {
    name: String,
    /// Number of buffers to produce.
    count: u64,
    /// Current sequence number.
    current: u64,
    /// Size of each buffer in bytes.
    buffer_size: usize,
}

impl NullSource {
    /// Create a new NullSource that produces `count` buffers.
    pub fn new(count: u64) -> Self {
        Self {
            name: "nullsource".to_string(),
            count,
            current: 0,
            buffer_size: 64,
        }
    }

    /// Create a new NullSource with a custom name.
    pub fn with_name(name: impl Into<String>, count: u64) -> Self {
        Self {
            name: name.into(),
            count,
            current: 0,
            buffer_size: 64,
        }
    }

    /// Set the buffer size in bytes.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Get the number of buffers remaining.
    pub fn remaining(&self) -> u64 {
        self.count.saturating_sub(self.current)
    }
}

impl Source for NullSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.current >= self.count {
            return Ok(None);
        }

        let segment = Arc::new(HeapSegment::new(self.buffer_size)?);
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::with_sequence(self.current));

        self.current += 1;
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_null_sink_consumes() {
        let mut sink = NullSink::new();
        assert_eq!(sink.count(), 0);

        let segment = Arc::new(HeapSegment::new(64).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::with_sequence(0));

        sink.consume(buffer).unwrap();
        assert_eq!(sink.count(), 1);
    }

    #[test]
    fn test_null_sink_custom_name() {
        let sink = NullSink::with_name("my_sink");
        assert_eq!(sink.name(), "my_sink");
    }

    #[test]
    fn test_null_source_produces_count() {
        let mut source = NullSource::new(5);
        assert_eq!(source.remaining(), 5);

        for i in 0..5 {
            let buffer = source.produce().unwrap();
            assert!(buffer.is_some());
            assert_eq!(buffer.unwrap().metadata().sequence, i);
        }

        assert_eq!(source.remaining(), 0);

        // Should return None (EOS)
        let buffer = source.produce().unwrap();
        assert!(buffer.is_none());
    }

    #[test]
    fn test_null_source_buffer_size() {
        let mut source = NullSource::new(1).with_buffer_size(1024);

        let buffer = source.produce().unwrap().unwrap();
        assert_eq!(buffer.len(), 1024);
    }

    #[test]
    fn test_null_source_zero_count() {
        let mut source = NullSource::new(0);

        // Should immediately return None
        let buffer = source.produce().unwrap();
        assert!(buffer.is_none());
    }
}
