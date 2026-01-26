//! Tee element - duplicates buffers to multiple outputs.

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;

/// An element that passes buffers through while allowing inspection.
///
/// Unlike a true multi-output tee (which would need special pipeline support),
/// this Tee element passes the buffer through and tracks statistics.
///
/// For true multi-output fanout, use multiple links from a single source node
/// in the pipeline graph, which the executor handles by cloning buffers.
///
/// This element is useful for:
/// - Counting buffers passing through
/// - Adding inspection points in pipelines
/// - Debugging data flow
///
/// # Example
///
/// ```rust
/// use parallax::elements::Tee;
/// use parallax::element::Element;
/// # use parallax::buffer::{Buffer, MemoryHandle};
/// # use parallax::memory::HeapSegment;
/// # use parallax::metadata::Metadata;
/// # use std::sync::Arc;
///
/// let mut tee = Tee::new();
///
/// // Create and process a buffer
/// # let segment = Arc::new(HeapSegment::new(8).unwrap());
/// # let handle = MemoryHandle::from_segment(segment);
/// # let buffer = Buffer::new(handle, Metadata::from_sequence(0));
///
/// let result = tee.process(buffer).unwrap();
/// assert!(result.is_some());
/// assert_eq!(tee.count(), 1);
/// ```
pub struct Tee {
    name: String,
    /// Number of buffers that have passed through.
    count: u64,
    /// Total bytes that have passed through.
    bytes: u64,
}

impl Tee {
    /// Create a new Tee element.
    pub fn new() -> Self {
        Self {
            name: "tee".to_string(),
            count: 0,
            bytes: 0,
        }
    }

    /// Create a new Tee element with a custom name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            count: 0,
            bytes: 0,
        }
    }

    /// Get the number of buffers that have passed through.
    pub fn count(&self) -> u64 {
        self.count
    }

    /// Get the total bytes that have passed through.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Reset the statistics.
    pub fn reset(&mut self) {
        self.count = 0;
        self.bytes = 0;
    }
}

impl Default for Tee {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Tee {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.count += 1;
        self.bytes += buffer.len() as u64;
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
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    #[test]
    fn test_tee_passes_buffer() {
        let mut tee = Tee::new();

        let segment = Arc::new(HeapSegment::new(64).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));

        let result = tee.process(buffer).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    #[test]
    fn test_tee_tracks_statistics() {
        let mut tee = Tee::new();
        assert_eq!(tee.count(), 0);
        assert_eq!(tee.bytes(), 0);

        // Process a 64-byte buffer
        let segment = Arc::new(HeapSegment::new(64).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));
        tee.process(buffer).unwrap();

        assert_eq!(tee.count(), 1);
        assert_eq!(tee.bytes(), 64);

        // Process another 128-byte buffer
        let segment = Arc::new(HeapSegment::new(128).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::from_sequence(1));
        tee.process(buffer).unwrap();

        assert_eq!(tee.count(), 2);
        assert_eq!(tee.bytes(), 192);
    }

    #[test]
    fn test_tee_reset() {
        let mut tee = Tee::new();

        let segment = Arc::new(HeapSegment::new(64).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));
        tee.process(buffer).unwrap();

        assert_eq!(tee.count(), 1);
        tee.reset();
        assert_eq!(tee.count(), 0);
        assert_eq!(tee.bytes(), 0);
    }

    #[test]
    fn test_tee_custom_name() {
        let tee = Tee::with_name("my_tee");
        assert_eq!(tee.name(), "my_tee");
    }
}
