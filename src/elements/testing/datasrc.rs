//! DataSrc element for generating buffers from inline data.
//!
//! Similar to GStreamer's dataurisrc, but accepts raw bytes directly.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Source;
use crate::error::Result;
use crate::memory::HeapSegment;
use crate::metadata::Metadata;
use std::sync::Arc;

/// A source that produces buffers from inline data.
///
/// Useful for testing and embedding small amounts of data directly in code.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::DataSrc;
///
/// // From bytes
/// let src = DataSrc::from_bytes(b"hello world");
///
/// // From string
/// let src = DataSrc::from_string("hello world");
///
/// // From vec with chunking
/// let src = DataSrc::from_vec(data).with_chunk_size(1024);
/// ```
pub struct DataSrc {
    name: String,
    data: Vec<u8>,
    position: usize,
    chunk_size: Option<usize>,
    sequence: u64,
    repeat: bool,
    repeat_count: Option<usize>,
    repeats_done: usize,
}

impl DataSrc {
    /// Create a DataSrc from a byte slice.
    pub fn from_bytes(data: &[u8]) -> Self {
        Self {
            name: "datasrc".to_string(),
            data: data.to_vec(),
            position: 0,
            chunk_size: None,
            sequence: 0,
            repeat: false,
            repeat_count: None,
            repeats_done: 0,
        }
    }

    /// Create a DataSrc from a Vec<u8>.
    pub fn from_vec(data: Vec<u8>) -> Self {
        Self {
            name: "datasrc".to_string(),
            data,
            position: 0,
            chunk_size: None,
            sequence: 0,
            repeat: false,
            repeat_count: None,
            repeats_done: 0,
        }
    }

    /// Create a DataSrc from a string.
    pub fn from_string(data: &str) -> Self {
        Self::from_bytes(data.as_bytes())
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the chunk size for splitting data into multiple buffers.
    ///
    /// If not set, all data is returned in a single buffer.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = Some(size);
        self
    }

    /// Enable infinite repeat.
    pub fn repeat(mut self) -> Self {
        self.repeat = true;
        self.repeat_count = None;
        self
    }

    /// Repeat a specific number of times.
    pub fn repeat_n(mut self, count: usize) -> Self {
        self.repeat = true;
        self.repeat_count = Some(count);
        self
    }

    /// Get the total data size.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if the data is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the current position.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Get the number of buffers produced.
    pub fn buffers_produced(&self) -> u64 {
        self.sequence
    }

    /// Reset to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
        self.sequence = 0;
        self.repeats_done = 0;
    }
}

impl Source for DataSrc {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.data.is_empty() {
            return Ok(None);
        }

        // Check if we've finished
        if self.position >= self.data.len() {
            // Check if we should repeat
            if self.repeat {
                if let Some(max_repeats) = self.repeat_count {
                    if self.repeats_done >= max_repeats {
                        return Ok(None);
                    }
                }
                self.position = 0;
                self.repeats_done += 1;
            } else {
                return Ok(None);
            }
        }

        // Determine chunk to return
        let end = match self.chunk_size {
            Some(size) => (self.position + size).min(self.data.len()),
            None => self.data.len(),
        };

        let chunk = &self.data[self.position..end];
        let chunk_len = chunk.len();

        // Create buffer
        let segment = Arc::new(HeapSegment::new(chunk_len)?);
        unsafe {
            use crate::memory::MemorySegment;
            if let Some(ptr) = segment.as_mut_ptr() {
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), ptr, chunk_len);
            }
        }

        let handle = MemoryHandle::from_segment_with_len(segment, chunk_len);
        let metadata = Metadata::from_sequence(self.sequence);

        self.position = end;
        self.sequence += 1;

        Ok(Some(Buffer::new(handle, metadata)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_datasrc_from_bytes() {
        let mut src = DataSrc::from_bytes(b"hello");

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b"hello");

        // Should be EOS
        let buf = src.produce().unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn test_datasrc_from_string() {
        let mut src = DataSrc::from_string("hello world");

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b"hello world");
    }

    #[test]
    fn test_datasrc_chunked() {
        let mut src = DataSrc::from_bytes(b"hello world").with_chunk_size(5);

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b"hello");

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b" worl");

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b"d");

        // Should be EOS
        let buf = src.produce().unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn test_datasrc_repeat() {
        let mut src = DataSrc::from_bytes(b"ab").repeat_n(2);

        // First iteration
        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b"ab");

        // Second iteration (repeat 1)
        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b"ab");

        // Third iteration (repeat 2)
        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b"ab");

        // Should be EOS (repeated 2 times = 3 total)
        let buf = src.produce().unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn test_datasrc_empty() {
        let mut src = DataSrc::from_bytes(b"");

        let buf = src.produce().unwrap();
        assert!(buf.is_none());
    }

    #[test]
    fn test_datasrc_reset() {
        let mut src = DataSrc::from_bytes(b"hello");

        let _ = src.produce().unwrap();
        assert!(src.produce().unwrap().is_none());

        src.reset();

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b"hello");
    }

    #[test]
    fn test_datasrc_sequence() {
        let mut src = DataSrc::from_bytes(b"hello world").with_chunk_size(5);

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 0);

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 1);

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.metadata().sequence, 2);
    }

    #[test]
    fn test_datasrc_with_name() {
        let src = DataSrc::from_bytes(b"test").with_name("my-data");
        assert_eq!(src.name(), "my-data");
    }
}
