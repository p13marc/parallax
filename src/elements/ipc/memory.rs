//! Memory-based source and sink elements.
//!
//! Read from and write to in-memory buffers.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Sink, Source};
use crate::error::Result;
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::sync::{Arc, Mutex};

/// A source that reads from a memory buffer/slice.
///
/// Produces buffers from data stored in memory. Useful for testing
/// or when data is already loaded in memory.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::MemorySrc;
///
/// let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
/// let mut src = MemorySrc::new(data)
///     .with_chunk_size(4);
///
/// // Produces two 4-byte buffers
/// ```
pub struct MemorySrc {
    name: String,
    data: Vec<u8>,
    position: usize,
    chunk_size: usize,
    sequence: u64,
}

impl MemorySrc {
    /// Create a new memory source from data.
    pub fn new(data: impl Into<Vec<u8>>) -> Self {
        Self {
            name: "memorysrc".to_string(),
            data: data.into(),
            position: 0,
            chunk_size: 4096,
            sequence: 0,
        }
    }

    /// Create from a slice (copies the data).
    pub fn from_slice(data: &[u8]) -> Self {
        Self::new(data.to_vec())
    }

    /// Set the chunk size for produced buffers.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(1);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
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

    /// Get remaining bytes.
    pub fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.position)
    }

    /// Reset to the beginning.
    pub fn reset(&mut self) {
        self.position = 0;
        self.sequence = 0;
    }
}

impl Source for MemorySrc {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.position >= self.data.len() {
            return Ok(None);
        }

        let remaining = self.data.len() - self.position;
        let chunk_len = remaining.min(self.chunk_size);

        let segment = Arc::new(HeapSegment::new(chunk_len)?);
        let ptr = segment.as_mut_ptr().unwrap();

        // Copy data to segment
        unsafe {
            std::ptr::copy_nonoverlapping(self.data[self.position..].as_ptr(), ptr, chunk_len);
        }

        let handle = MemoryHandle::from_segment_with_len(segment, chunk_len);
        let metadata = Metadata::from_sequence(self.sequence);

        self.position += chunk_len;
        self.sequence += 1;

        Ok(Some(Buffer::new(handle, metadata)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A sink that writes to an in-memory buffer.
///
/// Collects all incoming buffers into a single contiguous memory buffer.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::MemorySink;
///
/// let mut sink = MemorySink::new();
///
/// // After processing...
/// let data = sink.take_data();
/// ```
pub struct MemorySink {
    name: String,
    data: Vec<u8>,
    max_size: Option<usize>,
    buffer_count: u64,
}

impl MemorySink {
    /// Create a new memory sink.
    pub fn new() -> Self {
        Self {
            name: "memorysink".to_string(),
            data: Vec::new(),
            max_size: None,
            buffer_count: 0,
        }
    }

    /// Create with pre-allocated capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            name: "memorysink".to_string(),
            data: Vec::with_capacity(capacity),
            max_size: None,
            buffer_count: 0,
        }
    }

    /// Set a maximum size limit.
    pub fn with_max_size(mut self, max_size: usize) -> Self {
        self.max_size = Some(max_size);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get a reference to the collected data.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Take ownership of the collected data.
    pub fn take_data(&mut self) -> Vec<u8> {
        self.buffer_count = 0;
        std::mem::take(&mut self.data)
    }

    /// Get the current size.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get the number of buffers received.
    pub fn buffer_count(&self) -> u64 {
        self.buffer_count
    }

    /// Clear the collected data.
    pub fn clear(&mut self) {
        self.data.clear();
        self.buffer_count = 0;
    }

    /// Get statistics.
    pub fn stats(&self) -> MemorySinkStats {
        MemorySinkStats {
            buffer_count: self.buffer_count,
            byte_count: self.data.len() as u64,
        }
    }
}

impl Default for MemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for MemorySink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let bytes = buffer.as_bytes();

        // Check max size limit
        if let Some(max) = self.max_size {
            let available = max.saturating_sub(self.data.len());
            if available == 0 {
                // Already at max, drop buffer
                return Ok(());
            }
            // Only take what fits
            let to_take = bytes.len().min(available);
            self.data.extend_from_slice(&bytes[..to_take]);
        } else {
            self.data.extend_from_slice(bytes);
        }

        self.buffer_count += 1;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for MemorySink.
#[derive(Debug, Clone, Copy)]
pub struct MemorySinkStats {
    /// Number of buffers received.
    pub buffer_count: u64,
    /// Total bytes collected.
    pub byte_count: u64,
}

/// Thread-safe version of MemorySink for concurrent access.
pub struct SharedMemorySink {
    inner: Arc<Mutex<MemorySink>>,
}

impl SharedMemorySink {
    /// Create a new shared memory sink.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(MemorySink::new())),
        }
    }

    /// Get a clone of the inner Arc for sharing.
    pub fn handle(&self) -> Arc<Mutex<MemorySink>> {
        Arc::clone(&self.inner)
    }

    /// Get the collected data (cloned).
    pub fn data(&self) -> Vec<u8> {
        self.inner.lock().unwrap().data().to_vec()
    }

    /// Get statistics.
    pub fn stats(&self) -> MemorySinkStats {
        self.inner.lock().unwrap().stats()
    }
}

impl Default for SharedMemorySink {
    fn default() -> Self {
        Self::new()
    }
}

impl Sink for SharedMemorySink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        self.inner.lock().unwrap().consume(buffer)
    }

    fn name(&self) -> &str {
        "shared-memorysink"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memorysrc_basic() {
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut src = MemorySrc::new(data.clone()).with_chunk_size(4);

        let mut collected = Vec::new();
        while let Some(buf) = src.produce().unwrap() {
            collected.extend_from_slice(buf.as_bytes());
        }

        assert_eq!(collected, data);
    }

    #[test]
    fn test_memorysrc_sequence() {
        let data = vec![0u8; 100];
        let mut src = MemorySrc::new(data).with_chunk_size(25);

        let buf1 = src.produce().unwrap().unwrap();
        let buf2 = src.produce().unwrap().unwrap();
        let buf3 = src.produce().unwrap().unwrap();
        let buf4 = src.produce().unwrap().unwrap();
        let eos = src.produce().unwrap();

        assert_eq!(buf1.metadata().sequence, 0);
        assert_eq!(buf2.metadata().sequence, 1);
        assert_eq!(buf3.metadata().sequence, 2);
        assert_eq!(buf4.metadata().sequence, 3);
        assert!(eos.is_none());
    }

    #[test]
    fn test_memorysrc_reset() {
        let data = vec![1u8, 2, 3, 4];
        let mut src = MemorySrc::new(data).with_chunk_size(2);

        src.produce().unwrap();
        src.produce().unwrap();
        assert!(src.produce().unwrap().is_none());

        src.reset();
        assert_eq!(src.position(), 0);
        assert!(src.produce().unwrap().is_some());
    }

    #[test]
    fn test_memorysrc_remaining() {
        let data = vec![0u8; 100];
        let mut src = MemorySrc::new(data).with_chunk_size(30);

        assert_eq!(src.remaining(), 100);
        src.produce().unwrap();
        assert_eq!(src.remaining(), 70);
    }

    #[test]
    fn test_memorysink_basic() {
        let mut sink = MemorySink::new();

        let segment = Arc::new(HeapSegment::new(4).unwrap());
        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            ptr.copy_from_nonoverlapping([1u8, 2, 3, 4].as_ptr(), 4);
        }
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::new());

        sink.consume(buffer).unwrap();

        assert_eq!(sink.data(), &[1, 2, 3, 4]);
        assert_eq!(sink.buffer_count(), 1);
    }

    #[test]
    fn test_memorysink_max_size() {
        let mut sink = MemorySink::new().with_max_size(5);

        for i in 0..3 {
            let segment = Arc::new(HeapSegment::new(3).unwrap());
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                ptr.copy_from_nonoverlapping([i, i, i].as_ptr(), 3);
            }
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::new(handle, Metadata::new());
            sink.consume(buffer).unwrap();
        }

        // Only first 5 bytes should be stored
        assert_eq!(sink.len(), 5);
    }

    #[test]
    fn test_memorysink_take_data() {
        let mut sink = MemorySink::new();

        let segment = Arc::new(HeapSegment::new(4).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::new());
        sink.consume(buffer).unwrap();

        let data = sink.take_data();
        assert_eq!(data.len(), 4);
        assert!(sink.is_empty());
        assert_eq!(sink.buffer_count(), 0);
    }

    #[test]
    fn test_memorysink_clear() {
        let mut sink = MemorySink::new();

        let segment = Arc::new(HeapSegment::new(4).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::new());
        sink.consume(buffer).unwrap();

        sink.clear();
        assert!(sink.is_empty());
    }

    #[test]
    fn test_shared_memorysink() {
        let mut sink = SharedMemorySink::new();

        let segment = Arc::new(HeapSegment::new(4).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::new());
        sink.consume(buffer).unwrap();

        let stats = sink.stats();
        assert_eq!(stats.buffer_count, 1);
    }

    #[test]
    fn test_roundtrip() {
        let original = vec![1u8, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let mut src = MemorySrc::new(original.clone()).with_chunk_size(3);
        let mut sink = MemorySink::new();

        while let Some(buf) = src.produce().unwrap() {
            sink.consume(buf).unwrap();
        }

        assert_eq!(sink.data(), &original);
    }
}
