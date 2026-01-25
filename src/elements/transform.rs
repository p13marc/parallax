//! Transform elements for buffer manipulation.
//!
//! Map, FlatMap, and other transformation elements.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Element;
use crate::error::Result;
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A map element that transforms buffer contents.
///
/// Applies a function to each buffer's data, producing a new buffer.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Map;
///
/// // Double each byte value
/// let map = Map::new(|data: &[u8]| {
///     data.iter().map(|b| b.wrapping_mul(2)).collect()
/// });
///
/// // Convert to uppercase (ASCII)
/// let map = Map::new(|data: &[u8]| {
///     data.iter().map(|b| b.to_ascii_uppercase()).collect()
/// });
/// ```
pub struct Map<F>
where
    F: FnMut(&[u8]) -> Vec<u8> + Send,
{
    name: String,
    transform: F,
    count: AtomicU64,
}

impl<F> Map<F>
where
    F: FnMut(&[u8]) -> Vec<u8> + Send,
{
    /// Create a new map element.
    pub fn new(transform: F) -> Self {
        Self {
            name: "map".to_string(),
            transform,
            count: AtomicU64::new(0),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the number of buffers processed.
    pub fn buffer_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl<F> Element for Map<F>
where
    F: FnMut(&[u8]) -> Vec<u8> + Send,
{
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);

        let input = buffer.as_bytes();
        let output = (self.transform)(input);

        if output.is_empty() {
            // Return empty buffer
            let segment = Arc::new(HeapSegment::new(1)?);
            let handle = MemoryHandle::from_segment_with_len(segment, 0);
            return Ok(Some(Buffer::new(handle, buffer.metadata().clone())));
        }

        let segment = Arc::new(HeapSegment::new(output.len())?);
        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            std::ptr::copy_nonoverlapping(output.as_ptr(), ptr, output.len());
        }

        let handle = MemoryHandle::from_segment_with_len(segment, output.len());
        Ok(Some(Buffer::new(handle, buffer.metadata().clone())))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A map element that can optionally filter by returning None.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::FilterMap;
///
/// // Only pass non-empty results
/// let filter_map = FilterMap::new(|data: &[u8]| {
///     if data.is_empty() {
///         None
///     } else {
///         Some(data.to_vec())
///     }
/// });
/// ```
pub struct FilterMap<F>
where
    F: FnMut(&[u8]) -> Option<Vec<u8>> + Send,
{
    name: String,
    transform: F,
    passed: AtomicU64,
    filtered: AtomicU64,
}

impl<F> FilterMap<F>
where
    F: FnMut(&[u8]) -> Option<Vec<u8>> + Send,
{
    /// Create a new filter-map element.
    pub fn new(transform: F) -> Self {
        Self {
            name: "filter-map".to_string(),
            transform,
            passed: AtomicU64::new(0),
            filtered: AtomicU64::new(0),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the number of passed buffers.
    pub fn passed_count(&self) -> u64 {
        self.passed.load(Ordering::Relaxed)
    }

    /// Get the number of filtered buffers.
    pub fn filtered_count(&self) -> u64 {
        self.filtered.load(Ordering::Relaxed)
    }
}

impl<F> Element for FilterMap<F>
where
    F: FnMut(&[u8]) -> Option<Vec<u8>> + Send,
{
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input = buffer.as_bytes();

        match (self.transform)(input) {
            Some(output) => {
                self.passed.fetch_add(1, Ordering::Relaxed);

                let segment = Arc::new(HeapSegment::new(output.len().max(1))?);
                if !output.is_empty() {
                    unsafe {
                        let ptr = segment.as_mut_ptr().unwrap();
                        std::ptr::copy_nonoverlapping(output.as_ptr(), ptr, output.len());
                    }
                }

                let handle = MemoryHandle::from_segment_with_len(segment, output.len());
                Ok(Some(Buffer::new(handle, buffer.metadata().clone())))
            }
            None => {
                self.filtered.fetch_add(1, Ordering::Relaxed);
                Ok(None)
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Splits buffers into fixed-size chunks.
///
/// Each input buffer is split into multiple output buffers of the specified size.
/// The last chunk may be smaller if the input doesn't divide evenly.
///
/// Note: This element buffers data internally and produces multiple outputs per input.
/// Use `process_all` to get all chunks, or iterate with `process` (returns first chunk,
/// call again to get more).
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Chunk;
///
/// let mut chunk = Chunk::new(1024);
///
/// // Process and get all chunks
/// let chunks = chunk.process_all(large_buffer)?;
/// ```
pub struct Chunk {
    name: String,
    chunk_size: usize,
    pending: Vec<u8>,
    pending_metadata: Option<Metadata>,
    sequence_offset: u64,
    count: AtomicU64,
    chunks_produced: AtomicU64,
}

impl Chunk {
    /// Create a new chunk element.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            name: "chunk".to_string(),
            chunk_size: chunk_size.max(1),
            pending: Vec::new(),
            pending_metadata: None,
            sequence_offset: 0,
            count: AtomicU64::new(0),
            chunks_produced: AtomicU64::new(0),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the chunk size.
    pub fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    /// Get the number of input buffers processed.
    pub fn input_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the number of chunks produced.
    pub fn chunks_produced(&self) -> u64 {
        self.chunks_produced.load(Ordering::Relaxed)
    }

    /// Process a buffer and return all resulting chunks.
    pub fn process_all(&mut self, buffer: Buffer) -> Result<Vec<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);

        let data = buffer.as_bytes();
        let base_metadata = buffer.metadata().clone();

        // Add to pending
        self.pending.extend_from_slice(data);
        if self.pending_metadata.is_none() {
            self.pending_metadata = Some(base_metadata);
        }

        let mut chunks = Vec::new();

        while self.pending.len() >= self.chunk_size {
            let chunk_data: Vec<u8> = self.pending.drain(..self.chunk_size).collect();

            let segment = Arc::new(HeapSegment::new(self.chunk_size)?);
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(chunk_data.as_ptr(), ptr, self.chunk_size);
            }

            let handle = MemoryHandle::from_segment_with_len(segment, self.chunk_size);
            let mut meta = self.pending_metadata.clone().unwrap_or_else(Metadata::new);
            meta.sequence = self.sequence_offset;
            self.sequence_offset += 1;

            chunks.push(Buffer::new(handle, meta));
            self.chunks_produced.fetch_add(1, Ordering::Relaxed);
        }

        Ok(chunks)
    }

    /// Flush any remaining pending data as a final chunk.
    pub fn flush(&mut self) -> Result<Option<Buffer>> {
        if self.pending.is_empty() {
            return Ok(None);
        }

        let len = self.pending.len();
        let segment = Arc::new(HeapSegment::new(len)?);
        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            std::ptr::copy_nonoverlapping(self.pending.as_ptr(), ptr, len);
        }

        let handle = MemoryHandle::from_segment_with_len(segment, len);
        let mut meta = self.pending_metadata.take().unwrap_or_else(Metadata::new);
        meta.sequence = self.sequence_offset;
        self.sequence_offset += 1;

        self.pending.clear();
        self.chunks_produced.fetch_add(1, Ordering::Relaxed);

        Ok(Some(Buffer::new(handle, meta)))
    }
}

impl Element for Chunk {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // For Element trait, we process and return first chunk if available
        let mut chunks = self.process_all(buffer)?;
        if chunks.is_empty() {
            Ok(None)
        } else {
            Ok(Some(chunks.remove(0)))
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A flat-map element that produces multiple outputs per input.
///
/// Applies a function to each buffer that returns a vector of output buffers.
/// Useful for operations that expand one input into multiple outputs.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::FlatMap;
///
/// // Split each buffer into lines
/// let flat_map = FlatMap::new(|data: &[u8]| {
///     data.split(|&b| b == b'\n')
///         .filter(|line| !line.is_empty())
///         .map(|line| line.to_vec())
///         .collect()
/// });
/// ```
pub struct FlatMap<F>
where
    F: FnMut(&[u8]) -> Vec<Vec<u8>> + Send,
{
    name: String,
    transform: F,
    input_count: AtomicU64,
    output_count: AtomicU64,
    pending: Vec<Vec<u8>>,
    pending_metadata: Option<Metadata>,
    sequence_offset: u64,
}

impl<F> FlatMap<F>
where
    F: FnMut(&[u8]) -> Vec<Vec<u8>> + Send,
{
    /// Create a new flat-map element.
    pub fn new(transform: F) -> Self {
        Self {
            name: "flat-map".to_string(),
            transform,
            input_count: AtomicU64::new(0),
            output_count: AtomicU64::new(0),
            pending: Vec::new(),
            pending_metadata: None,
            sequence_offset: 0,
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the number of input buffers processed.
    pub fn input_count(&self) -> u64 {
        self.input_count.load(Ordering::Relaxed)
    }

    /// Get the number of output buffers produced.
    pub fn output_count(&self) -> u64 {
        self.output_count.load(Ordering::Relaxed)
    }

    /// Process a buffer and return all resulting outputs.
    pub fn process_all(&mut self, buffer: Buffer) -> Result<Vec<Buffer>> {
        self.input_count.fetch_add(1, Ordering::Relaxed);

        let input = buffer.as_bytes();
        let base_metadata = buffer.metadata().clone();
        let outputs = (self.transform)(input);

        let mut buffers = Vec::with_capacity(outputs.len());

        for output in outputs {
            let segment = Arc::new(HeapSegment::new(output.len().max(1))?);
            if !output.is_empty() {
                unsafe {
                    let ptr = segment.as_mut_ptr().unwrap();
                    std::ptr::copy_nonoverlapping(output.as_ptr(), ptr, output.len());
                }
            }

            let handle = MemoryHandle::from_segment_with_len(segment, output.len());
            let mut meta = base_metadata.clone();
            meta.sequence = self.sequence_offset;
            self.sequence_offset += 1;

            buffers.push(Buffer::new(handle, meta));
            self.output_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(buffers)
    }

    /// Get the next pending output buffer, if any.
    pub fn next_pending(&mut self) -> Result<Option<Buffer>> {
        if self.pending.is_empty() {
            return Ok(None);
        }

        let output = self.pending.remove(0);
        let segment = Arc::new(HeapSegment::new(output.len().max(1))?);
        if !output.is_empty() {
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(output.as_ptr(), ptr, output.len());
            }
        }

        let handle = MemoryHandle::from_segment_with_len(segment, output.len());
        let mut meta = self.pending_metadata.clone().unwrap_or_else(Metadata::new);
        meta.sequence = self.sequence_offset;
        self.sequence_offset += 1;

        self.output_count.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Buffer::new(handle, meta)))
    }
}

impl<F> Element for FlatMap<F>
where
    F: FnMut(&[u8]) -> Vec<Vec<u8>> + Send,
{
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // If we have pending outputs, return them first
        if !self.pending.is_empty() {
            return self.next_pending();
        }

        self.input_count.fetch_add(1, Ordering::Relaxed);

        let input = buffer.as_bytes();
        self.pending_metadata = Some(buffer.metadata().clone());
        let outputs = (self.transform)(input);

        if outputs.is_empty() {
            return Ok(None);
        }

        // Store remaining outputs for later calls
        self.pending = outputs;

        // Return the first one
        self.next_pending()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_buffer(data: &[u8], seq: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(data.len().max(1)).unwrap());
        if !data.is_empty() {
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        }
        let handle = MemoryHandle::from_segment_with_len(segment, data.len());
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    // Map tests

    #[test]
    fn test_map_double() {
        let mut map = Map::new(|data: &[u8]| data.iter().map(|b| b.wrapping_mul(2)).collect());

        let buffer = create_test_buffer(&[1, 2, 3, 4, 5], 0);
        let result = map.process(buffer).unwrap().unwrap();

        assert_eq!(result.as_bytes(), &[2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_map_uppercase() {
        let mut map = Map::new(|data: &[u8]| data.iter().map(|b| b.to_ascii_uppercase()).collect());

        let buffer = create_test_buffer(b"hello", 0);
        let result = map.process(buffer).unwrap().unwrap();

        assert_eq!(result.as_bytes(), b"HELLO");
    }

    #[test]
    fn test_map_preserves_metadata() {
        let mut map = Map::new(|data: &[u8]| data.to_vec());

        let buffer = create_test_buffer(&[1, 2, 3], 42);
        let result = map.process(buffer).unwrap().unwrap();

        assert_eq!(result.metadata().sequence, 42);
    }

    #[test]
    fn test_map_count() {
        let mut map = Map::new(|data: &[u8]| data.to_vec());

        map.process(create_test_buffer(&[1], 0)).unwrap();
        map.process(create_test_buffer(&[2], 1)).unwrap();

        assert_eq!(map.buffer_count(), 2);
    }

    // FilterMap tests

    #[test]
    fn test_filter_map_pass() {
        let mut fm = FilterMap::new(|data: &[u8]| {
            if data.len() > 2 {
                Some(data.to_vec())
            } else {
                None
            }
        });

        let short = create_test_buffer(&[1, 2], 0);
        let long = create_test_buffer(&[1, 2, 3, 4], 1);

        assert!(fm.process(short).unwrap().is_none());
        assert!(fm.process(long).unwrap().is_some());

        assert_eq!(fm.passed_count(), 1);
        assert_eq!(fm.filtered_count(), 1);
    }

    // Chunk tests

    #[test]
    fn test_chunk_exact() {
        let mut chunk = Chunk::new(3);

        let buffer = create_test_buffer(&[1, 2, 3, 4, 5, 6], 0);
        let chunks = chunk.process_all(buffer).unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].as_bytes(), &[1, 2, 3]);
        assert_eq!(chunks[1].as_bytes(), &[4, 5, 6]);
    }

    #[test]
    fn test_chunk_with_remainder() {
        let mut chunk = Chunk::new(3);

        let buffer = create_test_buffer(&[1, 2, 3, 4, 5], 0);
        let chunks = chunk.process_all(buffer).unwrap();

        assert_eq!(chunks.len(), 1); // Only one complete chunk
        assert_eq!(chunks[0].as_bytes(), &[1, 2, 3]);

        // Flush remainder
        let remainder = chunk.flush().unwrap().unwrap();
        assert_eq!(remainder.as_bytes(), &[4, 5]);
    }

    #[test]
    fn test_chunk_accumulates() {
        let mut chunk = Chunk::new(5);

        let chunks1 = chunk.process_all(create_test_buffer(&[1, 2], 0)).unwrap();
        assert!(chunks1.is_empty());

        let chunks2 = chunk
            .process_all(create_test_buffer(&[3, 4, 5], 1))
            .unwrap();
        assert_eq!(chunks2.len(), 1);
        assert_eq!(chunks2[0].as_bytes(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_chunk_stats() {
        let mut chunk = Chunk::new(2);

        chunk
            .process_all(create_test_buffer(&[1, 2, 3, 4, 5], 0))
            .unwrap();

        assert_eq!(chunk.input_count(), 1);
        assert_eq!(chunk.chunks_produced(), 2);
    }
}
