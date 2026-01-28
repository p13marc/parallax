//! Buffer manipulation elements.
//!
//! Elements for slicing, trimming, and manipulating buffer contents.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Element;
use crate::error::Result;
use crate::memory::{CpuSegment, MemorySegment};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Trims buffers to a maximum size.
///
/// Buffers larger than the maximum are truncated. Smaller buffers pass through unchanged.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::BufferTrim;
///
/// // Limit buffers to 1KB
/// let trim = BufferTrim::new(1024);
/// ```
pub struct BufferTrim {
    name: String,
    max_size: usize,
    count: AtomicU64,
    trimmed_count: AtomicU64,
    bytes_trimmed: AtomicU64,
}

impl BufferTrim {
    /// Create a new buffer trim element.
    pub fn new(max_size: usize) -> Self {
        Self {
            name: "buffer-trim".to_string(),
            max_size,
            count: AtomicU64::new(0),
            trimmed_count: AtomicU64::new(0),
            bytes_trimmed: AtomicU64::new(0),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the max size.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// Set a new max size.
    pub fn set_max_size(&mut self, max_size: usize) {
        self.max_size = max_size;
    }

    /// Get statistics.
    pub fn stats(&self) -> BufferTrimStats {
        BufferTrimStats {
            buffer_count: self.count.load(Ordering::Relaxed),
            trimmed_count: self.trimmed_count.load(Ordering::Relaxed),
            bytes_trimmed: self.bytes_trimmed.load(Ordering::Relaxed),
        }
    }
}

impl Element for BufferTrim {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);

        if buffer.len() <= self.max_size {
            return Ok(Some(buffer));
        }

        // Need to trim - create a new buffer with truncated data
        let bytes_removed = buffer.len() - self.max_size;
        self.trimmed_count.fetch_add(1, Ordering::Relaxed);
        self.bytes_trimmed
            .fetch_add(bytes_removed as u64, Ordering::Relaxed);

        let segment = Arc::new(CpuSegment::new(self.max_size)?);
        let src_data = buffer.as_bytes();

        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            std::ptr::copy_nonoverlapping(src_data.as_ptr(), ptr, self.max_size);
        }

        let handle = MemoryHandle::from_segment_with_len(segment, self.max_size);
        let new_buffer = Buffer::new(handle, buffer.metadata().clone());

        Ok(Some(new_buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for BufferTrim.
#[derive(Debug, Clone, Copy)]
pub struct BufferTrimStats {
    /// Total buffers processed.
    pub buffer_count: u64,
    /// Buffers that were trimmed.
    pub trimmed_count: u64,
    /// Total bytes removed by trimming.
    pub bytes_trimmed: u64,
}

/// Extracts a slice from each buffer.
///
/// Creates a new buffer containing only the specified range of bytes.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::BufferSlice;
///
/// // Extract bytes 10-20 from each buffer
/// let slice = BufferSlice::new(10, 10);
///
/// // Extract from offset to end
/// let slice = BufferSlice::from_offset(10);
/// ```
pub struct BufferSlice {
    name: String,
    offset: usize,
    length: Option<usize>,
    count: AtomicU64,
    skip_short: bool,
}

impl BufferSlice {
    /// Create a new buffer slice element.
    ///
    /// # Arguments
    /// * `offset` - Starting offset in the buffer
    /// * `length` - Number of bytes to extract
    pub fn new(offset: usize, length: usize) -> Self {
        Self {
            name: "buffer-slice".to_string(),
            offset,
            length: Some(length),
            count: AtomicU64::new(0),
            skip_short: false,
        }
    }

    /// Create a slice that takes from offset to end.
    pub fn from_offset(offset: usize) -> Self {
        Self {
            name: "buffer-slice".to_string(),
            offset,
            length: None,
            count: AtomicU64::new(0),
            skip_short: false,
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Skip buffers that are too short (instead of producing empty/partial).
    pub fn skip_short(mut self, skip: bool) -> Self {
        self.skip_short = skip;
        self
    }

    /// Get the offset.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the length (if fixed).
    pub fn length(&self) -> Option<usize> {
        self.length
    }

    /// Get buffer count.
    pub fn buffer_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl Element for BufferSlice {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);

        let buf_len = buffer.len();

        // Handle offset beyond buffer length
        if self.offset >= buf_len {
            if self.skip_short {
                return Ok(None);
            }
            // Return empty buffer (allocate minimum 1 byte)
            let segment = Arc::new(CpuSegment::new(1)?);
            let handle = MemoryHandle::from_segment_with_len(segment, 0);
            return Ok(Some(Buffer::new(handle, buffer.metadata().clone())));
        }

        let available = buf_len - self.offset;
        let slice_len = match self.length {
            Some(len) => {
                if self.skip_short && available < len {
                    return Ok(None);
                }
                len.min(available)
            }
            None => available,
        };

        if slice_len == 0 {
            let segment = Arc::new(CpuSegment::new(1)?); // Minimum 1 byte
            let handle = MemoryHandle::from_segment_with_len(segment, 0);
            return Ok(Some(Buffer::new(handle, buffer.metadata().clone())));
        }

        let segment = Arc::new(CpuSegment::new(slice_len)?);
        let src_data = buffer.as_bytes();

        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            std::ptr::copy_nonoverlapping(src_data[self.offset..].as_ptr(), ptr, slice_len);
        }

        let handle = MemoryHandle::from_segment_with_len(segment, slice_len);
        let mut metadata = buffer.metadata().clone();

        // Update offset metadata if present
        if let Some(off) = metadata.offset {
            metadata.offset = Some(off + self.offset as u64);
        }

        Ok(Some(Buffer::new(handle, metadata)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Pads buffers to a minimum size.
///
/// Buffers smaller than the minimum are padded with a fill byte.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::BufferPad;
///
/// // Pad to at least 1KB with zeros
/// let pad = BufferPad::new(1024, 0);
/// ```
pub struct BufferPad {
    name: String,
    min_size: usize,
    fill_byte: u8,
    count: AtomicU64,
    padded_count: AtomicU64,
}

impl BufferPad {
    /// Create a new buffer pad element.
    pub fn new(min_size: usize, fill_byte: u8) -> Self {
        Self {
            name: "buffer-pad".to_string(),
            min_size,
            fill_byte,
            count: AtomicU64::new(0),
            padded_count: AtomicU64::new(0),
        }
    }

    /// Create with zero fill byte.
    pub fn with_zeros(min_size: usize) -> Self {
        Self::new(min_size, 0)
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the minimum size.
    pub fn min_size(&self) -> usize {
        self.min_size
    }

    /// Get statistics.
    pub fn stats(&self) -> BufferPadStats {
        BufferPadStats {
            buffer_count: self.count.load(Ordering::Relaxed),
            padded_count: self.padded_count.load(Ordering::Relaxed),
        }
    }
}

impl Element for BufferPad {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);

        if buffer.len() >= self.min_size {
            return Ok(Some(buffer));
        }

        self.padded_count.fetch_add(1, Ordering::Relaxed);

        let segment = Arc::new(CpuSegment::new(self.min_size)?);
        let src_data = buffer.as_bytes();

        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            // Copy original data
            std::ptr::copy_nonoverlapping(src_data.as_ptr(), ptr, src_data.len());
            // Fill padding
            std::ptr::write_bytes(
                ptr.add(src_data.len()),
                self.fill_byte,
                self.min_size - src_data.len(),
            );
        }

        let handle = MemoryHandle::from_segment_with_len(segment, self.min_size);
        Ok(Some(Buffer::new(handle, buffer.metadata().clone())))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for BufferPad.
#[derive(Debug, Clone, Copy)]
pub struct BufferPadStats {
    /// Total buffers processed.
    pub buffer_count: u64,
    /// Buffers that were padded.
    pub padded_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Metadata;

    fn create_test_buffer(data: &[u8], seq: u64) -> Buffer {
        let segment = Arc::new(CpuSegment::new(data.len()).unwrap());
        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        let handle = MemoryHandle::from_segment_with_len(segment, data.len());
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    // BufferTrim tests

    #[test]
    fn test_buffer_trim_no_trim() {
        let mut trim = BufferTrim::new(100);
        let buffer = create_test_buffer(&[1, 2, 3, 4, 5], 0);

        let result = trim.process(buffer).unwrap().unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.as_bytes(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_buffer_trim_trims() {
        let mut trim = BufferTrim::new(3);
        let buffer = create_test_buffer(&[1, 2, 3, 4, 5], 0);

        let result = trim.process(buffer).unwrap().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn test_buffer_trim_stats() {
        let mut trim = BufferTrim::new(2);

        trim.process(create_test_buffer(&[1, 2, 3, 4], 0)).unwrap();
        trim.process(create_test_buffer(&[1], 1)).unwrap();
        trim.process(create_test_buffer(&[1, 2, 3], 2)).unwrap();

        let stats = trim.stats();
        assert_eq!(stats.buffer_count, 3);
        assert_eq!(stats.trimmed_count, 2);
        assert_eq!(stats.bytes_trimmed, 3); // 2 from first, 1 from third
    }

    // BufferSlice tests

    #[test]
    fn test_buffer_slice_basic() {
        let mut slice = BufferSlice::new(1, 3);
        let buffer = create_test_buffer(&[1, 2, 3, 4, 5], 0);

        let result = slice.process(buffer).unwrap().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.as_bytes(), &[2, 3, 4]);
    }

    #[test]
    fn test_buffer_slice_from_offset() {
        let mut slice = BufferSlice::from_offset(2);
        let buffer = create_test_buffer(&[1, 2, 3, 4, 5], 0);

        let result = slice.process(buffer).unwrap().unwrap();
        assert_eq!(result.len(), 3);
        assert_eq!(result.as_bytes(), &[3, 4, 5]);
    }

    #[test]
    fn test_buffer_slice_beyond_length() {
        let mut slice = BufferSlice::new(3, 10);
        let buffer = create_test_buffer(&[1, 2, 3, 4, 5], 0);

        let result = slice.process(buffer).unwrap().unwrap();
        assert_eq!(result.len(), 2); // Only 2 bytes available
        assert_eq!(result.as_bytes(), &[4, 5]);
    }

    #[test]
    fn test_buffer_slice_skip_short() {
        let mut slice = BufferSlice::new(0, 10).skip_short(true);
        let buffer = create_test_buffer(&[1, 2, 3], 0);

        let result = slice.process(buffer).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_buffer_slice_offset_beyond() {
        let mut slice = BufferSlice::new(100, 5);
        let buffer = create_test_buffer(&[1, 2, 3], 0);

        let result = slice.process(buffer).unwrap().unwrap();
        assert_eq!(result.len(), 0);
    }

    // BufferPad tests

    #[test]
    fn test_buffer_pad_no_pad() {
        let mut pad = BufferPad::with_zeros(3);
        let buffer = create_test_buffer(&[1, 2, 3, 4, 5], 0);

        let result = pad.process(buffer).unwrap().unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.as_bytes(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_buffer_pad_pads() {
        let mut pad = BufferPad::with_zeros(5);
        let buffer = create_test_buffer(&[1, 2, 3], 0);

        let result = pad.process(buffer).unwrap().unwrap();
        assert_eq!(result.len(), 5);
        assert_eq!(result.as_bytes(), &[1, 2, 3, 0, 0]);
    }

    #[test]
    fn test_buffer_pad_fill_byte() {
        let mut pad = BufferPad::new(5, 0xFF);
        let buffer = create_test_buffer(&[1, 2], 0);

        let result = pad.process(buffer).unwrap().unwrap();
        assert_eq!(result.as_bytes(), &[1, 2, 0xFF, 0xFF, 0xFF]);
    }

    #[test]
    fn test_buffer_pad_stats() {
        let mut pad = BufferPad::with_zeros(3);

        pad.process(create_test_buffer(&[1, 2], 0)).unwrap();
        pad.process(create_test_buffer(&[1, 2, 3, 4], 1)).unwrap();

        let stats = pad.stats();
        assert_eq!(stats.buffer_count, 2);
        assert_eq!(stats.padded_count, 1);
    }
}
