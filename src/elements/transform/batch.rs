//! Batch and unbatch elements for buffer aggregation.
//!
//! Combine multiple buffers into one, or split one into many.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Element;
use crate::error::Result;
use crate::memory::{CpuSegment, MemorySegment};
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Combines multiple input buffers into a single output buffer.
///
/// Batches can be triggered by count, size, or timeout.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Batch;
/// use std::time::Duration;
///
/// // Batch every 10 buffers
/// let batch = Batch::by_count(10);
///
/// // Batch up to 1KB
/// let batch = Batch::by_size(1024);
///
/// // Batch with timeout
/// let batch = Batch::by_count(10).with_timeout(Duration::from_secs(1));
/// ```
pub struct Batch {
    name: String,
    max_count: Option<usize>,
    max_bytes: Option<usize>,
    timeout: Option<Duration>,
    pending: Vec<Buffer>,
    pending_bytes: usize,
    batch_start: Option<Instant>,
    sequence: u64,
    batches_produced: AtomicU64,
    buffers_received: AtomicU64,
}

impl Batch {
    /// Create a batch element that batches by buffer count.
    pub fn by_count(count: usize) -> Self {
        Self {
            name: "batch".to_string(),
            max_count: Some(count.max(1)),
            max_bytes: None,
            timeout: None,
            pending: Vec::new(),
            pending_bytes: 0,
            batch_start: None,
            sequence: 0,
            batches_produced: AtomicU64::new(0),
            buffers_received: AtomicU64::new(0),
        }
    }

    /// Create a batch element that batches by total size.
    pub fn by_size(max_bytes: usize) -> Self {
        Self {
            name: "batch".to_string(),
            max_count: None,
            max_bytes: Some(max_bytes.max(1)),
            timeout: None,
            pending: Vec::new(),
            pending_bytes: 0,
            batch_start: None,
            sequence: 0,
            batches_produced: AtomicU64::new(0),
            buffers_received: AtomicU64::new(0),
        }
    }

    /// Create with both count and size limits.
    pub fn with_limits(max_count: usize, max_bytes: usize) -> Self {
        Self {
            name: "batch".to_string(),
            max_count: Some(max_count.max(1)),
            max_bytes: Some(max_bytes.max(1)),
            timeout: None,
            pending: Vec::new(),
            pending_bytes: 0,
            batch_start: None,
            sequence: 0,
            batches_produced: AtomicU64::new(0),
            buffers_received: AtomicU64::new(0),
        }
    }

    /// Set a timeout for automatic flushing.
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Check if a batch should be flushed.
    fn should_flush(&self) -> bool {
        if let Some(max) = self.max_count {
            if self.pending.len() >= max {
                return true;
            }
        }

        if let Some(max) = self.max_bytes {
            if self.pending_bytes >= max {
                return true;
            }
        }

        if let Some(timeout) = self.timeout {
            if let Some(start) = self.batch_start {
                if start.elapsed() >= timeout {
                    return true;
                }
            }
        }

        false
    }

    /// Create a batched buffer from pending buffers.
    fn create_batch(&mut self) -> Result<Option<Buffer>> {
        if self.pending.is_empty() {
            return Ok(None);
        }

        // Concatenate all pending buffer data
        let total_len = self.pending_bytes;
        let segment = Arc::new(CpuSegment::new(total_len.max(1))?);

        if total_len > 0 {
            let mut offset = 0;
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                for buf in &self.pending {
                    let data = buf.as_bytes();
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.add(offset), data.len());
                    offset += data.len();
                }
            }
        }

        // Use metadata from first buffer
        let mut metadata = self
            .pending
            .first()
            .map(|b| b.metadata().clone())
            .unwrap_or_default();
        metadata.sequence = self.sequence;
        self.sequence += 1;

        self.pending.clear();
        self.pending_bytes = 0;
        self.batch_start = None;
        self.batches_produced.fetch_add(1, Ordering::Relaxed);

        let handle = MemoryHandle::from_segment_with_len(segment, total_len);
        Ok(Some(Buffer::new(handle, metadata)))
    }

    /// Flush any pending buffers as a batch.
    pub fn flush(&mut self) -> Result<Option<Buffer>> {
        self.create_batch()
    }

    /// Check if timeout has expired and flush if needed.
    pub fn check_timeout(&mut self) -> Result<Option<Buffer>> {
        if let Some(timeout) = self.timeout {
            if let Some(start) = self.batch_start {
                if start.elapsed() >= timeout {
                    return self.flush();
                }
            }
        }
        Ok(None)
    }

    /// Get the number of pending buffers.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get the pending byte count.
    pub fn pending_bytes(&self) -> usize {
        self.pending_bytes
    }

    /// Get statistics.
    pub fn stats(&self) -> BatchStats {
        BatchStats {
            buffers_received: self.buffers_received.load(Ordering::Relaxed),
            batches_produced: self.batches_produced.load(Ordering::Relaxed),
            pending_count: self.pending.len(),
            pending_bytes: self.pending_bytes,
        }
    }
}

impl Element for Batch {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.buffers_received.fetch_add(1, Ordering::Relaxed);

        if self.batch_start.is_none() {
            self.batch_start = Some(Instant::now());
        }

        self.pending_bytes += buffer.len();
        self.pending.push(buffer);

        if self.should_flush() {
            self.create_batch()
        } else {
            Ok(None)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Batch element.
#[derive(Debug, Clone, Copy)]
pub struct BatchStats {
    /// Total buffers received.
    pub buffers_received: u64,
    /// Total batches produced.
    pub batches_produced: u64,
    /// Buffers currently pending.
    pub pending_count: usize,
    /// Bytes currently pending.
    pub pending_bytes: usize,
}

/// Splits a single input buffer into multiple output buffers.
///
/// The opposite of Batch - takes one large buffer and produces many smaller ones.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Unbatch;
///
/// // Split into 100-byte chunks
/// let unbatch = Unbatch::new(100);
/// ```
pub struct Unbatch {
    name: String,
    chunk_size: usize,
    pending_chunks: VecDeque<Buffer>,
    count: AtomicU64,
    chunks_produced: AtomicU64,
}

impl Unbatch {
    /// Create a new unbatch element.
    pub fn new(chunk_size: usize) -> Self {
        Self {
            name: "unbatch".to_string(),
            chunk_size: chunk_size.max(1),
            pending_chunks: VecDeque::new(),
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

    /// Process and return all chunks at once.
    pub fn process_all(&mut self, buffer: Buffer) -> Result<Vec<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);

        let data = buffer.as_bytes();
        let base_metadata = buffer.metadata().clone();
        let mut chunks = Vec::new();
        let mut offset = 0;
        let mut seq = 0u64;

        while offset < data.len() {
            let chunk_len = (data.len() - offset).min(self.chunk_size);
            let segment = Arc::new(CpuSegment::new(chunk_len)?);

            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(data[offset..].as_ptr(), ptr, chunk_len);
            }

            let handle = MemoryHandle::from_segment_with_len(segment, chunk_len);
            let mut meta = base_metadata.clone();
            meta.sequence = seq;
            if let Some(base_offset) = meta.offset {
                meta.offset = Some(base_offset + offset as u64);
            }

            chunks.push(Buffer::new(handle, meta));
            self.chunks_produced.fetch_add(1, Ordering::Relaxed);

            offset += chunk_len;
            seq += 1;
        }

        Ok(chunks)
    }

    /// Get statistics.
    pub fn stats(&self) -> UnbatchStats {
        UnbatchStats {
            buffers_received: self.count.load(Ordering::Relaxed),
            chunks_produced: self.chunks_produced.load(Ordering::Relaxed),
        }
    }
}

impl Element for Unbatch {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // If we have pending chunks, return the next one
        if let Some(chunk) = self.pending_chunks.pop_front() {
            // Put the new buffer's chunks at the back
            let new_chunks = self.process_all(buffer)?;
            self.pending_chunks.extend(new_chunks);
            return Ok(Some(chunk));
        }

        // Process the buffer and return first chunk
        let mut chunks = self.process_all(buffer)?;
        if chunks.is_empty() {
            return Ok(None);
        }

        let first = chunks.remove(0);
        self.pending_chunks.extend(chunks);
        Ok(Some(first))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Unbatch element.
#[derive(Debug, Clone, Copy)]
pub struct UnbatchStats {
    /// Total buffers received.
    pub buffers_received: u64,
    /// Total chunks produced.
    pub chunks_produced: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Metadata;

    fn create_test_buffer(data: &[u8], seq: u64) -> Buffer {
        let segment = Arc::new(CpuSegment::new(data.len().max(1)).unwrap());
        if !data.is_empty() {
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        }
        let handle = MemoryHandle::from_segment_with_len(segment, data.len());
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    // Batch tests

    #[test]
    fn test_batch_by_count() {
        let mut batch = Batch::by_count(3);

        // First two don't produce output
        assert!(
            batch
                .process(create_test_buffer(&[1], 0))
                .unwrap()
                .is_none()
        );
        assert!(
            batch
                .process(create_test_buffer(&[2], 1))
                .unwrap()
                .is_none()
        );

        // Third produces batch
        let result = batch.process(create_test_buffer(&[3], 2)).unwrap();
        assert!(result.is_some());
        let buf = result.unwrap();
        assert_eq!(buf.as_bytes(), &[1, 2, 3]);
    }

    #[test]
    fn test_batch_by_size() {
        let mut batch = Batch::by_size(5);

        // Small buffers accumulate
        assert!(
            batch
                .process(create_test_buffer(&[1, 2], 0))
                .unwrap()
                .is_none()
        );
        assert!(
            batch
                .process(create_test_buffer(&[3, 4], 1))
                .unwrap()
                .is_none()
        );

        // This one triggers flush
        let result = batch.process(create_test_buffer(&[5], 2)).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_bytes(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_batch_flush() {
        let mut batch = Batch::by_count(10);

        batch.process(create_test_buffer(&[1, 2], 0)).unwrap();
        batch.process(create_test_buffer(&[3, 4], 1)).unwrap();

        assert_eq!(batch.pending_count(), 2);

        let result = batch.flush().unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().as_bytes(), &[1, 2, 3, 4]);
        assert_eq!(batch.pending_count(), 0);
    }

    #[test]
    fn test_batch_stats() {
        let mut batch = Batch::by_count(2);

        batch.process(create_test_buffer(&[1], 0)).unwrap();
        batch.process(create_test_buffer(&[2], 1)).unwrap();
        batch.process(create_test_buffer(&[3], 2)).unwrap();
        batch.process(create_test_buffer(&[4], 3)).unwrap();

        let stats = batch.stats();
        assert_eq!(stats.buffers_received, 4);
        assert_eq!(stats.batches_produced, 2);
    }

    #[test]
    fn test_batch_with_limits() {
        let mut batch = Batch::with_limits(10, 3); // 10 count OR 3 bytes

        batch.process(create_test_buffer(&[1, 2], 0)).unwrap();
        let result = batch.process(create_test_buffer(&[3], 1)).unwrap();

        // Size limit triggers before count
        assert!(result.is_some());
    }

    // Unbatch tests

    #[test]
    fn test_unbatch_basic() {
        let mut unbatch = Unbatch::new(2);

        let buffer = create_test_buffer(&[1, 2, 3, 4, 5], 0);
        let chunks = unbatch.process_all(buffer).unwrap();

        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].as_bytes(), &[1, 2]);
        assert_eq!(chunks[1].as_bytes(), &[3, 4]);
        assert_eq!(chunks[2].as_bytes(), &[5]);
    }

    #[test]
    fn test_unbatch_exact() {
        let mut unbatch = Unbatch::new(2);

        let buffer = create_test_buffer(&[1, 2, 3, 4], 0);
        let chunks = unbatch.process_all(buffer).unwrap();

        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].as_bytes(), &[1, 2]);
        assert_eq!(chunks[1].as_bytes(), &[3, 4]);
    }

    #[test]
    fn test_unbatch_element_interface() {
        let mut unbatch = Unbatch::new(2);

        // First call returns first chunk
        let buf = create_test_buffer(&[1, 2, 3, 4], 0);
        let chunk1 = unbatch.process(buf).unwrap().unwrap();
        assert_eq!(chunk1.as_bytes(), &[1, 2]);

        // Subsequent calls return pending chunks before processing new input
        let buf2 = create_test_buffer(&[5, 6], 1);
        let chunk2 = unbatch.process(buf2).unwrap().unwrap();
        assert_eq!(chunk2.as_bytes(), &[3, 4]); // Still from first buffer
    }

    #[test]
    fn test_unbatch_stats() {
        let mut unbatch = Unbatch::new(3);

        unbatch
            .process_all(create_test_buffer(&[1, 2, 3, 4, 5, 6, 7], 0))
            .unwrap();

        let stats = unbatch.stats();
        assert_eq!(stats.buffers_received, 1);
        assert_eq!(stats.chunks_produced, 3); // 3, 3, 1
    }

    // Round-trip test
    #[test]
    fn test_batch_unbatch_roundtrip() {
        let original_data: Vec<u8> = (0..20).collect();

        // Create individual buffers and batch them
        let mut batch = Batch::by_count(10); // Set high so we use flush
        for (i, chunk) in original_data.chunks(4).enumerate() {
            batch.process(create_test_buffer(chunk, i as u64)).unwrap();
        }
        // Flush remaining to get the batch
        let batched = batch.flush().unwrap().unwrap();
        assert_eq!(batched.as_bytes(), &original_data);

        // Unbatch back
        let mut unbatch = Unbatch::new(4);
        let chunks = unbatch.process_all(batched).unwrap();
        assert_eq!(chunks.len(), 5);

        let mut recovered: Vec<u8> = Vec::new();
        for chunk in chunks {
            recovered.extend_from_slice(chunk.as_bytes());
        }
        assert_eq!(recovered, original_data);
    }
}
