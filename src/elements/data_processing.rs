//! Data processing elements for Tier 5.
//!
//! - [`DuplicateFilter`]: Remove duplicate buffers
//! - [`RangeFilter`]: Filter by value/size range
//! - [`RegexFilter`]: Filter by regex pattern
//! - [`MetadataExtract`]: Extract metadata to sideband
//! - [`BufferSplit`]: Split buffer at delimiter
//! - [`BufferJoin`]: Join buffers with delimiter
//! - [`BufferConcat`]: Concatenate buffer contents

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Element;
use crate::error::Result;
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A filter that removes duplicate buffers based on content hash.
///
/// Uses a hash set to track seen buffer contents and drops duplicates.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::DuplicateFilter;
///
/// let mut filter = DuplicateFilter::new();
/// // First occurrence passes, duplicates are dropped
/// ```
pub struct DuplicateFilter {
    name: String,
    seen: HashSet<u64>,
    max_entries: usize,
    passed: AtomicU64,
    dropped: AtomicU64,
}

impl DuplicateFilter {
    /// Create a new duplicate filter with default capacity.
    pub fn new() -> Self {
        Self {
            name: "duplicate-filter".to_string(),
            seen: HashSet::new(),
            max_entries: 100_000,
            passed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Set maximum number of entries to track.
    ///
    /// When exceeded, the oldest entries are cleared.
    pub fn with_max_entries(mut self, max: usize) -> Self {
        self.max_entries = max;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> DuplicateFilterStats {
        DuplicateFilterStats {
            passed: self.passed.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
            tracked: self.seen.len(),
        }
    }

    /// Clear the tracking set.
    pub fn clear(&mut self) {
        self.seen.clear();
    }

    fn hash_buffer(data: &[u8]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        data.hash(&mut hasher);
        hasher.finish()
    }
}

impl Default for DuplicateFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for DuplicateFilter {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let hash = Self::hash_buffer(buffer.as_bytes());

        if self.seen.contains(&hash) {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return Ok(None);
        }

        // Clear if at capacity
        if self.seen.len() >= self.max_entries {
            self.seen.clear();
        }

        self.seen.insert(hash);
        self.passed.fetch_add(1, Ordering::Relaxed);
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for duplicate filter.
#[derive(Debug, Clone, Copy)]
pub struct DuplicateFilterStats {
    /// Buffers that passed (unique).
    pub passed: u64,
    /// Buffers that were dropped (duplicates).
    pub dropped: u64,
    /// Number of entries being tracked.
    pub tracked: usize,
}

/// A filter that passes buffers based on size range.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RangeFilter;
///
/// // Only pass buffers between 100 and 1000 bytes
/// let filter = RangeFilter::by_size(100, 1000);
/// ```
pub struct RangeFilter {
    name: String,
    min_size: Option<usize>,
    max_size: Option<usize>,
    min_sequence: Option<u64>,
    max_sequence: Option<u64>,
    passed: AtomicU64,
    dropped: AtomicU64,
}

impl RangeFilter {
    /// Create a new range filter with no constraints.
    pub fn new() -> Self {
        Self {
            name: "range-filter".to_string(),
            min_size: None,
            max_size: None,
            min_sequence: None,
            max_sequence: None,
            passed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Create a filter by size range.
    pub fn by_size(min: usize, max: usize) -> Self {
        Self {
            name: "range-filter".to_string(),
            min_size: Some(min),
            max_size: Some(max),
            min_sequence: None,
            max_sequence: None,
            passed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Create a filter by sequence range.
    pub fn by_sequence(min: u64, max: u64) -> Self {
        Self {
            name: "range-filter".to_string(),
            min_size: None,
            max_size: None,
            min_sequence: Some(min),
            max_sequence: Some(max),
            passed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Set minimum size.
    pub fn with_min_size(mut self, min: usize) -> Self {
        self.min_size = Some(min);
        self
    }

    /// Set maximum size.
    pub fn with_max_size(mut self, max: usize) -> Self {
        self.max_size = Some(max);
        self
    }

    /// Set minimum sequence.
    pub fn with_min_sequence(mut self, min: u64) -> Self {
        self.min_sequence = Some(min);
        self
    }

    /// Set maximum sequence.
    pub fn with_max_sequence(mut self, max: u64) -> Self {
        self.max_sequence = Some(max);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> RangeFilterStats {
        RangeFilterStats {
            passed: self.passed.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}

impl Default for RangeFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RangeFilter {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let size = buffer.len();
        let seq = buffer.metadata().sequence;

        // Check size constraints
        if let Some(min) = self.min_size {
            if size < min {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
        }
        if let Some(max) = self.max_size {
            if size > max {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
        }

        // Check sequence constraints
        if let Some(min) = self.min_sequence {
            if seq < min {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
        }
        if let Some(max) = self.max_sequence {
            if seq > max {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return Ok(None);
            }
        }

        self.passed.fetch_add(1, Ordering::Relaxed);
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for range filter.
#[derive(Debug, Clone, Copy)]
pub struct RangeFilterStats {
    /// Buffers that passed.
    pub passed: u64,
    /// Buffers that were dropped.
    pub dropped: u64,
}

/// A filter that matches buffer content against a regex pattern.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RegexFilter;
///
/// // Only pass buffers containing "error" or "Error"
/// let filter = RegexFilter::new(r"(?i)error")?;
/// ```
pub struct RegexFilter {
    name: String,
    pattern: regex::Regex,
    invert: bool,
    passed: AtomicU64,
    dropped: AtomicU64,
}

impl RegexFilter {
    /// Create a new regex filter.
    pub fn new(pattern: &str) -> Result<Self> {
        let regex = regex::Regex::new(pattern)
            .map_err(|e| crate::error::Error::Config(format!("invalid regex: {}", e)))?;

        Ok(Self {
            name: "regex-filter".to_string(),
            pattern: regex,
            invert: false,
            passed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        })
    }

    /// Invert the match (pass buffers that don't match).
    pub fn inverted(mut self) -> Self {
        self.invert = true;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the pattern string.
    pub fn pattern(&self) -> &str {
        self.pattern.as_str()
    }

    /// Get statistics.
    pub fn stats(&self) -> RegexFilterStats {
        RegexFilterStats {
            passed: self.passed.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}

impl Element for RegexFilter {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let data = buffer.as_bytes();

        // Try to interpret as UTF-8, fallback to lossy
        let text = String::from_utf8_lossy(data);
        let matches = self.pattern.is_match(&text);

        let pass = if self.invert { !matches } else { matches };

        if pass {
            self.passed.fetch_add(1, Ordering::Relaxed);
            Ok(Some(buffer))
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for regex filter.
#[derive(Debug, Clone, Copy)]
pub struct RegexFilterStats {
    /// Buffers that passed.
    pub passed: u64,
    /// Buffers that were dropped.
    pub dropped: u64,
}

/// Extracts metadata from buffers and stores in a sideband channel.
///
/// The buffer passes through unchanged while metadata is sent to
/// a separate receiver.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::MetadataExtract;
///
/// let (extractor, receiver) = MetadataExtract::new();
/// // Use extractor in pipeline, receive metadata from receiver
/// ```
pub struct MetadataExtract {
    name: String,
    sender: kanal::Sender<ExtractedMetadata>,
    count: AtomicU64,
}

/// Extracted metadata from a buffer.
#[derive(Debug, Clone)]
pub struct ExtractedMetadata {
    /// The buffer's metadata.
    pub metadata: Metadata,
    /// The buffer size.
    pub size: usize,
}

impl MetadataExtract {
    /// Create a new metadata extractor and its receiver.
    pub fn new() -> (Self, kanal::Receiver<ExtractedMetadata>) {
        let (tx, rx) = kanal::unbounded();
        (
            Self {
                name: "metadata-extract".to_string(),
                sender: tx,
                count: AtomicU64::new(0),
            },
            rx,
        )
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the number of buffers processed.
    pub fn count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }
}

impl Element for MetadataExtract {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let extracted = ExtractedMetadata {
            metadata: buffer.metadata().clone(),
            size: buffer.len(),
        };

        // Send to sideband (ignore errors if receiver dropped)
        let _ = self.sender.send(extracted);
        self.count.fetch_add(1, Ordering::Relaxed);

        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Splits a buffer at delimiter boundaries.
///
/// Each occurrence of the delimiter causes a split, producing multiple
/// output buffers.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::BufferSplit;
///
/// // Split on newlines
/// let splitter = BufferSplit::new(b"\n");
/// ```
pub struct BufferSplit {
    name: String,
    delimiter: Vec<u8>,
    keep_delimiter: bool,
    pending: Vec<Vec<u8>>,
    pending_metadata: Option<Metadata>,
    sequence_offset: u64,
    input_count: AtomicU64,
    output_count: AtomicU64,
}

impl BufferSplit {
    /// Create a new buffer splitter.
    pub fn new(delimiter: impl Into<Vec<u8>>) -> Self {
        Self {
            name: "buffer-split".to_string(),
            delimiter: delimiter.into(),
            keep_delimiter: false,
            pending: Vec::new(),
            pending_metadata: None,
            sequence_offset: 0,
            input_count: AtomicU64::new(0),
            output_count: AtomicU64::new(0),
        }
    }

    /// Keep the delimiter at the end of each split part.
    pub fn keep_delimiter(mut self) -> Self {
        self.keep_delimiter = true;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the delimiter.
    pub fn delimiter(&self) -> &[u8] {
        &self.delimiter
    }

    /// Get statistics.
    pub fn stats(&self) -> BufferSplitStats {
        BufferSplitStats {
            input_count: self.input_count.load(Ordering::Relaxed),
            output_count: self.output_count.load(Ordering::Relaxed),
        }
    }

    /// Process a buffer and return all splits.
    pub fn process_all(&mut self, buffer: Buffer) -> Result<Vec<Buffer>> {
        self.input_count.fetch_add(1, Ordering::Relaxed);

        let data = buffer.as_bytes();
        let base_metadata = buffer.metadata().clone();
        let parts = self.split_data(data);

        let mut buffers = Vec::with_capacity(parts.len());

        for part in parts {
            if part.is_empty() {
                continue;
            }

            let segment = Arc::new(HeapSegment::new(part.len())?);
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(part.as_ptr(), ptr, part.len());
            }

            let handle = MemoryHandle::from_segment_with_len(segment, part.len());
            let mut meta = base_metadata.clone();
            meta.sequence = self.sequence_offset;
            self.sequence_offset += 1;

            buffers.push(Buffer::new(handle, meta));
            self.output_count.fetch_add(1, Ordering::Relaxed);
        }

        Ok(buffers)
    }

    fn split_data(&self, data: &[u8]) -> Vec<Vec<u8>> {
        if self.delimiter.is_empty() {
            return vec![data.to_vec()];
        }

        let mut parts = Vec::new();
        let mut start = 0;
        let delim_len = self.delimiter.len();

        while start < data.len() {
            if let Some(pos) = self.find_delimiter(&data[start..]) {
                let end = start + pos;
                if self.keep_delimiter {
                    parts.push(data[start..end + delim_len].to_vec());
                } else {
                    parts.push(data[start..end].to_vec());
                }
                start = end + delim_len;
            } else {
                parts.push(data[start..].to_vec());
                break;
            }
        }

        parts
    }

    fn find_delimiter(&self, data: &[u8]) -> Option<usize> {
        data.windows(self.delimiter.len())
            .position(|w| w == self.delimiter.as_slice())
    }
}

impl Element for BufferSplit {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // If we have pending outputs, return them first
        if !self.pending.is_empty() {
            let part = self.pending.remove(0);
            if part.is_empty() {
                return Ok(None);
            }

            let segment = Arc::new(HeapSegment::new(part.len())?);
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(part.as_ptr(), ptr, part.len());
            }

            let handle = MemoryHandle::from_segment_with_len(segment, part.len());
            let mut meta = self.pending_metadata.clone().unwrap_or_else(Metadata::new);
            meta.sequence = self.sequence_offset;
            self.sequence_offset += 1;

            self.output_count.fetch_add(1, Ordering::Relaxed);
            return Ok(Some(Buffer::new(handle, meta)));
        }

        self.input_count.fetch_add(1, Ordering::Relaxed);

        let data = buffer.as_bytes();
        self.pending_metadata = Some(buffer.metadata().clone());
        self.pending = self.split_data(data);

        // Return first part
        if self.pending.is_empty() {
            return Ok(None);
        }

        let part = self.pending.remove(0);
        if part.is_empty() && self.pending.is_empty() {
            return Ok(None);
        }

        let segment = Arc::new(HeapSegment::new(part.len().max(1))?);
        if !part.is_empty() {
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(part.as_ptr(), ptr, part.len());
            }
        }

        let handle = MemoryHandle::from_segment_with_len(segment, part.len());
        let mut meta = self.pending_metadata.clone().unwrap_or_else(Metadata::new);
        meta.sequence = self.sequence_offset;
        self.sequence_offset += 1;

        self.output_count.fetch_add(1, Ordering::Relaxed);
        Ok(Some(Buffer::new(handle, meta)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for buffer split.
#[derive(Debug, Clone, Copy)]
pub struct BufferSplitStats {
    /// Number of input buffers processed.
    pub input_count: u64,
    /// Number of output buffers produced.
    pub output_count: u64,
}

/// Joins multiple buffers with a delimiter.
///
/// Accumulates buffers and joins them with the specified delimiter
/// when flushed or when a certain count is reached.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::BufferJoin;
///
/// // Join buffers with newlines
/// let joiner = BufferJoin::new(b"\n").with_count(10);
/// ```
pub struct BufferJoin {
    name: String,
    delimiter: Vec<u8>,
    max_count: Option<usize>,
    pending: Vec<Vec<u8>>,
    pending_size: usize,
    input_count: AtomicU64,
    output_count: AtomicU64,
}

impl BufferJoin {
    /// Create a new buffer joiner.
    pub fn new(delimiter: impl Into<Vec<u8>>) -> Self {
        Self {
            name: "buffer-join".to_string(),
            delimiter: delimiter.into(),
            max_count: None,
            pending: Vec::new(),
            pending_size: 0,
            input_count: AtomicU64::new(0),
            output_count: AtomicU64::new(0),
        }
    }

    /// Set the number of buffers to accumulate before joining.
    pub fn with_count(mut self, count: usize) -> Self {
        self.max_count = Some(count);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the delimiter.
    pub fn delimiter(&self) -> &[u8] {
        &self.delimiter
    }

    /// Get the number of pending buffers.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> BufferJoinStats {
        BufferJoinStats {
            input_count: self.input_count.load(Ordering::Relaxed),
            output_count: self.output_count.load(Ordering::Relaxed),
            pending_count: self.pending.len(),
        }
    }

    /// Flush pending buffers and produce joined output.
    pub fn flush(&mut self) -> Result<Option<Buffer>> {
        if self.pending.is_empty() {
            return Ok(None);
        }

        let total_size = self.pending_size + (self.pending.len() - 1) * self.delimiter.len();
        let segment = Arc::new(HeapSegment::new(total_size.max(1))?);

        let ptr = segment.as_mut_ptr().unwrap();
        let mut offset = 0;

        for (i, part) in self.pending.iter().enumerate() {
            if i > 0 && !self.delimiter.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        self.delimiter.as_ptr(),
                        ptr.add(offset),
                        self.delimiter.len(),
                    );
                }
                offset += self.delimiter.len();
            }
            if !part.is_empty() {
                unsafe {
                    std::ptr::copy_nonoverlapping(part.as_ptr(), ptr.add(offset), part.len());
                }
                offset += part.len();
            }
        }

        self.pending.clear();
        self.pending_size = 0;

        let handle = MemoryHandle::from_segment_with_len(segment, total_size);
        self.output_count.fetch_add(1, Ordering::Relaxed);

        Ok(Some(Buffer::new(handle, Metadata::new())))
    }
}

impl Element for BufferJoin {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.input_count.fetch_add(1, Ordering::Relaxed);

        let data = buffer.as_bytes().to_vec();
        self.pending_size += data.len();
        self.pending.push(data);

        // Check if we should flush
        if let Some(max) = self.max_count {
            if self.pending.len() >= max {
                return self.flush();
            }
        }

        Ok(None)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for buffer join.
#[derive(Debug, Clone, Copy)]
pub struct BufferJoinStats {
    /// Number of input buffers received.
    pub input_count: u64,
    /// Number of output buffers produced.
    pub output_count: u64,
    /// Number of buffers pending.
    pub pending_count: usize,
}

/// Concatenates consecutive buffer contents into one.
///
/// Accumulates buffer contents until a flush is triggered or
/// a maximum size/count is reached.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::BufferConcat;
///
/// // Concatenate up to 10 buffers
/// let concat = BufferConcat::new().with_max_count(10);
/// ```
pub struct BufferConcat {
    name: String,
    max_count: Option<usize>,
    max_size: Option<usize>,
    pending: Vec<u8>,
    pending_count: usize,
    input_count: AtomicU64,
    output_count: AtomicU64,
}

impl BufferConcat {
    /// Create a new buffer concatenator.
    pub fn new() -> Self {
        Self {
            name: "buffer-concat".to_string(),
            max_count: None,
            max_size: None,
            pending: Vec::new(),
            pending_count: 0,
            input_count: AtomicU64::new(0),
            output_count: AtomicU64::new(0),
        }
    }

    /// Set maximum number of buffers to concatenate.
    pub fn with_max_count(mut self, count: usize) -> Self {
        self.max_count = Some(count);
        self
    }

    /// Set maximum output size.
    pub fn with_max_size(mut self, size: usize) -> Self {
        self.max_size = Some(size);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the pending byte count.
    pub fn pending_size(&self) -> usize {
        self.pending.len()
    }

    /// Get statistics.
    pub fn stats(&self) -> BufferConcatStats {
        BufferConcatStats {
            input_count: self.input_count.load(Ordering::Relaxed),
            output_count: self.output_count.load(Ordering::Relaxed),
            pending_size: self.pending.len(),
        }
    }

    /// Flush pending data and produce output.
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

        self.pending.clear();
        self.pending_count = 0;

        let handle = MemoryHandle::from_segment_with_len(segment, len);
        self.output_count.fetch_add(1, Ordering::Relaxed);

        Ok(Some(Buffer::new(handle, Metadata::new())))
    }
}

impl Default for BufferConcat {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for BufferConcat {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.input_count.fetch_add(1, Ordering::Relaxed);

        let data = buffer.as_bytes();

        // Check if adding this would exceed max size
        if let Some(max) = self.max_size {
            if !self.pending.is_empty() && self.pending.len() + data.len() > max {
                // Flush current, then start new
                let result = self.flush();
                self.pending.extend_from_slice(data);
                self.pending_count = 1;
                return result;
            }
        }

        self.pending.extend_from_slice(data);
        self.pending_count += 1;

        // Check if we should flush
        if let Some(max) = self.max_count {
            if self.pending_count >= max {
                return self.flush();
            }
        }

        if let Some(max) = self.max_size {
            if self.pending.len() >= max {
                return self.flush();
            }
        }

        Ok(None)
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for buffer concat.
#[derive(Debug, Clone, Copy)]
pub struct BufferConcatStats {
    /// Number of input buffers received.
    pub input_count: u64,
    /// Number of output buffers produced.
    pub output_count: u64,
    /// Pending data size.
    pub pending_size: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_buffer(data: &[u8], seq: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(data.len().max(1)).unwrap());
        if !data.is_empty() {
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        }
        let handle = MemoryHandle::from_segment_with_len(segment, data.len());
        Buffer::new(handle, Metadata::with_sequence(seq))
    }

    // DuplicateFilter tests

    #[test]
    fn test_duplicate_filter_passes_unique() {
        let mut filter = DuplicateFilter::new();

        let buf1 = make_buffer(b"hello", 0);
        let buf2 = make_buffer(b"world", 1);

        assert!(filter.process(buf1).unwrap().is_some());
        assert!(filter.process(buf2).unwrap().is_some());

        let stats = filter.stats();
        assert_eq!(stats.passed, 2);
        assert_eq!(stats.dropped, 0);
    }

    #[test]
    fn test_duplicate_filter_drops_duplicates() {
        let mut filter = DuplicateFilter::new();

        let buf1 = make_buffer(b"hello", 0);
        let buf2 = make_buffer(b"hello", 1); // Same content

        assert!(filter.process(buf1).unwrap().is_some());
        assert!(filter.process(buf2).unwrap().is_none());

        let stats = filter.stats();
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.dropped, 1);
    }

    // RangeFilter tests

    #[test]
    fn test_range_filter_by_size() {
        let mut filter = RangeFilter::by_size(3, 6);

        let small = make_buffer(b"ab", 0);
        let ok = make_buffer(b"abcd", 1);
        let large = make_buffer(b"abcdefgh", 2);

        assert!(filter.process(small).unwrap().is_none());
        assert!(filter.process(ok).unwrap().is_some());
        assert!(filter.process(large).unwrap().is_none());
    }

    #[test]
    fn test_range_filter_by_sequence() {
        let mut filter = RangeFilter::by_sequence(5, 10);

        let before = make_buffer(b"x", 3);
        let in_range = make_buffer(b"x", 7);
        let after = make_buffer(b"x", 15);

        assert!(filter.process(before).unwrap().is_none());
        assert!(filter.process(in_range).unwrap().is_some());
        assert!(filter.process(after).unwrap().is_none());
    }

    // RegexFilter tests

    #[test]
    fn test_regex_filter_matches() {
        let mut filter = RegexFilter::new(r"hello").unwrap();

        let matching = make_buffer(b"hello world", 0);
        let not_matching = make_buffer(b"goodbye world", 1);

        assert!(filter.process(matching).unwrap().is_some());
        assert!(filter.process(not_matching).unwrap().is_none());
    }

    #[test]
    fn test_regex_filter_inverted() {
        let mut filter = RegexFilter::new(r"error").unwrap().inverted();

        let has_error = make_buffer(b"error occurred", 0);
        let no_error = make_buffer(b"all good", 1);

        assert!(filter.process(has_error).unwrap().is_none());
        assert!(filter.process(no_error).unwrap().is_some());
    }

    // BufferSplit tests

    #[test]
    fn test_buffer_split_basic() {
        let mut splitter = BufferSplit::new(b",");

        let buf = make_buffer(b"a,b,c", 0);
        let parts = splitter.process_all(buf).unwrap();

        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].as_bytes(), b"a");
        assert_eq!(parts[1].as_bytes(), b"b");
        assert_eq!(parts[2].as_bytes(), b"c");
    }

    #[test]
    fn test_buffer_split_keep_delimiter() {
        let mut splitter = BufferSplit::new(b"\n").keep_delimiter();

        let buf = make_buffer(b"line1\nline2\n", 0);
        let parts = splitter.process_all(buf).unwrap();

        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].as_bytes(), b"line1\n");
        assert_eq!(parts[1].as_bytes(), b"line2\n");
    }

    // BufferJoin tests

    #[test]
    fn test_buffer_join_with_count() {
        let mut joiner = BufferJoin::new(b",").with_count(3);

        assert!(joiner.process(make_buffer(b"a", 0)).unwrap().is_none());
        assert!(joiner.process(make_buffer(b"b", 1)).unwrap().is_none());
        let result = joiner.process(make_buffer(b"c", 2)).unwrap().unwrap();

        assert_eq!(result.as_bytes(), b"a,b,c");
    }

    #[test]
    fn test_buffer_join_flush() {
        let mut joiner = BufferJoin::new(b"-");

        joiner.process(make_buffer(b"x", 0)).unwrap();
        joiner.process(make_buffer(b"y", 1)).unwrap();

        let result = joiner.flush().unwrap().unwrap();
        assert_eq!(result.as_bytes(), b"x-y");
    }

    // BufferConcat tests

    #[test]
    fn test_buffer_concat_by_count() {
        let mut concat = BufferConcat::new().with_max_count(2);

        assert!(concat.process(make_buffer(b"hello", 0)).unwrap().is_none());
        let result = concat.process(make_buffer(b"world", 1)).unwrap().unwrap();

        assert_eq!(result.as_bytes(), b"helloworld");
    }

    #[test]
    fn test_buffer_concat_by_size() {
        let mut concat = BufferConcat::new().with_max_size(10);

        concat.process(make_buffer(b"hello", 0)).unwrap();
        concat.process(make_buffer(b"wor", 1)).unwrap();
        // Next one would exceed, so flush happens
        let result = concat.process(make_buffer(b"12345", 2)).unwrap().unwrap();

        assert_eq!(result.as_bytes(), b"hellowor");
    }

    #[test]
    fn test_buffer_concat_flush() {
        let mut concat = BufferConcat::new();

        concat.process(make_buffer(b"abc", 0)).unwrap();
        concat.process(make_buffer(b"def", 1)).unwrap();

        let result = concat.flush().unwrap().unwrap();
        assert_eq!(result.as_bytes(), b"abcdef");
    }
}
