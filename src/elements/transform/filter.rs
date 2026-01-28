//! Filter elements for conditional buffer passing.
//!
//! Various filter implementations for different use cases.

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;
use std::sync::atomic::{AtomicU64, Ordering};

/// A generic predicate-based filter.
///
/// Passes buffers that match the predicate, drops those that don't.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Filter;
///
/// // Only pass buffers larger than 100 bytes
/// let filter = Filter::new(|buf| buf.len() > 100);
///
/// // Only pass even sequence numbers
/// let filter = Filter::new(|buf| buf.metadata().sequence % 2 == 0);
/// ```
pub struct Filter<F>
where
    F: FnMut(&Buffer) -> bool + Send,
{
    name: String,
    predicate: F,
    passed: AtomicU64,
    dropped: AtomicU64,
}

impl<F> Filter<F>
where
    F: FnMut(&Buffer) -> bool + Send,
{
    /// Create a new filter with the given predicate.
    pub fn new(predicate: F) -> Self {
        Self {
            name: "filter".to_string(),
            predicate,
            passed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
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

    /// Get the number of dropped buffers.
    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Get statistics.
    pub fn stats(&self) -> FilterStats {
        FilterStats {
            passed: self.passed.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}

impl<F> Element for Filter<F>
where
    F: FnMut(&Buffer) -> bool + Send,
{
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        if (self.predicate)(&buffer) {
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

/// Statistics for Filter elements.
#[derive(Debug, Clone, Copy)]
pub struct FilterStats {
    /// Buffers that passed the filter.
    pub passed: u64,
    /// Buffers that were dropped.
    pub dropped: u64,
}

/// A sample filter that passes every Nth buffer or random percentage.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::SampleFilter;
///
/// // Pass every 10th buffer
/// let filter = SampleFilter::every_nth(10);
///
/// // Pass ~50% of buffers randomly
/// let filter = SampleFilter::random_percent(50);
/// ```
pub struct SampleFilter {
    name: String,
    mode: SampleMode,
    count: AtomicU64,
    passed: AtomicU64,
    rng_state: u64,
}

/// Sampling mode for SampleFilter.
#[derive(Debug, Clone, Copy)]
pub enum SampleMode {
    /// Pass every Nth buffer.
    EveryNth(u64),
    /// Pass with given percentage probability (0-100).
    RandomPercent(u8),
    /// Pass first N buffers only.
    FirstN(u64),
    /// Pass last N buffers (requires knowing total).
    SkipFirst(u64),
}

impl SampleFilter {
    /// Create a filter that passes every Nth buffer.
    pub fn every_nth(n: u64) -> Self {
        Self {
            name: "sample-filter".to_string(),
            mode: SampleMode::EveryNth(n.max(1)),
            count: AtomicU64::new(0),
            passed: AtomicU64::new(0),
            rng_state: 0x853c49e6748fea9b,
        }
    }

    /// Create a filter that passes buffers with given probability.
    pub fn random_percent(percent: u8) -> Self {
        Self {
            name: "sample-filter".to_string(),
            mode: SampleMode::RandomPercent(percent.min(100)),
            count: AtomicU64::new(0),
            passed: AtomicU64::new(0),
            rng_state: 0x853c49e6748fea9b,
        }
    }

    /// Create a filter that passes only the first N buffers.
    pub fn first_n(n: u64) -> Self {
        Self {
            name: "sample-filter".to_string(),
            mode: SampleMode::FirstN(n),
            count: AtomicU64::new(0),
            passed: AtomicU64::new(0),
            rng_state: 0,
        }
    }

    /// Create a filter that skips the first N buffers.
    pub fn skip_first(n: u64) -> Self {
        Self {
            name: "sample-filter".to_string(),
            mode: SampleMode::SkipFirst(n),
            count: AtomicU64::new(0),
            passed: AtomicU64::new(0),
            rng_state: 0,
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set random seed for reproducible sampling.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> FilterStats {
        FilterStats {
            passed: self.passed.load(Ordering::Relaxed),
            dropped: self.count.load(Ordering::Relaxed) - self.passed.load(Ordering::Relaxed),
        }
    }

    // Simple xorshift PRNG
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }
}

impl Element for SampleFilter {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let count = self.count.fetch_add(1, Ordering::Relaxed);

        let pass = match self.mode {
            SampleMode::EveryNth(n) => count % n == 0,
            SampleMode::RandomPercent(pct) => {
                let r = self.next_random() % 100;
                r < pct as u64
            }
            SampleMode::FirstN(n) => count < n,
            SampleMode::SkipFirst(n) => count >= n,
        };

        if pass {
            self.passed.fetch_add(1, Ordering::Relaxed);
            Ok(Some(buffer))
        } else {
            Ok(None)
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Filter buffers by metadata values.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::MetadataFilter;
///
/// // Only pass buffers with stream_id == 1
/// let filter = MetadataFilter::by_stream_id(1);
/// ```
pub struct MetadataFilter {
    name: String,
    stream_id: Option<u32>,
    min_sequence: Option<u64>,
    max_sequence: Option<u64>,
    passed: AtomicU64,
    dropped: AtomicU64,
}

impl MetadataFilter {
    /// Create a new metadata filter.
    pub fn new() -> Self {
        Self {
            name: "metadata-filter".to_string(),
            stream_id: None,
            min_sequence: None,
            max_sequence: None,
            passed: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
        }
    }

    /// Filter by stream ID.
    pub fn by_stream_id(id: u32) -> Self {
        Self::new().with_stream_id(id)
    }

    /// Set stream ID filter.
    pub fn with_stream_id(mut self, id: u32) -> Self {
        self.stream_id = Some(id);
        self
    }

    /// Set minimum sequence number (inclusive).
    pub fn with_min_sequence(mut self, min: u64) -> Self {
        self.min_sequence = Some(min);
        self
    }

    /// Set maximum sequence number (inclusive).
    pub fn with_max_sequence(mut self, max: u64) -> Self {
        self.max_sequence = Some(max);
        self
    }

    /// Set sequence range (inclusive).
    pub fn with_sequence_range(mut self, min: u64, max: u64) -> Self {
        self.min_sequence = Some(min);
        self.max_sequence = Some(max);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> FilterStats {
        FilterStats {
            passed: self.passed.load(Ordering::Relaxed),
            dropped: self.dropped.load(Ordering::Relaxed),
        }
    }
}

impl Default for MetadataFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for MetadataFilter {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let meta = buffer.metadata();
        let mut pass = true;

        if let Some(expected_id) = self.stream_id {
            if meta.stream_id != expected_id {
                pass = false;
            }
        }

        if let Some(min) = self.min_sequence {
            if meta.sequence < min {
                pass = false;
            }
        }

        if let Some(max) = self.max_sequence {
            if meta.sequence > max {
                pass = false;
            }
        }

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(256, 2048).unwrap())
    }

    fn create_test_buffer(size: usize, seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, size);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    fn create_buffer_with_stream(seq: u64, stream_id: u32) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 64);
        let mut meta = Metadata::from_sequence(seq);
        meta.stream_id = stream_id;
        Buffer::new(handle, meta)
    }

    // Filter tests

    #[test]
    fn test_filter_basic() {
        let mut filter = Filter::new(|buf| buf.len() > 50);

        let small = create_test_buffer(30, 0);
        let large = create_test_buffer(100, 1);

        assert!(filter.process(small).unwrap().is_none());
        assert!(filter.process(large).unwrap().is_some());

        let stats = filter.stats();
        assert_eq!(stats.passed, 1);
        assert_eq!(stats.dropped, 1);
    }

    #[test]
    fn test_filter_by_sequence() {
        let mut filter = Filter::new(|buf| buf.metadata().sequence % 2 == 0);

        for i in 0..10 {
            let buf = create_test_buffer(64, i);
            let result = filter.process(buf).unwrap();
            if i % 2 == 0 {
                assert!(result.is_some());
            } else {
                assert!(result.is_none());
            }
        }

        assert_eq!(filter.passed_count(), 5);
        assert_eq!(filter.dropped_count(), 5);
    }

    // SampleFilter tests

    #[test]
    fn test_sample_filter_every_nth() {
        let mut filter = SampleFilter::every_nth(3);

        let mut passed = 0;
        for i in 0..9 {
            let buf = create_test_buffer(64, i);
            if filter.process(buf).unwrap().is_some() {
                passed += 1;
            }
        }

        assert_eq!(passed, 3); // 0, 3, 6
    }

    #[test]
    fn test_sample_filter_first_n() {
        let mut filter = SampleFilter::first_n(3);

        for i in 0..10 {
            let buf = create_test_buffer(64, i);
            let result = filter.process(buf).unwrap();
            if i < 3 {
                assert!(result.is_some());
            } else {
                assert!(result.is_none());
            }
        }
    }

    #[test]
    fn test_sample_filter_skip_first() {
        let mut filter = SampleFilter::skip_first(3);

        for i in 0..10 {
            let buf = create_test_buffer(64, i);
            let result = filter.process(buf).unwrap();
            if i < 3 {
                assert!(result.is_none());
            } else {
                assert!(result.is_some());
            }
        }
    }

    #[test]
    fn test_sample_filter_random() {
        let mut filter = SampleFilter::random_percent(50).with_seed(12345);

        let mut passed = 0;
        for i in 0..1000 {
            let buf = create_test_buffer(64, i);
            if filter.process(buf).unwrap().is_some() {
                passed += 1;
            }
        }

        // Should be roughly 50%, allow some variance
        assert!(passed > 400 && passed < 600);
    }

    // MetadataFilter tests

    #[test]
    fn test_metadata_filter_stream_id() {
        let mut filter = MetadataFilter::by_stream_id(1);

        let buf0 = create_buffer_with_stream(0, 0);
        let buf1 = create_buffer_with_stream(1, 1);
        let buf2 = create_buffer_with_stream(2, 1);

        assert!(filter.process(buf0).unwrap().is_none());
        assert!(filter.process(buf1).unwrap().is_some());
        assert!(filter.process(buf2).unwrap().is_some());

        assert_eq!(filter.stats().passed, 2);
    }

    #[test]
    fn test_metadata_filter_sequence_range() {
        let mut filter = MetadataFilter::new().with_sequence_range(5, 10);

        for i in 0..15 {
            let buf = create_test_buffer(64, i);
            let result = filter.process(buf).unwrap();
            if i >= 5 && i <= 10 {
                assert!(result.is_some(), "seq {} should pass", i);
            } else {
                assert!(result.is_none(), "seq {} should be dropped", i);
            }
        }
    }
}
