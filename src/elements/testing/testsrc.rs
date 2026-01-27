//! TestSrc element for generating test patterns.
//!
//! Generates various test patterns for pipeline testing and benchmarking.

use crate::element::{ProduceContext, ProduceResult, Source};
use crate::error::Result;
use std::time::{Duration, Instant};

/// Test pattern types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TestPattern {
    /// All zeros.
    #[default]
    Zero,
    /// All ones (0xFF).
    Ones,
    /// Incrementing bytes (0, 1, 2, ..., 255, 0, 1, ...).
    Counter,
    /// Random data.
    Random,
    /// Alternating 0x55/0xAA pattern.
    Alternating,
    /// Sequence number repeated to fill buffer.
    Sequence,
}

/// A source that generates test pattern buffers.
///
/// Useful for testing pipelines, benchmarking, and debugging.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::{TestSrc, TestPattern};
///
/// // Generate 100 buffers of 1KB zeros
/// let src = TestSrc::new()
///     .with_pattern(TestPattern::Zero)
///     .with_buffer_size(1024)
///     .with_num_buffers(100);
///
/// // Generate infinite random data at 1MB/s
/// let src = TestSrc::new()
///     .with_pattern(TestPattern::Random)
///     .with_buffer_size(4096)
///     .with_rate(1_000_000); // bytes per second
/// ```
pub struct TestSrc {
    name: String,
    pattern: TestPattern,
    buffer_size: usize,
    num_buffers: Option<u64>,
    sequence: u64,
    bytes_produced: u64,
    rate_limit: Option<u64>, // bytes per second
    last_produce: Option<Instant>,
    counter: u8,
    rng_state: u64,
}

impl TestSrc {
    /// Create a new test source with default settings.
    pub fn new() -> Self {
        Self {
            name: "testsrc".to_string(),
            pattern: TestPattern::default(),
            buffer_size: 4096,
            num_buffers: None,
            sequence: 0,
            bytes_produced: 0,
            rate_limit: None,
            last_produce: None,
            counter: 0,
            rng_state: 0x853c49e6748fea9b, // Arbitrary seed
        }
    }

    /// Set the test pattern.
    pub fn with_pattern(mut self, pattern: TestPattern) -> Self {
        self.pattern = pattern;
        self
    }

    /// Set the buffer size.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set the number of buffers to produce (None = infinite).
    pub fn with_num_buffers(mut self, count: u64) -> Self {
        self.num_buffers = Some(count);
        self
    }

    /// Set a rate limit in bytes per second.
    pub fn with_rate(mut self, bytes_per_second: u64) -> Self {
        self.rate_limit = Some(bytes_per_second);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the random seed (for reproducible Random pattern).
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = seed;
        self
    }

    /// Get the number of buffers produced.
    pub fn buffers_produced(&self) -> u64 {
        self.sequence
    }

    /// Get the total bytes produced.
    pub fn bytes_produced(&self) -> u64 {
        self.bytes_produced
    }

    /// Reset the source.
    pub fn reset(&mut self) {
        self.sequence = 0;
        self.bytes_produced = 0;
        self.counter = 0;
        self.last_produce = None;
    }

    // Simple xorshift64 PRNG
    fn next_random(&mut self) -> u64 {
        let mut x = self.rng_state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        x
    }

    fn fill_buffer(&mut self, data: &mut [u8]) {
        match self.pattern {
            TestPattern::Zero => {
                data.fill(0);
            }
            TestPattern::Ones => {
                data.fill(0xFF);
            }
            TestPattern::Counter => {
                for byte in data.iter_mut() {
                    *byte = self.counter;
                    self.counter = self.counter.wrapping_add(1);
                }
            }
            TestPattern::Random => {
                let mut i = 0;
                while i < data.len() {
                    let r = self.next_random();
                    let bytes = r.to_le_bytes();
                    let remaining = data.len() - i;
                    let to_copy = remaining.min(8);
                    data[i..i + to_copy].copy_from_slice(&bytes[..to_copy]);
                    i += to_copy;
                }
            }
            TestPattern::Alternating => {
                for (i, byte) in data.iter_mut().enumerate() {
                    *byte = if i % 2 == 0 { 0x55 } else { 0xAA };
                }
            }
            TestPattern::Sequence => {
                let seq_bytes = self.sequence.to_le_bytes();
                for (i, byte) in data.iter_mut().enumerate() {
                    *byte = seq_bytes[i % 8];
                }
            }
        }
    }

    fn apply_rate_limit(&mut self) {
        if let Some(rate) = self.rate_limit {
            if rate == 0 {
                return;
            }

            let now = Instant::now();

            if let Some(last) = self.last_produce {
                // Calculate expected time for bytes produced
                let expected_duration =
                    Duration::from_secs_f64(self.buffer_size as f64 / rate as f64);
                let elapsed = now.duration_since(last);

                if elapsed < expected_duration {
                    std::thread::sleep(expected_duration - elapsed);
                }
            }

            self.last_produce = Some(Instant::now());
        }
    }
}

impl Default for TestSrc {
    fn default() -> Self {
        Self::new()
    }
}

impl Source for TestSrc {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        // Check buffer limit
        if let Some(max) = self.num_buffers {
            if self.sequence >= max {
                return Ok(ProduceResult::Eos);
            }
        }

        // Apply rate limiting
        self.apply_rate_limit();

        // Fill the provided buffer
        let output = ctx.output();
        let len = output.len().min(self.buffer_size);
        self.fill_buffer(&mut output[..len]);

        // Set metadata
        ctx.set_sequence(self.sequence);

        self.sequence += 1;
        self.bytes_produced += len as u64;

        Ok(ProduceResult::Produced(len))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(self.buffer_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::CpuArena;
    use std::sync::Arc;

    // Helper to create a ProduceContext with an arena
    #[allow(dead_code)]
    fn produce_with_arena(src: &mut TestSrc, arena: &Arc<CpuArena>) -> Result<ProduceResult> {
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        src.produce(&mut ctx)
    }

    // Helper to produce and get buffer data
    fn produce_buffer(src: &mut TestSrc, arena: &Arc<CpuArena>) -> Option<Vec<u8>> {
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        match src.produce(&mut ctx).unwrap() {
            ProduceResult::Produced(n) => {
                let buffer = ctx.finalize(n);
                Some(buffer.as_bytes().to_vec())
            }
            ProduceResult::Eos => None,
            _ => None,
        }
    }

    #[test]
    fn test_testsrc_zero_pattern() {
        let mut src = TestSrc::new()
            .with_pattern(TestPattern::Zero)
            .with_buffer_size(100)
            .with_num_buffers(1);

        let arena = Arc::new(CpuArena::new(1024, 4).unwrap());
        let bytes = produce_buffer(&mut src, &arena).unwrap();
        assert!(bytes.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_testsrc_ones_pattern() {
        let mut src = TestSrc::new()
            .with_pattern(TestPattern::Ones)
            .with_buffer_size(100)
            .with_num_buffers(1);

        let arena = Arc::new(CpuArena::new(1024, 4).unwrap());
        let bytes = produce_buffer(&mut src, &arena).unwrap();
        assert!(bytes.iter().all(|&b| b == 0xFF));
    }

    #[test]
    fn test_testsrc_counter_pattern() {
        let mut src = TestSrc::new()
            .with_pattern(TestPattern::Counter)
            .with_buffer_size(256)
            .with_num_buffers(1);

        let arena = Arc::new(CpuArena::new(1024, 4).unwrap());
        let bytes = produce_buffer(&mut src, &arena).unwrap();

        for (i, &byte) in bytes.iter().enumerate() {
            assert_eq!(byte, i as u8);
        }
    }

    #[test]
    fn test_testsrc_alternating_pattern() {
        let mut src = TestSrc::new()
            .with_pattern(TestPattern::Alternating)
            .with_buffer_size(10)
            .with_num_buffers(1);

        let arena = Arc::new(CpuArena::new(1024, 4).unwrap());
        let bytes = produce_buffer(&mut src, &arena).unwrap();

        assert_eq!(bytes[0], 0x55);
        assert_eq!(bytes[1], 0xAA);
        assert_eq!(bytes[2], 0x55);
        assert_eq!(bytes[3], 0xAA);
    }

    #[test]
    fn test_testsrc_random_reproducible() {
        let mut src1 = TestSrc::new()
            .with_pattern(TestPattern::Random)
            .with_buffer_size(100)
            .with_seed(12345)
            .with_num_buffers(1);

        let mut src2 = TestSrc::new()
            .with_pattern(TestPattern::Random)
            .with_buffer_size(100)
            .with_seed(12345)
            .with_num_buffers(1);

        let arena = Arc::new(CpuArena::new(1024, 4).unwrap());
        let bytes1 = produce_buffer(&mut src1, &arena).unwrap();
        let bytes2 = produce_buffer(&mut src2, &arena).unwrap();

        assert_eq!(bytes1, bytes2);
    }

    #[test]
    fn test_testsrc_num_buffers() {
        let mut src = TestSrc::new().with_buffer_size(10).with_num_buffers(3);
        let arena = Arc::new(CpuArena::new(1024, 8).unwrap());

        assert!(produce_buffer(&mut src, &arena).is_some());
        assert!(produce_buffer(&mut src, &arena).is_some());
        assert!(produce_buffer(&mut src, &arena).is_some());
        assert!(produce_buffer(&mut src, &arena).is_none());

        assert_eq!(src.buffers_produced(), 3);
    }

    #[test]
    fn test_testsrc_sequence() {
        let mut src = TestSrc::new().with_buffer_size(10).with_num_buffers(3);
        let arena = Arc::new(CpuArena::new(1024, 8).unwrap());

        // Check sequence via metadata in context
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        src.produce(&mut ctx).unwrap();
        assert_eq!(ctx.metadata().sequence, 0);

        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        src.produce(&mut ctx).unwrap();
        assert_eq!(ctx.metadata().sequence, 1);

        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        src.produce(&mut ctx).unwrap();
        assert_eq!(ctx.metadata().sequence, 2);
    }

    #[test]
    fn test_testsrc_bytes_produced() {
        let mut src = TestSrc::new().with_buffer_size(100).with_num_buffers(5);
        let arena = Arc::new(CpuArena::new(1024, 8).unwrap());

        for _ in 0..5 {
            produce_buffer(&mut src, &arena);
        }

        assert_eq!(src.bytes_produced(), 500);
    }

    #[test]
    fn test_testsrc_reset() {
        let mut src = TestSrc::new().with_buffer_size(10).with_num_buffers(2);
        let arena = Arc::new(CpuArena::new(1024, 8).unwrap());

        produce_buffer(&mut src, &arena);
        produce_buffer(&mut src, &arena);
        assert!(produce_buffer(&mut src, &arena).is_none());

        src.reset();

        assert!(produce_buffer(&mut src, &arena).is_some());
        assert_eq!(src.buffers_produced(), 1);
    }

    #[test]
    fn test_testsrc_with_name() {
        let src = TestSrc::new().with_name("my-test");
        assert_eq!(src.name(), "my-test");
    }
}
