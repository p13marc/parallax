//! Null elements - NullSink and NullSource.

use crate::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use crate::error::{Error, Result};
use crate::memory::SharedArena;

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
/// use parallax::element::{ConsumeContext, Sink};
/// # use parallax::buffer::{Buffer, MemoryHandle};
/// # use parallax::memory::SharedArena;
/// # use parallax::metadata::Metadata;
///
/// let mut sink = NullSink::new();
///
/// // Create a test buffer
/// # let arena = SharedArena::new(8, 4).unwrap();
/// # let slot = arena.acquire().unwrap();
/// # let handle = MemoryHandle::new(slot);
/// # let buffer = Buffer::new(handle, Metadata::from_sequence(0));
/// # let ctx = ConsumeContext::new(&buffer);
///
/// // NullSink just discards the buffer
/// sink.consume(&ctx).unwrap();
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
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
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
/// use parallax::element::{ProduceContext, ProduceResult, Source};
/// use parallax::memory::SharedArena;
///
/// let arena = SharedArena::new(1024, 8).unwrap();
/// let mut source = NullSource::new(5);
///
/// // Produces 5 buffers, then EOS
/// for _ in 0..5 {
///     let slot = arena.acquire().unwrap();
///     let mut ctx = ProduceContext::new(slot);
///     let result = source.produce(&mut ctx).unwrap();
///     assert!(matches!(result, ProduceResult::Produced(_)));
/// }
///
/// // After 5 buffers, returns EOS
/// let slot = arena.acquire().unwrap();
/// let mut ctx = ProduceContext::new(slot);
/// let result = source.produce(&mut ctx).unwrap();
/// assert!(matches!(result, ProduceResult::Eos));
/// ```
pub struct NullSource {
    name: String,
    /// Number of buffers to produce.
    count: u64,
    /// Current sequence number.
    current: u64,
    /// Size of each buffer in bytes.
    buffer_size: usize,
    /// Arena for allocating buffers when no ProduceContext buffer is available.
    arena: Option<SharedArena>,
}

impl NullSource {
    /// Create a new NullSource that produces `count` buffers.
    pub fn new(count: u64) -> Self {
        Self {
            name: "nullsource".to_string(),
            count,
            current: 0,
            buffer_size: 64,
            arena: None,
        }
    }

    /// Create a new NullSource with a custom name.
    pub fn with_name(name: impl Into<String>, count: u64) -> Self {
        Self {
            name: name.into(),
            count,
            current: 0,
            buffer_size: 64,
            arena: None,
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
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.current >= self.count {
            return Ok(ProduceResult::Eos);
        }

        // Try to use the pre-allocated buffer if available, otherwise create our own
        if ctx.has_buffer() {
            let output = ctx.output();
            let len = output.len().min(self.buffer_size);
            output[..len].fill(0);
            ctx.set_sequence(self.current);
            self.current += 1;
            Ok(ProduceResult::Produced(len))
        } else {
            // Fallback: create our own buffer when no arena is configured
            use crate::buffer::{Buffer, MemoryHandle};
            use crate::metadata::Metadata;

            if self.arena.is_none() {
                // Use a reasonable number of slots for test/benchmark workloads
                self.arena = Some(SharedArena::new(self.buffer_size, 256)?);
            }
            let arena = self.arena.as_mut().unwrap();
            // Reclaim any freed slots before acquiring
            arena.reclaim();
            let slot = arena
                .acquire()
                .ok_or_else(|| Error::Element("arena exhausted".into()))?;
            // SharedArena slots are already zero-initialized
            let handle = MemoryHandle::with_len(slot, self.buffer_size);
            let buffer = Buffer::new(handle, Metadata::from_sequence(self.current));
            self.current += 1;
            Ok(ProduceResult::OwnBuffer(buffer))
        }
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
    use crate::buffer::{Buffer, MemoryHandle};
    use crate::metadata::Metadata;

    #[test]
    fn test_null_sink_consumes() {
        let mut sink = NullSink::new();
        assert_eq!(sink.count(), 0);

        let arena = SharedArena::new(64, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));
        let ctx = ConsumeContext::new(&buffer);

        sink.consume(&ctx).unwrap();
        assert_eq!(sink.count(), 1);
    }

    #[test]
    fn test_null_sink_custom_name() {
        let sink = NullSink::with_name("my_sink");
        assert_eq!(sink.name(), "my_sink");
    }

    #[test]
    fn test_null_source_produces_count() {
        use crate::memory::SharedArena;
        use std::sync::OnceLock;

        fn test_arena() -> &'static SharedArena {
            static ARENA: OnceLock<SharedArena> = OnceLock::new();
            ARENA.get_or_init(|| SharedArena::new(2048, 16).unwrap())
        }

        let mut source = NullSource::new(5);
        assert_eq!(source.remaining(), 5);

        for i in 0..5 {
            let slot = test_arena().acquire().unwrap();
            let mut ctx = ProduceContext::new(slot);
            let result = source.produce(&mut ctx).unwrap();
            match result {
                ProduceResult::Produced(n) => {
                    assert!(n > 0);
                    let buf = ctx.finalize(n);
                    assert_eq!(buf.metadata().sequence, i);
                }
                _ => panic!("Expected Produced result"),
            }
        }

        assert_eq!(source.remaining(), 0);

        // Should return Eos
        let slot = test_arena().acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        let result = source.produce(&mut ctx).unwrap();
        assert!(matches!(result, ProduceResult::Eos));
    }

    #[test]
    fn test_null_source_buffer_size() {
        use crate::memory::SharedArena;
        use std::sync::OnceLock;

        fn test_arena() -> &'static SharedArena {
            static ARENA: OnceLock<SharedArena> = OnceLock::new();
            ARENA.get_or_init(|| SharedArena::new(2048, 8).unwrap())
        }

        let mut source = NullSource::new(1).with_buffer_size(1024);

        let slot = test_arena().acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        let result = source.produce(&mut ctx).unwrap();
        match result {
            ProduceResult::Produced(n) => {
                assert_eq!(n, 1024);
                let buf = ctx.finalize(n);
                assert_eq!(buf.len(), 1024);
            }
            _ => panic!("Expected Produced result"),
        }
    }

    #[test]
    fn test_null_source_zero_count() {
        let mut source = NullSource::new(0);

        // Should immediately return Eos
        let mut ctx = ProduceContext::without_buffer();
        let result = source.produce(&mut ctx).unwrap();
        assert!(matches!(result, ProduceResult::Eos));
    }
}
