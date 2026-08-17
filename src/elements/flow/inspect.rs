//! Inspect element — a 1-in/1-out passthrough that counts what flows through it.

use crate::buffer::Buffer;
use crate::element::{Element, ExecutionHints};
use crate::error::Result;

/// A passthrough element that counts the buffers and bytes crossing it.
///
/// # This was called `Tee`, and it never was one
///
/// It has one input and one output. It does not duplicate anything. The name
/// promised a 1-to-N fan-out and delivered a counter, and the module docs
/// repeated the claim — so people reached for it to split a stream, and got a
/// single chain.
///
/// **Fan-out needs no element at all.** Src-pads are genuinely 1:N: link the
/// same node twice and the executor clones the buffer to each branch, which is
/// a refcount bump, not a copy.
///
/// ```rust,ignore
/// let src = pipeline.add_source("camera", V4l2Src::new("/dev/video0")?);
/// let rec = pipeline.add_sink("recorder", recorder);
/// let live = pipeline.add_sink("preview", preview);
///
/// pipeline.link(src, rec)?;         // both links leave the same src-pad
/// pipeline.link_lossy(src, live)?;  // and this one may drop when it falls behind
/// ```
///
/// Two constraints worth knowing:
///
/// * The **parse grammar is a strictly linear chain** (`a ! b ! c`) and cannot
///   express fan-out. It needs the programmatic API above.
/// * A `Block` branch that fills its channel back-pressures the source *and
///   every sibling branch*. Use
///   [`link_lossy`](crate::pipeline::Pipeline::link_lossy) on branches that are
///   allowed to fall behind — see [`LinkPolicy`](crate::pipeline::LinkPolicy).
///
/// `Inspect` is for what it actually does: counting buffers and bytes at a point
/// in the graph, as a debugging or accounting probe.
///
/// # Example
///
/// ```rust
/// use parallax::elements::Inspect;
/// use parallax::element::Element;
/// # use parallax::buffer::{Buffer, MemoryHandle};
/// # use parallax::memory::SharedArena;
/// # use parallax::metadata::Metadata;
///
/// let mut inspect = Inspect::new();
///
/// # let arena = SharedArena::new(64, 4).unwrap();
/// # let slot = arena.acquire().unwrap();
/// # let handle = MemoryHandle::new(slot);
/// # let buffer = Buffer::new(handle, Metadata::from_sequence(0));
/// let result = inspect.process(buffer).unwrap();
/// assert!(result.is_some());
/// assert_eq!(inspect.count(), 1);
/// ```
pub struct Inspect {
    name: String,
    /// Number of buffers that have passed through.
    count: u64,
    /// Total bytes that have passed through.
    bytes: u64,
}

impl Inspect {
    /// Create a new Inspect element.
    pub fn new() -> Self {
        Self {
            name: "inspect".to_string(),
            count: 0,
            bytes: 0,
        }
    }

    /// Create a new Inspect element with a custom name.
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

impl Default for Inspect {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Inspect {
    // #189: may forward the input buffer — the upstream producer's arena
    // budget accumulates through this element.
    fn passthrough(&self) -> bool {
        true
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.count += 1;
        self.bytes += buffer.len() as u64;
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::rt_safe()
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
        ARENA.get_or_init(|| SharedArena::new(256, 64).unwrap())
    }

    fn make_buffer(size: usize, seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, size);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn inspect_passes_buffer() {
        let mut inspect = Inspect::new();

        let buffer = make_buffer(64, 42);

        let result = inspect.process(buffer).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    #[test]
    fn inspect_tracks_statistics() {
        let mut inspect = Inspect::new();
        assert_eq!(inspect.count(), 0);
        assert_eq!(inspect.bytes(), 0);

        // Process a 64-byte buffer
        let buffer = make_buffer(64, 0);
        inspect.process(buffer).unwrap();

        assert_eq!(inspect.count(), 1);
        assert_eq!(inspect.bytes(), 64);

        // Process another 128-byte buffer
        let buffer = make_buffer(128, 1);
        inspect.process(buffer).unwrap();

        assert_eq!(inspect.count(), 2);
        assert_eq!(inspect.bytes(), 192);
    }

    #[test]
    fn inspect_reset() {
        let mut inspect = Inspect::new();

        let buffer = make_buffer(64, 0);
        inspect.process(buffer).unwrap();

        assert_eq!(inspect.count(), 1);
        inspect.reset();
        assert_eq!(inspect.count(), 0);
        assert_eq!(inspect.bytes(), 0);
    }

    #[test]
    fn inspect_custom_name() {
        let inspect = Inspect::with_name("my_counter");
        assert_eq!(inspect.name(), "my_counter");
    }
}
