//! PassThrough element - passes buffers unchanged.

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;

/// An element that passes buffers through unchanged.
///
/// This is useful for:
/// - Debugging pipelines (can add logging in a custom version)
/// - Testing pipeline infrastructure
/// - Placeholder elements during development
///
/// # Example
///
/// ```rust
/// use parallax::elements::PassThrough;
/// use parallax::element::Element;
/// # use parallax::buffer::{Buffer, MemoryHandle};
/// # use parallax::memory::HeapSegment;
/// # use parallax::metadata::Metadata;
/// # use std::sync::Arc;
///
/// let mut passthrough = PassThrough::new();
///
/// // Create a test buffer
/// # let segment = Arc::new(HeapSegment::new(8).unwrap());
/// # let handle = MemoryHandle::from_segment(segment);
/// # let buffer = Buffer::new(handle, Metadata::with_sequence(42));
///
/// // PassThrough returns the buffer unchanged
/// let result = passthrough.process(buffer).unwrap();
/// assert!(result.is_some());
/// ```
pub struct PassThrough {
    name: String,
}

impl PassThrough {
    /// Create a new PassThrough element with the default name.
    pub fn new() -> Self {
        Self {
            name: "passthrough".to_string(),
        }
    }

    /// Create a new PassThrough element with a custom name.
    pub fn with_name(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for PassThrough {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for PassThrough {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    #[test]
    fn test_passthrough_passes_buffer() {
        let mut passthrough = PassThrough::new();

        let segment = Arc::new(HeapSegment::new(64).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::with_sequence(42));

        let result = passthrough.process(buffer).unwrap();
        assert!(result.is_some());

        let out_buffer = result.unwrap();
        assert_eq!(out_buffer.metadata().sequence, 42);
    }

    #[test]
    fn test_passthrough_custom_name() {
        let passthrough = PassThrough::with_name("my_passthrough");
        assert_eq!(passthrough.name(), "my_passthrough");
    }

    #[test]
    fn test_passthrough_default_name() {
        let passthrough = PassThrough::new();
        assert_eq!(passthrough.name(), "passthrough");
    }
}
