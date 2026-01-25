//! Identity element with callbacks for debugging.
//!
//! A pass-through element that allows inspection of buffers via callbacks.

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Callback type for buffer inspection.
pub type BufferCallback = Box<dyn Fn(&Buffer) + Send + Sync>;

/// An identity element that passes buffers through unchanged while
/// optionally calling callbacks for inspection.
///
/// This is useful for debugging, logging, or metrics collection
/// without modifying the data flow.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Identity;
///
/// let identity = Identity::new()
///     .with_name("debug-point")
///     .on_buffer(|buf| {
///         println!("Buffer: seq={}, len={}", buf.metadata().sequence, buf.len());
///     });
/// ```
pub struct Identity {
    name: String,
    callback: Option<Arc<BufferCallback>>,
    count: AtomicU64,
    bytes: AtomicU64,
}

impl Identity {
    /// Create a new identity element.
    pub fn new() -> Self {
        Self {
            name: "identity".to_string(),
            callback: None,
            count: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set a callback to be called for each buffer.
    pub fn on_buffer<F>(mut self, callback: F) -> Self
    where
        F: Fn(&Buffer) + Send + Sync + 'static,
    {
        self.callback = Some(Arc::new(Box::new(callback)));
        self
    }

    /// Get the number of buffers processed.
    pub fn buffer_count(&self) -> u64 {
        self.count.load(Ordering::Relaxed)
    }

    /// Get the total bytes processed.
    pub fn byte_count(&self) -> u64 {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Get statistics.
    pub fn stats(&self) -> IdentityStats {
        IdentityStats {
            buffer_count: self.count.load(Ordering::Relaxed),
            byte_count: self.bytes.load(Ordering::Relaxed),
        }
    }

    /// Reset statistics.
    pub fn reset_stats(&self) {
        self.count.store(0, Ordering::Relaxed);
        self.bytes.store(0, Ordering::Relaxed);
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for Identity {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(buffer.len() as u64, Ordering::Relaxed);

        if let Some(ref cb) = self.callback {
            cb(&buffer);
        }

        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for Identity element.
#[derive(Debug, Clone, Copy)]
pub struct IdentityStats {
    /// Number of buffers processed.
    pub buffer_count: u64,
    /// Total bytes processed.
    pub byte_count: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::atomic::AtomicUsize;

    fn create_test_buffer(size: usize, seq: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(size).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::with_sequence(seq))
    }

    #[test]
    fn test_identity_passthrough() {
        let mut identity = Identity::new();

        let buffer = create_test_buffer(100, 42);
        let result = identity.process(buffer).unwrap();

        assert!(result.is_some());
        let buf = result.unwrap();
        assert_eq!(buf.metadata().sequence, 42);
        assert_eq!(buf.len(), 100);
    }

    #[test]
    fn test_identity_callback() {
        let call_count = Arc::new(AtomicUsize::new(0));
        let call_count_clone = Arc::clone(&call_count);

        let mut identity = Identity::new().on_buffer(move |_buf| {
            call_count_clone.fetch_add(1, Ordering::Relaxed);
        });

        for i in 0..5 {
            identity.process(create_test_buffer(50, i)).unwrap();
        }

        assert_eq!(call_count.load(Ordering::Relaxed), 5);
    }

    #[test]
    fn test_identity_stats() {
        let mut identity = Identity::new();

        identity.process(create_test_buffer(100, 0)).unwrap();
        identity.process(create_test_buffer(200, 1)).unwrap();
        identity.process(create_test_buffer(50, 2)).unwrap();

        let stats = identity.stats();
        assert_eq!(stats.buffer_count, 3);
        assert_eq!(stats.byte_count, 350);
    }

    #[test]
    fn test_identity_reset_stats() {
        let mut identity = Identity::new();

        identity.process(create_test_buffer(100, 0)).unwrap();
        assert_eq!(identity.buffer_count(), 1);

        identity.reset_stats();
        assert_eq!(identity.buffer_count(), 0);
        assert_eq!(identity.byte_count(), 0);
    }

    #[test]
    fn test_identity_with_name() {
        let identity = Identity::new().with_name("my-identity");
        assert_eq!(identity.name(), "my-identity");
    }
}
