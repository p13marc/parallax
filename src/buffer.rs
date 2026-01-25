//! Buffer types for zero-copy data passing.

use crate::memory::{MemorySegment, MemoryType};
use crate::metadata::Metadata;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// Validation state for archived buffers.
const NOT_VALIDATED: u8 = 0;
const VALIDATED_OK: u8 = 1;

/// Handle to a memory region within a segment.
///
/// This is cheap to clone (just Arc increment + copy of offset/len).
#[derive(Clone)]
pub struct MemoryHandle {
    /// The backing memory segment.
    segment: Arc<dyn MemorySegment>,
    /// Offset within the segment.
    offset: usize,
    /// Length of this buffer's data.
    len: usize,
}

impl MemoryHandle {
    /// Create a new memory handle.
    ///
    /// # Arguments
    ///
    /// * `segment` - The backing memory segment.
    /// * `offset` - Offset within the segment.
    /// * `len` - Length of the data.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > segment.len()`.
    pub fn new(segment: Arc<dyn MemorySegment>, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= segment.len(),
            "memory handle exceeds segment bounds"
        );
        Self {
            segment,
            offset,
            len,
        }
    }

    /// Create a memory handle covering an entire segment.
    pub fn from_segment(segment: Arc<dyn MemorySegment>) -> Self {
        let len = segment.len();
        Self {
            segment,
            offset: 0,
            len,
        }
    }

    /// Create a memory handle covering a portion of a segment starting at offset 0.
    ///
    /// Useful when you have a larger segment but only wrote `len` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `len > segment.len()`.
    pub fn from_segment_with_len(segment: Arc<dyn MemorySegment>, len: usize) -> Self {
        assert!(
            len <= segment.len(),
            "requested length exceeds segment size"
        );
        Self {
            segment,
            offset: 0,
            len,
        }
    }

    /// Get a pointer to the start of this handle's memory.
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.segment.as_ptr().add(self.offset) }
    }

    /// Get a mutable pointer to the start of this handle's memory.
    pub fn as_mut_ptr(&self) -> Option<*mut u8> {
        self.segment
            .as_mut_ptr()
            .map(|ptr| unsafe { ptr.add(self.offset) })
    }

    /// Get the length of this handle's data.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if this handle has zero length.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get this handle's data as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len) }
    }

    /// Get this handle's data as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        self.as_mut_ptr()
            .map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr, self.len) })
    }

    /// Get the memory type of the backing segment.
    pub fn memory_type(&self) -> MemoryType {
        self.segment.memory_type()
    }

    /// Get the offset within the segment.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get a reference to the backing segment.
    pub fn segment(&self) -> &Arc<dyn MemorySegment> {
        &self.segment
    }

    /// Create a sub-handle (a view into a portion of this handle).
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > self.len`.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len, "sub-handle exceeds parent bounds");
        Self {
            segment: Arc::clone(&self.segment),
            offset: self.offset + offset,
            len,
        }
    }
}

impl std::fmt::Debug for MemoryHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryHandle")
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("memory_type", &self.memory_type())
            .finish()
    }
}

/// A buffer containing data and metadata.
///
/// Buffers are the primary data container in Parallax pipelines. They consist of:
/// - A `MemoryHandle` pointing to the actual data
/// - `Metadata` with timestamps, flags, and extra fields
/// - A type parameter `T` for compile-time type safety (optional)
///
/// # Type Parameter
///
/// - `Buffer<()>` (or just `Buffer`): Dynamic buffer, type checked at runtime
/// - `Buffer<T>`: Typed buffer, provides compile-time safety
///
/// # Zero-Copy
///
/// Buffers are cheap to clone - only the Arc reference count is incremented.
/// The actual data is never copied during normal pipeline operations.
///
/// # Example
///
/// ```rust
/// use parallax::buffer::{Buffer, MemoryHandle};
/// use parallax::memory::HeapSegment;
/// use parallax::metadata::Metadata;
/// use std::sync::Arc;
///
/// // Create a buffer from a heap segment
/// let segment = Arc::new(HeapSegment::new(1024).unwrap());
/// let handle = MemoryHandle::from_segment(segment);
/// let buffer = Buffer::<()>::new(handle, Metadata::with_sequence(0));
///
/// // Clone is O(1) - just Arc increment
/// let buffer2 = buffer.clone();
/// ```
pub struct Buffer<T = ()> {
    /// Handle to the memory region containing the data.
    memory: MemoryHandle,
    /// Buffer metadata.
    metadata: Metadata,
    /// Validation state (for rkyv archived data).
    /// 0 = not validated, 1 = validated OK
    validated: AtomicU8,
    /// Type marker for compile-time safety.
    _marker: PhantomData<T>,
}

impl<T> Buffer<T> {
    /// Create a new buffer.
    ///
    /// # Arguments
    ///
    /// * `memory` - Handle to the memory containing the data.
    /// * `metadata` - Buffer metadata.
    pub fn new(memory: MemoryHandle, metadata: Metadata) -> Self {
        Self {
            memory,
            metadata,
            validated: AtomicU8::new(NOT_VALIDATED),
            _marker: PhantomData,
        }
    }

    /// Get a reference to the buffer's metadata.
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Get a mutable reference to the buffer's metadata.
    pub fn metadata_mut(&mut self) -> &mut Metadata {
        &mut self.metadata
    }

    /// Get a reference to the memory handle.
    pub fn memory(&self) -> &MemoryHandle {
        &self.memory
    }

    /// Get the buffer data as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        self.memory.as_slice()
    }

    /// Get the length of the buffer data.
    pub fn len(&self) -> usize {
        self.memory.len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.memory.is_empty()
    }

    /// Get the memory type of the backing segment.
    pub fn memory_type(&self) -> MemoryType {
        self.memory.memory_type()
    }

    /// Check if this buffer has been validated (for rkyv data).
    pub fn is_validated(&self) -> bool {
        self.validated.load(Ordering::Acquire) == VALIDATED_OK
    }

    /// Mark this buffer as validated.
    ///
    /// This is used internally by the rkyv access methods to cache
    /// validation state.
    pub fn mark_validated(&self) {
        self.validated.store(VALIDATED_OK, Ordering::Release);
    }

    /// Create a sub-buffer (a view into a portion of this buffer).
    ///
    /// The new buffer shares the same memory and metadata.
    ///
    /// # Arguments
    ///
    /// * `offset` - Offset within this buffer's data.
    /// * `len` - Length of the sub-buffer.
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > self.len()`.
    pub fn slice(&self, offset: usize, len: usize) -> Buffer<T> {
        Buffer {
            memory: self.memory.slice(offset, len),
            metadata: self.metadata.clone(),
            validated: AtomicU8::new(NOT_VALIDATED), // Sub-buffer needs revalidation
            _marker: PhantomData,
        }
    }

    /// Convert this buffer to a dynamically-typed buffer.
    ///
    /// This is always safe (type erasure).
    pub fn into_dynamic(self) -> Buffer<()> {
        Buffer {
            memory: self.memory,
            metadata: self.metadata,
            validated: AtomicU8::new(self.validated.load(Ordering::Relaxed)),
            _marker: PhantomData,
        }
    }
}

impl<T> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self {
            memory: self.memory.clone(),
            metadata: self.metadata.clone(),
            validated: AtomicU8::new(self.validated.load(Ordering::Relaxed)),
            _marker: PhantomData,
        }
    }
}

impl<T> std::fmt::Debug for Buffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Buffer")
            .field("memory", &self.memory)
            .field("metadata", &self.metadata)
            .field("validated", &self.is_validated())
            .finish()
    }
}

// Buffer is Send + Sync if T is Send + Sync (which () is)
unsafe impl<T: Send> Send for Buffer<T> {}
unsafe impl<T: Sync> Sync for Buffer<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HeapSegment;

    fn make_test_buffer(size: usize) -> Buffer {
        let segment = Arc::new(HeapSegment::new(size).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::with_sequence(42))
    }

    #[test]
    fn test_buffer_creation() {
        let buffer = make_test_buffer(1024);
        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.metadata().sequence, 42);
        assert!(!buffer.is_validated());
    }

    #[test]
    fn test_buffer_clone_is_cheap() {
        let buffer = make_test_buffer(1024);
        let buffer2 = buffer.clone();

        // Both should point to the same memory
        assert_eq!(buffer.as_bytes().as_ptr(), buffer2.as_bytes().as_ptr());
    }

    #[test]
    fn test_buffer_slice() {
        let buffer = make_test_buffer(1024);
        let sub = buffer.slice(100, 200);

        assert_eq!(sub.len(), 200);
        assert_eq!(sub.memory().offset(), 100);
    }

    #[test]
    fn test_buffer_validation_state() {
        let buffer = make_test_buffer(1024);
        assert!(!buffer.is_validated());

        buffer.mark_validated();
        assert!(buffer.is_validated());

        // Clone should preserve validation state
        let buffer2 = buffer.clone();
        assert!(buffer2.is_validated());
    }

    #[test]
    fn test_memory_handle_slice() {
        let segment = Arc::new(HeapSegment::new(1024).unwrap());
        let handle = MemoryHandle::from_segment(segment);

        let sub = handle.slice(100, 200);
        assert_eq!(sub.offset(), 100);
        assert_eq!(sub.len(), 200);
    }

    #[test]
    #[should_panic(expected = "sub-handle exceeds parent bounds")]
    fn test_memory_handle_slice_out_of_bounds() {
        let segment = Arc::new(HeapSegment::new(1024).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let _ = handle.slice(900, 200); // 900 + 200 > 1024
    }
}
