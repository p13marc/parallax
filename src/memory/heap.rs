//! Heap-backed memory segment.

use super::{IpcHandle, MemorySegment, MemoryType};
use crate::error::{Error, Result};

/// A memory segment backed by heap allocation.
///
/// This is the simplest memory backend, suitable for single-process pipelines.
/// It does not support cross-process sharing.
///
/// # Example
///
/// ```rust
/// use parallax::memory::{HeapSegment, MemorySegment};
///
/// let segment = HeapSegment::new(1024).unwrap();
/// assert_eq!(segment.len(), 1024);
/// ```
pub struct HeapSegment {
    /// The underlying memory allocation.
    /// Using a boxed slice ensures the memory is contiguous and won't be reallocated.
    data: Box<[u8]>,
}

impl HeapSegment {
    /// Create a new heap segment with the given size.
    ///
    /// The memory is zero-initialized.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes. Must be greater than 0.
    ///
    /// # Errors
    ///
    /// Returns an error if size is 0 or allocation fails.
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::AllocationFailed(
                "size must be greater than 0".into(),
            ));
        }

        // Allocate zeroed memory
        let data = vec![0u8; size].into_boxed_slice();

        Ok(Self { data })
    }

    /// Create a new heap segment with specific alignment.
    ///
    /// Useful when the memory needs to be aligned for SIMD or other requirements.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes.
    /// * `align` - Required alignment (must be a power of 2).
    ///
    /// # Errors
    ///
    /// Returns an error if allocation fails or alignment is invalid.
    pub fn with_alignment(size: usize, align: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::AllocationFailed(
                "size must be greater than 0".into(),
            ));
        }
        if !align.is_power_of_two() {
            return Err(Error::AllocationFailed(
                "alignment must be a power of 2".into(),
            ));
        }

        // For now, we just allocate normally and hope for the best.
        // A more sophisticated implementation would use aligned allocation.
        // The standard allocator typically provides at least 8 or 16 byte alignment.
        let segment = Self::new(size)?;

        // Verify alignment
        let ptr = segment.data.as_ptr();
        if (ptr as usize) % align != 0 {
            // In practice this is unlikely for reasonable alignments,
            // but for strict requirements we'd need a custom allocator.
            return Err(Error::AllocationFailed(format!(
                "could not achieve alignment of {} bytes",
                align
            )));
        }

        Ok(segment)
    }
}

impl MemorySegment for HeapSegment {
    fn as_ptr(&self) -> *const u8 {
        self.data.as_ptr()
    }

    fn as_mut_ptr(&self) -> Option<*mut u8> {
        // We have exclusive ownership, so we can provide mutable access.
        // This is safe because HeapSegment is not Clone.
        Some(self.data.as_ptr() as *mut u8)
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn ipc_handle(&self) -> Option<IpcHandle> {
        // Heap memory cannot be shared across processes
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heap_segment_creation() {
        let segment = HeapSegment::new(1024).unwrap();
        assert_eq!(segment.len(), 1024);
        assert_eq!(segment.memory_type(), MemoryType::Cpu);
        assert!(segment.ipc_handle().is_none());
    }

    #[test]
    fn test_heap_segment_zero_size_fails() {
        let result = HeapSegment::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_heap_segment_read_write() {
        let segment = HeapSegment::new(1024).unwrap();

        // Write some data
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::write(ptr, 42);
            std::ptr::write(ptr.add(1), 43);
        }

        // Read it back
        unsafe {
            let slice = segment.as_slice();
            assert_eq!(slice[0], 42);
            assert_eq!(slice[1], 43);
        }
    }

    #[test]
    fn test_heap_segment_is_zeroed() {
        let segment = HeapSegment::new(1024).unwrap();
        unsafe {
            let slice = segment.as_slice();
            assert!(slice.iter().all(|&b| b == 0));
        }
    }
}
