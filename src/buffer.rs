//! Buffer types for zero-copy data passing.

use crate::memory::{ArenaSlot, IpcSlotRef, MemorySegment, MemoryType};
use crate::metadata::Metadata;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// Validation state for archived buffers.
const NOT_VALIDATED: u8 = 0;
const VALIDATED_OK: u8 = 1;

/// Handle to a memory region.
///
/// This can be either:
/// - A reference to an `Arc<dyn MemorySegment>` (standalone allocation)
/// - An `ArenaSlot` (slot in a shared arena, more efficient for pools)
///
/// Cloning a `MemoryHandle` is cheap:
/// - Segment variant: Arc increment + copy of offset/len
/// - Arena variant: Arc increment (ArenaSlot contains `Arc<CpuArena>`)
pub enum MemoryHandle {
    /// Standalone segment with offset/length window.
    Segment {
        /// The backing memory segment.
        segment: Arc<dyn MemorySegment>,
        /// Offset within the segment.
        offset: usize,
        /// Length of this buffer's data.
        len: usize,
    },
    /// Slot in an arena (more efficient for pools).
    ///
    /// The ArenaSlot is an RAII guard that returns the slot to
    /// the arena when all references are dropped.
    Arena {
        /// The arena slot (owns the slot, returns on drop).
        slot: Arc<ArenaSlot>,
        /// Offset within the slot (for sub-buffers).
        offset: usize,
        /// Length of this buffer's data.
        len: usize,
    },
}

impl MemoryHandle {
    /// Create a new memory handle from a segment.
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
        Self::Segment {
            segment,
            offset,
            len,
        }
    }

    /// Create a memory handle covering an entire segment.
    pub fn from_segment(segment: Arc<dyn MemorySegment>) -> Self {
        let len = segment.len();
        Self::Segment {
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
        Self::Segment {
            segment,
            offset: 0,
            len,
        }
    }

    /// Create a memory handle from an arena slot.
    ///
    /// The slot is wrapped in an Arc so that cloning the handle
    /// doesn't duplicate the slot reservation.
    pub fn from_arena_slot(slot: ArenaSlot) -> Self {
        let len = slot.len();
        Self::Arena {
            slot: Arc::new(slot),
            offset: 0,
            len,
        }
    }

    /// Create a memory handle from an arena slot with a specific length.
    ///
    /// Useful when you have a slot but only wrote `len` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `len > slot.len()`.
    pub fn from_arena_slot_with_len(slot: ArenaSlot, len: usize) -> Self {
        assert!(len <= slot.len(), "requested length exceeds slot size");
        Self::Arena {
            slot: Arc::new(slot),
            offset: 0,
            len,
        }
    }

    /// Get a pointer to the start of this handle's memory.
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Segment {
                segment, offset, ..
            } => unsafe { segment.as_ptr().add(*offset) },
            Self::Arena { slot, offset, .. } => unsafe { slot.as_ptr().add(*offset) },
        }
    }

    /// Get a mutable pointer to the start of this handle's memory.
    pub fn as_mut_ptr(&self) -> Option<*mut u8> {
        match self {
            Self::Segment {
                segment, offset, ..
            } => segment.as_mut_ptr().map(|ptr| unsafe { ptr.add(*offset) }),
            Self::Arena { slot, offset, .. } => {
                // Arena slots are always mutable (we own the slot)
                // We need to get a mutable pointer through the Arc, which is safe
                // because we have exclusive access via the slot reservation
                Some(unsafe { slot.as_ptr().add(*offset) as *mut u8 })
            }
        }
    }

    /// Get the length of this handle's data.
    pub fn len(&self) -> usize {
        match self {
            Self::Segment { len, .. } | Self::Arena { len, .. } => *len,
        }
    }

    /// Check if this handle has zero length.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get this handle's data as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Get this handle's data as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        self.as_mut_ptr()
            .map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr, self.len()) })
    }

    /// Get the memory type of the backing segment.
    pub fn memory_type(&self) -> MemoryType {
        match self {
            Self::Segment { segment, .. } => segment.memory_type(),
            Self::Arena { .. } => MemoryType::Cpu,
        }
    }

    /// Get the offset within the segment/slot.
    pub fn offset(&self) -> usize {
        match self {
            Self::Segment { offset, .. } | Self::Arena { offset, .. } => *offset,
        }
    }

    /// Get a reference to the backing segment (if this is a segment handle).
    ///
    /// Returns `None` for arena handles.
    pub fn segment(&self) -> Option<&Arc<dyn MemorySegment>> {
        match self {
            Self::Segment { segment, .. } => Some(segment),
            Self::Arena { .. } => None,
        }
    }

    /// Get a reference to the arena slot (if this is an arena handle).
    ///
    /// Returns `None` for segment handles.
    pub fn arena_slot(&self) -> Option<&Arc<ArenaSlot>> {
        match self {
            Self::Segment { .. } => None,
            Self::Arena { slot, .. } => Some(slot),
        }
    }

    /// Get an IPC reference for cross-process sharing.
    ///
    /// Only available for arena handles. Segment handles must use
    /// fd passing via `IpcHandle` from the segment.
    pub fn ipc_ref(&self) -> Option<IpcSlotRef> {
        match self {
            Self::Segment { .. } => None,
            Self::Arena { slot, offset, len } => {
                let base_ref = slot.ipc_ref();
                Some(IpcSlotRef::new(
                    base_ref.arena_id,
                    base_ref.offset + offset,
                    *len,
                ))
            }
        }
    }

    /// Check if this handle is arena-backed.
    pub fn is_arena(&self) -> bool {
        matches!(self, Self::Arena { .. })
    }

    /// Check if this handle is segment-backed.
    pub fn is_segment(&self) -> bool {
        matches!(self, Self::Segment { .. })
    }

    /// Create a sub-handle (a view into a portion of this handle).
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > self.len()`.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        let current_len = self.len();
        assert!(
            offset + len <= current_len,
            "sub-handle exceeds parent bounds"
        );

        match self {
            Self::Segment {
                segment,
                offset: base_offset,
                ..
            } => Self::Segment {
                segment: Arc::clone(segment),
                offset: base_offset + offset,
                len,
            },
            Self::Arena {
                slot,
                offset: base_offset,
                ..
            } => Self::Arena {
                slot: Arc::clone(slot),
                offset: base_offset + offset,
                len,
            },
        }
    }
}

impl Clone for MemoryHandle {
    fn clone(&self) -> Self {
        match self {
            Self::Segment {
                segment,
                offset,
                len,
            } => Self::Segment {
                segment: Arc::clone(segment),
                offset: *offset,
                len: *len,
            },
            Self::Arena { slot, offset, len } => Self::Arena {
                slot: Arc::clone(slot),
                offset: *offset,
                len: *len,
            },
        }
    }
}

impl std::fmt::Debug for MemoryHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Segment { offset, len, .. } => f
                .debug_struct("MemoryHandle::Segment")
                .field("offset", offset)
                .field("len", len)
                .field("memory_type", &self.memory_type())
                .finish(),
            Self::Arena { offset, len, .. } => f
                .debug_struct("MemoryHandle::Arena")
                .field("offset", offset)
                .field("len", len)
                .finish(),
        }
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
/// use parallax::memory::CpuSegment;
/// use parallax::metadata::Metadata;
/// use std::sync::Arc;
///
/// // Create a buffer from a heap segment
/// let segment = Arc::new(CpuSegment::new(1024).unwrap());
/// let handle = MemoryHandle::from_segment(segment);
/// let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));
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
    use crate::memory::CpuSegment;

    fn make_test_buffer(size: usize) -> Buffer {
        let segment = Arc::new(CpuSegment::new(size).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::from_sequence(42))
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
        let segment = Arc::new(CpuSegment::new(1024).unwrap());
        let handle = MemoryHandle::from_segment(segment);

        let sub = handle.slice(100, 200);
        assert_eq!(sub.offset(), 100);
        assert_eq!(sub.len(), 200);
    }

    #[test]
    #[should_panic(expected = "sub-handle exceeds parent bounds")]
    fn test_memory_handle_slice_out_of_bounds() {
        let segment = Arc::new(CpuSegment::new(1024).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let _ = handle.slice(900, 200); // 900 + 200 > 1024
    }

    // Arena-backed buffer tests
    use crate::memory::CpuArena;

    fn make_arena_buffer(arena: &std::sync::Arc<CpuArena>) -> Buffer {
        let slot = arena.acquire().expect("arena not exhausted");
        let handle = MemoryHandle::from_arena_slot(slot);
        Buffer::new(handle, Metadata::from_sequence(100))
    }

    #[test]
    fn test_arena_buffer_creation() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let buffer = make_arena_buffer(&arena);

        assert_eq!(buffer.len(), 4096);
        assert_eq!(buffer.metadata().sequence, 100);
        assert_eq!(buffer.memory_type(), MemoryType::Cpu);
        assert!(buffer.memory().is_arena());
        assert!(!buffer.memory().is_segment());
    }

    #[test]
    fn test_arena_buffer_clone_is_cheap() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let buffer = make_arena_buffer(&arena);
        let buffer2 = buffer.clone();

        // Both should point to the same memory
        assert_eq!(buffer.as_bytes().as_ptr(), buffer2.as_bytes().as_ptr());

        // Slot should still be reserved (only 3 free)
        assert_eq!(arena.free_count(), 3);
    }

    #[test]
    fn test_arena_buffer_slot_released_on_drop() {
        let arena = CpuArena::new(4096, 4).unwrap();
        assert_eq!(arena.free_count(), 4);

        {
            let _buffer = make_arena_buffer(&arena);
            assert_eq!(arena.free_count(), 3);

            // Clone the buffer
            let _buffer2 = _buffer.clone();
            // Still only one slot used (Arc shared)
            assert_eq!(arena.free_count(), 3);
        }

        // Both dropped, slot should be released
        assert_eq!(arena.free_count(), 4);
    }

    #[test]
    fn test_arena_buffer_ipc_ref() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let buffer = make_arena_buffer(&arena);

        let ipc_ref = buffer
            .memory()
            .ipc_ref()
            .expect("arena should have ipc_ref");
        assert_eq!(ipc_ref.arena_id, arena.id());
        assert_eq!(ipc_ref.len, 4096);
    }

    #[test]
    fn test_arena_buffer_slice() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let buffer = make_arena_buffer(&arena);

        let sub = buffer.slice(100, 200);
        assert_eq!(sub.len(), 200);
        assert_eq!(sub.memory().offset(), 100);
        assert!(sub.memory().is_arena());

        // IPC ref should reflect the slice
        let ipc_ref = sub.memory().ipc_ref().unwrap();
        assert_eq!(ipc_ref.len, 200);
    }

    #[test]
    fn test_arena_buffer_read_write() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let mut handle = MemoryHandle::from_arena_slot(slot);

        // Write via handle
        handle.as_mut_slice().unwrap()[0..5].copy_from_slice(b"hello");

        // Read via buffer
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));
        assert_eq!(&buffer.as_bytes()[0..5], b"hello");
    }

    #[test]
    fn test_arena_buffer_with_len() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::from_arena_slot_with_len(slot, 100);

        assert_eq!(handle.len(), 100);
        assert!(handle.is_arena());
    }

    #[test]
    fn test_segment_buffer_has_no_ipc_ref() {
        let buffer = make_test_buffer(1024);
        assert!(buffer.memory().ipc_ref().is_none());
        assert!(buffer.memory().is_segment());
        assert!(!buffer.memory().is_arena());
    }

    #[test]
    fn test_arena_slot_accessor() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let buffer = make_arena_buffer(&arena);

        assert!(buffer.memory().arena_slot().is_some());
        assert!(buffer.memory().segment().is_none());
    }

    #[test]
    fn test_segment_accessor() {
        let buffer = make_test_buffer(1024);

        assert!(buffer.memory().segment().is_some());
        assert!(buffer.memory().arena_slot().is_none());
    }
}
