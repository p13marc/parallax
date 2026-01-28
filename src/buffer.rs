//! Buffer types for zero-copy data passing.

use crate::memory::{ArenaSlot, MemorySegment, MemoryType, SharedIpcSlotRef, SharedSlotRef};
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
/// - A `SharedSlotRef` from a `SharedArena` (preferred, cross-process refcounting)
/// - An `Arc<dyn MemorySegment>` for legacy/simple cases (deprecated)
///
/// Cloning a `MemoryHandle` is cheap:
/// - SharedSlot variant: Atomic increment of refcount in shared memory
/// - Segment variant: Arc increment (heap-based, single-process only)
///
/// For cross-process pipelines, always use the SharedSlot variant.
pub enum MemoryHandle {
    /// Slot from a SharedArena (preferred for cross-process).
    ///
    /// The refcount is in shared memory, enabling true cross-process
    /// reference counting and pool reclamation.
    SharedSlot {
        /// The slot reference (refcount in shared memory).
        slot: SharedSlotRef,
        /// Offset within the slot (for sub-buffers).
        offset: usize,
        /// Length of this buffer's data.
        len: usize,
    },
    /// Slot from a CpuArena (single-process pooling).
    ///
    /// Use this for efficient single-process buffer pooling.
    /// For cross-process, use SharedSlot instead.
    Arena {
        /// The arena slot (owns the slot, returns on drop).
        slot: Arc<ArenaSlot>,
        /// Offset within the slot (for sub-buffers).
        offset: usize,
        /// Length of this buffer's data.
        len: usize,
    },
    /// Standalone segment (deprecated, single-process only).
    ///
    /// This variant exists for backward compatibility. For new code,
    /// prefer using `SharedSlot` which works across processes.
    #[deprecated(
        since = "0.3.0",
        note = "Use SharedSlot variant for cross-process support"
    )]
    Segment {
        /// The backing memory segment.
        segment: Arc<dyn MemorySegment>,
        /// Offset within the segment.
        offset: usize,
        /// Length of this buffer's data.
        len: usize,
    },
}

impl MemoryHandle {
    /// Create a memory handle from a SharedSlotRef (preferred).
    ///
    /// This is the recommended way to create buffers. The refcount is in
    /// shared memory, enabling cross-process reference counting.
    pub fn from_shared_slot(slot: SharedSlotRef) -> Self {
        let len = slot.len();
        Self::SharedSlot {
            slot,
            offset: 0,
            len,
        }
    }

    /// Create a memory handle from a SharedSlotRef with a specific length.
    ///
    /// Useful when you have a slot but only wrote `len` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `len > slot.len()`.
    pub fn from_shared_slot_with_len(slot: SharedSlotRef, len: usize) -> Self {
        assert!(len <= slot.len(), "requested length exceeds slot size");
        Self::SharedSlot {
            slot,
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

    /// Create a new memory handle from a segment (deprecated).
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
    #[deprecated(
        since = "0.3.0",
        note = "Use from_shared_slot for cross-process support"
    )]
    #[allow(deprecated)]
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

    /// Create a memory handle covering an entire segment (deprecated).
    #[deprecated(
        since = "0.3.0",
        note = "Use from_shared_slot for cross-process support"
    )]
    #[allow(deprecated)]
    pub fn from_segment(segment: Arc<dyn MemorySegment>) -> Self {
        let len = segment.len();
        Self::Segment {
            segment,
            offset: 0,
            len,
        }
    }

    /// Create a memory handle covering a portion of a segment starting at offset 0 (deprecated).
    ///
    /// Useful when you have a larger segment but only wrote `len` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `len > segment.len()`.
    #[deprecated(
        since = "0.3.0",
        note = "Use from_shared_slot_with_len for cross-process support"
    )]
    #[allow(deprecated)]
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

    /// Get a pointer to the start of this handle's memory.
    #[allow(deprecated)]
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Self::SharedSlot { slot, offset, .. } => unsafe { slot.as_ptr().add(*offset) },
            Self::Arena { slot, offset, .. } => unsafe { slot.as_ptr().add(*offset) },
            Self::Segment {
                segment, offset, ..
            } => unsafe { segment.as_ptr().add(*offset) },
        }
    }

    /// Get a mutable pointer to the start of this handle's memory.
    #[allow(deprecated)]
    pub fn as_mut_ptr(&self) -> Option<*mut u8> {
        match self {
            Self::SharedSlot { slot, offset, .. } => {
                // SharedSlotRef data is always mutable
                Some(unsafe { slot.as_ptr().add(*offset) as *mut u8 })
            }
            Self::Arena { slot, offset, .. } => {
                // Arena slots are always mutable
                Some(unsafe { slot.as_ptr().add(*offset) as *mut u8 })
            }
            Self::Segment {
                segment, offset, ..
            } => segment.as_mut_ptr().map(|ptr| unsafe { ptr.add(*offset) }),
        }
    }

    /// Get the length of this handle's data.
    #[allow(deprecated)]
    pub fn len(&self) -> usize {
        match self {
            Self::SharedSlot { len, .. } | Self::Arena { len, .. } | Self::Segment { len, .. } => {
                *len
            }
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
    #[allow(deprecated)]
    pub fn memory_type(&self) -> MemoryType {
        match self {
            Self::SharedSlot { .. } | Self::Arena { .. } => MemoryType::Cpu,
            Self::Segment { segment, .. } => segment.memory_type(),
        }
    }

    /// Get the offset within the segment/slot.
    #[allow(deprecated)]
    pub fn offset(&self) -> usize {
        match self {
            Self::SharedSlot { offset, .. }
            | Self::Arena { offset, .. }
            | Self::Segment { offset, .. } => *offset,
        }
    }

    /// Get a reference to the backing segment (if this is a segment handle).
    ///
    /// Returns `None` for shared slot and arena handles.
    #[deprecated(since = "0.3.0", note = "SharedSlot handles don't expose the segment")]
    #[allow(deprecated)]
    pub fn segment(&self) -> Option<&Arc<dyn MemorySegment>> {
        match self {
            Self::Segment { segment, .. } => Some(segment),
            Self::SharedSlot { .. } | Self::Arena { .. } => None,
        }
    }

    /// Get a reference to the arena slot (if this is an arena handle).
    pub fn arena_slot(&self) -> Option<&Arc<ArenaSlot>> {
        match self {
            Self::Arena { slot, .. } => Some(slot),
            #[allow(deprecated)]
            Self::SharedSlot { .. } | Self::Segment { .. } => None,
        }
    }

    /// Get a reference to the shared slot (if this is a shared slot handle).
    pub fn shared_slot(&self) -> Option<&SharedSlotRef> {
        match self {
            Self::SharedSlot { slot, .. } => Some(slot),
            #[allow(deprecated)]
            Self::Arena { .. } | Self::Segment { .. } => None,
        }
    }

    /// Get an IPC reference for cross-process sharing.
    ///
    /// Only available for SharedSlot handles.
    /// Arena and Segment handles return `None`.
    #[allow(deprecated)]
    pub fn ipc_ref(&self) -> Option<SharedIpcSlotRef> {
        match self {
            Self::SharedSlot { slot, offset, len } => {
                let base_ref = slot.ipc_ref();
                Some(SharedIpcSlotRef {
                    arena_id: base_ref.arena_id,
                    slot_index: base_ref.slot_index,
                    data_offset: base_ref.data_offset + offset,
                    len: *len,
                })
            }
            Self::Arena { .. } | Self::Segment { .. } => None,
        }
    }

    /// Check if this handle is shared-slot-backed (preferred).
    #[allow(deprecated)]
    pub fn is_shared_slot(&self) -> bool {
        matches!(self, Self::SharedSlot { .. })
    }

    /// Check if this handle is segment-backed (deprecated).
    #[allow(deprecated)]
    pub fn is_segment(&self) -> bool {
        matches!(self, Self::Segment { .. })
    }

    /// Create a sub-handle (a view into a portion of this handle).
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > self.len()`.
    #[allow(deprecated)]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        let current_len = self.len();
        assert!(
            offset + len <= current_len,
            "sub-handle exceeds parent bounds"
        );

        match self {
            Self::SharedSlot {
                slot,
                offset: base_offset,
                ..
            } => Self::SharedSlot {
                slot: slot.slice(*base_offset + offset, len),
                offset: 0, // The slice already includes the offset
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
            Self::Segment {
                segment,
                offset: base_offset,
                ..
            } => Self::Segment {
                segment: Arc::clone(segment),
                offset: base_offset + offset,
                len,
            },
        }
    }
}

impl Clone for MemoryHandle {
    #[allow(deprecated)]
    fn clone(&self) -> Self {
        match self {
            Self::SharedSlot { slot, offset, len } => Self::SharedSlot {
                slot: slot.clone(), // Increments refcount in shared memory
                offset: *offset,
                len: *len,
            },
            Self::Arena { slot, offset, len } => Self::Arena {
                slot: Arc::clone(slot),
                offset: *offset,
                len: *len,
            },
            Self::Segment {
                segment,
                offset,
                len,
            } => Self::Segment {
                segment: Arc::clone(segment),
                offset: *offset,
                len: *len,
            },
        }
    }
}

impl std::fmt::Debug for MemoryHandle {
    #[allow(deprecated)]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SharedSlot { slot, offset, len } => f
                .debug_struct("MemoryHandle::SharedSlot")
                .field("arena_id", &slot.arena_id())
                .field("offset", offset)
                .field("len", len)
                .field("refcount", &slot.refcount())
                .finish(),
            Self::Arena { slot, offset, len } => f
                .debug_struct("MemoryHandle::Arena")
                .field("offset", offset)
                .field("len", len)
                .field("slot_len", &slot.len())
                .finish(),
            Self::Segment { offset, len, .. } => f
                .debug_struct("MemoryHandle::Segment")
                .field("offset", offset)
                .field("len", len)
                .field("memory_type", &self.memory_type())
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
    use crate::memory::SharedArena;

    fn make_test_buffer(arena: &SharedArena, size: usize) -> Buffer {
        let slot = arena.acquire().expect("arena not exhausted");
        // Only use the first `size` bytes if smaller than slot
        let len = size.min(slot.len());
        let handle = MemoryHandle::from_shared_slot_with_len(slot, len);
        Buffer::new(handle, Metadata::from_sequence(42))
    }

    #[test]
    fn test_buffer_creation() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let buffer = make_test_buffer(&arena, 1024);
        assert_eq!(buffer.len(), 1024);
        assert_eq!(buffer.metadata().sequence, 42);
        assert!(!buffer.is_validated());
    }

    #[test]
    fn test_buffer_clone_increments_shared_refcount() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let initial_refcount = slot.refcount();

        let handle = MemoryHandle::from_shared_slot(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

        // Clone increments refcount in shared memory
        let buffer2 = buffer.clone();
        assert_eq!(
            buffer.memory().shared_slot().unwrap().refcount(),
            initial_refcount + 1
        );

        // Both should point to the same memory
        assert_eq!(buffer.as_bytes().as_ptr(), buffer2.as_bytes().as_ptr());
    }

    #[test]
    fn test_buffer_slice() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let buffer = make_test_buffer(&arena, 1024);
        let sub = buffer.slice(100, 200);

        assert_eq!(sub.len(), 200);
        // SharedSlot slices have offset 0 (offset is baked into the slot)
        assert!(sub.memory().is_shared_slot());
    }

    #[test]
    fn test_buffer_validation_state() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let buffer = make_test_buffer(&arena, 1024);
        assert!(!buffer.is_validated());

        buffer.mark_validated();
        assert!(buffer.is_validated());

        // Clone should preserve validation state
        let buffer2 = buffer.clone();
        assert!(buffer2.is_validated());
    }

    #[test]
    fn test_shared_slot_buffer_creation() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::from_shared_slot(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(100));

        assert_eq!(buffer.len(), 4096);
        assert_eq!(buffer.metadata().sequence, 100);
        assert_eq!(buffer.memory_type(), MemoryType::Cpu);
        assert!(buffer.memory().is_shared_slot());
        assert!(!buffer.memory().is_segment());
    }

    #[test]
    fn test_shared_slot_released_on_drop() {
        let arena = SharedArena::new(4096, 4).unwrap();

        {
            let slot = arena.acquire().unwrap();
            assert_eq!(slot.refcount(), 1);

            let handle = MemoryHandle::from_shared_slot(slot);
            let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

            // Clone increases refcount
            let buffer2 = buffer.clone();
            assert_eq!(buffer.memory().shared_slot().unwrap().refcount(), 2);

            drop(buffer2);
            assert_eq!(buffer.memory().shared_slot().unwrap().refcount(), 1);
        }

        // After all references dropped, owner can reclaim
        arena.reclaim();
        // Slot should be available again (acquire should succeed)
        assert!(arena.acquire().is_some());
    }

    #[test]
    fn test_shared_slot_ipc_ref() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::from_shared_slot(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

        let ipc_ref = buffer
            .memory()
            .ipc_ref()
            .expect("shared slot should have ipc_ref");
        assert_eq!(ipc_ref.arena_id, arena.id());
        assert_eq!(ipc_ref.len, 4096);
    }

    #[test]
    fn test_shared_slot_slice() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::from_shared_slot(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

        let sub = buffer.slice(100, 200);
        assert_eq!(sub.len(), 200);
        assert!(sub.memory().is_shared_slot());

        // IPC ref should reflect the slice
        let ipc_ref = sub.memory().ipc_ref().unwrap();
        assert_eq!(ipc_ref.len, 200);
    }

    #[test]
    fn test_shared_slot_read_write() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().unwrap();

        // Write via slot
        slot.data_mut()[0..5].copy_from_slice(b"hello");

        let mut handle = MemoryHandle::from_shared_slot(slot);

        // Read via handle
        assert_eq!(&handle.as_slice()[0..5], b"hello");

        // Write via handle
        handle.as_mut_slice().unwrap()[5..10].copy_from_slice(b"world");

        // Read via buffer
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));
        assert_eq!(&buffer.as_bytes()[0..10], b"helloworld");
    }

    #[test]
    fn test_shared_slot_with_len() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::from_shared_slot_with_len(slot, 100);

        assert_eq!(handle.len(), 100);
        assert!(handle.is_shared_slot());
    }

    #[test]
    fn test_shared_slot_accessor() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::from_shared_slot(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

        assert!(buffer.memory().shared_slot().is_some());
    }

    #[test]
    #[should_panic(expected = "sub-handle exceeds parent bounds")]
    fn test_memory_handle_slice_out_of_bounds() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::from_shared_slot(slot);
        let _ = handle.slice(900, 200); // 900 + 200 > 1024
    }
}
