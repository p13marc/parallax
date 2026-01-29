//! Buffer types for zero-copy data passing.

use crate::memory::{MemoryType, SharedIpcSlotRef, SharedSlotRef};
use crate::metadata::Metadata;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU8, Ordering};

/// Validation state for archived buffers.
const NOT_VALIDATED: u8 = 0;
const VALIDATED_OK: u8 = 1;

/// Handle to a memory region backed by `SharedArena`.
///
/// All buffers use `SharedSlotRef` which stores its refcount in shared memory,
/// enabling true cross-process reference counting and zero-copy buffer sharing.
///
/// Cloning a `MemoryHandle` is cheap: just an atomic increment of the refcount
/// in shared memory.
pub struct MemoryHandle {
    /// The slot reference (refcount in shared memory).
    slot: SharedSlotRef,
    /// Offset within the slot (for sub-buffers).
    offset: usize,
    /// Length of this buffer's data.
    len: usize,
}

impl MemoryHandle {
    /// Create a memory handle from a SharedSlotRef.
    ///
    /// The refcount is in shared memory, enabling cross-process reference counting.
    pub fn new(slot: SharedSlotRef) -> Self {
        let len = slot.len();
        Self {
            slot,
            offset: 0,
            len,
        }
    }

    /// Create a memory handle with a specific length.
    ///
    /// Useful when you have a slot but only wrote `len` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `len > slot.len()`.
    pub fn with_len(slot: SharedSlotRef, len: usize) -> Self {
        assert!(len <= slot.len(), "requested length exceeds slot size");
        Self {
            slot,
            offset: 0,
            len,
        }
    }

    /// Get a pointer to the start of this handle's memory.
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.slot.as_ptr().add(self.offset) }
    }

    /// Get a mutable pointer to the start of this handle's memory.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { self.slot.as_mut_ptr().add(self.offset) }
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
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len) }
    }

    /// Get the memory type (always CPU for SharedArena).
    pub fn memory_type(&self) -> MemoryType {
        MemoryType::Cpu
    }

    /// Get the offset within the slot.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get a reference to the underlying slot.
    pub fn slot(&self) -> &SharedSlotRef {
        &self.slot
    }

    /// Get an IPC reference for cross-process sharing.
    pub fn ipc_ref(&self) -> SharedIpcSlotRef {
        let base_ref = self.slot.ipc_ref();
        SharedIpcSlotRef {
            arena_id: base_ref.arena_id,
            slot_index: base_ref.slot_index,
            data_offset: base_ref.data_offset + self.offset,
            len: self.len,
        }
    }

    /// Get the arena ID.
    pub fn arena_id(&self) -> u64 {
        self.slot.arena_id()
    }

    /// Get the arena's raw fd (for IPC).
    pub fn arena_fd(&self) -> i32 {
        self.slot.arena_fd()
    }

    /// Get the arena's total size (for IPC).
    pub fn arena_size(&self) -> usize {
        self.slot.arena_size()
    }

    /// Get the current refcount (for debugging).
    pub fn refcount(&self) -> u32 {
        self.slot.refcount()
    }

    /// Create a sub-handle (a view into a portion of this handle).
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > self.len()`.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(offset + len <= self.len, "sub-handle exceeds parent bounds");

        Self {
            slot: self.slot.slice(self.offset + offset, len),
            offset: 0, // The slice already includes the offset
            len,
        }
    }
}

impl Clone for MemoryHandle {
    fn clone(&self) -> Self {
        Self {
            slot: self.slot.clone(), // Increments refcount in shared memory
            offset: self.offset,
            len: self.len,
        }
    }
}

impl std::fmt::Debug for MemoryHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryHandle")
            .field("arena_id", &self.slot.arena_id())
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("refcount", &self.slot.refcount())
            .finish()
    }
}

/// A buffer containing data and metadata.
///
/// Buffers are the primary data container in Parallax pipelines. They consist of:
/// - A `MemoryHandle` pointing to the actual data (backed by SharedArena)
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
/// Buffers are cheap to clone - only the refcount in shared memory is incremented.
/// The actual data is never copied during normal pipeline operations.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::buffer::{Buffer, MemoryHandle};
/// use parallax::memory::SharedArena;
/// use parallax::metadata::Metadata;
///
/// // Create an arena and acquire a slot
/// let arena = SharedArena::new(1024, 4)?;
/// let slot = arena.acquire().unwrap();
///
/// // Create a buffer
/// let handle = MemoryHandle::new(slot);
/// let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));
///
/// // Clone is O(1) - just atomic increment in shared memory
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

// ============================================================================
// DmaBufBuffer - buffer backed by DMA-BUF segment
// ============================================================================

use crate::memory::DmaBufSegment;
use rustix::fd::BorrowedFd;

/// A buffer backed by a DMA-BUF file descriptor.
///
/// Unlike [`Buffer`] which uses [`SharedArena`](crate::memory::SharedArena),
/// this buffer directly owns a [`DmaBufSegment`] for zero-copy GPU integration.
///
/// # Use Cases
///
/// - V4L2 camera capture with `VIDIOC_EXPBUF`
/// - libcamera DMA-BUF frame buffers
/// - GPU encoder/decoder pipelines
/// - Zero-copy video processing
///
/// # Memory Type
///
/// Always reports [`MemoryType::DmaBuf`].
///
/// # Example
///
/// ```rust,ignore
/// use parallax::buffer::DmaBufBuffer;
/// use parallax::memory::DmaBufSegment;
/// use parallax::metadata::Metadata;
///
/// // From V4L2 exported buffer
/// let segment = DmaBufSegment::from_fd(dmabuf_fd, buffer_size)?;
/// let buffer = DmaBufBuffer::new(segment, Metadata::from_sequence(0));
///
/// // Access data
/// let data = buffer.as_bytes();
///
/// // Get fd for GPU import
/// let fd = buffer.as_fd();
/// ```
pub struct DmaBufBuffer {
    /// The DMA-BUF segment.
    segment: DmaBufSegment,
    /// Buffer metadata.
    metadata: Metadata,
}

impl DmaBufBuffer {
    /// Create a new DMA-BUF buffer.
    pub fn new(segment: DmaBufSegment, metadata: Metadata) -> Self {
        Self { segment, metadata }
    }

    /// Get the buffer data as a byte slice.
    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        DmaBufSegment::as_slice(&self.segment)
    }

    /// Get the buffer data as a mutable byte slice.
    ///
    /// Returns `None` if the segment is read-only.
    #[inline]
    pub fn as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        DmaBufSegment::as_mut_slice(&mut self.segment)
    }

    /// Get the length of the buffer data.
    #[inline]
    pub fn len(&self) -> usize {
        self.segment.size()
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.segment.size() == 0
    }

    /// Get the memory type (always DmaBuf).
    #[inline]
    pub fn memory_type(&self) -> MemoryType {
        MemoryType::DmaBuf
    }

    /// Get a reference to the buffer's metadata.
    #[inline]
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Get a mutable reference to the buffer's metadata.
    #[inline]
    pub fn metadata_mut(&mut self) -> &mut Metadata {
        &mut self.metadata
    }

    /// Get the DMA-BUF file descriptor.
    ///
    /// Use this for:
    /// - GPU import operations (Vulkan, VA-API)
    /// - IPC fd passing via `send_fds()`
    /// - Duplicating the fd with `try_clone()`
    #[inline]
    pub fn as_fd(&self) -> BorrowedFd<'_> {
        self.segment.as_fd()
    }

    /// Check if this buffer is read-only.
    #[inline]
    pub fn is_read_only(&self) -> bool {
        self.segment.is_read_only()
    }

    /// Get a reference to the underlying segment.
    #[inline]
    pub fn segment(&self) -> &DmaBufSegment {
        &self.segment
    }

    /// Consume the buffer and return the segment.
    #[inline]
    pub fn into_segment(self) -> DmaBufSegment {
        self.segment
    }

    /// Convert to a regular `Buffer` by copying data to an arena.
    ///
    /// Use this when you need to pass DMA-BUF data to an element that only
    /// accepts arena-backed buffers.
    ///
    /// # Arguments
    ///
    /// * `arena` - The arena to copy data into.
    ///
    /// # Errors
    ///
    /// Returns an error if the arena is exhausted.
    pub fn to_buffer(&self, arena: &crate::memory::SharedArena) -> crate::error::Result<Buffer> {
        let mut slot = arena
            .acquire()
            .ok_or_else(|| crate::error::Error::PoolExhausted)?;

        let data = self.as_bytes();
        if data.len() > slot.len() {
            return Err(crate::error::Error::InvalidSegment(format!(
                "DMA-BUF buffer ({} bytes) exceeds arena slot size ({} bytes)",
                data.len(),
                slot.len()
            )));
        }

        slot.data_mut()[..data.len()].copy_from_slice(data);

        let handle = MemoryHandle::with_len(slot, data.len());
        Ok(Buffer::new(handle, self.metadata.clone()))
    }
}

impl std::fmt::Debug for DmaBufBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DmaBufBuffer")
            .field("segment", &self.segment)
            .field("metadata", &self.metadata)
            .finish()
    }
}

// DmaBufBuffer is Send + Sync because DmaBufSegment is
unsafe impl Send for DmaBufBuffer {}
unsafe impl Sync for DmaBufBuffer {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SharedArena;

    fn make_test_buffer(arena: &SharedArena, size: usize) -> Buffer {
        let slot = arena.acquire().expect("arena not exhausted");
        let len = size.min(slot.len());
        let handle = MemoryHandle::with_len(slot, len);
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

        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

        // Clone increments refcount in shared memory
        let buffer2 = buffer.clone();
        assert_eq!(buffer.memory().refcount(), initial_refcount + 1);

        // Both should point to the same memory
        assert_eq!(buffer.as_bytes().as_ptr(), buffer2.as_bytes().as_ptr());
    }

    #[test]
    fn test_buffer_slice() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let buffer = make_test_buffer(&arena, 1024);
        let sub = buffer.slice(100, 200);

        assert_eq!(sub.len(), 200);
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
    fn test_buffer_creation_with_metadata() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(100));

        assert_eq!(buffer.len(), 4096);
        assert_eq!(buffer.metadata().sequence, 100);
        assert_eq!(buffer.memory_type(), MemoryType::Cpu);
    }

    #[test]
    fn test_slot_released_on_drop() {
        let arena = SharedArena::new(4096, 4).unwrap();

        {
            let slot = arena.acquire().unwrap();
            assert_eq!(slot.refcount(), 1);

            let handle = MemoryHandle::new(slot);
            let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

            // Clone increases refcount
            let buffer2 = buffer.clone();
            assert_eq!(buffer.memory().refcount(), 2);

            drop(buffer2);
            assert_eq!(buffer.memory().refcount(), 1);
        }

        // After all references dropped, owner can reclaim
        arena.reclaim();
        // Slot should be available again (acquire should succeed)
        assert!(arena.acquire().is_some());
    }

    #[test]
    fn test_ipc_ref() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

        let ipc_ref = buffer.memory().ipc_ref();
        assert_eq!(ipc_ref.arena_id, arena.id());
        assert_eq!(ipc_ref.len, 4096);
    }

    #[test]
    fn test_slice_ipc_ref() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

        let sub = buffer.slice(100, 200);
        assert_eq!(sub.len(), 200);

        // IPC ref should reflect the slice
        let ipc_ref = sub.memory().ipc_ref();
        assert_eq!(ipc_ref.len, 200);
    }

    #[test]
    fn test_read_write() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().unwrap();

        // Write via slot
        slot.data_mut()[0..5].copy_from_slice(b"hello");

        let mut handle = MemoryHandle::new(slot);

        // Read via handle
        assert_eq!(&handle.as_slice()[0..5], b"hello");

        // Write via handle
        handle.as_mut_slice()[5..10].copy_from_slice(b"world");

        // Read via buffer
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));
        assert_eq!(&buffer.as_bytes()[0..10], b"helloworld");
    }

    #[test]
    fn test_with_len() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 100);

        assert_eq!(handle.len(), 100);
    }

    #[test]
    fn test_slot_accessor() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(0));

        assert_eq!(buffer.memory().slot().arena_id(), arena.id());
    }

    #[test]
    #[should_panic(expected = "sub-handle exceeds parent bounds")]
    fn test_memory_handle_slice_out_of_bounds() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let _ = handle.slice(900, 200); // 900 + 200 > 1024
    }
}
