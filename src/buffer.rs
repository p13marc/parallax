//! Buffer types for zero-copy data passing.

use crate::memory::{DmaBufSlot, MemoryType, SharedIpcSlotRef, SharedSlotRef};
use crate::metadata::Metadata;
use std::marker::PhantomData;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// Validation state for archived buffers.
const NOT_VALIDATED: u8 = 0;
const VALIDATED_OK: u8 = 1;

/// Handle to a buffer's backing memory (#145): a first-class enum, so ONE
/// uniform [`Buffer`] type flows through the whole pipeline whatever the
/// memory is.
///
/// - [`Cpu`](Self::Cpu): a `SharedArena` slot — refcount in shared memory,
///   cross-process by construction. The overwhelmingly common case; the
///   `new`/`with_len` constructors build it, so CPU call sites never name
///   the variant.
/// - [`DmaBuf`](Self::DmaBuf): a refcounted [`DmaBufSlot`] — a mapped
///   dmabuf fd whose last drop fires the producer's release hook (a V4L2
///   source re-queues the buffer to the driver). The mapping lives as long
///   as the slot, so [`as_slice`](Self::as_slice) is infallible for both
///   variants.
///
/// Cloning is cheap for both: a shared-memory refcount increment (Cpu) or
/// an `Arc` bump (DmaBuf). Arena-specific accessors ([`slot`](Self::slot),
/// [`ipc_ref`](Self::ipc_ref), [`arena_id`](Self::arena_id), …) return
/// `Option` — `None` on a DmaBuf handle.
pub enum MemoryHandle {
    /// A `SharedArena` slot (refcount in shared memory).
    #[non_exhaustive]
    Cpu {
        /// The slot reference.
        slot: SharedSlotRef,
        /// Offset within the slot (for sub-buffers).
        offset: usize,
        /// Length of this buffer's data.
        len: usize,
    },
    /// A refcounted dmabuf slot (mapped fd + release hook).
    #[non_exhaustive]
    DmaBuf {
        /// The shared slot; last clone's drop fires the release hook.
        slot: Arc<DmaBufSlot>,
        /// Offset within the mapping (for sub-buffers).
        offset: usize,
        /// Length of this buffer's data.
        len: usize,
    },
}

impl MemoryHandle {
    /// Create a CPU memory handle from a SharedSlotRef.
    ///
    /// The refcount is in shared memory, enabling cross-process reference counting.
    pub fn new(slot: SharedSlotRef) -> Self {
        let len = slot.len();
        Self::Cpu {
            slot,
            offset: 0,
            len,
        }
    }

    /// Create a CPU memory handle with a specific length.
    ///
    /// Useful when you have a slot but only wrote `len` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `len > slot.len()`.
    pub fn with_len(slot: SharedSlotRef, len: usize) -> Self {
        assert!(len <= slot.len(), "requested length exceeds slot size");
        Self::Cpu {
            slot,
            offset: 0,
            len,
        }
    }

    /// Create a DmaBuf memory handle covering the slot's whole mapping.
    pub fn from_dmabuf(slot: Arc<DmaBufSlot>) -> Self {
        let len = slot.segment().size();
        Self::DmaBuf {
            slot,
            offset: 0,
            len,
        }
    }

    /// Create a DmaBuf memory handle over `offset..offset + len` of the
    /// mapping (a V4L2 frame's `bytesused` is usually smaller than the
    /// exported buffer).
    ///
    /// # Panics
    ///
    /// Panics if `offset + len` exceeds the mapping.
    pub fn from_dmabuf_with_len(slot: Arc<DmaBufSlot>, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= slot.segment().size(),
            "requested range exceeds dmabuf size"
        );
        Self::DmaBuf { slot, offset, len }
    }

    /// Get a pointer to the start of this handle's memory.
    pub fn as_ptr(&self) -> *const u8 {
        match self {
            Self::Cpu { slot, offset, .. } => unsafe { slot.as_ptr().add(*offset) },
            Self::DmaBuf { slot, offset, .. } => unsafe {
                slot.segment().as_slice().as_ptr().add(*offset)
            },
        }
    }

    /// Get a mutable pointer to the start of this handle's memory.
    ///
    /// Private: only [`as_mut_slice`](Self::as_mut_slice) uses it, and only
    /// after the writability check.
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_ptr() as *mut u8
    }

    /// Get the length of this handle's data.
    pub fn len(&self) -> usize {
        match self {
            Self::Cpu { len, .. } | Self::DmaBuf { len, .. } => *len,
        }
    }

    /// Check if this handle has zero length.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get this handle's data as a byte slice.
    ///
    /// Infallible for both variants: a DmaBuf slot keeps its mapping alive
    /// for its whole life.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Get this handle's data as a mutable byte slice.
    ///
    /// `None` for a read-only DmaBuf mapping
    /// ([`DmaBufSegment::from_fd_readonly`](crate::memory::DmaBufSegment::from_fd_readonly)) —
    /// CPU slots are always writable.
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        if let Self::DmaBuf { slot, .. } = self
            && slot.segment().is_read_only()
        {
            return None;
        }
        let len = self.len();
        Some(unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), len) })
    }

    /// Get the memory type of the backing.
    pub fn memory_type(&self) -> MemoryType {
        match self {
            Self::Cpu { .. } => MemoryType::Cpu,
            Self::DmaBuf { .. } => MemoryType::DmaBuf,
        }
    }

    /// Get the offset within the backing slot/mapping.
    pub fn offset(&self) -> usize {
        match self {
            Self::Cpu { offset, .. } | Self::DmaBuf { offset, .. } => *offset,
        }
    }

    /// The underlying arena slot; `None` for a DmaBuf handle.
    pub fn slot(&self) -> Option<&SharedSlotRef> {
        match self {
            Self::Cpu { slot, .. } => Some(slot),
            Self::DmaBuf { .. } => None,
        }
    }

    /// The underlying dmabuf slot; `None` for a Cpu handle.
    pub fn dmabuf_slot(&self) -> Option<&Arc<DmaBufSlot>> {
        match self {
            Self::Cpu { .. } => None,
            Self::DmaBuf { slot, .. } => Some(slot),
        }
    }

    /// The dmabuf fd, for GPU import (#62); `None` for a Cpu handle.
    pub fn dmabuf_fd(&self) -> Option<rustix::fd::BorrowedFd<'_>> {
        self.dmabuf_slot().map(|s| s.fd())
    }

    /// An IPC reference for cross-process sharing; `None` for a DmaBuf
    /// handle (no arena identity — fd passing is the follow-up, see
    /// `IpcSink`'s error).
    pub fn ipc_ref(&self) -> Option<SharedIpcSlotRef> {
        match self {
            Self::Cpu { slot, offset, len } => {
                let base_ref = slot.ipc_ref();
                Some(SharedIpcSlotRef {
                    arena_id: base_ref.arena_id,
                    slot_index: base_ref.slot_index,
                    data_offset: base_ref.data_offset + offset,
                    len: *len,
                })
            }
            Self::DmaBuf { .. } => None,
        }
    }

    /// The arena ID; `None` for a DmaBuf handle.
    pub fn arena_id(&self) -> Option<u64> {
        self.slot().map(|s| s.arena_id())
    }

    /// The arena's raw fd (for IPC); `None` for a DmaBuf handle.
    pub fn arena_fd(&self) -> Option<i32> {
        self.slot().map(|s| s.arena_fd())
    }

    /// The arena's total size (for IPC); `None` for a DmaBuf handle.
    pub fn arena_size(&self) -> Option<usize> {
        self.slot().map(|s| s.arena_size())
    }

    /// Get the current refcount (for debugging). Cpu: the shared-memory
    /// slot refcount; DmaBuf: the `Arc`'s strong count.
    pub fn refcount(&self) -> u32 {
        match self {
            Self::Cpu { slot, .. } => slot.refcount(),
            Self::DmaBuf { slot, .. } => Arc::strong_count(slot) as u32,
        }
    }

    /// Create a sub-handle (a view into a portion of this handle).
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > self.len()`.
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        assert!(
            offset + len <= self.len(),
            "sub-handle exceeds parent bounds"
        );
        match self {
            Self::Cpu {
                slot, offset: base, ..
            } => Self::Cpu {
                slot: slot.slice(base + offset, len),
                offset: 0, // The slice already includes the offset
                len,
            },
            Self::DmaBuf {
                slot, offset: base, ..
            } => Self::DmaBuf {
                slot: Arc::clone(slot),
                offset: base + offset,
                len,
            },
        }
    }
}

impl Clone for MemoryHandle {
    fn clone(&self) -> Self {
        match self {
            Self::Cpu { slot, offset, len } => Self::Cpu {
                slot: slot.clone(), // Increments refcount in shared memory
                offset: *offset,
                len: *len,
            },
            Self::DmaBuf { slot, offset, len } => Self::DmaBuf {
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
            Self::Cpu { slot, offset, len } => f
                .debug_struct("MemoryHandle::Cpu")
                .field("arena_id", &slot.arena_id())
                .field("offset", offset)
                .field("len", len)
                .field("refcount", &slot.refcount())
                .finish(),
            Self::DmaBuf { slot, offset, len } => f
                .debug_struct("MemoryHandle::DmaBuf")
                .field("slot", slot)
                .field("offset", offset)
                .field("len", len)
                .field("refs", &Arc::strong_count(slot))
                .finish(),
        }
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
/// // Clone is O(1): two atomic increments (slot refcount + arena refcount)
/// // in shared memory, plus a Metadata clone. No payload copy, no syscall.
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

    /// Get a mutable reference to the memory handle.
    ///
    /// This allows in-place modification of buffer data, which is
    /// essential for RT-safe elements that must not allocate.
    pub fn memory_mut(&mut self) -> &mut MemoryHandle {
        &mut self.memory
    }

    /// Get the buffer data as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        self.memory.as_slice()
    }

    /// Get the buffer data as a mutable byte slice.
    ///
    /// This allows in-place modification of buffer data without
    /// allocation, which is essential for RT-safe processing.
    ///
    /// # Panics
    ///
    /// Panics on a read-only DmaBuf mapping (#145) — nothing in-tree
    /// produces one into a pipeline; the in-place-mutating elements are CPU
    /// audio paths that never see dmabuf. Use
    /// [`try_as_bytes_mut`](Self::try_as_bytes_mut) where the backing is
    /// not statically known.
    pub fn as_bytes_mut(&mut self) -> &mut [u8] {
        self.memory
            .as_mut_slice()
            .expect("buffer backed by a read-only dmabuf mapping cannot be mutated")
    }

    /// Get the buffer data as a mutable byte slice, or `None` for a
    /// read-only DmaBuf mapping (#145).
    pub fn try_as_bytes_mut(&mut self) -> Option<&mut [u8]> {
        self.memory.as_mut_slice()
    }

    /// Copy this buffer's payload into a fresh CPU arena slot (#145).
    ///
    /// Variant-agnostic replacement for the old `DmaBufBuffer::to_buffer`:
    /// copies exactly [`len`](Self::len) bytes (not the whole mapping) and
    /// carries the metadata over. The bridge for CPU-only consumers of a
    /// dmabuf-producing element — `memorycopy` uses it.
    pub fn copy_to_cpu(&self, arena: &crate::memory::SharedArena) -> crate::error::Result<Self> {
        let mut slot = arena.acquire().ok_or(crate::error::Error::PoolExhausted)?;
        let data = self.as_bytes();
        if data.len() > slot.len() {
            return Err(crate::error::Error::InvalidSegment(format!(
                "arena slot too small: need {} bytes, slot is {}",
                data.len(),
                slot.len()
            )));
        }
        slot.data_mut()[..data.len()].copy_from_slice(data);
        Ok(Self::new(
            MemoryHandle::with_len(slot, data.len()),
            self.metadata.clone(),
        ))
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
        let mut slot = arena.acquire().ok_or(crate::error::Error::PoolExhausted)?;

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

        let ipc_ref = buffer.memory().ipc_ref().expect("cpu handle");
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
        let ipc_ref = sub.memory().ipc_ref().expect("cpu handle");
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
        handle.as_mut_slice().expect("cpu is writable")[5..10].copy_from_slice(b"world");

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

        assert_eq!(
            buffer.memory().slot().expect("cpu handle").arena_id(),
            arena.id()
        );
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

#[cfg(test)]
mod dmabuf_handle_tests {
    use super::*;
    use crate::memory::{DmaBufSegment, DmaBufSlot};
    use std::sync::atomic::{AtomicU32, Ordering as AtomicOrdering};

    fn memfd_segment(len: u64) -> DmaBufSegment {
        let fd = rustix::fs::memfd_create("buffer_dmabuf_test", rustix::fs::MemfdFlags::CLOEXEC)
            .unwrap();
        rustix::fs::ftruncate(&fd, len).unwrap();
        DmaBufSegment::from_fd(fd, len as usize).unwrap()
    }

    fn dmabuf_buffer(len: u64) -> Buffer {
        let slot = Arc::new(DmaBufSlot::new(memfd_segment(len)));
        Buffer::new(MemoryHandle::from_dmabuf(slot), Metadata::from_sequence(0))
    }

    #[test]
    fn dmabuf_handle_reports_its_type_and_no_arena() {
        let buf = dmabuf_buffer(4096);
        assert_eq!(buf.memory_type(), MemoryType::DmaBuf);
        assert!(buf.memory().slot().is_none());
        assert!(buf.memory().ipc_ref().is_none());
        assert!(buf.memory().arena_id().is_none());
        assert!(buf.memory().dmabuf_slot().is_some());
        assert!(buf.memory().dmabuf_fd().is_some());
    }

    #[test]
    fn clone_shares_the_slot_and_drop_releases_once() {
        let fired = Arc::new(AtomicU32::new(0));
        let hook_fired = fired.clone();
        let segment = Arc::new(memfd_segment(1024));
        let slot = Arc::new(DmaBufSlot::with_release(
            segment,
            3,
            Box::new(move |_| {
                hook_fired.fetch_add(1, AtomicOrdering::SeqCst);
            }),
        ));
        let buf = Buffer::<()>::new(
            MemoryHandle::from_dmabuf_with_len(slot, 0, 512),
            Metadata::from_sequence(0),
        );
        assert_eq!(buf.len(), 512);
        let clone = buf.clone();
        assert_eq!(buf.memory().refcount(), 2);
        drop(buf);
        assert_eq!(fired.load(AtomicOrdering::SeqCst), 0);
        drop(clone);
        assert_eq!(fired.load(AtomicOrdering::SeqCst), 1);
    }

    #[test]
    fn dmabuf_bytes_are_readable_and_writable() {
        let mut buf = dmabuf_buffer(1024);
        buf.as_bytes_mut()[..5].copy_from_slice(b"hello");
        assert_eq!(&buf.as_bytes()[..5], b"hello");
        assert!(buf.try_as_bytes_mut().is_some());
    }

    #[test]
    fn readonly_mapping_refuses_mutation() {
        let fd = rustix::fs::memfd_create("ro_dmabuf", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        rustix::fs::ftruncate(&fd, 1024).unwrap();
        let segment = DmaBufSegment::from_fd_readonly(fd, 1024).unwrap();
        let mut buf = Buffer::<()>::new(
            MemoryHandle::from_dmabuf(Arc::new(DmaBufSlot::new(segment))),
            Metadata::from_sequence(0),
        );
        assert!(buf.try_as_bytes_mut().is_none());
    }

    #[test]
    fn slice_offsets_into_the_mapping() {
        let mut buf = dmabuf_buffer(1024);
        buf.as_bytes_mut()[100..105].copy_from_slice(b"world");
        let sub = buf.slice(100, 5);
        assert_eq!(sub.as_bytes(), b"world");
        assert_eq!(sub.memory_type(), MemoryType::DmaBuf);
    }

    #[test]
    fn copy_to_cpu_copies_exactly_len_bytes() {
        let arena = crate::memory::SharedArena::new(4096, 4).unwrap();
        let mut buf = dmabuf_buffer(4096);
        buf.as_bytes_mut()[..4].copy_from_slice(b"data");
        let sub = buf.slice(0, 4);
        let cpu = sub.copy_to_cpu(&arena).unwrap();
        assert_eq!(cpu.memory_type(), MemoryType::Cpu);
        assert_eq!(cpu.len(), 4, "len(), not the whole mapping");
        assert_eq!(cpu.as_bytes(), b"data");
    }
}
