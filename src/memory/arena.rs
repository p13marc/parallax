//! Arena-based memory allocation for efficient buffer pools.
//!
//! This module provides `CpuArena`, which allocates a large memfd region
//! and subdivides it into fixed-size slots. This solves the fd limit problem:
//! instead of one fd per buffer, we have one fd per arena.
//!
//! # Design Rationale
//!
//! Linux has a default limit of 1024 file descriptors per process. With the
//! previous design (one memfd per buffer), a pipeline with many in-flight
//! buffers could exhaust this limit. With arena allocation:
//!
//! - 1 fd per arena (e.g., 256MB arena)
//! - Many slots per arena (e.g., 4096 slots of 64KB each)
//! - fd usage = O(arenas) ≈ O(pipelines), not O(buffers)
//!
//! # Cross-Process Sharing
//!
//! When sharing buffers across processes:
//!
//! 1. First buffer from arena: send arena fd + slot offset
//! 2. Subsequent buffers: just send offset (receiver caches the mmap)
//!
//! This is done via [`IpcSlotRef`] which contains arena_id + offset.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::{CpuArena, ArenaSlot};
//!
//! // Create arena with 16 slots of 64KB each (1MB total, 1 fd)
//! let arena = CpuArena::new(64 * 1024, 16)?;
//!
//! // Acquire a slot
//! let mut slot = arena.acquire().expect("arena not exhausted");
//!
//! // Write data
//! slot.as_mut_slice()[..5].copy_from_slice(b"hello");
//!
//! // Get IPC reference for cross-process sharing
//! let ipc_ref = slot.ipc_ref();
//! // Send ipc_ref over Unix socket...
//!
//! // Slot is returned to arena when dropped
//! ```

use super::{IpcHandle, MemorySegment, MemoryType};
use crate::error::{Error, Result};
use crate::memory::AtomicBitmap;
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::mm::{MapFlags, ProtFlags};
use std::ffi::CString;
use std::os::unix::io::{AsRawFd, RawFd};
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique arena IDs.
static ARENA_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique arena ID.
fn next_arena_id() -> u64 {
    ARENA_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Arena of CPU memory slots (single fd for entire pool).
///
/// This is the most efficient way to allocate buffers for pipelines:
/// - One memfd for the entire arena
/// - Lock-free slot allocation via atomic bitmap
/// - RAII slots that return to pool on drop
///
/// # Memory Layout
///
/// ```text
/// ┌─────────┬─────────┬─────────┬─────────┬─────────┐
/// │  Slot 0 │  Slot 1 │  Slot 2 │   ...   │ Slot N  │
/// └─────────┴─────────┴─────────┴─────────┴─────────┘
/// ^                                                  ^
/// base                                    base + total_size
/// ```
///
/// Each slot is `slot_size` bytes, starting at `base + (index * slot_size)`.
pub struct CpuArena {
    /// The memfd file descriptor (ONE fd for entire arena).
    fd: OwnedFd,
    /// Base pointer to the mmap'd region.
    base: NonNull<u8>,
    /// Total size of the arena in bytes.
    total_size: usize,
    /// Size of each slot in bytes.
    slot_size: usize,
    /// Number of slots.
    slot_count: usize,
    /// Bitmap tracking which slots are free (1 = free, 0 = in use).
    free_slots: AtomicBitmap,
    /// Unique ID for this arena (for cross-process identification).
    arena_id: u64,
    /// Optional debug name.
    name: Option<String>,
}

impl CpuArena {
    /// Create a new arena with fixed-size slots.
    ///
    /// # Arguments
    ///
    /// * `slot_size` - Size of each slot in bytes. Must be > 0.
    /// * `slot_count` - Number of slots. Must be > 0.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // 16 slots of 64KB each = 1MB arena, 1 fd
    /// let arena = CpuArena::new(64 * 1024, 16)?;
    /// ```
    pub fn new(slot_size: usize, slot_count: usize) -> Result<Arc<Self>> {
        Self::with_name("parallax-arena", slot_size, slot_count)
    }

    /// Create a new arena with a debug name.
    pub fn with_name(name: &str, slot_size: usize, slot_count: usize) -> Result<Arc<Self>> {
        if slot_size == 0 {
            return Err(Error::AllocationFailed("slot_size must be > 0".into()));
        }
        if slot_count == 0 {
            return Err(Error::AllocationFailed("slot_count must be > 0".into()));
        }

        let total_size = slot_size
            .checked_mul(slot_count)
            .ok_or_else(|| Error::AllocationFailed("arena size overflow".into()))?;

        // Create memfd
        let cname = CString::new(name).map_err(|e| Error::AllocationFailed(e.to_string()))?;
        let fd = rustix::fs::memfd_create(&cname, rustix::fs::MemfdFlags::CLOEXEC)?;

        // Set size
        rustix::fs::ftruncate(&fd, total_size as u64)?;

        // Map the entire region
        let base = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                total_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &fd,
                0,
            )?
        };

        let base = NonNull::new(base.cast::<u8>())
            .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;

        // Create bitmap with all slots free (0 = free, 1 = allocated)
        let free_slots = AtomicBitmap::new(slot_count);
        // AtomicBitmap starts with all slots free (all bits = 0)

        Ok(Arc::new(Self {
            fd,
            base,
            total_size,
            slot_size,
            slot_count,
            free_slots,
            arena_id: next_arena_id(),
            name: Some(name.to_string()),
        }))
    }

    /// Acquire a slot from the arena.
    ///
    /// Returns `None` if all slots are in use.
    /// The slot is automatically returned when dropped.
    pub fn acquire(self: &Arc<Self>) -> Option<ArenaSlot> {
        // Find a free slot and mark it as allocated
        let index = self.free_slots.acquire_slot()?;

        Some(ArenaSlot {
            arena: Arc::clone(self),
            index,
            offset: index * self.slot_size,
            len: self.slot_size,
        })
    }

    /// Try to acquire multiple slots at once.
    ///
    /// Returns `None` if not enough slots are available.
    /// All-or-nothing: either all slots are acquired or none.
    pub fn acquire_many(self: &Arc<Self>, count: usize) -> Option<Vec<ArenaSlot>> {
        let mut slots = Vec::with_capacity(count);

        for _ in 0..count {
            match self.acquire() {
                Some(slot) => slots.push(slot),
                None => {
                    // Not enough slots - return all acquired slots
                    // (they'll be released on drop)
                    return None;
                }
            }
        }

        Some(slots)
    }

    /// Get the unique arena ID.
    ///
    /// This is used for cross-process identification.
    #[inline]
    pub fn id(&self) -> u64 {
        self.arena_id
    }

    /// Get the file descriptor for sharing with other processes.
    #[inline]
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// Get the raw file descriptor.
    #[inline]
    pub fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Get the slot size.
    #[inline]
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Get the total number of slots.
    #[inline]
    pub fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Get the number of free slots.
    #[inline]
    pub fn free_count(&self) -> usize {
        self.free_slots.count_free()
    }

    /// Get the number of used slots.
    #[inline]
    pub fn used_count(&self) -> usize {
        self.slot_count - self.free_count()
    }

    /// Get the total arena size in bytes.
    #[inline]
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Get the debug name.
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Pre-fault all memory to avoid page faults during use.
    pub fn prefault(&self) {
        let page_size = 4096;
        let ptr = self.base.as_ptr();
        for offset in (0..self.total_size).step_by(page_size) {
            unsafe {
                std::ptr::read_volatile(ptr.add(offset));
            }
        }
    }

    /// Release a slot back to the arena (internal).
    fn release(&self, index: usize) {
        debug_assert!(index < self.slot_count);
        self.free_slots.release_slot(index);
    }

    /// Get a raw pointer to a slot (for reconstruction from IpcSlotRef).
    ///
    /// # Safety
    ///
    /// The caller must ensure `offset` is valid (aligned to slot boundaries).
    #[inline]
    pub unsafe fn slot_ptr(&self, offset: usize) -> *mut u8 {
        debug_assert!(offset < self.total_size);
        // SAFETY: Caller guarantees offset is valid.
        unsafe { self.base.as_ptr().add(offset) }
    }
}

impl Drop for CpuArena {
    fn drop(&mut self) {
        unsafe {
            let _ = rustix::mm::munmap(self.base.as_ptr().cast(), self.total_size);
        }
    }
}

// SAFETY: CpuArena is Send + Sync because:
// - Memory access is through atomic bitmap
// - The fd is kernel-reference-counted
// - No thread-local state
unsafe impl Send for CpuArena {}
unsafe impl Sync for CpuArena {}

impl AsFd for CpuArena {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

/// A slot within an arena (RAII guard, returns to arena on drop).
///
/// This is the handle you get when acquiring a slot from an arena.
/// When dropped, the slot is automatically returned to the arena for reuse.
pub struct ArenaSlot {
    /// Reference to the owning arena.
    arena: Arc<CpuArena>,
    /// Index of this slot in the arena.
    index: usize,
    /// Byte offset from arena base.
    offset: usize,
    /// Size of this slot (same as arena.slot_size).
    len: usize,
}

impl ArenaSlot {
    /// Get the slot data as a byte slice.
    #[inline]
    pub fn data(&self) -> &[u8] {
        // SAFETY: self.offset and self.len are valid by construction
        unsafe { std::slice::from_raw_parts(self.arena.base.as_ptr().add(self.offset), self.len) }
    }

    /// Get the slot data as a mutable byte slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        // SAFETY: self.offset and self.len are valid by construction,
        // and &mut self guarantees exclusive access
        unsafe {
            std::slice::from_raw_parts_mut(self.arena.base.as_ptr().add(self.offset), self.len)
        }
    }

    /// Get an IPC reference for cross-process sharing.
    ///
    /// The receiver can use this to access the same memory region
    /// (after mapping the arena fd). Defaults to read-write access.
    #[inline]
    pub fn ipc_ref(&self) -> IpcSlotRef {
        IpcSlotRef {
            arena_id: self.arena.arena_id,
            offset: self.offset,
            len: self.len,
            access: Access::ReadWrite,
        }
    }

    /// Get an IPC reference with specific access rights.
    ///
    /// Use this when you need to restrict access:
    /// - `Access::ReadOnly` for downstream consumers
    /// - `Access::WriteOnly` for upstream producers
    /// - `Access::ReadWrite` for transforms
    #[inline]
    pub fn ipc_ref_with_access(&self, access: Access) -> IpcSlotRef {
        IpcSlotRef {
            arena_id: self.arena.arena_id,
            offset: self.offset,
            len: self.len,
            access,
        }
    }

    /// Get the arena this slot belongs to.
    #[inline]
    pub fn arena(&self) -> &Arc<CpuArena> {
        &self.arena
    }

    /// Get the slot index.
    #[inline]
    pub fn index(&self) -> usize {
        self.index
    }

    /// Get the byte offset from arena base.
    #[inline]
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the slot size.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the slot is empty (always false, slots have fixed size).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the raw pointer to the slot data.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        unsafe { self.arena.base.as_ptr().add(self.offset) }
    }

    /// Get the mutable raw pointer to the slot data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        unsafe { self.arena.base.as_ptr().add(self.offset) }
    }
}

impl Drop for ArenaSlot {
    fn drop(&mut self) {
        self.arena.release(self.index);
    }
}

impl MemorySegment for ArenaSlot {
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        ArenaSlot::as_ptr(self)
    }

    #[inline]
    fn as_mut_ptr(&self) -> Option<*mut u8> {
        Some(unsafe { self.arena.base.as_ptr().add(self.offset) })
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn memory_type(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn ipc_handle(&self) -> Option<IpcHandle> {
        Some(IpcHandle::Fd {
            fd: self.arena.raw_fd(),
            size: self.arena.total_size,
        })
    }
}

// SAFETY: ArenaSlot is Send + Sync because the underlying memory is shared
// and thread-safety is managed by the arena's atomic bitmap.
unsafe impl Send for ArenaSlot {}
unsafe impl Sync for ArenaSlot {}

/// Access rights for shared memory regions.
///
/// These rights control how a process can access a shared memory slot.
/// The supervisor enforces these based on data flow direction:
/// - Sources produce data → WriteOnly for producer, ReadOnly for consumers
/// - Sinks consume data → ReadOnly access
/// - Transforms need both → ReadWrite (or separate read/write regions)
#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    PartialEq,
    Eq,
    Hash,
    rkyv::Archive,
    rkyv::Serialize,
    rkyv::Deserialize,
)]
#[rkyv(derive(Debug))]
#[repr(u8)]
pub enum Access {
    /// Read-only access (PROT_READ).
    ///
    /// Used for downstream elements that only consume buffer data.
    ReadOnly = 0,

    /// Write-only access (PROT_WRITE).
    ///
    /// Used for sources that produce data into buffers.
    /// Note: On most architectures, write implies read for the CPU.
    WriteOnly = 1,

    /// Read-write access (PROT_READ | PROT_WRITE).
    ///
    /// Used for transforms that read input and write output.
    #[default]
    ReadWrite = 2,
}

impl Access {
    /// Convert to mmap protection flags.
    pub fn to_prot_flags(self) -> ProtFlags {
        match self {
            Access::ReadOnly => ProtFlags::READ,
            Access::WriteOnly => ProtFlags::WRITE,
            Access::ReadWrite => ProtFlags::READ | ProtFlags::WRITE,
        }
    }

    /// Check if this access level allows reading.
    #[inline]
    pub fn can_read(self) -> bool {
        matches!(self, Access::ReadOnly | Access::ReadWrite)
    }

    /// Check if this access level allows writing.
    #[inline]
    pub fn can_write(self) -> bool {
        matches!(self, Access::WriteOnly | Access::ReadWrite)
    }

    /// Create from a role in the pipeline.
    ///
    /// - Sources/producers → WriteOnly (they produce data)
    /// - Sinks/consumers → ReadOnly (they consume data)
    /// - Transforms → ReadWrite (they read and write)
    pub fn for_role(is_producer: bool, is_consumer: bool) -> Self {
        match (is_producer, is_consumer) {
            (true, true) => Access::ReadWrite,
            (true, false) => Access::WriteOnly,
            (false, true) => Access::ReadOnly,
            (false, false) => Access::ReadOnly, // Default to read-only for safety
        }
    }
}

impl std::fmt::Display for Access {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Access::ReadOnly => write!(f, "r--"),
            Access::WriteOnly => write!(f, "-w-"),
            Access::ReadWrite => write!(f, "rw-"),
        }
    }
}

/// Cross-process slot reference (serializable).
///
/// This is sent over IPC channels to reference a buffer in shared memory.
/// The receiver uses `arena_id` to look up the cached mmap and `offset`
/// to locate the data.
///
/// # Wire Format
///
/// ```text
/// ┌──────────────────┬──────────────────┬──────────────────┬──────────────────┐
/// │    arena_id      │      offset      │       len        │     access       │
/// │     (8 bytes)    │     (8 bytes)    │     (8 bytes)    │    (1 byte)      │
/// └──────────────────┴──────────────────┴──────────────────┴──────────────────┘
/// ```
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[rkyv(derive(Debug))]
pub struct IpcSlotRef {
    /// Unique ID of the arena.
    pub arena_id: u64,
    /// Byte offset within the arena.
    pub offset: usize,
    /// Size of the slot data.
    pub len: usize,
    /// Access rights for this slot reference.
    pub access: Access,
}

impl IpcSlotRef {
    /// Create a new IPC slot reference with read-write access.
    pub const fn new(arena_id: u64, offset: usize, len: usize) -> Self {
        Self {
            arena_id,
            offset,
            len,
            access: Access::ReadWrite,
        }
    }

    /// Create a new IPC slot reference with specific access rights.
    pub const fn with_access(arena_id: u64, offset: usize, len: usize, access: Access) -> Self {
        Self {
            arena_id,
            offset,
            len,
            access,
        }
    }

    /// Create a read-only reference to this slot.
    ///
    /// Used when passing buffers downstream for consumption.
    pub const fn as_read_only(&self) -> Self {
        Self {
            arena_id: self.arena_id,
            offset: self.offset,
            len: self.len,
            access: Access::ReadOnly,
        }
    }

    /// Create a write-only reference to this slot.
    ///
    /// Used when granting a producer access to write data.
    pub const fn as_write_only(&self) -> Self {
        Self {
            arena_id: self.arena_id,
            offset: self.offset,
            len: self.len,
            access: Access::WriteOnly,
        }
    }

    /// Check if this reference allows reading.
    #[inline]
    pub fn can_read(&self) -> bool {
        self.access.can_read()
    }

    /// Check if this reference allows writing.
    #[inline]
    pub fn can_write(&self) -> bool {
        self.access.can_write()
    }
}

/// Cache for mapping received arena fds.
///
/// When receiving buffers from another process, we cache the mmap
/// so we only map each arena once. Each arena can have multiple mappings
/// with different access rights.
pub struct ArenaCache {
    /// Cached arena mappings: (arena_id, access) -> mapping
    mappings: std::collections::HashMap<(u64, Access), CachedArenaMapping>,
}

struct CachedArenaMapping {
    base: NonNull<u8>,
    size: usize,
}

impl ArenaCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            mappings: std::collections::HashMap::new(),
        }
    }

    /// Map an arena fd with read-write access and cache it.
    ///
    /// # Safety
    ///
    /// The caller must ensure `fd` is a valid arena fd and `size` is correct.
    pub unsafe fn map_arena(
        &mut self,
        arena_id: u64,
        fd: BorrowedFd<'_>,
        size: usize,
    ) -> Result<()> {
        // SAFETY: Caller guarantees fd and size are valid, we just forward to with_access
        unsafe { self.map_arena_with_access(arena_id, fd, size, Access::ReadWrite) }
    }

    /// Map an arena fd with specific access rights and cache it.
    ///
    /// # Safety
    ///
    /// The caller must ensure `fd` is a valid arena fd and `size` is correct.
    pub unsafe fn map_arena_with_access(
        &mut self,
        arena_id: u64,
        fd: BorrowedFd<'_>,
        size: usize,
        access: Access,
    ) -> Result<()> {
        let key = (arena_id, access);
        if self.mappings.contains_key(&key) {
            return Ok(()); // Already mapped with this access level
        }

        let prot = access.to_prot_flags();

        let base =
            unsafe { rustix::mm::mmap(std::ptr::null_mut(), size, prot, MapFlags::SHARED, fd, 0)? };

        let base = NonNull::new(base.cast::<u8>())
            .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;

        self.mappings.insert(key, CachedArenaMapping { base, size });

        Ok(())
    }

    /// Get a slice for an IPC slot reference.
    ///
    /// Returns `None` if the arena hasn't been mapped yet or if the slot
    /// doesn't have read access.
    pub fn get_slice(&self, slot_ref: &IpcSlotRef) -> Option<&[u8]> {
        // Check access rights
        if !slot_ref.access.can_read() {
            return None;
        }

        // Try to find a mapping with compatible access
        let mapping = self.find_mapping(slot_ref.arena_id, slot_ref.access)?;
        if slot_ref.offset + slot_ref.len > mapping.size {
            return None; // Out of bounds
        }
        Some(unsafe {
            std::slice::from_raw_parts(mapping.base.as_ptr().add(slot_ref.offset), slot_ref.len)
        })
    }

    /// Get a mutable slice for an IPC slot reference.
    ///
    /// Returns `None` if the arena hasn't been mapped yet or if the slot
    /// doesn't have write access.
    ///
    /// # Safety
    ///
    /// The caller must ensure exclusive access to this memory region.
    #[allow(clippy::mut_from_ref)] // Interior mutability via mmap is intentional
    pub unsafe fn get_mut_slice(&self, slot_ref: &IpcSlotRef) -> Option<&mut [u8]> {
        // Check access rights
        if !slot_ref.access.can_write() {
            return None;
        }

        // Try to find a mapping with compatible access
        let mapping = self.find_mapping(slot_ref.arena_id, slot_ref.access)?;
        if slot_ref.offset + slot_ref.len > mapping.size {
            return None; // Out of bounds
        }
        Some(unsafe {
            std::slice::from_raw_parts_mut(mapping.base.as_ptr().add(slot_ref.offset), slot_ref.len)
        })
    }

    /// Find a mapping with compatible access rights.
    fn find_mapping(&self, arena_id: u64, access: Access) -> Option<&CachedArenaMapping> {
        // First try exact match
        if let Some(mapping) = self.mappings.get(&(arena_id, access)) {
            return Some(mapping);
        }

        // ReadWrite mapping is compatible with any access
        if let Some(mapping) = self.mappings.get(&(arena_id, Access::ReadWrite)) {
            return Some(mapping);
        }

        // For ReadOnly access, a ReadWrite mapping also works (already checked above)
        // For WriteOnly access, a ReadWrite mapping also works (already checked above)
        None
    }

    /// Check if an arena is already mapped.
    pub fn is_mapped(&self, arena_id: u64) -> bool {
        self.mappings.keys().any(|(id, _)| *id == arena_id)
    }

    /// Check if an arena is mapped with specific access.
    pub fn is_mapped_with_access(&self, arena_id: u64, access: Access) -> bool {
        self.find_mapping(arena_id, access).is_some()
    }

    /// Remove all cached mappings for an arena.
    pub fn unmap(&mut self, arena_id: u64) {
        let keys: Vec<_> = self
            .mappings
            .keys()
            .filter(|(id, _)| *id == arena_id)
            .copied()
            .collect();

        for key in keys {
            if let Some(mapping) = self.mappings.remove(&key) {
                unsafe {
                    let _ = rustix::mm::munmap(mapping.base.as_ptr().cast(), mapping.size);
                }
            }
        }
    }

    /// Remove a specific mapping with given access.
    pub fn unmap_with_access(&mut self, arena_id: u64, access: Access) {
        if let Some(mapping) = self.mappings.remove(&(arena_id, access)) {
            unsafe {
                let _ = rustix::mm::munmap(mapping.base.as_ptr().cast(), mapping.size);
            }
        }
    }

    /// Clear all cached mappings.
    pub fn clear(&mut self) {
        for (_, mapping) in self.mappings.drain() {
            unsafe {
                let _ = rustix::mm::munmap(mapping.base.as_ptr().cast(), mapping.size);
            }
        }
    }
}

impl Default for ArenaCache {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ArenaCache {
    fn drop(&mut self) {
        self.clear();
    }
}

// SAFETY: CachedArenaMapping is Send + Sync
unsafe impl Send for CachedArenaMapping {}
unsafe impl Sync for CachedArenaMapping {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_access_properties() {
        assert!(Access::ReadOnly.can_read());
        assert!(!Access::ReadOnly.can_write());

        assert!(!Access::WriteOnly.can_read());
        assert!(Access::WriteOnly.can_write());

        assert!(Access::ReadWrite.can_read());
        assert!(Access::ReadWrite.can_write());
    }

    #[test]
    fn test_access_for_role() {
        assert_eq!(Access::for_role(true, false), Access::WriteOnly); // Producer
        assert_eq!(Access::for_role(false, true), Access::ReadOnly); // Consumer
        assert_eq!(Access::for_role(true, true), Access::ReadWrite); // Transform
        assert_eq!(Access::for_role(false, false), Access::ReadOnly); // Default
    }

    #[test]
    fn test_access_display() {
        assert_eq!(format!("{}", Access::ReadOnly), "r--");
        assert_eq!(format!("{}", Access::WriteOnly), "-w-");
        assert_eq!(format!("{}", Access::ReadWrite), "rw-");
    }

    #[test]
    fn test_ipc_slot_ref_access_conversion() {
        let slot_ref = IpcSlotRef::new(1, 0, 1024);
        assert_eq!(slot_ref.access, Access::ReadWrite);

        let readonly = slot_ref.as_read_only();
        assert_eq!(readonly.access, Access::ReadOnly);
        assert!(readonly.can_read());
        assert!(!readonly.can_write());

        let writeonly = slot_ref.as_write_only();
        assert_eq!(writeonly.access, Access::WriteOnly);
        assert!(!writeonly.can_read());
        assert!(writeonly.can_write());
    }

    #[test]
    fn test_ipc_slot_ref_with_access() {
        let slot_ref = IpcSlotRef::with_access(1, 0, 1024, Access::ReadOnly);
        assert_eq!(slot_ref.access, Access::ReadOnly);
        assert!(slot_ref.can_read());
        assert!(!slot_ref.can_write());
    }

    #[test]
    fn test_arena_creation() {
        let arena = CpuArena::new(4096, 16).unwrap();
        assert_eq!(arena.slot_size(), 4096);
        assert_eq!(arena.slot_count(), 16);
        assert_eq!(arena.total_size(), 4096 * 16);
        assert_eq!(arena.free_count(), 16);
        assert_eq!(arena.used_count(), 0);
    }

    #[test]
    fn test_arena_with_name() {
        let arena = CpuArena::with_name("test-arena", 1024, 8).unwrap();
        assert_eq!(arena.name(), Some("test-arena"));
    }

    #[test]
    fn test_arena_zero_slot_size_fails() {
        assert!(CpuArena::new(0, 16).is_err());
    }

    #[test]
    fn test_arena_zero_slot_count_fails() {
        assert!(CpuArena::new(4096, 0).is_err());
    }

    #[test]
    fn test_arena_acquire_release() {
        let arena = CpuArena::new(4096, 4).unwrap();
        assert_eq!(arena.free_count(), 4);

        let slot1 = arena.acquire().unwrap();
        assert_eq!(arena.free_count(), 3);
        assert_eq!(arena.used_count(), 1);

        let slot2 = arena.acquire().unwrap();
        assert_eq!(arena.free_count(), 2);

        drop(slot1);
        assert_eq!(arena.free_count(), 3);

        drop(slot2);
        assert_eq!(arena.free_count(), 4);
    }

    #[test]
    fn test_arena_exhaustion() {
        let arena = CpuArena::new(4096, 2).unwrap();

        let _slot1 = arena.acquire().unwrap();
        let _slot2 = arena.acquire().unwrap();

        // Arena is full
        assert!(arena.acquire().is_none());
    }

    #[test]
    fn test_slot_read_write() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().unwrap();

        // Write data
        slot.data_mut()[0] = 42;
        slot.data_mut()[1] = 43;
        slot.data_mut()[4095] = 99;

        // Read back
        assert_eq!(slot.data()[0], 42);
        assert_eq!(slot.data()[1], 43);
        assert_eq!(slot.data()[4095], 99);
    }

    #[test]
    fn test_slot_ipc_ref() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();

        let ipc_ref = slot.ipc_ref();
        assert_eq!(ipc_ref.arena_id, arena.id());
        assert_eq!(ipc_ref.offset, slot.offset());
        assert_eq!(ipc_ref.len, slot.len());
    }

    #[test]
    fn test_multiple_slots_different_offsets() {
        let arena = CpuArena::new(4096, 4).unwrap();

        let slot0 = arena.acquire().unwrap();
        let slot1 = arena.acquire().unwrap();
        let slot2 = arena.acquire().unwrap();

        // Slots should have different offsets
        let offsets: std::collections::HashSet<_> =
            [slot0.offset(), slot1.offset(), slot2.offset()]
                .into_iter()
                .collect();
        assert_eq!(offsets.len(), 3);

        // Offsets should be multiples of slot_size
        assert_eq!(slot0.offset() % 4096, 0);
        assert_eq!(slot1.offset() % 4096, 0);
        assert_eq!(slot2.offset() % 4096, 0);
    }

    #[test]
    fn test_acquire_many() {
        let arena = CpuArena::new(4096, 8).unwrap();

        // Acquire 4 slots
        let slots = arena.acquire_many(4).unwrap();
        assert_eq!(slots.len(), 4);
        assert_eq!(arena.free_count(), 4);

        // Try to acquire 5 more (should fail, only 4 available)
        assert!(arena.acquire_many(5).is_none());
        // The 4 slots should still be free (all-or-nothing)
        assert_eq!(arena.free_count(), 4);

        // Acquire remaining 4
        let more_slots = arena.acquire_many(4).unwrap();
        assert_eq!(more_slots.len(), 4);
        assert_eq!(arena.free_count(), 0);
    }

    #[test]
    fn test_arena_prefault() {
        let arena = CpuArena::new(4096, 4).unwrap();
        arena.prefault(); // Should not panic
    }

    #[test]
    fn test_arena_unique_ids() {
        let arena1 = CpuArena::new(4096, 4).unwrap();
        let arena2 = CpuArena::new(4096, 4).unwrap();
        let arena3 = CpuArena::new(4096, 4).unwrap();

        assert_ne!(arena1.id(), arena2.id());
        assert_ne!(arena2.id(), arena3.id());
        assert_ne!(arena1.id(), arena3.id());
    }

    #[test]
    fn test_slot_implements_memory_segment() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();

        assert_eq!(slot.memory_type(), MemoryType::Cpu);
        assert!(slot.ipc_handle().is_some());
        assert_eq!(MemorySegment::len(&slot), 4096);
    }

    #[test]
    fn test_arena_cache() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().unwrap();

        // Write some data
        slot.data_mut()[0] = 123;
        slot.data_mut()[100] = 234;

        let ipc_ref = slot.ipc_ref();

        // Simulate receiving in another "process" (same process for test)
        let mut cache = ArenaCache::new();
        assert!(!cache.is_mapped(arena.id()));

        // Map the arena
        unsafe {
            cache
                .map_arena(arena.id(), arena.fd(), arena.total_size())
                .unwrap();
        }
        assert!(cache.is_mapped(arena.id()));

        // Read via cache
        let slice = cache.get_slice(&ipc_ref).unwrap();
        assert_eq!(slice[0], 123);
        assert_eq!(slice[100], 234);
    }

    #[test]
    fn test_arena_cache_bounds_check() {
        let arena = CpuArena::new(4096, 4).unwrap();

        let mut cache = ArenaCache::new();
        unsafe {
            cache
                .map_arena(arena.id(), arena.fd(), arena.total_size())
                .unwrap();
        }

        // Valid reference
        let valid_ref = IpcSlotRef::new(arena.id(), 0, 4096);
        assert!(cache.get_slice(&valid_ref).is_some());

        // Out of bounds
        let invalid_ref = IpcSlotRef::new(arena.id(), arena.total_size(), 1);
        assert!(cache.get_slice(&invalid_ref).is_none());

        // Unknown arena
        let unknown_ref = IpcSlotRef::new(999999, 0, 4096);
        assert!(cache.get_slice(&unknown_ref).is_none());
    }

    #[test]
    fn test_arena_cache_access_control() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().unwrap();

        // Write data
        slot.data_mut()[0] = 42;

        let ipc_ref = slot.ipc_ref();

        // Map with read-write access
        let mut cache = ArenaCache::new();
        unsafe {
            cache
                .map_arena(arena.id(), arena.fd(), arena.total_size())
                .unwrap();
        }

        // Read-write reference should work for both read and write
        assert!(cache.get_slice(&ipc_ref).is_some());
        unsafe {
            assert!(cache.get_mut_slice(&ipc_ref).is_some());
        }

        // Read-only reference should only allow read
        let readonly_ref = ipc_ref.as_read_only();
        assert!(cache.get_slice(&readonly_ref).is_some());
        unsafe {
            assert!(cache.get_mut_slice(&readonly_ref).is_none());
        }

        // Write-only reference should only allow write
        let writeonly_ref = ipc_ref.as_write_only();
        assert!(cache.get_slice(&writeonly_ref).is_none());
        unsafe {
            assert!(cache.get_mut_slice(&writeonly_ref).is_some());
        }
    }

    #[test]
    fn test_arena_cache_readonly_mapping() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[0] = 99;

        let ipc_ref = slot.ipc_ref_with_access(Access::ReadOnly);

        // Map with read-only access
        let mut cache = ArenaCache::new();
        unsafe {
            cache
                .map_arena_with_access(arena.id(), arena.fd(), arena.total_size(), Access::ReadOnly)
                .unwrap();
        }

        // Should be able to read
        let slice = cache.get_slice(&ipc_ref).unwrap();
        assert_eq!(slice[0], 99);

        // Should NOT be able to get mutable slice (access denied)
        // Note: The underlying mmap is read-only, so this correctly fails
        unsafe {
            // The slot ref is read-only, so get_mut_slice returns None
            assert!(cache.get_mut_slice(&ipc_ref).is_none());
        }
    }

    #[test]
    fn test_arena_slot_ipc_ref_with_access() {
        let arena = CpuArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();

        // Default ipc_ref should be read-write
        let default_ref = slot.ipc_ref();
        assert_eq!(default_ref.access, Access::ReadWrite);

        // ipc_ref_with_access should respect the given access
        let readonly_ref = slot.ipc_ref_with_access(Access::ReadOnly);
        assert_eq!(readonly_ref.access, Access::ReadOnly);

        let writeonly_ref = slot.ipc_ref_with_access(Access::WriteOnly);
        assert_eq!(writeonly_ref.access, Access::WriteOnly);
    }

    #[test]
    fn test_arena_cache_multiple_access_levels() {
        let arena = CpuArena::new(4096, 4).unwrap();

        let mut cache = ArenaCache::new();

        // Map with both read-only and read-write access
        unsafe {
            cache
                .map_arena_with_access(arena.id(), arena.fd(), arena.total_size(), Access::ReadOnly)
                .unwrap();
            cache
                .map_arena_with_access(
                    arena.id(),
                    arena.fd(),
                    arena.total_size(),
                    Access::ReadWrite,
                )
                .unwrap();
        }

        assert!(cache.is_mapped(arena.id()));
        assert!(cache.is_mapped_with_access(arena.id(), Access::ReadOnly));
        assert!(cache.is_mapped_with_access(arena.id(), Access::ReadWrite));

        // Unmap just the read-only mapping
        cache.unmap_with_access(arena.id(), Access::ReadOnly);
        assert!(cache.is_mapped(arena.id())); // Still mapped via read-write
        assert!(cache.is_mapped_with_access(arena.id(), Access::ReadWrite));
    }
}
