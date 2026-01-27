//! Shared-memory reference counting for cross-process buffer management.
//!
//! This module provides true cross-process reference counting by storing
//! the refcount in shared memory (memfd). Unlike heap-based `Arc`, this
//! works across process boundaries because the refcount lives in the
//! same physical memory pages that all processes map.
//!
//! # Design
//!
//! Each arena has a header and per-slot headers stored in shared memory:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │ ArenaHeader (128 bytes, cache-line aligned)                     │
//! │ ┌─────────────────────────────────────────────────────────────┐ │
//! │ │ magic: u64          │ version: u32  │ slot_count: u32       │ │
//! │ │ slot_size: u32      │ data_offset: u32                      │ │
//! │ │ arena_id: u64                                               │ │
//! │ └─────────────────────────────────────────────────────────────┘ │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ ReleaseQueue (in shared memory, MPSC lock-free)                 │
//! │ ┌─────────────────────────────────────────────────────────────┐ │
//! │ │ head: AtomicU32     │ tail: AtomicU32                       │ │
//! │ │ slots: [AtomicU32; QUEUE_SIZE]  (ring buffer)               │ │
//! │ └─────────────────────────────────────────────────────────────┘ │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ SlotHeader[0..N] (8 bytes each, naturally aligned)              │
//! │ ┌────────────┬────────────┬────────────┬────────────┐          │
//! │ │refcount:u32│refcount:u32│refcount:u32│refcount:u32│ ...      │
//! │ │state:u32   │state:u32   │state:u32   │state:u32   │          │
//! │ └────────────┴────────────┴────────────┴────────────┘          │
//! ├─────────────────────────────────────────────────────────────────┤
//! │ SlotData[0..N] (slot_size bytes each)                           │
//! │ ┌────────────┬────────────┬────────────┬────────────┐          │
//! │ │  user data │  user data │  user data │  user data │ ...      │
//! │ └────────────┴────────────┴────────────┴────────────┘          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Lock-Free Release Queue
//!
//! When a slot's refcount drops to 0, it is pushed to a lock-free MPSC
//! (multiple-producer, single-consumer) queue in shared memory. The arena
//! owner drains this queue to reclaim slots.
//!
//! This avoids O(n) scanning - release is O(1) and reclaim is O(k) where
//! k is the number of released slots.
//!
//! # Cross-Process Semantics
//!
//! - **Clone**: Atomically increments refcount (works across processes)
//! - **Drop**: Atomically decrements refcount; if 0, pushes to release queue
//! - **Reclaim**: Owner drains the release queue and marks slots as free
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::{SharedArena, SharedSlotRef};
//!
//! // Create arena (owner process)
//! let arena = SharedArena::new(4096, 16)?;
//!
//! // Acquire a slot
//! let slot = arena.acquire()?;
//! slot.data_mut()[..5].copy_from_slice(b"hello");
//!
//! // Get IPC reference
//! let ipc_ref = slot.ipc_ref();
//! // Send ipc_ref + arena fd to another process...
//!
//! // In another process:
//! let slot2 = SharedSlotRef::from_ipc(ipc_ref, mapped_arena)?;
//! // slot2 incremented the shared refcount
//! // When slot2 drops, refcount decrements
//! // When refcount hits 0, slot index is pushed to release queue
//!
//! // Owner periodically drains the queue
//! arena.reclaim();
//! ```

use crate::error::{Error, Result};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::mm::{MapFlags, ProtFlags};
use std::ffi::CString;
use std::os::unix::io::{AsRawFd, RawFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

/// Magic number to identify valid arena headers.
const ARENA_MAGIC: u64 = 0x504C585F4152454E; // "PLX_AREN" in ASCII

/// Current arena format version.
const ARENA_VERSION: u32 = 2; // Bumped for queue-based release

/// Size of the release queue (must be power of 2 for efficient modulo).
/// This limits how many slots can be pending release at once.
const RELEASE_QUEUE_SIZE: usize = 1024;

/// Sentinel value indicating an empty queue slot.
const QUEUE_EMPTY: u32 = u32::MAX;

/// Slot states (stored in shared memory).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlotState {
    /// Slot is free and can be acquired.
    Free = 0,
    /// Slot is allocated and has active references.
    Allocated = 1,
}

impl SlotState {
    fn from_u32(v: u32) -> Self {
        match v {
            0 => SlotState::Free,
            1 => SlotState::Allocated,
            _ => SlotState::Free, // Invalid states treated as free
        }
    }
}

/// Lock-free MPSC (Multiple-Producer Single-Consumer) release queue.
///
/// This is stored in shared memory and allows any process to push
/// released slot indices, while only the owner process consumes them.
///
/// # Algorithm
///
/// Uses a bounded ring buffer with atomic head/tail pointers:
/// - **Push (any process)**: CAS on tail to reserve slot, then write index
/// - **Pop (owner only)**: Read head, check if slot is filled, advance head
///
/// The queue uses a two-phase commit for push:
/// 1. Reserve slot by advancing tail (CAS)
/// 2. Write the slot index
///
/// Pop checks that the slot has been written (not QUEUE_EMPTY) before consuming.
#[repr(C, align(64))]
struct ReleaseQueue {
    /// Head index (consumer reads from here). Only owner advances this.
    head: AtomicU32,
    /// Tail index (producers write here). Any process can advance this.
    tail: AtomicU32,
    /// Padding to separate head/tail from the ring buffer (avoid false sharing).
    _pad: [u8; 56],
    /// Ring buffer of slot indices. QUEUE_EMPTY means slot not yet written.
    slots: [AtomicU32; RELEASE_QUEUE_SIZE],
}

impl ReleaseQueue {
    /// Initialize the queue (all slots empty).
    fn init(&self) {
        self.head.store(0, Ordering::Release);
        self.tail.store(0, Ordering::Release);
        for slot in &self.slots {
            slot.store(QUEUE_EMPTY, Ordering::Release);
        }
    }

    /// Try to push a slot index to the queue.
    ///
    /// Returns `true` if successful, `false` if queue is full.
    /// This is safe to call from any process.
    fn try_push(&self, slot_index: u32) -> bool {
        loop {
            let tail = self.tail.load(Ordering::Acquire);
            let head = self.head.load(Ordering::Acquire);

            // Check if queue is full
            let next_tail = tail.wrapping_add(1);
            if next_tail.wrapping_sub(head) > RELEASE_QUEUE_SIZE as u32 {
                // Queue is full
                return false;
            }

            // Try to reserve this slot by advancing tail
            match self.tail.compare_exchange_weak(
                tail,
                next_tail,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // We reserved the slot, now write the index
                    let idx = (tail as usize) & (RELEASE_QUEUE_SIZE - 1);
                    self.slots[idx].store(slot_index, Ordering::Release);
                    return true;
                }
                Err(_) => {
                    // Another producer won, retry
                    std::hint::spin_loop();
                    continue;
                }
            }
        }
    }

    /// Try to pop a slot index from the queue.
    ///
    /// Returns `Some(slot_index)` if successful, `None` if queue is empty.
    /// Only the owner should call this.
    fn try_pop(&self) -> Option<u32> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);

            // Check if queue is empty
            if head == tail {
                return None;
            }

            let idx = (head as usize) & (RELEASE_QUEUE_SIZE - 1);
            let slot_index = self.slots[idx].load(Ordering::Acquire);

            // Check if the producer has finished writing
            if slot_index == QUEUE_EMPTY {
                // Producer reserved but hasn't written yet, spin
                std::hint::spin_loop();
                continue;
            }

            // Try to advance head
            match self.head.compare_exchange_weak(
                head,
                head.wrapping_add(1),
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    // Clear the slot for reuse
                    self.slots[idx].store(QUEUE_EMPTY, Ordering::Release);
                    return Some(slot_index);
                }
                Err(_) => {
                    // Shouldn't happen in single-consumer, but handle gracefully
                    std::hint::spin_loop();
                    continue;
                }
            }
        }
    }

    /// Get the number of items in the queue (approximate, for debugging).
    fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        tail.wrapping_sub(head) as usize
    }

    /// Check if the queue is empty.
    #[allow(dead_code)]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Header at the start of the arena (in shared memory).
///
/// This is cache-line aligned (64 bytes) to avoid false sharing.
#[repr(C, align(64))]
struct ArenaHeader {
    /// Magic number for validation.
    magic: AtomicU64,
    /// Format version.
    version: AtomicU32,
    /// Number of slots.
    slot_count: AtomicU32,
    /// Size of each slot's data region (excluding header).
    slot_size: AtomicU32,
    /// Offset from arena base to first slot's data.
    data_offset: AtomicU32,
    /// Unique arena ID (for cross-process identification).
    arena_id: AtomicU64,
    /// Offset from arena base to slot headers.
    slot_headers_offset: AtomicU32,
    /// Reserved for future use.
    _reserved: [u8; 20],
}

impl ArenaHeader {
    /// Validate the header is properly initialized.
    fn validate(&self) -> Result<()> {
        let magic = self.magic.load(Ordering::Acquire);
        if magic != ARENA_MAGIC {
            return Err(Error::InvalidSegment(format!(
                "invalid arena magic: expected {:x}, got {:x}",
                ARENA_MAGIC, magic
            )));
        }
        let version = self.version.load(Ordering::Acquire);
        if version != ARENA_VERSION {
            return Err(Error::InvalidSegment(format!(
                "unsupported arena version: expected {}, got {}",
                ARENA_VERSION, version
            )));
        }
        Ok(())
    }
}

/// Per-slot header (in shared memory).
///
/// This is 8 bytes to maintain natural alignment.
#[repr(C, align(8))]
struct SlotHeader {
    /// Reference count (atomic, works across processes).
    refcount: AtomicU32,
    /// Slot state (Free, Allocated).
    state: AtomicU32,
}

impl SlotHeader {
    /// Initialize a new slot header.
    fn init(&self) {
        self.refcount.store(0, Ordering::Release);
        self.state.store(SlotState::Free as u32, Ordering::Release);
    }

    /// Try to acquire this slot (transition Free -> Allocated with refcount=1).
    ///
    /// Returns true if successful, false if slot was not free.
    fn try_acquire(&self) -> bool {
        // CAS: Free -> Allocated
        let result = self.state.compare_exchange(
            SlotState::Free as u32,
            SlotState::Allocated as u32,
            Ordering::AcqRel,
            Ordering::Acquire,
        );

        if result.is_ok() {
            // Set initial refcount
            self.refcount.store(1, Ordering::Release);
            true
        } else {
            false
        }
    }

    /// Increment refcount (for clone).
    ///
    /// # Panics
    ///
    /// Panics if refcount would overflow (> 2^31).
    fn inc_ref(&self) {
        let old = self.refcount.fetch_add(1, Ordering::AcqRel);
        if old > i32::MAX as u32 {
            // Overflow protection - this should never happen in practice
            self.refcount.fetch_sub(1, Ordering::AcqRel);
            panic!("SharedSlotRef refcount overflow");
        }
    }

    /// Decrement refcount (for drop).
    ///
    /// Returns true if this was the last reference (refcount hit 0).
    fn dec_ref(&self) -> bool {
        let old = self.refcount.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(old > 0, "refcount underflow");
        old == 1
    }

    /// Mark slot as free (called by owner after draining from queue).
    fn mark_free(&self) {
        self.state.store(SlotState::Free as u32, Ordering::Release);
    }

    /// Get current refcount (for debugging).
    fn refcount(&self) -> u32 {
        self.refcount.load(Ordering::Acquire)
    }

    /// Get current state.
    fn state(&self) -> SlotState {
        SlotState::from_u32(self.state.load(Ordering::Acquire))
    }
}

/// Global counter for generating unique arena IDs.
static ARENA_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique arena ID.
fn next_arena_id() -> u64 {
    ARENA_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Calculate arena layout parameters.
///
/// Returns (total_size, queue_offset, slot_headers_offset, data_offset).
fn calculate_layout(slot_size: usize, slot_count: usize) -> (usize, usize, usize, usize) {
    let arena_header_size = std::mem::size_of::<ArenaHeader>();
    let queue_size = std::mem::size_of::<ReleaseQueue>();
    let slot_header_size = std::mem::size_of::<SlotHeader>();
    let all_slot_headers_size = slot_header_size * slot_count;

    // Layout: ArenaHeader | ReleaseQueue | SlotHeaders | Data
    let queue_offset = arena_header_size;
    let slot_headers_offset = queue_offset + queue_size;

    // Align data offset to 64 bytes for cache efficiency
    let header_region_size = slot_headers_offset + all_slot_headers_size;
    let data_offset = (header_region_size + 63) & !63;

    let data_region_size = slot_size * slot_count;
    let total_size = data_offset + data_region_size;

    (total_size, queue_offset, slot_headers_offset, data_offset)
}

/// Arena with shared-memory reference counting and lock-free release queue.
///
/// Unlike `CpuArena`, this arena stores refcounts in the shared memory
/// itself, enabling true cross-process reference counting without messages.
///
/// # Ownership Model
///
/// - **Owner process**: Creates the arena, can acquire slots, reclaims freed slots
/// - **Client processes**: Map the arena fd, can clone/drop slot references
///
/// The owner should periodically call `reclaim()` to drain the release queue
/// and recycle slots whose refcount has dropped to 0.
///
/// # Lock-Free Release
///
/// When a slot's refcount drops to 0 (from any process), its index is pushed
/// to a lock-free MPSC queue in shared memory. The owner drains this queue
/// in O(k) time where k is the number of released slots.
pub struct SharedArena {
    /// The memfd file descriptor.
    fd: OwnedFd,
    /// Base pointer to the mmap'd region.
    base: NonNull<u8>,
    /// Total size of the arena.
    total_size: usize,
    /// Pointer to arena header (kept for potential future use in validation).
    #[allow(dead_code)]
    header: NonNull<ArenaHeader>,
    /// Pointer to release queue.
    release_queue: NonNull<ReleaseQueue>,
    /// Pointer to first slot header.
    slot_headers: NonNull<SlotHeader>,
    /// Offset from base to data region.
    data_offset: usize,
    /// Size of each slot's data.
    slot_size: usize,
    /// Number of slots.
    slot_count: usize,
    /// Unique arena ID.
    arena_id: u64,
    /// Whether this is the owner (can acquire/reclaim) or a client (clone/drop only).
    is_owner: bool,
}

impl SharedArena {
    /// Create a new arena (owner process).
    ///
    /// # Arguments
    ///
    /// * `slot_size` - Size of each slot's data region in bytes.
    /// * `slot_count` - Number of slots in the arena.
    pub fn new(slot_size: usize, slot_count: usize) -> Result<Self> {
        Self::with_name("parallax-shared-arena", slot_size, slot_count)
    }

    /// Create a new arena with a debug name.
    pub fn with_name(name: &str, slot_size: usize, slot_count: usize) -> Result<Self> {
        if slot_size == 0 {
            return Err(Error::AllocationFailed("slot_size must be > 0".into()));
        }
        if slot_count == 0 {
            return Err(Error::AllocationFailed("slot_count must be > 0".into()));
        }

        let (total_size, queue_offset, slot_headers_offset, data_offset) =
            calculate_layout(slot_size, slot_count);

        // Create memfd
        let cname = CString::new(name).map_err(|e| Error::AllocationFailed(e.to_string()))?;
        let fd = rustix::fs::memfd_create(&cname, rustix::fs::MemfdFlags::CLOEXEC)?;

        // Set size
        rustix::fs::ftruncate(&fd, total_size as u64)?;

        // Map the region
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

        let arena_id = next_arena_id();

        // Initialize header
        let header = base.cast::<ArenaHeader>();
        unsafe {
            let h = header.as_ref();
            h.magic.store(ARENA_MAGIC, Ordering::Release);
            h.version.store(ARENA_VERSION, Ordering::Release);
            h.slot_count.store(slot_count as u32, Ordering::Release);
            h.slot_size.store(slot_size as u32, Ordering::Release);
            h.data_offset.store(data_offset as u32, Ordering::Release);
            h.arena_id.store(arena_id, Ordering::Release);
            h.slot_headers_offset
                .store(slot_headers_offset as u32, Ordering::Release);
        }

        // Initialize release queue
        let release_queue = unsafe {
            NonNull::new_unchecked(base.as_ptr().add(queue_offset).cast::<ReleaseQueue>())
        };
        unsafe {
            release_queue.as_ref().init();
        }

        // Initialize slot headers
        let slot_headers = unsafe {
            NonNull::new_unchecked(base.as_ptr().add(slot_headers_offset).cast::<SlotHeader>())
        };

        for i in 0..slot_count {
            unsafe {
                let sh = &*slot_headers.as_ptr().add(i);
                sh.init();
            }
        }

        Ok(Self {
            fd,
            base,
            total_size,
            header,
            release_queue,
            slot_headers,
            data_offset,
            slot_size,
            slot_count,
            arena_id,
            is_owner: true,
        })
    }

    /// Map an existing arena from a received file descriptor (client process).
    ///
    /// # Safety
    ///
    /// The caller must ensure `fd` is a valid SharedArena file descriptor.
    pub unsafe fn from_fd(fd: OwnedFd) -> Result<Self> {
        // Get the file size
        let stat = rustix::fs::fstat(&fd)?;
        let total_size = stat.st_size as usize;

        if total_size < std::mem::size_of::<ArenaHeader>() {
            return Err(Error::InvalidSegment("arena too small for header".into()));
        }

        // Map the region
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

        // Validate header
        let header = base.cast::<ArenaHeader>();
        unsafe {
            header.as_ref().validate()?;
        }

        // Read layout from header
        let (slot_count, slot_size, data_offset, arena_id, slot_headers_offset) = unsafe {
            let h = header.as_ref();
            (
                h.slot_count.load(Ordering::Acquire) as usize,
                h.slot_size.load(Ordering::Acquire) as usize,
                h.data_offset.load(Ordering::Acquire) as usize,
                h.arena_id.load(Ordering::Acquire),
                h.slot_headers_offset.load(Ordering::Acquire) as usize,
            )
        };

        let queue_offset = std::mem::size_of::<ArenaHeader>();
        let release_queue = unsafe {
            NonNull::new_unchecked(base.as_ptr().add(queue_offset).cast::<ReleaseQueue>())
        };

        let slot_headers = unsafe {
            NonNull::new_unchecked(base.as_ptr().add(slot_headers_offset).cast::<SlotHeader>())
        };

        Ok(Self {
            fd,
            base,
            total_size,
            header,
            release_queue,
            slot_headers,
            data_offset,
            slot_size,
            slot_count,
            arena_id,
            is_owner: false, // Client cannot acquire new slots
        })
    }

    /// Acquire a slot from the arena.
    ///
    /// Returns `None` if all slots are in use or if this is a client (not owner).
    pub fn acquire(&self) -> Option<SharedSlotRef> {
        if !self.is_owner {
            return None; // Only owner can acquire new slots
        }

        // Linear scan for a free slot
        // TODO: Could use a free list or bitmap for O(1) acquire
        for i in 0..self.slot_count {
            let sh = unsafe { &*self.slot_headers.as_ptr().add(i) };
            if sh.try_acquire() {
                return Some(SharedSlotRef {
                    release_queue: self.release_queue,
                    slot_header: unsafe { NonNull::new_unchecked(sh as *const _ as *mut _) },
                    data_ptr: unsafe {
                        NonNull::new_unchecked(
                            self.base
                                .as_ptr()
                                .add(self.data_offset + i * self.slot_size)
                                as *mut u8,
                        )
                    },
                    data_len: self.slot_size,
                    slot_index: i as u32,
                    arena_id: self.arena_id,
                    data_offset: self.data_offset + i * self.slot_size,
                });
            }
        }

        None // All slots in use
    }

    /// Reclaim slots from the release queue.
    ///
    /// This should be called periodically by the owner process.
    /// Returns the number of slots reclaimed.
    ///
    /// This is O(k) where k is the number of released slots, not O(n).
    pub fn reclaim(&self) -> usize {
        if !self.is_owner {
            return 0;
        }

        let mut reclaimed = 0;
        let queue = unsafe { self.release_queue.as_ref() };

        while let Some(slot_index) = queue.try_pop() {
            if (slot_index as usize) < self.slot_count {
                let sh = unsafe { &*self.slot_headers.as_ptr().add(slot_index as usize) };
                // Double-check refcount is still 0 (might have been re-acquired)
                if sh.refcount() == 0 {
                    sh.mark_free();
                    reclaimed += 1;
                }
            }
        }

        reclaimed
    }

    /// Get the number of slots pending in the release queue.
    pub fn pending_count(&self) -> usize {
        unsafe { self.release_queue.as_ref().len() }
    }

    /// Get the unique arena ID.
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
    pub fn free_count(&self) -> usize {
        let mut count = 0;
        for i in 0..self.slot_count {
            let sh = unsafe { &*self.slot_headers.as_ptr().add(i) };
            if sh.state() == SlotState::Free {
                count += 1;
            }
        }
        count
    }

    /// Get the number of allocated slots.
    pub fn allocated_count(&self) -> usize {
        let mut count = 0;
        for i in 0..self.slot_count {
            let sh = unsafe { &*self.slot_headers.as_ptr().add(i) };
            if sh.state() == SlotState::Allocated {
                count += 1;
            }
        }
        count
    }

    /// Check if this is the owner process.
    #[inline]
    pub fn is_owner(&self) -> bool {
        self.is_owner
    }

    /// Get total arena size.
    #[inline]
    pub fn total_size(&self) -> usize {
        self.total_size
    }

    /// Reconstruct a slot reference from an IPC reference.
    ///
    /// This is used by client processes to access a slot that was
    /// sent via IPC. The refcount is incremented atomically.
    ///
    /// Returns `None` if the slot is not in Allocated state.
    pub fn slot_from_ipc(&self, ipc_ref: &SharedIpcSlotRef) -> Option<SharedSlotRef> {
        if ipc_ref.arena_id != self.arena_id {
            return None; // Wrong arena
        }

        if ipc_ref.slot_index as usize >= self.slot_count {
            return None; // Invalid slot index
        }

        let sh = unsafe { &*self.slot_headers.as_ptr().add(ipc_ref.slot_index as usize) };

        // Check slot is allocated
        if sh.state() != SlotState::Allocated {
            return None;
        }

        // Increment refcount
        sh.inc_ref();

        Some(SharedSlotRef {
            release_queue: self.release_queue,
            slot_header: unsafe { NonNull::new_unchecked(sh as *const _ as *mut _) },
            data_ptr: unsafe {
                NonNull::new_unchecked(self.base.as_ptr().add(ipc_ref.data_offset) as *mut u8)
            },
            data_len: ipc_ref.len,
            slot_index: ipc_ref.slot_index,
            arena_id: self.arena_id,
            data_offset: ipc_ref.data_offset,
        })
    }
}

impl Drop for SharedArena {
    fn drop(&mut self) {
        unsafe {
            let _ = rustix::mm::munmap(self.base.as_ptr().cast(), self.total_size);
        }
    }
}

// SAFETY: SharedArena is Send + Sync because all mutable state is behind atomics.
unsafe impl Send for SharedArena {}
unsafe impl Sync for SharedArena {}

impl AsFd for SharedArena {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

/// Reference to a slot in a SharedArena.
///
/// This is the primary handle for accessing buffer data. Unlike `Arc`,
/// the refcount is stored in shared memory and works across processes.
///
/// # Cloning
///
/// Cloning a `SharedSlotRef` atomically increments the shared refcount.
/// This works even when the clone is in a different process.
///
/// # Dropping
///
/// Dropping a `SharedSlotRef` atomically decrements the shared refcount.
/// When the refcount reaches 0, the slot index is pushed to the release
/// queue for the owner to reclaim.
pub struct SharedSlotRef {
    /// Pointer to the release queue (for pushing on drop).
    release_queue: NonNull<ReleaseQueue>,
    /// Pointer to the slot header (contains refcount).
    slot_header: NonNull<SlotHeader>,
    /// Pointer to the slot data.
    data_ptr: NonNull<u8>,
    /// Length of the slot data.
    data_len: usize,
    /// Slot index in the arena (for pushing to release queue).
    slot_index: u32,
    /// Arena ID (for IPC serialization).
    arena_id: u64,
    /// Offset from arena base to data (for IPC).
    data_offset: usize,
}

impl SharedSlotRef {
    /// Get the slot data as a byte slice.
    #[inline]
    pub fn data(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.data_ptr.as_ptr(), self.data_len) }
    }

    /// Get the slot data as a mutable byte slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.data_ptr.as_ptr(), self.data_len) }
    }

    /// Get an IPC reference for cross-process sharing.
    ///
    /// Send this over a Unix socket along with the arena fd (first time only).
    #[inline]
    pub fn ipc_ref(&self) -> SharedIpcSlotRef {
        SharedIpcSlotRef {
            arena_id: self.arena_id,
            slot_index: self.slot_index,
            data_offset: self.data_offset,
            len: self.data_len,
        }
    }

    /// Get the slot index.
    #[inline]
    pub fn slot_index(&self) -> usize {
        self.slot_index as usize
    }

    /// Get the arena ID.
    #[inline]
    pub fn arena_id(&self) -> u64 {
        self.arena_id
    }

    /// Get the current refcount (for debugging).
    #[inline]
    pub fn refcount(&self) -> u32 {
        unsafe { self.slot_header.as_ref().refcount() }
    }

    /// Get the data length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data_len
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data_len == 0
    }

    /// Get raw pointer to data.
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.data_ptr.as_ptr()
    }

    /// Get mutable raw pointer to data.
    #[inline]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.data_ptr.as_ptr()
    }

    /// Create a sub-reference (view into a portion of the data).
    ///
    /// This increments the refcount (the sub-reference keeps the slot alive).
    ///
    /// # Panics
    ///
    /// Panics if `offset + len > self.len()`.
    pub fn slice(&self, offset: usize, len: usize) -> SharedSlotRef {
        assert!(
            offset + len <= self.data_len,
            "slice exceeds slot bounds: {} + {} > {}",
            offset,
            len,
            self.data_len
        );

        // Increment refcount for the new reference
        unsafe {
            self.slot_header.as_ref().inc_ref();
        }

        SharedSlotRef {
            release_queue: self.release_queue,
            slot_header: self.slot_header,
            data_ptr: unsafe { NonNull::new_unchecked(self.data_ptr.as_ptr().add(offset)) },
            data_len: len,
            slot_index: self.slot_index,
            arena_id: self.arena_id,
            data_offset: self.data_offset + offset,
        }
    }
}

impl Clone for SharedSlotRef {
    fn clone(&self) -> Self {
        // Increment refcount in shared memory
        unsafe {
            self.slot_header.as_ref().inc_ref();
        }

        Self {
            release_queue: self.release_queue,
            slot_header: self.slot_header,
            data_ptr: self.data_ptr,
            data_len: self.data_len,
            slot_index: self.slot_index,
            arena_id: self.arena_id,
            data_offset: self.data_offset,
        }
    }
}

impl Drop for SharedSlotRef {
    fn drop(&mut self) {
        // Decrement refcount in shared memory
        let was_last = unsafe { self.slot_header.as_ref().dec_ref() };

        if was_last {
            // Push slot index to release queue
            let queue = unsafe { self.release_queue.as_ref() };
            // If queue is full, the slot will be leaked until the owner
            // does a full scan. This is a tradeoff for lock-free operation.
            // In practice, the queue should be sized large enough.
            let _ = queue.try_push(self.slot_index);
        }
    }
}

// SAFETY: SharedSlotRef is Send + Sync because all state is in shared memory
// with atomic operations.
unsafe impl Send for SharedSlotRef {}
unsafe impl Sync for SharedSlotRef {}

impl std::fmt::Debug for SharedSlotRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedSlotRef")
            .field("arena_id", &self.arena_id)
            .field("slot_index", &self.slot_index)
            .field("len", &self.data_len)
            .field("refcount", &self.refcount())
            .finish()
    }
}

/// IPC reference to a slot (serializable).
///
/// Send this over a Unix socket to share a buffer reference.
/// The first time, also send the arena fd via SCM_RIGHTS.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, Hash, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize,
)]
#[rkyv(derive(Debug))]
pub struct SharedIpcSlotRef {
    /// Arena ID (receiver looks up cached mapping).
    pub arena_id: u64,
    /// Slot index in the arena.
    pub slot_index: u32,
    /// Offset from arena base to data.
    pub data_offset: usize,
    /// Length of the data.
    pub len: usize,
}

impl SharedIpcSlotRef {
    /// Create a new IPC slot reference.
    pub const fn new(arena_id: u64, slot_index: u32, data_offset: usize, len: usize) -> Self {
        Self {
            arena_id,
            slot_index,
            data_offset,
            len,
        }
    }
}

/// Cache for mapping received SharedArena file descriptors.
///
/// Client processes use this to cache arena mappings, avoiding
/// repeated mmap calls for the same arena.
pub struct SharedArenaCache {
    /// Cached arenas: arena_id -> SharedArena
    arenas: std::collections::HashMap<u64, SharedArena>,
}

impl SharedArenaCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self {
            arenas: std::collections::HashMap::new(),
        }
    }

    /// Map an arena from a received fd and cache it.
    ///
    /// # Safety
    ///
    /// The caller must ensure `fd` is a valid SharedArena file descriptor.
    pub unsafe fn map_arena(&mut self, fd: OwnedFd) -> Result<u64> {
        let arena = unsafe { SharedArena::from_fd(fd)? };
        let arena_id = arena.id();
        self.arenas.insert(arena_id, arena);
        Ok(arena_id)
    }

    /// Get a slot reference from an IPC reference.
    ///
    /// This increments the shared refcount.
    pub fn get_slot(&self, ipc_ref: &SharedIpcSlotRef) -> Option<SharedSlotRef> {
        self.arenas
            .get(&ipc_ref.arena_id)
            .and_then(|arena| arena.slot_from_ipc(ipc_ref))
    }

    /// Check if an arena is cached.
    pub fn is_cached(&self, arena_id: u64) -> bool {
        self.arenas.contains_key(&arena_id)
    }

    /// Remove an arena from the cache.
    pub fn remove(&mut self, arena_id: u64) -> Option<SharedArena> {
        self.arenas.remove(&arena_id)
    }

    /// Clear all cached arenas.
    pub fn clear(&mut self) {
        self.arenas.clear();
    }

    /// Get the number of cached arenas.
    pub fn len(&self) -> usize {
        self.arenas.len()
    }

    /// Check if cache is empty.
    pub fn is_empty(&self) -> bool {
        self.arenas.is_empty()
    }
}

impl Default for SharedArenaCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_release_queue_basic() {
        // Create a mock queue in regular memory for testing
        let queue = Box::new(ReleaseQueue {
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            _pad: [0; 56],
            slots: std::array::from_fn(|_| AtomicU32::new(QUEUE_EMPTY)),
        });

        assert!(queue.is_empty());
        assert_eq!(queue.len(), 0);

        // Push some items
        assert!(queue.try_push(1));
        assert!(queue.try_push(2));
        assert!(queue.try_push(3));

        assert_eq!(queue.len(), 3);
        assert!(!queue.is_empty());

        // Pop items
        assert_eq!(queue.try_pop(), Some(1));
        assert_eq!(queue.try_pop(), Some(2));
        assert_eq!(queue.try_pop(), Some(3));
        assert_eq!(queue.try_pop(), None);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_release_queue_wrap_around() {
        let queue = Box::new(ReleaseQueue {
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            _pad: [0; 56],
            slots: std::array::from_fn(|_| AtomicU32::new(QUEUE_EMPTY)),
        });

        // Fill and drain multiple times to test wrap-around
        for round in 0..3 {
            for i in 0..100 {
                assert!(queue.try_push(round * 100 + i));
            }
            for i in 0..100 {
                assert_eq!(queue.try_pop(), Some(round * 100 + i));
            }
            assert!(queue.is_empty());
        }
    }

    #[test]
    fn test_shared_arena_creation() {
        let arena = SharedArena::new(4096, 16).unwrap();
        assert_eq!(arena.slot_size(), 4096);
        assert_eq!(arena.slot_count(), 16);
        assert_eq!(arena.free_count(), 16);
        assert_eq!(arena.allocated_count(), 0);
        assert!(arena.is_owner());
        assert_eq!(arena.pending_count(), 0);
    }

    #[test]
    fn test_shared_arena_acquire_release() {
        let arena = SharedArena::new(4096, 4).unwrap();
        assert_eq!(arena.free_count(), 4);

        let slot1 = arena.acquire().unwrap();
        assert_eq!(arena.free_count(), 3);
        assert_eq!(arena.allocated_count(), 1);
        assert_eq!(slot1.refcount(), 1);

        let slot2 = arena.acquire().unwrap();
        assert_eq!(arena.free_count(), 2);

        // Clone increments refcount
        let slot1_clone = slot1.clone();
        assert_eq!(slot1.refcount(), 2);
        assert_eq!(slot1_clone.refcount(), 2);

        // Drop clone decrements refcount
        drop(slot1_clone);
        assert_eq!(slot1.refcount(), 1);

        // Drop original pushes to release queue
        drop(slot1);
        assert_eq!(arena.pending_count(), 1);

        // Reclaim drains queue and marks slot free
        let reclaimed = arena.reclaim();
        assert_eq!(reclaimed, 1);
        assert_eq!(arena.free_count(), 3);
        assert_eq!(arena.pending_count(), 0);

        drop(slot2);
        arena.reclaim();
        assert_eq!(arena.free_count(), 4);
    }

    #[test]
    fn test_shared_slot_read_write() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().unwrap();

        slot.data_mut()[0] = 42;
        slot.data_mut()[1] = 43;
        slot.data_mut()[4095] = 99;

        assert_eq!(slot.data()[0], 42);
        assert_eq!(slot.data()[1], 43);
        assert_eq!(slot.data()[4095], 99);
    }

    #[test]
    fn test_shared_slot_slice() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        assert_eq!(slot.refcount(), 1);

        let sub = slot.slice(100, 200);
        assert_eq!(sub.len(), 200);
        assert_eq!(slot.refcount(), 2); // Both share the refcount

        drop(sub);
        assert_eq!(slot.refcount(), 1);
    }

    #[test]
    fn test_shared_arena_exhaustion() {
        let arena = SharedArena::new(4096, 2).unwrap();

        let _slot1 = arena.acquire().unwrap();
        let _slot2 = arena.acquire().unwrap();

        // Arena is full
        assert!(arena.acquire().is_none());
    }

    #[test]
    fn test_shared_ipc_ref() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();

        let ipc_ref = slot.ipc_ref();
        assert_eq!(ipc_ref.arena_id, arena.id());
        assert_eq!(ipc_ref.slot_index as usize, slot.slot_index());
        assert_eq!(ipc_ref.len, slot.len());
    }

    #[test]
    fn test_shared_arena_from_fd() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[0] = 123;

        let ipc_ref = slot.ipc_ref();

        // Simulate receiving the fd in another "process"
        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&arena.fd, 0).unwrap();
        let client_arena = unsafe { SharedArena::from_fd(dup_fd).unwrap() };

        assert!(!client_arena.is_owner());
        assert_eq!(client_arena.id(), arena.id());

        // Client cannot acquire new slots
        assert!(client_arena.acquire().is_none());

        // Client can get existing slot from IPC ref
        let client_slot = client_arena.slot_from_ipc(&ipc_ref).unwrap();
        assert_eq!(client_slot.data()[0], 123);
        assert_eq!(slot.refcount(), 2); // Both owner and client have refs
    }

    #[test]
    fn test_cross_process_refcount() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let ipc_ref = slot.ipc_ref();

        // Simulate another process
        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&arena.fd, 0).unwrap();
        let client_arena = unsafe { SharedArena::from_fd(dup_fd).unwrap() };

        // Owner has ref
        assert_eq!(slot.refcount(), 1);

        // Client gets ref
        let client_slot = client_arena.slot_from_ipc(&ipc_ref).unwrap();
        assert_eq!(slot.refcount(), 2);

        // Client drops ref
        drop(client_slot);
        assert_eq!(slot.refcount(), 1);

        // Owner drops ref - slot pushed to queue
        drop(slot);
        assert_eq!(arena.pending_count(), 1);

        // Reclaim drains the queue
        let reclaimed = arena.reclaim();
        assert_eq!(reclaimed, 1);
        assert_eq!(arena.free_count(), 4);
    }

    #[test]
    fn test_shared_arena_cache() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[0] = 77;
        let ipc_ref = slot.ipc_ref();

        // Create cache and map arena
        let mut cache = SharedArenaCache::new();
        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&arena.fd, 0).unwrap();
        let cached_id = unsafe { cache.map_arena(dup_fd).unwrap() };

        assert_eq!(cached_id, arena.id());
        assert!(cache.is_cached(arena.id()));

        // Get slot from cache
        let cached_slot = cache.get_slot(&ipc_ref).unwrap();
        assert_eq!(cached_slot.data()[0], 77);
        assert_eq!(slot.refcount(), 2);
    }

    #[test]
    fn test_layout_calculation() {
        let (total, queue_offset, slot_headers_offset, data_offset) = calculate_layout(4096, 16);

        // ArenaHeader = 64 bytes
        // ReleaseQueue = 64 + 4096 = ~4160 bytes (64 for head/tail/pad, 4096 for slots)
        // SlotHeaders = 16 * 8 = 128 bytes
        // Data offset = aligned to 64

        assert_eq!(queue_offset, 64); // After ArenaHeader
        assert!(slot_headers_offset > queue_offset);
        assert!(data_offset >= slot_headers_offset + 128);
        assert_eq!(data_offset % 64, 0); // Cache-line aligned
        assert_eq!(total, data_offset + 16 * 4096);
    }

    #[test]
    fn test_concurrent_refcount() {
        use std::sync::Arc;
        use std::thread;

        let arena = Arc::new(SharedArena::new(4096, 4).unwrap());
        let slot = arena.acquire().unwrap();
        let ipc_ref = slot.ipc_ref();

        // Spawn multiple threads that clone/drop the slot
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let arena = Arc::clone(&arena);
                let ipc_ref = ipc_ref;
                thread::spawn(move || {
                    for _ in 0..100 {
                        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&arena.fd, 0).unwrap();
                        let client = unsafe { SharedArena::from_fd(dup_fd).unwrap() };
                        if let Some(s) = client.slot_from_ipc(&ipc_ref) {
                            let _ = s.clone();
                            // Both drop here
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // Original slot should still have refcount 1
        assert_eq!(slot.refcount(), 1);

        // Note: We don't assert pending_count > 0 because all the client
        // slots incremented and then decremented the refcount, so none of
        // them were the "last" reference - the original slot still holds it.
        // The release queue only gets entries when refcount drops to 0.
    }

    #[test]
    fn test_concurrent_release_queue() {
        use std::sync::Arc;
        use std::thread;

        let arena = Arc::new(SharedArena::new(4096, 64).unwrap());

        // Acquire all slots
        let slots: Vec<_> = (0..64).filter_map(|_| arena.acquire()).collect();
        assert_eq!(slots.len(), 64);
        assert_eq!(arena.free_count(), 0);

        // Drop slots from multiple threads
        let slots = Arc::new(std::sync::Mutex::new(slots));
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let slots = Arc::clone(&slots);
                thread::spawn(move || {
                    for _ in 0..8 {
                        let slot = slots.lock().unwrap().pop();
                        drop(slot);
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // All slots should be in the release queue
        assert_eq!(arena.pending_count(), 64);

        // Reclaim should free all slots
        let reclaimed = arena.reclaim();
        assert_eq!(reclaimed, 64);
        assert_eq!(arena.free_count(), 64);
        assert_eq!(arena.pending_count(), 0);
    }
}
