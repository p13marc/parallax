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
//! │ ArenaHeader (64 bytes, cache-line aligned)                      │
//! │ ┌─────────────────────────────────────────────────────────────┐ │
//! │ │ magic: u64          │ version: u32  │ slot_count: u32       │ │
//! │ │ slot_size: u32      │ data_offset: u32                      │ │
//! │ │ arena_id: u64       │ slot_headers_offset │ refcount        │ │
//! │ │ slot_stride │ alignment │ reclaim_lock │ orphaned           │ │
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
//! │ │ word: u64  │ word: u64  │ word: u64  │ word: u64  │ ...      │
//! │ │ (state<<32 | refcount, one atomic)                │          │
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
//! k is the number of released slots. If a release ever finds the ring
//! full, the drop is counted in `ArenaHeader::orphaned` and the owner's
//! next `reclaim()` does a one-shot O(n) sweep of the slot headers to
//! recover the orphans (#177) — sound because the packed slot word makes
//! (Allocated, rc=0) unambiguous.
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
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use super::eventfd::EventFd;

/// Magic number to identify valid arena headers.
const ARENA_MAGIC: u64 = 0x504C585F4152454E; // "PLX_AREN" in ASCII

/// Current arena format version.
///
/// v4 added `slot_stride` and `alignment` to [`ArenaHeader`]. Before that a
/// client mapping the arena had to *guess* the stride, and guessed 64-byte
/// rounding — silently mis-addressing every slot of a 32-byte-aligned arena
/// whose slot size was not already a multiple of 64.
///
/// v5 packs each [`SlotHeader`]'s state+refcount into one `AtomicU64` and
/// adds the `orphaned` counter (#177). The byte layout is unchanged on
/// little-endian, but the bump is required, not cosmetic: a v4 peer CASes
/// the state half-word independently, so its `try_acquire` could interleave
/// between a v5 peer's 64-bit transitions — e.g. succeed against a slot a
/// v5 orphan sweep is concurrently freeing — recreating the very
/// double-allocation race the packing removes. `validate()`'s exact
/// equality check makes mixed-version mapping impossible.
const ARENA_VERSION: u32 = 5;

/// Size of the release queue (must be power of 2 for efficient modulo).
/// This limits how many slots can be pending release at once.
const RELEASE_QUEUE_SIZE: usize = 1024;

/// Sentinel value indicating an empty queue slot.
const QUEUE_EMPTY: u32 = u32::MAX;

/// Comprehensive arena metrics for monitoring and debugging.
///
/// This struct provides a snapshot of arena state at a point in time.
/// Use `SharedArena::metrics()` to obtain these metrics.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::memory::SharedArena;
///
/// let arena = SharedArena::new(4096, 100)?;
///
/// // Acquire some slots...
/// let slots: Vec<_> = (0..50).filter_map(|_| arena.acquire()).collect();
///
/// let metrics = arena.metrics();
/// println!("Arena {} utilization: {:.1}%", metrics.arena_id, metrics.utilization_percent);
/// println!("Slots: {} allocated, {} free, {} pending",
///     metrics.allocated_slots, metrics.free_slots, metrics.pending_release);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ArenaMetrics {
    /// Unique arena identifier.
    pub arena_id: u64,
    /// Total number of slots in the arena.
    pub slot_count: usize,
    /// Size of each slot in bytes.
    pub slot_size: usize,
    /// Number of currently allocated slots.
    pub allocated_slots: usize,
    /// Number of free slots available for acquisition.
    pub free_slots: usize,
    /// Number of slots pending release (in the release queue).
    pub pending_release: usize,
    /// Slots orphaned by a full release queue, awaiting the sweep (#177).
    pub orphaned: usize,
    /// Total arena memory in bytes (including headers).
    pub total_bytes: usize,
    /// Bytes used by allocated slots (allocated_slots * slot_size).
    pub used_bytes: usize,
    /// Utilization as a percentage (0.0 to 100.0).
    pub utilization_percent: f64,
    /// Whether this is the owner process (can acquire/reclaim).
    pub is_owner: bool,
}

impl ArenaMetrics {
    /// Check if utilization is above a threshold.
    #[inline]
    pub fn is_above_threshold(&self, threshold_percent: f64) -> bool {
        self.utilization_percent > threshold_percent
    }

    /// Check if the arena is nearly full (>90% utilization).
    #[inline]
    pub fn is_nearly_full(&self) -> bool {
        self.is_above_threshold(90.0)
    }

    /// Check if the arena is completely exhausted.
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.free_slots == 0
    }

    /// Get available slots (free + pending that can be reclaimed).
    #[inline]
    pub fn available_after_reclaim(&self) -> usize {
        self.free_slots + self.pending_release
    }
}

impl std::fmt::Display for ArenaMetrics {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Arena[{}]: {}/{} slots ({:.1}%), {} pending, {} bytes",
            self.arena_id,
            self.allocated_slots,
            self.slot_count,
            self.utilization_percent,
            self.pending_release,
            self.total_bytes
        )
    }
}

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

    /// How long `try_pop` waits for a producer that has reserved a ring entry
    /// but not yet stored its index, before giving up on this call.
    ///
    /// The push is a two-phase commit (reserve tail, then write), so the gap
    /// is normally a few instructions — but the producer can be preempted
    /// inside it, and an unbounded wait here livelocked the consumer on
    /// loaded machines (#171). Giving up is always safe: the entry stays in
    /// the ring for the next `reclaim()` to collect.
    const POP_SPIN_BUDGET: u32 = 128;

    /// Try to pop a slot index from the queue.
    ///
    /// Returns `Some(slot_index)` if successful, `None` if the queue is empty
    /// or its head entry is still being written (see [`Self::POP_SPIN_BUDGET`]).
    /// Only the owner should call this, and one caller at a time — concurrent
    /// consumers can erase each other's entries (see `ArenaHeader::reclaim_lock`).
    fn try_pop(&self) -> Option<u32> {
        let mut spins = 0u32;
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
                // Producer reserved but hasn't written yet. Wait briefly; if
                // it was preempted mid-push, bail out rather than livelock.
                if spins >= Self::POP_SPIN_BUDGET {
                    return None;
                }
                spins += 1;
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
    /// Arena-level reference count (for cross-process lifetime management).
    /// When this reaches 0, the arena can be unmapped.
    refcount: AtomicU32,
    /// Distance in bytes between consecutive slots' data regions.
    ///
    /// This is `slot_size` rounded up to `alignment`, and it is **not**
    /// derivable from `slot_size` alone — which is exactly why it lives here.
    /// Added in format v4.
    slot_stride: AtomicU32,
    /// Alignment the slot data was laid out for (32 for AVX, 64 for AVX-512).
    ///
    /// Recorded so a client can validate the stride rather than trust it.
    /// Added in format v4.
    alignment: AtomicU32,
    /// Serializes [`SharedArena::reclaim`] across clones (0 = unlocked).
    ///
    /// The release queue is MPSC and reclaim is its single consumer; two
    /// threads draining it concurrently can pop the same ring entry, and the
    /// loser's slot-clear can erase an index a wrapped producer has already
    /// refilled — leaking that slot forever. Carved out of the v4 reserved
    /// bytes: memfd memory is zero-filled, so pre-existing arenas read as
    /// unlocked and the format version is unchanged.
    reclaim_lock: AtomicU32,
    /// Count of slots whose last reference dropped while the release queue
    /// was full (#177). Such a slot sits at (Allocated, rc=0), invisible to
    /// `free_count`/`has_free`. Lives in shared memory, not process-local:
    /// the last drop can happen in a *client* process, and only the owner
    /// can sweep. A nonzero value tells the next [`SharedArena::reclaim`]
    /// to sweep all slot headers. Carved out of v4's reserved bytes (struct
    /// size unchanged); v5 is a semantic break anyway — see
    /// [`ARENA_VERSION`]. Added in format v5.
    orphaned: AtomicU32,
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

/// Per-slot header (in shared memory): state in bits 63..32, refcount in
/// bits 31..0 of one atomic word.
///
/// One word so the Free→Allocated transition and the initial refcount are a
/// single CAS — there is no observable (Allocated, rc=0) window during
/// acquire, which is what makes the orphan sweep in [`SharedArena::reclaim`]
/// sound (#177). (Allocated, rc=0) has exactly one meaning: "released,
/// awaiting the owner". It is absorbing until the owner's `try_free`:
/// rc can only leave 0 via `try_acquire` (requires Free) or `try_inc_ref`
/// (refuses rc=0). On little-endian the packing reproduces the old
/// two-field byte layout (rc at bytes 0..4, state at 4..8), but nothing
/// depends on that — the packing is defined on the u64 value, and all
/// peers of one arena run the same format version on one host.
#[repr(C, align(8))]
struct SlotHeader {
    word: AtomicU64,
}

/// Mask of the refcount half of a [`SlotHeader`] word.
const SLOT_RC_MASK: u64 = u32::MAX as u64;

const fn slot_word(state: SlotState, rc: u32) -> u64 {
    ((state as u64) << 32) | rc as u64
}

/// (Free, 0) — also the memfd zero-fill value, so a fresh mapping is valid.
const SLOT_FREE: u64 = slot_word(SlotState::Free, 0);
/// (Allocated, 1) — the state right after a successful acquire.
const SLOT_ALLOCATED_1: u64 = slot_word(SlotState::Allocated, 1);
/// (Allocated, 0) — released, awaiting the owner's reclaim or sweep.
const SLOT_RELEASED: u64 = slot_word(SlotState::Allocated, 0);

impl SlotHeader {
    /// Initialize a new slot header.
    fn init(&self) {
        self.word.store(SLOT_FREE, Ordering::Release);
    }

    /// Try to acquire this slot: (Free, 0) → (Allocated, 1) in one CAS.
    ///
    /// Returns true if successful, false if the slot was not free.
    fn try_acquire(&self) -> bool {
        // AcqRel: Acquire pairs with try_free's Release so the new holder is
        // ordered after the previous holder's release chain; Release
        // publishes ownership. Failure Relaxed: nothing is dereferenced.
        self.word
            .compare_exchange(
                SLOT_FREE,
                SLOT_ALLOCATED_1,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Increment refcount (for clone) — caller already holds a reference.
    ///
    /// # Panics
    ///
    /// Panics if refcount would overflow (> 2^31).
    fn inc_ref(&self) {
        // Relaxed (Arc precedent): visibility of the slot data came with the
        // reference being cloned. The +1 cannot carry into the state bits:
        // the panic threshold keeps rc far below 2^32.
        let old = self.word.fetch_add(1, Ordering::Relaxed) & SLOT_RC_MASK;
        if old > i32::MAX as u64 {
            self.word.fetch_sub(1, Ordering::Relaxed);
            panic!("SharedSlotRef refcount overflow");
        }
    }

    /// Increment refcount only if the slot is (Allocated, rc >= 1).
    ///
    /// Refuses to resurrect a released slot: an IPC ref to a slot whose
    /// refcount already hit 0 is stale by protocol (the sender must keep the
    /// slot alive until the receiver maps it), and blindly incrementing
    /// would race the owner's reclaim/sweep into a double allocation.
    fn try_inc_ref(&self) -> bool {
        let mut cur = self.word.load(Ordering::Relaxed);
        loop {
            if (cur >> 32) != SlotState::Allocated as u64 || (cur & SLOT_RC_MASK) == 0 {
                return false;
            }
            if (cur & SLOT_RC_MASK) > i32::MAX as u64 {
                panic!("SharedSlotRef refcount overflow");
            }
            // Acquire on success: the mapper reads slot data the producer
            // wrote before publishing the ref.
            match self.word.compare_exchange_weak(
                cur,
                cur + 1,
                Ordering::Acquire,
                Ordering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(v) => cur = v,
            }
        }
    }

    /// Decrement refcount (for drop).
    ///
    /// Returns true if this was the last reference (refcount hit 0).
    fn dec_ref(&self) -> bool {
        // AcqRel: Release so this holder's data writes happen-before whoever
        // frees/reacquires the slot; Acquire so the last dropper is ordered
        // after every other dropper (Arc uses Release + an acquire fence on
        // the last drop; an unconditional AcqRel RMW is equivalent).
        let old = self.word.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(old & SLOT_RC_MASK > 0, "refcount underflow");
        old & SLOT_RC_MASK == 1
    }

    /// (Allocated, 0) → (Free, 0). Owner-only, under `reclaim_lock`.
    ///
    /// The single reclaim/sweep primitive: it can never free a live or
    /// mid-acquire slot, because neither ever *is* (Allocated, 0).
    fn try_free(&self) -> bool {
        // Acquire pairs with the final dec_ref's Release; Release makes the
        // Free store the tail of the chain the next try_acquire picks up.
        self.word
            .compare_exchange(
                SLOT_RELEASED,
                SLOT_FREE,
                Ordering::AcqRel,
                Ordering::Relaxed,
            )
            .is_ok()
    }

    /// Get current refcount (for debugging).
    fn refcount(&self) -> u32 {
        (self.word.load(Ordering::Acquire) & SLOT_RC_MASK) as u32
    }

    /// Get current state.
    fn state(&self) -> SlotState {
        SlotState::from_u32((self.word.load(Ordering::Acquire) >> 32) as u32)
    }
}

/// Per-process counter for generating unique arena IDs.
static ARENA_ID_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Random per-process base for arena IDs, drawn once from the kernel.
static ARENA_ID_BASE: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// Generate a process-unique arena ID: a random 64-bit base plus a counter.
///
/// The ID keys `SharedArenaCache` and the IPC arena registries across
/// processes, so it must not collide between peers (#178). A plain counter
/// starting at 1 made every process mint 1, 2, 3, …, which collides as soon
/// as one process maps arenas from two creators. The random base makes
/// cross-process collision negligible (and is pid-reuse-proof, unlike
/// mixing the pid in); the counter keeps in-process IDs distinct.
fn next_arena_id() -> u64 {
    let base = *ARENA_ID_BASE.get_or_init(|| {
        let mut bytes = [0u8; 8];
        let mut filled = 0;
        while filled < bytes.len() {
            match rustix::rand::getrandom(
                &mut bytes[filled..],
                rustix::rand::GetRandomFlags::empty(),
            ) {
                Ok(n) => filled += n,
                Err(rustix::io::Errno::INTR) => {}
                // getrandom cannot fail on any supported kernel; if it
                // somehow does, a time-derived base still beats `1`.
                Err(_) => {
                    let t = std::time::UNIX_EPOCH
                        .elapsed()
                        .map(|d| d.as_nanos() as u64)
                        .unwrap_or(0);
                    bytes = (t ^ (std::process::id() as u64).rotate_left(32)).to_ne_bytes();
                    break;
                }
            }
        }
        u64::from_ne_bytes(bytes)
    });
    base.wrapping_add(ARENA_ID_COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// Calculate memory layout for arena with custom alignment.
///
/// Returns (total_size, queue_offset, slot_headers_offset, data_offset, slot_stride).
fn calculate_layout_aligned(
    slot_size: usize,
    slot_count: usize,
    alignment: usize,
) -> (usize, usize, usize, usize, usize) {
    let arena_header_size = std::mem::size_of::<ArenaHeader>();
    let queue_size = std::mem::size_of::<ReleaseQueue>();
    let slot_header_size = std::mem::size_of::<SlotHeader>();
    let all_slot_headers_size = slot_header_size * slot_count;

    // Layout: ArenaHeader | ReleaseQueue | SlotHeaders | Data
    let queue_offset = arena_header_size;
    let slot_headers_offset = queue_offset + queue_size;

    // Align data offset to requested alignment for SIMD efficiency
    let header_region_size = slot_headers_offset + all_slot_headers_size;
    let data_offset = (header_region_size + alignment - 1) & !(alignment - 1);

    // Round up slot_size to alignment so each slot starts aligned
    let slot_stride = (slot_size + alignment - 1) & !(alignment - 1);
    let data_region_size = slot_stride * slot_count;
    let total_size = data_offset + data_region_size;

    (
        total_size,
        queue_offset,
        slot_headers_offset,
        data_offset,
        slot_stride,
    )
}

/// Process-local wake-up for slot releases (#180).
///
/// The release queue notifies nobody by design (release stays O(1) and
/// wait-free); the price was that waiters polled — `FixedBufferPool` in
/// 2 ms slices. The doorbell closes that gap: a last-reference drop rings
/// it after pushing to the queue, and blocking waiters
/// (`FixedBufferPool::acquire`, `OutputArena::admit_within`) park on the
/// eventfd instead of polling.
///
/// Deliberately NOT in shared memory: an eventfd is a per-process resource,
/// so a *remote* peer's drop cannot ring the owner's doorbell — waiters
/// carry a coarse safety-net timeout for that case (the queue entry is
/// there; only the wake-up is missing). The eventfd is created lazily on
/// first wait so the majority of arenas (no blocking waiters) never carry
/// an fd, and a zero-waiter ring costs one fence + one relaxed load, no
/// syscall — which keeps `SharedSlotRef::drop` cheap.
pub(crate) struct Doorbell {
    /// Created by the first waiter. `None` inside means creation failed
    /// (fd exhaustion); waiters then degrade to a short sleep-poll.
    event: std::sync::OnceLock<Option<EventFd>>,
    /// Threads currently between `register()` and their wake-up.
    waiters: AtomicU32,
}

impl Doorbell {
    fn new() -> Self {
        Self {
            event: std::sync::OnceLock::new(),
            waiters: AtomicU32::new(0),
        }
    }

    /// Producer side. Call AFTER making the resource visible (queue push /
    /// orphan count).
    ///
    /// # Lost-wakeup proof
    ///
    /// Producer: push (P), `fence(SeqCst)` (F1), load `waiters` (L).
    /// Waiter: init fd + `fetch_add(waiters)` (A), `fence(SeqCst)` (F2),
    /// re-check the resource (R), park on the fd. SeqCst fences are totally
    /// ordered, so either F1 < F2 — then P is visible to R and the waiter
    /// never parks on a stale view — or F2 < F1 — then A (and the fd
    /// initialization sequenced before it) is visible to L, the producer
    /// notifies, and the eventfd counter is sticky: even a notify landing
    /// between the waiter's failed re-check and its park leaves the fd
    /// readable, so the park returns immediately.
    #[inline]
    pub(crate) fn ring(&self) {
        std::sync::atomic::fence(Ordering::SeqCst);
        if self.waiters.load(Ordering::Relaxed) > 0
            && let Some(Some(event)) = self.event.get()
        {
            // A failed write is covered by the waiter's safety-net timeout.
            let _ = event.notify();
        }
    }

    /// Waiter side: register, then re-check the resource, then `wait`.
    /// RAII so a panicking or early-returning waiter cannot leave the
    /// count stuck high.
    pub(crate) fn register(&self) -> DoorbellWaiter<'_> {
        // fd first: a producer that observes our waiters increment must be
        // able to resolve the fd (see ring's proof).
        let _ = self.event.get_or_init(|| EventFd::new().ok());
        self.waiters.fetch_add(1, Ordering::Relaxed);
        std::sync::atomic::fence(Ordering::SeqCst);
        DoorbellWaiter { bell: self }
    }
}

/// RAII registration on a [`Doorbell`]; de-registers on drop.
pub(crate) struct DoorbellWaiter<'a> {
    bell: &'a Doorbell,
}

impl DoorbellWaiter<'_> {
    /// Park until rung or `timeout` elapses. Returns true if rung; the
    /// caller re-checks the resource either way.
    pub(crate) fn wait(&self, timeout: Duration) -> bool {
        match self.bell.event.get() {
            Some(Some(event)) => event.wait_timeout(timeout).unwrap_or(false),
            // Degraded mode (no fd): a bounded sleep-poll.
            _ => {
                std::thread::sleep(timeout.min(Duration::from_millis(10)));
                false
            }
        }
    }
}

impl Drop for DoorbellWaiter<'_> {
    fn drop(&mut self) {
        self.bell.waiters.fetch_sub(1, Ordering::Relaxed);
    }
}

/// Arena with shared-memory reference counting and lock-free release queue.
///
/// This arena stores refcounts in the shared memory itself, enabling true
/// cross-process reference counting without messages.
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
    /// The memfd file descriptor, shared between clones.
    ///
    /// `Arc`, not `OwnedFd`: `SharedSlotRef` holds a `SharedArena` by value, so
    /// every `Buffer::clone` clones an arena. Duplicating the fd there made a
    /// buffer clone cost an `fcntl` and a `close` per branch — two syscalls on
    /// the hot path of a feature (fan-out) whose whole premise is that sharing
    /// a buffer is nearly free. One handle shared by all clones is equally
    /// correct: there is exactly one owner, so no double-close, and the mmap
    /// keeps the file alive independently of the descriptor.
    fd: Arc<OwnedFd>,
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
    /// Logical size of each slot's data (usable bytes).
    slot_size: usize,
    /// Stride between slots (may be larger than slot_size for alignment).
    slot_stride: usize,
    /// Number of slots.
    slot_count: usize,
    /// Unique arena ID.
    arena_id: u64,
    /// Whether this is the owner (can acquire/reclaim) or a client (clone/drop only).
    is_owner: bool,
    /// Release doorbell, shared by all in-process clones (#180). A
    /// `from_fd` client gets its own — it cannot ring the owner's, and
    /// the owner's waiters cover that with a safety-net timeout.
    doorbell: Arc<Doorbell>,
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
        Self::with_alignment(name, slot_size, slot_count, 64)
    }

    /// Create a new arena with custom alignment.
    ///
    /// # Arguments
    ///
    /// * `name` - Debug name for the memfd.
    /// * `slot_size` - Logical size of each slot's data region in bytes.
    /// * `slot_count` - Number of slots in the arena.
    /// * `alignment` - Alignment for slot data (e.g., 32 for AVX, 64 for AVX-512).
    ///
    /// The actual stride between slots will be rounded up to `alignment`.
    pub fn with_alignment(
        name: &str,
        slot_size: usize,
        slot_count: usize,
        alignment: usize,
    ) -> Result<Self> {
        if slot_size == 0 {
            return Err(Error::AllocationFailed("slot_size must be > 0".into()));
        }
        if slot_count == 0 {
            return Err(Error::AllocationFailed("slot_count must be > 0".into()));
        }
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Error::AllocationFailed(
                "alignment must be a positive power of 2".into(),
            ));
        }

        let (total_size, queue_offset, slot_headers_offset, data_offset, slot_stride) =
            calculate_layout_aligned(slot_size, slot_count, alignment);

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
            h.slot_stride.store(slot_stride as u32, Ordering::Release);
            h.alignment.store(alignment as u32, Ordering::Release);
            h.refcount.store(1, Ordering::Release); // Initial refcount = 1
            h.reclaim_lock.store(0, Ordering::Release);
            h.orphaned.store(0, Ordering::Release);
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
            fd: Arc::new(fd),
            base,
            total_size,
            header,
            release_queue,
            slot_headers,
            data_offset,
            slot_size,
            slot_stride,
            slot_count,
            arena_id,
            is_owner: true,
            doorbell: Arc::new(Doorbell::new()),
        })
    }

    /// Create a new arena aligned for AVX2 SIMD (32-byte alignment).
    ///
    /// This is the recommended alignment for most modern SIMD operations.
    pub fn new_avx(slot_size: usize, slot_count: usize) -> Result<Self> {
        Self::with_alignment("parallax-shared-arena-avx", slot_size, slot_count, 32)
    }

    /// Create a new arena aligned for AVX-512 SIMD (64-byte alignment).
    ///
    /// Use this when targeting systems with AVX-512 support.
    pub fn new_avx512(slot_size: usize, slot_count: usize) -> Result<Self> {
        Self::with_alignment("parallax-shared-arena-avx512", slot_size, slot_count, 64)
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

        // Read layout from header and increment refcount
        let (slot_count, slot_size, data_offset, arena_id, slot_headers_offset, slot_stride, align) = unsafe {
            let h = header.as_ref();
            // Increment arena refcount (cross-process safe)
            let old_refcount = h.refcount.fetch_add(1, Ordering::AcqRel);
            if old_refcount > i32::MAX as u32 {
                // Overflow protection - decrement and fail
                h.refcount.fetch_sub(1, Ordering::AcqRel);
                // Unmap before returning error
                let _ = rustix::mm::munmap(base.as_ptr().cast(), total_size);
                return Err(Error::AllocationFailed("arena refcount overflow".into()));
            }
            (
                h.slot_count.load(Ordering::Acquire) as usize,
                h.slot_size.load(Ordering::Acquire) as usize,
                h.data_offset.load(Ordering::Acquire) as usize,
                h.arena_id.load(Ordering::Acquire),
                h.slot_headers_offset.load(Ordering::Acquire) as usize,
                h.slot_stride.load(Ordering::Acquire) as usize,
                h.alignment.load(Ordering::Acquire) as usize,
            )
        };

        // Validate the stride the owner recorded (format v4). This used to be
        // recomputed here as `(slot_size + 63) & !63`, which is right only when
        // the owner happened to use 64-byte alignment: an arena built with
        // `new_avx` (32) and a slot size that is a multiple of 32 but not of 64
        // got a client stride larger than the owner's, mis-addressing every
        // slot after the first and reading past the mapping at the last.
        //
        // On any inconsistency, unmap and refuse rather than hand back an arena
        // that silently corrupts data.
        let reject = |msg: String| -> Error {
            unsafe {
                header.as_ref().refcount.fetch_sub(1, Ordering::AcqRel);
                let _ = rustix::mm::munmap(base.as_ptr().cast(), total_size);
            }
            Error::InvalidSegment(msg)
        };

        if align == 0 || !align.is_power_of_two() {
            return Err(reject(format!(
                "arena declares a non-power-of-two slot alignment: {align}"
            )));
        }
        let expected_stride = slot_size.div_ceil(align) * align;
        if slot_stride != expected_stride {
            return Err(reject(format!(
                "arena slot_stride {slot_stride} disagrees with slot_size {slot_size} \
                 rounded up to alignment {align} ({expected_stride})"
            )));
        }
        if data_offset + slot_stride * slot_count > total_size {
            return Err(reject(format!(
                "arena slots do not fit the mapping: data_offset {data_offset} + \
                 {slot_count} x stride {slot_stride} > {total_size} bytes"
            )));
        }

        let queue_offset = std::mem::size_of::<ArenaHeader>();
        let release_queue = unsafe {
            NonNull::new_unchecked(base.as_ptr().add(queue_offset).cast::<ReleaseQueue>())
        };

        let slot_headers = unsafe {
            NonNull::new_unchecked(base.as_ptr().add(slot_headers_offset).cast::<SlotHeader>())
        };

        Ok(Self {
            fd: Arc::new(fd),
            base,
            total_size,
            header,
            release_queue,
            slot_headers,
            data_offset,
            slot_size,
            slot_stride,
            slot_count,
            arena_id,
            is_owner: false, // Client cannot acquire new slots
            doorbell: Arc::new(Doorbell::new()),
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
        // NOTE: Could use a free list or bitmap for O(1) acquire if this becomes a bottleneck
        for i in 0..self.slot_count {
            let sh = unsafe { &*self.slot_headers.as_ptr().add(i) };
            if sh.try_acquire() {
                return Some(SharedSlotRef {
                    arena: self.clone(),
                    release_queue: self.release_queue,
                    slot_header: unsafe { NonNull::new_unchecked(sh as *const _ as *mut _) },
                    data_ptr: unsafe {
                        NonNull::new_unchecked(
                            self.base
                                .as_ptr()
                                .add(self.data_offset + i * self.slot_stride),
                        )
                    },
                    data_len: self.slot_size, // Usable data size (not stride)
                    slot_index: i as u32,
                    data_offset: self.data_offset + i * self.slot_stride,
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

        // The release queue is MPSC and this is its single consumer. Arena
        // clones share the mapping, so two threads can reach this point at
        // once — serialize them. The loser returns 0 immediately, which is
        // fine: the holder is draining the same queue.
        let header = unsafe { self.header.as_ref() };
        if header.reclaim_lock.swap(1, Ordering::Acquire) != 0 {
            return 0;
        }

        let mut reclaimed = 0;
        let queue = unsafe { self.release_queue.as_ref() };

        while let Some(slot_index) = queue.try_pop() {
            if (slot_index as usize) < self.slot_count {
                let sh = unsafe { &*self.slot_headers.as_ptr().add(slot_index as usize) };
                // try_free CASes (Allocated, 0) -> (Free, 0), so a slot that
                // was re-referenced between push and pop is skipped; a later
                // drop re-pushes it.
                if sh.try_free() {
                    reclaimed += 1;
                }
            }
        }

        // Recover slots orphaned by a full ring (#177). swap(0) claims every
        // orphan counted so far; one that races in after the swap re-arms
        // the counter and the *next* reclaim sweeps. Deliberately not
        // "decrement by slots found": the sweep also frees slots pushed to
        // the ring after the drain above, so found-count and orphan-count
        // need not match, and residue arithmetic either leaks or pins the
        // counter above zero (an O(n) sweep on every reclaim).
        if header.orphaned.swap(0, Ordering::Acquire) > 0 {
            for i in 0..self.slot_count {
                let sh = unsafe { &*self.slot_headers.as_ptr().add(i) };
                if sh.try_free() {
                    reclaimed += 1;
                }
            }
        }

        header.reclaim_lock.store(0, Ordering::Release);
        reclaimed
    }

    /// Number of slots currently orphaned by a full release queue (#177).
    ///
    /// Nonzero means the next [`reclaim`](Self::reclaim) will sweep all slot
    /// headers to recover them.
    pub fn orphaned_count(&self) -> usize {
        unsafe { self.header.as_ref() }
            .orphaned
            .load(Ordering::Relaxed) as usize
    }

    /// The release doorbell (#180), for blocking waiters.
    pub(crate) fn doorbell(&self) -> &Doorbell {
        &self.doorbell
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

    /// Whether [`acquire`](Self::acquire) would succeed right now.
    ///
    /// Like `free_count() > 0`, but stops at the first free slot instead of
    /// scanning all of them. This is for *admission control*: an encoder can
    /// check before doing irreversible work (pushing a frame into a GOP) and
    /// skip the input cleanly, rather than encoding and then discovering it has
    /// nowhere to put the result.
    ///
    /// Call [`reclaim`](Self::reclaim) first — a slot released downstream stays
    /// `Allocated` until the owner reclaims it.
    ///
    /// Always `false` for a client (non-owner) arena, which cannot acquire.
    pub fn has_free(&self) -> bool {
        if !self.is_owner {
            return false;
        }
        (0..self.slot_count).any(|i| {
            let sh = unsafe { &*self.slot_headers.as_ptr().add(i) };
            sh.state() == SlotState::Free
        })
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

    /// Get utilization as a percentage (0.0 to 100.0).
    ///
    /// Utilization = allocated_count / slot_count * 100
    #[inline]
    pub fn utilization(&self) -> f64 {
        if self.slot_count == 0 {
            return 0.0;
        }
        (self.allocated_count() as f64 / self.slot_count as f64) * 100.0
    }

    /// Get comprehensive arena metrics.
    ///
    /// This provides a snapshot of arena state including slot counts,
    /// memory usage, and utilization. Useful for monitoring and debugging.
    pub fn metrics(&self) -> ArenaMetrics {
        let allocated = self.allocated_count();
        let free = self.slot_count - allocated;
        let pending = self.pending_count();

        ArenaMetrics {
            arena_id: self.arena_id,
            slot_count: self.slot_count,
            slot_size: self.slot_size,
            allocated_slots: allocated,
            free_slots: free,
            pending_release: pending,
            orphaned: self.orphaned_count(),
            total_bytes: self.total_size,
            used_bytes: allocated * self.slot_size,
            utilization_percent: if self.slot_count > 0 {
                (allocated as f64 / self.slot_count as f64) * 100.0
            } else {
                0.0
            },
            is_owner: self.is_owner,
        }
    }

    /// Check if the arena is nearly exhausted (utilization > threshold).
    ///
    /// Default threshold is 90%. Use this to trigger backpressure.
    #[inline]
    pub fn is_nearly_exhausted(&self) -> bool {
        self.is_nearly_exhausted_threshold(90.0)
    }

    /// Check if the arena is nearly exhausted with custom threshold.
    #[inline]
    pub fn is_nearly_exhausted_threshold(&self, threshold_percent: f64) -> bool {
        self.utilization() > threshold_percent
    }

    /// Check if the arena is completely exhausted (no free slots).
    #[inline]
    pub fn is_exhausted(&self) -> bool {
        self.free_count() == 0
    }

    /// Reconstruct a slot reference from an IPC reference.
    ///
    /// This is used by client processes to access a slot that was
    /// sent via IPC. The refcount is incremented atomically.
    ///
    /// Returns `None` if the slot is not live: not Allocated, or released
    /// (refcount already 0). Resurrection from rc=0 is refused — the sender
    /// must keep the slot alive until the receiver maps it, so a released
    /// slot means the ref is stale, and incrementing anyway would race the
    /// owner's reclaim into a double allocation (#177).
    pub fn slot_from_ipc(&self, ipc_ref: &SharedIpcSlotRef) -> Option<SharedSlotRef> {
        if ipc_ref.arena_id != self.arena_id {
            return None; // Wrong arena
        }

        if ipc_ref.slot_index as usize >= self.slot_count {
            return None; // Invalid slot index
        }

        let sh = unsafe { &*self.slot_headers.as_ptr().add(ipc_ref.slot_index as usize) };

        // Atomically: verify (Allocated, rc >= 1) and increment.
        if !sh.try_inc_ref() {
            return None;
        }

        Some(SharedSlotRef {
            arena: self.clone(),
            release_queue: self.release_queue,
            slot_header: unsafe { NonNull::new_unchecked(sh as *const _ as *mut _) },
            data_ptr: unsafe {
                NonNull::new_unchecked(self.base.as_ptr().add(ipc_ref.data_offset))
            },
            data_len: ipc_ref.len,
            slot_index: ipc_ref.slot_index,
            data_offset: ipc_ref.data_offset,
        })
    }
}

impl Clone for SharedArena {
    /// Clone the arena, incrementing the shared refcount.
    ///
    /// This works across processes - the refcount is stored in shared memory.
    /// In-process clones share the same mmap and only increment the refcount.
    /// The mmap is only unmapped when the last clone is dropped.
    fn clone(&self) -> Self {
        // Increment refcount in shared memory
        let header = unsafe { self.header.as_ref() };
        let old_refcount = header.refcount.fetch_add(1, Ordering::AcqRel);
        if old_refcount > i32::MAX as u32 {
            // Overflow protection
            header.refcount.fetch_sub(1, Ordering::AcqRel);
            panic!("SharedArena refcount overflow");
        }

        // Share the fd and the mmap — don't create new ones. Safe because the
        // mapping is MAP_SHARED, all clones share the arena-level refcount, and
        // the region is only unmapped when that refcount drops to 0.
        Self {
            fd: Arc::clone(&self.fd),
            base: self.base,
            total_size: self.total_size,
            header: self.header,
            release_queue: self.release_queue,
            slot_headers: self.slot_headers,
            data_offset: self.data_offset,
            slot_size: self.slot_size,
            slot_stride: self.slot_stride,
            slot_count: self.slot_count,
            arena_id: self.arena_id,
            is_owner: self.is_owner,
            doorbell: Arc::clone(&self.doorbell),
        }
    }
}

impl Drop for SharedArena {
    fn drop(&mut self) {
        // Decrement refcount in shared memory
        let header = unsafe { self.header.as_ref() };
        let old_refcount = header.refcount.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(old_refcount > 0, "SharedArena refcount underflow");

        // Only unmap when we're the last reference
        if old_refcount == 1 {
            unsafe {
                let _ = rustix::mm::munmap(self.base.as_ptr().cast(), self.total_size);
            }
        }
        // The fd closes when the last clone drops its Arc. That may happen
        // before or after the munmap above; either order is fine, because the
        // mapping holds its own reference to the file.
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
    /// Clone of the arena - keeps the mmap alive while slot is in use.
    /// This uses the shared-memory refcount, so it works across processes.
    arena: SharedArena,
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
            arena_id: self.arena.id(),
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
        self.arena.id()
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

    /// Get the arena's raw file descriptor (for IPC).
    ///
    /// Use this to send the arena fd via SCM_RIGHTS to other processes.
    #[inline]
    pub fn arena_fd(&self) -> i32 {
        self.arena.raw_fd()
    }

    /// Get the arena's total size (for IPC).
    #[inline]
    pub fn arena_size(&self) -> usize {
        self.arena.total_size()
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
            arena: self.arena.clone(),
            release_queue: self.release_queue,
            slot_header: self.slot_header,
            data_ptr: unsafe { NonNull::new_unchecked(self.data_ptr.as_ptr().add(offset)) },
            data_len: len,
            slot_index: self.slot_index,
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
            arena: self.arena.clone(),
            release_queue: self.release_queue,
            slot_header: self.slot_header,
            data_ptr: self.data_ptr,
            data_len: self.data_len,
            slot_index: self.slot_index,
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
            if !queue.try_push(self.slot_index) {
                // Ring full: the slot sits at (Allocated, rc=0), invisible
                // to has_free/free_count. Count it so the owner's next
                // reclaim() sweeps the slot headers and recovers it (#177).
                // Release: the released slot word is in the happens-before
                // past of a reclaim that Acquire-observes this count.
                let header = unsafe { self.arena.header.as_ref() };
                header.orphaned.fetch_add(1, Ordering::Release);
            }
            // Wake any blocking waiter in THIS process (#180). Rings on the
            // orphan path too — a sweep-capable reclaim frees either. With
            // no waiters this is a fence + relaxed load, no syscall.
            self.arena.doorbell.ring();
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
            .field("arena_id", &self.arena.id())
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
    fn a_stalled_producer_cannot_livelock_the_consumer() {
        let queue = Box::new(ReleaseQueue {
            head: AtomicU32::new(0),
            tail: AtomicU32::new(0),
            _pad: [0; 56],
            slots: std::array::from_fn(|_| AtomicU32::new(QUEUE_EMPTY)),
        });

        // Simulate a producer preempted mid-push: tail is reserved but the
        // index was never stored. Before #171 this spun forever.
        queue.tail.store(1, Ordering::Release);
        assert_eq!(queue.try_pop(), None);

        // The producer finishes its store — the entry becomes poppable.
        queue.slots[0].store(42, Ordering::Release);
        assert_eq!(queue.try_pop(), Some(42));
        assert!(queue.is_empty());
    }

    #[test]
    fn concurrent_reclaims_do_not_lose_slots() {
        use std::sync::Arc;

        // reclaim() is serialized internally; hammering it from many threads
        // while the ring wraps must never leak a slot. 3 rounds × 64 slots
        // wraps nothing by itself, so push the ring around with repetition.
        let arena = Arc::new(SharedArena::new(512, 64).unwrap());

        for _ in 0..20 {
            let slots: Vec<_> = std::iter::from_fn(|| arena.acquire()).collect();
            assert_eq!(slots.len(), 64);
            drop(slots);

            let handles: Vec<_> = (0..8)
                .map(|_| {
                    let arena = Arc::clone(&arena);
                    std::thread::spawn(move || arena.reclaim())
                })
                .collect();
            for h in handles {
                h.join().unwrap();
            }
            // A drain may have been cut short (lock held, spin budget); a few
            // follow-up reclaims must find everything. Bounded, so a genuinely
            // lost slot fails the assert instead of hanging the test.
            for _ in 0..100 {
                if arena.free_count() == 64 {
                    break;
                }
                arena.reclaim();
                std::thread::yield_now();
            }
            assert_eq!(arena.free_count(), 64);
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
    fn arena_ids_are_process_unique_not_a_counter_from_one() {
        // #178: ids key cross-process caches, so every process minting
        // 1, 2, 3, … collides as soon as one process maps arenas from two
        // creators. The id is now a random 64-bit base plus a counter:
        // distinct within the process, and the old deterministic low ids
        // only reachable at ~2^-50 odds (this assertion is probabilistic
        // by nature — a failure here means the base was not randomized).
        let a = SharedArena::new(64, 1).unwrap();
        let b = SharedArena::new(64, 1).unwrap();
        assert_ne!(a.id(), b.id());
        assert!(a.id().max(b.id()) > 1 << 12);
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
    fn has_free_agrees_with_acquire() {
        let arena = SharedArena::new(4096, 2).unwrap();
        assert!(arena.has_free());

        let slot1 = arena.acquire().unwrap();
        assert!(arena.has_free(), "one of two slots is still free");

        let _slot2 = arena.acquire().unwrap();
        assert!(!arena.has_free());
        assert!(arena.acquire().is_none(), "has_free lied about exhaustion");

        // A released slot only counts once the owner reclaims it — which is
        // exactly why admission control has to reclaim before it asks.
        drop(slot1);
        assert!(!arena.has_free(), "release alone must not free the slot");
        arena.reclaim();
        assert!(arena.has_free());
        assert!(arena.acquire().is_some());
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
    fn a_full_release_queue_orphans_and_reclaim_recovers() {
        // #177: more live slots than ring entries, all dropped at once —
        // the overflow beyond RELEASE_QUEUE_SIZE used to leak forever.
        let extra = 64;
        let count = RELEASE_QUEUE_SIZE + extra;
        let arena = SharedArena::new(64, count).unwrap();

        let slots: Vec<_> = (0..count).map(|_| arena.acquire().unwrap()).collect();
        assert_eq!(arena.free_count(), 0);
        drop(slots);

        assert_eq!(
            arena.orphaned_count(),
            extra,
            "drops past ring capacity must be counted, not lost"
        );
        assert_eq!(arena.free_count(), 0, "release alone frees nothing");

        let reclaimed = arena.reclaim();
        assert_eq!(reclaimed, count, "drain + sweep must recover every slot");
        assert_eq!(arena.free_count(), count);
        assert_eq!(arena.orphaned_count(), 0);

        // The arena is fully usable again.
        let again: Vec<_> = (0..count).map(|_| arena.acquire().unwrap()).collect();
        assert_eq!(again.len(), count);
    }

    #[test]
    fn sweep_never_frees_a_live_or_mid_acquire_slot() {
        // #177 soundness: acquire is a single CAS to (Allocated, 1), so the
        // sweep's (Allocated, 0) -> Free CAS can never free a slot that a
        // concurrent acquirer just handed out. Each thread writes a unique
        // token into its slot and verifies it survives; a double allocation
        // corrupts a token. Ring pressure (slot_count > ring) guarantees
        // orphans, so the sweep path really runs.
        use std::sync::Barrier;
        let count = RELEASE_QUEUE_SIZE + 76;
        let arena = SharedArena::new(64, count).unwrap();

        // Fill the ring with orphan-generating pressure first.
        let warm: Vec<_> = (0..count).map(|_| arena.acquire().unwrap()).collect();
        drop(warm);
        assert!(arena.orphaned_count() > 0);

        let barrier = std::sync::Arc::new(Barrier::new(10));
        let mut handles = Vec::new();
        for t in 0..8u64 {
            let arena = arena.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for i in 0..2000u64 {
                    let Some(mut slot) = arena.acquire() else {
                        arena.reclaim();
                        continue;
                    };
                    let token = (t << 32) | i;
                    slot.data_mut()[..8].copy_from_slice(&token.to_ne_bytes());
                    std::thread::yield_now();
                    let read = u64::from_ne_bytes(slot.data()[..8].try_into().unwrap());
                    assert_eq!(read, token, "double allocation corrupted a live slot");
                }
            }));
        }
        for _ in 0..2 {
            let arena = arena.clone();
            let barrier = barrier.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                for _ in 0..4000 {
                    arena.reclaim();
                    std::thread::yield_now();
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }

        arena.reclaim();
        assert_eq!(arena.free_count(), count, "every slot must come back");
    }

    #[test]
    fn slot_from_ipc_refuses_a_released_slot() {
        // #177: resurrection from rc=0 would race the owner's reclaim into
        // a double allocation, so a stale ipc_ref must resolve to None.
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let ipc_ref = slot.ipc_ref();

        // Live slot: the ref resolves.
        let mapped = arena.slot_from_ipc(&ipc_ref).unwrap();
        drop(mapped);

        // Released slot (rc hit 0, sitting in the ring): refused.
        drop(slot);
        assert!(
            arena.slot_from_ipc(&ipc_ref).is_none(),
            "a released slot must not be resurrected"
        );
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

    /// Regression for #1: a client must read the slot stride from the header,
    /// not guess it.
    ///
    /// `from_fd` used to recompute the stride as `(slot_size + 63) & !63`,
    /// which is only correct for 64-byte-aligned arenas. `new_avx` uses 32, so
    /// a slot size that is a multiple of 32 but not of 64 gave the client a
    /// *larger* stride than the owner: slot `i` was read at `data_offset +
    /// i*128` where the owner wrote it at `data_offset + i*96`. Silent
    /// cross-process corruption, and an out-of-bounds read at the last slot.
    #[test]
    fn from_fd_reads_a_non_64_byte_stride_from_the_header() {
        const SLOT_SIZE: usize = 96; // multiple of 32, not of 64
        const SLOT_COUNT: usize = 8;

        let arena = SharedArena::with_alignment("avx32-stride", SLOT_SIZE, SLOT_COUNT, 32).unwrap();
        assert_eq!(arena.slot_stride, 96, "owner stride");
        assert_ne!(
            arena.slot_stride,
            SLOT_SIZE.div_ceil(64) * 64,
            "test is pointless unless the 64-rounded guess differs"
        );

        // Owner writes a distinct pattern into every slot and keeps the refs so
        // nothing is recycled underneath us.
        let mut slots = Vec::new();
        for i in 0..SLOT_COUNT {
            let mut slot = arena.acquire().expect("slot available");
            slot.data_mut().fill(i as u8 + 1);
            slots.push(slot);
        }

        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&arena.fd, 0).unwrap();
        let client = unsafe { SharedArena::from_fd(dup_fd).unwrap() };
        assert_eq!(client.slot_stride, arena.slot_stride, "client stride");

        // Every slot must read back exactly what the owner wrote, including the
        // last one — which under the old guess sat past the end of the mapping.
        for (i, slot) in slots.iter().enumerate() {
            let client_slot = client.slot_from_ipc(&slot.ipc_ref()).unwrap();
            let data = client_slot.data();
            assert_eq!(data.len(), SLOT_SIZE, "slot {i} length");
            assert!(
                data.iter().all(|&b| b == i as u8 + 1),
                "slot {i} read back as {:?}, expected all {}",
                &data[..8.min(data.len())],
                i as u8 + 1
            );
        }
    }

    /// The default 64-byte path must keep working across the version bump.
    #[test]
    fn from_fd_still_handles_the_default_alignment() {
        let arena = SharedArena::new(1000, 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[0] = 200;

        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&arena.fd, 0).unwrap();
        let client = unsafe { SharedArena::from_fd(dup_fd).unwrap() };

        assert_eq!(client.slot_stride, 1024);
        assert_eq!(
            client.slot_from_ipc(&slot.ipc_ref()).unwrap().data()[0],
            200
        );
    }

    /// A header whose stride contradicts its own slot size / alignment is
    /// rejected instead of producing a mis-indexed arena.
    #[test]
    fn from_fd_rejects_an_inconsistent_stride() {
        let arena = SharedArena::with_alignment("bad-stride", 96, 4, 32).unwrap();

        // Corrupt the recorded stride the way a mismatched writer would.
        unsafe {
            arena
                .header
                .as_ref()
                .slot_stride
                .store(128, Ordering::Release);
        }

        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&arena.fd, 0).unwrap();
        match unsafe { SharedArena::from_fd(dup_fd) } {
            Ok(_) => panic!("an inconsistent stride must be rejected"),
            Err(e) => assert!(
                e.to_string().contains("slot_stride"),
                "expected a stride complaint, got: {e}"
            ),
        }
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
        let (total, queue_offset, slot_headers_offset, data_offset, slot_stride) =
            calculate_layout_aligned(4096, 16, 64);

        // ArenaHeader = 64 bytes
        // ReleaseQueue = 64 + 4096 = ~4160 bytes (64 for head/tail/pad, 4096 for slots)
        // SlotHeaders = 16 * 8 = 128 bytes
        // Data offset = aligned to 64

        assert_eq!(queue_offset, 64); // After ArenaHeader
        assert!(slot_headers_offset > queue_offset);
        assert!(data_offset >= slot_headers_offset + 128);
        assert_eq!(data_offset % 64, 0); // Cache-line aligned
        assert_eq!(slot_stride, 4096); // Already aligned to 64
        assert_eq!(total, data_offset + 16 * slot_stride);
    }

    #[test]
    fn test_layout_calculation_unaligned_slot() {
        // Slot size that's not a multiple of alignment
        let (total, _queue_offset, _slot_headers_offset, data_offset, slot_stride) =
            calculate_layout_aligned(1000, 10, 32);

        // Slot stride should be rounded up to 32-byte alignment
        assert_eq!(slot_stride, 1024); // ceil(1000/32)*32 = 1024
        assert_eq!(data_offset % 32, 0); // Data region aligned
        assert_eq!(total, data_offset + 10 * slot_stride);
    }

    #[test]
    fn test_concurrent_refcount() {
        use std::thread;

        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let ipc_ref = slot.ipc_ref();

        // Spawn multiple threads that clone/drop the slot
        let handles: Vec<_> = (0..10)
            .map(|_| {
                let arena = arena.clone();
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

        let arena = SharedArena::new(4096, 64).unwrap();

        // Acquire all slots
        let slots: Vec<_> = (0..64).filter_map(|_| arena.acquire()).collect();
        assert_eq!(slots.len(), 64);
        assert_eq!(arena.free_count(), 0);

        // Drop slots from multiple threads (Arc is for the Mutex, not SharedArena)
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

    #[test]
    fn test_arena_metrics() {
        let arena = SharedArena::new(4096, 100).unwrap();

        // Initial metrics
        let m = arena.metrics();
        assert_eq!(m.slot_count, 100);
        assert_eq!(m.slot_size, 4096);
        assert_eq!(m.allocated_slots, 0);
        assert_eq!(m.free_slots, 100);
        assert_eq!(m.pending_release, 0);
        assert!((m.utilization_percent - 0.0).abs() < 0.01);
        assert!(!m.is_nearly_full());
        assert!(!m.is_exhausted());
        assert!(m.is_owner);

        // Acquire 50 slots (50% utilization)
        let slots: Vec<_> = (0..50).filter_map(|_| arena.acquire()).collect();
        assert_eq!(slots.len(), 50);

        let m = arena.metrics();
        assert_eq!(m.allocated_slots, 50);
        assert_eq!(m.free_slots, 50);
        assert!((m.utilization_percent - 50.0).abs() < 0.01);
        assert!(!m.is_nearly_full());
        assert!(!m.is_exhausted());
        assert_eq!(m.used_bytes, 50 * 4096);

        // Acquire 45 more (95% utilization)
        let more_slots: Vec<_> = (0..45).filter_map(|_| arena.acquire()).collect();
        assert_eq!(more_slots.len(), 45);

        let m = arena.metrics();
        assert_eq!(m.allocated_slots, 95);
        assert_eq!(m.free_slots, 5);
        assert!((m.utilization_percent - 95.0).abs() < 0.01);
        assert!(m.is_nearly_full()); // > 90%
        assert!(!m.is_exhausted());

        // Test threshold methods
        assert!(arena.is_nearly_exhausted()); // > 90%
        assert!(arena.is_nearly_exhausted_threshold(90.0));
        assert!(!arena.is_nearly_exhausted_threshold(96.0));
        assert!(!arena.is_exhausted());

        // Test Display trait
        let display = format!("{}", m);
        assert!(display.contains("95/100"));
        assert!(display.contains("95.0%"));

        // Drop some slots, check pending
        drop(slots);
        let m = arena.metrics();
        assert_eq!(m.pending_release, 50);
        assert_eq!(m.available_after_reclaim(), 5 + 50); // free + pending

        // Reclaim and verify
        arena.reclaim();
        let m = arena.metrics();
        assert_eq!(m.pending_release, 0);
        assert_eq!(m.free_slots, 55);
        assert_eq!(m.allocated_slots, 45);
    }

    #[test]
    fn test_utilization_methods() {
        let arena = SharedArena::new(1024, 10).unwrap();

        assert!((arena.utilization() - 0.0).abs() < 0.01);
        assert!(!arena.is_exhausted());

        // Acquire all slots
        let slots: Vec<_> = (0..10).filter_map(|_| arena.acquire()).collect();
        assert_eq!(slots.len(), 10);

        assert!((arena.utilization() - 100.0).abs() < 0.01);
        assert!(arena.is_exhausted());
        assert!(arena.is_nearly_exhausted());

        // Can't acquire more
        assert!(arena.acquire().is_none());
    }
}

#[cfg(test)]
mod clone_drop_tests {
    use super::*;

    #[test]
    fn test_clone_then_drop_original() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let clone = arena.clone();

        // Drop original - clone should still work
        drop(arena);

        // Clone should still be usable
        let slot = clone.acquire().unwrap();
        assert_eq!(slot.len(), 4096);
    }

    #[test]
    fn test_clone_acquire_then_drop_original() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let clone = arena.clone();

        // Acquire from original
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[0] = 42;

        // Drop original
        drop(arena);

        // Slot should still be valid (data in shared memory)
        assert_eq!(slot.data()[0], 42);

        // Clone should still work
        let slot2 = clone.acquire().unwrap();
        assert_eq!(slot2.len(), 4096);
    }

    #[test]
    fn test_multiple_clones_sequential_drop() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let clone1 = arena.clone();
        let clone2 = arena.clone();
        let clone3 = arena.clone();

        drop(arena);
        drop(clone1);

        // clone2 and clone3 should still work
        let slot = clone2.acquire().unwrap();
        assert_eq!(slot.len(), 4096);

        drop(clone2);

        // clone3 should still work
        let slot2 = clone3.acquire().unwrap();
        assert_eq!(slot2.len(), 4096);
    }
}

#[cfg(test)]
mod pipeline_sim_tests {
    use super::*;

    // Simulate what the pipeline does: clone arena into adapter, drop original
    #[test]
    fn test_arena_moved_to_adapter() {
        struct MockSourceAdapter {
            arena: Option<SharedArena>,
        }

        let arena = SharedArena::new(64, 8).unwrap();

        // Clone the arena like PipelineBuilder does
        let cloned = arena.clone();

        // Move clone into adapter (like SourceAdapter::with_arena does)
        let adapter = MockSourceAdapter {
            arena: Some(cloned),
        };

        // Original arena is dropped (like in PipelineBuilder after .source() call)
        drop(arena);

        // Now use the arena in the adapter
        let arena_ref = adapter.arena.as_ref().unwrap();
        let mut slot = arena_ref.acquire().unwrap();

        // Write to the slot
        slot.data_mut()[0] = 42;
        assert_eq!(slot.data()[0], 42);
    }

    #[test]
    fn test_arena_with_box() {
        struct MockSourceAdapter {
            arena: Option<SharedArena>,
        }

        let arena = SharedArena::new(64, 8).unwrap();
        let cloned = arena.clone();

        // Box the adapter like Pipeline does
        let adapter = Box::new(MockSourceAdapter {
            arena: Some(cloned),
        });

        drop(arena);

        // Use boxed adapter
        let arena_ref = adapter.arena.as_ref().unwrap();
        let mut slot = arena_ref.acquire().unwrap();
        slot.data_mut()[0] = 42;
        assert_eq!(slot.data()[0], 42);
    }
}
