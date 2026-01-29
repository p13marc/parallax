//! Memory management for Parallax.
//!
//! This module provides the memory abstraction layer that enables zero-copy
//! buffer sharing across threads and processes.
//!
//! # Architecture
//!
//! - [`SharedArena`]: Arena allocator with cross-process reference counting
//! - [`SharedSlotRef`]: Zero-allocation slot reference
//! - [`FixedBufferPool`]: Pipeline-level buffer pool with backpressure
//!
//! # Memory Backends
//!
//! | Backend | Use Case |
//! |---------|----------|
//! | [`SharedArena`] | **Primary**: Cross-process zero-copy buffers |
//! | [`HugePageSegment`] | Large allocations, reduced TLB misses |
//! | [`MappedFileSegment`] | Persistent storage, file I/O |
//! | [`DmaBufSegment`] | GPU-importable buffers, zero-copy capture |
//!
//! # Design Rationale
//!
//! All CPU memory is backed by `memfd_create + MAP_SHARED`, which has zero
//! overhead vs malloc but is always shareable via fd passing. This means:
//!
//! - Every buffer is automatically shareable across processes
//! - No conversion needed before IPC
//! - Cross-process = same physical pages (true zero-copy)
//! - Reference counts stored in shared memory (not on heap)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::{SharedArena, SharedSlotRef};
//!
//! // Create arena with 16 slots of 64KB each
//! let arena = SharedArena::new(64 * 1024, 16)?;
//!
//! // Acquire a slot
//! let slot = arena.acquire().expect("arena not exhausted");
//!
//! // Write data
//! slot.data_mut()[..5].copy_from_slice(b"hello");
//!
//! // Get IPC reference for cross-process sharing
//! let ipc_ref = slot.ipc_ref();
//! // Send ipc_ref over Unix socket...
//! ```

mod bitmap;
mod buffer_pool;
pub mod defaults;
mod dmabuf;
mod huge_pages;
pub mod ipc;
mod mapped_file;
mod segment;
mod shared_refcount;

pub use bitmap::AtomicBitmap;
pub use buffer_pool::{BufferPool, FixedBufferPool, PoolStats, PooledBuffer};
pub use dmabuf::DmaBufSegment;
pub use huge_pages::{HugePageSegment, HugePageSize};
pub use mapped_file::MappedFileSegment;
pub use segment::{IpcHandle, MemorySegment, MemoryType};
pub use shared_refcount::{
    ArenaMetrics, SharedArena, SharedArenaCache, SharedIpcSlotRef, SharedSlotRef,
};
