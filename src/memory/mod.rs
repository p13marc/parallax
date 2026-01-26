//! Memory management for Parallax.
//!
//! This module provides the memory abstraction layer that enables zero-copy
//! buffer sharing across threads and processes.
//!
//! # Architecture
//!
//! - [`MemorySegment`]: Trait for different memory backends
//! - [`MemoryPool`]: Loan-based memory pool for efficient buffer allocation
//! - [`LoanedSlot`]: RAII guard that returns memory to pool on drop
//!
//! # Memory Backends
//!
//! | Backend | Use Case |
//! |---------|----------|
//! | [`CpuSegment`] | **Primary**: All CPU memory, always IPC-ready |
//! | [`HugePageSegment`] | Large allocations, reduced TLB misses |
//! | [`MappedFileSegment`] | Persistent storage, file I/O |
//!
//! # Deprecated Backends
//!
//! | Backend | Replacement |
//! |---------|-------------|
//! | `HeapSegment` | Use [`CpuSegment`] - same performance, always shareable |
//! | `SharedMemorySegment` | Use [`CpuSegment`] - unified type for all CPU memory |
//!
//! # Design Rationale
//!
//! Previously, Parallax had separate `HeapSegment` (malloc-backed) and
//! `SharedMemorySegment` (memfd-backed). This was unnecessary because
//! `memfd_create + MAP_SHARED` has zero overhead vs malloc but is always
//! shareable. Now all CPU memory uses [`CpuSegment`].
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::{CpuSegment, MemorySegment};
//!
//! // Allocate CPU memory (works like malloc, but shareable)
//! let segment = CpuSegment::new(64 * 1024)?;
//!
//! // Write data
//! segment.as_mut_slice()[..5].copy_from_slice(b"hello");
//!
//! // Get fd for cross-process sharing (always available!)
//! let fd = segment.fd();
//! // Send fd via SCM_RIGHTS...
//! ```

mod arena;
mod bitmap;
mod cpu;
mod heap;
mod huge_pages;
pub mod ipc;
mod mapped_file;
mod pool;
mod segment;
mod shared;
mod shared_refcount;

pub use arena::{Access, ArenaCache, ArenaSlot, CpuArena, IpcSlotRef};
pub use bitmap::AtomicBitmap;
pub use cpu::CpuSegment;
pub use huge_pages::{HugePageSegment, HugePageSize};
pub use mapped_file::MappedFileSegment;
pub use pool::{LoanedSlot, MemoryPool};
pub use segment::{IpcHandle, MemorySegment, MemoryType};
pub use shared_refcount::{SharedArena, SharedArenaCache, SharedIpcSlotRef, SharedSlotRef};

// Deprecated re-exports (kept for backward compatibility)
#[deprecated(since = "0.2.0", note = "Use CpuSegment instead")]
pub use heap::HeapSegment;
#[deprecated(since = "0.2.0", note = "Use CpuSegment instead")]
pub use shared::SharedMemorySegment;
