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
//! # Design Rationale
//!
//! All CPU memory is backed by `memfd_create + MAP_SHARED`, which has zero
//! overhead vs malloc but is always shareable via fd passing. This means:
//!
//! - Every buffer is automatically shareable across processes
//! - No conversion needed before IPC
//! - Cross-process = same physical pages (true zero-copy)
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
mod buffer_pool;
mod cpu;
mod huge_pages;
pub mod ipc;
mod mapped_file;
mod pool;
mod segment;
mod shared_refcount;

pub use arena::{Access, ArenaCache, ArenaSlot, CpuArena, IpcSlotRef};
pub use bitmap::AtomicBitmap;
pub use buffer_pool::{BufferPool, FixedBufferPool, PoolStats, PooledBuffer};
pub use cpu::CpuSegment;
pub use huge_pages::{HugePageSegment, HugePageSize};
pub use mapped_file::MappedFileSegment;
pub use pool::{LoanedSlot, MemoryPool};
pub use segment::{IpcHandle, MemorySegment, MemoryType};
pub use shared_refcount::{SharedArena, SharedArenaCache, SharedIpcSlotRef, SharedSlotRef};
