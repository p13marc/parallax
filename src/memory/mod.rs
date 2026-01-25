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
//! | [`HeapSegment`] | Default, single-process, simple |
//! | [`SharedMemorySegment`] | Multi-process, zero-copy IPC |
//! | [`HugePageSegment`] | Large allocations, reduced TLB misses |
//! | [`MappedFileSegment`] | Persistent storage, file I/O |
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::{MemoryPool, HeapSegment};
//!
//! // Create a pool with 16 slots of 64KB each
//! let pool = MemoryPool::new(HeapSegment::new(16 * 64 * 1024)?, 64 * 1024)?;
//!
//! // Loan a slot
//! let slot = pool.loan().expect("pool not exhausted");
//!
//! // Write data to the slot
//! slot.as_mut_slice()[..5].copy_from_slice(b"hello");
//!
//! // Slot is returned to pool when dropped
//! ```

mod bitmap;
mod heap;
mod huge_pages;
pub mod ipc;
mod mapped_file;
mod pool;
mod segment;
mod shared;

pub use bitmap::AtomicBitmap;
pub use heap::HeapSegment;
pub use huge_pages::{HugePageSegment, HugePageSize};
pub use mapped_file::MappedFileSegment;
pub use pool::{LoanedSlot, MemoryPool};
pub use segment::{IpcHandle, MemorySegment, MemoryType};
pub use shared::SharedMemorySegment;
