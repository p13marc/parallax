//! Inter-process communication elements.
//!
//! - [`IpcSrc`], [`IpcSink`]: Cross-process buffer transfer via shared memory
//! - [`MemorySrc`], [`MemorySink`]: In-memory buffer storage

mod ipc_elements;
mod memory;

pub use ipc_elements::{IpcSink, IpcSrc};
pub use memory::{MemorySink, MemorySinkStats, MemorySrc, SharedMemorySink};
