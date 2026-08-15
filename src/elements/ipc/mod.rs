//! Inter-process communication elements.
//!
//! - [`IpcSrc`], [`IpcSink`]: Cross-process buffer transfer via shared memory

mod ipc_elements;
pub(crate) mod protocol;

pub use ipc_elements::{IpcSink, IpcSrc};
