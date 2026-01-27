//! Pipeline links for connecting elements across process and network boundaries.
//!
//! This module provides abstractions for transferring buffers between pipeline
//! elements, whether they are in the same process, different processes on the
//! same machine (IPC), or across a network.
//!
//! ## Link Types
//!
//! - [`LocalLink`]: In-process buffered channel (using kanal)
//! - [`IpcPublisher`]/[`IpcSubscriber`]: Cross-process link using shared memory and Unix sockets
//! - [`NetworkSender`]/[`NetworkReceiver`]: TCP-based network link with rkyv serialization
//!
//! ## Zero-Copy Semantics
//!
//! - **Local**: Buffers are moved through channels (Arc clone only)
//! - **IPC**: Buffer data lives in shared memory; only metadata is transferred
//! - **Network**: Buffers are serialized with rkyv at send and validated at receive

mod ipc_link;
mod local;
mod network;

pub use ipc_link::{IpcPublisher, IpcSubscriber};
pub use local::LocalLink;
pub use network::{NetworkReceiver, NetworkSender};
