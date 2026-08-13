//! Pipeline links for connecting elements across process and network boundaries.
//!
//! This module provides abstractions for transferring buffers between pipeline
//! elements, whether they are in the same process, different processes on the
//! same machine (IPC), or across a network.
//!
//! ## Link Types
//!
//! - [`IpcPublisher`]/[`IpcSubscriber`]: Cross-process link using shared memory and Unix sockets
//! - [`NetworkSender`]/[`NetworkReceiver`]: TCP-based network link with rkyv serialization
//!
//! In-process links need no type here: the executor gives every graph edge its
//! own `tokio::sync::mpsc` channel, sized by `Pipeline::link_pads_full` and
//! governed by its [`LinkPolicy`](crate::pipeline::LinkPolicy).
//!
//! ## Zero-Copy Semantics
//!
//! - **In-process**: Buffers are moved through channels (refcount only)
//! - **IPC**: Buffer data lives in shared memory; only metadata is transferred
//! - **Network**: Buffers are serialized with rkyv at send and validated at receive

mod ipc_link;
mod network;

pub use ipc_link::{IpcPublisher, IpcSubscriber};
pub use network::{NetworkReceiver, NetworkSender};
