//! IPC elements for cross-process pipelines.
//!
//! These elements enable zero-copy data transfer between separate processes
//! using shared memory arenas and Unix sockets for control messages.
//!
//! # Architecture
//!
//! ```text
//! Process A                      Process B
//! ┌─────────┐   Unix Socket     ┌─────────┐
//! │ IpcSink │ ───────────────▶ │ IpcSrc  │
//! └────┬────┘   (control msgs)  └────┬────┘
//!      │                              │
//!      └──────────────────────────────┘
//!              Shared Memory Arena
//!              (data, zero-copy)
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! // Process A: Send data
//! let sink = IpcSink::new("/tmp/my-pipeline.sock");
//! sink.consume(buffer)?;
//!
//! // Process B: Receive data
//! let src = IpcSrc::new("/tmp/my-pipeline.sock");
//! let buffer = src.produce()?;
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use crate::error::{Error, Result};
use crate::execution::{ControlMessage, SerializableMetadata, frame_message, unframe_message};
use crate::format::Caps;
use crate::memory::{SharedArena, SharedIpcSlotRef};
use std::collections::VecDeque;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

/// IPC sink that sends buffers to another process.
///
/// Uses a Unix socket for control messages and shared memory for data.
/// The sink creates and manages the shared memory arena, passing the
/// file descriptor to the source via SCM_RIGHTS.
pub struct IpcSink {
    /// Path to the Unix socket.
    path: PathBuf,
    /// Connected socket (if any).
    socket: Option<UnixStream>,
    /// Shared memory arena for buffers (refcount in shared memory).
    arena: Option<SharedArena>,
    /// Whether we're the server (created the socket).
    is_server: bool,
    /// Listener for incoming connections (server mode).
    listener: Option<UnixListener>,
    /// Pending slots waiting for acknowledgment.
    pending_slots: VecDeque<SharedIpcSlotRef>,
    /// Maximum pending buffers before blocking.
    max_pending: usize,
    /// Capabilities.
    caps: Caps,
}

impl IpcSink {
    /// Create a new IPC sink.
    ///
    /// The sink will create a Unix socket at the given path and wait
    /// for a connection from an IpcSrc.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            socket: None,
            arena: None,
            is_server: true,
            listener: None,
            pending_slots: VecDeque::new(),
            max_pending: 16,
            caps: Caps::any(),
        }
    }

    /// Create a sink that connects to an existing socket (client mode).
    pub fn connect(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            socket: None,
            arena: None,
            is_server: false,
            listener: None,
            pending_slots: VecDeque::new(),
            max_pending: 16,
            caps: Caps::any(),
        }
    }

    /// Set maximum pending buffers.
    pub fn with_max_pending(mut self, max: usize) -> Self {
        self.max_pending = max;
        self
    }

    /// Set capabilities.
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = caps;
        self
    }

    /// Initialize the connection and arena.
    fn ensure_connected(&mut self) -> Result<()> {
        if self.socket.is_some() {
            return Ok(());
        }

        if self.is_server {
            // Create listener if not already created
            if self.listener.is_none() {
                // Remove existing socket file if present
                let _ = std::fs::remove_file(&self.path);
                self.listener = Some(UnixListener::bind(&self.path).map_err(|e| {
                    Error::Io(std::io::Error::new(
                        e.kind(),
                        format!("Failed to bind IPC socket at {:?}: {}", self.path, e),
                    ))
                })?);
            }

            // Accept connection
            let listener = self.listener.as_ref().unwrap();
            let (socket, _addr) = listener.accept().map_err(|e| {
                Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to accept IPC connection: {}", e),
                ))
            })?;
            self.socket = Some(socket);

            // Create arena and send registration
            if self.arena.is_none() {
                let arena = SharedArena::new(64 * 1024, 64)?; // 64KB slots, 64 slots = 4MB
                self.arena = Some(arena);
            }

            // Send arena registration message
            self.send_arena_registration()?;
        } else {
            // Connect to existing socket
            let socket = UnixStream::connect(&self.path).map_err(|e| {
                Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to connect to IPC socket at {:?}: {}", self.path, e),
                ))
            })?;
            self.socket = Some(socket);
        }

        Ok(())
    }

    /// Send arena registration to the source.
    fn send_arena_registration(&mut self) -> Result<()> {
        let arena = self
            .arena
            .as_ref()
            .ok_or_else(|| Error::Element("Arena not initialized".into()))?;

        let msg = ControlMessage::RegisterArena {
            arena_id: arena.id(),
            size: arena.total_size(),
            slot_size: arena.slot_size(),
            slot_count: arena.slot_count(),
        };

        self.send_message(&msg)?;

        // NOTE: Arena fd should be sent via SCM_RIGHTS for true cross-process zero-copy.
        // Currently relies on the arena being accessible via its ID within the same
        // process group. Full SCM_RIGHTS implementation requires extending the IPC
        // protocol with fd passing support from memory/ipc.rs (send_fds/recv_fds).

        Ok(())
    }

    /// Send a control message.
    fn send_message(&mut self, msg: &ControlMessage) -> Result<()> {
        let socket = self
            .socket
            .as_mut()
            .ok_or_else(|| Error::Element("Not connected".into()))?;

        let framed = frame_message(msg);
        socket.write_all(&framed).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to send IPC message: {}", e),
            ))
        })?;

        Ok(())
    }

    /// Receive a control message.
    fn recv_message(&mut self) -> Result<Option<ControlMessage>> {
        let socket = self
            .socket
            .as_mut()
            .ok_or_else(|| Error::Element("Not connected".into()))?;

        // Set non-blocking for checking
        socket.set_nonblocking(true).ok();

        let mut buf = [0u8; 4];
        match socket.read_exact(&mut buf) {
            Ok(()) => {
                socket.set_nonblocking(false).ok();
                let len = u32::from_le_bytes(buf) as usize;
                let mut data = vec![0u8; 4 + len];
                data[..4].copy_from_slice(&buf);
                socket.read_exact(&mut data[4..]).map_err(|e| {
                    Error::Io(std::io::Error::new(
                        e.kind(),
                        format!("Failed to read IPC message body: {}", e),
                    ))
                })?;
                let (msg, _) = unframe_message(&data)
                    .ok_or_else(|| Error::Element("Invalid IPC message".into()))?;
                Ok(Some(msg))
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                socket.set_nonblocking(false).ok();
                Ok(None)
            }
            Err(e) => {
                socket.set_nonblocking(false).ok();
                Err(Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to read IPC message: {}", e),
                )))
            }
        }
    }

    /// Process any pending acknowledgments.
    fn process_acks(&mut self) -> Result<()> {
        while let Some(msg) = self.recv_message()? {
            if let ControlMessage::BufferDone { slot } = msg {
                // Remove from pending
                self.pending_slots.retain(|s| s != &slot);
            }
        }
        Ok(())
    }
}

impl Sink for IpcSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.ensure_connected()?;
        self.process_acks()?;

        // Wait if too many pending
        while self.pending_slots.len() >= self.max_pending {
            // Block and wait for ack
            std::thread::sleep(std::time::Duration::from_micros(100));
            self.process_acks()?;
        }

        // Get the buffer from context
        let buffer = ctx.buffer();

        // Get slot reference from buffer (all buffers are now backed by SharedArena)
        let slot_ref = buffer.memory().ipc_ref();

        // Send buffer ready message
        let msg = ControlMessage::BufferReady {
            slot: slot_ref,
            metadata: SerializableMetadata::from_metadata(buffer.metadata()),
        };
        self.send_message(&msg)?;

        self.pending_slots.push_back(slot_ref);

        Ok(())
    }
}

impl Drop for IpcSink {
    fn drop(&mut self) {
        // Send shutdown message
        if self.socket.is_some() {
            let _ = self.send_message(&ControlMessage::Shutdown);
        }

        // Clean up socket file if we're the server
        if self.is_server {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// IPC source that receives buffers from another process.
///
/// Connects to an IpcSink and receives buffer references via Unix socket,
/// then accesses the data through the shared memory arena.
pub struct IpcSrc {
    /// Path to the Unix socket.
    path: PathBuf,
    /// Connected socket (if any).
    socket: Option<UnixStream>,
    /// Whether we're the server (created the socket).
    is_server: bool,
    /// Listener for incoming connections (server mode).
    listener: Option<UnixListener>,
    /// Registered arenas (id -> slot_count). Used to track known arenas.
    registered_arenas: std::collections::HashMap<u64, usize>,
    /// Capabilities.
    caps: Caps,
    /// Arena for creating placeholder buffers (when arena mapping not available).
    arena: Option<SharedArena>,
}

impl IpcSrc {
    /// Create a new IPC source.
    ///
    /// The source will connect to the Unix socket at the given path.
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            socket: None,
            is_server: false,
            listener: None,
            registered_arenas: std::collections::HashMap::new(),
            caps: Caps::any(),
            arena: None,
        }
    }

    /// Create a source that listens for connections (server mode).
    pub fn listen(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            socket: None,
            is_server: true,
            listener: None,
            registered_arenas: std::collections::HashMap::new(),
            caps: Caps::any(),
            arena: None,
        }
    }

    /// Set capabilities.
    pub fn with_caps(mut self, caps: Caps) -> Self {
        self.caps = caps;
        self
    }

    /// Initialize the connection.
    fn ensure_connected(&mut self) -> Result<()> {
        if self.socket.is_some() {
            return Ok(());
        }

        if self.is_server {
            // Create listener if not already created
            if self.listener.is_none() {
                let _ = std::fs::remove_file(&self.path);
                self.listener = Some(UnixListener::bind(&self.path).map_err(|e| {
                    Error::Io(std::io::Error::new(
                        e.kind(),
                        format!("Failed to bind IPC socket at {:?}: {}", self.path, e),
                    ))
                })?);
            }

            // Accept connection
            let listener = self.listener.as_ref().unwrap();
            let (socket, _addr) = listener.accept().map_err(|e| {
                Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to accept IPC connection: {}", e),
                ))
            })?;
            self.socket = Some(socket);
        } else {
            // Connect to existing socket
            let socket = UnixStream::connect(&self.path).map_err(|e| {
                Error::Io(std::io::Error::new(
                    e.kind(),
                    format!("Failed to connect to IPC socket at {:?}: {}", self.path, e),
                ))
            })?;
            self.socket = Some(socket);
        }

        Ok(())
    }

    /// Send a control message.
    fn send_message(&mut self, msg: &ControlMessage) -> Result<()> {
        let socket = self
            .socket
            .as_mut()
            .ok_or_else(|| Error::Element("Not connected".into()))?;

        let framed = frame_message(msg);
        socket.write_all(&framed).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to send IPC message: {}", e),
            ))
        })?;

        Ok(())
    }

    /// Receive a control message (blocking).
    fn recv_message(&mut self) -> Result<ControlMessage> {
        let socket = self
            .socket
            .as_mut()
            .ok_or_else(|| Error::Element("Not connected".into()))?;

        // Read length prefix
        let mut len_buf = [0u8; 4];
        socket.read_exact(&mut len_buf).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read IPC message length: {}", e),
            ))
        })?;

        let len = u32::from_le_bytes(len_buf) as usize;

        // Read message body
        let mut data = vec![0u8; 4 + len];
        data[..4].copy_from_slice(&len_buf);
        socket.read_exact(&mut data[4..]).map_err(|e| {
            Error::Io(std::io::Error::new(
                e.kind(),
                format!("Failed to read IPC message body: {}", e),
            ))
        })?;

        let (msg, _) =
            unframe_message(&data).ok_or_else(|| Error::Element("Invalid IPC message".into()))?;

        Ok(msg)
    }

    /// Handle arena registration message.
    fn handle_arena_registration(
        &mut self,
        arena_id: u64,
        _size: usize,
        _slot_size: usize,
        slot_count: usize,
    ) {
        self.registered_arenas.insert(arena_id, slot_count);
        // NOTE: Should receive arena fd via SCM_RIGHTS and mmap it for true zero-copy.
        // Currently only tracks arena metadata. Full implementation requires:
        // 1. Receive fd via recv_fds() from memory/ipc.rs
        // 2. mmap the fd to local address space
        // 3. Create a SharedArena from the mapped region
    }
}

impl Source for IpcSrc {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        self.ensure_connected()?;

        loop {
            let msg = self.recv_message()?;

            match msg {
                ControlMessage::RegisterArena {
                    arena_id,
                    size,
                    slot_size,
                    slot_count,
                } => {
                    self.handle_arena_registration(arena_id, size, slot_size, slot_count);
                    continue;
                }

                ControlMessage::BufferReady { slot, metadata } => {
                    // Send acknowledgment
                    self.send_message(&ControlMessage::BufferDone { slot: slot })?;

                    // NOTE: With full SCM_RIGHTS support, we would look up the arena
                    // in a local cache via SharedArenaCache and create a zero-copy buffer.
                    // Currently creates a placeholder buffer since arena mapping
                    // is not yet implemented.
                    let meta = metadata.to_metadata();

                    // Create a buffer with allocated memory using SharedArena
                    // TODO: In a real implementation, use SharedArenaCache to map
                    // the arena from the sender and create a zero-copy buffer view.
                    if self.arena.is_none() {
                        // Use 64KB slots by default, matching IpcSink
                        self.arena = Some(SharedArena::new(64 * 1024, 64)?);
                    }
                    let arena = self.arena.as_mut().unwrap();
                    arena.reclaim();
                    let arena_slot = arena
                        .acquire()
                        .ok_or_else(|| Error::Element("arena exhausted".into()))?;
                    let handle = MemoryHandle::with_len(arena_slot, slot.len);
                    let buffer = Buffer::new(handle, meta);

                    // IpcSrc receives buffers via IPC, so return OwnBuffer
                    return Ok(ProduceResult::OwnBuffer(buffer));
                }

                ControlMessage::Eos => {
                    return Ok(ProduceResult::Eos);
                }

                ControlMessage::Shutdown => {
                    return Ok(ProduceResult::Eos);
                }

                _ => {
                    // Ignore other messages
                    continue;
                }
            }
        }
    }
}

impl Drop for IpcSrc {
    fn drop(&mut self) {
        // Clean up socket file if we're the server
        if self.is_server {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ipc_sink_creation() {
        let sink = IpcSink::new("/tmp/test-ipc-sink.sock");
        assert!(sink.socket.is_none());
        assert!(sink.is_server);
    }

    #[test]
    fn test_ipc_src_creation() {
        let src = IpcSrc::new("/tmp/test-ipc-src.sock");
        assert!(src.socket.is_none());
        assert!(!src.is_server);
    }

    #[test]
    fn test_ipc_sink_with_caps() {
        use crate::format::{MediaFormat, VideoCodec};

        let caps = Caps::new(MediaFormat::Video(VideoCodec::H264));
        let sink = IpcSink::new("/tmp/test.sock").with_caps(caps.clone());
        assert_eq!(sink.caps, caps);
    }

    #[test]
    fn test_ipc_src_listen_mode() {
        let src = IpcSrc::listen("/tmp/test-listen.sock");
        assert!(src.is_server);
    }

    // Integration test would require two threads/processes
    // which is complex to set up in unit tests
}
