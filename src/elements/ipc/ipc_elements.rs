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

use super::protocol::{ControlMessage, SerializableMetadata, frame_message, unframe_message};
use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use crate::error::{Error, Result};
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

    /// Send arena registration to the source via SCM_RIGHTS.
    ///
    /// Sends both the control message and the arena file descriptor in a single
    /// `sendmsg()` call using Unix domain socket ancillary data.
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

        let msg_bytes = frame_message(&msg);

        // Send arena fd + registration message via SCM_RIGHTS
        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| Error::Element("Not connected".into()))?;

        crate::memory::ipc::send_fds(socket, &[arena.fd()], &msg_bytes)?;

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
    /// Mapped arenas received via SCM_RIGHTS (id -> SharedArena).
    arena_cache: std::collections::HashMap<u64, SharedArena>,
    /// Capabilities.
    caps: Caps,
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
            arena_cache: std::collections::HashMap::new(),
            caps: Caps::any(),
        }
    }

    /// Create a source that listens for connections (server mode).
    pub fn listen(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            socket: None,
            is_server: true,
            listener: None,
            arena_cache: std::collections::HashMap::new(),
            caps: Caps::any(),
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

    /// Receive arena registration with fd via SCM_RIGHTS.
    ///
    /// This receives both the control message and the arena fd in a single
    /// `recvmsg()` call, then maps the arena into this process's address space.
    fn receive_arena_registration(&mut self) -> Result<()> {
        let socket = self
            .socket
            .as_ref()
            .ok_or_else(|| Error::Element("Not connected".into()))?;

        let mut data_buf = vec![0u8; 4096];
        let (bytes_read, fds) = crate::memory::ipc::recv_fds(socket, &mut data_buf)?;

        if fds.is_empty() {
            return Err(Error::Element("No arena fd received".into()));
        }

        // Parse the control message
        let (msg, _) = unframe_message(&data_buf[..bytes_read])
            .ok_or_else(|| Error::Element("Invalid arena registration message".into()))?;

        if let ControlMessage::RegisterArena { arena_id, .. } = msg {
            // Map the arena from the received fd
            let fd = fds.into_iter().next().unwrap();
            let arena = unsafe { SharedArena::from_fd(fd)? };
            self.arena_cache.insert(arena_id, arena);
            Ok(())
        } else {
            Err(Error::Element(format!(
                "Expected RegisterArena, got {:?}",
                msg
            )))
        }
    }
}

impl Source for IpcSrc {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        self.ensure_connected()?;

        // If no arenas registered yet, receive the arena registration first
        if self.arena_cache.is_empty() {
            self.receive_arena_registration()?;
        }

        loop {
            let msg = self.recv_message()?;

            match msg {
                ControlMessage::BufferReady { slot, metadata } => {
                    let meta = metadata.to_metadata();

                    // Look up the arena by ID for true zero-copy access
                    let arena = self.arena_cache.get(&slot.arena_id).ok_or_else(|| {
                        Error::Element(format!("unknown arena {}", slot.arena_id))
                    })?;

                    // Get a reference to the same shared memory slot (zero-copy)
                    let arena_slot = arena.slot_from_ipc(&slot).ok_or_else(|| {
                        Error::Element(format!(
                            "invalid slot {} in arena {}",
                            slot.slot_index, slot.arena_id
                        ))
                    })?;

                    let handle = MemoryHandle::with_len(arena_slot, slot.len);
                    let buffer = Buffer::new(handle, meta);

                    // Send acknowledgment after we have a reference to the slot
                    self.send_message(&ControlMessage::BufferDone { slot })?;

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
