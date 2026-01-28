//! IPC (Inter-Process Communication) links using shared memory and Unix sockets.
//!
//! This module provides zero-copy buffer transfer between processes on the same
//! machine using:
//! - Shared memory (memfd_create) for buffer data
//! - Unix domain sockets for signaling and metadata transfer
//! - SCM_RIGHTS for passing file descriptors

use crate::buffer::{Buffer, MemoryHandle};
use crate::error::{Error, Result};
use crate::memory::{SharedArena, SharedArenaCache, ipc};
use crate::metadata::Metadata;

use std::collections::HashMap;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::Path;

/// Message types for IPC protocol.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MessageType {
    /// New shared memory segment (carries fd)
    NewSegment = 1,
    /// Buffer using existing segment (carries offset, len, metadata)
    Buffer = 2,
    /// End of stream
    Eos = 3,
    /// Error message
    Error = 4,
}

impl TryFrom<u8> for MessageType {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(MessageType::NewSegment),
            2 => Ok(MessageType::Buffer),
            3 => Ok(MessageType::Eos),
            4 => Ok(MessageType::Error),
            _ => Err(Error::Pipeline(format!("unknown message type: {}", value))),
        }
    }
}

/// Header for IPC messages.
/// Note: Using individual reads/writes to avoid packed struct alignment issues.
#[derive(Debug, Clone, Copy)]
struct MessageHeader {
    msg_type: u8,
    segment_id: u32,
    field1: u64,
    field2: u64,
    sequence: u64,
}

impl MessageHeader {
    const SIZE: usize = 1 + 4 + 8 + 8 + 8; // 29 bytes

    fn to_bytes(self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0] = self.msg_type;
        buf[1..5].copy_from_slice(&self.segment_id.to_le_bytes());
        buf[5..13].copy_from_slice(&self.field1.to_le_bytes());
        buf[13..21].copy_from_slice(&self.field2.to_le_bytes());
        buf[21..29].copy_from_slice(&self.sequence.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(Error::Pipeline("message header too short".into()));
        }
        Ok(Self {
            msg_type: bytes[0],
            segment_id: u32::from_le_bytes(bytes[1..5].try_into().unwrap()),
            field1: u64::from_le_bytes(bytes[5..13].try_into().unwrap()),
            field2: u64::from_le_bytes(bytes[13..21].try_into().unwrap()),
            sequence: u64::from_le_bytes(bytes[21..29].try_into().unwrap()),
        })
    }
}

/// Publisher side of an IPC link.
///
/// Sends buffers to subscribers using shared memory for zero-copy transfer.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::link::IpcPublisher;
/// use parallax::buffer::Buffer;
///
/// // Create publisher listening on a socket
/// let mut publisher = IpcPublisher::bind("/tmp/pipeline.sock")?;
///
/// // Wait for a subscriber
/// publisher.accept()?;
///
/// // Send buffers
/// publisher.send(buffer)?;
///
/// // Signal end of stream
/// publisher.send_eos()?;
/// ```
pub struct IpcPublisher {
    listener: Option<UnixListener>,
    stream: Option<UnixStream>,
    /// Map from segment base pointer to segment_id (for deduplication)
    segment_ids: HashMap<usize, u32>,
    next_segment_id: u32,
}

impl IpcPublisher {
    /// Create a new publisher bound to the given socket path.
    ///
    /// The socket file will be created. If it already exists, it will be removed.
    pub fn bind<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();

        // Remove existing socket if present
        if path.exists() {
            std::fs::remove_file(path)?;
        }

        let listener = UnixListener::bind(path)?;

        Ok(Self {
            listener: Some(listener),
            stream: None,
            segment_ids: HashMap::new(),
            next_segment_id: 0,
        })
    }

    /// Create a publisher from an existing Unix socket pair.
    ///
    /// Useful for testing or when sockets are created externally.
    pub fn from_stream(stream: UnixStream) -> Self {
        Self {
            listener: None,
            stream: Some(stream),
            segment_ids: HashMap::new(),
            next_segment_id: 0,
        }
    }

    /// Accept a subscriber connection.
    ///
    /// Blocks until a subscriber connects. Only one subscriber is supported
    /// per publisher.
    pub fn accept(&mut self) -> Result<()> {
        if self.stream.is_some() {
            return Err(Error::Pipeline("already have a subscriber".into()));
        }

        let listener = self
            .listener
            .as_ref()
            .ok_or_else(|| Error::Pipeline("no listener".into()))?;

        let (stream, _addr) = listener.accept()?;
        self.stream = Some(stream);

        Ok(())
    }

    /// Send a buffer to the subscriber.
    ///
    /// If the buffer's backing segment hasn't been sent before, it will be
    /// sent first (with the fd), then the buffer metadata is sent.
    pub fn send(&mut self, buffer: Buffer) -> Result<()> {
        let memory = buffer.memory();

        // Ensure the arena fd has been sent to the subscriber
        let segment_id = self.ensure_segment_sent(&buffer)?;

        // Get the IPC reference which has slot_index and data_offset
        let ipc_ref = memory.ipc_ref();

        // Pack slot_index in upper 32 bits of field1, data_offset in lower 32 bits
        let field1 =
            ((ipc_ref.slot_index as u64) << 32) | (ipc_ref.data_offset as u64 & 0xFFFFFFFF);

        // Send buffer message
        let header = MessageHeader {
            msg_type: MessageType::Buffer as u8,
            segment_id,
            field1,
            field2: ipc_ref.len as u64,
            sequence: buffer.metadata().sequence,
        };

        use std::io::Write;
        let stream = self.stream.as_mut().unwrap();
        stream.write_all(&header.to_bytes())?;

        Ok(())
    }

    /// Send end-of-stream signal.
    pub fn send_eos(&mut self) -> Result<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("no subscriber connected".into()))?;

        let header = MessageHeader {
            msg_type: MessageType::Eos as u8,
            segment_id: 0,
            field1: 0,
            field2: 0,
            sequence: 0,
        };

        use std::io::Write;
        stream.write_all(&header.to_bytes())?;

        Ok(())
    }

    fn ensure_segment_sent(&mut self, buffer: &Buffer) -> Result<u32> {
        let memory = buffer.memory();

        // MemoryHandle now only uses SharedSlotRef, use arena_id as the key
        let arena_id = memory.arena_id();
        let fd = memory.arena_fd();
        let size = memory.arena_size();

        // Check if we've already sent this arena
        if let Some(&id) = self.segment_ids.get(&(arena_id as usize)) {
            return Ok(id);
        }

        let id = self.next_segment_id;
        self.next_segment_id += 1;

        // Send NewSegment message with fd
        let header = MessageHeader {
            msg_type: MessageType::NewSegment as u8,
            segment_id: id,
            field1: size as u64,
            field2: 0,
            sequence: 0,
        };

        let stream = self
            .stream
            .as_ref()
            .ok_or_else(|| Error::Pipeline("no subscriber connected".into()))?;

        // Use BorrowedFd to satisfy AsFd trait
        use std::os::unix::io::BorrowedFd;
        let borrowed_fd = unsafe { BorrowedFd::borrow_raw(fd) };
        ipc::send_fds(stream, &[borrowed_fd], &header.to_bytes())?;

        // Remember this arena by its id
        self.segment_ids.insert(arena_id as usize, id);

        Ok(id)
    }
}

/// Subscriber side of an IPC link.
///
/// Receives buffers from a publisher using shared memory for zero-copy transfer.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::link::IpcSubscriber;
///
/// // Connect to a publisher
/// let mut subscriber = IpcSubscriber::connect("/tmp/pipeline.sock")?;
///
/// // Receive buffers
/// while let Some(buffer) = subscriber.recv()? {
///     // Process buffer - data is in shared memory, no copy!
///     println!("Got {} bytes", buffer.len());
/// }
/// ```
pub struct IpcSubscriber {
    stream: UnixStream,
    /// Arenas we've received from the publisher (segment_id -> arena)
    arenas: HashMap<u32, SharedArena>,
    /// Cache for slot references
    arena_cache: SharedArenaCache,
}

impl IpcSubscriber {
    /// Connect to a publisher at the given socket path.
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self> {
        let stream = UnixStream::connect(path)?;

        Ok(Self {
            stream,
            arenas: HashMap::new(),
            arena_cache: SharedArenaCache::new(),
        })
    }

    /// Create a subscriber from an existing Unix socket.
    pub fn from_stream(stream: UnixStream) -> Self {
        Self {
            stream,
            arenas: HashMap::new(),
            arena_cache: SharedArenaCache::new(),
        }
    }

    /// Receive a buffer from the publisher.
    ///
    /// Returns `Ok(None)` on end-of-stream.
    pub fn recv(&mut self) -> Result<Option<Buffer>> {
        use std::io::Read;

        let mut header_buf = [0u8; MessageHeader::SIZE];

        // Try to receive - might be a message with fd or without
        let mut ancillary_buf = [std::mem::MaybeUninit::uninit(); 64];
        let mut ancillary = rustix::net::RecvAncillaryBuffer::new(&mut ancillary_buf);

        let mut iov = [std::io::IoSliceMut::new(&mut header_buf)];
        let result = rustix::net::recvmsg(
            &self.stream,
            &mut iov,
            &mut ancillary,
            rustix::net::RecvFlags::empty(),
        )?;

        if result.bytes == 0 {
            // Connection closed
            return Ok(None);
        }

        if result.bytes < header_buf.len() {
            // Read the rest
            self.stream.read_exact(&mut header_buf[result.bytes..])?;
        }

        let header = MessageHeader::from_bytes(&header_buf)?;
        let msg_type = MessageType::try_from(header.msg_type)?;

        // Extract any file descriptors from ancillary data
        let mut received_fds = Vec::new();
        for msg in ancillary.drain() {
            if let rustix::net::RecvAncillaryMessage::ScmRights(rights) = msg {
                for fd in rights {
                    received_fds.push(fd);
                }
            }
        }

        match msg_type {
            MessageType::NewSegment => {
                // Should have received an fd
                let fd = received_fds
                    .into_iter()
                    .next()
                    .ok_or_else(|| Error::Pipeline("expected fd for NewSegment".into()))?;

                // Map the SharedArena from the received fd
                let arena = unsafe { SharedArena::from_fd(fd)? };
                let arena_id = arena.id();
                self.arenas.insert(header.segment_id, arena);

                // Arena ID is stored in the arena itself
                let _ = arena_id;

                // Recurse to get the actual buffer
                self.recv()
            }
            MessageType::Buffer => {
                let segment_id = header.segment_id;
                let arena = self.arenas.get(&segment_id).ok_or_else(|| {
                    Error::Pipeline(format!("unknown segment id: {}", segment_id))
                })?;

                // Unpack slot_index from upper 32 bits, data_offset from lower 32 bits
                let slot_index = (header.field1 >> 32) as u32;
                let data_offset = (header.field1 & 0xFFFFFFFF) as usize;
                let len = header.field2 as usize;

                // Create an IPC slot reference to reconstruct the slot
                let ipc_ref = crate::memory::SharedIpcSlotRef {
                    arena_id: arena.id(),
                    slot_index,
                    data_offset,
                    len,
                };

                // Get a slot reference from the arena
                // Since the old protocol doesn't track slot indices properly,
                // we need to create the handle differently.
                // For now, we acquire a new slot and copy the reference data.
                // In a proper implementation, the publisher would send the slot_index.

                // Actually, the publisher sends buffers that reference memory in the arena.
                // The offset is relative to the arena base. We need to find which slot
                // contains this offset, or use the arena's slot_from_ipc if available.

                let slot = arena
                    .slot_from_ipc(&ipc_ref)
                    .ok_or_else(|| Error::Pipeline("failed to reconstruct slot from IPC".into()))?;

                let handle = MemoryHandle::with_len(slot, len);
                let metadata = Metadata::from_sequence(header.sequence);

                Ok(Some(Buffer::new(handle, metadata)))
            }
            MessageType::Eos => Ok(None),
            MessageType::Error => Err(Error::Pipeline("publisher sent error".into())),
        }
    }

    /// Create an iterator over received buffers.
    pub fn iter(&mut self) -> impl Iterator<Item = Result<Buffer>> + '_ {
        std::iter::from_fn(move || match self.recv() {
            Ok(Some(buf)) => Some(Ok(buf)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SharedArena;
    use std::thread;

    #[test]
    fn test_ipc_link_socket_pair() {
        let (pub_stream, sub_stream) = UnixStream::pair().unwrap();

        let mut publisher = IpcPublisher::from_stream(pub_stream);
        let mut subscriber = IpcSubscriber::from_stream(sub_stream);

        // Create a shared arena with enough slots - clone for use across threads
        let arena = SharedArena::with_name("test-ipc-link", 4096, 16).unwrap();
        let arena_clone = arena.clone();

        // Producer thread
        let producer = thread::spawn(move || {
            for i in 0..10u64 {
                // Acquire a slot and write data
                let mut slot = arena_clone.acquire().expect("arena should have slots");
                let ptr = slot.data_mut().as_mut_ptr() as *mut u64;
                unsafe {
                    *ptr = i;
                }

                let handle = MemoryHandle::with_len(slot, 8);
                let buffer = Buffer::new(handle, Metadata::from_sequence(i));

                publisher.send(buffer).unwrap();
            }
            publisher.send_eos().unwrap();
        });

        // Consumer
        let mut received = Vec::new();
        while let Some(buffer) = subscriber.recv().unwrap() {
            let ptr = buffer.as_bytes().as_ptr() as *const u64;
            let value = unsafe { *ptr };
            received.push((buffer.metadata().sequence, value));
        }

        producer.join().unwrap();

        assert_eq!(received.len(), 10);
        for (i, (seq, val)) in received.iter().enumerate() {
            assert_eq!(*seq, i as u64);
            assert_eq!(*val, i as u64);
        }

        // Drop arena after all buffers are consumed
        drop(arena);
    }

    #[test]
    fn test_ipc_link_zero_copy() {
        let (pub_stream, sub_stream) = UnixStream::pair().unwrap();

        let mut publisher = IpcPublisher::from_stream(pub_stream);
        let mut subscriber = IpcSubscriber::from_stream(sub_stream);

        // Create shared arena
        let arena = SharedArena::with_name("test-zero-copy", 4096, 4).unwrap();

        // Acquire slot and write data
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[0] = 42u8;
        slot.data_mut()[1] = 43u8;

        // Keep a raw pointer for later modification
        let data_ptr = slot.data_mut().as_mut_ptr();

        // Send buffer referencing this slot
        let handle = MemoryHandle::with_len(slot, 2);
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));
        publisher.send(buffer).unwrap();
        publisher.send_eos().unwrap();

        // Receive
        let received = subscriber.recv().unwrap().unwrap();

        // Verify data - this is reading from the SAME shared memory
        assert_eq!(received.as_bytes(), &[42, 43]);

        // Modify through publisher's view
        unsafe {
            *data_ptr = 100u8;
        }

        // Should see the change through subscriber's buffer!
        // (This is the zero-copy property)
        assert_eq!(received.as_bytes()[0], 100);
    }

    #[test]
    fn test_ipc_link_multiple_arenas() {
        let (pub_stream, sub_stream) = UnixStream::pair().unwrap();

        let mut publisher = IpcPublisher::from_stream(pub_stream);
        let mut subscriber = IpcSubscriber::from_stream(sub_stream);

        // Create two different arenas - keep them alive until after assertions
        let arena1 = SharedArena::with_name("test-multi-1", 1024, 4).unwrap();
        let arena2 = SharedArena::with_name("test-multi-2", 1024, 4).unwrap();

        // Acquire slots and write different data to each
        let mut slot1 = arena1.acquire().unwrap();
        let mut slot2 = arena2.acquire().unwrap();
        slot1.data_mut()[0] = 1u8;
        slot2.data_mut()[0] = 2u8;

        // Clone slots so we can send and still verify data
        let slot1_clone = slot1.clone();
        let slot2_clone = slot2.clone();

        // Send buffers from both arenas
        let handle1 = MemoryHandle::with_len(slot1, 1);
        let handle2 = MemoryHandle::with_len(slot2, 1);

        publisher
            .send(Buffer::new(handle1, Metadata::from_sequence(0)))
            .unwrap();
        publisher
            .send(Buffer::new(handle2, Metadata::from_sequence(1)))
            .unwrap();
        publisher.send_eos().unwrap();

        // Receive both - subscriber maps its own view of the arenas
        let buf1 = subscriber.recv().unwrap().unwrap();
        let buf2 = subscriber.recv().unwrap().unwrap();

        assert_eq!(buf1.as_bytes(), &[1]);
        assert_eq!(buf2.as_bytes(), &[2]);

        // Keep slots alive to ensure refcount stays > 0
        drop(slot1_clone);
        drop(slot2_clone);
    }

    #[test]
    fn test_ipc_link_arena_reuse() {
        let (pub_stream, sub_stream) = UnixStream::pair().unwrap();

        let mut publisher = IpcPublisher::from_stream(pub_stream);
        let mut subscriber = IpcSubscriber::from_stream(sub_stream);

        // Create one arena with multiple slots
        let arena = SharedArena::with_name("test-reuse", 1024, 8).unwrap();

        // Send multiple buffers from the same arena
        for i in 0..5 {
            let mut slot = arena.acquire().unwrap();
            slot.data_mut()[0] = i as u8;

            let handle = MemoryHandle::with_len(slot, 1);
            publisher
                .send(Buffer::new(handle, Metadata::from_sequence(i as u64)))
                .unwrap();
        }
        publisher.send_eos().unwrap();

        // Receive all buffers
        let mut count = 0;
        while let Some(buf) = subscriber.recv().unwrap() {
            assert_eq!(buf.as_bytes()[0], count as u8);
            count += 1;
        }
        assert_eq!(count, 5);

        // Subscriber should only have one arena (the fd was sent once)
        assert_eq!(subscriber.arenas.len(), 1);
    }
}
