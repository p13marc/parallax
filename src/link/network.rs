//! Network links using TCP with rkyv serialization.
//!
//! This module provides buffer transfer over TCP networks. Unlike IPC links,
//! network links must serialize the buffer data since shared memory isn't
//! available across machines.
//!
//! ## Protocol
//!
//! ```text
//! ┌──────────────────────────────────────┐
//! │ Magic: "PRLX" (4 bytes)              │
//! │ Version: u16                         │
//! │ Flags: u16                           │
//! │ Payload length: u32 (LE)             │
//! │ Checksum: u32 (CRC32)                │
//! ├──────────────────────────────────────┤
//! │ Payload (rkyv-serialized)            │
//! │ - Metadata                           │
//! │ - Data bytes                         │
//! └──────────────────────────────────────┘
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::error::{Error, Result};
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;

/// Protocol magic bytes.
const MAGIC: [u8; 4] = *b"PRLX";

/// Protocol version.
const VERSION: u16 = 1;

/// Message flags.
#[repr(u16)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MessageFlags {
    /// Normal data message.
    Data = 0,
    /// End of stream.
    Eos = 1,
    /// Error message.
    Error = 2,
}

impl From<u16> for MessageFlags {
    fn from(value: u16) -> Self {
        match value {
            1 => MessageFlags::Eos,
            2 => MessageFlags::Error,
            _ => MessageFlags::Data,
        }
    }
}

/// Header for network messages.
/// Using manual serialization to avoid packed struct alignment issues.
#[derive(Debug, Clone, Copy)]
struct NetworkHeader {
    magic: [u8; 4],
    version: u16,
    flags: u16,
    payload_len: u32,
    checksum: u32,
}

impl NetworkHeader {
    const SIZE: usize = 4 + 2 + 2 + 4 + 4; // 16 bytes

    fn new(flags: MessageFlags, payload_len: usize, checksum: u32) -> Self {
        Self {
            magic: MAGIC,
            version: VERSION,
            flags: flags as u16,
            payload_len: payload_len as u32,
            checksum,
        }
    }

    fn to_bytes(&self) -> [u8; Self::SIZE] {
        let mut buf = [0u8; Self::SIZE];
        buf[0..4].copy_from_slice(&self.magic);
        buf[4..6].copy_from_slice(&self.version.to_le_bytes());
        buf[6..8].copy_from_slice(&self.flags.to_le_bytes());
        buf[8..12].copy_from_slice(&self.payload_len.to_le_bytes());
        buf[12..16].copy_from_slice(&self.checksum.to_le_bytes());
        buf
    }

    fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < Self::SIZE {
            return Err(Error::Pipeline("header too short".into()));
        }
        let mut magic = [0u8; 4];
        magic.copy_from_slice(&bytes[0..4]);
        Ok(Self {
            magic,
            version: u16::from_le_bytes(bytes[4..6].try_into().unwrap()),
            flags: u16::from_le_bytes(bytes[6..8].try_into().unwrap()),
            payload_len: u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            checksum: u32::from_le_bytes(bytes[12..16].try_into().unwrap()),
        })
    }

    fn validate(&self) -> Result<()> {
        if self.magic != MAGIC {
            return Err(Error::Pipeline(format!("invalid magic: {:?}", self.magic)));
        }
        if self.version != VERSION {
            return Err(Error::Pipeline(format!(
                "unsupported version: {}",
                self.version
            )));
        }
        Ok(())
    }
}

/// Simple CRC32 implementation for checksums.
fn crc32(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for byte in data {
        crc ^= *byte as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 {
                (crc >> 1) ^ 0xEDB88320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

/// Sender side of a network link.
///
/// Serializes buffers and sends them over TCP.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::link::NetworkSender;
///
/// // Connect to receiver
/// let mut sender = NetworkSender::connect("192.168.1.100:9000")?;
///
/// // Send buffers
/// sender.send(buffer)?;
///
/// // Signal end
/// sender.send_eos()?;
/// ```
pub struct NetworkSender {
    stream: TcpStream,
    bytes_sent: u64,
}

impl NetworkSender {
    /// Connect to a network receiver.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let stream = TcpStream::connect(addr)?;
        Ok(Self {
            stream,
            bytes_sent: 0,
        })
    }

    /// Create a sender from an existing TCP stream.
    pub fn from_stream(stream: TcpStream) -> Self {
        Self {
            stream,
            bytes_sent: 0,
        }
    }

    /// Send a buffer over the network.
    ///
    /// The buffer's data and metadata are serialized and sent.
    pub fn send(&mut self, buffer: Buffer) -> Result<()> {
        // Build payload: sequence (8 bytes) + data
        let data = buffer.as_bytes();
        let metadata = buffer.metadata();

        let mut payload = Vec::with_capacity(8 + data.len());
        payload.extend_from_slice(&metadata.sequence.to_le_bytes());
        payload.extend_from_slice(data);

        let checksum = crc32(&payload);
        let header = NetworkHeader::new(MessageFlags::Data, payload.len(), checksum);

        self.stream.write_all(&header.to_bytes())?;
        self.stream.write_all(&payload)?;
        self.bytes_sent += (NetworkHeader::SIZE + payload.len()) as u64;

        Ok(())
    }

    /// Send end-of-stream signal.
    pub fn send_eos(&mut self) -> Result<()> {
        let header = NetworkHeader::new(MessageFlags::Eos, 0, 0);
        self.stream.write_all(&header.to_bytes())?;
        self.bytes_sent += NetworkHeader::SIZE as u64;
        Ok(())
    }

    /// Get total bytes sent.
    pub fn bytes_sent(&self) -> u64 {
        self.bytes_sent
    }

    /// Get the local address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.stream.local_addr()?)
    }

    /// Get the peer address.
    pub fn peer_addr(&self) -> Result<SocketAddr> {
        Ok(self.stream.peer_addr()?)
    }

    /// Flush the send buffer.
    pub fn flush(&mut self) -> Result<()> {
        Ok(self.stream.flush()?)
    }
}

/// Receiver side of a network link.
///
/// Receives and deserializes buffers from TCP.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::link::NetworkReceiver;
///
/// // Listen for connections
/// let mut receiver = NetworkReceiver::listen("0.0.0.0:9000")?;
/// receiver.accept()?;
///
/// // Receive buffers
/// while let Some(buffer) = receiver.recv()? {
///     println!("Got {} bytes", buffer.len());
/// }
/// ```
pub struct NetworkReceiver {
    listener: Option<TcpListener>,
    stream: Option<TcpStream>,
    bytes_received: u64,
}

impl NetworkReceiver {
    /// Create a receiver listening on the given address.
    pub fn listen<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let listener = TcpListener::bind(addr)?;
        Ok(Self {
            listener: Some(listener),
            stream: None,
            bytes_received: 0,
        })
    }

    /// Create a receiver from an existing TCP stream.
    pub fn from_stream(stream: TcpStream) -> Self {
        Self {
            listener: None,
            stream: Some(stream),
            bytes_received: 0,
        }
    }

    /// Accept a connection from a sender.
    pub fn accept(&mut self) -> Result<()> {
        if self.stream.is_some() {
            return Err(Error::Pipeline("already connected".into()));
        }

        let listener = self
            .listener
            .as_ref()
            .ok_or_else(|| Error::Pipeline("no listener".into()))?;

        let (stream, _addr) = listener.accept()?;
        self.stream = Some(stream);

        Ok(())
    }

    /// Get the local address (if listening).
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.listener
            .as_ref()
            .and_then(|l| l.local_addr().ok())
            .or_else(|| self.stream.as_ref().and_then(|s| s.local_addr().ok()))
    }

    /// Receive a buffer from the network.
    ///
    /// Returns `Ok(None)` on end-of-stream.
    pub fn recv(&mut self) -> Result<Option<Buffer>> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("not connected".into()))?;

        // Read header
        let mut header_buf = [0u8; NetworkHeader::SIZE];
        match stream.read_exact(&mut header_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                return Ok(None);
            }
            Err(e) => return Err(Error::Io(e)),
        }

        let header = NetworkHeader::from_bytes(&header_buf)?;
        header.validate()?;

        self.bytes_received += NetworkHeader::SIZE as u64;

        let flags = MessageFlags::from(header.flags);
        match flags {
            MessageFlags::Eos => Ok(None),
            MessageFlags::Error => Err(Error::Pipeline("sender reported error".into())),
            MessageFlags::Data => {
                // Read payload
                let payload_len = header.payload_len as usize;
                let mut payload = vec![0u8; payload_len];
                stream.read_exact(&mut payload)?;

                self.bytes_received += payload_len as u64;

                // Verify checksum
                let computed_checksum = crc32(&payload);
                if computed_checksum != header.checksum {
                    return Err(Error::Pipeline(format!(
                        "checksum mismatch: expected {:08x}, got {:08x}",
                        header.checksum, computed_checksum
                    )));
                }

                // Parse payload: sequence (8 bytes) + data
                if payload.len() < 8 {
                    return Err(Error::Pipeline("payload too short".into()));
                }

                let sequence = u64::from_le_bytes(payload[..8].try_into().unwrap());
                let data = &payload[8..];

                // Create buffer with heap-backed memory
                let segment = Arc::new(HeapSegment::new(data.len())?);
                let ptr = segment
                    .as_mut_ptr()
                    .ok_or_else(|| Error::Element("cannot get mutable pointer".into()))?;
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                }

                let handle = MemoryHandle::from_segment(segment);
                let metadata = Metadata::from_sequence(sequence);

                Ok(Some(Buffer::new(handle, metadata)))
            }
        }
    }

    /// Get total bytes received.
    pub fn bytes_received(&self) -> u64 {
        self.bytes_received
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
    use std::thread;

    fn make_buffer(data: &[u8], seq: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(data.len()).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Buffer::new(
            MemoryHandle::from_segment(segment),
            Metadata::from_sequence(seq),
        )
    }

    #[test]
    fn test_crc32() {
        assert_eq!(crc32(b"hello"), 0x3610a686);
        assert_eq!(crc32(b""), 0x00000000);
        assert_eq!(crc32(b"123456789"), 0xcbf43926);
    }

    #[test]
    fn test_network_header() {
        let header = NetworkHeader::new(MessageFlags::Data, 100, 0x12345678);
        assert_eq!(header.magic, MAGIC);
        assert_eq!(header.version, VERSION);
        assert_eq!(header.flags, 0);
        assert_eq!(header.payload_len, 100);
        assert_eq!(header.checksum, 0x12345678);

        let bytes = header.to_bytes();
        let parsed = NetworkHeader::from_bytes(&bytes).unwrap();
        assert_eq!(parsed.magic, MAGIC);
        assert_eq!(parsed.version, VERSION);
        assert_eq!(parsed.flags, 0);
        assert_eq!(parsed.payload_len, 100);
        assert_eq!(parsed.checksum, 0x12345678);
    }

    #[test]
    fn test_network_roundtrip() {
        // Create a listener
        let mut receiver = NetworkReceiver::listen("127.0.0.1:0").unwrap();
        let addr = receiver.local_addr().unwrap();

        let producer = thread::spawn(move || {
            let mut sender = NetworkSender::connect(addr).unwrap();

            for i in 0..10u64 {
                let data = format!("message {}", i);
                sender.send(make_buffer(data.as_bytes(), i)).unwrap();
            }

            sender.send_eos().unwrap();
            sender.bytes_sent()
        });

        receiver.accept().unwrap();

        let mut received = Vec::new();
        while let Some(buffer) = receiver.recv().unwrap() {
            let data = String::from_utf8_lossy(buffer.as_bytes()).to_string();
            received.push((buffer.metadata().sequence, data));
        }

        let bytes_sent = producer.join().unwrap();

        assert_eq!(received.len(), 10);
        for (i, (seq, data)) in received.iter().enumerate() {
            assert_eq!(*seq, i as u64);
            assert_eq!(data, &format!("message {}", i));
        }

        assert!(bytes_sent > 0);
        assert!(receiver.bytes_received() > 0);
    }

    #[test]
    fn test_network_large_payload() {
        let mut receiver = NetworkReceiver::listen("127.0.0.1:0").unwrap();
        let addr = receiver.local_addr().unwrap();

        let producer = thread::spawn(move || {
            let mut sender = NetworkSender::connect(addr).unwrap();

            // Send 1MB buffer
            let large_data: Vec<u8> = (0..1_000_000).map(|i| (i % 256) as u8).collect();
            sender.send(make_buffer(&large_data, 0)).unwrap();
            sender.send_eos().unwrap();

            large_data
        });

        receiver.accept().unwrap();
        let buffer = receiver.recv().unwrap().unwrap();

        let original = producer.join().unwrap();
        assert_eq!(buffer.as_bytes(), original.as_slice());
    }

    #[test]
    fn test_network_from_stream() {
        // Use socket pair for testing
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let client = thread::spawn(move || {
            let stream = TcpStream::connect(addr).unwrap();
            let mut sender = NetworkSender::from_stream(stream);
            sender.send(make_buffer(b"test", 42)).unwrap();
            sender.send_eos().unwrap();
        });

        let (stream, _) = listener.accept().unwrap();
        let mut receiver = NetworkReceiver::from_stream(stream);

        let buffer = receiver.recv().unwrap().unwrap();
        assert_eq!(buffer.as_bytes(), b"test");
        assert_eq!(buffer.metadata().sequence, 42);

        assert!(receiver.recv().unwrap().is_none());

        client.join().unwrap();
    }
}
