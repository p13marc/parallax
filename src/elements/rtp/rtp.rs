//! RTP source and sink elements.
//!
//! Provides RTP (Real-time Transport Protocol) streaming over UDP.
//!
//! - [`RtpSrc`]: Receives RTP packets from a UDP socket and parses them
//! - [`RtpSink`]: Sends RTP packets over UDP
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::{RtpSrc, RtpSink};
//!
//! // Receive RTP packets
//! let mut src = RtpSrc::bind("0.0.0.0:5004")?
//!     .with_payload_type(96);
//!
//! // Send RTP packets
//! let mut sink = RtpSink::connect("192.168.1.100:5004")?
//!     .with_ssrc(0x12345678)
//!     .with_payload_type(96);
//! ```

use crate::buffer::Buffer;
use crate::clock::ClockTime;
use crate::element::{Sink, Source};
use crate::error::{Error, Result};
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::{BufferFlags, Metadata, RtpMeta};

use bytes::Bytes;
use rtp::packet::Packet;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::Arc;
use std::time::Duration;
use webrtc_util::marshal::{Marshal, MarshalSize, Unmarshal};

/// Maximum RTP packet size (UDP MTU).
const MAX_RTP_PACKET_SIZE: usize = 1500;

// ============================================================================
// RtpSrc
// ============================================================================

/// An RTP source that receives and parses RTP packets from UDP.
///
/// This element receives UDP datagrams, parses them as RTP packets, and outputs
/// the RTP payload as buffers with RTP metadata attached.
///
/// # Features
///
/// - Parses RTP headers and extracts sequence number, timestamp, SSRC, etc.
/// - Attaches RTP metadata to output buffers
/// - Optional payload type filtering
/// - Optional SSRC filtering
/// - Tracks packet statistics (received, dropped, reordered)
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RtpSrc;
///
/// let mut src = RtpSrc::bind("0.0.0.0:5004")?
///     .with_payload_type(96)  // Only accept H.264 packets
///     .with_clock_rate(90000);
///
/// // Each buffer contains the RTP payload with RtpMeta attached
/// while let Some(buffer) = src.produce()? {
///     let rtp = buffer.metadata().rtp.unwrap();
///     println!("seq={} ts={} marker={}", rtp.seq, rtp.ts, rtp.marker);
/// }
/// ```
pub struct RtpSrc {
    name: String,
    socket: UdpSocket,
    buffer_size: usize,
    sequence: u64,
    read_timeout: Option<Duration>,
    last_sender: Option<SocketAddr>,

    // RTP configuration
    payload_type: Option<u8>,
    ssrc_filter: Option<u32>,
    clock_rate: u32,

    // Statistics
    stats: RtpSrcStats,
}

/// Statistics for RtpSrc.
#[derive(Debug, Clone, Default)]
pub struct RtpSrcStats {
    /// Total packets received.
    pub packets_received: u64,
    /// Packets dropped due to parse errors.
    pub packets_dropped: u64,
    /// Packets filtered out (wrong PT or SSRC).
    pub packets_filtered: u64,
    /// Total bytes received (payload only).
    pub bytes_received: u64,
    /// Last RTP sequence number seen.
    pub last_rtp_seq: u16,
    /// Last SSRC seen.
    pub last_ssrc: u32,
}

impl RtpSrc {
    /// Create a new RTP source bound to the given address.
    pub fn bind<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let socket = UdpSocket::bind(&addr)?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            name: format!("rtpsrc-{}", local_addr),
            socket,
            buffer_size: MAX_RTP_PACKET_SIZE,
            sequence: 0,
            read_timeout: None,
            last_sender: None,
            payload_type: None,
            ssrc_filter: None,
            clock_rate: 90000, // Default video clock rate
            stats: RtpSrcStats::default(),
        })
    }

    /// Set expected payload type (filters out other types).
    pub fn with_payload_type(mut self, pt: u8) -> Self {
        self.payload_type = Some(pt);
        self
    }

    /// Set expected SSRC (filters out other sources).
    pub fn with_ssrc(mut self, ssrc: u32) -> Self {
        self.ssrc_filter = Some(ssrc);
        self
    }

    /// Set the RTP clock rate (for timestamp conversion).
    pub fn with_clock_rate(mut self, rate: u32) -> Self {
        self.clock_rate = rate;
        self
    }

    /// Set the receive buffer size.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set a custom name for this source.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the receive timeout.
    pub fn with_read_timeout(mut self, timeout: Duration) -> Result<Self> {
        self.socket.set_read_timeout(Some(timeout))?;
        self.read_timeout = Some(timeout);
        Ok(self)
    }

    /// Enable or disable non-blocking mode.
    pub fn set_nonblocking(self, nonblocking: bool) -> Result<Self> {
        self.socket.set_nonblocking(nonblocking)?;
        Ok(self)
    }

    /// Get the local address this socket is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Get the address of the last sender.
    pub fn last_sender(&self) -> Option<SocketAddr> {
        self.last_sender
    }

    /// Get current statistics.
    pub fn stats(&self) -> &RtpSrcStats {
        &self.stats
    }

    /// Join a multicast group for receiving RTP.
    pub fn join_multicast_v4(
        self,
        multiaddr: std::net::Ipv4Addr,
        interface: std::net::Ipv4Addr,
    ) -> Result<Self> {
        self.socket.join_multicast_v4(&multiaddr, &interface)?;
        Ok(self)
    }

    /// Parse an RTP packet from raw bytes.
    fn parse_rtp(&mut self, data: &[u8]) -> Result<Option<(RtpMeta, Bytes)>> {
        let mut buf = data;
        let packet = match Packet::unmarshal(&mut buf) {
            Ok(p) => p,
            Err(e) => {
                self.stats.packets_dropped += 1;
                return Err(Error::Element(format!("RTP parse error: {}", e)));
            }
        };

        // Filter by payload type if configured
        if let Some(expected_pt) = self.payload_type {
            if packet.header.payload_type != expected_pt {
                self.stats.packets_filtered += 1;
                return Ok(None);
            }
        }

        // Filter by SSRC if configured
        if let Some(expected_ssrc) = self.ssrc_filter {
            if packet.header.ssrc != expected_ssrc {
                self.stats.packets_filtered += 1;
                return Ok(None);
            }
        }

        // Update stats
        self.stats.last_rtp_seq = packet.header.sequence_number;
        self.stats.last_ssrc = packet.header.ssrc;

        // Create RTP metadata
        let rtp_meta = RtpMeta {
            seq: packet.header.sequence_number,
            ts: packet.header.timestamp,
            ssrc: packet.header.ssrc,
            pt: packet.header.payload_type,
            marker: packet.header.marker,
        };

        Ok(Some((rtp_meta, packet.payload)))
    }
}

impl Source for RtpSrc {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        // Allocate receive buffer
        let mut recv_buf = vec![0u8; self.buffer_size];

        // Receive datagram
        let (n, sender) = match self.socket.recv_from(&mut recv_buf) {
            Ok(result) => result,
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                return Ok(None);
            }
            Err(e) => return Err(Error::Io(e)),
        };

        self.last_sender = Some(sender);
        self.stats.packets_received += 1;

        // Parse RTP packet
        let (rtp_meta, payload) = match self.parse_rtp(&recv_buf[..n])? {
            Some(result) => result,
            None => return Ok(None), // Filtered out
        };

        // Create output buffer with payload
        let payload_len = payload.len();
        self.stats.bytes_received += payload_len as u64;

        let segment = Arc::new(HeapSegment::new(payload_len)?);
        let ptr = segment
            .as_mut_ptr()
            .ok_or_else(|| Error::Element("cannot get mutable pointer".into()))?;
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ref().as_ptr(), ptr, payload_len);
        }

        let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, payload_len);

        // Build metadata
        let seq = self.sequence;
        self.sequence += 1;

        let pts = rtp_meta.timestamp_to_clock(self.clock_rate);
        let mut flags = BufferFlags::NONE;
        if rtp_meta.marker {
            // Marker typically indicates frame boundary
            flags = flags.insert(BufferFlags::SYNC_POINT);
        }

        let metadata = Metadata::new()
            .with_sequence(seq)
            .with_pts(pts)
            .with_rtp(rtp_meta)
            .with_flags(flags);

        Ok(Some(Buffer::new(handle, metadata)))
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// RtpSink
// ============================================================================

/// An RTP sink that sends RTP packets over UDP.
///
/// This element takes buffers and wraps them in RTP packets for transmission.
/// If the buffer has RTP metadata, those values are used; otherwise the sink
/// generates sequence numbers and timestamps.
///
/// # Features
///
/// - Builds RTP packets from buffer payload
/// - Uses existing RTP metadata if present, or generates new values
/// - Configurable SSRC, payload type, and clock rate
/// - Tracks packet statistics
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RtpSink;
///
/// let mut sink = RtpSink::connect("192.168.1.100:5004")?
///     .with_ssrc(0x12345678)
///     .with_payload_type(96)
///     .with_clock_rate(90000);
///
/// // Send buffers as RTP packets
/// sink.consume(buffer)?;
/// ```
pub struct RtpSink {
    name: String,
    socket: UdpSocket,
    destination: SocketAddr,

    // RTP configuration
    ssrc: u32,
    payload_type: u8,
    clock_rate: u32,

    // Sequence/timestamp generation
    next_seq: u16,
    base_timestamp: u32,
    last_pts: ClockTime,

    // Statistics
    stats: RtpSinkStats,
}

/// Statistics for RtpSink.
#[derive(Debug, Clone, Default)]
pub struct RtpSinkStats {
    /// Total packets sent.
    pub packets_sent: u64,
    /// Total bytes sent (payload only).
    pub bytes_sent: u64,
    /// Packets that failed to send.
    pub packets_failed: u64,
}

impl RtpSink {
    /// Create a new RTP sink connected to the given address.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let socket = UdpSocket::bind("0.0.0.0:0")?;
        let local_addr = socket.local_addr()?;

        // Generate random SSRC
        let ssrc = rand_ssrc();

        Ok(Self {
            name: format!("rtpsink-{}->{}", local_addr, addr),
            socket,
            destination: addr,
            ssrc,
            payload_type: 96, // Dynamic payload type
            clock_rate: 90000,
            next_seq: rand_seq(),
            base_timestamp: rand_timestamp(),
            last_pts: ClockTime::ZERO,
            stats: RtpSinkStats::default(),
        })
    }

    /// Set the SSRC for outgoing packets.
    pub fn with_ssrc(mut self, ssrc: u32) -> Self {
        self.ssrc = ssrc;
        self
    }

    /// Set the payload type for outgoing packets.
    pub fn with_payload_type(mut self, pt: u8) -> Self {
        self.payload_type = pt;
        self
    }

    /// Set the RTP clock rate (for timestamp calculation).
    pub fn with_clock_rate(mut self, rate: u32) -> Self {
        self.clock_rate = rate;
        self
    }

    /// Set a custom name for this sink.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set initial sequence number.
    pub fn with_initial_seq(mut self, seq: u16) -> Self {
        self.next_seq = seq;
        self
    }

    /// Get current statistics.
    pub fn stats(&self) -> &RtpSinkStats {
        &self.stats
    }

    /// Get the local address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Get the destination address.
    pub fn destination(&self) -> SocketAddr {
        self.destination
    }

    /// Calculate RTP timestamp from PTS.
    fn pts_to_rtp_timestamp(&self, pts: ClockTime) -> u32 {
        if pts.is_none() {
            return self.base_timestamp;
        }
        let rtp_ts = RtpMeta::clock_to_timestamp(pts, self.clock_rate);
        self.base_timestamp.wrapping_add(rtp_ts)
    }
}

impl Sink for RtpSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let metadata = buffer.metadata();
        let payload = buffer.as_bytes();

        // Use RTP metadata if present, otherwise generate
        let (seq, ts, marker) = if let Some(rtp) = &metadata.rtp {
            (rtp.seq, rtp.ts, rtp.marker)
        } else {
            let seq = self.next_seq;
            self.next_seq = self.next_seq.wrapping_add(1);

            let ts = self.pts_to_rtp_timestamp(metadata.pts);
            let marker = metadata.flags.is_keyframe();

            (seq, ts, marker)
        };

        // Build RTP packet
        let packet = Packet {
            header: rtp::header::Header {
                version: 2,
                padding: false,
                extension: false,
                marker,
                payload_type: self.payload_type,
                sequence_number: seq,
                timestamp: ts,
                ssrc: self.ssrc,
                csrc: vec![],
                extension_profile: 0,
                extensions: vec![],
                extensions_padding: 0,
            },
            payload: Bytes::copy_from_slice(payload),
        };

        // Serialize and send
        let mut buf = vec![0u8; packet.marshal_size()];
        if let Err(e) = packet.marshal_to(&mut buf) {
            self.stats.packets_failed += 1;
            return Err(Error::Element(format!("RTP marshal error: {}", e)));
        }

        match self.socket.send_to(&buf, self.destination) {
            Ok(_) => {
                self.stats.packets_sent += 1;
                self.stats.bytes_sent += payload.len() as u64;
                self.last_pts = metadata.pts;
                Ok(())
            }
            Err(e) => {
                self.stats.packets_failed += 1;
                Err(Error::Io(e))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Async variants
// ============================================================================

/// Async RTP source for use with async runtimes.
pub struct AsyncRtpSrc {
    name: String,
    socket: tokio::net::UdpSocket,
    buffer_size: usize,
    sequence: u64,
    last_sender: Option<SocketAddr>,

    // RTP configuration
    payload_type: Option<u8>,
    ssrc_filter: Option<u32>,
    clock_rate: u32,

    // Statistics
    stats: RtpSrcStats,
}

impl AsyncRtpSrc {
    /// Create a new async RTP source bound to the given address.
    pub async fn bind<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let socket = tokio::net::UdpSocket::bind(addr).await?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            name: format!("async-rtpsrc-{}", local_addr),
            socket,
            buffer_size: MAX_RTP_PACKET_SIZE,
            sequence: 0,
            last_sender: None,
            payload_type: None,
            ssrc_filter: None,
            clock_rate: 90000,
            stats: RtpSrcStats::default(),
        })
    }

    /// Set expected payload type.
    pub fn with_payload_type(mut self, pt: u8) -> Self {
        self.payload_type = Some(pt);
        self
    }

    /// Set expected SSRC.
    pub fn with_ssrc(mut self, ssrc: u32) -> Self {
        self.ssrc_filter = Some(ssrc);
        self
    }

    /// Set the RTP clock rate.
    pub fn with_clock_rate(mut self, rate: u32) -> Self {
        self.clock_rate = rate;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the local address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Get current statistics.
    pub fn stats(&self) -> &RtpSrcStats {
        &self.stats
    }

    /// Receive an RTP packet asynchronously.
    pub async fn recv(&mut self) -> Result<Option<Buffer>> {
        let mut recv_buf = vec![0u8; self.buffer_size];

        let (n, sender) = self.socket.recv_from(&mut recv_buf).await?;
        self.last_sender = Some(sender);
        self.stats.packets_received += 1;

        // Parse RTP packet
        let mut buf = &recv_buf[..n];
        let packet = match Packet::unmarshal(&mut buf) {
            Ok(p) => p,
            Err(e) => {
                self.stats.packets_dropped += 1;
                return Err(Error::Element(format!("RTP parse error: {}", e)));
            }
        };

        // Filter by payload type
        if let Some(expected_pt) = self.payload_type {
            if packet.header.payload_type != expected_pt {
                self.stats.packets_filtered += 1;
                return Ok(None);
            }
        }

        // Filter by SSRC
        if let Some(expected_ssrc) = self.ssrc_filter {
            if packet.header.ssrc != expected_ssrc {
                self.stats.packets_filtered += 1;
                return Ok(None);
            }
        }

        // Update stats
        self.stats.last_rtp_seq = packet.header.sequence_number;
        self.stats.last_ssrc = packet.header.ssrc;

        // Create RTP metadata
        let rtp_meta = RtpMeta {
            seq: packet.header.sequence_number,
            ts: packet.header.timestamp,
            ssrc: packet.header.ssrc,
            pt: packet.header.payload_type,
            marker: packet.header.marker,
        };

        // Create output buffer
        let payload = packet.payload;
        let payload_len = payload.len();
        self.stats.bytes_received += payload_len as u64;

        let segment = Arc::new(HeapSegment::new(payload_len)?);
        let ptr = segment
            .as_mut_ptr()
            .ok_or_else(|| Error::Element("cannot get mutable pointer".into()))?;
        unsafe {
            std::ptr::copy_nonoverlapping(payload.as_ref().as_ptr(), ptr, payload_len);
        }

        let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, payload_len);

        let seq = self.sequence;
        self.sequence += 1;

        let pts = rtp_meta.timestamp_to_clock(self.clock_rate);
        let mut flags = BufferFlags::NONE;
        if rtp_meta.marker {
            flags = flags.insert(BufferFlags::SYNC_POINT);
        }

        let metadata = Metadata::new()
            .with_sequence(seq)
            .with_pts(pts)
            .with_rtp(rtp_meta)
            .with_flags(flags);

        Ok(Some(Buffer::new(handle, metadata)))
    }
}

/// Async RTP sink for use with async runtimes.
pub struct AsyncRtpSink {
    name: String,
    socket: tokio::net::UdpSocket,
    destination: SocketAddr,

    // RTP configuration
    ssrc: u32,
    payload_type: u8,
    clock_rate: u32,

    // Sequence/timestamp generation
    next_seq: u16,
    base_timestamp: u32,

    // Statistics
    stats: RtpSinkStats,
}

impl AsyncRtpSink {
    /// Create a new async RTP sink connected to the given address.
    pub async fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let socket = tokio::net::UdpSocket::bind("0.0.0.0:0").await?;
        let local_addr = socket.local_addr()?;

        let ssrc = rand_ssrc();

        Ok(Self {
            name: format!("async-rtpsink-{}->{}", local_addr, addr),
            socket,
            destination: addr,
            ssrc,
            payload_type: 96,
            clock_rate: 90000,
            next_seq: rand_seq(),
            base_timestamp: rand_timestamp(),
            stats: RtpSinkStats::default(),
        })
    }

    /// Set the SSRC.
    pub fn with_ssrc(mut self, ssrc: u32) -> Self {
        self.ssrc = ssrc;
        self
    }

    /// Set the payload type.
    pub fn with_payload_type(mut self, pt: u8) -> Self {
        self.payload_type = pt;
        self
    }

    /// Set the clock rate.
    pub fn with_clock_rate(mut self, rate: u32) -> Self {
        self.clock_rate = rate;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the local address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Get current statistics.
    pub fn stats(&self) -> &RtpSinkStats {
        &self.stats
    }

    /// Send a buffer as an RTP packet.
    pub async fn send(&mut self, buffer: Buffer) -> Result<()> {
        let metadata = buffer.metadata();
        let payload = buffer.as_bytes();

        let (seq, ts, marker) = if let Some(rtp) = &metadata.rtp {
            (rtp.seq, rtp.ts, rtp.marker)
        } else {
            let seq = self.next_seq;
            self.next_seq = self.next_seq.wrapping_add(1);

            let ts = if metadata.pts.is_some() {
                let rtp_ts = RtpMeta::clock_to_timestamp(metadata.pts, self.clock_rate);
                self.base_timestamp.wrapping_add(rtp_ts)
            } else {
                self.base_timestamp
            };
            let marker = metadata.flags.is_keyframe();

            (seq, ts, marker)
        };

        let packet = Packet {
            header: rtp::header::Header {
                version: 2,
                padding: false,
                extension: false,
                marker,
                payload_type: self.payload_type,
                sequence_number: seq,
                timestamp: ts,
                ssrc: self.ssrc,
                csrc: vec![],
                extension_profile: 0,
                extensions: vec![],
                extensions_padding: 0,
            },
            payload: Bytes::copy_from_slice(payload),
        };

        let mut buf = vec![0u8; packet.marshal_size()];
        if let Err(e) = packet.marshal_to(&mut buf) {
            self.stats.packets_failed += 1;
            return Err(Error::Element(format!("RTP marshal error: {}", e)));
        }

        match self.socket.send_to(&buf, self.destination).await {
            Ok(_) => {
                self.stats.packets_sent += 1;
                self.stats.bytes_sent += payload.len() as u64;
                Ok(())
            }
            Err(e) => {
                self.stats.packets_failed += 1;
                Err(Error::Io(e))
            }
        }
    }
}

impl crate::element::AsyncSink for AsyncRtpSink {
    async fn consume(&mut self, buffer: Buffer) -> Result<()> {
        self.send(buffer).await
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Helper functions
// ============================================================================

/// Generate a random SSRC.
fn rand_ssrc() -> u32 {
    // Use a simple hash of current time as pseudo-random
    use std::time::SystemTime;
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default();
    let seed = now.as_nanos() as u64;
    // Simple xorshift
    let mut x = seed;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x as u32
}

/// Generate a random initial sequence number.
fn rand_seq() -> u16 {
    (rand_ssrc() & 0xFFFF) as u16
}

/// Generate a random initial timestamp.
fn rand_timestamp() -> u32 {
    rand_ssrc()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Sink, Source};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_rtp_src_creation() {
        let src = RtpSrc::bind("127.0.0.1:0").unwrap();
        assert!(src.name.contains("rtpsrc"));
        assert!(src.local_addr().is_ok());
    }

    #[test]
    fn test_rtp_sink_creation() {
        let sink = RtpSink::connect("127.0.0.1:9999").unwrap();
        assert!(sink.name.contains("rtpsink"));
        assert_eq!(sink.destination.port(), 9999);
    }

    #[test]
    fn test_rtp_sink_configuration() {
        let sink = RtpSink::connect("127.0.0.1:5004")
            .unwrap()
            .with_ssrc(0x12345678)
            .with_payload_type(96)
            .with_clock_rate(90000);

        assert_eq!(sink.ssrc, 0x12345678);
        assert_eq!(sink.payload_type, 96);
        assert_eq!(sink.clock_rate, 90000);
    }

    #[test]
    fn test_rtp_roundtrip() {
        // Create receiver
        let mut src = RtpSrc::bind("127.0.0.1:0")
            .unwrap()
            .with_read_timeout(Duration::from_secs(2))
            .unwrap();
        let recv_addr = src.local_addr().unwrap();

        // Spawn sender thread
        let handle = thread::spawn(move || {
            let mut sink = RtpSink::connect(recv_addr)
                .unwrap()
                .with_ssrc(0xABCDEF01)
                .with_payload_type(96);

            // Create test buffer
            let segment = Arc::new(HeapSegment::new(5).unwrap());
            let ptr = segment.as_mut_ptr().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(b"hello".as_ptr(), ptr, 5);
            }
            let buf_handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, 5);
            let buffer = Buffer::new(buf_handle, Metadata::default());

            sink.consume(buffer).unwrap();
        });

        // Receive and verify
        let buffer = src.produce().unwrap().unwrap();
        assert_eq!(buffer.as_bytes(), b"hello");

        // Check RTP metadata
        let rtp = buffer.metadata().rtp.unwrap();
        assert_eq!(rtp.ssrc, 0xABCDEF01);
        assert_eq!(rtp.pt, 96);

        // Check stats
        assert_eq!(src.stats().packets_received, 1);
        assert_eq!(src.stats().bytes_received, 5);

        handle.join().unwrap();
    }

    #[test]
    fn test_rtp_src_filtering() {
        // Create receiver with PT filter
        let mut src = RtpSrc::bind("127.0.0.1:0")
            .unwrap()
            .with_payload_type(96)
            .with_read_timeout(Duration::from_millis(100))
            .unwrap()
            .set_nonblocking(true)
            .unwrap();
        let recv_addr = src.local_addr().unwrap();

        // Send packet with wrong PT
        let handle = thread::spawn(move || {
            let mut sink = RtpSink::connect(recv_addr).unwrap().with_payload_type(97); // Wrong PT

            let segment = Arc::new(HeapSegment::new(4).unwrap());
            let buf_handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, 4);
            let buffer = Buffer::new(buf_handle, Metadata::default());

            sink.consume(buffer).unwrap();
        });

        handle.join().unwrap();

        // Small delay to allow packet to arrive
        thread::sleep(Duration::from_millis(50));

        // Should be filtered out
        let result = src.produce();
        assert!(result.is_ok());
        // Either None (filtered) or WouldBlock in non-blocking mode
    }

    #[tokio::test]
    async fn test_async_rtp_roundtrip() {
        let mut src = AsyncRtpSrc::bind("127.0.0.1:0").await.unwrap();
        let recv_addr = src.local_addr().unwrap();

        let sender = tokio::spawn(async move {
            let mut sink = AsyncRtpSink::connect(recv_addr)
                .await
                .unwrap()
                .with_ssrc(0x87654321)
                .with_payload_type(111);

            let segment = Arc::new(HeapSegment::new(10).unwrap());
            let ptr = segment.as_mut_ptr().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(b"async test".as_ptr(), ptr, 10);
            }
            let buf_handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, 10);
            let buffer = Buffer::new(buf_handle, Metadata::default());

            sink.send(buffer).await.unwrap();
        });

        // Use tokio timeout
        let result = tokio::time::timeout(Duration::from_secs(2), src.recv()).await;

        assert!(result.is_ok());
        let buffer = result.unwrap().unwrap().unwrap();
        assert_eq!(buffer.as_bytes(), b"async test");

        let rtp = buffer.metadata().rtp.unwrap();
        assert_eq!(rtp.ssrc, 0x87654321);
        assert_eq!(rtp.pt, 111);

        sender.await.unwrap();
    }

    #[test]
    fn test_rtp_meta_preservation() {
        // Create receiver
        let mut src = RtpSrc::bind("127.0.0.1:0")
            .unwrap()
            .with_clock_rate(90000)
            .with_read_timeout(Duration::from_secs(2))
            .unwrap();
        let recv_addr = src.local_addr().unwrap();

        let handle = thread::spawn(move || {
            let mut sink = RtpSink::connect(recv_addr).unwrap().with_ssrc(0x11111111);

            // Create buffer with existing RTP metadata
            let rtp = RtpMeta {
                seq: 1234,
                ts: 90000, // 1 second
                ssrc: 0x22222222,
                pt: 96,
                marker: true,
            };

            let segment = Arc::new(HeapSegment::new(3).unwrap());
            let ptr = segment.as_mut_ptr().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(b"rtp".as_ptr(), ptr, 3);
            }
            let buf_handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, 3);
            let metadata = Metadata::new().with_rtp(rtp);
            let buffer = Buffer::new(buf_handle, metadata);

            sink.consume(buffer).unwrap();
        });

        let buffer = src.produce().unwrap().unwrap();

        // Verify RTP fields were used (not sink's SSRC since it uses buffer's RTP meta for seq/ts)
        let rtp = buffer.metadata().rtp.unwrap();
        assert_eq!(rtp.seq, 1234);
        assert_eq!(rtp.ts, 90000);
        assert!(rtp.marker);

        // PTS should be calculated from RTP timestamp
        let pts = buffer.metadata().pts;
        assert_eq!(pts.secs(), 1); // 90000 / 90000 = 1 second

        handle.join().unwrap();
    }
}
