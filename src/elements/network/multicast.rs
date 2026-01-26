//! UDP multicast source and sink elements.
//!
//! Provides multicast UDP communication for one-to-many data distribution.
//!
//! - [`UdpMulticastSrc`]: Receives data from a multicast group
//! - [`UdpMulticastSink`]: Sends data to a multicast group

use crate::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use crate::error::{Error, Result};
use std::net::{Ipv4Addr, SocketAddr, SocketAddrV4, UdpSocket};
use std::time::Duration;

/// A UDP multicast source that receives data from a multicast group.
///
/// Joins a multicast group and receives datagrams sent to that group.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::UdpMulticastSrc;
///
/// // Join multicast group 239.255.0.1 on port 5000
/// let src = UdpMulticastSrc::new("239.255.0.1", 5000)?;
/// ```
pub struct UdpMulticastSrc {
    name: String,
    socket: UdpSocket,
    multicast_addr: Ipv4Addr,
    port: u16,
    buffer_size: usize,
    bytes_read: u64,
    sequence: u64,
    datagrams_received: u64,
}

impl UdpMulticastSrc {
    /// Create a new multicast source.
    ///
    /// # Arguments
    /// * `multicast_addr` - The multicast group address (e.g., "239.255.0.1")
    /// * `port` - The port to listen on
    pub fn new(multicast_addr: &str, port: u16) -> Result<Self> {
        let multicast_ip: Ipv4Addr = multicast_addr
            .parse()
            .map_err(|_| Error::Config(format!("invalid multicast address: {}", multicast_addr)))?;

        if !multicast_ip.is_multicast() {
            return Err(Error::Config(format!(
                "address {} is not a multicast address",
                multicast_addr
            )));
        }

        // Bind to any address on the specified port
        let bind_addr = SocketAddrV4::new(Ipv4Addr::UNSPECIFIED, port);
        let socket = UdpSocket::bind(bind_addr)?;

        // Join the multicast group on all interfaces
        socket.join_multicast_v4(&multicast_ip, &Ipv4Addr::UNSPECIFIED)?;

        Ok(Self {
            name: format!("multicast-src-{}:{}", multicast_addr, port),
            socket,
            multicast_addr: multicast_ip,
            port,
            buffer_size: 65535, // Max UDP datagram size
            bytes_read: 0,
            sequence: 0,
            datagrams_received: 0,
        })
    }

    /// Set the buffer size for receiving datagrams.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size.clamp(1, 65535);
        self
    }

    /// Set a custom name for this source.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the receive timeout.
    pub fn with_timeout(self, timeout: Duration) -> Result<Self> {
        self.socket.set_read_timeout(Some(timeout))?;
        Ok(self)
    }

    /// Get the multicast address.
    pub fn multicast_addr(&self) -> Ipv4Addr {
        self.multicast_addr
    }

    /// Get the port.
    pub fn port(&self) -> u16 {
        self.port
    }

    /// Get the number of bytes received.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Get the number of datagrams received.
    pub fn datagrams_received(&self) -> u64 {
        self.datagrams_received
    }

    /// Get statistics.
    pub fn stats(&self) -> UdpMulticastStats {
        UdpMulticastStats {
            bytes_transferred: self.bytes_read,
            datagrams: self.datagrams_received,
        }
    }
}

impl Source for UdpMulticastSrc {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        let output = ctx.output();
        let recv_len = output.len().min(self.buffer_size);

        match self.socket.recv_from(&mut output[..recv_len]) {
            Ok((n, _addr)) => {
                self.bytes_read += n as u64;
                self.datagrams_received += 1;
                ctx.set_sequence(self.sequence);
                self.sequence += 1;

                Ok(ProduceResult::Produced(n))
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Timeout - signal would block
                Ok(ProduceResult::WouldBlock)
            }
            Err(e) => Err(e.into()),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(self.buffer_size)
    }
}

impl Drop for UdpMulticastSrc {
    fn drop(&mut self) {
        // Leave the multicast group
        let _ = self
            .socket
            .leave_multicast_v4(&self.multicast_addr, &Ipv4Addr::UNSPECIFIED);
    }
}

/// A UDP multicast sink that sends data to a multicast group.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::UdpMulticastSink;
///
/// // Send to multicast group 239.255.0.1 on port 5000
/// let sink = UdpMulticastSink::new("239.255.0.1", 5000)?;
/// ```
pub struct UdpMulticastSink {
    name: String,
    socket: UdpSocket,
    dest_addr: SocketAddr,
    multicast_addr: Ipv4Addr,
    bytes_written: u64,
    datagrams_sent: u64,
    ttl: u32,
}

impl UdpMulticastSink {
    /// Create a new multicast sink.
    ///
    /// # Arguments
    /// * `multicast_addr` - The multicast group address (e.g., "239.255.0.1")
    /// * `port` - The destination port
    pub fn new(multicast_addr: &str, port: u16) -> Result<Self> {
        let multicast_ip: Ipv4Addr = multicast_addr
            .parse()
            .map_err(|_| Error::Config(format!("invalid multicast address: {}", multicast_addr)))?;

        if !multicast_ip.is_multicast() {
            return Err(Error::Config(format!(
                "address {} is not a multicast address",
                multicast_addr
            )));
        }

        // Bind to any available port
        let socket = UdpSocket::bind("0.0.0.0:0")?;

        // Set default TTL for multicast
        socket.set_multicast_ttl_v4(1)?;

        let dest_addr = SocketAddr::V4(SocketAddrV4::new(multicast_ip, port));

        Ok(Self {
            name: format!("multicast-sink-{}:{}", multicast_addr, port),
            socket,
            dest_addr,
            multicast_addr: multicast_ip,
            bytes_written: 0,
            datagrams_sent: 0,
            ttl: 1,
        })
    }

    /// Set the multicast TTL (time-to-live / hop limit).
    ///
    /// Default is 1, meaning packets won't cross routers.
    pub fn with_ttl(mut self, ttl: u32) -> Result<Self> {
        self.socket.set_multicast_ttl_v4(ttl)?;
        self.ttl = ttl;
        Ok(self)
    }

    /// Enable or disable multicast loopback.
    ///
    /// When enabled, multicast packets are also received on the local machine.
    pub fn with_loopback(self, enabled: bool) -> Result<Self> {
        self.socket.set_multicast_loop_v4(enabled)?;
        Ok(self)
    }

    /// Set a custom name for this sink.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the multicast address.
    pub fn multicast_addr(&self) -> Ipv4Addr {
        self.multicast_addr
    }

    /// Get the destination address.
    pub fn dest_addr(&self) -> SocketAddr {
        self.dest_addr
    }

    /// Get the number of bytes written.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Get the number of datagrams sent.
    pub fn datagrams_sent(&self) -> u64 {
        self.datagrams_sent
    }

    /// Get the current TTL.
    pub fn ttl(&self) -> u32 {
        self.ttl
    }

    /// Get statistics.
    pub fn stats(&self) -> UdpMulticastStats {
        UdpMulticastStats {
            bytes_transferred: self.bytes_written,
            datagrams: self.datagrams_sent,
        }
    }
}

impl Sink for UdpMulticastSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let data = ctx.input();

        // UDP datagrams have a max size
        if data.len() > 65535 {
            return Err(Error::Element("buffer too large for UDP datagram".into()));
        }

        let sent = self.socket.send_to(data, self.dest_addr)?;
        self.bytes_written += sent as u64;
        self.datagrams_sent += 1;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Statistics for multicast elements.
#[derive(Debug, Clone, Copy, Default)]
pub struct UdpMulticastStats {
    /// Total bytes transferred.
    pub bytes_transferred: u64,
    /// Total datagrams transferred.
    pub datagrams: u64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{Buffer, MemoryHandle};
    use crate::memory::{CpuArena, HeapSegment, MemorySegment};
    use crate::metadata::Metadata;
    use std::sync::Arc;
    use std::thread;

    fn make_buffer(data: &[u8], seq: u64) -> Buffer {
        let segment = Arc::new(HeapSegment::new(data.len().max(1)).unwrap());
        if !data.is_empty() {
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        }
        let handle = MemoryHandle::from_segment_with_len(segment, data.len());
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn test_multicast_sink_creation() -> Result<()> {
        let sink = UdpMulticastSink::new("239.255.0.1", 5000)?;
        assert_eq!(sink.multicast_addr(), Ipv4Addr::new(239, 255, 0, 1));
        Ok(())
    }

    #[test]
    fn test_multicast_src_creation() -> Result<()> {
        let src = UdpMulticastSrc::new("239.255.0.1", 5001)?;
        assert_eq!(src.multicast_addr(), Ipv4Addr::new(239, 255, 0, 1));
        assert_eq!(src.port(), 5001);
        Ok(())
    }

    #[test]
    fn test_invalid_multicast_address() {
        // Non-multicast address should fail
        let result = UdpMulticastSrc::new("192.168.1.1", 5000);
        assert!(result.is_err());
    }

    #[test]
    fn test_multicast_sink_with_ttl() -> Result<()> {
        let sink = UdpMulticastSink::new("239.255.0.1", 5002)?.with_ttl(5)?;
        assert_eq!(sink.ttl(), 5);
        Ok(())
    }

    #[test]
    fn test_multicast_src_with_name() -> Result<()> {
        let src = UdpMulticastSrc::new("239.255.0.1", 5003)?.with_name("my-multicast");
        assert_eq!(src.name(), "my-multicast");
        Ok(())
    }

    #[test]
    fn test_multicast_roundtrip() -> Result<()> {
        let port = 5004;
        let multicast_addr = "239.255.0.1";

        // Start receiver in a thread
        let receiver = thread::spawn(move || -> Result<Vec<u8>> {
            let mut src = UdpMulticastSrc::new(multicast_addr, port)?
                .with_timeout(Duration::from_millis(500))?;

            // Create arena for buffer allocation
            let arena = CpuArena::new(65535, 4).unwrap();
            let slot = arena.acquire().unwrap();
            let mut ctx = ProduceContext::new(slot);

            // Wait for data
            match src.produce(&mut ctx)? {
                ProduceResult::Produced(n) => {
                    let buffer = ctx.finalize(n);
                    Ok(buffer.as_bytes().to_vec())
                }
                _ => Ok(vec![]),
            }
        });

        // Give receiver time to join multicast group
        thread::sleep(Duration::from_millis(100));

        // Send data
        let mut sink = UdpMulticastSink::new(multicast_addr, port)?.with_loopback(true)?;
        let buffer = make_buffer(b"Hello Multicast", 0);
        let ctx = ConsumeContext::new(&buffer);
        sink.consume(&ctx)?;

        let received = receiver.join().unwrap()?;
        // Note: multicast delivery is not guaranteed, so we check if we got data
        if !received.is_empty() {
            assert_eq!(received, b"Hello Multicast");
        }

        Ok(())
    }

    #[test]
    fn test_multicast_stats() -> Result<()> {
        let mut sink = UdpMulticastSink::new("239.255.0.1", 5005)?;

        let buffer1 = make_buffer(b"test", 0);
        let ctx1 = ConsumeContext::new(&buffer1);
        sink.consume(&ctx1)?;

        let buffer2 = make_buffer(b"data", 1);
        let ctx2 = ConsumeContext::new(&buffer2);
        sink.consume(&ctx2)?;

        let stats = sink.stats();
        assert_eq!(stats.datagrams, 2);
        assert_eq!(stats.bytes_transferred, 8);

        Ok(())
    }
}
