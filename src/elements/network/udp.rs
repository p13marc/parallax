//! UDP source and sink elements.
//!
//! Provides network-based streaming via UDP sockets.
//!
//! - [`UdpSrc`]: Reads datagrams from a UDP socket
//! - [`UdpSink`]: Writes datagrams to a UDP socket

use crate::buffer::Buffer;
use crate::element::{ConsumeContext, ProduceContext, ProduceResult};
use crate::error::{Error, Result};
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::net::{SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::Arc;
use std::time::Duration;

/// A UDP source that reads datagrams from a socket.
///
/// UDP is connectionless, so this source binds to a local address and receives
/// datagrams from any sender. Each datagram becomes a separate buffer.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::UdpSrc;
///
/// // Bind to a local address and receive datagrams
/// let mut src = UdpSrc::bind("0.0.0.0:8080")?;
///
/// // Optionally filter by sender
/// let mut src = UdpSrc::bind("0.0.0.0:8080")?
///     .connect("192.168.1.100:9000")?;
/// ```
pub struct UdpSrc {
    name: String,
    socket: UdpSocket,
    buffer_size: usize,
    bytes_read: u64,
    sequence: u64,
    read_timeout: Option<Duration>,
    last_sender: Option<SocketAddr>,
}

impl UdpSrc {
    /// Create a new UDP source bound to the given address.
    pub fn bind<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let socket = UdpSocket::bind(&addr)?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            name: format!("udpsrc-{}", local_addr),
            socket,
            buffer_size: 65535, // Max UDP datagram size
            bytes_read: 0,
            sequence: 0,
            read_timeout: None,
            last_sender: None,
        })
    }

    /// Connect to a specific remote address.
    ///
    /// After connecting, only datagrams from this address are received,
    /// and `send` can be used instead of `send_to`.
    pub fn connect<A: ToSocketAddrs>(self, addr: A) -> Result<Self> {
        self.socket.connect(addr)?;
        Ok(self)
    }

    /// Set the buffer size for receiving datagrams.
    ///
    /// Default is 65535 (maximum UDP datagram size).
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

    /// Get the number of bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Get the local address this socket is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Get the address of the last sender.
    pub fn last_sender(&self) -> Option<SocketAddr> {
        self.last_sender
    }

    /// Join a multicast group.
    ///
    /// The `multiaddr` is the multicast group address, and `interface` is the
    /// local interface address (use `0.0.0.0` for any interface).
    pub fn join_multicast_v4(
        self,
        multiaddr: std::net::Ipv4Addr,
        interface: std::net::Ipv4Addr,
    ) -> Result<Self> {
        self.socket.join_multicast_v4(&multiaddr, &interface)?;
        Ok(self)
    }

    /// Leave a multicast group.
    pub fn leave_multicast_v4(
        self,
        multiaddr: std::net::Ipv4Addr,
        interface: std::net::Ipv4Addr,
    ) -> Result<Self> {
        self.socket.leave_multicast_v4(&multiaddr, &interface)?;
        Ok(self)
    }
}

impl crate::element::Source for UdpSrc {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if ctx.has_buffer() {
            // Use the provided buffer
            let output = ctx.output();
            let max_len = output.len().min(self.buffer_size);

            // Receive datagram into the provided buffer
            match self.socket.recv_from(&mut output[..max_len]) {
                Ok((n, sender)) => {
                    self.bytes_read += n as u64;
                    self.last_sender = Some(sender);

                    ctx.set_sequence(self.sequence);
                    self.sequence += 1;

                    Ok(ProduceResult::Produced(n))
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Non-blocking mode, no data available
                    Ok(ProduceResult::WouldBlock)
                }
                Err(e) => Err(Error::Io(e)),
            }
        } else {
            // No buffer provided, allocate our own
            let segment = Arc::new(HeapSegment::new(self.buffer_size)?);
            let ptr = segment
                .as_mut_ptr()
                .ok_or_else(|| Error::Element("cannot get mutable pointer".into()))?;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.buffer_size) };

            // Receive datagram
            match self.socket.recv_from(slice) {
                Ok((n, sender)) => {
                    self.bytes_read += n as u64;
                    self.last_sender = Some(sender);
                    let seq = self.sequence;
                    self.sequence += 1;

                    let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, n);
                    let metadata = Metadata::from_sequence(seq);
                    Ok(ProduceResult::OwnBuffer(Buffer::new(handle, metadata)))
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // Non-blocking mode, no data available
                    Ok(ProduceResult::WouldBlock)
                }
                Err(e) => Err(Error::Io(e)),
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(self.buffer_size)
    }
}

/// A UDP sink that sends datagrams to a socket.
///
/// Can operate in two modes:
/// - **Connected mode**: Send to a single destination (more efficient)
/// - **Unconnected mode**: Send to different destinations per buffer
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::UdpSink;
///
/// // Connected mode - send all data to one destination
/// let mut sink = UdpSink::connect("127.0.0.1:8080")?;
///
/// // Unconnected mode - specify destination per send
/// let mut sink = UdpSink::bind("0.0.0.0:0")?;
/// sink.set_destination("192.168.1.100:9000")?;
/// ```
pub struct UdpSink {
    name: String,
    socket: UdpSocket,
    destination: Option<SocketAddr>,
    connected: bool,
    bytes_written: u64,
    write_timeout: Option<Duration>,
}

impl UdpSink {
    /// Create a new UDP sink bound to an ephemeral port.
    ///
    /// Use `set_destination` to specify where to send data.
    pub fn bind<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let socket = UdpSocket::bind(addr)?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            name: format!("udpsink-{}", local_addr),
            socket,
            destination: None,
            connected: false,
            bytes_written: 0,
            write_timeout: None,
        })
    }

    /// Create a new UDP sink connected to a specific destination.
    ///
    /// This is more efficient than using `set_destination` for each send.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let socket = UdpSocket::bind("0.0.0.0:0")?;
        socket.connect(addr)?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            name: format!("udpsink-{}->{}", local_addr, addr),
            socket,
            destination: Some(addr),
            connected: true,
            bytes_written: 0,
            write_timeout: None,
        })
    }

    /// Set the destination address for unconnected mode.
    pub fn set_destination<A: ToSocketAddrs>(&mut self, addr: A) -> Result<()> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;
        self.destination = Some(addr);
        Ok(())
    }

    /// Set a custom name for this sink.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the write timeout.
    pub fn with_write_timeout(mut self, timeout: Duration) -> Result<Self> {
        self.socket.set_write_timeout(Some(timeout))?;
        self.write_timeout = Some(timeout);
        Ok(self)
    }

    /// Enable or disable non-blocking mode.
    pub fn set_nonblocking(self, nonblocking: bool) -> Result<Self> {
        self.socket.set_nonblocking(nonblocking)?;
        Ok(self)
    }

    /// Get the number of bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Get the local address this socket is bound to.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Get the destination address (if set).
    pub fn destination(&self) -> Option<SocketAddr> {
        self.destination
    }

    /// Enable broadcast on this socket.
    pub fn set_broadcast(self, broadcast: bool) -> Result<Self> {
        self.socket.set_broadcast(broadcast)?;
        Ok(self)
    }

    /// Set the multicast TTL.
    pub fn set_multicast_ttl_v4(self, ttl: u32) -> Result<Self> {
        self.socket.set_multicast_ttl_v4(ttl)?;
        Ok(self)
    }
}

impl crate::element::Sink for UdpSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let data = ctx.input();

        if self.connected {
            // Connected mode - use send()
            self.socket.send(data)?;
        } else if let Some(dest) = self.destination {
            // Unconnected mode with destination set
            self.socket.send_to(data, dest)?;
        } else {
            return Err(Error::Config("UDP sink has no destination set".into()));
        }

        self.bytes_written += data.len() as u64;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Async UDP source for use with async runtimes.
pub struct AsyncUdpSrc {
    name: String,
    socket: tokio::net::UdpSocket,
    buffer_size: usize,
    bytes_read: u64,
    sequence: u64,
    last_sender: Option<SocketAddr>,
}

impl AsyncUdpSrc {
    /// Create a new async UDP source bound to the given address.
    pub async fn bind<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let socket = tokio::net::UdpSocket::bind(addr).await?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            name: format!("async-udpsrc-{}", local_addr),
            socket,
            buffer_size: 65535,
            bytes_read: 0,
            sequence: 0,
            last_sender: None,
        })
    }

    /// Connect to a specific remote address.
    pub async fn connect<A: ToSocketAddrs>(self, addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        self.socket.connect(addr).await?;
        Ok(self)
    }

    /// Set the buffer size for receiving datagrams.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set a custom name for this source.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the number of bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Get the local address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Get the last sender address.
    pub fn last_sender(&self) -> Option<SocketAddr> {
        self.last_sender
    }

    /// Get the preferred buffer size for this source.
    pub fn preferred_buffer_size(&self) -> Option<usize> {
        Some(self.buffer_size)
    }

    /// Receive a datagram asynchronously (convenience method).
    pub async fn recv(&mut self) -> Result<Option<Buffer>> {
        // Allocate buffer
        let segment = Arc::new(HeapSegment::new(self.buffer_size)?);
        let ptr = segment
            .as_mut_ptr()
            .ok_or_else(|| Error::Element("cannot get mutable pointer".into()))?;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.buffer_size) };

        // Receive datagram
        let (n, sender) = self.socket.recv_from(slice).await?;
        self.bytes_read += n as u64;
        self.last_sender = Some(sender);
        let seq = self.sequence;
        self.sequence += 1;

        let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, n);
        let metadata = Metadata::from_sequence(seq);
        Ok(Some(Buffer::new(handle, metadata)))
    }
}

impl crate::element::AsyncSource for AsyncUdpSrc {
    async fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        if ctx.has_buffer() {
            // Use the provided buffer
            let output = ctx.output();
            let max_len = output.len().min(self.buffer_size);

            // Receive datagram into the provided buffer
            let (n, sender) = self.socket.recv_from(&mut output[..max_len]).await?;
            self.bytes_read += n as u64;
            self.last_sender = Some(sender);

            ctx.set_sequence(self.sequence);
            self.sequence += 1;

            Ok(ProduceResult::Produced(n))
        } else {
            // No buffer provided, allocate our own
            let segment = Arc::new(HeapSegment::new(self.buffer_size)?);
            let ptr = segment
                .as_mut_ptr()
                .ok_or_else(|| Error::Element("cannot get mutable pointer".into()))?;
            let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.buffer_size) };

            // Receive datagram
            let (n, sender) = self.socket.recv_from(slice).await?;
            self.bytes_read += n as u64;
            self.last_sender = Some(sender);
            let seq = self.sequence;
            self.sequence += 1;

            let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, n);
            let metadata = Metadata::from_sequence(seq);
            Ok(ProduceResult::OwnBuffer(Buffer::new(handle, metadata)))
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(self.buffer_size)
    }
}

/// Async UDP sink for use with async runtimes.
pub struct AsyncUdpSink {
    name: String,
    socket: tokio::net::UdpSocket,
    destination: Option<SocketAddr>,
    connected: bool,
    bytes_written: u64,
}

impl AsyncUdpSink {
    /// Create a new async UDP sink bound to an ephemeral port.
    pub async fn bind<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let socket = tokio::net::UdpSocket::bind(addr).await?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            name: format!("async-udpsink-{}", local_addr),
            socket,
            destination: None,
            connected: false,
            bytes_written: 0,
        })
    }

    /// Create a new async UDP sink connected to a specific destination.
    pub async fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let socket = tokio::net::UdpSocket::bind("0.0.0.0:0").await?;
        socket.connect(&addr).await?;
        let local_addr = socket.local_addr()?;

        Ok(Self {
            name: format!("async-udpsink-{}->{}", local_addr, addr),
            socket,
            destination: Some(addr),
            connected: true,
            bytes_written: 0,
        })
    }

    /// Set the destination address for unconnected mode.
    pub fn set_destination<A: ToSocketAddrs>(&mut self, addr: A) -> Result<()> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;
        self.destination = Some(addr);
        Ok(())
    }

    /// Set a custom name for this sink.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the number of bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Get the local address.
    pub fn local_addr(&self) -> Result<SocketAddr> {
        Ok(self.socket.local_addr()?)
    }

    /// Send a buffer asynchronously.
    pub async fn send(&mut self, buffer: Buffer) -> Result<()> {
        let ctx = ConsumeContext::new(&buffer);
        <Self as crate::element::AsyncSink>::consume(self, &ctx).await
    }
}

impl crate::element::AsyncSink for AsyncUdpSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let data = ctx.input();

        if self.connected {
            self.socket.send(data).await?;
        } else if let Some(dest) = self.destination {
            self.socket.send_to(data, dest).await?;
        } else {
            return Err(Error::Config("UDP sink has no destination set".into()));
        }

        self.bytes_written += data.len() as u64;
        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{Sink, Source};
    use crate::memory::CpuArena;
    use std::thread;

    #[test]
    fn test_udp_src_creation() {
        let src = UdpSrc::bind("127.0.0.1:0").unwrap();
        assert!(src.name.contains("udpsrc"));
        assert!(src.local_addr().is_ok());
    }

    #[test]
    fn test_udp_sink_creation() {
        let sink = UdpSink::bind("127.0.0.1:0").unwrap();
        assert!(sink.name.contains("udpsink"));
        assert!(sink.local_addr().is_ok());
    }

    #[test]
    fn test_udp_sink_connected() {
        let sink = UdpSink::connect("127.0.0.1:9999").unwrap();
        assert!(sink.connected);
        assert_eq!(sink.destination, Some("127.0.0.1:9999".parse().unwrap()));
    }

    #[test]
    fn test_udp_roundtrip() {
        // Create receiver
        let mut src = UdpSrc::bind("127.0.0.1:0").unwrap();
        let recv_addr = src.local_addr().unwrap();

        // Spawn sender thread
        let handle = thread::spawn(move || {
            let socket = UdpSocket::bind("127.0.0.1:0").unwrap();
            socket.send_to(b"hello udp", recv_addr).unwrap();
        });

        // Create arena for receiving
        let arena = Arc::new(CpuArena::new(65535, 4).unwrap());
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);

        // Receive the datagram
        let result = src.produce(&mut ctx).unwrap();
        match result {
            ProduceResult::Produced(n) => {
                let buffer = ctx.finalize(n);
                assert_eq!(buffer.as_bytes(), b"hello udp");
            }
            _ => panic!("expected Produced result"),
        }
        assert!(src.last_sender().is_some());

        handle.join().unwrap();
    }

    #[test]
    fn test_udp_sink_roundtrip() {
        // Create receiver socket
        let receiver = UdpSocket::bind("127.0.0.1:0").unwrap();
        let recv_addr = receiver.local_addr().unwrap();

        // Create sink and send data
        let mut sink = UdpSink::connect(recv_addr).unwrap();

        let segment = Arc::new(HeapSegment::new(9).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(b"hello udp".as_ptr(), ptr, 9);
        }
        let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, 9);
        let buffer = Buffer::new(handle, Metadata::default());

        let ctx = ConsumeContext::new(&buffer);
        sink.consume(&ctx).unwrap();

        // Receive and verify
        let mut buf = [0u8; 64];
        let (n, _sender) = receiver.recv_from(&mut buf).unwrap();
        assert_eq!(&buf[..n], b"hello udp");
    }

    #[test]
    fn test_udp_sink_no_destination() {
        let mut sink = UdpSink::bind("127.0.0.1:0").unwrap();

        let segment = Arc::new(HeapSegment::new(4).unwrap());
        let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, 4);
        let buffer = Buffer::new(handle, Metadata::default());

        // Should fail without destination
        let ctx = ConsumeContext::new(&buffer);
        let result = sink.consume(&ctx);
        assert!(result.is_err());
    }

    #[test]
    fn test_udp_with_buffer_size() {
        let src = UdpSrc::bind("127.0.0.1:0").unwrap().with_buffer_size(1024);
        assert_eq!(src.buffer_size, 1024);
    }

    #[test]
    fn test_udp_with_name() {
        let src = UdpSrc::bind("127.0.0.1:0")
            .unwrap()
            .with_name("my-udp-source");
        assert_eq!(src.name(), "my-udp-source");
    }

    #[tokio::test]
    async fn test_async_udp_roundtrip() {
        // Create async receiver
        let mut src = AsyncUdpSrc::bind("127.0.0.1:0").await.unwrap();
        let recv_addr = src.local_addr().unwrap();

        // Spawn sender task
        let sender = tokio::spawn(async move {
            let socket = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
            socket.send_to(b"async hello", recv_addr).await.unwrap();
        });

        // Receive using the convenience method
        let buffer = src.recv().await.unwrap().unwrap();
        assert_eq!(buffer.as_bytes(), b"async hello");

        sender.await.unwrap();
    }

    #[tokio::test]
    async fn test_async_udp_sink() {
        // Create receiver
        let receiver = tokio::net::UdpSocket::bind("127.0.0.1:0").await.unwrap();
        let recv_addr = receiver.local_addr().unwrap();

        // Create sink and send
        let mut sink = AsyncUdpSink::connect(recv_addr).await.unwrap();

        let segment = Arc::new(HeapSegment::new(11).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(b"async hello".as_ptr(), ptr, 11);
        }
        let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, 11);
        let buffer = Buffer::new(handle, Metadata::default());

        sink.send(buffer).await.unwrap();

        // Verify
        let mut buf = [0u8; 64];
        let (n, _) = receiver.recv_from(&mut buf).await.unwrap();
        assert_eq!(&buf[..n], b"async hello");
    }
}
