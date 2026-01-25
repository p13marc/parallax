//! TCP source and sink elements.
//!
//! Provides network-based streaming via TCP sockets.
//!
//! - [`TcpSrc`]: Reads data from a TCP connection (client or listener)
//! - [`TcpSink`]: Writes data to a TCP connection

use crate::buffer::Buffer;
use crate::element::AsyncSource;
use crate::error::{Error, Result};
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
use std::sync::Arc;
use std::time::Duration;

/// Mode of operation for TCP source.
#[derive(Debug, Clone)]
pub enum TcpMode {
    /// Connect to a remote address as a client.
    Client(SocketAddr),
    /// Listen on an address and accept one connection.
    Server(SocketAddr),
}

/// A TCP source that reads data from a network connection.
///
/// Can operate in two modes:
/// - **Client mode**: Connects to a remote TCP server
/// - **Server mode**: Listens for an incoming connection
///
/// # Example
///
/// ```rust,ignore
/// // Client mode - connect to a server
/// let src = TcpSrc::connect("127.0.0.1:8080")?;
///
/// // Server mode - wait for a connection
/// let src = TcpSrc::listen("0.0.0.0:8080")?;
/// ```
pub struct TcpSrc {
    name: String,
    stream: Option<TcpStream>,
    listener: Option<TcpListener>,
    mode: TcpMode,
    buffer_size: usize,
    connected: bool,
    bytes_read: u64,
    sequence: u64,
    read_timeout: Option<Duration>,
}

impl TcpSrc {
    /// Create a new TCP source in client mode.
    ///
    /// Does not connect immediately - connection happens on first `produce()` call.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        Ok(Self {
            name: format!("tcpsrc-{}", addr),
            stream: None,
            listener: None,
            mode: TcpMode::Client(addr),
            buffer_size: 64 * 1024, // 64KB default
            connected: false,
            bytes_read: 0,
            sequence: 0,
            read_timeout: None,
        })
    }

    /// Create a new TCP source in server mode.
    ///
    /// Binds immediately but does not accept - acceptance happens on first `produce()` call.
    pub fn listen<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let listener = TcpListener::bind(addr)?;

        Ok(Self {
            name: format!("tcpsrc-listener-{}", addr),
            stream: None,
            listener: Some(listener),
            mode: TcpMode::Server(addr),
            buffer_size: 64 * 1024,
            connected: false,
            bytes_read: 0,
            sequence: 0,
            read_timeout: None,
        })
    }

    /// Set the buffer size for reads.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
        self
    }

    /// Set a custom name for this source.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the read timeout.
    pub fn with_read_timeout(mut self, timeout: Duration) -> Self {
        self.read_timeout = Some(timeout);
        self
    }

    /// Get the number of bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Get the local address (if bound/connected).
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.stream
            .as_ref()
            .and_then(|s| s.local_addr().ok())
            .or_else(|| self.listener.as_ref().and_then(|l| l.local_addr().ok()))
    }

    /// Get the peer address (if connected).
    pub fn peer_addr(&self) -> Option<SocketAddr> {
        self.stream.as_ref().and_then(|s| s.peer_addr().ok())
    }

    fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        match &self.mode {
            TcpMode::Client(addr) => {
                let stream = TcpStream::connect(addr)?;
                if let Some(timeout) = self.read_timeout {
                    stream.set_read_timeout(Some(timeout))?;
                }
                self.stream = Some(stream);
            }
            TcpMode::Server(_) => {
                if let Some(ref listener) = self.listener {
                    let (stream, _peer) = listener.accept()?;
                    if let Some(timeout) = self.read_timeout {
                        stream.set_read_timeout(Some(timeout))?;
                    }
                    self.stream = Some(stream);
                }
            }
        }

        self.connected = true;
        Ok(())
    }
}

impl crate::element::Source for TcpSrc {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        self.ensure_connected()?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("TCP stream not connected".into()))?;

        // Allocate buffer
        let segment = Arc::new(HeapSegment::new(self.buffer_size)?);
        let ptr = segment
            .as_mut_ptr()
            .ok_or_else(|| Error::Element("cannot get mutable pointer".into()))?;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.buffer_size) };

        // Read data
        match stream.read(slice) {
            Ok(0) => {
                // Connection closed
                Ok(None)
            }
            Ok(n) => {
                self.bytes_read += n as u64;
                let seq = self.sequence;
                self.sequence += 1;

                let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, n);
                let metadata = Metadata::with_sequence(seq);
                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => Err(Error::Io(e)),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Async TCP source for use with async runtimes.
pub struct AsyncTcpSrc {
    name: String,
    stream: Option<tokio::net::TcpStream>,
    listener: Option<tokio::net::TcpListener>,
    mode: TcpMode,
    buffer_size: usize,
    connected: bool,
    bytes_read: u64,
    sequence: u64,
}

impl AsyncTcpSrc {
    /// Create a new async TCP source in client mode.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        Ok(Self {
            name: format!("async-tcpsrc-{}", addr),
            stream: None,
            listener: None,
            mode: TcpMode::Client(addr),
            buffer_size: 64 * 1024,
            connected: false,
            bytes_read: 0,
            sequence: 0,
        })
    }

    /// Create a new async TCP source in server mode.
    pub async fn listen<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let listener = tokio::net::TcpListener::bind(addr).await?;

        Ok(Self {
            name: format!("async-tcpsrc-listener-{}", addr),
            stream: None,
            listener: Some(listener),
            mode: TcpMode::Server(addr),
            buffer_size: 64 * 1024,
            connected: false,
            bytes_read: 0,
            sequence: 0,
        })
    }

    /// Set the buffer size for reads.
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

    async fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        match &self.mode {
            TcpMode::Client(addr) => {
                let stream = tokio::net::TcpStream::connect(addr).await?;
                self.stream = Some(stream);
            }
            TcpMode::Server(_) => {
                if let Some(ref listener) = self.listener {
                    let (stream, _peer) = listener.accept().await?;
                    self.stream = Some(stream);
                }
            }
        }

        self.connected = true;
        Ok(())
    }
}

impl AsyncSource for AsyncTcpSrc {
    async fn produce(&mut self) -> Result<Option<Buffer>> {
        use tokio::io::AsyncReadExt;

        self.ensure_connected().await?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("TCP stream not connected".into()))?;

        // Allocate buffer
        let segment = Arc::new(HeapSegment::new(self.buffer_size)?);
        let ptr = segment
            .as_mut_ptr()
            .ok_or_else(|| Error::Element("cannot get mutable pointer".into()))?;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.buffer_size) };

        // Read data asynchronously
        match stream.read(slice).await {
            Ok(0) => {
                // Connection closed
                Ok(None)
            }
            Ok(n) => {
                self.bytes_read += n as u64;
                let seq = self.sequence;
                self.sequence += 1;

                let handle = crate::buffer::MemoryHandle::from_segment_with_len(segment, n);
                let metadata = Metadata::with_sequence(seq);
                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => Err(Error::Io(e)),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// A TCP sink that writes data to a network connection.
///
/// Can operate in two modes:
/// - **Client mode**: Connects to a remote TCP server
/// - **Server mode**: Listens for an incoming connection
pub struct TcpSink {
    name: String,
    stream: Option<TcpStream>,
    listener: Option<TcpListener>,
    mode: TcpMode,
    connected: bool,
    bytes_written: u64,
    write_timeout: Option<Duration>,
}

impl TcpSink {
    /// Create a new TCP sink in client mode.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        Ok(Self {
            name: format!("tcpsink-{}", addr),
            stream: None,
            listener: None,
            mode: TcpMode::Client(addr),
            connected: false,
            bytes_written: 0,
            write_timeout: None,
        })
    }

    /// Create a new TCP sink in server mode.
    pub fn listen<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let listener = TcpListener::bind(addr)?;

        Ok(Self {
            name: format!("tcpsink-listener-{}", addr),
            stream: None,
            listener: Some(listener),
            mode: TcpMode::Server(addr),
            connected: false,
            bytes_written: 0,
            write_timeout: None,
        })
    }

    /// Set a custom name for this sink.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the write timeout.
    pub fn with_write_timeout(mut self, timeout: Duration) -> Self {
        self.write_timeout = Some(timeout);
        self
    }

    /// Get the number of bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Get the local address (if bound/connected).
    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.stream
            .as_ref()
            .and_then(|s| s.local_addr().ok())
            .or_else(|| self.listener.as_ref().and_then(|l| l.local_addr().ok()))
    }

    /// Get the peer address (if connected).
    pub fn peer_addr(&self) -> Option<SocketAddr> {
        self.stream.as_ref().and_then(|s| s.peer_addr().ok())
    }

    fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        match &self.mode {
            TcpMode::Client(addr) => {
                let stream = TcpStream::connect(addr)?;
                if let Some(timeout) = self.write_timeout {
                    stream.set_write_timeout(Some(timeout))?;
                }
                self.stream = Some(stream);
            }
            TcpMode::Server(_) => {
                if let Some(ref listener) = self.listener {
                    let (stream, _peer) = listener.accept()?;
                    if let Some(timeout) = self.write_timeout {
                        stream.set_write_timeout(Some(timeout))?;
                    }
                    self.stream = Some(stream);
                }
            }
        }

        self.connected = true;
        Ok(())
    }
}

impl crate::element::Sink for TcpSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        self.ensure_connected()?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("TCP stream not connected".into()))?;

        let data = buffer.as_bytes();
        stream.write_all(data)?;
        self.bytes_written += data.len() as u64;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Async TCP sink for use with async runtimes.
pub struct AsyncTcpSink {
    name: String,
    stream: Option<tokio::net::TcpStream>,
    listener: Option<tokio::net::TcpListener>,
    mode: TcpMode,
    connected: bool,
    bytes_written: u64,
}

impl AsyncTcpSink {
    /// Create a new async TCP sink in client mode.
    pub fn connect<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        Ok(Self {
            name: format!("async-tcpsink-{}", addr),
            stream: None,
            listener: None,
            mode: TcpMode::Client(addr),
            connected: false,
            bytes_written: 0,
        })
    }

    /// Create a new async TCP sink in server mode.
    pub async fn listen<A: ToSocketAddrs>(addr: A) -> Result<Self> {
        let addr = addr
            .to_socket_addrs()?
            .next()
            .ok_or_else(|| Error::Config("invalid address".into()))?;

        let listener = tokio::net::TcpListener::bind(addr).await?;

        Ok(Self {
            name: format!("async-tcpsink-listener-{}", addr),
            stream: None,
            listener: Some(listener),
            mode: TcpMode::Server(addr),
            connected: false,
            bytes_written: 0,
        })
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

    async fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        match &self.mode {
            TcpMode::Client(addr) => {
                let stream = tokio::net::TcpStream::connect(addr).await?;
                self.stream = Some(stream);
            }
            TcpMode::Server(_) => {
                if let Some(ref listener) = self.listener {
                    let (stream, _peer) = listener.accept().await?;
                    self.stream = Some(stream);
                }
            }
        }

        self.connected = true;
        Ok(())
    }

    /// Write a buffer asynchronously.
    pub async fn write(&mut self, buffer: Buffer) -> Result<()> {
        <Self as crate::element::AsyncSink>::consume(self, buffer).await
    }
}

impl crate::element::AsyncSink for AsyncTcpSink {
    async fn consume(&mut self, buffer: Buffer) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        self.ensure_connected().await?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("TCP stream not connected".into()))?;

        let data = buffer.as_bytes();
        stream.write_all(data).await?;
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
    use crate::element::Source;
    use std::thread;

    #[test]
    fn test_tcp_client_mode_creation() {
        let src = TcpSrc::connect("127.0.0.1:0").unwrap();
        assert!(src.name.contains("tcpsrc"));
        assert!(!src.connected);
    }

    #[test]
    fn test_tcp_server_mode_creation() {
        let src = TcpSrc::listen("127.0.0.1:0").unwrap();
        assert!(src.name.contains("tcpsrc-listener"));
        assert!(!src.connected);
        // Should have a listener bound
        assert!(src.local_addr().is_some());
    }

    #[test]
    fn test_tcp_sink_creation() {
        let sink = TcpSink::connect("127.0.0.1:0").unwrap();
        assert!(sink.name.contains("tcpsink"));
        assert!(!sink.connected);
    }

    #[test]
    fn test_tcp_roundtrip() {
        // Create a server source
        let mut src = TcpSrc::listen("127.0.0.1:0").unwrap();
        let addr = src.local_addr().unwrap();

        // Spawn a client in another thread to send data
        let handle = thread::spawn(move || {
            let mut stream = TcpStream::connect(addr).unwrap();
            stream.write_all(b"hello world").unwrap();
            // Close the connection
            drop(stream);
        });

        // Read from the source
        let buffer = src.produce().unwrap().unwrap();
        assert_eq!(buffer.as_bytes(), b"hello world");

        // Next read should return None (connection closed)
        let next = src.produce().unwrap();
        assert!(next.is_none());

        handle.join().unwrap();
    }

    #[test]
    fn test_tcp_sink_roundtrip() {
        use crate::element::Sink;
        use std::io::Read;

        // Create a server that will receive data
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        // Spawn receiver
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = Vec::new();
            stream.read_to_end(&mut buf).unwrap();
            buf
        });

        // Create sink and send data
        let mut sink = TcpSink::connect(addr).unwrap();
        let segment = Arc::new(HeapSegment::new(11).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(b"hello world".as_ptr(), ptr, 11);
        }
        let handle_mem = crate::buffer::MemoryHandle::from_segment_with_len(segment, 11);
        let buffer = Buffer::new(handle_mem, Metadata::default());
        sink.consume(buffer).unwrap();

        // Close connection
        drop(sink);

        // Verify received data
        let received = handle.join().unwrap();
        assert_eq!(received, b"hello world");
    }

    #[test]
    fn test_tcp_with_buffer_size() {
        let src = TcpSrc::connect("127.0.0.1:0")
            .unwrap()
            .with_buffer_size(1024);
        assert_eq!(src.buffer_size, 1024);
    }

    #[test]
    fn test_tcp_with_name() {
        let src = TcpSrc::connect("127.0.0.1:0")
            .unwrap()
            .with_name("my-tcp-source");
        assert_eq!(src.name(), "my-tcp-source");
    }

    #[tokio::test]
    async fn test_async_tcp_roundtrip() {
        use tokio::io::AsyncWriteExt;

        // Create async server source
        let mut src = AsyncTcpSrc::listen("127.0.0.1:0").await.unwrap();
        let addr = match &src.mode {
            TcpMode::Server(a) => *a,
            _ => panic!("expected server mode"),
        };
        // Get the actual bound address from the listener
        let actual_addr = src.listener.as_ref().unwrap().local_addr().unwrap();

        // Spawn client task
        let client = tokio::spawn(async move {
            let mut stream = tokio::net::TcpStream::connect(actual_addr).await.unwrap();
            stream.write_all(b"async hello").await.unwrap();
            stream.shutdown().await.unwrap();
        });

        // Read from async source
        let buffer = src.produce().await.unwrap().unwrap();
        assert_eq!(buffer.as_bytes(), b"async hello");

        client.await.unwrap();
    }
}
