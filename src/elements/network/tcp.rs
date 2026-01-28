//! TCP source and sink elements.
//!
//! Provides network-based streaming via TCP sockets.
//!
//! - [`TcpSrc`]: Reads data from a TCP connection (client or listener)
//! - [`TcpSink`]: Writes data to a TCP connection

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{AsyncSource, ConsumeContext, ProduceContext, ProduceResult};
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use crate::metadata::Metadata;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream, ToSocketAddrs};
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
    /// Arena for buffer allocation.
    arena: Option<SharedArena>,
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
            arena: None,
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
            arena: None,
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
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        self.ensure_connected()?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("TCP stream not connected".into()))?;

        // Check if context has a buffer
        if ctx.has_buffer() {
            // Use the provided buffer from the context
            let output = ctx.output();
            match stream.read(output) {
                Ok(0) => {
                    // Connection closed
                    Ok(ProduceResult::Eos)
                }
                Ok(n) => {
                    self.bytes_read += n as u64;
                    ctx.set_sequence(self.sequence);
                    self.sequence += 1;
                    Ok(ProduceResult::Produced(n))
                }
                Err(e) => Err(Error::Io(e)),
            }
        } else {
            // No buffer provided, allocate from arena
            // Initialize arena lazily if needed
            if self.arena.is_none() {
                self.arena = Some(SharedArena::new(self.buffer_size, 8)?);
            }
            let arena = self.arena.as_ref().unwrap();

            let mut slot = arena
                .acquire()
                .ok_or_else(|| Error::Element("arena exhausted".into()))?;
            let slice = slot.data_mut();

            match stream.read(slice) {
                Ok(0) => {
                    // Connection closed
                    Ok(ProduceResult::Eos)
                }
                Ok(n) => {
                    self.bytes_read += n as u64;
                    let seq = self.sequence;
                    self.sequence += 1;

                    let handle = MemoryHandle::with_len(slot, n);
                    let metadata = Metadata::from_sequence(seq);
                    Ok(ProduceResult::OwnBuffer(Buffer::new(handle, metadata)))
                }
                Err(e) => Err(Error::Io(e)),
            }
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
    /// Arena for buffer allocation.
    arena: Option<SharedArena>,
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
            arena: None,
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
            arena: None,
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
    async fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        use tokio::io::AsyncReadExt;

        self.ensure_connected().await?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("TCP stream not connected".into()))?;

        // Check if context has a buffer
        if ctx.has_buffer() {
            // Use the provided buffer from the context
            let output = ctx.output();
            match stream.read(output).await {
                Ok(0) => {
                    // Connection closed
                    Ok(ProduceResult::Eos)
                }
                Ok(n) => {
                    self.bytes_read += n as u64;
                    ctx.set_sequence(self.sequence);
                    self.sequence += 1;
                    Ok(ProduceResult::Produced(n))
                }
                Err(e) => Err(Error::Io(e)),
            }
        } else {
            // No buffer provided, allocate from arena
            // Initialize arena lazily if needed
            if self.arena.is_none() {
                self.arena = Some(SharedArena::new(self.buffer_size, 8)?);
            }
            let arena = self.arena.as_ref().unwrap();

            let mut slot = arena
                .acquire()
                .ok_or_else(|| Error::Element("arena exhausted".into()))?;
            let slice = slot.data_mut();

            match stream.read(slice).await {
                Ok(0) => {
                    // Connection closed
                    Ok(ProduceResult::Eos)
                }
                Ok(n) => {
                    self.bytes_read += n as u64;
                    let seq = self.sequence;
                    self.sequence += 1;

                    let handle = MemoryHandle::with_len(slot, n);
                    let metadata = Metadata::from_sequence(seq);
                    Ok(ProduceResult::OwnBuffer(Buffer::new(handle, metadata)))
                }
                Err(e) => Err(Error::Io(e)),
            }
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
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.ensure_connected()?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("TCP stream not connected".into()))?;

        let data = ctx.input();
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
    pub async fn write(&mut self, buffer: &Buffer) -> Result<()> {
        let ctx = ConsumeContext::new(buffer);
        <Self as crate::element::AsyncSink>::consume(self, &ctx).await
    }
}

impl crate::element::AsyncSink for AsyncTcpSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        use tokio::io::AsyncWriteExt;

        self.ensure_connected().await?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Pipeline("TCP stream not connected".into()))?;

        let data = ctx.input();
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
        use crate::memory::SharedArena;
        use std::sync::OnceLock;

        fn test_arena() -> &'static SharedArena {
            static ARENA: OnceLock<SharedArena> = OnceLock::new();
            ARENA.get_or_init(|| SharedArena::new(1024, 16).unwrap())
        }

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

        // Read from the source using the new context-based API
        let slot = test_arena().acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        let result = src.produce(&mut ctx).unwrap();

        match result {
            ProduceResult::Produced(n) => {
                let buffer = ctx.finalize(n);
                assert_eq!(buffer.as_bytes(), b"hello world");
            }
            _ => panic!("expected Produced result"),
        }

        // Next read should return Eos (connection closed)
        let slot2 = test_arena().acquire().unwrap();
        let mut ctx2 = ProduceContext::new(slot2);
        let next = src.produce(&mut ctx2).unwrap();
        assert!(matches!(next, ProduceResult::Eos));

        handle.join().unwrap();
    }

    #[test]
    fn test_tcp_sink_roundtrip() {
        use crate::element::Sink;
        use crate::memory::SharedArena;
        use std::io::Read;
        use std::sync::OnceLock;

        fn test_arena() -> &'static SharedArena {
            static ARENA: OnceLock<SharedArena> = OnceLock::new();
            ARENA.get_or_init(|| SharedArena::new(1024, 16).unwrap())
        }

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
        let mut slot = test_arena().acquire().unwrap();
        slot.data_mut()[..11].copy_from_slice(b"hello world");
        let handle_mem = crate::buffer::MemoryHandle::with_len(slot, 11);
        let buffer = Buffer::new(handle_mem, Metadata::default());
        let ctx = ConsumeContext::new(&buffer);
        sink.consume(&ctx).unwrap();

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
        use crate::memory::SharedArena;
        use std::sync::OnceLock;
        use tokio::io::AsyncWriteExt;

        fn test_arena() -> &'static SharedArena {
            static ARENA: OnceLock<SharedArena> = OnceLock::new();
            ARENA.get_or_init(|| SharedArena::new(1024, 16).unwrap())
        }

        // Create async server source
        let mut src = AsyncTcpSrc::listen("127.0.0.1:0").await.unwrap();
        let _addr = match &src.mode {
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

        // Read from async source using the new context-based API
        let slot = test_arena().acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        let result = src.produce(&mut ctx).await.unwrap();

        match result {
            ProduceResult::Produced(n) => {
                let buffer = ctx.finalize(n);
                assert_eq!(buffer.as_bytes(), b"async hello");
            }
            _ => panic!("expected Produced result"),
        }

        client.await.unwrap();
    }
}
