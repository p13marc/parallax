//! Unix domain socket source and sink elements.
//!
//! Provides local IPC via Unix domain sockets with lower overhead than TCP.
//!
//! - [`UnixSrc`]: Reads data from a Unix socket connection
//! - [`UnixSink`]: Writes data to a Unix socket connection (an `AsyncSink`)
//!
//! The old `AsyncUnixSrc`/`AsyncUnixSink` pair was deleted with #172: neither
//! implemented an element trait, and both re-connected (server mode: re-bound
//! the socket path) on every call, which was never a usable transport.
//! `UnixSink` itself is the async one now.

use crate::element::{AsyncSink, ConsumeContext, ProduceContext, ProduceResult, Source};
use crate::error::{Error, Result};
use std::io::Read;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tokio::io::AsyncWriteExt;

/// Mode of operation for Unix socket source.
#[derive(Debug, Clone)]
pub enum UnixMode {
    /// Connect to a socket path as a client.
    Client(PathBuf),
    /// Listen on a socket path and accept one connection.
    Server(PathBuf),
}

/// A Unix domain socket source that reads data from a local connection.
///
/// Can operate in two modes:
/// - **Client mode**: Connects to an existing Unix socket
/// - **Server mode**: Creates a socket and waits for a connection
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::UnixSrc;
///
/// // Client mode - connect to an existing socket
/// let src = UnixSrc::connect("/tmp/my.sock")?;
///
/// // Server mode - create socket and wait for connection
/// let src = UnixSrc::listen("/tmp/my.sock")?;
/// ```
pub struct UnixSrc {
    name: String,
    stream: Option<UnixStream>,
    listener: Option<UnixListener>,
    mode: UnixMode,
    buffer_size: usize,
    connected: bool,
    bytes_read: u64,
    sequence: u64,
    read_timeout: Option<Duration>,
    cleanup_on_drop: bool,
}

impl UnixSrc {
    /// Create a new Unix socket source in client mode.
    ///
    /// Does not connect immediately - connection happens on first `produce()` call.
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let name = format!("unixsrc-{}", path.display());

        Ok(Self {
            name,
            stream: None,
            listener: None,
            mode: UnixMode::Client(path),
            buffer_size: 64 * 1024,
            connected: false,
            bytes_read: 0,
            sequence: 0,
            read_timeout: None,
            cleanup_on_drop: false,
        })
    }

    /// Create a new Unix socket source in server mode.
    ///
    /// Creates the socket immediately but does not accept - acceptance happens
    /// on first `produce()` call.
    pub fn listen<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let name = format!("unixsrc-listener-{}", path.display());

        // Remove existing socket file if present
        let _ = std::fs::remove_file(&path);

        let listener = UnixListener::bind(&path)?;

        Ok(Self {
            name,
            stream: None,
            listener: Some(listener),
            mode: UnixMode::Server(path),
            buffer_size: 64 * 1024,
            connected: false,
            bytes_read: 0,
            sequence: 0,
            read_timeout: None,
            cleanup_on_drop: true,
        })
    }

    /// Set the buffer size for reads.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size.max(1);
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

    /// Get the socket path.
    pub fn path(&self) -> &Path {
        match &self.mode {
            UnixMode::Client(p) | UnixMode::Server(p) => p,
        }
    }

    fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        match &self.mode {
            UnixMode::Client(path) => {
                let stream = UnixStream::connect(path)?;
                if let Some(timeout) = self.read_timeout {
                    stream.set_read_timeout(Some(timeout))?;
                }
                self.stream = Some(stream);
            }
            UnixMode::Server(_) => {
                if let Some(ref listener) = self.listener {
                    let (stream, _) = listener.accept()?;
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

impl Source for UnixSrc {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        self.ensure_connected()?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Element("not connected".into()))?;

        let output = ctx.output();

        match stream.read(output) {
            Ok(0) => Ok(ProduceResult::Eos), // EOF
            Ok(n) => {
                self.bytes_read += n as u64;
                ctx.set_sequence(self.sequence);
                self.sequence += 1;

                Ok(ProduceResult::Produced(n))
            }
            Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                // Timeout - return empty buffer with timeout flag
                ctx.set_sequence(self.sequence);
                ctx.metadata_mut().flags = ctx
                    .metadata()
                    .flags
                    .insert(crate::metadata::BufferFlags::TIMEOUT);
                self.sequence += 1;
                Ok(ProduceResult::Produced(0))
            }
            Err(e) => Err(e.into()),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for UnixSrc {
    fn drop(&mut self) {
        if self.cleanup_on_drop
            && let UnixMode::Server(ref path) = self.mode
        {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// A Unix domain socket sink that writes data to a local connection.
///
/// An [`AsyncSink`] (#172): the accept, the connect and every write await on
/// tokio's reactor, so a peer that never connects — or connects and stops
/// reading — pends this element's future instead of parking a tokio worker
/// (the same liveness class as `IpcSink`). Register with `add_async_sink`.
///
/// Can operate in two modes:
/// - **Client mode**: Connects to an existing Unix socket
/// - **Server mode**: Creates a socket and waits for a connection
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::UnixSink;
///
/// // Client mode - connect to an existing socket
/// let sink = UnixSink::connect("/tmp/my.sock")?;
///
/// // Server mode - create socket and wait for connection
/// let sink = UnixSink::listen("/tmp/my.sock")?;
/// ```
pub struct UnixSink {
    name: String,
    stream: Option<tokio::net::UnixStream>,
    /// Bound eagerly (std, no reactor needed at construction), converted to a
    /// tokio listener at the first `consume`.
    listener: Option<UnixListener>,
    tokio_listener: Option<tokio::net::UnixListener>,
    mode: UnixMode,
    connected: bool,
    bytes_written: u64,
    write_timeout: Option<Duration>,
    cleanup_on_drop: bool,
}

impl UnixSink {
    /// Create a new Unix socket sink in client mode.
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let name = format!("unixsink-{}", path.display());

        Ok(Self {
            name,
            stream: None,
            listener: None,
            tokio_listener: None,
            mode: UnixMode::Client(path),
            connected: false,
            bytes_written: 0,
            write_timeout: None,
            cleanup_on_drop: false,
        })
    }

    /// Create a new Unix socket sink in server mode.
    pub fn listen<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let name = format!("unixsink-listener-{}", path.display());

        // Remove existing socket file if present
        let _ = std::fs::remove_file(&path);

        let listener = UnixListener::bind(&path)?;

        Ok(Self {
            name,
            stream: None,
            listener: Some(listener),
            tokio_listener: None,
            mode: UnixMode::Server(path),
            connected: false,
            bytes_written: 0,
            write_timeout: None,
            cleanup_on_drop: true,
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

    /// Get the socket path.
    pub fn path(&self) -> &Path {
        match &self.mode {
            UnixMode::Client(p) | UnixMode::Server(p) => p,
        }
    }

    async fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        match &self.mode {
            UnixMode::Client(path) => {
                self.stream = Some(tokio::net::UnixStream::connect(path).await?);
            }
            UnixMode::Server(_) => {
                if let Some(listener) = self.listener.take() {
                    listener.set_nonblocking(true)?;
                    self.tokio_listener = Some(tokio::net::UnixListener::from_std(listener)?);
                }
                if let Some(ref listener) = self.tokio_listener {
                    let (stream, _) = listener.accept().await?;
                    self.stream = Some(stream);
                }
            }
        }

        self.connected = true;
        Ok(())
    }
}

impl AsyncSink for UnixSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        self.ensure_connected().await?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Element("not connected".into()))?;

        let data = ctx.input();
        // tokio streams have no socket-level write timeout; the deadline
        // wraps the future instead, which covers the same stall.
        match self.write_timeout {
            Some(limit) => tokio::time::timeout(limit, stream.write_all(data))
                .await
                .map_err(|_| {
                    Error::Element(format!("unix sink write timed out after {limit:?}"))
                })??,
            None => stream.write_all(data).await?,
        }
        self.bytes_written += data.len() as u64;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for UnixSink {
    fn drop(&mut self) {
        if self.cleanup_on_drop
            && let UnixMode::Server(ref path) = self.mode
        {
            let _ = std::fs::remove_file(path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{Buffer, MemoryHandle};
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::io::Write;
    use std::sync::OnceLock;
    use std::thread;
    use tempfile::tempdir;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(4096, 64).unwrap())
    }

    fn make_buffer(data: &[u8], seq: u64) -> Buffer {
        let mut slot = test_arena().acquire().unwrap();
        if !data.is_empty() {
            slot.data_mut()[..data.len()].copy_from_slice(data);
        }
        let handle = MemoryHandle::with_len(slot, data.len());
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[tokio::test]
    async fn test_unix_roundtrip() -> Result<()> {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("test.sock");

        let path_clone = socket_path.clone();
        let server = thread::spawn(move || -> Result<Vec<u8>> {
            let arena = SharedArena::new(4096, 8).unwrap();
            let mut src = UnixSrc::listen(&path_clone)?;
            let mut data = Vec::new();
            loop {
                let slot = arena.acquire().unwrap();
                let mut ctx = ProduceContext::new(slot);
                match src.produce(&mut ctx)? {
                    ProduceResult::Produced(n) => {
                        let buf = ctx.finalize(n);
                        if buf.metadata().flags.is_eos() {
                            break;
                        }
                        data.extend_from_slice(buf.as_bytes());
                        if data.len() >= 11 {
                            break;
                        }
                    }
                    ProduceResult::Eos => break,
                    _ => break,
                }
            }
            Ok(data)
        });

        // Give server time to start listening
        tokio::time::sleep(Duration::from_millis(50)).await;

        let mut sink = UnixSink::connect(&socket_path)?;
        let buf1 = make_buffer(b"Hello", 0);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1).await?;
        let buf2 = make_buffer(b" World", 1);
        let ctx2 = ConsumeContext::new(&buf2);
        sink.consume(&ctx2).await?;

        let received = server.join().unwrap()?;
        assert_eq!(received, b"Hello World");

        Ok(())
    }

    #[test]
    fn test_unix_src_client_mode() {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("client.sock");

        // Create a server first
        let path_clone = socket_path.clone();
        let _server = thread::spawn(move || {
            let listener = UnixListener::bind(&path_clone).unwrap();
            let (mut stream, _) = listener.accept().unwrap();
            stream.write_all(b"test data").unwrap();
        });

        thread::sleep(Duration::from_millis(50));

        let src = UnixSrc::connect(&socket_path);
        assert!(src.is_ok());
    }

    #[test]
    fn test_unix_sink_with_name() -> Result<()> {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("named.sock");

        let sink = UnixSink::listen(&socket_path)?.with_name("my-sink");
        assert_eq!(sink.name(), "my-sink");

        Ok(())
    }

    #[test]
    fn test_unix_src_with_buffer_size() -> Result<()> {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("buffered.sock");

        let src = UnixSrc::listen(&socket_path)?.with_buffer_size(1024);
        assert_eq!(src.buffer_size, 1024);

        Ok(())
    }

    #[test]
    fn test_unix_cleanup_on_drop() -> Result<()> {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("cleanup.sock");

        {
            let _src = UnixSrc::listen(&socket_path)?;
            assert!(socket_path.exists());
        }

        // Socket file should be cleaned up
        assert!(!socket_path.exists());

        Ok(())
    }
}
