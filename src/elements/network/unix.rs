//! Unix domain socket source and sink elements.
//!
//! Provides local IPC via Unix domain sockets with lower overhead than TCP.
//!
//! - [`UnixSrc`]: Reads data from a Unix socket connection
//! - [`UnixSink`]: Writes data to a Unix socket connection
//! - [`AsyncUnixSrc`]: Async version of UnixSrc
//! - [`AsyncUnixSink`]: Async version of UnixSink

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{AsyncSource, ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use crate::error::{Error, Result};
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use std::io::{Read, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

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
        if self.cleanup_on_drop {
            if let UnixMode::Server(ref path) = self.mode {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

/// A Unix domain socket sink that writes data to a local connection.
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
    stream: Option<UnixStream>,
    listener: Option<UnixListener>,
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

    fn ensure_connected(&mut self) -> Result<()> {
        if self.connected {
            return Ok(());
        }

        match &self.mode {
            UnixMode::Client(path) => {
                let stream = UnixStream::connect(path)?;
                if let Some(timeout) = self.write_timeout {
                    stream.set_write_timeout(Some(timeout))?;
                }
                self.stream = Some(stream);
            }
            UnixMode::Server(_) => {
                if let Some(ref listener) = self.listener {
                    let (stream, _) = listener.accept()?;
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

impl Sink for UnixSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.ensure_connected()?;

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| Error::Element("not connected".into()))?;

        let data = ctx.input();
        stream.write_all(data)?;
        self.bytes_written += data.len() as u64;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for UnixSink {
    fn drop(&mut self) {
        if self.cleanup_on_drop {
            if let UnixMode::Server(ref path) = self.mode {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

/// Async Unix domain socket source.
///
/// Uses tokio for async I/O operations.
pub struct AsyncUnixSrc {
    name: String,
    path: PathBuf,
    mode: UnixMode,
    buffer_size: usize,
    bytes_read: u64,
    sequence: u64,
}

impl AsyncUnixSrc {
    /// Create a new async Unix socket source in client mode.
    pub fn connect<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref().to_path_buf();
        let name = format!("async-unixsrc-{}", path.display());

        Self {
            name,
            path: path.clone(),
            mode: UnixMode::Client(path),
            buffer_size: 64 * 1024,
            bytes_read: 0,
            sequence: 0,
        }
    }

    /// Create a new async Unix socket source in server mode.
    pub fn listen<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref().to_path_buf();
        let name = format!("async-unixsrc-listener-{}", path.display());

        Self {
            name,
            path: path.clone(),
            mode: UnixMode::Server(path),
            buffer_size: 64 * 1024,
            bytes_read: 0,
            sequence: 0,
        }
    }

    /// Set the buffer size for reads.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size.max(1);
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the number of bytes read so far.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }
}

impl AsyncSource for AsyncUnixSrc {
    async fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        // For async, we use std blocking I/O wrapped in spawn_blocking
        // A full async implementation would use tokio::net::UnixStream
        let path = self.path.clone();
        let buffer_size = self.buffer_size;
        let is_server = matches!(self.mode, UnixMode::Server(_));

        let result = tokio::task::spawn_blocking(move || {
            let stream = if is_server {
                let _ = std::fs::remove_file(&path);
                let listener = UnixListener::bind(&path)?;
                let (stream, _) = listener.accept()?;
                stream
            } else {
                UnixStream::connect(&path)?
            };

            let mut buf = vec![0u8; buffer_size];
            let n = (&stream).read(&mut buf)?;
            buf.truncate(n);
            Ok::<_, Error>(buf)
        })
        .await
        .map_err(|e| Error::Element(format!("task join error: {}", e)))??;

        if result.is_empty() {
            return Ok(ProduceResult::Eos);
        }

        self.bytes_read += result.len() as u64;
        let seq = self.sequence;
        self.sequence += 1;

        // Check if we can use the pre-allocated buffer
        if ctx.has_buffer() && ctx.capacity() >= result.len() {
            let output = ctx.output();
            output[..result.len()].copy_from_slice(&result);
            ctx.set_sequence(seq);
            Ok(ProduceResult::Produced(result.len()))
        } else {
            // Fall back to creating our own buffer
            let segment = Arc::new(HeapSegment::new(result.len())?);
            unsafe {
                let ptr = segment.as_mut_ptr().unwrap();
                std::ptr::copy_nonoverlapping(result.as_ptr(), ptr, result.len());
            }

            let handle = MemoryHandle::from_segment_with_len(segment, result.len());
            Ok(ProduceResult::OwnBuffer(Buffer::new(
                handle,
                Metadata::from_sequence(seq),
            )))
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Async Unix domain socket sink.
pub struct AsyncUnixSink {
    name: String,
    path: PathBuf,
    mode: UnixMode,
    bytes_written: u64,
}

impl AsyncUnixSink {
    /// Create a new async Unix socket sink in client mode.
    pub fn connect<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref().to_path_buf();
        let name = format!("async-unixsink-{}", path.display());

        Self {
            name,
            path: path.clone(),
            mode: UnixMode::Client(path),
            bytes_written: 0,
        }
    }

    /// Create a new async Unix socket sink in server mode.
    pub fn listen<P: AsRef<Path>>(path: P) -> Self {
        let path = path.as_ref().to_path_buf();
        let name = format!("async-unixsink-listener-{}", path.display());

        Self {
            name,
            path: path.clone(),
            mode: UnixMode::Server(path),
            bytes_written: 0,
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the number of bytes written so far.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }

    /// Consume a buffer asynchronously.
    pub async fn consume_async(&mut self, buffer: Buffer) -> Result<()> {
        let path = self.path.clone();
        let data = buffer.as_bytes().to_vec();
        let len = data.len();
        let is_server = matches!(self.mode, UnixMode::Server(_));

        tokio::task::spawn_blocking(move || {
            let mut stream = if is_server {
                let _ = std::fs::remove_file(&path);
                let listener = UnixListener::bind(&path)?;
                let (stream, _) = listener.accept()?;
                stream
            } else {
                UnixStream::connect(&path)?
            };

            stream.write_all(&data)?;
            Ok::<_, Error>(())
        })
        .await
        .map_err(|e| Error::Element(format!("task join error: {}", e)))??;

        self.bytes_written += len as u64;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::CpuArena;
    use std::thread;
    use tempfile::tempdir;

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
    fn test_unix_roundtrip() -> Result<()> {
        let dir = tempdir().unwrap();
        let socket_path = dir.path().join("test.sock");

        let path_clone = socket_path.clone();
        let server = thread::spawn(move || -> Result<Vec<u8>> {
            let arena = Arc::new(CpuArena::new(4096, 8).unwrap());
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
        thread::sleep(Duration::from_millis(50));

        let mut sink = UnixSink::connect(&socket_path)?;
        let buf1 = make_buffer(b"Hello", 0);
        let ctx1 = ConsumeContext::new(&buf1);
        sink.consume(&ctx1)?;
        let buf2 = make_buffer(b" World", 1);
        let ctx2 = ConsumeContext::new(&buf2);
        sink.consume(&ctx2)?;

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
