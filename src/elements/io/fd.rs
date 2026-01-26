//! File descriptor source and sink elements.
//!
//! Read from and write to raw file descriptors.

use crate::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use crate::error::{Error, Result};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};

/// A source that reads from a file descriptor.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::FdSrc;
/// use std::os::unix::io::AsRawFd;
///
/// // Read from stdin
/// let src = FdSrc::new(std::io::stdin().as_raw_fd());
///
/// // Or take ownership of an fd
/// let src = FdSrc::from_owned(owned_fd);
/// ```
pub struct FdSrc {
    name: String,
    fd: FdHolder,
    bytes_read: u64,
    sequence: u64,
}

/// A sink that writes to a file descriptor.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::FdSink;
/// use std::os::unix::io::AsRawFd;
///
/// // Write to stdout
/// let sink = FdSink::new(std::io::stdout().as_raw_fd());
///
/// // Or take ownership of an fd
/// let sink = FdSink::from_owned(owned_fd);
/// ```
pub struct FdSink {
    name: String,
    fd: FdHolder,
    bytes_written: u64,
}

enum FdHolder {
    Borrowed(RawFd),
    Owned(OwnedFd),
}

impl FdHolder {
    fn as_raw_fd(&self) -> RawFd {
        match self {
            FdHolder::Borrowed(fd) => *fd,
            FdHolder::Owned(fd) => fd.as_raw_fd(),
        }
    }
}

impl AsFd for FdHolder {
    fn as_fd(&self) -> BorrowedFd<'_> {
        unsafe { BorrowedFd::borrow_raw(self.as_raw_fd()) }
    }
}

impl FdSrc {
    /// Create a new FdSrc from a raw file descriptor.
    ///
    /// The fd is borrowed - the caller retains ownership.
    pub fn new(fd: RawFd) -> Self {
        Self {
            name: format!("fdsrc-{}", fd),
            fd: FdHolder::Borrowed(fd),
            bytes_read: 0,
            sequence: 0,
        }
    }

    /// Create a new FdSrc that takes ownership of the file descriptor.
    pub fn from_owned(fd: OwnedFd) -> Self {
        let raw = fd.as_raw_fd();
        Self {
            name: format!("fdsrc-{}", raw),
            fd: FdHolder::Owned(fd),
            bytes_read: 0,
            sequence: 0,
        }
    }

    /// Create a new FdSrc from a raw fd, taking ownership.
    ///
    /// # Safety
    /// The fd must be valid and not used elsewhere.
    pub unsafe fn from_raw_fd(fd: RawFd) -> Self {
        Self::from_owned(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the raw file descriptor.
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Get the number of bytes read.
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read
    }

    /// Get the number of buffers produced.
    pub fn buffers_produced(&self) -> u64 {
        self.sequence
    }
}

impl Source for FdSrc {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        let output = ctx.output();

        match rustix::io::read(&self.fd, output) {
            Ok(0) => {
                // EOF
                Ok(ProduceResult::Eos)
            }
            Ok(n) => {
                self.bytes_read += n as u64;
                ctx.set_sequence(self.sequence);
                self.sequence += 1;

                Ok(ProduceResult::Produced(n))
            }
            Err(e) => Err(Error::Io(e.into())),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl FdSink {
    /// Create a new FdSink from a raw file descriptor.
    ///
    /// The fd is borrowed - the caller retains ownership.
    pub fn new(fd: RawFd) -> Self {
        Self {
            name: format!("fdsink-{}", fd),
            fd: FdHolder::Borrowed(fd),
            bytes_written: 0,
        }
    }

    /// Create a new FdSink that takes ownership of the file descriptor.
    pub fn from_owned(fd: OwnedFd) -> Self {
        let raw = fd.as_raw_fd();
        Self {
            name: format!("fdsink-{}", raw),
            fd: FdHolder::Owned(fd),
            bytes_written: 0,
        }
    }

    /// Create a new FdSink from a raw fd, taking ownership.
    ///
    /// # Safety
    /// The fd must be valid and not used elsewhere.
    pub unsafe fn from_raw_fd(fd: RawFd) -> Self {
        Self::from_owned(unsafe { OwnedFd::from_raw_fd(fd) })
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the raw file descriptor.
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Get the number of bytes written.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

impl Sink for FdSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let data = ctx.input();
        let mut written = 0;

        while written < data.len() {
            match rustix::io::write(&self.fd, &data[written..]) {
                Ok(n) => {
                    written += n;
                }
                Err(e) => {
                    return Err(Error::Io(e.into()));
                }
            }
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
    use crate::memory::CpuArena;

    #[test]
    fn test_fdsrc_from_pipe() {
        // Create a pipe using rustix
        let (read_fd, write_fd) = rustix::pipe::pipe().unwrap();

        // Write some data
        rustix::io::write(&write_fd, b"hello from pipe").unwrap();
        drop(write_fd);

        // Create arena and source
        let arena = CpuArena::new(1024, 4).unwrap();
        let mut src = FdSrc::from_owned(read_fd);

        // Acquire slot and create context
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);

        // Produce buffer
        let result = src.produce(&mut ctx).unwrap();
        match result {
            ProduceResult::Produced(n) => {
                assert_eq!(n, 15);
                let buf = ctx.finalize(n);
                assert_eq!(buf.as_bytes(), b"hello from pipe");
            }
            _ => panic!("Expected Produced result"),
        }

        // Should be EOF on next read
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        let result = src.produce(&mut ctx).unwrap();
        assert!(matches!(result, ProduceResult::Eos));

        assert_eq!(src.bytes_read(), 15);
    }

    #[test]
    fn test_fdsink_to_pipe() {
        // Create a pipe using rustix
        let (read_fd, write_fd) = rustix::pipe::pipe().unwrap();

        // Create arena and sink
        let arena = CpuArena::new(1024, 4).unwrap();
        let mut sink = FdSink::from_owned(write_fd);

        // Create buffer using arena
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        let output = ctx.output();
        output[..11].copy_from_slice(b"hello world");
        let buffer = ctx.finalize(11);

        // Consume buffer
        let consume_ctx = ConsumeContext::new(&buffer);
        sink.consume(&consume_ctx).unwrap();
        drop(sink);

        // Read and verify using rustix
        let mut output = [0u8; 64];
        let n = rustix::io::read(&read_fd, &mut output).unwrap();
        assert_eq!(&output[..n], b"hello world");
    }

    #[test]
    fn test_fdsrc_with_name() {
        let (read_fd, _write_fd) = rustix::pipe::pipe().unwrap();
        let src = FdSrc::from_owned(read_fd).with_name("my-fdsrc");
        assert_eq!(src.name(), "my-fdsrc");
    }

    #[test]
    fn test_fdsink_with_name() {
        let (_read_fd, write_fd) = rustix::pipe::pipe().unwrap();
        let sink = FdSink::from_owned(write_fd).with_name("my-fdsink");
        assert_eq!(sink.name(), "my-fdsink");
    }

    #[test]
    fn test_fdsrc_multiple_reads() {
        // Create a pipe using rustix
        let (read_fd, write_fd) = rustix::pipe::pipe().unwrap();

        // Write multiple chunks
        rustix::io::write(&write_fd, b"chunk1").unwrap();
        rustix::io::write(&write_fd, b"chunk2").unwrap();
        drop(write_fd);

        // Create arena and source
        let arena = CpuArena::new(1024, 4).unwrap();
        let mut src = FdSrc::from_owned(read_fd);

        // Read first chunk
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        let result = src.produce(&mut ctx).unwrap();
        match result {
            ProduceResult::Produced(n) => {
                let buf = ctx.finalize(n);
                // Might read both chunks at once or separately
                assert!(buf.len() > 0);
            }
            _ => panic!("Expected Produced result"),
        }

        assert_eq!(src.buffers_produced(), 1);
    }

    #[test]
    fn test_fdsink_bytes_written() {
        let (_read_fd, write_fd) = rustix::pipe::pipe().unwrap();

        let arena = CpuArena::new(1024, 4).unwrap();
        let mut sink = FdSink::from_owned(write_fd);

        // Write first buffer
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        ctx.output()[..5].copy_from_slice(b"hello");
        let buffer = ctx.finalize(5);
        let consume_ctx = ConsumeContext::new(&buffer);
        sink.consume(&consume_ctx).unwrap();

        assert_eq!(sink.bytes_written(), 5);

        // Write second buffer
        let slot = arena.acquire().unwrap();
        let mut ctx = ProduceContext::new(slot);
        ctx.output()[..5].copy_from_slice(b"world");
        let buffer = ctx.finalize(5);
        let consume_ctx = ConsumeContext::new(&buffer);
        sink.consume(&consume_ctx).unwrap();

        assert_eq!(sink.bytes_written(), 10);
    }
}
