//! File descriptor source and sink elements.
//!
//! Read from and write to raw file descriptors.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Sink, Source};
use crate::error::{Error, Result};
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::io::{AsRawFd, FromRawFd, RawFd};
use std::sync::Arc;

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
    buffer_size: usize,
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
            buffer_size: 64 * 1024,
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
            buffer_size: 64 * 1024,
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

    /// Set the buffer size for reads.
    pub fn with_buffer_size(mut self, size: usize) -> Self {
        self.buffer_size = size;
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
    fn produce(&mut self) -> Result<Option<Buffer>> {
        let segment = Arc::new(HeapSegment::new(self.buffer_size)?);
        let ptr = segment
            .as_mut_ptr()
            .ok_or_else(|| Error::Element("cannot get mutable pointer".into()))?;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr, self.buffer_size) };

        match rustix::io::read(&self.fd, slice) {
            Ok(0) => {
                // EOF
                Ok(None)
            }
            Ok(n) => {
                self.bytes_read += n as u64;
                let seq = self.sequence;
                self.sequence += 1;

                let handle = MemoryHandle::from_segment_with_len(segment, n);
                let metadata = Metadata::with_sequence(seq);

                Ok(Some(Buffer::new(handle, metadata)))
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
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let data = buffer.as_bytes();
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

    #[test]
    fn test_fdsrc_from_pipe() {
        // Create a pipe using rustix
        let (read_fd, write_fd) = rustix::pipe::pipe().unwrap();

        // Write some data
        rustix::io::write(&write_fd, b"hello from pipe").unwrap();
        drop(write_fd);

        // Read using FdSrc
        let mut src = FdSrc::from_owned(read_fd).with_buffer_size(1024);

        let buf = src.produce().unwrap().unwrap();
        assert_eq!(buf.as_bytes(), b"hello from pipe");

        // Should be EOF
        let buf = src.produce().unwrap();
        assert!(buf.is_none());

        assert_eq!(src.bytes_read(), 15);
    }

    #[test]
    fn test_fdsink_to_pipe() {
        // Create a pipe using rustix
        let (read_fd, write_fd) = rustix::pipe::pipe().unwrap();

        // Write using FdSink
        let mut sink = FdSink::from_owned(write_fd);

        let segment = Arc::new(HeapSegment::new(11).unwrap());
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(b"hello world".as_ptr(), ptr, 11);
        }
        let handle = MemoryHandle::from_segment_with_len(segment, 11);
        let buffer = Buffer::new(handle, Metadata::default());

        sink.consume(buffer).unwrap();
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
    fn test_fdsrc_buffer_size() {
        let (read_fd, _write_fd) = rustix::pipe::pipe().unwrap();
        let src = FdSrc::from_owned(read_fd).with_buffer_size(256);
        assert_eq!(src.buffer_size, 256);
    }
}
