//! DMA-BUF memory segment for GPU-importable buffers.
//!
//! DMA-BUF is the Linux kernel's buffer sharing mechanism, enabling:
//! - Zero-copy sharing between processes (via SCM_RIGHTS fd passing)
//! - Direct GPU import (Vulkan, VA-API, etc.)
//! - Efficient camera/video capture pipelines
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::DmaBufSegment;
//!
//! // From V4L2 VIDIOC_EXPBUF
//! let dmabuf_fd: OwnedFd = v4l2_export_buffer(device, buffer_index)?;
//! let segment = DmaBufSegment::from_fd(dmabuf_fd, buffer_size)?;
//!
//! // CPU access via mmap
//! let data = segment.as_slice();
//!
//! // IPC: send fd to another process
//! send_fds(&socket, &[segment.as_fd()], &[])?;
//! ```

use crate::error::Result;
use crate::memory::{IpcHandle, MemorySegment, MemoryType};
use rustix::fd::{AsFd, AsRawFd, BorrowedFd, OwnedFd};
use rustix::mm::{MapFlags, ProtFlags};
use std::ptr::NonNull;

/// A memory segment backed by a DMA-BUF file descriptor.
///
/// DMA-BUF is the Linux kernel's buffer sharing mechanism. This segment type
/// wraps a DMA-BUF fd and provides CPU access via mmap.
///
/// # Use Cases
///
/// - V4L2 camera capture with `VIDIOC_EXPBUF`
/// - libcamera frame buffers
/// - DRM/KMS buffer export
/// - GPU driver buffer export
/// - Zero-copy video pipelines
///
/// # Memory Type
///
/// Reports `MemoryType::DmaBuf`, which:
/// - Supports IPC via fd passing (SCM_RIGHTS)
/// - Does NOT support network transfer (fd is local)
/// - Is CPU-accessible (via mmap)
/// - Can be imported by GPU drivers
///
/// # Thread Safety
///
/// `DmaBufSegment` is `Send + Sync`. The underlying fd can be used from any
/// thread, and concurrent reads are safe. Mutable access requires `&mut self`.
pub struct DmaBufSegment {
    /// The DMA-BUF file descriptor.
    fd: OwnedFd,
    /// Memory-mapped pointer for CPU access.
    ptr: NonNull<u8>,
    /// Size in bytes.
    len: usize,
    /// Whether this segment is read-only.
    read_only: bool,
}

impl DmaBufSegment {
    /// Create a DMA-BUF segment from an existing file descriptor.
    ///
    /// The fd is typically obtained from:
    /// - V4L2 `VIDIOC_EXPBUF` ioctl
    /// - libcamera frame buffer
    /// - DRM/KMS buffer export
    /// - GPU driver export
    ///
    /// The segment will mmap the fd for CPU access with read/write permissions.
    ///
    /// # Arguments
    ///
    /// * `fd` - The DMA-BUF file descriptor (ownership transferred)
    /// * `len` - Size of the buffer in bytes
    ///
    /// # Errors
    ///
    /// Returns an error if mmap fails (e.g., invalid fd, insufficient permissions).
    pub fn from_fd(fd: OwnedFd, len: usize) -> Result<Self> {
        let ptr = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                len,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &fd,
                0,
            )
            .map_err(|e| {
                crate::error::Error::InvalidSegment(format!("mmap DMA-BUF failed: {}", e))
            })?
        };

        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| crate::error::Error::InvalidSegment("mmap returned null".into()))?;

        Ok(Self {
            fd,
            ptr,
            len,
            read_only: false,
        })
    }

    /// Create a read-only DMA-BUF segment.
    ///
    /// Use this when the buffer should not be modified (e.g., camera output
    /// that will be consumed by an encoder).
    ///
    /// # Arguments
    ///
    /// * `fd` - The DMA-BUF file descriptor (ownership transferred)
    /// * `len` - Size of the buffer in bytes
    pub fn from_fd_readonly(fd: OwnedFd, len: usize) -> Result<Self> {
        let ptr = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                len,
                ProtFlags::READ,
                MapFlags::SHARED,
                &fd,
                0,
            )
            .map_err(|e| {
                crate::error::Error::InvalidSegment(format!("mmap DMA-BUF failed: {}", e))
            })?
        };

        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| crate::error::Error::InvalidSegment("mmap returned null".into()))?;

        Ok(Self {
            fd,
            ptr,
            len,
            read_only: true,
        })
    }

    /// Get a borrowed reference to the underlying file descriptor.
    ///
    /// Use this for:
    /// - GPU import operations
    /// - IPC fd passing via `send_fds()`
    /// - Duplicating the fd with `try_clone()`
    #[inline]
    pub fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// Consume the segment and return the file descriptor.
    ///
    /// **Warning**: This unmaps the memory. The returned fd is still valid,
    /// but the CPU-accessible pointer is no longer usable.
    ///
    /// Use this when transferring ownership of the fd to another system
    /// (e.g., GPU import that takes ownership).
    pub fn into_fd(self) -> OwnedFd {
        // Unmap first, then extract fd
        unsafe {
            let _ = rustix::mm::munmap(self.ptr.as_ptr().cast(), self.len);
        }
        // Use ManuallyDrop to prevent Drop from running (which would double-unmap)
        let this = std::mem::ManuallyDrop::new(self);
        // SAFETY: We're consuming self and have already unmapped
        unsafe { std::ptr::read(&this.fd) }
    }

    /// Get the segment as a byte slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr is valid for len bytes, properly aligned for u8
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    /// Get the segment as a mutable byte slice.
    ///
    /// Returns `None` if this segment is read-only.
    #[inline]
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]> {
        if self.read_only {
            None
        } else {
            // SAFETY: ptr is valid for len bytes, we have &mut self
            Some(unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) })
        }
    }

    /// Check if this segment is read-only.
    #[inline]
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Get the size in bytes.
    #[inline]
    pub fn size(&self) -> usize {
        self.len
    }
}

impl Drop for DmaBufSegment {
    fn drop(&mut self) {
        // Unmap before fd is closed
        unsafe {
            let _ = rustix::mm::munmap(self.ptr.as_ptr().cast(), self.len);
        }
        // fd is dropped automatically (OwnedFd)
    }
}

// SAFETY: DMA-BUF fds can be sent between threads.
// The fd itself is just a number; the kernel handles synchronization.
unsafe impl Send for DmaBufSegment {}

// SAFETY: Concurrent reads of the mmap'd region are safe.
// Mutable access requires &mut self, which Rust enforces.
unsafe impl Sync for DmaBufSegment {}

impl MemorySegment for DmaBufSegment {
    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn as_mut_ptr(&self) -> Option<*mut u8> {
        if self.read_only {
            None
        } else {
            Some(self.ptr.as_ptr())
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::DmaBuf
    }

    fn ipc_handle(&self) -> Option<IpcHandle> {
        Some(IpcHandle::Fd {
            fd: self.fd.as_raw_fd(),
            size: self.len,
        })
    }
}

impl std::fmt::Debug for DmaBufSegment {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DmaBufSegment")
            .field("fd", &self.fd.as_raw_fd())
            .field("len", &self.len)
            .field("read_only", &self.read_only)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dmabuf_from_memfd() {
        // Use memfd as a DMA-BUF-like fd for testing.
        // Real DMA-BUF requires a device driver, but memfd works for API testing.
        let fd = rustix::fs::memfd_create("test_dmabuf", rustix::fs::MemfdFlags::CLOEXEC).unwrap();

        rustix::fs::ftruncate(&fd, 4096).unwrap();

        let segment = DmaBufSegment::from_fd(fd, 4096).unwrap();

        assert_eq!(segment.len(), 4096);
        assert_eq!(segment.size(), 4096);
        assert_eq!(segment.memory_type(), MemoryType::DmaBuf);
        assert!(segment.ipc_handle().is_some());
        assert!(!segment.is_read_only());
    }

    #[test]
    fn test_dmabuf_read_write() {
        let fd = rustix::fs::memfd_create("test_rw", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        rustix::fs::ftruncate(&fd, 1024).unwrap();

        let mut segment = DmaBufSegment::from_fd(fd, 1024).unwrap();

        // Write some data using the DmaBufSegment's own method (explicit call)
        let data = b"Hello, DMA-BUF!";
        DmaBufSegment::as_mut_slice(&mut segment).unwrap()[..data.len()].copy_from_slice(data);

        // Read it back using the DmaBufSegment's own method (explicit call)
        assert_eq!(&DmaBufSegment::as_slice(&segment)[..data.len()], data);
    }

    #[test]
    fn test_dmabuf_readonly() {
        let fd = rustix::fs::memfd_create("test_ro", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        rustix::fs::ftruncate(&fd, 512).unwrap();

        let segment = DmaBufSegment::from_fd_readonly(fd, 512).unwrap();

        assert!(segment.is_read_only());
        // Use the MemorySegment trait's as_mut_ptr which returns Option
        assert!(segment.as_mut_ptr().is_none());

        // Reading should still work
        let _ = segment.as_slice();
    }

    #[test]
    fn test_dmabuf_ipc_handle() {
        let fd = rustix::fs::memfd_create("test_ipc", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        rustix::fs::ftruncate(&fd, 2048).unwrap();

        let segment = DmaBufSegment::from_fd(fd, 2048).unwrap();
        let handle = segment.ipc_handle().unwrap();

        match handle {
            IpcHandle::Fd { fd: _, size } => {
                assert_eq!(size, 2048);
            }
            _ => panic!("Expected IpcHandle::Fd"),
        }
    }

    #[test]
    fn test_dmabuf_into_fd() {
        let fd = rustix::fs::memfd_create("test_into", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        let raw_fd = fd.as_raw_fd();
        rustix::fs::ftruncate(&fd, 256).unwrap();

        let segment = DmaBufSegment::from_fd(fd, 256).unwrap();

        // Consume and get fd back
        let recovered_fd = segment.into_fd();
        assert_eq!(recovered_fd.as_raw_fd(), raw_fd);

        // fd should still be valid - we can fstat it
        let stat = rustix::fs::fstat(&recovered_fd).unwrap();
        assert_eq!(stat.st_size, 256);
    }

    #[test]
    fn test_dmabuf_debug() {
        let fd = rustix::fs::memfd_create("test_debug", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        rustix::fs::ftruncate(&fd, 128).unwrap();

        let segment = DmaBufSegment::from_fd(fd, 128).unwrap();
        let debug_str = format!("{:?}", segment);

        assert!(debug_str.contains("DmaBufSegment"));
        assert!(debug_str.contains("len: 128"));
    }
}
