//! Shared memory segment using Linux memfd.
//!
//! This module provides a memory segment backed by anonymous shared memory
//! created via `memfd_create`. This allows zero-copy sharing between processes
//! by passing the file descriptor over a Unix socket.

use super::{IpcHandle, MemorySegment, MemoryType};
use crate::error::{Error, Result};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::mm::{MapFlags, ProtFlags};
use std::ffi::CString;
use std::os::unix::io::{AsRawFd, RawFd};
use std::ptr::NonNull;

/// A memory segment backed by Linux memfd (anonymous shared memory).
///
/// This is the primary memory backend for multi-process pipelines. The segment
/// can be shared with other processes by passing the file descriptor via
/// `SCM_RIGHTS` over a Unix socket.
///
/// # Features
///
/// - Anonymous: No filesystem visibility (unlike `shm_open`)
/// - Auto-cleanup: Kernel reclaims memory when all references are closed
/// - Sealable: Can be made immutable with `F_SEAL_*` (future feature)
///
/// # Example
///
/// ```rust,ignore
/// use parallax::memory::{SharedMemorySegment, MemorySegment};
///
/// // Create a 1MB shared memory segment
/// let segment = SharedMemorySegment::new("my-buffer", 1024 * 1024)?;
///
/// // Get IPC handle to share with another process
/// let handle = segment.ipc_handle().unwrap();
/// // Send handle.fd over Unix socket...
/// ```
pub struct SharedMemorySegment {
    /// The memfd file descriptor.
    fd: OwnedFd,
    /// Pointer to the mmap'd region.
    ptr: NonNull<u8>,
    /// Size of the segment.
    len: usize,
    /// Optional name (for debugging).
    name: Option<String>,
}

impl SharedMemorySegment {
    /// Create a new shared memory segment.
    ///
    /// # Arguments
    ///
    /// * `name` - Debug name for the segment (visible in `/proc/self/fd/`).
    /// * `size` - Size in bytes. Must be greater than 0.
    ///
    /// # Errors
    ///
    /// Returns an error if `memfd_create`, `ftruncate`, or `mmap` fails.
    pub fn new(name: &str, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::AllocationFailed(
                "size must be greater than 0".into(),
            ));
        }

        // Create anonymous memfd
        let cname = CString::new(name).map_err(|e| Error::AllocationFailed(e.to_string()))?;
        let fd = rustix::fs::memfd_create(&cname, rustix::fs::MemfdFlags::CLOEXEC)?;

        // Set the size
        rustix::fs::ftruncate(&fd, size as u64)?;

        // Memory-map the region
        let ptr = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &fd,
                0,
            )?
        };

        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;

        Ok(Self {
            fd,
            ptr,
            len: size,
            name: Some(name.to_string()),
        })
    }

    /// Open an existing shared memory segment from a file descriptor.
    ///
    /// This is used by the receiving process after getting the fd via `SCM_RIGHTS`.
    ///
    /// # Arguments
    ///
    /// * `fd` - File descriptor of the memfd.
    /// * `size` - Expected size of the segment.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `fd` is a valid memfd and that `size`
    /// matches the actual size of the memfd.
    pub unsafe fn from_fd(fd: OwnedFd, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::AllocationFailed(
                "size must be greater than 0".into(),
            ));
        }

        // Memory-map the region
        let ptr = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &fd,
                0,
            )?
        };

        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;

        Ok(Self {
            fd,
            ptr,
            len: size,
            name: None,
        })
    }

    /// Get the file descriptor for this segment.
    ///
    /// Use this to send the fd to another process via `SCM_RIGHTS`.
    pub fn as_fd(&self) -> &OwnedFd {
        &self.fd
    }

    /// Get the raw file descriptor.
    pub fn as_raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Open an existing shared memory segment from a raw file descriptor.
    ///
    /// This creates a new mapping from an existing fd without taking ownership.
    /// The original fd remains open and the segment maintains its own reference.
    ///
    /// # Arguments
    ///
    /// * `fd` - Raw file descriptor of the memfd.
    /// * `size` - Expected size of the segment.
    ///
    /// # Safety
    ///
    /// The caller must ensure that `fd` is a valid memfd and that `size`
    /// matches the actual size of the memfd.
    pub unsafe fn from_raw_fd(fd: RawFd, size: usize) -> Result<Self> {
        // Duplicate the fd so we have our own reference
        let dup_fd = rustix::io::fcntl_dupfd_cloexec(unsafe { BorrowedFd::borrow_raw(fd) }, 0)?;
        unsafe { Self::from_fd(dup_fd, size) }
    }

    /// Get the debug name of this segment.
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Pre-fault the memory to avoid page faults during use.
    ///
    /// This touches every page to ensure physical memory is allocated.
    /// Useful for latency-sensitive applications.
    pub fn prefault(&self) {
        let page_size = 4096; // Could use sysconf, but 4K is standard
        let ptr = self.ptr.as_ptr();
        for offset in (0..self.len).step_by(page_size) {
            unsafe {
                // Volatile read to prevent optimization
                std::ptr::read_volatile(ptr.add(offset));
            }
        }
    }
}

impl MemorySegment for SharedMemorySegment {
    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    fn as_mut_ptr(&self) -> Option<*mut u8> {
        Some(self.ptr.as_ptr())
    }

    fn len(&self) -> usize {
        self.len
    }

    fn memory_type(&self) -> MemoryType {
        MemoryType::SharedMemory
    }

    fn ipc_handle(&self) -> Option<IpcHandle> {
        Some(IpcHandle::Fd {
            fd: self.fd.as_raw_fd(),
            size: self.len,
        })
    }
}

impl Drop for SharedMemorySegment {
    fn drop(&mut self) {
        // Unmap the memory region
        unsafe {
            let _ = rustix::mm::munmap(self.ptr.as_ptr().cast(), self.len);
        }
        // fd is automatically closed when OwnedFd is dropped
    }
}

// SAFETY: SharedMemorySegment is Send + Sync because:
// - The memory is shared and can be accessed from any thread
// - The fd is reference-counted by the kernel
// - We don't hold any thread-local state
unsafe impl Send for SharedMemorySegment {}
unsafe impl Sync for SharedMemorySegment {}

impl AsFd for SharedMemorySegment {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shared_memory_creation() {
        let segment = SharedMemorySegment::new("test-segment", 4096).unwrap();
        assert_eq!(segment.len(), 4096);
        assert_eq!(segment.memory_type(), MemoryType::SharedMemory);
        assert!(segment.ipc_handle().is_some());
        assert_eq!(segment.name(), Some("test-segment"));
    }

    #[test]
    fn test_shared_memory_zero_size_fails() {
        let result = SharedMemorySegment::new("test", 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_shared_memory_read_write() {
        let segment = SharedMemorySegment::new("test-rw", 4096).unwrap();

        // Write some data
        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::write(ptr, 42);
            std::ptr::write(ptr.add(1), 43);
            std::ptr::write(ptr.add(4095), 99);
        }

        // Read it back
        unsafe {
            let slice = segment.as_slice();
            assert_eq!(slice[0], 42);
            assert_eq!(slice[1], 43);
            assert_eq!(slice[4095], 99);
        }
    }

    #[test]
    fn test_shared_memory_prefault() {
        let segment = SharedMemorySegment::new("test-prefault", 1024 * 1024).unwrap();
        segment.prefault(); // Should not panic
    }

    #[test]
    fn test_shared_memory_from_fd() {
        // Create a segment
        let original = SharedMemorySegment::new("test-dup", 4096).unwrap();

        // Write some data
        unsafe {
            let slice = std::slice::from_raw_parts_mut(original.as_mut_ptr().unwrap(), 4096);
            slice[0] = 123;
            slice[100] = 234;
        }

        // Duplicate the fd (simulating receiving it from another process)
        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&original.fd, 0).unwrap();

        // Open from the duplicated fd
        let reopened = unsafe { SharedMemorySegment::from_fd(dup_fd, 4096).unwrap() };

        // Verify we can read the same data
        unsafe {
            let slice = reopened.as_slice();
            assert_eq!(slice[0], 123);
            assert_eq!(slice[100], 234);
        }
    }

    #[test]
    fn test_shared_memory_modifications_visible() {
        // Create a segment
        let segment1 = SharedMemorySegment::new("test-shared", 4096).unwrap();

        // Duplicate fd to simulate another process
        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&segment1.fd, 0).unwrap();
        let segment2 = unsafe { SharedMemorySegment::from_fd(dup_fd, 4096).unwrap() };

        // Write via segment1
        unsafe {
            *segment1.as_mut_ptr().unwrap() = 77;
        }

        // Should be visible via segment2
        unsafe {
            assert_eq!(*segment2.as_ptr(), 77);
        }

        // Write via segment2
        unsafe {
            *segment2.as_mut_ptr().unwrap().add(100) = 88;
        }

        // Should be visible via segment1
        unsafe {
            assert_eq!(*segment1.as_ptr().add(100), 88);
        }
    }
}
