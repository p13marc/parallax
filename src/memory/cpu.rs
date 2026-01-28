//! Unified CPU memory segment using Linux memfd.
//!
//! This module provides `CpuSegment`, the primary memory backend for Parallax.
//! All CPU memory is backed by `memfd_create`, which has zero overhead vs malloc
//! but is always shareable via fd passing.
//!
//! # Design Rationale
//!
//! Previously, Parallax had separate `CpuSegment` (malloc-backed) and
//! a separate heap-backed segment. This distinction was unnecessary:
//!
//! - `memfd_create` + `MAP_SHARED` has zero overhead vs `malloc`
//! - Every buffer is automatically shareable via fd passing
//! - No conversion needed before IPC
//! - Cross-process = same physical pages (true zero-copy)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::{CpuSegment, MemorySegment};
//!
//! // Allocate 1MB of CPU memory (works like malloc, but shareable)
//! let segment = CpuSegment::new(1024 * 1024)?;
//!
//! // Write some data
//! segment.as_mut_slice().unwrap()[..5].copy_from_slice(b"hello");
//!
//! // Get fd for IPC (always available - no conversion needed!)
//! let fd = segment.fd();
//! // Send fd over Unix socket via SCM_RIGHTS...
//!
//! // In another process:
//! let received = CpuSegment::from_fd(received_fd)?;
//! // Same physical memory - true zero-copy!
//! ```

use super::{IpcHandle, MemorySegment, MemoryType};
use crate::error::{Error, Result};
use rustix::fd::{AsFd, BorrowedFd, OwnedFd};
use rustix::mm::{MapFlags, ProtFlags};
use std::ffi::CString;
use std::os::unix::io::{AsRawFd, RawFd};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};

/// Global counter for generating unique segment IDs.
static SEGMENT_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Generate a unique segment ID.
fn next_segment_id() -> u64 {
    SEGMENT_ID_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Unified CPU memory segment - always memfd-backed, always IPC-ready.
///
/// This is the primary memory backend for Parallax. It replaces both
/// separate heap and shared memory types with a single type that's
/// always shareable across processes with zero overhead.
///
/// # Memory Model
///
/// - Backed by `memfd_create` (anonymous file in memory)
/// - Mapped with `MAP_SHARED` (changes visible across mappings)
/// - File descriptor can be sent via `SCM_RIGHTS` for cross-process sharing
/// - Same physical pages in all processes (true zero-copy)
///
/// # Performance
///
/// Despite using `memfd_create` instead of `malloc`, there's no performance
/// penalty for single-process use:
///
/// - Same page fault behavior as anonymous mmap
/// - No filesystem overhead (memfd is purely in-memory)
/// - No IPC overhead until fd is actually shared
///
/// # Safety
///
/// The segment is `Send + Sync` because:
/// - The underlying memory can be safely accessed from any thread
/// - The fd is reference-counted by the kernel
/// - Concurrent access requires external synchronization (same as any shared memory)
pub struct CpuSegment {
    /// The memfd file descriptor.
    fd: OwnedFd,
    /// Pointer to the mmap'd region.
    ptr: NonNull<u8>,
    /// Size of the segment in bytes.
    len: usize,
    /// Unique ID for this segment (for cross-process identification).
    id: u64,
    /// Optional debug name.
    name: Option<String>,
}

impl CpuSegment {
    /// Allocate new CPU memory.
    ///
    /// This works like `malloc` but the memory is always shareable via fd.
    ///
    /// # Arguments
    ///
    /// * `size` - Size in bytes. Must be greater than 0.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `size` is 0
    /// - `memfd_create` fails (unlikely, kernel resource exhaustion)
    /// - `ftruncate` fails (unlikely)
    /// - `mmap` fails (unlikely, address space exhaustion)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let segment = CpuSegment::new(4096)?;
    /// assert_eq!(segment.len(), 4096);
    /// ```
    pub fn new(size: usize) -> Result<Self> {
        Self::with_name("parallax", size)
    }

    /// Allocate new CPU memory with a debug name.
    ///
    /// The name is visible in `/proc/self/fd/` and helps with debugging.
    ///
    /// # Arguments
    ///
    /// * `name` - Debug name (will be prefixed with "parallax:")
    /// * `size` - Size in bytes. Must be greater than 0.
    pub fn with_name(name: &str, size: usize) -> Result<Self> {
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

        // Memory-map the region with MAP_SHARED
        // MAP_SHARED is crucial: it means other processes mapping the same fd
        // see the same physical pages (true zero-copy)
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
            id: next_segment_id(),
            name: Some(name.to_string()),
        })
    }

    /// Reconstruct a segment from a received file descriptor (safe version).
    ///
    /// Use this in the receiving process after getting the fd via `SCM_RIGHTS`.
    /// The resulting segment shares the same physical memory as the sender.
    ///
    /// This method queries the actual file size from the fd and validates it,
    /// making it safe to call with any file descriptor.
    ///
    /// # Arguments
    ///
    /// * `fd` - File descriptor received from another process (should be a memfd)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // In receiving process, after getting fd via SCM_RIGHTS:
    /// let segment = CpuSegment::from_fd(received_fd)?;
    /// // segment now shares memory with sender
    /// ```
    pub fn from_fd(fd: OwnedFd) -> Result<Self> {
        // Query the actual size from the fd
        let stat = rustix::fs::fstat(&fd)?;
        let size = stat.st_size as usize;

        if size == 0 {
            return Err(Error::AllocationFailed("memfd has zero size".into()));
        }

        // Map the received fd
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
            id: next_segment_id(),
            name: None,
        })
    }

    /// Reconstruct a segment from a received file descriptor with explicit size.
    ///
    /// Use this when you already know the size and want to skip the fstat call,
    /// or when you want to map only a portion of the memfd.
    ///
    /// # Arguments
    ///
    /// * `fd` - File descriptor received from another process
    /// * `size` - Size of the memory region to map
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `fd` is a valid memfd file descriptor
    /// - `size` does not exceed the actual memfd size
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // In receiving process, when you know the size:
    /// let segment = unsafe { CpuSegment::from_fd_with_size(received_fd, known_size)? };
    /// ```
    pub unsafe fn from_fd_with_size(fd: OwnedFd, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::AllocationFailed(
                "size must be greater than 0".into(),
            ));
        }

        // Map the received fd
        // SAFETY: Caller guarantees fd is a valid memfd and size is correct
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
            id: next_segment_id(),
            name: None,
        })
    }

    /// Reconstruct from a raw file descriptor (safe version).
    ///
    /// This duplicates the fd so the segment owns its own reference.
    /// The size is queried from the fd automatically.
    ///
    /// # Safety
    ///
    /// The caller must ensure `fd` is a valid file descriptor.
    pub unsafe fn from_raw_fd(fd: RawFd) -> Result<Self> {
        // SAFETY: Caller guarantees fd is valid
        let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
        let owned = rustix::io::fcntl_dupfd_cloexec(borrowed, 0)?;
        Self::from_fd(owned)
    }

    /// Reconstruct from a raw file descriptor with explicit size.
    ///
    /// This duplicates the fd so the segment owns its own reference.
    ///
    /// # Safety
    ///
    /// The caller must ensure:
    /// - `fd` is a valid file descriptor
    /// - `size` does not exceed the actual file size
    pub unsafe fn from_raw_fd_with_size(fd: RawFd, size: usize) -> Result<Self> {
        // SAFETY: Caller guarantees fd is valid
        let borrowed = unsafe { BorrowedFd::borrow_raw(fd) };
        let owned = rustix::io::fcntl_dupfd_cloexec(borrowed, 0)?;
        // SAFETY: We're forwarding the caller's safety guarantees
        unsafe { Self::from_fd_with_size(owned, size) }
    }

    /// Get the file descriptor for IPC.
    ///
    /// Use this to send the fd to another process via `SCM_RIGHTS`.
    /// The fd is always available (no conversion needed).
    #[inline]
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }

    /// Get the raw file descriptor.
    #[inline]
    pub fn raw_fd(&self) -> RawFd {
        self.fd.as_raw_fd()
    }

    /// Get the unique segment ID.
    ///
    /// This is useful for cross-process identification when using arenas.
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Get the debug name.
    #[inline]
    pub fn name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    /// Get a mutable slice of the memory.
    ///
    /// # Safety
    ///
    /// The caller must ensure exclusive access to this memory region.
    /// If the memory is shared with other processes, external synchronization
    /// is required.
    #[inline]
    #[allow(clippy::mut_from_ref)] // Interior mutability via mmap is intentional
    pub fn as_mut_slice(&self) -> &mut [u8] {
        // SAFETY: We own the memory and the caller ensures exclusive access
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }

    /// Pre-fault the memory to avoid page faults during use.
    ///
    /// This touches every page to ensure physical memory is allocated.
    /// Useful for latency-sensitive applications.
    pub fn prefault(&self) {
        let page_size = 4096; // Standard page size
        let ptr = self.ptr.as_ptr();
        for offset in (0..self.len).step_by(page_size) {
            unsafe {
                // Volatile read to prevent optimization
                std::ptr::read_volatile(ptr.add(offset));
            }
        }
    }

    /// Duplicate the segment (creates a new mapping of the same memory).
    ///
    /// The new segment shares the same physical memory as the original.
    /// This is useful when you need multiple handles to the same memory.
    pub fn try_clone(&self) -> Result<Self> {
        let new_fd = rustix::io::fcntl_dupfd_cloexec(&self.fd, 0)?;
        Self::from_fd(new_fd)
    }
}

impl MemorySegment for CpuSegment {
    #[inline]
    fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    #[inline]
    fn as_mut_ptr(&self) -> Option<*mut u8> {
        Some(self.ptr.as_ptr())
    }

    #[inline]
    fn len(&self) -> usize {
        self.len
    }

    #[inline]
    fn memory_type(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn ipc_handle(&self) -> Option<IpcHandle> {
        Some(IpcHandle::Fd {
            fd: self.fd.as_raw_fd(),
            size: self.len,
        })
    }
}

impl Drop for CpuSegment {
    fn drop(&mut self) {
        // Unmap the memory region
        unsafe {
            let _ = rustix::mm::munmap(self.ptr.as_ptr().cast(), self.len);
        }
        // fd is automatically closed when OwnedFd is dropped
    }
}

// SAFETY: CpuSegment is Send + Sync because:
// - The memory region can be accessed from any thread
// - The fd is reference-counted by the kernel
// - We don't hold any thread-local state
// - Concurrent access requires external synchronization (same as any shared memory)
unsafe impl Send for CpuSegment {}
unsafe impl Sync for CpuSegment {}

impl AsFd for CpuSegment {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_segment_creation() {
        let segment = CpuSegment::new(4096).unwrap();
        assert_eq!(segment.len(), 4096);
        assert_eq!(segment.memory_type(), MemoryType::Cpu);
        assert!(segment.ipc_handle().is_some());
        assert!(segment.id() > 0);
    }

    #[test]
    fn test_cpu_segment_with_name() {
        let segment = CpuSegment::with_name("test-buffer", 4096).unwrap();
        assert_eq!(segment.name(), Some("test-buffer"));
        assert_eq!(segment.len(), 4096);
    }

    #[test]
    fn test_cpu_segment_zero_size_fails() {
        let result = CpuSegment::new(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cpu_segment_read_write() {
        let segment = CpuSegment::new(4096).unwrap();

        // Write some data
        let slice = segment.as_mut_slice();
        slice[0] = 42;
        slice[1] = 43;
        slice[4095] = 99;

        // Read it back
        unsafe {
            let read_slice = segment.as_slice();
            assert_eq!(read_slice[0], 42);
            assert_eq!(read_slice[1], 43);
            assert_eq!(read_slice[4095], 99);
        }
    }

    #[test]
    fn test_cpu_segment_prefault() {
        let segment = CpuSegment::new(1024 * 1024).unwrap();
        segment.prefault(); // Should not panic
    }

    #[test]
    fn test_cpu_segment_from_fd() {
        // Create a segment
        let original = CpuSegment::new(4096).unwrap();

        // Write some data
        original.as_mut_slice()[0] = 123;
        original.as_mut_slice()[100] = 234;

        // Duplicate the fd (simulating receiving it from another process)
        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&original.fd, 0).unwrap();

        // Open from the duplicated fd
        let reopened = CpuSegment::from_fd(dup_fd).unwrap();

        // Verify we can read the same data (same physical memory!)
        unsafe {
            let slice = reopened.as_slice();
            assert_eq!(slice[0], 123);
            assert_eq!(slice[100], 234);
        }
    }

    #[test]
    fn test_cpu_segment_modifications_visible() {
        // Create a segment
        let segment1 = CpuSegment::new(4096).unwrap();

        // Duplicate fd to simulate another process
        let dup_fd = rustix::io::fcntl_dupfd_cloexec(&segment1.fd, 0).unwrap();
        let segment2 = CpuSegment::from_fd(dup_fd).unwrap();

        // Write via segment1
        segment1.as_mut_slice()[0] = 77;

        // Should be visible via segment2 (same physical memory)
        unsafe {
            assert_eq!(segment2.as_slice()[0], 77);
        }

        // Write via segment2
        segment2.as_mut_slice()[100] = 88;

        // Should be visible via segment1
        unsafe {
            assert_eq!(segment1.as_slice()[100], 88);
        }
    }

    #[test]
    fn test_cpu_segment_try_clone() {
        let original = CpuSegment::new(4096).unwrap();
        original.as_mut_slice()[0] = 42;

        let cloned = original.try_clone().unwrap();

        // Different segment IDs
        assert_ne!(original.id(), cloned.id());

        // But same memory content
        unsafe {
            assert_eq!(cloned.as_slice()[0], 42);
        }

        // Modifications in clone visible in original
        cloned.as_mut_slice()[1] = 99;
        unsafe {
            assert_eq!(original.as_slice()[1], 99);
        }
    }

    #[test]
    fn test_cpu_segment_unique_ids() {
        let seg1 = CpuSegment::new(4096).unwrap();
        let seg2 = CpuSegment::new(4096).unwrap();
        let seg3 = CpuSegment::new(4096).unwrap();

        assert_ne!(seg1.id(), seg2.id());
        assert_ne!(seg2.id(), seg3.id());
        assert_ne!(seg1.id(), seg3.id());
    }

    #[test]
    fn test_cpu_segment_large_allocation() {
        // 16 MB allocation
        let segment = CpuSegment::new(16 * 1024 * 1024).unwrap();
        assert_eq!(segment.len(), 16 * 1024 * 1024);

        // Write at the end
        segment.as_mut_slice()[16 * 1024 * 1024 - 1] = 0xFF;
        unsafe {
            assert_eq!(segment.as_slice()[16 * 1024 * 1024 - 1], 0xFF);
        }
    }
}
