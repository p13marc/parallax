//! Memory-mapped file segment for persistent buffers.
//!
//! This module provides a memory segment backed by a file on disk.
//! Changes to the memory are persisted to the file, making this
//! useful for:
//!
//! - Persistent buffers that survive process restarts
//! - Memory-mapped I/O for large files
//! - Sharing data between processes via the filesystem
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::MappedFileSegment;
//!
//! // Create a new mapped file
//! let segment = MappedFileSegment::create("/tmp/buffer.dat", 1024 * 1024)?;
//!
//! // Write some data
//! segment.as_mut_slice()[..5].copy_from_slice(b"hello");
//!
//! // Sync to disk
//! segment.sync()?;
//!
//! // Later, open the same file
//! let segment2 = MappedFileSegment::open("/tmp/buffer.dat")?;
//! assert_eq!(&segment2.as_slice()[..5], b"hello");
//! ```

use super::{IpcHandle, MemorySegment, MemoryType};
use crate::error::{Error, Result};
use rustix::fd::OwnedFd;
use rustix::mm::{MapFlags, ProtFlags};
use std::path::{Path, PathBuf};
use std::ptr::NonNull;

/// A memory segment backed by a file on disk.
///
/// The file is memory-mapped, providing zero-copy access to file contents.
/// Changes are automatically synced to disk (though you can call `sync()`
/// to force immediate persistence).
pub struct MappedFileSegment {
    /// File descriptor.
    fd: OwnedFd,
    /// Pointer to the mmap'd region.
    ptr: NonNull<u8>,
    /// Size of the segment.
    len: usize,
    /// Path to the file.
    path: PathBuf,
    /// Whether the mapping is read-only.
    read_only: bool,
}

impl MappedFileSegment {
    /// Create a new mapped file segment.
    ///
    /// Creates a new file (or truncates an existing one) and maps it.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the file to create.
    /// * `size` - Size of the file in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if file creation, truncation, or mapping fails.
    pub fn create<P: AsRef<Path>>(path: P, size: usize) -> Result<Self> {
        let path = path.as_ref();

        if size == 0 {
            return Err(Error::AllocationFailed(
                "size must be greater than 0".into(),
            ));
        }

        // Open/create the file
        use rustix::fs::{Mode, OFlags};
        let fd = rustix::fs::open(
            path,
            OFlags::RDWR | OFlags::CREATE | OFlags::TRUNC,
            Mode::from_raw_mode(0o644),
        )?;

        // Set the file size
        rustix::fs::ftruncate(&fd, size as u64)?;

        // Memory-map the file
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
            path: path.to_path_buf(),
            read_only: false,
        })
    }

    /// Open an existing file as a mapped segment.
    ///
    /// The file is mapped read-write by default.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the existing file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file doesn't exist or mapping fails.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_mode(path, false)
    }

    /// Open an existing file as a read-only mapped segment.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the existing file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file doesn't exist or mapping fails.
    pub fn open_readonly<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::open_with_mode(path, true)
    }

    fn open_with_mode<P: AsRef<Path>>(path: P, read_only: bool) -> Result<Self> {
        let path = path.as_ref();

        // Open the file
        use rustix::fs::OFlags;
        let flags = if read_only {
            OFlags::RDONLY
        } else {
            OFlags::RDWR
        };
        let fd = rustix::fs::open(path, flags, rustix::fs::Mode::empty())?;

        // Get file size
        let stat = rustix::fs::fstat(&fd)?;
        let size = stat.st_size as usize;

        if size == 0 {
            return Err(Error::AllocationFailed("file is empty".into()));
        }

        // Memory-map the file
        let prot = if read_only {
            ProtFlags::READ
        } else {
            ProtFlags::READ | ProtFlags::WRITE
        };

        let ptr = unsafe {
            rustix::mm::mmap(std::ptr::null_mut(), size, prot, MapFlags::SHARED, &fd, 0)?
        };

        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;

        Ok(Self {
            fd,
            ptr,
            len: size,
            path: path.to_path_buf(),
            read_only,
        })
    }

    /// Resize the mapped file.
    ///
    /// This unmaps the current region, truncates the file, and remaps.
    ///
    /// # Arguments
    ///
    /// * `new_size` - New size in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the segment is read-only or resizing fails.
    pub fn resize(&mut self, new_size: usize) -> Result<()> {
        if self.read_only {
            return Err(Error::InvalidSegment("segment is read-only".into()));
        }

        if new_size == 0 {
            return Err(Error::AllocationFailed(
                "size must be greater than 0".into(),
            ));
        }

        // Unmap current region
        unsafe {
            rustix::mm::munmap(self.ptr.as_ptr().cast(), self.len)?;
        }

        // Resize the file
        rustix::fs::ftruncate(&self.fd, new_size as u64)?;

        // Remap
        let ptr = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                new_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &self.fd,
                0,
            )?
        };

        self.ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;
        self.len = new_size;

        Ok(())
    }

    /// Sync changes to disk.
    ///
    /// This calls `msync` to ensure all modifications are written to the file.
    pub fn sync(&self) -> Result<()> {
        unsafe {
            rustix::mm::msync(
                self.ptr.as_ptr().cast(),
                self.len,
                rustix::mm::MsyncFlags::SYNC,
            )?;
        }
        Ok(())
    }

    /// Sync changes asynchronously.
    ///
    /// This schedules a sync but doesn't wait for it to complete.
    pub fn sync_async(&self) -> Result<()> {
        unsafe {
            rustix::mm::msync(
                self.ptr.as_ptr().cast(),
                self.len,
                rustix::mm::MsyncFlags::ASYNC,
            )?;
        }
        Ok(())
    }

    /// Get the path to the backing file.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Check if the segment is read-only.
    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    /// Pre-fault all pages to avoid page faults during use.
    pub fn prefault(&self) {
        let page_size = 4096;
        let ptr = self.ptr.as_ptr();
        for offset in (0..self.len).step_by(page_size) {
            unsafe {
                std::ptr::read_volatile(ptr.add(offset));
            }
        }
    }
}

impl MemorySegment for MappedFileSegment {
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
        MemoryType::MappedFile
    }

    fn ipc_handle(&self) -> Option<IpcHandle> {
        // Return the file path for named sharing
        Some(IpcHandle::Named {
            name: self.path.to_string_lossy().into_owned(),
            size: self.len,
        })
    }
}

impl Drop for MappedFileSegment {
    fn drop(&mut self) {
        // Sync before unmapping to ensure data is persisted
        let _ = self.sync();

        unsafe {
            let _ = rustix::mm::munmap(self.ptr.as_ptr().cast(), self.len);
        }
        // fd is automatically closed when OwnedFd is dropped
    }
}

// SAFETY: MappedFileSegment is Send + Sync because:
// - File mappings can be safely accessed from any thread
// - The kernel handles synchronization for SHARED mappings
unsafe impl Send for MappedFileSegment {}
unsafe impl Sync for MappedFileSegment {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("parallax-test-{}-{}", name, std::process::id()))
    }

    #[test]
    fn test_create_and_open() {
        let path = temp_path("create-open");

        // Create a new mapped file
        {
            let segment = MappedFileSegment::create(&path, 4096).unwrap();
            assert_eq!(segment.len(), 4096);
            assert!(!segment.is_read_only());
            assert_eq!(segment.memory_type(), MemoryType::MappedFile);

            // Write some data
            let ptr = segment.as_mut_ptr().unwrap();
            unsafe {
                std::ptr::copy_nonoverlapping(b"hello".as_ptr(), ptr, 5);
            }

            segment.sync().unwrap();
        }

        // Reopen and verify
        {
            let segment = MappedFileSegment::open(&path).unwrap();
            assert_eq!(segment.len(), 4096);

            let data = unsafe { std::slice::from_raw_parts(segment.as_ptr(), 5) };
            assert_eq!(data, b"hello");
        }

        // Cleanup
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_readonly() {
        let path = temp_path("readonly");

        // Create file
        {
            let segment = MappedFileSegment::create(&path, 1024).unwrap();
            let ptr = segment.as_mut_ptr().unwrap();
            unsafe {
                *ptr = 42;
            }
        }

        // Open read-only
        {
            let segment = MappedFileSegment::open_readonly(&path).unwrap();
            assert!(segment.is_read_only());
            assert!(segment.as_mut_ptr().is_none());

            // Can still read
            unsafe {
                assert_eq!(*segment.as_ptr(), 42);
            }
        }

        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_resize() {
        let path = temp_path("resize");

        let mut segment = MappedFileSegment::create(&path, 1024).unwrap();
        assert_eq!(segment.len(), 1024);

        // Write marker
        unsafe {
            *segment.as_mut_ptr().unwrap() = 99;
        }

        // Resize larger
        segment.resize(4096).unwrap();
        assert_eq!(segment.len(), 4096);

        // Data should be preserved
        unsafe {
            assert_eq!(*segment.as_ptr(), 99);
        }

        drop(segment);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_ipc_handle() {
        let path = temp_path("ipc-handle");

        let segment = MappedFileSegment::create(&path, 1024).unwrap();
        let handle = segment.ipc_handle().unwrap();

        match handle {
            IpcHandle::Named { name, size } => {
                assert!(name.contains("ipc-handle"));
                assert_eq!(size, 1024);
            }
            _ => panic!("expected Named handle"),
        }

        drop(segment);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_zero_size_fails() {
        let path = temp_path("zero-size");
        let result = MappedFileSegment::create(&path, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_prefault() {
        let path = temp_path("prefault");
        let segment = MappedFileSegment::create(&path, 4096).unwrap();
        segment.prefault(); // Should not panic
        drop(segment);
        fs::remove_file(&path).unwrap();
    }

    #[test]
    fn test_sync() {
        let path = temp_path("sync");
        let segment = MappedFileSegment::create(&path, 4096).unwrap();

        unsafe {
            *segment.as_mut_ptr().unwrap() = 123;
        }

        segment.sync().unwrap();
        segment.sync_async().unwrap();

        drop(segment);
        fs::remove_file(&path).unwrap();
    }
}
