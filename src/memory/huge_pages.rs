//! Huge page memory segment for reduced TLB misses.
//!
//! This module provides a memory segment backed by huge pages (2MB or 1GB).
//! Huge pages reduce TLB (Translation Lookaside Buffer) misses, which can
//! significantly improve performance for memory-intensive workloads.
//!
//! # Requirements
//!
//! - Linux kernel with huge page support
//! - Sufficient huge pages reserved (see `/proc/sys/vm/nr_hugepages`)
//! - Appropriate permissions (usually root or CAP_IPC_LOCK)
//!
//! # Huge Page Sizes
//!
//! - **2MB**: Standard huge pages on x86_64
//! - **1GB**: Gigantic pages (requires kernel support and explicit reservation)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::{HugePageSegment, HugePageSize};
//!
//! // Allocate 2MB huge page
//! let segment = HugePageSegment::new(HugePageSize::MB2, 2 * 1024 * 1024)?;
//!
//! // Use like any other segment
//! let ptr = segment.as_mut_ptr().unwrap();
//! ```

use super::{IpcHandle, MemorySegment, MemoryType};
use crate::error::{Error, Result};
use rustix::mm::{MapFlags, ProtFlags};
use std::ptr::NonNull;

/// Size of huge pages.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HugePageSize {
    /// 2MB huge pages (standard on x86_64).
    MB2,
    /// 1GB gigantic pages.
    GB1,
}

impl HugePageSize {
    /// Get the size in bytes.
    pub fn bytes(self) -> usize {
        match self {
            HugePageSize::MB2 => 2 * 1024 * 1024,
            HugePageSize::GB1 => 1024 * 1024 * 1024,
        }
    }

    /// Get the MAP_HUGETLB shift value for mmap.
    fn shift(self) -> u32 {
        match self {
            HugePageSize::MB2 => 21, // log2(2MB) = 21
            HugePageSize::GB1 => 30, // log2(1GB) = 30
        }
    }
}

/// A memory segment backed by huge pages.
///
/// This provides better performance for large allocations by reducing
/// TLB misses. The trade-off is that huge pages must be pre-reserved
/// at the system level.
pub struct HugePageSegment {
    /// Pointer to the mmap'd region.
    ptr: NonNull<u8>,
    /// Size of the segment (rounded up to huge page size).
    len: usize,
    /// The huge page size used.
    page_size: HugePageSize,
}

impl HugePageSegment {
    /// Allocate a new huge page segment.
    ///
    /// # Arguments
    ///
    /// * `page_size` - Size of huge pages to use (2MB or 1GB).
    /// * `size` - Minimum size in bytes. Will be rounded up to huge page boundary.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `mmap` fails (insufficient huge pages, permissions, etc.)
    /// - Size is zero
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let segment = HugePageSegment::new(HugePageSize::MB2, 4 * 1024 * 1024)?;
    /// assert_eq!(segment.len(), 4 * 1024 * 1024); // Exactly 2 huge pages
    /// ```
    pub fn new(page_size: HugePageSize, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(Error::AllocationFailed(
                "size must be greater than 0".into(),
            ));
        }

        // Round up to huge page boundary
        let page_bytes = page_size.bytes();
        let aligned_size = (size + page_bytes - 1) / page_bytes * page_bytes;

        // Build mmap flags
        // MAP_HUGETLB | (shift << MAP_HUGE_SHIFT)
        // MAP_HUGE_SHIFT is 26, so we shift the size log2 by 26 bits
        let huge_shift = page_size.shift();
        let huge_flags =
            MapFlags::from_bits_retain(MapFlags::HUGETLB.bits() | ((huge_shift as u32) << 26));

        let flags = MapFlags::PRIVATE | huge_flags;

        // Allocate with mmap_anonymous (handles ANONYMOUS flag internally)
        let ptr = unsafe {
            rustix::mm::mmap_anonymous(
                std::ptr::null_mut(),
                aligned_size,
                ProtFlags::READ | ProtFlags::WRITE,
                flags,
            )?
        };

        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;

        Ok(Self {
            ptr,
            len: aligned_size,
            page_size,
        })
    }

    /// Try to allocate with huge pages, falling back to regular pages on failure.
    ///
    /// This is useful when huge pages are preferred but not required.
    pub fn new_or_fallback(page_size: HugePageSize, size: usize) -> Result<Self> {
        match Self::new(page_size, size) {
            Ok(segment) => Ok(segment),
            Err(_) => {
                // Fall back to anonymous mmap without huge pages
                if size == 0 {
                    return Err(Error::AllocationFailed(
                        "size must be greater than 0".into(),
                    ));
                }

                let flags = MapFlags::PRIVATE;

                let ptr = unsafe {
                    rustix::mm::mmap_anonymous(
                        std::ptr::null_mut(),
                        size,
                        ProtFlags::READ | ProtFlags::WRITE,
                        flags,
                    )?
                };

                let ptr = NonNull::new(ptr.cast::<u8>())
                    .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;

                // Note: This isn't actually using huge pages, but provides
                // a fallback for when huge pages aren't available
                Ok(Self {
                    ptr,
                    len: size,
                    page_size,
                })
            }
        }
    }

    /// Get the huge page size used.
    pub fn page_size(&self) -> HugePageSize {
        self.page_size
    }

    /// Get the number of huge pages allocated.
    pub fn page_count(&self) -> usize {
        self.len / self.page_size.bytes()
    }

    /// Pre-fault all pages to avoid page faults during use.
    ///
    /// This touches every huge page to ensure physical memory is allocated.
    pub fn prefault(&self) {
        let page_bytes = self.page_size.bytes();
        let ptr = self.ptr.as_ptr();
        for offset in (0..self.len).step_by(page_bytes) {
            unsafe {
                // Volatile read to prevent optimization
                std::ptr::read_volatile(ptr.add(offset));
            }
        }
    }
}

impl MemorySegment for HugePageSegment {
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
        MemoryType::HugePages
    }

    fn ipc_handle(&self) -> Option<IpcHandle> {
        // Anonymous huge pages can't be shared directly via fd
        // Would need to use memfd with huge page flags
        None
    }
}

impl Drop for HugePageSegment {
    fn drop(&mut self) {
        unsafe {
            let _ = rustix::mm::munmap(self.ptr.as_ptr().cast(), self.len);
        }
    }
}

// SAFETY: HugePageSegment is Send + Sync because the memory is private
// and can be accessed from any thread safely.
unsafe impl Send for HugePageSegment {}
unsafe impl Sync for HugePageSegment {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_huge_page_size_bytes() {
        assert_eq!(HugePageSize::MB2.bytes(), 2 * 1024 * 1024);
        assert_eq!(HugePageSize::GB1.bytes(), 1024 * 1024 * 1024);
    }

    #[test]
    fn test_huge_page_fallback() {
        // This should always succeed (falls back to regular pages)
        let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, 4096).unwrap();
        assert!(segment.len() >= 4096);
        assert_eq!(segment.memory_type(), MemoryType::HugePages);
    }

    #[test]
    fn test_huge_page_read_write() {
        // Use fallback since we can't guarantee huge pages are available
        let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, 4096).unwrap();

        let ptr = segment.as_mut_ptr().unwrap();
        unsafe {
            std::ptr::write(ptr, 42);
            std::ptr::write(ptr.add(1), 43);
            assert_eq!(std::ptr::read(ptr), 42);
            assert_eq!(std::ptr::read(ptr.add(1)), 43);
        }
    }

    #[test]
    fn test_huge_page_zero_size_fails() {
        let result = HugePageSegment::new(HugePageSize::MB2, 0);
        assert!(result.is_err());

        let result = HugePageSegment::new_or_fallback(HugePageSize::MB2, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_huge_page_no_ipc() {
        let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, 4096).unwrap();
        assert!(segment.ipc_handle().is_none());
    }

    #[test]
    fn test_prefault() {
        let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, 4096).unwrap();
        segment.prefault(); // Should not panic
    }
}
