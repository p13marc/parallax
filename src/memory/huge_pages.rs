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
use std::os::fd::{AsRawFd, OwnedFd};
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

    /// The matching `MFD_HUGE_*` flag for `memfd_create`.
    ///
    /// This replaces the old `MAP_HUGE_* << MAP_HUGE_SHIFT` mmap flag: the
    /// segment is now memfd-backed, so the page size is selected when the fd
    /// is created rather than when it is mapped.
    fn memfd_flag(self) -> rustix::fs::MemfdFlags {
        match self {
            HugePageSize::MB2 => rustix::fs::MemfdFlags::HUGE_2MB,
            HugePageSize::GB1 => rustix::fs::MemfdFlags::HUGE_1GB,
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
    /// Size of the segment (rounded up to the *effective* page size).
    len: usize,
    /// The huge page size requested.
    page_size: HugePageSize,
    /// Backing memfd, so the segment can be shared over SCM_RIGHTS.
    ///
    /// `MemoryType::HugePages.supports_ipc()` returns `true`, and before this
    /// existed the segment was an anonymous `MAP_HUGETLB` mapping with no fd,
    /// so `ipc_handle()` always returned `None` and the capability matrix
    /// advertised something the segment could not deliver.
    fd: OwnedFd,
    /// Whether [`new_or_fallback`](Self::new_or_fallback) had to degrade to
    /// normal pages.
    ///
    /// Tracked because the fallback used to keep reporting
    /// `MemoryType::HugePages` — and, worse, kept `page_size` as the divisor
    /// for `page_count()` and the step for `prefault()`, so a sub-2 MiB
    /// fallback reported zero pages and pre-faulted nothing.
    fell_back: bool,
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
    /// - No huge pages are available (see `/proc/meminfo`), permissions are
    ///   insufficient, or `memfd_create`/`mmap` otherwise fails
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
        let aligned_size = size.div_ceil(page_bytes) * page_bytes;

        // Back the segment with a hugetlb memfd rather than an anonymous
        // MAP_HUGETLB mapping, so it has a real fd and can cross a process
        // boundary via SCM_RIGHTS. `MemoryType::HugePages.supports_ipc()`
        // claims it can; this is what makes that true.
        let fd = rustix::fs::memfd_create(
            c"parallax-hugepage",
            rustix::fs::MemfdFlags::CLOEXEC
                | rustix::fs::MemfdFlags::HUGETLB
                | page_size.memfd_flag(),
        )?;
        rustix::fs::ftruncate(&fd, aligned_size as u64)?;

        // MAP_SHARED, not PRIVATE: a private mapping of a shared fd would
        // copy-on-write and defeat the point of having the fd at all.
        let ptr = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                aligned_size,
                ProtFlags::READ | ProtFlags::WRITE,
                MapFlags::SHARED,
                &fd,
                0,
            )?
        };

        let ptr = NonNull::new(ptr.cast::<u8>())
            .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;

        Ok(Self {
            ptr,
            len: aligned_size,
            page_size,
            fd,
            fell_back: false,
        })
    }

    /// Try to allocate with huge pages, falling back to regular pages on failure.
    ///
    /// This is useful when huge pages are preferred but not required. A
    /// fallback segment is still memfd-backed and still IPC-shareable — it
    /// just uses normal pages, and says so: [`fell_back`](Self::fell_back)
    /// returns `true`, [`memory_type`](MemorySegment::memory_type) reports
    /// [`MemoryType::Cpu`], and [`page_count`](Self::page_count) and
    /// [`prefault`](Self::prefault) use the real page size.
    pub fn new_or_fallback(page_size: HugePageSize, size: usize) -> Result<Self> {
        match Self::new(page_size, size) {
            Ok(segment) => Ok(segment),
            Err(e) => {
                if size == 0 {
                    return Err(Error::AllocationFailed(
                        "size must be greater than 0".into(),
                    ));
                }
                tracing::debug!(
                    "huge pages unavailable ({e}), falling back to normal pages; \
                     check HugePages_Free in /proc/meminfo"
                );

                let normal_page = normal_page_size();
                let aligned_size = size.div_ceil(normal_page) * normal_page;

                let fd = rustix::fs::memfd_create(
                    c"parallax-hugepage-fallback",
                    rustix::fs::MemfdFlags::CLOEXEC,
                )?;
                rustix::fs::ftruncate(&fd, aligned_size as u64)?;

                let ptr = unsafe {
                    rustix::mm::mmap(
                        std::ptr::null_mut(),
                        aligned_size,
                        ProtFlags::READ | ProtFlags::WRITE,
                        MapFlags::SHARED,
                        &fd,
                        0,
                    )?
                };

                let ptr = NonNull::new(ptr.cast::<u8>())
                    .ok_or_else(|| Error::AllocationFailed("mmap returned null".into()))?;

                Ok(Self {
                    ptr,
                    len: aligned_size,
                    page_size,
                    fd,
                    fell_back: true,
                })
            }
        }
    }

    /// The huge page size that was *requested*.
    ///
    /// This is not necessarily what backs the segment — check
    /// [`fell_back`](Self::fell_back).
    pub fn page_size(&self) -> HugePageSize {
        self.page_size
    }

    /// Whether this segment degraded to normal pages.
    ///
    /// Only [`new_or_fallback`](Self::new_or_fallback) can return `true`;
    /// [`new`](Self::new) either gets huge pages or errors.
    pub fn fell_back(&self) -> bool {
        self.fell_back
    }

    /// The size of the pages actually backing this segment.
    pub fn effective_page_size(&self) -> usize {
        if self.fell_back {
            normal_page_size()
        } else {
            self.page_size.bytes()
        }
    }

    /// Get the number of pages allocated, in whatever size actually backs it.
    pub fn page_count(&self) -> usize {
        self.len / self.effective_page_size()
    }

    /// Pre-fault all pages to avoid page faults during use.
    ///
    /// This touches every huge page to ensure physical memory is allocated.
    pub fn prefault(&self) {
        let page_bytes = self.effective_page_size();
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
        // Report what actually backs the segment. Claiming `HugePages` after
        // silently degrading to 4 KiB pages made the reported type useless for
        // any caller trying to reason about TLB behaviour.
        if self.fell_back {
            MemoryType::Cpu
        } else {
            MemoryType::HugePages
        }
    }

    fn ipc_handle(&self) -> Option<IpcHandle> {
        Some(IpcHandle::Fd {
            fd: self.fd.as_raw_fd(),
            size: self.len,
        })
    }
}

/// The system's normal page size, for the fallback path.
fn normal_page_size() -> usize {
    // rustix exposes this without a syscall on Linux (it comes from the auxv).
    rustix::param::page_size()
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

    /// True when the system actually has a hugetlb pool reserved. Most dev
    /// boxes have `HugePages_Total: 0`, so the huge path cannot be exercised
    /// there and the tests below assert on whichever path was taken.
    fn huge_pages_available() -> bool {
        std::fs::read_to_string("/proc/meminfo")
            .ok()
            .and_then(|s| {
                s.lines()
                    .find(|l| l.starts_with("HugePages_Free:"))
                    .and_then(|l| l.split_whitespace().nth(1)?.parse::<u64>().ok())
            })
            .is_some_and(|free| free > 0)
    }

    #[test]
    fn test_huge_page_fallback() {
        // This should always succeed: huge pages if the pool allows, normal
        // pages otherwise.
        let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, 4096).unwrap();
        assert!(segment.len() >= 4096);

        // The reported type must match what actually backs it. It used to say
        // HugePages unconditionally, even after degrading to 4 KiB pages.
        if segment.fell_back() {
            assert_eq!(segment.memory_type(), MemoryType::Cpu);
        } else {
            assert_eq!(segment.memory_type(), MemoryType::HugePages);
        }
    }

    #[test]
    fn fallback_reports_a_real_page_count() {
        // A 4 KiB request that degrades to normal pages used to report
        // `4096 / 2MiB == 0` pages, because page_count() divided by the
        // *requested* huge page size.
        let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, 4096).unwrap();
        assert!(
            segment.page_count() >= 1,
            "a non-empty segment must hold at least one page, got {} (len {}, page {})",
            segment.page_count(),
            segment.len(),
            segment.effective_page_size()
        );
        assert_eq!(segment.len() % segment.effective_page_size(), 0);
    }

    #[test]
    fn new_errors_rather_than_degrading() {
        // `new` must never silently hand back normal pages — that is what
        // `new_or_fallback` is for.
        match HugePageSegment::new(HugePageSize::MB2, 4096) {
            Ok(seg) => {
                assert!(!seg.fell_back());
                assert_eq!(seg.memory_type(), MemoryType::HugePages);
                assert!(huge_pages_available(), "got huge pages with an empty pool?");
            }
            Err(_) => assert!(
                !huge_pages_available(),
                "huge pages are available but new() failed"
            ),
        }
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

    /// The segment is memfd-backed, so `supports_ipc()` is no longer a lie.
    ///
    /// This used to assert `ipc_handle().is_none()` while
    /// `MemoryType::HugePages.supports_ipc()` returned `true` — the
    /// contradiction was test-enshrined.
    #[test]
    fn huge_page_segment_is_fd_shareable() {
        let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, 4096).unwrap();

        match segment.ipc_handle() {
            Some(IpcHandle::Fd { fd, size }) => {
                assert!(fd >= 0, "expected a real fd, got {fd}");
                assert_eq!(size, segment.len());
            }
            other => panic!("expected an fd handle, got {other:?}"),
        }

        assert!(
            segment.memory_type().supports_ipc(),
            "the reported memory type must agree with the handle"
        );
    }

    /// Writes through the mapping are visible through a second mapping of the
    /// same fd — i.e. it is genuinely MAP_SHARED, not a private copy.
    #[test]
    fn writes_are_visible_through_the_shared_fd() {
        let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, 4096).unwrap();
        let Some(IpcHandle::Fd { fd, size }) = segment.ipc_handle() else {
            panic!("expected an fd handle");
        };

        unsafe { std::ptr::write(segment.as_mut_ptr().unwrap(), 0xAB) };

        // Map the same fd again, as a peer process would after SCM_RIGHTS.
        let borrowed = unsafe { std::os::fd::BorrowedFd::borrow_raw(fd) };
        let second = unsafe {
            rustix::mm::mmap(
                std::ptr::null_mut(),
                size,
                ProtFlags::READ,
                MapFlags::SHARED,
                borrowed,
                0,
            )
            .unwrap()
        };

        assert_eq!(unsafe { std::ptr::read(second.cast::<u8>()) }, 0xAB);
        unsafe { rustix::mm::munmap(second, size).unwrap() };
    }

    #[test]
    fn test_prefault() {
        let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, 4096).unwrap();
        segment.prefault(); // Should not panic
    }
}
