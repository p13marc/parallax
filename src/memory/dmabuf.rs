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
use crate::memory::MemoryType;
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

impl DmaBufSegment {
    /// The memory type this segment reports in caps terms.
    pub fn memory_type(&self) -> MemoryType {
        MemoryType::DmaBuf
    }
}

// ============================================================================
// DmaBufSlot — the refcounted, releasable unit behind MemoryHandle::DmaBuf
// ============================================================================

/// Release hook: called with the producer-side buffer index when the last
/// reference to a [`DmaBufSlot`] drops. For a V4L2 source this re-queues the
/// buffer to the driver (QBUF); tests use it to observe recycling.
pub type DmaBufReleaseHook = Box<dyn Fn(u32) + Send + Sync>;

/// `DRM_FORMAT_MOD_LINEAR` — rows end to end at the declared stride.
pub const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// `I915_FORMAT_MOD_Y_TILED` — Intel Y-tiles, 128 bytes by 32 rows, each
/// tile stored as eight 16-byte columns. What Intel's VA driver renders
/// decode output into (see `crate::gpu::vaapi`).
pub const I915_FORMAT_MOD_Y_TILED: u64 = (1u64 << 56) | 2;

/// A refcounted dmabuf-backed slot (#145) — the DmaBuf counterpart of
/// [`SharedSlotRef`](crate::memory::SharedSlotRef)'s discipline: cloning a
/// buffer clones an `Arc<DmaBufSlot>`, and the LAST drop fires the release
/// hook so the producer can recycle the underlying driver buffer.
///
/// The `segment` is itself `Arc`'d so a pooled producer keeps one long-lived
/// mmap per driver buffer across recycles; dropping the slot returns the
/// *index* to the producer, it does not unmap anything.
pub struct DmaBufSlot {
    /// The mapped dmabuf. Shared with the producer's pool for pooled slots.
    segment: std::sync::Arc<DmaBufSegment>,
    /// Producer-side buffer index (V4L2 QBUF index); 0 for one-shot slots.
    index: u32,
    /// Fired exactly once, from `Drop` — i.e. when the last
    /// `Arc<DmaBufSlot>` clone goes away.
    release: Option<DmaBufReleaseHook>,
    /// How the bytes are arranged in memory, as a DRM format modifier.
    ///
    /// A property of the *allocation*, not of the frame, which is why it
    /// lives here and not in `Metadata`: a transform that copies a dmabuf
    /// into CPU memory must not carry it along, and geometry is already
    /// `Metadata`'s job.
    modifier: u64,
}

impl DmaBufSlot {
    /// One-shot slot: owns its segment, no recycle hook (imports, tests).
    pub fn new(segment: DmaBufSegment) -> Self {
        Self {
            segment: std::sync::Arc::new(segment),
            index: 0,
            release: None,
            modifier: DRM_FORMAT_MOD_LINEAR,
        }
    }

    /// Pooled slot: shared segment plus a recycle hook called with `index`
    /// on last drop (the V4L2 re-queue path).
    pub fn with_release(
        segment: std::sync::Arc<DmaBufSegment>,
        index: u32,
        hook: DmaBufReleaseHook,
    ) -> Self {
        Self {
            segment,
            index,
            release: Some(hook),
            modifier: DRM_FORMAT_MOD_LINEAR,
        }
    }

    /// Declare the allocation's DRM format modifier.
    ///
    /// The default is [`DRM_FORMAT_MOD_LINEAR`], which is what every
    /// CPU-readable producer has: rows laid end to end at the declared
    /// stride. A producer whose memory is *not* linear — a GPU decode
    /// target, say — must say so, because the bytes are then meaningless
    /// to anything but an importer that knows the layout.
    pub fn with_modifier(mut self, modifier: u64) -> Self {
        self.modifier = modifier;
        self
    }

    /// The allocation's DRM format modifier.
    pub fn modifier(&self) -> u64 {
        self.modifier
    }

    /// Whether the bytes can be read as rows at the declared stride.
    ///
    /// `false` means [`crate::buffer::Buffer::as_bytes`] on this slot is a
    /// live mapping of memory laid out in some tiled or compressed order:
    /// readable, but not a picture. Only an importer told the modifier can
    /// make sense of it.
    pub fn is_linear(&self) -> bool {
        self.modifier == DRM_FORMAT_MOD_LINEAR
    }

    /// The mapped segment, shared.
    ///
    /// A GPU importer keys its cache on this pointer: a pooled producer
    /// hands out a fresh `DmaBufSlot` per frame over a *stable* segment, so
    /// the segment is the identity of the underlying allocation and holding
    /// an `Arc` of it keeps that identity from being reused underneath.
    pub fn shared_segment(&self) -> &std::sync::Arc<DmaBufSegment> {
        &self.segment
    }

    /// The mapped segment.
    pub fn segment(&self) -> &DmaBufSegment {
        &self.segment
    }

    /// Producer-side buffer index.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// The dmabuf fd — the GPU-import hook (#62: `import_dmabuf` takes
    /// exactly this).
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.segment.as_fd()
    }
}

impl Drop for DmaBufSlot {
    fn drop(&mut self) {
        if let Some(hook) = self.release.take() {
            hook(self.index);
        }
    }
}

impl std::fmt::Debug for DmaBufSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DmaBufSlot")
            .field("fd", &self.segment.as_fd().as_raw_fd())
            .field("index", &self.index)
            .field("len", &self.segment.size())
            .field("read_only", &self.segment.is_read_only())
            .field("has_release", &self.release.is_some())
            .finish()
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

        assert_eq!(segment.size(), 4096);
        assert_eq!(segment.memory_type(), MemoryType::DmaBuf);
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

        let mut segment = DmaBufSegment::from_fd_readonly(fd, 512).unwrap();

        assert!(segment.is_read_only());
        assert!(segment.as_mut_slice().is_none());

        // Reading should still work
        let _ = segment.as_slice();
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

#[cfg(test)]
mod slot_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

    fn memfd_segment(len: u64) -> DmaBufSegment {
        let fd =
            rustix::fs::memfd_create("dmabuf_slot_test", rustix::fs::MemfdFlags::CLOEXEC).unwrap();
        rustix::fs::ftruncate(&fd, len).unwrap();
        DmaBufSegment::from_fd(fd, len as usize).unwrap()
    }

    #[test]
    fn release_hook_fires_once_on_last_drop_with_index() {
        let fired = Arc::new(AtomicU32::new(0));
        let index_seen = Arc::new(AtomicU64::new(u64::MAX));
        let fired_hook = fired.clone();
        let index_hook = index_seen.clone();

        let segment = Arc::new(memfd_segment(4096));
        let slot = Arc::new(DmaBufSlot::with_release(
            segment,
            7,
            Box::new(move |idx| {
                fired_hook.fetch_add(1, Ordering::SeqCst);
                index_hook.store(idx as u64, Ordering::SeqCst);
            }),
        ));

        let a = Arc::clone(&slot);
        let b = Arc::clone(&slot);
        drop(slot);
        drop(a);
        assert_eq!(fired.load(Ordering::SeqCst), 0, "clones still alive");
        drop(b);
        assert_eq!(fired.load(Ordering::SeqCst), 1, "fires exactly once");
        assert_eq!(index_seen.load(Ordering::SeqCst), 7);
    }

    #[test]
    fn one_shot_slot_has_no_hook_and_index_zero() {
        let slot = DmaBufSlot::new(memfd_segment(1024));
        assert_eq!(slot.index(), 0);
        drop(slot); // no hook — must not panic
    }

    #[test]
    fn pooled_slot_shares_the_segment_mapping() {
        let segment = Arc::new(memfd_segment(1024));
        let slot = DmaBufSlot::with_release(Arc::clone(&segment), 0, Box::new(|_| {}));
        assert_eq!(
            slot.segment().as_slice().as_ptr(),
            segment.as_slice().as_ptr(),
            "one mapping, shared"
        );
    }

    /// Real dmabuf validation via /dev/udmabuf when available (skips
    /// otherwise): a udmabuf-created fd exercises actual dma-buf f_ops.
    #[test]
    fn udmabuf_backed_segment_when_available() {
        use rustix::fs::{MemfdFlags, SealFlags};

        let Ok(dev) = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open("/dev/udmabuf")
        else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };

        const SIZE: u64 = 4096;
        let memfd = rustix::fs::memfd_create(
            "udmabuf_source",
            MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
        )
        .unwrap();
        rustix::fs::ftruncate(&memfd, SIZE).unwrap();
        rustix::fs::fcntl_add_seals(&memfd, SealFlags::SHRINK).unwrap();

        #[repr(C)]
        struct UdmabufCreate {
            memfd: u32,
            flags: u32,
            offset: u64,
            size: u64,
        }
        // _IOW('u', 0x42, struct udmabuf_create)
        const UDMABUF_CREATE: libc::c_ulong = 0x4018_7542;
        let arg = UdmabufCreate {
            memfd: memfd.as_raw_fd() as u32,
            flags: 0,
            offset: 0,
            size: SIZE,
        };
        let raw = unsafe { libc::ioctl(dev.as_fd().as_raw_fd(), UDMABUF_CREATE, &arg) };
        if raw < 0 {
            eprintln!(
                "skipping: UDMABUF_CREATE failed ({})",
                std::io::Error::last_os_error()
            );
            return;
        }
        let dmabuf_fd = unsafe { <OwnedFd as rustix::fd::FromRawFd>::from_raw_fd(raw) };

        let segment = DmaBufSegment::from_fd(dmabuf_fd, SIZE as usize).unwrap();
        assert_eq!(segment.size(), SIZE as usize);
        // A real dma-buf mmap: write through it and read back.
        let mut segment = segment;
        segment.as_mut_slice().unwrap()[..4].copy_from_slice(b"dmab");
        assert_eq!(&segment.as_slice()[..4], b"dmab");
    }
}
