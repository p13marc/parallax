//! Decode output frames the pipeline owns (#193).
//!
//! `cros-codecs` makes the *caller* supply the frames a decoder renders
//! into, and keeps the resulting `Surface` `pub(crate)` — so the only way to
//! get pixels back out is [`VideoFrame::map`] on memory we allocated
//! ourselves. This module is that memory.
//!
//! # Why udmabuf and not gbm
//!
//! The obvious choice is a gbm buffer object, which is what `cros-codecs`'
//! own frame types use. Measured on this project's reference device (Comet
//! Lake Gen9.5, iHD 26.2.4), gbm refuses to allocate NV12 at all — every
//! flag combination (`LINEAR`, `RENDERING`, `SCANOUT`, none) returns
//! `EINVAL`. And a surface the *driver* allocates comes back Y-tiled
//! (modifier `0x100000000000002`), so mmapping its exported dmabuf yields
//! swizzled bytes that would need de-tiling on the CPU — which is most of
//! the cost hardware decode was supposed to remove.
//!
//! A udmabuf wraps an ordinary memfd as a dma-buf. That gives us:
//!
//! - **linear** memory, so a CPU read is a plain row copy;
//! - memory the pipeline owns, so [`map`](VideoFrame::map) is our own mmap
//!   rather than a driver round-trip;
//! - a dma-buf fd, which is what `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`
//!   wants — and which is already a first-class buffer backing in this
//!   pipeline ([`crate::memory::DmaBufSlot`], #145).
//!
//! Verified against the reference device: iHD accepts a linear
//! udmabuf-backed NV12 surface as a decode render target.
//!
//! `cros-codecs`' own `GenericDmaVideoFrame` is deliberately not used even
//! where it would fit: its `map()` runs `_mm_clflush` **per byte** on drop
//! (3.1 M instructions for one 1080p frame) and its unmap is a lazy
//! `iter().map(..)` that is never consumed, so the mapping is never
//! released.

use std::os::fd::{AsFd, AsRawFd, OwnedFd};
use std::sync::Arc;

use cros_codecs::libva;
use cros_codecs::video_frame::{ReadMapping, VideoFrame, WriteMapping};
use cros_codecs::{Fourcc, Resolution};

use crate::error::{Error, Result};

/// `DRM_FORMAT_NV12`, as a fourcc.
const DRM_FORMAT_NV12: u32 = u32::from_le_bytes(*b"NV12");
/// `DRM_FORMAT_MOD_LINEAR`.
const DRM_FORMAT_MOD_LINEAR: u64 = 0;

/// A linear NV12 frame in memory this process owns, exposed to VA-API as a
/// dma-buf.
///
/// Cheap to clone into the pipeline: the payload lives behind an `Arc`, so a
/// decoded frame can ride onward as a `Buffer` while the pool still tracks
/// the slot.
#[derive(Debug)]
pub struct VaFrame {
    inner: Arc<FrameMemory>,
    /// Visible size. The allocation is the *coded* size, which is larger
    /// whenever the stream's dimensions are not macroblock-aligned.
    visible: Resolution,
}

/// The allocation itself: a memfd, its dma-buf view, and a live mapping.
#[derive(Debug)]
struct FrameMemory {
    /// The dma-buf handed to VA-API. Kept alive for the frame's whole life;
    /// each import `dup`s it, so the driver never owns this one.
    dmabuf: OwnedFd,
    /// Mapped once at allocation rather than per frame — a decode pipeline
    /// maps every frame it produces, so the map/unmap pair would be pure
    /// per-frame syscall overhead.
    ptr: *mut u8,
    len: usize,
    /// Coded (allocated) geometry, which is what the strides describe.
    coded: Resolution,
    /// Byte offset and row pitch of the luma and chroma planes.
    offsets: [usize; 2],
    pitches: [usize; 2],
}

// SAFETY: `ptr` is a private MAP_SHARED mapping owned by this struct and
// unmapped exactly once in `Drop`. Nothing hands out a raw pointer; access
// goes through `&self`/`&mut self` slices, so Rust's borrow rules provide
// the synchronisation.
unsafe impl Send for FrameMemory {}
unsafe impl Sync for FrameMemory {}

impl Drop for FrameMemory {
    fn drop(&mut self) {
        // SAFETY: `ptr`/`len` are exactly what `mmap` returned, and this
        // runs once — `FrameMemory` is only ever behind an `Arc`.
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.len);
        }
    }
}

impl VaFrame {
    /// Allocate one NV12 frame at `coded` size, visible at `visible`.
    ///
    /// The coded size is what the decoder renders into (macroblock-aligned);
    /// the visible size is what the stream actually shows. Keeping both is
    /// what lets the readback crop without a second copy.
    pub fn new(coded: Resolution, visible: Resolution) -> Result<Self> {
        let pitch = coded.width as usize;
        let luma = pitch * coded.height as usize;
        let chroma = pitch * coded.height.div_ceil(2) as usize;
        // udmabuf requires a page-multiple size.
        let len = (luma + chroma).next_multiple_of(page_size());

        let dmabuf = udmabuf(len)?;
        // SAFETY: `dmabuf` is a live dma-buf fd of at least `len` bytes;
        // MAP_SHARED is what makes writes by the GPU visible here.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_SHARED,
                dmabuf.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(Error::Element(format!(
                "vaapi: mapping a {len}-byte output frame failed: {}",
                std::io::Error::last_os_error()
            )));
        }

        Ok(Self {
            inner: Arc::new(FrameMemory {
                dmabuf,
                ptr: ptr as *mut u8,
                len,
                coded,
                offsets: [0, luma],
                pitches: [pitch, pitch],
            }),
            visible,
        })
    }

    /// The frame's bytes.
    pub fn as_bytes(&self) -> &[u8] {
        // SAFETY: the mapping is live for as long as `inner` is, and this
        // borrow cannot outlive `&self`.
        unsafe { std::slice::from_raw_parts(self.inner.ptr, self.inner.len) }
    }

    /// Plane byte offsets within the frame: luma, then interleaved chroma.
    pub fn offsets(&self) -> [usize; 2] {
        self.inner.offsets
    }

    /// Plane row pitches, in bytes.
    pub fn pitches(&self) -> [usize; 2] {
        self.inner.pitches
    }

    /// Coded (allocated) geometry.
    pub fn coded(&self) -> Resolution {
        self.inner.coded
    }

    /// Borrow the dma-buf, e.g. to hand the pipeline a
    /// [`DmaBufSlot`](crate::memory::DmaBufSlot).
    pub fn dmabuf(&self) -> std::os::fd::BorrowedFd<'_> {
        self.inner.dmabuf.as_fd()
    }
}

/// Wrap `len` bytes of fresh memfd as a dma-buf.
fn udmabuf(len: usize) -> Result<OwnedFd> {
    use rustix::fs::{MemfdFlags, SealFlags};

    let dev = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open("/dev/udmabuf")
        .map_err(|e| {
            Error::Element(format!(
                "vaapi: /dev/udmabuf unavailable ({e}) — hardware decode needs it to \
                 allocate output frames the driver can render into"
            ))
        })?;

    let memfd = rustix::fs::memfd_create(
        "parallax-vaapi-frame",
        MemfdFlags::CLOEXEC | MemfdFlags::ALLOW_SEALING,
    )
    .map_err(|e| Error::Element(format!("vaapi: memfd_create failed: {e}")))?;
    rustix::fs::ftruncate(&memfd, len as u64)
        .map_err(|e| Error::Element(format!("vaapi: ftruncate failed: {e}")))?;
    // udmabuf refuses a memfd that can still shrink under it.
    rustix::fs::fcntl_add_seals(&memfd, SealFlags::SHRINK)
        .map_err(|e| Error::Element(format!("vaapi: sealing the memfd failed: {e}")))?;

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
        size: len as u64,
    };
    // SAFETY: `arg` matches the kernel's `struct udmabuf_create` and lives
    // across the call; `dev` is a live /dev/udmabuf handle.
    let raw = unsafe { libc::ioctl(dev.as_fd().as_raw_fd(), UDMABUF_CREATE, &arg) };
    if raw < 0 {
        return Err(Error::Element(format!(
            "vaapi: UDMABUF_CREATE failed: {}",
            std::io::Error::last_os_error()
        )));
    }
    // SAFETY: the ioctl returned a fresh owned fd.
    Ok(unsafe { <OwnedFd as rustix::fd::FromRawFd>::from_raw_fd(raw) })
}

fn page_size() -> usize {
    // SAFETY: a plain sysconf query.
    (unsafe { libc::sysconf(libc::_SC_PAGESIZE) }) as usize
}

/// The DRM_PRIME_2 import descriptor for one [`VaFrame`].
///
/// Holds a `dup` of the frame's dma-buf: `create_surfaces` takes the
/// descriptor by value and the driver keeps a reference, so the frame's own
/// fd must not be the one handed over.
#[derive(Debug)]
pub struct VaFrameDescriptor {
    fd: OwnedFd,
    len: usize,
    coded: Resolution,
    offsets: [usize; 2],
    pitches: [usize; 2],
}

impl libva::ExternalBufferDescriptor for VaFrameDescriptor {
    const MEMORY_TYPE: libva::MemoryType = libva::MemoryType::DrmPrime2;
    type DescriptorAttribute = libva::VADRMPRIMESurfaceDescriptor;

    fn va_surface_attribute(&mut self) -> Self::DescriptorAttribute {
        let mut desc: libva::VADRMPRIMESurfaceDescriptor = Default::default();
        desc.fourcc = libva::VA_FOURCC_NV12;
        desc.width = self.coded.width;
        desc.height = self.coded.height;
        desc.num_objects = 1;
        desc.objects[0].fd = self.fd.as_raw_fd();
        desc.objects[0].size = self.len as u32;
        desc.objects[0].drm_format_modifier = DRM_FORMAT_MOD_LINEAR;
        desc.num_layers = 1;
        desc.layers[0].drm_format = DRM_FORMAT_NV12;
        desc.layers[0].num_planes = 2;
        desc.layers[0].object_index = [0; 4];
        desc.layers[0].offset = [self.offsets[0] as u32, self.offsets[1] as u32, 0, 0];
        desc.layers[0].pitch = [self.pitches[0] as u32, self.pitches[1] as u32, 0, 0];
        desc
    }
}

/// Read-only view of a frame's planes, in `cros-codecs`' shape.
struct VaFrameMapping<'a> {
    planes: Vec<&'a [u8]>,
}

impl<'a> ReadMapping<'a> for VaFrameMapping<'a> {
    fn get(&self) -> Vec<&[u8]> {
        self.planes.clone()
    }
}

impl VideoFrame for VaFrame {
    type MemDescriptor = VaFrameDescriptor;
    type NativeHandle = libva::Surface<VaFrameDescriptor>;

    fn fourcc(&self) -> Fourcc {
        Fourcc::from(b"NV12")
    }

    fn resolution(&self) -> Resolution {
        self.visible
    }

    fn get_plane_size(&self) -> Vec<usize> {
        let h = self.inner.coded.height as usize;
        vec![
            self.inner.pitches[0] * h,
            self.inner.pitches[1] * h.div_ceil(2),
        ]
    }

    fn get_plane_pitch(&self) -> Vec<usize> {
        self.inner.pitches.to_vec()
    }

    fn map<'a>(&'a self) -> std::result::Result<Box<dyn ReadMapping<'a> + 'a>, String> {
        let bytes = self.as_bytes();
        let sizes = self.get_plane_size();
        let planes = self
            .inner
            .offsets
            .iter()
            .zip(sizes)
            .map(|(&off, size)| &bytes[off..off + size])
            .collect();
        Ok(Box::new(VaFrameMapping { planes }))
    }

    fn map_mut<'a>(&'a mut self) -> std::result::Result<Box<dyn WriteMapping<'a> + 'a>, String> {
        // Decode output is read-only from our side: the GPU writes it, the
        // pipeline reads it. Nothing in the decode path needs this, and
        // handing out `&mut` to memory the driver may be rendering into
        // would be unsound.
        Err("parallax VA-API frames are read-only".to_string())
    }

    fn to_native_handle(
        &self,
        display: &Arc<libva::Display>,
    ) -> std::result::Result<Self::NativeHandle, String> {
        // A fresh `dup` per import: `create_surfaces` takes the descriptor
        // by value and the driver holds the fd for the surface's life.
        let fd = self
            .inner
            .dmabuf
            .try_clone()
            .map_err(|e| format!("vaapi: duplicating the frame fd failed: {e}"))?;

        let mut surfaces = display
            .create_surfaces(
                libva::VA_RT_FORMAT_YUV420,
                Some(libva::VA_FOURCC_NV12),
                self.inner.coded.width,
                self.inner.coded.height,
                Some(libva::UsageHint::USAGE_HINT_DECODER),
                vec![VaFrameDescriptor {
                    fd,
                    len: self.inner.len,
                    coded: self.inner.coded,
                    offsets: self.inner.offsets,
                    pitches: self.inner.pitches,
                }],
            )
            .map_err(|e| format!("vaapi: importing an output frame failed: {e:?}"))?;

        if surfaces.len() != 1 {
            return Err(format!(
                "vaapi: expected one surface per frame, got {}",
                surfaces.len()
            ));
        }
        Ok(surfaces.remove(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn res(width: u32, height: u32) -> Resolution {
        Resolution { width, height }
    }

    /// Allocation, geometry and mapping, with no GPU involved — udmabuf is
    /// a kernel facility, not a driver one.
    #[test]
    fn a_frame_is_linear_nv12_of_the_coded_size() {
        let Ok(frame) = VaFrame::new(res(1920, 1088), res(1920, 1080)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };

        assert_eq!(frame.resolution(), res(1920, 1080), "visible size");
        assert_eq!(frame.coded(), res(1920, 1088), "coded size");
        assert_eq!(frame.pitches(), [1920, 1920]);
        assert_eq!(frame.offsets(), [0, 1920 * 1088]);
        assert_eq!(frame.get_plane_size(), vec![1920 * 1088, 1920 * 544]);
        // Page-rounded, so at least the two planes.
        assert!(frame.as_bytes().len() >= 1920 * 1088 * 3 / 2);

        // cros-codecs' own consistency check, which is what the decoder
        // backend runs before rendering into a frame.
        frame.validate_frame().expect("a valid NV12 frame");
    }

    /// The mapping addresses both planes at their declared offsets, and
    /// writes through it are visible — i.e. it really is one shared
    /// allocation, not a copy.
    #[test]
    fn the_mapping_addresses_both_planes() {
        let Ok(mut frame) = VaFrame::new(res(64, 48), res(64, 48)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };

        // Write a marker at the start of each plane through the raw memory.
        let offsets = frame.offsets();
        {
            // SAFETY: the mapping is live and exclusively borrowed here.
            let bytes = unsafe { std::slice::from_raw_parts_mut(frame.inner.ptr, frame.inner.len) };
            bytes[offsets[0]] = 0xA1;
            bytes[offsets[1]] = 0xB2;
        }

        {
            let mapping = frame.map().expect("mapping");
            let planes = mapping.get();
            assert_eq!(planes.len(), 2);
            assert_eq!(planes[0][0], 0xA1, "luma plane starts at offset 0");
            assert_eq!(planes[1][0], 0xB2, "chroma plane starts after luma");
            assert_eq!(planes[0].len(), 64 * 48);
            assert_eq!(planes[1].len(), 64 * 24);
        }

        // Decode output is never written from our side.
        assert!(frame.map_mut().is_err());
    }

    /// The driver really accepts one of these as a decode render target.
    ///
    /// This is the load-bearing assumption of the whole design — if it were
    /// false, the pipeline could not own its decode output and would have to
    /// de-tile driver-allocated surfaces on the CPU. Green-skips without a
    /// VA display, so it is only meaningful where there is hardware.
    #[test]
    fn a_frame_imports_as_a_va_decode_target() {
        let Some(display) = super::super::VaDisplay::open() else {
            eprintln!("skipping: no VA display");
            return;
        };
        let Ok(frame) = VaFrame::new(res(1920, 1088), res(1920, 1080)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };

        frame
            .to_native_handle(&display.handle())
            .expect("iHD accepts a linear udmabuf-backed NV12 surface");

        // Importing twice must work: `to_native_handle` is called once per
        // *picture*, not once per frame, so a pooled frame is imported again
        // every time it is reused. Each import dups the fd; if it handed
        // over the frame's own, the second would fail.
        frame
            .to_native_handle(&display.handle())
            .expect("a frame can be imported more than once");
    }

    /// Odd heights round the chroma plane up rather than truncating it —
    /// a half-row short would have the driver writing past the allocation.
    #[test]
    fn odd_coded_height_rounds_chroma_up() {
        let Ok(frame) = VaFrame::new(res(16, 18), res(16, 17)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };
        assert_eq!(frame.get_plane_size(), vec![16 * 18, 16 * 9]);
    }
}
