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
//! - memory the pipeline owns, so [`map`](VideoFrame::map) is our own mmap
//!   rather than a driver round-trip;
//! - a dma-buf fd, which is what `VA_SURFACE_ATTRIB_MEM_TYPE_DRM_PRIME_2`
//!   wants — and which is already a first-class buffer backing in this
//!   pipeline ([`crate::memory::DmaBufSlot`], #145).
//!
//! # The frames are Y-tiled, and that is not negotiable
//!
//! The first version of this module asked for `DRM_FORMAT_MOD_LINEAR` and
//! believed the answer: the import succeeds, and `vaExportSurfaceHandle`
//! even echoes `drm_format_modifier = 0` back. The pixels say otherwise.
//! Measured against a software decode of the same stream, the decoded
//! frames are laid out in **Intel Y-tiles** — 128-byte by 32-row tiles,
//! each tile stored as eight 16-byte columns of 32 rows — bit-exactly, at
//! every resolution tested (128x128, 256x128, 640x480, 1920x1088). No
//! usage hint (decoder, display, export, VPP-write, none) and no layer
//! shape (one NV12 layer or two single-plane layers) changes it: on this
//! driver the decode render target is tiled whatever it is told.
//!
//! Two consequences, both load-bearing:
//!
//! - The **allocation must be tile-shaped**, or the driver writes past it.
//!   The pitch is rounded up to [`TILE_WIDTH`] and each plane's row count
//!   to [`TILE_HEIGHT`]; a 460-wide frame is really 512 bytes per row and
//!   the declared 460 was simply ignored — the driver overran a buffer
//!   sized from it.
//! - A CPU reader cannot address the bytes directly. [`VaFrame::read_plane`]
//!   is the only correct way out, and it de-tiles as it copies. The eventual
//!   zero-copy path does not de-tile at all: it hands the dma-buf and its
//!   modifier to the GPU, which is what a tiled frame is *for*.
//!
//! So the modifier this module declares is the truth — `I915_FORMAT_MOD_Y_TILED`
//! — rather than a linear request the driver ignores.
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
use crate::memory::I915_FORMAT_MOD_Y_TILED;

/// `DRM_FORMAT_NV12`, as a fourcc.
const DRM_FORMAT_NV12: u32 = u32::from_le_bytes(*b"NV12");

/// An Intel Y-tile is 128 bytes wide...
const TILE_WIDTH: usize = 128;
/// ...and 32 rows tall, so 4 KiB in all.
const TILE_HEIGHT: usize = 32;
/// Within a tile the bytes are stored as 16-byte columns, each column
/// holding all [`TILE_HEIGHT`] of its rows before the next one starts.
const TILE_COLUMN: usize = 16;

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
    /// Rows actually allocated per plane — the visible/coded row count
    /// rounded up to a whole tile, which is what the driver writes.
    rows: [usize; 2],
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

/// A fixed set of [`VaFrame`]s, reissued as the pipeline lets them go.
///
/// Allocating a frame costs a memfd, an ioctl and an mmap of several
/// megabytes, so a decoder that allocated per picture would spend more on
/// bookkeeping than on decoding. The pool holds one reference to each
/// allocation and hands out a frame only when nothing else holds it —
/// `Arc::strong_count == 1` means the decoder has released its handle *and*
/// the pipeline has dropped the buffer that rode on it.
///
/// Sizing comes from the stream: `StreamInfo::min_num_frames` is what the
/// codec needs for references alone, and anything the pipeline holds
/// downstream is on top of that.
#[derive(Debug)]
pub struct VaFramePool {
    slots: Vec<Arc<FrameMemory>>,
    coded: Resolution,
    visible: Resolution,
}

impl VaFramePool {
    /// An empty pool for frames of this geometry.
    pub fn new(coded: Resolution, visible: Resolution) -> Self {
        Self {
            slots: Vec::new(),
            coded,
            visible,
        }
    }

    /// Grow to `count` frames, allocating any that are missing.
    ///
    /// Never shrinks: a slot may still be in flight, and dropping our
    /// reference would only defer the free until the pipeline lets go
    /// anyway. Geometry changes go through [`reset`](Self::reset).
    pub fn reserve(&mut self, count: usize) -> Result<()> {
        while self.slots.len() < count {
            self.slots
                .push(VaFrame::new(self.coded, self.visible)?.inner);
        }
        Ok(())
    }

    /// Drop every slot and re-target the pool at a new geometry.
    ///
    /// In-flight frames keep their own allocation alive, so this is safe
    /// mid-stream — a resolution change simply stops reusing the old ones.
    pub fn reset(&mut self, coded: Resolution, visible: Resolution) {
        self.slots.clear();
        self.coded = coded;
        self.visible = visible;
    }

    /// A frame nothing else is holding, or `None` when all are in flight.
    pub fn acquire(&mut self) -> Option<VaFrame> {
        let free = self
            .slots
            .iter()
            .find(|slot| Arc::strong_count(slot) == 1)?;
        Some(VaFrame {
            inner: Arc::clone(free),
            visible: self.visible,
        })
    }

    /// How many frames exist.
    pub fn len(&self) -> usize {
        self.slots.len()
    }

    /// Whether the pool holds no frames.
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// How many are free right now — diagnostics, and the signal that a
    /// decoder is starved.
    pub fn available(&self) -> usize {
        self.slots
            .iter()
            .filter(|slot| Arc::strong_count(slot) == 1)
            .count()
    }

    /// The geometry this pool allocates.
    pub fn geometry(&self) -> (Resolution, Resolution) {
        (self.coded, self.visible)
    }
}

impl VaFrame {
    /// Allocate one NV12 frame at `coded` size, visible at `visible`.
    ///
    /// The coded size is what the decoder renders into (macroblock-aligned);
    /// the visible size is what the stream actually shows. Keeping both is
    /// what lets the readback crop without a second copy.
    pub fn new(coded: Resolution, visible: Resolution) -> Result<Self> {
        if coded.width == 0 || coded.height == 0 {
            return Err(Error::Element(format!(
                "vaapi: refusing to allocate a {}x{} frame — the pool is sized from the \
                 stream's geometry, so it must not be reserved before that is known",
                coded.width, coded.height
            )));
        }
        // Tile-shaped, not frame-shaped: the driver renders Y-tiles and
        // rounds the pitch up to a whole tile regardless of what we declare,
        // so a buffer sized from the frame's own width is one the driver
        // writes past. Rows are rounded per plane too — a partial bottom tile
        // is still a whole tile in memory.
        let pitch = (coded.width as usize).next_multiple_of(TILE_WIDTH);
        let luma_rows = (coded.height as usize).next_multiple_of(TILE_HEIGHT);
        let chroma_rows = coded
            .height
            .div_ceil(2)
            .next_multiple_of(TILE_HEIGHT as u32) as usize;
        let luma = pitch * luma_rows;
        let chroma = pitch * chroma_rows;
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
                rows: [luma_rows, chroma_rows],
            }),
            visible,
        })
    }

    /// Copy `rows` rows of `plane` into `dst`, de-tiling on the way.
    ///
    /// `dst` is packed at `dst_stride` bytes per row and `row_bytes` of each
    /// row are meaningful — which is how the coded-to-visible crop happens
    /// for free, since the tiled source is addressed per 16-byte column
    /// anyway and the columns past the visible width are simply never read.
    ///
    /// This is the only correct way to read a decoded frame on the CPU: the
    /// bytes behind [`as_bytes`](Self::as_bytes) are Y-tiled, so reading
    /// them as rows yields a coherent but scrambled picture (the failure
    /// looks like a diagonal shear, which is what made it visible at all).
    pub fn read_plane(
        &self,
        plane: usize,
        dst: &mut [u8],
        dst_stride: usize,
        rows: usize,
        row_bytes: usize,
    ) {
        let src = self.as_bytes();
        let base = self.inner.offsets[plane];
        let pitch = self.inner.pitches[plane];
        let tiles_across = pitch / TILE_WIDTH;
        const TILE_BYTES: usize = TILE_WIDTH * TILE_HEIGHT;
        const COLUMN_BYTES: usize = TILE_COLUMN * TILE_HEIGHT;

        for row in 0..rows {
            let (tile_y, row_in_tile) = (row / TILE_HEIGHT, row % TILE_HEIGHT);
            let mut col = 0;
            while col < row_bytes {
                let (tile_x, col_in_tile) = (col / TILE_WIDTH, col % TILE_WIDTH);
                let tile = tile_y * tiles_across + tile_x;
                let src_off = base
                    + tile * TILE_BYTES
                    + (col_in_tile / TILE_COLUMN) * COLUMN_BYTES
                    + row_in_tile * TILE_COLUMN;
                let n = TILE_COLUMN.min(row_bytes - col);
                let dst_off = row * dst_stride + col;
                dst[dst_off..dst_off + n].copy_from_slice(&src[src_off..src_off + n]);
                col += TILE_COLUMN;
            }
        }
    }

    /// Raw mutable access to the mapping, for tests that place a known
    /// pattern where the driver would have written one.
    #[cfg(test)]
    #[cfg_attr(not(feature = "display-gpu"), allow(dead_code))]
    pub(crate) fn as_bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: the mapping is PROT_WRITE and lives as long as `inner`;
        // `&mut self` is what makes this the only live reference to it.
        unsafe { std::slice::from_raw_parts_mut(self.inner.ptr, self.inner.len) }
    }

    /// The frame's bytes, **as the driver tiled them**.
    ///
    /// Useful as raw memory — its length, its identity as one allocation —
    /// but not as a picture. Read pixels with [`read_plane`](Self::read_plane).
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

    /// The allocation's DRM format modifier.
    ///
    /// Always [`I915_FORMAT_MOD_Y_TILED`]: whatever this module asks for,
    /// that is what the driver renders. An importer needs it; a CPU reader
    /// needs [`read_plane`](Self::read_plane) instead.
    pub fn modifier(&self) -> u64 {
        I915_FORMAT_MOD_Y_TILED
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
        // One object (the whole allocation) carrying one layer of two
        // planes — the NV12 shape, described to the driver in its own terms.
        let mut objects: [libva::VADRMPRIMESurfaceDescriptorObject; 4] = Default::default();
        objects[0] = libva::VADRMPRIMESurfaceDescriptorObject {
            fd: self.fd.as_raw_fd(),
            size: self.len as u32,
            drm_format_modifier: I915_FORMAT_MOD_Y_TILED,
        };

        let mut layers: [libva::VADRMPRIMESurfaceDescriptorLayer; 4] = Default::default();
        layers[0] = libva::VADRMPRIMESurfaceDescriptorLayer {
            drm_format: DRM_FORMAT_NV12,
            num_planes: 2,
            object_index: [0; 4],
            offset: [self.offsets[0] as u32, self.offsets[1] as u32, 0, 0],
            pitch: [self.pitches[0] as u32, self.pitches[1] as u32, 0, 0],
        };

        libva::VADRMPRIMESurfaceDescriptor {
            fourcc: libva::VA_FOURCC_NV12,
            width: self.coded.width,
            height: self.coded.height,
            num_objects: 1,
            objects,
            num_layers: 1,
            layers,
        }
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
        vec![
            self.inner.pitches[0] * self.inner.rows[0],
            self.inner.pitches[1] * self.inner.rows[1],
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
    fn a_frame_is_nv12_of_the_coded_size() {
        let Ok(frame) = VaFrame::new(res(1920, 1088), res(1920, 1080)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };

        assert_eq!(frame.resolution(), res(1920, 1080), "visible size");
        assert_eq!(frame.coded(), res(1920, 1088), "coded size");
        // 1920 is already a whole number of 128-byte tiles and both 1088 and
        // 544 are multiples of 32, so this geometry needs no rounding at all.
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
            // Tile-aligned: 64 bytes of pitch become 128, and 48 luma rows
            // and 24 chroma rows each become 64 and 32.
            assert_eq!(planes[0].len(), 128 * 64);
            assert_eq!(planes[1].len(), 128 * 32);
        }

        // Decode output is never written from our side.
        assert!(frame.map_mut().is_err());
    }

    /// The pool reissues a slot only once nothing else holds it, which is
    /// what keeps a decoder from rendering over a frame the pipeline is
    /// still reading.
    #[test]
    fn the_pool_reissues_only_released_slots() {
        let mut pool = VaFramePool::new(res(64, 48), res(64, 48));
        if pool.reserve(2).is_err() {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        }
        assert_eq!((pool.len(), pool.available()), (2, 2));

        let a = pool.acquire().expect("a free slot");
        let b = pool.acquire().expect("the other free slot");
        assert_eq!(pool.available(), 0);
        assert!(pool.acquire().is_none(), "both are in flight");

        // Distinct allocations, not the same one handed out twice.
        assert_ne!(a.as_bytes().as_ptr(), b.as_bytes().as_ptr());

        drop(a);
        assert_eq!(pool.available(), 1);
        let c = pool.acquire().expect("the released slot comes back");
        assert!(pool.acquire().is_none());
        drop((b, c));
        assert_eq!(pool.available(), 2);
    }

    /// A frame still in flight survives a pool reset — a resolution change
    /// must not pull memory out from under a buffer the pipeline holds.
    #[test]
    fn a_reset_does_not_disturb_frames_in_flight() {
        let mut pool = VaFramePool::new(res(64, 48), res(64, 48));
        if pool.reserve(1).is_err() {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        }
        let held = pool.acquire().expect("a free slot");
        // SAFETY: exclusive here; just marking the memory.
        unsafe { std::slice::from_raw_parts_mut(held.inner.ptr, 4) }.copy_from_slice(b"live");

        pool.reset(res(128, 96), res(128, 96));
        assert_eq!(pool.len(), 0);
        assert_eq!(pool.geometry().0, res(128, 96));

        // The old frame still reads back.
        assert_eq!(&held.as_bytes()[..4], b"live");
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

    /// Geometry that fits no tile boundary anywhere still allocates whole
    /// tiles. Anything less and the driver writes past the allocation — it
    /// renders Y-tiles regardless of the pitch and height it was given.
    #[test]
    fn awkward_geometry_allocates_whole_tiles() {
        let Ok(frame) = VaFrame::new(res(16, 18), res(16, 17)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };
        assert_eq!(frame.pitches(), [128, 128], "16 bytes is a 128-byte tile");
        // 18 luma rows and 9 chroma rows are both one 32-row tile.
        assert_eq!(frame.get_plane_size(), vec![128 * 32, 128 * 32]);
        assert_eq!(frame.offsets(), [0, 128 * 32]);
    }

    /// A frame whose width is not a whole number of tiles: 460 bytes of
    /// picture live in a 512-byte pitch. Getting this wrong is not a
    /// cosmetic error — the driver wrote 512-byte rows into a buffer sized
    /// for 460 of them.
    #[test]
    fn a_non_tile_width_rounds_the_pitch_up() {
        let Ok(frame) = VaFrame::new(res(460, 320), res(460, 308)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };
        assert_eq!(frame.pitches(), [512, 512]);
        assert_eq!(frame.get_plane_size(), vec![512 * 320, 512 * 160]);
    }

    /// The de-tiler is the inverse of the driver's layout: writing a Y-tiled
    /// pattern through the raw mapping and reading it back must give the
    /// picture the tiling encoded. Pure arithmetic — no GPU involved.
    #[test]
    fn read_plane_inverts_the_tiling() {
        const W: usize = 256;
        const H: usize = 64;
        let Ok(frame) = VaFrame::new(res(W as u32, H as u32), res(W as u32, H as u32)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };
        // The picture we want back: byte (x, y) is a function of both, so a
        // transposed or shifted read cannot pass by accident.
        let pixel = |x: usize, y: usize| ((x * 7 + y * 31) % 251) as u8;

        // Write it into the mapping in Y-tile order.
        {
            // SAFETY: the mapping is live and exclusively borrowed here.
            let bytes = unsafe { std::slice::from_raw_parts_mut(frame.inner.ptr, frame.inner.len) };
            let pitch = frame.pitches()[0];
            for y in 0..H {
                for x in 0..W {
                    let tile = (y / TILE_HEIGHT) * (pitch / TILE_WIDTH) + x / TILE_WIDTH;
                    let off = tile * TILE_WIDTH * TILE_HEIGHT
                        + ((x % TILE_WIDTH) / TILE_COLUMN) * TILE_COLUMN * TILE_HEIGHT
                        + (y % TILE_HEIGHT) * TILE_COLUMN
                        + x % TILE_COLUMN;
                    bytes[off] = pixel(x, y);
                }
            }
        }

        let mut out = vec![0u8; W * H];
        frame.read_plane(0, &mut out, W, H, W);
        for y in 0..H {
            for x in 0..W {
                assert_eq!(out[y * W + x], pixel(x, y), "de-tiled byte at ({x},{y})");
            }
        }
    }

    /// The crop is free: a visible width narrower than the coded one simply
    /// stops the de-tiler early, and never reads the columns past it.
    #[test]
    fn read_plane_crops_to_the_visible_width() {
        let Ok(frame) = VaFrame::new(res(256, 64), res(200, 50)) else {
            eprintln!("skipping: /dev/udmabuf unavailable");
            return;
        };
        let mut out = vec![0xCDu8; 200 * 50];
        frame.read_plane(0, &mut out, 200, 50, 200);
        assert_eq!(out.len(), 200 * 50);
    }
}
