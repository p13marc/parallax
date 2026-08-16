//! VP8/VP9 software decoder using libvpx.
//!
//! libvpx is the reference implementation for both codecs — no viable
//! pure-Rust decoder exists for either, so this binds the system library,
//! like `Dav1dDecoder` does for AV1 and `OpusDecoder` for Opus.
//!
//! Requires the libvpx development package at build time:
//!
//! - **Fedora/RHEL**: `sudo dnf install libvpx-devel`
//! - **Debian/Ubuntu**: `sudo apt install libvpx-dev`
//! - **Arch**: `sudo pacman -S libvpx`
//!
//! Input is one whole frame per buffer, exactly as a WebM/MKV or MP4
//! demuxer emits it (VP9 superframes included). Output is packed I420 with
//! geometry in metadata; VP8/VP9 decode in display order, so timestamps map
//! one-to-one from input to output.
//!
//! ```rust,ignore
//! use parallax::elements::codec::VpxDecoder;
//!
//! let decoder = VpxDecoder::vp9()?; // or ::vp8()
//! pipeline.add_filter("decode", decoder);
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Element, ExecutionHints};
use crate::error::{Error, Result};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;

use std::collections::VecDeque;
use std::ffi::CStr;
use vpx_sys as ffi;

/// Which of the two libvpx codecs a [`VpxDecoder`] speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VpxCodec {
    /// VP8.
    Vp8,
    /// VP9.
    Vp9,
}

impl std::fmt::Display for VpxCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VpxCodec::Vp8 => write!(f, "VP8"),
            VpxCodec::Vp9 => write!(f, "VP9"),
        }
    }
}

/// Bound on the timing queue; mirrors `H264Decoder`'s reasoning — a stream
/// that never produces a picture must not grow it without bound.
const MAX_PENDING_METADATA: usize = 64;

/// VP8/VP9 software decoder (libvpx), usable directly as an [`Element`].
///
/// Emits packed I420 with `set_video_dims` metadata; a mid-stream
/// resolution change rebuilds the output arena. libvpx outputs at most one
/// *shown* frame per input frame and holds nothing back for reordering, so
/// `flush()` has nothing to drain.
pub struct VpxDecoder {
    ctx: ffi::vpx_codec_ctx_t,
    codec: VpxCodec,
    output: OutputArena,
    last_dims: Option<(u32, u32)>,
    /// Input timings awaiting their decoded frame (FIFO — no reordering).
    pending: VecDeque<Metadata>,
    frames_out: u64,
    frame_count: u64,
    bytes_decoded: u64,
    /// ACCURATE-seek clipping (#165): decoded frames below the current Time
    /// segment's start are dropped after decoding.
    clip: super::common::SegmentClip,
}

// The context is only ever touched from one element task at a time; the
// raw pointers inside vpx_codec_ctx_t are to decoder-owned state.
unsafe impl Send for VpxDecoder {}

impl VpxDecoder {
    /// Create a VP8 decoder.
    pub fn vp8() -> Result<Self> {
        // SAFETY: returns a pointer to a static interface table.
        Self::new(VpxCodec::Vp8, unsafe { ffi::vpx_codec_vp8_dx() })
    }

    /// Create a VP9 decoder.
    pub fn vp9() -> Result<Self> {
        // SAFETY: returns a pointer to a static interface table.
        Self::new(VpxCodec::Vp9, unsafe { ffi::vpx_codec_vp9_dx() })
    }

    /// Create a decoder for `codec`.
    pub fn for_codec(codec: VpxCodec) -> Result<Self> {
        match codec {
            VpxCodec::Vp8 => Self::vp8(),
            VpxCodec::Vp9 => Self::vp9(),
        }
    }

    fn new(codec: VpxCodec, iface: *const ffi::vpx_codec_iface) -> Result<Self> {
        // SAFETY: zeroed vpx_codec_ctx_t is the documented pre-init state;
        // a null config makes libvpx read stream parameters from the data.
        let mut ctx: ffi::vpx_codec_ctx_t = unsafe { std::mem::zeroed() };
        let err = unsafe {
            ffi::vpx_codec_dec_init_ver(
                &mut ctx,
                iface,
                std::ptr::null(),
                0,
                ffi::VPX_DECODER_ABI_VERSION as std::os::raw::c_int,
            )
        };
        if err != ffi::vpx_codec_err_t::VPX_CODEC_OK {
            return Err(Error::Config(format!(
                "failed to initialize libvpx {codec} decoder: {err:?}"
            )));
        }
        Ok(Self {
            ctx,
            codec,
            output: OutputArena::new(defaults::VIDEO_DECODER_SLOT_COUNT)
                .with_min_slot_size(4 * 1024 * 1024),
            last_dims: None,
            pending: VecDeque::new(),
            frames_out: 0,
            frame_count: 0,
            bytes_decoded: 0,
            clip: Default::default(),
        })
    }

    /// Which codec this decoder speaks.
    pub fn codec(&self) -> VpxCodec {
        self.codec
    }

    /// Frames decoded so far.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Compressed bytes consumed so far.
    pub fn bytes_decoded(&self) -> u64 {
        self.bytes_decoded
    }

    fn codec_error(&self, what: &str) -> Error {
        // SAFETY: ctx is initialized; vpx_codec_error returns a static or
        // context-owned NUL-terminated string.
        let detail = unsafe {
            let ptr = ffi::vpx_codec_error(&self.ctx);
            if ptr.is_null() {
                "unknown".into()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Error::Config(format!("libvpx {}: {what}: {detail}", self.codec))
    }

    /// Copy a decoded image into a packed I420 output buffer.
    fn image_to_buffer(&mut self, img: &ffi::vpx_image_t, source: Metadata) -> Result<Buffer> {
        if img.fmt != ffi::vpx_img_fmt::VPX_IMG_FMT_I420 {
            return Err(Error::Config(format!(
                "libvpx {}: unsupported output format {:?} (only 8-bit 4:2:0 is wired up)",
                self.codec, img.fmt
            )));
        }
        let (w, h) = (img.d_w as usize, img.d_h as usize);
        let dims = (w as u32, h as u32);
        if self.last_dims.is_some_and(|last| last != dims) {
            tracing::info!(
                "vpxdecoder: resolution changed to {}x{}, rebuilding the output arena",
                dims.0,
                dims.1
            );
            self.output.reset();
        }
        self.last_dims = Some(dims);

        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
        let total = w * h + 2 * cw * ch;
        let mut slot = self.output.acquire(total, "vpxdecoder")?;
        {
            let dst = &mut slot.data_mut()[..total];
            // SAFETY: libvpx guarantees each plane pointer covers
            // stride × rows bytes for the image it just returned.
            let mut off = 0;
            for (plane, rows, cols) in [(0usize, h, w), (1, ch, cw), (2, ch, cw)] {
                let stride = img.stride[plane] as usize;
                let src = img.planes[plane];
                for row in 0..rows {
                    let src_row =
                        unsafe { std::slice::from_raw_parts(src.add(row * stride), cols) };
                    dst[off..off + cols].copy_from_slice(src_row);
                    off += cols;
                }
            }
        }
        let handle = MemoryHandle::with_len(slot, total);

        let mut metadata = source;
        metadata.sequence = self.frames_out;
        self.frames_out += 1;
        metadata.set_video_dims(dims.0, dims.1, crate::format::PixelFormat::I420);
        Ok(Buffer::new(handle, metadata))
    }
}

impl Drop for VpxDecoder {
    fn drop(&mut self) {
        // SAFETY: ctx was initialized in new(); destroy exactly once.
        unsafe {
            ffi::vpx_codec_destroy(&mut self.ctx);
        }
    }
}

impl Element for VpxDecoder {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    // Decoders never skip an input (it may be a reference frame); on arena
    // exhaustion the *output copy* fails after libvpx already consumed the
    // data, and the executor sheds that single frame.
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        if self.pending.len() >= MAX_PENDING_METADATA {
            self.pending.pop_front();
        }
        self.pending.push_back(buffer.metadata().clone());

        let data = buffer.as_bytes();
        self.bytes_decoded += data.len() as u64;
        // SAFETY: data/len describe a live slice; no user_priv; deadline 0
        // (decode without giving libvpx a time budget).
        let err = unsafe {
            ffi::vpx_codec_decode(
                &mut self.ctx,
                data.as_ptr(),
                data.len() as std::os::raw::c_uint,
                std::ptr::null_mut(),
                0,
            )
        };
        if err != ffi::vpx_codec_err_t::VPX_CODEC_OK {
            return Err(self.codec_error("decode failed"));
        }

        // At most one *shown* frame comes back per input (a VP9 superframe
        // carries hidden alt-refs plus one shown frame).
        let mut iter: ffi::vpx_codec_iter_t = std::ptr::null();
        let mut out = None;
        loop {
            // SAFETY: ctx valid; iter is the opaque cursor this API expects.
            let img = unsafe { ffi::vpx_codec_get_frame(&mut self.ctx, &mut iter) };
            if img.is_null() {
                break;
            }
            let source = self.pending.pop_front().unwrap_or_default();
            self.frame_count += 1;
            // SAFETY: img points at a decoder-owned image, valid until the
            // next decode call on this context.
            out = Some(self.image_to_buffer(unsafe { &*img }, source)?);
        }
        // ACCURATE clipping (#165): decoded, but out-of-segment.
        if let Some(b) = &out
            && self.clip.clips(b.metadata().pts)
        {
            return Ok(None);
        }
        Ok(out)
    }

    fn handle_downstream_event(
        &mut self,
        event: crate::event::Event,
    ) -> Option<crate::event::Event> {
        self.clip.observe(&event);
        Some(event)
    }

    // libvpx (without frame-parallel mode) holds no shown frames back, so
    // there is nothing to drain at EOS.
    fn flush(&mut self) -> Result<Option<Buffer>> {
        Ok(None)
    }

    fn name(&self) -> &str {
        match self.codec {
            VpxCodec::Vp8 => "vp8decoder",
            VpxCodec::Vp9 => "vp9decoder",
        }
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::native() // Native code (FFI), might crash on bad input
    }
}

impl std::fmt::Debug for VpxDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VpxDecoder")
            .field("codec", &self.codec)
            .field("frame_count", &self.frame_count)
            .field("last_dims", &self.last_dims)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construct_both_codecs() {
        assert_eq!(VpxDecoder::vp8().unwrap().codec(), VpxCodec::Vp8);
        assert_eq!(VpxDecoder::vp9().unwrap().codec(), VpxCodec::Vp9);
        assert_eq!(
            VpxDecoder::for_codec(VpxCodec::Vp9).unwrap().name(),
            "vp9decoder"
        );
    }

    #[test]
    fn garbage_input_errors_without_crashing() {
        use crate::memory::SharedArena;

        let data = [0xde, 0xad, 0xbe, 0xef];
        let arena = SharedArena::new(data.len(), 2).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..data.len()].copy_from_slice(&data);
        let buf = Buffer::new(
            MemoryHandle::with_len(slot, data.len()),
            Metadata::default(),
        );

        let mut dec = VpxDecoder::vp9().unwrap();
        assert!(dec.process(buf).is_err());
    }
}
