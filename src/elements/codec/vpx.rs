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
    /// Caller-supplied frame buffers, when the link negotiated External
    /// (#198). `None` on VP8, which libvpx cannot do this for, and on any
    /// link that asked for CPU memory.
    pool: Option<std::sync::Arc<super::vpx_pool::VpxFramePool>>,
    /// What the downstream link negotiated; `None` before `prepare()`.
    negotiated: Option<crate::memory::MemoryType>,
    /// Whether `vpx_codec_set_frame_buffer_functions` has been attempted.
    /// It must be called before the first decode, and only once.
    frame_buffers_installed: bool,
    /// Rate-limits the fall-back-to-copy warning to one per decoder.
    warned_external_fallback: bool,
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
            pool: None,
            negotiated: None,
            frame_buffers_installed: false,
            warned_external_fallback: false,
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

    /// How many frame buffers this decoder has handed libvpx (#198), or `None`
    /// when it is decoding into the arena instead.
    ///
    /// The one number that says whether the zero-copy path is behaving: libvpx
    /// needs about sixteen for its own references, and anything far above that
    /// means the pipeline is holding frames longer than its budget allowed.
    pub fn pooled_frame_buffers(&self) -> Option<usize> {
        self.pool.as_ref().map(|p| p.len())
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

    /// Hand libvpx our own frame buffers, once, before the first decode.
    ///
    /// Only when the link negotiated [`MemoryType::External`] — supplying
    /// buffers costs a pool of at least sixteen full frames, which is pure
    /// overhead if the output is going to be copied into an arena slot anyway.
    ///
    /// VP8 is skipped without trying: `vpx_decoder.h` says the API "currently
    /// only works with VP9", and a VP8 context answers `VPX_CODEC_INCAPABLE`.
    /// A refusal from any other cause is not fatal either — the decoder simply
    /// keeps the packed copy path, which is always legal on this link because
    /// `external_preferred()` consumers also accept CPU memory.
    fn install_frame_buffers(&mut self) {
        if self.frame_buffers_installed {
            return;
        }
        self.frame_buffers_installed = true;

        if self.codec != VpxCodec::Vp9
            || self.negotiated != Some(crate::memory::MemoryType::External)
        {
            return;
        }

        let pool = super::vpx_pool::VpxFramePool::new();
        // SAFETY: `cb_priv` must outlive the codec context. It does: the pool
        // is a field of this struct, and `Drop for VpxDecoder` destroys the
        // context before any field is dropped.
        let err = unsafe {
            ffi::vpx_codec_set_frame_buffer_functions(
                &mut self.ctx,
                Some(super::vpx_pool::get_frame_buffer),
                Some(super::vpx_pool::release_frame_buffer),
                std::sync::Arc::as_ptr(&pool) as *mut std::ffi::c_void,
            )
        };
        if err != ffi::vpx_codec_err_t::VPX_CODEC_OK {
            tracing::debug!(
                "vpxdecoder: libvpx refused caller-supplied frame buffers ({err:?}); \
                 decoding into the arena instead"
            );
            return;
        }
        self.pool = Some(pool);
        tracing::debug!("vpxdecoder: decoding into caller-supplied frame buffers (#198)");
    }

    /// Emit a decoded frame without copying it (#198).
    ///
    /// The frame already lives in one of our pooled buffers, so the `Buffer`
    /// wraps that allocation directly and declares libvpx's real strides. The
    /// `PipelineHold` owner is what keeps the memory alive and what tells the
    /// pool the frame is off screen.
    ///
    /// `Ok(None)` whenever the planes do not sit inside the buffer the way
    /// [`PlaneLayout`](crate::format::PlaneLayout) requires — every plane
    /// addressable for its whole `stride * rows`, per #196 — and the caller
    /// falls back to the packed copy.
    fn image_to_external(
        &mut self,
        img: &ffi::vpx_image_t,
        source: Metadata,
    ) -> Result<Option<Buffer>> {
        let Some(pool) = self.pool.clone() else {
            return Ok(None);
        };
        if img.fb_priv.is_null() {
            return Ok(None);
        }
        let Some(owner) = pool.find(img.fb_priv) else {
            return Ok(None);
        };

        let (base, span) = owner.span();
        let (w, h) = (img.d_w as usize, img.d_h as usize);
        let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));

        let mut descs = Vec::with_capacity(3);
        for (plane, rows, row_bytes) in [(0usize, h, w), (1, ch, cw), (2, ch, cw)] {
            let ptr = img.planes[plane];
            let stride = img.stride[plane] as usize;
            if ptr.is_null() || stride < row_bytes {
                return Ok(None);
            }
            // The plane must start inside the buffer...
            if (ptr as usize) < base as usize {
                return Ok(None);
            }
            let offset = ptr as usize - base as usize;
            // ...and be addressable for its whole stride * rows. libvpx's
            // frame buffers carry a border below the last row, so this
            // normally holds with room to spare — but it is the invariant the
            // SIMD colorspace backend walks planes on, so check rather than
            // assume.
            if offset + stride.saturating_mul(rows) > span {
                return Ok(None);
            }
            descs.push(crate::format::PlaneDesc { offset, stride });
        }

        let dims = (w as u32, h as u32);
        if self.last_dims.is_some_and(|last| last != dims) {
            self.output.reset();
        }
        self.last_dims = Some(dims);

        let hold = pool.claim_for_pipeline(img.fb_priv).ok_or_else(|| {
            Error::Config("vpxdecoder: frame buffer vanished from the pool".into())
        })?;

        // SAFETY: `base..base + span` is the pooled allocation. `hold` keeps it
        // alive, and the pool will not hand it back to libvpx until `hold`
        // drops, so nothing writes to it while this buffer lives. External
        // buffers are never written by the pipeline.
        let slot =
            unsafe { crate::memory::ExternalSlot::new(base, span, hold, "vpx-frame-buffer") };

        self.frame_count += 1;
        let mut metadata = source;
        metadata.sequence = self.frames_out;
        self.frames_out += 1;
        metadata.set_video_planes(
            dims.0,
            dims.1,
            crate::format::PixelFormat::I420,
            crate::format::PlaneLayout::from_planes(&descs),
        );

        Ok(Some(Buffer::new(
            MemoryHandle::from_external(std::sync::Arc::new(slot)),
            metadata,
        )))
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
        // What the bitstream said, filled in from what the container said
        // where VPx left an axis unspecified (#265). `source` is the input
        // buffer's metadata, so the container's answer is already on it.
        let inherited = metadata.color_info().unwrap_or_default();
        metadata.set_color_info(image_color(img).or(inherited));
        Ok(Buffer::new(handle, metadata))
    }
}

/// What a decoded VPx image says about its own colour (#265).
///
/// libvpx has its own small vocabulary rather than H.273 codes, so unlike
/// dav1d this needs a real table. VP9 signals both axes in the uncompressed
/// header; VP8 signals neither, and answers `UNKNOWN` + studio range — which
/// is what the VP8 spec mandates anyway, so the range is honest and the
/// matrix falls through to the consumer's own guess.
fn image_color(img: &ffi::vpx_image_t) -> crate::format::ColorInfo {
    use crate::format::{ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, TransferFunction};
    use ffi::vpx_color_space::*;

    ColorInfo {
        matrix: match img.cs {
            VPX_CS_BT_709 => ColorMatrix::Bt709,
            VPX_CS_BT_601 | VPX_CS_SMPTE_170 => ColorMatrix::Bt601,
            VPX_CS_BT_2020 => ColorMatrix::Bt2020Ncl,
            // SMPTE 240M and sRGB have no variant here, and guessing a near
            // neighbour would be a lie about which coefficients apply.
            _ => ColorMatrix::Unspecified,
        },
        primaries: match img.cs {
            VPX_CS_BT_709 => ColorPrimaries::Bt709,
            VPX_CS_BT_601 => ColorPrimaries::Bt470bg,
            VPX_CS_SMPTE_170 => ColorPrimaries::Smpte170m,
            VPX_CS_BT_2020 => ColorPrimaries::Bt2020,
            _ => ColorPrimaries::Unspecified,
        },
        // libvpx conflates matrix, primaries and transfer into one `cs`, so
        // there is nothing separate to read for this axis.
        transfer: TransferFunction::Unspecified,
        range: match img.range {
            ffi::vpx_color_range::VPX_CR_FULL_RANGE => ColorRange::Full,
            ffi::vpx_color_range::VPX_CR_STUDIO_RANGE => ColorRange::Limited,
        },
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

    fn set_negotiated_memory(&mut self, memory: crate::memory::MemoryType) {
        self.negotiated = Some(memory);
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, MemoryLayout,
            VideoFormatCaps,
        };
        // #198: VP9 can decode straight into buffers we supply, so an opted-in
        // consumer takes the frame zero-copy and everyone else gets the packed
        // Cpu copy the solver downgrades to. VP8 cannot — libvpx's
        // caller-supplied frame buffers are VP9-only — so it says so rather
        // than advertising a capability it would silently fall back from.
        let memory = match self.codec {
            VpxCodec::Vp9 => MemoryCaps::external_preferred(),
            VpxCodec::Vp8 => MemoryCaps::cpu_only(),
        };
        ElementMediaCaps::new([FormatMemoryCap::new(
            FormatCaps::VideoRaw(VideoFormatCaps {
                width: CapsValue::Any,
                height: CapsValue::Any,
                // libvpx is asked for nothing else here; `image_to_buffer`
                // errors on any other `vpx_img_fmt`.
                pixel_format: CapsValue::Fixed(crate::format::PixelFormat::I420),
                framerate: CapsValue::Any,
                layout: MemoryLayout::NONE,
            }),
            memory,
        )])
    }

    // Decoders never skip an input (it may be a reference frame); on arena
    // exhaustion the *output copy* fails after libvpx already consumed the
    // data, and the executor sheds that single frame.
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        if self.pending.len() >= MAX_PENDING_METADATA {
            self.pending.pop_front();
        }
        self.pending.push_back(buffer.metadata().clone());

        self.install_frame_buffers();

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
            // SAFETY: img points at an image valid until the next decode call
            // on this context (or, with caller-supplied buffers, until libvpx
            // releases the frame buffer — which it cannot do before then).
            let img = unsafe { &*img };
            out = match self.image_to_external(img, source.clone())? {
                Some(buffer) => Some(buffer),
                None => {
                    if self.pool.is_some() && !self.warned_external_fallback {
                        self.warned_external_fallback = true;
                        tracing::warn!(
                            "vpxdecoder: decoded frame's planes do not satisfy the \
                             stride*rows layout contract; copying into the arena instead"
                        );
                    }
                    self.frame_count += 1;
                    Some(self.image_to_buffer(img, source)?)
                }
            };
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
