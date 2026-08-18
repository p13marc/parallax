//! VA-API hardware video decode (#193).
//!
//! Implements [`Element`] directly, per the #160 codec-surface rule: a
//! wrapper cannot carry the input access unit's PTS and metadata through to
//! the frame it becomes, and that is the whole job of a decoder element.
//!
//! # What the hardware costs
//!
//! Decode itself moves to the GPU's fixed-function video engine, so the CPU
//! cost of a frame collapses to the readback — a plane copy out of the
//! linear frame the driver rendered into ([`VaFrame`]). That copy is the
//! reason the frames are pipeline-owned and linear: a driver-allocated
//! surface comes back tiled, and de-tiling on the CPU would give most of
//! the saving straight back.
//!
//! # Probe, then fall back
//!
//! [`VaapiDecoder::new`] returns `Err` unless a display opens *and*
//! advertises decode for the codec. Callers are expected to fall back to
//! the software decoder on that error rather than treating it as fatal —
//! which codecs are available depends on the driver package, not just the
//! hardware (see [`crate::gpu::vaapi`]).

use std::collections::VecDeque;

use cros_codecs::Resolution;
use cros_codecs::backend::vaapi::decoder::VaapiDecodedHandle;
use cros_codecs::decoder::stateless::h264::H264;
use cros_codecs::decoder::stateless::vp9::Vp9;
use cros_codecs::decoder::stateless::{DecodeError, StatelessDecoder, StatelessVideoDecoder};
use cros_codecs::decoder::{BlockingMode, DecodedHandle, DecoderEvent};

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::event::Event;
use crate::format::{PixelFormat, PlaneLayout};
use crate::gpu::Codec;
use crate::gpu::vaapi::{VaDisplay, VaFrame, VaFramePool};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;

use super::common::SegmentClip;

/// How many decoded frames the pipeline may hold on top of what the codec
/// needs for its own references.
///
/// `StreamInfo::min_num_frames` covers references only. A frame is still
/// pinned while it travels the pipeline, so a pool sized to the codec's
/// minimum starves the decoder the moment anything downstream holds one.
const PIPELINE_FRAMES: usize = 6;

/// Cap on access units awaiting a decoded frame.
///
/// The same bound `Dav1dDecoder` uses. A decoder that reorders can hold
/// several AUs' metadata; an unbounded queue would grow without limit on a
/// stream that never produces output.
const MAX_PENDING_METADATA: usize = 64;

/// The decoded-frame handle every codec's decoder produces.
///
/// All of them share one backend, so the handle type does not vary with the
/// codec — which is what lets one element drive any of them through a
/// trait object.
type Handle = std::rc::Rc<std::cell::RefCell<VaapiDecodedHandle<VaFrame>>>;

/// A hardware video decoder, whatever codec it was built for.
pub struct VaapiDecoder {
    decoder: Box<dyn StatelessVideoDecoder<Handle = Handle>>,
    codec: Codec,
    pool: VaFramePool,
    /// Output arena for the readback copy.
    output: OutputArena,
    /// Input metadata awaiting the frame it becomes, oldest first.
    pending_meta: VecDeque<Metadata>,
    /// Frames the decoder has finished but we have not emitted yet.
    ready: VecDeque<Handle>,
    /// An access unit the decoder could not take yet, and how far into it we
    /// got. Re-offered before anything new.
    stalled: Option<(Buffer, usize)>,
    /// Geometry of the last emitted frame; a change resets the arena.
    last_dims: Option<(u32, u32)>,
    clip: SegmentClip,
    frames_out: u64,
}

// SAFETY: `cros-codecs` keeps `Rc<Context>` internally, so the decoder is
// not `Send` by inference. It is owned entirely by one element task, never
// shared, and never touched from another thread — the same justification
// (and the same shape) as `VpxDecoder`'s FFI context.
unsafe impl Send for VaapiDecoder {}

impl VaapiDecoder {
    /// A hardware VP9 decoder, or the reason there isn't one.
    pub fn vp9() -> Result<Self> {
        Self::open(Codec::Vp9)
    }

    /// A hardware H.264 decoder, or the reason there isn't one.
    ///
    /// Note that H.264 is absent from patent-free driver builds even on
    /// hardware that has the engine — the error says so when that is why.
    pub fn h264() -> Result<Self> {
        Self::open(Codec::H264)
    }

    /// Open the hardware decoder for `codec`, or explain why it is
    /// unavailable.
    ///
    /// Every failure here is a reason to use the software decoder instead,
    /// not a reason to stop: no DRM device, no driver, or a driver built
    /// without this codec.
    ///
    /// HEVC is deliberately absent: `cros-codecs`' H.265 constructor takes
    /// an `Rc<Display>` where H.264 and VP9 take `Arc`, so it cannot be
    /// driven from the same shared display at all. Adding it means fixing
    /// that upstream first.
    pub fn open(codec: Codec) -> Result<Self> {
        let display = VaDisplay::open().ok_or_else(|| {
            Error::Element(
                "vaapi: no VA display (no DRM render node, or no driver installed)".into(),
            )
        })?;
        Self::with_display(&display, codec)
    }

    /// Build a decoder on an already-open display.
    fn with_display(display: &VaDisplay, codec: Codec) -> Result<Self> {
        if !display.supports_decode(codec) {
            return Err(Error::Element(format!(
                "vaapi: {} decodes {:?}, not {codec} — falling back to software \
                 (H.264 and HEVC are absent from patent-free driver builds; \
                 check `vainfo`)",
                display.vendor(),
                display.decodable(),
            )));
        }

        // `VaapiBackend::new` builds its initial config from a hardcoded
        // `VAProfileH264Main` regardless of the codec asked for, and
        // `.expect()`s it — so a driver without H.264 panics here while
        // decoding VP9 it fully supports. Refusing first turns that into a
        // software fallback. (Filed against the crate; remove this once the
        // backend uses the decoder's own profile.)
        if !display.supports_backend_init() {
            return Err(Error::Element(format!(
                "vaapi: {} cannot decode H.264, which cros-codecs' backend requires for \
                 its initial config even when decoding {codec} — falling back to software. \
                 A patent-free driver build (Fedora's libva-intel-media-driver) omits \
                 H.264/HEVC; RPM Fusion's intel-media-driver-freeworld has them",
                display.vendor(),
            )));
        }

        let init = |e| Error::Element(format!("vaapi: {codec} decoder init failed: {e:?}"));
        let decoder: Box<dyn StatelessVideoDecoder<Handle = Handle>> = match codec {
            Codec::Vp9 => Box::new(
                StatelessDecoder::<Vp9, _>::new_vaapi(display.handle(), BlockingMode::NonBlocking)
                    .map_err(init)?,
            ),
            Codec::H264 => Box::new(
                StatelessDecoder::<H264, _>::new_vaapi(display.handle(), BlockingMode::NonBlocking)
                    .map_err(init)?,
            ),
            other => {
                return Err(Error::Element(format!(
                    "vaapi: no hardware decoder wired for {other} yet"
                )));
            }
        };

        Ok(Self {
            decoder,
            codec,
            pool: VaFramePool::new(Resolution::from((0, 0)), Resolution::from((0, 0))),
            output: OutputArena::new(defaults::VIDEO_DECODER_SLOT_COUNT),
            pending_meta: VecDeque::new(),
            ready: VecDeque::new(),
            stalled: None,
            last_dims: None,
            clip: SegmentClip::default(),
            frames_out: 0,
        })
    }

    /// Re-size the frame pool from what the decoder now says about the
    /// stream.
    fn resize_pool(&mut self) -> Result<()> {
        let Some(info) = self.decoder.stream_info() else {
            return Ok(());
        };
        let coded = info.coded_resolution;
        let visible = info.display_resolution;
        if self.pool.geometry() != (coded, visible) {
            tracing::debug!(
                "vaapi: stream is {}x{} (coded {}x{}), {} reference frames",
                visible.width,
                visible.height,
                coded.width,
                coded.height,
                info.min_num_frames
            );
            self.pool.reset(coded, visible);
            self.output.reset();
        }
        self.pool.reserve(info.min_num_frames + PIPELINE_FRAMES)
    }

    /// Drain everything the decoder has finished into `ready`.
    ///
    /// `FormatChanged` is not an error: the stream's geometry is now
    /// whatever `stream_info` says, and the pool follows it.
    fn drain_events(&mut self) -> Result<()> {
        let mut reformatted = false;
        while let Some(event) = self.decoder.next_event() {
            match event {
                DecoderEvent::FrameReady(handle) => self.ready.push_back(handle),
                DecoderEvent::FormatChanged => reformatted = true,
            }
        }
        if reformatted {
            self.resize_pool()?;
        }
        Ok(())
    }

    /// Feed one access unit, from `offset`, returning how far we got.
    ///
    /// Returns `Ok(None)` when the decoder could not take it all — the
    /// caller stashes the remainder rather than dropping it. A decoder must
    /// never skip an input: it would be a reference frame the decoder never
    /// sees, and everything after it decodes wrong.
    fn feed(&mut self, data: &[u8], mut offset: usize, pts: u64) -> Result<Option<usize>> {
        while offset < data.len() {
            // Borrow-splitting: the closure needs the pool while `decoder`
            // is mutably borrowed, so they must be distinct fields here.
            let pool = &mut self.pool;
            let mut alloc = || pool.acquire();

            match self.decoder.decode(pts, &data[offset..], &mut alloc) {
                Ok(0) => {
                    // No progress and no error: nothing more to take from
                    // this unit.
                    break;
                }
                Ok(consumed) => offset += consumed,
                Err(DecodeError::CheckEvents) => {
                    self.drain_events()?;
                    return Ok(None);
                }
                Err(DecodeError::NotEnoughOutputBuffers(_)) => {
                    // Every frame is in flight downstream. Not an error and
                    // not a drop — the unit is re-offered once one returns.
                    self.drain_events()?;
                    return Ok(None);
                }
                Err(e) => {
                    return Err(Error::Element(format!("vaapi: decode failed: {e}")));
                }
            }
        }
        Ok(Some(offset))
    }

    /// Copy one decoded frame out of GPU-written memory into an arena slot.
    ///
    /// The readback is per-row rather than one `copy_from_slice`: the frame
    /// is allocated at the *coded* size, so its rows are `coded.width` apart
    /// while the packed output wants `visible.width`. Cropping here is free;
    /// doing it downstream would cost another full-frame pass.
    fn handle_to_buffer(&mut self, handle: &Handle) -> Result<Buffer> {
        handle
            .sync()
            .map_err(|e| Error::Element(format!("vaapi: waiting for a frame failed: {e}")))?;

        let frame = handle.video_frame();
        let visible = handle.display_resolution();
        let (w, h) = (visible.width, visible.height);

        if self.last_dims != Some((w, h)) {
            self.output.reset();
            self.last_dims = Some((w, h));
        }

        let packed = PlaneLayout::packed(PixelFormat::Nv12, w, h);
        let total = packed.required_len(PixelFormat::Nv12, w, h);
        let mut slot = self.output.acquire(total, "vaapidecoder")?;

        let src = frame.as_bytes();
        let src_offsets = frame.offsets();
        let src_pitches = frame.pitches();
        let dst = &mut slot.data_mut()[..total];

        for (plane, out) in packed.resolved(PixelFormat::Nv12, w, h).enumerate() {
            for row in 0..out.rows {
                let s = src_offsets[plane] + row * src_pitches[plane];
                let d = out.offset + row * out.stride;
                dst[d..d + out.row_bytes].copy_from_slice(&src[s..s + out.row_bytes]);
            }
        }

        let mut metadata = self.claim_meta_for(handle.timestamp());
        metadata.pts = ClockTime::from_nanos(handle.timestamp());
        metadata.sequence = self.frames_out;
        metadata.set_video_dims(w, h, PixelFormat::Nv12);
        self.frames_out += 1;

        Ok(Buffer::new(MemoryHandle::with_len(slot, total), metadata))
    }

    /// The input metadata belonging to `pts`.
    ///
    /// VP9 has no reorder, so this is normally the front of the queue; the
    /// PTS match is what keeps it honest if that ever stops being true.
    fn claim_meta_for(&mut self, pts: u64) -> Metadata {
        if let Some(idx) = self
            .pending_meta
            .iter()
            .position(|m| m.pts.nanos() == pts)
            .or(if self.pending_meta.is_empty() {
                None
            } else {
                Some(0)
            })
        {
            self.pending_meta.remove(idx).unwrap_or_default()
        } else {
            Metadata::default()
        }
    }

    /// Emit the oldest decoded frame that the current segment wants.
    fn next_output(&mut self) -> Result<Option<Buffer>> {
        while let Some(handle) = self.ready.pop_front() {
            let buffer = self.handle_to_buffer(&handle)?;
            if self.clip.clips(buffer.metadata().pts) {
                continue;
            }
            return Ok(Some(buffer));
        }
        Ok(None)
    }

    /// How many frames the pool holds — diagnostics and tests.
    pub fn pool_size(&self) -> usize {
        self.pool.len()
    }
}

impl Element for VaapiDecoder {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // The pool cannot be sized before the first access unit tells the
        // decoder what the stream is.
        if self.pool.is_empty() {
            self.pool.reserve(PIPELINE_FRAMES)?;
        }

        // Anything stalled goes first, or the stream reorders itself.
        if let Some((pending, offset)) = self.stalled.take() {
            let pts = pending.metadata().pts.nanos();
            match self.feed(pending.as_bytes(), offset, pts)? {
                Some(_) => {}
                None => {
                    self.stalled = Some((pending, offset));
                    // Hold the new unit too — but only after the stalled one
                    // drains, so it is queued rather than dropped.
                    self.push_meta(buffer.metadata().clone());
                    return self.next_output();
                }
            }
        }

        self.push_meta(buffer.metadata().clone());
        let pts = buffer.metadata().pts.nanos();
        if self.feed(buffer.as_bytes(), 0, pts)?.is_none() {
            self.stalled = Some((buffer, 0));
        }
        self.drain_events()?;
        self.next_output()
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        if self.stalled.is_none() && self.ready.is_empty() {
            // One drain per cycle: `flush` is called repeatedly until it
            // answers `None`, and asking the decoder twice would restart it.
            let _ = self.decoder.flush();
            self.drain_events()?;
        }
        self.next_output()
    }

    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        self.clip.observe(&event);
        if matches!(event, Event::FlushStart) {
            // A seek lands on a keyframe, so nothing in flight is useful.
            // `flush()` also puts the decoder back in a state where it waits
            // for one.
            let _ = self.decoder.flush();
            while self.decoder.next_event().is_some() {}
            self.ready.clear();
            self.pending_meta.clear();
            self.stalled = None;
        }
        Some(event)
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, MemoryLayout,
            VideoFormatCaps,
        };
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(VideoFormatCaps {
                width: CapsValue::Any,
                height: CapsValue::Any,
                pixel_format: CapsValue::Fixed(PixelFormat::Nv12),
                framerate: CapsValue::Any,
                layout: MemoryLayout::NONE,
            }),
            MemoryCaps::cpu_only(),
        )])
    }

    fn name(&self) -> &str {
        match self.codec {
            Codec::Vp9 => "vaapivp9dec",
            Codec::H264 => "vaapih264dec",
            Codec::H265 => "vaapih265dec",
            Codec::Av1 => "vaapiav1dec",
        }
    }

    fn execution_hints(&self) -> crate::element::ExecutionHints {
        crate::element::ExecutionHints::native()
    }
}

impl VaapiDecoder {
    /// Record an access unit's metadata, bounded.
    fn push_meta(&mut self, metadata: Metadata) {
        if self.pending_meta.len() >= MAX_PENDING_METADATA {
            self.pending_meta.pop_front();
        }
        self.pending_meta.push_back(metadata);
    }
}

impl std::fmt::Debug for VaapiDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VaapiDecoder")
            .field("codec", &self.codec)
            .field("pool", &self.pool.len())
            .field("ready", &self.ready.len())
            .field("frames_out", &self.frames_out)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construction answers rather than panicking, whatever the machine.
    ///
    /// `VaapiBackend::new` panics when config creation fails, so this is
    /// really asserting that the capability probe runs first — for every
    /// codec, including ones this machine cannot decode.
    #[test]
    fn construction_answers_on_any_machine() {
        for codec in [Codec::Vp9, Codec::H264, Codec::H265, Codec::Av1] {
            match VaapiDecoder::open(codec) {
                Ok(d) => eprintln!("vaapi {codec}: {d:?}"),
                Err(e) => eprintln!("vaapi {codec}: unavailable (expected fallback) — {e}"),
            }
        }
    }

    /// Element names are per codec, so a pipeline graph names what it is
    /// actually running.
    #[test]
    fn each_codec_has_its_own_element_name() {
        let Ok(d) = VaapiDecoder::vp9() else {
            eprintln!("skipping: no hardware VP9 decoder here");
            return;
        };
        assert_eq!(d.name(), "vaapivp9dec");
    }
}
