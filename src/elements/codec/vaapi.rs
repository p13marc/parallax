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
use cros_codecs::decoder::stateless::vp8::Vp8;
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

/// Cap on access units the decoder has not been able to swallow yet.
///
/// A stall is normally one `NotEnoughOutputBuffers` cleared by the next
/// `drain_events`, so this is never approached in a healthy pipeline. It is
/// a bound rather than an unbounded queue because the alternative to failing
/// is silently growing, and it is deliberately *not* handled by dropping:
/// a skipped access unit is a reference frame the decoder never sees, and
/// everything after it decodes wrong until the next keyframe.
const MAX_PENDING_INPUT: usize = 8;

/// How many times an access unit may make no progress before its tail is
/// abandoned.
///
/// `decode` returning `Ok(0)` without an error is not something the wired
/// codecs do — VP9 reports the whole bitstream consumed, H.264 at least one
/// NAL unit — so this is a spin guard, not a code path with a purpose.
const MAX_ZERO_PROGRESS: u32 = 2;

/// Consecutive undecodable access units tolerated before giving up.
///
/// Matches `H264Decoder`'s policy for `dsRefLost`: after a seek into an
/// open GOP the leading pictures reference frames that were never decoded,
/// and refusing them is correct behaviour, not failure. Erring only after a
/// long run of them keeps a corrupt stream from looping forever.
const MAX_CONSECUTIVE_REFUSALS: u32 = 300;

/// The decoded-frame handle every codec's decoder produces.
///
/// All of them share one backend, so the handle type does not vary with the
/// codec — which is what lets one element drive any of them through a
/// trait object.
type Handle = std::rc::Rc<std::cell::RefCell<VaapiDecodedHandle<VaFrame>>>;

/// One access unit and how far into it the decoder has got.
struct PendingAu {
    buffer: Buffer,
    offset: usize,
    /// How many times in a row `decode` reported no progress on it.
    zero_progress: u32,
}

/// What [`drive_input`] managed to do with the queue.
#[derive(Debug, PartialEq, Eq)]
enum DriveOutcome {
    /// Every queued access unit was consumed.
    Drained,
    /// The decoder could take no more for now; the queue front is where to
    /// resume. Not an error: the caller drains events and tries again.
    Stalled,
}

/// Feed queued access units to `decode`, front first, until one stalls.
///
/// Split out from the element so the offset and queue bookkeeping — which is
/// where all three of the data-loss bugs lived — can be tested without a GPU.
/// `decode` is the raw call: bytes and PTS in, bytes-consumed out.
fn drive_input<F>(
    pending: &mut VecDeque<PendingAu>,
    refusals: &mut u32,
    mut decode: F,
) -> Result<DriveOutcome>
where
    F: FnMut(&[u8], u64) -> std::result::Result<usize, DecodeError>,
{
    while let Some(au) = pending.front_mut() {
        let pts = au.buffer.metadata().pts.nanos();
        let data = au.buffer.as_bytes();
        if au.offset >= data.len() {
            pending.pop_front();
            continue;
        }

        match decode(&data[au.offset..], pts) {
            Ok(0) => {
                au.zero_progress += 1;
                if au.zero_progress < MAX_ZERO_PROGRESS {
                    // Give it one more chance after the caller drains events;
                    // treating it as a stall keeps the unit queued.
                    return Ok(DriveOutcome::Stalled);
                }
                let left = data.len() - au.offset;
                tracing::warn!(
                    "vaapi: decoder made no progress on {left} bytes of an access unit;                      abandoning its tail"
                );
                pending.pop_front();
            }
            Ok(consumed) => {
                // Advance *before* anything can return: the whole first bug
                // was reporting a stall while forgetting how far we had got,
                // so the already-consumed prefix was submitted again.
                au.offset += consumed;
                au.zero_progress = 0;
                *refusals = 0;
                if au.offset >= data.len() {
                    pending.pop_front();
                }
            }
            Err(DecodeError::CheckEvents) | Err(DecodeError::NotEnoughOutputBuffers(_)) => {
                // Back-pressure, not failure. The failing call consumed
                // nothing, so resuming at `offset` re-offers exactly the unit
                // that was refused — which is what the trait asks for.
                return Ok(DriveOutcome::Stalled);
            }
            Err(e @ DecodeError::ParseFrameError(_)) => {
                // An access unit the parser cannot make sense of. After a
                // seek into an open GOP that is expected, not broken, so it
                // is skipped like `H264Decoder` skips a lost reference —
                // fatal only if it never stops.
                *refusals += 1;
                if *refusals >= MAX_CONSECUTIVE_REFUSALS {
                    return Err(Error::Element(format!(
                        "vaapi: {MAX_CONSECUTIVE_REFUSALS} consecutive access units                          could not be parsed; last error: {e}"
                    )));
                }
                if refusals.is_power_of_two() {
                    tracing::warn!("vaapi: skipping an unparseable access unit ({e})");
                }
                pending.pop_front();
            }
            Err(e) => return Err(Error::Element(format!("vaapi: decode failed: {e}"))),
        }
    }
    Ok(DriveOutcome::Drained)
}

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
    /// Access units the decoder has not finished swallowing, oldest first.
    ///
    /// A single slot is not enough: while one unit is stalled the executor
    /// keeps delivering, and there is nowhere to hand a buffer back to — the
    /// only recoverable error the executor knows, `PoolExhausted`, makes it
    /// *shed* the buffer, which for a decoder is the one thing that must
    /// never happen.
    pending_input: VecDeque<PendingAu>,
    /// Consecutive access units the decoder refused to parse.
    refusals: u32,
    /// Whether the end-of-stream drain has already been asked for this cycle.
    eos_drained: bool,
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

    /// A hardware VP8 decoder, or the reason there isn't one.
    pub fn vp8() -> Result<Self> {
        Self::open(Codec::Vp8)
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
                "vaapi: {} decodes {:?}, not {codec} — falling back to software. \
                 Which codecs a driver offers is a packaging decision as much as a \
                 hardware one: patent-free builds omit H.264 and HEVC on silicon that \
                 has the engines. `vainfo` is the ground truth",
                display.vendor(),
                display.decodable(),
            )));
        }

        // Advertising a profile and building a config for it are different
        // questions, and only the second one is the one the decoder needs
        // answered. Asking it here keeps a driver that reports optimistically
        // from failing at the first frame instead of at construction, where a
        // caller can still choose software.
        if !display.decode_config_works(codec) {
            return Err(Error::Element(format!(
                "vaapi: {} advertises {codec} decode but will not create a config for it \
                 — falling back to software",
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
            Codec::Vp8 => Box::new(
                StatelessDecoder::<Vp8, _>::new_vaapi(display.handle(), BlockingMode::NonBlocking)
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
            pending_input: VecDeque::new(),
            refusals: 0,
            eos_drained: false,
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

    /// Push queued access units into the decoder until it stops taking them.
    ///
    /// Draining events between attempts is not bookkeeping — it is what
    /// releases the decoded frame the decoder is waiting for, so a stall that
    /// looks terminal often clears on the next pass. The loop stops as soon
    /// as a pass changes nothing, which is the only honest definition of "it
    /// really cannot take more right now".
    fn pump(&mut self) -> Result<()> {
        loop {
            let progress_marker = self
                .pending_input
                .front()
                .map(|au| (self.pending_input.len(), au.offset));

            // Borrow-splitting: the allocation closure needs the pool while
            // `decoder` is mutably borrowed, so they must be distinct fields.
            let (decoder, pool) = (&mut self.decoder, &mut self.pool);
            let outcome = drive_input(&mut self.pending_input, &mut self.refusals, |data, pts| {
                let mut alloc = || pool.acquire();
                decoder.decode(pts, data, &mut alloc)
            })?;
            self.drain_events()?;

            if outcome == DriveOutcome::Drained {
                return Ok(());
            }
            let after = self
                .pending_input
                .front()
                .map(|au| (self.pending_input.len(), au.offset));
            if after == progress_marker {
                // The pass moved nothing, so another one would not either.
                // The queue keeps what it holds; the next `process` or
                // `flush` will try again once downstream has released a frame.
                return Ok(());
            }
        }
    }

    /// Copy one decoded frame out of GPU-written memory into an arena slot.
    ///
    /// The copy goes through [`VaFrame::read_plane`] because the driver
    /// renders **Y-tiled** frames whatever modifier it is asked for, so the
    /// bytes are not rows. De-tiling moves the same volume of data as a
    /// straight copy would and crops coded-to-visible on the way, which is
    /// why it is still worth doing here rather than downstream.
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

        let dst = &mut slot.data_mut()[..total];
        for (plane, out) in packed.resolved(PixelFormat::Nv12, w, h).enumerate() {
            frame.read_plane(
                plane,
                &mut dst[out.offset..],
                out.stride,
                out.rows,
                out.row_bytes,
            );
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

    /// Access units are queued rather than dropped when the decoder stalls,
    /// so up to [`MAX_PENDING_INPUT`] upstream buffers can be pinned here.
    /// Declaring it is what sizes the *producer's* arena to survive it (#189).
    fn retained_buffers(&self) -> usize {
        MAX_PENDING_INPUT
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // No pool is reserved here on purpose. Frame geometry is not known
        // until the decoder has read a sequence header, and reserving before
        // that would ask for zero-sized frames. The decoder requests one only
        // after it reports `FormatChanged`, which `drain_events` turns into a
        // correctly-sized `resize_pool`; until then an allocation request
        // answers `None`, which is the ordinary back-pressure stall.

        if self.pending_input.len() >= MAX_PENDING_INPUT {
            return Err(Error::Element(format!(
                "vaapi: {MAX_PENDING_INPUT} access units are queued and the decoder has \
                 taken none of them — the frame pool ({} frames, {} free) is too small \
                 for what this graph holds downstream",
                self.pool.len(),
                self.pool.available(),
            )));
        }

        self.push_meta(buffer.metadata().clone());
        self.pending_input.push_back(PendingAu {
            buffer,
            offset: 0,
            zero_progress: 0,
        });
        self.eos_drained = false;
        self.pump()?;
        self.next_output()
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        // Whatever is still queued has to reach the decoder before the drain,
        // or the tail of the stream is simply lost.
        self.pump()?;

        if !self.eos_drained && self.pending_input.is_empty() && self.ready.is_empty() {
            // Once per cycle: `flush` is called repeatedly until it answers
            // `None`, and asking the decoder to drain twice restarts it.
            self.eos_drained = true;
            if !self.pending_input.is_empty() {
                tracing::warn!(
                    "vaapi: {} access units never reached the decoder before the drain",
                    self.pending_input.len()
                );
            }
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
            self.pending_input.clear();
            self.refusals = 0;
            self.eos_drained = false;
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
            Codec::Vp8 => "vaapivp8dec",
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

    /// An access unit of `len` bytes, PTS `pts`.
    fn au(arena: &crate::memory::SharedArena, len: usize, pts: u64) -> PendingAu {
        let slot = arena.acquire().expect("arena not exhausted");
        let mut metadata = Metadata::default();
        metadata.pts = ClockTime::from_nanos(pts);
        PendingAu {
            buffer: Buffer::new(MemoryHandle::with_len(slot, len), metadata),
            offset: 0,
            zero_progress: 0,
        }
    }

    fn queue(arena: &crate::memory::SharedArena, lens: &[usize]) -> VecDeque<PendingAu> {
        lens.iter()
            .enumerate()
            .map(|(i, len)| au(arena, *len, i as u64))
            .collect()
    }

    /// A stall must not forget how far the decoder got.
    ///
    /// The original bug reported the stall without the offset, so the caller
    /// re-stashed the pre-call one and the already-consumed prefix was
    /// submitted again. For H.264 that means re-sending an SPS, which can
    /// itself provoke the next stall — a livelock, not just waste.
    #[test]
    fn a_stall_resumes_where_it_stopped() {
        let arena = crate::memory::SharedArena::new(4096, 4).unwrap();
        let mut pending = queue(&arena, &[100]);
        let mut refusals = 0;
        let mut calls = 0;

        let outcome = drive_input(&mut pending, &mut refusals, |_data, _pts| {
            calls += 1;
            match calls {
                1 => Ok(40),
                _ => Err(DecodeError::CheckEvents),
            }
        })
        .unwrap();

        assert_eq!(outcome, DriveOutcome::Stalled);
        assert_eq!(pending.len(), 1, "the unit is kept, not dropped");
        assert_eq!(pending.front().unwrap().offset, 40, "progress is remembered");

        // Resuming must offer only the remaining 60 bytes.
        let mut seen = 0;
        drive_input(&mut pending, &mut refusals, |data, _pts| {
            seen = data.len();
            Ok(data.len())
        })
        .unwrap();
        assert_eq!(seen, 60, "resumed at the offset, not from the start");
        assert!(pending.is_empty());
    }

    /// A unit arriving while another is stalled is queued behind it, in
    /// order, and neither is lost.
    ///
    /// This is the bug that could not be fixed by returning an error: the
    /// executor's only recoverable error, `PoolExhausted`, makes it *shed*
    /// the buffer.
    #[test]
    fn a_second_access_unit_waits_its_turn() {
        let arena = crate::memory::SharedArena::new(4096, 4).unwrap();
        let mut pending = queue(&arena, &[10, 20]);
        let mut refusals = 0;

        // Everything stalls at first.
        let outcome = drive_input(&mut pending, &mut refusals, |_, _| {
            Err(DecodeError::NotEnoughOutputBuffers(1))
        })
        .unwrap();
        assert_eq!(outcome, DriveOutcome::Stalled);
        assert_eq!(pending.len(), 2, "both units still queued");

        // Then everything drains, oldest first.
        let mut order = Vec::new();
        let outcome = drive_input(&mut pending, &mut refusals, |data, pts| {
            order.push((pts, data.len()));
            Ok(data.len())
        })
        .unwrap();
        assert_eq!(outcome, DriveOutcome::Drained);
        assert_eq!(order, vec![(0, 10), (1, 20)], "decode order preserved");
        assert!(pending.is_empty());
    }

    /// `Ok(0)` neither spins forever nor silently abandons the unit on the
    /// first try: it gets one more pass, then gives up loudly.
    #[test]
    fn no_progress_gives_up_rather_than_spinning() {
        let arena = crate::memory::SharedArena::new(4096, 4).unwrap();
        let mut pending = queue(&arena, &[64]);
        let mut refusals = 0;

        let outcome = drive_input(&mut pending, &mut refusals, |_, _| Ok(0)).unwrap();
        assert_eq!(outcome, DriveOutcome::Stalled, "first zero is a stall");
        assert_eq!(pending.len(), 1);

        let outcome = drive_input(&mut pending, &mut refusals, |_, _| Ok(0)).unwrap();
        assert_eq!(outcome, DriveOutcome::Drained);
        assert!(pending.is_empty(), "the unit is dropped, not retried forever");
    }

    /// An unparseable unit is skipped like a lost reference, and only a long
    /// run of them is fatal.
    #[test]
    fn unparseable_units_are_skipped_then_eventually_fatal() {
        let arena = crate::memory::SharedArena::new(4096, 64).unwrap();
        let mut refusals = 0;

        let mut pending = queue(&arena, &[8]);
        let outcome = drive_input(&mut pending, &mut refusals, |_, _| {
            Err(DecodeError::ParseFrameError("bad NAL".to_string()))
        })
        .unwrap();
        assert_eq!(outcome, DriveOutcome::Drained);
        assert!(pending.is_empty());
        assert_eq!(refusals, 1);

        // A successful decode clears the count.
        let mut pending = queue(&arena, &[8]);
        drive_input(&mut pending, &mut refusals, |d, _| Ok(d.len())).unwrap();
        assert_eq!(refusals, 0, "one good unit resets the run");

        // A long enough run is fatal.
        refusals = MAX_CONSECUTIVE_REFUSALS - 1;
        let mut pending = queue(&arena, &[8]);
        let err = drive_input(&mut pending, &mut refusals, |_, _| {
            Err(DecodeError::ParseFrameError("bad NAL".to_string()))
        });
        assert!(err.is_err(), "a stream that never parses must not loop forever");
    }

    /// Construction answers rather than panicking, whatever the machine.
    ///
    /// `VaapiBackend::new` panics when config creation fails, so this is
    /// really asserting that the capability probe runs first — for every
    /// codec, including ones this machine cannot decode.
    #[test]
    fn construction_answers_on_any_machine() {
        for codec in [Codec::Vp9, Codec::Vp8, Codec::H264, Codec::H265, Codec::Av1] {
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
