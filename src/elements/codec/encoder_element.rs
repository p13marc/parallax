//! Generic encoder element wrapper.
//!
//! This module provides [`EncoderElement`], a wrapper that adapts any
//! [`VideoEncoder`] to work as a pipeline element.
//!
//! # Features
//!
//! - Handles variable output (0, 1, or multiple packets per frame)
//! - Automatic flush at end-of-stream
//! - Timestamp preservation
//! - Statistics tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{EncoderElement, Rav1eEncoder, Rav1eConfig};
//!
//! let encoder = Rav1eEncoder::new(Rav1eConfig::default())?;
//! let element = EncoderElement::new(encoder);
//!
//! pipeline.add_node("encoder", DynAsyncElement::new_box(TransformAdapter::new(element)));
//! ```

use super::common::VideoFrameRef;
use super::traits::VideoEncoder;
use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::{Error, Result};
use crate::format::MediaFormat;
use crate::memory::{OutputArena, OutputBudget, defaults};
use std::collections::VecDeque;

/// Wraps a [`VideoEncoder`] to work as a pipeline [`Transform`] element.
///
/// This wrapper handles:
/// - Viewing input buffers as [`VideoFrameRef`]s (no copy)
/// - Managing encoder buffering (B-frames, lookahead)
/// - Flushing remaining packets at EOS
/// - Preserving timestamps
///
/// # Usage
///
/// ```rust,ignore
/// // Create encoder
/// let encoder = Rav1eEncoder::new(config)?;
///
/// // Wrap in EncoderElement
/// let element = EncoderElement::new(encoder);
///
/// // Add to pipeline
/// let node = pipeline.add_node(
///     "encoder",
///     DynAsyncElement::new_box(TransformAdapter::new(element)),
/// );
/// ```
pub struct EncoderElement<E: VideoEncoder> {
    /// The underlying encoder.
    encoder: E,
    /// Queue of pending output packets (for multiple outputs per frame).
    pending_packets: VecDeque<(Vec<u8>, i64)>, // (data, pts)
    /// Whether we've started flushing.
    flushing: bool,
    /// Whether flush is complete.
    flushed: bool,
    /// Current input format, learned from the first buffer's `Metadata` and
    /// re-learned whenever it changes.
    ///
    /// `None` until a buffer has declared one. It is deliberately not seeded
    /// from a constructor argument: a constructor value is a claim about data
    /// that has not arrived yet, and on any pipeline with a scaler in it that
    /// claim is a lie (#38).
    format: Option<crate::format::VideoFormat>,
    /// Statistics: frames received.
    frames_in: u64,
    /// Statistics: packets produced.
    packets_out: u64,
    /// Runtime control: keyframe requests plus bitrate/GOP/QP changes
    /// (shared with [`Self::control_handle`] and [`Self::keyframe_handle`]).
    control: super::EncoderControl,
    /// Frames, bytes, rate-control drops and encode time, readable while the
    /// element is inside its executor task (see [`Self::stats`]).
    stats: crate::control::EncoderStatsHandle,
    /// The control generation last applied to the encoder.
    applied_generation: u64,
    /// Arena for output buffer allocation, sized from the first frame and
    /// from the executor's link-capacity budget.
    output: OutputArena,
}

impl<E: VideoEncoder> EncoderElement<E> {
    /// Create a new encoder element wrapper.
    ///
    /// Takes **no format**: geometry and pixel layout travel in-band, in each
    /// buffer's [`Metadata`](crate::metadata::Metadata), and the wrapper reads
    /// them per frame. A buffer that declares no `VideoRaw` format is an
    /// error rather than an invitation to guess.
    ///
    /// Only planar/semi-planar YUV formats are encodable — packed or RGB input
    /// needs a `VideoConvert` upstream.
    pub fn new(encoder: E) -> Self {
        Self {
            encoder,
            pending_packets: VecDeque::new(),
            flushing: false,
            flushed: false,
            format: None,
            frames_in: 0,
            packets_out: 0,
            control: super::EncoderControl::new(),
            stats: crate::control::EncoderStatsHandle::default(),
            applied_generation: 0,
            output: OutputArena::new(defaults::VIDEO_ENCODER_SLOT_COUNT),
        }
    }

    /// The input format most recently declared by a buffer.
    ///
    /// `None` before the first frame arrives.
    pub fn format(&self) -> Option<crate::format::VideoFormat> {
        self.format
    }

    /// Get a cloneable handle for requesting keyframes at runtime.
    ///
    /// Clone this *before* the pipeline starts; a
    /// [`request()`](super::KeyframeHandle::request) makes the wrapper call
    /// [`VideoEncoder::force_keyframe`] before encoding its next frame.
    pub fn keyframe_handle(&self) -> super::KeyframeHandle {
        self.control.keyframe_handle()
    }

    /// Get the number of frames received.
    pub fn frames_in(&self) -> u64 {
        self.frames_in
    }

    /// A cloneable handle to this encoder's counters.
    ///
    /// Clone it *before* `executor.start()`: the element is moved into its
    /// executor task there, so the plain `&self` counters below can never be
    /// read while it is actually encoding. This handle can.
    pub fn stats(&self) -> crate::control::EncoderStatsHandle {
        self.stats.clone()
    }

    /// Get the number of packets produced.
    pub fn packets_out(&self) -> u64 {
        self.packets_out
    }

    /// Get a reference to the inner encoder.
    pub fn encoder(&self) -> &E {
        &self.encoder
    }

    /// Get a mutable reference to the inner encoder.
    pub fn encoder_mut(&mut self) -> &mut E {
        &mut self.encoder
    }

    /// View the input buffer as a frame, honoring format renegotiation.
    ///
    /// The buffer is authoritative about its own geometry and layout. If it
    /// declares nothing, this errors instead of reusing a stale value — the
    /// silent-corruption mode #38 exists to remove.
    ///
    /// The lifetime is named on purpose: the returned view borrows `buffer`,
    /// not `self`, so the caller can go on to `self.encoder.encode(frame)`
    /// while the view is live.
    fn buffer_to_frame<'a>(&mut self, buffer: &'a Buffer) -> Result<VideoFrameRef<'a>> {
        match buffer.metadata().format {
            Some(MediaFormat::VideoRaw(vf)) => {
                if self.format != Some(vf) {
                    match self.format {
                        Some(old) => tracing::debug!(
                            "encoder input renegotiated: {}x{} {:?} -> {}x{} {:?}",
                            old.width,
                            old.height,
                            old.pixel_format,
                            vf.width,
                            vf.height,
                            vf.pixel_format
                        ),
                        None => tracing::debug!(
                            "encoder input format: {}x{} {:?}",
                            vf.width,
                            vf.height,
                            vf.pixel_format
                        ),
                    }
                    self.format = Some(vf);
                    // The arena is sized from the frame; a new geometry needs a
                    // new one.
                    self.output.reset();
                }
            }
            _ => {
                return Err(Error::Element(
                    "EncoderElement: buffer carries no VideoRaw format metadata, so its \
                     geometry and pixel layout are unknown. Geometry travels in-band — \
                     set Metadata::set_video_dims() upstream, or insert an element that \
                     does (VideoScale, VideoConvert, a device source)."
                        .into(),
                ));
            }
        }

        let declared = self.format.expect("set immediately above");
        let format = map_pixel_format(declared.pixel_format)?;

        // Plane geometry comes from the buffer (#196): packed for an ordinary
        // arena frame, the producer's real strides for a codec-owned one.
        // Deriving it from width here is what the strided guard used to be
        // protecting against.
        let layout = buffer
            .metadata()
            .plane_layout()
            .expect("the VideoRaw arm above proved the buffer declares geometry");
        let need = layout.full_span_len(declared.pixel_format, declared.width, declared.height);
        if buffer.as_bytes().len() < need {
            return Err(Error::Element(format!(
                "EncoderElement: {}x{} {:?} needs {need} bytes for its declared plane \
                 layout, buffer has {}",
                declared.width,
                declared.height,
                declared.pixel_format,
                buffer.as_bytes().len()
            )));
        }

        Ok(VideoFrameRef {
            width: declared.width,
            height: declared.height,
            format,
            pts: buffer.metadata().pts.nanos() as i64,
            data: buffer.as_bytes(),
            layout,
        })
    }

    /// The output arena, created on first use and sized from the input frame.
    ///
    /// A compressed frame is smaller than its raw source, so the raw size is a
    /// safe upper bound. Sizing it lazily is what lets the constructor take no
    /// dimensions: there is nothing to size it from until a frame arrives.
    fn slot_ceiling(&self) -> usize {
        let (w, h) = self
            .format
            .map(|f| (f.width as usize, f.height as usize))
            .unwrap_or((0, 0));
        (w * h * 3).max(4096)
    }

    /// Convert encoded packet to output buffer, preserving input metadata.
    fn packet_to_buffer(
        &mut self,
        data: Vec<u8>,
        pts: i64,
        input_metadata: &crate::metadata::Metadata,
    ) -> Result<Buffer> {
        self.output.set_min_slot_size(self.slot_ceiling());
        let mut slot = self.output.acquire(data.len(), "encoderelement")?;
        slot.data_mut()[..data.len()].copy_from_slice(&data);

        // Preserve input metadata and update PTS
        let mut metadata = input_metadata.clone();
        metadata.pts = crate::clock::ClockTime::from_nanos(pts as u64);

        Ok(Buffer::new(
            MemoryHandle::with_len(slot, data.len()),
            metadata,
        ))
    }

    /// Convert encoded packet to output buffer during flush (no input metadata).
    fn packet_to_buffer_flush(&mut self, data: Vec<u8>, pts: i64) -> Result<Buffer> {
        self.output.set_min_slot_size(self.slot_ceiling());
        let mut slot = self.output.acquire(data.len(), "encoderelement")?;
        slot.data_mut()[..data.len()].copy_from_slice(&data);

        let metadata =
            crate::metadata::Metadata::from_pts(crate::clock::ClockTime::from_nanos(pts as u64));

        Ok(Buffer::new(
            MemoryHandle::with_len(slot, data.len()),
            metadata,
        ))
    }
}

impl<E: VideoEncoder> super::Controllable for EncoderElement<E> {
    type Control = super::EncoderControl;

    /// A handle for changing bitrate / keyframe interval / QP at runtime.
    ///
    /// Clone it *before* `executor.start()`. Changes are applied through
    /// [`VideoEncoder::set_bitrate`] and friends before the next frame is
    /// encoded; an encoder that cannot honor a parameter reports it, and the
    /// wrapper logs a warning and keeps encoding rather than failing the
    /// stream. See [`crate::control`].
    fn control(&self) -> super::EncoderControl {
        self.control.clone()
    }
}

/// Map a caps-level pixel format to the codec-level one.
///
/// Only planar/semi-planar YUV layouts are representable as encoder input;
/// packed YUV, RGB and grayscale need a `VideoConvert` upstream.
fn map_pixel_format(format: crate::format::PixelFormat) -> Result<super::common::PixelFormat> {
    use crate::format::PixelFormat as Caps;
    Ok(match format {
        Caps::I420 => super::common::PixelFormat::I420,
        Caps::I420_10Le => super::common::PixelFormat::I420p10,
        Caps::I422 => super::common::PixelFormat::I422,
        Caps::I444 => super::common::PixelFormat::I444,
        Caps::Nv12 => super::common::PixelFormat::Nv12,
        other => {
            return Err(Error::InvalidCaps(format!(
                "encoder cannot take {other:?} input; insert VideoConvert upstream"
            )));
        }
    })
}

impl<E: VideoEncoder + 'static> EncoderElement<E> {
    /// Apply any parameter changes made through [`Self::control_handle`].
    ///
    /// A rejected parameter is logged, not propagated: an encoder that cannot
    /// change its bitrate should keep encoding at the old one, not tear the
    /// stream down.
    fn apply_pending_control(&mut self) {
        let Some(params) = self.control.poll(&mut self.applied_generation) else {
            return;
        };

        if let Some(bps) = params.bitrate_bps
            && let Err(e) = self.encoder.set_bitrate(bps)
        {
            tracing::warn!("encoder bitrate change to {bps} bps rejected: {e}");
        }
        if let Some(frames) = params.keyframe_interval
            && let Err(e) = self.encoder.set_keyframe_interval(frames)
        {
            tracing::warn!("encoder keyframe interval change to {frames} rejected: {e}");
        }
        if let Some(qp) = params.qp
            && let Err(e) = self.encoder.set_qp(qp)
        {
            tracing::warn!("encoder QP change to {qp} rejected: {e}");
        }
        // rate_control and skip_frames are OpenH264-specific knobs; the
        // VideoEncoder trait has no setter for them, so they are ignored here
        // rather than silently pretending to apply.
    }
}

impl<E: VideoEncoder + 'static> Transform for EncoderElement<E> {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        // Admission control before the frame enters the encoder. This is not an
        // optimisation: a frame pushed into the GOP whose packet is then shed
        // leaves a hole the decoder cannot fill until the next IDR. Skipping the
        // input keeps the bitstream coherent and just lowers the frame rate,
        // which is what the existing `skip_frames` knob does deliberately.
        self.output.admit()?;

        // Runtime reconfiguration (bitrate / GOP / QP) from the control handle.
        self.apply_pending_control();

        // Runtime keyframe requests: from the shared handle or stamped
        // in-band on the buffer's metadata.
        if self.control.take_keyframe()
            || buffer
                .metadata()
                .get::<bool>(super::KEYFRAME_REQUEST)
                .copied()
                .unwrap_or(false)
        {
            self.encoder.force_keyframe();
        }

        // Convert buffer to frame (may renegotiate the input format)
        let frame = self.buffer_to_frame(&buffer)?;
        let input_metadata = buffer.metadata();
        let pts = frame.pts;
        self.frames_in += 1;

        // Encode frame
        let started = std::time::Instant::now();
        let packets = self.encoder.encode(frame)?;
        let elapsed_ns = started.elapsed().as_nanos() as u64;

        // If no packets, the encoder is buffering (lookahead) or rate control
        // swallowed the frame. Either way nothing came out for this input.
        if packets.is_empty() {
            self.stats.record_rc_drop(elapsed_ns);
            return Ok(Output::None);
        }

        // Convert packets to buffers, preserving input metadata
        let mut buffers = Vec::with_capacity(packets.len());
        let mut encoded_bytes = 0usize;
        for packet in packets {
            let data = packet.as_ref().to_vec();
            encoded_bytes += data.len();
            buffers.push(self.packet_to_buffer(data, pts, input_metadata)?);
            self.packets_out += 1;
        }
        self.stats.record_frame(encoded_bytes, elapsed_ns);

        Ok(Output::from(buffers))
    }

    fn flush(&mut self) -> Result<Output> {
        if self.flushed {
            return Ok(Output::None);
        }

        // Check for pending packets from previous flush call
        if let Some((data, pts)) = self.pending_packets.pop_front() {
            self.packets_out += 1;
            return Ok(Output::single(self.packet_to_buffer_flush(data, pts)?));
        }

        // First flush call: get all remaining packets
        if !self.flushing {
            self.flushing = true;
            let packets = self.encoder.flush()?;

            for packet in packets {
                let data = packet.as_ref().to_vec();
                // Use 0 as PTS for flushed packets (could be improved)
                self.pending_packets.push_back((data, 0));
            }
        }

        // Return next pending packet
        match self.pending_packets.pop_front() {
            Some((data, pts)) => {
                self.packets_out += 1;
                Ok(Output::single(self.packet_to_buffer_flush(data, pts)?))
            }
            None => {
                self.flushed = true;
                Ok(Output::None)
            }
        }
    }

    fn name(&self) -> &str {
        "EncoderElement"
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatMemoryCap, MemoryCaps, PixelFormat, VideoFormatCaps,
        };
        // Pin the *pixel formats* the wrapped encoders can take, so negotiation
        // auto-inserts a VideoConvert for RGB or packed input instead of
        // failing at the first frame. This list mirrors `map_pixel_format`.
        //
        // Geometry is deliberately `Any`: the encoder does not constrain it —
        // it encodes whatever each buffer declares. Advertising a fixed size
        // here would be re-asserting the constructor dimensions that #38
        // removed, and would make a scaler upstream look like a caps conflict.
        let caps = VideoFormatCaps {
            pixel_format: CapsValue::List(vec![
                PixelFormat::I420,
                PixelFormat::I420_10Le,
                PixelFormat::I422,
                PixelFormat::I444,
                PixelFormat::Nv12,
            ]),
            ..VideoFormatCaps::any()
        };
        // #196: an encoder that reads planes through `VideoFrameRef::plane`
        // can take a codec-owned frame as it is; one that assumes packed
        // bytes must not be offered one.
        let memory = if self.encoder.accepts_strided_input() {
            MemoryCaps::external_or_cpu()
        } else {
            MemoryCaps::cpu_only()
        };
        ElementMediaCaps::new(vec![FormatMemoryCap::new(caps.into(), memory)])
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::cpu_intensive()
    }
}

impl<E: VideoEncoder> std::fmt::Debug for EncoderElement<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderElement")
            .field("format", &self.format)
            .field("frames_in", &self.frames_in)
            .field("packets_out", &self.packets_out)
            .field("flushing", &self.flushing)
            .field("flushed", &self.flushed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{Framerate, PixelFormat, VideoFormat};
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;

    /// A no-op encoder that records the frames it is given.
    struct RecordingEncoder {
        frames: Vec<(
            u32,
            u32,
            super::super::common::PixelFormat,
            Vec<(usize, usize)>,
        )>,
    }

    impl VideoEncoder for RecordingEncoder {
        type Packet = Vec<u8>;
        fn encode(&mut self, frame: VideoFrameRef<'_>) -> Result<Vec<Vec<u8>>> {
            let planes = (0..4)
                .filter_map(|i| frame.plane(i))
                .map(|p| (p.offset, p.stride))
                .collect();
            self.frames
                .push((frame.width, frame.height, frame.format, planes));
            Ok(vec![vec![0xAB]])
        }
        fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
            Ok(Vec::new())
        }
        fn accepts_strided_input(&self) -> bool {
            true
        }
    }

    fn video_format(width: u32, height: u32, pixel_format: PixelFormat) -> VideoFormat {
        VideoFormat {
            width,
            height,
            pixel_format,
            framerate: Framerate { num: 30, den: 1 },
        }
    }

    /// A buffer that really holds the frame it claims to.
    ///
    /// `buffer_to_frame` validates the declared layout against the byte
    /// count (#196) — a 16-byte buffer claiming 640x480 is exactly the
    /// silent-corruption case it now refuses.
    fn frame_buffer(format: Option<VideoFormat>) -> Buffer {
        let size =
            format.map_or(16, |vf| {
                crate::format::PlaneLayout::packed(vf.pixel_format, vf.width, vf.height)
                    .required_len(vf.pixel_format, vf.width, vf.height)
            });
        let arena = SharedArena::new(size.max(16), 2).unwrap();
        let slot = arena.acquire().unwrap();
        let mut metadata = Metadata::from_sequence(0);
        metadata.format = format.map(MediaFormat::VideoRaw);
        Buffer::new(MemoryHandle::with_len(slot, size), metadata)
    }

    /// Plane offsets and strides for a packed buffer come out of
    /// `PlaneLayout`, so NV12 is two planes with a full-width chroma plane —
    /// not three with a phantom `stride_v = 0`.
    #[test]
    fn planes_follow_pixel_format() {
        for (pf, expect) in [
            (
                PixelFormat::I420,
                vec![(0usize, 640usize), (307_200, 320), (384_000, 320)],
            ),
            (PixelFormat::Nv12, vec![(0, 640), (307_200, 640)]),
            (
                PixelFormat::I444,
                vec![(0, 640), (307_200, 640), (614_400, 640)],
            ),
            (
                PixelFormat::I420_10Le,
                vec![(0, 1280), (614_400, 640), (768_000, 640)],
            ),
        ] {
            let mut element = EncoderElement::new(RecordingEncoder { frames: vec![] });
            element
                .transform(frame_buffer(Some(video_format(640, 480, pf))))
                .unwrap();
            let (w, h, _, ref planes) = element.encoder().frames[0];
            assert_eq!((w, h), (640, 480));
            assert_eq!(*planes, expect, "planes for {pf:?}");
        }
    }

    #[test]
    fn metadata_renegotiates_format() {
        let mut element = EncoderElement::new(RecordingEncoder { frames: vec![] });
        element
            .transform(frame_buffer(Some(video_format(
                640,
                480,
                PixelFormat::I420,
            ))))
            .unwrap();
        element
            .transform(frame_buffer(Some(video_format(
                1280,
                720,
                PixelFormat::Nv12,
            ))))
            .unwrap();

        let frames = &element.encoder().frames;
        assert_eq!((frames[0].0, frames[0].1), (640, 480));
        assert_eq!((frames[1].0, frames[1].1), (1280, 720));
        assert_eq!(frames[1].2, super::super::common::PixelFormat::Nv12);
        assert_eq!(frames[1].3.len(), 2, "NV12 is two planes");
        assert_eq!(
            element.format().unwrap().width,
            1280,
            "format sticks after renegotiation"
        );
    }

    /// #38: geometry travels in-band, so a buffer that declares none is an
    /// error — never an invitation to reuse a stale value.
    #[test]
    fn a_buffer_without_format_metadata_errors() {
        let mut element = EncoderElement::new(RecordingEncoder { frames: vec![] });
        assert!(
            element.format().is_none(),
            "nothing known before the first frame"
        );

        let err = element.transform(frame_buffer(None)).unwrap_err();
        assert!(
            err.to_string().contains("no VideoRaw format metadata"),
            "the error must say what is missing, got: {err}"
        );
        assert!(element.encoder().frames.is_empty(), "nothing was encoded");
    }

    /// A format-less buffer *after* a good one is still an error: the previous
    /// frame's geometry says nothing about this one.
    #[test]
    fn a_stale_format_is_not_reused() {
        let mut element = EncoderElement::new(RecordingEncoder { frames: vec![] });
        element
            .transform(frame_buffer(Some(video_format(
                640,
                480,
                PixelFormat::I420,
            ))))
            .unwrap();
        assert!(element.transform(frame_buffer(None)).is_err());
        assert_eq!(element.encoder().frames.len(), 1);
    }

    #[test]
    fn unmappable_formats_error() {
        // RGB input needs a VideoConvert upstream. It is now caught on the
        // first buffer that declares it, rather than at construction.
        let mut element = EncoderElement::new(RecordingEncoder { frames: vec![] });
        assert!(
            element
                .transform(frame_buffer(Some(video_format(
                    640,
                    480,
                    PixelFormat::Rgb24
                ))))
                .is_err(),
            "RGB input must be rejected (needs VideoConvert)"
        );

        // Via renegotiation:
        let mut element = EncoderElement::new(RecordingEncoder { frames: vec![] });
        element
            .transform(frame_buffer(Some(video_format(
                640,
                480,
                PixelFormat::I420,
            ))))
            .unwrap();
        assert!(
            element
                .transform(frame_buffer(Some(video_format(
                    640,
                    480,
                    PixelFormat::Yuyv
                ))))
                .is_err(),
            "packed YUV renegotiation must be rejected"
        );
    }
    /// An encoder that reads planes gets a codec-owned frame as it is; one
    /// that assumes packed bytes never sees one.
    #[test]
    fn input_memory_caps_follow_the_encoder() {
        use crate::memory::MemoryType;

        struct PackedOnly;
        impl VideoEncoder for PackedOnly {
            type Packet = Vec<u8>;
            fn encode(&mut self, _frame: VideoFrameRef<'_>) -> Result<Vec<Vec<u8>>> {
                Ok(Vec::new())
            }
            fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
                Ok(Vec::new())
            }
        }

        let strided = EncoderElement::new(RecordingEncoder { frames: vec![] });
        let caps = strided.input_media_caps();
        assert!(
            caps.iter()
                .any(|c| c.memory.lists_memory(MemoryType::External)),
            "a stride-reading encoder opts into External"
        );

        let packed = EncoderElement::new(PackedOnly);
        assert!(
            !packed
                .input_media_caps()
                .iter()
                .any(|c| c.memory.lists_memory(MemoryType::External)),
            "a packed-only encoder must never be offered a strided frame"
        );
    }

    /// A strided frame reaches the encoder with real plane offsets and
    /// strides — the layout it declared, not one derived from its width.
    #[test]
    fn a_strided_buffer_reaches_the_encoder_with_its_own_layout() {
        use crate::format::{PixelFormat as Caps, PlaneDesc, PlaneLayout};
        const W: u32 = 64;
        const H: u32 = 48;
        const PAD: usize = 16;

        let packed = PlaneLayout::packed(Caps::I420, W, H);
        let mut descs = Vec::new();
        let mut offset = 0usize;
        for p in packed.resolved(Caps::I420, W, H) {
            descs.push(PlaneDesc {
                offset,
                stride: p.stride + PAD,
            });
            offset += (p.stride + PAD) * p.rows;
        }
        let layout = PlaneLayout::from_planes(&descs);
        let len = layout.full_span_len(Caps::I420, W, H);

        let arena = SharedArena::new(len, 2).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..len].fill(0x40);
        let mut metadata = Metadata::from_sequence(0);
        metadata.set_video_planes(W, H, Caps::I420, layout);
        let buffer = Buffer::new(MemoryHandle::with_len(slot, len), metadata);

        let mut element = EncoderElement::new(RecordingEncoder { frames: vec![] });
        element.transform(buffer).unwrap();

        let (w, h, _, ref planes) = element.encoder().frames[0];
        assert_eq!((w, h), (W, H));
        assert_eq!(
            *planes,
            vec![
                (0usize, 64 + PAD),
                ((64 + PAD) * 48, 32 + PAD),
                ((64 + PAD) * 48 + (32 + PAD) * 24, 32 + PAD),
            ]
        );
    }
}
