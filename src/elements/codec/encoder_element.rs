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

use super::common::VideoFrame;
use super::traits::VideoEncoder;
use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::{Error, Result};
use crate::format::MediaFormat;
use crate::memory::SharedArena;
use std::collections::VecDeque;

/// Wraps a [`VideoEncoder`] to work as a pipeline [`Transform`] element.
///
/// This wrapper handles:
/// - Converting input buffers to [`VideoFrame`]
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
    /// Current input format (seeded by the constructor, updated by
    /// per-buffer format metadata).
    format: crate::format::VideoFormat,
    /// Statistics: frames received.
    frames_in: u64,
    /// Statistics: packets produced.
    packets_out: u64,
    /// Runtime control: keyframe requests plus bitrate/GOP/QP changes
    /// (shared with [`Self::control_handle`] and [`Self::keyframe_handle`]).
    control: super::EncoderControl,
    /// The control generation last applied to the encoder.
    applied_generation: u64,
    /// Arena for output buffer allocation.
    arena: SharedArena,
}

impl<E: VideoEncoder> EncoderElement<E> {
    /// Create a new encoder element wrapper.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The video encoder to wrap
    /// * `format` - Expected input format (width, height, pixel format,
    ///   framerate). A buffer carrying different `VideoRaw` format metadata
    ///   renegotiates on the fly. Only planar/semi-planar YUV formats are
    ///   encodable — packed or RGB input needs `VideoConvert` upstream.
    pub fn new(encoder: E, format: crate::format::VideoFormat) -> Result<Self> {
        // Fail fast on formats no wrapped encoder can take.
        map_pixel_format(format.pixel_format)?;

        // Estimate max packet size (compressed should be smaller than raw frame)
        let max_packet_size = (format.width as usize) * (format.height as usize) * 3;
        let arena = SharedArena::new(max_packet_size.max(4096), 16)
            .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?;

        Ok(Self {
            encoder,
            pending_packets: VecDeque::new(),
            flushing: false,
            flushed: false,
            format,
            frames_in: 0,
            packets_out: 0,
            control: super::EncoderControl::new(),
            applied_generation: 0,
            arena,
        })
    }

    /// The current input format (may change via renegotiation).
    pub fn format(&self) -> crate::format::VideoFormat {
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

    /// Convert input buffer to VideoFrame, honoring format renegotiation.
    fn buffer_to_frame(&mut self, buffer: &Buffer) -> Result<VideoFrame> {
        // Renegotiation: format metadata (set on the first buffer or on
        // format changes) overrides the configured format.
        if let Some(MediaFormat::VideoRaw(vf)) = buffer.metadata().format
            && vf != self.format
        {
            tracing::debug!(
                "encoder input renegotiated: {}x{} {:?} -> {}x{} {:?}",
                self.format.width,
                self.format.height,
                self.format.pixel_format,
                vf.width,
                vf.height,
                vf.pixel_format
            );
            self.format = vf;
        }

        let format = map_pixel_format(self.format.pixel_format)?;
        let width = self.format.width;
        let bpc = format.bytes_per_component();
        let stride_y = width as usize * bpc;
        let (stride_u, stride_v) = match format {
            // Semi-planar: one interleaved UV plane at full row width.
            super::common::PixelFormat::Nv12 => (stride_y, 0),
            super::common::PixelFormat::I444 => (stride_y, stride_y),
            // Planar 4:2:0 / 4:2:2: half-width chroma planes.
            _ => (stride_y / 2, stride_y / 2),
        };

        Ok(VideoFrame {
            width,
            height: self.format.height,
            format,
            pts: buffer.metadata().pts.nanos() as i64,
            data: buffer.as_bytes().to_vec(),
            stride_y,
            stride_u,
            stride_v,
        })
    }

    /// Convert encoded packet to output buffer, preserving input metadata.
    fn packet_to_buffer(
        &self,
        data: Vec<u8>,
        pts: i64,
        input_metadata: &crate::metadata::Metadata,
    ) -> Result<Buffer> {
        self.arena.reclaim();
        let mut slot = self
            .arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
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
    fn packet_to_buffer_flush(&self, data: Vec<u8>, pts: i64) -> Result<Buffer> {
        self.arena.reclaim();
        let mut slot = self
            .arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
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
    }
}

impl<E: VideoEncoder + 'static> Transform for EncoderElement<E> {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
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
        let packets = self.encoder.encode(&frame)?;

        // If no packets, encoder is buffering
        if packets.is_empty() {
            return Ok(Output::None);
        }

        // Convert packets to buffers, preserving input metadata
        let mut buffers = Vec::with_capacity(packets.len());
        for packet in packets {
            let data = packet.as_ref().to_vec();
            buffers.push(self.packet_to_buffer(data, pts, input_metadata)?);
            self.packets_out += 1;
        }

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
            CapsValue, ElementMediaCaps, FormatMemoryCap, MemoryCaps, VideoFormatCaps,
        };
        // Advertise the configured format so graph negotiation can catch
        // mismatches (and auto-insert a converter) instead of failing at the
        // first encoded frame.
        let caps = VideoFormatCaps {
            width: CapsValue::Fixed(self.format.width),
            height: CapsValue::Fixed(self.format.height),
            pixel_format: CapsValue::Fixed(self.format.pixel_format),
            ..VideoFormatCaps::any()
        };
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            caps.into(),
            MemoryCaps::cpu_only(),
        )])
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
            usize,
            usize,
            usize,
        )>,
    }

    impl VideoEncoder for RecordingEncoder {
        type Packet = Vec<u8>;
        fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Vec<u8>>> {
            self.frames.push((
                frame.width,
                frame.height,
                frame.format,
                frame.stride_y,
                frame.stride_u,
                frame.stride_v,
            ));
            Ok(vec![vec![0xAB]])
        }
        fn flush(&mut self) -> Result<Vec<Vec<u8>>> {
            Ok(Vec::new())
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

    fn frame_buffer(format: Option<VideoFormat>) -> Buffer {
        let arena = SharedArena::new(64, 8).unwrap();
        let slot = arena.acquire().unwrap();
        let mut metadata = Metadata::from_sequence(0);
        metadata.format = format.map(MediaFormat::VideoRaw);
        Buffer::new(MemoryHandle::with_len(slot, 16), metadata)
    }

    #[test]
    fn strides_follow_pixel_format() {
        for (pf, expect) in [
            (PixelFormat::I420, (640usize, 320usize, 320usize)),
            (PixelFormat::Nv12, (640, 640, 0)),
            (PixelFormat::I444, (640, 640, 640)),
            (PixelFormat::I420_10Le, (1280, 640, 640)),
        ] {
            let mut element = EncoderElement::new(
                RecordingEncoder { frames: vec![] },
                video_format(640, 480, pf),
            )
            .unwrap();
            element.transform(frame_buffer(None)).unwrap();
            let (w, h, _, sy, su, sv) = element.encoder().frames[0];
            assert_eq!((w, h), (640, 480));
            assert_eq!((sy, su, sv), expect, "strides for {pf:?}");
        }
    }

    #[test]
    fn metadata_renegotiates_format() {
        let mut element = EncoderElement::new(
            RecordingEncoder { frames: vec![] },
            video_format(640, 480, PixelFormat::I420),
        )
        .unwrap();
        element.transform(frame_buffer(None)).unwrap();
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
        assert_eq!(
            element.format().width,
            1280,
            "format sticks after renegotiation"
        );
    }

    #[test]
    fn unmappable_formats_error() {
        // At construction:
        assert!(
            EncoderElement::new(
                RecordingEncoder { frames: vec![] },
                video_format(640, 480, PixelFormat::Rgb24)
            )
            .is_err(),
            "RGB input must be rejected (needs VideoConvert)"
        );
        // Via renegotiation:
        let mut element = EncoderElement::new(
            RecordingEncoder { frames: vec![] },
            video_format(640, 480, PixelFormat::I420),
        )
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
}
