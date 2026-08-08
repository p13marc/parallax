//! Hardware decoder element wrapper.
//!
//! This module provides [`HwDecoderElement`], a wrapper that adapts any
//! [`HwVideoDecoder`] (like `VulkanH264Decoder`) to work as a pipeline element.
//!
//! # Features
//!
//! - Hardware-accelerated video decoding
//! - Decoded pixels copied into shared-memory `Buffer`s with in-band
//!   geometry (`Metadata::set_video_dims`), ready for any downstream element
//! - Automatic flush at end-of-stream
//! - Statistics tracking
//!
//! DMA-BUF export of decoded frames (zero-copy output) is tracked in #62.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::gpu::{VulkanContext, VulkanH264Decoder};
//! use parallax::elements::codec::HwDecoderElement;
//!
//! let ctx = std::sync::Arc::new(VulkanContext::new()?);
//! let decoder = VulkanH264Decoder::new(ctx)?;
//! let element = HwDecoderElement::new(decoder);
//!
//! pipeline.add_node("hw_decoder", DynAsyncElement::new_box(TransformAdapter::new(element)));
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::{Error, Result};
use crate::gpu::{GpuFrame, HwVideoDecoder};
use crate::memory::SharedArena;
use std::collections::VecDeque;

/// Wraps a [`HwVideoDecoder`] to work as a pipeline [`Transform`] element.
///
/// This wrapper handles:
/// - Converting encoded packets to GPU frames
/// - Managing decoder buffering (B-frame reordering)
/// - Copying decoded pixels into shared-memory buffers with in-band geometry
/// - Flushing remaining frames at EOS
/// - Preserving timestamps
///
/// # GPU Memory Handling
///
/// Decoded frames are copied to CPU shared memory (via
/// [`HwVideoDecoder::read_frame`]) for compatibility with every pipeline
/// element; the `GpuFrame`'s device memory is freed on drop. Zero-copy
/// DMA-BUF output is tracked in #62.
///
/// # Usage
///
/// ```rust,ignore
/// // Create Vulkan context and decoder (geometry comes from the SPS)
/// let ctx = std::sync::Arc::new(VulkanContext::new()?);
/// let decoder = VulkanH264Decoder::new(ctx)?;
///
/// // Wrap in HwDecoderElement
/// let element = HwDecoderElement::new(decoder);
///
/// // Add to pipeline
/// let node = pipeline.add_node(
///     "hw_decoder",
///     DynAsyncElement::new_box(TransformAdapter::new(element)),
/// );
/// ```
pub struct HwDecoderElement<D: HwVideoDecoder> {
    /// The underlying hardware decoder.
    decoder: D,
    /// Queue of pending output frames (for multiple outputs per packet).
    pending_frames: VecDeque<GpuFrame>,
    /// Whether we've started flushing.
    flushing: bool,
    /// Whether flush is complete.
    flushed: bool,
    /// Statistics: packets received.
    packets_in: u64,
    /// Statistics: frames produced.
    frames_out: u64,
    /// Arena for output buffer allocation (CPU fallback).
    arena: Option<SharedArena>,
    /// Expected output width (for arena sizing).
    width: u32,
    /// Expected output height (for arena sizing).
    height: u32,
}

impl<D: HwVideoDecoder> HwDecoderElement<D> {
    /// Create a new hardware decoder element wrapper.
    ///
    /// # Arguments
    ///
    /// * `decoder` - The hardware video decoder to wrap
    pub fn new(decoder: D) -> Self {
        Self {
            decoder,
            pending_frames: VecDeque::new(),
            flushing: false,
            flushed: false,
            packets_in: 0,
            frames_out: 0,
            arena: None,
            width: 0,
            height: 0,
        }
    }

    /// Get the number of packets received.
    pub fn packets_in(&self) -> u64 {
        self.packets_in
    }

    /// Get the number of frames produced.
    pub fn frames_out(&self) -> u64 {
        self.frames_out
    }

    /// Get a reference to the inner decoder.
    pub fn decoder(&self) -> &D {
        &self.decoder
    }

    /// Get a mutable reference to the inner decoder.
    pub fn decoder_mut(&mut self) -> &mut D {
        &mut self.decoder
    }

    /// Copy a decoded [`GpuFrame`] into a pipeline [`Buffer`].
    ///
    /// The pixels are read through [`HwVideoDecoder::read_frame`] into a
    /// `SharedArena` slot, and geometry is stamped with
    /// `Metadata::set_video_dims` so downstream elements can read it. The
    /// `GpuFrame` is consumed; dropping it frees its GPU memory (RAII).
    fn frame_to_buffer(
        &mut self,
        frame: GpuFrame,
        input_metadata: Option<&crate::metadata::Metadata>,
    ) -> Result<Buffer> {
        let frame_size = frame.format.frame_size(frame.width, frame.height);

        // Observability only — never a fallback.
        self.width = frame.width;
        self.height = frame.height;

        // (Re)build the arena when the frame no longer fits — a mid-stream
        // resolution change must not truncate frames into stale slots.
        if self
            .arena
            .as_ref()
            .is_none_or(|a| a.slot_size() < frame_size)
        {
            self.arena = Some(
                SharedArena::new(frame_size, 8)
                    .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
            );
        }

        let arena = self.arena.as_mut().expect("just ensured");
        arena.reclaim();
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;

        // The actual bridge: decoded pixels land in shared memory here.
        self.decoder
            .read_frame(&frame, &mut slot.data_mut()[..frame_size])?;

        // Propagate the input's metadata (stream_id, custom keys) where we
        // have it; flush has no input to inherit from.
        let mut metadata = input_metadata.cloned().unwrap_or_default();
        metadata.pts = ClockTime::from_nanos(frame.pts as u64);
        metadata.set_video_dims(frame.width, frame.height, frame.format.into());
        if frame.is_keyframe {
            metadata.flags |= crate::metadata::BufferFlags::SYNC_POINT;
        }

        Ok(Buffer::new(
            MemoryHandle::with_len(slot, frame_size),
            metadata,
        ))
        // `frame` drops here: its GPU memory is freed by the handle's Drop.
    }
}

impl<D: HwVideoDecoder + 'static> Transform for HwDecoderElement<D> {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let packet = buffer.as_bytes();
        let pts = buffer
            .metadata()
            .pts
            .to_option()
            .map(|t| t.nanos())
            .unwrap_or(self.frames_out * 33_333_333) as i64; // Default 30fps

        self.packets_in += 1;

        // Decode packet
        let frames = self.decoder.decode(packet, pts)?;

        // If no frames, decoder is buffering
        if frames.is_empty() {
            return Ok(Output::None);
        }

        // Convert frames to buffers, inheriting the input's metadata.
        let input_metadata = buffer.metadata().clone();
        let mut buffers = Vec::with_capacity(frames.len());
        for frame in frames {
            buffers.push(self.frame_to_buffer(frame, Some(&input_metadata))?);
            self.frames_out += 1;
        }

        Ok(Output::from(buffers))
    }

    fn flush(&mut self) -> Result<Output> {
        if self.flushed {
            return Ok(Output::None);
        }

        // Check for pending frames from previous flush call
        if let Some(frame) = self.pending_frames.pop_front() {
            self.frames_out += 1;
            return Ok(Output::single(self.frame_to_buffer(frame, None)?));
        }

        // First flush call: get all remaining frames
        if !self.flushing {
            self.flushing = true;
            let frames = self.decoder.flush()?;

            for frame in frames {
                self.pending_frames.push_back(frame);
            }
        }

        // Return next pending frame
        match self.pending_frames.pop_front() {
            Some(frame) => {
                self.frames_out += 1;
                Ok(Output::single(self.frame_to_buffer(frame, None)?))
            }
            None => {
                self.flushed = true;
                Ok(Output::None)
            }
        }
    }

    fn name(&self) -> &str {
        "HwDecoderElement"
    }

    fn execution_hints(&self) -> ExecutionHints {
        // Hardware decoders use native code (Vulkan driver)
        // They should be isolated for safety
        ExecutionHints::native()
    }
}

impl<D: HwVideoDecoder> std::fmt::Debug for HwDecoderElement<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HwDecoderElement")
            .field("packets_in", &self.packets_in)
            .field("frames_out", &self.frames_out)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("flushing", &self.flushing)
            .field("flushed", &self.flushed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{MediaFormat, PixelFormat};
    use crate::gpu::{Codec, GpuBuffer, GpuPixelFormat, GpuUsage, VideoProfile};
    use crate::metadata::{BufferFlags, Metadata};
    use std::collections::HashMap;

    /// A synthetic frame over a `None` handle: no GPU anywhere near the test.
    fn gpu_frame(pts: i64, keyframe: bool, w: u32, h: u32) -> GpuFrame {
        GpuFrame {
            buffer: GpuBuffer {
                handle: Default::default(),
                size: GpuPixelFormat::Nv12.frame_size(w, h),
                usage: GpuUsage::default(),
            },
            format: GpuPixelFormat::Nv12,
            width: w,
            height: h,
            stride: w,
            pts,
            is_keyframe: keyframe,
        }
    }

    /// Serves pre-baked pixel bytes per pts through `read_frame`, which is
    /// exactly how the element is supposed to reach decoded data.
    struct MockDecoder {
        pixels: HashMap<i64, Vec<u8>>,
        queued: Vec<GpuFrame>,
        flush_queue: Vec<GpuFrame>,
        fail_read: bool,
        profile: VideoProfile,
    }

    impl MockDecoder {
        fn new() -> Self {
            Self {
                pixels: HashMap::new(),
                queued: Vec::new(),
                flush_queue: Vec::new(),
                fail_read: false,
                profile: VideoProfile::default(),
            }
        }

        fn queue(&mut self, frame: GpuFrame, pixels: Vec<u8>) {
            self.pixels.insert(frame.pts, pixels);
            self.queued.push(frame);
        }

        fn queue_flush(&mut self, frame: GpuFrame, pixels: Vec<u8>) {
            self.pixels.insert(frame.pts, pixels);
            self.flush_queue.push(frame);
        }
    }

    impl HwVideoDecoder for MockDecoder {
        fn decode(&mut self, _packet: &[u8], _pts: i64) -> Result<Vec<GpuFrame>> {
            Ok(std::mem::take(&mut self.queued))
        }

        fn flush(&mut self) -> Result<Vec<GpuFrame>> {
            Ok(std::mem::take(&mut self.flush_queue))
        }

        fn reset(&mut self) -> Result<()> {
            Ok(())
        }

        fn codec(&self) -> Codec {
            Codec::H264
        }

        fn profile(&self) -> &VideoProfile {
            &self.profile
        }

        fn output_format(&self) -> GpuPixelFormat {
            GpuPixelFormat::Nv12
        }

        fn read_frame(&self, frame: &GpuFrame, out: &mut [u8]) -> Result<()> {
            if self.fail_read {
                return Err(Error::Element("mock read failure".into()));
            }
            let bytes = &self.pixels[&frame.pts];
            out[..bytes.len()].copy_from_slice(bytes);
            Ok(())
        }
    }

    fn input_packet() -> Buffer {
        let arena = SharedArena::new(64, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let mut metadata = Metadata::new();
        metadata.stream_id = 7;
        Buffer::new(MemoryHandle::with_len(slot, 16), metadata)
    }

    fn buffers(output: Output) -> Vec<Buffer> {
        match output {
            Output::None => vec![],
            Output::Single(b) => vec![b],
            Output::Multiple(bs) => bs,
        }
    }

    /// The headline regression: decoded pixels must actually be in the
    /// output buffer, not whatever the arena slot last held.
    #[test]
    fn decoded_bytes_reach_the_output_buffer() {
        let mut mock = MockDecoder::new();
        let size = GpuPixelFormat::Nv12.frame_size(16, 16);
        let pattern: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        mock.queue(gpu_frame(1000, true, 16, 16), pattern.clone());

        let mut element = HwDecoderElement::new(mock);
        let out = buffers(element.transform(input_packet()).unwrap());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].as_bytes(), &pattern[..]);
    }

    /// Geometry travels in-band, in the shape every downstream element reads.
    #[test]
    fn output_metadata_uses_set_video_dims() {
        let mut mock = MockDecoder::new();
        let size = GpuPixelFormat::Nv12.frame_size(32, 16);
        mock.queue(gpu_frame(0, true, 32, 16), vec![0x22; size]);

        let mut element = HwDecoderElement::new(mock);
        let out = buffers(element.transform(input_packet()).unwrap());
        let meta = out[0].metadata();

        assert_eq!(meta.video_dims(), Some((32, 16)));
        assert_eq!(meta.video_pixel_format(), Some(PixelFormat::Nv12));
        assert!(matches!(
            meta.format,
            Some(MediaFormat::VideoRaw(vf)) if vf.width == 32 && vf.height == 16
        ));
        // Legacy convention too (AutoVideoSink reads it).
        assert_eq!(meta.get::<u64>("width"), Some(&32));
        // Input metadata (stream_id) is inherited.
        assert_eq!(meta.stream_id, 7);
    }

    #[test]
    fn keyframes_carry_sync_point_and_delta_frames_do_not() {
        let mut mock = MockDecoder::new();
        let size = GpuPixelFormat::Nv12.frame_size(16, 16);
        mock.queue(gpu_frame(0, true, 16, 16), vec![1; size]);
        mock.queue(gpu_frame(1, false, 16, 16), vec![2; size]);

        let mut element = HwDecoderElement::new(mock);
        let out = buffers(element.transform(input_packet()).unwrap());
        assert_eq!(out.len(), 2);
        assert!(out[0].metadata().flags.contains(BufferFlags::SYNC_POINT));
        assert!(!out[1].metadata().flags.contains(BufferFlags::SYNC_POINT));
    }

    #[test]
    fn pts_maps_into_metadata_and_order_is_preserved() {
        let mut mock = MockDecoder::new();
        let size = GpuPixelFormat::Nv12.frame_size(16, 16);
        for pts in [33, 66, 99] {
            mock.queue(gpu_frame(pts, false, 16, 16), vec![pts as u8; size]);
        }

        let mut element = HwDecoderElement::new(mock);
        let out = buffers(element.transform(input_packet()).unwrap());
        let ptss: Vec<u64> = out.iter().map(|b| b.metadata().pts.nanos()).collect();
        assert_eq!(ptss, vec![33, 66, 99]);
        assert_eq!(element.frames_out(), 3);
    }

    #[test]
    fn flush_drains_decoder_frames() {
        let mut mock = MockDecoder::new();
        let size = GpuPixelFormat::Nv12.frame_size(16, 16);
        mock.queue_flush(gpu_frame(500, false, 16, 16), vec![9; size]);

        let mut element = HwDecoderElement::new(mock);
        let out = buffers(element.flush().unwrap());
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].metadata().pts.nanos(), 500);
        // Second flush: drained.
        assert!(matches!(element.flush().unwrap(), Output::None));
    }

    #[test]
    fn read_frame_failure_propagates_as_an_error() {
        let mut mock = MockDecoder::new();
        let size = GpuPixelFormat::Nv12.frame_size(16, 16);
        mock.queue(gpu_frame(0, true, 16, 16), vec![0; size]);
        mock.fail_read = true;

        let mut element = HwDecoderElement::new(mock);
        assert!(element.transform(input_packet()).is_err());
    }

    /// A mid-stream resolution change must re-size the arena, not truncate.
    #[test]
    fn a_resolution_change_rebuilds_the_arena() {
        let mut mock = MockDecoder::new();
        let small = GpuPixelFormat::Nv12.frame_size(16, 16);
        mock.queue(gpu_frame(0, true, 16, 16), vec![1; small]);

        let mut element = HwDecoderElement::new(mock);
        buffers(element.transform(input_packet()).unwrap());

        let big = GpuPixelFormat::Nv12.frame_size(64, 64);
        element
            .decoder_mut()
            .queue(gpu_frame(1, true, 64, 64), vec![3; big]);
        let out = buffers(element.transform(input_packet()).unwrap());
        assert_eq!(out[0].len(), big);
        assert_eq!(out[0].metadata().video_dims(), Some((64, 64)));
    }
}
