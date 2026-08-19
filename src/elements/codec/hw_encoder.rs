//! Hardware encoder element wrapper.
//!
//! This module provides [`HwEncoderElement`], a wrapper that adapts any
//! [`HwVideoEncoder`] (like a future `VulkanH264Encoder`) to work as a pipeline element.
//!
//! # Features
//!
//! - Hardware-accelerated video encoding
//! - GPU frame input (zero-copy when possible)
//! - DMA-BUF import for cross-process sharing
//! - Automatic flush at end-of-stream
//! - Keyframe forcing
//! - Statistics tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::gpu::{VulkanContext, VulkanH264Encoder};
//! use parallax::elements::codec::HwEncoderElement;
//!
//! let ctx = VulkanContext::new()?;
//! let encoder = VulkanH264Encoder::new(&ctx, 1920, 1080)?;
//! let element = HwEncoderElement::new(encoder);
//!
//! pipeline.add_node("hw_encoder", DynAsyncElement::new_box(TransformAdapter::new(element)));
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::{Error, Result};
use crate::gpu::{GpuBuffer, GpuBufferHandle, GpuFrame, GpuPixelFormat, GpuUsage, HwVideoEncoder};
use crate::memory::{OutputArena, OutputBudget, defaults};
use std::collections::VecDeque;

/// Wraps a [`HwVideoEncoder`] to work as a pipeline [`Transform`] element.
///
/// This wrapper handles:
/// - Converting raw frame buffers to GPU frames
/// - Managing encoder buffering (B-frame reordering)
/// - Converting encoded packets to output buffers
/// - Flushing remaining packets at EOS
/// - Preserving timestamps
///
/// # GPU Memory Handling
///
/// The element can accept raw video frames in CPU memory (which are uploaded
/// to GPU) or GPU frames via DMA-BUF for zero-copy operation.
///
/// # Usage
///
/// ```rust,ignore
/// // Create Vulkan context and encoder
/// let ctx = VulkanContext::new()?;
/// let encoder = VulkanH264Encoder::new(&ctx, 1920, 1080)?;
///
/// // Wrap in HwEncoderElement
/// let element = HwEncoderElement::new(encoder);
///
/// // Add to pipeline
/// let node = pipeline.add_node(
///     "hw_encoder",
///     DynAsyncElement::new_box(TransformAdapter::new(element)),
/// );
/// ```
pub struct HwEncoderElement<E: HwVideoEncoder> {
    /// The underlying hardware encoder.
    encoder: E,
    /// Queue of pending output packets (for multiple outputs per frame).
    pending_packets: VecDeque<Vec<u8>>,
    /// Whether we've started flushing.
    flushing: bool,
    /// Whether flush is complete.
    flushed: bool,
    /// Statistics: frames received.
    frames_in: u64,
    /// Statistics: packets produced.
    packets_out: u64,
    /// Arena for output buffer allocation.
    output: OutputArena,
    /// Pending runtime keyframe requests (shared with [`Self::keyframe_handle`]).
    keyframe_requests: super::KeyframeHandle,
    /// Counters readable while the pipeline runs (shared with [`Self::stats`]).
    stats: crate::control::EncoderStatsHandle,
    /// Expected input width.
    width: u32,
    /// Expected input height.
    height: u32,
    /// Input pixel format.
    format: GpuPixelFormat,
}

impl<E: HwVideoEncoder> HwEncoderElement<E> {
    /// Create a new hardware encoder element wrapper.
    ///
    /// Takes **no dimensions**: geometry travels in-band, in each buffer's
    /// [`Metadata`](crate::metadata::Metadata). A buffer that declares none is
    /// an error, not an invitation to reuse the last frame's size (#38).
    ///
    /// # Arguments
    ///
    /// * `encoder` - The hardware video encoder to wrap
    pub fn new(encoder: E) -> Self {
        Self {
            encoder,
            pending_packets: VecDeque::new(),
            flushing: false,
            flushed: false,
            frames_in: 0,
            packets_out: 0,
            output: OutputArena::new(defaults::VIDEO_ENCODER_SLOT_COUNT)
                .with_min_slot_size(256 * 1024),
            keyframe_requests: super::KeyframeHandle::new(),
            stats: crate::control::EncoderStatsHandle::default(),
            width: 0,
            height: 0,
            format: GpuPixelFormat::Nv12,
        }
    }

    /// Force the next encoded frame to be a keyframe.
    pub fn request_keyframe(&mut self) {
        self.keyframe_requests.request();
    }

    /// Get a cloneable handle for requesting keyframes at runtime.
    ///
    /// Clone this *before* the pipeline starts; a
    /// [`request()`](super::KeyframeHandle::request) makes the wrapper call
    /// [`HwVideoEncoder::force_keyframe`] before encoding its next frame.
    pub fn keyframe_handle(&self) -> super::KeyframeHandle {
        self.keyframe_requests.clone()
    }

    /// Get the number of frames received.
    pub fn frames_in(&self) -> u64 {
        self.frames_in
    }

    /// A cloneable handle to this encoder's counters.
    ///
    /// Clone it *before* `executor.start()`: the element is moved into its
    /// executor task there, so the plain `&self` counters above can never be
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

    /// Get codec data (SPS/PPS for H.264, VPS/SPS/PPS for H.265).
    ///
    /// This should be called after the first encode to get header data
    /// that must be sent out-of-band or at the start of the stream.
    pub fn codec_data(&self) -> Option<Vec<u8>> {
        self.encoder.codec_data()
    }

    /// Convert input buffer to GpuFrame.
    ///
    /// Currently creates a placeholder GPU frame. Full implementation would:
    /// - Upload CPU data to GPU memory
    /// - Or import DMA-BUF for zero-copy
    fn buffer_to_frame(&mut self, buffer: &Buffer) -> Result<GpuFrame> {
        // Geometry travels in-band. Reusing the previous frame's size when a
        // buffer declares none is the silent-corruption mode #38 removes: put
        // a scaler upstream and the encoder would quietly encode the wrong
        // rectangle.
        let (width, height) = buffer.metadata().video_dims().ok_or_else(|| {
            Error::Element(
                "HwEncoderElement: buffer carries no video dimensions. Geometry travels \
                 in-band — set Metadata::set_video_dims() upstream, or insert an element \
                 that does (VideoScale, VideoConvert, a device source)."
                    .into(),
            )
        })?;

        if (self.width, self.height) != (width, height) {
            if self.width != 0 {
                tracing::debug!(
                    "hw encoder input resized: {}x{} -> {width}x{height}",
                    self.width,
                    self.height
                );
            }
            self.width = width;
            self.height = height;
        }

        let pts = buffer
            .metadata()
            .pts
            .to_option()
            .map(|t| t.nanos())
            .unwrap_or(self.frames_in * 33_333_333) as i64;

        let is_keyframe = buffer
            .metadata()
            .get::<bool>("video/keyframe")
            .copied()
            .unwrap_or(false);

        // In a full implementation, we would:
        // 1. Check if buffer is backed by DMA-BUF
        // 2. If so, import directly to GPU
        // 3. If CPU buffer, upload to GPU memory

        Ok(GpuFrame {
            buffer: GpuBuffer {
                handle: GpuBufferHandle::None,
                size: buffer.len(),
                usage: GpuUsage::encode_input(),
            },
            format: self.format,
            width,
            height,
            stride: width,
            pts,
            is_keyframe,
        })
    }

    /// Convert encoded packet to output buffer.
    fn packet_to_buffer(&mut self, packet: Vec<u8>, pts: i64, is_keyframe: bool) -> Result<Buffer> {
        let packet_size = packet.len();

        // `acquire` sizes the arena on first use and rejects an oversized
        // packet with a message naming the cause.
        let mut slot = self.output.acquire(packet_size, "hwencoderelement")?;
        slot.data_mut()[..packet_size].copy_from_slice(&packet);

        let mut metadata = crate::metadata::Metadata::new();
        metadata.pts = ClockTime::from_nanos(pts as u64);
        metadata.set("video/keyframe", is_keyframe);
        metadata.set("video/hw_encoded", true);
        metadata.set("video/codec", codec_name(self.encoder.codec()));

        Ok(Buffer::new(
            MemoryHandle::with_len(slot, packet_size),
            metadata,
        ))
    }
}

impl<E: HwVideoEncoder> super::Controllable for HwEncoderElement<E> {
    /// Hardware encoders expose only keyframe forcing today — there is no
    /// `HwVideoEncoder::set_bitrate`, so there is nothing wider to hand out.
    type Control = super::KeyframeHandle;

    fn control(&self) -> super::KeyframeHandle {
        self.keyframe_requests.clone()
    }
}

impl<E: HwVideoEncoder + 'static> Transform for HwEncoderElement<E> {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let pts = buffer
            .metadata()
            .pts
            .to_option()
            .map(|t| t.nanos())
            .unwrap_or(self.frames_in * 33_333_333) as i64;

        // Check for keyframe request from upstream or the runtime handle
        let upstream_keyframe = buffer
            .metadata()
            .get::<bool>(super::KEYFRAME_REQUEST)
            .copied()
            .unwrap_or(false);

        if self.keyframe_requests.take() || upstream_keyframe {
            self.encoder.force_keyframe();
        }

        self.frames_in += 1;

        // Convert buffer to GPU frame
        let frame = self.buffer_to_frame(&buffer)?;

        // Encode frame
        let started = std::time::Instant::now();
        let packets = self.encoder.encode(&frame)?;
        let elapsed_ns = started.elapsed().as_nanos() as u64;

        // If no packets, the encoder is buffering (B-frame reorder) or rate
        // control swallowed the frame. Either way nothing came out for this
        // input, which is exactly the event a sender-side bitrate loop needs
        // to see.
        if packets.is_empty() {
            self.stats.record_rc_drop(elapsed_ns);
            return Ok(Output::None);
        }

        // Convert packets to buffers
        let mut buffers = Vec::with_capacity(packets.len());
        let mut encoded_bytes = 0usize;
        for (i, packet) in packets.into_iter().enumerate() {
            let is_keyframe = i == 0
                && buffer
                    .metadata()
                    .get::<bool>("video/keyframe")
                    .copied()
                    .unwrap_or(false);
            let data = packet.as_ref().to_vec();
            encoded_bytes += data.len();
            buffers.push(self.packet_to_buffer(data, pts, is_keyframe)?);
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
        if let Some(packet) = self.pending_packets.pop_front() {
            self.packets_out += 1;
            let pts = self.frames_in as i64 * 33_333_333;
            return Ok(Output::single(self.packet_to_buffer(packet, pts, false)?));
        }

        // First flush call: get all remaining packets
        if !self.flushing {
            self.flushing = true;
            let packets = self.encoder.flush()?;

            for packet in packets {
                self.pending_packets.push_back(packet.as_ref().to_vec());
            }
        }

        // Return next pending packet
        match self.pending_packets.pop_front() {
            Some(packet) => {
                self.packets_out += 1;
                let pts = self.frames_in as i64 * 33_333_333;
                Ok(Output::single(self.packet_to_buffer(packet, pts, false)?))
            }
            None => {
                self.flushed = true;
                Ok(Output::None)
            }
        }
    }

    fn name(&self) -> &str {
        "HwEncoderElement"
    }

    fn execution_hints(&self) -> ExecutionHints {
        // Hardware encoders use native code (Vulkan driver)
        // They should be isolated for safety
        ExecutionHints::native()
    }
}

impl<E: HwVideoEncoder> std::fmt::Debug for HwEncoderElement<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HwEncoderElement")
            .field("frames_in", &self.frames_in)
            .field("packets_out", &self.packets_out)
            .field("width", &self.width)
            .field("height", &self.height)
            .field("format", &self.format)
            .field("flushing", &self.flushing)
            .field("flushed", &self.flushed)
            .finish()
    }
}

/// Get a human-readable codec name.
fn codec_name(codec: crate::gpu::Codec) -> &'static str {
    match codec {
        crate::gpu::Codec::H264 => "H.264",
        crate::gpu::Codec::H265 => "H.265",
        crate::gpu::Codec::Av1 => "AV1",
        crate::gpu::Codec::Vp9 => "VP9",
        crate::gpu::Codec::Vp8 => "VP8",
    }
}
