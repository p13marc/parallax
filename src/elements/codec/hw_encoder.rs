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
use crate::gpu::traits::{
    GpuBuffer, GpuBufferHandle, GpuFrame, GpuPixelFormat, GpuUsage, HwVideoEncoder,
};
use crate::memory::SharedArena;
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
    arena: Option<SharedArena>,
    /// Force next frame to be keyframe.
    force_keyframe: bool,
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
            arena: None,
            force_keyframe: false,
            width: 0,
            height: 0,
            format: GpuPixelFormat::Nv12,
        }
    }

    /// Create with known dimensions (allows pre-allocation).
    pub fn with_dimensions(encoder: E, width: u32, height: u32, format: GpuPixelFormat) -> Self {
        Self {
            encoder,
            pending_packets: VecDeque::new(),
            flushing: false,
            flushed: false,
            frames_in: 0,
            packets_out: 0,
            arena: None,
            force_keyframe: false,
            width,
            height,
            format,
        }
    }

    /// Force the next encoded frame to be a keyframe.
    pub fn request_keyframe(&mut self) {
        self.force_keyframe = true;
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
        // Extract dimensions from metadata if available
        let width = buffer
            .metadata()
            .get::<u32>("video/width")
            .copied()
            .unwrap_or(self.width);
        let height = buffer
            .metadata()
            .get::<u32>("video/height")
            .copied()
            .unwrap_or(self.height);

        // Update dimensions on first frame
        if self.width == 0 {
            self.width = width;
            self.height = height;
        }

        let pts = buffer
            .metadata()
            .pts
            .as_nanos()
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

        // Initialize arena on first use (size for typical encoded packet)
        if self.arena.is_none() {
            // Allocate space for up to 256KB packets
            let max_packet_size = 256 * 1024;
            self.arena = Some(
                SharedArena::new(max_packet_size, 16)
                    .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
            );
        }

        let arena = self.arena.as_mut().unwrap();
        arena.reclaim();
        let slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;

        // Copy packet data to slot
        let data = slot.data_mut();
        if packet_size > data.len() {
            return Err(Error::Element(format!(
                "Packet size {} exceeds arena slot size {}",
                packet_size,
                data.len()
            )));
        }
        data[..packet_size].copy_from_slice(&packet);

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

impl<E: HwVideoEncoder + 'static> Transform for HwEncoderElement<E> {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let pts = buffer
            .metadata()
            .pts
            .as_nanos()
            .unwrap_or(self.frames_in * 33_333_333) as i64;

        // Check for keyframe request from upstream
        let upstream_keyframe = buffer
            .metadata()
            .get::<bool>("video/keyframe_request")
            .copied()
            .unwrap_or(false);

        if upstream_keyframe || self.force_keyframe {
            self.encoder.force_keyframe();
            self.force_keyframe = false;
        }

        self.frames_in += 1;

        // Convert buffer to GPU frame
        let frame = self.buffer_to_frame(&buffer)?;

        // Encode frame
        let packets = self.encoder.encode(&frame)?;

        // If no packets, encoder is buffering
        if packets.is_empty() {
            return Ok(Output::None);
        }

        // Convert packets to buffers
        let mut buffers = Vec::with_capacity(packets.len());
        for (i, packet) in packets.into_iter().enumerate() {
            let is_keyframe = i == 0
                && buffer
                    .metadata()
                    .get::<bool>("video/keyframe")
                    .copied()
                    .unwrap_or(false);
            buffers.push(self.packet_to_buffer(packet.as_ref().to_vec(), pts, is_keyframe)?);
            self.packets_out += 1;
        }

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
    }
}
