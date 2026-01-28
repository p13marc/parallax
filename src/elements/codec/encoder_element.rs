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
    /// Frame width (for buffer-to-frame conversion).
    width: u32,
    /// Frame height (for buffer-to-frame conversion).
    height: u32,
    /// Statistics: frames received.
    frames_in: u64,
    /// Statistics: packets produced.
    packets_out: u64,
    /// Arena for output buffer allocation.
    arena: SharedArena,
}

impl<E: VideoEncoder> EncoderElement<E> {
    /// Create a new encoder element wrapper.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The video encoder to wrap
    /// * `width` - Expected frame width
    /// * `height` - Expected frame height
    pub fn new(encoder: E, width: u32, height: u32) -> Result<Self> {
        // Estimate max packet size (compressed should be smaller than raw frame)
        let max_packet_size = (width as usize) * (height as usize) * 3;
        let arena = SharedArena::new(max_packet_size, 16)
            .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?;

        Ok(Self {
            encoder,
            pending_packets: VecDeque::new(),
            flushing: false,
            flushed: false,
            width,
            height,
            frames_in: 0,
            packets_out: 0,
            arena,
        })
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

    /// Convert input buffer to VideoFrame.
    fn buffer_to_frame(&self, buffer: &Buffer) -> VideoFrame {
        let data = buffer.as_bytes().to_vec();
        let pts = buffer.metadata().pts.nanos() as i64;

        VideoFrame {
            width: self.width,
            height: self.height,
            format: super::common::PixelFormat::I420, // Assume I420 for now
            pts,
            data,
            stride_y: self.width as usize,
            stride_u: self.width as usize / 2,
            stride_v: self.width as usize / 2,
        }
    }

    /// Convert encoded packet to output buffer.
    fn packet_to_buffer(&self, data: Vec<u8>, pts: i64) -> Result<Buffer> {
        let mut slot = self
            .arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
        slot.data_mut()[..data.len()].copy_from_slice(&data);

        let mut metadata = crate::metadata::Metadata::new();
        metadata.pts = crate::clock::ClockTime::from_nanos(pts as u64);

        Ok(Buffer::new(
            MemoryHandle::with_len(slot, data.len()),
            metadata,
        ))
    }
}

impl<E: VideoEncoder + 'static> Transform for EncoderElement<E> {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        // Convert buffer to frame
        let frame = self.buffer_to_frame(&buffer);
        let pts = frame.pts;
        self.frames_in += 1;

        // Encode frame
        let packets = self.encoder.encode(&frame)?;

        // If no packets, encoder is buffering
        if packets.is_empty() {
            return Ok(Output::None);
        }

        // Convert packets to buffers
        let mut buffers = Vec::with_capacity(packets.len());
        for packet in packets {
            let data = packet.as_ref().to_vec();
            buffers.push(self.packet_to_buffer(data, pts)?);
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
            return Ok(Output::single(self.packet_to_buffer(data, pts)?));
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
                Ok(Output::single(self.packet_to_buffer(data, pts)?))
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

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::cpu_intensive()
    }
}

impl<E: VideoEncoder> std::fmt::Debug for EncoderElement<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncoderElement")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("frames_in", &self.frames_in)
            .field("packets_out", &self.packets_out)
            .field("flushing", &self.flushing)
            .field("flushed", &self.flushed)
            .finish()
    }
}
