//! Generic decoder element wrapper.
//!
//! This module provides [`DecoderElement`], a wrapper that adapts any
//! [`VideoDecoder`] to work as a pipeline element.
//!
//! # Features
//!
//! - Handles variable output (0, 1, or multiple frames per packet)
//! - Automatic flush at end-of-stream
//! - Timestamp preservation
//! - Statistics tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{DecoderElement, Dav1dDecoder};
//!
//! let decoder = Dav1dDecoder::new()?;
//! let element = DecoderElement::new(decoder);
//!
//! pipeline.add_node("decoder", DynAsyncElement::new_box(TransformAdapter::new(element)));
//! ```

use super::common::VideoFrame;
use super::traits::VideoDecoder;
use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use std::collections::VecDeque;

/// Wraps a [`VideoDecoder`] to work as a pipeline [`Transform`] element.
///
/// This wrapper handles:
/// - Converting encoded packets to frames
/// - Managing decoder buffering (B-frame reordering)
/// - Flushing remaining frames at EOS
/// - Preserving timestamps
///
/// # Usage
///
/// ```rust,ignore
/// // Create decoder
/// let decoder = Dav1dDecoder::new()?;
///
/// // Wrap in DecoderElement
/// let element = DecoderElement::new(decoder);
///
/// // Add to pipeline
/// let node = pipeline.add_node(
///     "decoder",
///     DynAsyncElement::new_box(TransformAdapter::new(element)),
/// );
/// ```
pub struct DecoderElement<D: VideoDecoder> {
    /// The underlying decoder.
    decoder: D,
    /// Queue of pending output frames (for multiple outputs per packet).
    pending_frames: VecDeque<VideoFrame>,
    /// Whether we've started flushing.
    flushing: bool,
    /// Whether flush is complete.
    flushed: bool,
    /// Statistics: packets received.
    packets_in: u64,
    /// Statistics: frames produced.
    frames_out: u64,
    /// Arena for output buffer allocation.
    arena: Option<SharedArena>,
}

impl<D: VideoDecoder> DecoderElement<D> {
    /// Create a new decoder element wrapper.
    pub fn new(decoder: D) -> Self {
        Self {
            decoder,
            pending_frames: VecDeque::new(),
            flushing: false,
            flushed: false,
            packets_in: 0,
            frames_out: 0,
            arena: None,
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

    /// Convert VideoFrame to output buffer.
    fn frame_to_buffer(&mut self, frame: VideoFrame) -> Result<Buffer> {
        // Initialize arena on first use with frame size
        if self.arena.is_none() {
            self.arena = Some(
                SharedArena::new(frame.data.len(), 16)
                    .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
            );
        }

        let arena = self.arena.as_ref().unwrap();
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
        slot.data_mut()[..frame.data.len()].copy_from_slice(&frame.data);

        let mut metadata = crate::metadata::Metadata::new();
        metadata.pts = ClockTime::from_nanos(frame.pts as u64);

        // Store frame dimensions in metadata for downstream elements
        metadata.set("video/width", frame.width);
        metadata.set("video/height", frame.height);
        metadata.set("video/format", format!("{:?}", frame.format));

        Ok(Buffer::new(
            MemoryHandle::with_len(slot, frame.data.len()),
            metadata,
        ))
    }
}

impl<D: VideoDecoder + 'static> Transform for DecoderElement<D> {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let packet = buffer.as_bytes();
        self.packets_in += 1;

        // Decode packet
        let frames = self.decoder.decode(packet)?;

        // If no frames, decoder is buffering
        if frames.is_empty() {
            return Ok(Output::None);
        }

        // Convert frames to buffers
        let mut buffers = Vec::with_capacity(frames.len());
        for frame in frames {
            buffers.push(self.frame_to_buffer(frame)?);
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
            return Ok(Output::single(self.frame_to_buffer(frame)?));
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
                Ok(Output::single(self.frame_to_buffer(frame)?))
            }
            None => {
                self.flushed = true;
                Ok(Output::None)
            }
        }
    }

    fn name(&self) -> &str {
        "DecoderElement"
    }

    fn execution_hints(&self) -> ExecutionHints {
        // Decoders often use native code (FFI)
        ExecutionHints::native()
    }
}

impl<D: VideoDecoder> std::fmt::Debug for DecoderElement<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecoderElement")
            .field("packets_in", &self.packets_in)
            .field("frames_out", &self.frames_out)
            .field("flushing", &self.flushing)
            .field("flushed", &self.flushed)
            .finish()
    }
}
