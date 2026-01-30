//! Generic audio decoder element wrapper.
//!
//! This module provides [`AudioDecoderElement`], a wrapper that adapts any
//! [`AudioDecoder`] to work as a pipeline element.
//!
//! # Features
//!
//! - Handles decoding packets to PCM samples
//! - Automatic flush at end-of-stream
//! - Timestamp preservation
//! - Statistics tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{AudioDecoderElement, OpusDecoder};
//!
//! let decoder = OpusDecoder::new(48000, 2)?;
//! let element = AudioDecoderElement::new(decoder);
//!
//! pipeline.add_node("decoder", DynAsyncElement::new_box(TransformAdapter::new(element)));
//! ```

use super::audio_traits::AudioDecoder;
use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::{Error, Result};
use crate::memory::SharedArena;

/// Wraps an [`AudioDecoder`] to work as a pipeline [`Transform`] element.
///
/// This wrapper handles:
/// - Converting input buffers (compressed packets) to decoded PCM
/// - Managing decoder state
/// - Flushing at EOS
/// - Preserving timestamps
pub struct AudioDecoderElement<D: AudioDecoder> {
    /// The underlying decoder.
    decoder: D,
    /// Whether we've started flushing.
    flushing: bool,
    /// Whether flush is complete.
    flushed: bool,
    /// Statistics: packets received.
    packets_in: u64,
    /// Statistics: frames produced.
    frames_out: u64,
    /// Current timestamp in nanoseconds.
    current_pts: i64,
    /// Arena for output buffer allocation.
    arena: Option<SharedArena>,
}

impl<D: AudioDecoder> AudioDecoderElement<D> {
    /// Create a new audio decoder element wrapper.
    ///
    /// # Arguments
    ///
    /// * `decoder` - The audio decoder to wrap
    pub fn new(decoder: D) -> Self {
        Self {
            decoder,
            flushing: false,
            flushed: false,
            packets_in: 0,
            frames_out: 0,
            current_pts: 0,
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

    /// Ensure arena is initialized with appropriate size.
    fn ensure_arena(&mut self, min_size: usize) -> Result<()> {
        if self.arena.is_none()
            || self.arena.as_ref().map(|a| a.slot_size()).unwrap_or(0) < min_size
        {
            // Allocate arena for decoded audio
            // Typical max: 120ms at 48kHz, stereo, 16-bit = 5760 * 2 * 2 = 23040 bytes
            let size = min_size.max(64 * 1024);
            self.arena = Some(
                SharedArena::new(size, 16)
                    .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
            );
        }
        Ok(())
    }
}

impl<D: AudioDecoder + 'static> Transform for AudioDecoderElement<D> {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let packet = buffer.as_bytes();
        let input_pts = buffer.metadata().pts.nanos() as i64;
        self.packets_in += 1;

        // Decode packet
        let mut samples = self.decoder.decode(packet)?;

        // Set timestamp
        samples.pts = if input_pts != 0 {
            input_pts
        } else {
            self.current_pts
        };

        // Update PTS for next frame
        self.current_pts = samples.pts + samples.duration_nanos() as i64;
        self.frames_out += 1;

        // Ensure arena is large enough
        self.ensure_arena(samples.data.len())?;
        let arena = self.arena.as_ref().unwrap();
        arena.reclaim();

        // Copy to output buffer
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
        slot.data_mut()[..samples.data.len()].copy_from_slice(&samples.data);

        // Preserve input metadata and update PTS/duration
        let mut metadata = buffer.metadata().clone();
        metadata.pts = crate::clock::ClockTime::from_nanos(samples.pts as u64);
        metadata.duration = crate::clock::ClockTime::from_nanos(samples.duration_nanos() as u64);

        Ok(Output::single(Buffer::new(
            MemoryHandle::with_len(slot, samples.data.len()),
            metadata,
        )))
    }

    fn flush(&mut self) -> Result<Output> {
        if self.flushed {
            return Ok(Output::None);
        }

        if !self.flushing {
            self.flushing = true;
        }

        // Try to flush decoder
        match self.decoder.flush()? {
            Some(samples) => {
                self.frames_out += 1;

                self.ensure_arena(samples.data.len())?;
                let arena = self.arena.as_ref().unwrap();
                arena.reclaim();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
                slot.data_mut()[..samples.data.len()].copy_from_slice(&samples.data);

                let mut metadata = crate::metadata::Metadata::new();
                metadata.pts = crate::clock::ClockTime::from_nanos(self.current_pts as u64);

                Ok(Output::single(Buffer::new(
                    MemoryHandle::with_len(slot, samples.data.len()),
                    metadata,
                )))
            }
            None => {
                self.flushed = true;
                Ok(Output::None)
            }
        }
    }

    fn name(&self) -> &str {
        "AudioDecoderElement"
    }

    fn execution_hints(&self) -> ExecutionHints {
        // Audio decoders typically use native code (libopus, etc.)
        ExecutionHints::native()
    }
}

impl<D: AudioDecoder> std::fmt::Debug for AudioDecoderElement<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioDecoderElement")
            .field("packets_in", &self.packets_in)
            .field("frames_out", &self.frames_out)
            .field("flushing", &self.flushing)
            .field("flushed", &self.flushed)
            .finish()
    }
}
