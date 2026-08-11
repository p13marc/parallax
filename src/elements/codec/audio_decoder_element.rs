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

use super::audio_traits::{AudioDecoder, AudioSamples};
use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::Result;
use crate::memory::{OutputArena, OutputBudget, defaults};

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
    output: OutputArena,
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
            output: OutputArena::new(defaults::AUDIO_SLOT_COUNT).with_min_slot_size(64 * 1024),
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

    /// Widen the arena when a decoded frame outgrows its slots.
    ///
    /// Slot size is fixed at build time, so a longer packet than anything seen
    /// so far needs a rebuild.
    fn ensure_arena(&mut self, min_size: usize) {
        if self
            .output
            .arena()
            .is_some_and(|a| a.slot_size() < min_size)
        {
            self.output.reset();
        }
        self.output.set_min_slot_size(min_size.max(64 * 1024));
    }
}

/// The in-band description of decoded PCM, from the decoder's own account of
/// what it produced.
fn pcm_media_format(samples: &AudioSamples) -> crate::format::MediaFormat {
    use super::audio_traits::AudioSampleFormat;
    crate::format::MediaFormat::AudioRaw(crate::format::AudioFormat::new(
        samples.sample_rate,
        samples.channels as u16,
        match samples.format {
            AudioSampleFormat::S16 => crate::format::SampleFormat::S16,
            AudioSampleFormat::S32 => crate::format::SampleFormat::S32,
            AudioSampleFormat::F32 => crate::format::SampleFormat::F32,
        },
    ))
}

impl<D: AudioDecoder + 'static> Transform for AudioDecoderElement<D> {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let packet = buffer.as_bytes();
        let input_pts = buffer.metadata().pts.nanos() as i64;
        self.packets_in += 1;

        // Decode packet
        let mut samples = self.decoder.decode(packet)?;

        // A packet can legitimately decode to nothing (Vorbis primes its
        // MDCT window on the first packet after init/seek) — emitting an
        // empty buffer downstream helps nobody.
        if samples.samples_per_channel == 0 {
            return Ok(Output::None);
        }

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
        self.ensure_arena(samples.data.len());
        let mut slot = self
            .output
            .acquire(samples.data.len(), "audiodecoderelement")?;
        slot.data_mut()[..samples.data.len()].copy_from_slice(&samples.data);
        // Hand the spent Vec back so the decoder reuses it next packet (#143).
        let data_len = samples.data.len();
        self.decoder.recycle(std::mem::take(&mut samples.data));

        // Preserve input metadata and update PTS/duration
        let mut metadata = buffer.metadata().clone();
        metadata.pts = crate::clock::ClockTime::from_nanos(samples.pts as u64);
        metadata.duration = crate::clock::ClockTime::from_nanos(samples.duration_nanos());

        // Describe the PCM in-band so downstream (audioconvert, sinks) can
        // configure from the buffer instead of out-of-band knowledge (#68).
        metadata.format = Some(pcm_media_format(&samples));

        Ok(Output::single(Buffer::new(
            MemoryHandle::with_len(slot, data_len),
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

                self.ensure_arena(samples.data.len());
                let mut slot = self
                    .output
                    .acquire(samples.data.len(), "audiodecoderelement")?;
                slot.data_mut()[..samples.data.len()].copy_from_slice(&samples.data);

                let mut metadata = crate::metadata::Metadata::new();
                metadata.pts = crate::clock::ClockTime::from_nanos(self.current_pts as u64);
                metadata.format = Some(pcm_media_format(&samples));

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
