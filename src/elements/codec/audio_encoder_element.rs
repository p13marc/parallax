//! Generic audio encoder element wrapper.
//!
//! This module provides [`AudioEncoderElement`], a wrapper that adapts any
//! [`AudioEncoder`] to work as a pipeline element.
//!
//! # Features
//!
//! - Handles variable output (0, 1, or multiple packets per frame)
//! - Automatic flush at end-of-stream
//! - Timestamp preservation and calculation
//! - Statistics tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{AudioEncoderElement, OpusEncoder, OpusApplication};
//!
//! let encoder = OpusEncoder::new(48000, 2, 128000, OpusApplication::Audio)?;
//! let element = AudioEncoderElement::new(encoder);
//!
//! pipeline.add_node("encoder", DynAsyncElement::new_box(TransformAdapter::new(element)));
//! ```

use super::audio_traits::{AudioEncoder, AudioSampleFormat, AudioSamplesRef};
use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::Result;
use crate::memory::{OutputArena, OutputBudget, defaults};
use std::collections::VecDeque;

/// Wraps an [`AudioEncoder`] to work as a pipeline [`Transform`] element.
///
/// This wrapper handles:
/// - Viewing input buffers as [`AudioSamplesRef`]s (no copy)
/// - Managing encoder buffering
/// - Flushing remaining packets at EOS
/// - Calculating timestamps
pub struct AudioEncoderElement<E: AudioEncoder> {
    /// The underlying encoder.
    encoder: E,
    /// Queue of pending output packets (for multiple outputs per frame).
    pending_packets: VecDeque<(Vec<u8>, i64)>, // (data, pts)
    /// Whether we've started flushing.
    flushing: bool,
    /// Whether flush is complete.
    flushed: bool,
    /// Input sample rate.
    sample_rate: u32,
    /// Input channels.
    channels: u32,
    /// Input sample format.
    format: AudioSampleFormat,
    /// Statistics: frames received.
    frames_in: u64,
    /// Statistics: packets produced.
    packets_out: u64,
    /// Current timestamp in nanoseconds.
    current_pts: i64,
    /// Arena for output buffer allocation.
    output: OutputArena,
}

impl<E: AudioEncoder> AudioEncoderElement<E> {
    /// Create a new audio encoder element wrapper.
    ///
    /// # Arguments
    ///
    /// * `encoder` - The audio encoder to wrap
    /// * `sample_rate` - Expected input sample rate
    /// * `channels` - Expected input channels
    /// * `format` - Expected input sample format
    pub fn new(
        encoder: E,
        sample_rate: u32,
        channels: u32,
        format: AudioSampleFormat,
    ) -> Result<Self> {
        // Sized by the executor from link capacity; the slot floor is a
        // generous ceiling for a compressed audio packet (Opus tops out near
        // 4000 bytes).
        let output = OutputArena::new(defaults::AUDIO_SLOT_COUNT).with_min_slot_size(8192);

        Ok(Self {
            encoder,
            pending_packets: VecDeque::new(),
            flushing: false,
            flushed: false,
            sample_rate,
            channels,
            format,
            frames_in: 0,
            packets_out: 0,
            current_pts: 0,
            output,
        })
    }

    /// Create with default S16 format.
    pub fn new_s16(encoder: E, sample_rate: u32, channels: u32) -> Result<Self> {
        Self::new(encoder, sample_rate, channels, AudioSampleFormat::S16)
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

    /// View the input buffer as audio samples (no copy).
    ///
    /// The lifetime is named on purpose: the returned view borrows `buffer`,
    /// not `self`, so the caller can go on to `self.encoder.encode(samples)`
    /// while the view is live.
    fn buffer_to_samples<'a>(&self, buffer: &'a Buffer) -> AudioSamplesRef<'a> {
        let data = buffer.as_bytes();
        let pts = buffer.metadata().pts.nanos() as i64;

        let bytes_per_sample = self.format.bytes_per_sample();
        let total_samples = data.len() / bytes_per_sample;
        let samples_per_channel = total_samples / self.channels as usize;

        AudioSamplesRef {
            data,
            format: self.format,
            channels: self.channels,
            sample_rate: self.sample_rate,
            samples_per_channel,
            pts,
        }
    }

    /// Convert encoded packet to output buffer, preserving input metadata.
    fn packet_to_buffer(
        &mut self,
        data: Vec<u8>,
        pts: i64,
        input_metadata: &crate::metadata::Metadata,
    ) -> Result<Buffer> {
        let mut slot = self.output.acquire(data.len(), "audioencoderelement")?;
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
        let mut slot = self.output.acquire(data.len(), "audioencoderelement")?;
        slot.data_mut()[..data.len()].copy_from_slice(&data);

        let metadata =
            crate::metadata::Metadata::from_pts(crate::clock::ClockTime::from_nanos(pts as u64));

        Ok(Buffer::new(
            MemoryHandle::with_len(slot, data.len()),
            metadata,
        ))
    }
}

impl<E: AudioEncoder + 'static> Transform for AudioEncoderElement<E> {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        // Convert buffer to samples
        let samples = self.buffer_to_samples(&buffer);
        let input_metadata = buffer.metadata();
        let pts = if samples.pts != 0 {
            samples.pts
        } else {
            self.current_pts
        };

        // Update current PTS based on samples
        let duration_nanos = samples.duration_nanos() as i64;
        self.current_pts = pts + duration_nanos;
        self.frames_in += 1;

        // Encode samples
        let packets = self.encoder.encode(samples)?;

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
                self.pending_packets.push_back((data, self.current_pts));
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
        "AudioEncoderElement"
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::cpu_intensive()
    }
}

impl<E: AudioEncoder> std::fmt::Debug for AudioEncoderElement<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioEncoderElement")
            .field("sample_rate", &self.sample_rate)
            .field("channels", &self.channels)
            .field("format", &self.format)
            .field("frames_in", &self.frames_in)
            .field("packets_out", &self.packets_out)
            .field("flushing", &self.flushing)
            .field("flushed", &self.flushed)
            .finish()
    }
}
