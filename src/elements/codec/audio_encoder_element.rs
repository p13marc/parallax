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

use super::audio_traits::{AudioEncoder, AudioSampleFormat, AudioSamples};
use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use std::collections::VecDeque;

/// Wraps an [`AudioEncoder`] to work as a pipeline [`Transform`] element.
///
/// This wrapper handles:
/// - Converting input buffers to [`AudioSamples`]
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
    arena: SharedArena,
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
        // Estimate max packet size (typical Opus max is ~4000 bytes)
        let max_packet_size = 8192;
        let arena = SharedArena::new(max_packet_size, 32)
            .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?;

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
            arena,
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

    /// Convert input buffer to AudioSamples.
    fn buffer_to_samples(&self, buffer: &Buffer) -> AudioSamples {
        let data = buffer.as_bytes().to_vec();
        let pts = buffer.metadata().pts.nanos() as i64;

        let bytes_per_sample = self.format.bytes_per_sample();
        let total_samples = data.len() / bytes_per_sample;
        let samples_per_channel = total_samples / self.channels as usize;

        AudioSamples {
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

impl<E: AudioEncoder + 'static> Transform for AudioEncoderElement<E> {
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
        let packets = self.encoder.encode(&samples)?;

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
