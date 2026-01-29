//! Opus audio codec implementation.
//!
//! This module provides Opus encoding and decoding using the `opus` crate
//! (bindings to libopus).
//!
//! # Features
//!
//! - High-quality audio at low bitrates (6-510 kbps)
//! - Supports speech and music
//! - Low latency (2.5ms to 60ms frames)
//! - Variable bitrate (VBR) and constant bitrate (CBR)
//!
//! # Frame Sizes
//!
//! Opus supports specific frame sizes at 48kHz:
//! - 120 samples (2.5ms)
//! - 240 samples (5ms)
//! - 480 samples (10ms)
//! - 960 samples (20ms) - default
//! - 1920 samples (40ms)
//! - 2880 samples (60ms)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{OpusEncoder, OpusDecoder, OpusApplication};
//!
//! // Create encoder for music at 128 kbps
//! let mut encoder = OpusEncoder::new(48000, 2, 128000, OpusApplication::Audio)?;
//!
//! // Create decoder
//! let mut decoder = OpusDecoder::new(48000, 2)?;
//!
//! // Encode samples (must be 960 samples per channel for 20ms frames)
//! let samples = AudioSamples::from_s16(&pcm_data, 2, 48000);
//! let packets = encoder.encode(&samples)?;
//!
//! // Decode packets
//! for packet in packets {
//!     let decoded = decoder.decode(packet.as_ref())?;
//! }
//! ```
//!
//! # Build Dependencies
//!
//! Requires libopus system library:
//!
//! - **Fedora/RHEL**: `sudo dnf install opus-devel`
//! - **Debian/Ubuntu**: `sudo apt install libopus-dev`
//! - **Arch**: `sudo pacman -S opus`
//! - **macOS**: `brew install opus`

use super::audio_traits::{AudioDecoder, AudioEncoder, AudioSampleFormat, AudioSamples};
use crate::error::{Error, Result};

use opus::{Application, Channels, Decoder as OpusDecoderInner, Encoder as OpusEncoderInner};

/// Opus application mode.
///
/// Affects encoding optimization strategy.
#[derive(Clone, Copy, Debug, Default)]
pub enum OpusApplication {
    /// Optimized for voice/speech (VOIP).
    Voip,
    /// Optimized for music and general audio.
    #[default]
    Audio,
    /// Restricted low-delay mode for real-time applications.
    LowDelay,
}

impl From<OpusApplication> for Application {
    fn from(app: OpusApplication) -> Self {
        match app {
            OpusApplication::Voip => Application::Voip,
            OpusApplication::Audio => Application::Audio,
            OpusApplication::LowDelay => Application::LowDelay,
        }
    }
}

/// Validate sample rate for Opus.
fn validate_sample_rate(rate: u32) -> Result<u32> {
    match rate {
        8000 | 12000 | 16000 | 24000 | 48000 => Ok(rate),
        _ => Err(Error::Config(format!(
            "Opus only supports sample rates: 8000, 12000, 16000, 24000, 48000 Hz (got {})",
            rate
        ))),
    }
}

/// Convert channel count to opus Channels.
fn to_opus_channels(channels: u32) -> Result<Channels> {
    match channels {
        1 => Ok(Channels::Mono),
        2 => Ok(Channels::Stereo),
        _ => Err(Error::Config(format!(
            "Opus only supports mono (1) or stereo (2) channels (got {})",
            channels
        ))),
    }
}

/// Opus audio encoder.
///
/// Encodes PCM audio to Opus packets. Input must be S16 interleaved samples.
///
/// # Frame Size Requirements
///
/// Opus requires input to be exact frame sizes. At 48kHz, valid sizes are:
/// - 120, 240, 480, 960, 1920, 2880 samples per channel
///
/// The encoder buffers input and produces packets when enough samples are available.
pub struct OpusEncoder {
    /// Inner opus encoder.
    encoder: OpusEncoderInner,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Number of channels.
    channels: u32,
    /// Target bitrate in bits per second.
    #[allow(dead_code)]
    bitrate: u32,
    /// Frame size in samples per channel (default: 960 = 20ms at 48kHz).
    frame_size: usize,
    /// Internal buffer for accumulating samples.
    buffer: Vec<i16>,
    /// Packets produced.
    packets_out: u64,
}

impl OpusEncoder {
    /// Create a new Opus encoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate (8000, 12000, 16000, 24000, or 48000 Hz)
    /// * `channels` - Number of channels (1 or 2)
    /// * `bitrate` - Target bitrate in bits per second (6000-510000)
    /// * `application` - Encoding mode optimization
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let encoder = OpusEncoder::new(48000, 2, 128000, OpusApplication::Audio)?;
    /// ```
    pub fn new(
        sample_rate: u32,
        channels: u32,
        bitrate: u32,
        application: OpusApplication,
    ) -> Result<Self> {
        let _ = validate_sample_rate(sample_rate)?;
        let opus_channels = to_opus_channels(channels)?;

        let mut encoder = OpusEncoderInner::new(sample_rate, opus_channels, application.into())
            .map_err(|e| Error::Config(format!("Failed to create Opus encoder: {:?}", e)))?;

        // Set bitrate
        encoder
            .set_bitrate(opus::Bitrate::Bits(bitrate as i32))
            .map_err(|e| Error::Config(format!("Failed to set Opus bitrate: {:?}", e)))?;

        // Default frame size: 20ms at the given sample rate
        let frame_size = (sample_rate as usize * 20) / 1000;

        Ok(Self {
            encoder,
            sample_rate,
            channels,
            bitrate,
            frame_size,
            buffer: Vec::new(),
            packets_out: 0,
        })
    }

    /// Set the frame size in samples per channel.
    ///
    /// Valid values at 48kHz: 120, 240, 480, 960, 1920, 2880
    pub fn set_frame_size(&mut self, samples_per_channel: usize) -> Result<()> {
        // Validate frame size (must correspond to 2.5, 5, 10, 20, 40, or 60 ms)
        let duration_us = (samples_per_channel * 1_000_000) / self.sample_rate as usize;
        match duration_us {
            2500 | 5000 | 10000 | 20000 | 40000 | 60000 => {
                self.frame_size = samples_per_channel;
                Ok(())
            }
            _ => Err(Error::Config(format!(
                "Invalid Opus frame size: {} samples ({} us). Must be 2.5, 5, 10, 20, 40, or 60 ms",
                samples_per_channel, duration_us
            ))),
        }
    }

    /// Get the number of packets produced.
    pub fn packets_out(&self) -> u64 {
        self.packets_out
    }

    /// Encode a single frame (internal).
    fn encode_frame(&mut self, frame: &[i16]) -> Result<Vec<u8>> {
        // Maximum Opus packet size
        let output = self
            .encoder
            .encode_vec(frame, 4000)
            .map_err(|e| Error::Element(format!("Opus encode error: {:?}", e)))?;

        self.packets_out += 1;
        Ok(output)
    }
}

impl AudioEncoder for OpusEncoder {
    type Packet = Vec<u8>;

    fn encode(&mut self, samples: &AudioSamples) -> Result<Vec<Self::Packet>> {
        // Convert input to S16 if needed
        let input_s16: Vec<i16> = match samples.format {
            AudioSampleFormat::S16 => samples
                .as_s16()
                .ok_or_else(|| Error::Element("Failed to get S16 samples".to_string()))?
                .to_vec(),
            AudioSampleFormat::F32 => {
                // Convert F32 to S16
                samples
                    .as_f32()
                    .ok_or_else(|| Error::Element("Failed to get F32 samples".to_string()))?
                    .iter()
                    .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0) as i16)
                    .collect()
            }
            AudioSampleFormat::S32 => {
                // Convert S32 to S16
                let s32_data = unsafe {
                    std::slice::from_raw_parts(
                        samples.data.as_ptr() as *const i32,
                        samples.data.len() / 4,
                    )
                };
                s32_data.iter().map(|&s| (s >> 16) as i16).collect()
            }
        };

        // Add to buffer
        self.buffer.extend_from_slice(&input_s16);

        // Encode complete frames
        let frame_samples = self.frame_size * self.channels as usize;
        let mut packets = Vec::new();

        while self.buffer.len() >= frame_samples {
            let frame: Vec<i16> = self.buffer.drain(..frame_samples).collect();
            packets.push(self.encode_frame(&frame)?);
        }

        Ok(packets)
    }

    fn flush(&mut self) -> Result<Vec<Self::Packet>> {
        if self.buffer.is_empty() {
            return Ok(Vec::new());
        }

        // Pad remaining samples with silence to complete a frame
        let frame_samples = self.frame_size * self.channels as usize;
        let padding_needed = frame_samples - self.buffer.len();
        self.buffer
            .extend(std::iter::repeat(0i16).take(padding_needed));

        let frame: Vec<i16> = self.buffer.drain(..).collect();
        Ok(vec![self.encode_frame(&frame)?])
    }

    fn frame_size(&self) -> Option<usize> {
        Some(self.frame_size)
    }

    fn codec_data(&self) -> Option<Vec<u8>> {
        // Opus doesn't require codec-specific data for basic operation
        // OpusHead would go here for Ogg encapsulation
        None
    }

    fn has_pending(&self) -> bool {
        !self.buffer.is_empty()
    }

    fn output_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn output_channels(&self) -> u32 {
        self.channels
    }
}

/// Opus audio decoder.
///
/// Decodes Opus packets to PCM audio (S16 interleaved samples).
pub struct OpusDecoder {
    /// Inner opus decoder.
    decoder: OpusDecoderInner,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Number of channels.
    channels: u32,
    /// Packets decoded.
    packets_in: u64,
}

impl OpusDecoder {
    /// Create a new Opus decoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Output sample rate (8000, 12000, 16000, 24000, or 48000 Hz)
    /// * `channels` - Number of output channels (1 or 2)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let decoder = OpusDecoder::new(48000, 2)?;
    /// ```
    pub fn new(sample_rate: u32, channels: u32) -> Result<Self> {
        let _ = validate_sample_rate(sample_rate)?;
        let opus_channels = to_opus_channels(channels)?;

        let decoder = OpusDecoderInner::new(sample_rate, opus_channels)
            .map_err(|e| Error::Config(format!("Failed to create Opus decoder: {:?}", e)))?;

        Ok(Self {
            decoder,
            sample_rate,
            channels,
            packets_in: 0,
        })
    }

    /// Get the number of packets decoded.
    pub fn packets_in(&self) -> u64 {
        self.packets_in
    }
}

impl AudioDecoder for OpusDecoder {
    fn decode(&mut self, packet: &[u8]) -> Result<AudioSamples> {
        // Maximum frame size: 120ms at 48kHz = 5760 samples per channel
        // With stereo: 5760 * 2 = 11520 samples
        let max_samples = 5760 * self.channels as usize;
        let mut output = vec![0i16; max_samples];

        let decoded_samples = self
            .decoder
            .decode(packet, &mut output, false)
            .map_err(|e| Error::Element(format!("Opus decode error: {:?}", e)))?;

        self.packets_in += 1;

        // Trim to actual decoded size
        output.truncate(decoded_samples * self.channels as usize);

        // Convert to bytes
        let bytes: Vec<u8> = output.iter().flat_map(|s| s.to_le_bytes()).collect();

        Ok(AudioSamples {
            data: bytes,
            format: AudioSampleFormat::S16,
            channels: self.channels,
            sample_rate: self.sample_rate,
            samples_per_channel: decoded_samples,
            pts: 0, // Will be set by element wrapper
        })
    }

    fn flush(&mut self) -> Result<Option<AudioSamples>> {
        // Opus decoder doesn't buffer, nothing to flush
        Ok(None)
    }

    fn output_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn output_channels(&self) -> u32 {
        self.channels
    }

    fn output_format(&self) -> AudioSampleFormat {
        AudioSampleFormat::S16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_opus_encoder_creation() {
        let encoder = OpusEncoder::new(48000, 2, 128000, OpusApplication::Audio);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_opus_encoder_invalid_sample_rate() {
        let encoder = OpusEncoder::new(44100, 2, 128000, OpusApplication::Audio);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_opus_encoder_invalid_channels() {
        let encoder = OpusEncoder::new(48000, 4, 128000, OpusApplication::Audio);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_opus_decoder_creation() {
        let decoder = OpusDecoder::new(48000, 2);
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_opus_encode_decode_roundtrip() {
        let mut encoder = OpusEncoder::new(48000, 2, 128000, OpusApplication::Audio).unwrap();
        let mut decoder = OpusDecoder::new(48000, 2).unwrap();

        // Generate a 20ms sine wave (960 samples per channel at 48kHz)
        let samples_per_channel = 960;
        let mut pcm: Vec<i16> = Vec::with_capacity(samples_per_channel * 2);
        for i in 0..samples_per_channel {
            let t = i as f32 / 48000.0;
            let sample = ((t * 440.0 * 2.0 * std::f32::consts::PI).sin() * 16000.0) as i16;
            pcm.push(sample); // Left
            pcm.push(sample); // Right
        }

        let input = AudioSamples::from_s16(&pcm, 2, 48000);
        let packets = encoder.encode(&input).unwrap();

        // Should produce exactly 1 packet for 20ms frame
        assert_eq!(packets.len(), 1);

        // Decode the packet
        let decoded = decoder.decode(&packets[0]).unwrap();

        // Should have same number of samples
        assert_eq!(decoded.samples_per_channel, samples_per_channel);
        assert_eq!(decoded.channels, 2);
        assert_eq!(decoded.sample_rate, 48000);
    }

    #[test]
    fn test_opus_encoder_buffering() {
        let mut encoder = OpusEncoder::new(48000, 2, 128000, OpusApplication::Audio).unwrap();

        // Send less than a full frame (960 samples)
        let small_input: Vec<i16> = vec![0i16; 480 * 2]; // 480 samples stereo
        let samples = AudioSamples::from_s16(&small_input, 2, 48000);

        let packets = encoder.encode(&samples).unwrap();
        // Should buffer, no output yet
        assert!(packets.is_empty());
        assert!(encoder.has_pending());

        // Send more samples to complete the frame
        let packets = encoder.encode(&samples).unwrap();
        // Now should output a packet
        assert_eq!(packets.len(), 1);
    }

    #[test]
    fn test_opus_encoder_flush() {
        let mut encoder = OpusEncoder::new(48000, 2, 128000, OpusApplication::Audio).unwrap();

        // Send partial frame
        let small_input: Vec<i16> = vec![0i16; 100 * 2]; // 100 samples stereo
        let samples = AudioSamples::from_s16(&small_input, 2, 48000);
        encoder.encode(&samples).unwrap();

        // Flush should pad and encode remaining
        let packets = encoder.flush().unwrap();
        assert_eq!(packets.len(), 1);
        assert!(!encoder.has_pending());
    }

    #[test]
    fn test_opus_frame_size_setting() {
        let mut encoder = OpusEncoder::new(48000, 2, 128000, OpusApplication::Audio).unwrap();

        // Valid frame sizes
        assert!(encoder.set_frame_size(120).is_ok()); // 2.5ms
        assert!(encoder.set_frame_size(240).is_ok()); // 5ms
        assert!(encoder.set_frame_size(480).is_ok()); // 10ms
        assert!(encoder.set_frame_size(960).is_ok()); // 20ms
        assert!(encoder.set_frame_size(1920).is_ok()); // 40ms
        assert!(encoder.set_frame_size(2880).is_ok()); // 60ms

        // Invalid frame size
        assert!(encoder.set_frame_size(1000).is_err());
    }
}
