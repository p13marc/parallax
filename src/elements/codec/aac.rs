//! AAC audio encoder implementation.
//!
//! This module provides AAC encoding using the `fdk-aac` crate
//! (bindings to Fraunhofer FDK-AAC).
//!
//! # License Warning
//!
//! **FDK-AAC has patent license restrictions for commercial use.**
//! Please review the FDK-AAC license before using in commercial products.
//!
//! # Features
//!
//! - High-quality AAC-LC encoding
//! - Supports mono and stereo
//! - Configurable bitrate (64-320 kbps typical)
//! - Common for streaming (HLS, DASH)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::AacEncoder;
//!
//! // Create encoder for stereo at 128 kbps
//! let mut encoder = AacEncoder::new(44100, 2, 128000)?;
//!
//! // Encode samples (S16 interleaved)
//! let samples = AudioSamples::from_s16(&pcm_data, 2, 44100);
//! let packets = encoder.encode(&samples)?;
//! ```
//!
//! # Build Dependencies
//!
//! Requires FDK-AAC library:
//!
//! - **Fedora/RHEL**: `sudo dnf install fdk-aac-devel` (from RPM Fusion)
//! - **Debian/Ubuntu**: `sudo apt install libfdk-aac-dev`
//! - **Arch**: `sudo pacman -S libfdk-aac`
//! - **macOS**: `brew install fdk-aac`

use super::audio_traits::{AudioEncoder, AudioSampleFormat, AudioSamples};
use crate::error::{Error, Result};

use fdk_aac::enc::{BitRate, ChannelMode, Encoder, EncoderParams};

/// AAC audio encoder.
///
/// Encodes PCM audio to AAC-LC packets using FDK-AAC.
///
/// # License Note
///
/// FDK-AAC is not fully royalty-free. Review license terms before commercial use.
pub struct AacEncoder {
    /// Inner FDK-AAC encoder.
    encoder: Encoder,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Number of channels.
    channels: u32,
    /// Target bitrate in bits per second.
    bitrate: u32,
    /// Frame size (samples per channel per frame).
    frame_size: usize,
    /// Internal buffer for accumulating samples.
    buffer: Vec<i16>,
    /// Packets produced.
    packets_out: u64,
}

impl AacEncoder {
    /// Create a new AAC encoder.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate (typically 44100 or 48000 Hz)
    /// * `channels` - Number of channels (1 or 2)
    /// * `bitrate` - Target bitrate in bits per second (64000-320000 typical)
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let encoder = AacEncoder::new(44100, 2, 128000)?;
    /// ```
    pub fn new(sample_rate: u32, channels: u32, bitrate: u32) -> Result<Self> {
        let channel_mode = match channels {
            1 => ChannelMode::Mono,
            2 => ChannelMode::Stereo,
            _ => {
                return Err(Error::Config(format!(
                    "AAC only supports mono (1) or stereo (2) channels (got {})",
                    channels
                )));
            }
        };

        let params = EncoderParams {
            bit_rate: BitRate::Cbr(bitrate),
            sample_rate,
            transport: fdk_aac::enc::Transport::Raw,
            channels: channel_mode,
        };

        let encoder = Encoder::new(params)
            .map_err(|e| Error::Config(format!("Failed to create AAC encoder: {:?}", e)))?;

        // AAC-LC typically uses 1024 samples per frame
        let frame_size = 1024;

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

    /// Get the number of packets produced.
    pub fn packets_out(&self) -> u64 {
        self.packets_out
    }

    /// Get the required frame size in samples per channel.
    pub fn required_frame_size(&self) -> usize {
        self.frame_size
    }

    /// Encode a single frame (internal).
    fn encode_frame(&mut self, frame: &[i16]) -> Result<Vec<u8>> {
        let output = self
            .encoder
            .encode(frame)
            .map_err(|e| Error::Element(format!("AAC encode error: {:?}", e)))?;

        self.packets_out += 1;
        Ok(output)
    }
}

impl AudioEncoder for AacEncoder {
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
        // AAC AudioSpecificConfig would go here
        // FDK-AAC provides this via encoder.info()
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_aac_encoder_creation() {
        let encoder = AacEncoder::new(44100, 2, 128000);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_aac_encoder_invalid_channels() {
        let encoder = AacEncoder::new(44100, 4, 128000);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_aac_encode_frame() {
        let mut encoder = AacEncoder::new(44100, 2, 128000).unwrap();

        // Generate 1024 samples (one AAC frame) of silence, stereo
        let pcm: Vec<i16> = vec![0i16; 1024 * 2];
        let samples = AudioSamples::from_s16(&pcm, 2, 44100);

        let packets = encoder.encode(&samples).unwrap();
        // Should produce exactly 1 packet
        assert_eq!(packets.len(), 1);
        // AAC packet should have data
        assert!(!packets[0].is_empty());
    }

    #[test]
    fn test_aac_encoder_buffering() {
        let mut encoder = AacEncoder::new(44100, 2, 128000).unwrap();

        // Send less than a full frame (1024 samples)
        let small_input: Vec<i16> = vec![0i16; 512 * 2];
        let samples = AudioSamples::from_s16(&small_input, 2, 44100);

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
    fn test_aac_encoder_flush() {
        let mut encoder = AacEncoder::new(44100, 2, 128000).unwrap();

        // Send partial frame
        let small_input: Vec<i16> = vec![0i16; 100 * 2];
        let samples = AudioSamples::from_s16(&small_input, 2, 44100);
        encoder.encode(&samples).unwrap();

        // Flush should pad and encode remaining
        let packets = encoder.flush().unwrap();
        assert_eq!(packets.len(), 1);
        assert!(!encoder.has_pending());
    }
}
