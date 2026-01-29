//! Codec traits for audio encoders and decoders.
//!
//! This module defines standard traits that audio codecs must implement to work
//! with the pipeline element wrappers.
//!
//! # Overview
//!
//! - [`AudioEncoder`] - Trait for audio encoders (produce packets from samples)
//! - [`AudioDecoder`] - Trait for audio decoders (produce samples from packets)
//!
//! # Example: Implementing AudioEncoder
//!
//! ```rust,ignore
//! impl AudioEncoder for MyEncoder {
//!     type Packet = Vec<u8>;
//!
//!     fn encode(&mut self, samples: &AudioSamples) -> Result<Vec<Self::Packet>> {
//!         // Encode samples, may return 0 or more packets
//!     }
//!
//!     fn flush(&mut self) -> Result<Vec<Self::Packet>> {
//!         // Drain any buffered samples at EOS
//!     }
//! }
//! ```

use crate::error::Result;

/// Sample format for audio data.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AudioSampleFormat {
    /// 16-bit signed integer (little-endian).
    #[default]
    S16,
    /// 32-bit signed integer (little-endian).
    S32,
    /// 32-bit float.
    F32,
}

impl AudioSampleFormat {
    /// Bytes per sample.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::S16 => 2,
            Self::S32 | Self::F32 => 4,
        }
    }
}

/// Audio samples container for codec input/output.
///
/// Samples are stored interleaved: [L0, R0, L1, R1, ...] for stereo.
#[derive(Clone, Debug)]
pub struct AudioSamples {
    /// Raw sample data (interleaved).
    pub data: Vec<u8>,
    /// Sample format.
    pub format: AudioSampleFormat,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u32,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of samples per channel.
    pub samples_per_channel: usize,
    /// Presentation timestamp in nanoseconds.
    pub pts: i64,
}

impl AudioSamples {
    /// Create new audio samples container.
    pub fn new(
        data: Vec<u8>,
        format: AudioSampleFormat,
        channels: u32,
        sample_rate: u32,
        samples_per_channel: usize,
    ) -> Self {
        Self {
            data,
            format,
            channels,
            sample_rate,
            samples_per_channel,
            pts: 0,
        }
    }

    /// Create audio samples from S16 data.
    pub fn from_s16(data: &[i16], channels: u32, sample_rate: u32) -> Self {
        let samples_per_channel = data.len() / channels as usize;
        let bytes: Vec<u8> = data.iter().flat_map(|s| s.to_le_bytes()).collect();
        Self::new(
            bytes,
            AudioSampleFormat::S16,
            channels,
            sample_rate,
            samples_per_channel,
        )
    }

    /// Create audio samples from F32 data.
    pub fn from_f32(data: &[f32], channels: u32, sample_rate: u32) -> Self {
        let samples_per_channel = data.len() / channels as usize;
        let bytes: Vec<u8> = data.iter().flat_map(|s| s.to_le_bytes()).collect();
        Self::new(
            bytes,
            AudioSampleFormat::F32,
            channels,
            sample_rate,
            samples_per_channel,
        )
    }

    /// Get samples as S16 slice (only valid if format is S16).
    pub fn as_s16(&self) -> Option<&[i16]> {
        if self.format != AudioSampleFormat::S16 {
            return None;
        }
        // SAFETY: data is aligned and was created from i16 values
        Some(unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const i16, self.data.len() / 2)
        })
    }

    /// Get samples as F32 slice (only valid if format is F32).
    pub fn as_f32(&self) -> Option<&[f32]> {
        if self.format != AudioSampleFormat::F32 {
            return None;
        }
        // SAFETY: data is aligned and was created from f32 values
        Some(unsafe {
            std::slice::from_raw_parts(self.data.as_ptr() as *const f32, self.data.len() / 4)
        })
    }

    /// Total frame size in bytes.
    pub fn frame_size(&self) -> usize {
        self.samples_per_channel * self.channels as usize * self.format.bytes_per_sample()
    }

    /// Duration in nanoseconds.
    pub fn duration_nanos(&self) -> u64 {
        if self.sample_rate == 0 {
            return 0;
        }
        (self.samples_per_channel as u64 * 1_000_000_000) / self.sample_rate as u64
    }
}

/// Trait for audio encoders.
///
/// Audio encoders take raw PCM samples and produce encoded packets.
/// Due to frame buffering, there may not be a 1:1 correspondence between
/// input samples and output packets.
///
/// # Buffering Behavior
///
/// - `encode()` may return 0 packets (encoder is buffering)
/// - `encode()` may return 1 packet (typical case)
/// - `encode()` may return multiple packets (accumulated frames)
/// - `flush()` must be called at EOS to drain all remaining packets
///
/// # Example
///
/// ```rust,ignore
/// let mut encoder = OpusEncoder::new(48000, 2, 128000)?;
///
/// // Encode audio frames
/// for samples in audio_frames {
///     for packet in encoder.encode(&samples)? {
///         // Process encoded packet
///     }
/// }
///
/// // Flush remaining frames at EOS
/// for packet in encoder.flush()? {
///     // Process remaining packets
/// }
/// ```
pub trait AudioEncoder: Send {
    /// Encoded packet type (usually `Vec<u8>`).
    type Packet: AsRef<[u8]> + Send;

    /// Encode audio samples.
    ///
    /// Returns zero or more encoded packets. The encoder may buffer
    /// samples internally to form complete frames.
    fn encode(&mut self, samples: &AudioSamples) -> Result<Vec<Self::Packet>>;

    /// Flush any buffered samples at end-of-stream.
    ///
    /// Must be called after all samples have been sent to drain the
    /// encoder's internal buffers.
    fn flush(&mut self) -> Result<Vec<Self::Packet>>;

    /// Get the required frame size in samples per channel.
    ///
    /// Some codecs (like Opus) require specific frame sizes.
    /// Returns `None` if the encoder accepts any frame size.
    fn frame_size(&self) -> Option<usize> {
        None
    }

    /// Get codec-specific header data (optional).
    ///
    /// For Opus: OpusHead structure
    /// For AAC: AudioSpecificConfig
    fn codec_data(&self) -> Option<Vec<u8>> {
        None
    }

    /// Check if encoder has buffered samples.
    fn has_pending(&self) -> bool {
        false
    }

    /// Get the output sample rate (may differ from input for resampling encoders).
    fn output_sample_rate(&self) -> u32;

    /// Get the number of output channels.
    fn output_channels(&self) -> u32;
}

/// Trait for audio decoders.
///
/// Audio decoders take encoded packets and produce raw PCM samples.
///
/// # Example
///
/// ```rust,ignore
/// let mut decoder = OpusDecoder::new(48000, 2)?;
///
/// for packet in packets {
///     let samples = decoder.decode(&packet)?;
///     // Process decoded samples
/// }
///
/// // Flush at EOS
/// let remaining = decoder.flush()?;
/// ```
pub trait AudioDecoder: Send {
    /// Decode an encoded packet.
    ///
    /// Returns decoded audio samples.
    fn decode(&mut self, packet: &[u8]) -> Result<AudioSamples>;

    /// Flush any buffered samples at end-of-stream.
    fn flush(&mut self) -> Result<Option<AudioSamples>>;

    /// Check if decoder has buffered samples.
    fn has_pending(&self) -> bool {
        false
    }

    /// Get the output sample rate.
    fn output_sample_rate(&self) -> u32;

    /// Get the number of output channels.
    fn output_channels(&self) -> u32;

    /// Get the output sample format.
    fn output_format(&self) -> AudioSampleFormat;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_format_bytes() {
        assert_eq!(AudioSampleFormat::S16.bytes_per_sample(), 2);
        assert_eq!(AudioSampleFormat::S32.bytes_per_sample(), 4);
        assert_eq!(AudioSampleFormat::F32.bytes_per_sample(), 4);
    }

    #[test]
    fn test_audio_samples_from_s16() {
        let data: Vec<i16> = vec![0, 100, -100, 32767, -32768, 0];
        let samples = AudioSamples::from_s16(&data, 2, 48000);

        assert_eq!(samples.channels, 2);
        assert_eq!(samples.sample_rate, 48000);
        assert_eq!(samples.samples_per_channel, 3);
        assert_eq!(samples.format, AudioSampleFormat::S16);
        assert_eq!(samples.data.len(), 12); // 6 samples * 2 bytes
    }

    #[test]
    fn test_audio_samples_from_f32() {
        let data: Vec<f32> = vec![0.0, 0.5, -0.5, 1.0];
        let samples = AudioSamples::from_f32(&data, 2, 44100);

        assert_eq!(samples.channels, 2);
        assert_eq!(samples.sample_rate, 44100);
        assert_eq!(samples.samples_per_channel, 2);
        assert_eq!(samples.format, AudioSampleFormat::F32);
        assert_eq!(samples.data.len(), 16); // 4 samples * 4 bytes
    }

    #[test]
    fn test_audio_samples_as_s16() {
        let original: Vec<i16> = vec![100, -100, 32767, -32768];
        let samples = AudioSamples::from_s16(&original, 2, 48000);

        let recovered = samples.as_s16().unwrap();
        assert_eq!(recovered, &original[..]);
    }

    #[test]
    fn test_audio_samples_duration() {
        // 48000 samples at 48000 Hz = 1 second = 1_000_000_000 ns
        let samples = AudioSamples::new(
            vec![0; 48000 * 2 * 2], // 48000 samples * 2 channels * 2 bytes
            AudioSampleFormat::S16,
            2,
            48000,
            48000,
        );
        assert_eq!(samples.duration_nanos(), 1_000_000_000);
    }

    #[test]
    fn test_audio_samples_frame_size() {
        let samples = AudioSamples::new(
            vec![0; 1920 * 2 * 2], // 1920 samples * 2 channels * 2 bytes
            AudioSampleFormat::S16,
            2,
            48000,
            1920, // 40ms at 48kHz
        );
        assert_eq!(samples.frame_size(), 1920 * 2 * 2);
    }
}
