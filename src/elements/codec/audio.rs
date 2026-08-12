//! Audio codec elements using pure Rust implementations.
//!
//! This module provides audio decoding elements using Symphonia (pure Rust).
//!
//! # Supported Codecs
//!
//! | Codec | Feature Flag | Decoder | Encoder |
//! |-------|--------------|---------|---------|
//! | FLAC | `audio-flac` | Yes | No |
//! | MP3 | `audio-mp3` | Yes | No |
//! | AAC | `audio-aac` | Yes | No |
//! | Vorbis | `audio-vorbis` | Yes | No |
//!
//! All decoders are implemented using Symphonia, a pure Rust audio library.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{SymphoniaDecoder, AudioFormat};
//!
//! // Create a decoder that auto-detects format
//! let decoder = SymphoniaDecoder::new()?;
//!
//! // Or specify format explicitly
//! let decoder = SymphoniaDecoder::for_format(AudioFormat::Mp3)?;
//! ```

use crate::error::{Error, Result};

/// Supported audio formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AudioFormat {
    /// FLAC lossless audio
    #[cfg(feature = "audio-flac")]
    Flac,
    /// MP3 (MPEG-1 Audio Layer 3)
    #[cfg(feature = "audio-mp3")]
    Mp3,
    /// AAC (Advanced Audio Coding)
    #[cfg(feature = "audio-aac")]
    Aac,
    /// Vorbis (Ogg Vorbis)
    #[cfg(feature = "audio-vorbis")]
    Vorbis,
    /// Raw PCM samples
    Pcm,
}

/// Sample format for audio data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleFormat {
    /// 16-bit signed integer
    S16,
    /// 32-bit signed integer
    S32,
    /// 32-bit float
    F32,
}

impl SampleFormat {
    /// Bytes per sample.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            Self::S16 => 2,
            Self::S32 | Self::F32 => 4,
        }
    }
}

/// Audio frame metadata.
#[derive(Clone, Debug)]
pub struct AudioFrameInfo {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// Sample format.
    pub sample_format: SampleFormat,
    /// Number of samples per channel in this frame.
    pub samples_per_channel: usize,
    /// Timestamp in samples.
    pub timestamp: u64,
}

impl AudioFrameInfo {
    /// Calculate total frame size in bytes.
    pub fn frame_size(&self) -> usize {
        self.samples_per_channel * self.channels as usize * self.sample_format.bytes_per_sample()
    }
}

// ============================================================================
// Symphonia-based Audio Decoder
// ============================================================================

#[cfg(any(
    feature = "audio-flac",
    feature = "audio-mp3",
    feature = "audio-aac",
    feature = "audio-vorbis"
))]
mod symphonia_decoder {
    use super::*;
    use symphonia::core::codecs::CodecParameters;
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;

    /// One-shot, container-probing audio decoder using Symphonia (pure Rust).
    ///
    /// Supports FLAC, MP3, AAC, and Vorbis depending on enabled features.
    ///
    /// **Not a pipeline element** (#160): it re-probes its accumulated input
    /// as a whole container on every call, which makes it unusable for
    /// demuxed streams — the streaming decoders (`AacDecoder`,
    /// `VorbisDecoder`, ...) are the element-system route. This remains as a
    /// convenience for decoding a complete in-memory file to PCM.
    pub struct SymphoniaDecoder {
        /// Hint for format detection.
        hint: Option<String>,
        /// Decoded frame count.
        frame_count: u64,
        /// Internal buffer for accumulating input data.
        input_buffer: Vec<u8>,
        /// Sample rate from last decoded frame.
        last_sample_rate: Option<u32>,
        /// Channel count from last decoded frame.
        last_channels: Option<u16>,
    }

    impl SymphoniaDecoder {
        /// Create a new decoder with auto-detection.
        pub fn new() -> Result<Self> {
            Ok(Self {
                hint: None,
                frame_count: 0,
                input_buffer: Vec::new(),
                last_sample_rate: None,
                last_channels: None,
            })
        }

        /// Create a decoder for a specific format.
        pub fn for_format(format: AudioFormat) -> Result<Self> {
            let hint = match format {
                #[cfg(feature = "audio-flac")]
                AudioFormat::Flac => Some("flac".to_string()),
                #[cfg(feature = "audio-mp3")]
                AudioFormat::Mp3 => Some("mp3".to_string()),
                #[cfg(feature = "audio-aac")]
                AudioFormat::Aac => Some("aac".to_string()),
                #[cfg(feature = "audio-vorbis")]
                AudioFormat::Vorbis => Some("ogg".to_string()),
                _ => None,
            };

            Ok(Self {
                hint,
                frame_count: 0,
                input_buffer: Vec::new(),
                last_sample_rate: None,
                last_channels: None,
            })
        }

        /// Get the number of frames decoded.
        pub fn frame_count(&self) -> u64 {
            self.frame_count
        }

        /// Feed a chunk of an audio *file* and try to decode.
        ///
        /// Accumulates input across calls; once enough of the container has
        /// arrived to probe, returns interleaved f32 samples plus their
        /// description. Feeding the entire file in one call is the intended
        /// use.
        pub fn decode_chunk(&mut self, input: &[u8]) -> Result<Option<(Vec<f32>, AudioFrameInfo)>> {
            // Accumulate input data
            self.input_buffer.extend_from_slice(input);

            // Need at least some data to probe
            if self.input_buffer.len() < 1024 {
                return Ok(None);
            }

            // Create a cursor over the accumulated data
            let cursor = std::io::Cursor::new(self.input_buffer.clone());
            let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

            // Set up hint
            let mut hint = Hint::new();
            if let Some(ref ext) = self.hint {
                hint.with_extension(ext);
            }

            // Probe the format
            let mut format = match symphonia::default::get_probe().probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            ) {
                Ok(f) => f,
                Err(_) => return Ok(None), // Not enough data or unknown format
            };

            // Find the first audio track with usable codec parameters
            let (track_id, audio_params) =
                match format.tracks().iter().find_map(|t| match &t.codec_params {
                    Some(CodecParameters::Audio(params)) => Some((t.id, params.clone())),
                    _ => None,
                }) {
                    Some(found) => found,
                    None => return Ok(None),
                };

            // Create decoder
            let mut decoder = match symphonia::default::get_codecs()
                .make_audio_decoder(&audio_params, &AudioDecoderOptions::default())
            {
                Ok(d) => d,
                Err(e) => {
                    return Err(Error::Config(format!(
                        "Failed to create audio decoder: {:?}",
                        e
                    )));
                }
            };

            // Get codec parameters
            let sample_rate = audio_params.sample_rate.unwrap_or(44100);
            let channels = audio_params
                .channels
                .as_ref()
                .map(|c| c.count() as u16)
                .unwrap_or(2);

            self.last_sample_rate = Some(sample_rate);
            self.last_channels = Some(channels);

            // Try to decode a packet
            let packet = match format.next_packet() {
                Ok(Some(p)) if p.track_id == track_id => p,
                Ok(_) => return Ok(None), // other track or end of stream
                Err(_) => return Ok(None),
            };

            let decoded = match decoder.decode(&packet) {
                Ok(d) => d,
                Err(_) => return Ok(None),
            };

            // Convert to interleaved f32 samples
            let mut samples: Vec<f32> = Vec::new();
            decoded.copy_to_vec_interleaved(&mut samples);
            let samples_per_channel = samples.len() / channels as usize;

            // Clear processed data (simplified - in real impl would track position)
            self.input_buffer.clear();

            let info = AudioFrameInfo {
                sample_rate,
                channels,
                sample_format: SampleFormat::F32,
                samples_per_channel,
                timestamp: self.frame_count,
            };

            self.frame_count += 1;

            Ok(Some((samples, info)))
        }
    }

    impl Default for SymphoniaDecoder {
        fn default() -> Self {
            Self::new().expect("Failed to create default decoder")
        }
    }
}

#[cfg(any(
    feature = "audio-flac",
    feature = "audio-mp3",
    feature = "audio-aac",
    feature = "audio-vorbis"
))]
pub use symphonia_decoder::SymphoniaDecoder;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_format_bytes() {
        assert_eq!(SampleFormat::S16.bytes_per_sample(), 2);
        assert_eq!(SampleFormat::S32.bytes_per_sample(), 4);
        assert_eq!(SampleFormat::F32.bytes_per_sample(), 4);
    }

    #[test]
    fn test_audio_frame_info_size() {
        let info = AudioFrameInfo {
            sample_rate: 48000,
            channels: 2,
            sample_format: SampleFormat::F32,
            samples_per_channel: 1024,
            timestamp: 0,
        };
        // 1024 samples * 2 channels * 4 bytes = 8192
        assert_eq!(info.frame_size(), 8192);
    }
}
