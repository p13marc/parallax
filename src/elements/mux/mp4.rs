//! MP4/MOV container muxer.
//!
//! This module provides an MP4 muxer that creates MP4/MOV container files
//! from elementary streams (video, audio).
//!
//! # Supported Codecs
//!
//! | Type | Codec | Notes |
//! |------|-------|-------|
//! | Video | H.264/AVC | Most common |
//! | Video | H.265/HEVC | High efficiency |
//! | Video | VP9 | WebM compatible |
//! | Audio | AAC | Most common |
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig, Mp4AudioTrackConfig};
//! use std::fs::File;
//! use std::io::BufWriter;
//!
//! let file = File::create("output.mp4")?;
//! let writer = BufWriter::new(file);
//!
//! let config = Mp4MuxConfig::default();
//! let mut mux = Mp4Mux::new(writer, config)?;
//!
//! // Add video track
//! let video_config = Mp4VideoTrackConfig::h264(1920, 1080, &sps, &pps);
//! let video_track = mux.add_video_track(video_config)?;
//!
//! // Add audio track
//! let audio_config = Mp4AudioTrackConfig::aac(44100, 2);
//! let audio_track = mux.add_audio_track(audio_config)?;
//!
//! // Write samples
//! mux.write_sample(video_track, &video_data, pts, dts, is_keyframe)?;
//! mux.write_sample(audio_track, &audio_data, pts, dts, true)?;
//!
//! // Finalize
//! mux.finish()?;
//! ```

use crate::error::{Error, Result};

use mp4::{
    AacConfig, AudioObjectType, AvcConfig, ChannelConfig, HevcConfig, MediaConfig, Mp4Config,
    Mp4Sample, Mp4Writer, SampleFreqIndex, TrackConfig, Vp9Config,
};
use std::io::{Seek, Write};

// ============================================================================
// Muxer Configuration
// ============================================================================

/// MP4 muxer configuration.
#[derive(Debug, Clone)]
pub struct Mp4MuxConfig {
    /// Major brand (default: "isom").
    pub major_brand: String,
    /// Minor version (default: 512).
    pub minor_version: u32,
    /// Compatible brands (default: ["isom", "iso2", "avc1", "mp41"]).
    pub compatible_brands: Vec<String>,
    /// Timescale (ticks per second, default: 1000).
    pub timescale: u32,
}

impl Default for Mp4MuxConfig {
    fn default() -> Self {
        Self {
            major_brand: "isom".to_string(),
            minor_version: 512,
            compatible_brands: vec![
                "isom".to_string(),
                "iso2".to_string(),
                "avc1".to_string(),
                "mp41".to_string(),
            ],
            timescale: 1000,
        }
    }
}

impl Mp4MuxConfig {
    /// Create a configuration optimized for H.264 video.
    pub fn h264() -> Self {
        Self::default()
    }

    /// Create a configuration optimized for H.265/HEVC video.
    pub fn hevc() -> Self {
        Self {
            major_brand: "isom".to_string(),
            minor_version: 512,
            compatible_brands: vec![
                "isom".to_string(),
                "iso2".to_string(),
                "hev1".to_string(),
                "mp41".to_string(),
            ],
            timescale: 1000,
        }
    }

    /// Create a configuration optimized for VP9 video.
    pub fn vp9() -> Self {
        Self {
            major_brand: "isom".to_string(),
            minor_version: 512,
            compatible_brands: vec!["isom".to_string(), "iso2".to_string(), "mp41".to_string()],
            timescale: 1000,
        }
    }
}

// ============================================================================
// Video Track Configuration
// ============================================================================

/// Video track configuration for the MP4 muxer.
#[derive(Debug, Clone)]
pub struct Mp4VideoTrackConfig {
    /// Video width in pixels.
    pub width: u16,
    /// Video height in pixels.
    pub height: u16,
    /// Video codec configuration.
    pub codec: VideoCodecConfig,
}

/// Video codec configuration.
#[derive(Debug, Clone)]
pub enum VideoCodecConfig {
    /// H.264/AVC configuration.
    H264 {
        /// Sequence Parameter Set (raw bytes without start code).
        sps: Vec<u8>,
        /// Picture Parameter Set (raw bytes without start code).
        pps: Vec<u8>,
    },
    /// H.265/HEVC configuration.
    /// Note: mp4 crate v0.14 only stores dimensions, not VPS/SPS/PPS.
    H265,
    /// VP9 configuration.
    /// Note: mp4 crate v0.14 only stores dimensions.
    Vp9,
}

impl Mp4VideoTrackConfig {
    /// Create H.264/AVC video track configuration.
    ///
    /// # Arguments
    ///
    /// * `width` - Video width in pixels.
    /// * `height` - Video height in pixels.
    /// * `sps` - Sequence Parameter Set (raw bytes without start code).
    /// * `pps` - Picture Parameter Set (raw bytes without start code).
    pub fn h264(width: u16, height: u16, sps: &[u8], pps: &[u8]) -> Self {
        Self {
            width,
            height,
            codec: VideoCodecConfig::H264 {
                sps: sps.to_vec(),
                pps: pps.to_vec(),
            },
        }
    }

    /// Create H.265/HEVC video track configuration.
    ///
    /// # Arguments
    ///
    /// * `width` - Video width in pixels.
    /// * `height` - Video height in pixels.
    ///
    /// Note: The mp4 crate v0.14 does not store VPS/SPS/PPS for HEVC.
    pub fn hevc(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            codec: VideoCodecConfig::H265,
        }
    }

    /// Create VP9 video track configuration.
    ///
    /// # Arguments
    ///
    /// * `width` - Video width in pixels.
    /// * `height` - Video height in pixels.
    pub fn vp9(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            codec: VideoCodecConfig::Vp9,
        }
    }
}

// ============================================================================
// Audio Track Configuration
// ============================================================================

/// Audio track configuration for the MP4 muxer.
#[derive(Debug, Clone)]
pub struct Mp4AudioTrackConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Audio codec configuration.
    pub codec: AudioCodecConfig,
}

/// Audio codec configuration.
#[derive(Debug, Clone)]
pub enum AudioCodecConfig {
    /// AAC configuration.
    Aac {
        /// AAC profile (1 = AAC-LC, 2 = HE-AAC, etc.).
        profile: u8,
    },
}

impl Mp4AudioTrackConfig {
    /// Create AAC audio track configuration.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (e.g., 44100, 48000).
    /// * `channels` - Number of channels (1 = mono, 2 = stereo).
    pub fn aac(sample_rate: u32, channels: u8) -> Self {
        Self {
            sample_rate,
            channels,
            codec: AudioCodecConfig::Aac {
                profile: 2, // AAC-LC
            },
        }
    }

    /// Create AAC audio track configuration with custom profile.
    pub fn aac_with_profile(sample_rate: u32, channels: u8, profile: u8) -> Self {
        Self {
            sample_rate,
            channels,
            codec: AudioCodecConfig::Aac { profile },
        }
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for the MP4 muxer.
#[derive(Debug, Clone, Default)]
pub struct Mp4MuxStats {
    /// Total samples written.
    pub samples_written: u64,
    /// Total bytes written.
    pub bytes_written: u64,
    /// Video samples written.
    pub video_samples: u64,
    /// Audio samples written.
    pub audio_samples: u64,
    /// Keyframes written.
    pub keyframes: u64,
}

// ============================================================================
// MP4 Muxer
// ============================================================================

/// MP4/MOV container muxer.
///
/// Creates MP4 files from elementary streams using the `mp4` crate (pure Rust).
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
/// use std::fs::File;
/// use std::io::BufWriter;
///
/// let file = File::create("output.mp4")?;
/// let writer = BufWriter::new(file);
///
/// let mut mux = Mp4Mux::new(writer, Mp4MuxConfig::default())?;
///
/// // Add H.264 video track (SPS/PPS from stream)
/// let video_track = mux.add_video_track(Mp4VideoTrackConfig::h264(
///     1920, 1080, &sps_bytes, &pps_bytes
/// ))?;
///
/// // Write video frames
/// for frame in frames {
///     mux.write_video_sample(video_track, &frame.data, frame.pts_ms, frame.is_keyframe)?;
/// }
///
/// // Finalize the file
/// mux.finish()?;
/// ```
pub struct Mp4Mux<W: Write + Seek> {
    writer: Mp4Writer<W>,
    stats: Mp4MuxStats,
    #[allow(dead_code)]
    timescale: u32,
    track_count: u32,
}

impl<W: Write + Seek> Mp4Mux<W> {
    /// Create a new MP4 muxer.
    ///
    /// # Arguments
    ///
    /// * `writer` - A writer implementing `Write + Seek` (e.g., `BufWriter<File>`).
    /// * `config` - Muxer configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the MP4 header cannot be written.
    pub fn new(writer: W, config: Mp4MuxConfig) -> Result<Self> {
        let mp4_config = Mp4Config {
            major_brand: config
                .major_brand
                .parse()
                .map_err(|_| Error::Config("Invalid major brand".into()))?,
            minor_version: config.minor_version,
            compatible_brands: config
                .compatible_brands
                .iter()
                .filter_map(|s| s.parse().ok())
                .collect(),
            timescale: config.timescale,
        };

        let mp4_writer = Mp4Writer::write_start(writer, &mp4_config)
            .map_err(|e| Error::Config(format!("Failed to write MP4 header: {}", e)))?;

        Ok(Self {
            writer: mp4_writer,
            stats: Mp4MuxStats::default(),
            timescale: config.timescale,
            track_count: 0,
        })
    }

    /// Add a video track to the MP4 file.
    ///
    /// Returns the track ID (1-based).
    pub fn add_video_track(&mut self, config: Mp4VideoTrackConfig) -> Result<u32> {
        let media_config = match config.codec {
            VideoCodecConfig::H264 { sps, pps } => MediaConfig::AvcConfig(AvcConfig {
                width: config.width,
                height: config.height,
                seq_param_set: sps,
                pic_param_set: pps,
            }),
            VideoCodecConfig::H265 => MediaConfig::HevcConfig(HevcConfig {
                width: config.width,
                height: config.height,
            }),
            VideoCodecConfig::Vp9 => MediaConfig::Vp9Config(Vp9Config {
                width: config.width,
                height: config.height,
            }),
        };

        let track_config = TrackConfig::from(media_config);

        self.writer
            .add_track(&track_config)
            .map_err(|e| Error::Config(format!("Failed to add video track: {}", e)))?;

        self.track_count += 1;
        Ok(self.track_count)
    }

    /// Add an audio track to the MP4 file.
    ///
    /// Returns the track ID (1-based).
    pub fn add_audio_track(&mut self, config: Mp4AudioTrackConfig) -> Result<u32> {
        let media_config = match config.codec {
            AudioCodecConfig::Aac { profile } => {
                let freq_index = Self::sample_rate_to_index(config.sample_rate);
                let chan_conf = Self::channels_to_config(config.channels);
                let audio_profile = Self::profile_to_audio_object_type(profile);

                MediaConfig::AacConfig(AacConfig {
                    bitrate: 0, // Unknown/VBR
                    profile: audio_profile,
                    freq_index,
                    chan_conf,
                })
            }
        };

        let track_config = TrackConfig::from(media_config);

        self.writer
            .add_track(&track_config)
            .map_err(|e| Error::Config(format!("Failed to add audio track: {}", e)))?;

        self.track_count += 1;
        Ok(self.track_count)
    }

    /// Convert sample rate to AAC frequency index.
    fn sample_rate_to_index(sample_rate: u32) -> SampleFreqIndex {
        match sample_rate {
            96000 => SampleFreqIndex::Freq96000,
            88200 => SampleFreqIndex::Freq88200,
            64000 => SampleFreqIndex::Freq64000,
            48000 => SampleFreqIndex::Freq48000,
            44100 => SampleFreqIndex::Freq44100,
            32000 => SampleFreqIndex::Freq32000,
            24000 => SampleFreqIndex::Freq24000,
            22050 => SampleFreqIndex::Freq22050,
            16000 => SampleFreqIndex::Freq16000,
            12000 => SampleFreqIndex::Freq12000,
            11025 => SampleFreqIndex::Freq11025,
            8000 => SampleFreqIndex::Freq8000,
            7350 => SampleFreqIndex::Freq7350,
            _ => SampleFreqIndex::Freq44100, // Default to 44100 Hz
        }
    }

    /// Convert channel count to ChannelConfig.
    fn channels_to_config(channels: u8) -> ChannelConfig {
        match channels {
            1 => ChannelConfig::Mono,
            2 => ChannelConfig::Stereo,
            3 => ChannelConfig::Three,
            4 => ChannelConfig::Four,
            5 => ChannelConfig::Five,
            6 => ChannelConfig::FiveOne,
            8 => ChannelConfig::SevenOne,
            _ => ChannelConfig::Stereo, // Default to stereo
        }
    }

    /// Convert profile number to AudioObjectType.
    fn profile_to_audio_object_type(profile: u8) -> AudioObjectType {
        match profile {
            1 => AudioObjectType::AacMain,
            2 => AudioObjectType::AacLowComplexity,
            3 => AudioObjectType::AacScalableSampleRate,
            4 => AudioObjectType::AacLongTermPrediction,
            5 => AudioObjectType::SpectralBandReplication,
            _ => AudioObjectType::AacLowComplexity, // Default to AAC-LC
        }
    }

    /// Write a sample to a track.
    ///
    /// # Arguments
    ///
    /// * `track_id` - The track ID (returned from `add_video_track` or `add_audio_track`).
    /// * `data` - The sample data (encoded frame).
    /// * `pts_ms` - Presentation timestamp in milliseconds.
    /// * `dts_ms` - Decode timestamp in milliseconds (use same as PTS if no B-frames).
    /// * `is_sync` - Whether this is a sync point (keyframe for video, always true for audio).
    pub fn write_sample(
        &mut self,
        track_id: u32,
        data: &[u8],
        pts_ms: u64,
        dts_ms: u64,
        is_sync: bool,
    ) -> Result<()> {
        let sample = Mp4Sample {
            start_time: dts_ms,
            duration: 0, // Will be calculated from next sample
            rendering_offset: (pts_ms as i64 - dts_ms as i64) as i32,
            is_sync,
            bytes: bytes::Bytes::copy_from_slice(data),
        };

        self.writer
            .write_sample(track_id, &sample)
            .map_err(|e| Error::Config(format!("Failed to write sample: {}", e)))?;

        self.stats.samples_written += 1;
        self.stats.bytes_written += data.len() as u64;

        if is_sync {
            self.stats.keyframes += 1;
        }

        Ok(())
    }

    /// Write a video sample with automatic timestamp handling.
    ///
    /// # Arguments
    ///
    /// * `track_id` - The video track ID.
    /// * `data` - The encoded video frame data.
    /// * `pts_ms` - Presentation timestamp in milliseconds.
    /// * `is_keyframe` - Whether this is a keyframe (IDR frame).
    pub fn write_video_sample(
        &mut self,
        track_id: u32,
        data: &[u8],
        pts_ms: u64,
        is_keyframe: bool,
    ) -> Result<()> {
        self.write_sample(track_id, data, pts_ms, pts_ms, is_keyframe)?;
        self.stats.video_samples += 1;
        Ok(())
    }

    /// Write an audio sample.
    ///
    /// Audio samples are always sync points.
    ///
    /// # Arguments
    ///
    /// * `track_id` - The audio track ID.
    /// * `data` - The encoded audio frame data.
    /// * `pts_ms` - Presentation timestamp in milliseconds.
    pub fn write_audio_sample(&mut self, track_id: u32, data: &[u8], pts_ms: u64) -> Result<()> {
        self.write_sample(track_id, data, pts_ms, pts_ms, true)?;
        self.stats.audio_samples += 1;
        Ok(())
    }

    /// Get muxer statistics.
    pub fn stats(&self) -> &Mp4MuxStats {
        &self.stats
    }

    /// Finalize the MP4 file.
    ///
    /// This writes the moov box and completes the file.
    /// The muxer cannot be used after calling this method.
    pub fn finish(mut self) -> Result<W> {
        self.writer
            .write_end()
            .map_err(|e| Error::Config(format!("Failed to finalize MP4: {}", e)))?;

        Ok(self.writer.into_writer())
    }
}

impl<W: Write + Seek> std::fmt::Debug for Mp4Mux<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp4Mux")
            .field("stats", &self.stats)
            .field("track_count", &self.track_count)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_mp4_mux_config_default() {
        let config = Mp4MuxConfig::default();
        assert_eq!(config.major_brand, "isom");
        assert_eq!(config.timescale, 1000);
    }

    #[test]
    fn test_mp4_mux_create() {
        let buffer = Cursor::new(Vec::new());
        let mux = Mp4Mux::new(buffer, Mp4MuxConfig::default());
        assert!(mux.is_ok());
    }

    #[test]
    fn test_sample_rate_to_index() {
        assert_eq!(
            Mp4Mux::<Cursor<Vec<u8>>>::sample_rate_to_index(44100),
            SampleFreqIndex::Freq44100
        );
        assert_eq!(
            Mp4Mux::<Cursor<Vec<u8>>>::sample_rate_to_index(48000),
            SampleFreqIndex::Freq48000
        );
        assert_eq!(
            Mp4Mux::<Cursor<Vec<u8>>>::sample_rate_to_index(96000),
            SampleFreqIndex::Freq96000
        );
    }

    #[test]
    fn test_video_track_config_h264() {
        let config = Mp4VideoTrackConfig::h264(1920, 1080, &[0, 1, 2], &[3, 4, 5]);
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        match config.codec {
            VideoCodecConfig::H264 { sps, pps } => {
                assert_eq!(sps, vec![0, 1, 2]);
                assert_eq!(pps, vec![3, 4, 5]);
            }
            _ => panic!("Expected H264 codec"),
        }
    }

    #[test]
    fn test_audio_track_config_aac() {
        let config = Mp4AudioTrackConfig::aac(44100, 2);
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
    }
}
