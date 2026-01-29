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
// MP4 File Sink Element
// ============================================================================

use crate::element::{ConsumeContext, Sink};
use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

/// Configuration for the MP4 file sink.
#[derive(Debug, Clone)]
pub struct Mp4FileSinkConfig {
    /// Video width in pixels.
    pub width: u16,
    /// Video height in pixels.
    pub height: u16,
    /// Frame rate (frames per second).
    pub framerate: f32,
    /// H.264 SPS data (extracted from first keyframe if not provided).
    pub sps: Option<Vec<u8>>,
    /// H.264 PPS data (extracted from first keyframe if not provided).
    pub pps: Option<Vec<u8>>,
}

impl Mp4FileSinkConfig {
    /// Create a new MP4 file sink configuration.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            framerate: 30.0,
            sps: None,
            pps: None,
        }
    }

    /// Set the frame rate.
    pub fn with_framerate(mut self, fps: f32) -> Self {
        self.framerate = fps;
        self
    }

    /// Set the H.264 SPS/PPS data.
    pub fn with_codec_data(mut self, sps: Vec<u8>, pps: Vec<u8>) -> Self {
        self.sps = Some(sps);
        self.pps = Some(pps);
        self
    }
}

/// MP4 file sink element for H.264 video.
///
/// This element writes H.264 encoded video frames to an MP4 file.
/// It automatically extracts SPS/PPS from the first keyframe and
/// creates a properly formatted MP4 file.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::mux::{Mp4FileSink, Mp4FileSinkConfig};
///
/// let config = Mp4FileSinkConfig::new(1920, 1080).with_framerate(30.0);
/// let sink = Mp4FileSink::new("output.mp4", config)?;
///
/// // Use in pipeline
/// pipeline.add_sink("mp4sink", sink);
/// ```
pub struct Mp4FileSink {
    path: PathBuf,
    config: Mp4FileSinkConfig,
    muxer: Option<Mp4Mux<BufWriter<File>>>,
    video_track: Option<u32>,
    frame_count: u64,
    frame_duration_ms: u64,
    /// Extracted SPS from first keyframe
    sps: Option<Vec<u8>>,
    /// Extracted PPS from first keyframe
    pps: Option<Vec<u8>>,
}

impl Mp4FileSink {
    /// Create a new MP4 file sink.
    pub fn new<P: AsRef<Path>>(path: P, config: Mp4FileSinkConfig) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        let frame_duration_ms = (1000.0 / config.framerate) as u64;

        Ok(Self {
            path,
            config,
            muxer: None,
            video_track: None,
            frame_count: 0,
            frame_duration_ms,
            sps: None,
            pps: None,
        })
    }

    /// Extract SPS and PPS from H.264 NAL units.
    ///
    /// Scans the data for NAL units and extracts SPS (type 7) and PPS (type 8).
    pub fn extract_sps_pps(data: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
        let mut sps = None;
        let mut pps = None;
        let mut i = 0;

        while i + 4 < data.len() {
            // Look for start code (0x00 0x00 0x00 0x01 or 0x00 0x00 0x01)
            let start_code_len = if i + 4 <= data.len()
                && data[i] == 0
                && data[i + 1] == 0
                && data[i + 2] == 0
                && data[i + 3] == 1
            {
                4
            } else if i + 3 <= data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                3
            } else {
                i += 1;
                continue;
            };

            let nal_start = i + start_code_len;
            if nal_start >= data.len() {
                break;
            }

            let nal_type = data[nal_start] & 0x1F;

            // Find end of this NAL unit (next start code or end of data)
            let mut nal_end = data.len();
            for j in (nal_start + 1)..(data.len().saturating_sub(2)) {
                if (data[j] == 0 && data[j + 1] == 0 && data[j + 2] == 1)
                    || (j + 3 < data.len()
                        && data[j] == 0
                        && data[j + 1] == 0
                        && data[j + 2] == 0
                        && data[j + 3] == 1)
                {
                    nal_end = j;
                    break;
                }
            }

            let nal_data = &data[nal_start..nal_end];

            match nal_type {
                7 => {
                    // SPS
                    sps = Some(nal_data.to_vec());
                }
                8 => {
                    // PPS
                    pps = Some(nal_data.to_vec());
                }
                _ => {}
            }

            i = nal_end;
        }

        (sps, pps)
    }

    /// Convert Annex-B format to AVCC format.
    ///
    /// Replaces start codes (0x00 0x00 0x00 0x01) with 4-byte length prefixes.
    pub fn annex_b_to_avcc(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len());
        let mut i = 0;

        while i < data.len() {
            // Find start code
            let start_code_len = if i + 4 <= data.len()
                && data[i] == 0
                && data[i + 1] == 0
                && data[i + 2] == 0
                && data[i + 3] == 1
            {
                4
            } else if i + 3 <= data.len() && data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
                3
            } else {
                // No start code at this position, just copy byte
                result.push(data[i]);
                i += 1;
                continue;
            };

            let nal_start = i + start_code_len;
            if nal_start >= data.len() {
                break;
            }

            // Find end of this NAL unit
            let mut nal_end = data.len();
            for j in (nal_start + 1)..(data.len().saturating_sub(2)) {
                if (data[j] == 0 && data[j + 1] == 0 && data[j + 2] == 1)
                    || (j + 3 < data.len()
                        && data[j] == 0
                        && data[j + 1] == 0
                        && data[j + 2] == 0
                        && data[j + 3] == 1)
                {
                    nal_end = j;
                    break;
                }
            }

            let nal_data = &data[nal_start..nal_end];
            let nal_len = nal_data.len() as u32;

            // Write 4-byte length prefix (big-endian)
            result.extend_from_slice(&nal_len.to_be_bytes());
            result.extend_from_slice(nal_data);

            i = nal_end;
        }

        result
    }

    /// Initialize the muxer with the given SPS/PPS.
    fn initialize_muxer(&mut self, sps: &[u8], pps: &[u8]) -> Result<()> {
        let file = File::create(&self.path)?;
        let writer = BufWriter::new(file);

        let mut muxer = Mp4Mux::new(writer, Mp4MuxConfig::h264())?;

        let video_config =
            Mp4VideoTrackConfig::h264(self.config.width, self.config.height, sps, pps);
        let track_id = muxer.add_video_track(video_config)?;

        self.muxer = Some(muxer);
        self.video_track = Some(track_id);

        tracing::info!(
            "MP4 muxer initialized: {}x{} @ {} fps",
            self.config.width,
            self.config.height,
            self.config.framerate
        );

        Ok(())
    }

    /// Get the number of frames written.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Finalize and close the MP4 file.
    pub fn finish(mut self) -> Result<()> {
        if let Some(muxer) = self.muxer.take() {
            muxer.finish()?;
            tracing::info!(
                "MP4 file finalized: {} frames written to {:?}",
                self.frame_count,
                self.path
            );
        }
        Ok(())
    }
}

impl Sink for Mp4FileSink {
    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Accept H.264 encoded video
        use crate::format::{
            ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, VideoCodec,
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Video(VideoCodec::H264),
            MemoryCaps::cpu_only(),
        )])
    }

    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let data = ctx.buffer().as_bytes();

        // Check if this is a keyframe (contains SPS/PPS or IDR NAL)
        let is_keyframe = data.len() > 4 && {
            // Look for IDR NAL unit (type 5) or SPS (type 7)
            let mut found_idr = false;
            let mut i = 0;
            while i + 4 < data.len() {
                if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
                    let nal_type = data[i + 4] & 0x1F;
                    if nal_type == 5 || nal_type == 7 {
                        found_idr = true;
                        break;
                    }
                    i += 4;
                } else {
                    i += 1;
                }
            }
            found_idr
        };

        // If muxer not initialized, try to extract SPS/PPS from this frame
        if self.muxer.is_none() {
            // Try config first
            let (sps, pps) = if self.config.sps.is_some() && self.config.pps.is_some() {
                (
                    self.config.sps.clone().unwrap(),
                    self.config.pps.clone().unwrap(),
                )
            } else {
                // Extract from frame data
                let (extracted_sps, extracted_pps) = Self::extract_sps_pps(data);
                match (extracted_sps, extracted_pps) {
                    (Some(s), Some(p)) => (s, p),
                    _ => {
                        // Can't initialize yet, skip this frame
                        tracing::debug!("Waiting for keyframe with SPS/PPS...");
                        return Ok(());
                    }
                }
            };

            self.sps = Some(sps.clone());
            self.pps = Some(pps.clone());
            self.initialize_muxer(&sps, &pps)?;
        }

        let muxer = self.muxer.as_mut().unwrap();
        let track_id = self.video_track.unwrap();

        // Convert from Annex-B to AVCC format
        let avcc_data = Self::annex_b_to_avcc(data);

        // Calculate PTS
        let pts_ms = self.frame_count * self.frame_duration_ms;

        // Write sample
        muxer.write_video_sample(track_id, &avcc_data, pts_ms, is_keyframe)?;
        self.frame_count += 1;

        if self.frame_count % 30 == 0 {
            tracing::debug!("MP4: {} frames written", self.frame_count);
        }

        Ok(())
    }
}

impl Drop for Mp4FileSink {
    fn drop(&mut self) {
        if let Some(muxer) = self.muxer.take() {
            if let Err(e) = muxer.finish() {
                tracing::error!("Failed to finalize MP4 file: {}", e);
            }
        }
    }
}

impl std::fmt::Debug for Mp4FileSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp4FileSink")
            .field("path", &self.path)
            .field("config", &self.config)
            .field("frame_count", &self.frame_count)
            .field("initialized", &self.muxer.is_some())
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// MP4 Mux Transform Element
// ============================================================================

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Element;
use crate::memory::SharedArena;
use crate::metadata::Metadata;
use std::io::Cursor;

/// Configuration for the MP4 mux transform.
#[derive(Debug, Clone)]
pub struct Mp4MuxTransformConfig {
    /// Video width in pixels.
    pub width: u16,
    /// Video height in pixels.
    pub height: u16,
    /// Frame rate (frames per second).
    pub framerate: f32,
    /// H.264 SPS data (extracted from first keyframe if not provided).
    pub sps: Option<Vec<u8>>,
    /// H.264 PPS data (extracted from first keyframe if not provided).
    pub pps: Option<Vec<u8>>,
}

impl Mp4MuxTransformConfig {
    /// Create a new MP4 mux transform configuration.
    pub fn new(width: u16, height: u16) -> Self {
        Self {
            width,
            height,
            framerate: 30.0,
            sps: None,
            pps: None,
        }
    }

    /// Set the frame rate.
    pub fn with_framerate(mut self, fps: f32) -> Self {
        self.framerate = fps;
        self
    }

    /// Set the H.264 SPS/PPS data.
    pub fn with_codec_data(mut self, sps: Vec<u8>, pps: Vec<u8>) -> Self {
        self.sps = Some(sps);
        self.pps = Some(pps);
        self
    }
}

/// MP4 mux transform element for H.264 video.
///
/// This element muxes H.264 encoded video frames into MP4 container format.
/// Unlike `Mp4FileSink`, this element does NOT write to disk - it outputs
/// the MP4 data as buffers that can be sent to any sink (file, network, etc.).
///
/// **Important**: MP4 format requires seeking, so this element buffers all
/// input frames in memory and outputs the complete MP4 file on `flush()`.
/// For streaming use cases, consider using fragmented MP4 (fMP4) or MPEG-TS.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::mux::{Mp4MuxTransform, Mp4MuxTransformConfig};
/// use parallax::elements::io::FileSink;
///
/// let config = Mp4MuxTransformConfig::new(1920, 1080).with_framerate(30.0);
/// let mux = Mp4MuxTransform::new(config);
/// let sink = FileSink::new("output.mp4");
///
/// // Pipeline: ... -> H264Encoder -> Mp4MuxTransform -> FileSink
/// ```
pub struct Mp4MuxTransform {
    config: Mp4MuxTransformConfig,
    /// In-memory buffer for MP4 data
    buffer: Cursor<Vec<u8>>,
    /// The muxer (initialized when SPS/PPS is available)
    muxer: Option<Mp4Mux<Cursor<Vec<u8>>>>,
    video_track: Option<u32>,
    frame_count: u64,
    frame_duration_ms: u64,
    /// Extracted SPS from first keyframe
    sps: Option<Vec<u8>>,
    /// Extracted PPS from first keyframe
    pps: Option<Vec<u8>>,
    /// Whether we've already flushed
    flushed: bool,
}

impl Mp4MuxTransform {
    /// Create a new MP4 mux transform.
    pub fn new(config: Mp4MuxTransformConfig) -> Self {
        let frame_duration_ms = (1000.0 / config.framerate) as u64;

        Self {
            config,
            buffer: Cursor::new(Vec::new()),
            muxer: None,
            video_track: None,
            frame_count: 0,
            frame_duration_ms,
            sps: None,
            pps: None,
            flushed: false,
        }
    }

    /// Initialize the muxer with the given SPS/PPS.
    fn initialize_muxer(&mut self, sps: &[u8], pps: &[u8]) -> Result<()> {
        // Take the current buffer and create a fresh one
        let buffer = std::mem::replace(&mut self.buffer, Cursor::new(Vec::new()));
        self.buffer = buffer;

        let mut muxer = Mp4Mux::new(
            Cursor::new(Vec::with_capacity(1024 * 1024)), // Pre-allocate 1MB
            Mp4MuxConfig::h264(),
        )?;

        let video_config =
            Mp4VideoTrackConfig::h264(self.config.width, self.config.height, sps, pps);
        let track_id = muxer.add_video_track(video_config)?;

        self.muxer = Some(muxer);
        self.video_track = Some(track_id);

        tracing::info!(
            "MP4 mux transform initialized: {}x{} @ {} fps",
            self.config.width,
            self.config.height,
            self.config.framerate
        );

        Ok(())
    }

    /// Get the number of frames muxed.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Check if a NAL unit is a keyframe (contains SPS, PPS, or IDR).
    fn is_keyframe(data: &[u8]) -> bool {
        if data.len() <= 4 {
            return false;
        }

        let mut i = 0;
        while i + 4 < data.len() {
            if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1 {
                let nal_type = data[i + 4] & 0x1F;
                if nal_type == 5 || nal_type == 7 {
                    // IDR or SPS
                    return true;
                }
                i += 4;
            } else {
                i += 1;
            }
        }
        false
    }
}

impl Element for Mp4MuxTransform {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let data = buffer.as_bytes();

        let is_keyframe = Self::is_keyframe(data);

        // If muxer not initialized, try to extract SPS/PPS from this frame
        if self.muxer.is_none() {
            // Try config first
            let (sps, pps) = if self.config.sps.is_some() && self.config.pps.is_some() {
                (
                    self.config.sps.clone().unwrap(),
                    self.config.pps.clone().unwrap(),
                )
            } else {
                // Extract from frame data
                let (extracted_sps, extracted_pps) = Mp4FileSink::extract_sps_pps(data);
                match (extracted_sps, extracted_pps) {
                    (Some(s), Some(p)) => (s, p),
                    _ => {
                        // Can't initialize yet, skip this frame
                        tracing::debug!("MP4 mux: waiting for keyframe with SPS/PPS...");
                        return Ok(None);
                    }
                }
            };

            self.sps = Some(sps.clone());
            self.pps = Some(pps.clone());
            self.initialize_muxer(&sps, &pps)?;
        }

        let muxer = self.muxer.as_mut().unwrap();
        let track_id = self.video_track.unwrap();

        // Convert from Annex-B to AVCC format
        let avcc_data = Mp4FileSink::annex_b_to_avcc(data);

        // Calculate PTS
        let pts_ms = self.frame_count * self.frame_duration_ms;

        // Write sample
        muxer.write_video_sample(track_id, &avcc_data, pts_ms, is_keyframe)?;
        self.frame_count += 1;

        if self.frame_count % 30 == 0 {
            tracing::debug!("MP4 mux: {} frames buffered", self.frame_count);
        }

        // Don't output anything during processing - MP4 requires all data for moov
        Ok(None)
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        if self.flushed {
            return Ok(None);
        }
        self.flushed = true;

        if let Some(muxer) = self.muxer.take() {
            // Finalize the muxer and get the complete MP4 data
            let cursor = muxer.finish()?;
            let mp4_data = cursor.into_inner();

            tracing::info!(
                "MP4 mux: finalized {} frames, {} bytes",
                self.frame_count,
                mp4_data.len()
            );

            if mp4_data.is_empty() {
                return Ok(None);
            }

            // Create an arena for the output buffer
            let arena = SharedArena::new(mp4_data.len(), 1)
                .map_err(|e| Error::AllocationFailed(format!("Failed to create arena: {}", e)))?;

            let mut slot = arena.acquire().ok_or(Error::PoolExhausted)?;

            slot.data_mut()[..mp4_data.len()].copy_from_slice(&mp4_data);

            let handle = MemoryHandle::with_len(slot, mp4_data.len());
            let buffer = Buffer::new(handle, Metadata::new());

            Ok(Some(buffer))
        } else {
            tracing::warn!("MP4 mux: no muxer initialized, no output");
            Ok(None)
        }
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Accept H.264 encoded video
        use crate::format::{
            ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, VideoCodec,
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Video(VideoCodec::H264),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Output MP4 as raw bytes (container format)
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Bytes,
            MemoryCaps::cpu_only(),
        )])
    }
}

impl std::fmt::Debug for Mp4MuxTransform {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp4MuxTransform")
            .field("config", &self.config)
            .field("frame_count", &self.frame_count)
            .field("initialized", &self.muxer.is_some())
            .field("flushed", &self.flushed)
            .finish()
    }
}

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

    #[test]
    fn test_mp4_mux_transform_config() {
        let config = Mp4MuxTransformConfig::new(1920, 1080)
            .with_framerate(60.0)
            .with_codec_data(vec![1, 2, 3], vec![4, 5, 6]);

        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.framerate, 60.0);
        assert_eq!(config.sps, Some(vec![1, 2, 3]));
        assert_eq!(config.pps, Some(vec![4, 5, 6]));
    }
}
