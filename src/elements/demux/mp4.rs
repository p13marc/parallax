//! MP4/MOV container demuxer.
//!
//! This module provides an MP4 demuxer that extracts elementary streams
//! (video, audio) from MP4/MOV container files.
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
//! use parallax::elements::Mp4Demux;
//! use std::fs::File;
//! use std::io::BufReader;
//!
//! let file = File::open("video.mp4")?;
//! let size = file.metadata()?.len();
//! let reader = BufReader::new(file);
//! let mut demux = Mp4Demux::new(reader, size)?;
//!
//! // Get track information
//! for track in demux.tracks() {
//!     println!("Track {}: {:?}", track.id, track.codec);
//! }
//!
//! // Read samples
//! while let Some(sample) = demux.read_sample(track_id)? {
//!     // Process sample.data, sample.pts, sample.dts, etc.
//! }
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use crate::metadata::{BufferFlags, Metadata};

use mp4::{MediaType, Mp4Reader, TrackType};
use std::io::{Read, Seek};
use std::sync::OnceLock;

/// Shared arena for MP4 demuxer buffers.
fn mp4_demux_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    // Video frames can be large, use generous slot size
    ARENA.get_or_init(|| SharedArena::new(4 * 1024 * 1024, 32).unwrap())
}

// ============================================================================
// Codec Types
// ============================================================================

/// MP4 codec type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Mp4Codec {
    /// H.264/AVC video.
    H264,
    /// H.265/HEVC video.
    H265,
    /// VP9 video.
    Vp9,
    /// AAC audio.
    Aac,
    /// TTML/TTXT subtitles.
    Ttxt,
    /// Unknown codec.
    Unknown,
}

impl Mp4Codec {
    /// Returns true if this is a video codec.
    pub fn is_video(&self) -> bool {
        matches!(self, Mp4Codec::H264 | Mp4Codec::H265 | Mp4Codec::Vp9)
    }

    /// Returns true if this is an audio codec.
    pub fn is_audio(&self) -> bool {
        matches!(self, Mp4Codec::Aac)
    }

    /// Returns true if this is a subtitle codec.
    pub fn is_subtitle(&self) -> bool {
        matches!(self, Mp4Codec::Ttxt)
    }
}

impl std::fmt::Display for Mp4Codec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Mp4Codec::H264 => write!(f, "H.264/AVC"),
            Mp4Codec::H265 => write!(f, "H.265/HEVC"),
            Mp4Codec::Vp9 => write!(f, "VP9"),
            Mp4Codec::Aac => write!(f, "AAC"),
            Mp4Codec::Ttxt => write!(f, "TTXT"),
            Mp4Codec::Unknown => write!(f, "Unknown"),
        }
    }
}

impl From<MediaType> for Mp4Codec {
    fn from(mt: MediaType) -> Self {
        match mt {
            MediaType::H264 => Mp4Codec::H264,
            MediaType::H265 => Mp4Codec::H265,
            MediaType::VP9 => Mp4Codec::Vp9,
            MediaType::AAC => Mp4Codec::Aac,
            MediaType::TTXT => Mp4Codec::Ttxt,
        }
    }
}

// ============================================================================
// Track Information
// ============================================================================

/// Information about a track in the MP4 file.
#[derive(Debug, Clone)]
pub struct Mp4Track {
    /// Track ID (1-based).
    pub id: u32,
    /// Track type (video, audio, subtitle).
    pub track_type: Mp4TrackType,
    /// Codec used by this track.
    pub codec: Mp4Codec,
    /// Duration in nanoseconds.
    pub duration_ns: u64,
    /// Timescale (ticks per second).
    pub timescale: u32,
    /// Number of samples in this track.
    pub sample_count: u32,
    /// Video-specific information (if applicable).
    pub video_info: Option<Mp4VideoInfo>,
    /// Audio-specific information (if applicable).
    pub audio_info: Option<Mp4AudioInfo>,
}

/// Track type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mp4TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle track.
    Subtitle,
    /// Unknown track type.
    Unknown,
}

impl From<TrackType> for Mp4TrackType {
    fn from(tt: TrackType) -> Self {
        match tt {
            TrackType::Video => Mp4TrackType::Video,
            TrackType::Audio => Mp4TrackType::Audio,
            TrackType::Subtitle => Mp4TrackType::Subtitle,
        }
    }
}

/// Video track information.
#[derive(Debug, Clone)]
pub struct Mp4VideoInfo {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Frame rate (frames per second), if available.
    pub frame_rate: Option<f64>,
}

/// Audio track information.
#[derive(Debug, Clone)]
pub struct Mp4AudioInfo {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
}

// ============================================================================
// Sample
// ============================================================================

/// A sample (frame) extracted from an MP4 track.
#[derive(Debug)]
pub struct Mp4Sample {
    /// The buffer containing the sample data.
    pub buffer: Buffer,
    /// Track ID this sample belongs to.
    pub track_id: u32,
    /// Presentation timestamp in nanoseconds.
    pub pts_ns: u64,
    /// Decode timestamp in nanoseconds (may differ from PTS for B-frames).
    pub dts_ns: u64,
    /// Duration of this sample in nanoseconds.
    pub duration_ns: u64,
    /// Whether this is a keyframe (sync sample).
    pub is_keyframe: bool,
    /// Sample index within the track.
    pub sample_index: u32,
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for the MP4 demuxer.
#[derive(Debug, Clone, Default)]
pub struct Mp4DemuxStats {
    /// Total samples read.
    pub samples_read: u64,
    /// Total bytes read.
    pub bytes_read: u64,
    /// Video samples read.
    pub video_samples: u64,
    /// Audio samples read.
    pub audio_samples: u64,
    /// Keyframes read.
    pub keyframes: u64,
}

// ============================================================================
// MP4 Demuxer
// ============================================================================

/// MP4/MOV container demuxer.
///
/// Extracts elementary streams from MP4 containers using the `mp4` crate (pure Rust).
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Mp4Demux;
/// use std::fs::File;
/// use std::io::BufReader;
///
/// let file = File::open("video.mp4")?;
/// let size = file.metadata()?.len();
/// let reader = BufReader::new(file);
///
/// let mut demux = Mp4Demux::new(reader, size)?;
///
/// // Iterate through tracks
/// for track in demux.tracks() {
///     println!("Track {}: {} ({:?})", track.id, track.codec, track.track_type);
/// }
///
/// // Read video samples
/// if let Some(video_track) = demux.video_track_id() {
///     while let Some(sample) = demux.read_sample(video_track)? {
///         println!("Frame: pts={}ns, keyframe={}", sample.pts_ns, sample.is_keyframe);
///     }
/// }
/// ```
pub struct Mp4Demux<R: Read + Seek> {
    reader: Mp4Reader<R>,
    tracks: Vec<Mp4Track>,
    stats: Mp4DemuxStats,
    /// Current sample index per track (track_id -> next sample index).
    sample_indices: std::collections::HashMap<u32, u32>,
}

impl<R: Read + Seek> Mp4Demux<R> {
    /// Create a new MP4 demuxer from a reader.
    ///
    /// # Arguments
    ///
    /// * `reader` - A reader implementing `Read + Seek` (e.g., `BufReader<File>`).
    /// * `size` - Total size of the MP4 data in bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the MP4 header cannot be parsed.
    pub fn new(reader: R, size: u64) -> Result<Self> {
        let mp4_reader = Mp4Reader::read_header(reader, size)
            .map_err(|e| Error::Config(format!("Failed to read MP4 header: {}", e)))?;

        let mut tracks = Vec::new();
        let mut sample_indices = std::collections::HashMap::new();

        for track in mp4_reader.tracks().values() {
            let track_type = track
                .track_type()
                .map(Mp4TrackType::from)
                .unwrap_or(Mp4TrackType::Unknown);

            let codec = track
                .media_type()
                .map(Mp4Codec::from)
                .unwrap_or(Mp4Codec::Unknown);

            let video_info = if track_type == Mp4TrackType::Video {
                Some(Mp4VideoInfo {
                    width: track.width() as u32,
                    height: track.height() as u32,
                    frame_rate: Some(track.frame_rate()),
                })
            } else {
                None
            };

            let audio_info = if track_type == Mp4TrackType::Audio {
                Some(Mp4AudioInfo {
                    sample_rate: track
                        .sample_freq_index()
                        .map(|i| Self::sample_rate_from_index(i))
                        .unwrap_or(44100),
                    channels: track
                        .channel_config()
                        .map(|c| Self::channel_count(c))
                        .unwrap_or(2),
                })
            } else {
                None
            };

            // Duration in nanoseconds
            let duration = track.duration();
            let duration_ns = duration.as_nanos() as u64;

            tracks.push(Mp4Track {
                id: track.track_id(),
                track_type,
                codec,
                duration_ns,
                timescale: track.timescale(),
                sample_count: track.sample_count(),
                video_info,
                audio_info,
            });

            sample_indices.insert(track.track_id(), 1); // Samples are 1-indexed
        }

        Ok(Self {
            reader: mp4_reader,
            tracks,
            stats: Mp4DemuxStats::default(),
            sample_indices,
        })
    }

    /// Convert AAC sample frequency index to Hz.
    fn sample_rate_from_index(index: mp4::SampleFreqIndex) -> u32 {
        match index {
            mp4::SampleFreqIndex::Freq96000 => 96000,
            mp4::SampleFreqIndex::Freq88200 => 88200,
            mp4::SampleFreqIndex::Freq64000 => 64000,
            mp4::SampleFreqIndex::Freq48000 => 48000,
            mp4::SampleFreqIndex::Freq44100 => 44100,
            mp4::SampleFreqIndex::Freq32000 => 32000,
            mp4::SampleFreqIndex::Freq24000 => 24000,
            mp4::SampleFreqIndex::Freq22050 => 22050,
            mp4::SampleFreqIndex::Freq16000 => 16000,
            mp4::SampleFreqIndex::Freq12000 => 12000,
            mp4::SampleFreqIndex::Freq11025 => 11025,
            mp4::SampleFreqIndex::Freq8000 => 8000,
            mp4::SampleFreqIndex::Freq7350 => 7350,
        }
    }

    /// Convert AAC channel config to channel count.
    fn channel_count(config: mp4::ChannelConfig) -> u16 {
        match config {
            mp4::ChannelConfig::Mono => 1,
            mp4::ChannelConfig::Stereo => 2,
            mp4::ChannelConfig::Three => 3,
            mp4::ChannelConfig::Four => 4,
            mp4::ChannelConfig::Five => 5,
            mp4::ChannelConfig::FiveOne => 6,
            mp4::ChannelConfig::SevenOne => 8,
        }
    }

    /// Get all tracks in the MP4 file.
    pub fn tracks(&self) -> &[Mp4Track] {
        &self.tracks
    }

    /// Get the first video track ID, if any.
    pub fn video_track_id(&self) -> Option<u32> {
        self.tracks
            .iter()
            .find(|t| t.track_type == Mp4TrackType::Video)
            .map(|t| t.id)
    }

    /// Get the first audio track ID, if any.
    pub fn audio_track_id(&self) -> Option<u32> {
        self.tracks
            .iter()
            .find(|t| t.track_type == Mp4TrackType::Audio)
            .map(|t| t.id)
    }

    /// Get track information by ID.
    pub fn track(&self, track_id: u32) -> Option<&Mp4Track> {
        self.tracks.iter().find(|t| t.id == track_id)
    }

    /// Get demuxer statistics.
    pub fn stats(&self) -> &Mp4DemuxStats {
        &self.stats
    }

    /// Get total duration of the MP4 file in nanoseconds.
    pub fn duration_ns(&self) -> u64 {
        self.reader.duration().as_nanos() as u64
    }

    /// Reset the read position for a track to the beginning.
    pub fn seek_to_start(&mut self, track_id: u32) {
        self.sample_indices.insert(track_id, 1);
    }

    /// Read the next sample from a track.
    ///
    /// Returns `None` when all samples have been read.
    ///
    /// # Arguments
    ///
    /// * `track_id` - The track ID to read from.
    ///
    /// # Errors
    ///
    /// Returns an error if the track doesn't exist or sample reading fails.
    pub fn read_sample(&mut self, track_id: u32) -> Result<Option<Mp4Sample>> {
        let track = self
            .tracks
            .iter()
            .find(|t| t.id == track_id)
            .ok_or_else(|| Error::Config(format!("Track {} not found", track_id)))?
            .clone();

        let sample_index = *self.sample_indices.get(&track_id).unwrap_or(&1);

        if sample_index > track.sample_count {
            return Ok(None); // No more samples
        }

        // Read the sample
        let sample = self
            .reader
            .read_sample(track_id, sample_index)
            .map_err(|e| Error::Config(format!("Failed to read sample: {}", e)))?;

        let sample = match sample {
            Some(s) => s,
            None => return Ok(None),
        };

        // Create buffer
        let buffer = self.create_buffer(&sample.bytes, track_id, &sample, &track)?;

        // Calculate timestamps in nanoseconds
        let timescale = track.timescale as u128;
        let pts_ns = if timescale > 0 {
            (sample.start_time as u128 * 1_000_000_000 / timescale) as u64
        } else {
            0
        };

        // For DTS, use rendering_offset if available
        let dts_ns = if sample.rendering_offset != 0 && timescale > 0 {
            let dts = sample.start_time as i64 - sample.rendering_offset as i64;
            if dts >= 0 {
                (dts as u128 * 1_000_000_000 / timescale) as u64
            } else {
                0
            }
        } else {
            pts_ns
        };

        let duration_ns = if timescale > 0 {
            (sample.duration as u128 * 1_000_000_000 / timescale) as u64
        } else {
            0
        };

        // Update statistics
        self.stats.samples_read += 1;
        self.stats.bytes_read += sample.bytes.len() as u64;

        if track.track_type == Mp4TrackType::Video {
            self.stats.video_samples += 1;
        } else if track.track_type == Mp4TrackType::Audio {
            self.stats.audio_samples += 1;
        }

        if sample.is_sync {
            self.stats.keyframes += 1;
        }

        // Advance to next sample
        self.sample_indices.insert(track_id, sample_index + 1);

        Ok(Some(Mp4Sample {
            buffer,
            track_id,
            pts_ns,
            dts_ns,
            duration_ns,
            is_keyframe: sample.is_sync,
            sample_index,
        }))
    }

    /// Create a buffer from sample data.
    fn create_buffer(
        &self,
        data: &bytes::Bytes,
        track_id: u32,
        sample: &mp4::Mp4Sample,
        track: &Mp4Track,
    ) -> Result<Buffer> {
        let mut slot = mp4_demux_arena()
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;

        slot.data_mut()[..data.len()].copy_from_slice(data);

        let handle = MemoryHandle::with_len(slot, data.len());

        // Build metadata
        let timescale = track.timescale as u128;

        let mut metadata = Metadata::new();
        metadata.stream_id = track_id;

        if timescale > 0 {
            metadata.pts = ClockTime::from_nanos(
                (sample.start_time as u128 * 1_000_000_000 / timescale) as u64,
            );

            let dts = sample.start_time as i64 - sample.rendering_offset as i64;
            if dts >= 0 {
                metadata.dts =
                    ClockTime::from_nanos((dts as u128 * 1_000_000_000 / timescale) as u64);
            }

            metadata.duration =
                ClockTime::from_nanos((sample.duration as u128 * 1_000_000_000 / timescale) as u64);
        }

        if sample.is_sync {
            metadata.flags |= BufferFlags::SYNC_POINT;
        }

        Ok(Buffer::new(handle, metadata))
    }

    /// Read all samples from a track into a vector.
    ///
    /// This is a convenience method for reading entire tracks at once.
    /// For large files, prefer using `read_sample` in a loop.
    pub fn read_all_samples(&mut self, track_id: u32) -> Result<Vec<Mp4Sample>> {
        self.seek_to_start(track_id);
        let mut samples = Vec::new();

        while let Some(sample) = self.read_sample(track_id)? {
            samples.push(sample);
        }

        Ok(samples)
    }
}

impl<R: Read + Seek> std::fmt::Debug for Mp4Demux<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mp4Demux")
            .field("tracks", &self.tracks)
            .field("stats", &self.stats)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mp4_codec_display() {
        assert_eq!(format!("{}", Mp4Codec::H264), "H.264/AVC");
        assert_eq!(format!("{}", Mp4Codec::H265), "H.265/HEVC");
        assert_eq!(format!("{}", Mp4Codec::Vp9), "VP9");
        assert_eq!(format!("{}", Mp4Codec::Aac), "AAC");
        assert_eq!(format!("{}", Mp4Codec::Ttxt), "TTXT");
    }

    #[test]
    fn test_mp4_codec_classification() {
        assert!(Mp4Codec::H264.is_video());
        assert!(Mp4Codec::H265.is_video());
        assert!(Mp4Codec::Vp9.is_video());
        assert!(!Mp4Codec::Aac.is_video());

        assert!(Mp4Codec::Aac.is_audio());
        assert!(!Mp4Codec::H264.is_audio());

        assert!(Mp4Codec::Ttxt.is_subtitle());
        assert!(!Mp4Codec::H264.is_subtitle());
    }

    #[test]
    fn test_sample_rate_from_index() {
        assert_eq!(
            Mp4Demux::<std::io::Cursor<Vec<u8>>>::sample_rate_from_index(
                mp4::SampleFreqIndex::Freq44100
            ),
            44100
        );
        assert_eq!(
            Mp4Demux::<std::io::Cursor<Vec<u8>>>::sample_rate_from_index(
                mp4::SampleFreqIndex::Freq48000
            ),
            48000
        );
        assert_eq!(
            Mp4Demux::<std::io::Cursor<Vec<u8>>>::sample_rate_from_index(
                mp4::SampleFreqIndex::Freq96000
            ),
            96000
        );
    }

    #[test]
    fn test_mp4_track_type_from() {
        assert_eq!(Mp4TrackType::from(TrackType::Video), Mp4TrackType::Video);
        assert_eq!(Mp4TrackType::from(TrackType::Audio), Mp4TrackType::Audio);
        assert_eq!(
            Mp4TrackType::from(TrackType::Subtitle),
            Mp4TrackType::Subtitle
        );
    }

    #[test]
    fn test_mp4_codec_from_media_type() {
        assert_eq!(Mp4Codec::from(MediaType::H264), Mp4Codec::H264);
        assert_eq!(Mp4Codec::from(MediaType::H265), Mp4Codec::H265);
        assert_eq!(Mp4Codec::from(MediaType::VP9), Mp4Codec::Vp9);
        assert_eq!(Mp4Codec::from(MediaType::AAC), Mp4Codec::Aac);
        assert_eq!(Mp4Codec::from(MediaType::TTXT), Mp4Codec::Ttxt);
    }
}
