//! MP4/MOV container demuxer.
//!
//! This module provides an MP4 demuxer that extracts elementary streams
//! (video, audio) from MP4/MOV container files.
//!
//! # Supported Codecs
//!
//! | Type | Codec | Notes |
//! |------|-------|-------|
//! | Video | H.264/AVC | Samples converted to Annex-B with in-band SPS/PPS on keyframes |
//! | Video | H.265/HEVC | Passes through in container (length-prefixed) form |
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
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::{BufferFlags, Metadata};

use mp4::{MediaType, Mp4Reader, TrackType};
use std::io::{Read, Seek};

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
// Seeking
// ============================================================================

/// Where a time-based seek landed.
///
/// Returned by [`Mp4Demux::seek_to_time`] and [`Mp4Demux::seek_all_to_time`].
/// The landing time is at or before the requested target (except when the
/// target precedes the track's first sync sample, where the first sync sample
/// is the earliest decodable position).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Mp4SeekPoint {
    /// Track the seek was resolved on.
    pub track_id: u32,
    /// 1-based sample index the next [`Mp4Demux::read_sample`] will return.
    pub sample_index: u32,
    /// Start time of that sample in nanoseconds.
    pub time_ns: u64,
    /// Whether the landing sample is a sync sample (keyframe).
    pub is_sync: bool,
}

/// Locate the last sample whose start time is at or before `target_ticks`.
///
/// `entries` are stts runs as `(sample_count, sample_delta)`. Returns the
/// 1-based sample index and its start time in ticks, clamped to the last
/// sample; `None` for a track with no samples.
fn stts_locate(
    entries: impl IntoIterator<Item = (u32, u32)>,
    target_ticks: u64,
) -> Option<(u32, u64)> {
    let mut index: u64 = 1;
    let mut time: u64 = 0;
    let mut last: Option<(u32, u64)> = None;
    for (count, delta) in entries {
        let (count, delta) = (count as u64, delta as u64);
        if count == 0 {
            continue;
        }
        let span = count * delta;
        if delta > 0 && target_ticks < time + span {
            let n = (target_ticks - time) / delta;
            return Some(((index + n) as u32, time + n * delta));
        }
        // Target lies beyond this run: remember the run's last sample.
        last = Some(((index + count - 1) as u32, time + span - delta));
        index += count;
        time += span;
    }
    last
}

/// Start time in ticks of a 1-based sample index, per the stts runs.
fn stts_time_of(entries: impl IntoIterator<Item = (u32, u32)>, sample: u32) -> Option<u64> {
    let mut index: u64 = 1;
    let mut time: u64 = 0;
    for (count, delta) in entries {
        let (count, delta) = (count as u64, delta as u64);
        if (sample as u64) < index + count {
            return Some(time + (sample as u64 - index) * delta);
        }
        index += count;
        time += count * delta;
    }
    None
}

/// Snap a 1-based sample index back to the nearest sync sample at or before
/// it. `sync_samples` is the sorted stss entry list; `None` means every
/// sample is a sync sample. A target before the first sync sample snaps
/// *forward* to it — there is nothing decodable earlier.
fn snap_to_sync(sync_samples: Option<&[u32]>, sample: u32) -> (u32, bool) {
    match sync_samples {
        None => (sample, true),
        Some([]) => (sample, false),
        Some(entries) => match entries.binary_search(&sample) {
            Ok(_) => (sample, true),
            Err(0) => (entries[0], true),
            Err(i) => (entries[i - 1], true),
        },
    }
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
    /// H.264 decoder configuration per track, lifted from the avcC box.
    /// Tracks in the map get their samples converted to Annex-B.
    avc_configs: std::collections::HashMap<u32, AvcDecoderConfig>,
    /// Per-instance output arena for demuxed samples.
    ///
    /// This used to be a process-wide `static`, so two demuxers in one process
    /// drew from the same 32 slots and neither could be sized for its own
    /// stream (#95). `Mp4Demux` implements no element trait, so the executor
    /// cannot reach it — [`set_output_budget`](Self::set_output_budget) is
    /// there for whatever wraps it (see #76).
    output: OutputArena,
}

/// Per-track H.264 decoder configuration from the avcC box.
///
/// MP4 stores H.264 as AVCC: length-prefixed NALs with the parameter sets
/// out-of-band in the sample entry. Decoders consume Annex-B, so the demuxer
/// needs both the prefix size and the parameter sets to emit self-contained
/// access units.
struct AvcDecoderConfig {
    /// NAL length prefix size in bytes (1, 2 or 4).
    length_size: u8,
    /// Raw SPS NALs (no start codes).
    sps: Vec<Vec<u8>>,
    /// Raw PPS NALs (no start codes).
    pps: Vec<Vec<u8>>,
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
        let mut avc_configs = std::collections::HashMap::new();

        for track in mp4_reader.tracks().values() {
            let track_type = track
                .track_type()
                .map(Mp4TrackType::from)
                .unwrap_or(Mp4TrackType::Unknown);

            let codec = track
                .media_type()
                .map(Mp4Codec::from)
                .unwrap_or(Mp4Codec::Unknown);

            // H.264 tracks: lift the avcC record so samples can be converted
            // to Annex-B with in-band parameter sets. A track without one
            // passes through unconverted (and undecodable — say so).
            if codec == Mp4Codec::H264 {
                match &track.trak.mdia.minf.stbl.stsd.avc1 {
                    Some(avc1) => {
                        let avcc = &avc1.avcc;
                        avc_configs.insert(
                            track.track_id(),
                            AvcDecoderConfig {
                                length_size: avcc.length_size_minus_one + 1,
                                sps: avcc
                                    .sequence_parameter_sets
                                    .iter()
                                    .map(|n| n.bytes.clone())
                                    .collect(),
                                pps: avcc
                                    .picture_parameter_sets
                                    .iter()
                                    .map(|n| n.bytes.clone())
                                    .collect(),
                            },
                        );
                    }
                    None => tracing::warn!(
                        "H.264 track {} has no avc1/avcC box; samples pass through \
                         length-prefixed and will not decode",
                        track.track_id()
                    ),
                }
            }

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
            avc_configs,
            output: OutputArena::new(defaults::MP4_DEMUX_SLOT_COUNT)
                .with_min_slot_size(defaults::MP4_DEMUX_SLOT_SIZE)
                .grow_to_fit(),
        })
    }

    /// Size this demuxer's output arena from the graph below it.
    ///
    /// `Mp4Demux` implements no element trait, so nothing calls this yet — the
    /// wrapper added by #76 will. Without it the arena falls back to
    /// [`defaults::MP4_DEMUX_SLOT_COUNT`].
    pub fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    /// Convert one AVCC video sample to a self-contained Annex-B access unit.
    ///
    /// Sync samples get every SPS/PPS from the avcC record prepended so a
    /// decoder can join at any keyframe (MP4 keeps parameter sets out-of-band;
    /// Annex-B consumers expect them in-band).
    fn avcc_sample_to_annex_b(
        cfg: &AvcDecoderConfig,
        data: &[u8],
        is_sync: bool,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(data.len() + 64);
        if is_sync {
            for nal in cfg.sps.iter().chain(cfg.pps.iter()) {
                out.extend_from_slice(&[0, 0, 0, 1]);
                out.extend_from_slice(nal);
            }
        }
        out.extend_from_slice(&crate::codec::annexb::avcc_to_annex_b(
            data,
            cfg.length_size,
        )?);
        Ok(out)
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

    /// Seek one track to a target time, landing on a decodable sample.
    ///
    /// Resolves `target_ns` to a sample via the time-to-sample table (stts),
    /// then snaps back to the nearest sync sample via the sync-sample table
    /// (stss) — when a track has no stss box every sample is a sync sample,
    /// which is the normal case for audio. The landing time is therefore at
    /// or before the target, except when the target precedes the first sync
    /// sample (nothing earlier is decodable, so the seek snaps forward to it).
    ///
    /// The next [`read_sample`](Self::read_sample) on this track returns the
    /// landing sample. On a video track the demuxer re-emits SPS/PPS in-band
    /// on keyframes, so a decoder can restart cleanly at the landing point.
    ///
    /// To seek several tracks consistently, use
    /// [`seek_all_to_time`](Self::seek_all_to_time) — seeking each track to
    /// the same target independently can land video and audio a whole GOP
    /// apart, because only video snaps to keyframes.
    ///
    /// # Errors
    ///
    /// Returns an error if the track does not exist, has no samples, or has
    /// a zero timescale.
    pub fn seek_to_time(&mut self, track_id: u32, target_ns: u64) -> Result<Mp4SeekPoint> {
        let track = self
            .reader
            .tracks()
            .get(&track_id)
            .ok_or_else(|| Error::Config(format!("Track {} not found", track_id)))?;

        let timescale = track.timescale();
        if timescale == 0 {
            return Err(Error::Config(format!(
                "Track {} has a zero timescale",
                track_id
            )));
        }

        let target_ticks = (target_ns as u128 * timescale as u128 / 1_000_000_000) as u64;

        let stbl = &track.trak.mdia.minf.stbl;
        let stts = || {
            stbl.stts
                .entries
                .iter()
                .map(|e| (e.sample_count, e.sample_delta))
        };

        let (sample, _) = stts_locate(stts(), target_ticks).ok_or_else(|| {
            Error::Config(format!("Track {} has no samples to seek in", track_id))
        })?;

        let sync_entries = stbl.stss.as_ref().map(|s| s.entries.as_slice());
        let (landing, is_sync) = snap_to_sync(sync_entries, sample);

        // The snap changed the index, so recompute its start time.
        let landing_ticks = stts_time_of(stts(), landing).unwrap_or(0);
        let time_ns = (landing_ticks as u128 * 1_000_000_000 / timescale as u128) as u64;

        self.sample_indices.insert(track_id, landing);

        Ok(Mp4SeekPoint {
            track_id,
            sample_index: landing,
            time_ns,
            is_sync,
        })
    }

    /// Seek every track to a consistent position at or before `target_ns`.
    ///
    /// The target is first resolved on the video track (keyframe snap via
    /// [`seek_to_time`](Self::seek_to_time)); every other track is then
    /// sought to the video's landing time, so audio starts where the video
    /// keyframe does instead of up to a GOP later. Files without a video
    /// track resolve the target on the first track instead.
    ///
    /// Returns the seek point of the reference track.
    pub fn seek_all_to_time(&mut self, target_ns: u64) -> Result<Mp4SeekPoint> {
        let reference = self
            .video_track_id()
            .or_else(|| self.tracks.first().map(|t| t.id))
            .ok_or_else(|| Error::Config("MP4 has no tracks to seek".into()))?;

        let point = self.seek_to_time(reference, target_ns)?;

        let other_ids: Vec<u32> = self
            .tracks
            .iter()
            .map(|t| t.id)
            .filter(|id| *id != reference)
            .collect();
        for id in other_ids {
            self.seek_to_time(id, point.time_ns)?;
        }

        Ok(point)
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

        // H.264: convert the AVCC sample to a self-contained Annex-B access
        // unit; anything else passes through in container form.
        let payload: std::borrow::Cow<'_, [u8]> = match self.avc_configs.get(&track_id) {
            Some(cfg) => std::borrow::Cow::Owned(Self::avcc_sample_to_annex_b(
                cfg,
                &sample.bytes,
                sample.is_sync,
            )?),
            None => std::borrow::Cow::Borrowed(&sample.bytes),
        };

        // Create buffer
        let buffer = self.create_buffer(&payload, track_id, &sample, &track)?;

        // Calculate timestamps in nanoseconds
        let timescale = track.timescale as u128;
        let pts_ns = (sample.start_time as u128 * 1_000_000_000)
            .checked_div(timescale)
            .unwrap_or(0) as u64;

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

        let duration_ns = (sample.duration as u128 * 1_000_000_000)
            .checked_div(timescale)
            .unwrap_or(0) as u64;

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
        &mut self,
        data: &[u8],
        track_id: u32,
        sample: &mp4::Mp4Sample,
        track: &Mp4Track,
    ) -> Result<Buffer> {
        // `grow_to_fit`: a sample larger than the current slot rebuilds the
        // arena rather than failing, because a demuxer's output size follows
        // the file, not a ceiling it chose.
        let mut slot = self.output.acquire(data.len(), "mp4demux")?;
        slot.data_mut()[..data.len()].copy_from_slice(data);

        let handle = MemoryHandle::with_len(slot, data.len());

        // Build metadata
        let timescale = track.timescale as u128;

        let mut metadata = Metadata::new();
        metadata.stream_id = track_id;

        if let Some(pts) = (sample.start_time as u128 * 1_000_000_000).checked_div(timescale) {
            metadata.pts = ClockTime::from_nanos(pts as u64);

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

    type TestDemux = Mp4Demux<std::io::Cursor<Vec<u8>>>;

    fn test_avc_config() -> AvcDecoderConfig {
        AvcDecoderConfig {
            length_size: 4,
            sps: vec![vec![0x67, 0x42, 0x00, 0x1F]],
            pps: vec![vec![0x68, 0xCE, 0x3C, 0x80]],
        }
    }

    #[test]
    fn avcc_sync_sample_gets_in_band_parameter_sets() {
        // One IDR NAL, 4-byte length prefix.
        let sample = [0u8, 0, 0, 2, 0x65, 0xAA];
        let out = TestDemux::avcc_sample_to_annex_b(&test_avc_config(), &sample, true).unwrap();

        let types: Vec<u8> = crate::codec::annexb::nal_units(&out)
            .map(|n| n.nal_type())
            .collect();
        assert_eq!(types, vec![7, 8, 5], "SPS, PPS, then the slice");
    }

    #[test]
    fn avcc_delta_sample_converts_without_prefix() {
        let sample = [0u8, 0, 0, 3, 0x41, 0x9A, 0x02];
        let out = TestDemux::avcc_sample_to_annex_b(&test_avc_config(), &sample, false).unwrap();
        assert_eq!(out, vec![0, 0, 0, 1, 0x41, 0x9A, 0x02]);
    }

    #[test]
    fn avcc_truncated_sample_errors() {
        // Length prefix claims 9 bytes; only 1 remains.
        let sample = [0u8, 0, 0, 9, 0x41];
        assert!(TestDemux::avcc_sample_to_annex_b(&test_avc_config(), &sample, false).is_err());
    }

    // 10 samples of 100 ticks, then 5 of 200 ticks: track spans 0..2000.
    const STTS: [(u32, u32); 2] = [(10, 100), (5, 200)];

    #[test]
    fn stts_locate_finds_containing_sample() {
        assert_eq!(stts_locate(STTS, 0), Some((1, 0)));
        assert_eq!(
            stts_locate(STTS, 99),
            Some((1, 0)),
            "mid-sample rounds down"
        );
        assert_eq!(stts_locate(STTS, 100), Some((2, 100)));
        assert_eq!(stts_locate(STTS, 999), Some((10, 900)), "last of first run");
        assert_eq!(
            stts_locate(STTS, 1000),
            Some((11, 1000)),
            "first of second run"
        );
        assert_eq!(stts_locate(STTS, 1350), Some((12, 1200)));
    }

    #[test]
    fn stts_locate_clamps_past_the_end() {
        assert_eq!(
            stts_locate(STTS, 5000),
            Some((15, 1800)),
            "clamps to last sample"
        );
        assert_eq!(stts_locate([], 0), None, "no samples");
        assert_eq!(
            stts_locate([(0, 100)], 50),
            None,
            "empty runs carry no samples"
        );
    }

    #[test]
    fn stts_time_of_inverts_locate() {
        assert_eq!(stts_time_of(STTS, 1), Some(0));
        assert_eq!(stts_time_of(STTS, 10), Some(900));
        assert_eq!(stts_time_of(STTS, 11), Some(1000));
        assert_eq!(stts_time_of(STTS, 15), Some(1800));
        assert_eq!(stts_time_of(STTS, 16), None, "past the end");
    }

    #[test]
    fn snap_to_sync_snaps_backwards() {
        let stss = [1u32, 6, 11];
        assert_eq!(
            snap_to_sync(Some(&stss), 6),
            (6, true),
            "already a keyframe"
        );
        assert_eq!(snap_to_sync(Some(&stss), 9), (6, true), "snaps back");
        assert_eq!(snap_to_sync(Some(&stss), 15), (11, true));
        assert_eq!(
            snap_to_sync(None, 7),
            (7, true),
            "no stss: everything is sync"
        );
    }

    #[test]
    fn snap_to_sync_before_first_keyframe_goes_forward() {
        // Pathological file whose first sync sample is not sample 1: nothing
        // before it is decodable, so the seek moves forward.
        let stss = [4u32, 8];
        assert_eq!(snap_to_sync(Some(&stss), 2), (4, true));
    }
}
