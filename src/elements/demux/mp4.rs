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
//! | Video | AV1 | `av01` recovered by the stsd side-scan; samples are plain OBUs |
//! | Audio | AAC | Most common |
//! | Audio | Opus | `Opus`/`dOps` recovered by the stsd side-scan; raw Opus packets |
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
use crate::event::SeekSnap;
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::{BufferFlags, Metadata};

use super::mp4_extra::{PatchedReader, config_first_patches};
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
    /// AV1 video (`av01` sample entry, recovered by the stsd scan — the
    /// mp4 crate itself does not model it).
    Av1,
    /// AAC audio.
    Aac,
    /// Opus audio (`Opus` sample entry, recovered by the stsd scan).
    Opus,
    /// TTML/TTXT subtitles.
    Ttxt,
    /// Unknown codec.
    Unknown,
}

impl Mp4Codec {
    /// The pipeline's video codec for this track, if it is video (#238).
    ///
    /// `None` for audio, subtitles, and codecs parallax has no variant for —
    /// an undeclared format beats a mislabelled one.
    fn video_codec(&self) -> Option<crate::format::VideoCodec> {
        use crate::format::VideoCodec;
        Some(match self {
            Self::H264 => VideoCodec::H264,
            Self::H265 => VideoCodec::H265,
            Self::Vp9 => VideoCodec::Vp9,
            Self::Av1 => VideoCodec::Av1,
            _ => return None,
        })
    }

    /// The pipeline's audio codec for this track, if it is audio (#238).
    fn audio_codec(&self) -> Option<crate::format::AudioCodec> {
        use crate::format::AudioCodec;
        Some(match self {
            Self::Aac => AudioCodec::Aac,
            Self::Opus => AudioCodec::Opus,
            _ => return None,
        })
    }

    /// Returns true if this is a video codec.
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            Mp4Codec::H264 | Mp4Codec::H265 | Mp4Codec::Vp9 | Mp4Codec::Av1
        )
    }

    /// Returns true if this is an audio codec.
    pub fn is_audio(&self) -> bool {
        matches!(self, Mp4Codec::Aac | Mp4Codec::Opus)
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
            Mp4Codec::Av1 => write!(f, "AV1"),
            Mp4Codec::Aac => write!(f, "AAC"),
            Mp4Codec::Opus => write!(f, "Opus"),
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

/// What the container says about a track, beyond its codec (#278).
///
/// MP4 spreads this across three boxes and mp4-rs exposes one of them
/// directly: `mdhd`'s packed 5-bit-per-letter language code comes through
/// `track.language()`, while `hdlr`'s name — where several muxers write a
/// track title — and `tkhd`'s enabled flag are reachable only because
/// `mp4::Mp4Track::trak` is a public field. There is no forced flag in MP4 at
/// all, and `elng` (the BCP-47 extended language tag) is not modelled by the
/// crate; both are an honest `None`/`false` rather than something invented.
fn track_label(track: &mp4::Mp4Track) -> super::TrackLabel {
    const TKHD_TRACK_ENABLED: u32 = 0x0000_0001;
    let hdlr = &track.trak.mdia.hdlr;
    // Muxers write "VideoHandler"/"SoundHandler"/"Core Media Video" here as a
    // matter of course; that is the tool talking about itself, not a title
    // anybody wants in a menu.
    let name = Some(hdlr.name.trim())
        .filter(|name| {
            !name.is_empty()
                && !name.ends_with("Handler")
                && !name.eq_ignore_ascii_case("core media video")
                && !name.eq_ignore_ascii_case("core media audio")
        })
        .map(str::to_string);

    super::TrackLabel {
        language: Some(track.language().to_string()),
        name,
        // MP4 has no per-track default flag; `alternate_group` says which
        // tracks are alternatives to each other, not which one to pick.
        default: false,
        forced: false,
        enabled: track.trak.tkhd.flags & TKHD_TRACK_ENABLED != 0,
    }
}

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
    /// Language, name and flags — what a picker labels a row with (#278).
    pub label: super::TrackLabel,
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
    /// What the sample entry's `colr` box said (#265), if it carried one.
    pub color: Option<crate::format::ColorInfo>,
    /// Pixel aspect ratio (`pasp`) and rotation (`tkhd` matrix), #266.
    pub display: Option<crate::format::DisplayGeometry>,
}

/// Audio track information.
///
/// For AAC tracks the esds `DecoderSpecificDescriptor` fields are lifted so a
/// decoder can be initialized from what the demuxer provides — the audio
/// equivalent of the avcC treatment H.264 tracks get.
#[derive(Debug, Clone)]
pub struct Mp4AudioInfo {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
    /// AAC audio object type from the esds box (2 = AAC-LC), if present.
    pub object_type: Option<u8>,
    /// AAC sampling-frequency index from the esds box, if present.
    pub freq_index: Option<u8>,
    /// AAC channel configuration from the esds box, if present.
    pub channel_config: Option<u8>,
    /// The `AudioSpecificConfig` bytes an AAC decoder initializes from.
    ///
    /// Rebuilt from the three esds fields above (the mp4 crate parses the
    /// descriptor into fields rather than keeping the raw bytes), which is
    /// exact for the plain AAC family — object types 1–30 with a standard
    /// frequency index. An explicit-frequency stream (index 15) or an ASC
    /// with extension payloads (HE-AAC's SBR block) cannot be reconstructed
    /// from the parsed fields, so those stay `None`.
    pub audio_specific_config: Option<Vec<u8>>,
    /// Raw codec configuration for non-esds codecs — the `dOps` payload for
    /// Opus tracks (mirrors `MkvAudioInfo::codec_private`).
    pub codec_private: Option<Vec<u8>>,
}

impl Mp4AudioInfo {
    /// Build the 7-byte ADTS header for a raw AAC frame of `frame_len` bytes.
    ///
    /// `None` when the track carried no esds configuration, or when the frame
    /// is too large for the 13-bit ADTS length field. See
    /// [`Mp4Demux::with_adts_aac`] to have the demuxer prepend this to every
    /// AAC sample instead.
    pub fn adts_header(&self, frame_len: usize) -> Option<[u8; 7]> {
        Self::adts_header_from(
            self.object_type?,
            self.freq_index?,
            self.channel_config?,
            frame_len,
        )
    }

    /// [`adts_header`](Self::adts_header) from the bare esds fields — lets
    /// `read_sample` keep per-track Copy state instead of cloning the whole
    /// track (config Vecs included) per sample (#144).
    pub(crate) fn adts_header_from(
        object_type: u8,
        freq_index: u8,
        chan_conf: u8,
        frame_len: usize,
    ) -> Option<[u8; 7]> {
        let full_len = frame_len.checked_add(7)?;
        if full_len > 0x1FFF || object_type == 0 {
            return None;
        }
        let len = full_len as u16;
        // ADTS profile is the object type minus one, two bits wide.
        let profile = (object_type - 1) & 0x3;

        Some([
            0xFF, // syncword
            0xF1, // syncword end, MPEG-4, layer 0, no CRC
            (profile << 6) | ((freq_index & 0xF) << 2) | ((chan_conf >> 2) & 0x1),
            ((chan_conf & 0x3) << 6) | ((len >> 11) as u8 & 0x3),
            (len >> 3) as u8,
            (((len & 0x7) as u8) << 5) | 0x1F, // buffer fullness: VBR
            0xFC,                              // buffer fullness end, 1 AAC frame
        ])
    }
}

/// Rebuild the two-byte `AudioSpecificConfig` from parsed esds fields.
///
/// Layout: 5 bits object type, 4 bits frequency index, 4 bits channel
/// configuration, 3 zero bits (frame length 1024, no core coder, no
/// extension). Only valid for object types 1–30 and frequency indices 0–14 —
/// the escape encodings need bytes the parsed fields no longer carry.
fn synthesize_asc(object_type: u8, freq_index: u8, chan_conf: u8) -> Option<Vec<u8>> {
    if object_type == 0 || object_type > 30 || freq_index > 14 || chan_conf > 15 {
        return None;
    }
    Some(vec![
        (object_type << 3) | (freq_index >> 1),
        ((freq_index & 1) << 7) | (chan_conf << 3),
    ])
}

// ============================================================================
// Seeking
// ============================================================================

/// Where a time-based seek landed.
///
/// Returned by [`Mp4Demux::seek_to_time`] and [`Mp4Demux::seek_all_to_time`].
/// The landing time is on a sync sample in the direction the caller's
/// [`SeekSnap`] chose: `Before` lands at or before the target (except when
/// the target precedes the track's first sync sample, where the first sync
/// sample is the earliest decodable position), `After` at or after it
/// (clamped back to the last sync sample past the end of the track), and
/// `Nearest` on whichever is closer in time.
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

/// Composition-time offset of a 1-based sample index, per the ctts runs.
///
/// `entries` are ctts runs as `(sample_count, sample_offset)`. Zero for a
/// track with no `ctts` box — no reordering, so composition time *is* decode
/// time.
///
/// This exists because the seek tables and the emitted timestamps are on
/// different clocks (#265). `stts_locate` and `stts_time_of` walk `stts`
/// only, which is **decode** time, while `read_sample` stamps
/// **composition** time (see `sample_times_ns`). So `Mp4SeekPoint.time_ns`
/// — which becomes the synthesized segment's `start`, the `SeekDone`
/// position, and the target every non-reference track is re-sought to —
/// has to add the ctts offset, or it is a DTS being compared against CTS
/// everywhere downstream. On any stream with B-frames that puts audio ahead
/// of video by the reorder delay, and leaves the first video frame above the
/// segment start where nothing would clip it.
///
/// Before #304, `read_sample` stamped the decode time as the PTS, so adding
/// this offset was what made a seek's landing disagree with the sample it
/// landed on.
fn ctts_offset_of(entries: impl IntoIterator<Item = (u32, i32)>, sample: u32) -> i64 {
    let mut index: u64 = 1;
    for (count, offset) in entries {
        let count = count as u64;
        if (sample as u64) < index + count {
            return offset as i64;
        }
        index += count;
    }
    0
}

/// A sample's `(pts, dts)` in nanoseconds, or `None` for a zero timescale.
///
/// `mp4::Mp4Sample::start_time` is the sample's **decode** time — the crate
/// sums `stts` deltas for it — and `rendering_offset` is its `ctts` entry, so
/// composition time is their sum. `shift` is the track's edit-list shift
/// (`edit_shift_ticks`), taken off both so that the first frame shown is
/// presented at zero. `read_sample` computes them once, for both the
/// buffer's `Metadata` and the `Mp4Sample` it returns, so they cannot
/// disagree.
///
/// It used to be read the other way round (#304): `start_time` as the PTS and
/// `start_time - rendering_offset` as the DTS. A B-frame stream then reached
/// the decoder with monotonic decode times in decode order, a reordering
/// decoder (VA-API) carried them into display order, and every frame that
/// came out behind its predecessor was dropped as late — about half of them.
/// `Mp4Mux::write_sample` has always written the crate's meaning, so the two
/// ends of a round trip disagreed.
///
/// A time that comes out negative clamps to zero: the DTS of the first
/// frames of any reordered stream with an edit list, or the PTS of samples
/// an edit cuts off, which this demuxer does not discard.
///
/// So the first frames of an x264 file share a DTS of zero, and that is the
/// honest limit of an unsigned clock rather than an oversight: their decode
/// times really are before the presentation origin, which is what "the first
/// frame is shown at zero and something had to be decoded before it" means.
/// The two ways out both give up something worth more. Leaving the DTS
/// unshifted puts it *after* the PTS on every B frame, which is not a decode
/// time at all. Shifting both by less, so the DTS stays positive, hands the
/// reorder delay back to the PTS — the A/V offset the edit list is there to
/// remove. What the clamp keeps is what a consumer can actually lean on: the
/// DTS never runs backwards and never passes the PTS
/// (`under_an_edit_shift_the_dts_neither_runs_backwards_nor_passes_the_pts`).
/// A muxer that needs it *strictly* increasing has to offset the whole
/// timeline, as every muxer facing a negative DTS does.
fn sample_times_ns(
    start_time: u64,
    rendering_offset: i32,
    shift: i64,
    timescale: u32,
) -> Option<(u64, u64)> {
    if timescale == 0 {
        return None;
    }
    let to_ns = |ticks: i128| (ticks.max(0) as u128 * 1_000_000_000 / timescale as u128) as u64;
    let dts = start_time as i128 - shift as i128;
    Some((to_ns(dts + rendering_offset as i128), to_ns(dts)))
}

/// How far a track's composition timeline sits ahead of its presentation
/// timeline, in track ticks: presentation = composition − shift.
///
/// From the edit list (`edts/elst`), as `(segment_duration, media_time)`
/// pairs. A reordering encoder writes one: ffmpeg's x264 output sets the
/// first edit's `media_time` to the first frame's ctts offset, so that the
/// first frame shown is presented at zero. Ignoring it puts a correctly
/// composed B-frame stream (#304) one to three frames behind its audio.
/// Leading empty edits (`media_time` of -1) delay the presentation by their
/// duration, which is in the **movie** timescale.
///
/// Only the first media edit is honoured: a multi-segment edit list (cuts,
/// repeats) needs a timeline, not a shift.
fn edit_shift_ticks(
    entries: impl IntoIterator<Item = (u64, i64)>,
    track_timescale: u32,
    movie_timescale: u32,
) -> i64 {
    // Leading empty edits, in the movie timescale.
    let mut empty: u64 = 0;
    for (segment_duration, media_time) in entries {
        if media_time < 0 {
            empty = empty.saturating_add(segment_duration);
            continue;
        }
        let delay = (empty as u128 * track_timescale as u128)
            .checked_div(movie_timescale as u128)
            .unwrap_or(0) as i64;
        return media_time.saturating_sub(delay);
    }
    0
}

/// An `elst` entry's `media_time`, with an empty edit's -1 restored: mp4 0.14
/// reads both box versions as unsigned and widens them to `u64`.
fn elst_media_time(version: u8, raw: u64) -> i64 {
    if version == 1 {
        raw as i64
    } else {
        raw as u32 as i32 as i64
    }
}

/// A track's edit list as `(segment_duration, media_time)` pairs.
fn elst_entries(track: &mp4::Mp4Track) -> Vec<(u64, i64)> {
    let Some(elst) = track.trak.edts.as_ref().and_then(|e| e.elst.as_ref()) else {
        return Vec::new();
    };
    let version = elst.version;
    elst.entries
        .iter()
        .map(|e| (e.segment_duration, elst_media_time(version, e.media_time)))
        .collect()
}

/// A seek target, moved from the clock the pipeline speaks onto the one
/// `stts` is written in.
///
/// `read_sample` stamps presentation time: decode + ctts − shift
/// (`sample_times_ns`). The time-to-sample table knows decode time only, so a
/// target has to be taken back through the same arithmetic before
/// `stts_locate` can be asked about it — or the lookup answers a different
/// question from the one the landing is then reported on. `sync_ctts` is the
/// composition offset of the track's sync samples, which are the only samples
/// a seek lands on; the caller reads it off the first one.
///
/// For ffmpeg's x264 output the two terms cancel (the edit's media time *is*
/// the keyframes' ctts offset), which is why looking the target up as a
/// decode time appeared to work. They do not cancel on a track with no ctts
/// and an edit list — audio behind a leading empty edit was looked up that
/// edit's whole duration late, and audio is re-sought to the video's landing
/// on every seek, so it opened that far from the picture — nor on a B-frame
/// track with no edit list, where a target between a keyframe's decode time
/// and its presentation time landed on that keyframe and `SeekSnap::Before`
/// came back *after* the target.
fn seek_target_on_decode_clock(target_ticks: u64, shift: i64, sync_ctts: i64) -> u64 {
    (target_ticks as i128 + shift as i128 - sync_ctts as i128).max(0) as u64
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

/// Snap a 1-based sample index forward to the nearest sync sample at or
/// after it. A target past the last sync sample clamps *back* to it — there
/// is nothing decodable later.
fn snap_to_sync_after(sync_samples: Option<&[u32]>, sample: u32) -> (u32, bool) {
    match sync_samples {
        None => (sample, true),
        Some([]) => (sample, false),
        Some(entries) => match entries.binary_search(&sample) {
            Ok(_) => (sample, true),
            Err(i) if i < entries.len() => (entries[i], true),
            Err(_) => (*entries.last().expect("non-empty stss"), true),
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
    /// The crate's reader, over the file with each first sample entry's
    /// configuration box moved to the front (#305, `config_first_patches`).
    reader: Mp4Reader<PatchedReader<R>>,
    tracks: Vec<Mp4Track>,
    stats: Mp4DemuxStats,
    /// Current sample index per track (track_id -> next sample index).
    sample_indices: std::collections::HashMap<u32, u32>,
    /// H.264 decoder configuration per track, lifted from the avcC box.
    /// Tracks in the map get their samples converted to Annex-B.
    nal_configs: std::collections::HashMap<u32, crate::codec::annexb::NalConfig>,
    /// Edit-list shift per track, in track ticks (`edit_shift_ticks`).
    edit_shifts: std::collections::HashMap<u32, i64>,
    /// Per-instance output arena for demuxed samples.
    ///
    /// This used to be a process-wide `static`, so two demuxers in one process
    /// drew from the same 32 slots and neither could be sized for its own
    /// stream (#95). `Mp4Demux` implements no element trait, so the executor
    /// cannot reach it — [`set_output_budget`](Self::set_output_budget) is
    /// there for whatever wraps it (see #76).
    output: OutputArena,
    /// Prepend an ADTS header to every AAC sample (see [`with_adts_aac`](Self::with_adts_aac)).
    adts_aac: bool,
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
        // Recover sample entries the mp4 crate drops (av01, Opus) before it
        // consumes the reader; tolerant — an unscannable file just leaves
        // the affected tracks Unknown, as before.
        let mut reader = reader;
        let stsd_extras = super::mp4_extra::scan(&mut reader).unwrap_or_default();

        // mp4-rs takes a sample entry's first child for its configuration
        // and rejects the whole file when it is not (#305). Show it the
        // entries with that box first; tolerant like the scan above, so an
        // unwalkable file reaches the crate exactly as before.
        let patches = config_first_patches(&mut reader).unwrap_or_default();
        let reader = PatchedReader::new(reader, patches)
            .map_err(|e| Error::Config(format!("Failed to read MP4 header: {}", e)))?;

        let mp4_reader = Mp4Reader::read_header(reader, size)
            .map_err(|e| Error::Config(format!("Failed to read MP4 header: {}", e)))?;

        let mut tracks = Vec::new();
        let mut sample_indices = std::collections::HashMap::new();
        let mut nal_configs = std::collections::HashMap::new();
        let mut edit_shifts = std::collections::HashMap::new();
        let movie_timescale = mp4_reader.timescale();

        for track in mp4_reader.tracks().values() {
            let track_type = track
                .track_type()
                .map(Mp4TrackType::from)
                .unwrap_or(Mp4TrackType::Unknown);

            let mut codec = track
                .media_type()
                .map(Mp4Codec::from)
                .unwrap_or(Mp4Codec::Unknown);
            let extra = stsd_extras
                .iter()
                .find(|(id, _)| *id == track.track_id())
                .map(|(_, e)| e);
            if codec == Mp4Codec::Unknown
                && let Some(extra) = extra
            {
                codec = match &extra.fourcc {
                    b"av01" => Mp4Codec::Av1,
                    b"Opus" => Mp4Codec::Opus,
                    _ => Mp4Codec::Unknown,
                };
            }

            // H.264 tracks: lift the avcC record so samples can be converted
            // to Annex-B with in-band parameter sets. A track without one
            // passes through unconverted (and undecodable — say so).
            if codec == Mp4Codec::H265 {
                // HEVC is stored exactly like H.264 — length-prefixed NALs
                // with the parameter sets out of band — so it wants the same
                // Annex-B conversion. Only the record differs, and the mp4
                // crate does not expose it, so the raw scan recovers it.
                match extra.and_then(|e| e.hvcc.as_ref()) {
                    Some(hvcc) => {
                        nal_configs
                            .insert(track.track_id(), crate::codec::annexb::parse_hvcc(hvcc)?);
                    }
                    None => tracing::warn!(
                        "HEVC track {} has no hvcC record; samples pass through \
                         length-prefixed and will not decode",
                        track.track_id()
                    ),
                }
            }
            if codec == Mp4Codec::H264 {
                match &track.trak.mdia.minf.stbl.stsd.avc1 {
                    Some(avc1) => {
                        let avcc = &avc1.avcc;
                        nal_configs.insert(
                            track.track_id(),
                            crate::codec::annexb::NalConfig {
                                length_size: avcc.length_size_minus_one + 1,
                                params: avcc
                                    .sequence_parameter_sets
                                    .iter()
                                    .chain(avcc.picture_parameter_sets.iter())
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
                    color: extra.and_then(|e| e.color),
                    display: extra.and_then(|e| e.display),
                })
            } else {
                None
            };

            let audio_info = if track_type == Mp4TrackType::Audio {
                // Lift the esds DecoderSpecificDescriptor fields directly —
                // this is what an AAC decoder needs to initialize (#69).
                let dec_specific = track
                    .trak
                    .mdia
                    .minf
                    .stbl
                    .stsd
                    .mp4a
                    .as_ref()
                    .and_then(|mp4a| mp4a.esds.as_ref())
                    .map(|esds| &esds.es_desc.dec_config.dec_specific);
                let dops = extra.and_then(|e| e.dops.as_ref());
                Some(Mp4AudioInfo {
                    // Opus always decodes at 48 kHz regardless of the
                    // original input rate the dOps box records.
                    sample_rate: if codec == Mp4Codec::Opus {
                        48_000
                    } else {
                        track
                            .sample_freq_index()
                            .map(|i| Self::sample_rate_from_index(i))
                            .unwrap_or(44100)
                    },
                    channels: dops.map(|d| d.channels as u16).unwrap_or_else(|| {
                        track
                            .channel_config()
                            .map(|c| Self::channel_count(c))
                            .unwrap_or(2)
                    }),
                    object_type: dec_specific.map(|d| d.profile),
                    freq_index: dec_specific.map(|d| d.freq_index),
                    channel_config: dec_specific.map(|d| d.chan_conf),
                    audio_specific_config: dec_specific
                        .and_then(|d| synthesize_asc(d.profile, d.freq_index, d.chan_conf)),
                    codec_private: dops.map(|d| d.raw.clone()),
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
                label: track_label(track),
            });

            sample_indices.insert(track.track_id(), 1); // Samples are 1-indexed

            let edits = elst_entries(track);
            let shift = edit_shift_ticks(edits, track.timescale(), movie_timescale);
            edit_shifts.insert(track.track_id(), shift);
        }

        Ok(Self {
            reader: mp4_reader,
            tracks,
            stats: Mp4DemuxStats::default(),
            sample_indices,
            nal_configs,
            edit_shifts,
            output: OutputArena::new(defaults::MP4_DEMUX_SLOT_COUNT)
                .with_min_slot_size(defaults::DEMUX_MIN_SLOT_SIZE)
                .grow_to_fit(),
            adts_aac: false,
        })
    }

    /// Wrap every AAC sample in an ADTS header on the way out.
    ///
    /// MP4 stores AAC raw (the configuration lives out-of-band in the esds
    /// box); consumers that expect a self-describing stream — ADTS parsers,
    /// files meant to be playable on their own — need the 7-byte header back
    /// on every frame. Samples of tracks with no esds configuration pass
    /// through unwrapped.
    pub fn with_adts_aac(mut self, adts: bool) -> Self {
        self.adts_aac = adts;
        self
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
    #[cfg(test)] // production converts straight into the slot (#144)
    fn avcc_sample_to_annex_b(
        cfg: &crate::codec::annexb::NalConfig,
        data: &[u8],
        is_sync: bool,
    ) -> Result<Vec<u8>> {
        let mut out = Vec::with_capacity(data.len() + 64);
        if is_sync {
            for nal in &cfg.params {
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
    /// Resolves `target_ns` to a sample via the time-to-sample table (stts)
    /// — after moving it onto that table's clock, which is decode time
    /// (`seek_target_on_decode_clock`) — then snaps to a sync sample via the sync-sample table (stss) in the
    /// direction `snap` chose — when a track has no stss box every sample is
    /// a sync sample, which is the normal case for audio, and the snap is a
    /// no-op. `SeekSnap::Before` lands at or before the target (a target
    /// before the first sync sample snaps forward to it — nothing earlier is
    /// decodable), `After` at or after it (a target past the last sync
    /// sample clamps back to it), `Nearest` on whichever is closer in time,
    /// ties breaking toward `Before`.
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
    pub fn seek_to_time(
        &mut self,
        track_id: u32,
        target_ns: u64,
        snap: SeekSnap,
    ) -> Result<Mp4SeekPoint> {
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

        let ctts = || {
            stbl.ctts
                .iter()
                .flat_map(|c| c.entries.iter().map(|e| (e.sample_count, e.sample_offset)))
        };
        let sync_entries = stbl.stss.as_ref().map(|s| s.entries.as_slice());
        let shift = self.edit_shifts.get(&track_id).copied().unwrap_or(0);

        // `stts` is decode time and the target is presentation time: convert
        // before looking it up. See `seek_target_on_decode_clock`.
        let first_sync = sync_entries.and_then(|s| s.first().copied()).unwrap_or(1);
        let target_ticks =
            seek_target_on_decode_clock(target_ticks, shift, ctts_offset_of(ctts(), first_sync));

        let (sample, _) = stts_locate(stts(), target_ticks).ok_or_else(|| {
            Error::Config(format!("Track {} has no samples to seek in", track_id))
        })?;

        let (landing, is_sync) = match snap {
            SeekSnap::Before => snap_to_sync(sync_entries, sample),
            SeekSnap::After => snap_to_sync_after(sync_entries, sample),
            SeekSnap::Nearest => {
                let before = snap_to_sync(sync_entries, sample);
                let after = snap_to_sync_after(sync_entries, sample);
                if before.0 == after.0 {
                    before
                } else {
                    // Compare distances in time, not sample indices — GOP
                    // lengths vary. Ties break toward Before.
                    let dist = |s: u32| {
                        stts_time_of(stts(), s)
                            .map(|t| t.abs_diff(target_ticks))
                            .unwrap_or(u64::MAX)
                    };
                    if dist(after.0) < dist(before.0) {
                        after
                    } else {
                        before
                    }
                }
            }
        };

        // The snap changed the index, so recompute its start time — in
        // *presentation* time (composition time less the edit-list shift),
        // because that is the clock `read_sample` stamps its buffers on and
        // therefore the only one the rest of the pipeline ever sees. See
        // `ctts_offset_of` and `sample_times_ns`.
        let landing_dts = stts_time_of(stts(), landing).unwrap_or(0) as i64;
        let landing_ticks = landing_dts
            .saturating_add(ctts_offset_of(ctts(), landing))
            .saturating_sub(shift)
            .max(0) as u64;
        let time_ns = (landing_ticks as u128 * 1_000_000_000 / timescale as u128) as u64;

        self.sample_indices.insert(track_id, landing);

        Ok(Mp4SeekPoint {
            track_id,
            sample_index: landing,
            time_ns,
            is_sync,
        })
    }

    /// Seek every track to a consistent position around `target_ns`.
    ///
    /// The target is first resolved on the video track with the caller's
    /// `snap` direction (keyframe snap via
    /// [`seek_to_time`](Self::seek_to_time)); every other track is then
    /// sought to the video's landing time with `SeekSnap::Before` — audio
    /// must start at or just before the video keyframe or playback opens
    /// with an audio gap, so the snap direction only steers the *reference*
    /// track. Files without a video track resolve the target on the first
    /// track instead.
    ///
    /// Returns the seek point of the reference track.
    pub fn seek_all_to_time(&mut self, target_ns: u64, snap: SeekSnap) -> Result<Mp4SeekPoint> {
        let reference = self
            .video_track_id()
            .or_else(|| self.tracks.first().map(|t| t.id))
            .ok_or_else(|| Error::Config("MP4 has no tracks to seek".into()))?;

        let point = self.seek_to_time(reference, target_ns, snap)?;

        let other_ids: Vec<u32> = self
            .tracks
            .iter()
            .map(|t| t.id)
            .filter(|id| *id != reference)
            .collect();
        for id in other_ids {
            self.seek_to_time(id, point.time_ns, SeekSnap::Before)?;
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
        // Plain-Copy per-track state only — cloning the whole Mp4Track
        // re-allocated its config Vecs on every sample (#144).
        let (sample_count, timescale, codec, track_type, adts_fields) = {
            let t = self
                .tracks
                .iter()
                .find(|t| t.id == track_id)
                .ok_or_else(|| Error::Config(format!("Track {} not found", track_id)))?;
            let adts = t
                .audio_info
                .as_ref()
                .and_then(|i| Some((i.object_type?, i.freq_index?, i.channel_config?)));
            (t.sample_count, t.timescale, t.codec, t.track_type, adts)
        };

        let sample_index = *self.sample_indices.get(&track_id).unwrap_or(&1);

        if sample_index > sample_count {
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

        // Build the outgoing payload directly in the arena slot (#144):
        // H.264 converts to a self-contained Annex-B access unit (exact size
        // via a length-prefix pre-pass), ADTS-wrapped AAC gets its 7-byte
        // header prepended, everything else copies through once.
        use crate::codec::annexb;
        let handle = if let Some(cfg) = self.nal_configs.get(&track_id) {
            let params_len: usize = if sample.is_sync {
                cfg.params.iter().map(|n| 4 + n.len()).sum()
            } else {
                0
            };
            let total = params_len + annexb::annex_b_len(&sample.bytes, cfg.length_size)?;
            let mut slot = self.output.acquire(total, "mp4demux")?;
            {
                let out = &mut slot.data_mut()[..total];
                let mut pos = 0usize;
                if sample.is_sync {
                    for nal in &cfg.params {
                        out[pos..pos + 4].copy_from_slice(&[0, 0, 0, 1]);
                        out[pos + 4..pos + 4 + nal.len()].copy_from_slice(nal);
                        pos += 4 + nal.len();
                    }
                }
                annexb::avcc_to_annex_b_into_slice(
                    &sample.bytes,
                    cfg.length_size,
                    &mut out[pos..],
                )?;
            }
            MemoryHandle::with_len(slot, total)
        } else {
            let adts_header = if self.adts_aac && codec == Mp4Codec::Aac {
                adts_fields.and_then(|(ot, fi, cc)| {
                    Mp4AudioInfo::adts_header_from(ot, fi, cc, sample.bytes.len())
                })
            } else {
                None
            };
            let header = adts_header.as_ref().map(|h| &h[..]).unwrap_or(&[]);
            let total = header.len() + sample.bytes.len();
            let mut slot = self.output.acquire(total, "mp4demux")?;
            {
                let out = &mut slot.data_mut()[..total];
                out[..header.len()].copy_from_slice(header);
                out[header.len()..].copy_from_slice(&sample.bytes);
            }
            MemoryHandle::with_len(slot, total)
        };
        // Presentation time for the PTS, decode time for the DTS (#304) —
        // once, for both the buffer's metadata and the returned sample.
        let shift = self.edit_shifts.get(&track_id).copied().unwrap_or(0);
        let times = sample_times_ns(sample.start_time, sample.rendering_offset, shift, timescale);
        let buffer = self.finish_buffer(handle, track_id, &sample, times, timescale);
        let (pts_ns, dts_ns) = times.unwrap_or((0, 0));

        let timescale = timescale as u128;
        let duration_ns = (sample.duration as u128 * 1_000_000_000)
            .checked_div(timescale)
            .unwrap_or(0) as u64;

        // Update statistics
        self.stats.samples_read += 1;
        self.stats.bytes_read += sample.bytes.len() as u64;

        if track_type == Mp4TrackType::Video {
            self.stats.video_samples += 1;
        } else if track_type == Mp4TrackType::Audio {
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

    /// Wrap an already-written arena handle with the sample's metadata.
    ///
    /// `times` is the sample's `(pts, dts)` from `sample_times_ns`.
    fn finish_buffer(
        &mut self,
        handle: MemoryHandle,
        track_id: u32,
        sample: &mp4::Mp4Sample,
        times: Option<(u64, u64)>,
        timescale: u32,
    ) -> Buffer {
        let timescale = timescale as u128;

        let mut metadata = Metadata::new();
        metadata.stream_id = track_id;

        if let Some((pts, dts)) = times {
            metadata.pts = ClockTime::from_nanos(pts);
            metadata.dts = ClockTime::from_nanos(dts);

            metadata.duration =
                ClockTime::from_nanos((sample.duration as u128 * 1_000_000_000 / timescale) as u64);
        }

        if sample.is_sync {
            metadata.flags |= BufferFlags::SYNC_POINT;
        }

        // Say what this is (#238) — the codec and the geometry were both
        // parsed out of the track header and then discarded, so a
        // `mp4demux ! zenohsink` published H.264 as `zenoh/bytes`.
        if let Some(track) = self.tracks.iter().find(|t| t.id == track_id) {
            if let Some(codec) = track.codec.video_codec() {
                match &track.video_info {
                    Some(v) if v.width > 0 && v.height > 0 => {
                        metadata.set_coded_video_dims(codec, v.width, v.height)
                    }
                    _ => metadata.format = Some(crate::format::MediaFormat::Video(codec)),
                }
                // What the sample entry's `colr` box said (#265). It rides the
                // *encoded* buffer, so each decoder carries it across onto the
                // frame it produces — the demuxer never emits raw video.
                if let Some(color) = track.video_info.as_ref().and_then(|v| v.color) {
                    metadata.set_color_info(color);
                }
                // Non-square pixels and rotation (#266) ride the encoded
                // buffer alongside the colour, and reach the display the same
                // way — via the decoder's inherited metadata.
                if let Some(display) = track.video_info.as_ref().and_then(|v| v.display) {
                    metadata.set_display_geometry(display);
                }
            } else if let Some(codec) = track.codec.audio_codec() {
                metadata.format = Some(crate::format::MediaFormat::Audio(codec));
            }
        }

        Buffer::new(handle, metadata)
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

// ============================================================================
// Mp4DemuxSource (pipeline Demuxer)
// ============================================================================

/// [`Mp4Demux`] as a *source-style* pipeline element with routed A/V pads.
///
/// The demuxer owns its reader (MP4 needs `Seek` over the whole file, so it
/// cannot be fed chunks), which makes it the pipeline's source — add it with
/// [`add_demuxer`](crate::pipeline::Pipeline::add_demuxer) and give it no
/// input link:
///
/// ```rust,ignore
/// let file = std::fs::File::open("movie.mp4")?;
/// let size = file.metadata()?.len();
/// let demux = Mp4Demux::new(std::io::BufReader::new(file), size)?;
///
/// let node = pipeline.add_demuxer("mp4demux", Mp4DemuxSource::new(demux));
/// pipeline.link_pads(node, "video", video_branch, "sink")?;
/// pipeline.link_pads(node, "audio", audio_branch, "sink")?;
/// ```
///
/// Samples are emitted in decode order across the selected tracks (smallest
/// DTS first), so audio and video interleave the way the file laid them out.
/// The first video track routes to the `"video"` pad and the first audio
/// track to `"audio"`; further tracks are ignored. Buffer metadata carries
/// pts/dts/duration, the keyframe flag, and the track id as `stream_id`, and
/// H.264 keyframes arrive as self-contained Annex-B — exactly what the
/// app-driven feeder thread used to assemble by hand.
pub struct Mp4DemuxSource<R: Read + Seek + Send> {
    demux: Mp4Demux<R>,
    outputs: Vec<(crate::element::PadId, crate::format::Caps)>,
    /// `(track_id, pad)` for each routed track.
    tracks: Vec<(u32, crate::element::PadId)>,
    /// Where to report what this file holds (#278).
    bus: Option<crate::pipeline::bus::BusHandle>,
    /// Whether the stream collection has gone out.
    reported: bool,
    /// Read-ahead of one sample per track, for DTS-ordered interleaving.
    pending: Vec<Option<Mp4Sample>>,
    primed: bool,
    /// Restart from the top at EOS instead of ending (see [`with_loop`](Self::with_loop)).
    looping: bool,
    /// A Segment staged for the executor to emit on every pad before the
    /// next buffer (#265). The `with_loop` rewind is not a seek, so nothing
    /// else re-establishes the timeline at the lap boundary.
    staged_segment: Option<crate::event::SegmentEvent>,
    /// Trick-play rate from the last handled seek (#165): while > 1.0,
    /// only video keyframes are emitted (audio muted) — fast-forward as
    /// GStreamer's keyframe trick mode. A rate-1.0 seek restores normal
    /// playback. PTS untouched: the post-seek Segment carries the rate
    /// and sinks pace by segment running time.
    trick_rate: f64,
    /// Playback stop from the last handled seek's `stop` (#165): a track
    /// whose next sample's PTS is at or past this counts as exhausted, so
    /// playback really ends at `stop` (the executor turns the resulting Eos
    /// into `SegmentDone` for a SEGMENT-flagged seek). Cleared by a seek
    /// with no stop.
    stop_ns: Option<u64>,
    /// Reverse playback cursor (#165, `trick_rate < 0`): the next backward
    /// keyframe probe position. `None` while reverse is active means the
    /// walk passed the range's bottom — Eos. Keyframe-only, video-only
    /// (GStreamer's keyframe reverse trick mode); PTS stay true (decreasing)
    /// and the reverse segment maps them to increasing running time.
    reverse_cursor_ns: Option<u64>,
    /// Bottom of the reverse range (the seek's `start`).
    reverse_end_ns: u64,
}

impl<R: Read + Seek + Send> Mp4DemuxSource<R> {
    /// Wrap an opened [`Mp4Demux`], selecting its first video and first
    /// audio track.
    pub fn new(demux: Mp4Demux<R>) -> Self {
        use crate::element::PadId;
        use crate::format::Caps;
        let mut outputs = Vec::new();
        let mut tracks = Vec::new();
        if let Some(id) = demux.video_track_id() {
            outputs.push((PadId(0), Caps::any()));
            tracks.push((id, PadId(0)));
        }
        if let Some(id) = demux.audio_track_id() {
            outputs.push((PadId(1), Caps::any()));
            tracks.push((id, PadId(1)));
        }
        let pending = (0..tracks.len()).map(|_| None).collect();
        Self {
            demux,
            outputs,
            tracks,
            bus: None,
            reported: false,
            pending,
            primed: false,
            looping: false,
            staged_segment: None,
            trick_rate: 1.0,
            stop_ns: None,
            reverse_cursor_ns: None,
            reverse_end_ns: 0,
        }
    }

    /// Wrap an opened [`Mp4Demux`], selecting only its first video track.
    ///
    /// The audio pad never exists, so nothing is routed to it — for players
    /// running without an audio branch, this beats leaving the pad unlinked
    /// (which drops each audio sample with a rate-limited warning).
    pub fn video_only(demux: Mp4Demux<R>) -> Self {
        Self::new(demux).drop_audio()
    }

    /// Drop the audio pad, keeping whatever video track is selected.
    ///
    /// Distinct from `with_tracks(video, None)`, which means "take the
    /// container's default audio track" — the difference between *no* audio
    /// and *the usual* audio, and getting it wrong leaves an unlinked pad
    /// dropping every audio buffer it is handed.
    pub fn drop_audio(mut self) -> Self {
        self.tracks.retain(|(_, pad)| pad.0 == 0);
        self.outputs.retain(|(pad, _)| pad.0 == 0);
        self.pending.truncate(self.tracks.len());
        self
    }

    /// Route specific tracks instead of the first of each type (#278).
    ///
    /// `None` keeps the default pick. An id that is not in the file, or names
    /// a track of the wrong type, is an error rather than a silent fall-back.
    ///
    /// Construction-time by design: switching a track mid-stream would mean a
    /// different decoder element, and the graph is fixed at `start()`.
    pub fn with_tracks(mut self, video: Option<u32>, audio: Option<u32>) -> Result<Self> {
        use crate::element::PadId;
        use crate::format::Caps;

        let video = match video {
            Some(id) => Some(self.require_track(id, Mp4TrackType::Video)?),
            None => self.demux.video_track_id(),
        };
        let audio = match audio {
            Some(id) => Some(self.require_track(id, Mp4TrackType::Audio)?),
            None => self.demux.audio_track_id(),
        };

        self.outputs.clear();
        self.tracks.clear();
        if let Some(id) = video {
            self.outputs.push((PadId(0), Caps::any()));
            self.tracks.push((id, PadId(0)));
        }
        if let Some(id) = audio {
            self.outputs.push((PadId(1), Caps::any()));
            self.tracks.push((id, PadId(1)));
        }
        self.pending = (0..self.tracks.len()).map(|_| None).collect();
        self.primed = false;
        Ok(self)
    }

    /// Check that `id` names a track of `kind`, and return it.
    fn require_track(&self, id: u32, kind: Mp4TrackType) -> Result<u32> {
        match self.demux.track(id) {
            Some(track) if track.track_type == kind => Ok(id),
            Some(track) => Err(Error::Config(format!(
                "MP4 track {id} is {:?}, not {kind:?}",
                track.track_type
            ))),
            None => Err(Error::Config(format!("MP4 has no track {id}"))),
        }
    }

    /// Tell the bus what this file holds, once (#278).
    ///
    /// From `produce` rather than from `set_bus`: the executor hands the bus
    /// over before the pipeline is running, and a message posted then has
    /// nobody listening — the application takes the bus off `start()`.
    fn report_streams(&mut self) {
        if self.reported {
            return;
        }
        self.reported = true;
        let Some(bus) = self.bus.clone() else {
            return;
        };
        let reports = self
            .demux
            .tracks()
            .iter()
            .map(|track| super::StreamReport {
                id: format!("mp4:{}", track.id),
                kind: match track.track_type {
                    Mp4TrackType::Video => crate::element::muxer::StreamType::Video,
                    Mp4TrackType::Audio => crate::element::muxer::StreamType::Audio,
                    Mp4TrackType::Subtitle => crate::element::muxer::StreamType::Subtitle,
                    Mp4TrackType::Unknown => crate::element::muxer::StreamType::Data,
                },
                label: track.label.clone(),
                caps: track.codec.to_string(),
                selected: self.tracks.iter().any(|(id, _)| *id == track.id),
            })
            .collect();
        super::post_streams(
            &bus,
            "mp4",
            crate::clock::ClockTime::from_nanos(self.demux.duration_ns()),
            reports,
        );
    }

    /// Loop playback: at end of stream, seek back to the start and keep
    /// producing instead of reporting EOS.
    ///
    /// This is the only way to loop a windowed player in-process — winit
    /// allows one event loop per process ever, so tearing the pipeline down
    /// and rebuilding it cannot reopen the window. The stream then never
    /// ends on its own; stop it with [`PipelineHandle::stop`].
    ///
    /// The rewind emits a fresh `Segment` on every pad (#265), so running
    /// time stays monotonic across the lap boundary and paced sinks keep
    /// A/V locked through it — but it is not a flush, so anything already
    /// queued downstream plays out first. For a lap boundary the pipeline
    /// treats as a real segment end, drive a
    /// [`SeekFlags::SEGMENT`](crate::event::SeekFlags::SEGMENT) seek loop
    /// from the bus instead (respond to `MessageKind::SegmentDone` with a
    /// non-flushing SEGMENT seek back to the start, #165).
    ///
    /// [`PipelineHandle::stop`]: crate::pipeline::PipelineHandle::stop
    pub fn with_loop(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// The wrapped demuxer (e.g. for `seek_all_to_time`).
    pub fn demux_mut(&mut self) -> &mut Mp4Demux<R> {
        &mut self.demux
    }

    /// Refill the read-ahead slot for track index `i` if it is empty.
    fn refill(&mut self, i: usize) -> Result<()> {
        if self.pending[i].is_none() {
            self.pending[i] = self.demux.read_sample(self.tracks[i].0)?;
        }
        Ok(())
    }

    /// Reverse trick mode (#165): walk video keyframes backward from the
    /// cursor. Each call snaps the cursor to the previous sync sample,
    /// emits it, and moves the cursor just below it; passing the range's
    /// bottom (or keyframe 0) ends the walk. Audio is muted — its samples
    /// are never read.
    fn produce_reverse(&mut self) -> Result<crate::element::DemuxerProduce> {
        use crate::element::{DemuxerProduce, RoutedOutput};

        let Some((video_track, video_pad)) =
            self.tracks.iter().find(|(_, pad)| pad.0 == 0).copied()
        else {
            return Ok(DemuxerProduce::Eos);
        };
        let Some(cursor) = self.reverse_cursor_ns else {
            return Ok(DemuxerProduce::Eos);
        };
        let point = self.demux.seek_all_to_time(cursor, SeekSnap::Before)?;
        let k = point.time_ns;
        if k < self.reverse_end_ns {
            self.reverse_cursor_ns = None;
            return Ok(DemuxerProduce::Eos);
        }
        let Some(sample) = self.demux.read_sample(video_track)? else {
            self.reverse_cursor_ns = None;
            return Ok(DemuxerProduce::Eos);
        };
        // Strictly decreasing: the next probe sits below this keyframe, so
        // SeekSnap::Before cannot land on it again — termination guaranteed.
        self.reverse_cursor_ns = (k > self.reverse_end_ns && k > 0).then(|| k - 1);
        Ok(DemuxerProduce::Routed(RoutedOutput::single(
            video_pad,
            sample.buffer,
        )))
    }
}

impl<R: Read + Seek + Send> crate::element::Demuxer for Mp4DemuxSource<R> {
    fn set_bus(&mut self, bus: crate::pipeline::bus::BusHandle) {
        self.bus = Some(bus);
    }

    fn take_downstream_event(&mut self) -> Option<crate::event::Event> {
        self.staged_segment.take().map(crate::event::Event::Segment)
    }

    fn demux(&mut self, _buffer: Buffer) -> Result<crate::element::RoutedOutput> {
        Err(Error::Config(
            "Mp4DemuxSource owns its reader; add it with no input link".into(),
        ))
    }

    fn produce(&mut self) -> Result<crate::element::DemuxerProduce> {
        use crate::element::{DemuxerProduce, RoutedOutput};

        self.report_streams();

        if self.trick_rate < 0.0 {
            return self.produce_reverse();
        }

        if !self.primed {
            for i in 0..self.tracks.len() {
                self.refill(i)?;
            }
            self.primed = true;
        }

        // Fast-forward (#165): drop pending samples that are not video
        // keyframes before interleaving, refilling as we go. Reads still
        // touch every sample (a stss-walk fast path is a later slice);
        // only decode-side work is saved, which is the expensive part.
        if self.trick_rate > 1.0 {
            for i in 0..self.tracks.len() {
                loop {
                    match &self.pending[i] {
                        Some(s) if self.tracks[i].1.0 != 0 || !s.is_keyframe => {
                            self.pending[i] = None;
                            self.refill(i)?;
                        }
                        _ => break,
                    }
                }
            }
        }

        // Emit the pending sample with the smallest DTS. A sample at or past
        // the seek's stop counts its track as exhausted (#165) — playback
        // ends at `stop`, not at the file's end.
        let past_stop = |s: &Mp4Sample| {
            self.stop_ns
                .is_some_and(|stop| s.buffer.metadata().pts.nanos() >= stop)
        };
        let next = self
            .pending
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|s| (i, s)))
            .filter(|(_, s)| !past_stop(s))
            .min_by_key(|(_, s)| s.dts_ns)
            .map(|(i, _)| i);

        let Some(i) = next else {
            if self.looping {
                // Rewind and keep going — one keyframe-snapped seek to zero,
                // then re-prime the read-ahead on the next call.
                self.demux.seek_all_to_time(0, SeekSnap::Before)?;
                self.primed = false;
                // Re-establish the timeline for the new lap: PTS restart at
                // zero, running time must not. The executor fills in the
                // base already consumed (#265).
                self.staged_segment = Some(crate::event::SegmentEvent::new_time(
                    crate::clock::ClockTime::ZERO,
                    None,
                ));
                return self.produce();
            }
            return Ok(DemuxerProduce::Eos);
        };
        let sample = self.pending[i].take().expect("selected a pending sample");
        self.refill(i)?;

        Ok(DemuxerProduce::Routed(RoutedOutput::single(
            self.tracks[i].1,
            sample.buffer,
        )))
    }

    fn pad_name(&self, pad: crate::element::PadId) -> String {
        // Exhaustive on purpose: the catch-all used to be `_ => "audio"`, so
        // a third pad would silently collide with the audio one (#279).
        match pad.0 {
            0 => "video".into(),
            1 => "audio".into(),
            _ => "subtitle".into(),
        }
    }

    fn handle_upstream_event(&mut self, event: &crate::event::Event) -> crate::event::EventResult {
        use crate::event::{Event, EventResult, SeekType, SegmentFormat};

        // PipelineHandle::seek reaches a source-style demuxer here: land all
        // tracks on a consistent keyframe-snapped position and drop the
        // read-ahead so the next produce() starts from the seek point. The
        // executor then runs FlushStart → FlushStop → Segment on every pad.
        let Event::Seek(seek) = event else {
            return EventResult::NotHandled;
        };
        if seek.format != SegmentFormat::Time || seek.start.seek_type != SeekType::Set {
            return EventResult::NotHandled;
        }

        // Reverse (#165): keyframe-only backward walk over [start, stop).
        // The reported landing is the TOP of the range — where playback
        // begins; the executor builds the segment as [start, stop] with the
        // seek's negative rate.
        if seek.rate < 0.0 {
            let start_ns = seek.start.position.max(0) as u64;
            let stop_ns = match seek.stop.seek_type {
                SeekType::Set => seek.stop.position.max(0) as u64,
                _ => {
                    let d = self.demux.duration_ns();
                    if d == 0 {
                        tracing::warn!(
                            "mp4demux: reverse seek needs a stop or a known duration; refused"
                        );
                        return EventResult::Error;
                    }
                    d
                }
            };
            if stop_ns <= start_ns {
                tracing::warn!("mp4demux: reverse seek range is empty ({start_ns}..{stop_ns})");
                return EventResult::Error;
            }
            for slot in &mut self.pending {
                *slot = None;
            }
            self.staged_segment = None;
            self.primed = true; // reverse path bypasses forward priming
            self.trick_rate = seek.rate;
            self.stop_ns = None;
            self.reverse_end_ns = start_ns;
            self.reverse_cursor_ns = Some(stop_ns - 1);
            tracing::info!(
                "mp4demux: reverse seek over [{start_ns}, {stop_ns}) ns, keyframes only"
            );
            return EventResult::handled_at(stop_ns as i64);
        }

        let snap = seek.flags.snap().unwrap_or(SeekSnap::Before);
        match self
            .demux
            .seek_all_to_time(seek.start.position.max(0) as u64, snap)
        {
            Ok(point) => {
                tracing::info!(
                    "mp4demux: seek to {} ns landed at {} ns (keyframe, {snap:?})",
                    seek.start.position,
                    point.time_ns
                );
                for slot in &mut self.pending {
                    *slot = None;
                }
                self.primed = false;
                // A loop rewind may have staged a Segment that this seek just
                // superseded; emitting it afterwards would re-anchor both pads
                // at zero on top of the seek's own segment (#265).
                self.staged_segment = None;
                // Trick play (#165): the seek's rate governs playback until
                // the next handled seek changes it (rate 0 is nonsense —
                // treat as normal). A forward seek always leaves reverse.
                self.trick_rate = if seek.rate > 0.0 { seek.rate } else { 1.0 };
                self.reverse_cursor_ns = None;
                // Honor the seek's stop (#165): playback ends there. A seek
                // without a stop clears any previous one.
                self.stop_ns = match seek.stop.seek_type {
                    SeekType::Set => Some(seek.stop.position.max(0) as u64),
                    _ => None,
                };
                // The keyframe actually landed on (may be before the
                // request) — Segment/SeekDone report it (#162).
                EventResult::handled_at(point.time_ns as i64)
            }
            Err(e) => {
                tracing::warn!("mp4demux: seek failed: {e}");
                EventResult::Error
            }
        }
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        let d = self.demux.duration_ns();
        Some(crate::pipeline::seek::DurationQuery {
            format: crate::event::SegmentFormat::Time,
            // 0 from the mvhd means "unknown", not "empty file".
            duration: (d > 0).then_some(d),
        })
    }

    fn outputs(&self) -> &[(crate::element::PadId, crate::format::Caps)] {
        &self.outputs
    }

    fn on_pad_added(&mut self, _callback: crate::element::PadAddedCallback) {}

    fn name(&self) -> &str {
        "mp4demux"
    }

    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.demux.set_output_budget(budget);
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

    fn test_avc_config() -> crate::codec::annexb::NalConfig {
        crate::codec::annexb::NalConfig {
            length_size: 4,
            params: vec![vec![0x67, 0x42, 0x00, 0x1F], vec![0x68, 0xCE, 0x3C, 0x80]],
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

    /// #265: the seek tables are in decode time, the emitted timestamps in
    /// composition time, and the landing has to be reported on the second.
    #[test]
    fn ctts_offset_of_walks_the_runs() {
        // Three samples offset by two frame durations (a typical B-frame
        // reorder delay), then five with no offset.
        const CTTS: [(u32, i32); 2] = [(3, 200), (5, 0)];
        assert_eq!(ctts_offset_of(CTTS, 1), 200);
        assert_eq!(ctts_offset_of(CTTS, 3), 200);
        assert_eq!(ctts_offset_of(CTTS, 4), 0);
        assert_eq!(ctts_offset_of(CTTS, 8), 0);
    }

    #[test]
    fn a_track_without_ctts_composes_at_its_decode_time() {
        // No ctts box at all — the common case, and the one where the old
        // DTS-only arithmetic happened to be right.
        assert_eq!(ctts_offset_of([], 1), 0);
        // Past the last run: no reordering left to describe.
        assert_eq!(ctts_offset_of([(2u32, 300i32)], 9), 0);
    }

    #[test]
    fn a_negative_ctts_offset_is_honoured() {
        // Version-1 ctts allows negative offsets (composition before decode).
        assert_eq!(ctts_offset_of([(4u32, -150i32)], 2), -150);
    }

    /// #304: `start_time` is decode time; composition is decode + ctts.
    #[test]
    fn a_sample_is_presented_at_decode_time_plus_its_ctts_offset() {
        // 90 kHz, 24 fps (3750 ticks a frame): a P frame decoded at 1 s and
        // shown three frames later, as in a B-frame pyramid.
        assert_eq!(
            sample_times_ns(90_000, 11_250, 0, 90_000),
            Some((1_125_000_000, 1_000_000_000))
        );
    }

    #[test]
    fn a_sample_without_ctts_has_equal_pts_and_dts() {
        assert_eq!(
            sample_times_ns(45_000, 0, 0, 90_000),
            Some((500_000_000, 500_000_000))
        );
    }

    #[test]
    fn a_negative_composition_time_clamps_to_zero() {
        // Version-1 ctts on the first frame, with no edit list to shift it.
        assert_eq!(sample_times_ns(0, -3_750, 0, 90_000), Some((0, 0)));
        assert_eq!(
            sample_times_ns(3_750, -3_750, 0, 90_000),
            Some((0, 41_666_666))
        );
    }

    /// The edit list is what puts an x264 file's first frame at zero: ffmpeg
    /// writes the first frame's ctts offset as the first edit's media time.
    #[test]
    fn the_edit_shift_brings_the_first_frame_to_zero() {
        // 15360 Hz, 30 fps (512 ticks a frame), two frames of reorder delay.
        assert_eq!(sample_times_ns(0, 1024, 1024, 15_360), Some((0, 0)));
        // The next frame decoded is a P frame shown third: composition 2560,
        // presented at 1536 (100 ms). Its decode time clamps to zero.
        assert_eq!(
            sample_times_ns(512, 2048, 1024, 15_360),
            Some((100_000_000, 0))
        );
    }

    /// What the clamp leaves a consumer to rely on. An x264-style stream:
    /// 512 ticks a frame, decoded I P B B P B B, two frames of reorder delay,
    /// and the edit list taking those two frames back off.
    #[test]
    fn under_an_edit_shift_the_dts_neither_runs_backwards_nor_passes_the_pts() {
        const FRAME: i64 = 512;
        const DELAY: i64 = 2;
        // Where each frame is shown, in the order it is decoded.
        let shown = [0i64, 3, 1, 2, 6, 4, 5];
        let mut last_dts = 0;
        for (decoded, shown) in shown.into_iter().enumerate() {
            let decoded = decoded as i64;
            let ctts = (shown - decoded + DELAY) * FRAME;
            let (pts, dts) =
                sample_times_ns((decoded * FRAME) as u64, ctts as i32, DELAY * FRAME, 15_360)
                    .unwrap();
            assert_eq!(pts, shown as u64 * 100_000_000 / 3, "frame {decoded}");
            assert!(dts <= pts, "frame {decoded}: DTS {dts} is past PTS {pts}");
            assert!(dts >= last_dts, "frame {decoded}: DTS ran backwards");
            last_dts = dts;
        }
        // The frames decoded at or before the origin are the ones that tie at
        // zero; the first one decoded after it is the first to move.
        assert_eq!(
            sample_times_ns(1024, 512, 1024, 15_360),
            Some((33_333_333, 0))
        );
        assert!(sample_times_ns(1536, 512, 1024, 15_360).unwrap().1 > 0);
    }

    /// The x264 case: the edit's media time is the keyframes' ctts offset,
    /// so a keyframe is presented at its decode time and the target is
    /// already on the decode clock.
    #[test]
    fn a_seek_target_is_unchanged_when_the_edit_cancels_the_reorder_delay() {
        assert_eq!(seek_target_on_decode_clock(15_360, 1024, 1024), 15_360);
        assert_eq!(seek_target_on_decode_clock(15_360, 0, 0), 15_360);
    }

    /// Audio delayed 100 ms by a leading empty edit (shift −4800 at 48 kHz):
    /// the sample heard at 1 s is the one decoded at 900 ms.
    #[test]
    fn a_seek_target_behind_an_empty_edit_is_looked_up_that_much_earlier() {
        assert_eq!(seek_target_on_decode_clock(48_000, -4_800, 0), 43_200);
        // Inside the silence the edit inserts, the first sample is the answer.
        assert_eq!(seek_target_on_decode_clock(2_400, -4_800, 0), 0);
        // Opus pre-skip is the other direction: 312 samples are never heard.
        assert_eq!(seek_target_on_decode_clock(48_000, 312, 0), 48_312);
    }

    /// B frames and no edit list: a keyframe is shown one reorder delay after
    /// it is decoded, so the target comes back by that delay.
    #[test]
    fn a_seek_target_on_a_reordered_track_comes_back_by_the_keyframe_offset() {
        assert_eq!(seek_target_on_decode_clock(90_000, 0, 3_750), 86_250);
        assert_eq!(seek_target_on_decode_clock(1_000, 0, 3_750), 0);
    }

    #[test]
    fn a_zero_timescale_has_no_times() {
        assert_eq!(sample_times_ns(1_000, 0, 0, 0), None);
    }

    #[test]
    fn a_media_edit_is_the_shift() {
        assert_eq!(edit_shift_ticks([(165_267, 512)], 15_360, 1_000), 512);
        assert_eq!(edit_shift_ticks([], 15_360, 1_000), 0, "no edit list");
    }

    #[test]
    fn a_leading_empty_edit_delays_the_presentation() {
        // 100 ms of nothing (movie timescale 1000), then the media from its
        // start: the track's samples are presented 100 ms late.
        assert_eq!(
            edit_shift_ticks([(100, -1), (145_074, 0)], 48_000, 1_000),
            -4_800
        );
    }

    #[test]
    fn only_empty_edits_mean_no_shift() {
        assert_eq!(edit_shift_ticks([(100, -1)], 48_000, 1_000), 0);
        // A zero movie timescale cannot convert the delay; it is dropped.
        assert_eq!(edit_shift_ticks([(100, -1), (0, 312)], 48_000, 0), 312);
    }

    #[test]
    fn an_empty_edit_reads_as_minus_one_in_both_box_versions() {
        assert_eq!(elst_media_time(0, u32::MAX as u64), -1);
        assert_eq!(elst_media_time(1, u64::MAX), -1);
        assert_eq!(elst_media_time(0, 512), 512);
        assert_eq!(elst_media_time(1, 512), 512);
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
    fn asc_synthesis_matches_the_spec_layout() {
        // AAC-LC (2), 48 kHz (index 3), stereo (2):
        // 00010 0011 0010 000 → 0x11 0x90.
        assert_eq!(synthesize_asc(2, 3, 2), Some(vec![0x11, 0x90]));
        // AAC-LC, 44.1 kHz (index 4), mono (1): 00010 0100 0001 000.
        assert_eq!(synthesize_asc(2, 4, 1), Some(vec![0x12, 0x08]));
    }

    #[test]
    fn asc_synthesis_refuses_escape_encodings() {
        assert_eq!(synthesize_asc(31, 3, 2), None, "object-type escape");
        assert_eq!(synthesize_asc(2, 15, 2), None, "explicit frequency");
        assert_eq!(synthesize_asc(0, 3, 2), None, "null object type");
    }

    #[test]
    fn adts_header_round_trips_its_fields() {
        let info = Mp4AudioInfo {
            sample_rate: 48000,
            channels: 2,
            object_type: Some(2),
            freq_index: Some(3),
            channel_config: Some(2),
            audio_specific_config: synthesize_asc(2, 3, 2),
            codec_private: None,
        };
        let header = info.adts_header(100).unwrap();

        assert_eq!(header[0], 0xFF);
        assert_eq!(header[1], 0xF1, "MPEG-4, layer 0, no CRC");
        assert_eq!(header[2] >> 6, 1, "profile = object type - 1");
        assert_eq!((header[2] >> 2) & 0xF, 3, "frequency index");
        let chan = ((header[2] & 0x1) << 2) | (header[3] >> 6);
        assert_eq!(chan, 2, "channel configuration");
        let len =
            ((header[3] as u16 & 0x3) << 11) | ((header[4] as u16) << 3) | (header[5] as u16 >> 5);
        assert_eq!(len, 107, "13-bit length includes the header");
    }

    #[test]
    fn adts_header_needs_esds_and_a_representable_length() {
        let no_esds = Mp4AudioInfo {
            sample_rate: 44100,
            channels: 2,
            object_type: None,
            freq_index: None,
            channel_config: None,
            audio_specific_config: None,
            codec_private: None,
        };
        assert_eq!(no_esds.adts_header(100), None);

        let info = Mp4AudioInfo {
            sample_rate: 48000,
            channels: 2,
            object_type: Some(2),
            freq_index: Some(3),
            channel_config: Some(2),
            audio_specific_config: None,
            codec_private: None,
        };
        assert!(info.adts_header(0x1FFF - 6).is_none(), "overflows 13 bits");
        assert!(info.adts_header(0x1FFF - 7).is_some());
    }

    #[test]
    fn snap_to_sync_before_first_keyframe_goes_forward() {
        // Pathological file whose first sync sample is not sample 1: nothing
        // before it is decodable, so the seek moves forward.
        let stss = [4u32, 8];
        assert_eq!(snap_to_sync(Some(&stss), 2), (4, true));
    }

    #[test]
    fn snap_to_sync_after_snaps_forward() {
        let stss = [1u32, 6, 11];
        assert_eq!(
            snap_to_sync_after(Some(&stss), 6),
            (6, true),
            "already a keyframe"
        );
        assert_eq!(
            snap_to_sync_after(Some(&stss), 9),
            (11, true),
            "snaps forward"
        );
        assert_eq!(
            snap_to_sync_after(Some(&stss), 15),
            (11, true),
            "past the last keyframe clamps back"
        );
        assert_eq!(
            snap_to_sync_after(Some(&stss), 2),
            (6, true),
            "between the first two"
        );
        assert_eq!(
            snap_to_sync_after(None, 7),
            (7, true),
            "no stss: everything is sync"
        );
        assert_eq!(
            snap_to_sync_after(Some(&[]), 7),
            (7, false),
            "empty stss: nothing to snap to"
        );
    }
}
