//! Matroska/WebM container demuxer.
//!
//! Extracts elementary streams from Matroska (`.mkv`) and WebM (`.webm`)
//! files, backed by the pure-Rust `matroska-demuxer` crate.
//!
//! # Supported codecs
//!
//! | Type | Codec ID | Notes |
//! |------|----------|-------|
//! | Video | `V_MPEG4/ISO/AVC` | Blocks converted to Annex-B with in-band SPS/PPS on keyframes |
//! | Video | `V_VP8`, `V_VP9`, `V_AV1`, `V_MPEGH/ISO/HEVC` | Pass through as stored |
//! | Audio | `A_AAC*` | CodecPrivate carries the real AudioSpecificConfig |
//! | Audio | `A_OPUS`, `A_VORBIS`, `A_EAC3`, `A_AC3`, `A_MPEG/L3` | Pass through as stored |
//!
//! Subtitle and unrecognized tracks are listed in [`MkvDemux::tracks`] but
//! never routed to a pad — a player can show them without choking on them.
//!
//! Unlike MP4 (per-track sample cursors), Matroska stores frames interleaved
//! in cluster order, so [`MkvDemux`] is itself the source-style
//! [`Demuxer`](crate::element::Demuxer): add it with
//! [`add_demuxer`](crate::pipeline::Pipeline::add_demuxer) and no input link.
//!
//! ```rust,ignore
//! use parallax::elements::MkvDemux;
//!
//! let file = std::fs::File::open("movie.mkv")?;
//! let demux = MkvDemux::new(std::io::BufReader::new(file))?;
//! let node = pipeline.add_demuxer("mkvdemux", demux);
//! pipeline.link_pads(node, "video", decode, "sink")?;
//! pipeline.link_pads(node, "audio", audio_decode, "sink")?;
//! ```
//!
//! Matroska blocks carry presentation timestamps only, so `dts == pts` on
//! every buffer. H.264 WEB-DL streams mux in decode order with B-frame
//! reordering expressed purely through the PTS sequence; downstream decoders
//! that reorder (openh264 does) handle this.

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::codec::annexb::{NalConfig, parse_avcc, parse_hvcc};
use crate::error::{Error, Result};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::{BufferFlags, Metadata};

use matroska_demuxer::{ContentCompAlgo, ContentEncodingValue, Frame, MatroskaFile, TrackType};
use std::io::{Read, Seek};

// ============================================================================
// Codec / track types
// ============================================================================

/// Codec of a Matroska track, mapped from its codec ID string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MkvCodec {
    /// H.264/AVC (`V_MPEG4/ISO/AVC`).
    H264,
    /// H.265/HEVC (`V_MPEGH/ISO/HEVC`).
    H265,
    /// VP8 (`V_VP8`).
    Vp8,
    /// VP9 (`V_VP9`).
    Vp9,
    /// AV1 (`V_AV1`).
    Av1,
    /// AAC (`A_AAC` and its object-type suffixed forms).
    Aac,
    /// Opus (`A_OPUS`).
    Opus,
    /// Vorbis (`A_VORBIS`).
    Vorbis,
    /// E-AC-3 / Dolby Digital Plus (`A_EAC3`).
    Eac3,
    /// AC-3 / Dolby Digital (`A_AC3`).
    Ac3,
    /// MPEG audio layer 3 (`A_MPEG/L3`).
    Mp3,
    /// Any `S_*` subtitle codec.
    Subtitle,
    /// Unrecognized codec ID (kept verbatim in [`MkvTrack::codec_id`]).
    Unknown,
}

impl MkvCodec {
    fn from_codec_id(id: &str) -> Self {
        match id {
            "V_MPEG4/ISO/AVC" => MkvCodec::H264,
            "V_MPEGH/ISO/HEVC" => MkvCodec::H265,
            "V_VP8" => MkvCodec::Vp8,
            "V_VP9" => MkvCodec::Vp9,
            "V_AV1" => MkvCodec::Av1,
            "A_OPUS" => MkvCodec::Opus,
            "A_VORBIS" => MkvCodec::Vorbis,
            "A_EAC3" => MkvCodec::Eac3,
            "A_AC3" => MkvCodec::Ac3,
            "A_MPEG/L3" => MkvCodec::Mp3,
            _ if id.starts_with("A_AAC") => MkvCodec::Aac,
            _ if id.starts_with("S_") => MkvCodec::Subtitle,
            _ => MkvCodec::Unknown,
        }
    }

    /// Returns true if this is a video codec.
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            MkvCodec::H264 | MkvCodec::H265 | MkvCodec::Vp8 | MkvCodec::Vp9 | MkvCodec::Av1
        )
    }

    /// Returns true if this is an audio codec.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            MkvCodec::Aac
                | MkvCodec::Opus
                | MkvCodec::Vorbis
                | MkvCodec::Eac3
                | MkvCodec::Ac3
                | MkvCodec::Mp3
        )
    }

    /// Returns true if this is a subtitle codec.
    pub fn is_subtitle(&self) -> bool {
        matches!(self, MkvCodec::Subtitle)
    }
}

impl std::fmt::Display for MkvCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MkvCodec::H264 => write!(f, "H.264/AVC"),
            MkvCodec::H265 => write!(f, "H.265/HEVC"),
            MkvCodec::Vp8 => write!(f, "VP8"),
            MkvCodec::Vp9 => write!(f, "VP9"),
            MkvCodec::Av1 => write!(f, "AV1"),
            MkvCodec::Aac => write!(f, "AAC"),
            MkvCodec::Opus => write!(f, "Opus"),
            MkvCodec::Vorbis => write!(f, "Vorbis"),
            MkvCodec::Eac3 => write!(f, "E-AC-3"),
            MkvCodec::Ac3 => write!(f, "AC-3"),
            MkvCodec::Mp3 => write!(f, "MP3"),
            MkvCodec::Subtitle => write!(f, "Subtitle"),
            MkvCodec::Unknown => write!(f, "Unknown"),
        }
    }
}

/// Type of a Matroska track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MkvTrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle track.
    Subtitle,
    /// Anything else (complex, logo, control, …).
    Unknown,
}

impl From<TrackType> for MkvTrackType {
    fn from(tt: TrackType) -> Self {
        match tt {
            TrackType::Video => MkvTrackType::Video,
            TrackType::Audio => MkvTrackType::Audio,
            TrackType::Subtitle => MkvTrackType::Subtitle,
            _ => MkvTrackType::Unknown,
        }
    }
}

/// Video parameters of a Matroska track.
#[derive(Debug, Clone)]
pub struct MkvVideoInfo {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame rate derived from the track's default frame duration, if stated.
    pub frame_rate: Option<f64>,
    /// What the track's `Colour` element said (#265), if it carried one.
    pub color: Option<crate::format::ColorInfo>,
    /// Pixel aspect ratio derived from `DisplayWidth`/`DisplayHeight` (#266).
    ///
    /// Rotation is always [`Rotation::None`](crate::format::Rotation::None)
    /// here: MKV carries it in the `Projection` element, which
    /// `matroska-demuxer` 0.8.1 does not model at all — not even an element
    /// id — so reading it would mean a raw EBML parse of our own. Out of
    /// scope for #266; `docs/formats.md` says so.
    pub display: Option<crate::format::DisplayGeometry>,
}

/// Derive the pixel aspect ratio from a track's display dimensions.
///
/// One formula covers every `DisplayUnit`: the element pair always states a
/// *display* aspect, whether it is spelled in pixels, centimetres, inches or
/// as a bare ratio, so
///
/// ```text
/// PAR = (display_width x pixel_height) : (display_height x pixel_width)
/// ```
///
/// holds throughout. `Unknown` is the one exception — the numbers mean
/// nothing without a unit, and inventing one would squash the picture on the
/// strength of a guess.
///
/// `None` when nothing was stated, or when the result is square: square
/// pixels are what a consumer assumes anyway.
fn display_par(video: &matroska_demuxer::Video) -> Option<(u32, u32)> {
    use matroska_demuxer::DisplayUnit;

    if matches!(video.display_unit(), Some(DisplayUnit::Unknown)) {
        return None;
    }
    par_from_display(
        video.display_width()?.get(),
        video.display_height()?.get(),
        video.pixel_width().get(),
        video.pixel_height().get(),
    )
}

/// The arithmetic half of [`display_par`], split out because
/// `matroska_demuxer::Video` cannot be constructed outside its crate — so
/// this is the part a test can reach.
fn par_from_display(dw: u64, dh: u64, pw: u64, ph: u64) -> Option<(u32, u32)> {
    let num = dw.checked_mul(ph)?;
    let den = dh.checked_mul(pw)?;
    if num == 0 || den == 0 || num == den {
        return None;
    }
    // Reduce, or a 3840x2160 track states its aspect as 8294400:8294400 and
    // every consumer carries two eight-digit numbers around forever.
    let divisor = gcd(num, den);
    Some((
        u32::try_from(num / divisor).ok()?,
        u32::try_from(den / divisor).ok()?,
    ))
}

/// Greatest common divisor, for reducing the aspect fraction.
fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a.max(1)
}

/// Read a track's `Colour` element into a [`ColorInfo`](crate::format::ColorInfo).
///
/// MKV codes every axis with the H.273 numbers, so this is a translation of
/// `matroska-demuxer`'s enums back to the numbers the rest of the tree uses —
/// not a second opinion about what they mean.
///
/// `None` when the element is absent *or* says nothing, so an empty `Colour`
/// (which is legal, and common) does not out-rank the height heuristic.
///
/// Deliberately not read: `max_cll`/`max_fall`. `matroska-demuxer` 0.8.1 reads
/// them from the wrong element id.
fn colour_of(video: &matroska_demuxer::Video) -> Option<crate::format::ColorInfo> {
    use crate::format::{ColorInfo, ColorMatrix, ColorPrimaries, ColorRange, TransferFunction};
    use matroska_demuxer::TransferCharacteristics as T;
    use matroska_demuxer::{MatrixCoefficients as M, Primaries as P, Range as R};

    let colour = video.colour()?;
    let info = ColorInfo {
        matrix: match colour.matrix_coefficients() {
            Some(M::Bt709) => ColorMatrix::Bt709,
            Some(M::Fcc73682 | M::Bt470bg | M::Smpte170) => ColorMatrix::Bt601,
            Some(M::Bt2020Ncl) => ColorMatrix::Bt2020Ncl,
            _ => ColorMatrix::Unspecified,
        },
        range: match colour.range() {
            // `Defined` means "read it off the matrix/transfer", which is not
            // a claim this type can hold — leave it unsignalled and let the
            // consumer's own fallback decide.
            Some(R::Broadcast) => ColorRange::Limited,
            Some(R::Full) => ColorRange::Full,
            _ => ColorRange::Unspecified,
        },
        primaries: match colour.primaries() {
            Some(P::Bt709) => ColorPrimaries::Bt709,
            Some(P::Bt601) => ColorPrimaries::Bt470bg,
            Some(P::Smpte170) => ColorPrimaries::Smpte170m,
            Some(P::Bt2020) => ColorPrimaries::Bt2020,
            _ => ColorPrimaries::Unspecified,
        },
        transfer: match colour.transfer_characteristics() {
            Some(T::Bt709) => TransferFunction::Bt709,
            Some(T::Iec61966_2_1) => TransferFunction::Srgb,
            Some(T::Bt220_10) => TransferFunction::Bt2020_10,
            Some(T::Bt2100) => TransferFunction::Pq,
            Some(T::Hlg) => TransferFunction::Hlg,
            _ => TransferFunction::Unspecified,
        },
    };
    (!info.is_unspecified()).then_some(info)
}

impl MkvCodec {
    /// The pipeline's video codec for this track, if it is video (#238).
    ///
    /// `None` for audio, subtitles, and codecs parallax has no variant for —
    /// an undeclared format beats a mislabelled one.
    fn video_codec(&self) -> Option<crate::format::VideoCodec> {
        use crate::format::VideoCodec;
        Some(match self {
            Self::H264 => VideoCodec::H264,
            Self::H265 => VideoCodec::H265,
            Self::Vp8 => VideoCodec::Vp8,
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
            Self::Vorbis => AudioCodec::Vorbis,
            Self::Eac3 => AudioCodec::Eac3,
            Self::Mp3 => AudioCodec::Mp3,
            // No `Ac3` variant in the pipeline's vocabulary; leave it
            // undeclared rather than calling it E-AC-3.
            _ => return None,
        })
    }
}

/// Audio parameters of a Matroska track.
#[derive(Debug, Clone)]
pub struct MkvAudioInfo {
    /// Output sample rate in Hz (the SBR output rate when one is declared).
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u16,
    /// Raw CodecPrivate bytes — the AudioSpecificConfig for AAC, the OpusHead
    /// for Opus, the Xiph-laced header triple for Vorbis.
    pub codec_private: Option<Vec<u8>>,
}

/// Information about one track in a Matroska file.
#[derive(Debug, Clone)]
pub struct MkvTrack {
    /// Track number (what block frames reference).
    pub id: u64,
    /// Track type.
    pub track_type: MkvTrackType,
    /// Mapped codec.
    pub codec: MkvCodec,
    /// Raw codec ID string (`V_MPEG4/ISO/AVC`, `A_OPUS`, …).
    pub codec_id: String,
    /// Video parameters (video tracks only).
    pub video_info: Option<MkvVideoInfo>,
    /// Audio parameters (audio tracks only).
    pub audio_info: Option<MkvAudioInfo>,
    /// Language, name and flags — what a picker labels a row with (#278).
    pub label: super::TrackLabel,
}

/// VP9 uncompressed-header keyframe probe (profiles 0–3).
///
/// Used only when the container stores the frame in a BlockGroup without a
/// keyframe flag; WebM normally uses SimpleBlocks where the flag is present.
fn vp9_is_keyframe(data: &[u8]) -> bool {
    let Some(&b0) = data.first() else {
        return false;
    };
    if b0 >> 6 != 0b10 {
        return false; // not a frame marker
    }
    let profile = ((b0 >> 5) & 1) | (((b0 >> 4) & 1) << 1);
    // Bit cursor after frame_marker + profile bits (profile 3 has a reserved bit).
    let mut bit = if profile == 3 { 5 } else { 4 };
    let get = |i: u32| (b0 >> (7 - i)) & 1;
    let show_existing = get(bit);
    bit += 1;
    if show_existing == 1 {
        return false;
    }
    get(bit) == 0 // frame_type: 0 = KEY_FRAME
}

// ============================================================================
// MkvDemux
// ============================================================================

/// Matroska/WebM demuxer — a source-style pipeline [`Demuxer`].
///
/// The first video track routes to the `"video"` pad and the first audio
/// track to `"audio"`; subtitle and additional tracks are listed but never
/// routed. See the module documentation for payload conversion and seek
/// behavior.
///
/// [`Demuxer`]: crate::element::Demuxer
pub struct MkvDemux<R: Read + Seek> {
    file: MatroskaFile<super::mkv_cues::SharedReader<R>>,
    /// Second handle to the reader inside `file`, for cue-indexed seeks:
    /// the crate keeps its reader private, so repositioning goes through
    /// this shared handle (state reset via `file.seek(0)` first).
    raw: super::mkv_cues::SharedReader<R>,
    /// Parallax-owned cue table, scanned before `MatroskaFile::open` — the
    /// crate's own cue path mis-resolves ffmpeg's CueRelativePosition (see
    /// `mkv_cues`). Empty for cue-less files.
    cues: Vec<super::mkv_cues::CueEntry>,
    tracks: Vec<MkvTrack>,
    /// Nanoseconds per Matroska timestamp tick.
    timestamp_scale: u64,
    outputs: Vec<(crate::element::PadId, crate::format::Caps)>,
    /// `(track_number, pad)` for each routed track.
    routed: Vec<(u64, crate::element::PadId)>,
    /// avcC/hvcC config per routed track.
    ///
    /// A map, not a single `Option`, because routing a second video track
    /// used to clobber the first one's parameter sets outright — the
    /// asymmetry with MP4's `nal_configs` was a latent bug waiting for the
    /// first caller that selected a track (#278).
    nal_configs: Vec<(u64, NalConfig)>,
    /// Bytes to prepend per frame for tracks using header-stripping
    /// compression, keyed like `routed`.
    stripped_prefix: Vec<(u64, Vec<u8>)>,
    /// Per-track default frame duration in ns (Matroska DefaultDuration).
    default_duration_ns: Vec<(u64, u64)>,
    output: OutputArena,
    /// Where to report what this file holds (#278). `None` outside a running
    /// pipeline — `tracks()` is the answer for a caller that has not started
    /// one yet, and is how the player builds its picker before `start`.
    bus: Option<crate::pipeline::bus::BusHandle>,
    /// Whether the stream collection has gone out. Once per pipeline: the
    /// track table is fixed at construction, so repeating it says nothing.
    reported: bool,
    /// Subtitle track to route, if any (#279). Off unless asked for.
    subtitle_pick: Option<u64>,
    /// Reusable frame to avoid a fresh allocation per block.
    frame: Frame,
    looping: bool,
    /// A Segment staged for the executor to emit on every pad before the
    /// next buffer (#265). The `with_loop` rewind is not a seek, so nothing
    /// else re-establishes the timeline at the lap boundary.
    staged_segment: Option<crate::event::SegmentEvent>,
    /// Trick-play rate from the last handled seek (#165): while > 1.0,
    /// only video keyframes are emitted — fast-forward as GStreamer's
    /// keyframe trick mode (audio is muted; every skipped frame would
    /// need decode work the rate makes pointless). A rate-1.0 seek
    /// restores normal playback. PTS are untouched: the post-seek
    /// Segment carries the rate and sinks pace by segment running time.
    trick_rate: f64,
    /// Drop non-keyframe video frames until the first keyframe (set after a
    /// seek so a decoder never sees a mid-GOP entry).
    video_resync: bool,
    /// Playback stop from the last handled seek's `stop`, in ticks (#165):
    /// the first routed frame at or past it ends the run (the executor
    /// turns that Eos into `SegmentDone` for a SEGMENT-flagged seek).
    /// Cleared by a seek with no stop.
    stop_ticks: Option<u64>,
    /// Reverse playback cursor (#165, `trick_rate < 0`): the next backward
    /// cue probe position in ticks. `None` while reverse is active means
    /// the walk passed the range's bottom — Eos. Cue-granular (one keyframe
    /// per cue point), video-only, true decreasing PTS; the reverse segment
    /// maps them to increasing running time. Cue-less files refuse reverse.
    reverse_cursor_ticks: Option<u64>,
    /// Bottom of the reverse range (the seek's `start`), in ticks.
    reverse_end_ticks: u64,
    /// Converted frame waiting for an output slot. Arena exhaustion is flow
    /// control (the executor retries produce()), so the frame taken from the
    /// parser must survive the retry instead of being lost.
    pending: Option<PendingFrame>,
    /// Post-seek skip, **per routed track**: `(track_number, target_ticks)`
    /// entries, each cleared only by that track's own first frame at/after
    /// the target. A single global skip used to be cleared by whichever
    /// track crossed the target first — audio interleaves ahead, so stale
    /// pre-target video leaked through (#158).
    skip: Vec<(u64, u64)>,
    /// Frames the produce() scan may discard (skip + resync) before
    /// returning an empty `Routed` to yield — the executor drains a
    /// superseding seek between produce() calls, so a long scan must not
    /// hold the task.
    scan_budget: usize,
}

/// Default [`MkvDemux::with_scan_budget`]: a whole cluster of discards per
/// produce() call, cheap enough to stay responsive.
const DEFAULT_SCAN_BUDGET: usize = 256;

/// A converted frame awaiting arena space.
struct PendingFrame {
    pad: crate::element::PadId,
    /// The raw block payload — conversion is idempotent, so the retry
    /// re-converts straight into the slot once one is free (#144).
    raw: Vec<u8>,
    track: u64,
    timestamp_ticks: u64,
    duration_ticks: Option<u64>,
    is_keyframe: bool,
}

impl<R: Read + Seek> MkvDemux<R> {
    /// Open a Matroska/WebM stream and select its first video and first
    /// audio track for routing.
    ///
    /// # Errors
    ///
    /// Returns an error if the EBML/Matroska structure cannot be parsed or
    /// a selected track uses an unsupported content encoding (encryption,
    /// zlib compression).
    pub fn new(reader: R) -> Result<Self> {
        use crate::element::PadId;
        use crate::format::Caps;

        // Own cue pre-scan on the raw reader BEFORE the crate consumes it
        // (the mp4_extra precedent), then keep a second handle for seeks.
        let mut reader = reader;
        let cues = super::mkv_cues::scan_cues(&mut reader);
        let shared = super::mkv_cues::SharedReader::new(reader);
        let raw = shared.clone();

        let file = MatroskaFile::open(shared)
            .map_err(|e| Error::Config(format!("failed to parse Matroska file: {e:?}")))?;
        let timestamp_scale = file.info().timestamp_scale().get();

        let mut tracks = Vec::new();
        for entry in file.tracks() {
            let codec_id = entry.codec_id().to_string();
            let codec = MkvCodec::from_codec_id(&codec_id);
            let track_type = MkvTrackType::from(entry.track_type());
            let video_info = entry.video().map(|v| MkvVideoInfo {
                width: v.pixel_width().get() as u32,
                height: v.pixel_height().get() as u32,
                frame_rate: entry
                    .default_duration()
                    .map(|d| 1_000_000_000.0 / d.get() as f64),
                color: colour_of(v),
                display: display_par(v).map(|par| crate::format::DisplayGeometry {
                    par,
                    rotation: crate::format::Rotation::None,
                }),
            });
            let audio_info = entry.audio().map(|a| MkvAudioInfo {
                sample_rate: a
                    .output_sampling_frequency()
                    .unwrap_or_else(|| a.sampling_frequency()) as u32,
                channels: a.channels().get() as u16,
                codec_private: entry.codec_private().map(|p| p.to_vec()),
            });
            tracks.push(MkvTrack {
                id: entry.track_number().get(),
                track_type,
                codec,
                codec_id,
                video_info,
                audio_info,
                label: super::TrackLabel {
                    // BCP-47 overrides the ISO-639-2 field when present —
                    // that is the spec's own rule, not a preference.
                    language: entry
                        .language_bcp47()
                        .or_else(|| entry.language())
                        .map(str::to_string),
                    name: entry.name().map(str::to_string),
                    default: entry.flag_default(),
                    forced: entry.flag_forced(),
                    enabled: entry.flag_enabled(),
                },
            });
        }

        let mut this = Self {
            file,
            raw,
            cues,
            tracks,
            timestamp_scale,
            outputs: Vec::new(),
            routed: Vec::new(),
            nal_configs: Vec::new(),
            stripped_prefix: Vec::new(),
            default_duration_ns: Vec::new(),
            output: OutputArena::new(defaults::MP4_DEMUX_SLOT_COUNT)
                .with_min_slot_size(defaults::DEMUX_MIN_SLOT_SIZE)
                .grow_to_fit(),
            bus: None,
            reported: false,
            subtitle_pick: None,
            frame: Frame::default(),
            looping: false,
            staged_segment: None,
            trick_rate: 1.0,
            video_resync: false,
            stop_ticks: None,
            reverse_cursor_ticks: None,
            reverse_end_ticks: 0,
            pending: None,
            skip: Vec::new(),
            scan_budget: DEFAULT_SCAN_BUDGET,
        };

        let video = this
            .tracks
            .iter()
            .find(|t| t.track_type == MkvTrackType::Video && t.codec.is_video())
            .map(|t| t.id);
        let audio = this
            .tracks
            .iter()
            .find(|t| t.track_type == MkvTrackType::Audio && t.codec.is_audio())
            .map(|t| t.id);

        if let Some(id) = video {
            this.route(id, PadId(0), Caps::any())?;
        }
        if let Some(id) = audio {
            this.route(id, PadId(1), Caps::any())?;
        }
        Ok(this)
    }

    /// Route track `id` to `pad`, lifting codec config and validating
    /// content encodings.
    ///
    /// Public since #278: a caller that has read `tracks()` and wants a
    /// different one needs a way to say so. Prefer
    /// [`with_tracks`](Self::with_tracks), which clears the previous pick's
    /// state first — routing over a live pad leaves the old track's
    /// parameter sets behind.
    pub fn route(
        &mut self,
        id: u64,
        pad: crate::element::PadId,
        caps: crate::format::Caps,
    ) -> Result<()> {
        let entry_idx = self
            .file
            .tracks()
            .iter()
            .position(|e| e.track_number().get() == id)
            .ok_or_else(|| Error::Config(format!("MKV track {id} not found")))?;
        let entry = &self.file.tracks()[entry_idx];

        // Content encodings: header stripping is reversible per frame; zlib
        // and encryption are not supported — refuse to route such a track
        // rather than emit garbage.
        if let Some(encodings) = entry.content_encodings() {
            for enc in encodings {
                match enc.encoding() {
                    ContentEncodingValue::Compression(c) => match c.algo() {
                        ContentCompAlgo::Stripping => {
                            let prefix = c.settings().map(<[u8]>::to_vec).unwrap_or_default();
                            self.stripped_prefix.push((id, prefix));
                        }
                        algo => {
                            return Err(Error::Config(format!(
                                "MKV track {id} uses unsupported compression {algo:?}"
                            )));
                        }
                    },
                    ContentEncodingValue::Encryption(_) => {
                        return Err(Error::Config(format!("MKV track {id} is encrypted")));
                    }
                    ContentEncodingValue::Unknown => {}
                }
            }
        }

        let track = self
            .tracks
            .iter()
            .find(|t| t.id == id)
            .expect("routed track exists in track table");
        let record = match track.codec {
            MkvCodec::H264 => Some(("avcC", parse_avcc as fn(&[u8]) -> Result<NalConfig>)),
            MkvCodec::H265 => Some(("hvcC", parse_hvcc as fn(&[u8]) -> Result<NalConfig>)),
            _ => None,
        };
        if let Some((name, parse)) = record {
            match entry.codec_private() {
                Some(private) => {
                    let config = parse(private)?;
                    self.nal_configs.retain(|(track, _)| *track != id);
                    self.nal_configs.push((id, config));
                }
                None => tracing::warn!(
                    "{} MKV track {id} has no {name} CodecPrivate; frames pass \
                     through length-prefixed and will not decode",
                    track.codec
                ),
            }
        }
        if let Some(d) = entry.default_duration() {
            self.default_duration_ns.push((id, d.get()));
        }

        self.routed.push((id, pad));
        self.outputs.push((pad, caps));
        Ok(())
    }

    /// Loop playback: at end of stream, seek back to the start and keep
    /// producing instead of reporting EOS.
    ///
    /// The rewind emits a fresh `Segment` on every pad (#265), so running
    /// time stays monotonic across the lap boundary and paced sinks keep
    /// A/V locked through it — but it is not a flush, so anything already
    /// queued downstream plays out first. For a lap boundary the pipeline
    /// treats as a real segment end, drive a
    /// [`SeekFlags::SEGMENT`](crate::event::SeekFlags::SEGMENT) seek loop
    /// from the bus instead (respond to `MessageKind::SegmentDone` with a
    /// non-flushing SEGMENT seek back to the start, #165).
    pub fn with_loop(mut self, looping: bool) -> Self {
        self.looping = looping;
        self
    }

    /// Cap how many frames one `produce()` call may discard while scanning
    /// toward a seek target before yielding an empty result (default 256).
    ///
    /// Yielding lets the executor drain a superseding seek between calls, so
    /// a second arrow press does not queue behind a long scan.
    pub fn with_scan_budget(mut self, budget: usize) -> Self {
        self.scan_budget = budget.max(1);
        self
    }

    /// Drop the audio pad so nothing is routed to it (for video-only players;
    /// beats leaving the pad unlinked, which warns per dropped sample).
    pub fn video_only(mut self) -> Self {
        self.routed.retain(|(_, pad)| pad.0 == 0);
        self.outputs.retain(|(pad, _)| pad.0 == 0);
        self
    }

    /// The text codec a subtitle track's id names, if parallax can draw it.
    ///
    /// `S_HDMV/PGS` and `S_VOBSUB` are **bitmap** subtitles — a completely
    /// different rendering path from timed text — so they are listed and not
    /// routed rather than routed and shown as garbage.
    pub fn text_codec(codec_id: &str) -> Option<crate::format::TextCodec> {
        use crate::format::TextCodec;
        match codec_id {
            "S_TEXT/UTF8" => Some(TextCodec::Utf8),
            "S_TEXT/ASS" | "S_TEXT/SSA" => Some(TextCodec::Ass),
            "S_TEXT/WEBVTT" => Some(TextCodec::WebVtt),
            _ => None,
        }
    }

    /// Route specific tracks instead of the first of each type (#278).
    ///
    /// `None` keeps the default pick, so `with_tracks(None, Some(3))` changes
    /// only the audio. An id that is not in the file is an error rather than
    /// a silent fall-back: a picker offered it, so something is wrong
    /// upstream if it does not exist.
    ///
    /// This is a **construction-time** choice, and deliberately so. Switching
    /// a track mid-stream is not a matter of re-pointing a pad: an AAC track
    /// and an E-AC-3 track need different decoder elements, and the graph is
    /// fixed at `start()`. An application changes tracks by building again —
    /// which is what the player does, resuming where it was.
    pub fn with_tracks(self, video: Option<u64>, audio: Option<u64>) -> Result<Self> {
        let subtitle = self.subtitle_pick;
        self.with_all_tracks(video, audio, subtitle)
    }

    /// Route a subtitle track as well (#279).
    ///
    /// `None` routes none: subtitles are off by default, which is what a
    /// player does until somebody asks for them. There is no "the
    /// container's default subtitle" here for the same reason — a file's
    /// forced-subtitle flag is a hint for a *language preference* the engine
    /// does not have.
    pub fn with_subtitle_track(mut self, subtitle: Option<u64>) -> Result<Self> {
        self.subtitle_pick = subtitle;
        let (video, audio) = self.current_picks();
        self.with_all_tracks(video, audio, subtitle)
    }

    /// The video and audio tracks currently routed.
    fn current_picks(&self) -> (Option<u64>, Option<u64>) {
        let at = |pad: u32| {
            self.routed
                .iter()
                .find(|(_, p)| p.0 == pad)
                .map(|(id, _)| *id)
        };
        (at(0), at(1))
    }

    fn with_all_tracks(
        mut self,
        video: Option<u64>,
        audio: Option<u64>,
        subtitle: Option<u64>,
    ) -> Result<Self> {
        let video = match video {
            Some(id) => Some(self.require_track(id, MkvTrackType::Video)?),
            None => self.default_track(MkvTrackType::Video),
        };
        let audio = match audio {
            Some(id) => Some(self.require_track(id, MkvTrackType::Audio)?),
            None => self.default_track(MkvTrackType::Audio),
        };
        let subtitle = match subtitle {
            Some(id) => {
                let id = self.require_track(id, MkvTrackType::Subtitle)?;
                let codec_id = self
                    .tracks
                    .iter()
                    .find(|t| t.id == id)
                    .map(|t| t.codec_id.clone())
                    .unwrap_or_default();
                if Self::text_codec(&codec_id).is_none() {
                    return Err(Error::Config(format!(
                        "MKV track {id} is {codec_id}, which is a bitmap subtitle \
                         format parallax cannot draw"
                    )));
                }
                Some(id)
            }
            None => None,
        };
        self.subtitle_pick = subtitle;

        // Everything `route` accumulated for the previous pick has to go, or
        // the old track's parameter sets and header-stripping prefixes follow
        // the new one around.
        self.routed.clear();
        self.outputs.clear();
        self.nal_configs.clear();
        self.stripped_prefix.clear();
        self.default_duration_ns.clear();

        if let Some(id) = video {
            self.route(id, crate::element::PadId(0), crate::format::Caps::any())?;
        }
        if let Some(id) = audio {
            self.route(id, crate::element::PadId(1), crate::format::Caps::any())?;
        }
        if let Some(id) = subtitle {
            self.route(id, crate::element::PadId(2), crate::format::Caps::any())?;
        }
        Ok(self)
    }

    /// Check that `id` names a track of `kind`, and return it.
    fn require_track(&self, id: u64, kind: MkvTrackType) -> Result<u64> {
        match self.tracks.iter().find(|t| t.id == id) {
            Some(track) if track.track_type == kind => Ok(id),
            Some(track) => Err(Error::Config(format!(
                "MKV track {id} is {:?}, not {kind:?}",
                track.track_type
            ))),
            None => Err(Error::Config(format!("MKV has no track {id}"))),
        }
    }

    /// The track this demuxer picks for `kind` when nobody says otherwise.
    ///
    /// The container's own default flag first — that is what it is for — then
    /// the first playable track, which is what this always did.
    fn default_track(&self, kind: MkvTrackType) -> Option<u64> {
        let playable = |t: &&MkvTrack| {
            t.track_type == kind
                && t.label.enabled
                && match kind {
                    MkvTrackType::Video => t.codec.is_video(),
                    MkvTrackType::Audio => t.codec.is_audio(),
                    _ => true,
                }
        };
        self.tracks
            .iter()
            .find(|t| playable(t) && t.label.default)
            .or_else(|| self.tracks.iter().find(playable))
            .map(|t| t.id)
    }

    /// Tell the bus what this file holds, once (#278).
    ///
    /// From `produce` rather than from `set_bus`, because the executor calls
    /// `set_bus` before the pipeline is running and a message posted then has
    /// nobody listening yet — the application takes the bus off the handle
    /// `start()` returns.
    fn report_streams(&mut self) {
        if self.reported {
            return;
        }
        self.reported = true;
        let Some(bus) = self.bus.clone() else {
            return;
        };
        let reports = self
            .tracks
            .iter()
            .map(|track| super::StreamReport {
                id: format!("mkv:{}", track.id),
                kind: match track.track_type {
                    MkvTrackType::Video => crate::element::muxer::StreamType::Video,
                    MkvTrackType::Audio => crate::element::muxer::StreamType::Audio,
                    MkvTrackType::Subtitle => crate::element::muxer::StreamType::Subtitle,
                    MkvTrackType::Unknown => crate::element::muxer::StreamType::Data,
                },
                label: track.label.clone(),
                caps: track.codec_id.clone(),
                selected: self.routed.iter().any(|(id, _)| *id == track.id),
            })
            .collect();
        super::post_streams(
            &bus,
            "matroska",
            self.duration_ns()
                .map(crate::clock::ClockTime::from_nanos)
                .unwrap_or(crate::clock::ClockTime::NONE),
            reports,
        );
    }

    /// All tracks in the file, including unrouted subtitle/extra tracks.
    pub fn tracks(&self) -> &[MkvTrack] {
        &self.tracks
    }

    /// The first video track, if any.
    pub fn video_track(&self) -> Option<&MkvTrack> {
        self.tracks
            .iter()
            .find(|t| t.track_type == MkvTrackType::Video && t.codec.is_video())
    }

    /// The first audio track, if any.
    pub fn audio_track(&self) -> Option<&MkvTrack> {
        self.tracks
            .iter()
            .find(|t| t.track_type == MkvTrackType::Audio && t.codec.is_audio())
    }

    /// Total duration in nanoseconds, when the segment declares one.
    pub fn duration_ns(&self) -> Option<u64> {
        self.file
            .info()
            .duration()
            .map(|ticks| (ticks * self.timestamp_scale as f64) as u64)
    }

    /// Convert one frame's payload directly into an arena slot and wrap it
    /// with timing metadata — no intermediate `Vec` (#144).
    ///
    /// H.264 becomes a self-contained Annex-B access unit (SPS/PPS in-band
    /// on keyframes) sized exactly via a length-prefix pre-pass;
    /// header-stripped tracks get their prefix restored; everything else
    /// copies through once.
    fn emit_frame(
        &mut self,
        track: u64,
        data: &[u8],
        timestamp_ticks: u64,
        duration_ticks: Option<u64>,
        is_keyframe: bool,
    ) -> Result<Buffer> {
        use crate::codec::annexb;

        let prefix = self
            .stripped_prefix
            .iter()
            .find(|(id, _)| *id == track)
            .map(|(_, p)| p.as_slice())
            .unwrap_or(&[]);

        let nal_config = self
            .nal_configs
            .iter()
            .find(|(id, _)| *id == track)
            .map(|(_, cfg)| cfg);
        let is_length_prefixed_video = nal_config.is_some()
            && self
                .routed
                .iter()
                .any(|(id, pad)| *id == track && pad.0 == 0);

        let handle = if is_length_prefixed_video && prefix.is_empty() {
            let cfg = nal_config.expect("checked above");
            let params_len: usize = if is_keyframe {
                cfg.params.iter().map(|n| 4 + n.len()).sum()
            } else {
                0
            };
            let total = params_len + annexb::annex_b_len(data, cfg.length_size)?;
            let mut slot = self.output.acquire(total, "mkvdemux")?;
            {
                let out = &mut slot.data_mut()[..total];
                let mut pos = 0usize;
                if is_keyframe {
                    for nal in &cfg.params {
                        out[pos..pos + 4].copy_from_slice(&[0, 0, 0, 1]);
                        out[pos + 4..pos + 4 + nal.len()].copy_from_slice(nal);
                        pos += 4 + nal.len();
                    }
                }
                pos += annexb::avcc_to_annex_b_into_slice(data, cfg.length_size, &mut out[pos..])?;
                debug_assert_eq!(pos, total);
            }
            MemoryHandle::with_len(slot, total)
        } else if is_length_prefixed_video {
            // Header-stripped (rare): the length prefixes span the
            // prefix/data boundary, so materialize once, then convert.
            let mut joined = Vec::with_capacity(prefix.len() + data.len());
            joined.extend_from_slice(prefix);
            joined.extend_from_slice(data);
            let cfg = nal_config.expect("checked above");
            let params_len: usize = if is_keyframe {
                cfg.params.iter().map(|n| 4 + n.len()).sum()
            } else {
                0
            };
            let total = params_len + annexb::annex_b_len(&joined, cfg.length_size)?;
            let mut slot = self.output.acquire(total, "mkvdemux")?;
            {
                let out = &mut slot.data_mut()[..total];
                let mut pos = 0usize;
                if is_keyframe {
                    for nal in &cfg.params {
                        out[pos..pos + 4].copy_from_slice(&[0, 0, 0, 1]);
                        out[pos + 4..pos + 4 + nal.len()].copy_from_slice(nal);
                        pos += 4 + nal.len();
                    }
                }
                annexb::avcc_to_annex_b_into_slice(&joined, cfg.length_size, &mut out[pos..])?;
            }
            MemoryHandle::with_len(slot, total)
        } else {
            // Passthrough (optionally prefix-restored): one copy into the slot.
            let total = prefix.len() + data.len();
            let mut slot = self.output.acquire(total, "mkvdemux")?;
            {
                let out = &mut slot.data_mut()[..total];
                out[..prefix.len()].copy_from_slice(prefix);
                out[prefix.len()..].copy_from_slice(data);
            }
            MemoryHandle::with_len(slot, total)
        };

        let mut metadata = Metadata::new();
        metadata.stream_id = track as u32;
        let pts = timestamp_ticks.saturating_mul(self.timestamp_scale);
        metadata.pts = ClockTime::from_nanos(pts);
        // Matroska blocks carry presentation time only (see module docs).
        metadata.dts = ClockTime::from_nanos(pts);
        let duration_ns = duration_ticks
            .map(|d| d.saturating_mul(self.timestamp_scale))
            .or_else(|| {
                self.default_duration_ns
                    .iter()
                    .find(|(id, _)| *id == track)
                    .map(|(_, d)| *d)
            });
        if let Some(d) = duration_ns {
            metadata.duration = ClockTime::from_nanos(d);
        }
        if is_keyframe {
            metadata.flags |= BufferFlags::SYNC_POINT;
        }

        // Say what this is (#238). The demuxer parsed the codec and the pixel
        // dimensions out of the track entry and used to throw both away, so a
        // `mkvdemux ! zenohsink` published H.264 as `zenoh/bytes` with no
        // geometry, and `EncoderElement` errors outright on a buffer that
        // declares no format.
        if let Some(track_info) = self.tracks.iter().find(|t| t.id == track) {
            if let Some(codec) = track_info.codec.video_codec() {
                match &track_info.video_info {
                    Some(v) if v.width > 0 && v.height > 0 => {
                        metadata.set_coded_video_dims(codec, v.width, v.height)
                    }
                    _ => metadata.format = Some(crate::format::MediaFormat::Video(codec)),
                }
                // What the track's `Colour` element said (#265). It rides the
                // *encoded* buffer, so each decoder carries it across onto the
                // frame it produces — the demuxer never emits raw video.
                if let Some(color) = track_info.video_info.as_ref().and_then(|v| v.color) {
                    metadata.set_color_info(color);
                }
                // Non-square pixels (#266), by the same route as the colour.
                if let Some(display) = track_info.video_info.as_ref().and_then(|v| v.display) {
                    metadata.set_display_geometry(display);
                }
            } else if let Some(codec) = track_info.codec.audio_codec() {
                metadata.format = Some(crate::format::MediaFormat::Audio(codec));
            } else if let Some(codec) = Self::text_codec(&track_info.codec_id) {
                // A subtitle block is already a cue: Matroska strips the
                // timing into the block, and this buffer's pts/duration are
                // that timing (#279).
                metadata.format = Some(crate::format::MediaFormat::Text(codec));
            }
        }

        Ok(Buffer::new(handle, metadata))
    }

    /// Best-effort keyframe determination for a video frame.
    ///
    /// SimpleBlock files carry the flag; BlockGroup files (typical for H.264
    /// MKVs) don't expose one through the parser, so probe the bitstream.
    fn is_keyframe(&self, codec: &MkvCodec, flag: Option<bool>, data: &[u8]) -> bool {
        if let Some(k) = flag {
            return k;
        }
        match codec {
            MkvCodec::H264 | MkvCodec::H265 => {
                // Length-prefixed at this point; walk NALs by prefix.
                let len_size = self
                    .nal_configs
                    .first()
                    .map(|(_, c)| c.length_size as usize)
                    .unwrap_or(4);
                let is_irap = |first: u8, second: u8| match codec {
                    // H.264: nal_unit_type 5 is an IDR slice.
                    MkvCodec::H264 => first & 0x1f == 5,
                    // HEVC: types 16..=23 are the IRAP set (BLA, IDR, CRA),
                    // and the type lives in the first byte's bits 1..6 of a
                    // two-byte header.
                    _ => {
                        let _ = second;
                        matches!((first >> 1) & 0x3f, 16..=23)
                    }
                };
                let mut pos = 0usize;
                while pos + len_size <= data.len() {
                    let mut len = 0usize;
                    for &b in &data[pos..pos + len_size] {
                        len = (len << 8) | b as usize;
                    }
                    pos += len_size;
                    if let Some(&first) = data.get(pos)
                        && is_irap(first, data.get(pos + 1).copied().unwrap_or(0))
                    {
                        return true;
                    }
                    pos = pos.saturating_add(len);
                }
                false
            }
            MkvCodec::Vp8 => data.first().is_some_and(|b| b & 1 == 0),
            MkvCodec::Vp9 => vp9_is_keyframe(data),
            // AV1 keyframe detection needs OBU parsing; treat unknown as
            // non-key so seek resync waits for a flagged frame.
            _ if codec.is_video() => false,
            _ => true, // audio frames are all sync points
        }
    }

    /// Reverse trick mode (#165): walk cue points backward from the cursor.
    ///
    /// Each call repositions at the last cue at or before the cursor (the
    /// same `file.seek(0)` + raw-reposition primitive the seek path uses),
    /// scans that cluster forward for the cue's video keyframe, emits it,
    /// and moves the cursor just below the cue. Granularity is the cue
    /// table's — a keyframe between cue points is skipped, coarser than
    /// MP4's exact stss walk but the honest best an index-driven walk can
    /// do. Audio is muted; a cue whose keyframe the heuristic rejects is
    /// skipped with an empty yield so termination never depends on the
    /// bitstream probes.
    fn produce_reverse(&mut self) -> Result<crate::element::DemuxerProduce> {
        use crate::element::{DemuxerProduce, RoutedOutput};

        let Some(&(video_track, video_pad)) = self.routed.iter().find(|(_, pad)| pad.0 == 0) else {
            return Ok(DemuxerProduce::Eos);
        };
        let Some(cursor) = self.reverse_cursor_ticks else {
            return Ok(DemuxerProduce::Eos);
        };
        let Some((cue_ticks, cluster_offset)) = self
            .cues
            .iter()
            .rev()
            .find(|c| c.time_ticks <= cursor)
            .map(|c| (c.time_ticks, c.cluster_offset))
        else {
            self.reverse_cursor_ticks = None;
            return Ok(DemuxerProduce::Eos);
        };
        if cue_ticks < self.reverse_end_ticks {
            self.reverse_cursor_ticks = None;
            return Ok(DemuxerProduce::Eos);
        }
        // Strictly decreasing across calls: the next probe sits below this
        // cue, so the same cue can never be selected twice — termination is
        // guaranteed by the walk, not by the scan below succeeding.
        let next_cursor =
            (cue_ticks > 0 && cue_ticks > self.reverse_end_ticks).then(|| cue_ticks - 1);

        self.file
            .seek(0)
            .map_err(|e| Error::Config(format!("MKV reverse state reset failed: {e:?}")))?;
        self.raw
            .seek(std::io::SeekFrom::Start(cluster_offset))
            .map_err(|e| Error::Config(format!("MKV reverse reposition failed: {e}")))?;

        let codec = self
            .tracks
            .iter()
            .find(|t| t.id == video_track)
            .map(|t| t.codec.clone())
            .unwrap_or(MkvCodec::Unknown);
        loop {
            let more = self
                .file
                .next_frame(&mut self.frame)
                .map_err(|e| Error::Config(format!("MKV read error: {e:?}")))?;
            let step = |this: &mut Self| {
                this.reverse_cursor_ticks = next_cursor;
                if next_cursor.is_some() {
                    // Empty yield: the executor calls produce() again and
                    // the walk continues from the previous cue.
                    Ok(DemuxerProduce::Routed(RoutedOutput::new()))
                } else {
                    Ok(DemuxerProduce::Eos)
                }
            };
            if !more {
                return step(self);
            }
            if self.frame.track != video_track {
                continue;
            }
            let ts = self.frame.timestamp;
            if ts > cursor {
                // Passed the window without finding the cue's keyframe.
                return step(self);
            }
            if ts < cue_ticks {
                continue;
            }
            if !self.is_keyframe(&codec, self.frame.is_keyframe, &self.frame.data) {
                continue;
            }
            let duration = self.frame.duration;
            let data = std::mem::take(&mut self.frame.data);
            let result = self.emit_frame(video_track, &data, ts, duration, true);
            self.frame.data = data;
            // Advance BEFORE returning either way: on arena exhaustion the
            // frame survives in `pending` and the retry must emit it and
            // move on, not re-walk the cluster.
            self.reverse_cursor_ticks = next_cursor;
            return match result {
                Ok(buffer) => Ok(DemuxerProduce::Routed(RoutedOutput::single(
                    video_pad, buffer,
                ))),
                Err(e) => {
                    self.pending = Some(PendingFrame {
                        pad: video_pad,
                        raw: self.frame.data.clone(),
                        track: video_track,
                        timestamp_ticks: ts,
                        duration_ticks: duration,
                        is_keyframe: true,
                    });
                    Err(e)
                }
            };
        }
    }
}

impl<R: Read + Seek + Send> crate::element::Demuxer for MkvDemux<R> {
    fn set_bus(&mut self, bus: crate::pipeline::bus::BusHandle) {
        self.bus = Some(bus);
    }

    fn take_downstream_event(&mut self) -> Option<crate::event::Event> {
        self.staged_segment.take().map(crate::event::Event::Segment)
    }

    fn demux(&mut self, _buffer: Buffer) -> Result<crate::element::RoutedOutput> {
        Err(Error::Config(
            "MkvDemux owns its reader; add it with no input link".into(),
        ))
    }

    fn produce(&mut self) -> Result<crate::element::DemuxerProduce> {
        use crate::element::{DemuxerProduce, RoutedOutput};

        self.report_streams();

        // A frame parked by arena exhaustion goes out before anything new is
        // read — the executor treats PoolExhausted as flow control and calls
        // produce() again.
        if let Some(p) = self.pending.take() {
            match self.emit_frame(
                p.track,
                &p.raw,
                p.timestamp_ticks,
                p.duration_ticks,
                p.is_keyframe,
            ) {
                Ok(buffer) => {
                    return Ok(DemuxerProduce::Routed(RoutedOutput::single(p.pad, buffer)));
                }
                Err(e) => {
                    self.pending = Some(p);
                    return Err(e);
                }
            }
        }

        // Reverse trick mode (#165): dispatched after the pending drain so
        // an arena-exhausted reverse keyframe is replayed, not re-walked.
        if self.trick_rate < 0.0 {
            return self.produce_reverse();
        }

        let mut discarded = 0usize;
        loop {
            // Bound the scan work per call: past the budget, yield an empty
            // result (a free cooperative yield — no sleep) so the executor
            // can drain a superseding seek before the scan continues.
            if discarded >= self.scan_budget {
                return Ok(DemuxerProduce::Routed(RoutedOutput::new()));
            }

            let more = self
                .file
                .next_frame(&mut self.frame)
                .map_err(|e| Error::Config(format!("MKV read error: {e:?}")))?;
            if !more {
                if self.looping {
                    self.file
                        .seek(0)
                        .map_err(|e| Error::Config(format!("MKV rewind failed: {e:?}")))?;
                    self.video_resync = true;
                    // Re-establish the timeline for the new lap: PTS restart
                    // at zero, running time must not. The executor fills in
                    // the base already consumed (#265).
                    self.staged_segment =
                        Some(crate::event::SegmentEvent::new_time(ClockTime::ZERO, None));
                    // A seek past the last frame leaves skip entries no frame
                    // can ever satisfy; the loop rewind must drop them or it
                    // would discard every frame forever.
                    self.skip.clear();
                    continue;
                }
                return Ok(DemuxerProduce::Eos);
            }

            let track = self.frame.track;
            let Some(&(_, pad)) = self.routed.iter().find(|(id, _)| *id == track) else {
                continue; // subtitle or extra track
            };

            // Seek stop (#165): frames interleave in cluster time order, so
            // the first routed frame at or past `stop` means every track is
            // there — playback ends at the seek's stop, not the file's end.
            if self
                .stop_ticks
                .is_some_and(|stop| self.frame.timestamp >= stop)
            {
                return Ok(DemuxerProduce::Eos);
            }

            // Post-seek skip, strictly per track: only this track's own
            // first frame at/after the target clears its entry, so audio
            // interleaved ahead of video can no longer let pre-target video
            // through.
            if let Some(i) = self.skip.iter().position(|(id, _)| *id == track) {
                if self.frame.timestamp < self.skip[i].1 {
                    discarded += 1;
                    continue;
                }
                self.skip.swap_remove(i);
            }

            let codec = self
                .tracks
                .iter()
                .find(|t| t.id == track)
                .map(|t| t.codec.clone())
                .unwrap_or(MkvCodec::Unknown);

            let is_video = pad.0 == 0;
            let flag = self.frame.is_keyframe;
            let is_key = if is_video {
                self.is_keyframe(&codec, flag, &self.frame.data)
            } else {
                true
            };

            // After a seek, hold video until a keyframe so a decoder never
            // joins mid-GOP. Evaluated only after the video track's own skip
            // has cleared — a pre-target keyframe (the t=0 one, during a
            // rewind scan) can no longer satisfy the resync while the skip is
            // still discarding (#158).
            //
            // Audio passes through (every frame decodes). It is *not* held
            // with the video: a seek past the last keyframe legitimately
            // emits no video at all, and holding audio for a keyframe that
            // will never come would turn that into silence. What keeps the
            // two branches opening together instead is the skip floor — see
            // the forward-snap landing in `handle_upstream_event` (#265).
            if self.video_resync && is_video {
                if !is_key {
                    discarded += 1;
                    continue;
                }
                self.video_resync = false;
            }

            // Fast-forward (#165): keyframes only, no audio.
            if self.trick_rate > 1.0 && (!is_video || !is_key) {
                discarded += 1;
                continue;
            }

            // Convert straight into the slot (the take/restore dance frees
            // the `&self.frame` borrow for the `&mut self` call). On arena
            // exhaustion the raw payload is copied once into the pending
            // stash — an allocation only on that rare path.
            let ticks = self.frame.timestamp;
            let duration = self.frame.duration;
            let data = std::mem::take(&mut self.frame.data);
            let result = self.emit_frame(track, &data, ticks, duration, is_key);
            self.frame.data = data;
            match result {
                Ok(buffer) => {
                    return Ok(DemuxerProduce::Routed(RoutedOutput::single(pad, buffer)));
                }
                Err(e) => {
                    self.pending = Some(PendingFrame {
                        pad,
                        raw: self.frame.data.clone(),
                        track,
                        timestamp_ticks: ticks,
                        duration_ticks: duration,
                        is_keyframe: is_key,
                    });
                    return Err(e);
                }
            }
        }
    }

    fn pad_name(&self, pad: crate::element::PadId) -> String {
        // Exhaustive on purpose. The catch-all used to be `_ => "audio"`, so
        // adding a third pad silently collided with the audio one — every
        // subtitle buffer would have gone down the audio branch (#279).
        match pad.0 {
            0 => "video".into(),
            1 => "audio".into(),
            _ => "subtitle".into(),
        }
    }

    fn handle_upstream_event(&mut self, event: &crate::event::Event) -> crate::event::EventResult {
        use crate::event::{Event, EventResult, SeekType, SegmentFormat};

        let Event::Seek(seek) = event else {
            return EventResult::NotHandled;
        };
        if seek.format != SegmentFormat::Time || seek.start.seek_type != SeekType::Set {
            return EventResult::NotHandled;
        }
        // Reverse (#165): a backward cue walk. Needs a resolvable top of
        // range and a cue table to walk — cue-less Matroska has no backward
        // index at all, and refusing beats a whole-file keyframe scan.
        if seek.rate < 0.0 {
            let start_ns = seek.start.position.max(0) as u64;
            let stop_ns = match seek.stop.seek_type {
                SeekType::Set => seek.stop.position.max(0) as u64,
                _ => match self.duration_ns() {
                    Some(d) if d > 0 => d,
                    _ => {
                        tracing::warn!(
                            "mkvdemux: reverse seek needs a stop or a known duration; refused"
                        );
                        return EventResult::Error;
                    }
                },
            };
            if stop_ns <= start_ns {
                tracing::warn!("mkvdemux: reverse seek range is empty ({start_ns}..{stop_ns})");
                return EventResult::Error;
            }
            if self.cues.is_empty() {
                tracing::warn!("mkvdemux: reverse playback needs cues; cue-less file refused");
                return EventResult::Error;
            }
            self.pending = None;
            self.skip.clear();
            self.video_resync = false;
            self.staged_segment = None;
            self.stop_ticks = None;
            self.trick_rate = seek.rate;
            self.reverse_end_ticks = start_ns / self.timestamp_scale;
            self.reverse_cursor_ticks = Some((stop_ns - 1) / self.timestamp_scale);
            tracing::info!(
                "mkvdemux: reverse seek over [{start_ns}, {stop_ns}) ns, cue keyframes only"
            );
            return EventResult::handled_at(stop_ns as i64);
        }
        let target_ns = seek.start.position.max(0) as u64;
        let ticks = target_ns / self.timestamp_scale;

        // Snap direction (#166). Default is forward — MKV's scan-based skip
        // naturally lands on the first keyframe at or after the target.
        // Resolution happens at cue granularity: `prev`/`next` bracket the
        // target in the cue table, and a non-cued keyframe between cues can
        // make the true forward landing closer than `next` suggests — best
        // effort, same as GStreamer's demuxers. Cue-less files can only
        // scan forward, so Before/Nearest degrade.
        let requested = seek.flags.snap().unwrap_or(crate::event::SeekSnap::After);
        let prev = self
            .cues
            .iter()
            .rev()
            .find(|c| c.time_ticks <= ticks)
            .map(|c| (c.time_ticks, c.cluster_offset));
        let next_ticks = self
            .cues
            .iter()
            .find(|c| c.time_ticks > ticks)
            .map(|c| c.time_ticks);
        use crate::event::SeekSnap;
        let snap = if self.cues.is_empty() {
            if requested != SeekSnap::After {
                tracing::debug!(
                    "mkvdemux: {requested:?} requested but the file has no cues; forward snap"
                );
            }
            SeekSnap::After
        } else {
            match requested {
                SeekSnap::Nearest => match (prev, next_ticks) {
                    (Some((p, _)), Some(n)) if n - ticks < ticks - p => SeekSnap::After,
                    (Some(_), _) => SeekSnap::Before,
                    (None, _) => SeekSnap::After,
                },
                s => s,
            }
        };
        // Backward snap lands exactly on a cue (skip to the cue's own time,
        // landing knowable up front). Forward lands on the next cue when
        // there is one — cues *are* video keyframes, so that is the honest
        // landing and it can be reported up front like the backward one
        // (#265; it used to report `None`, which made the executor synthesize
        // the segment around the *requested* time while video started a GOP
        // later). Without cues there is no index to resolve against and the
        // scan has to find the keyframe, so the landing stays unknown and
        // `video_resync` holds both tracks until it does.
        let (cue_target, skip_ticks, landing_ticks) = match snap {
            SeekSnap::Before => {
                // `prev`, or the first cue when the target precedes it —
                // nothing decodable earlier (mirrors MP4's rule for a target
                // before the first keyframe).
                let (cue_ticks, offset) = prev.unwrap_or_else(|| {
                    let c = &self.cues[0];
                    (c.time_ticks, c.cluster_offset)
                });
                (Some(offset), cue_ticks, Some(cue_ticks))
            }
            // Forward. Enter at the *previous* cue's cluster either way: the
            // landing frame's cluster may begin before it, and the per-track
            // skip discards everything below the floor anyway.
            _ => match next_ticks {
                // A cue ahead of the target is the landing.
                Some(landing) => (prev.map(|(_, offset)| offset), landing, Some(landing)),
                // Past the last cue: only the scan can find the keyframe.
                None => (prev.map(|(_, offset)| offset), ticks, None),
            },
        };

        // Cue-indexed path first, on OUR cue table (`mkv_cues`): the crate's
        // own cue seek mis-resolves ffmpeg's CueRelativePosition against the
        // first cluster — sometimes erroring, sometimes silently landing
        // wrong. `file.seek(0)` is a cheap, always-correct parser-state
        // reset (clears the crate's queued frames); the shared raw handle
        // then repositions the stream at the cue's Cluster element start,
        // from which `next_frame` resynchronizes off the cluster's own
        // Timestamp. Cue-less files take the crate's linear path, and a
        // rewind-plus-skip remains the last resort.
        let sought = match cue_target {
            Some(cluster_offset) => match self.file.seek(0) {
                Ok(()) => match self.raw.seek(std::io::SeekFrom::Start(cluster_offset)) {
                    Ok(_) => {
                        tracing::debug!(
                            "mkvdemux: cue seek to tick {ticks} lands at cluster offset \
                             {cluster_offset}"
                        );
                        true
                    }
                    Err(e) => {
                        tracing::warn!("mkvdemux: cue reposition failed: {e}");
                        false
                    }
                },
                Err(e) => {
                    tracing::warn!("mkvdemux: seek state reset failed: {e:?}");
                    false
                }
            },
            None => match self.file.seek(ticks) {
                Ok(()) => true,
                Err(fast) => {
                    tracing::debug!(
                        "mkvdemux: crate seek to tick {ticks} failed ({fast:?}); \
                         falling back to linear scan from the start"
                    );
                    self.file.seek(0).is_ok()
                }
            },
        };
        if sought {
            tracing::info!("mkvdemux: seek to {target_ns} ns (tick {ticks}, {snap:?})");
            self.pending = None;
            self.video_resync = true;
            // A loop rewind may have staged a Segment that this seek just
            // superseded; emitting it afterwards would re-anchor both pads at
            // zero on top of the seek's own segment (#265).
            self.staged_segment = None;
            // Trick play (#165): the seek's rate governs playback until the
            // next handled seek changes it. A forward seek leaves reverse.
            self.trick_rate = if seek.rate > 0.0 { seek.rate } else { 1.0 };
            self.reverse_cursor_ticks = None;
            // Honor the seek's stop (#165): playback ends there. A seek
            // without a stop clears any previous one.
            self.stop_ticks = match seek.stop.seek_type {
                SeekType::Set => Some(seek.stop.position.max(0) as u64 / self.timestamp_scale),
                _ => None,
            };
            // Per-track skip on EVERY path: the cue lands at (or before) the
            // skip target's cluster, the crate's narrow phase only guarantees
            // the first block, and the rewind fallback starts at zero — in
            // all three cases each routed track discards its own frames
            // below the target.
            self.skip = self
                .routed
                .iter()
                .map(|&(id, _)| (id, skip_ticks))
                .collect();
            match landing_ticks {
                // Backward snap: the cue IS the landing, report it so the
                // Segment and SeekDone carry the honest position. Slightly
                // optimistic if the keyframe heuristic rejects the cue's
                // block and the resync scans past it — rare, self-correcting
                // (video simply starts at the next keyframe).
                Some(t) => EventResult::handled_at((t * self.timestamp_scale) as i64),
                // Forward snap on a cue-less file: the landing is unknown
                // until produce() finds the first keyframe at or after the
                // target — report None and let the executor fall back to the
                // requested position for the Segment (#162). `video_resync`
                // then holds *both* tracks to that keyframe, so the branches
                // still open together (#265).
                None => EventResult::handled(),
            }
        } else {
            EventResult::Error
        }
    }

    fn is_seekable(&self) -> bool {
        // Cue-indexed when the file carries Cues; cue-less files still
        // seek via the crate's linear path and the rewind fallback.
        true
    }

    fn query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        // None when the Segment declares no Duration (streamed/remuxed
        // WebM) — the handle then reports ClockTime::NONE, and players
        // keep their upper seek clamp open.
        self.duration_ns()
            .map(|d| crate::pipeline::seek::DurationQuery {
                format: crate::event::SegmentFormat::Time,
                duration: Some(d),
            })
    }

    fn outputs(&self) -> &[(crate::element::PadId, crate::format::Caps)] {
        &self.outputs
    }

    fn on_pad_added(&mut self, _callback: crate::element::PadAddedCallback) {}

    fn name(&self) -> &str {
        "mkvdemux"
    }

    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }
}

impl<R: Read + Seek> std::fmt::Debug for MkvDemux<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MkvDemux")
            .field("tracks", &self.tracks)
            .field("timestamp_scale", &self.timestamp_scale)
            .finish()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod display_tests {
    use super::par_from_display;

    #[test]
    fn the_display_elements_become_a_reduced_pixel_ratio() {
        // Anamorphic PAL: 720x576 stored, displayed 16:9. One formula covers
        // every DisplayUnit, because the pair always states a *display*
        // aspect whatever it is spelled in — pixels here, a bare ratio in the
        // next case, and the answer is the same.
        assert_eq!(par_from_display(1024, 576, 720, 576), Some((64, 45)));
        assert_eq!(par_from_display(16, 9, 720, 576), Some((64, 45)));

        // Square pixels say nothing: it is what a consumer assumes, so
        // stamping it would cost a wire field per buffer for no answer.
        assert_eq!(par_from_display(1920, 1080, 1920, 1080), None);
        // ...including when the numbers differ but the ratio does not.
        assert_eq!(par_from_display(960, 540, 1920, 1080), None);

        // A zero is a division waiting to happen further downstream.
        assert_eq!(par_from_display(0, 576, 720, 576), None);
        assert_eq!(par_from_display(1024, 0, 720, 576), None);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codec_id_mapping() {
        assert_eq!(MkvCodec::from_codec_id("V_MPEG4/ISO/AVC"), MkvCodec::H264);
        assert_eq!(MkvCodec::from_codec_id("V_VP8"), MkvCodec::Vp8);
        assert_eq!(MkvCodec::from_codec_id("V_VP9"), MkvCodec::Vp9);
        assert_eq!(MkvCodec::from_codec_id("V_AV1"), MkvCodec::Av1);
        assert_eq!(MkvCodec::from_codec_id("A_AAC"), MkvCodec::Aac);
        assert_eq!(MkvCodec::from_codec_id("A_AAC/MPEG4/LC"), MkvCodec::Aac);
        assert_eq!(MkvCodec::from_codec_id("A_OPUS"), MkvCodec::Opus);
        assert_eq!(MkvCodec::from_codec_id("A_VORBIS"), MkvCodec::Vorbis);
        assert_eq!(MkvCodec::from_codec_id("A_EAC3"), MkvCodec::Eac3);
        assert_eq!(MkvCodec::from_codec_id("S_TEXT/UTF8"), MkvCodec::Subtitle);
        assert_eq!(
            MkvCodec::from_codec_id("V_MS/VFW/FOURCC"),
            MkvCodec::Unknown
        );
    }

    #[test]
    fn codec_display() {
        assert_eq!(format!("{}", MkvCodec::H264), "H.264/AVC");
        assert_eq!(format!("{}", MkvCodec::Vp9), "VP9");
        assert_eq!(format!("{}", MkvCodec::Eac3), "E-AC-3");
    }

    #[test]
    fn avcc_parse_roundtrip() {
        // version 1, profile/compat/level, 4-byte lengths, 1 SPS, 1 PPS.
        let sps = [0x67, 0x42, 0x00, 0x1e];
        let pps = [0x68, 0xce, 0x38, 0x80];
        let mut avcc = vec![1, 0x42, 0x00, 0x1e, 0xff, 0xe1];
        avcc.extend_from_slice(&(sps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&sps);
        avcc.push(1);
        avcc.extend_from_slice(&(pps.len() as u16).to_be_bytes());
        avcc.extend_from_slice(&pps);

        let cfg = parse_avcc(&avcc).unwrap();
        assert_eq!(cfg.length_size, 4);
        assert_eq!(cfg.params, vec![sps.to_vec(), pps.to_vec()], "SPS then PPS");
    }

    /// hvcC keeps its parameter sets in typed arrays rather than two counted
    /// lists, and puts the length-prefix width 17 bytes further along.
    #[test]
    fn hvcc_parse_extracts_vps_sps_pps_in_order() {
        let vps = [0x40u8, 0x01, 0x0c];
        let sps = [0x42u8, 0x01, 0x01];
        let pps = [0x44u8, 0x01, 0xc1];
        let mut hvcc = vec![1u8];
        hvcc.extend_from_slice(&[0; 20]); // profile/tier/level and friends
        hvcc.push(0xf3); // reserved(6) | lengthSizeMinusOne(2) = 3 -> 4 bytes
        hvcc.push(3); // three arrays
        for (nal_type, nal) in [(32u8, &vps[..]), (33, &sps[..]), (34, &pps[..])] {
            hvcc.push(nal_type);
            hvcc.extend_from_slice(&1u16.to_be_bytes());
            hvcc.extend_from_slice(&(nal.len() as u16).to_be_bytes());
            hvcc.extend_from_slice(nal);
        }

        let cfg = parse_hvcc(&hvcc).unwrap();
        assert_eq!(cfg.length_size, 4);
        assert_eq!(
            cfg.params,
            vec![vps.to_vec(), sps.to_vec(), pps.to_vec()],
            "VPS, SPS, PPS in the order the record listed them"
        );
    }

    #[test]
    fn hvcc_parse_rejects_garbage() {
        assert!(parse_hvcc(&[]).is_err());
        assert!(parse_hvcc(&[2; 32]).is_err(), "wrong version");
        let mut truncated = vec![1u8];
        truncated.extend_from_slice(&[0; 21]);
        truncated.push(1); // claims one array, then stops
        assert!(parse_hvcc(&truncated).is_err());
    }

    #[test]
    fn avcc_parse_rejects_garbage() {
        assert!(parse_avcc(&[]).is_err());
        assert!(parse_avcc(&[2, 0, 0, 0, 0xff, 0xe1]).is_err());
        assert!(parse_avcc(&[1, 0, 0, 0, 0xff, 0xe1, 0x00]).is_err());
    }

    #[test]
    fn vp8_vp9_keyframe_probe() {
        // VP8: bit 0 of the first byte is 0 for keyframes.
        assert!(MkvCodec::Vp8.is_video());
        // VP9 keyframe: marker 0b10, profile 0, show_existing 0, frame_type 0.
        assert!(vp9_is_keyframe(&[0b1000_0000]));
        // Non-key (frame_type = 1).
        assert!(!vp9_is_keyframe(&[0b1000_0100]));
        // show_existing_frame set.
        assert!(!vp9_is_keyframe(&[0b1000_1000]));
        // Bad marker.
        assert!(!vp9_is_keyframe(&[0b0100_0000]));
    }
}
