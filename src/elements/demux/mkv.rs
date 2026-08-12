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
}

// ============================================================================
// avcC parsing (Matroska keeps H.264 parameter sets in CodecPrivate)
// ============================================================================

/// Parsed avcC decoder configuration from a track's CodecPrivate.
struct AvcConfig {
    /// NAL length prefix size in bytes (1, 2 or 4).
    length_size: u8,
    /// Raw SPS NALs (no start codes).
    sps: Vec<Vec<u8>>,
    /// Raw PPS NALs (no start codes).
    pps: Vec<Vec<u8>>,
}

/// Parse an AVCDecoderConfigurationRecord (ISO 14496-15 §5.3.3.1).
fn parse_avcc(data: &[u8]) -> Result<AvcConfig> {
    let err = || Error::Config("truncated avcC record in CodecPrivate".into());
    if data.len() < 7 || data[0] != 1 {
        return Err(Error::Config(format!(
            "CodecPrivate is not an avcC record (len {}, version {})",
            data.len(),
            data.first().copied().unwrap_or(0)
        )));
    }
    let length_size = (data[4] & 0x03) + 1;
    let mut pos = 5usize;
    let read_sets = |pos: &mut usize, count: usize| -> Result<Vec<Vec<u8>>> {
        let mut sets = Vec::with_capacity(count);
        for _ in 0..count {
            let len = data
                .get(*pos..*pos + 2)
                .map(|b| u16::from_be_bytes([b[0], b[1]]) as usize)
                .ok_or_else(err)?;
            *pos += 2;
            sets.push(data.get(*pos..*pos + len).ok_or_else(err)?.to_vec());
            *pos += len;
        }
        Ok(sets)
    };
    let num_sps = (*data.get(pos).ok_or_else(err)? & 0x1f) as usize;
    pos += 1;
    let sps = read_sets(&mut pos, num_sps)?;
    let num_pps = *data.get(pos).ok_or_else(err)? as usize;
    pos += 1;
    let pps = read_sets(&mut pos, num_pps)?;
    Ok(AvcConfig {
        length_size,
        sps,
        pps,
    })
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
    /// avcC config of the routed H.264 video track, if any.
    avc_config: Option<AvcConfig>,
    /// Bytes to prepend per frame for tracks using header-stripping
    /// compression, keyed like `routed`.
    stripped_prefix: Vec<(u64, Vec<u8>)>,
    /// Per-track default frame duration in ns (Matroska DefaultDuration).
    default_duration_ns: Vec<(u64, u64)>,
    output: OutputArena,
    /// Reusable frame to avoid a fresh allocation per block.
    frame: Frame,
    looping: bool,
    /// Drop non-keyframe video frames until the first keyframe (set after a
    /// seek so a decoder never sees a mid-GOP entry).
    video_resync: bool,
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
            avc_config: None,
            stripped_prefix: Vec::new(),
            default_duration_ns: Vec::new(),
            output: OutputArena::new(defaults::MP4_DEMUX_SLOT_COUNT)
                .with_min_slot_size(defaults::MP4_DEMUX_SLOT_SIZE)
                .grow_to_fit(),
            frame: Frame::default(),
            looping: false,
            video_resync: false,
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
    fn route(
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
        if track.codec == MkvCodec::H264 {
            match entry.codec_private() {
                Some(private) => self.avc_config = Some(parse_avcc(private)?),
                None => tracing::warn!(
                    "H.264 MKV track {id} has no avcC CodecPrivate; frames pass \
                     through length-prefixed and will not decode"
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

        let is_h264_video = self.avc_config.is_some()
            && self
                .routed
                .first()
                .is_some_and(|(id, pad)| *id == track && pad.0 == 0);

        let handle = if is_h264_video && prefix.is_empty() {
            let cfg = self.avc_config.as_ref().expect("checked above");
            let params_len: usize = if is_keyframe {
                cfg.sps
                    .iter()
                    .chain(cfg.pps.iter())
                    .map(|n| 4 + n.len())
                    .sum()
            } else {
                0
            };
            let total = params_len + annexb::annex_b_len(data, cfg.length_size)?;
            let mut slot = self.output.acquire(total, "mkvdemux")?;
            {
                let out = &mut slot.data_mut()[..total];
                let mut pos = 0usize;
                if is_keyframe {
                    for nal in cfg.sps.iter().chain(cfg.pps.iter()) {
                        out[pos..pos + 4].copy_from_slice(&[0, 0, 0, 1]);
                        out[pos + 4..pos + 4 + nal.len()].copy_from_slice(nal);
                        pos += 4 + nal.len();
                    }
                }
                pos += annexb::avcc_to_annex_b_into_slice(data, cfg.length_size, &mut out[pos..])?;
                debug_assert_eq!(pos, total);
            }
            MemoryHandle::with_len(slot, total)
        } else if is_h264_video {
            // Header-stripped H.264 (rare): the length prefixes span the
            // prefix/data boundary, so materialize once, then convert.
            let mut joined = Vec::with_capacity(prefix.len() + data.len());
            joined.extend_from_slice(prefix);
            joined.extend_from_slice(data);
            let cfg = self.avc_config.as_ref().expect("checked above");
            let params_len: usize = if is_keyframe {
                cfg.sps
                    .iter()
                    .chain(cfg.pps.iter())
                    .map(|n| 4 + n.len())
                    .sum()
            } else {
                0
            };
            let total = params_len + annexb::annex_b_len(&joined, cfg.length_size)?;
            let mut slot = self.output.acquire(total, "mkvdemux")?;
            {
                let out = &mut slot.data_mut()[..total];
                let mut pos = 0usize;
                if is_keyframe {
                    for nal in cfg.sps.iter().chain(cfg.pps.iter()) {
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
            MkvCodec::H264 => {
                // Length-prefixed at this point; walk NALs by prefix.
                let len_size = self
                    .avc_config
                    .as_ref()
                    .map(|c| c.length_size as usize)
                    .unwrap_or(4);
                let mut pos = 0usize;
                while pos + len_size <= data.len() {
                    let mut len = 0usize;
                    for &b in &data[pos..pos + len_size] {
                        len = (len << 8) | b as usize;
                    }
                    pos += len_size;
                    if let Some(&first) = data.get(pos)
                        && first & 0x1f == 5
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
}

impl<R: Read + Seek + Send> crate::element::Demuxer for MkvDemux<R> {
    fn demux(&mut self, _buffer: Buffer) -> Result<crate::element::RoutedOutput> {
        Err(Error::Config(
            "MkvDemux owns its reader; add it with no input link".into(),
        ))
    }

    fn produce(&mut self) -> Result<crate::element::DemuxerProduce> {
        use crate::element::{DemuxerProduce, RoutedOutput};

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
            // joins mid-GOP. Audio passes through (every frame decodes).
            // Evaluated only after the video track's own skip has cleared —
            // a pre-target keyframe (the t=0 one, during a rewind scan) can
            // no longer satisfy the resync while the skip is still
            // discarding (#158).
            if self.video_resync && is_video {
                if !is_key {
                    discarded += 1;
                    continue;
                }
                self.video_resync = false;
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
        match pad.0 {
            0 => "video".into(),
            _ => "audio".into(),
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
        let target_ns = seek.start.position.max(0) as u64;
        let ticks = target_ns / self.timestamp_scale;

        // Cue-indexed path first, on OUR cue table (`mkv_cues`): the crate's
        // own cue seek mis-resolves ffmpeg's CueRelativePosition against the
        // first cluster — sometimes erroring, sometimes silently landing
        // wrong. `file.seek(0)` is a cheap, always-correct parser-state
        // reset (clears the crate's queued frames); the shared raw handle
        // then repositions the stream at the cue's Cluster element start,
        // from which `next_frame` resynchronizes off the cluster's own
        // Timestamp. Cue-less files take the crate's linear path, and a
        // rewind-plus-skip remains the last resort.
        let cue = self.cues.iter().rev().find(|c| c.time_ticks <= ticks);
        let sought = match cue {
            Some(cue) => match self.file.seek(0) {
                Ok(()) => match self.raw.seek(std::io::SeekFrom::Start(cue.cluster_offset)) {
                    Ok(_) => {
                        tracing::debug!(
                            "mkvdemux: cue seek to tick {ticks} lands at cluster offset {}",
                            cue.cluster_offset
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
            tracing::info!("mkvdemux: seek to {target_ns} ns (tick {ticks})");
            self.pending = None;
            self.video_resync = true;
            // Per-track skip on EVERY path: the cue lands at (or before) the
            // target's cluster, the crate's narrow phase only guarantees the
            // first block, and the rewind fallback starts at zero — in all
            // three cases each routed track discards its own frames below
            // the target.
            self.skip = self.routed.iter().map(|&(id, _)| (id, ticks)).collect();
            // Landing unknown until produce() finds the first keyframe at or
            // after the target — report None and let the executor fall back
            // to the requested position for the Segment (#162).
            EventResult::handled()
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
        assert_eq!(cfg.sps, vec![sps.to_vec()]);
        assert_eq!(cfg.pps, vec![pps.to_vec()]);
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
