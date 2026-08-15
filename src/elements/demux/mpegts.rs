//! MPEG Transport Stream demultiplexer.
//!
//! This module provides an MPEG-TS demuxer that extracts elementary streams
//! (video, audio) from transport stream data.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::TsDemux;
//!
//! let mut demux = TsDemux::new();
//!
//! // Feed TS packets (188 bytes each)
//! while let Some(ts_data) = source.read().await? {
//!     for frame in demux.push(&ts_data)? {
//!         match frame.stream_type {
//!             TsStreamType::H264 => { /* video frame */ },
//!             TsStreamType::Aac => { /* audio frame */ },
//!             _ => {}
//!         }
//!     }
//! }
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::error::{Error, Result};
use crate::event::{
    Event, EventResult, SeekEvent, SeekFlags, SeekPosition, SeekType, SegmentFormat,
};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;

use mpeg2ts_reader::StreamType;
use mpeg2ts_reader::demultiplex::{
    self, DemuxContext, FilterChangeset, FilterRequest, NullPacketFilter, PacketFilter,
    PatPacketFilter, PmtPacketFilter,
};
use mpeg2ts_reader::pes::{self, ElementaryStreamConsumer, PesContents, PesHeader};
use mpeg2ts_reader::psi::pat::PAT_PID;

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Size of a single MPEG-TS packet.
pub const TS_PACKET_SIZE: usize = 188;

// ============================================================================
// Byte index (time → offset, for seeking a fed demuxer)
// ============================================================================

/// Sparse time-to-byte index built from the PCR clock while demuxing (#163).
///
/// A *fed* demuxer cannot seek: it does not own the reader. What it can do is
/// tell the source **where** to seek, which needs a mapping from stream time
/// to byte offset. Bisection — the obvious way to build one — is
/// architecturally unavailable here: each probe would be a full flush round
/// trip through the executor. So the index is built passively from data that
/// already flows past, and answers with a single-shot estimate.
///
/// The anchors come from the PCR, not from PES PTS: PCR is denser (a
/// conformant stream repeats it at least every 100 ms), appears in the
/// adaptation field of packets we can parse without any PSI, and is present
/// from the first packet of the program rather than the first complete access
/// unit.
///
/// # Accuracy
///
/// A linear interpolation between anchors is exact for CBR (the overwhelmingly
/// common case for TS, which pads to a constant rate) and approximate for VBR
/// or ad-spliced streams. **The honesty does not come from the estimate**: it
/// comes from reporting the first PTS that actually arrives after the seek, so
/// an application learns where it really landed even when the guess was off.
#[derive(Debug, Default, Clone)]
pub struct TsByteIndex {
    /// `(nanoseconds since the first PCR, absolute byte offset)`, sorted.
    anchors: Vec<(u64, u64)>,
    /// 90 kHz base of the first PCR ever seen — the stream's time origin.
    /// Deliberately *not* cleared by a flush: it is what makes post-seek
    /// anchors comparable with pre-seek ones.
    first_pcr: Option<u64>,
    /// The PID the first PCR came from. A multi-program stream carries one
    /// PCR per program; mixing two clocks into one index would corrupt it.
    pcr_pid: Option<u16>,
    /// Stream size in bytes, learned from the source's byte segment.
    total: Option<u64>,
}

impl TsByteIndex {
    /// Anchors closer together than this are redundant for a linear estimate.
    const MIN_SPACING_NS: u64 = 100_000_000;
    /// Above this the index is decimated (every other anchor dropped), which
    /// halves the resolution and doubles the span it can cover.
    const MAX_ANCHORS: usize = 8192;

    /// Number of anchors currently held.
    pub fn len(&self) -> usize {
        self.anchors.len()
    }

    /// Whether the index holds no anchors at all.
    pub fn is_empty(&self) -> bool {
        self.anchors.is_empty()
    }

    /// Record the stream's total size, learned from a byte-format segment.
    pub fn set_total(&mut self, total: Option<u64>) {
        if let Some(t) = total.filter(|t| *t > 0) {
            self.total = Some(t);
        }
    }

    /// Scan one 188-byte packet at absolute `offset` and record its PCR.
    fn observe(&mut self, packet: &[u8], offset: u64) {
        let Some((pid, pcr)) = packet_pcr(packet) else {
            return;
        };
        match self.pcr_pid {
            Some(locked) if locked != pid => return,
            Some(_) => {}
            None => self.pcr_pid = Some(pid),
        }
        let origin = *self.first_pcr.get_or_insert(pcr);
        // A PCR below the origin is either a 33-bit wrap (~26.5 h) or a
        // discontinuity. Either way it cannot be placed on this timeline.
        let Some(delta) = pcr.checked_sub(origin) else {
            return;
        };
        // 90 kHz ticks → ns, exactly: 1e9 / 90_000 = 100_000 / 9.
        let time_ns = delta.saturating_mul(100_000) / 9;
        self.insert(time_ns, offset);
    }

    fn insert(&mut self, time_ns: u64, offset: u64) {
        let pos = self.anchors.partition_point(|(t, _)| *t < time_ns);
        // Redundant against either neighbour: a linear estimate gains nothing.
        let crowded = |(t, _): &(u64, u64)| t.abs_diff(time_ns) < Self::MIN_SPACING_NS;
        if pos > 0 && crowded(&self.anchors[pos - 1]) {
            return;
        }
        if pos < self.anchors.len() && crowded(&self.anchors[pos]) {
            return;
        }
        self.anchors.insert(pos, (time_ns, offset));
        if self.anchors.len() > Self::MAX_ANCHORS {
            let mut keep = false;
            self.anchors.retain(|_| {
                keep = !keep;
                keep
            });
        }
    }

    /// Estimate the byte offset holding `target_ns` of stream time.
    ///
    /// `None` when the index is too thin to interpolate — fewer than two
    /// anchors, i.e. under ~200 ms of stream seen. The caller must not invent
    /// an offset in that case; refusing the seek is the honest answer.
    ///
    /// The result is clamped to the stream size (when known) and snapped down
    /// to a 188-byte packet boundary, so the source resumes at a sync byte.
    pub fn estimate_byte_offset(&self, target_ns: u64) -> Option<u64> {
        if self.anchors.len() < 2 {
            return None;
        }
        let (first_t, first_o) = self.anchors[0];
        let n = self.anchors.len();
        let raw = if target_ns <= first_t {
            first_o
        } else {
            // Interpolate within the bracketing pair, or extrapolate along the
            // last one when the target is past everything seen so far — which
            // is the normal case for a forward seek into unread data.
            let i = self.anchors.partition_point(|(t, _)| *t <= target_ns);
            let (lo, hi) = if i >= n { (n - 2, n - 1) } else { (i - 1, i) };
            interpolate(self.anchors[lo], self.anchors[hi], target_ns)
        };
        let clamped = match self.total {
            Some(total) => raw.min(total.saturating_sub(TS_PACKET_SIZE as u64)),
            None => raw,
        };
        Some(clamped / TS_PACKET_SIZE as u64 * TS_PACKET_SIZE as u64)
    }
}

/// Linear interpolation (or extrapolation) of an offset at `target`.
fn interpolate((t0, o0): (u64, u64), (t1, o1): (u64, u64), target: u64) -> u64 {
    if t1 <= t0 {
        return o1;
    }
    let span_t = (t1 - t0) as u128;
    let span_o = o1.saturating_sub(o0) as u128;
    let into = (target.saturating_sub(t0)) as u128;
    let offset = o0 as u128 + span_o * into / span_t;
    offset.min(u64::MAX as u128) as u64
}

/// Extract `(PID, PCR base)` from a TS packet's adaptation field.
///
/// The PCR base is the 33-bit 90 kHz counter; the 9-bit 27 MHz extension is
/// deliberately discarded — it buys 11 ns of precision on an estimate whose
/// error is measured in packets.
fn packet_pcr(packet: &[u8]) -> Option<(u16, u64)> {
    if packet.len() < TS_PACKET_SIZE || packet[0] != 0x47 {
        return None;
    }
    // adaptation_field_control: 0b10 = AF only, 0b11 = AF + payload.
    let afc = (packet[3] >> 4) & 0b11;
    if afc != 0b10 && afc != 0b11 {
        return None;
    }
    let af_len = packet[4] as usize;
    // Need the flags byte plus 6 PCR bytes, all inside the packet.
    if af_len < 7 || 5 + af_len > TS_PACKET_SIZE {
        return None;
    }
    if packet[5] & 0x10 == 0 {
        return None; // PCR_flag clear
    }
    let pid = (((packet[1] & 0x1F) as u16) << 8) | packet[2] as u16;
    let b = &packet[6..12];
    let base = ((b[0] as u64) << 25)
        | ((b[1] as u64) << 17)
        | ((b[2] as u64) << 9)
        | ((b[3] as u64) << 1)
        | ((b[4] as u64) >> 7);
    Some((pid, base))
}

/// Find a packet boundary in `data`, confirmed by the sync bytes that should
/// follow it.
///
/// A lone 0x47 is worth nothing — it is a perfectly ordinary payload byte, and
/// locking onto one mid-packet desynchronises the parser for the rest of the
/// stream. This checks up to three packets ahead, using as many as the data
/// actually contains (a 200-byte read can only confirm one).
fn find_sync(data: &[u8]) -> Option<usize> {
    const CONFIRM: usize = 3;
    for start in 0..data.len() {
        if data[start] != 0x47 {
            continue;
        }
        let confirmed = (1..CONFIRM).all(|k| {
            let at = start + k * TS_PACKET_SIZE;
            // Beyond the data is "not contradicted", not "confirmed" — with
            // short reads there is nothing else to go on.
            at >= data.len() || data[at] == 0x47
        });
        if confirmed {
            return Some(start);
        }
    }
    None
}

// ============================================================================
// Stream Types
// ============================================================================

/// MPEG-TS stream type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TsStreamType {
    /// H.264/AVC video.
    H264,
    /// H.265/HEVC video.
    H265,
    /// MPEG-2 video.
    Mpeg2Video,
    /// AAC audio (ADTS).
    AacAdts,
    /// AAC audio (LATM).
    AacLatm,
    /// MPEG audio (layer 1/2/3).
    MpegAudio,
    /// AC-3 audio.
    Ac3,
    /// Private data stream.
    PrivateData,
    /// Unknown or unsupported stream type.
    Unknown(u8),
}

impl From<StreamType> for TsStreamType {
    fn from(st: StreamType) -> Self {
        match st {
            StreamType::H264 => TsStreamType::H264,
            StreamType::H265 => TsStreamType::H265,
            StreamType::H262 => TsStreamType::Mpeg2Video,
            StreamType::ADTS => TsStreamType::AacAdts,
            StreamType::LATM => TsStreamType::AacLatm,
            StreamType::ISO_11172_AUDIO | StreamType::ISO_138183_AUDIO => TsStreamType::MpegAudio,
            StreamType::H222_0_PES_PRIVATE_DATA => TsStreamType::PrivateData,
            other => TsStreamType::Unknown(other.0),
        }
    }
}

impl TsStreamType {
    /// Returns true if this is a video stream type.
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            TsStreamType::H264 | TsStreamType::H265 | TsStreamType::Mpeg2Video
        )
    }

    /// Returns true if this is an audio stream type.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            TsStreamType::AacAdts
                | TsStreamType::AacLatm
                | TsStreamType::MpegAudio
                | TsStreamType::Ac3
        )
    }
}

// ============================================================================
// Elementary Stream Frame
// ============================================================================

/// A frame extracted from an elementary stream.
#[derive(Debug)]
pub struct TsFrame {
    /// The buffer containing the frame data.
    pub buffer: Buffer,
    /// The PID of the elementary stream.
    pub pid: u16,
    /// The stream type.
    pub stream_type: TsStreamType,
    /// Presentation timestamp (if available).
    pub pts: Option<ClockTime>,
    /// Decode timestamp (if available).
    pub dts: Option<ClockTime>,
}

// ============================================================================
// Program Information
// ============================================================================

/// Information about a program in the transport stream.
#[derive(Debug, Clone)]
pub struct TsProgram {
    /// Program number.
    pub program_number: u16,
    /// PMT PID.
    pub pmt_pid: u16,
    /// Elementary streams in this program.
    pub streams: Vec<TsElementaryStream>,
}

/// Information about an elementary stream.
#[derive(Debug, Clone)]
pub struct TsElementaryStream {
    /// Elementary stream PID.
    pub pid: u16,
    /// Stream type.
    pub stream_type: TsStreamType,
    /// Original MPEG-TS stream type code.
    pub stream_type_code: u8,
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for the TS demuxer.
#[derive(Debug, Clone, Default)]
pub struct TsDemuxStats {
    /// Total TS packets processed.
    pub packets_processed: u64,
    /// Total bytes processed.
    pub bytes_processed: u64,
    /// PES packets extracted.
    pub pes_packets: u64,
    /// Video frames extracted.
    pub video_frames: u64,
    /// Audio frames extracted.
    pub audio_frames: u64,
    /// Sync errors (invalid sync bytes).
    pub sync_errors: u64,
}

// ============================================================================
// Output Queue (shared between demuxer and consumers)
// ============================================================================

/// Shared output queue for extracted frames.
type OutputQueue = Arc<Mutex<VecDeque<TsFrame>>>;
type SharedStats = Arc<Mutex<TsDemuxStats>>;
/// The demuxer's per-instance output arena, shared with its frame collectors.
///
/// This used to be a process-wide `static OnceLock<SharedArena>`: every
/// `TsDemux` in the process drew from the same 64 slots and none could be
/// sized for its own stream (#95). `Arc<Mutex<..>>` because the collectors
/// live inside mpeg2ts_reader's demultiplexer state, out of reach of a plain
/// borrow.
type SharedOutput = Arc<Mutex<OutputArena>>;

// ============================================================================
// Elementary Stream Consumer Implementation
// ============================================================================

/// Consumer that collects PES data into frames.
pub struct FrameCollector {
    pid: u16,
    stream_type: TsStreamType,
    output: OutputQueue,
    stats: SharedStats,
    arena: SharedOutput,
    current_data: Vec<u8>,
    current_pts: Option<u64>,
    current_dts: Option<u64>,
}

impl FrameCollector {
    fn new(
        pid: u16,
        stream_type: TsStreamType,
        output: OutputQueue,
        stats: SharedStats,
        arena: SharedOutput,
    ) -> Self {
        Self {
            pid,
            stream_type,
            output,
            stats,
            arena,
            current_data: Vec::new(),
            current_pts: None,
            current_dts: None,
        }
    }

    fn flush_frame(&mut self) {
        if self.current_data.is_empty() {
            return;
        }

        // Create buffer from collected data
        if let Ok(buffer) = self.create_buffer() {
            let pts = self.current_pts.map(|v| {
                // PTS is in 90kHz units
                ClockTime::from_nanos((v as u128 * 1_000_000_000 / 90_000) as u64)
            });
            let dts = self
                .current_dts
                .map(|v| ClockTime::from_nanos((v as u128 * 1_000_000_000 / 90_000) as u64));

            let frame = TsFrame {
                buffer,
                pid: self.pid,
                stream_type: self.stream_type,
                pts,
                dts,
            };

            self.output.lock().unwrap().push_back(frame);
            self.stats.lock().unwrap().pes_packets += 1;

            if self.stream_type.is_video() {
                self.stats.lock().unwrap().video_frames += 1;
            } else if self.stream_type.is_audio() {
                self.stats.lock().unwrap().audio_frames += 1;
            }
        }

        self.current_data.clear();
        self.current_pts = None;
        self.current_dts = None;
    }

    fn create_buffer(&self) -> Result<Buffer> {
        let data = &self.current_data;
        if data.is_empty() {
            return Err(Error::Element("Empty buffer data".into()));
        }

        let mut arena = self.arena.lock().unwrap();
        let mut slot = arena.acquire(data.len(), "tsdemux")?;
        slot.data_mut()[..data.len()].copy_from_slice(data);

        let handle = MemoryHandle::with_len(slot, data.len());

        // Build metadata
        let mut metadata = Metadata::new();
        metadata.stream_id = self.pid as u32;

        if let Some(pts) = self.current_pts {
            metadata.pts = ClockTime::from_nanos((pts as u128 * 1_000_000_000 / 90_000) as u64);
        }
        if let Some(dts) = self.current_dts {
            metadata.dts = ClockTime::from_nanos((dts as u128 * 1_000_000_000 / 90_000) as u64);
        }

        Ok(Buffer::new(handle, metadata))
    }
}

impl<Ctx: DemuxContext> ElementaryStreamConsumer<Ctx> for FrameCollector {
    fn start_stream(&mut self, _ctx: &mut Ctx) {
        // Stream started, clear any partial data
        self.current_data.clear();
        self.current_pts = None;
        self.current_dts = None;
    }

    fn begin_packet(&mut self, _ctx: &mut Ctx, header: PesHeader<'_>) {
        // Flush previous frame if any
        self.flush_frame();

        // Extract timestamps and payload
        match header.contents() {
            PesContents::Parsed(Some(parsed)) => {
                // Extract PTS/DTS
                if let Ok(pts_dts) = parsed.pts_dts() {
                    match pts_dts {
                        pes::PtsDts::PtsOnly(Ok(pts)) => {
                            self.current_pts = Some(pts.value());
                        }
                        pes::PtsDts::Both {
                            pts: Ok(pts),
                            dts: Ok(dts),
                        } => {
                            self.current_pts = Some(pts.value());
                            self.current_dts = Some(dts.value());
                        }
                        _ => {}
                    }
                }

                // Append payload
                self.current_data.extend_from_slice(parsed.payload());
            }
            PesContents::Parsed(None) => {
                // No parsed content
            }
            PesContents::Payload(payload) => {
                // Raw payload without header
                self.current_data.extend_from_slice(payload);
            }
        }
    }

    fn continue_packet(&mut self, _ctx: &mut Ctx, data: &[u8]) {
        // Continuation of PES packet
        self.current_data.extend_from_slice(data);
    }

    fn end_packet(&mut self, _ctx: &mut Ctx) {
        // Packet complete, flush the frame
        self.flush_frame();
    }

    fn continuity_error(&mut self, _ctx: &mut Ctx) {
        // Continuity error, discard partial data
        self.current_data.clear();
        self.current_pts = None;
        self.current_dts = None;
    }
}

// ============================================================================
// Packet Filter Switch
// ============================================================================

/// Packet filter for handling different PID types.
pub enum TsPacketFilter {
    /// PAT filter.
    Pat(PatPacketFilter<TsDemuxContext>),
    /// PMT filter.
    Pmt(PmtPacketFilter<TsDemuxContext>),
    /// PES filter for elementary streams.
    Pes(pes::PesPacketFilter<TsDemuxContext, FrameCollector>),
    /// Null filter for ignored streams.
    Null(NullPacketFilter<TsDemuxContext>),
}

impl PacketFilter for TsPacketFilter {
    type Ctx = TsDemuxContext;

    fn consume(&mut self, ctx: &mut Self::Ctx, pk: &mpeg2ts_reader::packet::Packet<'_>) {
        match self {
            TsPacketFilter::Pat(f) => f.consume(ctx, pk),
            TsPacketFilter::Pmt(f) => f.consume(ctx, pk),
            TsPacketFilter::Pes(f) => f.consume(ctx, pk),
            TsPacketFilter::Null(f) => f.consume(ctx, pk),
        }
    }
}

// ============================================================================
// Demux Context
// ============================================================================

/// Context for the TS demuxer.
pub struct TsDemuxContext {
    /// Output queue for extracted frames.
    output: OutputQueue,
    /// Statistics.
    stats: SharedStats,
    /// Per-instance output arena, handed to each new frame collector.
    arena: SharedOutput,
    /// Stream type filter (None = accept all).
    stream_filter: Option<Vec<TsStreamType>>,
    /// Filter changeset for dynamic filter updates.
    changeset: FilterChangeset<TsPacketFilter>,
}

impl TsDemuxContext {
    fn new(output: OutputQueue, stats: SharedStats, arena: SharedOutput) -> Self {
        Self {
            output,
            stats,
            arena,
            stream_filter: None,
            changeset: FilterChangeset::default(),
        }
    }

    fn with_filter(
        output: OutputQueue,
        stats: SharedStats,
        arena: SharedOutput,
        filter: Vec<TsStreamType>,
    ) -> Self {
        Self {
            output,
            stats,
            arena,
            stream_filter: Some(filter),
            changeset: FilterChangeset::default(),
        }
    }

    fn should_handle_stream(&self, stream_type: TsStreamType) -> bool {
        match &self.stream_filter {
            Some(filter) => filter.contains(&stream_type),
            None => true,
        }
    }
}

impl DemuxContext for TsDemuxContext {
    type F = TsPacketFilter;

    fn filter_changeset(&mut self) -> &mut FilterChangeset<Self::F> {
        &mut self.changeset
    }

    fn construct(&mut self, req: FilterRequest<'_, '_>) -> Self::F {
        match req {
            FilterRequest::ByPid(PAT_PID) => TsPacketFilter::Pat(PatPacketFilter::default()),
            FilterRequest::ByPid(mpeg2ts_reader::STUFFING_PID) => {
                TsPacketFilter::Null(NullPacketFilter::default())
            }
            FilterRequest::ByPid(_) => TsPacketFilter::Null(NullPacketFilter::default()),
            FilterRequest::ByStream {
                stream_type,
                stream_info,
                ..
            } => {
                let ts_type: TsStreamType = stream_type.into();

                if self.should_handle_stream(ts_type) {
                    let collector = FrameCollector::new(
                        stream_info.elementary_pid().into(),
                        ts_type,
                        self.output.clone(),
                        self.stats.clone(),
                        self.arena.clone(),
                    );
                    TsPacketFilter::Pes(pes::PesPacketFilter::new(collector))
                } else {
                    TsPacketFilter::Null(NullPacketFilter::default())
                }
            }
            FilterRequest::Pmt {
                pid,
                program_number,
            } => TsPacketFilter::Pmt(PmtPacketFilter::new(pid, program_number)),
            FilterRequest::Nit { .. } => TsPacketFilter::Null(NullPacketFilter::default()),
        }
    }
}

// ============================================================================
// TsDemux Element
// ============================================================================

/// MPEG Transport Stream demultiplexer.
///
/// Extracts elementary streams (video, audio) from MPEG-TS data.
pub struct TsDemux {
    /// The underlying demultiplexer.
    demux: demultiplex::Demultiplex<TsDemuxContext>,
    /// The demux context (must be kept alive and passed to push).
    ctx: TsDemuxContext,
    /// Output queue for extracted frames.
    output: OutputQueue,
    /// Statistics.
    stats: SharedStats,
    /// Per-instance output arena (see [`SharedOutput`]).
    arena: SharedOutput,
    /// Partial packet buffer for handling non-aligned input.
    partial_packet: Vec<u8>,
    /// Absolute byte offset of the next byte to be consumed, in the *source's*
    /// address space. Advanced by `push`, re-anchored by
    /// [`set_stream_position`](Self::set_stream_position) after a seek — which
    /// is the only reason it is not just `stats.bytes_processed`.
    stream_pos: u64,
    /// Time-to-byte index built from the PCR clock; see [`TsByteIndex`].
    index: TsByteIndex,
}

impl TsDemux {
    /// Create a new TS demuxer.
    pub fn new() -> Self {
        let output = Arc::new(Mutex::new(VecDeque::new()));
        let stats = Arc::new(Mutex::new(TsDemuxStats::default()));
        let arena = Arc::new(Mutex::new(
            OutputArena::new(defaults::TS_DEMUX_SLOT_COUNT)
                .with_min_slot_size(defaults::TS_DEMUX_SLOT_SIZE)
                .grow_to_fit(),
        ));
        let mut ctx = TsDemuxContext::new(output.clone(), stats.clone(), arena.clone());
        let demux = demultiplex::Demultiplex::new(&mut ctx);

        Self {
            demux,
            ctx,
            output,
            stats,
            arena,
            partial_packet: Vec::new(),
            stream_pos: 0,
            index: TsByteIndex::default(),
        }
    }

    /// Create a demuxer that only extracts specific stream types.
    pub fn with_stream_filter(stream_types: Vec<TsStreamType>) -> Self {
        let output = Arc::new(Mutex::new(VecDeque::new()));
        let stats = Arc::new(Mutex::new(TsDemuxStats::default()));
        let arena = Arc::new(Mutex::new(
            OutputArena::new(defaults::TS_DEMUX_SLOT_COUNT)
                .with_min_slot_size(defaults::TS_DEMUX_SLOT_SIZE)
                .grow_to_fit(),
        ));
        let mut ctx =
            TsDemuxContext::with_filter(output.clone(), stats.clone(), arena.clone(), stream_types);
        let demux = demultiplex::Demultiplex::new(&mut ctx);

        Self {
            demux,
            ctx,
            output,
            stats,
            arena,
            partial_packet: Vec::new(),
            stream_pos: 0,
            index: TsByteIndex::default(),
        }
    }

    /// Create a demuxer for video streams only.
    pub fn video_only() -> Self {
        Self::with_stream_filter(vec![
            TsStreamType::H264,
            TsStreamType::H265,
            TsStreamType::Mpeg2Video,
        ])
    }

    /// Create a demuxer for audio streams only.
    pub fn audio_only() -> Self {
        Self::with_stream_filter(vec![
            TsStreamType::AacAdts,
            TsStreamType::AacLatm,
            TsStreamType::MpegAudio,
            TsStreamType::Ac3,
        ])
    }

    /// Size this demuxer's output arena from the graph below it.
    ///
    /// Called by the executor through [`TsDemuxElement`]; standalone users
    /// fall back to [`defaults::TS_DEMUX_SLOT_COUNT`].
    pub fn set_output_budget(&mut self, budget: OutputBudget) {
        self.arena.lock().unwrap().set_budget(budget);
    }

    /// Get current statistics.
    pub fn stats(&self) -> TsDemuxStats {
        self.stats.lock().unwrap().clone()
    }

    /// Push TS data into the demuxer.
    ///
    /// Returns extracted frames. Input data can be any size; the demuxer
    /// handles packet boundary alignment internally.
    pub fn push(&mut self, data: &[u8]) -> Result<Vec<TsFrame>> {
        // Combine with any partial packet from previous push
        let to_process = if self.partial_packet.is_empty() {
            data.to_vec()
        } else {
            let mut combined = std::mem::take(&mut self.partial_packet);
            combined.extend_from_slice(data);
            combined
        };

        // Find a *confirmed* packet boundary: a lone 0x47 is an ordinary
        // payload byte, and locking onto one desynchronises the rest of the
        // stream. With nothing confirmable, keep the data as partial and wait
        // for more rather than guessing.
        let start = match find_sync(&to_process) {
            Some(start) => start,
            None => {
                if !to_process.is_empty() {
                    self.stats.lock().unwrap().sync_errors += 1;
                }
                // Anything before the last possible boundary is unusable.
                let keep = to_process.len().saturating_sub(TS_PACKET_SIZE * 3);
                self.stream_pos += keep as u64;
                self.partial_packet = to_process[keep..].to_vec();
                return Ok(self.output.lock().unwrap().drain(..).collect());
            }
        };
        if start > 0 {
            self.stats.lock().unwrap().sync_errors += 1;
        }

        let aligned = &to_process[start..];
        self.stream_pos += start as u64;

        // Calculate how many complete packets we have
        let complete_packets = aligned.len() / TS_PACKET_SIZE;
        let complete_bytes = complete_packets * TS_PACKET_SIZE;

        if complete_bytes > 0 {
            // Index before demuxing: the PCR sits in the adaptation field and
            // needs no PSI, so this works from the very first packet — before
            // the parser has even seen a PAT.
            for i in 0..complete_packets {
                let at = i * TS_PACKET_SIZE;
                self.index.observe(
                    &aligned[at..at + TS_PACKET_SIZE],
                    self.stream_pos + at as u64,
                );
            }
            // Process complete packets
            self.demux.push(&mut self.ctx, &aligned[..complete_bytes]);
            self.stats.lock().unwrap().packets_processed += complete_packets as u64;
            self.stats.lock().unwrap().bytes_processed += complete_bytes as u64;
            self.stream_pos += complete_bytes as u64;
        }

        // Save remaining partial packet
        if complete_bytes < aligned.len() {
            self.partial_packet = aligned[complete_bytes..].to_vec();
        }

        // Collect extracted frames
        let frames: Vec<TsFrame> = self.output.lock().unwrap().drain(..).collect();
        Ok(frames)
    }

    /// Flush any remaining partial data.
    ///
    /// Call this at end of stream to ensure all frames are extracted.
    pub fn flush(&mut self) -> Vec<TsFrame> {
        self.partial_packet.clear();
        self.output.lock().unwrap().drain(..).collect()
    }

    /// The time-to-byte index built so far; see [`TsByteIndex`].
    pub fn index(&self) -> &TsByteIndex {
        &self.index
    }

    /// Estimate the byte offset holding `target` of stream time.
    ///
    /// Shorthand for [`TsByteIndex::estimate_byte_offset`]; `None` when the
    /// index is too thin to answer.
    pub fn estimate_byte_offset(&self, target: ClockTime) -> Option<u64> {
        self.index.estimate_byte_offset(target.nanos())
    }

    /// Tell the demuxer where in the source the next pushed byte comes from.
    ///
    /// Called after a seek, from the source's byte-format segment. Without it
    /// every anchor recorded after the seek would be placed at the wrong
    /// offset and the index would degrade with every seek instead of
    /// improving.
    pub fn set_stream_position(&mut self, offset: u64) {
        self.stream_pos = offset;
    }

    /// The absolute offset of the next byte this demuxer expects.
    pub fn stream_position(&self) -> u64 {
        self.stream_pos
    }

    /// Record the stream's total size (from a byte segment), so byte
    /// estimates can be clamped to it.
    pub fn set_stream_total(&mut self, total: Option<u64>) {
        self.index.set_total(total);
    }

    /// Rebuild the parser without touching statistics or the byte index.
    ///
    /// What a flush needs: half-assembled access units and PSI state are
    /// invalid after a seek, but the index built from the bytes already read
    /// is exactly what makes the *next* seek possible, and counters are not
    /// timeline state.
    pub fn reset_parser(&mut self) {
        self.partial_packet.clear();
        self.output.lock().unwrap().clear();
        self.rebuild();
    }

    /// Reset the demuxer state.
    ///
    /// Rebuilds the parser (PAT/PMT included), so the caller must expect
    /// nothing to come out until the next PSI in the stream. Also clears the
    /// statistics and the byte index — use [`reset_parser`](Self::reset_parser)
    /// for a flush, which must keep both.
    ///
    /// The stream filter is carried across: recreating the context
    /// unconditionally with `TsDemuxContext::new` silently turned a
    /// `video_only()` demuxer into one that accepted everything.
    pub fn reset(&mut self) {
        self.partial_packet.clear();
        self.output.lock().unwrap().clear();
        *self.stats.lock().unwrap() = TsDemuxStats::default();
        self.index = TsByteIndex::default();
        self.stream_pos = 0;
        self.rebuild();
    }

    /// Recreate the parser context, preserving the filter this demuxer was
    /// built with.
    fn rebuild(&mut self) {
        self.ctx = match self.ctx.stream_filter.clone() {
            Some(filter) => TsDemuxContext::with_filter(
                self.output.clone(),
                self.stats.clone(),
                self.arena.clone(),
                filter,
            ),
            None => {
                TsDemuxContext::new(self.output.clone(), self.stats.clone(), self.arena.clone())
            }
        };
        self.demux = demultiplex::Demultiplex::new(&mut self.ctx);
    }
}

impl Default for TsDemux {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

// ============================================================================
// TsDemuxElement (pipeline Demuxer)
// ============================================================================

/// [`TsDemux`] as a pipeline element: 1 TS-byte input, routed A/V outputs.
///
/// Feed it from any byte source (`filesrc ! tsdemux`-style, programmatically):
///
/// ```rust,ignore
/// let src = pipeline.add_source("src", FileSrc::new("capture.ts"));
/// let demux = pipeline.add_demuxer("tsdemux", TsDemuxElement::new());
/// pipeline.link(src, demux)?;
/// pipeline.link_pads(demux, "video", video_branch, "sink")?;
/// pipeline.link_pads(demux, "audio", audio_branch, "sink")?;
/// ```
///
/// Video PES streams route to the `"video"` pad, audio to `"audio"`, anything
/// else to `"data"`; a pad with no links drops its frames (the executor warns).
/// Frame metadata carries the PID as `stream_id` plus PTS/DTS.
pub struct TsDemuxElement {
    demux: TsDemux,
    outputs: Vec<(crate::element::PadId, crate::format::Caps)>,
    /// EOS drain state: the executor calls `produce()` until Eos.
    drained: bool,
    /// In-flight ACCURATE seek being iterated on (#173).
    refine: Option<RefineState>,
    /// A corrected seek staged for the executor's `take_upstream_event` poll.
    refined_out: Option<Event>,
}

/// State of an [`SeekFlags::ACCURATE`] seek across refinement rounds (#173).
///
/// A single-shot linear estimate is exact for CBR and approximate for VBR.
/// With ACCURATE set, the demuxer keeps the seek in flight after forwarding:
/// each flush re-arms `awaiting_pts`, the first post-flush PTS is compared
/// against the target, and a miss beyond [`TsDemuxElement::REFINE_THRESHOLD`]
/// forwards a corrected BYTES seek — same seqnum, one refinement round
/// deeper — re-estimated from an index that now holds anchors observed at
/// the mis-landing. Bounded by `rounds_left`; the last landing is reported
/// regardless.
///
/// [`SeekFlags::ACCURATE`]: crate::event::SeekFlags::ACCURATE
#[derive(Debug)]
struct RefineState {
    /// The last BYTES seek forwarded (carries seqnum + current round).
    seek: SeekEvent,
    /// The TIME target the application asked for.
    target: ClockTime,
    /// Byte offset of the last estimate, to detect a stalled index.
    last_offset: u64,
    /// Corrections still allowed.
    rounds_left: u8,
    /// Armed by each flush: the next PTS seen is a landing to judge.
    awaiting_pts: bool,
}

impl TsDemuxElement {
    /// Wrap a fresh [`TsDemux`].
    pub fn new() -> Self {
        Self::with_demux(TsDemux::new())
    }

    /// Wrap a configured [`TsDemux`] (stream filters etc.).
    pub fn with_demux(demux: TsDemux) -> Self {
        use crate::element::PadId;
        use crate::format::Caps;
        Self {
            demux,
            outputs: vec![
                (PadId(0), Caps::any()),
                (PadId(1), Caps::any()),
                (PadId(2), Caps::any()),
            ],
            drained: false,
            refine: None,
            refined_out: None,
        }
    }

    /// The wrapped demuxer.
    pub fn demux(&self) -> &TsDemux {
        &self.demux
    }

    fn pad_for(stream_type: TsStreamType) -> crate::element::PadId {
        use crate::element::PadId;
        if stream_type.is_video() {
            PadId(0)
        } else if stream_type.is_audio() {
            PadId(1)
        } else {
            PadId(2)
        }
    }

    fn route(frames: Vec<TsFrame>) -> crate::element::RoutedOutput {
        let mut routed = crate::element::RoutedOutput::new();
        for frame in frames {
            routed.push(Self::pad_for(frame.stream_type), frame.buffer);
        }
        routed
    }
}

impl Default for TsDemuxElement {
    fn default() -> Self {
        Self::new()
    }
}

impl TsDemuxElement {
    /// A landing this far off an ACCURATE target triggers a correction.
    ///
    /// Half a second: comfortably above the jitter between a byte offset and
    /// the first PES PTS behind it (a GOP is typically shorter), and small
    /// enough that a VBR miss of seconds is always corrected.
    const REFINE_THRESHOLD: ClockTime = ClockTime::from_millis(500);
    /// Corrections per seek. Each costs a full flush round trip through the
    /// executor; past this the landing is reported as-is.
    const MAX_REFINE_ROUNDS: u8 = 3;

    /// Judge the first post-flush landing of an in-flight ACCURATE seek and
    /// stage a corrected BYTES seek when it missed (#173).
    fn maybe_refine(&mut self, frames: &[TsFrame]) {
        let Some(state) = &mut self.refine else {
            return;
        };
        if !state.awaiting_pts {
            return;
        }
        let Some(pts) = frames.iter().find_map(|f| f.pts) else {
            return;
        };
        state.awaiting_pts = false;

        let err = ClockTime::from_nanos(pts.nanos().abs_diff(state.target.nanos()));
        if err <= Self::REFINE_THRESHOLD || state.rounds_left == 0 {
            tracing::debug!(
                "tsdemux: ACCURATE seek landed at {pts} for target {} (err {err}), done",
                state.target
            );
            self.refine = None;
            return;
        }
        // The index has been improving the whole time: the mis-landed round's
        // packets were observed at their true offsets, so the anchors now
        // bracket (or closely precede) the target.
        let next = self.demux.estimate_byte_offset(state.target);
        match next {
            Some(offset) if offset != state.last_offset => {
                state.rounds_left -= 1;
                state.last_offset = offset;
                let refined = state
                    .seek
                    .derive_refined(SegmentFormat::Bytes, SeekPosition::set(offset as i64));
                tracing::debug!(
                    "tsdemux: ACCURATE seek landed at {pts} for target {} (err {err}), \
                     round {} corrects to byte {offset}",
                    state.target,
                    refined.refine_round(),
                );
                state.seek = refined.clone();
                self.refined_out = Some(Event::Seek(refined));
            }
            _ => {
                // Same estimate again (or none): the index can do no better,
                // and repeating the seek would loop on the same landing.
                tracing::debug!(
                    "tsdemux: ACCURATE seek stuck at {pts} for target {} (err {err}), \
                     index cannot improve — reporting the landing",
                    state.target
                );
                self.refine = None;
            }
        }
    }
}

impl crate::element::Demuxer for TsDemuxElement {
    fn demux(&mut self, buffer: Buffer) -> Result<crate::element::RoutedOutput> {
        let frames = self.demux.push(buffer.as_bytes())?;
        self.maybe_refine(&frames);
        Ok(Self::route(frames))
    }

    fn produce(&mut self) -> Result<crate::element::DemuxerProduce> {
        // Reached at EOS: drain the trailing partial frames once, keeping
        // their routing, then report end of stream.
        if self.drained {
            return Ok(crate::element::DemuxerProduce::Eos);
        }
        self.drained = true;
        Ok(crate::element::DemuxerProduce::Routed(Self::route(
            self.demux.flush(),
        )))
    }

    fn pad_name(&self, pad: crate::element::PadId) -> String {
        match pad.0 {
            0 => "video".into(),
            1 => "audio".into(),
            _ => "data".into(),
        }
    }

    fn outputs(&self) -> &[(crate::element::PadId, crate::format::Caps)] {
        &self.outputs
    }

    fn on_pad_added(&mut self, _callback: crate::element::PadAddedCallback) {}

    fn name(&self) -> &str {
        "tsdemux"
    }

    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.demux.set_output_budget(budget);
    }

    /// TIME in, BYTES out — the whole reason this element participates in
    /// seeking at all. Declared so `filesrc ! tsdemux` reports itself
    /// TIME-seekable before anyone tries.
    fn seek_translations(&self) -> Vec<crate::pipeline::seek::SeekTranslation> {
        vec![crate::pipeline::seek::SeekTranslation {
            from: crate::event::SegmentFormat::Time,
            to: crate::event::SegmentFormat::Bytes,
            // A fed TS demuxer never learns the duration: it sees a byte
            // stream from wherever the source happens to be, and the last PCR
            // it has seen is a floor, not a total.
            duration: None,
        }]
    }

    /// Translate a TIME seek into a BYTES seek on the source.
    ///
    /// Refuses (leaves the event to travel further upstream) when the index
    /// cannot answer — under ~200 ms of stream seen, or a stream with no
    /// usable PCR. Inventing an offset would land the source in the middle of
    /// a packet and desynchronise the parse.
    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        let Event::Seek(seek) = event else {
            return EventResult::NotHandled;
        };
        if seek.format != SegmentFormat::Time {
            return EventResult::NotHandled;
        }
        // Only absolute targets: Current/End-relative would need a position
        // and a duration this element does not have.
        if seek.start.seek_type != SeekType::Set {
            return EventResult::NotHandled;
        }
        let target = ClockTime::from_nanos(seek.start.position.max(0) as u64);
        let Some(offset) = self.demux.estimate_byte_offset(target) else {
            tracing::debug!(
                "tsdemux: byte index too thin ({} anchors) to place {target}",
                self.demux.index().len()
            );
            return EventResult::NotHandled;
        };
        tracing::debug!("tsdemux: TIME {target} estimated at byte {offset}");
        let derived = seek.derive(SegmentFormat::Bytes, SeekPosition::set(offset as i64));
        // ACCURATE (#173): keep the seek in flight — the first post-flush PTS
        // is judged in `maybe_refine`, which stages corrections through
        // `take_upstream_event`. A new seek (any flavour) supersedes an old
        // refinement: its epoch outranks every round of the previous seqnum.
        self.refine = if seek.flags.contains(SeekFlags::ACCURATE) {
            Some(RefineState {
                seek: derived.clone(),
                target,
                last_offset: offset,
                rounds_left: Self::MAX_REFINE_ROUNDS,
                awaiting_pts: false,
            })
        } else {
            None
        };
        self.refined_out = None;
        EventResult::forward(Event::Seek(derived))
    }

    /// Learn where the source landed, and how big the stream is.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        if let Event::Segment(seg) = &event
            && seg.format == SegmentFormat::Bytes
        {
            self.demux.set_stream_position(seg.start.max(0) as u64);
            self.demux
                .set_stream_total(u64::try_from(seg.stop).ok().filter(|_| seg.stop >= 0));
        }
        Some(event)
    }

    /// Drop the half-assembled access unit and the PSI state, keeping the
    /// byte index — that index is what makes the *next* seek possible.
    fn flush(&mut self) -> Result<crate::element::RoutedOutput> {
        let routed = Self::route(self.demux.flush());
        self.demux.reset_parser();
        // A post-seek EOS must drain again.
        self.drained = false;
        // Each flush starts a fresh landing: the next PTS judges it (#173).
        if let Some(state) = &mut self.refine {
            state.awaiting_pts = true;
        }
        Ok(routed)
    }

    fn take_upstream_event(&mut self) -> Option<Event> {
        self.refined_out.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ts_demux_creation() {
        let demux = TsDemux::new();
        assert_eq!(demux.stats().packets_processed, 0);
    }

    #[test]
    fn test_ts_demux_video_only() {
        let demux = TsDemux::video_only();
        assert_eq!(demux.stats().video_frames, 0);
    }

    #[test]
    fn test_ts_demux_audio_only() {
        let demux = TsDemux::audio_only();
        assert_eq!(demux.stats().audio_frames, 0);
    }

    #[test]
    fn test_ts_stream_type_classification() {
        assert!(TsStreamType::H264.is_video());
        assert!(TsStreamType::H265.is_video());
        assert!(!TsStreamType::H264.is_audio());

        assert!(TsStreamType::AacAdts.is_audio());
        assert!(TsStreamType::MpegAudio.is_audio());
        assert!(!TsStreamType::AacAdts.is_video());
    }

    #[test]
    fn test_ts_demux_sync_error_handling() {
        let mut demux = TsDemux::new();

        // Feed invalid data (no sync byte)
        let invalid_data = vec![0x00; 188];
        let frames = demux.push(&invalid_data).unwrap();

        assert!(frames.is_empty());
        assert!(demux.stats().sync_errors > 0);
    }

    #[test]
    fn test_ts_demux_partial_packet() {
        let mut demux = TsDemux::new();

        // Feed partial packet (starts with sync but less than 188 bytes)
        let mut partial = vec![0x47];
        partial.extend_from_slice(&[0x00; 99]);
        let frames = demux.push(&partial).unwrap();

        assert!(frames.is_empty());
        assert_eq!(demux.stats().packets_processed, 0);
    }

    #[test]
    fn test_ts_demux_stats_default() {
        let stats = TsDemuxStats::default();
        assert_eq!(stats.packets_processed, 0);
        assert_eq!(stats.bytes_processed, 0);
        assert_eq!(stats.video_frames, 0);
        assert_eq!(stats.audio_frames, 0);
    }

    #[test]
    fn test_ts_demux_reset() {
        let mut demux = TsDemux::new();

        // Feed some data (even invalid)
        let _ = demux.push(&[0x00; 200]);

        // Reset
        demux.reset();

        assert_eq!(demux.stats().packets_processed, 0);
        assert_eq!(demux.stats().sync_errors, 0);
    }

    /// A reset must not widen what the demuxer accepts. `reset()` rebuilds
    /// the parsing context, and rebuilding it unconditionally with
    /// `TsDemuxContext::new` turned a `video_only()` demuxer into one that
    /// accepted audio too — which matters because a seek resets the parser.
    #[test]
    fn reset_preserves_the_stream_filter() {
        let mut demux = TsDemux::video_only();
        let before = demux.ctx.stream_filter.clone();
        assert!(before.is_some(), "video_only() must install a filter");

        demux.reset();

        assert_eq!(
            demux.ctx.stream_filter, before,
            "reset dropped the stream filter"
        );

        // And an unfiltered demuxer stays unfiltered.
        let mut all = TsDemux::new();
        all.reset();
        assert!(all.ctx.stream_filter.is_none());
    }

    // ========================================================================
    // Byte index / sync (#163 phase B)
    // ========================================================================

    /// Build a 188-byte packet on `pid` carrying `pcr` (90 kHz ticks) in its
    /// adaptation field. Payload is 0xFF so a stray 0x47 can never appear.
    fn pcr_packet(pid: u16, pcr: u64) -> Vec<u8> {
        let mut p = vec![0xFFu8; TS_PACKET_SIZE];
        p[0] = 0x47;
        p[1] = ((pid >> 8) as u8) & 0x1F;
        p[2] = (pid & 0xFF) as u8;
        p[3] = 0x20; // adaptation field only
        p[4] = (TS_PACKET_SIZE - 5) as u8;
        p[5] = 0x10; // PCR_flag
        p[6] = (pcr >> 25) as u8;
        p[7] = (pcr >> 17) as u8;
        p[8] = (pcr >> 9) as u8;
        p[9] = (pcr >> 1) as u8;
        p[10] = ((pcr & 1) as u8) << 7;
        p[11] = 0;
        p
    }

    /// A packet with no adaptation field, so no PCR.
    fn plain_packet(pid: u16) -> Vec<u8> {
        let mut p = vec![0xFFu8; TS_PACKET_SIZE];
        p[0] = 0x47;
        p[1] = ((pid >> 8) as u8) & 0x1F;
        p[2] = (pid & 0xFF) as u8;
        p[3] = 0x10; // payload only
        p
    }

    #[test]
    fn pcr_is_parsed_from_the_adaptation_field() {
        // 1 second at 90 kHz.
        let pkt = pcr_packet(0x100, 90_000);
        assert_eq!(packet_pcr(&pkt), Some((0x100, 90_000)));
        assert_eq!(packet_pcr(&plain_packet(0x100)), None);
        // A 33-bit PCR near the wrap point survives the bit shuffling.
        let big = (1u64 << 33) - 1;
        assert_eq!(packet_pcr(&pcr_packet(0x1FFF, big)), Some((0x1FFF, big)));
    }

    #[test]
    fn sync_needs_confirmation_from_the_packets_that_follow() {
        // A lone 0x47 in the middle of payload is not a packet boundary.
        let mut data = vec![0x00u8; 4];
        data.push(0x47); // decoy at 4, contradicted 188 bytes later
        data.extend(vec![0x00u8; TS_PACKET_SIZE * 3]);
        // The real stream starts after it.
        let real = data.len();
        for _ in 0..3 {
            data.extend(plain_packet(0x100));
        }
        assert_eq!(find_sync(&data), Some(real));

        // With too little data to contradict anything, the first 0x47 wins —
        // there is nothing else to go on.
        assert_eq!(find_sync(&[0x00, 0x47, 0x00]), Some(1));
        assert_eq!(find_sync(&[0x00, 0x01, 0x02]), None);
    }

    #[test]
    fn a_decoy_sync_byte_no_longer_desynchronises_the_parse() {
        let mut demux = TsDemux::new();
        // A 0x47 inside 100 bytes of leading junk, deliberately *off* the real
        // packet grid. The old "first 0x47 wins" rule locked onto it and every
        // packet after it was parsed 97 bytes out of phase.
        let mut data = vec![0x00u8; 100];
        data[3] = 0x47;
        for second in 0..3u64 {
            data.extend(pcr_packet(0x100, second * 90_000));
        }
        demux.push(&data).unwrap();

        assert_eq!(demux.stats().sync_errors, 1, "the junk prefix is reported");
        assert_eq!(demux.stats().packets_processed, 3);
        assert_eq!(
            demux.stream_position(),
            100 + 3 * TS_PACKET_SIZE as u64,
            "consumed the junk and exactly three packets"
        );
        // The clincher: a misaligned parse finds no PCR at all, because the
        // adaptation-field bits land on payload.
        assert_eq!(demux.index().len(), 3, "PCRs were found, so alignment held");
    }

    #[test]
    fn the_index_maps_time_to_bytes_from_pcr() {
        let mut demux = TsDemux::new();
        // 10 seconds of CBR: one PCR packet per second, 1000 packets apart.
        for second in 0..10u64 {
            let mut chunk = pcr_packet(0x100, second * 90_000);
            for _ in 0..999 {
                chunk.extend(plain_packet(0x100));
            }
            demux.push(&chunk).unwrap();
        }
        assert_eq!(demux.index().len(), 10, "one anchor per second");

        // Second 5 sits 5000 packets in, exactly.
        let at5 = demux
            .estimate_byte_offset(ClockTime::from_secs(5))
            .expect("index has 10 anchors");
        assert_eq!(at5, 5000 * TS_PACKET_SIZE as u64);

        // Between anchors, linearly.
        let at_2s5 = demux
            .estimate_byte_offset(ClockTime::from_millis(2500))
            .unwrap();
        assert_eq!(at_2s5, 2500 * TS_PACKET_SIZE as u64);

        // Before the start clamps to the first anchor; every answer is on the
        // packet grid.
        assert_eq!(demux.estimate_byte_offset(ClockTime::ZERO), Some(0));
        for target in [0, 1, 1234, 9_999_999_999u64] {
            let off = demux
                .estimate_byte_offset(ClockTime::from_nanos(target))
                .unwrap();
            assert_eq!(off % TS_PACKET_SIZE as u64, 0, "target {target}");
        }
    }

    /// Hand-build a routed frame carrying only what `maybe_refine` reads.
    fn frame_with_pts(pts: ClockTime) -> TsFrame {
        let arena = crate::memory::SharedArena::new(64, 2).unwrap();
        let slot = arena.acquire().unwrap();
        TsFrame {
            buffer: Buffer::new(MemoryHandle::with_len(slot, 8), Metadata::from_sequence(0)),
            pid: 0x100,
            stream_type: TsStreamType::H264,
            pts: Some(pts),
            dts: None,
        }
    }

    /// #173: an ACCURATE seek that lands off-target stages a corrected BYTES
    /// seek — same seqnum, next refinement round — from an index improved by
    /// the mis-landing itself, and stops once the landing is close enough.
    #[test]
    fn an_accurate_seek_stages_corrections_until_it_lands() {
        let mut el = TsDemuxElement::new();
        use crate::element::Demuxer;

        // A misleadingly slow start: anchors at (0s, 0) and (1s, packet 100).
        // Extrapolating 5s from these lands at packet 500 — far short on a
        // stream whose real rate ramps up.
        let mut head = pcr_packet(0x100, 0);
        for _ in 0..99 {
            head.extend(plain_packet(0x100));
        }
        head.extend(pcr_packet(0x100, 90_000));
        el.demux.push(&head).unwrap();

        let target = ClockTime::from_secs(5);
        let seek = SeekEvent::new_time(target)
            .with_flags(SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::ACCURATE);
        let result = el.handle_upstream_event(&Event::Seek(seek.clone()));
        let EventResult::Forward(translated) = result else {
            panic!("expected a translated seek, got {result:?}");
        };
        let Event::Seek(bytes_seek) = &*translated else {
            panic!("expected a BYTES seek");
        };
        let first_offset = bytes_seek.start.position as u64;
        assert_eq!(first_offset, 500 * TS_PACKET_SIZE as u64);

        // The flush trio arrives; the demuxer arms itself to judge the next
        // PTS. The source's Segment reports where it actually resumed.
        el.flush().unwrap();
        el.handle_downstream_event(Event::Segment(crate::event::SegmentEvent::new_bytes(
            first_offset,
            None,
        )));
        // A PCR observed at the landing: the index learns (2s, packet 500).
        el.demux.push(&pcr_packet(0x100, 2 * 90_000)).unwrap();

        // The first post-flush frame says 2s — a 3-second miss.
        el.maybe_refine(&[frame_with_pts(ClockTime::from_secs(2))]);
        let refined = el.take_upstream_event().expect("a correction was staged");
        let Event::Seek(refined) = refined else {
            panic!("expected a refined seek");
        };
        assert_eq!(refined.seqnum(), seek.seqnum(), "same logical seek");
        assert_eq!(refined.refine_round(), 1);
        assert!(refined.epoch() > seek.epoch(), "the correction outranks");
        // Local slope at the landing: (1s, 100p) -> (2s, 500p) is 400 p/s, so
        // 5s ≈ packet 500 + 3 x 400 = 1700.
        assert_eq!(refined.start.position as u64, 1700 * TS_PACKET_SIZE as u64);
        assert!(el.take_upstream_event().is_none(), "staged exactly once");

        // Round 2 lands within the threshold: refinement ends, nothing more
        // is staged, and the executor is free to post SeekDone.
        el.flush().unwrap();
        el.maybe_refine(&[frame_with_pts(ClockTime::from_millis(4_800))]);
        assert!(el.take_upstream_event().is_none());
        assert!(el.refine.is_none(), "the seek is no longer in flight");
    }

    /// Without ACCURATE nothing changes: one shot, no in-flight state.
    #[test]
    fn a_plain_seek_stages_no_corrections() {
        let mut el = TsDemuxElement::new();
        use crate::element::Demuxer;

        let mut head = pcr_packet(0x100, 0);
        for _ in 0..99 {
            head.extend(plain_packet(0x100));
        }
        head.extend(pcr_packet(0x100, 90_000));
        el.demux.push(&head).unwrap();

        let seek = SeekEvent::new_time(ClockTime::from_secs(5));
        assert!(matches!(
            el.handle_upstream_event(&Event::Seek(seek)),
            EventResult::Forward(_)
        ));
        el.flush().unwrap();
        el.maybe_refine(&[frame_with_pts(ClockTime::from_secs(2))]);
        assert!(el.take_upstream_event().is_none());
    }

    #[test]
    fn a_thin_index_refuses_to_guess() {
        let mut demux = TsDemux::new();
        assert_eq!(demux.estimate_byte_offset(ClockTime::from_secs(1)), None);
        demux.push(&pcr_packet(0x100, 0)).unwrap();
        assert_eq!(
            demux.estimate_byte_offset(ClockTime::from_secs(1)),
            None,
            "one anchor gives a point, not a slope"
        );
    }

    #[test]
    fn estimates_are_clamped_to_the_stream_size() {
        let mut demux = TsDemux::new();
        for second in 0..3u64 {
            let mut chunk = pcr_packet(0x100, second * 90_000);
            for _ in 0..99 {
                chunk.extend(plain_packet(0x100));
            }
            demux.push(&chunk).unwrap();
        }
        let total = 300 * TS_PACKET_SIZE as u64;
        demux.set_stream_total(Some(total));
        // An hour into a 3-second file: the last packet, not past the end.
        let off = demux
            .estimate_byte_offset(ClockTime::from_secs(3600))
            .unwrap();
        assert!(off < total, "{off} must be inside a {total}-byte stream");
        assert_eq!(off, total - TS_PACKET_SIZE as u64);
    }

    #[test]
    fn a_second_pcr_pid_does_not_corrupt_the_index() {
        let mut demux = TsDemux::new();
        // Program A ticks forward; program B carries a wildly different clock
        // on another PID and must be ignored entirely.
        for second in 0..4u64 {
            let mut chunk = pcr_packet(0x100, second * 90_000);
            chunk.extend(pcr_packet(0x200, 9_000_000 + second * 90_000));
            for _ in 0..98 {
                chunk.extend(plain_packet(0x100));
            }
            demux.push(&chunk).unwrap();
        }
        assert_eq!(demux.index().len(), 4);
        assert_eq!(
            demux.estimate_byte_offset(ClockTime::from_secs(1)),
            Some(100 * TS_PACKET_SIZE as u64)
        );
    }

    #[test]
    fn a_flush_keeps_the_index_and_a_reset_drops_it() {
        let mut demux = TsDemux::new();
        for second in 0..3u64 {
            let mut chunk = pcr_packet(0x100, second * 90_000);
            for _ in 0..99 {
                chunk.extend(plain_packet(0x100));
            }
            demux.push(&chunk).unwrap();
        }
        assert_eq!(demux.index().len(), 3);

        demux.reset_parser();
        assert_eq!(
            demux.index().len(),
            3,
            "the index survives a flush — it is what makes the next seek work"
        );
        assert!(
            demux.stats().packets_processed > 0,
            "counters are not timeline state"
        );

        demux.reset();
        assert!(demux.index().is_empty(), "a full reset starts over");
    }

    #[test]
    fn post_seek_anchors_land_at_their_real_offsets() {
        let mut demux = TsDemux::new();
        // Read the first 2 seconds...
        for second in 0..2u64 {
            let mut chunk = pcr_packet(0x100, second * 90_000);
            for _ in 0..99 {
                chunk.extend(plain_packet(0x100));
            }
            demux.push(&chunk).unwrap();
        }
        // ...then the source seeks to second 8 and says where it landed.
        demux.reset_parser();
        demux.set_stream_position(800 * TS_PACKET_SIZE as u64);
        let mut chunk = pcr_packet(0x100, 8 * 90_000);
        for _ in 0..99 {
            chunk.extend(plain_packet(0x100));
        }
        demux.push(&chunk).unwrap();

        assert_eq!(demux.index().len(), 3);
        // The new anchor is placed at 800 packets, so the estimate for second
        // 4 interpolates across the gap instead of being nonsense.
        assert_eq!(
            demux.estimate_byte_offset(ClockTime::from_secs(4)),
            Some(400 * TS_PACKET_SIZE as u64)
        );
    }

    #[test]
    fn test_ts_packet_size() {
        assert_eq!(TS_PACKET_SIZE, 188);
    }

    #[test]
    fn test_ts_stream_type_from_mpeg() {
        let h264: TsStreamType = StreamType::H264.into();
        assert_eq!(h264, TsStreamType::H264);

        let h265: TsStreamType = StreamType::H265.into();
        assert_eq!(h265, TsStreamType::H265);

        let adts: TsStreamType = StreamType::ADTS.into();
        assert_eq!(adts, TsStreamType::AacAdts);
    }
}
