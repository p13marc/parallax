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
use crate::memory::SharedArena;
use crate::metadata::Metadata;

use mpeg2ts_reader::StreamType;
use mpeg2ts_reader::demultiplex::{
    self, DemuxContext, FilterChangeset, FilterRequest, NullPacketFilter, PacketFilter,
    PatPacketFilter, PmtPacketFilter,
};
use mpeg2ts_reader::pes::{self, ElementaryStreamConsumer, PesContents, PesHeader};
use mpeg2ts_reader::psi::pat::PAT_PID;

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::OnceLock;

/// Shared arena for MPEG-TS demuxer buffers.
fn ts_demux_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    // PES packets can be large for video, use generous slot size
    ARENA.get_or_init(|| SharedArena::new(2 * 1024 * 1024, 64).unwrap())
}

/// Size of a single MPEG-TS packet.
pub const TS_PACKET_SIZE: usize = 188;

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
type OutputQueue = Rc<RefCell<VecDeque<TsFrame>>>;
type SharedStats = Rc<RefCell<TsDemuxStats>>;

// ============================================================================
// Elementary Stream Consumer Implementation
// ============================================================================

/// Consumer that collects PES data into frames.
pub struct FrameCollector {
    pid: u16,
    stream_type: TsStreamType,
    output: OutputQueue,
    stats: SharedStats,
    current_data: Vec<u8>,
    current_pts: Option<u64>,
    current_dts: Option<u64>,
}

impl FrameCollector {
    fn new(pid: u16, stream_type: TsStreamType, output: OutputQueue, stats: SharedStats) -> Self {
        Self {
            pid,
            stream_type,
            output,
            stats,
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

            self.output.borrow_mut().push_back(frame);
            self.stats.borrow_mut().pes_packets += 1;

            if self.stream_type.is_video() {
                self.stats.borrow_mut().video_frames += 1;
            } else if self.stream_type.is_audio() {
                self.stats.borrow_mut().audio_frames += 1;
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

        let mut slot = ts_demux_arena()
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;

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
    /// Stream type filter (None = accept all).
    stream_filter: Option<Vec<TsStreamType>>,
    /// Filter changeset for dynamic filter updates.
    changeset: FilterChangeset<TsPacketFilter>,
}

impl TsDemuxContext {
    fn new(output: OutputQueue, stats: SharedStats) -> Self {
        Self {
            output,
            stats,
            stream_filter: None,
            changeset: FilterChangeset::default(),
        }
    }

    fn with_filter(output: OutputQueue, stats: SharedStats, filter: Vec<TsStreamType>) -> Self {
        Self {
            output,
            stats,
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
    /// Partial packet buffer for handling non-aligned input.
    partial_packet: Vec<u8>,
}

impl TsDemux {
    /// Create a new TS demuxer.
    pub fn new() -> Self {
        let output = Rc::new(RefCell::new(VecDeque::new()));
        let stats = Rc::new(RefCell::new(TsDemuxStats::default()));
        let mut ctx = TsDemuxContext::new(output.clone(), stats.clone());
        let demux = demultiplex::Demultiplex::new(&mut ctx);

        Self {
            demux,
            ctx,
            output,
            stats,
            partial_packet: Vec::new(),
        }
    }

    /// Create a demuxer that only extracts specific stream types.
    pub fn with_stream_filter(stream_types: Vec<TsStreamType>) -> Self {
        let output = Rc::new(RefCell::new(VecDeque::new()));
        let stats = Rc::new(RefCell::new(TsDemuxStats::default()));
        let mut ctx = TsDemuxContext::with_filter(output.clone(), stats.clone(), stream_types);
        let demux = demultiplex::Demultiplex::new(&mut ctx);

        Self {
            demux,
            ctx,
            output,
            stats,
            partial_packet: Vec::new(),
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

    /// Get current statistics.
    pub fn stats(&self) -> TsDemuxStats {
        self.stats.borrow().clone()
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

        // Find first sync byte
        let start = to_process
            .iter()
            .position(|&b| b == 0x47)
            .unwrap_or(to_process.len());
        if start > 0 {
            self.stats.borrow_mut().sync_errors += 1;
        }

        let aligned = &to_process[start..];

        // Calculate how many complete packets we have
        let complete_packets = aligned.len() / TS_PACKET_SIZE;
        let complete_bytes = complete_packets * TS_PACKET_SIZE;

        if complete_bytes > 0 {
            // Process complete packets
            self.demux.push(&mut self.ctx, &aligned[..complete_bytes]);
            self.stats.borrow_mut().packets_processed += complete_packets as u64;
            self.stats.borrow_mut().bytes_processed += complete_bytes as u64;
        }

        // Save remaining partial packet
        if complete_bytes < aligned.len() {
            self.partial_packet = aligned[complete_bytes..].to_vec();
        }

        // Collect extracted frames
        let frames: Vec<TsFrame> = self.output.borrow_mut().drain(..).collect();
        Ok(frames)
    }

    /// Flush any remaining partial data.
    ///
    /// Call this at end of stream to ensure all frames are extracted.
    pub fn flush(&mut self) -> Vec<TsFrame> {
        self.partial_packet.clear();
        self.output.borrow_mut().drain(..).collect()
    }

    /// Reset the demuxer state.
    pub fn reset(&mut self) {
        self.partial_packet.clear();
        self.output.borrow_mut().clear();
        *self.stats.borrow_mut() = TsDemuxStats::default();

        // Recreate the demuxer
        self.ctx = TsDemuxContext::new(self.output.clone(), self.stats.clone());
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
