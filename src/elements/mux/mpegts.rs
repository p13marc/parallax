//! MPEG Transport Stream multiplexer.
//!
//! This module provides an MPEG-TS muxer that combines elementary streams
//! (video, audio, metadata/KLV) into transport stream data.
//!
//! # Features
//!
//! - Video streams: H.264, H.265, MPEG-2
//! - Audio streams: AAC, MPEG audio
//! - Private data streams: KLV (SMPTE ST 0336), STANAG 4609
//! - Configurable PCR interval and bitrate
//! - PSI table generation (PAT, PMT)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::mux::{TsMux, TsMuxConfig, TsMuxTrack, TsMuxStreamType};
//!
//! // Create muxer with video and metadata tracks
//! let config = TsMuxConfig::new()
//!     .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
//!     .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());
//!
//! let mut mux = TsMux::new(config);
//!
//! // Write video frame
//! let ts_packets = mux.write_pes(256, &video_data, Some(pts), None)?;
//!
//! // Write KLV metadata
//! let ts_packets = mux.write_pes(257, &klv_data, Some(pts), None)?;
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use crate::metadata::Metadata;

use std::collections::HashMap;
use std::sync::OnceLock;

/// Shared arena for MPEG-TS muxer buffers.
fn ts_mux_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    // TS packets are 188 bytes, but we batch them; use moderate slot size
    ARENA.get_or_init(|| SharedArena::new(1024 * 1024, 32).unwrap())
}

// ============================================================================
// Constants
// ============================================================================

/// Size of a single MPEG-TS packet.
pub const TS_PACKET_SIZE: usize = 188;

/// Sync byte for TS packets.
const SYNC_BYTE: u8 = 0x47;

/// Maximum payload size in a TS packet (no adaptation field).
const MAX_PAYLOAD_SIZE: usize = 184;

/// PMT default PID.
const PMT_PID_DEFAULT: u16 = 0x1000;

/// Null packet PID.
#[allow(dead_code)]
const NULL_PID: u16 = 0x1FFF;

/// PCR PID (usually same as video PID).
const PCR_INTERVAL_MS: u64 = 40; // 40ms = 25 PCR per second

/// 90kHz clock for PTS/DTS.
const CLOCK_90KHZ: u64 = 90_000;

/// 27MHz clock for PCR.
const CLOCK_27MHZ: u64 = 27_000_000;

// ============================================================================
// Stream Types (ISO/IEC 13818-1)
// ============================================================================

/// MPEG-TS stream type for muxing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TsMuxStreamType {
    /// H.264/AVC video (0x1B).
    H264,
    /// H.265/HEVC video (0x24).
    H265,
    /// AV1 video (0x06 private + descriptor, or custom).
    /// Note: AV1 in MPEG-TS uses private data with AV1 descriptor per AOM spec.
    Av1,
    /// MPEG-2 video (0x02).
    Mpeg2Video,
    /// AAC audio with ADTS transport (0x0F).
    AacAdts,
    /// AAC audio with LATM transport (0x11).
    AacLatm,
    /// MPEG-1/2 audio (0x03/0x04).
    MpegAudio,
    /// AC-3 audio (0x81).
    Ac3,
    /// Private PES data (0x06) - for KLV/STANAG.
    PrivateData,
    /// SMPTE ST 0336 KLV (0x15).
    Klv,
    /// Custom stream type.
    Custom(u8),
}

impl TsMuxStreamType {
    /// Get the ISO/IEC 13818-1 stream type code.
    pub fn stream_type_code(&self) -> u8 {
        match self {
            TsMuxStreamType::H264 => 0x1B,
            TsMuxStreamType::H265 => 0x24,
            TsMuxStreamType::Av1 => 0x06, // Private data, requires AV1 descriptor
            TsMuxStreamType::Mpeg2Video => 0x02,
            TsMuxStreamType::AacAdts => 0x0F,
            TsMuxStreamType::AacLatm => 0x11,
            TsMuxStreamType::MpegAudio => 0x03,
            TsMuxStreamType::Ac3 => 0x81,
            TsMuxStreamType::PrivateData => 0x06,
            TsMuxStreamType::Klv => 0x15, // SMPTE metadata
            TsMuxStreamType::Custom(code) => *code,
        }
    }

    /// Returns true if this is a video stream type.
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            TsMuxStreamType::H264
                | TsMuxStreamType::H265
                | TsMuxStreamType::Av1
                | TsMuxStreamType::Mpeg2Video
        )
    }

    /// Returns true if this is an audio stream type.
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            TsMuxStreamType::AacAdts
                | TsMuxStreamType::AacLatm
                | TsMuxStreamType::MpegAudio
                | TsMuxStreamType::Ac3
        )
    }

    /// Returns true if this is a data/metadata stream type.
    pub fn is_data(&self) -> bool {
        matches!(self, TsMuxStreamType::PrivateData | TsMuxStreamType::Klv)
    }
}

// ============================================================================
// Track Configuration
// ============================================================================

/// Configuration for a track in the mux.
#[derive(Debug, Clone)]
pub struct TsMuxTrack {
    /// Elementary stream PID (13-bit, 0x0010-0x1FFE).
    pub pid: u16,
    /// Stream type.
    pub stream_type: TsMuxStreamType,
    /// Stream ID for PES header (0xE0-0xEF for video, 0xC0-0xDF for audio).
    pub stream_id: u8,
    /// Optional descriptor data for PMT.
    pub descriptors: Vec<u8>,
    /// Whether this track carries PCR.
    pub is_pcr_pid: bool,
}

impl TsMuxTrack {
    /// Create a new track with default settings.
    pub fn new(pid: u16, stream_type: TsMuxStreamType) -> Self {
        let stream_id = match stream_type {
            TsMuxStreamType::H264
            | TsMuxStreamType::H265
            | TsMuxStreamType::Av1
            | TsMuxStreamType::Mpeg2Video => 0xE0,
            TsMuxStreamType::AacAdts
            | TsMuxStreamType::AacLatm
            | TsMuxStreamType::MpegAudio
            | TsMuxStreamType::Ac3 => 0xC0,
            TsMuxStreamType::PrivateData | TsMuxStreamType::Klv => 0xBD,
            TsMuxStreamType::Custom(_) => 0xBD,
        };

        Self {
            pid,
            stream_type,
            stream_id,
            descriptors: Vec::new(),
            is_pcr_pid: false,
        }
    }

    /// Set this track as the PCR PID.
    pub fn with_pcr(mut self) -> Self {
        self.is_pcr_pid = true;
        self
    }

    /// Set custom stream ID.
    pub fn with_stream_id(mut self, stream_id: u8) -> Self {
        self.stream_id = stream_id;
        self
    }

    /// Add descriptor data for PMT.
    pub fn with_descriptor(mut self, descriptor: Vec<u8>) -> Self {
        self.descriptors = descriptor;
        self
    }

    /// Mark as video track (sets appropriate stream ID).
    pub fn video(mut self) -> Self {
        self.stream_id = 0xE0;
        self.is_pcr_pid = true; // Video typically carries PCR
        self
    }

    /// Mark as audio track (sets appropriate stream ID).
    pub fn audio(mut self) -> Self {
        self.stream_id = 0xC0;
        self
    }

    /// Mark as private data track (for KLV/metadata).
    pub fn private_data(mut self) -> Self {
        self.stream_id = 0xBD;
        self
    }
}

// ============================================================================
// Mux Configuration
// ============================================================================

/// Configuration for the TS muxer.
#[derive(Debug, Clone)]
pub struct TsMuxConfig {
    /// Program number (default: 1).
    pub program_number: u16,
    /// PMT PID (default: 0x1000).
    pub pmt_pid: u16,
    /// Tracks in this program.
    pub tracks: Vec<TsMuxTrack>,
    /// PCR interval in milliseconds (default: 40ms).
    pub pcr_interval_ms: u64,
    /// Transport stream ID (default: 1).
    pub ts_id: u16,
    /// Include PSI at start (PAT/PMT).
    pub include_psi: bool,
    /// PSI repeat interval in packets (0 = only at start).
    pub psi_interval: u32,
}

impl Default for TsMuxConfig {
    fn default() -> Self {
        Self {
            program_number: 1,
            pmt_pid: PMT_PID_DEFAULT,
            tracks: Vec::new(),
            pcr_interval_ms: PCR_INTERVAL_MS,
            ts_id: 1,
            include_psi: true,
            psi_interval: 0,
        }
    }
}

impl TsMuxConfig {
    /// Create a new configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a track to the configuration.
    pub fn add_track(mut self, track: TsMuxTrack) -> Self {
        self.tracks.push(track);
        self
    }

    /// Set the program number.
    pub fn program_number(mut self, number: u16) -> Self {
        self.program_number = number;
        self
    }

    /// Set the PMT PID.
    pub fn pmt_pid(mut self, pid: u16) -> Self {
        self.pmt_pid = pid;
        self
    }

    /// Set the PCR interval in milliseconds.
    pub fn pcr_interval_ms(mut self, interval: u64) -> Self {
        self.pcr_interval_ms = interval;
        self
    }

    /// Set PSI repeat interval (in packets, 0 = only at start).
    pub fn psi_interval(mut self, interval: u32) -> Self {
        self.psi_interval = interval;
        self
    }

    /// Get the PCR PID (first track with is_pcr_pid set, or first video track).
    pub fn pcr_pid(&self) -> Option<u16> {
        self.tracks
            .iter()
            .find(|t| t.is_pcr_pid)
            .or_else(|| self.tracks.iter().find(|t| t.stream_type.is_video()))
            .map(|t| t.pid)
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for the TS muxer.
#[derive(Debug, Clone, Default)]
pub struct TsMuxStats {
    /// Total TS packets written.
    pub packets_written: u64,
    /// Total bytes written.
    pub bytes_written: u64,
    /// PES packets written.
    pub pes_packets: u64,
    /// PAT packets written.
    pub pat_packets: u64,
    /// PMT packets written.
    pub pmt_packets: u64,
    /// PCR count.
    pub pcr_count: u64,
}

// ============================================================================
// Track State
// ============================================================================

/// Runtime state for a track.
#[derive(Debug)]
struct TrackState {
    /// Continuity counter (4-bit, 0-15).
    continuity_counter: u8,
}

impl Default for TrackState {
    fn default() -> Self {
        Self {
            continuity_counter: 0,
        }
    }
}

// ============================================================================
// TS Muxer
// ============================================================================

/// MPEG Transport Stream multiplexer.
///
/// Combines elementary streams into MPEG-TS format.
pub struct TsMux {
    /// Configuration.
    config: TsMuxConfig,
    /// Track states (keyed by PID).
    track_states: HashMap<u16, TrackState>,
    /// PSI continuity counters.
    pat_cc: u8,
    pmt_cc: u8,
    /// Statistics.
    stats: TsMuxStats,
    /// Last PCR timestamp.
    last_pcr: Option<u64>,
    /// Packet counter for PSI interval.
    packet_counter: u32,
    /// Whether PSI has been written.
    psi_written: bool,
}

impl TsMux {
    /// Create a new TS muxer with the given configuration.
    pub fn new(config: TsMuxConfig) -> Self {
        let mut track_states = HashMap::new();
        for track in &config.tracks {
            track_states.insert(track.pid, TrackState::default());
        }

        Self {
            config,
            track_states,
            pat_cc: 0,
            pmt_cc: 0,
            stats: TsMuxStats::default(),
            last_pcr: None,
            packet_counter: 0,
            psi_written: false,
        }
    }

    /// Get current statistics.
    pub fn stats(&self) -> &TsMuxStats {
        &self.stats
    }

    /// Get the configuration.
    pub fn config(&self) -> &TsMuxConfig {
        &self.config
    }

    /// Generate PSI tables (PAT + PMT).
    pub fn write_psi(&mut self) -> Vec<u8> {
        let mut output = Vec::new();

        // Write PAT
        output.extend(self.write_pat());

        // Write PMT
        output.extend(self.write_pmt());

        self.psi_written = true;
        output
    }

    /// Generate PAT (Program Association Table).
    fn write_pat(&mut self) -> Vec<u8> {
        let mut packet = [0u8; TS_PACKET_SIZE];

        // TS header
        packet[0] = SYNC_BYTE;
        packet[1] = 0x40; // PUSI=1, PID high bits = 0
        packet[2] = 0x00; // PID low bits = 0 (PAT)
        packet[3] = 0x10 | (self.pat_cc & 0x0F); // Adaptation=01 (payload only), CC
        self.pat_cc = (self.pat_cc + 1) & 0x0F;

        // Pointer field (required for PSI)
        packet[4] = 0x00;

        // PAT section
        let mut section = Vec::new();
        section.push(0x00); // table_id = PAT

        // Section length placeholder (filled later)
        let section_length_pos = section.len();
        section.push(0x00);
        section.push(0x00);

        // Transport stream ID
        section.push((self.config.ts_id >> 8) as u8);
        section.push((self.config.ts_id & 0xFF) as u8);

        // Version (5 bits), current_next (1)
        section.push(0xC1); // version=0, current_next=1, reserved=11

        // Section number
        section.push(0x00);

        // Last section number
        section.push(0x00);

        // Program entries
        // Program number
        section.push((self.config.program_number >> 8) as u8);
        section.push((self.config.program_number & 0xFF) as u8);
        // PMT PID
        section.push(0xE0 | ((self.config.pmt_pid >> 8) as u8 & 0x1F));
        section.push((self.config.pmt_pid & 0xFF) as u8);

        // Fix section length (includes CRC)
        let section_length = section.len() - 3 + 4; // -3 for header before length, +4 for CRC
        section[section_length_pos] = 0xB0 | ((section_length >> 8) as u8 & 0x0F);
        section[section_length_pos + 1] = (section_length & 0xFF) as u8;

        // Calculate CRC32
        let crc = crc32_mpeg(&section);
        section.push((crc >> 24) as u8);
        section.push((crc >> 16) as u8);
        section.push((crc >> 8) as u8);
        section.push((crc & 0xFF) as u8);

        // Copy section to packet
        let payload_start = 5;
        let section_len = section.len().min(TS_PACKET_SIZE - payload_start);
        packet[payload_start..payload_start + section_len].copy_from_slice(&section[..section_len]);

        // Fill rest with stuffing
        for i in payload_start + section_len..TS_PACKET_SIZE {
            packet[i] = 0xFF;
        }

        self.stats.pat_packets += 1;
        self.stats.packets_written += 1;
        self.stats.bytes_written += TS_PACKET_SIZE as u64;

        packet.to_vec()
    }

    /// Generate PMT (Program Map Table).
    fn write_pmt(&mut self) -> Vec<u8> {
        let mut packet = [0u8; TS_PACKET_SIZE];

        // TS header
        packet[0] = SYNC_BYTE;
        packet[1] = 0x40 | ((self.config.pmt_pid >> 8) as u8 & 0x1F);
        packet[2] = (self.config.pmt_pid & 0xFF) as u8;
        packet[3] = 0x10 | (self.pmt_cc & 0x0F);
        self.pmt_cc = (self.pmt_cc + 1) & 0x0F;

        // Pointer field
        packet[4] = 0x00;

        // PMT section
        let mut section = Vec::new();
        section.push(0x02); // table_id = PMT

        // Section length placeholder
        let section_length_pos = section.len();
        section.push(0x00);
        section.push(0x00);

        // Program number
        section.push((self.config.program_number >> 8) as u8);
        section.push((self.config.program_number & 0xFF) as u8);

        // Version, current_next
        section.push(0xC1);

        // Section number
        section.push(0x00);

        // Last section number
        section.push(0x00);

        // PCR PID
        let pcr_pid = self.config.pcr_pid().unwrap_or(0x1FFF);
        section.push(0xE0 | ((pcr_pid >> 8) as u8 & 0x1F));
        section.push((pcr_pid & 0xFF) as u8);

        // Program info length (no program descriptors)
        section.push(0xF0);
        section.push(0x00);

        // Elementary stream info
        for track in &self.config.tracks {
            // Stream type
            section.push(track.stream_type.stream_type_code());

            // Elementary PID
            section.push(0xE0 | ((track.pid >> 8) as u8 & 0x1F));
            section.push((track.pid & 0xFF) as u8);

            // ES info length
            let desc_len = track.descriptors.len();
            section.push(0xF0 | ((desc_len >> 8) as u8 & 0x0F));
            section.push((desc_len & 0xFF) as u8);

            // Descriptors
            section.extend_from_slice(&track.descriptors);
        }

        // Fix section length
        let section_length = section.len() - 3 + 4;
        section[section_length_pos] = 0xB0 | ((section_length >> 8) as u8 & 0x0F);
        section[section_length_pos + 1] = (section_length & 0xFF) as u8;

        // CRC32
        let crc = crc32_mpeg(&section);
        section.push((crc >> 24) as u8);
        section.push((crc >> 16) as u8);
        section.push((crc >> 8) as u8);
        section.push((crc & 0xFF) as u8);

        // Copy section to packet
        let payload_start = 5;
        let section_len = section.len().min(TS_PACKET_SIZE - payload_start);
        packet[payload_start..payload_start + section_len].copy_from_slice(&section[..section_len]);

        // Fill rest with stuffing
        for i in payload_start + section_len..TS_PACKET_SIZE {
            packet[i] = 0xFF;
        }

        self.stats.pmt_packets += 1;
        self.stats.packets_written += 1;
        self.stats.bytes_written += TS_PACKET_SIZE as u64;

        packet.to_vec()
    }

    /// Write a PES packet for a given PID.
    ///
    /// Returns the TS packets containing the PES data.
    pub fn write_pes(
        &mut self,
        pid: u16,
        data: &[u8],
        pts: Option<ClockTime>,
        dts: Option<ClockTime>,
    ) -> Result<Vec<u8>> {
        // Find track
        let track = self
            .config
            .tracks
            .iter()
            .find(|t| t.pid == pid)
            .ok_or_else(|| Error::Element(format!("Unknown PID: {}", pid)))?
            .clone();

        // Ensure track state exists
        if !self.track_states.contains_key(&pid) {
            self.track_states.insert(pid, TrackState::default());
        }

        let mut output = Vec::new();

        // Check if we should write PSI
        if self.config.include_psi && (!self.psi_written || self.should_write_psi()) {
            output.extend(self.write_psi());
        }

        // Build PES packet
        let pes_packet = self.build_pes_packet(&track, data, pts, dts);

        // Check if we need PCR (only for PCR PID)
        let need_pcr = track.is_pcr_pid && self.should_write_pcr(pts);

        // Split PES packet into TS packets
        let ts_packets = self.packetize_pes(pid, &pes_packet, need_pcr, pts);
        output.extend(ts_packets);

        self.stats.pes_packets += 1;
        Ok(output)
    }

    /// Build a PES packet header + payload.
    fn build_pes_packet(
        &self,
        track: &TsMuxTrack,
        data: &[u8],
        pts: Option<ClockTime>,
        dts: Option<ClockTime>,
    ) -> Vec<u8> {
        let mut pes = Vec::new();

        // PES start code (00 00 01)
        pes.push(0x00);
        pes.push(0x00);
        pes.push(0x01);

        // Stream ID
        pes.push(track.stream_id);

        // Calculate header extension length
        let has_pts = pts.is_some();
        let has_dts = dts.is_some() && pts.is_some();
        let header_data_length = if has_pts && has_dts {
            10 // PTS + DTS
        } else if has_pts {
            5 // PTS only
        } else {
            0
        };

        // PES packet length (0 = unbounded for video)
        let pes_packet_length = if track.stream_type.is_video() {
            0 // Unbounded for video
        } else {
            // 3 (header) + header_data_length + data length
            let len = 3 + header_data_length + data.len();
            if len > 65535 {
                0 // Too large, use unbounded
            } else {
                len as u16
            }
        };
        pes.push((pes_packet_length >> 8) as u8);
        pes.push((pes_packet_length & 0xFF) as u8);

        // PES header flags
        // Byte 1: '10' + scrambling(2) + priority(1) + alignment(1) + copyright(1) + original(1)
        pes.push(0x80); // '10' + no scrambling, etc.

        // Byte 2: PTS_DTS_flags(2) + ESCR(1) + ES_rate(1) + DSM_trick(1) + additional_copy(1) + CRC(1) + extension(1)
        let pts_dts_flags = if has_pts && has_dts {
            0xC0 // Both PTS and DTS
        } else if has_pts {
            0x80 // PTS only
        } else {
            0x00 // Neither
        };
        pes.push(pts_dts_flags);

        // PES header data length
        pes.push(header_data_length as u8);

        // Write PTS
        if let Some(pts_time) = pts {
            let pts_90khz = pts_time.nanos() as u64 * CLOCK_90KHZ / 1_000_000_000;
            if has_dts {
                // PTS with DTS flag (0011)
                pes.extend(encode_timestamp(pts_90khz, 0x03));
            } else {
                // PTS only flag (0010)
                pes.extend(encode_timestamp(pts_90khz, 0x02));
            }
        }

        // Write DTS
        if let Some(dts_time) = dts {
            if has_pts {
                let dts_90khz = dts_time.nanos() as u64 * CLOCK_90KHZ / 1_000_000_000;
                pes.extend(encode_timestamp(dts_90khz, 0x01));
            }
        }

        // Payload
        pes.extend_from_slice(data);

        pes
    }

    /// Split a PES packet into TS packets.
    fn packetize_pes(
        &mut self,
        pid: u16,
        pes_data: &[u8],
        include_pcr: bool,
        pts: Option<ClockTime>,
    ) -> Vec<u8> {
        let mut output = Vec::new();
        let mut offset = 0;
        let mut first_packet = true;

        while offset < pes_data.len() {
            let mut packet = [0u8; TS_PACKET_SIZE];

            // Get continuity counter
            let state = self.track_states.get_mut(&pid).unwrap();
            let cc = state.continuity_counter;
            state.continuity_counter = (state.continuity_counter + 1) & 0x0F;

            // TS header
            packet[0] = SYNC_BYTE;

            // Flags and PID
            let pusi = if first_packet { 0x40 } else { 0x00 };
            packet[1] = pusi | ((pid >> 8) as u8 & 0x1F);
            packet[2] = (pid & 0xFF) as u8;

            // Determine adaptation field needs
            let remaining = pes_data.len() - offset;
            let need_adaptation = include_pcr && first_packet;

            let (adaptation_length, payload_start) = if need_adaptation {
                // Adaptation field with PCR (7 bytes: 1 length + 1 flags + 6 PCR)
                let pcr_time = pts
                    .map(|t| t.nanos() as u64 * CLOCK_27MHZ / 1_000_000_000)
                    .unwrap_or(0);

                packet[3] = 0x30 | (cc & 0x0F); // Adaptation + payload
                packet[4] = 7; // Adaptation field length (excluding this byte)
                packet[5] = 0x10; // PCR flag set

                // PCR (33 bits base + 6 reserved + 9 bits extension)
                let pcr_base = pcr_time / 300;
                let pcr_ext = (pcr_time % 300) as u16;
                packet[6] = (pcr_base >> 25) as u8;
                packet[7] = (pcr_base >> 17) as u8;
                packet[8] = (pcr_base >> 9) as u8;
                packet[9] = (pcr_base >> 1) as u8;
                packet[10] = ((pcr_base & 0x01) << 7) as u8 | 0x7E | ((pcr_ext >> 8) as u8 & 0x01);
                packet[11] = (pcr_ext & 0xFF) as u8;

                self.stats.pcr_count += 1;
                self.last_pcr = Some(pts.map(|t| t.nanos()).unwrap_or(0));

                (8, 12) // 1 (length byte) + 7 (PCR data), payload starts at 12
            } else if remaining < MAX_PAYLOAD_SIZE {
                // Need stuffing - adaptation field for padding
                let stuffing_needed = MAX_PAYLOAD_SIZE - remaining;
                if stuffing_needed == 1 {
                    // Special case: 1 byte adaptation field (just length=0)
                    packet[3] = 0x30 | (cc & 0x0F);
                    packet[4] = 0; // Adaptation field length = 0
                    (1, 5)
                } else {
                    // Adaptation field with stuffing
                    packet[3] = 0x30 | (cc & 0x0F);
                    packet[4] = (stuffing_needed - 1) as u8; // Adaptation field length
                    packet[5] = 0x00; // No flags
                    // Fill with stuffing bytes
                    for i in 6..4 + stuffing_needed {
                        packet[i] = 0xFF;
                    }
                    (stuffing_needed, 4 + stuffing_needed)
                }
            } else {
                // Payload only
                packet[3] = 0x10 | (cc & 0x0F);
                (0, 4)
            };

            // Copy payload
            let payload_space = TS_PACKET_SIZE - payload_start;
            let copy_len = remaining.min(payload_space);
            packet[payload_start..payload_start + copy_len]
                .copy_from_slice(&pes_data[offset..offset + copy_len]);

            // If there's remaining space (shouldn't happen with proper stuffing), fill with 0xFF
            for i in payload_start + copy_len..TS_PACKET_SIZE {
                packet[i] = 0xFF;
            }

            output.extend_from_slice(&packet);
            offset += copy_len;
            first_packet = false;

            self.stats.packets_written += 1;
            self.stats.bytes_written += TS_PACKET_SIZE as u64;
            self.packet_counter += 1;

            // Avoid unused variable warning
            let _ = adaptation_length;
        }

        output
    }

    /// Check if we should write PCR based on interval.
    fn should_write_pcr(&self, pts: Option<ClockTime>) -> bool {
        match (self.last_pcr, pts) {
            (Some(last), Some(current)) => {
                let elapsed_ns = current.nanos().saturating_sub(last);
                let interval_ns = self.config.pcr_interval_ms * 1_000_000;
                elapsed_ns >= interval_ns
            }
            (None, Some(_)) => true, // First PCR
            _ => false,
        }
    }

    /// Check if we should write PSI based on interval.
    fn should_write_psi(&self) -> bool {
        if self.config.psi_interval == 0 {
            return false;
        }
        self.packet_counter >= self.config.psi_interval
    }

    /// Reset the muxer state.
    pub fn reset(&mut self) {
        for state in self.track_states.values_mut() {
            state.continuity_counter = 0;
        }
        self.pat_cc = 0;
        self.pmt_cc = 0;
        self.stats = TsMuxStats::default();
        self.last_pcr = None;
        self.packet_counter = 0;
        self.psi_written = false;
    }

    /// Write a Buffer as PES data.
    ///
    /// Extracts PTS/DTS from buffer metadata.
    pub fn write_buffer(&mut self, pid: u16, buffer: &Buffer) -> Result<Vec<u8>> {
        let data = buffer.as_bytes();
        let pts = if buffer.metadata().pts.nanos() > 0 {
            Some(buffer.metadata().pts)
        } else {
            None
        };
        let dts = if buffer.metadata().dts.nanos() > 0 {
            Some(buffer.metadata().dts)
        } else {
            None
        };
        self.write_pes(pid, data, pts, dts)
    }

    /// Create a Buffer containing TS packets.
    pub fn create_ts_buffer(&self, ts_data: Vec<u8>) -> Result<Buffer> {
        if ts_data.is_empty() {
            return Err(Error::Element("Empty TS data".into()));
        }

        let arena = ts_mux_arena();
        arena.reclaim();

        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;

        slot.data_mut()[..ts_data.len()].copy_from_slice(&ts_data);

        let handle = MemoryHandle::with_len(slot, ts_data.len());
        let metadata = Metadata::new();

        Ok(Buffer::new(handle, metadata))
    }
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Encode a 33-bit timestamp for PES header.
fn encode_timestamp(ts: u64, marker: u8) -> [u8; 5] {
    let mut bytes = [0u8; 5];

    // Format: marker(4) + ts[32:30](3) + 1 + ts[29:15](15) + 1 + ts[14:0](15) + 1
    bytes[0] = (marker << 4) | (((ts >> 30) & 0x07) as u8) << 1 | 0x01;
    bytes[1] = ((ts >> 22) & 0xFF) as u8;
    bytes[2] = (((ts >> 15) & 0x7F) << 1) as u8 | 0x01;
    bytes[3] = ((ts >> 7) & 0xFF) as u8;
    bytes[4] = (((ts & 0x7F) << 1) | 0x01) as u8;

    bytes
}

/// CRC32 calculation for MPEG-TS PSI tables.
fn crc32_mpeg(data: &[u8]) -> u32 {
    const CRC_TABLE: [u32; 256] = [
        0x00000000, 0x04c11db7, 0x09823b6e, 0x0d4326d9, 0x130476dc, 0x17c56b6b, 0x1a864db2,
        0x1e475005, 0x2608edb8, 0x22c9f00f, 0x2f8ad6d6, 0x2b4bcb61, 0x350c9b64, 0x31cd86d3,
        0x3c8ea00a, 0x384fbdbd, 0x4c11db70, 0x48d0c6c7, 0x4593e01e, 0x4152fda9, 0x5f15adac,
        0x5bd4b01b, 0x569796c2, 0x52568b75, 0x6a1936c8, 0x6ed82b7f, 0x639b0da6, 0x675a1011,
        0x791d4014, 0x7ddc5da3, 0x709f7b7a, 0x745e66cd, 0x9823b6e0, 0x9ce2ab57, 0x91a18d8e,
        0x95609039, 0x8b27c03c, 0x8fe6dd8b, 0x82a5fb52, 0x8664e6e5, 0xbe2b5b58, 0xbaea46ef,
        0xb7a96036, 0xb3687d81, 0xad2f2d84, 0xa9ee3033, 0xa4ad16ea, 0xa06c0b5d, 0xd4326d90,
        0xd0f37027, 0xddb056fe, 0xd9714b49, 0xc7361b4c, 0xc3f706fb, 0xceb42022, 0xca753d95,
        0xf23a8028, 0xf6fb9d9f, 0xfbb8bb46, 0xff79a6f1, 0xe13ef6f4, 0xe5ffeb43, 0xe8bccd9a,
        0xec7dd02d, 0x34867077, 0x30476dc0, 0x3d044b19, 0x39c556ae, 0x278206ab, 0x23431b1c,
        0x2e003dc5, 0x2ac12072, 0x128e9dcf, 0x164f8078, 0x1b0ca6a1, 0x1fcdbb16, 0x018aeb13,
        0x054bf6a4, 0x0808d07d, 0x0cc9cdca, 0x7897ab07, 0x7c56b6b0, 0x71159069, 0x75d48dde,
        0x6b93dddb, 0x6f52c06c, 0x6211e6b5, 0x66d0fb02, 0x5e9f46bf, 0x5a5e5b08, 0x571d7dd1,
        0x53dc6066, 0x4d9b3063, 0x495a2dd4, 0x44190b0d, 0x40d816ba, 0xaca5c697, 0xa864db20,
        0xa527fdf9, 0xa1e6e04e, 0xbfa1b04b, 0xbb60adfc, 0xb6238b25, 0xb2e29692, 0x8aad2b2f,
        0x8e6c3698, 0x832f1041, 0x87ee0df6, 0x99a95df3, 0x9d684044, 0x902b669d, 0x94ea7b2a,
        0xe0b41de7, 0xe4750050, 0xe9362689, 0xedf73b3e, 0xf3b06b3b, 0xf771768c, 0xfa325055,
        0xfef34de2, 0xc6bcf05f, 0xc27dede8, 0xcf3ecb31, 0xcbffd686, 0xd5b88683, 0xd1799b34,
        0xdc3abded, 0xd8fba05a, 0x690ce0ee, 0x6dcdfd59, 0x608edb80, 0x644fc637, 0x7a089632,
        0x7ec98b85, 0x738aad5c, 0x774bb0eb, 0x4f040d56, 0x4bc510e1, 0x46863638, 0x42472b8f,
        0x5c007b8a, 0x58c1663d, 0x558240e4, 0x51435d53, 0x251d3b9e, 0x21dc2629, 0x2c9f00f0,
        0x285e1d47, 0x36194d42, 0x32d850f5, 0x3f9b762c, 0x3b5a6b9b, 0x0315d626, 0x07d4cb91,
        0x0a97ed48, 0x0e56f0ff, 0x1011a0fa, 0x14d0bd4d, 0x19939b94, 0x1d528623, 0xf12f560e,
        0xf5ee4bb9, 0xf8ad6d60, 0xfc6c70d7, 0xe22b20d2, 0xe6ea3d65, 0xeba91bbc, 0xef68060b,
        0xd727bbb6, 0xd3e6a601, 0xdea580d8, 0xda649d6f, 0xc423cd6a, 0xc0e2d0dd, 0xcda1f604,
        0xc960ebb3, 0xbd3e8d7e, 0xb9ff90c9, 0xb4bcb610, 0xb07daba7, 0xae3afba2, 0xaafbe615,
        0xa7b8c0cc, 0xa379dd7b, 0x9b3660c6, 0x9ff77d71, 0x92b45ba8, 0x9675461f, 0x8832161a,
        0x8cf30bad, 0x81b02d74, 0x857130c3, 0x5d8a9099, 0x594b8d2e, 0x5408abf7, 0x50c9b640,
        0x4e8ee645, 0x4a4ffbf2, 0x470cdd2b, 0x43cdc09c, 0x7b827d21, 0x7f436096, 0x7200464f,
        0x76c15bf8, 0x68860bfd, 0x6c47164a, 0x61043093, 0x65c52d24, 0x119b4be9, 0x155a565e,
        0x18197087, 0x1cd86d30, 0x029f3d35, 0x065e2082, 0x0b1d065b, 0x0fdc1bec, 0x3793a651,
        0x3352bbe6, 0x3e119d3f, 0x3ad08088, 0x2497d08d, 0x2056cd3a, 0x2d15ebe3, 0x29d4f654,
        0xc5a92679, 0xc1683bce, 0xcc2b1d17, 0xc8ea00a0, 0xd6ad50a5, 0xd26c4d12, 0xdf2f6bcb,
        0xdbee767c, 0xe3a1cbc1, 0xe760d676, 0xea23f0af, 0xeee2ed18, 0xf0a5bd1d, 0xf464a0aa,
        0xf9278673, 0xfde69bc4, 0x89b8fd09, 0x8d79e0be, 0x803ac667, 0x84fbdbd0, 0x9abc8bd5,
        0x9e7d9662, 0x933eb0bb, 0x97ffad0c, 0xafb010b1, 0xab710d06, 0xa6322bdf, 0xa2f33668,
        0xbcb4666d, 0xb8757bda, 0xb5365d03, 0xb1f740b4,
    ];

    let mut crc = 0xFFFFFFFFu32;
    for &byte in data {
        crc = CRC_TABLE[((crc >> 24) as u8 ^ byte) as usize] ^ (crc << 8);
    }
    crc
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ts_mux_creation() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mux = TsMux::new(config);
        assert_eq!(mux.stats().packets_written, 0);
    }

    #[test]
    fn test_ts_stream_type_codes() {
        assert_eq!(TsMuxStreamType::H264.stream_type_code(), 0x1B);
        assert_eq!(TsMuxStreamType::H265.stream_type_code(), 0x24);
        assert_eq!(TsMuxStreamType::AacAdts.stream_type_code(), 0x0F);
        assert_eq!(TsMuxStreamType::Klv.stream_type_code(), 0x15);
        assert_eq!(TsMuxStreamType::PrivateData.stream_type_code(), 0x06);
    }

    #[test]
    fn test_ts_stream_type_classification() {
        assert!(TsMuxStreamType::H264.is_video());
        assert!(TsMuxStreamType::H265.is_video());
        assert!(!TsMuxStreamType::H264.is_audio());
        assert!(!TsMuxStreamType::H264.is_data());

        assert!(TsMuxStreamType::AacAdts.is_audio());
        assert!(!TsMuxStreamType::AacAdts.is_video());

        assert!(TsMuxStreamType::Klv.is_data());
        assert!(TsMuxStreamType::PrivateData.is_data());
    }

    #[test]
    fn test_ts_track_builder() {
        let track = TsMuxTrack::new(256, TsMuxStreamType::H264)
            .video()
            .with_pcr();

        assert_eq!(track.pid, 256);
        assert_eq!(track.stream_id, 0xE0);
        assert!(track.is_pcr_pid);
    }

    #[test]
    fn test_ts_mux_write_psi() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mut mux = TsMux::new(config);

        let psi = mux.write_psi();

        // Should have PAT + PMT (2 packets × 188 bytes)
        assert_eq!(psi.len(), TS_PACKET_SIZE * 2);

        // Check sync bytes
        assert_eq!(psi[0], SYNC_BYTE);
        assert_eq!(psi[188], SYNC_BYTE);

        assert_eq!(mux.stats().pat_packets, 1);
        assert_eq!(mux.stats().pmt_packets, 1);
    }

    #[test]
    fn test_ts_mux_write_pes() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mut mux = TsMux::new(config);

        let data = vec![0x00, 0x00, 0x00, 0x01, 0x67]; // NAL start + SPS
        let pts = ClockTime::from_millis(1000);

        let ts_data = mux.write_pes(256, &data, Some(pts), None).unwrap();

        // Should have at least PSI + 1 PES packet
        assert!(ts_data.len() >= TS_PACKET_SIZE * 3);

        // All packets should have sync byte
        for i in (0..ts_data.len()).step_by(TS_PACKET_SIZE) {
            assert_eq!(ts_data[i], SYNC_BYTE);
        }
    }

    #[test]
    fn test_ts_mux_multiple_tracks() {
        let config = TsMuxConfig::new()
            .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
            .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());

        let mut mux = TsMux::new(config);

        // Write video
        let video_data = vec![0x00, 0x00, 0x00, 0x01, 0x67];
        let _ = mux.write_pes(256, &video_data, Some(ClockTime::from_millis(0)), None);

        // Write KLV
        let klv_data = vec![0x06, 0x0E, 0x2B]; // KLV key start
        let _ = mux.write_pes(257, &klv_data, Some(ClockTime::from_millis(0)), None);

        assert_eq!(mux.stats().pes_packets, 2);
    }

    #[test]
    fn test_ts_mux_reset() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mut mux = TsMux::new(config);

        // Write some data
        let _ = mux.write_pes(
            256,
            &[0x00, 0x00, 0x01],
            Some(ClockTime::from_millis(0)),
            None,
        );

        assert!(mux.stats().packets_written > 0);

        mux.reset();

        assert_eq!(mux.stats().packets_written, 0);
        assert_eq!(mux.stats().pes_packets, 0);
    }

    #[test]
    fn test_ts_mux_config_pcr_pid() {
        let config = TsMuxConfig::new()
            .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
            .add_track(TsMuxTrack::new(257, TsMuxStreamType::AacAdts).audio());

        // Video track should be PCR PID by default
        assert_eq!(config.pcr_pid(), Some(256));
    }

    #[test]
    fn test_timestamp_encoding() {
        // Test encoding 90kHz timestamp
        let ts: u64 = 90_000; // 1 second at 90kHz
        let encoded = encode_timestamp(ts, 0x02); // PTS only marker

        // Verify marker bits
        assert_eq!(encoded[0] & 0xF0, 0x20);
        // Verify marker bits (0x01) at specific positions
        assert_eq!(encoded[0] & 0x01, 0x01);
        assert_eq!(encoded[2] & 0x01, 0x01);
        assert_eq!(encoded[4] & 0x01, 0x01);
    }

    #[test]
    fn test_crc32_mpeg() {
        // Known CRC test vector for PAT-like data
        let data = [
            0x00, 0xB0, 0x0D, 0x00, 0x01, 0xC1, 0x00, 0x00, 0x00, 0x01, 0xE0, 0x10,
        ];
        let crc = crc32_mpeg(&data);
        // CRC should be non-zero for non-trivial data
        assert_ne!(crc, 0);
    }

    #[test]
    fn test_ts_mux_large_pes() {
        let config =
            TsMuxConfig::new().add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video());
        let mut mux = TsMux::new(config);

        // Large frame that requires multiple TS packets
        let large_data = vec![0xAB; 1000];
        let ts_data = mux
            .write_pes(256, &large_data, Some(ClockTime::from_millis(0)), None)
            .unwrap();

        // Should produce multiple packets
        let packet_count = ts_data.len() / TS_PACKET_SIZE;
        assert!(packet_count > 5); // At least PSI + multiple PES packets
    }
}
