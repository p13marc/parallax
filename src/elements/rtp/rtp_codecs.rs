//! RTP codec payloaders and depayloaders.
//!
//! This module provides elements for converting between raw media frames and
//! RTP packets for various codecs.
//!
//! ## Depayloaders (RTP → Raw)
//! - [`RtpH264Depay`]: H.264/AVC depacketizer
//! - [`RtpH265Depay`]: H.265/HEVC depacketizer
//! - [`RtpVp8Depay`]: VP8 depacketizer
//! - [`RtpVp9Depay`]: VP9 depacketizer
//! - [`RtpOpusDepay`]: Opus audio depacketizer
//!
//! ## Payloaders (Raw → RTP)
//! - [`RtpH264Pay`]: H.264/AVC packetizer
//! - [`RtpH265Pay`]: H.265/HEVC packetizer
//! - [`RtpVp8Pay`]: VP8 packetizer
//! - [`RtpVp9Pay`]: VP9 packetizer
//! - [`RtpOpusPay`]: Opus audio packetizer
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::{RtpSrc, RtpH264Depay, FileSink};
//!
//! // Pipeline: RtpSrc -> RtpH264Depay -> FileSink
//! let src = RtpSrc::bind("0.0.0.0:5004")?;
//! let depay = RtpH264Depay::new();
//!
//! // depay outputs complete H.264 NAL units in Annex B format
//! ```

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::format::{MediaFormat, RtpEncoding, RtpFormat, VideoCodec};
use crate::memory::SharedArena;
use crate::metadata::BufferFlags;

use bytes::Bytes;
use rtp::codecs::h264::{H264Packet, H264Payloader};
use rtp::codecs::h265::{H265Packet, HevcPayloader};
use rtp::codecs::opus::OpusPacket;
use rtp::codecs::vp8::{Vp8Packet, Vp8Payloader};
use rtp::codecs::vp9::{Vp9Packet, Vp9Payloader};
use rtp::packetizer::{Depacketizer, Payloader};

/// Default MTU for RTP payloaders.
const DEFAULT_MTU: usize = 1400;

// ============================================================================
// H.264 Depayloader
// ============================================================================

/// H.264/AVC RTP depacketizer.
///
/// Converts RTP packets containing H.264 payload into NAL units.
/// Handles FU-A fragmentation and STAP-A aggregation.
///
/// # Output Format
///
/// By default, outputs NAL units in Annex B format (with 0x00000001 start codes).
/// Set `is_avc = true` for AVC format (with 4-byte length prefix).
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RtpH264Depay;
///
/// let depay = RtpH264Depay::new();
/// // or for AVC format:
/// let depay = RtpH264Depay::new().avc_format();
/// ```
pub struct RtpH264Depay {
    name: String,
    depacketizer: H264Packet,
    stats: DepayStats,
    arena: Option<SharedArena>,
}

impl RtpH264Depay {
    /// Create a new H.264 depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-h264-depay".into(),
            depacketizer: H264Packet::default(),
            stats: DepayStats::default(),
            arena: None,
        }
    }

    /// Set AVC output format (4-byte length prefix instead of Annex B start codes).
    pub fn avc_format(mut self) -> Self {
        self.depacketizer.is_avc = true;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &DepayStats {
        &self.stats
    }
}

impl Default for RtpH264Depay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpH264Depay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.packets_in += 1;

        match self.depacketizer.depacketize(&payload) {
            Ok(output) => {
                if output.is_empty() {
                    // Fragment not complete yet
                    return Ok(None);
                }

                self.stats.frames_out += 1;
                self.stats.bytes_out += output.len() as u64;

                // Lazily initialize arena
                if self.arena.is_none() {
                    self.arena =
                        Some(SharedArena::new(256 * 1024, 32).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }
                let arena = self.arena.as_ref().unwrap();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
                slot.data_mut()[..output.len()].copy_from_slice(output.as_ref());

                let handle = crate::buffer::MemoryHandle::with_len(slot, output.len());

                // Preserve metadata, update format
                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Video(VideoCodec::H264));

                // Check if this is a keyframe (IDR NAL unit)
                if output.len() > 4 {
                    let nal_type = output[4] & 0x1F;
                    if nal_type == 5 {
                        // IDR
                        metadata.flags = metadata.flags.insert(BufferFlags::SYNC_POINT);
                    }
                }

                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("H.264 depacketize error: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// H.264 Payloader
// ============================================================================

/// H.264/AVC RTP packetizer.
///
/// Converts H.264 NAL units into RTP packets with proper fragmentation.
///
/// # Input Format
///
/// Accepts NAL units in Annex B format (with start codes) or raw NAL units.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RtpH264Pay;
///
/// let pay = RtpH264Pay::new()
///     .with_mtu(1400);
/// ```
pub struct RtpH264Pay {
    name: String,
    payloader: H264Payloader,
    mtu: usize,
    stats: PayStats,
    arena: Option<SharedArena>,
}

impl RtpH264Pay {
    /// Create a new H.264 packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-h264-pay".into(),
            payloader: H264Payloader::default(),
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            arena: None,
        }
    }

    /// Set the MTU (maximum transmission unit).
    pub fn with_mtu(mut self, mtu: usize) -> Self {
        self.mtu = mtu;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &PayStats {
        &self.stats
    }
}

impl Default for RtpH264Pay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpH264Pay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => {
                if packets.is_empty() {
                    return Ok(None);
                }

                // For now, concatenate all packets into one buffer
                // In a real implementation, we'd use Output::Multiple
                let total_len: usize = packets.iter().map(|p| p.len()).sum();
                self.stats.packets_out += packets.len() as u64;
                self.stats.bytes_out += total_len as u64;

                // Lazily initialize arena
                if self.arena.is_none() {
                    self.arena =
                        Some(SharedArena::new(256 * 1024, 32).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }
                let arena = self.arena.as_ref().unwrap();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;

                let mut offset = 0;
                for packet in &packets {
                    slot.data_mut()[offset..offset + packet.len()].copy_from_slice(packet.as_ref());
                    offset += packet.len();
                }

                let handle = crate::buffer::MemoryHandle::with_len(slot, total_len);

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Rtp(RtpFormat::H264));

                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("H.264 packetize error: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// H.265 Depayloader
// ============================================================================

/// H.265/HEVC RTP depacketizer.
///
/// Converts RTP packets containing H.265 payload into NAL units.
pub struct RtpH265Depay {
    name: String,
    depacketizer: H265Packet,
    stats: DepayStats,
    arena: Option<SharedArena>,
}

impl RtpH265Depay {
    /// Create a new H.265 depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-h265-depay".into(),
            depacketizer: H265Packet::default(),
            stats: DepayStats::default(),
            arena: None,
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &DepayStats {
        &self.stats
    }
}

impl Default for RtpH265Depay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpH265Depay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.packets_in += 1;

        match self.depacketizer.depacketize(&payload) {
            Ok(output) => {
                if output.is_empty() {
                    return Ok(None);
                }

                self.stats.frames_out += 1;
                self.stats.bytes_out += output.len() as u64;

                // Lazily initialize arena
                if self.arena.is_none() {
                    self.arena =
                        Some(SharedArena::new(256 * 1024, 32).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }
                let arena = self.arena.as_ref().unwrap();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
                slot.data_mut()[..output.len()].copy_from_slice(output.as_ref());

                let handle = crate::buffer::MemoryHandle::with_len(slot, output.len());

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Video(VideoCodec::H265));

                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("H.265 depacketize error: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// H.265 Payloader
// ============================================================================

/// H.265/HEVC RTP packetizer.
pub struct RtpH265Pay {
    name: String,
    payloader: HevcPayloader,
    mtu: usize,
    stats: PayStats,
    arena: Option<SharedArena>,
}

impl RtpH265Pay {
    /// Create a new H.265 packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-h265-pay".into(),
            payloader: HevcPayloader::default(),
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            arena: None,
        }
    }

    /// Set the MTU.
    pub fn with_mtu(mut self, mtu: usize) -> Self {
        self.mtu = mtu;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &PayStats {
        &self.stats
    }
}

impl Default for RtpH265Pay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpH265Pay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => {
                let packets: Vec<Bytes> = packets;
                if packets.is_empty() {
                    return Ok(None);
                }

                let total_len: usize = packets.iter().map(|p: &Bytes| p.len()).sum();
                self.stats.packets_out += packets.len() as u64;
                self.stats.bytes_out += total_len as u64;

                // Lazily initialize arena
                if self.arena.is_none() {
                    self.arena =
                        Some(SharedArena::new(256 * 1024, 32).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }
                let arena = self.arena.as_ref().unwrap();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;

                let mut offset = 0;
                for packet in &packets {
                    slot.data_mut()[offset..offset + packet.len()].copy_from_slice(packet.as_ref());
                    offset += packet.len();
                }

                let handle = crate::buffer::MemoryHandle::with_len(slot, total_len);

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Rtp(RtpFormat::H265));

                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("H.265 packetize error: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// VP8 Depayloader
// ============================================================================

/// VP8 RTP depacketizer.
pub struct RtpVp8Depay {
    name: String,
    depacketizer: Vp8Packet,
    stats: DepayStats,
    arena: Option<SharedArena>,
}

impl RtpVp8Depay {
    /// Create a new VP8 depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-vp8-depay".into(),
            depacketizer: Vp8Packet::default(),
            stats: DepayStats::default(),
            arena: None,
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &DepayStats {
        &self.stats
    }
}

impl Default for RtpVp8Depay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpVp8Depay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.packets_in += 1;

        match self.depacketizer.depacketize(&payload) {
            Ok(output) => {
                if output.is_empty() {
                    return Ok(None);
                }

                self.stats.frames_out += 1;
                self.stats.bytes_out += output.len() as u64;

                // Lazily initialize arena
                if self.arena.is_none() {
                    self.arena =
                        Some(SharedArena::new(256 * 1024, 32).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }
                let arena = self.arena.as_ref().unwrap();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
                slot.data_mut()[..output.len()].copy_from_slice(output.as_ref());

                let handle = crate::buffer::MemoryHandle::with_len(slot, output.len());

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Video(VideoCodec::Vp8));

                // VP8 keyframe detection
                if !output.is_empty() && (output[0] & 0x01) == 0 {
                    metadata.flags = metadata.flags.insert(BufferFlags::SYNC_POINT);
                }

                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("VP8 depacketize error: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// VP8 Payloader
// ============================================================================

/// VP8 RTP packetizer.
pub struct RtpVp8Pay {
    name: String,
    payloader: Vp8Payloader,
    mtu: usize,
    stats: PayStats,
    arena: Option<SharedArena>,
}

impl RtpVp8Pay {
    /// Create a new VP8 packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-vp8-pay".into(),
            payloader: Vp8Payloader::default(),
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            arena: None,
        }
    }

    /// Set the MTU.
    pub fn with_mtu(mut self, mtu: usize) -> Self {
        self.mtu = mtu;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &PayStats {
        &self.stats
    }
}

impl Default for RtpVp8Pay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpVp8Pay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => {
                if packets.is_empty() {
                    return Ok(None);
                }

                let total_len: usize = packets.iter().map(|p| p.len()).sum();
                self.stats.packets_out += packets.len() as u64;
                self.stats.bytes_out += total_len as u64;

                // Lazily initialize arena
                if self.arena.is_none() {
                    self.arena =
                        Some(SharedArena::new(256 * 1024, 32).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }
                let arena = self.arena.as_ref().unwrap();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;

                let mut offset = 0;
                for packet in &packets {
                    slot.data_mut()[offset..offset + packet.len()].copy_from_slice(packet.as_ref());
                    offset += packet.len();
                }

                let handle = crate::buffer::MemoryHandle::with_len(slot, total_len);

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Rtp(RtpFormat::new(
                    96,
                    90000,
                    RtpEncoding::Vp8,
                )));

                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("VP8 packetize error: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// VP9 Depayloader
// ============================================================================

/// VP9 RTP depacketizer.
pub struct RtpVp9Depay {
    name: String,
    depacketizer: Vp9Packet,
    stats: DepayStats,
    arena: Option<SharedArena>,
}

impl RtpVp9Depay {
    /// Create a new VP9 depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-vp9-depay".into(),
            depacketizer: Vp9Packet::default(),
            stats: DepayStats::default(),
            arena: None,
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &DepayStats {
        &self.stats
    }
}

impl Default for RtpVp9Depay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpVp9Depay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.packets_in += 1;

        match self.depacketizer.depacketize(&payload) {
            Ok(output) => {
                if output.is_empty() {
                    return Ok(None);
                }

                self.stats.frames_out += 1;
                self.stats.bytes_out += output.len() as u64;

                // Lazily initialize arena
                if self.arena.is_none() {
                    self.arena =
                        Some(SharedArena::new(256 * 1024, 32).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }
                let arena = self.arena.as_ref().unwrap();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
                slot.data_mut()[..output.len()].copy_from_slice(output.as_ref());

                let handle = crate::buffer::MemoryHandle::with_len(slot, output.len());

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Video(VideoCodec::Vp9));

                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("VP9 depacketize error: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// VP9 Payloader
// ============================================================================

/// VP9 RTP packetizer.
pub struct RtpVp9Pay {
    name: String,
    payloader: Vp9Payloader,
    mtu: usize,
    stats: PayStats,
    arena: Option<SharedArena>,
}

impl RtpVp9Pay {
    /// Create a new VP9 packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-vp9-pay".into(),
            payloader: Vp9Payloader::default(),
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            arena: None,
        }
    }

    /// Set the MTU.
    pub fn with_mtu(mut self, mtu: usize) -> Self {
        self.mtu = mtu;
        self
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &PayStats {
        &self.stats
    }
}

impl Default for RtpVp9Pay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpVp9Pay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => {
                if packets.is_empty() {
                    return Ok(None);
                }

                let total_len: usize = packets.iter().map(|p| p.len()).sum();
                self.stats.packets_out += packets.len() as u64;
                self.stats.bytes_out += total_len as u64;

                // Lazily initialize arena
                if self.arena.is_none() {
                    self.arena =
                        Some(SharedArena::new(256 * 1024, 32).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }
                let arena = self.arena.as_ref().unwrap();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;

                let mut offset = 0;
                for packet in &packets {
                    slot.data_mut()[offset..offset + packet.len()].copy_from_slice(packet.as_ref());
                    offset += packet.len();
                }

                let handle = crate::buffer::MemoryHandle::with_len(slot, total_len);

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Rtp(RtpFormat::new(
                    98,
                    90000,
                    RtpEncoding::Vp9,
                )));

                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("VP9 packetize error: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Opus Depayloader
// ============================================================================

/// Opus audio RTP depacketizer.
pub struct RtpOpusDepay {
    name: String,
    depacketizer: OpusPacket,
    stats: DepayStats,
    arena: Option<SharedArena>,
}

impl RtpOpusDepay {
    /// Create a new Opus depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-opus-depay".into(),
            depacketizer: OpusPacket::default(),
            stats: DepayStats::default(),
            arena: None,
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get statistics.
    pub fn stats(&self) -> &DepayStats {
        &self.stats
    }
}

impl Default for RtpOpusDepay {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for RtpOpusDepay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.packets_in += 1;

        match self.depacketizer.depacketize(&payload) {
            Ok(output) => {
                if output.is_empty() {
                    return Ok(None);
                }

                self.stats.frames_out += 1;
                self.stats.bytes_out += output.len() as u64;

                // Lazily initialize arena
                if self.arena.is_none() {
                    self.arena =
                        Some(SharedArena::new(64 * 1024, 32).map_err(|e| {
                            Error::Element(format!("Failed to create arena: {}", e))
                        })?);
                }
                let arena = self.arena.as_ref().unwrap();

                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
                slot.data_mut()[..output.len()].copy_from_slice(output.as_ref());

                let handle = crate::buffer::MemoryHandle::with_len(slot, output.len());

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Audio(crate::format::AudioCodec::Opus));

                Ok(Some(Buffer::new(handle, metadata)))
            }
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("Opus depacketize error: {}", e)))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics for depacketizers.
#[derive(Debug, Clone, Default)]
pub struct DepayStats {
    /// RTP packets received.
    pub packets_in: u64,
    /// Complete frames output.
    pub frames_out: u64,
    /// Bytes output.
    pub bytes_out: u64,
    /// Errors encountered.
    pub errors: u64,
}

/// Statistics for packetizers.
#[derive(Debug, Clone, Default)]
pub struct PayStats {
    /// Frames received.
    pub frames_in: u64,
    /// RTP packets output.
    pub packets_out: u64,
    /// Bytes output.
    pub bytes_out: u64,
    /// Errors encountered.
    pub errors: u64,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(256 * 1024, 64).unwrap())
    }

    fn create_test_buffer(data: &[u8]) -> Buffer {
        let mut slot = test_arena().acquire().unwrap();
        slot.data_mut()[..data.len()].copy_from_slice(data);
        let handle = crate::buffer::MemoryHandle::with_len(slot, data.len());
        Buffer::new(handle, Metadata::default())
    }

    #[test]
    fn test_h264_depay_creation() {
        let depay = RtpH264Depay::new();
        assert_eq!(depay.name(), "rtp-h264-depay");
        assert!(!depay.depacketizer.is_avc);
    }

    #[test]
    fn test_h264_depay_avc_format() {
        let depay = RtpH264Depay::new().avc_format();
        assert!(depay.depacketizer.is_avc);
    }

    #[test]
    fn test_h264_depay_single_nalu() {
        let mut depay = RtpH264Depay::new();

        // Single NAL unit (type 1 = non-IDR slice)
        // Format: [nal_header, payload...]
        let nal_data = vec![0x41, 0x01, 0x02, 0x03, 0x04];
        let buffer = create_test_buffer(&nal_data);

        let result = depay.process(buffer).unwrap();
        assert!(result.is_some());

        let output = result.unwrap();
        // Should have Annex B start code prepended
        assert!(output.as_bytes().starts_with(&[0x00, 0x00, 0x00, 0x01]));
        assert_eq!(depay.stats().packets_in, 1);
        assert_eq!(depay.stats().frames_out, 1);
    }

    #[test]
    fn test_h264_pay_creation() {
        let pay = RtpH264Pay::new().with_mtu(1200);
        assert_eq!(pay.name(), "rtp-h264-pay");
        assert_eq!(pay.mtu, 1200);
    }

    #[test]
    fn test_h265_depay_creation() {
        let depay = RtpH265Depay::new();
        assert_eq!(depay.name(), "rtp-h265-depay");
    }

    #[test]
    fn test_vp8_depay_creation() {
        let depay = RtpVp8Depay::new();
        assert_eq!(depay.name(), "rtp-vp8-depay");
    }

    #[test]
    fn test_vp9_depay_creation() {
        let depay = RtpVp9Depay::new();
        assert_eq!(depay.name(), "rtp-vp9-depay");
    }

    #[test]
    fn test_opus_depay_creation() {
        let depay = RtpOpusDepay::new();
        assert_eq!(depay.name(), "rtp-opus-depay");
    }

    #[test]
    fn test_opus_depay_process() {
        let mut depay = RtpOpusDepay::new();

        // Opus packet (just raw data, no special header)
        let opus_data = vec![0xFC, 0x01, 0x02, 0x03, 0x04, 0x05];
        let buffer = create_test_buffer(&opus_data);

        let result = depay.process(buffer).unwrap();
        assert!(result.is_some());

        let output = result.unwrap();
        assert_eq!(output.as_bytes(), &opus_data);
        assert_eq!(depay.stats().packets_in, 1);
        assert_eq!(depay.stats().frames_out, 1);
    }

    #[test]
    fn test_depay_stats() {
        let mut depay = RtpH264Depay::new();

        // Process a valid packet
        let nal_data = vec![0x41, 0x01, 0x02, 0x03];
        let buffer = create_test_buffer(&nal_data);
        let _ = depay.process(buffer);

        let stats = depay.stats();
        assert_eq!(stats.packets_in, 1);
    }

    #[test]
    fn test_pay_stats() {
        let pay = RtpH264Pay::new();
        let stats = pay.stats();
        assert_eq!(stats.frames_in, 0);
        assert_eq!(stats.packets_out, 0);
    }
}
