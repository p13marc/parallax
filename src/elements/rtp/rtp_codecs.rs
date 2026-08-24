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
//! - [`RtpAv1Pay`]: AV1 packetizer (no depayloader yet)
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
use crate::element::{Element, Output, Transform};
use crate::error::{Error, Result};
use crate::format::{MediaFormat, RtpEncoding, RtpFormat, VideoCodec};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::{BufferFlags, RtpMeta};

use bytes::Bytes;
use rtp::codecs::av1::Av1Payloader;
use rtp::codecs::h264::{H264Packet, H264Payloader};
use rtp::codecs::h265::{H265Packet, HevcPayloader};
use rtp::codecs::opus::{OpusPacket, OpusPayloader};
use rtp::codecs::vp8::{Vp8Packet, Vp8Payloader};
use rtp::codecs::vp9::{Vp9Packet, Vp9Payloader};
use rtp::packetizer::{Depacketizer, Payloader};

/// Default MTU for RTP payloaders.
const DEFAULT_MTU: usize = 1400;

// ============================================================================
// H.264 Depayloader
// ============================================================================

/// Whether a length-prefixed (AVC/HVCC) buffer contains a decoder entry point.
///
/// `annexb::is_entry_point` scans for start codes, so it cannot read the
/// `avc_format()` output — 4-byte big-endian lengths instead. Same NAL-type
/// rules, different framing.
fn length_prefixed_has_entry_point(data: &[u8], codec: crate::codec::annexb::NalCodec) -> bool {
    use crate::codec::annexb::NalCodec;

    let mut i = 0usize;
    while i + 4 <= data.len() {
        let len = u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize;
        let start = i + 4;
        let end = match start.checked_add(len) {
            Some(e) if e <= data.len() => e,
            _ => break,
        };
        if start < end {
            let nal_type = match codec {
                NalCodec::H264 => data[start] & 0x1F,
                NalCodec::H265 => (data[start] >> 1) & 0x3F,
            };
            let entry = match codec {
                NalCodec::H264 => nal_type == 5,
                NalCodec::H265 => (16..=23).contains(&nal_type),
            };
            if entry {
                return true;
            }
        }
        i = end;
    }
    false
}

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
    /// Mirrors `depacketizer.is_avc`: the keyframe scan needs to know which
    /// framing it is reading.
    is_avc: bool,
    depacketizer: H264Packet,
    stats: DepayStats,
    output: OutputArena,
}

impl RtpH264Depay {
    /// Create a new H.264 depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-h264-depay".into(),
            is_avc: false,
            depacketizer: H264Packet::default(),
            stats: DepayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::VIDEO_ENCODED_SLOT_SIZE)
                .grow_to_fit(),
        }
    }

    /// Set AVC output format (4-byte length prefix instead of Annex B start codes).
    pub fn avc_format(mut self) -> Self {
        self.depacketizer.is_avc = true;
        self.is_avc = true;
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
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

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

                let mut slot = self.output.acquire(output.len(), "rtph264depay")?;
                slot.data_mut()[..output.len()].copy_from_slice(output.as_ref());

                let handle = crate::buffer::MemoryHandle::with_len(slot, output.len());

                // Preserve metadata, update format
                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Video(VideoCodec::H264));

                // Scan every NAL, not just the first (#239). The canonical
                // keyframe an encoder or RTSP server ships is a STAP-A of
                // `SPS | PPS | IDR`, so reading only `output[4]` saw the SPS
                // (type 7), decided "not a keyframe", and left the one access
                // unit that carries the parameter sets unflagged.
                let entry_point = if self.is_avc {
                    length_prefixed_has_entry_point(&output, crate::codec::annexb::NalCodec::H264)
                } else {
                    crate::codec::annexb::is_entry_point(
                        &output,
                        crate::codec::annexb::NalCodec::H264,
                    )
                };
                metadata.flags = if entry_point {
                    metadata.flags.insert(BufferFlags::SYNC_POINT)
                } else {
                    // The inherited flag came from RtpSrc, which sets
                    // SYNC_POINT from the RTP *marker* bit — "last packet of an
                    // access unit", not "keyframe". Carrying it through would
                    // be worse than saying nothing.
                    metadata.flags.remove(BufferFlags::SYNC_POINT)
                };

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
// Packetization (#218)
// ============================================================================

/// The RTP clock for video, in Hz. Every video payload type in RFC 3551 and
/// its successors uses 90 kHz.
const VIDEO_CLOCK_RATE: u32 = 90_000;

/// Per-packet stamping, shared by every payloader (#218).
///
/// A payloader used to concatenate an access unit's packets into one buffer,
/// which `RtpSink` then wrapped in a *single* RTP header — so any frame that
/// fragmented (every keyframe) went out as one malformed, oversized packet.
/// The fix is one buffer per packet, and stamping the header fields that only
/// the packetizer knows:
///
/// - `seq` is monotonic across the stream, not per access unit;
/// - `ts` is **shared** by every packet of one access unit — that is what
///   makes them one frame to a receiver;
/// - `marker` is set on the **last** packet of the access unit, which is what
///   the bit means. `RtpSink`'s fallback set it from `is_keyframe()`, which is
///   a different question entirely.
///
/// `ssrc` and `pt` stay 0: they are transport identity, owned by the sink,
/// which already overrides them with its own.
struct PayStamp {
    next_seq: u16,
    clock_rate: u32,
}

impl PayStamp {
    fn new(clock_rate: u32) -> Self {
        Self {
            next_seq: 0,
            clock_rate,
        }
    }

    /// One buffer per packet, over **one** arena slot.
    ///
    /// The packets are copied contiguously and handed out as zero-copy slices
    /// of that slot. A slot each would be the obvious shape and the wrong one:
    /// `Output::Multiple` hands the executor every buffer at once, so a
    /// 60 KB keyframe at a 1200-byte MTU would need fifty slots live
    /// simultaneously and exhaust an arena sized for the link.
    fn emit(
        &mut self,
        packets: &[Bytes],
        input: &Buffer,
        format: MediaFormat,
        arena: &mut OutputArena,
        stats: &mut PayStats,
        name: &'static str,
    ) -> Result<Output> {
        if packets.is_empty() {
            return Ok(Output::None);
        }

        let total: usize = packets.iter().map(|p| p.len()).sum();
        let mut slot = arena.acquire(total, name)?;
        let mut offset = 0;
        for packet in packets {
            slot.data_mut()[offset..offset + packet.len()].copy_from_slice(packet.as_ref());
            offset += packet.len();
        }
        let whole = crate::buffer::MemoryHandle::with_len(slot, total);

        let ts = RtpMeta::clock_to_timestamp(input.metadata().pts, self.clock_rate);
        let last = packets.len() - 1;
        let mut out = Vec::with_capacity(packets.len());
        let mut offset = 0;
        for (i, packet) in packets.iter().enumerate() {
            let mut metadata = input.metadata().clone();
            metadata.format = Some(format.clone());
            metadata.rtp = Some(RtpMeta::new(self.next_seq, ts, 0, 0, i == last));
            self.next_seq = self.next_seq.wrapping_add(1);
            out.push(Buffer::new(whole.slice(offset, packet.len()), metadata));
            offset += packet.len();
        }

        stats.packets_out += packets.len() as u64;
        stats.bytes_out += total as u64;
        Ok(Output::Multiple(out))
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
    /// Per-packet RTP header stamping (#218).
    stamp: PayStamp,
    payloader: H264Payloader,
    mtu: usize,
    stats: PayStats,
    output: OutputArena,
}

impl RtpH264Pay {
    /// Create a new H.264 packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-h264-pay".into(),
            stamp: PayStamp::new(VIDEO_CLOCK_RATE),
            payloader: H264Payloader::default(),
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::VIDEO_ENCODED_SLOT_SIZE)
                .grow_to_fit(),
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

impl Transform for RtpH264Pay {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    /// One buffer per RTP packet (#218).
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => self.stamp.emit(
                &packets,
                &buffer,
                MediaFormat::Rtp(RtpFormat::H264),
                &mut self.output,
                &mut self.stats,
                "rtph264pay",
            ),
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("H.264 packetize error: {}", e)))
            }
        }
    }

    /// A payloader accepts the codec it packetizes and emits RTP (#218).
    ///
    /// No RTP element declared caps before this, so a mis-wired graph got no
    /// answer from `prepare()` at all.
    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Video(VideoCodec::H264),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Rtp(RtpFormat::H264),
            MemoryCaps::cpu_only(),
        )])
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
    output: OutputArena,
}

impl RtpH265Depay {
    /// Create a new H.265 depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-h265-depay".into(),
            depacketizer: H265Packet::default(),
            stats: DepayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::VIDEO_ENCODED_SLOT_SIZE)
                .grow_to_fit(),
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
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

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

                let mut slot = self.output.acquire(output.len(), "rtph265depay")?;
                slot.data_mut()[..output.len()].copy_from_slice(output.as_ref());

                let handle = crate::buffer::MemoryHandle::with_len(slot, output.len());

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Video(VideoCodec::H265));

                // H.265 had no keyframe detection at all (#239), and inherited
                // a SYNC_POINT that RtpSrc set from the RTP marker bit — which
                // means "end of access unit", not "keyframe".
                metadata.flags = if crate::codec::annexb::is_entry_point(
                    &output,
                    crate::codec::annexb::NalCodec::H265,
                ) {
                    metadata.flags.insert(BufferFlags::SYNC_POINT)
                } else {
                    metadata.flags.remove(BufferFlags::SYNC_POINT)
                };

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
    /// Per-packet RTP header stamping (#218).
    stamp: PayStamp,
    payloader: HevcPayloader,
    mtu: usize,
    stats: PayStats,
    output: OutputArena,
}

impl RtpH265Pay {
    /// Create a new H.265 packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-h265-pay".into(),
            stamp: PayStamp::new(VIDEO_CLOCK_RATE),
            payloader: HevcPayloader::default(),
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::VIDEO_ENCODED_SLOT_SIZE)
                .grow_to_fit(),
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

impl Transform for RtpH265Pay {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    /// One buffer per RTP packet (#218).
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => self.stamp.emit(
                &packets,
                &buffer,
                MediaFormat::Rtp(RtpFormat::H265),
                &mut self.output,
                &mut self.stats,
                "rtph265pay",
            ),
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("H.265 packetize error: {}", e)))
            }
        }
    }

    /// A payloader accepts the codec it packetizes and emits RTP (#218).
    ///
    /// No RTP element declared caps before this, so a mis-wired graph got no
    /// answer from `prepare()` at all.
    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Video(VideoCodec::H265),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Rtp(RtpFormat::H265),
            MemoryCaps::cpu_only(),
        )])
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
    output: OutputArena,
}

impl RtpVp8Depay {
    /// Create a new VP8 depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-vp8-depay".into(),
            depacketizer: Vp8Packet::default(),
            stats: DepayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::VIDEO_ENCODED_SLOT_SIZE)
                .grow_to_fit(),
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
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

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

                let mut slot = self.output.acquire(output.len(), "rtpvp8depay")?;
                slot.data_mut()[..output.len()].copy_from_slice(output.as_ref());

                let handle = crate::buffer::MemoryHandle::with_len(slot, output.len());

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Video(VideoCodec::Vp8));

                // VP8 read the frame-type bit correctly but only ever *set*
                // the flag (#239), so an inter frame kept the SYNC_POINT
                // `RtpSrc` had set from the RTP marker bit — which means "end
                // of access unit", not "keyframe". The verdict has to clear it
                // too. An unreadable frame leaves the flag alone rather than
                // asserting anything about bytes it could not parse.
                metadata.flags = match crate::codec::vpx::vp8_is_keyframe(&output) {
                    Some(true) => metadata.flags.insert(BufferFlags::SYNC_POINT),
                    Some(false) => metadata.flags.remove(BufferFlags::SYNC_POINT),
                    None => metadata.flags,
                };

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
    /// Per-packet RTP header stamping (#218).
    stamp: PayStamp,
    payloader: Vp8Payloader,
    mtu: usize,
    stats: PayStats,
    output: OutputArena,
}

impl RtpVp8Pay {
    /// Create a new VP8 packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-vp8-pay".into(),
            stamp: PayStamp::new(VIDEO_CLOCK_RATE),
            payloader: Vp8Payloader::default(),
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::VIDEO_ENCODED_SLOT_SIZE)
                .grow_to_fit(),
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

impl Transform for RtpVp8Pay {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    /// One buffer per RTP packet (#218).
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => self.stamp.emit(
                &packets,
                &buffer,
                MediaFormat::Rtp(RtpFormat::new(96, 90000, RtpEncoding::Vp8)),
                &mut self.output,
                &mut self.stats,
                "rtpvp8pay",
            ),
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("VP8 packetize error: {}", e)))
            }
        }
    }

    /// A payloader accepts the codec it packetizes and emits RTP (#218).
    ///
    /// No RTP element declared caps before this, so a mis-wired graph got no
    /// answer from `prepare()` at all.
    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Video(VideoCodec::Vp8),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Rtp(RtpFormat::new(96, 90000, RtpEncoding::Vp8)),
            MemoryCaps::cpu_only(),
        )])
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
    output: OutputArena,
}

impl RtpVp9Depay {
    /// Create a new VP9 depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-vp9-depay".into(),
            depacketizer: Vp9Packet::default(),
            stats: DepayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::VIDEO_ENCODED_SLOT_SIZE)
                .grow_to_fit(),
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
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

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

                let mut slot = self.output.acquire(output.len(), "rtpvp9depay")?;
                slot.data_mut()[..output.len()].copy_from_slice(output.as_ref());

                let handle = crate::buffer::MemoryHandle::with_len(slot, output.len());

                let mut metadata = buffer.metadata().clone();
                metadata.format = Some(MediaFormat::Video(VideoCodec::Vp9));

                // VP9 had no detection at all (#239) and inherited the RTP
                // marker bit's SYNC_POINT, so roughly every frame claimed to
                // be an entry point. `annexb` does not apply here — VP9 has no
                // start codes — so the uncompressed header answers it.
                metadata.flags = match crate::codec::vpx::vp9_is_keyframe(&output) {
                    Some(true) => metadata.flags.insert(BufferFlags::SYNC_POINT),
                    Some(false) => metadata.flags.remove(BufferFlags::SYNC_POINT),
                    None => metadata.flags,
                };

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
    /// Per-packet RTP header stamping (#218).
    stamp: PayStamp,
    payloader: Vp9Payloader,
    mtu: usize,
    stats: PayStats,
    output: OutputArena,
}

impl RtpVp9Pay {
    /// Create a new VP9 packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-vp9-pay".into(),
            stamp: PayStamp::new(VIDEO_CLOCK_RATE),
            payloader: Vp9Payloader::default(),
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::VIDEO_ENCODED_SLOT_SIZE)
                .grow_to_fit(),
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

impl Transform for RtpVp9Pay {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    /// One buffer per RTP packet (#218).
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => self.stamp.emit(
                &packets,
                &buffer,
                MediaFormat::Rtp(RtpFormat::new(98, 90000, RtpEncoding::Vp9)),
                &mut self.output,
                &mut self.stats,
                "rtpvp9pay",
            ),
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("VP9 packetize error: {}", e)))
            }
        }
    }

    /// A payloader accepts the codec it packetizes and emits RTP (#218).
    ///
    /// No RTP element declared caps before this, so a mis-wired graph got no
    /// answer from `prepare()` at all.
    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Video(VideoCodec::Vp9),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Rtp(RtpFormat::new(98, 90000, RtpEncoding::Vp9)),
            MemoryCaps::cpu_only(),
        )])
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
    output: OutputArena,
}

impl RtpOpusDepay {
    /// Create a new Opus depacketizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-opus-depay".into(),
            depacketizer: OpusPacket,
            stats: DepayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::NETWORK_SLOT_SIZE)
                .grow_to_fit(),
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
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

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

                let mut slot = self.output.acquire(output.len(), "rtpopusdepay")?;
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
// Opus Payloader
// ============================================================================

/// Opus RTP packetizer (RFC 7587).
///
/// Wraps encoded Opus frames in RTP payloads — one frame per packet, no
/// fragmentation (Opus frames are far below any sane MTU).
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::RtpOpusPay;
///
/// let pay = RtpOpusPay::new();
/// ```
pub struct RtpOpusPay {
    name: String,
    /// Per-packet RTP header stamping (#218).
    stamp: PayStamp,
    payloader: OpusPayloader,
    mtu: usize,
    stats: PayStats,
    output: OutputArena,
}

impl RtpOpusPay {
    /// Create a new Opus packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-opus-pay".into(),
            stamp: PayStamp::new(48_000),
            payloader: OpusPayloader,
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::NETWORK_SLOT_SIZE)
                .grow_to_fit(),
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

impl Default for RtpOpusPay {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for RtpOpusPay {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    /// One buffer per RTP packet (#218).
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => self.stamp.emit(
                &packets,
                &buffer,
                MediaFormat::Rtp(RtpFormat::OPUS),
                &mut self.output,
                &mut self.stats,
                "rtpopuspay",
            ),
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("Opus packetize error: {}", e)))
            }
        }
    }

    /// A payloader accepts the codec it packetizes and emits RTP (#218).
    ///
    /// No RTP element declared caps before this, so a mis-wired graph got no
    /// answer from `prepare()` at all.
    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Audio(crate::format::AudioCodec::Opus),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Rtp(RtpFormat::OPUS),
            MemoryCaps::cpu_only(),
        )])
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// ============================================================================
// AV1 Payloader
// ============================================================================

/// AV1 RTP packetizer (AV1 RTP spec: OBU aggregation with LEB128 sizes).
///
/// Packetizes AV1 temporal units into RTP payloads. There is no matching
/// depayloader yet — the rtp crate (0.14) ships only the packetizer — so
/// this is one-way interop towards RTP-native AV1 receivers.
pub struct RtpAv1Pay {
    name: String,
    /// Per-packet RTP header stamping (#218).
    stamp: PayStamp,
    payloader: Av1Payloader,
    mtu: usize,
    stats: PayStats,
    output: OutputArena,
}

impl RtpAv1Pay {
    /// Create a new AV1 packetizer.
    pub fn new() -> Self {
        Self {
            name: "rtp-av1-pay".into(),
            stamp: PayStamp::new(VIDEO_CLOCK_RATE),
            payloader: Av1Payloader {},
            mtu: DEFAULT_MTU,
            stats: PayStats::default(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::VIDEO_ENCODED_SLOT_SIZE)
                .grow_to_fit(),
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

impl Default for RtpAv1Pay {
    fn default() -> Self {
        Self::new()
    }
}

impl Transform for RtpAv1Pay {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    /// One buffer per RTP packet (#218).
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let payload = Bytes::copy_from_slice(buffer.as_bytes());
        self.stats.frames_in += 1;

        match self.payloader.payload(self.mtu, &payload) {
            Ok(packets) => self.stamp.emit(
                &packets,
                &buffer,
                MediaFormat::Rtp(RtpFormat::new(98, 90000, crate::format::RtpEncoding::Av1)),
                &mut self.output,
                &mut self.stats,
                "rtpav1pay",
            ),
            Err(e) => {
                self.stats.errors += 1;
                Err(Error::Element(format!("AV1 packetize error: {}", e)))
            }
        }
    }

    /// A payloader accepts the codec it packetizes and emits RTP (#218).
    ///
    /// No RTP element declared caps before this, so a mis-wired graph got no
    /// answer from `prepare()` at all.
    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Video(VideoCodec::Av1),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps};
        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Rtp(RtpFormat::new(98, 90000, crate::format::RtpEncoding::Av1)),
            MemoryCaps::cpu_only(),
        )])
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
    use crate::memory::SharedArena;
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
    fn a_stap_a_keyframe_is_recognised() {
        // #239: the canonical keyframe is SPS|PPS|IDR, so a scan that looked
        // only at the first NAL saw type 7 and called it a delta frame.
        use crate::codec::annexb::{NalCodec, is_entry_point};

        let stap_a = [
            0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E, // SPS
            0, 0, 0, 1, 0x68, 0xCE, 0x38, 0x80, // PPS
            0, 0, 0, 1, 0x65, 0xAA, 0xBB, // IDR
        ];
        assert!(is_entry_point(&stap_a, NalCodec::H264));
        // The old first-NAL-only reading:
        assert_ne!(stap_a[4] & 0x1F, 5, "first NAL is the SPS, not the IDR");

        let delta = [0, 0, 0, 1, 0x41, 0x9A, 0x02];
        assert!(!is_entry_point(&delta, NalCodec::H264));
    }

    #[test]
    fn length_prefixed_keyframes_are_recognised_too() {
        use crate::codec::annexb::NalCodec;

        // avcC framing: 4-byte big-endian length, then the NAL.
        let mut avcc = Vec::new();
        for nal in [
            vec![0x67u8, 0x42, 0x00, 0x1E],
            vec![0x68, 0xCE, 0x38, 0x80],
            vec![0x65, 0xAA, 0xBB],
        ] {
            avcc.extend_from_slice(&(nal.len() as u32).to_be_bytes());
            avcc.extend_from_slice(&nal);
        }
        assert!(super::length_prefixed_has_entry_point(
            &avcc,
            NalCodec::H264
        ));

        let mut delta = Vec::new();
        let nal = [0x41u8, 0x9A, 0x02];
        delta.extend_from_slice(&(nal.len() as u32).to_be_bytes());
        delta.extend_from_slice(&nal);
        assert!(!super::length_prefixed_has_entry_point(
            &delta,
            NalCodec::H264
        ));

        // A truncated length must not panic or over-read.
        assert!(!super::length_prefixed_has_entry_point(
            &[0, 0, 0, 99, 0x65],
            NalCodec::H264
        ));
    }

    #[test]
    fn hevc_irap_types_are_recognised() {
        use crate::codec::annexb::{NalCodec, is_entry_point};

        // IDR_W_RADL is type 19; the two-byte header puts it in bits 1..6.
        let irap = [0, 0, 0, 1, 19u8 << 1, 0x01, 0xAA];
        assert!(is_entry_point(&irap, NalCodec::H265));
        // TRAIL_R (type 1) is not an entry point.
        let trail = [0, 0, 0, 1, 1u8 << 1, 0x01, 0xAA];
        assert!(!is_entry_point(&trail, NalCodec::H265));
    }

    /// Payload a frame and return the packets, one buffer each (#218).
    fn packetize(out: Output) -> Vec<Buffer> {
        match out {
            Output::Multiple(v) => v,
            Output::Single(b) => vec![b],
            Output::None => Vec::new(),
        }
    }

    #[test]
    fn test_h264_depay_creation() {
        let depay = RtpH264Depay::new();
        assert_eq!(Element::name(&depay), "rtp-h264-depay");
        assert!(!depay.depacketizer.is_avc);
    }

    /// Opus pay -> depay round-trip: one frame per packet (RFC 7587), the
    /// payload is the frame itself.
    #[test]
    fn test_opus_pay_depay_roundtrip() {
        let opus_frame = b"\xfc\x01\x02\x03\x04opus-ish";
        let mut pay = RtpOpusPay::new();
        let packets = packetize(pay.transform(create_test_buffer(opus_frame)).unwrap());
        assert_eq!(packets.len(), 1, "one Opus frame is one RTP packet");
        let paid = packets.into_iter().next().unwrap();
        assert_eq!(pay.stats().packets_out, 1);
        assert!(matches!(paid.metadata().format, Some(MediaFormat::Rtp(_))));
        // #218: the last (here only) packet of an access unit carries the
        // marker bit, and every packet carries its own header fields.
        let rtp = paid.metadata().rtp.expect("per-packet RtpMeta");
        assert!(rtp.marker);

        let mut depay = RtpOpusDepay::new();
        let depaid = depay.process(paid).unwrap().expect("frame out");
        assert_eq!(depaid.as_bytes(), opus_frame);
    }

    /// AV1 payloads carry the aggregation header + LEB128-sized OBUs; a
    /// small temporal unit fits one packet, a large one fragments per MTU.
    #[test]
    fn test_av1_pay_packetizes() {
        // OBU_FRAME (type 6, has-size), leb128 size 3, 3 payload bytes.
        // (Temporal delimiters alone are stripped per the AV1 RTP spec.)
        let small = [0x32u8, 0x03, 1, 2, 3];
        let mut pay = RtpAv1Pay::new();
        let packets = packetize(pay.transform(create_test_buffer(&small)).unwrap());
        assert_eq!(packets.len(), 1);
        assert!(!packets[0].is_empty());
        assert_eq!(pay.stats().packets_out, 1);

        // A large payload must fragment into multiple packets.
        let mut pay = RtpAv1Pay::new().with_mtu(200);
        let mut large = vec![0x12u8, 0x00];
        // OBU_FRAME (type 6) with has_size flag, LEB128 size 600.
        large.push(0x32);
        large.extend_from_slice(&[0xD8, 0x04]); // leb128(600)
        large.extend(std::iter::repeat_n(0xAB, 600));
        pay.transform(create_test_buffer(&large)).unwrap();
        assert!(
            pay.stats().packets_out > 1,
            "600-byte OBU at MTU 200 must fragment, got {} packets",
            pay.stats().packets_out
        );
    }

    /// A buffer whose metadata already claims SYNC_POINT — which is what
    /// `RtpSrc` hands a depayloader, because it sets the flag from the RTP
    /// *marker* bit ("end of access unit"), not from the bitstream.
    fn marked_buffer(data: &[u8]) -> Buffer {
        let arena = test_arena();
        arena.reclaim();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..data.len()].copy_from_slice(data);
        let handle = crate::buffer::MemoryHandle::with_len(slot, data.len());
        let mut metadata = Metadata::default();
        metadata.flags = metadata.flags.insert(BufferFlags::SYNC_POINT);
        Buffer::new(handle, metadata)
    }

    /// #239: the VPx depayloaders must *clear* the inherited marker-bit flag
    /// on an inter frame, not merely fail to set it.
    #[test]
    fn the_vpx_depayloaders_clear_an_inherited_sync_point() {
        // VP8 frame tag: bit 0 is frame_type, inverted.
        let vp8_key = [0x00u8, 0x9D, 0x01, 0x2A];
        let vp8_inter = [0x01u8, 0x9D, 0x01, 0x2A];
        for (frame, expect_key) in [(&vp8_key[..], true), (&vp8_inter[..], false)] {
            let mut pay = RtpVp8Pay::new();
            let packets = packetize(pay.transform(create_test_buffer(frame)).unwrap());
            let packet = packets.into_iter().next().expect("one packet");
            let mut depay = RtpVp8Depay::new();
            let out = depay
                .process(marked_buffer(packet.as_bytes()))
                .unwrap()
                .expect("frame out");
            // Proves the round-trip really carried the frame — without this
            // an unparsed payload would keep the inherited flag and the
            // keyframe half of the assertion would pass for the wrong reason.
            assert_eq!(out.as_bytes(), frame, "the descriptor is stripped");
            assert_eq!(
                out.metadata().flags.is_keyframe(),
                expect_key,
                "VP8 keyframe verdict must come from the bitstream, not the marker bit"
            );
        }

        // VP9 profile 0: frame_marker=10, profile bits 0, show_existing=0,
        // then frame_type.
        let vp9_key = [0b1000_0000u8, 0x49, 0x83, 0x42];
        let vp9_inter = [0b1000_0100u8, 0x49, 0x83, 0x42];
        for (frame, expect_key) in [(&vp9_key[..], true), (&vp9_inter[..], false)] {
            let mut pay = RtpVp9Pay::new();
            let packets = packetize(pay.transform(create_test_buffer(frame)).unwrap());
            let packet = packets.into_iter().next().expect("one packet");
            let mut depay = RtpVp9Depay::new();
            let out = depay
                .process(marked_buffer(packet.as_bytes()))
                .unwrap()
                .expect("frame out");
            // Proves the round-trip really carried the frame — without this
            // an unparsed payload would keep the inherited flag and the
            // keyframe half of the assertion would pass for the wrong reason.
            assert_eq!(out.as_bytes(), frame, "the descriptor is stripped");
            assert_eq!(
                out.metadata().flags.is_keyframe(),
                expect_key,
                "VP9 keyframe verdict must come from the uncompressed header"
            );
        }
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
        assert_eq!(Transform::name(&pay), "rtp-h264-pay");
        assert_eq!(pay.mtu, 1200);
    }

    #[test]
    fn test_h265_depay_creation() {
        let depay = RtpH265Depay::new();
        assert_eq!(Element::name(&depay), "rtp-h265-depay");
    }

    #[test]
    fn test_vp8_depay_creation() {
        let depay = RtpVp8Depay::new();
        assert_eq!(Element::name(&depay), "rtp-vp8-depay");
    }

    #[test]
    fn test_vp9_depay_creation() {
        let depay = RtpVp9Depay::new();
        assert_eq!(Element::name(&depay), "rtp-vp9-depay");
    }

    #[test]
    fn test_opus_depay_creation() {
        let depay = RtpOpusDepay::new();
        assert_eq!(Element::name(&depay), "rtp-opus-depay");
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
