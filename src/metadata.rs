//! Buffer metadata types.
//!
//! Metadata accompanies every buffer in the pipeline and contains:
//! - Timestamps (PTS, DTS, duration)
//! - Sequence numbers and stream identifiers
//! - Flags (keyframe, EOS, discontinuity, etc.)
//! - Optional RTP-specific metadata
//! - Optional media format information

use crate::clock::ClockTime;
use crate::format::MediaFormat;
use rkyv::{Archive, Deserialize, Serialize};

// ============================================================================
// Buffer Flags
// ============================================================================

/// Buffer flags (1 byte).
///
/// Flags indicate buffer properties like keyframes, end-of-stream, etc.
/// Uses bitflags for efficient storage and combination.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Archive, Serialize, Deserialize)]
#[repr(transparent)]
pub struct BufferFlags(u8);

impl BufferFlags {
    /// No flags set.
    pub const NONE: Self = Self(0);
    /// Buffer is a sync point (keyframe).
    pub const SYNC_POINT: Self = Self(1 << 0);
    /// Buffer marks end of stream.
    pub const EOS: Self = Self(1 << 1);
    /// Buffer data may be corrupted or incomplete.
    pub const CORRUPTED: Self = Self(1 << 2);
    /// Buffer follows a discontinuity (gap in stream).
    pub const DISCONT: Self = Self(1 << 3);
    /// Buffer depends on previous buffers (not independently decodable).
    pub const DELTA: Self = Self(1 << 4);
    /// Buffer contains header/config data (SPS/PPS, etc.).
    pub const HEADER: Self = Self(1 << 5);
    /// Buffer should not be displayed (decode only).
    pub const DECODE_ONLY: Self = Self(1 << 6);
    /// Buffer was generated due to timeout (fallback/heartbeat).
    pub const TIMEOUT: Self = Self(1 << 7);

    /// Check if a flag is set.
    #[inline]
    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) != 0
    }

    /// Set a flag.
    #[inline]
    pub const fn insert(self, flag: Self) -> Self {
        Self(self.0 | flag.0)
    }

    /// Clear a flag.
    #[inline]
    pub const fn remove(self, flag: Self) -> Self {
        Self(self.0 & !flag.0)
    }

    /// Toggle a flag.
    #[inline]
    pub const fn toggle(self, flag: Self) -> Self {
        Self(self.0 ^ flag.0)
    }

    /// Check if this is a keyframe (sync point).
    #[inline]
    pub const fn is_keyframe(self) -> bool {
        self.contains(Self::SYNC_POINT)
    }

    /// Check if this marks end of stream.
    #[inline]
    pub const fn is_eos(self) -> bool {
        self.contains(Self::EOS)
    }

    /// Check if this follows a discontinuity.
    #[inline]
    pub const fn is_discont(self) -> bool {
        self.contains(Self::DISCONT)
    }

    /// Check if this is a delta frame (not independently decodable).
    #[inline]
    pub const fn is_delta(self) -> bool {
        self.contains(Self::DELTA)
    }

    /// Check if data may be corrupted.
    #[inline]
    pub const fn is_corrupted(self) -> bool {
        self.contains(Self::CORRUPTED)
    }

    /// Check if this contains header data.
    #[inline]
    pub const fn is_header(self) -> bool {
        self.contains(Self::HEADER)
    }

    /// Check if this is decode-only (not for display).
    #[inline]
    pub const fn is_decode_only(self) -> bool {
        self.contains(Self::DECODE_ONLY)
    }

    /// Check if this was generated from a timeout.
    #[inline]
    pub const fn is_timeout(self) -> bool {
        self.contains(Self::TIMEOUT)
    }

    /// Get the raw bits.
    #[inline]
    pub const fn bits(self) -> u8 {
        self.0
    }

    /// Create from raw bits.
    #[inline]
    pub const fn from_bits(bits: u8) -> Self {
        Self(bits)
    }
}

impl std::ops::BitOr for BufferFlags {
    type Output = Self;

    #[inline]
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for BufferFlags {
    #[inline]
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for BufferFlags {
    type Output = Self;

    #[inline]
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::BitAndAssign for BufferFlags {
    #[inline]
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

impl std::ops::Not for BufferFlags {
    type Output = Self;

    #[inline]
    fn not(self) -> Self {
        Self(!self.0)
    }
}

// ============================================================================
// RTP Metadata
// ============================================================================

/// RTP header fields (12 bytes, Copy).
///
/// Contains the essential RTP header fields needed for media processing.
/// Stored as an optional field in `Metadata` to avoid overhead for non-RTP buffers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtpMeta {
    /// RTP sequence number (16-bit, wraps).
    pub seq: u16,
    /// RTP timestamp (32-bit, codec clock rate dependent).
    pub ts: u32,
    /// Synchronization source identifier.
    pub ssrc: u32,
    /// Payload type (0-127).
    pub pt: u8,
    /// Marker bit (usually indicates frame boundary).
    pub marker: bool,
}

impl RtpMeta {
    /// Create new RTP metadata.
    pub const fn new(seq: u16, ts: u32, ssrc: u32, pt: u8, marker: bool) -> Self {
        Self {
            seq,
            ts,
            ssrc,
            pt,
            marker,
        }
    }

    /// Convert RTP timestamp to ClockTime given the clock rate.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::metadata::RtpMeta;
    /// use parallax::clock::ClockTime;
    ///
    /// let rtp = RtpMeta { ts: 90000, ..Default::default() };
    /// let time = rtp.timestamp_to_clock(90000); // 90kHz clock for video
    /// assert_eq!(time.secs(), 1);
    /// ```
    #[inline]
    pub fn timestamp_to_clock(self, clock_rate: u32) -> ClockTime {
        if clock_rate == 0 {
            return ClockTime::NONE;
        }
        ClockTime::from_nanos((self.ts as u64 * 1_000_000_000) / clock_rate as u64)
    }

    /// Convert ClockTime to RTP timestamp given the clock rate.
    #[inline]
    pub fn clock_to_timestamp(time: ClockTime, clock_rate: u32) -> u32 {
        if time.is_none() || clock_rate == 0 {
            return 0;
        }
        ((time.nanos() * clock_rate as u64) / 1_000_000_000) as u32
    }

    /// Calculate the expected next sequence number.
    #[inline]
    pub fn next_seq(self) -> u16 {
        self.seq.wrapping_add(1)
    }

    /// Check if another sequence number follows this one.
    #[inline]
    pub fn is_next_seq(self, other: u16) -> bool {
        other == self.next_seq()
    }
}

// ============================================================================
// Metadata
// ============================================================================

/// Metadata associated with a buffer.
///
/// Contains timing information, sequence numbers, flags, and optional
/// format-specific data.
///
/// # Timestamps
///
/// - `pts`: Presentation timestamp - when buffer should be displayed/rendered
/// - `dts`: Decode timestamp - when buffer should be decoded (for B-frames)
/// - `duration`: How long this buffer's content lasts
///
/// # Example
///
/// ```rust
/// use parallax::metadata::Metadata;
/// use parallax::clock::ClockTime;
///
/// let meta = Metadata::new()
///     .with_pts(ClockTime::from_millis(100))
///     .with_sequence(42)
///     .keyframe();
///
/// assert!(meta.flags.is_keyframe());
/// assert_eq!(meta.sequence, 42);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Metadata {
    /// Presentation timestamp (when this buffer should be displayed).
    pub pts: ClockTime,

    /// Decode timestamp (when this buffer should be decoded).
    /// Usually same as PTS except for B-frames.
    pub dts: ClockTime,

    /// Duration of this buffer's content.
    pub duration: ClockTime,

    /// Monotonic sequence number within a stream.
    pub sequence: u64,

    /// Stream identifier (for demultiplexed streams).
    pub stream_id: u32,

    /// Buffer flags.
    pub flags: BufferFlags,

    /// RTP-specific metadata (only for RTP buffers).
    pub rtp: Option<RtpMeta>,

    /// Media format (set on format changes or first buffer).
    pub format: Option<MediaFormat>,

    /// Byte offset in the original source (optional).
    pub offset: Option<u64>,
}

impl Metadata {
    /// Create new metadata with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create metadata with a sequence number (constructor).
    pub fn from_sequence(sequence: u64) -> Self {
        Self {
            sequence,
            ..Default::default()
        }
    }

    // Builder methods

    /// Set the sequence number (builder).
    pub fn with_sequence(mut self, sequence: u64) -> Self {
        self.sequence = sequence;
        self
    }

    /// Set the presentation timestamp.
    pub fn with_pts(mut self, pts: ClockTime) -> Self {
        self.pts = pts;
        self
    }

    /// Set the decode timestamp.
    pub fn with_dts(mut self, dts: ClockTime) -> Self {
        self.dts = dts;
        self
    }

    /// Set the duration.
    pub fn with_duration(mut self, duration: ClockTime) -> Self {
        self.duration = duration;
        self
    }

    /// Set the stream ID.
    pub fn with_stream_id(mut self, stream_id: u32) -> Self {
        self.stream_id = stream_id;
        self
    }

    /// Set the flags.
    pub fn with_flags(mut self, flags: BufferFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Set RTP metadata.
    pub fn with_rtp(mut self, rtp: RtpMeta) -> Self {
        self.rtp = Some(rtp);
        self
    }

    /// Set the media format.
    pub fn with_format(mut self, format: MediaFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Set the byte offset.
    pub fn with_offset(mut self, offset: u64) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Mark as keyframe (sync point).
    pub fn keyframe(mut self) -> Self {
        self.flags = self.flags.insert(BufferFlags::SYNC_POINT);
        self
    }

    /// Mark as end of stream.
    pub fn eos(mut self) -> Self {
        self.flags = self.flags.insert(BufferFlags::EOS);
        self
    }

    /// Mark as discontinuity.
    pub fn discont(mut self) -> Self {
        self.flags = self.flags.insert(BufferFlags::DISCONT);
        self
    }

    /// Mark as delta frame.
    pub fn delta(mut self) -> Self {
        self.flags = self.flags.insert(BufferFlags::DELTA);
        self
    }

    /// Mark as header data.
    pub fn header(mut self) -> Self {
        self.flags = self.flags.insert(BufferFlags::HEADER);
        self
    }

    // Query methods

    /// Check if this is a keyframe.
    #[inline]
    pub fn is_keyframe(&self) -> bool {
        self.flags.is_keyframe()
    }

    /// Check if this is end of stream.
    #[inline]
    pub fn is_eos(&self) -> bool {
        self.flags.is_eos()
    }

    /// Check if this is a discontinuity.
    #[inline]
    pub fn is_discont(&self) -> bool {
        self.flags.is_discont()
    }

    /// Get the effective decode timestamp (DTS if set, otherwise PTS).
    #[inline]
    pub fn effective_dts(&self) -> ClockTime {
        if self.dts.is_some() {
            self.dts
        } else {
            self.pts
        }
    }
}

// ============================================================================
// Legacy compatibility types (for rkyv serialization)
// ============================================================================

/// A key-value pair for extra metadata (legacy support).
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub struct ExtraField {
    /// Field name.
    pub key: String,
    /// Field value.
    pub value: MetadataValue,
}

/// Possible values for extra metadata fields (legacy support).
#[derive(Debug, Clone, PartialEq, Archive, Serialize, Deserialize)]
pub enum MetadataValue {
    /// String value.
    String(String),
    /// Integer value.
    Int(i64),
    /// Floating-point value.
    Float(f64),
    /// Boolean value.
    Bool(bool),
    /// Raw bytes.
    Bytes(Vec<u8>),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_flags_basic() {
        let flags = BufferFlags::NONE;
        assert!(!flags.is_keyframe());
        assert!(!flags.is_eos());

        let flags = flags.insert(BufferFlags::SYNC_POINT);
        assert!(flags.is_keyframe());
        assert!(!flags.is_eos());

        let flags = flags.insert(BufferFlags::EOS);
        assert!(flags.is_keyframe());
        assert!(flags.is_eos());
    }

    #[test]
    fn test_buffer_flags_operators() {
        let flags = BufferFlags::SYNC_POINT | BufferFlags::DISCONT;
        assert!(flags.is_keyframe());
        assert!(flags.is_discont());

        let flags = flags & BufferFlags::SYNC_POINT;
        assert!(flags.is_keyframe());
        assert!(!flags.is_discont());
    }

    #[test]
    fn test_buffer_flags_remove() {
        let flags = BufferFlags::SYNC_POINT | BufferFlags::EOS;
        let flags = flags.remove(BufferFlags::SYNC_POINT);
        assert!(!flags.is_keyframe());
        assert!(flags.is_eos());
    }

    #[test]
    fn test_rtp_meta_timestamp_conversion() {
        let rtp = RtpMeta {
            seq: 1000,
            ts: 90000, // 1 second at 90kHz
            ssrc: 12345,
            pt: 96,
            marker: true,
        };

        let time = rtp.timestamp_to_clock(90000);
        assert_eq!(time.secs(), 1);

        let ts_back = RtpMeta::clock_to_timestamp(time, 90000);
        assert_eq!(ts_back, 90000);
    }

    #[test]
    fn test_rtp_meta_sequence() {
        let rtp = RtpMeta {
            seq: 65535,
            ..Default::default()
        };
        assert_eq!(rtp.next_seq(), 0);
        assert!(rtp.is_next_seq(0));

        let rtp = RtpMeta {
            seq: 100,
            ..Default::default()
        };
        assert_eq!(rtp.next_seq(), 101);
    }

    #[test]
    fn test_metadata_builder() {
        let meta = Metadata::new()
            .with_pts(ClockTime::from_millis(100))
            .with_dts(ClockTime::from_millis(50))
            .with_duration(ClockTime::from_millis(33))
            .with_sequence(42)
            .with_stream_id(1)
            .keyframe();

        assert_eq!(meta.pts.millis(), 100);
        assert_eq!(meta.dts.millis(), 50);
        assert_eq!(meta.duration.millis(), 33);
        assert_eq!(meta.sequence, 42);
        assert_eq!(meta.stream_id, 1);
        assert!(meta.is_keyframe());
    }

    #[test]
    fn test_metadata_with_rtp() {
        let rtp = RtpMeta::new(1000, 90000, 12345, 96, true);
        let meta = Metadata::new().with_rtp(rtp);

        assert!(meta.rtp.is_some());
        let rtp = meta.rtp.unwrap();
        assert_eq!(rtp.seq, 1000);
        assert!(rtp.marker);
    }

    #[test]
    fn test_metadata_effective_dts() {
        // When DTS is set, use DTS
        let meta = Metadata::new()
            .with_pts(ClockTime::from_millis(100))
            .with_dts(ClockTime::from_millis(50));
        assert_eq!(meta.effective_dts().millis(), 50);

        // When DTS is explicitly NONE, use PTS
        let meta = Metadata::new()
            .with_pts(ClockTime::from_millis(100))
            .with_dts(ClockTime::NONE);
        assert_eq!(meta.effective_dts().millis(), 100);

        // When DTS is default (ZERO), it's considered "set" and returns 0
        let meta = Metadata::new().with_pts(ClockTime::from_millis(100));
        assert_eq!(meta.effective_dts().millis(), 0); // DTS defaults to ZERO
    }
}
