//! Buffer metadata types.
//!
//! Metadata accompanies every buffer in the pipeline and contains:
//! - Timestamps (PTS, DTS, duration)
//! - Sequence numbers and stream identifiers
//! - Flags (keyframe, EOS, discontinuity, etc.)
//! - Optional RTP-specific metadata
//! - Optional media format information
//! - Extensible custom metadata (KLV, SEI, closed captions, etc.)

use crate::clock::ClockTime;
use crate::format::MediaFormat;
use rkyv::{Archive, Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;

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
// Custom Metadata
// ============================================================================

/// Type alias for the clone function stored in MetaBox.
type CloneFn = fn(&(dyn Any + Send + Sync)) -> Box<dyn Any + Send + Sync>;

/// Type alias for the debug function stored in MetaBox.
type DebugFn = fn(&(dyn Any + Send + Sync), &mut std::fmt::Formatter<'_>) -> std::fmt::Result;

/// Wrapper for custom metadata values that can be stored in buffers.
///
/// This wrapper stores a type-erased value that can be downcast to its
/// original type. Cloning is supported via a stored function pointer.
///
/// # Example
///
/// ```rust
/// use parallax::metadata::Metadata;
///
/// #[derive(Clone, Debug)]
/// struct GpsPosition {
///     lat: f64,
///     lon: f64,
/// }
///
/// let mut meta = Metadata::new();
/// meta.set("app/gps", GpsPosition { lat: 37.0, lon: -122.0 });
/// ```
struct MetaBox {
    /// The stored value as a type-erased Any.
    value: Box<dyn Any + Send + Sync>,
    /// Function to clone the value.
    clone_fn: CloneFn,
    /// Function to debug-format the value.
    debug_fn: DebugFn,
}

impl std::fmt::Debug for MetaBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (self.debug_fn)(&*self.value, f)
    }
}

impl Clone for MetaBox {
    fn clone(&self) -> Self {
        Self {
            value: (self.clone_fn)(&*self.value),
            clone_fn: self.clone_fn,
            debug_fn: self.debug_fn,
        }
    }
}

impl MetaBox {
    fn new<T: Clone + Send + Sync + std::fmt::Debug + 'static>(value: T) -> Self {
        Self {
            value: Box::new(value),
            clone_fn: |any| {
                // SAFETY: We know the concrete type is T because we created this MetaBox with T.
                let t = any
                    .downcast_ref::<T>()
                    .expect("MetaBox type mismatch in clone");
                Box::new(t.clone())
            },
            debug_fn: |any, f| {
                // SAFETY: We know the concrete type is T because we created this MetaBox with T.
                let t = any
                    .downcast_ref::<T>()
                    .expect("MetaBox type mismatch in debug");
                std::fmt::Debug::fmt(t, f)
            },
        }
    }

    fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.value.downcast_ref()
    }

    fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.value.downcast_mut()
    }

    fn into_inner<T: 'static>(self) -> Option<T> {
        self.value.downcast().ok().map(|b| *b)
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
/// # Custom Metadata
///
/// The `custom` field allows attaching arbitrary typed data to buffers.
/// Use namespaced keys to avoid collisions: `"domain/type"`.
///
/// Common namespaces:
/// - `stanag/*` - STANAG/MISB metadata (KLV, VMTI)
/// - `h264/*` - H.264 specific (SEI, SPS, PPS)
/// - `h265/*` - H.265/HEVC specific
/// - `av1/*` - AV1 specific (metadata OBUs)
/// - `caption/*` - Closed captions (CEA-608, CEA-708)
/// - `audio/*` - Audio metadata (loudness, language)
/// - `app/*` - Application-specific data
///
/// # Example
///
/// ```rust
/// use parallax::metadata::Metadata;
/// use parallax::clock::ClockTime;
///
/// let mut meta = Metadata::new()
///     .with_pts(ClockTime::from_millis(100))
///     .with_sequence(42)
///     .keyframe();
///
/// // Attach custom data
/// meta.set("app/frame_number", 1234u32);
/// meta.set_bytes("stanag/klv", vec![0x06, 0x0E, 0x2B, 0x34]);
///
/// assert!(meta.flags.is_keyframe());
/// assert_eq!(meta.sequence, 42);
/// assert_eq!(meta.get::<u32>("app/frame_number"), Some(&1234));
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

    /// Custom metadata storage.
    ///
    /// Use `set()`, `get()`, and related methods to access.
    /// Keys should be namespaced: `"domain/type"` (e.g., `"stanag/klv"`).
    custom: HashMap<&'static str, MetaBox>,
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

    /// Create metadata with a PTS (presentation timestamp).
    pub fn from_pts(pts: crate::clock::ClockTime) -> Self {
        Self {
            pts,
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

    // ========================================================================
    // Custom Metadata Methods
    // ========================================================================

    /// Set custom metadata with a namespaced key.
    ///
    /// Keys should use the format `"domain/type"` to avoid collisions.
    /// Common namespaces: `stanag/`, `h264/`, `h265/`, `av1/`, `caption/`, `app/`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::metadata::Metadata;
    ///
    /// let mut meta = Metadata::new();
    ///
    /// // Store typed data
    /// meta.set("app/frame_id", 12345u64);
    /// meta.set("app/quality", 0.95f64);
    ///
    /// // Store a custom struct (must be Clone + Send + Sync + 'static)
    /// #[derive(Clone, Debug)]
    /// struct GpsPosition { lat: f64, lon: f64 }
    /// meta.set("app/gps", GpsPosition { lat: 37.0, lon: -122.0 });
    /// ```
    pub fn set<T: Clone + Send + Sync + std::fmt::Debug + 'static>(
        &mut self,
        key: &'static str,
        value: T,
    ) {
        self.custom.insert(key, MetaBox::new(value));
    }

    /// Get custom metadata by key.
    ///
    /// Returns `None` if the key doesn't exist or the type doesn't match.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::metadata::Metadata;
    ///
    /// let mut meta = Metadata::new();
    /// meta.set("app/count", 42u32);
    ///
    /// assert_eq!(meta.get::<u32>("app/count"), Some(&42));
    /// assert_eq!(meta.get::<u64>("app/count"), None); // Wrong type
    /// assert_eq!(meta.get::<u32>("app/other"), None); // Key not found
    /// ```
    pub fn get<T: 'static>(&self, key: &'static str) -> Option<&T> {
        self.custom.get(key)?.downcast_ref()
    }

    /// Get mutable custom metadata by key.
    ///
    /// Returns `None` if the key doesn't exist or the type doesn't match.
    pub fn get_mut<T: 'static>(&mut self, key: &'static str) -> Option<&mut T> {
        self.custom.get_mut(key)?.downcast_mut()
    }

    /// Check if custom metadata exists for a key.
    #[inline]
    pub fn has(&self, key: &'static str) -> bool {
        self.custom.contains_key(key)
    }

    /// Remove custom metadata by key.
    ///
    /// Returns the value if it existed and matched the type.
    pub fn remove<T: 'static>(&mut self, key: &'static str) -> Option<T> {
        let meta_box = self.custom.remove(key)?;
        meta_box.into_inner()
    }

    /// Get all custom metadata keys.
    pub fn custom_keys(&self) -> impl Iterator<Item = &'static str> + '_ {
        self.custom.keys().copied()
    }

    /// Get the number of custom metadata entries.
    #[inline]
    pub fn custom_len(&self) -> usize {
        self.custom.len()
    }

    /// Check if there is no custom metadata.
    #[inline]
    pub fn custom_is_empty(&self) -> bool {
        self.custom.is_empty()
    }

    /// Clear all custom metadata.
    pub fn clear_custom(&mut self) {
        self.custom.clear();
    }

    // ========================================================================
    // Convenience Methods for Common Types
    // ========================================================================

    /// Set raw bytes as custom metadata.
    ///
    /// This is a convenience method for `set(key, Vec<u8>)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::metadata::Metadata;
    ///
    /// let mut meta = Metadata::new();
    /// meta.set_bytes("h264/sei", vec![0x06, 0x05, 0x04]);
    ///
    /// assert_eq!(meta.get_bytes("h264/sei"), Some(&[0x06, 0x05, 0x04][..]));
    /// ```
    pub fn set_bytes(&mut self, key: &'static str, data: Vec<u8>) {
        self.set(key, data);
    }

    /// Get raw bytes from custom metadata.
    ///
    /// Returns `None` if the key doesn't exist or isn't stored as `Vec<u8>`.
    pub fn get_bytes(&self, key: &'static str) -> Option<&[u8]> {
        self.get::<Vec<u8>>(key).map(|v| v.as_slice())
    }

    /// Set KLV data (STANAG 4609 / MISB).
    ///
    /// Convenience method for `set_bytes("stanag/klv", data)`.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::metadata::Metadata;
    ///
    /// let mut meta = Metadata::new();
    /// meta.set_klv(vec![0x06, 0x0E, 0x2B, 0x34]);
    ///
    /// assert_eq!(meta.klv(), Some(&[0x06, 0x0E, 0x2B, 0x34][..]));
    /// ```
    pub fn set_klv(&mut self, data: Vec<u8>) {
        self.set_bytes("stanag/klv", data);
    }

    /// Get KLV data (STANAG 4609 / MISB).
    ///
    /// Convenience method for `get_bytes("stanag/klv")`.
    pub fn klv(&self) -> Option<&[u8]> {
        self.get_bytes("stanag/klv")
    }

    /// Set SEI NAL units (H.264/H.265 Supplemental Enhancement Information).
    ///
    /// Each inner `Vec<u8>` is a complete SEI NAL unit.
    pub fn set_sei(&mut self, nalus: Vec<Vec<u8>>) {
        self.set("h264/sei", nalus);
    }

    /// Get SEI NAL units.
    pub fn sei(&self) -> Option<&Vec<Vec<u8>>> {
        self.get::<Vec<Vec<u8>>>("h264/sei")
    }

    /// Set closed caption data (CEA-608 or CEA-708).
    pub fn set_captions(&mut self, cc_type: &'static str, data: Vec<u8>) {
        self.set_bytes(cc_type, data);
    }

    /// Get closed caption data.
    pub fn captions(&self, cc_type: &'static str) -> Option<&[u8]> {
        self.get_bytes(cc_type)
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

    // ========================================================================
    // Custom Metadata Tests
    // ========================================================================

    #[test]
    fn test_custom_metadata_set_get() {
        let mut meta = Metadata::new();

        // Set and get u32
        meta.set("app/count", 42u32);
        assert_eq!(meta.get::<u32>("app/count"), Some(&42));

        // Set and get String
        meta.set("app/name", "test".to_string());
        assert_eq!(meta.get::<String>("app/name"), Some(&"test".to_string()));

        // Set and get f64
        meta.set("app/quality", 0.95f64);
        assert_eq!(meta.get::<f64>("app/quality"), Some(&0.95));
    }

    #[test]
    fn test_custom_metadata_type_mismatch() {
        let mut meta = Metadata::new();
        meta.set("app/count", 42u32);

        // Wrong type should return None
        assert_eq!(meta.get::<u64>("app/count"), None);
        assert_eq!(meta.get::<String>("app/count"), None);
    }

    #[test]
    fn test_custom_metadata_missing_key() {
        let meta = Metadata::new();
        assert_eq!(meta.get::<u32>("nonexistent"), None);
    }

    #[test]
    fn test_custom_metadata_has() {
        let mut meta = Metadata::new();
        assert!(!meta.has("app/test"));

        meta.set("app/test", 1u32);
        assert!(meta.has("app/test"));
    }

    #[test]
    fn test_custom_metadata_remove() {
        let mut meta = Metadata::new();
        meta.set("app/count", 42u32);

        let removed = meta.remove::<u32>("app/count");
        assert_eq!(removed, Some(42));
        assert!(!meta.has("app/count"));

        // Remove non-existent
        let removed = meta.remove::<u32>("app/count");
        assert_eq!(removed, None);
    }

    #[test]
    fn test_custom_metadata_get_mut() {
        let mut meta = Metadata::new();
        meta.set("app/count", 42u32);

        if let Some(count) = meta.get_mut::<u32>("app/count") {
            *count = 100;
        }

        assert_eq!(meta.get::<u32>("app/count"), Some(&100));
    }

    #[test]
    fn test_custom_metadata_bytes() {
        let mut meta = Metadata::new();
        meta.set_bytes("test/data", vec![1, 2, 3, 4]);

        assert!(meta.has("test/data"));
        assert_eq!(meta.get_bytes("test/data"), Some(&[1, 2, 3, 4][..]));
        assert_eq!(meta.get_bytes("nonexistent"), None);
    }

    #[test]
    fn test_custom_metadata_klv() {
        let mut meta = Metadata::new();
        let klv_data = vec![0x06, 0x0E, 0x2B, 0x34, 0x01, 0x02];

        meta.set_klv(klv_data.clone());

        assert!(meta.has("stanag/klv"));
        assert_eq!(meta.klv(), Some(&klv_data[..]));
    }

    #[test]
    fn test_custom_metadata_sei() {
        let mut meta = Metadata::new();
        let sei_nalus = vec![vec![0x06, 0x05, 0x10], vec![0x06, 0x01, 0x20]];

        meta.set_sei(sei_nalus.clone());

        assert!(meta.has("h264/sei"));
        assert_eq!(meta.sei(), Some(&sei_nalus));
    }

    #[test]
    fn test_custom_metadata_clone() {
        let mut meta = Metadata::new();
        meta.set("app/count", 42u32);
        meta.set_bytes("app/data", vec![1, 2, 3]);

        let cloned = meta.clone();

        // Cloned metadata should have the same values
        assert_eq!(cloned.get::<u32>("app/count"), Some(&42));
        assert_eq!(cloned.get_bytes("app/data"), Some(&[1, 2, 3][..]));
    }

    #[test]
    fn test_custom_metadata_keys_and_len() {
        let mut meta = Metadata::new();
        assert!(meta.custom_is_empty());
        assert_eq!(meta.custom_len(), 0);

        meta.set("app/a", 1u32);
        meta.set("app/b", 2u32);
        meta.set("app/c", 3u32);

        assert!(!meta.custom_is_empty());
        assert_eq!(meta.custom_len(), 3);

        let keys: Vec<_> = meta.custom_keys().collect();
        assert!(keys.contains(&"app/a"));
        assert!(keys.contains(&"app/b"));
        assert!(keys.contains(&"app/c"));
    }

    #[test]
    fn test_custom_metadata_clear() {
        let mut meta = Metadata::new();
        meta.set("app/a", 1u32);
        meta.set("app/b", 2u32);

        assert_eq!(meta.custom_len(), 2);

        meta.clear_custom();

        assert!(meta.custom_is_empty());
        assert_eq!(meta.custom_len(), 0);
    }

    #[test]
    fn test_custom_metadata_overwrite() {
        let mut meta = Metadata::new();
        meta.set("app/count", 42u32);
        meta.set("app/count", 100u32);

        assert_eq!(meta.get::<u32>("app/count"), Some(&100));
        assert_eq!(meta.custom_len(), 1);
    }

    #[test]
    fn test_custom_metadata_struct() {
        #[derive(Clone, Debug, PartialEq)]
        struct GpsPosition {
            lat: f64,
            lon: f64,
        }

        let mut meta = Metadata::new();
        meta.set(
            "app/gps",
            GpsPosition {
                lat: 37.7749,
                lon: -122.4194,
            },
        );

        let gps = meta.get::<GpsPosition>("app/gps").unwrap();
        assert_eq!(gps.lat, 37.7749);
        assert_eq!(gps.lon, -122.4194);

        // Clone should work
        let cloned = meta.clone();
        let cloned_gps = cloned.get::<GpsPosition>("app/gps").unwrap();
        assert_eq!(cloned_gps, gps);
    }
}
