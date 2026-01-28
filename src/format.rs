//! Media format and capabilities types.
//!
//! This module provides type-safe media format descriptions for buffers
//! and element capabilities (caps) for format negotiation.
//!
//! # Design Principles
//!
//! - **Type safety**: Use enums instead of stringly-typed formats
//! - **Zero-cost**: Small, Copy types wherever possible
//! - **Explicit**: Clear format descriptions, no implicit conversions
//!
//! # Caps Negotiation
//!
//! The caps system supports constraint-based negotiation:
//!
//! - [`CapsValue<T>`]: A value that can be fixed, range, list, or any
//! - [`VideoFormatCaps`]: Video format with constraints
//! - [`AudioFormatCaps`]: Audio format with constraints
//! - [`MemoryCaps`]: Memory type constraints
//! - [`MediaCaps`]: Combined format + memory caps
//!
//! ```rust,ignore
//! use parallax::format::{CapsValue, VideoFormatCaps, PixelFormat, Framerate};
//!
//! // Element accepts 1080p or 720p, any framerate
//! let caps = VideoFormatCaps {
//!     width: CapsValue::List(vec![1920, 1280]),
//!     height: CapsValue::List(vec![1080, 720]),
//!     pixel_format: CapsValue::Fixed(PixelFormat::I420),
//!     framerate: CapsValue::Any,
//! };
//!
//! // Find common ground with another element
//! let negotiated = caps.intersect(&other_caps)?;
//! let fixed = negotiated.fixate()?;
//! ```

use crate::memory::MemoryType;
use smallvec::SmallVec;

// ============================================================================
// CapsValue - constraint value for negotiation
// ============================================================================

/// A value that can be fixed, range, list, or any.
///
/// Used in caps negotiation to express constraints on format parameters.
/// Supports intersection (finding common ground) and fixation (choosing a value).
///
/// # Examples
///
/// ```rust
/// use parallax::format::CapsValue;
///
/// // Fixed value
/// let fixed: CapsValue<u32> = CapsValue::Fixed(1920);
///
/// // Range of acceptable values
/// let range: CapsValue<u32> = CapsValue::Range { min: 720, max: 1920 };
///
/// // List of acceptable values (ordered by preference)
/// let list: CapsValue<u32> = CapsValue::List(vec![1920, 1280, 720]);
///
/// // Any value accepted
/// let any: CapsValue<u32> = CapsValue::Any;
///
/// // Intersection finds common ground
/// assert_eq!(fixed.intersect(&range), Some(CapsValue::Fixed(1920)));
/// ```
#[derive(Clone, Debug, PartialEq, Default)]
pub enum CapsValue<T> {
    /// Exact value (fully constrained).
    Fixed(T),
    /// Range of acceptable values (inclusive).
    Range {
        /// Minimum acceptable value.
        min: T,
        /// Maximum acceptable value.
        max: T,
    },
    /// List of acceptable values (ordered by preference, first is best).
    List(Vec<T>),
    /// Any value accepted (unconstrained).
    #[default]
    Any,
}

impl<T: Clone + Ord> CapsValue<T> {
    /// Check if a value is accepted by this constraint.
    pub fn accepts(&self, value: &T) -> bool {
        match self {
            Self::Fixed(v) => v == value,
            Self::Range { min, max } => value >= min && value <= max,
            Self::List(values) => values.contains(value),
            Self::Any => true,
        }
    }

    /// Intersect two constraints, finding common values.
    ///
    /// Returns `None` if there's no overlap.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            // Any intersects with anything
            (Self::Any, other) => Some(other.clone()),
            (self_, Self::Any) => Some(self_.clone()),

            // Fixed vs Fixed: must be equal
            (Self::Fixed(a), Self::Fixed(b)) => {
                if a == b {
                    Some(Self::Fixed(a.clone()))
                } else {
                    None
                }
            }

            // Fixed vs Range: fixed must be in range
            (Self::Fixed(v), Self::Range { min, max })
            | (Self::Range { min, max }, Self::Fixed(v)) => {
                if v >= min && v <= max {
                    Some(Self::Fixed(v.clone()))
                } else {
                    None
                }
            }

            // Fixed vs List: fixed must be in list
            (Self::Fixed(v), Self::List(list)) | (Self::List(list), Self::Fixed(v)) => {
                if list.contains(v) {
                    Some(Self::Fixed(v.clone()))
                } else {
                    None
                }
            }

            // Range vs Range: overlap
            (
                Self::Range {
                    min: min1,
                    max: max1,
                },
                Self::Range {
                    min: min2,
                    max: max2,
                },
            ) => {
                let new_min = min1.max(min2);
                let new_max = max1.min(max2);
                if new_min <= new_max {
                    if new_min == new_max {
                        Some(Self::Fixed(new_min.clone()))
                    } else {
                        Some(Self::Range {
                            min: new_min.clone(),
                            max: new_max.clone(),
                        })
                    }
                } else {
                    None
                }
            }

            // Range vs List: filter list to values in range
            (Self::Range { min, max }, Self::List(list))
            | (Self::List(list), Self::Range { min, max }) => {
                let filtered: Vec<T> = list
                    .iter()
                    .filter(|v| *v >= min && *v <= max)
                    .cloned()
                    .collect();
                match filtered.len() {
                    0 => None,
                    1 => Some(Self::Fixed(filtered.into_iter().next().unwrap())),
                    _ => Some(Self::List(filtered)),
                }
            }

            // List vs List: common values (preserving order from first list)
            (Self::List(list1), Self::List(list2)) => {
                let common: Vec<T> = list1
                    .iter()
                    .filter(|v| list2.contains(v))
                    .cloned()
                    .collect();
                match common.len() {
                    0 => None,
                    1 => Some(Self::Fixed(common.into_iter().next().unwrap())),
                    _ => Some(Self::List(common)),
                }
            }
        }
    }

    /// Fixate: choose a single value from the constraint.
    ///
    /// Returns the preferred value (first in list, min in range).
    /// Returns `None` for `Any` (cannot fixate without default).
    pub fn fixate(&self) -> Option<T> {
        match self {
            Self::Fixed(v) => Some(v.clone()),
            Self::Range { min, .. } => Some(min.clone()),
            Self::List(values) => values.first().cloned(),
            Self::Any => None,
        }
    }

    /// Fixate with a default value for `Any`.
    pub fn fixate_with_default(&self, default: T) -> T {
        self.fixate().unwrap_or(default)
    }

    /// Check if this is a fixed value.
    #[inline]
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Fixed(_))
    }

    /// Check if this accepts any value.
    #[inline]
    pub fn is_any(&self) -> bool {
        matches!(self, Self::Any)
    }

    /// Get the fixed value if this is fixed.
    #[inline]
    pub fn as_fixed(&self) -> Option<&T> {
        match self {
            Self::Fixed(v) => Some(v),
            _ => None,
        }
    }
}

impl<T: Clone + Ord> From<T> for CapsValue<T> {
    fn from(value: T) -> Self {
        Self::Fixed(value)
    }
}

impl<T: Clone + Ord> From<std::ops::RangeInclusive<T>> for CapsValue<T> {
    fn from(range: std::ops::RangeInclusive<T>) -> Self {
        let (min, max) = range.into_inner();
        Self::Range { min, max }
    }
}

impl<T: Clone + Ord> From<Vec<T>> for CapsValue<T> {
    fn from(values: Vec<T>) -> Self {
        match values.len() {
            0 => Self::Any,
            1 => Self::Fixed(values.into_iter().next().unwrap()),
            _ => Self::List(values),
        }
    }
}

// ============================================================================
// Media Formats
// ============================================================================

/// Media format - describes buffer contents.
///
/// This enum provides type-safe format descriptions for different media types.
/// Each variant carries the specific format details needed for that media type.
#[derive(Clone, Debug, PartialEq)]
pub enum MediaFormat {
    /// Raw video frames (uncompressed).
    VideoRaw(VideoFormat),
    /// Encoded video (compressed).
    Video(VideoCodec),
    /// Raw audio samples (uncompressed).
    AudioRaw(AudioFormat),
    /// Encoded audio (compressed).
    Audio(AudioCodec),
    /// RTP packet.
    Rtp(RtpFormat),
    /// MPEG-TS packet.
    MpegTs,
    /// Raw bytes (no format constraints).
    Bytes,
}

impl MediaFormat {
    /// Check compatibility (can data flow between these formats?).
    ///
    /// Two formats are compatible if:
    /// - Either is `Bytes` (accepts anything)
    /// - They are the same variant with matching parameters
    pub fn compatible(&self, other: &MediaFormat) -> bool {
        match (self, other) {
            (Self::Bytes, _) | (_, Self::Bytes) => true,
            (Self::VideoRaw(a), Self::VideoRaw(b)) => a == b,
            (Self::Video(a), Self::Video(b)) => a == b,
            (Self::AudioRaw(a), Self::AudioRaw(b)) => a == b,
            (Self::Audio(a), Self::Audio(b)) => a == b,
            (Self::Rtp(a), Self::Rtp(b)) => a.payload_type == b.payload_type,
            (Self::MpegTs, Self::MpegTs) => true,
            _ => false,
        }
    }

    /// Get the buffer size in bytes for this format, if determinable.
    ///
    /// Returns `Some(size)` for formats with known sizes (raw video/audio),
    /// or `None` for variable-size formats (encoded, RTP, etc.).
    pub fn buffer_size(&self) -> Option<usize> {
        match self {
            Self::VideoRaw(vf) => Some(vf.frame_size()),
            Self::AudioRaw(af) => Some(af.frame_size()),
            // Encoded formats have variable size
            Self::Video(_) | Self::Audio(_) => None,
            // RTP packets are variable size
            Self::Rtp(_) => None,
            // MPEG-TS packets are 188 bytes, but typically batched
            Self::MpegTs => Some(188 * 7), // Common batch size
            // Raw bytes have no size constraint
            Self::Bytes => None,
        }
    }
}

// ============================================================================
// Video Formats
// ============================================================================

/// Raw video format (24 bytes, Copy).
///
/// Describes uncompressed video frames with resolution, pixel format, and framerate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VideoFormat {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format (color space and layout).
    pub pixel_format: PixelFormat,
    /// Frame rate.
    pub framerate: Framerate,
}

impl Default for VideoFormat {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            pixel_format: PixelFormat::default(),
            framerate: Framerate::default(),
        }
    }
}

impl VideoFormat {
    /// Create a new video format.
    pub const fn new(
        width: u32,
        height: u32,
        pixel_format: PixelFormat,
        framerate: Framerate,
    ) -> Self {
        Self {
            width,
            height,
            pixel_format,
            framerate,
        }
    }

    /// Calculate the frame size in bytes for this format.
    pub const fn frame_size(&self) -> usize {
        let pixels = self.width as usize * self.height as usize;
        match self.pixel_format {
            // YUV 4:2:0 (1.5 bytes per pixel)
            PixelFormat::I420 => pixels * 3 / 2,
            PixelFormat::Nv12 => pixels * 3 / 2,
            // YUV 4:2:0 10-bit (2.25 bytes per pixel, rounded up)
            PixelFormat::I420_10Le => pixels * 3, // Each plane is 2 bytes/sample
            PixelFormat::P010 => pixels * 3,
            // YUV 4:2:2 (2 bytes per pixel)
            PixelFormat::I422 => pixels * 2,
            PixelFormat::Yuyv => pixels * 2,
            PixelFormat::Uyvy => pixels * 2,
            // YUV 4:4:4 (3 bytes per pixel)
            PixelFormat::I444 => pixels * 3,
            // RGB (3 or 4 bytes per pixel)
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => pixels * 3,
            PixelFormat::Rgba | PixelFormat::Bgra | PixelFormat::Argb => pixels * 4,
            // Grayscale
            PixelFormat::Gray8 => pixels,
            PixelFormat::Gray16Le => pixels * 2,
        }
    }
}

/// Pixel formats (color space and memory layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[repr(u8)]
pub enum PixelFormat {
    // ========================================================================
    // YUV 4:2:0 formats (most common)
    // ========================================================================
    /// YUV 4:2:0 planar (Y plane, then U plane, then V plane).
    /// Most common format for video codecs.
    #[default]
    I420 = 0,
    /// YUV 4:2:0 semi-planar (Y plane, then interleaved UV plane).
    /// Common for hardware decoders.
    Nv12,
    /// YUV 4:2:0 planar, 10-bit little endian.
    /// Used by 10-bit HEVC, AV1.
    I420_10Le,
    /// YUV 4:2:0 semi-planar, 10-bit.
    /// Common for 10-bit hardware decoders.
    P010,

    // ========================================================================
    // YUV 4:2:2 formats (broadcast quality)
    // ========================================================================
    /// YUV 4:2:2 planar (Y plane, then U plane, then V plane).
    I422,
    /// YUV 4:2:2 packed (Y0 U Y1 V).
    Yuyv,
    /// YUV 4:2:2 packed (U Y0 V Y1).
    Uyvy,

    // ========================================================================
    // YUV 4:4:4 formats (full chroma)
    // ========================================================================
    /// YUV 4:4:4 planar.
    I444,

    // ========================================================================
    // RGB formats
    // ========================================================================
    /// RGB 8-bit per channel, packed (24 bits/pixel).
    Rgb24,
    /// RGBA 8-bit per channel, packed (32 bits/pixel).
    Rgba,
    /// BGR 8-bit per channel, packed (24 bits/pixel).
    Bgr24,
    /// BGRA 8-bit per channel, packed (32 bits/pixel).
    Bgra,
    /// ARGB 8-bit per channel, packed (32 bits/pixel).
    Argb,

    // ========================================================================
    // Grayscale formats
    // ========================================================================
    /// 8-bit grayscale.
    Gray8,
    /// 16-bit grayscale little endian.
    Gray16Le,
}

/// Video codecs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VideoCodec {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC.
    H265,
    /// VP8.
    Vp8,
    /// VP9.
    Vp9,
    /// AV1.
    Av1,
}

/// Frame rate as numerator/denominator (8 bytes, Copy).
///
/// Using a fraction allows exact representation of common framerates
/// like 29.97 fps (30000/1001) and 23.976 fps (24000/1001).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Framerate {
    /// Numerator (frames).
    pub num: u32,
    /// Denominator (time units).
    pub den: u32,
}

impl Framerate {
    /// Create a new framerate.
    pub const fn new(num: u32, den: u32) -> Self {
        Self { num, den }
    }

    /// 24 fps (film).
    pub const FPS_24: Self = Self::new(24, 1);
    /// 25 fps (PAL).
    pub const FPS_25: Self = Self::new(25, 1);
    /// 30 fps.
    pub const FPS_30: Self = Self::new(30, 1);
    /// 60 fps.
    pub const FPS_60: Self = Self::new(60, 1);
    /// 29.97 fps (NTSC).
    pub const FPS_29_97: Self = Self::new(30000, 1001);
    /// 23.976 fps (film on NTSC).
    pub const FPS_23_976: Self = Self::new(24000, 1001);

    /// Get the framerate as a floating-point value.
    #[inline]
    pub fn fps(&self) -> f64 {
        self.num as f64 / self.den.max(1) as f64
    }

    /// Get frame duration in nanoseconds.
    #[inline]
    pub const fn frame_duration_ns(&self) -> u64 {
        if self.num == 0 {
            return 0;
        }
        (self.den as u64 * 1_000_000_000) / self.num as u64
    }
}

impl Default for Framerate {
    fn default() -> Self {
        Self::FPS_30
    }
}

impl PartialOrd for Framerate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Framerate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Compare as fractions: a/b vs c/d => a*d vs c*b
        let lhs = (self.num as u64) * (other.den as u64);
        let rhs = (other.num as u64) * (self.den as u64);
        lhs.cmp(&rhs)
    }
}

// ============================================================================
// Audio Formats
// ============================================================================

/// Raw audio format (8 bytes, Copy).
///
/// Describes uncompressed audio samples.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AudioFormat {
    /// Sample rate in Hz (e.g., 44100, 48000).
    pub sample_rate: u32,
    /// Number of channels (1 = mono, 2 = stereo).
    pub channels: u16,
    /// Sample format (bit depth and type).
    pub sample_format: SampleFormat,
}

impl Default for AudioFormat {
    fn default() -> Self {
        Self {
            sample_rate: 0,
            channels: 0,
            sample_format: SampleFormat::default(),
        }
    }
}

impl AudioFormat {
    /// Create a new audio format.
    pub const fn new(sample_rate: u32, channels: u16, sample_format: SampleFormat) -> Self {
        Self {
            sample_rate,
            channels,
            sample_format,
        }
    }

    /// CD quality: 44100 Hz, stereo, 16-bit signed.
    pub const CD_QUALITY: Self = Self::new(44100, 2, SampleFormat::S16);

    /// DVD quality: 48000 Hz, stereo, 16-bit signed.
    pub const DVD_QUALITY: Self = Self::new(48000, 2, SampleFormat::S16);

    /// Get bytes per sample (for one channel).
    pub const fn bytes_per_sample(&self) -> usize {
        self.sample_format.bytes()
    }

    /// Get bytes per frame (all channels for one sample time).
    pub const fn bytes_per_frame(&self) -> usize {
        self.sample_format.bytes() * self.channels as usize
    }

    /// Get a typical buffer size for this audio format.
    ///
    /// Returns size for ~10ms of audio, which is a common buffer duration
    /// used in real-time audio processing.
    pub const fn frame_size(&self) -> usize {
        // 10ms worth of samples
        let samples_per_buffer = self.sample_rate as usize / 100;
        samples_per_buffer * self.bytes_per_frame()
    }
}

/// Audio sample formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, PartialOrd, Ord)]
#[repr(u8)]
pub enum SampleFormat {
    /// Signed 16-bit integer (most common).
    #[default]
    S16 = 0,
    /// Signed 32-bit integer.
    S32,
    /// 32-bit floating point.
    F32,
    /// Unsigned 8-bit integer.
    U8,
}

impl SampleFormat {
    /// Get bytes per sample.
    pub const fn bytes(&self) -> usize {
        match self {
            Self::S16 => 2,
            Self::S32 | Self::F32 => 4,
            Self::U8 => 1,
        }
    }
}

/// Audio codecs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AudioCodec {
    /// Opus (modern, efficient).
    Opus,
    /// AAC.
    Aac,
    /// MP3.
    Mp3,
    /// G.711 μ-law.
    Pcmu,
    /// G.711 A-law.
    Pcma,
}

// ============================================================================
// RTP Format
// ============================================================================

/// RTP stream format (8 bytes, Copy).
///
/// Describes the format of an RTP stream including payload type and encoding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RtpFormat {
    /// RTP payload type (0-127).
    pub payload_type: u8,
    /// Clock rate in Hz.
    pub clock_rate: u32,
    /// What's inside the RTP payload.
    pub encoding: RtpEncoding,
}

impl RtpFormat {
    /// Create a new RTP format.
    pub const fn new(payload_type: u8, clock_rate: u32, encoding: RtpEncoding) -> Self {
        Self {
            payload_type,
            clock_rate,
            encoding,
        }
    }

    /// H.264 video over RTP (dynamic payload type 96).
    pub const H264: Self = Self::new(96, 90000, RtpEncoding::H264);

    /// H.265 video over RTP (dynamic payload type 97).
    pub const H265: Self = Self::new(97, 90000, RtpEncoding::H265);

    /// Opus audio over RTP (dynamic payload type 111).
    pub const OPUS: Self = Self::new(111, 48000, RtpEncoding::Opus);

    /// PCMU audio (G.711 μ-law, payload type 0).
    pub const PCMU: Self = Self::new(0, 8000, RtpEncoding::Pcmu);

    /// PCMA audio (G.711 A-law, payload type 8).
    pub const PCMA: Self = Self::new(8, 8000, RtpEncoding::Pcma);
}

/// What's inside the RTP payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RtpEncoding {
    /// H.264 / AVC video.
    H264,
    /// H.265 / HEVC video.
    H265,
    /// VP8 video.
    Vp8,
    /// VP9 video.
    Vp9,
    /// Opus audio.
    Opus,
    /// G.711 μ-law audio.
    Pcmu,
    /// G.711 A-law audio.
    Pcma,
    /// Dynamic/unknown payload type.
    Dynamic(u8),
}

// ============================================================================
// Codec Data
// ============================================================================

/// Codec initialization data (SPS/PPS for H.264, etc.).
///
/// Only allocated when codec requires out-of-band configuration.
#[derive(Clone, Debug, PartialEq)]
pub struct CodecData(Box<[u8]>);

impl CodecData {
    /// Create new codec data from bytes.
    pub fn new(data: impl Into<Box<[u8]>>) -> Self {
        Self(data.into())
    }

    /// Get the data as a slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    /// Get the length of the codec data.
    #[inline]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl AsRef<[u8]> for CodecData {
    fn as_ref(&self) -> &[u8] {
        &self.0
    }
}

impl From<Vec<u8>> for CodecData {
    fn from(v: Vec<u8>) -> Self {
        Self(v.into_boxed_slice())
    }
}

impl From<&[u8]> for CodecData {
    fn from(s: &[u8]) -> Self {
        Self(s.into())
    }
}

// ============================================================================
// Format Caps - constraint-based format negotiation
// ============================================================================

/// Video format with constraints for negotiation.
///
/// Each field can be fixed, a range, a list of options, or any value.
/// Used during caps negotiation to find compatible formats.
#[derive(Clone, Debug, PartialEq)]
pub struct VideoFormatCaps {
    /// Width constraint.
    pub width: CapsValue<u32>,
    /// Height constraint.
    pub height: CapsValue<u32>,
    /// Pixel format constraint.
    pub pixel_format: CapsValue<PixelFormat>,
    /// Framerate constraint.
    pub framerate: CapsValue<Framerate>,
}

impl VideoFormatCaps {
    /// Create caps that accept any video format.
    pub fn any() -> Self {
        Self {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Any,
            framerate: CapsValue::Any,
        }
    }

    /// Create caps for a fixed video format.
    pub fn fixed(format: VideoFormat) -> Self {
        Self {
            width: CapsValue::Fixed(format.width),
            height: CapsValue::Fixed(format.height),
            pixel_format: CapsValue::Fixed(format.pixel_format),
            framerate: CapsValue::Fixed(format.framerate),
        }
    }

    // ========================================================================
    // Convenience constructors for common formats
    // ========================================================================

    /// Create caps for YUV 4:2:0 video (I420) of any size.
    ///
    /// This is the most common format for video codecs.
    pub fn yuv420() -> Self {
        Self {
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            ..Self::any()
        }
    }

    /// Create caps for YUV 4:2:0 video with specific dimensions.
    pub fn yuv420_size(width: u32, height: u32) -> Self {
        Self {
            width: CapsValue::Fixed(width),
            height: CapsValue::Fixed(height),
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            framerate: CapsValue::Any,
        }
    }

    /// Create caps accepting multiple YUV formats.
    pub fn yuv() -> Self {
        Self {
            pixel_format: CapsValue::List(vec![
                PixelFormat::I420,
                PixelFormat::Nv12,
                PixelFormat::I422,
                PixelFormat::I444,
            ]),
            ..Self::any()
        }
    }

    /// Create caps for RGB formats.
    pub fn rgb() -> Self {
        Self {
            pixel_format: CapsValue::List(vec![PixelFormat::Rgb24, PixelFormat::Rgba]),
            ..Self::any()
        }
    }

    /// Create caps for RGBA video.
    pub fn rgba() -> Self {
        Self {
            pixel_format: CapsValue::Fixed(PixelFormat::Rgba),
            ..Self::any()
        }
    }

    /// Create caps with a specific size constraint.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = CapsValue::Fixed(width);
        self.height = CapsValue::Fixed(height);
        self
    }

    /// Create caps with a size range.
    pub fn with_size_range(mut self, min_w: u32, max_w: u32, min_h: u32, max_h: u32) -> Self {
        self.width = CapsValue::Range {
            min: min_w,
            max: max_w,
        };
        self.height = CapsValue::Range {
            min: min_h,
            max: max_h,
        };
        self
    }

    /// Create caps with a specific framerate.
    pub fn with_framerate(mut self, framerate: Framerate) -> Self {
        self.framerate = CapsValue::Fixed(framerate);
        self
    }

    /// Intersect with another video caps.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        Some(Self {
            width: self.width.intersect(&other.width)?,
            height: self.height.intersect(&other.height)?,
            pixel_format: self.pixel_format.intersect(&other.pixel_format)?,
            framerate: self.framerate.intersect(&other.framerate)?,
        })
    }

    /// Fixate to a concrete video format.
    pub fn fixate(&self) -> Option<VideoFormat> {
        Some(VideoFormat {
            width: self.width.fixate()?,
            height: self.height.fixate()?,
            pixel_format: self.pixel_format.fixate()?,
            framerate: self.framerate.fixate()?,
        })
    }

    /// Fixate with defaults for unconstrained values.
    ///
    /// Unlike `fixate()`, this always returns a value by using sensible defaults
    /// for Any constraints: 1920x1080, I420, 30fps.
    pub fn fixate_with_defaults(&self) -> VideoFormat {
        VideoFormat {
            width: self.width.fixate_with_default(1920),
            height: self.height.fixate_with_default(1080),
            pixel_format: self.pixel_format.fixate_with_default(PixelFormat::I420),
            framerate: self.framerate.fixate_with_default(Framerate::FPS_30),
        }
    }

    /// Check if fully fixed.
    pub fn is_fixed(&self) -> bool {
        self.width.is_fixed()
            && self.height.is_fixed()
            && self.pixel_format.is_fixed()
            && self.framerate.is_fixed()
    }
}

impl Default for VideoFormatCaps {
    fn default() -> Self {
        Self::any()
    }
}

impl From<VideoFormat> for VideoFormatCaps {
    fn from(format: VideoFormat) -> Self {
        Self::fixed(format)
    }
}

/// Audio format with constraints for negotiation.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioFormatCaps {
    /// Sample rate constraint.
    pub sample_rate: CapsValue<u32>,
    /// Number of channels constraint.
    pub channels: CapsValue<u16>,
    /// Sample format constraint.
    pub sample_format: CapsValue<SampleFormat>,
}

impl AudioFormatCaps {
    /// Create caps that accept any audio format.
    pub fn any() -> Self {
        Self {
            sample_rate: CapsValue::Any,
            channels: CapsValue::Any,
            sample_format: CapsValue::Any,
        }
    }

    /// Create caps for a fixed audio format.
    pub fn fixed(format: AudioFormat) -> Self {
        Self {
            sample_rate: CapsValue::Fixed(format.sample_rate),
            channels: CapsValue::Fixed(format.channels),
            sample_format: CapsValue::Fixed(format.sample_format),
        }
    }

    // ========================================================================
    // Convenience constructors for common formats
    // ========================================================================

    /// Create caps for S16 audio (most common format).
    pub fn s16() -> Self {
        Self {
            sample_format: CapsValue::Fixed(SampleFormat::S16),
            ..Self::any()
        }
    }

    /// Create caps for F32 audio (common for processing).
    pub fn f32() -> Self {
        Self {
            sample_format: CapsValue::Fixed(SampleFormat::F32),
            ..Self::any()
        }
    }

    /// Create caps for stereo audio.
    pub fn stereo() -> Self {
        Self {
            channels: CapsValue::Fixed(2),
            ..Self::any()
        }
    }

    /// Create caps for mono audio.
    pub fn mono() -> Self {
        Self {
            channels: CapsValue::Fixed(1),
            ..Self::any()
        }
    }

    /// Create caps for standard rates (44100, 48000).
    pub fn standard_rates() -> Self {
        Self {
            sample_rate: CapsValue::List(vec![48000, 44100]),
            ..Self::any()
        }
    }

    /// Create caps with a specific sample rate.
    pub fn with_rate(mut self, rate: u32) -> Self {
        self.sample_rate = CapsValue::Fixed(rate);
        self
    }

    /// Create caps with a specific channel count.
    pub fn with_channels(mut self, channels: u16) -> Self {
        self.channels = CapsValue::Fixed(channels);
        self
    }

    /// Create caps with a specific sample format.
    pub fn with_format(mut self, format: SampleFormat) -> Self {
        self.sample_format = CapsValue::Fixed(format);
        self
    }

    /// Intersect with another audio caps.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        Some(Self {
            sample_rate: self.sample_rate.intersect(&other.sample_rate)?,
            channels: self.channels.intersect(&other.channels)?,
            sample_format: self.sample_format.intersect(&other.sample_format)?,
        })
    }

    /// Fixate to a concrete audio format.
    pub fn fixate(&self) -> Option<AudioFormat> {
        Some(AudioFormat {
            sample_rate: self.sample_rate.fixate()?,
            channels: self.channels.fixate()?,
            sample_format: self.sample_format.fixate()?,
        })
    }

    /// Fixate with defaults for unconstrained values.
    ///
    /// Unlike `fixate()`, this always returns a value by using sensible defaults
    /// for Any constraints: 48000 Hz, stereo, S16.
    pub fn fixate_with_defaults(&self) -> AudioFormat {
        AudioFormat {
            sample_rate: self.sample_rate.fixate_with_default(48000),
            channels: self.channels.fixate_with_default(2),
            sample_format: self.sample_format.fixate_with_default(SampleFormat::S16),
        }
    }

    /// Check if fully fixed.
    pub fn is_fixed(&self) -> bool {
        self.sample_rate.is_fixed() && self.channels.is_fixed() && self.sample_format.is_fixed()
    }
}

impl Default for AudioFormatCaps {
    fn default() -> Self {
        Self::any()
    }
}

impl From<AudioFormat> for AudioFormatCaps {
    fn from(format: AudioFormat) -> Self {
        Self::fixed(format)
    }
}

/// Format caps - constraints for any format type.
#[derive(Clone, Debug, PartialEq, Default)]
pub enum FormatCaps {
    /// Raw video with constraints.
    VideoRaw(VideoFormatCaps),
    /// Encoded video (codec is fixed, but format params may vary).
    Video(VideoCodec),
    /// Raw audio with constraints.
    AudioRaw(AudioFormatCaps),
    /// Encoded audio.
    Audio(AudioCodec),
    /// RTP stream.
    Rtp(RtpFormat),
    /// MPEG-TS.
    MpegTs,
    /// Raw bytes.
    Bytes,
    /// Any format.
    #[default]
    Any,
}

impl FormatCaps {
    /// Intersect with another format caps.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (Self::Any, other) => Some(other.clone()),
            (self_, Self::Any) => Some(self_.clone()),

            (Self::VideoRaw(a), Self::VideoRaw(b)) => Some(Self::VideoRaw(a.intersect(b)?)),
            (Self::Video(a), Self::Video(b)) if a == b => Some(Self::Video(*a)),
            (Self::AudioRaw(a), Self::AudioRaw(b)) => Some(Self::AudioRaw(a.intersect(b)?)),
            (Self::Audio(a), Self::Audio(b)) if a == b => Some(Self::Audio(*a)),
            (Self::Rtp(a), Self::Rtp(b)) if a == b => Some(Self::Rtp(*a)),
            (Self::MpegTs, Self::MpegTs) => Some(Self::MpegTs),
            (Self::Bytes, Self::Bytes) => Some(Self::Bytes),
            (Self::Bytes, other) | (other, Self::Bytes) => Some(other.clone()),

            _ => None,
        }
    }

    /// Fixate to a concrete media format.
    pub fn fixate(&self) -> Option<MediaFormat> {
        match self {
            Self::VideoRaw(caps) => Some(MediaFormat::VideoRaw(caps.fixate()?)),
            Self::Video(codec) => Some(MediaFormat::Video(*codec)),
            Self::AudioRaw(caps) => Some(MediaFormat::AudioRaw(caps.fixate()?)),
            Self::Audio(codec) => Some(MediaFormat::Audio(*codec)),
            Self::Rtp(rtp) => Some(MediaFormat::Rtp(*rtp)),
            Self::MpegTs => Some(MediaFormat::MpegTs),
            Self::Bytes => Some(MediaFormat::Bytes),
            Self::Any => None,
        }
    }

    /// Fixate with defaults for unconstrained values.
    ///
    /// Unlike `fixate()`, this always returns a value by using sensible defaults.
    /// For `Any`, defaults to raw bytes.
    pub fn fixate_with_defaults(&self) -> MediaFormat {
        match self {
            Self::VideoRaw(caps) => MediaFormat::VideoRaw(caps.fixate_with_defaults()),
            Self::Video(codec) => MediaFormat::Video(*codec),
            Self::AudioRaw(caps) => MediaFormat::AudioRaw(caps.fixate_with_defaults()),
            Self::Audio(codec) => MediaFormat::Audio(*codec),
            Self::Rtp(rtp) => MediaFormat::Rtp(*rtp),
            Self::MpegTs => MediaFormat::MpegTs,
            Self::Bytes => MediaFormat::Bytes,
            Self::Any => MediaFormat::Bytes,
        }
    }

    /// Check if this is a video format (raw or encoded).
    pub fn is_video(&self) -> bool {
        matches!(self, Self::VideoRaw(_) | Self::Video(_))
    }

    /// Check if this is an audio format (raw or encoded).
    pub fn is_audio(&self) -> bool {
        matches!(self, Self::AudioRaw(_) | Self::Audio(_))
    }
}

impl From<MediaFormat> for FormatCaps {
    fn from(format: MediaFormat) -> Self {
        match format {
            MediaFormat::VideoRaw(v) => Self::VideoRaw(v.into()),
            MediaFormat::Video(c) => Self::Video(c),
            MediaFormat::AudioRaw(a) => Self::AudioRaw(a.into()),
            MediaFormat::Audio(c) => Self::Audio(c),
            MediaFormat::Rtp(r) => Self::Rtp(r),
            MediaFormat::MpegTs => Self::MpegTs,
            MediaFormat::Bytes => Self::Bytes,
        }
    }
}

impl From<VideoFormatCaps> for FormatCaps {
    fn from(caps: VideoFormatCaps) -> Self {
        Self::VideoRaw(caps)
    }
}

impl From<AudioFormatCaps> for FormatCaps {
    fn from(caps: AudioFormatCaps) -> Self {
        Self::AudioRaw(caps)
    }
}

// ============================================================================
// Memory Caps - memory type constraints
// ============================================================================

/// Memory capabilities for negotiation.
///
/// Describes what memory types an element can work with.
#[derive(Clone, Debug, PartialEq)]
pub struct MemoryCaps {
    /// Supported memory types (ordered by preference).
    pub types: CapsValue<MemoryType>,
    /// Can import (receive) from these types via conversion.
    pub can_import: Vec<MemoryType>,
    /// Can export (send) to these types via conversion.
    pub can_export: Vec<MemoryType>,
}

impl MemoryCaps {
    /// Accept only CPU memory.
    pub fn cpu_only() -> Self {
        Self {
            types: CapsValue::Fixed(MemoryType::Cpu),
            can_import: vec![MemoryType::Cpu],
            can_export: vec![MemoryType::Cpu],
        }
    }

    /// Accept any memory type.
    pub fn any() -> Self {
        Self {
            types: CapsValue::Any,
            can_import: vec![],
            can_export: vec![],
        }
    }

    /// Prefer GPU memory but can work with CPU.
    #[allow(deprecated)]
    pub fn gpu_preferred() -> Self {
        Self {
            types: CapsValue::List(vec![MemoryType::GpuDevice, MemoryType::Cpu]),
            can_import: vec![MemoryType::Cpu, MemoryType::DmaBuf],
            can_export: vec![MemoryType::Cpu, MemoryType::DmaBuf],
        }
    }

    /// Intersect with another memory caps.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        Some(Self {
            types: self.types.intersect(&other.types)?,
            // For imports/exports, take union (more capable)
            can_import: self
                .can_import
                .iter()
                .chain(other.can_import.iter())
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect(),
            can_export: self
                .can_export
                .iter()
                .chain(other.can_export.iter())
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect(),
        })
    }

    /// Fixate to a concrete memory type.
    pub fn fixate(&self) -> Option<MemoryType> {
        self.types.fixate()
    }
}

impl Default for MemoryCaps {
    fn default() -> Self {
        Self::any()
    }
}

// ============================================================================
// Media Caps - combined format + memory
// ============================================================================

/// Combined format and memory capabilities.
///
/// This is the complete specification for what an element can handle.
#[derive(Clone, Debug, PartialEq)]
pub struct MediaCaps {
    /// Format constraints.
    pub format: FormatCaps,
    /// Memory constraints.
    pub memory: MemoryCaps,
}

impl MediaCaps {
    /// Create caps that accept anything.
    pub fn any() -> Self {
        Self {
            format: FormatCaps::Any,
            memory: MemoryCaps::any(),
        }
    }

    /// Create caps with specific format, any memory.
    pub fn from_format(format: FormatCaps) -> Self {
        Self {
            format,
            memory: MemoryCaps::any(),
        }
    }

    /// Create caps with specific format and memory.
    pub fn new(format: FormatCaps, memory: MemoryCaps) -> Self {
        Self { format, memory }
    }

    /// Intersect with another media caps.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        Some(Self {
            format: self.format.intersect(&other.format)?,
            memory: self.memory.intersect(&other.memory)?,
        })
    }

    /// Fixate format (memory type chosen separately).
    pub fn fixate_format(&self) -> Option<MediaFormat> {
        self.format.fixate()
    }

    /// Fixate format with defaults for unconstrained values.
    pub fn fixate_format_with_defaults(&self) -> MediaFormat {
        self.format.fixate_with_defaults()
    }

    /// Fixate memory type.
    pub fn fixate_memory(&self) -> Option<MemoryType> {
        self.memory.fixate()
    }
}

impl Default for MediaCaps {
    fn default() -> Self {
        Self::any()
    }
}

impl From<MediaFormat> for MediaCaps {
    fn from(format: MediaFormat) -> Self {
        Self {
            format: format.into(),
            memory: MemoryCaps::any(),
        }
    }
}

impl From<Caps> for MediaCaps {
    fn from(caps: Caps) -> Self {
        if caps.is_any() {
            Self::any()
        } else if let Some(format) = caps.preferred() {
            Self::from_format(format.clone().into())
        } else {
            Self::any()
        }
    }
}

// ============================================================================
// ElementMediaCaps - format + memory combinations
// ============================================================================

/// A single format+memory capability.
///
/// This couples a format constraint with its supported memory types.
/// This is the key insight from GStreamer's GstCapsFeatures: memory type
/// is part of the format constraint, not separate.
///
/// # Example
///
/// ```rust
/// use parallax::format::{FormatMemoryCap, VideoFormatCaps, MemoryCaps, PixelFormat, CapsValue};
/// use parallax::memory::MemoryType;
///
/// // RGB format supported on GPU only
/// let gpu_rgb = FormatMemoryCap {
///     format: VideoFormatCaps {
///         pixel_format: CapsValue::Fixed(PixelFormat::Rgba),
///         ..VideoFormatCaps::any()
///     }.into(),
///     memory: MemoryCaps {
///         types: CapsValue::Fixed(MemoryType::GpuDevice),
///         ..MemoryCaps::any()
///     },
/// };
///
/// // YUV formats supported on CPU
/// let cpu_yuv = FormatMemoryCap {
///     format: VideoFormatCaps {
///         pixel_format: CapsValue::List(vec![PixelFormat::I420, PixelFormat::Nv12]),
///         ..VideoFormatCaps::any()
///     }.into(),
///     memory: MemoryCaps::cpu_only(),
/// };
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct FormatMemoryCap {
    /// Format constraints.
    pub format: FormatCaps,
    /// Memory type constraints.
    pub memory: MemoryCaps,
}

impl FormatMemoryCap {
    /// Create a new format+memory capability.
    pub fn new(format: FormatCaps, memory: MemoryCaps) -> Self {
        Self { format, memory }
    }

    /// Create a capability for any format with CPU memory.
    pub fn any_cpu() -> Self {
        Self {
            format: FormatCaps::Any,
            memory: MemoryCaps::cpu_only(),
        }
    }

    /// Create a capability for any format and memory.
    pub fn any() -> Self {
        Self {
            format: FormatCaps::Any,
            memory: MemoryCaps::any(),
        }
    }

    /// Try to intersect this capability with another.
    ///
    /// Returns `Some(intersected)` if there's a common format+memory,
    /// or `None` if they're incompatible.
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        Some(Self {
            format: self.format.intersect(&other.format)?,
            memory: self.memory.intersect(&other.memory)?,
        })
    }

    /// Convert to MediaCaps for backward compatibility.
    pub fn into_media_caps(self) -> MediaCaps {
        MediaCaps {
            format: self.format,
            memory: self.memory,
        }
    }
}

impl From<MediaCaps> for FormatMemoryCap {
    fn from(caps: MediaCaps) -> Self {
        Self {
            format: caps.format,
            memory: caps.memory,
        }
    }
}

impl From<FormatMemoryCap> for MediaCaps {
    fn from(cap: FormatMemoryCap) -> Self {
        MediaCaps {
            format: cap.format,
            memory: cap.memory,
        }
    }
}

/// Element capabilities: multiple format+memory combinations.
///
/// This allows elements to express complex constraints like:
/// - "I support RGB on GPU, but only YUV on CPU"
/// - "I accept YUYV, MJPEG, or NV12"
///
/// Each capability in the list represents an independent option.
/// The list is ordered by preference (first is best).
///
/// # Design Rationale
///
/// This is similar to GStreamer's GstCaps which can contain multiple
/// GstStructure entries, each with its own GstCapsFeatures for memory type.
/// Unlike GStreamer, we couple format and memory tightly in each entry.
///
/// # Examples
///
/// ```rust
/// use parallax::format::{ElementMediaCaps, FormatMemoryCap, VideoFormatCaps, MemoryCaps, PixelFormat, CapsValue};
/// use parallax::memory::MemoryType;
///
/// // Camera that supports multiple formats
/// let camera_caps = ElementMediaCaps::new(vec![
///     // YUYV at high resolution
///     FormatMemoryCap::new(
///         VideoFormatCaps {
///             width: CapsValue::Fixed(1920),
///             height: CapsValue::Fixed(1080),
///             pixel_format: CapsValue::Fixed(PixelFormat::Yuyv),
///             ..VideoFormatCaps::any()
///         }.into(),
///         MemoryCaps::cpu_only(),
///     ),
///     // NV12 at any resolution (from hardware encoder path)
///     FormatMemoryCap::new(
///         VideoFormatCaps {
///             pixel_format: CapsValue::Fixed(PixelFormat::Nv12),
///             ..VideoFormatCaps::any()
///         }.into(),
///         MemoryCaps::cpu_only(),
///     ),
/// ]);
///
/// // GPU filter that only supports RGB on GPU
/// let gpu_filter_caps = ElementMediaCaps::new(vec![
///     FormatMemoryCap::new(
///         VideoFormatCaps::rgba().into(),
///         MemoryCaps {
///             types: CapsValue::Fixed(MemoryType::GpuDevice),
///             ..MemoryCaps::any()
///         },
///     ),
/// ]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ElementMediaCaps {
    /// List of supported format+memory combinations, ordered by preference.
    caps: SmallVec<[FormatMemoryCap; 2]>,
}

impl ElementMediaCaps {
    /// Create new element caps with the given capabilities.
    pub fn new(caps: impl IntoIterator<Item = FormatMemoryCap>) -> Self {
        Self {
            caps: caps.into_iter().collect(),
        }
    }

    /// Create caps that accept any format and memory.
    pub fn any() -> Self {
        Self {
            caps: smallvec::smallvec![FormatMemoryCap::any()],
        }
    }

    /// Create caps that accept any format but only CPU memory.
    pub fn any_cpu() -> Self {
        Self {
            caps: smallvec::smallvec![FormatMemoryCap::any_cpu()],
        }
    }

    /// Create caps for a single format+memory combination.
    pub fn single(cap: FormatMemoryCap) -> Self {
        Self {
            caps: smallvec::smallvec![cap],
        }
    }

    /// Create caps from a simple MediaCaps (single format+memory).
    pub fn from_media_caps(caps: MediaCaps) -> Self {
        Self::single(caps.into())
    }

    /// Check if this accepts any format (no constraints).
    pub fn is_any(&self) -> bool {
        self.caps.len() == 1
            && matches!(self.caps[0].format, FormatCaps::Any)
            && self.caps[0].memory.types.is_any()
    }

    /// Check if this has exactly one fixed capability.
    pub fn is_fixed(&self) -> bool {
        self.caps.len() == 1
    }

    /// Get the capabilities list.
    pub fn capabilities(&self) -> &[FormatMemoryCap] {
        &self.caps
    }

    /// Iterate over all capabilities.
    pub fn iter(&self) -> impl Iterator<Item = &FormatMemoryCap> {
        self.caps.iter()
    }

    /// Get the number of capabilities.
    pub fn len(&self) -> usize {
        self.caps.len()
    }

    /// Check if empty (no capabilities).
    pub fn is_empty(&self) -> bool {
        self.caps.is_empty()
    }

    /// Get the preferred (first) capability.
    pub fn preferred(&self) -> Option<&FormatMemoryCap> {
        self.caps.first()
    }

    /// Add a capability.
    pub fn add(&mut self, cap: FormatMemoryCap) {
        self.caps.push(cap);
    }

    /// Find the first compatible capability between two caps.
    ///
    /// Returns the intersected capability if one exists, preferring
    /// earlier entries in both lists.
    pub fn intersect(&self, other: &Self) -> Option<FormatMemoryCap> {
        // Try each of our caps against each of theirs
        for our_cap in &self.caps {
            for their_cap in &other.caps {
                if let Some(intersected) = our_cap.intersect(their_cap) {
                    return Some(intersected);
                }
            }
        }
        None
    }

    /// Check if there's any compatible format+memory between two caps.
    pub fn intersects(&self, other: &Self) -> bool {
        self.intersect(other).is_some()
    }

    /// Find all compatible capabilities (not just the first).
    pub fn intersect_all(&self, other: &Self) -> Vec<FormatMemoryCap> {
        let mut result = Vec::new();
        for our_cap in &self.caps {
            for their_cap in &other.caps {
                if let Some(intersected) = our_cap.intersect(their_cap) {
                    result.push(intersected);
                }
            }
        }
        result
    }

    /// Convert to the best matching MediaCaps for backward compatibility.
    ///
    /// Returns the first (preferred) capability as MediaCaps.
    pub fn to_media_caps(&self) -> MediaCaps {
        self.caps
            .first()
            .cloned()
            .map(|c| c.into())
            .unwrap_or_else(MediaCaps::any)
    }

    /// Convert from simple Caps (format list without memory).
    ///
    /// Each format in the Caps becomes a separate capability with CPU memory.
    pub fn from_caps(caps: &Caps) -> Self {
        if caps.is_any() {
            return Self::any_cpu();
        }

        let caps_vec: SmallVec<[FormatMemoryCap; 2]> = caps
            .formats()
            .iter()
            .map(|f| FormatMemoryCap::new(f.clone().into(), MemoryCaps::cpu_only()))
            .collect();

        if caps_vec.is_empty() {
            Self::any_cpu()
        } else {
            Self { caps: caps_vec }
        }
    }
}

impl Default for ElementMediaCaps {
    fn default() -> Self {
        Self::any()
    }
}

impl From<MediaCaps> for ElementMediaCaps {
    fn from(caps: MediaCaps) -> Self {
        Self::single(caps.into())
    }
}

impl From<Caps> for ElementMediaCaps {
    fn from(caps: Caps) -> Self {
        Self::from_caps(&caps)
    }
}

impl From<FormatMemoryCap> for ElementMediaCaps {
    fn from(cap: FormatMemoryCap) -> Self {
        Self::single(cap)
    }
}

// ============================================================================
// Caps (Capabilities) - legacy API
// ============================================================================

/// Capabilities: what formats an element accepts/produces.
///
/// Caps describe the formats an element can handle. They're used for:
/// - Pipeline validation (ensuring connected elements are compatible)
/// - Format negotiation (choosing the best format for a connection)
///
/// # Examples
///
/// ```rust
/// use parallax::format::{Caps, MediaFormat, VideoCodec};
///
/// // Element that accepts any format
/// let any_caps = Caps::any();
///
/// // Element that only produces H.264
/// let h264_caps = Caps::new(MediaFormat::Video(VideoCodec::H264));
///
/// // Element that accepts multiple formats
/// let multi_caps = Caps::many([
///     MediaFormat::Video(VideoCodec::H264),
///     MediaFormat::Video(VideoCodec::H265),
/// ]);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct Caps(SmallVec<[MediaFormat; 2]>);

impl Caps {
    /// Create caps that accept any format.
    ///
    /// This is the default for elements that don't care about format.
    pub fn any() -> Self {
        Self(SmallVec::new())
    }

    /// Create caps with a single format.
    pub fn new(format: MediaFormat) -> Self {
        let mut v = SmallVec::new();
        v.push(format);
        Self(v)
    }

    /// Create caps with multiple acceptable formats.
    ///
    /// The first format is the preferred one.
    pub fn many(formats: impl IntoIterator<Item = MediaFormat>) -> Self {
        Self(formats.into_iter().collect())
    }

    /// Is this "any format"?
    #[inline]
    pub fn is_any(&self) -> bool {
        self.0.is_empty()
    }

    /// Is this a single fixed format?
    #[inline]
    pub fn is_fixed(&self) -> bool {
        self.0.len() == 1
    }

    /// Get the formats.
    #[inline]
    pub fn formats(&self) -> &[MediaFormat] {
        &self.0
    }

    /// Get the preferred format (first one).
    #[inline]
    pub fn preferred(&self) -> Option<&MediaFormat> {
        self.0.first()
    }

    /// Check if compatible with another caps.
    ///
    /// Two caps are compatible if there exists at least one format
    /// that both can handle.
    pub fn intersects(&self, other: &Caps) -> bool {
        if self.is_any() || other.is_any() {
            return true;
        }
        self.0
            .iter()
            .any(|a| other.0.iter().any(|b| a.compatible(b)))
    }

    /// Find the first compatible format between two caps.
    ///
    /// Returns the format from `self` that is compatible with `other`.
    /// If either is "any", returns the other's preferred format.
    pub fn negotiate(&self, other: &Caps) -> Option<MediaFormat> {
        if self.is_any() {
            return other.preferred().cloned();
        }
        if other.is_any() {
            return self.preferred().cloned();
        }
        self.0
            .iter()
            .find(|a| other.0.iter().any(|b| a.compatible(b)))
            .cloned()
    }

    // ========================================================================
    // Convenience constructors for video formats
    // ========================================================================

    /// Create caps for raw video with specific dimensions and pixel format.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::format::{Caps, PixelFormat};
    ///
    /// let caps = Caps::video_raw(1920, 1080, PixelFormat::Rgba);
    /// assert!(!caps.is_any());
    /// ```
    pub fn video_raw(width: u32, height: u32, format: PixelFormat) -> Self {
        Self::new(MediaFormat::VideoRaw(VideoFormat {
            width,
            height,
            pixel_format: format,
            ..Default::default()
        }))
    }

    /// Create caps for raw video with any dimensions but specific pixel format.
    ///
    /// This is useful for sinks that can display any resolution but require
    /// a specific pixel format (e.g., RGBA for display).
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::format::{Caps, PixelFormat};
    ///
    /// let caps = Caps::video_raw_any_resolution(PixelFormat::Rgba);
    /// assert!(!caps.is_any());
    /// ```
    pub fn video_raw_any_resolution(format: PixelFormat) -> Self {
        // We use 0x0 dimensions to indicate "any resolution"
        // The negotiation system will recognize this as a wildcard
        Self::new(MediaFormat::VideoRaw(VideoFormat {
            width: 0,
            height: 0,
            pixel_format: format,
            ..Default::default()
        }))
    }

    /// Create caps for any raw video format.
    ///
    /// This matches any raw video, regardless of dimensions or pixel format.
    pub fn video_raw_any() -> Self {
        Self::new(MediaFormat::VideoRaw(VideoFormat::default()))
    }

    // ========================================================================
    // Convenience constructors for audio formats
    // ========================================================================

    /// Create caps for raw audio with specific parameters.
    ///
    /// # Example
    ///
    /// ```rust
    /// use parallax::format::{Caps, SampleFormat};
    ///
    /// let caps = Caps::audio_raw(48000, 2, SampleFormat::F32);
    /// assert!(!caps.is_any());
    /// ```
    pub fn audio_raw(sample_rate: u32, channels: u16, format: SampleFormat) -> Self {
        Self::new(MediaFormat::AudioRaw(AudioFormat {
            sample_rate,
            channels,
            sample_format: format,
            ..Default::default()
        }))
    }

    /// Create caps for raw audio with any sample rate but specific format and channels.
    pub fn audio_raw_any_rate(channels: u16, format: SampleFormat) -> Self {
        Self::new(MediaFormat::AudioRaw(AudioFormat {
            sample_rate: 0,
            channels,
            sample_format: format,
            ..Default::default()
        }))
    }

    /// Create caps for any raw audio format.
    pub fn audio_raw_any() -> Self {
        Self::new(MediaFormat::AudioRaw(AudioFormat::default()))
    }

    // ========================================================================
    // Inspection methods
    // ========================================================================

    /// Check if this caps represents raw video.
    pub fn is_video_raw(&self) -> bool {
        self.0.iter().any(|f| matches!(f, MediaFormat::VideoRaw(_)))
    }

    /// Check if this caps represents raw audio.
    pub fn is_audio_raw(&self) -> bool {
        self.0.iter().any(|f| matches!(f, MediaFormat::AudioRaw(_)))
    }

    /// Get the pixel format if this is a single video format.
    pub fn video_pixel_format(&self) -> Option<PixelFormat> {
        match self.preferred()? {
            MediaFormat::VideoRaw(v) => Some(v.pixel_format),
            _ => None,
        }
    }

    /// Get the video dimensions if this is a single video format.
    pub fn video_dimensions(&self) -> Option<(u32, u32)> {
        match self.preferred()? {
            MediaFormat::VideoRaw(v) => Some((v.width, v.height)),
            _ => None,
        }
    }
}

impl Default for Caps {
    fn default() -> Self {
        Self::any()
    }
}

// Convenient conversions
impl From<VideoFormat> for MediaFormat {
    fn from(v: VideoFormat) -> Self {
        Self::VideoRaw(v)
    }
}

impl From<AudioFormat> for MediaFormat {
    fn from(v: AudioFormat) -> Self {
        Self::AudioRaw(v)
    }
}

impl From<VideoCodec> for MediaFormat {
    fn from(v: VideoCodec) -> Self {
        Self::Video(v)
    }
}

impl From<AudioCodec> for MediaFormat {
    fn from(v: AudioCodec) -> Self {
        Self::Audio(v)
    }
}

impl From<RtpFormat> for MediaFormat {
    fn from(v: RtpFormat) -> Self {
        Self::Rtp(v)
    }
}

impl From<MediaFormat> for Caps {
    fn from(f: MediaFormat) -> Self {
        Self::new(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_format() {
        let fmt = VideoFormat::new(1920, 1080, PixelFormat::I420, Framerate::FPS_30);
        assert_eq!(fmt.width, 1920);
        assert_eq!(fmt.height, 1080);
        assert_eq!(fmt.frame_size(), 1920 * 1080 * 3 / 2); // I420 = 1.5 bytes per pixel
    }

    #[test]
    fn test_framerate() {
        assert_eq!(Framerate::FPS_30.fps(), 30.0);
        assert!((Framerate::FPS_29_97.fps() - 29.97).abs() < 0.01);
        assert_eq!(Framerate::FPS_30.frame_duration_ns(), 33_333_333);
    }

    #[test]
    fn test_audio_format() {
        let fmt = AudioFormat::CD_QUALITY;
        assert_eq!(fmt.sample_rate, 44100);
        assert_eq!(fmt.channels, 2);
        assert_eq!(fmt.bytes_per_frame(), 4); // 2 bytes * 2 channels
    }

    #[test]
    fn test_media_format_compatibility() {
        let h264 = MediaFormat::Video(VideoCodec::H264);
        let h265 = MediaFormat::Video(VideoCodec::H265);
        let bytes = MediaFormat::Bytes;

        assert!(h264.compatible(&h264));
        assert!(!h264.compatible(&h265));
        assert!(h264.compatible(&bytes));
        assert!(bytes.compatible(&h264));
    }

    #[test]
    fn test_caps_any() {
        let any = Caps::any();
        assert!(any.is_any());
        assert!(any.formats().is_empty());
    }

    #[test]
    fn test_caps_fixed() {
        let caps = Caps::new(MediaFormat::Video(VideoCodec::H264));
        assert!(caps.is_fixed());
        assert!(!caps.is_any());
        assert_eq!(caps.formats().len(), 1);
    }

    #[test]
    fn test_caps_intersects() {
        let h264 = Caps::new(MediaFormat::Video(VideoCodec::H264));
        let h265 = Caps::new(MediaFormat::Video(VideoCodec::H265));
        let multi = Caps::many([
            MediaFormat::Video(VideoCodec::H264),
            MediaFormat::Video(VideoCodec::H265),
        ]);
        let any = Caps::any();

        assert!(h264.intersects(&h264));
        assert!(!h264.intersects(&h265));
        assert!(h264.intersects(&multi));
        assert!(h264.intersects(&any));
        assert!(any.intersects(&h264));
    }

    #[test]
    fn test_caps_negotiate() {
        let h264 = Caps::new(MediaFormat::Video(VideoCodec::H264));
        let multi = Caps::many([
            MediaFormat::Video(VideoCodec::H265),
            MediaFormat::Video(VideoCodec::H264),
        ]);
        let any = Caps::any();

        assert_eq!(
            h264.negotiate(&multi),
            Some(MediaFormat::Video(VideoCodec::H264))
        );
        assert_eq!(
            any.negotiate(&h264),
            Some(MediaFormat::Video(VideoCodec::H264))
        );
        assert_eq!(
            h264.negotiate(&any),
            Some(MediaFormat::Video(VideoCodec::H264))
        );
    }

    #[test]
    fn test_codec_data() {
        let data = CodecData::new(vec![0x00, 0x00, 0x01, 0x67]);
        assert_eq!(data.len(), 4);
        assert_eq!(data.as_slice(), &[0x00, 0x00, 0x01, 0x67]);
    }

    #[test]
    fn test_rtp_format() {
        let rtp = RtpFormat::H264;
        assert_eq!(rtp.payload_type, 96);
        assert_eq!(rtp.clock_rate, 90000);
        assert_eq!(rtp.encoding, RtpEncoding::H264);
    }

    // ========================================================================
    // CapsValue tests
    // ========================================================================

    #[test]
    fn test_caps_value_fixed() {
        let fixed: CapsValue<u32> = CapsValue::Fixed(1920);
        assert!(fixed.is_fixed());
        assert!(!fixed.is_any());
        assert!(fixed.accepts(&1920));
        assert!(!fixed.accepts(&1080));
        assert_eq!(fixed.fixate(), Some(1920));
    }

    #[test]
    fn test_caps_value_range() {
        let range: CapsValue<u32> = CapsValue::Range {
            min: 720,
            max: 1920,
        };
        assert!(!range.is_fixed());
        assert!(range.accepts(&720));
        assert!(range.accepts(&1080));
        assert!(range.accepts(&1920));
        assert!(!range.accepts(&480));
        assert!(!range.accepts(&4096));
        assert_eq!(range.fixate(), Some(720)); // min is preferred
    }

    #[test]
    fn test_caps_value_list() {
        let list: CapsValue<u32> = CapsValue::List(vec![1920, 1280, 720]);
        assert!(list.accepts(&1920));
        assert!(list.accepts(&1280));
        assert!(!list.accepts(&1080));
        assert_eq!(list.fixate(), Some(1920)); // first is preferred
    }

    #[test]
    fn test_caps_value_any() {
        let any: CapsValue<u32> = CapsValue::Any;
        assert!(any.is_any());
        assert!(any.accepts(&0));
        assert!(any.accepts(&u32::MAX));
        assert_eq!(any.fixate(), None);
        assert_eq!(any.fixate_with_default(1080), 1080);
    }

    #[test]
    fn test_caps_value_intersect_fixed_fixed() {
        let a: CapsValue<u32> = CapsValue::Fixed(1920);
        let b: CapsValue<u32> = CapsValue::Fixed(1920);
        let c: CapsValue<u32> = CapsValue::Fixed(1080);

        assert_eq!(a.intersect(&b), Some(CapsValue::Fixed(1920)));
        assert_eq!(a.intersect(&c), None);
    }

    #[test]
    fn test_caps_value_intersect_fixed_range() {
        let fixed: CapsValue<u32> = CapsValue::Fixed(1080);
        let range: CapsValue<u32> = CapsValue::Range {
            min: 720,
            max: 1920,
        };
        let out_of_range: CapsValue<u32> = CapsValue::Fixed(480);

        assert_eq!(fixed.intersect(&range), Some(CapsValue::Fixed(1080)));
        assert_eq!(range.intersect(&fixed), Some(CapsValue::Fixed(1080)));
        assert_eq!(out_of_range.intersect(&range), None);
    }

    #[test]
    fn test_caps_value_intersect_range_range() {
        let a: CapsValue<u32> = CapsValue::Range {
            min: 720,
            max: 1920,
        };
        let b: CapsValue<u32> = CapsValue::Range {
            min: 1080,
            max: 4096,
        };
        let c: CapsValue<u32> = CapsValue::Range { min: 100, max: 500 };

        // Overlap: 1080-1920
        assert_eq!(
            a.intersect(&b),
            Some(CapsValue::Range {
                min: 1080,
                max: 1920
            })
        );
        // No overlap
        assert_eq!(a.intersect(&c), None);
    }

    #[test]
    fn test_caps_value_intersect_list_list() {
        let a: CapsValue<u32> = CapsValue::List(vec![1920, 1280, 720]);
        let b: CapsValue<u32> = CapsValue::List(vec![1080, 1280, 720, 480]);

        // Common: 1280, 720 (order from first list)
        assert_eq!(a.intersect(&b), Some(CapsValue::List(vec![1280, 720])));
    }

    #[test]
    fn test_caps_value_intersect_any() {
        let any: CapsValue<u32> = CapsValue::Any;
        let fixed: CapsValue<u32> = CapsValue::Fixed(1920);

        assert_eq!(any.intersect(&fixed), Some(CapsValue::Fixed(1920)));
        assert_eq!(fixed.intersect(&any), Some(CapsValue::Fixed(1920)));
    }

    #[test]
    fn test_caps_value_from() {
        let from_value: CapsValue<u32> = 1920.into();
        assert_eq!(from_value, CapsValue::Fixed(1920));

        let from_range: CapsValue<u32> = (720..=1920).into();
        assert_eq!(
            from_range,
            CapsValue::Range {
                min: 720,
                max: 1920
            }
        );

        let from_vec: CapsValue<u32> = vec![1920, 1080].into();
        assert_eq!(from_vec, CapsValue::List(vec![1920, 1080]));

        let from_single_vec: CapsValue<u32> = vec![1920].into();
        assert_eq!(from_single_vec, CapsValue::Fixed(1920));

        let from_empty_vec: CapsValue<u32> = Vec::new().into();
        assert_eq!(from_empty_vec, CapsValue::Any);
    }

    // ========================================================================
    // VideoFormatCaps tests
    // ========================================================================

    #[test]
    fn test_video_format_caps_any() {
        let caps = VideoFormatCaps::any();
        assert!(caps.width.is_any());
        assert!(caps.height.is_any());
        assert!(caps.pixel_format.is_any());
        assert!(caps.framerate.is_any());
    }

    #[test]
    fn test_video_format_caps_fixed() {
        let format = VideoFormat::new(1920, 1080, PixelFormat::I420, Framerate::FPS_30);
        let caps = VideoFormatCaps::fixed(format);

        assert!(caps.is_fixed());
        assert_eq!(caps.fixate(), Some(format));
    }

    #[test]
    fn test_video_format_caps_intersect() {
        let producer = VideoFormatCaps {
            width: CapsValue::List(vec![1920, 1280]),
            height: CapsValue::List(vec![1080, 720]),
            pixel_format: CapsValue::List(vec![PixelFormat::I420, PixelFormat::Nv12]),
            framerate: CapsValue::Any,
        };

        let consumer = VideoFormatCaps {
            width: CapsValue::Range {
                min: 1280,
                max: 1920,
            },
            height: CapsValue::Fixed(720),
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            framerate: CapsValue::Fixed(Framerate::FPS_30),
        };

        let result = producer.intersect(&consumer).unwrap();
        assert_eq!(result.width, CapsValue::List(vec![1920, 1280]));
        assert_eq!(result.height, CapsValue::Fixed(720));
        assert_eq!(result.pixel_format, CapsValue::Fixed(PixelFormat::I420));
        assert_eq!(result.framerate, CapsValue::Fixed(Framerate::FPS_30));

        let fixed = result.fixate().unwrap();
        assert_eq!(fixed.width, 1920);
        assert_eq!(fixed.height, 720);
    }

    // ========================================================================
    // AudioFormatCaps tests
    // ========================================================================

    #[test]
    fn test_audio_format_caps_fixed() {
        let format = AudioFormat::CD_QUALITY;
        let caps = AudioFormatCaps::fixed(format);

        assert!(caps.is_fixed());
        assert_eq!(caps.fixate(), Some(format));
    }

    #[test]
    fn test_audio_format_caps_intersect() {
        let a = AudioFormatCaps {
            sample_rate: CapsValue::List(vec![48000, 44100]),
            channels: CapsValue::Range { min: 1, max: 8 },
            sample_format: CapsValue::Any,
        };

        let b = AudioFormatCaps {
            sample_rate: CapsValue::Fixed(48000),
            channels: CapsValue::Fixed(2),
            sample_format: CapsValue::Fixed(SampleFormat::S16),
        };

        let result = a.intersect(&b).unwrap();
        assert_eq!(result.sample_rate, CapsValue::Fixed(48000));
        assert_eq!(result.channels, CapsValue::Fixed(2));
        assert_eq!(result.sample_format, CapsValue::Fixed(SampleFormat::S16));
    }

    // ========================================================================
    // FormatCaps tests
    // ========================================================================

    #[test]
    fn test_format_caps_intersect() {
        let a = FormatCaps::VideoRaw(VideoFormatCaps::any());
        let b = FormatCaps::VideoRaw(VideoFormatCaps::fixed(VideoFormat::new(
            1920,
            1080,
            PixelFormat::I420,
            Framerate::FPS_30,
        )));

        let result = a.intersect(&b).unwrap();
        assert!(matches!(result, FormatCaps::VideoRaw(_)));
    }

    #[test]
    fn test_format_caps_incompatible() {
        let video = FormatCaps::VideoRaw(VideoFormatCaps::any());
        let audio = FormatCaps::AudioRaw(AudioFormatCaps::any());

        assert!(video.intersect(&audio).is_none());
    }

    #[test]
    fn test_format_caps_bytes_compatible() {
        let bytes = FormatCaps::Bytes;
        let video = FormatCaps::VideoRaw(VideoFormatCaps::any());

        // Bytes is compatible with anything
        assert!(bytes.intersect(&video).is_some());
    }

    // ========================================================================
    // MemoryCaps tests
    // ========================================================================

    #[test]
    fn test_memory_caps_cpu_only() {
        let caps = MemoryCaps::cpu_only();
        assert_eq!(caps.fixate(), Some(MemoryType::Cpu));
    }

    #[test]
    fn test_memory_caps_intersect() {
        let a = MemoryCaps::cpu_only();
        let b = MemoryCaps::any();

        let result = a.intersect(&b).unwrap();
        assert_eq!(result.fixate(), Some(MemoryType::Cpu));
    }

    // ========================================================================
    // MediaCaps tests
    // ========================================================================

    #[test]
    fn test_media_caps_from_format() {
        let format = MediaFormat::Video(VideoCodec::H264);
        let caps: MediaCaps = format.into();

        assert!(matches!(caps.format, FormatCaps::Video(VideoCodec::H264)));
        assert!(caps.memory.types.is_any());
    }

    #[test]
    fn test_media_caps_intersect() {
        let a = MediaCaps::new(
            FormatCaps::VideoRaw(VideoFormatCaps::any()),
            MemoryCaps::any(),
        );
        let b = MediaCaps::new(
            FormatCaps::VideoRaw(VideoFormatCaps::fixed(VideoFormat::new(
                1920,
                1080,
                PixelFormat::I420,
                Framerate::FPS_30,
            ))),
            MemoryCaps::cpu_only(),
        );

        let result = a.intersect(&b).unwrap();
        assert_eq!(result.fixate_memory(), Some(MemoryType::Cpu));
        assert!(result.fixate_format().is_some());
    }

    #[test]
    fn test_framerate_ord() {
        assert!(Framerate::FPS_24 < Framerate::FPS_30);
        assert!(Framerate::FPS_30 < Framerate::FPS_60);
        assert!(Framerate::FPS_29_97 < Framerate::FPS_30);
    }

    // ========================================================================
    // FormatMemoryCap tests
    // ========================================================================

    #[test]
    fn test_format_memory_cap_any() {
        let cap = FormatMemoryCap::any();
        assert!(matches!(cap.format, FormatCaps::Any));
        assert!(cap.memory.types.is_any());
    }

    #[test]
    fn test_format_memory_cap_intersect() {
        // GPU RGB
        let gpu_rgb = FormatMemoryCap::new(
            VideoFormatCaps::rgba().into(),
            MemoryCaps {
                types: CapsValue::Fixed(MemoryType::GpuDevice),
                ..MemoryCaps::any()
            },
        );

        // Any format on GPU
        let any_gpu = FormatMemoryCap::new(
            FormatCaps::Any,
            MemoryCaps {
                types: CapsValue::Fixed(MemoryType::GpuDevice),
                ..MemoryCaps::any()
            },
        );

        // Should intersect: RGBA on GPU
        let result = gpu_rgb.intersect(&any_gpu);
        assert!(result.is_some());
        let intersected = result.unwrap();
        assert!(matches!(intersected.format, FormatCaps::VideoRaw(_)));
        assert_eq!(
            intersected.memory.types,
            CapsValue::Fixed(MemoryType::GpuDevice)
        );
    }

    #[test]
    fn test_format_memory_cap_no_intersect_memory_mismatch() {
        // GPU only
        let gpu = FormatMemoryCap::new(
            FormatCaps::Any,
            MemoryCaps {
                types: CapsValue::Fixed(MemoryType::GpuDevice),
                ..MemoryCaps::any()
            },
        );

        // CPU only
        let cpu = FormatMemoryCap::any_cpu();

        // Should NOT intersect: memory types don't match
        assert!(gpu.intersect(&cpu).is_none());
    }

    // ========================================================================
    // ElementMediaCaps tests
    // ========================================================================

    #[test]
    fn test_element_media_caps_any() {
        let caps = ElementMediaCaps::any();
        assert!(caps.is_any());
        assert_eq!(caps.capabilities().len(), 1);
    }

    #[test]
    fn test_element_media_caps_multi_format() {
        // Camera supports YUYV and NV12
        let camera_caps = ElementMediaCaps::new(vec![
            FormatMemoryCap::new(
                VideoFormatCaps {
                    pixel_format: CapsValue::Fixed(PixelFormat::Yuyv),
                    ..VideoFormatCaps::any()
                }
                .into(),
                MemoryCaps::cpu_only(),
            ),
            FormatMemoryCap::new(
                VideoFormatCaps {
                    pixel_format: CapsValue::Fixed(PixelFormat::Nv12),
                    ..VideoFormatCaps::any()
                }
                .into(),
                MemoryCaps::cpu_only(),
            ),
        ]);

        assert!(!camera_caps.is_any());
        assert_eq!(camera_caps.capabilities().len(), 2);

        // Consumer wants NV12
        let consumer = ElementMediaCaps::single(FormatMemoryCap::new(
            VideoFormatCaps {
                pixel_format: CapsValue::Fixed(PixelFormat::Nv12),
                ..VideoFormatCaps::any()
            }
            .into(),
            MemoryCaps::cpu_only(),
        ));

        // Should find NV12 as common format
        let result = camera_caps.intersect(&consumer);
        assert!(result.is_some());
    }

    #[test]
    fn test_element_media_caps_format_memory_coupling() {
        // GPU filter supports RGB on GPU only, but YUV on CPU
        let gpu_filter = ElementMediaCaps::new(vec![
            // RGB on GPU (preferred)
            FormatMemoryCap::new(
                VideoFormatCaps::rgba().into(),
                MemoryCaps {
                    types: CapsValue::Fixed(MemoryType::GpuDevice),
                    ..MemoryCaps::any()
                },
            ),
            // YUV on CPU (fallback)
            FormatMemoryCap::new(VideoFormatCaps::yuv420().into(), MemoryCaps::cpu_only()),
        ]);

        // Source produces RGB on CPU
        let rgb_cpu_source = ElementMediaCaps::single(FormatMemoryCap::new(
            VideoFormatCaps::rgba().into(),
            MemoryCaps::cpu_only(),
        ));

        // Source produces YUV on CPU
        let yuv_cpu_source = ElementMediaCaps::single(FormatMemoryCap::new(
            VideoFormatCaps::yuv420().into(),
            MemoryCaps::cpu_only(),
        ));

        // Source produces RGB on GPU
        let rgb_gpu_source = ElementMediaCaps::single(FormatMemoryCap::new(
            VideoFormatCaps::rgba().into(),
            MemoryCaps {
                types: CapsValue::Fixed(MemoryType::GpuDevice),
                ..MemoryCaps::any()
            },
        ));

        // RGB CPU should NOT match (filter only supports RGB on GPU)
        assert!(gpu_filter.intersect(&rgb_cpu_source).is_none());

        // YUV CPU should match (filter supports YUV on CPU)
        assert!(gpu_filter.intersect(&yuv_cpu_source).is_some());

        // RGB GPU should match (filter's preferred option)
        let result = gpu_filter.intersect(&rgb_gpu_source);
        assert!(result.is_some());
        let intersected = result.unwrap();
        assert_eq!(
            intersected.memory.types,
            CapsValue::Fixed(MemoryType::GpuDevice)
        );
    }

    #[test]
    fn test_element_media_caps_from_simple_caps() {
        let simple = Caps::many([
            MediaFormat::Video(VideoCodec::H264),
            MediaFormat::Video(VideoCodec::H265),
        ]);

        let elem_caps = ElementMediaCaps::from_caps(&simple);
        assert_eq!(elem_caps.capabilities().len(), 2);

        // Each should have CPU memory (default)
        for cap in elem_caps.capabilities() {
            assert_eq!(cap.memory.types, CapsValue::Fixed(MemoryType::Cpu));
        }
    }

    #[test]
    fn test_element_media_caps_intersect_all() {
        // Source supports both formats on CPU
        let source = ElementMediaCaps::new(vec![
            FormatMemoryCap::new(VideoFormatCaps::rgba().into(), MemoryCaps::cpu_only()),
            FormatMemoryCap::new(VideoFormatCaps::yuv420().into(), MemoryCaps::cpu_only()),
        ]);

        // Sink accepts both too
        let sink = ElementMediaCaps::new(vec![
            FormatMemoryCap::new(VideoFormatCaps::yuv420().into(), MemoryCaps::cpu_only()),
            FormatMemoryCap::new(VideoFormatCaps::rgba().into(), MemoryCaps::cpu_only()),
        ]);

        // Should find both as common (RGBA matches RGBA, YUV matches YUV)
        // Note: RGBA doesn't match YUV and vice versa, so we get 2 matches not 4
        let all = source.intersect_all(&sink);
        assert_eq!(all.len(), 2); // RGBA-RGBA and YUV-YUV

        // First match should be RGBA (source's first preference matched against sink)
        let first = source.intersect(&sink).unwrap();
        assert!(matches!(first.format, FormatCaps::VideoRaw(_)));
    }
}
