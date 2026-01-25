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

use smallvec::SmallVec;

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
            PixelFormat::I420 => pixels * 3 / 2, // Y + U/4 + V/4
            PixelFormat::Nv12 => pixels * 3 / 2, // Y + UV/2
            PixelFormat::Yuyv => pixels * 2,     // 2 bytes per pixel
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => pixels * 3,
            PixelFormat::Rgba | PixelFormat::Bgra => pixels * 4,
        }
    }
}

/// Pixel formats (color space and memory layout).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum PixelFormat {
    /// YUV 4:2:0 planar (Y plane, then U plane, then V plane).
    /// Most common format for video codecs.
    #[default]
    I420 = 0,
    /// YUV 4:2:0 semi-planar (Y plane, then interleaved UV plane).
    /// Common for hardware decoders.
    Nv12,
    /// YUV 4:2:2 packed (Y0 U Y1 V).
    Yuyv,
    /// RGB 8-bit per channel, packed.
    Rgb24,
    /// RGBA 8-bit per channel, packed.
    Rgba,
    /// BGR 8-bit per channel, packed.
    Bgr24,
    /// BGRA 8-bit per channel, packed.
    Bgra,
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
}

/// Audio sample formats.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
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
// Caps (Capabilities)
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
}
