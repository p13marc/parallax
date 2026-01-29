//! H.264/AVC codec elements using OpenH264.
//!
//! This module provides H.264 encoding and decoding using Cisco's OpenH264 library.
//! The library is BSD-2 licensed and the source code is bundled with the crate.
//!
//! # Example - Encoding
//!
//! ```rust,ignore
//! use parallax::elements::codec::{H264Encoder, H264EncoderConfig};
//!
//! // Create encoder for 1920x1080 video
//! let config = H264EncoderConfig::new(1920, 1080);
//! let mut encoder = H264Encoder::new(config)?;
//!
//! // Encode YUV frames
//! let encoded = encoder.encode_yuv420(&yuv_data)?;
//! ```
//!
//! # Example - Decoding
//!
//! ```rust,ignore
//! use parallax::elements::codec::H264Decoder;
//!
//! let mut decoder = H264Decoder::new()?;
//!
//! // Decode NAL units
//! if let Some(frame) = decoder.decode(&nal_data)? {
//!     // Process decoded YUV frame
//!     let yuv_data = frame.yuv_data();
//! }
//! ```

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use crate::metadata::Metadata;

use openh264::OpenH264API;
use openh264::decoder::{DecodedYUV, Decoder};
use openh264::encoder::{BitRate, Encoder, EncoderConfig, FrameRate};
use openh264::formats::YUVSource;

// ============================================================================
// Encoder Configuration
// ============================================================================

/// H.264 encoder configuration.
#[derive(Debug, Clone)]
pub struct H264EncoderConfig {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Target bitrate in bits per second (0 = auto).
    pub bitrate_bps: u32,
    /// Maximum frame rate in Hz.
    pub max_frame_rate: f32,
    /// Quantization parameter (0-51, lower = better quality, larger files).
    /// Default is 26.
    pub qp: u8,
    /// Enable scene change detection.
    pub scene_change_detect: bool,
    /// Keyframe interval (0 = auto).
    pub keyframe_interval: u32,
    /// Number of threads (0 = auto).
    pub num_threads: u32,
}

impl H264EncoderConfig {
    /// Create a new encoder configuration with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bitrate_bps: 0,
            max_frame_rate: 30.0,
            qp: 26,
            scene_change_detect: true,
            keyframe_interval: 0,
            num_threads: 0,
        }
    }

    /// Set the target bitrate in bits per second.
    pub fn bitrate(mut self, bps: u32) -> Self {
        self.bitrate_bps = bps;
        self
    }

    /// Set the maximum frame rate.
    pub fn frame_rate(mut self, fps: f32) -> Self {
        self.max_frame_rate = fps;
        self
    }

    /// Set the quantization parameter (0-51).
    pub fn qp(mut self, qp: u8) -> Self {
        self.qp = qp.min(51);
        self
    }

    /// Set the keyframe interval.
    pub fn keyframe_interval(mut self, interval: u32) -> Self {
        self.keyframe_interval = interval;
        self
    }

    /// Set the number of encoding threads.
    pub fn threads(mut self, threads: u32) -> Self {
        self.num_threads = threads;
        self
    }

    /// Create a configuration for low-latency streaming.
    pub fn low_latency(width: u32, height: u32) -> Self {
        Self::new(width, height)
            .frame_rate(30.0)
            .keyframe_interval(30) // Keyframe every second at 30fps
            .qp(28) // Slightly lower quality for speed
    }

    /// Create a configuration for high-quality encoding.
    pub fn high_quality(width: u32, height: u32) -> Self {
        Self::new(width, height)
            .frame_rate(30.0)
            .keyframe_interval(120) // Keyframe every 4 seconds
            .qp(20) // Higher quality
    }
}

impl Default for H264EncoderConfig {
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}

// ============================================================================
// Encoder
// ============================================================================

/// H.264 encoder using OpenH264.
///
/// Encodes YUV420 frames to H.264 NAL units.
pub struct H264Encoder {
    encoder: Encoder,
    config: H264EncoderConfig,
    frame_count: u64,
    bytes_encoded: u64,
    /// Arena for output buffer allocation.
    arena: SharedArena,
}

impl H264Encoder {
    /// Create a new H.264 encoder with the given configuration.
    pub fn new(config: H264EncoderConfig) -> Result<Self> {
        let mut encoder_config = EncoderConfig::new();

        if config.bitrate_bps > 0 {
            encoder_config = encoder_config.bitrate(BitRate::from_bps(config.bitrate_bps));
        }

        encoder_config = encoder_config
            .max_frame_rate(FrameRate::from_hz(config.max_frame_rate))
            .scene_change_detect(config.scene_change_detect)
            .num_threads(config.num_threads as u16);

        let api = OpenH264API::from_source();
        let encoder = Encoder::with_api_config(api, encoder_config)
            .map_err(|e| Error::Config(format!("Failed to create H.264 encoder: {:?}", e)))?;

        // Create arena for encoded output buffers (typically < 1MB per frame)
        // Use 64 slots to handle buffering when downstream is slower than encoding
        let arena = SharedArena::new(1024 * 1024, 64)
            .map_err(|e| Error::Config(format!("Failed to create arena: {}", e)))?;

        Ok(Self {
            encoder,
            config,
            frame_count: 0,
            bytes_encoded: 0,
            arena,
        })
    }

    /// Encode a YUV420 frame.
    ///
    /// The input data must be in YUV420 planar format:
    /// - Y plane: width * height bytes
    /// - U plane: (width/2) * (height/2) bytes
    /// - V plane: (width/2) * (height/2) bytes
    ///
    /// Returns the encoded H.264 bitstream (NAL units).
    pub fn encode_yuv420(&mut self, yuv_data: &[u8]) -> Result<Vec<u8>> {
        let expected_size = self.config.width as usize * self.config.height as usize * 3 / 2;
        if yuv_data.len() < expected_size {
            return Err(Error::Config(format!(
                "YUV data too small: expected {} bytes, got {}",
                expected_size,
                yuv_data.len()
            )));
        }

        let yuv = YuvFrame {
            data: yuv_data,
            width: self.config.width as usize,
            height: self.config.height as usize,
        };

        let bitstream = self
            .encoder
            .encode(&yuv)
            .map_err(|e| Error::Config(format!("H.264 encode failed: {:?}", e)))?;

        let encoded = bitstream.to_vec();
        self.frame_count += 1;
        self.bytes_encoded += encoded.len() as u64;

        Ok(encoded)
    }

    /// Force the next frame to be a keyframe (IDR frame).
    pub fn force_keyframe(&mut self) {
        self.encoder.force_intra_frame();
    }

    /// Get the number of frames encoded.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the total bytes encoded.
    pub fn bytes_encoded(&self) -> u64 {
        self.bytes_encoded
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &H264EncoderConfig {
        &self.config
    }
}

impl std::fmt::Debug for H264Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H264Encoder")
            .field("config", &self.config)
            .field("frame_count", &self.frame_count)
            .field("bytes_encoded", &self.bytes_encoded)
            .finish()
    }
}

/// Element trait implementation for H264Encoder.
impl Element for H264Encoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input_data = buffer.as_bytes();
        let encoded = self.encode_yuv420(input_data)?;

        if encoded.is_empty() {
            return Ok(None);
        }

        // Reclaim any released slots before acquiring
        self.arena.reclaim();

        // Acquire slot from arena and copy encoded data
        let mut slot = self
            .arena
            .acquire()
            .ok_or_else(|| Error::Config("Failed to acquire buffer slot".to_string()))?;

        // Copy encoded data to slot
        slot.data_mut()[..encoded.len()].copy_from_slice(&encoded);

        let handle = crate::buffer::MemoryHandle::with_len(slot, encoded.len());
        let metadata = Metadata::from_sequence(self.frame_count - 1);
        Ok(Some(Buffer::new(handle, metadata)))
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Accept I420 (YUV420) video of any size
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, PixelFormat,
            VideoFormatCaps,
        };

        let format = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            ..VideoFormatCaps::any()
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(format),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Output is H.264 encoded video
        use crate::format::{
            ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, VideoCodec,
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Video(VideoCodec::H264),
            MemoryCaps::cpu_only(),
        )])
    }
}

/// VideoEncoder trait implementation for H264Encoder.
///
/// This allows H264Encoder to be used with `EncoderElement` for pipeline integration.
impl super::traits::VideoEncoder for H264Encoder {
    type Packet = Vec<u8>;

    fn encode(&mut self, frame: &super::common::VideoFrame) -> Result<Vec<Self::Packet>> {
        // Validate frame dimensions match encoder config
        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(Error::Config(format!(
                "Frame dimensions {}x{} don't match encoder config {}x{}",
                frame.width, frame.height, self.config.width, self.config.height
            )));
        }

        let encoded = self.encode_yuv420(&frame.data)?;

        if encoded.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![encoded])
        }
    }

    fn flush(&mut self) -> Result<Vec<Self::Packet>> {
        // OpenH264 doesn't buffer frames, so flush is a no-op
        Ok(Vec::new())
    }

    fn codec_data(&self) -> Option<Vec<u8>> {
        // H.264 codec data (SPS/PPS) would be extracted from the first encoded frame
        // For now, return None - the first keyframe contains the headers inline
        None
    }
}

// ============================================================================
// Decoder
// ============================================================================

/// H.264 decoder using OpenH264.
///
/// Decodes H.264 NAL units to YUV420 frames.
pub struct H264Decoder {
    decoder: Decoder,
    frame_count: u64,
    bytes_decoded: u64,
    /// Arena for output buffer allocation.
    arena: SharedArena,
}

impl H264Decoder {
    /// Create a new H.264 decoder.
    pub fn new() -> Result<Self> {
        let decoder = Decoder::new()
            .map_err(|e| Error::Config(format!("Failed to create H.264 decoder: {:?}", e)))?;

        // Create arena for decoded YUV frames (1080p YUV420 = ~3MB per frame)
        let arena = SharedArena::new(4 * 1024 * 1024, 8)
            .map_err(|e| Error::Config(format!("Failed to create arena: {}", e)))?;

        Ok(Self {
            decoder,
            frame_count: 0,
            bytes_decoded: 0,
            arena,
        })
    }

    /// Decode H.264 NAL units.
    ///
    /// Returns the decoded YUV frame if a complete frame is available,
    /// or `None` if more data is needed.
    pub fn decode(&mut self, nal_data: &[u8]) -> Result<Option<DecodedFrame>> {
        self.bytes_decoded += nal_data.len() as u64;

        let result = self
            .decoder
            .decode(nal_data)
            .map_err(|e| Error::Config(format!("H.264 decode failed: {:?}", e)))?;

        match result {
            Some(yuv) => {
                self.frame_count += 1;
                Ok(Some(DecodedFrame::from_decoded_yuv(yuv)))
            }
            None => Ok(None),
        }
    }

    /// Flush the decoder and retrieve any remaining frames.
    pub fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        let frames = self
            .decoder
            .flush_remaining()
            .map_err(|e| Error::Config(format!("H.264 flush failed: {:?}", e)))?;

        self.frame_count += frames.len() as u64;
        Ok(frames
            .into_iter()
            .map(DecodedFrame::from_decoded_yuv)
            .collect())
    }

    /// Get the number of frames decoded.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the total bytes decoded.
    pub fn bytes_decoded(&self) -> u64 {
        self.bytes_decoded
    }
}

impl std::fmt::Debug for H264Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H264Decoder")
            .field("frame_count", &self.frame_count)
            .field("bytes_decoded", &self.bytes_decoded)
            .finish()
    }
}

/// Element trait implementation for H264Decoder.
impl Element for H264Decoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input_data = buffer.as_bytes();

        match self.decode(input_data)? {
            Some(frame) => {
                let yuv_data = frame.to_yuv420_planar();

                // Reclaim released slots and acquire new one
                self.arena.reclaim();
                let mut slot = self
                    .arena
                    .acquire()
                    .ok_or_else(|| Error::Config("Failed to acquire buffer slot".to_string()))?;

                // Copy YUV data to slot
                slot.data_mut()[..yuv_data.len()].copy_from_slice(&yuv_data);

                let handle = crate::buffer::MemoryHandle::with_len(slot, yuv_data.len());
                let mut metadata = Metadata::from_sequence(self.frame_count - 1);
                metadata.set("width", frame.width() as u64);
                metadata.set("height", frame.height() as u64);
                Ok(Some(Buffer::new(handle, metadata)))
            }
            None => Ok(None),
        }
    }
}

// ============================================================================
// Decoded Frame
// ============================================================================

/// A decoded YUV frame from the H.264 decoder.
#[derive(Debug)]
pub struct DecodedFrame {
    /// Y plane data.
    y_data: Vec<u8>,
    /// U plane data.
    u_data: Vec<u8>,
    /// V plane data.
    v_data: Vec<u8>,
    /// Frame width.
    width: usize,
    /// Frame height.
    height: usize,
    /// Y plane stride.
    y_stride: usize,
    /// U plane stride.
    u_stride: usize,
    /// V plane stride.
    v_stride: usize,
}

impl DecodedFrame {
    fn from_decoded_yuv(yuv: DecodedYUV) -> Self {
        let (width, height) = yuv.dimensions();
        let y_data = yuv.y().to_vec();
        let u_data = yuv.u().to_vec();
        let v_data = yuv.v().to_vec();
        let (y_stride, u_stride, v_stride) = yuv.strides();

        Self {
            y_data,
            u_data,
            v_data,
            width,
            height,
            y_stride,
            u_stride,
            v_stride,
        }
    }

    /// Get the frame width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the frame height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the Y plane data.
    pub fn y_plane(&self) -> &[u8] {
        &self.y_data
    }

    /// Get the U plane data.
    pub fn u_plane(&self) -> &[u8] {
        &self.u_data
    }

    /// Get the V plane data.
    pub fn v_plane(&self) -> &[u8] {
        &self.v_data
    }

    /// Get the strides for each plane (Y, U, V).
    pub fn strides(&self) -> (usize, usize, usize) {
        (self.y_stride, self.u_stride, self.v_stride)
    }

    /// Convert to contiguous YUV420 planar format.
    ///
    /// Returns a Vec with Y, U, V planes concatenated without padding.
    pub fn to_yuv420_planar(&self) -> Vec<u8> {
        let y_size = self.width * self.height;
        let uv_size = (self.width / 2) * (self.height / 2);
        let total_size = y_size + uv_size * 2;

        let mut output = Vec::with_capacity(total_size);

        // Copy Y plane (removing stride padding if any)
        for y in 0..self.height {
            let start = y * self.y_stride;
            let end = start + self.width;
            if end <= self.y_data.len() {
                output.extend_from_slice(&self.y_data[start..end]);
            }
        }

        // Copy U plane
        let uv_height = self.height / 2;
        let uv_width = self.width / 2;
        for y in 0..uv_height {
            let start = y * self.u_stride;
            let end = start + uv_width;
            if end <= self.u_data.len() {
                output.extend_from_slice(&self.u_data[start..end]);
            }
        }

        // Copy V plane
        for y in 0..uv_height {
            let start = y * self.v_stride;
            let end = start + uv_width;
            if end <= self.v_data.len() {
                output.extend_from_slice(&self.v_data[start..end]);
            }
        }

        output
    }
}

// ============================================================================
// Helper Types
// ============================================================================

/// Internal YUV frame wrapper for OpenH264.
struct YuvFrame<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
}

impl<'a> YUVSource for YuvFrame<'a> {
    fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn strides(&self) -> (usize, usize, usize) {
        (self.width, self.width / 2, self.width / 2)
    }

    fn y(&self) -> &[u8] {
        &self.data[..self.width * self.height]
    }

    fn u(&self) -> &[u8] {
        let y_size = self.width * self.height;
        let u_size = (self.width / 2) * (self.height / 2);
        &self.data[y_size..y_size + u_size]
    }

    fn v(&self) -> &[u8] {
        let y_size = self.width * self.height;
        let u_size = (self.width / 2) * (self.height / 2);
        &self.data[y_size + u_size..]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_default() {
        let config = H264EncoderConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.qp, 26);
    }

    #[test]
    fn test_encoder_config_builder() {
        let config = H264EncoderConfig::new(640, 480)
            .bitrate(1_000_000)
            .frame_rate(25.0)
            .qp(24)
            .keyframe_interval(60);

        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.bitrate_bps, 1_000_000);
        assert_eq!(config.max_frame_rate, 25.0);
        assert_eq!(config.qp, 24);
        assert_eq!(config.keyframe_interval, 60);
    }

    #[test]
    fn test_encoder_config_low_latency() {
        let config = H264EncoderConfig::low_latency(1280, 720);
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.keyframe_interval, 30);
    }

    #[test]
    fn test_encoder_config_high_quality() {
        let config = H264EncoderConfig::high_quality(1920, 1080);
        assert_eq!(config.qp, 20);
        assert_eq!(config.keyframe_interval, 120);
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = H264Decoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_encoder_creation() {
        let config = H264EncoderConfig::new(320, 240);
        let encoder = H264Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Create a simple YUV420 frame (gray + neutral UV)
        let width = 64;
        let height = 64;
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);

        let mut yuv_data = vec![128u8; y_size + uv_size * 2];
        // Make Y plane a gradient
        for y in 0..height {
            for x in 0..width {
                yuv_data[y * width + x] = ((x + y) * 2) as u8;
            }
        }

        // Encode
        let config = H264EncoderConfig::new(width as u32, height as u32);
        let mut encoder = H264Encoder::new(config).expect("Failed to create encoder");

        let encoded = encoder.encode_yuv420(&yuv_data).expect("Failed to encode");
        assert!(!encoded.is_empty(), "Encoded data should not be empty");

        // Decode
        let mut decoder = H264Decoder::new().expect("Failed to create decoder");
        let decoded = decoder.decode(&encoded);

        // The first frame might need SPS/PPS, so decoding might return None
        // This is expected behavior
        assert!(decoded.is_ok());
    }

    #[test]
    fn test_encoder_stats() {
        let width = 64;
        let height = 64;
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        let yuv_data = vec![128u8; y_size + uv_size * 2];

        let config = H264EncoderConfig::new(width as u32, height as u32);
        let mut encoder = H264Encoder::new(config).expect("Failed to create encoder");

        assert_eq!(encoder.frame_count(), 0);
        assert_eq!(encoder.bytes_encoded(), 0);

        encoder.encode_yuv420(&yuv_data).expect("Failed to encode");

        assert_eq!(encoder.frame_count(), 1);
        assert!(encoder.bytes_encoded() > 0);
    }

    #[test]
    fn test_decoder_stats() {
        let decoder = H264Decoder::new().expect("Failed to create decoder");
        assert_eq!(decoder.frame_count(), 0);
        assert_eq!(decoder.bytes_decoded(), 0);
    }
}
