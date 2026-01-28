//! AV1 software encoder using rav1e.
//!
//! rav1e is an AV1 encoder written in Rust, known for its safety and quality.
//! It is pure Rust and does not require any system dependencies.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{Rav1eEncoder, Rav1eConfig};
//!
//! let config = Rav1eConfig::default()
//!     .dimensions(1920, 1080)
//!     .speed(6);
//! let encoder = Rav1eEncoder::new(config)?;
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Element, ExecutionHints};
use crate::error::{Error, Result};
use crate::memory::{CpuSegment, MemorySegment};
use std::sync::Arc;

use super::common::{PixelFormat, VideoFrame};
use super::traits::VideoEncoder;

/// Configuration for the rav1e AV1 encoder.
#[derive(Clone, Debug)]
pub struct Rav1eConfig {
    /// Video width.
    pub width: usize,
    /// Video height.
    pub height: usize,
    /// Speed preset (0-10, higher = faster but lower quality).
    pub speed: usize,
    /// Quantizer (0-255, lower = higher quality).
    pub quantizer: usize,
    /// Bitrate in bits per second (0 = constant quality mode).
    pub bitrate: usize,
    /// Frames per second numerator.
    pub timebase_num: u64,
    /// Frames per second denominator.
    pub timebase_den: u64,
    /// Pixel format.
    pub pixel_format: PixelFormat,
    /// Bit depth (8 or 10).
    pub bit_depth: usize,
}

impl Default for Rav1eConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            speed: 6,
            quantizer: 100,
            bitrate: 0,
            timebase_num: 1,
            timebase_den: 30,
            pixel_format: PixelFormat::I420,
            bit_depth: 8,
        }
    }
}

impl Rav1eConfig {
    /// Set video dimensions.
    pub fn dimensions(mut self, width: usize, height: usize) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Set width.
    pub fn width(mut self, width: usize) -> Self {
        self.width = width;
        self
    }

    /// Set height.
    pub fn height(mut self, height: usize) -> Self {
        self.height = height;
        self
    }

    /// Set speed preset (0-10).
    pub fn speed(mut self, speed: usize) -> Self {
        self.speed = speed.min(10);
        self
    }

    /// Set quantizer (0-255).
    pub fn quantizer(mut self, quantizer: usize) -> Self {
        self.quantizer = quantizer.min(255);
        self
    }

    /// Set target bitrate in bits per second.
    pub fn bitrate(mut self, bitrate: usize) -> Self {
        self.bitrate = bitrate;
        self
    }

    /// Set frame rate.
    pub fn framerate(mut self, fps: u64) -> Self {
        self.timebase_num = 1;
        self.timebase_den = fps;
        self
    }

    /// Set fractional frame rate.
    pub fn framerate_rational(mut self, num: u64, den: u64) -> Self {
        self.timebase_num = num;
        self.timebase_den = den;
        self
    }

    /// Set bit depth (8 or 10).
    pub fn bit_depth(mut self, depth: usize) -> Self {
        self.bit_depth = if depth >= 10 { 10 } else { 8 };
        self
    }
}

/// AV1 software encoder using rav1e.
///
/// rav1e is an AV1 encoder written in Rust, known for its safety and quality.
///
/// # Input
///
/// Expects raw video frames in I420 format.
///
/// # Output
///
/// Produces AV1 OBU bitstream packets.
///
/// # Example
///
/// ```rust,ignore
/// let config = Rav1eConfig::default()
///     .dimensions(1920, 1080)
///     .speed(6)
///     .bitrate(5_000_000);
///
/// let encoder = Rav1eEncoder::new(config)?;
/// pipeline.add_node("av1enc", DynAsyncElement::new_box(ElementAdapter::new(encoder)));
/// ```
pub struct Rav1eEncoder {
    context: rav1e::Context<u8>,
    config: Rav1eConfig,
    frame_count: u64,
}

impl Rav1eEncoder {
    /// Create a new rav1e encoder.
    pub fn new(config: Rav1eConfig) -> Result<Self> {
        let enc_config = Self::build_config(&config)?;
        let context = enc_config
            .new_context()
            .map_err(|e| Error::Config(format!("Failed to create rav1e context: {:?}", e)))?;

        Ok(Self {
            context,
            config,
            frame_count: 0,
        })
    }

    fn build_config(config: &Rav1eConfig) -> Result<rav1e::Config> {
        let mut enc = rav1e::EncoderConfig::default();

        enc.width = config.width;
        enc.height = config.height;
        enc.speed_settings = rav1e::config::SpeedSettings::from_preset(config.speed as u8);
        enc.quantizer = config.quantizer;
        enc.bitrate = config.bitrate as i32;
        enc.time_base = rav1e::data::Rational::new(config.timebase_num, config.timebase_den);

        let cfg = rav1e::Config::new()
            .with_encoder_config(enc)
            .with_threads(0); // Auto-detect thread count

        Ok(cfg)
    }

    /// Get the configuration.
    pub fn config(&self) -> &Rav1eConfig {
        &self.config
    }

    /// Get the number of frames encoded.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Encode a frame from raw I420 data.
    fn encode_frame(&mut self, input: &[u8], _pts: u64) -> Result<Option<Vec<u8>>> {
        // Create rav1e frame
        let mut frame = self.context.new_frame();

        // Calculate plane sizes
        let width = self.config.width;
        let height = self.config.height;
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);

        if input.len() < y_size + 2 * uv_size {
            return Err(Error::InvalidSegment(format!(
                "Input buffer too small: {} < {}",
                input.len(),
                y_size + 2 * uv_size
            )));
        }

        // Copy Y plane
        for y in 0..height {
            let src_offset = y * width;
            let dst_row = &mut frame.planes[0].rows_iter_mut().nth(y).unwrap();
            dst_row[..width].copy_from_slice(&input[src_offset..src_offset + width]);
        }

        // Copy U plane
        let u_start = y_size;
        let uv_width = width / 2;
        let uv_height = height / 2;
        for y in 0..uv_height {
            let src_offset = u_start + y * uv_width;
            let dst_row = &mut frame.planes[1].rows_iter_mut().nth(y).unwrap();
            dst_row[..uv_width].copy_from_slice(&input[src_offset..src_offset + uv_width]);
        }

        // Copy V plane
        let v_start = y_size + uv_size;
        for y in 0..uv_height {
            let src_offset = v_start + y * uv_width;
            let dst_row = &mut frame.planes[2].rows_iter_mut().nth(y).unwrap();
            dst_row[..uv_width].copy_from_slice(&input[src_offset..src_offset + uv_width]);
        }

        // Send frame to encoder
        self.context
            .send_frame(frame)
            .map_err(|e| Error::InvalidSegment(format!("rav1e send_frame failed: {:?}", e)))?;

        // Try to receive encoded packet
        match self.context.receive_packet() {
            Ok(packet) => {
                self.frame_count += 1;
                Ok(Some(packet.data))
            }
            Err(rav1e::EncoderStatus::NeedMoreData) => Ok(None),
            Err(rav1e::EncoderStatus::Encoded) => {
                // Frame was encoded but no packet ready yet
                Ok(None)
            }
            Err(e) => Err(Error::InvalidSegment(format!(
                "rav1e encode failed: {:?}",
                e
            ))),
        }
    }

    /// Flush remaining frames from the encoder (internal implementation).
    fn flush_internal(&mut self) -> Result<Vec<Vec<u8>>> {
        self.context.flush();

        let mut packets = Vec::new();
        loop {
            match self.context.receive_packet() {
                Ok(packet) => {
                    self.frame_count += 1;
                    packets.push(packet.data);
                }
                Err(rav1e::EncoderStatus::LimitReached) => break,
                Err(rav1e::EncoderStatus::Encoded) => continue,
                Err(rav1e::EncoderStatus::NeedMoreData) => continue,
                Err(e) => {
                    return Err(Error::InvalidSegment(format!(
                        "rav1e flush failed: {:?}",
                        e
                    )));
                }
            }
        }
        Ok(packets)
    }
}

impl Element for Rav1eEncoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input = buffer.as_bytes();
        let pts = buffer.metadata().pts.nanos();

        match self.encode_frame(input, pts)? {
            Some(packet) => {
                // Create output buffer with encoded data
                let segment = Arc::new(CpuSegment::new(packet.len())?);
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        packet.as_ptr(),
                        segment.as_mut_ptr().unwrap(),
                        packet.len(),
                    );
                }

                let metadata = buffer.metadata().clone();
                // Note: codec info could be added via MediaFormat if needed

                Ok(Some(Buffer::new(
                    MemoryHandle::from_segment(segment),
                    metadata,
                )))
            }
            None => Ok(None), // Encoder buffering, no output yet
        }
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::cpu_intensive() // Pure Rust, memory-safe, CPU intensive
    }
}

impl Drop for Rav1eEncoder {
    fn drop(&mut self) {
        // Flush is called automatically, but we ignore remaining packets
        let _ = self.flush_internal();
    }
}

impl VideoEncoder for Rav1eEncoder {
    type Packet = Vec<u8>;

    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Self::Packet>> {
        // Validate frame format
        if frame.format != PixelFormat::I420 {
            return Err(Error::InvalidSegment(format!(
                "Rav1eEncoder only supports I420, got {:?}",
                frame.format
            )));
        }

        // Validate dimensions match config
        if frame.width as usize != self.config.width || frame.height as usize != self.config.height
        {
            return Err(Error::InvalidSegment(format!(
                "Frame dimensions {}x{} don't match encoder config {}x{}",
                frame.width, frame.height, self.config.width, self.config.height
            )));
        }

        // Encode the frame
        match self.encode_frame(&frame.data, frame.pts as u64)? {
            Some(packet) => Ok(vec![packet]),
            None => Ok(vec![]), // Encoder buffering
        }
    }

    fn flush(&mut self) -> Result<Vec<Self::Packet>> {
        self.flush_internal()
    }

    fn has_pending(&self) -> bool {
        // rav1e may have frames in lookahead buffer
        // We can't easily query this, so assume true if we've sent any frames
        self.frame_count > 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rav1e_config_builder() {
        let config = Rav1eConfig::default()
            .dimensions(1280, 720)
            .speed(8)
            .quantizer(150)
            .framerate(60);

        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.speed, 8);
        assert_eq!(config.quantizer, 150);
        assert_eq!(config.timebase_den, 60);
    }

    #[test]
    fn test_rav1e_config_clamp() {
        let config = Rav1eConfig::default().speed(100).quantizer(500);

        assert_eq!(config.speed, 10); // Clamped to max
        assert_eq!(config.quantizer, 255); // Clamped to max
    }
}
