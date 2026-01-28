//! Image codec elements using pure Rust implementations.
//!
//! This module provides image encoding and decoding elements.
//!
//! # Supported Formats
//!
//! | Format | Feature Flag | Decoder | Encoder |
//! |--------|--------------|---------|---------|
//! | JPEG | `image-jpeg` | Yes | No |
//! | PNG | `image-png` | Yes | Yes |
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{JpegDecoder, PngDecoder, PngEncoder};
//!
//! // Decode JPEG
//! let decoder = JpegDecoder::new();
//!
//! // Decode PNG
//! let decoder = PngDecoder::new();
//!
//! // Encode to PNG
//! let encoder = PngEncoder::new(1920, 1080, ColorType::Rgba);
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Element, ExecutionHints};
use crate::error::{Error, Result};
use crate::memory::{CpuSegment, MemorySegment};
use std::sync::Arc;

/// Color type for image data.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorType {
    /// Grayscale (1 channel)
    Gray,
    /// Grayscale with alpha (2 channels)
    GrayAlpha,
    /// RGB (3 channels)
    Rgb,
    /// RGBA (4 channels)
    Rgba,
}

impl ColorType {
    /// Number of channels.
    pub fn channels(&self) -> usize {
        match self {
            Self::Gray => 1,
            Self::GrayAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }

    /// Bytes per pixel (assuming 8-bit channels).
    pub fn bytes_per_pixel(&self) -> usize {
        self.channels()
    }
}

/// Decoded image frame.
#[derive(Clone, Debug)]
pub struct ImageFrame {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Color type.
    pub color_type: ColorType,
    /// Raw pixel data (row-major, top-to-bottom).
    pub data: Vec<u8>,
}

impl ImageFrame {
    /// Create a new image frame.
    pub fn new(width: u32, height: u32, color_type: ColorType) -> Self {
        let size = width as usize * height as usize * color_type.bytes_per_pixel();
        Self {
            width,
            height,
            color_type,
            data: vec![0u8; size],
        }
    }

    /// Calculate the row stride in bytes.
    pub fn stride(&self) -> usize {
        self.width as usize * self.color_type.bytes_per_pixel()
    }

    /// Get a row of pixels.
    pub fn row(&self, y: u32) -> &[u8] {
        let stride = self.stride();
        let start = y as usize * stride;
        &self.data[start..start + stride]
    }
}

// ============================================================================
// JPEG Decoder (using zune-jpeg)
// ============================================================================

#[cfg(feature = "image-jpeg")]
mod jpeg_codec {
    use super::*;
    use zune_jpeg::JpegDecoder as ZuneJpegDecoder;

    /// JPEG decoder using zune-jpeg (pure Rust).
    ///
    /// Decodes JPEG images to raw RGB pixel data.
    pub struct JpegDecoder {
        frame_count: u64,
    }

    impl JpegDecoder {
        /// Create a new JPEG decoder.
        pub fn new() -> Self {
            Self { frame_count: 0 }
        }

        /// Get the number of frames decoded.
        pub fn frame_count(&self) -> u64 {
            self.frame_count
        }
    }

    impl Default for JpegDecoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Element for JpegDecoder {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            let input = buffer.as_bytes();

            // Create decoder
            let mut decoder = ZuneJpegDecoder::new(input);

            // Decode header to get dimensions
            decoder.decode_headers().map_err(|e| {
                Error::InvalidSegment(format!("JPEG header decode failed: {:?}", e))
            })?;

            let info = decoder
                .info()
                .ok_or_else(|| Error::InvalidSegment("Failed to get JPEG info".to_string()))?;

            let _width = info.width as u32;
            let _height = info.height as u32;

            // Decode pixels
            let pixels = decoder
                .decode()
                .map_err(|e| Error::InvalidSegment(format!("JPEG decode failed: {:?}", e)))?;

            // Determine color type based on output (for future metadata use)
            let _color_type = match info.components {
                1 => ColorType::Gray,
                3 => ColorType::Rgb,
                4 => ColorType::Rgba,
                _ => ColorType::Rgb,
            };

            // Create output buffer
            let segment = Arc::new(CpuSegment::new(pixels.len())?);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pixels.as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    pixels.len(),
                );
            }

            self.frame_count += 1;

            let metadata = buffer.metadata().clone();
            // Could store width/height in metadata if needed

            Ok(Some(Buffer::new(
                MemoryHandle::from_segment(segment),
                metadata,
            )))
        }

        fn execution_hints(&self) -> ExecutionHints {
            ExecutionHints::cpu_intensive()
        }
    }
}

#[cfg(feature = "image-jpeg")]
pub use jpeg_codec::JpegDecoder;

// ============================================================================
// PNG Codec (using png crate)
// ============================================================================

#[cfg(feature = "image-png")]
mod png_codec {
    use super::*;

    /// PNG decoder using the png crate (pure Rust).
    ///
    /// Decodes PNG images to raw pixel data.
    pub struct PngDecoder {
        frame_count: u64,
    }

    impl PngDecoder {
        /// Create a new PNG decoder.
        pub fn new() -> Self {
            Self { frame_count: 0 }
        }

        /// Get the number of frames decoded.
        pub fn frame_count(&self) -> u64 {
            self.frame_count
        }
    }

    impl Default for PngDecoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Element for PngDecoder {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            let input = buffer.as_bytes();

            // Create decoder
            let decoder = png::Decoder::new(std::io::Cursor::new(input));
            let mut reader = decoder
                .read_info()
                .map_err(|e| Error::InvalidSegment(format!("PNG header decode failed: {:?}", e)))?;

            let mut pixels = vec![0u8; reader.output_buffer_size()];
            let info = reader
                .next_frame(&mut pixels)
                .map_err(|e| Error::InvalidSegment(format!("PNG decode failed: {:?}", e)))?;

            // Truncate to actual size
            pixels.truncate(info.buffer_size());

            // Create output buffer
            let segment = Arc::new(CpuSegment::new(pixels.len())?);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    pixels.as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    pixels.len(),
                );
            }

            self.frame_count += 1;

            let metadata = buffer.metadata().clone();
            Ok(Some(Buffer::new(
                MemoryHandle::from_segment(segment),
                metadata,
            )))
        }

        fn execution_hints(&self) -> ExecutionHints {
            ExecutionHints::cpu_intensive()
        }
    }

    /// PNG encoder using the png crate (pure Rust).
    ///
    /// Encodes raw pixel data to PNG format.
    pub struct PngEncoder {
        width: u32,
        height: u32,
        color_type: ColorType,
        frame_count: u64,
    }

    impl PngEncoder {
        /// Create a new PNG encoder.
        ///
        /// # Arguments
        /// * `width` - Image width in pixels
        /// * `height` - Image height in pixels
        /// * `color_type` - Color type (Gray, GrayAlpha, Rgb, Rgba)
        pub fn new(width: u32, height: u32, color_type: ColorType) -> Self {
            Self {
                width,
                height,
                color_type,
                frame_count: 0,
            }
        }

        /// Get the number of frames encoded.
        pub fn frame_count(&self) -> u64 {
            self.frame_count
        }

        fn to_png_color_type(&self) -> png::ColorType {
            match self.color_type {
                ColorType::Gray => png::ColorType::Grayscale,
                ColorType::GrayAlpha => png::ColorType::GrayscaleAlpha,
                ColorType::Rgb => png::ColorType::Rgb,
                ColorType::Rgba => png::ColorType::Rgba,
            }
        }
    }

    impl Element for PngEncoder {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            let input = buffer.as_bytes();

            // Expected input size
            let expected_size =
                self.width as usize * self.height as usize * self.color_type.bytes_per_pixel();
            if input.len() < expected_size {
                return Err(Error::InvalidSegment(format!(
                    "Input buffer too small: {} < {}",
                    input.len(),
                    expected_size
                )));
            }

            // Encode to PNG
            let mut output = Vec::new();
            {
                let mut encoder = png::Encoder::new(&mut output, self.width, self.height);
                encoder.set_color(self.to_png_color_type());
                encoder.set_depth(png::BitDepth::Eight);

                let mut writer = encoder.write_header().map_err(|e| {
                    Error::InvalidSegment(format!("PNG header write failed: {:?}", e))
                })?;

                writer
                    .write_image_data(&input[..expected_size])
                    .map_err(|e| Error::InvalidSegment(format!("PNG encode failed: {:?}", e)))?;
            }

            // Create output buffer
            let segment = Arc::new(CpuSegment::new(output.len())?);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    output.as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    output.len(),
                );
            }

            self.frame_count += 1;

            let metadata = buffer.metadata().clone();
            Ok(Some(Buffer::new(
                MemoryHandle::from_segment(segment),
                metadata,
            )))
        }

        fn execution_hints(&self) -> ExecutionHints {
            ExecutionHints::cpu_intensive()
        }
    }
}

#[cfg(feature = "image-png")]
pub use png_codec::{PngDecoder, PngEncoder};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_type_channels() {
        assert_eq!(ColorType::Gray.channels(), 1);
        assert_eq!(ColorType::GrayAlpha.channels(), 2);
        assert_eq!(ColorType::Rgb.channels(), 3);
        assert_eq!(ColorType::Rgba.channels(), 4);
    }

    #[test]
    fn test_image_frame_stride() {
        let frame = ImageFrame::new(100, 50, ColorType::Rgba);
        assert_eq!(frame.stride(), 400); // 100 * 4
        assert_eq!(frame.data.len(), 20000); // 100 * 50 * 4
    }
}
