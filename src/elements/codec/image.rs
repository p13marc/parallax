//! Image codec elements using pure Rust implementations.
//!
//! This module provides image encoding and decoding elements.
//!
//! # Supported Formats
//!
//! | Format | Feature Flag | Decoder | Encoder |
//! |--------|--------------|---------|---------|
//! | JPEG | `image-jpeg` | Yes | Yes |
//! | PNG | `image-png` | Yes | Yes |
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{JpegDecoder, JpegEncoder, PngDecoder, PngEncoder};
//!
//! // Decode JPEG
//! let decoder = JpegDecoder::new();
//!
//! // Encode to JPEG (e.g. a low-fps preview branch)
//! let encoder = JpegEncoder::new(640, 480, ColorType::Rgb).with_quality(80);
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
use crate::memory::SharedArena;

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
        arena: Option<SharedArena>,
    }

    impl JpegDecoder {
        /// Create a new JPEG decoder.
        pub fn new() -> Self {
            Self {
                frame_count: 0,
                arena: None,
            }
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
            let mut decoder = ZuneJpegDecoder::new(std::io::Cursor::new(input));

            // Decode header to get dimensions
            decoder.decode_headers().map_err(|e| {
                Error::InvalidSegment(format!("JPEG header decode failed: {:?}", e))
            })?;

            let info = decoder
                .info()
                .ok_or_else(|| Error::InvalidSegment("Failed to get JPEG info".to_string()))?;

            let width = info.width as u32;
            let height = info.height as u32;

            // Decode pixels
            let pixels = decoder
                .decode()
                .map_err(|e| Error::InvalidSegment(format!("JPEG decode failed: {:?}", e)))?;

            let pixel_format = match info.components {
                1 => crate::format::PixelFormat::Gray8,
                4 => crate::format::PixelFormat::Rgba,
                _ => crate::format::PixelFormat::Rgb24,
            };

            // Lazily initialize arena
            if self.arena.is_none() {
                self.arena = Some(
                    SharedArena::new(pixels.len().max(1024 * 1024), 16)
                        .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
                );
            }
            let arena = self.arena.as_mut().unwrap();

            arena.reclaim();

            let mut slot = arena
                .acquire()
                .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
            slot.data_mut()[..pixels.len()].copy_from_slice(&pixels);

            self.frame_count += 1;

            // Propagate input metadata and describe the decoded frame so
            // downstream elements know its dimensions and pixel layout.
            let mut metadata = buffer.metadata().clone();
            metadata.format = Some(crate::format::MediaFormat::VideoRaw(
                crate::format::VideoFormat {
                    width,
                    height,
                    pixel_format,
                    framerate: crate::format::Framerate::default(),
                },
            ));

            Ok(Some(Buffer::new(
                MemoryHandle::with_len(slot, pixels.len()),
                metadata,
            )))
        }

        fn execution_hints(&self) -> ExecutionHints {
            ExecutionHints::cpu_intensive()
        }
    }

    /// JPEG encoder using the jpeg-encoder crate (pure Rust, SIMD on x86).
    ///
    /// Encodes raw interleaved pixel data (RGB/RGBA/grayscale) to JPEG —
    /// e.g. for a low-fps preview branch published over the network. Camera
    /// YUV output (I420/NV12/YUYV) must be converted upstream first (see
    /// `VideoConvert`).
    pub struct JpegEncoder {
        width: u32,
        height: u32,
        color_type: ColorType,
        /// Shared with [`JpegQualityControl`] handles so a running pipeline can
        /// change it.
        quality: std::sync::Arc<std::sync::atomic::AtomicU8>,
        frame_count: u64,
        arena: Option<SharedArena>,
    }

    /// Cloneable handle to change a running [`JpegEncoder`]'s quality.
    ///
    /// A preview branch is a real per-viewer cost — at 640x480 and quality 75,
    /// even 2 fps is on the order of 50-100 kB/s each — and quality is the
    /// cheapest knob on it.
    ///
    /// Like the other runtime control handles, clone it from the element
    /// **before** `executor.start()`.
    #[derive(Clone, Debug)]
    pub struct JpegQualityControl(std::sync::Arc<std::sync::atomic::AtomicU8>);

    impl JpegQualityControl {
        /// Set the encoding quality (clamped to 1-100).
        pub fn set_quality(&self, quality: u8) {
            self.0
                .store(quality.clamp(1, 100), std::sync::atomic::Ordering::Release);
        }

        /// The current encoding quality.
        pub fn quality(&self) -> u8 {
            self.0.load(std::sync::atomic::Ordering::Acquire)
        }
    }

    impl JpegEncoder {
        /// Create a new JPEG encoder.
        ///
        /// # Arguments
        /// * `width` - Image width in pixels (max 65535)
        /// * `height` - Image height in pixels (max 65535)
        /// * `color_type` - Input pixel layout (Gray, Rgb, Rgba; alpha is
        ///   ignored during encoding)
        pub fn new(width: u32, height: u32, color_type: ColorType) -> Self {
            Self {
                width,
                height,
                color_type,
                quality: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(80)),
                frame_count: 0,
                arena: None,
            }
        }

        /// Set the encoding quality (1-100, default 80). Higher = better
        /// quality, larger output.
        pub fn with_quality(self, quality: u8) -> Self {
            JpegQualityControl(std::sync::Arc::clone(&self.quality)).set_quality(quality);
            self
        }

        /// The current encoding quality (1-100).
        pub fn quality(&self) -> u8 {
            self.quality.load(std::sync::atomic::Ordering::Relaxed)
        }

        /// Get the number of frames encoded.
        pub fn frame_count(&self) -> u64 {
            self.frame_count
        }

        fn to_jpeg_color_type(&self) -> Result<jpeg_encoder::ColorType> {
            Ok(match self.color_type {
                ColorType::Gray => jpeg_encoder::ColorType::Luma,
                ColorType::Rgb => jpeg_encoder::ColorType::Rgb,
                ColorType::Rgba => jpeg_encoder::ColorType::Rgba,
                ColorType::GrayAlpha => {
                    return Err(Error::InvalidCaps(
                        "JPEG cannot encode gray+alpha input".to_string(),
                    ));
                }
            })
        }
    }

    impl crate::control::Controllable for JpegEncoder {
        type Control = JpegQualityControl;

        /// A handle for changing the JPEG quality on a running pipeline.
        ///
        /// Clone it *before* `executor.start()` — see [`crate::control`].
        fn control(&self) -> JpegQualityControl {
            JpegQualityControl(std::sync::Arc::clone(&self.quality))
        }
    }

    impl Element for JpegEncoder {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            let input = buffer.as_bytes();

            if self.width > u16::MAX as u32 || self.height > u16::MAX as u32 {
                return Err(Error::InvalidCaps(format!(
                    "JPEG dimensions limited to 65535, got {}x{}",
                    self.width, self.height
                )));
            }

            let expected_size =
                self.width as usize * self.height as usize * self.color_type.bytes_per_pixel();
            if input.len() < expected_size {
                return Err(Error::InvalidSegment(format!(
                    "Input buffer too small: {} < {}",
                    input.len(),
                    expected_size
                )));
            }

            let mut output = Vec::new();
            // Read per frame: a JpegQualityControl handle may have changed it
            // while the pipeline runs.
            let encoder = jpeg_encoder::Encoder::new(&mut output, self.quality());
            encoder
                .encode(
                    &input[..expected_size],
                    self.width as u16,
                    self.height as u16,
                    self.to_jpeg_color_type()?,
                )
                .map_err(|e| Error::InvalidSegment(format!("JPEG encode failed: {:?}", e)))?;

            // Lazily initialize arena
            if self.arena.is_none() {
                self.arena = Some(
                    SharedArena::new(output.len().max(1024 * 1024), 16)
                        .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
                );
            }
            let arena = self.arena.as_mut().unwrap();

            arena.reclaim();

            let mut slot = arena
                .acquire()
                .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
            slot.data_mut()[..output.len()].copy_from_slice(&output);

            self.frame_count += 1;

            // Propagate metadata (PTS!); every JPEG is independently decodable.
            let mut metadata = buffer.metadata().clone();
            metadata.flags |= crate::metadata::BufferFlags::SYNC_POINT;
            Ok(Some(Buffer::new(
                MemoryHandle::with_len(slot, output.len()),
                metadata,
            )))
        }

        fn execution_hints(&self) -> ExecutionHints {
            ExecutionHints::cpu_intensive()
        }
    }
}

#[cfg(feature = "image-jpeg")]
pub use jpeg_codec::{JpegDecoder, JpegEncoder, JpegQualityControl};

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
        arena: Option<SharedArena>,
    }

    impl PngDecoder {
        /// Create a new PNG decoder.
        pub fn new() -> Self {
            Self {
                frame_count: 0,
                arena: None,
            }
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

            let buffer_size = reader.output_buffer_size().ok_or_else(|| {
                Error::InvalidSegment("PNG output buffer size overflow".to_string())
            })?;
            let mut pixels = vec![0u8; buffer_size];
            let info = reader
                .next_frame(&mut pixels)
                .map_err(|e| Error::InvalidSegment(format!("PNG decode failed: {:?}", e)))?;

            // Truncate to actual size
            pixels.truncate(info.buffer_size());

            // Lazily initialize arena
            if self.arena.is_none() {
                self.arena = Some(
                    SharedArena::new(pixels.len().max(1024 * 1024), 16)
                        .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
                );
            }
            let arena = self.arena.as_mut().unwrap();

            arena.reclaim();

            let mut slot = arena
                .acquire()
                .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
            slot.data_mut()[..pixels.len()].copy_from_slice(&pixels);

            self.frame_count += 1;

            let metadata = buffer.metadata().clone();
            Ok(Some(Buffer::new(
                MemoryHandle::with_len(slot, pixels.len()),
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
        arena: Option<SharedArena>,
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
                arena: None,
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

            // Lazily initialize arena
            if self.arena.is_none() {
                // PNG compression varies; allocate generous size
                self.arena = Some(
                    SharedArena::new(output.len().max(1024 * 1024), 16)
                        .map_err(|e| Error::Element(format!("Failed to create arena: {}", e)))?,
                );
            }
            let arena = self.arena.as_mut().unwrap();

            arena.reclaim();

            let mut slot = arena
                .acquire()
                .ok_or_else(|| Error::Element("Failed to acquire buffer slot".to_string()))?;
            slot.data_mut()[..output.len()].copy_from_slice(&output);

            self.frame_count += 1;

            let metadata = buffer.metadata().clone();
            Ok(Some(Buffer::new(
                MemoryHandle::with_len(slot, output.len()),
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
    #[cfg(feature = "image-jpeg")]
    use crate::control::Controllable;

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

    #[cfg(feature = "image-jpeg")]
    mod jpeg {
        use super::*;
        use crate::buffer::{Buffer, MemoryHandle};
        use crate::format::{MediaFormat, PixelFormat};
        use crate::memory::SharedArena;
        use crate::metadata::{BufferFlags, Metadata};

        const W: u32 = 32;
        const H: u32 = 16;

        /// A smooth RGB gradient (JPEG-friendly so round-trips are close).
        fn gradient_rgb() -> Vec<u8> {
            let mut data = Vec::with_capacity((W * H * 3) as usize);
            for y in 0..H {
                for x in 0..W {
                    data.push((x * 8) as u8);
                    data.push((y * 16) as u8);
                    data.push(128);
                }
            }
            data
        }

        fn rgb_buffer(data: &[u8]) -> Buffer {
            let arena = SharedArena::new(data.len().max(1024), 8).unwrap();
            let mut slot = arena.acquire().unwrap();
            slot.data_mut()[..data.len()].copy_from_slice(data);
            let mut metadata = Metadata::from_sequence(7);
            metadata.pts = crate::clock::ClockTime::from_millis(42);
            Buffer::new(MemoryHandle::with_len(slot, data.len()), metadata)
        }

        /// Preview quality is a live knob: a viewer on a thin link asks for a
        /// cheaper preview and gets one without the pipeline restarting.
        #[test]
        fn quality_change_reaches_a_running_encoder() {
            let original = gradient_rgb();
            let mut encoder = JpegEncoder::new(W, H, ColorType::Rgb).with_quality(95);
            let control = encoder.control();

            let high = encoder
                .process(rgb_buffer(&original))
                .unwrap()
                .expect("encoder output");
            let high_len = high.as_bytes().len();

            control.set_quality(15);
            let low = encoder
                .process(rgb_buffer(&original))
                .unwrap()
                .expect("encoder output");
            let low_len = low.as_bytes().len();

            assert!(
                low_len < high_len,
                "quality 15 must produce a smaller JPEG than quality 95 \
                 (got {low_len} vs {high_len} bytes)"
            );

            // Both must still be valid JPEGs a viewer can decode.
            let mut decoder = JpegDecoder::new();
            for encoded in [high, low] {
                assert_eq!(&encoded.as_bytes()[..2], &[0xFF, 0xD8], "JPEG SOI marker");
                assert!(decoder.process(encoded).unwrap().is_some());
            }
        }

        #[test]
        fn quality_is_clamped_on_every_path() {
            let encoder = JpegEncoder::new(W, H, ColorType::Rgb).with_quality(200);
            assert_eq!(encoder.quality(), 100);

            let control = encoder.control();
            control.set_quality(0);
            assert_eq!(control.quality(), 1, "0 would be a library-level surprise");
            assert_eq!(encoder.quality(), 1, "handle and element share state");
        }

        #[test]
        fn encode_decode_roundtrip() {
            let original = gradient_rgb();
            let mut encoder = JpegEncoder::new(W, H, ColorType::Rgb).with_quality(95);
            let encoded = encoder
                .process(rgb_buffer(&original))
                .unwrap()
                .expect("encoder output");

            // Metadata propagated, JPEG flagged as an independent sync point.
            assert_eq!(encoded.metadata().sequence, 7);
            assert_eq!(
                encoded.metadata().pts,
                crate::clock::ClockTime::from_millis(42)
            );
            assert!(encoded.metadata().flags.contains(BufferFlags::SYNC_POINT));
            assert_eq!(&encoded.as_bytes()[..2], &[0xFF, 0xD8], "JPEG SOI marker");

            // Decode back: dimensions + format in metadata, pixels close.
            let mut decoder = JpegDecoder::new();
            let decoded = decoder.process(encoded).unwrap().expect("decoder output");
            match decoded.metadata().format {
                Some(MediaFormat::VideoRaw(vf)) => {
                    assert_eq!((vf.width, vf.height), (W, H));
                    assert_eq!(vf.pixel_format, PixelFormat::Rgb24);
                }
                ref other => panic!("expected VideoRaw format metadata, got {other:?}"),
            }
            let pixels = decoded.as_bytes();
            assert_eq!(pixels.len(), original.len());
            let max_diff = pixels
                .iter()
                .zip(&original)
                .map(|(a, b)| a.abs_diff(*b))
                .max()
                .unwrap();
            assert!(max_diff < 24, "lossy but close (max diff {max_diff})");
        }

        #[test]
        fn quality_orders_output_size() {
            let data = gradient_rgb();
            let size_at = |quality: u8| -> usize {
                let mut encoder = JpegEncoder::new(W, H, ColorType::Rgb).with_quality(quality);
                encoder.process(rgb_buffer(&data)).unwrap().unwrap().len()
            };
            let high = size_at(95);
            let low = size_at(10);
            assert!(
                high > low,
                "higher quality must produce more bytes: q95={high} q10={low}"
            );
        }

        #[test]
        fn rejects_undersized_input() {
            let mut encoder = JpegEncoder::new(W, H, ColorType::Rgb);
            let result = encoder.process(rgb_buffer(&[0u8; 16]));
            assert!(result.is_err());
        }

        #[test]
        fn rejects_gray_alpha() {
            let mut encoder = JpegEncoder::new(W, H, ColorType::GrayAlpha);
            let data = vec![0u8; (W * H * 2) as usize];
            assert!(encoder.process(rgb_buffer(&data)).is_err());
        }
    }
}
