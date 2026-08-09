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
//! let encoder = JpegEncoder::new().with_quality(80);
//!
//! // Decode PNG
//! let decoder = PngDecoder::new();
//!
//! // Encode to PNG
//! let encoder = PngEncoder::new();
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{Element, ExecutionHints};
use crate::error::{Error, Result};
use crate::memory::{OutputArena, OutputBudget, defaults};

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

/// Resolve what an image encoder is being handed, from the buffer itself.
///
/// Geometry travels in-band, so both the size and the pixel layout come from the
/// same [`Metadata`] as the bytes and cannot disagree with them. A `hint` (from
/// `with_color_type`) covers hand-built buffers that declare no pixel format; a
/// buffer that *does* declare one always wins.
///
/// Errors rather than falling back to a stale constructor value. That fallback
/// is precisely what made a scaler upstream of an image encoder either throw
/// "Input buffer too small" or — worse — silently encode the top-left corner of
/// a larger frame.
#[cfg(any(feature = "image-jpeg", feature = "image-png"))]
fn resolve_image_input(
    metadata: &crate::metadata::Metadata,
    hint: Option<ColorType>,
    who: &str,
) -> Result<(u32, u32, ColorType)> {
    use crate::format::PixelFormat;

    let (width, height) = metadata.video_dims().ok_or_else(|| {
        Error::InvalidCaps(format!(
            "{who}: buffer carries no video dimensions — the upstream element must call \
             Metadata::set_video_dims()"
        ))
    })?;

    // A declared format is authoritative. Only interleaved 8-bit layouts can be
    // encoded; a YUV frame here means someone forgot a VideoConvertElement, and
    // encoding its planes as if they were RGB is a bug we used to commit
    // silently.
    let color_type = match metadata.video_pixel_format() {
        Some(PixelFormat::Gray8) => ColorType::Gray,
        Some(PixelFormat::Rgb24) => ColorType::Rgb,
        Some(PixelFormat::Rgba) => ColorType::Rgba,
        Some(other) => {
            return Err(Error::InvalidCaps(format!(
                "{who} cannot encode {other:?} — insert a VideoConvertElement upstream to reach \
                 Rgb24, Rgba or Gray8"
            )));
        }
        None => hint.ok_or_else(|| {
            Error::InvalidCaps(format!(
                "{who}: buffer is {width}x{height} but declares no pixel format, and no \
                 with_color_type() hint was given"
            ))
        })?,
    };

    Ok((width, height, color_type))
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
        output: OutputArena,
    }

    impl JpegDecoder {
        /// The output arena, sized by the executor at start.
        ///
        /// A 1 MiB floor on the slot: compressed frame sizes vary, and the
        /// slot size is fixed when the arena is built from the first one.
        fn new_output_arena() -> OutputArena {
            OutputArena::new(defaults::VIDEO_DECODER_SLOT_COUNT).with_min_slot_size(1024 * 1024)
        }

        /// Create a new JPEG decoder.
        pub fn new() -> Self {
            Self {
                frame_count: 0,
                output: Self::new_output_arena(),
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
        fn set_output_budget(&mut self, budget: OutputBudget) {
            self.output.set_budget(budget);
        }

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

            let mut slot = self.output.acquire(pixels.len(), "jpegdecoder")?;
            slot.data_mut()[..pixels.len()].copy_from_slice(&pixels);

            self.frame_count += 1;

            // Propagate input metadata and describe the decoded frame so
            // downstream elements know its dimensions and pixel layout.
            // `set_video_dims` rather than writing `format` directly: it also
            // writes the legacy "width"/"height" keys, and readers are split
            // across both conventions.
            let mut metadata = buffer.metadata().clone();
            metadata.set_video_dims(width, height, pixel_format);

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
    ///
    /// Takes **no dimensions**: geometry travels in-band, in [`Metadata`](crate::metadata::Metadata), and
    /// the encoder reads it from each buffer. A colour-type *hint* is available
    /// via [`with_color_type`](Self::with_color_type) for buffers that carry no
    /// pixel format, but a buffer that declares one wins.
    pub struct JpegEncoder {
        /// Colour-type hint, used only when the buffer declares no pixel format.
        color_type: Option<ColorType>,
        /// Shared with [`JpegQualityControl`] handles so a running pipeline can
        /// change it.
        quality: std::sync::Arc<std::sync::atomic::AtomicU8>,
        frame_count: u64,
        output: OutputArena,
        /// Counters readable while the pipeline runs (shared with [`Self::stats`]).
        stats: crate::control::EncoderStatsHandle,
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
        /// The output arena, sized by the executor at start.
        ///
        /// A 1 MiB floor on the slot: compressed frame sizes vary, and the
        /// slot size is fixed when the arena is built from the first one.
        fn new_output_arena() -> OutputArena {
            OutputArena::new(defaults::VIDEO_ENCODER_SLOT_COUNT).with_min_slot_size(1024 * 1024)
        }

        /// Create a new JPEG encoder.
        ///
        /// Geometry and pixel layout come from each buffer's [`Metadata`](crate::metadata::Metadata).
        pub fn new() -> Self {
            Self {
                color_type: None,
                quality: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(80)),
                frame_count: 0,
                output: Self::new_output_arena(),
                stats: crate::control::EncoderStatsHandle::default(),
            }
        }

        /// Hint the input pixel layout, for buffers that declare no pixel format.
        ///
        /// A buffer that *does* declare one always wins — this is a fallback for
        /// hand-built buffers, not an override.
        pub fn with_color_type(mut self, color_type: ColorType) -> Self {
            self.color_type = Some(color_type);
            self
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

        /// A cloneable handle to this encoder's counters.
        ///
        /// Clone it *before* `executor.start()`: the element is moved into its
        /// executor task there, so [`frame_count`](Self::frame_count) can never
        /// be read while it is actually encoding. This handle can.
        ///
        /// JPEG has no rate control, so `frames_dropped_by_rc` stays zero;
        /// `frames_encoded`, `bytes_encoded` and `last_encode_ns` are live.
        pub fn stats(&self) -> crate::control::EncoderStatsHandle {
            self.stats.clone()
        }

        fn to_jpeg_color_type(color_type: ColorType) -> Result<jpeg_encoder::ColorType> {
            Ok(match color_type {
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

    impl Default for JpegEncoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Element for JpegEncoder {
        fn set_output_budget(&mut self, budget: OutputBudget) {
            self.output.set_budget(budget);
        }

        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            // Admission control before the encode: if downstream is still
            // holding every output slot, spending CPU on a frame with
            // nowhere to go helps nobody. The executor sheds it and the
            // next frame proceeds normally.
            self.output.admit()?;

            let input = buffer.as_bytes();

            // Geometry travels in-band. Both the size *and* the pixel layout come
            // from the buffer that carries the bytes, so they can never disagree.
            let (width, height, color_type) =
                resolve_image_input(buffer.metadata(), self.color_type, "JpegEncoder")?;

            if width > u16::MAX as u32 || height > u16::MAX as u32 {
                return Err(Error::InvalidCaps(format!(
                    "JPEG dimensions limited to 65535, got {width}x{height}"
                )));
            }

            let expected_size = width as usize * height as usize * color_type.bytes_per_pixel();
            if input.len() < expected_size {
                return Err(Error::InvalidSegment(format!(
                    "Input buffer too small: {} < {} ({width}x{height} {color_type:?})",
                    input.len(),
                    expected_size
                )));
            }

            let mut output = Vec::new();
            // Read per frame: a JpegQualityControl handle may have changed it
            // while the pipeline runs.
            let started = std::time::Instant::now();
            let encoder = jpeg_encoder::Encoder::new(&mut output, self.quality());
            encoder
                .encode(
                    &input[..expected_size],
                    width as u16,
                    height as u16,
                    Self::to_jpeg_color_type(color_type)?,
                )
                .map_err(|e| Error::InvalidSegment(format!("JPEG encode failed: {:?}", e)))?;
            // A preview branch is a real per-viewer cost, and quality is the
            // knob operators reach for — so bytes-per-frame and encode time
            // need to be readable while the pipeline runs, not just after.
            // JPEG has no rate control, so frames_dropped_by_rc stays zero.
            self.stats
                .record_frame(output.len(), started.elapsed().as_nanos() as u64);

            let mut slot = self.output.acquire(output.len(), "jpegencoder")?;
            slot.data_mut()[..output.len()].copy_from_slice(&output);

            self.frame_count += 1;

            // Propagate metadata (PTS!); every JPEG is independently decodable.
            let mut metadata = buffer.metadata().clone();
            metadata.flags |= crate::metadata::BufferFlags::SYNC_POINT;
            // These bytes are a JPEG, not raw video. Carrying the input's
            // VideoRaw format forward would tell downstream elements they can
            // index into planes that are no longer there.
            metadata.format = None;
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
        output: OutputArena,
    }

    impl PngDecoder {
        /// The output arena, sized by the executor at start.
        ///
        /// A 1 MiB floor on the slot: compressed frame sizes vary, and the
        /// slot size is fixed when the arena is built from the first one.
        fn new_output_arena() -> OutputArena {
            OutputArena::new(defaults::VIDEO_DECODER_SLOT_COUNT).with_min_slot_size(1024 * 1024)
        }

        /// Create a new PNG decoder.
        pub fn new() -> Self {
            Self {
                frame_count: 0,
                output: Self::new_output_arena(),
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
        fn set_output_budget(&mut self, budget: OutputBudget) {
            self.output.set_budget(budget);
        }

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

            let mut slot = self.output.acquire(pixels.len(), "pngdecoder")?;
            slot.data_mut()[..pixels.len()].copy_from_slice(&pixels);

            self.frame_count += 1;

            // Geometry travels in-band: describe what we decoded, or refuse to
            // emit a buffer nobody downstream can interpret.
            let pixel_format = png_pixel_format(info.color_type, info.bit_depth)?;
            let mut metadata = buffer.metadata().clone();
            metadata.set_video_dims(info.width, info.height, pixel_format);

            Ok(Some(Buffer::new(
                MemoryHandle::with_len(slot, pixels.len()),
                metadata,
            )))
        }

        fn execution_hints(&self) -> ExecutionHints {
            ExecutionHints::cpu_intensive()
        }
    }

    /// Name a decoded PNG's pixel layout in the caps vocabulary.
    ///
    /// PNG can express layouts parallax has no `PixelFormat` for — 16-bit
    /// channels, grayscale+alpha, and palettes. Emitting such a buffer without
    /// a format leaves every downstream element to guess at its bytes, which is
    /// the class of bug the in-band-geometry invariant exists to kill. So we
    /// refuse it instead. (Widening this via `png::Transformations` to expand
    /// those into Rgb8/Gray8 would be a strict improvement — a good follow-up.)
    fn png_pixel_format(
        color_type: png::ColorType,
        bit_depth: png::BitDepth,
    ) -> Result<crate::format::PixelFormat> {
        use crate::format::PixelFormat as Pf;
        use png::{BitDepth, ColorType};

        match (color_type, bit_depth) {
            (ColorType::Grayscale, BitDepth::Eight) => Ok(Pf::Gray8),
            (ColorType::Rgb, BitDepth::Eight) => Ok(Pf::Rgb24),
            (ColorType::Rgba, BitDepth::Eight) => Ok(Pf::Rgba),
            _ => Err(Error::InvalidSegment(format!(
                "PNG is {color_type:?}/{bit_depth:?}, which has no raw-video pixel format \
                 parallax can describe (supported: 8-bit Grayscale, Rgb, Rgba)"
            ))),
        }
    }

    /// PNG encoder using the png crate (pure Rust).
    ///
    /// Encodes raw pixel data to PNG format.
    ///
    /// Takes **no dimensions**: geometry travels in-band, in [`Metadata`](crate::metadata::Metadata), and
    /// the encoder reads it from each buffer. A colour-type *hint* is available
    /// via [`with_color_type`](Self::with_color_type) for buffers that carry no
    /// pixel format.
    pub struct PngEncoder {
        /// Colour-type hint, used only when the buffer declares no pixel format.
        color_type: Option<ColorType>,
        frame_count: u64,
        output: OutputArena,
        /// Counters readable while the pipeline runs (shared with [`Self::stats`]).
        stats: crate::control::EncoderStatsHandle,
    }

    impl PngEncoder {
        /// The output arena, sized by the executor at start.
        ///
        /// A 1 MiB floor on the slot: compressed frame sizes vary, and the
        /// slot size is fixed when the arena is built from the first one.
        fn new_output_arena() -> OutputArena {
            OutputArena::new(defaults::VIDEO_ENCODER_SLOT_COUNT).with_min_slot_size(1024 * 1024)
        }

        /// Create a new PNG encoder.
        ///
        /// Geometry and pixel layout come from each buffer's [`Metadata`](crate::metadata::Metadata).
        pub fn new() -> Self {
            Self {
                color_type: None,
                frame_count: 0,
                output: Self::new_output_arena(),
                stats: crate::control::EncoderStatsHandle::default(),
            }
        }

        /// Hint the input pixel layout, for buffers that declare no pixel format.
        ///
        /// A buffer that *does* declare one always wins.
        pub fn with_color_type(mut self, color_type: ColorType) -> Self {
            self.color_type = Some(color_type);
            self
        }

        /// Get the number of frames encoded.
        pub fn frame_count(&self) -> u64 {
            self.frame_count
        }

        /// A cloneable handle to this encoder's counters.
        ///
        /// Clone it *before* `executor.start()`: the element is moved into its
        /// executor task there, so [`frame_count`](Self::frame_count) can never
        /// be read while it is actually encoding. This handle can.
        ///
        /// PNG has no rate control, so `frames_dropped_by_rc` stays zero;
        /// `frames_encoded`, `bytes_encoded` and `last_encode_ns` are live.
        pub fn stats(&self) -> crate::control::EncoderStatsHandle {
            self.stats.clone()
        }

        fn to_png_color_type(color_type: ColorType) -> png::ColorType {
            match color_type {
                ColorType::Gray => png::ColorType::Grayscale,
                ColorType::GrayAlpha => png::ColorType::GrayscaleAlpha,
                ColorType::Rgb => png::ColorType::Rgb,
                ColorType::Rgba => png::ColorType::Rgba,
            }
        }
    }

    impl Default for PngEncoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Element for PngEncoder {
        fn set_output_budget(&mut self, budget: OutputBudget) {
            self.output.set_budget(budget);
        }

        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            // Admission control before the encode: if downstream is still
            // holding every output slot, spending CPU on a frame with
            // nowhere to go helps nobody. The executor sheds it and the
            // next frame proceeds normally.
            self.output.admit()?;

            let input = buffer.as_bytes();

            // Geometry travels in-band: size and layout come from the same
            // metadata as the bytes, so they cannot disagree.
            let (width, height, color_type) =
                resolve_image_input(buffer.metadata(), self.color_type, "PngEncoder")?;

            // Expected input size
            let expected_size = width as usize * height as usize * color_type.bytes_per_pixel();
            if input.len() < expected_size {
                return Err(Error::InvalidSegment(format!(
                    "Input buffer too small: {} < {} ({width}x{height} {color_type:?})",
                    input.len(),
                    expected_size
                )));
            }

            // Encode to PNG
            let started = std::time::Instant::now();
            let mut output = Vec::new();
            {
                let mut encoder = png::Encoder::new(&mut output, width, height);
                encoder.set_color(Self::to_png_color_type(color_type));
                encoder.set_depth(png::BitDepth::Eight);

                let mut writer = encoder.write_header().map_err(|e| {
                    Error::InvalidSegment(format!("PNG header write failed: {:?}", e))
                })?;

                writer
                    .write_image_data(&input[..expected_size])
                    .map_err(|e| Error::InvalidSegment(format!("PNG encode failed: {:?}", e)))?;
            }
            // Same rationale as JpegEncoder: the element is moved into its
            // executor task at start, so per-frame cost has to be readable
            // through a handle. PNG has no rate control either.
            self.stats
                .record_frame(output.len(), started.elapsed().as_nanos() as u64);

            let mut slot = self.output.acquire(output.len(), "pngencoder")?;
            slot.data_mut()[..output.len()].copy_from_slice(&output);

            self.frame_count += 1;

            let mut metadata = buffer.metadata().clone();
            // These bytes are a PNG, not raw video.
            metadata.format = None;
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
            // Geometry travels in-band, so a test frame describes itself too.
            metadata.set_video_dims(W, H, crate::format::PixelFormat::Rgb24);
            Buffer::new(MemoryHandle::with_len(slot, data.len()), metadata)
        }

        /// #44: the preview branch's byte cost must be readable on a running
        /// pipeline, not only after it stops.
        ///
        /// `frame_count()` is `&self` on the element, and `Executor::start`
        /// moves elements into their tasks — so it can never be read while the
        /// encoder is encoding. The stats handle is cloned before start and
        /// can be.
        #[test]
        fn stats_track_frames_and_bytes() {
            let mut encoder = JpegEncoder::new();
            let stats = encoder.stats();

            assert_eq!(stats.snapshot().frames_encoded, 0);

            let data = gradient_rgb();
            let first = encoder
                .process(rgb_buffer(&data))
                .unwrap()
                .expect("encoder output");
            let first_len = first.as_bytes().len();

            let snap = stats.snapshot();
            assert_eq!(snap.frames_encoded, 1);
            assert_eq!(
                snap.bytes_encoded as usize, first_len,
                "bytes_encoded must match what actually came out"
            );
            assert!(snap.last_encode_ns > 0, "encode duration must be recorded");
            assert_eq!(
                snap.frames_dropped_by_rc, 0,
                "JPEG has no rate control, so this counter stays zero"
            );

            encoder.process(rgb_buffer(&data)).unwrap();
            let snap = stats.snapshot();
            assert_eq!(snap.frames_encoded, 2);
            assert!(snap.bytes_encoded as usize > first_len, "bytes accumulate");
        }

        /// Preview quality is a live knob: a viewer on a thin link asks for a
        /// cheaper preview and gets one without the pipeline restarting.
        #[test]
        fn quality_change_reaches_a_running_encoder() {
            let original = gradient_rgb();
            let mut encoder = JpegEncoder::new().with_quality(95);
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
            let encoder = JpegEncoder::new().with_quality(200);
            assert_eq!(encoder.quality(), 100);

            let control = encoder.control();
            control.set_quality(0);
            assert_eq!(control.quality(), 1, "0 would be a library-level surprise");
            assert_eq!(encoder.quality(), 1, "handle and element share state");
        }

        #[test]
        fn encode_decode_roundtrip() {
            let original = gradient_rgb();
            let mut encoder = JpegEncoder::new().with_quality(95);
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
                let mut encoder = JpegEncoder::new().with_quality(quality);
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
            let mut encoder = JpegEncoder::new();
            let result = encoder.process(rgb_buffer(&[0u8; 16]));
            assert!(result.is_err());
        }

        #[test]
        fn rejects_gray_alpha() {
            let mut encoder = JpegEncoder::new();
            let data = vec![0u8; (W * H * 2) as usize];
            assert!(encoder.process(rgb_buffer(&data)).is_err());
        }
    }
}
