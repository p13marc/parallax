//! Integration tests for media codec elements.
//!
//! These tests verify the codec elements work correctly with the pipeline system.

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;

/// Shared arena for test buffer allocation
fn test_arena() -> SharedArena {
    // Use thread_local to avoid creating too many arenas
    thread_local! {
        static ARENA: SharedArena = SharedArena::new(1024 * 1024, 64)
            .expect("failed to create test arena");
    }
    ARENA.with(|a| a.clone())
}

/// Test helper to create a buffer with given data
#[allow(dead_code)]
fn create_test_buffer(data: &[u8]) -> Buffer {
    let arena = test_arena();
    let mut slot = arena.acquire().expect("failed to acquire slot from arena");
    slot.data_mut()[..data.len()].copy_from_slice(data);
    let handle = MemoryHandle::with_len(slot, data.len());
    Buffer::new(handle, Metadata::from_sequence(0))
}

/// A raw-video buffer that describes its own geometry, the way every source and
/// decoder now does. Image encoders read their size and pixel layout from here.
#[allow(dead_code)]
fn create_video_buffer(
    data: &[u8],
    width: u32,
    height: u32,
    pixel_format: parallax::format::PixelFormat,
) -> Buffer {
    let arena = test_arena();
    let mut slot = arena.acquire().expect("failed to acquire slot from arena");
    slot.data_mut()[..data.len()].copy_from_slice(data);
    let handle = MemoryHandle::with_len(slot, data.len());

    let mut metadata = Metadata::from_sequence(0);
    metadata.set_video_dims(width, height, pixel_format);
    Buffer::new(handle, metadata)
}

// ============================================================================
// PNG Codec Tests
// ============================================================================

#[cfg(feature = "image-png")]
mod png_tests {
    use super::*;
    use parallax::element::Element;
    use parallax::elements::codec::{PngDecoder, PngEncoder};

    /// Create a simple 2x2 RGB test image
    fn create_rgb_image(width: u32, height: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                // Create a simple gradient pattern
                data.push((x * 255 / width.max(1)) as u8); // R
                data.push((y * 255 / height.max(1)) as u8); // G
                data.push(128); // B
            }
        }
        data
    }

    /// Create a simple grayscale test image
    fn create_gray_image(width: u32, height: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                data.push(((x + y) * 255 / (width + height).max(1)) as u8);
            }
        }
        data
    }

    #[test]
    fn test_png_roundtrip_rgb() {
        // Create a test RGB image
        let width = 16;
        let height = 16;
        let raw_image = create_rgb_image(width, height);

        // Encode to PNG
        let mut encoder = PngEncoder::new();
        let input_buffer = create_video_buffer(
            &raw_image,
            width,
            height,
            parallax::format::PixelFormat::Rgb24,
        );
        let encoded = encoder.process(input_buffer).unwrap().unwrap();

        // Verify PNG signature
        let encoded_data = encoded.as_bytes();
        assert!(encoded_data.len() > 8, "PNG should have at least header");
        assert_eq!(
            &encoded_data[..8],
            &[137, 80, 78, 71, 13, 10, 26, 10],
            "Invalid PNG signature"
        );

        // Decode the PNG
        let mut decoder = PngDecoder::new();
        let decoded = decoder.process(encoded).unwrap().unwrap();

        // Verify the decoded image matches original
        let decoded_data = decoded.as_bytes();
        assert_eq!(decoded_data.len(), raw_image.len(), "Decoded size mismatch");
        assert_eq!(decoded_data, raw_image.as_slice(), "Decoded data mismatch");
    }

    #[test]
    fn test_png_roundtrip_grayscale() {
        let width = 8;
        let height = 8;
        let raw_image = create_gray_image(width, height);

        // Encode to PNG
        let mut encoder = PngEncoder::new();
        let input_buffer = create_video_buffer(
            &raw_image,
            width,
            height,
            parallax::format::PixelFormat::Gray8,
        );
        let encoded = encoder.process(input_buffer).unwrap().unwrap();

        // Decode the PNG
        let mut decoder = PngDecoder::new();
        let decoded = decoder.process(encoded).unwrap().unwrap();

        // Verify the decoded image matches original
        assert_eq!(decoded.as_bytes().len(), raw_image.len());
        assert_eq!(decoded.as_bytes(), raw_image.as_slice());
    }

    #[test]
    fn test_png_roundtrip_rgba() {
        let width = 4;
        let height = 4;
        let mut raw_image = Vec::with_capacity((width * height * 4) as usize);
        for y in 0..height {
            for x in 0..width {
                raw_image.push((x * 64) as u8); // R
                raw_image.push((y * 64) as u8); // G
                raw_image.push(128); // B
                raw_image.push(255); // A (fully opaque)
            }
        }

        // Encode to PNG
        let mut encoder = PngEncoder::new();
        let input_buffer = create_video_buffer(
            &raw_image,
            width,
            height,
            parallax::format::PixelFormat::Rgba,
        );
        let encoded = encoder.process(input_buffer).unwrap().unwrap();

        // Decode the PNG
        let mut decoder = PngDecoder::new();
        let decoded = decoder.process(encoded).unwrap().unwrap();

        assert_eq!(decoded.as_bytes().len(), raw_image.len());
        assert_eq!(decoded.as_bytes(), raw_image.as_slice());
    }

    #[test]
    fn test_png_encoder_compression() {
        // Create a larger image to test compression
        let width = 64;
        let height = 64;
        let raw_image = create_rgb_image(width, height);
        let raw_size = raw_image.len();

        // Encode to PNG
        let mut encoder = PngEncoder::new();
        let input_buffer = create_video_buffer(
            &raw_image,
            width,
            height,
            parallax::format::PixelFormat::Rgb24,
        );
        let encoded = encoder.process(input_buffer).unwrap().unwrap();

        // PNG should provide some compression for this pattern
        let compressed_size = encoded.as_bytes().len();

        // Note: compression ratio varies by content, but PNG should not be much larger
        // than raw for simple patterns
        assert!(
            compressed_size < raw_size * 2,
            "PNG encoding should not dramatically increase size: raw={}, compressed={}",
            raw_size,
            compressed_size
        );
    }

    #[test]
    fn test_png_decoder_invalid_data() {
        let mut decoder = PngDecoder::new();
        let invalid_buffer = create_test_buffer(b"not a valid png file");

        // Should return an error for invalid PNG data
        let result = decoder.process(invalid_buffer);
        assert!(result.is_err(), "Should fail on invalid PNG data");
    }

    #[test]
    fn test_png_multiple_frames() {
        let width = 8;
        let height = 8;

        let mut encoder = PngEncoder::new();
        let mut decoder = PngDecoder::new();

        // Process multiple frames
        for i in 0..5 {
            let mut raw_image = vec![0u8; (width * height * 3) as usize];
            // Fill with pattern based on frame number
            for pixel in raw_image.chunks_mut(3) {
                pixel[0] = (i * 50) as u8;
                pixel[1] = (i * 40) as u8;
                pixel[2] = (i * 30) as u8;
            }

            let input_buffer = create_video_buffer(
                &raw_image,
                width,
                height,
                parallax::format::PixelFormat::Rgb24,
            );
            let encoded = encoder.process(input_buffer).unwrap().unwrap();
            let decoded = decoder.process(encoded).unwrap().unwrap();

            assert_eq!(
                decoded.as_bytes(),
                raw_image.as_slice(),
                "Frame {} mismatch",
                i
            );
        }
    }
}

// ============================================================================
// JPEG Codec Tests
// ============================================================================

#[cfg(feature = "image-jpeg")]
mod jpeg_tests {
    use super::*;
    use parallax::element::Element;
    use parallax::elements::codec::JpegDecoder;

    #[test]
    fn test_jpeg_decoder_creation() {
        let decoder = JpegDecoder::new();
        assert_eq!(decoder.frame_count(), 0);
    }

    #[test]
    fn test_jpeg_decoder_invalid() {
        let mut decoder = JpegDecoder::new();
        let input = create_test_buffer(b"not a jpeg file");

        let result = decoder.process(input);
        assert!(result.is_err(), "Should fail on invalid JPEG");
    }

    #[test]
    fn test_jpeg_decoder_truncated() {
        let mut decoder = JpegDecoder::new();
        // Just a JPEG marker, truncated
        let input = create_test_buffer(&[0xFF, 0xD8, 0xFF, 0xE0]);

        let result = decoder.process(input);
        assert!(result.is_err(), "Should fail on truncated JPEG");
    }

    #[test]
    fn test_jpeg_decoder_empty() {
        let mut decoder = JpegDecoder::new();
        let input = create_test_buffer(&[0u8; 10]);

        let result = decoder.process(input);
        assert!(result.is_err(), "Should fail on empty/invalid data");
    }
}

// ============================================================================
// Audio Codec Tests
// ============================================================================

#[cfg(any(
    feature = "audio-flac",
    feature = "audio-mp3",
    feature = "audio-aac",
    feature = "audio-vorbis"
))]
mod audio_tests {
    use super::*;
    use parallax::element::Element;
    use parallax::elements::codec::SymphoniaDecoder;

    #[test]
    fn test_symphonia_decoder_creation() {
        // Test creating decoder with auto-detection
        let decoder_auto = SymphoniaDecoder::new();
        assert!(decoder_auto.is_ok(), "Should create auto-detect decoder");
    }

    #[test]
    fn test_symphonia_decoder_invalid_data() {
        let mut decoder = SymphoniaDecoder::new().expect("Should create decoder");
        let input = create_test_buffer(b"not audio data - this is garbage");

        // The decoder buffers data and returns Ok(None) for small/unrecognized inputs
        // This is expected behavior - it waits for more data
        let result = decoder.process(input);
        // Should succeed but return None (not enough data to decode)
        assert!(result.is_ok(), "Should not error on small invalid data");
        assert!(
            result.unwrap().is_none(),
            "Should return None when data is insufficient"
        );
    }

    /// Real decode: a 0.3s 440 Hz mono MP3 generated with ffmpeg. Exercises
    /// the full probe -> track select -> decode -> interleave path.
    #[cfg(feature = "audio-mp3")]
    #[test]
    fn test_symphonia_decoder_real_mp3() {
        let mut decoder = SymphoniaDecoder::for_format(parallax::elements::codec::AudioFormat::Mp3)
            .expect("Should create decoder");
        let data = include_bytes!("fixtures/sine.mp3");
        let input = create_test_buffer(data);

        let output = decoder.process(input).expect("decode should not error");
        let output = output.expect("a full MP3 file must yield decoded samples");
        assert!(
            !output.as_bytes().is_empty(),
            "decoded PCM must be non-empty"
        );
        // f32 samples: length must be a whole number of samples
        assert_eq!(output.as_bytes().len() % 4, 0);
    }

    #[test]
    fn test_symphonia_decoder_small_data() {
        let mut decoder = SymphoniaDecoder::new().expect("Should create decoder");
        // Create a small buffer (less than 1024 bytes threshold)
        let input = create_test_buffer(&[0u8; 100]);

        // The decoder buffers data and waits for more
        let result = decoder.process(input);
        assert!(result.is_ok(), "Should handle small input gracefully");
        assert!(
            result.unwrap().is_none(),
            "Should return None for small input"
        );
    }
}

// ============================================================================
// AV1 Encoder Tests (requires av1-encode feature)
// ============================================================================

#[cfg(feature = "av1-encode")]
mod av1_encode_tests {
    use super::*;
    use parallax::element::Element;
    use parallax::elements::codec::{Rav1eConfig, Rav1eEncoder};

    fn create_yuv420_frame(width: usize, height: usize) -> Vec<u8> {
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        let total_size = y_size + uv_size * 2;

        let mut data = vec![0u8; total_size];

        // Fill Y plane with gradient
        for y in 0..height {
            for x in 0..width {
                data[y * width + x] = ((x + y) * 255 / (width + height)) as u8;
            }
        }

        // Fill U plane (neutral)
        for i in 0..uv_size {
            data[y_size + i] = 128;
        }

        // Fill V plane (neutral)
        for i in 0..uv_size {
            data[y_size + uv_size + i] = 128;
        }

        data
    }

    #[test]
    fn test_rav1e_encoder_creation() {
        let config = Rav1eConfig::default()
            .dimensions(64, 64)
            .speed(10)
            .quantizer(100);

        let encoder = Rav1eEncoder::new(config);
        assert!(encoder.is_ok(), "Should create encoder");
    }

    #[test]
    fn test_rav1e_encoder_invalid_dimensions() {
        let config = Rav1eConfig::default()
            .dimensions(0, 0)
            .speed(10)
            .quantizer(100);

        let encoder = Rav1eEncoder::new(config);
        assert!(encoder.is_err(), "Should fail with zero dimensions");
    }

    #[test]
    fn test_rav1e_encode_frame() {
        let width = 64;
        let height = 64;

        let config = Rav1eConfig::default()
            .dimensions(width, height)
            .speed(10)
            .quantizer(100);

        let mut encoder = Rav1eEncoder::new(config).expect("Should create encoder");

        // Create a YUV420 frame
        let frame_data = create_yuv420_frame(width, height);
        let input = create_test_buffer(&frame_data);

        // First frame may not produce immediate output (encoder buffering)
        let result = encoder.process(input);
        assert!(result.is_ok(), "Should process frame without error");
    }
}

// ============================================================================
// Codec Integration Tests
// ============================================================================

#[cfg(feature = "image-png")]
mod integration_tests {
    use super::*;
    use parallax::element::Element;
    use parallax::elements::codec::{PngDecoder, PngEncoder};

    #[test]
    fn test_encoder_decoder_pipeline() {
        // Simulate a pipeline: raw image -> PNG encode -> PNG decode -> verify
        let width = 32;
        let height = 32;

        // Create raw image data
        let mut raw_data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                raw_data.push((x * 8) as u8);
                raw_data.push((y * 8) as u8);
                raw_data.push(((x + y) * 4) as u8);
            }
        }

        let mut encoder = PngEncoder::new();
        let mut decoder = PngDecoder::new();

        // Process through encoder
        let input = create_video_buffer(
            &raw_data,
            width,
            height,
            parallax::format::PixelFormat::Rgb24,
        );
        let encoded = encoder.process(input).unwrap().unwrap();

        // Process through decoder
        let decoded = decoder.process(encoded).unwrap().unwrap();

        // Verify round-trip
        assert_eq!(decoded.as_bytes(), raw_data.as_slice());
    }

    #[test]
    fn test_streaming_encode_decode() {
        // Test streaming multiple frames through codec pipeline
        let width = 16;
        let height = 16;
        let frame_count = 10;

        let mut encoder = PngEncoder::new();
        let mut decoder = PngDecoder::new();

        for frame_idx in 0..frame_count {
            // Generate unique frame data
            let mut raw_data = vec![0u8; (width * height * 3) as usize];
            for (i, byte) in raw_data.iter_mut().enumerate() {
                *byte = ((i + frame_idx * 100) % 256) as u8;
            }

            let input = create_video_buffer(
                &raw_data,
                width,
                height,
                parallax::format::PixelFormat::Rgb24,
            );
            let encoded = encoder.process(input).unwrap().unwrap();
            let decoded = decoder.process(encoded).unwrap().unwrap();

            assert_eq!(
                decoded.as_bytes(),
                raw_data.as_slice(),
                "Frame {} round-trip failed",
                frame_idx
            );
        }
    }
}

// ============================================================================
// Geometry travels in-band (#38)
// ============================================================================

/// The invariant these tests pin down:
///
/// > **Geometry travels in-band, in `Metadata`. No element takes dimensions at
/// > construction. An element that cannot determine its geometry from metadata
/// > errors — it never falls back to a stale constructor value.**
///
/// Before this, elements disagreed about where geometry lived, and that
/// disagreement *was* a bug class: put a scaler upstream of an image encoder
/// constructed at the source size and it either threw "Input buffer too small"
/// when the frame shrank, or silently encoded the top-left corner of a larger
/// one.
#[cfg(feature = "image-jpeg")]
mod geometry_in_metadata {
    use super::*;
    use parallax::element::Element;
    use parallax::elements::codec::{JpegDecoder, JpegEncoder};
    use parallax::elements::transform::VideoScale;
    use parallax::format::PixelFormat;

    fn rgb_gradient(width: u32, height: u32) -> Vec<u8> {
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                data.push((x * 255 / width.max(1)) as u8);
                data.push((y * 255 / height.max(1)) as u8);
                data.push(64);
            }
        }
        data
    }

    /// The zensight preview branch, and the thing 0.2.0 could not express at all:
    /// decode a JPEG, scale it down, re-encode it. Previews were being encoded at
    /// *full source size* — a 1080p camera's thumbnail was a 1080p JPEG rendered
    /// into a 320x180 tile — because the scaler corrupted RGB (#36/#37) and the
    /// encoder's geometry was frozen at construction (#38). Neither fix alone is
    /// enough; this needs both.
    ///
    /// Note what is *not* here: no dimensions are passed to any constructor. Every
    /// element learns the geometry from the buffer it is handed.
    #[test]
    fn jpeg_decode_scale_encode_rgb_chain() {
        const SRC_W: u32 = 32;
        const SRC_H: u32 = 16;

        // A source JPEG at 32x16.
        let mut jpeg_encoder = JpegEncoder::new().with_quality(95);
        let source = create_video_buffer(
            &rgb_gradient(SRC_W, SRC_H),
            SRC_W,
            SRC_H,
            PixelFormat::Rgb24,
        );
        let jpeg = jpeg_encoder.process(source).unwrap().unwrap();

        // Decode -> the decoder stamps 32x16 Rgb24.
        let mut decoder = JpegDecoder::new();
        let decoded = decoder.process(jpeg).unwrap().unwrap();
        assert_eq!(decoded.metadata().video_dims(), Some((SRC_W, SRC_H)));
        assert_eq!(
            decoded.metadata().video_pixel_format(),
            Some(PixelFormat::Rgb24),
            "the scaler needs to know this is RGB, not planar YUV"
        );

        // Scale to half size, in RGB, with no colourspace round-trip.
        let mut scaler = VideoScale::new();
        scaler.control().set_target(SRC_W / 2, SRC_H / 2);
        let scaled = scaler.process(decoded).unwrap().unwrap();

        assert_eq!(scaled.metadata().video_dims(), Some((16, 8)));
        assert_eq!(
            scaled.metadata().video_pixel_format(),
            Some(PixelFormat::Rgb24),
            "scaling preserves the pixel format"
        );
        assert_eq!(scaled.len(), 16 * 8 * 3);

        // Re-encode. The encoder takes 16x8 from the buffer — it was never told a
        // size, so it cannot be encoding a stale one.
        let mut preview_encoder = JpegEncoder::new().with_quality(75);
        let preview = preview_encoder.process(scaled).unwrap().unwrap();

        let mut check = JpegDecoder::new();
        let final_frame = check.process(preview).unwrap().unwrap();
        assert_eq!(
            final_frame.metadata().video_dims(),
            Some((16, 8)),
            "the preview must be a *small* JPEG, not a full-size one"
        );

        // And it must still look like the gradient: left darker than right.
        let px = final_frame.as_bytes();
        let left_red = px[0] as i32;
        let right_red = px[15 * 3] as i32;
        assert!(
            right_red > left_red + 40,
            "the horizontal gradient did not survive the chain \
             (left R={left_red}, right R={right_red})"
        );
    }

    /// An image encoder follows the buffer, frame by frame. Constructed once, fed
    /// two different sizes, it must encode each at its own size — where before it
    /// would have cropped the larger one to the corner or errored on the smaller.
    #[test]
    fn image_encoder_geometry_follows_the_buffer() {
        let mut encoder = JpegEncoder::new();
        let mut decoder = JpegDecoder::new();

        for (w, h) in [(32u32, 16u32), (16, 8), (48, 24)] {
            let raw = create_video_buffer(&rgb_gradient(w, h), w, h, PixelFormat::Rgb24);
            let jpeg = encoder.process(raw).unwrap().unwrap();
            let decoded = decoder.process(jpeg).unwrap().unwrap();

            assert_eq!(
                decoded.metadata().video_dims(),
                Some((w, h)),
                "encoder did not follow the buffer to {w}x{h}"
            );
        }
    }

    /// A YUV frame handed to an image encoder means someone forgot a
    /// VideoConvertElement. It used to be encoded as if its planes were
    /// interleaved RGB — garbage, silently. Say so instead.
    #[test]
    fn an_image_encoder_rejects_yuv_input() {
        let yuv = create_video_buffer(&vec![128u8; 16 * 16 * 3 / 2], 16, 16, PixelFormat::I420);

        let err = JpegEncoder::new().process(yuv).unwrap_err().to_string();
        assert!(
            err.contains("I420"),
            "the error must name the format: {err}"
        );
        assert!(
            err.contains("VideoConvertElement"),
            "the error must say how to fix it: {err}"
        );
    }

    /// No geometry, no encode. There is no constructor value left to fall back to,
    /// and inventing one produces sheared or truncated pictures.
    #[test]
    fn an_image_encoder_without_geometry_errors() {
        let bare = create_test_buffer(&[0u8; 768]);

        let err = JpegEncoder::new().process(bare).unwrap_err().to_string();
        assert!(
            err.contains("no video dimensions"),
            "unhelpful error: {err}"
        );
        assert!(err.contains("set_video_dims"), "unhelpful error: {err}");
    }
}
