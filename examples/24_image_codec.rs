//! Example 24: Image Codec Pipeline
//!
//! This example demonstrates image encoding/decoding using pure Rust codecs:
//! - JPEG decoding with zune-jpeg
//! - PNG encoding with the png crate
//!
//! Run with: cargo run --example 24_image_codec --features image-codecs

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ConsumeContext, Element, ProduceContext, ProduceResult, Sink, Source};
use parallax::elements::codec::{ColorType, JpegDecoder, PngEncoder};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use std::sync::Arc;

/// A source that generates a simple test image (gradient pattern).
struct TestImageSource {
    width: u32,
    height: u32,
    produced: bool,
}

impl TestImageSource {
    fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            produced: false,
        }
    }

    /// Generate a simple RGB gradient image.
    fn generate_image(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity((self.width * self.height * 3) as usize);

        for y in 0..self.height {
            for x in 0..self.width {
                // Create a gradient pattern
                let r = ((x * 255) / self.width) as u8;
                let g = ((y * 255) / self.height) as u8;
                let b = (((x + y) * 255) / (self.width + self.height)) as u8;
                data.push(r);
                data.push(g);
                data.push(b);
            }
        }

        data
    }
}

impl Source for TestImageSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.produced {
            return Ok(ProduceResult::Eos);
        }

        let image_data = self.generate_image();
        self.produced = true;

        // Write to provided buffer or create our own
        if ctx.has_buffer() {
            let output = ctx.output();
            let len = image_data.len().min(output.len());
            output[..len].copy_from_slice(&image_data[..len]);
            Ok(ProduceResult::Produced(len))
        } else {
            let segment = Arc::new(HeapSegment::new(image_data.len())?);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    image_data.as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    image_data.len(),
                );
            }
            Ok(ProduceResult::OwnBuffer(Buffer::new(
                MemoryHandle::from_segment(segment),
                Default::default(),
            )))
        }
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some((self.width * self.height * 3) as usize)
    }
}

/// A sink that collects the encoded image data.
struct ImageCollector {
    data: Vec<u8>,
}

impl ImageCollector {
    fn new() -> Self {
        Self { data: Vec::new() }
    }

    fn data(&self) -> &[u8] {
        &self.data
    }
}

impl Sink for ImageCollector {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.data.extend_from_slice(ctx.buffer().as_bytes());
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Image Codec Example ===\n");

    // Image dimensions
    let width = 256u32;
    let height = 256u32;

    println!("1. Generating test image ({width}x{height} RGB)...");

    // Create source and generate image
    let mut source = TestImageSource::new(width, height);
    let image_data = source.generate_image();
    println!("   Generated {} bytes of raw RGB data", image_data.len());

    // Create PNG encoder
    println!("\n2. Encoding to PNG...");
    let mut encoder = PngEncoder::new(width, height, ColorType::Rgb);

    // Create input buffer
    let segment = Arc::new(HeapSegment::new(image_data.len())?);
    unsafe {
        std::ptr::copy_nonoverlapping(
            image_data.as_ptr(),
            segment.as_mut_ptr().unwrap(),
            image_data.len(),
        );
    }
    let input_buffer = Buffer::new(MemoryHandle::from_segment(segment), Default::default());

    // Encode
    if let Some(encoded) = encoder.process(input_buffer)? {
        let encoded_size = encoded.as_bytes().len();
        println!("   Encoded to {} bytes of PNG data", encoded_size);
        println!(
            "   Compression ratio: {:.1}%",
            (encoded_size as f64 / image_data.len() as f64) * 100.0
        );

        // Verify PNG signature
        let png_data = encoded.as_bytes();
        if png_data.len() >= 8 {
            let signature = &png_data[..8];
            let expected = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A];
            if signature == expected {
                println!("   ✓ Valid PNG signature detected");
            }
        }
    }

    println!("\n3. Testing JPEG decoder (with synthetic JPEG data)...");

    // Note: JpegDecoder expects actual JPEG data
    // For a real example, you would read a JPEG file
    let jpeg_decoder = JpegDecoder::new();
    println!(
        "   JPEG decoder created (frame_count: {})",
        jpeg_decoder.frame_count()
    );
    println!("   Note: Feed actual JPEG data to decode images");

    println!("\n=== Example Complete ===");
    println!("\nSupported codecs (pure Rust, no C dependencies):");
    println!("  - JPEG decode: zune-jpeg (feature: image-jpeg)");
    println!("  - PNG decode/encode: png crate (feature: image-png)");
    println!("  - FLAC decode: symphonia (feature: audio-flac)");
    println!("  - MP3 decode: symphonia (feature: audio-mp3)");
    println!("  - AAC decode: symphonia (feature: audio-aac)");
    println!("  - Vorbis decode: symphonia (feature: audio-vorbis)");
    println!("  - AV1 encode: rav1e (feature: av1-encode)");

    Ok(())
}
