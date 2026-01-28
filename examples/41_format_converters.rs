//! # Format Converters Example
//!
//! Demonstrates the use of video and audio format converters in pipelines.
//!
//! This example shows:
//! - Video pixel format conversion (YUYV -> RGBA)
//! - Audio sample format conversion (S16 -> F32)
//! - Audio resampling (48kHz -> 44.1kHz)
//!
//! Run: `cargo run --example 41_format_converters`

use parallax::converters::{PixelFormat, SampleFormat};
use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::elements::transform::{
    AudioConvertElement, AudioResampleElement, VideoConvertElement,
};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::pipeline::Pipeline;

// =============================================================================
// Video Format Conversion Demo
// =============================================================================

/// A test source that produces YUYV video frames.
struct YuyvSource {
    frame_count: u32,
    max_frames: u32,
    width: u32,
    height: u32,
}

impl YuyvSource {
    fn new(width: u32, height: u32, max_frames: u32) -> Self {
        Self {
            frame_count: 0,
            max_frames,
            width,
            height,
        }
    }
}

impl Source for YuyvSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.frame_count >= self.max_frames {
            return Ok(ProduceResult::Eos);
        }

        self.frame_count += 1;

        // YUYV is 2 bytes per pixel (packed YUV 4:2:2)
        let frame_size = (self.width * self.height * 2) as usize;

        if ctx.capacity() < frame_size {
            return Ok(ProduceResult::WouldBlock);
        }

        // Generate a simple test pattern:
        // Y increases horizontally, U and V are fixed
        let output = ctx.output();
        for y in 0..self.height {
            for x in (0..self.width).step_by(2) {
                let idx = ((y * self.width + x) * 2) as usize;
                let y_val = ((x * 255) / self.width) as u8;
                output[idx] = y_val; // Y0
                output[idx + 1] = 128; // U
                output[idx + 2] = y_val; // Y1
                output[idx + 3] = 128; // V
            }
        }

        println!(
            "[YuyvSource] Produced frame {} ({} bytes YUYV)",
            self.frame_count, frame_size
        );
        Ok(ProduceResult::Produced(frame_size))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some((self.width * self.height * 2) as usize)
    }
}

/// A sink that receives RGBA video frames.
struct RgbaSink {
    received: u32,
    expected_size: usize,
}

impl RgbaSink {
    fn new(width: u32, height: u32) -> Self {
        Self {
            received: 0,
            expected_size: (width * height * 4) as usize,
        }
    }
}

impl Sink for RgbaSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.received += 1;
        let data = ctx.input();

        // Verify size is correct for RGBA
        let size_ok = data.len() == self.expected_size;

        // Sample some pixels
        let first_pixel = if data.len() >= 4 {
            format!("R={} G={} B={} A={}", data[0], data[1], data[2], data[3])
        } else {
            "N/A".to_string()
        };

        println!(
            "[RgbaSink] Frame {}: {} bytes (expected: {}, ok: {}), first pixel: {}",
            self.received,
            data.len(),
            self.expected_size,
            size_ok,
            first_pixel
        );

        Ok(())
    }
}

// =============================================================================
// Audio Format Conversion Demo
// =============================================================================

/// A test source that produces S16 audio samples.
struct S16AudioSource {
    sample_count: u32,
    max_samples: u32,
    sample_rate: u32,
    channels: u32,
}

impl S16AudioSource {
    fn new(sample_rate: u32, channels: u32, duration_ms: u32) -> Self {
        let max_samples = (sample_rate * duration_ms / 1000) * channels;
        Self {
            sample_count: 0,
            max_samples,
            sample_rate,
            channels,
        }
    }
}

impl Source for S16AudioSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.sample_count >= self.max_samples {
            return Ok(ProduceResult::Eos);
        }

        // Produce 10ms of audio at a time
        let samples_per_chunk = (self.sample_rate / 100) * self.channels;
        let remaining = self.max_samples - self.sample_count;
        let to_produce = samples_per_chunk.min(remaining);
        let bytes_needed = (to_produce * 2) as usize; // S16 = 2 bytes

        if ctx.capacity() < bytes_needed {
            return Ok(ProduceResult::WouldBlock);
        }

        // Generate a 440Hz sine wave
        let output = ctx.output();
        for i in 0..to_produce {
            let sample_idx = self.sample_count + i;
            let t = sample_idx as f32 / self.sample_rate as f32;
            let value = (t * 440.0 * 2.0 * std::f32::consts::PI).sin();
            let s16_value = (value * 32767.0) as i16;

            let byte_idx = (i * 2) as usize;
            let bytes = s16_value.to_le_bytes();
            output[byte_idx] = bytes[0];
            output[byte_idx + 1] = bytes[1];
        }

        self.sample_count += to_produce;
        println!(
            "[S16AudioSource] Produced {} S16 samples ({} bytes)",
            to_produce, bytes_needed
        );
        Ok(ProduceResult::Produced(bytes_needed))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(((self.sample_rate / 100) * self.channels * 2) as usize)
    }
}

/// A sink that receives F32 audio samples.
struct F32AudioSink {
    received_bytes: usize,
    chunks: u32,
}

impl F32AudioSink {
    fn new() -> Self {
        Self {
            received_bytes: 0,
            chunks: 0,
        }
    }
}

impl Sink for F32AudioSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.chunks += 1;
        let data = ctx.input();
        self.received_bytes += data.len();

        // Sample some values
        let sample_count = data.len() / 4;
        let first_sample = if data.len() >= 4 {
            f32::from_le_bytes([data[0], data[1], data[2], data[3]])
        } else {
            0.0
        };

        println!(
            "[F32AudioSink] Chunk {}: {} F32 samples, first value: {:.4}",
            self.chunks, sample_count, first_sample
        );

        Ok(())
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Format Converters Example ===\n");

    // Example 1: Video pixel format conversion (YUYV -> RGBA)
    println!("--- Example 1: Video YUYV -> RGBA ---");
    {
        let width = 320;
        let height = 240;

        let arena = SharedArena::new((width * height * 2) as usize, 8)?;
        let mut pipeline = Pipeline::new();

        let src =
            pipeline.add_source_with_arena("yuyv_src", YuyvSource::new(width, height, 3), arena);

        // VideoConvertElement auto-detects input format and converts to RGBA
        let convert = pipeline.add_transform(
            "convert",
            VideoConvertElement::new()
                .with_input_format(PixelFormat::Yuyv)
                .with_output_format(PixelFormat::Rgba)
                .with_size(width, height),
        );

        let sink = pipeline.add_sink("rgba_sink", RgbaSink::new(width, height));

        pipeline.link(src, convert)?;
        pipeline.link(convert, sink)?;

        pipeline.run().await?;
        println!("Video conversion complete!\n");
    }

    // Example 2: Audio sample format conversion (S16 -> F32)
    println!("--- Example 2: Audio S16 -> F32 ---");
    {
        let sample_rate = 48000;
        let channels = 2;
        let duration_ms = 50; // 50ms of audio

        let arena = SharedArena::new((sample_rate / 100 * channels * 2) as usize, 8)?;
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_source_with_arena(
            "s16_src",
            S16AudioSource::new(sample_rate, channels, duration_ms),
            arena,
        );

        // AudioConvertElement converts S16 to F32
        let convert = pipeline.add_transform(
            "convert",
            AudioConvertElement::new()
                .with_input_format(SampleFormat::S16Le)
                .with_output_format(SampleFormat::F32Le)
                .with_channels(channels),
        );

        let sink = pipeline.add_sink("f32_sink", F32AudioSink::new());

        pipeline.link(src, convert)?;
        pipeline.link(convert, sink)?;

        pipeline.run().await?;
        println!("Audio format conversion complete!\n");
    }

    // Example 3: Audio resampling (48kHz -> 44.1kHz)
    println!("--- Example 3: Audio Resampling 48kHz -> 44.1kHz ---");
    {
        let input_rate = 48000;
        let output_rate = 44100;
        let channels = 1;
        let duration_ms = 100; // 100ms of audio

        let arena = SharedArena::new((input_rate / 100 * channels * 4) as usize, 8)?;
        let mut pipeline = Pipeline::new();

        // For resampling, we need F32 samples
        // First convert S16 -> F32, then resample
        let src = pipeline.add_source_with_arena(
            "s16_src",
            S16AudioSource::new(input_rate, channels, duration_ms),
            arena,
        );

        let convert = pipeline.add_transform(
            "convert",
            AudioConvertElement::new()
                .with_input_format(SampleFormat::S16Le)
                .with_output_format(SampleFormat::F32Le)
                .with_channels(channels),
        );

        let resample = pipeline.add_transform(
            "resample",
            AudioResampleElement::new()
                .with_input_rate(input_rate)
                .with_output_rate(output_rate)
                .with_channels(channels)
                .with_format(SampleFormat::F32Le),
        );

        let sink = pipeline.add_sink("f32_sink", F32AudioSink::new());

        pipeline.link(src, convert)?;
        pipeline.link(convert, resample)?;
        pipeline.link(resample, sink)?;

        pipeline.run().await?;
        println!("Audio resampling complete!\n");
    }

    // Example 4: Direct converter usage (without pipeline)
    println!("--- Example 4: Direct Converter Usage ---");
    {
        use parallax::converters::{AudioConvert, VideoConvert};

        // Video: Convert a small YUYV frame to RGBA
        let width = 4;
        let height = 2;
        let converter = VideoConvert::new(PixelFormat::Yuyv, PixelFormat::Rgba, width, height)?;

        // YUYV: Y0 U Y1 V pattern (2 pixels = 4 bytes)
        let yuyv_input: Vec<u8> = vec![
            128, 128, 128, 128, // 2 gray pixels
            200, 128, 200, 128, // 2 bright pixels
            50, 128, 50, 128, // 2 dark pixels
            128, 128, 128, 128, // 2 gray pixels
        ];

        let mut rgba_output = vec![0u8; (width * height * 4) as usize];
        converter.convert(&yuyv_input, &mut rgba_output)?;

        println!(
            "Converted {}x{} YUYV ({} bytes) to RGBA ({} bytes)",
            width,
            height,
            yuyv_input.len(),
            rgba_output.len()
        );
        println!(
            "First pixel RGBA: [{}, {}, {}, {}]",
            rgba_output[0], rgba_output[1], rgba_output[2], rgba_output[3]
        );

        // Audio: Convert S16 samples to F32
        let audio_converter = AudioConvert::new(SampleFormat::S16Le, SampleFormat::F32Le, 1)?;

        let s16_samples: Vec<i16> = vec![0, 16384, -16384, 32767, -32768];
        let s16_bytes: Vec<u8> = s16_samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let mut f32_output = vec![0u8; s16_samples.len() * 4];
        audio_converter.convert(&s16_bytes, &mut f32_output)?;

        println!("\nAudio S16 -> F32 conversion:");
        for (i, s16) in s16_samples.iter().enumerate() {
            let f32_bytes = &f32_output[i * 4..(i + 1) * 4];
            let f32_val =
                f32::from_le_bytes([f32_bytes[0], f32_bytes[1], f32_bytes[2], f32_bytes[3]]);
            println!("  S16 {} -> F32 {:.4}", s16, f32_val);
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
