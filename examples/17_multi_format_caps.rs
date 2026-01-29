//! # Multi-Format Caps Negotiation
//!
//! Demonstrates how elements can declare multiple supported formats and how
//! the pipeline automatically negotiates the best common format.
//!
//! Key concepts:
//! - `ElementMediaCaps`: Holds multiple format+memory combinations
//! - `FormatMemoryCap`: Couples a format with memory type constraints
//! - Formats are listed in preference order
//! - Negotiation finds the first (highest preference) common format
//!
//! Run: `cargo run --example 17_multi_format_caps`

use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::format::{
    CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, PixelFormat,
    VideoFormatCaps,
};
use parallax::memory::SharedArena;
use parallax::pipeline::Pipeline;

// =============================================================================
// Multi-Format Video Source
// =============================================================================

/// A video source that supports multiple pixel formats.
///
/// This simulates a camera or capture device that can output video in
/// multiple formats. The formats are listed in preference order.
struct MultiFormatVideoSource {
    frame_count: u32,
    max_frames: u32,
}

impl MultiFormatVideoSource {
    fn new(max_frames: u32) -> Self {
        Self {
            frame_count: 0,
            max_frames,
        }
    }
}

impl Source for MultiFormatVideoSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.frame_count >= self.max_frames {
            return Ok(ProduceResult::Eos);
        }

        self.frame_count += 1;

        // Create a simple test frame
        let frame_data = format!("Frame {}", self.frame_count);
        let data = frame_data.as_bytes();

        // Write to context buffer
        let len = data.len().min(ctx.capacity());
        ctx.output()[..len].copy_from_slice(&data[..len]);
        Ok(ProduceResult::Produced(len))
    }

    fn output_media_caps(&self) -> ElementMediaCaps {
        // Declare support for multiple formats at 640x480
        // Listed in preference order: YUYV (most preferred), then RGB24, then I420

        let yuyv = VideoFormatCaps {
            width: CapsValue::Fixed(640),
            height: CapsValue::Fixed(480),
            pixel_format: CapsValue::Fixed(PixelFormat::Yuyv),
            ..VideoFormatCaps::any()
        };

        let rgb24 = VideoFormatCaps {
            width: CapsValue::Fixed(640),
            height: CapsValue::Fixed(480),
            pixel_format: CapsValue::Fixed(PixelFormat::Rgb24),
            ..VideoFormatCaps::any()
        };

        let i420 = VideoFormatCaps {
            width: CapsValue::Fixed(640),
            height: CapsValue::Fixed(480),
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            ..VideoFormatCaps::any()
        };

        ElementMediaCaps::new(vec![
            FormatMemoryCap::new(yuyv.into(), MemoryCaps::cpu_only()),
            FormatMemoryCap::new(rgb24.into(), MemoryCaps::cpu_only()),
            FormatMemoryCap::new(i420.into(), MemoryCaps::cpu_only()),
        ])
    }
}

// =============================================================================
// Format-Specific Sinks
// =============================================================================

/// A sink that only accepts I420 format.
struct I420Sink {
    received: u32,
}

impl I420Sink {
    fn new() -> Self {
        Self { received: 0 }
    }
}

impl Sink for I420Sink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.received += 1;
        println!(
            "[I420Sink] Received frame {}: {} bytes",
            self.received,
            ctx.input().len()
        );
        Ok(())
    }

    fn input_media_caps(&self) -> ElementMediaCaps {
        let i420 = VideoFormatCaps {
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            ..VideoFormatCaps::any()
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            i420.into(),
            MemoryCaps::cpu_only(),
        )])
    }
}

/// A sink that accepts multiple formats (RGB24 preferred, then RGBA).
struct RgbSink {
    received: u32,
}

impl RgbSink {
    fn new() -> Self {
        Self { received: 0 }
    }
}

impl Sink for RgbSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.received += 1;
        println!(
            "[RgbSink] Received frame {}: {} bytes",
            self.received,
            ctx.input().len()
        );
        Ok(())
    }

    fn input_media_caps(&self) -> ElementMediaCaps {
        let rgb24 = VideoFormatCaps {
            pixel_format: CapsValue::Fixed(PixelFormat::Rgb24),
            ..VideoFormatCaps::any()
        };

        let rgba = VideoFormatCaps {
            pixel_format: CapsValue::Fixed(PixelFormat::Rgba),
            ..VideoFormatCaps::any()
        };

        ElementMediaCaps::new(vec![
            FormatMemoryCap::new(rgb24.into(), MemoryCaps::cpu_only()),
            FormatMemoryCap::new(rgba.into(), MemoryCaps::cpu_only()),
        ])
    }
}

// =============================================================================
// Main
// =============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Multi-Format Caps Negotiation Example ===\n");

    let arena = SharedArena::new(1024, 4)?;

    // Example 1: Source supports [YUYV, RGB24, I420], Sink accepts [I420]
    // Should negotiate to I420 (the only common format)
    println!("--- Example 1: Negotiate to I420 ---");
    println!("Source outputs: [YUYV, RGB24, I420]");
    println!("Sink accepts:   [I420]");
    println!("Expected:       I420 (the only common format)");
    {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_source_with_arena(
            "video_src",
            MultiFormatVideoSource::new(3),
            arena.clone(),
        );
        let sink = pipeline.add_sink("i420_sink", I420Sink::new());

        pipeline.link(src, sink)?;

        // Negotiate - should find I420 as common format
        pipeline.negotiate()?;
        println!("Negotiation successful!");

        // Run the pipeline
        println!("\nRunning pipeline...");
        pipeline.run().await?;
    }

    println!("\n--- Example 2: Negotiate to RGB24 ---");
    println!("Source outputs: [YUYV, RGB24, I420]");
    println!("Sink accepts:   [RGB24, RGBA]");
    println!("Expected:       RGB24 (source's 2nd choice matches sink's 1st)");
    {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_source_with_arena(
            "video_src",
            MultiFormatVideoSource::new(3),
            arena.clone(),
        );
        let sink = pipeline.add_sink("rgb_sink", RgbSink::new());

        pipeline.link(src, sink)?;

        // Negotiate - should find RGB24 as common format
        pipeline.negotiate()?;
        println!("Negotiation successful!");

        // Run the pipeline
        println!("\nRunning pipeline...");
        pipeline.run().await?;
    }

    println!("\n--- Example 3: Inspect Source Caps ---");
    {
        let mut pipeline = Pipeline::new();
        let src = pipeline.add_source_with_arena(
            "video_src",
            MultiFormatVideoSource::new(1),
            arena.clone(),
        );

        // Get the node and inspect its caps
        if let Some(node) = pipeline.get_node(src) {
            let caps = node.output_media_caps();
            println!("Source supports {} format(s):", caps.len());
            for (i, cap) in caps.iter().enumerate() {
                if let FormatCaps::VideoRaw(v) = &cap.format {
                    println!(
                        "  {}. {:?} @ {:?}x{:?}",
                        i + 1,
                        v.pixel_format,
                        v.width,
                        v.height
                    );
                }
            }
        }
    }

    println!("\n=== Done ===");
    Ok(())
}
