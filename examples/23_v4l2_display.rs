//! V4L2 video capture with window display.
//!
//! This example demonstrates capturing video frames from a V4L2 device
//! (webcam) and displaying them in a native window using AutoVideoSink.
//!
//! The pipeline uses caps negotiation to automatically insert a videoconvert
//! element when the source format (YUYV) doesn't match the sink format (RGBA).
//!
//! Run with: `cargo run --example 23_v4l2_display --features "v4l2,display"`
//!
//! Requirements:
//! - Linux system with V4L2 support
//! - A video capture device (webcam, etc.)
//! - User must be in the 'video' group or have permissions to /dev/video*

use parallax::elements::device::enumerate_video_devices;
use parallax::error::Result;
use parallax::pipeline::Pipeline;

#[tokio::main]
async fn main() -> Result<()> {
    // Enable debug logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("parallax=info".parse().unwrap()),
        )
        .init();

    println!("V4L2 Video Capture with Display");
    println!("================================\n");

    // List available video devices
    println!("Scanning for video capture devices...\n");

    let devices = enumerate_video_devices()?;

    if devices.is_empty() {
        println!("No video capture devices found!");
        println!("\nPossible reasons:");
        println!("  - No camera connected");
        println!("  - Missing permissions (try: sudo usermod -aG video $USER)");
        println!("  - V4L2 drivers not loaded");
        return Ok(());
    }

    println!("Found {} device(s):\n", devices.len());
    for (i, dev) in devices.iter().enumerate() {
        println!(
            "  [{}] {} - {} (backend: {:?})",
            i, dev.id, dev.name, dev.backend
        );
    }

    // Use first device
    let device_path = &devices[0].id;
    println!("\nUsing device: {}", device_path);
    println!("Close window to stop.\n");

    // Pipeline without explicit videoconvert - negotiation will auto-insert it!
    // v4l2src outputs YUYV, autovideosink expects RGBA -> videoconvert inserted
    let pipeline_str = format!("v4l2src device={} ! autovideosink", device_path);

    println!("Pipeline: {}", pipeline_str);

    let mut pipeline = Pipeline::parse(&pipeline_str)?;

    // Prepare the pipeline (runs negotiation and inserts converters)
    pipeline.prepare()?;

    // Show the final pipeline with any auto-inserted converters
    println!("\nFinal pipeline after negotiation:");
    println!("{}", pipeline.describe());

    // Run the pipeline
    pipeline.run().await?;

    println!("\nDone!");
    Ok(())
}
