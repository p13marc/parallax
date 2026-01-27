//! V4L2 video capture example.
//!
//! This example demonstrates capturing video frames from a V4L2 device
//! (webcam, capture card, etc.) and discarding them (NullSink).
//!
//! Run with: `cargo run --example 22_v4l2_capture --features v4l2`
//!
//! Requirements:
//! - Linux system with V4L2 support
//! - A video capture device (webcam, etc.)
//! - User must be in the 'video' group or have permissions to /dev/video*

use parallax::elements::NullSink;
use parallax::elements::device::{V4l2Src, enumerate_video_devices};
use parallax::error::Result;
use parallax::memory::CpuArena;
use parallax::pipeline::Pipeline;
use std::time::Duration;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("parallax=info")
        .init();

    println!("V4L2 Video Capture Example");
    println!("==========================\n");

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
        if let Some(model) = &dev.model {
            println!("      Model: {}", model);
        }
        if let Some(loc) = &dev.location {
            println!("      Location: {:?}", loc);
        }
    }

    // Use first device
    let device_path = &devices[0].id;
    println!("\nUsing device: {}\n", device_path);

    // Create V4L2 source
    let src = V4l2Src::new(device_path)?;

    println!("Capture configuration:");
    println!("  Resolution: {}x{}", src.width(), src.height());
    println!(
        "  Format: {}",
        std::str::from_utf8(src.fourcc()).unwrap_or("????")
    );

    // Calculate frame size for buffer
    let frame_size = match src.fourcc() {
        b"MJPG" | b"JPEG" => (src.width() * src.height() / 4) as usize,
        b"YUYV" | b"UYVY" => (src.width() * src.height() * 2) as usize,
        _ => (src.width() * src.height() * 3) as usize,
    };

    println!("  Estimated frame size: {} bytes", frame_size);
    println!("\nCapturing for 3 seconds...\n");

    // Create arena for buffers
    let arena = CpuArena::new(frame_size, 4)?;

    // Build pipeline: v4l2src → nullsink
    let mut pipeline = Pipeline::new();

    let src_node = pipeline.add_source_with_arena("v4l2src", src, arena);
    let sink_node = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src_node, sink_node)?;

    // Run for a short duration
    tokio::select! {
        result = pipeline.run() => {
            if let Err(e) = result {
                eprintln!("Pipeline error: {}", e);
            }
        }
        _ = tokio::time::sleep(Duration::from_secs(3)) => {
            println!("Capture complete!");
        }
    }

    println!("\nDone!");
    Ok(())
}
