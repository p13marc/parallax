//! V4L2 video capture with window display.
//!
//! This example demonstrates capturing video frames from a V4L2 device
//! (webcam) and displaying them in a native window using AutoVideoSink.
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

    // Simple pipeline: v4l2src -> videoconvert -> autovideosink
    let pipeline_str = format!(
        "v4l2src device={} ! videoconvert ! autovideosink",
        device_path
    );

    let mut pipeline = Pipeline::parse(&pipeline_str)?;
    pipeline.run().await?;

    println!("\nDone!");
    Ok(())
}
