//! Screen capture via XDG Desktop Portal.
//!
//! This example captures screen content and saves it to a raw video file
//! that can be played with ffplay or converted with ffmpeg.
//!
//! Run with: cargo run --example 46_screen_capture --features screen-capture
//!
//! After capture, play with:
//!   ffplay -f rawvideo -pixel_format bgra -video_size WIDTHxHEIGHT screen_capture.raw
//!
//! Or convert to MP4:
//!   ffmpeg -f rawvideo -pixel_format bgra -video_size WIDTHxHEIGHT -framerate 30 \
//!          -i screen_capture.raw -c:v libx264 -pix_fmt yuv420p screen_capture.mp4
//!
//! Requirements:
//! - XDG Desktop Portal service running
//! - PipeWire session manager
//! - Portal backend (xdg-desktop-portal-gnome, xdg-desktop-portal-kde, etc.)

use std::fs::File;
use std::io::Write;
use std::time::{Duration, Instant};

use parallax::elements::device::{CaptureSourceType, ScreenCaptureConfig, ScreenCaptureSrc};
use parallax::error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter("parallax=info")
        .init();

    println!("Screen Capture Example");
    println!("======================");
    println!();
    println!("This will prompt you to select a screen or window to capture.");
    println!();

    // Create screen capture configuration
    let config = ScreenCaptureConfig {
        source_type: CaptureSourceType::Any, // Let user choose monitor or window
        show_cursor: true,
        persist_session: false,
    };

    // Create the screen capture source
    let mut capture = ScreenCaptureSrc::new(config);

    // Initialize the portal session (this shows the permission dialog)
    println!("Requesting screen capture permission...");
    let info = capture.initialize().await?;

    println!();
    println!("Screen capture initialized:");
    println!("  Resolution: {}x{}", info.width, info.height);
    println!("  PipeWire node ID: {}", info.node_id);
    println!();

    // Open output file
    let output_filename = "screen_capture.raw";
    let mut output_file = File::create(output_filename)?;

    // Capture frames
    let capture_duration = Duration::from_secs(5);
    let mut frame_count = 0;
    let mut total_bytes = 0usize;
    let start = Instant::now();

    // Track actual frame dimensions (may differ from info)
    let mut actual_width = 0u32;
    let mut actual_height = 0u32;

    println!("Capturing for {} seconds...", capture_duration.as_secs());
    println!("(Select a window/screen in the portal dialog that appears)");
    println!();

    while start.elapsed() < capture_duration {
        // Try to receive a frame with timeout
        if let Some(frame) = capture.recv_frame_timeout(Duration::from_millis(100)) {
            frame_count += 1;
            total_bytes += frame.data.len();

            // Track dimensions from first frame
            if frame_count == 1 {
                actual_width = frame.width;
                actual_height = frame.height;
                println!(
                    "First frame: {}x{}, stride={}, {} bytes",
                    frame.width,
                    frame.height,
                    frame.stride,
                    frame.data.len()
                );
            }

            // Write raw frame data to file
            output_file.write_all(&frame.data)?;

            // Progress update every 10 frames
            if frame_count % 10 == 0 {
                println!(
                    "  Captured {} frames ({:.1} MB)...",
                    frame_count,
                    total_bytes as f64 / 1_000_000.0
                );
            }
        }
    }

    println!();
    println!("Capture complete!");
    println!("  Frames: {}", frame_count);
    println!("  Duration: {:.1}s", start.elapsed().as_secs_f64());
    println!(
        "  FPS: {:.1}",
        frame_count as f64 / start.elapsed().as_secs_f64()
    );
    println!("  Total size: {:.1} MB", total_bytes as f64 / 1_000_000.0);
    println!();
    println!("Output saved to: {}", output_filename);
    println!();

    if actual_width > 0 && actual_height > 0 {
        println!("To play the raw video:");
        println!(
            "  ffplay -f rawvideo -pixel_format bgra -video_size {}x{} {}",
            actual_width, actual_height, output_filename
        );
        println!();
        println!("To convert to MP4:");
        println!(
            "  ffmpeg -f rawvideo -pixel_format bgra -video_size {}x{} -framerate {:.0} \\",
            actual_width,
            actual_height,
            frame_count as f64 / start.elapsed().as_secs_f64()
        );
        println!(
            "         -i {} -c:v libx264 -pix_fmt yuv420p screen_capture.mp4",
            output_filename
        );
    }

    Ok(())
}
