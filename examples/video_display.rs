//! Video display example using VideoTestSrc and IcedVideoSink.
//!
//! This example demonstrates displaying video test patterns in a GUI window.
//! It creates a VideoTestSrc that generates test patterns and pipes them
//! to an IcedVideoSink for display.
//!
//! Run with: cargo run --example video_display --features iced-sink
//!
//! Controls:
//! - Close the window to exit

use parallax::element::{Sink, Source};
use parallax::elements::{
    IcedVideoSink, IcedVideoSinkConfig, InputPixelFormat, VideoPattern, VideoTestSrc,
};
use std::thread;
use std::time::Duration;

fn main() -> iced::Result {
    println!("=== Video Display Example ===");
    println!("Displaying SMPTE color bars at 640x480 @ 30fps");
    println!("Close the window to exit.\n");

    // Configuration
    let width = 640;
    let height = 480;
    let pattern = VideoPattern::MovingBall;

    // Create the video source
    let src = VideoTestSrc::new()
        .with_pattern(pattern)
        .with_resolution(width, height)
        .with_framerate(30, 1)
        .with_name("test-pattern");

    // Create the video sink with stats overlay
    let config = IcedVideoSinkConfig {
        title: format!("Parallax Video - {:?}", pattern),
        width,
        height,
        show_stats: true,
        pixel_format: InputPixelFormat::Rgb24, // VideoTestSrc outputs RGB24 by default
    };

    let (sink, handle) = IcedVideoSink::with_config(config);

    // Spawn a thread to produce frames
    let producer = thread::spawn(move || {
        run_producer(src, sink);
    });

    // Run the Iced window (blocks until closed)
    let result = handle.run();

    // Wait for producer to finish
    let _ = producer.join();

    println!("Window closed. Exiting.");
    result
}

fn run_producer(mut src: VideoTestSrc, mut sink: IcedVideoSink) {
    println!("Producer thread started");

    loop {
        // Check if window is still open
        if !sink.is_window_open() {
            println!("Window closed, stopping producer");
            break;
        }

        // Produce a frame
        match src.produce() {
            Ok(Some(buffer)) => {
                // Send to sink
                if let Err(e) = sink.consume(buffer) {
                    println!("Sink error: {}", e);
                    break;
                }
            }
            Ok(None) => {
                // Source exhausted (shouldn't happen with infinite source)
                println!("Source exhausted");
                break;
            }
            Err(e) => {
                println!("Source error: {}", e);
                break;
            }
        }

        // Small sleep to not overwhelm the system
        // The VideoTestSrc already has framerate limiting, but this adds safety
        thread::sleep(Duration::from_micros(100));
    }

    let stats = sink.stats();
    println!(
        "Producer finished. Frames: received={}, displayed={}, dropped={}",
        stats.frames_received, stats.frames_displayed, stats.frames_dropped
    );
}
