//! Screen capture to MP4 video file using Pipeline.
//!
//! This example captures screen content and encodes it to an MP4 file
//! that can be played in VLC, mpv, or any standard video player.
//!
//! Run with:
//!   cargo run --example 46_screen_capture --features "screen-capture,h264,mp4-demux,simd-colorspace"
//!
//! For non-interactive capture (using restore token from previous session):
//!   cargo run --example 46_screen_capture --features "screen-capture,h264,mp4-demux,simd-colorspace" -- --token "<your-token>"
//!
//! Requirements:
//! - XDG Desktop Portal service running
//! - PipeWire session manager
//! - Portal backend (xdg-desktop-portal-gnome, xdg-desktop-portal-kde, etc.)
//!
//! ## How Restore Tokens Work
//!
//! When you first capture, the portal prompts you to select a screen/window.
//! After capture completes, this example prints a "restore token" - a string that
//! grants permission to capture the same source again without prompting.
//!
//! Save that token and pass it with `--token` to skip the permission dialog:
//!
//! ```bash
//! # First run - shows permission dialog, prints token at the end
//! cargo run --example 46_screen_capture --features "screen-capture,h264,mp4-demux,simd-colorspace"
//!
//! # Subsequent runs - no dialog (if token is still valid)
//! cargo run --example 46_screen_capture --features "screen-capture,h264,mp4-demux,simd-colorspace" -- --token "your_token_here"
//! ```
//!
//! Note: Tokens may expire or become invalid when the captured window closes,
//! the system reboots, or the portal revokes permission.

use std::time::Instant;

use parallax::converters::PixelFormat;
use parallax::elements::codec::{H264Encoder, H264EncoderConfig};
use parallax::elements::device::{CaptureSourceType, ScreenCaptureConfig, ScreenCaptureSrc};
use parallax::elements::io::FileSink;
use parallax::elements::mux::{Mp4MuxTransform, Mp4MuxTransformConfig};
use parallax::elements::transform::VideoConvertElement;
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::pipeline::Pipeline;
use parallax::pipeline::flow::FlowPolicy;

fn parse_args() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--token" && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
        i += 1;
    }
    None
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter("parallax=debug,warn")
        .init();

    let restore_token = parse_args();

    println!("Screen Capture to MP4 Pipeline Example");
    println!("======================================");
    println!();

    if let Some(ref token) = restore_token {
        println!("Using restore token: {}...", &token[..token.len().min(20)]);
        println!("(Will attempt non-interactive capture)");
    } else {
        println!("No restore token provided.");
        println!("This will prompt you to select a screen or window to capture.");
        println!();
        println!("Tip: After capture, a restore token will be printed.");
        println!("     Use --token <token> to capture without prompting next time.");
    }
    println!();

    // Capture settings
    let capture_duration_seconds = 5;
    let framerate = 30.0;
    let max_frames = (capture_duration_seconds as f32 * framerate) as u32;

    // Create screen capture configuration with optional restore token
    let mut capture_config = ScreenCaptureConfig::default()
        .with_source_type(CaptureSourceType::Any)
        .with_cursor(true)
        .with_max_frames(max_frames)
        .with_flow_policy(FlowPolicy::Drop {
            log_drops: true,
            max_consecutive: None,
        });

    // If we have a restore token, use it for non-interactive capture
    if let Some(token) = restore_token {
        capture_config = capture_config.with_restore_token(token);
    }

    // Create the pipeline elements
    // 1. Screen capture source (BGRA format from PipeWire)
    let capture = ScreenCaptureSrc::new(capture_config);

    // Screen capture typically returns 1920x1080 (full HD)
    // Note: The actual size is determined at runtime by the portal/PipeWire
    let capture_width = 1920;
    let capture_height = 1080;

    // 2. Video converter: BGRA -> I420 (YUV420) for H.264 encoding
    // BGRx from PipeWire is treated as BGRA (alpha ignored)
    let converter = VideoConvertElement::new()
        .with_input_format(PixelFormat::Bgra)
        .with_output_format(PixelFormat::I420)
        .with_size(capture_width, capture_height);

    // 3. H.264 encoder with multi-threading enabled
    let encoder_config = H264EncoderConfig::new(1920, 1080)
        .frame_rate(framerate)
        .bitrate(4_000_000) // 4 Mbps for good quality screen capture
        .threads(0); // 0 = auto-detect (uses all available cores)
    let encoder = H264Encoder::new(encoder_config)?;

    // 4. MP4 muxer (transform, not sink - outputs complete MP4 on flush)
    let mp4_config = Mp4MuxTransformConfig::new(1920, 1080).with_framerate(framerate);
    let mp4_mux = Mp4MuxTransform::new(mp4_config);

    // 5. File sink to write the MP4 data
    let output_filename = "screen_capture.mp4";
    let file_sink = FileSink::new(output_filename);

    // Build the pipeline with arena for 1920x1080 BGRA frames
    // BGRA = 4 bytes per pixel, 1920x1080 = ~8.3 MB per frame
    // 200 slots provides headroom for capture/encode rate differences
    let frame_size = 1920 * 1080 * 4;
    let arena = SharedArena::new(frame_size, 200)?;

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source_with_arena("screen_capture", capture, arena);
    let convert = pipeline.add_filter("videoconvert", converter);
    let enc = pipeline.add_filter("h264enc", encoder);
    let mux = pipeline.add_filter("mp4mux", mp4_mux);
    let sink = pipeline.add_sink("filesink", file_sink);

    pipeline.link(src, convert)?;
    pipeline.link(convert, enc)?;
    pipeline.link(enc, mux)?;
    pipeline.link(mux, sink)?;

    println!("Pipeline: ScreenCapture -> VideoConvert -> H264Encoder -> Mp4Mux -> FileSink");
    println!();
    println!("Acceleration enabled:");
    #[cfg(feature = "simd-colorspace")]
    println!("  - VideoConvert: SIMD (AVX2/SSE4.1/NEON)");
    #[cfg(not(feature = "simd-colorspace"))]
    println!("  - VideoConvert: Scalar (enable simd-colorspace for acceleration)");
    println!("  - H264Encoder: OpenH264 multi-threaded (auto-detect cores)");
    println!();
    println!(
        "Capturing {} seconds ({} frames at {} fps)...",
        capture_duration_seconds, max_frames, framerate
    );
    println!();

    // Run the pipeline - it will stop when max_frames is reached
    let start = Instant::now();

    match pipeline.run().await {
        Ok(()) => {
            let elapsed = start.elapsed();
            println!();
            println!("Capture complete!");
            println!(
                "Total time: {:.2}s ({:.1} fps effective)",
                elapsed.as_secs_f64(),
                max_frames as f64 / elapsed.as_secs_f64()
            );
            println!();
            println!(
                "Note: The restore token was logged above (look for 'Screen capture restore token')."
            );
            println!("      Copy that token and use --token <token> for non-interactive capture.");
        }
        Err(e) => {
            println!();
            println!("Pipeline error: {}", e);
        }
    }

    println!();
    println!("Output saved to: {}", output_filename);
    println!();
    println!("To play:");
    println!("  mpv {}", output_filename);
    println!("  vlc {}", output_filename);
    println!("  ffplay {}", output_filename);

    Ok(())
}
