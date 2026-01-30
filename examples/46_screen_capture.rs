//! Screen capture to MP4 video file using Pipeline.
//!
//! This example captures screen content and encodes it to an MP4 file
//! that can be played in VLC, mpv, or any standard video player.
//!
//! Run with:
//!   cargo run --example 46_screen_capture --features "screen-capture,h264,mp4-demux"
//!
//! Requirements:
//! - XDG Desktop Portal service running
//! - PipeWire session manager
//! - Portal backend (xdg-desktop-portal-gnome, xdg-desktop-portal-kde, etc.)

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

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter("parallax=debug,warn")
        .init();

    println!("Screen Capture to MP4 Pipeline Example");
    println!("======================================");
    println!();
    println!("This will prompt you to select a screen or window to capture.");
    println!();

    // Capture settings
    // Note: OpenH264 (pure Rust) is slow for 1080p - encoding takes longer than capture
    // The pipeline buffers captured frames and encodes them after capture completes
    let capture_duration_seconds = 5;
    let framerate = 30.0;
    let max_frames = (capture_duration_seconds as f32 * framerate) as u32;

    // Create screen capture configuration with frame limit
    let capture_config = ScreenCaptureConfig {
        source_type: CaptureSourceType::Any,
        show_cursor: true,
        persist_session: false,
        max_frames: Some(max_frames),
        flow_policy: FlowPolicy::Drop {
            log_drops: true,
            max_consecutive: None,
        },
    };

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

    // 3. H.264 encoder
    let encoder_config = H264EncoderConfig::new(1920, 1080)
        .frame_rate(framerate)
        .bitrate(4_000_000); // 4 Mbps for good quality screen capture
    let encoder = H264Encoder::new(encoder_config)?;

    // 4. MP4 muxer (transform, not sink - outputs complete MP4 on flush)
    let mp4_config = Mp4MuxTransformConfig::new(1920, 1080).with_framerate(framerate);
    let mp4_mux = Mp4MuxTransform::new(mp4_config);

    // 5. File sink to write the MP4 data
    let output_filename = "screen_capture.mp4";
    let file_sink = FileSink::new(output_filename);

    // Build the pipeline with a large enough arena for 1920x1080 BGRA frames
    // BGRA = 4 bytes per pixel, 1920x1080 = ~8.3 MB per frame
    //
    // IMPORTANT: OpenH264 (pure Rust) encodes at ~1-2 fps for 1080p
    // PipeWire captures at 30fps, so we need a large buffer to hold
    // captured frames while waiting for encoding.
    //
    // For 5 seconds at 30fps = 150 frames = ~1.2GB buffer needed
    // The pipeline will capture all frames quickly, then spend time encoding.
    let frame_size = 1920 * 1080 * 4;
    let arena = SharedArena::new(frame_size, 200)?; // 200 frame slots (~1.6GB)

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
    println!(
        "Capturing {} seconds ({} frames at {} fps)...",
        capture_duration_seconds, max_frames, framerate
    );
    println!("(Permission dialog will appear)");
    println!();

    // Run the pipeline - it will stop when max_frames is reached
    match pipeline.run().await {
        Ok(()) => {
            println!();
            println!("Capture complete!");
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
