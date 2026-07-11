//! RTSP playback in a native window.
//!
//! Connects to an RTSP server and shows the live video in a window:
//! an `RtspSession` feeds H.264 access units into an `AppSrc`, and the
//! pipeline decodes and displays them:
//!
//! ```text
//! RtspSession --> AppSrc -> H264Decoder -> VideoConvert (I420->RGBA) -> AutoVideoSink
//! (tokio task)  [pipeline ....................................................]
//! ```
//!
//! This is the receive side of the zensight media-plane design: a network
//! session bridged into a parallax pipeline through `AppSrc`.
//!
//! Run with:
//!
//! ```text
//! cargo run --example 58_rtsp_display --features "rtsp,h264,display" -- \
//!     rtsp://127.0.0.1:8554/stream
//! ```
//!
//! Serve a local test stream first (no VLC needed):
//!
//! ```text
//! just rtsp-server        # = ./scripts/rtsp_test_server.py
//! ```
//!
//! Close the window (or Ctrl-C) to stop.

use std::time::Duration;

use parallax::converters::PixelFormat;
use parallax::elements::transform::VideoConvertElement;
use parallax::elements::{AppSrc, AutoVideoSink, H264Decoder, MediaType, RtspSrc, RtspTransport};
use parallax::error::{Error, Result};
use parallax::pipeline::Pipeline;

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("parallax=info".parse().unwrap()),
        )
        .init();

    let url = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "rtsp://127.0.0.1:8554/stream".to_string());

    println!("RTSP Display");
    println!("============\n");
    println!("Connecting to {url} ...");

    let mut session = RtspSrc::new(&url)
        .with_transport(RtspTransport::TcpInterleaved)
        .video_only()
        .connect()
        .await?;

    let video = session
        .streams()
        .iter()
        .find(|s| s.media_type == MediaType::Video)
        .cloned()
        .ok_or_else(|| Error::Element("server advertises no video stream".into()))?;
    if video.codec != "h264" {
        return Err(Error::Element(format!(
            "this example decodes H.264 only, server offers '{}'",
            video.codec
        )));
    }

    let dims = video
        .dimensions
        .map(|(w, h)| format!("{w}x{h}"))
        .unwrap_or_else(|| "unknown (will use decoder metadata)".into());
    println!("Connected: h264 {dims}. Close the window to stop.\n");

    // Build the playback pipeline. VideoConvert gets I420 pinned (the decoder's
    // output); dimensions come from the SDP if present, otherwise from the
    // decoder's per-buffer width/height metadata.
    let appsrc = AppSrc::with_max_buffers(32).with_name("rtsp");
    let handle = appsrc.handle();

    let mut convert = VideoConvertElement::new()
        .with_input_format(PixelFormat::I420)
        .with_output_format(PixelFormat::Rgba);
    if let Some((w, h)) = video.dimensions {
        convert = convert.with_size(w, h);
    }

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", appsrc);
    let dec = pipeline.add_filter("decode", H264Decoder::new()?);
    let cvt = pipeline.add_filter("convert", convert);
    let sink = pipeline.add_sink("display", AutoVideoSink::new().with_title("Parallax RTSP"));
    pipeline.link(src, dec)?;
    pipeline.link(dec, cvt)?;
    pipeline.link(cvt, sink)?;

    // Feed RTSP frames into the pipeline. Skip until the first keyframe so the
    // decoder starts on an IDR (joining a live stream lands mid-GOP).
    let feeder = tokio::spawn(async move {
        let mut saw_keyframe = false;
        loop {
            match session.next_frame().await {
                Ok(Some(frame)) => {
                    if !frame.is_video() {
                        continue;
                    }
                    if !saw_keyframe {
                        if !frame.buffer().metadata().is_keyframe() {
                            continue;
                        }
                        saw_keyframe = true;
                    }
                    if handle
                        .push_buffer_timeout(frame.into_buffer(), Some(Duration::from_secs(5)))
                        .is_err()
                    {
                        // Pipeline gone (window closed) or wedged; stop feeding.
                        break;
                    }
                }
                Ok(None) => {
                    println!("Server ended the stream.");
                    handle.end_stream();
                    break;
                }
                Err(e) => {
                    eprintln!("RTSP error: {e}");
                    handle.end_stream();
                    break;
                }
            }
        }
    });

    let result = pipeline.run().await;
    feeder.abort();

    // Window close surfaces as a sink error — that's a normal way to quit.
    match result {
        Ok(()) => println!("Stream finished."),
        Err(e) if format!("{e}").contains("window closed") => println!("Window closed."),
        Err(e) => return Err(e),
    }
    Ok(())
}
