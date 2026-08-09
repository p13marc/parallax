//! RTSP playback in a native window.
//!
//! Connects to an RTSP server and shows the live video in a window:
//!
//! ```text
//! RtspSrc -> H264Decoder -> VideoConvert (I420->RGBA) -> AutoVideoSink
//! ```
//!
//! A connected `RtspSession` is an `AsyncSource` like any other, so it goes
//! straight into the graph. This example used to spawn a task, pump the session
//! by hand, skip to the first keyframe, and shovel the buffers through an
//! `AppSrc` — about forty lines of bridge that every caller was writing. The
//! keyframe skip now lives in the session (`skip_until_keyframe`, on by default:
//! joining a live stream lands mid-GOP).
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

use parallax::converters::PixelFormat;
use parallax::elements::transform::VideoConvertElement;
use parallax::elements::{AutoVideoSink, H264Decoder, MediaType, RtspSrc, RtspTransport};
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

    let session = RtspSrc::new(&url)
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
        .unwrap_or_else(|| "not in the SDP — waiting for the first SPS".into());
    println!("Connected: h264 {dims}. Close the window to stop.\n");

    // The session is about to be moved into the pipeline, so take the metadata
    // handle now. A camera whose SDP carries no geometry announces it in the
    // first in-band SPS instead, which arrives once frames start flowing.
    let info = session.stream_info_handle();
    let index = video.index;

    // Build the playback pipeline. VideoConvert gets I420 pinned (the decoder's
    // output); dimensions come from the SDP if present, otherwise from the
    // decoder's per-buffer width/height metadata.
    let mut convert = VideoConvertElement::new()
        .with_input_format(PixelFormat::I420)
        .with_output_format(PixelFormat::Rgba);
    if let Some((w, h)) = video.dimensions {
        convert = convert.with_size(w, h);
    }

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_async_source("rtsp", session);
    let dec = pipeline.add_filter("decode", H264Decoder::new()?);
    let cvt = pipeline.add_filter("convert", convert);
    let sink = pipeline.add_sink("display", AutoVideoSink::new().with_title("Parallax RTSP"));
    pipeline.link(src, dec)?;
    pipeline.link(dec, cvt)?;
    pipeline.link(cvt, sink)?;

    // Report the real geometry as soon as it is known. For an SDP that already
    // carried it this resolves at once; otherwise it waits one keyframe. The
    // task ends by itself when the session drops.
    if video.dimensions.is_none() {
        tokio::spawn(async move {
            if let Some((w, h)) = info.wait_for_dimensions(index).await {
                println!("Stream geometry from the in-band SPS: {w}x{h}");
            }
        });
    }

    let result = pipeline.run().await;

    // Window close surfaces as a sink error — that's a normal way to quit.
    match result {
        Ok(()) => println!("Stream finished."),
        Err(e) if format!("{e}").contains("window closed") => println!("Window closed."),
        Err(e) => return Err(e),
    }
    Ok(())
}
