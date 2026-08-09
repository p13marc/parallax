//! RTSP capture: connect to an RTSP server and dump the video stream to disk.
//!
//! Connects to an RTSP server, prints the streams it advertises, then records
//! the video elementary stream to a file. With the default
//! [`RtspFrameFormat::AnnexB`] framing, H.264/H.265 dumps are self-contained
//! Annex-B bytestreams (SPS/PPS repeated on every keyframe), directly playable:
//!
//! ```text
//! ffplay rtsp_capture.h264      # or: mpv rtsp_capture.h264
//! ```
//!
//! Run with:
//!
//! ```text
//! cargo run --example 57_rtsp_capture --features rtsp -- \
//!     rtsp://127.0.0.1:8554/stream [output.h264] [seconds]
//! ```
//!
//! Serving a test stream yourself (public RTSP test endpoints are all dead or
//! token-gated — run a local server instead):
//!
//! ```text
//! # GStreamer test pattern (no extra install on most distros):
//! just rtsp-server        # = ./scripts/rtsp_test_server.py
//!
//! # VLC (loops a file, transcodes to H.264):
//! cvlc input.mp4 --loop \
//!     --sout '#transcode{vcodec=h264,vb=2000,acodec=none}:rtp{sdp=rtsp://:8554/stream}'
//!
//! # ffmpeg + mediamtx (mediamtx serves rtsp://127.0.0.1:8554/stream):
//! mediamtx & ffmpeg -re -stream_loop -1 -i input.mp4 -c:v libx264 -an \
//!     -f rtsp rtsp://127.0.0.1:8554/stream
//! ```
//!
//! Camera URLs with embedded credentials (`rtsp://user:pass@host/...`) work;
//! they are lifted into RTSP digest/basic auth automatically.
//!
//! Stop early with Ctrl-C; the file is complete up to the last frame written.

use std::io::Write;
use std::time::Duration;

use parallax::elements::{RtspSrc, RtspTransport};
use parallax::error::{Error, Result};

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("parallax=info".parse().unwrap()),
        )
        .init();

    let mut args = std::env::args().skip(1);
    let url = args
        .next()
        .unwrap_or_else(|| "rtsp://127.0.0.1:8554/stream".to_string());
    let output = args
        .next()
        .unwrap_or_else(|| "rtsp_capture.h264".to_string());
    let seconds: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(10);

    println!("RTSP Capture");
    println!("============\n");
    println!("Connecting to {url} ...");

    // TCP-interleaved is the most robust transport (single connection, works
    // through NAT). Video-only keeps the dump a valid elementary stream.
    let mut session = RtspSrc::new(&url)
        .with_transport(RtspTransport::TcpInterleaved)
        .video_only()
        .connect()
        .await?;

    println!("Connected. Streams advertised by the server:\n");
    for s in session.streams() {
        let dims = s
            .dimensions
            .map(|(w, h)| format!("{w}x{h}"))
            .unwrap_or_else(|| "?".into());
        let fps = s
            .framerate
            .map(|f| format!("{f:.1} fps"))
            .unwrap_or_else(|| "? fps".into());
        println!(
            "  [{}] {:?} codec={} {} {} clock={} Hz",
            s.index, s.media_type, s.codec, dims, fps, s.clock_rate
        );
    }

    let video = session
        .streams()
        .iter()
        .find(|s| s.media_type == parallax::elements::MediaType::Video)
        .cloned()
        .ok_or_else(|| Error::Element("server advertises no video stream".into()))?;
    if video.codec != "h264" && video.codec != "h265" {
        println!(
            "\nnote: video codec is '{}' — the dump will be raw frames, not an \
             H.26x bytestream",
            video.codec
        );
    }

    let mut file = std::fs::File::create(&output)?;
    println!("\nRecording to {output} for {seconds}s (Ctrl-C to stop early)...\n");

    let deadline = tokio::time::sleep(Duration::from_secs(seconds));
    tokio::pin!(deadline);
    let mut last_report = std::time::Instant::now();
    // Joining a live stream lands mid-GOP; frames before the first keyframe
    // reference parameter sets and frames we never saw, so skip them.
    let mut saw_keyframe = false;

    loop {
        let frame = tokio::select! {
            f = session.next_frame() => f?,
            _ = &mut deadline => {
                println!("\nCapture window elapsed.");
                break;
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\nInterrupted.");
                break;
            }
        };

        let Some(frame) = frame else {
            println!("\nServer ended the stream.");
            break;
        };
        if !frame.is_video() {
            continue;
        }
        if !saw_keyframe {
            if !frame.buffer().metadata().is_keyframe() {
                continue;
            }
            saw_keyframe = true;
        }
        file.write_all(frame.buffer().as_bytes())?;

        if last_report.elapsed() >= Duration::from_secs(2) {
            let stats = session.stats();
            let elapsed = stats
                .connected_at
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0);
            println!(
                "  {:>5} frames ({} keyframes), {:.2} MiB, {:.1} fps",
                stats.video_frames,
                stats.video_keyframes,
                stats.bytes_received as f64 / (1024.0 * 1024.0),
                stats.video_frames as f64 / elapsed.max(0.001),
            );
            last_report = std::time::Instant::now();
        }
    }

    file.flush()?;

    // Geometry the SDP did not advertise is picked up from the first in-band
    // SPS, so `streams()` can say more now than it did at connect time.
    if video.dimensions.is_none()
        && let Some((w, h)) = session
            .streams()
            .get(video.index)
            .and_then(|s| s.dimensions)
    {
        println!("\nGeometry learned from the in-band SPS: {w}x{h}");
    }

    let stats = session.stats();
    println!(
        "\nDone: {} frames ({} keyframes), {:.2} MiB written to {}",
        stats.video_frames,
        stats.video_keyframes,
        stats.bytes_received as f64 / (1024.0 * 1024.0),
        output
    );
    if video.codec == "h264" || video.codec == "h265" {
        println!("Play it with: ffplay {output}   (or: mpv {output})");
    }

    Ok(())
}
