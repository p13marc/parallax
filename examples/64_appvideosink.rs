//! # AppVideoSink — rendering video yourself
//!
//! `AutoVideoSink` opens its own window, which means it owns the process's
//! one winit event loop. A GUI application owns that loop itself, so it needs
//! a sink that hands frames *out* instead of drawing them.
//!
//! This example is the smallest possible host: no window, no GPU. It drains
//! the mailbox exactly as a real renderer's redraw loop would — ask the clock
//! for running time, take the frame that is due, ask when to come back — and
//! checksums the pixels instead of uploading them.
//!
//! ```text
//! [VideoTestSrc] → [AppVideoSink] ⇄ mailbox ⇄ [this program]
//! ```
//!
//! Run: `cargo run --example 64_appvideosink`

use std::time::Duration;

use parallax::elements::{AppVideoSink, VideoPattern, VideoTestSrc};
use parallax::error::Result;
use parallax::pipeline::{Executor, Pipeline};

/// Frames a real renderer holds beyond the mailbox: the one on screen.
const HOST_RETENTION: usize = 1;

#[tokio::main]
async fn main() -> Result<()> {
    println!("AppVideoSink Example");
    println!("====================\n");

    let source = VideoTestSrc::new()
        .with_pattern(VideoPattern::MovingBall)
        .with_resolution(320, 240)
        .with_framerate(25, 1)
        .with_num_frames(50);

    let sink = AppVideoSink::new()
        .with_sync(true)
        .with_host_retention(HOST_RETENTION);

    // Before `start`, like every handle in parallax: `Executor::start` moves
    // the element into its task and there is no getting at it afterwards.
    let frames = sink.mailbox();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", source);
    let dst = pipeline.add_async_sink("display", sink);
    pipeline.link(src, dst)?;

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline)?;

    // The host's "redraw loop". A real one is driven by the compositor; this
    // one polls, which is the same shape with a worse clock.
    let mut shown = 0u64;
    let mut checksum = 0u64;
    loop {
        // `clock()` is `None` until the first buffer — an element only sees
        // the pipeline clock through its consume context, so there is nothing
        // to hand out before then. Show frames on arrival until it appears.
        let now = frames
            .clock()
            .map(|c| c.now())
            .unwrap_or(parallax::clock::ClockTime::NONE);

        while let Some(paced) = frames.take_due(now, Duration::from_millis(40)) {
            let bytes = paced.frame.data.as_bytes();
            checksum = checksum
                .wrapping_mul(31)
                .wrapping_add(bytes.iter().map(|b| *b as u64).sum::<u64>());
            shown += 1;
            if shown.is_multiple_of(10) {
                println!(
                    "  {shown:3} frames · {}x{} {:?} · due at {}",
                    paced.frame.width, paced.frame.height, paced.frame.format, paced.target
                );
            }
        }

        if frames.end_reason().is_some() && !frames.has_frame() {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }

    let stats = frames.stats();
    println!("\nend: {:?}", frames.end_reason());
    println!("shown {shown} frames, checksum {checksum:#x}");
    println!(
        "dropped: {} late, {} stale, {} because this host was too slow",
        stats.dropped_late, stats.dropped_stale, stats.dropped_full
    );

    handle.stop();
    let _ = handle.wait().await;
    Ok(())
}
