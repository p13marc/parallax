//! Video test source generating pattern frames.
//!
//! Run with: cargo run --example 15_video_testsrc

use parallax::element::{ConsumeContext, DynAsyncElement, Sink, SinkAdapter, SourceAdapter};
use parallax::elements::{PixelFormat, VideoPattern, VideoTestSrc};
use parallax::error::Result;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

struct FrameCounter {
    count: Arc<AtomicU64>,
    width: u32,
    height: u32,
}

impl Sink for FrameCounter {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let n = self.count.fetch_add(1, Ordering::Relaxed);
        let expected_size = (self.width * self.height * 4) as usize; // RGBA32
        println!(
            "Frame {}: {} bytes (expected {})",
            n,
            ctx.len(),
            expected_size
        );
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let width = 320;
    let height = 240;
    let num_frames = 5;

    println!(
        "VideoTestSrc: {}x{} SMPTE bars, {} frames",
        width, height, num_frames
    );
    println!();

    let counter = Arc::new(AtomicU64::new(0));

    let src = VideoTestSrc::new()
        .with_resolution(width, height)
        .with_pattern(VideoPattern::SmpteColorBars)
        .with_pixel_format(PixelFormat::Rgba32)
        .with_num_frames(num_frames);

    let mut pipeline = Pipeline::new();

    let src_node = pipeline.add_node(
        "videotestsrc",
        DynAsyncElement::new_box(SourceAdapter::new(src)),
    );
    let sink_node = pipeline.add_node(
        "counter",
        DynAsyncElement::new_box(SinkAdapter::new(FrameCounter {
            count: counter.clone(),
            width,
            height,
        })),
    );

    pipeline.link(src_node, sink_node)?;
    pipeline.run().await?;

    println!();
    println!("Generated {} frames", counter.load(Ordering::Relaxed));

    Ok(())
}
