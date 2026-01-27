//! # Image Codecs
//!
//! Encode and decode images using PNG codec.
//!
//! ```text
//! [VideoTestSrc] → [PngEncoder] → [PngDecoder] → [Sink]
//! ```
//!
//! Run: `cargo run --example 13_image --features image-codecs`

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ConsumeContext, DynAsyncElement, Sink, SinkAdapter};
use parallax::elements::VideoTestSrc;
use parallax::elements::codec::{PngDecoder, PngEncoder};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;

struct ImageSink {
    count: u32,
}

impl Sink for ImageSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        self.count += 1;
        // After decode, we have raw RGB data
        let bytes = ctx.input();
        println!(
            "[Sink] Frame {}: {} bytes (decoded RGB)",
            self.count,
            bytes.len()
        );
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Image Codec Pipeline ===\n");

    let mut pipeline = Pipeline::new();

    // Video test source: 64x64 RGB frames
    let src = pipeline.add_node(
        "videotestsrc",
        DynAsyncElement::new_box(VideoTestSrc::new(64, 64, 3)), // 3 frames
    );

    // PNG encoder
    let encoder = pipeline.add_node("png_enc", DynAsyncElement::new_box(PngEncoder::new(64, 64)));

    // PNG decoder
    let decoder = pipeline.add_node("png_dec", DynAsyncElement::new_box(PngDecoder::new()));

    // Sink
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(ImageSink { count: 0 })),
    );

    pipeline.link(src, encoder)?;
    pipeline.link(encoder, decoder)?;
    pipeline.link(decoder, sink)?;

    pipeline.run().await?;

    println!("\nPipeline complete!");
    Ok(())
}
