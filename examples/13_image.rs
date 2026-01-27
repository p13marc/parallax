//! # Image Codecs
//!
//! Encode and decode images using PNG codec.
//!
//! ```text
//! [VideoTestSrc] → [PngEncoder] → [PngDecoder] → [Sink]
//! ```
//!
//! Run: `cargo run --example 13_image --features image-codecs`

use parallax::element::{ConsumeContext, Sink};
use parallax::elements::VideoTestSrc;
use parallax::elements::codec::{PngDecoder, PngEncoder};
use parallax::error::Result;
use parallax::pipeline::Pipeline;

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

    // Video test source: 64x64 RGB frames (3 frames)
    let src = pipeline.add_source(
        "videotestsrc",
        VideoTestSrc::new().with_size(64, 64).with_num_frames(3),
    );

    // PNG encoder
    let encoder = pipeline.add_filter("png_enc", PngEncoder::new(64, 64));

    // PNG decoder
    let decoder = pipeline.add_filter("png_dec", PngDecoder::new());

    // Sink
    let sink = pipeline.add_sink("sink", ImageSink { count: 0 });

    pipeline.link(src, encoder)?;
    pipeline.link(encoder, decoder)?;
    pipeline.link(decoder, sink)?;

    pipeline.run().await?;

    println!("\nPipeline complete!");
    Ok(())
}
