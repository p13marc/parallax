//! # AV1 Codec
//!
//! Encode video frames to AV1 using rav1e (pure Rust).
//!
//! ```text
//! [VideoTestSrc] → [AV1Encoder] → [FileSink]
//! ```
//!
//! Run: `cargo run --example 15_av1 --features av1-encode`

use parallax::element::DynAsyncElement;
use parallax::elements::codec::{Rav1eEncoder, Rav1eEncoderConfig};
use parallax::elements::{FileSink, VideoTestSrc};
use parallax::error::Result;
use parallax::pipeline::Pipeline;
use tempfile::tempdir;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== AV1 Encoding Pipeline ===\n");

    let dir = tempdir()?;
    let output_path = dir.path().join("output.av1");

    let mut pipeline = Pipeline::new();

    // Video test source: 320x240, 10 frames (AV1 encoding is slow)
    let src = pipeline.add_node(
        "videotestsrc",
        DynAsyncElement::new_box(VideoTestSrc::new(320, 240, 10)),
    );

    // AV1 encoder with speed preset for faster encoding
    let config = Rav1eEncoderConfig {
        width: 320,
        height: 240,
        speed: 10,      // Fastest speed preset
        quantizer: 100, // Lower quality for speed
        ..Default::default()
    };
    let encoder = pipeline.add_node(
        "av1enc",
        DynAsyncElement::new_box(Rav1eEncoder::with_config(config)?),
    );

    // File sink
    let sink = pipeline.add_node(
        "filesink",
        DynAsyncElement::new_box(FileSink::new(&output_path)),
    );

    pipeline.link(src, encoder)?;
    pipeline.link(encoder, sink)?;

    println!("Encoding 10 frames at 320x240 (AV1)...");
    println!("(This may take a moment - AV1 encoding is CPU-intensive)\n");
    pipeline.run().await?;

    let file_size = std::fs::metadata(&output_path)?.len();
    println!("Output: {:?}", output_path);
    println!("Size: {} bytes", file_size);

    Ok(())
}
