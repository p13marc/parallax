//! # H.264 Codec
//!
//! Encode video frames to H.264 using OpenH264.
//!
//! ```text
//! [VideoTestSrc] → [H264Encoder] → [FileSink]
//! ```
//!
//! Run: `cargo run --example 14_h264 --features h264`

use parallax::element::DynAsyncElement;
use parallax::elements::codec::OpenH264Encoder;
use parallax::elements::{FileSink, VideoTestSrc};
use parallax::error::Result;
use parallax::pipeline::Pipeline;
use tempfile::tempdir;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== H.264 Encoding Pipeline ===\n");

    let dir = tempdir()?;
    let output_path = dir.path().join("output.h264");

    let mut pipeline = Pipeline::new();

    // Video test source: 320x240, 30 frames
    let src = pipeline.add_node(
        "videotestsrc",
        DynAsyncElement::new_box(VideoTestSrc::new(320, 240, 30)),
    );

    // H.264 encoder
    let encoder = pipeline.add_node(
        "h264enc",
        DynAsyncElement::new_box(OpenH264Encoder::new(320, 240)?),
    );

    // File sink
    let sink = pipeline.add_node(
        "filesink",
        DynAsyncElement::new_box(FileSink::new(&output_path)),
    );

    pipeline.link(src, encoder)?;
    pipeline.link(encoder, sink)?;

    println!("Encoding 30 frames at 320x240...");
    pipeline.run().await?;

    let file_size = std::fs::metadata(&output_path)?.len();
    println!("\nOutput: {:?}", output_path);
    println!("Size: {} bytes", file_size);

    Ok(())
}
