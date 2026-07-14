//! # H.264 Codec
//!
//! Encode video frames to H.264 using OpenH264.
//!
//! ```text
//! [VideoTestSrc] → [VideoConvert(→I420)] → [H264Encoder] → [FileSink]
//! ```
//!
//! Run: `cargo run --example 14_h264 --features h264`

use parallax::converters::PixelFormat as ConverterPixelFormat;
use parallax::elements::codec::{H264Encoder, H264EncoderConfig};
use parallax::elements::transform::VideoConvertElement;
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
    let src = pipeline.add_source(
        "videotestsrc",
        VideoTestSrc::new()
            .with_resolution(320, 240)
            .with_num_frames(30),
    );

    // VideoTestSrc emits RGB24; H.264 encodes I420 planes. Without this the
    // encoder would read RGB bytes as if they were Y/U/V — which is exactly what
    // this example used to do, silently.
    let convert = pipeline.add_filter(
        "videoconvert",
        VideoConvertElement::new().with_output_format(ConverterPixelFormat::I420),
    );

    // H.264 encoder. No dimensions: it encodes whatever each frame declares.
    let encoder = pipeline.add_filter("h264enc", H264Encoder::new(H264EncoderConfig::new())?);

    // File sink
    let sink = pipeline.add_sink("filesink", FileSink::new(&output_path));

    pipeline.link(src, convert)?;
    pipeline.link(convert, encoder)?;
    pipeline.link(encoder, sink)?;

    println!("Encoding 30 frames at 320x240...");
    pipeline.run().await?;

    let file_size = std::fs::metadata(&output_path)?.len();
    println!("\nOutput: {:?}", output_path);
    println!("Size: {} bytes", file_size);

    Ok(())
}
