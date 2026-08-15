//! # AV1 Codec
//!
//! Encode video frames to AV1 using rav1e (pure Rust).
//!
//! ```text
//! [AppSrc (I420 frames)] → [Rav1eEncoder] → [FileSink]
//! ```
//!
//! Run: `cargo run --example 15_av1 --features av1-encode`

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::elements::FileSink;
use parallax::elements::app::AppSrc;
use parallax::elements::codec::{Rav1eConfig, Rav1eEncoder};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use tempfile::tempdir;

const WIDTH: usize = 320;
const HEIGHT: usize = 240;
const FRAME_SIZE: usize = WIDTH * HEIGHT * 3 / 2; // I420

/// A synthetic I420 frame: moving luma gradient, flat chroma.
fn i420_frame(arena: &SharedArena, seq: u64) -> Buffer {
    arena.reclaim();
    let mut slot = arena.acquire().expect("arena slot");
    let data = slot.data_mut();
    data.fill(128);
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            data[y * WIDTH + x] = ((x + y) as u8).wrapping_add(seq as u8 * 8);
        }
    }
    let mut metadata = Metadata::from_sequence(seq);
    metadata.pts = parallax::clock::ClockTime::from_millis(seq * 33);
    // Geometry travels in-band: the encoder takes its size from the frame,
    // not from a number handed to a constructor at startup.
    metadata.set_video_dims(
        WIDTH as u32,
        HEIGHT as u32,
        parallax::format::PixelFormat::I420,
    );
    Buffer::new(MemoryHandle::with_len(slot, FRAME_SIZE), metadata)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== AV1 Encoding Pipeline ===\n");

    let dir = tempdir()?;
    let output_path = dir.path().join("output.av1");

    // AV1 encoder: fastest speed preset, low quality for demo speed
    let config = Rav1eConfig::default().speed(10).quantizer(200);
    let encoder = Rav1eEncoder::new(config)?;

    let src = AppSrc::new();
    let src_handle = src.handle();

    let mut pipeline = Pipeline::new();
    let s = pipeline.add_source("appsrc", src);
    let e = pipeline.add_filter("av1enc", encoder);
    let k = pipeline.add_async_sink("filesink", FileSink::new(&output_path));
    pipeline.link(s, e)?;
    pipeline.link(e, k)?;

    // Feed 10 I420 frames, then end the stream.
    let arena = SharedArena::new(FRAME_SIZE, 16)
        .map_err(|e| parallax::error::Error::Element(format!("arena: {e}")))?;
    for seq in 0..10 {
        src_handle.push_buffer(i420_frame(&arena, seq)).await?;
    }
    src_handle.end_stream();

    println!("Encoding 10 frames at {WIDTH}x{HEIGHT} (AV1)...");
    println!("(This may take a moment - AV1 encoding is CPU-intensive)\n");
    pipeline.run().await?;

    let file_size = std::fs::metadata(&output_path)?.len();
    println!("Output: {:?}", output_path);
    println!("Size: {} bytes", file_size);

    Ok(())
}
