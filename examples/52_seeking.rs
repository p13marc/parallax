//! # File Seeking & Position Queries
//!
//! Demonstrates seeking in a file-based pipeline and querying
//! position/duration. FileSrc supports byte-based seeking.
//!
//! ```text
//! [FileSrc] → seek to offset → read → [NullSink]
//! ```
//!
//! Run: `cargo run --example 52_seeking`

use parallax::elements::{FileSrc, NullSink};
use parallax::pipeline::Pipeline;
use std::io::Write;

#[tokio::main]
async fn main() -> parallax::error::Result<()> {
    // Create a temp file with known content
    let mut temp = tempfile::NamedTempFile::new().unwrap();
    let content = b"AAAA_BBBB_CCCC_DDDD_EEEE";
    temp.write_all(content).unwrap();
    temp.flush().unwrap();

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source("src", FileSrc::open(temp.path())?);
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink)?;

    // Query seekability
    let seekable = pipeline.query_seekable();
    println!(
        "Seekable: {} (range: {} - {} bytes)",
        seekable.seekable, seekable.start, seekable.stop
    );

    // Query duration
    if let Some(dur) = pipeline.query_duration() {
        println!("Duration: {:?} = {:?}", dur.format, dur.duration);
    }

    // Query initial position
    if let Some(pos) = pipeline.query_position() {
        println!("Position before seek: {:?}", pos.position);
    }

    // Seek to byte 10 ("CCCC_DDDD_EEEE")
    let handled = pipeline.seek_bytes(10)?;
    println!("Seek to byte 10: handled={}", handled);

    // Query position after seek
    if let Some(pos) = pipeline.query_position() {
        println!("Position after seek: {:?}", pos.position);
    }

    // Run the pipeline from the seeked position
    pipeline.run().await?;

    println!("Done!");
    Ok(())
}
