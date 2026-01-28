//! AutoVideoSink example - Display video in a window.
//!
//! This example demonstrates using the autovideosink element to display
//! video frames in a window, just like GStreamer's autovideosink.
//!
//! Run with: `cargo run --example 24_autovideosink --features display`

use parallax::error::Result;
use parallax::pipeline::Pipeline;

#[tokio::main]
async fn main() -> Result<()> {
    println!("AutoVideoSink Example");
    println!("=====================\n");
    println!("Displaying video test pattern. Close the window to exit.\n");

    // Simple pipeline: videotestsrc generates frames, autovideosink displays them
    let mut pipeline = Pipeline::parse("videotestsrc ! autovideosink")?;
    pipeline.run().await?;

    println!("\nDone!");
    Ok(())
}
