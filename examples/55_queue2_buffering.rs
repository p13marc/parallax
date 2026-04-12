//! # Queue2 Network Buffering
//!
//! Demonstrates the advanced Queue2 element with stream buffering mode.
//! Queue2 pauses at low watermark and resumes at high watermark,
//! posting buffering progress messages to the pipeline bus.
//!
//! Modes available:
//! - Stream: in-memory ring buffer with watermarks (shown here)
//! - Download: file-backed progressive download
//! - Timeshift: circular file buffer for DVR rewind
//!
//! ```text
//! [NullSource] → [Queue2 (stream mode)] → [NullSink]
//! ```
//!
//! Run: `cargo run --example 55_queue2_buffering`

use parallax::elements::flow::Queue2;
use parallax::elements::{NullSink, NullSource};
use parallax::pipeline::{Executor, Pipeline};

#[tokio::main]
async fn main() -> parallax::error::Result<()> {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source("src", NullSource::new(30));

    // Queue2 in stream mode: 4KB buffer, pause at 10%, resume at 90%
    let queue = Queue2::stream(4096).with_watermarks(10, 90);
    let q = pipeline.add_filter("queue2", queue);

    let sink = pipeline.add_sink("sink", NullSink::new());

    pipeline.link(src, q)?;
    pipeline.link(q, sink)?;

    // Take bus to observe buffering messages
    let mut bus = pipeline.take_bus().expect("bus");

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline)?;
    handle.wait().await?;

    // Check for buffering messages
    let mut buffering_count = 0;
    while let Some(msg) = bus.poll() {
        if let parallax::pipeline::bus::MessageKind::Buffering {
            percent, mode, ..
        } = &msg.kind
        {
            buffering_count += 1;
            if buffering_count <= 5 {
                println!("[Bus] Buffering: {}% ({:?})", percent, mode);
            }
        }
    }

    if buffering_count > 5 {
        println!("[Bus] ... and {} more buffering messages", buffering_count - 5);
    }
    println!("\nTotal buffering messages: {}", buffering_count);

    Ok(())
}
