//! # Pipeline Bus Messages
//!
//! Demonstrates the pipeline message bus for element-to-application
//! communication. The bus carries typed messages (errors, warnings,
//! tags, QoS, buffering, state changes) that applications can poll
//! synchronously or stream asynchronously.
//!
//! ```text
//! [NullSource] → [NullSink]
//!       ↓ bus messages ↓
//!   [Application polls bus]
//! ```
//!
//! Run: `cargo run --example 51_bus_messages`

use parallax::elements::{NullSink, NullSource};
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::{Executor, Pipeline};

#[tokio::main]
async fn main() -> parallax::error::Result<()> {
    // Create pipeline — bus is created automatically
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source("src", NullSource::new(10));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink)?;

    // Take the bus before starting (it moves to PipelineHandle)
    let mut bus = pipeline.take_bus().expect("bus should be available");

    // Start the pipeline
    let executor = Executor::new();
    let handle = executor.start(&mut pipeline)?;

    // Wait for pipeline to finish
    handle.wait().await?;

    // Drain all bus messages
    println!("--- Bus Messages ---");
    while let Some(msg) = bus.poll() {
        match &msg.kind {
            MessageKind::StateChanged { old, new } => {
                println!("[{}] State: {:?} → {:?}", msg.source, old, new);
            }
            MessageKind::Eos => {
                println!("[{}] End of stream", msg.source);
            }
            MessageKind::Error { error, debug } => {
                println!("[{}] ERROR: {} ({:?})", msg.source, error, debug);
            }
            MessageKind::Tag { tags } => {
                println!("[{}] Tags: {}", msg.source, tags);
            }
            other => {
                println!("[{}] {}", msg.source, other);
            }
        }
    }
    println!("--- End ---");

    Ok(())
}
