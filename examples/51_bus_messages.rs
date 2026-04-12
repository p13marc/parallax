//! # Pipeline Bus Messages
//!
//! Demonstrates three patterns for consuming pipeline bus messages:
//!
//! 1. **Sync polling** — `bus.poll()` for non-blocking draining
//! 2. **Async Stream** — `bus.into_stream()` with `StreamExt` / `select!`
//! 3. **Message handler** — `pipeline.run_with_bus()` callback
//!
//! ```text
//! [NullSource] → [NullSink]
//!       ↓ bus messages ↓
//!   [Application consumes messages]
//! ```
//!
//! Run: `cargo run --example 51_bus_messages`

use futures::StreamExt;
use parallax::elements::{NullSink, NullSource};
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::{Executor, Pipeline};

#[tokio::main]
async fn main() -> parallax::error::Result<()> {
    // ================================================================
    // Pattern 1: Sync polling (after pipeline finishes)
    // ================================================================
    println!("=== Pattern 1: Sync Polling ===\n");
    {
        let mut pipeline = Pipeline::new();
        let src = pipeline.add_source("src", NullSource::new(5));
        let sink = pipeline.add_sink("sink", NullSink::new());
        pipeline.link(src, sink)?;

        let mut bus = pipeline.take_bus().expect("bus");
        let executor = Executor::new();
        let handle = executor.start(&mut pipeline)?;
        handle.wait().await?;

        // Drain messages synchronously
        while let Some(msg) = bus.poll() {
            println!("  [{}] {}", msg.source, msg.kind);
        }
    }

    // ================================================================
    // Pattern 2: Async Stream (use with select!, StreamExt)
    // ================================================================
    println!("\n=== Pattern 2: Async Stream ===\n");
    {
        let mut pipeline = Pipeline::new();
        let src = pipeline.add_source("src", NullSource::new(5));
        let sink = pipeline.add_sink("sink", NullSink::new());
        pipeline.link(src, sink)?;

        let bus = pipeline.take_bus().expect("bus");
        let executor = Executor::new();
        let handle = executor.start(&mut pipeline)?;
        handle.wait().await?;

        // Consume as an async Stream
        let mut stream = bus.into_stream();
        while let Some(msg) = stream.next().await {
            let is_eos = matches!(msg.kind, MessageKind::Eos);
            println!("  [{}] {}", msg.source, msg.kind);
            if is_eos {
                break;
            }
        }
    }

    // ================================================================
    // Pattern 3: Message handler callback
    // ================================================================
    println!("\n=== Pattern 3: run_with_bus() ===\n");
    {
        let mut pipeline = Pipeline::new();
        let src = pipeline.add_source("src", NullSource::new(5));
        let sink = pipeline.add_sink("sink", NullSink::new());
        pipeline.link(src, sink)?;

        pipeline
            .run_with_bus(|msg| {
                println!("  [{}] {}", msg.source, msg.kind);
                true // continue
            })
            .await?;
    }

    println!("\nDone!");
    Ok(())
}
