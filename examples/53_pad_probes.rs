//! # Pad Probes
//!
//! Demonstrates pad probes for intercepting buffers at any point
//! in the pipeline. Probes can inspect, drop, or count buffers.
//!
//! ```text
//! [NullSource] → (probe here) → [NullSink]
//! ```
//!
//! Run: `cargo run --example 53_pad_probes`

use parallax::elements::{NullSink, NullSource};
use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
use parallax::pipeline::{Executor, Pipeline};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

#[tokio::main]
async fn main() -> parallax::error::Result<()> {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source("src", NullSource::new(20));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink)?;

    // --- Probe 1: Count all buffers ---
    let buffer_count = Arc::new(AtomicU64::new(0));
    let count_clone = buffer_count.clone();

    let pad = PadRef::src(src);
    pipeline.add_probe(pad, ProbeType::BUFFER, move |data| {
        if let ProbeData::Buffer(buf) = data {
            let n = count_clone.fetch_add(1, Ordering::Relaxed) + 1;
            if n <= 3 || n == 20 {
                println!("[Probe] Buffer #{}: {} bytes", n, buf.len());
            } else if n == 4 {
                println!("[Probe] ... (counting remaining buffers)");
            }
        }
        ProbeReturn::Ok // Pass through
    });

    // --- Probe 2: Count total bytes ---
    let total_bytes = Arc::new(AtomicU64::new(0));
    let bytes_clone = total_bytes.clone();

    pipeline.add_probe(pad, ProbeType::BUFFER, move |data| {
        if let ProbeData::Buffer(buf) = data {
            bytes_clone.fetch_add(buf.len() as u64, Ordering::Relaxed);
        }
        ProbeReturn::Ok
    });

    // Run
    let executor = Executor::new();
    executor.run(&mut pipeline).await?;

    println!(
        "\nResults: {} buffers, {} total bytes",
        buffer_count.load(Ordering::Relaxed),
        total_bytes.load(Ordering::Relaxed),
    );

    Ok(())
}
