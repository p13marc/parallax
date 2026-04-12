//! # Tracer Framework
//!
//! Demonstrates the pluggable tracer framework for pipeline debugging.
//! Tracers observe pipeline behavior without modifying data flow.
//!
//! Built-in tracers:
//! - LatencyTracer: per-element processing time
//! - FramerateTracer: buffers/sec at each element
//! - DropTracer: counts dropped buffers
//!
//! ```text
//! [NullSource] → [PassThrough] → [NullSink]
//!       ↑ tracers observe all buffer flow ↑
//! ```
//!
//! Run: `cargo run --example 54_tracers`
//!
//! Or activate via environment variable:
//! ```bash
//! PARALLAX_TRACERS="latency;framerate" cargo run --example 54_tracers
//! ```

use parallax::elements::{NullSink, NullSource, PassThrough};
use parallax::pipeline::tracer::{FramerateTracer, LatencyTracer, TracerRegistry};
use parallax::pipeline::{Executor, Pipeline};

#[tokio::main]
async fn main() -> parallax::error::Result<()> {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source("src", NullSource::new(100));
    let xfm = pipeline.add_filter("transform", PassThrough::new());
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, xfm)?;
    pipeline.link(xfm, sink)?;

    // Set up tracers programmatically
    let registry = TracerRegistry::new();
    registry.add(Box::new(LatencyTracer::new()));
    registry.add(Box::new(FramerateTracer::new()));
    pipeline.set_tracer_registry(registry.clone());

    println!("Running pipeline with tracers...\n");

    // Run the pipeline
    let executor = Executor::new();
    executor.run(&mut pipeline).await?;

    // Print tracer reports
    for (name, report) in registry.reports() {
        println!("=== {} ===", name);
        println!("{}", report);
    }

    // Also show pipeline stats snapshot
    let stats = pipeline.stats_snapshot();
    println!("Pipeline: {} elements, {} links, state={:?}",
        stats.element_count, stats.link_count, stats.state);

    Ok(())
}
