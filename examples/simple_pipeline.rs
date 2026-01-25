//! Simple dynamic pipeline example.
//!
//! This example demonstrates building and running a pipeline using the
//! dynamic (string-based) API. Pipelines are constructed at runtime and
//! connected through the graph structure.
//!
//! Run with: cargo run --example simple_pipeline

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{Element, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter};
use parallax::error::Result;
use parallax::memory::HeapSegment;
use parallax::metadata::Metadata;
use parallax::pipeline::{Pipeline, PipelineExecutor};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A source that produces N buffers with sequential data.
struct CountingSource {
    current: u64,
    max: u64,
    buffer_size: usize,
}

impl CountingSource {
    fn new(count: u64, buffer_size: usize) -> Self {
        Self {
            current: 0,
            max: count,
            buffer_size,
        }
    }
}

impl Source for CountingSource {
    fn produce(&mut self) -> Result<Option<Buffer<()>>> {
        if self.current >= self.max {
            return Ok(None);
        }

        // Create a buffer
        let segment = Arc::new(HeapSegment::new(self.buffer_size)?);
        let handle = MemoryHandle::from_segment_with_len(segment, self.buffer_size);
        let buffer = Buffer::new(handle, Metadata::from_sequence(self.current));

        self.current += 1;
        Ok(Some(buffer))
    }
}

/// A transform that doubles values in the buffer.
struct Doubler;

impl Element for Doubler {
    fn process(&mut self, buffer: Buffer<()>) -> Result<Option<Buffer<()>>> {
        // In a real implementation, you'd modify the buffer contents
        // For this example, we just pass through
        Ok(Some(buffer))
    }
}

/// A transform that filters buffers based on sequence number.
struct EvenFilter;

impl Element for EvenFilter {
    fn process(&mut self, buffer: Buffer<()>) -> Result<Option<Buffer<()>>> {
        // Only pass buffers with even sequence numbers
        if buffer.metadata().sequence % 2 == 0 {
            Ok(Some(buffer))
        } else {
            Ok(None) // Filter out odd sequences
        }
    }
}

/// A sink that counts and prints received buffers.
struct PrintingSink {
    name: String,
    count: Arc<AtomicU64>,
    verbose: bool,
}

impl PrintingSink {
    fn new(name: &str, count: Arc<AtomicU64>, verbose: bool) -> Self {
        Self {
            name: name.to_string(),
            count,
            verbose,
        }
    }
}

impl Sink for PrintingSink {
    fn consume(&mut self, buffer: Buffer<()>) -> Result<()> {
        let n = self.count.fetch_add(1, Ordering::Relaxed) + 1;
        if self.verbose {
            println!(
                "[{}] Received buffer #{}: seq={}, len={}",
                self.name,
                n,
                buffer.metadata().sequence,
                buffer.len()
            );
        }
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("=== Simple Pipeline Example ===\n");

    // Example 1: Linear pipeline
    println!("1. Linear pipeline: Source -> Transform -> Sink");
    run_linear_pipeline().await?;

    // Example 2: Pipeline with filtering
    println!("\n2. Filtering pipeline: Source -> Filter -> Sink");
    run_filtering_pipeline().await?;

    // Example 3: Branching pipeline (tee)
    println!("\n3. Branching pipeline: Source -> Tee -> [Sink1, Sink2]");
    run_branching_pipeline().await?;

    println!("\n=== All examples completed ===");
    Ok(())
}

async fn run_linear_pipeline() -> Result<()> {
    let mut pipeline = Pipeline::new();

    // Create elements
    let source = pipeline.add_node(
        "counter_source",
        Box::new(SourceAdapter::new(CountingSource::new(10, 64))),
    );

    let transform = pipeline.add_node("doubler", Box::new(ElementAdapter::new(Doubler)));

    let sink_count = Arc::new(AtomicU64::new(0));
    let sink = pipeline.add_node(
        "printing_sink",
        Box::new(SinkAdapter::new(PrintingSink::new(
            "linear",
            sink_count.clone(),
            true,
        ))),
    );

    // Link elements
    pipeline.link(source, transform)?;
    pipeline.link(transform, sink)?;

    // Run the pipeline
    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await?;

    println!(
        "   Total buffers received: {}",
        sink_count.load(Ordering::Relaxed)
    );
    Ok(())
}

async fn run_filtering_pipeline() -> Result<()> {
    let mut pipeline = Pipeline::new();

    // Create elements
    let source = pipeline.add_node(
        "counter_source",
        Box::new(SourceAdapter::new(CountingSource::new(10, 64))),
    );

    let filter = pipeline.add_node("even_filter", Box::new(ElementAdapter::new(EvenFilter)));

    let sink_count = Arc::new(AtomicU64::new(0));
    let sink = pipeline.add_node(
        "printing_sink",
        Box::new(SinkAdapter::new(PrintingSink::new(
            "filter",
            sink_count.clone(),
            true,
        ))),
    );

    // Link elements
    pipeline.link(source, filter)?;
    pipeline.link(filter, sink)?;

    // Run the pipeline
    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await?;

    println!(
        "   Total buffers received (should be 5): {}",
        sink_count.load(Ordering::Relaxed)
    );
    Ok(())
}

async fn run_branching_pipeline() -> Result<()> {
    let mut pipeline = Pipeline::new();

    // Create elements
    let source = pipeline.add_node(
        "counter_source",
        Box::new(SourceAdapter::new(CountingSource::new(5, 64))),
    );

    // Two separate sinks
    let sink1_count = Arc::new(AtomicU64::new(0));
    let sink1 = pipeline.add_node(
        "sink1",
        Box::new(SinkAdapter::new(PrintingSink::new(
            "branch-1",
            sink1_count.clone(),
            true,
        ))),
    );

    let sink2_count = Arc::new(AtomicU64::new(0));
    let sink2 = pipeline.add_node(
        "sink2",
        Box::new(SinkAdapter::new(PrintingSink::new(
            "branch-2",
            sink2_count.clone(),
            true,
        ))),
    );

    // Link source to both sinks (fan-out)
    pipeline.link(source, sink1)?;
    pipeline.link(source, sink2)?;

    // Run the pipeline
    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await?;

    println!(
        "   Sink 1 received: {}, Sink 2 received: {}",
        sink1_count.load(Ordering::Relaxed),
        sink2_count.load(Ordering::Relaxed)
    );
    Ok(())
}
