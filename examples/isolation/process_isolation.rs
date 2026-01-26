//! Process isolation example demonstrating automatic IPC injection.
//!
//! This example shows how to run pipelines with transparent process isolation.
//! The executor automatically handles IPC boundaries - users write normal
//! pipelines and specify which elements should be isolated.
//!
//! # Key Concepts
//!
//! - **ExecutionMode::InProcess**: Everything runs in one process (default)
//! - **ExecutionMode::isolated()**: Each element runs in its own process
//! - **ExecutionMode::grouped(patterns)**: Elements matching patterns are isolated
//!
//! # Use Cases
//!
//! 1. **Fault Isolation**: Isolate untrusted decoders that might crash
//! 2. **Security**: Run potentially unsafe operations in sandboxed processes
//! 3. **Resource Control**: Limit memory/CPU per element via cgroups
//!
//! Run with: cargo run --example process_isolation

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    DynAsyncElement, Element, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter,
};
use parallax::error::Result;
use parallax::execution::{ExecutionMode, ExecutionPlan, IsolatedExecutor, IsolatedExecutorConfig};
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// A source that produces numbered buffers.
struct CountingSource {
    current: u64,
    max: u64,
}

impl CountingSource {
    fn new(count: u64) -> Self {
        Self {
            current: 0,
            max: count,
        }
    }
}

impl Source for CountingSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.current >= self.max {
            return Ok(None);
        }

        // Create buffer with sequence number in data
        let data = self.current.to_le_bytes();
        let segment = Arc::new(HeapSegment::new(8)?);
        // Write data using raw pointer (HeapSegment provides as_mut_ptr)
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), segment.as_mut_ptr().unwrap(), 8);
        }

        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::from_sequence(self.current));

        self.current += 1;
        Ok(Some(buffer))
    }
}

/// A "decoder" that simulates expensive processing.
/// In real use, this might be an H.264 decoder, image processor, etc.
struct SimulatedDecoder {
    name: String,
    processed: u64,
}

impl SimulatedDecoder {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            processed: 0,
        }
    }
}

impl Element for SimulatedDecoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.processed += 1;

        // Simulate some CPU work
        let mut sum = 0u64;
        for _ in 0..1000 {
            sum = sum.wrapping_add(self.processed);
        }
        let _ = sum; // Prevent optimization

        // In a real decoder, we might:
        // - Parse compressed data
        // - Decode to raw frames
        // - Potentially crash on malformed input (hence isolation!)

        Ok(Some(buffer))
    }
}

/// A sink that counts received buffers.
struct CountingSink {
    received: Arc<AtomicU64>,
    name: String,
}

impl CountingSink {
    fn new(name: &str, counter: Arc<AtomicU64>) -> Self {
        Self {
            received: counter,
            name: name.to_string(),
        }
    }
}

impl Sink for CountingSink {
    fn consume(&mut self, _buffer: Buffer) -> Result<()> {
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn main() -> Result<()> {
    println!("=== Process Isolation Example ===\n");

    // Example 1: Plan analysis (in-process)
    println!("1. In-Process Execution Plan");
    example_in_process_plan()?;

    // Example 2: Plan analysis (fully isolated)
    println!("\n2. Fully Isolated Execution Plan");
    example_isolated_plan()?;

    // Example 3: Plan analysis (selective isolation)
    println!("\n3. Selective Isolation (isolate decoders only)");
    example_grouped_plan()?;

    // Example 4: Running with in-process mode
    println!("\n4. Running Pipeline In-Process");
    tokio::runtime::Runtime::new()?.block_on(example_run_in_process())?;

    // Example 5: Using Pipeline convenience methods
    println!("\n5. Pipeline Convenience Methods");
    tokio::runtime::Runtime::new()?.block_on(example_convenience_methods())?;

    println!("\n=== Process Isolation Example Complete ===");
    Ok(())
}

/// Build a sample pipeline: src -> decoder -> encoder -> sink
fn build_sample_pipeline(sink_counter: Arc<AtomicU64>) -> Pipeline {
    let mut pipeline = Pipeline::new();

    // Add source
    let src = pipeline.add_node(
        "video_source",
        DynAsyncElement::new_box(SourceAdapter::new(CountingSource::new(100))),
    );

    // Add decoder (might crash on bad input - good candidate for isolation)
    let decoder = pipeline.add_node(
        "h264_decoder",
        DynAsyncElement::new_box(ElementAdapter::new(SimulatedDecoder::new("h264_decoder"))),
    );

    // Add encoder (CPU intensive - might want to isolate)
    let encoder = pipeline.add_node(
        "h264_encoder",
        DynAsyncElement::new_box(ElementAdapter::new(SimulatedDecoder::new("h264_encoder"))),
    );

    // Add sink
    let sink = pipeline.add_node(
        "file_sink",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink::new(
            "file_sink",
            sink_counter,
        ))),
    );

    // Connect: src -> decoder -> encoder -> sink
    pipeline.link(src, decoder).unwrap();
    pipeline.link(decoder, encoder).unwrap();
    pipeline.link(encoder, sink).unwrap();

    pipeline
}

fn example_in_process_plan() -> Result<()> {
    let counter = Arc::new(AtomicU64::new(0));
    let pipeline = build_sample_pipeline(counter);

    // Create executor with in-process mode
    let executor = IsolatedExecutor::new(ExecutionMode::InProcess);
    let plan = executor.plan(&pipeline)?;

    print_plan(&plan);

    println!("   All elements run in the same process.");
    println!("   No IPC overhead, but no fault isolation.");

    Ok(())
}

fn example_isolated_plan() -> Result<()> {
    let counter = Arc::new(AtomicU64::new(0));
    let pipeline = build_sample_pipeline(counter);

    // Create executor with full isolation
    let executor = IsolatedExecutor::new(ExecutionMode::isolated());
    let plan = executor.plan(&pipeline)?;

    print_plan(&plan);

    println!("   Each element runs in its own process.");
    println!("   Maximum isolation, but higher IPC overhead.");

    Ok(())
}

fn example_grouped_plan() -> Result<()> {
    let counter = Arc::new(AtomicU64::new(0));
    let pipeline = build_sample_pipeline(counter);

    // Isolate only decoders and encoders (pattern matching)
    let executor = IsolatedExecutor::new(ExecutionMode::grouped(vec![
        "*decoder*".to_string(),
        "*encoder*".to_string(),
    ]));
    let plan = executor.plan(&pipeline)?;

    print_plan(&plan);

    println!("   Source and sink run in supervisor process.");
    println!("   Decoder and encoder each run in isolated processes.");
    println!("   Balanced: isolates risky elements, minimizes IPC.");

    Ok(())
}

async fn example_run_in_process() -> Result<()> {
    let counter = Arc::new(AtomicU64::new(0));
    let pipeline = build_sample_pipeline(counter.clone());

    // Run with explicit executor configuration
    let config = IsolatedExecutorConfig {
        arena_size: 16 * 1024 * 1024, // 16 MB
        arena_slots: 32,
        slot_size: 512 * 1024, // 512 KB per slot
        ..Default::default()
    };

    let executor = IsolatedExecutor::with_config(ExecutionMode::InProcess, config);
    executor.run(pipeline).await?;

    println!("   Processed {} buffers", counter.load(Ordering::Relaxed));

    Ok(())
}

async fn example_convenience_methods() -> Result<()> {
    // Method 1: Simple run (in-process)
    {
        let counter = Arc::new(AtomicU64::new(0));
        let mut pipeline = build_sample_pipeline(counter.clone());
        pipeline.run().await?;
        println!(
            "   pipeline.run(): {} buffers (in-process)",
            counter.load(Ordering::Relaxed)
        );
    }

    // Method 2: Run with execution mode
    {
        let counter = Arc::new(AtomicU64::new(0));
        let pipeline = build_sample_pipeline(counter.clone());
        pipeline.run_with_mode(ExecutionMode::InProcess).await?;
        println!(
            "   pipeline.run_with_mode(): {} buffers",
            counter.load(Ordering::Relaxed)
        );
    }

    // Method 3: Run with full isolation (each element in own process)
    {
        let counter = Arc::new(AtomicU64::new(0));
        let pipeline = build_sample_pipeline(counter.clone());
        // Note: This currently falls back to in-process but demonstrates the API
        pipeline.run_isolated().await?;
        println!(
            "   pipeline.run_isolated(): {} buffers",
            counter.load(Ordering::Relaxed)
        );
    }

    // Method 4: Selective isolation with patterns
    {
        let counter = Arc::new(AtomicU64::new(0));
        let pipeline = build_sample_pipeline(counter.clone());
        // Isolate elements matching these patterns
        pipeline
            .run_isolating(vec!["*decoder*", "*encoder*"])
            .await?;
        println!(
            "   pipeline.run_isolating([\"*decoder*\", \"*encoder*\"]): {} buffers",
            counter.load(Ordering::Relaxed)
        );
    }

    Ok(())
}

fn print_plan(plan: &ExecutionPlan) {
    println!("   Execution Plan:");
    println!("   - Groups: {}", plan.groups.len());
    for (id, group) in &plan.groups {
        let process_type = if group.is_supervisor {
            "supervisor"
        } else {
            "isolated"
        };
        println!(
            "     - Group {:?} ({}): {} nodes",
            id,
            process_type,
            group.nodes.len()
        );
    }
    println!("   - IPC Boundaries: {}", plan.boundaries.len());
    for boundary in &plan.boundaries {
        println!(
            "     - {:?} -> {:?}",
            boundary.src_group, boundary.sink_group
        );
    }
}
