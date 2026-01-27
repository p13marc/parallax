//! Example 20: Dynamic Pipeline State Changes
//!
//! This example demonstrates the PipeWire-inspired 3-state pipeline model
//! and how to dynamically transition between states.
//!
//! Pipeline States:
//! ```text
//! Suspended <──> Idle <──> Running
//!     │                        │
//!     └────── Error ◄──────────┘
//! ```
//!
//! - **Suspended**: Minimal memory footprint, resources deallocated
//! - **Idle**: Ready to process (paused), resources allocated
//! - **Running**: Actively processing data
//! - **Error**: Unrecoverable error state
//!
//! State Transitions:
//! - `prepare()`: Suspended → Idle (validate, negotiate caps, allocate)
//! - `activate()`: Idle → Running (start processing)
//! - `pause()`: Running → Idle (stop processing, keep resources)
//! - `suspend()`: Idle → Suspended (release resources)
//!
//! Key insight from PipeWire: "paused" and "stopped" are the same state (Idle)
//! - the difference is just intent.
//!
//! Run with: cargo run --example 20_dynamic_state

use parallax::element::{
    ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
    SourceAdapter,
};
use parallax::error::Result;
use parallax::pipeline::{Executor, Pipeline, PipelineState};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A source that produces numbered buffers.
struct NumberSource {
    current: u64,
    max: u64,
}

impl Source for NumberSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.current >= self.max {
            return Ok(ProduceResult::Eos);
        }
        let data = self.current.to_le_bytes();
        let output = ctx.output();
        let len = data.len().min(output.len());
        output[..len].copy_from_slice(&data[..len]);
        ctx.set_sequence(self.current);
        self.current += 1;
        Ok(ProduceResult::Produced(len))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(8)
    }
}

/// A sink that counts received buffers.
struct CountingSink {
    count: Arc<AtomicU64>,
}

impl Sink for CountingSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn print_state(pipeline: &Pipeline) {
    let state = pipeline.state();
    let state_str = match state {
        PipelineState::Suspended => "Suspended (resources deallocated)",
        PipelineState::Idle => "Idle (ready to process, resources allocated)",
        PipelineState::Running => "Running (actively processing)",
        PipelineState::Error => "Error (unrecoverable)",
    };
    println!("   Current state: {:?} - {}", state, state_str);
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Dynamic Pipeline State Changes Example ===\n");

    // Create a pipeline
    let count = Arc::new(AtomicU64::new(0));
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(NumberSource {
            current: 0,
            max: 100,
        })),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            count: count.clone(),
        })),
    );
    pipeline.link(src, sink)?;

    println!("1. Initial State:");
    print_state(&pipeline);
    assert_eq!(pipeline.state(), PipelineState::Suspended);
    println!();

    // Transition: Suspended -> Idle
    println!("2. Calling prepare() (Suspended -> Idle):");
    println!("   - Validates pipeline structure");
    println!("   - Negotiates caps between elements");
    println!("   - Allocates resources");
    pipeline.prepare()?;
    print_state(&pipeline);
    assert_eq!(pipeline.state(), PipelineState::Idle);
    println!();

    // At this point, the pipeline is ready but not running
    println!("3. Pipeline is now Idle (paused/ready):");
    println!("   - Resources are allocated");
    println!("   - Caps are negotiated");
    println!("   - Ready to start processing at any time");
    println!("   - Negotiated: {}", pipeline.is_negotiated());
    println!();

    // Transition: Idle -> Running
    println!("4. Calling activate() (Idle -> Running):");
    println!("   - Starts data processing");
    pipeline.activate()?;
    print_state(&pipeline);
    assert_eq!(pipeline.state(), PipelineState::Running);
    println!();

    // Let it run a bit using the executor
    println!("5. Running the pipeline with Executor:");
    let executor = Executor::new();
    let handle = executor.start(&mut pipeline)?;

    // Wait for completion
    handle.wait().await?;
    println!("   Processed {} buffers", count.load(Ordering::Relaxed));
    print_state(&pipeline);
    println!();

    // Transition: Running -> Idle (pause)
    println!("6. Calling pause() (Running -> Idle):");
    println!("   - Stops data processing");
    println!("   - Keeps resources allocated");
    println!("   - Can resume quickly");
    pipeline.pause()?;
    print_state(&pipeline);
    assert_eq!(pipeline.state(), PipelineState::Idle);
    println!();

    // Demonstrate that pausing preserves resources
    println!("7. While Idle (paused):");
    println!("   - Resources still allocated");
    println!("   - Can call activate() to resume");
    println!("   - Or suspend() to release resources");
    println!();

    // Transition: Idle -> Suspended (release resources)
    println!("8. Calling suspend() (Idle -> Suspended):");
    println!("   - Releases resources");
    println!("   - Minimal memory footprint");
    pipeline.suspend()?;
    print_state(&pipeline);
    assert_eq!(pipeline.state(), PipelineState::Suspended);
    println!();

    // Show that we can prepare again
    println!("9. Can prepare again to restart:");
    pipeline.prepare()?;
    print_state(&pipeline);
    pipeline.suspend()?;
    println!();

    // Summary
    println!("=== State Transition Summary ===\n");
    println!("  Suspended  ──prepare()──>  Idle  ──activate()──>  Running");
    println!("      ^                        │                       │");
    println!("      │                        │                       │");
    println!("      └──────suspend()─────────┘<──────pause()─────────┘");
    println!();
    println!("Key points:");
    println!("  - prepare() allocates resources and negotiates caps");
    println!("  - activate() starts processing (can be called multiple times)");
    println!("  - pause() stops processing but keeps resources");
    println!("  - suspend() releases resources for minimal footprint");
    println!();
    println!("The Idle state represents both 'paused' and 'stopped' - the");
    println!("difference is only in intent. This simplifies state management.");
    println!();

    // Demonstrate state queries
    println!("=== State Query Methods ===\n");
    let mut demo = Pipeline::new();
    demo.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NumberSource { current: 0, max: 1 })),
    );

    println!("  pipeline.state() returns: {:?}", demo.state());
    println!(
        "  pipeline.is_negotiated() returns: {}",
        demo.is_negotiated()
    );
    println!();

    demo.prepare()?;
    println!("  After prepare():");
    println!("    state: {:?}", demo.state());
    println!("    is_negotiated: {}", demo.is_negotiated());

    println!("\n=== Example Complete ===");

    Ok(())
}
