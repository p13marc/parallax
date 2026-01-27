//! Pipeline Builder DSL - Fluent API for constructing pipelines.
//!
//! This example demonstrates the ergonomic builder API for creating pipelines
//! without manual adapter wrapping and linking.
//!
//! Run with: cargo run --example 36_builder_dsl

use parallax::element::{
    ConsumeContext, Output, ProduceContext, ProduceResult, Sink, Source, Transform,
};
use parallax::error::Result;
use parallax::memory::MemorySegment;
use parallax::pipeline::PipelineBuilder;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

// ============================================================================
// Example Elements
// ============================================================================

/// A source that produces numbers from 0 to max-1.
struct NumberSource {
    current: u32,
    max: u32,
}

impl NumberSource {
    fn new(max: u32) -> Self {
        Self { current: 0, max }
    }
}

impl Source for NumberSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.current >= self.max {
            return Ok(ProduceResult::Eos);
        }

        let data = self.current.to_le_bytes();
        let output = ctx.output();
        output[..4].copy_from_slice(&data);
        ctx.set_sequence(self.current as u64);

        self.current += 1;
        Ok(ProduceResult::Produced(4))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(4)
    }
}

/// A transform that doubles each number.
struct DoubleTransform;

impl Transform for DoubleTransform {
    fn transform(&mut self, buffer: parallax::buffer::Buffer) -> Result<Output> {
        let value = u32::from_le_bytes(buffer.as_bytes().try_into().unwrap());
        let doubled = value * 2;

        // Create new buffer with doubled value
        let segment = std::sync::Arc::new(parallax::memory::HeapSegment::new(4)?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                doubled.to_le_bytes().as_ptr(),
                segment.as_mut_ptr().unwrap(),
                4,
            );
        }

        Ok(Output::Single(parallax::buffer::Buffer::new(
            parallax::buffer::MemoryHandle::from_segment(segment),
            buffer.metadata().clone(),
        )))
    }
}

/// A transform that adds a constant to each number.
struct AddTransform {
    value: u32,
}

impl AddTransform {
    fn new(value: u32) -> Self {
        Self { value }
    }
}

impl Transform for AddTransform {
    fn transform(&mut self, buffer: parallax::buffer::Buffer) -> Result<Output> {
        let value = u32::from_le_bytes(buffer.as_bytes().try_into().unwrap());
        let added = value + self.value;

        let segment = std::sync::Arc::new(parallax::memory::HeapSegment::new(4)?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                added.to_le_bytes().as_ptr(),
                segment.as_mut_ptr().unwrap(),
                4,
            );
        }

        Ok(Output::Single(parallax::buffer::Buffer::new(
            parallax::buffer::MemoryHandle::from_segment(segment),
            buffer.metadata().clone(),
        )))
    }
}

/// A sink that prints and counts received values.
struct PrintSink {
    name: String,
    counter: Arc<AtomicU32>,
}

impl PrintSink {
    fn new(name: impl Into<String>, counter: Arc<AtomicU32>) -> Self {
        Self {
            name: name.into(),
            counter,
        }
    }
}

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let value = u32::from_le_bytes(ctx.input().try_into().unwrap());
        println!("[{}] Received: {}", self.name, value);
        self.counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// A sink that just counts (no printing).
struct CountingSink {
    counter: Arc<AtomicU32>,
}

impl CountingSink {
    fn new(counter: Arc<AtomicU32>) -> Self {
        Self { counter }
    }
}

impl Sink for CountingSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        self.counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

// ============================================================================
// Examples
// ============================================================================

/// Example 1: Simple linear pipeline with builder
async fn example_linear_pipeline() -> Result<()> {
    println!("=== Example 1: Linear Pipeline ===\n");

    let counter = Arc::new(AtomicU32::new(0));

    // Builder pattern: source -> transform -> sink
    let mut pipeline = PipelineBuilder::new()
        .with_new_arena(64, 8)?
        .source(NumberSource::new(5))
        .then(DoubleTransform)
        .sink(PrintSink::new("output", counter.clone()))
        .build()?;

    pipeline.run().await?;

    println!(
        "\nReceived {} buffers (expected 5)\n",
        counter.load(Ordering::Relaxed)
    );

    Ok(())
}

/// Example 2: Multiple transforms chained
async fn example_chained_transforms() -> Result<()> {
    println!("=== Example 2: Chained Transforms ===\n");

    let counter = Arc::new(AtomicU32::new(0));

    // Chain multiple transforms: double, then add 10
    let mut pipeline = PipelineBuilder::new()
        .with_new_arena(64, 8)?
        .source(NumberSource::new(5))
        .then(DoubleTransform)
        .then(AddTransform::new(10))
        .sink(PrintSink::new("output", counter.clone()))
        .build()?;

    pipeline.run().await?;

    println!("\nReceived {} buffers\n", counter.load(Ordering::Relaxed));

    Ok(())
}

/// Example 3: Named elements for debugging
async fn example_named_elements() -> Result<()> {
    println!("=== Example 3: Named Elements ===\n");

    let counter = Arc::new(AtomicU32::new(0));

    let pipeline = PipelineBuilder::new()
        .with_new_arena(64, 8)?
        .source_named("number_generator", NumberSource::new(3))
        .then_named("doubler", DoubleTransform)
        .then_named("adder", AddTransform::new(100))
        .sink_named("printer", PrintSink::new("final", counter.clone()))
        .build()?;

    // Access nodes by name
    println!("Pipeline nodes:");
    for (id, node) in pipeline.nodes() {
        println!(
            "  [{}] {} ({:?})",
            id.index(),
            node.name(),
            node.element_type()
        );
    }
    println!();

    // Verify named elements exist
    assert!(pipeline.get_node_id("number_generator").is_some());
    assert!(pipeline.get_node_id("doubler").is_some());
    assert!(pipeline.get_node_id("adder").is_some());
    assert!(pipeline.get_node_id("printer").is_some());

    let mut pipeline = pipeline;
    pipeline.run().await?;

    println!("\nReceived {} buffers\n", counter.load(Ordering::Relaxed));

    Ok(())
}

/// Example 4: Tee with multiple branches
async fn example_tee_branching() -> Result<()> {
    println!("=== Example 4: Tee Branching ===\n");

    let counter_a = Arc::new(AtomicU32::new(0));
    let counter_b = Arc::new(AtomicU32::new(0));

    // Tee splits to two branches with different processing
    let mut pipeline = PipelineBuilder::new()
        .with_new_arena(64, 16)?
        .source(NumberSource::new(5))
        .tee(|t| {
            // Branch A: just count
            t.branch(|b| b.sink(CountingSink::new(counter_a.clone())));

            // Branch B: double, then count
            t.branch(|b| {
                b.then(DoubleTransform)
                    .sink(CountingSink::new(counter_b.clone()))
            });
        })
        .build()?;

    pipeline.run().await?;

    println!(
        "Branch A received: {} (raw values)",
        counter_a.load(Ordering::Relaxed)
    );
    println!(
        "Branch B received: {} (doubled values)",
        counter_b.load(Ordering::Relaxed)
    );
    println!();

    Ok(())
}

/// Example 5: Compare with traditional API
async fn example_compare_apis() -> Result<()> {
    println!("=== Example 5: API Comparison ===\n");

    let counter = Arc::new(AtomicU32::new(0));

    // Traditional API (verbose)
    println!("Traditional API:");
    println!("  let mut pipeline = Pipeline::new();");
    println!(
        "  let src = pipeline.add_node(\"src\", DynAsyncElement::new_box(SourceAdapter::new(...)));"
    );
    println!(
        "  let transform = pipeline.add_node(\"transform\", DynAsyncElement::new_box(TransformAdapter::new(...)));"
    );
    println!(
        "  let sink = pipeline.add_node(\"sink\", DynAsyncElement::new_box(SinkAdapter::new(...)));"
    );
    println!("  pipeline.link(src, transform)?;");
    println!("  pipeline.link(transform, sink)?;");
    println!();

    // Builder API (concise)
    println!("Builder API:");
    println!("  let pipeline = PipelineBuilder::new()");
    println!("      .source(NumberSource::new(5))");
    println!("      .then(DoubleTransform)");
    println!("      .sink(PrintSink::new(...))");
    println!("      .build()?;");
    println!();

    // Actually run the builder version
    let mut pipeline = PipelineBuilder::new()
        .with_new_arena(64, 8)?
        .source(NumberSource::new(3))
        .then(DoubleTransform)
        .sink(PrintSink::new("output", counter.clone()))
        .build()?;

    println!("Running the builder version:\n");
    pipeline.run().await?;

    println!("\nReceived {} buffers\n", counter.load(Ordering::Relaxed));

    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    example_linear_pipeline().await?;
    example_chained_transforms().await?;
    example_named_elements().await?;
    example_tee_branching().await?;
    example_compare_apis().await?;

    println!("All examples completed successfully!");
    Ok(())
}
