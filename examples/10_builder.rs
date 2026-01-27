//! # Pipeline Builder DSL
//!
//! Fluent builder API for constructing pipelines with less boilerplate.
//! Uses the `>>` operator for linking elements.
//!
//! Run: `cargo run --example 10_builder`

use parallax::buffer::Buffer;
use parallax::element::{
    ConsumeContext, Output, ProduceContext, ProduceResult, Sink, Source, Transform,
};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::pipeline::{FromSource, PipelineBuilder, to};
use std::sync::Arc;

struct NumberSource {
    current: u32,
    max: u32,
}

impl Source for NumberSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.current >= self.max {
            return Ok(ProduceResult::Eos);
        }
        self.current += 1;
        ctx.output()[..4].copy_from_slice(&self.current.to_le_bytes());
        Ok(ProduceResult::Produced(4))
    }
}

struct SquareTransform;

impl Transform for SquareTransform {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let value = u32::from_le_bytes(buffer.as_bytes()[..4].try_into().unwrap());
        let squared = value * value;

        let segment = Arc::new(HeapSegment::new(4)?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                squared.to_le_bytes().as_ptr(),
                segment.as_mut_ptr().unwrap(),
                4,
            );
        }
        Ok(Output::Single(Buffer::new(
            parallax::buffer::MemoryHandle::from_segment(segment),
            buffer.metadata().clone(),
        )))
    }
}

struct PrintSink {
    label: &'static str,
}

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let value = u32::from_le_bytes(ctx.input()[..4].try_into().unwrap());
        println!("[{}] {}", self.label, value);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Pipeline Builder DSL ===\n");

    // Example 1: Simple chain with >> operator
    // Note: Use FromSource wrapper and to() for sink
    println!("--- Linear Pipeline (>> operator) ---");
    let pipeline = FromSource(NumberSource { current: 0, max: 5 })
        >> SquareTransform
        >> to(PrintSink { label: "Square" });

    pipeline.run().await?;

    // Example 2: Fluent builder API
    println!("\n--- Fluent Builder API ---");
    PipelineBuilder::new()
        .source(NumberSource { current: 0, max: 3 })
        .then(SquareTransform)
        .sink(PrintSink { label: "Result" })
        .build()?
        .run()
        .await?;

    // Example 3: Named elements
    println!("\n--- Named Elements ---");
    PipelineBuilder::new()
        .source_named("numbers", NumberSource { current: 0, max: 3 })
        .then_named("square", SquareTransform)
        .sink_named("print", PrintSink { label: "Named" })
        .build()?
        .run()
        .await?;

    Ok(())
}
