//! # Pipeline Builder DSL
//!
//! Fluent builder API for constructing pipelines with less boilerplate.
//! Uses the `>>` operator for linking elements.
//!
//! Run: `cargo run --example 10_builder`

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    ConsumeContext, Output, ProduceContext, ProduceResult, Sink, Source, Transform,
};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::pipeline::{FromSource, PipelineBuilder, to};

struct NumberSource {
    current: u32,
    max: u32,
    arena: SharedArena,
}

impl NumberSource {
    fn new(max: u32) -> Result<Self> {
        Ok(Self {
            current: 0,
            max,
            arena: SharedArena::new(64, 8)?,
        })
    }
}

impl Source for NumberSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.current >= self.max {
            return Ok(ProduceResult::Eos);
        }
        self.current += 1;

        // Allocate from our own arena
        let mut slot = self.arena.acquire().expect("source arena exhausted");
        slot.data_mut()[..4].copy_from_slice(&self.current.to_le_bytes());

        let buffer = Buffer::new(MemoryHandle::with_len(slot, 4), Default::default());
        Ok(ProduceResult::OwnBuffer(buffer))
    }
}

struct SquareTransform {
    arena: SharedArena,
}

impl SquareTransform {
    fn new() -> Result<Self> {
        Ok(Self {
            arena: SharedArena::new(64, 8)?,
        })
    }
}

impl Transform for SquareTransform {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let value = u32::from_le_bytes(buffer.as_bytes()[..4].try_into().unwrap());
        let squared = value * value;

        let mut slot = self.arena.acquire().expect("transform arena exhausted");
        slot.data_mut()[..4].copy_from_slice(&squared.to_le_bytes());

        Ok(Output::Single(Buffer::new(
            MemoryHandle::with_len(slot, 4),
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
    let pipeline = FromSource(NumberSource::new(5)?)
        >> SquareTransform::new()?
        >> to(PrintSink { label: "Square" });

    pipeline.run().await?;

    // Example 2: Fluent builder API
    println!("\n--- Fluent Builder API ---");
    PipelineBuilder::new()
        .source(NumberSource::new(3)?)
        .then(SquareTransform::new()?)
        .sink(PrintSink { label: "Result" })
        .build()?
        .run()
        .await?;

    // Example 3: Named elements
    println!("\n--- Named Elements ---");
    PipelineBuilder::new()
        .source_named("numbers", NumberSource::new(3)?)
        .then_named("square", SquareTransform::new()?)
        .sink_named("print", PrintSink { label: "Named" })
        .build()?
        .run()
        .await?;

    Ok(())
}
