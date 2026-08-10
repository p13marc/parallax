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
use parallax::memory::{OutputArena, OutputBudget, defaults};
use parallax::pipeline::{FromSource, PipelineBuilder, to};

struct NumberSource {
    current: u32,
    max: u32,
    /// Sized by the executor from link capacity; the floor below only applies
    /// when nothing is driving this element.
    output: OutputArena,
}

impl NumberSource {
    fn new(max: u32) -> Result<Self> {
        Ok(Self {
            current: 0,
            max,
            output: OutputArena::new(defaults::SOURCE_SLOT_COUNT),
        })
    }
}

impl Source for NumberSource {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.current >= self.max {
            return Ok(ProduceResult::Eos);
        }

        // `try_acquire`, because this is a source: nothing sheds a source's
        // `PoolExhausted`, so a full arena means "wait", not "fail".
        let Some(mut slot) = self.output.try_acquire(4, "numbers")? else {
            return Ok(ProduceResult::WouldBlock);
        };
        // The counter advances only once the slot is secured.
        self.current += 1;
        slot.data_mut()[..4].copy_from_slice(&self.current.to_le_bytes());

        let buffer = Buffer::new(MemoryHandle::with_len(slot, 4), Default::default());
        Ok(ProduceResult::OwnBuffer(buffer))
    }
}

struct SquareTransform {
    output: OutputArena,
}

impl SquareTransform {
    fn new() -> Result<Self> {
        Ok(Self {
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT),
        })
    }
}

impl Transform for SquareTransform {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let value = u32::from_le_bytes(buffer.as_bytes()[..4].try_into().unwrap());
        let squared = value * value;

        let mut slot = self.output.acquire(4, "square")?;
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
