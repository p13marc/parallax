//! # Transform Pipeline
//!
//! A pipeline with a transform element that modifies data in-flight.
//! Numbers are doubled as they pass through.
//!
//! ```text
//! [CounterSource] → [DoubleTransform] → [PrintSink]
//! ```
//!
//! Run: `cargo run --example 02_transform`

use parallax::buffer::Buffer;
use parallax::element::{
    ConsumeContext, Output, ProduceContext, ProduceResult, Sink, Source, Transform,
};
use parallax::error::Result;
use parallax::memory::CpuArena;
use parallax::pipeline::Pipeline;

struct CounterSource {
    count: u32,
    max: u32,
}

impl Source for CounterSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }
        self.count += 1;
        let bytes = self.count.to_le_bytes();
        ctx.output()[..4].copy_from_slice(&bytes);
        Ok(ProduceResult::Produced(4))
    }
}

struct DoubleTransform;

impl Transform for DoubleTransform {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let value = u32::from_le_bytes(buffer.as_bytes()[..4].try_into().unwrap());
        let doubled = value * 2;

        let segment = std::sync::Arc::new(parallax::memory::HeapSegment::new(4)?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                doubled.to_le_bytes().as_ptr(),
                parallax::memory::MemorySegment::as_mut_ptr(&*segment).unwrap(),
                4,
            );
        }
        Ok(Output::Single(Buffer::new(
            parallax::buffer::MemoryHandle::from_segment(segment),
            buffer.metadata().clone(),
        )))
    }
}

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let value = u32::from_le_bytes(ctx.input()[..4].try_into().unwrap());
        println!("Value: {}", value);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arena = CpuArena::new(64, 8)?;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source_with_arena("src", CounterSource { count: 0, max: 5 }, arena);
    let transform = pipeline.add_transform("double", DoubleTransform);
    let sink = pipeline.add_sink("sink", PrintSink);

    pipeline.link(src, transform)?;
    pipeline.link(transform, sink)?;

    pipeline.run().await
}
