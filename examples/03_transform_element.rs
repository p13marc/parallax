//! Adding a transform element between source and sink.
//!
//! Run with: cargo run --example 03_transform_element

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    ConsumeContext, DynAsyncElement, Element, ElementAdapter, ProduceContext, ProduceResult, Sink,
    SinkAdapter, Source, SourceAdapter,
};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::pipeline::Pipeline;
use std::sync::Arc;

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

/// Doubles every number that passes through.
struct DoubleTransform;

impl Element for DoubleTransform {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let value = u64::from_le_bytes(buffer.as_bytes().try_into().unwrap());
        let doubled = value * 2;

        // Create new buffer with doubled value
        let segment = Arc::new(HeapSegment::new(8)?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                doubled.to_le_bytes().as_ptr(),
                segment.as_mut_ptr().unwrap(),
                8,
            );
        }

        Ok(Some(Buffer::new(
            MemoryHandle::from_segment(segment),
            buffer.metadata().clone(),
        )))
    }
}

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let value = u64::from_le_bytes(ctx.input().try_into().unwrap());
        println!("Received: {}", value);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Pipeline: Source(0..5) -> Double -> Print");
    println!();

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "numbers",
        DynAsyncElement::new_box(SourceAdapter::new(NumberSource { current: 0, max: 5 })),
    );
    let double = pipeline.add_node(
        "double",
        DynAsyncElement::new_box(ElementAdapter::new(DoubleTransform)),
    );
    let sink = pipeline.add_node(
        "print",
        DynAsyncElement::new_box(SinkAdapter::new(PrintSink)),
    );

    pipeline.link(src, double)?;
    pipeline.link(double, sink)?;

    pipeline.run().await
}
