//! Default in-process execution (no isolation).
//!
//! Run with: cargo run --example 11_isolate_in_process

use parallax::buffer::Buffer;
use parallax::element::{
    ConsumeContext, DynAsyncElement, Element, ElementAdapter, ProduceContext, ProduceResult, Sink,
    SinkAdapter, Source, SourceAdapter,
};
use parallax::error::Result;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

struct Source10 {
    n: u64,
}
impl Source for Source10 {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.n >= 10 {
            return Ok(ProduceResult::Eos);
        }
        let data = self.n.to_le_bytes();
        let output = ctx.output();
        let len = data.len().min(output.len());
        output[..len].copy_from_slice(&data[..len]);
        ctx.set_sequence(self.n);
        self.n += 1;
        Ok(ProduceResult::Produced(len))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(8)
    }
}

struct Passthrough;
impl Element for Passthrough {
    fn process(&mut self, buf: Buffer) -> Result<Option<Buffer>> {
        Ok(Some(buf))
    }
}

struct Counter(Arc<AtomicU64>);
impl Sink for Counter {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("In-process execution: all elements run in same process");
    println!();

    let counter = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(Source10 { n: 0 })),
    );
    let proc = pipeline.add_node(
        "processor",
        DynAsyncElement::new_box(ElementAdapter::new(Passthrough)),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(Counter(counter.clone()))),
    );

    pipeline.link(src, proc)?;
    pipeline.link(proc, sink)?;

    // Default: pipeline.run() runs everything in-process
    pipeline.run().await?;

    println!("Processed: {} buffers", counter.load(Ordering::Relaxed));
    Ok(())
}
