//! # Tee (Fan-out)
//!
//! A pipeline that splits one stream into two using a Tee element.
//! Each buffer is cloned and sent to both sinks.
//!
//! ```text
//!                  ┌→ [PrintSink "A"]
//! [CounterSource] →│
//!                  └→ [PrintSink "B"]
//! ```
//!
//! Run: `cargo run --example 03_tee`

use parallax::element::{
    ConsumeContext, DynAsyncElement, ElementAdapter, ProduceContext, ProduceResult, Sink,
    SinkAdapter, Source, SourceAdapter,
};
use parallax::elements::Tee;
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

struct NamedSink {
    name: &'static str,
}

impl Sink for NamedSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let value = u32::from_le_bytes(ctx.input()[..4].try_into().unwrap());
        println!("[{}] Received: {}", self.name, value);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arena = CpuArena::new(64, 8)?;

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::with_arena(
            CounterSource { count: 0, max: 3 },
            arena,
        )),
    );
    let tee = pipeline.add_node(
        "tee",
        DynAsyncElement::new_box(ElementAdapter::new(Tee::new())),
    );
    let sink_a = pipeline.add_node(
        "sink_a",
        DynAsyncElement::new_box(SinkAdapter::new(NamedSink { name: "A" })),
    );
    let sink_b = pipeline.add_node(
        "sink_b",
        DynAsyncElement::new_box(SinkAdapter::new(NamedSink { name: "B" })),
    );

    pipeline.link(src, tee)?;
    pipeline.link(tee, sink_a)?;
    pipeline.link(tee, sink_b)?;

    pipeline.run().await
}
