//! Using Tee to send data to multiple sinks.
//!
//! Run with: cargo run --example 04_tee_fanout

use parallax::element::{
    ConsumeContext, DynAsyncElement, ElementAdapter, ProduceContext, ProduceResult, Sink,
    SinkAdapter, Source, SourceAdapter,
};
use parallax::elements::Tee;
use parallax::error::Result;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

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

struct CountingSink {
    name: String,
    counter: Arc<AtomicU64>,
}

impl Sink for CountingSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        self.counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Pipeline: Source -> Tee -> [Sink A, Sink B, Sink C]");
    println!();

    let counter_a = Arc::new(AtomicU64::new(0));
    let counter_b = Arc::new(AtomicU64::new(0));
    let counter_c = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(NumberSource {
            current: 0,
            max: 10,
        })),
    );

    // Tee duplicates buffers to all connected sinks
    let tee = pipeline.add_node(
        "tee",
        DynAsyncElement::new_box(ElementAdapter::new(Tee::new())),
    );

    let sink_a = pipeline.add_node(
        "sink_a",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            name: "A".into(),
            counter: counter_a.clone(),
        })),
    );
    let sink_b = pipeline.add_node(
        "sink_b",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            name: "B".into(),
            counter: counter_b.clone(),
        })),
    );
    let sink_c = pipeline.add_node(
        "sink_c",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            name: "C".into(),
            counter: counter_c.clone(),
        })),
    );

    pipeline.link(src, tee)?;
    pipeline.link(tee, sink_a)?;
    pipeline.link(tee, sink_b)?;
    pipeline.link(tee, sink_c)?;

    pipeline.run().await?;

    println!(
        "Sink A received: {} buffers",
        counter_a.load(Ordering::Relaxed)
    );
    println!(
        "Sink B received: {} buffers",
        counter_b.load(Ordering::Relaxed)
    );
    println!(
        "Sink C received: {} buffers",
        counter_c.load(Ordering::Relaxed)
    );

    Ok(())
}
