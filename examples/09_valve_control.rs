//! Valve for on/off flow control.
//!
//! Run with: cargo run --example 09_valve_control

use parallax::element::{
    ConsumeContext, DynAsyncElement, ElementAdapter, ProduceContext, ProduceResult, Sink,
    SinkAdapter, Source, SourceAdapter,
};
use parallax::elements::Valve;
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
    received: Arc<AtomicU64>,
}

impl Sink for CountingSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Valve starts closed, drops all buffers");
    println!();

    let received = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(NumberSource {
            current: 0,
            max: 10,
        })),
    );

    // Valve starts closed (open=false)
    let valve = pipeline.add_node(
        "valve",
        DynAsyncElement::new_box(ElementAdapter::new(Valve::with_state(false))),
    );

    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            received: received.clone(),
        })),
    );

    pipeline.link(src, valve)?;
    pipeline.link(valve, sink)?;

    pipeline.run().await?;

    println!("Sent: 10 buffers");
    println!(
        "Received: {} buffers (valve was closed)",
        received.load(Ordering::Relaxed)
    );

    Ok(())
}
