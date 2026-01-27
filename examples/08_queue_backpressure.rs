//! Queue element with backpressure control.
//!
//! Run with: cargo run --example 08_queue_backpressure

use parallax::element::{
    ConsumeContext, DynAsyncElement, ElementAdapter, ProduceContext, ProduceResult, Sink,
    SinkAdapter, Source, SourceAdapter,
};
use parallax::elements::{LeakyMode, Queue};
use parallax::error::Result;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

struct FastSource {
    current: u64,
    max: u64,
}

impl Source for FastSource {
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

struct SlowSink {
    received: Arc<AtomicU64>,
}

impl Sink for SlowSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        // Simulate slow processing
        std::thread::sleep(Duration::from_millis(10));
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Queue with capacity 4, leaky mode drops old buffers when full");
    println!();

    let received = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "fast_source",
        DynAsyncElement::new_box(SourceAdapter::new(FastSource {
            current: 0,
            max: 20,
        })),
    );

    // Queue with capacity 4, drops oldest when full
    let queue = pipeline.add_node(
        "queue",
        DynAsyncElement::new_box(ElementAdapter::new(
            Queue::new(4).leaky(LeakyMode::Upstream),
        )),
    );

    let sink = pipeline.add_node(
        "slow_sink",
        DynAsyncElement::new_box(SinkAdapter::new(SlowSink {
            received: received.clone(),
        })),
    );

    pipeline.link(src, queue)?;
    pipeline.link(queue, sink)?;

    pipeline.run().await?;

    println!("Sent: 20 buffers");
    println!(
        "Received: {} buffers (some dropped due to backpressure)",
        received.load(Ordering::Relaxed)
    );

    Ok(())
}
