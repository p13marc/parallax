//! # Queue (Backpressure)
//!
//! A pipeline with a Queue element that buffers data between source and sink.
//! The queue provides backpressure when full.
//!
//! ```text
//! [FastSource] → [Queue] → [SlowSink]
//! ```
//!
//! Run: `cargo run --example 05_queue`

use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::elements::Queue;
use parallax::error::Result;
use parallax::memory::CpuArena;
use parallax::pipeline::Pipeline;
use std::time::Duration;

struct FastSource {
    count: u32,
    max: u32,
}

impl Source for FastSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }
        self.count += 1;
        println!("[Source] Producing {}", self.count);
        let bytes = self.count.to_le_bytes();
        ctx.output()[..4].copy_from_slice(&bytes);
        Ok(ProduceResult::Produced(4))
    }
}

struct SlowSink;

impl Sink for SlowSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let value = u32::from_le_bytes(ctx.input()[..4].try_into().unwrap());
        println!("[Sink] Processing {}...", value);
        std::thread::sleep(Duration::from_millis(100)); // Simulate slow processing
        println!("[Sink] Done with {}", value);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arena = CpuArena::new(64, 8)?;

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source_with_arena("src", FastSource { count: 0, max: 5 }, arena);
    // Queue with capacity of 2 buffers
    let queue = pipeline.add_filter("queue", Queue::new(2));
    let sink = pipeline.add_sink("sink", SlowSink);

    pipeline.link(src, queue)?;
    pipeline.link(queue, sink)?;

    pipeline.run().await
}
