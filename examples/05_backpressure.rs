//! # Backpressure (the link is the queue)
//!
//! Every element runs in its own task behind a bounded channel, so queueing
//! is a property of the **link**, not an element: `link_pads_full` sets the
//! channel capacity, and a full channel back-pressures the producer.
//!
//! ```text
//! [FastSource] ──channel(2)──> [SlowSink]
//! ```
//!
//! Run: `cargo run --example 05_backpressure`

use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::pipeline::{LinkPolicy, Pipeline};
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
    let arena = SharedArena::new(64, 8)?;

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source_with_arena("src", FastSource { count: 0, max: 5 }, arena);
    let sink = pipeline.add_sink("sink", SlowSink);

    // The channel between the two nodes holds at most 2 buffers; once it is
    // full the source's send waits — that IS the backpressure. `Block` is the
    // default policy (no data loss); `LinkPolicy::Drop` would shed instead.
    pipeline.link_pads_full(src, "src", sink, "sink", LinkPolicy::Block, Some(2))?;

    pipeline.run().await
}
