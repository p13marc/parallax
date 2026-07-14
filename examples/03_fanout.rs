//! # Fan-out (one source, many sinks)
//!
//! Fan-out needs **no element**. Src-pads are 1:N: link the same node twice and
//! the executor hands each branch a clone of the buffer — which is a refcount
//! bump on shared memory, not a copy.
//!
//! ```text
//!                  ┌→ [PrintSink "A"]   link()        must not lose data
//! [CounterSource] →│
//!                  └→ [SlowSink  "B"]   link_lossy()  may drop when behind
//! ```
//!
//! The second link is the important one. A `Block` branch that fills its channel
//! back-pressures the source *and every sibling* — so one slow consumer would
//! otherwise drag down the fast one. `link_lossy` lets branch B degrade alone.
//!
//! (This example used to put a `Tee` element in the middle and imply the element
//! did the splitting. It never did — it is a passthrough counter, now honestly
//! named `Inspect`. The fan-out was always in the links.)
//!
//! The parse grammar is a strictly linear chain and cannot express this, so
//! fan-out requires the programmatic API used below.
//!
//! Run: `cargo run --example 03_fanout`

use std::time::Duration;

use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::memory::SharedArena;
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
    delay: Option<Duration>,
}

impl Sink for NamedSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        if let Some(delay) = self.delay {
            std::thread::sleep(delay);
        }
        let value = u32::from_le_bytes(ctx.input()[..4].try_into().unwrap());
        println!("[{}] Received: {}", self.name, value);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arena = SharedArena::new(64, 8)?;

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source_with_arena("src", CounterSource { count: 0, max: 5 }, arena);
    let fast = pipeline.add_sink(
        "sink_a",
        NamedSink {
            name: "A lossless",
            delay: None,
        },
    );
    let slow = pipeline.add_sink(
        "sink_b",
        NamedSink {
            name: "B lossy   ",
            delay: Some(Duration::from_millis(50)),
        },
    );

    // Two links out of one src-pad. That is the whole fan-out.
    pipeline.link(src, fast)?;
    pipeline.link_lossy(src, slow)?;

    pipeline.run().await
}
