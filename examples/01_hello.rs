//! # Hello Pipeline
//!
//! The simplest possible pipeline: a source that produces one message,
//! connected to a sink that prints it.
//!
//! ```text
//! [HelloSource] → [PrintSink]
//! ```
//!
//! Run: `cargo run --example 01_hello`

use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::pipeline::Pipeline;

struct HelloSource {
    sent: bool,
}

impl Source for HelloSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.sent {
            return Ok(ProduceResult::Eos);
        }
        self.sent = true;

        let msg = b"Hello, Parallax!";
        ctx.output()[..msg.len()].copy_from_slice(msg);
        Ok(ProduceResult::Produced(msg.len()))
    }
}

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let text = std::str::from_utf8(ctx.input()).unwrap_or("<invalid>");
        println!("Received: {}", text);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arena = SharedArena::new(1024, 4)?;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source_with_arena("src", HelloSource { sent: false }, arena);
    let sink = pipeline.add_sink("sink", PrintSink);
    pipeline.link(src, sink)?;

    pipeline.run().await
}
