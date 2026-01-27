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

use parallax::element::{
    ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
    SourceAdapter,
};
use parallax::error::Result;
use parallax::memory::CpuArena;
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
    let arena = CpuArena::new(1024, 4)?;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::with_arena(
            HelloSource { sent: false },
            arena,
        )),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(PrintSink)),
    );
    pipeline.link(src, sink)?;

    pipeline.run().await
}
