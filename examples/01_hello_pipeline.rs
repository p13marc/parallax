//! Hello World pipeline - the simplest possible example.
//!
//! Run with: cargo run --example 01_hello_pipeline

use parallax::element::{
    ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
    SourceAdapter,
};
use parallax::error::Result;
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

        let data = b"Hello, Pipeline!";
        let output = ctx.output();
        let len = data.len().min(output.len());
        output[..len].copy_from_slice(&data[..len]);

        Ok(ProduceResult::Produced(len))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(16) // "Hello, Pipeline!" is 16 bytes
    }
}

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let text = std::str::from_utf8(ctx.input()).unwrap_or("<invalid utf8>");
        println!("Received: {}", text);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(HelloSource { sent: false })),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(PrintSink)),
    );

    pipeline.link(src, sink)?;
    pipeline.run().await
}
