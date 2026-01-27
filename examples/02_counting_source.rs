//! A source that produces multiple buffers with sequence numbers.
//!
//! Run with: cargo run --example 02_counting_source

use parallax::element::{
    ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
    SourceAdapter,
};
use parallax::error::Result;
use parallax::pipeline::Pipeline;

struct CountingSource {
    current: u64,
    max: u64,
}

impl Source for CountingSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.current >= self.max {
            return Ok(ProduceResult::Eos); // End of stream
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
        Some(8) // We need 8 bytes for a u64
    }
}

struct PrintSink {
    count: u64,
}

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let value = u64::from_le_bytes(ctx.input().try_into().unwrap());
        println!("Buffer {}: value = {}", self.count, value);
        self.count += 1;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "counter",
        DynAsyncElement::new_box(SourceAdapter::new(CountingSource { current: 0, max: 5 })),
    );
    let sink = pipeline.add_node(
        "printer",
        DynAsyncElement::new_box(SinkAdapter::new(PrintSink { count: 0 })),
    );

    pipeline.link(src, sink)?;
    pipeline.run().await
}
