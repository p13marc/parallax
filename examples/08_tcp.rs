//! # TCP Streaming
//!
//! Stream data over TCP between two pipelines.
//! One pipeline sends, another receives.
//!
//! ```text
//! Pipeline 1: [Source] → [TcpSink]
//! Pipeline 2: [TcpSrc] → [Sink]
//! ```
//!
//! Run: `cargo run --example 08_tcp`

use parallax::element::{
    ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
    SourceAdapter,
};
use parallax::elements::{TcpSink, TcpSrc};
use parallax::error::Result;
use parallax::memory::CpuArena;
use parallax::pipeline::Pipeline;
use std::time::Duration;

struct MessageSource {
    count: u32,
    max: u32,
}

impl Source for MessageSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }
        self.count += 1;
        let msg = format!("TCP Message #{}\n", self.count);
        let bytes = msg.as_bytes();
        ctx.output()[..bytes.len()].copy_from_slice(bytes);
        Ok(ProduceResult::Produced(bytes.len()))
    }
}

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let text = std::str::from_utf8(ctx.input()).unwrap_or("<invalid>");
        print!("[Receiver] Got: {}", text);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let addr = "127.0.0.1:9876";

    // Start receiver pipeline (server - listens first)
    let receiver = tokio::spawn(async move {
        let arena = CpuArena::new(4096, 8)?;
        let mut pipeline = Pipeline::new();
        let src = pipeline.add_node(
            "tcpsrc",
            DynAsyncElement::new_box(SourceAdapter::with_arena(TcpSrc::listen(addr)?, arena)),
        );
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(PrintSink)),
        );
        pipeline.link(src, sink)?;
        pipeline.run().await
    });

    // Give server time to start listening
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Sender pipeline (client - connects to server)
    let arena = CpuArena::new(256, 8)?;
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::with_arena(
            MessageSource { count: 0, max: 3 },
            arena,
        )),
    );
    let sink = pipeline.add_node(
        "tcpsink",
        DynAsyncElement::new_box(SinkAdapter::new(TcpSink::connect(addr)?)),
    );
    pipeline.link(src, sink)?;

    println!("[Sender] Starting...");
    pipeline.run().await?;
    println!("[Sender] Done\n");

    // Give receiver time to process
    tokio::time::sleep(Duration::from_millis(200)).await;

    // Note: receiver will block waiting for more data
    receiver.abort();

    Ok(())
}
