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

use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
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
        let src = pipeline.add_source_with_arena("tcpsrc", TcpSrc::listen(addr)?, arena);
        let sink = pipeline.add_sink("sink", PrintSink);
        pipeline.link(src, sink)?;
        pipeline.run().await
    });

    // Give server time to start listening
    tokio::time::sleep(Duration::from_millis(100)).await;

    // Sender pipeline (client - connects to server)
    let arena = CpuArena::new(256, 8)?;
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source_with_arena("src", MessageSource { count: 0, max: 3 }, arena);
    let sink = pipeline.add_sink("tcpsink", TcpSink::connect(addr)?);
    pipeline.link(src, sink)?;

    println!("[Sender] Starting...");
    pipeline.run().await?;
    println!("[Sender] Done\n");

    // Receiver should detect connection close and terminate gracefully
    println!("[Main] Waiting for receiver to complete...");
    receiver.await.expect("receiver task panicked")?;
    println!("[Main] Receiver completed");

    Ok(())
}
