//! TCP network streaming example.
//!
//! Demonstrates streaming data between processes over TCP.
//! Run the server first, then the client in a separate terminal.
//!
//! Server: cargo run --example 21_tcp_streaming -- server
//! Client: cargo run --example 21_tcp_streaming -- client

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
    SourceAdapter,
};
use parallax::error::Result;
use parallax::memory::{CpuArena, HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::env;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

const ADDRESS: &str = "127.0.0.1:9876";
const MESSAGE_COUNT: u64 = 100;

/// A source that produces numbered messages.
struct MessageSource {
    count: u64,
    max: u64,
}

impl Source for MessageSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }

        let msg = format!("Message #{}: Hello from Parallax!\n", self.count);
        let data = msg.as_bytes();

        if ctx.has_buffer() {
            let output = ctx.output();
            let len = data.len().min(output.len());
            output[..len].copy_from_slice(&data[..len]);
            ctx.set_sequence(self.count);
            self.count += 1;
            Ok(ProduceResult::Produced(len))
        } else {
            // Fallback: create our own buffer
            let segment = Arc::new(HeapSegment::new(data.len())?);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    data.len(),
                );
            }
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(self.count));
            self.count += 1;
            Ok(ProduceResult::OwnBuffer(buffer))
        }
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(256)
    }
}

/// A sink that prints received messages.
struct PrintSink {
    received: Arc<AtomicU64>,
}

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let data = ctx.input();
        if let Ok(text) = std::str::from_utf8(data) {
            print!("Received: {}", text);
        } else {
            println!("Received {} bytes (binary)", data.len());
        }
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

async fn run_server() -> Result<()> {
    use parallax::elements::network::TcpSink;

    println!("Starting TCP server on {}", ADDRESS);
    println!("Waiting for client connection...");

    let arena = CpuArena::new(256, 16)?;

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "message_source",
        DynAsyncElement::new_box(SourceAdapter::with_arena(
            MessageSource {
                count: 0,
                max: MESSAGE_COUNT,
            },
            arena,
        )),
    );

    // TcpSink in server mode - waits for a client to connect
    // Wrap in SinkAdapter since TcpSink implements Sink trait
    let tcp_sink = TcpSink::listen(ADDRESS)?;
    let sink = pipeline.add_node(
        "tcp_sink",
        DynAsyncElement::new_box(SinkAdapter::new(tcp_sink)),
    );

    pipeline.link(src, sink)?;

    println!("Sending {} messages to client...", MESSAGE_COUNT);
    pipeline.run().await?;

    println!("Server finished sending all messages.");
    Ok(())
}

async fn run_client() -> Result<()> {
    use parallax::elements::network::TcpSrc;

    println!("Connecting to TCP server at {}", ADDRESS);

    let received = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();

    // TcpSrc in client mode - connects to the server
    // Wrap in SourceAdapter since TcpSrc implements Source trait
    let tcp_src = TcpSrc::connect(ADDRESS)?;
    let src = pipeline.add_node(
        "tcp_source",
        DynAsyncElement::new_box(SourceAdapter::new(tcp_src)),
    );

    let sink = pipeline.add_node(
        "print_sink",
        DynAsyncElement::new_box(SinkAdapter::new(PrintSink {
            received: received.clone(),
        })),
    );

    pipeline.link(src, sink)?;

    println!("Receiving messages from server...\n");
    pipeline.run().await?;

    println!(
        "\nClient finished. Received {} messages.",
        received.load(Ordering::Relaxed)
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        println!("Usage: {} <server|client>", args[0]);
        println!();
        println!("Run server first in one terminal:");
        println!("  cargo run --example 21_tcp_streaming -- server");
        println!();
        println!("Then run client in another terminal:");
        println!("  cargo run --example 21_tcp_streaming -- client");
        return Ok(());
    }

    match args[1].as_str() {
        "server" => run_server().await,
        "client" => run_client().await,
        _ => {
            println!("Unknown mode: {}. Use 'server' or 'client'.", args[1]);
            Ok(())
        }
    }
}
