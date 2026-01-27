//! # Funnel (Fan-in)
//!
//! Merge multiple streams into one using a Funnel.
//! Funnel collects buffers from multiple inputs and produces them in order.
//!
//! ```text
//! [SourceA] → FunnelInput ─┐
//!                          ├→ [Funnel] → [PrintSink]
//! [SourceB] → FunnelInput ─┘
//! ```
//!
//! Note: Funnel uses a push model with explicit input handles rather than
//! pipeline links. This is useful for dynamic fan-in scenarios.
//!
//! Run: `cargo run --example 04_funnel`

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ConsumeContext, Sink};
use parallax::elements::Funnel;
use parallax::error::Result;
use parallax::memory::{CpuArena, HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::thread;

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let text = std::str::from_utf8(ctx.input()).unwrap_or("<invalid>");
        println!("[Sink] Received: {}", text);
        Ok(())
    }
}

fn make_buffer(data: &str, seq: u64) -> Buffer {
    let bytes = data.as_bytes();
    let segment = Arc::new(HeapSegment::new(bytes.len()).unwrap());
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), segment.as_mut_ptr().unwrap(), bytes.len());
    }
    Buffer::new(
        MemoryHandle::from_segment_with_len(segment, bytes.len()),
        Metadata::from_sequence(seq),
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create funnel and get input handles
    let funnel = Funnel::new();
    let input_a = funnel.new_input();
    let input_b = funnel.new_input();

    // Spawn threads to push data into the funnel
    let producer_a = thread::spawn(move || {
        for i in 1..=3 {
            let msg = format!("A:{}", i);
            let _ = input_a.push(make_buffer(&msg, i));
            println!("[Producer A] Sent: {}", msg);
        }
        input_a.end_stream();
    });

    let producer_b = thread::spawn(move || {
        for i in 1..=3 {
            let msg = format!("B:{}", i);
            let _ = input_b.push(make_buffer(&msg, i));
            println!("[Producer B] Sent: {}", msg);
        }
        input_b.end_stream();
    });

    // Build pipeline: Funnel (as source) -> Sink
    let arena = CpuArena::new(256, 8)?;
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source_with_arena("funnel", funnel, arena);
    let sink = pipeline.add_sink("sink", PrintSink);
    pipeline.link(src, sink)?;

    // Run pipeline
    pipeline.run().await?;

    // Wait for producers
    producer_a.join().unwrap();
    producer_b.join().unwrap();

    Ok(())
}
