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
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::thread;

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let text = std::str::from_utf8(ctx.input()).unwrap_or("<invalid>");
        println!("[Sink] Received: {}", text);
        Ok(())
    }
}

fn make_buffer(arena: &SharedArena, data: &str, seq: u64) -> Buffer {
    let bytes = data.as_bytes();
    let mut slot = arena.acquire().expect("arena exhausted");
    slot.data_mut()[..bytes.len()].copy_from_slice(bytes);
    Buffer::new(
        MemoryHandle::with_len(slot, bytes.len()),
        Metadata::from_sequence(seq),
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create a shared arena for producer buffers
    let producer_arena = SharedArena::new(256, 16)?;

    // Create funnel and get input handles
    let funnel = Funnel::new();
    let input_a = funnel.new_input();
    let input_b = funnel.new_input();

    // Clone arenas for each producer thread
    let arena_a = producer_arena.clone();
    let arena_b = producer_arena.clone();

    // Spawn threads to push data into the funnel
    let producer_a = thread::spawn(move || {
        for i in 1..=3 {
            let msg = format!("A:{}", i);
            let _ = input_a.push(make_buffer(&arena_a, &msg, i));
            println!("[Producer A] Sent: {}", msg);
        }
        input_a.end_stream();
    });

    let producer_b = thread::spawn(move || {
        for i in 1..=3 {
            let msg = format!("B:{}", i);
            let _ = input_b.push(make_buffer(&arena_b, &msg, i));
            println!("[Producer B] Sent: {}", msg);
        }
        input_b.end_stream();
    });

    // Build pipeline: Funnel (as source) -> Sink
    let pipeline_arena = SharedArena::new(256, 8)?;
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source_with_arena("funnel", funnel, pipeline_arena);
    let sink = pipeline.add_sink("sink", PrintSink);
    pipeline.link(src, sink)?;

    // Run pipeline
    pipeline.run().await?;

    // Wait for producers
    producer_a.join().unwrap();
    producer_b.join().unwrap();

    Ok(())
}
