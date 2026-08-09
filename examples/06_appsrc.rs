//! # AppSrc and AppSink
//!
//! Application integration: push data into a pipeline with AppSrc,
//! and pull results out with AppSink.
//!
//! ```text
//! [AppSrc] → [AppSink]
//! ```
//!
//! Run: `cargo run --example 06_appsrc`

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::elements::{AppSink, AppSrc, EndReason, Pulled};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::thread;
use std::time::Duration;

fn make_buffer(arena: &SharedArena, data: &[u8], seq: u64) -> Buffer {
    let mut slot = arena.acquire().expect("arena exhausted");
    slot.data_mut()[..data.len()].copy_from_slice(data);
    Buffer::new(
        MemoryHandle::with_len(slot, data.len()),
        Metadata::from_sequence(seq),
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create AppSrc and get handle for pushing
    let appsrc = AppSrc::new();
    let src_handle = appsrc.handle();

    // Create AppSink and get handle for pulling
    let appsink = AppSink::new();
    let sink_handle = appsink.handle();

    // Create arena for producer buffers
    let producer_arena = SharedArena::new(256, 16)?;

    // Build pipeline
    let pipeline_arena = SharedArena::new(256, 8)?;
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source_with_arena("appsrc", appsrc, pipeline_arena);
    let sink = pipeline.add_sink("appsink", appsink);
    pipeline.link(src, sink)?;

    // Spawn producer thread
    let producer = thread::spawn(move || {
        for i in 0..5u64 {
            let msg = format!("Message {}", i);
            src_handle
                .push_buffer_blocking(make_buffer(&producer_arena, msg.as_bytes(), i))
                .unwrap();
            println!("[Producer] Pushed: {}", msg);
            thread::sleep(Duration::from_millis(10));
        }
        src_handle.end_stream();
        println!("[Producer] Sent EOS");
    });

    // Spawn consumer thread
    let consumer = thread::spawn(move || {
        loop {
            // Every way a pull can end has a name, so the consumer can react
            // to each one differently instead of guessing what `None` meant.
            match sink_handle.pull_buffer_blocking() {
                Pulled::Buffer(buffer) => {
                    let text = std::str::from_utf8(buffer.as_bytes()).unwrap_or("<invalid>");
                    println!("[Consumer] Received: {}", text);
                }
                Pulled::Ended(EndReason::Eos) => {
                    println!("[Consumer] Stream ended cleanly");
                    break;
                }
                Pulled::Ended(EndReason::Error(err)) => {
                    println!("[Consumer] Pipeline failed: {err}");
                    break;
                }
                Pulled::Ended(EndReason::Aborted) => {
                    println!("[Consumer] Pipeline aborted");
                    break;
                }
                // The blocking pull has no timeout, so neither of these can
                // happen here; a `_timeout` variant would return Empty.
                Pulled::Empty | Pulled::Flushing => continue,
            }
        }
    });

    // Run pipeline
    pipeline.run().await?;

    producer.join().unwrap();
    consumer.join().unwrap();

    Ok(())
}
