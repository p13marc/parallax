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
use parallax::element::{DynAsyncElement, SinkAdapter, SourceAdapter};
use parallax::elements::{AppSink, AppSrc};
use parallax::error::Result;
use parallax::memory::{CpuArena, HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

fn make_buffer(data: &[u8], seq: u64) -> Buffer {
    let segment = Arc::new(HeapSegment::new(data.len()).unwrap());
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), segment.as_mut_ptr().unwrap(), data.len());
    }
    Buffer::new(
        MemoryHandle::from_segment_with_len(segment, data.len()),
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

    // Build pipeline
    let arena = CpuArena::new(256, 8)?;
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node(
        "appsrc",
        DynAsyncElement::new_box(SourceAdapter::with_arena(appsrc, arena)),
    );
    let sink = pipeline.add_node(
        "appsink",
        DynAsyncElement::new_box(SinkAdapter::new(appsink)),
    );
    pipeline.link(src, sink)?;

    // Spawn producer thread
    let producer = thread::spawn(move || {
        for i in 0..5u64 {
            let msg = format!("Message {}", i);
            src_handle
                .push_buffer(make_buffer(msg.as_bytes(), i))
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
            match sink_handle.pull_buffer() {
                Ok(Some(buffer)) => {
                    let text = std::str::from_utf8(buffer.as_bytes()).unwrap_or("<invalid>");
                    println!("[Consumer] Received: {}", text);
                }
                Ok(None) => {
                    println!("[Consumer] Got EOS");
                    break;
                }
                Err(e) => {
                    println!("[Consumer] Error: {}", e);
                    break;
                }
            }
        }
    });

    // Run pipeline
    pipeline.run().await?;

    producer.join().unwrap();
    consumer.join().unwrap();

    Ok(())
}
