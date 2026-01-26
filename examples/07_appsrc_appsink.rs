//! AppSrc and AppSink for application integration.
//!
//! Run with: cargo run --example 07_appsrc_appsink

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{DynAsyncElement, SinkAdapter, SourceAdapter};
use parallax::elements::{AppSink, AppSrc};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::thread;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Injecting and extracting data via AppSrc/AppSink");
    println!();

    let appsrc = AppSrc::new();
    let src_handle = appsrc.handle();

    let appsink = AppSink::new();
    let sink_handle = appsink.handle();

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "appsrc",
        DynAsyncElement::new_box(SourceAdapter::new(appsrc)),
    );
    let sink = pipeline.add_node(
        "appsink",
        DynAsyncElement::new_box(SinkAdapter::new(appsink)),
    );

    pipeline.link(src, sink)?;

    // Producer thread: push data into pipeline
    let producer = thread::spawn(move || {
        for i in 0..5u64 {
            let segment = Arc::new(HeapSegment::new(8).unwrap());
            unsafe {
                std::ptr::copy_nonoverlapping(
                    i.to_le_bytes().as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    8,
                );
            }
            let buffer = Buffer::new(
                MemoryHandle::from_segment(segment),
                Metadata::from_sequence(i),
            );
            src_handle.push_buffer(buffer).unwrap();
            println!("Pushed: {}", i);
        }
        src_handle.end_stream();
    });

    // Consumer thread: pull data from pipeline
    let consumer = thread::spawn(move || {
        while let Ok(Some(buffer)) = sink_handle.pull_buffer() {
            let bytes: [u8; 8] = buffer.as_bytes().try_into().unwrap();
            let value = u64::from_le_bytes(bytes);
            println!("Pulled: {}", value);
        }
    });

    // Run pipeline
    pipeline.run().await?;

    producer.join().unwrap();
    consumer.join().unwrap();

    Ok(())
}
