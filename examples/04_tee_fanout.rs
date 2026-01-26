//! Using Tee to send data to multiple sinks.
//!
//! Run with: cargo run --example 04_tee_fanout

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    DynAsyncElement, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter,
};
use parallax::elements::Tee;
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

struct NumberSource {
    current: u64,
    max: u64,
}

impl Source for NumberSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.current >= self.max {
            return Ok(None);
        }
        let segment = Arc::new(HeapSegment::new(8)?);
        let data = self.current.to_le_bytes();
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), segment.as_mut_ptr().unwrap(), 8);
        }
        let buffer = Buffer::new(
            MemoryHandle::from_segment(segment),
            Metadata::from_sequence(self.current),
        );
        self.current += 1;
        Ok(Some(buffer))
    }
}

struct CountingSink {
    name: String,
    counter: Arc<AtomicU64>,
}

impl Sink for CountingSink {
    fn consume(&mut self, _buffer: Buffer) -> Result<()> {
        self.counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Pipeline: Source -> Tee -> [Sink A, Sink B, Sink C]");
    println!();

    let counter_a = Arc::new(AtomicU64::new(0));
    let counter_b = Arc::new(AtomicU64::new(0));
    let counter_c = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(NumberSource {
            current: 0,
            max: 10,
        })),
    );

    // Tee duplicates buffers to all connected sinks
    let tee = pipeline.add_node(
        "tee",
        DynAsyncElement::new_box(ElementAdapter::new(Tee::new())),
    );

    let sink_a = pipeline.add_node(
        "sink_a",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            name: "A".into(),
            counter: counter_a.clone(),
        })),
    );
    let sink_b = pipeline.add_node(
        "sink_b",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            name: "B".into(),
            counter: counter_b.clone(),
        })),
    );
    let sink_c = pipeline.add_node(
        "sink_c",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            name: "C".into(),
            counter: counter_c.clone(),
        })),
    );

    pipeline.link(src, tee)?;
    pipeline.link(tee, sink_a)?;
    pipeline.link(tee, sink_b)?;
    pipeline.link(tee, sink_c)?;

    pipeline.run().await?;

    println!(
        "Sink A received: {} buffers",
        counter_a.load(Ordering::Relaxed)
    );
    println!(
        "Sink B received: {} buffers",
        counter_b.load(Ordering::Relaxed)
    );
    println!(
        "Sink C received: {} buffers",
        counter_c.load(Ordering::Relaxed)
    );

    Ok(())
}
