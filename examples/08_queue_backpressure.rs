//! Queue element with backpressure control.
//!
//! Run with: cargo run --example 08_queue_backpressure

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    DynAsyncElement, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter,
};
use parallax::elements::{LeakyMode, Queue};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

struct FastSource {
    current: u64,
    max: u64,
}

impl Source for FastSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.current >= self.max {
            return Ok(None);
        }
        let segment = Arc::new(HeapSegment::new(8)?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.current.to_le_bytes().as_ptr(),
                segment.as_mut_ptr().unwrap(),
                8,
            );
        }
        let buffer = Buffer::new(
            MemoryHandle::from_segment(segment),
            Metadata::from_sequence(self.current),
        );
        self.current += 1;
        Ok(Some(buffer))
    }
}

struct SlowSink {
    received: Arc<AtomicU64>,
}

impl Sink for SlowSink {
    fn consume(&mut self, _buffer: Buffer) -> Result<()> {
        // Simulate slow processing
        std::thread::sleep(Duration::from_millis(10));
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Queue with capacity 4, leaky mode drops old buffers when full");
    println!();

    let received = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "fast_source",
        DynAsyncElement::new_box(SourceAdapter::new(FastSource {
            current: 0,
            max: 20,
        })),
    );

    // Queue with capacity 4, drops oldest when full
    let queue = pipeline.add_node(
        "queue",
        DynAsyncElement::new_box(ElementAdapter::new(
            Queue::new(4).leaky(LeakyMode::Upstream),
        )),
    );

    let sink = pipeline.add_node(
        "slow_sink",
        DynAsyncElement::new_box(SinkAdapter::new(SlowSink {
            received: received.clone(),
        })),
    );

    pipeline.link(src, queue)?;
    pipeline.link(queue, sink)?;

    pipeline.run().await?;

    println!("Sent: 20 buffers");
    println!(
        "Received: {} buffers (some dropped due to backpressure)",
        received.load(Ordering::Relaxed)
    );

    Ok(())
}
