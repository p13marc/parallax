//! Valve for on/off flow control.
//!
//! Run with: cargo run --example 09_valve_control

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    DynAsyncElement, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter,
};
use parallax::elements::Valve;
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

struct CountingSink {
    received: Arc<AtomicU64>,
}

impl Sink for CountingSink {
    fn consume(&mut self, _buffer: Buffer) -> Result<()> {
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Valve starts closed, drops all buffers");
    println!();

    let received = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(NumberSource {
            current: 0,
            max: 10,
        })),
    );

    // Valve starts closed (open=false)
    let valve = pipeline.add_node(
        "valve",
        DynAsyncElement::new_box(ElementAdapter::new(Valve::with_state(false))),
    );

    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            received: received.clone(),
        })),
    );

    pipeline.link(src, valve)?;
    pipeline.link(valve, sink)?;

    pipeline.run().await?;

    println!("Sent: 10 buffers");
    println!(
        "Received: {} buffers (valve was closed)",
        received.load(Ordering::Relaxed)
    );

    Ok(())
}
