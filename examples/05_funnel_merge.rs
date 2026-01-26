//! Using Funnel to merge multiple sources into one sink.
//!
//! Run with: cargo run --example 05_funnel_merge

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{DynAsyncElement, Sink, SinkAdapter, Source, SourceAdapter};
use parallax::elements::{Funnel, FunnelInput};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

struct TaggedSource {
    tag: u8,
    current: u64,
    max: u64,
}

impl Source for TaggedSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.current >= self.max {
            return Ok(None);
        }
        // First byte is tag, rest is counter
        let segment = Arc::new(HeapSegment::new(9)?);
        unsafe {
            let ptr = segment.as_mut_ptr().unwrap();
            *ptr = self.tag;
            std::ptr::copy_nonoverlapping(self.current.to_le_bytes().as_ptr(), ptr.add(1), 8);
        }
        let buffer = Buffer::new(
            MemoryHandle::from_segment(segment),
            Metadata::from_sequence(self.current),
        );
        self.current += 1;
        Ok(Some(buffer))
    }
}

struct PrintSink {
    received: Arc<AtomicU64>,
}

impl Sink for PrintSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let data = buffer.as_bytes();
        let tag = data[0];
        let value = u64::from_le_bytes(data[1..9].try_into().unwrap());
        println!("From source {}: value {}", tag as char, value);
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Pipeline: [Source A, Source B] -> Funnel -> Sink");
    println!();

    let received = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();

    let src_a = pipeline.add_node(
        "source_a",
        DynAsyncElement::new_box(SourceAdapter::new(TaggedSource {
            tag: b'A',
            current: 0,
            max: 3,
        })),
    );
    let src_b = pipeline.add_node(
        "source_b",
        DynAsyncElement::new_box(SourceAdapter::new(TaggedSource {
            tag: b'B',
            current: 0,
            max: 3,
        })),
    );

    // Funnel merges multiple inputs into one output (it's a Source that reads from FunnelInputs)
    let funnel = pipeline.add_node(
        "funnel",
        DynAsyncElement::new_box(SourceAdapter::new(Funnel::new())),
    );

    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(PrintSink {
            received: received.clone(),
        })),
    );

    pipeline.link(src_a, funnel)?;
    pipeline.link(src_b, funnel)?;
    pipeline.link(funnel, sink)?;

    pipeline.run().await?;

    println!();
    println!(
        "Total received: {} buffers",
        received.load(Ordering::Relaxed)
    );

    Ok(())
}
