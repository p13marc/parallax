//! Adding a transform element between source and sink.
//!
//! Run with: cargo run --example 03_transform_element

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    DynAsyncElement, Element, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter,
};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;

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

/// Doubles every number that passes through.
struct DoubleTransform;

impl Element for DoubleTransform {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let value = u64::from_le_bytes(buffer.as_bytes().try_into().unwrap());
        let doubled = value * 2;

        // Create new buffer with doubled value
        let segment = Arc::new(HeapSegment::new(8)?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                doubled.to_le_bytes().as_ptr(),
                segment.as_mut_ptr().unwrap(),
                8,
            );
        }

        Ok(Some(Buffer::new(
            MemoryHandle::from_segment(segment),
            buffer.metadata().clone(),
        )))
    }
}

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let value = u64::from_le_bytes(buffer.as_bytes().try_into().unwrap());
        println!("Received: {}", value);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Pipeline: Source(0..5) -> Double -> Print");
    println!();

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "numbers",
        DynAsyncElement::new_box(SourceAdapter::new(NumberSource { current: 0, max: 5 })),
    );
    let double = pipeline.add_node(
        "double",
        DynAsyncElement::new_box(ElementAdapter::new(DoubleTransform)),
    );
    let sink = pipeline.add_node(
        "print",
        DynAsyncElement::new_box(SinkAdapter::new(PrintSink)),
    );

    pipeline.link(src, double)?;
    pipeline.link(double, sink)?;

    pipeline.run().await
}
