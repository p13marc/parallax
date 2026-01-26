//! A source that produces multiple buffers with sequence numbers.
//!
//! Run with: cargo run --example 02_counting_source

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{DynAsyncElement, Sink, SinkAdapter, Source, SourceAdapter};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;

struct CountingSource {
    current: u64,
    max: u64,
}

impl Source for CountingSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.current >= self.max {
            return Ok(None); // End of stream
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

struct PrintSink {
    count: u64,
}

impl Sink for PrintSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let value = u64::from_le_bytes(buffer.as_bytes().try_into().unwrap());
        println!("Buffer {}: value = {}", self.count, value);
        self.count += 1;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "counter",
        DynAsyncElement::new_box(SourceAdapter::new(CountingSource { current: 0, max: 5 })),
    );
    let sink = pipeline.add_node(
        "printer",
        DynAsyncElement::new_box(SinkAdapter::new(PrintSink { count: 0 })),
    );

    pipeline.link(src, sink)?;
    pipeline.run().await
}
