//! Hello World pipeline - the simplest possible example.
//!
//! Run with: cargo run --example 01_hello_pipeline

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{DynAsyncElement, Sink, SinkAdapter, Source, SourceAdapter};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;

struct HelloSource {
    sent: bool,
}

impl Source for HelloSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.sent {
            return Ok(None);
        }
        self.sent = true;

        let data = b"Hello, Pipeline!";
        let segment = Arc::new(HeapSegment::new(data.len())?);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), segment.as_mut_ptr().unwrap(), data.len());
        }

        Ok(Some(Buffer::new(
            MemoryHandle::from_segment(segment),
            Metadata::new(),
        )))
    }
}

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let text = std::str::from_utf8(buffer.as_bytes()).unwrap_or("<invalid utf8>");
        println!("Received: {}", text);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(HelloSource { sent: false })),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(PrintSink)),
    );

    pipeline.link(src, sink)?;
    pipeline.run().await
}
