//! Isolate specific elements by name pattern.
//!
//! Run with: cargo run --example 12_isolate_by_pattern

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    DynAsyncElement, Element, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter,
};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

struct Source10 {
    n: u64,
}
impl Source for Source10 {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.n >= 10 {
            return Ok(None);
        }
        let seg = Arc::new(HeapSegment::new(8)?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                self.n.to_le_bytes().as_ptr(),
                seg.as_mut_ptr().unwrap(),
                8,
            );
        }
        let buf = Buffer::new(
            MemoryHandle::from_segment(seg),
            Metadata::from_sequence(self.n),
        );
        self.n += 1;
        Ok(Some(buf))
    }
}

struct Passthrough;
impl Element for Passthrough {
    fn process(&mut self, buf: Buffer) -> Result<Option<Buffer>> {
        Ok(Some(buf))
    }
}

struct Counter(Arc<AtomicU64>);
impl Sink for Counter {
    fn consume(&mut self, _: Buffer) -> Result<()> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Isolate elements matching '*decoder*' pattern");
    println!();
    println!("Pipeline: source -> h264_decoder -> sink");
    println!("          (main)    (isolated)     (main)");
    println!();

    let counter = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node(
        "video_source",
        DynAsyncElement::new_box(SourceAdapter::new(Source10 { n: 0 })),
    );
    let dec = pipeline.add_node(
        "h264_decoder",
        DynAsyncElement::new_box(ElementAdapter::new(Passthrough)),
    );
    let sink = pipeline.add_node(
        "display_sink",
        DynAsyncElement::new_box(SinkAdapter::new(Counter(counter.clone()))),
    );

    pipeline.link(src, dec)?;
    pipeline.link(dec, sink)?;

    // Isolate elements matching "*decoder*"
    pipeline.run_isolating(vec!["*decoder*"]).await?;

    println!("Processed: {} buffers", counter.load(Ordering::Relaxed));
    Ok(())
}
