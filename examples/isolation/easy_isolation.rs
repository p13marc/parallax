//! Easy process isolation - the simple API.
//!
//! This example shows the easiest way to use process isolation.
//! Just write a normal pipeline and call the right run method!
//!
//! # TL;DR
//!
//! ```rust,ignore
//! // Normal pipeline - no IPC elements needed
//! let pipeline = Pipeline::new();
//! pipeline.add_node("src", source);
//! pipeline.add_node("risky_decoder", decoder);  // Might crash!
//! pipeline.add_node("sink", sink);
//!
//! // Option 1: Run everything in one process (default)
//! pipeline.run().await?;
//!
//! // Option 2: Isolate specific elements by name pattern
//! pipeline.run_isolating(vec!["*decoder*"]).await?;
//!
//! // Option 3: Isolate everything
//! pipeline.run_isolated().await?;
//! ```
//!
//! Run with: cargo run --example easy_isolation

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    DynAsyncElement, Element, ElementAdapter, Sink, SinkAdapter, Source, SourceAdapter,
};
use parallax::error::Result;
use parallax::execution::ExecutionMode;
use parallax::memory::HeapSegment;
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Simple elements for demonstration
// ============================================================================

/// Produces N buffers.
struct SimpleSource {
    count: u64,
    max: u64,
}

impl SimpleSource {
    fn new(n: u64) -> Self {
        Self { count: 0, max: n }
    }
}

impl Source for SimpleSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.count >= self.max {
            return Ok(None);
        }
        let seg = Arc::new(HeapSegment::new(64)?);
        let buf = Buffer::new(
            MemoryHandle::from_segment(seg),
            Metadata::from_sequence(self.count),
        );
        self.count += 1;
        Ok(Some(buf))
    }
}

/// Passes buffers through (simulates decoder/encoder).
struct PassthroughTransform {
    name: String,
}

impl PassthroughTransform {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
        }
    }
}

impl Element for PassthroughTransform {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        Ok(Some(buffer))
    }
}

/// Counts received buffers.
struct CountingSink {
    counter: Arc<AtomicU64>,
}

impl CountingSink {
    fn new(c: Arc<AtomicU64>) -> Self {
        Self { counter: c }
    }
}

impl Sink for CountingSink {
    fn consume(&mut self, _: Buffer) -> Result<()> {
        self.counter.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

// ============================================================================
// Helper to build a typical video processing pipeline
// ============================================================================

fn build_video_pipeline(sink_counter: Arc<AtomicU64>) -> Pipeline {
    let mut p = Pipeline::new();

    // Typical video pipeline: source -> decoder -> filter -> encoder -> sink
    let src = p.add_node(
        "camera_source",
        DynAsyncElement::new_box(SourceAdapter::new(SimpleSource::new(100))),
    );

    let dec = p.add_node(
        "h264_decoder", // Decoders can crash on bad input!
        DynAsyncElement::new_box(ElementAdapter::new(PassthroughTransform::new("decoder"))),
    );

    let filt = p.add_node(
        "blur_filter",
        DynAsyncElement::new_box(ElementAdapter::new(PassthroughTransform::new("filter"))),
    );

    let enc = p.add_node(
        "h265_encoder", // Encoders are CPU-heavy
        DynAsyncElement::new_box(ElementAdapter::new(PassthroughTransform::new("encoder"))),
    );

    let sink = p.add_node(
        "file_sink",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink::new(sink_counter))),
    );

    p.link(src, dec).unwrap();
    p.link(dec, filt).unwrap();
    p.link(filt, enc).unwrap();
    p.link(enc, sink).unwrap();

    p
}

// ============================================================================
// Main examples
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           Easy Process Isolation Examples                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝\n");

    // -------------------------------------------------------------------------
    // Example 1: Default - everything in one process
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Example 1: In-Process (Default)                             │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!("  Code: pipeline.run().await?");
    println!("  ");
    println!("  ┌────────┐   ┌─────────┐   ┌────────┐   ┌─────────┐   ┌──────┐");
    println!("  │ Source │──▶│ Decoder │──▶│ Filter │──▶│ Encoder │──▶│ Sink │");
    println!("  └────────┘   └─────────┘   └────────┘   └─────────┘   └──────┘");
    println!("  └─────────────────────── Same Process ───────────────────────┘");

    let counter = Arc::new(AtomicU64::new(0));
    let mut pipeline = build_video_pipeline(counter.clone());
    pipeline.run().await?;
    println!(
        "\n  ✓ Processed {} buffers\n",
        counter.load(Ordering::Relaxed)
    );

    // -------------------------------------------------------------------------
    // Example 2: Isolate decoders (they might crash on bad input!)
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Example 2: Isolate Decoders (Crash Protection)              │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!("  Code: pipeline.run_isolating(vec![\"*decoder*\"]).await?");
    println!("  ");
    println!("  ┌────────┐        ╔═════════════════╗        ┌──────┐");
    println!("  │ Source │──IPC──▶║ Decoder Process ║──IPC──▶│ Rest │");
    println!("  └────────┘        ╚═════════════════╝        └──────┘");
    println!("  └─ Main Process ─┘                           └──────┘");

    let counter = Arc::new(AtomicU64::new(0));
    let pipeline = build_video_pipeline(counter.clone());
    pipeline.run_isolating(vec!["*decoder*"]).await?;
    println!(
        "\n  ✓ Processed {} buffers (decoder was isolated)\n",
        counter.load(Ordering::Relaxed)
    );

    // -------------------------------------------------------------------------
    // Example 3: Isolate both decoders AND encoders
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Example 3: Isolate Decoders and Encoders                    │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!("  Code: pipeline.run_isolating(vec![\"*decoder*\", \"*encoder*\"]).await?");
    println!("  ");
    println!(
        "  ┌────────┐        ╔═════════╗        ┌────────┐        ╔═════════╗        ┌──────┐"
    );
    println!(
        "  │ Source │──IPC──▶║ Decoder ║──IPC──▶│ Filter │──IPC──▶║ Encoder ║──IPC──▶│ Sink │"
    );
    println!(
        "  └────────┘        ╚═════════╝        └────────┘        ╚═════════╝        └──────┘"
    );
    println!(
        "  └─ Main ─┘        └ Process ┘        └─ Main ─┘        └ Process ┘        └ Main ┘"
    );

    let counter = Arc::new(AtomicU64::new(0));
    let pipeline = build_video_pipeline(counter.clone());
    pipeline
        .run_isolating(vec!["*decoder*", "*encoder*"])
        .await?;
    println!(
        "\n  ✓ Processed {} buffers (decoder + encoder isolated)\n",
        counter.load(Ordering::Relaxed)
    );

    // -------------------------------------------------------------------------
    // Example 4: Full isolation (each element in its own process)
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Example 4: Full Isolation (Maximum Security)                │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!("  Code: pipeline.run_isolated().await?");
    println!("  ");
    println!("  ╔════════╗       ╔═════════╗       ╔════════╗       ╔═════════╗       ╔══════╗");
    println!("  ║ Source ║──IPC──║ Decoder ║──IPC──║ Filter ║──IPC──║ Encoder ║──IPC──║ Sink ║");
    println!("  ╚════════╝       ╚═════════╝       ╚════════╝       ╚═════════╝       ╚══════╝");
    println!("  └Process 1       └Process 2        └Process 3       └Process 4        └Process 5");

    let counter = Arc::new(AtomicU64::new(0));
    let pipeline = build_video_pipeline(counter.clone());
    pipeline.run_isolated().await?;
    println!(
        "\n  ✓ Processed {} buffers (each element isolated)\n",
        counter.load(Ordering::Relaxed)
    );

    // -------------------------------------------------------------------------
    // Example 5: Using ExecutionMode directly for more control
    // -------------------------------------------------------------------------
    println!("┌─────────────────────────────────────────────────────────────┐");
    println!("│ Example 5: Custom ExecutionMode                             │");
    println!("└─────────────────────────────────────────────────────────────┘");
    println!("  Code: pipeline.run_with_mode(ExecutionMode::grouped(patterns)).await?");

    let counter = Arc::new(AtomicU64::new(0));
    let pipeline = build_video_pipeline(counter.clone());

    // Custom mode: isolate anything with "h264" or "h265" in the name
    let mode = ExecutionMode::grouped(vec!["*h264*".to_string(), "*h265*".to_string()]);

    pipeline.run_with_mode(mode).await?;
    println!(
        "\n  ✓ Processed {} buffers (h264/h265 elements isolated)\n",
        counter.load(Ordering::Relaxed)
    );

    // -------------------------------------------------------------------------
    // Summary
    // -------------------------------------------------------------------------
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║                         Summary                               ║");
    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║ Method                        │ Use Case                      ║");
    println!("╟───────────────────────────────┼───────────────────────────────╢");
    println!("║ pipeline.run()                │ Default, no isolation         ║");
    println!("║ pipeline.run_isolating([...]) │ Isolate specific elements     ║");
    println!("║ pipeline.run_isolated()       │ Isolate everything            ║");
    println!("║ pipeline.run_with_mode(mode)  │ Full control over isolation   ║");
    println!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
