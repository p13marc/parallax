//! Example 19: Automatic Execution Strategy
//!
//! This example demonstrates how the unified Executor automatically determines
//! the optimal execution strategy for each element based on ExecutionHints.
//!
//! Key concepts:
//! - **ExecutionHints**: Describes element characteristics (trust, latency, etc.)
//! - **Automatic Strategy**: The executor analyzes hints to choose Async/RT/Isolated
//! - **No Developer Insight Required**: Just run the pipeline and it "does the right thing"
//!
//! The executor uses these rules:
//! - Untrusted elements OR native code without crash safety -> Isolated process
//! - RT affinity + RT-safe elements -> RT thread
//! - Low latency + RT-safe elements -> RT thread
//! - I/O-bound elements -> Async (Tokio task)
//! - Everything else -> Async (default)
//!
//! Run with: cargo run --example 19_auto_execution

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    DynAsyncElement, Element, ElementAdapter, ExecutionHints, ProcessingHint, Sink, SinkAdapter,
    Source, SourceAdapter,
};
use parallax::error::Result;
use parallax::memory::HeapSegment;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline, UnifiedExecutorConfig};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A simple source that produces N buffers.
struct TestSource {
    current: u64,
    max: u64,
}

impl Source for TestSource {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        if self.current >= self.max {
            return Ok(None);
        }
        let segment = Arc::new(HeapSegment::new(8)?);
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::from_sequence(self.current));
        self.current += 1;
        Ok(Some(buffer))
    }

    // Override execution_hints to indicate this is I/O-bound
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

/// A "native" decoder element that uses FFI (simulated).
/// This would trigger isolation in a real pipeline.
struct NativeDecoder;

impl Element for NativeDecoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // In reality, this would call into a C library
        Ok(Some(buffer))
    }

    // Indicate this uses native code and is not crash-safe
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::native()
    }
}

/// An untrusted parser element.
/// This would trigger isolation because it processes untrusted input.
struct UntrustedParser;

impl Element for UntrustedParser {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // In reality, this would parse untrusted data
        Ok(Some(buffer))
    }

    // Indicate this is untrusted
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::untrusted()
    }
}

/// A low-latency audio processor.
/// This would run in RT threads if RT-safe.
struct LowLatencyProcessor;

impl Element for LowLatencyProcessor {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Low-latency audio processing
        Ok(Some(buffer))
    }

    // Indicate low latency requirements
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::low_latency()
    }

    // Indicate this is RT-safe (no allocations in hot path)
    fn is_rt_safe(&self) -> bool {
        true
    }
}

/// A CPU-bound encoder element.
struct CpuBoundEncoder;

impl Element for CpuBoundEncoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // CPU-intensive encoding
        Ok(Some(buffer))
    }

    // Indicate this is CPU-bound
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints {
            processing: ProcessingHint::CpuBound,
            ..ExecutionHints::default()
        }
    }
}

/// A trusted sink that writes to disk.
struct TrustedSink {
    received: Arc<AtomicU64>,
}

impl Sink for TrustedSink {
    fn consume(&mut self, _buffer: Buffer) -> Result<()> {
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    // Default hints: trusted, I/O-bound
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing to see executor decisions
    tracing_subscriber::fmt()
        .with_env_filter("parallax=debug")
        .init();

    println!("=== Automatic Execution Strategy Example ===\n");

    // Example 1: Simple pipeline with default (auto) strategy
    println!("1. Default (Auto) Strategy:");
    println!("   The executor analyzes each element's ExecutionHints\n");

    let counter = Arc::new(AtomicU64::new(0));
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "test_source",
        DynAsyncElement::new_box(SourceAdapter::new(TestSource { current: 0, max: 5 })),
    );
    let sink = pipeline.add_node(
        "trusted_sink",
        DynAsyncElement::new_box(SinkAdapter::new(TrustedSink {
            received: counter.clone(),
        })),
    );
    pipeline.link(src, sink)?;

    // Run with default (auto) configuration
    let executor = Executor::new();
    println!("   Running pipeline with auto strategy...");
    executor.run(&mut pipeline).await?;
    println!(
        "   Processed: {} buffers\n",
        counter.load(Ordering::Relaxed)
    );

    // Example 2: Pipeline with mixed element types
    println!("2. Mixed Element Types:");
    println!("   - TestSource (I/O-bound) -> Async");
    println!("   - NativeDecoder (native code) -> would be Isolated");
    println!("   - LowLatencyProcessor (low latency + RT-safe) -> would be RT");
    println!("   - TrustedSink (I/O-bound) -> Async\n");

    let counter2 = Arc::new(AtomicU64::new(0));
    let mut pipeline2 = Pipeline::new();

    let src2 = pipeline2.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(TestSource { current: 0, max: 5 })),
    );
    let decoder = pipeline2.add_node(
        "native_decoder",
        DynAsyncElement::new_box(ElementAdapter::new(NativeDecoder)),
    );
    let processor = pipeline2.add_node(
        "low_latency",
        DynAsyncElement::new_box(ElementAdapter::new(LowLatencyProcessor)),
    );
    let sink2 = pipeline2.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(TrustedSink {
            received: counter2.clone(),
        })),
    );

    pipeline2.link(src2, decoder)?;
    pipeline2.link(decoder, processor)?;
    pipeline2.link(processor, sink2)?;

    println!("   Running pipeline with auto strategy...");
    println!("   (Note: isolation not fully implemented yet, will log warnings)");
    executor.run(&mut pipeline2).await?;
    println!(
        "   Processed: {} buffers\n",
        counter2.load(Ordering::Relaxed)
    );

    // Example 3: Explicit async-only configuration
    println!("3. Explicit Async-Only Configuration:");
    println!("   Disables auto-detection, all elements run as Tokio tasks\n");

    let counter3 = Arc::new(AtomicU64::new(0));
    let mut pipeline3 = Pipeline::new();

    let src3 = pipeline3.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(TestSource { current: 0, max: 5 })),
    );
    let sink3 = pipeline3.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(TrustedSink {
            received: counter3.clone(),
        })),
    );
    pipeline3.link(src3, sink3)?;

    let executor = Executor::with_config(UnifiedExecutorConfig::async_only());
    println!("   Running pipeline with async_only config...");
    executor.run(&mut pipeline3).await?;
    println!(
        "   Processed: {} buffers\n",
        counter3.load(Ordering::Relaxed)
    );

    // Example 4: Print execution hints for various elements
    println!("4. Inspecting Element ExecutionHints:");
    println!();

    let hints = [
        ("TestSource (I/O)", ExecutionHints::io_bound()),
        ("NativeDecoder", ExecutionHints::native()),
        ("UntrustedParser", ExecutionHints::untrusted()),
        ("LowLatencyProcessor", ExecutionHints::low_latency()),
        (
            "CpuBoundEncoder",
            ExecutionHints {
                processing: ProcessingHint::CpuBound,
                ..ExecutionHints::default()
            },
        ),
        ("Default (trusted)", ExecutionHints::default()),
    ];

    for (name, hint) in hints {
        println!("   {}:", name);
        println!("     trust_level: {:?}", hint.trust_level);
        println!("     processing: {:?}", hint.processing);
        println!("     latency: {:?}", hint.latency);
        println!("     uses_native_code: {}", hint.uses_native_code);
        println!("     crash_safe: {}", hint.crash_safe);
        println!();
    }

    println!("=== Example Complete ===");
    println!();
    println!("Key takeaways:");
    println!("  - ExecutionHints let elements describe their characteristics");
    println!("  - The Executor automatically chooses the best strategy");
    println!("  - No manual configuration needed for most pipelines");
    println!("  - Use ExecutorConfig::async_only() to disable auto-detection");
    println!("  - Use ExecutorConfig::hybrid() for explicit RT scheduling");

    Ok(())
}
