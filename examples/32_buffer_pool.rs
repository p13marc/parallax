//! Buffer Pool Example - Demonstrates pipeline-level buffer pooling.
//!
//! This example shows how to use the BufferPool API for efficient buffer
//! management with backpressure, statistics tracking, and zero-allocation
//! production.
//!
//! Key concepts demonstrated:
//! - Creating a fixed-size buffer pool
//! - Using pool from ProduceContext via acquire_buffer()
//! - Monitoring pool statistics
//!
//! Run with: cargo run --example 32_buffer_pool

use parallax::element::{
    ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
    SourceAdapter,
};
use parallax::error::Result;
use parallax::memory::{BufferPool, FixedBufferPool};
use parallax::pipeline::Pipeline;

/// A source that uses the buffer pool for zero-allocation production.
///
/// This source demonstrates acquiring buffers from the pool.
struct PoolAwareSource {
    count: u64,
    max_buffers: u64,
}

impl Source for PoolAwareSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max_buffers {
            println!("[Source] Reached max buffers, sending EOS");
            return Ok(ProduceResult::Eos);
        }

        // Acquire a buffer from the pool
        println!("[Source] Acquiring buffer #{} from pool...", self.count);
        let mut pooled = ctx.acquire_buffer()?;
        println!("[Source] Got buffer with capacity {}", pooled.capacity());

        // Write data to the pooled buffer
        let data = format!("Buffer #{} from pool", self.count);
        let len = data.len().min(pooled.capacity());
        pooled.data_mut()[..len].copy_from_slice(data.as_bytes());
        pooled.set_len(len);
        pooled.metadata_mut().sequence = self.count;

        self.count += 1;

        // Convert to Buffer for downstream (detaches from pool)
        Ok(ProduceResult::OwnBuffer(pooled.into_buffer()))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(256) // Small buffers for this example
    }
}

/// A simple sink that prints received data.
struct PrintSink {
    received: u64,
}

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let text = std::str::from_utf8(ctx.input()).unwrap_or("<invalid>");
        println!(
            "[Sink] Received #{}: {} ({} bytes)",
            ctx.sequence(),
            text,
            ctx.len()
        );
        self.received += 1;
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Buffer Pool Example ===\n");

    // Create a buffer pool: 8 buffers of 256 bytes each
    let pool = FixedBufferPool::new(256, 8)?;
    println!(
        "Created pool: {} buffers x {} bytes",
        pool.capacity(),
        pool.buffer_size()
    );
    println!("Initial stats: {:?}\n", pool.stats());

    // Clone pool for stats monitoring
    let pool_for_stats = pool.clone();

    // Create source with pool access
    let source = PoolAwareSource {
        count: 0,
        max_buffers: 5,
    };

    // Create sink
    let sink = PrintSink { received: 0 };

    // Create pipeline
    let mut pipeline = Pipeline::new();

    // Add source with pool attached
    let src_id = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::with_pool(source, pool)),
    );

    let sink_id = pipeline.add_node("sink", DynAsyncElement::new_box(SinkAdapter::new(sink)));

    pipeline.link(src_id, sink_id)?;

    // Run the pipeline
    println!("Starting pipeline...\n");
    pipeline.run().await?;

    // Print final stats
    println!("\n=== Final Pool Statistics ===");
    let stats = pool_for_stats.stats();
    println!("Total acquisitions: {}", stats.acquisitions);
    println!("Waits (backpressure events): {}", stats.waits);
    println!("Available buffers: {}/{}", stats.available, stats.capacity);

    Ok(())
}
