//! # Buffer Pool
//!
//! Pre-allocated buffer pool for zero-allocation streaming.
//! Demonstrates backpressure when pool is exhausted.
//!
//! ```text
//! [PooledSource] → [Sink]
//! ```
//!
//! Run: `cargo run --example 11_buffer_pool`

use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::memory::{BufferPool, FixedBufferPool};
use parallax::pipeline::Pipeline;

struct PooledSource {
    count: u64,
    max: u64,
}

impl Source for PooledSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }
        self.count += 1;

        // Acquire buffer from pool (blocks if pool exhausted)
        let mut pooled = ctx.acquire_buffer()?;

        let msg = format!("Pooled buffer #{}", self.count);
        let bytes = msg.as_bytes();
        pooled.data_mut()[..bytes.len()].copy_from_slice(bytes);
        pooled.set_len(bytes.len());

        println!("[Source] Produced #{}", self.count);
        Ok(ProduceResult::OwnBuffer(pooled.into_buffer()))
    }
}

struct SlowSink;

impl Sink for SlowSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let text = std::str::from_utf8(ctx.input()).unwrap_or("<invalid>");
        println!("[Sink] Processing: {}", text);
        std::thread::sleep(std::time::Duration::from_millis(50));
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create pool: 4 buffers of 256 bytes each
    let pool = FixedBufferPool::new(256, 4)?;
    println!(
        "Created pool: {} buffers x {} bytes\n",
        pool.capacity(),
        pool.buffer_size()
    );

    let pool_stats = pool.clone();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source_with_pool("src", PooledSource { count: 0, max: 8 }, pool);
    let sink = pipeline.add_sink("sink", SlowSink);
    pipeline.link(src, sink)?;

    pipeline.run().await?;

    // Print pool statistics
    let stats = pool_stats.stats();
    println!("\n=== Pool Statistics ===");
    println!("Total acquisitions: {}", stats.acquisitions);
    println!("Waits (backpressure): {}", stats.waits);

    Ok(())
}
