//! # Process Isolation
//!
//! Run pipeline elements in isolated processes for security.
//! Demonstrates different execution modes.
//!
//! Run: `cargo run --example 12_isolation`

use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::memory::CpuArena;
use parallax::pipeline::Pipeline;

struct CounterSource {
    count: u32,
    max: u32,
}

impl Source for CounterSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }
        self.count += 1;
        let bytes = self.count.to_le_bytes();
        ctx.output()[..4].copy_from_slice(&bytes);
        println!(
            "[Source PID {}] Produced {}",
            std::process::id(),
            self.count
        );
        Ok(ProduceResult::Produced(4))
    }
}

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let value = u32::from_le_bytes(ctx.input()[..4].try_into().unwrap());
        println!("[Sink PID {}] Received {}", std::process::id(), value);
        Ok(())
    }
}

fn build_pipeline() -> Result<Pipeline> {
    let arena = CpuArena::new(64, 8)?;
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source_with_arena("src", CounterSource { count: 0, max: 3 }, arena);
    let sink = pipeline.add_sink("sink", PrintSink);
    pipeline.link(src, sink)?;

    Ok(pipeline)
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Execution Modes ===\n");
    println!("Main process PID: {}\n", std::process::id());

    // Mode 1: In-process (default, fastest)
    println!("--- Mode: InProcess (default) ---");
    println!("All elements run in the same process.\n");
    build_pipeline()?.run().await?;

    println!("\n--- Isolation Summary ---");
    println!("run()           - All in one process (fastest, default)");
    println!("run_isolated()  - Each element in separate process (most secure)");
    println!("run_isolating() - Selective isolation by pattern (balanced)");
    println!("\nExample of isolated execution:");
    println!("  pipeline.run_isolated().await?");
    println!("  pipeline.run_isolating(vec![\"*decoder*\"]).await?");

    Ok(())
}
