//! # Hybrid Pipeline (Async I/O + RT Processing)
//!
//! Demonstrates a pipeline where:
//! - `FileSrc`-style source runs as async Tokio task (I/O-bound)
//! - `Gain` element runs in a dedicated RT thread (CPU-bound, RT-safe)
//! - `FileSink`-style sink runs as async Tokio task (I/O-bound)
//!
//! The executor automatically detects that `Gain` is RT-safe and
//! schedules it in the RT data thread, while I/O elements stay in Tokio.
//!
//! ```text
//! [AsyncSource] ──bridge──> [Gain (RT thread)] ──bridge──> [AsyncSink]
//!   (Tokio)                   (SCHED_FIFO)                   (Tokio)
//! ```
//!
//! Run: `cargo run --example 50_hybrid_pipeline`

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    Affinity, ConsumeContext, Element, ExecutionHints, LatencyHint, ProcessingHint, ProduceContext,
    ProduceResult, Sink, Source,
};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline, UnifiedExecutorConfig as ExecutorConfig};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

fn shared_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(1024, 128).unwrap())
}

// ---------------------------------------------------------------------------
// Async source: simulates I/O-bound file reading
// ---------------------------------------------------------------------------

struct AudioSource {
    count: u64,
    max: u64,
}

impl Source for AudioSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }

        // Generate f32 "audio" samples (a simple ramp)
        let arena = shared_arena();
        let slot = arena.acquire().unwrap();
        let num_samples = 256; // 256 f32 samples per buffer
        let handle = MemoryHandle::with_len(slot, num_samples * 4);
        let mut buffer = Buffer::new(handle, Metadata::from_sequence(self.count));

        let data = buffer.as_bytes_mut();
        for i in 0..num_samples {
            let sample = (i as f32) / (num_samples as f32); // 0.0 to ~1.0
            let offset = i * 4;
            data[offset..offset + 4].copy_from_slice(&sample.to_le_bytes());
        }

        self.count += 1;
        Ok(ProduceResult::OwnBuffer(buffer))
    }

    fn name(&self) -> &str {
        "audio_source"
    }

    // I/O-bound: stays in Tokio
    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

// ---------------------------------------------------------------------------
// RT-safe gain element
// ---------------------------------------------------------------------------

struct RtGain {
    factor: f32,
}

impl Element for RtGain {
    fn process(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
        let data = buffer.as_bytes_mut();
        for sample in data.chunks_exact_mut(4) {
            let val = f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
            let out = val * self.factor;
            sample.copy_from_slice(&out.to_le_bytes());
        }
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        "rt_gain"
    }

    fn is_rt_safe(&self) -> bool {
        true // No allocations, no I/O, no locks
    }

    fn affinity(&self) -> Affinity {
        Affinity::RealTime // Request RT scheduling
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints {
            processing: ProcessingHint::CpuBound,
            latency: LatencyHint::Low,
            ..ExecutionHints::trusted()
        }
    }
}

// ---------------------------------------------------------------------------
// Async sink: simulates I/O-bound file writing
// ---------------------------------------------------------------------------

struct VerifySink {
    received: Arc<AtomicU64>,
    expected_factor: f32,
}

impl Sink for VerifySink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let count = self.received.fetch_add(1, Ordering::Relaxed);

        // Verify a few samples were actually gained
        let data = ctx.input();
        if data.len() >= 4 {
            let first_sample = f32::from_le_bytes([data[0], data[1], data[2], data[3]]);
            // First sample in each buffer is 0.0 * factor = 0.0
            assert!(
                first_sample.abs() < 0.001,
                "buffer {count}: first sample should be ~0.0, got {first_sample}"
            );
        }

        if data.len() >= 8 {
            let second_sample = f32::from_le_bytes([data[4], data[5], data[6], data[7]]);
            // Second sample: (1/256) * factor
            let expected = (1.0 / 256.0) * self.expected_factor;
            assert!(
                (second_sample - expected).abs() < 0.001,
                "buffer {count}: second sample should be ~{expected}, got {second_sample}"
            );
        }

        Ok(())
    }

    fn name(&self) -> &str {
        "verify_sink"
    }

    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for visibility into scheduling decisions
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("parallax=info".parse().unwrap()),
        )
        .init();

    let gain_factor = 2.0;
    let num_buffers = 100;
    let received = Arc::new(AtomicU64::new(0));

    // Build pipeline manually so we can set affinity per element
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source(
        "audio_src",
        AudioSource {
            count: 0,
            max: num_buffers,
        },
    );
    let gain = pipeline.add_filter(
        "rt_gain",
        RtGain {
            factor: gain_factor,
        },
    );
    let sink = pipeline.add_sink(
        "verify_sink",
        VerifySink {
            received: received.clone(),
            expected_factor: gain_factor,
        },
    );

    pipeline.link(src, gain)?;
    pipeline.link(gain, sink)?;

    // Run with hybrid scheduling (auto-detected from element hints)
    let config = ExecutorConfig {
        auto_strategy: true, // Let executor detect RT elements
        ..ExecutorConfig::default()
    };
    let executor = Executor::with_config(config);
    executor.run(&mut pipeline).await?;

    let total = received.load(Ordering::Relaxed);
    println!("Hybrid pipeline complete: {total} buffers processed through RT gain");
    assert_eq!(total, num_buffers);

    Ok(())
}
