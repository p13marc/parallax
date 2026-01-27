//! Benchmarks comparing InProcess vs Isolated execution modes.
//!
//! This benchmark measures the overhead of process isolation, including:
//! - Pipeline startup latency
//! - Per-buffer IPC overhead
//! - End-to-end throughput

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parallax::buffer::Buffer;
use parallax::element::{
    ConsumeContext, DynAsyncElement, Element, ElementAdapter, ProduceContext, ProduceResult, Sink,
    SinkAdapter, Source, SourceAdapter,
};
use parallax::error::Result;
use parallax::memory::CpuArena;
use parallax::pipeline::Pipeline;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Source that produces N buffers.
/// Uses OwnBuffer pattern to work with or without pre-allocated arena.
struct CountingSource {
    remaining: u64,
    total: u64,
}

impl CountingSource {
    fn new(count: u64) -> Self {
        Self {
            remaining: count,
            total: count,
        }
    }
}

impl Source for CountingSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        use parallax::buffer::MemoryHandle;
        use parallax::memory::HeapSegment;
        use parallax::metadata::Metadata;

        if self.remaining == 0 {
            return Ok(ProduceResult::Eos);
        }

        self.remaining -= 1;
        let seq = self.total - self.remaining - 1;

        if ctx.has_buffer() {
            // Use pre-allocated buffer from arena
            let output = ctx.output();
            let data = seq.to_le_bytes();
            let len = data.len().min(output.len());
            output[..len].copy_from_slice(&data[..len]);
            ctx.set_sequence(seq);
            Ok(ProduceResult::Produced(len))
        } else {
            // Fallback: create own buffer
            let segment = Arc::new(HeapSegment::new(64).unwrap());
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(seq));
            Ok(ProduceResult::OwnBuffer(buffer))
        }
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(64)
    }
}

/// Passthrough element (simulates processing).
struct Passthrough;

impl Element for Passthrough {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        Ok(Some(buffer))
    }
}

/// Counting sink.
struct CountingSink {
    count: Arc<AtomicU64>,
}

impl CountingSink {
    fn new(counter: Arc<AtomicU64>) -> Self {
        Self { count: counter }
    }
}

impl Sink for CountingSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Build a test pipeline: source -> passthrough -> sink
fn build_pipeline(buffer_count: u64, counter: Arc<AtomicU64>) -> Pipeline {
    let mut pipeline = Pipeline::new();

    // Use SourceAdapter without arena - source will use OwnBuffer fallback
    let src = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(CountingSource::new(buffer_count))),
    );

    let pass = pipeline.add_node(
        "passthrough",
        DynAsyncElement::new_box(ElementAdapter::new(Passthrough)),
    );

    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink::new(counter))),
    );

    pipeline.link(src, pass).expect("link src->pass");
    pipeline.link(pass, sink).expect("link pass->sink");

    pipeline
}

/// Benchmark InProcess execution (baseline).
fn bench_inprocess(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("execution_inprocess");

    for buffer_count in [100u64, 1000, 10000] {
        group.throughput(Throughput::Elements(buffer_count));
        group.bench_with_input(
            BenchmarkId::new("buffers", buffer_count),
            &buffer_count,
            |b, &count| {
                b.iter(|| {
                    let counter = Arc::new(AtomicU64::new(0));
                    let mut pipeline = build_pipeline(count, counter.clone());

                    rt.block_on(async {
                        pipeline.run().await.expect("pipeline run");
                    });

                    black_box(counter.load(Ordering::Relaxed))
                });
            },
        );
    }

    group.finish();
}

/// Benchmark per-buffer latency in InProcess mode.
fn bench_inprocess_latency(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("latency_inprocess");

    // Use a small buffer count to measure per-iteration latency
    let buffer_count = 10u64;

    group.throughput(Throughput::Elements(buffer_count));
    group.bench_function("10_buffers", |b| {
        b.iter(|| {
            let counter = Arc::new(AtomicU64::new(0));
            let mut pipeline = build_pipeline(buffer_count, counter.clone());

            rt.block_on(async {
                pipeline.run().await.expect("pipeline run");
            });

            black_box(counter.load(Ordering::Relaxed))
        });
    });

    group.finish();
}

/// Benchmark comparing raw element processing (no pipeline overhead).
fn bench_raw_element(c: &mut Criterion) {
    use parallax::buffer::MemoryHandle;
    use parallax::memory::HeapSegment;
    use parallax::metadata::Metadata;

    let mut group = c.benchmark_group("raw_element");

    for buffer_count in [100usize, 1000, 10000] {
        group.throughput(Throughput::Elements(buffer_count as u64));
        group.bench_with_input(
            BenchmarkId::new("buffers", buffer_count),
            &buffer_count,
            |b, &count| {
                b.iter(|| {
                    let mut passthrough = Passthrough;
                    let mut processed = 0usize;

                    for i in 0..count {
                        let segment = Arc::new(HeapSegment::new(64).unwrap());
                        let handle = MemoryHandle::from_segment(segment);
                        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(i as u64));

                        if let Ok(Some(_)) = passthrough.process(buffer) {
                            processed += 1;
                        }
                    }

                    black_box(processed)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark SharedArena acquire/release (IPC-ready memory).
fn bench_shared_arena(c: &mut Criterion) {
    use parallax::memory::SharedArena;

    let mut group = c.benchmark_group("shared_arena");

    // Use large arena to avoid running out of slots during benchmark iterations
    let slot_count = 1024;
    let arena = SharedArena::new(1024, slot_count).expect("create arena");

    group.throughput(Throughput::Elements(1));
    group.bench_function("acquire_release_reclaim", |b| {
        b.iter(|| {
            let slot = arena.acquire().expect("acquire slot");
            black_box(&slot);
            drop(slot); // Slot pushed to release queue
            arena.reclaim(); // Reclaim immediately
        });
    });

    // Also benchmark batch acquire/release
    group.throughput(Throughput::Elements(10));
    group.bench_function("batch_10", |b| {
        b.iter(|| {
            let slots: Vec<_> = (0..10).map(|_| arena.acquire().expect("acquire")).collect();
            black_box(&slots);
            drop(slots); // All slots pushed to release queue
            arena.reclaim(); // Reclaim immediately
        });
    });

    group.finish();
}

/// Benchmark CpuArena vs SharedArena comparison.
fn bench_arena_comparison(c: &mut Criterion) {
    use parallax::memory::SharedArena;

    let mut group = c.benchmark_group("arena_comparison");

    let slot_size = 1024;
    let slot_count = 1024; // Large enough for benchmark iterations

    // CpuArena (single-process, simpler)
    let cpu_arena = CpuArena::new(slot_size, slot_count).expect("create cpu arena");
    group.throughput(Throughput::Elements(1));
    group.bench_function("cpu_arena", |b| {
        b.iter(|| {
            let slot = cpu_arena.acquire().expect("acquire");
            black_box(slot);
        });
    });

    // SharedArena (cross-process capable, refcounts in shared memory)
    let shared_arena = SharedArena::new(slot_size, slot_count).expect("create shared arena");
    group.bench_function("shared_arena", |b| {
        b.iter(|| {
            let slot = shared_arena.acquire().expect("acquire");
            black_box(&slot);
            drop(slot);
            shared_arena.reclaim();
        });
    });

    group.finish();
}

/// Benchmark pipeline startup time.
fn bench_pipeline_startup(c: &mut Criterion) {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let mut group = c.benchmark_group("pipeline_startup");

    // Measure just pipeline creation (no execution)
    group.bench_function("create_3_element", |b| {
        b.iter(|| {
            let counter = Arc::new(AtomicU64::new(0));
            let pipeline = build_pipeline(1, counter);
            black_box(pipeline)
        });
    });

    // Measure pipeline creation + single buffer run
    group.bench_function("create_and_run_1_buffer", |b| {
        b.iter(|| {
            let counter = Arc::new(AtomicU64::new(0));
            let mut pipeline = build_pipeline(1, counter.clone());

            rt.block_on(async {
                pipeline.run().await.expect("run");
            });

            black_box(counter.load(Ordering::Relaxed))
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_inprocess,
    bench_inprocess_latency,
    bench_raw_element,
    bench_shared_arena,
    bench_arena_comparison,
    bench_pipeline_startup,
);

criterion_main!(benches);
