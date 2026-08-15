//! End-to-end pipeline throughput and the primitives underneath it.
//!
//! Three layers, from the top down:
//!
//! 1. A whole pipeline, run to EOS, in async and hybrid scheduling — what a
//!    user actually gets.
//! 2. Fan-out, at 1/2/4/8 branches and under both `LinkPolicy` values, because
//!    "buffers are shared by refcount, not copied" is a claim that should be
//!    visible in the numbers.
//! 3. `AsyncRtBridge` and its `EventFd`, the async<->RT boundary that hybrid
//!    mode pays for at every partition edge.
//!
//! Run with:
//!   cargo bench --bench throughput
//!
//! Smoke-run without timing:
//!   cargo bench --bench throughput -- --test
//!
//! Note there is no `Tee`: fan-out is several links leaving one src-pad, which
//! is why the fan-out benchmarks build graphs rather than elements.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::Element;
use parallax::elements::{NullSink, NullSource, PassThrough};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::rt_bridge::{AsyncRtBridge, BridgeConfig, EventFd};
use parallax::pipeline::rt_scheduler::{RtConfig, SchedulingMode};
use parallax::pipeline::{Executor, LinkPolicy, Pipeline, UnifiedExecutorConfig as ExecutorConfig};
use std::hint::black_box;

/// How many buffers each end-to-end run pushes. Large enough that per-run
/// setup (arena creation, task spawning) does not dominate.
const BUFFERS: u64 = 2_000;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()
        .expect("tokio runtime")
}

/// A linear pipeline of `filters` passthrough stages, run to EOS.
///
/// This is the number to quote for "buffers/sec through a pipeline": it
/// includes channel sends, task wakeups, and the executor's per-buffer
/// bookkeeping (probes and tracers are registered but empty).
fn bench_pipeline_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_linear");
    group.sample_size(20);
    group.throughput(Throughput::Elements(BUFFERS));
    let rt = runtime();

    for &filters in &[0usize, 1, 4] {
        group.bench_function(BenchmarkId::new("passthrough_stages", filters), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut pipeline = Pipeline::new();
                    let src = pipeline.add_source("src", NullSource::new(BUFFERS));
                    let mut prev = src;
                    for i in 0..filters {
                        let f = pipeline.add_filter(format!("f{i}"), PassThrough::new());
                        pipeline.link(prev, f).expect("link");
                        prev = f;
                    }
                    let sink = pipeline.add_sink("sink", NullSink::new());
                    pipeline.link(prev, sink).expect("link");

                    Executor::new().run(&mut pipeline).await.expect("run");
                });
            });
        });
    }

    group.finish();
}

/// The same pipeline under `SchedulingMode::Hybrid`.
///
/// `PassThrough` reports `ExecutionHints::rt_safe()`, so it lands on an RT
/// thread and both edges get an `AsyncRtBridge`. The delta against
/// `pipeline_linear` is what the bridges cost.
fn bench_pipeline_hybrid(c: &mut Criterion) {
    let mut group = c.benchmark_group("pipeline_hybrid");
    group.sample_size(20);
    group.throughput(Throughput::Elements(BUFFERS));
    let rt = runtime();

    for (label, mode) in [
        ("async", SchedulingMode::Async),
        ("hybrid", SchedulingMode::Hybrid),
    ] {
        group.bench_function(BenchmarkId::new("scheduling", label), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut pipeline = Pipeline::new();
                    let src = pipeline.add_source("src", NullSource::new(BUFFERS));
                    let filter = pipeline.add_filter("rt", PassThrough::new());
                    let sink = pipeline.add_sink("sink", NullSink::new());
                    pipeline.link(src, filter).expect("link");
                    pipeline.link(filter, sink).expect("link");

                    // `scheduling` is authoritative; the executor derives
                    // RtConfig::mode from it.
                    let config = ExecutorConfig::default().with_scheduling(mode);
                    Executor::with_config(config)
                        .run(&mut pipeline)
                        .await
                        .expect("run");
                });
            });
        });
    }

    // Keep the import meaningful even if the hints ever change out from under
    // this benchmark: if PassThrough stops being rt_safe, "hybrid" silently
    // becomes a second "async" run and the comparison is worthless.
    assert!(
        PassThrough::new().execution_hints().is_rt_safe(),
        "pipeline_hybrid is only meaningful while PassThrough is rt_safe"
    );

    group.finish();
}

/// Fan-out to N sinks from one src-pad.
///
/// Buffers are shared by refcount, and the broadcast is concurrent
/// (`join_all`), so the per-buffer cost should grow far more slowly than N.
fn bench_fanout_width(c: &mut Criterion) {
    let mut group = c.benchmark_group("fanout_width");
    group.sample_size(20);
    group.throughput(Throughput::Elements(BUFFERS));
    let rt = runtime();

    for &branches in &[1usize, 2, 4, 8] {
        group.bench_function(BenchmarkId::new("branches", branches), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut pipeline = Pipeline::new();
                    let src = pipeline.add_source("src", NullSource::new(BUFFERS));
                    for i in 0..branches {
                        let sink = pipeline.add_sink(format!("sink{i}"), NullSink::new());
                        pipeline.link(src, sink).expect("link");
                    }
                    Executor::new().run(&mut pipeline).await.expect("run");
                });
            });
        });
    }

    group.finish();
}

/// `LinkPolicy::Block` against the two lossy policies on an asymmetric fan-out.
///
/// One branch is deliberately slower (four passthrough stages deep). Under
/// `Block` the source is throttled to the slow branch; under the lossy
/// policies the slow branch degrades itself — DropNewest sheds the incoming
/// buffer, DropOldest (#169) evicts the queued one. The gap against Block is
/// the head-of-line blocking that #39 removed.
fn bench_fanout_policy(c: &mut Criterion) {
    let mut group = c.benchmark_group("fanout_policy");
    group.sample_size(20);
    group.throughput(Throughput::Elements(BUFFERS));
    let rt = runtime();

    for policy in [
        LinkPolicy::Block,
        LinkPolicy::DropNewest,
        LinkPolicy::DropOldest,
    ] {
        let label = match policy {
            LinkPolicy::Block => "block",
            LinkPolicy::DropNewest => "drop_newest",
            LinkPolicy::DropOldest => "drop_oldest",
        };
        group.bench_function(BenchmarkId::new("slow_branch", label), |b| {
            b.iter(|| {
                rt.block_on(async {
                    let mut pipeline = Pipeline::new();
                    let src = pipeline.add_source("src", NullSource::new(BUFFERS));

                    // Fast branch: straight to a sink.
                    let fast = pipeline.add_sink("fast", NullSink::new());
                    pipeline.link(src, fast).expect("link");

                    // Slow branch: a chain of stages, so it drains later.
                    let head = pipeline.add_filter("slow0", PassThrough::new());
                    pipeline.link_with(src, head, policy).expect("link");
                    let mut prev = head;
                    for i in 1..4 {
                        let f = pipeline.add_filter(format!("slow{i}"), PassThrough::new());
                        pipeline.link(prev, f).expect("link");
                        prev = f;
                    }
                    let slow_sink = pipeline.add_sink("slow_sink", NullSink::new());
                    pipeline.link(prev, slow_sink).expect("link");

                    Executor::new().run(&mut pipeline).await.expect("run");
                });
            });
        });
    }

    group.finish();
}

/// The lock-free SPSC ring underneath every async<->RT boundary.
///
/// No eventfd here: this is the ring alone, which is what an RT thread touches
/// on its hot path.
fn bench_rt_bridge_ring(c: &mut Criterion) {
    let mut group = c.benchmark_group("rt_bridge_ring");

    for &capacity in &[16usize, 256] {
        let bridge = AsyncRtBridge::new(BridgeConfig::with_capacity(capacity)).expect("bridge");
        let arena = SharedArena::new(4096, capacity * 2).expect("arena");

        group.throughput(Throughput::Elements(1));
        group.bench_function(BenchmarkId::new("push_pop", capacity), |b| {
            b.iter(|| {
                let slot = arena.acquire().expect("slot");
                let buffer = Buffer::new(MemoryHandle::with_len(slot, 4096), Metadata::new());
                bridge.try_push(buffer).expect("ring not full");
                black_box(bridge.try_pop().expect("ring not empty"));
                arena.reclaim();
            });
        });

        // Fill the ring, then drain it: amortises the per-slot cost the way a
        // driver-paced RT cycle does.
        group.bench_function(BenchmarkId::new("fill_drain", capacity), |b| {
            b.iter(|| {
                let mut pushed = 0;
                while pushed < capacity {
                    let slot = match arena.acquire() {
                        Some(s) => s,
                        None => break,
                    };
                    let buffer = Buffer::new(MemoryHandle::with_len(slot, 4096), Metadata::new());
                    if bridge.try_push(buffer).is_err() {
                        break;
                    }
                    pushed += 1;
                }
                while let Some(buf) = bridge.try_pop() {
                    black_box(&buf);
                }
                arena.reclaim();
            });
        });
    }

    group.finish();
}

/// `EventFd` notify/wait — the wakeup an RT thread costs an async task.
///
/// Measured without a blocking wait, so this is the syscall pair, not the
/// scheduler latency.
fn bench_eventfd(c: &mut Criterion) {
    let mut group = c.benchmark_group("eventfd");
    let efd = EventFd::new().expect("eventfd");

    group.bench_function("notify_then_try_wait", |b| {
        b.iter(|| {
            efd.notify().expect("notify");
            black_box(efd.try_wait().expect("try_wait"));
        });
    });

    group.bench_function("try_wait_empty", |b| {
        b.iter(|| {
            black_box(efd.try_wait().expect("try_wait"));
        });
    });

    group.finish();
}

/// Confirm the RT config presets still name what these benchmarks assume.
fn bench_noop_config_sanity(_c: &mut Criterion) {
    assert_eq!(RtConfig::realtime().mode, SchedulingMode::RealTime);
    assert_eq!(RtConfig::hybrid().mode, SchedulingMode::Hybrid);
}

criterion_group!(
    benches,
    bench_pipeline_linear,
    bench_pipeline_hybrid,
    bench_fanout_width,
    bench_fanout_policy,
    bench_rt_bridge_ring,
    bench_eventfd,
    bench_noop_config_sanity,
);
criterion_main!(benches);
