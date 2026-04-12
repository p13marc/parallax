//! Throughput benchmarks for Parallax pipeline.
//!
//! NOTE: These benchmarks are currently stubs. The original benchmarks used
//! types (HeapSegment, MemoryPool, CpuArena, SharedMemorySegment) that have
//! been replaced by SharedArena. Benchmarks need to be rewritten.

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("placeholder", |b| {
        b.iter(|| {
            // TODO: Rewrite benchmarks using SharedArena API
            std::hint::black_box(42)
        });
    });
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
