//! Memory pool benchmarks.
//!
//! NOTE: These benchmarks are currently stubs. The original benchmarks used
//! types (HeapSegment, MemoryPool) that have been replaced by SharedArena
//! and FixedBufferPool. Benchmarks need to be rewritten.

use criterion::{Criterion, criterion_group, criterion_main};

fn bench_placeholder(c: &mut Criterion) {
    c.bench_function("pool_placeholder", |b| {
        b.iter(|| {
            // TODO: Rewrite benchmarks using SharedArena/FixedBufferPool API
            std::hint::black_box(42)
        });
    });
}

criterion_group!(benches, bench_placeholder);
criterion_main!(benches);
