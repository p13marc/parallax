//! Throughput benchmarks for Parallax pipelines.

use criterion::{Criterion, criterion_group, criterion_main};

fn placeholder_benchmark(_c: &mut Criterion) {
    // TODO: Add pipeline throughput benchmarks in Phase 4
}

criterion_group!(benches, placeholder_benchmark);
criterion_main!(benches);
