//! Memory pool benchmarks.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parallax::memory::{HeapSegment, MemoryPool};
use std::sync::Arc;

fn bench_pool_acquire_release(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_acquire_release");

    for num_slots in [16, 64, 256, 1024] {
        let segment = HeapSegment::new(num_slots * 1024).unwrap();
        let pool = Arc::new(MemoryPool::new(segment, 1024).unwrap());

        group.throughput(Throughput::Elements(1));
        group.bench_with_input(BenchmarkId::from_parameter(num_slots), &pool, |b, pool| {
            b.iter(|| {
                let slot = pool.loan().expect("pool not exhausted");
                drop(slot);
            });
        });
    }

    group.finish();
}

fn bench_pool_concurrent(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_concurrent");

    let segment = HeapSegment::new(1024 * 1024).unwrap();
    let pool = Arc::new(MemoryPool::new(segment, 1024).unwrap());

    group.throughput(Throughput::Elements(100));
    group.bench_function("4_threads_100_ops_each", |b| {
        b.iter(|| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    let pool = Arc::clone(&pool);
                    std::thread::spawn(move || {
                        for _ in 0..100 {
                            if let Some(slot) = pool.loan() {
                                std::hint::black_box(slot.as_ptr());
                            }
                        }
                    })
                })
                .collect();

            for h in handles {
                h.join().unwrap();
            }
        });
    });

    group.finish();
}

criterion_group!(benches, bench_pool_acquire_release, bench_pool_concurrent);
criterion_main!(benches);
