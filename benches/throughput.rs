//! Throughput benchmarks for Parallax pipeline.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{Element, Sink, Source};
use parallax::error::Result;
use parallax::link::LocalLink;
use parallax::memory::{HeapSegment, MemoryPool, SharedMemorySegment};
use parallax::metadata::Metadata;
use std::hint::black_box;
use std::sync::Arc;

/// A simple passthrough element for benchmarking.
struct BenchPassthrough;

impl Element for BenchPassthrough {
    fn process(&mut self, buffer: Buffer<()>) -> Result<Option<Buffer<()>>> {
        Ok(Some(buffer))
    }
}

/// A source that produces N buffers of a given size.
struct BenchSource {
    remaining: usize,
    buffer_size: usize,
    sequence: u64,
}

impl BenchSource {
    fn new(count: usize, buffer_size: usize) -> Self {
        Self {
            remaining: count,
            buffer_size,
            sequence: 0,
        }
    }
}

impl Source for BenchSource {
    fn produce(&mut self) -> Result<Option<Buffer<()>>> {
        if self.remaining == 0 {
            return Ok(None);
        }

        self.remaining -= 1;
        let segment = Arc::new(HeapSegment::new(self.buffer_size).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::<()>::new(handle, Metadata::from_sequence(self.sequence));
        self.sequence += 1;
        Ok(Some(buffer))
    }
}

/// A sink that just counts buffers.
struct BenchSink {
    count: usize,
    bytes: usize,
}

impl BenchSink {
    fn new() -> Self {
        Self { count: 0, bytes: 0 }
    }
}

impl Sink for BenchSink {
    fn consume(&mut self, buffer: Buffer<()>) -> Result<()> {
        self.count += 1;
        self.bytes += buffer.len();
        Ok(())
    }
}

fn bench_buffer_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_creation");

    for size in [64, 1024, 64 * 1024, 1024 * 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            b.iter(|| {
                let segment = Arc::new(HeapSegment::new(size).unwrap());
                let handle = MemoryHandle::from_segment(segment);
                black_box(Buffer::<()>::new(handle, Metadata::default()))
            });
        });
    }

    group.finish();
}

fn bench_buffer_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_clone");

    for size in [64, 1024, 64 * 1024, 1024 * 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let segment = Arc::new(HeapSegment::new(size).unwrap());
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::<()>::new(handle, Metadata::default());

            b.iter(|| black_box(buffer.clone()));
        });
    }

    group.finish();
}

fn bench_pool_loan_return(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_loan_return");

    for slot_size in [1024, 64 * 1024, 1024 * 1024].iter() {
        group.throughput(Throughput::Elements(1));
        group.bench_with_input(
            BenchmarkId::from_parameter(slot_size),
            slot_size,
            |b, &slot_size| {
                let segment = HeapSegment::new(slot_size * 16).unwrap();
                let pool = MemoryPool::new(segment, slot_size).unwrap();

                b.iter(|| {
                    let slot = pool.loan().unwrap();
                    black_box(slot);
                    // Slot returned on drop
                });
            },
        );
    }

    group.finish();
}

fn bench_channel_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("channel_throughput");

    for buffer_count in [100, 1000, 10000].iter() {
        group.throughput(Throughput::Elements(*buffer_count as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_count),
            buffer_count,
            |b, &buffer_count| {
                b.iter(|| {
                    let (tx, rx) = LocalLink::bounded(64);

                    // Producer
                    std::thread::scope(|s| {
                        let producer = s.spawn(|| {
                            for i in 0..buffer_count {
                                let segment = Arc::new(HeapSegment::new(64).unwrap());
                                let handle = MemoryHandle::from_segment(segment);
                                let buffer =
                                    Buffer::<()>::new(handle, Metadata::from_sequence(i as u64));
                                tx.send(buffer).unwrap();
                            }
                        });

                        let consumer = s.spawn(|| {
                            let mut count = 0;
                            while count < buffer_count {
                                if rx.recv().is_some() {
                                    count += 1;
                                }
                            }
                            count
                        });

                        producer.join().unwrap();
                        black_box(consumer.join().unwrap());
                    });
                });
            },
        );
    }

    group.finish();
}

fn bench_shared_memory_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("shared_memory");

    for size in [4096, 64 * 1024, 1024 * 1024].iter() {
        group.throughput(Throughput::Bytes(*size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), size, |b, &size| {
            let mut counter = 0u64;
            b.iter(|| {
                let name = format!("bench-shm-{}", counter);
                counter += 1;
                black_box(SharedMemorySegment::new(&name, size).unwrap())
            });
        });
    }

    group.finish();
}

fn bench_element_passthrough(c: &mut Criterion) {
    let mut group = c.benchmark_group("element_passthrough");

    for buffer_size in [64, 1024, 64 * 1024].iter() {
        group.throughput(Throughput::Bytes(*buffer_size as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(buffer_size),
            buffer_size,
            |b, &buffer_size| {
                let mut element = BenchPassthrough;
                let segment = Arc::new(HeapSegment::new(buffer_size).unwrap());
                let handle = MemoryHandle::from_segment(segment);
                let buffer = Buffer::<()>::new(handle, Metadata::default());

                b.iter(|| {
                    let input = buffer.clone();
                    black_box(element.process(input).unwrap())
                });
            },
        );
    }

    group.finish();
}

fn bench_source_to_sink(c: &mut Criterion) {
    let mut group = c.benchmark_group("source_to_sink");

    for (count, size) in [(1000, 64), (1000, 1024), (100, 64 * 1024)].iter() {
        let total_bytes = count * size;
        group.throughput(Throughput::Bytes(total_bytes as u64));
        group.bench_with_input(
            BenchmarkId::new("count_size", format!("{}x{}", count, size)),
            &(*count, *size),
            |b, &(count, size)| {
                b.iter(|| {
                    let mut source = BenchSource::new(count, size);
                    let mut sink = BenchSink::new();

                    while let Some(buffer) = source.produce().unwrap() {
                        sink.consume(buffer).unwrap();
                    }

                    black_box(sink.count)
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_buffer_creation,
    bench_buffer_clone,
    bench_pool_loan_return,
    bench_channel_throughput,
    bench_shared_memory_creation,
    bench_element_passthrough,
    bench_source_to_sink,
);

criterion_main!(benches);
