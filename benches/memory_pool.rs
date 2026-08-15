//! Benchmarks for the shared-memory allocation path.
//!
//! Everything in parallax that carries payload goes through a `SharedArena`:
//! it is memfd-backed so every buffer is IPC-ready, refcounts live inside the
//! shared mapping, and released slots come back through a lock-free MPSC
//! queue that the owner drains with `reclaim()`. These benchmarks measure that
//! path, because it sits on the hot side of every element.
//!
//! Run with:
//!   cargo bench --bench memory_pool
//!
//! Smoke-run without timing:
//!   cargo bench --bench memory_pool -- --test

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use parallax::buffer::{Buffer, MemoryHandle};
use parallax::memory::{BufferPool, FixedBufferPool, SharedArena};
use parallax::metadata::Metadata;
use std::hint::black_box;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Slot sizes spanning the range real elements ask for: a control message, an
/// audio quantum, a 720p I420 frame, a 1080p I420 frame.
const SLOT_SIZES: &[(usize, &str)] = &[
    (256, "256B"),
    (4096, "4KiB"),
    (1_382_400, "720p_i420"),
    (3_110_400, "1080p_i420"),
];

/// Acquire a slot and drop it, per iteration.
///
/// This is the allocation cost an element pays per buffer. The drop pushes the
/// slot onto the release queue rather than freeing it, so the `reclaim()` is
/// what actually makes it available again — measuring the pair together is the
/// only honest way to report a per-buffer cost.
fn bench_arena_acquire_release(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_acquire_release");

    // No Throughput::Bytes: acquiring a slot never touches its payload, so a
    // bytes/sec figure here would report ~10 TiB/s for a 1080p slot and mean
    // nothing. The interesting result is that the time is flat in slot size.
    group.throughput(Throughput::Elements(1));

    for &(slot_size, name) in SLOT_SIZES {
        let arena = SharedArena::new(slot_size, 64).expect("arena");

        group.bench_function(BenchmarkId::new("acquire_drop_reclaim", name), |b| {
            b.iter(|| {
                let slot = arena.acquire().expect("slot");
                black_box(&slot);
                drop(slot);
                black_box(arena.reclaim());
            });
        });
    }

    group.finish();
}

/// Acquire the whole arena, then reclaim it in one sweep.
///
/// `reclaim()` drains the release queue in O(k) for k pending slots, so the
/// per-slot cost should fall as the batch grows. An element that reclaims once
/// per wakeup rather than once per buffer is relying on exactly that.
fn bench_arena_batch_reclaim(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_batch_reclaim");

    for &slot_count in &[8usize, 64, 512] {
        let arena = SharedArena::new(4096, slot_count).expect("arena");
        group.throughput(Throughput::Elements(slot_count as u64));

        group.bench_function(BenchmarkId::new("slots", slot_count), |b| {
            b.iter(|| {
                let slots: Vec<_> = (0..slot_count)
                    .map(|_| arena.acquire().expect("slot"))
                    .collect();
                drop(slots);
                black_box(arena.reclaim());
            });
        });
    }

    group.finish();
}

/// What one fan-out branch actually costs.
///
/// The payload is genuinely shared: no bytes move, and the cost is flat in
/// slot size — a 1080p frame clones as fast as a 256-byte control message.
///
/// This benchmark is the reason that is true. It originally reported hundreds
/// of nanoseconds that grew with the arena, because `SharedSlotRef` holds a
/// `SharedArena` by value and `SharedArena::clone` dup'd the arena fd: `strace
/// -c` over 1000 clones showed ~2000 `fcntl` and ~1000 `close` calls, i.e. two
/// syscalls per branch on the hot path of the feature whose premise is that
/// sharing is free. The fd is now shared via `Arc` and the syscalls are gone.
/// Keep an eye on these numbers: a regression here means something started
/// duplicating an owned resource per clone again.
///
/// `slot_ref_only` vs `buffer_empty_meta` separates the memory handle from the
/// `Metadata` clone — `Metadata` derives `Clone` and carries a
/// `HashMap<&'static str, MetaBox>`, so a branch copies the metadata too.
/// `buffer_rich_meta` prices a populated custom map.
///
/// Deliberately no `Throughput::Bytes`: no payload bytes move, so a bytes/sec
/// figure would be a fiction.
fn bench_buffer_refcount(c: &mut Criterion) {
    let mut group = c.benchmark_group("buffer_clone");
    group.throughput(Throughput::Elements(1));

    for &(slot_size, name) in SLOT_SIZES {
        let arena = SharedArena::new(slot_size, 8).expect("arena");

        // The refcount on its own.
        let slot = arena.acquire().expect("slot");
        group.bench_function(BenchmarkId::new("slot_ref_only", name), |b| {
            b.iter(|| {
                let c = slot.clone();
                black_box(&c);
            });
        });

        // A whole buffer with empty metadata: refcount + an empty-map clone.
        let buffer = Buffer::new(
            MemoryHandle::with_len(arena.acquire().expect("slot"), slot_size),
            Metadata::new(),
        );
        group.bench_function(BenchmarkId::new("buffer_empty_meta", name), |b| {
            b.iter(|| {
                let c = buffer.clone();
                black_box(&c);
            });
        });

        // Fan-out shape: one clone per branch, all alive at once.
        group.bench_function(BenchmarkId::new("buffer_x8_branches", name), |b| {
            b.iter(|| {
                let clones: [Buffer; 8] = std::array::from_fn(|_| buffer.clone());
                black_box(&clones);
            });
        });
    }

    // Metadata weight, at one slot size: a buffer carrying custom entries
    // pays for copying them on every branch.
    let arena = SharedArena::new(4096, 8).expect("arena");
    let mut rich = Metadata::new();
    rich.set("app/frame_number", 1234u32);
    rich.set_bytes("stanag/klv", vec![0u8; 64]);
    let heavy: Buffer = Buffer::new(
        MemoryHandle::with_len(arena.acquire().expect("slot"), 4096),
        rich,
    );
    group.bench_function("buffer_rich_meta/4KiB", |b| {
        b.iter(|| {
            let c = heavy.clone();
            black_box(&c);
        });
    });

    group.finish();
}

/// `FixedBufferPool::try_acquire` (non-blocking) against `acquire` (condvar).
///
/// `try_acquire` calls `reclaim()` first, so on an uncontended pool the two
/// should be nearly identical — `acquire`'s cost only appears when the pool is
/// actually exhausted, which is the back-pressure path.
fn bench_pool_acquire(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_acquire");
    let buffer_size = 4096;
    group.throughput(Throughput::Bytes(buffer_size as u64));

    let pool = FixedBufferPool::new(buffer_size, 64).expect("pool");

    group.bench_function("try_acquire_uncontended", |b| {
        b.iter(|| {
            let buf = pool.try_acquire().expect("pool not exhausted");
            black_box(&buf);
        });
    });

    group.bench_function("acquire_uncontended", |b| {
        b.iter(|| {
            let buf = pool.acquire().expect("pool not exhausted");
            black_box(&buf);
        });
    });

    // Exhausted pool: try_acquire must report failure rather than block. This
    // is what a `LinkPolicy::DropNewest` branch does when it falls behind.
    let small = FixedBufferPool::new(buffer_size, 2).expect("pool");
    let _held: Vec<_> = (0..2).map(|_| small.acquire().expect("held")).collect();
    group.bench_function("try_acquire_exhausted", |b| {
        b.iter(|| {
            black_box(small.try_acquire().is_none());
        });
    });

    group.finish();
}

/// The pool under N threads competing for the same slots.
///
/// Each iteration is one acquire/drop on the main thread while `threads`
/// background threads hammer the same pool, so the number reported is the
/// cost *under* contention, not the contention itself.
fn bench_pool_contention(c: &mut Criterion) {
    let mut group = c.benchmark_group("pool_contention");
    // Sample less: each iteration touches a lock-free queue shared with live
    // threads, so criterion's default 100 samples takes a while.
    group.sample_size(30);

    for &threads in &[1usize, 2, 4] {
        let pool = FixedBufferPool::new(4096, 256).expect("pool");
        let stop = Arc::new(AtomicBool::new(false));

        let workers: Vec<_> = (0..threads)
            .map(|_| {
                let pool = Arc::clone(&pool);
                let stop = Arc::clone(&stop);
                std::thread::spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        if let Some(buf) = pool.try_acquire() {
                            black_box(&buf);
                        }
                    }
                })
            })
            .collect();

        group.bench_function(BenchmarkId::new("competing_threads", threads), |b| {
            b.iter(|| {
                let buf = pool.acquire().expect("pool");
                black_box(&buf);
            });
        });

        stop.store(true, Ordering::Relaxed);
        for w in workers {
            w.join().expect("worker");
        }
    }

    group.finish();
}

/// Mapping an arena from a received fd — the cost a subscriber pays once per
/// arena, not per buffer.
///
/// Worth watching because `SharedArenaCache` exists specifically to make this
/// a once-per-arena cost; if it ever crept onto the per-buffer path the
/// difference would be visible here.
fn bench_arena_from_fd(c: &mut Criterion) {
    let mut group = c.benchmark_group("arena_from_fd");

    for &slot_count in &[16usize, 256] {
        let arena = SharedArena::new(4096, slot_count).expect("arena");

        group.bench_function(BenchmarkId::new("map", slot_count), |b| {
            b.iter(|| {
                let fd = rustix::io::fcntl_dupfd_cloexec(arena.fd(), 0).expect("dup");
                // SAFETY: the fd is a dup of a live SharedArena's own memfd.
                let client = unsafe { SharedArena::from_fd(fd).expect("map") };
                black_box(client.slot_count());
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_arena_acquire_release,
    bench_arena_batch_reclaim,
    bench_buffer_refcount,
    bench_pool_acquire,
    bench_pool_contention,
    bench_arena_from_fd,
);
criterion_main!(benches);
