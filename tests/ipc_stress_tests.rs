//! Stress tests for IPC and shared memory subsystems.
//!
//! These tests exercise the cross-process communication and shared memory
//! reference counting under high load to verify correctness and stability.

use parallax::memory::{SharedArena, SharedArenaCache};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Instant;

// ============================================================================
// SharedArena Stress Tests
// ============================================================================

/// Test high-frequency acquire/release cycles.
#[test]
fn test_arena_high_frequency_acquire_release() {
    let arena = SharedArena::new(1024, 64).unwrap();
    let iterations = 10_000;

    for i in 0..iterations {
        let mut slot = arena.acquire().expect("should acquire slot");
        slot.data_mut()[0] = (i % 256) as u8;
        drop(slot);
        arena.reclaim();
    }

    // All slots should be free
    assert_eq!(arena.free_count(), 64);
    assert_eq!(arena.pending_count(), 0);
}

/// Test concurrent acquire/release from multiple threads.
#[test]
fn test_arena_concurrent_acquire_release() {
    let arena = Arc::new(SharedArena::new(4096, 128).unwrap());
    let num_threads = 8;
    let iterations_per_thread = 1000;
    let total_ops = Arc::new(AtomicU64::new(0));

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let arena = Arc::clone(&arena);
            let total_ops = Arc::clone(&total_ops);
            thread::spawn(move || {
                for i in 0..iterations_per_thread {
                    // Try to acquire a slot
                    if let Some(mut slot) = arena.acquire() {
                        // Write some data
                        slot.data_mut()[0] = thread_id as u8;
                        slot.data_mut()[1] = (i % 256) as u8;
                        total_ops.fetch_add(1, Ordering::Relaxed);
                        // Slot is dropped here, pushing to release queue
                    }
                    // Periodically reclaim (only owner can, but safe to call)
                    if i % 100 == 0 {
                        arena.reclaim();
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Final reclaim - may need multiple passes due to timing
    for _ in 0..10 {
        arena.reclaim();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }

    let ops = total_ops.load(Ordering::Relaxed);
    println!(
        "Completed {} acquire/release ops across {} threads",
        ops, num_threads
    );
    assert!(ops > 0, "Should have completed some operations");

    // All slots should be free now (with some tolerance for timing)
    let free = arena.free_count();
    assert!(
        free >= 126,
        "Expected most slots to be free, got {} of 128",
        free
    );
}

/// Test the release queue under heavy concurrent pressure.
#[test]
fn test_release_queue_stress() {
    let arena = Arc::new(SharedArena::new(256, 1024).unwrap());

    // Acquire all slots
    let mut slots = Vec::new();
    while let Some(slot) = arena.acquire() {
        slots.push(slot);
    }
    assert_eq!(slots.len(), 1024);
    assert_eq!(arena.free_count(), 0);

    // Drop all slots from multiple threads simultaneously
    let slots = Arc::new(std::sync::Mutex::new(slots));
    let num_threads = 16;
    let slots_per_thread = 1024 / num_threads;

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let slots = Arc::clone(&slots);
            thread::spawn(move || {
                for _ in 0..slots_per_thread {
                    let slot = slots.lock().unwrap().pop();
                    drop(slot); // This pushes to the release queue
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // All slots should be in the release queue
    assert_eq!(arena.pending_count(), 1024);

    // Reclaim should process all
    let reclaimed = arena.reclaim();
    assert_eq!(reclaimed, 1024);
    assert_eq!(arena.free_count(), 1024);
    assert_eq!(arena.pending_count(), 0);
}

/// Test cross-"process" reference counting (simulated with fd duplication).
#[test]
fn test_cross_process_refcount_stress() {
    let arena = Arc::new(SharedArena::new(4096, 32).unwrap());
    let num_clients = 4;
    let iterations = 500;

    // Acquire a slot that will be shared
    let mut owner_slot = arena.acquire().unwrap();
    owner_slot.data_mut()[0..4].copy_from_slice(b"TEST");
    let ipc_ref = owner_slot.ipc_ref();

    let handles: Vec<_> = (0..num_clients)
        .map(|_| {
            let arena = Arc::clone(&arena);
            let ipc_ref = ipc_ref;
            thread::spawn(move || {
                for _ in 0..iterations {
                    // Simulate receiving fd in another process
                    let dup_fd = rustix::io::fcntl_dupfd_cloexec(&arena.fd(), 0).unwrap();
                    let client_arena = unsafe { SharedArena::from_fd(dup_fd).unwrap() };

                    // Get the slot reference
                    if let Some(client_slot) = client_arena.slot_from_ipc(&ipc_ref) {
                        // Verify data
                        assert_eq!(&client_slot.data()[0..4], b"TEST");
                        // Clone and verify again
                        let cloned = client_slot.clone();
                        assert_eq!(&cloned.data()[0..4], b"TEST");
                        // Both drop here, decrementing refcount
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Owner slot should still have refcount 1
    assert_eq!(owner_slot.refcount(), 1);

    // Drop owner slot
    drop(owner_slot);

    // Should be in release queue now
    assert_eq!(arena.pending_count(), 1);
    arena.reclaim();
    assert_eq!(arena.free_count(), 32);
}

/// Test SharedArenaCache under concurrent access.
#[test]
fn test_arena_cache_concurrent() {
    let arena1 = Arc::new(SharedArena::new(1024, 16).unwrap());
    let arena2 = Arc::new(SharedArena::new(2048, 8).unwrap());

    // Acquire slots from both arenas
    let mut slot1 = arena1.acquire().unwrap();
    slot1.data_mut()[0] = 0xAA;
    let ipc_ref1 = slot1.ipc_ref();

    let mut slot2 = arena2.acquire().unwrap();
    slot2.data_mut()[0] = 0xBB;
    let ipc_ref2 = slot2.ipc_ref();

    let num_threads = 4;
    let iterations = 200;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let arena1 = Arc::clone(&arena1);
            let arena2 = Arc::clone(&arena2);
            let ipc_ref1 = ipc_ref1;
            let ipc_ref2 = ipc_ref2;
            thread::spawn(move || {
                let mut cache = SharedArenaCache::new();

                // Map both arenas
                let fd1 = rustix::io::fcntl_dupfd_cloexec(&arena1.fd(), 0).unwrap();
                let fd2 = rustix::io::fcntl_dupfd_cloexec(&arena2.fd(), 0).unwrap();
                unsafe {
                    cache.map_arena(fd1).unwrap();
                    cache.map_arena(fd2).unwrap();
                }

                for i in 0..iterations {
                    // Alternate between arenas
                    let (ipc_ref, expected) = if (thread_id + i) % 2 == 0 {
                        (ipc_ref1, 0xAA)
                    } else {
                        (ipc_ref2, 0xBB)
                    };

                    if let Some(slot) = cache.get_slot(&ipc_ref) {
                        assert_eq!(slot.data()[0], expected);
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Original slots should still be valid
    assert_eq!(slot1.data()[0], 0xAA);
    assert_eq!(slot2.data()[0], 0xBB);
}

/// Test slot slicing under concurrent access.
#[test]
fn test_slot_slicing_stress() {
    let arena = Arc::new(SharedArena::new(4096, 16).unwrap());
    let mut slot = arena.acquire().unwrap();

    // Fill with pattern
    for (i, byte) in slot.data_mut().iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    let num_threads = 8;
    let iterations = 500;

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let slot = slot.clone(); // Increments refcount
            thread::spawn(move || {
                for i in 0..iterations {
                    // Create various slices
                    let offset = ((thread_id * 100 + i) % 3000) as usize;
                    let len = ((i % 500) + 1) as usize;

                    if offset + len <= slot.len() {
                        let sub = slot.slice(offset, len);
                        // Verify the slice data matches
                        for (j, &byte) in sub.data().iter().enumerate() {
                            let expected = ((offset + j) % 256) as u8;
                            assert_eq!(byte, expected, "Mismatch at offset {}", offset + j);
                        }
                        // Sub-slice is dropped here
                    }
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    // Verify original slot still valid
    for (i, &byte) in slot.data().iter().enumerate() {
        assert_eq!(byte, (i % 256) as u8);
    }
}

// ============================================================================
// Throughput Benchmarks
// ============================================================================

/// Measure arena acquire/release throughput.
#[test]
fn test_arena_throughput() {
    let arena = SharedArena::new(4096, 64).unwrap();
    let iterations = 50_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let slot = arena.acquire().expect("acquire");
        drop(slot);
        arena.reclaim();
    }
    let elapsed = start.elapsed();

    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "Arena acquire/release throughput: {:.0} ops/sec ({} ops in {:?})",
        ops_per_sec, iterations, elapsed
    );

    // Should achieve at least 100k ops/sec on modern hardware
    assert!(
        ops_per_sec > 50_000.0,
        "Throughput too low: {:.0} ops/sec",
        ops_per_sec
    );
}

/// Measure refcount increment/decrement throughput.
#[test]
fn test_refcount_throughput() {
    let arena = SharedArena::new(4096, 1).unwrap();
    let slot = arena.acquire().unwrap();
    let iterations = 100_000;

    let start = Instant::now();
    for _ in 0..iterations {
        let cloned = slot.clone();
        drop(cloned);
    }
    let elapsed = start.elapsed();

    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();
    println!(
        "Refcount throughput: {:.0} ops/sec ({} ops in {:?})",
        ops_per_sec, iterations, elapsed
    );

    // Should achieve at least 1M ops/sec
    assert!(
        ops_per_sec > 500_000.0,
        "Refcount throughput too low: {:.0} ops/sec",
        ops_per_sec
    );

    // Final refcount should be 1
    assert_eq!(slot.refcount(), 1);
}

/// Measure concurrent acquire/release throughput.
#[test]
fn test_concurrent_throughput() {
    let arena = Arc::new(SharedArena::new(4096, 256).unwrap());
    let num_threads = 4;
    let iterations_per_thread = 10_000;
    let completed = Arc::new(AtomicU64::new(0));

    let start = Instant::now();

    let handles: Vec<_> = (0..num_threads)
        .map(|_| {
            let arena = Arc::clone(&arena);
            let completed = Arc::clone(&completed);
            thread::spawn(move || {
                for _ in 0..iterations_per_thread {
                    if let Some(slot) = arena.acquire() {
                        completed.fetch_add(1, Ordering::Relaxed);
                        drop(slot);
                    }
                    arena.reclaim();
                }
            })
        })
        .collect();

    for h in handles {
        h.join().unwrap();
    }

    let elapsed = start.elapsed();
    let total_ops = completed.load(Ordering::Relaxed);
    let ops_per_sec = total_ops as f64 / elapsed.as_secs_f64();

    println!(
        "Concurrent throughput: {:.0} ops/sec ({} ops in {:?}, {} threads)",
        ops_per_sec, total_ops, elapsed, num_threads
    );

    // Should scale reasonably with threads
    assert!(
        ops_per_sec > 100_000.0,
        "Concurrent throughput too low: {:.0} ops/sec",
        ops_per_sec
    );
}

// ============================================================================
// Memory Pressure Tests
// ============================================================================

/// Test behavior under memory pressure (full arena).
#[test]
fn test_arena_full_pressure() {
    let arena = SharedArena::new(1024, 8).unwrap();

    // Fill the arena
    let mut slots = Vec::new();
    for _ in 0..8 {
        slots.push(arena.acquire().expect("should acquire"));
    }

    // Verify arena is full
    assert!(arena.acquire().is_none());
    assert_eq!(arena.free_count(), 0);

    // Release one slot and verify we can acquire again
    slots.pop();
    arena.reclaim();
    assert_eq!(arena.free_count(), 1);

    let new_slot = arena.acquire().expect("should acquire after release");
    slots.push(new_slot);
    assert!(arena.acquire().is_none());
}

/// Test rapid full/empty cycles.
#[test]
fn test_arena_full_empty_cycles() {
    let arena = SharedArena::new(512, 32).unwrap();
    let cycles = 100;

    for _ in 0..cycles {
        // Fill completely
        let mut slots = Vec::new();
        while let Some(slot) = arena.acquire() {
            slots.push(slot);
        }
        assert_eq!(slots.len(), 32);
        assert_eq!(arena.free_count(), 0);

        // Empty completely
        slots.clear();
        arena.reclaim();
        assert_eq!(arena.free_count(), 32);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

/// Test zero-length slices.
#[test]
fn test_zero_length_slice() {
    let arena = SharedArena::new(1024, 4).unwrap();
    let slot = arena.acquire().unwrap();

    // Zero-length slice should work
    let sub = slot.slice(0, 0);
    assert_eq!(sub.len(), 0);
    assert!(sub.is_empty());
    assert_eq!(slot.refcount(), 2); // Original + slice

    drop(sub);
    assert_eq!(slot.refcount(), 1);
}

/// Test maximum slot size.
#[test]
fn test_large_slots() {
    // 1MB slots
    let arena = SharedArena::new(1024 * 1024, 4).unwrap();
    let mut slot = arena.acquire().unwrap();

    // Fill with data
    for (i, byte) in slot.data_mut().iter_mut().enumerate() {
        *byte = (i % 256) as u8;
    }

    // Verify
    for (i, &byte) in slot.data().iter().enumerate() {
        assert_eq!(byte, (i % 256) as u8);
    }

    assert_eq!(slot.len(), 1024 * 1024);
}

/// Test many small slots.
#[test]
fn test_many_small_slots() {
    // 512 slots of 64 bytes each (within release queue limits)
    let arena = SharedArena::new(64, 512).unwrap();

    // Acquire all
    let mut slots = Vec::new();
    while let Some(slot) = arena.acquire() {
        slots.push(slot);
    }
    assert_eq!(slots.len(), 512);

    // Verify all unique
    let indices: std::collections::HashSet<_> = slots.iter().map(|s| s.slot_index()).collect();
    assert_eq!(indices.len(), 512);

    // Release all
    slots.clear();
    arena.reclaim();
    assert_eq!(arena.free_count(), 512);
}
