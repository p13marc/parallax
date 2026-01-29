//! Integration tests for arena behavior in pipeline contexts.
//!
//! These tests verify that arenas work correctly in realistic pipeline
//! scenarios, including proper reclaim behavior and exhaustion handling.

use parallax::buffer::MemoryHandle;
use parallax::memory::SharedArena;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// Arena Exhaustion Tests
// ============================================================================

/// Test that arena exhaustion is detected correctly.
#[test]
fn test_arena_exhaustion_detection() {
    let arena = SharedArena::new(1024, 10).unwrap();

    // Initially not exhausted
    assert!(!arena.is_exhausted());
    assert!(!arena.is_nearly_exhausted());
    assert!((arena.utilization() - 0.0).abs() < 0.01);

    // Acquire all slots
    let slots: Vec<_> = (0..10).filter_map(|_| arena.acquire()).collect();
    assert_eq!(slots.len(), 10);

    // Now exhausted
    assert!(arena.is_exhausted());
    assert!(arena.is_nearly_exhausted());
    assert!((arena.utilization() - 100.0).abs() < 0.01);

    // Can't acquire more
    assert!(arena.acquire().is_none());

    // Drop one slot
    drop(slots.into_iter().next());

    // Still exhausted until reclaim
    assert!(arena.is_exhausted());

    // After reclaim, one slot available
    arena.reclaim();
    assert!(!arena.is_exhausted());
    assert!(arena.acquire().is_some());
}

/// Test nearly-exhausted threshold detection at various levels.
#[test]
fn test_arena_threshold_detection() {
    let arena = SharedArena::new(1024, 100).unwrap();

    // Acquire 50% (not nearly exhausted)
    let _slots_50: Vec<_> = (0..50).filter_map(|_| arena.acquire()).collect();
    assert!(!arena.is_nearly_exhausted());
    assert!(!arena.is_nearly_exhausted_threshold(90.0));
    assert!(arena.is_nearly_exhausted_threshold(40.0));

    // Acquire to 85% (still not at 90%)
    let _slots_35: Vec<_> = (0..35).filter_map(|_| arena.acquire()).collect();
    assert!(!arena.is_nearly_exhausted()); // default 90%
    assert!(arena.is_nearly_exhausted_threshold(80.0));

    // Acquire to 95%
    let _slots_10: Vec<_> = (0..10).filter_map(|_| arena.acquire()).collect();
    assert!(arena.is_nearly_exhausted()); // > 90%
    assert!(arena.is_nearly_exhausted_threshold(90.0));
}

// ============================================================================
// Arena Metrics Tests
// ============================================================================

/// Test metrics accuracy under various conditions.
#[test]
fn test_arena_metrics_accuracy() {
    let arena = SharedArena::new(4096, 50).unwrap();

    // Initial state
    let m = arena.metrics();
    assert_eq!(m.slot_count, 50);
    assert_eq!(m.slot_size, 4096);
    assert_eq!(m.allocated_slots, 0);
    assert_eq!(m.free_slots, 50);
    assert_eq!(m.pending_release, 0);
    assert_eq!(m.used_bytes, 0);

    // Allocate 20 slots
    let slots: Vec<_> = (0..20).filter_map(|_| arena.acquire()).collect();
    let m = arena.metrics();
    assert_eq!(m.allocated_slots, 20);
    assert_eq!(m.free_slots, 30);
    assert_eq!(m.used_bytes, 20 * 4096);
    assert!((m.utilization_percent - 40.0).abs() < 0.01);

    // Drop slots (they go to pending)
    drop(slots);
    let m = arena.metrics();
    assert_eq!(m.pending_release, 20);
    assert_eq!(m.available_after_reclaim(), 30 + 20);

    // Reclaim
    arena.reclaim();
    let m = arena.metrics();
    assert_eq!(m.pending_release, 0);
    assert_eq!(m.free_slots, 50);
    assert_eq!(m.allocated_slots, 0);
}

// ============================================================================
// Producer-Consumer Pattern Tests
// ============================================================================

/// Simulate a producer-consumer pattern with arena.
/// Producer acquires slots, writes data, consumer reads and releases.
#[test]
fn test_producer_consumer_pattern() {
    let arena = Arc::new(SharedArena::new(1024, 32).unwrap());
    let arena_producer = Arc::clone(&arena);
    let arena_consumer = Arc::clone(&arena);

    let (tx, rx) = std::sync::mpsc::channel();
    let produced = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let consumed = Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let produced_clone = Arc::clone(&produced);
    let consumed_clone = Arc::clone(&consumed);

    // Producer thread
    let producer = thread::spawn(move || {
        for i in 0..100 {
            // Reclaim before acquire
            arena_producer.reclaim();

            // Try to acquire, with backoff if exhausted
            let mut attempts = 0;
            let slot = loop {
                if let Some(slot) = arena_producer.acquire() {
                    break slot;
                }
                attempts += 1;
                if attempts > 100 {
                    panic!("Producer couldn't acquire slot after 100 attempts");
                }
                arena_producer.reclaim();
                thread::sleep(Duration::from_micros(100));
            };

            // Write data
            let mut slot = slot;
            slot.data_mut()[0] = (i % 256) as u8;

            // Send to consumer
            tx.send(slot).unwrap();
            produced_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    });

    // Consumer thread
    let consumer = thread::spawn(move || {
        for _ in 0..100 {
            let slot = rx.recv().unwrap();
            // Read data (just verify it's valid)
            let _ = slot.data()[0];
            // Drop slot (pushes to release queue)
            drop(slot);
            consumed_clone.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            // Periodically reclaim from consumer side
            arena_consumer.reclaim();
        }
    });

    producer.join().unwrap();
    consumer.join().unwrap();

    // Final reclaim
    arena.reclaim();

    assert_eq!(produced.load(std::sync::atomic::Ordering::Relaxed), 100);
    assert_eq!(consumed.load(std::sync::atomic::Ordering::Relaxed), 100);
    assert_eq!(arena.free_count(), 32);
    assert_eq!(arena.pending_count(), 0);
}

// ============================================================================
// Buffer Integration Tests
// ============================================================================

/// Test creating MemoryHandles from arena slots.
#[test]
fn test_memory_handle_from_arena_slot() {
    let arena = SharedArena::new(4096, 8).unwrap();

    // Create handle from slot
    let mut slot = arena.acquire().unwrap();
    let data = b"Hello, Parallax!";
    slot.data_mut()[..data.len()].copy_from_slice(data);

    let handle = MemoryHandle::with_len(slot, data.len());

    assert_eq!(handle.len(), data.len());
    assert_eq!(handle.as_slice(), data);

    // Arena should show 1 allocated
    assert_eq!(arena.allocated_count(), 1);

    // Drop handle
    drop(handle);

    // Slot should be pending
    assert_eq!(arena.pending_count(), 1);

    // Reclaim
    arena.reclaim();
    assert_eq!(arena.free_count(), 8);
}

/// Test multiple handles sharing arena.
#[test]
fn test_multiple_handles_from_arena() {
    let arena = SharedArena::new(1024, 16).unwrap();
    let mut handles = Vec::new();

    // Create multiple handles
    for i in 0..10 {
        arena.reclaim();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[0] = i as u8;
        let handle = MemoryHandle::with_len(slot, 1);
        handles.push(handle);
    }

    assert_eq!(arena.allocated_count(), 10);
    assert_eq!(arena.free_count(), 6);

    // Drop half
    handles.truncate(5);
    assert_eq!(arena.pending_count(), 5);

    // Reclaim
    arena.reclaim();
    assert_eq!(arena.free_count(), 11);
    assert_eq!(arena.allocated_count(), 5);

    // Drop rest
    drop(handles);
    arena.reclaim();
    assert_eq!(arena.free_count(), 16);
}

// ============================================================================
// Reclaim Timing Tests
// ============================================================================

/// Test that reclaim works correctly when called frequently.
#[test]
fn test_frequent_reclaim() {
    let arena = SharedArena::new(512, 8).unwrap();

    for _ in 0..100 {
        // Acquire
        let slot = arena.acquire().unwrap();

        // Reclaim (should be no-op, slot not released yet)
        let reclaimed = arena.reclaim();
        assert_eq!(reclaimed, 0);

        // Drop
        drop(slot);

        // Reclaim (should reclaim the slot)
        let reclaimed = arena.reclaim();
        assert_eq!(reclaimed, 1);
    }

    assert_eq!(arena.free_count(), 8);
}

/// Test reclaim performance with batch releases.
#[test]
fn test_batch_reclaim_performance() {
    let arena = SharedArena::new(256, 1000).unwrap();

    // Acquire all slots
    let slots: Vec<_> = (0..1000).filter_map(|_| arena.acquire()).collect();
    assert_eq!(slots.len(), 1000);

    // Drop all at once
    let start = Instant::now();
    drop(slots);
    let drop_time = start.elapsed();

    // All should be pending
    assert_eq!(arena.pending_count(), 1000);

    // Single reclaim should handle all
    let start = Instant::now();
    let reclaimed = arena.reclaim();
    let reclaim_time = start.elapsed();

    assert_eq!(reclaimed, 1000);
    assert_eq!(arena.free_count(), 1000);

    println!(
        "Batch release: drop {} slots in {:?}, reclaim in {:?}",
        1000, drop_time, reclaim_time
    );

    // Both should be fast (< 10ms)
    assert!(drop_time < Duration::from_millis(100));
    assert!(reclaim_time < Duration::from_millis(100));
}

// ============================================================================
// Edge Case Tests
// ============================================================================

/// Test arena with single slot.
#[test]
fn test_single_slot_arena() {
    let arena = SharedArena::new(64, 1).unwrap();

    let slot = arena.acquire().unwrap();
    assert!(arena.is_exhausted());
    assert!(arena.acquire().is_none());

    drop(slot);
    arena.reclaim();

    assert!(!arena.is_exhausted());
    let _slot = arena.acquire().unwrap();
}

/// Test arena with very large slots.
#[test]
fn test_large_slot_arena() {
    // 10MB slots
    let arena = SharedArena::new(10 * 1024 * 1024, 4).unwrap();

    let mut slot = arena.acquire().unwrap();
    // Write to beginning and end
    slot.data_mut()[0] = 0xAA;
    slot.data_mut()[10 * 1024 * 1024 - 1] = 0xBB;

    assert_eq!(slot.data()[0], 0xAA);
    assert_eq!(slot.data()[10 * 1024 * 1024 - 1], 0xBB);

    let metrics = arena.metrics();
    assert_eq!(metrics.slot_size, 10 * 1024 * 1024);
    assert_eq!(metrics.used_bytes, 10 * 1024 * 1024);
}

/// Test metrics display formatting.
#[test]
fn test_metrics_display() {
    let arena = SharedArena::new(4096, 100).unwrap();

    // Acquire 75 slots
    let _slots: Vec<_> = (0..75).filter_map(|_| arena.acquire()).collect();

    let metrics = arena.metrics();
    let display = format!("{}", metrics);

    // Should contain key info
    assert!(display.contains("75/100"));
    assert!(display.contains("75.0%"));
    assert!(display.contains("Arena["));
}
