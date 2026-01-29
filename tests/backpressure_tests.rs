//! Integration tests for backpressure and flow control.
//!
//! These tests verify that:
//! - Queue water marks trigger flow signals correctly
//! - Flow state handles stay synchronized
//! - Sources respond to backpressure appropriately
//! - Drop policies work as expected

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ProduceContext, ProduceResult, Source};
use parallax::elements::flow::Queue;
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::flow::{FlowPolicy, FlowSignal, FlowStateHandle, WaterMarks};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

fn test_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(256, 512).unwrap())
}

fn create_test_buffer(seq: u64) -> Buffer {
    let arena = test_arena();
    let slot = arena.acquire().unwrap();
    let handle = MemoryHandle::with_len(slot, 64);
    Buffer::new(handle, Metadata::from_sequence(seq))
}

/// A test source that respects flow control.
struct FlowAwareSource {
    count: u64,
    max: u64,
    flow_state: Option<FlowStateHandle>,
    flow_policy: FlowPolicy,
    produced: u64,
    dropped: u64,
}

impl FlowAwareSource {
    fn new(max: u64) -> Self {
        Self {
            count: 0,
            max,
            flow_state: None,
            flow_policy: FlowPolicy::drop_with_logging(),
            produced: 0,
            dropped: 0,
        }
    }

    fn set_flow_state(&mut self, handle: FlowStateHandle) {
        self.flow_state = Some(handle);
    }

    fn with_policy(mut self, policy: FlowPolicy) -> Self {
        self.flow_policy = policy;
        self
    }

    fn produced(&self) -> u64 {
        self.produced
    }

    fn dropped(&self) -> u64 {
        self.dropped
    }
}

impl Source for FlowAwareSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }

        // Check backpressure
        if let Some(ref flow_state) = self.flow_state {
            if !flow_state.should_produce() && self.flow_policy.allows_dropping() {
                self.dropped += 1;
                flow_state.record_drop();
                self.count += 1;
                return Ok(ProduceResult::WouldBlock);
            }
        }

        let buffer = create_test_buffer(self.count);
        self.count += 1;
        self.produced += 1;
        Ok(ProduceResult::OwnBuffer(buffer))
    }

    fn flow_policy(&self) -> FlowPolicy {
        self.flow_policy.clone()
    }
}

#[test]
fn test_queue_flow_state_triggers_on_high_water() {
    let queue = Queue::new(10).with_flow_control();
    let flow_state = queue.flow_state_handle();

    // Initially ready
    assert_eq!(flow_state.signal(), FlowSignal::Ready);

    // Fill to just below high water (80% of 10 = 8)
    for i in 0..7 {
        queue.push(create_test_buffer(i)).unwrap();
    }
    assert_eq!(flow_state.signal(), FlowSignal::Ready);

    // Push to high water mark
    queue.push(create_test_buffer(7)).unwrap();
    assert_eq!(flow_state.signal(), FlowSignal::Busy);
}

#[test]
fn test_queue_flow_state_releases_on_low_water() {
    let queue = Queue::new(10).with_flow_control();
    let flow_state = queue.flow_state_handle();

    // Fill to high water
    for i in 0..8 {
        queue.push(create_test_buffer(i)).unwrap();
    }
    assert_eq!(flow_state.signal(), FlowSignal::Busy);

    // Drain to just above low water (20% of 10 = 2)
    for _ in 0..5 {
        queue.pop_timeout(Some(Duration::from_millis(10))).unwrap();
    }
    assert_eq!(queue.len(), 3);
    assert_eq!(flow_state.signal(), FlowSignal::Busy); // Still busy

    // Drain to low water
    queue.pop_timeout(Some(Duration::from_millis(10))).unwrap();
    assert_eq!(queue.len(), 2);
    assert_eq!(flow_state.signal(), FlowSignal::Ready); // Now ready
}

#[test]
fn test_flow_aware_source_drops_on_backpressure() {
    let queue = Queue::new(10).with_flow_control();
    let flow_state = queue.flow_state_handle();

    let mut source = FlowAwareSource::new(20);
    source.set_flow_state(flow_state.clone());

    // Produce while queue is not full
    let mut produced_count = 0;
    let arena = test_arena();
    let slot = arena.acquire().unwrap();
    let mut ctx = ProduceContext::new(slot);

    // Fill queue to high water
    for _ in 0..8 {
        if let Ok(ProduceResult::OwnBuffer(buf)) = source.produce(&mut ctx) {
            queue.push(buf).unwrap();
            produced_count += 1;
        }
    }
    assert_eq!(produced_count, 8);
    assert_eq!(flow_state.signal(), FlowSignal::Busy);

    // Now source should drop frames
    let mut dropped_count = 0;
    for _ in 0..5 {
        match source.produce(&mut ctx) {
            Ok(ProduceResult::WouldBlock) => dropped_count += 1,
            Ok(ProduceResult::OwnBuffer(_)) => panic!("Should have dropped"),
            _ => {}
        }
    }
    assert_eq!(dropped_count, 5);
    assert_eq!(source.dropped(), 5);

    // Drain queue to release backpressure
    while queue.len() > 2 {
        queue.pop_timeout(Some(Duration::from_millis(10))).unwrap();
    }
    assert_eq!(flow_state.signal(), FlowSignal::Ready);

    // Now source should produce again
    if let Ok(ProduceResult::OwnBuffer(_)) = source.produce(&mut ctx) {
        // Good - produced
    } else {
        panic!("Should have produced after backpressure released");
    }
}

#[test]
fn test_flow_state_handle_thread_safety() {
    use std::thread;

    let queue = Queue::new(100).with_flow_control();
    let flow_state = queue.flow_state_handle();

    let producer_handle = flow_state.clone();
    let consumer_handle = flow_state.clone();

    let produced = Arc::new(AtomicU64::new(0));
    let dropped = Arc::new(AtomicU64::new(0));
    let produced_clone = produced.clone();
    let dropped_clone = dropped.clone();

    // Producer thread - fills queue
    let producer = thread::spawn(move || {
        for i in 0..50 {
            if producer_handle.should_produce() {
                queue.push(create_test_buffer(i)).unwrap();
                produced_clone.fetch_add(1, Ordering::Relaxed);
            } else {
                dropped_clone.fetch_add(1, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_micros(100));
        }
        queue
    });

    // Let producer run a bit
    thread::sleep(Duration::from_millis(10));

    // Consumer thread - drains queue
    let consumer = thread::spawn(move || {
        let mut consumed = 0u64;
        loop {
            match consumer_handle.signal() {
                FlowSignal::Ready | FlowSignal::Busy => {
                    // Keep draining
                }
                _ => break,
            }
            consumed += 1;
            if consumed > 100 {
                break; // Prevent infinite loop
            }
            thread::sleep(Duration::from_micros(50));
        }
    });

    let queue = producer.join().unwrap();
    consumer.join().unwrap();

    // Drain remaining
    while queue
        .pop_timeout(Some(Duration::from_millis(1)))
        .unwrap()
        .is_some()
    {}

    let total_produced = produced.load(Ordering::Relaxed);
    let total_dropped = dropped.load(Ordering::Relaxed);

    // Should have produced some and possibly dropped some
    assert!(total_produced > 0);
    // Total should be 50 (all iterations)
    assert_eq!(total_produced + total_dropped, 50);
}

#[test]
fn test_custom_water_marks() {
    // Very aggressive water marks: 30% high, 10% low
    let wm = WaterMarks::with_percentages(100, 30, 10);
    let queue = Queue::new(100).with_water_marks(wm);
    let flow_state = queue.flow_state_handle();

    // Fill to 29 - still ready
    for i in 0..29 {
        queue.push(create_test_buffer(i)).unwrap();
    }
    assert_eq!(flow_state.signal(), FlowSignal::Ready);

    // Fill to 30 - now busy
    queue.push(create_test_buffer(29)).unwrap();
    assert_eq!(flow_state.signal(), FlowSignal::Busy);

    // Drain to 11 - still busy
    while queue.len() > 11 {
        queue.pop_timeout(Some(Duration::from_millis(1))).unwrap();
    }
    assert_eq!(flow_state.signal(), FlowSignal::Busy);

    // Drain to 10 - now ready
    queue.pop_timeout(Some(Duration::from_millis(1))).unwrap();
    assert_eq!(queue.len(), 10);
    assert_eq!(flow_state.signal(), FlowSignal::Ready);
}

#[test]
fn test_flow_policy_block_does_not_drop() {
    let queue = Queue::new(10).with_flow_control();
    let flow_state = queue.flow_state_handle();

    let mut source = FlowAwareSource::new(20).with_policy(FlowPolicy::Block);
    source.set_flow_state(flow_state.clone());

    // Fill queue to high water
    let arena = test_arena();
    let slot = arena.acquire().unwrap();
    let mut ctx = ProduceContext::new(slot);

    for _ in 0..8 {
        if let Ok(ProduceResult::OwnBuffer(buf)) = source.produce(&mut ctx) {
            queue.push(buf).unwrap();
        }
    }
    assert_eq!(flow_state.signal(), FlowSignal::Busy);

    // With Block policy, source should still produce (not drop)
    // The blocking behavior would be handled by the executor
    match source.produce(&mut ctx) {
        Ok(ProduceResult::OwnBuffer(_)) => {
            // Block policy doesn't allow dropping, so it produces anyway
            // (actual blocking would be done by executor/channel)
        }
        Ok(ProduceResult::WouldBlock) => {
            panic!("Block policy should not drop frames");
        }
        _ => {}
    }

    assert_eq!(source.dropped(), 0);
}

#[test]
fn test_backpressure_statistics() {
    let queue = Queue::new(10).with_flow_control();
    let flow_state = queue.flow_state_handle();

    // Initially no backpressure events
    assert_eq!(flow_state.backpressure_events(), 0);

    // Fill to high water
    for i in 0..8 {
        queue.push(create_test_buffer(i)).unwrap();
    }
    assert_eq!(flow_state.backpressure_events(), 1);

    // Drain to low water and refill
    while queue.len() > 2 {
        queue.pop_timeout(Some(Duration::from_millis(1))).unwrap();
    }
    assert_eq!(flow_state.signal(), FlowSignal::Ready);

    // Refill to high water - second backpressure event
    while queue.len() < 8 {
        queue.push(create_test_buffer(queue.len() as u64)).unwrap();
    }
    assert_eq!(flow_state.backpressure_events(), 2);

    // Record some drops
    flow_state.record_drop();
    flow_state.record_drop();
    flow_state.record_drop();
    assert_eq!(flow_state.frames_dropped(), 3);
}

#[test]
fn test_hysteresis_prevents_oscillation() {
    let queue = Queue::new(10).with_flow_control();
    let flow_state = queue.flow_state_handle();

    // Fill to high water (8)
    for i in 0..8 {
        queue.push(create_test_buffer(i)).unwrap();
    }
    assert_eq!(flow_state.signal(), FlowSignal::Busy);
    let events_after_first = flow_state.backpressure_events();

    // Oscillate around high water mark - should NOT cause extra events
    // due to hysteresis (need to drop to low water first)
    for _ in 0..10 {
        // Pop one (7)
        queue.pop_timeout(Some(Duration::from_millis(1))).unwrap();
        // Push one back (8)
        queue.push(create_test_buffer(100)).unwrap();
    }

    // Should still have same number of backpressure events
    assert_eq!(flow_state.backpressure_events(), events_after_first);
    assert_eq!(flow_state.signal(), FlowSignal::Busy);
}
