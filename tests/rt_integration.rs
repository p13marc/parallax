//! Integration tests for the RT scheduling system.
//!
//! These tests verify the complete RT path: graph partitioning,
//! bridge creation, data thread spawning, and end-to-end data flow
//! through hybrid pipelines mixing async and RT elements.
//!
//! This file also contains a tracking allocator to verify RT-safe
//! elements do not allocate during processing.

use parallax::buffer::Buffer;
use parallax::element::{
    Affinity, AsyncElementDyn, Element, ExecutionHints, LatencyHint, ProcessingHint,
};
use parallax::elements::{Gain, NullSink, NullSource, PassThrough};
use parallax::error::Result;
use parallax::pipeline::rt_scheduler::{ActivationRecord, RtConfig, RtScheduler, SchedulingMode};
use parallax::pipeline::{Executor, Pipeline, UnifiedExecutorConfig as ExecutorConfig};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

// ============================================================================
// Tracking allocator for allocation-free verification
// ============================================================================

use std::alloc::{GlobalAlloc, Layout, System};

/// A global allocator that counts allocations per thread.
struct TrackingAllocator;

// Thread-local allocation counter. Only counts when tracking is enabled.
std::thread_local! {
    static ALLOC_TRACKING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    static ALLOC_COUNT: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_TRACKING.with(|tracking| {
            if tracking.get() {
                ALLOC_COUNT.with(|count| count.set(count.get() + 1));
            }
        });
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

/// Start tracking allocations on the current thread. Returns the count at start.
fn start_alloc_tracking() -> u64 {
    ALLOC_COUNT.with(|c| {
        let current = c.get();
        ALLOC_TRACKING.with(|t| t.set(true));
        current
    })
}

/// Stop tracking and return the number of allocations since tracking started.
fn stop_alloc_tracking(start_count: u64) -> u64 {
    ALLOC_TRACKING.with(|t| t.set(false));
    ALLOC_COUNT.with(|c| c.get() - start_count)
}

// ============================================================================
// Helper: RT-safe test element that records processing
// ============================================================================

/// An RT-safe element that doubles each byte and counts buffers processed.
struct RtDoubler {
    count: Arc<AtomicU64>,
    name: String,
}

impl RtDoubler {
    fn new(name: &str, count: Arc<AtomicU64>) -> Self {
        Self {
            count,
            name: name.to_string(),
        }
    }
}

impl Element for RtDoubler {
    fn process(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);
        // Double each byte (wrapping)
        for byte in buffer.as_bytes_mut() {
            *byte = byte.wrapping_mul(2);
        }
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_rt_safe(&self) -> bool {
        true
    }

    fn affinity(&self) -> Affinity {
        Affinity::RealTime
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints {
            processing: ProcessingHint::CpuBound,
            latency: LatencyHint::Low,
            ..ExecutionHints::trusted()
        }
    }
}

/// A non-RT-safe element for testing that the async path works alongside RT.
struct AsyncCounter {
    count: Arc<AtomicU64>,
    name: String,
}

impl AsyncCounter {
    fn new(name: &str, count: Arc<AtomicU64>) -> Self {
        Self {
            count,
            name: name.to_string(),
        }
    }
}

impl Element for AsyncCounter {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_rt_safe(&self) -> bool {
        false
    }

    fn affinity(&self) -> Affinity {
        Affinity::Async
    }
}

/// Counting sink that records the total number of buffers received.
struct CountingSink {
    count: Arc<AtomicU64>,
}

impl CountingSink {
    fn new(count: Arc<AtomicU64>) -> Self {
        Self { count }
    }
}

impl parallax::element::Sink for CountingSink {
    fn consume(&mut self, _ctx: &parallax::element::ConsumeContext) -> Result<()> {
        self.count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn name(&self) -> &str {
        "counting_sink"
    }
}

// ============================================================================
// Test: Activation Record basics
// ============================================================================

#[test]
fn test_activation_record_dependency_tracking() {
    let activation = ActivationRecord::new().unwrap();

    // No dependencies → immediately ready
    activation.set_required(0);
    activation.reset_pending();
    assert!(activation.is_ready());

    // 3 dependencies → not ready until all satisfied
    activation.set_required(3);
    activation.reset_pending();
    assert!(!activation.is_ready());

    assert!(!activation.decrement_pending()); // 3→2
    assert!(!activation.is_ready());

    assert!(!activation.decrement_pending()); // 2→1
    assert!(!activation.is_ready());

    assert!(activation.decrement_pending()); // 1→0 → ready!
    assert!(activation.is_ready());
}

#[test]
fn test_activation_record_reset_between_cycles() {
    let activation = ActivationRecord::new().unwrap();
    activation.set_required(2);

    // Cycle 1
    activation.reset_pending();
    assert!(!activation.decrement_pending());
    assert!(activation.decrement_pending());
    assert!(activation.is_ready());

    // Cycle 2 — reset should restore pending
    activation.reset_pending();
    assert!(!activation.is_ready());
    assert!(!activation.decrement_pending());
    assert!(activation.decrement_pending());
    assert!(activation.is_ready());
}

// ============================================================================
// Test: Graph partitioning
// ============================================================================

#[tokio::test]
async fn test_partition_all_async() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let pt = pipeline.add_filter("pt", PassThrough::new());
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, pt).unwrap();
    pipeline.link(pt, sink).unwrap();

    // In Async mode, all nodes should be async even if RT-safe
    let scheduler = RtScheduler::new(RtConfig {
        mode: SchedulingMode::Async,
        ..Default::default()
    });
    let partition = scheduler.partition_graph(&pipeline).unwrap();

    assert!(!partition.has_rt_nodes());
    assert!(!partition.has_boundaries());
    assert_eq!(partition.async_nodes.len(), 3);
}

#[tokio::test]
async fn test_partition_hybrid_with_rt_element() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let rt_elem = pipeline.add_filter("rt", RtDoubler::new("rt", Arc::new(AtomicU64::new(0))));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, rt_elem).unwrap();
    pipeline.link(rt_elem, sink).unwrap();

    let scheduler = RtScheduler::new(RtConfig::hybrid());
    let partition = scheduler.partition_graph(&pipeline).unwrap();

    // RT element should be in RT partition
    assert!(partition.has_rt_nodes());
    assert_eq!(partition.rt_nodes.len(), 1);
    // Source + sink should be async
    assert_eq!(partition.async_nodes.len(), 2);
    // Two boundaries: src→rt (AsyncToRt) and rt→sink (RtToAsync)
    assert_eq!(partition.boundary_edges.len(), 2);
}

#[tokio::test]
async fn test_partition_mixed_pipeline() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let async_elem = pipeline.add_filter(
        "async_filter",
        AsyncCounter::new("async_filter", Arc::new(AtomicU64::new(0))),
    );
    let rt_elem = pipeline.add_filter("rt", RtDoubler::new("rt", Arc::new(AtomicU64::new(0))));
    let pt = pipeline.add_filter("pt", PassThrough::new()); // RT-safe with Auto affinity
    let sink = pipeline.add_sink("sink", NullSink::new());

    pipeline.link(src, async_elem).unwrap();
    pipeline.link(async_elem, rt_elem).unwrap();
    pipeline.link(rt_elem, pt).unwrap();
    pipeline.link(pt, sink).unwrap();

    let scheduler = RtScheduler::new(RtConfig::hybrid());
    let partition = scheduler.partition_graph(&pipeline).unwrap();

    // rt_elem has RealTime affinity → RT
    // PassThrough has Auto affinity + is_rt_safe → RT
    assert_eq!(partition.rt_nodes.len(), 2);
    // src + async_filter + sink → async
    assert_eq!(partition.async_nodes.len(), 3);
    assert!(partition.has_boundaries());
}

// ============================================================================
// Test: Bridge creation and data flow
// ============================================================================

#[tokio::test]
async fn test_bridge_creation_for_boundaries() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let rt_elem = pipeline.add_filter("rt", RtDoubler::new("rt", Arc::new(AtomicU64::new(0))));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, rt_elem).unwrap();
    pipeline.link(rt_elem, sink).unwrap();

    let mut scheduler = RtScheduler::new(RtConfig::hybrid());
    let partition = scheduler.partition_graph(&pipeline).unwrap();
    scheduler.create_bridges(&partition).unwrap();

    // Should have 2 bridges (one per boundary edge)
    assert_eq!(scheduler.bridges().len(), 2);

    // Bridges should be accessible by edge endpoints
    for edge in &partition.boundary_edges {
        assert!(
            scheduler.get_bridge(edge.source, edge.sink).is_some(),
            "bridge should exist for edge {:?} → {:?}",
            edge.source,
            edge.sink
        );
    }
}

// ============================================================================
// Test: Processing order computation
// ============================================================================

#[tokio::test]
async fn test_processing_order_respects_topology() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let rt1 = pipeline.add_filter("rt1", RtDoubler::new("rt1", Arc::new(AtomicU64::new(0))));
    let rt2 = pipeline.add_filter("rt2", RtDoubler::new("rt2", Arc::new(AtomicU64::new(0))));
    let sink = pipeline.add_sink("sink", NullSink::new());

    pipeline.link(src, rt1).unwrap();
    pipeline.link(rt1, rt2).unwrap();
    pipeline.link(rt2, sink).unwrap();

    let mut scheduler = RtScheduler::new(RtConfig::hybrid());
    let partition = scheduler.partition_graph(&pipeline).unwrap();
    scheduler
        .compute_processing_order(&partition, &pipeline)
        .unwrap();

    let order = scheduler.processing_order();
    assert_eq!(order.len(), 2); // rt1, rt2

    // rt1 must come before rt2
    let pos_rt1 = order.iter().position(|&id| id == rt1).unwrap();
    let pos_rt2 = order.iter().position(|&id| id == rt2).unwrap();
    assert!(
        pos_rt1 < pos_rt2,
        "rt1 (pos {}) must be processed before rt2 (pos {})",
        pos_rt1,
        pos_rt2
    );
}

// ============================================================================
// Test: Downstream map
// ============================================================================

#[tokio::test]
async fn test_downstream_map() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let rt1 = pipeline.add_filter("rt1", RtDoubler::new("rt1", Arc::new(AtomicU64::new(0))));
    let rt2 = pipeline.add_filter("rt2", RtDoubler::new("rt2", Arc::new(AtomicU64::new(0))));
    let sink = pipeline.add_sink("sink", NullSink::new());

    pipeline.link(src, rt1).unwrap();
    pipeline.link(rt1, rt2).unwrap();
    pipeline.link(rt2, sink).unwrap();

    let scheduler = RtScheduler::new(RtConfig::hybrid());
    let partition = scheduler.partition_graph(&pipeline).unwrap();
    let downstream_map = scheduler.build_downstream_map(&partition, &pipeline);

    // rt1 should have rt2 as downstream
    assert_eq!(downstream_map.get(&rt1).unwrap(), &vec![rt2]);
    // rt2 should have no RT downstream (sink is async)
    assert!(downstream_map.get(&rt2).unwrap().is_empty());
}

// ============================================================================
// Test: End-to-end hybrid pipeline (data flows through RT thread)
// ============================================================================

#[tokio::test]
async fn test_hybrid_pipeline_data_flow() {
    let rt_count = Arc::new(AtomicU64::new(0));
    let sink_count = Arc::new(AtomicU64::new(0));
    let num_buffers = 20u64;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(num_buffers));
    let rt_elem = pipeline.add_filter("rt", RtDoubler::new("rt", rt_count.clone()));
    let sink = pipeline.add_sink("sink", CountingSink::new(sink_count.clone()));

    pipeline.link(src, rt_elem).unwrap();
    pipeline.link(rt_elem, sink).unwrap();

    let config = ExecutorConfig::hybrid();
    let executor = Executor::with_config(config);

    // Run with a timeout to avoid hanging on failure
    let result = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        executor.run(&mut pipeline).await
    })
    .await;

    match result {
        Ok(Ok(())) => {
            // RT element should have processed all buffers
            assert_eq!(
                rt_count.load(Ordering::Relaxed),
                num_buffers,
                "RT element should have processed {} buffers",
                num_buffers
            );
            // Sink should have received all buffers
            assert_eq!(
                sink_count.load(Ordering::Relaxed),
                num_buffers,
                "Sink should have received {} buffers",
                num_buffers
            );
        }
        Ok(Err(e)) => panic!("Pipeline error: {}", e),
        Err(_) => panic!("Pipeline timed out after 10 seconds"),
    }
}

/// Test that PassThrough (RT-safe with Auto affinity) runs in the RT thread
/// when the executor is in Hybrid mode.
#[tokio::test]
async fn test_passthrough_in_hybrid_mode() {
    let sink_count = Arc::new(AtomicU64::new(0));
    let num_buffers = 15u64;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(num_buffers));
    let pt = pipeline.add_filter("pt", PassThrough::new());
    let sink = pipeline.add_sink("sink", CountingSink::new(sink_count.clone()));

    pipeline.link(src, pt).unwrap();
    pipeline.link(pt, sink).unwrap();

    let config = ExecutorConfig::hybrid();
    let executor = Executor::with_config(config);

    let result = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        executor.run(&mut pipeline).await
    })
    .await;

    match result {
        Ok(Ok(())) => {
            assert_eq!(
                sink_count.load(Ordering::Relaxed),
                num_buffers,
                "Sink should have received {} buffers",
                num_buffers
            );
        }
        Ok(Err(e)) => panic!("Pipeline error: {}", e),
        Err(_) => panic!("Pipeline timed out after 10 seconds"),
    }
}

// ============================================================================
// Test: Gain element in hybrid pipeline
// ============================================================================

#[tokio::test]
async fn test_gain_hybrid_pipeline() {
    let sink_count = Arc::new(AtomicU64::new(0));
    let num_buffers = 10u64;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(num_buffers));
    let gain = pipeline.add_filter("gain", Gain::new(2.0));
    let sink = pipeline.add_sink("sink", CountingSink::new(sink_count.clone()));

    pipeline.link(src, gain).unwrap();
    pipeline.link(gain, sink).unwrap();

    // Gain declares is_rt_safe()=true with Low latency hint and CpuBound processing,
    // so auto_strategy should detect it as RT
    let config = ExecutorConfig {
        auto_strategy: true,
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let result = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        executor.run(&mut pipeline).await
    })
    .await;

    match result {
        Ok(Ok(())) => {
            assert_eq!(
                sink_count.load(Ordering::Relaxed),
                num_buffers,
                "Sink should have received {} buffers through Gain in hybrid mode",
                num_buffers
            );
        }
        Ok(Err(e)) => panic!("Pipeline error: {}", e),
        Err(_) => panic!("Pipeline timed out after 10 seconds"),
    }
}

// ============================================================================
// Test: Mixed pipeline — 3 async elements + 2 RT elements
// ============================================================================

#[tokio::test]
async fn test_mixed_async_rt_pipeline() {
    let async_count = Arc::new(AtomicU64::new(0));
    let rt1_count = Arc::new(AtomicU64::new(0));
    let rt2_count = Arc::new(AtomicU64::new(0));
    let sink_count = Arc::new(AtomicU64::new(0));
    let num_buffers = 10u64;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(num_buffers));
    let async_elem = pipeline.add_filter(
        "async_filter",
        AsyncCounter::new("async_filter", async_count.clone()),
    );
    let rt1 = pipeline.add_filter("rt1", RtDoubler::new("rt1", rt1_count.clone()));
    let rt2 = pipeline.add_filter("rt2", RtDoubler::new("rt2", rt2_count.clone()));
    let sink = pipeline.add_sink("sink", CountingSink::new(sink_count.clone()));

    pipeline.link(src, async_elem).unwrap();
    pipeline.link(async_elem, rt1).unwrap();
    pipeline.link(rt1, rt2).unwrap();
    pipeline.link(rt2, sink).unwrap();

    let config = ExecutorConfig::hybrid();
    let executor = Executor::with_config(config);

    let result = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        executor.run(&mut pipeline).await
    })
    .await;

    match result {
        Ok(Ok(())) => {
            assert_eq!(
                async_count.load(Ordering::Relaxed),
                num_buffers,
                "Async element should have processed all buffers"
            );
            assert_eq!(
                rt1_count.load(Ordering::Relaxed),
                num_buffers,
                "RT1 should have processed all buffers"
            );
            assert_eq!(
                rt2_count.load(Ordering::Relaxed),
                num_buffers,
                "RT2 should have processed all buffers"
            );
            assert_eq!(
                sink_count.load(Ordering::Relaxed),
                num_buffers,
                "Sink should have received all buffers"
            );
        }
        Ok(Err(e)) => panic!("Pipeline error: {}", e),
        Err(_) => panic!("Pipeline timed out after 10 seconds"),
    }
}

// ============================================================================
// Test: RT thread is actually spawned (thread name check)
// ============================================================================

#[tokio::test]
async fn test_rt_thread_name() {
    use std::sync::Mutex;

    // We'll check that a thread named "parallax-rt-*" exists during execution
    let thread_found = Arc::new(Mutex::new(false));
    let thread_found_clone = thread_found.clone();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(100));
    let rt_elem = pipeline.add_filter("rt", RtDoubler::new("rt", Arc::new(AtomicU64::new(0))));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, rt_elem).unwrap();
    pipeline.link(rt_elem, sink).unwrap();

    let config = ExecutorConfig::hybrid();
    let executor = Executor::with_config(config);

    let handle = executor.start(&mut pipeline).unwrap();

    // Give the RT thread time to start
    tokio::time::sleep(std::time::Duration::from_millis(50)).await;

    // Check /proc for threads with our naming pattern
    if let Ok(entries) = std::fs::read_dir("/proc/self/task") {
        for entry in entries.flatten() {
            if let Ok(comm) = std::fs::read_to_string(entry.path().join("comm")) {
                let comm = comm.trim();
                if comm.starts_with("parallax-rt") {
                    *thread_found_clone.lock().unwrap() = true;
                    break;
                }
            }
        }
    }

    // Abort to avoid waiting for all buffers
    handle.abort();

    assert!(
        *thread_found.lock().unwrap(),
        "Expected to find a thread named 'parallax-rt-*' in /proc/self/task"
    );
}

// ============================================================================
// Test: SyncElement trait dispatch
// ============================================================================

#[test]
fn test_sync_element_dispatch_on_rt_safe_element() {
    use parallax::element::{DynAsyncElement, ElementAdapter};

    let passthrough = PassThrough::new();
    let mut adapter = DynAsyncElement::new_box(ElementAdapter::new(passthrough));

    // PassThrough is RT-safe, so as_sync_element should return Some
    assert!(
        adapter.as_sync_element().is_some(),
        "RT-safe element should provide SyncElement interface"
    );
}

#[test]
fn test_sync_element_dispatch_on_non_rt_safe_element() {
    use parallax::element::{DynAsyncElement, ElementAdapter};

    let counter = AsyncCounter::new("test", Arc::new(AtomicU64::new(0)));
    let mut adapter = DynAsyncElement::new_box(ElementAdapter::new(counter));

    // AsyncCounter is not RT-safe, so as_sync_element should return None
    assert!(
        adapter.as_sync_element().is_none(),
        "Non-RT-safe element should NOT provide SyncElement interface"
    );
}

// ============================================================================
// Test: Dependency setup in activations
// ============================================================================

#[tokio::test]
async fn test_dependency_setup() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let rt1 = pipeline.add_filter("rt1", RtDoubler::new("rt1", Arc::new(AtomicU64::new(0))));
    let rt2 = pipeline.add_filter("rt2", RtDoubler::new("rt2", Arc::new(AtomicU64::new(0))));
    let sink = pipeline.add_sink("sink", NullSink::new());

    pipeline.link(src, rt1).unwrap();
    pipeline.link(rt1, rt2).unwrap();
    pipeline.link(rt2, sink).unwrap();

    let mut scheduler = RtScheduler::new(RtConfig::hybrid());
    let partition = scheduler.partition_graph(&pipeline).unwrap();
    scheduler.setup_activations(&partition).unwrap();
    scheduler.setup_dependencies(&partition, &pipeline).unwrap();

    // rt1 has 0 RT→RT dependencies (its input comes from async bridge, not an RT node)
    let rt1_activation = scheduler.get_activation(rt1).unwrap();
    assert_eq!(
        rt1_activation.required.load(Ordering::Relaxed),
        0,
        "rt1 should have 0 RT dependencies (bridge inputs are polled, not activation-tracked)"
    );

    // rt2 has 1 RT dependency (rt1) → required=1
    let rt2_activation = scheduler.get_activation(rt2).unwrap();
    assert_eq!(
        rt2_activation.required.load(Ordering::Relaxed),
        1,
        "rt2 should have 1 dependency (from rt1)"
    );
}

// ============================================================================
// Test: Auto-strategy detection
// ============================================================================

#[tokio::test]
async fn test_auto_strategy_detects_rt_elements() {
    let sink_count = Arc::new(AtomicU64::new(0));
    let rt_count = Arc::new(AtomicU64::new(0));
    let num_buffers = 5u64;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(num_buffers));
    let rt_elem = pipeline.add_filter("rt", RtDoubler::new("rt", rt_count.clone()));
    let sink = pipeline.add_sink("sink", CountingSink::new(sink_count.clone()));

    pipeline.link(src, rt_elem).unwrap();
    pipeline.link(rt_elem, sink).unwrap();

    // Use auto_strategy — should detect the RT element and use hybrid
    let config = ExecutorConfig {
        auto_strategy: true,
        ..Default::default()
    };
    let executor = Executor::with_config(config);

    let result = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        executor.run(&mut pipeline).await
    })
    .await;

    match result {
        Ok(Ok(())) => {
            assert_eq!(rt_count.load(Ordering::Relaxed), num_buffers);
            assert_eq!(sink_count.load(Ordering::Relaxed), num_buffers);
        }
        Ok(Err(e)) => panic!("Pipeline error: {}", e),
        Err(_) => panic!("Pipeline timed out"),
    }
}

// ============================================================================
// Test: Pure async fallback (no RT elements)
// ============================================================================

#[tokio::test]
async fn test_hybrid_falls_back_to_async_when_no_rt() {
    let sink_count = Arc::new(AtomicU64::new(0));
    let num_buffers = 10u64;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(num_buffers));
    let async_elem = pipeline.add_filter(
        "async_filter",
        AsyncCounter::new("async_filter", Arc::new(AtomicU64::new(0))),
    );
    let sink = pipeline.add_sink("sink", CountingSink::new(sink_count.clone()));

    pipeline.link(src, async_elem).unwrap();
    pipeline.link(async_elem, sink).unwrap();

    // Hybrid mode with no RT elements should fall back to pure async
    let config = ExecutorConfig::hybrid();
    let executor = Executor::with_config(config);

    let result = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        executor.run(&mut pipeline).await
    })
    .await;

    match result {
        Ok(Ok(())) => {
            assert_eq!(sink_count.load(Ordering::Relaxed), num_buffers);
        }
        Ok(Err(e)) => panic!("Pipeline error: {}", e),
        Err(_) => panic!("Pipeline timed out"),
    }
}

// ============================================================================
// Test: ManualDriver gated processing
// ============================================================================

/// Verify that the RT data thread only processes when the driver fires.
///
/// This uses `spawn_data_thread` directly with a ManualDriver's eventfd,
/// pushes buffers into the input bridge, fires the driver a specific number
/// of times, and verifies the output bridge receives the expected data.
#[test]
fn test_manual_driver_gated_processing() {
    use parallax::buffer::MemoryHandle;
    use parallax::memory::SharedArena;
    use parallax::metadata::Metadata;
    use parallax::pipeline::rt_scheduler::spawn_data_thread;
    use parallax::pipeline::{AsyncRtBridge, BridgeConfig, DriverConfig, ManualDriver};
    use std::collections::HashMap;

    let arena = SharedArena::new(1024, 64).unwrap();

    // Build a pipeline to get valid NodeIds, then extract the RT element
    let mut pipeline = Pipeline::new();
    let _src = pipeline.add_source("src", NullSource::new(100));
    let rt_node = pipeline.add_filter("rt", PassThrough::new());
    let _sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(_src, rt_node).unwrap();
    pipeline.link(rt_node, _sink).unwrap();

    // Extract the RT element from the pipeline
    let node = pipeline.get_node_mut(rt_node).unwrap();
    let element = node.take_element().unwrap();

    let mut elements = HashMap::new();
    elements.insert(rt_node, element);

    // Create activation record (0 RT deps — input comes from bridge)
    let activation = Arc::new(ActivationRecord::new().unwrap());
    activation.set_required(0);
    let mut activations = HashMap::new();
    activations.insert(rt_node, activation);

    // Create input and output bridges
    let input_bridge = Arc::new(AsyncRtBridge::new(BridgeConfig::with_capacity(16)).unwrap());
    let output_bridge = Arc::new(AsyncRtBridge::new(BridgeConfig::with_capacity(16)).unwrap());

    let mut input_bridges = HashMap::new();
    input_bridges.insert(rt_node, input_bridge.clone());
    let mut output_bridges = HashMap::new();
    output_bridges.insert(rt_node, output_bridge.clone());

    // Create driver
    let driver = ManualDriver::new(DriverConfig::default()).unwrap();
    let driver_trigger = driver.eventfd().clone();

    // Spawn the data thread
    let handle = spawn_data_thread(
        "test-manual-driver".to_string(),
        RtConfig::hybrid(),
        vec![rt_node],
        activations,
        elements,
        input_bridges,
        output_bridges,
        HashMap::new(), // no downstream RT nodes
        driver_trigger,
    )
    .unwrap();

    // Push 5 buffers into the input bridge
    for i in 0..5u64 {
        let slot = arena.acquire().unwrap();
        let mem_handle = MemoryHandle::with_len(slot, 64);
        let buffer = Buffer::new(mem_handle, Metadata::from_sequence(i));
        input_bridge.try_push(buffer).unwrap();
    }

    // Before firing the driver, output should be empty
    assert!(
        output_bridge.is_empty(),
        "Output bridge should be empty before driver fires"
    );

    // Fire the driver exactly 3 times — each cycle should process one buffer
    for _ in 0..3 {
        driver.trigger().unwrap();
        // Give data thread time to wake and process
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Should have ~3 buffers in output (one per cycle)
    let output_count = output_bridge.len();
    assert!(
        output_count >= 2 && output_count <= 4,
        "Expected ~3 buffers in output after 3 driver cycles, got {output_count}"
    );

    // Fire 2 more times to drain remaining
    for _ in 0..2 {
        driver.trigger().unwrap();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }

    // Now signal EOS on input bridge and fire driver to let thread notice
    input_bridge.signal_eos();
    driver.trigger().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(50));

    // Drain remaining output and count total
    let mut total = 0;
    while output_bridge.try_pop().is_some() {
        total += 1;
    }
    assert_eq!(total, 5, "All 5 buffers should have been processed");

    // Stop the data thread
    handle.signal_stop();
    // Fire driver once more to unblock the thread's wait
    // (ignore error if eventfd was already consumed)
    let _ = driver.trigger();
    handle.join().unwrap();

    // Verify driver stats
    let stats = driver.stats();
    assert!(
        stats.cycles >= 6,
        "Expected at least 6 driver cycles, got {}",
        stats.cycles
    );
}

// ============================================================================
// Test: RT processing is allocation-free
// ============================================================================

/// Verify that SyncElement::process_sync() on RT-safe elements does not allocate.
///
/// This uses the tracking allocator to count heap allocations during
/// the sync processing path of PassThrough and Gain elements.
#[test]
fn test_rt_processing_allocation_free() {
    use parallax::buffer::MemoryHandle;
    use parallax::element::{DynAsyncElement, ElementAdapter};
    use parallax::memory::SharedArena;
    use parallax::metadata::Metadata;

    // Use 512 slots: each test section does 1 warmup + 50 iterations,
    // and slots require reclaim() to reuse after drop.
    let arena = SharedArena::new(1024, 512).unwrap();

    // --- Test PassThrough ---
    {
        let passthrough = PassThrough::new();
        let mut adapter = DynAsyncElement::new_box(ElementAdapter::new(passthrough));
        let sync_elem = adapter
            .as_sync_element()
            .expect("PassThrough should be SyncElement");

        // Warm up: first call may do lazy initialization
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 64);
        let warmup = Buffer::new(handle, Metadata::from_sequence(0));
        let _ = sync_elem.process_sync(warmup);

        // Now measure: process 50 buffers and count allocations
        let start = start_alloc_tracking();
        for i in 1..=50u64 {
            let slot = arena.acquire().unwrap();
            let handle = MemoryHandle::with_len(slot, 64);
            let buffer = Buffer::new(handle, Metadata::from_sequence(i));
            let result = sync_elem.process_sync(buffer);
            assert!(result.is_ok());
            drop(result);
        }
        let allocs = stop_alloc_tracking(start);

        assert_eq!(
            allocs, 0,
            "PassThrough::process_sync() allocated {} times in 50 calls (expected 0)",
            allocs
        );
    }

    // --- Test Gain ---
    {
        let gain = Gain::new(2.0);
        let mut adapter = DynAsyncElement::new_box(ElementAdapter::new(gain));
        let sync_elem = adapter
            .as_sync_element()
            .expect("Gain should be SyncElement");

        // Warm up
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 64);
        let warmup = Buffer::new(handle, Metadata::from_sequence(0));
        let _ = sync_elem.process_sync(warmup);

        // Measure
        let start = start_alloc_tracking();
        for i in 1..=50u64 {
            let slot = arena.acquire().unwrap();
            let handle = MemoryHandle::with_len(slot, 64);
            let buffer = Buffer::new(handle, Metadata::from_sequence(i));
            let result = sync_elem.process_sync(buffer);
            assert!(result.is_ok());
            drop(result);
        }
        let allocs = stop_alloc_tracking(start);

        assert_eq!(
            allocs, 0,
            "Gain::process_sync() allocated {} times in 50 calls (expected 0)",
            allocs
        );
    }

    // --- Test RtDoubler (custom RT element) ---
    {
        let doubler = RtDoubler::new("test", Arc::new(AtomicU64::new(0)));
        let mut adapter = DynAsyncElement::new_box(ElementAdapter::new(doubler));
        let sync_elem = adapter
            .as_sync_element()
            .expect("RtDoubler should be SyncElement");

        // Warm up
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 64);
        let warmup = Buffer::new(handle, Metadata::from_sequence(0));
        let _ = sync_elem.process_sync(warmup);

        // Measure
        let start = start_alloc_tracking();
        for i in 1..=50u64 {
            let slot = arena.acquire().unwrap();
            let handle = MemoryHandle::with_len(slot, 64);
            let buffer = Buffer::new(handle, Metadata::from_sequence(i));
            let result = sync_elem.process_sync(buffer);
            assert!(result.is_ok());
            drop(result);
        }
        let allocs = stop_alloc_tracking(start);

        assert_eq!(
            allocs, 0,
            "RtDoubler::process_sync() allocated {} times in 50 calls (expected 0)",
            allocs
        );
    }
}
