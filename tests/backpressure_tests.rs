//! Integration tests for backpressure and flow control.
//!
//! The link channel is the queue (#167): capacity + `LinkPolicy` per link,
//! occupancy-driven `FlowSignal`s via `Pipeline::monitor_link`. These tests
//! verify the executor-produced signals end to end, plus the thread-safety
//! of the shared handle itself.

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ProduceContext, ProduceResult, Source};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::flow::{FlowSignal, FlowStateHandle, WaterMarks, new_flow_state};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

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

/// The shared handle is a plain Arc of atomics: signals written on one
/// thread are visible on another, and drop/backpressure counters merge.
#[test]
fn test_flow_state_handle_thread_safety() {
    use std::thread;

    let state = new_flow_state();
    let writer = state.clone();
    let reader = state.clone();

    let t = thread::spawn(move || {
        for i in 0..100u64 {
            writer.set_signal(if i % 2 == 0 {
                FlowSignal::Busy
            } else {
                FlowSignal::Ready
            });
            writer.record_drop();
        }
    });
    // Concurrent reads must never see anything but the two valid states.
    for _ in 0..100 {
        let s = reader.signal();
        assert!(matches!(s, FlowSignal::Ready | FlowSignal::Busy));
    }
    t.join().unwrap();
    assert_eq!(state.frames_dropped(), 100);
    assert_eq!(state.signal(), FlowSignal::Ready, "last write wins");
    assert!(state.backpressure_events() >= 1);
}

// ============================================================================
// Executor-produced link flow signals (#167): the link channel IS the queue,
// and `Pipeline::monitor_link` drives a FlowStateHandle from its occupancy —
// sampled on the send side (fills) and the receive side (drains).
// ============================================================================

mod link_monitoring {
    use super::*;
    use parallax::elements::{AppSink, AppSrc, Pulled};
    use parallax::pipeline::{Executor, LinkPolicy, Pipeline};

    /// Poll a condition without timers: an AppSink whose queue is full holds
    /// its task in `consume`, and timer-based waits against blocked elements
    /// can stall the tokio time driver (see CLAUDE.md).
    async fn wait_for(what: &str, mut cond: impl FnMut() -> bool) {
        for _ in 0..200_000 {
            if cond() {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!("timed out waiting for {what}");
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn monitored_link_reports_busy_then_ready() {
        let mut pipeline = Pipeline::new();
        let src = AppSrc::with_max_buffers(64);
        let src_handle = src.handle();
        let sink = AppSink::with_max_buffers(1);
        let sink_handle = sink.handle();
        let s = pipeline.add_source("appsrc", src);
        let k = pipeline.add_async_sink("appsink", sink);
        let link = pipeline
            .link_pads_full(s, "src", k, "sink", LinkPolicy::Block, Some(8))
            .unwrap();
        let flow = pipeline
            .monitor_link_with(link, WaterMarks::new(6, 1))
            .unwrap();

        let executor = Executor::new();
        let handle = executor.start(&mut pipeline).unwrap();

        // The sink is not pulled, so the link channel backs up past the
        // high mark; the send-side samples flip the signal to Busy.
        for i in 0..20u64 {
            src_handle.push_buffer(create_test_buffer(i)).await.unwrap();
        }
        wait_for("Busy at the high mark", || {
            flow.signal() == FlowSignal::Busy
        })
        .await;
        assert!(!flow.should_produce());
        assert!(flow.backpressure_events() >= 1);

        // Draining the sink empties the channel; the receive-side samples
        // flip it back to Ready at the low mark.
        src_handle.end_stream();
        let mut got = 0;
        while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {
            got += 1;
        }
        assert_eq!(got, 20, "Block policy loses nothing");
        wait_for("Ready at the low mark", || {
            flow.signal() == FlowSignal::Ready
        })
        .await;
        handle.wait().await.unwrap();
    }

    /// A live-source stand-in: checks `should_produce()` before the
    /// expensive part, the way v4l2/screen-capture do.
    struct GatedSource {
        flow: Option<FlowStateHandle>,
        skipped: Arc<AtomicU64>,
        produced: u64,
        total: u64,
    }

    impl GatedSource {
        fn new(total: u64, skipped: Arc<AtomicU64>) -> Self {
            Self {
                flow: None,
                skipped,
                produced: 0,
                total,
            }
        }

        fn set_flow_state(&mut self, flow: FlowStateHandle) {
            self.flow = Some(flow);
        }
    }

    impl Source for GatedSource {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
            if self.produced >= self.total {
                return Ok(ProduceResult::Eos);
            }
            if let Some(flow) = &self.flow
                && !flow.should_produce()
            {
                self.skipped.fetch_add(1, Ordering::Relaxed);
                return Ok(ProduceResult::WouldBlock);
            }
            let seq = self.produced;
            self.produced += 1;
            Ok(ProduceResult::OwnBuffer(create_test_buffer(seq)))
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn flow_aware_source_skips_while_busy_and_resumes() {
        let skipped = Arc::new(AtomicU64::new(0));
        let mut pipeline = Pipeline::new();
        let sink = AppSink::with_max_buffers(1);
        let sink_handle = sink.handle();
        let s = pipeline.add_source("gated", GatedSource::new(40, skipped.clone()));
        let k = pipeline.add_async_sink("appsink", sink);
        // Drop policy, as a live source would use — but the gate closes at
        // the high mark (6), before the channel (8) can fill, so nothing is
        // actually dropped: the source skips instead.
        let link = pipeline
            .link_pads_full(s, "src", k, "sink", LinkPolicy::DropNewest, Some(8))
            .unwrap();
        let flow = pipeline
            .monitor_link_with(link, WaterMarks::new(6, 1))
            .unwrap();
        pipeline
            .get_element_mut::<GatedSource>("gated")
            .unwrap()
            .set_flow_state(flow.clone());

        let executor = Executor::new();
        let handle = executor.start(&mut pipeline).unwrap();

        // Un-pulled sink → channel backs up → the source starts skipping.
        wait_for("source skips while Busy", || {
            skipped.load(Ordering::Relaxed) > 0
        })
        .await;

        // Drain everything: the source resumes at the low mark and finishes.
        let mut got = 0;
        while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {
            got += 1;
        }
        handle.wait().await.unwrap();
        assert_eq!(got, 40, "skipping deferred production, lost nothing");
        assert!(flow.backpressure_events() >= 1);
    }
}
