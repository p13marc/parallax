//! #168: a bounded `AppSink` back-pressures by *awaiting* space, never by
//! parking its tokio worker.
//!
//! Every part of the shape below is load-bearing, so don't "simplify" it:
//!
//! - a **burst source** with no await between buffers, so messages are already
//!   queued on the link channel when the sink task is polled — that is what
//!   makes the sink run several `consume`s inside a single poll;
//! - a bounded sink with `drop_on_full` **off**, so the second consume in that
//!   poll has to wait;
//! - a **spawned** puller: the notify from consume #1 pushes it into the sink
//!   worker's non-stealable LIFO slot. A puller inlined in the test task is
//!   immune to the bug and would prove nothing;
//! - `worker_threads = 2`: with one worker the wedge takes the time driver
//!   down with it, so no timeout could ever fire and the test would hang
//!   instead of failing. With two, worker A is the casualty and worker B still
//!   runs the timeout, which turns the deadlock into a diagnosis.
//!
//! Verified to fail against the pre-fix `AppSink` (blocking condvar `Sink`).

use std::sync::OnceLock;
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ProduceContext, ProduceResult, Source};
use parallax::elements::{AppSink, EndReason, Pulled};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, LinkPolicy, Pipeline};

fn arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 128).unwrap())
}

/// Produces `total` buffers as fast as it is polled, with no await and no
/// sleep — the shape that fills a link channel before the sink is polled once.
struct BurstSource {
    produced: u64,
    total: u64,
}

impl Source for BurstSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.produced >= self.total {
            return Ok(ProduceResult::Eos);
        }
        // Released slots come back through the arena's release queue and only
        // become free once the owner drains it — an endless in-test producer
        // that never reclaims runs the arena dry and stalls forever.
        arena().reclaim();
        let Some(slot) = arena().acquire() else {
            return Ok(ProduceResult::WouldBlock);
        };
        self.produced += 1;
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            Metadata::from_sequence(self.produced),
        )))
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_bounded_appsink_backpressures_without_wedging_its_worker() {
    const TOTAL: u64 = 200;

    let mut pipeline = Pipeline::new();
    // max_buffers(1) and NO drop_on_full: every buffer after the first must
    // wait for the application to pull.
    let sink = AppSink::with_max_buffers(1);
    let sink_handle = sink.handle();
    let src = pipeline.add_source(
        "burst",
        BurstSource {
            produced: 0,
            total: TOTAL,
        },
    );
    let snk = pipeline.add_async_sink("appsink", sink);
    pipeline
        .link_pads_full(src, "src", snk, "sink", LinkPolicy::Block, Some(32))
        .unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let puller = tokio::spawn(async move {
        let mut got = 0u64;
        loop {
            match sink_handle.pull_buffer().await {
                Pulled::Buffer(_) => got += 1,
                Pulled::Ended(reason) => break (got, reason),
                Pulled::Empty | Pulled::Flushing => tokio::task::yield_now().await,
            }
        }
    });

    let (got, reason) = tokio::time::timeout(Duration::from_secs(10), puller)
        .await
        .expect("the sink parked its worker: a full AppSink must await space (#168)")
        .expect("puller task");

    assert_eq!(got, TOTAL, "Block back-pressure must not lose buffers");
    assert_eq!(reason, EndReason::Eos);

    tokio::time::timeout(Duration::from_secs(5), handle.wait())
        .await
        .expect("pipeline shutdown")
        .unwrap();
}
