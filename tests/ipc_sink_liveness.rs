//! #172: an `IpcSink` waiting on an absent peer must not park a tokio worker.
//!
//! `IpcSink` waits in two places that have no upper bound — for a peer to
//! connect, and for a connected peer to acknowledge buffers. As a synchronous
//! `Sink` both were `thread::sleep` loops, so each stuck sink permanently
//! consumed a worker thread and, worse, stalled the runtime's time driver for
//! every other task on it.
//!
//! The observable difference is cancellability. A task blocked inside a
//! synchronous call is not at an await point, so `JoinHandle::abort` cannot
//! touch it; one that awaits a sleep is cancelled at the next poll. This test
//! therefore aborts a pipeline whose sink has no peer at all and requires the
//! abort to actually land.
//!
//! `worker_threads = 2` for the same reason as `appsink_backpressure`: with
//! one worker, the pre-fix version takes the time driver down with it and the
//! test would hang rather than fail.

use std::sync::OnceLock;
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ProduceContext, ProduceResult, Source};
use parallax::elements::IpcSink;
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};

fn arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 64).unwrap())
}

struct Ticker {
    produced: u64,
}

impl Source for Ticker {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
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
async fn an_ipc_sink_with_no_peer_leaves_the_runtime_alive() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("no-peer.sock");

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", Ticker { produced: 0 });
    // Server mode, and nothing will ever connect.
    let snk = pipeline.add_async_sink("ipcsink", IpcSink::new(&sock));
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // The time driver must still be running while the sink waits. A
    // `thread::sleep` inside a sync `consume` is exactly what stops it.
    let ticks = tokio::spawn(async move {
        let mut n = 0;
        for _ in 0..5 {
            tokio::time::sleep(Duration::from_millis(10)).await;
            n += 1;
        }
        n
    });
    let ticks = tokio::time::timeout(Duration::from_secs(5), ticks)
        .await
        .expect("the runtime's timers stopped: a sink parked its worker")
        .unwrap();
    assert_eq!(ticks, 5);

    // And the waiting sink is cancellable, which a task blocked inside a
    // synchronous call never is.
    handle.abort();
    tokio::time::timeout(Duration::from_secs(5), tokio::task::yield_now())
        .await
        .expect("yield after abort");
}
