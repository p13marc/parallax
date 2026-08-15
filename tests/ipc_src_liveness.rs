//! #179: an `IpcSrc` waiting on an absent peer must not park a tokio worker.
//!
//! The pre-#179 `IpcSrc` was a sync `Source` that, once connected, blocked
//! a worker inside a socket `read_exact` — the #172 bug class, unfixed on
//! the source side. As an `AsyncSource` its waits are awaits: bounded
//! doorbell waits and `WouldBlock` polls, so the runtime's time driver
//! keeps running and `abort` lands.

use std::time::Duration;

use parallax::element::{AsyncSink, ConsumeContext};
use parallax::elements::IpcSrc;
use parallax::error::Result;
use parallax::pipeline::{Executor, Pipeline};

struct CountingSink;

impl AsyncSink for CountingSink {
    async fn consume(&mut self, _ctx: &ConsumeContext<'_>) -> Result<()> {
        Ok(())
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_ipc_src_with_no_peer_leaves_the_runtime_alive() {
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("no-peer.sock");

    let mut pipeline = Pipeline::new();
    // Server mode, and nothing will ever connect.
    let src = pipeline.add_async_source("ipcsrc", IpcSrc::listen(&sock));
    let snk = pipeline.add_async_sink("sink", CountingSink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // The time driver must still be running while the source polls.
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
        .expect("the runtime's timers stopped: a source parked its worker")
        .unwrap();
    assert_eq!(ticks, 5);

    // And the waiting source is cancellable.
    handle.abort();
    tokio::time::timeout(Duration::from_secs(5), tokio::task::yield_now())
        .await
        .expect("yield after abort");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_src_retries_until_the_socket_exists() {
    // Client mode against a path nobody has bound: the old code errored
    // out of produce; now it polls (start-order independence).
    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("never-bound.sock");

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_async_source("ipcsrc", IpcSrc::new(&sock));
    let snk = pipeline.add_async_sink("sink", CountingSink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Give it time to hit the missing socket repeatedly; the pipeline must
    // still be alive (an error would have ended it).
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        handle.end_reason().is_none(),
        "connect retry must not error"
    );

    handle.abort();
    tokio::time::timeout(Duration::from_secs(5), tokio::task::yield_now())
        .await
        .expect("yield after abort");
}
