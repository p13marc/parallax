//! Regression tests for cooperative stop (#23): a live (infinite) source has
//! no natural EOS, and `abort()` alone cannot end a produce loop that never
//! reaches an await point. `PipelineHandle::stop()` must end the loop
//! gracefully — EOS reaches the sink and `wait()` returns `Ok` — and
//! `abort()` must also raise the flag so live sources exit.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ProduceContext, ProduceResult, Source};
use parallax::elements::{AppSink, AppSinkHandle, EndReason};
use parallax::error::{Error, Result};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};

/// A live source: produces small buffers forever, never EOS on its own.
/// Counts produce() calls so tests can observe whether the loop still runs.
struct InfiniteSource {
    sequence: u64,
    calls: Arc<AtomicU64>,
    arena: SharedArena,
}

impl InfiniteSource {
    fn new(calls: Arc<AtomicU64>) -> Self {
        Self {
            sequence: 0,
            calls,
            arena: SharedArena::new(64, 256).unwrap(),
        }
    }
}

impl Source for InfiniteSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.arena.reclaim();
        let slot = self
            .arena
            .acquire()
            .ok_or_else(|| Error::Element("arena exhausted".into()))?;
        let buffer = Buffer::new(
            MemoryHandle::with_len(slot, 64),
            Metadata::from_sequence(self.sequence),
        );
        self.sequence += 1;
        Ok(ProduceResult::OwnBuffer(buffer))
    }

    fn name(&self) -> &str {
        "infinite-source"
    }
}

/// Wait for the sink to learn the stream ended, and assert it ended cleanly.
///
/// `end_reason()`, not `ended()`: nobody pulls in these tests, so the queue
/// never drains and `ended()` — which waits for that, so a `select!` consumer
/// loses nothing — would never fire. Knowing *that* it stopped and waiting
/// until everything has been consumed are different questions.
async fn wait_for_eos(handle: &AppSinkHandle, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    let reason = loop {
        if let Some(reason) = handle.end_reason() {
            break reason;
        }
        assert!(
            Instant::now() < deadline,
            "sink never terminated after {what}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert_eq!(
        reason,
        EndReason::Eos,
        "a cooperative stop should read as a clean end of stream"
    );
}

/// Assert the source's produce loop goes quiet within a bound.
async fn wait_for_source_exit(calls: &Arc<AtomicU64>, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    loop {
        let before = calls.load(Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(100)).await;
        if calls.load(Ordering::Relaxed) == before {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "live source still producing after {what}"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_ends_live_source_with_clean_eos() {
    let calls = Arc::new(AtomicU64::new(0));
    // Live-video shape: bounded sink that sheds frames instead of
    // back-pressuring the pipeline (nobody pulls in this test).
    let sink = AppSink::with_max_buffers(4).drop_on_full(true);
    let handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", InfiniteSource::new(calls.clone()));
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    // Let it actually run first.
    let started = Instant::now();
    while calls.load(Ordering::Relaxed) < 3 {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "live source never started producing"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    pipeline_handle.stop();

    wait_for_eos(&handle, "stop()").await;
    assert!(
        pipeline_handle.wait().await.is_ok(),
        "cooperative stop is a clean shutdown, not an error"
    );
    wait_for_source_exit(&calls, "stop()+wait()").await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_also_ends_live_source() {
    let calls = Arc::new(AtomicU64::new(0));
    let sink = AppSink::with_max_buffers(4).drop_on_full(true);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", InfiniteSource::new(calls.clone()));
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    let started = Instant::now();
    while calls.load(Ordering::Relaxed) < 3 {
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "live source never started producing"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    pipeline_handle.abort();

    wait_for_source_exit(&calls, "abort()").await;
}
