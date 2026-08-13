//! Regression tests for #88: the pipeline's terminal outcome must be reachable
//! by someone who does not consume the handle.
//!
//! `PipelineEvent::Eos` was emitted in exactly one place — the last line of
//! `PipelineHandle::wait()` — and `wait()` takes the handle by value. So the
//! caller who keeps the handle to *control* the pipeline, which is precisely
//! the caller who wants to know why it stopped, could never find out. The event
//! channel could not fill the gap: it is a broadcast, so a terminal event sent
//! before you subscribed is simply gone.
//!
//! `ended()` answers instead, and it retains the answer.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    ConsumeContext, Element, ExecutionHints, ProduceContext, ProduceResult, Sink, Source,
};
use parallax::elements::{AppSink, NullSink, NullSource, PassThrough};
use parallax::error::{Error, Result};
use parallax::event::Event;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{EndReason, Executor, ExecutorConfig, Pipeline, PipelineHandle};

const LIMIT: Duration = Duration::from_secs(10);

struct FailingSource;

impl Source for FailingSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        Err(Error::Element("simulated device failure".into()))
    }

    fn name(&self) -> &str {
        "failing-source"
    }
}

struct PanickingTransform;

impl Element for PanickingTransform {
    fn process(&mut self, _buffer: Buffer) -> Result<Option<Buffer>> {
        panic!("encoder hit an impossible state");
    }
}

/// A live source: produces forever, never EOS on its own.
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

/// An RT-safe source and sink, so a hybrid partition can claim the whole graph
/// and leave the async side — and therefore the live task count — empty.
struct RtSource {
    remaining: u64,
    sequence: u64,
    arena: SharedArena,
}

impl RtSource {
    fn new(count: u64) -> Self {
        Self {
            remaining: count,
            sequence: 0,
            arena: SharedArena::new(64, 256).unwrap(),
        }
    }
}

impl Source for RtSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.remaining == 0 {
            return Ok(ProduceResult::Eos);
        }
        self.remaining -= 1;
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
        "rt-source"
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::rt_safe()
    }
}

/// Panics on the terminal event, which the sink task calls *outside* the
/// `catch_unwind` that wraps `consume()`.
struct PanickingOnEosSink;

impl Sink for PanickingOnEosSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "panicking-on-eos-sink"
    }

    fn handle_downstream_event(&mut self, _event: Event) -> Option<Event> {
        panic!("teardown fell over");
    }
}

struct RtSink;

impl Sink for RtSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "rt-sink"
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::rt_safe()
    }
}

async fn ended(handle: &PipelineHandle, what: &str) -> EndReason {
    tokio::time::timeout(LIMIT, handle.ended())
        .await
        .unwrap_or_else(|_| panic!("the pipeline never reported an outcome: {what}"))
}

/// A live pipeline whose sink never blocks the source.
fn live_pipeline(calls: Arc<AtomicU64>) -> Pipeline {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", InfiniteSource::new(calls));
    let snk = pipeline.add_async_sink("sink", AppSink::with_max_buffers(4).drop_on_full(true));
    pipeline.link(src, snk).unwrap();
    pipeline
}

/// The test that catches a missing live-task counter: nothing here ever calls
/// `wait()`, which is the only thing that used to report EOS.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn eos_without_ever_calling_wait() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let snk = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, snk).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();

    assert_eq!(ended(&handle, "a five-buffer run").await, EndReason::Eos);
}

/// One chain fails while a much longer one is still running. The failure is
/// recorded before the failing task releases its share of the live count, so
/// the outcome cannot be overwritten by whichever chain finishes last — no
/// sleeps needed to make that deterministic.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn error_wins_over_eos() {
    let mut pipeline = Pipeline::new();

    let failing = pipeline.add_source("src_fail", FailingSource);
    let dead_end = pipeline.add_sink("sink_fail", NullSink::new());
    pipeline.link(failing, dead_end).unwrap();

    let long = pipeline.add_source("src_long", NullSource::new(1000));
    let long_sink = pipeline.add_sink("sink_long", NullSink::new());
    pipeline.link(long, long_sink).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();

    match ended(&handle, "one failing chain of two").await {
        EndReason::Error(err) => {
            assert_eq!(err.node(), Some("src_fail"));
            assert!(
                err.message().contains("simulated device failure"),
                "unexpected message: {err}"
            );
        }
        other => panic!("a failed chain reported {other:?}"),
    }
}

/// `Ended` is owned precisely so it can be created before the handle is
/// consumed and awaited afterwards — `abort()` also takes `self`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_reports_aborted() {
    let calls = Arc::new(AtomicU64::new(0));
    let mut pipeline = live_pipeline(calls.clone());
    let handle = Executor::new().start(&mut pipeline).unwrap();

    let ended = handle.ended();
    while calls.load(Ordering::Relaxed) == 0 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    handle.abort();

    let reason = tokio::time::timeout(LIMIT, ended)
        .await
        .expect("abort never reported an outcome");
    assert_eq!(reason, EndReason::Aborted);
}

/// The watch-not-broadcast property. A broadcast receiver created after the
/// terminal event was sent waits forever; this one must answer immediately.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_late_observer_still_sees_the_outcome() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(3));
    let snk = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, snk).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();

    // Let the pipeline finish, and be seen to finish, before subscribing.
    while handle.end_reason().is_none() {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    tokio::time::sleep(Duration::from_millis(100)).await;

    let reason = tokio::time::timeout(Duration::from_millis(500), handle.ended())
        .await
        .expect("a late observer waited for an outcome that had already been decided");
    assert_eq!(reason, EndReason::Eos);
}

/// `stop()` is the graceful path: the sources end their loop, EOS flows
/// downstream as usual, and the run reports a clean end — not an abort.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stop_is_a_clean_end_not_an_abort() {
    let calls = Arc::new(AtomicU64::new(0));
    let mut pipeline = live_pipeline(calls.clone());
    let handle = Executor::new().start(&mut pipeline).unwrap();

    while calls.load(Ordering::Relaxed) < 4 {
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    handle.stop();

    assert_eq!(
        ended(&handle, "a stopped live source").await,
        EndReason::Eos
    );
}

/// A panicking element is a failure like any other: #85 turned it into an
/// `Err` at the call site, so it takes the ordinary error path here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panicking_element_ends_the_pipeline_with_an_error() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let xfm = pipeline.add_filter("xfm", PanickingTransform);
    let snk = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    let ended = handle.ended();

    let waited = tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("a panicking element wedged the pipeline");
    assert!(waited.is_err(), "a panic must not report success");

    let reason = tokio::time::timeout(LIMIT, ended)
        .await
        .expect("the panic never reached the outcome");
    assert!(
        matches!(reason, EndReason::Error(_)),
        "a panicking element reported {reason:?}"
    );
}

/// Not every element call sits inside the `catch_unwind` that turns a panic into
/// an ordinary error: the sink task's `handle_downstream_event` is called on the
/// terminal path and its result is discarded, so a panic there unwinds the whole
/// task instead. The share of the live count is released by `Drop`, which runs
/// during unwinding too — so without the `thread::panicking()` check the task
/// would count as a clean finish and this run would report EOS to `ended()`
/// while `wait()` returned `Error::Panic`. Two observers, two answers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panic_outside_the_guarded_calls_is_still_an_error() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let snk = pipeline.add_sink("sink", PanickingOnEosSink);
    pipeline.link(src, snk).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    let ended = handle.ended();

    let waited = tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("the panicking sink wedged the pipeline");
    assert!(waited.is_err(), "a panicking sink must not report success");

    let reason = tokio::time::timeout(LIMIT, ended)
        .await
        .expect("the panic never reached the outcome");
    assert!(
        matches!(reason, EndReason::Error(_)),
        "a panic outside the guarded calls reported {reason:?}"
    );
}

/// Every node RT-safe, so the async side of the graph is empty. The live count
/// would hit zero the moment `start()` released the seed, declaring EOS before
/// the RT thread had run a single cycle — which is why the seed is *retained*
/// whenever there are RT threads and only released once `wait()` has joined
/// them. They loop until told to stop, so they can never be counted like tasks.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_all_rt_graph_does_not_report_eos_before_it_runs() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", RtSource::new(20));
    let xfm = pipeline.add_filter("rt", PassThrough::new());
    let snk = pipeline.add_sink("sink", RtSink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::with_config(ExecutorConfig::hybrid());
    let handle = executor.start(&mut pipeline).unwrap();

    assert_eq!(
        handle.end_reason(),
        None,
        "an all-RT graph reported an outcome before it had run"
    );

    tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("the RT threads were never joined")
        .expect("a clean all-RT run reported an error");
}

/// Hybrid mode holds the seed share of the live count until the RT threads are
/// joined, so the answer arrives with `wait()` rather than before it — the
/// shape the docs promise.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hybrid_reports_eos_once_the_rt_threads_are_joined() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(20));
    let xfm = pipeline.add_filter("rt", PassThrough::new());
    let snk = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::with_config(ExecutorConfig::hybrid());
    let handle = executor.start(&mut pipeline).unwrap();
    let ended = handle.ended();

    tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("the hybrid pipeline never finished")
        .expect("a clean hybrid run reported an error");

    let reason = tokio::time::timeout(Duration::from_millis(500), ended)
        .await
        .expect("the seed was never released after the RT join");
    assert_eq!(reason, EndReason::Eos);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ended_can_be_awaited_after_wait_consumed_the_handle() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let snk = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, snk).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    let ended = handle.ended();

    tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("the pipeline never finished")
        .expect("a clean run reported an error");

    let reason = tokio::time::timeout(Duration::from_millis(500), ended)
        .await
        .expect("the outcome was not retained past wait()");
    assert_eq!(reason, EndReason::Eos);
}
