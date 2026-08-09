//! #85: no element can fail without telling the graph.
//!
//! Error propagation used to exist in two of six task exits. A sink, demuxer or
//! muxer that failed returned straight out of its task, so everything below it
//! waited forever — and a source whose only sink had died did something worse
//! than hang: `OutputBranch::send_buffer` discarded its result, and kanal
//! reports a closed channel *immediately* rather than blocking, so the source
//! spun at 100% CPU producing into nothing.
//!
//! Every test here is wrapped in a timeout, so a regression fails the suite
//! instead of hanging CI until it is killed.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    ConsumeContext, Muxer, MuxerInput, PadAddedCallback, PadId, ProduceContext, ProduceResult,
    Sink, Source,
};
use parallax::elements::{AppSink, EndReason, NullSource};
use parallax::error::{Error, Result};
use parallax::format::Caps;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};

const LIMIT: Duration = Duration::from_secs(10);

fn arena() -> &'static SharedArena {
    use std::sync::OnceLock;
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 512).unwrap())
}

/// Produces forever, counting calls so a test can watch the loop go quiet.
struct CountingSource {
    calls: Arc<AtomicU64>,
}

impl Source for CountingSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        arena().reclaim();
        let Some(slot) = arena().acquire() else {
            return Ok(ProduceResult::WouldBlock);
        };
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            Metadata::from_sequence(0),
        )))
    }

    fn name(&self) -> &str {
        "counting-source"
    }
}

struct FailingSink;

impl Sink for FailingSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        Err(Error::Element("sink gave up".into()))
    }

    fn name(&self) -> &str {
        "failing-sink"
    }
}

struct FailingMuxer {
    inputs: Vec<(PadId, Caps)>,
}

impl FailingMuxer {
    fn new() -> Self {
        Self {
            inputs: vec![(PadId::new(0), Caps::any()), (PadId::new(1), Caps::any())],
        }
    }
}

impl Muxer for FailingMuxer {
    fn mux(&mut self, _input: MuxerInput) -> Result<Option<Buffer>> {
        Err(Error::Element("muxer gave up".into()))
    }

    fn name(&self) -> &str {
        "failing-muxer"
    }

    fn inputs(&self) -> &[(PadId, Caps)] {
        &self.inputs
    }

    fn on_pad_added(&mut self, _callback: PadAddedCallback) {}
}

/// The livelock. A source whose only consumer has died must stop, not spin.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dead_sink_stops_the_source_instead_of_spinning() {
    let calls = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        CountingSource {
            calls: Arc::clone(&calls),
        },
    );
    let snk = pipeline.add_sink("sink", FailingSink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let result = tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("pipeline hung after the sink failed");
    assert!(result.is_err(), "the sink error should surface");

    // The real assertion: the produce loop has actually stopped. Before the
    // fix this counter climbed for as long as the handle existed.
    let before = calls.load(Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        calls.load(Ordering::Relaxed),
        before,
        "the source is still producing into a closed channel"
    );
}

/// A muxer that fails must tell its output, not just return.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failing_muxer_terminates_its_sink() {
    let sink = AppSink::new();
    let handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let a = pipeline.add_source("a", NullSource::new(4));
    let b = pipeline.add_source("b", NullSource::new(4));
    let mux = pipeline.add_muxer("mux", FailingMuxer::new());
    let snk = pipeline.add_sink("sink", sink);
    // Muxer nodes start with a single "sink" pad; the executor keeps a
    // receiver per link, so both sources feed it.
    pipeline.link(a, mux).unwrap();
    pipeline.link(b, mux).unwrap();
    pipeline.link(mux, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    let reason = tokio::time::timeout(LIMIT, handle.ended())
        .await
        .expect("the sink below a failed muxer never terminated");
    match reason {
        EndReason::Error(err) => assert!(
            err.message().contains("muxer gave up"),
            "wrong reason: {err}"
        ),
        other => panic!("a failed muxer must not report a clean end: {other:?}"),
    }

    let _ = tokio::time::timeout(LIMIT, pipeline_handle.wait())
        .await
        .expect("pipeline hung on teardown");
}

/// A sink whose element fails still has to end its own consumer's wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failing_pipeline_does_not_leave_wait_hanging() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(8));
    let snk = pipeline.add_sink("sink", FailingSink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let result = tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("wait() hung on a failed sink");
    assert!(result.is_err());
}
