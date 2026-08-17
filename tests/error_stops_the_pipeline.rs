//! #191: one dead element must wind the whole run down.
//!
//! A fatal element error used to only *record* `EndReason::Error` — no stop
//! flag, no cascade. In a demuxer-rooted pipeline (exactly the player's
//! shape) every sibling task then ran to the media's natural end and
//! `PipelineHandle::wait()` blocked for the whole file. These tests pin the
//! new contract: a fatal error, an [`Error::Shutdown`] request, and a
//! `stop()` while paused all end the run promptly.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    ConsumeContext, Demuxer, DemuxerProduce, PadAddedCallback, PadId, ProduceContext,
    ProduceResult, RoutedOutput, Sink, Source,
};
use parallax::elements::EndReason;
use parallax::error::{Error, Result};
use parallax::format::Caps;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};

const LIMIT: Duration = Duration::from_secs(5);

fn arena() -> &'static SharedArena {
    use std::sync::OnceLock;
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 512).unwrap())
}

fn buffer_with_seq(seq: u64) -> Option<Buffer> {
    arena().reclaim();
    let slot = arena().acquire()?;
    Some(Buffer::new(
        MemoryHandle::with_len(slot, 8),
        Metadata::from_sequence(seq),
    ))
}

/// Source-style demuxer emitting alternately on pads "a" and "b" forever —
/// the player's shape (a demuxer root with two branches, no natural end
/// within the test's lifetime).
struct EndlessDemuxer {
    seq: u64,
    outputs: Vec<(PadId, Caps)>,
}

impl EndlessDemuxer {
    fn new() -> Self {
        Self {
            seq: 0,
            outputs: vec![(PadId(0), Caps::any()), (PadId(1), Caps::any())],
        }
    }
}

impl Demuxer for EndlessDemuxer {
    fn demux(&mut self, _buffer: Buffer) -> Result<RoutedOutput> {
        unreachable!("driven through produce()")
    }

    fn produce(&mut self) -> Result<DemuxerProduce> {
        let Some(buffer) = buffer_with_seq(self.seq) else {
            return Ok(DemuxerProduce::WouldBlock);
        };
        let pad = PadId((self.seq % 2) as u32);
        self.seq += 1;
        Ok(DemuxerProduce::Routed(RoutedOutput::single(pad, buffer)))
    }

    fn pad_name(&self, pad: PadId) -> String {
        match pad.0 {
            0 => "a".into(),
            _ => "b".into(),
        }
    }

    fn outputs(&self) -> &[(PadId, Caps)] {
        &self.outputs
    }

    fn on_pad_added(&mut self, _callback: PadAddedCallback) {}

    fn name(&self) -> &str {
        "endlessdemux"
    }
}

/// Fails fatally on the Nth buffer.
struct FailsAfter {
    remaining: u64,
}

impl Sink for FailsAfter {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        if self.remaining == 0 {
            return Err(Error::Element("sink gave up".into()));
        }
        self.remaining -= 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "fails-after"
    }
}

/// Requests a cooperative shutdown on the Nth buffer (#191) — the window-close
/// shape.
struct ShutsDownAfter {
    remaining: u64,
}

impl Sink for ShutsDownAfter {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        if self.remaining == 0 {
            return Err(Error::Shutdown);
        }
        self.remaining -= 1;
        Ok(())
    }

    fn name(&self) -> &str {
        "shuts-down-after"
    }
}

/// Consumes happily forever.
struct QuietSink;

impl Sink for QuietSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "quiet-sink"
    }
}

/// Produces forever, for the non-demuxer cases.
struct EndlessSource {
    calls: Arc<AtomicU64>,
}

impl Source for EndlessSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        self.calls.fetch_add(1, Ordering::Relaxed);
        match buffer_with_seq(0) {
            Some(buffer) => Ok(ProduceResult::OwnBuffer(buffer)),
            None => Ok(ProduceResult::WouldBlock),
        }
    }

    fn name(&self) -> &str {
        "endless-source"
    }
}

/// The headline regression (#191): a demuxer-rooted pipeline whose video-side
/// sink dies must end promptly — not play the audio branch to the media's
/// natural end. Before the fix, `wait()` here hung until the timeout: the
/// error was recorded but nothing stopped the demuxer or the healthy branch.
#[tokio::test(flavor = "multi_thread")]
async fn a_failing_sink_stops_a_demuxer_rooted_pipeline_promptly() {
    let mut pipeline = Pipeline::new();
    let demux = pipeline.add_demuxer("demux", EndlessDemuxer::new());
    let bad = pipeline.add_sink("bad", FailsAfter { remaining: 3 });
    let good = pipeline.add_sink("good", QuietSink);
    pipeline.link_pads(demux, "a", bad, "sink").unwrap();
    pipeline.link_pads(demux, "b", good, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    let ended = handle.ended();

    let result = tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("a fatal sink error must end the run promptly, not at natural EOS");
    assert!(result.is_err(), "the failure must surface from wait()");

    let reason = tokio::time::timeout(LIMIT, ended).await.unwrap();
    match reason {
        EndReason::Error(e) => assert!(
            e.message().contains("sink gave up"),
            "unexpected error: {e:?}"
        ),
        other => panic!("expected Error, got {other:?}"),
    }
}

/// `Error::Shutdown` (#191) is the window-close contract: the run winds down
/// cleanly and ends as `Eos` — `wait()` succeeds, nothing to string-match.
#[tokio::test(flavor = "multi_thread")]
async fn a_shutdown_request_ends_the_run_as_eos() {
    let calls = Arc::new(AtomicU64::new(0));
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        EndlessSource {
            calls: calls.clone(),
        },
    );
    let snk = pipeline.add_sink("snk", ShutsDownAfter { remaining: 3 });
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    let ended = handle.ended();

    let result = tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("a shutdown request must end the run promptly");
    assert!(result.is_ok(), "shutdown is not a failure: {result:?}");

    let reason = tokio::time::timeout(LIMIT, ended).await.unwrap();
    assert_eq!(reason, EndReason::Eos, "shutdown ends as a clean Eos");
}

/// The same contract on the demuxer-rooted shape — the actual player graph
/// (window close on the video branch, audio branch healthy).
#[tokio::test(flavor = "multi_thread")]
async fn a_shutdown_request_stops_a_demuxer_rooted_pipeline() {
    let mut pipeline = Pipeline::new();
    let demux = pipeline.add_demuxer("demux", EndlessDemuxer::new());
    let closing = pipeline.add_sink("closing", ShutsDownAfter { remaining: 3 });
    let audio = pipeline.add_sink("audio", QuietSink);
    pipeline.link_pads(demux, "a", closing, "sink").unwrap();
    pipeline.link_pads(demux, "b", audio, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    let ended = handle.ended();

    let result = tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("window close must end the run promptly");
    assert!(result.is_ok(), "shutdown is not a failure: {result:?}");
    let reason = tokio::time::timeout(LIMIT, ended).await.unwrap();
    assert_eq!(reason, EndReason::Eos);
}

/// `stop()` on a *paused* pipeline must still complete (#191): the wind-down
/// un-pauses, so the gated loops and frozen-clock pacers can reach their stop
/// checks. Hung forever before the fix.
#[tokio::test(flavor = "multi_thread")]
async fn stop_while_paused_completes() {
    let calls = Arc::new(AtomicU64::new(0));
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        EndlessSource {
            calls: calls.clone(),
        },
    );
    let snk = pipeline.add_sink("snk", QuietSink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Let it flow, then pause with buffers in flight.
    tokio::time::timeout(LIMIT, async {
        while calls.load(Ordering::Relaxed) < 5 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("source never produced");
    handle.pause();
    handle.stop();

    let result = tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("stop() while paused must not hang");
    assert!(result.is_ok(), "{result:?}");
}
