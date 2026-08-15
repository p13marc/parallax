//! #163 phase C: flush scoping follows the seek's path, not the pipeline.
//!
//! The flush epoch used to be one pipeline-global counter: a flushing seek
//! handled on chain A raised it for everyone, and chain B's in-flight
//! buffers — stamped before the bump, owed no flush — were silently dropped
//! at B's sink with no FlushStop and no re-anchoring Segment. Since phase C
//! every producing node has its own epoch cell, consumers judge staleness
//! against the cell of the branch a buffer arrived on, and the shed
//! propagates hop-by-hop exactly along the seek's path.
//!
//! The rigs pull continuously from background tasks: sources produce at
//! full speed, so the link channels stay full and there are always
//! in-flight pre-seek-stamped buffers when the seek lands — which is
//! exactly what the pre-fix global bump destroyed on innocent chains.
//! (Pulling must not stop around the seek: a sink blocked inside
//! `consume()` on a full AppSink never reaches the loop head that services
//! its upstream inbox, and the seek would stall, not propagate.)
//!
//! Also covers #183: fan-in into a non-muxer is rejected at link time
//! instead of silently dropping the extra branch.

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{
    Muxer, MuxerInput, PadAddedCallback, PadId, ProduceContext, ProduceResult, Source,
};
use parallax::elements::{AppSink, AppSinkHandle, PassThrough, Pulled};
use parallax::error::Result;
use parallax::event::{Event, EventResult};
use parallax::format::Caps;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

fn arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 512).unwrap())
}

/// A seekable source whose position jumps on a handled seek; PTS exposes
/// the jump so tests can tell pre-seek buffers from post-seek ones.
struct SeekJumpSource {
    pos_ms: u64,
    seeks: Arc<AtomicU64>,
    stream_id: u32,
}

impl Source for SeekJumpSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        arena().reclaim();
        let Some(slot) = arena().acquire() else {
            return Ok(ProduceResult::WouldBlock);
        };
        self.pos_ms += 10;
        let mut meta = Metadata::from_sequence(self.pos_ms / 10);
        meta.pts = ClockTime::from_millis(self.pos_ms);
        meta.stream_id = self.stream_id;
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            meta,
        )))
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        if let Event::Seek(seek) = event {
            self.pos_ms = (seek.start.position.max(0) as u64) / 1_000_000; // ns → ms
            self.seeks.fetch_add(1, Ordering::SeqCst);
            EventResult::handled_at(seek.start.position)
        } else {
            EventResult::NotHandled
        }
    }
}

/// An unseekable source producing a gapless sequence — the innocent chain.
/// Any epoch-dropped buffer shows up as a hole in the sequence numbers.
struct PlainSource {
    produced: u64,
    stream_id: u32,
}

impl Source for PlainSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        arena().reclaim();
        let Some(slot) = arena().acquire() else {
            return Ok(ProduceResult::WouldBlock);
        };
        self.produced += 1;
        let mut meta = Metadata::from_sequence(self.produced);
        meta.stream_id = self.stream_id;
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            meta,
        )))
    }
}

/// A minimal N-to-1 muxer that forwards whatever it is handed, recording
/// the quiet stream's sequences as `mux()` receives them — the per-branch
/// guarantee is at the muxer INPUT (below the merge point the muxed stream
/// is one stream, and a flush there legitimately sheds interleaved data).
struct PassMux {
    inputs: Vec<(PadId, Caps)>,
    seen_quiet: Arc<Mutex<Vec<u64>>>,
}

impl Muxer for PassMux {
    fn mux(&mut self, input: MuxerInput) -> Result<Option<Buffer>> {
        if input.buffer.metadata().stream_id == 2 {
            self.seen_quiet
                .lock()
                .unwrap()
                .push(input.buffer.metadata().sequence);
        }
        Ok(Some(input.buffer))
    }
    fn name(&self) -> &str {
        "passmux"
    }
    fn inputs(&self) -> &[(PadId, Caps)] {
        &self.inputs
    }
    fn on_pad_added(&mut self, _callback: PadAddedCallback) {}
}

type Collected = Arc<Mutex<Vec<Metadata>>>;

/// Continuously pull a sink into a shared vec until the sink ends.
fn spawn_puller(handle: AppSinkHandle) -> Collected {
    let collected: Collected = Arc::new(Mutex::new(Vec::new()));
    let out = collected.clone();
    tokio::spawn(async move {
        loop {
            match handle.pull_buffer().await {
                Pulled::Buffer(b) => out.lock().unwrap().push(b.metadata().clone()),
                Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
                _ => break,
            }
        }
    });
    collected
}

async fn wait_until(mut cond: impl FnMut() -> bool, what: &str) {
    for _ in 0..2000 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for {what}");
}

fn assert_gapless(seqs: &[u64], what: &str) {
    for w in seqs.windows(2) {
        assert_eq!(
            w[1],
            w[0] + 1,
            "{what}: sequence gap {} -> {} — an innocent chain's buffers were epoch-dropped",
            w[0],
            w[1]
        );
    }
}

/// The phase-C headline: a flushing seek on chain A must not shed chain B's
/// in-flight buffers. Failed on the pipeline-global epoch (B's queued
/// backlog was dropped at its sink with no re-anchoring Segment).
#[tokio::test(flavor = "multi_thread")]
async fn a_seek_on_one_chain_never_drops_the_other_chains_buffers() {
    let seeks = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src_a = pipeline.add_source(
        "seekable",
        SeekJumpSource {
            pos_ms: 0,
            seeks: seeks.clone(),
            stream_id: 1,
        },
    );
    let sink_a = AppSink::new();
    let handle_a = sink_a.handle();
    let a = pipeline.add_async_sink("sink_a", sink_a);
    pipeline.link(src_a, a).unwrap();

    let src_b = pipeline.add_source(
        "plain",
        PlainSource {
            produced: 0,
            stream_id: 2,
        },
    );
    let sink_b = AppSink::new();
    let handle_b = sink_b.handle();
    let b = pipeline.add_async_sink("sink_b", sink_b);
    pipeline.link(src_b, b).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    let got_a = spawn_puller(handle_a);
    let got_b = spawn_puller(handle_b);

    // Warm up: both chains flowing, channels full of in-flight buffers.
    wait_until(|| got_b.lock().unwrap().len() >= 10, "chain B warmup").await;

    let before = got_b.lock().unwrap().len();
    assert!(handle.seek_time(ClockTime::from_secs(100)).await);
    wait_until(|| seeks.load(Ordering::SeqCst) > 0, "seek to reach A").await;

    // Chain B keeps flowing and its stream stays gapless across the seek.
    wait_until(
        || got_b.lock().unwrap().len() >= before + 30,
        "chain B to keep flowing",
    )
    .await;
    // Chain A genuinely sought.
    wait_until(
        || {
            got_a
                .lock()
                .unwrap()
                .iter()
                .any(|m| m.pts >= ClockTime::from_secs(100))
        },
        "chain A to reach its seek target",
    )
    .await;
    handle.abort();

    let seqs: Vec<u64> = got_b.lock().unwrap().iter().map(|m| m.sequence).collect();
    assert!(seqs.len() >= 40);
    assert_gapless(&seqs, "chain B");
}

/// Fan-out from one source: both branches share the source's epoch cell and
/// both receive the flush trio — a seek still sheds the backlog on both.
#[tokio::test(flavor = "multi_thread")]
async fn a_diamond_fanout_seek_still_sheds_both_branches() {
    let seeks = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "seekable",
        SeekJumpSource {
            pos_ms: 0,
            seeks: seeks.clone(),
            stream_id: 1,
        },
    );
    let sink1 = AppSink::new();
    let h1 = sink1.handle();
    let s1 = pipeline.add_async_sink("s1", sink1);
    let sink2 = AppSink::new();
    let h2 = sink2.handle();
    let s2 = pipeline.add_async_sink("s2", sink2);
    pipeline.link(src, s1).unwrap();
    pipeline.link(src, s2).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    let got1 = spawn_puller(h1);
    let got2 = spawn_puller(h2);

    wait_until(|| got1.lock().unwrap().len() >= 5, "warmup").await;

    let target = ClockTime::from_secs(100);
    let mark1 = got1.lock().unwrap().len();
    let mark2 = got2.lock().unwrap().len();
    assert!(handle.seek_time(target).await);
    wait_until(
        || seeks.load(Ordering::SeqCst) > 0,
        "seek to reach the source",
    )
    .await;

    // Both branches reach post-target PTS...
    for (name, got) in [("s1", &got1), ("s2", &got2)] {
        wait_until(|| got.lock().unwrap().iter().any(|m| m.pts >= target), name).await;
    }
    handle.abort();

    // ...and the flush ordering held on both: once a post-target buffer
    // appears, no pre-target buffer may follow it — a branch that missed
    // the shed would interleave stale backlog after the jump.
    let _ = (mark1, mark2);
    for (name, got) in [("s1", &got1), ("s2", &got2)] {
        let got = got.lock().unwrap();
        if let Some(first_post) = got.iter().position(|m| m.pts >= target) {
            assert!(
                got[first_post..].iter().all(|m| m.pts >= target),
                "{name}: pre-seek buffers delivered after the jump — backlog not shed"
            );
        }
    }
}

/// A muxer with a seeking input and a quiet input: the quiet branch's
/// buffers must survive the sibling's seek (per-branch staleness), while
/// the seeking branch still sheds and jumps.
#[tokio::test(flavor = "multi_thread")]
async fn a_muxer_keeps_the_quiet_branchs_buffers_across_a_sibling_seek() {
    let seeks = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src_a = pipeline.add_source(
        "seekable",
        SeekJumpSource {
            pos_ms: 0,
            seeks: seeks.clone(),
            stream_id: 1,
        },
    );
    let src_b = pipeline.add_source(
        "plain",
        PlainSource {
            produced: 0,
            stream_id: 2,
        },
    );
    let seen_quiet: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let mux = pipeline.add_muxer(
        "mux",
        PassMux {
            inputs: Vec::new(),
            seen_quiet: seen_quiet.clone(),
        },
    );
    let sink = AppSink::new();
    let handle_s = sink.handle();
    let s = pipeline.add_async_sink("sink", sink);
    pipeline.link(src_a, mux).unwrap();
    pipeline.link(src_b, mux).unwrap();
    pipeline.link(mux, s).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    let got = spawn_puller(handle_s);

    wait_until(
        || {
            got.lock()
                .unwrap()
                .iter()
                .filter(|m| m.stream_id == 2)
                .count()
                >= 10
        },
        "quiet branch warmup",
    )
    .await;

    assert!(handle.seek_time(ClockTime::from_secs(100)).await);
    wait_until(|| seeks.load(Ordering::SeqCst) > 0, "seek to reach A").await;

    // The seeking branch jumps...
    wait_until(
        || {
            got.lock()
                .unwrap()
                .iter()
                .any(|m| m.stream_id == 1 && m.pts >= ClockTime::from_secs(100))
        },
        "seeking branch to reach its target through the muxer",
    )
    .await;
    // ...while the quiet branch keeps flowing.
    let quiet_now = got
        .lock()
        .unwrap()
        .iter()
        .filter(|m| m.stream_id == 2)
        .count();
    wait_until(
        || {
            got.lock()
                .unwrap()
                .iter()
                .filter(|m| m.stream_id == 2)
                .count()
                >= quiet_now + 20
        },
        "quiet branch to keep flowing",
    )
    .await;
    handle.abort();

    // The guarantee lives at the muxer input: every one of the quiet
    // branch's buffers reached mux() — none were epoch-dropped by the
    // sibling's seek. (Downstream of the merge the muxed stream is one
    // stream; the forwarded flush legitimately sheds interleaved data
    // there, so end-to-end gaplessness is NOT the invariant.)
    let b_seqs = seen_quiet.lock().unwrap();
    assert!(b_seqs.len() >= 30, "muxer starved the quiet branch");
    assert_gapless(&b_seqs, "quiet branch at the muxer input");
}

/// #183: fan-in into anything but a muxer is a link-time error instead of a
/// silently dropped branch.
#[test]
fn fan_in_into_a_non_muxer_is_rejected() {
    // Into a sink: rejected.
    let mut p = Pipeline::new();
    let a = p.add_source(
        "a",
        PlainSource {
            produced: 0,
            stream_id: 1,
        },
    );
    let b = p.add_source(
        "b",
        PlainSource {
            produced: 0,
            stream_id: 2,
        },
    );
    let sink = p.add_async_sink("sink", AppSink::new());
    p.link(a, sink).unwrap();
    let err = p.link(b, sink).unwrap_err();
    assert!(
        err.to_string().contains("only muxers"),
        "unexpected error: {err}"
    );

    // Into a transform: rejected.
    let mut p = Pipeline::new();
    let a = p.add_source(
        "a",
        PlainSource {
            produced: 0,
            stream_id: 1,
        },
    );
    let b = p.add_source(
        "b",
        PlainSource {
            produced: 0,
            stream_id: 2,
        },
    );
    let f = p.add_filter("f", PassThrough::new());
    p.link(a, f).unwrap();
    assert!(p.link(b, f).is_err());

    // Into a muxer: fine.
    let mut p = Pipeline::new();
    let a = p.add_source(
        "a",
        PlainSource {
            produced: 0,
            stream_id: 1,
        },
    );
    let b = p.add_source(
        "b",
        PlainSource {
            produced: 0,
            stream_id: 2,
        },
    );
    let m = p.add_muxer(
        "m",
        PassMux {
            inputs: Vec::new(),
            seen_quiet: Arc::new(Mutex::new(Vec::new())),
        },
    );
    p.link(a, m).unwrap();
    p.link(b, m).unwrap();
}
