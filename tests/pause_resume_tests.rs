//! Integration tests for runtime pause/resume/position (#71).

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::elements::{AppSrc, NullSink, NullSource};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::probe::{PadRef, ProbeReturn, ProbeType};
use parallax::pipeline::{Executor, Pipeline, PipelineState};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

fn arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 512).unwrap())
}

fn buffer_with_pts(pts_ns: u64) -> Buffer {
    // Endless producers (CountingDemux) cycle far more buffers than the
    // arena has slots; reclaim returns the released ones first.
    arena().reclaim();
    let slot = arena().acquire().expect("test arena exhausted");
    let mut metadata = Metadata::new();
    metadata.pts = ClockTime::from_nanos(pts_ns);
    Buffer::new(MemoryHandle::with_len(slot, 8), metadata)
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

/// Pause stops delivery; resume restarts it; the counters prove both.
#[tokio::test(flavor = "multi_thread")]
async fn pause_stops_the_stream_and_resume_restarts_it() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(u64::MAX));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let delivered = Arc::new(AtomicU64::new(0));
    let delivered_probe = delivered.clone();
    let _ = pipeline.add_probe(PadRef::sink(sink), ProbeType::BUFFER, move |_| {
        delivered_probe.fetch_add(1, Ordering::Relaxed);
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    wait_until(|| delivered.load(Ordering::Relaxed) > 0, "first delivery").await;

    handle.pause();
    assert!(handle.is_paused());
    // Let the gated source's in-flight buffers drain, then the count must
    // hold still.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let frozen = delivered.load(Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        delivered.load(Ordering::Relaxed),
        frozen,
        "buffers were delivered while paused"
    );

    handle.resume();
    assert!(!handle.is_paused());
    wait_until(
        || delivered.load(Ordering::Relaxed) > frozen,
        "delivery after resume",
    )
    .await;

    handle.stop();
    handle.wait().await.unwrap();
}

/// Pause/resume post the matching StateChanged transitions on the bus.
#[tokio::test(flavor = "multi_thread")]
async fn pause_and_resume_post_state_changes() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(u64::MAX));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    handle.pause();
    handle.pause(); // idempotent: must not post a second transition
    handle.resume();

    handle.stop();
    handle.wait().await.unwrap();

    let mut transitions = Vec::new();
    while let Some(msg) = bus.poll() {
        if let MessageKind::StateChanged { old, new } = msg.kind {
            transitions.push((old, new));
        }
    }
    // Startup itself posts Suspended→Idle and Idle→Running, so anchor on the
    // pause transition and count from there.
    let paused = (PipelineState::Running, PipelineState::Idle);
    let resumed = (PipelineState::Idle, PipelineState::Running);
    assert_eq!(
        transitions.iter().filter(|t| **t == paused).count(),
        1,
        "exactly one pause transition (second pause() was a no-op): {transitions:?}"
    );
    let pause_at = transitions.iter().position(|t| *t == paused).unwrap();
    assert_eq!(
        transitions[pause_at + 1..]
            .iter()
            .filter(|t| **t == resumed)
            .count(),
        1,
        "exactly one resume transition after the pause: {transitions:?}"
    );
}

/// Pause gates a *source-style demuxer* — the topology where it used to gate
/// nothing at all (#156): the player's only producer is `add_demuxer`, and
/// `spawn_demuxer_task` had no pause_rx.
#[tokio::test(flavor = "multi_thread")]
async fn pause_gates_a_source_style_demuxer() {
    use parallax::element::{Demuxer, DemuxerProduce, PadAddedCallback, PadId, RoutedOutput};
    use parallax::format::Caps;

    /// Endless one-pad demuxer that owns its "reader" (a counter).
    struct CountingDemux {
        seq: u64,
        outputs: Vec<(PadId, Caps)>,
    }

    impl Demuxer for CountingDemux {
        fn demux(&mut self, _buffer: Buffer) -> parallax::error::Result<RoutedOutput> {
            unreachable!("source-style: driven through produce()")
        }

        fn produce(&mut self) -> parallax::error::Result<DemuxerProduce> {
            let buffer = buffer_with_pts(self.seq * 1_000);
            self.seq += 1;
            Ok(DemuxerProduce::Routed(RoutedOutput::single(
                PadId(0),
                buffer,
            )))
        }

        fn pad_name(&self, _pad: PadId) -> String {
            "data".into()
        }

        fn outputs(&self) -> &[(PadId, Caps)] {
            &self.outputs
        }

        fn on_pad_added(&mut self, _callback: PadAddedCallback) {}
    }

    let mut pipeline = Pipeline::new();
    let node = pipeline.add_demuxer(
        "demux",
        CountingDemux {
            seq: 0,
            outputs: vec![(PadId(0), Caps::any())],
        },
    );
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link_pads(node, "data", sink, "sink").unwrap();

    let delivered = Arc::new(AtomicU64::new(0));
    let delivered_probe = delivered.clone();
    let _ = pipeline.add_probe(PadRef::sink(sink), ProbeType::BUFFER, move |_| {
        delivered_probe.fetch_add(1, Ordering::Relaxed);
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    wait_until(|| delivered.load(Ordering::Relaxed) > 0, "first delivery").await;

    handle.pause();
    // Drain the in-flight tail, then the count must hold still — before the
    // fix the demuxer kept producing and this grew by thousands.
    tokio::time::sleep(Duration::from_millis(100)).await;
    let frozen = delivered.load(Ordering::Relaxed);
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        delivered.load(Ordering::Relaxed),
        frozen,
        "a paused source-style demuxer kept delivering"
    );

    handle.resume();
    wait_until(
        || delivered.load(Ordering::Relaxed) > frozen,
        "delivery after resume",
    )
    .await;

    handle.stop();
    handle.wait().await.unwrap();
}

/// A paused sink delivers Pause/Resume to its element exactly once per
/// transition, holds (not drops) the buffer it stashed while paused, and
/// replays everything in order on resume.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_paused_sink_delivers_pause_resume_and_replays_the_stash() {
    use parallax::element::{ConsumeContext, Sink};
    use parallax::event::Event;
    use std::sync::Mutex;

    /// Records every consume and every downstream event, in arrival order.
    struct LoggingSink {
        log: Arc<Mutex<Vec<String>>>,
        first: bool,
    }

    impl Sink for LoggingSink {
        fn consume(&mut self, ctx: &ConsumeContext) -> parallax::error::Result<()> {
            let pts = ctx.metadata().pts.nanos();
            self.log.lock().unwrap().push(format!("buf:{pts}"));
            // The first consume is long enough that the test's pause() —
            // issued a few ms after "buf:1000" appears — deterministically
            // lands while this buffer is in flight and the rest of the burst
            // is still queued. If the sink went idle first, Pause delivery
            // would defer to the next message (the documented caveat) and
            // the test would race.
            if self.first {
                self.first = false;
                std::thread::sleep(Duration::from_millis(150));
            } else {
                std::thread::sleep(Duration::from_millis(10));
            }
            Ok(())
        }

        fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
            self.log
                .lock()
                .unwrap()
                .push(format!("ev:{}", event.name()));
            Some(event)
        }
    }

    let log = Arc::new(Mutex::new(Vec::new()));
    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let sink = pipeline.add_sink(
        "sink",
        LoggingSink {
            log: log.clone(),
            first: true,
        },
    );
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // A burst of five; the sink is chewing on the first when pause() lands.
    for pts in [1_000u64, 2_000, 3_000, 4_000, 5_000] {
        src_handle.push_buffer(buffer_with_pts(pts)).await.unwrap();
    }
    // Poll with yield_now, not a timer: the sink's deliberately *blocking*
    // consume can stall the tokio time driver, and a timer-based wait would
    // wake only after the whole burst has drained — pause() must land inside
    // the first consume's window.
    let t0 = std::time::Instant::now();
    while !log.lock().unwrap().iter().any(|e| e == "buf:1000") {
        assert!(
            t0.elapsed() < Duration::from_secs(10),
            "first consume never came"
        );
        tokio::task::yield_now().await;
    }
    handle.pause();
    handle.pause(); // idempotent: must not deliver a second Pause event

    // While paused nothing more is consumed (the stash is held, not played).
    tokio::time::sleep(Duration::from_millis(150)).await;
    let frozen = log.lock().unwrap().len();
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(
        log.lock().unwrap().len(),
        frozen,
        "the sink kept working while paused: {:?}",
        log.lock().unwrap()
    );

    handle.resume();
    wait_until(
        || log.lock().unwrap().iter().any(|e| e == "buf:5000"),
        "the full burst after resume",
    )
    .await;

    src_handle.end_stream();
    handle.wait().await.unwrap();

    let log = log.lock().unwrap();
    let buffers: Vec<&String> = log.iter().filter(|e| e.starts_with("buf:")).collect();
    assert_eq!(
        buffers,
        ["buf:1000", "buf:2000", "buf:3000", "buf:4000", "buf:5000"],
        "pause lost or reordered buffers: {log:?}"
    );
    assert_eq!(
        log.iter().filter(|e| *e == "ev:pause").count(),
        1,
        "exactly one Pause event: {log:?}"
    );
    assert_eq!(
        log.iter().filter(|e| *e == "ev:resume").count(),
        1,
        "exactly one Resume event: {log:?}"
    );
    let pause_at = log.iter().position(|e| e == "ev:pause").unwrap();
    let resume_at = log.iter().position(|e| e == "ev:resume").unwrap();
    assert!(pause_at < resume_at, "Pause precedes Resume: {log:?}");
    assert!(
        !log[pause_at..resume_at]
            .iter()
            .any(|e| e.starts_with("buf:")),
        "a buffer was consumed between Pause and Resume: {log:?}"
    );
}

/// Pausing a pipeline whose Block links are completely full must not
/// deadlock: resume un-parks every blocked send and the run reaches EOS.
#[tokio::test(flavor = "multi_thread")]
async fn pause_with_full_block_links_resumes_to_eos() {
    use parallax::pipeline::LinkPolicy;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(50));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline
        .link_pads_full(src, "src", sink, "sink", LinkPolicy::Block, Some(2))
        .unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    tokio::time::sleep(Duration::from_millis(50)).await;
    handle.pause();
    tokio::time::sleep(Duration::from_millis(100)).await;
    handle.resume();

    tokio::time::timeout(Duration::from_secs(10), handle.wait())
        .await
        .expect("paused-then-resumed pipeline reached EOS")
        .unwrap();
}

/// position() follows the last-presented PTS monotonically, holds across a
/// pause, and is re-anchored backwards by a flushing seek's Segment.
#[tokio::test(flavor = "multi_thread")]
async fn position_tracks_presented_pts_and_seeks() {
    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for pts in [10_000u64, 20_000, 30_000] {
        src_handle.push_buffer(buffer_with_pts(pts)).await.unwrap();
    }
    wait_until(
        || handle.position() == ClockTime::from_nanos(30_000),
        "position to reach the last PTS",
    )
    .await;

    // Paused: nothing is presented, so the position holds still.
    handle.pause();
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert_eq!(handle.position(), ClockTime::from_nanos(30_000));
    handle.resume();

    // A flushing seek re-anchors the position at the segment start, and the
    // first post-seek buffer advances it from there — backwards moves work
    // because FlushStop reset the max().
    assert!(handle.seek_time(ClockTime::from_nanos(5_000)).await);
    wait_until(
        || handle.position() == ClockTime::from_nanos(5_000),
        "position at the segment start",
    )
    .await;

    src_handle
        .push_buffer(buffer_with_pts(6_000))
        .await
        .unwrap();
    wait_until(
        || handle.position() == ClockTime::from_nanos(6_000),
        "position at the post-seek PTS",
    )
    .await;

    src_handle.end_stream();
    handle.wait().await.unwrap();
}

/// A seek lands WHILE paused (#71): since #163 the seek's path runs through
/// the (gated) sink, which must keep draining its upstream inbox during the
/// hold — otherwise the seek would wedge until resume.
#[tokio::test(flavor = "multi_thread")]
async fn seek_lands_while_paused() {
    use parallax::pipeline::bus::MessageKind;

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    src_handle
        .push_buffer(buffer_with_pts(10_000))
        .await
        .unwrap();
    wait_until(
        || handle.position() == ClockTime::from_nanos(10_000),
        "first buffer presented",
    )
    .await;

    handle.pause();
    assert!(handle.seek_time(ClockTime::from_nanos(500)).await);

    // The seek completes during the pause: SeekDone posts without a resume.
    let mut seek_done = false;
    for _ in 0..400 {
        while let Some(msg) = bus.poll() {
            if matches!(msg.kind, MessageKind::SeekDone { .. }) {
                seek_done = true;
            }
        }
        if seek_done {
            break;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    assert!(seek_done, "seek must land while paused");
    assert!(handle.is_paused());

    handle.resume();
    src_handle.end_stream();
    handle.wait().await.unwrap();
}
