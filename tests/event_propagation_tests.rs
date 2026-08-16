//! Integration tests for runtime event propagation (#72).
//!
//! FlushStart/FlushStop/Segment must traverse the inter-element channels of a
//! *running* pipeline, in order relative to buffers, and a flushing seek must
//! invoke `flush()` on transforms and keep stale buffers from surfacing after
//! FlushStop.

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{Output, Transform};
use parallax::elements::{AppSink, AppSrc, FileSrc, NullSink, NullSource, Pulled};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
use parallax::pipeline::{Executor, Pipeline};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

fn arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 512).unwrap())
}

fn buffer_with_pts(pts_ns: u64) -> Buffer {
    let slot = arena().acquire().expect("test arena exhausted");
    let mut metadata = Metadata::new();
    metadata.pts = ClockTime::from_nanos(pts_ns);
    Buffer::new(MemoryHandle::with_len(slot, 8), metadata)
}

/// Ordered log of everything a pad saw: `buffer:<pts>` or the event's name.
type PadLog = Arc<Mutex<Vec<String>>>;

fn log_probe(pipeline: &mut Pipeline, pad: PadRef) -> PadLog {
    let log: PadLog = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();
    let _ = pipeline.add_probe(
        pad,
        ProbeType::BUFFER | ProbeType::EVENT_DOWN | ProbeType::EVENT_FLUSH,
        move |data| {
            let entry = match data {
                ProbeData::Buffer(b) => format!("buffer:{}", b.metadata().pts.nanos()),
                ProbeData::Event(e) => e.name().to_string(),
                _ => return ProbeReturn::Ok,
            };
            log_clone.lock().unwrap().push(entry);
            ProbeReturn::Ok
        },
    );
    log
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

/// Passthrough transform that parks one designated buffer as pending output
/// (a stand-in for a decoder's reorder queue) and counts `flush()` calls.
struct HoldingTransform {
    hold_pts: u64,
    pending: Option<Buffer>,
    flushes: Arc<AtomicU64>,
}

impl Transform for HoldingTransform {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        if buffer.metadata().pts.nanos() == self.hold_pts {
            self.pending = Some(buffer);
            Ok(Output::None)
        } else {
            Ok(Output::Single(buffer))
        }
    }

    fn flush(&mut self) -> Result<Output> {
        self.flushes.fetch_add(1, Ordering::SeqCst);
        Ok(match self.pending.take() {
            Some(b) => Output::Single(b),
            None => Output::None,
        })
    }

    fn name(&self) -> &str {
        "holding"
    }
}

/// The headline test: a flushing seek mid-stream reaches every pad of a
/// 3-element chain as `flush-start, flush-stop, segment`, strictly between the
/// pre-seek and post-seek buffers; the transform's `flush()` runs on
/// FlushStart and its pending output is discarded, not forwarded.
#[tokio::test(flavor = "multi_thread")]
async fn flush_seek_traverses_chain_in_order() {
    let mut pipeline = Pipeline::new();

    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let flushes = Arc::new(AtomicU64::new(0));
    let src = pipeline.add_source("src", appsrc);
    let xfm = pipeline.add_transform(
        "hold",
        HoldingTransform {
            hold_pts: 2,
            pending: None,
            flushes: flushes.clone(),
        },
    );
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, sink).unwrap();

    let src_log = log_probe(&mut pipeline, PadRef::src(src));
    let xfm_log = log_probe(&mut pipeline, PadRef::src(xfm));
    let sink_log = log_probe(&mut pipeline, PadRef::sink(sink));

    // Pre-seek buffers, including the one the transform parks (pts 2).
    for pts in 0..4u64 {
        src_handle.push_buffer(buffer_with_pts(pts)).await.unwrap();
    }

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    // pts 2 is held by the transform, so the sink sees 3 of the 4.
    wait_until(
        || {
            sink_log
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.starts_with("buffer:"))
                .count()
                >= 3
        },
        "pre-seek buffers at the sink",
    )
    .await;

    assert!(handle.seek_time(ClockTime::from_nanos(1_000)).await);

    // The whole sequence must arrive at the sink before we resume pushing.
    // The FIRST segment on the wire is the initial one (#165); the seek's
    // segment is the second.
    wait_until(
        || {
            sink_log
                .lock()
                .unwrap()
                .iter()
                .filter(|e| *e == "segment")
                .count()
                >= 2
        },
        "the seek's segment at the sink",
    )
    .await;

    // Post-seek data, then EOS.
    src_handle.push_buffer(buffer_with_pts(100)).await.unwrap();
    src_handle.end_stream();
    handle.wait().await.unwrap();

    for (name, log) in [("src", src_log), ("xfm", xfm_log), ("sink", sink_log)] {
        let log = log.lock().unwrap();
        let pos = |needle: &str| {
            log.iter()
                .position(|e| e == needle)
                .unwrap_or_else(|| panic!("{name}: no '{needle}' in {log:?}"))
        };
        // #165: every pad's wire begins stream-start, then the initial
        // segment, before any buffer.
        assert_eq!(log[0], "stream-start", "{name}: {log:?}");
        let initial_segment = pos("segment");
        let first_buffer = log
            .iter()
            .position(|e| e.starts_with("buffer:"))
            .unwrap_or_else(|| panic!("{name}: no buffers in {log:?}"));
        assert!(
            initial_segment < first_buffer,
            "{name}: initial segment must precede the first buffer: {log:?}"
        );
        let flush_start = pos("flush-start");
        let flush_stop = pos("flush-stop");
        // The seek's segment is the first one AFTER flush-stop.
        let seek_segment = flush_stop
            + 1
            + log[flush_stop + 1..]
                .iter()
                .position(|e| e == "segment")
                .unwrap_or_else(|| panic!("{name}: no post-flush segment in {log:?}"));
        assert!(
            flush_start < flush_stop && flush_stop < seek_segment,
            "{name}: events out of order: {log:?}"
        );
        for (i, entry) in log.iter().enumerate() {
            if let Some(pts) = entry.strip_prefix("buffer:") {
                let pts: u64 = pts.parse().unwrap();
                if pts < 100 {
                    assert!(
                        i < flush_start,
                        "{name}: pre-seek buffer {pts} surfaced after FlushStart: {log:?}"
                    );
                } else {
                    assert!(
                        i > seek_segment,
                        "{name}: post-seek buffer {pts} surfaced before Segment: {log:?}"
                    );
                }
            }
        }
        // The parked pts-2 buffer must have been discarded by the flush, so it
        // never reaches the wire on any pad past the transform.
        if name != "src" {
            assert!(
                !log.iter().any(|e| e == "buffer:2"),
                "{name}: flushed pending buffer leaked: {log:?}"
            );
        }
    }

    // flush() ran once for the FlushStart and once at EOS.
    assert_eq!(flushes.load(Ordering::SeqCst), 2);

    // The bus reported the seek: AppSrc reports no landing position, so
    // the requested target is echoed, tagged with the seek's format.
    let mut seek_done = None;
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone {
            format, position, ..
        } = msg.kind
        {
            seek_done = Some((format, position));
        }
    }
    assert_eq!(
        seek_done,
        Some((parallax::event::SegmentFormat::Time, Some(1_000)))
    );
}

/// A NON-flushing seek is a queued seek (#162, GStreamer's queue-behind-data
/// semantics): the source repositions, but nothing is discarded — the new
/// Segment travels FIFO behind the already-queued buffers, so downstream
/// sees old data → segment → new data, with no flush events anywhere.
#[tokio::test(flavor = "multi_thread")]
async fn non_flushing_seek_queues_behind_data() {
    use parallax::event::{SeekEvent, SeekFlags, SeekPosition, SegmentFormat};

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let appsink = AppSink::with_max_buffers(32);
    let sink_handle = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, snk).unwrap();

    let sink_log = log_probe(&mut pipeline, PadRef::sink(snk));

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    for pts in [1_000u64, 2_000, 3_000] {
        src_handle.push_buffer(buffer_with_pts(pts)).await.unwrap();
    }
    // Let the pre-seek buffers reach the sink pad before the seek, so their
    // ordering relative to the Segment is unambiguous in the log.
    wait_until(
        || {
            sink_log
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.starts_with("buffer:"))
                .count()
                >= 3
        },
        "pre-seek buffers at the sink",
    )
    .await;

    let seek = SeekEvent::new(SegmentFormat::Time, SeekPosition::set(1_000_000))
        .with_flags(SeekFlags::empty());
    assert!(handle.seek(seek).await);

    wait_until(
        || {
            sink_log
                .lock()
                .unwrap()
                .iter()
                .filter(|e| *e == "segment")
                .count()
                >= 2
        },
        "the queued segment (the initial one is first, #165)",
    )
    .await;
    src_handle
        .push_buffer(buffer_with_pts(1_000_000))
        .await
        .unwrap();

    src_handle.end_stream();
    while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {}
    handle.wait().await.unwrap();

    let log = sink_log.lock().unwrap().clone();
    assert!(
        !log.iter().any(|e| e.starts_with("flush")),
        "a non-flushing seek must not flush: {log:?}"
    );
    let segment_at = log.iter().rposition(|e| e == "segment").unwrap();
    let pre = &log[..segment_at];
    assert_eq!(
        pre.iter().filter(|e| e.starts_with("buffer:")).count(),
        3,
        "all pre-seek buffers arrive before the segment, none dropped: {log:?}"
    );
    assert!(
        log[segment_at..].contains(&"buffer:1000000".to_string()),
        "post-seek data follows the segment: {log:?}"
    );

    // SeekDone still posts — it means "source repositioned", not "the
    // segment reached the sinks".
    let mut seek_done = false;
    while let Some(msg) = bus.poll() {
        if matches!(msg.kind, MessageKind::SeekDone { .. }) {
            seek_done = true;
        }
    }
    assert!(seek_done, "SeekDone posted for a non-flushing seek");
}

/// A source that cannot seek reports a warning on the bus and keeps running.
#[tokio::test(flavor = "multi_thread")]
async fn unseekable_pipeline_rejects_seek() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(u64::MAX));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    // GStreamer discipline (#162): no element declared seek support, so the
    // seek is rejected up front — nothing dispatched, nothing flushed,
    // nothing on the bus. The old behavior fired it at every source and
    // warned after the fact.
    assert!(!handle.seekable());
    assert!(!handle.query_seekable().seekable);
    assert!(!handle.seek_time(ClockTime::from_nanos(0)).await);

    // Give the pipeline a moment, then prove the seek left no trace and the
    // stream is still running.
    tokio::time::sleep(Duration::from_millis(100)).await;
    while let Some(msg) = bus.poll() {
        assert!(
            !matches!(
                msg.kind,
                MessageKind::Warning { .. } | MessageKind::SeekDone { .. }
            ),
            "a rejected seek must leave no bus trace, got {:?}",
            msg.kind
        );
    }

    handle.stop();
    handle.wait().await.unwrap();
}

/// End-to-end byte seek on the one seekable built-in: FileSrc repositions and
/// post-seek payloads start at the requested offset.
#[tokio::test(flavor = "multi_thread")]
async fn filesrc_seek_bytes_repositions() {
    let dir = std::env::temp_dir().join(format!("parallax-seek-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("pattern.bin");
    // Recognizable content: byte i is i % 251.
    let data: Vec<u8> = (0..64 * 1024u32).map(|i| (i % 251) as u8).collect();
    std::fs::write(&path, &data).unwrap();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::new(&path).with_chunk_size(1024));
    let appsink = AppSink::with_max_buffers(4);
    let sink_handle = appsink.handle();
    let sink = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    // Pull a few pre-seek chunks; the bounded AppSink back-pressures the
    // source, so the file cannot run out before the seek lands.
    for _ in 0..3 {
        match sink_handle.pull_buffer().await {
            Pulled::Buffer(_) => {}
            other => panic!("expected a buffer, got {other:?}"),
        }
    }

    let offset = 40 * 1024u64;
    assert!(handle.seek_bytes(offset).await);

    // Drain to EOS; after the flush the payloads must restart at `offset`.
    let mut post_flush = Vec::new();
    while let Pulled::Buffer(b) = sink_handle.pull_buffer().await {
        post_flush.push(b.as_bytes().to_vec());
    }
    handle.wait().await.unwrap();

    let expected = &data[offset as usize..offset as usize + 16];
    assert!(
        post_flush
            .iter()
            .any(|p| p.len() >= 16 && &p[..16] == expected),
        "no post-seek chunk starts at offset {offset}"
    );

    let mut seek_done = false;
    while let Some(msg) = bus.poll() {
        if matches!(msg.kind, MessageKind::SeekDone { .. }) {
            seek_done = true;
        }
    }
    assert!(seek_done, "no SeekDone on the bus");

    let _ = std::fs::remove_dir_all(&dir);
}

// ============================================================================
// #165 slice 1: initial StreamStart/Segment
// ============================================================================

/// Captured segments: `(format, start, rate, stop)`.
type SegmentLog = Arc<Mutex<Vec<(parallax::event::SegmentFormat, i64, f64, i64)>>>;

fn segment_probe(pipeline: &mut Pipeline, pad: PadRef) -> SegmentLog {
    let log: SegmentLog = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();
    let _ = pipeline.add_probe(pad, ProbeType::EVENT_DOWN, move |data| {
        if let ProbeData::Event(parallax::event::Event::Segment(seg)) = data {
            log_clone
                .lock()
                .unwrap()
                .push((seg.format, seg.start, seg.rate, seg.stop));
        }
        ProbeReturn::Ok
    });
    log
}

/// Every pad's wire begins stream-start → segment → buffers, and the initial
/// segment anchors at the FIRST buffer's PTS — not zero — so `position()` is
/// honest for streams that start late.
#[tokio::test(flavor = "multi_thread")]
async fn pipeline_start_emits_stream_start_then_segment() {
    use parallax::event::SegmentFormat;

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(8);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let snk = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, snk).unwrap();

    let src_log = log_probe(&mut pipeline, PadRef::src(src));
    let sink_log = log_probe(&mut pipeline, PadRef::sink(snk));
    let segments = segment_probe(&mut pipeline, PadRef::sink(snk));

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // First PTS is 10 s, not zero.
    src_handle
        .push_buffer(buffer_with_pts(10_000_000_000))
        .await
        .unwrap();
    src_handle
        .push_buffer(buffer_with_pts(10_100_000_000))
        .await
        .unwrap();
    src_handle.end_stream();
    handle.wait().await.unwrap();

    for (name, log) in [("src", src_log), ("sink", sink_log)] {
        let log = log.lock().unwrap();
        assert_eq!(
            &log[..3],
            &[
                "stream-start".to_string(),
                "segment".to_string(),
                "buffer:10000000000".to_string()
            ],
            "{name}: wire must start stream-start → segment → first buffer: {log:?}"
        );
    }
    let segments = segments.lock().unwrap();
    assert_eq!(
        segments.as_slice(),
        &[(SegmentFormat::Time, 10_000_000_000, 1.0, -1)],
        "initial segment anchors at the first PTS at rate 1.0, no stop"
    );
}

/// The initial segment re-anchors `position()` immediately: a stream starting
/// at 10 s reports ~10 s from its first buffer, not the running-time fallback.
#[tokio::test(flavor = "multi_thread")]
async fn initial_segment_anchors_position_for_nonzero_streams() {
    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(8);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let appsink = AppSink::with_max_buffers(4);
    let sink_handle = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    src_handle
        .push_buffer(buffer_with_pts(10_000_000_000))
        .await
        .unwrap();
    wait_until(
        || handle.position().to_option() == Some(ClockTime::from_secs(10)),
        "position anchored at the stream's real start",
    )
    .await;

    src_handle.end_stream();
    while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {}
    handle.wait().await.unwrap();
}

/// An untimestamped stream (FileSrc) gets a Bytes segment from zero — the
/// sink's Time-only position store ignores it.
#[tokio::test(flavor = "multi_thread")]
async fn untimestamped_source_emits_bytes_segment() {
    use parallax::event::SegmentFormat;

    let dir = std::env::temp_dir().join(format!("parallax-165-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("data.bin");
    std::fs::write(&path, vec![7u8; 4096]).unwrap();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::new(&path).with_chunk_size(1024));
    let snk = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, snk).unwrap();
    let segments = segment_probe(&mut pipeline, PadRef::sink(snk));

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    let segments = segments.lock().unwrap();
    assert_eq!(
        segments.as_slice(),
        &[(SegmentFormat::Bytes, 0, 1.0, -1)],
        "untimestamped stream anchors a Bytes segment at zero"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// The post-seek segment now carries the seek's rate and stop (#165), so the
/// segment's shape is trick-play-ready even though playback rate is not yet
/// consumed anywhere else.
#[tokio::test(flavor = "multi_thread")]
async fn post_seek_segment_carries_rate_and_stop() {
    use parallax::event::{SeekEvent, SeekPosition, SegmentFormat};

    let dir = std::env::temp_dir().join(format!("parallax-165b-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("data.bin");
    std::fs::write(&path, vec![7u8; 64 * 1024]).unwrap();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::new(&path).with_chunk_size(1024));
    let appsink = AppSink::with_max_buffers(2);
    let sink_handle = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, snk).unwrap();
    let segments = segment_probe(&mut pipeline, PadRef::sink(snk));

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for _ in 0..2 {
        assert!(matches!(sink_handle.pull_buffer().await, Pulled::Buffer(_)));
    }
    let seek = SeekEvent::new_bytes(32 * 1024)
        .with_rate(2.0)
        .with_stop(SeekPosition::set(48 * 1024));
    assert!(handle.seek(seek).await);

    while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {}
    handle.wait().await.unwrap();

    let segments = segments.lock().unwrap();
    let last = segments.last().expect("seek segment captured");
    assert_eq!(
        *last,
        (SegmentFormat::Bytes, 32 * 1024, 2.0, 48 * 1024),
        "seek segment carries landing, rate and stop: {segments:?}"
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// A non-flushing seek's queued Segment carries `base` = the running time
/// already consumed under the outgoing segment (#165), so downstream running
/// time stays monotonic across the queued boundary. (A flushing seek keeps
/// base 0 — running time restarts.)
#[tokio::test(flavor = "multi_thread")]
async fn non_flushing_seek_accumulates_base() {
    use parallax::event::{Event, SeekEvent, SeekFlags, SeekPosition, SegmentEvent, SegmentFormat};

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let appsink = AppSink::with_max_buffers(32);
    let sink_handle = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, snk).unwrap();

    let segments: Arc<Mutex<Vec<SegmentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let segments_probe = segments.clone();
    let _ = pipeline.add_probe(PadRef::sink(snk), ProbeType::EVENT_DOWN, move |data| {
        if let ProbeData::Event(Event::Segment(seg)) = data {
            segments_probe.lock().unwrap().push(seg.clone());
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Three buffers, 1 ms apart, starting at t = 1 ms.
    for pts in [1_000_000u64, 2_000_000, 3_000_000] {
        src_handle.push_buffer(buffer_with_pts(pts)).await.unwrap();
    }
    wait_until(
        || segments.lock().unwrap().len() >= 1,
        "the lazy initial segment",
    )
    .await;
    // The tracker's `last_pts` advances in the source task; wait for the
    // data to be fully produced (visible at the sink) before seeking.
    let mut pulled = 0;
    while pulled < 3 {
        if let Pulled::Buffer(_) = sink_handle.pull_buffer().await {
            pulled += 1;
        }
    }

    let seek = SeekEvent::new(SegmentFormat::Time, SeekPosition::set(10_000_000))
        .with_flags(SeekFlags::empty());
    assert!(handle.seek(seek).await);

    wait_until(|| segments.lock().unwrap().len() >= 2, "the queued segment").await;

    src_handle
        .push_buffer(buffer_with_pts(10_000_000))
        .await
        .unwrap();
    src_handle.end_stream();
    while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {}
    handle.wait().await.unwrap();

    let segs = segments.lock().unwrap().clone();
    let initial = &segs[0];
    let queued = &segs[1];
    assert_eq!(initial.base, 0, "initial segment has no accumulated base");
    assert_eq!(
        initial.start, 1_000_000,
        "initial segment anchors at the first buffer's PTS"
    );
    assert_eq!(queued.start, 10_000_000, "queued segment starts at target");
    // Running time consumed under the initial segment: last PTS observed was
    // 3 ms, anchored at 1 ms -> 2 ms.
    assert_eq!(
        queued.base, 2_000_000,
        "queued segment's base is the running time consumed: {segs:?}"
    );
    // Monotonicity across the boundary: the new segment's start maps to at
    // least where the old one left off.
    let before = initial
        .to_running_time(ClockTime::from_nanos(3_000_000))
        .nanos();
    let after = queued
        .to_running_time(ClockTime::from_nanos(10_000_000))
        .nanos();
    assert!(
        after >= before,
        "running time is monotonic across the queued boundary ({before} -> {after})"
    );
}

/// A FLUSHING seek after playback restarts running time: base stays 0.
#[tokio::test(flavor = "multi_thread")]
async fn flushing_seek_resets_base() {
    use parallax::event::{Event, SeekEvent, SeekPosition, SegmentEvent, SegmentFormat};

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let appsink = AppSink::with_max_buffers(32);
    let sink_handle = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, snk).unwrap();

    let segments: Arc<Mutex<Vec<SegmentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let segments_probe = segments.clone();
    let _ = pipeline.add_probe(PadRef::sink(snk), ProbeType::EVENT_DOWN, move |data| {
        if let ProbeData::Event(Event::Segment(seg)) = data {
            segments_probe.lock().unwrap().push(seg.clone());
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for pts in [1_000_000u64, 2_000_000, 3_000_000] {
        src_handle.push_buffer(buffer_with_pts(pts)).await.unwrap();
    }
    let mut pulled = 0;
    while pulled < 3 {
        if let Pulled::Buffer(_) = sink_handle.pull_buffer().await {
            pulled += 1;
        }
    }

    // Default flags include FLUSH.
    let seek = SeekEvent::new(SegmentFormat::Time, SeekPosition::set(10_000_000));
    assert!(handle.seek(seek).await);
    wait_until(
        || segments.lock().unwrap().len() >= 2,
        "the post-seek segment",
    )
    .await;

    src_handle.end_stream();
    while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {}
    handle.wait().await.unwrap();

    let segs = segments.lock().unwrap().clone();
    assert_eq!(
        segs[1].base, 0,
        "a flushing seek restarts running time: {segs:?}"
    );
}
