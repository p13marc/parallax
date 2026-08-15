//! Integration tests for Queue2's buffering-aware seeking (#164).
//!
//! Download-mode Queue2 sits mid-graph (`src ! queue2 ! sink`), records true
//! byte offsets from `Metadata.offset`, and absorbs forward byte seeks that
//! the arriving stream will satisfy anyway — the flush trio originates from
//! the queue's task and the source never repositions. Seeks it cannot absorb
//! pass through to the source, and the downloaded-ranges bookkeeping stays
//! honest across the resulting discontinuity.

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::control::Controllable;
use parallax::element::{ProduceContext, ProduceResult, Source};
use parallax::elements::flow::Queue2;
use parallax::elements::{AppSink, AppSinkHandle, Pulled};
use parallax::error::Result;
use parallax::event::{Event, EventResult, SegmentFormat};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
use parallax::pipeline::{Executor, Pipeline};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

const CHUNK: u64 = 64;

fn arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 512).unwrap())
}

/// Drain an AppSink to its terminal state (EOS/disconnect), tolerating
/// flush windows — a sink nobody pulls back-pressures its task forever.
async fn drain_all(handle: AppSinkHandle) {
    loop {
        match handle.pull_buffer().await {
            Pulled::Buffer(_) => {}
            Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
            _ => break,
        }
    }
}

/// Drain an AppSink in the background, recording each pulled buffer's
/// length. Keeping the sink drained is what lets upstream events keep
/// moving: a sink blocked inside `consume` only polls its inbox between
/// buffers.
fn spawn_recording_drain(
    handle: AppSinkHandle,
) -> (tokio::task::JoinHandle<()>, Arc<Mutex<Vec<usize>>>) {
    let lens: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let lens_clone = lens.clone();
    let task = tokio::spawn(async move {
        loop {
            match handle.pull_buffer().await {
                Pulled::Buffer(b) => lens_clone.lock().unwrap().push(b.len()),
                Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
                _ => break,
            }
        }
    });
    (task, lens)
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

/// A byte source that stamps `Metadata.offset` like FileSrc/HttpSrc do,
/// produces up to a runtime-adjustable byte limit (then WouldBlock), and
/// counts + honors the byte seeks that reach it.
struct StampedByteSource {
    pos: u64,
    limit: Arc<AtomicU64>,
    seeks: Arc<AtomicU64>,
}

impl Source for StampedByteSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.pos >= self.limit.load(Ordering::SeqCst) {
            return Ok(ProduceResult::WouldBlock);
        }
        arena().reclaim();
        let Some(slot) = arena().acquire() else {
            return Ok(ProduceResult::WouldBlock);
        };
        let mut meta = Metadata::from_sequence(self.pos / CHUNK);
        meta.offset = Some(self.pos);
        self.pos += CHUNK;
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, CHUNK as usize),
            meta,
        )))
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        if let Event::Seek(seek) = event
            && seek.format == SegmentFormat::Bytes
        {
            self.seeks.fetch_add(1, Ordering::SeqCst);
            self.pos = seek.start.position.max(0) as u64;
            return EventResult::handled_at(self.pos as i64);
        }
        EventResult::NotHandled
    }
}

struct Rig {
    handle: parallax::pipeline::PipelineHandle,
    bus: parallax::pipeline::bus::Bus,
    sink: AppSinkHandle,
    source_seeks: Arc<AtomicU64>,
    limit: Arc<AtomicU64>,
    ranges: parallax::elements::flow::Queue2RangesHandle,
    queue_events: Arc<Mutex<Vec<String>>>,
    _temp: tempfile::NamedTempFile,
}

/// Build `src ! queue2(download) ! appsink`, started, with the byte source
/// initially limited to `initial_bytes`.
fn start_rig(initial_bytes: u64, total: u64) -> Rig {
    let source_seeks = Arc::new(AtomicU64::new(0));
    let limit = Arc::new(AtomicU64::new(initial_bytes));
    let temp = tempfile::NamedTempFile::new().unwrap();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        StampedByteSource {
            pos: 0,
            limit: limit.clone(),
            seeks: source_seeks.clone(),
        },
    );
    let queue = Queue2::download(temp.path(), Some(total));
    let ranges = queue.control();
    let q = pipeline.add_filter("queue2", queue);
    let appsink = AppSink::with_max_buffers(4);
    let sink = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, q).unwrap();
    pipeline.link(q, snk).unwrap();

    // Record events leaving the queue's src pad: an absorbed seek's flush
    // trio must originate here.
    let queue_events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = queue_events.clone();
    let _ = pipeline.add_probe(
        PadRef::src(q),
        ProbeType::EVENT_DOWN | ProbeType::EVENT_FLUSH,
        move |data| {
            if let ProbeData::Event(e) = data {
                events_clone.lock().unwrap().push(e.name().to_string());
            }
            ProbeReturn::Ok
        },
    );

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let bus = handle.take_bus().unwrap();
    Rig {
        handle,
        bus,
        sink,
        source_seeks,
        limit,
        ranges,
        queue_events,
        _temp: temp,
    }
}

/// A forward byte seek within the threshold is absorbed by the queue: the
/// flush trio fires from the queue's src pad, SeekDone names "queue2", the
/// source never repositions, and the post-seek stream starts at the target.
#[tokio::test(flavor = "multi_thread")]
async fn a_forward_seek_within_threshold_is_absorbed_by_queue2() {
    let mut rig = start_rig(10 * CHUNK, 100_000);
    let (drain, lens) = spawn_recording_drain(rig.sink);

    // Let the head of the stream flow through the queue.
    wait_until(
        || rig.ranges.write_pos() == 10 * CHUNK,
        "the whole limited stream to be recorded",
    )
    .await;

    // 60 bytes past the write position: absorbable.
    let target = 10 * CHUNK + 60;
    assert!(rig.handle.seek_bytes(target).await, "the seek dispatches");

    let mut seek_done_source = None;
    wait_until(
        || {
            while let Some(msg) = rig.bus.poll() {
                if let MessageKind::SeekDone { source, .. } = msg.kind {
                    seek_done_source = Some(source);
                }
            }
            seek_done_source.is_some()
        },
        "SeekDone",
    )
    .await;
    assert_eq!(
        seek_done_source.as_deref(),
        Some("queue2"),
        "the queue absorbed the seek"
    );
    assert_eq!(
        rig.source_seeks.load(Ordering::SeqCst),
        0,
        "the source never saw the absorbed seek"
    );
    {
        let events = rig.queue_events.lock().unwrap();
        let fs = events.iter().position(|e| e == "flush-start");
        let fstop = events.iter().position(|e| e == "flush-stop");
        assert!(
            fs.is_some() && fstop.is_some() && fs < fstop,
            "the flush trio originated from the queue's task: {events:?}"
        );
    }

    // Release more data; the queue skips to the target and forwards its
    // straddling tail first — CHUNK - 60 = 4 bytes.
    rig.limit.store(20 * CHUNK, Ordering::SeqCst);
    wait_until(
        || lens.lock().unwrap().contains(&((CHUNK - 60) as usize)),
        "the post-target straddler tail",
    )
    .await;

    // The skipped bytes are still part of the download.
    assert!(
        rig.ranges.contains(10 * CHUNK + 30),
        "skipped-through bytes were recorded"
    );

    rig.handle.stop();
    drain.await.unwrap();
    rig.handle.wait().await.unwrap();
}

/// A seek beyond the threshold passes through to the source; the resulting
/// discontinuity leaves two honest downloaded ranges.
#[tokio::test(flavor = "multi_thread")]
async fn a_far_seek_reaches_the_source_and_splits_the_ranges() {
    let mut rig = start_rig(4 * CHUNK, 10_000_000);
    let (drain, _lens) = spawn_recording_drain(rig.sink);

    wait_until(
        || rig.ranges.write_pos() == 4 * CHUNK,
        "the head to be recorded",
    )
    .await;

    // Far beyond the 512 KiB default threshold.
    let target = 2_000_000;
    rig.limit.store(u64::MAX, Ordering::SeqCst);
    assert!(rig.handle.seek_bytes(target).await);

    let mut seek_done_source = None;
    wait_until(
        || {
            while let Some(msg) = rig.bus.poll() {
                if let MessageKind::SeekDone { source, .. } = msg.kind {
                    seek_done_source = Some(source);
                }
            }
            seek_done_source.is_some()
        },
        "SeekDone",
    )
    .await;
    assert_eq!(
        seek_done_source.as_deref(),
        Some("src"),
        "a far seek is the source's to handle"
    );
    assert_eq!(rig.source_seeks.load(Ordering::SeqCst), 1);

    // Post-seek data lands at its true offset: two spans, honest hole.
    wait_until(
        || rig.ranges.ranges().len() == 2 && rig.ranges.write_pos() > target,
        "the post-seek span to be recorded",
    )
    .await;
    let spans = rig.ranges.ranges();
    // The source may race a few chunks past the initial limit before the
    // seek reaches it; the head span just has to start at 0 and end well
    // short of the target.
    assert_eq!(spans[0].0, 0);
    assert!(
        spans[0].1 >= 4 * CHUNK && spans[0].1 < target,
        "head span {:?} ends between the initial limit and the target",
        spans[0]
    );
    assert_eq!(spans[1].0, target, "the second span starts at the target");
    assert!(!rig.ranges.contains(1_000_000), "the hole is reported");

    rig.handle.stop();
    drain.await.unwrap();
    rig.handle.wait().await.unwrap();
}

/// End-to-end over a real file: `filesrc ! queue2(download) ! appsink`
/// downloads the whole file into one span, and the bus carries a
/// DownloadProgress message.
#[tokio::test(flavor = "multi_thread")]
async fn filesrc_through_queue2_reports_ranges_and_progress() {
    use std::io::Write;
    let mut media = tempfile::NamedTempFile::new().unwrap();
    let payload = vec![0x5Au8; 8192];
    media.write_all(&payload).unwrap();
    media.flush().unwrap();
    let download = tempfile::NamedTempFile::new().unwrap();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "filesrc",
        parallax::elements::FileSrc::open(media.path()).unwrap(),
    );
    let queue = Queue2::download(download.path(), Some(8192));
    let ranges = queue.control();
    let q = pipeline.add_filter("queue2", queue);
    let appsink = AppSink::new();
    let sink = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, q).unwrap();
    pipeline.link(q, snk).unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    let mut received = 0usize;
    loop {
        match sink.pull_buffer().await {
            Pulled::Buffer(b) => received += b.len(),
            Pulled::Empty | Pulled::Flushing => tokio::task::yield_now().await,
            _ => break,
        }
    }
    handle.wait().await.unwrap();

    assert_eq!(received, 8192, "every byte flowed through the queue");
    assert_eq!(ranges.ranges(), vec![(0, 8192)]);
    assert_eq!(ranges.total(), Some(8192));

    let mut saw_progress = false;
    while let Some(msg) = bus.poll() {
        if let MessageKind::DownloadProgress { ranges, total, .. } = msg.kind {
            saw_progress = true;
            assert_eq!(total, Some(8192));
            assert_eq!(ranges, vec![(0, 8192)]);
        }
    }
    assert!(saw_progress, "DownloadProgress was posted");
}

/// The adapter fix, end-to-end: a source-handled flushing seek travels back
/// down through an `add_filter`-wrapped stream-mode Queue2, whose FlushStart
/// handling (ring drop + re-entering buffering) now actually runs.
#[tokio::test(flavor = "multi_thread")]
async fn a_flush_from_upstream_reaches_queue2_through_the_adapter() {
    let source_seeks = Arc::new(AtomicU64::new(0));
    let limit = Arc::new(AtomicU64::new(u64::MAX));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        StampedByteSource {
            pos: 0,
            limit: limit.clone(),
            seeks: source_seeks.clone(),
        },
    );
    // Tiny stream-mode queue: fills to 100% immediately, so the post-flush
    // re-entry into buffering is observable as a percent drop.
    let q = pipeline.add_filter("queue2", Queue2::stream(256));
    let appsink = AppSink::with_max_buffers(4);
    let sink = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, q).unwrap();
    pipeline.link(q, snk).unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    // Keep the sink drained in the background: a sink blocked inside
    // `consume` (full AppSink queue) only polls its inbox between buffers.
    let drain = tokio::spawn(drain_all(sink));
    assert!(handle.seek_bytes(0).await, "the source handles the seek");

    // After the source's flush trio passes through the queue, its FlushStart
    // arm re-enters buffering and posts an honest low percentage.
    let mut seek_done_at = None;
    let mut post_seek_low_percent = false;
    wait_until(
        || {
            while let Some(msg) = bus.poll() {
                match msg.kind {
                    MessageKind::SeekDone { .. } => seek_done_at = Some(true),
                    MessageKind::Buffering { percent, .. }
                        if seek_done_at.is_some() && percent < 100 =>
                    {
                        post_seek_low_percent = true;
                    }
                    _ => {}
                }
            }
            post_seek_low_percent
        },
        "a post-seek re-buffering message (proof FlushStart reached Queue2)",
    )
    .await;

    handle.stop();
    drain.await.unwrap();
    handle.wait().await.unwrap();
}
