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
    wait_until(
        || sink_log.lock().unwrap().iter().any(|e| e == "segment"),
        "segment at the sink",
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
        let flush_start = pos("flush-start");
        let flush_stop = pos("flush-stop");
        let segment = pos("segment");
        assert!(
            flush_start < flush_stop && flush_stop < segment,
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
                        i > segment,
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

    // The bus reported the seek, at the requested position.
    let mut seek_done = None;
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone { position } = msg.kind {
            seek_done = Some(position);
        }
    }
    assert_eq!(seek_done, Some(ClockTime::from_nanos(1_000)));
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
    let sink = pipeline.add_sink("sink", appsink);
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
