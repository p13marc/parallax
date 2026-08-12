//! Integration tests for seeking, position queries, and segment arithmetic.

use parallax::clock::ClockTime;
use parallax::element::Source;
use parallax::elements::FileSrc;
use parallax::event::{SegmentEvent, SegmentFormat};
use parallax::pipeline::Pipeline;
use std::io::Write;
use tempfile::NamedTempFile;

// ============================================================================
// Segment Timestamp Mapping Tests
// ============================================================================

#[test]
fn test_segment_running_time_normal_playback() {
    let seg = SegmentEvent::new_time(ClockTime::ZERO, None);
    assert_eq!(
        seg.to_running_time(ClockTime::from_secs(5)),
        ClockTime::from_secs(5)
    );
    assert_eq!(seg.to_running_time(ClockTime::ZERO), ClockTime::ZERO);
}

#[test]
fn test_segment_running_time_after_seek() {
    // Seek to 10s with 5s already elapsed
    let seg = SegmentEvent {
        format: SegmentFormat::Time,
        start: 10_000_000_000,
        stop: -1,
        position: 10_000_000_000,
        rate: 1.0,
        applied_rate: 1.0,
        base: 5_000_000_000,
        flags: Default::default(),
    };

    // PTS=10s → running_time = (10-10)/1 + 5 = 5s
    assert_eq!(
        seg.to_running_time(ClockTime::from_secs(10)),
        ClockTime::from_secs(5)
    );
    // PTS=15s → running_time = (15-10)/1 + 5 = 10s
    assert_eq!(
        seg.to_running_time(ClockTime::from_secs(15)),
        ClockTime::from_secs(10)
    );
}

#[test]
fn test_segment_running_time_double_speed() {
    let seg = SegmentEvent::new_time(ClockTime::ZERO, None).with_rate(2.0);
    // At 2x speed, 10s of content plays in 5s of real time
    assert_eq!(
        seg.to_running_time(ClockTime::from_secs(10)),
        ClockTime::from_secs(5)
    );
}

#[test]
fn test_segment_running_time_half_speed() {
    let seg = SegmentEvent::new_time(ClockTime::ZERO, None).with_rate(0.5);
    // At 0.5x speed, 10s of content plays in 20s of real time
    assert_eq!(
        seg.to_running_time(ClockTime::from_secs(10)),
        ClockTime::from_secs(20)
    );
}

#[test]
fn test_segment_running_time_roundtrip() {
    let seg = SegmentEvent {
        format: SegmentFormat::Time,
        start: 10_000_000_000,
        stop: -1,
        position: 10_000_000_000,
        rate: 2.0,
        applied_rate: 1.0,
        base: 5_000_000_000,
        flags: Default::default(),
    };

    let pts = ClockTime::from_secs(20);
    let rt = seg.to_running_time(pts);
    let roundtrip = seg.running_time_to_pts(rt);
    assert_eq!(roundtrip, pts);
}

#[test]
fn test_segment_none_values() {
    let seg = SegmentEvent::default();
    assert_eq!(seg.to_running_time(ClockTime::NONE), ClockTime::NONE);
    assert_eq!(seg.to_stream_time(ClockTime::NONE), ClockTime::NONE);
    assert_eq!(seg.running_time_to_pts(ClockTime::NONE), ClockTime::NONE);
}

#[test]
fn test_segment_before_start_returns_none() {
    let seg = SegmentEvent::new_time(ClockTime::from_secs(10), None);
    assert_eq!(
        seg.to_running_time(ClockTime::from_secs(5)),
        ClockTime::NONE
    );
}

// ============================================================================
// FileSrc Seeking Tests
// ============================================================================

fn create_test_file(content: &[u8]) -> NamedTempFile {
    let mut temp = NamedTempFile::new().unwrap();
    temp.write_all(content).unwrap();
    temp.flush().unwrap();
    temp
}

#[test]
fn test_filesrc_is_seekable() {
    let temp = create_test_file(b"hello world");
    let src = FileSrc::open(temp.path()).unwrap();
    assert!(src.is_seekable());
}

#[test]
fn test_filesrc_query_duration() {
    let content = b"hello world, this is test data";
    let temp = create_test_file(content);
    let src = FileSrc::open(temp.path()).unwrap();

    let dur = src.query_duration().unwrap();
    assert_eq!(dur.format, SegmentFormat::Bytes);
    assert_eq!(dur.duration, Some(content.len() as u64));
}

#[test]
fn test_filesrc_query_position() {
    let temp = create_test_file(b"hello world");
    let src = FileSrc::open(temp.path()).unwrap();

    let pos = src.query_position().unwrap();
    assert_eq!(pos.format, SegmentFormat::Bytes);
    assert_eq!(pos.position, Some(0));
}

#[test]
fn test_filesrc_seek_bytes() {
    use parallax::element::{ProduceContext, ProduceResult, Source};
    use parallax::event::{Event, SeekEvent};
    use parallax::memory::SharedArena;

    let content = b"ABCDEFGHIJKLMNOP";
    let temp = create_test_file(content);
    let mut src = FileSrc::open(temp.path()).unwrap().with_chunk_size(4);

    // Read first 4 bytes
    let arena = SharedArena::new(64, 4).unwrap();
    let slot = arena.acquire().unwrap();
    let mut ctx = ProduceContext::new(slot);
    match src.produce(&mut ctx).unwrap() {
        ProduceResult::Produced(n) => {
            let buf = ctx.finalize(n);
            assert_eq!(buf.as_bytes(), b"ABCD");
        }
        _ => panic!("Expected Produced"),
    }

    // Seek to byte 8
    let seek = SeekEvent::new_bytes(8);
    let result = src.handle_upstream_event(&Event::Seek(seek));
    assert!(result.is_handled());

    // Read from new position
    arena.reclaim();
    let slot = arena.acquire().unwrap();
    let mut ctx = ProduceContext::new(slot);
    match src.produce(&mut ctx).unwrap() {
        ProduceResult::Produced(n) => {
            let buf = ctx.finalize(n);
            assert_eq!(buf.as_bytes(), b"IJKL");
        }
        _ => panic!("Expected Produced after seek"),
    }

    // Position should reflect the seek
    let pos = src.query_position().unwrap();
    assert_eq!(pos.position, Some(12)); // 8 + 4 bytes read
}

// ============================================================================
// Pipeline-Level Query Tests
// ============================================================================

#[test]
fn test_pipeline_query_seekable_with_filesrc() {
    let content = b"test data for seekable query";
    let temp = create_test_file(content);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::open(temp.path()).unwrap());
    let sink = pipeline.add_sink("sink", parallax::elements::NullSink::new());
    pipeline.link(src, sink).unwrap();

    let seekable = pipeline.query_seekable();
    assert!(seekable.seekable);
    assert_eq!(seekable.start, 0);
    assert_eq!(seekable.stop, content.len() as u64);
}

#[test]
fn test_pipeline_query_duration_with_filesrc() {
    let content = b"test data for duration query";
    let temp = create_test_file(content);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::open(temp.path()).unwrap());
    let sink = pipeline.add_sink("sink", parallax::elements::NullSink::new());
    pipeline.link(src, sink).unwrap();

    let dur = pipeline.query_duration().unwrap();
    assert_eq!(dur.format, SegmentFormat::Bytes);
    assert_eq!(dur.duration, Some(content.len() as u64));
}

#[test]
fn test_pipeline_query_seekable_with_null_source() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", parallax::elements::NullSource::new(10));
    let sink = pipeline.add_sink("sink", parallax::elements::NullSink::new());
    pipeline.link(src, sink).unwrap();

    // NullSource is not seekable
    let seekable = pipeline.query_seekable();
    assert!(!seekable.seekable);
}

// ============================================================================
// Runtime Flush-Seek Tests (#157)
// ============================================================================

/// A flushing runtime seek sheds the queued backlog instead of playing it.
///
/// Before the flush epoch, FlushStart/FlushStop rode the same FIFO channels as
/// the buffers, back-to-back — every buffer already queued sat *ahead* of them
/// and played out normally (~2.5 s of stale audio on a real file). Now the
/// seek bumps a pipeline-wide epoch and consumers drop older-stamped buffers
/// at receive speed, so only the handful already inside elements can surface.
#[tokio::test(flavor = "multi_thread")]
async fn runtime_flush_seek_sheds_the_queued_backlog() {
    use parallax::buffer::{Buffer, MemoryHandle};
    use parallax::element::{ProduceContext, ProduceResult, Source};
    use parallax::elements::util::PassThrough;
    use parallax::elements::{AppSink, Pulled};
    use parallax::event::{Event, EventResult};
    use parallax::memory::SharedArena;
    use parallax::metadata::Metadata;
    use parallax::pipeline::{Executor, LinkPolicy, Pipeline};
    use std::sync::Arc;

    /// Endless counter; PTS = sequence in milliseconds. A time seek jumps the
    /// counter to the target millisecond.
    struct SeekableCounter {
        seq: u64,
        arena: Arc<SharedArena>,
    }

    impl Source for SeekableCounter {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> parallax::error::Result<ProduceResult> {
            self.arena.reclaim();
            let Some(slot) = self.arena.acquire() else {
                return Ok(ProduceResult::WouldBlock);
            };
            let mut metadata = Metadata::from_sequence(self.seq);
            metadata.pts = ClockTime::from_millis(self.seq);
            let buffer = Buffer::new(MemoryHandle::new(slot), metadata);
            self.seq += 1;
            Ok(ProduceResult::OwnBuffer(buffer))
        }

        fn is_seekable(&self) -> bool {
            true
        }

        fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
            match event {
                Event::Seek(seek) => {
                    self.seq = (seek.start.position.max(0) as u64) / 1_000_000;
                    EventResult::handled()
                }
                _ => EventResult::NotHandled,
            }
        }
    }

    let source = SeekableCounter {
        seq: 0,
        arena: Arc::new(SharedArena::new(64, 64).unwrap()),
    };

    let mut pipeline = Pipeline::new();
    let sink = AppSink::with_max_buffers(2);
    let sink_handle = sink.handle();
    let src = pipeline.add_source("src", source);
    let mid = pipeline.add_filter("mid", PassThrough::new());
    let snk = pipeline.add_sink("snk", sink);
    // The wide queue sits *upstream* of the transform: its 16 stale buffers
    // can only reach the sink after the transform re-checks them, and by then
    // the seek's epoch bump has landed. The narrow sink-side queue bounds the
    // worst-case leak (4 + AppSink's 2 + in-flight) below the assertion even
    // if the bump loses every scheduling race.
    pipeline
        .link_pads_full(src, "src", mid, "sink", LinkPolicy::Block, Some(16))
        .unwrap();
    pipeline
        .link_pads_full(mid, "src", snk, "sink", LinkPolicy::Block, Some(4))
        .unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    // Prove flow, then let every queue fill (~34 buffers park the source).
    assert!(matches!(sink_handle.pull_buffer().await, Pulled::Buffer(_)));
    tokio::time::sleep(std::time::Duration::from_millis(150)).await;

    assert!(handle.seek_time(ClockTime::from_secs(60)).await);

    // Drain until the seek lands, pacing the stale phase like a presenting
    // sink would — a zero-delay drain can empty the sink-side queue before
    // the parked source even gets scheduled to handle the seek.
    let mut stale_after_seek = 0u64;
    loop {
        match sink_handle.pull_buffer().await {
            Pulled::Buffer(b) => {
                if b.metadata().pts.nanos() >= 60_000_000_000 {
                    break;
                }
                stale_after_seek += 1;
                tokio::time::sleep(std::time::Duration::from_millis(2)).await;
            }
            other => panic!("stream ended before the seek landed: {other:?}"),
        }
    }

    // ~22 buffers were queued at seek time; without the epoch every one of
    // them plays out before the seek is visible. With it, at most the
    // sink-side queue (4) plus AppSink's own 2 and a couple in flight can
    // surface — the 16 upstream of the transform are always epoch-dropped.
    assert!(
        stale_after_seek <= 10,
        "queued backlog leaked past the flush: {stale_after_seek} stale buffers"
    );

    handle.stop();
    while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {}
    handle.wait().await.unwrap();

    let mut seek_done = false;
    while let Some(msg) = bus.poll() {
        if matches!(
            msg.kind,
            parallax::pipeline::bus::MessageKind::SeekDone { .. }
        ) {
            seek_done = true;
        }
    }
    assert!(seek_done, "SeekDone posted");
}

/// A seek fanned out to several seekable sources performs exactly ONE
/// epoch transition (the epoch becomes the seek's seqnum via fetch_max) —
/// under the old per-source fetch_add, whichever source handled the seek
/// second stale-stamped the first source's post-seek output, and one
/// branch went silent after every multi-source seek (#162).
#[tokio::test(flavor = "multi_thread")]
async fn multi_source_seek_bumps_epoch_once() {
    use parallax::buffer::{Buffer, MemoryHandle};
    use parallax::element::{ProduceContext, ProduceResult, Source};
    use parallax::elements::{AppSink, Pulled};
    use parallax::event::{Event, EventResult};
    use parallax::memory::SharedArena;
    use parallax::metadata::Metadata;
    use parallax::pipeline::Executor;
    use parallax::pipeline::bus::MessageKind;
    use std::sync::Arc;

    struct SeekableCounter {
        seq: u64,
        arena: Arc<SharedArena>,
    }

    impl Source for SeekableCounter {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> parallax::error::Result<ProduceResult> {
            self.arena.reclaim();
            let Some(slot) = self.arena.acquire() else {
                return Ok(ProduceResult::WouldBlock);
            };
            let mut metadata = Metadata::from_sequence(self.seq);
            metadata.pts = ClockTime::from_millis(self.seq);
            let buffer = Buffer::new(MemoryHandle::new(slot), metadata);
            self.seq += 1;
            Ok(ProduceResult::OwnBuffer(buffer))
        }

        fn is_seekable(&self) -> bool {
            true
        }

        fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
            match event {
                Event::Seek(seek) => {
                    self.seq = (seek.start.position.max(0) as u64) / 1_000_000;
                    EventResult::handled()
                }
                _ => EventResult::NotHandled,
            }
        }
    }

    let mut pipeline = Pipeline::new();
    let mut handles = Vec::new();
    for name in ["a", "b"] {
        let src = pipeline.add_source(
            format!("src_{name}"),
            SeekableCounter {
                seq: 0,
                arena: Arc::new(SharedArena::new(64, 64).unwrap()),
            },
        );
        let sink = AppSink::with_max_buffers(2);
        handles.push(sink.handle());
        let snk = pipeline.add_sink(format!("sink_{name}"), sink);
        pipeline.link(src, snk).unwrap();
    }

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    for h in &handles {
        assert!(matches!(h.pull_buffer().await, Pulled::Buffer(_)));
    }

    assert!(handle.seek_time(ClockTime::from_secs(60)).await);

    // BOTH branches must reach post-seek data: neither source's fresh
    // output may be stale-dropped by the sibling handling the same seek.
    for (i, h) in handles.iter().enumerate() {
        loop {
            match h.pull_buffer().await {
                Pulled::Buffer(b) => {
                    if b.metadata().pts.nanos() >= 60_000_000_000 {
                        break;
                    }
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                other => panic!("branch {i} ended before the seek landed: {other:?}"),
            }
        }
    }

    handle.stop();
    for h in &handles {
        while let Pulled::Buffer(_) = h.pull_buffer().await {}
    }
    handle.wait().await.unwrap();

    // One SeekDone per handling source, all correlated by the same seqnum.
    let mut dones: Vec<(u64, String)> = Vec::new();
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone { seqnum, source, .. } = msg.kind {
            dones.push((seqnum, source));
        }
    }
    assert_eq!(
        dones.len(),
        2,
        "one SeekDone per handling source: {dones:?}"
    );
    assert_eq!(dones[0].0, dones[1].0, "same seek, same seqnum");
    assert_ne!(dones[0].1, dones[1].1, "distinct handling sources");
}

/// The running handle answers seekability and duration from the snapshot
/// taken at start — the runtime counterpart of `Pipeline::query_*` (#162).
#[tokio::test(flavor = "multi_thread")]
async fn handle_reports_seekability_and_duration() {
    use parallax::elements::NullSink;

    // FileSrc pipeline: seekable, byte-format range = file length.
    let content = vec![0u8; 4096];
    let temp = create_test_file(&content);
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::open(temp.path()).unwrap());
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let executor = parallax::pipeline::Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    assert!(handle.seekable());
    let q = handle.query_seekable();
    assert!(q.seekable);
    assert_eq!(q.stop, 4096, "byte range bounded by the file length");
    // FileSrc's duration is Bytes-format, so the Time-format duration is NONE.
    assert_eq!(handle.duration(), ClockTime::NONE);
    assert_eq!(
        handle.query_duration().map(|d| (d.format, d.duration)),
        Some((SegmentFormat::Bytes, Some(4096)))
    );
    handle.stop();
    handle.wait().await.unwrap();

    // NullSource pipeline: nothing seekable, no duration.
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", parallax::elements::NullSource::new(4));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();
    let executor = parallax::pipeline::Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    assert!(!handle.seekable());
    assert!(!handle.query_seekable().seekable);
    assert_eq!(handle.duration(), ClockTime::NONE);
    assert_eq!(handle.query_duration(), None);
    handle.wait().await.unwrap();
}

#[test]
fn test_pipeline_seek_bytes() {
    let content = b"ABCDEFGHIJKLMNOP";
    let temp = create_test_file(content);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::open(temp.path()).unwrap());
    let sink = pipeline.add_sink("sink", parallax::elements::NullSink::new());
    pipeline.link(src, sink).unwrap();

    // Seek to byte 8
    let handled = pipeline.seek_bytes(8).unwrap();
    assert!(handled);

    // Position should reflect the seek
    let pos = pipeline.query_position().unwrap();
    assert_eq!(pos.position, Some(8));
}
