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
    assert_eq!(seg.to_running_time(ClockTime::from_secs(5)), ClockTime::from_secs(5));
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
    assert_eq!(seg.to_running_time(ClockTime::from_secs(10)), ClockTime::from_secs(5));
    // PTS=15s → running_time = (15-10)/1 + 5 = 10s
    assert_eq!(seg.to_running_time(ClockTime::from_secs(15)), ClockTime::from_secs(10));
}

#[test]
fn test_segment_running_time_double_speed() {
    let seg = SegmentEvent::new_time(ClockTime::ZERO, None).with_rate(2.0);
    // At 2x speed, 10s of content plays in 5s of real time
    assert_eq!(seg.to_running_time(ClockTime::from_secs(10)), ClockTime::from_secs(5));
}

#[test]
fn test_segment_running_time_half_speed() {
    let seg = SegmentEvent::new_time(ClockTime::ZERO, None).with_rate(0.5);
    // At 0.5x speed, 10s of content plays in 20s of real time
    assert_eq!(seg.to_running_time(ClockTime::from_secs(10)), ClockTime::from_secs(20));
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
    assert_eq!(seg.to_running_time(ClockTime::from_secs(5)), ClockTime::NONE);
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
