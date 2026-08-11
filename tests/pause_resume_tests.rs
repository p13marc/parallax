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
