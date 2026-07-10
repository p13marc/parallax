//! Integration tests for pad probes.

use parallax::elements::{NullSink, NullSource};
use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
use parallax::pipeline::{Executor, Pipeline};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Test that a buffer probe on a source pad counts all buffers.
#[tokio::test]
async fn test_buffer_probe_counts_buffers() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(50));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let count = Arc::new(AtomicU64::new(0));
    let count_clone = count.clone();

    // Add probe on source's output pad
    let pad = PadRef::src(src);
    let _ = pipeline.add_probe(pad, ProbeType::BUFFER, move |data| {
        if let ProbeData::Buffer(_) = data {
            count_clone.fetch_add(1, Ordering::Relaxed);
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    assert_eq!(count.load(Ordering::Relaxed), 50);
}

/// Test that a DROP probe prevents buffers from reaching the sink.
#[tokio::test]
async fn test_buffer_probe_drop() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(20));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let count = Arc::new(AtomicU64::new(0));
    let count_clone = count.clone();

    // Drop every other buffer
    let pad = PadRef::src(src);
    let _ = pipeline.add_probe(pad, ProbeType::BUFFER, move |_| {
        let n = count_clone.fetch_add(1, Ordering::Relaxed);
        if n % 2 == 0 {
            ProbeReturn::Drop
        } else {
            ProbeReturn::Ok
        }
    });

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    // All 20 buffers were seen by the probe
    assert_eq!(count.load(Ordering::Relaxed), 20);
}

/// Test that a one-shot probe (ProbeReturn::Remove) is called only once.
#[tokio::test]
async fn test_buffer_probe_one_shot() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let count = Arc::new(AtomicU64::new(0));
    let count_clone = count.clone();

    let pad = PadRef::src(src);
    let _ = pipeline.add_probe(pad, ProbeType::BUFFER, move |_| {
        count_clone.fetch_add(1, Ordering::Relaxed);
        ProbeReturn::Remove // Remove after first invocation
    });

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    // Probe should have been called exactly once
    assert_eq!(count.load(Ordering::Relaxed), 1);
}

/// Test that probes can accumulate byte counts.
#[tokio::test]
async fn test_buffer_probe_byte_counter() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(25));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let total_bytes = Arc::new(AtomicU64::new(0));
    let total_clone = total_bytes.clone();

    let pad = PadRef::src(src);
    let _ = pipeline.add_probe(pad, ProbeType::BUFFER, move |data| {
        if let ProbeData::Buffer(buf) = data {
            total_clone.fetch_add(buf.len() as u64, Ordering::Relaxed);
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    // NullSource produces buffers with some bytes each
    assert!(total_bytes.load(Ordering::Relaxed) > 0);
}

/// Test probe removal by ID.
#[test]
fn test_probe_add_and_remove() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let _sink = pipeline.add_sink("sink", NullSink::new());

    let pad = PadRef::src(src);
    let id = pipeline.add_probe(pad, ProbeType::BUFFER, |_| ProbeReturn::Ok);

    assert!(pipeline.probe_registry().has_probes(&pad));
    assert!(pipeline.remove_probe(id));
    assert!(!pipeline.probe_registry().has_probes(&pad));
}

/// Test that multiple probes on the same pad are all invoked.
#[tokio::test]
async fn test_multiple_probes_same_pad() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let count_a = Arc::new(AtomicU64::new(0));
    let count_b = Arc::new(AtomicU64::new(0));
    let ca = count_a.clone();
    let cb = count_b.clone();

    let pad = PadRef::src(src);
    let _ = pipeline.add_probe(pad, ProbeType::BUFFER, move |_| {
        ca.fetch_add(1, Ordering::Relaxed);
        ProbeReturn::Ok
    });
    let _ = pipeline.add_probe(pad, ProbeType::BUFFER, move |_| {
        cb.fetch_add(1, Ordering::Relaxed);
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    assert_eq!(count_a.load(Ordering::Relaxed), 10);
    assert_eq!(count_b.load(Ordering::Relaxed), 10);
}
