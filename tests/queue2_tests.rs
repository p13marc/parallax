//! Integration tests for Queue2 network buffering element.

use parallax::elements::flow::Queue2;
use parallax::elements::{NullSink, NullSource};
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::{Executor, Pipeline};

/// Test Queue2 in stream mode within a pipeline.
#[tokio::test]
async fn test_queue2_stream_mode_pipeline() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source("src", NullSource::new(20));
    let queue = Queue2::stream(4096).with_watermarks(10, 90);
    let q = pipeline.add_filter("queue2", queue);
    let sink = pipeline.add_sink("sink", NullSink::new());

    pipeline.link(src, q).unwrap();
    pipeline.link(q, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test Queue2 posts buffering messages to bus.
#[tokio::test]
async fn test_queue2_buffering_messages() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source("src", NullSource::new(10));
    let queue = Queue2::stream(4096).with_watermarks(10, 90);
    let q = pipeline.add_filter("queue2", queue);
    let sink = pipeline.add_sink("sink", NullSink::new());

    pipeline.link(src, q).unwrap();
    pipeline.link(q, sink).unwrap();

    let mut bus = pipeline.take_bus().unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    // Check for buffering messages on bus
    let mut saw_buffering = false;
    while let Some(msg) = bus.poll() {
        if matches!(msg.kind, MessageKind::Buffering { .. }) {
            saw_buffering = true;
        }
    }

    // Queue2 should post buffering messages when starting (in buffering mode)
    assert!(saw_buffering, "Expected buffering messages on bus");
}

/// Test Queue2 statistics.
#[test]
fn test_queue2_stats_available() {
    let queue = Queue2::stream(1024);
    let stats = queue.stats();
    assert_eq!(stats.buffers_in, 0);
    assert_eq!(stats.buffers_out, 0);
    assert!(stats.is_buffering); // Starts in buffering mode
}
