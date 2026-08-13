//! #85: an element that panics is reported, not silently orphaned.
//!
//! There was no `catch_unwind` anywhere in the executor. A panicking task never
//! reached its own error arm, so it told nothing downstream and every sink below
//! it waited forever; `wait()` eventually surfaced a bare JoinError as
//! `InvalidSegment("task panicked")`, with no element name, and only if someone
//! called it. That is the failure mode behind the ten-second first-frame
//! watchdog the reporter was carrying downstream.
//!
//! Panics here print their backtrace to stderr — `catch_unwind` does not
//! suppress the default hook, and installing one is process-global and unsafe
//! across parallel tests. The noise is expected.

use std::time::Duration;

use parallax::buffer::Buffer;
use parallax::element::{ConsumeContext, Element, ProduceContext, ProduceResult, Sink, Source};
use parallax::elements::{AppSink, EndReason, NullSource};
use parallax::error::{Error, Result};
use parallax::pipeline::{Executor, Pipeline};

const LIMIT: Duration = Duration::from_secs(10);

struct PanickingSource;

impl Source for PanickingSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        panic!("capture thread fell over");
    }

    fn name(&self) -> &str {
        "panicking-source"
    }
}

struct PanickingTransform;

impl Element for PanickingTransform {
    fn process(&mut self, _buffer: Buffer) -> Result<Option<Buffer>> {
        panic!("encoder hit an impossible state");
    }
}

struct PanickingSink;

impl Sink for PanickingSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        panic!("writer fell over");
    }

    fn name(&self) -> &str {
        "panicking-sink"
    }
}

/// The one the downstream watchdog existed for: a source that dies without a
/// word. The sink must be told, and told *why*.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panicking_source_terminates_the_sink_with_a_reason() {
    let sink = AppSink::new();
    let handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", PanickingSource);
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    let reason = tokio::time::timeout(LIMIT, handle.ended())
        .await
        .expect("the sink was orphaned by the panicking source");

    match reason {
        EndReason::Error(err) => {
            assert_eq!(err.node(), Some("src"));
            assert!(
                err.message().contains("capture thread fell over"),
                "the panic message was lost: {err}"
            );
        }
        other => panic!("a panic must not read as a clean end of stream: {other:?}"),
    }

    let result = tokio::time::timeout(LIMIT, pipeline_handle.wait())
        .await
        .expect("wait() hung");
    match result.expect_err("wait() should report the panic") {
        Error::Panic { node, message } => {
            assert_eq!(node, "src");
            assert!(message.contains("capture thread fell over"));
        }
        other => panic!("a panic should be Error::Panic, got: {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panicking_transform_terminates_the_sink_with_a_reason() {
    let sink = AppSink::new();
    let handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(4));
    let xfm = pipeline.add_filter("enc", PanickingTransform);
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    let reason = tokio::time::timeout(LIMIT, handle.ended())
        .await
        .expect("the sink was orphaned by the panicking transform");

    match reason {
        EndReason::Error(err) => {
            assert_eq!(err.node(), Some("enc"));
            assert!(
                err.message().contains("impossible state"),
                "the panic message was lost: {err}"
            );
        }
        other => panic!("a panic must not read as a clean end of stream: {other:?}"),
    }

    assert!(
        tokio::time::timeout(LIMIT, pipeline_handle.wait())
            .await
            .expect("wait() hung")
            .is_err()
    );
}

/// A panicking sink has nothing downstream, but must still fail the run rather
/// than leaving `wait()` to guess.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_panicking_sink_fails_the_run() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(4));
    let snk = pipeline.add_sink("writer", PanickingSink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let result = tokio::time::timeout(LIMIT, handle.wait())
        .await
        .expect("wait() hung on a panicking sink");

    match result.expect_err("a panicking sink should fail the run") {
        Error::Panic { node, message } => {
            assert_eq!(node, "writer");
            assert!(message.contains("writer fell over"));
        }
        other => panic!("expected Error::Panic, got: {other:?}"),
    }
}
