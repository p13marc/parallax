//! Regression tests: an upstream failure must reach the app boundary *as a
//! failure*.
//!
//! Two properties, and the second one is #85. A dead element must terminate the
//! sinks below it rather than wedging the pipeline (the failure mode behind a
//! live camera source dying mid-stream) — and the consumer must be able to tell
//! that apart from the stream simply ending. This file used to assert the
//! opposite of the second: an error arrived as a plain EOS, indistinguishable
//! from success.

use std::time::Duration;

use parallax::buffer::Buffer;
use parallax::element::{Element, ProduceContext, ProduceResult, Source};
use parallax::elements::{AppSink, AppSinkHandle, EndReason, NullSource};
use parallax::error::{Error, Result};
use parallax::pipeline::{Executor, Pipeline};

struct FailingSource;

impl Source for FailingSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        Err(Error::Element("simulated device failure".into()))
    }

    fn name(&self) -> &str {
        "failing-source"
    }
}

struct FailingTransform;

impl Element for FailingTransform {
    fn process(&mut self, _buffer: Buffer) -> Result<Option<Buffer>> {
        Err(Error::Element("simulated encoder failure".into()))
    }
}

/// The reason the sink saw, or a failure if it never saw one.
///
/// No polling loop: `ended()` resolves when the stream is over and drained.
/// Deleting the 10-second poll that used to live here is itself part of the
/// point — it existed only because EOS was the sole observable signal.
async fn end_reason(handle: &AppSinkHandle, what: &str) -> EndReason {
    tokio::time::timeout(Duration::from_secs(10), handle.ended())
        .await
        .unwrap_or_else(|_| panic!("sink never terminated after the {what} error"))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_source_error_reaches_the_sink_as_an_error() {
    let sink = AppSink::new();
    let handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FailingSource);
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    match end_reason(&handle, "source").await {
        EndReason::Error(err) => {
            assert_eq!(err.node(), Some("src"));
            assert!(
                err.message().contains("simulated device failure"),
                "the reason was lost on the way down: {err}"
            );
        }
        other => panic!("a dead source must not look like a clean stream end: {other:?}"),
    }
    assert!(
        pipeline_handle.wait().await.is_err(),
        "pipeline should report the source error"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_transform_error_reaches_the_sink_as_an_error() {
    let sink = AppSink::new();
    let handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let xfm = pipeline.add_filter("failing-transform", FailingTransform);
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    match end_reason(&handle, "transform").await {
        EndReason::Error(err) => {
            assert_eq!(err.node(), Some("failing-transform"));
            assert!(
                err.message().contains("simulated encoder failure"),
                "the reason was lost on the way down: {err}"
            );
        }
        other => panic!("a dead transform must not look like a clean stream end: {other:?}"),
    }
    assert!(
        pipeline_handle.wait().await.is_err(),
        "pipeline should report the transform error"
    );
}

/// The other half of the distinction: a stream that really did just end.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_clean_run_still_ends_cleanly() {
    let sink = AppSink::new();
    let handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    let mut seen = 0;
    let reason = loop {
        match handle.pull_buffer().await {
            parallax::elements::Pulled::Buffer(_) => seen += 1,
            parallax::elements::Pulled::Ended(reason) => break reason,
            other => panic!("unexpected outcome: {other:?}"),
        }
    };

    assert_eq!(reason, EndReason::Eos);
    assert_eq!(seen, 5, "buffers must arrive before the end is reported");
    pipeline_handle.wait().await.unwrap();
}
