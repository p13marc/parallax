//! Regression tests: a source or transform error must propagate EOS
//! downstream so sinks terminate instead of waiting forever on a wedged
//! pipeline (the failure mode behind a live camera source dying mid-stream).

use std::time::{Duration, Instant};

use parallax::buffer::Buffer;
use parallax::element::{Element, ProduceContext, ProduceResult, Source};
use parallax::elements::{AppSink, AppSinkHandle, NullSource};
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

async fn wait_for_eos(handle: &AppSinkHandle, what: &str) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !handle.is_eos() {
        assert!(
            Instant::now() < deadline,
            "sink never reached EOS after {what} error"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn source_error_propagates_eos_to_sink() {
    let sink = AppSink::new();
    let handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FailingSource);
    let snk = pipeline.add_sink("sink", sink);
    pipeline.link(src, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    wait_for_eos(&handle, "source").await;
    assert!(
        pipeline_handle.wait().await.is_err(),
        "pipeline should report the source error"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn transform_error_propagates_eos_to_sink() {
    let sink = AppSink::new();
    let handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let xfm = pipeline.add_filter("failing-transform", FailingTransform);
    let snk = pipeline.add_sink("sink", sink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::new();
    let pipeline_handle = executor.start(&mut pipeline).unwrap();

    wait_for_eos(&handle, "transform").await;
    assert!(
        pipeline_handle.wait().await.is_err(),
        "pipeline should report the transform error"
    );
}
