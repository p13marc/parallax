//! Static latency aggregation (#184): elements declare the latency they
//! deliberately introduce; the pipeline sums along each source→sink path
//! and reports the worst path — once on the bus at start, and on demand
//! via `PipelineHandle::latency()`.

use parallax::buffer::Buffer;
use parallax::clock::ClockTime;
use parallax::element::{ConsumeContext, Element, Sink};
use parallax::elements::{NullSink, NullSource};
use parallax::error::Result;
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::{Executor, LatencyRange, Pipeline};

/// Passthrough that declares a fixed latency.
struct Latent(u64);

impl Element for Latent {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        "latent"
    }

    fn latency(&self) -> Option<LatencyRange> {
        Some(LatencyRange::fixed(ClockTime::from_millis(self.0)))
    }
}

/// Sink that declares an up-to latency (a pacing budget).
struct LatentSink(u64);

impl Sink for LatentSink {
    fn consume(&mut self, _ctx: &ConsumeContext<'_>) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "latent_sink"
    }

    fn latency(&self) -> Option<LatencyRange> {
        Some(LatencyRange::up_to(ClockTime::from_millis(self.0)))
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn linear_pipeline_sums_declared_latency() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let mid = pipeline.add_filter("mid", Latent(100));
    let snk = pipeline.add_sink("snk", LatentSink(40));
    pipeline.link(src, mid).unwrap();
    pipeline.link(mid, snk).unwrap();

    // Pre-start query.
    let latency = pipeline.query_latency().expect("declared latency");
    assert_eq!(latency.min, ClockTime::from_millis(100));
    assert_eq!(latency.max, ClockTime::from_millis(140));

    // Snapshot on the handle + a single LatencyChanged on the bus.
    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();
    assert_eq!(handle.latency(), Some(latency));
    handle.wait().await.unwrap();

    let mut posted = Vec::new();
    while let Some(msg) = bus.poll() {
        if let MessageKind::LatencyChanged { min, max } = msg.kind {
            posted.push((min, max));
        }
    }
    assert_eq!(
        posted,
        vec![(ClockTime::from_millis(100), ClockTime::from_millis(140))]
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn diamond_reports_the_worst_path() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let slow = pipeline.add_filter("slow", Latent(100));
    let fast = pipeline.add_filter("fast", Latent(30));
    let snk_a = pipeline.add_sink("snk_a", NullSink::new());
    let snk_b = pipeline.add_sink("snk_b", NullSink::new());
    pipeline.link(src, slow).unwrap();
    pipeline.link(src, fast).unwrap();
    pipeline.link(slow, snk_a).unwrap();
    pipeline.link(fast, snk_b).unwrap();

    let latency = pipeline.query_latency().expect("declared latency");
    assert_eq!(latency.max, ClockTime::from_millis(100), "worst path wins");
}

#[tokio::test(flavor = "multi_thread")]
async fn no_declared_latency_means_none_and_no_message() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let snk = pipeline.add_sink("snk", NullSink::new());
    pipeline.link(src, snk).unwrap();

    assert!(pipeline.query_latency().is_none());

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();
    assert_eq!(handle.latency(), None);
    handle.wait().await.unwrap();

    while let Some(msg) = bus.poll() {
        assert!(
            !matches!(msg.kind, MessageKind::LatencyChanged { .. }),
            "no honest value, no message"
        );
    }
}
