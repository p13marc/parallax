//! Integration tests for the tracer framework.

use parallax::elements::{NullSink, NullSource};
use parallax::pipeline::tracer::{FramerateTracer, LatencyTracer, TracerRegistry};
use parallax::pipeline::{Executor, Pipeline};

/// Test that the latency tracer collects data during pipeline execution.
#[tokio::test]
async fn test_latency_tracer_pipeline() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(50));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let registry = TracerRegistry::new();
    registry.add(Box::new(LatencyTracer::new()));
    pipeline.set_tracer_registry(registry.clone());

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    let reports = registry.reports();
    // Latency tracer should have at least one report
    assert!(!reports.is_empty(), "Expected latency tracer report");
    let (name, report) = &reports[0];
    assert_eq!(name, "latency");
    assert!(report.contains("Latency Report"), "Report: {report}");
}

/// Test that the framerate tracer collects data during pipeline execution.
#[tokio::test]
async fn test_framerate_tracer_pipeline() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(100));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let registry = TracerRegistry::new();
    registry.add(Box::new(FramerateTracer::new()));
    pipeline.set_tracer_registry(registry.clone());

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    let reports = registry.reports();
    assert!(!reports.is_empty(), "Expected framerate tracer report");
    let (name, report) = &reports[0];
    assert_eq!(name, "framerate");
    assert!(report.contains("buf/s"), "Report: {report}");
}

/// Test multiple tracers running simultaneously.
#[tokio::test]
async fn test_multiple_tracers() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(25));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let registry = TracerRegistry::new();
    registry.add(Box::new(LatencyTracer::new()));
    registry.add(Box::new(FramerateTracer::new()));
    pipeline.set_tracer_registry(registry.clone());

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    let reports = registry.reports();
    assert_eq!(reports.len(), 2, "Expected 2 tracer reports");
}

/// Test pipeline stats snapshot.
#[test]
fn test_pipeline_stats_snapshot() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let filter = pipeline.add_filter("filter", parallax::elements::PassThrough::new());
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, filter).unwrap();
    pipeline.link(filter, sink).unwrap();

    let stats = pipeline.stats_snapshot();
    assert_eq!(stats.element_count, 3);
    assert_eq!(stats.link_count, 2);
    assert_eq!(stats.elements.len(), 3);

    // Check element names
    let names: Vec<&str> = stats.elements.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"src"));
    assert!(names.contains(&"filter"));
    assert!(names.contains(&"sink"));
}

/// Test DOT dump (already existed, verify it still works with new fields).
#[test]
fn test_pipeline_dot_dump() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(10));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let dot = pipeline.to_dot();
    assert!(dot.contains("digraph pipeline"));
    assert!(dot.contains("src"));
    assert!(dot.contains("sink"));
}

/// Test tracer registry is empty by default (no env var).
#[test]
fn test_default_tracer_registry_empty() {
    // SAFETY: Only modifying test-specific env var, no other threads using it
    unsafe {
        std::env::remove_var("PARALLAX_TRACERS");
    }
    let pipeline = Pipeline::new();
    assert!(pipeline.tracer_registry().is_empty());
}

/// #43: a transform must appear in the latency report.
///
/// `spawn_transform_task` took no tracers at all, so `LatencyTracer` could never
/// produce a number for an *encoder* — the one element whose cost you most want
/// to see. The pipeline was observable at its edges and opaque in the middle.
#[tokio::test]
async fn a_transform_appears_in_the_latency_report() {
    use parallax::elements::PassThrough;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(50));
    let filter = pipeline.add_filter("filter", PassThrough::new());
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, filter).unwrap();
    pipeline.link(filter, sink).unwrap();

    let registry = TracerRegistry::new();
    registry.add(Box::new(LatencyTracer::new()));
    pipeline.set_tracer_registry(registry.clone());

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    let (_, report) = &registry.reports()[0];
    assert!(
        report.contains("filter"),
        "the transform must be measured, not just the source and sink. Report:\n{report}"
    );
}

/// A minimal N-to-1 muxer that concatenates whatever it is handed.
struct PassThroughMuxer {
    inputs: Vec<(parallax::element::PadId, parallax::format::Caps)>,
    seen: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl parallax::element::Muxer for PassThroughMuxer {
    fn mux(
        &mut self,
        input: parallax::element::MuxerInput,
    ) -> parallax::error::Result<Option<parallax::buffer::Buffer>> {
        self.seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(Some(input.buffer))
    }

    fn name(&self) -> &str {
        "passthrough_muxer"
    }

    fn inputs(&self) -> &[(parallax::element::PadId, parallax::format::Caps)] {
        &self.inputs
    }

    fn on_pad_added(&mut self, _callback: parallax::element::PadAddedCallback) {}
}

/// #43: a muxer must appear in the latency report.
///
/// `spawn_muxer_task` took no `probe_registry` and never notified the tracer
/// registry — it received `tracers` only to hand to `broadcast()` for drop
/// accounting. So a muxer, which is exactly the element you suspect when
/// interleaving stalls, was unmeasurable.
#[tokio::test]
async fn a_muxer_appears_in_the_latency_report() {
    let mut pipeline = Pipeline::new();
    let a = pipeline.add_source("a", NullSource::new(25));
    let b = pipeline.add_source("b", NullSource::new(25));
    let seen = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let mux = pipeline.add_muxer(
        "mux",
        PassThroughMuxer {
            inputs: Vec::new(),
            seen: seen.clone(),
        },
    );
    let sink = pipeline.add_sink("sink", NullSink::new());
    // Several links into the muxer's one declared sink pad — the same shape
    // `examples/16_mpegts.rs` uses.
    pipeline.link(a, mux).unwrap();
    pipeline.link(b, mux).unwrap();
    pipeline.link(mux, sink).unwrap();

    let registry = TracerRegistry::new();
    registry.add(Box::new(LatencyTracer::new()));
    pipeline.set_tracer_registry(registry.clone());

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    // Two pre-existing bugs in spawn_muxer_task had to be fixed to get here:
    // the dispatch drained the channel map before the Muxer arm could read it
    // per pad (so the muxer received nothing at all), and the receive futures
    // were never re-armed (so it would have got one buffer per input pad).
    assert_eq!(
        seen.load(std::sync::atomic::Ordering::Relaxed),
        50,
        "the muxer must see every buffer from both sources"
    );

    let (_, report) = &registry.reports()[0];
    assert!(
        report.contains("mux"),
        "the muxer must be measured. Report:\n{report}"
    );
}

/// #43: framerate through a transform is measurable too.
#[tokio::test]
async fn a_transform_appears_in_the_framerate_report() {
    use parallax::elements::PassThrough;

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(50));
    let filter = pipeline.add_filter("filter", PassThrough::new());
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, filter).unwrap();
    pipeline.link(filter, sink).unwrap();

    let registry = TracerRegistry::new();
    registry.add(Box::new(FramerateTracer::new()));
    pipeline.set_tracer_registry(registry.clone());

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    let (_, report) = &registry.reports()[0];
    assert!(report.contains("filter"), "Report:\n{report}");
}
