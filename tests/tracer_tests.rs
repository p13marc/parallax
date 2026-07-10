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
