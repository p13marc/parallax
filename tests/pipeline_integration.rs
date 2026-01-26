//! Integration tests for the Parallax pipeline system.

use parallax::element::{DynAsyncElement, ElementAdapter, SinkAdapter, SourceAdapter};
use parallax::elements::{NullSink, NullSource, PassThrough, Tee};
use parallax::pipeline::{Executor, Pipeline};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Test a simple source -> sink pipeline using built-in elements.
#[tokio::test]
async fn test_null_source_to_null_sink() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(100))),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())),
    );

    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test a pipeline with a PassThrough element in the middle.
#[tokio::test]
async fn test_source_passthrough_sink() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(50))),
    );
    let passthrough = pipeline.add_node(
        "passthrough",
        DynAsyncElement::new_box(ElementAdapter::new(PassThrough::new())),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())),
    );

    pipeline.link(src, passthrough).unwrap();
    pipeline.link(passthrough, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test a pipeline with a Tee element for statistics.
#[tokio::test]
async fn test_source_tee_sink() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(25))),
    );
    let tee = pipeline.add_node(
        "tee",
        DynAsyncElement::new_box(ElementAdapter::new(Tee::new())),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())),
    );

    pipeline.link(src, tee).unwrap();
    pipeline.link(tee, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test a longer pipeline with multiple elements.
#[tokio::test]
async fn test_long_pipeline() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(10))),
    );
    let p1 = pipeline.add_node(
        "p1",
        DynAsyncElement::new_box(ElementAdapter::new(PassThrough::new())),
    );
    let p2 = pipeline.add_node(
        "p2",
        DynAsyncElement::new_box(ElementAdapter::new(PassThrough::new())),
    );
    let p3 = pipeline.add_node(
        "p3",
        DynAsyncElement::new_box(ElementAdapter::new(PassThrough::new())),
    );
    let tee = pipeline.add_node(
        "tee",
        DynAsyncElement::new_box(ElementAdapter::new(Tee::new())),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())),
    );

    pipeline.link(src, p1).unwrap();
    pipeline.link(p1, p2).unwrap();
    pipeline.link(p2, p3).unwrap();
    pipeline.link(p3, tee).unwrap();
    pipeline.link(tee, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test that the pipeline correctly counts buffers with a custom sink.
#[tokio::test]
async fn test_buffer_counting() {
    use parallax::buffer::Buffer;
    use parallax::element::Sink;
    use parallax::error::Result;

    struct CountingSink {
        count: Arc<AtomicU64>,
    }

    impl Sink for CountingSink {
        fn consume(&mut self, _buffer: Buffer) -> Result<()> {
            self.count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    let count = Arc::new(AtomicU64::new(0));
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(42))),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            count: count.clone(),
        })),
    );

    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    assert_eq!(count.load(Ordering::Relaxed), 42);
}

/// Test pipeline with a filter element.
#[tokio::test]
async fn test_filter_element() {
    use parallax::buffer::Buffer;
    use parallax::element::{Element, Sink};
    use parallax::error::Result;

    /// Filter that only passes every Nth buffer.
    struct EveryNth {
        n: u64,
        current: u64,
    }

    impl Element for EveryNth {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            self.current += 1;
            if self.current % self.n == 0 {
                Ok(Some(buffer))
            } else {
                Ok(None) // Filter out
            }
        }
    }

    struct CountingSink {
        count: Arc<AtomicU64>,
    }

    impl Sink for CountingSink {
        fn consume(&mut self, _buffer: Buffer) -> Result<()> {
            self.count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    let count = Arc::new(AtomicU64::new(0));
    let mut pipeline = Pipeline::new();

    // Source produces 100 buffers, filter passes every 10th (10 buffers)
    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(100))),
    );
    let filter = pipeline.add_node(
        "filter",
        DynAsyncElement::new_box(ElementAdapter::new(EveryNth { n: 10, current: 0 })),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            count: count.clone(),
        })),
    );

    pipeline.link(src, filter).unwrap();
    pipeline.link(filter, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    assert_eq!(count.load(Ordering::Relaxed), 10);
}

/// Test pipeline validation catches errors.
#[tokio::test]
async fn test_pipeline_validation_errors() {
    use parallax::error::Error;

    // Empty pipeline should fail
    let pipeline = Pipeline::new();
    let result = pipeline.validate();
    assert!(matches!(result, Err(Error::InvalidSegment(_))));

    // Pipeline with only source (no sink) should fail
    let mut pipeline = Pipeline::new();
    pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(10))),
    );
    let result = pipeline.validate();
    assert!(matches!(result, Err(Error::InvalidSegment(_))));

    // Pipeline with only sink (no source) should fail
    let mut pipeline = Pipeline::new();
    pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())),
    );
    let result = pipeline.validate();
    assert!(matches!(result, Err(Error::InvalidSegment(_))));
}

/// Test that cycles are detected and rejected.
#[tokio::test]
async fn test_cycle_detection() {
    let mut pipeline = Pipeline::new();

    let a = pipeline.add_node(
        "a",
        DynAsyncElement::new_box(ElementAdapter::new(PassThrough::new())),
    );
    let b = pipeline.add_node(
        "b",
        DynAsyncElement::new_box(ElementAdapter::new(PassThrough::new())),
    );
    let c = pipeline.add_node(
        "c",
        DynAsyncElement::new_box(ElementAdapter::new(PassThrough::new())),
    );

    // Create a chain: a -> b -> c
    pipeline.link(a, b).unwrap();
    pipeline.link(b, c).unwrap();

    // Attempting to link c -> a would create a cycle
    let result = pipeline.link(c, a);
    assert!(result.is_err());
}

/// Test abort functionality on a long-running pipeline.
#[tokio::test]
async fn test_pipeline_abort() {
    use parallax::buffer::{Buffer, MemoryHandle};
    use parallax::element::Source;
    use parallax::error::Result;
    use parallax::memory::HeapSegment;
    use parallax::metadata::Metadata;

    /// A source that produces buffers forever.
    struct InfiniteSource {
        seq: u64,
    }

    impl Source for InfiniteSource {
        fn produce(&mut self) -> Result<Option<Buffer>> {
            let segment = std::sync::Arc::new(HeapSegment::new(64)?);
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::new(handle, Metadata::from_sequence(self.seq));
            self.seq += 1;
            Ok(Some(buffer))
        }
    }

    let count = Arc::new(AtomicU64::new(0));
    let mut pipeline = Pipeline::new();

    struct CountingSink {
        count: Arc<AtomicU64>,
    }

    impl parallax::element::Sink for CountingSink {
        fn consume(&mut self, _buffer: Buffer) -> Result<()> {
            self.count.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(InfiniteSource { seq: 0 })),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
            count: count.clone(),
        })),
    );

    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Let it run for a bit
    tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;

    // Should have processed some buffers
    let processed = count.load(Ordering::Relaxed);
    assert!(processed > 0, "Expected some buffers to be processed");

    // Abort the pipeline
    handle.abort();
}

/// Test configurable channel capacity.
#[tokio::test]
async fn test_executor_config() {
    use parallax::pipeline::UnifiedExecutorConfig;

    let config = UnifiedExecutorConfig::default().with_channel_capacity(64);
    assert_eq!(config.channel_capacity, 64);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(10))),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())),
    );
    pipeline.link(src, sink).unwrap();

    let executor = Executor::with_config(config);
    executor.run(&mut pipeline).await.unwrap();
}

// ============================================================================
// Parser Integration Tests
// ============================================================================

/// Test parsing and running a simple pipeline from a string.
#[tokio::test]
async fn test_parse_simple_pipeline() {
    let mut pipeline = Pipeline::parse("nullsource count=10 ! nullsink").unwrap();
    assert_eq!(pipeline.node_count(), 2);

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test parsing a pipeline with a transform element.
#[tokio::test]
async fn test_parse_pipeline_with_transform() {
    let mut pipeline = Pipeline::parse("nullsource count=5 ! passthrough ! nullsink").unwrap();
    assert_eq!(pipeline.node_count(), 3);

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test parsing a pipeline with multiple transforms.
#[tokio::test]
async fn test_parse_long_pipeline() {
    let mut pipeline =
        Pipeline::parse("nullsource count=3 ! passthrough ! tee ! passthrough ! nullsink").unwrap();
    assert_eq!(pipeline.node_count(), 5);

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test parsing with different property types.
#[tokio::test]
async fn test_parse_with_properties() {
    let mut pipeline = Pipeline::parse("nullsource count=100 buffer-size=1024 ! nullsink").unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test that parsing unknown elements fails.
#[test]
fn test_parse_unknown_element_fails() {
    let result = Pipeline::parse("unknown_element ! nullsink");
    assert!(result.is_err());
}

/// Test that parsing empty string fails.
#[test]
fn test_parse_empty_fails() {
    let result = Pipeline::parse("");
    assert!(result.is_err());
}

/// Test that parsing filesrc without location fails.
#[test]
fn test_parse_filesrc_without_location_fails() {
    let result = Pipeline::parse("filesrc ! nullsink");
    assert!(result.is_err());
}

/// Test roundtrip with files using parsed pipeline.
#[tokio::test]
async fn test_parse_file_pipeline() {
    use std::io::Write;
    use tempfile::NamedTempFile;

    // Create input file
    let mut input = NamedTempFile::new().unwrap();
    let data = b"Test data for pipeline";
    input.write_all(data).unwrap();
    input.flush().unwrap();

    // Create output file
    let output = NamedTempFile::new().unwrap();

    // Build pipeline string
    let pipeline_str = format!(
        "filesrc location='{}' ! filesink location='{}'",
        input.path().display(),
        output.path().display()
    );

    let mut pipeline = Pipeline::parse(&pipeline_str).unwrap();
    assert_eq!(pipeline.node_count(), 2);

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    // Verify output
    let output_data = std::fs::read(output.path()).unwrap();
    assert_eq!(output_data, data);
}

// ============================================================================
// Typed Pipeline Integration Tests
// ============================================================================

/// Test typed pipeline with map operator.
#[test]
fn test_typed_pipeline_map() {
    use parallax::typed::{collect, from_iter, map, pipeline};

    let source = from_iter(vec![1i32, 2, 3, 4, 5]);
    let result = pipeline(source)
        .then(map(|x: i32| x * x))
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![1, 4, 9, 16, 25]);
}

/// Test typed pipeline with filter operator.
#[test]
fn test_typed_pipeline_filter() {
    use parallax::typed::{collect, filter, from_iter, pipeline};

    let source = from_iter(1..=10);
    let result = pipeline(source)
        .then(filter(|x: &i32| x % 3 == 0))
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![3, 6, 9]);
}

/// Test typed pipeline with chained operators.
#[test]
fn test_typed_pipeline_chain() {
    use parallax::typed::{collect, filter, from_iter, map, pipeline};

    let source = from_iter(1..=20);
    let result = pipeline(source)
        .then(filter(|x: &i32| x % 2 == 0)) // Keep even: 2,4,6,...,20
        .then(map(|x: i32| x / 2)) // Divide by 2: 1,2,3,...,10
        .then(filter(|x: &i32| *x > 5)) // Keep > 5: 6,7,8,9,10
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![6, 7, 8, 9, 10]);
}

/// Test typed pipeline with take operator.
#[test]
fn test_typed_pipeline_take() {
    use parallax::typed::{collect, from_iter, pipeline, take};

    let source = from_iter(1..=100);
    let result = pipeline(source)
        .then(take(5))
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![1, 2, 3, 4, 5]);
}

/// Test typed pipeline with skip operator.
#[test]
fn test_typed_pipeline_skip() {
    use parallax::typed::{collect, from_iter, pipeline, skip};

    let source = from_iter(1..=10);
    let result = pipeline(source)
        .then(skip(7))
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![8, 9, 10]);
}

/// Test typed pipeline with filter_map operator.
#[test]
fn test_typed_pipeline_filter_map() {
    use parallax::typed::{collect, filter_map, from_iter, pipeline};

    let source = from_iter(vec!["1", "two", "3", "four", "5"]);
    let result = pipeline(source)
        .then(filter_map(|s: &str| s.parse::<i32>().ok()))
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![1, 3, 5]);
}

/// Test typed pipeline with >> operator.
#[test]
fn test_typed_pipeline_shr_operator() {
    use parallax::typed::{collect, from_iter, map, pipeline};

    let source = from_iter(vec![1, 2, 3]);
    let result = (pipeline(source) >> map(|x: i32| x + 100))
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![101, 102, 103]);
}

/// Test typed pipeline with multiple >> operators.
#[test]
fn test_typed_pipeline_shr_chain() {
    use parallax::typed::{collect, filter, from_iter, map, pipeline};

    let source = from_iter(1..=10);
    let result = (pipeline(source) >> filter(|x: &i32| x % 2 == 0) >> map(|x: i32| x * 10))
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![20, 40, 60, 80, 100]);
}

/// Test typed pipeline with discard sink.
#[test]
fn test_typed_pipeline_discard() {
    use parallax::typed::{discard, from_iter, map, pipeline};

    let source = from_iter(1..=1000);
    let _ = pipeline(source)
        .then(map(|x: i32| x * 2))
        .sink(discard())
        .run()
        .unwrap();
    // Just verify it completes without error
}

/// Test typed pipeline with for_each sink.
#[test]
fn test_typed_pipeline_for_each() {
    use parallax::typed::{for_each, from_iter, pipeline};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicI32, Ordering};

    let sum = Arc::new(AtomicI32::new(0));
    let sum_clone = sum.clone();

    let source = from_iter(1..=10);
    let _ = pipeline(source)
        .sink(for_each(move |x: i32| {
            sum_clone.fetch_add(x, Ordering::Relaxed);
        }))
        .run()
        .unwrap();

    // 1+2+3+...+10 = 55
    assert_eq!(sum.load(Ordering::Relaxed), 55);
}

/// Test typed pipeline type safety (compile-time check).
#[test]
fn test_typed_pipeline_type_conversion() {
    use parallax::typed::{collect, from_iter, map, pipeline};

    // i32 -> String -> usize pipeline
    let source = from_iter(vec![1i32, 22, 333]);
    let result = pipeline(source)
        .then(map(|x: i32| x.to_string()))
        .then(map(|s: String| s.len()))
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![1, 2, 3]);
}

/// Test typed pipeline with once source.
#[test]
fn test_typed_pipeline_once() {
    use parallax::typed::{collect, map, once, pipeline};

    let source = once(42i32);
    let result = pipeline(source)
        .then(map(|x: i32| x * 2))
        .sink(collect())
        .run()
        .unwrap();

    assert_eq!(result.into_inner(), vec![84]);
}

// ============================================================================
// Phase 5: Graph Export and Event System Tests
// ============================================================================

/// Test DOT graph export.
#[test]
fn test_pipeline_to_dot() {
    use parallax::pipeline::Pipeline;

    let pipeline = Pipeline::parse("nullsource count=5 ! passthrough ! nullsink").unwrap();
    let dot = pipeline.to_dot();

    // Check that the DOT output contains expected elements
    assert!(dot.contains("digraph pipeline"));
    assert!(dot.contains("rankdir=LR"));
    assert!(dot.contains("nullsource_0"));
    assert!(dot.contains("passthrough_1"));
    assert!(dot.contains("nullsink_2"));
    assert!(dot.contains("->")); // Edge connections
}

/// Test DOT graph export with options.
#[test]
fn test_pipeline_to_dot_with_options() {
    use parallax::pipeline::{DotOptions, Pipeline};

    let pipeline = Pipeline::parse("nullsource ! nullsink").unwrap();

    // Test verbose options
    let dot_verbose = pipeline.to_dot_with_options(&DotOptions::verbose());
    assert!(dot_verbose.contains("Legend"));
    assert!(dot_verbose.contains("src -> sink")); // Pad names shown

    // Test minimal options
    let dot_minimal = pipeline.to_dot_with_options(&DotOptions::minimal());
    assert!(!dot_minimal.contains("Legend"));
}

/// Test JSON graph export.
#[test]
fn test_pipeline_to_json() {
    use parallax::pipeline::Pipeline;

    let pipeline = Pipeline::parse("nullsource count=10 ! passthrough ! nullsink").unwrap();
    let json = pipeline.to_json();

    // Check that the JSON output contains expected elements
    assert!(json.contains("\"state\": \"Suspended\""));
    assert!(json.contains("\"node_count\": 3"));
    assert!(json.contains("\"edge_count\": 2"));
    assert!(json.contains("\"nodes\": ["));
    assert!(json.contains("\"edges\": ["));
    assert!(json.contains("\"name\": \"nullsource_0\""));
    assert!(json.contains("\"type\": \"Source\""));
}

/// Test pipeline event system.
#[tokio::test]
async fn test_pipeline_events() {
    use parallax::element::{SinkAdapter, SourceAdapter};
    use parallax::elements::{NullSink, NullSource};
    use parallax::pipeline::{Executor, Pipeline, PipelineEvent};

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(10))),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())),
    );
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Subscribe to events - note that some early events (Started, NodeStarted)
    // may have already been emitted before we subscribe, since the broadcast
    // channel drops messages for late subscribers.
    let mut receiver = handle.subscribe();

    // Wait for the pipeline to complete
    handle.wait().await.unwrap();

    // Check that we received the events that occur after subscription.
    // EOS is emitted in wait() after all tasks complete, so it should always be received.
    // NodeFinished events may or may not be received depending on timing.
    let mut node_finished_count = 0;
    let mut eos_count = 0;

    while let Some(event) = receiver.try_recv() {
        match event {
            PipelineEvent::NodeFinished { .. } => node_finished_count += 1,
            PipelineEvent::Eos => eos_count += 1,
            _ => {}
        }
    }

    // EOS should always be received since it's emitted after wait() tasks join
    assert_eq!(eos_count, 1, "Expected 1 Eos event");
    // NodeFinished events should be received (emitted before tasks join)
    assert_eq!(node_finished_count, 2, "Expected 2 NodeFinished events");
}

/// Test wait_eos on event receiver.
#[tokio::test]
async fn test_event_wait_eos() {
    use parallax::element::{SinkAdapter, SourceAdapter};
    use parallax::elements::{NullSink, NullSource};
    use parallax::pipeline::{Executor, Pipeline};

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(5))),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())),
    );
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let mut receiver = handle.subscribe();

    // Wait for EOS using the convenience method
    let wait_handle = tokio::spawn(async move { receiver.wait_eos().await });

    // Wait for the pipeline
    handle.wait().await.unwrap();

    // The wait_eos should have completed successfully
    let result = wait_handle.await.unwrap();
    assert!(result.is_ok());
}

/// Test pipeline state transitions.
#[tokio::test]
async fn test_pipeline_state_transitions() {
    use parallax::element::{SinkAdapter, SourceAdapter};
    use parallax::elements::{NullSink, NullSource};
    use parallax::pipeline::{Executor, Pipeline, PipelineState};

    let mut pipeline = Pipeline::new();

    // Initially suspended
    assert_eq!(pipeline.state(), PipelineState::Suspended);

    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(NullSource::new(10))),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(NullSink::new())),
    );
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Should be running now
    assert_eq!(pipeline.state(), PipelineState::Running);

    handle.wait().await.unwrap();
}

/// Test pipeline abort emits stopped event.
#[tokio::test]
async fn test_pipeline_abort_event() {
    use parallax::buffer::{Buffer, MemoryHandle};
    use parallax::element::{SinkAdapter, Source, SourceAdapter};
    use parallax::memory::HeapSegment;
    use parallax::metadata::Metadata;
    use parallax::pipeline::{Executor, Pipeline, PipelineEvent};

    /// A source that produces buffers forever.
    struct InfiniteSource;

    impl Source for InfiniteSource {
        fn produce(&mut self) -> parallax::error::Result<Option<Buffer>> {
            let segment = std::sync::Arc::new(HeapSegment::new(64)?);
            let handle = MemoryHandle::from_segment(segment);
            Ok(Some(Buffer::new(handle, Metadata::default())))
        }
    }

    struct DiscardSink;
    impl parallax::element::Sink for DiscardSink {
        fn consume(&mut self, _buffer: Buffer) -> parallax::error::Result<()> {
            Ok(())
        }
    }

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node(
        "src",
        DynAsyncElement::new_box(SourceAdapter::new(InfiniteSource)),
    );
    let sink = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(DiscardSink)),
    );
    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let mut receiver = handle.subscribe();

    // Let it run briefly
    tokio::time::sleep(tokio::time::Duration::from_millis(10)).await;

    // Abort the pipeline
    handle.abort();

    // Check for Stopped event
    let mut found_stopped = false;
    while let Some(event) = receiver.try_recv() {
        if matches!(event, PipelineEvent::Stopped) {
            found_stopped = true;
            break;
        }
    }

    assert!(found_stopped, "Expected Stopped event after abort");
}

/// Test dynamic pipeline builder from typed module.
#[test]
fn test_dynamic_pipeline_builder_creation() {
    use parallax::typed::bridge::DynamicPipelineBuilder;

    let builder = DynamicPipelineBuilder::new();
    let pipeline = builder.build();

    assert_eq!(pipeline.node_count(), 0);
    assert!(pipeline.is_empty());
}

// ============================================================================
// Phase 6: Plugin System Tests
// ============================================================================

/// Test plugin registry creation.
#[test]
fn test_plugin_registry_creation() {
    use parallax::plugin::PluginRegistry;

    let registry = PluginRegistry::new();
    assert!(registry.list_plugins().is_empty());
    assert!(registry.list_elements().is_empty());
}

/// Test plugin registry with custom search paths.
#[test]
fn test_plugin_registry_search_paths() {
    use parallax::plugin::PluginRegistry;

    let registry = PluginRegistry::with_search_paths(["/custom/path", "/another/path"]);
    assert!(registry.list_plugins().is_empty());
}

/// Test plugin loader creation.
#[test]
fn test_plugin_loader_creation() {
    use parallax::plugin::PluginLoader;

    let loader = PluginLoader::new();
    // Loader should be created successfully
    let _ = loader;
}

/// Test loading nonexistent plugin fails gracefully.
#[test]
fn test_load_nonexistent_plugin() {
    use parallax::plugin::{PluginError, PluginLoader};

    let loader = PluginLoader::new();
    let result = unsafe { loader.load_by_name("nonexistent_plugin_that_does_not_exist") };
    assert!(matches!(result, Err(PluginError::LoadFailed(_))));
}

/// Test element factory lists built-in elements.
#[test]
fn test_element_factory_list_elements() {
    use parallax::pipeline::ElementFactory;

    let factory = ElementFactory::new();
    let elements = factory.list_elements();

    assert!(elements.contains(&"nullsource".to_string()));
    assert!(elements.contains(&"nullsink".to_string()));
    assert!(elements.contains(&"passthrough".to_string()));
    assert!(elements.contains(&"tee".to_string()));
    assert!(elements.contains(&"filesrc".to_string()));
    assert!(elements.contains(&"filesink".to_string()));
}

/// Test element factory with plugin registry.
#[test]
fn test_element_factory_with_plugin_registry() {
    use parallax::pipeline::ElementFactory;
    use parallax::plugin::PluginRegistry;
    use std::sync::Arc;

    let registry = Arc::new(PluginRegistry::new());
    let factory = ElementFactory::with_plugin_registry(registry);

    // Built-in elements should still be available
    assert!(factory.is_registered("nullsource"));
    assert!(factory.is_registered("nullsink"));
}

/// Test plugin ABI version constant.
#[test]
fn test_plugin_abi_version() {
    use parallax::plugin::PARALLAX_ABI_VERSION;

    // ABI version should be 1 for this initial implementation
    assert_eq!(PARALLAX_ABI_VERSION, 1);
}

/// Test plugin descriptor struct sizes are non-zero.
#[test]
fn test_plugin_descriptor_sizes() {
    use parallax::plugin::{ElementDescriptor, PluginDescriptor};

    assert!(std::mem::size_of::<PluginDescriptor>() > 0);
    assert!(std::mem::size_of::<ElementDescriptor>() > 0);
}

/// Test creating element not in registry fails.
#[test]
fn test_create_element_not_in_registry() {
    use parallax::plugin::{PluginError, PluginRegistry};

    let registry = PluginRegistry::new();
    let result = registry.create_element("nonexistent");
    assert!(matches!(result, Err(PluginError::ElementNotFound(_))));
}

/// Test registry has_element returns false for unknown elements.
#[test]
fn test_registry_has_element() {
    use parallax::plugin::PluginRegistry;

    let registry = PluginRegistry::new();
    assert!(!registry.has_element("nonexistent"));
    assert!(!registry.has_element("doubler"));
}

/// Test unloading nonexistent plugin returns false.
#[test]
fn test_unload_nonexistent_plugin() {
    use parallax::plugin::PluginRegistry;

    let registry = PluginRegistry::new();
    assert!(!registry.unload_plugin("nonexistent"));
}

// ============================================================================
// Phase 7: Multi-source Fusion and Temporal Types Tests
// ============================================================================

/// Test timestamp creation and arithmetic.
#[test]
fn test_timestamp_basics() {
    use parallax::temporal::Timestamp;
    use std::time::Duration;

    let ts1 = Timestamp::from_millis(1000);
    let ts2 = Timestamp::from_millis(1500);

    assert_eq!(ts1.as_millis(), 1000);
    assert_eq!(ts1.as_secs(), 1);

    // Arithmetic
    let ts3 = ts1 + Duration::from_millis(500);
    assert_eq!(ts3.as_millis(), 1500);

    // Difference
    let diff: Duration = ts2 - ts1;
    assert_eq!(diff, Duration::from_millis(500));
}

/// Test timestamp ordering.
#[test]
fn test_timestamp_ordering() {
    use parallax::temporal::Timestamp;

    let t1 = Timestamp::from_millis(100);
    let t2 = Timestamp::from_millis(200);
    let t3 = Timestamp::from_millis(100);

    assert!(t1 < t2);
    assert!(t2 > t1);
    assert_eq!(t1, t3);
}

/// Test time range operations.
#[test]
fn test_time_range() {
    use parallax::temporal::{TimeRange, Timestamp};

    let range = TimeRange::new(Timestamp::from_secs(10), Timestamp::from_secs(20));

    assert!(range.contains(Timestamp::from_secs(15)));
    assert!(!range.contains(Timestamp::from_secs(5)));
    assert!(!range.contains(Timestamp::from_secs(20))); // exclusive end
}

/// Test time range overlap detection.
#[test]
fn test_time_range_overlap() {
    use parallax::temporal::{TimeRange, Timestamp};

    let r1 = TimeRange::new(Timestamp::from_secs(10), Timestamp::from_secs(20));
    let r2 = TimeRange::new(Timestamp::from_secs(15), Timestamp::from_secs(25));
    let r3 = TimeRange::new(Timestamp::from_secs(25), Timestamp::from_secs(30));

    assert!(r1.overlaps(&r2));
    assert!(!r1.overlaps(&r3));

    let intersection = r1.intersection(&r2).unwrap();
    assert_eq!(intersection.start, Timestamp::from_secs(15));
    assert_eq!(intersection.end, Timestamp::from_secs(20));
}

/// Test clock source types.
#[test]
fn test_clock_source() {
    use parallax::temporal::{ClockSource, Timestamp};

    let ts = Timestamp::from_nanos_with_source(1000, ClockSource::WallClock);
    assert_eq!(ts.source(), ClockSource::WallClock);

    let ts2 = ts.with_source(ClockSource::Monotonic);
    assert_eq!(ts2.source(), ClockSource::Monotonic);
}

/// Test merge operator combines two sources.
#[test]
fn test_typed_merge() {
    use parallax::typed::{collect, from_iter, merge, pipeline};

    let left = from_iter(vec![1, 3, 5]);
    let right = from_iter(vec![2, 4, 6]);
    let merged = merge(left, right);

    let result = pipeline(merged).sink(collect()).run().unwrap();
    let items = result.into_inner();

    // Should have all items (order depends on alternation)
    assert_eq!(items.len(), 6);
    assert!(items.contains(&1));
    assert!(items.contains(&2));
    assert!(items.contains(&5));
    assert!(items.contains(&6));
}

/// Test zip operator pairs items from two sources.
#[test]
fn test_typed_zip() {
    use parallax::typed::{collect, from_iter, pipeline, zip};

    let left = from_iter(vec!["a", "b", "c"]);
    let right = from_iter(vec![1, 2, 3]);
    let zipped = zip(left, right);

    let result = pipeline(zipped).sink(collect()).run().unwrap();
    let items = result.into_inner();

    assert_eq!(items, vec![("a", 1), ("b", 2), ("c", 3)]);
}

/// Test zip stops at shorter source.
#[test]
fn test_typed_zip_unequal() {
    use parallax::typed::{collect, from_iter, pipeline, zip};

    let left = from_iter(vec![1, 2, 3, 4, 5]);
    let right = from_iter(vec![10, 20]);
    let zipped = zip(left, right);

    let result = pipeline(zipped).sink(collect()).run().unwrap();
    let items = result.into_inner();

    assert_eq!(items.len(), 2);
    assert_eq!(items, vec![(1, 10), (2, 20)]);
}

/// Test join operator joins by key.
#[test]
fn test_typed_join() {
    use parallax::typed::{collect, from_iter, join, pipeline};

    #[derive(Clone, Debug, PartialEq)]
    struct User {
        id: i32,
        name: String,
    }
    #[derive(Clone, Debug, PartialEq)]
    struct Order {
        user_id: i32,
        amount: i32,
    }

    let users = from_iter(vec![
        User {
            id: 1,
            name: "Alice".to_string(),
        },
        User {
            id: 2,
            name: "Bob".to_string(),
        },
    ]);
    let orders = from_iter(vec![
        Order {
            user_id: 2,
            amount: 100,
        },
        Order {
            user_id: 1,
            amount: 50,
        },
    ]);

    let joined = join(users, orders, |u| u.id, |o| o.user_id);
    let result = pipeline(joined).sink(collect()).run().unwrap();
    let items = result.into_inner();

    assert_eq!(items.len(), 2);
    // Both users should be joined with their orders
    assert!(
        items
            .iter()
            .any(|(u, o)| u.name == "Alice" && o.amount == 50)
    );
    assert!(
        items
            .iter()
            .any(|(u, o)| u.name == "Bob" && o.amount == 100)
    );
}

/// Test temporal join with exact matching.
#[test]
fn test_typed_temporal_join() {
    use parallax::typed::{Timestamped, collect, from_iter, pipeline, temporal_join};
    use std::time::Duration;

    let sensor1 = from_iter(vec![
        Timestamped::from_millis(100, "A"),
        Timestamped::from_millis(200, "B"),
        Timestamped::from_millis(300, "C"),
    ]);
    let sensor2 = from_iter(vec![
        Timestamped::from_millis(100, 1),
        Timestamped::from_millis(200, 2),
        Timestamped::from_millis(300, 3),
    ]);

    let joined = temporal_join(
        sensor1,
        sensor2,
        Duration::from_millis(10),
        |s| s.timestamp,
        |s| s.timestamp,
    );

    let result = pipeline(joined).sink(collect()).run().unwrap();
    let items = result.into_inner();

    assert_eq!(items.len(), 3);
    // All should be matched
    assert!(items.iter().any(|(a, b)| a.value == "A" && b.value == 1));
    assert!(items.iter().any(|(a, b)| a.value == "B" && b.value == 2));
    assert!(items.iter().any(|(a, b)| a.value == "C" && b.value == 3));
}

/// Test temporal join with tolerance.
#[test]
fn test_typed_temporal_join_tolerance() {
    use parallax::typed::{Timestamped, collect, from_iter, pipeline, temporal_join};
    use std::time::Duration;

    // Timestamps are slightly off but within tolerance
    let sensor1 = from_iter(vec![
        Timestamped::from_millis(100, "A"),
        Timestamped::from_millis(200, "B"),
    ]);
    let sensor2 = from_iter(vec![
        Timestamped::from_millis(95, 1),  // 5ms before A
        Timestamped::from_millis(210, 2), // 10ms after B
    ]);

    let joined = temporal_join(
        sensor1,
        sensor2,
        Duration::from_millis(15), // 15ms tolerance
        |s| s.timestamp,
        |s| s.timestamp,
    );

    let result = pipeline(joined).sink(collect()).run().unwrap();
    let items = result.into_inner();

    assert_eq!(items.len(), 2);
}

/// Test join window configuration.
#[test]
fn test_join_window_config() {
    use parallax::temporal::{AlignmentStrategy, JoinWindow};
    use std::time::Duration;

    let window = JoinWindow::with_max_delay(Duration::from_millis(50))
        .with_strategy(AlignmentStrategy::Exact)
        .with_max_buffer_size(32);

    assert_eq!(window.max_delay, Duration::from_millis(50));
    assert_eq!(window.strategy, AlignmentStrategy::Exact);
    assert_eq!(window.max_buffer_size, 32);
}

/// Test temporal join operators.
#[test]
fn test_temporal_join_direct() {
    use parallax::temporal::{JoinResult, JoinWindow, TemporalJoin, Timestamp};

    let mut joiner: TemporalJoin<i32, i32> = TemporalJoin::with_config(JoinWindow::default());

    joiner.push_left(Timestamp::from_millis(100), 1);
    joiner.push_right(Timestamp::from_millis(100), 10);

    match joiner.try_emit() {
        Some(JoinResult::Matched(a, b)) => {
            assert_eq!(a, 1);
            assert_eq!(b, 10);
        }
        other => panic!("Expected Matched, got {:?}", other),
    }
}

/// Test alignment strategies.
#[test]
fn test_alignment_strategies() {
    use parallax::temporal::AlignmentStrategy;
    use std::time::Duration;

    let exact = AlignmentStrategy::Exact;
    let tolerance = AlignmentStrategy::Tolerance(Duration::from_millis(10));
    let nearest = AlignmentStrategy::Nearest(Duration::from_millis(50));

    assert_eq!(exact, AlignmentStrategy::Exact);
    assert_eq!(
        tolerance,
        AlignmentStrategy::Tolerance(Duration::from_millis(10))
    );
    assert_eq!(
        nearest,
        AlignmentStrategy::Nearest(Duration::from_millis(50))
    );
}
