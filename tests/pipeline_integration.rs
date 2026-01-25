//! Integration tests for the Parallax pipeline system.

use parallax::element::{ElementAdapter, SinkAdapter, SourceAdapter};
use parallax::elements::{NullSink, NullSource, PassThrough, Tee};
use parallax::pipeline::{Pipeline, PipelineExecutor};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// Test a simple source -> sink pipeline using built-in elements.
#[tokio::test]
async fn test_null_source_to_null_sink() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node("src", Box::new(SourceAdapter::new(NullSource::new(100))));
    let sink = pipeline.add_node("sink", Box::new(SinkAdapter::new(NullSink::new())));

    pipeline.link(src, sink).unwrap();

    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test a pipeline with a PassThrough element in the middle.
#[tokio::test]
async fn test_source_passthrough_sink() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node("src", Box::new(SourceAdapter::new(NullSource::new(50))));
    let passthrough = pipeline.add_node(
        "passthrough",
        Box::new(ElementAdapter::new(PassThrough::new())),
    );
    let sink = pipeline.add_node("sink", Box::new(SinkAdapter::new(NullSink::new())));

    pipeline.link(src, passthrough).unwrap();
    pipeline.link(passthrough, sink).unwrap();

    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test a pipeline with a Tee element for statistics.
#[tokio::test]
async fn test_source_tee_sink() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node("src", Box::new(SourceAdapter::new(NullSource::new(25))));
    let tee = pipeline.add_node("tee", Box::new(ElementAdapter::new(Tee::new())));
    let sink = pipeline.add_node("sink", Box::new(SinkAdapter::new(NullSink::new())));

    pipeline.link(src, tee).unwrap();
    pipeline.link(tee, sink).unwrap();

    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test a longer pipeline with multiple elements.
#[tokio::test]
async fn test_long_pipeline() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_node("src", Box::new(SourceAdapter::new(NullSource::new(10))));
    let p1 = pipeline.add_node("p1", Box::new(ElementAdapter::new(PassThrough::new())));
    let p2 = pipeline.add_node("p2", Box::new(ElementAdapter::new(PassThrough::new())));
    let p3 = pipeline.add_node("p3", Box::new(ElementAdapter::new(PassThrough::new())));
    let tee = pipeline.add_node("tee", Box::new(ElementAdapter::new(Tee::new())));
    let sink = pipeline.add_node("sink", Box::new(SinkAdapter::new(NullSink::new())));

    pipeline.link(src, p1).unwrap();
    pipeline.link(p1, p2).unwrap();
    pipeline.link(p2, p3).unwrap();
    pipeline.link(p3, tee).unwrap();
    pipeline.link(tee, sink).unwrap();

    let executor = PipelineExecutor::new();
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

    let src = pipeline.add_node("src", Box::new(SourceAdapter::new(NullSource::new(42))));
    let sink = pipeline.add_node(
        "sink",
        Box::new(SinkAdapter::new(CountingSink {
            count: count.clone(),
        })),
    );

    pipeline.link(src, sink).unwrap();

    let executor = PipelineExecutor::new();
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
    let src = pipeline.add_node("src", Box::new(SourceAdapter::new(NullSource::new(100))));
    let filter = pipeline.add_node(
        "filter",
        Box::new(ElementAdapter::new(EveryNth { n: 10, current: 0 })),
    );
    let sink = pipeline.add_node(
        "sink",
        Box::new(SinkAdapter::new(CountingSink {
            count: count.clone(),
        })),
    );

    pipeline.link(src, filter).unwrap();
    pipeline.link(filter, sink).unwrap();

    let executor = PipelineExecutor::new();
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
    pipeline.add_node("src", Box::new(SourceAdapter::new(NullSource::new(10))));
    let result = pipeline.validate();
    assert!(matches!(result, Err(Error::InvalidSegment(_))));

    // Pipeline with only sink (no source) should fail
    let mut pipeline = Pipeline::new();
    pipeline.add_node("sink", Box::new(SinkAdapter::new(NullSink::new())));
    let result = pipeline.validate();
    assert!(matches!(result, Err(Error::InvalidSegment(_))));
}

/// Test that cycles are detected and rejected.
#[tokio::test]
async fn test_cycle_detection() {
    let mut pipeline = Pipeline::new();

    let a = pipeline.add_node("a", Box::new(ElementAdapter::new(PassThrough::new())));
    let b = pipeline.add_node("b", Box::new(ElementAdapter::new(PassThrough::new())));
    let c = pipeline.add_node("c", Box::new(ElementAdapter::new(PassThrough::new())));

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
            let buffer = Buffer::new(handle, Metadata::with_sequence(self.seq));
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
        Box::new(SourceAdapter::new(InfiniteSource { seq: 0 })),
    );
    let sink = pipeline.add_node(
        "sink",
        Box::new(SinkAdapter::new(CountingSink {
            count: count.clone(),
        })),
    );

    pipeline.link(src, sink).unwrap();

    let executor = PipelineExecutor::new();
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
    use parallax::pipeline::ExecutorConfig;

    let config = ExecutorConfig::with_capacity(64);
    assert_eq!(config.channel_capacity, 64);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node("src", Box::new(SourceAdapter::new(NullSource::new(10))));
    let sink = pipeline.add_node("sink", Box::new(SinkAdapter::new(NullSink::new())));
    pipeline.link(src, sink).unwrap();

    let executor = PipelineExecutor::with_config(config);
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

    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test parsing a pipeline with a transform element.
#[tokio::test]
async fn test_parse_pipeline_with_transform() {
    let mut pipeline = Pipeline::parse("nullsource count=5 ! passthrough ! nullsink").unwrap();
    assert_eq!(pipeline.node_count(), 3);

    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test parsing a pipeline with multiple transforms.
#[tokio::test]
async fn test_parse_long_pipeline() {
    let mut pipeline =
        Pipeline::parse("nullsource count=3 ! passthrough ! tee ! passthrough ! nullsink").unwrap();
    assert_eq!(pipeline.node_count(), 5);

    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await.unwrap();
}

/// Test parsing with different property types.
#[tokio::test]
async fn test_parse_with_properties() {
    let mut pipeline = Pipeline::parse("nullsource count=100 buffer-size=1024 ! nullsink").unwrap();

    let executor = PipelineExecutor::new();
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

    let executor = PipelineExecutor::new();
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
