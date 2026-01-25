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
