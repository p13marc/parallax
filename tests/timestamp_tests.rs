//! Tests for timestamp (PTS/DTS) preservation through the pipeline.

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{ConsumeContext, Element, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};
use std::sync::OnceLock;
use std::sync::{Arc, Mutex};

fn test_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(1024, 64).unwrap())
}

/// Source that produces buffers with known PTS values.
struct TimestampedSource {
    count: u32,
    max: u32,
    pts_interval_ns: u64,
}

impl TimestampedSource {
    fn new(max: u32, pts_interval_ns: u64) -> Self {
        Self {
            count: 0,
            max,
            pts_interval_ns,
        }
    }
}

impl Source for TimestampedSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.count >= self.max {
            return Ok(ProduceResult::Eos);
        }

        let pts = ClockTime::from_nanos(self.count as u64 * self.pts_interval_ns);
        let arena = test_arena();
        let mut slot = arena.acquire().unwrap();
        // Write frame number as data
        slot.data_mut()[..4].copy_from_slice(&self.count.to_le_bytes());

        let mut metadata = Metadata::new();
        metadata.pts = pts;
        metadata.sequence = self.count as u64;

        self.count += 1;

        let handle = MemoryHandle::with_len(slot, 4);
        Ok(ProduceResult::OwnBuffer(Buffer::new(handle, metadata)))
    }

    fn name(&self) -> &str {
        "TimestampedSource"
    }
}

/// Transform that verifies and passes through PTS.
struct PtsPassthroughTransform;

impl Element for PtsPassthroughTransform {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Verify PTS is set (allow ZERO for first frame)
        let pts = buffer.metadata().pts;
        assert!(pts != ClockTime::NONE, "Expected valid PTS, got {:?}", pts);

        // Create new buffer with same metadata (simulates transform behavior)
        let arena = test_arena();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..buffer.len()].copy_from_slice(buffer.as_bytes());

        let handle = MemoryHandle::with_len(slot, buffer.len());
        // CRITICAL: Preserve metadata including PTS
        Ok(Some(Buffer::new(handle, buffer.metadata().clone())))
    }

    fn name(&self) -> &str {
        "PtsPassthroughTransform"
    }
}

/// Sink that records all received PTS values for verification.
struct PtsRecordingSink {
    received_pts: Arc<Mutex<Vec<u64>>>,
}

impl PtsRecordingSink {
    fn new(received_pts: Arc<Mutex<Vec<u64>>>) -> Self {
        Self { received_pts }
    }
}

impl Sink for PtsRecordingSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let pts = ctx.buffer().metadata().pts;
        self.received_pts.lock().unwrap().push(pts.nanos());
        Ok(())
    }

    fn name(&self) -> &str {
        "PtsRecordingSink"
    }
}

/// Test that PTS flows through a simple source -> sink pipeline.
#[tokio::test]
async fn test_pts_preservation_source_to_sink() {
    let received_pts = Arc::new(Mutex::new(Vec::new()));

    let mut pipeline = Pipeline::new();

    // 10 frames at 33ms interval (30fps)
    let src = pipeline.add_source(
        "src",
        TimestampedSource::new(10, 33_333_333), // 33.33ms in nanoseconds
    );
    let sink = pipeline.add_sink("sink", PtsRecordingSink::new(received_pts.clone()));

    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    // Verify all PTS values were received correctly
    let pts_values = received_pts.lock().unwrap();
    assert_eq!(pts_values.len(), 10, "Should have received 10 buffers");

    for (i, &pts) in pts_values.iter().enumerate() {
        let expected = i as u64 * 33_333_333;
        assert_eq!(
            pts, expected,
            "Buffer {} should have PTS {} but got {}",
            i, expected, pts
        );
    }
}

/// Test that PTS flows through a pipeline with a transform element.
#[tokio::test]
async fn test_pts_preservation_through_transform() {
    let received_pts = Arc::new(Mutex::new(Vec::new()));

    let mut pipeline = Pipeline::new();

    // 5 frames at 40ms interval (25fps)
    let src = pipeline.add_source(
        "src",
        TimestampedSource::new(5, 40_000_000), // 40ms in nanoseconds
    );
    let transform = pipeline.add_filter("transform", PtsPassthroughTransform);
    let sink = pipeline.add_sink("sink", PtsRecordingSink::new(received_pts.clone()));

    pipeline.link(src, transform).unwrap();
    pipeline.link(transform, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    // Verify all PTS values were received correctly
    let pts_values = received_pts.lock().unwrap();
    assert_eq!(pts_values.len(), 5, "Should have received 5 buffers");

    for (i, &pts) in pts_values.iter().enumerate() {
        let expected = i as u64 * 40_000_000;
        assert_eq!(
            pts, expected,
            "Buffer {} should have PTS {} but got {}",
            i, expected, pts
        );
    }
}

/// Test that PTS flows through multiple chained transforms.
#[tokio::test]
async fn test_pts_preservation_through_chain() {
    let received_pts = Arc::new(Mutex::new(Vec::new()));

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source(
        "src",
        TimestampedSource::new(3, 100_000_000), // 100ms
    );
    let t1 = pipeline.add_filter("t1", PtsPassthroughTransform);
    let t2 = pipeline.add_filter("t2", PtsPassthroughTransform);
    let t3 = pipeline.add_filter("t3", PtsPassthroughTransform);
    let sink = pipeline.add_sink("sink", PtsRecordingSink::new(received_pts.clone()));

    pipeline.link(src, t1).unwrap();
    pipeline.link(t1, t2).unwrap();
    pipeline.link(t2, t3).unwrap();
    pipeline.link(t3, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    // Verify
    let pts_values = received_pts.lock().unwrap();
    assert_eq!(pts_values.len(), 3);
    assert_eq!(pts_values[0], 0);
    assert_eq!(pts_values[1], 100_000_000);
    assert_eq!(pts_values[2], 200_000_000);
}

/// Test that clock is provided to sources via ProduceContext.
#[tokio::test]
async fn test_clock_provided_to_source() {
    use std::sync::atomic::{AtomicBool, Ordering};

    let clock_was_available = Arc::new(AtomicBool::new(false));
    let clock_check = clock_was_available.clone();

    /// Source that checks if clock is available.
    struct ClockCheckSource {
        clock_available: Arc<AtomicBool>,
        produced: bool,
    }

    impl Source for ClockCheckSource {
        fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
            if self.produced {
                return Ok(ProduceResult::Eos);
            }

            // Check if clock is available
            if ctx.clock().is_some() {
                self.clock_available.store(true, Ordering::SeqCst);
            }

            self.produced = true;

            let arena = test_arena();
            let slot = arena.acquire().unwrap();
            let handle = MemoryHandle::with_len(slot, 4);
            Ok(ProduceResult::OwnBuffer(Buffer::new(
                handle,
                Metadata::new(),
            )))
        }

        fn name(&self) -> &str {
            "ClockCheckSource"
        }
    }

    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source(
        "src",
        ClockCheckSource {
            clock_available: clock_check,
            produced: false,
        },
    );
    let sink = pipeline.add_sink("sink", parallax::elements::NullSink::new());

    pipeline.link(src, sink).unwrap();

    let executor = Executor::new();
    executor.run(&mut pipeline).await.unwrap();

    assert!(
        clock_was_available.load(Ordering::SeqCst),
        "Clock should be available in ProduceContext"
    );
}
