//! Flow Control and Backpressure Example
//!
//! Demonstrates executor-produced link flow signals:
//! - `Pipeline::monitor_link` attaches watermarks to a link and returns a
//!   `FlowStateHandle` driven from the link channel's occupancy
//! - a live source polls `should_produce()` and skips capture work while
//!   downstream is backed up — cheaper than dropping after the copy
//!
//! ```text
//! [SimulatedCamera] ──Drop link(8), monitored──> [SlowSink]
//!        ^                                            │
//!        └───────── FlowStateHandle (Busy/Ready) ─────┘
//! ```
//!
//! Run with: cargo run --example 47_flow_control

use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::memory::FixedBufferPool;
use parallax::pipeline::flow::{FlowStateHandle, WaterMarks};
use parallax::pipeline::{Executor, LinkPolicy, Pipeline};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// A simulated live video source. When downstream is congested it skips the
/// frame *before* doing the capture work, the way v4l2/screen-capture do.
struct SimulatedCameraSource {
    frame_count: u64,
    max_frames: u64,
    flow_state: Option<FlowStateHandle>,
    produced: Arc<AtomicU64>,
    skipped: Arc<AtomicU64>,
}

impl SimulatedCameraSource {
    fn new(max_frames: u64) -> Self {
        Self {
            frame_count: 0,
            max_frames,
            flow_state: None,
            produced: Arc::new(AtomicU64::new(0)),
            skipped: Arc::new(AtomicU64::new(0)),
        }
    }

    fn set_flow_state(&mut self, handle: FlowStateHandle) {
        self.flow_state = Some(handle);
    }

    fn counters(&self) -> (Arc<AtomicU64>, Arc<AtomicU64>) {
        (self.produced.clone(), self.skipped.clone())
    }
}

impl Source for SimulatedCameraSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.frame_count >= self.max_frames {
            return Ok(ProduceResult::Eos);
        }

        // The gate: skip the (simulated) capture + copy while Busy.
        if let Some(flow) = &self.flow_state
            && !flow.should_produce()
        {
            self.skipped.fetch_add(1, Ordering::Relaxed);
            flow.record_drop();
            return Ok(ProduceResult::WouldBlock);
        }

        // Pool-aware acquisition: the pool reclaims slots that return
        // through the arena's release queue, so sustained production works.
        let Ok(mut pooled) = ctx.acquire_buffer() else {
            return Ok(ProduceResult::WouldBlock);
        };
        self.frame_count += 1;
        self.produced.fetch_add(1, Ordering::Relaxed);
        // Simulated capture into the pooled slot.
        pooled.data_mut()[..64].fill(0x42);
        pooled.set_len(64);
        pooled.metadata_mut().sequence = self.frame_count;
        Ok(ProduceResult::OwnBuffer(pooled.into_buffer()))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        Some(64)
    }
}

/// A sink that consumes slower than the source produces.
struct SlowSink {
    consumed: u64,
}

impl Sink for SlowSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        self.consumed += 1;
        // Simulate slow processing (an encoder that can't keep up).
        std::thread::sleep(Duration::from_millis(3));
        Ok(())
    }
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let mut pipeline = Pipeline::new();
    let camera = SimulatedCameraSource::new(200);
    let (produced, skipped) = camera.counters();

    let src = pipeline.add_source_with_pool("camera", camera, FixedBufferPool::new(256, 64)?);
    let sink = pipeline.add_sink("slow_sink", SlowSink { consumed: 0 });

    // A live branch: Drop policy (a stalled sink must not wedge the camera),
    // shallow channel, monitored with explicit watermarks.
    let link = pipeline.link_pads_full(src, "src", sink, "sink", LinkPolicy::Drop, Some(8))?;
    let flow = pipeline.monitor_link_with(link, WaterMarks::new(6, 2))?;

    // Hand the source its gate before start (elements move into their tasks).
    pipeline
        .get_element_mut::<SimulatedCameraSource>("camera")
        .unwrap()
        .set_flow_state(flow.clone());

    println!("Running: fast camera → Drop link(8, marks 6/2) → slow sink\n");
    let executor = Executor::new();
    let handle = executor.start(&mut pipeline)?;
    handle.wait().await?;

    println!("Frames produced : {}", produced.load(Ordering::Relaxed));
    println!("Frames skipped  : {}", skipped.load(Ordering::Relaxed));
    println!("Backpressure events: {}", flow.backpressure_events());
    println!(
        "\nThe skip counter is work the camera never did — the gate closed at\n\
         the high mark, before capture, instead of dropping after the copy."
    );
    Ok(())
}
