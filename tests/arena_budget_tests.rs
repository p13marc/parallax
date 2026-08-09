//! Regression tests for #84: a slow consumer must not kill the pipeline.
//!
//! An element that allocates its own output buffers pins one arena slot per
//! buffer in flight. When those arenas were hard-coded to 16 slots behind a
//! 16-deep channel, a consumer that paused for a moment filled the channel, the
//! next `acquire()` found nothing, and the element returned a fatal error — on a
//! live-video path, where the correct answer is to wait or to shed.
//!
//! These use a purpose-built element rather than `JpegEncoder`, which is what
//! the issue reported: `elements::codec` does not compile under default
//! features, so a test written against it would not run in the default CI job.
//! `ArenaTransform` reproduces the same shape — lazy arena, budget-sized,
//! `Error::PoolExhausted` on exhaustion.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ConsumeContext, Element, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::memory::{OutputArena, OutputBudget, SharedArena, defaults};
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, ExecutorConfig, LinkPolicy, Pipeline};

/// Emits `count` small buffers from its own arena, then EOS.
struct CountingSource {
    arena: SharedArena,
    emitted: u64,
    count: u64,
}

impl CountingSource {
    fn new(count: u64) -> Self {
        Self {
            // Deliberately roomy: this test is about the *transform's* arena.
            arena: SharedArena::new(64, 512).unwrap(),
            emitted: 0,
            count,
        }
    }
}

impl Source for CountingSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.emitted >= self.count {
            return Ok(ProduceResult::Eos);
        }
        self.arena.reclaim();
        let Some(slot) = self.arena.acquire() else {
            return Ok(ProduceResult::WouldBlock);
        };
        let buffer = Buffer::new(
            MemoryHandle::with_len(slot, 8),
            Metadata::from_sequence(self.emitted),
        );
        self.emitted += 1;
        Ok(ProduceResult::OwnBuffer(buffer))
    }

    fn name(&self) -> &str {
        "counting-source"
    }
}

/// The shape every codec element has: an output arena built on the first frame,
/// sized by the executor, reporting exhaustion as `Error::PoolExhausted`.
struct ArenaTransform {
    output: OutputArena,
    /// Slot count of the arena once built, so the test can assert the sizing.
    built_slots: Arc<AtomicUsize>,
    explicit: Option<usize>,
}

impl ArenaTransform {
    fn new() -> Self {
        Self {
            output: OutputArena::new(defaults::MIN_OUTPUT_SLOT_COUNT),
            built_slots: Arc::new(AtomicUsize::new(0)),
            explicit: None,
        }
    }

    fn with_output_slots(mut self, slots: usize) -> Self {
        self.explicit = Some(slots);
        self.output.set_slots(slots);
        self
    }

    fn built_slots(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.built_slots)
    }
}

impl Element for ArenaTransform {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let mut slot = self.output.acquire(64, "arenatransform")?;
        slot.data_mut()[..8].copy_from_slice(&[0u8; 8]);

        if let Some(arena) = self.output.arena() {
            self.built_slots
                .store(arena.slot_count(), Ordering::Relaxed);
        }

        Ok(Some(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            buffer.metadata().clone(),
        )))
    }

    fn name(&self) -> &str {
        "arena-transform"
    }
}

/// The "slow consumer" from the issue, made deterministic.
///
/// Retains the first `hold_first` buffers — pinning their arena slots exactly
/// as an application sitting on what it pulled would — then releases everything
/// and keeps up from there. Gating on the buffer count rather than the clock
/// matters: these pipelines finish in microseconds, so any wall-clock gate
/// races the run instead of shaping it.
///
/// `hold_first == usize::MAX` is the pathological consumer that never lets go.
struct GatedSink {
    held: Vec<Buffer>,
    hold_first: usize,
    received: Arc<AtomicUsize>,
}

impl GatedSink {
    fn holding(hold_first: usize) -> Self {
        Self {
            held: Vec::new(),
            hold_first,
            received: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// A consumer that keeps up: nothing is retained.
    fn prompt() -> Self {
        Self::holding(0)
    }
}

impl Sink for GatedSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let n = self.received.fetch_add(1, Ordering::SeqCst) + 1;
        if n <= self.hold_first {
            self.held.push(ctx.buffer().clone());
        } else {
            self.held.clear();
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "gated-sink"
    }
}

/// The headline regression, in the exact shape the issue describes: a consumer
/// momentarily holding a full channel's worth of buffers.
///
/// With the old hard-coded 16 slots behind a 16-deep channel, the 17th
/// `acquire()` found nothing and killed the pipeline. The budget makes the
/// arena strictly deeper than the channel, so this is now unreachable — no
/// shedding, no error, every frame delivered.
#[tokio::test]
async fn a_full_channel_no_longer_exhausts_the_arena() {
    let capacity = ExecutorConfig::default().channel_capacity;
    // Two past the old cliff: enough to prove the inequality holds, few enough
    // that no legitimate shedding can occur.
    let frames = capacity as u64 + 2;

    // Retain every one of them: this is the consumer holding a full
    // channel's worth, which is what used to be fatal.
    let sink = GatedSink::holding(frames as usize);
    let received = Arc::clone(&sink.received);
    let transform = ArenaTransform::new();
    let slots = transform.built_slots();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new(frames));
    let xfm = pipeline.add_filter("xfm", transform);
    let snk = pipeline.add_sink("sink", sink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();

    let result = tokio::time::timeout(Duration::from_secs(10), handle.wait())
        .await
        .expect("pipeline hung");

    assert!(
        result.is_ok(),
        "a consumer holding one channel's worth of buffers killed the pipeline: {:?}",
        result.err()
    );
    assert_eq!(
        received.load(Ordering::SeqCst) as u64,
        frames,
        "frames were shed even though the arena should outsize the channel"
    );
    assert!(
        slots.load(Ordering::Relaxed) > capacity,
        "the arena ({} slots) must outgrow the {capacity}-deep channel it feeds",
        slots.load(Ordering::Relaxed)
    );
}

/// Retention beyond the in-flight margin *does* shed — and then the pipeline
/// recovers, which is the property that matters.
///
/// The budget covers what the channels hold; an application sitting on buffers
/// it has already pulled is invisible to the executor and can pin arbitrarily
/// many slots (see the module docs on `OutputBudget`). So a consumer that hoards
/// a full channel's worth loses frames — deliberately. What must not happen is
/// the pipeline dying, or staying broken once the consumer catches up.
#[tokio::test]
async fn a_consumer_that_hoards_then_recovers_sheds_but_keeps_running() {
    let capacity = ExecutorConfig::default().channel_capacity;
    let frames = 200u64;
    let hold = capacity + 2;

    let sink = GatedSink::holding(hold);
    let received = Arc::clone(&sink.received);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new(frames));
    let xfm = pipeline.add_filter("xfm", ArenaTransform::new());
    let snk = pipeline.add_sink("sink", sink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();

    let result = tokio::time::timeout(Duration::from_secs(15), handle.wait())
        .await
        .expect("pipeline hung");

    assert!(
        result.is_ok(),
        "hoarding must shed, not kill the pipeline: {:?}",
        result.err()
    );

    // Recovery is the assertion: the run continued well past the burst rather
    // than wedging at the point the arena first ran dry.
    let got = received.load(Ordering::SeqCst) as u64;
    assert!(
        got > hold as u64 + 50,
        "only {got} of {frames} frames arrived — the pipeline never recovered \
         from the burst"
    );
}

/// The case no budget can bound: the consumer never lets go. The pipeline must
/// degrade to shedding, not die.
#[tokio::test]
async fn a_consumer_that_never_releases_sheds_instead_of_dying() {
    // Never lets go of anything: no budget can bound this.
    let sink = GatedSink::holding(usize::MAX);
    let received = Arc::clone(&sink.received);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new(200));
    let xfm = pipeline.add_filter("xfm", ArenaTransform::new());
    let snk = pipeline.add_sink("sink", sink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();

    let result = tokio::time::timeout(Duration::from_secs(15), handle.wait())
        .await
        .expect("pipeline hung instead of shedding");

    assert!(
        result.is_ok(),
        "arena exhaustion should shed, not fail the pipeline: {:?}",
        result.err()
    );
    // Some frames were shed — that is the point — but the run completed and the
    // sink saw real work rather than an immediate error.
    let got = received.load(Ordering::SeqCst);
    assert!(got > 0, "the sink received nothing at all");
    assert!(
        got < 200,
        "expected some shedding with a consumer holding every buffer, got all {got}"
    );
}

/// `shed_fatal_after` is the opt-in for work where silent degradation is worse
/// than stopping.
#[tokio::test]
async fn shed_fatal_after_turns_shedding_back_into_a_failure() {
    let sink = GatedSink::holding(usize::MAX);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new(500));
    let xfm = pipeline.add_filter("xfm", ArenaTransform::new());
    let snk = pipeline.add_sink("sink", sink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::with_config(ExecutorConfig::default().with_shed_fatal_after(5));
    let handle = executor.start(&mut pipeline).unwrap();

    let result = tokio::time::timeout(Duration::from_secs(15), handle.wait())
        .await
        .expect("pipeline hung");

    let err = result.expect_err("shed_fatal_after(5) should have failed the run");
    assert!(format!("{err}").contains("shed"), "unhelpful error: {err}");
}

/// The arena tracks the channel it feeds, whatever the channel's depth.
#[tokio::test]
async fn the_arena_scales_with_the_channel_capacity() {
    for capacity in [4usize, 64, 128] {
        let transform = ArenaTransform::new();
        let slots = transform.built_slots();

        let mut pipeline = Pipeline::new();
        let src = pipeline.add_source("src", CountingSource::new(4));
        let xfm = pipeline.add_filter("xfm", transform);
        let snk = pipeline.add_sink("sink", GatedSink::prompt());
        pipeline.link(src, xfm).unwrap();
        pipeline.link(xfm, snk).unwrap();

        let executor =
            Executor::with_config(ExecutorConfig::default().with_channel_capacity(capacity));
        let handle = executor.start(&mut pipeline).unwrap();
        tokio::time::timeout(Duration::from_secs(10), handle.wait())
            .await
            .expect("pipeline hung")
            .unwrap();

        let built = slots.load(Ordering::Relaxed);
        assert!(
            built > capacity,
            "capacity {capacity} got only {built} slots — the inequality this \
             whole mechanism exists to maintain is broken"
        );
    }
}

/// Fan-out shares slots, so the budget follows the deepest branch rather than
/// the sum. Summing would over-allocate by up to N×.
#[tokio::test]
async fn fan_out_sizes_from_the_deepest_branch_not_the_total() {
    let transform = ArenaTransform::new();
    let slots = transform.built_slots();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new(4));
    let xfm = pipeline.add_filter("xfm", transform);
    let a = pipeline.add_sink("a", GatedSink::prompt());
    let b = pipeline.add_sink("b", GatedSink::prompt());
    pipeline.link(src, xfm).unwrap();
    pipeline
        .link_pads_full(xfm, "src", a, "sink", LinkPolicy::Block, Some(8))
        .unwrap();
    pipeline
        .link_pads_full(xfm, "src", b, "sink", LinkPolicy::Block, Some(64))
        .unwrap();

    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();
    tokio::time::timeout(Duration::from_secs(10), handle.wait())
        .await
        .expect("pipeline hung")
        .unwrap();

    let built = slots.load(Ordering::Relaxed);
    let margin = defaults::IN_FLIGHT_MARGIN;
    assert_eq!(
        built,
        64 + margin,
        "expected max(8, 64) + {margin}; a sum would have given {}",
        8 + 64 + margin
    );
}

/// An explicit slot count wins over the executor's, even when it is too small.
#[tokio::test]
async fn an_explicit_slot_count_overrides_the_budget() {
    let transform = ArenaTransform::new().with_output_slots(3);
    let slots = transform.built_slots();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new(4));
    let xfm = pipeline.add_filter("xfm", transform);
    let snk = pipeline.add_sink("sink", GatedSink::prompt());
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();
    tokio::time::timeout(Duration::from_secs(10), handle.wait())
        .await
        .expect("pipeline hung")
        .unwrap();

    assert_eq!(slots.load(Ordering::Relaxed), 3);
}
