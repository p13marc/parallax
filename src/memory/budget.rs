//! How many slots an element's output arena needs.
//!
//! An element that produces buffers owns an arena, and every buffer it emits
//! pins a slot until the last downstream reference drops. Size the arena below
//! what the graph can hold in flight and the element runs out of slots — which,
//! before this existed, killed the pipeline.
//!
//! The executor is the only party that knows how much the graph can hold: link
//! capacity is its configuration ([`channel_capacity`], or a per-link override).
//! So it computes an [`OutputBudget`] per node and hands it to the element
//! before the element builds anything. Elements build their arenas lazily from
//! the first frame's geometry, so the budget is stored and consulted at build
//! time — including on the rebuild that follows a resolution change.
//!
//! # This is a floor, not a guarantee
//!
//! The budget bounds what the *channels* can hold. It cannot bound what
//! downstream *elements* hold: an `AppSink` queues up to its `max_buffers`, a
//! `Queue` up to its depth, and an application can retain every `Buffer` it
//! pulls for as long as it likes. None of that is visible to the executor.
//!
//! So exhaustion remains possible, and is handled where it happens rather than
//! prevented here: an element that cannot acquire returns
//! [`Error::PoolExhausted`](crate::error::Error::PoolExhausted), which the
//! executor treats as a shed buffer rather than a fatal error. The budget makes
//! the common case impossible; shedding keeps the pathological case alive.
//!
//! [`channel_capacity`]: crate::pipeline::ExecutorConfig::channel_capacity

use super::defaults;
use super::shared_refcount::{SharedArena, SharedSlotRef};
use crate::error::{Error, Result};
use std::time::Duration;

/// The slot count an element's output arena should have, as computed by the
/// executor from the graph it is about to run.
///
/// Obtained through `set_output_budget`; elements do not construct one, except
/// in tests. [`Default`] is the standalone case (no executor), which resolves to
/// the caller's floor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OutputBudget {
    /// How many buffers the downstream links can hold at once.
    ///
    /// The **maximum** over the links of a src pad, not the sum: fan-out clones
    /// a `Buffer`, and a clone is a refcount bump on the *same* slot, so N
    /// branches holding the same buffer pin one slot, not N. Summing would
    /// over-allocate by up to N×. Genuinely distinct pads (a demuxer's) *are*
    /// summed, because those emit different buffers.
    pub downstream_capacity: usize,

    /// Slack for buffers that are in neither the arena's free list nor a
    /// channel: one in the consumer's hand, one in the producer's between
    /// `process()` returning and the send completing, one for a probe or tracer
    /// holding a clone.
    pub in_flight_margin: usize,
}

impl OutputBudget {
    /// Build a budget directly. The executor is the normal source of these.
    pub fn new(downstream_capacity: usize, in_flight_margin: usize) -> Self {
        Self {
            downstream_capacity,
            in_flight_margin,
        }
    }

    /// The raw slot count this budget asks for, before any floor or clamp.
    pub fn slots(&self) -> usize {
        self.downstream_capacity
            .saturating_add(self.in_flight_margin)
    }

    /// The slot count to actually allocate, given a per-element `floor` and the
    /// byte size of one slot.
    ///
    /// Two adjustments to [`slots`](Self::slots):
    ///
    /// - the `floor` wins when it is larger, so an element that knows it needs
    ///   depth (a lookahead encoder) keeps it even in a shallow graph;
    /// - the total is clamped to [`defaults::MAX_OUTPUT_ARENA_BYTES`] slots' worth. A 4K
    ///   RGBA frame is 33 MB, so an unclamped `channel_capacity: 200` would ask
    ///   for 6.6 GB. Degrading to fewer slots sheds frames; allocating 6.6 GB
    ///   takes the machine down.
    ///
    /// Never returns 0 — an arena with no slots is useless, and a slot larger
    /// than the whole clamp still gets one.
    pub fn resolve(&self, floor: usize, slot_size: usize) -> usize {
        let want = self.slots().max(floor).max(1);

        if slot_size == 0 {
            return want;
        }
        let affordable = (defaults::MAX_OUTPUT_ARENA_BYTES / slot_size).max(1);
        want.min(affordable)
    }

    /// [`resolve`](Self::resolve), honouring an explicit constructor override.
    ///
    /// An explicit count always wins — a caller who sized their arena by hand
    /// meant it. But a count *below* the computed budget re-creates the bug this
    /// machinery exists to remove, so it is reported: `element` names the
    /// offender, and the caller can raise it or accept the shedding.
    ///
    /// The reverse (an override above the budget) is silent — deliberately
    /// buffering more is a legitimate choice.
    pub fn resolve_with_override(
        &self,
        explicit: Option<usize>,
        floor: usize,
        slot_size: usize,
        element: &str,
    ) -> usize {
        let Some(explicit) = explicit else {
            return self.resolve(floor, slot_size);
        };

        let explicit = explicit.max(1);
        let needed = self.slots();
        if explicit < needed {
            tracing::warn!(
                "{element}: output arena pinned to {explicit} slots but the graph can hold \
                 {needed} buffers in flight ({} of link capacity + {} margin); frames will be \
                 shed under load",
                self.downstream_capacity,
                self.in_flight_margin,
            );
        }
        explicit
    }
}

/// A lazily-built output arena that sizes itself from the executor's budget.
///
/// Elements that emit buffers nearly all want the same thing: an arena built on
/// the first frame (because only then is the payload size known), rebuilt on a
/// resize, sized from the budget, and reporting exhaustion in the one way the
/// executor treats as recoverable. Before this existed each of them spelled it
/// out by hand with a hard-coded slot count, which is precisely how a dozen
/// elements ended up with 16-slot arenas behind a 16-deep channel.
///
/// ```rust,ignore
/// struct MyEncoder {
///     output: OutputArena,
/// }
///
/// impl MyEncoder {
///     fn new() -> Self {
///         Self { output: OutputArena::new(defaults::VIDEO_ENCODER_SLOT_COUNT) }
///     }
/// }
///
/// impl Element for MyEncoder {
///     fn set_output_budget(&mut self, budget: OutputBudget) {
///         self.output.set_budget(budget);
///     }
///
///     fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
///         self.output.admit()?;                     // before irreversible work
///         let encoded = self.encode(&buffer)?;
///         let mut slot = self.output.acquire(encoded.len(), "myencoder")?;
///         slot.data_mut()[..encoded.len()].copy_from_slice(&encoded);
///         // ...
///     }
/// }
/// ```
/// Every slot is 64-byte aligned — [`SharedArena::new`] asks for a cache line,
/// which also satisfies `MemoryLayout::{SSE, AVX, AVX512}`. An element that used
/// to reach for `SharedArena::new_avx` for its 32-byte alignment gets a stronger
/// guarantee here, not a weaker one.
pub struct OutputArena {
    arena: Option<SharedArena>,
    budget: Option<OutputBudget>,
    explicit: Option<usize>,
    floor: usize,
    min_slot_size: usize,
    grow: bool,
}

impl std::fmt::Debug for OutputArena {
    // Hand-written because SharedArena is not Debug, and the useful facts here
    // are the shape of the arena rather than its mapping.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputArena")
            .field("built", &self.arena.is_some())
            .field("slots", &self.arena.as_ref().map(|a| a.slot_count()))
            .field("slot_size", &self.arena.as_ref().map(|a| a.slot_size()))
            .field("budget", &self.budget)
            .field("explicit", &self.explicit)
            .field("floor", &self.floor)
            .field("grow", &self.grow)
            .finish()
    }
}

impl OutputArena {
    /// A new, unbuilt arena. `floor` is the slot count to use when no executor
    /// budget arrives — pick one of the `memory::defaults` counts.
    pub fn new(floor: usize) -> Self {
        Self {
            arena: None,
            budget: None,
            explicit: None,
            floor,
            min_slot_size: 0,
            grow: false,
        }
    }

    /// Rebuild the arena when a payload outgrows the slot, instead of failing.
    ///
    /// Slot size is fixed when the arena is built, so the default is strict: an
    /// oversized payload is a bug and [`acquire`](Self::acquire) says so. That
    /// is right for an element whose output size is *bounded* — an encoder
    /// writing under a ceiling, a capture source copying a frame of the driver's
    /// negotiated `sizeimage`.
    ///
    /// It is wrong for an element whose output size follows its **input**: a
    /// `Map` over arbitrary bytes, an RTP depayloader that meets a bigger
    /// keyframe than the last one. Those have no ceiling to enforce, and failing
    /// on the first large frame would shed every large frame after it — a live
    /// stream that never recovers. Pair this with
    /// [`with_min_slot_size`](Self::with_min_slot_size) so the common case still
    /// builds once.
    ///
    /// Growth is one-way: a smaller payload reuses the slot it already fits in.
    pub fn grow_to_fit(mut self) -> Self {
        self.grow = true;
        self
    }

    /// [`grow_to_fit`](Self::grow_to_fit) after construction.
    pub fn set_grow_to_fit(&mut self, grow: bool) {
        self.grow = grow;
    }

    /// Never build slots smaller than `bytes`.
    ///
    /// Slot size is fixed when the arena is built, from the first frame — but a
    /// compressed frame's size varies, and the *next* one may well be bigger.
    /// Encoders should set this to a generous ceiling so a slightly larger
    /// frame does not have to rebuild the arena (or, worse, fail to fit).
    pub fn with_min_slot_size(mut self, bytes: usize) -> Self {
        self.min_slot_size = bytes;
        self
    }

    /// [`with_min_slot_size`](Self::with_min_slot_size) after construction, for
    /// elements that only learn their upper bound from the first frame's
    /// geometry. Takes effect on the next build, so pair it with
    /// [`reset`](Self::reset) to re-size an existing arena.
    pub fn set_min_slot_size(&mut self, bytes: usize) {
        self.min_slot_size = bytes;
    }

    /// Adopt the executor's budget. Wire this to `set_output_budget`.
    pub fn set_budget(&mut self, budget: OutputBudget) {
        self.budget = Some(budget);
    }

    /// Pin the slot count explicitly, overriding the budget.
    pub fn set_slots(&mut self, slots: usize) {
        self.explicit = Some(slots);
    }

    /// Whether the arena has been built yet.
    pub fn is_built(&self) -> bool {
        self.arena.is_some()
    }

    /// The built arena, if any.
    pub fn arena(&self) -> Option<&SharedArena> {
        self.arena.as_ref()
    }

    /// Drop the arena so the next [`acquire`](Self::acquire) rebuilds it.
    ///
    /// For a geometry change: the new frames need differently-sized slots, and
    /// the budget is re-consulted on the way through.
    pub fn reset(&mut self) {
        self.arena = None;
    }

    /// Admission control: is there room for another output buffer?
    ///
    /// Call this **before** doing work that cannot be undone — pushing a frame
    /// into an encoder's GOP, advancing a muxer's continuity counter. On
    /// [`Error::PoolExhausted`] the caller should return immediately, leaving
    /// its state untouched; the executor sheds the buffer and the next one
    /// proceeds normally.
    ///
    /// Succeeds trivially before the arena exists — there is nothing to be full
    /// yet, and the first [`acquire`](Self::acquire) will build it.
    pub fn admit(&mut self) -> Result<()> {
        let Some(arena) = self.arena.as_mut() else {
            return Ok(());
        };
        arena.reclaim();
        if arena.has_free() {
            Ok(())
        } else {
            Err(Error::PoolExhausted)
        }
    }

    /// [`admit`](Self::admit), but wait up to `timeout` for a slot before
    /// giving up (#180).
    ///
    /// For call sites that would rather stall briefly than shed — an encoder
    /// shedding a packet leaves a `frame_num` gap until the next IDR, so one
    /// or two frame intervals of waiting can be the better trade. Not wired
    /// into any built-in element; opting in is a per-element latency
    /// decision. Blocks the calling thread — keep it off Tokio workers for
    /// anything longer than a frame interval.
    pub fn admit_within(&mut self, timeout: Duration) -> Result<()> {
        if self.admit().is_ok() {
            return Ok(());
        }
        let deadline = std::time::Instant::now() + timeout;
        // Clone so the waiter's borrow doesn't overlap `admit(&mut self)`;
        // the doorbell Arc is shared across clones, so rings still land.
        let arena = self
            .arena
            .as_ref()
            .expect("admit only fails on a built arena")
            .clone();
        let waiter = arena.doorbell().register();
        loop {
            // Re-check after registering (doorbell fence pair closes the
            // lost-wakeup window); admit() reclaims — and sweeps orphans.
            if self.admit().is_ok() {
                return Ok(());
            }
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Err(Error::PoolExhausted);
            }
            waiter.wait(remaining.min(Duration::from_millis(250)));
        }
    }

    /// Acquire a slot big enough for `len` bytes, building the arena if needed.
    ///
    /// The arena's slot size is fixed at build time, from `len` and
    /// [`with_min_slot_size`](Self::with_min_slot_size); later frames must fit
    /// within it. Call [`reset`](Self::reset) when the geometry changes.
    ///
    /// Returns [`Error::PoolExhausted`] when every slot is taken, which the
    /// executor treats as a shed buffer rather than a fatal error. A frame too
    /// big for the slot is a plain error naming the cause — the hand-written
    /// versions of this used to index straight into the slot and panic.
    ///
    /// Sources want [`try_acquire`](Self::try_acquire) instead: nothing sheds a
    /// source's `PoolExhausted`, so returning it would kill the pipeline.
    pub fn acquire(&mut self, len: usize, element: &str) -> Result<SharedSlotRef> {
        self.try_acquire(len, element)?.ok_or(Error::PoolExhausted)
    }

    /// [`acquire`](Self::acquire), but a full arena is `Ok(None)` rather than an
    /// error.
    ///
    /// For **sources**. `Error::PoolExhausted` is only shed inside the
    /// executor's transform task; a source that returned it would take its own
    /// pipeline down. `Ok(None)` lets the source answer
    /// `ProduceResult::WouldBlock` instead, which the executor already
    /// understands — it sleeps briefly and asks again — so the source stalls
    /// until downstream releases a slot and then carries on.
    ///
    /// Note what does *not* become `Ok(None)`: a frame too large for the slot,
    /// and a failed mapping, are still errors. Reporting those as "try again"
    /// would spin forever on a condition retrying cannot fix.
    pub fn try_acquire(&mut self, len: usize, element: &str) -> Result<Option<SharedSlotRef>> {
        // A payload past the slot size needs a differently-sized arena, and the
        // budget is re-consulted on the way through.
        if self.grow && self.arena.as_ref().is_some_and(|a| a.slot_size() < len) {
            self.reset();
        }

        if self.arena.is_none() {
            let slot_size = len.max(self.min_slot_size).max(1);
            let budget = self.budget.unwrap_or_default();
            let slots = budget.resolve_with_override(self.explicit, self.floor, slot_size, element);

            tracing::debug!(
                "{element}: output arena {slots} slots x {slot_size} bytes \
                 (budget {} + {})",
                budget.downstream_capacity,
                budget.in_flight_margin,
            );
            self.arena =
                Some(SharedArena::new(slot_size, slots).map_err(|e| {
                    Error::AllocationFailed(format!("{element}: output arena: {e}"))
                })?);
        }

        let arena = self.arena.as_mut().expect("built immediately above");
        arena.reclaim();

        let Some(slot) = arena.acquire() else {
            return Ok(None);
        };
        if slot.len() < len {
            return Err(Error::InvalidSegment(format!(
                "{element}: output slot is {} bytes but the frame needs {len} — the arena was \
                 built for a smaller payload; call reset() on a geometry change, or \
                 grow_to_fit() if the size follows the input",
                slot.len(),
            )));
        }
        Ok(Some(slot))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slots_is_capacity_plus_margin() {
        assert_eq!(OutputBudget::new(16, 4).slots(), 20);
    }

    #[test]
    fn the_floor_wins_in_a_shallow_graph() {
        // A one-deep link must not leave a lookahead encoder with 5 slots.
        let budget = OutputBudget::new(1, 4);
        assert_eq!(budget.resolve(64, 1024), 64);
    }

    #[test]
    fn the_budget_wins_in_a_deep_graph() {
        let budget = OutputBudget::new(256, 4);
        assert_eq!(budget.resolve(64, 1024), 260);
    }

    #[test]
    fn a_default_budget_falls_back_to_the_floor() {
        // No executor: an element constructed and driven by hand.
        assert_eq!(OutputBudget::default().resolve(16, 1024), 16);
    }

    #[test]
    fn a_huge_slot_is_clamped_rather_than_allocated() {
        // 4K RGBA at a 200-deep link would be ~6.6 GB unclamped.
        let slot = 3840 * 2160 * 4;
        let budget = OutputBudget::new(200, 4);
        let slots = budget.resolve(64, slot);

        assert!(slots < 204, "expected a clamp, got {slots} slots");
        assert!(
            slots * slot <= defaults::MAX_OUTPUT_ARENA_BYTES,
            "clamp exceeded its own bound"
        );
    }

    #[test]
    fn even_an_oversized_slot_gets_one() {
        let slot = defaults::MAX_OUTPUT_ARENA_BYTES * 2;
        assert_eq!(OutputBudget::new(16, 4).resolve(16, slot), 1);
    }

    #[test]
    fn a_zero_slot_size_is_not_a_division_by_zero() {
        assert_eq!(OutputBudget::new(16, 4).resolve(8, 0), 20);
    }

    #[test]
    fn resolve_never_returns_zero() {
        assert_eq!(OutputBudget::new(0, 0).resolve(0, 1024), 1);
    }

    #[test]
    fn an_explicit_count_wins_in_both_directions() {
        let budget = OutputBudget::new(16, 4);
        assert_eq!(budget.resolve_with_override(Some(4), 64, 1024, "t"), 4);
        assert_eq!(budget.resolve_with_override(Some(512), 64, 1024, "t"), 512);
    }

    #[test]
    fn no_override_resolves_normally() {
        let budget = OutputBudget::new(16, 4);
        assert_eq!(budget.resolve_with_override(None, 64, 1024, "t"), 64);
    }

    #[test]
    fn an_explicit_zero_is_raised_to_one() {
        let budget = OutputBudget::new(16, 4);
        assert_eq!(budget.resolve_with_override(Some(0), 64, 1024, "t"), 1);
    }

    // ---- OutputArena ------------------------------------------------------

    #[test]
    fn the_arena_is_built_on_first_acquire_and_sized_from_the_budget() {
        let mut out = OutputArena::new(defaults::MIN_OUTPUT_SLOT_COUNT);
        out.set_budget(OutputBudget::new(32, 4));
        assert!(!out.is_built());

        let slot = out.acquire(1024, "t").unwrap();
        assert!(out.is_built());
        assert_eq!(out.arena().unwrap().slot_count(), 36);
        assert!(slot.len() >= 1024);
    }

    #[test]
    fn without_a_budget_the_arena_falls_back_to_its_floor() {
        let mut out = OutputArena::new(8);
        out.acquire(1024, "t").unwrap();
        assert_eq!(out.arena().unwrap().slot_count(), 8);
    }

    #[test]
    fn admit_passes_before_the_arena_exists() {
        let mut out = OutputArena::new(2);
        assert!(
            out.admit().is_ok(),
            "nothing can be full before it is built"
        );
    }

    #[test]
    fn a_full_arena_reports_pool_exhausted_from_both_paths() {
        let mut out = OutputArena::new(2);
        out.set_slots(2);

        let held: Vec<_> = (0..2).map(|_| out.acquire(64, "t").unwrap()).collect();

        assert!(matches!(out.admit(), Err(Error::PoolExhausted)));
        assert!(matches!(out.acquire(64, "t"), Err(Error::PoolExhausted)));

        // Releasing is not enough — the owner has to reclaim, which both
        // admit() and acquire() do for the caller.
        drop(held);
        assert!(out.admit().is_ok());
        assert!(out.acquire(64, "t").is_ok());
    }

    #[test]
    fn admit_within_passes_trivially_before_the_arena_exists() {
        let mut out = OutputArena::new(2);
        assert!(out.admit_within(Duration::from_millis(1)).is_ok());
    }

    #[test]
    fn admit_within_wakes_when_a_slot_frees() {
        // #180: a slot dropped from another thread rings the arena doorbell
        // and unblocks admit_within well before its deadline.
        let mut out = OutputArena::new(2);
        out.set_slots(2);
        let held: Vec<_> = (0..2).map(|_| out.acquire(64, "t").unwrap()).collect();
        assert!(matches!(out.admit(), Err(Error::PoolExhausted)));

        let releaser = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            drop(held);
        });

        let started = std::time::Instant::now();
        out.admit_within(Duration::from_secs(5)).unwrap();
        let waited = started.elapsed();
        assert!(
            waited < Duration::from_millis(200),
            "admit_within took {waited:?} — the doorbell missed and only a \
             timeout slice woke it"
        );
        releaser.join().unwrap();
    }

    #[test]
    fn admit_within_gives_up_at_the_deadline() {
        let mut out = OutputArena::new(2);
        out.set_slots(2);
        let _held: Vec<_> = (0..2).map(|_| out.acquire(64, "t").unwrap()).collect();

        let started = std::time::Instant::now();
        let result = out.admit_within(Duration::from_millis(50));
        assert!(matches!(result, Err(Error::PoolExhausted)));
        let waited = started.elapsed();
        assert!(
            waited >= Duration::from_millis(45),
            "returned early: {waited:?}"
        );
        assert!(waited < Duration::from_secs(2), "overslept: {waited:?}");
    }

    #[test]
    fn reset_rebuilds_at_the_new_size() {
        let mut out = OutputArena::new(4);
        out.acquire(1024, "t").unwrap();
        let small = out.arena().unwrap().slot_count();

        out.reset();
        assert!(!out.is_built());
        let slot = out.acquire(64 * 1024, "t").unwrap();
        assert!(
            slot.len() >= 64 * 1024,
            "a resize must widen the slot, not reuse the old one"
        );
        assert_eq!(out.arena().unwrap().slot_count(), small);
    }

    #[test]
    fn a_min_slot_size_leaves_headroom_for_a_bigger_next_frame() {
        // Compressed frame sizes vary, and slot size is fixed at build time.
        let mut out = OutputArena::new(4).with_min_slot_size(1024 * 1024);

        let slot = out.acquire(900, "t").unwrap();
        assert!(slot.len() >= 1024 * 1024, "no headroom: {}", slot.len());
        drop(slot);

        // A later, much larger frame still fits.
        assert!(out.acquire(900_000, "t").is_ok());
    }

    #[test]
    fn a_frame_larger_than_the_slot_is_an_error_not_a_silent_truncation() {
        let mut out = OutputArena::new(4);
        out.acquire(64, "t").unwrap();

        let err = out.acquire(1024 * 1024, "t").unwrap_err();
        assert!(
            format!("{err}").contains("geometry change"),
            "unhelpful message: {err}"
        );
    }

    #[test]
    fn try_acquire_reports_a_full_arena_as_none_not_an_error() {
        // What lets a source answer WouldBlock instead of dying: nothing sheds
        // a source's PoolExhausted.
        let mut out = OutputArena::new(2);
        out.set_slots(2);

        let held: Vec<_> = (0..2)
            .map(|_| out.try_acquire(64, "t").unwrap().unwrap())
            .collect();
        assert!(matches!(out.try_acquire(64, "t"), Ok(None)));

        drop(held);
        assert!(out.try_acquire(64, "t").unwrap().is_some());
    }

    #[test]
    fn try_acquire_still_errors_on_an_oversized_frame() {
        // Not Ok(None): retrying cannot make the frame smaller, so a source
        // told to try again would spin forever.
        let mut out = OutputArena::new(4);
        out.try_acquire(64, "t").unwrap().unwrap();

        assert!(matches!(
            out.try_acquire(1024 * 1024, "t"),
            Err(Error::InvalidSegment(_))
        ));
    }

    #[test]
    fn grow_to_fit_rebuilds_instead_of_erroring() {
        let mut out = OutputArena::new(4).grow_to_fit();
        out.acquire(64, "t").unwrap();

        let slot = out
            .acquire(1024 * 1024, "t")
            .expect("grow_to_fit must widen rather than fail");
        assert!(slot.len() >= 1024 * 1024);
    }

    #[test]
    fn growth_is_one_way() {
        let mut out = OutputArena::new(4).grow_to_fit();
        out.acquire(1024 * 1024, "t").unwrap();
        let wide = out.arena().unwrap().slot_size();

        out.acquire(64, "t").unwrap();
        assert_eq!(
            out.arena().unwrap().slot_size(),
            wide,
            "a smaller payload must reuse the slot it fits in, not shrink it"
        );
    }

    #[test]
    fn slots_are_aligned_for_avx512() {
        // `SharedArena::new` asks for a cache line, which covers every
        // MemoryLayout an element can request. Elements that used to reach for
        // `new_avx` (32-byte) rely on this being at least as strong.
        let mut out = OutputArena::new(4);
        let slot = out.acquire(4096, "t").unwrap();

        assert_eq!(
            slot.data().as_ptr() as usize % 64,
            0,
            "output slots must stay 64-byte aligned"
        );
    }
}
