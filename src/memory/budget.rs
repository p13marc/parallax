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
    /// - the total is clamped to [`MAX_OUTPUT_ARENA_BYTES`] slots' worth. A 4K
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
}
