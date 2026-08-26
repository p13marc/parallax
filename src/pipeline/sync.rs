//! The pipeline's shared, re-anchorable running-time base.
//!
//! # Why this exists
//!
//! A pacing sink decides when to present a buffer by comparing the buffer's
//! *segment running time* — `(pts - segment.start) / |rate| + segment.base` —
//! against the pipeline clock. That comparison needs a third quantity: the
//! clock reading that corresponds to running time zero. GStreamer calls it
//! `base_time`, and recalculating it on a flushing seek is what keeps the
//! audio and video sinks agreeing about *now*.
//!
//! Parallax used to hand every element a `base_time` **copied by value** at
//! `Executor::start`, so nothing could re-anchor it afterwards. A flushing
//! seek emits one `Segment` per pad with `base = 0` — running time restarts —
//! while an audio-provided master clock keeps counting monotonically upward,
//! so the two numbers stop meaning the same thing the instant you seek.
//! `AutoVideoSink` worked around that with a private, arrival-time anchor:
//! whatever wall-clock instant its first post-seek frame happened to reach the
//! sink became "running time zero for video". `AlsaSink` had no anchor at all.
//! The resulting A/V offset was the arrival skew between the two branches, and
//! it was frozen in until the next seek.
//!
//! [`SyncBase`] replaces the copied value with one shared cell. Reads are a
//! single relaxed-ish atomic load on the hot path; the *decision* to re-anchor
//! is taken under a mutex, because it happens once per sink per seek and being
//! obviously correct matters more there than being fast.
//!
//! # The anchoring contract
//!
//! - **Generations.** Every anchor is taken *for a generation*. Generation 0
//!   is start-up; afterwards the generation is the seek's
//!   [`epoch`](crate::event::SeekEvent::epoch), delivered in band by
//!   [`FlushStopEvent::epoch`](crate::event::FlushStopEvent), so every sink
//!   derives the same number from the same event with no cross-task
//!   handshake. Epochs are monotonic by construction, so a late anchor from an
//!   older seek can never overwrite a newer one.
//! - **The master wins.** The sink whose element provided the selected clock
//!   (`AlsaSink`, priority 150) is the sync master: its clock *is* the
//!   timeline, so its answer is authoritative. It may override a non-master
//!   that anchored first, once per generation. Each element learns its role
//!   from the [`SyncHandle`] the executor hands it, so a caller cannot claim
//!   an authority it was not given.
//! - **Nobody stalls.** A non-master that has waited `ANCHOR_GRACE` without
//!   the master showing up anchors itself. A video-only pipeline has no master
//!   at all and every sink is first-come — which is exactly the old
//!   `PtsPacer` behaviour, except now it is shared rather than private.
//!
//! # Scope: one origin per pipeline
//!
//! The base is pipeline-wide, as GStreamer's `base_time` is. A flushing seek
//! that reaches only some of a multi-source graph's chains still moves the
//! origin for all of them — the chains that did not flush keep their own
//! accumulated `segment.base`, so their running time is continuous, but they
//! are measured against the new origin. Single-source graphs (every player,
//! every capture pipeline) are unaffected; a graph that seeks one of several
//! independent sources while pacing the others should not rely on this.
//!
//! # What re-anchoring does *not* touch
//!
//! The clock itself. It stays monotonic:
//! [`PausableClock`](crate::clock::PausableClock) still folds a pause span out
//! of `now()`, and `AlsaPosition::release` still lets the clock run at wall
//! rate after the audio stream ends. Only the origin moves.

use crate::clock::ClockTime;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// How long a non-master sink waits for the master to anchor a generation
/// before anchoring it itself.
///
/// Long enough to cover a decoder restart on the other branch, short enough
/// that a dead audio branch cannot visibly stall the picture.
pub const ANCHOR_GRACE: Duration = Duration::from_millis(250);

/// The generation used before any seek has happened.
pub const INITIAL_GENERATION: u64 = 0;

/// Whether an element may define running-time zero authoritatively.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SyncRole {
    /// The sink whose element provided the selected clock. Its device *is*
    /// the timeline, so its anchor overrides anyone else's for a generation.
    Master,
    /// Everyone else. May anchor a generation the master has not claimed —
    /// which is what keeps a pipeline with no master, or one whose master
    /// branch has stalled, from waiting forever.
    Slave,
}

/// Which anchor is currently installed.
#[derive(Debug)]
struct AnchorState {
    /// Generation the installed base was anchored for.
    generation: u64,
    /// Whether *any* anchor has been taken yet.
    anchored: bool,
    /// Whether the elected master took this generation's anchor.
    master_claimed: bool,
}

/// The clock reading that maps to running time zero, shared by every element.
///
/// See the [module docs](self) for the contract. Cheap to read, cheap to
/// clone (it lives behind an `Arc`), and safe to anchor from several sink
/// tasks at once.
#[derive(Debug)]
pub struct SyncBase {
    /// The base, in clock nanoseconds. [`ClockTime::NONE`]'s sentinel is
    /// stored verbatim, so a pipeline with no started clock reads back NONE
    /// and every derived running time is NONE too.
    base_ns: AtomicU64,
    /// `generation + 1` of the installed anchor; 0 means "never anchored".
    /// Separate from the mutex so [`Self::is_anchored`] is a plain load on
    /// the polling path.
    anchored_generation: AtomicU64,
    /// The node index of the sync master, if the graph elected one.
    /// `None` means nobody is coming, so a slave anchors immediately instead
    /// of burning [`ANCHOR_GRACE`] first.
    master: Option<usize>,
    state: Mutex<AnchorState>,
}

impl SyncBase {
    /// A base starting at `base_time`, not yet anchored by any sink.
    ///
    /// Pass [`ClockTime::NONE`] when the pipeline has no started clock; every
    /// running time derived from it is then NONE, which is the "consume as
    /// fast as buffers arrive" signal sinks already understand.
    /// `master` is the graph index of the node whose element provided the
    /// selected clock; `None` when no node did.
    pub fn new(base_time: ClockTime, master: Option<usize>) -> Self {
        Self {
            base_ns: AtomicU64::new(base_time.nanos()),
            anchored_generation: AtomicU64::new(0),
            master,
            state: Mutex::new(AnchorState {
                generation: INITIAL_GENERATION,
                anchored: false,
                master_claimed: false,
            }),
        }
    }

    /// Whether any element in this pipeline is the sync master.
    #[inline]
    pub fn has_master(&self) -> bool {
        self.master.is_some()
    }

    /// The role the node at graph index `node` holds.
    #[inline]
    pub fn role_for(&self, node: usize) -> SyncRole {
        match self.master {
            Some(master) if master == node => SyncRole::Master,
            _ => SyncRole::Slave,
        }
    }

    /// The clock time that currently maps to running time zero.
    #[inline]
    pub fn base_time(&self) -> ClockTime {
        ClockTime::from_nanos(self.base_ns.load(Ordering::Acquire))
    }

    /// Running time for a clock reading.
    ///
    /// NONE in, NONE out — and NONE out as well when the pipeline has no
    /// started clock, because the base is NONE then.
    #[inline]
    pub fn running_time(&self, now: ClockTime) -> ClockTime {
        now.saturating_sub(self.base_time())
    }

    /// Whether `generation` (or anything newer) has been anchored.
    #[inline]
    pub fn is_anchored(&self, generation: u64) -> bool {
        match self.anchored_generation.load(Ordering::Acquire) {
            0 => false,
            // `installed` is generation + 1, so this is `>= generation`.
            installed => installed > generation,
        }
    }

    /// The generation currently anchored, if any.
    #[inline]
    pub fn anchored_generation(&self) -> Option<u64> {
        self.anchored_generation
            .load(Ordering::Acquire)
            .checked_sub(1)
    }

    /// Anchor `generation` so that stream running time `rt` maps to clock
    /// reading `now`.
    ///
    /// Returns whether this call installed the anchor. Declines when a newer
    /// generation is already installed, or when this generation is installed
    /// and the caller has no authority to replace it (see the module docs:
    /// the master may override a non-master once).
    ///
    /// A NONE `now` or `rt` is refused rather than stored — an unmappable
    /// buffer must not redefine the timeline.
    pub fn anchor(&self, generation: u64, now: ClockTime, rt: ClockTime, role: SyncRole) -> bool {
        let master = role == SyncRole::Master;
        let (Some(now), Some(rt)) = (now.to_option(), rt.to_option()) else {
            return false;
        };
        // De-poison like the probe/tracer registries: one panicking sink must
        // not turn into a second failure in every other sink task.
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        if state.anchored {
            if generation < state.generation {
                return false;
            }
            if generation == state.generation && (!master || state.master_claimed) {
                return false;
            }
        }

        // Publish the base before the generation, so a reader that observes
        // the new generation observes the base that goes with it.
        self.base_ns
            .store(now.nanos().saturating_sub(rt.nanos()), Ordering::Release);
        state.generation = generation;
        state.anchored = true;
        state.master_claimed = master;
        self.anchored_generation
            .store(generation.saturating_add(1), Ordering::Release);
        true
    }
}

impl Default for SyncBase {
    fn default() -> Self {
        Self::new(ClockTime::NONE, None)
    }
}

/// One element's view of the pipeline's [`SyncBase`], carrying the role the
/// executor gave it.
///
/// Handed to every element through `set_clock` and reachable from
/// `ProduceContext::sync` / `ConsumeContext::sync`. Cloning is an `Arc`
/// bump. The role is baked in so [`anchor`](Self::anchor) cannot be called
/// with an authority the element was not granted.
#[derive(Debug, Clone)]
pub struct SyncHandle {
    base: Arc<SyncBase>,
    role: SyncRole,
}

impl SyncHandle {
    /// Bind `base` to `role`.
    pub fn new(base: Arc<SyncBase>, role: SyncRole) -> Self {
        Self { base, role }
    }

    /// This element's role.
    #[inline]
    pub fn role(&self) -> SyncRole {
        self.role
    }

    /// Whether this element defines running-time zero authoritatively.
    #[inline]
    pub fn is_master(&self) -> bool {
        self.role == SyncRole::Master
    }

    /// Whether *any* element in the pipeline is the master — i.e. whether it
    /// is worth waiting [`ANCHOR_GRACE`] for one.
    #[inline]
    pub fn has_master(&self) -> bool {
        self.base.has_master()
    }

    /// The shared base itself.
    #[inline]
    pub fn base(&self) -> &Arc<SyncBase> {
        &self.base
    }

    /// The clock time that currently maps to running time zero.
    #[inline]
    pub fn base_time(&self) -> ClockTime {
        self.base.base_time()
    }

    /// Running time for a clock reading.
    #[inline]
    pub fn running_time(&self, now: ClockTime) -> ClockTime {
        self.base.running_time(now)
    }

    /// Whether `generation` (or anything newer) has been anchored.
    #[inline]
    pub fn is_anchored(&self, generation: u64) -> bool {
        self.base.is_anchored(generation)
    }

    /// The same base, bound to the role the node at graph index `node`
    /// holds. This is how the executor gives each element its handle without
    /// threading the master's identity through every spawn signature.
    pub fn for_node(&self, node: usize) -> Self {
        Self {
            base: Arc::clone(&self.base),
            role: self.base.role_for(node),
        }
    }

    /// Anchor `generation` with this handle's role. See
    /// [`SyncBase::anchor`].
    pub fn anchor(&self, generation: u64, now: ClockTime, rt: ClockTime) -> bool {
        self.base.anchor(generation, now, rt, self.role)
    }
}

// ============================================================================
// Audio-side helpers shared by every sink that maps stream time to the clock
// ============================================================================
/// The frames of a buffer that fall inside the segment (#265).
///
/// A seek lands the audio track at or *before* the reference track's
/// keyframe — MP4 seeks every non-reference track with `SeekSnap::Before`
/// precisely so playback does not open with an audio gap — so the first
/// post-seek buffer routinely straddles `segment.start`. The video path has
/// clipped that pre-roll since #165 (`SegmentClip` in the decoders, and
/// `AutoVideoSink` as the last line); the audio path clipped it nowhere at
/// all, so the sink played samples the segment says are not in it and the
/// stream-time-to-clock mapping was wrong at its origin.
///
/// Returns `(skip, keep)` in frames: `skip` frames trimmed off the head,
/// `keep` frames written. `keep == 0` means the whole buffer is out of
/// segment. Pure integer arithmetic over the sink's own configured rate, so
/// it is testable without a device — same reasoning as `AlsaSink`'s `rebase_written`.
///
/// Reverse segments are passed through untouched: a PTS below `start` there
/// is the walk's own termination, not pre-roll, which is the same rule
/// `AutoVideoSink`'s own clip follows.
pub fn clip_to_segment(
    segment: &crate::event::SegmentEvent,
    pts: ClockTime,
    frames: usize,
    sample_rate: u32,
) -> (usize, usize) {
    let (Some(pts), true) = (pts.to_option(), segment.rate > 0.0) else {
        return (0, frames);
    };
    let rate = sample_rate.max(1) as u128;
    let pts_ns = pts.nanos() as i128;

    // Head: round *up*, so a frame straddling the boundary is dropped
    // rather than played early.
    let skip = if pts_ns < segment.start as i128 {
        let gap = (segment.start as i128 - pts_ns) as u128;
        let f = gap.saturating_mul(rate).div_ceil(1_000_000_000);
        (f.min(frames as u128)) as usize
    } else {
        0
    };

    // Tail: a seek with a `stop` ends playback there, so a buffer straddling
    // it keeps only the part below. Round *down* for the mirror reason.
    let mut keep = frames - skip;
    if segment.stop >= 0 {
        let first_ns = pts_ns + (skip as u128).saturating_mul(1_000_000_000).div_ceil(rate) as i128;
        if first_ns >= segment.stop as i128 {
            return (skip, 0);
        }
        let room = (segment.stop as i128 - first_ns) as u128;
        let f = room.saturating_mul(rate) / 1_000_000_000;
        keep = keep.min(f as usize);
    }
    (skip, keep)
}

/// How long `frames` last at `sample_rate`, in nanoseconds.
pub fn frames_to_nanos(frames: u64, sample_rate: u32) -> u64 {
    frames
        .saturating_mul(1_000_000_000)
        .checked_div(sample_rate.max(1) as u64)
        .unwrap_or(0)
}

/// Drives one sink's part of the per-generation anchor handshake (#265).
///
/// Held by a pacing sink; see [`SyncBase`] for the contract. The rules it
/// encodes, in one place so every sink follows the same ones:
///
/// - the master anchors immediately — its device's progress *is* the clock;
/// - with no master in the graph, whoever gets there first anchors, which is
///   what a video-only pipeline wants and what the old private per-sink
///   anchor already did;
/// - otherwise wait for the master, but only for `grace`: a stalled or
///   absent audio branch must not freeze the picture.
#[derive(Debug)]
pub struct AnchorGate {
    generation: u64,
    grace: Duration,
    waiting_since: Option<std::time::Instant>,
}

impl Default for AnchorGate {
    fn default() -> Self {
        Self {
            generation: INITIAL_GENERATION,
            grace: ANCHOR_GRACE,
            waiting_since: None,
        }
    }
}

impl AnchorGate {
    /// A gate that waits at most `grace` for the master.
    pub fn with_grace(grace: Duration) -> Self {
        Self {
            grace,
            ..Self::default()
        }
    }

    /// How long this gate waits for the master.
    pub fn grace(&self) -> Duration {
        self.grace
    }

    /// Set how long this gate waits for the master.
    pub fn set_grace(&mut self, grace: Duration) {
        self.grace = grace;
    }

    /// The generation currently being anchored.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// A flush opened a new generation, named in band by `FlushStop`.
    pub fn open(&mut self, generation: u64) {
        self.generation = generation;
        self.waiting_since = None;
    }

    /// Make sure this generation's origin is fixed before pacing against it.
    ///
    /// `target` is the buffer's segment running time — the value that must
    /// map onto `now` on the clock if this sink ends up anchoring.
    pub async fn settle(&mut self, sync: &SyncHandle, now: ClockTime, target: ClockTime) {
        if sync.is_master() || !sync.has_master() {
            sync.anchor(self.generation, now, target);
            return;
        }
        if sync.is_anchored(self.generation) {
            return;
        }
        let started = *self
            .waiting_since
            .get_or_insert_with(std::time::Instant::now);
        while !sync.is_anchored(self.generation) {
            if started.elapsed() >= self.grace {
                tracing::debug!(
                    "sync: no anchor for generation {} after {:?}; anchoring this branch itself",
                    self.generation,
                    self.grace
                );
                sync.anchor(self.generation, now, target);
                return;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    }
}

// ============================================================================
// Presentation pacing
// ============================================================================

/// Longest single wait a pacing sink will ask for.
///
/// A stream whose timestamps jump forward (a bad demuxer, a wrapped RTP
/// clock) must not park the sink for minutes. Past this the buffer is
/// presented instead.
pub const MAX_WAIT: Duration = Duration::from_secs(1);

/// What to do with a buffer whose presentation time is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pace {
    /// Show it now.
    Present,
    /// Show it after this long.
    Wait(Duration),
    /// Too late to be useful — drop it, `late_ns` past the deadline.
    DropLate {
        /// How far past `max_lateness` the buffer was (running time, ns).
        late_ns: u64,
    },
}

/// Decide what to do with a buffer whose segment running time is `target`,
/// given the clock now reads `now` in running time.
///
/// Both are measured against the same shared [`SyncBase`], which is what
/// makes two sinks on two branches agree. There is deliberately no state
/// here: the pacer used to carry a private anchor pinning its first buffer's
/// mapped time to whatever running time it happened to *arrive* at, which
/// silently discarded the segment's absolute mapping on every seek and made
/// the A/V offset the arrival skew between the branches.
pub fn schedule(target: ClockTime, now: ClockTime, max_lateness: Duration) -> Pace {
    let (Some(target), Some(now)) = (target.to_option(), now.to_option()) else {
        return Pace::Present;
    };
    let (target, now) = (target.nanos(), now.nanos());

    if target > now {
        let wait = Duration::from_nanos(target - now);
        return if wait > MAX_WAIT {
            Pace::Present
        } else {
            Pace::Wait(wait)
        };
    }

    let late_ns = now - target;
    if late_ns > max_lateness.as_nanos() as u64 {
        Pace::DropLate { late_ns }
    } else {
        Pace::Present
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(secs: u64) -> ClockTime {
        ClockTime::from_secs(secs)
    }

    #[test]
    fn an_unanchored_base_reports_what_it_was_built_with() {
        let base = SyncBase::new(ClockTime::ZERO, Some(0));
        assert_eq!(base.base_time(), ClockTime::ZERO);
        assert_eq!(base.running_time(t(5)), t(5));
        assert!(!base.is_anchored(INITIAL_GENERATION));
    }

    #[test]
    fn a_none_base_makes_every_running_time_none() {
        let base = SyncBase::default();
        assert_eq!(base.running_time(t(5)), ClockTime::NONE);
    }

    #[test]
    fn anchoring_maps_the_given_running_time_onto_the_given_clock_reading() {
        let base = SyncBase::new(ClockTime::ZERO, Some(0));
        // The first post-seek sample, whose running time is 0, is committed
        // when the clock reads 12 s.
        assert!(base.anchor(16, t(12), ClockTime::ZERO, SyncRole::Master));
        assert_eq!(base.running_time(t(12)), ClockTime::ZERO);
        assert_eq!(base.running_time(t(14)), t(2));
        assert!(base.is_anchored(16));
    }

    #[test]
    fn a_newer_generation_replaces_an_older_one() {
        let base = SyncBase::new(ClockTime::ZERO, Some(0));
        assert!(base.anchor(16, t(12), ClockTime::ZERO, SyncRole::Master));
        assert!(base.anchor(32, t(20), ClockTime::ZERO, SyncRole::Master));
        assert_eq!(base.running_time(t(20)), ClockTime::ZERO);
    }

    #[test]
    fn an_older_generation_never_overwrites_a_newer_one() {
        // Overlapping seeks: the slow branch's anchor for seek N lands after
        // seek N+1 already re-anchored. It must be ignored.
        let base = SyncBase::new(ClockTime::ZERO, Some(0));
        assert!(base.anchor(32, t(20), ClockTime::ZERO, SyncRole::Master));
        assert!(!base.anchor(16, t(12), ClockTime::ZERO, SyncRole::Master));
        assert_eq!(base.running_time(t(20)), ClockTime::ZERO);
    }

    #[test]
    fn the_master_overrides_a_non_master_once_per_generation() {
        let base = SyncBase::new(ClockTime::ZERO, Some(0));
        // Video got there first.
        assert!(base.anchor(16, t(12), ClockTime::ZERO, SyncRole::Slave));
        assert_eq!(base.running_time(t(12)), ClockTime::ZERO);
        // Audio — the clock owner — corrects it.
        assert!(base.anchor(16, t(11), ClockTime::ZERO, SyncRole::Master));
        assert_eq!(base.running_time(t(11)), ClockTime::ZERO);
        // And is not itself overridable within the generation.
        assert!(!base.anchor(16, t(13), ClockTime::ZERO, SyncRole::Master));
        assert!(!base.anchor(16, t(13), ClockTime::ZERO, SyncRole::Slave));
        assert_eq!(base.running_time(t(11)), ClockTime::ZERO);
    }

    #[test]
    fn a_non_master_cannot_replace_a_non_master_within_a_generation() {
        let base = SyncBase::new(ClockTime::ZERO, Some(0));
        assert!(base.anchor(0, t(1), ClockTime::ZERO, SyncRole::Slave));
        assert!(!base.anchor(0, t(2), ClockTime::ZERO, SyncRole::Slave));
        assert_eq!(base.base_time(), t(1));
    }

    #[test]
    fn an_unmappable_anchor_is_refused_rather_than_stored() {
        let base = SyncBase::new(ClockTime::ZERO, Some(0));
        assert!(!base.anchor(0, ClockTime::NONE, ClockTime::ZERO, SyncRole::Master));
        assert!(!base.anchor(0, t(1), ClockTime::NONE, SyncRole::Master));
        assert!(!base.is_anchored(0));
        assert_eq!(base.base_time(), ClockTime::ZERO);
    }

    #[test]
    fn a_running_time_above_the_clock_reading_saturates_instead_of_wrapping() {
        // A sink anchoring a segment whose running time already exceeds the
        // clock (a stream starting mid-timeline against a fresh clock).
        let base = SyncBase::new(ClockTime::ZERO, Some(0));
        assert!(base.anchor(0, t(1), t(5), SyncRole::Master));
        assert_eq!(base.base_time(), ClockTime::ZERO);
    }

    #[test]
    fn is_anchored_answers_for_older_generations_too() {
        let base = SyncBase::new(ClockTime::ZERO, Some(0));
        assert!(base.anchor(48, t(1), ClockTime::ZERO, SyncRole::Master));
        assert!(base.is_anchored(16));
        assert!(base.is_anchored(48));
        assert!(!base.is_anchored(64));
        assert_eq!(base.anchored_generation(), Some(48));
    }
}

#[cfg(test)]
mod pacing_tests {
    use super::*;

    const LATE: Duration = Duration::from_millis(40);

    #[test]
    fn a_buffer_due_later_waits_exactly_that_long() {
        let pace = schedule(
            ClockTime::from_millis(100),
            ClockTime::from_millis(60),
            LATE,
        );
        assert_eq!(pace, Pace::Wait(Duration::from_millis(40)));
    }

    #[test]
    fn a_buffer_within_the_lateness_budget_is_presented() {
        assert_eq!(
            schedule(ClockTime::from_millis(60), ClockTime::from_millis(90), LATE),
            Pace::Present
        );
    }

    #[test]
    fn a_buffer_past_the_lateness_budget_is_dropped() {
        assert_eq!(
            schedule(
                ClockTime::from_millis(60),
                ClockTime::from_millis(200),
                LATE
            ),
            Pace::DropLate {
                late_ns: 140_000_000
            }
        );
    }

    #[test]
    fn an_absurd_wait_presents_instead_of_parking_the_sink() {
        assert_eq!(
            schedule(ClockTime::from_secs(3600), ClockTime::ZERO, LATE),
            Pace::Present
        );
    }

    #[test]
    fn an_unmappable_time_presents_rather_than_guessing() {
        assert_eq!(
            schedule(ClockTime::NONE, ClockTime::ZERO, LATE),
            Pace::Present
        );
        assert_eq!(
            schedule(ClockTime::ZERO, ClockTime::NONE, LATE),
            Pace::Present
        );
    }
}

#[cfg(test)]
mod clip_tests {
    use super::*;
    use crate::event::SegmentEvent;

    const RATE: u32 = 48_000;

    /// 480 frames = 10 ms at 48 kHz.
    fn ms(n: u64) -> ClockTime {
        ClockTime::from_millis(n)
    }

    #[test]
    fn a_buffer_wholly_inside_the_segment_is_untouched() {
        let seg = SegmentEvent::new_time(ms(1_000), None);
        assert_eq!(clip_to_segment(&seg, ms(1_000), 480, RATE), (0, 480));
        assert_eq!(clip_to_segment(&seg, ms(5_000), 480, RATE), (0, 480));
    }

    #[test]
    fn a_buffer_wholly_below_the_segment_is_dropped() {
        let seg = SegmentEvent::new_time(ms(1_000), None);
        // Ends exactly at the boundary: nothing of it is in segment.
        assert_eq!(clip_to_segment(&seg, ms(990), 480, RATE), (480, 0));
        assert_eq!(clip_to_segment(&seg, ms(500), 480, RATE), (480, 0));
    }

    #[test]
    fn a_straddling_buffer_is_trimmed_sample_accurately() {
        // MP4 seeks every non-reference track with SeekSnap::Before, so the
        // first post-seek audio buffer routinely starts below the segment.
        let seg = SegmentEvent::new_time(ms(1_000), None);
        // Starts 5 ms early: 240 frames of pre-roll, 240 kept.
        assert_eq!(clip_to_segment(&seg, ms(995), 480, RATE), (240, 240));
    }

    #[test]
    fn a_frame_straddling_the_boundary_is_dropped_not_played_early() {
        // 1 ns before the start: the first frame is partly out of segment,
        // so the ceiling drops it rather than playing it a hair early.
        let seg = SegmentEvent::new_time(ms(1_000), None);
        let pts = ClockTime::from_nanos(ms(1_000).nanos() - 1);
        assert_eq!(clip_to_segment(&seg, pts, 480, RATE), (1, 479));
    }

    #[test]
    fn a_segment_stop_ends_playback_inside_the_buffer() {
        // A seek with a stop (#165) really ends there.
        let seg = SegmentEvent::new_time(ms(1_000), Some(ms(1_005)));
        assert_eq!(clip_to_segment(&seg, ms(1_000), 480, RATE), (0, 240));
        // Entirely past the stop.
        assert_eq!(clip_to_segment(&seg, ms(1_005), 480, RATE), (0, 0));
    }

    #[test]
    fn head_and_tail_trims_compose() {
        let seg = SegmentEvent::new_time(ms(1_000), Some(ms(1_005)));
        // Starts 5 ms early and runs 5 ms past the stop: only the middle
        // 5 ms survives.
        assert_eq!(clip_to_segment(&seg, ms(995), 720, RATE), (240, 240));
    }

    #[test]
    fn an_unmappable_or_reverse_segment_clips_nothing() {
        let seg = SegmentEvent::new_time(ms(1_000), None);
        assert_eq!(clip_to_segment(&seg, ClockTime::NONE, 480, RATE), (0, 480));
        // Reverse: a PTS below start is the walk's own termination, the same
        // rule AutoVideoSink::clips_out_of_segment follows.
        let rev = SegmentEvent::new_time(ms(0), Some(ms(2_000))).with_rate(-1.0);
        assert_eq!(clip_to_segment(&rev, ms(500), 480, RATE), (0, 480));
    }

    #[test]
    fn frames_convert_to_nanos_without_dividing_by_zero() {
        assert_eq!(frames_to_nanos(48_000, RATE), 1_000_000_000);
        assert_eq!(frames_to_nanos(4_096, RATE), 85_333_333);
        assert_eq!(frames_to_nanos(0, RATE), 0);
        // A zero rate cannot happen (the device configured one), but it must
        // clamp rather than divide by zero.
        assert_eq!(frames_to_nanos(1, 0), 1_000_000_000);
    }
}
