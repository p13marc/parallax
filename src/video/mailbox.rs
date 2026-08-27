//! The seam between a paced pipeline and a GUI's redraw loop.
//!
//! [`AppVideoSink`](crate::elements::AppVideoSink) fills a [`FrameMailbox`];
//! a host toolkit drains it. The split is deliberate and it is where this
//! differs from `AutoVideoSink`, which owns a window and can therefore wait
//! for a frame's presentation time inside `consume`:
//!
//! - the **sink** owns the anchor handshake, segment mapping, flush
//!   generations and back-pressure — everything that needs a
//!   [`ConsumeContext`](crate::element::ConsumeContext);
//! - the **host** owns per-frame timing, because it is the only side that
//!   knows when the next vertical blank is.
//!
//! A host needs three calls per redraw: [`FrameMailbox::clock`] once to learn
//! how to read running time, then [`FrameMailbox::take_due`] for the frame to
//! show and [`FrameMailbox::next_due`] for when to wake up again.
//!
//! ```no_run
//! # use std::time::Duration;
//! # use parallax::video::FrameMailbox;
//! # fn redraw(mailbox: &FrameMailbox) {
//! let Some(clock) = mailbox.clock() else { return };
//! if let Some(paced) = mailbox.take_due(clock.now(), Duration::from_millis(40)) {
//!     // upload paced.frame and draw it
//! }
//! let wake_at = mailbox.next_due();
//! # let _ = wake_at;
//! # }
//! ```

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use tokio::sync::Notify;

use crate::clock::{Clock, ClockTime};
use crate::pipeline::{EndReason, Pace, SyncHandle, schedule};

use super::VideoFrame;

/// Frames a mailbox holds before the sink starts shedding.
///
/// Three is what `AutoVideoSink`'s display channel holds, for the same
/// reason: deep enough that a late redraw does not starve the screen, shallow
/// enough that a seek's stale frames are gone in one frame interval.
pub const DEFAULT_QUEUE_DEPTH: usize = 3;

/// How long a frame may sit in the mailbox, for latency accounting.
///
/// One frame at 25 fps, matching the max-lateness default — the queue's
/// contribution to end-to-end latency is its depth times this.
const NOMINAL_FRAME: Duration = Duration::from_millis(40);

/// Everything a host needs to read pipeline running time on its own thread.
///
/// Both halves are cheap clones (an `Arc` bump each), which is what lets the
/// host evaluate [`schedule`] where the vertical blank is instead of asking
/// the sink to guess.
#[derive(Clone)]
pub struct PresentationClock {
    clock: Arc<dyn Clock>,
    sync: SyncHandle,
}

impl std::fmt::Debug for PresentationClock {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PresentationClock")
            .field("now", &self.now())
            .field("paused", &self.is_paused())
            .finish()
    }
}

impl PresentationClock {
    /// Wrap a pipeline clock and this node's sync handle.
    pub fn new(clock: Arc<dyn Clock>, sync: SyncHandle) -> Self {
        Self { clock, sync }
    }

    /// Running time now — the number frame targets are expressed in.
    pub fn now(&self) -> ClockTime {
        self.sync.running_time(self.clock.now())
    }

    /// The raw clock reading behind [`now`](Self::now).
    ///
    /// `now()` is this minus the running-time origin, so the difference is
    /// the origin itself — which is what a host measuring its own sync
    /// against another branch needs.
    pub fn clock_time(&self) -> ClockTime {
        self.clock.now()
    }

    /// Whether the pipeline is paused, which freezes [`now`](Self::now).
    ///
    /// A host must distinguish this from a stalled clock: a paused clock is
    /// supposed to stop, so the right response is to park, not to present.
    pub fn is_paused(&self) -> bool {
        self.sync.is_paused()
    }

    /// The current flush epoch — raised the instant a flushing seek is
    /// dispatched, before any in-band event could arrive.
    pub fn flush_epoch(&self) -> u64 {
        self.sync.flush_epoch()
    }

    /// The underlying handle, for a host that wants the rest of the sync API.
    pub fn sync(&self) -> &SyncHandle {
        &self.sync
    }
}

/// A decoded frame and the contract for showing it.
#[derive(Debug, Clone)]
pub struct PacedFrame {
    /// The pixels, and how to read them.
    pub frame: VideoFrame,
    /// Running time this frame is due at.
    ///
    /// [`ClockTime::NONE`] means unpaced — no clock, no timestamp, or `sync`
    /// off — and a host should show it on arrival.
    pub target: ClockTime,
    /// The flush generation this frame belongs to.
    pub generation: u64,
    /// The first frame of a generation, which goes up however late it is.
    ///
    /// After a seek it is the frame the user is waiting to see; dropping it
    /// for lateness would leave the pre-seek picture on screen until the
    /// branch caught up.
    pub must_present: bool,
}

/// What a host can render, injected into the sink's caps.
///
/// The sink cannot know this — only the host knows which pixel formats its
/// shader binds and whether its device can import a dma-buf — so it is told
/// rather than guessed. Defaults to the four presentable formats in CPU
/// memory, which every host can take.
#[derive(Debug, Clone)]
pub struct FrameAcceptance {
    /// Pixel formats the host can render, most preferred first.
    pub formats: Vec<crate::format::PixelFormat>,
    /// Memory types the host can read.
    pub memory: crate::format::MemoryCaps,
}

impl Default for FrameAcceptance {
    fn default() -> Self {
        use crate::format::PixelFormat;
        Self {
            formats: vec![
                PixelFormat::I420,
                PixelFormat::Nv12,
                PixelFormat::Bgra,
                PixelFormat::Rgba,
            ],
            memory: crate::format::MemoryCaps::cpu_only(),
        }
    }
}

/// Counters a host can show and the sink turns into QoS.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct FrameStats {
    /// Frames waiting to be shown.
    pub queued: usize,
    /// Frames the host took.
    pub presented: u64,
    /// Frames dropped for arriving past `max_lateness`.
    pub dropped_late: u64,
    /// Frames dropped because the host stopped draining and the ring filled.
    pub dropped_full: u64,
    /// Frames discarded because a seek superseded the position they belong to.
    pub dropped_stale: u64,
    /// Frames clipped as out-of-segment pre-roll after an ACCURATE seek.
    pub clipped: u64,
    /// Worst lateness observed, in nanoseconds.
    pub worst_late_ns: u64,
}

#[derive(Default)]
struct MailboxState {
    queue: VecDeque<PacedFrame>,
    stats: FrameStats,
    clock: Option<PresentationClock>,
    end: Option<EndReason>,
}

/// A bounded, drop-oldest handoff of paced frames from a sink to a host.
///
/// Cloneable by `Arc`, `Send + Sync`, and usable from any toolkit — nothing
/// here knows what a window is.
pub struct FrameMailbox {
    state: Mutex<MailboxState>,
    /// Rung when a frame arrives on an empty ring, and on end-of-stream.
    data: Notify,
    /// Rung when the host takes a frame, which is what paces the pipeline.
    space: Notify,
    id: u64,
    capacity: usize,
    generation: AtomicU64,
    paused: AtomicBool,
}

impl std::fmt::Debug for FrameMailbox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameMailbox")
            .field("id", &self.id)
            .field("capacity", &self.capacity)
            .field("stats", &self.stats())
            .finish()
    }
}

impl FrameMailbox {
    /// A mailbox holding `capacity` frames (clamped to at least one).
    pub fn new(capacity: usize) -> Self {
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self {
            state: Mutex::new(MailboxState::default()),
            data: Notify::new(),
            space: Notify::new(),
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            capacity: capacity.max(1),
            generation: AtomicU64::new(crate::pipeline::INITIAL_GENERATION),
            paused: AtomicBool::new(false),
        }
    }

    /// A process-unique identity for this mailbox.
    ///
    /// Stable for the mailbox's life and different for every one built, which
    /// is what lets a host key a subscription on it: rebuilding the pipeline
    /// makes a new mailbox with a new id, so frames from the old one cannot
    /// be mistaken for the new stream's.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// How many frames it holds before shedding.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// How to read running time, once the pipeline has delivered a clock.
    ///
    /// `None` until the first buffer arrives: an element only sees the clock
    /// through its `ConsumeContext`, so there is nothing to hand out before
    /// then. A host should treat `None` as "show frames on arrival".
    pub fn clock(&self) -> Option<PresentationClock> {
        self.lock().clock.clone()
    }

    /// Whether the pipeline is paused.
    pub fn is_paused(&self) -> bool {
        self.paused.load(Ordering::Acquire)
    }

    /// The current flush generation. Frames below it are stale.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Why the stream ended, once it has.
    pub fn end_reason(&self) -> Option<EndReason> {
        self.lock().end.clone()
    }

    /// Snapshot of the counters.
    pub fn stats(&self) -> FrameStats {
        let state = self.lock();
        FrameStats {
            queued: state.queue.len(),
            ..state.stats
        }
    }

    /// Rung when a frame lands on an empty ring, or the stream ends.
    ///
    /// Only those two cases: a non-empty ring means the host already knows to
    /// come back, so waking it again per frame would be pure overhead.
    pub fn doorbell(&self) -> &Notify {
        &self.data
    }

    /// Take the frame that should be on screen at `now`, if any.
    ///
    /// Discards stale generations and frames later than `max_lateness`,
    /// counting both. Returns `None` when the next frame is not due yet — ask
    /// [`next_due`](Self::next_due) when to come back.
    pub fn take_due(&self, now: ClockTime, max_lateness: Duration) -> Option<PacedFrame> {
        let generation = self.generation();
        let mut state = self.lock();
        let taken = loop {
            let front = state.queue.front()?;
            if front.generation < generation {
                let _ = state.queue.pop_front();
                state.stats.dropped_stale += 1;
                continue;
            }
            if front.must_present {
                break state.queue.pop_front();
            }
            match schedule(front.target, now, max_lateness) {
                Pace::Present => break state.queue.pop_front(),
                Pace::Wait(_) => return None,
                Pace::DropLate { late_ns } => {
                    let _ = state.queue.pop_front();
                    state.stats.dropped_late += 1;
                    state.stats.worst_late_ns = state.stats.worst_late_ns.max(late_ns);
                }
            }
        };
        if taken.is_some() {
            state.stats.presented += 1;
        }
        drop(state);
        self.space.notify_one();
        taken
    }

    /// Running time the next queued frame is due at.
    ///
    /// `Some(ClockTime::NONE)` means one is waiting with no timing contract —
    /// show it now. `None` means the ring is empty.
    pub fn next_due(&self) -> Option<ClockTime> {
        let generation = self.generation();
        let state = self.lock();
        state
            .queue
            .iter()
            .find(|f| f.generation >= generation)
            .map(|f| f.target)
    }

    /// Whether anything is waiting.
    pub fn has_frame(&self) -> bool {
        !self.lock().queue.is_empty()
    }

    // ---- sink side ----

    /// Publish the clock, once, on the first buffer.
    pub(crate) fn attach_clock(&self, clock: Arc<dyn Clock>, sync: SyncHandle) {
        let mut state = self.lock();
        if state.clock.is_none() {
            state.clock = Some(PresentationClock::new(clock, sync));
        }
    }

    /// Try to enqueue. `Err` hands the frame back because the ring is full.
    pub(crate) fn try_push(&self, frame: PacedFrame) -> std::result::Result<(), PacedFrame> {
        // Deliberately a plain fn: `AsyncSink::consume` is `Send`, so no
        // `MutexGuard` may live across an await. The awaiting wrapper is
        // `AppVideoSink::enqueue`, which drops this guard before it waits.
        let mut state = self.lock();
        if state.queue.len() >= self.capacity {
            return Err(frame);
        }
        let was_empty = state.queue.is_empty();
        state.queue.push_back(frame);
        drop(state);
        if was_empty {
            self.data.notify_one();
        }
        Ok(())
    }

    /// Enqueue, evicting the oldest frame to make room.
    pub(crate) fn push_evicting(&self, frame: PacedFrame) {
        let mut state = self.lock();
        while state.queue.len() >= self.capacity {
            let _ = state.queue.pop_front();
            state.stats.dropped_full += 1;
        }
        let was_empty = state.queue.is_empty();
        state.queue.push_back(frame);
        drop(state);
        if was_empty {
            self.data.notify_one();
        }
    }

    /// A flushing seek: everything queued belongs to the position being left.
    pub(crate) fn retire(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        let mut state = self.lock();
        let dropped = state.queue.len();
        state.queue.clear();
        state.stats.dropped_stale += dropped as u64;
        drop(state);
        // Space appeared, and a host parked on the doorbell should re-evaluate.
        self.space.notify_waiters();
        self.data.notify_waiters();
    }

    /// A `FlushStop` names the generation the next frames belong to.
    pub(crate) fn open(&self, generation: u64) {
        self.generation.store(generation, Ordering::Release);
    }

    pub(crate) fn set_paused(&self, paused: bool) {
        self.paused.store(paused, Ordering::Release);
        if !paused {
            self.data.notify_waiters();
        }
    }

    pub(crate) fn count_clipped(&self) {
        self.lock().stats.clipped += 1;
    }

    pub(crate) fn count_stale(&self) {
        self.lock().stats.dropped_stale += 1;
    }

    /// Record the terminal outcome and wake anyone waiting for a frame.
    pub(crate) fn end(&self, reason: EndReason) {
        let mut state = self.lock();
        if state.end.is_none() {
            state.end = Some(reason);
        }
        drop(state);
        self.data.notify_waiters();
        self.space.notify_waiters();
    }

    /// The counters the sink turns into a QoS report.
    ///
    /// `presented` and `dropped_late` are cumulative — a diagnostics panel
    /// reads them and would be baffled by a counter that goes down — so the
    /// sink keeps its own window baselines. Only the worst-lateness watermark
    /// is taken, because "worst since the last report" is the only useful
    /// reading of it.
    pub(crate) fn presentation_counters(&self) -> (u64, u64, u64) {
        let mut state = self.lock();
        let worst = std::mem::take(&mut state.stats.worst_late_ns);
        (state.stats.presented, state.stats.dropped_late, worst)
    }

    pub(crate) fn space(&self) -> &Notify {
        &self.space
    }

    /// De-poison on the way in: a panicking element task must not take the
    /// host's redraw loop down with it.
    fn lock(&self) -> MutexGuard<'_, MailboxState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl FrameMailbox {
    /// The queue's contribution to end-to-end latency at a nominal 25 fps.
    pub(crate) fn nominal_latency(&self) -> Duration {
        NOMINAL_FRAME * self.capacity as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::Buffer;
    use crate::buffer::MemoryHandle;
    use crate::format::PixelFormat;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;

    fn frame(target_ms: u64, generation: u64) -> PacedFrame {
        let arena = SharedArena::new(4096, 4).expect("arena");
        let slot = arena.acquire().expect("slot");
        let buffer = Buffer::new(MemoryHandle::with_len(slot, 16), Metadata::default());
        PacedFrame {
            frame: VideoFrame {
                data: buffer,
                width: 2,
                height: 2,
                format: PixelFormat::Rgba,
                layout: None,
                color: None,
                display: None,
            },
            target: ClockTime::from_millis(target_ms),
            generation,
            must_present: false,
        }
    }

    fn ms(v: u64) -> ClockTime {
        ClockTime::from_millis(v)
    }

    #[test]
    fn a_frame_is_held_until_it_is_due() {
        let mb = FrameMailbox::new(3);
        mb.try_push(frame(100, 0)).unwrap();
        assert!(mb.take_due(ms(0), Duration::from_millis(40)).is_none());
        assert_eq!(mb.next_due(), Some(ms(100)));
        assert!(mb.take_due(ms(100), Duration::from_millis(40)).is_some());
        assert_eq!(mb.stats().presented, 1);
    }

    #[test]
    fn late_frames_are_skipped_to_reach_the_current_one() {
        let mb = FrameMailbox::new(4);
        for t in [0, 40, 80, 120] {
            mb.try_push(frame(t, 0)).unwrap();
        }
        // 130 ms in with a 40 ms budget: the first three are past saving.
        let got = mb
            .take_due(ms(130), Duration::from_millis(40))
            .expect("the current frame");
        assert_eq!(got.target, ms(120));
        assert_eq!(mb.stats().dropped_late, 3);
        assert!(mb.stats().worst_late_ns > 0);
    }

    #[test]
    fn the_first_frame_of_a_generation_goes_up_however_late() {
        let mb = FrameMailbox::new(3);
        let mut f = frame(0, 0);
        f.must_present = true;
        mb.try_push(f).unwrap();
        // Ten seconds late, and still shown: after a seek this is the frame
        // the user is waiting for.
        let got = mb.take_due(ms(10_000), Duration::from_millis(40));
        assert!(got.is_some());
        assert_eq!(mb.stats().dropped_late, 0);
    }

    #[test]
    fn a_full_ring_hands_the_frame_back_and_eviction_is_counted() {
        let mb = FrameMailbox::new(2);
        mb.try_push(frame(0, 0)).unwrap();
        mb.try_push(frame(40, 0)).unwrap();
        let rejected = mb.try_push(frame(80, 0));
        assert!(rejected.is_err(), "a full ring never silently drops");
        mb.push_evicting(rejected.unwrap_err());
        assert_eq!(mb.stats().dropped_full, 1);
        assert_eq!(mb.stats().queued, 2);
        // The oldest went, not the newest.
        assert_eq!(mb.next_due(), Some(ms(40)));
    }

    #[test]
    fn retiring_a_generation_discards_what_belonged_to_the_old_position() {
        let mb = FrameMailbox::new(4);
        mb.try_push(frame(0, 0)).unwrap();
        mb.try_push(frame(40, 0)).unwrap();
        mb.retire();
        assert!(!mb.has_frame());
        assert_eq!(mb.stats().dropped_stale, 2);

        // A frame stamped with the old generation still cannot be shown.
        mb.open(7);
        mb.try_push(frame(0, 1)).unwrap();
        assert!(mb.take_due(ms(1000), Duration::from_millis(40)).is_none());
        assert_eq!(mb.stats().dropped_stale, 3);
    }

    #[test]
    fn an_unpaced_frame_is_due_immediately() {
        let mb = FrameMailbox::new(2);
        let mut f = frame(0, 0);
        f.target = ClockTime::NONE;
        mb.try_push(f).unwrap();
        assert_eq!(mb.next_due(), Some(ClockTime::NONE));
        assert!(
            mb.take_due(ClockTime::NONE, Duration::from_millis(40))
                .is_some()
        );
    }

    #[test]
    fn mailboxes_are_distinguishable() {
        let a = FrameMailbox::new(1);
        let b = FrameMailbox::new(1);
        assert_ne!(
            a.id(),
            b.id(),
            "a rebuild must not look like the old stream"
        );
    }
}
