//! AppVideoSink — hand paced video frames to a GUI toolkit.
//!
//! [`AutoVideoSink`](super::AutoVideoSink) opens its own window, which means
//! it owns the process's one winit event loop. An application built on Iced,
//! egui or anything else owns that loop itself, so the two cannot coexist and
//! a GUI player needs a different sink.
//!
//! This is it. Everything that needs a
//! [`ConsumeContext`](crate::element::ConsumeContext) stays here — the anchor
//! handshake, segment mapping, flush generations, out-of-segment clipping,
//! QoS, back-pressure — and the frame, with the running time it is due at,
//! goes into a [`FrameMailbox`] the host drains from its redraw loop.
//!
//! Per-frame *timing* is deliberately the host's: it is the only side that
//! knows when the next vertical blank is, and
//! [`schedule`](crate::pipeline::schedule) is a pure function it can call
//! where that knowledge lives.
//!
//! ```no_run
//! use parallax::elements::AppVideoSink;
//! use parallax::pipeline::{Executor, Pipeline};
//!
//! # fn demo(pipeline: &mut Pipeline) -> parallax::error::Result<()> {
//! let mut sink = AppVideoSink::new().with_sync(true);
//! // Before `start`, like every control handle.
//! let frames = sink.mailbox();
//! let node = pipeline.add_async_sink("display", sink);
//! # let _ = (frames, node);
//! # Ok(())
//! # }
//! ```
//!
//! # Back-pressure, and what happens when the host stops
//!
//! `consume` waits for room in the mailbox. Room appears when the host takes
//! a frame, which happens at presentation rate — so that wait *is* what makes
//! the pipeline run at playback speed instead of as fast as the disk allows.
//!
//! A host can stop draining entirely: an occluded or minimised window stops
//! receiving redraws, and a compositor that withholds frame callbacks can
//! block the whole GUI event loop inside its own present call. Waiting there
//! fills the video link, backs up the demuxer, starves the audio branch,
//! underruns the device and freezes the master clock — a hidden window becomes
//! a stopped pipeline.
//!
//! Bounding each wait is *not* enough, and that is the part worth reading
//! twice. Every rate in a paced graph is defined by every other one: this
//! sink's consumption sets the demuxer's routing rate, which sets the audio
//! branch's feed rate, which sets playback, which drives the master clock,
//! which — through [`schedule`](crate::pipeline::schedule) — decides when this
//! sink consumes again. That loop is self-consistent at *any* rate. Shedding
//! one frame per wall-clock deadline just chooses a slow one and stays there;
//! the pipeline settles at a fraction of playback speed and never climbs out,
//! because nothing in the loop is pulling it back up.
//!
//! So once the host has missed a deadline this sink stops waiting on it and
//! paces itself instead: it holds each frame until its presentation time and
//! then sheds it. The pipeline runs at exactly playback speed with nobody
//! watching, and the picture is the only thing that pays.
//!
//! **Pacing here does not starve the graph, and the difference is worth being
//! precise about**, because the obvious reading says it must. The sink
//! consumes one frame per frame interval, so it unblocks the demuxer at
//! exactly the rate the other branches are being drained at; the video link
//! stays full, the demuxer blocks on *that* pad, and audio is fed at 1x. What
//! puts the sink inside the bad loop is consuming *slower* than the frames
//! come due — which is what a wall-clock deadline does and what a due-time
//! wait cannot. A clock that stops for reasons no pause explains is the one
//! case left, and `StallWatch` sheds through it.
//!
//! The tempting alternative — shed at source rate and let some other branch's
//! device do the pacing — is worse, and measurably so. With no back-pressure
//! at all the demuxer runs as far ahead as the *other* branch's link allows;
//! in `parallax-player` that is 2.05 s of audio against 1.24 s of video, so
//! frames reach the mailbox 2 s early, `schedule`'s
//! [`MAX_WAIT`](crate::pipeline::MAX_WAIT) hatch presents them on arrival, and
//! the picture runs two seconds ahead of the sound. That is
//! `a_host_that_comes_back_gets_a_picture_in_sync` measuring 2050 ms.
//!
//! The latch clears the moment the host takes a frame again. One rule around
//! it earns its place: a host that has taken nothing because nothing is *due*
//! is a correctly paced host, not a stopped one, and latching at it would
//! throw the picture away for doing its job (see `host_is_merely_waiting`).
//!
//! A *paused* pipeline is excluded throughout: the clock is supposed to stop
//! there, and the executor gates this task anyway.

use std::sync::Arc;
use std::time::Duration;

use crate::clock::ClockTime;
use crate::element::{AsyncSink, ConsumeContext};
use crate::error::Result;
use crate::event::SegmentEvent;
use crate::pipeline::{Anchor, AnchorGate, EndReason, Pace, schedule};
use crate::video::{
    DEFAULT_QUEUE_DEPTH, FrameAcceptance, FrameMailbox, FrameView, PacedFrame, PresentationClock,
    VideoFrame,
};

/// Default lateness past which a frame is dropped rather than shown.
///
/// One frame at 25 fps. A frame this late is already superseded by the one
/// behind it, so showing it costs the *next* frame its slot too.
const DEFAULT_MAX_LATENESS: Duration = Duration::from_millis(40);

/// How long `consume` waits for the host before deciding it has stopped.
///
/// Same 500 ms as [`STALL_LIMIT`](crate::pipeline::STALL_LIMIT), and for the
/// same reason: past this the host is not merely slow, it has stopped, and
/// continuing to wait would take the rest of the pipeline down with it.
///
/// This is the *entry* condition for the shed latch, paid once. It is not a
/// per-frame budget: charging it to every frame is what pinned the video
/// branch at two frames a second and starved everything behind it.
const DEFAULT_SHED_AFTER: Duration = Duration::from_millis(500);

/// Longest single sleep while pacing a frame this sink is holding itself.
///
/// Short slices rather than one sleep to the presentation time, for the reason
/// `AutoVideoSink`'s pacing loop uses them: a pause, a flush or a re-anchor
/// arriving mid-wait is only observable by looking again.
const PACE_SLICE: Duration = Duration::from_millis(10);

/// What [`AppVideoSink::enqueue`] should do with a frame the host has no room
/// for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ShedMode {
    /// Wait for the host to take one, bounded by `shed_after`. The normal
    /// path: that wait is what makes the pipeline run at playback speed.
    WaitForHost,
    /// Hold the frame until it is *due*, then shed it. What a stopped host
    /// gets: the pipeline keeps running at exactly playback speed and the
    /// picture is what pays.
    PaceOurselves,
}

/// Which of the two behaviours applies to this frame.
///
/// Pure, so the decision is testable without a pipeline, a clock or a host —
/// the same reason [`StallWatch`](crate::pipeline::StallWatch) and
/// [`AnchorGate`](crate::pipeline::AnchorGate) are.
fn shed_mode(stalled: bool, target: ClockTime) -> ShedMode {
    // With nothing to pace against — `sync` off, or no clock yet — the bounded
    // wait is the only brake in the graph, so it stays. Deliberately
    // unchanged: an unsynced sink has no timing contract to keep, and shedding
    // freely would let a hidden window run the source to EOS.
    if !stalled || target == ClockTime::NONE {
        ShedMode::WaitForHost
    } else {
        ShedMode::PaceOurselves
    }
}

/// Length of one QoS aggregation window in running time (#184).
const QOS_WINDOW_NS: u64 = 1_000_000_000;

/// A sink that hands paced frames to application code for rendering.
///
/// The sink owns everything that needs a
/// [`ConsumeContext`](crate::element::ConsumeContext) — the anchor handshake,
/// segment mapping, flush generations, back-pressure — and the host owns
/// per-frame timing, because only it knows when the next vertical blank is.
pub struct AppVideoSink {
    mailbox: Arc<FrameMailbox>,
    name: String,
    /// Present frames at their PTS instead of as fast as they arrive.
    sync: bool,
    /// How late a frame may be and still be shown, when `sync` is on.
    max_lateness: Duration,
    /// How long to wait for the host before shedding.
    shed_after: Duration,
    /// This sink's part of the per-generation anchor handshake (#265).
    anchor: AnchorGate,
    /// Whether the first buffer of the current generation is still owed a
    /// present.
    generation_opened: bool,
    /// The current Time segment, for mapping PTS to running time.
    segment: Option<SegmentEvent>,
    /// PTS of the last buffer that reached the mailbox — the pivot an instant
    /// rate change bends the segment around.
    last_pts: ClockTime,
    /// What the host can render.
    acceptance: FrameAcceptance,
    /// Frames the host holds beyond the mailbox, for the output budget.
    host_retention: usize,
    /// Running-time window boundary for the next QoS report.
    qos_window_start: Option<u64>,
    /// Cumulative counter values at the last report, so the window's deltas
    /// can be derived without resetting what the interface reads.
    qos_baseline: (u64, u64),
    /// The staged QoS event, taken by the executor.
    qos_pending: Option<crate::event::QosEvent>,
    /// Whether the host has stopped taking frames.
    ///
    /// Sticky: set when a wait for room runs out, cleared the moment a push
    /// succeeds. Room appearing *is* the host draining, and it is the only
    /// signal of that this side gets.
    host_stalled: bool,
    /// How long this sink paces against a motionless clock before giving up
    /// on it. Only reachable from [`ShedMode::PaceOurselves`].
    stall_limit: Duration,
}

impl Default for AppVideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl AppVideoSink {
    /// A sink with a default-depth mailbox and pacing off.
    ///
    /// Pacing is opt-in for the same reason it is on `AutoVideoSink`: a
    /// clock-less preview should keep showing frames on arrival.
    pub fn new() -> Self {
        Self::with_queue_depth(DEFAULT_QUEUE_DEPTH)
    }

    /// A sink whose mailbox holds `depth` frames.
    ///
    /// Depth one makes the sink publish a frame only when it is due, which is
    /// the simplest thing a host can consume; deeper lets the host choose the
    /// frame closest to the next vertical blank. Deeper also means more
    /// producer slots pinned — see [`retained_buffers`](AsyncSink::retained_buffers).
    pub fn with_queue_depth(depth: usize) -> Self {
        Self {
            mailbox: Arc::new(FrameMailbox::new(depth)),
            name: "appvideosink".to_string(),
            sync: false,
            max_lateness: DEFAULT_MAX_LATENESS,
            shed_after: DEFAULT_SHED_AFTER,
            anchor: AnchorGate::default(),
            generation_opened: false,
            segment: None,
            last_pts: ClockTime::NONE,
            acceptance: FrameAcceptance::default(),
            host_retention: 1,
            qos_window_start: None,
            qos_baseline: (0, 0),
            qos_pending: None,
            host_stalled: false,
            stall_limit: crate::pipeline::STALL_LIMIT,
        }
    }

    /// The mailbox the host drains. Take it **before** `Executor::start`,
    /// which moves the element into its task.
    pub fn mailbox(&self) -> Arc<FrameMailbox> {
        Arc::clone(&self.mailbox)
    }

    /// Present frames at their timestamps rather than on arrival.
    pub fn with_sync(mut self, sync: bool) -> Self {
        self.sync = sync;
        self
    }

    /// How late a frame may be and still be worth showing.
    pub fn with_max_lateness(mut self, max_lateness: Duration) -> Self {
        self.max_lateness = max_lateness;
        self
    }

    /// How long to wait for the host before concluding it has stopped.
    ///
    /// Paid once per stall, not per frame: past it this sink stops
    /// back-pressuring altogether until the host takes a frame again.
    pub fn with_shed_after(mut self, shed_after: Duration) -> Self {
        self.shed_after = shed_after;
        self
    }

    /// How long to pace against a clock that is not moving before shedding
    /// anyway. Mirrors `AutoVideoSink::with_stall_limit` (no link: it lives
    /// behind the `display` feature, and an intra-doc link into a gated module
    /// resolves in one build and breaks in another).
    ///
    /// Only reaches a graph with no sync master, where this sink is the only
    /// thing pacing anything; elsewhere a stalled host is answered by shedding
    /// at source rate, which never waits on a clock at all.
    pub fn with_stall_limit(mut self, limit: Duration) -> Self {
        self.stall_limit = limit;
        self
    }

    /// How long to wait for the clock-owning sink to fix the running-time
    /// origin before doing it here.
    pub fn with_anchor_grace(mut self, grace: Duration) -> Self {
        self.anchor.set_grace(grace);
        self
    }

    /// Declare what the host can render; drives negotiation.
    pub fn with_acceptance(mut self, acceptance: FrameAcceptance) -> Self {
        self.acceptance = acceptance;
        self
    }

    /// How many frames the host holds beyond the mailbox — the frame it is
    /// currently displaying, plus anything its renderer keeps in flight.
    ///
    /// Folded into [`retained_buffers`](AsyncSink::retained_buffers), which is
    /// what sizes the producer's arena. Under-declaring it starves the decoder.
    pub fn with_host_retention(mut self, frames: usize) -> Self {
        self.host_retention = frames;
        self
    }

    /// Name this element.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Out-of-segment pre-roll after an ACCURATE seek (#165): data starts at
    /// the snapped keyframe, the segment at the requested time.
    fn clips_out_of_segment(&self, pts: ClockTime) -> bool {
        self.sync
            && pts != ClockTime::NONE
            && self
                .segment
                .as_ref()
                .is_some_and(|seg| seg.rate > 0.0 && (pts.nanos() as i64) < seg.start)
    }

    /// Define running-time zero for this generation, or wait for the sink
    /// that owns the clock to do it.
    async fn settle_anchor(
        &mut self,
        ctx: &ConsumeContext<'_>,
        target: ClockTime,
        entered_at: u64,
    ) -> Anchor {
        let Some(sync) = ctx.sync() else {
            return Anchor::Ready;
        };
        match self.anchor.settle(sync, entered_at).await {
            // The clock is read *here*, after any wait the gate did: reading
            // it before would put however long that wait took straight into
            // the origin.
            Anchor::Yours => {
                sync.anchor(self.anchor.generation(), ctx.clock_time(), target);
                Anchor::Ready
            }
            other => other,
        }
    }

    /// Put the frame in the mailbox, by whichever of the three routes applies.
    async fn enqueue(&mut self, frame: PacedFrame) {
        let clock = self.mailbox.clock();
        // No clock yet means no timeline: the first buffer has not been
        // through `consume` anywhere, so there is nothing to pace against.
        let mode = match &clock {
            Some(_) => shed_mode(self.host_stalled, frame.target),
            None => ShedMode::WaitForHost,
        };
        match mode {
            ShedMode::PaceOurselves => {
                let clock = clock.expect("PaceOurselves is only reachable with a clock");
                self.pace_then_shed(frame, &clock).await;
            }
            ShedMode::WaitForHost => self.wait_for_host(frame, clock.as_ref()).await,
        }
    }

    /// Wait for the host to make room, and give up on it after `shed_after`.
    async fn wait_for_host(&mut self, mut frame: PacedFrame, clock: Option<&PresentationClock>) {
        // A local handle, not `self.mailbox`: the `Notified` future borrows the
        // mailbox for as long as it is alive, and the latch transitions below
        // need `&mut self` inside that span.
        let mailbox = Arc::clone(&self.mailbox);
        let mut deadline = tokio::time::Instant::now() + self.shed_after;
        let mut stall = crate::pipeline::StallWatch::new(self.stall_limit);
        // `Some` means the host never made room and the frame has to be shed.
        let shed = loop {
            // Register before the check, and eagerly: `notify_one` from a
            // host taking a frame leaves a permit for a later waiter, but
            // `notify_waiters` from a flush wakes only those already
            // registered. Without `enable()` a flush landing between the
            // check and the first poll would be lost and this would wait on
            // a mailbox nobody is going to drain. (Same reasoning as
            // `AppSink::enqueue`.)
            let mut wait = std::pin::pin!(mailbox.space().notified());
            wait.as_mut().enable();

            match mailbox.try_push(frame) {
                Ok(()) => break None,
                Err(returned) => frame = returned,
            }

            if tokio::time::timeout_at(deadline, wait).await.is_err() {
                // A paused pipeline is not a stalled host: the clock is meant
                // to have stopped, and the executor gates this task on the
                // next message anyway. Give it a fresh window rather than
                // shedding a frame nobody asked us to drop — and a fresh one,
                // not the remains of the old, or the `continue` spins on an
                // already-expired deadline.
                //
                // Read through the clock, not `mailbox.is_paused()`: the
                // mailbox flag is set by *this element's* own
                // `handle_downstream_event`, which the executor calls from
                // this same task, so it cannot change while `consume` is
                // running. The shared flag is raised by `PipelineHandle::pause`
                // before the executor's gate takes effect, which is precisely
                // the window that matters here.
                if clock.is_some_and(|clock| clock.is_paused()) {
                    tokio::time::sleep(PACE_SLICE).await;
                    deadline = tokio::time::Instant::now() + self.shed_after;
                    continue;
                }
                // Nor is a host that has taken nothing because nothing is
                // *due*: that is a correctly paced host, and shedding at it
                // would throw the picture away for doing its job.
                if self.host_is_merely_waiting(&mailbox, clock, &mut stall) {
                    deadline = tokio::time::Instant::now() + self.shed_after;
                    continue;
                }
                break Some(frame);
            }
        };
        match shed {
            Some(frame) => {
                self.host_has_stopped();
                mailbox.push_evicting(frame);
            }
            None => self.host_is_back(),
        }
    }

    /// Whether the host has a reason not to have taken anything yet.
    ///
    /// Distinguishing "stopped" from "waiting" is the whole difficulty. The
    /// case that forced this: coming back from a stall, the queue is dated
    /// ahead of the clock by however far the un-back-pressured demuxer ran,
    /// so the host cannot present for as long as that lead takes to discharge
    /// — and treating that as a stall re-latches the shed, drains the link
    /// again and rebuilds the lead. The two states then alternate forever and
    /// the picture arrives in bursts.
    ///
    /// The exception is a clock that has stopped: the front will never come
    /// due, and the reason may well be that this very wait is starving the
    /// branch that drives it.
    fn host_is_merely_waiting(
        &self,
        mailbox: &FrameMailbox,
        clock: Option<&PresentationClock>,
        stall: &mut crate::pipeline::StallWatch,
    ) -> bool {
        let Some(clock) = clock else { return false };
        let now = clock.now();
        if stall.stalled(now, clock.is_paused()) {
            return false;
        }
        matches!(
            mailbox
                .next_due()
                .map(|due| schedule(due, now, self.max_lateness)),
            Some(Pace::Wait(_))
        )
    }

    /// Hold the frame until it is due, then shed it.
    ///
    /// With the host gone this sink is the only thing keeping the graph at
    /// playback speed, so pacing has to happen here. Shaped like
    /// `AutoVideoSink`'s pacing loop, and for the same reasons.
    ///
    /// Waiting here does not starve the rest of the graph, which is the part
    /// that is easy to get wrong: the sink consumes one frame per frame
    /// interval, so it unblocks the demuxer at exactly the rate the other
    /// branches are being drained at. What it must never do is consume
    /// *slower* than that — which is precisely what a wait bounded by a
    /// wall-clock deadline, rather than by the frame's own due time, does.
    async fn pace_then_shed(&mut self, mut frame: PacedFrame, clock: &PresentationClock) {
        let entered_at = clock.flush_epoch();
        let mut stall = crate::pipeline::StallWatch::new(self.stall_limit);
        loop {
            // Checked every slice, not just at the top: the host coming back
            // is what ends the latch, and this is the only place it shows.
            match self.mailbox.try_push(frame) {
                Ok(()) => return self.host_is_back(),
                Err(returned) => frame = returned,
            }

            // A seek dispatched while we waited abandoned this position.
            if clock.flush_epoch() > entered_at {
                self.mailbox.count_stale();
                return;
            }
            if clock.is_paused() {
                tokio::time::sleep(PACE_SLICE).await;
                continue;
            }

            let now = clock.now();
            // A clock that stops for reasons no pause explains means some
            // branch is starved — possibly by this very wait. Shed and move
            // on, which is what unblocks it.
            if stall.stalled(now, clock.is_paused()) {
                break;
            }
            match schedule(frame.target, now, self.max_lateness) {
                Pace::Wait(delay) => tokio::time::sleep(delay.min(PACE_SLICE)).await,
                Pace::Present | Pace::DropLate { .. } => break,
            }
        }
        self.mailbox.push_evicting(frame);
    }

    /// The host missed its deadline. Said once per stall, not per frame.
    fn host_has_stopped(&mut self) {
        if !self.host_stalled {
            self.host_stalled = true;
            tracing::warn!(
                "{}: the host has stopped taking frames after {:?}; shedding video so the \
                 rest of the pipeline keeps running at playback speed",
                self.name,
                self.shed_after
            );
        }
    }

    /// A push succeeded, so the host is draining again.
    fn host_is_back(&mut self) {
        if self.host_stalled {
            self.host_stalled = false;
            tracing::info!("{}: the host is taking frames again", self.name);
        }
    }

    /// Fold the host's presentation counters into a rate-limited QoS report.
    ///
    /// Counting at presentation rather than at enqueue is the difference from
    /// `AutoVideoSink`: this sink does not decide whether a frame was shown,
    /// so `proportion` can only mean anything if it comes from the side that
    /// did.
    fn roll_qos(&mut self, now_ns: u64) {
        let start = *self.qos_window_start.get_or_insert(now_ns);
        if now_ns.saturating_sub(start) < QOS_WINDOW_NS {
            return;
        }
        let (presented, dropped_late, worst_late_ns) = self.mailbox.presentation_counters();
        let processed = presented.saturating_sub(self.qos_baseline.0);
        let dropped = dropped_late.saturating_sub(self.qos_baseline.1);
        self.qos_baseline = (presented, dropped_late);
        if dropped > 0 {
            self.qos_pending = Some(crate::event::QosEvent {
                qos_type: crate::event::QosType::Underflow,
                proportion: (processed + dropped) as f64 / processed.max(1) as f64,
                jitter_ns: worst_late_ns as i64,
                timestamp: ClockTime::from_nanos(now_ns),
                processed,
                dropped,
            });
        }
        self.qos_window_start = Some(now_ns);
    }
}

impl AsyncSink for AppVideoSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let view = FrameView::from_metadata(ctx.metadata(), ctx.len())?;

        // The clock reaches an element only through its context, so the first
        // buffer is the earliest the host can be told how to read running
        // time. Until then `FrameMailbox::clock()` is `None` and a host shows
        // frames on arrival.
        if let (Some(clock), Some(sync)) = (ctx.clock(), ctx.sync()) {
            self.mailbox.attach_clock(clock.clone(), sync.clone());
        }

        let pts = ctx.metadata().pts;
        if self.clips_out_of_segment(pts) {
            self.mailbox.count_clipped();
            return Ok(());
        }
        self.last_pts = pts;

        // Map through the current segment (#165): (pts − start)/|rate| + base.
        // A PTS past the segment stop has no valid mapping — `NONE` reaches
        // the host as "show it now", which is what `AutoVideoSink` does too.
        let target = if self.sync
            && ctx.clock().is_some()
            && let Some(pts) = pts.to_option()
            && ctx.running_time().is_some()
        {
            self.segment
                .as_ref()
                .map_or(ctx.metadata().pts, |seg| seg.to_running_time(pts))
        } else {
            ClockTime::NONE
        };

        if target != ClockTime::NONE {
            // The flush epoch this frame is being handled under. A seek
            // raises it the moment it is dispatched, which is the only signal
            // that reaches a sink already inside the anchor wait (#265).
            let entered_at = ctx.sync().map(|s| s.flush_epoch()).unwrap_or(0);
            if self.settle_anchor(ctx, target, entered_at).await == Anchor::Superseded {
                self.mailbox.count_stale();
                return Ok(());
            }
            self.roll_qos(ctx.running_time().nanos());
        }

        let must_present = std::mem::take(&mut self.generation_opened);
        self.enqueue(PacedFrame {
            // Refcount bump, not a copy: the host maps the same arena slot —
            // or pins the same decoder picture (#194) — the producer wrote.
            frame: VideoFrame::new(ctx.buffer().clone(), view),
            target,
            generation: self.mailbox.generation(),
            must_present,
        })
        .await;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }

    /// Slots the producer must keep beyond the link's own capacity: the
    /// mailbox's depth plus whatever the host declared it holds.
    fn retained_buffers(&self) -> usize {
        self.mailbox.capacity() + self.host_retention
    }

    /// Pacing lateness plus the mailbox's own hold time.
    ///
    /// The queue is real held time — unlike `AutoVideoSink`, which hands each
    /// frame straight to its display thread — so a pipeline that does not
    /// count it under-reports its latency by the depth of this ring.
    fn latency(&self) -> Option<crate::pipeline::seek::LatencyRange> {
        self.sync.then(|| {
            let held = self.max_lateness + self.mailbox.nominal_latency();
            crate::pipeline::seek::LatencyRange::up_to(
                ClockTime::from_nanos(held.as_nanos() as u64),
            )
        })
    }

    /// QoS origination (#184): the executor polls this after each consume.
    fn take_upstream_event(&mut self) -> Option<crate::event::Event> {
        self.qos_pending.take().map(crate::event::Event::Qos)
    }

    fn handle_upstream_event(&mut self, event: &crate::event::Event) -> crate::event::EventResult {
        use crate::event::{Event, EventResult, SeekFlags};
        if let Event::Seek(seek) = event
            && seek.flags.contains(SeekFlags::INSTANT_RATE_CHANGE)
        {
            let Some(seg) = self.segment.as_mut() else {
                return EventResult::NotHandled;
            };
            if seek.rate == 0.0 || seek.rate.signum() != seg.rate.signum() {
                tracing::warn!(
                    "appvideosink: instant rate {} rejected (direction change from {})",
                    seek.rate,
                    seg.rate
                );
                return EventResult::NotHandled;
            }
            // Pivot on the last frame seen, or the segment start when nothing
            // has played yet: `to_running_time(pivot)` must read the same
            // before and after.
            let pivot = self
                .last_pts
                .to_option()
                .unwrap_or(ClockTime::from_nanos(seg.start.max(0) as u64));
            let before = seg.to_running_time(pivot);
            seg.rate = seek.rate;
            if let (Some(before), Some(after)) =
                (before.to_option(), seg.to_running_time(pivot).to_option())
            {
                seg.base += before.nanos() as i64 - after.nanos() as i64;
            }
            return EventResult::handled();
        }
        EventResult::NotHandled
    }

    fn handle_downstream_event(
        &mut self,
        event: crate::event::Event,
    ) -> Option<crate::event::Event> {
        use crate::event::Event;
        match &event {
            // Everything queued belongs to the position being left. Bumping
            // the generation is what stops the host blitting the pre-seek
            // picture back over the new one.
            Event::FlushStart => self.mailbox.retire(),
            Event::FlushStop(stop) => {
                self.anchor.open(stop.epoch);
                self.mailbox.open(stop.epoch);
                self.generation_opened = true;
            }
            Event::Segment(seg) if seg.format == crate::event::SegmentFormat::Time => {
                self.segment = Some(seg.clone());
            }
            // `AutoVideoSink` can ignore these and let the frozen clock stall
            // its pacing wait. This sink queues, so a host polling the
            // mailbox would spin against a clock going nowhere; tell it.
            Event::Pause => self.mailbox.set_paused(true),
            Event::Resume => self.mailbox.set_paused(false),
            Event::Eos => self.mailbox.end(EndReason::Eos),
            Event::Error(err) => self.mailbox.end(EndReason::Error(err.clone())),
            _ => {}
        }
        Some(event)
    }

    async fn finish(&mut self, reason: EndReason) -> Result<()> {
        // The host is watching the mailbox, not the bus: without this a clean
        // stop looks identical to a stream that simply went quiet.
        self.mailbox.end(reason);
        Ok(())
    }

    fn input_caps(&self) -> crate::format::Caps {
        crate::format::Caps::video_raw_any_resolution(
            self.acceptance
                .formats
                .first()
                .copied()
                .unwrap_or(crate::format::PixelFormat::Rgba),
        )
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryLayout, VideoFormatCaps,
        };
        ElementMediaCaps::new([FormatMemoryCap::new(
            FormatCaps::VideoRaw(VideoFormatCaps {
                width: CapsValue::Any,
                height: CapsValue::Any,
                pixel_format: CapsValue::List(self.acceptance.formats.clone()),
                framerate: CapsValue::Any,
                layout: MemoryLayout::NONE,
            }),
            self.acceptance.memory.clone(),
        )])
    }
}

impl Drop for AppVideoSink {
    fn drop(&mut self) {
        // A sink dropped without `finish` (an `abort()`, or a build that was
        // never started) still has to release a host parked on the doorbell.
        if self.mailbox.end_reason().is_none() {
            self.mailbox.end(EndReason::Aborted);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::{MemoryCaps, PixelFormat};

    #[test]
    fn the_mailbox_outlives_the_element() {
        let sink = AppVideoSink::new();
        let mailbox = sink.mailbox();
        let id = mailbox.id();
        drop(sink);
        // Still readable, and told the stream is over rather than left
        // waiting on a doorbell nobody will ring.
        assert_eq!(mailbox.id(), id);
        assert_eq!(mailbox.end_reason(), Some(EndReason::Aborted));
    }

    #[test]
    fn retention_is_the_ring_plus_what_the_host_declares() {
        let sink = AppVideoSink::with_queue_depth(4).with_host_retention(3);
        assert_eq!(sink.retained_buffers(), 7);
        // The default host holds exactly the frame on screen.
        assert_eq!(AppVideoSink::with_queue_depth(2).retained_buffers(), 3);
    }

    #[test]
    fn latency_counts_the_queue_and_only_when_pacing() {
        let unpaced = AppVideoSink::new();
        assert!(unpaced.latency().is_none(), "no pacing, no added latency");

        let paced = AppVideoSink::with_queue_depth(3).with_sync(true);
        let declared = paced.latency().expect("paced sinks declare latency");
        // 40 ms of lateness budget plus three frames of queue at 25 fps.
        assert_eq!(declared.max, ClockTime::from_millis(160));
    }

    /// The policy in one place.
    mod shedding {
        use super::*;

        #[test]
        fn a_draining_host_is_never_shed_against() {
            for target in [ClockTime::NONE, ClockTime::from_millis(40)] {
                assert_eq!(shed_mode(false, target), ShedMode::WaitForHost);
            }
        }

        /// A stopped host costs the picture, not the pipeline: the sink paces
        /// itself so the graph keeps running at exactly playback speed.
        #[test]
        fn a_stopped_host_makes_this_sink_the_pacing() {
            assert_eq!(
                shed_mode(true, ClockTime::from_millis(40)),
                ShedMode::PaceOurselves
            );
        }

        /// With no timeline there is nothing to pace against, so the bounded
        /// wait stays — it is the only brake an unsynced graph has.
        #[test]
        fn with_no_timeline_the_bounded_wait_stays() {
            assert_eq!(shed_mode(true, ClockTime::NONE), ShedMode::WaitForHost);
        }
    }

    #[test]
    fn caps_come_from_the_host_not_from_a_guess() {
        let sink = AppVideoSink::new().with_acceptance(FrameAcceptance {
            formats: vec![PixelFormat::Nv12],
            memory: MemoryCaps::external_or_cpu(),
        });
        let caps = sink.input_media_caps();
        let rendered = format!("{caps:?}");
        assert!(rendered.contains("Nv12"), "{rendered}");
        assert!(!rendered.contains("I420"), "{rendered}");
    }

    #[test]
    fn out_of_segment_preroll_is_clipped_only_when_pacing() {
        let mut sink = AppVideoSink::new().with_sync(true);
        sink.segment = Some(SegmentEvent::new_time(ClockTime::from_millis(3000), None));
        assert!(sink.clips_out_of_segment(ClockTime::from_millis(2500)));
        assert!(!sink.clips_out_of_segment(ClockTime::from_millis(3500)));
        // Unpaced previews keep every frame.
        sink.sync = false;
        assert!(!sink.clips_out_of_segment(ClockTime::from_millis(2500)));
    }
}
