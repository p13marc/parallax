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
//! # Back-pressure, and why it is bounded
//!
//! `consume` waits for room in the mailbox. Room appears when the host takes
//! a frame, which happens at presentation rate — so that wait *is* what makes
//! the pipeline run at playback speed instead of as fast as the disk allows.
//!
//! The wait is bounded, because a host can stop draining entirely: an
//! occluded or minimised window stops receiving redraws. Blocking there would
//! fill the video link, back up the demuxer, starve the audio branch, underrun
//! the device and freeze the master clock — turning a hidden window into a
//! stopped pipeline. Past the deadline the oldest frame is shed instead, which
//! degrades video and leaves audio playing. That is the right trade for a
//! player. A *paused* pipeline is excluded: the clock is supposed to stop
//! there, and the executor gates this task anyway.

use std::sync::Arc;
use std::time::Duration;

use crate::clock::ClockTime;
use crate::element::{AsyncSink, ConsumeContext};
use crate::error::Result;
use crate::event::SegmentEvent;
use crate::pipeline::{Anchor, AnchorGate, EndReason};
use crate::video::{
    DEFAULT_QUEUE_DEPTH, FrameAcceptance, FrameMailbox, FrameView, PacedFrame, VideoFrame,
};

/// Default lateness past which a frame is dropped rather than shown.
///
/// One frame at 25 fps. A frame this late is already superseded by the one
/// behind it, so showing it costs the *next* frame its slot too.
const DEFAULT_MAX_LATENESS: Duration = Duration::from_millis(40);

/// How long `consume` waits for the host to free a slot before shedding.
///
/// Same 500 ms as [`STALL_LIMIT`](crate::pipeline::STALL_LIMIT), and for the
/// same reason: past this the host is not merely slow, it has stopped, and
/// continuing to wait would take the rest of the pipeline down with it.
const DEFAULT_SHED_AFTER: Duration = Duration::from_millis(500);

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

    /// How long to wait for a host that has stopped draining before shedding.
    pub fn with_shed_after(mut self, shed_after: Duration) -> Self {
        self.shed_after = shed_after;
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

    /// Put the frame in the mailbox, waiting for room but never forever.
    async fn enqueue(&self, mut frame: PacedFrame) {
        let deadline = tokio::time::Instant::now() + self.shed_after;
        loop {
            // Register before the check, and eagerly: `notify_one` from a
            // host taking a frame leaves a permit for a later waiter, but
            // `notify_waiters` from a flush wakes only those already
            // registered. Without `enable()` a flush landing between the
            // check and the first poll would be lost and this would wait on
            // a mailbox nobody is going to drain. (Same reasoning as
            // `AppSink::enqueue`.)
            let mut wait = std::pin::pin!(self.mailbox.space().notified());
            wait.as_mut().enable();

            match self.mailbox.try_push(frame) {
                Ok(()) => return,
                Err(returned) => frame = returned,
            }

            if tokio::time::timeout_at(deadline, wait).await.is_err() {
                // A paused pipeline is not a stalled host: the clock is meant
                // to have stopped, and the executor gates this task on the
                // next message anyway. Give it another window rather than
                // shedding a frame nobody asked us to drop.
                if self.mailbox.is_paused() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                    continue;
                }
                self.mailbox.push_evicting(frame);
                return;
            }
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
