//! Headless A/V-sync scaffolding: an audio sink that owns the clock and a
//! video sink that paces against it, without a sound card or a display.
//!
//! The real `AlsaSink` needs a device and the real `AutoVideoSink` needs a
//! window, so neither can run in CI. These two stand in for them — but they
//! deliberately reuse the *shared* machinery those sinks use
//! (`pipeline::sync::{clip_to_segment, AnchorGate, schedule}`), so what the
//! tests prove about the contract is what the real sinks obey.
//!
//! [`ClockAudioSink`] is the master, exactly as `AlsaSink` is: it provides
//! the pipeline clock, and that clock counts the audio it has actually
//! "played". It plays in real time, so the whole graph runs at stream rate —
//! the fixtures are one and two seconds long.

use parallax::clock::{Clock, ClockProvider, ClockTime};
use parallax::element::{AsyncSink, ConsumeContext};
use parallax::error::Result;
use parallax::event::{Event, SegmentEvent, SegmentFormat};
use parallax::pipeline::{
    Anchor, AnchorGate, Pace, StallWatch, clip_to_segment, frames_to_nanos, schedule,
};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// What a sink saw for one buffer, in the terms A/V sync is argued in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Presented {
    /// The buffer's own timestamp.
    pub pts: ClockTime,
    /// Its segment running time — where the *stream* says it belongs.
    pub running_time: ClockTime,
    /// The clock reading when it was presented or heard.
    pub at: ClockTime,
}

impl Presented {
    /// The origin this presentation implies: the clock reading that maps to
    /// running time zero.
    ///
    /// Two sinks are in sync exactly when they agree on this number. That is
    /// the whole invariant, and it is what the arrival-anchored pacer could
    /// not maintain across a seek.
    pub fn implied_base_ns(&self) -> i128 {
        self.at.nanos() as i128 - self.running_time.nanos() as i128
    }
}

/// A clock reporting media a device has consumed, like `AlsaClock`.
#[derive(Debug, Default)]
pub struct PlayedClock {
    played: AtomicU64,
    /// Set at EOS, mirroring `AlsaPosition::release`: the stream *ending* is
    /// not a stall, and a frozen master clock would park every other paced
    /// branch forever instead of letting it present its tail.
    released: Mutex<Option<(std::time::Instant, ClockTime)>>,
}

impl PlayedClock {
    /// Advance by `d` of played media.
    pub fn advance(&self, d: ClockTime) {
        self.played.fetch_add(d.nanos(), Ordering::AcqRel);
    }

    /// The stream ended: run at wall rate from here.
    pub fn release(&self) {
        let base = ClockTime::from_nanos(self.played.load(Ordering::Acquire));
        let mut guard = self.released.lock().unwrap_or_else(|e| e.into_inner());
        guard.get_or_insert((std::time::Instant::now(), base));
    }
}

impl Clock for PlayedClock {
    fn now(&self) -> ClockTime {
        let played = ClockTime::from_nanos(self.played.load(Ordering::Acquire));
        match *self.released.lock().unwrap_or_else(|e| e.into_inner()) {
            Some((at, base)) => {
                let wall = ClockTime::from_nanos(at.elapsed().as_nanos() as u64);
                base.saturating_add(wall).max(played)
            }
            None => played,
        }
    }

    fn name(&self) -> &str {
        "test-played-clock"
    }
}

struct PlayedClockProvider {
    clock: Arc<PlayedClock>,
    priority: u32,
}

impl ClockProvider for PlayedClockProvider {
    fn provide_clock(&self) -> Option<Arc<dyn Clock>> {
        Some(self.clock.clone() as Arc<dyn Clock>)
    }

    fn clock_priority(&self) -> u32 {
        self.priority
    }
}

/// Everything a test wants to read back from a sink after the run.
#[derive(Debug, Default)]
pub struct Log {
    /// One entry per buffer that reached the device or the screen.
    pub presented: Vec<Presented>,
    /// Buffers dropped for being outside the segment, in frames for audio
    /// and whole buffers for video.
    pub clipped: u64,
    /// Buffers dropped for arriving too late to be worth showing.
    pub late: u64,
    /// Time segments seen, in order.
    pub segments: Vec<SegmentEvent>,
    /// Flush generations opened, in order.
    pub generations: Vec<u64>,
    /// Times this sink gave up on a motionless clock.
    pub stalled: u64,
    /// Buffers abandoned mid-wait because a seek superseded them.
    pub superseded: u64,
}

impl Log {
    /// Entries at or after `from` on the clock — i.e. after a seek landed.
    pub fn after(&self, from: ClockTime) -> Vec<Presented> {
        self.presented
            .iter()
            .copied()
            .filter(|p| p.at >= from)
            .collect()
    }
}

/// A handle onto a sink's log, cloned before `start()` moves the element.
pub type LogHandle = Arc<Mutex<Log>>;

pub fn record(log: &LogHandle, f: impl FnOnce(&mut Log)) {
    f(&mut log.lock().unwrap_or_else(|e| e.into_inner()))
}

/// The stand-in for `AlsaSink`: provides the pipeline clock, clips to the
/// segment, anchors the shared base, and plays in real time.
pub struct ClockAudioSink {
    clock: Arc<PlayedClock>,
    provider: PlayedClockProvider,
    sample_rate: u32,
    frame_size: usize,
    segment: Option<SegmentEvent>,
    anchor: AnchorGate,
    log: LogHandle,
}

impl ClockAudioSink {
    /// A sink for `sample_rate` Hz, `channels`-channel PCM of
    /// `bytes_per_sample`-wide samples.
    pub fn new(sample_rate: u32, channels: u32, bytes_per_sample: usize) -> Self {
        let clock = Arc::new(PlayedClock::default());
        Self {
            provider: PlayedClockProvider {
                clock: Arc::clone(&clock),
                // Above any software clock, like AlsaSink's 150.
                priority: 150,
            },
            clock,
            sample_rate,
            frame_size: bytes_per_sample * channels as usize,
            segment: None,
            anchor: AnchorGate::default(),
            log: Arc::new(Mutex::new(Log::default())),
        }
    }

    /// The log, cloned before `start()` moves the element.
    pub fn log(&self) -> LogHandle {
        Arc::clone(&self.log)
    }
}

impl AsyncSink for ClockAudioSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let frames = ctx.input().len() / self.frame_size;
        if frames == 0 {
            return Ok(());
        }
        let pts = ctx.metadata().pts;

        let (skip, frames) = match &self.segment {
            Some(seg) => clip_to_segment(seg, pts, frames, self.sample_rate),
            None => (0, frames),
        };
        if skip > 0 {
            record(&self.log, |l| l.clipped += skip as u64);
        }
        if frames == 0 {
            return Ok(());
        }

        let first = ClockTime::from_nanos(
            pts.nanos()
                .saturating_add(frames_to_nanos(skip as u64, self.sample_rate)),
        );
        let running_time = self
            .segment
            .as_ref()
            .map_or(first, |seg| seg.to_running_time(first));

        // The master's anchor: this sample is heard as the clock reads now
        // (nothing is queued ahead of it in this stand-in).
        if let Some(sync) = ctx.sync()
            && self.anchor.settle(sync, sync.flush_epoch()).await == Anchor::Yours
        {
            sync.anchor(self.anchor.generation(), ctx.clock_time(), running_time);
        }

        let played = ClockTime::from_nanos(frames_to_nanos(frames as u64, self.sample_rate));
        record(&self.log, |l| {
            l.presented.push(Presented {
                pts: first,
                running_time,
                at: ctx.clock_time(),
            })
        });
        // Play it at stream rate: this sink is the pipeline's clock, so this
        // sleep is what makes the whole graph run at real time.
        tokio::time::sleep(Duration::from_nanos(played.nanos())).await;
        self.clock.advance(played);
        Ok(())
    }

    fn name(&self) -> &str {
        "test-clock-audio-sink"
    }

    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        match &event {
            Event::Eos | Event::Error(_) => self.clock.release(),
            Event::FlushStart => self.segment = None,
            Event::FlushStop(stop) => {
                self.anchor.open(stop.epoch);
                record(&self.log, |l| l.generations.push(stop.epoch));
            }
            Event::Segment(seg) if seg.format == SegmentFormat::Time => {
                self.segment = Some(seg.clone());
                record(&self.log, |l| l.segments.push(seg.clone()));
            }
            _ => {}
        }
        Some(event)
    }

    fn as_clock_provider(&self) -> Option<&dyn ClockProvider> {
        Some(&self.provider)
    }
}

/// The stand-in for a paced `AutoVideoSink`: same segment clipping, same
/// anchor handshake, same `schedule` policy, no window.
pub struct PacedVideoSink {
    segment: Option<SegmentEvent>,
    anchor: AnchorGate,
    max_lateness: Duration,
    generation_opened: bool,
    log: LogHandle,
}

impl Default for PacedVideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl PacedVideoSink {
    /// A sink with `AutoVideoSink`'s default 40 ms lateness budget.
    pub fn new() -> Self {
        Self {
            segment: None,
            anchor: AnchorGate::default(),
            max_lateness: Duration::from_millis(40),
            generation_opened: false,
            log: Arc::new(Mutex::new(Log::default())),
        }
    }

    /// The log, cloned before `start()` moves the element.
    pub fn log(&self) -> LogHandle {
        Arc::clone(&self.log)
    }
}

impl AsyncSink for PacedVideoSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let pts = ctx.metadata().pts;
        // Out-of-segment pre-roll, as AutoVideoSink clips it.
        if let Some(seg) = &self.segment
            && seg.rate > 0.0
            && pts != ClockTime::NONE
            && (pts.nanos() as i64) < seg.start
        {
            record(&self.log, |l| l.clipped += 1);
            return Ok(());
        }

        let Some(target) = self
            .segment
            .as_ref()
            .map_or(pts, |seg| seg.to_running_time(pts))
            .to_option()
        else {
            return Ok(());
        };
        let Some(sync) = ctx.sync() else {
            return Ok(());
        };

        let entered_at = sync.flush_epoch();
        match self.anchor.settle(sync, entered_at).await {
            Anchor::Superseded => {
                record(&self.log, |l| l.superseded += 1);
                return Ok(());
            }
            // Read the clock after the gate's wait, not before it.
            Anchor::Yours => {
                sync.anchor(self.anchor.generation(), ctx.clock_time(), target);
            }
            Anchor::Ready => {}
        }

        let mut pace = schedule(target, ctx.running_time(), self.max_lateness);
        let mut stall = StallWatch::default();
        while let Pace::Wait(delay) = pace {
            tokio::time::sleep(delay.min(Duration::from_millis(10))).await;
            // Same two escapes as `AutoVideoSink`: a seek abandons the frame
            // outright, and a clock going nowhere must not hold this sink
            // forever — because this sink holding on is one of the reasons a
            // clock goes nowhere.
            if sync.flush_epoch() > entered_at {
                record(&self.log, |l| l.superseded += 1);
                return Ok(());
            }
            let now = ctx.running_time();
            if stall.stalled(now, sync.is_paused()) {
                record(&self.log, |l| l.stalled += 1);
                break;
            }
            pace = schedule(target, now, self.max_lateness);
        }
        if self.generation_opened {
            self.generation_opened = false;
        } else if matches!(pace, Pace::DropLate { .. }) {
            record(&self.log, |l| l.late += 1);
            return Ok(());
        }

        record(&self.log, |l| {
            l.presented.push(Presented {
                pts,
                running_time: target,
                at: ctx.clock_time(),
            })
        });
        Ok(())
    }

    fn name(&self) -> &str {
        "test-paced-video-sink"
    }

    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        match &event {
            Event::FlushStop(stop) => {
                self.anchor.open(stop.epoch);
                self.generation_opened = true;
                record(&self.log, |l| l.generations.push(stop.epoch));
            }
            Event::Segment(seg) if seg.format == SegmentFormat::Time => {
                self.segment = Some(seg.clone());
                record(&self.log, |l| l.segments.push(seg.clone()));
            }
            _ => {}
        }
        Some(event)
    }
}

/// Stands in for a video decoder starting, and restarting after a flush.
///
/// This is the asymmetry that made the A/V offset arbitrary. The real
/// `H264Decoder` rebuilds its openh264 instance from scratch on `FlushStart`
/// and then has to decode a keyframe before anything comes out, while the
/// audio branch's first buffer is accepted by a freshly emptied device
/// straight away. A pacer that pins running-time zero to its own first
/// arrival therefore pins it however late the video branch happened to be —
/// and holds that offset until the next seek re-rolls it.
///
/// Armed at construction as well as on every flush, because start-up has the
/// same shape: the audio link is ~2 s deep in a real player and audio flows
/// before the first frame clears decode.
pub struct SlowRestart {
    delay: Duration,
    armed: bool,
}

impl SlowRestart {
    /// Stall the first buffer, and the first after each flush, by `delay`.
    pub fn new(delay: Duration) -> Self {
        Self { delay, armed: true }
    }
}

impl parallax::element::AsyncTransform for SlowRestart {
    async fn transform(
        &mut self,
        buffer: parallax::buffer::Buffer,
    ) -> Result<parallax::element::Output> {
        if std::mem::take(&mut self.armed) {
            tokio::time::sleep(self.delay).await;
        }
        Ok(parallax::element::Output::single(buffer))
    }

    fn name(&self) -> &str {
        "test-slow-restart"
    }

    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        if matches!(event, Event::FlushStart) {
            self.armed = true;
        }
        Some(event)
    }
}
