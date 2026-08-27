//! AutoVideoSink - Display video frames in a window.
//!
//! This sink automatically creates a window and displays video frames using
//! winit + softbuffer. Like GStreamer's xvimagesink, it runs its own event
//! thread and doesn't require any special lifecycle management.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::Pipeline;
//!
//! // Simple usage - just works!
//! Pipeline::parse("videotestsrc ! autovideosink")?.run().await?;
//! ```
//!
//! # Architecture
//!
//! The sink spawns a dedicated display thread that:
//! 1. Creates a winit window with `any_thread(true)` (Linux only)
//! 2. Runs its own event loop
//! 3. Receives frames via a bounded channel
//! 4. Blits frames to the window using softbuffer
//!
//! This design mirrors GStreamer's xvimagesink which also runs its own
//! X11 event thread.

use super::present::{self, DisplayFrame};
use crate::clock::ClockTime;
use crate::element::{AsyncSink, ConsumeContext};
use crate::error::{Error, Result};
use crate::format::{Caps, PixelFormat};
use crate::pipeline::{Anchor, Pace, schedule};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use winit::event_loop::EventLoopProxy;

/// The winit user event: "state changed, wake up and look".
///
/// The frame channel stays the data path; this is only the doorbell that lets
/// the event loop idle in [`ControlFlow::Wait`] instead of the historical
/// `ControlFlow::Poll` spin, which burned a full core on `try_recv` (#155).
#[derive(Debug)]
struct WakeUp;

/// The display loop's doorbell, parked where every waker can reach it.
///
/// Filled by the display thread right after it builds the event loop, cleared
/// when the loop exits. `None` simply means there is nothing to wake yet — the
/// loop drains pending state on startup, so a wake missed in that window is
/// harmless.
type SharedProxy = Arc<Mutex<Option<EventLoopProxy<WakeUp>>>>;

/// Ring the display loop's doorbell, if the loop exists.
fn wake_display(proxy: &SharedProxy) {
    let guard = proxy.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(p) = guard.as_ref() {
        // EventLoopClosed means the loop is already gone — nothing to wake.
        let _ = p.send_event(WakeUp);
    }
}

/// Default lateness past which a frame is dropped rather than shown.
///
/// One frame at 25 fps. A frame this late is already superseded by the one
/// behind it, so showing it costs the *next* frame its slot too.
const DEFAULT_MAX_LATENESS: Duration = Duration::from_millis(40);

/// Depth of the consume-task → display-thread frame channel. Each queued
/// frame is a zero-copy `Buffer` clone pinning a producer arena slot, so
/// this plus the presented frame is what `retained_buffers()` reports.
const DISPLAY_QUEUE_DEPTH: usize = 3;

/// Length of one QoS aggregation window in running time (#184).
const QOS_WINDOW_NS: u64 = 1_000_000_000;

/// Aggregates presentation results into rate-limited QoS reports (#184).
///
/// The upstream transport has no dedup or rate limiting for non-seek
/// events, so the origin throttles: counters accumulate over a ≥1 s window
/// of running time and a single [`QosEvent`](crate::event::QosEvent) is
/// staged at the window boundary — and only when something was actually
/// dropped. The executor polls it via `AsyncSink::take_upstream_event`.
#[derive(Debug, Default)]
struct QosAggregator {
    /// Running time when the current window opened.
    window_start: Option<u64>,
    processed: u64,
    dropped: u64,
    /// Worst lateness observed in the window.
    worst_jitter_ns: i64,
    /// The staged event, taken by the executor.
    pending: Option<crate::event::QosEvent>,
}

impl QosAggregator {
    fn presented(&mut self, now_ns: u64) {
        self.processed += 1;
        self.roll(now_ns);
    }

    fn dropped_late(&mut self, now_ns: u64, late_ns: u64) {
        self.dropped += 1;
        self.worst_jitter_ns = self.worst_jitter_ns.max(late_ns as i64);
        self.roll(now_ns);
    }

    fn roll(&mut self, now_ns: u64) {
        let start = *self.window_start.get_or_insert(now_ns);
        if now_ns.saturating_sub(start) < QOS_WINDOW_NS {
            return;
        }
        if self.dropped > 0 {
            self.pending = Some(crate::event::QosEvent {
                qos_type: crate::event::QosType::Underflow,
                proportion: (self.processed + self.dropped) as f64 / self.processed.max(1) as f64,
                jitter_ns: self.worst_jitter_ns,
                timestamp: ClockTime::from_nanos(now_ns),
                processed: self.processed,
                dropped: self.dropped,
            });
        }
        self.window_start = Some(now_ns);
        self.processed = 0;
        self.dropped = 0;
        self.worst_jitter_ns = 0;
    }

    fn take(&mut self) -> Option<crate::event::QosEvent> {
        self.pending.take()
    }
}

/// A video sink that automatically creates a window and displays frames.
///
/// This sink spawns its own display thread with a winit event loop,
/// similar to how GStreamer's xvimagesink works. No special lifecycle
/// management is required - it's just a regular sink.
///
/// # Platform Support
///
/// - **Linux (X11/Wayland)**: Fully supported via `any_thread`
/// - **Windows**: Supported via `any_thread`
/// - **macOS**: Not supported (macOS requires GUI on main thread)
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::app::AutoVideoSink;
/// use parallax::pipeline::Pipeline;
///
/// // Via pipeline string
/// Pipeline::parse("videotestsrc ! autovideosink")?.run().await?;
///
/// // Or programmatically
/// let sink = AutoVideoSink::new();
/// ```
pub struct AutoVideoSink {
    /// Channel sender for frames
    sender: Option<SyncSender<DisplayFrame>>,
    /// Handle to the display thread
    display_thread: Option<JoinHandle<()>>,
    /// Flag to signal shutdown
    running: Arc<AtomicBool>,
    /// Window title
    title: String,
    /// Frame dimensions (detected from first frame if 0)
    width: Arc<AtomicU32>,
    height: Arc<AtomicU32>,
    /// Element name
    name: String,
    /// Present frames at their PTS instead of as fast as they arrive.
    sync: bool,
    /// How late a frame may be and still be shown, when `sync` is on.
    max_lateness: Duration,
    /// This sink's part of the per-generation anchor handshake (#265). A
    /// `FlushStop` names the generation in band, so both branches of an A/V
    /// graph anchor the same one from the same event.
    anchor: crate::pipeline::AnchorGate,
    /// How long to wait on a motionless clock before presenting anyway; see
    /// [`with_stall_limit`](Self::with_stall_limit).
    stall_limit: Duration,
    /// Whether the first buffer of the current generation is still owed a
    /// present. It is the frame the user is waiting to see after a seek, so
    /// it goes up however late it is.
    generation_opened: bool,
    /// PTS of the last buffer that reached the pacer — the pivot an instant
    /// rate change bends the segment around, so the change is continuous at
    /// the position actually being played.
    last_pts: ClockTime,
    /// Frames stamped below this are stale — a flushing seek bumped it. Read
    /// by the display thread, which skips them rather than blitting the
    /// pre-seek picture back over the new one.
    display_generation: Arc<AtomicU64>,
    /// Current segment (#165): PTS map through
    /// [`SegmentEvent::to_running_time`] before pacing, so a segment's
    /// start/rate/base take effect. `None` until the initial segment
    /// arrives (raw-PTS behavior).
    segment: Option<crate::event::SegmentEvent>,
    /// Frames dropped for being late or for a full display channel.
    dropped: u64,
    /// QoS origination (#184): lateness aggregated into rate-limited
    /// upstream reports, polled via `take_upstream_event`.
    qos: QosAggregator,
    /// Window-event channel, created when [`handle`](Self::handle) is called.
    /// `None` means no handle was taken and the event loop sends nothing —
    /// exactly the historical behavior.
    events: Option<(
        crossbeam_channel::Sender<VideoWindowEvent>,
        crossbeam_channel::Receiver<VideoWindowEvent>,
    )>,
    /// Desired fullscreen state, applied by the event loop when woken.
    fullscreen: Arc<AtomicBool>,
    /// Doorbell into the display loop; see [`SharedProxy`].
    proxy: SharedProxy,
    /// Prefer the GPU presentation backend (#190). Only effective when the
    /// `display-gpu` feature is compiled and the process-wide probe finds a
    /// real adapter; the caps flip to accept I420/NV12 exactly then.
    gpu: bool,
}

/// A key press from the video window, winit-agnostic.
///
/// The named variants cover what a player binds (transport control); anything
/// else arrives as [`Character`](Self::Character) text or a named
/// [`Other`](Self::Other).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoKey {
    /// A printable character key, as text (e.g. `"q"`).
    Character(String),
    /// The space bar.
    Space,
    /// Escape.
    Escape,
    /// Enter/Return.
    Enter,
    /// Left arrow.
    ArrowLeft,
    /// Right arrow.
    ArrowRight,
    /// Up arrow.
    ArrowUp,
    /// Down arrow.
    ArrowDown,
    /// Any other key, by its winit debug name.
    Other(String),
}

/// A user-interaction event from the video window.
#[derive(Debug, Clone, PartialEq)]
pub enum VideoWindowEvent {
    /// A key was pressed (repeats are filtered out).
    KeyPressed(VideoKey),
    /// The left mouse button was pressed at this window position.
    MousePressed {
        /// X position in window coordinates.
        x: f64,
        /// Y position in window coordinates.
        y: f64,
    },
    /// The user asked to close the window. The window *will* close and the
    /// sink requests a cooperative pipeline shutdown on its next frame
    /// ([`Error::Shutdown`](crate::error::Error::Shutdown), #191 — the run
    /// ends as `EndReason::Eos`); this event is the zero-latency signal for
    /// apps that want to react before that.
    CloseRequested,
    /// The window was resized.
    Resized {
        /// New inner width in pixels.
        width: u32,
        /// New inner height in pixels.
        height: u32,
    },
}

/// Runtime handle to an [`AutoVideoSink`]'s window.
///
/// Follows the [`Controllable`](crate::control::Controllable) convention:
/// take it (and clone it freely) **before** `Executor::start` moves the sink
/// into its task. Events arrive on a bounded channel — a consumer that stops
/// reading loses the oldest events, never blocks the window.
///
/// ```rust,ignore
/// let mut sink = AutoVideoSink::new().with_sync(true);
/// let window = sink.handle();                    // BEFORE start
/// let handle = executor.start(&mut pipeline)?;
/// while let Some(event) = window.try_event() {
///     match event {
///         VideoWindowEvent::KeyPressed(VideoKey::Space) => handle.pause(),
///         VideoWindowEvent::KeyPressed(VideoKey::Escape) => handle.stop(),
///         _ => {}
///     }
/// }
/// ```
#[derive(Clone)]
pub struct AutoVideoSinkHandle {
    events: crossbeam_channel::Receiver<VideoWindowEvent>,
    fullscreen: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    proxy: SharedProxy,
}

impl AutoVideoSinkHandle {
    /// Next pending window event, if any. Non-blocking.
    pub fn try_event(&self) -> Option<VideoWindowEvent> {
        self.events.try_recv().ok()
    }

    /// Wait for the next window event, giving up after `timeout`.
    pub fn event_timeout(&self, timeout: Duration) -> Option<VideoWindowEvent> {
        self.events.recv_timeout(timeout).ok()
    }

    /// Ask the window to enter or leave borderless fullscreen.
    ///
    /// Wakes the event loop to apply it; safe to call from any thread, before
    /// or after the window exists.
    pub fn set_fullscreen(&self, fullscreen: bool) {
        self.fullscreen.store(fullscreen, Ordering::SeqCst);
        wake_display(&self.proxy);
    }

    /// Whether fullscreen is currently requested.
    pub fn is_fullscreen(&self) -> bool {
        self.fullscreen.load(Ordering::SeqCst)
    }

    /// Whether the display window is still open (mirrors
    /// [`AutoVideoSink::is_open`]).
    pub fn is_open(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }
}

impl AutoVideoSink {
    /// Create a new auto video sink with default settings.
    pub fn new() -> Self {
        Self {
            sender: None,
            display_thread: None,
            running: Arc::new(AtomicBool::new(false)),
            title: "Parallax Video".to_string(),
            width: Arc::new(AtomicU32::new(0)),
            height: Arc::new(AtomicU32::new(0)),
            name: "autovideosink".to_string(),
            sync: false,
            max_lateness: DEFAULT_MAX_LATENESS,
            anchor: crate::pipeline::AnchorGate::default(),
            stall_limit: crate::pipeline::STALL_LIMIT,
            generation_opened: false,
            last_pts: ClockTime::NONE,
            display_generation: Arc::new(AtomicU64::new(0)),
            segment: None,
            dropped: 0,
            qos: QosAggregator::default(),
            events: None,
            fullscreen: Arc::new(AtomicBool::new(false)),
            proxy: Arc::new(Mutex::new(None)),
            gpu: true,
        }
    }

    /// Prefer (or refuse) the GPU presentation backend (#190). Default on;
    /// without the `display-gpu` feature or a usable adapter it is inert.
    /// `with_gpu(false)` also drops I420/NV12 from the caps, restoring the
    /// pure CPU pipeline shape (converter upstream, softbuffer blit).
    pub fn with_gpu(mut self, gpu: bool) -> Self {
        self.gpu = gpu;
        self
    }

    /// Whether this sink will negotiate YUV input for GPU presentation.
    fn gpu_effective(&self) -> bool {
        self.gpu && present::gpu_present_available()
    }

    /// Runtime handle to the window: events + fullscreen control.
    ///
    /// Must be taken **before** `Executor::start` moves the sink into its
    /// task (the [`Controllable`](crate::control::Controllable) convention).
    /// A sink whose handle was never taken sends no events at all.
    pub fn handle(&mut self) -> AutoVideoSinkHandle {
        // Bounded and lossy on the sender side: the window must never block
        // on a consumer that stopped reading.
        let (_, rx) = self
            .events
            .get_or_insert_with(|| crossbeam_channel::bounded::<VideoWindowEvent>(64));
        AutoVideoSinkHandle {
            events: rx.clone(),
            fullscreen: self.fullscreen.clone(),
            running: self.running.clone(),
            proxy: self.proxy.clone(),
        }
    }

    /// Set the window title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
    }

    /// Present frames at their PTS rather than as fast as they arrive.
    ///
    /// **Off by default.** A capture preview wants every frame the camera
    /// produces, shown the moment it arrives; a media player wants the stream
    /// to play at its own speed. Turning this on makes the sink wait until each
    /// frame's presentation instant on the pipeline clock, which also
    /// back-pressures everything upstream to real time.
    ///
    /// Frames without a PTS, and pipelines with no clock, are unaffected — they
    /// keep the blit-as-fast-as-they-arrive behaviour.
    pub fn with_sync(mut self, sync: bool) -> Self {
        self.sync = sync;
        self
    }

    /// How late a frame may be and still be shown (default 40 ms).
    ///
    /// How long to keep waiting on a pipeline clock that is not moving
    /// before presenting anyway (#265).
    ///
    /// The clock a paced sink waits on is usually a device's, and this sink
    /// can be the reason it stopped: holding a frame fills its link, which
    /// backs the producer up, which starves the branch driving the clock.
    /// Giving up breaks that circle — pacing degrades to blit-on-arrival
    /// until the clock moves again, which it then does. A deliberate
    /// [`PipelineHandle::pause`](crate::pipeline::PipelineHandle::pause) is
    /// excluded, so pausing still stalls presentation indefinitely.
    pub fn with_stall_limit(mut self, limit: Duration) -> Self {
        self.stall_limit = limit;
        self
    }

    /// How long to wait for the sync master to define running-time zero
    /// before doing it ourselves (#265).
    ///
    /// The master is the sink that provides the pipeline clock — an
    /// `AlsaSink`, whose device's progress *is* the timeline. This sink
    /// normally arrives second (a video branch restarting after a seek has
    /// to rebuild its decoder and decode a keyframe), so the wait usually
    /// costs nothing. It exists so a stalled or absent audio branch cannot
    /// freeze the picture; a pipeline with no master never waits at all.
    pub fn with_anchor_grace(mut self, grace: Duration) -> Self {
        self.anchor.set_grace(grace);
        self
    }

    /// Only consulted when [`with_sync`](Self::with_sync) is on. A later frame
    /// is dropped and counted rather than displayed out of time.
    pub fn with_max_lateness(mut self, max_lateness: Duration) -> Self {
        self.max_lateness = max_lateness;
        self
    }

    /// Frames dropped so far — late, or displaced by a full display channel.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Record a dropped frame, once, in one place.
    fn drop_frame(&mut self, reason: &str) {
        self.dropped += 1;
        crate::observability::record_buffer_dropped("pipeline", &self.name);
        tracing::debug!("autovideosink: dropping frame ({reason})");
    }

    /// A flush opened sync generation `epoch` (#265).
    ///
    /// Arms the anchor handshake for it and marks the next frame as the one
    /// that must be shown however late it is.
    fn open_generation(&mut self, epoch: u64) {
        self.anchor.open(epoch);
        self.generation_opened = true;
    }

    /// Retire the frames already handed to the display thread.
    ///
    /// They belong to the stream position a flushing seek just abandoned.
    /// The receiver lives on the display thread and `std::sync::mpsc` has no
    /// second consumer, so instead of draining the channel this bumps a
    /// shared counter and the display loop skips anything stamped below it —
    /// which also catches the frame it may have dequeued a moment ago.
    fn retire_queued_frames(&mut self) {
        self.display_generation.fetch_add(1, Ordering::AcqRel);
    }

    /// Make sure this generation's running-time origin is fixed before
    /// pacing against it (#265).
    ///
    /// The master — the sink whose element provides the pipeline clock — is
    /// the one whose answer counts, because its device's progress *is* the
    /// timeline. This sink is normally not it, so it waits: a video branch
    /// restarting after a seek is slower than an audio one (the decoder is
    /// rebuilt, a keyframe must be decoded), so the master has usually
    /// anchored before the first frame even arrives and the wait costs
    /// nothing. The rules live in [`AnchorGate`](crate::pipeline::AnchorGate)
    /// so every pacing sink follows the same ones.
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

    /// Whether this PTS is out-of-segment pre-roll a sync sink should drop
    /// (#165 ACCURATE clipping): below the current forward Time segment's
    /// start. `sync=false` previews never clip.
    fn clips_out_of_segment(&self, pts: crate::clock::ClockTime) -> bool {
        self.sync
            && pts != crate::clock::ClockTime::NONE
            && self
                .segment
                .as_ref()
                .is_some_and(|seg| seg.rate > 0.0 && (pts.nanos() as i64) < seg.start)
    }

    /// Set expected dimensions (optional, auto-detected from first frame).
    pub fn with_size(self, width: u32, height: u32) -> Self {
        self.width.store(width, Ordering::SeqCst);
        self.height.store(height, Ordering::SeqCst);
        self
    }

    /// Check if the display window is still open.
    pub fn is_open(&self) -> bool {
        self.running.load(Ordering::SeqCst)
    }

    /// Start the display thread.
    fn start_display(&mut self, initial_width: u32, initial_height: u32) -> Result<()> {
        if self.display_thread.is_some() {
            return Ok(()); // Already started
        }

        // Store dimensions
        self.width.store(initial_width, Ordering::SeqCst);
        self.height.store(initial_height, Ordering::SeqCst);

        // Shallow on purpose: each queued frame pins an upstream arena slot
        // (see DisplayFrame). Declared to the executor via
        // `retained_buffers()` (#189) so the producer's arena is sized for
        // the queue plus the presented frame.
        let (sender, receiver) = mpsc::sync_channel::<DisplayFrame>(DISPLAY_QUEUE_DEPTH);

        let running = Arc::clone(&self.running);
        let title = self.title.clone();
        let events_tx = self.events.as_ref().map(|(tx, _)| tx.clone());
        let fullscreen = self.fullscreen.clone();
        let proxy = self.proxy.clone();

        running.store(true, Ordering::SeqCst);

        let prefer_gpu = self.gpu;
        let display_generation = Arc::clone(&self.display_generation);
        let handle = thread::spawn(move || {
            if let Err(e) = run_display_loop(
                receiver,
                running,
                &title,
                initial_width,
                initial_height,
                events_tx,
                fullscreen,
                proxy.clone(),
                prefer_gpu,
                display_generation,
            ) {
                eprintln!("Display error: {}", e);
            }
            // The loop is gone; wakes from here on have nowhere to go.
            *proxy.lock().unwrap_or_else(|e| e.into_inner()) = None;
        });

        self.sender = Some(sender);
        self.display_thread = Some(handle);

        Ok(())
    }

    /// Stop the display thread.
    fn stop_display(&mut self) {
        self.running.store(false, Ordering::SeqCst);
        // The loop idles in `Wait`; without a wake it would only notice the
        // flag on the next window event.
        wake_display(&self.proxy);

        // Drop sender to unblock receiver
        self.sender.take();

        // Wait for the display thread — but bounded (#191). This runs from
        // the sink element's Drop, i.e. inside a tokio task; the sequence
        // above makes the join normally instant, but a display thread
        // wedged in driver teardown must degrade to a leaked thread, not an
        // unkillable `PipelineHandle::wait()`. (winit's one-event-loop-per-
        // process rule means no restart was ever possible anyway.)
        if let Some(handle) = self.display_thread.take() {
            let deadline = std::time::Instant::now() + Duration::from_secs(2);
            while !handle.is_finished() && std::time::Instant::now() < deadline {
                thread::sleep(Duration::from_millis(10));
            }
            if handle.is_finished() {
                let _ = handle.join();
            } else {
                tracing::warn!(
                    "autovideosink: display thread did not exit within 2s — detaching it"
                );
            }
        }
    }
}

impl Default for AutoVideoSink {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for AutoVideoSink {
    fn drop(&mut self) {
        self.stop_display();
    }
}

impl AsyncSink for AutoVideoSink {
    /// QoS origination (#184): the executor polls this after each consume;
    /// the aggregator stages at most ~one event per second of running time.
    fn take_upstream_event(&mut self) -> Option<crate::event::Event> {
        self.qos.take().map(crate::event::Event::Qos)
    }

    /// The display thread holds up to `DISPLAY_QUEUE_DEPTH` queued frames
    /// plus the currently presented one, each a zero-copy `Buffer` clone
    /// pinning a producer arena slot (#189).
    fn retained_buffers(&self) -> usize {
        // Plus the submissions the GPU may still be reading from when the
        // frames are imported rather than uploaded: an imported frame's
        // memory belongs to the producer until its draw retires, so the
        // backend holds it that much longer (see `GPU_IN_FLIGHT`).
        DISPLAY_QUEUE_DEPTH
            + 1
            + if present::gpu_dmabuf_import_available() {
                2
            } else {
                0
            }
    }

    /// Instant rate change (#165): `seek.rate` replaces the current
    /// segment's rate in this sink's own mapping, effective at the next
    /// frame. `base` is recomputed so the *current* position keeps the
    /// running time it already had — the mapping bends, it does not jump.
    /// (The old code cleared the pacer's private anchor instead, which hid
    /// the discontinuity; with absolute pacing there is nowhere to hide it.)
    /// Sign flips are rejected (reverse needs a real seek: the data order
    /// itself must change); a later Segment (any real seek) re-establishes
    /// its own rate. Everything else is not ours.
    fn handle_upstream_event(&mut self, event: &crate::event::Event) -> crate::event::EventResult {
        use crate::event::{Event, EventResult, SeekFlags};
        if let Event::Seek(seek) = event
            && seek.flags.contains(SeekFlags::INSTANT_RATE_CHANGE)
        {
            let Some(seg) = self.segment.as_mut() else {
                // No mapping yet (nothing played): nothing to change.
                return EventResult::NotHandled;
            };
            if seek.rate == 0.0 || seek.rate.signum() != seg.rate.signum() {
                tracing::warn!(
                    "autovideosink: instant rate {} rejected (direction change from {})",
                    seek.rate,
                    seg.rate
                );
                return EventResult::NotHandled;
            }
            // Pivot on the last frame presented, or the segment start when
            // nothing has been shown yet: `to_running_time(pivot)` must read
            // the same before and after.
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

    /// Declared latency (#184): pacing may hold a frame up to its lateness
    /// budget past the ideal presentation time.
    fn latency(&self) -> Option<crate::pipeline::seek::LatencyRange> {
        self.sync.then(|| {
            crate::pipeline::seek::LatencyRange::up_to(ClockTime::from_nanos(
                self.max_lateness.as_nanos() as u64,
            ))
        })
    }

    fn handle_downstream_event(
        &mut self,
        event: crate::event::Event,
    ) -> Option<crate::event::Event> {
        match &event {
            // A flushing seek abandoned the stream position the queued
            // frames belong to. Drop them: the display channel holds up to
            // DISPLAY_QUEUE_DEPTH + 1 of them, and without this they are
            // blitted *after* the seek — the pre-seek picture flashing back
            // over the new one.
            crate::event::Event::FlushStart => {
                self.retire_queued_frames();
            }
            // The flush names the sync generation to anchor for (#265), so
            // both branches of an A/V graph derive it from the same event.
            crate::event::Event::FlushStop(stop) => {
                self.open_generation(stop.epoch);
            }
            // A new segment re-defines the PTS → running-time mapping
            // (#165). It no longer re-anchors anything: the origin is
            // shared, and the segment's own `start`/`base`/`rate` are the
            // whole mapping.
            crate::event::Event::Segment(seg)
                if seg.format == crate::event::SegmentFormat::Time =>
            {
                self.segment = Some(seg.clone());
            }
            _ => {}
        }
        Some(event)
    }

    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let data = ctx.input();

        tracing::debug!("AutoVideoSink: received buffer with {} bytes", data.len());

        // Format, geometry and plane layout, read and validated once by
        // `crate::video` — the same read a host renderer does, so an Iced or
        // egui presenter cannot disagree with this sink about what a buffer
        // is (#263).
        let meta = ctx.metadata();
        let view = crate::video::FrameView::from_metadata(meta, data.len())?;
        let (format, width, height, strided) = (view.format, view.width, view.height, view.layout);

        tracing::debug!("AutoVideoSink: detected dimensions {}x{}", width, height);

        // ACCURATE-seek clipping (#165): a PTS below the segment start is
        // out-of-segment pre-roll — data legitimately starts at the snapped
        // keyframe, the segment at the requested time. A decoder normally
        // drops these before they arrive; for pipelines without one the
        // sync sink is the last line, and dropping beats the old
        // blit-at-full-speed. `sync=false` previews keep every frame.
        if self.clips_out_of_segment(meta.pts) {
            self.drop_frame("out-of-segment");
            return Ok(());
        }

        self.last_pts = meta.pts;

        // Presentation pacing (#66). Opt-in: a clock-less pipeline, an
        // un-timestamped stream, or plain `sync=false` all keep the historical
        // blit-on-arrival behaviour, so capture previews do not regress.
        if self.sync
            && ctx.clock().is_some()
            && let Some(pts) = meta.pts.to_option()
            && ctx.running_time().is_some()
            // Map through the current segment (#165): (pts − start)/|rate| +
            // base. A PTS past the segment stop has no valid mapping
            // (below-start ones were dropped above) — present it immediately
            // rather than pace it.
            && let Some(target) = self
                .segment
                .as_ref()
                .map_or(meta.pts, |seg| seg.to_running_time(pts))
                .to_option()
        {
            // The flush epoch this frame is being handled under. A seek
            // raises it the moment it is dispatched, which is the only signal
            // that reaches a sink already inside a wait — and both waits
            // below are waits (#265).
            let entered_at = ctx.sync().map(|s| s.flush_epoch()).unwrap_or(0);

            // Define running-time zero for this generation, or wait for the
            // sink that owns the clock to do it. Absolute pacing is only
            // meaningful once the origin is fixed.
            if self.settle_anchor(ctx, target, entered_at).await == Anchor::Superseded {
                self.drop_frame("superseded by a seek");
                return Ok(());
            }

            let now = ctx.running_time();
            let mut now_ns = now.nanos();
            let mut pace = schedule(target, now, self.max_lateness);
            // Waiting here is the point: it is what back-pressures the
            // decoder and the source down to real time. Since #172 the wait
            // is an await, so it pends this element's future instead of
            // parking a tokio worker.
            //
            // The wait polls the clock in short slices instead of sleeping the
            // full delay: against a clock frozen by `PipelineHandle::pause`
            // a one-shot sleep would keep presenting at wall-clock rate,
            // while re-reading running time stalls right here — and resumes
            // gap-free, because the clock does. Re-reading is also what makes
            // a mid-wait re-anchor by the master self-correcting. `MAX_WAIT`
            // still applies to each *computed* deficit (the PTS-jump escape
            // hatch), not to the frozen-clock stall, where the deficit never
            // shrinks but stays under the cap.
            let mut stall = crate::pipeline::StallWatch::new(self.stall_limit);
            let mut abandoned = false;
            while let Pace::Wait(delay) = pace {
                tokio::time::sleep(delay.min(Duration::from_millis(10))).await;
                let Some(now) = ctx.running_time().to_option() else {
                    break;
                };
                if let Some(sync) = ctx.sync() {
                    if sync.flush_epoch() > entered_at {
                        // A flushing seek abandoned the position this frame
                        // belongs to. Waiting out its presentation time would
                        // hold the seek up by however long that is — and the
                        // `FlushStart` that would have freed us is queued
                        // behind this very buffer.
                        abandoned = true;
                        break;
                    }
                    // A clock provided by a device stops when that device
                    // does — and this sink can be why, by holding its buffer
                    // until its own link backs the producer up and starves
                    // the branch that drives the clock. Give up on a clock
                    // going nowhere and present: that drains the link and
                    // restarts everything. A deliberate pause is excluded.
                    if stall.stalled(now, sync.is_paused()) {
                        tracing::warn!(
                            "autovideosink: the pipeline clock has not moved for {:?}; \
                             presenting instead of waiting on it",
                            self.stall_limit
                        );
                        break;
                    }
                }
                now_ns = now.nanos();
                pace = schedule(target, now, self.max_lateness);
            }
            if abandoned {
                self.drop_frame("superseded by a seek");
                return Ok(());
            }
            // The first frame of a generation goes up however late it is:
            // after a seek it is the frame the user is waiting to see, and
            // dropping it would leave the pre-seek picture on screen until
            // the branch caught up.
            if self.generation_opened {
                self.generation_opened = false;
            } else if let Pace::DropLate { late_ns } = pace {
                self.qos.dropped_late(now_ns, late_ns);
                self.drop_frame("late");
                return Ok(());
            }
            self.qos.presented(now_ns);
        }

        // Start display thread on first frame
        if self.sender.is_none() {
            tracing::info!("AutoVideoSink: starting display thread");
            self.start_display(width, height)?;
            tracing::info!("AutoVideoSink: display thread started");
        }

        let sender = self
            .sender
            .as_ref()
            .ok_or_else(|| Error::Element("Display not started".into()))?;

        // The user closed the window: ask for a cooperative pipeline
        // shutdown (#191). Not an error — the run ends as EndReason::Eos,
        // and nothing has to string-match a message to tell a close from a
        // failure any more.
        if !self.running.load(Ordering::SeqCst) {
            return Err(Error::Shutdown);
        }

        let frame = DisplayFrame {
            generation: self.display_generation.load(Ordering::Acquire),
            // Refcount bump, not a copy: the display thread maps the same
            // arena slot — or pins the same decoder picture (#194) — the
            // producer wrote (#141).
            data: ctx.buffer().clone(),
            width,
            height,
            format,
            layout: strided,
            // What the stream said, if it said anything (#265). The
            // presenter falls back to the height heuristic; a file that
            // signalled its matrix outranks that guess.
            color: ctx.buffer().metadata().color_info(),
        };

        // A full channel means the display cannot keep up. Shed this frame and
        // count it — the previous code logged "drain one old frame", drained
        // nothing, and retried the same send with the result discarded.
        match sender.try_send(frame) {
            Ok(()) => {
                // Doorbell only — the sync_channel above is the data path. A
                // full channel needs no wake: its frames already sent one each.
                wake_display(&self.proxy);
                Ok(())
            }
            Err(mpsc::TrySendError::Full(_)) => {
                self.drop_frame("display too slow");
                Ok(())
            }
            // Same close, seen from the channel side (the display thread
            // already dropped the receiver) — same clean shutdown (#191).
            // This arm used to return a *different* string ("Display
            // closed") than the arm above, so apps matching one missed the
            // other and reported a failed playback for a normal close.
            Err(mpsc::TrySendError::Disconnected(_)) => Err(Error::Shutdown),
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn input_caps(&self) -> Caps {
        // AutoVideoSink expects RGBA data (any resolution)
        Caps::video_raw_any_resolution(PixelFormat::Rgba)
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, MemoryLayout,
            VideoFormatCaps,
        };
        // With a usable GPU backend (#190) the sink takes the decoder's
        // I420/NV12 directly — colorspace conversion and scaling move into
        // the fragment shader, and no `videoconvert` is needed at all.
        // Otherwise: RGBA first, because the converter matrix reaches Rgba
        // from every source format, so auto-inserted converters keep
        // working everywhere; BGRA is the CPU blit's fast path (its LE u32
        // is softbuffer's 0RGB layout).
        // With the GPU path the sink also opts into External memory
        // (#194): wgpu uploads strided decoder-owned frames natively
        // (bytes_per_row), so a dav1d picture reaches the texture with
        // zero CPU copies; the CpuBackend runtime degradation repacks.
        // Without a GPU the caps stay cpu_only — the packed contract.
        let (formats, memory) = if self.gpu_effective() {
            (
                vec![
                    PixelFormat::I420,
                    PixelFormat::Nv12,
                    PixelFormat::Bgra,
                    PixelFormat::Rgba,
                ],
                // Importing a dma-buf costs nothing at all where it is
                // available, so it goes first — but only where it really is:
                // advertising it on a GL adapter would strand a producer
                // that owns GPU memory with frames the sink cannot read.
                if present::gpu_dmabuf_import_available() {
                    MemoryCaps::gpu_import_or_cpu()
                } else {
                    MemoryCaps::external_or_cpu()
                },
            )
        } else {
            (
                vec![PixelFormat::Rgba, PixelFormat::Bgra],
                MemoryCaps::cpu_only(),
            )
        };
        ElementMediaCaps::new([FormatMemoryCap::new(
            FormatCaps::VideoRaw(VideoFormatCaps {
                width: CapsValue::Any,
                height: CapsValue::Any,
                pixel_format: CapsValue::List(formats),
                framerate: CapsValue::Any,
                layout: MemoryLayout::NONE,
            }),
            memory,
        )])
    }
}

/// Run the winit display loop in the display thread.
///
/// The loop is event-driven: it idles in `ControlFlow::Wait` and is woken by
/// window events or by a [`WakeUp`] rung through `proxy_slot` (new frame,
/// fullscreen request, shutdown). It never polls.
#[allow(clippy::too_many_arguments)]
fn run_display_loop(
    receiver: mpsc::Receiver<DisplayFrame>,
    running: Arc<AtomicBool>,
    title: &str,
    initial_width: u32,
    initial_height: u32,
    events_tx: Option<crossbeam_channel::Sender<VideoWindowEvent>>,
    fullscreen: Arc<AtomicBool>,
    proxy_slot: SharedProxy,
    prefer_gpu: bool,
    display_generation: Arc<AtomicU64>,
) -> Result<()> {
    use super::present::RenderBackend;
    use winit::application::ApplicationHandler;
    use winit::dpi::PhysicalSize;
    use winit::event::{ElementState, MouseButton, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::keyboard::{Key, NamedKey};
    use winit::platform::x11::EventLoopBuilderExtX11;
    use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

    struct VideoApp {
        // Declared before `window` so drop order tears the backend's
        // surface down before the window it was created from.
        backend: Option<Box<dyn RenderBackend>>,
        window: Option<Arc<Window>>,
        receiver: mpsc::Receiver<DisplayFrame>,
        running: Arc<AtomicBool>,
        current_frame: Option<DisplayFrame>,
        title: String,
        initial_width: u32,
        initial_height: u32,
        events_tx: Option<crossbeam_channel::Sender<VideoWindowEvent>>,
        prefer_gpu: bool,
        /// Desired state, set by the handle; compared against `is_fullscreen`.
        fullscreen: Arc<AtomicBool>,
        is_fullscreen: bool,
        /// Last cursor position, for MousePressed coordinates.
        cursor: (f64, f64),
        /// Frames stamped below this were retired by a flushing seek.
        display_generation: Arc<AtomicU64>,
    }

    impl VideoApp {
        /// Best-effort event delivery: a full channel drops the event (the
        /// consumer stopped reading) and a missing sender means no handle was
        /// ever taken.
        fn emit(&self, event: VideoWindowEvent) {
            if let Some(tx) = &self.events_tx {
                let _ = tx.try_send(event);
            }
        }

        /// Present every frame queued since the last wake.
        ///
        /// Rendering here instead of via `RedrawRequested` bypasses compositor
        /// vsync throttling — same rationale as the old poll loop, minus the
        /// polling. Runs on every wake, so a doorbell missed while the loop
        /// was still being built is made up on the next event.
        fn drain_frames(&mut self) {
            while let Ok(frame) = self.receiver.try_recv() {
                // A flushing seek retired everything queued before it (#265);
                // blitting those now would flash the pre-seek picture back
                // over the new position.
                if frame.generation < self.display_generation.load(Ordering::Acquire) {
                    continue;
                }
                self.current_frame = Some(frame);
                self.render();
            }
        }
    }

    impl ApplicationHandler<WakeUp> for VideoApp {
        fn resumed(&mut self, event_loop: &ActiveEventLoop) {
            if self.window.is_some() {
                return; // Already have a window
            }

            // Physical size, deliberately: a logical size is multiplied by
            // the display scale factor, so on a scaled desktop the surface
            // never matches the video and every frame takes the scaling blit.
            // At physical size the default window is pixel-perfect AND hits
            // the 1:1 row-copy fast path (#161).
            let attrs = WindowAttributes::default()
                .with_title(&self.title)
                .with_inner_size(PhysicalSize::new(self.initial_width, self.initial_height));

            match event_loop.create_window(attrs) {
                Ok(window) => {
                    let window = Arc::new(window);
                    // GPU first (feature + preference + probe), softbuffer
                    // as the fallback; see `present::create_backend`.
                    match super::present::create_backend(window.clone(), self.prefer_gpu) {
                        Ok(backend) => {
                            tracing::debug!("autovideosink: {} backend ready", backend.name());
                            self.backend = Some(backend);
                            self.window = Some(window);
                        }
                        Err(e) => {
                            eprintln!("Failed to create render backend: {}", e);
                            self.running.store(false, Ordering::SeqCst);
                            event_loop.exit();
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Failed to create window: {}", e);
                    self.running.store(false, Ordering::SeqCst);
                    event_loop.exit();
                }
            }
        }

        fn window_event(
            &mut self,
            event_loop: &ActiveEventLoop,
            _window_id: WindowId,
            event: WindowEvent,
        ) {
            match event {
                WindowEvent::CloseRequested => {
                    self.emit(VideoWindowEvent::CloseRequested);
                    self.running.store(false, Ordering::SeqCst);
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    self.render();
                }
                WindowEvent::Resized(size) => {
                    self.emit(VideoWindowEvent::Resized {
                        width: size.width,
                        height: size.height,
                    });
                    if let Some(backend) = &mut self.backend {
                        backend.resized(size.width, size.height);
                    }
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.state == ElementState::Pressed && !event.repeat {
                        let key = match &event.logical_key {
                            Key::Named(NamedKey::Space) => VideoKey::Space,
                            Key::Named(NamedKey::Escape) => VideoKey::Escape,
                            Key::Named(NamedKey::Enter) => VideoKey::Enter,
                            Key::Named(NamedKey::ArrowLeft) => VideoKey::ArrowLeft,
                            Key::Named(NamedKey::ArrowRight) => VideoKey::ArrowRight,
                            Key::Named(NamedKey::ArrowUp) => VideoKey::ArrowUp,
                            Key::Named(NamedKey::ArrowDown) => VideoKey::ArrowDown,
                            Key::Character(text) => VideoKey::Character(text.to_string()),
                            other => VideoKey::Other(format!("{other:?}")),
                        };
                        self.emit(VideoWindowEvent::KeyPressed(key));
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    self.cursor = (position.x, position.y);
                }
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => {
                    self.emit(VideoWindowEvent::MousePressed {
                        x: self.cursor.0,
                        y: self.cursor.1,
                    });
                }
                _ => {}
            }
        }

        fn user_event(&mut self, _event_loop: &ActiveEventLoop, _event: WakeUp) {
            // The doorbell: a frame was queued, fullscreen changed, or the
            // sink is shutting down. State checks happen in `about_to_wait`,
            // which winit runs right after this.
            self.drain_frames();
        }

        fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
            // Tear the render backend down HERE, while the event loop (and
            // with it the display connection) is still alive (#191). The
            // `app` local outlives `run_app`, so without this the wgpu
            // Surface/Device would drop *after* winit destroyed the
            // EventLoop — the classic swapchain-against-a-dead-connection
            // driver hang. Also release every pinned arena slot promptly:
            // a queued frame held by a dead thread pins a producer slot
            // forever.
            while self.receiver.try_recv().is_ok() {}
            self.current_frame = None;
            self.backend = None;
            self.window = None;
        }

        fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
            // Check if we should exit
            if !self.running.load(Ordering::SeqCst) {
                event_loop.exit();
                return;
            }

            // Apply a fullscreen request from the handle.
            let want_fullscreen = self.fullscreen.load(Ordering::SeqCst);
            if want_fullscreen != self.is_fullscreen
                && let Some(window) = &self.window
            {
                window.set_fullscreen(if want_fullscreen {
                    Some(Fullscreen::Borderless(None))
                } else {
                    None
                });
                self.is_fullscreen = want_fullscreen;
            }

            // Catch frames whose doorbell rang before the loop was built (the
            // first frames of a stream race `start_display`).
            self.drain_frames();

            // Sleep until the next window event or WakeUp — the frame rate is
            // set by the producer ringing the doorbell, not by polling (#155:
            // ControlFlow::Poll here spun a full core on try_recv).
            event_loop.set_control_flow(ControlFlow::Wait);
        }
    }

    impl VideoApp {
        fn render(&mut self) {
            let Some(window) = &self.window else {
                return;
            };
            let Some(backend) = &mut self.backend else {
                return;
            };
            let Some(frame) = &self.current_frame else {
                return;
            };

            let size = window.inner_size();
            if let Err(e) = backend.render(frame, (size.width, size.height)) {
                // One frame failing to present is a glitch, not a teardown;
                // persistent failure shows up as a black window plus logs.
                tracing::warn!("autovideosink: {} render failed: {e}", backend.name());
            }
        }
    }

    // Create event loop with any_thread enabled (Linux only)
    let event_loop = EventLoop::<WakeUp>::with_user_event()
        .with_any_thread(true)
        .build()
        .map_err(|e| Error::Element(format!("Failed to create event loop: {}", e)))?;

    // Publish the doorbell. Wakes rung before this point are covered by the
    // drain in the first `about_to_wait`.
    *proxy_slot.lock().unwrap_or_else(|e| e.into_inner()) = Some(event_loop.create_proxy());

    let mut app = VideoApp {
        backend: None,
        window: None,
        receiver,
        running,
        current_frame: None,
        title: title.to_string(),
        initial_width,
        initial_height,
        events_tx,
        prefer_gpu,
        fullscreen,
        is_fullscreen: false,
        cursor: (0.0, 0.0),
        display_generation,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| Error::Element(format!("Event loop error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clock stopped at one reading, so an anchor's arithmetic is exact.
    struct FixedClock(ClockTime);

    impl crate::clock::Clock for FixedClock {
        fn now(&self) -> ClockTime {
            self.0
        }
        fn name(&self) -> &str {
            "fixed"
        }
    }

    /// The smallest real buffer a `ConsumeContext` can be built over.
    fn one_byte_buffer() -> crate::buffer::Buffer {
        let arena = crate::memory::SharedArena::new(64, 2).unwrap();
        let slot = arena.acquire().unwrap();
        crate::buffer::Buffer::new(
            crate::buffer::MemoryHandle::with_len(slot, 1),
            crate::metadata::Metadata::new(),
        )
    }

    #[test]
    fn handle_events_flow_and_fullscreen_toggles() {
        let mut sink = AutoVideoSink::new();
        let handle = sink.handle();
        let handle2 = handle.clone();

        assert_eq!(handle.try_event(), None);
        // The element side owns the sender; emit as the event loop would.
        let (tx, _) = sink.events.as_ref().unwrap().clone();
        tx.try_send(VideoWindowEvent::KeyPressed(VideoKey::Space))
            .unwrap();
        assert_eq!(
            handle.try_event(),
            Some(VideoWindowEvent::KeyPressed(VideoKey::Space))
        );

        assert!(!handle2.is_fullscreen());
        handle2.set_fullscreen(true);
        assert!(handle.is_fullscreen());
    }

    #[test]
    fn test_sink_creation() {
        let sink = AutoVideoSink::new();
        assert_eq!(sink.name(), "autovideosink");
        assert!(!sink.is_open()); // Not started yet
    }

    #[test]
    fn test_sink_with_title() {
        let sink = AutoVideoSink::new().with_title("My Video");
        assert_eq!(sink.title, "My Video");
    }

    // ------------------------------------------------------------------
    // PTS-paced presentation (#66)
    // ------------------------------------------------------------------

    const FRAME_25FPS: u64 = 40_000_000;

    fn ns(n: u64) -> ClockTime {
        ClockTime::from_nanos(n)
    }

    #[test]
    fn pacing_is_off_by_default() {
        // Every pipeline has a started clock, so the property — not the
        // presence of a clock — is what turns pacing on. Examples 23/24/58
        // depend on this.
        let sink = AutoVideoSink::new();
        assert!(!sink.sync);
        assert_eq!(sink.max_lateness, DEFAULT_MAX_LATENESS);
    }

    #[test]
    fn a_time_segment_is_stored_as_the_whole_mapping() {
        use crate::element::AsyncSink;
        use crate::event::{Event, SegmentEvent};

        let mut sink = AutoVideoSink::new();

        // A Time segment installs the mapping, and that is all it does: the
        // running-time origin is shared and re-anchored by the flush, not by
        // the segment (#265).
        let seg = SegmentEvent::new_time(ClockTime::from_secs(10), None).with_rate(2.0);
        let fwd = sink.handle_downstream_event(Event::Segment(seg));
        assert!(matches!(fwd, Some(Event::Segment(_))), "segment forwarded");
        let stored = sink.segment.as_ref().expect("segment stored");
        assert_eq!(stored.start, 10_000_000_000);
        assert_eq!(stored.rate, 2.0);

        // The mapping halves inter-frame spacing at rate 2.0.
        let mapped_a = stored.to_running_time(ClockTime::from_secs(10));
        let mapped_b = stored.to_running_time(ns(10_000_000_000 + 2 * FRAME_25FPS));
        assert_eq!(mapped_a, ns(0));
        assert_eq!(mapped_b, ns(FRAME_25FPS), "rate 2.0 halves the spacing");

        // Out-of-segment PTS has no mapping — consume presents it directly.
        assert_eq!(
            stored.to_running_time(ClockTime::from_secs(5)),
            ClockTime::NONE
        );

        // A Bytes segment is not a video timeline; ignored.
        let bytes_seg = SegmentEvent::new_bytes(0, None);
        let _ = sink.handle_downstream_event(Event::Segment(bytes_seg));
        assert_eq!(
            sink.segment.as_ref().unwrap().format,
            crate::event::SegmentFormat::Time
        );
    }

    #[test]
    fn a_25fps_stream_waits_one_frame_interval() {
        // Decoder runs flat out: the clock has barely moved, so each frame
        // is held back to its own presentation instant. Absolute now — the
        // origin is the shared base, not this sink's first arrival.
        for frame in 1..10u64 {
            assert_eq!(
                schedule(ns(frame * FRAME_25FPS), ns(0), DEFAULT_MAX_LATENESS),
                Pace::Wait(Duration::from_nanos(frame * FRAME_25FPS)),
                "frame {frame}"
            );
        }
    }

    #[test]
    fn qos_aggregator_stages_only_after_a_window_with_drops() {
        let mut agg = QosAggregator::default();
        // A full window of clean presentation stages nothing.
        agg.presented(0);
        agg.presented(500_000_000);
        agg.presented(1_100_000_000); // window rolls, dropped == 0
        assert!(agg.take().is_none());

        // Next window: 3 presented, 1 dropped -> proportion (3+1)/3.
        agg.presented(1_200_000_000);
        agg.presented(1_300_000_000);
        agg.dropped_late(1_400_000_000, 70_000_000);
        agg.presented(2_200_000_000); // rolls the window
        let qos = agg.take().expect("staged after a window with drops");
        assert_eq!(qos.qos_type, crate::event::QosType::Underflow);
        assert_eq!(qos.processed, 3);
        assert_eq!(qos.dropped, 1);
        assert_eq!(qos.jitter_ns, 70_000_000);
        assert!((qos.proportion - 4.0 / 3.0).abs() < 1e-9);
        // One event per window: nothing new until the next roll.
        assert!(agg.take().is_none());
    }

    #[test]
    fn qos_aggregator_survives_zero_presented() {
        let mut agg = QosAggregator::default();
        agg.dropped_late(0, 10);
        agg.dropped_late(1_500_000_000, 20); // rolls
        let qos = agg.take().expect("staged");
        assert_eq!(qos.processed, 0);
        assert_eq!(qos.dropped, 2);
        // Denominator clamps at 1 — no division by zero.
        assert!((qos.proportion - 2.0).abs() < 1e-9);
    }

    #[test]
    fn a_frozen_clock_keeps_the_pacer_waiting() {
        // PipelineHandle::pause freezes running time. Re-scheduling against
        // the same `now` must keep answering Wait with a constant deficit —
        // never Present — which is what makes the consume() poll loop stall
        // for as long as the pause lasts.
        let frozen_now = ns(0);
        for _ in 0..100 {
            assert_eq!(
                schedule(ns(FRAME_25FPS), frozen_now, DEFAULT_MAX_LATENESS),
                Pace::Wait(Duration::from_nanos(FRAME_25FPS))
            );
        }
        // Resume: time advances past the target and the frame presents.
        assert_eq!(
            schedule(ns(FRAME_25FPS), ns(FRAME_25FPS), DEFAULT_MAX_LATENESS),
            Pace::Present
        );
    }

    /// #165 ACCURATE clipping at the sink: below-segment-start PTS clip in
    /// sync mode (forward rate only); reverse segments and sync=false never
    /// clip, and an unmapped stream (no segment) never clips.
    #[test]
    fn out_of_segment_preroll_clips_only_in_sync_forward() {
        use crate::event::SegmentEvent;

        let mut sink = AutoVideoSink::new();
        sink.sync = true;
        assert!(!sink.clips_out_of_segment(ns(100)), "no segment, no clip");

        sink.segment = Some(SegmentEvent::new_time(ns(700_000_000), None));
        assert!(sink.clips_out_of_segment(ns(500_000_000)));
        assert!(!sink.clips_out_of_segment(ns(700_000_000)));
        assert!(!sink.clips_out_of_segment(ns(900_000_000)));
        assert!(!sink.clips_out_of_segment(crate::clock::ClockTime::NONE));

        // Reverse: PTS below start is the walk's own termination, not
        // pre-roll.
        sink.segment = Some(SegmentEvent::new_time(ns(0), Some(ns(2_000_000_000))).with_rate(-1.0));
        assert!(!sink.clips_out_of_segment(ns(500_000_000)));

        // sync=false previews keep everything.
        sink.sync = false;
        sink.segment = Some(SegmentEvent::new_time(ns(700_000_000), None));
        assert!(!sink.clips_out_of_segment(ns(500_000_000)));
    }

    /// #265: a flush names the generation to anchor, opens the "show the
    /// next frame however late" window, and retires whatever the display
    /// thread has queued from before the seek.
    #[test]
    fn a_flush_opens_the_generation_it_names() {
        use crate::element::AsyncSink;
        use crate::event::{Event, FlushStopEvent};

        let mut sink = AutoVideoSink::new();
        assert_eq!(
            sink.anchor.generation(),
            crate::pipeline::INITIAL_GENERATION
        );
        assert!(!sink.generation_opened);

        let before = sink.display_generation.load(Ordering::Acquire);
        let _ = sink.handle_downstream_event(Event::FlushStart);
        assert!(
            sink.display_generation.load(Ordering::Acquire) > before,
            "queued pre-seek frames are retired, not blitted after the seek"
        );

        let _ = sink
            .handle_downstream_event(Event::FlushStop(FlushStopEvent::new(true).with_epoch(48)));
        assert_eq!(
            sink.anchor.generation(),
            48,
            "the flush names the generation"
        );
        assert!(
            sink.generation_opened,
            "the first frame after a seek is the one the user is waiting for"
        );
    }

    /// A pipeline with no clock provider has no master, so the sink anchors
    /// its own first frame — which is exactly what the old private per-sink
    /// anchor did, only now the answer is shared with every other sink.
    #[tokio::test]
    async fn a_masterless_pipeline_anchors_on_the_spot() {
        use crate::pipeline::{SyncBase, SyncHandle, SyncRole};
        use std::sync::Arc;

        let base = Arc::new(SyncBase::new(ClockTime::ZERO, None));
        let handle = SyncHandle::new(Arc::clone(&base), SyncRole::Slave);
        let clock: Arc<dyn crate::clock::Clock> = Arc::new(FixedClock(ns(7_000_000_000)));
        let buffer = one_byte_buffer();
        let mut ctx = crate::element::ConsumeContext::new(&buffer);
        ctx.set_clock(&clock, &handle);

        let mut sink = AutoVideoSink::new();
        assert_eq!(
            sink.settle_anchor(&ctx, ns(2_000_000_000), 0).await,
            Anchor::Ready
        );

        assert!(base.is_anchored(0));
        // Running time 2 s maps to clock 7 s, so the origin sits at 5 s.
        assert_eq!(base.base_time(), ns(5_000_000_000));
    }

    /// With a master in the graph the video sink waits for it — bounded, so
    /// a stalled audio branch cannot freeze the picture forever.
    #[tokio::test(start_paused = true)]
    async fn a_slave_gives_up_waiting_after_the_grace() {
        use crate::pipeline::{SyncBase, SyncHandle, SyncRole};
        use std::sync::Arc;

        let base = Arc::new(SyncBase::new(ClockTime::ZERO, Some(9)));
        let handle = SyncHandle::new(Arc::clone(&base), SyncRole::Slave);
        let clock: Arc<dyn crate::clock::Clock> = Arc::new(FixedClock(ns(3_000_000_000)));
        let buffer = one_byte_buffer();
        let mut ctx = crate::element::ConsumeContext::new(&buffer);
        ctx.set_clock(&clock, &handle);

        let mut sink = AutoVideoSink::new().with_anchor_grace(Duration::from_millis(100));
        assert_eq!(
            sink.settle_anchor(&ctx, ClockTime::ZERO, 0).await,
            Anchor::Ready
        );

        assert!(base.is_anchored(0), "the wait is bounded");
        assert_eq!(base.base_time(), ns(3_000_000_000));
    }

    /// A seek dispatched while this sink waits for the master abandons the
    /// frame: it belongs to a position nobody is playing any more, and
    /// waiting the grace out would hold the seek up by whatever is left of
    /// it (#265).
    #[tokio::test]
    async fn a_seek_during_the_anchor_wait_abandons_the_frame() {
        use crate::pipeline::{SyncBase, SyncHandle, SyncRole};
        use std::sync::Arc;

        let base = Arc::new(SyncBase::new(ClockTime::ZERO, Some(9)));
        let handle = SyncHandle::new(Arc::clone(&base), SyncRole::Slave);
        let clock: Arc<dyn crate::clock::Clock> = Arc::new(FixedClock(ns(3_000_000_000)));
        let buffer = one_byte_buffer();
        let mut ctx = crate::element::ConsumeContext::new(&buffer);
        ctx.set_clock(&clock, &handle);

        // The master never anchors; a seek lands mid-wait instead.
        base.note_flush(16);
        let mut sink = AutoVideoSink::new().with_anchor_grace(Duration::from_secs(30));
        assert_eq!(
            sink.settle_anchor(&ctx, ClockTime::ZERO, 0).await,
            Anchor::Superseded
        );
        assert!(!base.is_anchored(0), "and nothing was anchored from it");
    }

    /// #165 instant rate change: same-sign rates replace the segment's rate
    /// while `base` absorbs the change so the current position keeps the
    /// running time it already had; sign flips, rate 0, and a mapping-less
    /// sink decline. Only INSTANT_RATE_CHANGE seeks are ours.
    #[test]
    fn instant_rate_change_updates_the_mapping() {
        use crate::element::AsyncSink;
        use crate::event::{Event, EventResult, SeekEvent, SegmentEvent};

        let mut sink = AutoVideoSink::new();

        // No segment yet: nothing to change.
        let ev = Event::Seek(SeekEvent::new_instant_rate(2.0));
        assert!(matches!(
            sink.handle_upstream_event(&ev),
            EventResult::NotHandled
        ));

        sink.segment = Some(SegmentEvent::new_time(ns(0), None));
        // Four seconds in, mapping to running time 4 s at rate 1.0.
        sink.last_pts = ns(4_000_000_000);
        let before = sink
            .segment
            .as_ref()
            .unwrap()
            .to_running_time(sink.last_pts);
        assert_eq!(before, ns(4_000_000_000));

        assert!(matches!(
            sink.handle_upstream_event(&ev),
            EventResult::Handled { .. }
        ));
        assert_eq!(sink.segment.as_ref().unwrap().rate, 2.0);
        // The mapping bends around the current position instead of jumping:
        // at rate 2.0 the raw mapping would read 2 s, so `base` makes up the
        // difference. With the private arrival anchor gone there is nowhere
        // left to hide a discontinuity here.
        assert_eq!(
            sink.segment
                .as_ref()
                .unwrap()
                .to_running_time(sink.last_pts),
            before,
            "instant rate change is continuous at the current position"
        );
        // And from here on, one second of content takes half a second.
        assert_eq!(
            sink.segment
                .as_ref()
                .unwrap()
                .to_running_time(ns(5_000_000_000)),
            ns(4_500_000_000)
        );

        // Sign flip and rate 0 decline; the mapping is untouched.
        for bad in [-1.0, 0.0] {
            let ev = Event::Seek(SeekEvent::new_instant_rate(bad));
            assert!(matches!(
                sink.handle_upstream_event(&ev),
                EventResult::NotHandled
            ));
            assert_eq!(sink.segment.as_ref().unwrap().rate, 2.0);
        }

        // A regular seek is not ours (the executor routes it upstream).
        let regular = Event::Seek(SeekEvent::new_time(ns(1_000)));
        assert!(matches!(
            sink.handle_upstream_event(&regular),
            EventResult::NotHandled
        ));
    }
}
