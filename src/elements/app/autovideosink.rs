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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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

/// Longest single wait the pacer will ask for.
///
/// A stream whose timestamps jump forward (a bad demuxer, a wrapped RTP clock)
/// must not park the sink for minutes. Past this the frame is shown instead.
const MAX_WAIT: Duration = Duration::from_secs(1);

/// What to do with a frame whose presentation time is known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Pace {
    /// Show it now.
    Present,
    /// Show it after this long.
    Wait(Duration),
    /// Too late to be useful — drop it, `late_ns` past the deadline.
    DropLate {
        /// How far past `max_lateness` the frame was (running time, ns).
        late_ns: u64,
    },
}

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

/// Paces presentation in segment running time.
///
/// The input is already segment-mapped (#165): `consume` runs each PTS
/// through [`SegmentEvent::to_running_time`], so segment start/rate/base
/// take effect upstream of this struct. The first frame still anchors the
/// stream — its mapped time is pinned to the running time at which it
/// arrived — because parallax has no LATENCY query: pacing purely against
/// the clock would mark the first frames late by the pipeline's startup
/// latency and drop them. The anchor absorbs that latency; a new segment
/// (seek or start) clears it so the next frame re-anchors.
///
/// Reverse playback (#165) needs no special casing here: a reverse
/// segment's `to_running_time` maps decreasing PTS to *increasing* running
/// times, so this pacer sees a normal forward timeline.
///
/// [`SegmentEvent::to_running_time`]: crate::event::SegmentEvent::to_running_time
#[derive(Debug, Default)]
struct PtsPacer {
    /// `(first PTS, running time when it arrived)`.
    anchor: Option<(ClockTime, ClockTime)>,
}

impl PtsPacer {
    /// Decide what to do with a frame stamped `pts`, given the clock now reads
    /// `now` in running time.
    ///
    /// Both times must be known; the caller checks that before asking.
    fn schedule(&mut self, pts: ClockTime, now: ClockTime, max_lateness: Duration) -> Pace {
        let (first_pts, anchor_running) = *self.anchor.get_or_insert((pts, now));

        // A PTS before the anchor means the stream went backwards (a seek, a
        // reordered first frame). Re-anchor rather than dropping everything.
        let Some(offset) = pts.nanos().checked_sub(first_pts.nanos()) else {
            self.anchor = Some((pts, now));
            return Pace::Present;
        };

        let target = anchor_running.nanos().saturating_add(offset);
        let now = now.nanos();

        if target > now {
            let wait = Duration::from_nanos(target - now);
            return if wait > MAX_WAIT {
                Pace::Present
            } else {
                Pace::Wait(wait)
            };
        }

        let deficit = now - target;
        if deficit > max_lateness.as_nanos() as u64 {
            Pace::DropLate { late_ns: deficit }
        } else {
            Pace::Present
        }
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
    /// PTS → running-time mapping, built from the first frame.
    pacer: PtsPacer,
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
            pacer: PtsPacer::default(),
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
    /// frame — the pacer re-anchors so running time continues from here
    /// instead of jumping. Sign flips are rejected (reverse needs a real
    /// seek: the data order itself must change); a later Segment (any real
    /// seek) re-establishes its own rate. Everything else is not ours.
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
            seg.rate = seek.rate;
            self.pacer.anchor = None;
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
            // A flushing seek moved the stream: drop the PTS anchor so the
            // first post-seek frame re-anchors at its own arrival time.
            // Without this a forward seek would schedule every new frame
            // `seek distance` in the future (capped by MAX_WAIT, but still
            // a stall).
            crate::event::Event::FlushStop(_) => {
                self.pacer.anchor = None;
            }
            // A new segment re-defines the mapping (#165) — and re-anchors,
            // which also covers the forward NON-flushing seek (the queued
            // Segment arrives with no FlushStop to clear the anchor).
            crate::event::Event::Segment(seg)
                if seg.format == crate::event::SegmentFormat::Time =>
            {
                self.segment = Some(seg.clone());
                self.pacer.anchor = None;
            }
            _ => {}
        }
        Some(event)
    }

    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let data = ctx.input();

        tracing::debug!("AutoVideoSink: received buffer with {} bytes", data.len());

        // Format: from per-buffer metadata; an undeclared payload is
        // treated as RGBA (the legacy contract).
        let meta = ctx.metadata();
        let format = meta.video_pixel_format().unwrap_or(PixelFormat::Rgba);

        // A strided frame (#194: External zero-copy from a decoder)
        // declares its real layout; the size gate must judge against THAT,
        // not the packed size — a strided frame is larger.
        let strided = meta
            .has_strided_planes()
            .then(|| meta.plane_layout())
            .flatten();

        // Dimensions: prefer per-buffer metadata (`MediaFormat::VideoRaw`,
        // the single geometry representation since #160); for RGBA/BGRA,
        // fall back to guessing from the buffer size. YUV frames REQUIRE
        // declared geometry — plane layout is unrecoverable without it.
        let (width, height) = match format {
            PixelFormat::I420 | PixelFormat::Nv12 => match meta.video_dims() {
                Some((w, h)) if w > 0 && h > 0 => {
                    let need = strided
                        .map(|l| l.required_len(format, w, h))
                        .unwrap_or_else(|| present::yuv420_len(w, h));
                    if data.len() < need {
                        return Err(Error::Element(format!(
                            "autovideosink: {format:?} buffer is {} bytes but {w}x{h} needs {need}",
                            data.len(),
                        )));
                    }
                    (w, h)
                }
                Some((w, h)) => {
                    return Err(Error::Element(format!(
                        "autovideosink: {format:?} buffer declares degenerate geometry {w}x{h}"
                    )));
                }
                None => {
                    return Err(Error::Element(format!(
                        "autovideosink: {format:?} buffer carries no video dimensions \
                         (decoders set them via Metadata::set_video_dims)"
                    )));
                }
            },
            _ => match meta.video_dims() {
                Some((w, h)) if w > 0 && h > 0 && (w as usize * h as usize * 4) == data.len() => {
                    (w, h)
                }
                _ => detect_dimensions(data.len()),
            },
        };

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

        // Presentation pacing (#66). Opt-in: a clock-less pipeline, an
        // un-timestamped stream, or plain `sync=false` all keep the historical
        // blit-on-arrival behaviour, so capture previews do not regress.
        if self.sync
            && ctx.clock().is_some()
            && let Some(pts) = meta.pts.to_option()
            && let Some(now) = ctx.running_time().to_option()
            // Map through the current segment (#165): (pts − start)/|rate| +
            // base. A PTS past the segment stop has no valid mapping
            // (below-start ones were dropped above) — present it immediately
            // rather than pace it.
            && let Some(pts) = self
                .segment
                .as_ref()
                .map_or(meta.pts, |seg| seg.to_running_time(pts))
                .to_option()
        {
            let mut now_ns = now.nanos();
            let mut pace = self.pacer.schedule(pts, now, self.max_lateness);
            // Waiting here is the point: it is what back-pressures the
            // decoder and the source down to real time. Since #172 the wait
            // is an await, so it pends this element's future instead of
            // parking a tokio worker.
            //
            // The wait polls the clock in short slices instead of sleeping the
            // full delay: against a clock frozen by `PipelineHandle::pause`
            // a one-shot sleep would keep presenting at wall-clock rate,
            // while re-reading running time stalls right here — and resumes
            // gap-free, because the clock does. `MAX_WAIT` still applies to
            // each *computed* deficit (the PTS-jump escape hatch), not to the
            // frozen-clock stall, where the deficit never shrinks but stays
            // under the cap.
            while let Pace::Wait(delay) = pace {
                tokio::time::sleep(delay.min(Duration::from_millis(10))).await;
                match ctx.running_time().to_option() {
                    Some(now) => {
                        now_ns = now.nanos();
                        pace = self.pacer.schedule(pts, now, self.max_lateness);
                    }
                    None => break,
                }
            }
            if let Pace::DropLate { late_ns } = pace {
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
            // Refcount bump, not a copy: the display thread maps the same
            // arena slot — or pins the same decoder picture (#194) — the
            // producer wrote (#141).
            data: ctx.buffer().clone(),
            width,
            height,
            format,
            layout: strided,
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

/// Detect frame dimensions from buffer size (assuming RGBA format).
fn detect_dimensions(size: usize) -> (u32, u32) {
    // Size = width * height * 4 (RGBA)
    let pixels = size / 4;

    // Try common resolutions
    const COMMON: &[(u32, u32)] = &[
        (640, 480),   // VGA
        (800, 600),   // SVGA
        (1024, 768),  // XGA
        (1280, 720),  // 720p
        (1280, 960),  // SXGA-
        (1920, 1080), // 1080p
        (1920, 1200), // WUXGA
        (2560, 1440), // 1440p
        (3840, 2160), // 4K
        (320, 240),   // QVGA
        (176, 144),   // QCIF
        (352, 288),   // CIF
    ];

    for &(w, h) in COMMON {
        if (w * h) as usize == pixels {
            return (w, h);
        }
    }

    // Fallback: assume 4:3 aspect ratio
    let height = ((pixels as f64).sqrt() * 0.866) as u32; // sqrt(3/4)
    let width = pixels as u32 / height.max(1);
    (width.max(1), height.max(1))
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
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| Error::Element(format!("Event loop error: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_detect_dimensions() {
        // 640x480 RGBA = 1,228,800 bytes
        assert_eq!(detect_dimensions(640 * 480 * 4), (640, 480));

        // 1280x720 RGBA = 3,686,400 bytes
        assert_eq!(detect_dimensions(1280 * 720 * 4), (1280, 720));

        // 1920x1080 RGBA = 8,294,400 bytes
        assert_eq!(detect_dimensions(1920 * 1080 * 4), (1920, 1080));
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
    fn a_time_segment_is_stored_and_re_anchors() {
        use crate::element::AsyncSink;
        use crate::event::{Event, SegmentEvent};

        let mut sink = AutoVideoSink::new();
        sink.pacer.anchor = Some((ns(0), ns(0)));

        // A Time segment installs the mapping and clears the anchor, so the
        // next frame re-anchors in the new segment's running-time domain —
        // this is also what un-stalls a forward NON-flushing seek (#165).
        let seg = SegmentEvent::new_time(ClockTime::from_secs(10), None).with_rate(2.0);
        let fwd = sink.handle_downstream_event(Event::Segment(seg));
        assert!(matches!(fwd, Some(Event::Segment(_))), "segment forwarded");
        assert!(sink.pacer.anchor.is_none(), "new segment re-anchors");
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
        sink.pacer.anchor = Some((ns(0), ns(0)));
        let bytes_seg = SegmentEvent::new_bytes(0, None);
        let _ = sink.handle_downstream_event(Event::Segment(bytes_seg));
        assert!(
            sink.pacer.anchor.is_some(),
            "bytes segment leaves pacing alone"
        );
        assert_eq!(
            sink.segment.as_ref().unwrap().format,
            crate::event::SegmentFormat::Time
        );
    }

    #[test]
    fn the_first_frame_anchors_the_stream() {
        let mut pacer = PtsPacer::default();

        // A stream that starts at 10s of PTS must not stall for 10s: the
        // anchor pins that PTS to the running time it arrived at.
        assert_eq!(
            pacer.schedule(ns(10_000_000_000), ns(5_000_000), DEFAULT_MAX_LATENESS),
            Pace::Present
        );
        // ...and the next frame is one frame interval after it.
        assert_eq!(
            pacer.schedule(
                ns(10_000_000_000 + FRAME_25FPS),
                ns(5_000_000),
                DEFAULT_MAX_LATENESS
            ),
            Pace::Wait(Duration::from_nanos(FRAME_25FPS))
        );
    }

    #[test]
    fn a_25fps_stream_waits_one_frame_interval() {
        let mut pacer = PtsPacer::default();
        let start = 1_000_000_000;

        assert_eq!(
            pacer.schedule(ns(0), ns(start), DEFAULT_MAX_LATENESS),
            Pace::Present
        );

        // Decoder runs flat out: the clock has barely moved, so each frame is
        // held back to its own presentation instant.
        for frame in 1..10u64 {
            assert_eq!(
                pacer.schedule(ns(frame * FRAME_25FPS), ns(start), DEFAULT_MAX_LATENESS),
                Pace::Wait(Duration::from_nanos(frame * FRAME_25FPS)),
                "frame {frame}"
            );
        }
    }

    #[test]
    fn a_frame_that_is_on_time_is_presented() {
        let mut pacer = PtsPacer::default();
        pacer.schedule(ns(0), ns(0), DEFAULT_MAX_LATENESS);

        // Arrived 39 ms late — inside the 40 ms budget, so still worth showing.
        assert_eq!(
            pacer.schedule(
                ns(FRAME_25FPS),
                ns(FRAME_25FPS + 39_000_000),
                DEFAULT_MAX_LATENESS
            ),
            Pace::Present
        );
    }

    #[test]
    fn a_frame_past_the_lateness_budget_is_dropped() {
        let mut pacer = PtsPacer::default();
        pacer.schedule(ns(0), ns(0), DEFAULT_MAX_LATENESS);

        assert_eq!(
            pacer.schedule(
                ns(FRAME_25FPS),
                ns(FRAME_25FPS + 41_000_000),
                DEFAULT_MAX_LATENESS
            ),
            Pace::DropLate {
                late_ns: 41_000_000
            }
        );
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
    fn the_lateness_budget_is_configurable() {
        let generous = Duration::from_millis(500);
        let mut pacer = PtsPacer::default();
        pacer.schedule(ns(0), ns(0), generous);

        // The same 41 ms of lateness that the default drops, this one shows.
        assert_eq!(
            pacer.schedule(ns(FRAME_25FPS), ns(FRAME_25FPS + 41_000_000), generous),
            Pace::Present
        );
    }

    #[test]
    fn an_absurd_pts_jump_does_not_park_the_sink() {
        let mut pacer = PtsPacer::default();
        pacer.schedule(ns(0), ns(0), DEFAULT_MAX_LATENESS);

        // An hour into the future — a wrapped RTP clock or a broken demuxer.
        // Show it rather than sleeping through the rest of the stream.
        assert_eq!(
            pacer.schedule(ns(3_600_000_000_000), ns(0), DEFAULT_MAX_LATENESS),
            Pace::Present
        );
    }

    #[test]
    fn a_backwards_pts_re_anchors_instead_of_dropping_the_stream() {
        let mut pacer = PtsPacer::default();
        pacer.schedule(ns(10 * FRAME_25FPS), ns(0), DEFAULT_MAX_LATENESS);

        // A seek back to the start: without re-anchoring, every frame from
        // here on would look eternally late and be dropped forever.
        assert_eq!(
            pacer.schedule(ns(0), ns(500_000_000), DEFAULT_MAX_LATENESS),
            Pace::Present
        );
        assert_eq!(
            pacer.schedule(ns(FRAME_25FPS), ns(500_000_000), DEFAULT_MAX_LATENESS),
            Pace::Wait(Duration::from_nanos(FRAME_25FPS))
        );
    }

    #[test]
    fn a_frozen_clock_keeps_the_pacer_waiting() {
        // PipelineHandle::pause freezes running time. Re-scheduling against
        // the same `now` must keep answering Wait with a constant deficit —
        // never Present — which is what makes the consume() poll loop stall
        // for as long as the pause lasts.
        let mut pacer = PtsPacer::default();
        pacer.schedule(ns(0), ns(0), DEFAULT_MAX_LATENESS);

        let frozen_now = ns(0);
        for _ in 0..100 {
            assert_eq!(
                pacer.schedule(ns(FRAME_25FPS), frozen_now, DEFAULT_MAX_LATENESS),
                Pace::Wait(Duration::from_nanos(FRAME_25FPS))
            );
        }
        // Resume: time advances past the target and the frame presents.
        assert_eq!(
            pacer.schedule(ns(FRAME_25FPS), ns(FRAME_25FPS), DEFAULT_MAX_LATENESS),
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

    /// #165 instant rate change: same-sign rates replace the segment's rate
    /// and re-anchor the pacer; sign flips, rate 0, and a mapping-less sink
    /// decline. Only INSTANT_RATE_CHANGE seeks are ours.
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
        sink.pacer.anchor = Some((ns(0), ns(0)));
        assert!(matches!(
            sink.handle_upstream_event(&ev),
            EventResult::Handled { .. }
        ));
        assert_eq!(sink.segment.as_ref().unwrap().rate, 2.0);
        assert!(sink.pacer.anchor.is_none(), "pacer re-anchors");

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
