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

use crate::buffer::Buffer;
use crate::clock::ClockTime;
use crate::element::{ConsumeContext, Sink};
use crate::error::{Error, Result};
use crate::format::{Caps, PixelFormat};
use std::num::NonZeroU32;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::Duration;
use winit::event_loop::EventLoopProxy;

/// Frame data sent to the display thread.
///
/// Carries the pipeline `Buffer` itself — a clone is three atomic
/// increments and (since #138) an allocation-free metadata copy, so the
/// display thread shares the arena slot instead of receiving a
/// full-frame `Vec` copy (#141). The slot stays pinned until the frame
/// is replaced, which is why the display channel is kept shallow.
struct DisplayFrame {
    /// RGBA pixel data, arena-backed.
    data: Buffer,
    /// Frame width in pixels
    width: u32,
    /// Frame height in pixels
    height: u32,
}

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
    /// Too late to be useful — drop it.
    DropLate,
}

/// Maps buffer PTS onto pipeline running time.
///
/// There are no `SegmentEvent`s anywhere in the tree yet (the executor's
/// inter-element `Message` carries only buffers, EOS and errors), so the
/// segment mapping in [`SegmentEvent::to_running_time`] has nothing to feed it.
/// Instead the first frame anchors the stream: its PTS is pinned to the running
/// time at which it arrived, and every later frame is scheduled at that anchor
/// plus its PTS offset. A stream that does not start at zero therefore plays
/// from wherever it starts, without a leading stall.
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

        if now - target > max_lateness.as_nanos() as u64 {
            Pace::DropLate
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
    /// Frames dropped for being late or for a full display channel.
    dropped: u64,
    /// Window-event channel, created when [`handle`](Self::handle) is called.
    /// `None` means no handle was taken and the event loop sends nothing —
    /// exactly the historical behavior.
    events: Option<(
        kanal::Sender<VideoWindowEvent>,
        kanal::Receiver<VideoWindowEvent>,
    )>,
    /// Desired fullscreen state, applied by the event loop when woken.
    fullscreen: Arc<AtomicBool>,
    /// Doorbell into the display loop; see [`SharedProxy`].
    proxy: SharedProxy,
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
    /// pipeline sees EOS via `is_open()`; this event lets the app react too.
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
    events: kanal::Receiver<VideoWindowEvent>,
    fullscreen: Arc<AtomicBool>,
    running: Arc<AtomicBool>,
    proxy: SharedProxy,
}

impl AutoVideoSinkHandle {
    /// Next pending window event, if any. Non-blocking.
    pub fn try_event(&self) -> Option<VideoWindowEvent> {
        self.events.try_recv().ok().flatten()
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
            dropped: 0,
            events: None,
            fullscreen: Arc::new(AtomicBool::new(false)),
            proxy: Arc::new(Mutex::new(None)),
        }
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
            .get_or_insert_with(|| kanal::bounded::<VideoWindowEvent>(64));
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

        // Bounded channel for backpressure (8 frames buffer)
        // Shallow on purpose: each queued frame pins an upstream arena slot
        // (see DisplayFrame). 3 in flight + 1 presented stays well inside the
        // producer's IN_FLIGHT_MARGIN.
        let (sender, receiver) = mpsc::sync_channel::<DisplayFrame>(3);

        let running = Arc::clone(&self.running);
        let title = self.title.clone();
        let events_tx = self.events.as_ref().map(|(tx, _)| tx.clone());
        let fullscreen = self.fullscreen.clone();
        let proxy = self.proxy.clone();

        running.store(true, Ordering::SeqCst);

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

        // Wait for thread to finish
        if let Some(handle) = self.display_thread.take() {
            let _ = handle.join();
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

impl Sink for AutoVideoSink {
    fn handle_downstream_event(
        &mut self,
        event: crate::event::Event,
    ) -> Option<crate::event::Event> {
        // A flushing seek moved the stream: drop the PTS anchor so the first
        // post-seek frame re-anchors at its own arrival time. Without this a
        // forward seek would schedule every new frame `seek distance` in the
        // future (capped by MAX_WAIT, but still a stall).
        if matches!(event, crate::event::Event::FlushStop(_)) {
            self.pacer.anchor = None;
        }
        Some(event)
    }

    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let data = ctx.input();

        tracing::debug!("AutoVideoSink: received buffer with {} bytes", data.len());

        // Dimensions: prefer per-buffer metadata, fall back to guessing from
        // the RGBA buffer size. `video_dims` reads both conventions — the
        // `MediaFormat::VideoRaw` one and the legacy "width"/"height" keys — so
        // an upstream element that set only one of them still works.
        let meta = ctx.metadata();
        let (width, height) = match meta.video_dims() {
            Some((w, h)) if w > 0 && h > 0 && (w as usize * h as usize * 4) == data.len() => (w, h),
            _ => detect_dimensions(data.len()),
        };

        tracing::debug!("AutoVideoSink: detected dimensions {}x{}", width, height);

        // Presentation pacing (#66). Opt-in: a clock-less pipeline, an
        // un-timestamped stream, or plain `sync=false` all keep the historical
        // blit-on-arrival behaviour, so capture previews do not regress.
        if self.sync
            && ctx.clock().is_some()
            && let Some(pts) = meta.pts.to_option()
            && let Some(now) = ctx.running_time().to_option()
        {
            let mut pace = self.pacer.schedule(pts, now, self.max_lateness);
            // Blocking here is the point: it is what back-pressures the
            // decoder and the source down to real time. Other sync sinks
            // block on their I/O for the same reason.
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
                thread::sleep(delay.min(Duration::from_millis(10)));
                match ctx.running_time().to_option() {
                    Some(now) => pace = self.pacer.schedule(pts, now, self.max_lateness),
                    None => break,
                }
            }
            if pace == Pace::DropLate {
                self.drop_frame("late");
                return Ok(());
            }
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

        // Check if display is still running
        if !self.running.load(Ordering::SeqCst) {
            return Err(Error::Element("Display window closed".into()));
        }

        let frame = DisplayFrame {
            // Refcount bump, not a copy: the display thread maps the same
            // arena slot the converter wrote (#141).
            data: ctx.buffer().clone(),
            width,
            height,
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
            Err(mpsc::TrySendError::Disconnected(_)) => {
                Err(Error::Element("Display closed".into()))
            }
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn input_caps(&self) -> Caps {
        // AutoVideoSink expects RGBA data (any resolution)
        Caps::video_raw_any_resolution(PixelFormat::Rgba)
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
    events_tx: Option<kanal::Sender<VideoWindowEvent>>,
    fullscreen: Arc<AtomicBool>,
    proxy_slot: SharedProxy,
) -> Result<()> {
    use winit::application::ApplicationHandler;
    use winit::dpi::LogicalSize;
    use winit::event::{ElementState, MouseButton, WindowEvent};
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::keyboard::{Key, NamedKey};
    use winit::platform::x11::EventLoopBuilderExtX11;
    use winit::window::{Fullscreen, Window, WindowAttributes, WindowId};

    struct VideoApp {
        window: Option<std::rc::Rc<Window>>,
        surface: Option<softbuffer::Surface<std::rc::Rc<Window>, std::rc::Rc<Window>>>,
        context: Option<softbuffer::Context<std::rc::Rc<Window>>>,
        receiver: mpsc::Receiver<DisplayFrame>,
        running: Arc<AtomicBool>,
        current_frame: Option<DisplayFrame>,
        title: String,
        initial_width: u32,
        initial_height: u32,
        events_tx: Option<kanal::Sender<VideoWindowEvent>>,
        blit_cache: BlitCache,
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

            let attrs = WindowAttributes::default()
                .with_title(&self.title)
                .with_inner_size(LogicalSize::new(self.initial_width, self.initial_height));

            match event_loop.create_window(attrs) {
                Ok(window) => {
                    let window = std::rc::Rc::new(window);

                    // Create softbuffer context and surface
                    match softbuffer::Context::new(window.clone()) {
                        Ok(context) => match softbuffer::Surface::new(&context, window.clone()) {
                            Ok(surface) => {
                                self.context = Some(context);
                                self.surface = Some(surface);
                                self.window = Some(window);
                            }
                            Err(e) => {
                                eprintln!("Failed to create surface: {}", e);
                                self.running.store(false, Ordering::SeqCst);
                                event_loop.exit();
                            }
                        },
                        Err(e) => {
                            eprintln!("Failed to create softbuffer context: {}", e);
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
                    // Surface will be resized on next render
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
            let Some(surface) = &mut self.surface else {
                return;
            };
            let Some(frame) = &self.current_frame else {
                return;
            };

            let size = window.inner_size();
            let width = size.width;
            let height = size.height;

            if width == 0 || height == 0 {
                return;
            }

            // Resize surface if needed
            if let (Some(w), Some(h)) = (NonZeroU32::new(width), NonZeroU32::new(height))
                && surface.resize(w, h).is_err()
            {
                return;
            }

            // Get buffer and blit frame
            if let Ok(mut buffer) = surface.buffer_mut() {
                blit_frame(
                    frame,
                    &mut buffer,
                    width as usize,
                    height as usize,
                    &mut self.blit_cache,
                );
                let _ = buffer.present();
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
        window: None,
        surface: None,
        context: None,
        receiver,
        running,
        current_frame: None,
        blit_cache: BlitCache::default(),
        title: title.to_string(),
        initial_width,
        initial_height,
        events_tx,
        fullscreen,
        is_fullscreen: false,
        cursor: (0.0, 0.0),
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| Error::Element(format!("Event loop error: {}", e)))
}

/// Blit an RGBA frame to the softbuffer surface with scaling.
/// Horizontal nearest-neighbour map, cached across frames — rebuilding it
/// costs one small allocation only when the source or window geometry
/// changes.
#[derive(Default)]
struct BlitCache {
    key: (usize, usize),
    x_map: Vec<u32>,
}

fn blit_frame(
    frame: &DisplayFrame,
    buffer: &mut [u32],
    dst_width: usize,
    dst_height: usize,
    cache: &mut BlitCache,
) {
    let src_width = frame.width as usize;
    let src_height = frame.height as usize;
    let src = frame.data.as_bytes();

    if src_width == 0
        || src_height == 0
        || dst_width == 0
        || dst_height == 0
        || src.len() < src_width * src_height * 4
        || buffer.len() < dst_width * dst_height
    {
        return;
    }

    // Aspect-preserving letterbox: scale to fit, center, black bars around.
    // Stretching to the window distorted anything whose aspect ratio did not
    // match — obvious the moment a 16:9 stream met a fullscreen 16:10 panel.
    let (out_width, out_height) = if src_width * dst_height >= src_height * dst_width {
        // Width-bound: pillar-free, bars top/bottom.
        (dst_width, (src_height * dst_width / src_width).max(1))
    } else {
        // Height-bound: bars left/right.
        ((src_width * dst_height / src_height).max(1), dst_height)
    };
    let x0 = (dst_width - out_width) / 2;
    let y0 = (dst_height - out_height) / 2;

    if cache.key != (src_width, out_width) {
        cache.key = (src_width, out_width);
        cache.x_map.clear();
        cache
            .x_map
            .extend((0..out_width).map(|x| ((x * src_width) / out_width) as u32));
    }

    // Row-wise: bars fill by slice, the image row converts RGBA → 0RGB with
    // all bounds established once per row (#141) — the previous per-pixel
    // loop paid a bounds check, two `contains`, and a mul+div per pixel.
    for dst_y in 0..dst_height {
        let row = &mut buffer[dst_y * dst_width..(dst_y + 1) * dst_width];
        if dst_y < y0 || dst_y >= y0 + out_height {
            row.fill(0); // letterbox bar
            continue;
        }
        row[..x0].fill(0);
        row[x0 + out_width..].fill(0);

        let src_y = ((dst_y - y0) * src_height) / out_height;
        let src_row = &src[src_y * src_width * 4..(src_y + 1) * src_width * 4];
        let out_row = &mut row[x0..x0 + out_width];
        if out_width == src_width {
            // 1:1 horizontal: straight zip, no index math.
            for (dst, px) in out_row.iter_mut().zip(src_row.chunks_exact(4)) {
                // softbuffer expects 0xRRGGBB (no alpha, RGB in the low 24 bits).
                *dst = ((px[0] as u32) << 16) | ((px[1] as u32) << 8) | (px[2] as u32);
            }
        } else {
            for (dst, &sx) in out_row.iter_mut().zip(&cache.x_map[..out_width]) {
                let o = sx as usize * 4;
                *dst = ((src_row[o] as u32) << 16)
                    | ((src_row[o + 1] as u32) << 8)
                    | (src_row[o + 2] as u32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An arena-backed white RGBA frame of `w`x`h` for blit tests.
    fn white_frame(w: u32, h: u32) -> DisplayFrame {
        let len = (w * h * 4) as usize;
        let arena = crate::memory::SharedArena::new(len, 2).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..len].fill(0xFF);
        DisplayFrame {
            data: Buffer::new(
                crate::buffer::MemoryHandle::with_len(slot, len),
                crate::metadata::Metadata::new(),
            ),
            width: w,
            height: h,
        }
    }

    #[test]
    fn letterbox_centers_and_paints_bars_black() {
        // 2x2 white frame into a 4x2 window: 1:1 content in the middle two
        // columns, black pillars on columns 0 and 3.
        let frame = white_frame(2, 2);
        let mut buffer = vec![0x123456u32; 4 * 2];
        blit_frame(&frame, &mut buffer, 4, 2, &mut BlitCache::default());

        for y in 0..2 {
            assert_eq!(buffer[y * 4], 0, "left pillar is black");
            assert_eq!(buffer[y * 4 + 3], 0, "right pillar is black");
            assert_eq!(buffer[y * 4 + 1], 0xFFFFFF, "content");
            assert_eq!(buffer[y * 4 + 2], 0xFFFFFF, "content");
        }
    }

    #[test]
    fn matching_aspect_fills_the_window() {
        let frame = white_frame(2, 2);
        let mut buffer = vec![0u32; 8 * 8];
        blit_frame(&frame, &mut buffer, 8, 8, &mut BlitCache::default());
        assert!(buffer.iter().all(|p| *p == 0xFFFFFF), "no bars");
    }

    /// The scaled path via the cached x-map produces the same result when
    /// geometry changes between frames (cache rebuild).
    #[test]
    fn blit_cache_survives_geometry_changes() {
        let mut cache = BlitCache::default();
        let frame = white_frame(2, 2);
        let mut a = vec![0u32; 8 * 8];
        blit_frame(&frame, &mut a, 8, 8, &mut cache);
        let mut b = vec![0u32; 6 * 6];
        blit_frame(&frame, &mut b, 6, 6, &mut cache);
        assert!(a.iter().all(|p| *p == 0xFFFFFF));
        assert!(b.iter().all(|p| *p == 0xFFFFFF));
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
            Pace::DropLate
        );
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
}
