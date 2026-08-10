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

use crate::clock::ClockTime;
use crate::element::{ConsumeContext, Sink};
use crate::error::{Error, Result};
use crate::format::{Caps, PixelFormat};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Frame data sent to the display thread.
struct DisplayFrame {
    /// RGBA pixel data
    data: Vec<u8>,
    /// Frame width in pixels
    width: u32,
    /// Frame height in pixels
    height: u32,
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
        let (sender, receiver) = mpsc::sync_channel::<DisplayFrame>(8);

        let running = Arc::clone(&self.running);
        let title = self.title.clone();

        running.store(true, Ordering::SeqCst);

        let handle = thread::spawn(move || {
            if let Err(e) =
                run_display_loop(receiver, running, &title, initial_width, initial_height)
            {
                eprintln!("Display error: {}", e);
            }
        });

        self.sender = Some(sender);
        self.display_thread = Some(handle);

        Ok(())
    }

    /// Stop the display thread.
    fn stop_display(&mut self) {
        self.running.store(false, Ordering::SeqCst);

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
            match self.pacer.schedule(pts, now, self.max_lateness) {
                Pace::Present => {}
                // Blocking here is the point: it is what back-pressures the
                // decoder and the source down to real time. Other sync sinks
                // block on their I/O for the same reason.
                Pace::Wait(delay) => thread::sleep(delay),
                Pace::DropLate => {
                    self.drop_frame("late");
                    return Ok(());
                }
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
            data: data.to_vec(),
            width,
            height,
        };

        // A full channel means the display cannot keep up. Shed this frame and
        // count it — the previous code logged "drain one old frame", drained
        // nothing, and retried the same send with the result discarded.
        match sender.try_send(frame) {
            Ok(()) => Ok(()),
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
fn run_display_loop(
    receiver: mpsc::Receiver<DisplayFrame>,
    running: Arc<AtomicBool>,
    title: &str,
    initial_width: u32,
    initial_height: u32,
) -> Result<()> {
    use winit::application::ApplicationHandler;
    use winit::dpi::LogicalSize;
    use winit::event::WindowEvent;
    use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
    use winit::platform::x11::EventLoopBuilderExtX11;
    use winit::window::{Window, WindowAttributes, WindowId};

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
    }

    impl ApplicationHandler for VideoApp {
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
                    self.running.store(false, Ordering::SeqCst);
                    event_loop.exit();
                }
                WindowEvent::RedrawRequested => {
                    self.render();
                }
                WindowEvent::Resized(_) => {
                    // Surface will be resized on next render
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
                _ => {}
            }
        }

        fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
            // Check if we should exit
            if !self.running.load(Ordering::SeqCst) {
                event_loop.exit();
                return;
            }

            // Check for ONE new frame (don't drain - render each frame)
            if let Ok(frame) = self.receiver.try_recv() {
                self.current_frame = Some(frame);
                // Render immediately instead of waiting for RedrawRequested
                // This bypasses compositor vsync throttling
                self.render();
            }

            // Poll continuously - don't add artificial delay
            // The frame rate is controlled by the source, not the sink
            event_loop.set_control_flow(ControlFlow::Poll);
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
                blit_frame(frame, &mut buffer, width as usize, height as usize);
                let _ = buffer.present();
            }
        }
    }

    // Create event loop with any_thread enabled (Linux only)
    let event_loop = EventLoop::builder()
        .with_any_thread(true)
        .build()
        .map_err(|e| Error::Element(format!("Failed to create event loop: {}", e)))?;

    let mut app = VideoApp {
        window: None,
        surface: None,
        context: None,
        receiver,
        running,
        current_frame: None,
        title: title.to_string(),
        initial_width,
        initial_height,
    };

    event_loop
        .run_app(&mut app)
        .map_err(|e| Error::Element(format!("Event loop error: {}", e)))
}

/// Blit an RGBA frame to the softbuffer surface with scaling.
fn blit_frame(frame: &DisplayFrame, buffer: &mut [u32], dst_width: usize, dst_height: usize) {
    let src_width = frame.width as usize;
    let src_height = frame.height as usize;

    if src_width == 0 || src_height == 0 {
        return;
    }

    // Simple nearest-neighbor scaling
    for dst_y in 0..dst_height {
        let src_y = (dst_y * src_height) / dst_height;
        for dst_x in 0..dst_width {
            let src_x = (dst_x * src_width) / dst_width;

            let src_idx = (src_y * src_width + src_x) * 4;
            if src_idx + 3 < frame.data.len() {
                let r = frame.data[src_idx] as u32;
                let g = frame.data[src_idx + 1] as u32;
                let b = frame.data[src_idx + 2] as u32;
                // softbuffer expects 0xRRGGBB format (no alpha, RGB in low 24 bits)
                buffer[dst_y * dst_width + dst_x] = (r << 16) | (g << 8) | b;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
