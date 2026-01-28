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

use crate::element::{ConsumeContext, Sink};
use crate::error::{Error, Result};
use std::num::NonZeroU32;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::thread::{self, JoinHandle};

/// Frame data sent to the display thread.
struct DisplayFrame {
    /// RGBA pixel data
    data: Vec<u8>,
    /// Frame width in pixels
    width: u32,
    /// Frame height in pixels
    height: u32,
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
        }
    }

    /// Set the window title.
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = title.into();
        self
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

        // Bounded channel for backpressure (4 frames buffer)
        let (sender, receiver) = mpsc::sync_channel::<DisplayFrame>(4);

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

        // Detect frame dimensions from data size (assuming RGBA)
        // Common resolutions: 640x480=1228800, 1280x720=3686400, 1920x1080=8294400
        let (width, height) = detect_dimensions(data.len());

        // Start display thread on first frame
        if self.sender.is_none() {
            self.start_display(width, height)?;
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

        // Send frame (blocks if display is slow - natural backpressure)
        sender
            .send(frame)
            .map_err(|_| Error::Element("Display closed".into()))
    }

    fn name(&self) -> &str {
        &self.name
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
        window: Option<Window>,
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
                                self.window = Some(std::rc::Rc::try_unwrap(window).ok().unwrap());
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

            // Check for new frames (non-blocking)
            while let Ok(frame) = self.receiver.try_recv() {
                self.current_frame = Some(frame);
            }

            // Request redraw if we have a frame
            if self.current_frame.is_some() {
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            // Poll again after a short delay (~60fps)
            event_loop.set_control_flow(ControlFlow::wait_duration(
                std::time::Duration::from_millis(16),
            ));
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
            if let (Some(w), Some(h)) = (NonZeroU32::new(width), NonZeroU32::new(height)) {
                if surface.resize(w, h).is_err() {
                    return;
                }
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
}
