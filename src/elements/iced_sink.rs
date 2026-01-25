//! IcedVideoSink - Display video frames in an Iced GUI window.
//!
//! This module provides a video sink that displays frames in a native window
//! using the Iced GUI framework.
//!
//! # Architecture
//!
//! Since Iced runs its own event loop, we use a channel-based approach:
//! - `IcedVideoSink` implements `Sink` and sends frames through a channel
//! - `VideoWindow` is the Iced application that receives and displays frames
//! - `IcedVideoSinkHandle` provides control over the window (close, resize, etc.)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::{VideoTestSrc, VideoPattern, IcedVideoSink};
//! use parallax::element::{Source, Sink};
//!
//! // Create a video source
//! let mut src = VideoTestSrc::new()
//!     .with_pattern(VideoPattern::SmpteColorBars)
//!     .with_resolution(640, 480);
//!
//! // Create the sink and get a handle to run the window
//! let (mut sink, handle) = IcedVideoSink::new(640, 480);
//!
//! // Spawn a thread to feed frames
//! std::thread::spawn(move || {
//!     while let Ok(Some(buf)) = src.produce() {
//!         if sink.consume(buf).is_err() {
//!             break;
//!         }
//!     }
//! });
//!
//! // Run the Iced window (blocks until closed)
//! handle.run().unwrap();
//! ```

use crate::buffer::Buffer;
use crate::element::Sink;
use crate::error::{Error, Result};
use iced::widget::{column, container, image, text};
use iced::{Element, Length, Subscription, Task, Theme};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Statistics for the video sink.
#[derive(Debug, Clone, Default)]
pub struct IcedVideoSinkStats {
    /// Number of frames received.
    pub frames_received: u64,
    /// Number of frames displayed.
    pub frames_displayed: u64,
    /// Number of frames dropped (buffer full).
    pub frames_dropped: u64,
    /// Current display FPS.
    pub display_fps: f64,
}

/// Shared state between the sink and the Iced window.
struct SharedState {
    /// Current frame data (RGBA pixels).
    frame_data: Mutex<Option<FrameData>>,
    /// Whether the window is still open.
    window_open: AtomicBool,
    /// Frame counter for statistics.
    frames_received: AtomicU64,
    /// Frames displayed counter.
    frames_displayed: AtomicU64,
    /// Frames dropped counter.
    frames_dropped: AtomicU64,
}

/// Frame data passed to the window.
struct FrameData {
    /// RGBA pixel data.
    pixels: Vec<u8>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Presentation timestamp (for display).
    #[allow(dead_code)]
    pts_nanos: u64,
    /// Sequence number.
    #[allow(dead_code)]
    sequence: u64,
}

/// Configuration for the video sink.
#[derive(Debug, Clone)]
pub struct IcedVideoSinkConfig {
    /// Window title.
    pub title: String,
    /// Initial window width.
    pub width: u32,
    /// Initial window height.
    pub height: u32,
    /// Show statistics overlay.
    pub show_stats: bool,
    /// Expected input pixel format.
    pub pixel_format: InputPixelFormat,
}

impl Default for IcedVideoSinkConfig {
    fn default() -> Self {
        Self {
            title: "Video".to_string(),
            width: 640,
            height: 480,
            show_stats: false,
            pixel_format: InputPixelFormat::Rgba32,
        }
    }
}

/// Expected input pixel format for the sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputPixelFormat {
    /// RGBA with 8 bits per channel (32 bits per pixel).
    #[default]
    Rgba32,
    /// RGB with 8 bits per channel (24 bits per pixel) - will be converted to RGBA.
    Rgb24,
    /// BGRA with 8 bits per channel (32 bits per pixel) - will be converted to RGBA.
    Bgra32,
    /// BGR with 8 bits per channel (24 bits per pixel) - will be converted to RGBA.
    Bgr24,
}

impl InputPixelFormat {
    /// Get bytes per pixel for this format.
    pub fn bytes_per_pixel(&self) -> usize {
        match self {
            InputPixelFormat::Rgba32 | InputPixelFormat::Bgra32 => 4,
            InputPixelFormat::Rgb24 | InputPixelFormat::Bgr24 => 3,
        }
    }

    /// Convert pixels from this format to RGBA.
    fn to_rgba(&self, input: &[u8], width: u32, height: u32) -> Vec<u8> {
        let pixel_count = (width * height) as usize;
        let mut output = vec![0u8; pixel_count * 4];

        match self {
            InputPixelFormat::Rgba32 => {
                output.copy_from_slice(input);
            }
            InputPixelFormat::Rgb24 => {
                for i in 0..pixel_count {
                    output[i * 4] = input[i * 3];
                    output[i * 4 + 1] = input[i * 3 + 1];
                    output[i * 4 + 2] = input[i * 3 + 2];
                    output[i * 4 + 3] = 255;
                }
            }
            InputPixelFormat::Bgra32 => {
                for i in 0..pixel_count {
                    output[i * 4] = input[i * 4 + 2]; // R from B
                    output[i * 4 + 1] = input[i * 4 + 1]; // G
                    output[i * 4 + 2] = input[i * 4]; // B from R
                    output[i * 4 + 3] = input[i * 4 + 3]; // A
                }
            }
            InputPixelFormat::Bgr24 => {
                for i in 0..pixel_count {
                    output[i * 4] = input[i * 3 + 2]; // R from B
                    output[i * 4 + 1] = input[i * 3 + 1]; // G
                    output[i * 4 + 2] = input[i * 3]; // B from R
                    output[i * 4 + 3] = 255;
                }
            }
        }

        output
    }
}

/// A video sink that displays frames in an Iced GUI window.
///
/// This sink sends frames to an Iced application for display. The actual
/// window is run separately using `IcedVideoSinkHandle::run()`.
pub struct IcedVideoSink {
    name: String,
    config: IcedVideoSinkConfig,
    state: Arc<SharedState>,
    frames_consumed: u64,
}

impl IcedVideoSink {
    /// Create a new video sink with the given dimensions.
    ///
    /// Returns the sink and a handle to run the window.
    pub fn new(width: u32, height: u32) -> (Self, IcedVideoSinkHandle) {
        Self::with_config(IcedVideoSinkConfig {
            width,
            height,
            ..Default::default()
        })
    }

    /// Create a new video sink with custom configuration.
    pub fn with_config(config: IcedVideoSinkConfig) -> (Self, IcedVideoSinkHandle) {
        let state = Arc::new(SharedState {
            frame_data: Mutex::new(None),
            window_open: AtomicBool::new(true),
            frames_received: AtomicU64::new(0),
            frames_displayed: AtomicU64::new(0),
            frames_dropped: AtomicU64::new(0),
        });

        let sink = Self {
            name: "iced_video_sink".to_string(),
            config: config.clone(),
            state: Arc::clone(&state),
            frames_consumed: 0,
        };

        let handle = IcedVideoSinkHandle { config, state };

        (sink, handle)
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Check if the window is still open.
    pub fn is_window_open(&self) -> bool {
        self.state.window_open.load(Ordering::Relaxed)
    }

    /// Get statistics.
    pub fn stats(&self) -> IcedVideoSinkStats {
        IcedVideoSinkStats {
            frames_received: self.state.frames_received.load(Ordering::Relaxed),
            frames_displayed: self.state.frames_displayed.load(Ordering::Relaxed),
            frames_dropped: self.state.frames_dropped.load(Ordering::Relaxed),
            display_fps: 0.0, // Would need timing to calculate
        }
    }
}

impl Sink for IcedVideoSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        // Check if window is still open
        if !self.state.window_open.load(Ordering::Relaxed) {
            return Err(Error::Element("Window closed".into()));
        }

        let data = buffer.as_bytes();
        let expected_size = (self.config.width * self.config.height) as usize
            * self.config.pixel_format.bytes_per_pixel();

        if data.len() != expected_size {
            return Err(Error::Element(format!(
                "Frame size mismatch: expected {} bytes, got {}",
                expected_size,
                data.len()
            )));
        }

        // Convert to RGBA
        let rgba_pixels =
            self.config
                .pixel_format
                .to_rgba(data, self.config.width, self.config.height);

        let frame = FrameData {
            pixels: rgba_pixels,
            width: self.config.width,
            height: self.config.height,
            pts_nanos: buffer.metadata().pts.nanos(),
            sequence: buffer.metadata().sequence,
        };

        // Try to update the frame
        {
            let mut frame_lock = self.state.frame_data.lock().unwrap();
            if frame_lock.is_some() {
                // Previous frame not yet consumed, drop it
                self.state.frames_dropped.fetch_add(1, Ordering::Relaxed);
            }
            *frame_lock = Some(frame);
        }

        self.state.frames_received.fetch_add(1, Ordering::Relaxed);
        self.frames_consumed += 1;

        Ok(())
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Handle to run the Iced video window.
///
/// This handle is used to start the Iced event loop and display the window.
/// The `run()` method blocks until the window is closed.
pub struct IcedVideoSinkHandle {
    config: IcedVideoSinkConfig,
    state: Arc<SharedState>,
}

/// Global state holder for Iced application initialization.
/// This is needed because Iced 0.14's boot function must be 'static.
static INIT_STATE: std::sync::OnceLock<VideoWindowInit> = std::sync::OnceLock::new();

struct VideoWindowInit {
    state: Arc<SharedState>,
    width: u32,
    height: u32,
    show_stats: bool,
}

impl IcedVideoSinkHandle {
    /// Run the video window.
    ///
    /// This method blocks until the window is closed.
    pub fn run(self) -> iced::Result {
        let settings = iced::window::Settings {
            size: iced::Size::new(self.config.width as f32, self.config.height as f32),
            ..Default::default()
        };

        // Store init state globally (only one video window per process)
        let _ = INIT_STATE.set(VideoWindowInit {
            state: self.state,
            width: self.config.width,
            height: self.config.height,
            show_stats: self.config.show_stats,
        });

        iced::application(VideoWindow::boot, VideoWindow::update, VideoWindow::view)
            .subscription(VideoWindow::subscription)
            .window(settings)
            .theme(VideoWindow::theme)
            .title(VideoWindow::title)
            .run()
    }

    /// Close the window.
    ///
    /// This signals the window to close. The actual close happens
    /// asynchronously when the Iced event loop processes the request.
    pub fn close(&self) {
        self.state.window_open.store(false, Ordering::Relaxed);
    }
}

/// Messages for the video window.
#[derive(Debug, Clone)]
enum Message {
    /// Tick for frame refresh.
    Tick,
}

/// The Iced application for displaying video.
struct VideoWindow {
    state: Arc<SharedState>,
    current_handle: Option<image::Handle>,
    width: u32,
    height: u32,
    show_stats: bool,
    last_frame_time: Instant,
    fps: f64,
    frame_count: u64,
    fps_update_time: Instant,
    fps_frame_count: u64,
}

impl VideoWindow {
    fn boot() -> Self {
        let init = INIT_STATE.get().expect("VideoWindow not initialized");
        let now = Instant::now();
        Self {
            state: Arc::clone(&init.state),
            current_handle: None,
            width: init.width,
            height: init.height,
            show_stats: init.show_stats,
            last_frame_time: now,
            fps: 0.0,
            frame_count: 0,
            fps_update_time: now,
            fps_frame_count: 0,
        }
    }

    fn theme(&self) -> Theme {
        Theme::Dark
    }

    fn title(&self) -> String {
        "Video".to_string()
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Tick => {
                // Check for new frame
                let mut frame_lock = self.state.frame_data.lock().unwrap();
                if let Some(frame) = frame_lock.take() {
                    // Update dimensions if frame size changed
                    self.width = frame.width;
                    self.height = frame.height;

                    // Create new image handle
                    self.current_handle = Some(image::Handle::from_rgba(
                        frame.width,
                        frame.height,
                        frame.pixels,
                    ));

                    self.state.frames_displayed.fetch_add(1, Ordering::Relaxed);
                    self.frame_count += 1;

                    // Update FPS calculation
                    let now = Instant::now();
                    self.fps_frame_count += 1;
                    let elapsed = now.duration_since(self.fps_update_time);
                    if elapsed >= Duration::from_secs(1) {
                        self.fps = self.fps_frame_count as f64 / elapsed.as_secs_f64();
                        self.fps_update_time = now;
                        self.fps_frame_count = 0;
                    }
                    self.last_frame_time = now;
                }
                drop(frame_lock);

                // Check if we should close
                if !self.state.window_open.load(Ordering::Relaxed) {
                    return iced::exit();
                }

                Task::none()
            }
        }
    }

    fn view(&self) -> Element<'_, Message> {
        let content: Element<Message> = if let Some(handle) = &self.current_handle {
            let img = image::Image::new(handle.clone())
                .content_fit(iced::ContentFit::Contain)
                .width(Length::Fill)
                .height(Length::Fill);

            if self.show_stats {
                let stats_text = format!(
                    "Frame: {} | FPS: {:.1} | Dropped: {}",
                    self.frame_count,
                    self.fps,
                    self.state.frames_dropped.load(Ordering::Relaxed)
                );
                column![img, text(stats_text).size(14),].into()
            } else {
                img.into()
            }
        } else {
            let placeholder = text("Waiting for video...")
                .size(24)
                .width(Length::Fill)
                .align_x(iced::alignment::Horizontal::Center);

            if self.show_stats {
                column![
                    container(placeholder)
                        .width(Length::Fill)
                        .height(Length::Fill)
                        .center_x(Length::Fill)
                        .center_y(Length::Fill),
                    text("No frames received").size(14),
                ]
                .into()
            } else {
                container(placeholder)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .center_x(Length::Fill)
                    .center_y(Length::Fill)
                    .into()
            }
        };

        container(content)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        // Poll for new frames at ~60Hz
        iced::time::every(Duration::from_millis(16)).map(|_| Message::Tick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_input_pixel_format_bytes_per_pixel() {
        assert_eq!(InputPixelFormat::Rgba32.bytes_per_pixel(), 4);
        assert_eq!(InputPixelFormat::Rgb24.bytes_per_pixel(), 3);
        assert_eq!(InputPixelFormat::Bgra32.bytes_per_pixel(), 4);
        assert_eq!(InputPixelFormat::Bgr24.bytes_per_pixel(), 3);
    }

    #[test]
    fn test_rgb24_to_rgba() {
        let format = InputPixelFormat::Rgb24;
        let input = vec![255, 0, 0, 0, 255, 0, 0, 0, 255]; // Red, Green, Blue pixels
        let output = format.to_rgba(&input, 3, 1);

        assert_eq!(output.len(), 12); // 3 pixels * 4 bytes
        // Red pixel
        assert_eq!(&output[0..4], &[255, 0, 0, 255]);
        // Green pixel
        assert_eq!(&output[4..8], &[0, 255, 0, 255]);
        // Blue pixel
        assert_eq!(&output[8..12], &[0, 0, 255, 255]);
    }

    #[test]
    fn test_bgr24_to_rgba() {
        let format = InputPixelFormat::Bgr24;
        let input = vec![0, 0, 255, 0, 255, 0, 255, 0, 0]; // Red, Green, Blue in BGR
        let output = format.to_rgba(&input, 3, 1);

        assert_eq!(output.len(), 12);
        // Red pixel (was BGR: 0, 0, 255)
        assert_eq!(&output[0..4], &[255, 0, 0, 255]);
        // Green pixel (was BGR: 0, 255, 0)
        assert_eq!(&output[4..8], &[0, 255, 0, 255]);
        // Blue pixel (was BGR: 255, 0, 0)
        assert_eq!(&output[8..12], &[0, 0, 255, 255]);
    }

    #[test]
    fn test_bgra32_to_rgba() {
        let format = InputPixelFormat::Bgra32;
        let input = vec![0, 0, 255, 128]; // Red with alpha 128 in BGRA
        let output = format.to_rgba(&input, 1, 1);

        assert_eq!(output.len(), 4);
        assert_eq!(&output[0..4], &[255, 0, 0, 128]);
    }

    #[test]
    fn test_rgba32_passthrough() {
        let format = InputPixelFormat::Rgba32;
        let input = vec![255, 128, 64, 32];
        let output = format.to_rgba(&input, 1, 1);

        assert_eq!(output, input);
    }

    #[test]
    fn test_sink_creation() {
        let (sink, _handle) = IcedVideoSink::new(640, 480);
        assert!(sink.is_window_open());
        assert_eq!(sink.name(), "iced_video_sink");
    }

    #[test]
    fn test_sink_with_config() {
        let config = IcedVideoSinkConfig {
            title: "Test Window".to_string(),
            width: 1920,
            height: 1080,
            show_stats: true,
            pixel_format: InputPixelFormat::Rgb24,
        };

        let (sink, _handle) = IcedVideoSink::with_config(config);
        assert!(sink.is_window_open());
    }

    #[test]
    fn test_sink_stats_initial() {
        let (sink, _handle) = IcedVideoSink::new(640, 480);
        let stats = sink.stats();

        assert_eq!(stats.frames_received, 0);
        assert_eq!(stats.frames_displayed, 0);
        assert_eq!(stats.frames_dropped, 0);
    }

    #[test]
    fn test_handle_close() {
        let (sink, handle) = IcedVideoSink::new(640, 480);
        assert!(sink.is_window_open());

        handle.close();
        assert!(!sink.is_window_open());
    }
}
