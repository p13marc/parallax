//! Screen capture via XDG Desktop Portal.
//!
//! This module provides screen capture on Wayland and X11 using the
//! XDG Desktop Portal's ScreenCast interface. The portal handles:
//! - User permission dialogs
//! - Window/monitor selection
//! - PipeWire stream setup
//!
//! # Requirements
//!
//! - XDG Desktop Portal service running (standard on most Linux desktops)
//! - PipeWire session manager
//! - Portal backend (e.g., xdg-desktop-portal-gnome, xdg-desktop-portal-kde)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::device::screen_capture::{ScreenCaptureSrc, ScreenCaptureConfig};
//!
//! // Capture the entire screen (user selects which monitor)
//! let config = ScreenCaptureConfig::default();
//! let mut capture = ScreenCaptureSrc::new(config);
//!
//! // Initialize (shows permission dialog)
//! capture.initialize().await?;
//!
//! // Use in a pipeline
//! let pipeline = Pipeline::new();
//! pipeline.add_element("screen", Src(capture));
//! ```
//!
//! # How It Works
//!
//! 1. Create a ScreenCast session via D-Bus portal
//! 2. User is prompted to select a screen/window
//! 3. Portal returns a PipeWire node ID and fd
//! 4. We connect to PipeWire using the fd and capture frames from that node
//! 5. Frames are delivered as BGRA or other negotiated format

use std::os::fd::OwnedFd;
use std::thread;

use ashpd::desktop::PersistMode;
use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
use ashpd::enumflags2::BitFlags;
use kanal::{Receiver, Sender, bounded};
use pipewire as pw;

use crate::buffer::Buffer;
use crate::element::{Affinity, ExecutionHints, ProduceContext, ProduceResult, Source};
use crate::error::{Error, Result};
use crate::format::{
    CapsValue, ElementMediaCaps, FormatMemoryCap, MemoryCaps, PixelFormat, VideoFormatCaps,
};
use crate::memory::SharedArena;
use crate::metadata::Metadata;

use super::DeviceError;

/// Screen capture source type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CaptureSourceType {
    /// Capture a monitor (full screen).
    #[default]
    Monitor,
    /// Capture a specific window.
    Window,
    /// Allow user to choose either.
    Any,
}

impl CaptureSourceType {
    /// Convert to BitFlags for the portal API.
    fn to_source_type(self) -> BitFlags<SourceType> {
        match self {
            CaptureSourceType::Monitor => SourceType::Monitor.into(),
            CaptureSourceType::Window => SourceType::Window.into(),
            CaptureSourceType::Any => SourceType::Monitor | SourceType::Window,
        }
    }
}

/// Screen capture configuration.
#[derive(Debug, Clone)]
pub struct ScreenCaptureConfig {
    /// Type of source to capture (monitor, window, or any).
    pub source_type: CaptureSourceType,
    /// Whether to show the cursor in the capture.
    pub show_cursor: bool,
    /// Whether to persist the session across restarts.
    /// If true, the user won't be prompted again if they previously granted permission.
    pub persist_session: bool,
}

impl Default for ScreenCaptureConfig {
    fn default() -> Self {
        Self {
            source_type: CaptureSourceType::Monitor,
            show_cursor: true,
            persist_session: false,
        }
    }
}

/// Information about an active screen capture session.
#[derive(Debug, Clone)]
pub struct ScreenCaptureInfo {
    /// Width of the captured stream.
    pub width: u32,
    /// Height of the captured stream.
    pub height: u32,
    /// PipeWire node ID for the stream.
    pub node_id: u32,
}

/// Captured frame data.
#[derive(Debug, Clone)]
pub struct CapturedFrame {
    /// Raw pixel data.
    pub data: Vec<u8>,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Bytes per row (stride).
    pub stride: u32,
    /// Pixel format (typically BGRA on most systems).
    pub format: PixelFormat,
}

/// Screen capture source element.
///
/// Captures screen content via XDG Desktop Portal and PipeWire.
/// The user will be prompted to select a screen or window when the
/// element is first started.
pub struct ScreenCaptureSrc {
    config: ScreenCaptureConfig,
    /// Receiver for captured frames from PipeWire thread.
    frame_receiver: Option<Receiver<CapturedFrame>>,
    /// Sender to signal shutdown.
    shutdown_sender: Option<Sender<()>>,
    /// Capture thread handle.
    capture_thread: Option<thread::JoinHandle<()>>,
    /// Capture info (set after session is started).
    info: Option<ScreenCaptureInfo>,
    /// Arena for output buffers.
    arena: Option<SharedArena>,
    /// Whether the session has been initialized.
    initialized: bool,
}

impl std::fmt::Debug for ScreenCaptureSrc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScreenCaptureSrc")
            .field("config", &self.config)
            .field("info", &self.info)
            .field("initialized", &self.initialized)
            .finish()
    }
}

impl ScreenCaptureSrc {
    /// Create a new screen capture source with the given configuration.
    ///
    /// This only sets up the configuration. The actual portal session
    /// is created when `initialize()` is called (typically during pipeline prepare).
    pub fn new(config: ScreenCaptureConfig) -> Self {
        Self {
            config,
            frame_receiver: None,
            shutdown_sender: None,
            capture_thread: None,
            info: None,
            arena: None,
            initialized: false,
        }
    }

    /// Create a screen capture source with default configuration.
    pub fn default_config() -> Self {
        Self::new(ScreenCaptureConfig::default())
    }

    /// Initialize the portal session and start capturing.
    ///
    /// This will prompt the user to select a screen/window to capture.
    /// Must be called before producing frames.
    pub async fn initialize(&mut self) -> Result<ScreenCaptureInfo> {
        if self.initialized {
            return self
                .info
                .clone()
                .ok_or_else(|| Error::Device(DeviceError::NotFound("No capture info".into())));
        }

        // Create the screencast portal proxy
        let screencast = Screencast::new()
            .await
            .map_err(|e| Error::Device(DeviceError::PipeWire(format!("Portal error: {}", e))))?;

        // Create a session
        let session = screencast
            .create_session()
            .await
            .map_err(|e| Error::Device(DeviceError::PipeWire(format!("Session error: {}", e))))?;

        // Configure the cursor mode
        let cursor_mode = if self.config.show_cursor {
            CursorMode::Embedded
        } else {
            CursorMode::Hidden
        };

        // Select sources (this will show the portal dialog)
        screencast
            .select_sources(
                &session,
                cursor_mode,
                self.config.source_type.to_source_type(),
                false, // multiple sources
                None,  // restore token
                PersistMode::DoNot,
            )
            .await
            .map_err(|e| {
                Error::Device(DeviceError::PipeWire(format!(
                    "Select sources error: {}",
                    e
                )))
            })?;

        // Start the screencast (this shows the permission dialog)
        let response = screencast
            .start(&session, None)
            .await
            .map_err(|e| Error::Device(DeviceError::PipeWire(format!("Start error: {}", e))))?
            .response()
            .map_err(|_| Error::Device(DeviceError::PortalDenied))?;

        // Get the streams
        let streams = response.streams();
        if streams.is_empty() {
            return Err(Error::Device(DeviceError::NotFound(
                "No streams returned from portal".into(),
            )));
        }

        let stream = &streams[0];
        let node_id = stream.pipe_wire_node_id();

        // Get the PipeWire remote fd
        let pw_fd = screencast
            .open_pipe_wire_remote(&session)
            .await
            .map_err(|e| {
                Error::Device(DeviceError::PipeWire(format!(
                    "Failed to open PipeWire remote: {}",
                    e
                )))
            })?;

        // Get stream dimensions (if available)
        let (width, height) = stream.size().unwrap_or((1920, 1080));

        let info = ScreenCaptureInfo {
            width: width as u32,
            height: height as u32,
            node_id,
        };

        // Start the capture thread
        let (frame_tx, frame_rx) = bounded::<CapturedFrame>(8);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);

        let capture_node_id = node_id;
        let capture_thread = thread::spawn(move || {
            Self::capture_thread(pw_fd, capture_node_id, frame_tx, shutdown_rx);
        });

        self.frame_receiver = Some(frame_rx);
        self.shutdown_sender = Some(shutdown_tx);
        self.capture_thread = Some(capture_thread);
        self.info = Some(info.clone());
        self.initialized = true;

        Ok(info)
    }

    /// The PipeWire capture thread.
    fn capture_thread(
        pw_fd: OwnedFd,
        node_id: u32,
        frame_tx: Sender<CapturedFrame>,
        shutdown_rx: Receiver<()>,
    ) {
        pw::init();

        let main_loop = match pw::main_loop::MainLoop::new(None) {
            Ok(ml) => ml,
            Err(e) => {
                tracing::error!("Failed to create PipeWire main loop: {}", e);
                return;
            }
        };

        let context = match pw::context::Context::new(&main_loop) {
            Ok(ctx) => ctx,
            Err(e) => {
                tracing::error!("Failed to create PipeWire context: {}", e);
                return;
            }
        };

        // Connect using the fd from the portal
        let core = match context.connect_fd(pw_fd, None) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!("Failed to connect to PipeWire via fd: {}", e);
                return;
            }
        };

        let props = pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        };

        let stream = match pw::stream::Stream::new(&core, "parallax-screen-capture", props) {
            Ok(s) => s,
            Err(e) => {
                tracing::error!("Failed to create PipeWire stream: {}", e);
                return;
            }
        };

        let frame_tx_clone = frame_tx.clone();

        let _listener = stream
            .add_local_listener_with_user_data(())
            .process(move |stream, _| {
                if let Some(mut pw_buffer) = stream.dequeue_buffer() {
                    if let Some(data) = pw_buffer.datas_mut().first_mut() {
                        let chunk = data.chunk();
                        let size = chunk.size() as usize;
                        let stride = chunk.stride() as u32;

                        if size > 0 {
                            if let Some(slice) = data.data() {
                                let bytes = slice[..size].to_vec();

                                // Try to determine dimensions from stride and size
                                // Assuming BGRA (4 bytes per pixel)
                                let width = if stride > 0 { stride / 4 } else { 0 };
                                let height = if width > 0 && size > 0 {
                                    (size as u32) / stride
                                } else {
                                    0
                                };

                                let frame = CapturedFrame {
                                    data: bytes,
                                    width,
                                    height,
                                    stride,
                                    format: PixelFormat::Bgra,
                                };

                                let _ = frame_tx_clone.try_send(frame);
                            }
                        }
                    }
                }
            })
            .register();

        // Connect to the specific node from the portal
        let params: &mut [&pw::spa::pod::Pod] = &mut [];

        if let Err(e) = stream.connect(
            pw::spa::utils::Direction::Input,
            Some(node_id),
            pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
            params,
        ) {
            tracing::error!(
                "Failed to connect PipeWire stream to node {}: {:?}",
                node_id,
                e
            );
            return;
        }

        tracing::info!("Screen capture started, connected to node {}", node_id);

        // Run main loop until shutdown
        loop {
            if shutdown_rx.try_recv().is_ok() {
                tracing::debug!("Screen capture shutdown requested");
                break;
            }

            main_loop
                .loop_()
                .iterate(std::time::Duration::from_millis(10));
        }

        tracing::debug!("Screen capture thread exiting");
    }

    /// Get the capture info (available after initialization).
    pub fn info(&self) -> Option<&ScreenCaptureInfo> {
        self.info.as_ref()
    }

    /// Check if the session has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Set the arena for output buffers.
    pub fn set_arena(&mut self, arena: SharedArena) {
        self.arena = Some(arena);
    }

    /// Try to receive a frame without blocking.
    pub fn try_recv_frame(&mut self) -> Option<CapturedFrame> {
        self.frame_receiver
            .as_ref()
            .and_then(|rx| rx.try_recv().ok().flatten())
    }

    /// Receive a frame, blocking until one is available.
    pub fn recv_frame(&mut self) -> Option<CapturedFrame> {
        self.frame_receiver.as_ref().and_then(|rx| rx.recv().ok())
    }

    /// Receive a frame with a timeout.
    pub fn recv_frame_timeout(&mut self, timeout: std::time::Duration) -> Option<CapturedFrame> {
        self.frame_receiver
            .as_ref()
            .and_then(|rx| rx.recv_timeout(timeout).ok())
    }
}

impl Source for ScreenCaptureSrc {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        // Try to receive a frame
        let frame = match self.try_recv_frame() {
            Some(f) => f,
            None => return Ok(ProduceResult::WouldBlock),
        };

        // Get arena for output
        let arena = self.arena.as_ref().ok_or_else(|| {
            Error::InvalidSegment("Screen capture not initialized with arena".into())
        })?;

        let frame_size = frame.data.len();
        let mut slot = arena.acquire().ok_or_else(|| {
            Error::AllocationFailed(format!(
                "No slots available in arena (need {} bytes)",
                frame_size
            ))
        })?;

        if slot.len() < frame_size {
            return Err(Error::AllocationFailed(format!(
                "Arena slot too small: {} < {}",
                slot.len(),
                frame_size
            )));
        }

        // Copy frame data
        slot.data_mut()[..frame_size].copy_from_slice(&frame.data);

        let handle = crate::buffer::MemoryHandle::with_len(slot, frame_size);
        let mut metadata = Metadata::new();
        metadata.set("video/width", frame.width);
        metadata.set("video/height", frame.height);
        metadata.set("video/stride", frame.stride);
        metadata.set("video/format", format!("{:?}", frame.format));

        let buffer = Buffer::new(handle, metadata);
        Ok(ProduceResult::OwnBuffer(buffer))
    }

    fn output_media_caps(&self) -> ElementMediaCaps {
        // Get dimensions from info if available, otherwise use defaults
        let (width, height) = self
            .info
            .as_ref()
            .map(|i| (i.width, i.height))
            .unwrap_or((1920, 1080));

        let format = VideoFormatCaps {
            width: CapsValue::Fixed(width),
            height: CapsValue::Fixed(height),
            pixel_format: CapsValue::List(vec![
                PixelFormat::Bgra,
                PixelFormat::Rgba,
                PixelFormat::Rgb24,
            ]),
            framerate: CapsValue::Any,
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            format.into(),
            MemoryCaps::cpu_only(),
        )])
    }

    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    fn is_rt_safe(&self) -> bool {
        false
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

impl Drop for ScreenCaptureSrc {
    fn drop(&mut self) {
        // Signal the capture thread to stop
        if let Some(tx) = self.shutdown_sender.take() {
            let _ = tx.send(());
        }

        // Wait for the thread to finish
        if let Some(thread) = self.capture_thread.take() {
            let _ = thread.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = ScreenCaptureConfig::default();
        assert_eq!(config.source_type, CaptureSourceType::Monitor);
        assert!(config.show_cursor);
        assert!(!config.persist_session);
    }

    #[test]
    fn test_source_type_conversion() {
        let monitor_flags = CaptureSourceType::Monitor.to_source_type();
        assert!(monitor_flags.contains(SourceType::Monitor));
        assert!(!monitor_flags.contains(SourceType::Window));

        let window_flags = CaptureSourceType::Window.to_source_type();
        assert!(window_flags.contains(SourceType::Window));
        assert!(!window_flags.contains(SourceType::Monitor));

        let any_flags = CaptureSourceType::Any.to_source_type();
        assert!(any_flags.contains(SourceType::Monitor));
        assert!(any_flags.contains(SourceType::Window));
    }

    #[test]
    fn test_create_source() {
        let src = ScreenCaptureSrc::default_config();
        assert!(!src.is_initialized());
        assert!(src.info().is_none());
    }
}
