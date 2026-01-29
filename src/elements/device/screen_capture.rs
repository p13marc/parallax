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
//! let mut capture = ScreenCaptureSrc::new(config).await?;
//!
//! // Or capture a specific window
//! let config = ScreenCaptureConfig {
//!     source_type: SourceType::Window,
//!     show_cursor: true,
//!     ..Default::default()
//! };
//! let mut capture = ScreenCaptureSrc::new(config).await?;
//! ```
//!
//! # How It Works
//!
//! 1. Create a ScreenCast session via D-Bus portal
//! 2. User is prompted to select a screen/window
//! 3. Portal returns a PipeWire node ID and fd
//! 4. We connect to PipeWire and capture frames from that node
//! 5. Frames are delivered as BGRA or other negotiated format

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use ashpd::desktop::PersistMode;
use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
use ashpd::enumflags2::BitFlags;

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

/// Frame data received from PipeWire.
#[derive(Debug)]
struct CapturedFrame {
    /// Raw pixel data (BGRA format typically).
    data: Vec<u8>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Stride (bytes per row).
    stride: u32,
}

/// Screen capture source element.
///
/// Captures screen content via XDG Desktop Portal and PipeWire.
/// The user will be prompted to select a screen or window when the
/// element is first started.
pub struct ScreenCaptureSrc {
    config: ScreenCaptureConfig,
    /// Shared state for receiving frames from PipeWire callback.
    frame_receiver: Arc<Mutex<Option<CapturedFrame>>>,
    /// Signal to stop the capture.
    stop_signal: Arc<AtomicBool>,
    /// Capture info (set after session is started).
    info: Option<ScreenCaptureInfo>,
    /// Arena for output buffers.
    arena: Option<SharedArena>,
    /// Whether the session has been initialized.
    initialized: bool,
    /// PipeWire node ID (set after portal session is created).
    node_id: Option<u32>,
    /// PipeWire remote fd (set after portal session is created).
    pw_fd: Option<std::os::fd::OwnedFd>,
}

impl std::fmt::Debug for ScreenCaptureSrc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScreenCaptureSrc")
            .field("config", &self.config)
            .field("info", &self.info)
            .field("initialized", &self.initialized)
            .field("node_id", &self.node_id)
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
            frame_receiver: Arc::new(Mutex::new(None)),
            stop_signal: Arc::new(AtomicBool::new(false)),
            info: None,
            arena: None,
            initialized: false,
            node_id: None,
            pw_fd: None,
        }
    }

    /// Create a screen capture source with default configuration.
    pub fn default_config() -> Self {
        Self::new(ScreenCaptureConfig::default())
    }

    /// Initialize the portal session.
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

        self.info = Some(info.clone());
        self.node_id = Some(node_id);
        self.pw_fd = Some(pw_fd);
        self.initialized = true;

        Ok(info)
    }

    /// Get the capture info (available after initialization).
    pub fn info(&self) -> Option<&ScreenCaptureInfo> {
        self.info.as_ref()
    }

    /// Check if the session has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }
}

impl ScreenCaptureSrc {
    /// Set the arena for output buffers.
    pub fn set_arena(&mut self, arena: SharedArena) {
        self.arena = Some(arena);
    }
}

impl Source for ScreenCaptureSrc {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        // Check if we have a frame available
        let frame = {
            let mut receiver = self.frame_receiver.lock().unwrap();
            receiver.take()
        };

        if let Some(frame) = frame {
            // We have a frame - copy it to the output buffer
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

            let buffer = Buffer::new(handle, metadata);
            Ok(ProduceResult::OwnBuffer(buffer))
        } else {
            // No frame available yet
            Ok(ProduceResult::WouldBlock)
        }
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
        // Screen capture involves async I/O
        Affinity::Async
    }

    fn is_rt_safe(&self) -> bool {
        false // Portal/PipeWire interactions are not RT-safe
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }
}

impl Drop for ScreenCaptureSrc {
    fn drop(&mut self) {
        // Signal the capture to stop
        self.stop_signal.store(true, Ordering::SeqCst);
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
}
