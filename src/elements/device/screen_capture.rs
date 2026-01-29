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
//! // Receive frames
//! while let Some(frame) = capture.recv_frame_timeout(Duration::from_millis(100)) {
//!     println!("Got frame: {}x{}", frame.width, frame.height);
//! }
//! ```

use std::os::fd::OwnedFd;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;

use ashpd::desktop::PersistMode;
use ashpd::desktop::screencast::{CursorMode, Screencast, SourceType};
use ashpd::enumflags2::BitFlags;
use kanal::{Receiver, Sender, bounded};
use pipewire as pw;
use pw::spa;

use crate::buffer::Buffer;
use crate::element::{Affinity, ExecutionHints, ProduceContext, ProduceResult, Source};
use crate::error::{Error, Result};
use crate::format::{
    CapsValue, ElementMediaCaps, FormatMemoryCap, MemoryCaps, PixelFormat, VideoFormatCaps,
};
use crate::memory::SharedArena;
use crate::metadata::Metadata;
use crate::pipeline::flow::{FlowPolicy, FlowSignal, FlowStateHandle};

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
    pub persist_session: bool,
    /// Maximum number of frames to capture before EOS.
    /// None means unlimited (capture until manually stopped).
    pub max_frames: Option<u32>,
    /// Flow policy for handling downstream backpressure.
    /// Default: Drop frames when downstream is busy (prevents capture lag).
    pub flow_policy: FlowPolicy,
}

impl Default for ScreenCaptureConfig {
    fn default() -> Self {
        Self {
            source_type: CaptureSourceType::Monitor,
            show_cursor: true,
            persist_session: false,
            max_frames: None,
            // Screen capture should drop frames when downstream is busy
            // to prevent building up lag. This is the standard behavior
            // for live video sources.
            flow_policy: FlowPolicy::drop_with_logging(),
        }
    }
}

impl ScreenCaptureConfig {
    /// Set the maximum number of frames to capture.
    pub fn with_max_frames(mut self, max_frames: u32) -> Self {
        self.max_frames = Some(max_frames);
        self
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
    /// Pixel format.
    pub format: PixelFormat,
}

/// Screen capture source element.
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
    /// Frame counter for PipeWire thread (debugging).
    frame_count: Arc<AtomicU32>,
    /// Frames output by produce() - used for max_frames limit.
    frames_produced: u32,
    /// Flow state handle for downstream backpressure monitoring.
    flow_state: Option<FlowStateHandle>,
    /// Frames dropped due to backpressure.
    frames_dropped: u64,
}

impl std::fmt::Debug for ScreenCaptureSrc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScreenCaptureSrc")
            .field("config", &self.config)
            .field("info", &self.info)
            .field("initialized", &self.initialized)
            .field("frame_count", &self.frame_count.load(Ordering::Relaxed))
            .field("frames_produced", &self.frames_produced)
            .field("frames_dropped", &self.frames_dropped)
            .finish()
    }
}

impl ScreenCaptureSrc {
    /// Create a new screen capture source with the given configuration.
    pub fn new(config: ScreenCaptureConfig) -> Self {
        Self {
            config,
            frame_receiver: None,
            shutdown_sender: None,
            capture_thread: None,
            info: None,
            arena: None,
            initialized: false,
            frame_count: Arc::new(AtomicU32::new(0)),
            frames_produced: 0,
            flow_state: None,
            frames_dropped: 0,
        }
    }

    /// Create a screen capture source with default configuration.
    pub fn default_config() -> Self {
        Self::new(ScreenCaptureConfig::default())
    }

    /// Initialize the portal session and start capturing.
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
        let (frame_tx, frame_rx) = bounded::<CapturedFrame>(16);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);

        let capture_node_id = node_id;
        let frame_count = self.frame_count.clone();

        let capture_thread = thread::spawn(move || {
            if let Err(e) =
                Self::capture_thread(pw_fd, capture_node_id, frame_tx, shutdown_rx, frame_count)
            {
                tracing::error!("Screen capture thread error: {}", e);
            }
        });

        self.frame_receiver = Some(frame_rx);
        self.shutdown_sender = Some(shutdown_tx);
        self.capture_thread = Some(capture_thread);
        self.info = Some(info.clone());
        self.initialized = true;

        tracing::info!(
            "Screen capture initialized: {}x{}, node_id={}",
            info.width,
            info.height,
            info.node_id
        );

        Ok(info)
    }

    /// The PipeWire capture thread.
    fn capture_thread(
        pw_fd: OwnedFd,
        node_id: u32,
        frame_tx: Sender<CapturedFrame>,
        shutdown_rx: Receiver<()>,
        frame_count: Arc<AtomicU32>,
    ) -> std::result::Result<(), String> {
        pw::init();

        // Use ThreadLoop instead of MainLoop - it runs in its own thread
        // and is what OBS and other applications use for portal screen capture
        let thread_loop =
            unsafe { pw::thread_loop::ThreadLoop::new(Some("parallax-capture"), None) }
                .map_err(|e| format!("Failed to create thread loop: {}", e))?;

        // Create context with the thread loop
        let context = pw::context::Context::new(&thread_loop)
            .map_err(|e| format!("Failed to create context: {}", e))?;

        // Start the thread loop BEFORE connecting (following OBS pattern)
        thread_loop.start();
        tracing::debug!("Thread loop started");

        // Lock the thread loop for all PipeWire operations
        let _lock = thread_loop.lock();
        tracing::debug!("Thread loop locked");

        // Connect using the fd from the portal (with lock held)
        let core = context
            .connect_fd(pw_fd, None)
            .map_err(|e| format!("Failed to connect via fd: {}", e))?;
        tracing::debug!("Connected to core via fd");

        let stream_running = Arc::new(AtomicBool::new(false));
        let stream_running_clone = stream_running.clone();

        let props = pw::properties::properties! {
            *pw::keys::MEDIA_TYPE => "Video",
            *pw::keys::MEDIA_CATEGORY => "Capture",
            *pw::keys::MEDIA_ROLE => "Screen",
        };

        let stream = pw::stream::Stream::new(&core, "parallax-screen-capture", props)
            .map_err(|e| format!("Failed to create stream: {}", e))?;
        tracing::debug!("Stream created");

        // Shared state for video format info
        let video_width = Arc::new(AtomicU32::new(0));
        let video_height = Arc::new(AtomicU32::new(0));
        let video_width_process = video_width.clone();
        let video_height_process = video_height.clone();

        let frame_tx_clone = frame_tx.clone();
        let frame_count_clone = frame_count.clone();

        let _listener = stream
            .add_local_listener_with_user_data(())
            .state_changed(move |_stream, _user_data, old, new| {
                tracing::debug!("Stream state changed: {:?} -> {:?}", old, new);
                if new == pw::stream::StreamState::Streaming {
                    stream_running_clone.store(true, Ordering::SeqCst);
                    tracing::info!("Screen capture stream is now streaming");
                }
            })
            .param_changed(move |_stream, _user_data, id, param| {
                let Some(param) = param else { return };

                if id != spa::param::ParamType::Format.as_raw() {
                    return;
                }

                // Try to parse video format
                if let Ok((media_type, media_subtype)) =
                    spa::param::format_utils::parse_format(param)
                {
                    tracing::info!(
                        "Format: media_type={:?}, media_subtype={:?}",
                        media_type,
                        media_subtype
                    );
                }

                // Parse video info to get dimensions
                let mut video_info = spa::param::video::VideoInfoRaw::default();
                if video_info.parse(param).is_ok() {
                    let size = video_info.size();
                    video_width.store(size.width, Ordering::SeqCst);
                    video_height.store(size.height, Ordering::SeqCst);
                    tracing::info!(
                        "Video format: {}x{}, format={:?}",
                        size.width,
                        size.height,
                        video_info.format()
                    );
                }
            })
            .process(move |stream, _user_data| {
                tracing::trace!("Process callback entered");

                let Some(mut buffer) = stream.dequeue_buffer() else {
                    tracing::trace!("No buffer available to dequeue");
                    return;
                };

                let datas = buffer.datas_mut();
                if datas.is_empty() {
                    tracing::trace!("Buffer has no data planes");
                    return;
                }

                let data = &mut datas[0];
                let chunk = data.chunk();
                let size = chunk.size() as usize;
                let stride = chunk.stride() as u32;

                if size == 0 {
                    tracing::trace!("Buffer chunk size is 0");
                    return;
                }

                let Some(slice) = data.data() else {
                    return;
                };

                // Get dimensions from parsed format or calculate from stride
                let width = video_width_process.load(Ordering::SeqCst);
                let height = video_height_process.load(Ordering::SeqCst);

                // If we don't have dimensions from format, estimate from stride
                let (width, height) = if width > 0 && height > 0 {
                    (width, height)
                } else if stride > 0 {
                    // Assume BGRA (4 bytes per pixel)
                    let w = stride / 4;
                    let h = if w > 0 { size as u32 / stride } else { 0 };
                    (w, h)
                } else {
                    (0, 0)
                };

                if width == 0 || height == 0 {
                    tracing::warn!(
                        "Got frame with unknown dimensions, size={}, stride={}",
                        size,
                        stride
                    );
                    return;
                }

                let bytes = slice[..size].to_vec();
                let frame = CapturedFrame {
                    data: bytes,
                    width,
                    height,
                    stride,
                    format: PixelFormat::Bgra, // Most common for screen capture
                };

                let count = frame_count_clone.fetch_add(1, Ordering::Relaxed) + 1;

                if count <= 5 || count % 30 == 0 {
                    tracing::debug!(
                        "Captured frame {}: {}x{}, {} bytes",
                        count,
                        width,
                        height,
                        frame.data.len()
                    );
                }

                if let Err(e) = frame_tx_clone.try_send(frame) {
                    tracing::warn!("Frame channel full or closed: {}", e);
                }
            })
            .register()
            .map_err(|e| format!("Failed to register listener: {}", e))?;

        // Build format params - specify what video formats we accept
        let format_obj = spa::pod::object!(
            spa::utils::SpaTypes::ObjectParamFormat,
            spa::param::ParamType::EnumFormat,
            spa::pod::property!(
                spa::param::format::FormatProperties::MediaType,
                Id,
                spa::param::format::MediaType::Video
            ),
            spa::pod::property!(
                spa::param::format::FormatProperties::MediaSubtype,
                Id,
                spa::param::format::MediaSubtype::Raw
            ),
            spa::pod::property!(
                spa::param::format::FormatProperties::VideoFormat,
                Choice,
                Enum,
                Id,
                spa::param::video::VideoFormat::BGRx,
                spa::param::video::VideoFormat::BGRx,
                spa::param::video::VideoFormat::BGRA,
                spa::param::video::VideoFormat::RGBx,
                spa::param::video::VideoFormat::RGBA,
                spa::param::video::VideoFormat::RGB,
            ),
            spa::pod::property!(
                spa::param::format::FormatProperties::VideoSize,
                Choice,
                Range,
                Rectangle,
                spa::utils::Rectangle {
                    width: 1920,
                    height: 1080
                },
                spa::utils::Rectangle {
                    width: 1,
                    height: 1
                },
                spa::utils::Rectangle {
                    width: 4096,
                    height: 4096
                }
            ),
            spa::pod::property!(
                spa::param::format::FormatProperties::VideoFramerate,
                Choice,
                Range,
                Fraction,
                spa::utils::Fraction { num: 30, denom: 1 },
                spa::utils::Fraction { num: 0, denom: 1 },
                spa::utils::Fraction { num: 120, denom: 1 }
            ),
        );

        let format_bytes: Vec<u8> = spa::pod::serialize::PodSerializer::serialize(
            std::io::Cursor::new(Vec::new()),
            &spa::pod::Value::Object(format_obj),
        )
        .map_err(|e| format!("Failed to serialize format params: {:?}", e))?
        .0
        .into_inner();

        let format_pod = spa::pod::Pod::from_bytes(&format_bytes)
            .ok_or_else(|| "Failed to create format pod".to_string())?;

        let mut params = [format_pod];
        tracing::debug!("Built format params, connecting stream...");

        // Connect to the portal's screencast node
        stream
            .connect(
                spa::utils::Direction::Input,
                Some(node_id),
                pw::stream::StreamFlags::AUTOCONNECT | pw::stream::StreamFlags::MAP_BUFFERS,
                &mut params,
            )
            .map_err(|e| format!("Failed to connect stream to node {}: {:?}", node_id, e))?;

        tracing::info!("Screen capture stream connecting to node {}", node_id);

        // Unlock the thread loop - the PipeWire thread will now process events
        drop(_lock);
        tracing::debug!("Thread loop unlocked, PipeWire processing events");

        // Wait for shutdown signal while the thread loop runs
        loop {
            match shutdown_rx.recv_timeout(std::time::Duration::from_millis(100)) {
                Ok(()) => {
                    tracing::debug!("Screen capture shutdown requested (received signal)");
                    break;
                }
                Err(kanal::ReceiveErrorTimeout::Closed)
                | Err(kanal::ReceiveErrorTimeout::SendClosed) => {
                    tracing::debug!("Screen capture shutdown (channel closed)");
                    break;
                }
                Err(kanal::ReceiveErrorTimeout::Timeout) => {
                    // No shutdown yet, continue
                    let count = frame_count.load(Ordering::Relaxed);
                    if count > 0 && count % 30 == 0 {
                        tracing::debug!("Still capturing, total frames: {}", count);
                    }
                }
            }
        }

        // Stop the thread loop
        thread_loop.stop();

        tracing::info!("PipeWire thread loop stopped");

        tracing::debug!(
            "Screen capture thread exiting, captured {} frames",
            frame_count.load(Ordering::Relaxed)
        );
        Ok(())
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

    /// Get the number of frames captured so far.
    pub fn frame_count(&self) -> u32 {
        self.frame_count.load(Ordering::Relaxed)
    }

    /// Get the number of frames dropped due to backpressure.
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }

    /// Get the number of frames successfully produced.
    pub fn frames_produced(&self) -> u32 {
        self.frames_produced
    }

    /// Set the flow state handle for downstream backpressure monitoring.
    ///
    /// When set, the source will check this handle before producing frames.
    /// If downstream signals backpressure (Busy), frames will be dropped
    /// according to the configured flow policy.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let queue = Queue::new(100).with_flow_control();
    /// let flow_state = queue.flow_state_handle();
    ///
    /// let mut capture = ScreenCaptureSrc::default_config();
    /// capture.set_flow_state(flow_state);
    /// ```
    pub fn set_flow_state(&mut self, handle: FlowStateHandle) {
        self.flow_state = Some(handle);
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
        // Check if we've reached the frame limit (check frames we've actually output)
        if let Some(max) = self.config.max_frames {
            if self.frames_produced >= max {
                tracing::info!("Screen capture: reached max frames ({})", max);
                return Ok(ProduceResult::Eos);
            }
        }

        // Lazy initialization: if not initialized, do it now
        // This uses block_in_place to run the async portal interaction
        if !self.initialized {
            tracing::info!("Screen capture: lazy initialization starting...");
            let info = tokio::task::block_in_place(|| {
                tokio::runtime::Handle::current().block_on(self.initialize())
            })?;
            tracing::info!("Screen capture initialized: {}x{}", info.width, info.height);
            // Note: Arena is created lazily when we see the actual frame size
        }

        let frame = match self.try_recv_frame() {
            Some(f) => f,
            None => return Ok(ProduceResult::WouldBlock),
        };

        // Check for downstream backpressure
        if let Some(ref flow_state) = self.flow_state {
            if !flow_state.should_produce() && self.config.flow_policy.allows_dropping() {
                // Drop this frame due to backpressure
                self.frames_dropped += 1;
                flow_state.record_drop();

                if self.config.flow_policy.should_log_drops() {
                    if self.frames_dropped == 1 || self.frames_dropped % 30 == 0 {
                        tracing::warn!(
                            "Screen capture: dropping frame due to backpressure (total dropped: {})",
                            self.frames_dropped
                        );
                    }
                }

                // Return WouldBlock to signal we didn't produce, but aren't at EOS
                return Ok(ProduceResult::WouldBlock);
            }
        }

        // Ensure we have an arena large enough for this frame
        // The actual frame size may differ from portal-reported dimensions
        let frame_size = frame.data.len();
        if self.arena.is_none()
            || self.arena.as_ref().map(|a| a.slot_size()).unwrap_or(0) < frame_size
        {
            tracing::debug!(
                "Creating arena for frame size: {} bytes ({}x{})",
                frame_size,
                frame.width,
                frame.height
            );
            // Default to 200 slots to buffer ~6 seconds at 30fps
            // This is needed because downstream elements (encoders) may be slower
            // than capture rate. For production, use set_arena() with appropriate size.
            let arena = SharedArena::new(frame_size, 200)
                .map_err(|e| Error::AllocationFailed(format!("Failed to create arena: {}", e)))?;
            self.arena = Some(arena);
        }

        let arena = self.arena.as_ref().unwrap();

        // Reclaim any slots that have been released by downstream elements
        arena.reclaim();

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

        slot.data_mut()[..frame_size].copy_from_slice(&frame.data);

        let handle = crate::buffer::MemoryHandle::with_len(slot, frame_size);
        let mut metadata = Metadata::new();
        metadata.set("video/width", frame.width);
        metadata.set("video/height", frame.height);
        metadata.set("video/stride", frame.stride);
        metadata.set("video/format", format!("{:?}", frame.format));

        // Increment frames produced counter
        self.frames_produced += 1;

        let buffer = Buffer::new(handle, metadata);
        Ok(ProduceResult::OwnBuffer(buffer))
    }

    fn output_media_caps(&self) -> ElementMediaCaps {
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

    fn handle_flow_signal(&mut self, signal: FlowSignal) {
        // Update our internal state based on downstream signal
        if let Some(ref flow_state) = self.flow_state {
            flow_state.set_signal(signal);
        }
    }

    fn flow_policy(&self) -> FlowPolicy {
        self.config.flow_policy.clone()
    }
}

impl Drop for ScreenCaptureSrc {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown_sender.take() {
            let _ = tx.send(());
        }
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
        assert_eq!(src.frame_count(), 0);
    }
}
