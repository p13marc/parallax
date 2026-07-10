//! libcamera video capture.
//!
//! libcamera is the modern camera stack for Linux, providing a unified API
//! for complex camera hardware (ISP, 3A algorithms, etc.).
//!
//! ## Features
//!
//! - Unified API for all cameras (USB webcams, MIPI CSI, Raspberry Pi, etc.)
//! - Automatic ISP configuration
//! - 3A algorithms (auto-exposure, auto-white-balance, auto-focus)
//! - DMA-BUF support for zero-copy
//!
//! ## Example
//!
//! ```rust,ignore
//! use parallax::elements::device::libcamera::{LibCameraSrc, LibCameraConfig};
//!
//! // Use default camera with auto configuration
//! let camera = LibCameraSrc::new()?;
//!
//! // Or configure specific format
//! let config = LibCameraConfig {
//!     width: 1920,
//!     height: 1080,
//!     format: PixelFormat::NV12,
//!     buffer_count: 4,
//! };
//! let camera = LibCameraSrc::with_config(config)?;
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, mpsc as std_mpsc};
use std::thread;
use std::time::Duration;

use kanal::{Receiver, Sender, bounded};
use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::{CameraManager, HotplugEvent},
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    properties,
    stream::StreamRole,
};

use crate::element::{AsyncSource, ExecutionHints, ProduceContext, ProduceResult};
use crate::error::Result;
use crate::pipeline::flow::{FlowPolicy, FlowSignal, FlowStateHandle};

use super::{CameraLocation, DeviceError};

/// Process-wide [`CameraManager`] plus the hotplug subscription made at init.
///
/// libcamera enforces a single `CameraManager` per process — constructing a
/// second one while the first is alive is a fatal error in the C++ library
/// ("Multiple CameraManager objects are not allowed"). Everything in this
/// crate therefore shares this one instance, created on first use and kept
/// for the lifetime of the process.
struct SharedCameraManager {
    manager: CameraManager,
    /// Hotplug event receiver, subscribed once at init (the crate replaces
    /// the internal sender on re-subscription, so it must never be called
    /// again). Taken by `DeviceMonitor` and returned when it shuts down.
    #[cfg_attr(not(feature = "hotplug"), allow(dead_code))]
    hotplug_rx: Mutex<Option<std_mpsc::Receiver<HotplugEvent>>>,
}

// SAFETY: `CameraManager` is `!Send`/`!Sync` because it holds raw pointers,
// but the C++ `CameraManager` methods reachable through `&self` on the Rust
// wrapper (`cameras()`, `get()`, `version()`, `is_started()`) are documented
// thread-safe in libcamera (internal locking). All `&mut self` methods
// (`subscribe_hotplug_events`) are called exactly once inside `shared()`
// before the reference is published, and the leaked reference makes further
// `&mut` access impossible. The hotplug callbacks registered by the crate
// run on libcamera's internal thread and only touch the mpsc sender, which
// is what `subscribe_hotplug_events` set up for exactly that purpose.
// `hotplug_rx` is independently synchronized by its `Mutex`.
unsafe impl Send for SharedCameraManager {}
unsafe impl Sync for SharedCameraManager {}

/// Cached init result: the leaked shared manager, or a sticky error string
/// (libcamera has no daemon that could "come up later", so retrying is
/// pointless and would risk the fatal double-instantiation).
static SHARED: OnceLock<std::result::Result<&'static SharedCameraManager, String>> =
    OnceLock::new();

fn shared() -> Result<&'static SharedCameraManager> {
    SHARED
        .get_or_init(|| {
            let mut manager = CameraManager::new().map_err(|e| e.to_string())?;
            let hotplug_rx = manager.subscribe_hotplug_events();
            Ok(&*Box::leak(Box::new(SharedCameraManager {
                manager,
                hotplug_rx: Mutex::new(Some(hotplug_rx)),
            })))
        })
        .clone()
        .map_err(|e| DeviceError::LibCamera(e).into())
}

/// Get the process-wide shared [`CameraManager`].
///
/// The first call creates (and leaks) the manager; a failure to create it is
/// cached and returned on every subsequent call.
pub(crate) fn shared_manager() -> Result<&'static CameraManager> {
    shared().map(|s| &s.manager)
}

/// Take the process-wide hotplug event receiver, draining any stale events.
///
/// Returns `None` if libcamera is unavailable or another holder (a running
/// `DeviceMonitor`) already took it. Hand it back with
/// [`return_hotplug_receiver`] so a later monitor can use it.
#[cfg(feature = "hotplug")]
#[allow(dead_code)] // wired up by DeviceMonitor's libcamera integration
pub(crate) fn take_hotplug_receiver() -> Option<std_mpsc::Receiver<HotplugEvent>> {
    let rx = shared().ok()?.hotplug_rx.lock().unwrap().take();
    if let Some(rx) = &rx {
        // Events that accumulated while nobody was listening are stale.
        while rx.try_recv().is_ok() {}
    }
    rx
}

/// Return the hotplug receiver taken with [`take_hotplug_receiver`].
#[cfg(feature = "hotplug")]
#[allow(dead_code)] // wired up by DeviceMonitor's libcamera integration
pub(crate) fn return_hotplug_receiver(rx: std_mpsc::Receiver<HotplugEvent>) {
    if let Ok(s) = shared() {
        *s.hotplug_rx.lock().unwrap() = Some(rx);
    }
}

/// Check if libcamera is available on this system.
pub fn is_available() -> bool {
    shared_manager()
        .map(|m| !m.cameras().is_empty())
        .unwrap_or(false)
}

/// Information about a libcamera camera.
#[derive(Debug, Clone)]
pub struct LibCameraInfo {
    /// Camera ID (unique identifier).
    pub id: String,
    /// Camera model name.
    pub model: String,
    /// Camera physical location.
    pub location: CameraLocation,
}

/// Enumerate cameras available via libcamera.
pub fn enumerate_cameras() -> Result<Vec<LibCameraInfo>> {
    let cm = shared_manager()?;

    let camera_list = cm.cameras();
    let mut cameras = Vec::new();

    for i in 0..camera_list.len() {
        if let Some(camera) = camera_list.get(i) {
            let id = camera.id().to_string();

            // Get model from properties - use id as fallback
            let model = camera
                .properties()
                .get::<properties::Model>()
                .map(|m| m.to_string())
                .unwrap_or_else(|_| id.clone());

            // Get location from properties
            // NOTE: The libcamera properties API varies by version.
            // Default to External for compatibility.
            let location = CameraLocation::External;

            cameras.push(LibCameraInfo {
                id,
                model,
                location,
            });
        }
    }

    Ok(cameras)
}

/// libcamera capture configuration.
#[derive(Debug, Clone)]
pub struct LibCameraConfig {
    /// Desired width (0 for auto).
    pub width: u32,
    /// Desired height (0 for auto).
    pub height: u32,
    /// Pixel format (None for auto).
    pub format: Option<PixelFormat>,
    /// Number of buffers to allocate.
    pub buffer_count: usize,
}

impl Default for LibCameraConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            format: None,
            buffer_count: 4,
        }
    }
}

/// Captured frame from libcamera.
#[allow(dead_code)]
struct CapturedFrame {
    /// Frame data.
    data: Vec<u8>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Timestamp in microseconds.
    timestamp_us: i64,
}

/// libcamera video capture source.
pub struct LibCameraSrc {
    /// Receiver for captured frames.
    receiver: Receiver<CapturedFrame>,
    /// Sender to request shutdown.
    shutdown: Sender<()>,
    /// Thread handle.
    thread: Option<thread::JoinHandle<()>>,
    /// Configuration used.
    config: LibCameraConfig,
    /// Camera ID being used.
    camera_id: String,
    /// Flow state handle for downstream backpressure monitoring.
    flow_state: Option<FlowStateHandle>,
    /// Frames dropped due to backpressure.
    frames_dropped: u64,
}

impl LibCameraSrc {
    /// Create a capture source using the default camera.
    pub fn new() -> Result<Self> {
        Self::with_config(LibCameraConfig::default())
    }

    /// Create a capture source with specific configuration.
    pub fn with_config(config: LibCameraConfig) -> Result<Self> {
        let cameras = enumerate_cameras()?;
        if cameras.is_empty() {
            return Err(DeviceError::NotFound("No cameras available".into()).into());
        }

        Self::with_camera_and_config(&cameras[0].id, config)
    }

    /// Create a capture source for a specific camera.
    pub fn with_camera(camera_id: &str) -> Result<Self> {
        Self::with_camera_and_config(camera_id, LibCameraConfig::default())
    }

    /// Create a capture source for a specific camera with configuration.
    pub fn with_camera_and_config(camera_id: &str, config: LibCameraConfig) -> Result<Self> {
        let (frame_tx, frame_rx) = bounded::<CapturedFrame>(config.buffer_count);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);

        let camera_id_owned = camera_id.to_string();
        let config_clone = config.clone();

        let thread = thread::spawn(move || {
            if let Err(e) =
                Self::capture_thread(camera_id_owned, config_clone, frame_tx, shutdown_rx)
            {
                tracing::error!("libcamera capture thread error: {}", e);
            }
        });

        Ok(Self {
            receiver: frame_rx,
            shutdown: shutdown_tx,
            thread: Some(thread),
            config,
            camera_id: camera_id.to_string(),
            flow_state: None,
            frames_dropped: 0,
        })
    }

    /// Main capture thread.
    fn capture_thread(
        camera_id: String,
        config: LibCameraConfig,
        _frame_tx: Sender<CapturedFrame>,
        shutdown_rx: Receiver<()>,
    ) -> Result<()> {
        // Look up the camera via the process-wide shared manager
        let cm = shared_manager()?;
        let camera = cm
            .get(&camera_id)
            .ok_or_else(|| DeviceError::NotFound(camera_id.clone()))?;

        // Acquire camera
        let mut camera = camera
            .acquire()
            .map_err(|e| DeviceError::LibCamera(e.to_string()))?;

        // Generate configuration
        let mut cam_config = camera
            .generate_configuration(&[StreamRole::VideoRecording])
            .ok_or_else(|| {
                DeviceError::LibCamera("Failed to generate configuration".to_string())
            })?;

        // Modify configuration if requested
        if let Some(mut stream_config) = cam_config.get_mut(0) {
            if config.width > 0 && config.height > 0 {
                stream_config.set_size(libcamera::geometry::Size {
                    width: config.width,
                    height: config.height,
                });
            }
            if let Some(format) = config.format {
                stream_config.set_pixel_format(format);
            }
        }

        // Validate and apply configuration
        match cam_config.validate() {
            CameraConfigurationStatus::Valid => {}
            CameraConfigurationStatus::Adjusted => {
                tracing::warn!("Camera configuration was adjusted");
            }
            CameraConfigurationStatus::Invalid => {
                return Err(DeviceError::FormatNotSupported("Invalid configuration".into()).into());
            }
        }

        camera
            .configure(&mut cam_config)
            .map_err(|e| DeviceError::LibCamera(e.to_string()))?;

        // Get stream and allocate buffers
        let stream = cam_config.get(0).unwrap().stream().unwrap();
        let stream_config = cam_config.get(0).unwrap();
        let _width = stream_config.get_size().width;
        let _height = stream_config.get_size().height;

        let mut allocator = FrameBufferAllocator::new(&camera);
        let buffers = allocator
            .alloc(&stream)
            .map_err(|e| DeviceError::LibCamera(e.to_string()))?;

        // Create memory-mapped buffers
        let buffers: Vec<MemoryMappedFrameBuffer<FrameBuffer>> = buffers
            .into_iter()
            .map(|buf| MemoryMappedFrameBuffer::new(buf).unwrap())
            .collect();

        // Create requests
        let requests: Vec<_> = buffers
            .into_iter()
            .enumerate()
            .map(|(i, buf)| {
                let mut request = camera.create_request(Some(i as u64)).unwrap();
                request.add_buffer(&stream, buf).unwrap();
                request
            })
            .collect();

        // Start camera before queueing so completed requests flow immediately
        camera
            .start(None)
            .map_err(|e| DeviceError::LibCamera(e.to_string()))?;

        // Queue all requests (queue_request hands the request back on failure)
        for request in requests {
            camera
                .queue_request(request)
                .map_err(|(_, e)| DeviceError::LibCamera(e.to_string()))?;
        }

        let running = Arc::new(AtomicBool::new(true));

        // Main capture loop
        while running.load(Ordering::SeqCst) {
            // Check for shutdown
            if shutdown_rx.try_recv().is_ok() {
                running.store(false, Ordering::SeqCst);
                break;
            }

            // Wait for and process completed requests
            // Note: In a real implementation, we'd use camera.poll() or similar
            // For now, we simulate with a small sleep
            thread::sleep(Duration::from_millis(1));

            // Process any completed requests
            // This is a simplified version - real implementation would use callbacks
        }

        // Stop camera
        let _ = camera.stop();

        Ok(())
    }

    /// Get the camera ID being used.
    pub fn camera_id(&self) -> &str {
        &self.camera_id
    }

    /// Get the configuration being used.
    pub fn config(&self) -> &LibCameraConfig {
        &self.config
    }

    /// Set the flow state handle for downstream backpressure monitoring.
    ///
    /// When set, the source will check this handle before producing frames.
    /// If downstream signals backpressure (Busy), frames will be dropped
    /// to prevent lag buildup.
    pub fn set_flow_state(&mut self, handle: FlowStateHandle) {
        self.flow_state = Some(handle);
    }

    /// Get the number of frames dropped due to backpressure.
    pub fn frames_dropped(&self) -> u64 {
        self.frames_dropped
    }
}

impl Drop for LibCameraSrc {
    fn drop(&mut self) {
        // Signal shutdown
        let _ = self.shutdown.send(());

        // Wait for thread to finish
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl AsyncSource for LibCameraSrc {
    async fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        // Check for downstream backpressure before receiving
        // libcamera is a live source - dropping frames is better than accumulating lag
        if let Some(ref flow_state) = self.flow_state {
            if !flow_state.should_produce() {
                // Drop this frame due to backpressure
                self.frames_dropped += 1;
                flow_state.record_drop();

                if self.frames_dropped == 1 || self.frames_dropped % 30 == 0 {
                    tracing::warn!(
                        "libcamera: dropping frame due to backpressure (total dropped: {})",
                        self.frames_dropped
                    );
                }

                // Drain one frame from the receiver to keep the capture thread running
                let _ = self.receiver.as_async().recv().await;

                return Ok(ProduceResult::WouldBlock);
            }
        }

        match self.receiver.as_async().recv().await {
            Ok(frame) => {
                let len = frame.data.len();
                if len > 0 && len <= ctx.output().len() {
                    ctx.output()[..len].copy_from_slice(&frame.data);
                    // NOTE: Metadata (timestamp, width, height) should be set
                    // via ProduceContext when buffer metadata API is extended.
                    Ok(ProduceResult::Produced(len))
                } else if len > ctx.output().len() {
                    // Buffer too small - request larger buffer
                    tracing::warn!(
                        "libcamera frame ({} bytes) exceeds output buffer ({} bytes)",
                        len,
                        ctx.output().len()
                    );
                    Ok(ProduceResult::WouldBlock)
                } else {
                    Ok(ProduceResult::WouldBlock)
                }
            }
            Err(_) => Ok(ProduceResult::Eos),
        }
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        // Estimate based on config or default to 1080p
        let width = if self.config.width > 0 {
            self.config.width
        } else {
            1920
        };
        let height = if self.config.height > 0 {
            self.config.height
        } else {
            1080
        };
        // Assume worst case (RGB24)
        Some((width * height * 3) as usize)
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
        // libcamera is a live source - always use Drop policy to prevent lag
        FlowPolicy::Drop {
            log_drops: true,
            max_consecutive: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        let available = is_available();
        println!("libcamera available: {}", available);
    }

    #[test]
    fn test_enumerate_cameras() {
        match enumerate_cameras() {
            Ok(cameras) => {
                println!("Found {} cameras:", cameras.len());
                for camera in &cameras {
                    println!("  {} - {} ({:?})", camera.id, camera.model, camera.location);
                }
            }
            Err(e) => {
                println!("Failed to enumerate cameras: {}", e);
            }
        }
    }
}
