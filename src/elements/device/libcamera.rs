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
//! // Or configure format and frame rate
//! let config = LibCameraConfig {
//!     width: 1920,
//!     height: 1080,
//!     framerate: Some((30, 1)),
//!     ..Default::default()
//! };
//! let camera = LibCameraSrc::with_config(config)?;
//! ```

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Mutex, OnceLock, mpsc as std_mpsc};
use std::thread;
use std::time::Duration;

use kanal::{Receiver, Sender, bounded};
use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::{CameraManager, HotplugEvent},
    control::{ControlInfoMap, ControlList},
    controls,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    properties,
    request::{RequestStatus, ReuseFlag},
    stream::StreamRole,
};

use crate::clock::ClockTime;
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
    /// Desired frame rate as frames-per-second numerator/denominator
    /// (e.g. `(30, 1)` for 30 fps, `(30000, 1001)` for 29.97 fps).
    /// `None` keeps the camera default.
    ///
    /// Applied by locking the `FrameDurationLimits` control (min = max =
    /// one frame duration) at start; the requested duration is clamped to
    /// the camera's advertised bounds — read the effective rate back with
    /// [`LibCameraSrc::framerate`]. Best-effort: some pipeline handlers
    /// (notably UVC webcams) do not honor the control, so the true rate is
    /// only observable from buffer PTS deltas.
    pub framerate: Option<(u32, u32)>,
}

impl Default for LibCameraConfig {
    fn default() -> Self {
        Self {
            width: 0,
            height: 0,
            format: None,
            buffer_count: 4,
            framerate: None,
        }
    }
}

/// Convert an fps fraction to a frame duration in microseconds.
fn framerate_to_duration_us(num: u32, den: u32) -> Result<u64> {
    if num == 0 || den == 0 {
        return Err(crate::error::Error::Config(format!(
            "invalid framerate {num}/{den}"
        )));
    }
    Ok((1_000_000u64 * u64::from(den)) / u64::from(num))
}

/// Clamp a requested frame duration (µs) to advertised control bounds.
///
/// `bounds` is `(min, max)` from the camera's `FrameDurationLimits` control
/// info; requests pass through unclamped when the camera advertises no
/// usable bounds.
fn clamp_frame_duration_us(requested_us: u64, bounds: Option<(i64, i64)>) -> u64 {
    match bounds {
        Some((min, max)) if min >= 0 && max >= min => requested_us.clamp(min as u64, max as u64),
        _ => requested_us,
    }
}

/// Read the camera's advertised `FrameDurationLimits` bounds in µs, if any.
fn frame_duration_bounds(controls: &ControlInfoMap) -> Option<(i64, i64)> {
    let info = controls
        .at(controls::ControlId::FrameDurationLimits as u32)
        .ok()?;
    let min = i64::try_from(info.min()).ok()?;
    let max = i64::try_from(info.max()).ok()?;
    Some((min, max))
}

/// Reduce an fps fraction by its greatest common divisor.
fn reduce_fraction(num: u32, den: u32) -> (u32, u32) {
    fn gcd(a: u32, b: u32) -> u32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    let g = gcd(num, den).max(1);
    (num / g, den / g)
}

/// Captured frame from libcamera.
struct CapturedFrame {
    /// Frame data (all planes concatenated, `bytes_used` per plane).
    data: Vec<u8>,
    /// Capture timestamp in nanoseconds (CLOCK_MONOTONIC).
    timestamp_ns: i64,
    /// Frame sequence number from libcamera.
    sequence: u64,
}

/// Stream parameters negotiated by the capture thread, reported through the
/// startup handshake.
#[derive(Debug, Clone, Copy)]
struct NegotiatedInfo {
    /// Actual frame width (drivers may adjust the requested size).
    width: u32,
    /// Actual frame height.
    height: u32,
    /// Frame size in bytes as reported by the stream configuration.
    frame_size: u32,
    /// Frame duration applied via `FrameDurationLimits` (µs), after
    /// clamping to the camera's advertised bounds. `None` when no rate was
    /// requested.
    frame_duration_us: Option<u64>,
}

/// How long the constructor waits for the capture thread to configure and
/// start the camera.
const STARTUP_TIMEOUT: Duration = Duration::from_secs(5);

/// Poll interval of the capture loop for shutdown checks.
const CAPTURE_POLL_INTERVAL: Duration = Duration::from_millis(100);

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
    /// Stream parameters actually negotiated with the camera.
    negotiated: NegotiatedInfo,
    /// Effective frame rate as an fps fraction, when one was requested.
    framerate: Option<(u32, u32)>,
    /// Frame duration hint stamped into buffer metadata.
    frame_duration: ClockTime,
    /// Camera ID being used.
    camera_id: String,
    /// First frame timestamp for relative PTS calculation (ns; `i64::MIN`
    /// until the first frame arrives).
    first_timestamp_ns: AtomicI64,
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
    ///
    /// Blocks until the capture thread has configured and started the camera
    /// (or failed to), so configuration errors surface here instead of a
    /// source that silently never produces.
    pub fn with_camera_and_config(camera_id: &str, config: LibCameraConfig) -> Result<Self> {
        let (frame_tx, frame_rx) = bounded::<CapturedFrame>(config.buffer_count);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
        let (startup_tx, startup_rx) = std_mpsc::channel::<Result<NegotiatedInfo>>();

        let camera_id_owned = camera_id.to_string();
        let config_clone = config.clone();

        let thread = thread::Builder::new()
            .name("parallax-libcamera".to_string())
            .spawn(move || {
                if let Err(e) = Self::capture_thread(
                    camera_id_owned,
                    config_clone,
                    frame_tx,
                    shutdown_rx,
                    &startup_tx,
                ) {
                    tracing::error!("libcamera capture thread error: {}", e);
                    // No-op if startup already succeeded (receiver dropped).
                    let _ = startup_tx.send(Err(e));
                }
            })
            .map_err(|e| DeviceError::LibCamera(format!("failed to spawn thread: {e}")))?;

        let negotiated: NegotiatedInfo = match startup_rx.recv_timeout(STARTUP_TIMEOUT) {
            Ok(Ok(info)) => info,
            Ok(Err(e)) => {
                let _ = thread.join();
                return Err(e);
            }
            Err(_) => {
                // Thread is stuck or died without reporting; tell it to stop
                // and detach. It self-terminates once its startup send fails.
                let _ = shutdown_tx.send(());
                return Err(DeviceError::LibCamera(format!(
                    "camera {camera_id} did not start within {STARTUP_TIMEOUT:?}"
                ))
                .into());
            }
        };

        // Report the effective rate: echo the requested fraction when it
        // survived clamping unchanged, otherwise derive one from the
        // clamped duration.
        let framerate = negotiated
            .frame_duration_us
            .map(|us| match config.framerate {
                Some((num, den)) if framerate_to_duration_us(num, den).ok() == Some(us) => {
                    (num, den)
                }
                _ => reduce_fraction(1_000_000, us.min(u64::from(u32::MAX)) as u32),
            });
        let frame_duration = negotiated
            .frame_duration_us
            .map(|us| ClockTime::from_nanos(us * 1000))
            .unwrap_or(ClockTime::ZERO);

        Ok(Self {
            receiver: frame_rx,
            shutdown: shutdown_tx,
            thread: Some(thread),
            config,
            negotiated,
            framerate,
            frame_duration,
            camera_id: camera_id.to_string(),
            first_timestamp_ns: AtomicI64::new(i64::MIN),
            flow_state: None,
            frames_dropped: 0,
        })
    }

    /// Main capture thread.
    ///
    /// Reports startup success (with the negotiated stream parameters) or
    /// failure through `startup_tx`, then loops shipping completed frames to
    /// `frame_tx` until shut down or the consumer goes away.
    fn capture_thread(
        camera_id: String,
        config: LibCameraConfig,
        frame_tx: Sender<CapturedFrame>,
        shutdown_rx: Receiver<()>,
        startup_tx: &std_mpsc::Sender<Result<NegotiatedInfo>>,
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
            stream_config.set_buffer_count(config.buffer_count as u32);
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

        // Frame rate: lock FrameDurationLimits to a single duration, clamped
        // to the camera's advertised bounds (libcamera has no read-back of
        // the applied rate, so the clamped request is what we report).
        let frame_duration_us = match config.framerate {
            Some((num, den)) => {
                let requested = framerate_to_duration_us(num, den)?;
                let clamped =
                    clamp_frame_duration_us(requested, frame_duration_bounds(camera.controls()));
                if clamped != requested {
                    tracing::warn!(
                        "libcamera: requested frame duration {}µs clamped to {}µs",
                        requested,
                        clamped
                    );
                }
                Some(clamped)
            }
            None => None,
        };

        // Get stream and the actually-applied parameters
        let stream = cam_config.get(0).unwrap().stream().unwrap();
        let stream_config = cam_config.get(0).unwrap();
        let negotiated = NegotiatedInfo {
            width: stream_config.get_size().width,
            height: stream_config.get_size().height,
            frame_size: stream_config.get_frame_size(),
            frame_duration_us,
        };

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

        // Subscribe to completion events before anything can complete
        let completed_rx = camera.subscribe_request_completed();

        let start_controls = match frame_duration_us {
            Some(us) => {
                let mut list = ControlList::new();
                list.set(controls::FrameDurationLimits([us as i64, us as i64]))
                    .map_err(|e| {
                        DeviceError::LibCamera(format!("setting FrameDurationLimits: {e}"))
                    })?;
                Some(list)
            }
            None => None,
        };

        // Start camera before queueing so completed requests flow immediately
        camera
            .start(start_controls.as_deref())
            .map_err(|e| DeviceError::LibCamera(e.to_string()))?;

        // Queue all requests (queue_request hands the request back on failure)
        for request in requests {
            camera
                .queue_request(request)
                .map_err(|(_, e)| DeviceError::LibCamera(e.to_string()))?;
        }

        if startup_tx.send(Ok(negotiated)).is_err() {
            // The constructor gave up waiting; nobody will consume frames.
            let _ = camera.stop();
            return Ok(());
        }

        // Fallback timestamp base for pipelines that report no SensorTimestamp
        let started_at = std::time::Instant::now();

        // Main capture loop: receive completed requests, ship the frame,
        // recycle the request.
        loop {
            // Stop on a shutdown message or a closed shutdown channel
            // (kanal returns Ok(None) when the channel is just empty)
            if !matches!(shutdown_rx.try_recv(), Ok(None)) {
                break;
            }

            let mut request = match completed_rx.recv_timeout(CAPTURE_POLL_INTERVAL) {
                Ok(req) => req,
                Err(std_mpsc::RecvTimeoutError::Timeout) => continue,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => break,
            };

            if request.status() == RequestStatus::Complete {
                let timestamp_ns = request
                    .metadata()
                    .get::<controls::SensorTimestamp>()
                    .map(|t| t.0)
                    .unwrap_or_else(|_| started_at.elapsed().as_nanos() as i64);

                if let Some(fb) = request.buffer::<MemoryMappedFrameBuffer<FrameBuffer>>(&stream) {
                    // Concatenate the used part of each plane
                    let planes = fb.data();
                    let plane_meta = fb.metadata().map(|m| m.planes());
                    let mut data = Vec::with_capacity(planes.iter().map(|p| p.len()).sum());
                    for (i, plane) in planes.iter().enumerate() {
                        let used = plane_meta
                            .as_ref()
                            .and_then(|pm| pm.get(i))
                            .map(|pm| pm.bytes_used as usize)
                            .unwrap_or(plane.len())
                            .min(plane.len());
                        data.extend_from_slice(&plane[..used]);
                    }

                    let frame = CapturedFrame {
                        data,
                        timestamp_ns,
                        sequence: u64::from(request.sequence()),
                    };
                    match frame_tx.try_send(frame) {
                        Ok(true) => {}
                        Ok(false) => {
                            // Consumer is behind; drop the frame (live source).
                            tracing::trace!("libcamera: frame channel full, dropping frame");
                        }
                        Err(_) => break, // LibCameraSrc was dropped
                    }
                }
            }

            request.reuse(ReuseFlag::REUSE_BUFFERS);
            if let Err((_, e)) = camera.queue_request(request) {
                tracing::error!("libcamera: failed to re-queue request: {}", e);
                break;
            }
        }

        // Stop camera; in-flight requests complete as Cancelled into the
        // soon-dropped completion channel.
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

    /// Actual frame width negotiated with the camera (drivers may adjust
    /// the requested size).
    pub fn width(&self) -> u32 {
        self.negotiated.width
    }

    /// Actual frame height negotiated with the camera.
    pub fn height(&self) -> u32 {
        self.negotiated.height
    }

    /// Effective frame rate as an fps fraction (numerator, denominator).
    ///
    /// This is the requested rate after clamping to the camera's advertised
    /// `FrameDurationLimits` bounds; `None` when no rate was requested
    /// (camera default). Best-effort: pipeline handlers that ignore the
    /// control (e.g. UVC webcams) run at their own rate, observable from
    /// buffer PTS deltas.
    pub fn framerate(&self) -> Option<(u32, u32)> {
        self.framerate
    }

    /// Calculate relative PTS from a capture timestamp.
    ///
    /// libcamera reports absolute CLOCK_MONOTONIC timestamps; convert to
    /// time relative to the first captured frame.
    fn calculate_pts(&self, timestamp_ns: i64) -> ClockTime {
        // Only the first frame sets the base timestamp
        let _ = self.first_timestamp_ns.compare_exchange(
            i64::MIN,
            timestamp_ns,
            Ordering::SeqCst,
            Ordering::SeqCst,
        );

        let first_ns = self.first_timestamp_ns.load(Ordering::SeqCst);
        ClockTime::from_nanos((timestamp_ns - first_ns).max(0) as u64)
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
                    ctx.set_pts(self.calculate_pts(frame.timestamp_ns));
                    ctx.set_sequence(frame.sequence);
                    if self.frame_duration > ClockTime::ZERO {
                        ctx.metadata_mut().duration = self.frame_duration;
                    }
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
        if self.negotiated.frame_size > 0 {
            Some(self.negotiated.frame_size as usize)
        } else {
            // Assume worst case (RGB24) for the negotiated size
            Some((self.negotiated.width * self.negotiated.height * 3) as usize)
        }
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
    fn test_framerate_to_duration() {
        assert_eq!(framerate_to_duration_us(30, 1).unwrap(), 33333);
        assert_eq!(framerate_to_duration_us(30000, 1001).unwrap(), 33366);
        assert_eq!(framerate_to_duration_us(15, 1).unwrap(), 66666);
        assert_eq!(framerate_to_duration_us(1, 1).unwrap(), 1_000_000);
        assert!(framerate_to_duration_us(0, 1).is_err());
        assert!(framerate_to_duration_us(30, 0).is_err());
    }

    #[test]
    fn test_clamp_frame_duration() {
        // In range: unchanged
        assert_eq!(
            clamp_frame_duration_us(33333, Some((10_000, 100_000))),
            33333
        );
        // Below min / above max: clamped
        assert_eq!(
            clamp_frame_duration_us(5_000, Some((10_000, 100_000))),
            10_000
        );
        assert_eq!(
            clamp_frame_duration_us(200_000, Some((10_000, 100_000))),
            100_000
        );
        // No or nonsensical bounds: passthrough
        assert_eq!(clamp_frame_duration_us(33333, None), 33333);
        assert_eq!(clamp_frame_duration_us(33333, Some((-1, 100))), 33333);
        assert_eq!(clamp_frame_duration_us(33333, Some((100, 10))), 33333);
    }

    #[test]
    fn test_reduce_fraction() {
        assert_eq!(reduce_fraction(1_000_000, 33333), (1_000_000, 33333)); // coprime
        assert_eq!(reduce_fraction(1_000_000, 50_000), (20, 1));
        assert_eq!(reduce_fraction(1_000_000, 66666), (500_000, 33333));
        assert_eq!(reduce_fraction(30, 0), (1, 0)); // gcd(30, 0) = 30; no panic
    }

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
