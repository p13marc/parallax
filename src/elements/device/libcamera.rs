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

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

use kanal::{Receiver, Sender, bounded};
use libcamera::{
    camera::CameraConfigurationStatus,
    camera_manager::CameraManager,
    framebuffer::AsFrameBuffer,
    framebuffer_allocator::{FrameBuffer, FrameBufferAllocator},
    framebuffer_map::MemoryMappedFrameBuffer,
    pixel_format::PixelFormat,
    properties,
    request::ReuseFlag,
    stream::StreamRole,
};

use crate::buffer::Buffer;
use crate::element::ProduceContext;
use crate::element::traits::{Affinity, AsyncSource, ExecutionHints, ProduceResult};
use crate::error::Result;

use super::{CameraLocation, DeviceError};

/// Check if libcamera is available on this system.
pub fn is_available() -> bool {
    match CameraManager::new() {
        Ok(cm) => !cm.cameras().is_empty(),
        Err(_) => false,
    }
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
    let cm = CameraManager::new().map_err(|e| DeviceError::LibCamera(e.to_string()))?;

    let mut cameras = Vec::new();
    for camera in cm.cameras() {
        let id = camera.id().to_string();

        // Get model from properties
        let model = camera
            .properties()
            .get::<properties::Model>()
            .map(|m| m.to_string())
            .unwrap_or_else(|| "Unknown".to_string());

        // Get location from properties
        let location = camera
            .properties()
            .get::<properties::Location>()
            .map(|loc| match loc {
                properties::CameraLocation::Front => CameraLocation::Front,
                properties::CameraLocation::Back => CameraLocation::Back,
                properties::CameraLocation::External => CameraLocation::External,
            })
            .unwrap_or(CameraLocation::External);

        cameras.push(LibCameraInfo {
            id,
            model,
            location,
        });
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
        })
    }

    /// Main capture thread.
    fn capture_thread(
        camera_id: String,
        config: LibCameraConfig,
        frame_tx: Sender<CapturedFrame>,
        shutdown_rx: Receiver<()>,
    ) -> Result<()> {
        // Create camera manager
        let cm = CameraManager::new().map_err(|e| DeviceError::LibCamera(e.to_string()))?;

        // Find camera
        let camera = cm
            .cameras()
            .into_iter()
            .find(|c| c.id() == camera_id)
            .ok_or_else(|| DeviceError::NotFound(camera_id.clone()))?;

        // Acquire camera
        let mut camera = camera
            .acquire()
            .map_err(|e| DeviceError::LibCamera(e.to_string()))?;

        // Generate configuration
        let mut cam_config = camera
            .generate_configuration(&[StreamRole::VideoRecording])
            .map_err(|e| DeviceError::LibCamera(e.to_string()))?;

        // Modify configuration if requested
        if let Some(stream_config) = cam_config.get_mut(0) {
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
        let width = stream_config.get_size().width;
        let height = stream_config.get_size().height;

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
        let mut requests: Vec<_> = buffers
            .into_iter()
            .enumerate()
            .map(|(i, buf)| {
                let mut request = camera.create_request(Some(i as u64)).unwrap();
                request.add_buffer(&stream, buf).unwrap();
                request
            })
            .collect();

        // Queue all requests
        for request in &mut requests {
            camera
                .queue_request(request)
                .map_err(|e| DeviceError::LibCamera(e.to_string()))?;
        }

        // Start camera
        camera
            .start()
            .map_err(|e| DeviceError::LibCamera(e.to_string()))?;

        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        // Request completed callback storage
        let completed_requests = Arc::new(std::sync::Mutex::new(Vec::new()));
        let completed_clone = completed_requests.clone();

        // Main capture loop
        while running.load(Ordering::SeqCst) {
            // Check for shutdown
            if shutdown_rx.try_recv().is_ok() {
                running_clone.store(false, Ordering::SeqCst);
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
        match self.receiver.recv_async().await {
            Ok(frame) => {
                let len = frame.data.len();
                if len > 0 && len <= ctx.output().len() {
                    ctx.output()[..len].copy_from_slice(&frame.data);
                    // NOTE: Metadata (timestamp, width, height) should be set
                    // via ProduceContext when buffer metadata API is extended.
                    Ok(ProduceResult::Produced(len))
                } else if len > ctx.output().len() {
                    // Buffer too small
                    Ok(ProduceResult::OwnBuffer(Buffer::from(frame.data)))
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
