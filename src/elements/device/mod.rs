//! Device capture and playback elements.
//!
//! This module provides elements for capturing from and playing to hardware devices
//! on Linux. The modern stack (PipeWire, libcamera) is preferred, with fallbacks
//! to V4L2 and ALSA for compatibility.
//!
//! ## Recommended Stack (Modern Linux)
//!
//! | Device Type | Primary | Fallback |
//! |-------------|---------|----------|
//! | Camera | libcamera | V4L2 |
//! | Audio | PipeWire | ALSA |
//! | Screen | PipeWire (portal) | - |
//!
//! ## Feature Flags
//!
//! - `pipewire` - PipeWire audio/video capture and playback
//! - `libcamera` - Modern camera API (Raspberry Pi, embedded, complex cameras)
//! - `screen-capture` - Screen capture via XDG portal (requires pipewire)
//! - `v4l2` - V4L2 video capture (fallback for simple webcams)
//! - `alsa` - ALSA audio capture/playback (fallback)
//! - `device-capture` - Recommended stack (pipewire + libcamera)
//! - `device-all` - All device capture features
//!
//! ## Usage
//!
//! ```rust,ignore
//! use parallax::elements::device::{camera_src, audio_src, audio_sink};
//!
//! // Auto-detect best backend
//! let camera = camera_src()?;  // Returns libcamera or V4L2 source
//! let mic = audio_src()?;      // Returns PipeWire or ALSA source
//! let speaker = audio_sink()?; // Returns PipeWire or ALSA sink
//! ```

use crate::error::Result;
use thiserror::Error;

// Module declarations
#[cfg(feature = "pipewire")]
pub mod pipewire;

#[cfg(feature = "libcamera")]
pub mod libcamera;

#[cfg(feature = "v4l2")]
pub mod v4l2;

#[cfg(feature = "alsa")]
pub mod alsa;

#[cfg(feature = "screen-capture")]
pub mod screen_capture;

// Re-exports
#[cfg(feature = "pipewire")]
pub use self::pipewire::{PipeWireSink, PipeWireSrc, PipeWireTarget};

#[cfg(feature = "screen-capture")]
pub use self::screen_capture::{
    CaptureSourceType, ScreenCaptureConfig, ScreenCaptureInfo, ScreenCaptureSrc,
};

#[cfg(feature = "libcamera")]
pub use self::libcamera::{LibCameraConfig, LibCameraInfo, LibCameraSrc};

#[cfg(feature = "v4l2")]
pub use self::v4l2::{V4l2Config, V4l2DeviceInfo, V4l2Src};

#[cfg(feature = "alsa")]
pub use self::alsa::{AlsaDeviceInfo, AlsaFormat, AlsaSampleFormat, AlsaSink, AlsaSrc};

/// Device capture/playback errors.
#[derive(Debug, Error)]
pub enum DeviceError {
    /// Device not found.
    #[error("Device not found: {0}")]
    NotFound(String),

    /// Device is busy (in use by another application).
    #[error("Device busy: {0}")]
    Busy(String),

    /// Permission denied to access device.
    #[error("Permission denied: {0}")]
    PermissionDenied(String),

    /// Requested format is not supported by the device.
    #[error("Format not supported: {0}")]
    FormatNotSupported(String),

    /// Device was disconnected.
    #[error("Device disconnected")]
    Disconnected,

    /// PipeWire is not available on this system.
    #[error("PipeWire not available")]
    PipeWireNotAvailable,

    /// libcamera is not available on this system.
    #[error("libcamera not available")]
    LibCameraNotAvailable,

    /// Portal request was denied by the user.
    #[error("Portal request denied")]
    PortalDenied,

    /// PipeWire-specific error.
    #[error("PipeWire error: {0}")]
    PipeWire(String),

    /// libcamera-specific error.
    #[error("libcamera error: {0}")]
    LibCamera(String),

    /// V4L2-specific error.
    #[error("V4L2 error: {0}")]
    V4l2(#[from] std::io::Error),

    /// ALSA-specific error.
    #[error("ALSA error: {0}")]
    Alsa(String),

    /// No suitable backend available.
    #[error("No capture backend available")]
    NoBackendAvailable,
}

/// Available capture backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaptureBackend {
    /// PipeWire (recommended for audio and screen capture).
    PipeWire,
    /// libcamera (recommended for camera capture).
    LibCamera,
    /// V4L2 (fallback for simple webcams).
    V4l2,
    /// ALSA (fallback for audio).
    Alsa,
    /// No backend available.
    None,
}

/// Detect the best available video capture backend.
pub fn detect_video_backend() -> CaptureBackend {
    #[cfg(feature = "libcamera")]
    {
        if libcamera_available() {
            return CaptureBackend::LibCamera;
        }
    }

    #[cfg(feature = "v4l2")]
    {
        if v4l2_available() {
            return CaptureBackend::V4l2;
        }
    }

    CaptureBackend::None
}

/// Detect the best available audio capture backend.
pub fn detect_audio_backend() -> CaptureBackend {
    #[cfg(feature = "pipewire")]
    {
        if pipewire_available() {
            return CaptureBackend::PipeWire;
        }
    }

    #[cfg(feature = "alsa")]
    {
        if alsa_available() {
            return CaptureBackend::Alsa;
        }
    }

    CaptureBackend::None
}

/// Check if PipeWire is available on this system.
#[cfg(feature = "pipewire")]
pub fn pipewire_available() -> bool {
    // Try to initialize PipeWire - if it fails, it's not available
    pipewire::is_available()
}

/// Check if PipeWire is available (returns false when feature disabled).
#[cfg(not(feature = "pipewire"))]
pub fn pipewire_available() -> bool {
    false
}

/// Check if libcamera is available on this system.
#[cfg(feature = "libcamera")]
pub fn libcamera_available() -> bool {
    libcamera::is_available()
}

/// Check if libcamera is available (returns false when feature disabled).
#[cfg(not(feature = "libcamera"))]
pub fn libcamera_available() -> bool {
    false
}

/// Check if V4L2 is available on this system.
#[cfg(feature = "v4l2")]
pub fn v4l2_available() -> bool {
    v4l2::is_available()
}

/// Check if V4L2 is available (returns false when feature disabled).
#[cfg(not(feature = "v4l2"))]
pub fn v4l2_available() -> bool {
    false
}

/// Check if ALSA is available on this system.
#[cfg(feature = "alsa")]
pub fn alsa_available() -> bool {
    alsa::is_available()
}

/// Check if ALSA is available (returns false when feature disabled).
#[cfg(not(feature = "alsa"))]
pub fn alsa_available() -> bool {
    false
}

/// Video capture device information (unified across backends).
#[derive(Debug, Clone)]
pub struct VideoCaptureDevice {
    /// Device identifier (path or name).
    pub id: String,
    /// Human-readable device name.
    pub name: String,
    /// Backend used for this device.
    pub backend: CaptureBackend,
    /// Device model (if available).
    pub model: Option<String>,
    /// Device location (front, back, external).
    pub location: Option<CameraLocation>,
}

/// Camera physical location.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CameraLocation {
    /// Front-facing camera (e.g., laptop webcam, phone front camera).
    Front,
    /// Back-facing camera (e.g., phone rear camera).
    Back,
    /// External camera (USB webcam, etc.).
    External,
}

/// Audio capture device information (unified across backends).
#[derive(Debug, Clone)]
pub struct AudioCaptureDevice {
    /// Device identifier.
    pub id: String,
    /// Human-readable device name.
    pub name: String,
    /// Backend used for this device.
    pub backend: CaptureBackend,
    /// Whether this is a capture device.
    pub is_capture: bool,
    /// Whether this is a playback device.
    pub is_playback: bool,
}

/// Enumerate all available video capture devices.
pub fn enumerate_video_devices() -> Result<Vec<VideoCaptureDevice>> {
    #[allow(unused_mut)]
    let mut devices = Vec::new();

    #[cfg(feature = "libcamera")]
    {
        if let Ok(libcamera_devices) = libcamera::enumerate_cameras() {
            for dev in libcamera_devices {
                devices.push(VideoCaptureDevice {
                    id: dev.id.clone(),
                    name: dev.model.clone(),
                    backend: CaptureBackend::LibCamera,
                    model: Some(dev.model),
                    location: Some(dev.location),
                });
            }
        }
    }

    #[cfg(feature = "v4l2")]
    {
        if let Ok(v4l2_devices) = v4l2::enumerate_devices() {
            for dev in v4l2_devices {
                let dev_id = dev.path.to_string_lossy().into_owned();
                // Skip if we already have this device from libcamera
                if !devices.iter().any(|d: &VideoCaptureDevice| d.id == dev_id) {
                    devices.push(VideoCaptureDevice {
                        id: dev_id,
                        name: dev.name.clone(),
                        backend: CaptureBackend::V4l2,
                        model: Some(dev.name),
                        location: None,
                    });
                }
            }
        }
    }

    Ok(devices)
}

/// Enumerate all available audio devices.
#[allow(unused_mut)]
pub fn enumerate_audio_devices() -> Result<Vec<AudioCaptureDevice>> {
    let mut devices = Vec::new();

    #[cfg(feature = "pipewire")]
    {
        if let Ok(pw_devices) = pipewire::enumerate_audio_nodes() {
            for dev in pw_devices {
                devices.push(AudioCaptureDevice {
                    id: dev.id.to_string(),
                    name: dev.description.clone(),
                    backend: CaptureBackend::PipeWire,
                    is_capture: dev.is_capture,
                    is_playback: dev.is_playback,
                });
            }
        }
    }

    #[cfg(feature = "alsa")]
    {
        if let Ok(alsa_devices) = alsa::enumerate_devices() {
            for dev in alsa_devices {
                // Skip if we already have this device from PipeWire
                if !devices
                    .iter()
                    .any(|d: &AudioCaptureDevice| d.name == dev.name)
                {
                    devices.push(AudioCaptureDevice {
                        id: dev.name.clone(),
                        name: dev.description.clone(),
                        backend: CaptureBackend::Alsa,
                        is_capture: dev.is_capture,
                        is_playback: dev.is_playback,
                    });
                }
            }
        }
    }

    Ok(devices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_detection() {
        // These tests just verify the detection functions don't panic
        let video = detect_video_backend();
        let audio = detect_audio_backend();
        println!("Video backend: {:?}", video);
        println!("Audio backend: {:?}", audio);
    }

    #[test]
    fn test_enumerate_devices() {
        // These may return empty lists if no devices are available
        let video = enumerate_video_devices().unwrap_or_default();
        let audio = enumerate_audio_devices().unwrap_or_default();
        println!("Found {} video devices", video.len());
        println!("Found {} audio devices", audio.len());
    }
}
