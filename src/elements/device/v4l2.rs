//! V4L2 video capture (fallback).
//!
//! V4L2 (Video4Linux2) provides direct access to video capture devices.
//! This is a fallback for when libcamera is not available or for simple
//! webcams that don't need ISP processing.
//!
//! ## Example
//!
//! ```rust,ignore
//! use parallax::elements::device::v4l2::{V4l2Src, V4l2DeviceInfo};
//!
//! // List available devices
//! let devices = V4l2Src::enumerate_devices()?;
//! for dev in &devices {
//!     println!("{}: {}", dev.path.display(), dev.name);
//! }
//!
//! // Open default camera
//! let camera = V4l2Src::new("/dev/video0")?;
//! ```
//!
//! ## DMA-BUF Export
//!
//! For zero-copy GPU pipelines, enable DMA-BUF export mode:
//!
//! ```rust,ignore
//! use parallax::elements::device::v4l2::{V4l2Src, V4l2Config};
//!
//! let config = V4l2Config {
//!     dmabuf_export: true,  // Export via VIDIOC_EXPBUF
//!     ..Default::default()
//! };
//! let camera = V4l2Src::with_config("/dev/video0", config)?;
//!
//! // Camera now declares DmaBuf as preferred memory type
//! // Pipeline will automatically select DMA-BUF path when downstream supports it
//! ```

use std::os::fd::{FromRawFd, OwnedFd};
use std::path::PathBuf;

use v4l::buffer::Type;
use v4l::io::mmap::Stream as MmapStream;
use v4l::io::traits::CaptureStream;
use v4l::prelude::*;
use v4l::video::Capture;

use crate::buffer::{Buffer, DmaBufBuffer, MemoryHandle};
use crate::element::{Affinity, ExecutionHints, ProduceContext, ProduceResult, Source};
use crate::error::Result;
use crate::format::{
    Caps, CapsValue, ElementMediaCaps, FormatMemoryCap, MemoryCaps, PixelFormat, VideoFormatCaps,
};
use crate::memory::{DmaBufSegment, SharedArena};
use crate::metadata::Metadata;

use super::DeviceError;

/// Check if V4L2 is available on this system.
pub fn is_available() -> bool {
    // Check if any video devices exist
    std::path::Path::new("/dev/video0").exists()
        || enumerate_devices().map(|d| !d.is_empty()).unwrap_or(false)
}

/// Information about a V4L2 device.
#[derive(Debug, Clone)]
pub struct V4l2DeviceInfo {
    /// Device path (e.g., /dev/video0).
    pub path: PathBuf,
    /// Device name.
    pub name: String,
    /// Driver name.
    pub driver: String,
    /// Bus information.
    pub bus_info: String,
}

/// Enumerate V4L2 devices.
pub fn enumerate_devices() -> Result<Vec<V4l2DeviceInfo>> {
    let mut devices = Vec::new();

    // Scan /dev/video* devices
    for i in 0..64 {
        let path = PathBuf::from(format!("/dev/video{}", i));
        if !path.exists() {
            continue;
        }

        // Try to open and query the device
        match Device::with_path(&path) {
            Ok(dev) => {
                if let Ok(caps) = dev.query_caps() {
                    // Only include capture devices
                    if caps
                        .capabilities
                        .contains(v4l::capability::Flags::VIDEO_CAPTURE)
                    {
                        devices.push(V4l2DeviceInfo {
                            path,
                            name: caps.card.clone(),
                            driver: caps.driver.clone(),
                            bus_info: caps.bus.clone(),
                        });
                    }
                }
            }
            Err(_) => continue,
        }
    }

    Ok(devices)
}

/// V4L2 video capture configuration.
#[derive(Debug, Clone)]
pub struct V4l2Config {
    /// Desired width.
    pub width: u32,
    /// Desired height.
    pub height: u32,
    /// Desired fourcc format (e.g., "YUYV", "MJPG").
    pub fourcc: Option<String>,
    /// Number of buffers.
    pub buffer_count: u32,
    /// Export buffers as DMA-BUF file descriptors.
    ///
    /// When enabled:
    /// - Buffers are exported via VIDIOC_EXPBUF
    /// - Zero-copy path to GPU or other processes
    /// - `output_media_caps()` declares DmaBuf as preferred memory type
    ///
    /// When disabled (default):
    /// - Standard mmap capture with copy to arena
    /// - Compatible with any downstream element
    pub dmabuf_export: bool,
}

impl Default for V4l2Config {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            fourcc: None,
            buffer_count: 4,
            dmabuf_export: false,
        }
    }
}

/// A supported format from the V4L2 device.
#[derive(Debug, Clone)]
pub struct V4l2SupportedFormat {
    /// FourCC code.
    pub fourcc: [u8; 4],
    /// Supported resolutions (width, height).
    pub resolutions: Vec<(u32, u32)>,
}

// VIDIOC_EXPBUF ioctl for exporting V4L2 buffers as DMA-BUF fds.
// The v4l crate doesn't expose this directly, so we use raw ioctls.
// See: https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/vidioc-expbuf.html

/// VIDIOC_EXPBUF ioctl number: _IOWR('V', 16, struct v4l2_exportbuffer)
const VIDIOC_EXPBUF: libc::c_ulong = 0xc040_5610;

/// V4L2 buffer type for video capture.
const V4L2_BUF_TYPE_VIDEO_CAPTURE: u32 = 1;

/// v4l2_exportbuffer structure for VIDIOC_EXPBUF ioctl.
#[repr(C)]
struct V4l2ExportBuffer {
    /// Buffer type (V4L2_BUF_TYPE_VIDEO_CAPTURE).
    type_: u32,
    /// Buffer index to export.
    index: u32,
    /// Plane index (0 for single-planar formats).
    plane: u32,
    /// Flags (O_CLOEXEC, O_RDONLY, etc.).
    flags: u32,
    /// OUTPUT: Exported DMA-BUF file descriptor.
    fd: i32,
    /// Reserved for future use.
    reserved: [u32; 11],
}

/// V4L2 video capture source.
///
/// This source captures video frames from a V4L2 device using memory-mapped
/// buffers for efficient zero-copy capture.
///
/// # Device Lifecycle
///
/// The device is properly released when V4l2Src is dropped. The stream
/// buffers are unmapped and streaming is stopped before the device is closed.
///
/// # DMA-BUF Export Mode
///
/// When `dmabuf_export` is enabled in the config, buffers are exported as
/// DMA-BUF file descriptors using VIDIOC_EXPBUF. This enables zero-copy
/// sharing with GPU pipelines or other processes.
pub struct V4l2Src {
    /// The V4L2 device - must be kept alive for the stream to work.
    /// Option allows us to control drop order (stream first, then device).
    device: Option<Device>,
    /// The mmap stream - dropped before device.
    stream: Option<MmapStream<'static>>,
    /// Device path.
    path: PathBuf,
    /// Actual format being used.
    width: u32,
    height: u32,
    fourcc: [u8; 4],
    /// All supported formats from the device (cached for caps negotiation).
    supported_formats: Vec<V4l2SupportedFormat>,
    /// Arena for buffer allocation (per-source to avoid contention).
    arena: Option<SharedArena>,
    /// Whether to export buffers as DMA-BUF file descriptors.
    dmabuf_export: bool,
    /// Exported DMA-BUF file descriptors (one per buffer, when dmabuf_export is true).
    exported_fds: Vec<OwnedFd>,
    /// Frame sizes for each exported buffer (needed to create DmaBufSegment).
    exported_sizes: Vec<usize>,
}

impl V4l2Src {
    /// Create a capture source for the given device path.
    pub fn new(device_path: &str) -> Result<Self> {
        Self::with_config(device_path, V4l2Config::default())
    }

    /// Create a capture source with specific configuration.
    pub fn with_config(device_path: &str, config: V4l2Config) -> Result<Self> {
        let path = PathBuf::from(device_path);

        // Open device
        let dev = Device::with_path(&path).map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                DeviceError::NotFound(device_path.to_string())
            } else if e.kind() == std::io::ErrorKind::PermissionDenied {
                DeviceError::PermissionDenied(device_path.to_string())
            } else {
                DeviceError::V4l2(e)
            }
        })?;

        // Query supported formats for caps negotiation
        let supported_formats = Self::query_supported_formats(&dev);

        // Set format
        let fourcc = if let Some(ref fcc) = config.fourcc {
            let bytes = fcc.as_bytes();
            if bytes.len() >= 4 {
                [bytes[0], bytes[1], bytes[2], bytes[3]]
            } else {
                // Default to YUYV
                *b"YUYV"
            }
        } else {
            // Try MJPG first (compressed, less bandwidth), fallback to YUYV
            *b"MJPG"
        };

        let mut format = v4l::Format::new(config.width, config.height, v4l::FourCC::new(&fourcc));
        format = dev.set_format(&format).map_err(DeviceError::V4l2)?;

        let width = format.width;
        let height = format.height;
        let actual_fourcc = format.fourcc.repr;

        // Create mmap stream
        let stream = MmapStream::with_buffers(&dev, Type::VideoCapture, config.buffer_count)
            .map_err(DeviceError::V4l2)?;

        // SAFETY: We store both `dev` and `stream` in the struct, ensuring
        // the device outlives the stream. The stream is dropped first in our
        // Drop implementation, then the device.
        let stream: MmapStream<'static> = unsafe { std::mem::transmute(stream) };

        // Export buffers as DMA-BUF fds if requested
        let (exported_fds, exported_sizes) = if config.dmabuf_export {
            Self::export_buffers(&dev, config.buffer_count, width, height, &actual_fourcc)?
        } else {
            (Vec::new(), Vec::new())
        };

        Ok(Self {
            device: Some(dev),
            stream: Some(stream),
            path,
            width,
            height,
            fourcc: actual_fourcc,
            supported_formats,
            arena: None,
            dmabuf_export: config.dmabuf_export,
            exported_fds,
            exported_sizes,
        })
    }

    /// Export V4L2 buffers as DMA-BUF file descriptors using VIDIOC_EXPBUF.
    fn export_buffers(
        dev: &Device,
        buffer_count: u32,
        width: u32,
        height: u32,
        fourcc: &[u8; 4],
    ) -> Result<(Vec<OwnedFd>, Vec<usize>)> {
        let device_fd = dev.handle().fd();

        // Calculate frame size for this format
        let fourcc_str = std::str::from_utf8(fourcc).unwrap_or("????");
        let frame_size = match fourcc_str {
            "MJPG" | "JPEG" => (width * height) as usize, // Estimate for compressed
            "YUYV" | "UYVY" => (width * height * 2) as usize,
            "NV12" | "NV21" => (width * height * 3 / 2) as usize,
            "I420" | "YU12" => (width * height * 3 / 2) as usize,
            "RGB3" | "BGR3" => (width * height * 3) as usize,
            "RGBP" | "RGB4" | "BA24" => (width * height * 4) as usize,
            "GREY" | "Y800" => (width * height) as usize,
            _ => (width * height * 4) as usize, // Worst case
        };

        let mut fds = Vec::with_capacity(buffer_count as usize);
        let mut sizes = Vec::with_capacity(buffer_count as usize);

        for i in 0..buffer_count {
            let mut expbuf = V4l2ExportBuffer {
                type_: V4L2_BUF_TYPE_VIDEO_CAPTURE,
                index: i,
                plane: 0,
                flags: libc::O_CLOEXEC as u32 | libc::O_RDWR as u32,
                fd: 0,
                reserved: [0; 11],
            };

            // SAFETY: VIDIOC_EXPBUF is a standard V4L2 ioctl that exports a buffer
            // as a DMA-BUF fd. The device fd is valid and the struct is properly
            // initialized.
            let ret = unsafe { libc::ioctl(device_fd, VIDIOC_EXPBUF, &mut expbuf) };

            if ret < 0 {
                let err = std::io::Error::last_os_error();
                return Err(DeviceError::V4l2(err).into());
            }

            // SAFETY: On success, VIDIOC_EXPBUF returns a valid DMA-BUF fd in expbuf.fd
            let owned_fd = unsafe { OwnedFd::from_raw_fd(expbuf.fd) };
            fds.push(owned_fd);
            sizes.push(frame_size);
        }

        Ok((fds, sizes))
    }

    /// Query supported formats from the device.
    fn query_supported_formats(dev: &Device) -> Vec<V4l2SupportedFormat> {
        let mut formats = Vec::new();

        // Enumerate supported formats
        if let Ok(format_descs) = dev.enum_formats() {
            for fmt_desc in format_descs {
                let fourcc = fmt_desc.fourcc.repr;

                // Query supported frame sizes for this format
                let mut resolutions = Vec::new();
                if let Ok(sizes) = dev.enum_framesizes(fmt_desc.fourcc) {
                    for size in sizes {
                        match size.size {
                            v4l::framesize::FrameSizeEnum::Discrete(d) => {
                                resolutions.push((d.width, d.height));
                            }
                            v4l::framesize::FrameSizeEnum::Stepwise(s) => {
                                // For stepwise, add common resolutions within range
                                for (w, h) in &[
                                    (640, 480),
                                    (800, 600),
                                    (1280, 720),
                                    (1920, 1080),
                                    (3840, 2160),
                                ] {
                                    if *w >= s.min_width
                                        && *w <= s.max_width
                                        && *h >= s.min_height
                                        && *h <= s.max_height
                                    {
                                        resolutions.push((*w, *h));
                                    }
                                }
                            }
                        }
                    }
                }

                // If no resolutions found, add a default
                if resolutions.is_empty() {
                    resolutions.push((640, 480));
                }

                formats.push(V4l2SupportedFormat {
                    fourcc,
                    resolutions,
                });
            }
        }

        formats
    }

    /// Get all supported formats from this device.
    pub fn supported_formats(&self) -> &[V4l2SupportedFormat] {
        &self.supported_formats
    }

    /// Get the device path.
    pub fn path(&self) -> &PathBuf {
        &self.path
    }

    /// Get the actual capture width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get the actual capture height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get the actual fourcc format.
    pub fn fourcc(&self) -> &[u8; 4] {
        &self.fourcc
    }

    /// Enumerate available devices.
    pub fn enumerate_devices() -> Result<Vec<V4l2DeviceInfo>> {
        enumerate_devices()
    }

    /// Stop capturing and release the device.
    ///
    /// This is called automatically on drop, but can be called explicitly
    /// to release the device early.
    pub fn stop(&mut self) {
        // Drop stream first to stop streaming and unmap buffers
        self.stream.take();
        // Then drop the device to close the file descriptor
        self.device.take();
    }
}

impl Drop for V4l2Src {
    fn drop(&mut self) {
        // Ensure proper drop order: stream first, then device
        self.stop();
    }
}

impl Source for V4l2Src {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        // DMA-BUF export path: capture frame and return DmaBufBuffer
        if self.dmabuf_export && !self.exported_fds.is_empty() {
            let stream = self
                .stream
                .as_mut()
                .ok_or_else(|| DeviceError::NotFound("device closed".to_string()))?;

            let (_buffer, meta) = stream.next().map_err(DeviceError::V4l2)?;
            let buffer_index = (meta.sequence as usize) % self.exported_fds.len();

            // Clone the fd so the segment owns it (we keep the original for reuse)
            let fd = self.exported_fds[buffer_index]
                .try_clone()
                .map_err(DeviceError::V4l2)?;

            let size = self.exported_sizes[buffer_index];
            let segment = DmaBufSegment::from_fd(fd, size)?;

            return Ok(ProduceResult::OwnDmaBuf(DmaBufBuffer::new(
                segment,
                Metadata::new(),
            )));
        }

        // Standard mmap path: lazily initialize per-source arena if needed
        if self.arena.is_none() {
            let slot_size = self
                .preferred_buffer_size()
                .unwrap_or(self.width as usize * self.height as usize * 4);
            self.arena = Some(SharedArena::new(slot_size, 32)?);
        }

        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| DeviceError::NotFound("device closed".to_string()))?;

        // Capture a frame
        let (buffer, _meta) = stream.next().map_err(DeviceError::V4l2)?;
        let len = buffer.len();

        if !ctx.has_buffer() || len > ctx.capacity() {
            // No buffer provided or buffer too small, return our own buffer from arena
            let arena = self.arena.as_mut().unwrap();
            arena.reclaim();

            let mut slot = arena
                .acquire()
                .ok_or_else(|| crate::error::Error::Element("V4L2 arena exhausted".to_string()))?;

            slot.data_mut()[..len].copy_from_slice(buffer);

            let handle = MemoryHandle::with_len(slot, len);

            return Ok(ProduceResult::OwnBuffer(Buffer::new(
                handle,
                Metadata::new(),
            )));
        }

        ctx.output()[..len].copy_from_slice(buffer);

        Ok(ProduceResult::Produced(len))
    }

    fn preferred_buffer_size(&self) -> Option<usize> {
        // Estimate based on format
        let fourcc_str = std::str::from_utf8(&self.fourcc).unwrap_or("????");
        let size = match fourcc_str {
            "MJPG" | "JPEG" => {
                // MJPEG is compressed, estimate max size
                (self.width * self.height) as usize
            }
            "YUYV" | "UYVY" => {
                // YUV 4:2:2 = 2 bytes per pixel
                (self.width * self.height * 2) as usize
            }
            "NV12" | "NV21" => {
                // YUV 4:2:0 = 1.5 bytes per pixel
                (self.width * self.height * 3 / 2) as usize
            }
            "RGB3" | "BGR3" => {
                // RGB24 = 3 bytes per pixel
                (self.width * self.height * 3) as usize
            }
            _ => {
                // Unknown, assume worst case
                (self.width * self.height * 4) as usize
            }
        };
        Some(size)
    }

    fn affinity(&self) -> Affinity {
        // V4L2 capture is blocking I/O
        Affinity::Async
    }

    fn is_rt_safe(&self) -> bool {
        false
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::io_bound()
    }

    fn output_caps(&self) -> Caps {
        // Convert V4L2 fourcc to our PixelFormat
        let fourcc_str = std::str::from_utf8(&self.fourcc).unwrap_or("????");
        let pixel_format = match fourcc_str {
            "YUYV" => PixelFormat::Yuyv,
            "UYVY" => PixelFormat::Uyvy,
            "NV12" => PixelFormat::Nv12,
            "I420" | "YU12" => PixelFormat::I420,
            "RGB3" => PixelFormat::Rgb24,
            "BGR3" => PixelFormat::Bgr24,
            "RGBP" | "RGB4" => PixelFormat::Rgba,
            "BA24" => PixelFormat::Bgra,
            "GREY" | "Y800" => PixelFormat::Gray8,
            // MJPEG and other compressed formats - we can't negotiate these
            // through the raw video pipeline, return any() to allow the
            // pipeline to fail with "cannot negotiate" if no decoder is available
            "MJPG" | "JPEG" => return Caps::any(),
            _ => {
                tracing::warn!("Unknown V4L2 format: {}, returning Caps::any()", fourcc_str);
                return Caps::any();
            }
        };

        Caps::video_raw(self.width, self.height, pixel_format)
    }

    fn output_media_caps(&self) -> ElementMediaCaps {
        // Build caps from all supported formats on the device
        // This demonstrates the enhanced caps negotiation where an element
        // can declare multiple format+memory combinations.

        let mut caps = Vec::new();

        for supported in &self.supported_formats {
            let fourcc_str = std::str::from_utf8(&supported.fourcc).unwrap_or("????");

            // Convert V4L2 fourcc to our PixelFormat
            let pixel_format = match fourcc_str {
                "YUYV" => Some(PixelFormat::Yuyv),
                "UYVY" => Some(PixelFormat::Uyvy),
                "NV12" => Some(PixelFormat::Nv12),
                "I420" | "YU12" => Some(PixelFormat::I420),
                "RGB3" => Some(PixelFormat::Rgb24),
                "BGR3" => Some(PixelFormat::Bgr24),
                "RGBP" | "RGB4" => Some(PixelFormat::Rgba),
                "BA24" => Some(PixelFormat::Bgra),
                "GREY" | "Y800" => Some(PixelFormat::Gray8),
                // Skip compressed formats like MJPEG
                _ => None,
            };

            if let Some(pixel_format) = pixel_format {
                // Build resolution constraint from supported resolutions
                let (widths, heights): (Vec<u32>, Vec<u32>) =
                    supported.resolutions.iter().cloned().unzip();

                let width_caps = if widths.len() == 1 {
                    CapsValue::Fixed(widths[0])
                } else {
                    CapsValue::List(widths)
                };

                let height_caps = if heights.len() == 1 {
                    CapsValue::Fixed(heights[0])
                } else {
                    CapsValue::List(heights)
                };

                let format_caps = VideoFormatCaps {
                    width: width_caps,
                    height: height_caps,
                    pixel_format: CapsValue::Fixed(pixel_format),
                    framerate: CapsValue::Any,
                };

                if self.dmabuf_export {
                    // DMA-BUF preferred when export is enabled
                    caps.push(FormatMemoryCap::new(
                        format_caps.clone().into(),
                        MemoryCaps::dmabuf_only(),
                    ));
                }

                // CPU memory always available (fallback or primary)
                caps.push(FormatMemoryCap::new(
                    format_caps.into(),
                    MemoryCaps::cpu_only(),
                ));
            }
        }

        if caps.is_empty() {
            // Fallback: use current format only
            let fourcc_str = std::str::from_utf8(&self.fourcc).unwrap_or("????");
            let pixel_format = match fourcc_str {
                "YUYV" => Some(PixelFormat::Yuyv),
                "UYVY" => Some(PixelFormat::Uyvy),
                "NV12" => Some(PixelFormat::Nv12),
                "I420" | "YU12" => Some(PixelFormat::I420),
                "RGB3" => Some(PixelFormat::Rgb24),
                "BGR3" => Some(PixelFormat::Bgr24),
                "RGBP" | "RGB4" => Some(PixelFormat::Rgba),
                "BA24" => Some(PixelFormat::Bgra),
                "GREY" | "Y800" => Some(PixelFormat::Gray8),
                _ => None,
            };

            if let Some(pixel_format) = pixel_format {
                let format_caps = VideoFormatCaps {
                    width: CapsValue::Fixed(self.width),
                    height: CapsValue::Fixed(self.height),
                    pixel_format: CapsValue::Fixed(pixel_format),
                    framerate: CapsValue::Any,
                };

                if self.dmabuf_export {
                    caps.push(FormatMemoryCap::new(
                        format_caps.clone().into(),
                        MemoryCaps::dmabuf_only(),
                    ));
                }

                caps.push(FormatMemoryCap::new(
                    format_caps.into(),
                    MemoryCaps::cpu_only(),
                ));
            }
        }

        if caps.is_empty() {
            // Still empty (e.g., MJPEG only) - return any CPU
            ElementMediaCaps::any_cpu()
        } else {
            ElementMediaCaps::new(caps)
        }
    }
}

impl V4l2Src {
    /// Whether DMA-BUF export is enabled.
    pub fn is_dmabuf_export(&self) -> bool {
        self.dmabuf_export
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_available() {
        let available = is_available();
        println!("V4L2 available: {}", available);
    }

    #[test]
    fn test_enumerate_devices() {
        match enumerate_devices() {
            Ok(devices) => {
                println!("Found {} V4L2 devices:", devices.len());
                for dev in &devices {
                    println!(
                        "  {} - {} (driver: {}, bus: {})",
                        dev.path.display(),
                        dev.name,
                        dev.driver,
                        dev.bus_info
                    );
                }
            }
            Err(e) => {
                println!("Failed to enumerate: {}", e);
            }
        }
    }
}
