//! GPU codec traits for hardware video encoding/decoding.
//!
//! These traits define the interface for hardware-accelerated video codecs.
//! They are designed to work with the Vulkan Video backend but could support
//! other backends (VA-API, NVENC) in the future.

use crate::error::Result;
use std::os::fd::OwnedFd;

use super::{Codec, GpuUsage, VideoProfile};

/// GPU buffer handle.
///
/// This is an opaque handle to GPU memory. The actual implementation
/// depends on the backend (Vulkan, VA-API, etc.).
#[derive(Debug)]
pub struct GpuBuffer {
    /// Backend-specific handle.
    pub(crate) handle: GpuBufferHandle,
    /// Buffer size in bytes.
    pub size: usize,
    /// Usage flags.
    pub usage: GpuUsage,
}

/// Backend-specific buffer handle.
#[derive(Debug)]
pub(crate) enum GpuBufferHandle {
    /// Placeholder for when no backend is available.
    None,
    /// Vulkan memory handle.
    #[cfg(feature = "vulkan-video")]
    Vulkan {
        memory: ash::vk::DeviceMemory,
        image: Option<ash::vk::Image>,
    },
}

impl Default for GpuBufferHandle {
    fn default() -> Self {
        Self::None
    }
}

/// GPU video frame.
///
/// Represents a decoded video frame in GPU memory.
#[derive(Debug)]
pub struct GpuFrame {
    /// GPU buffer containing frame data.
    pub buffer: GpuBuffer,
    /// Pixel format.
    pub format: GpuPixelFormat,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Row stride in bytes.
    pub stride: u32,
    /// Presentation timestamp in nanoseconds.
    pub pts: i64,
    /// Frame is a keyframe (IDR for H.264/H.265).
    pub is_keyframe: bool,
}

/// GPU pixel format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GpuPixelFormat {
    /// NV12: Y plane followed by interleaved UV plane.
    #[default]
    Nv12,
    /// P010: 10-bit NV12 (Y and UV in 16-bit containers).
    P010,
    /// I420: Y, U, V planes (4:2:0).
    I420,
    /// I420 10-bit.
    I420p10,
}

impl GpuPixelFormat {
    /// Bits per pixel for luma channel.
    pub fn luma_bits(&self) -> u8 {
        match self {
            Self::Nv12 | Self::I420 => 8,
            Self::P010 | Self::I420p10 => 10,
        }
    }

    /// Calculate frame size in bytes.
    pub fn frame_size(&self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;
        match self {
            Self::Nv12 | Self::I420 => w * h * 3 / 2,
            Self::P010 | Self::I420p10 => w * h * 3, // 16-bit per component
        }
    }
}

/// GPU memory management trait.
///
/// Implementations provide GPU memory allocation and DMA-BUF import/export.
pub trait GpuMemory: Send + Sync {
    /// Import a DMA-BUF file descriptor as GPU memory.
    ///
    /// The GPU takes ownership of the file descriptor.
    fn import_dmabuf(&mut self, fd: OwnedFd, size: usize) -> Result<GpuBuffer>;

    /// Export a GPU buffer as a DMA-BUF file descriptor.
    ///
    /// The returned fd can be shared with other processes or devices.
    fn export_dmabuf(&self, buffer: &GpuBuffer) -> Result<OwnedFd>;

    /// Allocate GPU-local memory.
    fn allocate(&mut self, size: usize, usage: GpuUsage) -> Result<GpuBuffer>;

    /// Allocate a GPU image suitable for video decode/encode.
    fn allocate_image(
        &mut self,
        width: u32,
        height: u32,
        format: GpuPixelFormat,
        usage: GpuUsage,
    ) -> Result<GpuBuffer>;

    /// Free a GPU buffer.
    fn free(&mut self, buffer: GpuBuffer);

    /// Map GPU memory for CPU access (if supported).
    fn map(&self, buffer: &GpuBuffer) -> Result<*mut u8>;

    /// Unmap previously mapped memory.
    fn unmap(&self, buffer: &GpuBuffer);
}

/// Hardware video decoder trait.
///
/// Decodes compressed video packets to GPU frames.
pub trait HwVideoDecoder: Send {
    /// Decode a compressed video packet.
    ///
    /// Returns zero or more decoded frames. Due to B-frame reordering,
    /// output frames may not correspond 1:1 with input packets.
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<GpuFrame>>;

    /// Flush any buffered frames at end-of-stream.
    fn flush(&mut self) -> Result<Vec<GpuFrame>>;

    /// Reset decoder state (e.g., after seek).
    fn reset(&mut self) -> Result<()>;

    /// Get the codec this decoder handles.
    fn codec(&self) -> Codec;

    /// Get the current video profile.
    fn profile(&self) -> &VideoProfile;

    /// Get output pixel format.
    fn output_format(&self) -> GpuPixelFormat;

    /// Check if decoder has buffered frames.
    fn has_pending(&self) -> bool {
        false
    }
}

/// Hardware video encoder trait.
///
/// Encodes raw video frames to compressed packets.
pub trait HwVideoEncoder: Send {
    /// Encoded packet type.
    type Packet: AsRef<[u8]> + Send;

    /// Encode a GPU frame.
    ///
    /// Returns zero or more encoded packets. Due to B-frame reordering
    /// and rate control, output may not be immediate.
    fn encode(&mut self, frame: &GpuFrame) -> Result<Vec<Self::Packet>>;

    /// Flush any buffered packets at end-of-stream.
    fn flush(&mut self) -> Result<Vec<Self::Packet>>;

    /// Force an IDR frame on next encode.
    fn force_keyframe(&mut self);

    /// Reset encoder state.
    fn reset(&mut self) -> Result<()>;

    /// Get the codec this encoder produces.
    fn codec(&self) -> Codec;

    /// Get codec-specific header data (SPS/PPS for H.264, VPS/SPS/PPS for H.265).
    fn codec_data(&self) -> Option<Vec<u8>> {
        None
    }

    /// Check if encoder has buffered frames.
    fn has_pending(&self) -> bool {
        false
    }
}

/// Video decode capabilities for a specific codec.
#[derive(Debug, Clone)]
pub struct DecodeCapabilities {
    /// Codec type.
    pub codec: Codec,
    /// Supported profiles (codec-specific).
    pub profiles: Vec<u32>,
    /// Maximum supported level.
    pub max_level: u32,
    /// Maximum width.
    pub max_width: u32,
    /// Maximum height.
    pub max_height: u32,
    /// Supported bit depths.
    pub bit_depths: Vec<u8>,
    /// Supported output formats.
    pub output_formats: Vec<GpuPixelFormat>,
}

/// Video encode capabilities for a specific codec.
#[derive(Debug, Clone)]
pub struct EncodeCapabilities {
    /// Codec type.
    pub codec: Codec,
    /// Supported profiles (codec-specific).
    pub profiles: Vec<u32>,
    /// Maximum supported level.
    pub max_level: u32,
    /// Maximum width.
    pub max_width: u32,
    /// Maximum height.
    pub max_height: u32,
    /// Supported bit depths.
    pub bit_depths: Vec<u8>,
    /// Supported rate control modes.
    pub rate_control_modes: Vec<RateControlMode>,
    /// Maximum bitrate in bits per second.
    pub max_bitrate: u64,
}

/// Rate control mode for encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RateControlMode {
    /// Constant bitrate.
    Cbr,
    /// Variable bitrate.
    Vbr,
    /// Constant quality (CRF/CQP).
    Cq,
    /// Disabled (maximum quality).
    Disabled,
}

/// Encoder configuration.
#[derive(Debug, Clone)]
pub struct EncoderConfig {
    /// Target width.
    pub width: u32,
    /// Target height.
    pub height: u32,
    /// Target framerate (numerator).
    pub framerate_num: u32,
    /// Target framerate (denominator).
    pub framerate_den: u32,
    /// Target bitrate in bits per second.
    pub bitrate: u64,
    /// Rate control mode.
    pub rate_control: RateControlMode,
    /// GOP size (keyframe interval).
    pub gop_size: u32,
    /// Number of B-frames between P-frames.
    pub b_frames: u32,
    /// Profile to use.
    pub profile: u32,
    /// Level to use.
    pub level: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            framerate_num: 30,
            framerate_den: 1,
            bitrate: 5_000_000, // 5 Mbps
            rate_control: RateControlMode::Vbr,
            gop_size: 60,
            b_frames: 0,
            profile: 100, // High profile
            level: 41,    // Level 4.1
        }
    }
}

impl EncoderConfig {
    /// Create config for H.264 at given resolution and bitrate.
    pub fn h264(width: u32, height: u32, bitrate: u64) -> Self {
        Self {
            width,
            height,
            bitrate,
            profile: 100, // High profile
            level: if width > 1920 { 51 } else { 41 },
            ..Default::default()
        }
    }

    /// Create config for H.265 at given resolution and bitrate.
    pub fn h265(width: u32, height: u32, bitrate: u64) -> Self {
        Self {
            width,
            height,
            bitrate,
            profile: 1, // Main profile
            level: if width > 1920 { 51 } else { 41 },
            ..Default::default()
        }
    }
}
