//! Vulkan Video backend for hardware video encoding/decoding.
//!
//! This module provides Vulkan Video-based implementations of the GPU codec traits.
//!
//! # Requirements
//!
//! - Vulkan 1.3+
//! - Video extensions:
//!   - `VK_KHR_video_queue`
//!   - `VK_KHR_video_decode_queue` (for decode)
//!   - `VK_KHR_video_encode_queue` (for encode)
//!   - Codec-specific extensions (H.264, H.265, AV1)
//!
//! # Supported Hardware
//!
//! - AMD: RADV driver (Mesa 23.1+)
//! - Intel: ANV driver (Mesa 23.1+)
//! - NVIDIA: Proprietary driver 525+ or NVK

mod context;
mod decode;
mod decode_commands;
mod dpb;
mod error;
mod h264_parser;
mod h264_refs;
mod h264_std;
mod memory;
mod session;

pub use context::VulkanContext;
pub use decode::VulkanH264Decoder;
pub use decode_commands::{DecodeCommandRecorder, FrameDecodeInfo, RefSlotDesc};
pub use dpb::{Dpb, DpbReference, DpbSlot};
pub use error::VulkanError;
pub use h264_parser::{H264ParameterSets, ParsedPps, ParsedSliceHeader, ParsedSps, parse_annexb};
pub use h264_refs::{
    CurrentPicture, FramePoc, PictureId, PocParams, PocState, RefPicture, RefTracker,
};
pub use memory::VulkanGpuMemory;
pub use session::{
    SessionCapabilities, VideoProfileData, VideoSession, VideoSessionConfig, VideoSessionParameters,
};

use ash::vk;

/// Vulkan Video extension names.
pub mod extensions {
    /// Core video queue extension.
    pub const VIDEO_QUEUE: &std::ffi::CStr = c"VK_KHR_video_queue";

    /// Video decode queue extension.
    pub const VIDEO_DECODE_QUEUE: &std::ffi::CStr = c"VK_KHR_video_decode_queue";

    /// Video encode queue extension.
    pub const VIDEO_ENCODE_QUEUE: &std::ffi::CStr = c"VK_KHR_video_encode_queue";

    /// H.264 decode extension.
    pub const VIDEO_DECODE_H264: &std::ffi::CStr = c"VK_KHR_video_decode_h264";

    /// H.265 decode extension.
    pub const VIDEO_DECODE_H265: &std::ffi::CStr = c"VK_KHR_video_decode_h265";

    /// AV1 decode extension.
    pub const VIDEO_DECODE_AV1: &std::ffi::CStr = c"VK_KHR_video_decode_av1";

    /// H.264 encode extension.
    pub const VIDEO_ENCODE_H264: &std::ffi::CStr = c"VK_KHR_video_encode_h264";

    /// H.265 encode extension.
    pub const VIDEO_ENCODE_H265: &std::ffi::CStr = c"VK_KHR_video_encode_h265";

    /// External memory extension.
    pub const EXTERNAL_MEMORY: &std::ffi::CStr = c"VK_KHR_external_memory";

    /// External memory FD extension.
    pub const EXTERNAL_MEMORY_FD: &std::ffi::CStr = c"VK_KHR_external_memory_fd";

    /// DMA-BUF external memory extension.
    pub const EXTERNAL_MEMORY_DMABUF: &std::ffi::CStr = c"VK_EXT_external_memory_dma_buf";
}

/// Convert Vulkan result to our error type.
#[allow(dead_code)]
pub(crate) fn check_vk_result(result: vk::Result) -> Result<(), VulkanError> {
    match result {
        vk::Result::SUCCESS => Ok(()),
        vk::Result::ERROR_OUT_OF_HOST_MEMORY => Err(VulkanError::OutOfMemory),
        vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => Err(VulkanError::OutOfMemory),
        vk::Result::ERROR_INITIALIZATION_FAILED => Err(VulkanError::InitializationFailed),
        vk::Result::ERROR_DEVICE_LOST => Err(VulkanError::DeviceLost),
        vk::Result::ERROR_EXTENSION_NOT_PRESENT => Err(VulkanError::ExtensionNotSupported),
        vk::Result::ERROR_FEATURE_NOT_PRESENT => Err(VulkanError::FeatureNotSupported),
        vk::Result::ERROR_FORMAT_NOT_SUPPORTED => Err(VulkanError::FormatNotSupported),
        _ => Err(VulkanError::Other(format!("Vulkan error: {:?}", result))),
    }
}
