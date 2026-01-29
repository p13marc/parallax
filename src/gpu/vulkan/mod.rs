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
mod error;
mod memory;

pub use context::VulkanContext;
pub use decode::VulkanH264Decoder;
pub use error::VulkanError;
pub use memory::VulkanGpuMemory;

use ash::vk;

/// Vulkan Video extension names.
pub mod extensions {
    /// Core video queue extension.
    pub const VIDEO_QUEUE: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_queue\0") };

    /// Video decode queue extension.
    pub const VIDEO_DECODE_QUEUE: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_decode_queue\0") };

    /// Video encode queue extension.
    pub const VIDEO_ENCODE_QUEUE: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_encode_queue\0") };

    /// H.264 decode extension.
    pub const VIDEO_DECODE_H264: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_decode_h264\0") };

    /// H.265 decode extension.
    pub const VIDEO_DECODE_H265: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_decode_h265\0") };

    /// AV1 decode extension.
    pub const VIDEO_DECODE_AV1: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_decode_av1\0") };

    /// H.264 encode extension.
    pub const VIDEO_ENCODE_H264: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_encode_h264\0") };

    /// H.265 encode extension.
    pub const VIDEO_ENCODE_H265: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_video_encode_h265\0") };

    /// External memory extension.
    pub const EXTERNAL_MEMORY: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_memory\0") };

    /// External memory FD extension.
    pub const EXTERNAL_MEMORY_FD: &std::ffi::CStr =
        unsafe { std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_KHR_external_memory_fd\0") };

    /// DMA-BUF external memory extension.
    pub const EXTERNAL_MEMORY_DMABUF: &std::ffi::CStr = unsafe {
        std::ffi::CStr::from_bytes_with_nul_unchecked(b"VK_EXT_external_memory_dma_buf\0")
    };
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
