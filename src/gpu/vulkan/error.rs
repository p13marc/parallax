//! Vulkan-specific error types.

use std::fmt;

/// Vulkan Video error type.
#[derive(Debug)]
pub enum VulkanError {
    /// Vulkan library not found.
    LibraryNotFound,
    /// No compatible GPU found.
    NoCompatibleDevice,
    /// No video decode/encode queue available.
    NoVideoQueue,
    /// Required extension not supported.
    ExtensionNotSupported,
    /// Required feature not supported.
    FeatureNotSupported,
    /// Video codec not supported on this device.
    CodecNotSupported(crate::gpu::Codec),
    /// Video format not supported.
    FormatNotSupported,
    /// Out of GPU memory.
    OutOfMemory,
    /// Vulkan initialization failed.
    InitializationFailed,
    /// GPU device lost (driver crash or device removal).
    DeviceLost,
    /// Invalid parameter.
    InvalidParameter(String),
    /// Video session error.
    VideoSessionError(String),
    /// Decode error.
    DecodeError(String),
    /// Encode error.
    EncodeError(String),
    /// DMA-BUF import/export error.
    DmaBufError(String),
    /// Other Vulkan error.
    Other(String),
}

impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LibraryNotFound => write!(f, "Vulkan library not found"),
            Self::NoCompatibleDevice => write!(f, "No Vulkan-compatible GPU found"),
            Self::ExtensionNotSupported => write!(f, "Required Vulkan extension not supported"),
            Self::FeatureNotSupported => write!(f, "Required Vulkan feature not supported"),
            Self::NoVideoQueue => write!(f, "No video decode/encode queue available"),
            Self::CodecNotSupported(codec) => {
                write!(f, "Video codec {} not supported on this device", codec)
            }
            Self::FormatNotSupported => write!(f, "Video format not supported"),
            Self::OutOfMemory => write!(f, "Out of GPU memory"),
            Self::InitializationFailed => write!(f, "Vulkan initialization failed"),
            Self::DeviceLost => write!(f, "GPU device lost"),
            Self::InvalidParameter(msg) => write!(f, "Invalid parameter: {}", msg),
            Self::VideoSessionError(msg) => write!(f, "Video session error: {}", msg),
            Self::DecodeError(msg) => write!(f, "Decode error: {}", msg),
            Self::EncodeError(msg) => write!(f, "Encode error: {}", msg),
            Self::DmaBufError(msg) => write!(f, "DMA-BUF error: {}", msg),
            Self::Other(msg) => write!(f, "Vulkan error: {}", msg),
        }
    }
}

impl std::error::Error for VulkanError {}

impl From<VulkanError> for crate::error::Error {
    fn from(e: VulkanError) -> Self {
        crate::error::Error::Element(e.to_string())
    }
}

impl From<ash::LoadingError> for VulkanError {
    fn from(_: ash::LoadingError) -> Self {
        Self::LibraryNotFound
    }
}

impl From<ash::vk::Result> for VulkanError {
    fn from(result: ash::vk::Result) -> Self {
        match result {
            ash::vk::Result::ERROR_OUT_OF_HOST_MEMORY => Self::OutOfMemory,
            ash::vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => Self::OutOfMemory,
            ash::vk::Result::ERROR_INITIALIZATION_FAILED => Self::InitializationFailed,
            ash::vk::Result::ERROR_DEVICE_LOST => Self::DeviceLost,
            ash::vk::Result::ERROR_EXTENSION_NOT_PRESENT => Self::ExtensionNotSupported,
            ash::vk::Result::ERROR_FEATURE_NOT_PRESENT => Self::FeatureNotSupported,
            ash::vk::Result::ERROR_FORMAT_NOT_SUPPORTED => Self::FormatNotSupported,
            _ => Self::Other(format!("Vulkan error: {:?}", result)),
        }
    }
}
