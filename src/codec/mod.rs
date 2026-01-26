//! Video codec support for Parallax.
//!
//! This module provides hardware-accelerated and software video encoding/decoding
//! through a unified abstraction.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                   Codec Abstraction Layer                       │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  VideoDecoder trait    │    VideoEncoder trait                  │
//! │  - decode()            │    - encode()                          │
//! │  - flush()             │    - flush()                           │
//! │  - output_format()     │    - codec_data()                      │
//! └────────────┬───────────┴────────────┬───────────────────────────┘
//!              │                        │
//!     ┌────────┴────────┐      ┌────────┴────────┐
//!     │  Vulkan Video   │      │  Software       │
//!     │  (GPU accel)    │      │  (rav1e/dav1d)  │
//!     └─────────────────┘      └─────────────────┘
//! ```
//!
//! # Supported Codecs
//!
//! | Codec | Vulkan Decode | Vulkan Encode | Software |
//! |-------|---------------|---------------|----------|
//! | H.264 | ✅            | ✅            | ❌       |
//! | H.265 | ✅            | ✅            | ❌       |
//! | AV1   | ✅            | ✅            | ✅ (dav1d/rav1e) |
//! | VP9   | ✅            | ❌            | ❌       |
//!
//! # Features
//!
//! - `vulkan-video`: Enable Vulkan Video hardware acceleration
//! - `software-codecs`: Enable software AV1 codecs (rav1e/dav1d)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::codec::{VideoDecoder, create_decoder, VideoCodec};
//!
//! // Create a decoder (automatically selects best backend)
//! let mut decoder = create_decoder(VideoCodec::H264)?;
//!
//! // Decode a frame
//! let frame = decoder.decode(bitstream)?;
//! ```

mod traits;

#[cfg(feature = "vulkan-video")]
pub mod vulkan;

#[cfg(feature = "software-codecs")]
pub mod software;

pub use traits::*;

/// Supported video codecs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VideoCodec {
    /// H.264/AVC
    H264,
    /// H.265/HEVC
    H265,
    /// AV1 (AOMedia Video 1)
    AV1,
    /// VP9
    VP9,
}

impl VideoCodec {
    /// Get the human-readable name of the codec.
    pub fn name(&self) -> &'static str {
        match self {
            Self::H264 => "H.264/AVC",
            Self::H265 => "H.265/HEVC",
            Self::AV1 => "AV1",
            Self::VP9 => "VP9",
        }
    }

    /// Get the common file extension for this codec.
    pub fn extension(&self) -> &'static str {
        match self {
            Self::H264 => "h264",
            Self::H265 => "h265",
            Self::AV1 => "av1",
            Self::VP9 => "vp9",
        }
    }

    /// Get the MIME type for this codec.
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::H264 => "video/avc",
            Self::H265 => "video/hevc",
            Self::AV1 => "video/av1",
            Self::VP9 => "video/vp9",
        }
    }
}

impl std::fmt::Display for VideoCodec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Create a decoder for the specified codec.
///
/// This function automatically selects the best available backend:
/// 1. Vulkan Video (if available and supported for this codec)
/// 2. Software decoder (if available)
///
/// # Errors
///
/// Returns an error if no backend supports the requested codec.
#[allow(unused_variables)]
pub fn create_decoder(codec: VideoCodec) -> crate::error::Result<Box<dyn VideoDecoder>> {
    // Try Vulkan Video first
    #[cfg(feature = "vulkan-video")]
    {
        if let Ok(instance) = vulkan::VulkanVideoInstance::new() {
            let caps = instance.supported_codecs();
            let supported = match codec {
                VideoCodec::H264 => caps.h264_decode,
                VideoCodec::H265 => caps.h265_decode,
                VideoCodec::AV1 => caps.av1_decode,
                VideoCodec::VP9 => caps.vp9_decode,
            };
            if supported {
                return vulkan::create_decoder(instance, codec);
            }
        }
    }

    // Try software fallback
    #[cfg(feature = "software-codecs")]
    {
        if codec == VideoCodec::AV1 {
            return software::create_av1_decoder();
        }
    }

    Err(crate::error::Error::Unsupported(format!(
        "No decoder available for codec: {}",
        codec
    )))
}

/// Create an encoder for the specified codec.
///
/// This function automatically selects the best available backend:
/// 1. Vulkan Video (if available and supported for this codec)
/// 2. Software encoder (if available)
///
/// # Errors
///
/// Returns an error if no backend supports the requested codec.
#[allow(unused_variables)]
pub fn create_encoder(
    codec: VideoCodec,
    config: EncoderConfig,
) -> crate::error::Result<Box<dyn VideoEncoder>> {
    // Try Vulkan Video first
    #[cfg(feature = "vulkan-video")]
    {
        if let Ok(instance) = vulkan::VulkanVideoInstance::new() {
            let caps = instance.supported_codecs();
            let supported = match codec {
                VideoCodec::H264 => caps.h264_encode,
                VideoCodec::H265 => caps.h265_encode,
                VideoCodec::AV1 => caps.av1_encode,
                VideoCodec::VP9 => false, // VP9 encode not available in Vulkan
            };
            if supported {
                return vulkan::create_encoder(instance, codec, config);
            }
        }
    }

    // Try software fallback
    #[cfg(feature = "software-codecs")]
    {
        if codec == VideoCodec::AV1 {
            return software::create_av1_encoder(config);
        }
    }

    Err(crate::error::Error::Unsupported(format!(
        "No encoder available for codec: {}",
        codec
    )))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_codec_properties() {
        assert_eq!(VideoCodec::H264.name(), "H.264/AVC");
        assert_eq!(VideoCodec::H264.extension(), "h264");
        assert_eq!(VideoCodec::H264.mime_type(), "video/avc");

        assert_eq!(VideoCodec::AV1.name(), "AV1");
        assert_eq!(VideoCodec::AV1.extension(), "av1");
        assert_eq!(VideoCodec::AV1.mime_type(), "video/av1");
    }

    #[test]
    fn test_video_codec_display() {
        assert_eq!(format!("{}", VideoCodec::H264), "H.264/AVC");
        assert_eq!(format!("{}", VideoCodec::H265), "H.265/HEVC");
    }
}
