//! Hardware video decode.
//!
//! Two backends, complementary rather than rival, and only the *software*
//! decoders are a fallback for either:
//!
//! - `vaapi` — the path that runs on commodity Intel and AMD graphics
//!   today, and the one validated on this project's reference device
//!   (Comet Lake Gen9.5). H.264, HEVC, VP8 and VP9 decode, each checked
//!   bit-exact against its software counterpart where one exists.
//! - `vulkan` — Vulkan Video, targeting newer hardware (Gen12+ / RADV).
//!   H.264 decode is fully wired (real command submission, POC/DPB
//!   reference management, RAII GPU memory) but **unvalidated on silicon**;
//!   H.265/AV1 are enum arms and encode is an unimplemented trait. See #3.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Parallax Memory Types                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  Cpu (memfd)  │  DmaBuf  │  GpuDevice  │  GpuAccessible        │
//! └───────┬───────┴────┬─────┴──────┬──────┴─────────┬─────────────┘
//! │                     Vulkan Video Pipeline                      │
//! │  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
//! │  │ Import      │───▶│ Decode/     │───▶│ Export          │   │
//! │  │ DMA-BUF     │    │ Encode      │    │ DMA-BUF         │   │
//! │  └─────────────┘    └─────────────┘    └─────────────────┘   │
//! └───────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Feature Flags
//!
//! - `vulkan-video`: Enable Vulkan Video codec support
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::gpu::{VulkanContext, VulkanH264Decoder, Codec};
//!
//! // Check if GPU decoding is available
//! if let Ok(ctx) = VulkanContext::new() {
//!     if ctx.supports_decode(Codec::H264) {
//!         let decoder = VulkanH264Decoder::new(std::sync::Arc::new(ctx))?;
//!         // Use decoder in pipeline...
//!     }
//! }
//! ```
//!
//! # Requirements
//!
//! - Vulkan 1.3+ with video extensions
//! - GPU with video decode/encode support:
//!   - AMD: RADV driver (Mesa 23.1+)
//!   - Intel: ANV driver (Mesa 23.1+)
//!   - NVIDIA: Proprietary driver 525+ or NVK

mod traits;
pub use traits::*;

#[cfg(feature = "vulkan-video")]
pub mod vulkan;

#[cfg(feature = "vaapi")]
pub mod vaapi;

#[cfg(feature = "vaapi")]
pub use vaapi::VaDisplay;

#[cfg(feature = "vulkan-video")]
pub use vulkan::{
    DecodeCommandRecorder, Dpb, DpbReference, DpbSlot, FrameDecodeInfo, H264ParameterSets,
    ParsedPps, ParsedSliceHeader, ParsedSps, RefSlotDesc, SessionCapabilities, VideoSession,
    VideoSessionConfig, VideoSessionParameters, VulkanContext, VulkanError, VulkanGpuMemory,
    VulkanH264Decoder, parse_annexb,
};

/// Video codec types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Codec {
    /// H.264 / AVC
    H264,
    /// H.265 / HEVC
    H265,
    /// AV1
    Av1,
    /// VP9
    Vp9,
    /// VP8
    Vp8,
}

impl std::fmt::Display for Codec {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Codec::H264 => write!(f, "H.264"),
            Codec::H265 => write!(f, "H.265"),
            Codec::Av1 => write!(f, "AV1"),
            Codec::Vp9 => write!(f, "VP9"),
            Codec::Vp8 => write!(f, "VP8"),
        }
    }
}

/// Video profile for decode/encode operations.
#[derive(Debug, Clone)]
pub struct VideoProfile {
    /// Codec type.
    pub codec: Codec,
    /// Profile level (codec-specific).
    pub profile: u32,
    /// Level (codec-specific).
    pub level: u32,
    /// Chroma subsampling (420, 422, 444).
    pub chroma_format: ChromaFormat,
    /// Bit depth (8, 10, 12).
    pub bit_depth: u8,
}

impl Default for VideoProfile {
    fn default() -> Self {
        Self {
            codec: Codec::H264,
            profile: 100, // High profile
            level: 51,    // Level 5.1
            chroma_format: ChromaFormat::Yuv420,
            bit_depth: 8,
        }
    }
}

/// Chroma subsampling format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChromaFormat {
    /// Monochrome (no chroma).
    Monochrome,
    /// 4:2:0 subsampling (most common).
    #[default]
    Yuv420,
    /// 4:2:2 subsampling.
    Yuv422,
    /// 4:4:4 (no subsampling).
    Yuv444,
}

/// GPU buffer usage flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuUsage {
    /// Can be used as video decode output.
    pub decode_dst: bool,
    /// Can be used as video decode source (DPB).
    pub decode_src: bool,
    /// Can be used as video encode input.
    pub encode_src: bool,
    /// Can be used as video encode output.
    pub encode_dst: bool,
    /// Can be transferred to/from CPU.
    pub transfer: bool,
}

impl Default for GpuUsage {
    fn default() -> Self {
        Self {
            decode_dst: false,
            decode_src: false,
            encode_src: false,
            encode_dst: false,
            transfer: true,
        }
    }
}

impl GpuUsage {
    /// Usage for decode output frames.
    pub fn decode_output() -> Self {
        Self {
            decode_dst: true,
            decode_src: true, // For reference frames
            transfer: true,
            ..Default::default()
        }
    }

    /// Usage for encode input frames.
    pub fn encode_input() -> Self {
        Self {
            encode_src: true,
            transfer: true,
            ..Default::default()
        }
    }

    /// Usage for encode output (compressed data).
    pub fn encode_output() -> Self {
        Self {
            encode_dst: true,
            transfer: true,
            ..Default::default()
        }
    }
}

/// Check if Vulkan Video is available on this system.
///
/// Returns `true` if:
/// - Vulkan 1.3+ is available
/// - At least one GPU supports video decode or encode
#[cfg(feature = "vulkan-video")]
pub fn vulkan_video_available() -> bool {
    vulkan::VulkanContext::new().is_ok()
}

/// Whether VA-API hardware decode is available on this system.
///
/// A *display opens* — it says nothing about which codecs that display will
/// decode, which varies with the driver package (see [`vaapi`]). Ask
/// [`VaDisplay::supports_decode`] before choosing a hardware decoder.
#[cfg(feature = "vaapi")]
pub fn vaapi_available() -> bool {
    vaapi::VaDisplay::open().is_some()
}

/// Whether VA-API hardware decode is available on this system.
///
/// Always `false` without the `vaapi` feature.
#[cfg(not(feature = "vaapi"))]
pub fn vaapi_available() -> bool {
    false
}

#[cfg(not(feature = "vulkan-video"))]
/// Check whether Vulkan Video hardware acceleration is available on this system.
pub fn vulkan_video_available() -> bool {
    false
}
