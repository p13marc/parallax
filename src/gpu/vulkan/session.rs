//! Vulkan Video session management.
//!
//! Manages video sessions for decode and encode operations. A video session
//! encapsulates the state needed for hardware video processing.
//!
//! # Architecture
//!
//! Based on FFmpeg and GStreamer implementations:
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────┐
//! │                    VideoSession                              │
//! │  ┌─────────────────┐  ┌─────────────────────────────────┐  │
//! │  │ VkVideoSession  │  │ VkVideoSessionParameters        │  │
//! │  │ (Vulkan handle) │  │ (SPS/PPS for H.264, VPS for 265)│  │
//! │  └─────────────────┘  └─────────────────────────────────┘  │
//! │                                                              │
//! │  ┌─────────────────────────────────────────────────────┐   │
//! │  │              Session Memory Bindings                 │   │
//! │  │  (Driver-required memory for internal structures)    │   │
//! │  └─────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! # DPB (Decoded Picture Buffer) Configurations
//!
//! Vulkan supports three DPB modes (from Lynne's FFmpeg documentation):
//!
//! 1. **Coincident** - Output frames are usable as references (Intel)
//! 2. **Distinct Separate** - Separate DPB images per reference (AMD, NVIDIA)
//! 3. **Distinct Layered** - Single multi-layer image for all references

use super::context::VulkanContext;
use super::error::VulkanError;
use crate::error::Result;
use crate::gpu::{ChromaFormat, Codec, VideoProfile};

use ash::vk;
use std::ptr;
use std::sync::Arc;

/// Maximum DPB slots (H.264 level 5.1 max is 16)
pub const MAX_DPB_SLOTS: u32 = 17;

/// Maximum active reference pictures
pub const MAX_ACTIVE_REFS: u32 = 16;

/// Video session for decode or encode operations.
///
/// Wraps a `VkVideoSessionKHR` and manages associated resources.
pub struct VideoSession {
    /// Vulkan device.
    device: Arc<ash::Device>,
    /// Video queue extension function pointers.
    video_queue_fp: ash::khr::video_queue::DeviceFn,
    /// Video session handle.
    session: vk::VideoSessionKHR,
    /// Memory bound to the session.
    session_memory: Vec<vk::DeviceMemory>,
    /// Video profile used for this session.
    profile: VideoProfile,
    /// Coded extent (max resolution).
    coded_extent: vk::Extent2D,
    /// Picture format.
    picture_format: vk::Format,
    /// Reference picture format.
    #[allow(dead_code)]
    reference_format: vk::Format,
    /// Maximum DPB slots.
    max_dpb_slots: u32,
    /// Maximum active references.
    max_active_refs: u32,
    /// Session capabilities (queried from driver).
    capabilities: SessionCapabilities,
}

/// Session parameters (codec-specific: SPS/PPS for H.264, etc.)
pub struct VideoSessionParameters {
    /// Vulkan device.
    device: Arc<ash::Device>,
    /// Video queue extension function pointers.
    video_queue_fp: ash::khr::video_queue::DeviceFn,
    /// Session parameters handle.
    params: vk::VideoSessionParametersKHR,
    /// Parent session.
    #[allow(dead_code)]
    session: vk::VideoSessionKHR,
}

/// Capabilities queried from the driver.
#[derive(Debug, Clone, Default)]
pub struct SessionCapabilities {
    /// Minimum coded extent.
    pub min_coded_extent: vk::Extent2D,
    /// Maximum coded extent.
    pub max_coded_extent: vk::Extent2D,
    /// Maximum DPB slots supported.
    pub max_dpb_slots: u32,
    /// Maximum active reference pictures.
    pub max_active_reference_pictures: u32,
    /// Standard header version.
    pub std_header_version: vk::ExtensionProperties,
    /// Decode capability flags.
    pub decode_caps: vk::VideoDecodeCapabilityFlagsKHR,
}

/// Configuration for creating a video session.
#[derive(Debug, Clone)]
pub struct VideoSessionConfig {
    /// Video profile.
    pub profile: VideoProfile,
    /// Maximum coded width.
    pub max_width: u32,
    /// Maximum coded height.
    pub max_height: u32,
    /// Picture format (output format).
    pub picture_format: vk::Format,
    /// Reference picture format (usually same as picture format).
    pub reference_format: Option<vk::Format>,
    /// Maximum DPB slots (None = use driver max).
    pub max_dpb_slots: Option<u32>,
}

impl Default for VideoSessionConfig {
    fn default() -> Self {
        Self {
            profile: VideoProfile::default(),
            max_width: 1920,
            max_height: 1080,
            picture_format: vk::Format::G8_B8R8_2PLANE_420_UNORM, // NV12
            reference_format: None,
            max_dpb_slots: None,
        }
    }
}

impl VideoSession {
    /// Create a new video session for decoding.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Vulkan context with video support
    /// * `config` - Session configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the codec is not supported or session creation fails.
    pub fn new_decode(ctx: &VulkanContext, config: VideoSessionConfig) -> Result<Self> {
        let device = ctx.device().clone();
        let instance = ctx.instance();

        // Load video queue extension function pointers
        let video_queue_fp = ash::khr::video_queue::DeviceFn::load(|name| unsafe {
            std::mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr()))
        });

        // Query capabilities
        let capabilities = Self::query_capabilities(ctx, &config)?;

        // Determine formats and limits
        let reference_format = config.reference_format.unwrap_or(config.picture_format);
        let max_dpb_slots = config
            .max_dpb_slots
            .unwrap_or(capabilities.max_dpb_slots)
            .min(MAX_DPB_SLOTS);
        let max_active_refs = capabilities
            .max_active_reference_pictures
            .min(MAX_ACTIVE_REFS);

        let coded_extent = vk::Extent2D {
            width: config.max_width,
            height: config.max_height,
        };

        // Build video profile
        let mut h264_profile_info = vk::VideoDecodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH)
            .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);

        let (chroma_subsampling, luma_depth, chroma_depth) =
            Self::get_format_flags(&config.profile);

        let codec_op = match config.profile.codec {
            Codec::H264 => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            Codec::H265 => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
            Codec::Av1 => vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
            Codec::Vp9 => {
                return Err(VulkanError::CodecNotSupported(Codec::Vp9).into());
            }
        };

        let mut video_profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(codec_op)
            .chroma_subsampling(chroma_subsampling)
            .luma_bit_depth(luma_depth)
            .chroma_bit_depth(chroma_depth);

        if config.profile.codec == Codec::H264 {
            video_profile_info = video_profile_info.push_next(&mut h264_profile_info);
        }

        // Create video session
        let queue_family = ctx.decode_queue_family().ok_or(VulkanError::NoVideoQueue)?;

        let session_create_info = vk::VideoSessionCreateInfoKHR::default()
            .queue_family_index(queue_family)
            .video_profile(&video_profile_info)
            .picture_format(config.picture_format)
            .max_coded_extent(coded_extent)
            .reference_picture_format(reference_format)
            .max_dpb_slots(max_dpb_slots)
            .max_active_reference_pictures(max_active_refs)
            .std_header_version(&capabilities.std_header_version);

        let mut session = vk::VideoSessionKHR::null();
        let result = unsafe {
            (video_queue_fp.create_video_session_khr)(
                device.handle(),
                &session_create_info,
                ptr::null(),
                &mut session,
            )
        };

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::from(result).into());
        }

        // Bind session memory
        let session_memory = Self::bind_session_memory(ctx, &device, &video_queue_fp, session)?;

        Ok(Self {
            device,
            video_queue_fp,
            session,
            session_memory,
            profile: config.profile,
            coded_extent,
            picture_format: config.picture_format,
            reference_format,
            max_dpb_slots,
            max_active_refs,
            capabilities,
        })
    }

    /// Get format flags from profile.
    fn get_format_flags(
        profile: &VideoProfile,
    ) -> (
        vk::VideoChromaSubsamplingFlagsKHR,
        vk::VideoComponentBitDepthFlagsKHR,
        vk::VideoComponentBitDepthFlagsKHR,
    ) {
        let chroma = match profile.chroma_format {
            ChromaFormat::Yuv420 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_420,
            ChromaFormat::Yuv422 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_422,
            ChromaFormat::Yuv444 => vk::VideoChromaSubsamplingFlagsKHR::TYPE_444,
            ChromaFormat::Monochrome => vk::VideoChromaSubsamplingFlagsKHR::MONOCHROME,
        };

        let depth = if profile.bit_depth == 10 {
            vk::VideoComponentBitDepthFlagsKHR::TYPE_10
        } else {
            vk::VideoComponentBitDepthFlagsKHR::TYPE_8
        };

        (chroma, depth, depth)
    }

    /// Query capabilities for a given profile.
    fn query_capabilities(
        ctx: &VulkanContext,
        config: &VideoSessionConfig,
    ) -> Result<SessionCapabilities> {
        let instance = ctx.instance();

        // Load instance-level video queue functions
        let entry = unsafe { ash::Entry::load().map_err(VulkanError::from)? };
        let video_queue_instance_fp = ash::khr::video_queue::InstanceFn::load(|name| unsafe {
            std::mem::transmute(entry.get_instance_proc_addr(instance.handle(), name.as_ptr()))
        });

        // Build profile info for capability query
        let mut h264_profile_info = vk::VideoDecodeH264ProfileInfoKHR::default()
            .std_profile_idc(vk::native::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH)
            .picture_layout(vk::VideoDecodeH264PictureLayoutFlagsKHR::PROGRESSIVE);

        let (chroma, luma_depth, chroma_depth) = Self::get_format_flags(&config.profile);

        let codec_op = match config.profile.codec {
            Codec::H264 => vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            Codec::H265 => vk::VideoCodecOperationFlagsKHR::DECODE_H265,
            Codec::Av1 => vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
            Codec::Vp9 => {
                return Err(VulkanError::CodecNotSupported(Codec::Vp9).into());
            }
        };

        let mut video_profile_info = vk::VideoProfileInfoKHR::default()
            .video_codec_operation(codec_op)
            .chroma_subsampling(chroma)
            .luma_bit_depth(luma_depth)
            .chroma_bit_depth(chroma_depth);

        if config.profile.codec == Codec::H264 {
            video_profile_info = video_profile_info.push_next(&mut h264_profile_info);
        }

        // Query capabilities
        let mut decode_caps = vk::VideoDecodeCapabilitiesKHR::default();
        let mut video_caps = vk::VideoCapabilitiesKHR::default().push_next(&mut decode_caps);

        let result = unsafe {
            (video_queue_instance_fp.get_physical_device_video_capabilities_khr)(
                ctx.physical_device(),
                &video_profile_info,
                &mut video_caps,
            )
        };

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::from(result).into());
        }

        Ok(SessionCapabilities {
            min_coded_extent: video_caps.min_coded_extent,
            max_coded_extent: video_caps.max_coded_extent,
            max_dpb_slots: video_caps.max_dpb_slots,
            max_active_reference_pictures: video_caps.max_active_reference_pictures,
            std_header_version: video_caps.std_header_version,
            decode_caps: decode_caps.flags,
        })
    }

    /// Bind memory to the video session.
    fn bind_session_memory(
        ctx: &VulkanContext,
        device: &ash::Device,
        video_queue_fp: &ash::khr::video_queue::DeviceFn,
        session: vk::VideoSessionKHR,
    ) -> Result<Vec<vk::DeviceMemory>> {
        let instance = ctx.instance();

        // Query memory requirements count first
        let mut requirement_count = 0u32;
        let result = unsafe {
            (video_queue_fp.get_video_session_memory_requirements_khr)(
                device.handle(),
                session,
                &mut requirement_count,
                ptr::null_mut(),
            )
        };

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::from(result).into());
        }

        if requirement_count == 0 {
            return Ok(Vec::new());
        }

        // Query actual requirements
        let mut requirements =
            vec![vk::VideoSessionMemoryRequirementsKHR::default(); requirement_count as usize];
        let result = unsafe {
            (video_queue_fp.get_video_session_memory_requirements_khr)(
                device.handle(),
                session,
                &mut requirement_count,
                requirements.as_mut_ptr(),
            )
        };

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::from(result).into());
        }

        // Get memory properties
        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(ctx.physical_device()) };

        let mut memories = Vec::new();
        let mut bind_infos = Vec::new();

        for req in &requirements {
            // Find suitable memory type
            let memory_type = Self::find_memory_type(
                &memory_properties,
                req.memory_requirements.memory_type_bits,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .ok_or_else(|| VulkanError::Other("No suitable memory type for session".to_string()))?;

            // Allocate memory
            let alloc_info = vk::MemoryAllocateInfo::default()
                .allocation_size(req.memory_requirements.size)
                .memory_type_index(memory_type);

            let memory = unsafe {
                device
                    .allocate_memory(&alloc_info, None)
                    .map_err(VulkanError::from)?
            };

            memories.push(memory);

            bind_infos.push(
                vk::BindVideoSessionMemoryInfoKHR::default()
                    .memory_bind_index(req.memory_bind_index)
                    .memory(memory)
                    .memory_offset(0)
                    .memory_size(req.memory_requirements.size),
            );
        }

        // Bind all memory
        let result = unsafe {
            (video_queue_fp.bind_video_session_memory_khr)(
                device.handle(),
                session,
                bind_infos.len() as u32,
                bind_infos.as_ptr(),
            )
        };

        if result != vk::Result::SUCCESS {
            // Clean up on failure
            for memory in &memories {
                unsafe { device.free_memory(*memory, None) };
            }
            return Err(VulkanError::from(result).into());
        }

        Ok(memories)
    }

    /// Find suitable memory type.
    fn find_memory_type(
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        type_bits: u32,
        required_flags: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..memory_properties.memory_type_count {
            if (type_bits & (1 << i)) != 0 {
                let memory_type = memory_properties.memory_types[i as usize];
                if memory_type.property_flags.contains(required_flags) {
                    return Some(i);
                }
            }
        }
        None
    }

    /// Get the video session handle.
    pub fn handle(&self) -> vk::VideoSessionKHR {
        self.session
    }

    /// Get the video profile.
    pub fn profile(&self) -> &VideoProfile {
        &self.profile
    }

    /// Get the coded extent.
    pub fn coded_extent(&self) -> vk::Extent2D {
        self.coded_extent
    }

    /// Get the picture format.
    pub fn picture_format(&self) -> vk::Format {
        self.picture_format
    }

    /// Get max DPB slots.
    pub fn max_dpb_slots(&self) -> u32 {
        self.max_dpb_slots
    }

    /// Get max active references.
    pub fn max_active_refs(&self) -> u32 {
        self.max_active_refs
    }

    /// Get session capabilities.
    pub fn capabilities(&self) -> &SessionCapabilities {
        &self.capabilities
    }

    /// Get the video queue function pointers.
    pub fn video_queue_fp(&self) -> &ash::khr::video_queue::DeviceFn {
        &self.video_queue_fp
    }

    /// Get the device.
    pub fn device(&self) -> &ash::Device {
        &self.device
    }
}

impl Drop for VideoSession {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Destroy session
            (self.video_queue_fp.destroy_video_session_khr)(
                self.device.handle(),
                self.session,
                ptr::null(),
            );

            // Free session memory
            for memory in &self.session_memory {
                self.device.free_memory(*memory, None);
            }
        }
    }
}

impl VideoSessionParameters {
    /// Create empty session parameters (for later update).
    pub fn new_empty(session: &VideoSession) -> Result<Self> {
        let device = session.device.clone();
        let video_queue_fp = session.video_queue_fp.clone();

        let create_info =
            vk::VideoSessionParametersCreateInfoKHR::default().video_session(session.session);

        let mut params = vk::VideoSessionParametersKHR::null();
        let result = unsafe {
            (video_queue_fp.create_video_session_parameters_khr)(
                device.handle(),
                &create_info,
                ptr::null(),
                &mut params,
            )
        };

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::from(result).into());
        }

        Ok(Self {
            device,
            video_queue_fp,
            params,
            session: session.session,
        })
    }

    /// Get the parameters handle.
    pub fn handle(&self) -> vk::VideoSessionParametersKHR {
        self.params
    }
}

impl Drop for VideoSessionParameters {
    fn drop(&mut self) {
        unsafe {
            (self.video_queue_fp.destroy_video_session_parameters_khr)(
                self.device.handle(),
                self.params,
                ptr::null(),
            );
        }
    }
}

// Safety: VideoSession manages its own synchronization
unsafe impl Send for VideoSession {}
unsafe impl Sync for VideoSession {}

unsafe impl Send for VideoSessionParameters {}
unsafe impl Sync for VideoSessionParameters {}
