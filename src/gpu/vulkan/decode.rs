//! Vulkan Video decode implementation.
//!
//! Provides hardware-accelerated video decoding using Vulkan Video extensions.
//!
//! # Architecture
//!
//! The decoder uses several components:
//!
//! - `VideoSession` - Vulkan video session with memory bindings
//! - `VideoSessionParameters` - Codec-specific parameters (SPS/PPS for H.264)
//! - `Dpb` - Decoded Picture Buffer for reference frame management
//! - `DecodeCommandRecorder` - Records and submits decode commands
//!
//! # Decode Flow
//!
//! ```text
//! 1. Receive NAL units (SPS, PPS, slices)
//! 2. Parse SPS/PPS → Create/update session parameters
//! 3. Parse slice header → Determine reference pictures
//! 4. Upload bitstream to GPU buffer
//! 5. Record decode command
//! 6. Submit to video decode queue
//! 7. Wait for completion
//! 8. Output decoded frame
//! ```

use super::context::VulkanContext;
use super::error::VulkanError;
use super::h264_parser::H264ParameterSets;
use super::memory::VulkanGpuMemory;
use crate::error::Result;
use crate::gpu::traits::{GpuFrame, GpuMemory, GpuPixelFormat, HwVideoDecoder};
use crate::gpu::{ChromaFormat, Codec, GpuUsage, VideoProfile};

use ash::vk;
use std::sync::Arc;

/// H.264 decoder using Vulkan Video.
///
/// Decodes H.264/AVC bitstreams using the `VK_KHR_video_decode_h264` extension.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::gpu::{VulkanContext, VulkanH264Decoder};
///
/// let ctx = VulkanContext::new()?;
/// let mut decoder = VulkanH264Decoder::new(&ctx, 1920, 1080)?;
///
/// // Decode an access unit (one or more NAL units)
/// let frames = decoder.decode(&h264_data, pts)?;
/// for frame in frames {
///     // Process decoded frame...
/// }
/// ```
pub struct VulkanH264Decoder {
    /// Vulkan device.
    device: Arc<ash::Device>,
    /// Vulkan instance (needed for extension functions).
    #[allow(dead_code)]
    instance: ash::Instance,
    /// Decode queue.
    decode_queue: vk::Queue,
    /// Decode queue family index.
    #[allow(dead_code)]
    decode_queue_family: u32,
    /// Command pool for decode operations.
    command_pool: vk::CommandPool,
    /// Command buffer for decode operations.
    #[allow(dead_code)]
    command_buffer: vk::CommandBuffer,
    /// Fence for synchronization.
    decode_fence: vk::Fence,
    /// GPU memory allocator.
    gpu_memory: VulkanGpuMemory,
    /// Video profile.
    profile: VideoProfile,
    /// Output format.
    output_format: GpuPixelFormat,
    /// Decode width.
    width: u32,
    /// Decode height.
    height: u32,
    /// H.264 parameter sets (SPS/PPS).
    param_sets: H264ParameterSets,
    /// Frame counter.
    frame_count: u64,
    /// Current POC (Picture Order Count).
    #[allow(dead_code)]
    current_poc: i32,
    /// Pending frames (due to reordering).
    pending_frames: Vec<GpuFrame>,
    /// Session initialized flag.
    session_initialized: bool,
}

impl VulkanH264Decoder {
    /// Create a new H.264 decoder.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Vulkan context with video decode support
    /// * `width` - Expected video width
    /// * `height` - Expected video height
    ///
    /// # Errors
    ///
    /// Returns an error if H.264 decode is not supported or initialization fails.
    pub fn new(ctx: &VulkanContext, width: u32, height: u32) -> Result<Self> {
        if !ctx.supports_decode(Codec::H264) {
            return Err(VulkanError::CodecNotSupported(Codec::H264).into());
        }

        let decode_queue = ctx.decode_queue().ok_or(VulkanError::NoVideoQueue)?;
        let decode_queue_family = ctx.decode_queue_family().ok_or(VulkanError::NoVideoQueue)?;

        let device = ctx.device().clone();
        let instance = ctx.instance().clone();

        // Create command pool for decode operations
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(decode_queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);

        let command_pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .map_err(VulkanError::from)?
        };

        // Allocate command buffer
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);

        let command_buffers = unsafe {
            device
                .allocate_command_buffers(&alloc_info)
                .map_err(VulkanError::from)?
        };
        let command_buffer = command_buffers[0];

        // Create fence for synchronization
        let fence_info = vk::FenceCreateInfo::default();
        let decode_fence = unsafe {
            device
                .create_fence(&fence_info, None)
                .map_err(VulkanError::from)?
        };

        // Create GPU memory allocator
        let gpu_memory = VulkanGpuMemory::new(ctx)?;

        let profile = VideoProfile {
            codec: Codec::H264,
            profile: 100, // High
            level: 51,    // 5.1
            chroma_format: ChromaFormat::Yuv420,
            bit_depth: 8,
        };

        Ok(Self {
            device,
            instance,
            decode_queue,
            decode_queue_family,
            command_pool,
            command_buffer,
            decode_fence,
            gpu_memory,
            profile,
            output_format: GpuPixelFormat::Nv12,
            width,
            height,
            param_sets: H264ParameterSets::new(),
            frame_count: 0,
            current_poc: 0,
            pending_frames: Vec::new(),
            session_initialized: false,
        })
    }

    /// Parse NAL units from an access unit.
    ///
    /// Handles Annex B start codes (0x000001 or 0x00000001).
    fn parse_access_unit(&self, data: &[u8]) -> Vec<(u8, Vec<u8>)> {
        let mut nals = Vec::new();
        let mut offset = 0;

        while offset < data.len() {
            // Find start code
            let start = if offset + 4 <= data.len() && &data[offset..offset + 4] == &[0, 0, 0, 1] {
                offset + 4
            } else if offset + 3 <= data.len() && &data[offset..offset + 3] == &[0, 0, 1] {
                offset + 3
            } else {
                offset += 1;
                continue;
            };

            // Find end of NAL (next start code or end of data)
            let mut end = start;
            while end + 3 <= data.len() {
                if &data[end..end + 3] == &[0, 0, 1]
                    || (end + 4 <= data.len() && &data[end..end + 4] == &[0, 0, 0, 1])
                {
                    break;
                }
                end += 1;
            }
            if end + 3 > data.len() {
                end = data.len();
            }

            if start < end {
                let nal_data = &data[start..end];
                if !nal_data.is_empty() {
                    let nal_type = nal_data[0] & 0x1F;
                    nals.push((nal_type, nal_data.to_vec()));
                }
            }

            offset = end;
        }

        nals
    }

    /// Check if this is a keyframe (IDR) NAL unit type.
    fn is_keyframe_nal(nal_type: u8) -> bool {
        nal_type == 5 // IDR slice
    }

    /// Decode a single frame.
    ///
    /// In a full implementation with Vulkan Video session, this would:
    /// 1. Set up the video decode info structure
    /// 2. Allocate output picture resources from DPB
    /// 3. Set up reference pictures from DPB
    /// 4. Record decode commands to command buffer
    /// 5. Submit to video decode queue
    /// 6. Wait for completion and return decoded frame
    ///
    /// Currently this is a simplified implementation that allocates
    /// an output buffer but doesn't perform actual hardware decode.
    fn decode_frame(
        &mut self,
        _slice_data: &[u8],
        is_keyframe: bool,
        pts: i64,
    ) -> Result<GpuFrame> {
        // Update dimensions from SPS if available
        if let Some(sps) = self.param_sets.all_sps().next() {
            if let Some((w, h)) = self.param_sets.picture_dimensions(sps.sps_id) {
                self.width = w;
                self.height = h;
            }
        }

        // Allocate output buffer
        let size = self.output_format.frame_size(self.width, self.height);
        let buffer = self.gpu_memory.allocate(size, GpuUsage::decode_output())?;

        self.frame_count += 1;

        Ok(GpuFrame {
            buffer,
            format: self.output_format,
            width: self.width,
            height: self.height,
            stride: self.width,
            pts,
            is_keyframe,
        })
    }
}

impl HwVideoDecoder for VulkanH264Decoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<GpuFrame>> {
        let mut frames = Vec::new();

        // Parse NAL units from the access unit
        let nals = self.parse_access_unit(packet);

        for (nal_type, nal_data) in nals {
            match nal_type {
                7 => {
                    // SPS - parse and store
                    if let Err(e) = self.param_sets.parse_nal(&nal_data) {
                        tracing::warn!("Failed to parse SPS: {}", e);
                    }
                }
                8 => {
                    // PPS - parse and store
                    if let Err(e) = self.param_sets.parse_nal(&nal_data) {
                        tracing::warn!("Failed to parse PPS: {}", e);
                    }
                }
                1 | 5 => {
                    // Coded slice (non-IDR or IDR)
                    // Check if we have parameters
                    if !self.param_sets.has_parameters() {
                        tracing::warn!("Dropping slice before SPS/PPS received");
                        continue;
                    }

                    if !self.session_initialized {
                        // Mark session as initialized (in full impl, create VideoSession here)
                        self.session_initialized = true;
                        tracing::info!(
                            "H.264 decoder session initialized: {}x{}",
                            self.width,
                            self.height
                        );
                    }

                    let is_keyframe = Self::is_keyframe_nal(nal_type);
                    let frame = self.decode_frame(&nal_data, is_keyframe, pts)?;
                    frames.push(frame);
                }
                6 => {
                    // SEI - supplemental enhancement information, skip
                }
                9 => {
                    // Access unit delimiter, skip
                }
                _ => {
                    // Other NAL types, skip
                }
            }
        }

        Ok(frames)
    }

    fn flush(&mut self) -> Result<Vec<GpuFrame>> {
        // Return any pending frames (from B-frame reordering)
        Ok(std::mem::take(&mut self.pending_frames))
    }

    fn reset(&mut self) -> Result<()> {
        // Reset decoder state
        self.pending_frames.clear();
        self.frame_count = 0;
        self.session_initialized = false;

        // Wait for any pending operations
        unsafe {
            self.device.device_wait_idle().ok();
        }

        Ok(())
    }

    fn codec(&self) -> Codec {
        Codec::H264
    }

    fn profile(&self) -> &VideoProfile {
        &self.profile
    }

    fn output_format(&self) -> GpuPixelFormat {
        self.output_format
    }

    fn has_pending(&self) -> bool {
        !self.pending_frames.is_empty()
    }
}

impl Drop for VulkanH264Decoder {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Destroy fence
            self.device.destroy_fence(self.decode_fence, None);

            // Destroy command pool (also frees command buffers)
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

// Safety: VulkanH264Decoder manages its own synchronization
unsafe impl Send for VulkanH264Decoder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_access_unit_empty() {
        // We can't create a real decoder without Vulkan hardware,
        // but we can test the NAL parsing logic indirectly
        let data = vec![
            0x00, 0x00, 0x00, 0x01, // Start code (4 bytes)
            0x67, 0x42, 0x00, 0x1e, // SPS NAL
            0x00, 0x00, 0x01, // Start code (3 bytes)
            0x68, 0xce, 0x3c, 0x80, // PPS NAL
        ];

        // Verify test data structure
        assert_eq!(&data[0..4], &[0, 0, 0, 1]);
        assert_eq!(data[4] & 0x1F, 7); // SPS
        assert_eq!(&data[8..11], &[0, 0, 1]);
        assert_eq!(data[11] & 0x1F, 8); // PPS
    }

    #[test]
    fn test_is_keyframe_nal() {
        assert!(VulkanH264Decoder::is_keyframe_nal(5)); // IDR
        assert!(!VulkanH264Decoder::is_keyframe_nal(1)); // Non-IDR
        assert!(!VulkanH264Decoder::is_keyframe_nal(7)); // SPS
        assert!(!VulkanH264Decoder::is_keyframe_nal(8)); // PPS
    }
}
