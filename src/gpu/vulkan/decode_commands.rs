//! Vulkan Video decode command buffer recording.
//!
//! This module handles recording decode operations to Vulkan command buffers.
//!
//! # Decode Flow
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    Decode Command Recording                     │
//! │                                                                  │
//! │  1. Begin command buffer                                        │
//! │  2. Begin video coding scope (vkCmdBeginVideoCodingKHR)        │
//! │  3. Decode inline (vkCmdDecodeVideoKHR)                        │
//! │  4. End video coding scope (vkCmdEndVideoCodingKHR)            │
//! │  5. End command buffer                                          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Reference Picture Setup
//!
//! For each decode operation, we must specify:
//! - The DPB slot for the output (decoded picture destination)
//! - Reference pictures from DPB (for inter-frame prediction)
//! - Codec-specific decode info (H.264 slice parameters, etc.)

use super::context::VulkanContext;
use super::dpb::{Dpb, DpbReference};
use super::error::VulkanError;
use super::h264_parser::{ParsedPps, ParsedSps};
use super::session::{VideoSession, VideoSessionParameters};
use crate::error::Result;

use ash::vk;
use std::ptr;

/// Decode operation for a single picture.
#[derive(Debug)]
pub struct DecodeOperation {
    /// Output slot index in the DPB.
    pub output_slot: u32,
    /// Picture Order Count for the decoded frame.
    pub poc: i32,
    /// Frame number (H.264).
    pub frame_num: u32,
    /// Is this an IDR frame?
    pub is_idr: bool,
    /// Reference list 0 (for P and B slices).
    pub ref_list_0: Vec<DpbReference>,
    /// Reference list 1 (for B slices only).
    pub ref_list_1: Vec<DpbReference>,
    /// Bitstream data (NAL unit payload).
    pub bitstream: Vec<u8>,
    /// Bitstream offset within the buffer.
    pub bitstream_offset: u64,
    /// Bitstream size.
    pub bitstream_size: u64,
}

/// H.264 specific decode parameters.
#[derive(Debug, Clone)]
pub struct H264DecodeParams {
    /// Active SPS ID.
    pub sps_id: u8,
    /// Active PPS ID.
    pub pps_id: u8,
    /// First MB in slice.
    pub first_mb_in_slice: u32,
    /// Slice type (0=P, 1=B, 2=I, etc.).
    pub slice_type: u8,
    /// Frame number from slice header.
    pub frame_num: u32,
    /// Field/frame mode.
    pub field_pic_flag: bool,
    /// Bottom field flag.
    pub bottom_field_flag: bool,
    /// IDR picture ID.
    pub idr_pic_id: u16,
    /// Picture Order Count LSB.
    pub pic_order_cnt_lsb: u16,
    /// Delta POC bottom.
    pub delta_pic_order_cnt_bottom: i32,
    /// Delta POC[0].
    pub delta_pic_order_cnt_0: i32,
    /// Delta POC[1].
    pub delta_pic_order_cnt_1: i32,
    /// Num ref idx L0 active.
    pub num_ref_idx_l0_active_minus1: u8,
    /// Num ref idx L1 active.
    pub num_ref_idx_l1_active_minus1: u8,
    /// Cabac init IDC.
    pub cabac_init_idc: u8,
    /// Slice QP delta.
    pub slice_qp_delta: i8,
    /// Disable deblocking filter IDC.
    pub disable_deblocking_filter_idc: u8,
    /// Slice alpha C0 offset.
    pub slice_alpha_c0_offset_div2: i8,
    /// Slice beta offset.
    pub slice_beta_offset_div2: i8,
}

impl Default for H264DecodeParams {
    fn default() -> Self {
        Self {
            sps_id: 0,
            pps_id: 0,
            first_mb_in_slice: 0,
            slice_type: 2, // I slice
            frame_num: 0,
            field_pic_flag: false,
            bottom_field_flag: false,
            idr_pic_id: 0,
            pic_order_cnt_lsb: 0,
            delta_pic_order_cnt_bottom: 0,
            delta_pic_order_cnt_0: 0,
            delta_pic_order_cnt_1: 0,
            num_ref_idx_l0_active_minus1: 0,
            num_ref_idx_l1_active_minus1: 0,
            cabac_init_idc: 0,
            slice_qp_delta: 0,
            disable_deblocking_filter_idc: 0,
            slice_alpha_c0_offset_div2: 0,
            slice_beta_offset_div2: 0,
        }
    }
}

/// Decoder command buffer manager.
///
/// Handles recording decode operations to Vulkan command buffers.
pub struct DecodeCommandRecorder {
    /// Vulkan device.
    device: ash::Device,
    /// Video decode queue extension function pointers.
    decode_queue_fp: ash::khr::video_decode_queue::DeviceFn,
    /// Video queue extension function pointers.
    video_queue_fp: ash::khr::video_queue::DeviceFn,
    /// Command pool for decode operations.
    command_pool: vk::CommandPool,
    /// Command buffer.
    command_buffer: vk::CommandBuffer,
    /// Fence for synchronization.
    fence: vk::Fence,
    /// Bitstream buffer (GPU-accessible).
    bitstream_buffer: vk::Buffer,
    /// Bitstream buffer memory.
    bitstream_memory: vk::DeviceMemory,
    /// Bitstream buffer size.
    bitstream_size: u64,
    /// Bitstream buffer mapped pointer.
    bitstream_ptr: *mut u8,
}

impl DecodeCommandRecorder {
    /// Create a new decode command recorder.
    ///
    /// # Arguments
    ///
    /// * `ctx` - Vulkan context
    /// * `session` - Video session to use for decoding
    /// * `queue_family` - Queue family index for the decode queue
    /// * `max_bitstream_size` - Maximum size for the bitstream buffer
    pub fn new(
        ctx: &VulkanContext,
        session: &VideoSession,
        queue_family: u32,
        max_bitstream_size: u64,
    ) -> Result<Self> {
        let device = session.device().clone();
        let instance = ctx.instance();
        let instance_fp = session.video_queue_fp();

        // Load video decode queue extension
        let video_queue_fp = instance_fp.clone();
        let decode_queue_fp = ash::khr::video_decode_queue::DeviceFn::load(|name| unsafe {
            std::mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr()))
        });

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
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

        // Create fence
        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            device
                .create_fence(&fence_info, None)
                .map_err(VulkanError::from)?
        };

        // Create bitstream buffer
        let (bitstream_buffer, bitstream_memory, bitstream_ptr) =
            Self::create_bitstream_buffer(&device, max_bitstream_size)?;

        Ok(Self {
            device,
            decode_queue_fp,
            video_queue_fp,
            command_pool,
            command_buffer,
            fence,
            bitstream_buffer,
            bitstream_memory,
            bitstream_size: max_bitstream_size,
            bitstream_ptr,
        })
    }

    /// Create the bitstream buffer for decode input.
    fn create_bitstream_buffer(
        device: &ash::Device,
        size: u64,
    ) -> Result<(vk::Buffer, vk::DeviceMemory, *mut u8)> {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer = unsafe {
            device
                .create_buffer(&buffer_info, None)
                .map_err(VulkanError::from)?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        // Allocate host-visible memory for the bitstream
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(mem_requirements.size)
            .memory_type_index(0); // Should query for HOST_VISIBLE type

        let memory = unsafe {
            device.allocate_memory(&alloc_info, None).map_err(|e| {
                device.destroy_buffer(buffer, None);
                VulkanError::from(e)
            })?
        };

        unsafe {
            device.bind_buffer_memory(buffer, memory, 0).map_err(|e| {
                device.free_memory(memory, None);
                device.destroy_buffer(buffer, None);
                VulkanError::from(e)
            })?;
        }

        // Map buffer memory
        let ptr = unsafe {
            device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| {
                    device.free_memory(memory, None);
                    device.destroy_buffer(buffer, None);
                    VulkanError::from(e)
                })? as *mut u8
        };

        Ok((buffer, memory, ptr))
    }

    /// Upload bitstream data to the GPU buffer.
    ///
    /// Returns the offset and size in the buffer.
    pub fn upload_bitstream(&mut self, data: &[u8]) -> Result<(u64, u64)> {
        if data.len() as u64 > self.bitstream_size {
            return Err(VulkanError::Other("Bitstream too large".to_string()).into());
        }

        // Copy data to mapped buffer
        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), self.bitstream_ptr, data.len());
        }

        // Flush to make visible to GPU
        let flush_range = vk::MappedMemoryRange::default()
            .memory(self.bitstream_memory)
            .offset(0)
            .size(vk::WHOLE_SIZE);

        unsafe {
            self.device
                .flush_mapped_memory_ranges(&[flush_range])
                .map_err(VulkanError::from)?;
        }

        Ok((0, data.len() as u64))
    }

    /// Record H.264 decode commands.
    ///
    /// # Arguments
    ///
    /// * `session` - Video session
    /// * `params` - Session parameters (SPS/PPS)
    /// * `dpb` - Decoded Picture Buffer
    /// * `operation` - Decode operation parameters
    /// * `h264_params` - H.264 specific parameters
    /// * `sps` - Active SPS
    /// * `pps` - Active PPS
    #[allow(clippy::too_many_arguments)]
    pub fn record_h264_decode(
        &mut self,
        session: &VideoSession,
        params: &VideoSessionParameters,
        dpb: &Dpb,
        operation: &DecodeOperation,
        h264_params: &H264DecodeParams,
        _sps: &ParsedSps,
        _pps: &ParsedPps,
    ) -> Result<()> {
        let device = &self.device;

        // Reset command buffer
        unsafe {
            device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())
                .map_err(VulkanError::from)?;
        }

        // Begin command buffer
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        unsafe {
            device
                .begin_command_buffer(self.command_buffer, &begin_info)
                .map_err(VulkanError::from)?;
        }

        // Set up reference pictures
        let ref_slots =
            self.build_reference_slots(dpb, &operation.ref_list_0, &operation.ref_list_1);

        // Get output slot
        let output_slot = dpb
            .slot(operation.output_slot)
            .ok_or_else(|| VulkanError::Other("Invalid output slot".to_string()))?;

        // Build H.264 picture info using the native type
        let std_picture_info = self.build_std_h264_picture_info(h264_params);

        // Build the Vulkan H.264 picture info
        // Note: slice_count and slice_offsets are fields, not methods in ash 0.38
        let mut h264_picture_info =
            vk::VideoDecodeH264PictureInfoKHR::default().std_picture_info(&std_picture_info);
        // Set slice count directly on the struct
        h264_picture_info.slice_count = 1;

        // Build output picture resource
        let output_picture_resource = vk::VideoPictureResourceInfoKHR::default()
            .image_view_binding(output_slot.image_view)
            .coded_extent(session.coded_extent())
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .base_array_layer(0);

        // Build setup reference slot
        let setup_slot = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(operation.output_slot as i32)
            .picture_resource(&output_picture_resource);

        // Build decode info
        let mut decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.bitstream_buffer)
            .src_buffer_offset(operation.bitstream_offset)
            .src_buffer_range(operation.bitstream_size)
            .dst_picture_resource(output_picture_resource)
            .setup_reference_slot(&setup_slot)
            .push_next(&mut h264_picture_info);

        if !ref_slots.is_empty() {
            decode_info = decode_info.reference_slots(&ref_slots);
        }

        // Begin video coding scope
        let begin_coding_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(session.handle())
            .video_session_parameters(params.handle());

        unsafe {
            (self.video_queue_fp.cmd_begin_video_coding_khr)(
                self.command_buffer,
                &begin_coding_info,
            );
        }

        // Issue decode command
        unsafe {
            (self.decode_queue_fp.cmd_decode_video_khr)(self.command_buffer, &decode_info);
        }

        // End video coding scope
        let end_coding_info = vk::VideoEndCodingInfoKHR::default();
        unsafe {
            (self.video_queue_fp.cmd_end_video_coding_khr)(self.command_buffer, &end_coding_info);
        }

        // End command buffer
        unsafe {
            device
                .end_command_buffer(self.command_buffer)
                .map_err(VulkanError::from)?;
        }

        Ok(())
    }

    /// Build reference slot info from DPB references.
    fn build_reference_slots(
        &self,
        dpb: &Dpb,
        ref_list_0: &[DpbReference],
        ref_list_1: &[DpbReference],
    ) -> Vec<vk::VideoReferenceSlotInfoKHR<'static>> {
        let mut slots = Vec::new();

        for r in ref_list_0.iter().chain(ref_list_1.iter()) {
            if let Some(_slot) = dpb.slot(r.slot_index) {
                // Note: This creates temporary objects that don't live long enough.
                // In a real implementation, we'd need to maintain these structures
                // with proper lifetimes.
                let slot_info =
                    vk::VideoReferenceSlotInfoKHR::default().slot_index(r.slot_index as i32);
                slots.push(slot_info);
            }
        }

        slots
    }

    /// Build StdVideoDecodeH264PictureInfo from decode params.
    fn build_std_h264_picture_info(
        &self,
        params: &H264DecodeParams,
    ) -> vk::native::StdVideoDecodeH264PictureInfo {
        // Create flags manually since Default may not be implemented
        let mut flags: vk::native::StdVideoDecodeH264PictureInfoFlags =
            unsafe { std::mem::zeroed() };

        if params.field_pic_flag {
            flags.set_field_pic_flag(1);
        }
        if params.bottom_field_flag {
            flags.set_bottom_field_flag(1);
        }

        vk::native::StdVideoDecodeH264PictureInfo {
            flags,
            seq_parameter_set_id: params.sps_id,
            pic_parameter_set_id: params.pps_id,
            reserved1: 0,
            reserved2: 0,
            frame_num: params.frame_num as u16,
            idr_pic_id: params.idr_pic_id,
            PicOrderCnt: [
                params.pic_order_cnt_lsb as i32,
                params.delta_pic_order_cnt_bottom,
            ],
        }
    }

    /// Submit the recorded commands to the decode queue.
    pub fn submit(&mut self, queue: vk::Queue) -> Result<()> {
        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));

        unsafe {
            // Reset fence
            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::from)?;

            // Submit
            self.device
                .queue_submit(queue, &[submit_info], self.fence)
                .map_err(VulkanError::from)?;
        }

        Ok(())
    }

    /// Wait for decode completion.
    pub fn wait(&self, timeout_ns: u64) -> Result<bool> {
        let result = unsafe { self.device.wait_for_fences(&[self.fence], true, timeout_ns) };

        match result {
            Ok(()) => Ok(true),
            Err(vk::Result::TIMEOUT) => Ok(false),
            Err(e) => Err(VulkanError::from(e).into()),
        }
    }

    /// Get the command buffer handle.
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.command_buffer
    }

    /// Get the fence handle.
    pub fn fence(&self) -> vk::Fence {
        self.fence
    }
}

impl Drop for DecodeCommandRecorder {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            // Unmap and free bitstream buffer
            self.device.unmap_memory(self.bitstream_memory);
            self.device.destroy_buffer(self.bitstream_buffer, None);
            self.device.free_memory(self.bitstream_memory, None);

            // Destroy fence and command pool
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

// Safety: DecodeCommandRecorder manages its own synchronization
unsafe impl Send for DecodeCommandRecorder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_h264_decode_params_default() {
        let params = H264DecodeParams::default();
        assert_eq!(params.slice_type, 2); // I slice
        assert!(!params.field_pic_flag);
    }

    #[test]
    fn test_decode_operation() {
        let op = DecodeOperation {
            output_slot: 0,
            poc: 0,
            frame_num: 0,
            is_idr: true,
            ref_list_0: vec![],
            ref_list_1: vec![],
            bitstream: vec![],
            bitstream_offset: 0,
            bitstream_size: 0,
        };
        assert!(op.is_idr);
        assert!(op.ref_list_0.is_empty());
    }
}
