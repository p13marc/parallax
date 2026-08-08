//! Vulkan Video decode command recording and submission.
//!
//! One `DecodeCommandRecorder` drives one video session, synchronously: per
//! picture it uploads the slice NALs, records the video-coding scope
//! (`vkCmdBeginVideoCodingKHR` → `vkCmdDecodeVideoKHR` →
//! `vkCmdEndVideoCodingKHR`) on the decode queue, fences, then copies the
//! decoded NV12 image to a host-visible staging buffer on the graphics queue
//! (decode queues are not required to support transfer operations; graphics
//! queues are). Images are created `CONCURRENT` across the two families when
//! they differ, so no ownership transfers are needed.
//!
//! Frames-in-flight pipelining is explicitly out of scope — one picture is
//! decoded and read back at a time, which is the simple-and-correct shape for
//! bring-up on real hardware.

use super::context::VulkanContext;
use super::dpb::Dpb;
use super::error::VulkanError;
use super::session::{VideoSession, VideoSessionParameters};
use crate::error::Result;

use ash::vk;
use std::ptr;

/// A reference picture bound for the current decode, described the way the
/// H.264 std structs want it.
#[derive(Debug, Clone, Copy)]
pub struct RefSlotDesc {
    /// DPB slot the reference lives in.
    pub slot_index: u32,
    /// `FrameNum` of the reference.
    pub frame_num: u16,
    /// `TopFieldOrderCnt`.
    pub poc_top: i32,
    /// `BottomFieldOrderCnt`.
    pub poc_bottom: i32,
    /// Whether the reference is long-term.
    pub is_long_term: bool,
}

/// Everything needed to decode one frame.
pub struct FrameDecodeInfo<'a> {
    /// DPB slot receiving the reconstructed picture.
    pub setup_slot: u32,
    /// Whether the picture becomes a reference (activates the setup slot).
    pub is_reference: bool,
    /// The `StdVideoDecodeH264PictureInfo` for this picture.
    pub std_picture_info: vk::native::StdVideoDecodeH264PictureInfo,
    /// Byte offsets of each slice's start code within the uploaded range.
    pub slice_offsets: &'a [u32],
    /// Offset of the bitstream in the recorder's buffer (currently always 0).
    pub bitstream_offset: u64,
    /// Aligned size of the uploaded bitstream range.
    pub bitstream_range: u64,
    /// Active references, in no particular order (lists L0/L1 order matters
    /// to the driver only through the std picture info, which Vulkan H.264
    /// decode derives itself from the DPB slot metadata).
    pub references: &'a [RefSlotDesc],
}

/// Decoder command buffer manager. See the module docs for the shape.
pub struct DecodeCommandRecorder {
    /// Vulkan device.
    device: ash::Device,
    /// Video decode queue extension function pointers.
    decode_queue_fp: ash::khr::video_decode_queue::DeviceFn,
    /// Video queue extension function pointers.
    video_queue_fp: ash::khr::video_queue::DeviceFn,
    /// Command pool on the decode queue family.
    decode_pool: vk::CommandPool,
    /// Command buffer for the video coding scope.
    decode_cmd: vk::CommandBuffer,
    /// Command pool on the graphics queue family (transfer-capable).
    transfer_pool: vk::CommandPool,
    /// Command buffer for the copy-out.
    transfer_cmd: vk::CommandBuffer,
    /// Fence reused for both submissions (they are sequential).
    fence: vk::Fence,
    /// Queues.
    decode_queue: vk::Queue,
    graphics_queue: vk::Queue,
    /// Bitstream buffer (host-visible, VIDEO_DECODE_SRC).
    bitstream_buffer: vk::Buffer,
    bitstream_memory: vk::DeviceMemory,
    bitstream_size: u64,
    bitstream_ptr: *mut u8,
    /// Host-visible staging buffer the decoded frame is copied into.
    staging_buffer: vk::Buffer,
    staging_memory: vk::DeviceMemory,
    staging_size: u64,
    staging_ptr: *mut u8,
    /// The session needs a RESET control command the first time it is used.
    needs_reset: bool,
    /// Bitstream alignment requirements from the session capabilities.
    offset_alignment: u64,
    size_alignment: u64,
}

impl DecodeCommandRecorder {
    /// Create a recorder for `session`.
    ///
    /// `max_bitstream_size` bounds one access unit; `max_frame_size` bounds
    /// the decoded NV12 frame (staging buffer size).
    pub fn new(
        ctx: &VulkanContext,
        session: &VideoSession,
        max_bitstream_size: u64,
        max_frame_size: u64,
    ) -> Result<Self> {
        let device = session.device().clone();
        let instance = ctx.instance();

        let video_queue_fp = session.video_queue_fp().clone();
        let decode_queue_fp = ash::khr::video_decode_queue::DeviceFn::load(|name| unsafe {
            std::mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr()))
        });

        let decode_family = ctx.decode_queue_family().ok_or(VulkanError::NoVideoQueue)?;
        let decode_queue = ctx.decode_queue().ok_or(VulkanError::NoVideoQueue)?;
        let graphics_family = ctx.graphics_queue_family();
        let graphics_queue = ctx.graphics_queue();

        let (decode_pool, decode_cmd) = Self::create_pool_and_buffer(&device, decode_family)?;
        let (transfer_pool, transfer_cmd) = Self::create_pool_and_buffer(&device, graphics_family)?;

        let fence_info = vk::FenceCreateInfo::default();
        let fence = unsafe {
            device
                .create_fence(&fence_info, None)
                .map_err(VulkanError::from)?
        };

        let memory_properties =
            unsafe { instance.get_physical_device_memory_properties(ctx.physical_device()) };

        let caps = session.capabilities();
        let size_alignment = caps.min_bitstream_buffer_size_alignment.max(1);
        let offset_alignment = caps.min_bitstream_buffer_offset_alignment.max(1);
        let bitstream_size = max_bitstream_size.div_ceil(size_alignment) * size_alignment;

        // The bitstream buffer is a video-usage resource: it must carry the
        // video profile list.
        let bitstream_buffer = session.profile_data().with_profile_list(|profile_list| {
            let mut profile_list = *profile_list;
            let buffer_info = vk::BufferCreateInfo::default()
                .size(bitstream_size)
                .usage(vk::BufferUsageFlags::VIDEO_DECODE_SRC_KHR)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .push_next(&mut profile_list);
            unsafe {
                device
                    .create_buffer(&buffer_info, None)
                    .map_err(VulkanError::from)
            }
        })?;
        let (bitstream_memory, bitstream_ptr) = Self::bind_host_visible(
            &device,
            &memory_properties,
            bitstream_buffer,
            bitstream_size,
        )?;

        // Plain transfer-destination staging buffer for the decoded pixels.
        let staging_info = vk::BufferCreateInfo::default()
            .size(max_frame_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let staging_buffer = unsafe {
            device
                .create_buffer(&staging_info, None)
                .map_err(VulkanError::from)?
        };
        let (staging_memory, staging_ptr) =
            Self::bind_host_visible(&device, &memory_properties, staging_buffer, max_frame_size)?;

        Ok(Self {
            device,
            decode_queue_fp,
            video_queue_fp,
            decode_pool,
            decode_cmd,
            transfer_pool,
            transfer_cmd,
            fence,
            decode_queue,
            graphics_queue,
            bitstream_buffer,
            bitstream_memory,
            bitstream_size,
            bitstream_ptr,
            staging_buffer,
            staging_memory,
            staging_size: max_frame_size,
            staging_ptr,
            needs_reset: true,
            offset_alignment,
            size_alignment,
        })
    }

    fn create_pool_and_buffer(
        device: &ash::Device,
        queue_family: u32,
    ) -> Result<(vk::CommandPool, vk::CommandBuffer)> {
        let pool_info = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let pool = unsafe {
            device
                .create_command_pool(&pool_info, None)
                .map_err(VulkanError::from)?
        };

        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let buffers = unsafe {
            device.allocate_command_buffers(&alloc_info).map_err(|e| {
                device.destroy_command_pool(pool, None);
                VulkanError::from(e)
            })?
        };
        Ok((pool, buffers[0]))
    }

    /// Allocate HOST_VISIBLE|HOST_COHERENT memory for `buffer`, bind and map it.
    fn bind_host_visible(
        device: &ash::Device,
        memory_properties: &vk::PhysicalDeviceMemoryProperties,
        buffer: vk::Buffer,
        size: u64,
    ) -> Result<(vk::DeviceMemory, *mut u8)> {
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let wanted = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
        let memory_type = (0..memory_properties.memory_type_count)
            .find(|&i| {
                (requirements.memory_type_bits & (1 << i)) != 0
                    && memory_properties.memory_types[i as usize]
                        .property_flags
                        .contains(wanted)
            })
            .ok_or_else(|| {
                VulkanError::Other("No host-visible memory type for buffer".to_string())
            })?;

        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type);
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

        let ptr = unsafe {
            device
                .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
                .map_err(|e| {
                    device.free_memory(memory, None);
                    device.destroy_buffer(buffer, None);
                    VulkanError::from(e)
                })? as *mut u8
        };

        Ok((memory, ptr))
    }

    /// Upload the slice NALs of one picture, re-emitting Annex-B start codes.
    ///
    /// Returns `(range, slice_offsets)`: the aligned byte range and the offset
    /// of each slice's start code within it, as `vkCmdDecodeVideoKHR` wants.
    pub fn upload_bitstream(&mut self, slices: &[&[u8]]) -> Result<(u64, Vec<u32>)> {
        let (total, offsets) = bitstream_layout(slices, self.size_alignment)?;
        if total > self.bitstream_size {
            return Err(VulkanError::DecodeError(format!(
                "Access unit needs {} bytes, bitstream buffer holds {}",
                total, self.bitstream_size
            ))
            .into());
        }

        let mut cursor = 0usize;
        for slice in slices {
            unsafe {
                ptr::copy_nonoverlapping(
                    START_CODE.as_ptr(),
                    self.bitstream_ptr.add(cursor),
                    START_CODE.len(),
                );
                ptr::copy_nonoverlapping(
                    slice.as_ptr(),
                    self.bitstream_ptr.add(cursor + START_CODE.len()),
                    slice.len(),
                );
            }
            cursor += START_CODE.len() + slice.len();
        }
        // Zero the alignment tail so the driver never reads stale bytes.
        unsafe {
            ptr::write_bytes(self.bitstream_ptr.add(cursor), 0, total as usize - cursor);
        }

        // HOST_COHERENT memory: no flush needed.
        let _ = self.offset_alignment; // offset stays 0, trivially aligned

        Ok((total, offsets))
    }

    /// Decode one frame synchronously and copy the result into the staging
    /// buffer. On return the decoded NV12 bytes are readable via
    /// [`read_output`](Self::read_output).
    ///
    /// `coincide` selects Intel-style DPB-and-output-coincide (the setup slot
    /// image doubles as the decode output). Distinct mode decodes into the
    /// setup slot's DPB image as well — the DPB images carry
    /// `VIDEO_DECODE_DST` usage for exactly that — which keeps v1 to one image
    /// per picture on both driver families.
    pub fn decode_frame(
        &mut self,
        session: &VideoSession,
        params: &VideoSessionParameters,
        dpb: &mut Dpb,
        frame: &FrameDecodeInfo<'_>,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let device = self.device.clone();

        // ---- Phase 1: the video coding scope, on the decode queue ----
        unsafe {
            device
                .reset_command_buffer(self.decode_cmd, vk::CommandBufferResetFlags::empty())
                .map_err(VulkanError::from)?;
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.decode_cmd, &begin_info)
                .map_err(VulkanError::from)?;
        }

        // Transition the setup image to the DPB layout, and any reference that
        // somehow is not there yet (should not happen; belt and braces).
        {
            let mut barriers = Vec::new();
            let setup = dpb
                .slot(frame.setup_slot)
                .ok_or_else(|| VulkanError::Other("Invalid setup slot".to_string()))?;
            barriers.push(image_barrier(
                setup.image,
                setup.layout,
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            ));
            for r in frame.references {
                if let Some(slot) = dpb.slot(r.slot_index)
                    && slot.layout != vk::ImageLayout::VIDEO_DECODE_DPB_KHR
                {
                    barriers.push(image_barrier(
                        slot.image,
                        slot.layout,
                        vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                    ));
                }
            }
            let dep = vk::DependencyInfo::default().image_memory_barriers(&barriers);
            unsafe { device.cmd_pipeline_barrier2(self.decode_cmd, &dep) };

            if let Some(slot) = dpb.slot_mut(frame.setup_slot) {
                slot.layout = vk::ImageLayout::VIDEO_DECODE_DPB_KHR;
            }
            for r in frame.references {
                if let Some(slot) = dpb.slot_mut(r.slot_index) {
                    slot.layout = vk::ImageLayout::VIDEO_DECODE_DPB_KHR;
                }
            }
        }

        // Per-reference std info arrays. Build each vector completely before
        // anything takes a pointer into it, and keep them alive until the
        // commands are recorded.
        let std_ref_infos: Vec<vk::native::StdVideoDecodeH264ReferenceInfo> = frame
            .references
            .iter()
            .map(|r| {
                let mut flags: vk::native::StdVideoDecodeH264ReferenceInfoFlags =
                    unsafe { std::mem::zeroed() };
                flags.set_used_for_long_term_reference(r.is_long_term.into());
                vk::native::StdVideoDecodeH264ReferenceInfo {
                    flags,
                    FrameNum: r.frame_num,
                    reserved: 0,
                    PicOrderCnt: [r.poc_top, r.poc_bottom],
                }
            })
            .collect();

        let mut h264_slot_infos: Vec<vk::VideoDecodeH264DpbSlotInfoKHR<'_>> = std_ref_infos
            .iter()
            .map(|std| vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(std))
            .collect();

        let coded_extent = session.coded_extent();
        let ref_resources: Vec<vk::VideoPictureResourceInfoKHR<'_>> = frame
            .references
            .iter()
            .map(|r| {
                let slot = dpb.slot(r.slot_index).expect("validated above");
                vk::VideoPictureResourceInfoKHR::default()
                    .image_view_binding(slot.image_view)
                    .coded_extent(coded_extent)
                    .coded_offset(vk::Offset2D { x: 0, y: 0 })
                    .base_array_layer(0)
            })
            .collect();

        // Active reference slots (real indices), used both in the begin-info
        // and in the decode-info.
        let mut ref_slots: Vec<vk::VideoReferenceSlotInfoKHR<'_>> = Vec::new();
        for ((r, resource), h264_info) in frame
            .references
            .iter()
            .zip(ref_resources.iter())
            .zip(h264_slot_infos.iter_mut())
        {
            ref_slots.push(
                vk::VideoReferenceSlotInfoKHR::default()
                    .slot_index(r.slot_index as i32)
                    .picture_resource(resource)
                    .push_next(h264_info),
            );
        }

        // The setup slot's picture resource (reconstructed picture target).
        let setup_slot_view = dpb
            .slot(frame.setup_slot)
            .expect("validated above")
            .image_view;
        let setup_resource = vk::VideoPictureResourceInfoKHR::default()
            .image_view_binding(setup_slot_view)
            .coded_extent(coded_extent)
            .coded_offset(vk::Offset2D { x: 0, y: 0 })
            .base_array_layer(0);

        // The begin-info must declare every DPB slot the scope touches: the
        // active references with their indices, plus the setup slot as -1
        // (it only becomes active through the decode op itself).
        let mut begin_slots = ref_slots.clone();
        let mut setup_std_flags: vk::native::StdVideoDecodeH264ReferenceInfoFlags =
            unsafe { std::mem::zeroed() };
        setup_std_flags.set_used_for_long_term_reference(0);
        let setup_std_info = vk::native::StdVideoDecodeH264ReferenceInfo {
            flags: setup_std_flags,
            FrameNum: frame.std_picture_info.frame_num,
            reserved: 0,
            PicOrderCnt: frame.std_picture_info.PicOrderCnt,
        };
        let mut setup_h264_info =
            vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&setup_std_info);
        begin_slots.push(
            vk::VideoReferenceSlotInfoKHR::default()
                .slot_index(-1)
                .picture_resource(&setup_resource)
                .push_next(&mut setup_h264_info),
        );

        let begin_coding_info = vk::VideoBeginCodingInfoKHR::default()
            .video_session(session.handle())
            .video_session_parameters(params.handle())
            .reference_slots(&begin_slots);

        unsafe {
            (self.video_queue_fp.cmd_begin_video_coding_khr)(self.decode_cmd, &begin_coding_info);
        }

        if self.needs_reset {
            let control_info = vk::VideoCodingControlInfoKHR::default()
                .flags(vk::VideoCodingControlFlagsKHR::RESET);
            unsafe {
                (self.video_queue_fp.cmd_control_video_coding_khr)(self.decode_cmd, &control_info);
            }
            self.needs_reset = false;
        }

        // H.264 picture info with real slice offsets.
        let mut h264_picture_info = vk::VideoDecodeH264PictureInfoKHR::default()
            .std_picture_info(&frame.std_picture_info)
            .slice_offsets(frame.slice_offsets);

        // The activated version of the setup slot (real index) for decode-info.
        let mut setup_h264_info_active =
            vk::VideoDecodeH264DpbSlotInfoKHR::default().std_reference_info(&setup_std_info);
        let setup_slot_active = vk::VideoReferenceSlotInfoKHR::default()
            .slot_index(frame.setup_slot as i32)
            .picture_resource(&setup_resource)
            .push_next(&mut setup_h264_info_active);

        let mut decode_info = vk::VideoDecodeInfoKHR::default()
            .src_buffer(self.bitstream_buffer)
            .src_buffer_offset(frame.bitstream_offset)
            .src_buffer_range(frame.bitstream_range)
            .dst_picture_resource(setup_resource)
            .setup_reference_slot(&setup_slot_active)
            .push_next(&mut h264_picture_info);
        if !ref_slots.is_empty() {
            decode_info = decode_info.reference_slots(&ref_slots);
        }

        unsafe {
            (self.decode_queue_fp.cmd_decode_video_khr)(self.decode_cmd, &decode_info);
            let end_coding_info = vk::VideoEndCodingInfoKHR::default();
            (self.video_queue_fp.cmd_end_video_coding_khr)(self.decode_cmd, &end_coding_info);
            device
                .end_command_buffer(self.decode_cmd)
                .map_err(VulkanError::from)?;
        }

        self.submit_and_wait(self.decode_cmd, self.decode_queue)?;

        // ---- Phase 2: copy-out, on the graphics queue ----
        let frame_size = nv12_frame_size(width, height);
        if frame_size as u64 > self.staging_size {
            return Err(VulkanError::DecodeError(format!(
                "Decoded frame needs {} bytes, staging buffer holds {}",
                frame_size, self.staging_size
            ))
            .into());
        }

        let decoded_image = dpb.slot(frame.setup_slot).expect("validated above").image;
        unsafe {
            device
                .reset_command_buffer(self.transfer_cmd, vk::CommandBufferResetFlags::empty())
                .map_err(VulkanError::from)?;
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(self.transfer_cmd, &begin_info)
                .map_err(VulkanError::from)?;

            let to_transfer = [image_barrier(
                decoded_image,
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            )];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&to_transfer);
            device.cmd_pipeline_barrier2(self.transfer_cmd, &dep);

            // NV12: plane 0 is full-res luma (1 byte/texel), plane 1 is
            // half-res interleaved chroma (2 bytes/texel). Tightly packed,
            // this lays out exactly as NV12 in the staging buffer.
            let regions = [
                vk::BufferImageCopy::default()
                    .buffer_offset(0)
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::PLANE_0,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width,
                        height,
                        depth: 1,
                    }),
                vk::BufferImageCopy::default()
                    .buffer_offset((width as u64) * (height as u64))
                    .buffer_row_length(0)
                    .buffer_image_height(0)
                    .image_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::PLANE_1,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .image_offset(vk::Offset3D { x: 0, y: 0, z: 0 })
                    .image_extent(vk::Extent3D {
                        width: width / 2,
                        height: height / 2,
                        depth: 1,
                    }),
            ];
            device.cmd_copy_image_to_buffer(
                self.transfer_cmd,
                decoded_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                self.staging_buffer,
                &regions,
            );

            // Back to the DPB layout: the picture may serve as a reference.
            let to_dpb = [image_barrier(
                decoded_image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::VIDEO_DECODE_DPB_KHR,
            )];
            let dep = vk::DependencyInfo::default().image_memory_barriers(&to_dpb);
            device.cmd_pipeline_barrier2(self.transfer_cmd, &dep);

            device
                .end_command_buffer(self.transfer_cmd)
                .map_err(VulkanError::from)?;
        }

        self.submit_and_wait(self.transfer_cmd, self.graphics_queue)?;

        Ok(())
    }

    fn submit_and_wait(&self, cmd: vk::CommandBuffer, queue: vk::Queue) -> Result<()> {
        let submit_info = vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&cmd));
        unsafe {
            self.device
                .reset_fences(&[self.fence])
                .map_err(VulkanError::from)?;
            self.device
                .queue_submit(queue, &[submit_info], self.fence)
                .map_err(VulkanError::from)?;
            match self
                .device
                .wait_for_fences(&[self.fence], true, DECODE_TIMEOUT_NS)
            {
                Ok(()) => Ok(()),
                Err(vk::Result::TIMEOUT) => {
                    Err(VulkanError::DecodeError("GPU decode timed out".to_string()).into())
                }
                Err(e) => Err(VulkanError::from(e).into()),
            }
        }
    }

    /// Copy the decoded NV12 frame out of the staging buffer.
    ///
    /// Valid after a successful [`decode_frame`](Self::decode_frame); `out`
    /// must hold at least `width * height * 3 / 2` bytes for that frame.
    pub fn read_output(&self, out: &mut [u8]) -> Result<()> {
        if out.len() as u64 > self.staging_size {
            return Err(VulkanError::DecodeError(
                "Output slice larger than the staging buffer".to_string(),
            )
            .into());
        }
        unsafe {
            ptr::copy_nonoverlapping(self.staging_ptr, out.as_mut_ptr(), out.len());
        }
        Ok(())
    }

    /// Ask for a session RESET before the next decode (after `reset()`).
    pub fn request_session_reset(&mut self) {
        self.needs_reset = true;
    }
}

/// 4-byte Annex-B start code re-emitted in front of every uploaded slice.
const START_CODE: [u8; 4] = [0, 0, 0, 1];

/// Fence timeout: a frame decode that takes a second is a hang, not a frame.
const DECODE_TIMEOUT_NS: u64 = 1_000_000_000;

/// Decoded NV12 byte size.
fn nv12_frame_size(width: u32, height: u32) -> usize {
    (width as usize * height as usize * 3) / 2
}

/// Compute the uploaded layout of an access unit: total aligned size and the
/// offset of each slice's start code. Pure, so it is unit-testable.
fn bitstream_layout(slices: &[&[u8]], size_alignment: u64) -> Result<(u64, Vec<u32>)> {
    if slices.is_empty() {
        return Err(VulkanError::DecodeError("Access unit contains no slices".to_string()).into());
    }
    let mut offsets = Vec::with_capacity(slices.len());
    let mut cursor = 0u64;
    for slice in slices {
        offsets.push(
            u32::try_from(cursor).map_err(|_| {
                VulkanError::DecodeError("Bitstream offset exceeds u32".to_string())
            })?,
        );
        cursor += (START_CODE.len() + slice.len()) as u64;
    }
    let alignment = size_alignment.max(1);
    let total = cursor.div_ceil(alignment) * alignment;
    Ok((total, offsets))
}

/// A full-image layout-transition barrier with video-safe scopes.
///
/// Uses `ALL_COMMANDS` on both sides: with strictly serialized, fence-waited
/// submissions the extra breadth costs nothing and avoids per-queue stage
/// validity edge cases (`VIDEO_DECODE` stages are invalid on graphics queues
/// and vice versa).
fn image_barrier(
    image: vk::Image,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) -> vk::ImageMemoryBarrier2<'static> {
    vk::ImageMemoryBarrier2::default()
        .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
        .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
        .image(image)
        .subresource_range(vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: 1,
            base_array_layer: 0,
            layer_count: 1,
        })
}

impl Drop for DecodeCommandRecorder {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();

            self.device.unmap_memory(self.bitstream_memory);
            self.device.destroy_buffer(self.bitstream_buffer, None);
            self.device.free_memory(self.bitstream_memory, None);

            self.device.unmap_memory(self.staging_memory);
            self.device.destroy_buffer(self.staging_buffer, None);
            self.device.free_memory(self.staging_memory, None);

            self.device.destroy_fence(self.fence, None);
            self.device.destroy_command_pool(self.decode_pool, None);
            self.device.destroy_command_pool(self.transfer_pool, None);
        }
    }
}

// Safety: DecodeCommandRecorder serializes all GPU work behind a fence and is
// used from one thread at a time.
unsafe impl Send for DecodeCommandRecorder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bitstream_layout_places_slices_after_start_codes() {
        let a = [0x65u8; 10];
        let b = [0x41u8; 7];
        let (total, offsets) = bitstream_layout(&[&a, &b], 1).unwrap();
        assert_eq!(offsets, vec![0, 14], "second slice after 4+10 bytes");
        assert_eq!(total, (4 + 10 + 4 + 7) as u64);
    }

    #[test]
    fn bitstream_layout_rounds_total_up_to_alignment() {
        let a = [0x65u8; 10];
        let (total, _) = bitstream_layout(&[&a], 256).unwrap();
        assert_eq!(total, 256);
    }

    #[test]
    fn bitstream_layout_rejects_empty_access_unit() {
        assert!(bitstream_layout(&[], 1).is_err());
    }

    #[test]
    fn nv12_size_is_three_halves() {
        assert_eq!(nv12_frame_size(64, 64), 64 * 64 * 3 / 2);
        assert_eq!(nv12_frame_size(1920, 1080), 1920 * 1080 * 3 / 2);
    }
}
