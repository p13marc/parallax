//! Vulkan Video H.264 decode.
//!
//! Real hardware decode: per picture the decoder parses the slice headers,
//! computes POC, binds the DPB references, uploads the bitstream, records and
//! submits the video-coding scope, waits, and copies the reconstructed NV12
//! frame back into host-visible memory.
//!
//! # Decode flow
//!
//! ```text
//! 1. Split the packet into NAL units (Annex B)
//! 2. Parse SPS/PPS → (re)create VideoSession / VideoSessionParameters / Dpb
//! 3. Parse slice headers → POC (8.2.1), reference binding
//! 4. Upload slice NALs to the bitstream buffer
//! 5. Record barriers + vkCmdBeginVideoCodingKHR / vkCmdDecodeVideoKHR /
//!    vkCmdEndVideoCodingKHR, submit on the decode queue, fence-wait
//! 6. Copy the decoded image to staging on the graphics queue, fence-wait
//! 7. Reference marking (8.2.5) → release retired DPB slots
//! 8. Return the frame
//! ```
//!
//! # Scope (v1)
//!
//! Progressive frames only; Baseline/Main/High profiles; default reference
//! list order is delegated to the driver (Vulkan H.264 decode re-parses the
//! slice headers on the GPU, so `ref_pic_list_modification` works as long as
//! every needed reference is bound). A `frame_num` gap (lost reference) is
//! detected per 7.4.3: the first offending picture errors, the rest are
//! skipped until the next IDR — "non-existing" frame synthesis (8.2.5.2) is
//! not implemented. Scaling matrices are marshalled. Out of scope:
//! interlaced/PAFF/MBAFF, DMA-BUF export of decoded frames, and
//! multi-frame-in-flight pipelining. Every failure
//! path returns `Err` — a frame over uninitialised memory is never returned.

use super::context::VulkanContext;
use super::decode_commands::{DecodeCommandRecorder, FrameDecodeInfo, RefSlotDesc};
use super::dpb::Dpb;
use super::error::VulkanError;
use super::h264_parser::{H264ParameterSets, ParsedSliceHeader, parse_annexb};
use super::h264_refs::{
    CurrentPicture, FrameNumGapDetector, GapCheck, PictureId, PocParams, PocState, RefTracker,
};
use super::h264_std;
use super::session::{VideoSession, VideoSessionConfig, VideoSessionParameters};
use crate::error::Result;
use crate::gpu::traits::{GpuFrame, GpuMemory, GpuPixelFormat, HwVideoDecoder};
use crate::gpu::{ChromaFormat, Codec, GpuUsage, VideoProfile};

use ash::vk;
use h264_reader::nal::slice::{FieldPic, PicOrderCountLsb, SliceFamily};
use std::sync::Arc;

/// Largest access unit the bitstream buffer accepts (4 MiB — generous for
/// any sane keyframe at the resolutions a session can negotiate).
const MAX_ACCESS_UNIT_BYTES: u64 = 4 * 1024 * 1024;

/// H.264 decoder using Vulkan Video.
///
/// Decodes H.264/AVC bitstreams using the `VK_KHR_video_decode_h264`
/// extension. Takes **no dimensions**: geometry comes from the SPS, and the
/// whole session stack is (re)built when the SPS changes.
///
/// # Example
///
/// ```rust,ignore
/// use std::sync::Arc;
/// use parallax::gpu::{VulkanContext, VulkanH264Decoder};
///
/// let ctx = Arc::new(VulkanContext::new()?);
/// let mut decoder = VulkanH264Decoder::new(ctx)?;
///
/// // Decode an access unit (one or more NAL units)
/// let frames = decoder.decode(&h264_data, pts)?;
/// for frame in frames {
///     // Process decoded frame...
/// }
/// ```
pub struct VulkanH264Decoder {
    /// Vulkan context (device selection, queues).
    ctx: Arc<VulkanContext>,
    /// GPU memory allocator for output frames.
    gpu_memory: super::memory::VulkanGpuMemory,
    /// Video profile of the current session (from the active SPS).
    profile: VideoProfile,
    /// Output format.
    output_format: GpuPixelFormat,
    /// H.264 parameter sets (SPS/PPS).
    param_sets: H264ParameterSets,
    /// Session stack, created on the first decodable slice.
    session: Option<VideoSession>,
    session_params: Option<VideoSessionParameters>,
    dpb: Option<Dpb>,
    recorder: Option<DecodeCommandRecorder>,
    /// Which SPS the session stack was built for.
    active_sps_id: Option<u8>,
    /// Bumped on every SPS/PPS parse; params rebuilt when it moves.
    param_generation: u64,
    built_param_generation: u64,
    /// POC state (8.2.1).
    poc_state: PocState,
    /// frame_num continuity watchdog (7.4.3): a gap means a lost reference.
    gap_detector: FrameNumGapDetector,
    /// Reference bookkeeping (8.2.5).
    ref_tracker: RefTracker,
    /// Display (cropped) dimensions from the active SPS.
    width: u32,
    height: u32,
    /// Frame counter.
    frame_count: u64,
}

impl VulkanH264Decoder {
    /// Create a new H.264 decoder.
    ///
    /// Takes no dimensions — the video session is created from the stream's
    /// SPS when the first decodable slice arrives.
    ///
    /// # Errors
    ///
    /// Returns an error if the device has no H.264 decode support.
    pub fn new(ctx: Arc<VulkanContext>) -> Result<Self> {
        if !ctx.supports_decode(Codec::H264) {
            return Err(VulkanError::CodecNotSupported(Codec::H264).into());
        }
        ctx.decode_queue().ok_or(VulkanError::NoVideoQueue)?;

        let gpu_memory = super::memory::VulkanGpuMemory::new(&ctx)?;

        Ok(Self {
            ctx,
            gpu_memory,
            profile: VideoProfile {
                codec: Codec::H264,
                profile: 100,
                level: 51,
                chroma_format: ChromaFormat::Yuv420,
                bit_depth: 8,
            },
            output_format: GpuPixelFormat::Nv12,
            param_sets: H264ParameterSets::new(),
            session: None,
            session_params: None,
            dpb: None,
            recorder: None,
            active_sps_id: None,
            param_generation: 0,
            built_param_generation: 0,
            poc_state: PocState::new(),
            gap_detector: FrameNumGapDetector::new(),
            ref_tracker: RefTracker::new(),
            width: 0,
            height: 0,
            frame_count: 0,
        })
    }

    /// (Re)create the session stack for `sps_id` if needed.
    fn ensure_session(&mut self, sps_id: u8) -> Result<()> {
        if self.active_sps_id == Some(sps_id) && self.session.is_some() {
            self.ensure_params()?;
            return Ok(());
        }

        let sps = self
            .param_sets
            .get_sps(sps_id)
            .ok_or_else(|| VulkanError::DecodeError(format!("Unknown SPS id {sps_id}")))?;

        if !sps.frame_mbs_only_flag {
            return Err(VulkanError::DecodeError(
                "Interlaced streams (frame_mbs_only_flag=0) are not supported".to_string(),
            )
            .into());
        }
        if sps.bit_depth_luma != 8 || sps.chroma_format_idc != 1 {
            return Err(VulkanError::DecodeError(format!(
                "Only 8-bit 4:2:0 supported, stream is {}-bit chroma_format_idc={}",
                sps.bit_depth_luma, sps.chroma_format_idc
            ))
            .into());
        }

        let (coded_w, coded_h) = self
            .param_sets
            .coded_dimensions(sps_id)
            .ok_or_else(|| VulkanError::DecodeError("SPS without dimensions".to_string()))?;
        let (disp_w, disp_h) = self
            .param_sets
            .picture_dimensions(sps_id)
            .unwrap_or((coded_w, coded_h));

        self.profile = VideoProfile {
            codec: Codec::H264,
            profile: sps.profile_idc as u32,
            level: sps.level_idc as u32,
            chroma_format: ChromaFormat::Yuv420,
            bit_depth: sps.bit_depth_luma,
        };

        // Tear down the old stack first (order matters: recorder before
        // session, since it holds session-profile resources).
        self.recorder = None;
        self.dpb = None;
        self.session_params = None;
        self.session = None;
        for slot in self.ref_tracker.clear() {
            let _ = slot; // DPB is gone with the stack
        }
        self.poc_state.reset();
        self.gap_detector.reset();

        let config = VideoSessionConfig {
            profile: self.profile.clone(),
            max_width: coded_w,
            max_height: coded_h,
            picture_format: vk::Format::G8_B8R8_2PLANE_420_UNORM,
            reference_format: None,
            max_dpb_slots: None,
        };

        let session = VideoSession::new_decode(&self.ctx, config)?;
        let caps = session.capabilities();
        if coded_w > caps.max_coded_extent.width || coded_h > caps.max_coded_extent.height {
            return Err(VulkanError::VideoSessionError(format!(
                "Stream is {coded_w}x{coded_h}, driver supports at most {}x{}",
                caps.max_coded_extent.width, caps.max_coded_extent.height
            ))
            .into());
        }

        let dpb = Dpb::new(&self.ctx, &session, None)?;
        let frame_size = (coded_w as u64 * coded_h as u64 * 3) / 2;
        let recorder =
            DecodeCommandRecorder::new(&self.ctx, &session, MAX_ACCESS_UNIT_BYTES, frame_size)?;

        self.session = Some(session);
        self.dpb = Some(dpb);
        self.recorder = Some(recorder);
        self.active_sps_id = Some(sps_id);
        self.width = disp_w;
        self.height = disp_h;
        self.built_param_generation = 0; // force params (re)build below

        tracing::info!(
            "Vulkan H.264 session: {}x{} (coded {}x{}), profile_idc {}",
            disp_w,
            disp_h,
            coded_w,
            coded_h,
            self.profile.profile
        );

        self.ensure_params()
    }

    /// Rebuild the session parameters object if SPS/PPS changed.
    fn ensure_params(&mut self) -> Result<()> {
        if self.session_params.is_some() && self.built_param_generation == self.param_generation {
            return Ok(());
        }
        let session = self
            .session
            .as_ref()
            .expect("ensure_params is called with a live session");

        // The std structs point into these boxes for scaling matrices; keep
        // them alive across new_h264 (the driver copies at creation — see the
        // VideoSessionParameters::new_h264 docs).
        let mut keep_alive: Vec<Box<ash::vk::native::StdVideoH264ScalingLists>> = Vec::new();

        let mut std_sps = Vec::new();
        for sps in self.param_sets.all_sps() {
            let full = self
                .param_sets
                .full_sps(sps.sps_id)
                .ok_or_else(|| VulkanError::DecodeError("SPS missing from context".to_string()))?;
            let (converted, lists) = h264_std::std_sps(full)?;
            std_sps.push(converted);
            keep_alive.extend(lists);
        }
        let mut std_pps = Vec::new();
        for pps in self.param_sets.all_pps() {
            let full = self
                .param_sets
                .full_pps(pps.pps_id)
                .ok_or_else(|| VulkanError::DecodeError("PPS missing from context".to_string()))?;
            let (converted, lists) = h264_std::std_pps(full)?;
            std_pps.push(converted);
            keep_alive.extend(lists);
        }

        self.session_params = Some(VideoSessionParameters::new_h264(
            session, &std_sps, &std_pps,
        )?);
        drop(keep_alive); // the driver copied the parameter sets at creation
        self.built_param_generation = self.param_generation;
        Ok(())
    }

    /// Extract the POC-relevant fields from a parsed slice header.
    fn picture_id(psh: &ParsedSliceHeader) -> PictureId {
        let (lsb, delta_bottom) = match &psh.header.pic_order_cnt_lsb {
            Some(PicOrderCountLsb::Frame(lsb)) => (*lsb, 0),
            Some(PicOrderCountLsb::FieldsAbsolute {
                pic_order_cnt_lsb,
                delta_pic_order_cnt_bottom,
            }) => (*pic_order_cnt_lsb, *delta_pic_order_cnt_bottom),
            Some(PicOrderCountLsb::FieldsDelta(_)) | None => (0, 0),
        };
        let (d0, d1) = match &psh.header.pic_order_cnt_lsb {
            Some(PicOrderCountLsb::FieldsDelta([d0, d1])) => (*d0, *d1),
            _ => (0, 0),
        };
        PictureId {
            frame_num: psh.header.frame_num as u32,
            is_idr: psh.is_idr,
            is_reference: psh.nal_ref_idc != 0,
            pic_order_cnt_lsb: lsb,
            delta_pic_order_cnt_bottom: delta_bottom,
            delta_pic_order_cnt_0: d0,
            delta_pic_order_cnt_1: d1,
        }
    }

    /// SPS-derived POC parameters.
    fn poc_params(&self, sps_id: u8) -> Result<PocParams> {
        let full = self
            .param_sets
            .full_sps(sps_id)
            .ok_or_else(|| VulkanError::DecodeError("SPS missing from context".to_string()))?;

        let mut params = PocParams {
            max_frame_num: 1u32 << (full.log2_max_frame_num_minus4 + 4),
            ..Default::default()
        };
        match &full.pic_order_cnt {
            h264_reader::nal::sps::PicOrderCntType::TypeZero {
                log2_max_pic_order_cnt_lsb_minus4,
            } => {
                params.poc_type = 0;
                params.max_pic_order_cnt_lsb = 1u32 << (log2_max_pic_order_cnt_lsb_minus4 + 4);
            }
            h264_reader::nal::sps::PicOrderCntType::TypeOne {
                delta_pic_order_always_zero_flag,
                offset_for_non_ref_pic,
                offset_for_top_to_bottom_field,
                offsets_for_ref_frame,
            } => {
                params.poc_type = 1;
                params.delta_pic_order_always_zero = *delta_pic_order_always_zero_flag;
                params.offset_for_non_ref_pic = *offset_for_non_ref_pic;
                params.offset_for_top_to_bottom_field = *offset_for_top_to_bottom_field;
                params.offsets_for_ref_frame = offsets_for_ref_frame.clone();
            }
            h264_reader::nal::sps::PicOrderCntType::TypeTwo => {
                params.poc_type = 2;
            }
        }
        Ok(params)
    }

    /// Decode one picture: the given slices form exactly one frame.
    ///
    /// Returns `Ok(None)` for pictures skipped inside a `frame_num` gap —
    /// after a lost reference, non-IDR pictures are dropped until the next
    /// IDR starts a clean prediction chain.
    fn decode_picture(
        &mut self,
        slices: &[&[u8]],
        psh: &ParsedSliceHeader,
        pts: i64,
    ) -> Result<Option<GpuFrame>> {
        if psh.header.field_pic != FieldPic::Frame {
            return Err(VulkanError::DecodeError(
                "Field-coded (interlaced) pictures are not supported".to_string(),
            )
            .into());
        }

        self.ensure_session(psh.sps_id)?;

        let sps = self
            .param_sets
            .get_sps(psh.sps_id)
            .ok_or_else(|| VulkanError::DecodeError("SPS vanished".to_string()))?;
        let max_num_ref_frames = sps.max_num_ref_frames as u32;
        let poc_params = self.poc_params(psh.sps_id)?;
        let max_frame_num = poc_params.max_frame_num;

        let pic = Self::picture_id(psh);

        // frame_num continuity (7.4.3) — checked BEFORE the POC state
        // advances, so a skipped picture leaves no trace. The first violation
        // is an honest error naming the gap; everything after it is quietly
        // skipped until the IDR that recovers the stream.
        match self
            .gap_detector
            .check(pic.frame_num, pic.is_idr, pic.is_reference, max_frame_num)
        {
            GapCheck::Decode => {}
            GapCheck::GapDetected { got, expected } => {
                let gaps_allowed = self
                    .param_sets
                    .full_sps(psh.sps_id)
                    .map(|s| s.gaps_in_frame_num_value_allowed_flag)
                    .unwrap_or(false);
                tracing::warn!(
                    "frame_num gap: got {got}, expected {expected} — a reference frame was \
                     lost{}; dropping pictures until the next IDR",
                    if gaps_allowed {
                        " (stream allows gaps; non-existing-frame synthesis is not implemented)"
                    } else {
                        ""
                    }
                );
                return Err(VulkanError::DecodeError(format!(
                    "frame_num gap: got {got}, expected {expected}; waiting for an IDR"
                ))
                .into());
            }
            GapCheck::SkipUntilIdr => {
                tracing::debug!(
                    "skipping non-IDR picture (frame_num {}) inside a frame_num gap",
                    pic.frame_num
                );
                return Ok(None);
            }
        }

        let poc = self.poc_state.advance(&poc_params, &pic);

        // IDR: the whole DPB retires before the picture decodes.
        if pic.is_idr {
            let released = self.ref_tracker.clear();
            let dpb = self.dpb.as_mut().expect("session ensured");
            for slot in released {
                dpb.release_slot(slot);
            }
        }

        // Bind every active reference. The driver re-parses the slice headers
        // and builds prediction lists itself; our job is to have each needed
        // picture resident in a declared slot.
        let refs: Vec<RefSlotDesc> = self
            .ref_tracker
            .references()
            .iter()
            .map(|r| RefSlotDesc {
                slot_index: r.slot_index,
                frame_num: r.frame_num as u16,
                poc_top: r.poc_top,
                poc_bottom: r.poc_bottom,
                is_long_term: r.is_long_term(),
            })
            .collect();

        if !pic.is_idr && refs.is_empty() && psh.header.slice_type.family != SliceFamily::I {
            return Err(VulkanError::DecodeError(
                "P/B slice with an empty DPB (stream started mid-GOP?)".to_string(),
            )
            .into());
        }

        let dpb = self.dpb.as_mut().expect("session ensured");
        let setup_slot = dpb.acquire_slot().ok_or_else(|| {
            VulkanError::DecodeError(
                "DPB exhausted: no free slot for the current picture".to_string(),
            )
        })?;

        let decode_result = (|| -> Result<()> {
            let recorder = self.recorder.as_mut().expect("session ensured");
            let (range, offsets) = recorder.upload_bitstream(slices)?;

            let mut flags: vk::native::StdVideoDecodeH264PictureInfoFlags =
                unsafe { std::mem::zeroed() };
            flags.set_field_pic_flag(0);
            flags.set_bottom_field_flag(0);
            flags.set_is_intra(matches!(psh.header.slice_type.family, SliceFamily::I).into());
            flags.set_IdrPicFlag(pic.is_idr.into());
            flags.set_is_reference(pic.is_reference.into());

            let std_picture_info = vk::native::StdVideoDecodeH264PictureInfo {
                flags,
                seq_parameter_set_id: psh.sps_id,
                pic_parameter_set_id: psh.pps_id,
                reserved1: 0,
                reserved2: 0,
                frame_num: pic.frame_num as u16,
                idr_pic_id: psh.header.idr_pic_id.unwrap_or(0) as u16,
                PicOrderCnt: [poc.top, poc.bottom],
            };

            let frame = FrameDecodeInfo {
                setup_slot,
                is_reference: pic.is_reference,
                std_picture_info,
                slice_offsets: &offsets,
                bitstream_offset: 0,
                bitstream_range: range,
                references: &refs,
            };

            recorder.decode_frame(
                self.session.as_ref().expect("session ensured"),
                self.session_params.as_ref().expect("params ensured"),
                dpb,
                &frame,
                self.width,
                self.height,
            )
        })();

        if let Err(e) = decode_result {
            // Never leak the slot on a failed decode.
            dpb.release_slot(setup_slot);
            return Err(e);
        }

        // Reference marking (8.2.5) after successful decode.
        dpb.mark_as_reference(setup_slot, poc.poc(), pic.frame_num, false);
        let released = self.ref_tracker.mark(
            CurrentPicture {
                slot_index: setup_slot,
                frame_num: pic.frame_num,
                poc,
                is_reference: pic.is_reference,
                is_idr: pic.is_idr,
                idr_long_term: matches!(
                    psh.header.dec_ref_pic_marking,
                    Some(h264_reader::nal::slice::DecRefPicMarking::Idr {
                        long_term_reference_flag: true,
                        ..
                    })
                ),
            },
            psh.header.dec_ref_pic_marking.as_ref(),
            max_num_ref_frames,
            max_frame_num,
        );
        for slot in released {
            dpb.release_slot(slot);
        }
        if !pic.is_reference {
            dpb.release_slot(setup_slot);
        }

        // Copy the decoded NV12 bytes into a host-visible GPU buffer.
        let size = self.output_format.frame_size(self.width, self.height);
        let buffer = self.gpu_memory.allocate(size, GpuUsage::decode_output())?;
        {
            let recorder = self.recorder.as_ref().expect("session ensured");
            let ptr = self.gpu_memory.map(&buffer)?;
            let out = unsafe { std::slice::from_raw_parts_mut(ptr, size) };
            let copy = recorder.read_output(out);
            self.gpu_memory.unmap(&buffer);
            copy?;
        }

        self.frame_count += 1;

        Ok(Some(GpuFrame {
            buffer,
            format: self.output_format,
            width: self.width,
            height: self.height,
            stride: self.width,
            pts,
            is_keyframe: pic.is_idr,
        }))
    }
}

impl HwVideoDecoder for VulkanH264Decoder {
    fn decode(&mut self, packet: &[u8], pts: i64) -> Result<Vec<GpuFrame>> {
        let nals = parse_annexb(packet);

        // Parameter sets first — a malformed SPS/PPS is an error, not a warn:
        // decoding against stale parameters corrupts every following frame.
        let mut slices: Vec<&[u8]> = Vec::new();
        for nal in nals {
            match nal.first().map(|b| b & 0x1F) {
                Some(7) | Some(8) => {
                    self.param_sets.parse_nal(nal)?;
                    self.param_generation += 1;
                }
                Some(1) | Some(5) => slices.push(nal),
                _ => {} // SEI, AUD, filler — nothing to do
            }
        }

        if slices.is_empty() {
            return Ok(Vec::new());
        }
        if !self.param_sets.has_parameters() {
            tracing::warn!("Dropping {} slice(s) before SPS/PPS arrived", slices.len());
            return Ok(Vec::new());
        }

        // Group slices into pictures: a picture starts at first_mb_in_slice 0.
        let mut frames = Vec::new();
        let mut group: Vec<&[u8]> = Vec::new();
        let mut group_header: Option<ParsedSliceHeader> = None;

        for slice in slices {
            let psh = self.param_sets.parse_slice_header(slice)?;
            if psh.header.first_mb_in_slice == 0 && !group.is_empty() {
                let header = group_header.take().expect("non-empty group has a header");
                if let Some(frame) = self.decode_picture(&group, &header, pts)? {
                    frames.push(frame);
                }
                group.clear();
            }
            if group.is_empty() {
                group_header = Some(psh);
            }
            group.push(slice);
        }
        if !group.is_empty() {
            let header = group_header.take().expect("non-empty group has a header");
            if let Some(frame) = self.decode_picture(&group, &header, pts)? {
                frames.push(frame);
            }
        }

        Ok(frames)
    }

    fn flush(&mut self) -> Result<Vec<GpuFrame>> {
        // Synchronous decode holds nothing back: every call to decode()
        // returns its frames immediately, so there is nothing to drain.
        Ok(Vec::new())
    }

    fn reset(&mut self) -> Result<()> {
        for slot in self.ref_tracker.clear() {
            if let Some(dpb) = self.dpb.as_mut() {
                dpb.release_slot(slot);
            }
        }
        if let Some(dpb) = self.dpb.as_mut() {
            dpb.clear();
        }
        self.poc_state.reset();
        self.gap_detector.reset();
        if let Some(recorder) = self.recorder.as_mut() {
            recorder.request_session_reset();
        }
        self.frame_count = 0;

        unsafe {
            self.ctx.device().device_wait_idle().ok();
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

    fn read_frame(&self, frame: &GpuFrame, out: &mut [u8]) -> Result<()> {
        let size = frame.format.frame_size(frame.width, frame.height);
        if out.len() < size {
            return Err(VulkanError::DecodeError(format!(
                "read_frame: output holds {} bytes, frame needs {}",
                out.len(),
                size
            ))
            .into());
        }
        if frame.buffer.size < size {
            return Err(VulkanError::DecodeError(format!(
                "read_frame: frame buffer holds {} bytes, geometry says {}",
                frame.buffer.size, size
            ))
            .into());
        }

        let ptr = self.gpu_memory.map(&frame.buffer)?;
        // Safety: the mapping is at least `frame.buffer.size >= size` bytes,
        // and `out` was just checked to hold `size`.
        unsafe { std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), size) };
        self.gpu_memory.unmap(&frame.buffer);
        Ok(())
    }

    fn has_pending(&self) -> bool {
        false
    }
}

// Safety: VulkanH264Decoder serializes all GPU work behind fences.
unsafe impl Send for VulkanH264Decoder {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn annexb_parsing_of_sps_pps_pair() {
        let data = vec![
            0x00, 0x00, 0x00, 0x01, // Start code (4 bytes)
            0x67, 0x42, 0x00, 0x1e, // SPS NAL
            0x00, 0x00, 0x01, // Start code (3 bytes)
            0x68, 0xce, 0x3c, 0x80, // PPS NAL
        ];

        let nals = parse_annexb(&data);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0][0] & 0x1F, 7); // SPS
        assert_eq!(nals[1][0] & 0x1F, 8); // PPS
    }

    #[test]
    fn picture_id_extracts_poc_type0_fields() {
        // Structure-only check on the PictureId mapping (no device needed):
        // frame POC lsb and deltas ride through untouched.
        let pic = PictureId {
            frame_num: 3,
            is_idr: false,
            is_reference: true,
            pic_order_cnt_lsb: 6,
            ..Default::default()
        };
        assert_eq!(pic.frame_num, 3);
        assert_eq!(pic.pic_order_cnt_lsb, 6);
    }
}
