//! Conversions from `h264_reader` parameter sets to the `vk::native`
//! `StdVideoH264*` structs Vulkan Video session parameters consume.
//!
//! The std structs carry raw pointers (`pScalingLists`,
//! `pSequenceParameterSetVui`, `pOffsetForRefFrame`); v1 leaves them null —
//! streams with scaling matrices are rejected upstream, VUI is ignorable for
//! decode, and POC type 1's offset array is only read by *our* POC code, not
//! by the driver's parameter object.

use super::error::VulkanError;
use crate::error::Result;

use ash::vk::native as std_video;
use h264_reader::nal::pps::PicParameterSet;
use h264_reader::nal::sps::SeqParameterSet;

/// Map an H.264 `profile_idc` to the std enum, rejecting profiles the decode
/// path does not handle.
pub fn std_profile_idc(profile_idc: u32) -> Result<std_video::StdVideoH264ProfileIdc> {
    match profile_idc {
        66 => Ok(std_video::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_BASELINE),
        77 => Ok(std_video::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_MAIN),
        100 => Ok(std_video::StdVideoH264ProfileIdc_STD_VIDEO_H264_PROFILE_IDC_HIGH),
        other => Err(VulkanError::UnsupportedProfile(other).into()),
    }
}

/// Map an H.264 `level_idc` (10 = 1.0 … 62 = 6.2) to the std enum.
pub fn std_level_idc(level_idc: u8) -> std_video::StdVideoH264LevelIdc {
    match level_idc {
        0..=10 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_0,
        11 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_1,
        12 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_2,
        13 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_1_3,
        14..=20 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_0,
        21 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_2_1,
        22..=30 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_0,
        31 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1,
        32..=40 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_0,
        41 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_4_1,
        42..=50 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_0,
        51 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1,
        52..=60 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_0,
        61 => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_1,
        _ => std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_6_2,
    }
}

// The level table has deliberate gaps (22→3.0, 32→4.0…): mapping unknown
// intermediate values *up* keeps the driver's conformance window at least as
// large as the stream claims to need. Exact values (30, 40, 50, 60) land on
// their own entries via the range arms.

/// Convert a full `h264_reader` SPS to `StdVideoH264SequenceParameterSet`.
///
/// Streams carrying scaling matrices are rejected — v1 does not marshal
/// `pScalingLists`, and silently dropping them would corrupt every frame.
pub fn std_sps(sps: &SeqParameterSet) -> Result<std_video::StdVideoH264SequenceParameterSet> {
    if sps.chroma_info.scaling_matrix.is_some() {
        return Err(VulkanError::DecodeError(
            "SPS carries scaling matrices, which this decoder does not support yet".to_string(),
        )
        .into());
    }

    let mut flags: std_video::StdVideoH264SpsFlags = unsafe { std::mem::zeroed() };
    let frame_mbs_only = matches!(
        sps.frame_mbs_flags,
        h264_reader::nal::sps::FrameMbsFlags::Frames
    );
    flags.set_frame_mbs_only_flag(frame_mbs_only.into());
    if let h264_reader::nal::sps::FrameMbsFlags::Fields {
        mb_adaptive_frame_field_flag,
    } = sps.frame_mbs_flags
    {
        flags.set_mb_adaptive_frame_field_flag(mb_adaptive_frame_field_flag.into());
    }
    flags.set_direct_8x8_inference_flag(sps.direct_8x8_inference_flag.into());
    flags.set_gaps_in_frame_num_value_allowed_flag(sps.gaps_in_frame_num_value_allowed_flag.into());
    flags.set_separate_colour_plane_flag(sps.chroma_info.separate_colour_plane_flag.into());
    flags.set_qpprime_y_zero_transform_bypass_flag(
        sps.chroma_info.qpprime_y_zero_transform_bypass_flag.into(),
    );
    if sps.frame_cropping.is_some() {
        flags.set_frame_cropping_flag(1);
    }
    // VUI is not marshalled; leave vui_parameters_present_flag unset even if
    // the bitstream had one — the std struct must agree with its pointers.

    let chroma_format_idc = match sps.chroma_info.chroma_format {
        h264_reader::nal::sps::ChromaFormat::Monochrome => {
            std_video::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_MONOCHROME
        }
        h264_reader::nal::sps::ChromaFormat::YUV420 => {
            std_video::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_420
        }
        h264_reader::nal::sps::ChromaFormat::YUV422 => {
            std_video::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_422
        }
        h264_reader::nal::sps::ChromaFormat::YUV444 => {
            std_video::StdVideoH264ChromaFormatIdc_STD_VIDEO_H264_CHROMA_FORMAT_IDC_444
        }
        h264_reader::nal::sps::ChromaFormat::Invalid(v) => {
            return Err(VulkanError::DecodeError(format!("Invalid chroma_format_idc {v}")).into());
        }
    };

    let (pic_order_cnt_type, log2_max_pic_order_cnt_lsb_minus4, delta_always_zero) =
        match &sps.pic_order_cnt {
            h264_reader::nal::sps::PicOrderCntType::TypeZero {
                log2_max_pic_order_cnt_lsb_minus4,
            } => (
                std_video::StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0,
                *log2_max_pic_order_cnt_lsb_minus4,
                false,
            ),
            h264_reader::nal::sps::PicOrderCntType::TypeOne {
                delta_pic_order_always_zero_flag,
                ..
            } => (
                std_video::StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_1,
                0,
                *delta_pic_order_always_zero_flag,
            ),
            h264_reader::nal::sps::PicOrderCntType::TypeTwo => (
                std_video::StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_2,
                0,
                false,
            ),
        };
    flags.set_delta_pic_order_always_zero_flag(delta_always_zero.into());

    let (offset_for_non_ref_pic, offset_for_top_to_bottom_field, num_offsets) =
        match &sps.pic_order_cnt {
            h264_reader::nal::sps::PicOrderCntType::TypeOne {
                offset_for_non_ref_pic,
                offset_for_top_to_bottom_field,
                offsets_for_ref_frame,
                ..
            } => (
                *offset_for_non_ref_pic,
                *offset_for_top_to_bottom_field,
                offsets_for_ref_frame.len() as u8,
            ),
            _ => (0, 0, 0),
        };

    let (crop_left, crop_right, crop_top, crop_bottom) = match &sps.frame_cropping {
        Some(c) => (c.left_offset, c.right_offset, c.top_offset, c.bottom_offset),
        None => (0, 0, 0, 0),
    };

    Ok(std_video::StdVideoH264SequenceParameterSet {
        flags,
        profile_idc: std_profile_idc(u8::from(sps.profile_idc) as u32)?,
        level_idc: std_level_idc(sps.level_idc),
        chroma_format_idc,
        seq_parameter_set_id: sps.seq_parameter_set_id.id(),
        bit_depth_luma_minus8: sps.chroma_info.bit_depth_luma_minus8,
        bit_depth_chroma_minus8: sps.chroma_info.bit_depth_chroma_minus8,
        log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4,
        pic_order_cnt_type,
        offset_for_non_ref_pic,
        offset_for_top_to_bottom_field,
        log2_max_pic_order_cnt_lsb_minus4,
        num_ref_frames_in_pic_order_cnt_cycle: num_offsets,
        max_num_ref_frames: sps.max_num_ref_frames as u8,
        reserved1: 0,
        pic_width_in_mbs_minus1: sps.pic_width_in_mbs_minus1,
        pic_height_in_map_units_minus1: sps.pic_height_in_map_units_minus1,
        frame_crop_left_offset: crop_left,
        frame_crop_right_offset: crop_right,
        frame_crop_top_offset: crop_top,
        frame_crop_bottom_offset: crop_bottom,
        reserved2: 0,
        // v1: POC type 1 offset arrays, scaling lists and VUI are not
        // marshalled (nulls are legal; the flags above agree).
        pOffsetForRefFrame: std::ptr::null(),
        pScalingLists: std::ptr::null(),
        pSequenceParameterSetVui: std::ptr::null(),
    })
}

/// Convert a full `h264_reader` PPS to `StdVideoH264PictureParameterSet`.
pub fn std_pps(pps: &PicParameterSet) -> Result<std_video::StdVideoH264PictureParameterSet> {
    if pps
        .extension
        .as_ref()
        .is_some_and(|e| e.pic_scaling_matrix.is_some())
    {
        return Err(VulkanError::DecodeError(
            "PPS carries scaling matrices, which this decoder does not support yet".to_string(),
        )
        .into());
    }

    let mut flags: std_video::StdVideoH264PpsFlags = unsafe { std::mem::zeroed() };
    flags.set_entropy_coding_mode_flag(pps.entropy_coding_mode_flag.into());
    flags.set_bottom_field_pic_order_in_frame_present_flag(
        pps.bottom_field_pic_order_in_frame_present_flag.into(),
    );
    flags.set_weighted_pred_flag(pps.weighted_pred_flag.into());
    flags.set_deblocking_filter_control_present_flag(
        pps.deblocking_filter_control_present_flag.into(),
    );
    flags.set_constrained_intra_pred_flag(pps.constrained_intra_pred_flag.into());
    flags.set_redundant_pic_cnt_present_flag(pps.redundant_pic_cnt_present_flag.into());
    if let Some(ext) = &pps.extension {
        flags.set_transform_8x8_mode_flag(ext.transform_8x8_mode_flag.into());
    }

    Ok(std_video::StdVideoH264PictureParameterSet {
        flags,
        seq_parameter_set_id: pps.seq_parameter_set_id.id(),
        pic_parameter_set_id: pps.pic_parameter_set_id.id(),
        num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1 as u8,
        num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1 as u8,
        weighted_bipred_idc: pps.weighted_bipred_idc as std_video::StdVideoH264WeightedBipredIdc,
        pic_init_qp_minus26: pps.pic_init_qp_minus26 as i8,
        pic_init_qs_minus26: pps.pic_init_qs_minus26 as i8,
        chroma_qp_index_offset: pps.chroma_qp_index_offset as i8,
        second_chroma_qp_index_offset: pps
            .extension
            .as_ref()
            .map(|e| e.second_chroma_qp_index_offset as i8)
            .unwrap_or(pps.chroma_qp_index_offset as i8),
        pScalingLists: std::ptr::null(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use h264_reader::nal::sps::{
        ChromaFormat, ChromaInfo, FrameMbsFlags, PicOrderCntType, SeqParamSetId,
    };

    /// A synthetic 64x64 High-profile SPS, built directly (all `h264_reader`
    /// fields are public), so the test exercises conversion, not parsing.
    fn test_sps() -> SeqParameterSet {
        SeqParameterSet {
            profile_idc: 100u8.into(),
            constraint_flags: 0u8.into(),
            level_idc: 10,
            seq_parameter_set_id: SeqParamSetId::from_u32(0).unwrap(),
            chroma_info: ChromaInfo {
                chroma_format: ChromaFormat::YUV420,
                separate_colour_plane_flag: false,
                bit_depth_luma_minus8: 0,
                bit_depth_chroma_minus8: 0,
                qpprime_y_zero_transform_bypass_flag: false,
                scaling_matrix: None,
            },
            log2_max_frame_num_minus4: 0,
            pic_order_cnt: PicOrderCntType::TypeZero {
                log2_max_pic_order_cnt_lsb_minus4: 2,
            },
            max_num_ref_frames: 4,
            gaps_in_frame_num_value_allowed_flag: false,
            pic_width_in_mbs_minus1: 3,
            pic_height_in_map_units_minus1: 3,
            frame_mbs_flags: FrameMbsFlags::Frames,
            direct_8x8_inference_flag: true,
            frame_cropping: None,
            vui_parameters: None,
        }
    }

    fn test_pps() -> PicParameterSet {
        PicParameterSet {
            pic_parameter_set_id: h264_reader::nal::pps::PicParamSetId::from_u32(0).unwrap(),
            seq_parameter_set_id: SeqParamSetId::from_u32(0).unwrap(),
            entropy_coding_mode_flag: true,
            bottom_field_pic_order_in_frame_present_flag: false,
            slice_groups: None,
            num_ref_idx_l0_default_active_minus1: 0,
            num_ref_idx_l1_default_active_minus1: 0,
            weighted_pred_flag: false,
            weighted_bipred_idc: 0,
            pic_init_qp_minus26: 0,
            pic_init_qs_minus26: 0,
            chroma_qp_index_offset: 0,
            deblocking_filter_control_present_flag: true,
            constrained_intra_pred_flag: false,
            redundant_pic_cnt_present_flag: false,
            extension: None,
        }
    }

    #[test]
    fn sps_converts_with_matching_geometry() {
        let std = std_sps(&test_sps()).expect("convertible");
        assert_eq!(std.profile_idc, 100);
        assert_eq!(std.seq_parameter_set_id, 0);
        assert_eq!((std.pic_width_in_mbs_minus1 + 1) * 16, 64);
        assert_eq!((std.pic_height_in_map_units_minus1 + 1) * 16, 64);
        assert_eq!(std.max_num_ref_frames, 4);
        assert_eq!(
            std.pic_order_cnt_type,
            std_video::StdVideoH264PocType_STD_VIDEO_H264_POC_TYPE_0
        );
        assert_eq!(std.log2_max_pic_order_cnt_lsb_minus4, 2);
        assert!(std.pScalingLists.is_null());
        assert!(std.pSequenceParameterSetVui.is_null());
    }

    #[test]
    fn sps_with_scaling_matrix_is_rejected() {
        let mut sps = test_sps();
        sps.chroma_info.scaling_matrix = Some(h264_reader::nal::sps::SeqScalingMatrix::default());
        assert!(std_sps(&sps).is_err(), "scaling lists are not marshalled");
    }

    #[test]
    fn pps_converts_against_its_sps() {
        let std = std_pps(&test_pps()).expect("convertible");
        assert_eq!(std.pic_parameter_set_id, 0);
        assert_eq!(std.seq_parameter_set_id, 0);
        assert_eq!(
            std.second_chroma_qp_index_offset, std.chroma_qp_index_offset,
            "no extension: second offset falls back to the first"
        );
        assert!(std.pScalingLists.is_null());
    }

    #[test]
    fn profile_mapping_rejects_extended() {
        assert!(std_profile_idc(88).is_err(), "Extended profile unsupported");
        assert_eq!(std_profile_idc(66).unwrap(), 66);
        assert_eq!(std_profile_idc(100).unwrap(), 100);
    }

    #[test]
    fn level_mapping_hits_known_levels() {
        assert_eq!(
            std_level_idc(31),
            std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_3_1
        );
        assert_eq!(
            std_level_idc(51),
            std_video::StdVideoH264LevelIdc_STD_VIDEO_H264_LEVEL_IDC_5_1
        );
    }
}
