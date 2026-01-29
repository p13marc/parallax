//! H.264 bitstream parsing for Vulkan Video.
//!
//! Parses H.264 NAL units to extract SPS/PPS data needed for Vulkan Video
//! session parameter creation.

use super::error::VulkanError;
use crate::error::Result;

use h264_reader::Context;
use h264_reader::nal::UnitType;
use h264_reader::nal::pps::PicParameterSet;
use h264_reader::nal::sps::{ChromaFormat, SeqParameterSet};
use h264_reader::rbsp::BitReader;

/// Parsed H.264 parameter sets.
pub struct H264ParameterSets {
    /// Sequence Parameter Sets (indexed by sps_id).
    sps: Vec<Option<ParsedSps>>,
    /// Picture Parameter Sets (indexed by pps_id).
    pps: Vec<Option<ParsedPps>>,
    /// h264_reader context for parsing.
    context: Context,
}

impl std::fmt::Debug for H264ParameterSets {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H264ParameterSets")
            .field(
                "sps_count",
                &self.sps.iter().filter(|s| s.is_some()).count(),
            )
            .field(
                "pps_count",
                &self.pps.iter().filter(|p| p.is_some()).count(),
            )
            .finish()
    }
}

/// Parsed SPS with both raw bytes and decoded fields.
#[derive(Debug, Clone)]
pub struct ParsedSps {
    /// Raw NAL bytes (without start code, with NAL header).
    pub raw_bytes: Vec<u8>,
    /// SPS ID (0-31).
    pub sps_id: u8,
    /// Profile IDC (66=Baseline, 77=Main, 100=High, etc.).
    pub profile_idc: u8,
    /// Level IDC (e.g., 51 for level 5.1).
    pub level_idc: u8,
    /// Constraint flags.
    pub constraint_flags: u8,
    /// Chroma format (0=mono, 1=420, 2=422, 3=444).
    pub chroma_format_idc: u8,
    /// Bit depth for luma (8, 9, 10, 11, 12, 13, or 14).
    pub bit_depth_luma: u8,
    /// Bit depth for chroma.
    pub bit_depth_chroma: u8,
    /// Picture width in macroblocks.
    pub pic_width_in_mbs: u32,
    /// Picture height in map units.
    pub pic_height_in_map_units: u32,
    /// Frame MBs only flag.
    pub frame_mbs_only_flag: bool,
    /// Max number of reference frames.
    pub max_num_ref_frames: u8,
    /// log2_max_frame_num_minus4.
    pub log2_max_frame_num_minus4: u8,
    /// Picture order count type.
    pub pic_order_cnt_type: u8,
    /// log2_max_pic_order_cnt_lsb_minus4 (for poc_type 0).
    pub log2_max_pic_order_cnt_lsb_minus4: u8,
    /// Delta pic order always zero flag (for poc_type 1).
    pub delta_pic_order_always_zero_flag: bool,
}

/// Parsed PPS with both raw bytes and key fields.
#[derive(Debug, Clone)]
pub struct ParsedPps {
    /// Raw NAL bytes (without start code, with NAL header).
    pub raw_bytes: Vec<u8>,
    /// PPS ID (0-255).
    pub pps_id: u8,
    /// Referenced SPS ID.
    pub sps_id: u8,
    /// Entropy coding mode (0=CAVLC, 1=CABAC).
    pub entropy_coding_mode_flag: bool,
    /// Bottom field pic order in frame present flag.
    pub bottom_field_pic_order_in_frame_present_flag: bool,
    /// Number of ref idx L0 default active minus 1.
    pub num_ref_idx_l0_default_active_minus1: u8,
    /// Number of ref idx L1 default active minus 1.
    pub num_ref_idx_l1_default_active_minus1: u8,
    /// Weighted prediction flag.
    pub weighted_pred_flag: bool,
    /// Weighted bipred IDC.
    pub weighted_bipred_idc: u8,
    /// Initial QP minus 26.
    pub pic_init_qp_minus26: i8,
    /// Initial QS minus 26.
    pub pic_init_qs_minus26: i8,
    /// Chroma QP index offset.
    pub chroma_qp_index_offset: i8,
    /// Deblocking filter control present flag.
    pub deblocking_filter_control_present_flag: bool,
    /// Constrained intra prediction flag.
    pub constrained_intra_pred_flag: bool,
    /// Redundant pic cnt present flag.
    pub redundant_pic_cnt_present_flag: bool,
    /// Transform 8x8 mode flag (High profile).
    pub transform_8x8_mode_flag: bool,
    /// Second chroma QP index offset.
    pub second_chroma_qp_index_offset: i8,
}

impl Default for H264ParameterSets {
    fn default() -> Self {
        Self::new()
    }
}

impl H264ParameterSets {
    /// Create a new empty parameter set storage.
    pub fn new() -> Self {
        Self {
            sps: vec![None; 32],  // H.264 allows up to 32 SPS
            pps: vec![None; 256], // H.264 allows up to 256 PPS
            context: Context::default(),
        }
    }

    /// Parse a NAL unit and extract any SPS/PPS.
    ///
    /// Returns the NAL unit type.
    pub fn parse_nal(&mut self, nal_data: &[u8]) -> Result<UnitType> {
        if nal_data.is_empty() {
            return Err(VulkanError::DecodeError("Empty NAL unit".to_string()).into());
        }

        let nal_header = nal_data[0];
        let nal_type_val = nal_header & 0x1F;

        let unit_type = match nal_type_val {
            1 => UnitType::SliceLayerWithoutPartitioningNonIdr,
            5 => UnitType::SliceLayerWithoutPartitioningIdr,
            6 => UnitType::SEI,
            7 => UnitType::SeqParameterSet,
            8 => UnitType::PicParameterSet,
            9 => UnitType::AccessUnitDelimiter,
            _ => UnitType::Unspecified(nal_type_val),
        };

        match unit_type {
            UnitType::SeqParameterSet => {
                self.parse_sps(nal_data)?;
            }
            UnitType::PicParameterSet => {
                self.parse_pps(nal_data)?;
            }
            _ => {}
        }

        Ok(unit_type)
    }

    /// Parse an SPS NAL unit.
    fn parse_sps(&mut self, nal_data: &[u8]) -> Result<()> {
        if nal_data.len() < 4 {
            return Err(VulkanError::DecodeError("SPS too short".to_string()).into());
        }

        // Create a BitReader from the NAL data (skip NAL header byte)
        let rbsp_data = &nal_data[1..];
        let reader = BitReader::new(rbsp_data);

        // Parse using h264_reader
        let sps = SeqParameterSet::from_bits(reader)
            .map_err(|e| VulkanError::DecodeError(format!("Failed to parse SPS: {:?}", e)))?;

        // Extract key fields
        let sps_id = sps.seq_parameter_set_id.id() as u8;
        let profile_idc: u8 = sps.profile_idc.into();
        let level_idc = sps.level_idc;
        let constraint_flags: u8 = sps.constraint_flags.into();

        let chroma_format_idc = match sps.chroma_info.chroma_format {
            ChromaFormat::Monochrome => 0,
            ChromaFormat::YUV420 => 1,
            ChromaFormat::YUV422 => 2,
            ChromaFormat::YUV444 => 3,
            ChromaFormat::Invalid(_) => 1, // Default to 420
        };

        let bit_depth_luma = sps.chroma_info.bit_depth_luma_minus8 + 8;
        let bit_depth_chroma = sps.chroma_info.bit_depth_chroma_minus8 + 8;

        let (
            log2_max_pic_order_cnt_lsb_minus4,
            delta_pic_order_always_zero_flag,
            pic_order_cnt_type,
        ) = match &sps.pic_order_cnt {
            h264_reader::nal::sps::PicOrderCntType::TypeZero {
                log2_max_pic_order_cnt_lsb_minus4,
            } => (*log2_max_pic_order_cnt_lsb_minus4 as u8, false, 0u8),
            h264_reader::nal::sps::PicOrderCntType::TypeOne {
                delta_pic_order_always_zero_flag,
                ..
            } => (0, *delta_pic_order_always_zero_flag, 1u8),
            h264_reader::nal::sps::PicOrderCntType::TypeTwo => (0, false, 2u8),
        };

        let parsed = ParsedSps {
            raw_bytes: nal_data.to_vec(),
            sps_id,
            profile_idc,
            level_idc,
            constraint_flags,
            chroma_format_idc,
            bit_depth_luma,
            bit_depth_chroma,
            pic_width_in_mbs: sps.pic_width_in_mbs_minus1 + 1,
            pic_height_in_map_units: sps.pic_height_in_map_units_minus1 + 1,
            frame_mbs_only_flag: matches!(
                sps.frame_mbs_flags,
                h264_reader::nal::sps::FrameMbsFlags::Frames
            ),
            max_num_ref_frames: sps.max_num_ref_frames as u8,
            log2_max_frame_num_minus4: sps.log2_max_frame_num_minus4 as u8,
            pic_order_cnt_type,
            log2_max_pic_order_cnt_lsb_minus4,
            delta_pic_order_always_zero_flag,
        };

        // Store in context for PPS parsing
        self.context.put_seq_param_set(sps);

        // Store parsed SPS
        if (sps_id as usize) < self.sps.len() {
            self.sps[sps_id as usize] = Some(parsed);
        }

        Ok(())
    }

    /// Parse a PPS NAL unit.
    fn parse_pps(&mut self, nal_data: &[u8]) -> Result<()> {
        if nal_data.len() < 2 {
            return Err(VulkanError::DecodeError("PPS too short".to_string()).into());
        }

        // Create a BitReader from the NAL data (skip NAL header byte)
        let rbsp_data = &nal_data[1..];
        let reader = BitReader::new(rbsp_data);

        // Parse using h264_reader
        let pps = PicParameterSet::from_bits(&self.context, reader)
            .map_err(|e| VulkanError::DecodeError(format!("Failed to parse PPS: {:?}", e)))?;

        let pps_id = pps.pic_parameter_set_id.id() as u8;
        let sps_id = pps.seq_parameter_set_id.id() as u8;

        let parsed = ParsedPps {
            raw_bytes: nal_data.to_vec(),
            pps_id,
            sps_id,
            entropy_coding_mode_flag: pps.entropy_coding_mode_flag,
            bottom_field_pic_order_in_frame_present_flag: pps
                .bottom_field_pic_order_in_frame_present_flag,
            num_ref_idx_l0_default_active_minus1: pps.num_ref_idx_l0_default_active_minus1 as u8,
            num_ref_idx_l1_default_active_minus1: pps.num_ref_idx_l1_default_active_minus1 as u8,
            weighted_pred_flag: pps.weighted_pred_flag,
            weighted_bipred_idc: pps.weighted_bipred_idc,
            pic_init_qp_minus26: pps.pic_init_qp_minus26 as i8,
            pic_init_qs_minus26: pps.pic_init_qs_minus26 as i8,
            chroma_qp_index_offset: pps.chroma_qp_index_offset as i8,
            deblocking_filter_control_present_flag: pps.deblocking_filter_control_present_flag,
            constrained_intra_pred_flag: pps.constrained_intra_pred_flag,
            redundant_pic_cnt_present_flag: pps.redundant_pic_cnt_present_flag,
            transform_8x8_mode_flag: pps
                .extension
                .as_ref()
                .map(|e| e.transform_8x8_mode_flag)
                .unwrap_or(false),
            second_chroma_qp_index_offset: pps
                .extension
                .as_ref()
                .map(|e| e.second_chroma_qp_index_offset as i8)
                .unwrap_or(pps.chroma_qp_index_offset as i8),
        };

        // Store in context
        self.context.put_pic_param_set(pps);

        // Store parsed PPS
        if (pps_id as usize) < self.pps.len() {
            self.pps[pps_id as usize] = Some(parsed);
        }

        Ok(())
    }

    /// Get SPS by ID.
    pub fn get_sps(&self, sps_id: u8) -> Option<&ParsedSps> {
        self.sps.get(sps_id as usize).and_then(|s| s.as_ref())
    }

    /// Get PPS by ID.
    pub fn get_pps(&self, pps_id: u8) -> Option<&ParsedPps> {
        self.pps.get(pps_id as usize).and_then(|p| p.as_ref())
    }

    /// Get all parsed SPS.
    pub fn all_sps(&self) -> impl Iterator<Item = &ParsedSps> {
        self.sps.iter().filter_map(|s| s.as_ref())
    }

    /// Get all parsed PPS.
    pub fn all_pps(&self) -> impl Iterator<Item = &ParsedPps> {
        self.pps.iter().filter_map(|p| p.as_ref())
    }

    /// Check if we have at least one SPS and PPS.
    pub fn has_parameters(&self) -> bool {
        self.sps.iter().any(|s| s.is_some()) && self.pps.iter().any(|p| p.is_some())
    }

    /// Get picture dimensions from active SPS.
    pub fn picture_dimensions(&self, sps_id: u8) -> Option<(u32, u32)> {
        self.get_sps(sps_id).map(|sps| {
            let width = sps.pic_width_in_mbs * 16;
            let height =
                sps.pic_height_in_map_units * 16 * (if sps.frame_mbs_only_flag { 1 } else { 2 });
            (width, height)
        })
    }
}

/// Parse NAL units from Annex B byte stream.
///
/// Finds start codes (0x00 0x00 0x01 or 0x00 0x00 0x00 0x01) and extracts NAL units.
pub fn parse_annexb(data: &[u8]) -> Vec<&[u8]> {
    let mut nals = Vec::new();
    let mut i = 0;

    while i < data.len() {
        // Find start code
        let start = if i + 4 <= data.len() && &data[i..i + 4] == &[0, 0, 0, 1] {
            i + 4
        } else if i + 3 <= data.len() && &data[i..i + 3] == &[0, 0, 1] {
            i + 3
        } else {
            i += 1;
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
            nals.push(&data[start..end]);
        }

        i = end;
    }

    nals
}

/// Check if a NAL unit is an IDR (keyframe).
pub fn is_idr(nal_data: &[u8]) -> bool {
    if nal_data.is_empty() {
        return false;
    }
    (nal_data[0] & 0x1F) == 5
}

/// Check if a NAL unit is a coded slice (IDR or non-IDR).
pub fn is_slice(nal_data: &[u8]) -> bool {
    if nal_data.is_empty() {
        return false;
    }
    let nal_type = nal_data[0] & 0x1F;
    nal_type == 1 || nal_type == 5
}

/// Get the NAL unit type.
pub fn nal_type(nal_data: &[u8]) -> u8 {
    if nal_data.is_empty() {
        0
    } else {
        nal_data[0] & 0x1F
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_annexb() {
        // Test with 3-byte and 4-byte start codes
        let data = [
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1e, // SPS
            0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80, // PPS
        ];

        let nals = parse_annexb(&data);
        assert_eq!(nals.len(), 2);
        assert_eq!(nals[0][0] & 0x1F, 7); // SPS
        assert_eq!(nals[1][0] & 0x1F, 8); // PPS
    }

    #[test]
    fn test_nal_type() {
        assert_eq!(nal_type(&[0x67]), 7); // SPS
        assert_eq!(nal_type(&[0x68]), 8); // PPS
        assert_eq!(nal_type(&[0x65]), 5); // IDR
        assert_eq!(nal_type(&[0x41]), 1); // Non-IDR
    }

    #[test]
    fn test_is_idr() {
        assert!(is_idr(&[0x65])); // IDR
        assert!(!is_idr(&[0x41])); // Non-IDR
        assert!(!is_idr(&[0x67])); // SPS
    }
}
