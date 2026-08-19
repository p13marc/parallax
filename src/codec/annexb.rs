//! Annex-B bitstream helpers: NAL units, parameter sets, and AVCC conversion.
//!
//! H.264 (and H.265) elementary streams are Annex-B: a sequence of NAL units,
//! each prefixed by a `00 00 01` or `00 00 00 01` start code. Every part of a
//! pipeline that touches an encoded stream ends up needing the same three
//! things — *is there an IDR in here*, *what are the SPS/PPS*, and *convert
//! between start codes and MP4's length prefixes* (both directions). They were reimplemented three
//! separate times in this crate and exported zero times, with the two most
//! useful ones stranded on an MP4 *file sink* behind the `mp4-demux` feature.
//!
//! This module is compiled unconditionally: it is byte-slice arithmetic, it
//! depends on no codec library, and a downstream crate must not have to enable
//! `h264` to ask whether a packet it received is a keyframe.
//!
//! # Example
//!
//! ```rust
//! use parallax::codec::annexb;
//!
//! // start code, SPS (nal type 7), start code, IDR (nal type 5)
//! let stream = [0, 0, 0, 1, 0x67, 0x42, 0, 0, 1, 0x65, 0xAA];
//!
//! assert!(annexb::has_idr(&stream));
//! assert_eq!(annexb::nal_units(&stream).count(), 2);
//!
//! let (sps, pps) = annexb::extract_param_sets(&stream);
//! assert_eq!(sps.as_deref(), Some(&[0x67, 0x42][..]));
//! assert!(pps.is_none());
//! ```

/// NAL unit type for a coded slice of an IDR picture (H.264).
pub const NAL_IDR: u8 = 5;
/// NAL unit type for a sequence parameter set (H.264).
pub const NAL_SPS: u8 = 7;
/// NAL unit type for a picture parameter set (H.264).
pub const NAL_PPS: u8 = 8;

/// One NAL unit found in an Annex-B stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NalUnit<'a> {
    /// The NAL payload, **without** the start code. Never empty.
    pub data: &'a [u8],
}

impl NalUnit<'_> {
    /// The H.264 NAL unit type (the low 5 bits of the first byte).
    pub fn nal_type(&self) -> u8 {
        self.data[0] & 0x1F
    }

    /// Whether this is a coded slice of an IDR picture.
    pub fn is_idr(&self) -> bool {
        self.nal_type() == NAL_IDR
    }
}

/// Length of the start code at `data[i..]`, if one begins there.
fn start_code_len(data: &[u8], i: usize) -> Option<usize> {
    if data[i..].starts_with(&[0, 0, 0, 1]) {
        Some(4)
    } else if data[i..].starts_with(&[0, 0, 1]) {
        Some(3)
    } else {
        None
    }
}

/// Iterate the NAL units of an Annex-B stream.
///
/// Bytes before the first start code, and empty NAL units, are skipped. This is
/// the one scanner the rest of the module is built on — the copies it replaces
/// each had their own subtly different loop.
pub fn nal_units(data: &[u8]) -> impl Iterator<Item = NalUnit<'_>> {
    let mut cursor = 0usize;

    std::iter::from_fn(move || {
        // Find the next start code.
        while cursor < data.len() {
            let Some(len) = start_code_len(data, cursor) else {
                cursor += 1;
                continue;
            };

            let start = cursor + len;
            if start >= data.len() {
                cursor = data.len();
                return None;
            }

            // The NAL runs to the next start code, or to the end.
            let mut end = data.len();
            let mut j = start + 1;
            while j < data.len() {
                if start_code_len(data, j).is_some() {
                    end = j;
                    break;
                }
                j += 1;
            }

            cursor = end;
            return Some(NalUnit {
                data: &data[start..end],
            });
        }
        None
    })
}

/// Whether the stream contains a coded slice of an IDR picture — i.e. whether a
/// decoder joining here can start decoding.
pub fn has_idr(data: &[u8]) -> bool {
    nal_units(data).any(|nal| nal.is_idr())
}

/// The last SPS and PPS in the stream, without start codes.
///
/// Returns `(None, None)` when the stream carries no parameter sets, which is
/// normal for a delta frame.
pub fn extract_param_sets(data: &[u8]) -> (Option<Vec<u8>>, Option<Vec<u8>>) {
    let mut sps = None;
    let mut pps = None;

    for nal in nal_units(data) {
        match nal.nal_type() {
            NAL_SPS => sps = Some(nal.data.to_vec()),
            NAL_PPS => pps = Some(nal.data.to_vec()),
            _ => {}
        }
    }

    (sps, pps)
}

/// Prepend `sps` and `pps` (raw, no start codes) to an Annex-B access unit.
///
/// Use this to make a keyframe self-contained for a late joiner when the encoder
/// only emits parameter sets once.
pub fn prepend_param_sets(sps: &[u8], pps: &[u8], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(sps.len() + pps.len() + data.len() + 8);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(sps);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend_from_slice(pps);
    out.extend_from_slice(data);
    out
}

/// Convert an Annex-B stream to AVCC (length-prefixed) form, as MP4 requires.
///
/// Each start code is replaced by a 4-byte big-endian length.
pub fn annex_b_to_avcc(data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(data.len());

    for nal in nal_units(data) {
        out.extend_from_slice(&(nal.data.len() as u32).to_be_bytes());
        out.extend_from_slice(nal.data);
    }

    out
}

/// Convert an AVCC (length-prefixed) sample to Annex-B start-code form —
/// the reverse of [`annex_b_to_avcc`], for feeding demuxed MP4 samples to a
/// decoder.
///
/// `length_size` is `lengthSizeMinusOne + 1` from the avcC box and must be
/// 1, 2 or 4 (ISO/IEC 14496-15 forbids 3). Zero-length NALs are skipped;
/// a truncated length prefix or a NAL length overrunning the sample is an
/// error — silently emitting a torn NAL would corrupt the stream downstream.
pub fn avcc_to_annex_b(data: &[u8], length_size: u8) -> crate::error::Result<Vec<u8>> {
    let mut out = Vec::with_capacity(data.len() + 16);
    avcc_to_annex_b_into(data, length_size, &mut out)?;
    Ok(out)
}

/// The exact Annex-B size [`avcc_to_annex_b`] would produce.
///
/// A cheap pre-pass over the length prefixes only — lets a caller size an
/// arena slot exactly and convert into it with
/// [`avcc_to_annex_b_into_slice`], skipping the intermediate `Vec`.
pub fn annex_b_len(data: &[u8], length_size: u8) -> crate::error::Result<usize> {
    let mut total = 0usize;
    walk_avcc(data, length_size, |nal| {
        total += 4 + nal.len();
        Ok(())
    })?;
    Ok(total)
}

/// [`avcc_to_annex_b`], appending onto a caller-provided `Vec`.
pub fn avcc_to_annex_b_into(
    data: &[u8],
    length_size: u8,
    out: &mut Vec<u8>,
) -> crate::error::Result<()> {
    walk_avcc(data, length_size, |nal| {
        out.extend_from_slice(&[0, 0, 0, 1]);
        out.extend_from_slice(nal);
        Ok(())
    })
}

/// [`avcc_to_annex_b`], writing into a caller-provided slice (e.g. an arena
/// slot sized with [`annex_b_len`]). Returns the bytes written.
pub fn avcc_to_annex_b_into_slice(
    data: &[u8],
    length_size: u8,
    out: &mut [u8],
) -> crate::error::Result<usize> {
    let mut pos = 0usize;
    walk_avcc(data, length_size, |nal| {
        let end = pos + 4 + nal.len();
        let Some(dst) = out.get_mut(pos..end) else {
            return Err(crate::error::Error::Config(
                "Annex-B output slice too small".into(),
            ));
        };
        dst[..4].copy_from_slice(&[0, 0, 0, 1]);
        dst[4..].copy_from_slice(nal);
        pos = end;
        Ok(())
    })?;
    Ok(pos)
}

/// Walk the length-prefixed NALs of an AVCC sample, calling `emit` for each
/// non-empty one. Shared validation for every conversion flavour above.
fn walk_avcc(
    data: &[u8],
    length_size: u8,
    mut emit: impl FnMut(&[u8]) -> crate::error::Result<()>,
) -> crate::error::Result<()> {
    if !matches!(length_size, 1 | 2 | 4) {
        return Err(crate::error::Error::Config(format!(
            "invalid AVCC length_size {length_size} (must be 1, 2 or 4)"
        )));
    }
    let length_size = length_size as usize;

    let mut i = 0usize;
    while i < data.len() {
        if i + length_size > data.len() {
            return Err(crate::error::Error::Config(format!(
                "truncated AVCC NAL length prefix at offset {i}"
            )));
        }
        let mut len = 0usize;
        for &byte in &data[i..i + length_size] {
            len = (len << 8) | byte as usize;
        }
        i += length_size;

        if len > data.len() - i {
            return Err(crate::error::Error::Config(format!(
                "AVCC NAL length {len} overruns sample ({} bytes left)",
                data.len() - i
            )));
        }
        if len > 0 {
            emit(&data[i..i + len])?;
        }
        i += len;
    }

    Ok(())
}

/// A track's length-prefixed NAL configuration.
///
/// H.264 and HEVC store their bitstream the same way in a container — NAL
/// units behind a fixed-width length prefix, with the parameter sets kept out
/// of band — and want the same treatment on the way out: start codes, with
/// the parameter sets prepended to every keyframe so a decoder joining
/// mid-stream or after a seek can start. Only the record that carries them
/// differs, so only its parser is per-codec.
pub struct NalConfig {
    /// NAL length prefix size in bytes (1, 2 or 4).
    pub length_size: u8,
    /// Parameter-set NALs to prepend to a keyframe, in the order the record
    /// listed them: SPS then PPS for H.264, VPS then SPS then PPS for HEVC.
    pub params: Vec<Vec<u8>>,
}

/// Parse an AVCDecoderConfigurationRecord (ISO 14496-15 §5.3.3.1).
pub fn parse_avcc(data: &[u8]) -> crate::error::Result<NalConfig> {
    let err = || crate::error::Error::Config("truncated avcC record in CodecPrivate".into());
    if data.len() < 7 || data[0] != 1 {
        return Err(crate::error::Error::Config(format!(
            "CodecPrivate is not an avcC record (len {}, version {})",
            data.len(),
            data.first().copied().unwrap_or(0)
        )));
    }
    let length_size = (data[4] & 0x03) + 1;
    let mut pos = 5usize;
    let read_sets = |pos: &mut usize, count: usize| -> crate::error::Result<Vec<Vec<u8>>> {
        let mut sets = Vec::with_capacity(count);
        for _ in 0..count {
            let len = data
                .get(*pos..*pos + 2)
                .map(|b| u16::from_be_bytes([b[0], b[1]]) as usize)
                .ok_or_else(err)?;
            *pos += 2;
            sets.push(data.get(*pos..*pos + len).ok_or_else(err)?.to_vec());
            *pos += len;
        }
        Ok(sets)
    };
    let num_sps = (*data.get(pos).ok_or_else(err)? & 0x1f) as usize;
    pos += 1;
    let sps = read_sets(&mut pos, num_sps)?;
    let num_pps = *data.get(pos).ok_or_else(err)? as usize;
    pos += 1;
    let pps = read_sets(&mut pos, num_pps)?;
    Ok(NalConfig {
        length_size,
        params: sps.into_iter().chain(pps).collect(),
    })
}

/// Parse an HEVCDecoderConfigurationRecord (ISO 14496-15 §8.3.3.1).
///
/// Only two of its 22 fixed header bytes matter here — the version and the
/// length-prefix width — because everything else (profile, tier, chroma
/// format, bit depth) is repeated in the SPS the decoder is about to be
/// handed anyway. The arrays that follow hold the VPS, SPS and PPS, and are
/// taken in file order: a decoder needs them in dependency order, and every
/// muxer writes them that way because the spec's own array order is.
pub fn parse_hvcc(data: &[u8]) -> crate::error::Result<NalConfig> {
    let err = || crate::error::Error::Config("truncated hvcC record in CodecPrivate".into());
    if data.len() < 23 || data[0] != 1 {
        return Err(crate::error::Error::Config(format!(
            "CodecPrivate is not an hvcC record (len {}, version {})",
            data.len(),
            data.first().copied().unwrap_or(0)
        )));
    }
    let length_size = (data[21] & 0x03) + 1;
    let num_arrays = data[22] as usize;
    let mut pos = 23usize;
    let mut params = Vec::new();
    for _ in 0..num_arrays {
        // array_completeness(1) | reserved(1) | NAL_unit_type(6)
        pos += 1;
        let count = data
            .get(pos..pos + 2)
            .map(|b| u16::from_be_bytes([b[0], b[1]]) as usize)
            .ok_or_else(err)?;
        pos += 2;
        for _ in 0..count {
            let len = data
                .get(pos..pos + 2)
                .map(|b| u16::from_be_bytes([b[0], b[1]]) as usize)
                .ok_or_else(err)?;
            pos += 2;
            params.push(data.get(pos..pos + len).ok_or_else(err)?.to_vec());
            pos += len;
        }
    }
    Ok(NalConfig {
        length_size,
        params,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// SPS (7), PPS (8), IDR (5): a complete keyframe access unit.
    fn keyframe() -> Vec<u8> {
        let mut data = Vec::new();
        data.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E]); // SPS
        data.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x38, 0x80]); // PPS
        data.extend_from_slice(&[0, 0, 1, 0x65, 0xAA, 0xBB]); // IDR, 3-byte start code
        data
    }

    /// A non-IDR slice (type 1).
    fn delta_frame() -> Vec<u8> {
        vec![0, 0, 0, 1, 0x41, 0x9A, 0x02]
    }

    #[test]
    fn nal_units_handles_both_start_code_lengths() {
        let keyframe = keyframe();
        let nals: Vec<_> = nal_units(&keyframe).collect();
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0].nal_type(), NAL_SPS);
        assert_eq!(nals[1].nal_type(), NAL_PPS);
        assert_eq!(nals[2].nal_type(), NAL_IDR);
        assert_eq!(nals[2].data, &[0x65, 0xAA, 0xBB]);
    }

    #[test]
    fn has_idr_distinguishes_keyframes_from_delta_frames() {
        assert!(has_idr(&keyframe()));

        assert!(!has_idr(&delta_frame()));
        assert!(!has_idr(&[]));
        assert!(!has_idr(&[0, 0, 0, 1]), "a start code with no NAL after it");
    }

    #[test]
    fn extract_param_sets_returns_the_raw_sets() {
        let (sps, pps) = extract_param_sets(&keyframe());
        assert_eq!(sps.unwrap(), vec![0x67, 0x42, 0x00, 0x1E]);
        assert_eq!(pps.unwrap(), vec![0x68, 0xCE, 0x38, 0x80]);

        let (sps, pps) = extract_param_sets(&delta_frame());
        assert!(sps.is_none() && pps.is_none());
    }

    #[test]
    fn prepend_param_sets_makes_a_frame_self_contained() {
        let (sps, pps) = extract_param_sets(&keyframe());
        let bare_idr = vec![0, 0, 0, 1, 0x65, 0xAA];

        let complete = prepend_param_sets(&sps.unwrap(), &pps.unwrap(), &bare_idr);

        let types: Vec<_> = nal_units(&complete).map(|n| n.nal_type()).collect();
        assert_eq!(types, vec![NAL_SPS, NAL_PPS, NAL_IDR]);
    }

    #[test]
    fn annex_b_to_avcc_replaces_start_codes_with_lengths() {
        let avcc = annex_b_to_avcc(&delta_frame());
        assert_eq!(avcc, vec![0, 0, 0, 3, 0x41, 0x9A, 0x02]);

        // Round-trip the lengths of a multi-NAL access unit.
        let avcc = annex_b_to_avcc(&keyframe());
        let mut lengths = Vec::new();
        let mut i = 0;
        while i + 4 <= avcc.len() {
            let len = u32::from_be_bytes(avcc[i..i + 4].try_into().unwrap()) as usize;
            lengths.push(len);
            i += 4 + len;
        }
        assert_eq!(lengths, vec![4, 4, 3]);
        assert_eq!(i, avcc.len(), "no trailing bytes");
    }

    #[test]
    fn leading_garbage_before_the_first_start_code_is_skipped() {
        let mut data = vec![0xDE, 0xAD, 0xBE, 0xEF];
        data.extend_from_slice(&delta_frame());

        let nals: Vec<_> = nal_units(&data).collect();
        assert_eq!(nals.len(), 1);
        assert_eq!(nals[0].nal_type(), 1);
    }

    #[test]
    fn avcc_round_trips_with_annex_b_to_avcc() {
        let avcc = annex_b_to_avcc(&keyframe());
        let back = avcc_to_annex_b(&avcc, 4).unwrap();

        let orig: Vec<_> = nal_units(&keyframe()).map(|n| n.data.to_vec()).collect();
        let round: Vec<_> = nal_units(&back).map(|n| n.data.to_vec()).collect();
        assert_eq!(round, orig, "payloads survive the round trip");
        let types: Vec<_> = nal_units(&back).map(|n| n.nal_type()).collect();
        assert_eq!(types, vec![NAL_SPS, NAL_PPS, NAL_IDR]);
    }

    #[test]
    fn avcc_to_annex_b_handles_short_length_prefixes() {
        // length_size 2: one 3-byte NAL.
        let out = avcc_to_annex_b(&[0, 3, 0x41, 0x9A, 0x02], 2).unwrap();
        assert_eq!(out, vec![0, 0, 0, 1, 0x41, 0x9A, 0x02]);

        // length_size 1: two NALs back to back.
        let out = avcc_to_annex_b(&[2, 0x41, 0x9A, 1, 0x65], 1).unwrap();
        let types: Vec<_> = nal_units(&out).map(|n| n.nal_type()).collect();
        assert_eq!(types, vec![1, NAL_IDR]);
    }

    #[test]
    fn avcc_to_annex_b_rejects_truncation() {
        // Length claims 9 bytes, only 2 remain.
        assert!(avcc_to_annex_b(&[0, 0, 0, 9, 0x41, 0x9A], 4).is_err());
        // A 3-byte tail where a 4-byte prefix should be.
        assert!(avcc_to_annex_b(&[0, 0, 0, 2, 0x41, 0x9A, 0, 0, 0], 4).is_err());
    }

    #[test]
    fn avcc_to_annex_b_rejects_bad_length_sizes() {
        assert!(avcc_to_annex_b(&[0, 0, 1, 0x41], 3).is_err());
        assert!(avcc_to_annex_b(&[0x41], 0).is_err());
    }

    #[test]
    fn avcc_to_annex_b_skips_zero_length_nals() {
        let out = avcc_to_annex_b(&[0, 0, 0, 0, 0, 0, 0, 1, 0x65], 4).unwrap();
        assert_eq!(out, vec![0, 0, 0, 1, 0x65]);
    }
}
