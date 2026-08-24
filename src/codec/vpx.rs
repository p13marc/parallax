//! Keyframe detection for VP8 and VP9 bitstreams.
//!
//! Pure byte arithmetic over the frame's uncompressed header, so it needs no
//! libvpx — a receiver must be able to ask "can a decoder start here?" without
//! linking a decoder. The Annex-B codecs get the same question answered by
//! [`annexb::is_entry_point`](super::annexb::is_entry_point); this is the VPx
//! half of the same job.

/// Whether a VP8 frame is a keyframe.
///
/// The frame tag's first byte carries `frame_type` in bit 0, inverted: `0`
/// means key frame (RFC 6386 §9.1). `None` for an empty frame — the caller
/// decides what to do with a payload that says nothing.
pub fn vp8_is_keyframe(data: &[u8]) -> Option<bool> {
    Some(data.first()? & 0x01 == 0)
}

/// Whether a VP9 frame is a keyframe.
///
/// Reads the uncompressed header's leading bits (VP9 bitstream spec §6.2):
///
/// ```text
/// frame_marker            f(2)   == 0b10
/// profile_low_bit         f(1)
/// profile_high_bit        f(1)
/// [profile == 3] reserved f(1)
/// show_existing_frame     f(1)
/// [show_existing_frame] frame_to_show_map_idx f(3), and the frame ends
/// frame_type              f(1)   0 = KEY_FRAME
/// ```
///
/// A `show_existing_frame` frame re-displays a reference and carries no coded
/// data at all, so it is never an entry point however the bit that follows it
/// would have read.
///
/// `None` when the bytes are not a VP9 frame header (empty, or a bad frame
/// marker) — absent is honest, and better than guessing "not a keyframe" for
/// something that was never VP9.
pub fn vp9_is_keyframe(data: &[u8]) -> Option<bool> {
    let b = *data.first()?;

    // frame_marker must be 0b10.
    if b >> 6 != 0b10 {
        return None;
    }

    let profile = ((b >> 4) & 0x01) << 1 | ((b >> 5) & 0x01);
    // Profile 3 spends one reserved bit before show_existing_frame.
    let show_existing_shift = if profile == 3 { 2 } else { 3 };
    if (b >> show_existing_shift) & 0x01 == 1 {
        return Some(false);
    }

    Some((b >> (show_existing_shift - 1)) & 0x01 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vp8_reads_the_inverted_frame_type_bit() {
        assert_eq!(vp8_is_keyframe(&[0x00]), Some(true));
        assert_eq!(vp8_is_keyframe(&[0x01]), Some(false));
        assert_eq!(vp8_is_keyframe(&[]), None);
    }

    /// Build a VP9 first byte from the fields this reads.
    fn vp9_byte(profile: u8, show_existing: bool, key: bool) -> u8 {
        let mut bits: Vec<u8> = vec![1, 0]; // frame_marker
        bits.push(profile & 0x01); // profile_low_bit
        bits.push((profile >> 1) & 0x01); // profile_high_bit
        if profile == 3 {
            bits.push(0); // reserved_zero
        }
        bits.push(show_existing as u8);
        if !show_existing {
            bits.push(!key as u8); // frame_type: 0 = KEY_FRAME
        }
        let mut byte = 0u8;
        for (i, bit) in bits.iter().take(8).enumerate() {
            byte |= bit << (7 - i);
        }
        byte
    }

    #[test]
    fn vp9_reads_the_frame_type_across_every_profile() {
        for profile in 0..4u8 {
            assert_eq!(
                vp9_is_keyframe(&[vp9_byte(profile, false, true)]),
                Some(true),
                "profile {profile} keyframe"
            );
            assert_eq!(
                vp9_is_keyframe(&[vp9_byte(profile, false, false)]),
                Some(false),
                "profile {profile} inter frame"
            );
        }
    }

    /// A re-displayed reference carries no coded data, so a decoder cannot
    /// start there whatever the following bit would have said.
    #[test]
    fn a_show_existing_frame_is_never_an_entry_point() {
        for profile in 0..4u8 {
            assert_eq!(
                vp9_is_keyframe(&[vp9_byte(profile, true, true)]),
                Some(false),
                "profile {profile}"
            );
        }
    }

    #[test]
    fn a_bad_frame_marker_is_absent_not_false() {
        // 0b00 / 0b01 / 0b11 in the top two bits are not VP9 frames.
        assert_eq!(vp9_is_keyframe(&[0b0000_0000]), None);
        assert_eq!(vp9_is_keyframe(&[0b0100_0000]), None);
        assert_eq!(vp9_is_keyframe(&[0b1100_0000]), None);
        assert_eq!(vp9_is_keyframe(&[]), None);
    }
}
