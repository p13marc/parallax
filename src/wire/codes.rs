//! The one stable code vocabulary, shared by every wire out of a parallax
//! process (#219).
//!
//! # Why one table and not two
//!
//! There used to be two, and they had already diverged. The zenoh attachment
//! hand-wrote tables like the ones below; the IPC descriptor cast **enum
//! discriminants** (`v.pixel_format as u32`) and pinned them with a lock test
//! so nobody reordered `PixelFormat`. Two vocabularies that happen to agree is
//! not one vocabulary — it is a coincidence with a test holding it in place,
//! and the coincidence was already broken: `MediaFormat::Rtp` had a descriptor
//! tag and *no* zenoh mapping at all, so an RTP-formatted buffer crossed a
//! process boundary with its format intact and a machine boundary with its
//! format silently gone.
//!
//! With one table the enums are free to be reordered, renamed or renumbered:
//! only this file is the wire.
//!
//! # The rule
//!
//! **Codes are append-only.** A number, once shipped, means what it meant.
//! Adding a variant appends a code; removing one leaves a hole that
//! `*_from_code` answers `None` for. The lock tests at the bottom pin every
//! assignment, so changing one is a deliberate act with a failing test
//! attached rather than a refactor's side effect.

use crate::format::{AudioCodec, PixelFormat, RtpEncoding, SampleFormat, VideoCodec};

// ---------------------------------------------------------------------------
// MediaFormat tags
// ---------------------------------------------------------------------------

/// No format declared.
pub const FMT_NONE: u8 = 0;
/// [`MediaFormat::VideoRaw`](crate::format::MediaFormat::VideoRaw).
pub const FMT_VIDEO_RAW: u8 = 1;
/// [`MediaFormat::Video`](crate::format::MediaFormat::Video).
pub const FMT_VIDEO: u8 = 2;
/// [`MediaFormat::AudioRaw`](crate::format::MediaFormat::AudioRaw).
pub const FMT_AUDIO_RAW: u8 = 3;
/// [`MediaFormat::Audio`](crate::format::MediaFormat::Audio).
pub const FMT_AUDIO: u8 = 4;
/// [`MediaFormat::Rtp`](crate::format::MediaFormat::Rtp).
pub const FMT_RTP: u8 = 5;
/// [`MediaFormat::MpegTs`](crate::format::MediaFormat::MpegTs).
pub const FMT_MPEG_TS: u8 = 6;
/// [`MediaFormat::Bytes`](crate::format::MediaFormat::Bytes).
pub const FMT_BYTES: u8 = 7;

// ---------------------------------------------------------------------------
// Video codecs
// ---------------------------------------------------------------------------

/// Stable wire code for a [`VideoCodec`].
pub fn video_codec_to_code(codec: VideoCodec) -> u8 {
    match codec {
        VideoCodec::H264 => 0,
        VideoCodec::H265 => 1,
        VideoCodec::Vp8 => 2,
        VideoCodec::Vp9 => 3,
        VideoCodec::Av1 => 4,
        VideoCodec::Mjpeg => 5,
    }
}

/// [`VideoCodec`] for a stable wire code.
pub fn video_codec_from_code(code: u8) -> Option<VideoCodec> {
    Some(match code {
        0 => VideoCodec::H264,
        1 => VideoCodec::H265,
        2 => VideoCodec::Vp8,
        3 => VideoCodec::Vp9,
        4 => VideoCodec::Av1,
        5 => VideoCodec::Mjpeg,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Audio codecs
// ---------------------------------------------------------------------------

/// Stable wire code for an [`AudioCodec`].
pub fn audio_codec_to_code(codec: AudioCodec) -> u8 {
    match codec {
        AudioCodec::Opus => 0,
        AudioCodec::Aac => 1,
        AudioCodec::Mp3 => 2,
        AudioCodec::Pcmu => 3,
        AudioCodec::Pcma => 4,
        AudioCodec::Vorbis => 5,
        AudioCodec::Eac3 => 6,
    }
}

/// [`AudioCodec`] for a stable wire code.
pub fn audio_codec_from_code(code: u8) -> Option<AudioCodec> {
    Some(match code {
        0 => AudioCodec::Opus,
        1 => AudioCodec::Aac,
        2 => AudioCodec::Mp3,
        3 => AudioCodec::Pcmu,
        4 => AudioCodec::Pcma,
        5 => AudioCodec::Vorbis,
        6 => AudioCodec::Eac3,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Pixel formats
// ---------------------------------------------------------------------------

/// Stable wire code for a [`PixelFormat`].
pub fn pixel_format_to_code(format: PixelFormat) -> u8 {
    match format {
        PixelFormat::I420 => 0,
        PixelFormat::Nv12 => 1,
        PixelFormat::I420_10Le => 2,
        PixelFormat::P010 => 3,
        PixelFormat::I422 => 4,
        PixelFormat::Yuyv => 5,
        PixelFormat::Uyvy => 6,
        PixelFormat::I444 => 7,
        PixelFormat::Rgb24 => 8,
        PixelFormat::Rgba => 9,
        PixelFormat::Bgr24 => 10,
        PixelFormat::Bgra => 11,
        PixelFormat::Argb => 12,
        PixelFormat::Gray8 => 13,
        PixelFormat::Gray16Le => 14,
    }
}

/// [`PixelFormat`] for a stable wire code.
pub fn pixel_format_from_code(code: u8) -> Option<PixelFormat> {
    Some(match code {
        0 => PixelFormat::I420,
        1 => PixelFormat::Nv12,
        2 => PixelFormat::I420_10Le,
        3 => PixelFormat::P010,
        4 => PixelFormat::I422,
        5 => PixelFormat::Yuyv,
        6 => PixelFormat::Uyvy,
        7 => PixelFormat::I444,
        8 => PixelFormat::Rgb24,
        9 => PixelFormat::Rgba,
        10 => PixelFormat::Bgr24,
        11 => PixelFormat::Bgra,
        12 => PixelFormat::Argb,
        13 => PixelFormat::Gray8,
        14 => PixelFormat::Gray16Le,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// Sample formats
// ---------------------------------------------------------------------------

/// Stable wire code for a [`SampleFormat`].
pub fn sample_format_to_code(format: SampleFormat) -> u8 {
    match format {
        SampleFormat::S16 => 0,
        SampleFormat::S32 => 1,
        SampleFormat::F32 => 2,
        SampleFormat::U8 => 3,
    }
}

/// [`SampleFormat`] for a stable wire code.
pub fn sample_format_from_code(code: u8) -> Option<SampleFormat> {
    Some(match code {
        0 => SampleFormat::S16,
        1 => SampleFormat::S32,
        2 => SampleFormat::F32,
        3 => SampleFormat::U8,
        _ => return None,
    })
}

// ---------------------------------------------------------------------------
// RTP encodings
// ---------------------------------------------------------------------------

/// Stable wire code for an [`RtpEncoding`], plus its dynamic payload type.
///
/// Two values rather than one because `RtpEncoding` is not `repr(u8)`: it
/// carries `Dynamic(u8)`, so the code names the *shape* and the second value
/// carries the payload type for that one variant. It is 0 for every other.
pub fn rtp_encoding_to_code(encoding: RtpEncoding) -> (u8, u8) {
    match encoding {
        RtpEncoding::H264 => (0, 0),
        RtpEncoding::H265 => (1, 0),
        RtpEncoding::Vp8 => (2, 0),
        RtpEncoding::Vp9 => (3, 0),
        RtpEncoding::Opus => (4, 0),
        RtpEncoding::Pcmu => (5, 0),
        RtpEncoding::Pcma => (6, 0),
        RtpEncoding::Av1 => (7, 0),
        RtpEncoding::Dynamic(pt) => (8, pt),
    }
}

/// [`RtpEncoding`] for a stable wire code.
pub fn rtp_encoding_from_code(code: u8, dynamic_pt: u8) -> Option<RtpEncoding> {
    Some(match code {
        0 => RtpEncoding::H264,
        1 => RtpEncoding::H265,
        2 => RtpEncoding::Vp8,
        3 => RtpEncoding::Vp9,
        4 => RtpEncoding::Opus,
        5 => RtpEncoding::Pcmu,
        6 => RtpEncoding::Pcma,
        7 => RtpEncoding::Av1,
        8 => RtpEncoding::Dynamic(dynamic_pt),
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The lock. Every assignment written out, so a change to one is a
    /// deliberate act with a failing test attached — this replaces the
    /// `descriptor_enum_discriminants_locked` test the IPC descriptor used to
    /// need, and covers both wires at once rather than one of them.
    #[test]
    fn every_code_is_pinned() {
        use AudioCodec as A;
        use PixelFormat as P;
        use RtpEncoding as R;
        use SampleFormat as S;
        use VideoCodec as V;

        for (codec, code) in [
            (V::H264, 0),
            (V::H265, 1),
            (V::Vp8, 2),
            (V::Vp9, 3),
            (V::Av1, 4),
            (V::Mjpeg, 5),
        ] {
            assert_eq!(video_codec_to_code(codec), code, "{codec:?}");
            assert_eq!(video_codec_from_code(code), Some(codec));
        }

        for (codec, code) in [
            (A::Opus, 0),
            (A::Aac, 1),
            (A::Mp3, 2),
            (A::Pcmu, 3),
            (A::Pcma, 4),
            (A::Vorbis, 5),
            (A::Eac3, 6),
        ] {
            assert_eq!(audio_codec_to_code(codec), code, "{codec:?}");
            assert_eq!(audio_codec_from_code(code), Some(codec));
        }

        for (format, code) in [
            (P::I420, 0),
            (P::Nv12, 1),
            (P::I420_10Le, 2),
            (P::P010, 3),
            (P::I422, 4),
            (P::Yuyv, 5),
            (P::Uyvy, 6),
            (P::I444, 7),
            (P::Rgb24, 8),
            (P::Rgba, 9),
            (P::Bgr24, 10),
            (P::Bgra, 11),
            (P::Argb, 12),
            (P::Gray8, 13),
            (P::Gray16Le, 14),
        ] {
            assert_eq!(pixel_format_to_code(format), code, "{format:?}");
            assert_eq!(pixel_format_from_code(code), Some(format));
        }

        for (format, code) in [(S::S16, 0), (S::S32, 1), (S::F32, 2), (S::U8, 3)] {
            assert_eq!(sample_format_to_code(format), code, "{format:?}");
            assert_eq!(sample_format_from_code(code), Some(format));
        }

        for (encoding, code) in [
            (R::H264, 0),
            (R::H265, 1),
            (R::Vp8, 2),
            (R::Vp9, 3),
            (R::Opus, 4),
            (R::Pcmu, 5),
            (R::Pcma, 6),
            (R::Av1, 7),
        ] {
            assert_eq!(rtp_encoding_to_code(encoding), (code, 0), "{encoding:?}");
            assert_eq!(rtp_encoding_from_code(code, 0), Some(encoding));
        }
        assert_eq!(rtp_encoding_to_code(R::Dynamic(97)), (8, 97));
        assert_eq!(rtp_encoding_from_code(8, 97), Some(R::Dynamic(97)));

        assert_eq!(
            [
                FMT_NONE,
                FMT_VIDEO_RAW,
                FMT_VIDEO,
                FMT_AUDIO_RAW,
                FMT_AUDIO,
                FMT_RTP,
                FMT_MPEG_TS,
                FMT_BYTES
            ],
            [0, 1, 2, 3, 4, 5, 6, 7]
        );
    }

    /// A code no table knows must answer `None`, not panic and not guess: a
    /// newer peer sends codes this build has never heard of, and on the zenoh
    /// side that is a *legitimate* thing for it to do.
    #[test]
    fn an_unknown_code_is_absent_not_an_error() {
        assert_eq!(video_codec_from_code(200), None);
        assert_eq!(audio_codec_from_code(200), None);
        assert_eq!(pixel_format_from_code(200), None);
        assert_eq!(sample_format_from_code(200), None);
        assert_eq!(rtp_encoding_from_code(200, 0), None);
    }
}
