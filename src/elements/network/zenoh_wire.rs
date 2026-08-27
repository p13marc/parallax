//! The zenoh side of parallax's wire: how the shared schema reaches a sample.
//!
//! [`ZenohSink`](super::ZenohSink) publishes each buffer's payload bytes
//! unmodified and serializes the buffer [`Metadata`](crate::metadata::Metadata)
//! into the sample's **attachment**, so non-parallax subscribers still see plain media bytes
//! while parallax subscribers ([`ZenohSrc`](super::ZenohSrc)) reconstruct full
//! metadata.
//!
//! ```text
//! payload    = raw buffer bytes, unmodified
//! attachment = [ 0x50 'P', 0x58 'X', version ] ++ rkyv(WireMetadata)
//! ```
//!
//! **The schema itself lives in [`crate::wire`]**, not here (#219). It used to
//! live in this module, which put it behind the `zenoh` feature and let it
//! drift from the IPC descriptor: `rtp` was carried by one wire and dropped by
//! the other, `planes` was dropped by both, and `MediaFormat::Rtp` had a
//! descriptor tag and no zenoh mapping at all. One schema, two encodings, and
//! `tests/wire_projection.rs` proving they agree.
//!
//! This module keeps what is genuinely zenoh's: the sample `Encoding`
//! derivation, and the metadata keys a zenoh receiver stamps.
//!
//! The `@media` plane's CBOR [`FrameMeta`](crate::wire::FrameMeta) is a third
//! schema and is deliberately **not** merged into either — it is a
//! consumer-facing contract owned by its specification. See
//! `docs/zenoh-wire.md`.

use crate::format::{AudioCodec, MediaFormat, TextCodec, VideoCodec};

pub use crate::wire::{
    CUSTOM_META, WIRE_MAGIC, WIRE_VERSION, WireFormat, WireMetadata, WirePlanes, WireRtp,
    WireTerminal,
};

/// Metadata key under which [`ZenohSrc`](super::ZenohSrc) stores the concrete
/// key expression a sample arrived on (a `String`; useful with wildcard
/// subscriptions).
pub const KEY_EXPR_META: &str = "zenoh/key_expr";

/// Derive a zenoh sample [`Encoding`](zenoh::bytes::Encoding) from a buffer's
/// media format, so non-parallax subscribers get a standard content hint.
pub fn encoding_for_format(format: Option<&MediaFormat>) -> zenoh::bytes::Encoding {
    use zenoh::bytes::Encoding;
    match format {
        Some(MediaFormat::Video(VideoCodec::H264)) => Encoding::VIDEO_H264,
        Some(MediaFormat::Video(VideoCodec::H265)) => Encoding::VIDEO_H265,
        Some(MediaFormat::Video(VideoCodec::Vp8)) => Encoding::VIDEO_VP8,
        Some(MediaFormat::Video(VideoCodec::Vp9)) => Encoding::VIDEO_VP9,
        Some(MediaFormat::Video(VideoCodec::Av1)) => Encoding::from("video/AV1"),
        Some(MediaFormat::Video(VideoCodec::Mjpeg)) => Encoding::IMAGE_JPEG,
        Some(MediaFormat::VideoRaw(vf)) => Encoding::VIDEO_RAW
            .with_schema(format!("{:?};{}x{}", vf.pixel_format, vf.width, vf.height)),
        Some(MediaFormat::Audio(AudioCodec::Opus)) => Encoding::from("audio/opus"),
        Some(MediaFormat::Audio(AudioCodec::Aac)) => Encoding::from("audio/aac"),
        Some(MediaFormat::Audio(AudioCodec::Mp3)) => Encoding::from("audio/mpeg"),
        Some(MediaFormat::Audio(AudioCodec::Pcmu)) => Encoding::from("audio/PCMU"),
        Some(MediaFormat::Audio(AudioCodec::Pcma)) => Encoding::from("audio/PCMA"),
        Some(MediaFormat::Audio(AudioCodec::Vorbis)) => Encoding::from("audio/vorbis"),
        Some(MediaFormat::Audio(AudioCodec::Eac3)) => Encoding::from("audio/eac3"),
        Some(MediaFormat::AudioRaw(af)) => Encoding::from("audio/raw").with_schema(format!(
            "{:?};{}ch;{}Hz",
            af.sample_format, af.channels, af.sample_rate
        )),
        Some(MediaFormat::MpegTs) => Encoding::from("video/mp2t"),
        // A subtitle cue, so a non-parallax subscriber gets a content hint
        // rather than `zenoh/bytes` (#279 — the same drift #238 closed for
        // the demuxers' video and audio).
        Some(MediaFormat::Text(TextCodec::Utf8)) => Encoding::TEXT_PLAIN,
        Some(MediaFormat::Text(TextCodec::Ass)) => Encoding::from("text/x-ssa"),
        Some(MediaFormat::Text(TextCodec::WebVtt)) => Encoding::from("text/vtt"),
        Some(MediaFormat::Rtp(_)) | Some(MediaFormat::Bytes) | None => Encoding::ZENOH_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoding_derivation() {
        assert_eq!(
            encoding_for_format(Some(&MediaFormat::Video(VideoCodec::H264))).to_string(),
            "video/h264"
        );
        assert_eq!(
            encoding_for_format(None),
            zenoh::bytes::Encoding::ZENOH_BYTES
        );
    }

    #[test]
    fn mjpeg_derives_the_jpeg_encoding() {
        // A preview stream must announce itself as image/jpeg, not
        // zenoh/bytes — the whole point of declaring the codec (#236).
        assert_eq!(
            encoding_for_format(Some(&MediaFormat::Video(VideoCodec::Mjpeg))).to_string(),
            "image/jpeg"
        );
    }
}
