//! Wire format for buffer metadata carried in zenoh attachments.
//!
//! [`ZenohSink`](super::ZenohSink) publishes each buffer's payload bytes
//! unmodified and serializes the buffer [`Metadata`] into the sample's
//! **attachment**, so non-parallax subscribers still see plain media bytes
//! while parallax subscribers ([`ZenohSrc`](super::ZenohSrc)) reconstruct
//! full metadata (PTS/DTS/duration/sequence/flags/format).
//!
//! # Wire contract (version 1)
//!
//! ```text
//! payload    = raw buffer bytes, unmodified
//! attachment = [ 0x50 'P', 0x58 'X', 0x01 version ] ++ rkyv(WireMetadata)
//! ```
//!
//! - A missing attachment, unknown magic, or unknown version means "foreign
//!   or legacy publisher": receivers fall back to fabricated metadata and
//!   must not error.
//! - [`WireMetadata`] is deliberately decoupled from the in-memory
//!   [`Metadata`] type so internal refactors are not wire breaks; any change
//!   to the serialized layout requires bumping [`WIRE_VERSION`].
//! - Gaps in the received `sequence` indicate samples lost on the wire
//!   (congestion drop, late join); [`ZenohSrc`](super::ZenohSrc) marks them
//!   with [`BufferFlags::DISCONT`](crate::metadata::BufferFlags::DISCONT).
//!
//! This format is for parallax↔parallax links. It is intentionally distinct
//! from consumer-facing media planes (e.g. zensight's `@media` plane, which
//! uses a serde/CBOR attachment): formats are plane-scoped, not unified.

use crate::clock::ClockTime;
use crate::format::{
    AudioCodec, AudioFormat, Framerate, MediaFormat, PixelFormat, SampleFormat, VideoCodec,
    VideoFormat,
};
use crate::metadata::{BufferFlags, Metadata};

/// Attachment magic bytes (`"PX"`).
pub const WIRE_MAGIC: [u8; 2] = [0x50, 0x58];

/// Attachment wire-format version.
pub const WIRE_VERSION: u8 = 1;

/// Metadata key under which [`ZenohSrc`](super::ZenohSrc) stores the concrete
/// key expression a sample arrived on (a `String`; useful with wildcard
/// subscriptions).
pub const KEY_EXPR_META: &str = "zenoh/key_expr";

/// Metadata key under which custom wire entries with unknown keys are
/// aggregated as a `Vec<(String, Vec<u8>)>` (keys can't be interned into the
/// `&'static str`-keyed metadata map without leaking).
pub const CUSTOM_META: &str = "zenoh/custom";

/// Custom byte-entry keys that [`ZenohSrc`](super::ZenohSrc) recognizes and
/// re-inserts under their own metadata key via [`Metadata::set_bytes`].
/// One list for every wire (the IPC overflow path uses the same one);
/// extend it there as new well-known keys appear.
use crate::elements::ipc::protocol::KNOWN_CUSTOM_KEYS;

/// Buffer metadata as serialized on the zenoh wire (version 1).
///
/// Timestamps are nanoseconds with `u64::MAX` as the
/// [`ClockTime::NONE`] sentinel, matching `ClockTime`'s own representation.
#[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
#[rkyv(derive(Debug))]
pub struct WireMetadata {
    /// Presentation timestamp in ns (`u64::MAX` = none).
    pub pts_nanos: u64,
    /// Decode timestamp in ns (`u64::MAX` = none).
    pub dts_nanos: u64,
    /// Duration in ns (`u64::MAX` = none).
    pub duration_nanos: u64,
    /// Monotonic sequence number within the published stream.
    pub sequence: u64,
    /// Stream identifier.
    pub stream_id: u32,
    /// [`BufferFlags`] bits.
    pub flags: u8,
    /// Byte offset in the original source.
    pub offset: Option<u64>,
    /// Media format of the payload.
    pub format: Option<WireFormat>,
    /// Whitelisted custom byte entries (e.g. KLV, SEI), as (key, bytes).
    pub custom: Vec<(String, Vec<u8>)>,
}

mod wire_format {
    //! Inner module so `allow(missing_docs)` covers rkyv's generated
    //! resolver types (their fields cannot carry doc comments).
    #![allow(missing_docs)]

    /// Media format as serialized on the wire.
    ///
    /// Codec and pixel/sample format codes are the stable wire encoding — they
    /// mirror the in-memory enums today but must never be renumbered without a
    /// [`WIRE_VERSION`](super::WIRE_VERSION) bump.
    #[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    #[rkyv(derive(Debug))]
    pub enum WireFormat {
        /// Raw (uncompressed) video.
        VideoRaw {
            /// Frame width in pixels.
            width: u32,
            /// Frame height in pixels.
            height: u32,
            /// Pixel format code (see [`pixel_format_to_code`](super::pixel_format_to_code)).
            pixel_format: u8,
            /// Framerate numerator.
            fps_num: u32,
            /// Framerate denominator.
            fps_den: u32,
        },
        /// Encoded video.
        Video {
            /// Codec code (see [`video_codec_to_code`](super::video_codec_to_code)).
            codec: u8,
        },
        /// Raw (uncompressed) audio.
        AudioRaw {
            /// Sample rate in Hz.
            sample_rate: u32,
            /// Channel count.
            channels: u16,
            /// Sample format code (see [`sample_format_to_code`](super::sample_format_to_code)).
            sample_format: u8,
        },
        /// Encoded audio.
        Audio {
            /// Codec code (see [`audio_codec_to_code`](super::audio_codec_to_code)).
            codec: u8,
        },
        /// MPEG-TS packets.
        MpegTs,
        /// Raw bytes.
        Bytes,
    }
}
pub use wire_format::WireFormat;

/// Stable wire code for a [`VideoCodec`].
pub fn video_codec_to_code(codec: VideoCodec) -> u8 {
    match codec {
        VideoCodec::H264 => 0,
        VideoCodec::H265 => 1,
        VideoCodec::Vp8 => 2,
        VideoCodec::Vp9 => 3,
        VideoCodec::Av1 => 4,
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
        _ => return None,
    })
}

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

impl WireFormat {
    /// Convert an in-memory [`MediaFormat`] to its wire representation.
    ///
    /// Returns `None` for formats without a wire mapping (currently
    /// [`MediaFormat::Rtp`] — RTP payloads carry their own headers).
    pub fn from_media_format(format: &MediaFormat) -> Option<Self> {
        Some(match format {
            MediaFormat::VideoRaw(vf) => WireFormat::VideoRaw {
                width: vf.width,
                height: vf.height,
                pixel_format: pixel_format_to_code(vf.pixel_format),
                fps_num: vf.framerate.num,
                fps_den: vf.framerate.den,
            },
            MediaFormat::Video(codec) => WireFormat::Video {
                codec: video_codec_to_code(*codec),
            },
            MediaFormat::AudioRaw(af) => WireFormat::AudioRaw {
                sample_rate: af.sample_rate,
                channels: af.channels,
                sample_format: sample_format_to_code(af.sample_format),
            },
            MediaFormat::Audio(codec) => WireFormat::Audio {
                codec: audio_codec_to_code(*codec),
            },
            MediaFormat::MpegTs => WireFormat::MpegTs,
            MediaFormat::Bytes => WireFormat::Bytes,
            MediaFormat::Rtp(_) => return None,
        })
    }

    /// Convert back to an in-memory [`MediaFormat`].
    ///
    /// Returns `None` if a code is unknown (sent by a newer peer).
    pub fn to_media_format(&self) -> Option<MediaFormat> {
        Some(match self {
            WireFormat::VideoRaw {
                width,
                height,
                pixel_format,
                fps_num,
                fps_den,
            } => MediaFormat::VideoRaw(VideoFormat {
                width: *width,
                height: *height,
                pixel_format: pixel_format_from_code(*pixel_format)?,
                framerate: Framerate {
                    num: *fps_num,
                    den: *fps_den,
                },
            }),
            WireFormat::Video { codec } => MediaFormat::Video(video_codec_from_code(*codec)?),
            WireFormat::AudioRaw {
                sample_rate,
                channels,
                sample_format,
            } => MediaFormat::AudioRaw(AudioFormat {
                sample_rate: *sample_rate,
                channels: *channels,
                sample_format: sample_format_from_code(*sample_format)?,
            }),
            WireFormat::Audio { codec } => MediaFormat::Audio(audio_codec_from_code(*codec)?),
            WireFormat::MpegTs => MediaFormat::MpegTs,
            WireFormat::Bytes => MediaFormat::Bytes,
        })
    }
}

impl WireMetadata {
    /// Capture a buffer's [`Metadata`] for the wire.
    ///
    /// `forward_custom_keys` selects which byte-valued custom entries
    /// ([`Metadata::get_bytes`]) are serialized; everything else in the
    /// custom map is type-erased and cannot cross the wire.
    pub fn from_metadata(metadata: &Metadata, forward_custom_keys: &[&'static str]) -> Self {
        let custom = forward_custom_keys
            .iter()
            .filter_map(|key| {
                metadata
                    .get_bytes(key)
                    .map(|bytes| (key.to_string(), bytes.to_vec()))
            })
            .collect();

        Self {
            pts_nanos: metadata.pts.nanos(),
            dts_nanos: metadata.dts.nanos(),
            duration_nanos: metadata.duration.nanos(),
            sequence: metadata.sequence,
            stream_id: metadata.stream_id,
            flags: metadata.flags.bits(),
            offset: metadata.offset,
            format: metadata
                .format
                .as_ref()
                .and_then(WireFormat::from_media_format),
            custom,
        }
    }

    /// Reconstruct in-memory [`Metadata`] from the wire representation.
    pub fn to_metadata(&self) -> Metadata {
        let mut metadata = Metadata::new();
        metadata.pts = ClockTime::from_nanos(self.pts_nanos);
        metadata.dts = ClockTime::from_nanos(self.dts_nanos);
        metadata.duration = ClockTime::from_nanos(self.duration_nanos);
        metadata.sequence = self.sequence;
        metadata.stream_id = self.stream_id;
        metadata.flags = BufferFlags::from_bits(self.flags);
        metadata.offset = self.offset;
        metadata.format = self.format.as_ref().and_then(WireFormat::to_media_format);

        let mut unknown: Vec<(String, Vec<u8>)> = Vec::new();
        for (key, bytes) in &self.custom {
            match KNOWN_CUSTOM_KEYS.iter().find(|known| *known == key) {
                Some(known) => metadata.set_bytes(known, bytes.clone()),
                None => unknown.push((key.clone(), bytes.clone())),
            }
        }
        if !unknown.is_empty() {
            metadata.set(CUSTOM_META, unknown);
        }

        metadata
    }

    /// Serialize into attachment bytes: magic + version + rkyv payload.
    pub fn encode(&self) -> Vec<u8> {
        let body = rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("WireMetadata serialization cannot fail");
        let mut out = Vec::with_capacity(3 + body.len());
        out.extend_from_slice(&WIRE_MAGIC);
        out.push(WIRE_VERSION);
        out.extend_from_slice(&body);
        out
    }

    /// Deserialize from attachment bytes.
    ///
    /// Returns `None` for foreign attachments (wrong magic), unknown
    /// versions, or malformed payloads — receivers fall back to legacy
    /// (fabricated) metadata in that case.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 3 || bytes[..2] != WIRE_MAGIC || bytes[2] != WIRE_VERSION {
            return None;
        }
        // Copy into an aligned buffer for rkyv validation/access.
        let mut aligned = rkyv::util::AlignedVec::<16>::new();
        aligned.extend_from_slice(&bytes[3..]);
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(&aligned).ok()
    }
}

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
        Some(MediaFormat::Rtp(_)) | Some(MediaFormat::Bytes) | None => Encoding::ZENOH_BYTES,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn full_metadata() -> Metadata {
        let mut metadata = Metadata::new();
        metadata.pts = ClockTime::from_nanos(1_000_000_007);
        metadata.dts = ClockTime::from_nanos(999_999_999);
        metadata.duration = ClockTime::from_nanos(33_333_333);
        metadata.sequence = 42;
        metadata.stream_id = 7;
        metadata.flags = BufferFlags::SYNC_POINT | BufferFlags::DISCONT;
        metadata.offset = Some(123_456);
        metadata.format = Some(MediaFormat::Video(VideoCodec::H264));
        metadata.set_bytes("stanag/klv", vec![0x06, 0x0E, 0x2B, 0x34]);
        metadata
    }

    #[test]
    fn roundtrip_preserves_all_fields() {
        let original = full_metadata();
        let wire = WireMetadata::from_metadata(&original, &["stanag/klv"]);
        let bytes = wire.encode();
        let decoded = WireMetadata::decode(&bytes).expect("decode");
        let restored = decoded.to_metadata();

        assert_eq!(restored.pts, original.pts);
        assert_eq!(restored.dts, original.dts);
        assert_eq!(restored.duration, original.duration);
        assert_eq!(restored.sequence, original.sequence);
        assert_eq!(restored.stream_id, original.stream_id);
        assert_eq!(restored.flags, original.flags);
        assert_eq!(restored.offset, original.offset);
        assert_eq!(restored.format, original.format);
        assert_eq!(
            restored.get_bytes("stanag/klv"),
            original.get_bytes("stanag/klv")
        );
    }

    #[test]
    fn none_sentinels_survive() {
        let mut metadata = Metadata::new();
        metadata.pts = ClockTime::NONE;
        metadata.dts = ClockTime::NONE;
        metadata.duration = ClockTime::NONE;
        let wire = WireMetadata::from_metadata(&metadata, &[]);
        assert_eq!(wire.pts_nanos, u64::MAX);
        let restored = WireMetadata::decode(&wire.encode()).unwrap().to_metadata();
        assert!(restored.pts.is_none());
        assert!(restored.dts.is_none());
        assert!(restored.duration.is_none());
    }

    #[test]
    fn video_raw_format_roundtrips() {
        let mut metadata = Metadata::new();
        metadata.format = Some(MediaFormat::VideoRaw(VideoFormat {
            width: 1920,
            height: 1080,
            pixel_format: PixelFormat::Nv12,
            framerate: Framerate {
                num: 30000,
                den: 1001,
            },
        }));
        let wire = WireMetadata::from_metadata(&metadata, &[]);
        let restored = WireMetadata::decode(&wire.encode()).unwrap().to_metadata();
        assert_eq!(restored.format, metadata.format);
    }

    #[test]
    fn unknown_custom_keys_are_aggregated() {
        let wire = WireMetadata {
            pts_nanos: u64::MAX,
            dts_nanos: u64::MAX,
            duration_nanos: u64::MAX,
            sequence: 0,
            stream_id: 0,
            flags: 0,
            offset: None,
            format: None,
            custom: vec![("future/thing".to_string(), vec![1, 2, 3])],
        };
        let restored = wire.to_metadata();
        let aggregated = restored
            .get::<Vec<(String, Vec<u8>)>>(CUSTOM_META)
            .expect("unknown keys aggregated");
        assert_eq!(aggregated[0].0, "future/thing");
        assert_eq!(aggregated[0].1, vec![1, 2, 3]);
    }

    #[test]
    fn foreign_and_legacy_attachments_are_rejected_gracefully() {
        assert!(WireMetadata::decode(b"").is_none());
        assert!(WireMetadata::decode(b"hi").is_none());
        assert!(WireMetadata::decode(b"not an attachment").is_none());
        // Right magic, unknown version.
        let mut bytes = WireMetadata::from_metadata(&Metadata::new(), &[]).encode();
        bytes[2] = 99;
        assert!(WireMetadata::decode(&bytes).is_none());
        // Right magic + version, garbage payload.
        assert!(WireMetadata::decode(&[0x50, 0x58, WIRE_VERSION, 0xFF]).is_none());
    }

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
    fn all_codes_roundtrip() {
        for codec in [
            VideoCodec::H264,
            VideoCodec::H265,
            VideoCodec::Vp8,
            VideoCodec::Vp9,
            VideoCodec::Av1,
        ] {
            assert_eq!(
                video_codec_from_code(video_codec_to_code(codec)),
                Some(codec)
            );
        }
        for codec in [
            AudioCodec::Opus,
            AudioCodec::Aac,
            AudioCodec::Mp3,
            AudioCodec::Pcmu,
            AudioCodec::Pcma,
            AudioCodec::Vorbis,
            AudioCodec::Eac3,
        ] {
            assert_eq!(
                audio_codec_from_code(audio_codec_to_code(codec)),
                Some(codec)
            );
        }
        for code in 0..15u8 {
            let format = pixel_format_from_code(code).unwrap();
            assert_eq!(pixel_format_to_code(format), code);
        }
    }
}
