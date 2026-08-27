//! `WireMetadata` — the internal metadata schema, shared by every parallax
//! wire (#219).
//!
//! # One semantic mapping, two encodings
//!
//! This type *is* the definition of what a `Metadata` means on a wire. Two
//! things encode it, and they encode the same thing:
//!
//! - the **zenoh attachment** serializes this struct with rkyv, because a
//!   network attachment is variable-length and a schema with `Option`s and a
//!   `Vec` costs nothing when the fields are absent;
//! - the **IPC descriptor** ([`descriptor`](super::descriptor)) is a
//!   fixed-size `#[repr(C)]` *projection* of it, written scalar-by-scalar
//!   straight from `Metadata` — no intermediate, no rkyv, nothing allocated on
//!   the per-buffer path.
//!
//! The descriptor is a projection rather than a second schema, and
//! `tests/wire_projection.rs` is what makes that a fact instead of an
//! intention: it round-trips a corpus of metadata through both encodings and
//! asserts they restore an identical `Metadata`, modulo the two losses the
//! POD form documents (the custom map, which overflows through the control
//! socket, and nothing else).
//!
//! Both live here, always compiled, because this module is the intersection of
//! a feature-gated transport (`zenoh`) and a default-compiled one (the IPC
//! ring) — and because a projection-equality test that only ran with `zenoh`
//! enabled would not be guarding the default build.

use crate::clock::ClockTime;
use crate::format::{
    AudioFormat, Framerate, MediaFormat, PlaneDesc, PlaneLayout, RtpFormat, VideoFormat,
};
use crate::metadata::{BufferFlags, Metadata, RtpMeta};

use super::codes;

/// Metadata key under which custom wire entries that could **not** be
/// interned are aggregated as a `Vec<(String, Vec<u8>)>`.
///
/// Since #220 this is the overflow bucket, not the normal path: a key is
/// re-attached under its real name unless it fails
/// [`Metadata::intern`](crate::metadata::Metadata::intern) — past the
/// per-process cap, or not shaped like `"domain/type"`. Nothing is discarded
/// either way.
pub const CUSTOM_META: &str = "wire/overflow";

/// Buffer metadata as serialized on a parallax wire.
///
/// Timestamps are nanoseconds with `u64::MAX` as the [`ClockTime::NONE`]
/// sentinel, matching `ClockTime`'s own representation.
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
    /// Coded picture geometry for encoded video (#236): `(width, height,
    /// fps_num, fps_den)`. `None` when the producer did not know it — the
    /// RTP depayloaders, for instance, know the codec and nothing else.
    pub coded: Option<(u32, u32, u32, u32)>,
    /// RTP header fields (#219).
    ///
    /// The zenoh attachment used to drop these while the IPC descriptor
    /// carried them, so `rtpsrc ! ipcsink` preserved the sequence number and
    /// `rtpsrc ! zenohsink` did not.
    pub rtp: Option<WireRtp>,
    /// Plane layout for a strided raw-video buffer (#194, #219).
    ///
    /// Both wires used to drop this, which meant a strided producer's frame
    /// arrived describing itself as packed — geometry travels in-band
    /// (the Geometry-in-Metadata invariant) and this is part of the geometry.
    pub planes: Option<WirePlanes>,
    /// What the stream says about its colour (#265), as H.273 codes:
    /// `(matrix, range, primaries, transfer)`.
    pub color: Option<(u8, u8, u8, u8)>,
    /// Whitelisted custom byte entries (e.g. KLV, SEI), as (key, bytes).
    pub custom: Vec<(String, Vec<u8>)>,
    /// Set on a terminal sample (#224); `None` on every media sample.
    ///
    /// It is a field rather than a separate attachment kind so a terminal
    /// carries the same `sequence` accounting as the frames before it: a
    /// receiver's gap detection keeps working right up to the last sample,
    /// and a zero-length *media* buffer stays distinguishable from a goodbye.
    pub terminal: Option<WireTerminal>,
}

mod wire_types {
    //! Inner module so `allow(missing_docs)` covers rkyv's generated resolver
    //! types (their fields cannot carry doc comments).
    #![allow(missing_docs)]

    /// Media format as serialized on the wire.
    ///
    /// Every code here comes from [`codes`](super::codes) — see that module
    /// for why there is exactly one table.
    #[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    #[rkyv(derive(Debug))]
    pub enum WireFormat {
        /// Raw (uncompressed) video.
        VideoRaw {
            /// Frame width in pixels.
            width: u32,
            /// Frame height in pixels.
            height: u32,
            /// Pixel format code.
            pixel_format: u8,
            /// Framerate numerator.
            fps_num: u32,
            /// Framerate denominator.
            fps_den: u32,
        },
        /// Encoded video.
        Video {
            /// Codec code.
            codec: u8,
        },
        /// Raw (uncompressed) audio.
        AudioRaw {
            /// Sample rate in Hz.
            sample_rate: u32,
            /// Channel count.
            channels: u16,
            /// Sample format code.
            sample_format: u8,
        },
        /// Encoded audio.
        Audio {
            /// Codec code.
            codec: u8,
        },
        /// An RTP stream (#219).
        ///
        /// This variant did not exist, so `MediaFormat::Rtp` was the one
        /// format that survived a *process* boundary and vanished across a
        /// *machine* boundary.
        Rtp {
            /// Payload type (0–127).
            payload_type: u8,
            /// RTP clock rate in Hz.
            clock_rate: u32,
            /// Encoding code.
            encoding: u8,
            /// Payload type for `RtpEncoding::Dynamic`; 0 otherwise.
            dynamic_pt: u8,
        },
        /// MPEG-TS packets.
        MpegTs,
        /// Raw bytes.
        Bytes,
    }

    /// RTP header fields, as serialized (#219).
    #[derive(Debug, Clone, Copy, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    #[rkyv(derive(Debug))]
    pub struct WireRtp {
        pub seq: u16,
        pub ts: u32,
        pub ssrc: u32,
        pub pt: u8,
        pub marker: bool,
    }

    /// A strided frame's plane layout, as serialized (#194, #219).
    ///
    /// `offset`/`stride` are `usize` in memory and `u32` here: a plane offset
    /// past 4 GiB inside one buffer is not a thing parallax can produce, and
    /// pinning the width is what keeps the POD projection a fixed size.
    #[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    #[rkyv(derive(Debug))]
    pub struct WirePlanes {
        /// `(offset, stride)` per plane, in plane order.
        pub planes: Vec<(u32, u32)>,
    }

    /// Why a published stream ended (#224).
    ///
    /// Parallax-private plane only. The `@media` plane forbids anything but
    /// pixels on the media key (RFC 07 §1), and learns the same fact from the
    /// publisher's liveliness token disappearing instead — which also covers
    /// `kill -9` and network partition, where no graceful sample is ever sent.
    #[derive(Debug, Clone, PartialEq, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
    #[rkyv(derive(Debug))]
    pub enum WireTerminal {
        /// The stream ended normally.
        Eos,
        /// The publishing pipeline failed.
        Error {
            /// What went wrong, for the receiver's bus and logs.
            message: String,
        },
    }
}
pub use wire_types::{WireFormat, WirePlanes, WireRtp, WireTerminal};

impl WireFormat {
    /// Convert an in-memory [`MediaFormat`] to its wire representation.
    ///
    /// Total since #219 — `MediaFormat::Rtp` used to return `None` here, which
    /// is how an RTP-formatted buffer lost its format on the way out.
    pub fn from_media_format(format: &MediaFormat) -> Option<Self> {
        Some(match format {
            MediaFormat::VideoRaw(vf) => WireFormat::VideoRaw {
                width: vf.width,
                height: vf.height,
                pixel_format: codes::pixel_format_to_code(vf.pixel_format),
                fps_num: vf.framerate.num,
                fps_den: vf.framerate.den,
            },
            MediaFormat::Video(codec) => WireFormat::Video {
                codec: codes::video_codec_to_code(*codec),
            },
            MediaFormat::AudioRaw(af) => WireFormat::AudioRaw {
                sample_rate: af.sample_rate,
                channels: af.channels,
                sample_format: codes::sample_format_to_code(af.sample_format),
            },
            MediaFormat::Audio(codec) => WireFormat::Audio {
                codec: codes::audio_codec_to_code(*codec),
            },
            MediaFormat::Rtp(rf) => {
                let (encoding, dynamic_pt) = codes::rtp_encoding_to_code(rf.encoding);
                WireFormat::Rtp {
                    payload_type: rf.payload_type,
                    clock_rate: rf.clock_rate,
                    encoding,
                    dynamic_pt,
                }
            }
            MediaFormat::MpegTs => WireFormat::MpegTs,
            MediaFormat::Bytes => WireFormat::Bytes,
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
                pixel_format: codes::pixel_format_from_code(*pixel_format)?,
                framerate: Framerate {
                    num: *fps_num,
                    den: *fps_den,
                },
            }),
            WireFormat::Video { codec } => {
                MediaFormat::Video(codes::video_codec_from_code(*codec)?)
            }
            WireFormat::AudioRaw {
                sample_rate,
                channels,
                sample_format,
            } => MediaFormat::AudioRaw(AudioFormat {
                sample_rate: *sample_rate,
                channels: *channels,
                sample_format: codes::sample_format_from_code(*sample_format)?,
            }),
            WireFormat::Audio { codec } => {
                MediaFormat::Audio(codes::audio_codec_from_code(*codec)?)
            }
            WireFormat::Rtp {
                payload_type,
                clock_rate,
                encoding,
                dynamic_pt,
            } => MediaFormat::Rtp(RtpFormat {
                payload_type: *payload_type,
                clock_rate: *clock_rate,
                encoding: codes::rtp_encoding_from_code(*encoding, *dynamic_pt)?,
            }),
            WireFormat::MpegTs => MediaFormat::MpegTs,
            WireFormat::Bytes => MediaFormat::Bytes,
        })
    }
}

impl WirePlanes {
    /// Capture a [`PlaneLayout`].
    pub fn from_layout(layout: &PlaneLayout) -> Self {
        Self {
            planes: layout
                .planes()
                .iter()
                .map(|p| (p.offset as u32, p.stride as u32))
                .collect(),
        }
    }

    /// Rebuild a [`PlaneLayout`].
    pub fn to_layout(&self) -> PlaneLayout {
        let planes: Vec<PlaneDesc> = self
            .planes
            .iter()
            .map(|(offset, stride)| PlaneDesc {
                offset: *offset as usize,
                stride: *stride as usize,
            })
            .collect();
        PlaneLayout::from_planes(&planes)
    }
}

impl WireMetadata {
    /// Capture a buffer's [`Metadata`] for the wire.
    ///
    /// **Every** byte-valued custom entry travels (#220). `forward_custom_keys`
    /// is an optional *filter*, not an allowlist: empty means "all of them".
    /// It used to be the other way round, with a two-entry hardcoded list in
    /// the core crate deciding what a downstream application was allowed to
    /// send.
    ///
    /// What still cannot cross is the type-erased half of the map — a
    /// `Box<dyn Any>` has no wire form. **Store it as bytes if you want it to
    /// travel.**
    pub fn from_metadata(metadata: &Metadata, forward_custom_keys: &[&'static str]) -> Self {
        let custom = metadata
            .byte_entries()
            .filter(|(key, _)| forward_custom_keys.is_empty() || forward_custom_keys.contains(key))
            .map(|(key, bytes)| (key.to_string(), bytes.to_vec()))
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
            coded: metadata
                .coded
                .map(|c| (c.width, c.height, c.framerate.num, c.framerate.den)),
            rtp: metadata.rtp.map(|r| WireRtp {
                seq: r.seq,
                ts: r.ts,
                ssrc: r.ssrc,
                pt: r.pt,
                marker: r.marker,
            }),
            // Only the *declared* layout travels: `Metadata::plane_layout()`
            // synthesises a packed one from the geometry, and re-deriving that
            // on the far side is free.
            planes: metadata.planes.as_ref().map(WirePlanes::from_layout),
            color: metadata.color.map(|c| {
                (
                    codes::color_matrix_to_code(c.matrix),
                    codes::color_range_to_code(c.range),
                    codes::color_primaries_to_code(c.primaries),
                    codes::transfer_to_code(c.transfer),
                )
            }),
            custom,
            terminal: None,
        }
    }

    /// A terminal marker carrying `sequence` (#224).
    ///
    /// The sequence is the publisher's, one past the last frame, so a
    /// receiver's gap detection covers the goodbye too — a terminal that
    /// arrives after samples were lost still says so.
    pub fn terminal(reason: WireTerminal, sequence: u64) -> Self {
        Self {
            pts_nanos: ClockTime::NONE.nanos(),
            dts_nanos: ClockTime::NONE.nanos(),
            duration_nanos: ClockTime::NONE.nanos(),
            sequence,
            stream_id: 0,
            flags: 0,
            offset: None,
            format: None,
            coded: None,
            rtp: None,
            planes: None,
            color: None,
            custom: Vec::new(),
            terminal: Some(reason),
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
        if let (Some(MediaFormat::Video(codec)), Some((w, h, num, den))) =
            (&metadata.format, self.coded)
        {
            metadata.coded = Some(crate::format::CodedVideoFormat::new(
                *codec,
                w,
                h,
                Framerate::new(num, den),
            ));
        }
        metadata.rtp = self.rtp.map(|r| RtpMeta {
            seq: r.seq,
            ts: r.ts,
            ssrc: r.ssrc,
            pt: r.pt,
            marker: r.marker,
        });
        metadata.planes = self.planes.as_ref().map(WirePlanes::to_layout);
        // A code with no variant here drops the whole `ColorInfo` rather than
        // guessing a component: a half-understood colour description is worse
        // than an honest "nothing was said", which consumers already handle.
        metadata.color = self.color.and_then(|(m, r, p, t)| {
            Some(crate::format::ColorInfo {
                matrix: codes::color_matrix_from_code(m)?,
                range: codes::color_range_from_code(r)?,
                primaries: codes::color_primaries_from_code(p)?,
                transfer: codes::transfer_from_code(t)?,
            })
        });

        // Re-attach under the real key (#220). `intern` leaks the key once,
        // bounded, so a receiver is no longer limited to a hardcoded list —
        // an application's own `myapp/telemetry` survives the hop.
        let mut overflow: Vec<(String, Vec<u8>)> = Vec::new();
        for (key, bytes) in &self.custom {
            match Metadata::intern(key) {
                Some(interned) => metadata.set_bytes(interned, bytes.clone()),
                // Past the intern cap, or not a `"domain/type"` key. The
                // entry is not dropped — it lands in the aggregate bucket,
                // where an application that knows what it is looking for can
                // still find it.
                None => overflow.push((key.clone(), bytes.clone())),
            }
        }
        if !overflow.is_empty() {
            metadata.set(CUSTOM_META, overflow);
        }

        metadata
    }

    /// Serialize into attachment bytes: magic + version + rkyv payload.
    pub fn encode(&self) -> Vec<u8> {
        let body = rkyv::to_bytes::<rkyv::rancor::Error>(self)
            .expect("WireMetadata serialization cannot fail");
        let mut out = Vec::with_capacity(3 + body.len());
        out.extend_from_slice(&super::WIRE_MAGIC);
        out.push(super::WIRE_VERSION);
        out.extend_from_slice(&body);
        out
    }

    /// Deserialize from attachment bytes.
    ///
    /// Returns `None` for foreign attachments (wrong magic), unknown
    /// versions, or malformed payloads. That is the **zenoh** skew policy and
    /// it is deliberately not the IPC one: an open network legitimately
    /// carries publishers parallax has never heard of, so a receiver falls
    /// back to fabricated metadata rather than erroring. Both ends of an IPC
    /// channel are parallax, where a mismatch is a deployment error and the
    /// handshake refuses it.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < 3 || bytes[..2] != super::WIRE_MAGIC || bytes[2] != super::WIRE_VERSION {
            return None;
        }
        // Copy into an aligned buffer for rkyv validation/access.
        let mut aligned = rkyv::util::AlignedVec::<16>::new();
        aligned.extend_from_slice(&bytes[3..]);
        rkyv::from_bytes::<Self, rkyv::rancor::Error>(&aligned).ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ClockTime;
    use crate::format::{AudioCodec, PixelFormat, VideoCodec};
    use crate::metadata::BufferFlags;
    use crate::wire::WIRE_VERSION;
    use codes::*;

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

    /// #220's headline: a key this build has never heard of arrives under its
    /// own name. It used to reach a two-entry hardcoded allowlist and, failing
    /// that, an aggregate bucket the application had to know to look in.
    #[test]
    fn a_key_a_receiver_never_heard_of_arrives_under_its_own_name() {
        let wire = WireMetadata {
            pts_nanos: u64::MAX,
            dts_nanos: u64::MAX,
            duration_nanos: u64::MAX,
            sequence: 0,
            stream_id: 0,
            flags: 0,
            offset: None,
            format: None,
            coded: None,
            rtp: None,
            planes: None,
            color: None,
            custom: vec![("myapp/telemetry".to_string(), vec![1, 2, 3])],
            terminal: None,
        };
        let restored = wire.to_metadata();
        let key = Metadata::intern("myapp/telemetry").expect("interns");
        assert_eq!(restored.get_bytes(key), Some(&[1u8, 2, 3][..]));
        assert!(
            restored
                .get::<Vec<(String, Vec<u8>)>>(CUSTOM_META)
                .is_none(),
            "an interned key must not also land in the overflow bucket"
        );
    }

    /// A key that is not shaped like one is not dropped either — it spills
    /// into the bucket, where an application that knows what it is looking for
    /// can still find it.
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
            coded: None,
            rtp: None,
            planes: None,
            color: None,
            custom: vec![("no slash here".to_string(), vec![1, 2, 3])],
            terminal: None,
        };
        let restored = wire.to_metadata();
        let aggregated = restored
            .get::<Vec<(String, Vec<u8>)>>(CUSTOM_META)
            .expect("unknown keys aggregated");
        assert_eq!(aggregated[0].0, "no slash here");
        assert_eq!(aggregated[0].1, vec![1, 2, 3]);
    }

    #[test]
    fn a_terminal_round_trips_and_is_distinguishable_from_an_empty_frame() {
        for reason in [
            WireTerminal::Eos,
            WireTerminal::Error {
                message: "the camera went away".to_string(),
            },
        ] {
            let wire = WireMetadata::terminal(reason.clone(), 41);
            let decoded = WireMetadata::decode(&wire.encode()).expect("our own attachment");
            assert_eq!(decoded.terminal, Some(reason));
            // The sequence rides along, so a receiver's gap detection covers
            // the goodbye too.
            assert_eq!(decoded.sequence, 41);
        }

        // A zero-length *media* sample is not a goodbye.
        let ordinary = WireMetadata::from_metadata(&Metadata::new(), &[]);
        assert_eq!(ordinary.terminal, None);
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
    fn coded_geometry_survives_the_attachment() {
        let mut metadata = Metadata::new();
        metadata.set_coded_video_dims(VideoCodec::H264, 1920, 1080);

        let wire = WireMetadata::from_metadata(&metadata, &[]);
        let restored = WireMetadata::decode(&wire.encode()).unwrap().to_metadata();

        assert_eq!(restored.format, Some(MediaFormat::Video(VideoCodec::H264)));
        assert_eq!(restored.coded_video_dims(), Some((1920, 1080)));
        assert_eq!(restored.video_dims(), None);
    }

    #[test]
    fn a_codec_without_geometry_stays_without_geometry() {
        // The RTP depayloaders stamp a codec and nothing else.
        let mut metadata = Metadata::new();
        metadata.format = Some(MediaFormat::Video(VideoCodec::Vp9));

        let wire = WireMetadata::from_metadata(&metadata, &[]);
        let restored = WireMetadata::decode(&wire.encode()).unwrap().to_metadata();

        assert_eq!(restored.format, Some(MediaFormat::Video(VideoCodec::Vp9)));
        assert!(restored.coded.is_none());
    }

    #[test]
    fn all_codes_roundtrip() {
        for codec in [
            VideoCodec::H264,
            VideoCodec::H265,
            VideoCodec::Vp8,
            VideoCodec::Vp9,
            VideoCodec::Av1,
            VideoCodec::Mjpeg,
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
