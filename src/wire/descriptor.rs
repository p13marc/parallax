//! `IpcDescriptor` — the fixed-size POD **projection** of
//! [`WireMetadata`](super::WireMetadata) (#219).
//!
//! # Projection, not a second schema
//!
//! The zenoh attachment and this descriptor carry the same information; they
//! differ only in how. A network attachment is variable-length, so rkyv over
//! [`WireMetadata`](super::WireMetadata) costs nothing for absent fields. The
//! IPC ring is a fixed-stride array in shared memory on the **per-buffer hot
//! path**, so it is plain old data written scalar-by-scalar straight from
//! `Metadata` — no intermediate struct, no serialization, no allocation.
//!
//! That is a performance decision, not a schema decision, and the difference
//! matters: `tests/wire_projection.rs` round-trips a corpus through *both*
//! encodings and asserts they restore an identical `Metadata`. When they
//! disagree, one of them is wrong.
//!
//! Both encodings read the same [`codes`] table, so neither is
//! sensitive to the order of the enums it describes.
//!
//! # What a projection cannot carry
//!
//! One thing, documented and tested: the **custom map**. It is type-erased and
//! unbounded; its known byte-valued entries overflow through the control
//! socket (`ControlMessage::MetaOverflow`), flagged by presence bit 2. The
//! zenoh attachment carries them inline, because it can.
//!
//! # Layout
//!
//! 192 bytes, three cache lines. It was 128 until #219 gave it the plane
//! layout: a strided frame is the common case for dav1d and VA-API producers,
//! and pushing a `PlaneLayout` to the socket overflow would defeat the premise
//! of the descriptor ring — that a buffer crosses a process boundary without
//! touching the socket at all.

use super::codes;
use crate::clock::ClockTime;
use crate::error::{Error, Result};
use crate::format::{
    AudioFormat, Framerate, MAX_PLANES, MediaFormat, PlaneDesc, PlaneLayout, RtpFormat, VideoFormat,
};
use crate::memory::SharedIpcSlotRef;
use crate::metadata::{BufferFlags, Metadata, RtpMeta};

/// Bit 0: `byte_offset` present.
const PRESENCE_OFFSET: u8 = 1 << 0;
/// Bit 1: the RTP fields are meaningful.
const PRESENCE_RTP: u8 = 1 << 1;
/// Bit 2: a `MetaOverflow` control message precedes this descriptor.
const PRESENCE_META_OVERFLOW: u8 = 1 << 2;
/// Bit 3: the plane layout is meaningful (#219).
const PRESENCE_PLANES: u8 = 1 << 3;
/// `Metadata.color` is present (#265).
const PRESENCE_COLOR: u8 = 1 << 4;

/// One buffer's worth of wire state: the slot reference plus every fixed
/// metadata field. 192 bytes, plain old data.
///
/// See the module docs for why this is a projection of
/// [`WireMetadata`](super::WireMetadata) rather than a schema of its own.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct IpcDescriptor {
    /// Arena the slot lives in (process-unique since #178).
    pub arena_id: u64,
    /// Channel-monotonic sequence — the ack correlation id.
    pub seq: u64,
    /// Raw `ClockTime` nanos; `u64::MAX` = NONE. Round-trips exactly.
    pub pts: u64,
    /// See `pts`.
    pub dts: u64,
    /// See `pts`.
    pub duration: u64,
    /// `Metadata.sequence`.
    pub sequence: u64,
    /// `Metadata.offset`, valid iff presence bit 0.
    pub byte_offset: u64,
    /// Slot data offset within the arena.
    pub data_offset: u64,
    /// Payload length.
    pub len: u64,
    /// `Metadata.stream_id`.
    pub stream_id: u32,
    /// Slot index within the arena.
    pub slot_index: u32,
    /// `BufferFlags::bits()`.
    pub flags: u8,
    /// Presence bits; see the `PRESENCE_*` constants.
    pub presence: u8,
    /// `MediaFormat` tag — [`codes::FMT_NONE`] and friends.
    pub fmt_tag: u8,
    /// `RtpMeta.pt`.
    pub rtp_pt: u8,
    /// `RtpMeta.seq`.
    pub rtp_seq: u16,
    /// `RtpMeta.marker` (0/1).
    pub rtp_marker: u8,
    /// How many entries of `plane_offset`/`plane_stride` are meaningful,
    /// valid iff presence bit 3.
    pub plane_count: u8,
    /// `RtpMeta.ts`.
    pub rtp_ts: u32,
    /// `RtpMeta.ssrc`.
    pub rtp_ssrc: u32,
    /// MediaFormat payload word 1; meaning per `fmt_tag` (see `encode`).
    pub fmt_a: u32,
    /// MediaFormat payload word 2.
    pub fmt_b: u32,
    /// MediaFormat payload word 3.
    pub fmt_c: u32,
    /// MediaFormat payload word 4.
    pub fmt_d: u32,
    /// MediaFormat payload word 5.
    pub fmt_e: u32,
    /// Per-plane byte offsets (#219), valid iff presence bit 3.
    pub plane_offset: [u32; MAX_PLANES],
    /// Per-plane byte strides (#219), valid iff presence bit 3.
    pub plane_stride: [u32; MAX_PLANES],
    /// Colorimetry as H.273 codes (#265) — `[matrix, range, primaries,
    /// transfer]`, valid iff presence bit 4.
    ///
    /// This is the field the pad below was reserved for, and it cost nothing:
    /// four bytes out of forty-four, so the descriptor is still 192 bytes and
    /// `IPC_RING_VERSION` is untouched.
    pub color: [u8; 4],
    /// Spare. 192 rather than the bytes the fields need: three whole cache
    /// lines, and room so the next field to cross a process boundary is not
    /// another layout break.
    _pad: [u8; 40],
}

const _: () = assert!(std::mem::size_of::<IpcDescriptor>() == 192);
const _: () = assert!(std::mem::align_of::<IpcDescriptor>() == 8);

fn bad_wire(what: String) -> Error {
    // Unreachable when the version gate holds: both sides run the same
    // descriptor lineage. Reaching this means segment corruption.
    Error::InvalidSegment(format!("ipc descriptor: unknown {what}"))
}

impl IpcDescriptor {
    /// Encode a slot reference plus metadata. Drops the custom map — the
    /// element layer overflows its known entries through the control socket
    /// and flags them with [`set_meta_overflow`](Self::set_meta_overflow).
    pub fn encode(seq: u64, slot: &SharedIpcSlotRef, meta: &Metadata) -> Self {
        let mut d = Self {
            arena_id: slot.arena_id,
            seq,
            pts: meta.pts.nanos(),
            dts: meta.dts.nanos(),
            duration: meta.duration.nanos(),
            sequence: meta.sequence,
            byte_offset: meta.offset.unwrap_or(0),
            data_offset: slot.data_offset as u64,
            len: slot.len as u64,
            stream_id: meta.stream_id,
            slot_index: slot.slot_index,
            flags: meta.flags.bits(),
            presence: 0,
            color: [0; 4],
            fmt_tag: codes::FMT_NONE,
            rtp_pt: 0,
            rtp_seq: 0,
            rtp_marker: 0,
            plane_count: 0,
            rtp_ts: 0,
            rtp_ssrc: 0,
            fmt_a: 0,
            fmt_b: 0,
            fmt_c: 0,
            fmt_d: 0,
            fmt_e: 0,
            plane_offset: [0; MAX_PLANES],
            plane_stride: [0; MAX_PLANES],
            _pad: [0; 40],
        };
        if meta.offset.is_some() {
            d.presence |= PRESENCE_OFFSET;
        }
        if let Some(rtp) = meta.rtp {
            d.presence |= PRESENCE_RTP;
            d.rtp_pt = rtp.pt;
            d.rtp_seq = rtp.seq;
            d.rtp_marker = rtp.marker as u8;
            d.rtp_ts = rtp.ts;
            d.rtp_ssrc = rtp.ssrc;
        }
        // Only the *declared* layout travels (#219). `plane_layout()`
        // synthesises a packed one from the geometry, which the far side can
        // do for itself for free.
        if let Some(layout) = &meta.planes {
            d.presence |= PRESENCE_PLANES;
            let planes = layout.planes();
            d.plane_count = planes.len() as u8;
            for (i, plane) in planes.iter().enumerate() {
                d.plane_offset[i] = plane.offset as u32;
                d.plane_stride[i] = plane.stride as u32;
            }
        }
        // Colour (#265) rides its own bytes rather than the `fmt_*` words:
        // those are exhausted by `VideoRaw`'s five, and colour has to travel
        // on an *encoded* buffer too, where the demuxer stamps it.
        if let Some(color) = meta.color {
            d.color = [
                codes::color_matrix_to_code(color.matrix),
                codes::color_range_to_code(color.range),
                codes::color_primaries_to_code(color.primaries),
                codes::transfer_to_code(color.transfer),
            ];
            d.presence |= PRESENCE_COLOR;
        }
        match &meta.format {
            None => {}
            Some(MediaFormat::VideoRaw(v)) => {
                d.fmt_tag = codes::FMT_VIDEO_RAW;
                d.fmt_a = v.width;
                d.fmt_b = v.height;
                d.fmt_c = codes::pixel_format_to_code(v.pixel_format) as u32;
                d.fmt_d = v.framerate.num;
                d.fmt_e = v.framerate.den;
            }
            Some(MediaFormat::Video(c)) => {
                d.fmt_tag = codes::FMT_VIDEO;
                d.fmt_a = codes::video_codec_to_code(*c) as u32;
                // Coded geometry (#236) rides the same four words
                // FMT_VIDEO_RAW already uses. Zero means "the producer did
                // not know".
                if let Some(cf) = &meta.coded {
                    d.fmt_b = cf.width;
                    d.fmt_c = cf.height;
                    d.fmt_d = cf.framerate.num;
                    d.fmt_e = cf.framerate.den;
                }
            }
            Some(MediaFormat::AudioRaw(a)) => {
                d.fmt_tag = codes::FMT_AUDIO_RAW;
                d.fmt_a = a.sample_rate;
                d.fmt_b = a.channels as u32;
                d.fmt_c = codes::sample_format_to_code(a.sample_format) as u32;
            }
            Some(MediaFormat::Audio(c)) => {
                d.fmt_tag = codes::FMT_AUDIO;
                d.fmt_a = codes::audio_codec_to_code(*c) as u32;
            }
            Some(MediaFormat::Rtp(r)) => {
                d.fmt_tag = codes::FMT_RTP;
                d.fmt_a = r.payload_type as u32;
                d.fmt_b = r.clock_rate;
                let (code, dynamic) = codes::rtp_encoding_to_code(r.encoding);
                d.fmt_c = code as u32;
                d.fmt_d = dynamic as u32;
            }
            Some(MediaFormat::MpegTs) => d.fmt_tag = codes::FMT_MPEG_TS,
            Some(MediaFormat::Bytes) => d.fmt_tag = codes::FMT_BYTES,
        }
        d
    }

    /// Mark that a `MetaOverflow` control message precedes this descriptor.
    pub fn set_meta_overflow(&mut self) {
        self.presence |= PRESENCE_META_OVERFLOW;
    }

    /// Whether the element layer must collect overflow metadata first.
    pub fn has_meta_overflow(&self) -> bool {
        self.presence & PRESENCE_META_OVERFLOW != 0
    }

    /// Decode back into a slot reference and metadata (custom map empty —
    /// the element layer attaches overflow entries afterwards).
    ///
    /// Unknown codes are an **error** here, unlike on the zenoh side. Both
    /// ends of an IPC channel are parallax and the handshake gates on exact
    /// version equality, so an unknown code means segment corruption rather
    /// than a peer this build has not heard of.
    pub fn decode(&self) -> Result<(SharedIpcSlotRef, Metadata)> {
        let slot = SharedIpcSlotRef {
            arena_id: self.arena_id,
            slot_index: self.slot_index,
            data_offset: self.data_offset as usize,
            len: self.len as usize,
        };
        // Field-by-field: `custom` is private, so no struct literal/FRU.
        let mut meta = Metadata::new();
        meta.pts = ClockTime::from_nanos(self.pts);
        meta.dts = ClockTime::from_nanos(self.dts);
        meta.duration = ClockTime::from_nanos(self.duration);
        meta.sequence = self.sequence;
        meta.stream_id = self.stream_id;
        meta.flags = BufferFlags::from_bits(self.flags);
        if self.presence & PRESENCE_OFFSET != 0 {
            meta.offset = Some(self.byte_offset);
        }
        if self.presence & PRESENCE_RTP != 0 {
            meta.rtp = Some(RtpMeta {
                seq: self.rtp_seq,
                ts: self.rtp_ts,
                ssrc: self.rtp_ssrc,
                pt: self.rtp_pt,
                marker: self.rtp_marker != 0,
            });
        }
        if self.presence & PRESENCE_COLOR != 0 {
            let [m, r, p, t] = self.color;
            // Both ends of an IPC channel are parallax, so an unknown code is
            // a deployment error and says so — unlike the zenoh side, where a
            // foreign publisher is an expected thing to meet.
            meta.color = Some(crate::format::ColorInfo {
                matrix: codes::color_matrix_from_code(m)
                    .ok_or_else(|| bad_wire(format!("ColorMatrix {m}")))?,
                range: codes::color_range_from_code(r)
                    .ok_or_else(|| bad_wire(format!("ColorRange {r}")))?,
                primaries: codes::color_primaries_from_code(p)
                    .ok_or_else(|| bad_wire(format!("ColorPrimaries {p}")))?,
                transfer: codes::transfer_from_code(t)
                    .ok_or_else(|| bad_wire(format!("TransferFunction {t}")))?,
            });
        }
        meta.format = match self.fmt_tag {
            codes::FMT_NONE => None,
            codes::FMT_VIDEO_RAW => Some(MediaFormat::VideoRaw(VideoFormat {
                width: self.fmt_a,
                height: self.fmt_b,
                pixel_format: codes::pixel_format_from_code(self.fmt_c as u8)
                    .ok_or_else(|| bad_wire(format!("PixelFormat {}", self.fmt_c)))?,
                framerate: Framerate::new(self.fmt_d, self.fmt_e),
            })),
            codes::FMT_VIDEO => {
                let codec = codes::video_codec_from_code(self.fmt_a as u8)
                    .ok_or_else(|| bad_wire(format!("VideoCodec {}", self.fmt_a)))?;
                if self.fmt_b > 0 && self.fmt_c > 0 {
                    meta.coded = Some(crate::format::CodedVideoFormat::new(
                        codec,
                        self.fmt_b,
                        self.fmt_c,
                        Framerate::new(self.fmt_d, self.fmt_e),
                    ));
                }
                Some(MediaFormat::Video(codec))
            }
            codes::FMT_AUDIO_RAW => Some(MediaFormat::AudioRaw(AudioFormat {
                sample_rate: self.fmt_a,
                channels: self.fmt_b as u16,
                sample_format: codes::sample_format_from_code(self.fmt_c as u8)
                    .ok_or_else(|| bad_wire(format!("SampleFormat {}", self.fmt_c)))?,
            })),
            codes::FMT_AUDIO => Some(MediaFormat::Audio(
                codes::audio_codec_from_code(self.fmt_a as u8)
                    .ok_or_else(|| bad_wire(format!("AudioCodec {}", self.fmt_a)))?,
            )),
            codes::FMT_RTP => Some(MediaFormat::Rtp(RtpFormat {
                payload_type: self.fmt_a as u8,
                clock_rate: self.fmt_b,
                encoding: codes::rtp_encoding_from_code(self.fmt_c as u8, self.fmt_d as u8)
                    .ok_or_else(|| bad_wire(format!("RtpEncoding {}", self.fmt_c)))?,
            })),
            codes::FMT_MPEG_TS => Some(MediaFormat::MpegTs),
            codes::FMT_BYTES => Some(MediaFormat::Bytes),
            other => return Err(bad_wire(format!("format tag {other}"))),
        };
        // After the format, because `set_video_planes` would need geometry —
        // this writes the layout directly, and a layout without geometry is
        // exactly what the producer sent.
        if self.presence & PRESENCE_PLANES != 0 {
            let count = (self.plane_count as usize).min(MAX_PLANES);
            let planes: Vec<PlaneDesc> = (0..count)
                .map(|i| PlaneDesc {
                    offset: self.plane_offset[i] as usize,
                    stride: self.plane_stride[i] as usize,
                })
                .collect();
            meta.planes = Some(PlaneLayout::from_planes(&planes));
        }
        Ok((slot, meta))
    }
}
