//! #219: the IPC descriptor is a *projection* of `WireMetadata`, not a second
//! schema — and this is what makes that a fact rather than an intention.
//!
//! parallax has two ways out of a process, and until #219 they had two
//! independent metadata schemas that had already drifted apart:
//!
//! | | IPC descriptor | zenoh attachment |
//! |---|---|---|
//! | Format codes | enum **discriminants** | hand-written tables |
//! | `Metadata.rtp` | carried | **dropped** |
//! | `Metadata.planes` | dropped | **dropped** |
//! | `MediaFormat::Rtp` | `FMT_RTP` | **no mapping — silently lost** |
//!
//! Two vocabularies that happen to agree is the defect. So: one schema, two
//! encodings, and a corpus round-tripped through **both**, asserting they
//! restore an identical `Metadata`.
//!
//! It runs under **default features** on purpose. A test that only ran with
//! `zenoh` enabled would not be guarding the build that ships the IPC ring.
//!
//! ## The documented losses
//!
//! The projection is fixed-size POD, so exactly one thing cannot ride it: the
//! **custom map**, which is type-erased and unbounded. Its known byte-valued
//! entries overflow through the control socket instead
//! (`ControlMessage::MetaOverflow`). The attachment carries them inline,
//! because it can. Every other field must match, and the corpus below is
//! built to make a difference fail rather than pass by omission.

use parallax::clock::ClockTime;
use parallax::format::{
    AudioCodec, AudioFormat, Framerate, MediaFormat, PixelFormat, PlaneDesc, PlaneLayout,
    RtpEncoding, RtpFormat, SampleFormat, VideoCodec, VideoFormat,
};
use parallax::memory::SharedIpcSlotRef;
use parallax::metadata::{BufferFlags, Metadata, RtpMeta};
use parallax::wire::WireMetadata;
use parallax::wire::descriptor::IpcDescriptor;

fn slot() -> SharedIpcSlotRef {
    SharedIpcSlotRef {
        arena_id: 0xABCD,
        slot_index: 7,
        data_offset: 4096,
        len: 1234,
    }
}

/// Everything the descriptor cannot carry, listed once so a reader can check
/// the list against the module docs rather than against the code.
fn strip_projection_losses(meta: &Metadata) -> Metadata {
    let mut stripped = meta.clone();
    // The custom map: type-erased and unbounded. `Metadata` has no "clear
    // custom", so rebuild the fixed part.
    let mut clean = Metadata::new();
    clean.pts = stripped.pts;
    clean.dts = stripped.dts;
    clean.duration = stripped.duration;
    clean.sequence = stripped.sequence;
    clean.stream_id = stripped.stream_id;
    clean.flags = stripped.flags;
    clean.offset = stripped.offset;
    clean.format = stripped.format.take();
    clean.coded = stripped.coded;
    clean.rtp = stripped.rtp;
    clean.planes = stripped.planes;
    clean
}

/// Compare the two encodings' output against the input and against each other.
fn assert_projections_agree(original: &Metadata, what: &str) {
    let expected = strip_projection_losses(original);

    // Encoding 1: the fixed-size POD projection.
    let via_descriptor = IpcDescriptor::encode(0, &slot(), original)
        .decode()
        .unwrap_or_else(|e| panic!("{what}: descriptor decode failed: {e}"))
        .1;

    // Encoding 2: the rkyv attachment, through its real byte form.
    let wire = WireMetadata::from_metadata(original, &[]);
    let via_attachment = WireMetadata::decode(&wire.encode())
        .unwrap_or_else(|| panic!("{what}: attachment decode failed"))
        .to_metadata();

    for (label, got) in [
        ("descriptor", &via_descriptor),
        ("attachment", &via_attachment),
    ] {
        assert_eq!(got.pts, expected.pts, "{what}/{label}: pts");
        assert_eq!(got.dts, expected.dts, "{what}/{label}: dts");
        assert_eq!(got.duration, expected.duration, "{what}/{label}: duration");
        assert_eq!(got.sequence, expected.sequence, "{what}/{label}: sequence");
        assert_eq!(
            got.stream_id, expected.stream_id,
            "{what}/{label}: stream_id"
        );
        assert_eq!(got.flags, expected.flags, "{what}/{label}: flags");
        assert_eq!(got.offset, expected.offset, "{what}/{label}: offset");
        assert_eq!(got.format, expected.format, "{what}/{label}: format");
        assert_eq!(got.coded, expected.coded, "{what}/{label}: coded");
        assert_eq!(got.rtp, expected.rtp, "{what}/{label}: rtp");
        assert_eq!(got.planes, expected.planes, "{what}/{label}: planes");
    }
}

fn base() -> Metadata {
    let mut meta = Metadata::new();
    meta.pts = ClockTime::from_nanos(1_000_000_007);
    meta.dts = ClockTime::from_nanos(999_999_999);
    meta.duration = ClockTime::from_nanos(33_333_333);
    meta.sequence = 42;
    meta.stream_id = 7;
    meta.flags = BufferFlags::SYNC_POINT | BufferFlags::DISCONT;
    meta.offset = Some(123_456);
    meta
}

/// Every `MediaFormat` variant, including the two the corpus exists to catch:
/// `Rtp`, which the attachment used to lose entirely, and `Video` with coded
/// geometry, which shares its payload words with `VideoRaw`.
fn every_format() -> Vec<(&'static str, Metadata)> {
    let mut out = Vec::new();

    let mut m = base();
    m.format = None;
    out.push(("no format", m));

    for pixel_format in [
        PixelFormat::I420,
        PixelFormat::Nv12,
        PixelFormat::P010,
        PixelFormat::Bgra,
        PixelFormat::Gray16Le,
    ] {
        let mut m = base();
        m.format = Some(MediaFormat::VideoRaw(VideoFormat {
            width: 1920,
            height: 1080,
            pixel_format,
            framerate: Framerate::new(30000, 1001),
        }));
        out.push(("video raw", m));
    }

    for codec in [
        VideoCodec::H264,
        VideoCodec::H265,
        VideoCodec::Vp8,
        VideoCodec::Vp9,
        VideoCodec::Av1,
        VideoCodec::Mjpeg,
    ] {
        let mut m = base();
        m.format = Some(MediaFormat::Video(codec));
        out.push(("encoded video, no geometry", m));

        let mut m = base();
        m.set_coded_video_dims(codec, 1280, 720);
        out.push(("encoded video, with geometry", m));
    }

    for sample_format in [
        SampleFormat::S16,
        SampleFormat::S32,
        SampleFormat::F32,
        SampleFormat::U8,
    ] {
        let mut m = base();
        m.format = Some(MediaFormat::AudioRaw(AudioFormat {
            sample_rate: 48000,
            channels: 2,
            sample_format,
        }));
        out.push(("audio raw", m));
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
        let mut m = base();
        m.format = Some(MediaFormat::Audio(codec));
        out.push(("encoded audio", m));
    }

    // The variant the attachment used to drop on the floor.
    for encoding in [
        RtpEncoding::H264,
        RtpEncoding::H265,
        RtpEncoding::Vp8,
        RtpEncoding::Vp9,
        RtpEncoding::Opus,
        RtpEncoding::Pcmu,
        RtpEncoding::Pcma,
        RtpEncoding::Av1,
        RtpEncoding::Dynamic(97),
    ] {
        let mut m = base();
        m.format = Some(MediaFormat::Rtp(RtpFormat {
            payload_type: 96,
            clock_rate: 90000,
            encoding,
        }));
        out.push(("rtp", m));
    }

    let mut m = base();
    m.format = Some(MediaFormat::MpegTs);
    out.push(("mpeg-ts", m));

    let mut m = base();
    m.format = Some(MediaFormat::Bytes);
    out.push(("bytes", m));

    out
}

#[test]
fn every_media_format_projects_identically() {
    let corpus = every_format();
    assert!(corpus.len() >= 30, "corpus is {} cases", corpus.len());
    for (what, meta) in corpus {
        assert_projections_agree(&meta, what);
    }
}

/// The field the attachment used to drop while the descriptor carried it, so
/// `rtpsrc ! ipcsink` preserved the sequence number and `rtpsrc ! zenohsink`
/// did not.
#[test]
fn rtp_header_fields_survive_both_encodings() {
    let mut meta = base();
    meta.format = Some(MediaFormat::Video(VideoCodec::H264));
    meta.rtp = Some(RtpMeta {
        seq: 0xBEEF,
        ts: 0xDEADBEEF,
        ssrc: 0x12345678,
        pt: 96,
        marker: true,
    });
    assert_projections_agree(&meta, "rtp meta, marker set");

    meta.rtp = Some(RtpMeta {
        seq: 0,
        ts: 0,
        ssrc: 0,
        pt: 0,
        marker: false,
    });
    assert_projections_agree(&meta, "rtp meta, all zero");
}

/// The field **both** encodings dropped, which meant a strided producer's
/// frame arrived describing itself as packed. Geometry travels in-band; a
/// plane layout is part of the geometry.
#[test]
fn a_strided_plane_layout_survives_both_encodings() {
    // A dav1d-shaped I420 frame: plane heights aligned, so the strides are
    // wider than the visible width and the offsets are not derivable from it.
    let layout = PlaneLayout::from_planes(&[
        PlaneDesc {
            offset: 0,
            stride: 2048,
        },
        PlaneDesc {
            offset: 2048 * 1088,
            stride: 1024,
        },
        PlaneDesc {
            offset: 2048 * 1088 + 1024 * 544,
            stride: 1024,
        },
    ]);

    let mut meta = base();
    meta.set_video_planes(1920, 1080, PixelFormat::I420, layout);
    assert_projections_agree(&meta, "strided i420");

    // …and a packed frame declares no layout at all, which must also survive:
    // `plane_layout()` synthesises one from the geometry, and travelling that
    // would turn "packed" into "strided by coincidence".
    let mut packed = base();
    packed.set_video_dims(1920, 1080, PixelFormat::I420);
    assert_eq!(packed.planes, None);
    assert_projections_agree(&packed, "packed i420");
}

/// The sentinels have to survive as sentinels, not as very large numbers.
#[test]
fn absent_timestamps_survive_both_encodings() {
    let mut meta = Metadata::new();
    meta.pts = ClockTime::NONE;
    meta.dts = ClockTime::NONE;
    meta.duration = ClockTime::NONE;
    meta.offset = None;
    assert_projections_agree(&meta, "all absent");

    // Zero is a real timestamp, not an absent one.
    let zeroed = Metadata::new();
    assert_eq!(zeroed.pts, ClockTime::ZERO);
    assert_projections_agree(&zeroed, "zero timestamps");
}

/// The one documented loss, asserted rather than assumed: the descriptor drops
/// the custom map and the attachment carries it.
#[test]
fn the_custom_map_is_the_only_thing_the_projection_loses() {
    let mut meta = base();
    meta.format = Some(MediaFormat::Video(VideoCodec::H264));
    meta.set_bytes("stanag/klv", vec![0x06, 0x0E, 0x2B, 0x34]);

    let via_descriptor = IpcDescriptor::encode(0, &slot(), &meta).decode().unwrap().1;
    assert_eq!(
        via_descriptor.get_bytes("stanag/klv"),
        None,
        "the POD projection cannot carry the custom map — it overflows through \
         the control socket instead"
    );

    let wire = WireMetadata::from_metadata(&meta, &["stanag/klv"]);
    let via_attachment = WireMetadata::decode(&wire.encode()).unwrap().to_metadata();
    assert_eq!(
        via_attachment.get_bytes("stanag/klv"),
        meta.get_bytes("stanag/klv"),
        "the attachment carries it, because it can"
    );

    // Everything else still has to agree.
    assert_projections_agree(&meta, "with a custom entry");
}
