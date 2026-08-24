//! #218: a payloader emits one RTP packet per buffer.
//!
//! Every payloader used to concatenate an access unit's packets into a single
//! buffer — the source comment said so ("For now, concatenate all packets into
//! one buffer") — and `RtpSink` then wrapped the whole concatenation in **one**
//! RTP header and one `send_to`. So `… ! rtph264pay ! rtpsink` emitted a
//! malformed, oversized packet for any frame that fragmented, which is every
//! keyframe.
//!
//! What is tested here is the property that was missing, not the mechanism: a
//! frame larger than the MTU comes out as several packets, each within the
//! MTU, carrying the header fields only the packetizer can know — and the
//! depayloader puts the original bytes back together.

#![cfg(feature = "rtp")]

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{Element, Output, Transform};
use parallax::elements::{RtpH264Depay, RtpH264Pay, RtpH265Depay, RtpH265Pay, RtpOpusPay};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use std::sync::OnceLock;

fn arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(128 * 1024, 64).unwrap())
}

fn buffer(data: &[u8], pts_ns: u64) -> Buffer {
    let arena = arena();
    arena.reclaim();
    let mut slot = arena.acquire().expect("arena slot");
    slot.data_mut()[..data.len()].copy_from_slice(data);
    let mut metadata = Metadata::new();
    metadata.pts = ClockTime::from_nanos(pts_ns);
    Buffer::new(MemoryHandle::with_len(slot, data.len()), metadata)
}

fn packets(out: Output) -> Vec<Buffer> {
    match out {
        Output::Multiple(v) => v,
        Output::Single(b) => vec![b],
        Output::None => Vec::new(),
    }
}

/// An Annex-B keyframe well past any sane MTU.
fn big_keyframe() -> Vec<u8> {
    let mut au = Vec::new();
    au.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1E]); // SPS
    au.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x38, 0x80]); // PPS
    au.extend_from_slice(&[0, 0, 0, 1, 0x65]); // IDR
    // Deterministic filler so a reassembly check means something.
    au.extend((0..8000u32).map(|i| (i % 251) as u8));
    au
}

const MTU: usize = 1200;

#[test]
fn a_frame_larger_than_the_mtu_becomes_several_packets() {
    let au = big_keyframe();
    let mut pay = RtpH264Pay::new().with_mtu(MTU);
    let out = packets(pay.transform(buffer(&au, 1_000_000)).unwrap());

    assert!(
        out.len() > 5,
        "an 8 KB access unit at MTU {MTU} must fragment, got {} packet(s)",
        out.len()
    );
    for (i, packet) in out.iter().enumerate() {
        assert!(
            packet.len() <= MTU,
            "packet {i} is {} bytes, over the {MTU}-byte MTU",
            packet.len()
        );
        assert!(!packet.is_empty(), "packet {i} is empty");
    }
    assert_eq!(pay.stats().packets_out, out.len() as u64);
}

/// The three header fields only the packetizer knows. `RtpSink`'s fallback got
/// all three wrong for a fragmented frame: one sequence number per *buffer*, a
/// timestamp re-derived per buffer, and a marker bit taken from
/// `is_keyframe()` — which is a different question from "last packet of the
/// access unit".
#[test]
fn every_packet_carries_its_own_header_fields() {
    let mut pay = RtpH264Pay::new().with_mtu(MTU);
    let first = packets(pay.transform(buffer(&big_keyframe(), 1_000_000)).unwrap());
    let second = packets(pay.transform(buffer(&big_keyframe(), 2_000_000)).unwrap());

    let metas: Vec<_> = first
        .iter()
        .chain(second.iter())
        .map(|b| b.metadata().rtp.expect("per-packet RtpMeta"))
        .collect();

    // Sequence numbers are monotonic across the whole stream, not per frame.
    for pair in metas.windows(2) {
        assert_eq!(
            pair[1].seq,
            pair[0].seq.wrapping_add(1),
            "sequence numbers must be consecutive across access units"
        );
    }

    // One timestamp per access unit — that is what makes the packets one
    // frame to a receiver — and a different one for the next.
    let ts_first = metas[0].ts;
    assert!(
        first
            .iter()
            .all(|b| b.metadata().rtp.unwrap().ts == ts_first),
        "every packet of an access unit shares its timestamp"
    );
    let ts_second = second[0].metadata().rtp.unwrap().ts;
    assert_ne!(ts_first, ts_second, "a later frame has a later timestamp");
    // 90 kHz video clock, 1 ms apart.
    assert_eq!(ts_second.wrapping_sub(ts_first), 90);

    // The marker means "last packet of the access unit", and nothing else.
    for (i, packet) in first.iter().enumerate() {
        let marker = packet.metadata().rtp.unwrap().marker;
        assert_eq!(
            marker,
            i == first.len() - 1,
            "packet {i} of {}: marker must be set on the last packet only",
            first.len()
        );
    }
}

/// The round trip is the real proof: whatever the fragmentation, the
/// depayloader must hand back the bytes that went in.
#[test]
fn the_depayloader_reassembles_the_original_access_unit() {
    let au = big_keyframe();
    let mut pay = RtpH264Pay::new().with_mtu(MTU);
    let out = packets(pay.transform(buffer(&au, 0)).unwrap());
    assert!(out.len() > 1, "the fixture must fragment");

    let mut depay = RtpH264Depay::new();
    let mut reassembled = Vec::new();
    for packet in &out {
        // The depayloader is fed the RTP *payload*; RtpSrc strips the header.
        if let Some(frame) = depay.process(packet.clone()).unwrap() {
            reassembled.extend_from_slice(frame.as_bytes());
        }
    }

    assert_eq!(
        reassembled, au,
        "the reassembled access unit must be byte-identical"
    );
}

/// One Opus frame is one RTP packet (RFC 7587), so the marker is on it.
#[test]
fn an_audio_frame_that_fits_is_one_packet_with_the_marker_set() {
    let mut pay = RtpOpusPay::new();
    let out = packets(pay.transform(buffer(b"\xfc\x01\x02\x03opus", 0)).unwrap());

    assert_eq!(out.len(), 1);
    assert!(out[0].metadata().rtp.expect("RtpMeta").marker);
}

/// #240: the H.265 depayloader used to hand back the RTP payload verbatim —
/// payload header attached, no start codes, fragments never joined. This is
/// the same round-trip proof as the H.264 one, on the path that had none.
#[test]
fn the_h265_depayloader_reassembles_a_fragmented_access_unit() {
    // An IRAP slice (type 19) far past the MTU, plus a VPS/SPS/PPS ahead of
    // it — the shape a real keyframe takes.
    let mut au = Vec::new();
    for (nal_type, len) in [(32u8, 8usize), (33, 24), (34, 6)] {
        au.extend_from_slice(&[0, 0, 0, 1, nal_type << 1, 0x01]);
        au.extend((0..len).map(|i| (i % 251) as u8));
    }
    au.extend_from_slice(&[0, 0, 0, 1, 19 << 1, 0x01]);
    au.extend((0..6000u32).map(|i| (i % 251) as u8));

    let mut pay = RtpH265Pay::new().with_mtu(MTU);
    let out = packets(pay.transform(buffer(&au, 0)).unwrap());
    assert!(out.len() > 1, "the fixture must fragment");

    let mut depay = RtpH265Depay::new();
    let mut reassembled = Vec::new();
    let mut saw_keyframe = false;
    for packet in &out {
        if let Some(frame) = depay.process(packet.clone()).unwrap() {
            saw_keyframe |= frame.metadata().flags.is_keyframe();
            reassembled.extend_from_slice(frame.as_bytes());
        }
    }

    assert_eq!(
        reassembled, au,
        "the reassembled access unit must be byte-identical"
    );
    assert!(
        saw_keyframe,
        "the IRAP slice must be flagged as a decoder entry point"
    );
}
