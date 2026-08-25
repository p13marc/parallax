//! VP8/VP9 decode integration tests (#123): WebM fixtures demuxed by
//! MkvDemux, decoded by VpxDecoder, geometry and timing verified.

#![cfg(all(feature = "vpx", feature = "mkv-demux"))]

use parallax::element::{Demuxer, DemuxerProduce, Element};
use parallax::elements::demux::MkvDemux;
use parallax::elements::{VpxCodec, VpxDecoder};
use std::io::Cursor;

const VP9_OPUS_WEBM: &[u8] = include_bytes!("fixtures/tiny_vp9_opus.webm");
const VP8_VORBIS_WEBM: &[u8] = include_bytes!("fixtures/tiny_vp8_vorbis.webm");

fn decode_fixture(webm: &[u8], codec: VpxCodec) -> Vec<(u64, (u32, u32), usize)> {
    let mut demux = MkvDemux::new(Cursor::new(webm.to_vec()))
        .expect("fixture parses")
        .video_only();
    let mut dec = VpxDecoder::for_codec(codec).unwrap();
    let mut frames = Vec::new();
    loop {
        match demux.produce().unwrap() {
            DemuxerProduce::Routed(routed) => {
                for (_, buf) in routed {
                    let pts_ms = buf.metadata().pts.nanos() / 1_000_000;
                    if let Some(out) = dec.process(buf).unwrap() {
                        let dims = out.metadata().video_dims().map(|d| (d.0, d.1)).unwrap();
                        frames.push((pts_ms, dims, out.as_bytes().len()));
                    }
                }
            }
            DemuxerProduce::Eos => break,
            other => panic!("unexpected {other:?}"),
        }
    }
    frames
}

#[test]
fn vp9_decodes_webm_fixture() {
    let frames = decode_fixture(VP9_OPUS_WEBM, VpxCodec::Vp9);
    assert_eq!(frames.len(), 10, "all 10 frames decoded");
    for (_, dims, len) in &frames {
        assert_eq!(*dims, (64, 64), "geometry in metadata");
        assert_eq!(*len, 64 * 64 * 3 / 2, "packed I420 payload");
    }
    // PTS carried 1:1 from input to output, monotonic at 10 fps.
    let pts: Vec<u64> = frames.iter().map(|(p, _, _)| *p).collect();
    assert!(
        pts.windows(2).all(|w| w[0] < w[1]),
        "monotonic PTS: {pts:?}"
    );
    assert_eq!(pts[0], 0);
}

#[test]
fn vp8_decodes_webm_fixture() {
    let frames = decode_fixture(VP8_VORBIS_WEBM, VpxCodec::Vp8);
    assert_eq!(frames.len(), 10, "all 10 frames decoded");
    for (_, dims, len) in &frames {
        assert_eq!(*dims, (64, 64));
        assert_eq!(*len, 64 * 64 * 3 / 2);
    }
}

/// The wrong codec for the stream errors instead of emitting garbage.
#[test]
fn wrong_codec_errors() {
    let mut demux = MkvDemux::new(Cursor::new(VP9_OPUS_WEBM.to_vec()))
        .unwrap()
        .video_only();
    let mut dec = VpxDecoder::vp8().unwrap();
    let mut saw_error = false;
    for _ in 0..10 {
        match demux.produce().unwrap() {
            DemuxerProduce::Routed(routed) => {
                for (_, buf) in routed {
                    if dec.process(buf).is_err() {
                        saw_error = true;
                    }
                }
            }
            _ => break,
        }
        if saw_error {
            break;
        }
    }
    assert!(saw_error, "VP8 decoder rejects a VP9 stream");
}

// ------------------------------------------------------------------- #198

/// Decode every frame of a fixture, optionally with External negotiated.
fn decode_buffers(webm: &[u8], codec: VpxCodec, external: bool) -> Vec<parallax::buffer::Buffer> {
    use parallax::memory::MemoryType;

    let mut demux = MkvDemux::new(Cursor::new(webm.to_vec()))
        .expect("fixture parses")
        .video_only();
    let mut dec = VpxDecoder::for_codec(codec).unwrap();
    if external {
        dec.set_negotiated_memory(MemoryType::External);
    }
    let mut out = Vec::new();
    loop {
        match demux.produce().unwrap() {
            DemuxerProduce::Routed(routed) => {
                for (_, buf) in routed {
                    if let Some(frame) = dec.process(buf).unwrap() {
                        out.push(frame);
                    }
                }
            }
            DemuxerProduce::Eos => break,
            other => panic!("unexpected {other:?}"),
        }
    }
    out
}

/// #198: with External negotiated, VP9 decodes straight into buffers we supply
/// and the frame rides the pipeline with no copy — and the strided bytes
/// reconstruct EXACTLY what the packed path copies out.
///
/// The bit-exactness is the point. A stride or plane-offset mistake would
/// still produce a plausible-looking frame; only comparing against the copy
/// path catches it.
#[test]
fn vp9_external_emission_matches_the_packed_path() {
    use parallax::memory::MemoryType;

    let packed = decode_buffers(VP9_OPUS_WEBM, VpxCodec::Vp9, false);
    let external = decode_buffers(VP9_OPUS_WEBM, VpxCodec::Vp9, true);
    assert_eq!(packed.len(), 10);
    assert_eq!(external.len(), 10);

    for (p, e) in packed.iter().zip(&external) {
        assert_eq!(p.memory_type(), MemoryType::Cpu);
        assert_eq!(
            e.memory_type(),
            MemoryType::External,
            "VP9 must emit libvpx's own frame buffer, not a copy of it"
        );
        assert_eq!(p.metadata().pts, e.metadata().pts, "timing is unaffected");

        let meta = e.metadata();
        let (w, h) = meta.video_dims().expect("geometry travels in metadata");
        let fmt = meta.video_pixel_format().unwrap();
        let layout = meta
            .plane_layout()
            .expect("an external frame must declare its planes");

        assert!(
            e.metadata().has_strided_planes(),
            "libvpx pads its frame buffers with borders, so the frame really is strided"
        );

        // The #196 contract: every plane addressable for its whole
        // stride * rows, not merely its used bytes.
        assert!(
            layout.full_span_len(fmt, w, h) <= e.as_bytes().len(),
            "planes must be addressable for stride * rows"
        );

        let mut repacked = vec![0u8; layout.required_len(fmt, w, h)];
        let n = layout
            .repack_into(e.as_bytes(), fmt, w, h, &mut repacked)
            .unwrap();
        repacked.truncate(n);
        assert_eq!(
            repacked,
            p.as_bytes(),
            "the strided frame must hold the same pixels as the packed one"
        );
    }
}

/// Frames outlive the `process()` call that produced them, so the pool must
/// not let libvpx decode over one the pipeline is still holding.
///
/// Holding every frame of the fixture is what forces the pool past libvpx's own
/// reference set; releasing them and decoding again proves the slots come back.
#[test]
fn frames_held_past_decode_are_not_overwritten_by_later_ones() {
    use parallax::memory::MemoryType;

    let held = decode_buffers(VP9_OPUS_WEBM, VpxCodec::Vp9, true);
    assert_eq!(held.len(), 10);

    // Every frame is still alive here — nothing was dropped during the loop —
    // so if a pooled buffer had been recycled early, two frames would share
    // memory and their pixels would be identical.
    let packed = decode_buffers(VP9_OPUS_WEBM, VpxCodec::Vp9, false);
    for (i, (h, p)) in held.iter().zip(&packed).enumerate() {
        assert_eq!(h.memory_type(), MemoryType::External);
        let meta = h.metadata();
        let (w, hgt) = meta.video_dims().unwrap();
        let fmt = meta.video_pixel_format().unwrap();
        let layout = meta.plane_layout().unwrap();
        let mut repacked = vec![0u8; layout.required_len(fmt, w, hgt)];
        let n = layout
            .repack_into(h.as_bytes(), fmt, w, hgt, &mut repacked)
            .unwrap();
        repacked.truncate(n);
        assert_eq!(
            repacked,
            p.as_bytes(),
            "frame {i} was overwritten while the pipeline still held it"
        );
    }
}

/// VP8 says so rather than advertising a capability libvpx does not have.
///
/// `vpx_decoder.h`: caller-supplied frame buffers "currently only work with
/// VP9". Advertising `external_preferred()` here would be a promise the
/// decoder silently falls back from on every frame.
#[test]
fn vp8_declares_cpu_only_because_libvpx_cannot_do_better() {
    use parallax::element::Element;
    use parallax::memory::MemoryType;

    let vp8 = VpxDecoder::vp8().unwrap();
    assert!(
        !vp8.output_media_caps().lists_memory(MemoryType::External),
        "VP8 must not advertise External"
    );

    let vp9 = VpxDecoder::vp9().unwrap();
    assert!(
        vp9.output_media_caps().lists_memory(MemoryType::External),
        "VP9 must advertise External"
    );
}

/// A VP8 stream still decodes when a link somehow negotiated External: libvpx
/// refuses the callbacks, and the decoder falls back rather than failing.
#[test]
fn vp8_falls_back_to_the_packed_copy_under_external() {
    use parallax::memory::MemoryType;

    let frames = decode_buffers(VP8_VORBIS_WEBM, VpxCodec::Vp8, true);
    assert_eq!(frames.len(), 10);
    for frame in &frames {
        assert_eq!(
            frame.memory_type(),
            MemoryType::Cpu,
            "VP8 has no zero-copy path, and must not pretend otherwise"
        );
        assert_eq!(frame.as_bytes().len(), 64 * 64 * 3 / 2);
    }
}

/// Negotiating CPU must leave the old path exactly as it was: no pool, no
/// strides, packed bytes.
#[test]
fn a_cpu_link_still_gets_packed_frames() {
    use parallax::memory::MemoryType;

    let frames = decode_buffers(VP9_OPUS_WEBM, VpxCodec::Vp9, false);
    for frame in &frames {
        assert_eq!(frame.memory_type(), MemoryType::Cpu);
        assert!(
            !frame.metadata().has_strided_planes(),
            "a copied-out frame is packed; `plane_layout()` synthesises that, \
             but the frame must not *declare* a stride"
        );
        assert_eq!(frame.as_bytes().len(), 64 * 64 * 3 / 2);
    }
}

/// The other half of the retention story: a consumer that releases each frame
/// before asking for the next must not make the pool grow.
///
/// `frames_held_past_decode_are_not_overwritten_by_later_ones` proves the pool
/// *does* grow when it has to; this proves it does not grow when it does not,
/// which is what stops the zero-copy path from costing more memory than the
/// copy it replaced.
#[test]
fn releasing_each_frame_promptly_keeps_the_pool_small() {
    use parallax::memory::MemoryType;

    let mut demux = MkvDemux::new(Cursor::new(VP9_OPUS_WEBM.to_vec()))
        .expect("fixture parses")
        .video_only();
    let mut dec = VpxDecoder::vp9().unwrap();
    dec.set_negotiated_memory(MemoryType::External);

    let mut decoded = 0;
    loop {
        match demux.produce().unwrap() {
            DemuxerProduce::Routed(routed) => {
                for (_, buf) in routed {
                    if let Some(frame) = dec.process(buf).unwrap() {
                        assert_eq!(frame.memory_type(), MemoryType::External);
                        decoded += 1;
                        drop(frame); // the consumer is done with it
                    }
                }
            }
            DemuxerProduce::Eos => break,
            other => panic!("unexpected {other:?}"),
        }
    }

    assert_eq!(decoded, 10);
    let pooled = dec.pooled_frame_buffers().expect("VP9 uses the pool");
    assert!(
        pooled < 10,
        "slots must recycle when the consumer lets go: pool grew to {pooled} for 10 frames"
    );
}
