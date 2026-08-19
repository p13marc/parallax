//! VA-API hardware decode integration tests (#193).
//!
//! Every test here green-skips where there is no hardware: the reference
//! device is one machine, CI's runner has no GPU, and which codecs a driver
//! offers depends on the *package* as much as the silicon (a patent-free
//! build such as Fedora's `libva-intel-media-driver` omits H.264 and HEVC on
//! a chip that has the engines).

#![cfg(all(feature = "vaapi", feature = "mkv-demux"))]

use parallax::element::{Demuxer, DemuxerProduce, Element};
use parallax::elements::demux::MkvDemux;
use parallax::elements::VaapiDecoder;
use std::io::Cursor;

const VP9_OPUS_WEBM: &[u8] = include_bytes!("fixtures/tiny_vp9_opus.webm");

/// One decoded frame, as the pipeline would see it.
struct Frame {
    pts_ms: u64,
    dims: (u32, u32),
    bytes: Vec<u8>,
}

/// Drive a decoder over the whole fixture, including the EOS drain.
///
/// `Element::process` returns at most one buffer, so a decoder that has
/// several frames ready hands them over across later calls — and whatever is
/// still held at the end comes out of `flush`, which is called until it
/// answers `None`.
fn drive(mut dec: impl Element, webm: &[u8]) -> Vec<Frame> {
    let mut demux = MkvDemux::new(Cursor::new(webm.to_vec()))
        .expect("fixture parses")
        .video_only();
    let mut frames = Vec::new();
    let push = |out: parallax::buffer::Buffer, frames: &mut Vec<Frame>| {
        let dims = out.metadata().video_dims().map(|d| (d.0, d.1)).unwrap();
        frames.push(Frame {
            pts_ms: out.metadata().pts.nanos() / 1_000_000,
            dims,
            bytes: out.as_bytes().to_vec(),
        });
    };
    loop {
        match demux.produce().expect("demux") {
            DemuxerProduce::Routed(routed) => {
                for (_, buf) in routed {
                    if let Some(out) = dec.process(buf).expect("decode") {
                        push(out, &mut frames);
                    }
                }
            }
            DemuxerProduce::Eos => break,
            other => panic!("unexpected {other:?}"),
        }
    }
    while let Some(out) = dec.flush().expect("flush") {
        push(out, &mut frames);
    }
    frames
}

/// Skip-or-decoder: the constructor's `Err` is the documented "use software
/// instead" answer, not a failure.
fn vp9_hw() -> Option<VaapiDecoder> {
    match VaapiDecoder::vp9() {
        Ok(dec) => Some(dec),
        Err(e) => {
            eprintln!("skipping: no VA-API VP9 decoder here — {e}");
            None
        }
    }
}

#[test]
fn vp9_decodes_the_whole_fixture_on_hardware() {
    let Some(dec) = vp9_hw() else { return };
    let frames = drive(dec, VP9_OPUS_WEBM);

    assert_eq!(frames.len(), 10, "all 10 frames decoded");
    for f in &frames {
        assert_eq!(f.dims, (64, 64), "geometry travels in metadata");
        assert_eq!(f.bytes.len(), 64 * 64 * 3 / 2, "packed NV12 payload");
    }
    let pts: Vec<u64> = frames.iter().map(|f| f.pts_ms).collect();
    assert!(pts.windows(2).all(|w| w[0] < w[1]), "monotonic PTS: {pts:?}");
    assert_eq!(pts[0], 0);
}

/// The frame is not merely the right size — it is the right *picture*.
///
/// VP9 decoding is normative: the spec's integer transforms leave no room
/// for a hardware decoder and a software one to disagree, so a bit-exact
/// comparison is available and is worth far more than eyeballing a dump. It
/// is also the only test that distinguishes the three ways this can go
/// wrong: a stride mistake shears the image, a plane-offset mistake shifts
/// the chroma, and a modifier mistake checkerboards everything.
#[cfg(feature = "vpx")]
#[test]
fn hardware_decode_is_bit_exact_with_software() {
    use parallax::elements::VpxDecoder;

    let Some(hw_dec) = vp9_hw() else { return };
    let hw = drive(hw_dec, VP9_OPUS_WEBM);
    let sw = drive(VpxDecoder::vp9().expect("software VP9"), VP9_OPUS_WEBM);
    assert_eq!(hw.len(), sw.len(), "same frame count");

    for (i, (h, s)) in hw.iter().zip(&sw).enumerate() {
        let (w, ht) = h.dims;
        let (w, ht) = (w as usize, ht as usize);
        let (cw, ch) = (w.div_ceil(2), ht.div_ceil(2));

        // Luma is laid out identically in NV12 and I420.
        assert_eq!(
            &h.bytes[..w * ht],
            &s.bytes[..w * ht],
            "frame {i}: luma differs between hardware and software"
        );

        // NV12 interleaves chroma; I420 keeps the planes apart.
        let hw_uv = &h.bytes[w * ht..];
        let sw_u = &s.bytes[w * ht..w * ht + cw * ch];
        let sw_v = &s.bytes[w * ht + cw * ch..];
        for row in 0..ch {
            for col in 0..cw {
                assert_eq!(
                    hw_uv[row * cw * 2 + col * 2],
                    sw_u[row * cw + col],
                    "frame {i}: U differs at ({col},{row})"
                );
                assert_eq!(
                    hw_uv[row * cw * 2 + col * 2 + 1],
                    sw_v[row * cw + col],
                    "frame {i}: V differs at ({col},{row})"
                );
            }
        }
    }
}
