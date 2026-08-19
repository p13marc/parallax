//! VA-API hardware decode integration tests (#193).
//!
//! Every test here green-skips where there is no hardware: the reference
//! device is one machine, CI's runner has no GPU, and which codecs a driver
//! offers depends on the *package* as much as the silicon (a patent-free
//! build such as Fedora's `libva-intel-media-driver` omits H.264 and HEVC on
//! a chip that has the engines).

#![cfg(all(feature = "vaapi", feature = "mkv-demux"))]

use parallax::element::{Demuxer, DemuxerProduce, Element};
use parallax::elements::VaapiDecoder;
use parallax::elements::demux::MkvDemux;
use std::io::Cursor;

const VP9_OPUS_WEBM: &[u8] = include_bytes!("fixtures/tiny_vp9_opus.webm");
#[cfg(feature = "h264")]
const H264_AAC_MKV: &[u8] = include_bytes!("fixtures/tiny_h264_aac.mkv");
#[cfg(feature = "vpx")]
const VP8_VORBIS_WEBM: &[u8] = include_bytes!("fixtures/tiny_vp8_vorbis.webm");
const HEVC_MKV: &[u8] = include_bytes!("fixtures/tiny_hevc.mkv");
#[cfg(feature = "mp4-demux")]
const HEVC_MP4: &[u8] = include_bytes!("fixtures/tiny_hevc.mp4");

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
    assert!(
        pts.windows(2).all(|w| w[0] < w[1]),
        "monotonic PTS: {pts:?}"
    );
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
/// Compare hardware NV12 against software I420, frame by frame.
#[cfg(any(feature = "vpx", feature = "h264"))]
fn assert_bit_exact(hw: &[Frame], sw: &[Frame]) {
    assert!(!hw.is_empty(), "hardware decoded nothing");
    assert_eq!(hw.len(), sw.len(), "same frame count");

    for (i, (h, s)) in hw.iter().zip(sw).enumerate() {
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

#[cfg(feature = "vpx")]
#[test]
fn hardware_decode_is_bit_exact_with_software() {
    let Some(hw_dec) = vp9_hw() else { return };
    let hw = drive(hw_dec, VP9_OPUS_WEBM);
    let sw = drive(
        parallax::elements::VpxDecoder::vp9().expect("software VP9"),
        VP9_OPUS_WEBM,
    );
    assert_bit_exact(&hw, &sw);
}

/// VP8, the third codec this driver decodes.
#[cfg(feature = "vpx")]
#[test]
fn vp8_hardware_decode_is_bit_exact_with_software() {
    let hw_dec = match VaapiDecoder::vp8() {
        Ok(dec) => dec,
        Err(e) => {
            eprintln!("skipping: no VA-API VP8 decoder here — {e}");
            return;
        }
    };
    let hw = drive(hw_dec, VP8_VORBIS_WEBM);
    let sw = drive(
        parallax::elements::VpxDecoder::vp8().expect("software VP8"),
        VP8_VORBIS_WEBM,
    );
    assert_bit_exact(&hw, &sw);
}

/// The same proof for H.264, which is where the corpus really lives.
///
/// H.264 is absent from patent-free driver builds even on hardware that has
/// the engine, so this green-skips more often than the VP9 one — but where
/// it runs, it runs against `openh264` and must agree byte for byte.
#[cfg(feature = "h264")]
#[test]
fn h264_hardware_decode_is_bit_exact_with_software() {
    let hw_dec = match VaapiDecoder::h264() {
        Ok(dec) => dec,
        Err(e) => {
            eprintln!("skipping: no VA-API H.264 decoder here — {e}");
            return;
        }
    };
    let hw = drive(hw_dec, H264_AAC_MKV);
    let sw = drive(
        parallax::elements::H264Decoder::new().expect("software H.264"),
        H264_AAC_MKV,
    );
    assert_bit_exact(&hw, &sw);
}

/// HEVC has no software decoder in this tree, so there is nothing to compare
/// it against byte for byte. What there *is* is two independent container
/// paths to the same elementary stream — Matroska's `hvcC` in CodecPrivate
/// and MP4's in the sample entry — and they must produce identical pictures.
/// A mistake in either parameter-set path shows up as one side failing to
/// decode or the two disagreeing.
#[cfg(feature = "mp4-demux")]
#[test]
fn hevc_decodes_identically_from_both_containers() {
    let hw = |data| match VaapiDecoder::h265() {
        Ok(dec) => Some(drive(dec, data)),
        Err(e) => {
            eprintln!("skipping: no VA-API HEVC decoder here — {e}");
            None
        }
    };
    let Some(from_mkv) = hw(HEVC_MKV) else { return };
    let from_mp4 = {
        use parallax::element::{Demuxer, DemuxerProduce};
        use parallax::elements::demux::{Mp4Demux, Mp4DemuxSource};
        let mut demux = Mp4DemuxSource::new(
            Mp4Demux::new(
                std::io::Cursor::new(HEVC_MP4.to_vec()),
                HEVC_MP4.len() as u64,
            )
            .expect("fixture parses"),
        );
        let mut dec = VaapiDecoder::h265().expect("checked above");
        let mut frames = Vec::new();
        loop {
            match demux.produce().expect("demux") {
                DemuxerProduce::Routed(routed) => {
                    for (_, buf) in routed {
                        if let Some(out) = dec.process(buf).expect("decode") {
                            frames.push(out);
                        }
                    }
                }
                DemuxerProduce::Eos => break,
                other => panic!("unexpected {other:?}"),
            }
        }
        while let Some(out) = dec.flush().expect("flush") {
            frames.push(out);
        }
        frames
    };

    assert_eq!(
        from_mkv.len(),
        10,
        "all fixture frames decoded from Matroska"
    );
    assert_eq!(from_mp4.len(), from_mkv.len(), "same frame count from MP4");
    for (i, (m, p)) in from_mkv.iter().zip(&from_mp4).enumerate() {
        assert_eq!(m.dims, (320, 240));
        assert_eq!(
            m.bytes,
            p.as_bytes(),
            "frame {i} differs between the Matroska and MP4 paths"
        );
    }
}
