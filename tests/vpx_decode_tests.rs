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
