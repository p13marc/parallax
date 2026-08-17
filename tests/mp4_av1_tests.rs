//! AV1 + Opus in MP4 (#125): the stsd side-scan recovers what the mp4
//! crate drops, and samples pass through as plain OBUs / Opus packets.

#![cfg(feature = "mp4-demux")]

use parallax::elements::demux::{Mp4Codec, Mp4Demux};
use std::io::Cursor;

const AV1_OPUS_MP4: &[u8] = include_bytes!("fixtures/tiny_av1_opus.mp4");

fn open() -> Mp4Demux<Cursor<Vec<u8>>> {
    Mp4Demux::new(
        Cursor::new(AV1_OPUS_MP4.to_vec()),
        AV1_OPUS_MP4.len() as u64,
    )
    .unwrap()
}

#[test]
fn av1_and_opus_tracks_are_recognized() {
    let demux = open();

    let video_id = demux.video_track_id().expect("video track");
    let video = demux.track(video_id).unwrap();
    assert_eq!(video.codec, Mp4Codec::Av1);
    assert!(video.codec.is_video());
    let vinfo = video.video_info.as_ref().unwrap();
    assert_eq!((vinfo.width, vinfo.height), (64, 64));

    let audio_id = demux.audio_track_id().expect("audio track");
    let audio = demux.track(audio_id).unwrap();
    assert_eq!(audio.codec, Mp4Codec::Opus);
    assert!(audio.codec.is_audio());
    let ainfo = audio.audio_info.as_ref().unwrap();
    assert_eq!(ainfo.sample_rate, 48_000, "Opus decode rate");
    assert_eq!(ainfo.channels, 1, "mono sine fixture");
    let dops = ainfo.codec_private.as_ref().expect("dOps payload");
    assert!(dops.len() >= 11, "dOps has the fixed fields");
    assert_eq!(dops[1], 1, "dOps channel count");
}

/// AV1 samples come out untouched (OBUs, no length-prefix conversion) and
/// keyframes carry the sync flag from stss.
#[test]
fn av1_samples_pass_through_with_sync_flags() {
    let mut demux = open();
    let video_id = demux.video_track_id().unwrap();

    let mut keyframes = 0;
    let mut total = 0;
    let mut last_pts = None;
    while let Some(sample) = demux.read_sample(video_id).unwrap() {
        total += 1;
        if sample.is_keyframe {
            keyframes += 1;
        }
        // OBU stream: first byte's forbidden bit (0x80) must be clear.
        let first = sample.buffer.as_bytes()[0];
        assert_eq!(first & 0x80, 0, "OBU header forbidden bit clear");
        if let Some(last) = last_pts {
            assert!(sample.pts_ns > last, "monotonic PTS");
        }
        last_pts = Some(sample.pts_ns);
    }
    assert_eq!(total, 10, "all fixture frames");
    assert!(keyframes >= 1, "at least the leading keyframe is flagged");
}

/// The recovered Opus track decodes end-to-end when the codec feature is on.
#[cfg(feature = "opus")]
#[test]
fn opus_track_decodes() {
    use parallax::elements::OpusDecoder;
    use parallax::elements::codec::AudioDecoder;

    let mut demux = open();
    let audio_id = demux.audio_track_id().unwrap();
    let info = demux.track(audio_id).unwrap().audio_info.clone().unwrap();
    let mut dec = OpusDecoder::new(info.sample_rate, info.channels as u32).unwrap();

    let mut samples_out = 0usize;
    while let Some(sample) = demux.read_sample(audio_id).unwrap() {
        let decoded = dec.decode(sample.buffer.as_bytes()).unwrap();
        samples_out += decoded.samples_per_channel;
    }
    // ~1 s of audio at 48 kHz (minus priming).
    assert!(
        samples_out > 40_000,
        "decoded ~1s of PCM, got {samples_out} samples/ch"
    );
}

/// The recovered AV1 track decodes end-to-end when the codec feature is on:
/// OBUs straight into dav1d, display-ordered output with carried PTS and
/// packed I420 geometry in metadata.
#[cfg(feature = "av1-decode")]
#[test]
fn av1_track_decodes() {
    use parallax::element::Element;
    use parallax::elements::Dav1dDecoder;

    let mut demux = open();
    let video_id = demux.video_track_id().unwrap();
    let mut dec = Dav1dDecoder::new().unwrap();

    let mut out = Vec::new();
    while let Some(sample) = demux.read_sample(video_id).unwrap() {
        if let Some(buf) = dec.process(sample.buffer).unwrap() {
            out.push(buf);
        }
    }
    while let Some(buf) = dec.flush().unwrap() {
        out.push(buf);
    }

    assert_eq!(out.len(), 10, "all fixture frames decoded");
    let mut last = None;
    for buf in &out {
        let dims = buf.metadata().video_dims().unwrap();
        assert_eq!((dims.0, dims.1), (64, 64));
        assert_eq!(buf.as_bytes().len(), 64 * 64 * 3 / 2, "packed I420");
        let pts = buf.metadata().pts.nanos();
        if let Some(last) = last {
            assert!(pts > last, "display-ordered carried PTS");
        }
        last = Some(pts);
    }
}

/// #195: `send_data` holds the input `Buffer` zero-copy (no `to_vec`),
/// releasing it from dav1d's data-release callback. Dropping the decoder
/// joins dav1d's threads synchronously, so afterwards every input slot's
/// refcount must be back to exactly the clone this test kept.
#[cfg(feature = "av1-decode")]
#[test]
fn dav1d_releases_zero_copy_input_buffers() {
    use parallax::element::Element;
    use parallax::elements::Dav1dDecoder;

    let mut demux = open();
    let video_id = demux.video_track_id().unwrap();
    let mut dec = Dav1dDecoder::new().unwrap();

    let mut held = Vec::new();
    while let Some(sample) = demux.read_sample(video_id).unwrap() {
        held.push(sample.buffer.clone());
        let _ = dec.process(sample.buffer).unwrap();
    }
    while dec.flush().unwrap().is_some() {}
    drop(dec);

    for (i, buf) in held.iter().enumerate() {
        assert_eq!(
            buf.memory().refcount(),
            1,
            "input slot {i} still pinned after decoder drop"
        );
    }
}
