//! Subtitles, end to end (#279).
//!
//! The fixture is written here rather than checked in, because there is no
//! ffmpeg on this machine and every MKV in `tests/fixtures/` was generated
//! with one. Hand-building the EBML is the honest alternative and a better
//! one: it pins the exact bytes a cue arrives as, which a downloaded file
//! never would — and the timing this asserts (`BlockDuration` becoming the
//! cue's out-time) is invisible in a file you cannot see inside.

#![cfg(feature = "mkv-demux")]

use std::io::Cursor;

use parallax::clock::ClockTime;
use parallax::element::{Demuxer, DemuxerProduce};
use parallax::elements::demux::{MkvDemux, MkvTrackType};
use parallax::format::{MediaFormat, TextCodec};

// ---------------------------------------------------------------------------
// A minimal Matroska writer
// ---------------------------------------------------------------------------

/// EBML variable-length integer, in the smallest width that fits.
fn vint(mut value: u64) -> Vec<u8> {
    // The marker bit costs one bit of every byte, so a width holds
    // `7 * width` bits with the top value reserved for "unknown".
    let mut width = 1;
    while width < 8 && value >= (1u64 << (7 * width)) - 1 {
        width += 1;
    }
    let mut out = vec![0u8; width];
    for i in (0..width).rev() {
        out[i] = (value & 0xff) as u8;
        value >>= 8;
    }
    out[0] |= 0x80 >> (width - 1);
    out
}

/// An element: id bytes, then a length vint, then the payload.
fn el(id: &[u8], body: &[u8]) -> Vec<u8> {
    let mut out = id.to_vec();
    out.extend_from_slice(&vint(body.len() as u64));
    out.extend_from_slice(body);
    out
}

/// An unsigned-integer element, big-endian and minimally sized.
fn uint(id: &[u8], value: u64) -> Vec<u8> {
    let bytes = value.to_be_bytes();
    let first = bytes.iter().position(|b| *b != 0).unwrap_or(7);
    el(id, &bytes[first..])
}

/// A float element, always 64-bit — Matroska allows 4 or 8.
fn float(id: &[u8], value: f64) -> Vec<u8> {
    el(id, &value.to_be_bytes())
}

fn string(id: &[u8], value: &str) -> Vec<u8> {
    el(id, value.as_bytes())
}

fn concat(parts: &[Vec<u8>]) -> Vec<u8> {
    parts.iter().flatten().copied().collect()
}

/// A file with one video track and one `S_TEXT/UTF8` subtitle track.
///
/// The video track exists because a subtitle-only file is not what a player
/// opens, and because routing has to be shown to keep the two apart.
fn fixture(cues: &[(u64, u64, &str)]) -> Vec<u8> {
    // EBML header. DocType matroska, version 2 — enough for SimpleBlock.
    let header = el(
        &[0x1A, 0x45, 0xDF, 0xA3],
        &concat(&[
            uint(&[0x42, 0x86], 1), // EBMLVersion
            uint(&[0x42, 0xF7], 1), // EBMLReadVersion
            uint(&[0x42, 0xF2], 4), // EBMLMaxIDLength
            uint(&[0x42, 0xF3], 8), // EBMLMaxSizeLength
            string(&[0x42, 0x82], "matroska"),
            uint(&[0x42, 0x87], 2), // DocTypeVersion
            uint(&[0x42, 0x85], 2), // DocTypeReadVersion
        ]),
    );

    // Info: one tick is one millisecond, and the file is ten seconds.
    let info = el(
        &[0x15, 0x49, 0xA9, 0x66],
        &concat(&[
            uint(&[0x2A, 0xD7, 0xB1], 1_000_000), // TimestampScale (ns)
            float(&[0x44, 0x89], 10_000.0),       // Duration, in ticks
            // matroska-demuxer requires both, and says so by name when they
            // are missing.
            string(&[0x4D, 0x80], "parallax-tests"), // MuxingApp
            string(&[0x57, 0x41], "parallax-tests"), // WritingApp
        ]),
    );

    let video_track = el(
        &[0xAE],
        &concat(&[
            uint(&[0xD7], 1),                  // TrackNumber
            uint(&[0x73, 0xC5], 1),            // TrackUID
            uint(&[0x83], 1),                  // TrackType: video
            string(&[0x86], "V_UNCOMPRESSED"), // CodecID
            el(
                &[0xE0],
                &concat(&[uint(&[0xB0], 16), uint(&[0xBA], 16)]), // PixelWidth/Height
            ),
        ]),
    );

    let subtitle_track = el(
        &[0xAE],
        &concat(&[
            uint(&[0xD7], 2),
            uint(&[0x73, 0xC5], 2),
            uint(&[0x83], 0x11), // TrackType: subtitle
            string(&[0x86], "S_TEXT/UTF8"),
            string(&[0x22, 0xB5, 0x9C], "eng"), // Language
            string(&[0x53, 0x6E], "English"),   // Name
            uint(&[0x88], 1),                   // FlagDefault
            uint(&[0x55, 0xAA], 1),             // FlagForced
        ]),
    );

    let tracks = el(
        &[0x16, 0x54, 0xAE, 0x6B],
        &concat(&[video_track, subtitle_track]),
    );

    // One cluster at timestamp zero. Each cue is a BlockGroup, because a
    // SimpleBlock has nowhere to put a duration — and the duration is the
    // whole point.
    let mut cluster_body = uint(&[0xE7], 0); // Timestamp
    // A video frame, so the routing has something to keep apart.
    let mut video_block = vint(1);
    video_block.extend_from_slice(&0i16.to_be_bytes());
    video_block.push(0x80); // keyframe
    video_block.extend_from_slice(&[0u8; 8]);
    cluster_body.extend_from_slice(&el(&[0xA3], &video_block));

    for (start_ms, duration_ms, text) in cues {
        let mut block = vint(2); // track number
        block.extend_from_slice(&(*start_ms as i16).to_be_bytes());
        block.push(0x00);
        block.extend_from_slice(text.as_bytes());
        cluster_body.extend_from_slice(&el(
            &[0xA0], // BlockGroup
            &concat(&[el(&[0xA1], &block), uint(&[0x9B], *duration_ms)]),
        ));
    }
    let cluster = el(&[0x1F, 0x43, 0xB6, 0x75], &cluster_body);

    let segment = el(&[0x18, 0x53, 0x80, 0x67], &concat(&[info, tracks, cluster]));
    concat(&[header, segment])
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn open(data: Vec<u8>) -> MkvDemux<Cursor<Vec<u8>>> {
    MkvDemux::new(Cursor::new(data)).expect("the hand-built fixture parses")
}

#[test]
fn the_fixture_is_a_matroska_file_with_a_subtitle_track() {
    let demux = open(fixture(&[(1_000, 2_000, "Hello there")]));
    let subtitle = demux
        .tracks()
        .iter()
        .find(|t| t.track_type == MkvTrackType::Subtitle)
        .expect("a subtitle track");

    assert_eq!(subtitle.codec_id, "S_TEXT/UTF8");
    // The flags a picker needs, which #278 taught the track table to carry.
    assert_eq!(subtitle.label.spoken_language(), Some("eng"));
    assert_eq!(subtitle.label.name.as_deref(), Some("English"));
    assert!(subtitle.label.default);
    assert!(subtitle.label.forced);
    assert_eq!(subtitle.label.describe().as_deref(), Some("English [eng]"));
}

#[test]
fn subtitles_are_off_until_asked_for() {
    let demux = open(fixture(&[(1_000, 2_000, "Hello there")]));
    // Two pads at most, and neither of them the subtitle track: a player that
    // routed subtitles by default would have to decode and draw them for
    // every file, including the ones nobody wants them on.
    assert!(
        demux.outputs().iter().all(|(pad, _)| pad.0 != 2),
        "no subtitle pad without asking"
    );

    let demux = open(fixture(&[(1_000, 2_000, "Hello there")]))
        .with_subtitle_track(Some(2))
        .expect("track 2 is a text subtitle track");
    assert!(demux.outputs().iter().any(|(pad, _)| pad.0 == 2));
    assert_eq!(demux.pad_name(parallax::element::PadId(2)), "subtitle");
}

#[test]
fn a_cue_arrives_as_text_with_the_containers_timing() {
    let mut demux = open(fixture(&[
        (1_000, 2_000, "Hello there"),
        (4_000, 1_500, "Two lines\nof dialogue"),
    ]))
    .with_subtitle_track(Some(2))
    .expect("it routes");

    let mut cues = Vec::new();
    for _ in 0..32 {
        match demux.produce().expect("it produces") {
            DemuxerProduce::Routed(routed) => {
                for (pad, buffer) in routed.0 {
                    if pad.0 != 2 {
                        continue;
                    }
                    let meta = buffer.metadata().clone();
                    let text = String::from_utf8(buffer.as_bytes().to_vec()).expect("utf-8");
                    cues.push((meta, text));
                }
            }
            DemuxerProduce::Eos => break,
            // Arena pressure; the executor retries, and so does this.
            DemuxerProduce::WouldBlock => continue,
        }
    }

    assert_eq!(cues.len(), 2, "both cues came out of the subtitle pad");

    let (meta, text) = &cues[0];
    assert_eq!(text, "Hello there");
    // The container's timing, not a guess: Matroska strips the in-file
    // timestamps into the block, so the buffer *is* the cue.
    assert_eq!(meta.pts, ClockTime::from_millis(1_000));
    assert_eq!(meta.duration, ClockTime::from_millis(2_000));
    // And it says what it is, rather than arriving as anonymous bytes.
    assert_eq!(meta.format, Some(MediaFormat::Text(TextCodec::Utf8)));

    let (meta, text) = &cues[1];
    assert_eq!(text, "Two lines\nof dialogue");
    assert_eq!(meta.pts, ClockTime::from_millis(4_000));
    assert_eq!(meta.duration, ClockTime::from_millis(1_500));
}

#[test]
fn a_bitmap_subtitle_track_is_refused_rather_than_shown_as_garbage() {
    // The demuxer has no way to draw PGS or VobSub, and routing one would
    // hand a host a blob it would try to read as UTF-8.
    assert_eq!(
        MkvDemux::<Cursor<Vec<u8>>>::text_codec("S_TEXT/UTF8"),
        Some(TextCodec::Utf8)
    );
    assert_eq!(
        MkvDemux::<Cursor<Vec<u8>>>::text_codec("S_TEXT/ASS"),
        Some(TextCodec::Ass)
    );
    assert_eq!(
        MkvDemux::<Cursor<Vec<u8>>>::text_codec("S_TEXT/SSA"),
        Some(TextCodec::Ass)
    );
    assert_eq!(MkvDemux::<Cursor<Vec<u8>>>::text_codec("S_HDMV/PGS"), None);
    assert_eq!(MkvDemux::<Cursor<Vec<u8>>>::text_codec("S_VOBSUB"), None);
}
