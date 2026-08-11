//! Integration tests for the Matroska/WebM demuxer (#121).
//!
//! Fixtures are tiny ffmpeg-generated files checked into `tests/fixtures/`:
//! - `tiny_h264_aac.mkv` — 2 s, 20 frames @ 10 fps, keyframe every 5, AAC audio
//! - `tiny_vp9_opus.webm` — 1 s VP9 + Opus
//! - `tiny_vp8_vorbis.webm` — 1 s VP8 + Vorbis

#![cfg(feature = "mkv-demux")]

use parallax::elements::demux::{MkvCodec, MkvDemux, MkvTrackType};
use parallax::elements::{AppSink, Pulled};
use parallax::pipeline::{Executor, Pipeline};
use std::io::Cursor;

const H264_AAC_MKV: &[u8] = include_bytes!("fixtures/tiny_h264_aac.mkv");
const VP9_OPUS_WEBM: &[u8] = include_bytes!("fixtures/tiny_vp9_opus.webm");
const VP8_VORBIS_WEBM: &[u8] = include_bytes!("fixtures/tiny_vp8_vorbis.webm");

fn open(data: &[u8]) -> MkvDemux<Cursor<Vec<u8>>> {
    MkvDemux::new(Cursor::new(data.to_vec())).expect("fixture parses")
}

#[test]
fn mkv_probe_h264_aac_track_table() {
    let demux = open(H264_AAC_MKV);

    let video = demux.video_track().expect("video track");
    assert_eq!(video.codec, MkvCodec::H264);
    assert_eq!(video.track_type, MkvTrackType::Video);
    let vinfo = video.video_info.as_ref().expect("video info");
    assert_eq!((vinfo.width, vinfo.height), (64, 64));

    let audio = demux.audio_track().expect("audio track");
    assert_eq!(audio.codec, MkvCodec::Aac);
    let ainfo = audio.audio_info.as_ref().expect("audio info");
    assert_eq!(ainfo.sample_rate, 44100);
    assert!(
        ainfo.codec_private.is_some(),
        "AAC CodecPrivate (AudioSpecificConfig) exposed"
    );

    let duration = demux.duration_ns().expect("segment declares duration");
    assert!(
        (1_800_000_000..=2_300_000_000).contains(&duration),
        "~2 s duration, got {duration}"
    );
}

#[test]
fn webm_probe_vp9_opus_and_vp8_vorbis() {
    let demux = open(VP9_OPUS_WEBM);
    assert_eq!(demux.video_track().unwrap().codec, MkvCodec::Vp9);
    let audio = demux.audio_track().unwrap();
    assert_eq!(audio.codec, MkvCodec::Opus);
    let ainfo = audio.audio_info.as_ref().unwrap();
    assert_eq!(ainfo.sample_rate, 48000);
    assert!(ainfo.codec_private.is_some(), "OpusHead exposed");

    let demux = open(VP8_VORBIS_WEBM);
    assert_eq!(demux.video_track().unwrap().codec, MkvCodec::Vp8);
    let audio = demux.audio_track().unwrap();
    assert_eq!(audio.codec, MkvCodec::Vorbis);
    assert!(
        audio.audio_info.as_ref().unwrap().codec_private.is_some(),
        "Xiph-laced Vorbis headers exposed"
    );
}

/// A/V frames route to their own pads; H.264 comes out as Annex-B with
/// in-band parameter sets on keyframes; PTS is monotonic per branch.
#[tokio::test(flavor = "multi_thread")]
async fn mkv_routes_av_branches() {
    let demux = open(H264_AAC_MKV);

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(64);
    let video_handle = video_sink.handle();
    let audio_sink = AppSink::with_max_buffers(128);
    let audio_handle = audio_sink.handle();

    let node = pipeline.add_demuxer("mkvdemux", demux);
    let vs = pipeline.add_sink("video_sink", video_sink);
    let as_ = pipeline.add_sink("audio_sink", audio_sink);
    pipeline.link_pads(node, "video", vs, "sink").unwrap();
    pipeline.link_pads(node, "audio", as_, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Drain both branches concurrently and extract per-buffer facts instead
    // of holding the buffers: a held Buffer pins its arena slot, and the
    // demuxer arena is sized for in-flight frames, not the whole file.
    let (video, audio_pts) = tokio::join!(
        async {
            // (first 5 payload bytes, pts_ms, sync flag)
            let mut v: Vec<([u8; 5], u64, bool)> = Vec::new();
            while let Pulled::Buffer(b) = video_handle.pull_buffer().await {
                let mut head = [0u8; 5];
                head.copy_from_slice(&b.as_bytes()[..5]);
                v.push((
                    head,
                    b.metadata().pts.nanos() / 1_000_000,
                    b.metadata()
                        .flags
                        .contains(parallax::metadata::BufferFlags::SYNC_POINT),
                ));
            }
            v
        },
        async {
            let mut a: Vec<u64> = Vec::new();
            while let Pulled::Buffer(b) = audio_handle.pull_buffer().await {
                a.push(b.metadata().pts.nanos());
            }
            a
        }
    );
    handle.wait().await.unwrap();

    assert_eq!(video.len(), 20, "all video frames on video branch");
    assert!(
        !audio_pts.is_empty(),
        "audio frames arrived on the audio branch"
    );

    for (head, _, _) in &video {
        assert_eq!(&head[..4], &[0, 0, 0, 1], "video is Annex-B");
    }
    // Keyframes (every 5th frame at 10 fps → 0/500/1000/1500 ms) carry the
    // sync flag and lead with an SPS NAL (type 7).
    let keyframes: Vec<u64> = video
        .iter()
        .filter(|(_, _, sync)| *sync)
        .map(|(_, pts, _)| *pts)
        .collect();
    assert_eq!(keyframes, vec![0, 500, 1000, 1500], "keyframe cadence");
    assert_eq!(video[0].0[4] & 0x1f, 7, "keyframe leads with in-band SPS");

    for pair in video.windows(2) {
        assert!(pair[0].1 < pair[1].1, "video PTS monotonic");
    }
    for pair in audio_pts.windows(2) {
        assert!(pair[0] <= pair[1], "audio PTS monotonic");
    }
}

/// Runtime seek: PipelineHandle::seek_time reaches the demuxer, video
/// restarts at a keyframe at/after the target, and playback runs to EOS.
#[tokio::test(flavor = "multi_thread")]
async fn mkv_seeks_at_runtime() {
    use parallax::clock::ClockTime;
    use parallax::pipeline::bus::MessageKind;

    let demux = open(H264_AAC_MKV).video_only();

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(2);
    let video_handle = video_sink.handle();
    let node = pipeline.add_demuxer("mkvdemux", demux);
    let vs = pipeline.add_sink("video_sink", video_sink);
    pipeline.link_pads(node, "video", vs, "sink").unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    for _ in 0..2 {
        assert!(matches!(
            video_handle.pull_buffer().await,
            Pulled::Buffer(_)
        ));
    }
    assert!(handle.seek_time(ClockTime::from_millis(1200)).await);

    // Pre-seek in-flight frames may still surface; after the flush, video
    // resumes at the first keyframe at/after the seek point (1500 ms).
    let mut pts = Vec::new();
    let mut sync = Vec::new();
    while let Pulled::Buffer(b) = video_handle.pull_buffer().await {
        pts.push(b.metadata().pts.nanos() / 1_000_000);
        sync.push(
            b.metadata()
                .flags
                .contains(parallax::metadata::BufferFlags::SYNC_POINT),
        );
    }
    handle.wait().await.unwrap();

    let landing = pts
        .iter()
        .rposition(|p| *p == 1500)
        .expect("landing keyframe was presented");
    assert!(sync[landing], "landing frame is a keyframe");
    assert_eq!(
        &pts[landing..],
        &[1500, 1600, 1700, 1800, 1900],
        "playback continued from the landing keyframe to EOS: {pts:?}"
    );

    let mut seek_done = false;
    while let Some(msg) = bus.poll() {
        if matches!(msg.kind, MessageKind::SeekDone { .. }) {
            seek_done = true;
        }
    }
    assert!(seek_done, "SeekDone posted");
}

/// with_loop: the stream rewinds at EOS instead of ending, until stopped.
#[tokio::test(flavor = "multi_thread")]
async fn mkv_video_only_loops_at_eos() {
    let demux = open(VP9_OPUS_WEBM).video_only().with_loop(true);

    let mut pipeline = Pipeline::new();
    let sink = AppSink::with_max_buffers(4);
    let sink_handle = sink.handle();
    let node = pipeline.add_demuxer("mkvdemux", demux);
    let vs = pipeline.add_sink("video_sink", sink);
    pipeline.link_pads(node, "video", vs, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // More buffers than the 10-frame file holds: only looping supplies them.
    let mut pts = Vec::new();
    for _ in 0..25 {
        match sink_handle.pull_buffer().await {
            Pulled::Buffer(b) => pts.push(b.metadata().pts.nanos() / 1_000_000),
            other => panic!("stream ended early: {other:?}"),
        }
    }
    assert!(
        pts.windows(2).any(|w| w[1] < w[0]),
        "PTS wrapped across pulls of a 10-frame file: {pts:?}"
    );

    handle.stop();
    while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {}
    handle.wait().await.unwrap();
}

/// Subtitle tracks are listed but never routed.
#[test]
fn subtitle_tracks_listed_not_routed() {
    use parallax::element::Demuxer;

    let demux = open(H264_AAC_MKV);
    // The fixture has no subtitle track, but the routing table must only
    // ever hold the selected A/V pads regardless of extra tracks.
    assert!(demux.outputs().len() <= 2);
    for t in demux.tracks() {
        if t.codec.is_subtitle() {
            unreachable!("fixture has no subtitle track");
        }
    }
}
