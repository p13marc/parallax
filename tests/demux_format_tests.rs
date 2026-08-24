//! #238: a demuxer must declare what it demuxed.
//!
//! Every demuxer knows its tracks' codecs — that is the entire job — and two
//! of the three also parse the geometry out of the track header. All three
//! then threw it away: `Metadata.format` stayed `None` on every buffer they
//! emitted, so `mp4demux ! zenohsink` published H.264 as `zenoh/bytes` with
//! `FrameMeta { width: 0, height: 0 }`, and anything gating on the format saw
//! an undeclared stream.
//!
//! What each demuxer can honestly say differs, and these tests pin the
//! difference rather than papering over it:
//!
//! | | codec | geometry |
//! |---|---|---|
//! | MKV | yes (`MkvTrack::codec`) | yes (`MkvVideoInfo`) |
//! | MP4 | yes (`Mp4Track::codec`) | yes (`Mp4VideoInfo`) |
//! | MPEG-TS | yes (`FrameCollector::stream_type`) | **no** — it parses no SPS |

// No demuxer feature, nothing to test — and the shared helpers below would
// otherwise be dead code under `-D warnings`.
#![cfg(any(feature = "mkv-demux", feature = "mp4-demux", feature = "mpeg-ts"))]

use parallax::elements::{AppSink, Pulled};
use parallax::format::{MediaFormat, VideoCodec};
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};

/// The codec a buffer declares, whichever way it was declared.
fn declared_codec(meta: &Metadata) -> Option<VideoCodec> {
    match meta.format {
        Some(MediaFormat::Video(codec)) => Some(codec),
        _ => None,
    }
}

#[cfg(feature = "mkv-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mkv_stamps_codec_and_geometry_on_both_branches() {
    use parallax::elements::demux::MkvDemux;
    use parallax::format::AudioCodec;
    use std::io::Cursor;

    const H264_AAC_MKV: &[u8] = include_bytes!("fixtures/tiny_h264_aac.mkv");
    let demux = MkvDemux::new(Cursor::new(H264_AAC_MKV.to_vec())).unwrap();

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(64);
    let video_handle = video_sink.handle();
    let audio_sink = AppSink::with_max_buffers(128);
    let audio_handle = audio_sink.handle();

    let node = pipeline.add_demuxer("mkvdemux", demux);
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    let as_ = pipeline.add_async_sink("audio_sink", audio_sink);
    pipeline.link_pads(node, "video", vs, "sink").unwrap();
    pipeline.link_pads(node, "audio", as_, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let (video, audio) = tokio::join!(
        async {
            let mut v = Vec::new();
            while let Pulled::Buffer(b) = video_handle.pull_buffer().await {
                v.push((b.metadata().format.clone(), b.metadata().coded_video_dims()));
            }
            v
        },
        async {
            let mut a = Vec::new();
            while let Pulled::Buffer(b) = audio_handle.pull_buffer().await {
                a.push(b.metadata().format.clone());
            }
            a
        }
    );
    handle.wait().await.unwrap();

    assert!(!video.is_empty() && !audio.is_empty());
    for (format, dims) in &video {
        assert_eq!(*format, Some(MediaFormat::Video(VideoCodec::H264)));
        // The fixture is 64x64, and the demuxer read that out of the track
        // header — a consumer must not have to parse the bitstream for it.
        assert_eq!(*dims, Some((64, 64)));
    }
    for format in &audio {
        assert_eq!(*format, Some(MediaFormat::Audio(AudioCodec::Aac)));
    }
}

#[cfg(feature = "mkv-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn webm_declares_vp9_not_h264() {
    use parallax::elements::demux::MkvDemux;
    use std::io::Cursor;

    const VP9_OPUS_WEBM: &[u8] = include_bytes!("fixtures/tiny_vp9_opus.webm");
    let demux = MkvDemux::new(Cursor::new(VP9_OPUS_WEBM.to_vec())).unwrap();

    let mut pipeline = Pipeline::new();
    let sink = AppSink::with_max_buffers(64);
    let sink_handle = sink.handle();
    let node = pipeline.add_demuxer("mkvdemux", demux);
    let s = pipeline.add_async_sink("sink", sink);
    pipeline.link_pads(node, "video", s, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    let mut codecs = Vec::new();
    while let Pulled::Buffer(b) = sink_handle.pull_buffer().await {
        codecs.push(declared_codec(b.metadata()));
    }
    handle.wait().await.unwrap();

    assert!(!codecs.is_empty());
    assert!(codecs.iter().all(|c| *c == Some(VideoCodec::Vp9)));
}

#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_stamps_codec_and_geometry_on_both_branches() {
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4AudioTrackConfig, Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use parallax::format::AudioCodec;
    use std::io::Cursor;

    let mut mux = Mp4Mux::new(Cursor::new(Vec::new()), Mp4MuxConfig::default()).unwrap();
    let sps = vec![0x67, 0x42, 0x00, 0x1f];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];
    let video = mux
        .add_video_track(Mp4VideoTrackConfig::h264(320, 240, &sps, &pps))
        .unwrap();
    let audio = mux
        .add_audio_track(Mp4AudioTrackConfig::aac(48000, 2))
        .unwrap();
    let keyframe = [0x00, 0x00, 0x00, 0x02, 0x65, 0xAA];
    let delta = [0x00, 0x00, 0x00, 0x02, 0x41, 0x9A];
    for i in 0..6u64 {
        let data: &[u8] = if i == 0 { &keyframe } else { &delta };
        mux.write_video_sample(video, data, i * 100, 100, i == 0)
            .unwrap();
        mux.write_audio_sample(audio, &[0xAAu8, 0xBB], i * 100, 100)
            .unwrap();
    }
    let mp4_data = mux.finish().unwrap().into_inner();
    let demux = Mp4Demux::new(Cursor::new(mp4_data.clone()), mp4_data.len() as u64).unwrap();

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(32);
    let video_handle = video_sink.handle();
    let audio_sink = AppSink::with_max_buffers(32);
    let audio_handle = audio_sink.handle();

    let node = pipeline.add_demuxer("mp4demux", Mp4DemuxSource::new(demux));
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    let as_ = pipeline.add_async_sink("audio_sink", audio_sink);
    pipeline.link_pads(node, "video", vs, "sink").unwrap();
    pipeline.link_pads(node, "audio", as_, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let mut video = Vec::new();
    while let Pulled::Buffer(b) = video_handle.pull_buffer().await {
        video.push((b.metadata().format.clone(), b.metadata().coded_video_dims()));
    }
    let mut audio = Vec::new();
    while let Pulled::Buffer(b) = audio_handle.pull_buffer().await {
        audio.push(b.metadata().format.clone());
    }
    handle.wait().await.unwrap();

    assert_eq!(video.len(), 6);
    for (format, dims) in &video {
        assert_eq!(*format, Some(MediaFormat::Video(VideoCodec::H264)));
        assert_eq!(
            *dims,
            Some((320, 240)),
            "geometry comes from the track header"
        );
    }
    assert!(!audio.is_empty());
    for format in &audio {
        assert_eq!(*format, Some(MediaFormat::Audio(AudioCodec::Aac)));
    }
}

/// MPEG-TS declares the codec and nothing else — and now sets `SYNC_POINT`,
/// which it never did before.
///
/// There is no geometry to stamp: this demuxer parses no sequence header, so
/// a width would have to be invented. The keyframe verdict, by contrast, is
/// readable straight off the elementary stream, which is Annex-B here.
#[cfg(feature = "mpeg-ts")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn ts_declares_the_codec_and_finds_its_entry_points() {
    use parallax::clock::ClockTime;
    use parallax::elements::{
        FileSrc, TsDemuxElement, TsMux, TsMuxConfig, TsMuxStreamType, TsMuxTrack,
    };
    use parallax::metadata::BufferFlags;

    const VIDEO_PID: u16 = 0x100;
    const FRAMES: u64 = 40;

    let config = TsMuxConfig::new()
        .add_track(TsMuxTrack::new(VIDEO_PID, TsMuxStreamType::H264).video())
        .psi_interval(50);
    let mut mux = TsMux::new(config);

    let mut out = Vec::new();
    for frame in 0..FRAMES {
        // Every 10th access unit is an SPS+IDR pair; the rest are non-IDR
        // slices. Both are Annex-B, which is what a TS elementary stream is.
        let mut au = Vec::new();
        if frame.is_multiple_of(10) {
            au.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x42, 0x00, 0x1f]);
            au.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xce, 0x3c, 0x80]);
            au.extend_from_slice(&[0, 0, 0, 1, 0x65]);
        } else {
            au.extend_from_slice(&[0, 0, 0, 1, 0x41]);
        }
        au.resize(2048, (frame % 251) as u8);
        let pts = ClockTime::from_nanos(frame * 1_000_000_000 / 25);
        out.extend(mux.write_pes(VIDEO_PID, &au, Some(pts), Some(pts)).unwrap());
    }

    let mut file = tempfile::NamedTempFile::new().unwrap();
    std::io::Write::write_all(file.as_file_mut(), &out).unwrap();
    std::io::Write::flush(file.as_file_mut()).unwrap();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::new(file.path()).with_chunk_size(100 * 188));
    let demux = pipeline.add_demuxer("tsdemux", TsDemuxElement::new());
    let sink = AppSink::with_max_buffers(128);
    let sink_handle = sink.handle();
    let s = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "video", s, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let mut frames = Vec::new();
    while let Pulled::Buffer(b) = sink_handle.pull_buffer().await {
        frames.push((
            declared_codec(b.metadata()),
            b.metadata().flags.contains(BufferFlags::SYNC_POINT),
            b.metadata().coded_video_dims(),
        ));
    }
    handle.wait().await.unwrap();

    assert!(frames.len() > 20, "got {} frames", frames.len());
    for (codec, _, dims) in &frames {
        assert_eq!(*codec, Some(VideoCodec::H264));
        assert_eq!(
            *dims, None,
            "MPEG-TS parses no SPS — it must not invent geometry"
        );
    }
    let syncs = frames.iter().filter(|(_, sync, _)| *sync).count();
    assert!(
        syncs >= 3,
        "the SPS+IDR access units are entry points, got {syncs}"
    );
    assert!(
        syncs < frames.len(),
        "non-IDR slices must not be flagged as entry points"
    );
}
