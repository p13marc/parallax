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
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    let as_ = pipeline.add_async_sink("audio_sink", audio_sink);
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
    let vs = pipeline.add_async_sink("video_sink", video_sink);
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

/// Runtime SNAP_BEFORE: the seek lands on the prior cue keyframe (1000 ms)
/// and, because the backward snap reports its landing, SeekDone carries the
/// honest position instead of the requested one (#166).
#[tokio::test(flavor = "multi_thread")]
async fn mkv_snap_before_seeks_at_runtime() {
    use parallax::clock::ClockTime;
    use parallax::event::{SeekEvent, SeekFlags};
    use parallax::pipeline::bus::MessageKind;

    let demux = open(H264_AAC_MKV).video_only();

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(2);
    let video_handle = video_sink.handle();
    let node = pipeline.add_demuxer("mkvdemux", demux);
    let vs = pipeline.add_async_sink("video_sink", video_sink);
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
    let seek = SeekEvent::new_time(ClockTime::from_millis(1200))
        .with_flags(SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::SNAP_BEFORE);
    assert!(handle.seek(seek).await);

    let mut pts = Vec::new();
    while let Pulled::Buffer(b) = video_handle.pull_buffer().await {
        pts.push(b.metadata().pts.nanos() / 1_000_000);
    }
    handle.wait().await.unwrap();

    let landing = pts
        .iter()
        .rposition(|p| *p == 1000)
        .expect("landing keyframe was presented");
    assert_eq!(
        &pts[landing..],
        &[1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        "playback continued from the prior cue keyframe to EOS: {pts:?}"
    );

    let mut seek_done_pos = None;
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone { position, .. } = msg.kind {
            seek_done_pos = Some(position);
        }
    }
    assert_eq!(
        seek_done_pos,
        Some(Some(1_000_000_000)),
        "SeekDone reports the cue landing, not the request"
    );
}

/// with_loop: the stream rewinds at EOS instead of ending, until stopped.
#[tokio::test(flavor = "multi_thread")]
async fn mkv_video_only_loops_at_eos() {
    let demux = open(VP9_OPUS_WEBM).video_only().with_loop(true);

    let mut pipeline = Pipeline::new();
    let sink = AppSink::with_max_buffers(4);
    let sink_handle = sink.handle();
    let node = pipeline.add_demuxer("mkvdemux", demux);
    let vs = pipeline.add_async_sink("video_sink", sink);
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

// ============================================================================
// Cue-indexed seek (#158) — driving the Demuxer trait directly, so the
// assertions are deterministic (no pipeline, no wall clock).
// ============================================================================

mod cue_seek {
    use super::*;
    use parallax::buffer::Buffer;
    use parallax::clock::ClockTime;
    use parallax::element::{Demuxer, DemuxerProduce};
    use parallax::event::{Event, EventResult, SeekEvent, SeekFlags};
    use parallax::metadata::BufferFlags;
    use std::io::{Read, Seek, SeekFrom};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Counts every byte read, to prove a seek is indexed, not a rescan.
    struct CountingReader {
        inner: Cursor<Vec<u8>>,
        read: Arc<AtomicU64>,
    }

    impl Read for CountingReader {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = self.inner.read(buf)?;
            self.read.fetch_add(n as u64, Ordering::Relaxed);
            Ok(n)
        }
    }

    impl Seek for CountingReader {
        fn seek(&mut self, from: SeekFrom) -> std::io::Result<u64> {
            self.inner.seek(from)
        }
    }

    fn counting(data: &[u8]) -> (MkvDemux<CountingReader>, Arc<AtomicU64>) {
        let read = Arc::new(AtomicU64::new(0));
        let reader = CountingReader {
            inner: Cursor::new(data.to_vec()),
            read: read.clone(),
        };
        (MkvDemux::new(reader).expect("fixture parses"), read)
    }

    /// `(pad, pts_ms, sync)` facts; the buffer itself is dropped so the
    /// demuxer's own arena never runs out of slots.
    fn facts(buffer: &Buffer) -> (u64, bool) {
        (
            buffer.metadata().pts.nanos() / 1_000_000,
            buffer.metadata().flags.contains(BufferFlags::SYNC_POINT),
        )
    }

    /// Produce until the next routed buffer; count empty yields on the way.
    fn next_buffer<R: Read + Seek + Send>(
        demux: &mut MkvDemux<R>,
        yields: &mut usize,
    ) -> Option<(u32, u64, bool)> {
        for _ in 0..10_000 {
            match demux.produce().expect("produce") {
                DemuxerProduce::Routed(routed) if routed.is_empty() => *yields += 1,
                DemuxerProduce::Routed(routed) => {
                    let (pad, buffer) = routed.into_iter().next().unwrap();
                    let (pts, sync) = facts(&buffer);
                    return Some((pad.0, pts, sync));
                }
                DemuxerProduce::Eos => return None,
                DemuxerProduce::WouldBlock => panic!("MkvDemux never blocks"),
            }
        }
        panic!("produce() made no progress");
    }

    fn seek_with(
        demux: &mut MkvDemux<impl Read + Seek + Send>,
        millis: u64,
        flags: SeekFlags,
    ) -> EventResult {
        let event =
            Event::Seek(SeekEvent::new_time(ClockTime::from_millis(millis)).with_flags(flags));
        demux.handle_upstream_event(&event)
    }

    fn seek_to(demux: &mut MkvDemux<impl Read + Seek + Send>, millis: u64) {
        let result = seek_with(demux, millis, SeekFlags::FLUSH | SeekFlags::KEY_UNIT);
        assert!(result.is_handled(), "seek handled");
    }

    /// (A) BlockGroup fixture: a cue-indexed seek lands on the keyframe at
    /// or after the target and reads only the target cluster — not the file.
    #[test]
    fn mkv_cue_seek_lands_without_rescan() {
        let (mut demux, read) = counting(H264_AAC_MKV);
        let mut yields = 0;

        // Play a couple of frames from the head first.
        for _ in 0..2 {
            next_buffer(&mut demux, &mut yields).expect("head frames");
        }

        let before = read.load(Ordering::Relaxed);
        seek_to(&mut demux, 1200);

        // First post-seek VIDEO buffer: the 1500 ms keyframe, sync-flagged;
        // nothing on any pad below the 1200 ms target.
        loop {
            let (pad, pts, sync) = next_buffer(&mut demux, &mut yields).expect("post-seek data");
            assert!(pts >= 1200, "pre-target frame leaked: pad {pad} pts {pts}");
            if pad == 0 {
                assert_eq!(pts, 1500, "landing keyframe");
                assert!(sync, "landing frame is a keyframe");
                break;
            }
        }

        // Cluster 2 spans ~7.6 KB; the old rewind fallback re-read the whole
        // file (≥ 15 KB) frame by frame.
        let delta = read.load(Ordering::Relaxed) - before;
        assert!(delta < 9_000, "seek re-read {delta} bytes — a rescan");
    }

    /// (B) SimpleBlock fixtures: same landing contract on both WebM codecs
    /// (keyframes verified at 0 and 500 ms).
    #[test]
    fn webm_cue_seek_simpleblock() {
        for (name, data) in [("vp9", VP9_OPUS_WEBM), ("vp8", VP8_VORBIS_WEBM)] {
            let (mut demux, _) = counting(data);
            let mut yields = 0;
            next_buffer(&mut demux, &mut yields).expect("head frame");

            seek_to(&mut demux, 300);
            loop {
                let (pad, pts, sync) = next_buffer(&mut demux, &mut yields)
                    .unwrap_or_else(|| panic!("{name}: stream ended before the seek landed"));
                assert!(pts >= 300, "{name}: pre-target frame leaked: {pts}");
                if pad == 0 {
                    assert_eq!(pts, 500, "{name}: landing keyframe");
                    assert!(sync, "{name}: sync-flagged");
                    break;
                }
            }
        }
    }

    /// (C) Per-track skip + resync composition: a seek past the last
    /// keyframe emits NO video at all (never a stale or mid-GOP frame),
    /// while audio at/after the target still flows.
    #[test]
    fn seek_no_pre_target_video_leak() {
        let (mut demux, _) = counting(H264_AAC_MKV);
        let mut yields = 0;
        next_buffer(&mut demux, &mut yields).expect("head frame");

        seek_to(&mut demux, 1600);
        let mut audio = 0u64;
        while let Some((pad, pts, _)) = next_buffer(&mut demux, &mut yields) {
            assert_ne!(
                pad, 0,
                "video emitted after a seek past the last keyframe (pts {pts})"
            );
            assert!(pts >= 1600, "pre-target audio leaked: {pts}");
            audio += 1;
        }
        assert!(audio > 0, "audio at/after the target still flows");
    }

    /// (D) The scan yields within its budget, so a superseding seek gets a
    /// produce() boundary to land on instead of queueing behind the scan.
    #[test]
    fn seek_scan_yields_within_budget() {
        let read = Arc::new(AtomicU64::new(0));
        let reader = CountingReader {
            inner: Cursor::new(H264_AAC_MKV.to_vec()),
            read,
        };
        let mut demux = MkvDemux::new(reader)
            .expect("fixture parses")
            .with_scan_budget(4);
        let mut yields = 0;
        next_buffer(&mut demux, &mut yields).expect("head frame");

        seek_to(&mut demux, 1600);
        // Draining to the first post-seek buffer discards more than 4 frames
        // (skips + resync), so at least one empty yield must interleave.
        while next_buffer(&mut demux, &mut yields).is_some() {}
        assert!(
            yields > 0,
            "a bounded scan must yield between produce() calls"
        );
    }

    /// (E) SNAP_BEFORE lands on the prior cue and — unlike the forward
    /// default — reports the landing up front, so the executor's Segment
    /// and SeekDone carry the honest position (#166). Cues in the fixture
    /// sit at 0/500/1000/1500 ms.
    #[test]
    fn cue_seek_snap_before_lands_on_prior_cue() {
        let (mut demux, _) = counting(H264_AAC_MKV);
        let mut yields = 0;
        next_buffer(&mut demux, &mut yields).expect("head frame");

        let result = seek_with(
            &mut demux,
            1200,
            SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::SNAP_BEFORE,
        );
        assert!(
            matches!(
                result,
                EventResult::Handled {
                    position: Some(1_000_000_000)
                }
            ),
            "backward snap knows its landing up front, got {result:?}"
        );

        // First post-seek video buffer is the 1000 ms cue keyframe; nothing
        // below the cue leaks on any pad.
        loop {
            let (pad, pts, sync) = next_buffer(&mut demux, &mut yields).expect("post-seek data");
            assert!(pts >= 1000, "pre-cue frame leaked: pad {pad} pts {pts}");
            if pad == 0 {
                assert_eq!(pts, 1000, "landing keyframe");
                assert!(sync, "landing frame is a keyframe");
                break;
            }
        }
    }

    /// (F) Both SNAP bits = nearest, resolved at cue granularity: 1200 ms is
    /// closer to the 1000 ms cue, 1300 ms to the 1500 ms one.
    #[test]
    fn cue_seek_snap_nearest() {
        let nearest =
            SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::SNAP_BEFORE | SeekFlags::SNAP_AFTER;

        let (mut demux, _) = counting(H264_AAC_MKV);
        let mut yields = 0;
        next_buffer(&mut demux, &mut yields).expect("head frame");
        assert!(
            matches!(
                seek_with(&mut demux, 1200, nearest),
                EventResult::Handled {
                    position: Some(1_000_000_000)
                }
            ),
            "1200 ms resolves backward (200 ms) over forward (300 ms)"
        );

        let (mut demux, _) = counting(H264_AAC_MKV);
        let mut yields = 0;
        next_buffer(&mut demux, &mut yields).expect("head frame");
        assert!(
            matches!(
                seek_with(&mut demux, 1300, nearest),
                EventResult::Handled { position: None }
            ),
            "1300 ms resolves forward, whose landing is scan-determined"
        );
        loop {
            let (pad, pts, sync) = next_buffer(&mut demux, &mut yields).expect("post-seek data");
            assert!(pts >= 1300, "pre-target frame leaked: pad {pad} pts {pts}");
            if pad == 0 {
                assert_eq!(pts, 1500, "forward landing keyframe");
                assert!(sync);
                break;
            }
        }
    }
}

/// A demuxer-rooted pipeline reports seekable + duration through the
/// running handle — it used to report `seekable() == false` while seeking
/// perfectly well, because `DemuxerAdapter` never forwarded the queries
/// (#162).
#[tokio::test(flavor = "multi_thread")]
async fn demuxer_rooted_pipeline_reports_seekable() {
    let demux = open(H264_AAC_MKV).video_only();

    let mut pipeline = Pipeline::new();
    let sink = AppSink::with_max_buffers(4);
    let sink_handle = sink.handle();
    let node = pipeline.add_demuxer("mkvdemux", demux);
    let vs = pipeline.add_async_sink("video_sink", sink);
    pipeline.link_pads(node, "video", vs, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    assert!(handle.seekable(), "MkvDemux declares seek support");
    let q = handle.query_seekable();
    assert!(q.seekable);
    assert!(
        (1_800_000_000..=2_300_000_000).contains(&q.stop),
        "range bounded by the ~2 s duration, got {}",
        q.stop
    );
    let duration = handle.duration().to_option().expect("Time duration known");
    assert!((1_800_000_000..=2_300_000_000).contains(&duration.nanos()));

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
