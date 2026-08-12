//! Integration tests for per-pad demuxer routing (#76).
//!
//! The executor used to broadcast every demuxer output to *all* pads — a
//! two-branch A/V pipeline received audio on the video branch and vice
//! versa. These tests pin the fixed behavior: a routed buffer reaches only
//! the links attached under its pad's name.

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{Demuxer, DemuxerProduce, PadAddedCallback, PadId, RoutedOutput};
use parallax::elements::{AppSink, AppSrc, Pulled};
use parallax::error::Result;
use parallax::format::Caps;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};
use std::sync::OnceLock;

fn arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 512).unwrap())
}

fn buffer_with_seq(seq: u64) -> Buffer {
    let slot = arena().acquire().expect("test arena exhausted");
    Buffer::new(
        MemoryHandle::with_len(slot, 8),
        Metadata::from_sequence(seq),
    )
}

/// Routes buffers by sequence parity to pads "even" and "odd".
struct ParityDemuxer {
    outputs: Vec<(PadId, Caps)>,
}

impl ParityDemuxer {
    fn new() -> Self {
        Self {
            outputs: vec![(PadId(0), Caps::any()), (PadId(1), Caps::any())],
        }
    }
}

impl Demuxer for ParityDemuxer {
    fn demux(&mut self, buffer: Buffer) -> Result<RoutedOutput> {
        let pad = PadId((buffer.metadata().sequence % 2) as u32);
        Ok(RoutedOutput::single(pad, buffer))
    }

    fn pad_name(&self, pad: PadId) -> String {
        match pad.0 {
            0 => "even".into(),
            _ => "odd".into(),
        }
    }

    fn outputs(&self) -> &[(PadId, Caps)] {
        &self.outputs
    }

    fn on_pad_added(&mut self, _callback: PadAddedCallback) {}

    fn name(&self) -> &str {
        "paritydemux"
    }
}

/// Source-style demuxer: owns its "reader" (a countdown), no input link.
struct CountdownDemuxer {
    remaining: u64,
    outputs: Vec<(PadId, Caps)>,
}

impl Demuxer for CountdownDemuxer {
    fn demux(&mut self, _buffer: Buffer) -> Result<RoutedOutput> {
        unreachable!("driven through produce()")
    }

    fn produce(&mut self) -> Result<DemuxerProduce> {
        if self.remaining == 0 {
            return Ok(DemuxerProduce::Eos);
        }
        self.remaining -= 1;
        let seq = self.remaining;
        Ok(DemuxerProduce::Routed(RoutedOutput::single(
            PadId((seq % 2) as u32),
            buffer_with_seq(seq),
        )))
    }

    fn pad_name(&self, pad: PadId) -> String {
        match pad.0 {
            0 => "even".into(),
            _ => "odd".into(),
        }
    }

    fn outputs(&self) -> &[(PadId, Caps)] {
        &self.outputs
    }

    fn on_pad_added(&mut self, _callback: PadAddedCallback) {}

    fn name(&self) -> &str {
        "countdowndemux"
    }
}

async fn drain(handle: &parallax::elements::AppSinkHandle) -> Vec<u64> {
    let mut seqs = Vec::new();
    while let Pulled::Buffer(b) = handle.pull_buffer().await {
        seqs.push(b.metadata().sequence);
    }
    seqs
}

/// The acceptance test: two branches, each pad's buffers arrive only on its
/// own branch.
#[tokio::test(flavor = "multi_thread")]
async fn routed_buffers_reach_only_their_pads_links() {
    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let even_sink = AppSink::with_max_buffers(32);
    let even_handle = even_sink.handle();
    let odd_sink = AppSink::with_max_buffers(32);
    let odd_handle = odd_sink.handle();

    let src = pipeline.add_source("src", appsrc);
    let demux = pipeline.add_demuxer("demux", ParityDemuxer::new());
    let even = pipeline.add_sink("even_sink", even_sink);
    let odd = pipeline.add_sink("odd_sink", odd_sink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "even", even, "sink").unwrap();
    pipeline.link_pads(demux, "odd", odd, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for seq in 0..10u64 {
        src_handle.push_buffer(buffer_with_seq(seq)).await.unwrap();
    }
    src_handle.end_stream();

    let evens = drain(&even_handle).await;
    let odds = drain(&odd_handle).await;
    handle.wait().await.unwrap();

    assert_eq!(
        evens,
        vec![0, 2, 4, 6, 8],
        "even branch got exactly the evens"
    );
    assert_eq!(odds, vec![1, 3, 5, 7, 9], "odd branch got exactly the odds");
}

/// A source-style demuxer (no input link) is driven through produce() and
/// still routes per pad; EOS reaches both branches.
#[tokio::test(flavor = "multi_thread")]
async fn source_style_demuxer_produces_and_routes() {
    let mut pipeline = Pipeline::new();
    let even_sink = AppSink::with_max_buffers(32);
    let even_handle = even_sink.handle();
    let odd_sink = AppSink::with_max_buffers(32);
    let odd_handle = odd_sink.handle();

    let demux = pipeline.add_demuxer(
        "demux",
        CountdownDemuxer {
            remaining: 6,
            outputs: vec![(PadId(0), Caps::any()), (PadId(1), Caps::any())],
        },
    );
    let even = pipeline.add_sink("even_sink", even_sink);
    let odd = pipeline.add_sink("odd_sink", odd_sink);
    pipeline.link_pads(demux, "even", even, "sink").unwrap();
    pipeline.link_pads(demux, "odd", odd, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let evens = drain(&even_handle).await;
    let odds = drain(&odd_handle).await;
    handle.wait().await.unwrap();

    assert_eq!(evens, vec![4, 2, 0]);
    assert_eq!(odds, vec![5, 3, 1]);
}

/// The roadmap's acceptance test: a real MP4 in the pipeline, demuxed by an
/// in-pipeline element, with video buffers only on the video branch and
/// audio only on the audio branch — in decode order per branch.
#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_demux_source_routes_av_branches() {
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4AudioTrackConfig, Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use std::io::Cursor;

    // Fixture: interleaved A/V, video AVCC samples, silent-ish audio.
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
    let vs = pipeline.add_sink("video_sink", video_sink);
    let as_ = pipeline.add_sink("audio_sink", audio_sink);
    pipeline.link_pads(node, "video", vs, "sink").unwrap();
    pipeline.link_pads(node, "audio", as_, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let mut video_buffers = Vec::new();
    while let Pulled::Buffer(b) = video_handle.pull_buffer().await {
        video_buffers.push(b);
    }
    let mut audio_buffers = Vec::new();
    while let Pulled::Buffer(b) = audio_handle.pull_buffer().await {
        audio_buffers.push(b);
    }
    handle.wait().await.unwrap();

    assert_eq!(
        video_buffers.len(),
        6,
        "all video samples on the video branch"
    );
    assert_eq!(
        audio_buffers.len(),
        6,
        "all audio samples on the audio branch"
    );
    // Video came back as Annex-B (converted), audio raw — proof no buffer
    // crossed branches.
    for b in &video_buffers {
        assert_eq!(&b.as_bytes()[..4], &[0, 0, 0, 1], "video is Annex-B");
    }
    for b in &audio_buffers {
        assert_eq!(b.as_bytes(), &[0xAA, 0xBB], "audio payload untouched");
    }
    // Decode order per branch: PTS strictly increasing.
    for pair in video_buffers.windows(2) {
        assert!(pair[0].metadata().pts < pair[1].metadata().pts);
    }
}

/// with_loop: the stream rewinds at EOS instead of ending, until stopped.
#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_demux_source_loops_at_eos() {
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use std::io::Cursor;

    let mut mux = Mp4Mux::new(Cursor::new(Vec::new()), Mp4MuxConfig::default()).unwrap();
    let sps = vec![0x67, 0x42, 0x00, 0x1f];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];
    let video = mux
        .add_video_track(Mp4VideoTrackConfig::h264(320, 240, &sps, &pps))
        .unwrap();
    for i in 0..4u64 {
        mux.write_video_sample(
            video,
            &[0x00, 0x00, 0x00, 0x02, 0x65, 0xAA],
            i * 100,
            100,
            true,
        )
        .unwrap();
    }
    let mp4_data = mux.finish().unwrap().into_inner();
    let demux = Mp4Demux::new(Cursor::new(mp4_data.clone()), mp4_data.len() as u64).unwrap();

    let mut pipeline = Pipeline::new();
    let sink = AppSink::with_max_buffers(4);
    let sink_handle = sink.handle();
    let node = pipeline.add_demuxer(
        "mp4demux",
        Mp4DemuxSource::video_only(demux).with_loop(true),
    );
    let vs = pipeline.add_sink("video_sink", sink);
    pipeline.link_pads(node, "video", vs, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Far more buffers than the file holds: only looping can supply them.
    let mut pts = Vec::new();
    for _ in 0..10 {
        match sink_handle.pull_buffer().await {
            Pulled::Buffer(b) => pts.push(b.metadata().pts.nanos() / 1_000_000),
            other => panic!("stream ended early: {other:?}"),
        }
    }
    assert!(
        pts.windows(2).filter(|w| w[1] < w[0]).count() >= 2,
        "PTS wrapped at least twice across 10 pulls of a 4-sample file: {pts:?}"
    );

    handle.stop();
    while let Pulled::Buffer(_) = sink_handle.pull_buffer().await {}
    handle.wait().await.unwrap();
}

/// Runtime seek reaches a source-style demuxer: PipelineHandle::seek lands
/// Mp4DemuxSource on the target GOP's keyframe and playback continues from
/// there to EOS.
#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_demux_source_seeks_at_runtime() {
    use parallax::clock::ClockTime;
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use parallax::pipeline::bus::MessageKind;
    use std::io::Cursor;

    // Video-only: 20 frames at 100 ms, keyframe every 5 (t = 0/500/1000/1500).
    let mut mux = Mp4Mux::new(Cursor::new(Vec::new()), Mp4MuxConfig::default()).unwrap();
    let sps = vec![0x67, 0x42, 0x00, 0x1f];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];
    let video = mux
        .add_video_track(Mp4VideoTrackConfig::h264(320, 240, &sps, &pps))
        .unwrap();
    let keyframe = [0x00, 0x00, 0x00, 0x02, 0x65, 0xAA];
    let delta = [0x00, 0x00, 0x00, 0x02, 0x41, 0x9A];
    for i in 0..20u64 {
        let is_key = i % 5 == 0;
        let data: &[u8] = if is_key { &keyframe } else { &delta };
        mux.write_video_sample(video, data, i * 100, 100, is_key)
            .unwrap();
    }
    let mp4_data = mux.finish().unwrap().into_inner();
    let demux = Mp4Demux::new(Cursor::new(mp4_data.clone()), mp4_data.len() as u64).unwrap();

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(2);
    let video_handle = video_sink.handle();
    let node = pipeline.add_demuxer("mp4demux", Mp4DemuxSource::new(demux));
    let vs = pipeline.add_sink("video_sink", video_sink);
    pipeline.link_pads(node, "video", vs, "sink").unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    // The demuxer-rooted pipeline advertises its seekability at runtime
    // (#162): the snapshot came from Mp4DemuxSource's Demuxer impl.
    assert!(handle.seekable());
    assert!(handle.query_seekable().seekable);
    assert!(
        handle.duration().to_option().is_some(),
        "muxed fixture declares a duration"
    );

    // Consume the first couple of frames, then seek into the third GOP.
    for _ in 0..2 {
        assert!(matches!(
            video_handle.pull_buffer().await,
            Pulled::Buffer(_)
        ));
    }
    assert!(handle.seek_time(ClockTime::from_millis(1200)).await);

    // Keep pulling (the bounded sink needs the app to drain) until EOS.
    // In-flight pre-seek frames may still surface before the flush lands;
    // after it, playback restarts at the 1000 ms keyframe.
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
        "playback continued from the landing keyframe to EOS: {pts:?}"
    );

    // SeekDone reports where the demuxer ACTUALLY landed — the 1000 ms
    // keyframe, not the 1200 ms request (#162: honest completion).
    let mut seek_done = None;
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone {
            format, position, ..
        } = msg.kind
        {
            seek_done = Some((format, position));
        }
    }
    assert_eq!(
        seek_done,
        Some((parallax::event::SegmentFormat::Time, Some(1_000_000_000))),
        "SeekDone carries the snapped keyframe position"
    );
}

/// TsDemuxElement in a pipeline: TS bytes in, video frames only on the
/// video branch, audio only on audio.
#[cfg(feature = "mpeg-ts")]
#[tokio::test(flavor = "multi_thread")]
async fn ts_demux_element_routes_av_branches() {
    use parallax::clock::ClockTime;
    use parallax::elements::TsDemuxElement;
    use parallax::elements::mux::{TsMux, TsMuxConfig, TsMuxStreamType, TsMuxTrack};

    // Fixture: mux distinct payloads onto a video PID and an audio PID.
    let mut mux = TsMux::new(
        TsMuxConfig::new()
            .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
            .add_track(TsMuxTrack::new(257, TsMuxStreamType::AacAdts).audio()),
    );
    let mut ts_bytes = Vec::new();
    for i in 0..4u64 {
        let pts = Some(ClockTime::from_millis(i * 40));
        ts_bytes.extend(mux.write_pes(256, &[0x56u8; 32], pts, pts).unwrap());
        ts_bytes.extend(mux.write_pes(257, &[0x41u8; 16], pts, pts).unwrap());
    }

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(8);
    let src_handle = appsrc.handle();
    let video_sink = AppSink::with_max_buffers(32);
    let video_handle = video_sink.handle();
    let audio_sink = AppSink::with_max_buffers(32);
    let audio_handle = audio_sink.handle();

    let src = pipeline.add_source("src", appsrc);
    let demux = pipeline.add_demuxer("tsdemux", TsDemuxElement::new());
    let vs = pipeline.add_sink("video_sink", video_sink);
    let as_ = pipeline.add_sink("audio_sink", audio_sink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "video", vs, "sink").unwrap();
    pipeline.link_pads(demux, "audio", as_, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Push the TS capture in arbitrary chunks (the demuxer re-aligns
    // packet boundaries internally).
    let ts_arena = SharedArena::new(2048, 64).unwrap();
    for chunk in ts_bytes.chunks(1000) {
        let mut slot = ts_arena.acquire().unwrap();
        slot.data_mut()[..chunk.len()].copy_from_slice(chunk);
        let buffer = Buffer::new(MemoryHandle::with_len(slot, chunk.len()), Metadata::new());
        src_handle.push_buffer(buffer).await.unwrap();
    }
    src_handle.end_stream();

    let mut video_frames = Vec::new();
    while let Pulled::Buffer(b) = video_handle.pull_buffer().await {
        video_frames.push(b);
    }
    let mut audio_frames = Vec::new();
    while let Pulled::Buffer(b) = audio_handle.pull_buffer().await {
        audio_frames.push(b);
    }
    handle.wait().await.unwrap();

    assert!(!video_frames.is_empty(), "video frames arrived");
    assert!(!audio_frames.is_empty(), "audio frames arrived");
    // The mux pads unbounded video PES packets with 0xFF stuffing, so check
    // the routing property: each branch carries its own payload bytes and
    // none of the other branch's.
    for b in &video_frames {
        assert!(
            b.as_bytes().starts_with(&[0x56; 32]),
            "video payload intact"
        );
        assert!(!b.as_bytes().contains(&0x41), "no audio bytes on video");
        assert_eq!(b.metadata().stream_id, 256);
    }
    for b in &audio_frames {
        assert!(
            b.as_bytes().starts_with(&[0x41; 16]),
            "audio payload intact"
        );
        assert!(!b.as_bytes().contains(&0x56), "no video bytes on audio");
        assert_eq!(b.metadata().stream_id, 257);
    }
}

/// A pad with no links drops its buffers (with a warning) instead of
/// spraying them across the other branches — and the linked branch is
/// unaffected.
#[tokio::test(flavor = "multi_thread")]
async fn unlinked_pads_drop_instead_of_broadcasting() {
    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let even_sink = AppSink::with_max_buffers(32);
    let even_handle = even_sink.handle();

    let src = pipeline.add_source("src", appsrc);
    let demux = pipeline.add_demuxer("demux", ParityDemuxer::new());
    let even = pipeline.add_sink("even_sink", even_sink);
    pipeline.link(src, demux).unwrap();
    // The "odd" pad is deliberately left unlinked.
    pipeline.link_pads(demux, "even", even, "sink").unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for seq in 0..6u64 {
        src_handle.push_buffer(buffer_with_seq(seq)).await.unwrap();
    }
    src_handle.end_stream();

    let evens = drain(&even_handle).await;
    handle.wait().await.unwrap();

    assert_eq!(
        evens,
        vec![0, 2, 4],
        "odd buffers were dropped, not rerouted"
    );
}
