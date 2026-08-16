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
    let even = pipeline.add_async_sink("even_sink", even_sink);
    let odd = pipeline.add_async_sink("odd_sink", odd_sink);
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
    let even = pipeline.add_async_sink("even_sink", even_sink);
    let odd = pipeline.add_async_sink("odd_sink", odd_sink);
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
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    let as_ = pipeline.add_async_sink("audio_sink", audio_sink);
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
    let vs = pipeline.add_async_sink("video_sink", sink);
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
        let is_key = i.is_multiple_of(5);
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
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    // Capacity 2 on purpose: with the default 16 (+ the sink's queue of 2)
    // the whole remainder of the 20-frame fixture fits in flight, so a fast
    // demuxer can reach EOS before the hop-by-hop seek arrives and the test
    // races scheduling. A small channel guarantees the producer is parked on
    // back-pressure when the seek lands.
    pipeline
        .link_pads_full(
            node,
            "video",
            vs,
            "sink",
            parallax::pipeline::LinkPolicy::Block,
            Some(2),
        )
        .unwrap();

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

/// SNAP_AFTER steers the runtime seek forward: the same 1200 ms request
/// lands on the NEXT keyframe (1500 ms) and SeekDone reports it (#166).
#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_demux_source_snap_after_seeks_at_runtime() {
    use parallax::clock::ClockTime;
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use parallax::event::{SeekEvent, SeekFlags};
    use parallax::pipeline::bus::MessageKind;
    use std::io::Cursor;

    // Same fixture as above: keyframes at 0/500/1000/1500 ms.
    let mut mux = Mp4Mux::new(Cursor::new(Vec::new()), Mp4MuxConfig::default()).unwrap();
    let sps = vec![0x67, 0x42, 0x00, 0x1f];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];
    let video = mux
        .add_video_track(Mp4VideoTrackConfig::h264(320, 240, &sps, &pps))
        .unwrap();
    let keyframe = [0x00, 0x00, 0x00, 0x02, 0x65, 0xAA];
    let delta = [0x00, 0x00, 0x00, 0x02, 0x41, 0x9A];
    for i in 0..20u64 {
        let is_key = i.is_multiple_of(5);
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
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    // Capacity 2 on purpose: with the default 16 (+ the sink's queue of 2)
    // the whole remainder of the 20-frame fixture fits in flight, so a fast
    // demuxer can reach EOS before the hop-by-hop seek arrives and the test
    // races scheduling. A small channel guarantees the producer is parked on
    // back-pressure when the seek lands.
    pipeline
        .link_pads_full(
            node,
            "video",
            vs,
            "sink",
            parallax::pipeline::LinkPolicy::Block,
            Some(2),
        )
        .unwrap();

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
        .with_flags(SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::SNAP_AFTER);
    assert!(handle.seek(seek).await);

    let mut pts = Vec::new();
    while let Pulled::Buffer(b) = video_handle.pull_buffer().await {
        pts.push(b.metadata().pts.nanos() / 1_000_000);
    }
    handle.wait().await.unwrap();

    let landing = pts
        .iter()
        .rposition(|p| *p == 1500)
        .expect("landing keyframe was presented");
    assert_eq!(
        &pts[landing..],
        &[1500, 1600, 1700, 1800, 1900],
        "playback continued from the forward keyframe to EOS: {pts:?}"
    );

    let mut seek_done = None;
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone { position, .. } = msg.kind {
            seek_done = Some(position);
        }
    }
    assert_eq!(
        seek_done,
        Some(Some(1_500_000_000)),
        "SeekDone carries the forward-snapped keyframe position"
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
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    let as_ = pipeline.add_async_sink("audio_sink", audio_sink);
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
    let even = pipeline.add_async_sink("even_sink", even_sink);
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

/// #165: a demuxer owns its pads' stream identity — each pad gets its own
/// StreamStart and a lazy Time segment anchored at that pad's first PTS.
/// Upstream StreamStart/Segment (the fed input's domain) is swallowed.
#[tokio::test(flavor = "multi_thread")]
async fn demuxer_pads_each_get_stream_start_and_segment() {
    use parallax::clock::ClockTime;
    use parallax::event::{Event, SegmentFormat};
    use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
    use std::sync::{Arc, Mutex};

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(16);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let demux = pipeline.add_demuxer("parity", ParityDemuxer::new());
    let even_sink = AppSink::with_max_buffers(16);
    let even_handle = even_sink.handle();
    let odd_sink = AppSink::with_max_buffers(16);
    let odd_handle = odd_sink.handle();
    let ev = pipeline.add_async_sink("even_sink", even_sink);
    let od = pipeline.add_async_sink("odd_sink", odd_sink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "even", ev, "sink").unwrap();
    pipeline.link_pads(demux, "odd", od, "sink").unwrap();

    // Capture events on the demuxer's src side (one PadRef per node).
    let events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let events_clone = events.clone();
    let _ = pipeline.add_probe(PadRef::src(demux), ProbeType::EVENT_DOWN, move |data| {
        if let ProbeData::Event(e) = data {
            let entry = match e {
                Event::Segment(seg) => format!("segment:{:?}:{}", seg.format, seg.start),
                other => other.name().to_string(),
            };
            events_clone.lock().unwrap().push(entry);
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // seq 0 (pts 100 ms) → even, seq 1 (pts 250 ms) → odd.
    for (seq, pts_ms) in [(0u64, 100u64), (1, 250), (2, 300), (3, 450)] {
        let slot = arena().acquire().unwrap();
        let mut meta = Metadata::from_sequence(seq);
        meta.pts = ClockTime::from_millis(pts_ms);
        src_handle
            .push_buffer(Buffer::new(MemoryHandle::with_len(slot, 8), meta))
            .await
            .unwrap();
    }
    src_handle.end_stream();
    while let Pulled::Buffer(_) = even_handle.pull_buffer().await {}
    while let Pulled::Buffer(_) = odd_handle.pull_buffer().await {}
    handle.wait().await.unwrap();

    let events = events.lock().unwrap();
    let stream_starts = events.iter().filter(|e| *e == "stream-start").count();
    assert_eq!(stream_starts, 2, "one StreamStart per pad: {events:?}");
    let segments: Vec<&String> = events
        .iter()
        .filter(|e| e.starts_with("segment:"))
        .collect();
    assert_eq!(
        segments,
        vec![
            &format!("segment:{:?}:100000000", SegmentFormat::Time),
            &format!("segment:{:?}:250000000", SegmentFormat::Time),
        ],
        "each pad anchors at its own first PTS; the upstream segment is \
         swallowed: {events:?}"
    );
}

/// #165: after an upstream flushing seek passes through a FED demuxer, each
/// pad re-anchors with a fresh Time segment at its first post-seek buffer —
/// the source's own (input-domain) seek segment is swallowed.
#[tokio::test(flavor = "multi_thread")]
async fn fed_demuxer_reanchors_segments_after_upstream_seek() {
    use parallax::clock::ClockTime;
    use parallax::event::Event;
    use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
    use std::sync::{Arc, Mutex};

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(16);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let demux = pipeline.add_demuxer("parity", ParityDemuxer::new());
    let even_sink = AppSink::with_max_buffers(16);
    let even_handle = even_sink.handle();
    let odd_sink = AppSink::with_max_buffers(16);
    let odd_handle = odd_sink.handle();
    let ev = pipeline.add_async_sink("even_sink", even_sink);
    let od = pipeline.add_async_sink("odd_sink", odd_sink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "even", ev, "sink").unwrap();
    pipeline.link_pads(demux, "odd", od, "sink").unwrap();

    let log: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log_clone = log.clone();
    let _ = pipeline.add_probe(
        PadRef::src(demux),
        ProbeType::EVENT_DOWN | ProbeType::EVENT_FLUSH,
        move |data| {
            if let ProbeData::Event(e) = data {
                let entry = match e {
                    Event::Segment(seg) => format!("segment:{}", seg.start),
                    other => other.name().to_string(),
                };
                log_clone.lock().unwrap().push(entry);
            }
            ProbeReturn::Ok
        },
    );

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for (seq, pts_ms) in [(0u64, 100u64), (1, 250)] {
        let slot = arena().acquire().unwrap();
        let mut meta = Metadata::from_sequence(seq);
        meta.pts = ClockTime::from_millis(pts_ms);
        src_handle
            .push_buffer(Buffer::new(MemoryHandle::with_len(slot, 8), meta))
            .await
            .unwrap();
    }
    // Both pads have anchored.
    for _ in 0..200 {
        if log
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.starts_with("segment:"))
            .count()
            >= 2
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    // AppSrc is seekable by proxy; the flush trio travels through the fed
    // demuxer to every pad.
    assert!(handle.seek_time(ClockTime::from_secs(10)).await);
    for _ in 0..200 {
        if log.lock().unwrap().iter().any(|e| e == "flush-stop") {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    // Post-seek data re-anchors each pad.
    for (seq, pts_ms) in [(2u64, 10_000u64), (3, 10_250)] {
        let slot = arena().acquire().unwrap();
        let mut meta = Metadata::from_sequence(seq);
        meta.pts = ClockTime::from_millis(pts_ms);
        src_handle
            .push_buffer(Buffer::new(MemoryHandle::with_len(slot, 8), meta))
            .await
            .unwrap();
    }
    src_handle.end_stream();
    while let Pulled::Buffer(_) = even_handle.pull_buffer().await {}
    while let Pulled::Buffer(_) = odd_handle.pull_buffer().await {}
    handle.wait().await.unwrap();

    let log = log.lock().unwrap();
    let flush_stop = log
        .iter()
        .position(|e| e == "flush-stop")
        .unwrap_or_else(|| panic!("no flush-stop: {log:?}"));
    let post: Vec<&String> = log[flush_stop..]
        .iter()
        .filter(|e| e.starts_with("segment:"))
        .collect();
    assert_eq!(
        post,
        vec![
            &"segment:10000000000".to_string(),
            &"segment:10250000000".to_string()
        ],
        "each pad re-anchors at its first post-seek PTS; the source's seek \
         segment is swallowed: {log:?}"
    );
}

/// #165 fast-forward: a rate>1 seek switches Mp4DemuxSource to
/// keyframe-only output; a rate-1.0 seek restores every frame. The
/// fixture loops — trick-mode discards generate no backpressure, so a
/// finite stream would race to EOS before the restore seek could land.
#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_fast_forward_emits_keyframes_only() {
    use parallax::clock::ClockTime;
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use parallax::event::SeekEvent;
    use std::io::Cursor;

    // 20 frames at 100 ms, keyframe every 5 (t = 0/500/1000/1500).
    let mut mux = Mp4Mux::new(Cursor::new(Vec::new()), Mp4MuxConfig::default()).unwrap();
    let sps = vec![0x67, 0x42, 0x00, 0x1f];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];
    let video = mux
        .add_video_track(Mp4VideoTrackConfig::h264(320, 240, &sps, &pps))
        .unwrap();
    let keyframe = [0x00, 0x00, 0x00, 0x02, 0x65, 0xAA];
    let delta = [0x00, 0x00, 0x00, 0x02, 0x41, 0x9A];
    for i in 0..20u64 {
        let is_key = i.is_multiple_of(5);
        let data: &[u8] = if is_key { &keyframe } else { &delta };
        mux.write_video_sample(video, data, i * 100, 100, is_key)
            .unwrap();
    }
    let mp4_data = mux.finish().unwrap().into_inner();
    let demux = Mp4Demux::new(Cursor::new(mp4_data.clone()), mp4_data.len() as u64).unwrap();

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(2);
    let video_handle = video_sink.handle();
    let node = pipeline.add_demuxer(
        "mp4demux",
        Mp4DemuxSource::video_only(demux).with_loop(true),
    );
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    pipeline
        .link_pads_full(
            node,
            "video",
            vs,
            "sink",
            parallax::pipeline::LinkPolicy::Block,
            Some(2),
        )
        .unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for _ in 0..2 {
        assert!(matches!(
            video_handle.pull_buffer().await,
            Pulled::Buffer(_)
        ));
    }

    // Phase 1: fast-forward. Snap-before lands the 500 ms keyframe; while
    // rate > 1 only keyframes flow (looping through 0/500/1000/1500).
    let seek = SeekEvent::new_time(ClockTime::from_millis(600)).with_rate(2.0);
    assert!(handle.seek(seek).await);

    let mut pts = Vec::new();
    while pts.len() < 12 {
        match video_handle.pull_buffer().await {
            Pulled::Buffer(b) => pts.push(b.metadata().pts.nanos() / 1_000_000),
            Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
            other => panic!("stream ended during fast-forward: {other:?}"),
        }
    }
    // Everything from the landing keyframe on is a keyframe time.
    let landing = pts.iter().position(|p| *p == 500).expect("landing at 500");
    assert!(
        pts.len() - landing >= 6,
        "expected several trick frames after the landing: {pts:?}"
    );
    assert!(
        pts[landing..].iter().all(|p| p.is_multiple_of(500)),
        "fast-forward emitted a non-keyframe: {pts:?}"
    );

    // Phase 2: a rate-1.0 seek restores normal playback (lands 1000, then
    // every 100 ms frame follows).
    assert!(handle.seek_time(ClockTime::from_millis(1200)).await);
    let mut pts2 = Vec::new();
    while pts2.len() < 16 {
        match video_handle.pull_buffer().await {
            Pulled::Buffer(b) => pts2.push(b.metadata().pts.nanos() / 1_000_000),
            Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
            other => panic!("stream ended after the restore seek: {other:?}"),
        }
    }
    handle.stop();
    while let Pulled::Buffer(_) = video_handle.pull_buffer().await {}
    handle.wait().await.unwrap();

    let expected = [1000, 1100, 1200, 1300, 1400];
    assert!(
        pts2.windows(expected.len()).any(|w| w == expected),
        "normal playback restored after the rate-1.0 seek: {pts2:?}"
    );
}

/// #165 SEGMENT seeks: a SEGMENT-flagged seek with a stop posts
/// `SegmentDone` on the bus when playback reaches the stop — the sink never
/// sees EOS, the producer idles — and the app's follow-up NON-flushing
/// SEGMENT seek starts the next lap gaplessly: its queued segment carries
/// the accumulated `base`, so running time is monotonic across laps.
#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_segment_seek_loops_gaplessly() {
    use parallax::clock::ClockTime;
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use parallax::event::{Event, SeekEvent, SeekFlags, SeekPosition, SegmentEvent};
    use parallax::pipeline::bus::MessageKind;
    use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    // 20 frames at 100 ms, keyframe every 5 (t = 0/500/1000/1500 ms).
    let mut mux = Mp4Mux::new(Cursor::new(Vec::new()), Mp4MuxConfig::default()).unwrap();
    let sps = vec![0x67, 0x42, 0x00, 0x1f];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];
    let video = mux
        .add_video_track(Mp4VideoTrackConfig::h264(320, 240, &sps, &pps))
        .unwrap();
    let keyframe = [0x00, 0x00, 0x00, 0x02, 0x65, 0xAA];
    let delta = [0x00, 0x00, 0x00, 0x02, 0x41, 0x9A];
    for i in 0..20u64 {
        let is_key = i.is_multiple_of(5);
        let data: &[u8] = if is_key { &keyframe } else { &delta };
        mux.write_video_sample(video, data, i * 100, 100, is_key)
            .unwrap();
    }
    let mp4_data = mux.finish().unwrap().into_inner();
    let demux = Mp4Demux::new(Cursor::new(mp4_data.clone()), mp4_data.len() as u64).unwrap();

    let mut pipeline = Pipeline::new();
    // Small queue + Block link: the producer stalls a few frames in, so a
    // seek lands before startup playback can run past the assertions below.
    let video_sink = AppSink::with_max_buffers(2);
    let video_handle = video_sink.handle();
    let node = pipeline.add_demuxer("mp4demux", Mp4DemuxSource::video_only(demux));
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    pipeline
        .link_pads_full(
            node,
            "video",
            vs,
            "sink",
            parallax::pipeline::LinkPolicy::Block,
            Some(2),
        )
        .unwrap();

    let segments: Arc<Mutex<Vec<SegmentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let segments_probe = segments.clone();
    let _ = pipeline.add_probe(PadRef::sink(vs), ProbeType::EVENT_DOWN, move |data| {
        if let ProbeData::Event(Event::Segment(seg)) = data {
            segments_probe.lock().unwrap().push(seg.clone());
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    for _ in 0..2 {
        assert!(matches!(
            video_handle.pull_buffer().await,
            Pulled::Buffer(_)
        ));
    }

    // Lap 1: flushing SEGMENT seek over [0, 800 ms).
    let seek = SeekEvent::new_time(ClockTime::ZERO)
        .with_flags(SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::SEGMENT)
        .with_stop(SeekPosition::set(800_000_000));
    assert!(handle.seek(seek).await);

    // Pull until SegmentDone arrives; the sink must never report Ended.
    let mut lap1 = Vec::new();
    let mut segment_done = None;
    for _ in 0..2000 {
        while let Some(msg) = bus.poll() {
            if let MessageKind::SegmentDone {
                seqnum, position, ..
            } = msg.kind
            {
                segment_done = Some((seqnum, position));
            }
        }
        match video_handle.try_pull_buffer() {
            Pulled::Buffer(b) => {
                lap1.push(b.metadata().pts.nanos() / 1_000_000);
                continue;
            }
            Pulled::Ended(reason) => panic!("SEGMENT seek must not end the stream: {reason:?}"),
            _ => {}
        }
        // SegmentDone posts when the PRODUCER exhausts the range; trailing
        // frames may still be in flight — drain until the last one lands.
        if segment_done.is_some() && lap1.contains(&700) {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    let (_seq1, done_pos) = segment_done.expect("SegmentDone posted");
    assert_eq!(done_pos, Some(800_000_000), "position reports the stop");
    assert!(
        lap1.iter().all(|p| *p < 800),
        "playback ended at the seek's stop: {lap1:?}"
    );
    assert!(
        lap1.contains(&700),
        "the last pre-stop frame was delivered: {lap1:?}"
    );

    // Lap 2: NON-flushing SEGMENT seek back to 0 — gapless.
    let lap2_seek = SeekEvent::new_time(ClockTime::ZERO)
        .with_flags(SeekFlags::KEY_UNIT | SeekFlags::SEGMENT)
        .with_stop(SeekPosition::set(800_000_000));
    assert!(handle.seek(lap2_seek).await);

    let mut lap2 = Vec::new();
    for _ in 0..2000 {
        match video_handle.try_pull_buffer() {
            Pulled::Buffer(b) => {
                lap2.push(b.metadata().pts.nanos() / 1_000_000);
                continue;
            }
            Pulled::Ended(reason) => panic!("lap 2 must not end the stream: {reason:?}"),
            _ => {}
        }
        if lap2.len() >= 4 {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }
    assert!(
        lap2.first() == Some(&0),
        "lap 2 restarts at the seek target: {lap2:?}"
    );

    // The lap-2 segment accumulated base: 8 frames of lap 1 (0..=700 ms,
    // anchored at 0, rate 1) -> 700 ms of running time.
    let segs = segments.lock().unwrap().clone();
    let lap2_seg = segs
        .iter()
        .rev()
        .find(|s| s.base > 0)
        .unwrap_or_else(|| panic!("lap-2 segment carries accumulated base: {segs:?}"));
    assert_eq!(lap2_seg.base, 700_000_000, "{segs:?}");
    assert_eq!(lap2_seg.start, 0);
    // Monotonic running time across the lap boundary.
    let lap1_seg = segs.iter().find(|s| s.base == 0 && s.start == 0).unwrap();
    let end_of_lap1 = lap1_seg
        .to_running_time(ClockTime::from_nanos(700_000_000))
        .nanos();
    let start_of_lap2 = lap2_seg.to_running_time(ClockTime::ZERO).nanos();
    assert!(
        start_of_lap2 >= end_of_lap1,
        "gapless: lap 2 running time continues ({end_of_lap1} -> {start_of_lap2})"
    );

    handle.stop();
    loop {
        match video_handle.pull_buffer().await {
            Pulled::Buffer(_) | Pulled::Flushing | Pulled::Empty => {}
            Pulled::Ended(_) => break,
        }
    }
    handle.wait().await.unwrap();
}

/// #165: a plain (non-SEGMENT) seek with a stop really ends at the stop —
/// the sink sees EOS after the last pre-stop frame.
#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_plain_seek_with_stop_ends_at_stop() {
    use parallax::clock::ClockTime;
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use parallax::event::{SeekEvent, SeekPosition};
    use std::io::Cursor;

    let mut mux = Mp4Mux::new(Cursor::new(Vec::new()), Mp4MuxConfig::default()).unwrap();
    let sps = vec![0x67, 0x42, 0x00, 0x1f];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];
    let video = mux
        .add_video_track(Mp4VideoTrackConfig::h264(320, 240, &sps, &pps))
        .unwrap();
    let keyframe = [0x00, 0x00, 0x00, 0x02, 0x65, 0xAA];
    let delta = [0x00, 0x00, 0x00, 0x02, 0x41, 0x9A];
    for i in 0..20u64 {
        let is_key = i.is_multiple_of(5);
        let data: &[u8] = if is_key { &keyframe } else { &delta };
        mux.write_video_sample(video, data, i * 100, 100, is_key)
            .unwrap();
    }
    let mp4_data = mux.finish().unwrap().into_inner();
    let demux = Mp4Demux::new(Cursor::new(mp4_data.clone()), mp4_data.len() as u64).unwrap();

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(2);
    let video_handle = video_sink.handle();
    let node = pipeline.add_demuxer("mp4demux", Mp4DemuxSource::video_only(demux));
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    pipeline
        .link_pads_full(
            node,
            "video",
            vs,
            "sink",
            parallax::pipeline::LinkPolicy::Block,
            Some(2),
        )
        .unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for _ in 0..2 {
        assert!(matches!(
            video_handle.pull_buffer().await,
            Pulled::Buffer(_)
        ));
    }

    // Default flags (FLUSH | KEY_UNIT), stop at 800 ms, no SEGMENT.
    let seek = SeekEvent::new_time(ClockTime::ZERO).with_stop(SeekPosition::set(800_000_000));
    assert!(handle.seek(seek).await);

    let mut pts = Vec::new();
    loop {
        match video_handle.pull_buffer().await {
            Pulled::Buffer(b) => pts.push(b.metadata().pts.nanos() / 1_000_000),
            Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
            Pulled::Ended(_) => break,
        }
    }
    // A stale pre-seek frame can race the flush; judge the post-landing
    // tail (the last restart at 0), which is the seek's own playback.
    let landing = pts.iter().rposition(|p| *p == 0).unwrap_or(0);
    let tail = &pts[landing..];
    assert!(
        tail.iter().all(|p| *p < 800),
        "nothing at/past the stop: {pts:?}"
    );
    assert!(tail.contains(&700), "last pre-stop frame arrived: {pts:?}");
    handle.wait().await.unwrap();
}

/// #165 reverse playback (MP4-only, keyframe trick mode): a rate<0 seek
/// walks video keyframes backward from the range's top; PTS strictly
/// decrease, every frame is a keyframe, mapped running times strictly
/// increase, and the walk ends with EOS after keyframe 0.
#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_reverse_seek_walks_keyframes_backward() {
    use parallax::clock::ClockTime;
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use parallax::event::{Event, SeekEvent, SegmentEvent};
    use parallax::pipeline::bus::MessageKind;
    use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    // 20 frames at 100 ms, keyframes at 0/500/1000/1500 ms; duration 2 s.
    let mut mux = Mp4Mux::new(Cursor::new(Vec::new()), Mp4MuxConfig::default()).unwrap();
    let sps = vec![0x67, 0x42, 0x00, 0x1f];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];
    let video = mux
        .add_video_track(Mp4VideoTrackConfig::h264(320, 240, &sps, &pps))
        .unwrap();
    let keyframe = [0x00, 0x00, 0x00, 0x02, 0x65, 0xAA];
    let delta = [0x00, 0x00, 0x00, 0x02, 0x41, 0x9A];
    for i in 0..20u64 {
        let is_key = i.is_multiple_of(5);
        let data: &[u8] = if is_key { &keyframe } else { &delta };
        mux.write_video_sample(video, data, i * 100, 100, is_key)
            .unwrap();
    }
    let mp4_data = mux.finish().unwrap().into_inner();
    let demux = Mp4Demux::new(Cursor::new(mp4_data.clone()), mp4_data.len() as u64).unwrap();

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(2);
    let video_handle = video_sink.handle();
    let node = pipeline.add_demuxer("mp4demux", Mp4DemuxSource::video_only(demux));
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    pipeline
        .link_pads_full(
            node,
            "video",
            vs,
            "sink",
            parallax::pipeline::LinkPolicy::Block,
            Some(2),
        )
        .unwrap();

    let segments: Arc<Mutex<Vec<SegmentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let segments_probe = segments.clone();
    let _ = pipeline.add_probe(PadRef::sink(vs), ProbeType::EVENT_DOWN, move |data| {
        if let ProbeData::Event(Event::Segment(seg)) = data {
            segments_probe.lock().unwrap().push(seg.clone());
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    for _ in 0..2 {
        assert!(matches!(
            video_handle.pull_buffer().await,
            Pulled::Buffer(_)
        ));
    }

    // Reverse over the whole file: stop unset -> the demuxer resolves it
    // from the duration and reports it as the landing.
    let seek = SeekEvent::new_time(ClockTime::ZERO).with_rate(-1.0);
    assert!(handle.seek(seek).await);

    let mut pts = Vec::new();
    loop {
        match video_handle.pull_buffer().await {
            Pulled::Buffer(b) => pts.push(b.metadata().pts.nanos() / 1_000_000),
            Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
            Pulled::Ended(_) => break,
        }
    }

    // Post-seek tail: everything from the first top-of-range keyframe on.
    let top = pts
        .iter()
        .position(|p| *p == 1_500)
        .unwrap_or_else(|| panic!("reverse walk starts at the last keyframe: {pts:?}"));
    let tail = &pts[top..];
    assert_eq!(tail, &[1_500, 1_000, 500, 0], "keyframes backward: {pts:?}");

    // SeekDone reports the range top for a reverse seek.
    let mut seek_done_pos = None;
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone { position, .. } = msg.kind {
            seek_done_pos = Some(position);
        }
    }
    assert_eq!(seek_done_pos, Some(Some(2_000_000_000)));

    // The reverse segment maps the decreasing PTS to increasing running
    // times.
    let segs = segments.lock().unwrap().clone();
    let rseg = segs
        .iter()
        .find(|s| s.rate < 0.0)
        .unwrap_or_else(|| panic!("a reverse segment was emitted: {segs:?}"));
    assert_eq!(rseg.start, 0);
    assert_eq!(rseg.stop, 2_000_000_000);
    let rts: Vec<u64> = tail
        .iter()
        .map(|p| rseg.to_running_time(ClockTime::from_millis(*p)).nanos())
        .collect();
    assert!(
        rts.windows(2).all(|w| w[0] < w[1]),
        "running time increases in reverse: {rts:?}"
    );

    handle.wait().await.unwrap();
}

/// #165 ACCURATE: the synthesized segment starts at the REQUESTED time while
/// data still starts at the snapped keyframe — the gap is out-of-segment on
/// purpose (decoders decode-but-drop it), and SeekDone keeps reporting the
/// honest landing.
#[cfg(feature = "mp4-demux")]
#[tokio::test(flavor = "multi_thread")]
async fn mp4_accurate_seek_segment_starts_at_the_request() {
    use parallax::clock::ClockTime;
    use parallax::elements::Mp4DemuxSource;
    use parallax::elements::demux::Mp4Demux;
    use parallax::elements::mux::{Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};
    use parallax::event::{Event, SeekEvent, SeekFlags, SegmentEvent};
    use parallax::pipeline::bus::MessageKind;
    use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
    use std::io::Cursor;
    use std::sync::{Arc, Mutex};

    // 20 frames at 100 ms, keyframes at 0/500/1000/1500 ms.
    let mut mux = Mp4Mux::new(Cursor::new(Vec::new()), Mp4MuxConfig::default()).unwrap();
    let sps = vec![0x67, 0x42, 0x00, 0x1f];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];
    let video = mux
        .add_video_track(Mp4VideoTrackConfig::h264(320, 240, &sps, &pps))
        .unwrap();
    let keyframe = [0x00, 0x00, 0x00, 0x02, 0x65, 0xAA];
    let delta = [0x00, 0x00, 0x00, 0x02, 0x41, 0x9A];
    for i in 0..20u64 {
        let is_key = i.is_multiple_of(5);
        let data: &[u8] = if is_key { &keyframe } else { &delta };
        mux.write_video_sample(video, data, i * 100, 100, is_key)
            .unwrap();
    }
    let mp4_data = mux.finish().unwrap().into_inner();
    let demux = Mp4Demux::new(Cursor::new(mp4_data.clone()), mp4_data.len() as u64).unwrap();

    let mut pipeline = Pipeline::new();
    let video_sink = AppSink::with_max_buffers(2);
    let video_handle = video_sink.handle();
    let node = pipeline.add_demuxer("mp4demux", Mp4DemuxSource::video_only(demux));
    let vs = pipeline.add_async_sink("video_sink", video_sink);
    pipeline
        .link_pads_full(
            node,
            "video",
            vs,
            "sink",
            parallax::pipeline::LinkPolicy::Block,
            Some(2),
        )
        .unwrap();

    let segments: Arc<Mutex<Vec<SegmentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let segments_probe = segments.clone();
    let _ = pipeline.add_probe(PadRef::sink(vs), ProbeType::EVENT_DOWN, move |data| {
        if let ProbeData::Event(Event::Segment(seg)) = data {
            segments_probe.lock().unwrap().push(seg.clone());
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    for _ in 0..2 {
        assert!(matches!(
            video_handle.pull_buffer().await,
            Pulled::Buffer(_)
        ));
    }

    // ACCURATE seek to mid-GOP 700 ms: MP4 snaps the data to the 500 ms
    // keyframe, but the segment must start at the request.
    let seek = SeekEvent::new_time(ClockTime::from_millis(700))
        .with_flags(SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::ACCURATE);
    assert!(handle.seek(seek).await);

    let mut pts = Vec::new();
    loop {
        match video_handle.pull_buffer().await {
            Pulled::Buffer(b) => pts.push(b.metadata().pts.nanos() / 1_000_000),
            Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
            Pulled::Ended(_) => break,
        }
    }

    // Data starts at the snapped keyframe (stale pre-seek frames may race
    // the flush; judge from the seek's own landing).
    let landing = pts
        .iter()
        .position(|p| *p == 500)
        .unwrap_or_else(|| panic!("data starts at the 500 ms keyframe: {pts:?}"));
    assert_eq!(
        &pts[landing..landing + 4],
        &[500, 600, 700, 800],
        "playback proceeds from the keyframe: {pts:?}"
    );

    // The segment starts at the REQUEST, not the keyframe.
    let segs = segments.lock().unwrap().clone();
    let accurate_seg = segs
        .iter()
        .find(|s| s.start == 700_000_000)
        .unwrap_or_else(|| panic!("segment starts at the requested 700 ms: {segs:?}"));
    assert_eq!(accurate_seg.rate, 1.0);

    // SeekDone still reports the honest landing (the keyframe).
    let mut seek_done_pos = None;
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone { position, .. } = msg.kind {
            seek_done_pos = Some(position);
        }
    }
    assert_eq!(
        seek_done_pos,
        Some(Some(500_000_000)),
        "SeekDone reports the snapped landing"
    );

    handle.wait().await.unwrap();
}
