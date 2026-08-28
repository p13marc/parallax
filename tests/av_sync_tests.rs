//! A/V stays in sync across a seek (#265).
//!
//! # What went wrong
//!
//! Running time was `clock.now() - base_time`, with `base_time` snapshotted
//! at `Executor::start` and copied into every element. A flushing seek
//! restarts every segment at running time zero (`base = 0`) while an
//! audio-provided master clock keeps counting monotonically upward, so after
//! a seek the two quantities no longer name the same instant. The video sink
//! papered over that with a *private, arrival-time* anchor — whatever moment
//! its first post-seek frame reached the sink became running time zero for
//! video — and the audio sink had no anchor at all, being entirely PTS-blind.
//! The resulting offset was the arrival skew between the branches: arbitrary
//! per seek, and frozen in until the next one.
//!
//! # What these tests assert
//!
//! Both sinks are asked, per buffer, for the origin their behaviour implies:
//! `clock_reading_when_it_played - its_segment_running_time`. Two sinks are
//! in sync exactly when they agree on that number. Every test here is a
//! statement about that agreement — before a seek, after one, after several,
//! and across a `--loop` rewind.
//!
//! The sinks are stand-ins (`tests/support/sync.rs`) because `AlsaSink` needs
//! a device and `AutoVideoSink` needs a window, but they run the *shared*
//! `pipeline::sync` machinery the real ones run.

#[path = "support/sync.rs"]
mod sync;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{Demuxer, DemuxerProduce, PadId, RoutedOutput};
use parallax::elements::AppVideoSink;
use parallax::error::Result;
use parallax::event::{Event, EventResult, SeekType};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, ExecutorConfig, LinkPolicy, Pipeline};
use parallax::video::FrameMailbox;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use sync::{ClockAudioSink, LogHandle, PacedVideoSink, Presented, SlowRestart, record};

/// 10 fps video, so a frame every 100 ms.
const FRAME_NS: u64 = 100_000_000;
/// One keyframe every half second, like `tiny_h264_aac.mkv`.
const GOP_FRAMES: u64 = 5;
/// 10 ms of 48 kHz stereo S16 per audio buffer.
const AUDIO_NS: u64 = 10_000_000;
const SAMPLE_RATE: u32 = 48_000;
const AUDIO_BYTES: usize = 480 * 2 * 2;

/// How far the two branches may disagree about running-time zero.
///
/// One audio buffer is 10 ms, so that much is pure quantisation — the sinks
/// report the clock at buffer granularity. A quarter of a video frame is a
/// generous ceiling on top of it, and an order of magnitude below the
/// hundreds of milliseconds the arrival anchor used to produce.
const TOLERANCE_MS: f64 = 25.0;

const VIDEO_PAD: PadId = PadId(0);
const AUDIO_PAD: PadId = PadId(1);

/// A two-pad demuxer with an MP4-shaped seek: the video track snaps *back* to
/// its keyframe and the audio track then lands at or before that keyframe, so
/// the first post-seek audio buffer straddles the segment start exactly as a
/// real one does.
struct AvSource {
    arena: SharedArena,
    /// Next video frame index.
    video: u64,
    /// Next audio buffer index.
    audio: u64,
    /// Stop after this many video frames.
    frames: u64,
    /// Rewind to zero at the end instead of reporting EOS.
    looping: bool,
    /// Nanoseconds per video frame. The frame rate decides how many seconds
    /// of stream the branch depths hold, which is the whole subject of the
    /// occlusion tests.
    frame_ns: u64,
    staged: Option<parallax::event::SegmentEvent>,
    outputs: Vec<(PadId, parallax::format::Caps)>,
}

impl AvSource {
    fn new(frames: u64) -> Self {
        Self {
            arena: SharedArena::new(AUDIO_BYTES.next_power_of_two(), 256).unwrap(),
            video: 0,
            audio: 0,
            frames,
            looping: false,
            frame_ns: FRAME_NS,
            staged: None,
            outputs: vec![
                (VIDEO_PAD, parallax::format::Caps::any()),
                (AUDIO_PAD, parallax::format::Caps::any()),
            ],
        }
    }

    fn looping(mut self) -> Self {
        self.looping = true;
        self
    }

    /// Run at a different frame rate than the default 10 fps.
    fn with_frame_ns(mut self, frame_ns: u64) -> Self {
        self.frame_ns = frame_ns;
        self
    }

    fn emit(&self, len: usize, pts: u64) -> Option<Buffer> {
        self.arena.reclaim();
        let slot = self.arena.acquire()?;
        let mut meta = Metadata::new();
        meta.pts = ClockTime::from_nanos(pts);
        Some(Buffer::new(MemoryHandle::with_len(slot, len), meta))
    }
}

impl Demuxer for AvSource {
    fn demux(&mut self, _buffer: Buffer) -> Result<RoutedOutput> {
        Ok(RoutedOutput::new())
    }

    fn produce(&mut self) -> Result<DemuxerProduce> {
        if self.video >= self.frames {
            if !self.looping {
                return Ok(DemuxerProduce::Eos);
            }
            self.video = 0;
            self.audio = 0;
            self.staged = Some(parallax::event::SegmentEvent::new_time(
                ClockTime::ZERO,
                None,
            ));
        }
        // Interleave in timestamp order, audio slightly ahead like a real
        // container's read-ahead.
        let video_at = self.video * self.frame_ns;
        let audio_at = self.audio * AUDIO_NS;
        let (pad, len, pts) = if audio_at <= video_at {
            self.audio += 1;
            (AUDIO_PAD, AUDIO_BYTES, audio_at)
        } else {
            self.video += 1;
            (VIDEO_PAD, 1, video_at)
        };
        match self.emit(len, pts) {
            Some(buffer) => Ok(DemuxerProduce::Routed(RoutedOutput::single(pad, buffer))),
            None => Ok(DemuxerProduce::WouldBlock),
        }
    }

    fn outputs(&self) -> &[(PadId, parallax::format::Caps)] {
        &self.outputs
    }

    fn on_pad_added(&mut self, _callback: parallax::element::PadAddedCallback) {}

    fn pad_name(&self, pad: PadId) -> String {
        match pad.0 {
            0 => "video".into(),
            _ => "audio".into(),
        }
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn take_downstream_event(&mut self) -> Option<Event> {
        self.staged.take().map(Event::Segment)
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        let Event::Seek(seek) = event else {
            return EventResult::NotHandled;
        };
        if seek.start.seek_type != SeekType::Set {
            return EventResult::NotHandled;
        }
        let target = seek.start.position.max(0) as u64;
        // Video snaps back to its GOP's keyframe...
        let frame = target / self.frame_ns;
        let keyframe = frame - (frame % GOP_FRAMES);
        self.video = keyframe;
        // ...and audio lands at or before it, which is what leaves the first
        // post-seek audio buffer straddling the segment start.
        self.audio = (keyframe * self.frame_ns) / AUDIO_NS;
        if self.audio > 0 {
            self.audio -= 1;
        }
        EventResult::handled_at((keyframe * self.frame_ns) as i64)
    }
}

/// How long the video branch takes to restart after a flush — a stand-in for
/// rebuilding a decoder and decoding a keyframe. This asymmetry is what made
/// the post-seek offset arbitrary, so every test here runs with it.
const VIDEO_RESTART: Duration = Duration::from_millis(150);

/// Wire the two-pad source to the two stand-in sinks, with the video branch
/// slow to restart the way a real one is.
fn build(source: AvSource) -> (Pipeline, LogHandle, LogHandle) {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_demuxer("demux", source);

    let audio = ClockAudioSink::new(SAMPLE_RATE, 2, 2);
    let audio_log = audio.log();
    let video = PacedVideoSink::new();
    let video_log = video.log();

    let decode = pipeline.add_async_transform("decode", SlowRestart::new(VIDEO_RESTART));
    let vsink = pipeline.add_async_sink("display", video);
    let asink = pipeline.add_async_sink("speaker", audio);
    // Branch depths matching `parallax-player`'s, and for its reason: the
    // demuxer emits in timestamp order from one task, so if the video branch
    // back-pressures at exactly the rate the audio branch is drained, audio
    // starves, the master clock freezes, and the paced video sink waits on a
    // clock nothing will advance. The queue is the link channel here.
    pipeline
        .link_pads_full(src, "video", decode, "sink", LinkPolicy::Block, Some(24))
        .unwrap();
    pipeline.link(decode, vsink).unwrap();
    pipeline
        .link_pads_full(src, "audio", asink, "sink", LinkPolicy::Block, Some(96))
        .unwrap();
    (pipeline, audio_log, video_log)
}

/// The same graph, but ending in the GUI-facing sink instead of the paced
/// stand-in.
///
/// `AppVideoSink` does not decide when a frame is shown — that is the host's
/// job, which is the whole point of it — so the drain task below *is* the
/// host, doing what a redraw loop does. It logs into the same `Presented`
/// records the stand-in produces, so both are measured with one yardstick.
/// `shed_after: None` means the production default, which is what the
/// occlusion tests must run at — a shorter deadline than the frame interval
/// hides the very behaviour they are about.
fn build_with_app_sink(
    source: AvSource,
    shed_after: Option<Duration>,
) -> (Pipeline, LogHandle, Arc<FrameMailbox>) {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_demuxer("demux", source);

    let audio = ClockAudioSink::new(SAMPLE_RATE, 2, 2);
    let audio_log = audio.log();
    let mut video = AppVideoSink::new().with_sync(true);
    if let Some(shed_after) = shed_after {
        video = video.with_shed_after(shed_after);
    }
    let mailbox = video.mailbox();

    let decode = pipeline.add_async_transform("decode", SlowRestart::new(VIDEO_RESTART));
    let vsink = pipeline.add_async_sink("display", video);
    let asink = pipeline.add_async_sink("speaker", audio);
    pipeline
        .link_pads_full(src, "video", decode, "sink", LinkPolicy::Block, Some(24))
        .unwrap();
    // The player's decode-ahead window, spelled out rather than inherited from
    // `ExecutorConfig::channel_capacity` (16): how many seconds of stream the
    // video branch holds is exactly what decides whether a starving audio
    // branch shows up inside a test's measuring window.
    pipeline
        .link_pads_full(decode, "src", vsink, "sink", LinkPolicy::Block, Some(4))
        .unwrap();
    pipeline
        .link_pads_full(src, "audio", asink, "sink", LinkPolicy::Block, Some(96))
        .unwrap();
    (pipeline, audio_log, mailbox)
}

/// A host's redraw loop, at roughly 200 Hz.
fn spawn_host(mailbox: Arc<FrameMailbox>) -> (LogHandle, tokio::task::JoinHandle<()>) {
    let log: LogHandle = Arc::new(Mutex::new(sync::Log::default()));
    let into = LogHandle::clone(&log);
    let task = tokio::spawn(async move {
        loop {
            if let Some(clock) = mailbox.clock() {
                let now = clock.now();
                while let Some(paced) = mailbox.take_due(now, Duration::from_millis(40)) {
                    let at = clock.clock_time();
                    record(&into, |l| {
                        l.presented.push(Presented {
                            pts: paced.frame.data.metadata().pts,
                            running_time: paced.target,
                            at,
                        });
                    });
                }
            }
            if mailbox.end_reason().is_some() && !mailbox.has_frame() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    (log, task)
}

/// The same redraw loop, with a window in which the host draws nothing at all.
///
/// This is what an occluded or minimised window looks like from the pipeline's
/// side: not a slow host, a stopped one. It neither takes frames nor reads the
/// clock while it is dark, exactly as a widget whose event loop is parked
/// inside its compositor's present call does not.
fn spawn_host_with_gap(
    mailbox: Arc<FrameMailbox>,
    draw_for: Duration,
    dark_for: Duration,
) -> (LogHandle, tokio::task::JoinHandle<()>) {
    let log: LogHandle = Arc::new(Mutex::new(sync::Log::default()));
    let into = LogHandle::clone(&log);
    let task = tokio::spawn(async move {
        let dark_from = std::time::Instant::now() + draw_for;
        let dark_until = dark_from + dark_for;
        loop {
            let now = std::time::Instant::now();
            if !(dark_from..dark_until).contains(&now)
                && let Some(clock) = mailbox.clock()
            {
                let running = clock.now();
                while let Some(paced) = mailbox.take_due(running, Duration::from_millis(40)) {
                    let at = clock.clock_time();
                    record(&into, |l| {
                        l.presented.push(Presented {
                            pts: paced.frame.data.metadata().pts,
                            running_time: paced.target,
                            at,
                        });
                    });
                }
            }
            if mailbox.end_reason().is_some() && !mailbox.has_frame() {
                return;
            }
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    });
    (log, task)
}

/// How much stream time a log covers, in nanoseconds.
///
/// `ClockAudioSink` sleeps for every buffer it plays, so measuring its running
/// time against wall time is measuring whether the pipeline is keeping up.
fn advanced_ns(from: &sync::Log, to: &sync::Log) -> u64 {
    let start = from.presented.last().expect("something played first");
    let end = to.presented.last().expect("something played after");
    end.running_time
        .nanos()
        .saturating_sub(start.running_time.nanos())
}

fn snapshot(log: &LogHandle) -> sync::Log {
    let l = log.lock().unwrap_or_else(|e| e.into_inner());
    sync::Log {
        presented: l.presented.clone(),
        clipped: l.clipped,
        late: l.late,
        segments: l.segments.clone(),
        generations: l.generations.clone(),
        stalled: l.stalled,
        superseded: l.superseded,
    }
}

/// The offset between what the two sinks think running-time zero is, in ms,
/// measured where it matters: once both branches have settled.
///
/// The first frame after a seek is deliberately shown however late it is —
/// it is the frame the user is waiting for — and the frames behind it are
/// dropped while the branch catches up. What must converge is everything
/// after that, and what must *not* happen is the offset becoming permanent.
fn settled_drift_ms(audio: &sync::Log, video: &sync::Log, after: ClockTime) -> f64 {
    let a = audio.after(after);
    let v = video.after(after);
    let last_audio = *a.last().expect("audio played after the mark");
    let settled: Vec<Presented> = v.into_iter().skip(1).collect();
    let last_video = *settled.last().expect("video settled after the mark");
    (last_audio.implied_base_ns() - last_video.implied_base_ns()) as f64 / 1e6
}

#[tokio::test(flavor = "multi_thread")]
async fn audio_and_video_agree_on_the_origin_from_the_start() {
    let (mut pipeline, audio_log, video_log) = build(AvSource::new(10));
    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();
    // Read the logs before the run winds down, for the same reason: teardown
    // detaches the clock.
    tokio::time::sleep(Duration::from_millis(900)).await;
    let audio = snapshot(&audio_log);
    let video = snapshot(&video_log);
    handle.stop();
    let _ = handle.wait().await;

    assert!(!audio.presented.is_empty(), "audio played");
    assert!(!video.presented.is_empty(), "video presented");

    // The video branch starts late — a decoder has to come up — so the first
    // frames are dropped while it catches up. What matters is that it catches
    // up *to the audio's* origin rather than adopting its own late arrival as
    // running-time zero, which is what the private anchor did.
    let drift = settled_drift_ms(&audio, &video, ClockTime::ZERO);
    assert!(
        drift.abs() < TOLERANCE_MS,
        "video settled {drift:.1} ms off the audio origin"
    );

    // And no frame is ever shown *early*: a frame presented before its
    // running time would mean the origin moved backwards under it.
    let base = audio.presented.last().unwrap().implied_base_ns();
    for frame in &video.presented {
        let early = (base - frame.implied_base_ns()) as f64 / 1e6;
        assert!(
            early < TOLERANCE_MS,
            "frame at pts {:?} was shown {early:.1} ms early",
            frame.pts
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_seek_leaves_the_two_branches_on_one_timeline() {
    // 60 frames = 6 s of stream. The seek asks for 3.3 s, which is not on a
    // keyframe, so video snaps back to 3.0 s and audio lands just before it
    // — the shape a real MP4 seek produces.
    let (mut pipeline, audio_log, video_log) = build(AvSource::new(60));
    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();

    tokio::time::sleep(Duration::from_millis(300)).await;
    assert!(
        handle.seek_time(ClockTime::from_millis(3_300)).await,
        "the seek was dispatched"
    );
    tokio::time::sleep(Duration::from_millis(900)).await;
    // Snapshot while it is still playing: `stop()` deliberately detaches the
    // clock so paced branches can drain their tail, and readings from that
    // window measure teardown rather than playback.
    let audio = snapshot(&audio_log);
    let video = snapshot(&video_log);
    handle.stop();
    let _ = handle.wait().await;

    // The seek's segment starts at the video keyframe, 2.5 s.
    let seg = audio.segments.last().expect("a post-seek segment");
    assert_eq!(seg.start, 3_000_000_000, "segment starts at the keyframe");

    // Audio landed *before* that keyframe, so the straddling buffer is
    // trimmed instead of played. Nothing clipped it before this work.
    assert!(
        audio.clipped > 0,
        "the pre-keyframe audio pre-roll must be clipped, not played"
    );

    // Both branches agree on where running-time zero sits after the seek.
    let mark = audio
        .presented
        .iter()
        .find(|p| p.pts.nanos() >= 3_000_000_000)
        .expect("audio resumed after the seek")
        .at;
    let drift = settled_drift_ms(&audio, &video, mark);
    assert!(
        drift.abs() < TOLERANCE_MS,
        "A/V drifted {drift:.1} ms across the seek"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn repeated_seeks_do_not_accumulate_drift() {
    // The failure this guards is not one bad seek but a *fresh* offset per
    // seek — holding the arrow key used to re-roll it every press, forward
    // and backward alike.
    let (mut pipeline, audio_log, video_log) = build(AvSource::new(300));
    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();
    tokio::time::sleep(Duration::from_millis(400)).await;

    let mut worst: f64 = 0.0;
    let mut measured = 0;
    for target in [4_000u64, 12_000, 2_000, 8_000] {
        // Everything already played belongs to the old position; the first
        // audio entry after this index is where the new one starts.
        let before = snapshot(&audio_log).presented.len();
        handle.seek_time(ClockTime::from_millis(target)).await;
        tokio::time::sleep(Duration::from_millis(1_200)).await;

        let audio = snapshot(&audio_log);
        let video = snapshot(&video_log);
        let mark = audio.presented[before].at;
        if video.after(mark).len() < 3 {
            continue;
        }
        measured += 1;
        worst = worst.max(settled_drift_ms(&audio, &video, mark).abs());
    }
    handle.stop();
    let _ = handle.wait().await;

    assert!(
        measured >= 3,
        "only {measured} seeks produced enough frames to judge"
    );
    assert!(
        worst < TOLERANCE_MS,
        "worst A/V drift across four seeks: {worst:.1} ms"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_silent_loop_rewind_keeps_running_time_monotonic() {
    // `with_loop` repositions without any seek, so nothing but the staged
    // Segment re-establishes the timeline. Without it lap two's PTS restart
    // below the segment start, every frame reads as hopelessly late, and the
    // picture freezes for good — which is what absolute pacing would have
    // done to `parallax-player --loop`.
    let (mut pipeline, audio_log, video_log) = build(AvSource::new(10).looping());
    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();
    tokio::time::sleep(Duration::from_millis(2_500)).await;
    handle.stop();
    let _ = handle.wait().await;

    let audio = snapshot(&audio_log);
    let video = snapshot(&video_log);

    assert!(
        audio.segments.len() > 1,
        "the rewind emits a fresh Segment on every pad"
    );
    let lap2 = audio.segments.last().unwrap();
    assert_eq!(lap2.start, 0, "lap two restarts at PTS zero");
    assert!(
        lap2.base > 0,
        "…but running time carries on: base {}",
        lap2.base
    );

    // Running time never goes backwards, on either branch.
    for log in [&audio, &video] {
        let mut previous = ClockTime::ZERO;
        for entry in &log.presented {
            assert!(
                entry.running_time >= previous,
                "running time went backwards at pts {:?}",
                entry.pts
            );
            previous = entry.running_time;
        }
    }
    // And lap two is played, not clipped away as out of segment.
    assert!(
        video.presented.len() > 10,
        "lap two must play: only {} frames presented",
        video.presented.len()
    );
}

// ===========================================================================
// The same claim, through the real demuxer and the real decoders
// ===========================================================================

/// End to end on `tiny_h264_aac.mkv` — 2 s, 20 frames at 10 fps with a
/// keyframe every 5, plus AAC — with only the two devices stood in for.
///
/// Gated on `mkv-demux,h264,audio-aac`, so it runs in CI's `codec-libs` job
/// (`just check-media-full`) — the `media` combo carries no H.264.
///
/// The synthetic source above models the *shape* of the problem; this one
/// exercises the parts that actually restart on a seek: `MkvDemux`'s cue seek
/// and per-track skip, `H264Decoder` rebuilding openh264 on `FlushStart` and
/// waiting for an IDR, and `AudioDecoderElement` resetting its decoder and
/// its running timestamp. Those are what make the two branches restart at
/// different instants in the first place.
#[cfg(all(feature = "mkv-demux", feature = "h264", feature = "audio-aac"))]
#[tokio::test(flavor = "multi_thread")]
async fn a_real_mkv_keeps_lip_sync_across_a_seek() {
    use parallax::elements::codec::AudioDecoderElement;
    use parallax::elements::demux::MkvDemux;
    use parallax::elements::{AacDecoder, H264Decoder};

    const H264_AAC_MKV: &[u8] = include_bytes!("fixtures/tiny_h264_aac.mkv");

    let demux = MkvDemux::new(std::io::Cursor::new(H264_AAC_MKV.to_vec()))
        .expect("the fixture is readable Matroska");
    let audio_track = demux.audio_track().expect("the fixture has audio");
    let info = audio_track.audio_info.as_ref().expect("audio info");
    let (rate, channels) = (info.sample_rate, info.channels as u32);
    let asc = info
        .codec_private
        .clone()
        .expect("AAC CodecPrivate is the AudioSpecificConfig");

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_demuxer("demux", demux);
    let dec = pipeline.add_filter("decode", H264Decoder::new().expect("openh264"));
    // AAC decodes to interleaved f32.
    let audio = ClockAudioSink::new(rate, channels, 4);
    let audio_log = audio.log();
    let video = PacedVideoSink::new();
    let video_log = video.log();
    let adec = pipeline.add_transform(
        "audiodec",
        AudioDecoderElement::new(AacDecoder::from_asc(&asc).expect("ASC")),
    );
    let vsink = pipeline.add_async_sink("display", video);
    let asink = pipeline.add_async_sink("speaker", audio);

    // Shallower than the player's 24/96 on purpose: the fixture is two
    // seconds long and the player's depths would swallow all of it, so the
    // demuxer would reach EOS before the seek was ever issued. Audio still
    // fills first (8 AAC frames ≈ 170 ms against 4 video frames ≈ 400 ms),
    // which is what keeps the video branch from starving it.
    pipeline
        .link_pads_full(src, "video", dec, "sink", LinkPolicy::Block, Some(4))
        .unwrap();
    pipeline.link(dec, vsink).unwrap();
    pipeline
        .link_pads_full(src, "audio", adec, "sink", LinkPolicy::Block, Some(8))
        .unwrap();
    pipeline.link(adec, asink).unwrap();

    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();

    tokio::time::sleep(Duration::from_millis(400)).await;
    let before = snapshot(&audio_log).presented.len();
    // 300 ms is not a keyframe; MKV's default snap is forward, so video lands
    // on the 500 ms cue — and since #265 audio's skip floor lands there too.
    assert!(handle.seek_time(ClockTime::from_millis(300)).await);
    tokio::time::sleep(Duration::from_millis(900)).await;

    let audio = snapshot(&audio_log);
    let video = snapshot(&video_log);
    handle.stop();
    let _ = handle.wait().await;

    assert!(
        audio.presented.len() > before,
        "audio resumed after the seek"
    );

    // MKV forward-snaps, and the landing is the cue — a real video keyframe —
    // reported up front and shared by both branches. It used to be the
    // *requested* time on the audio pad while video waited a GOP for its
    // keyframe.
    let landing = audio.segments.last().expect("a post-seek segment").start;
    assert_eq!(landing, 500_000_000, "landed on the 500 ms cue keyframe");
    assert_eq!(
        video.segments.last().map(|s| s.start),
        Some(landing),
        "both pads carry the same post-seek segment"
    );

    let mark = audio.presented[before].at;
    let resumed: Vec<_> = video
        .presented
        .iter()
        .filter(|p| p.pts.nanos() as i64 >= landing)
        .collect();
    assert!(
        resumed.len() >= 3,
        "video resumed after the seek: {} frames",
        resumed.len()
    );

    let drift = settled_drift_ms(&audio, &video, mark);
    assert!(
        drift.abs() < TOLERANCE_MS,
        "real MKV drifted {drift:.1} ms across the seek"
    );
}

/// Reproduction rig for the reported hang: a looping source hammered with
/// seeks, so rewinds and flushes interleave.
#[tokio::test(flavor = "multi_thread")]
async fn looping_under_repeated_seeks_does_not_wedge() {
    let (mut pipeline, audio_log, video_log) = build(AvSource::new(20).looping());
    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();

    // Seek across the lap boundary repeatedly, at intervals that do not line
    // up with it.
    for round in 0..12u64 {
        tokio::time::sleep(Duration::from_millis(130)).await;
        let target = (round * 370) % 1_900;
        handle.seek_time(ClockTime::from_millis(target)).await;
    }
    // Recovery, not throughput: under a seek every 130 ms most frames are
    // legitimately superseded before their presentation time.
    let audio_mark = snapshot(&audio_log).presented.len();
    let video_mark = snapshot(&video_log).presented.len();
    let recovered = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if snapshot(&audio_log).presented.len() > audio_mark + 4
                && snapshot(&video_log).presented.len() > video_mark + 2
            {
                return;
            }
        }
    })
    .await;

    let stopped = tokio::time::timeout(Duration::from_secs(10), async {
        handle.stop();
        let _ = handle.wait().await;
    })
    .await;
    assert!(recovered.is_ok(), "wedged after the seek storm");
    assert!(stopped.is_ok(), "teardown wedged");
}

/// The reported hang, on the real stack: `--loop` plus repeated seeks, with
/// the player's own link depths.
#[cfg(all(feature = "mkv-demux", feature = "h264", feature = "audio-aac"))]
#[tokio::test(flavor = "multi_thread")]
async fn a_looping_real_pipeline_survives_repeated_seeks() {
    use parallax::elements::codec::AudioDecoderElement;
    use parallax::elements::demux::MkvDemux;
    use parallax::elements::{AacDecoder, H264Decoder};

    const H264_AAC_MKV: &[u8] = include_bytes!("fixtures/tiny_h264_aac.mkv");

    let demux = MkvDemux::new(std::io::Cursor::new(H264_AAC_MKV.to_vec()))
        .expect("readable Matroska")
        .with_loop(true);
    let info = demux
        .audio_track()
        .and_then(|t| t.audio_info.clone())
        .expect("audio info");
    let asc = info.codec_private.clone().expect("ASC");

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_demuxer("demux", demux);
    let dec = pipeline.add_filter("decode", H264Decoder::new().expect("openh264"));
    let audio = ClockAudioSink::new(info.sample_rate, info.channels as u32, 4);
    let audio_log = audio.log();
    let video = PacedVideoSink::new();
    let video_log = video.log();
    let adec = pipeline.add_transform(
        "audiodec",
        AudioDecoderElement::new(AacDecoder::from_asc(&asc).expect("ASC")),
    );
    let vsink = pipeline.add_async_sink("display", video);
    let asink = pipeline.add_async_sink("speaker", audio);

    // The player's own depths.
    pipeline
        .link_pads_full(src, "video", dec, "sink", LinkPolicy::Block, Some(24))
        .unwrap();
    pipeline.link(dec, vsink).unwrap();
    pipeline
        .link_pads_full(src, "audio", adec, "sink", LinkPolicy::Block, Some(96))
        .unwrap();
    pipeline.link(adec, asink).unwrap();

    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();

    let _ = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::ERROR)
        .try_init();
    for round in 0..10u64 {
        tokio::time::sleep(Duration::from_millis(150)).await;
        handle
            .seek_time(ClockTime::from_millis((round * 430) % 1_900))
            .await;
    }

    // The claim is recovery, not throughput: under a seek every 150 ms most
    // frames are legitimately superseded before their presentation time, so
    // what matters is that both branches are moving again once the storm
    // stops — and that they get there without being nudged.
    let audio_mark = snapshot(&audio_log).presented.len();
    let video_mark = snapshot(&video_log).presented.len();
    let recovered = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            tokio::time::sleep(Duration::from_millis(50)).await;
            if snapshot(&audio_log).presented.len() > audio_mark + 4
                && snapshot(&video_log).presented.len() > video_mark + 2
            {
                return;
            }
        }
    })
    .await;

    let audio = snapshot(&audio_log);
    let video = snapshot(&video_log);
    let stopped = tokio::time::timeout(Duration::from_secs(10), async {
        handle.stop();
        let _ = handle.wait().await;
    })
    .await;

    assert!(
        recovered.is_ok(),
        "wedged after the seek storm: audio {} -> {}, video {} -> {} \
         (video late={} superseded={} stalled={})",
        audio_mark,
        audio.presented.len(),
        video_mark,
        video.presented.len(),
        video.late,
        video.superseded,
        video.stalled,
    );
    assert!(stopped.is_ok(), "teardown wedged");
}

/// The GUI-facing sink keeps A/V together, which is the whole reason it can
/// replace the one that owns a window.
///
/// `AppVideoSink` hands frames to a host rather than presenting them itself,
/// so the anchor handshake, the segment mapping and the flush generations all
/// have to work across that boundary. This runs the identical graph the paced
/// stand-in runs and holds it to the identical bound.
#[tokio::test(flavor = "multi_thread")]
async fn the_app_sink_settles_on_the_audio_origin_too() {
    let (mut pipeline, audio_log, mailbox) = build_with_app_sink(AvSource::new(10), None);
    let (video_log, host) = spawn_host(Arc::clone(&mailbox));

    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();
    tokio::time::sleep(Duration::from_millis(900)).await;
    let audio = snapshot(&audio_log);
    let video = snapshot(&video_log);
    handle.stop();
    let _ = handle.wait().await;
    let _ = tokio::time::timeout(Duration::from_secs(2), host).await;

    assert!(!audio.presented.is_empty(), "audio played");
    assert!(!video.presented.is_empty(), "video presented");

    let drift = settled_drift_ms(&audio, &video, ClockTime::ZERO);
    assert!(
        drift.abs() < TOLERANCE_MS,
        "video settled {drift:.1} ms off the audio origin"
    );

    // And the host is told the run ended, rather than being left waiting on
    // a doorbell nobody is going to ring.
    assert!(
        mailbox.end_reason().is_some(),
        "the host learns the outcome"
    );
}

/// A host that stops drawing must not *slow the pipeline down*.
///
/// Not merely "must not stop it", which is what the first version of this
/// test asserted — a count of audio buffers over 1.2 s, with the shed deadline
/// overridden to 60 ms so that shedding roughly kept pace with a 10 fps source.
/// A pipeline crawling at two frames a second passes that. It is the shape a
/// real one takes when the window goes behind another: `enqueue` restarted its
/// deadline for every frame, so the video branch consumed at 1/`shed_after`,
/// filled its links, backed the demuxer up, starved the audio branch and left
/// the master clock creeping. Every rate in that loop is defined by every other
/// one, so it settles wherever it lands and stays.
///
/// What an occluded window must cost is the picture and nothing else. So the
/// assertion is a *rate*, at the production deadline, measured over a window
/// that only opens once the branch depths have been used up.
#[tokio::test(flavor = "multi_thread")]
async fn a_host_that_stops_drawing_does_not_slow_the_pipeline_down() {
    // 25 fps, so the video branch's 24 + 4 + 3 frames are ~1.2 s of stream and
    // the audio link's 96 buffers are ~1 s — both spent well inside the settle
    // window below.
    let (mut pipeline, audio_log, mailbox) =
        build_with_app_sink(AvSource::new(400).with_frame_ns(40_000_000), None);
    // No host at all: nothing ever drains the mailbox.
    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();

    tokio::time::sleep(Duration::from_millis(2_000)).await;
    let mark = snapshot(&audio_log);
    let marked_at = std::time::Instant::now();
    let before = mailbox.stats().dropped_full;
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let audio = snapshot(&audio_log);
    let elapsed = marked_at.elapsed();
    let stats = mailbox.stats();
    handle.stop();
    let _ = handle.wait().await;

    let rate = advanced_ns(&mark, &audio) as f64 / elapsed.as_nanos() as f64;
    assert!(
        rate > 0.6,
        "audio ran at {rate:.2}x real time with nobody drawing — \
         the video branch was still back-pressuring the demuxer"
    );

    // And the video branch shed at the *source's* rate, not the shed timer's:
    // ~25 a second over the measured window, against 2 a second before.
    let shed = stats.dropped_full - before;
    assert!(
        shed > 20,
        "the video branch shed only {shed} frames in {elapsed:?} — it was still waiting"
    );
    assert!(
        stats.queued <= mailbox.capacity(),
        "the ring stayed bounded at {} frames",
        stats.queued
    );
}

/// And a host that comes back gets a picture again, in sync.
///
/// The other half of the same failure: the pipeline that crawled while the
/// window was hidden used to stay crawling once it came back, because the
/// equilibrium it had settled into was self-consistent and nothing pulled it
/// out. Recovering has to be automatic and quick.
#[tokio::test(flavor = "multi_thread")]
async fn a_host_that_comes_back_gets_a_picture_in_sync() {
    let (mut pipeline, audio_log, mailbox) =
        build_with_app_sink(AvSource::new(400).with_frame_ns(40_000_000), None);
    // Draw briefly, go dark for longer than the branch depths can absorb, come
    // back. 2.5 s of dark against ~1.2 s of video and ~1 s of audio buffering.
    let (video_log, host) = spawn_host_with_gap(
        Arc::clone(&mailbox),
        Duration::from_millis(700),
        Duration::from_millis(2_500),
    );

    let executor = Executor::with_config(ExecutorConfig::default());
    let handle = executor.start(&mut pipeline).unwrap();

    // Through the dark window, sampling the audio at its edges.
    tokio::time::sleep(Duration::from_millis(1_400)).await;
    let dark_mark = snapshot(&audio_log);
    let dark_at = std::time::Instant::now();
    tokio::time::sleep(Duration::from_millis(1_700)).await;
    let dark_end = snapshot(&audio_log);
    let dark_elapsed = dark_at.elapsed();

    // Back on screen. Give the branch time to re-fill and settle before
    // judging the drift: the first frames after the host returns discharge
    // whatever lead the un-back-pressured demuxer built up.
    tokio::time::sleep(Duration::from_millis(1_500)).await;
    let resumed = snapshot(&video_log);
    let resumed_at = match resumed.presented.last() {
        Some(last) => last.at,
        // The state this whole change exists to make impossible: the host is
        // drawing again and the pipeline is still wedged where the stall left
        // it, with nothing ever becoming due.
        None => panic!("no frame was ever presented — the picture never came back"),
    };
    tokio::time::sleep(Duration::from_millis(1_200)).await;

    let audio = snapshot(&audio_log);
    let video = snapshot(&video_log);
    handle.stop();
    let _ = handle.wait().await;
    let _ = tokio::time::timeout(Duration::from_secs(2), host).await;

    let dark_rate = advanced_ns(&dark_mark, &dark_end) as f64 / dark_elapsed.as_nanos() as f64;
    assert!(
        dark_rate > 0.6,
        "audio ran at {dark_rate:.2}x real time while the host was dark"
    );

    let after_resume = video.after(resumed_at).len();
    assert!(
        after_resume > 15,
        "only {after_resume} frames were presented in the 1.2 s after the picture \
         came back — the pipeline was still crawling"
    );

    let drift = settled_drift_ms(&audio, &video, resumed_at);
    assert!(
        drift.abs() < TOLERANCE_MS,
        "video settled {drift:.1} ms off the audio origin after the host returned"
    );
}
