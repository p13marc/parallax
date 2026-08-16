//! #163 phase B, end to end: `filesrc ! tsdemux ! appsink` seeks in TIME.
//!
//! Nothing in this pipeline can seek in time. `FileSrc` seeks in bytes and
//! `TsDemuxElement` does not own the reader — the seek only works because the
//! demuxer *translates* it, which is the whole point of the phase.
//!
//! The fixture is muxed here rather than checked in so its shape is visible:
//! **`psi_interval` must be non-zero**. The default (0) writes PAT/PMT once at
//! the head, and a seek resets the parser — after which a head-only stream
//! never delivers another frame, because the tables that describe it are
//! thousands of packets behind the read cursor.

#![cfg(feature = "mpeg-ts")]

use std::time::Duration;

use parallax::clock::ClockTime;
use parallax::elements::{
    AppSink, FileSrc, Pulled, TsDemuxElement, TsMux, TsMuxConfig, TsMuxStreamType, TsMuxTrack,
};
use parallax::event::SegmentFormat;
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::{Executor, Pipeline};
use tempfile::NamedTempFile;

const VIDEO_PID: u16 = 0x100;
const FPS: u64 = 25;
const FRAMES: u64 = 500; // 20 seconds

/// Mux a 10-second single-track TS whose PSI repeats often enough to survive
/// a mid-stream parser reset.
fn fixture() -> NamedTempFile {
    let config = TsMuxConfig::new()
        .add_track(TsMuxTrack::new(VIDEO_PID, TsMuxStreamType::H264).video())
        // Every 50 packets: frequent enough that a post-seek parser sees a
        // PAT/PMT within a few reads.
        .psi_interval(50)
        // Denser than the 100 ms conformance floor, so 10 seconds of stream
        // gives the byte index plenty of anchors.
        .pcr_interval_ms(40);
    let mut mux = TsMux::new(config);

    // The payload is inert: this test is about where the demuxer lands, not
    // about decoding. Each frame is distinctly sized so nothing accidentally
    // aliases.
    let mut out = Vec::new();
    for frame in 0..FRAMES {
        let pts = ClockTime::from_nanos(frame * 1_000_000_000 / FPS);
        // 4 KB per frame, so the whole fixture (~2 MB) cannot fit in the
        // pipeline's channel buffers. With a smaller one the source reads the
        // file to EOS and exits before the seek ever reaches it — the seek
        // then has nothing to act on, and the test passes or fails on I/O
        // timing rather than on seeking.
        let payload = vec![(frame % 251) as u8; 4096];
        out.extend(
            mux.write_pes(VIDEO_PID, &payload, Some(pts), Some(pts))
                .unwrap(),
        );
    }

    let mut file = NamedTempFile::new().unwrap();
    std::io::Write::write_all(file.as_file_mut(), &out).unwrap();
    std::io::Write::flush(file.as_file_mut()).unwrap();
    file
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fed_ts_demuxer_seeks_in_time() {
    let file = fixture();
    let size = file.as_file().metadata().unwrap().len();
    assert!(
        size > 1_500_000,
        "fixture is {size} bytes, too small to seek in"
    );

    let mut pipeline = Pipeline::new();
    // Small reads: 100 packets each, so the demuxer is pushed often and the
    // seek lands within a read or two of the estimate.
    let src = pipeline.add_source("src", FileSrc::new(file.path()).with_chunk_size(100 * 188));
    let demux = pipeline.add_demuxer("tsdemux", TsDemuxElement::new());
    let sink = AppSink::with_max_buffers(4);
    let sink_handle = sink.handle();
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "video", snk, "sink").unwrap();

    // Before start, with only FileSrc's answer to go on, the graph already
    // reports TIME — the demuxer's declared translation replaced BYTES.
    let pre = pipeline.query_seekable();
    assert!(pre.seekable);
    assert_eq!(pre.format, SegmentFormat::Time);

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    // Pull in the background: the sink must keep draining or the graph
    // back-pressures and the seek never reaches the demuxer.
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let puller = tokio::spawn(async move {
        loop {
            match sink_handle.pull_buffer().await {
                Pulled::Buffer(b) => {
                    if tx.send(b.metadata().pts).is_err() {
                        break;
                    }
                    // Paced on purpose. Unthrottled, the whole 10-second
                    // fixture is read and demuxed in about ten milliseconds,
                    // and the seek races end-of-stream instead of landing
                    // mid-play. A bounded AppSink with Block policy turns
                    // this delay into back-pressure on the source.
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                Pulled::Ended(_) => break,
                Pulled::Empty | Pulled::Flushing => tokio::task::yield_now().await,
            }
        }
    });

    // Let a couple of seconds of stream go past so the index has anchors.
    let mut seen = 0;
    while seen < 20 {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(_)) => seen += 1,
            Ok(None) => panic!("stream ended before the seek could be issued"),
            Err(_) => panic!("no frames within 5s; got {seen}"),
        }
    }

    let target = ClockTime::from_secs(7);
    assert!(handle.seek_time(target).await, "the seek was dispatched");

    // Drain until a frame lands at or after the target. Pre-seek frames
    // already in flight are stamped with the old epoch and dropped by the
    // executor, but the ones already pulled are still in `rx`.
    let mut landed = None;
    let mut after_seek = 0u32;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(pts)) if pts >= ClockTime::from_secs(6) => {
                landed = Some(pts);
                break;
            }
            Ok(Some(_)) => after_seek += 1,
            Ok(None) => break,
            Err(_) => break,
        }
    }
    let landed = landed.expect("no frame at or after the seek target arrived");
    assert!(
        landed >= ClockTime::from_secs(6) && landed <= ClockTime::from_secs(9),
        "landed at {landed}, which is not near the 7s target"
    );
    // Playing through from 0.8s to 6s would have delivered ~130 frames. The
    // pre-seek buffers still in flight are epoch-dropped, so only a handful
    // of already-pulled ones can precede the landing.
    assert!(
        after_seek < 30,
        "{after_seek} frames before the landing: this played through rather than seeking"
    );

    handle.stop();
    let _ = puller.await;
    handle.wait().await.unwrap();

    // The completion is reported in TIME — the format the application asked
    // in — carrying the position actually reached, not the byte estimate.
    let mut time_done = None;
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone {
            format, position, ..
        } = msg.kind
            && format == SegmentFormat::Time
        {
            time_done = Some(position);
        }
    }
    let position = time_done.expect("a TIME SeekDone was posted");
    let position = position.expect("the demuxer reported where it landed");
    assert!(
        (6_000_000_000..=9_000_000_000).contains(&position),
        "SeekDone reported {position} ns, which is not near the 7s target"
    );
}

/// #173: on a VBR stream the single-shot linear estimate is badly wrong, and
/// `SeekFlags::ACCURATE` iterates until the landing is honest.
///
/// The fixture's frame size grows linearly, so the byte curve is quadratic in
/// time: extrapolating from the early (small-frame) anchors undershoots a
/// far target by many seconds. Each refinement round observes the
/// mis-landing's PCRs, re-estimates from the local slope, and converges —
/// Newton's method by flush round trip.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_accurate_seek_on_vbr_iterates_to_the_target() {
    use parallax::event::{SeekEvent, SeekFlags};

    // 20 seconds at 25 fps, ramping 512 B → ~12.5 KB per frame (~3.3 MB).
    let config = TsMuxConfig::new()
        .add_track(TsMuxTrack::new(VIDEO_PID, TsMuxStreamType::H264).video())
        .psi_interval(50)
        .pcr_interval_ms(40);
    let mut mux = TsMux::new(config);
    let mut out = Vec::new();
    for frame in 0..FRAMES {
        let pts = ClockTime::from_nanos(frame * 1_000_000_000 / FPS);
        let payload = vec![(frame % 251) as u8; 512 + frame as usize * 24];
        out.extend(
            mux.write_pes(VIDEO_PID, &payload, Some(pts), Some(pts))
                .unwrap(),
        );
    }
    let mut file = NamedTempFile::new().unwrap();
    std::io::Write::write_all(file.as_file_mut(), &out).unwrap();
    std::io::Write::flush(file.as_file_mut()).unwrap();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::new(file.path()).with_chunk_size(100 * 188));
    let demux = pipeline.add_demuxer("tsdemux", TsDemuxElement::new());
    let sink = AppSink::with_max_buffers(4);
    let sink_handle = sink.handle();
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "video", snk, "sink").unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let puller = tokio::spawn(async move {
        loop {
            match sink_handle.pull_buffer().await {
                Pulled::Buffer(b) => {
                    if tx.send(b.metadata().pts).is_err() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                Pulled::Ended(_) => break,
                Pulled::Empty | Pulled::Flushing => tokio::task::yield_now().await,
            }
        }
    });

    // Anchor the index on the slow early stream, then aim deep into the
    // fast region.
    let mut seen = 0;
    while seen < 20 {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(_)) => seen += 1,
            Ok(None) => panic!("stream ended before the seek could be issued"),
            Err(_) => panic!("no frames within 5s; got {seen}"),
        }
    }

    let target = ClockTime::from_secs(15);
    let seek = SeekEvent::new_time(target)
        .with_flags(SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::ACCURATE);
    assert!(handle.seek(seek).await, "the seek was dispatched");

    // The intermediate rounds' mis-landed frames are epoch-stale and shed;
    // wait for a frame near the target.
    let mut landed = None;
    let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(pts)) if pts >= ClockTime::from_millis(14_400) => {
                landed = Some(pts);
                break;
            }
            Ok(Some(_)) => {}
            Ok(None) => break,
            Err(_) => break,
        }
    }
    let landed = landed.expect("no frame near the accurate target arrived");
    assert!(
        landed <= ClockTime::from_millis(15_700),
        "landed at {landed}, past the 15s target's tolerance"
    );

    handle.stop();
    let _ = puller.await;
    handle.wait().await.unwrap();

    // Exactly one TIME completion, reported near the target — the refinement
    // rounds held it back until the landing was worth reporting.
    let mut time_dones = Vec::new();
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone {
            format, position, ..
        } = msg.kind
            && format == SegmentFormat::Time
        {
            time_dones.push(position);
        }
    }
    assert_eq!(
        time_dones.len(),
        1,
        "refinement must complete once: {time_dones:?}"
    );
    let position = time_dones[0].expect("the demuxer reported where it landed");
    assert!(
        (14_400_000_000..=15_700_000_000).contains(&position),
        "SeekDone reported {position} ns — the ACCURATE landing missed 15s"
    );
}

/// A seek issued before the index has anything in it must not be answered
/// with a made-up offset. The demuxer refuses, the seek travels on to
/// `FileSrc`, which cannot service a TIME seek either — and the pipeline says
/// so instead of silently landing somewhere wrong.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_seek_before_any_pcr_is_refused_rather_than_guessed() {
    let file = fixture();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::new(file.path()));
    let demux = pipeline.add_demuxer("tsdemux", TsDemuxElement::new());
    let sink = AppSink::with_max_buffers(4).drop_on_full(true);
    let sink_handle = sink.handle();
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "video", snk, "sink").unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    // Immediately, before a single buffer has been demuxed.
    handle.seek_time(ClockTime::from_secs(5)).await;

    handle.stop();
    let _ = sink_handle;
    handle.wait().await.unwrap();

    // No TIME completion was posted: nothing claimed to have landed anywhere.
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone { format, .. } = msg.kind {
            assert_ne!(
                format,
                SegmentFormat::Time,
                "an unindexed seek must not report a TIME landing"
            );
        }
    }
}

/// #165: a fed demuxer's pads re-anchor with the translated seek's shape.
///
/// Two regressions pinned here. A flushing rate-2.0 TIME seek must come back
/// out of the demuxer's video pad as a Time segment carrying rate 2.0 — the
/// re-anchor used to be a hardcoded rate-1.0 `initial_segment_for`, so every
/// fed demuxer silently dropped trick-play rate. And a follow-up NON-flushing
/// seek must produce a new pad segment at all (pads only re-anchored on
/// FlushStop, which a queued seek never sends) with `base` = the running time
/// already consumed, taken at the in-band byte-Segment boundary so running
/// time stays monotonic across the queued handoff.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_fed_demuxer_reanchors_with_rate_and_base() {
    use std::sync::{Arc, Mutex};

    use parallax::event::{Event, SeekEvent, SeekFlags, SegmentEvent};
    use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};

    let file = fixture();
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FileSrc::new(file.path()).with_chunk_size(100 * 188));
    let demux = pipeline.add_demuxer("tsdemux", TsDemuxElement::new());
    let sink = AppSink::with_max_buffers(4);
    let sink_handle = sink.handle();
    let snk = pipeline.add_async_sink("sink", sink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "video", snk, "sink").unwrap();

    // The demuxer's own pad segments. The upstream byte segment is swallowed
    // before src-pad probes, so only Time re-anchors land here.
    let segments: Arc<Mutex<Vec<SegmentEvent>>> = Arc::new(Mutex::new(Vec::new()));
    let segments_probe = segments.clone();
    let _ = pipeline.add_probe(PadRef::src(demux), ProbeType::EVENT_DOWN, move |data| {
        if let ProbeData::Event(Event::Segment(seg)) = data
            && seg.format == SegmentFormat::Time
        {
            segments_probe.lock().unwrap().push(seg.clone());
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Pull in the background, paced (see a_fed_ts_demuxer_seeks_in_time).
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let puller = tokio::spawn(async move {
        loop {
            match sink_handle.pull_buffer().await {
                Pulled::Buffer(b) => {
                    if tx.send(b.metadata().pts).is_err() {
                        break;
                    }
                    tokio::time::sleep(Duration::from_millis(2)).await;
                }
                Pulled::Ended(_) => break,
                Pulled::Empty | Pulled::Flushing => tokio::task::yield_now().await,
            }
        }
    });

    let mut seen = 0;
    while seen < 20 {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(_)) => seen += 1,
            Ok(None) => panic!("stream ended before the first seek"),
            Err(_) => panic!("no frames within 5s; got {seen}"),
        }
    }

    // Flushing trick-play seek: 2x from 7s.
    let seek = SeekEvent::new_time(ClockTime::from_secs(7)).with_rate(2.0);
    assert!(handle.seek(seek).await, "the flushing seek was dispatched");

    // Drain until post-seek data lands.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut landed = false;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(pts)) if pts >= ClockTime::from_secs(6) => {
                landed = true;
                break;
            }
            Ok(Some(_)) => {}
            _ => break,
        }
    }
    assert!(landed, "no frame at or after the 7s target arrived");

    let trick_seg = segments
        .lock()
        .unwrap()
        .last()
        .cloned()
        .expect("a re-anchor segment after the flushing seek");
    assert_eq!(
        trick_seg.rate, 2.0,
        "the pad re-anchor carries the translated seek's rate: {trick_seg:?}"
    );
    assert_eq!(trick_seg.base, 0, "a flushing seek restarts running time");
    assert!(
        (6_000_000_000..=9_000_000_000).contains(&trick_seg.start),
        "re-anchor starts near the 7s target: {trick_seg:?}"
    );
    let n_before = segments.lock().unwrap().len();

    // Queued (non-flushing) seek back to 2s at rate 1.0. Nothing is flushed:
    // the source's byte segment rides FIFO behind the queued data and the
    // pad re-anchors exactly at that boundary.
    let queued = SeekEvent::new_time(ClockTime::from_secs(2)).with_flags(SeekFlags::KEY_UNIT); // no FLUSH
    assert!(handle.seek(queued).await, "the queued seek was dispatched");

    // Post-seek data shows up as a PTS drop back below 5s.
    let deadline = tokio::time::Instant::now() + Duration::from_secs(10);
    let mut dropped_back = false;
    while tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(5), rx.recv()).await {
            Ok(Some(pts)) if pts <= ClockTime::from_secs(5) => {
                dropped_back = true;
                break;
            }
            Ok(Some(_)) => {}
            _ => break,
        }
    }
    assert!(dropped_back, "no post-seek frame near 2s arrived");

    handle.stop();
    let _ = puller.await;
    handle.wait().await.unwrap();

    let segs = segments.lock().unwrap().clone();
    assert!(
        segs.len() > n_before,
        "a NON-flushing translated seek re-anchors the pad (got {} segments, had {n_before})",
        segs.len()
    );
    let queued_seg = &segs[n_before];
    assert_eq!(queued_seg.rate, 1.0, "rate restored: {queued_seg:?}");
    assert!(
        queued_seg.base > 0,
        "the queued re-anchor accumulates consumed running time: {queued_seg:?}"
    );
    assert!(
        (500_000_000..=4_000_000_000).contains(&queued_seg.start),
        "re-anchor starts near the 2s target: {queued_seg:?}"
    );
    // Monotonic across the boundary: mapping the new segment's own start
    // through it must not run backwards past the base it inherited.
    let mapped = queued_seg
        .to_running_time(ClockTime::from_nanos(queued_seg.start as u64))
        .nanos() as i64;
    assert!(
        mapped >= queued_seg.base,
        "running time continues from the accumulated base ({mapped} < {})",
        queued_seg.base
    );
}
