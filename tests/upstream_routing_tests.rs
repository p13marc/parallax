//! Integration tests for hop-by-hop upstream event routing (#163 phase A).
//!
//! Upstream events enter the graph at the sinks and travel toward the
//! sources; each hop's element may handle the event (running the flush trio
//! from its own task) or pass it on. Multi-path delivery (fan-out diamonds)
//! is deduplicated by seek seqnum.

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{Output, ProduceContext, ProduceResult, Source, Transform};
use parallax::elements::{AppSink, AppSinkHandle, Pulled};
use parallax::error::Result;
use parallax::event::{Event, EventResult, SegmentFormat};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};
use parallax::pipeline::{Executor, Pipeline};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

fn arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 512).unwrap())
}

/// Drain an AppSink to its terminal state. `Flushing` is transient (a seek's
/// flush window) and `Empty` non-terminal — keep pulling through both.
async fn drain_all(handle: AppSinkHandle) {
    loop {
        match handle.pull_buffer().await {
            Pulled::Buffer(_) => {}
            Pulled::Flushing | Pulled::Empty => tokio::task::yield_now().await,
            _ => break,
        }
    }
}

async fn wait_until(mut cond: impl FnMut() -> bool, what: &str) {
    for _ in 0..2000 {
        if cond() {
            return;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    panic!("timed out waiting for {what}");
}

/// A seekable source that counts the seeks reaching it.
struct CountingSeekableSource {
    produced: u64,
    seeks: Arc<AtomicU64>,
}

impl Source for CountingSeekableSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        // Slots return through the release queue, so an endless producer
        // must reclaim before it acquires.
        arena().reclaim();
        let slot = match arena().acquire() {
            Some(s) => s,
            None => return Ok(ProduceResult::WouldBlock),
        };
        self.produced += 1;
        let mut meta = Metadata::from_sequence(self.produced);
        meta.pts = ClockTime::from_millis(self.produced * 10);
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            meta,
        )))
    }

    fn is_seekable(&self) -> bool {
        true
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        if let Event::Seek(seek) = event {
            self.seeks.fetch_add(1, Ordering::SeqCst);
            EventResult::handled_at(seek.start.position)
        } else {
            EventResult::NotHandled
        }
    }
}

/// A mid-graph transform that handles Time seeks itself (a stand-in for a
/// fed demuxer translating/absorbing seeks).
struct SeekHandlingTransform {
    seeks: Arc<AtomicU64>,
}

impl Transform for SeekHandlingTransform {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        Ok(Output::Single(buffer))
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        if let Event::Seek(seek) = event
            && seek.format == SegmentFormat::Time
        {
            self.seeks.fetch_add(1, Ordering::SeqCst);
            return EventResult::handled_at(seek.start.position);
        }
        EventResult::NotHandled
    }

    fn name(&self) -> &str {
        "seekxfm"
    }
}

/// A mid-graph element can handle a seek: the flush trio originates from the
/// transform's own task, SeekDone names the transform, and the seek never
/// travels past it to the source.
#[tokio::test(flavor = "multi_thread")]
async fn transform_handles_upstream_seek_midgraph() {
    let source_seeks = Arc::new(AtomicU64::new(0));
    let xfm_seeks = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        CountingSeekableSource {
            produced: 0,
            seeks: source_seeks.clone(),
        },
    );
    let xfm = pipeline.add_transform(
        "seekxfm",
        SeekHandlingTransform {
            seeks: xfm_seeks.clone(),
        },
    );
    let appsink = AppSink::with_max_buffers(2);
    let sink_handle = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    // The flush trio must fire on the TRANSFORM's src pad — proof it
    // originated there.
    let xfm_events: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let xfm_events_clone = xfm_events.clone();
    let _ = pipeline.add_probe(
        PadRef::src(xfm),
        ProbeType::EVENT_DOWN | ProbeType::EVENT_FLUSH,
        move |data| {
            if let ProbeData::Event(e) = data {
                xfm_events_clone.lock().unwrap().push(e.name().to_string());
            }
            ProbeReturn::Ok
        },
    );

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    // Keep the sink drained in the background: a sink blocked inside
    // `consume` (full AppSink queue) only polls its inbox between buffers.
    let drain = tokio::spawn(drain_all(sink_handle));

    assert!(handle.seek_time(ClockTime::from_millis(500)).await);

    wait_until(
        || xfm_seeks.load(Ordering::SeqCst) == 1,
        "the transform to handle the seek",
    )
    .await;
    handle.stop();
    drain.await.unwrap();
    handle.wait().await.unwrap();

    assert_eq!(
        source_seeks.load(Ordering::SeqCst),
        0,
        "the seek stopped at the transform; the source never saw it"
    );

    let events = xfm_events.lock().unwrap();
    let fs = events.iter().position(|e| e == "flush-start");
    let fstop = events.iter().position(|e| e == "flush-stop");
    assert!(
        fs.is_some() && fstop.is_some() && fs < fstop,
        "flush trio originated from the transform's task: {events:?}"
    );

    let mut seek_done_source = None;
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone { source, .. } = msg.kind {
            seek_done_source = Some(source);
        }
    }
    assert_eq!(
        seek_done_source.as_deref(),
        Some("seekxfm"),
        "SeekDone names the handling element"
    );
}

/// EVENT_UP probes fire on every traversed hop, in downstream-to-upstream
/// order: the sink sees the seek before the source.
#[tokio::test(flavor = "multi_thread")]
async fn upstream_event_probes_fire_per_hop() {
    let source_seeks = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        CountingSeekableSource {
            produced: 0,
            seeks: source_seeks.clone(),
        },
    );
    let appsink = AppSink::with_max_buffers(2);
    let sink_handle = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, snk).unwrap();

    let log: Arc<Mutex<Vec<&'static str>>> = Arc::new(Mutex::new(Vec::new()));
    let log_sink = log.clone();
    let _ = pipeline.add_probe(PadRef::sink(snk), ProbeType::EVENT_UP, move |data| {
        if let ProbeData::Event(Event::Seek(_)) = data {
            log_sink.lock().unwrap().push("sink");
        }
        ProbeReturn::Ok
    });
    let log_src = log.clone();
    let _ = pipeline.add_probe(PadRef::src(src), ProbeType::EVENT_UP, move |data| {
        if let ProbeData::Event(Event::Seek(_)) = data {
            log_src.lock().unwrap().push("source");
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let drain = tokio::spawn(drain_all(sink_handle));
    assert!(handle.seek_time(ClockTime::from_millis(100)).await);

    wait_until(
        || source_seeks.load(Ordering::SeqCst) == 1,
        "the seek to reach the source",
    )
    .await;
    handle.stop();
    drain.await.unwrap();
    handle.wait().await.unwrap();

    let log = log.lock().unwrap();
    assert_eq!(
        log.as_slice(),
        &["sink", "source"],
        "the seek traversed sink → source, probed at each hop"
    );
}

/// A fan-out diamond delivers the seek along both branches; the shared
/// source handles it exactly once (seqnum dedup) and one SeekDone posts.
#[tokio::test(flavor = "multi_thread")]
async fn diamond_delivers_seek_once() {
    let source_seeks = Arc::new(AtomicU64::new(0));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        CountingSeekableSource {
            produced: 0,
            seeks: source_seeks.clone(),
        },
    );
    let a = AppSink::with_max_buffers(2);
    let a_handle = a.handle();
    let b = AppSink::with_max_buffers(2);
    let b_handle = b.handle();
    let sa = pipeline.add_async_sink("sink_a", a);
    let sb = pipeline.add_async_sink("sink_b", b);
    pipeline.link(src, sa).unwrap();
    pipeline.link_lossy(src, sb).unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    let drain_a = tokio::spawn(drain_all(a_handle));
    let drain_b = tokio::spawn(drain_all(b_handle));
    assert!(handle.seek_time(ClockTime::from_millis(100)).await);

    wait_until(
        || source_seeks.load(Ordering::SeqCst) >= 1,
        "the seek to reach the source",
    )
    .await;
    // Give the second branch's copy time to arrive (and be deduped).
    tokio::time::sleep(Duration::from_millis(50)).await;
    handle.stop();
    drain_a.await.unwrap();
    drain_b.await.unwrap();
    handle.wait().await.unwrap();

    assert_eq!(
        source_seeks.load(Ordering::SeqCst),
        1,
        "the source handled the seek exactly once"
    );
    let mut seek_dones = 0;
    while let Some(msg) = bus.poll() {
        if matches!(msg.kind, MessageKind::SeekDone { .. }) {
            seek_dones += 1;
        }
    }
    assert_eq!(seek_dones, 1, "one SeekDone for one logical seek");
}

/// #163 phase B: a mid-graph element can *translate* a seek — consume the one
/// it was given and send a different one upstream. This is how a push-mode
/// demuxer turns a TIME seek into a BYTES seek on its source.
///
/// No container involved on purpose: this is the executor-level contract, and
/// it runs under default features in every CI job.
#[tokio::test(flavor = "multi_thread")]
async fn transform_translates_seek_to_the_source() {
    use parallax::event::{SeekEvent, SeekPosition, SegmentFormat};

    let source_seeks = Arc::new(Mutex::new(Vec::<(u64, SegmentFormat, i64)>::new()));
    let seen = source_seeks.clone();

    /// Records every seek that reaches it, in whatever format.
    struct RecordingSource {
        produced: u64,
        seen: Arc<Mutex<Vec<(u64, SegmentFormat, i64)>>>,
    }

    impl Source for RecordingSource {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
            arena().reclaim();
            let Some(slot) = arena().acquire() else {
                return Ok(ProduceResult::WouldBlock);
            };
            self.produced += 1;
            Ok(ProduceResult::OwnBuffer(Buffer::new(
                MemoryHandle::with_len(slot, 8),
                Metadata::from_sequence(self.produced),
            )))
        }

        fn is_seekable(&self) -> bool {
            true
        }

        fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
            if let Event::Seek(seek) = event {
                self.seen
                    .lock()
                    .unwrap()
                    .push((seek.seqnum(), seek.format, seek.start.position));
                return EventResult::handled_at(seek.start.position);
            }
            EventResult::NotHandled
        }
    }

    /// Stands in for a fed demuxer: converts TIME to BYTES on the way up.
    struct TranslatingTransform;

    impl Transform for TranslatingTransform {
        fn transform(&mut self, buffer: Buffer) -> Result<Output> {
            Ok(Output::Single(buffer))
        }

        fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
            if let Event::Seek(seek) = event
                && seek.format == SegmentFormat::Time
            {
                // `derive`, not a fresh constructor: the seqnum must survive.
                return EventResult::forward(Event::Seek(
                    seek.derive(SegmentFormat::Bytes, SeekPosition::set(4096)),
                ));
            }
            EventResult::NotHandled
        }

        fn name(&self) -> &str {
            "translator"
        }
    }

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", RecordingSource { produced: 0, seen });
    let xfm = pipeline.add_transform("translator", TranslatingTransform);
    let appsink = AppSink::with_max_buffers(2);
    let sink_handle = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    let drain = tokio::spawn(drain_all(sink_handle));

    let seek = SeekEvent::new_time(ClockTime::from_secs(5));
    let seqnum = seek.seqnum();
    assert!(handle.seek(seek).await);

    wait_until(
        || !source_seeks.lock().unwrap().is_empty(),
        "the translated seek to reach the source",
    )
    .await;
    handle.stop();
    drain.await.unwrap();
    handle.wait().await.unwrap();

    let seen = source_seeks.lock().unwrap();
    assert_eq!(seen.len(), 1, "exactly one seek reached the source");
    assert_eq!(
        seen[0],
        (seqnum, SegmentFormat::Bytes, 4096),
        "the source saw a BYTES seek carrying the original seqnum"
    );

    // The source reports its own Bytes completion; the translator reports the
    // Time one the application actually asked for. Both share the seqnum.
    let mut dones = Vec::new();
    while let Some(msg) = bus.poll() {
        if let MessageKind::SeekDone {
            seqnum: sq, format, ..
        } = msg.kind
        {
            dones.push((sq, format));
        }
    }
    assert!(
        dones.contains(&(seqnum, SegmentFormat::Bytes)),
        "the source's byte completion is still reported: {dones:?}"
    );
}
