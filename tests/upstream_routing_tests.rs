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

/// #163 phase B: the demuxer half of the same round trip.
///
/// A *fed* demuxer that translated a TIME seek into a BYTES seek needs three
/// things back from the executor, and none of them existed before this test:
///
/// 1. the source's post-seek `Segment`, delivered to
///    `Demuxer::handle_downstream_event` — the demuxer's cue that the byte
///    cursor moved, carrying the file total so a byte estimate can be clamped;
/// 2. `Demuxer::flush()` on `FlushStart`, so a half-assembled access unit
///    from before the seek is dropped rather than welded onto the first
///    post-seek bytes (the output of *that* call is discarded);
/// 3. the same `flush()` at EOS, with the output **routed** — a fed demuxer's
///    `produce()` is never called, so this is its only chance to emit the
///    tail, and it must land on its own pad rather than every branch.
#[tokio::test(flavor = "multi_thread")]
async fn a_fed_demuxer_is_flushed_and_sees_the_translated_segment() {
    use parallax::element::{Demuxer, PadAddedCallback, PadId, RoutedOutput};
    use parallax::event::{SeekEvent, SeekPosition};
    use parallax::format::Caps;
    use parallax::pipeline::seek::{DurationQuery, SeekTranslation};

    const TOTAL: u64 = 4096;

    /// A byte-addressed source that knows its own size.
    struct SizedSource {
        produced: u64,
    }

    impl Source for SizedSource {
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

        fn query_duration(&self) -> Option<DurationQuery> {
            Some(DurationQuery {
                format: SegmentFormat::Bytes,
                duration: Some(TOTAL),
            })
        }

        fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
            match event {
                Event::Seek(seek) if seek.format == SegmentFormat::Bytes => {
                    EventResult::handled_at(seek.start.position)
                }
                _ => EventResult::NotHandled,
            }
        }
    }

    #[derive(Default)]
    struct Record {
        segments: Vec<(SegmentFormat, i64, i64)>,
        flushes: usize,
    }

    struct TranslatingDemuxer {
        outputs: Vec<(PadId, Caps)>,
        record: Arc<Mutex<Record>>,
    }

    impl Demuxer for TranslatingDemuxer {
        fn demux(&mut self, buffer: Buffer) -> Result<RoutedOutput> {
            Ok(RoutedOutput::single(PadId(0), buffer))
        }

        fn seek_translations(&self) -> Vec<SeekTranslation> {
            vec![SeekTranslation {
                from: SegmentFormat::Time,
                to: SegmentFormat::Bytes,
                duration: None,
            }]
        }

        fn pad_name(&self, _pad: PadId) -> String {
            "video".into()
        }

        fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
            if let Event::Seek(seek) = event
                && seek.format == SegmentFormat::Time
            {
                return EventResult::forward(Event::Seek(
                    seek.derive(SegmentFormat::Bytes, SeekPosition::set(1024)),
                ));
            }
            EventResult::NotHandled
        }

        fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
            if let Event::Segment(seg) = &event {
                self.record
                    .lock()
                    .unwrap()
                    .segments
                    .push((seg.format, seg.start, seg.stop));
            }
            Some(event)
        }

        fn flush(&mut self) -> Result<RoutedOutput> {
            self.record.lock().unwrap().flushes += 1;
            arena().reclaim();
            match arena().acquire() {
                // Sequence 0 marks the tail: no produced buffer ever uses it.
                Some(slot) => Ok(RoutedOutput::single(
                    PadId(0),
                    Buffer::new(MemoryHandle::with_len(slot, 8), Metadata::from_sequence(0)),
                )),
                None => Ok(RoutedOutput::new()),
            }
        }

        fn outputs(&self) -> &[(PadId, Caps)] {
            &self.outputs
        }

        fn on_pad_added(&mut self, _callback: PadAddedCallback) {}

        fn name(&self) -> &str {
            "translatingdemux"
        }
    }

    let record = Arc::new(Mutex::new(Record::default()));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", SizedSource { produced: 0 });
    let demux = pipeline.add_demuxer(
        "demux",
        TranslatingDemuxer {
            outputs: vec![(PadId(0), Caps::any())],
            record: record.clone(),
        },
    );
    let appsink = AppSink::with_max_buffers(2);
    let sink_handle = appsink.handle();
    let snk = pipeline.add_async_sink("sink", appsink);
    pipeline.link(src, demux).unwrap();
    pipeline.link_pads(demux, "video", snk, "sink").unwrap();

    // Before start: the graph already answers TIME, not the source's BYTES.
    let pre = pipeline.query_seekable();
    assert!(pre.seekable);
    assert_eq!(pre.format, SegmentFormat::Time);

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // And the running pipeline agrees — the two surfaces share the fold.
    let live = handle.query_seekable();
    assert_eq!(live.format, SegmentFormat::Time);
    assert_eq!(
        live.stop, 0,
        "the demuxer knows no duration, so the TIME range is open"
    );

    let tails = Arc::new(AtomicU64::new(0));
    let counted = tails.clone();
    let drain = tokio::spawn(async move {
        loop {
            match sink_handle.pull_buffer().await {
                Pulled::Buffer(b) => {
                    if b.metadata().sequence == 0 {
                        counted.fetch_add(1, Ordering::SeqCst);
                    }
                }
                Pulled::Ended(_) => break,
                Pulled::Empty | Pulled::Flushing => tokio::task::yield_now().await,
            }
        }
    });

    assert!(
        handle
            .seek(SeekEvent::new_time(ClockTime::from_secs(5)))
            .await
    );

    wait_until(
        || record.lock().unwrap().flushes > 0,
        "the demuxer to be flushed by FlushStart",
    )
    .await;
    wait_until(
        || !record.lock().unwrap().segments.is_empty(),
        "the source's post-seek segment to reach the demuxer",
    )
    .await;

    handle.stop();
    drain.await.unwrap();
    handle.wait().await.unwrap();

    let record = record.lock().unwrap();
    // The lazy initial segment (start 0) precedes the seek's; both carry the
    // total, which is what makes a byte estimate clampable at any point.
    assert!(
        record
            .segments
            .contains(&(SegmentFormat::Bytes, 1024, TOTAL as i64)),
        "the demuxer sees where the source landed and how big the stream is: {:?}",
        record.segments
    );
    assert!(
        record
            .segments
            .iter()
            .all(|(_, _, stop)| *stop == TOTAL as i64),
        "every byte segment carries the total: {:?}",
        record.segments
    );
    assert!(
        record.flushes >= 2,
        "flushed once on FlushStart and once at EOS, got {}",
        record.flushes
    );
    assert_eq!(
        tails.load(Ordering::SeqCst),
        1,
        "the FlushStart flush is discarded; only the EOS one is routed"
    );
}

// ============================================================================
// QoS origination and consumption (#184)
// ============================================================================

/// An async sink that stages one `Event::Qos` after consuming a set number
/// of buffers — the element-side origination API in miniature.
struct QosReportingSink {
    consumed: u64,
    report_after: u64,
    staged: Option<Event>,
    reported: bool,
}

impl QosReportingSink {
    fn new(report_after: u64) -> Self {
        Self {
            consumed: 0,
            report_after,
            staged: None,
            reported: false,
        }
    }
}

impl parallax::element::AsyncSink for QosReportingSink {
    async fn consume(&mut self, _ctx: &parallax::element::ConsumeContext<'_>) -> Result<()> {
        self.consumed += 1;
        if !self.reported && self.consumed >= self.report_after {
            self.reported = true;
            self.staged = Some(Event::Qos(parallax::event::QosEvent {
                qos_type: parallax::event::QosType::Underflow,
                proportion: 4.0,
                jitter_ns: 7_000_000,
                timestamp: ClockTime::from_nanos(123),
                processed: self.report_after,
                dropped: 3,
            }));
        }
        Ok(())
    }

    fn take_upstream_event(&mut self) -> Option<Event> {
        self.staged.take()
    }

    fn name(&self) -> &str {
        "qos_sink"
    }
}

/// #184: a sink-originated QoS event reaches the source's pad (hop-by-hop
/// through a transform) and is mirrored on the bus with its fields intact.
#[tokio::test(flavor = "multi_thread")]
async fn qos_travels_from_sink_to_source_and_bus() {
    use parallax::elements::{AppSrc, PassThrough};

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let xfm = pipeline.add_filter("mid", PassThrough::new());
    let snk = pipeline.add_async_sink("qos_sink", QosReportingSink::new(3));
    pipeline.link(src, xfm).unwrap();
    pipeline.link(xfm, snk).unwrap();

    // EVENT_UP probe on the source's src pad: the last hop of the route.
    let seen: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::new()));
    let seen_probe = seen.clone();
    let _ = pipeline.add_probe(PadRef::src(src), ProbeType::EVENT_UP, move |data| {
        if let ProbeData::Event(Event::Qos(q)) = data {
            seen_probe.lock().unwrap().push(q.proportion);
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let mut handle = executor.start(&mut pipeline).unwrap();
    let mut bus = handle.take_bus().unwrap();

    for seq in 0..5u64 {
        let slot = arena().acquire().expect("arena");
        src_handle
            .push_buffer(Buffer::new(
                MemoryHandle::with_len(slot, 8),
                Metadata::from_sequence(seq),
            ))
            .await
            .unwrap();
    }

    wait_until(
        || !seen.lock().unwrap().is_empty(),
        "the QoS event at the source pad",
    )
    .await;
    assert_eq!(seen.lock().unwrap().as_slice(), &[4.0]);

    // Mirrored on the bus with the staged fields.
    let mut bus_qos = None;
    wait_until(
        || {
            while let Some(msg) = bus.poll() {
                if let MessageKind::Qos {
                    qos_type,
                    proportion,
                    jitter_ns,
                    processed,
                    dropped,
                    ..
                } = msg.kind
                {
                    bus_qos = Some((qos_type, proportion, jitter_ns, processed, dropped));
                }
            }
            bus_qos.is_some()
        },
        "the QoS bus message",
    )
    .await;
    assert_eq!(
        bus_qos,
        Some((parallax::event::QosType::Underflow, 4.0, 7_000_000, 3, 3))
    );

    src_handle.end_stream();
    handle.wait().await.unwrap();
}

/// #184: a Throttle between source and sink observes a QoS underflow and
/// scales its interval up by the reported proportion (rate down), visible
/// through its pre-start control handle. The event still travels on to the
/// source (observe-and-forward).
#[tokio::test(flavor = "multi_thread")]
async fn throttle_degrades_on_qos_underflow() {
    use parallax::elements::{AppSrc, Throttle};

    let mut pipeline = Pipeline::new();
    let appsrc = AppSrc::with_max_buffers(32);
    let src_handle = appsrc.handle();
    let src = pipeline.add_source("src", appsrc);
    let throttle = Throttle::from_millis(1);
    let control = throttle.control();
    let thr = pipeline.add_filter("throttle", throttle);
    let snk = pipeline.add_async_sink("qos_sink", QosReportingSink::new(2));
    pipeline.link(src, thr).unwrap();
    pipeline.link(thr, snk).unwrap();

    let up_at_src: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let up_probe = up_at_src.clone();
    let _ = pipeline.add_probe(PadRef::src(src), ProbeType::EVENT_UP, move |data| {
        if let ProbeData::Event(Event::Qos(_)) = data {
            *up_probe.lock().unwrap() += 1;
        }
        ProbeReturn::Ok
    });

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    let before = control.min_interval();
    assert_eq!(before, Duration::from_millis(1));

    // Push spaced-out buffers so the 1 ms throttle passes them all.
    for seq in 0..4u64 {
        let slot = arena().acquire().expect("arena");
        src_handle
            .push_buffer(Buffer::new(
                MemoryHandle::with_len(slot, 8),
                Metadata::from_sequence(seq),
            ))
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(3)).await;
    }

    // proportion 4.0 on a 1 ms interval -> 4 ms.
    wait_until(
        || control.min_interval() == Duration::from_millis(4),
        "the throttle to scale its interval by the QoS proportion",
    )
    .await;

    // Observe-and-forward: the source still saw the event.
    wait_until(
        || *up_at_src.lock().unwrap() > 0,
        "the QoS event to continue to the source",
    )
    .await;

    src_handle.end_stream();
    handle.wait().await.unwrap();
}
