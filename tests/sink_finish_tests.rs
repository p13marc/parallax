//! #221: the terminal hook a sink needs when its goodbye is I/O.
//!
//! `handle_downstream_event` is synchronous and the sink task drops the
//! element immediately after it, so a sink whose last act is an `await` —
//! publishing an end-of-stream sample, sending a WebSocket Close frame — had
//! only two options, both wrong: a detached `tokio::spawn` racing teardown, or
//! blocking in `Drop`, which strands a worker (gotcha 15).
//!
//! What these tests pin is not that the hook exists but *when* it runs: on
//! every exit the sink task has, with the reason the run actually ended for,
//! bounded so an unreachable peer cannot hang the teardown.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use parallax::buffer::Buffer;
use parallax::element::{AsyncSink, ConsumeContext};
use parallax::elements::{NullSink, NullSource};
use parallax::error::{Error, Result};
use parallax::pipeline::bus::MessageKind;
use parallax::pipeline::{EndReason, Executor, ExecutorConfig, Pipeline};
use tokio::sync::Mutex;

use parallax::buffer::MemoryHandle;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;

/// A buffer of `bytes` carrying `metadata`, from a shared test arena.
fn payload(bytes: &[u8], metadata: Metadata) -> Buffer {
    use std::sync::OnceLock;
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    let arena = ARENA.get_or_init(|| SharedArena::new(4096, 32).unwrap());
    arena.reclaim();
    let mut slot = arena.acquire().expect("test arena slot");
    slot.data_mut()[..bytes.len()].copy_from_slice(bytes);
    Buffer::new(MemoryHandle::with_len(slot, bytes.len()), metadata)
}

/// A sink that records what its `finish` was told, from inside an `await`.
struct RecordingSink {
    /// `Some(reason)` once `finish` ran. A `tokio::sync::Mutex` because the
    /// hook is genuinely async and holding a std guard across the await is
    /// what the executor's own code had to avoid.
    seen: Arc<Mutex<Option<EndReason>>>,
    /// Fail the Nth consume, to reach the fatal-error exit.
    fail_after: Option<u64>,
    count: u64,
    /// Hang forever in `finish`, to reach the timeout.
    hang: bool,
}

impl RecordingSink {
    fn new(seen: Arc<Mutex<Option<EndReason>>>) -> Self {
        Self {
            seen,
            fail_after: None,
            count: 0,
            hang: false,
        }
    }

    fn failing_after(mut self, n: u64) -> Self {
        self.fail_after = Some(n);
        self
    }

    fn hanging(mut self) -> Self {
        self.hang = true;
        self
    }
}

impl AsyncSink for RecordingSink {
    async fn consume(&mut self, _ctx: &ConsumeContext<'_>) -> Result<()> {
        self.count += 1;
        if self.fail_after == Some(self.count) {
            return Err(Error::Element("deliberate failure".into()));
        }
        Ok(())
    }

    async fn finish(&mut self, reason: EndReason) -> Result<()> {
        if self.hang {
            // Longer than any test's patience; the executor must cut it off.
            tokio::time::sleep(Duration::from_secs(3600)).await;
        }
        // The await is the point: this is work a sync hook could not do.
        tokio::task::yield_now().await;
        *self.seen.lock().await = Some(reason);
        Ok(())
    }

    fn name(&self) -> &str {
        "recording_sink"
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_clean_end_says_goodbye_with_eos() {
    let seen = Arc::new(Mutex::new(None));
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let sink = pipeline.add_async_sink("sink", RecordingSink::new(seen.clone()));
    pipeline.link(src, sink).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    assert_eq!(*seen.lock().await, Some(EndReason::Eos));
}

/// The unhappy path is the one that matters: a sink that only committed on a
/// clean EOS would leave the half-written file for exactly the runs where
/// something went wrong.
#[tokio::test(flavor = "multi_thread")]
async fn a_failing_sink_still_says_goodbye_and_is_told_why() {
    let seen = Arc::new(Mutex::new(None));
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(20));
    let sink = pipeline.add_async_sink("sink", RecordingSink::new(seen.clone()).failing_after(3));
    pipeline.link(src, sink).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    let _ = handle.wait().await;

    match seen.lock().await.clone() {
        Some(EndReason::Error(err)) => {
            assert!(
                err.message().contains("deliberate failure"),
                "the reason carries what actually failed, got {err:?}"
            );
        }
        other => panic!("expected an error reason, got {other:?}"),
    }
}

/// A goodbye is I/O, and I/O wedges. The bound is what keeps one unreachable
/// peer from hanging the pipeline's teardown.
#[tokio::test(flavor = "multi_thread")]
async fn a_wedged_goodbye_is_bounded_and_reported() {
    let seen = Arc::new(Mutex::new(None));
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(3));
    let sink = pipeline.add_async_sink("sink", RecordingSink::new(seen.clone()).hanging());
    pipeline.link(src, sink).unwrap();

    let mut bus = pipeline.take_bus().unwrap();

    let config = ExecutorConfig::default().with_finish_timeout(Duration::from_millis(150));
    let handle = Executor::with_config(config).start(&mut pipeline).unwrap();

    // The run ends despite the hook never returning.
    tokio::time::timeout(Duration::from_secs(5), handle.wait())
        .await
        .expect("teardown must not wait on a wedged finish()")
        .unwrap();

    assert_eq!(
        *seen.lock().await,
        None,
        "the hook never completed, so it recorded nothing"
    );

    let mut warned = false;
    while let Some(msg) = bus.poll() {
        if let MessageKind::Warning { warning, .. } = msg.kind
            && warning.contains("finish()")
        {
            warned = true;
        }
    }
    assert!(warned, "a wedged goodbye is reported, not swallowed");
}

/// The default hook is a no-op, so a sink that does not implement it is
/// unaffected — no extra await, no behaviour change.
#[tokio::test(flavor = "multi_thread")]
async fn a_sink_without_the_hook_is_untouched() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(4));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    assert_eq!(handle.ended().await, EndReason::Eos);
}

/// `SimpleSink` is the third sink family, and the one that carries the two
/// sinks whose only close path used to be `Drop` (`HlsSink`, `DashSink`).
#[tokio::test(flavor = "multi_thread")]
async fn the_simple_sink_family_gets_the_hook_too() {
    use parallax::element::{SimpleSink, Snk};

    struct SimpleRecorder(Arc<AtomicBool>);
    impl SimpleSink for SimpleRecorder {
        fn consume(&mut self, _buffer: &Buffer) -> Result<()> {
            Ok(())
        }
        fn finish(&mut self, _reason: EndReason) -> Result<()> {
            self.0.store(true, Ordering::SeqCst);
            Ok(())
        }
        fn name(&self) -> &str {
            "simple_recorder"
        }
    }

    let said = Arc::new(AtomicBool::new(false));
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(3));
    let sink = pipeline.add_element("sink", Snk(SimpleRecorder(said.clone())));
    pipeline.link(src, sink).unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    assert!(said.load(Ordering::SeqCst));
}

/// #221's whole point, on the two sinks that actually motivated it.
///
/// The hook's *mechanism* was pinned above with a fake sink. What was never
/// checked is that `HlsSink` and `DashSink` use it to write the thing that
/// makes a stream readable as finished — and a live playlist without its
/// `#EXT-X-ENDLIST` reads as still-running to every player, forever. Before
/// this, the only `ENDLIST` assertion in the tree asserted its *absence* in a
/// string built by a pure function.
#[test]
fn hls_finish_writes_the_endlist_that_marks_the_stream_over() {
    use parallax::element::SimpleSink;
    use parallax::elements::streaming::{HlsConfig, HlsSink};

    let dir = tempfile::tempdir().expect("tempdir");
    let config = HlsConfig {
        output_dir: dir.path().to_path_buf(),
        segment_duration: 0.05,
        // Live, not VOD. A VOD playlist carries `#EXT-X-ENDLIST` from its
        // first write, so it could not tell us anything about `finish()`.
        is_vod: false,
        ..Default::default()
    };
    let playlist = dir.path().join(&config.playlist_name);
    let mut sink = HlsSink::new(config).expect("hls sink");

    // Two keyframes far enough apart to close a segment between them.
    for (i, pts_ms) in [0u64, 100].into_iter().enumerate() {
        let mut meta = Metadata::new();
        meta.pts = parallax::clock::ClockTime::from_millis(pts_ms);
        meta.flags |= parallax::metadata::BufferFlags::SYNC_POINT;
        meta.sequence = i as u64;
        SimpleSink::consume(&mut sink, &payload(&[0xAB; 512], meta)).expect("consume");
    }

    // Whatever is on disk now is a *live* playlist: still running.
    if let Ok(before) = std::fs::read_to_string(&playlist) {
        assert!(
            !before.contains("#EXT-X-ENDLIST"),
            "a running stream must not claim to be over: {before}"
        );
    }

    SimpleSink::finish(&mut sink, EndReason::Eos).expect("finish");

    let after = std::fs::read_to_string(&playlist).expect("a playlist on disk");
    assert!(
        after.contains("#EXT-X-ENDLIST"),
        "finish() must close the playlist, or players wait forever: {after}"
    );
}
