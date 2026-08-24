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
