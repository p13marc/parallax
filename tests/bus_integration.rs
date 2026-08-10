//! Integration tests for the pipeline bus messaging system.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ProduceContext, ProduceResult, Source};
use parallax::elements::{NullSink, NullSource};
use parallax::error::{Error, Result};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::bus::{Bus, MessageKind};
use parallax::pipeline::{Executor, Pipeline, PipelineState};

/// Test that a pipeline creates a bus and handle automatically.
#[test]
fn test_pipeline_has_bus() {
    let mut pipeline = Pipeline::new();
    assert!(pipeline.take_bus().is_some());
    // Second take returns None
    assert!(pipeline.take_bus().is_none());
}

/// Test basic bus posting and polling.
#[test]
fn test_bus_post_and_poll() {
    let (mut bus, handle) = Bus::new();

    handle.post(MessageKind::Eos);
    handle.post_error("test error", Some("debug info".into()));
    handle.post_warning("test warning", None);
    handle.post_info("test info");

    // Poll messages in order
    let msg = bus.poll().unwrap();
    assert!(matches!(msg.kind, MessageKind::Eos));
    assert_eq!(msg.source, "pipeline");

    let msg = bus.poll().unwrap();
    match &msg.kind {
        MessageKind::Error { error, debug } => {
            assert_eq!(error, "test error");
            assert_eq!(debug.as_deref(), Some("debug info"));
        }
        _ => panic!("Expected Error"),
    }

    let msg = bus.poll().unwrap();
    assert!(matches!(msg.kind, MessageKind::Warning { .. }));

    let msg = bus.poll().unwrap();
    assert!(matches!(msg.kind, MessageKind::Info { .. }));

    assert!(bus.poll().is_none());
}

/// Test element-specific handles have correct source names.
#[test]
fn test_bus_handle_source_names() {
    let (mut bus, handle) = Bus::new();

    let decoder = handle.for_element("decoder");
    let encoder = handle.for_element("encoder");

    decoder.post_info("decoded");
    encoder.post_info("encoded");

    let msg1 = bus.poll().unwrap();
    assert_eq!(msg1.source, "decoder");

    let msg2 = bus.poll().unwrap();
    assert_eq!(msg2.source, "encoder");
}

/// Test bus peek doesn't consume the message.
#[test]
fn test_bus_peek() {
    let (mut bus, handle) = Bus::new();
    handle.post(MessageKind::Eos);

    assert!(bus.peek().is_some());
    assert!(bus.peek().is_some()); // Still there

    let msg = bus.poll().unwrap();
    assert!(matches!(msg.kind, MessageKind::Eos));
    assert!(bus.poll().is_none());
}

/// Test async wait_for_eos_or_error with EOS.
#[tokio::test]
async fn test_bus_wait_eos() {
    let (mut bus, handle) = Bus::new();

    tokio::spawn(async move {
        handle.post_info("starting");
        handle.post(MessageKind::Eos);
    });

    let result = bus.wait_for_eos_or_error().await;
    assert!(result.is_ok());
}

/// Test async wait_for_eos_or_error with error.
#[tokio::test]
async fn test_bus_wait_error() {
    let (mut bus, handle) = Bus::new();

    tokio::spawn(async move {
        handle.post_error("fatal", None);
    });

    let result = bus.wait_for_eos_or_error().await;
    assert_eq!(result.unwrap_err(), "fatal");
}

/// Test broadcast subscriber receives messages.
#[test]
fn test_bus_broadcast_subscriber() {
    let (mut bus, handle) = Bus::new();
    let mut sub = bus.subscribe();

    handle.post(MessageKind::Eos);
    handle.post_info("hello");

    // Flush from mpsc to broadcast
    bus.flush_to_broadcast();

    let msg1 = sub.try_recv().unwrap();
    assert!(matches!(msg1.kind, MessageKind::Eos));

    let msg2 = sub.try_recv().unwrap();
    assert!(matches!(msg2.kind, MessageKind::Info { .. }));
}

/// Test that state change messages are posted to the bus during pipeline execution.
#[tokio::test]
async fn test_pipeline_bus_state_changes() {
    let mut pipeline = Pipeline::new();

    let src = pipeline.add_source("src", NullSource::new(5));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    // Subscribe to bus before starting
    let mut bus = pipeline.take_bus().unwrap();
    let _bus_handle = pipeline.bus_handle().clone();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    // Drain bus messages and check for state changes
    let mut saw_suspended_to_idle = false;
    let mut saw_idle_to_running = false;

    while let Some(msg) = bus.poll() {
        if let MessageKind::StateChanged { old, new } = &msg.kind {
            if *old == PipelineState::Suspended && *new == PipelineState::Idle {
                saw_suspended_to_idle = true;
            }
            if *old == PipelineState::Idle && *new == PipelineState::Running {
                saw_idle_to_running = true;
            }
        }
    }

    assert!(
        saw_suspended_to_idle,
        "Expected Suspended->Idle state change on bus"
    );
    assert!(
        saw_idle_to_running,
        "Expected Idle->Running state change on bus"
    );
}

// ---------------------------------------------------------------------------
// #89: terminal messages from a real pipeline.
//
// Everything above `test_pipeline_bus_state_changes` posts by hand on a
// standalone `Bus`, which is why the gap went unnoticed for so long: nothing in
// `src/` had ever called `post_eos` or `post_error`, so
// `Bus::wait_for_eos_or_error()` was public API that could not return and
// `run_with_bus`'s `Error` arm was unreachable.
//
// The contract these pin: a run posts **exactly one** terminal message — `Eos`
// or `Error`, never both, never twice — and `Error` is attributed to the element
// that failed.
// ---------------------------------------------------------------------------

const LIMIT: Duration = Duration::from_secs(10);

struct FailingSource;

impl Source for FailingSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        Err(Error::Element("simulated device failure".into()))
    }

    fn name(&self) -> &str {
        "failing-source"
    }
}

/// A live source: produces forever, so only the handler can end the run.
struct InfiniteSource {
    sequence: u64,
    arena: SharedArena,
}

impl InfiniteSource {
    fn new() -> Self {
        Self {
            sequence: 0,
            arena: SharedArena::new(64, 256).unwrap(),
        }
    }
}

impl Source for InfiniteSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        self.arena.reclaim();
        let slot = self
            .arena
            .acquire()
            .ok_or_else(|| Error::Element("arena exhausted".into()))?;
        let buffer = Buffer::new(
            MemoryHandle::with_len(slot, 64),
            Metadata::from_sequence(self.sequence),
        );
        self.sequence += 1;
        Ok(ProduceResult::OwnBuffer(buffer))
    }

    fn name(&self) -> &str {
        "infinite-source"
    }
}

fn clean_pipeline() -> Pipeline {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", NullSource::new(5));
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();
    pipeline
}

fn failing_pipeline() -> Pipeline {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", FailingSource);
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();
    pipeline
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_clean_pipeline_posts_eos_to_the_bus() {
    let mut pipeline = clean_pipeline();
    let mut bus = pipeline.take_bus().unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();

    let outcome = tokio::time::timeout(LIMIT, bus.wait_for_eos_or_error())
        .await
        .expect("wait_for_eos_or_error never returned on a clean run");
    assert_eq!(outcome, Ok(()));

    handle.wait().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failing_pipeline_posts_error_to_the_bus() {
    let mut pipeline = failing_pipeline();
    let mut bus = pipeline.take_bus().unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();

    let outcome = tokio::time::timeout(LIMIT, bus.wait_for_eos_or_error())
        .await
        .expect("wait_for_eos_or_error never returned on a failing run");
    match outcome {
        Err(msg) => assert!(msg.contains("simulated device failure"), "got: {msg}"),
        Ok(()) => panic!("a failing pipeline reported a clean end"),
    }

    assert!(handle.wait().await.is_err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_terminal_message_names_the_element_that_failed() {
    let mut pipeline = failing_pipeline();
    let mut bus = pipeline.take_bus().unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    let _ = handle.wait().await;

    let mut errors = Vec::new();
    while let Some(msg) = bus.poll() {
        if matches!(msg.kind, MessageKind::Error { .. }) {
            errors.push(msg);
        }
    }

    assert_eq!(errors.len(), 1, "expected exactly one Error message");
    assert_eq!(errors[0].source, "src");
}

/// Never both. A failure must not also report the stream running out.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_failing_pipeline_posts_no_eos() {
    let mut pipeline = failing_pipeline();
    let mut bus = pipeline.take_bus().unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    let _ = handle.wait().await;

    while let Some(msg) = bus.poll() {
        assert!(
            !matches!(msg.kind, MessageKind::Eos),
            "a failing run posted Eos as well as Error"
        );
    }
}

/// An aborted run is neither: there is no `MessageKind::Aborted`, and an `Eos`
/// would claim the stream ran out when the caller cut it off.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_aborted_pipeline_posts_no_terminal_message() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", InfiniteSource::new());
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();
    let mut bus = pipeline.take_bus().unwrap();

    let handle = Executor::new().start(&mut pipeline).unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    handle.abort();

    while let Some(msg) = bus.poll() {
        assert!(
            !matches!(msg.kind, MessageKind::Eos | MessageKind::Error { .. }),
            "an aborted run posted a terminal message: {:?}",
            msg.kind
        );
    }
}

/// The `Error` arm of `run_with_bus` was unreachable twice over: nothing posted
/// `Error`, and the drain ran after `handle.wait().await?` had already returned.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_with_bus_surfaces_the_error_through_the_handler() {
    let mut pipeline = failing_pipeline();
    let seen = Arc::new(Mutex::new(Vec::new()));

    let recorder = seen.clone();
    let result = tokio::time::timeout(
        LIMIT,
        pipeline.run_with_bus(move |msg| {
            recorder.lock().unwrap().push(format!("{:?}", msg.kind));
            true
        }),
    )
    .await
    .expect("run_with_bus never returned");

    assert!(result.is_err(), "a failing pipeline returned Ok");
    let seen = seen.lock().unwrap();
    assert!(
        seen.iter().any(|k| k.starts_with("Error")),
        "the handler never saw the Error message: {seen:?}"
    );
}

/// "Return `false` to stop the pipeline" has been documented all along and has
/// never worked: the handler only ever ran on an already-dead pipeline, so
/// `false` broke a drain loop and nothing else. With a live source, that is the
/// difference between returning and hanging forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn run_with_bus_returning_false_stops_a_live_source() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", InfiniteSource::new());
    let sink = pipeline.add_sink("sink", NullSink::new());
    pipeline.link(src, sink).unwrap();

    let result = tokio::time::timeout(LIMIT, pipeline.run_with_bus(|_| false))
        .await
        .expect("returning false did not stop a live pipeline");

    result.unwrap();
}

/// Test that tags can be posted and received.
#[test]
fn test_bus_tags() {
    use parallax::pipeline::tags::{TagList, TagValue, tag_keys};

    let (mut bus, handle) = Bus::new();

    let mut tags = TagList::new();
    tags.set(tag_keys::TITLE, TagValue::String("Test Song".into()));
    tags.set(tag_keys::BITRATE, TagValue::Uint(320_000));

    handle.post_tags(tags);

    let msg = bus.poll().unwrap();
    match msg.kind {
        MessageKind::Tag { tags } => {
            assert_eq!(tags.get_string(tag_keys::TITLE), Some("Test Song"));
            assert_eq!(tags.get_uint(tag_keys::BITRATE), Some(320_000));
        }
        _ => panic!("Expected Tag message"),
    }
}

/// Test message display formatting.
#[test]
fn test_message_display() {
    let (mut bus, handle) = Bus::new();

    let elem = handle.for_element("decoder");
    elem.post_error("codec failure", Some("frame 42".into()));

    let msg = bus.poll().unwrap();
    let display = format!("{msg}");
    assert!(display.contains("decoder"));
    assert!(display.contains("codec failure"));
}

/// Test that Bus implements Stream for use with select! and StreamExt.
#[tokio::test]
async fn test_bus_stream() {
    use futures::StreamExt;

    let (bus, handle) = Bus::new();

    tokio::spawn(async move {
        handle.post(MessageKind::Info {
            info: "first".into(),
        });
        handle.post(MessageKind::Info {
            info: "second".into(),
        });
        handle.post(MessageKind::Eos);
    });

    let mut stream = bus.into_stream();
    let mut messages = Vec::new();

    while let Some(msg) = stream.next().await {
        let is_eos = matches!(msg.kind, MessageKind::Eos);
        messages.push(msg);
        if is_eos {
            break;
        }
    }

    assert_eq!(messages.len(), 3);
    assert!(matches!(messages[0].kind, MessageKind::Info { .. }));
    assert!(matches!(messages[2].kind, MessageKind::Eos));
}

/// Test Bus stream with tokio::select!
#[tokio::test]
async fn test_bus_stream_select() {
    use futures::StreamExt;

    let (bus, handle) = Bus::new();

    handle.post(MessageKind::Eos);

    let mut stream = bus.into_stream();
    let timeout = tokio::time::sleep(std::time::Duration::from_secs(1));
    tokio::pin!(timeout);

    tokio::select! {
        msg = stream.next() => {
            let msg = msg.unwrap();
            assert!(matches!(msg.kind, MessageKind::Eos));
        }
        _ = &mut timeout => {
            panic!("Timed out waiting for bus message");
        }
    }
}
