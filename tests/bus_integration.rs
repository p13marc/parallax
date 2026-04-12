//! Integration tests for the pipeline bus messaging system.

use parallax::elements::{NullSink, NullSource};
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
