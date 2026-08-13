//! #85: every way a pull can end has its own name.
//!
//! `Ok(None)` used to mean "timed out", "ended cleanly" and "died upstream",
//! and a consumer could not tell them apart. Downstream that produced a 100 ms
//! poll loop whose only job was to interleave `is_eos()` checks, and a
//! ten-second watchdog to notice a source that had panicked.
//!
//! No pipeline here on purpose: driving the AppSink directly keeps every one of
//! these fully deterministic, with no sleeps and nothing to race.

use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{AsyncSink, ConsumeContext};
use parallax::elements::{AppSink, EndReason, Pulled};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::StreamError;

fn arena() -> &'static SharedArena {
    use std::sync::OnceLock;
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(64, 256).unwrap())
}

fn buffer(seq: u64) -> Buffer {
    let slot = arena().acquire().expect("test arena exhausted");
    Buffer::new(
        MemoryHandle::with_len(slot, 8),
        Metadata::from_sequence(seq),
    )
}

async fn push(sink: &mut AppSink, seq: u64) {
    let buf = buffer(seq);
    sink.consume(&ConsumeContext::new(&buf)).await.unwrap();
}

/// The headline. These two were the same value before.
#[tokio::test]
async fn a_timeout_is_not_an_end_of_stream() {
    let sink = AppSink::new();
    let handle = sink.handle();

    let timed_out = handle.pull_buffer_timeout(Duration::from_millis(20)).await;
    assert!(
        matches!(timed_out, Pulled::Empty),
        "an idle stream must report Empty, not Ended: {timed_out:?}"
    );

    sink.send_eos();

    let ended = handle.pull_buffer_timeout(Duration::from_millis(20)).await;
    assert!(
        matches!(ended, Pulled::Ended(EndReason::Eos)),
        "after EOS the same call must report the end: {ended:?}"
    );
}

#[tokio::test]
async fn the_blocking_timeout_makes_the_same_distinction() {
    let sink = AppSink::new();
    let handle = sink.handle();

    assert!(matches!(
        handle.pull_buffer_timeout_blocking(Duration::from_millis(20)),
        Pulled::Empty
    ));

    sink.send_eos();
    assert!(matches!(
        handle.pull_buffer_timeout_blocking(Duration::from_millis(20)),
        Pulled::Ended(EndReason::Eos)
    ));
}

/// `try_pull_buffer` returned a bare `Option`, so "nothing yet" and "over"
/// were the same answer — the asymmetry AppSrc's docs already complained about.
#[tokio::test]
async fn try_pull_separates_empty_from_ended() {
    let sink = AppSink::new();
    let handle = sink.handle();

    assert!(matches!(handle.try_pull_buffer(), Pulled::Empty));

    sink.send_eos();
    assert!(matches!(
        handle.try_pull_buffer(),
        Pulled::Ended(EndReason::Eos)
    ));
}

#[tokio::test]
async fn an_upstream_failure_is_not_an_end_of_stream() {
    let sink = AppSink::new();
    let handle = sink.handle();

    sink.send_error(StreamError::new("v4l2src", "device disappeared"));

    match handle.pull_buffer().await {
        Pulled::Ended(EndReason::Error(err)) => {
            assert_eq!(err.node(), Some("v4l2src"));
            assert_eq!(err.message(), "device disappeared");
        }
        other => panic!("a failure must not look like a clean end: {other:?}"),
    }
}

/// Buffers already delivered when an element dies are still valid.
#[tokio::test]
async fn queued_buffers_drain_before_the_end_is_reported() {
    let mut sink = AppSink::new();
    let handle = sink.handle();

    for seq in 0..3 {
        push(&mut sink, seq).await;
    }
    sink.send_error(StreamError::new("enc", "boom"));

    for expected in 0..3 {
        match handle.pull_buffer().await {
            Pulled::Buffer(b) => assert_eq!(b.metadata().sequence, expected),
            other => panic!("lost buffer {expected}: {other:?}"),
        }
    }
    assert!(matches!(
        handle.pull_buffer().await,
        Pulled::Ended(EndReason::Error(_))
    ));
}

/// A late EOS must not paper over why it really stopped.
#[tokio::test]
async fn the_first_reason_wins() {
    let sink = AppSink::new();
    let handle = sink.handle();

    sink.send_error(StreamError::new("src", "the real reason"));
    sink.send_eos();

    match handle.end_reason().expect("a reason was recorded") {
        EndReason::Error(err) => assert_eq!(err.message(), "the real reason"),
        other => panic!("EOS overwrote the failure: {other:?}"),
    }
}

/// Flushing is app-initiated and recoverable, so it is neither an error nor an
/// end — a fourth outcome the issue did not name.
#[tokio::test]
async fn flushing_is_not_ending() {
    let mut sink = AppSink::new();
    let handle = sink.handle();

    handle.set_flushing(true);
    assert!(matches!(handle.try_pull_buffer(), Pulled::Flushing));
    assert!(!handle.is_ended(), "flushing is not terminal");

    handle.set_flushing(false);
    push(&mut sink, 42).await;
    match handle.try_pull_buffer() {
        Pulled::Buffer(b) => assert_eq!(b.metadata().sequence, 42),
        other => panic!("the sink did not recover from flushing: {other:?}"),
    }
}

/// `ended()` waits for the drain, which is what lets a `select!` consumer use
/// it without dropping buffers on the floor.
#[tokio::test]
async fn ended_waits_until_the_queue_is_drained() {
    let mut sink = AppSink::new();
    let handle = sink.handle();

    push(&mut sink, 0).await;
    sink.send_eos();

    let waiter = {
        let handle = handle.clone();
        tokio::spawn(async move { handle.ended().await })
    };
    tokio::task::yield_now().await;
    assert!(
        !waiter.is_finished(),
        "ended() fired with a buffer still queued — a select! consumer would \
         have torn down and lost it"
    );

    // The reason is knowable early, though; that is what end_reason() is for.
    assert_eq!(handle.end_reason(), Some(EndReason::Eos));

    assert!(matches!(handle.pull_buffer().await, Pulled::Buffer(_)));
    assert_eq!(waiter.await.unwrap(), EndReason::Eos);
}

/// The idiom the issue asks for: no polling, no watchdog.
#[tokio::test]
async fn a_consumer_can_select_over_pull_and_ended() {
    let mut sink = AppSink::new();
    let handle = sink.handle();

    for seq in 0..4 {
        push(&mut sink, seq).await;
    }
    sink.send_error(StreamError::new("cam", "cable pulled"));

    let mut seen = 0;
    let reason = loop {
        tokio::select! {
            biased;
            pulled = handle.pull_buffer() => match pulled {
                Pulled::Buffer(_) => seen += 1,
                Pulled::Ended(reason) => break reason,
                other => panic!("unexpected: {other:?}"),
            },
            reason = handle.ended() => break reason,
        }
    };

    assert_eq!(seen, 4, "the select! lost buffers");
    match reason {
        EndReason::Error(err) => assert_eq!(err.message(), "cable pulled"),
        other => panic!("wrong reason: {other:?}"),
    }
}

#[tokio::test]
async fn stats_report_termination_without_saying_why() {
    let mut sink = AppSink::new();
    let handle = sink.handle();

    push(&mut sink, 0).await;
    assert!(!handle.stats().ended);

    sink.send_error(StreamError::anonymous("no node to blame"));
    let stats = handle.stats();
    assert!(stats.ended);
    assert_eq!(stats.queued_buffers, 1);
    // The reason lives on the handle, not in a stats struct.
    assert!(matches!(handle.end_reason(), Some(EndReason::Error(_))));
}

#[tokio::test]
async fn an_anonymous_failure_still_reads_sensibly() {
    let err = StreamError::anonymous("something went wrong");
    assert_eq!(err.node(), None);
    assert_eq!(err.to_string(), "something went wrong");

    let attributed = StreamError::new("enc", "bad input");
    assert_eq!(attributed.to_string(), "element 'enc': bad input");
}
