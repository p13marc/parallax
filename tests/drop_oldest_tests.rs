//! Integration tests for `LinkPolicy::DropOldest` (#169) — the
//! leaky-downstream link: when the channel is full, the *oldest queued*
//! buffer is evicted so the consumer always sees the freshest data.
//!
//! The mirror-image contract to `fanout_tests.rs`'s DropNewest coverage:
//! same isolation (a lossy branch degrades alone), same control-delivery
//! guarantees (EOS/events are never dropped), opposite selection of *which*
//! buffer is lost.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::{Error, Result};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, ExecutorConfig, LinkPolicy, Pipeline};

const FRAMES: u64 = 120;

/// Emits `FRAMES` tiny buffers, sequence-stamped, as fast as they are taken.
struct CountingSource {
    produced: u64,
    arena: SharedArena,
}

impl CountingSource {
    fn new() -> Self {
        Self {
            produced: 0,
            arena: SharedArena::new(64, 32).unwrap(),
        }
    }
}

impl Source for CountingSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.produced >= FRAMES {
            return Ok(ProduceResult::Eos);
        }
        self.arena.reclaim();
        let Some(slot) = self.arena.acquire() else {
            return Ok(ProduceResult::WouldBlock);
        };
        let metadata = Metadata::from_sequence(self.produced);
        self.produced += 1;
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            metadata,
        )))
    }
}

/// Records the sequence number of every buffer it consumes, slowly if asked.
struct RecordingSink {
    seen: Arc<Mutex<Vec<u64>>>,
    delay: Option<Duration>,
}

impl Sink for RecordingSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        if let Some(delay) = self.delay {
            std::thread::sleep(delay);
        }
        self.seen.lock().unwrap().push(ctx.metadata().sequence);
        Ok(())
    }
}

fn recorder() -> Arc<Mutex<Vec<u64>>> {
    Arc::new(Mutex::new(Vec::new()))
}

/// The policy's whole point: under overload the consumer's view stays
/// *current*. The last buffer delivered is the last one produced — a
/// DropNewest link under the same load ends on a stale one.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drop_oldest_link_keeps_the_newest_buffers() {
    let seen = recorder();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new());
    let sink = pipeline.add_sink(
        "slow",
        RecordingSink {
            seen: seen.clone(),
            delay: Some(Duration::from_millis(15)),
        },
    );
    pipeline
        .link_with(src, sink, LinkPolicy::DropOldest)
        .unwrap();

    let executor = Executor::with_config(ExecutorConfig {
        channel_capacity: 4,
        ..ExecutorConfig::default()
    });
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    let seen = seen.lock().unwrap();
    assert!(
        (seen.len() as u64) < FRAMES,
        "the slow sink was supposed to fall behind and lose buffers \
         ({} of {FRAMES} — if it kept up, the eviction path was not exercised)",
        seen.len()
    );
    assert!(!seen.is_empty(), "a lossy branch still delivers");
    assert_eq!(
        *seen.last().unwrap(),
        FRAMES - 1,
        "drop-oldest keeps the freshest data: the last delivery is the last \
         buffer produced"
    );
    assert!(
        seen.windows(2).all(|w| w[0] < w[1]),
        "surviving buffers arrive in order: {seen:?}"
    );
}

/// Same isolation contract as DropNewest: the lossy branch degrades alone,
/// its blocking sibling sees everything.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_slow_drop_oldest_branch_does_not_stall_its_fast_sibling() {
    let fast = Arc::new(AtomicU64::new(0));
    let fast_count = fast.clone();
    let slow = recorder();

    struct CountingSink(Arc<AtomicU64>);
    impl Sink for CountingSink {
        fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new());
    let fast_sink = pipeline.add_sink("fast", CountingSink(fast_count));
    let slow_sink = pipeline.add_sink(
        "slow",
        RecordingSink {
            seen: slow.clone(),
            delay: Some(Duration::from_millis(20)),
        },
    );

    pipeline.link(src, fast_sink).unwrap();
    pipeline
        .link_with(src, slow_sink, LinkPolicy::DropOldest)
        .unwrap();

    let executor = Executor::with_config(ExecutorConfig {
        channel_capacity: 4,
        ..ExecutorConfig::default()
    });
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    assert_eq!(
        fast.load(Ordering::Relaxed),
        FRAMES,
        "the fast branch must receive every buffer, however slow its sibling"
    );
    let slow = slow.lock().unwrap();
    assert!(
        (slow.len() as u64) < FRAMES && !slow.is_empty(),
        "the slow branch drops but still delivers ({} of {FRAMES})",
        slow.len()
    );
}

/// EOS is never dropped or evicted, whatever the policy. Termination is the
/// proof: a lost EOS leaves the sink waiting forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn eos_reaches_a_drop_oldest_branch_even_when_its_channel_is_full() {
    let seen = recorder();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new());
    let sink = pipeline.add_sink(
        "slow",
        RecordingSink {
            seen: seen.clone(),
            delay: Some(Duration::from_millis(10)),
        },
    );
    pipeline
        .link_with(src, sink, LinkPolicy::DropOldest)
        .unwrap();

    let executor = Executor::with_config(ExecutorConfig {
        channel_capacity: 2,
        ..ExecutorConfig::default()
    });
    let handle = executor.start(&mut pipeline).unwrap();

    tokio::time::timeout(Duration::from_secs(20), handle.wait())
        .await
        .expect("the pipeline must terminate: EOS is never dropped")
        .unwrap();
}

/// A deep enough queue never evicts: exact delivery, exact order. The
/// drop-oldest twin of `a_link_can_override_the_channel_capacity`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_drop_oldest_link_with_a_deep_queue_loses_nothing() {
    let seen = recorder();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new());
    let sink = pipeline.add_sink(
        "sink",
        RecordingSink {
            seen: seen.clone(),
            delay: None,
        },
    );
    pipeline
        .link_pads_full(
            src,
            "src",
            sink,
            "sink",
            LinkPolicy::DropOldest,
            Some(FRAMES as usize * 2),
        )
        .unwrap();

    let executor = Executor::with_config(ExecutorConfig {
        channel_capacity: 1,
        ..ExecutorConfig::default()
    });
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    let seen = seen.lock().unwrap();
    let expected: Vec<u64> = (0..FRAMES).collect();
    assert_eq!(
        *seen, expected,
        "a drop-oldest link with room never has to evict"
    );
}

/// #85's contract holds for the leaky channel too: when the only consumer
/// dies, the source must learn it (closed-channel detection) and the run must
/// end instead of spinning into a dead link. The source here produces
/// forever, so termination *is* the detection.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn source_stops_when_a_drop_oldest_consumer_dies() {
    /// Produces forever; only closed-channel detection can end this run.
    struct EndlessSource {
        arena: SharedArena,
    }
    impl Source for EndlessSource {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
            self.arena.reclaim();
            let Some(slot) = self.arena.acquire() else {
                return Ok(ProduceResult::WouldBlock);
            };
            Ok(ProduceResult::OwnBuffer(Buffer::new(
                MemoryHandle::with_len(slot, 8),
                Metadata::from_sequence(0),
            )))
        }
    }

    struct FailingSink {
        consumed: u32,
    }
    impl Sink for FailingSink {
        fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
            self.consumed += 1;
            if self.consumed >= 3 {
                return Err(Error::Element("sink died".into()));
            }
            Ok(())
        }
    }

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "src",
        EndlessSource {
            arena: SharedArena::new(64, 32).unwrap(),
        },
    );
    let sink = pipeline.add_sink("failing", FailingSink { consumed: 0 });
    pipeline
        .link_with(src, sink, LinkPolicy::DropOldest)
        .unwrap();

    let executor = Executor::with_config(ExecutorConfig {
        channel_capacity: 2,
        ..ExecutorConfig::default()
    });
    let handle = executor.start(&mut pipeline).unwrap();

    let outcome = tokio::time::timeout(Duration::from_secs(20), handle.wait())
        .await
        .expect("the run must end when the only consumer is gone");
    assert!(
        outcome.is_err(),
        "a died-sink run reports the failure, not a clean EOS"
    );
}
