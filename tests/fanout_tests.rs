//! Integration tests for per-link fan-out policy (#39).
//!
//! Fan-out already worked — src-pads are 1:N and buffers are refcounted clones —
//! but every branch was awaited into a bounded channel, so a persistently slow
//! branch filled its slots, blocked the send, and stalled the source *and every
//! sibling*. The failure is inverted from what anyone predicts: the cheap 2 fps
//! preview drags down the full-rate H.264 branch.
//!
//! Nothing in this repo tested that before. These do.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, ExecutorConfig, LinkPolicy, Pipeline};

const FRAMES: u64 = 120;

/// Emits `FRAMES` tiny buffers as fast as the pipeline will take them.
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

/// Counts what it receives, optionally taking its sweet time about it.
struct CountingSink {
    received: Arc<AtomicU64>,
    delay: Option<Duration>,
}

impl Sink for CountingSink {
    fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
        if let Some(delay) = self.delay {
            std::thread::sleep(delay);
        }
        self.received.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn counter() -> Arc<AtomicU64> {
    Arc::new(AtomicU64::new(0))
}

/// The simulcast prerequisite: a deliberately slow *lossy* branch must not hold
/// up its fast sibling. The fast branch sees every frame; the slow one drops.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_slow_lossy_branch_does_not_stall_its_fast_sibling() {
    let fast = counter();
    let slow = counter();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new());
    let fast_sink = pipeline.add_sink(
        "fast",
        CountingSink {
            received: fast.clone(),
            delay: None,
        },
    );
    let slow_sink = pipeline.add_sink(
        "slow",
        CountingSink {
            received: slow.clone(),
            delay: Some(Duration::from_millis(20)),
        },
    );

    pipeline.link(src, fast_sink).unwrap();
    pipeline.link_lossy(src, slow_sink).unwrap();

    let executor = Executor::with_config(ExecutorConfig {
        channel_capacity: 4,
        ..ExecutorConfig::default()
    });
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    let fast = fast.load(Ordering::Relaxed);
    let slow = slow.load(Ordering::Relaxed);

    assert_eq!(
        fast, FRAMES,
        "the fast branch must receive every buffer, however slow its sibling is"
    );
    assert!(
        slow < FRAMES,
        "the slow branch was supposed to fall behind and drop ({slow} of {FRAMES} — \
         if it kept up, the test is not exercising the drop path)"
    );
    assert!(
        slow > 0,
        "a lossy branch still delivers what it can keep up with"
    );
}

/// The other half of the contract: `Block` still back-pressures, so we have not
/// quietly made everything lossy. Both branches get everything.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_blocking_branch_still_back_pressures() {
    let fast = counter();
    let slow = counter();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new());
    let fast_sink = pipeline.add_sink(
        "fast",
        CountingSink {
            received: fast.clone(),
            delay: None,
        },
    );
    let slow_sink = pipeline.add_sink(
        "slow",
        CountingSink {
            received: slow.clone(),
            delay: Some(Duration::from_millis(2)),
        },
    );

    pipeline.link(src, fast_sink).unwrap();
    pipeline.link(src, slow_sink).unwrap(); // Block: the default

    let executor = Executor::with_config(ExecutorConfig {
        channel_capacity: 4,
        ..ExecutorConfig::default()
    });
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    assert_eq!(fast.load(Ordering::Relaxed), FRAMES);
    assert_eq!(
        slow.load(Ordering::Relaxed),
        FRAMES,
        "a Block link must lose nothing — the source waits for it"
    );
}

/// EOS is never dropped, whatever the policy. A sink that missed it would wait
/// forever, so the test is simply that a lossy pipeline terminates at all —
/// which `handle.wait()` returning proves.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn eos_reaches_a_lossy_branch_even_when_its_channel_is_full() {
    let slow = counter();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new());
    let slow_sink = pipeline.add_sink(
        "slow",
        CountingSink {
            received: slow.clone(),
            delay: Some(Duration::from_millis(10)),
        },
    );
    pipeline.link_lossy(src, slow_sink).unwrap();

    let executor = Executor::with_config(ExecutorConfig {
        channel_capacity: 2,
        ..ExecutorConfig::default()
    });
    let handle = executor.start(&mut pipeline).unwrap();

    // If EOS were dropped when the channel was full, this would hang.
    tokio::time::timeout(Duration::from_secs(20), handle.wait())
        .await
        .expect("the pipeline must terminate: EOS is never dropped")
        .unwrap();
}

/// Per-link capacity: a branch can be given a deeper queue than the executor
/// default, which is how a bursty branch absorbs a burst without dropping.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_link_can_override_the_channel_capacity() {
    let received = counter();

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", CountingSource::new());
    let sink = pipeline.add_sink(
        "sink",
        CountingSink {
            received: received.clone(),
            delay: None,
        },
    );
    pipeline
        .link_pads_full(
            src,
            "src",
            sink,
            "sink",
            LinkPolicy::Drop,
            Some(FRAMES as usize * 2), // deep enough that nothing can be dropped
        )
        .unwrap();

    let executor = Executor::with_config(ExecutorConfig {
        channel_capacity: 1, // the default would drop heavily
        ..ExecutorConfig::default()
    });
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await.unwrap();

    assert_eq!(
        received.load(Ordering::Relaxed),
        FRAMES,
        "a lossy link with a deep enough queue never has to drop"
    );
}
