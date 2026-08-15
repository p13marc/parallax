//! #179: the end-to-end IpcSink → IpcSrc round trip over the descriptor ring.
//!
//! This test did not exist before, which is how the old socket-message path
//! shipped a latent bug: the sink registered its *own* placeholder arena but
//! sent slot refs from the upstream element's arena — ids that could never
//! match once #178 randomized them. The buffers here deliberately come from
//! the test source's own arena, so register-on-first-sight is what makes the
//! transfer work at all.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{AsyncSink, ConsumeContext, ProduceContext, ProduceResult, Source};
use parallax::elements::{IpcSink, IpcSrc};
use parallax::error::Result;
use parallax::format::{Framerate, MediaFormat, PixelFormat, VideoFormat};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};

type Received = Arc<Mutex<Vec<(Vec<u8>, Metadata)>>>;

/// A finite source producing `total` stamped buffers from its own arena.
struct StampedSource {
    arena: &'static SharedArena,
    produced: u64,
    total: u64,
}

impl Source for StampedSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.produced >= self.total {
            return Ok(ProduceResult::Eos);
        }
        self.arena.reclaim();
        let Some(mut slot) = self.arena.acquire() else {
            return Ok(ProduceResult::WouldBlock);
        };
        let seq = self.produced;
        self.produced += 1;

        slot.data_mut()[..8].copy_from_slice(&seq.to_ne_bytes());

        let mut meta = Metadata::from_sequence(seq);
        meta.pts = ClockTime::from_nanos(seq * 33_000_000);
        meta.format = Some(MediaFormat::VideoRaw(VideoFormat {
            width: 320,
            height: 240,
            pixel_format: PixelFormat::I420,
            framerate: Framerate::new(30, 1),
        }));
        // Custom metadata on every third buffer: exercises the overflow
        // path interleaved with descriptor-only buffers.
        if seq.is_multiple_of(3) {
            meta.set_klv(vec![0x4B, seq as u8, (seq >> 1) as u8]);
        }

        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            meta,
        )))
    }
}

/// Collects `(payload, metadata)` pairs, optionally slowly.
struct Collector {
    got: Received,
    delay: Duration,
}

impl AsyncSink for Collector {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        if !self.delay.is_zero() {
            tokio::time::sleep(self.delay).await;
        }
        let buffer = ctx.buffer();
        self.got
            .lock()
            .unwrap()
            .push((buffer.as_bytes().to_vec(), buffer.metadata().clone()));
        Ok(())
    }
}

async fn run_roundtrip(
    sock: std::path::PathBuf,
    arena: &'static SharedArena,
    total: u64,
    sink: IpcSink,
    delay: Duration,
) -> Vec<(Vec<u8>, Metadata)> {
    let got = Arc::new(Mutex::new(Vec::new()));

    // Pipeline A: source → IpcSink (server side of the socket).
    let mut sender = Pipeline::new();
    let src = sender.add_source(
        "src",
        StampedSource {
            arena,
            produced: 0,
            total,
        },
    );
    let snk = sender.add_async_sink("ipcsink", sink);
    sender.link(src, snk).unwrap();

    // Pipeline B: IpcSrc (client) → collector.
    let mut receiver = Pipeline::new();
    let ipc_src = receiver.add_async_source("ipcsrc", IpcSrc::new(&sock));
    let col = receiver.add_async_sink(
        "collector",
        Collector {
            got: got.clone(),
            delay,
        },
    );
    receiver.link(ipc_src, col).unwrap();

    let ea = Executor::new();
    let eb = Executor::new();
    let ha = ea.start(&mut sender).unwrap();
    let hb = eb.start(&mut receiver).unwrap();

    let (ra, rb) = tokio::join!(
        tokio::time::timeout(Duration::from_secs(30), ha.wait()),
        tokio::time::timeout(Duration::from_secs(30), hb.wait()),
    );
    rb.expect("receiver pipeline hung: EOS did not propagate over the ring")
        .expect("receiver pipeline failed");
    ra.expect("sender pipeline hung")
        .expect("sender pipeline failed");

    Arc::try_unwrap(got).unwrap().into_inner().unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn buffers_metadata_and_eos_cross_the_ring() {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    let arena = ARENA.get_or_init(|| SharedArena::new(64, 64).unwrap());

    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("roundtrip.sock");
    const N: u64 = 100;

    let got = run_roundtrip(sock.clone(), arena, N, IpcSink::new(&sock), Duration::ZERO).await;

    assert_eq!(got.len(), N as usize, "every buffer must arrive");
    for (i, (bytes, meta)) in got.iter().enumerate() {
        let seq = u64::from_ne_bytes(bytes[..8].try_into().unwrap());
        assert_eq!(seq, i as u64, "payload bytes intact and in order");
        assert_eq!(meta.sequence, i as u64);
        assert_eq!(meta.pts.nanos(), i as u64 * 33_000_000, "pts crosses");
        match &meta.format {
            Some(MediaFormat::VideoRaw(v)) => {
                assert_eq!((v.width, v.height), (320, 240), "format crosses");
                assert_eq!(v.framerate, Framerate::new(30, 1));
            }
            other => panic!("format did not cross: {other:?}"),
        }
        if i.is_multiple_of(3) {
            let klv = meta
                .get_bytes("stanag/klv")
                .expect("klv overflow metadata crosses");
            assert_eq!(klv, &[0x4B, i as u8, (i >> 1) as u8][..]);
        } else {
            assert!(meta.get_bytes("stanag/klv").is_none());
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_slow_receiver_backpressures_without_stale_refs() {
    // Small ring + slow consumer: the sink's in-flight bound and pin
    // protocol under real pressure. Every payload must still arrive
    // intact — a pin released early would surface as a "stale ipc slot
    // ref" error and kill the receiver pipeline.
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    let arena = ARENA.get_or_init(|| SharedArena::new(64, 64).unwrap());

    let dir = tempfile::tempdir().unwrap();
    let sock = dir.path().join("slow.sock");
    const N: u64 = 64;

    let got = run_roundtrip(
        sock.clone(),
        arena,
        N,
        IpcSink::new(&sock).with_capacity(8),
        Duration::from_millis(5),
    )
    .await;

    assert_eq!(got.len(), N as usize);
    for (i, (bytes, meta)) in got.iter().enumerate() {
        assert_eq!(u64::from_ne_bytes(bytes[..8].try_into().unwrap()), i as u64);
        assert_eq!(meta.sequence, i as u64);
    }
}
