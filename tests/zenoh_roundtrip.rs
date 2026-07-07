//! Integration tests for the zenoh v2 wire format (#11).
//!
//! Exercises `ZenohSink` → zenoh → `ZenohSrc` in-process on a shared session
//! (multicast scouting disabled so tests can't cross-talk with LAN peers or
//! each other). The wire contract under test: payload = raw bytes,
//! attachment = versioned rkyv `WireMetadata`, sample encoding derived from
//! the buffer's media format, graceful fallback for foreign publishers.
#![cfg(feature = "zenoh")]

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{AsyncSink, AsyncSource, ConsumeContext, ProduceContext, ProduceResult};
use parallax::elements::network::zenoh_wire::{KEY_EXPR_META, WIRE_MAGIC};
use parallax::elements::{ZenohSink, ZenohSrc};
use parallax::format::{MediaFormat, VideoCodec};
use parallax::memory::SharedArena;
use parallax::metadata::{BufferFlags, Metadata};

fn test_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(4096, 64).unwrap())
}

/// One session for the whole test binary; scouting disabled so no external
/// zenoh peers are discovered.
async fn session() -> zenoh::Session {
    static SESSION: tokio::sync::OnceCell<zenoh::Session> = tokio::sync::OnceCell::const_new();
    SESSION
        .get_or_init(|| async {
            let mut config = zenoh::Config::default();
            config
                .insert_json5("scouting/multicast/enabled", "false")
                .unwrap();
            config
                .insert_json5("scouting/gossip/enabled", "false")
                .unwrap();
            zenoh::open(config).await.expect("open zenoh session")
        })
        .await
        .clone()
}

/// Unique key expression per test (parallel tests must not cross-talk).
fn unique_key(tag: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "parallax/test/{}/{}/{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
        tag
    )
}

fn payload_buffer(payload: &[u8], metadata: Metadata) -> Buffer {
    let arena = test_arena();
    arena.reclaim();
    let mut slot = arena.acquire().expect("arena slot");
    slot.data_mut()[..payload.len()].copy_from_slice(payload);
    Buffer::new(MemoryHandle::with_len(slot, payload.len()), metadata)
}

async fn produce_one(src: &mut ZenohSrc) -> Buffer {
    let mut ctx = ProduceContext::without_buffer();
    match tokio::time::timeout(Duration::from_secs(10), src.produce(&mut ctx))
        .await
        .expect("timed out waiting for sample")
        .expect("produce failed")
    {
        ProduceResult::OwnBuffer(buffer) => buffer,
        other => panic!("expected OwnBuffer, got {other:?}"),
    }
}

/// Full metadata round-trip: pts/dts/duration/sequence/flags/format/KLV all
/// survive the hop; the sample encoding is derived from the media format.
#[tokio::test(flavor = "multi_thread")]
async fn metadata_roundtrip_with_encoding() {
    let session = session().await;
    let key = unique_key("roundtrip");

    let mut src = ZenohSrc::with_session(session.clone(), key.clone())
        .await
        .unwrap();

    // Raw subscriber to inspect the wire sample itself.
    let (raw_tx, mut raw_rx) = tokio::sync::mpsc::unbounded_channel();
    let _raw_sub = session
        .declare_subscriber(&key)
        .callback(move |sample| {
            let _ = raw_tx.send(sample);
        })
        .await
        .unwrap();

    let mut sink = ZenohSink::with_session(session.clone(), key.clone())
        .await
        .unwrap()
        .with_forward_custom_keys(&["stanag/klv"]);

    let mut metadata = Metadata::new();
    metadata.pts = ClockTime::from_millis(1234);
    metadata.dts = ClockTime::from_millis(1200);
    metadata.duration = ClockTime::from_millis(33);
    metadata.sequence = 0;
    metadata.stream_id = 9;
    metadata.flags = BufferFlags::SYNC_POINT;
    metadata.offset = Some(777);
    metadata.format = Some(MediaFormat::Video(VideoCodec::H264));
    metadata.set_klv(vec![0x06, 0x0E, 0x2B, 0x34]);

    let buffer = payload_buffer(b"fake-access-unit", metadata);
    sink.consume(&ConsumeContext::new(&buffer)).await.unwrap();

    // Parallax side: full metadata restored.
    let received = produce_one(&mut src).await;
    assert_eq!(received.as_bytes(), b"fake-access-unit");
    let m = received.metadata();
    assert_eq!(m.pts, ClockTime::from_millis(1234));
    assert_eq!(m.dts, ClockTime::from_millis(1200));
    assert_eq!(m.duration, ClockTime::from_millis(33));
    assert_eq!(m.sequence, 0);
    assert_eq!(m.stream_id, 9);
    assert!(m.flags.contains(BufferFlags::SYNC_POINT));
    assert!(!m.flags.contains(BufferFlags::DISCONT), "first sample");
    assert_eq!(m.offset, Some(777));
    assert_eq!(m.format, Some(MediaFormat::Video(VideoCodec::H264)));
    assert_eq!(m.klv(), Some(&[0x06, 0x0E, 0x2B, 0x34][..]));

    // Wire side: encoding derived from format; payload is the raw bytes;
    // attachment carries the magic.
    let sample = tokio::time::timeout(Duration::from_secs(5), raw_rx.recv())
        .await
        .expect("raw sample timeout")
        .expect("raw channel closed");
    assert_eq!(sample.encoding().to_string(), "video/h264");
    assert_eq!(&sample.payload().to_bytes()[..], b"fake-access-unit");
    let attachment = sample.attachment().expect("attachment present");
    assert_eq!(&attachment.to_bytes()[..2], &WIRE_MAGIC);
}

/// A gap in the published sequence numbers flags DISCONT on the receiver.
#[tokio::test(flavor = "multi_thread")]
async fn wire_sequence_gap_sets_discont() {
    let session = session().await;
    let key = unique_key("discont");

    let mut src = ZenohSrc::with_session(session.clone(), key.clone())
        .await
        .unwrap();
    let mut sink = ZenohSink::with_session(session.clone(), key.clone())
        .await
        .unwrap();

    for seq in [0u64, 1, 5] {
        let buffer = payload_buffer(b"x", Metadata::from_sequence(seq));
        sink.consume(&ConsumeContext::new(&buffer)).await.unwrap();
    }

    assert!(
        !produce_one(&mut src)
            .await
            .metadata()
            .flags
            .contains(BufferFlags::DISCONT)
    );
    assert!(
        !produce_one(&mut src)
            .await
            .metadata()
            .flags
            .contains(BufferFlags::DISCONT)
    );
    let third = produce_one(&mut src).await;
    assert_eq!(third.metadata().sequence, 5);
    assert!(
        third.metadata().flags.contains(BufferFlags::DISCONT),
        "sequence 1 -> 5 gap must flag DISCONT"
    );
}

/// Samples from foreign publishers (no attachment) fall back to fabricated
/// sequence metadata without erroring.
#[tokio::test(flavor = "multi_thread")]
async fn foreign_publisher_fallback() {
    let session = session().await;
    let key = unique_key("foreign");

    let mut src = ZenohSrc::with_session(session.clone(), key.clone())
        .await
        .unwrap();

    session.put(&key, b"plain".to_vec()).await.unwrap();

    let received = produce_one(&mut src).await;
    assert_eq!(received.as_bytes(), b"plain");
    assert_eq!(received.metadata().sequence, 0, "fabricated sequence");
    assert!(received.metadata().pts == ClockTime::ZERO);
}

/// An attachment with the right magic but unknown version also falls back.
#[tokio::test(flavor = "multi_thread")]
async fn unknown_wire_version_fallback() {
    let session = session().await;
    let key = unique_key("version");

    let mut src = ZenohSrc::with_session(session.clone(), key.clone())
        .await
        .unwrap();

    session
        .put(&key, b"future".to_vec())
        .attachment(vec![WIRE_MAGIC[0], WIRE_MAGIC[1], 99, 1, 2, 3])
        .await
        .unwrap();

    let received = produce_one(&mut src).await;
    assert_eq!(received.as_bytes(), b"future");
    assert_eq!(received.metadata().sequence, 0, "fabricated sequence");
}

/// With a wildcard subscription, the concrete key a sample arrived on is
/// stored under the KEY_EXPR_META metadata key.
#[tokio::test(flavor = "multi_thread")]
async fn wildcard_subscription_captures_key_expr() {
    let session = session().await;
    let prefix = unique_key("wild");
    let concrete = format!("{prefix}/camera0/h264");

    let mut src = ZenohSrc::with_session(session.clone(), format!("{prefix}/**"))
        .await
        .unwrap();
    let mut sink = ZenohSink::with_session(session.clone(), concrete.clone())
        .await
        .unwrap();

    let buffer = payload_buffer(b"y", Metadata::from_sequence(0));
    sink.consume(&ConsumeContext::new(&buffer)).await.unwrap();

    let received = produce_one(&mut src).await;
    assert_eq!(
        received.metadata().get::<String>(KEY_EXPR_META),
        Some(&concrete)
    );
}

/// Interop mode: without_metadata() publishes attachment-less samples.
#[tokio::test(flavor = "multi_thread")]
async fn without_metadata_mode_omits_attachment() {
    let session = session().await;
    let key = unique_key("nometa");

    let (raw_tx, mut raw_rx) = tokio::sync::mpsc::unbounded_channel();
    let _raw_sub = session
        .declare_subscriber(&key)
        .callback(move |sample| {
            let _ = raw_tx.send(sample);
        })
        .await
        .unwrap();

    let mut sink = ZenohSink::with_session(session.clone(), key.clone())
        .await
        .unwrap()
        .without_metadata();

    let buffer = payload_buffer(b"raw-only", Metadata::from_sequence(3));
    sink.consume(&ConsumeContext::new(&buffer)).await.unwrap();

    let sample = tokio::time::timeout(Duration::from_secs(5), raw_rx.recv())
        .await
        .expect("raw sample timeout")
        .expect("raw channel closed");
    assert!(sample.attachment().is_none(), "interop mode: no attachment");
    assert_eq!(&sample.payload().to_bytes()[..], b"raw-only");
}
