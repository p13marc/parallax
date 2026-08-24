//! The `@media` plane, over two real zenoh sessions (#231, #223).
//!
//! `tests/zenoh_roundtrip.rs` shares one session for the whole binary, so
//! everything it exercises is intra-session loopback. That cannot catch what
//! this file is for: a publisher and a subscriber that only know each other
//! through the network, which is the shape the plane is actually deployed in.
//!
//! Two sessions, scouting fully off, explicit `127.0.0.1` endpoints — hermetic
//! on a shared runner, where multicast discovery would find whatever else
//! happens to be running.

#![cfg(feature = "zenoh")]

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::codec::annexb::{self, NalCodec};
use parallax::element::{AsyncSink, AsyncSource, ConsumeContext, ProduceContext, ProduceResult};
use parallax::elements::{ZenohSink, ZenohSrc};
use parallax::format::VideoCodec;
use parallax::memory::SharedArena;
use parallax::metadata::{BufferFlags, Metadata};
use parallax::wire::FrameMeta;

/// Slots big enough for the synthetic access units below.
fn test_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(8192, 64).unwrap())
}

/// An endpoint no other test can collide with.
///
/// A unix socket rather than a TCP port on purpose: nextest gives every test
/// its own process, so a pid-derived port range is not actually unique — two
/// concurrent test processes with nearby pids will pick the same port and one
/// of them fails to bind. A path under `TMPDIR` cannot collide.
fn unique_endpoint() -> (String, std::path::PathBuf) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let path = std::env::temp_dir().join(format!(
        "parallax-zenoh-{}-{}.sock",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    (format!("unixsock-stream/{}", path.display()), path)
}

fn unique_key(tag: &str) -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!(
        "test/v1/h-{:012x}/@media/parallax/cam{}/{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed),
        tag
    )
}

/// A listening session and a session connected to it.
///
/// Both have multicast *and* gossip scouting disabled, so they find each other
/// only through the endpoint we hand them and nothing else on the machine.
async fn session_pair() -> (zenoh::Session, zenoh::Session) {
    let (endpoint, path) = unique_endpoint();
    // The socket file outlives the sessions; clean it up rather than
    // littering TMPDIR across a test run.
    let _ = std::fs::remove_file(&path);

    let mut listen = zenoh::Config::default();
    listen
        .insert_json5("scouting/multicast/enabled", "false")
        .unwrap();
    listen
        .insert_json5("scouting/gossip/enabled", "false")
        .unwrap();
    listen
        .insert_json5("listen/endpoints", &format!("[\"{endpoint}\"]"))
        .unwrap();
    let publisher = zenoh::open(listen).await.expect("open listening session");

    let mut connect = zenoh::Config::default();
    connect
        .insert_json5("scouting/multicast/enabled", "false")
        .unwrap();
    connect
        .insert_json5("scouting/gossip/enabled", "false")
        .unwrap();
    connect
        .insert_json5("connect/endpoints", &format!("[\"{endpoint}\"]"))
        .unwrap();
    let subscriber = zenoh::open(connect).await.expect("open connecting session");

    // Let the transport come up before anyone publishes.
    tokio::time::sleep(Duration::from_millis(300)).await;
    (publisher, subscriber)
}

/// Wait for a freshly declared subscription to reach the other session.
///
/// Over a real transport, declaring a subscriber and publishing are not
/// ordered: the declaration has to propagate before the publisher will match
/// it, and a sample sent into that window is simply gone. Tests that publish
/// exactly one sample notice; tests that publish several accidentally do not.
async fn settle() {
    tokio::time::sleep(Duration::from_millis(300)).await;
}

fn buffer(payload: &[u8], metadata: Metadata) -> Buffer {
    let arena = test_arena();
    arena.reclaim();
    let mut slot = arena.acquire().expect("arena slot");
    slot.data_mut()[..payload.len()].copy_from_slice(payload);
    Buffer::new(MemoryHandle::with_len(slot, payload.len()), metadata)
}

/// SPS, PPS, IDR — a self-contained H.264 keyframe.
fn keyframe_au() -> Vec<u8> {
    let mut v = Vec::new();
    v.extend_from_slice(&[0, 0, 0, 1, 0x67, 0x64, 0x0c, 0x14]);
    v.extend_from_slice(&[0, 0, 0, 1, 0x68, 0xCE, 0x38, 0x80]);
    v.extend_from_slice(&[0, 0, 0, 1, 0x65, 0xAA, 0xBB, 0xCC]);
    v
}

/// A non-IDR slice.
fn delta_au(marker: u8) -> Vec<u8> {
    vec![0, 0, 0, 1, 0x41, 0x9A, marker]
}

async fn produce_one(src: &mut ZenohSrc) -> Option<Buffer> {
    let mut ctx = ProduceContext::without_buffer();
    match tokio::time::timeout(Duration::from_secs(10), src.produce(&mut ctx))
        .await
        .expect("timed out waiting for a sample")
        .expect("produce failed")
    {
        ProduceResult::OwnBuffer(b) => Some(b),
        ProduceResult::WouldBlock => None,
        other => panic!("unexpected produce result: {other:?}"),
    }
}

/// Pull until a buffer actually arrives (the keyframe gate answers
/// `WouldBlock` for everything it drops).
async fn next_buffer(src: &mut ZenohSrc) -> Buffer {
    for _ in 0..32 {
        if let Some(b) = produce_one(src).await {
            return b;
        }
    }
    panic!("no buffer after 32 produce calls");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_access_unit_crosses_two_sessions_intact() {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut src = ZenohSrc::media_with_session(sub_session, &key)
        .await
        .expect("media source");
    let mut sink = ZenohSink::media_with_session(pub_session, &key)
        .await
        .expect("media sink");
    settle().await;

    let au = keyframe_au();
    let mut meta = Metadata::new();
    meta.set_coded_video_dims(VideoCodec::H264, 1920, 1080);
    meta.flags |= BufferFlags::SYNC_POINT;
    meta.pts = parallax::clock::ClockTime::from_nanos(1_000_000);

    sink.consume(&ConsumeContext::new(&buffer(&au, meta)))
        .await
        .expect("publish");

    let received = next_buffer(&mut src).await;

    assert_eq!(
        received.as_bytes(),
        &au[..],
        "payload must be byte-identical"
    );
    let m = received.metadata();
    assert!(m.flags.is_keyframe());
    assert_eq!(m.pts, parallax::clock::ClockTime::from_nanos(1_000_000));
    assert_eq!(m.coded_video_dims(), Some((1920, 1080)));
    assert_eq!(
        m.format,
        Some(parallax::format::MediaFormat::Video(VideoCodec::H264))
    );
    // Joining a live stream is itself a discontinuity.
    assert!(
        m.flags.contains(BufferFlags::DISCONT),
        "the first sample of a subscription is a discontinuity"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn the_gate_holds_until_the_first_keyframe() {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/low");

    let mut src = ZenohSrc::media_with_session(sub_session, &key)
        .await
        .expect("media source");
    let mut sink = ZenohSink::media_with_session(pub_session, &key)
        .await
        .expect("media sink");
    settle().await;

    // Three delta frames, then a keyframe. A decoder must not see the deltas.
    for i in 0..3u8 {
        let meta = Metadata::new();
        sink.consume(&ConsumeContext::new(&buffer(&delta_au(i), meta)))
            .await
            .expect("publish delta");
    }
    let au = keyframe_au();
    let mut meta = Metadata::new();
    meta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&au, meta)))
        .await
        .expect("publish keyframe");

    let received = next_buffer(&mut src).await;
    assert_eq!(
        received.as_bytes(),
        &au[..],
        "the first delivered buffer must be the keyframe, not a delta"
    );
    assert!(src.dropped_pre_keyframe() >= 1, "deltas should be counted");
}

#[tokio::test(flavor = "multi_thread")]
async fn the_publisher_owns_the_sequence_and_a_gap_shows() {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/medium");

    let mut src = ZenohSrc::media_with_session(sub_session, &key)
        .await
        .expect("media source");
    let mut sink = ZenohSink::media_with_session(pub_session, &key)
        .await
        .expect("media sink");
    settle().await;

    // Wildly inconsistent buffer sequence numbers — an encoder rewrites them
    // mid-graph, so the publisher must not trust them.
    for (i, seq) in [0u64, 900, 17].into_iter().enumerate() {
        let mut meta = Metadata::new();
        meta.sequence = seq;
        meta.flags |= BufferFlags::SYNC_POINT;
        let au = if i == 0 {
            keyframe_au()
        } else {
            delta_au(i as u8)
        };
        // Every one is flagged as a keyframe by the buffer, but only the
        // first actually contains an IDR — the publisher judges the bytes.
        sink.consume(&ConsumeContext::new(&buffer(&au, meta)))
            .await
            .expect("publish");
    }

    let first = next_buffer(&mut src).await;
    assert!(first.metadata().flags.is_keyframe());
    assert_eq!(first.metadata().sequence, 0, "publisher-owned, starts at 0");

    let second = next_buffer(&mut src).await;
    assert_eq!(second.metadata().sequence, 1, "monotonic, not the buffer's");
    assert!(
        !second.metadata().flags.contains(BufferFlags::DISCONT),
        "consecutive wire sequences are not a discontinuity"
    );
    assert!(
        !second.metadata().flags.is_keyframe(),
        "a delta frame flagged as a keyframe upstream must not be published as one"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_bare_keyframe_is_repaired_on_the_way_out() {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut src = ZenohSrc::media_with_session(sub_session, &key)
        .await
        .expect("media source");
    let mut sink = ZenohSink::media_with_session(pub_session, &key)
        .await
        .expect("media sink");
    settle().await;

    // First a full keyframe, so the publisher learns the parameter sets.
    let mut meta = Metadata::new();
    meta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&keyframe_au(), meta)))
        .await
        .expect("publish full keyframe");
    let _ = next_buffer(&mut src).await;

    // Then an IDR with no parameter sets, as a camera that only announced
    // them in its SDP would send.
    let bare = vec![0, 0, 0, 1, 0x65, 0xDD, 0xEE];
    assert!(!annexb::has_param_sets(&bare, NalCodec::H264));
    let mut meta = Metadata::new();
    meta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&bare, meta)))
        .await
        .expect("publish bare keyframe");

    let received = next_buffer(&mut src).await;
    assert!(
        annexb::has_param_sets(received.as_bytes(), NalCodec::H264),
        "a keyframe on this plane must be a self-contained decoder entry point"
    );
    assert!(annexb::is_entry_point(received.as_bytes(), NalCodec::H264));
}

#[tokio::test(flavor = "multi_thread")]
async fn caps_are_known_before_a_single_sample_arrives() {
    use parallax::element::AsyncSource;
    use parallax::format::{FormatCaps, VideoCodec};

    let (_pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");

    let src = ZenohSrc::media_with_session(sub_session, &key)
        .await
        .expect("media source");

    // Nothing has been published. `prepare()` would run right here, and it
    // must get a real answer.
    let caps = src.output_media_caps();
    assert_eq!(
        caps.preferred().map(|c| c.format.clone()),
        Some(FormatCaps::Video(VideoCodec::H264)),
        "the codec is a literal chunk of the key; caps must not wait for data"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wildcard_key_is_refused_on_both_sides() {
    let (pub_session, sub_session) = session_pair().await;
    let wild = "test/v1/*/@media/parallax/cam0/video/h264/high";

    assert!(
        ZenohSrc::media_with_session(sub_session, wild)
            .await
            .is_err(),
        "a wildcard subscription would interleave every tier, or every host"
    );
    assert!(
        ZenohSink::media_with_session(pub_session, wild)
            .await
            .is_err(),
        "publishing under a wildcard is a lie about identity"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn media_refuses_the_parallax_plane_s_options() {
    let (pub_session, _sub) = session_pair().await;
    let key = unique_key("video/h264/high");

    let sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    assert!(
        sink.without_metadata().is_err(),
        "@media samples must carry a FrameMeta"
    );

    let sink = ZenohSink::media_with_session(pub_session, &key)
        .await
        .expect("media sink");
    assert!(
        sink.with_forward_custom_keys(&["stanag/klv"]).is_err(),
        "@media carries pixels and nothing else"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_foreign_publisher_does_not_break_the_subscriber() {
    // Something publishing plain bytes on a media key: not conformant, but a
    // subscriber must degrade rather than die.
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut src = ZenohSrc::media_with_session(sub_session, &key)
        .await
        .expect("media source");
    settle().await;

    pub_session
        .put(&key, b"not an access unit".to_vec())
        .await
        .expect("raw put");

    let received = next_buffer(&mut src).await;
    assert_eq!(received.as_bytes(), b"not an access unit");
}

#[tokio::test(flavor = "multi_thread")]
async fn a_keyframe_request_is_raised_upstream_and_coalesced() {
    let (pub_session, _sub) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut sink = ZenohSink::media_with_session(pub_session, &key)
        .await
        .expect("media sink");

    assert!(
        sink.take_upstream_event().is_none(),
        "nothing to send before anyone asks"
    );

    // A viewer under loss asks far faster than a GOP; those must collapse.
    sink.request_keyframe();
    sink.request_keyframe();
    sink.request_keyframe();
    assert_eq!(sink.keyframe_requests(), 1, "requests coalesce");

    match sink.take_upstream_event() {
        Some(parallax::event::Event::ForceKeyUnit(e)) => {
            assert!(e.all_headers, "a joining viewer needs the parameter sets");
            assert_eq!(e.count, 1);
        }
        other => panic!("expected a ForceKeyUnit, got {other:?}"),
    }

    // Collected, so the next ask is a fresh one.
    assert!(sink.take_upstream_event().is_none());
    sink.request_keyframe();
    assert_eq!(sink.keyframe_requests(), 2);
}

#[test]
fn frame_meta_matches_the_checked_in_corpus() {
    // The binding artifact between parallax's FrameMeta and zensight's: if
    // either side's encoder drifts, these bytes stop decoding.
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/framemeta");
    let entries = std::fs::read_dir(dir).expect("corpus directory");

    let mut seen = 0;
    for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|e| e.to_str()) != Some("cbor") {
            continue;
        }
        let bytes = std::fs::read(&path).expect("corpus vector");
        let decoded = FrameMeta::from_cbor(&bytes)
            .unwrap_or_else(|| panic!("{} failed to decode", path.display()));
        // Re-encoding must reproduce the vector byte for byte.
        assert_eq!(
            decoded.to_cbor(),
            bytes,
            "{} did not round-trip",
            path.display()
        );
        seen += 1;
    }
    assert!(seen >= 2, "expected a corpus, found {seen} vectors");
}
