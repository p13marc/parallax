//! Two real zenoh sessions: the `@media` plane (#231, #223) and the
//! cross-machine end-of-stream path (#224).
//!
//! `tests/zenoh_roundtrip.rs` shares one session for the whole binary, so
//! everything it exercises is intra-session loopback. That cannot catch what
//! this file is for: a publisher and a subscriber that only know each other
//! through the network, which is the shape the plane is actually deployed in.
//!
//! Two sessions, scouting fully off, explicit unix-socket endpoints —
//! hermetic on a shared runner, where multicast discovery would find whatever
//! else happens to be running.
//!
//! The #224 tests live here rather than in a file of their own because this is
//! where the two-session harness is, and a terminal that only ever crossed a
//! loopback session would prove nothing about ordering.

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

/// A listening session and a session connected to it, from two given configs.
///
/// Both get multicast *and* gossip scouting disabled and a unique unix-socket
/// endpoint, so they find each other only through what we hand them and
/// nothing else on the machine.
async fn session_pair_from(
    publisher: zenoh::Config,
    subscriber: zenoh::Config,
) -> (zenoh::Session, zenoh::Session) {
    let (endpoint, path) = unique_endpoint();
    // The socket file outlives the sessions; clean it up rather than
    // littering TMPDIR across a test run.
    let _ = std::fs::remove_file(&path);

    let mut listen = publisher;
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

    let mut connect = subscriber;
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

/// The pair every test in this file runs on.
///
/// The publisher runs **the shipped config** — `session_config()`, timestamping
/// on (#245) — so these tests exercise what actually ships rather than a
/// hand-rolled default.
///
/// The subscriber is deliberately left on zenoh's own default, which for a peer
/// means timestamping **off**. Do not "fix" that: a session with timestamping on
/// stamps unstamped *incoming* samples with its own HLC before delivering them
/// (`treat_timestamp!` in zenoh's `route_data`), so with both ends on,
/// `sample.timestamp().is_some()` would pass against a publisher that stamps
/// nothing. Off, any timestamp seen in this file provably came from the
/// publisher.
async fn session_pair() -> (zenoh::Session, zenoh::Session) {
    session_pair_from(
        parallax::elements::network::session_config().expect("parallax session config"),
        zenoh::Config::default(),
    )
    .await
}

/// Wait until a subscription declared on `subscriber` is genuinely matched by
/// `publisher`, rather than hoping a fixed sleep was long enough.
///
/// Over a real transport, declaring a subscriber and publishing are not
/// ordered: the declaration has to propagate before the publisher will match
/// it, and a sample sent into that window is simply **gone** — this plane is
/// best-effort and drop-by-contract. Tests that publish exactly one sample
/// notice; tests that publish several used to accidentally not.
///
/// A 300 ms sleep used to stand in for that, and it is what made
/// `the_gate_holds_until_the_first_keyframe` and
/// `a_stalled_consumer_bounds_memory_instead_of_growing_it` fail on a busy CI
/// runner: the whole burst went into the unmatched window, leaving zero samples
/// received and a failure that said nothing about the behaviour under test.
///
/// So this probes instead. It declares one more subscriber, on a throwaway key,
/// and publishes to it until something arrives. The load-bearing detail is the
/// **ordering**: the probe is declared *after* whatever the test declared, and
/// both travel the same link, which delivers declarations in order. So a
/// publisher that has matched the probe has necessarily already processed the
/// test's declaration.
///
/// Also faster than the sleep it replaced — it normally returns on the first
/// round trip, in single-digit milliseconds.
async fn settle(publisher: &zenoh::Session, subscriber: &zenoh::Session) {
    static PROBE: AtomicU64 = AtomicU64::new(0);
    let key = format!(
        "parallax-settle/{}/{}",
        std::process::id(),
        PROBE.fetch_add(1, Ordering::Relaxed)
    );

    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let _probe = subscriber
        .declare_subscriber(&key)
        .callback(move |_| {
            let _ = tx.send(());
        })
        .await
        .expect("probe subscriber");

    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        // Ignore the error: a put with no matching route is not a failure, it
        // is exactly the state we are waiting to leave.
        let _ = publisher.put(&key, vec![0u8]).await;
        if tokio::time::timeout(Duration::from_millis(20), rx.recv())
            .await
            .is_ok()
        {
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the two sessions never matched a probe subscription in 10 s"
        );
    }
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

    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

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

    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

    // Three delta frames, then a keyframe. A decoder must not see the deltas.
    //
    // Republished until a round lands, because this plane is best-effort and
    // drop-by-contract and really does drop. Instrumenting a starved run showed
    // the sink reporting four samples sent and the subscriber receiving two,
    // with `lost_wire == 0` — no wire gap, just samples the transport was
    // entitled to discard. A fixed burst that must all arrive is asserting
    // something `@media` never promised, and that is what timed this test out
    // on CI. What is under test is the *gate*, not delivery.
    let au = keyframe_au();
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let received = loop {
        for i in 0..3u8 {
            let meta = Metadata::new();
            sink.consume(&ConsumeContext::new(&buffer(&delta_au(i), meta)))
                .await
                .expect("publish delta");
        }
        let mut meta = Metadata::new();
        meta.flags |= BufferFlags::SYNC_POINT;
        sink.consume(&ConsumeContext::new(&buffer(&au, meta)))
            .await
            .expect("publish keyframe");

        // Safe to abandon: `LeakyReceiver::recv` is cancel-safe, and
        // `leaky::tests::cancelled_recv_loses_nothing` pins that.
        let mut ctx = ProduceContext::without_buffer();
        match tokio::time::timeout(Duration::from_millis(500), src.produce(&mut ctx)).await {
            Ok(Ok(ProduceResult::OwnBuffer(b))) => break b,
            Ok(Err(e)) => panic!("produce failed: {e}"),
            _ => {}
        }
        assert!(
            std::time::Instant::now() < deadline,
            "no round of deltas-then-keyframe was ever delivered"
        );
    };

    assert_eq!(
        received.as_bytes(),
        &au[..],
        "the first delivered buffer must be the keyframe, not a delta"
    );
    // Only meaningful if a delta actually arrived — one the transport dropped
    // was never the gate's to count.
    if src.control().samples_received() > 1 {
        assert!(
            src.dropped_pre_keyframe() >= 1,
            "a delta that arrived before the keyframe must be counted, not delivered"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn the_publisher_owns_the_sequence_and_a_gap_shows() {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/medium");

    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

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

    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

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

    let src = ZenohSrc::media_with_session(sub_session.clone(), &key)
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
        ZenohSrc::media_with_session(sub_session.clone(), wild)
            .await
            .is_err(),
        "a wildcard subscription would interleave every tier, or every host"
    );
    assert!(
        ZenohSink::media_with_session(pub_session.clone(), wild)
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

    let sink = ZenohSink::media_with_session(pub_session.clone(), &key)
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

    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    settle(&pub_session, &sub_session).await;

    pub_session
        .put(&key, b"not an access unit".to_vec())
        .await
        .expect("raw put");

    let received = next_buffer(&mut src).await;
    assert_eq!(received.as_bytes(), b"not an access unit");
}

/// #234: the publisher learns when someone starts watching.
///
/// This test only means anything on a build with `zenoh-unstable` — matching
/// status lives behind it — which is why that feature is now in the CI test
/// job's sensor combo rather than only in clippy and doc.
#[cfg(feature = "zenoh-unstable")]
#[tokio::test(flavor = "multi_thread")]
async fn a_viewer_arriving_raises_the_demand_signal_and_asks_for_a_keyframe() {
    use parallax::elements::ZenohDemandEvent;

    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    let demand = sink.control();

    // The publisher is lazy, so declare it (and the listener) up front.
    sink.ensure_publisher().await.expect("publisher");
    settle(&pub_session, &sub_session).await;

    assert!(!demand.has_viewers(), "nobody is watching yet");
    assert_eq!(demand.rising_edges(), 0);

    // A viewer appears.
    let _src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");

    let event = demand
        .event_timeout(Duration::from_secs(10))
        .await
        .expect("a demand event");
    assert_eq!(event, ZenohDemandEvent::ViewersArrived);
    assert!(demand.has_viewers());
    assert_eq!(demand.rising_edges(), 1);

    // The arrival asks upstream for an entry point on the viewer's behalf —
    // that is the whole point, so a late joiner does not wait out a GOP.
    assert_eq!(demand.keyframe_requests(), 1);
    match sink.take_upstream_event() {
        Some(parallax::event::Event::ForceKeyUnit(e)) => {
            assert!(e.all_headers, "a joining viewer needs the parameter sets");
        }
        other => panic!("expected a ForceKeyUnit, got {other:?}"),
    }
}

/// The falling edge, which is what an idle reaper runs on.
#[cfg(feature = "zenoh-unstable")]
#[tokio::test(flavor = "multi_thread")]
async fn the_last_viewer_leaving_is_observable() {
    use parallax::elements::ZenohDemandEvent;

    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/low");

    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    let demand = sink.control();
    sink.ensure_publisher().await.expect("publisher");
    settle(&pub_session, &sub_session).await;

    let src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    assert_eq!(
        demand.event_timeout(Duration::from_secs(10)).await,
        Some(ZenohDemandEvent::ViewersArrived)
    );

    drop(src);
    assert_eq!(
        demand.event_timeout(Duration::from_secs(10)).await,
        Some(ZenohDemandEvent::ViewersGone),
        "the last viewer leaving must be observable"
    );
    assert!(!demand.has_viewers());
    assert_eq!(demand.falling_edges(), 1);
}

/// Publish counters live on the handle, which outlives `start()`.
#[tokio::test(flavor = "multi_thread")]
async fn the_handle_counts_what_was_published() {
    let (pub_session, _sub) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    let stats = sink.control();

    let mut meta = Metadata::new();
    meta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&keyframe_au(), meta)))
        .await
        .expect("publish");

    assert_eq!(stats.samples_sent(), 1);
    assert!(stats.bytes_sent() > 0);
    // Meaningful since #246: the error path increments this now. It used to
    // have no writer at all, so the assertion passed vacuously.
    assert_eq!(stats.publish_errors(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_keyframe_request_is_raised_upstream_and_coalesced() {
    let (pub_session, _sub) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
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

#[tokio::test(flavor = "multi_thread")]
async fn the_three_drop_counters_mean_different_things() {
    // #225: they point at different machines. `dropped_queue` says this host
    // is too slow, `dropped_no_slot` says this pipeline's arena is undersized,
    // `lost_wire` says the samples never arrived. One number cannot say that.
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    let stats = src.control();
    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

    // A clean keyframe, then a delta: nothing lost, nothing dropped.
    for au in [keyframe_au(), delta_au(0)] {
        let mut meta = Metadata::new();
        meta.flags |= BufferFlags::SYNC_POINT;
        sink.consume(&ConsumeContext::new(&buffer(&au, meta)))
            .await
            .expect("publish");
    }
    let _ = next_buffer(&mut src).await;
    let _ = next_buffer(&mut src).await;

    assert_eq!(stats.lost_wire(), 0, "nothing was lost");
    assert_eq!(stats.dropped_queue(), 0, "the queue kept up");
    assert_eq!(stats.dropped_no_slot(), 0, "slots were available");
    assert_eq!(stats.samples_received(), 2);
    assert!(stats.bytes_received() > 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn a_stalled_consumer_bounds_memory_instead_of_growing_it() {
    // The receive queue used to be unbounded: a pipeline that stopped pulling
    // grew memory until the process died. It is now drop-oldest, and says so.
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/low");

    let src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    let stats = src.control();
    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

    // Publish far more than the queue holds, and never pull — and keep
    // publishing until the queue visibly evicts.
    //
    // The retry is what makes this test honest rather than lucky. Declaring a
    // subscriber and publishing are not ordered (see `settle`), and a sample
    // sent before the declaration propagates is simply gone — so under CPU
    // contention a single fixed-size burst can arrive *entirely* in that
    // window, leaving `samples_received == 0` and a failure that says nothing
    // about the bound under test. Republishing lets a slow handshake catch up.
    // The first burst is enough in the normal case, and the loop exits in a
    // few milliseconds.
    let deadline = std::time::Instant::now() + Duration::from_secs(20);
    let mut published = 0u32;
    while stats.dropped_queue() == 0 && std::time::Instant::now() < deadline {
        for i in 0..64u8 {
            let mut meta = Metadata::new();
            meta.flags |= BufferFlags::SYNC_POINT;
            sink.consume(&ConsumeContext::new(&buffer(&delta_au(i), meta)))
                .await
                .expect("publish");
            published += 1;
        }
        // Let the subscriber's callback run before deciding to publish more.
        tokio::time::sleep(Duration::from_millis(20)).await;
    }

    assert!(
        stats.dropped_queue() > 0,
        "a stalled consumer must evict, not accumulate \
         (published {published}, dropped_queue = {}, samples_received = {} — if \
          the last is 0 the subscription never matched, not the queue unbounded)",
        stats.dropped_queue(),
        stats.samples_received()
    );
    // And the eviction is attributed correctly, not lumped in elsewhere.
    assert_eq!(stats.dropped_no_slot(), 0, "no arena slot was ever refused");
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

// ===========================================================================
// #224 — end of stream across the wire
// ===========================================================================

use parallax::elements::{PublisherLoss, ZenohSrcHandle};
use parallax::pipeline::EndReason;

/// A parallax-plane pair, with one buffer already published so the sink's
/// publisher (and its liveliness token) exist.
async fn parallax_pair(tag: &str) -> (ZenohSrc, ZenohSink, String) {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key(tag);
    let src = ZenohSrc::with_session(sub_session.clone(), &key)
        .await
        .expect("source");
    let sink = ZenohSink::with_session(pub_session.clone(), &key)
        .await
        .expect("sink");
    settle(&pub_session, &sub_session).await;
    (src, sink, key)
}

/// `produce` once, expecting neither a buffer nor an error within `patience`.
async fn expect_no_buffer(src: &mut ZenohSrc, patience: Duration) {
    let mut ctx = ProduceContext::without_buffer();
    match tokio::time::timeout(patience, src.produce(&mut ctx)).await {
        // Parked on the wire — nothing arrived, which is the point.
        Err(_) => {}
        Ok(Ok(ProduceResult::WouldBlock)) => {}
        Ok(other) => panic!("expected nothing, got {other:?}"),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_eos_crosses_the_wire_and_ends_the_receiver() {
    let (mut src, mut sink, _key) = parallax_pair("terminal-eos").await;

    let mut meta = Metadata::new();
    meta.sequence = 1;
    sink.consume(&ConsumeContext::new(&buffer(b"frame", meta)))
        .await
        .expect("publish");
    assert_eq!(next_buffer(&mut src).await.as_bytes(), b"frame");

    sink.finish(EndReason::Eos).await.expect("goodbye");

    let mut ctx = ProduceContext::without_buffer();
    let result = tokio::time::timeout(Duration::from_secs(10), src.produce(&mut ctx))
        .await
        .expect("the terminal must arrive")
        .expect("a clean end is not an error");
    assert!(
        matches!(result, ProduceResult::Eos),
        "got {result:?}, expected Eos"
    );
}

/// #85's rule, extended across the machine boundary: a receiving pipeline must
/// not report a clean end for a stream that failed at the other end.
#[tokio::test(flavor = "multi_thread")]
async fn a_publisher_failure_crosses_the_wire_as_a_failure() {
    let (mut src, mut sink, _key) = parallax_pair("terminal-error").await;

    sink.consume(&ConsumeContext::new(&buffer(b"x", Metadata::new())))
        .await
        .expect("publish");
    let _ = next_buffer(&mut src).await;

    sink.finish(EndReason::Error(parallax::pipeline::StreamError::new(
        "v4l2src",
        "device disappeared",
    )))
    .await
    .expect("goodbye");

    let mut ctx = ProduceContext::without_buffer();
    let err = tokio::time::timeout(Duration::from_secs(10), src.produce(&mut ctx))
        .await
        .expect("the terminal must arrive")
        .expect_err("a failed publisher must not read as a clean EOS");
    let message = err.to_string();
    assert!(
        message.contains("device disappeared") && message.contains("v4l2src"),
        "the reason must survive the crossing, got {message}"
    );
}

/// RFC 07 §1: `@media` carries pixels and nothing else. A terminal sample there
/// would reach a decoder as a zero-length access unit.
#[tokio::test(flavor = "multi_thread")]
async fn the_media_plane_gets_no_terminal_sample() {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");
    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

    let mut meta = Metadata::new();
    meta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&keyframe_au(), meta)))
        .await
        .expect("publish");
    let _ = next_buffer(&mut src).await;

    sink.finish(EndReason::Eos).await.expect("goodbye");

    // Nothing at all — not a frame, and not an EOS either: the default loss
    // policy keeps a media subscription open across a publisher restart.
    expect_no_buffer(&mut src, Duration::from_millis(600)).await;
}

/// The ungraceful case no goodbye covers: the publisher's process is gone.
#[tokio::test(flavor = "multi_thread")]
async fn a_vanished_publisher_ends_the_stream_under_the_eos_policy() {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");
    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source")
        .on_publisher_loss(PublisherLoss::Eos);
    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

    let mut meta = Metadata::new();
    meta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&keyframe_au(), meta)))
        .await
        .expect("publish");
    let _ = next_buffer(&mut src).await;

    // No `finish`, no goodbye — just gone, the way a killed process is.
    drop(sink);

    let mut ctx = ProduceContext::without_buffer();
    let result = tokio::time::timeout(Duration::from_secs(10), src.produce(&mut ctx))
        .await
        .expect("the liveliness delete must arrive")
        .expect("a vanished publisher is not this pipeline's error");
    assert!(
        matches!(result, ProduceResult::Eos),
        "got {result:?}, expected Eos"
    );
    assert_eq!(src.control().publisher_losses(), 1);
}

/// The default, and why: a camera pipeline restarting is normal operation.
#[tokio::test(flavor = "multi_thread")]
async fn a_vanished_publisher_only_warns_by_default() {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");
    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source");
    let handle: ZenohSrcHandle = src.control();
    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

    let mut meta = Metadata::new();
    meta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&keyframe_au(), meta)))
        .await
        .expect("publish");
    let _ = next_buffer(&mut src).await;

    drop(sink);

    expect_no_buffer(&mut src, Duration::from_millis(800)).await;
    assert_eq!(
        handle.publisher_losses(),
        1,
        "the loss is counted even when it is not acted on"
    );
}

/// The lever an application uses to apply a policy parallax has no business
/// knowing — the `@media` `state` catalogue doc, for one.
#[tokio::test(flavor = "multi_thread")]
async fn end_stream_ends_a_source_parked_on_the_wire() {
    let (mut src, _sink, _key) = parallax_pair("end-stream").await;
    let handle = src.control();

    // Ask from another task while `produce` is parked with nothing to read.
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(100)).await;
        handle.end_stream();
    });

    let mut ctx = ProduceContext::without_buffer();
    let result = tokio::time::timeout(Duration::from_secs(5), src.produce(&mut ctx))
        .await
        .expect("end_stream must wake a parked produce")
        .expect("ending on request is not an error");
    assert!(matches!(result, ProduceResult::Eos), "got {result:?}");
}

/// The case neither a terminal nor a liveliness token covers: the publisher is
/// alive, has said nothing, and never will.
#[tokio::test(flavor = "multi_thread")]
async fn a_stream_timeout_ends_a_silent_stream() {
    let (_pub_session, sub_session) = session_pair().await;
    let mut src = ZenohSrc::with_session(sub_session.clone(), unique_key("stream-timeout"))
        .await
        .expect("source")
        .with_stream_timeout(Duration::from_millis(200));

    let mut ctx = ProduceContext::without_buffer();
    let result = tokio::time::timeout(Duration::from_secs(5), src.produce(&mut ctx))
        .await
        .expect("the stream timeout must fire")
        .expect("a timeout is a clean end, not an error");
    assert!(matches!(result, ProduceResult::Eos), "got {result:?}");
}

/// The heartbeat's empty TIMEOUT buffer is a defect on `@media`: the plane's
/// samples are whole access units, and a decoder cannot tell a zero-length
/// "heartbeat" apart from a malformed frame.
#[tokio::test(flavor = "multi_thread")]
async fn an_at_media_heartbeat_never_produces_a_frame() {
    let (_pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");
    let mut src = ZenohSrc::media_with_session(sub_session.clone(), &key)
        .await
        .expect("media source")
        .with_timeout(Duration::from_millis(100));

    // Several expiries, none of which may yield a buffer.
    for _ in 0..3 {
        let mut ctx = ProduceContext::without_buffer();
        let result = tokio::time::timeout(Duration::from_secs(5), src.produce(&mut ctx))
            .await
            .expect("the heartbeat must fire")
            .expect("a heartbeat is not an error");
        assert!(
            matches!(result, ProduceResult::WouldBlock),
            "@media must never emit a zero-length frame, got {result:?}"
        );
    }
}

// ------------------------------------------------------------------- #245

/// Wait for one sample on a raw subscriber, or fail loudly.
async fn next_sample(
    rx: &mut tokio::sync::mpsc::UnboundedReceiver<zenoh::sample::Sample>,
) -> zenoh::sample::Sample {
    tokio::time::timeout(Duration::from_secs(5), rx.recv())
        .await
        .expect("a sample within 5 s")
        .expect("the raw subscriber is still alive")
}

/// The acceptance criterion for #245.
///
/// `FrameMeta` carries no wallclock, so the zenoh HLC stamp on the sample is
/// the only clock a consumer has for frame age. A peer session leaves
/// timestamping off by default, which used to make every `@media` sample
/// parallax published unstamped.
///
/// The subscriber session here has timestamping off (see `session_pair`), so a
/// timestamp on the received sample can only have come from the publisher.
#[tokio::test(flavor = "multi_thread")]
async fn a_media_sample_carries_a_zenoh_timestamp_across_two_sessions() {
    let (pub_session, sub_session) = session_pair().await;
    let key = unique_key("video/h264/high");

    let (raw_tx, mut raw_rx) = tokio::sync::mpsc::unbounded_channel();
    let _raw_sub = sub_session
        .declare_subscriber(&key)
        .callback(move |sample| {
            let _ = raw_tx.send(sample);
        })
        .await
        .expect("raw subscriber");

    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

    let mut meta = Metadata::new();
    meta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&keyframe_au(), meta)))
        .await
        .expect("publish keyframe");

    let sample = next_sample(&mut raw_rx).await;
    assert!(
        sample.timestamp().is_some(),
        "the @media plane's only clock is the HLC stamp on the sample"
    );
}

/// An operator's config wins, and the plane still carries frames without it.
///
/// The sharper half is `timestamp().is_none()`: it proves parallax honoured the
/// file rather than quietly overriding it.
#[tokio::test(flavor = "multi_thread")]
async fn an_operator_config_with_timestamping_off_still_carries_frames() {
    use std::io::Write;

    let mut file = tempfile::Builder::new()
        .suffix(".json5")
        .tempfile()
        .expect("temp config");
    file.write_all(br#"{ timestamping: { enabled: false } }"#)
        .expect("write config");
    file.flush().expect("flush config");

    let publisher_config = zenoh::Config::from_file(file.path()).expect("load operator config");
    let (pub_session, sub_session) =
        session_pair_from(publisher_config, zenoh::Config::default()).await;
    let key = unique_key("video/h264/high");

    let (raw_tx, mut raw_rx) = tokio::sync::mpsc::unbounded_channel();
    let _raw_sub = sub_session
        .declare_subscriber(&key)
        .callback(move |sample| {
            let _ = raw_tx.send(sample);
        })
        .await
        .expect("raw subscriber");

    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    settle(&pub_session, &sub_session).await;

    let au = keyframe_au();
    let mut meta = Metadata::new();
    meta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&au, meta)))
        .await
        .expect("publish keyframe");

    let sample = next_sample(&mut raw_rx).await;
    assert_eq!(
        sample.payload().to_bytes().as_ref(),
        au.as_slice(),
        "the frame still crosses intact"
    );
    let frame = FrameMeta::from_cbor(
        &sample
            .attachment()
            .expect("FrameMeta attachment")
            .to_bytes(),
    )
    .expect("decodable FrameMeta");
    assert!(frame.keyframe, "and its metadata is intact too");
    assert!(
        sample.timestamp().is_none(),
        "the operator said no; parallax must not override them"
    );
}

/// `keyframes_sent` had no writer at all before #246 — the counter was
/// declared, exposed on the handle, and never incremented by anything.
#[tokio::test(flavor = "multi_thread")]
async fn the_handle_counts_keyframes_separately_from_samples() {
    let (pub_session, _sub) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    let stats = sink.control();

    let mut keyframe = Metadata::new();
    keyframe.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&keyframe_au(), keyframe)))
        .await
        .expect("publish keyframe");

    // A delta frame, even one upstream mislabelled as a sync point: on this
    // plane the flag is decided by reading the bytes.
    let mut delta = Metadata::new();
    delta.flags |= BufferFlags::SYNC_POINT;
    sink.consume(&ConsumeContext::new(&buffer(&delta_au(1), delta)))
        .await
        .expect("publish delta");

    assert_eq!(stats.samples_sent(), 2);
    assert_eq!(
        stats.keyframes_sent(),
        1,
        "only the access unit that really is a decoder entry point counts"
    );
}

/// The send rate is the figure a health panel reads; the counter behind it is
/// the figure an exporter reads. Both, and they must agree about the stream
/// having stopped.
#[tokio::test(flavor = "multi_thread")]
async fn the_send_rate_rises_with_published_samples_and_decays_when_publishing_stops() {
    let (pub_session, _sub) = session_pair().await;
    let key = unique_key("video/h264/high");

    let mut sink = ZenohSink::media_with_session(pub_session.clone(), &key)
        .await
        .expect("media sink");
    let stats = sink.control();

    assert_eq!(stats.samples_per_second(), 0.0, "nothing published yet");

    // ~30 samples over ~1 s.
    for i in 0..30u8 {
        let mut meta = Metadata::new();
        meta.flags |= BufferFlags::SYNC_POINT;
        sink.consume(&ConsumeContext::new(&buffer(&delta_au(i), meta)))
            .await
            .expect("publish");
        tokio::time::sleep(Duration::from_millis(33)).await;
    }

    let publishing = stats.samples_per_second();
    let bitrate = stats.bits_per_second();
    assert!(
        publishing > 5.0,
        "a stream publishing at ~30/s must not read as idle, got {publishing:.1}"
    );
    assert!(bitrate > 0.0, "and its bitrate must be positive");

    tokio::time::sleep(Duration::from_secs(3)).await;
    let stopped = stats.samples_per_second();
    assert!(
        stopped < publishing / 10.0,
        "a stalled publisher must read as stalled: {publishing:.1} -> {stopped:.3} /s"
    );
    assert_eq!(
        stats.samples_sent(),
        30,
        "the counter does not decay — that is the point of keeping both"
    );
}
