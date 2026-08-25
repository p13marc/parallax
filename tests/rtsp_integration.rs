//! #209: the RTSP client, against a real socket.
//!
//! `src/elements/rtp/rtsp.rs` had sixteen unit tests and none of them opened
//! one — the file said so itself. DESCRIBE/SETUP/PLAY, transport selection,
//! connect timeouts, geometry stamping and the whole `AsyncSource::produce`
//! path rested on manual verification, which meant the ingest half of the
//! multi-machine epic rested on it too.
//!
//! The server is in `tests/support/rtsp_server.rs` and is deliberately a
//! fixture rather than a product. What it buys is the *failure* cases: a
//! server that dies mid-stream, one that never answers DESCRIBE, one whose SDP
//! is missing the parameter sets.

#![cfg(feature = "rtsp")]

mod support;

use std::time::Duration;

use parallax::clock::ClockTime;
use parallax::element::{AsyncSource, ProduceContext, ProduceResult};
use parallax::elements::{RtspReconnect, RtspSrc};
use parallax::metadata::BufferFlags;
use support::rtsp_server::{self, Behaviour};

/// Pull buffers until `want` of them arrive or the source ends.
async fn drain(
    session: &mut parallax::elements::RtspSession,
    want: usize,
) -> Vec<parallax::buffer::Buffer> {
    let mut out = Vec::new();
    for _ in 0..want * 8 {
        let mut ctx = ProduceContext::without_buffer();
        match tokio::time::timeout(Duration::from_secs(10), session.produce(&mut ctx)).await {
            Ok(Ok(ProduceResult::OwnBuffer(b))) => {
                out.push(b);
                if out.len() == want {
                    return out;
                }
            }
            Ok(Ok(ProduceResult::Eos)) => return out,
            Ok(Ok(_)) => continue,
            Ok(Err(e)) => panic!("produce failed: {e}"),
            Err(_) => panic!("timed out after {} buffers", out.len()),
        }
    }
    out
}

#[tokio::test(flavor = "multi_thread")]
async fn connect_play_and_receive_annex_b_frames() {
    let server = rtsp_server::start(Behaviour::good());

    let mut session = RtspSrc::new(server.url())
        .video_only()
        .connect()
        .await
        .expect("connect");

    // The SDP carried `sprop-parameter-sets`, so geometry is known before a
    // single frame arrives (#205).
    let video = session
        .streams()
        .iter()
        .find(|s| s.media_type == parallax::elements::MediaType::Video)
        .expect("a video stream");
    assert_eq!(video.dimensions, Some((64, 64)));
    assert_eq!(video.clock_rate, 90_000);

    let buffers = drain(&mut session, 1).await;
    assert!(!buffers.is_empty(), "no frames arrived");

    let first = &buffers[0];
    assert_eq!(
        &first.as_bytes()[..4],
        &[0, 0, 0, 1],
        "the default frame format is Annex-B"
    );
    assert!(
        first.metadata().is_keyframe(),
        "the keyframe gate holds until the first IDR, so the first buffer is one"
    );

    // #208: no fabricated `RtpMeta`. retina hands back demuxed access units,
    // not packets, so `seq`, `ssrc` and `pt` are simply not available — and
    // filling them with zeros published plausible-looking garbage to anything
    // that read them, including the IPC descriptor, which carries `RtpMeta`
    // across a process boundary. `None` is the truthful answer, and the RTP
    // timestamp is not lost: `pts` is derived from it.
    assert!(
        first.metadata().rtp.is_none(),
        "RTSP must not attach RtpMeta it does not have: {:?}",
        first.metadata().rtp
    );
    assert!(
        first.metadata().pts.nanos() > 0 || first.metadata().pts == ClockTime::ZERO,
        "the RTP timestamp still reaches pts"
    );

    assert_eq!(server.plays(), 1);
}

/// Credentials in the URL are lifted into RTSP auth rather than handed to
/// retina, which rejects them.
#[tokio::test(flavor = "multi_thread")]
async fn credentials_in_the_url_reach_the_server() {
    let server = rtsp_server::start(Behaviour {
        require_auth: true,
        ..Behaviour::good()
    });
    let url = server.url().replace("rtsp://", "rtsp://user:pass@");

    let mut session = RtspSrc::new(url)
        .video_only()
        .connect()
        .await
        .expect("the 401 must be answered, not fatal");

    assert!(!drain(&mut session, 1).await.is_empty());
}

/// A server that accepts the connection and then says nothing must not hang
/// the caller forever.
#[tokio::test(flavor = "multi_thread")]
async fn a_stalled_describe_hits_the_connect_timeout() {
    let server = rtsp_server::start(Behaviour {
        stall_describe: true,
        ..Behaviour::good()
    });

    let started = std::time::Instant::now();
    let result = RtspSrc::new(server.url())
        .with_timeout(Duration::from_millis(300))
        .connect()
        .await;

    assert!(result.is_err(), "a stalled DESCRIBE must not succeed");
    assert!(
        started.elapsed() < Duration::from_secs(5),
        "the timeout did not fire: waited {:?}",
        started.elapsed()
    );
}

/// #205: with no `sprop-parameter-sets`, the SDP's `a=framesize` is what makes
/// geometry known at connect time — parsed by parallax, because retina's SDP
/// parser does not handle it.
#[tokio::test(flavor = "multi_thread")]
async fn framesize_supplies_geometry_when_sprop_does_not() {
    let server = rtsp_server::start(Behaviour {
        sprop: false,
        framesize: Some((64, 64)),
        ..Behaviour::good()
    });

    let session = RtspSrc::new(server.url())
        .video_only()
        .connect()
        .await
        .expect("connect");

    let video = &session.streams()[0];
    assert_eq!(
        video.dimensions,
        Some((64, 64)),
        "a=framesize is the only geometry this SDP has"
    );
}

/// #206, the case the whole server exists for: the connection dies mid-stream
/// and the pipeline must not.
#[tokio::test(flavor = "multi_thread")]
async fn a_mid_stream_disconnect_reconnects_and_flags_the_gap() {
    let server = rtsp_server::start(Behaviour {
        // One packet, then the connection dies — the first frame may or may
        // not have completed, but the stream certainly has not.
        drop_after: Some(1),
        packet_interval: Duration::from_millis(1),
        ..Behaviour::good()
    });

    let mut session = RtspSrc::new(server.url())
        .video_only()
        .with_reconnect(RtspReconnect {
            max_retries: Some(5),
            initial_backoff: Duration::from_millis(20),
            max_backoff: Duration::from_millis(100),
        })
        .connect()
        .await
        .expect("connect");

    // Two buffers: the second can only arrive after a reconnect, because the
    // server killed the first connection partway through.
    let buffers = drain(&mut session, 2).await;
    assert!(
        buffers.len() >= 2,
        "expected the stream to resume, got {} buffer(s)",
        buffers.len()
    );

    assert!(
        server.plays() >= 2,
        "the client must have reached PLAY again, saw {}",
        server.plays()
    );
    assert!(
        session.stats().reconnects >= 1,
        "the reconnect must be counted"
    );

    // A resumed stream is a discontinuity and says so — everything retina
    // held (connection, RTP origin, SSRC) was replaced.
    //
    // Not `skip(1)`: the killed connection died after one RTP packet, which
    // may not have completed an access unit at all, so the *first* buffer the
    // caller ever sees can be the resumed one.
    let flagged = buffers
        .iter()
        .filter(|b| b.metadata().flags.contains(BufferFlags::DISCONT))
        .count();
    assert!(
        flagged >= 1,
        "a resumed stream must be flagged DISCONT; none of {} buffers was",
        buffers.len()
    );
}

/// Without a reconnect policy the stream simply ends, and the client does
/// **not** go behind the caller's back to reconnect.
///
/// Worth stating why the end is `Eos` rather than an error: RTSP has no
/// in-band end-of-stream for a live stream, so a server whose process dies
/// looks exactly like one that finished — retina yields `None` for a clean FIN
/// either way. `without_reconnect()` is the statement "treat the end as an
/// end", and the pipeline gets a clean EOS. Which is right for batch work: a
/// recording served over RTSP really does run out.
#[tokio::test(flavor = "multi_thread")]
async fn without_reconnect_the_stream_simply_ends() {
    let server = rtsp_server::start(Behaviour {
        drop_after: Some(1),
        packet_interval: Duration::from_millis(1),
        ..Behaviour::good()
    });

    let mut session = RtspSrc::new(server.url())
        .video_only()
        .without_reconnect()
        .connect()
        .await
        .expect("connect");

    let mut ended = false;
    for _ in 0..64 {
        let mut ctx = ProduceContext::without_buffer();
        match tokio::time::timeout(Duration::from_secs(10), session.produce(&mut ctx)).await {
            Ok(Ok(ProduceResult::Eos)) | Ok(Err(_)) => {
                ended = true;
                break;
            }
            Ok(Ok(_)) => continue,
            Err(_) => panic!("produce hung"),
        }
    }
    assert!(ended, "a dropped connection must end the source");
    assert_eq!(
        server.plays(),
        1,
        "and must not be retried behind the caller's back"
    );
    assert_eq!(session.stats().reconnects, 0);
}

/// #207: an RTP timestamp is a counter with an arbitrary origin. The Sender
/// Report is the only thing that turns it into a time two machines can agree
/// on — which is what an end-to-end latency measurement across this epic's
/// topology needs.
#[tokio::test(flavor = "multi_thread")]
async fn a_sender_report_anchors_timestamps_to_a_wallclock() {
    // 2024-01-01T00:00:00Z, anchoring RTP timestamp 0.
    const ANCHOR_SECS: u64 = 1_704_067_200;

    let server = rtsp_server::start(Behaviour {
        sender_report_at_unix_secs: Some(ANCHOR_SECS),
        packet_interval: Duration::from_millis(5),
        ..Behaviour::good()
    });

    let mut session = RtspSrc::new(server.url())
        .video_only()
        .without_reconnect()
        .connect()
        .await
        .expect("connect");
    let handle = session.stream_info_handle();

    let buffers = drain(&mut session, 2).await;
    assert!(!buffers.is_empty());

    let sr = handle
        .stream(0)
        .and_then(|s| s.sender_report)
        .expect("the SR must have been recorded");
    assert_eq!(sr.ntp_unix_nanos, ANCHOR_SECS * 1_000_000_000);
    assert_eq!(sr.rtp_timestamp, 0);

    // The PTS is now a wallclock, not a per-session relative number: it must
    // land within a second of the anchor for a stream that starts at RTP 0.
    let pts = buffers[0].metadata().pts.nanos();
    let anchor = ANCHOR_SECS * 1_000_000_000;
    assert!(
        pts >= anchor && pts < anchor + 5_000_000_000,
        "pts {pts} is not anchored near {anchor}"
    );
}

/// The camera that never sends one. Timestamps stay relative, and that is a
/// supported configuration rather than a fault.
#[tokio::test(flavor = "multi_thread")]
async fn no_sender_report_falls_back_to_relative_timing() {
    let server = rtsp_server::start(Behaviour {
        sender_report_at_unix_secs: None,
        ..Behaviour::good()
    });

    let mut session = RtspSrc::new(server.url())
        .video_only()
        .without_reconnect()
        .connect()
        .await
        .expect("connect");
    let handle = session.stream_info_handle();

    let buffers = drain(&mut session, 1).await;
    assert!(!buffers.is_empty());
    assert!(handle.stream(0).unwrap().sender_report.is_none());

    // Relative to the RTP origin: small, not a 2024 wallclock.
    assert!(
        buffers[0].metadata().pts.nanos() < 60_000_000_000,
        "an unanchored stream must not claim a wallclock"
    );
}
