//! A minimal RTSP server, for testing parallax's client. Nothing more.
//!
//! # Why this exists
//!
//! `src/elements/rtp/rtsp.rs` had sixteen unit tests and not one of them
//! opened a socket, so DESCRIBE/SETUP/PLAY, transport selection, connect
//! timeouts, frame conversion and the whole `AsyncSource::produce` path rested
//! on manual verification against `just rtsp-server` — which needs PyGObject,
//! `gstreamer1-rtsp-server` and `gstreamer1-plugins-bad-free`. Putting a
//! GStreamer stack on a CI runner to test a *Rust* RTSP client is the wrong
//! trade, especially on a self-hosted instance that has already been tuned for
//! OOM.
//!
//! # What it is, and is not
//!
//! It speaks exactly enough RFC 7826 for retina to reach PLAY over
//! TCP-interleaved transport and then receive H.264 in RTP. It is a **test
//! fixture, not an RTSP server**: no UDP, no RTCP, no digest auth beyond a
//! canned challenge, no session timeouts, no seeking, one media stream. Hand
//! rolled rather than built on `rtsp-types` for the same reason the HTTP test
//! server in `network/http.rs` is hand rolled: the protocol surface needed is
//! small, and the hard part — producing RTP a real depacketizer accepts — is
//! not what a message parser helps with.
//!
//! The RTP packetization comes from parallax's own `RtpH264Pay`, which means
//! the fixture stream is as valid as the payloader is. That is a feature: a
//! bug in the payloader shows up here as a client that cannot decode.
//!
//! # The failure modes are the point
//!
//! Anyone can test a happy path. What is worth a server is the rest:
//! [`Behaviour::drop_after`] kills the connection mid-stream so the reconnect
//! path (#206) is exercised for real, [`Behaviour::stall_describe`] never
//! answers so the connect timeout fires, and [`Behaviour::require_auth`]
//! challenges once so credential lifting is exercised.

#![allow(dead_code)] // Each test uses a subset.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// How the server should misbehave, if at all.
#[derive(Debug, Clone, Default)]
pub struct Behaviour {
    /// Kill the **first** connection after this many RTP packets. Later
    /// connections stream the fixture in full.
    ///
    /// First-connection-only on purpose: it is the "a switch rebooted once"
    /// scenario, and it terminates. A server that dies on every connection
    /// would have the client reconnecting forever, which tests nothing except
    /// the test's own timeout.
    pub drop_after: Option<usize>,
    /// Never answer DESCRIBE. The client's connect timeout must fire.
    pub stall_describe: bool,
    /// Answer the first DESCRIBE with `401` + a Basic challenge.
    pub require_auth: bool,
    /// Include `a=framesize:96 <w>-<h>` in the SDP.
    pub framesize: Option<(u32, u32)>,
    /// Include `sprop-parameter-sets` in the SDP's fmtp line.
    pub sprop: bool,
    /// Pause this long between RTP packets, so a test can observe the stream
    /// rather than race it to EOF.
    pub packet_interval: Duration,
}

impl Behaviour {
    /// A well-behaved server: parameter sets and a framesize in the SDP.
    pub fn good() -> Self {
        Self {
            sprop: true,
            framesize: Some((64, 64)),
            packet_interval: Duration::from_millis(2),
            ..Default::default()
        }
    }
}

/// A running test server.
pub struct TestRtspServer {
    addr: SocketAddr,
    /// How many times a client has reached PLAY. The reconnect proof.
    plays: Arc<AtomicU64>,
    /// How many TCP connections have been accepted.
    connections: Arc<AtomicU64>,
}

impl TestRtspServer {
    /// The `rtsp://` URL a client should connect to.
    pub fn url(&self) -> String {
        format!("rtsp://{}/stream", self.addr)
    }

    /// How many times a client has reached PLAY.
    pub fn plays(&self) -> u64 {
        self.plays.load(Ordering::Relaxed)
    }

    /// How many TCP connections have been accepted.
    pub fn connections(&self) -> u64 {
        self.connections.load(Ordering::Relaxed)
    }
}

/// Start a server on an ephemeral loopback port.
///
/// It serves connections in a loop on a detached thread, so a client that
/// reconnects is served again — which is what makes #206 testable. The thread
/// and the listener live for the rest of the process; a test binary is short.
pub fn start(behaviour: Behaviour) -> TestRtspServer {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let plays = Arc::new(AtomicU64::new(0));
    let connections = Arc::new(AtomicU64::new(0));

    let thread_plays = Arc::clone(&plays);
    let thread_connections = Arc::clone(&connections);
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(stream) = stream else { continue };
            let connection = thread_connections.fetch_add(1, Ordering::Relaxed) + 1;
            let behaviour = behaviour.clone();
            let plays = Arc::clone(&thread_plays);
            // A connection per thread: a client that reconnects while the old
            // connection is still being torn down must not be made to wait.
            std::thread::spawn(move || {
                let _ = serve(stream, behaviour, plays, connection);
            });
        }
    });

    TestRtspServer {
        addr,
        plays,
        connections,
    }
}

// ---------------------------------------------------------------------------
// The conversation
// ---------------------------------------------------------------------------

const SESSION_ID: &str = "parallaxtest";

fn serve(
    mut stream: TcpStream,
    behaviour: Behaviour,
    plays: Arc<AtomicU64>,
    connection: u64,
) -> std::io::Result<()> {
    stream.set_nodelay(true)?;
    let mut pending = Vec::new();
    let mut authed = !behaviour.require_auth;

    loop {
        let Some(request) = read_request(&mut stream, &mut pending)? else {
            return Ok(()); // client hung up
        };
        let method = request.method.as_str();
        let cseq = request.header("cseq").unwrap_or_else(|| "1".into());

        match method {
            "OPTIONS" => {
                respond(
                    &mut stream,
                    200,
                    "OK",
                    &cseq,
                    &[("Public", "OPTIONS, DESCRIBE, SETUP, PLAY, TEARDOWN")],
                    &[],
                )?;
            }
            "DESCRIBE" => {
                if behaviour.stall_describe {
                    // Answer nothing, ever. The client's connect timeout is
                    // the only thing that can end this.
                    std::thread::sleep(Duration::from_secs(3600));
                    return Ok(());
                }
                if !authed {
                    authed = true;
                    respond(
                        &mut stream,
                        401,
                        "Unauthorized",
                        &cseq,
                        &[("WWW-Authenticate", "Basic realm=\"parallax\"")],
                        &[],
                    )?;
                    continue;
                }
                let sdp = sdp(&behaviour);
                respond(
                    &mut stream,
                    200,
                    "OK",
                    &cseq,
                    &[
                        ("Content-Type", "application/sdp"),
                        ("Content-Base", &format!("{}/", request.uri)),
                    ],
                    sdp.as_bytes(),
                )?;
            }
            "SETUP" => {
                respond(
                    &mut stream,
                    200,
                    "OK",
                    &cseq,
                    &[
                        (
                            "Transport",
                            "RTP/AVP/TCP;unicast;interleaved=0-1;ssrc=DEADBEEF",
                        ),
                        ("Session", &format!("{SESSION_ID};timeout=60")),
                    ],
                    &[],
                )?;
            }
            "PLAY" => {
                respond(
                    &mut stream,
                    200,
                    "OK",
                    &cseq,
                    &[
                        ("Session", SESSION_ID),
                        ("Range", "npt=0.000-"),
                        (
                            "RTP-Info",
                            &format!("url={}/streamid=0;seq=0;rtptime=0", request.uri),
                        ),
                    ],
                    &[],
                )?;
                plays.fetch_add(1, Ordering::Relaxed);
                return pump(&mut stream, &behaviour, connection);
            }
            "TEARDOWN" => {
                respond(
                    &mut stream,
                    200,
                    "OK",
                    &cseq,
                    &[("Session", SESSION_ID)],
                    &[],
                )?;
                return Ok(());
            }
            _ => {
                respond(&mut stream, 501, "Not Implemented", &cseq, &[], &[])?;
            }
        }
    }
}

/// Write the fixture as interleaved RTP until it runs out — or until
/// `drop_after` packets, whichever comes first.
fn pump(stream: &mut TcpStream, behaviour: &Behaviour, connection: u64) -> std::io::Result<()> {
    let packets = rtp_packets();
    let die_at = if connection == 1 {
        behaviour.drop_after
    } else {
        None
    };
    for (i, packet) in packets.iter().enumerate() {
        if die_at == Some(i) {
            // Slam the connection shut mid-stream: no TEARDOWN, no EOS, the
            // way a switch reboot looks from the client's side.
            let _ = stream.shutdown(std::net::Shutdown::Both);
            return Ok(());
        }
        // `$` channel length(u16be) payload — RFC 7826 §14.
        let mut framed = Vec::with_capacity(4 + packet.len());
        framed.push(b'$');
        framed.push(0); // interleaved channel 0 = RTP
        framed.extend_from_slice(&(packet.len() as u16).to_be_bytes());
        framed.extend_from_slice(packet);
        if stream.write_all(&framed).is_err() {
            return Ok(()); // client went away; that is its business
        }
        if !behaviour.packet_interval.is_zero() {
            std::thread::sleep(behaviour.packet_interval);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Message plumbing
// ---------------------------------------------------------------------------

struct Request {
    method: String,
    uri: String,
    headers: Vec<(String, String)>,
}

impl Request {
    fn header(&self, name: &str) -> Option<String> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.clone())
    }
}

/// Read one request, buffering across reads. `None` when the peer closed.
fn read_request(stream: &mut TcpStream, pending: &mut Vec<u8>) -> std::io::Result<Option<Request>> {
    loop {
        if let Some(end) = find_double_crlf(pending) {
            let head = pending.drain(..end).collect::<Vec<u8>>();
            let text = String::from_utf8_lossy(&head).into_owned();
            let mut lines = text.lines();
            let Some(request_line) = lines.next() else {
                return Ok(None);
            };
            let mut parts = request_line.split_whitespace();
            let method = parts.next().unwrap_or_default().to_string();
            let uri = parts.next().unwrap_or_default().to_string();
            let headers = lines
                .filter_map(|line| line.split_once(':'))
                .map(|(k, v)| (k.trim().to_string(), v.trim().to_string()))
                .collect();
            return Ok(Some(Request {
                method,
                uri,
                headers,
            }));
        }

        let mut chunk = [0u8; 2048];
        match stream.read(&mut chunk) {
            Ok(0) => return Ok(None),
            Ok(n) => pending.extend_from_slice(&chunk[..n]),
            Err(e) => return Err(e),
        }
    }
}

fn find_double_crlf(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
}

fn respond(
    stream: &mut TcpStream,
    status: u16,
    reason: &str,
    cseq: &str,
    headers: &[(&str, &str)],
    body: &[u8],
) -> std::io::Result<()> {
    let mut out = format!("RTSP/1.0 {status} {reason}\r\nCSeq: {cseq}\r\n");
    for (name, value) in headers {
        out.push_str(&format!("{name}: {value}\r\n"));
    }
    if !body.is_empty() {
        out.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    out.push_str("\r\n");
    stream.write_all(out.as_bytes())?;
    stream.write_all(body)?;
    stream.flush()
}

// ---------------------------------------------------------------------------
// The stream itself
// ---------------------------------------------------------------------------

const FIXTURE: &[u8] = include_bytes!("../fixtures/tiny_64x64.h264");

fn sdp(behaviour: &Behaviour) -> String {
    let mut fmtp = String::from("packetization-mode=1;profile-level-id=640C14");
    if behaviour.sprop {
        let (sps, pps) = parameter_sets();
        fmtp.push_str(&format!(
            ";sprop-parameter-sets={},{}",
            base64(&sps),
            base64(&pps)
        ));
    }
    let mut sdp = String::from(
        "v=0\r\n\
         o=- 0 0 IN IP4 127.0.0.1\r\n\
         s=parallax test fixture\r\n\
         t=0 0\r\n\
         m=video 0 RTP/AVP 96\r\n\
         c=IN IP4 0.0.0.0\r\n\
         a=rtpmap:96 H264/90000\r\n",
    );
    sdp.push_str(&format!("a=fmtp:96 {fmtp}\r\n"));
    if let Some((w, h)) = behaviour.framesize {
        sdp.push_str(&format!("a=framesize:96 {w}-{h}\r\n"));
    }
    sdp.push_str("a=control:streamid=0\r\n");
    sdp
}

/// The fixture's SPS and PPS, without start codes.
fn parameter_sets() -> (Vec<u8>, Vec<u8>) {
    let mut sps = Vec::new();
    let mut pps = Vec::new();
    for nal in parallax::codec::annexb::nal_units(FIXTURE) {
        match nal.data[0] & 0x1F {
            7 if sps.is_empty() => sps = nal.data.to_vec(),
            8 if pps.is_empty() => pps = nal.data.to_vec(),
            _ => {}
        }
    }
    (sps, pps)
}

/// Packetize the fixture into RTP packets, one access unit per timestamp.
///
/// Uses parallax's own `RtpH264Pay` for the payloads and the `rtp` crate for
/// the headers — the same code the library ships, so a client that cannot
/// decode this is telling us something true.
fn rtp_packets() -> Vec<Vec<u8>> {
    use parallax::buffer::{Buffer, MemoryHandle};
    use parallax::clock::ClockTime;
    use parallax::element::{Output, Transform};
    use parallax::elements::RtpH264Pay;
    use parallax::memory::SharedArena;
    use parallax::metadata::Metadata;

    // Split the fixture into access units: a new one starts at each VCL NAL
    // (types 1 and 5) that follows a non-VCL one.
    let mut access_units: Vec<Vec<u8>> = Vec::new();
    let mut current: Vec<u8> = Vec::new();
    let mut seen_vcl = false;
    for nal in parallax::codec::annexb::nal_units(FIXTURE) {
        let nal_type = nal.data[0] & 0x1F;
        let is_vcl = matches!(nal_type, 1 | 5);
        if is_vcl && seen_vcl {
            access_units.push(std::mem::take(&mut current));
            seen_vcl = false;
        }
        current.extend_from_slice(&[0, 0, 0, 1]);
        current.extend_from_slice(nal.data);
        seen_vcl |= is_vcl;
    }
    if !current.is_empty() {
        access_units.push(current);
    }

    let arena = SharedArena::new(64 * 1024, 8).expect("arena");
    let mut pay = RtpH264Pay::new().with_mtu(1200);
    let mut out = Vec::new();
    let mut seq: u16 = 0;

    for (i, au) in access_units.iter().enumerate() {
        arena.reclaim();
        let mut slot = arena.acquire().expect("slot");
        slot.data_mut()[..au.len()].copy_from_slice(au);
        let mut meta = Metadata::new();
        // 90 kHz, 10 fps.
        meta.pts = ClockTime::from_nanos(i as u64 * 100_000_000);
        let buffer = Buffer::new(MemoryHandle::with_len(slot, au.len()), meta);

        let packets = match pay.transform(buffer).expect("packetize") {
            Output::Multiple(v) => v,
            Output::Single(b) => vec![b],
            Output::None => Vec::new(),
        };
        let last = packets.len().saturating_sub(1);
        for (j, packet) in packets.iter().enumerate() {
            let rtp = packet.metadata().rtp.expect("per-packet RtpMeta");
            out.push(rtp_header(seq, rtp.ts, j == last, packet.as_bytes()));
            seq = seq.wrapping_add(1);
        }
    }
    out
}

/// A 12-byte RTP header plus payload. Payload type 96, SSRC 0xDEADBEEF.
fn rtp_header(seq: u16, timestamp: u32, marker: bool, payload: &[u8]) -> Vec<u8> {
    let mut packet = Vec::with_capacity(12 + payload.len());
    packet.push(0x80); // V=2, no padding, no extension, CC=0
    packet.push(if marker { 0x80 | 96 } else { 96 });
    packet.extend_from_slice(&seq.to_be_bytes());
    packet.extend_from_slice(&timestamp.to_be_bytes());
    packet.extend_from_slice(&0xDEAD_BEEFu32.to_be_bytes());
    packet.extend_from_slice(payload);
    packet
}

/// Standard base64, no line breaks. Fifteen lines beats a dependency for a
/// test fixture that encodes two short byte strings.
fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
        out.push(ALPHABET[(n >> 18) as usize & 0x3F] as char);
        out.push(ALPHABET[(n >> 12) as usize & 0x3F] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 0x3F] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 0x3F] as char
        } else {
            '='
        });
    }
    out
}
