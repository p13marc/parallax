//! Two processes over zenoh, on parallax's private plane (#228).
//!
//! One binary, two processes: run with no arguments and it plays the
//! *publisher* — a pipeline `source → ZenohSink` — and spawns itself again
//! with `--subscriber <key>` as the *subscriber*, `ZenohSrc → printing sink`
//! in a genuinely separate process. Compare `60_ipc.rs`, which is the same
//! shape over shared memory: that one is the same-machine answer, this one
//! crosses machines.
//!
//! What crosses per buffer is the payload bytes plus an rkyv `WireMetadata`
//! attachment — pts/dts/duration, sequence, flags, media format, coded
//! geometry, and whitelisted custom entries such as KLV. The payload is
//! published **straight out of the arena** (#226): a cloned `Buffer` pins the
//! slot for as long as zenoh holds the bytes, so nothing is copied on the way
//! out.
//!
//! The two processes find each other over loopback with scouting off and an
//! explicit endpoint, so this demo cannot wander onto a LAN. On two real
//! machines you would drop the endpoint config and let zenoh scout, or point
//! `connect/endpoints` at a router.
//!
//! Run: `cargo run --example 62_zenoh --features zenoh`

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{AsyncSink, AsyncSource, ConsumeContext, ProduceContext, ProduceResult};
use parallax::elements::{ZenohSink, ZenohSrc};
use parallax::error::Result;
use parallax::format::{MediaFormat, VideoCodec};
use parallax::memory::SharedArena;
use parallax::metadata::{BufferFlags, Metadata};
use parallax::pipeline::{EndReason, Executor, Pipeline};
use std::sync::Arc;

const FRAMES: u64 = 20;

/// Stand-in for an encoder: one "access unit" per frame, every fifth a
/// keyframe, with KLV riding along so the subscriber can show that the
/// attachment carried more than pixels.
///
/// An `AsyncSource` rather than a `Source` so it can *pace* itself at 25 fps.
/// That is not decoration: pushing twenty frames as fast as the loop runs
/// overruns the subscriber's receive queue, which is bounded and drop-oldest
/// on purpose (#225), and the demo would print a stream with a hole in it.
/// A real capture source is paced by its device.
struct StampedSource {
    arena: Arc<SharedArena>,
    produced: u64,
}

impl AsyncSource for StampedSource {
    async fn produce(&mut self, _ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        if self.produced >= FRAMES {
            return Ok(ProduceResult::Eos);
        }
        if self.produced > 0 {
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
        }
        self.arena.reclaim();
        let Some(mut slot) = self.arena.acquire() else {
            return Ok(ProduceResult::WouldBlock);
        };
        let n = self.produced;
        self.produced += 1;

        slot.data_mut()[..8].copy_from_slice(&n.to_ne_bytes());

        let mut meta = Metadata::from_sequence(n);
        meta.pts = ClockTime::from_millis(n * 40); // 25 fps
        meta.set_coded_video_dims(VideoCodec::H264, 1280, 720);
        if n.is_multiple_of(5) {
            meta.flags |= BufferFlags::SYNC_POINT;
            meta.set_klv(vec![0x4B, 0x4C, 0x56, n as u8]);
        }

        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            meta,
        )))
    }
}

/// Prints what arrived, proving payload and metadata crossed intact.
struct PrintingSink;

impl AsyncSink for PrintingSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let buffer = ctx.buffer();
        let meta = buffer.metadata();
        let payload = u64::from_ne_bytes(buffer.as_bytes()[..8].try_into().unwrap());
        let klv = meta
            .get_bytes("stanag/klv")
            .map(|b| format!(" klv={b:02x?}"))
            .unwrap_or_default();
        let geometry = match meta.coded_video_dims() {
            Some((w, h)) => format!("{w}x{h}"),
            None => "?".to_string(),
        };
        println!(
            "  [subscriber pid {}] frame {payload:2} pts={:?} {}{geometry}{klv}{}",
            std::process::id(),
            meta.pts,
            match &meta.format {
                Some(MediaFormat::Video(codec)) => format!("{codec:?} "),
                _ => String::new(),
            },
            if meta.flags.is_keyframe() {
                " [keyframe]"
            } else {
                ""
            }
        );
        Ok(())
    }
}

/// Scouting off, one explicit endpoint: the two processes find each other and
/// nothing else. A real deployment leaves scouting on, or points
/// `connect/endpoints` at a router.
fn config(endpoint: &str, listen: bool) -> zenoh::Config {
    let mut config = zenoh::Config::default();
    config
        .insert_json5("scouting/multicast/enabled", "false")
        .unwrap();
    config
        .insert_json5("scouting/gossip/enabled", "false")
        .unwrap();
    let field = if listen {
        "listen/endpoints"
    } else {
        "connect/endpoints"
    };
    config
        .insert_json5(field, &format!("[\"{endpoint}\"]"))
        .unwrap();
    config
}

async fn run_publisher(key: &str, session: zenoh::Session) -> Result<()> {
    println!(
        "[publisher pid {}] publishing {FRAMES} frames on {key}",
        std::process::id()
    );

    // The payload arena. `ZenohSink` publishes out of it directly, so its
    // slots stay pinned until the transport is done with them — which is why
    // the sink declares `retained_buffers()`.
    let arena = Arc::new(SharedArena::new(64, 64)?);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_async_source("frames", StampedSource { arena, produced: 0 });
    let sink = pipeline.add_async_sink("zenohsink", ZenohSink::with_session(session, key).await?);
    pipeline.link(src, sink)?;

    let handle = Executor::new().start(&mut pipeline)?;
    handle.wait().await?;
    // The terminal sample goes out from the sink's `finish()` hook, after the
    // last frame and on the same key — which is what makes it ordered.
    println!("[publisher] done — EOS published on the media key");
    Ok(())
}

async fn run_subscriber(key: &str, endpoint: &str) -> Result<()> {
    let session = zenoh::open(config(endpoint, false))
        .await
        .map_err(|e| parallax::error::Error::Element(e.to_string()))?;

    // A safety net, and an honest use of the knob (#224): `with_stream_timeout`
    // ends the stream after a silence, for the case neither a terminal sample
    // nor a liveliness token catches — here, the two processes never meeting.
    let src = ZenohSrc::with_session(session, key)
        .await?
        .with_stream_timeout(std::time::Duration::from_secs(15));

    // Cloned *before* `start()` — the element is moved into its task there.
    let counters = parallax::control::Controllable::control(&src);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_async_source("zenohsrc", src);
    let sink = pipeline.add_async_sink("printer", PrintingSink);
    pipeline.link(src, sink)?;

    let handle = Executor::new().start(&mut pipeline)?;
    let ended = handle.ended();
    handle.wait().await?;

    // #224: the publisher's goodbye crossed the wire, so this is a real EOS
    // and not a subscription that merely went quiet.
    println!(
        "  [subscriber] ended: {:?} — {} samples, {} bytes, dropped {}/{}/{} \
         (queue/no-slot/wire)",
        ended.await,
        counters.samples_received(),
        counters.bytes_received(),
        counters.dropped_queue(),
        counters.dropped_no_slot(),
        counters.lost_wire(),
    );
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let endpoint = format!("tcp/127.0.0.1:{}", 7447 + (std::process::id() % 500) as u16);
    let key = "parallax/examples/62/video";

    if let Some(pos) = args.iter().position(|a| a == "--subscriber") {
        return run_subscriber(key, &args[pos + 1]).await;
    }

    // Open the listening session *before* spawning the subscriber: it is the
    // one that connects, and it cannot connect to a port nobody is listening
    // on yet.
    let session = zenoh::open(config(&endpoint, true))
        .await
        .map_err(|e| parallax::error::Error::Element(e.to_string()))?;

    let mut child = std::process::Command::new(std::env::current_exe()?)
        .arg("--subscriber")
        .arg(&endpoint)
        .spawn()?;

    // Give the subscriber time to connect and declare. Over a real transport
    // a declaration has to propagate before the publisher will match it, and
    // a sample sent into that window is simply gone — which is why every
    // two-session test in this repo has a settle step too.
    tokio::time::sleep(std::time::Duration::from_millis(1500)).await;

    let published = run_publisher(key, session).await;

    let status = child.wait()?;
    published?;
    if !status.success() {
        return Err(parallax::error::Error::Pipeline(format!(
            "subscriber process failed: {status}"
        )));
    }
    println!("[publisher] subscriber exited cleanly — two-process zenoh round trip complete");
    let _ = EndReason::Eos;
    Ok(())
}
