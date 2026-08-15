//! Two-process zero-copy IPC over the shared-memory descriptor ring (#179).
//!
//! One binary, two processes: run with no arguments and it plays the
//! *sender* — a pipeline `source → IpcSink` — and spawns itself again with
//! `--receiver <socket>` as the *receiver*, a pipeline `IpcSrc → printing
//! sink` in a genuinely separate process.
//!
//! What actually crosses the process boundary per buffer is a 128-byte
//! descriptor in a shared-memory SPSC ring (slot reference + pts/dts/
//! duration/sequence/flags/format/RTP metadata) and a `u64` ack coming
//! back — the payload bytes never move, they live in the source's
//! memfd-backed arena, mapped by both processes. Eventfd doorbells provide
//! the wakeups; the Unix socket carries only registration (with fds via
//! SCM_RIGHTS), rare custom-metadata overflow (the KLV entries below), and
//! teardown. That is the data-plane/signaling-plane split of design.md
//! principle 8, applied across processes.
//!
//! Run: `cargo run --example 60_ipc`

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::{AsyncSink, ConsumeContext, ProduceContext, ProduceResult, Source};
use parallax::elements::{IpcSink, IpcSrc};
use parallax::error::Result;
use parallax::format::{Framerate, MediaFormat, PixelFormat, VideoFormat};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{Executor, Pipeline};
use std::sync::Arc;

const FRAMES: u64 = 25;

/// Sender-side source: stamps each buffer with a payload, timestamps, a
/// video format, and (every fifth frame) a KLV metadata entry — so the
/// receiver can show that all of it crossed the boundary.
struct StampedSource {
    arena: Arc<SharedArena>,
    produced: u64,
}

impl Source for StampedSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.produced >= FRAMES {
            return Ok(ProduceResult::Eos);
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
        meta.format = Some(MediaFormat::VideoRaw(VideoFormat {
            width: 320,
            height: 240,
            pixel_format: PixelFormat::I420,
            framerate: Framerate::new(25, 1),
        }));
        if n % 5 == 0 {
            // Rides the control socket as MetaOverflow, re-attached by the
            // receiver — the one part of Metadata too dynamic for the ring.
            meta.set_klv(vec![0x4B, 0x4C, 0x56, n as u8]);
        }

        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, 8),
            meta,
        )))
    }
}

/// Receiver-side sink: prints what arrived, proving payload + metadata
/// crossed the process boundary intact.
struct PrintingSink {
    received: u64,
}

impl AsyncSink for PrintingSink {
    async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
        let buffer = ctx.buffer();
        let meta = buffer.metadata();
        let payload = u64::from_ne_bytes(buffer.as_bytes()[..8].try_into().unwrap());
        let klv = meta
            .get_bytes("stanag/klv")
            .map(|b| format!(" klv={b:02x?}"))
            .unwrap_or_default();
        let format = match &meta.format {
            Some(MediaFormat::VideoRaw(v)) => format!("{}x{}", v.width, v.height),
            other => format!("{other:?}"),
        };
        println!(
            "  [receiver pid {}] frame {payload:2} pts={:?} format={format}{klv}",
            std::process::id(),
            meta.pts
        );
        self.received += 1;
        Ok(())
    }
}

async fn run_sender(socket: &std::path::Path) -> Result<()> {
    println!(
        "[sender pid {}] producing {FRAMES} frames into {}",
        std::process::id(),
        socket.display()
    );

    // The payload arena: 25 tiny frames' worth of slots. The receiver maps
    // this same memfd; IpcSink registers it on first sight.
    let arena = Arc::new(SharedArena::new(64, 64)?);

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("frames", StampedSource { arena, produced: 0 });
    let sink = pipeline.add_async_sink("ipcsink", IpcSink::new(socket));
    pipeline.link(src, sink)?;

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await?;
    println!("[sender] done — all frames pushed and acked");
    Ok(())
}

async fn run_receiver(socket: &std::path::Path) -> Result<()> {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_async_source("ipcsrc", IpcSrc::new(socket));
    let sink = pipeline.add_async_sink("printer", PrintingSink { received: 0 });
    pipeline.link(src, sink)?;

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();
    handle.wait().await?;
    println!("  [receiver] EOS — stream ended cleanly via the ring's state word");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    if let Some(pos) = args.iter().position(|a| a == "--receiver") {
        let socket = std::path::PathBuf::from(&args[pos + 1]);
        return run_receiver(&socket).await;
    }

    // Parent: pick a socket path, spawn the receiver process, run the sender.
    let dir = std::env::temp_dir().join(format!("parallax-ipc-demo-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    let socket = dir.join("demo.sock");

    let mut child = std::process::Command::new(std::env::current_exe()?)
        .arg("--receiver")
        .arg(&socket)
        .spawn()?;

    let sender = run_sender(&socket).await;

    let status = child.wait()?;
    let _ = std::fs::remove_dir_all(&dir);
    sender?;
    if !status.success() {
        return Err(parallax::error::Error::Pipeline(format!(
            "receiver process failed: {status}"
        )));
    }
    println!("[sender] receiver exited cleanly — two-process zero-copy round trip complete");
    Ok(())
}
