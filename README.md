# Parallax

[![crates.io](https://img.shields.io/crates/v/parallax-pipeline.svg)](https://crates.io/crates/parallax-pipeline)
[![docs.rs](https://img.shields.io/docsrs/parallax-pipeline)](https://docs.rs/parallax-pipeline)
[![license](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](#license)

A Rust-native streaming pipeline engine with zero-copy, multi-process shared memory at its core.

> Published on crates.io as **`parallax-pipeline`** (the bare `parallax` name was taken); the library is still imported as `parallax` — `use parallax::pipeline::Pipeline;` works as-is.

Parallax lets you build media and data pipelines the way GStreamer does — sources, transforms, sinks, connected into a graph — but with a design that is Rust-first throughout: memfd-backed buffers that are always ready for cross-process sharing, reference counts stored *in* shared memory so they work across processes, a hybrid Tokio + real-time-thread executor inspired by PipeWire, and a typed pipeline API with compile-time type checking.

> **Status:** Parallax is a young project (v0.1, pre-release). The core engine — memory, pipelines, executor, elements, caps negotiation, plugins — is implemented and covered by 1100+ tests. Some subsystems are still scaffolding (see [Project status](#project-status)). Expect API churn before 1.0.

**Requirements:** Linux only (memfd_create, SCM_RIGHTS, eventfd) · Rust **1.95+** (edition 2024)

## Highlights

- **Shared memory first** — every CPU buffer lives in a memfd-backed arena. There is no "convert to shared memory" step: any buffer can be sent to another process by passing one fd.
- **Cross-process reference counting** — refcounts are atomics stored *inside* the shared arena, so clone/drop work identically from any process. Slot release goes through a lock-free MPSC queue in shared memory; no coordination messages needed.
- **Hybrid scheduling** — the executor runs I/O-bound elements as Tokio tasks and RT-safe, low-latency elements on dedicated real-time threads (optionally `SCHED_FIFO`), connected by lock-free SPSC bridges with eventfd wakeup. Strategy is chosen automatically from element-declared `ExecutionHints`.
- **Progressive typing** — build pipelines dynamically from strings (`Pipeline::parse`) or programmatically, or use the typed API (`pipeline(src) >> map(..) >> filter(..)`) with compile-time type checking.
- **Caps negotiation** — elements declare multiple format+memory capabilities in preference order; the pipeline negotiates per link, including DMA-BUF vs CPU memory selection, and can auto-insert converters (opt-in).
- **Batteries included** — 100+ built-in elements: file/TCP/UDP/Unix/HTTP/WebSocket/Zenoh I/O, RTP/RTCP/RTSP, MPEG-TS and MP4 mux/demux, HLS/DASH output, V4L2/libcamera/PipeWire/ALSA/screen capture, codecs (H.264, AV1, Opus, AAC, FLAC/MP3/Vorbis, JPEG/PNG), KLV/STANAG metadata, and a rich set of flow/timing/transform utilities.
- **Pure Rust where possible** — rav1e, Symphonia, zune-jpeg, png, mp4, mpeg2ts-reader; C libraries only where unavoidable (OpenH264, dav1d, libopus, FDK-AAC).
- **Observability** — pipeline bus (GStreamer-style messages), pad probes, latency/framerate/drop tracers, `metrics`/`tracing` integration, DOT graph export.

## Quick start

```toml
[dependencies]
parallax-pipeline = "0.1"   # lib name is `parallax`: code writes `use parallax::...`
```

### Parse a pipeline from a string

```rust
use parallax::pipeline::Pipeline;

#[tokio::main]
async fn main() -> parallax::Result<()> {
    // GStreamer-like syntax: elements separated by `!`, properties as name=value
    let mut pipeline = Pipeline::parse(
        "videotestsrc width=320 height=240 num-buffers=60 ! videoconvert ! nullsink",
    )?;
    pipeline.run().await
}
```

The string grammar is a **linear chain**: `element prop=value ... ! element ...`. Values may be quoted strings, integers, floats, or booleans; `name=foo` gives the node a custom name for later retrieval. Branching (tee), caps filters, and bins are *not* expressible in the string syntax — use the programmatic API for those.

Built-in factory names usable in `parse`: `filesrc`, `filesink`, `videotestsrc`, `videoconvert`, `passthrough`, `inspect` (`tee` is a deprecated alias), `nullsource`, `nullsink`, plus `autovideosink` (feature `display`) and `v4l2src` (feature `v4l2`). More can be registered through a `PluginRegistry` (`Pipeline::parse_with_factory`).

### Build programmatically

```rust
use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::memory::SharedArena;
use parallax::pipeline::Pipeline;
use parallax::Result;

struct HelloSource { sent: bool }

impl Source for HelloSource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.sent {
            return Ok(ProduceResult::Eos);
        }
        self.sent = true;
        let msg = b"Hello, Parallax!";
        ctx.output()[..msg.len()].copy_from_slice(msg);
        Ok(ProduceResult::Produced(msg.len()))
    }
}

struct PrintSink;

impl Sink for PrintSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        println!("Received: {}", String::from_utf8_lossy(ctx.input()));
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let arena = SharedArena::new(1024, 4)?; // 4 slots × 1 KiB

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source_with_arena("src", HelloSource { sent: false }, arena);
    let sink = pipeline.add_sink("sink", PrintSink);
    pipeline.link(src, sink)?;

    pipeline.run().await
}
```

**Fan-out needs no element**: link one src-pad to several sinks and the executor hands each branch a refcounted clone. Each link carries a `LinkPolicy` — `Block` (default) back-pressures the source *and every sibling branch*, so use `link_lossy()` on branches allowed to fall behind. Fan-in uses the `Funnel` element; N-to-1 muxing and multi-pad linking use `link_pads`. There is also the fluent `PipelineBuilder` (see `examples/10_builder.rs`).

### Typed pipelines (compile-time checked)

```rust
use parallax::typed::{pipeline, from_iter, map, filter, collect};

fn main() -> parallax::Result<()> {
    let source = from_iter(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);

    let result = (pipeline(source)
        >> filter(|x: &i32| x % 2 == 0)   // `>>` is sugar for .then(...)
        >> map(|x: i32| x * 10))
        .sink(collect::<i32>())
        .run()?
        .into_inner();

    assert_eq!(result, vec![20, 40, 60, 80, 100]);
    Ok(())
}
```

Operators: `map`, `filter`, `filter_map`, `inspect`, `take`, `skip`; sources `from_iter`, `range`, `once`, `repeat_with`; sinks `collect`, `discard`, `for_each`, `fold`. Multi-source combinators: `zip`, `merge` (two-source interleave), `join` (hash join), and `temporal_join` (timestamp-aligned join with tolerance windows — useful for sensor fusion).

### Retrieve elements after construction

```rust
use parallax::elements::io::FileSrc;

let mut pipeline = parallax::pipeline::Pipeline::parse(
    "filesrc name=reader location=input.bin ! passthrough ! filesink location=out.bin",
)?;

// Downcast by name; wrong type returns None (no panic)
if let Some(src) = pipeline.get_element::<FileSrc>("reader") {
    println!("Reading from: {}", src.path().display());
}
if let Some(src) = pipeline.get_element_mut::<FileSrc>("reader") {
    *src = FileSrc::new("different.bin");
}
```

Unnamed elements get auto-generated names (`filesrc_0`, `passthrough_1`, …).

## Execution model

### Pipeline states (PipeWire-inspired)

```
Suspended <──> Idle <──> Running
                            │
          Error ◄───────────┘
```

| State | Resources | Meaning |
|-------|-----------|---------|
| `Suspended` | Deallocated | Minimal footprint (initial state) |
| `Idle` | Allocated | Negotiated and ready — "paused" |
| `Running` | Allocated | Actively processing |
| `Error` | Varies | Unrecoverable; recover via `Suspended` |

```rust
pipeline.prepare()?;   // Suspended → Idle: validate, negotiate caps, allocate
pipeline.activate()?;  // Idle → Running
pipeline.pause()?;     // Running → Idle (resources kept)
pipeline.suspend()?;   // Idle → Suspended (resources released)
```

`pipeline.run().await` (and `Executor::start`) auto-prepares a `Suspended` pipeline.

### Automatic strategy selection

Each element declares `ExecutionHints` (`rt_safe`, `processing`, `latency`, `memory`, `trust_level`, …). With the default `auto_strategy`, the executor assigns each element one of **two** strategies:

| Element characteristics | Strategy |
|--------------------------|----------|
| `rt_safe` **and** latency `UltraLow`/`Low` | **RealTime** — dedicated RT thread |
| I/O-bound, or anything else | **Async** — Tokio task |

If any element lands on an RT thread the pipeline runs in hybrid mode: the graph is partitioned, and async↔RT boundaries are bridged by lock-free SPSC ring buffers with eventfd signaling. A timer or hardware driver paces RT cycles (PipeWire-style activation records).

> Note: process isolation for untrusted elements was prototyped and removed; the `trust_level` and `uses_native_code` hints are currently informational only. All execution is in-process.

### Manual control

```rust
use parallax::pipeline::{Executor, ExecutorConfig, SchedulingMode, RtConfig};

let config = ExecutorConfig {
    auto_strategy: false,
    scheduling: SchedulingMode::Hybrid,  // Async | Hybrid | RealTime
    rt: RtConfig {
        quantum: 256,           // samples per cycle (~5.3 ms at 48 kHz)
        rt_priority: Some(50),  // SCHED_FIFO (requires CAP_SYS_NICE)
        ..Default::default()
    },
    ..Default::default()
};

let executor = Executor::with_config(config);
let handle = executor.start(&mut pipeline)?;  // synchronous; returns a PipelineHandle
handle.wait().await?;                         // or: executor.run(&mut pipeline).await
```

Presets: `ExecutorConfig::low_latency_audio()`, `ExecutorConfig::video(fps)`, `ExecutorConfig::hybrid()`.

## Memory model

All CPU memory is **memfd-backed** (`memfd_create` + `MAP_SHARED`): zero overhead compared to heap allocation, and always IPC-ready — one fd per arena, not per buffer.

| Backend | Use case | fd-shareable |
|---------|----------|--------------|
| `SharedArena` | Default arena; cross-process refcounting, lock-free release | Yes (memfd) |
| `FixedBufferPool` | Pipeline-level pool on top of `SharedArena`, blocking backpressure | Yes |
| `DmaBufSegment` / `DmaBufBuffer` | Zero-copy GPU/device path (V4L2 export, DRM, …) | Yes (DMA-BUF fd) |
| `MappedFileSegment` | Persistent, file-backed buffers | By path |
| `HugePageSegment` | 2 MB / 1 GB pages for TLB-heavy workloads | Not currently |

### Cross-process reference counting

`Arc` keeps its refcount on the process heap, so it cannot be shared. Parallax stores refcounts **in the shared memory itself**:

```
SharedArena layout (one memfd):
┌──────────────────────────────────────────────────────────────┐
│ ArenaHeader (64 B, cache-aligned)                            │
│   magic, version, slot_count, slot_size, arena_id            │
├──────────────────────────────────────────────────────────────┤
│ ReleaseQueue — lock-free MPSC ring in shared memory          │
│   head ← owner drains (single consumer)                      │
│   tail ← any process pushes (multi producer)                 │
├──────────────────────────────────────────────────────────────┤
│ SlotHeader[0..N] (8 B each): refcount + state atomics        │
├──────────────────────────────────────────────────────────────┤
│ SlotData[0..N] (aligned user data)                           │
└──────────────────────────────────────────────────────────────┘
```

- **Clone** → atomic increment in shared memory (works from any process)
- **Drop** → atomic decrement; on zero, push slot index to the release queue
- **Reclaim** → owner drains the queue in O(k), k = released slots

```rust
use parallax::memory::{SharedArena, SharedArenaCache};

// Process A: owner
let arena = SharedArena::new(4096, 16)?;          // 16 slots × 4 KiB
let mut slot = arena.acquire().expect("free slot");
slot.data_mut()[..5].copy_from_slice(b"hello");
let ipc_ref = slot.ipc_ref();                      // serializable reference
// send arena.fd() once + ipc_ref per buffer over a Unix socket (SCM_RIGHTS)

// Process B: client
let mut cache = SharedArenaCache::new();
// Safety: the fd must be a valid SharedArena memfd received over SCM_RIGHTS
unsafe { cache.map_arena(received_fd)? };
let client_slot = cache.get_slot(&ipc_ref).expect("live slot");
assert_eq!(&client_slot.data()[..5], b"hello");
// Both processes share the same atomic refcount — drop from either side is correct.
```

Helpers in `parallax::memory::ipc` (`send_fds`, `recv_fds`, `send_segment_handle`, …) wrap the SCM_RIGHTS plumbing, and the `IpcSrc`/`IpcSink` elements do all of it for you.

## Element library

Full catalog with feature flags in **[docs/elements.md](docs/elements.md)**. Summary:

| Category | Elements |
|----------|----------|
| **I/O** | `FileSrc`/`FileSink`, `FdSrc`/`FdSink`, `ConsoleSink` |
| **Testing** | `TestSrc`, `VideoTestSrc`, `DataSrc`, `NullSource`, `NullSink` |
| **App integration** | `AppSrc`/`AppSink` (+ handles), `AutoVideoSink` (display window) |
| **Network** | TCP/UDP/Unix src+sink (sync & async), UDP multicast, `HttpSrc` (byte-seekable via HTTP `Range`)/`HttpSink`, `WebSocketSrc`/`WebSocketSink`, Zenoh pub/sub/query |
| **RTP/RTSP** | `RtpSrc`/`RtpSink`, jitter buffer, `RtcpHandler`, payloaders/depayloaders for H.264/H.265/VP8/VP9 (+ Opus depay), `RtspSrc` client (Annex-B or length-prefixed framing, URL credentials, per-op timeouts). See `examples/57_rtsp_capture.rs` and `examples/58_rtsp_display.rs`; `just rtsp-server` serves a local test stream |
| **Flow** | `Queue` (watermarks, leaky modes), `Queue2` (stream/download/timeshift buffering), `Inspect` (passthrough counter), `Funnel`, `InputSelector`/`OutputSelector`, `Concat`, `Valve` |
| **Transforms** | `Map`, `Filter`, `FilterMap`, `FlatMap`, `Chunk`, `Batch`/`Unbatch`, buffer trim/slice/pad/split/join/concat, dedup/range/regex filters, `Gain`, `VideoScale`, `VideoConvertElement`, `AudioConvertElement`, `AudioResampleElement` |
| **Timing** | `Delay`, `Timeout`, `Debounce`, `Throttle`, `RateLimiter` |
| **Metadata** | `SequenceNumber`, `Timestamper`, `MetadataInject`/`MetadataExtract`, `TimestampDebug`, `KlvEncoder` (STANAG 4609/MISB) |
| **Mux/Demux** | `TsMux`/`TsMuxElement`, `TsDemux`, `Mp4Mux`/`Mp4FileSink`, `Mp4Demux`, `StreamIdDemux` |
| **Codecs** | H.264 (OpenH264), AV1 encode (rav1e) / decode (dav1d), Opus, AAC (FDK encode, Symphonia decode), FLAC/MP3/Vorbis (Symphonia), JPEG (zune-jpeg), PNG |
| **Devices** | `V4l2Src` (DMA-BUF export), `LibCameraSrc`, `PipeWireSrc`/`PipeWireSink`, `ScreenCaptureSrc` (XDG portal), `AlsaSrc`/`AlsaSink` (provides hardware clock) |
| **Streaming out** | `HlsSink` (TS segments + M3U8, ABR variants), `DashSink` (fMP4 + MPD, live & VOD) |
| **IPC** | `IpcSrc`/`IpcSink` (zero-copy cross-process), `MemorySrc`/`MemorySink` |

## Infrastructure at a glance

- **Bus & messages** — `Message`/`MessageKind` (Eos, Error, Warning, Tag, Qos, Buffering, StateChanged, …); poll, `await`, broadcast-subscribe, or consume as a `futures::Stream` (`bus.into_stream()`); `pipeline.run_with_bus(|msg| ...)` delivers them live. Every run posts exactly one terminal message — `Eos`, or `Error` naming the element that failed. See `examples/51_bus_messages.rs`.
- **Seeking** — `pipeline.seek_bytes(pos)` / `seek_time(t)`, `query_position()`, `query_duration()`, `query_seekable()`; segment events map PTS to running/stream time. See `examples/52_seeking.rs`.
- **Pad probes** — intercept buffers/events at any pad (`ProbeType::BUFFER`, `ProbeReturn::{Ok, Drop, Remove, Handled}`). See `examples/53_pad_probes.rs`.
- **Tracers** — `LatencyTracer`, `FramerateTracer`, `DropTracer`; activate with `PARALLAX_TRACERS="latency;framerate;drops"`; DOT graph dumps via `PARALLAX_DOT_DIR`. See `examples/54_tracers.rs`.
- **Flow control** — `FlowSignal`/`FlowPolicy` backpressure for live sources (block, drop, ring-buffer, adaptive), queue watermarks. See `examples/47_flow_control.rs`.
- **Type detection** — `TypeFindRegistry` sniffs common container/codec formats (MP4, Matroska, MPEG-TS, Ogg, WAV, FLAC, MP3, H.264, PNG, JPEG, …) from leading bytes, with extension fallback. See `examples/56_typefind.rs`.
- **Clocks** — `ClockTime` (ns, `NONE` sentinel), pluggable `Clock`/`ClockProvider`; the executor auto-selects the highest-priority provider (e.g. ALSA hardware clock) at start. See `examples/48_clock_provider.rs`.
- **Caps negotiation** — multi-format `ElementMediaCaps` with memory-type coupling (CPU vs DMA-BUF); converter auto-insertion controlled by `ConverterPolicy::{Deny, Warn, Allow}` (default `Deny`: fail with a helpful error instead of silently inserting converters). See `examples/17_multi_format_caps.rs` and `examples/45_dmabuf_negotiation.rs`.
- **Plugins** — dynamic loading of `cdylib` element plugins over a versioned `#[repr(C)]` ABI (`define_plugin!` or the `parallax-macros` attribute macros). See [docs/plugins.md](docs/plugins.md).

## Feature flags

Default features are **empty** — everything below is opt-in.

| Feature | Description | External deps |
|---------|-------------|---------------|
| `macros` | Plugin authoring proc-macros (`parallax-macros`) | — |
| `huge-pages` | 2 MB/1 GB huge page segments | — |
| **Network** | | |
| `http` | HTTP source/sink (ureq) | — |
| `websocket` | WebSocket source/sink (tungstenite) | — |
| `zenoh` | Zenoh pub/sub + query elements | — |
| `rtp` | RTP/RTCP elements, payloaders, jitter buffer | — |
| `rtsp` | RTSP client source (implies `rtp`) | — |
| **Containers** | | |
| `mpeg-ts` | MPEG-TS demuxer + muxer (pure Rust) | — |
| `mp4-demux` | MP4/MOV demuxer + muxer (pure Rust) | — |
| **Video codecs** | | |
| `h264` | H.264 encode/decode (OpenH264) | C++ compiler |
| `av1-encode` | AV1 encoder (rav1e, pure Rust) | nasm recommended |
| `av1-decode` | AV1 decoder (dav1d) | libdav1d |
| `software-codecs` | All of the above | |
| `vulkan-video` | Vulkan Video GPU decode (**experimental scaffold**, see status) | Vulkan 1.3 |
| **Audio codecs** | | |
| `audio-codecs` | FLAC+MP3+AAC+Vorbis decoders (Symphonia, pure Rust) | — |
| `audio-flac` / `audio-mp3` / `audio-aac` / `audio-vorbis` | Individual Symphonia decoders | — |
| `opus` | Opus encode/decode | libopus |
| `aac-encode` | AAC encoder (FDK-AAC — **license restrictions**) | libfdk-aac |
| **Images** | | |
| `image-codecs` | JPEG decode (zune-jpeg) + PNG encode/decode — pure Rust | — |
| `image-jpeg` / `image-png` | Individual image codecs | — |
| **Conversion** | | |
| `simd-colorspace` | SIMD YUV↔RGB via the `yuv` crate (AVX-512/AVX2/SSE4.1/NEON) | — |
| **Devices** | | |
| `v4l2` | V4L2 capture, DMA-BUF export | libv4l headers |
| `libcamera` | libcamera capture | libcamera |
| `pipewire` | PipeWire audio/video capture + playback | libpipewire |
| `alsa` | ALSA capture/playback + hardware clock | alsa-lib |
| `screen-capture` | XDG-portal screen capture (implies `pipewire`) | — |
| `device-capture` | pipewire + libcamera | |
| `device-all` | All device backends | |
| **Display** | | |
| `display` | `AutoVideoSink` window (winit + softbuffer) | — |
| **Placeholders** | | |
| `gpu`, `rdma` | Reserved for future use — currently no-ops | — |

## Examples

38 numbered examples, one concept each (`cargo run --example <name> [--features ...]`):

| Range | Topic |
|-------|-------|
| `01`–`08` | Basics: hello, transform, fan-out, funnel, queue, appsrc, file I/O, TCP |
| `09`–`11` | Typed pipelines, builder DSL, buffer pools |
| `13`–`18`, `20` | Codecs: PNG (`image-codecs`), H.264 (`h264`), AV1 (`av1-encode`), MPEG-TS (`mpeg-ts`), caps negotiation, GPU decode (`vulkan-video`), Opus (`opus`) |
| `22`–`26` | Devices & streaming: V4L2 (`v4l2`), display (`display`), HLS, DASH |
| `41`–`46` | Converters, PipeWire (`pipewire`), ALSA (`alsa`), libcamera (`libcamera`), DMA-BUF negotiation (`v4l2`), screen capture (`screen-capture,h264,mp4-demux`) |
| `47`–`56` | Infrastructure: flow control, clocks, element retrieval, hybrid scheduling, bus, seeking, probes, tracers, Queue2 buffering, typefind |

(Numbers 12, 19, 21, 27–40 are retired/unassigned.)

## Performance

| Operation | Cost | Notes |
|-----------|------|-------|
| Buffer clone (any process) | O(1) | Atomic increment in shared memory |
| Buffer access | O(1) | Direct pointer into the mapped arena |
| Slot release | O(1) | Lock-free MPSC queue push |
| Slot reclaim | O(k) | k = released slots, not pool size |
| Fan-out to N sinks | O(N) refcount increments | No data copies |
| Cross-process send | O(1) after setup | Arena fd sent once; then only tiny refs |
| 1080p I420→RGBA (`simd-colorspace`) | ~0.9 ms | AVX2/AVX-512 via `yuv` crate |

`cargo bench` runs the colorspace benchmark; the memory/throughput benches are being rewritten after a memory-API refactor.

## Project status

Honest accounting of where things stand:

- **Solid:** memory subsystem, pipeline graph + executor (async & hybrid RT), element library, caps negotiation, bus/probes/tracers/seek, plugin loading, typed pipelines. 1112 tests pass.
- **Functional but young:** RTSP client, HLS/DASH sinks, MP4/TS muxing, device capture (V4L2/PipeWire/ALSA/libcamera/screen).
- **Scaffolding:** `vulkan-video` — the Vulkan context, video session, DPB, and DMA-BUF import/export are real, but the H.264 decoder does not yet submit actual decode commands; no GPU encode. Treat it as a preview.
- **Removed:** process isolation/sandboxing for untrusted elements was prototyped and backed out (fork-safety concerns); it may return in a different form. See [docs/security.md](docs/security.md).
- **Not started:** RDMA, CUDA interop.

## Documentation

| Doc | Contents |
|-----|----------|
| [Getting started](docs/getting-started.md) | Install, first pipeline, custom elements |
| [Architecture](docs/architecture.md) | System overview: graph, executor, memory, negotiation |
| [Pipelines](docs/pipeline.md) | Parse syntax, states, bus, events, seeking, probes, tracers, flow control |
| [Scheduling](docs/scheduling.md) | Executor internals, RT threads, drivers, bridges, clocks |
| [Memory](docs/memory.md) | SharedArena, buffer pools, DMA-BUF, IPC |
| [Elements](docs/elements.md) | Complete element catalog with feature flags |
| [Formats & negotiation](docs/formats.md) | Caps model, negotiation, converters, SIMD |
| [Plugins](docs/plugins.md) | Writing and loading dynamic plugins |
| [API overview](docs/api.md) | Map of the public API (`cargo doc` for the full reference) |
| [Security](docs/security.md) | Current security model and its limits |
| [Design](docs/design.md) | Design rationale and competitive landscape |

## License

Licensed under either of

- MIT license ([LICENSE-MIT](LICENSE-MIT))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

## Contributing

Contributions are welcome! Run `just check` (format + clippy + tests) before submitting a PR.
