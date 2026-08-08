# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Parallax** is a Rust-native streaming pipeline engine (a GStreamer alternative) built on zero-copy shared memory, hybrid async+realtime scheduling, and progressive typing. Linux-only. Workspace: the root crate (package **`parallax-pipeline`** on crates.io, `[lib] name = "parallax"` — the bare crates.io name is squatted, so package and lib names intentionally differ; imports stay `use parallax::`), `parallax-macros/` (package `parallax-pipeline-macros`, lib `parallax_macros`), `examples/example-plugin` (cdylib plugin demo, publish = false).

- Edition **2024**, MSRV **1.95** (enforced by the CI `msrv` job; do not claim 1.75 or 1.85 — both stale).
- Default features are empty; most media functionality is feature-gated.
- ~61k lines of Rust; 1100+ tests, all passing with default features.

### Core Principles

1. **Shared memory first**: all CPU buffers are memfd-backed (always IPC-ready, one fd per arena)
2. **Cross-process refcounting**: refcounts live *inside* shared memory (atomics), release via lock-free MPSC queue
3. **Progressive typing**: dynamic pipelines (string/programmatic) + typed pipelines (compile-time checked)
4. **Sync processing, async orchestration**: element hot paths are sync; orchestration is Tokio; RT-safe elements can run on dedicated RT threads
5. **Pure Rust codecs where possible**; C libraries only when unavoidable

### Key Design Decisions

| Aspect | Choice |
|--------|--------|
| Async runtime | Tokio |
| Channels | kanal (MPMC sync+async) for local links; SPSC ring + eventfd for async↔RT bridges |
| Graph | daggy (enforces DAG) |
| Parser | winnow |
| Serialization | rkyv (network links, IPC types) |
| Errors | thiserror (`Error` enum in `src/error.rs`, `#[non_exhaustive]`) |
| Metrics/logging | metrics-rs + tracing |
| Linux APIs | rustix (memfd, mmap, SCM_RIGHTS, eventfd) |
| Plugin ABI | **Hand-rolled `#[repr(C)]` descriptor + `extern "C"` fns, loaded via libloading.** NOT stabby (old docs claiming stabby are wrong; stabby appears in Cargo.lock only as a transitive dep of zenoh). |
| Object-safe async traits | trait-variant + dynosaur (`DynAsyncElement`) |
| GPU | Vulkan Video via ash — **experimental scaffold** (see Gotchas) |

## Build & Test Commands

```bash
just test          # cargo nextest run
just test-one NAME # single test
just lint          # clippy -D warnings
just lint-all      # clippy --all-features
just check         # fmt-check + lint + test
just check-sensor  # check+test+clippy the sensor combo (zenoh,h264,v4l2,rtp,rtsp,image-jpeg,hotplug — mirrors CI)
just bench         # criterion benchmarks
just watch         # auto-run tests on change
just coverage      # cargo llvm-cov nextest

# Direct cargo
cargo nextest run
cargo clippy -- -D warnings
cargo doc --no-deps
cargo check --all-targets              # default features
cargo check --features "h264,mpeg-ts"  # etc. for gated code
```

Feature-gated code is NOT compiled by default — after touching gated modules (codecs, devices, rtp, vulkan…), check with the relevant features enabled. CI (`.github/workflows/ci.yml`) runs default tests plus the sensor combo; it deliberately does NOT run `--all-features` (dav1d/alsa/pipewire/libcamera need system libs, and vulkan-video carries lint debt tracked in #3).

## Source Tree

```
src/
├── lib.rs              # module decls + prelude
├── error.rs            # Error/Result (variants: PoolExhausted, BufferPool, AllocationFailed,
│                       #   InvalidSegment, ValidationFailed, InvalidCaps, Config, Pipeline,
│                       #   Element, Io, System, Device[feature-gated])
├── buffer.rs           # Buffer<T=()>, MemoryHandle, DmaBufBuffer
├── metadata.rs         # Metadata (pts/dts/duration/sequence/stream_id/flags/rtp/format/offset
│                       #   + typed custom map), BufferFlags, RtpMeta
├── clock.rs            # ClockTime (NONE sentinel), Clock, ClockProvider, SystemClock, PipelineClock
├── codec/annexb.rs     # Annex-B helpers (nal_units/has_idr/extract_param_sets/annex_b_to_avcc)
│                       #   — ALWAYS compiled, no codec feature needed
├── control.rs          # Controllable trait + every runtime handle (EncoderControl,
│                       #   EncoderStatsHandle, RateControlMode, KeyframeHandle, ScaleControl, …)
├── format.rs           # CapsValue, Video/AudioFormat(+Caps), PixelFormat, MediaFormat,
│                       #   MemoryCaps, MemoryLayout, FormatMemoryCap, ElementMediaCaps, Caps
├── event/              # Event enum (StreamStart/Segment/Tags/Eos/CapsChanged/Gap | Seek/Qos/
│                       #   LatencyQuery | FlushStart/FlushStop/Custom), PipelineItem, TagList
├── memory/             # SharedArena, SharedSlotRef, SharedIpcSlotRef, SharedArenaCache,
│                       #   BufferPool/FixedBufferPool/PooledBuffer, DmaBufSegment,
│                       #   HugePageSegment, MappedFileSegment, AtomicBitmap, ipc (SCM_RIGHTS),
│                       #   defaults (slot size/count constants)
├── element/            # traits.rs (Source/Sink/Element/Transform/Async*/Demuxer/Muxer/
│                       #   SyncElement, ExecutionHints, adapters, AsyncElementDyn/DynAsyncElement)
│                       # pipeline_element.rs (PipelineElement, ProcessOutput, SimpleSource/
│                       #   SimpleSink/SimpleTransform, Src/Snk/Xfm wrappers)
│                       # context.rs (ProduceContext/ConsumeContext/ProcessContext, ProduceResult)
│                       # pad.rs, muxer.rs (MuxerSyncState/SyncMode/PadInfo/StreamType)
├── elements/           # Built-ins by category: io, testing, network, rtp, flow, transform,
│                       #   metadata (KLV), app, ipc, timing, util, demux, mux, codec, device,
│                       #   streaming (HLS/DASH)
├── pipeline/           # graph.rs (Pipeline, states, ConverterPolicy, DOT/JSON export, probes/
│                       #   seek surface), unified_executor.rs (Executor/ExecutorConfig/
│                       #   PipelineHandle/ElementStrategy), rt_scheduler.rs, rt_bridge.rs,
│                       #   driver.rs, parser.rs, factory.rs, bus.rs, events.rs, tags.rs,
│                       #   seek.rs, probe.rs, tracer.rs, typefind.rs, flow.rs, builder.rs
├── negotiation/        # NegotiationSolver (per-link), ConverterRegistry, builtin registry
├── converters/         # REAL VideoConvert/AudioConvert/AudioResample/VideoScale impls
├── link/               # LocalLink (kanal), IpcPublisher/IpcSubscriber (memfd+SCM_RIGHTS),
│                       #   NetworkSender/Receiver (TCP+rkyv, "PRLX" magic)
├── typed/              # TypedSource/Sink/Transform, pipeline builder (>> operator), operators,
│                       #   multi_source (merge/zip/join/temporal_join), bridge to dynamic
├── temporal/           # Timestamp (SEPARATE from ClockTime), TimeRange, TemporalJoin,
│                       #   AlignmentStrategy, JoinWindow
├── gpu/                # Vulkan Video scaffold (context/session/dpb/memory real; decoder stub)
├── plugin/             # descriptor.rs (#[repr(C)] ABI, define_plugin!), loader.rs (libloading),
│                       #   registry.rs; entry symbol: parallax_plugin_descriptor; ABI version 1
└── observability/      # metrics-rs helpers (parallax_* metric names), tracing spans
```

Docs live in `docs/` (user guides + `docs/research/` for historical design notes); implementation plans in `plans/`.

## Element System

Two coexisting generations:

**Legacy traits** (`element/traits.rs`) — most built-ins use these:
- `Source::produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult>`
- `Sink::consume(&mut self, ctx: &ConsumeContext) -> Result<()>`
- `Element::process(&mut self, buffer: Buffer) -> Result<Option<Buffer>>` (1-to-0/1)
- `Transform::transform(&mut self, buffer) -> Result<Output>` (multi-output; blanket impl for all `Element`)
- `AsyncSource`/`AsyncSink`/`AsyncTransform` (async variants), `Demuxer` (1-to-N via `RoutedOutput`), `Muxer` (N-to-1 via `MuxerInput`), `SyncElement` (RT path)
- Optional methods: `output_caps()`, `output_media_caps()`/`input_media_caps()`, `execution_hints()`, `flush()`, `handle_upstream_event()`/`handle_downstream_event()`, `is_seekable()`, `query_position()`/`query_duration()`, `flow_policy()`, `handle_flow_signal()`, `as_clock_provider()`
- Adapters wrap author traits into the type-erased runtime trait: `SourceAdapter` (also `with_arena`/`with_pool`), `SinkAdapter`, `ElementAdapter`, `TransformAdapter`, `MuxerAdapter`, etc. → `DynAsyncElement` (dynosaur-generated)

**Unified simple traits** (`element/pipeline_element.rs`) — no adapter boilerplate:
- `SimpleSource::produce() -> Result<ProcessOutput>`, `SimpleSink::consume(&Buffer)`, `SimpleTransform::transform(Buffer) -> Result<ProcessOutput>`
- Wrap in `Src(...)`, `Snk(...)`, `Xfm(...)` and pass to `pipeline.add_element(name, ...)`
- `ProcessOutput::{None, Buffer(..), Buffers(..), Eos, Pending}`

**ProduceResult** (pool-aware sources): `Produced(usize)` (wrote into ctx buffer), `Eos`, `OwnBuffer(Buffer)`, `OwnDmaBuf(DmaBufBuffer)`, `WouldBlock`.

**ExecutionHints** `{ rt_safe, trust_level, crash_safe, uses_native_code, processing, latency, memory }` with profiles `ExecutionHints::rt_safe()`, `io_bound()`, `cpu_intensive()`, `low_latency()`, etc. There is **no `Affinity` type/`affinity()` method** — scheduling derives purely from hints.

**Flush**: executor calls `flush()` repeatedly at EOS until it returns `None`/`Output::None` — implement it in encoders/muxers to drain buffered data.

**Runtime control** (`src/control.rs`, always compiled): `Executor::start()` *moves* elements into their tasks, so `get_element_mut()` returns `None` on a running pipeline. The only way to mutate a live element is an `Arc<Atomic*>` **handle cloned before `start()`**. Every controllable element implements `Controllable`, so the accessor is always `control()`: `EncoderControl` (bitrate/GOP/QP/rate-control/skip-frames/keyframe), `EncoderStatsHandle` (read-only counters), `KeyframeHandle`, `ScaleControl`, `ThrottleControl`, `JpegQualityControl`, `ValveControl`, `FlowStateHandle`. See `docs/elements.md` § Runtime control.

**Geometry-in-Metadata invariant**: geometry travels in-band. **No element takes dimensions at construction** (`H264EncoderConfig::new()`, `JpegEncoder::new()`, `PngEncoder::new()`, `VideoScale::new()` are all no-arg); an element that cannot determine its geometry from `Metadata` **errors** rather than falling back to a stale constructor value. The exception still to fix: `EncoderElement::new(enc, format: VideoFormat)`.

## Pipeline

### Construction

```rust
let mut p = Pipeline::parse("filesrc location=in.bin ! passthrough ! filesink location=out.bin")?;
// or
let mut p = Pipeline::new();
let src  = p.add_source("src", MySource);           // also: add_source_with_arena / _with_pool
let xfm  = p.add_transform("xfm", MyTransform);     // add_filter for Element impls
let sink = p.add_sink("sink", MySink);
let el   = p.add_element("el", Src(MySimpleSource)); // unified API
p.link(src, xfm)?;                                   // = link_pads(src,"src",xfm,"sink")
p.link_pads(xfm, "src", sink, "sink")?;
```

- `get_element::<T>(name)` / `get_element_mut::<T>(name)` downcast by node name; `name=` in parse strings sets the name, otherwise `{factory}_{index}`.
- Fluent alternative: `PipelineBuilder` (typestate; `source().then().tee(..).sink().build()`), see `pipeline/builder.rs`.
- `add_node` is `pub(crate)` — never show it in docs/examples.

### Parse grammar (IMPORTANT limitations)

- Strictly **linear chains**: `elem prop=val ! elem ...`. NO caps filters, NO tee branching (`t. !`), NO bins. Fan-out requires the programmatic API.
- Property values: quoted strings, bare words, ints, floats, bools (`true/false/yes/no`).
- Registered factory names (only these work in `parse`): `nullsource`, `nullsink`, `passthrough`, `tee`, `filesrc`, `filesink`, `videoconvert`, `videotestsrc`, + `autovideosink` [display], `v4l2src` [v4l2]. Extendable via `ElementFactory::with_plugin_registry`. Do NOT write doc examples with unregistered names like `decoder`, `audiosrc`, `h264enc`.

### States (PipeWire-inspired)

`Suspended <-> Idle <-> Running`, plus `Error` (recover via Suspended).
`prepare()` (validate+negotiate+allocate; converter policy applies), `activate()`, `pause()`, `suspend()`. `run()`/`start()` auto-prepare from Suspended.

### Execution

- `pipeline.run().await` — run to completion. `pipeline.run_with_bus(|msg| bool).await` — with message handler.
- `Executor::with_config(cfg)`; **`executor.start(&mut p)` is SYNC** and returns `PipelineHandle`; `handle.wait().await`. Only `executor.run()` is async. Never write `executor.start(...).await`.
- `ExecutorConfig { scheduling: SchedulingMode::{Async|Hybrid|RealTime}, auto_strategy: bool (default true), channel_capacity, rt: RtConfig{quantum, rt_priority, data_threads, bridge_capacity}, driver }`. Presets: `low_latency_audio()`, `video(fps)`, `hybrid()`.
- **`ElementStrategy` has exactly two variants: `Async` and `RealTime`.** Auto rule: `rt_safe && latency ∈ {UltraLow, Low}` → RealTime; else Async. `trust_level`/`uses_native_code` are currently IGNORED (process isolation was removed in commit da6df59 — never document an "isolated process" strategy).
- Hybrid mode: `RtScheduler::partition_graph` splits nodes, boundary edges get `AsyncRtBridge` (lock-free SPSC + eventfd), RT threads use PipeWire-style `ActivationRecord`s, paced by `TimerDriver`/`ManualDriver`.
- `SchedulingMode::RealTime` with zero RT-safe nodes falls back to fully-async and logs a `warn!` (it does not error). `ExecutorConfig::scheduling` is authoritative — the executor overrides `RtConfig::mode` from it, so `with_scheduling(RealTime)` works on its own. `auto_strategy` only *reports* that RT-safe elements exist; it does not promote an Async config to Hybrid.

### Bus & events

- `Bus`/`BusHandle`; `MessageKind::{StateChanged, Eos, Error, Warning, Info, Tag, DurationChanged, StreamCollection, Qos, LatencyChanged, Buffering, AsyncStart, AsyncDone, ClockLost, NewClock, SeekDone, Element, Application}`.
- Consume: `bus.poll()`, `bus.next().await`, `bus.subscribe()` (broadcast), `bus.into_stream()` (`futures::Stream`, works with `select!`), `bus.wait_for_eos_or_error().await`.
- Separate typed event channel: `PipelineHandle::subscribe() -> EventReceiver` (`pipeline/events.rs`) — distinct from the bus.
- In-band events (`src/event/`): downstream `StreamStart/Segment/Tags/Eos/CapsChanged/Gap`, upstream `Seek/Qos/LatencyQuery`, bidirectional `FlushStart/FlushStop/Custom`; `PipelineItem::{Buffer, Event}` keeps ordering through channels.

### Seeking / probes / tracers / typefind / flow control

- Seek: `pipeline.query_seekable()/query_position()/query_duration()`, `seek_bytes(u64)`, `seek_time(ClockTime)`, `seek(&SeekEvent)`. `SegmentEvent::to_running_time/to_stream_time` for timestamp mapping. FileSrc implements `SeekableSource`.
- Probes: `pipeline.add_probe(PadRef::src(node), ProbeType::BUFFER, |data| ProbeReturn::Ok)`; types `BUFFER, EVENT_DOWN, EVENT_UP, EVENT_FLUSH, BLOCK, IDLE`; returns `Ok/Drop/Remove/Handled`.
- Tracers: `TracerRegistry` + `LatencyTracer`/`FramerateTracer`/`DropTracer`; env `PARALLAX_TRACERS="latency;framerate;drops"`; DOT dumps via `PARALLAX_DOT_DIR`; `pipeline.to_dot()`/`to_json()`; `pipeline.stats_snapshot()`.
- Typefind: `TypeFindRegistry::with_builtins()`, `detect(&bytes)`, `detect_from_extension`, `detect_with_fallback`.
- Flow control (`pipeline/flow.rs`): `FlowSignal::{Ready,Busy,Drop,EosAck,Pausing,Stopping}`, `FlowPolicy::{Block, Drop{..}, RingBuffer{..}, Adaptive{..}}`, `FlowStateHandle`, `WaterMarks`. Queue element: `Queue::new(n).with_flow_control()/.with_water_marks(wm)`, `queue.flow_state_handle()`; live sources accept `set_flow_state(handle)` and check `should_produce()`.
- `Queue2` (elements/flow): `Queue2::stream(bytes)`, `::download(path, total)`, `::timeshift(path, bytes)`, `.with_watermarks(low, high)`; posts `MessageKind::Buffering`.

## Memory Model

- `SharedArena::new(slot_size, slot_count)` — memfd + MAP_SHARED. Layout: `ArenaHeader (64B) | ReleaseQueue (MPSC ring, 1024 entries) | SlotHeader[N] (8B: refcount+state atomics) | SlotData[N]`.
- `arena.acquire() -> Option<SharedSlotRef>` (owner only), `slot.ipc_ref() -> SharedIpcSlotRef` (rkyv-serializable), `unsafe SharedArena::from_fd(fd)` / `SharedArenaCache::map_arena` (client), `arena.reclaim()` drains released slots O(k).
- `FixedBufferPool::new(buffer_size, count) -> Result<Arc<Self>>`; `acquire()` blocks (condvar backpressure), `try_acquire()`, `PooledBuffer` returns slot on drop or `into_buffer()` detaches. Sources use `ctx.acquire_buffer()` inside `produce()`.
- `Buffer<T=()>` = `MemoryHandle` (slot+offset+len) + `Metadata`; clone = atomic increment; `slice()` = zero-copy sub-buffer.
- `Metadata`: fields `pts/dts/duration: ClockTime`, `sequence`, `stream_id`, `flags: BufferFlags`, `rtp`, `format`, `offset`; typed custom map `set/get/get_mut/remove` with `&'static str` keys (`"domain/type"` convention: `stanag/*`, `h264/*`, `app/*`, …); helpers `set_bytes/get_bytes`, `set_klv()/klv()`, `set_sei()/sei()`. Transforms MUST clone/propagate metadata or PTS is lost.
- DMA-BUF: `DmaBufSegment::from_fd`, `DmaBufBuffer`, `ProduceResult::OwnDmaBuf`; V4L2 exports via `dmabuf_export: true` config.
- IPC helpers: `memory::ipc::{send_fds, recv_fds, send_segment_handle, recv_segment_handle}` (SCM_RIGHTS, max 4 fds/message).

## Caps & Negotiation

- Constraint model: `CapsValue<T>::{Fixed, Range, List, Any}`; `VideoFormatCaps`/`AudioFormatCaps` → `FormatCaps`; `MemoryCaps` (`cpu_only()`, `dmabuf_only()`, `dmabuf_preferred()`, `any()`); pair = `FormatMemoryCap`; element declares preference-ordered `ElementMediaCaps` via `output_media_caps()`/`input_media_caps()`.
- Solver (`negotiation/solver.rs`) negotiates **per link, first intersection wins** (NOT a global constraint solver).
- On a mismatch, `diff_caps()` reports which **`ConvertAxes`** (`FORMAT|GEOMETRY|RATE|MEMORY`) conflict, and `ConverterRegistry::plan()` covers them with the cheapest **chain** (several converters per link). It returns `None` — a negotiation error — if it cannot cover them ALL; a partial chain is never emitted. Chain order: downscale ⇒ scale-then-convert, upscale ⇒ convert-then-scale.
- Registered: `videoconvert` (FORMAT), `videoscale` (GEOMETRY), `audioconvert` (FORMAT), `audioresample` (RATE), `memorycopy` (MEMORY), `identity` (NONE — so it can never be auto-inserted). Factories take a `ConversionRequest` and configure themselves from the sink's caps.
- Auto-insertion only fires when the **source** side pins the property (`Any ∩ Fixed = Fixed` never conflicts) — i.e. v4l2/screen-capture-rooted graphs and `EncoderElement`.
- `ConverterPolicy::{Deny (default), Warn, Allow}`; `pipeline.prepare()` fails with a helpful error under Deny; `prepare_with_auto_converters()` = Warn; `set_converter_policy(...)`.
- `MemoryLayout::{NONE, SSE, AVX, AVX512}` requests aligned buffers (arena constructors `new_avx`/`new_avx512`).

## Clock System

- `ClockTime` — ns, `ZERO`/`MAX`/`NONE` sentinels, saturating+NONE-propagating arithmetic.
- `Clock` trait (`now/flags/resolution/name`), `SystemClock` (monotonic), `PipelineClock` (base_time; `running_time()`, async `wait_until/wait_for`).
- `ClockProvider::{provide_clock, clock_priority}` — priority bands: 0–99 software, 100–199 hardware audio, 200–299 network, 300+ PTP.
- **Automatic selection**: executor calls `pipeline.select_clock()` before start — highest-priority element-provided clock wins (e.g. `AlsaSink` provides a hardware clock at priority 100). Manual override: `set_clock`/`use_clock_from`.
- Sources read the clock via `ctx.clock()` / `ctx.running_time()` in `produce()`.

## Codecs & Devices (feature-gated)

| What | Types | Feature |
|------|-------|---------|
| H.264 | `H264Encoder`/`H264Decoder` (implement `Element` directly). `H264EncoderConfig::new()` takes **no dimensions**; geometry follows `Metadata` (a resize rebuilds the encoder + IDR). Live control via `control()`: a **bitrate change is seamless** (openh264-sys2 `SetOption(ENCODER_OPTION_BITRATE)`, no IDR); GOP/QP/rate-control/skip-frames rebuild (IDR). A no-op change does nothing. Counters via `stats()` → `EncoderStatsHandle`. **Defaults**: `rate_control = Bitrate`, `bitrate_bps = 2_000_000` (Bitrate + 0 is an error), `skip_frames = false`. In Bitrate mode with skipping off, `qp` is a quality *ceiling* (the band opens to 51) | `h264` |
| H.264 hardware | `V4l2M2mH264Encoder` (impl `VideoEncoder`, wrap in `EncoderElement`), `find_m2m_encoder(b"H264")` device probe; `V4l2CodedFormat::Fwht` is test-only (vicodec) | `v4l2-m2m` (build needs libclang + kernel headers) |
| AV1 | `Rav1eEncoder` (impl `Element` directly AND `VideoEncoder`; drains lookahead via `Element::flush`), `Dav1dDecoder` (impl `Element`) | `av1-encode` / `av1-decode` |
| Audio dec | `SymphoniaDecoder` (impl `Element`) | `audio-flac/mp3/aac/vorbis` |
| Opus | `OpusEncoder::new(rate, ch, bitrate, OpusApplication)` / `OpusDecoder` (impl `AudioEncoder`/`AudioDecoder`, wrap in `AudioEncoderElement`/`AudioDecoderElement`) | `opus` |
| AAC enc | `AacEncoder` | `aac-encode` (FDK license!) |
| Images | `JpegEncoder`/`JpegDecoder`, `PngEncoder`/`PngDecoder` | `image-*` |
| Containers | `TsMux`/`TsMuxElement`/`TsDemux` [`mpeg-ts`], `Mp4Mux`/`Mp4FileSink`/`Mp4Demux` [`mp4-demux`] | |
| RTP | `RtpSrc/Sink`, `RtpH264Pay/Depay`, H265/VP8/VP9 pay/depay, `RtpOpusDepay` (no Opus pay), `RtpJitterBuffer`, `RtcpHandler` | `rtp` |
| RTSP | `RtspSrc` (client only, via retina; session API — bridge into pipelines via `AppSrc`). `RtspFrameFormat::AnnexB` default (SPS/PPS in-band per keyframe, feeds `H264Decoder` directly; `LengthPrefixed` for MP4 mux); `connect_timeout` enforced per operation; `user:pass@` URL creds auto-lifted. Local test stream: `just rtsp-server`. Examples 57 (capture) / 58 (display) | `rtsp` |
| HLS/DASH | `HlsSink`, `DashSink` (+ configs) — NOT feature-gated | — |
| Devices | `V4l2Src` (DMA-BUF export, `framerate` knob), `LibCameraSrc` (libcamera 0.7; `framerate` via FrameDurationLimits, best-effort on UVC; process-wide shared `CameraManager` — a second live instance is fatal in libcamera), `PipeWireSrc/Sink`, `ScreenCaptureSrc`, `AlsaSrc/Sink` (clock provider) | `v4l2`/`libcamera`/`pipewire`/`screen-capture`/`alsa` |
| Hotplug | `DeviceMonitor` (udev `video4linux` + libcamera events folded in when both features on; one physical USB cam → one `Added` per backend) | `hotplug` (+`libcamera` for folding) |
| KLV | `KlvEncoder`, `StanagMetadataBuilder` (elements/metadata) | — |

Codec traits: `VideoEncoder`/`VideoDecoder`, `AudioEncoder`/`AudioDecoder` (with `flush()` to drain at EOS; `VideoEncoder` also has `force_keyframe()` plus defaulted `set_bitrate`/`set_keyframe_interval`/`set_qp` that `Err` when the codec cannot comply — rav1e/opus/aac do not override them). Note the inconsistency: some codecs implement the traits (rav1e, opus, aac, v4l2-m2m), others implement `Element` directly (openh264, dav1d, symphonia). `EncoderElement::new(enc, format: VideoFormat)` maps caps pixel formats to codec ones with per-format strides (I420/I422/I444/NV12/10-bit), errors on RGB/packed input (needs `VideoConvert` upstream), and renegotiates from per-buffer `Metadata.format`.

**Video dimensions in metadata**: two conventions coexist — `Metadata.format` (`MediaFormat::VideoRaw`, read by `EncoderElement`) and the legacy `"width"`/`"height"` `u64` custom keys (set by `H264Decoder`, read by `AutoVideoSink`). Use `Metadata::set_video_dims()` / `video_dims()`, which write and read **both**; updating only one leaves the other stale and silently mis-sizes frames downstream.

**`elements::codec` is compiled only when at least one codec feature is enabled** (see the `#[cfg(any(...))]` on `pub mod codec` in `elements/mod.rs`). Consequence: unit tests inside codec modules (including the always-present `EncoderElement`) do NOT run under default features — they run in CI's sensor combo and feature-specific jobs. When adding a codec feature, add it to that cfg list or the module silently won't compile.

Muxer sync: `MuxerSyncState`/`MuxerSyncConfig::new().with_mode(SyncMode::{Auto|Strict|Loose|Timed}).with_interval_ms(..)`, `PadInfo::new(name, StreamType).required()/.optional()`; `TsMuxConfig::new().add_track(TsMuxTrack::new(pid, TsMuxStreamType::H264).video())`.

## Plugin System

- ABI: `PluginDescriptor`/`ElementDescriptor` (`#[repr(C)]`), `PARALLAX_ABI_VERSION = 1`, entry symbol `parallax_plugin_descriptor`, loaded with libloading. Element instances cross the boundary as double-boxed `DynAsyncElement` raw pointers.
- Authoring: `define_plugin!` macro_rules (uses `paste`; what `examples/example-plugin` uses) or `parallax-macros` proc-macros (`#[pipeline_element(...)]` + `plugin!{}`, feature `macros`).
- `PluginLoader::load_from_path` (unsafe) validates ABI version + descriptor; `PluginRegistry` indexes elements and can back `Pipeline::parse` names.
- Search paths for `load_by_name`: `.`, `/usr/lib/parallax/plugins`, `/usr/local/lib/parallax/plugins`.

## Observability

- Metrics: `observability::metrics` — `init_metrics()`, `ElementMetrics`, `PipelineMetrics`, metric names `parallax_buffers_*`, `parallax_bytes_*`, `parallax_processing_time_ns`, etc.
- Tracing: `observability::tracing_support` (`TracingConfig`, span helpers).
- Env vars: `PARALLAX_TRACERS` (tracer activation), `PARALLAX_DOT_DIR` (DOT dumps on state transitions).

## Gotchas & Pitfalls (read before writing code or docs)

1. **No process isolation.** Removed in da6df59. `ElementStrategy` = Async|RealTime only. Old docs mentioning `run_isolated`, `ElementSandbox`, "isolated process" strategy, or `src/execution/` are describing deleted code.
2. **Two `PixelFormat` enums**: `format::PixelFormat` (15 variants, caps) vs `converters::PixelFormat` (9 variants, actual conversion). Separate types, but conversion is explicit: `From<converters::PixelFormat> for format::PixelFormat` (total) and `TryFrom` back (partial). Same split for `SampleFormat` (`format::` vs `converters::audio::`), with no conversion impls.
3. **`Tee` no longer exists** — it was a 1-in/1-out passthrough counter that never fanned out, and is now `Inspect` (`tee` remains a deprecated parse alias). **Fan-out needs no element**: link one src-pad to several sinks. Each `Link` carries a `LinkPolicy` (`Block` default | `Drop`) — a `Block` branch that fills its channel back-pressures the source *and every sibling*, so use `pipeline.link_lossy()` on branches allowed to fall behind. EOS is never dropped.
4. **`temporal::Timestamp` ≠ `clock::ClockTime`** — separate types, no conversion provided. Typed temporal joins use `Timestamp`; buffers/events use `ClockTime`.
5. **`typed::merge` is 2-source alternating interleave**, not N-way funnel. `typed::run_async` is `spawn_blocking` around the sync loop, not native async.
6. **`Executor::start` is sync**; only `run()` is async.
7. **Arena format is v4** (`ARENA_VERSION`): `ArenaHeader` records `slot_stride` and `alignment`, so `from_fd` reads the true stride instead of assuming 64-byte rounding, and validates it. Cross-process `new_avx` (32-byte) arenas are safe.
8. **`HugePageSegment` is memfd-backed** (`MFD_HUGETLB` + `MAP_SHARED`), so `ipc_handle()` returns a real fd and `MemoryType::HugePages.supports_ipc()` is honest. `new()` errors when the hugetlb pool is empty; `new_or_fallback()` degrades to normal pages and reports it via `fell_back()` / `memory_type() == Cpu` / `effective_page_size()`.
9. **Vulkan Video (`vulkan-video`) is a scaffold**: context/session/DPB/DMA-BUF memory are real, but `VulkanH264Decoder::decode_frame` does NOT submit hardware decode commands — it returns a `GpuFrame` over uninitialised memory; no encode; H.265/AV1 absent. Tracked in #3, which also records that Mesa ANV only exposes Vulkan Video on Gen12+ (so a Gen9.5 iGPU cannot even smoke-test it; RADV is the target). The empty `gpu` feature flag and the never-referenced `gpu-allocator` dependency have been removed.
10. **Every example file has an `[[example]]` entry in Cargo.toml**, feature-gated ones carrying `required-features`. Add one when you add an example: without it, cargo builds the file under default features and there is nowhere to hang the gate — which is how 42/43/44 ended up hand-rolling `#[cfg(not(feature))]` stub `main`s that printed a message instead of failing.
11. **benches/memory_pool.rs and benches/throughput.rs are stubs** (pending rewrite after a memory-API refactor); only `colorspace` is a real bench.
12. **`AlignmentStrategy::Interpolate` needs `B: Lerp`** — it resamples the *right* stream onto the left's timestamps, and only `TemporalJoin::try_emit_interpolated` honours it. Plain `try_emit` returns `None` for that variant (it used to silently run `Nearest(10ms)`, discarding the caller's `Duration`).
13. Types that do NOT exist (stale docs may mention them): `CpuArena`, `HeapSegment`, `MemoryPool`, `SharedMemorySegment`, `PipelineExecutor`, `ElementSandbox`, `Affinity`, `parallax-launch`/`parallax-inspect`/`parallax-top` binaries.

## Code Style

- rustfmt defaults; `#![warn(missing_docs)]` is enforced — every public item needs a doc comment.
- thiserror for errors; tracing for logs; keep element `process()` sync (async only for real I/O).
- Derive rkyv traits for IPC-crossing types.
- Tests colocated per module + integration tests in `tests/`; run `just check` before committing.

## Documentation Map

- `docs/README.md` — index
- `docs/getting-started.md`, `architecture.md`, `pipeline.md`, `scheduling.md`, `memory.md`, `elements.md`, `formats.md`, `plugins.md`, `api.md`, `security.md` — user guides
- `docs/design.md` — design rationale + competitive landscape
- `docs/research/` — historical research/design notes (may describe superseded designs)
- `plans/` — active implementation plans (`plans/README.md` for status)
