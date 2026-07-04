# Parallax Architecture

This document describes how the pieces of Parallax fit together. Detailed guides for each subsystem are linked throughout.

## Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                            Application                             │
│      Pipeline::parse("...") / programmatic API / typed API         │
└──────────────────────────────┬─────────────────────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│  Pipeline (daggy DAG)                                              │
│    nodes = elements (type-erased DynAsyncElement)                  │
│    edges = links (negotiated format + memory type)                 │
│    state machine: Suspended ⇄ Idle ⇄ Running (+ Error)             │
│    caps negotiation · converter policy · clock selection · bus     │
└──────────────────────────────┬─────────────────────────────────────┘
                               │  Executor (auto strategy)
        ┌──────────────────────┴──────────────────────┐
        ▼                                             ▼
┌───────────────────────┐    AsyncRtBridge    ┌───────────────────────┐
│  Tokio runtime        │◄══ lock-free SPSC ═►│  RT data thread(s)    │
│  I/O-bound elements   │    ring + eventfd   │  rt_safe + low-latency│
│  (network, file, …)   │                     │  elements, driver-    │
│                       │                     │  paced cycles         │
└───────────────────────┘                     └───────────────────────┘
                               │
┌──────────────────────────────▼─────────────────────────────────────┐
│  Shared-memory foundation (all CPU buffers are memfd-backed)       │
│  SharedArena · FixedBufferPool · DmaBufSegment · MappedFileSegment │
│  cross-process refcounts + lock-free release queue in shared mem   │
└────────────────────────────────────────────────────────────────────┘
```

## Layers

### 1. Memory (`src/memory/`, `src/buffer.rs`, `src/metadata.rs`)

Everything above sits on the shared-memory foundation:

- **`SharedArena`** — one memfd per pool, `MAP_SHARED`. Slot refcounts and a lock-free MPSC release queue live *inside* the mapping, so clone/drop work identically from any process that maps the arena. Layout: `ArenaHeader | ReleaseQueue | SlotHeader[N] | SlotData[N]`.
- **`Buffer<T = ()>`** — a `MemoryHandle` (arena slot + offset + len) plus `Metadata`. Cloning is an atomic increment; `slice()` produces zero-copy sub-buffers. `Buffer<()>` is the dynamic form used throughout the pipeline; `Buffer<T>` carries a compile-time type tag.
- **`Metadata`** — PTS/DTS/duration (`ClockTime`), sequence, stream id, `BufferFlags`, optional RTP header info and negotiated format, plus a typed extensible map for custom data (KLV, SEI, captions, app data).
- **`FixedBufferPool`** — pipeline-level pool on top of `SharedArena` with blocking `acquire()` (backpressure) and statistics.
- **DMA-BUF** (`DmaBufSegment`/`DmaBufBuffer`) — wraps device/GPU fds for the zero-copy capture path.

Details: [memory.md](memory.md).

### 2. Elements (`src/element/`, `src/elements/`)

An element is a `Source`, `Sink`, `Element`/`Transform`, `Demuxer`, or `Muxer` implementation (or their async variants). Author traits are wrapped by adapters into a single type-erased runtime trait (`DynAsyncElement`, generated with trait-variant + dynosaur) that the executor drives.

- Sources produce into pool-provided buffers via `ProduceContext` (`ProduceResult::Produced(n)`) or hand over their own buffers (`OwnBuffer`, `OwnDmaBuf`).
- Elements declare **`ExecutionHints`** — `rt_safe`, `processing` (CPU/IO/memory-bound), `latency`, `memory` — which the executor uses to pick a strategy.
- Elements may also expose caps (`output_media_caps`/`input_media_caps`), seeking (`SeekableSource`), flow policies for live sources, and clocks (`as_clock_provider`).
- A newer "simple" API (`SimpleSource`/`SimpleSink`/`SimpleTransform` + `Src`/`Snk`/`Xfm` wrappers) removes adapter boilerplate for straightforward elements.

The built-in library (~100 elements across io/network/rtp/flow/transform/timing/mux/demux/codec/device/streaming categories) is cataloged in [elements.md](elements.md).

### 3. Pipeline graph (`src/pipeline/graph.rs`)

`Pipeline` owns a daggy DAG (cycles are rejected at link time). Each node carries the type-erased element, its unique name, cached execution hints, and (optionally) the clock it can provide. Each edge records the negotiated format and memory type once `prepare()` has run.

The pipeline also owns:

- the **state machine** — `Suspended ⇄ Idle ⇄ Running` with `Error`; `prepare()` runs validation, caps negotiation, and allocation,
- the **bus** (element→application messages) and the typed event channel,
- the **probe registry** (buffer/event interception) and **tracer registry**,
- **clock selection** — the highest-priority `ClockProvider` among elements wins (e.g. an ALSA sink's hardware clock),
- introspection — `to_dot()`, `to_json()`, `describe()`, `stats_snapshot()`.

Details: [pipeline.md](pipeline.md).

### 4. Caps negotiation (`src/format.rs`, `src/negotiation/`)

Elements declare preference-ordered lists of `FormatMemoryCap` (a format constraint coupled with acceptable memory types). During `prepare()`, the solver walks each link and picks the first intersection of source and sink capabilities, fixating ranges/lists into concrete formats and choosing a memory type (e.g. DMA-BUF when both ends support it, CPU otherwise).

If no intersection exists, behavior follows the `ConverterPolicy`: `Deny` (default) fails with a message listing every attempted combination; `Warn`/`Allow` consult the `ConverterRegistry` and auto-insert a converter element (single hop).

Negotiation is per-link (first-match, preference-ordered) — not a global constraint solve. Details: [formats.md](formats.md).

### 5. Executor (`src/pipeline/unified_executor.rs`, `rt_scheduler.rs`, `rt_bridge.rs`, `driver.rs`)

One executor, two element strategies:

| Strategy | Runs as | Chosen when (auto) |
|----------|---------|--------------------|
| `Async` | Tokio task | I/O-bound, or anything not RT-eligible (default) |
| `RealTime` | dedicated RT thread | `rt_safe` **and** latency `UltraLow`/`Low` |

With `auto_strategy` (default), the executor analyzes hints, partitions the graph, inserts `AsyncRtBridge`s (lock-free SPSC ring + eventfd) at async↔RT boundaries, and runs RT cycles paced by a `TimerDriver` (or a hardware/manual driver) using PipeWire-style activation records. RT threads can request `SCHED_FIFO` priority.

All execution is in-process. (A process-isolation strategy for untrusted elements was prototyped and removed; `trust_level` hints are currently informational.)

Details: [scheduling.md](scheduling.md).

### 6. Links across boundaries (`src/link/`)

| Link | Transport | Copy semantics |
|------|-----------|----------------|
| `LocalLink` | kanal channel (in-process) | move — refcount only |
| `IpcPublisher`/`IpcSubscriber` | Unix socket + shared arena | zero-copy — only a small ref crosses the socket; the arena fd is passed once via SCM_RIGHTS |
| `NetworkSender`/`NetworkReceiver` | TCP | serialized with rkyv (framed: `PRLX` magic, version, CRC32) |

The `IpcSrc`/`IpcSink` elements wrap the IPC link so multi-process pipelines compose like everything else.

### 7. Typed layer (`src/typed/`, `src/temporal/`)

An independent, compile-time-checked pipeline builder: `pipeline(source).then(transform).sink(sink).run()`, with `>>` as `.then` sugar. Transform chains are typed cons-lists (`Chain<T, Tail>`), so type mismatches between stages are compile errors. Multi-source combinators include `zip`, `merge` (two-source interleave), hash `join`, and `temporal_join` — timestamp-aligned joining with configurable tolerance windows (`src/temporal/`), useful for sensor fusion. Bridges (`source_to_dyn` etc.) connect typed stages into dynamic pipelines.

Execution is a synchronous pull loop (`run()`); `run_async()` wraps it in `spawn_blocking`.

### 8. Plugins (`src/plugin/`)

Dynamic element loading from `cdylib`s over a versioned `#[repr(C)]` ABI: a plugin exports `parallax_plugin_descriptor` returning a `PluginDescriptor` (ABI version, element list, `extern "C"` constructors). `PluginLoader` (libloading) validates the ABI version and descriptor before use; `PluginRegistry` indexes elements and can extend `Pipeline::parse` names. Details: [plugins.md](plugins.md).

### 9. Observability (`src/observability/`, `src/pipeline/tracer.rs`)

- metrics-rs counters/histograms (`parallax_buffers_*`, `parallax_processing_time_ns`, …) and tracing spans.
- Runtime tracers (latency, framerate, drops) activated programmatically or via `PARALLAX_TRACERS`.
- DOT graph export (`pipeline.to_dot()`), auto-dumped on state transitions when `PARALLAX_DOT_DIR` is set.

## Data flow walkthrough

What happens for one buffer in `filesrc ! passthrough ! filesink`:

1. **prepare()** — graph validated, caps negotiated per link, buffer pool allocated, clock selected, converters inserted (policy permitting). State → `Idle`.
2. **activate()/run()** — executor analyzes hints (all three elements here are async), spawns one Tokio task per element connected by channels. State → `Running`.
3. The source task calls `FileSrc::produce(ctx)`; the file chunk is read directly into an arena slot; `Produced(n)` finalizes a `Buffer` (slot ref + metadata).
4. The buffer moves through the channel (refcount move, no copy). Probes registered on the pads run before forwarding; tracers observe timestamps.
5. `passthrough` returns the same buffer; `FileSink::consume(ctx)` writes it out.
6. On EOS the source returns `ProduceResult::Eos`; downstream elements get `flush()` calls until drained; the bus posts `Eos`; `run()` resolves.

For a hybrid pipeline, step 2 additionally partitions the graph, spawns RT data threads, and connects domains with bridges; the RT side is driven in fixed quanta by the driver instead of being channel-driven.

## Design lineage

Parallax borrows deliberately:

- **From GStreamer**: element/pad/caps vocabulary, the bus, probes, typefind, seeking/segment semantics, converter auto-insertion (as an opt-in), tracer subsystem.
- **From PipeWire**: the 3-state model (paused = idle), quantum/driver-based RT scheduling, activation records, fd-based buffer sharing — extended with refcounts stored directly in shared memory rather than coordinated per-process.

The rationale and competitive landscape are in [design.md](design.md).
