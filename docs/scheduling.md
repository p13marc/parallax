# Scheduling & Execution

How the unified executor runs pipelines: automatic strategy selection, hybrid async + real-time scheduling, drivers, bridges, and clocks.

## The unified executor

One executor (`parallax::pipeline::Executor`) runs every pipeline. Each element is assigned one of **two** strategies:

| `ElementStrategy` | Runs as | Best for |
|-------------------|---------|----------|
| `Async` | Tokio task | I/O-bound elements (network, file), anything without RT requirements |
| `RealTime` | dedicated RT data thread | RT-safe, low-latency processing (audio mixing, decode loops) |

There is no per-element thread spawning beyond this, and no process isolation (an isolated-process strategy was prototyped and removed — `ExecutionHints::trust_level` is currently informational).

### Automatic strategy (`auto_strategy: true`, the default)

Elements describe themselves with `ExecutionHints`:

```rust
pub struct ExecutionHints {
    pub rt_safe: bool,                 // no allocation/blocking in the hot path
    pub trust_level: TrustLevel,       // Trusted | SemiTrusted | Untrusted (informational)
    pub crash_safe: bool,
    pub uses_native_code: bool,        // FFI / unsafe (informational)
    pub processing: ProcessingHint,    // Unknown | CpuBound | IoBound | MemoryBound
    pub latency: LatencyHint,          // UltraLow | Low | Normal | Relaxed
    pub memory: MemoryHint,            // Normal | Low | High | Streaming
}
```

Profiles: `ExecutionHints::rt_safe()`, `io_bound()`, `cpu_intensive()`, `low_latency()`, `trusted()`, `native()`, `untrusted()`.

The decision rule is deliberately simple:

1. `rt_safe && latency ∈ {UltraLow, Low}` → **RealTime**
2. `processing == IoBound` → **Async**
3. everything else → **Async**

If any element lands on RealTime, the pipeline runs in hybrid mode; otherwise fully async.

### Manual configuration

```rust
use parallax::pipeline::{Executor, ExecutorConfig, SchedulingMode, RtConfig};

let config = ExecutorConfig {
    auto_strategy: false,
    scheduling: SchedulingMode::Hybrid,   // Async | Hybrid | RealTime
    channel_capacity: 16,                 // inter-element channel depth
    rt: RtConfig {
        quantum: 256,                     // samples/frames per RT cycle
        rt_priority: Some(50),            // SCHED_FIFO priority (needs CAP_SYS_NICE)
        data_threads: 1,                  // 0 = off, 1 = single, -1 = per-core
        bridge_capacity: 16,              // async↔RT ring size
        ..Default::default()
    },
    driver: None,                         // Some(DriverConfig) for explicit pacing
    shed_fatal_after: None,               // None = shed forever (right for live media)
};

let executor = Executor::with_config(config);

// start() is SYNCHRONOUS and returns a handle; run() = start() + wait()
let handle = executor.start(&mut pipeline)?;
handle.wait().await?;
// or: executor.run(&mut pipeline).await?;
```

Presets: `ExecutorConfig::auto()`, `async_only()`, `hybrid()`, `low_latency_audio()` (quantum 64, priority 50), `video(fps)`.

`channel_capacity` does double duty: besides the channel depth, the executor uses
it to size each element's **output arena**, so an element cannot run out of slots
just because the channel it feeds is full. Raising it therefore costs memory in
two places — see [memory.md § Output arenas](memory.md#output-arenas), which also
explains why exhaustion sheds a buffer rather than failing the pipeline, and what
`shed_fatal_after` is for.

`SchedulingMode` semantics:

- `Async` — everything in Tokio (default).
- `Hybrid` — nodes that are `rt_safe` **and** low-latency go to RT threads; the rest stay async.
- `RealTime` — every `rt_safe` node goes to an RT thread (unlike `Hybrid`, no low-latency hint is also required). If *no* node qualifies, the executor logs a warning and falls back to fully-async execution rather than erroring. Setting this via `ExecutorConfig::with_scheduling` is sufficient: the executor derives `RtConfig::mode` from it, so the two cannot fall out of sync.

## Hybrid pipeline anatomy

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tokio runtime                            │
│   ┌──────────┐    ┌──────────┐                                  │
│   │ TcpSrc   │    │ FileSrc  │        (async, channel-driven)   │
│   └────┬─────┘    └────┬─────┘                                  │
│        ▼               ▼                                        │
│  ═══ AsyncRtBridge ═══════════  lock-free SPSC ring + eventfd   │
└────────┼────────────────────────────────────────────────────────┘
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  RT data thread (optional SCHED_FIFO)                           │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐                    │
│   │ Decoder  │──▶│  Mixer   │──▶│AudioSink │  (driver-paced)    │
│   └──────────┘   └──────────┘   └──────────┘                    │
│   TimerDriver ticks every quantum; activation records order     │
│   node execution topologically within each cycle                │
└─────────────────────────────────────────────────────────────────┘
```

The pieces (all in `src/pipeline/`):

- **`RtScheduler::partition_graph`** classifies nodes per the scheduling mode and finds *boundary edges* where async and RT nodes connect (`BoundaryDirection::{AsyncToRt, RtToAsync}`).
- **`AsyncRtBridge`** (`rt_bridge.rs`) — a cache-line-padded lock-free SPSC ring buffer carrying `Buffer`s across the boundary. The async side uses `push_async`/eventfd-based waits; the RT side uses non-blocking `try_push`/`try_pop`. EOS propagates through `signal_eos`.
- **`ActivationRecord`** (PipeWire-style) — per-node atomics (`required`/`pending`/`status` + eventfd trigger). Each RT cycle resets pending counts; a node runs when its dependencies have decremented it to ready. Statuses: `Idle`, `NeedData`, `HaveData`, `Processing`.
- **Drivers** (`driver.rs`) pace the RT side:
  - `TimerDriver` — fixed-period ticks; presets `DriverConfig::low_latency_audio()` (quantum 64 @ 48 kHz), `standard_audio()`, `video(fps)`, `custom(period, quantum)`.
  - `ManualDriver` — you call `trigger()` (tests, externally-clocked sources).
  - Hardware-driven pacing arrives via clock providers (see below).
- **`DataThreadHandle`** — join/stop handle for RT threads.

### RT-safety contract

An element declaring `rt_safe: true` promises its hot path performs **no heap allocation, no locking, no blocking syscalls**. Use pooled buffers (`ctx.acquire_buffer()` from a pre-sized `FixedBufferPool`) and pre-computed state. The `Gain` element is the reference example of an RT-safe transform.

`SCHED_FIFO` priorities require `CAP_SYS_NICE` (or rtkit); without it, RT threads run as normal threads and a warning is logged.

## Clocks

Timing lives in `src/clock.rs`:

- **`ClockTime`** — nanoseconds in a transparent `u64`, with `ZERO` and a `NONE` sentinel (`u64::MAX`); arithmetic saturates and propagates `NONE`.
- **`Clock`** trait — `now()`, plus `flags()` (`CAN_BE_MASTER`, `HARDWARE`, `NETWORK`, …), `resolution()`, `name()`. `SystemClock` (monotonic) is the default.
- **`PipelineClock`** — pairs a clock with a `base_time` set at start; `running_time() = now − base_time`. Async helpers `wait_until(t)` / `wait_for(d)` poll in ≤10 ms steps so clock adjustments are honored.
- **`ClockProvider`** — elements that own a better clock (audio sinks, network sync) implement `provide_clock()` + `clock_priority()`. Priority bands: 0–99 software, 100–199 hardware audio, 200–299 network (NTP), 300+ precision (PTP).

### Automatic clock selection

Before starting, the executor calls `pipeline.select_clock()`: every node's `as_clock_provider()` is consulted and the **highest-priority clock wins** (e.g. `AlsaSink` provides its hardware clock at priority 100, beating the system clock). The bus posts `NewClock`. Override manually with `pipeline.set_clock(clock)` or `pipeline.use_clock_from(&provider)`.

Sources read time through their context: `ctx.clock()`, `ctx.running_time()`, `ctx.base_time()` — use these to pace or timestamp production.

### Timestamps through the pipeline

Buffers carry PTS/DTS in `Metadata`. Device sources extract hardware timestamps where available (V4L2 buffer timestamps, PipeWire `spa_meta_header.pts`, ALSA hardware tstamps with sample-count fallback). **Transforms must clone/propagate metadata** or timing information is lost:

```rust
fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
    let metadata = buffer.metadata().clone(); // keep PTS/flags/custom data
    // ... transform ...
    Ok(Some(Buffer::new(new_handle, metadata)))
}
```

Debugging: the `TimestampDebug` element logs/collects PTS statistics (missing, backwards, interval jitter) — see `examples/48_clock_provider.rs` and `examples/50_hybrid_pipeline.rs`.
