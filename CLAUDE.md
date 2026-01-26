# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Parallax** is a Rust-native streaming pipeline engine designed to compete with GStreamer, offering security-first process isolation, zero-copy memory management, and modern GPU integration.

### Core Principles

1. **Security-First**: Inter-process isolation by default, sandboxed elements with seccomp/namespaces
2. **Shared Memory First**: All CPU buffers are memfd-backed (zero overhead, always IPC-ready)
3. **Progressive Typing**: Dynamic pipelines (runtime flexibility) + Typed pipelines (compile-time safety)
4. **Zero-Copy by Design**: Arena allocation, fd passing, rkyv serialization
5. **Linux-Only**: Leverages memfd_create, SCM_RIGHTS, seccomp, namespaces, cgroups
6. **Sync Processing, Async Orchestration**: Element processing is sync; pipeline orchestration is async (Tokio)

### Key Design Decisions

| Aspect | Choice |
|--------|--------|
| Async runtime | Tokio (shared within process) |
| Channels | Kanal (MPMC, sync+async) |
| Parser | winnow |
| Graph structure | daggy (enforces DAG) |
| Serialization | rkyv (zero-copy) |
| Error handling | thiserror (library) |
| Metrics | metrics-rs + tracing |
| Linux APIs | rustix |
| Plugin ABI | stabby (stable Rust ABI) |
| GPU | Vulkan Video + rust-gpu |
| Codecs | Vulkan Video (primary) + rav1d/rav1e (fallback) |

### Execution Modes

```rust
pub enum ExecutionMode {
    /// All elements as Tokio tasks in ONE runtime (fastest, no isolation)
    InProcess,
    
    /// Each element in separate sandboxed process (max isolation)
    Isolated { sandbox: ElementSandbox },
    
    /// Group elements to minimize processes while isolating untrusted code
    Grouped {
        isolated_patterns: Vec<String>,
        sandbox: ElementSandbox,
        groups: Option<HashMap<String, GroupId>>,
    },
}
```

| Mode | 20 elements | Tokio Runtimes | Processes |
|------|-------------|----------------|-----------|
| InProcess | All trusted | 1 | 1 |
| Isolated | All untrusted | 21 | 21 |
| Grouped | 2 codecs untrusted | 2-3 | 2-3 |

### Hybrid Scheduling (PipeWire-inspired)

Parallax uses a hybrid scheduling model inspired by PipeWire that combines:
- **Tokio async tasks** for I/O-bound elements (network, file I/O)
- **Dedicated RT threads** for CPU-bound, real-time-safe elements (audio/video processing)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tokio Runtime                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  TcpSrc      │  │  FileSrc     │  │  HttpSrc     │          │
│  │  (async I/O) │  │  (async I/O) │  │  (async I/O) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              AsyncRtBridge (lock-free ring buffer)      │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RT Data Thread(s)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Decoder     │──│  Mixer       │──│  AudioSink   │          │
│  │  (RT-safe)   │  │  (RT-safe)   │  │  (RT-safe)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
│  Driver-based scheduling, deterministic latency                │
└─────────────────────────────────────────────────────────────────┘
```

**Key concepts:**

- **Element Affinity**: Each element declares its scheduling affinity (`Async`, `RealTime`, or `Auto`)
- **RT-Safety**: Elements declare `is_rt_safe()` - no allocations, no blocking in hot path
- **Graph Partitioning**: The scheduler automatically partitions the graph and inserts bridges
- **Driver-based Scheduling**: A driver node (timer or hardware) initiates each processing cycle

```rust
// Scheduling modes
pub enum SchedulingMode {
    Async,    // All in Tokio (default)
    Hybrid,   // RT-safe nodes in RT threads, rest in Tokio
    RealTime, // All RT-safe nodes in RT threads
}

// Configure hybrid execution
let config = RtConfig {
    mode: SchedulingMode::Hybrid,
    quantum: 256,        // samples per cycle (audio)
    rt_priority: Some(50), // SCHED_FIFO priority (requires CAP_SYS_NICE)
    data_threads: 1,
    bridge_capacity: 16,
};

let executor = HybridExecutor::new(config);
executor.run(&mut pipeline).await?;
```

### Pipeline State Model (PipeWire-inspired)

Parallax uses a 3-state model inspired by PipeWire:

```
Suspended <──> Idle <──> Running
    │                        │
    └────── Error ◄──────────┘
```

| State | Description | Resources |
|-------|-------------|-----------|
| **Suspended** | Minimal memory footprint | Deallocated |
| **Idle** | Ready to process (paused) | Allocated |
| **Running** | Actively processing data | Allocated |
| **Error** | Unrecoverable error | Varies |

**State transitions:**
- `prepare()`: Suspended → Idle (validate, negotiate caps, allocate)
- `activate()`: Idle → Running (start processing)
- `pause()`: Running → Idle (stop processing, keep resources)
- `suspend()`: Idle → Suspended (release resources)

The key insight from PipeWire: "paused" and "stopped" are the same state (Idle) - the difference is just intent.

## Build Commands

```bash
# Using just (recommended)
just test          # Run tests with nextest
just lint          # Run clippy
just check         # Format + lint + test
just bench         # Run benchmarks
just watch         # Auto-run tests on changes

# Or directly with cargo
cargo build
cargo nextest run
cargo clippy -- -D warnings
```

## Architecture

### Security Model

```
┌─────────────────────────────────────────────────────────────────┐
│                      SUPERVISOR PROCESS                         │
│  • Spawns element processes    • Owns shared memory allocation  │
│  • Routes control messages     • Handles crash recovery         │
└─────────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Element │    │ Element │    │ Element │    │ Element │
   │  (src)  │───▶│ (codec) │───▶│(filter) │───▶│ (sink)  │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘
   seccomp        seccomp        seccomp        seccomp
```

**Key principle**: Buffers are shared; authority is not.

### Current Implementation

```
parallax/
├── src/
│   ├── lib.rs              # Public API exports
│   ├── error.rs            # Error types (thiserror)
│   │
│   ├── memory/             # Memory management
│   │   ├── segment.rs      # MemorySegment trait, MemoryType
│   │   ├── heap.rs         # HeapSegment (simple heap allocation)
│   │   ├── shared.rs       # SharedMemorySegment (memfd_create)
│   │   ├── cpu.rs          # CpuArena (arena allocator for IPC)
│   │   ├── pool.rs         # MemoryPool, LoanedSlot
│   │   ├── bitmap.rs       # AtomicBitmap (lock-free slot tracking)
│   │   └── ipc.rs          # send_fds/recv_fds (SCM_RIGHTS)
│   │
│   ├── buffer.rs           # Buffer, MemoryHandle
│   ├── metadata.rs         # Metadata, BufferFlags
│   │
│   ├── element/            # Element system
│   │   ├── traits.rs       # Source, Sink, Element, AsyncSource, AsyncSink
│   │   ├── pad.rs          # Pad, PadDirection, PadTemplate
│   │   └── context.rs      # ElementContext
│   │
│   ├── pipeline/           # Pipeline execution
│   │   ├── graph.rs        # Pipeline DAG (daggy-based)
│   │   ├── executor.rs     # PipelineExecutor (Tokio tasks + Kanal channels)
│   │   ├── hybrid_executor.rs # HybridExecutor (async + RT threads)
│   │   ├── rt_scheduler.rs # RT scheduler (graph partitioning, activation records)
│   │   ├── rt_bridge.rs    # AsyncRtBridge (lock-free SPSC ring buffer + eventfd)
│   │   ├── driver.rs       # TimerDriver, ManualDriver (PipeWire-style drivers)
│   │   ├── parser.rs       # Pipeline string parser (winnow)
│   │   └── factory.rs      # ElementFactory + PluginRegistry
│   │
│   ├── execution/          # Process isolation
│   │   ├── mode.rs         # ExecutionMode (InProcess, Isolated, Grouped)
│   │   ├── isolated_executor.rs  # Transparent IPC injection
│   │   ├── supervisor.rs   # Process supervision
│   │   └── protocol.rs     # Control message protocol
│   │
│   ├── elements/           # Built-in elements (organized by category)
│   │   ├── network/        # TCP, UDP, Unix, multicast, HTTP, WebSocket, Zenoh
│   │   ├── rtp/            # RTP, RTCP, codecs, jitter buffer, RTSP
│   │   ├── io/             # FileSrc/Sink, FdSrc/Sink, ConsoleSink
│   │   ├── testing/        # TestSrc, VideoTestSrc, DataSrc, Null
│   │   ├── flow/           # Queue, Tee, Funnel, Selector, Concat, Valve
│   │   ├── transform/      # Filter, Map, Batch, buffer/metadata ops
│   │   ├── app/            # AppSrc, AppSink, IcedVideoSink
│   │   ├── ipc/            # IpcSrc, IpcSink, MemorySrc/Sink
│   │   ├── timing/         # Delay, Timeout, RateLimiter
│   │   ├── demux/          # StreamIdDemux, TsDemux
│   │   └── util/           # PassThrough, Identity
│   │
│   ├── typed/              # Type-safe pipeline API
│   │   ├── pipeline.rs     # PipelineWithSource, PipelineWithTransforms
│   │   ├── operators.rs    # map, filter, take, skip, collect, etc.
│   │   └── multi_source.rs # merge, zip, join, temporal_join
│   │
│   └── plugin/             # Plugin system
│       ├── registry.rs     # PluginRegistry
│       ├── loader.rs       # Dynamic loading
│       └── descriptor.rs   # Plugin metadata (C-compatible ABI)
│
├── examples/               # One concept per file, numbered
│   ├── 01_hello_pipeline.rs      # Simplest pipeline
│   ├── 02_counting_source.rs     # Multiple buffers
│   ├── 03_transform_element.rs   # Transform element
│   ├── 04_tee_fanout.rs          # 1-to-N fanout
│   ├── 05_funnel_merge.rs        # N-to-1 merge
│   ├── 06_typed_pipeline.rs      # Type-safe pipelines
│   ├── 07_appsrc_appsink.rs      # Application integration
│   ├── 08_queue_backpressure.rs  # Backpressure
│   ├── 09_valve_control.rs       # Flow control
│   ├── 10_file_io.rs             # File read/write
│   ├── 11_isolate_in_process.rs  # Default execution
│   ├── 12_isolate_by_pattern.rs  # Selective isolation
│   ├── 13_isolate_all.rs         # Full isolation
│   ├── 14_ipc_manual.rs          # Manual IPC
│   ├── 15_video_testsrc.rs       # Video test patterns
│   └── 16_video_display.rs       # GUI display (iced-sink)
│
├── docs/
│   ├── FINAL_DESIGN_PARALLAX.md  # Complete design document
│   ├── PLAN_CAPS_NEGOTIATION.md  # Caps negotiation design
│   └── getting-started.md        # Quick start guide
```

### Key Types

```rust
// Element affinity (for hybrid scheduling)
pub enum Affinity {
    Async,     // Always run in Tokio
    RealTime,  // Always run in RT thread
    Auto,      // Let scheduler decide based on is_rt_safe()
}

// Element traits (sync)
pub trait Source: Send {
    fn produce(&mut self) -> Result<Option<Buffer>>;
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn is_rt_safe(&self) -> bool { false }
}

pub trait Sink: Send {
    fn consume(&mut self, buffer: Buffer) -> Result<()>;
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn is_rt_safe(&self) -> bool { false }
}

pub trait Element: Send {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>>;
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn is_rt_safe(&self) -> bool { false }
}

// Element traits (async - for I/O bound operations)
pub trait AsyncSource: Send {
    fn produce(&mut self) -> impl Future<Output = Result<Option<Buffer>>> + Send;
    fn affinity(&self) -> Affinity { Affinity::Async }  // Default to async
    fn is_rt_safe(&self) -> bool { false }
}

pub trait AsyncSink: Send {
    fn consume(&mut self, buffer: Buffer) -> impl Future<Output = Result<()>> + Send;
    fn affinity(&self) -> Affinity { Affinity::Async }
    fn is_rt_safe(&self) -> bool { false }
}

// Pipeline usage
let mut pipeline = Pipeline::parse("videotestsrc ! h264enc ! filesink location=out.h264")?;
pipeline.run().await?;
```

## Implementation Roadmap

See `docs/FINAL_DESIGN_PARALLAX.md` for full details.

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Memory Foundation (CpuSegment, CpuArena) | ✅ Complete |
| 2 | Caps Negotiation (global constraint solving) | ✅ Complete |
| 3 | Cross-Process IPC (IpcSrc/IpcSink) | ✅ Complete |
| 4 | GPU Integration (Vulkan Video) | Planned |
| 5 | Pure Rust Codecs (rav1d/rav1e) | Planned |
| 6 | Process Isolation (transparent auto-IPC) | ✅ Complete |
| 7 | Plugin System (C-compatible ABI) | ✅ Complete |
| 8 | Distribution (Zenoh) | ✅ Complete |
| 9 | Hybrid Scheduling (PipeWire-inspired) | ✅ Complete |

### Transparent Process Isolation

Users write normal pipelines - isolation is automatic:

```rust
// Write a normal pipeline
let pipeline = Pipeline::parse("filesrc ! h264dec ! displaysink")?;

// Run with selective isolation (decoders in isolated processes)
pipeline.run_isolating(vec!["*dec*"]).await?;

// Or run fully isolated (each element in its own process)
pipeline.run_isolated().await?;

// Or run in-process (default, fastest)
pipeline.run().await?;
```

The executor automatically injects IPC boundaries where needed.

## Code Style Guidelines

- Use `rustfmt` defaults
- Prefer `thiserror` for error types
- Use `tracing` for logging/instrumentation
- Derive `rkyv::Archive`, `rkyv::Serialize`, `rkyv::Deserialize` for IPC types
- Keep element `process()` sync; async only for I/O
- Document public APIs with examples
- Write tests for each module

## Testing

```bash
just test              # Run all tests with nextest
just test-one NAME     # Run specific test
just test-verbose      # Run with output capture disabled
just watch             # Auto-run on changes
```

## Pipeline Deployment Modes

### Single Binary (gst-launch equivalent)
```bash
parallax-launch "videotestsrc ! h264enc ! filesink location=out.h264"
```

### Multi-Binary (federated pipelines)
```rust
// Binary A
Pipeline::parse("v4l2src ! ipc_sink path=/run/parallax/camera")?;

// Binary B
Pipeline::parse("ipc_src path=/run/parallax/camera ! encoder ! zenoh_pub key=video")?;
```

### Cross-Machine (Zenoh)
```rust
// Machine A
Pipeline::parse("camera ! zenoh_pub key=factory/camera/1")?;

// Machine B
Pipeline::parse("zenoh_sub key=factory/camera/1 ! display")?;
```

## Performance Notes

- Buffer cloning is O(1) (Arc increment)
- Pool slot acquire/release is O(1) amortized (atomic bitmap)
- All CPU memory is memfd-backed (zero overhead vs malloc, always IPC-ready)
- Cross-process is true zero-copy (same physical pages via mmap)
- Arena allocation: 1 fd per pool, not per buffer (avoids fd limits)

## Documentation

- `docs/FINAL_DESIGN_PARALLAX.md` - Complete design document and competitive analysis
- `docs/PLAN_CAPS_NEGOTIATION.md` - Detailed caps negotiation design
- `FINAL_PLAN.md` - Original implementation plan (partially outdated)
