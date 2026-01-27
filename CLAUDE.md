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
| GPU | Vulkan Video + rust-gpu (planned) |
| Video Codecs | rav1e (AV1 encode), dav1d (AV1 decode) |
| Audio Codecs | Symphonia (pure Rust: FLAC, MP3, AAC, Vorbis) |
| Image Codecs | zune-jpeg, png crate (pure Rust) |
| Container Formats | mp4 crate (pure Rust: MP4 demux/mux) |

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

### Unified Executor with Automatic Strategy

Parallax uses a unified `Executor` that automatically determines the optimal execution strategy for each element based on **ExecutionHints**. No developer insight required - just run your pipeline and the executor figures out the best approach.

```rust
// Simply run your pipeline - the executor auto-negotiates strategy
let mut pipeline = Pipeline::parse("filesrc ! decoder ! videosink")?;
pipeline.run().await?;  // Automatic: filesrc=async, decoder=isolated, videosink=async
```

#### Automatic Strategy Detection

Each element declares `ExecutionHints` describing its characteristics:

```rust
pub struct ExecutionHints {
    pub trust_level: TrustLevel,      // Trusted, SemiTrusted, Untrusted
    pub processing: ProcessingHint,    // CpuBound, IoBound, MemoryBound, Unknown
    pub latency: LatencyHint,          // UltraLow, Low, Normal, Relaxed
    pub crash_safe: bool,              // Can recover from crashes?
    pub uses_native_code: bool,        // FFI, unsafe, external libs?
    pub memory: MemoryHint,            // Normal, Low, High, Streaming
}

// Elements provide hints (defaults are safe)
impl Source for MyDecoder {
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::native()  // Uses FFI -> will be isolated
    }
}
```

The executor analyzes all hints and chooses:

| Element Characteristics | Strategy |
|------------------------|----------|
| Untrusted OR native+unsafe | **Isolated** (separate process) |
| RT affinity + RT-safe | **RealTime** (dedicated RT thread) |
| Low latency + RT-safe | **RealTime** |
| I/O-bound OR async affinity | **Async** (Tokio task) |
| Everything else | **Async** (default) |

#### Manual Override

You can still configure manually if needed:

```rust
let config = ExecutorConfig {
    auto_strategy: false,  // Disable auto-detection
    scheduling: SchedulingMode::Hybrid,
    rt: RtConfig {
        quantum: 256,
        rt_priority: Some(50),
        ..Default::default()
    },
    ..Default::default()
};

let executor = Executor::with_config(config);
executor.start(&mut pipeline).await?;
```

### Hybrid Scheduling (PipeWire-inspired)

Under the hood, the unified executor combines:
- **Tokio async tasks** for I/O-bound elements (network, file I/O)
- **Dedicated RT threads** for CPU-bound, real-time-safe elements (audio/video processing)
- **Isolated processes** for untrusted or native code elements

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tokio Runtime                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  TcpSrc      │  │  FileSrc     │  │  HttpSrc     │          │
│  │  (async I/O) │  │  (async I/O) │  │  (async I/O) │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │              AsyncRtBridge (lock-free ring buffer)          ││
│  └─────────────────────────────────────────────────────────────┘│
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
│  Driver-based scheduling, deterministic latency                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                    Isolated Process(es)                         │
│  ┌──────────────┐                                               │
│  │  FFmpeg      │  seccomp sandbox, IPC via shared memory       │
│  │  (untrusted) │                                               │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

**Key concepts:**

- **ExecutionHints**: Each element declares its characteristics for automatic strategy detection
- **Element Affinity**: Elements can declare scheduling affinity (`Async`, `RealTime`, or `Auto`)
- **RT-Safety**: Elements declare `is_rt_safe()` - no allocations, no blocking in hot path
- **Graph Partitioning**: The scheduler automatically partitions the graph and inserts bridges
- **Driver-based Scheduling**: A driver node (timer or hardware) initiates each processing cycle

```rust
// Scheduling modes (used when auto_strategy is false)
pub enum SchedulingMode {
    Async,    // All in Tokio (default)
    Hybrid,   // RT-safe nodes in RT threads, rest in Tokio
    RealTime, // All RT-safe nodes in RT threads
}

// Per-element execution strategy (determined automatically)
pub enum ElementStrategy {
    Async,     // Run as Tokio task
    RealTime,  // Run in RT thread
    Isolated,  // Run in separate process
}
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

### Shared-Memory Reference Counting (SharedArena)

Parallax implements true cross-process reference counting by storing refcounts in shared memory (memfd), not on the heap like `Arc`. This avoids the double-allocation problem and enables zero-copy buffer sharing across processes.

```
SharedArena Memory Layout:
┌─────────────────────────────────────────────────────────────────┐
│ ArenaHeader (64 bytes, cache-aligned)                           │
│   magic, version, slot_count, slot_size, arena_id               │
├─────────────────────────────────────────────────────────────────┤
│ ReleaseQueue (MPSC lock-free queue in shared memory)            │
│   head: AtomicU32     ← Owner reads here (single consumer)      │
│   tail: AtomicU32     ← Any process writes here (multi producer)│
│   slots: [AtomicU32; 1024]  ← Ring buffer of slot indices       │
├─────────────────────────────────────────────────────────────────┤
│ SlotHeader[0..N] (8 bytes each)                                 │
│   refcount: AtomicU32  ← Works across processes!                │
│   state: AtomicU32     ← Free or Allocated                      │
├─────────────────────────────────────────────────────────────────┤
│ SlotData[0..N] (user data)                                      │
└─────────────────────────────────────────────────────────────────┘
```

**Key types:**
- `SharedArena` - Arena with refcounts in shared memory
- `SharedSlotRef` - Zero-allocation slot reference (no Arc needed)
- `SharedIpcSlotRef` - Serializable IPC reference (rkyv-compatible)
- `SharedArenaCache` - Cache for client processes to map arenas

**Cross-process semantics:**
- **Clone**: Atomic increment in shared memory (works across processes)
- **Drop**: Atomic decrement; if 0, push slot index to release queue
- **Reclaim**: Owner drains queue in O(k) where k = released slots

This improves on PipeWire's approach which uses per-process refcounting with message-based coordination.

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
│   │   ├── cpu.rs          # CpuSegment (unified memfd-backed memory)
│   │   ├── arena.rs        # CpuArena (arena allocator, 1 fd per pool)
│   │   ├── shared_refcount.rs # SharedArena (cross-process refcounting)
│   │   ├── pool.rs         # MemoryPool, LoanedSlot
│   │   ├── bitmap.rs       # AtomicBitmap (lock-free slot tracking)
│   │   └── ipc.rs          # send_fds/recv_fds (SCM_RIGHTS)
│   │
│   ├── buffer.rs           # Buffer, MemoryHandle
│   ├── metadata.rs         # Metadata, BufferFlags
│   │
│   ├── element/            # Element system
│   │   ├── traits.rs       # Source, Sink, Element, AsyncSource, AsyncSink, ExecutionHints
│   │   ├── pad.rs          # Pad, PadDirection, PadTemplate
│   │   └── context.rs      # ElementContext
│   │
│   ├── pipeline/           # Pipeline execution
│   │   ├── graph.rs        # Pipeline DAG (daggy-based)
│   │   ├── executor.rs     # Legacy PipelineExecutor (deprecated)
│   │   ├── unified_executor.rs # Unified Executor (async + RT + isolation)
│   │   ├── hybrid_executor.rs # Legacy HybridExecutor (deprecated)
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
│   │   ├── demux/          # StreamIdDemux, TsDemux, Mp4Demux
│   │   ├── mux/            # Mp4Mux
│   │   ├── codec/          # Media codecs (AV1, audio, image - feature-gated)
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
│   ├── 16_video_display.rs       # GUI display (iced-sink)
│   ├── 17_introspection.rs       # Pipeline introspection and caps
│   ├── 18_demuxer_muxer.rs       # Demuxer and muxer elements
│   ├── 19_auto_execution.rs      # Automatic execution strategy
│   ├── 20_dynamic_state.rs       # Dynamic pipeline state changes
│   └── 24_image_codec.rs         # Image encoding/decoding (PNG)
│
├── docs/
│   ├── FINAL_DESIGN_PARALLAX.md  # Complete design document
│   ├── PLAN_CAPS_NEGOTIATION.md  # Caps negotiation design
│   └── getting-started.md        # Quick start guide
```

### Key Types

```rust
// Execution hints for automatic strategy detection
pub struct ExecutionHints {
    pub trust_level: TrustLevel,      // Trusted, SemiTrusted, Untrusted
    pub processing: ProcessingHint,    // CpuBound, IoBound, MemoryBound, Unknown
    pub latency: LatencyHint,          // UltraLow, Low, Normal, Relaxed
    pub crash_safe: bool,
    pub uses_native_code: bool,
    pub memory: MemoryHint,            // Normal, Low, High, Streaming
}

// Element affinity (for hybrid scheduling)
pub enum Affinity {
    Async,     // Always run in Tokio
    RealTime,  // Always run in RT thread
    Auto,      // Let scheduler decide based on is_rt_safe()
}

// Buffer production context (PipeWire-style zero-allocation)
pub enum ProduceResult {
    Produced(usize),   // Wrote n bytes to provided buffer
    Eos,               // End of stream
    OwnBuffer(Buffer), // Source provides its own buffer (fallback)
    WouldBlock,        // No data available yet
}

pub struct ProduceContext<'a> { /* pre-allocated buffer slot */ }
pub struct ConsumeContext<'a> { /* reference to buffer */ }

// Element traits (sync)
pub trait Source: Send {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult>;
    fn preferred_buffer_size(&self) -> Option<usize> { None }
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn is_rt_safe(&self) -> bool { false }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
}

pub trait Sink: Send {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()>;
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn is_rt_safe(&self) -> bool { false }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
}

pub trait Element: Send {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>>;
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn is_rt_safe(&self) -> bool { false }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
}

// Element traits (async - for I/O bound operations)
pub trait AsyncSource: Send {
    fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> impl Future<Output = Result<ProduceResult>> + Send;
    fn preferred_buffer_size(&self) -> Option<usize> { None }
    fn affinity(&self) -> Affinity { Affinity::Async }  // Default to async
    fn is_rt_safe(&self) -> bool { false }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::io_bound() }
}

pub trait AsyncSink: Send {
    fn consume(&mut self, ctx: &ConsumeContext<'_>) -> impl Future<Output = Result<()>> + Send;
    fn affinity(&self) -> Affinity { Affinity::Async }
    fn is_rt_safe(&self) -> bool { false }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::io_bound() }
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
| 5 | Pure Rust Codecs | ✅ Partial (audio/image complete, video AV1 ready) |
| 6 | Process Isolation (transparent auto-IPC) | ✅ Complete |
| 7 | Plugin System (C-compatible ABI) | ✅ Complete |
| 8 | Distribution (Zenoh) | ✅ Complete |
| 9 | Hybrid Scheduling (PipeWire-inspired) | ✅ Complete |
| 10 | Unified Executor (automatic strategy) | ✅ Complete |

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

## Media Codecs

Parallax includes feature-gated media codecs prioritizing **pure Rust implementations** for security and portability.

### Codec Feature Flags

```toml
[dependencies]
# Video codecs
parallax = { version = "0.1", features = ["av1-encode"] }  # AV1 encoder (rav1e, pure Rust)
parallax = { version = "0.1", features = ["av1-decode"] }  # AV1 decoder (dav1d, C library)

# Audio codecs (all pure Rust via Symphonia)
parallax = { version = "0.1", features = ["audio-codecs"] }  # All: FLAC, MP3, AAC, Vorbis
parallax = { version = "0.1", features = ["audio-flac"] }    # FLAC only
parallax = { version = "0.1", features = ["audio-mp3"] }     # MP3 only
parallax = { version = "0.1", features = ["audio-aac"] }     # AAC only
parallax = { version = "0.1", features = ["audio-vorbis"] }  # Vorbis only

# Image codecs (all pure Rust)
parallax = { version = "0.1", features = ["image-codecs"] }  # All: JPEG, PNG
parallax = { version = "0.1", features = ["image-jpeg"] }    # JPEG decoder (zune-jpeg)
parallax = { version = "0.1", features = ["image-png"] }     # PNG encoder/decoder (png crate)

# Container formats (all pure Rust)
parallax = { version = "0.1", features = ["mp4-demux"] }     # MP4 demuxer/muxer
parallax = { version = "0.1", features = ["mpeg-ts"] }       # MPEG-TS demuxer
```

### Codec Summary

| Type | Codec | Feature | Crate | Pure Rust | Notes |
|------|-------|---------|-------|-----------|-------|
| Video | AV1 encode | `av1-encode` | rav1e | Yes | Install nasm for SIMD optimizations |
| Video | AV1 decode | `av1-decode` | dav1d | No | Requires libdav1d system library |
| Audio | FLAC | `audio-flac` | symphonia | Yes | Lossless audio |
| Audio | MP3 | `audio-mp3` | symphonia | Yes | Common lossy format |
| Audio | AAC | `audio-aac` | symphonia | Yes | Common in video containers |
| Audio | Vorbis | `audio-vorbis` | symphonia | Yes | Open source lossy format |
| Image | JPEG | `image-jpeg` | zune-jpeg | Yes | Decoder only |
| Image | PNG | `image-png` | png | Yes | Encoder and decoder |
| Container | MP4 | `mp4-demux` | mp4 | Yes | Demuxer and muxer |
| Container | MPEG-TS | `mpeg-ts` | mpeg2ts-reader | Yes | Demuxer only |

### Build Dependencies

Most codecs are pure Rust with no external dependencies. Exceptions:

- **av1-encode**: Optionally install `nasm` for x86_64 SIMD optimizations
- **av1-decode**: Requires `libdav1d-devel` (Fedora) / `libdav1d-dev` (Debian)

## Performance Notes

- Buffer cloning is O(1) (Arc increment)
- Pool slot acquire/release is O(1) amortized (atomic bitmap)
- All CPU memory is memfd-backed (zero overhead vs malloc, always IPC-ready)
- Cross-process is true zero-copy (same physical pages via mmap)
- Arena allocation: 1 fd per pool, not per buffer (avoids fd limits)
- SharedArena: Cross-process refcounting with O(1) release via lock-free MPSC queue

## Documentation

- `docs/FINAL_DESIGN_PARALLAX.md` - Complete design document and competitive analysis
- `docs/PLAN_CAPS_NEGOTIATION.md` - Detailed caps negotiation design
- `FINAL_PLAN.md` - Original implementation plan (partially outdated)
