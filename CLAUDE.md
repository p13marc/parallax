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
| Video Codecs | OpenH264 (H.264), rav1e (AV1 encode), dav1d (AV1 decode) |
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

### Pipeline Buffer Pool

Parallax provides a pipeline-level buffer pool for efficient buffer management:

```rust
// Create a pool with 10 buffers of 1MB each
let pool = FixedBufferPool::new(1024 * 1024, 10)?;

// Attach pool to source
let src = SourceAdapter::with_pool(my_source, pool.clone());

// Or set on pipeline for auto-sizing based on caps
pipeline.create_pool_from_caps(10)?;
```

**Key types:**
- `BufferPool` - Trait for buffer pool implementations
- `FixedBufferPool` - Fixed-size pool backed by SharedArena
- `PooledBuffer` - RAII buffer that returns to pool on drop
- `PoolStats` - Statistics (acquisitions, waits, availability)

**Features:**
- **Backpressure**: `acquire()` blocks when pool is exhausted
- **Zero-allocation**: Pre-allocated buffers, no malloc during processing
- **Statistics**: Track acquisitions, wait events, and availability
- **ProduceContext integration**: Sources can use `ctx.acquire_buffer()`

```rust
// In a Source::produce() implementation
fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
    let mut pooled = ctx.acquire_buffer()?;  // Blocks if pool exhausted
    pooled.data_mut()[..n].copy_from_slice(&data);
    pooled.set_len(n);
    Ok(ProduceResult::OwnBuffer(pooled.into_buffer()))
}
```

### Muxer Synchronization (N-to-1)

Parallax provides PTS-based synchronization for N-to-1 muxer elements:

```rust
use parallax::element::muxer::{MuxerSyncState, MuxerSyncConfig, PadInfo, StreamType, SyncMode};

// Create sync state with 40ms output interval (25fps video)
let config = MuxerSyncConfig::new()
    .with_mode(SyncMode::Strict)
    .with_interval_ms(40);

let mut sync = MuxerSyncState::new(config);

// Add input pads
let video_pad = sync.add_pad(PadInfo::new("video", StreamType::Video).required());
let audio_pad = sync.add_pad(PadInfo::new("audio", StreamType::Audio).required());
let data_pad = sync.add_pad(PadInfo::new("klv", StreamType::Data).optional());

// Push buffers from each input
sync.push(video_pad, video_buffer)?;
sync.push(audio_pad, audio_buffer)?;

// Check if ready and collect synchronized output
if sync.ready_to_output() {
    let collected = sync.collect_for_output();
    // Process collected buffers...
    sync.advance();  // Move to next output interval
}
```

**Key types:**
- `MuxerSyncState` - Core synchronization state machine
- `MuxerSyncConfig` - Configuration (mode, interval, live)
- `PadInfo` - Per-pad configuration (name, stream type, required)
- `StreamType` - Video, Audio, Subtitle, Data
- `SyncMode` - Synchronization strategy

**Sync modes:**
| Mode | Behavior |
|------|----------|
| `Strict` | Wait for all required pads to have data |
| `Loose` | Output when primary stream (video) is ready |
| `Timed { interval_ms }` | Fixed interval output |
| `Auto` | Strict for non-live, Loose for live sources |

**Pipeline-ready muxer elements:**
```rust
use parallax::elements::mux::{TsMuxElement, TsMuxConfig, TsMuxTrack, TsMuxStreamType};

// Create MPEG-TS muxer with video and KLV tracks
let config = TsMuxConfig::new()
    .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
    .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());

let mut mux = TsMuxElement::new(config)?;

// Push data via MuxerInput
mux.push(MuxerInput::new(video_pad_id, video_buffer))?;
mux.push(MuxerInput::new(data_pad_id, klv_buffer))?;

// Pull synchronized output
if mux.can_output() {
    if let Some(ts_buffer) = mux.pull()? {
        // Write TS packets...
    }
}
```

See `examples/39_muxer_element.rs` for a complete example.

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
│   │   ├── shared_refcount.rs # SharedArena (cross-process refcounting)
│   │   ├── buffer_pool.rs  # BufferPool trait, FixedBufferPool, PooledBuffer
│   │   ├── bitmap.rs       # AtomicBitmap (lock-free slot tracking)
│   │   ├── ipc.rs          # send_fds/recv_fds (SCM_RIGHTS)
│   │   ├── huge_pages.rs   # Huge page support
│   │   └── mapped_file.rs  # Memory-mapped file support
│   │
│   ├── buffer.rs           # Buffer, MemoryHandle
│   ├── metadata.rs         # Metadata, BufferFlags
│   │
│   ├── element/            # Element system
│   │   ├── traits.rs       # Source, Sink, Element, AsyncSource, AsyncSink, ExecutionHints
│   │   ├── pad.rs          # Pad, PadDirection, PadTemplate
│   │   ├── context.rs      # ElementContext
│   │   └── muxer.rs        # MuxerSyncState, PadInfo, StreamType, SyncMode
│   │
│   ├── pipeline/           # Pipeline execution
│   │   ├── graph.rs        # Pipeline DAG (daggy-based)
│   │   ├── unified_executor.rs # Unified Executor (async + RT + isolation)
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
│   │   ├── mux/            # TsMux, TsMuxElement, Mp4Mux (N-to-1 multiplexing)
│   │   ├── codec/          # Media codecs (AV1, audio, image - feature-gated)
│   │   ├── device/         # Hardware devices (V4L2, PipeWire, ALSA, libcamera)
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
├── examples/               # One concept per file, all use Pipeline
│   │   # Basic examples (no features required)
│   ├── 01_hello.rs               # Simplest pipeline: src -> sink
│   ├── 02_transform.rs           # Transform element: src -> xfm -> sink
│   ├── 03_tee.rs                 # Fan-out: src -> tee -> [sink, sink]
│   ├── 04_funnel.rs              # Fan-in: [src, src] -> funnel -> sink
│   ├── 05_queue.rs               # Backpressure with queue
│   ├── 06_appsrc.rs              # Application integration
│   ├── 07_file_io.rs             # File read/write
│   ├── 08_tcp.rs                 # TCP streaming
│   ├── 09_typed.rs               # Type-safe pipeline API
│   ├── 10_builder.rs             # Fluent builder DSL with >> operator
│   ├── 11_buffer_pool.rs         # Pre-allocated buffer pooling
│   ├── 12_isolation.rs           # Process isolation modes
│   │   # Codec examples (require feature flags)
│   ├── 13_image.rs               # PNG codec (--features image-codecs)
│   ├── 14_h264.rs                # H.264 encoding (--features h264)
│   ├── 15_av1.rs                 # AV1 encoding (--features av1-encode)
│   ├── 16_mpegts.rs              # MPEG-TS muxing (--features mpeg-ts)
│   ├── 17_multi_format_caps.rs   # Multi-format caps negotiation
│   │   # Device examples (require feature flags)
│   ├── 22_v4l2_capture.rs        # V4L2 camera capture (--features v4l2)
│   ├── 23_v4l2_display.rs        # V4L2 display output (--features v4l2)
│   ├── 24_autovideosink.rs       # Auto video sink selection
│   ├── 41_format_converters.rs   # Format conversion elements
│   ├── 42_pipewire_audio.rs      # PipeWire audio (--features pipewire)
│   ├── 43_alsa_audio.rs          # ALSA audio (--features alsa)
│   └── 44_libcamera_capture.rs   # libcamera capture (--features libcamera)
│
├── docs/                   # Documentation
│   ├── design.md           # Complete design document and competitive analysis
│   ├── architecture.md     # High-level architecture overview
│   ├── api.md              # API reference
│   ├── memory.md           # Memory management details
│   ├── plugins.md          # Plugin development guide
│   ├── security.md         # Security model and sandboxing
│   ├── getting-started.md  # Quick start guide
│   │   # Design documents
│   ├── caps-negotiation-research.md  # Caps negotiation research
│   ├── vulkan-video-design.md        # Vulkan Video integration design
│   ├── iced-integration-design.md    # Iced GUI integration design
│   ├── foundation-design.md          # Foundation layer design
│   ├── elements-roadmap.md           # Elements implementation roadmap
│   └── media-streaming-plan.md       # Media streaming plan
│
└── plans/                  # Implementation plans (internal)
    └── README.md           # Plan index and status
```

### Unified Element System (Plan 05)

Parallax provides a simplified element API that eliminates adapter boilerplate:

```rust
use parallax::element::{SimpleSource, SimpleSink, SimpleTransform, ProcessOutput};
use parallax::element::{Src, Snk, Xfm};
use parallax::pipeline::Pipeline;

// Define a simple source
struct Counter { count: u32, max: u32 }

impl SimpleSource for Counter {
    fn produce(&mut self) -> Result<ProcessOutput> {
        if self.count >= self.max {
            return Ok(ProcessOutput::Eos);
        }
        self.count += 1;
        Ok(ProcessOutput::buffer(create_buffer(self.count)))
    }
}

// Define a simple sink
struct Logger;

impl SimpleSink for Logger {
    fn consume(&mut self, buffer: &Buffer) -> Result<()> {
        println!("Received: {} bytes", buffer.len());
        Ok(())
    }
}

// Define a simple transform
struct Doubler;

impl SimpleTransform for Doubler {
    fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput> {
        // Transform and return
        Ok(ProcessOutput::buffer(transformed))
    }
}

// Use with Pipeline.add_element() - no adapters needed!
let mut pipeline = Pipeline::new();
let src = pipeline.add_element("src", Src(Counter { count: 0, max: 10 }));
let xfm = pipeline.add_element("xfm", Xfm(Doubler));
let sink = pipeline.add_element("sink", Snk(Logger));
pipeline.link(src, xfm)?;
pipeline.link(xfm, sink)?;
```

**Key types:**
- `ProcessOutput` - Unified output enum (None, Buffer, Buffers, Eos, Pending)
- `SimpleSource` - Trait for sync sources
- `SimpleSink` - Trait for sync sinks
- `SimpleTransform` - Trait for sync transforms
- `Src<T>`, `Snk<T>`, `Xfm<T>` - Wrapper types implementing `PipelineElement`
- `PipelineElementAdapter` - Bridge to legacy `AsyncElementDyn`

See `examples/40_unified_elements.rs` for a complete example.

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
    fn flush(&mut self) -> Result<Option<Buffer>> { Ok(None) }  // Called at EOS
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn is_rt_safe(&self) -> bool { false }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
}

pub trait Transform: Send {
    fn transform(&mut self, buffer: Buffer) -> Result<Output>;
    fn flush(&mut self) -> Result<Output> { Ok(Output::None) }  // Called at EOS
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn is_rt_safe(&self) -> bool { false }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
}

// Codec traits for video encoders/decoders
pub trait VideoEncoder: Send {
    type Packet: AsRef<[u8]> + Send;
    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Self::Packet>>;
    fn flush(&mut self) -> Result<Vec<Self::Packet>>;  // Drain buffered frames at EOS
}

pub trait VideoDecoder: Send {
    fn decode(&mut self, packet: &[u8]) -> Result<Vec<VideoFrame>>;
    fn flush(&mut self) -> Result<Vec<VideoFrame>>;  // Drain buffered frames at EOS
}

// EncoderElement/DecoderElement wrap codec traits for pipeline use
let encoder = Rav1eEncoder::new(config)?;
let element = EncoderElement::new(encoder, width, height);
pipeline.add_node("enc", DynAsyncElement::new_box(TransformAdapter::new(element)));

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

### Custom Metadata API

Buffers can carry extensible typed metadata for domain-specific data like KLV, SEI NALUs, closed captions, or application-specific values.

```rust
use parallax::metadata::Metadata;

let mut meta = Metadata::new();

// Store and retrieve typed data (any Clone + Send + Sync + Debug + 'static type)
meta.set("app/frame_id", 12345u64);
meta.set("app/quality", 0.95f64);
assert_eq!(meta.get::<u64>("app/frame_id"), Some(&12345));

// Store custom structs
#[derive(Clone, Debug)]
struct GpsPosition { lat: f64, lon: f64 }
meta.set("sensor/gps", GpsPosition { lat: 37.0, lon: -122.0 });

// Convenience methods for raw bytes
meta.set_bytes("h264/sei", vec![0x06, 0x05, 0x10]);
assert_eq!(meta.get_bytes("h264/sei"), Some(&[0x06, 0x05, 0x10][..]));

// KLV/STANAG metadata (common in defense/ISR applications)
meta.set_klv(vec![0x06, 0x0E, 0x2B, 0x34, /* ... */]);
assert!(meta.klv().is_some());

// Mutate in place
if let Some(count) = meta.get_mut::<u32>("app/count") {
    *count += 1;
}

// Remove metadata
let removed: Option<u64> = meta.remove("app/frame_id");
```

**Key namespaces** (use `"domain/type"` format to avoid collisions):
- `stanag/*` - STANAG/MISB metadata (KLV, VMTI)
- `h264/*`, `h265/*`, `av1/*` - Codec-specific (SEI, OBUs)
- `caption/*` - Closed captions (CEA-608, CEA-708)
- `audio/*` - Audio metadata (loudness, language)
- `sensor/*` - Sensor data (GPS, IMU, gimbal)
- `app/*` - Application-specific data

See `examples/31_av1_pipeline_stanag.rs` for a complete example of attaching KLV metadata to video frames.

### Multi-Format Caps Negotiation

Elements can declare multiple supported formats with memory type coupling, and the pipeline automatically negotiates the best common format. This is inspired by GStreamer's GstCapsFeatures.

**Key types:**
- `ElementMediaCaps` - Holds multiple format+memory combinations, ordered by preference
- `FormatMemoryCap` - Couples a format constraint with memory type constraints
- `VideoFormatCaps`, `AudioFormatCaps` - Format constraints with ranges/lists for dimensions, pixel format, etc.

```rust
use parallax::format::{
    CapsValue, ElementMediaCaps, FormatMemoryCap, MemoryCaps, PixelFormat, VideoFormatCaps,
};

// Declare multiple supported formats (e.g., for a camera source)
impl Source for MyCamera {
    fn output_media_caps(&self) -> ElementMediaCaps {
        let yuyv = VideoFormatCaps {
            width: CapsValue::Fixed(640),
            height: CapsValue::Fixed(480),
            pixel_format: CapsValue::Fixed(PixelFormat::Yuyv),
            framerate: CapsValue::Any,
        };
        let rgb24 = VideoFormatCaps {
            width: CapsValue::Fixed(640),
            height: CapsValue::Fixed(480),
            pixel_format: CapsValue::Fixed(PixelFormat::Rgb24),
            framerate: CapsValue::Any,
        };
        
        // Formats listed in preference order
        ElementMediaCaps::new(vec![
            FormatMemoryCap::new(yuyv.into(), MemoryCaps::cpu_only()),
            FormatMemoryCap::new(rgb24.into(), MemoryCaps::cpu_only()),
        ])
    }
}

// Sink declares what it accepts
impl Sink for MyDisplay {
    fn input_media_caps(&self) -> ElementMediaCaps {
        let rgba = VideoFormatCaps {
            pixel_format: CapsValue::Fixed(PixelFormat::Rgba),
            ..VideoFormatCaps::any()
        };
        ElementMediaCaps::new(vec![
            FormatMemoryCap::new(rgba.into(), MemoryCaps::cpu_only()),
        ])
    }
}
```

**Negotiation behavior:**
- The solver iterates source formats against sink formats
- Returns the first (highest preference) intersection
- If no direct match, looks for converters in the registry
- Error messages list all attempted format combinations

See `examples/17_multi_format_caps.rs` for a complete example.

## Implementation Roadmap

See `docs/design.md` for full details.

| Phase | Focus | Status |
|-------|-------|--------|
| 1 | Memory Foundation (SharedArena) | Complete |
| 2 | Caps Negotiation (global constraint solving) | Complete |
| 3 | Cross-Process IPC (IpcSrc/IpcSink) | Complete |
| 4 | GPU Integration (Vulkan Video) | Planned |
| 5 | Pure Rust Codecs | Partial (audio/image complete, video AV1 ready) |
| 6 | Process Isolation (transparent auto-IPC) | Complete |
| 7 | Plugin System (C-compatible ABI) | Complete |
| 8 | Distribution (Zenoh) | Complete |
| 9 | Hybrid Scheduling (PipeWire-inspired) | Complete |
| 10 | Unified Executor (automatic strategy) | Complete |
| 11 | Device Support (V4L2, PipeWire, ALSA, libcamera) | Complete |

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
parallax = { version = "0.1", features = ["h264"] }        # H.264 encoder/decoder (OpenH264)
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
| Video | H.264 | `h264` | openh264 | No | Requires C++ compiler (g++) |
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

## Device Support

Parallax provides feature-gated device elements for hardware capture and output.

### Device Feature Flags

```toml
[dependencies]
# Video capture
parallax = { version = "0.1", features = ["v4l2"] }       # V4L2 camera capture (Linux)
parallax = { version = "0.1", features = ["libcamera"] }  # libcamera capture (Linux)

# Audio capture/playback
parallax = { version = "0.1", features = ["pipewire"] }   # PipeWire audio (Linux)
parallax = { version = "0.1", features = ["alsa"] }       # ALSA audio (Linux)
```

### Device Summary

| Type | Device | Feature | Crate | Notes |
|------|--------|---------|-------|-------|
| Video | V4L2 | `v4l2` | v4l | Linux video capture |
| Video | libcamera | `libcamera` | libcamera | Modern camera API |
| Audio | PipeWire | `pipewire` | pipewire-rs | Modern Linux audio |
| Audio | ALSA | `alsa` | alsa-rs | Direct ALSA access |

### Build Dependencies

- **v4l2**: Requires `v4l-utils-devel` (Fedora) / `libv4l-dev` (Debian)
- **libcamera**: Requires `libcamera-devel` (Fedora) / `libcamera-dev` (Debian)
- **pipewire**: Requires `pipewire-devel` (Fedora) / `libpipewire-0.3-dev` (Debian)
- **alsa**: Requires `alsa-lib-devel` (Fedora) / `libasound2-dev` (Debian)

### DMA-BUF Export (Zero-Copy GPU Path)

V4L2 sources can export buffers as DMA-BUF file descriptors for zero-copy
integration with GPU pipelines:

```rust
use parallax::elements::device::{V4l2Src, V4l2Config};

let config = V4l2Config {
    dmabuf_export: true,  // Export via VIDIOC_EXPBUF
    ..Default::default()
};
let camera = V4l2Src::with_config("/dev/video0", config)?;

// Camera now declares DmaBuf as preferred memory type
// Pipeline will automatically select DMA-BUF path when downstream supports it
```

The caps negotiation system automatically selects the best memory type:
- If sink prefers DmaBuf and source offers it -> zero-copy DMA-BUF path
- If sink only accepts CPU -> fallback to mmap with copy

**Key types for DMA-BUF support:**

| Type | Description |
|------|-------------|
| `DmaBufSegment` | Memory segment backed by DMA-BUF file descriptor |
| `DmaBufBuffer` | Buffer wrapping a DmaBufSegment with metadata |
| `MemoryCaps::dmabuf_only()` | Accept only DMA-BUF memory |
| `MemoryCaps::dmabuf_preferred()` | Prefer DMA-BUF, fall back to CPU |
| `ProduceResult::OwnDmaBuf` | Return a DmaBufBuffer from a source |

**Example: DMA-BUF negotiation**

```rust
// Source declares multiple memory types (DmaBuf preferred)
fn output_media_caps(&self) -> ElementMediaCaps {
    ElementMediaCaps::new(vec![
        FormatMemoryCap::new(format.into(), MemoryCaps::dmabuf_only()),  // Preferred
        FormatMemoryCap::new(format.into(), MemoryCaps::cpu_only()),     // Fallback
    ])
}

// Sink declares what it accepts
fn input_media_caps(&self) -> ElementMediaCaps {
    ElementMediaCaps::new(vec![
        FormatMemoryCap::new(VideoFormatCaps::any().into(), MemoryCaps::dmabuf_only()),
        FormatMemoryCap::new(VideoFormatCaps::any().into(), MemoryCaps::cpu_only()),
    ])
}

// Pipeline negotiation picks DmaBuf (first common match)
```

See `examples/45_dmabuf_negotiation.rs` for a complete example.

## Performance Notes

- Buffer cloning is O(1) (Arc increment)
- Pool slot acquire/release is O(1) amortized (atomic bitmap)
- All CPU memory is memfd-backed (zero overhead vs malloc, always IPC-ready)
- Cross-process is true zero-copy (same physical pages via mmap)
- Arena allocation: 1 fd per pool, not per buffer (avoids fd limits)
- SharedArena: Cross-process refcounting with O(1) release via lock-free MPSC queue
- BufferPool: Pipeline-level pooling with natural backpressure when exhausted

## Documentation

- `docs/design.md` - Complete design document and competitive analysis
- `docs/architecture.md` - High-level architecture overview
- `docs/api.md` - API reference
- `docs/memory.md` - Memory management details
- `docs/plugins.md` - Plugin development guide
- `docs/security.md` - Security model and sandboxing
- `docs/getting-started.md` - Quick start guide
