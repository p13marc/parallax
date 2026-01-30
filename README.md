# Parallax

A Rust-native streaming pipeline engine with zero-copy multi-process support.

Parallax provides both dynamic (runtime-configured) and typed (compile-time safe) pipeline construction, with buffers backed by shared memory for efficient multi-process communication.

## Features

- **Zero-copy buffers**: Shared memory with cross-process reference counting
- **Progressive typing**: Start dynamic, graduate to typed
- **Multi-process pipelines**: memfd + Unix socket IPC
- **Hybrid scheduling**: PipeWire-inspired async + RT thread execution
- **rkyv serialization**: Zero-copy deserialization at boundaries
- **Linux-optimized**: memfd_create, huge pages, memory-mapped files
- **Plugin system**: Dynamic element loading with C ABI compatibility
- **Observability**: Built-in metrics and tracing support

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
parallax = "0.1"
```

### Dynamic Pipeline

Build pipelines at runtime using a GStreamer-like syntax:

```rust
use parallax::pipeline::Pipeline;

#[tokio::main]
async fn main() -> parallax::Result<()> {
    // Parse pipeline from string
    let mut pipeline = Pipeline::parse("filesrc location=input.bin ! passthrough ! nullsink")?;
    
    // Or build programmatically
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("src", source);
    let sink = pipeline.add_sink("sink", sink);
    pipeline.link(src, sink)?;
    
    // Run the pipeline - executor auto-negotiates strategy
    pipeline.run().await?;
    
    Ok(())
}
```

### Element Retrieval (GStreamer-like)

Retrieve and modify elements after pipeline creation:

```rust
use parallax::pipeline::Pipeline;
use parallax::elements::io::FileSrc;

// Use name= property to give elements predictable names
let mut pipeline = Pipeline::parse(
    "filesrc name=source location=input.bin ! passthrough ! nullsink"
)?;

// Retrieve element by name and downcast to concrete type
if let Some(src) = pipeline.get_element::<FileSrc>("source") {
    println!("Reading from: {}", src.path());
}

// Mutable access for modifying properties
if let Some(src) = pipeline.get_element_mut::<FileSrc>("source") {
    *src = FileSrc::new("different.bin");
}

// Without name=, elements get auto-generated names: filesrc_0, passthrough_1, etc.
```

### Typed Pipeline

Build pipelines with compile-time type checking:

```rust
use parallax::typed::{pipeline, from_iter, map, filter, collect};

fn main() -> parallax::Result<()> {
    let source = from_iter(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    let result = pipeline(source)
        .then(filter(|x: &i32| x % 2 == 0))  // Keep even numbers
        .then(map(|x: i32| x * 10))           // Multiply by 10
        .sink(collect::<i32>())
        .run()?
        .into_inner();
    
    println!("Result: {:?}", result);  // [20, 40, 60, 80, 100]
    Ok(())
}
```

### Multi-Process Pipeline

Use shared memory for zero-copy IPC with true cross-process reference counting:

```rust
use parallax::memory::{SharedArena, SharedArenaCache};

// Process A: Owner (creates arena, acquires slots)
let arena = SharedArena::new(4096, 16)?;  // 16 slots of 4KB each
let mut slot = arena.acquire()?;
slot.data_mut()[..5].copy_from_slice(b"hello");

// Send arena fd + IPC reference to Process B
let ipc_ref = slot.ipc_ref();
send_fd_and_ref(arena.fd(), ipc_ref)?;

// Process B: Client (maps arena, accesses slots)
let mut cache = SharedArenaCache::new();
cache.map_arena(received_fd)?;

let client_slot = cache.get_slot(&ipc_ref)?;
assert_eq!(client_slot.data(), b"hello");
// client_slot shares the refcount with Process A's slot!
// When both drop, slot is released via lock-free queue
```

The refcount is stored in the shared memory itself, so atomic operations work across processes. No messages needed for reference counting.

### Automatic Execution Strategy

The unified executor automatically determines the optimal strategy for each element:

```rust
use parallax::pipeline::Pipeline;

// Just run - the executor analyzes element hints and chooses strategies
let mut pipeline = Pipeline::parse("audiosrc ! decoder ! mixer ! audiosink")?;
pipeline.run().await?;  // Automatic: audiosrc=async, decoder=RT, mixer=RT, audiosink=async
```

Elements declare `ExecutionHints` describing their characteristics:
- **trust_level**: Trusted, SemiTrusted, Untrusted
- **processing**: CpuBound, IoBound, MemoryBound
- **latency**: UltraLow, Low, Normal, Relaxed
- **uses_native_code**: true if uses FFI

The executor automatically chooses:
| Characteristics | Strategy |
|----------------|----------|
| Untrusted or uses native code | Isolated process |
| RT affinity + RT-safe | RT thread |
| Low latency + RT-safe | RT thread |
| I/O-bound | Tokio async |

### Manual Execution Control

For advanced cases, configure the executor manually:

```rust
use parallax::pipeline::{Pipeline, Executor, ExecutorConfig, SchedulingMode, RtConfig};

let mut pipeline = Pipeline::parse("audiosrc ! decoder ! mixer ! audiosink")?;

// Disable auto-strategy and configure manually
let config = ExecutorConfig {
    auto_strategy: false,
    scheduling: SchedulingMode::Hybrid,
    rt: RtConfig {
        quantum: 256,           // samples per cycle (5.3ms at 48kHz)
        rt_priority: Some(50),  // SCHED_FIFO (requires CAP_SYS_NICE)
        data_threads: 1,
        bridge_capacity: 16,
        ..Default::default()
    },
    ..Default::default()
};

let executor = Executor::with_config(config);
executor.start(&mut pipeline).await?;
```

The scheduler automatically partitions the graph:
- **I/O-bound elements** (network, file) → Tokio async tasks
- **RT-safe elements** (decoders, mixers) → Dedicated RT threads
- **Untrusted elements** → Isolated processes
- **Lock-free bridges** connect different domains

### Pipeline States

Parallax uses a PipeWire-inspired 3-state model:

```
Suspended <──> Idle <──> Running
```

| State | Resources | Description |
|-------|-----------|-------------|
| **Suspended** | Deallocated | Minimal memory footprint |
| **Idle** | Allocated | Ready to process (paused) |
| **Running** | Allocated | Actively processing |

```rust
let mut pipeline = Pipeline::parse("src ! sink")?;

pipeline.prepare()?;   // Suspended → Idle (allocate, negotiate)
pipeline.activate()?;  // Idle → Running (start processing)
pipeline.pause()?;     // Running → Idle (stop, keep resources)
pipeline.suspend()?;   // Idle → Suspended (release resources)
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pipeline                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Source  │───▶│Transform│───▶│   Tee   │───▶│  Sink   │      │
│  └─────────┘    └─────────┘    └────┬────┘    └─────────┘      │
│                                     │                           │
│                                     ▼                           │
│                               ┌─────────┐                       │
│                               │  Sink   │                       │
│                               └─────────┘                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Shared Memory Foundation                     │
├─────────────────────────────────────────────────────────────────┤
│  All buffers are memfd-backed (zero overhead, always IPC-ready) │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  CpuArena    │  │ SharedArena  │  │  HugePages   │          │
│  │  (1 fd/pool) │  │ (cross-proc) │  │  (2MB/1GB)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │ MappedFile   │  │  GPU Pinned  │                             │
│  │ (persistent) │  │  (planned)   │                             │
│  └──────────────┘  └──────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## Built-in Elements

### Sources

| Element | Description |
|---------|-------------|
| `FileSrc` | Reads buffers from a file |
| `TcpSrc` / `AsyncTcpSrc` | Reads from TCP connection |
| `UdpSrc` / `AsyncUdpSrc` | Reads datagrams from UDP socket |
| `FdSrc` | Reads from a raw file descriptor |
| `AppSrc` | Injects buffers from application code |
| `DataSrc` | Generates buffers from inline data |
| `TestSrc` | Generates test pattern buffers |
| `NullSource` | Produces empty buffers |

### Sinks

| Element | Description |
|---------|-------------|
| `FileSink` | Writes buffers to a file |
| `TcpSink` / `AsyncTcpSink` | Writes to TCP connection |
| `UdpSink` / `AsyncUdpSink` | Sends datagrams to UDP socket |
| `FdSink` | Writes to a raw file descriptor |
| `AppSink` | Extracts buffers to application code |
| `ConsoleSink` | Prints buffers to console for debugging |
| `NullSink` | Discards all buffers |

### Sources (Memory/Test)

| Element | Description |
|---------|-------------|
| `MemorySrc` | Reads buffers from in-memory data |

### Sinks (Memory)

| Element | Description |
|---------|-------------|
| `MemorySink` | Collects buffers into memory |
| `SharedMemorySink` | Thread-safe memory sink |

### Transforms

| Element | Description |
|---------|-------------|
| `PassThrough` | Passes buffers unchanged |
| `RateLimiter` | Limits buffer throughput rate |
| `Valve` | Drops or passes buffers (on/off switch) |
| `Queue` | Async buffer queue with backpressure |
| `Identity` | Pass-through with callbacks for debugging |
| `Delay` / `AsyncDelay` | Adds fixed delay between buffers |
| `Map` | Transforms buffer data with a function |
| `FilterMap` | Transform and filter in one step |
| `Chunk` | Splits buffers into fixed-size chunks |
| `FlatMap` | One-to-many buffer transformation |

### Filtering

| Element | Description |
|---------|-------------|
| `Filter` | Filter buffers by predicate |
| `SampleFilter` | Sample buffers (every Nth, random %, first N) |
| `MetadataFilter` | Filter by stream ID or sequence range |
| `DuplicateFilter` | Remove duplicate buffers by content hash |
| `RangeFilter` | Filter by buffer size or sequence range |
| `RegexFilter` | Filter by regex pattern match |

### Batching

| Element | Description |
|---------|-------------|
| `Batch` | Aggregate multiple buffers into one |
| `Unbatch` | Split aggregated buffer back into chunks |

### Buffer Operations

| Element | Description |
|---------|-------------|
| `BufferTrim` | Trim buffers to maximum size |
| `BufferSlice` | Extract a slice from each buffer |
| `BufferPad` | Pad buffers to minimum size |
| `BufferSplit` | Split buffer at delimiter boundaries |
| `BufferJoin` | Join buffers with delimiter |
| `BufferConcat` | Concatenate buffer contents |

### Metadata Operations

| Element | Description |
|---------|-------------|
| `SequenceNumber` | Adds sequence numbers to buffers |
| `Timestamper` | Adds timestamps (system, monotonic) |
| `MetadataInject` | Injects stream ID, duration, offset |
| `MetadataExtract` | Extract metadata to sideband channel |

### Timing Control

| Element | Description |
|---------|-------------|
| `Timeout` | Generate fallback data on timeout |
| `Debounce` | Suppress rapid buffer bursts |
| `Throttle` | Limit buffer rate (drop excess) |

### Routing

| Element | Description |
|---------|-------------|
| `Tee` | Duplicates buffers to multiple outputs (1-to-N fanout) |
| `Funnel` | Merges multiple inputs into one output (N-to-1) |
| `InputSelector` | Selects one of N inputs (N-to-1 switching) |
| `OutputSelector` | Routes to one of N outputs (1-to-N routing) |
| `Concat` | Concatenates streams sequentially |
| `StreamIdDemux` | Demultiplexes by stream ID |

### Network (Unix/Multicast)

| Element | Description |
|---------|-------------|
| `UnixSrc` / `UnixSink` | Unix domain socket I/O |
| `AsyncUnixSrc` / `AsyncUnixSink` | Async Unix socket I/O |
| `UdpMulticastSrc` | Receive from multicast group |
| `UdpMulticastSink` | Send to multicast group |

### Network (HTTP, requires `http` feature)

| Element | Description |
|---------|-------------|
| `HttpSrc` | HTTP GET source (fetch from URL) |
| `HttpSink` | HTTP POST/PUT sink |

### Network (WebSocket, requires `websocket` feature)

| Element | Description |
|---------|-------------|
| `WebSocketSrc` | WebSocket message receiver |
| `WebSocketSink` | WebSocket message sender |

### Zenoh (requires `zenoh` feature)

| Element | Description |
|---------|-------------|
| `ZenohSrc` | Subscribe to Zenoh key expression |
| `ZenohSink` | Publish to Zenoh key expression |
| `ZenohQueryable` | Handle Zenoh queries |
| `ZenohQuerier` | Send Zenoh queries |

## Memory Model

All CPU memory in Parallax is **memfd-backed by default**. This means:
- Zero overhead compared to regular heap allocation
- Always IPC-ready (can send fd to other processes)
- Cross-process reference counting works automatically

| Backend | Use Case | IPC Support |
|---------|----------|-------------|
| `CpuArena` | Default arena allocation (1 fd per pool) | Yes |
| `SharedArena` | Cross-process refcounting with lock-free release | Yes |
| `HugePageSegment` | High-throughput (reduced TLB misses) | Yes |
| `MappedFileSegment` | Persistent buffers | Yes |

### Cross-Process Reference Counting

Traditional reference counting (like `Arc`) stores the refcount on the heap, which doesn't work across processes. Parallax stores refcounts **in the shared memory itself**:

```
SharedArena Memory Layout:
┌─────────────────────────────────────────────────────────────────┐
│ ArenaHeader (cache-aligned)                                     │
│   magic, version, slot_count, slot_size, arena_id               │
├─────────────────────────────────────────────────────────────────┤
│ ReleaseQueue (lock-free MPSC in shared memory)                  │
│   head: AtomicU32     ← Owner drains here (single consumer)     │
│   tail: AtomicU32     ← Any process pushes here (multi producer)│
│   slots: [AtomicU32]  ← Ring buffer of slot indices             │
├─────────────────────────────────────────────────────────────────┤
│ SlotHeader[0..N] (8 bytes each)                                 │
│   refcount: AtomicU32  ← Works across processes!                │
│   state: AtomicU32     ← Free or Allocated                      │
├─────────────────────────────────────────────────────────────────┤
│ SlotData[0..N] (user data)                                      │
└─────────────────────────────────────────────────────────────────┘
```

**How it works:**
- **Clone**: Atomic increment in shared memory (works across processes)
- **Drop**: Atomic decrement; if 0, push slot index to release queue
- **Reclaim**: Owner drains queue in O(k) where k = released slots

```rust
use parallax::memory::{SharedArena, SharedArenaCache};

// Owner process
let arena = SharedArena::new(4096, 64)?;  // 64 slots of 4KB
let slot = arena.acquire()?;

// Client process (after receiving arena fd)
let mut cache = SharedArenaCache::new();
cache.map_arena(received_fd)?;

let client_slot = cache.get_slot(&ipc_ref)?;
// Both slots share the same atomic refcount in shared memory!
// Drop from any process correctly decrements the shared refcount
```

## Typed Operators

| Operator | Description |
|----------|-------------|
| `map(fn)` | Transform each item |
| `filter(predicate)` | Keep items matching predicate |
| `filter_map(fn)` | Transform and filter in one step |
| `take(n)` | Take first n items |
| `skip(n)` | Skip first n items |
| `inspect(fn)` | Side-effect without modification |

## Performance

| Operation | Cost | Notes |
|-----------|------|-------|
| Buffer clone (same process) | O(1) | Atomic increment |
| Buffer clone (cross-process) | O(1) | Atomic increment in shared memory |
| Buffer access | O(1) | Direct pointer |
| Pool slot acquire | O(1) amortized | Atomic bitmap scan |
| Slot release (SharedArena) | O(1) | Lock-free queue push |
| Slot reclaim (SharedArena) | O(k) | k = released slots, not total slots |
| Tee (N outputs) | O(N) clones | No data copying |
| Cross-process send | O(1) | Just send IPC ref (no copy) |
| Cross-process receive | O(1) | Map arena once, then direct access |

## Documentation

- [Getting Started](docs/getting-started.md) - Quick start guide
- [Architecture](docs/architecture.md) - System design overview
- [API Reference](docs/api.md) - Public API documentation
- [Memory Model](docs/memory.md) - Memory management details
- [Plugin Development](docs/plugins.md) - Creating plugins
- [Design Document](docs/design.md) - Comprehensive design reference
- [Security](docs/security.md) - Sandbox and isolation

## Examples

Run the examples:

```bash
# Basics
cargo run --example 01_hello               # Simplest pipeline
cargo run --example 02_transform           # Custom transforms
cargo run --example 03_tee                 # 1-to-N fanout
cargo run --example 04_funnel              # N-to-1 merge
cargo run --example 05_queue               # Backpressure handling

# Application integration
cargo run --example 06_appsrc              # Inject buffers from app
cargo run --example 07_file_io             # Read/write files

# Network
cargo run --example 08_tcp                 # TCP streaming

# Typed pipelines
cargo run --example 09_typed               # Type-safe pipelines
cargo run --example 10_builder             # Fluent builder DSL

# Memory management
cargo run --example 11_buffer_pool         # Pre-allocated buffer pooling

# Process isolation
cargo run --example 12_isolation           # Execution modes

# Codecs (require feature flags)
cargo run --example 13_image --features image-codecs  # PNG codec
cargo run --example 14_h264 --features h264           # H.264 encoding
cargo run --example 15_av1 --features av1-encode      # AV1 encoding
cargo run --example 16_mpegts --features mpeg-ts      # MPEG-TS muxing

# Caps negotiation
cargo run --example 17_multi_format_caps   # Multi-format negotiation
```

## Benchmarks

Run benchmarks:

```bash
cargo bench
```

## Requirements

- Rust 1.75+
- Linux (uses Linux-specific APIs: memfd_create, SCM_RIGHTS)

## Feature Flags

| Feature | Description |
|---------|-------------|
| `macros` | Plugin authoring convenience macros |
| `huge-pages` | Enable huge page support |
| `http` | HTTP source/sink elements (uses ureq) |
| `websocket` | WebSocket source/sink elements (uses tungstenite) |
| `zenoh` | Zenoh pub/sub and query elements |
| **Device Capture** | |
| `pipewire` | PipeWire audio/video capture (requires libpipewire) |
| `libcamera` | Camera capture via libcamera (requires libcamera) |
| `alsa` | ALSA audio capture/playback (requires libasound) |
| `v4l2` | V4L2 video capture |
| **Codecs** | |
| `image-codecs` | All image codecs (JPEG, PNG) |
| `image-jpeg` | JPEG decoder (zune-jpeg, pure Rust) |
| `image-png` | PNG encoder/decoder (png crate, pure Rust) |
| `audio-codecs` | All audio codecs (FLAC, MP3, AAC, Vorbis) |
| `audio-flac` | FLAC decoder (Symphonia, pure Rust) |
| `audio-mp3` | MP3 decoder (Symphonia, pure Rust) |
| `audio-aac` | AAC decoder (Symphonia, pure Rust) |
| `audio-vorbis` | Vorbis decoder (Symphonia, pure Rust) |
| `av1-encode` | AV1 encoder (rav1e, pure Rust) |
| `av1-decode` | AV1 decoder (dav1d, requires libdav1d) |
| `h264` | H.264 encoder/decoder (OpenH264) |
| `mpeg-ts` | MPEG-TS demuxer |

## License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
