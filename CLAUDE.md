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
│   │   ├── segment.rs      # MemorySegment trait, MemoryType, IpcHandle
│   │   ├── heap.rs         # HeapSegment (to be replaced by CpuSegment)
│   │   ├── shared.rs       # SharedMemorySegment (memfd_create)
│   │   ├── pool.rs         # MemoryPool, LoanedSlot (loan semantics)
│   │   ├── bitmap.rs       # AtomicBitmap (lock-free slot tracking)
│   │   └── ipc.rs          # send_fds/recv_fds (SCM_RIGHTS)
│   │
│   ├── buffer.rs           # Buffer, MemoryHandle
│   ├── metadata.rs         # Metadata, BufferFlags
│   │
│   ├── element/            # Element system
│   │   ├── traits.rs       # Source, Sink, Element, AsyncSource, AsyncSink, Transform
│   │   ├── pad.rs          # Pad, PadDirection, PadTemplate
│   │   └── context.rs      # ElementContext
│   │
│   ├── pipeline/           # Pipeline execution
│   │   ├── graph.rs        # Pipeline DAG (daggy-based)
│   │   ├── executor.rs     # PipelineExecutor (Tokio tasks + Kanal channels)
│   │   ├── parser.rs       # Pipeline string parser (winnow)
│   │   └── factory.rs      # ElementFactory + PluginRegistry
│   │
│   ├── elements/           # Built-in elements (25+)
│   │   ├── null.rs         # NullSink, NullSource
│   │   ├── passthrough.rs  # PassThrough
│   │   ├── tee.rs          # Tee (fanout)
│   │   ├── file.rs         # FileSrc, FileSink
│   │   ├── tcp.rs          # TcpSrc/Sink (sync + async)
│   │   ├── udp.rs          # UdpSrc/Sink (sync + async)
│   │   ├── zenoh.rs        # ZenohPub, ZenohSub
│   │   ├── queue.rs        # Queue (backpressure)
│   │   ├── valve.rs        # Valve (flow control)
│   │   └── ...             # Many more
│   │
│   └── plugin/             # Plugin system
│       ├── registry.rs     # PluginRegistry
│       ├── loader.rs       # Dynamic loading
│       └── descriptor.rs   # Plugin metadata
│
├── docs/
│   ├── FINAL_DESIGN_PARALLAX.md    # Complete design document
│   └── PLAN_CAPS_NEGOTIATION.md    # Caps negotiation design
```

### Key Types

```rust
// Element traits (sync)
pub trait Source: Send {
    fn produce(&mut self) -> Result<Option<Buffer>>;
}

pub trait Sink: Send {
    fn consume(&mut self, buffer: Buffer) -> Result<()>;
}

pub trait Element: Send {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>>;
}

// Element traits (async - for I/O bound operations)
pub trait AsyncSource: Send {
    fn produce(&mut self) -> impl Future<Output = Result<Option<Buffer>>> + Send;
}

pub trait AsyncSink: Send {
    fn consume(&mut self, buffer: Buffer) -> impl Future<Output = Result<()>> + Send;
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
