# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Overview

**Parallax** is a Rust-native streaming pipeline engine inspired by GStreamer, designed for zero-copy data processing across single and multi-process pipelines.

### Core Principles

1. **Shared Memory First**: Buffers backed by shared memory for zero-copy multi-process pipelines
2. **Progressive Typing**: Dynamic pipelines (runtime flexibility) + Typed pipelines (compile-time safety)
3. **Zero-Copy by Design**: rkyv serialization, loan-based memory pools, Arc sharing
4. **Linux-Only**: Leverages memfd_create, SCM_RIGHTS, huge pages, potential io_uring
5. **Sync Processing, Async Orchestration**: Element processing is sync; pipeline orchestration is async

### Key Design Decisions

| Aspect | Choice |
|--------|--------|
| Async runtime | Tokio |
| Channels | Kanal (MPMC, sync+async) |
| Parser | winnow |
| Graph structure | daggy (enforces DAG) |
| Serialization | rkyv (zero-copy) |
| Error handling | thiserror (library) + anyhow (app) |
| Metrics | metrics-rs + tracing |
| Linux APIs | rustix |

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

### Current Implementation (Phase 1 & 2 Complete)

```
parallax/
├── src/
│   ├── lib.rs              # Public API exports
│   ├── error.rs            # Error types (thiserror)
│   │
│   ├── memory/             # Memory management (COMPLETE)
│   │   ├── mod.rs          # Module exports
│   │   ├── segment.rs      # MemorySegment trait, MemoryType, IpcHandle
│   │   ├── heap.rs         # HeapSegment (heap-backed, single process)
│   │   ├── shared.rs       # SharedMemorySegment (memfd_create, multi-process)
│   │   ├── pool.rs         # MemoryPool, LoanedSlot (loan semantics)
│   │   ├── bitmap.rs       # AtomicBitmap (lock-free slot tracking)
│   │   └── ipc.rs          # send_fds/recv_fds (SCM_RIGHTS fd passing)
│   │
│   ├── buffer.rs           # Buffer<T>, MemoryHandle (COMPLETE)
│   ├── metadata.rs         # Metadata, BufferFlags (COMPLETE)
│   │
│   ├── element/            # Element system (COMPLETE)
│   │   ├── mod.rs          # Module exports
│   │   ├── traits.rs       # Element, Source, Sink, AsyncSource, adapters
│   │   ├── pad.rs          # Pad, PadDirection, PadTemplate
│   │   └── context.rs      # ElementContext for runtime info
│   │
│   ├── pipeline/           # Pipeline execution (COMPLETE)
│   │   ├── mod.rs          # Module exports
│   │   ├── graph.rs        # Pipeline DAG (daggy-based)
│   │   └── executor.rs     # PipelineExecutor, task spawning, channel wiring
│   │
│   └── elements/           # Built-in elements (COMPLETE - 25+ elements)
│       ├── mod.rs          # Module exports
│       ├── passthrough.rs  # PassThrough - identity element
│       ├── tee.rs          # Tee - fanout (1-to-N)
│       ├── null.rs         # NullSink, NullSource
│       ├── file.rs         # FileSrc, FileSink
│       ├── tcp.rs          # TcpSrc, TcpSink, AsyncTcpSrc, AsyncTcpSink
│       ├── udp.rs          # UdpSrc, UdpSink, AsyncUdpSrc, AsyncUdpSink
│       ├── fd.rs           # FdSrc, FdSink (raw file descriptors)
│       ├── appsrc.rs       # AppSrc - inject from application code
│       ├── appsink.rs      # AppSink - extract to application code
│       ├── datasrc.rs      # DataSrc - inline data source
│       ├── testsrc.rs      # TestSrc - test pattern generator
│       ├── console.rs      # ConsoleSink - debug output
│       ├── queue.rs        # Queue - async buffering with backpressure
│       ├── valve.rs        # Valve - on/off flow control
│       ├── rate_limiter.rs # RateLimiter - throughput limiting
│       ├── funnel.rs       # Funnel - merge N-to-1
│       ├── selector.rs     # InputSelector, OutputSelector
│       ├── concat.rs       # Concat - sequential stream concatenation
│       └── streamid_demux.rs # StreamIdDemux - demux by stream ID
│
├── tests/
│   └── pipeline_integration.rs  # Integration tests
```

### Key Types

```rust
// Buffer with pluggable memory backend
pub struct Buffer<T = ()> {
    memory: MemoryHandle,
    metadata: Metadata,
    validated: AtomicU8,  // rkyv validation cache
    _marker: PhantomData<T>,
}

// Memory segment abstraction
pub trait MemorySegment: Send + Sync {
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&self) -> Option<*mut u8>;
    fn len(&self) -> usize;
    fn memory_type(&self) -> MemoryType;
    fn ipc_handle(&self) -> Option<IpcHandle>;
}

// Loan-based memory pool
pub struct MemoryPool { /* ... */ }
pub struct LoanedSlot { /* RAII guard, returns to pool on drop */ }

// Element traits
pub trait Source: Send {
    fn produce(&mut self) -> Result<Option<Buffer>>;
}

pub trait Sink: Send {
    fn consume(&mut self, buffer: Buffer) -> Result<()>;
}

pub trait Element: Send {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>>;
}

pub trait AsyncSource: Send {
    fn produce(&mut self) -> impl Future<Output = Result<Option<Buffer>>> + Send;
}

pub trait AsyncSink: Send {
    fn consume(&mut self, buffer: Buffer) -> impl Future<Output = Result<()>> + Send;
}
```

## Implementation Phases

1. **Phase 1**: Core Types & Memory Foundation - COMPLETE
   - MemorySegment trait + HeapSegment + SharedMemorySegment
   - MemoryPool with loan semantics
   - Buffer<T> with MemoryHandle
   - IPC fd passing (SCM_RIGHTS)

2. **Phase 2**: Elements & Pipeline Core - COMPLETE
   - Element traits (Source, Sink, Element, ElementDyn)
   - Pad abstraction (Pad, PadDirection, PadTemplate)
   - Pipeline DAG with daggy (cycle detection, validation)
   - PipelineExecutor with Tokio tasks + Kanal channels
   - Built-in elements: PassThrough, Tee, NullSink, NullSource
   - Integration tests

3. **Phase 3**: Sources, Sinks & GStreamer-Equivalent Elements - COMPLETE
   - File I/O: FileSrc, FileSink
   - Network: TcpSrc/Sink, UdpSrc/Sink (sync + async variants)
   - Raw FD: FdSrc, FdSink
   - Application integration: AppSrc, AppSink (with handles)
   - Test/utility: DataSrc, TestSrc, ConsoleSink
   - Transforms: Queue (backpressure/leaky), Valve, RateLimiter
   - Routing: Funnel (N-to-1), InputSelector, OutputSelector, Concat, StreamIdDemux
   - AsyncSource and AsyncSink traits

4. **Phase 4**: Typed Pipeline Builder - COMPLETE
5. **Phase 5**: Events & Observability
6. **Phase 6**: Advanced Memory Backends
7. **Phase 7**: Plugin System
8. **Phase 8**: Optimizations & Polish

## Code Style Guidelines

- Use `rustfmt` defaults
- Prefer `thiserror` for error types
- Use `tracing` for logging/instrumentation
- Derive `rkyv::Archive`, `rkyv::Serialize`, `rkyv::Deserialize` for types that cross process boundaries
- Keep element processing (`process()`) sync; async only for I/O and orchestration
- Document public APIs with examples
- Write tests for each module

## Testing

```bash
just test              # Run all tests with nextest
just test-one NAME     # Run specific test
just test-verbose      # Run with output capture disabled
just watch             # Auto-run on changes
```

## Zenoh Integration

Parallax is designed to be compatible with Zenoh for distributed pipelines:
- Buffers provide `as_bytes()` for ZBytes conversion
- rkyv format is transparent to Zenoh (just bytes)
- Zero-copy possible via shared memory bridge
- Works with all Zenoh topologies

## Performance Notes

- Buffer cloning is O(1) (Arc increment)
- Pool slot acquire/release is O(1) amortized (atomic bitmap)
- rkyv validation is cached (validate once, then zero-cost)
- Cross-process is true zero-copy via shared memory

## Feature Flags

```toml
[features]
default = []
huge-pages = []     # MAP_HUGETLB support
io-uring = []       # Future: io_uring async I/O
gpu = []            # Future: CUDA/Vulkan pinned memory
rdma = []           # Future: RDMA support
```

## Documentation

- `FINAL_PLAN.md` - Complete implementation plan and architecture
