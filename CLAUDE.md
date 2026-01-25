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
# Build
cargo build

# Build release
cargo build --release

# Run tests
cargo test

# Run tests with nextest (if installed)
cargo nextest run

# Run a specific test
cargo test test_name

# Run benchmarks
cargo bench

# Check formatting
cargo fmt --check

# Lint
cargo clippy -- -D warnings
```

## Architecture

### Crate Organization

```
parallax/
├── src/
│   ├── lib.rs              # Public API exports
│   ├── error.rs            # Error types (thiserror)
│   │
│   ├── memory/             # Memory management
│   │   ├── segment.rs      # MemorySegment trait
│   │   ├── heap.rs         # HeapSegment (default)
│   │   ├── shared.rs       # PosixSharedMemory (memfd)
│   │   ├── pool.rs         # MemoryPool, LoanedSlot
│   │   └── bitmap.rs       # AtomicBitmap for slot tracking
│   │
│   ├── buffer.rs           # Buffer<T>, MemoryHandle
│   ├── metadata.rs         # Metadata, BufferFlags
│   ├── caps.rs             # Caps trait, DynCaps
│   │
│   ├── element/            # Element system
│   │   ├── traits.rs       # Element, Source, Sink, AsyncSource
│   │   ├── dynamic.rs      # ElementDyn (type-erased)
│   │   ├── pad.rs          # Pad, PadDirection
│   │   └── registry.rs     # Element factory registry
│   │
│   ├── pipeline/           # Pipeline execution
│   │   ├── graph.rs        # Pipeline DAG (daggy-based)
│   │   ├── executor.rs     # Task spawning, channel wiring
│   │   ├── parser.rs       # String parser (winnow)
│   │   └── events.rs       # Event enum, event stream
│   │
│   ├── typed/              # Typed pipeline builder
│   │   └── builder.rs      # TypedPipeline, .then(), .tee()
│   │
│   ├── link/               # Inter-element connections
│   │   ├── local.rs        # In-process Kanal links
│   │   ├── ipc.rs          # Cross-process (memfd + Unix socket)
│   │   └── network.rs      # TCP links
│   │
│   └── elements/           # Built-in elements
│       ├── filesrc.rs
│       ├── tcpsrc.rs
│       ├── passthrough.rs
│       ├── tee.rs
│       └── sinks.rs        # ConsoleSink, NullSink, ZenohSink
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
pub trait Element: Send {
    type Input: Caps;
    type Output: Caps;
    fn process(&mut self, input: Buffer<Self::Input>) -> Result<Buffer<Self::Output>>;
}

pub trait ElementDyn: Send {
    fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>>;
}
```

## Implementation Phases

1. **Phase 1**: Core Types & Memory Foundation (current)
2. **Phase 2**: Elements & Pipeline Core
3. **Phase 3**: Sources, Parser & Shared Memory
4. **Phase 4**: Typed Pipeline Builder & Serialization
5. **Phase 5**: Events, Observability & Validation
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

## Testing Strategy

```bash
# Unit tests
cargo test

# Integration tests
cargo test --test integration

# Benchmarks
cargo bench

# Property-based tests (with proptest)
cargo test proptest_
```

## Zenoh Integration

Parallax is designed to be compatible with Zenoh for distributed pipelines:

```rust
// ZenohSink publishes buffers to Zenoh topics
impl Sink for ZenohSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        self.publisher.put(buffer.as_bytes()).await?;
        Ok(())
    }
}
```

- Buffers use rkyv format; mark with encoding `"application/rkyv"`
- Zero-copy possible via Zenoh's SharedMemoryManager
- Works with all Zenoh topologies (peer, router, client)

## Performance Notes

- Buffer cloning is O(1) (Arc increment)
- Pool slot acquire/release is O(1) amortized (atomic bitmap)
- rkyv validation is cached (validate once, then zero-cost)
- Use huge pages for large buffer pools (`huge-pages` feature)
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
- `docs/` - Additional design documents (future)

## License

TBD (recommend Apache-2.0 or MIT)
