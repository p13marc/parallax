# Parallax - Final Implementation Plan

## Project Name: **Parallax**

The name evokes:
- **Parallel** data flows through pipeline branches
- **Parallax effect** - different perspectives on the same data (typed vs dynamic views)
- **Zero displacement** - like zero-copy, the data doesn't move, only the viewpoint changes

Alternative candidates considered:
- **Conduit** - too generic, already used
- **Aqueduct** - water metaphor, bit heavy
- **Flux** - taken by many projects
- **Torrent** - association with P2P
- **Rill** - small stream, nice but obscure

**Parallax** is unique in the Rust ecosystem (no existing crate with this name) and memorable.

---

## Executive Summary

This document synthesizes insights from three prior analyses into a unified implementation plan for a Rust-native streaming pipeline engine. The design prioritizes:

1. **Shared Memory First**: Buffers backed by shared memory for zero-copy multi-process pipelines
2. **rkyv Serialization**: Zero-copy deserialization at process/network boundaries
3. **Progressive Typing**: Dynamic pipelines for flexibility, typed pipelines for safety
4. **Performance**: Zero-copy, allocation reuse, minimal overhead

---

## Core Architecture

### Memory Model: Shared Memory First

Inspired by iceoryx2's loan-based model, all buffers are backed by memory that can be shared across processes without copying.

```
┌─────────────────────────────────────────────────────────────────┐
│                     Memory Backend Abstraction                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   HeapPool   │  │  SharedMem   │  │   HugePages  │          │
│  │  (default)   │  │   (POSIX)    │  │  (2MB/1GB)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  GPU Pinned  │  │    RDMA      │  │  MappedFile  │          │
│  │  (Phase 3+)  │  │  (Phase 4+)  │  │  (Phase 2)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Buffer Design

```rust
/// A buffer backed by any memory region
pub struct Buffer<T = ()> {
    /// Handle to the memory region
    memory: MemoryHandle,
    /// Metadata (always in regular heap for fast access)
    metadata: Metadata,
    /// Type marker for compile-time safety
    _marker: PhantomData<T>,
}

/// Handle to a memory region (cheap to clone - just Arc + offsets)
#[derive(Clone)]
pub struct MemoryHandle {
    /// The backing segment (shared across clones)
    segment: Arc<dyn MemorySegment>,
    /// Offset within segment
    offset: usize,
    /// Length of this buffer's data
    len: usize,
}

/// Trait for different memory backends
pub trait MemorySegment: Send + Sync {
    /// Raw pointer to segment start
    fn as_ptr(&self) -> *const u8;
    
    /// Mutable pointer (if exclusive access)
    fn as_mut_ptr(&self) -> Option<*mut u8>;
    
    /// Total size of the segment
    fn len(&self) -> usize;
    
    /// Memory type for capability checking
    fn memory_type(&self) -> MemoryType;
    
    /// For IPC: handle that can be sent to another process
    fn ipc_handle(&self) -> Option<IpcHandle>;
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MemoryType {
    /// Regular heap memory (single-process only)
    Heap,
    /// POSIX shared memory (shm_open + mmap)
    SharedMemory,
    /// Huge pages (2MB or 1GB)
    HugePages,
    /// Memory-mapped file
    MappedFile,
    /// GPU-accessible pinned host memory
    GpuAccessible,
    /// GPU device memory
    GpuDevice,
    /// RDMA-registered memory
    RdmaRegistered,
}
```

### Memory Pool with Loan Semantics

Following iceoryx2's pattern, publishers "loan" memory slots from pools:

```rust
/// A pool that manages memory slots for buffers
pub struct MemoryPool {
    /// The backing memory segment
    segment: Arc<dyn MemorySegment>,
    /// Fixed slot size (all slots same size)
    slot_size: usize,
    /// Number of slots
    num_slots: usize,
    /// Bitmap tracking free slots (lock-free)
    free_slots: AtomicBitmap,
}

impl MemoryPool {
    /// Loan a slot from the pool (returns uninitialized memory)
    pub fn loan(&self) -> Option<LoanedSlot> {
        let slot_idx = self.free_slots.acquire_slot()?;
        Some(LoanedSlot {
            pool: self.clone(),
            slot_idx,
            ptr: unsafe { self.slot_ptr(slot_idx) },
            len: self.slot_size,
        })
    }
}

/// RAII guard for a loaned memory slot - returns to pool on drop
pub struct LoanedSlot {
    pool: MemoryPool,
    slot_idx: usize,
    ptr: *mut u8,
    len: usize,
}

impl LoanedSlot {
    /// Write typed data (serialized via rkyv) and convert to Buffer
    pub fn write<T: Archive + Serialize<...>>(self, data: &T) -> Buffer<T> {
        let bytes = rkyv::to_bytes(data).unwrap();
        assert!(bytes.len() <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr, bytes.len());
        }
        Buffer::from_loaned_slot(self, bytes.len())
    }
    
    /// Write raw bytes
    pub fn write_bytes(self, data: &[u8]) -> Buffer {
        assert!(data.len() <= self.len);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr, data.len());
        }
        Buffer::from_loaned_slot(self, data.len())
    }
}

impl Drop for LoanedSlot {
    fn drop(&mut self) {
        // Return slot to pool automatically
        self.pool.free_slots.release_slot(self.slot_idx);
    }
}
```

### Serialization: rkyv at Boundaries

rkyv provides zero-copy deserialization - the receiver reads data directly from shared memory without any parsing or allocation.

```rust
/// Payload representation - optimized for each context
pub enum Payload<T> {
    /// In-process with known type: direct reference, no serialization
    Local {
        slot: LoanedSlot,
        typed: Arc<T>,  // Typed view into the slot's memory
    },
    /// Serialized rkyv bytes: ready for IPC or network
    Archived {
        slot: LoanedSlot,
        len: usize,
    },
}

impl<T: Archive + Serialize<...>> Payload<T> {
    /// Access data - zero-copy in both cases
    pub fn access(&self) -> PayloadRef<'_, T> {
        match self {
            Payload::Local { typed, .. } => PayloadRef::Native(typed.as_ref()),
            Payload::Archived { slot, len } => {
                let bytes = unsafe { std::slice::from_raw_parts(slot.ptr, *len) };
                let archived = rkyv::access::<Archived<T>, _>(bytes).unwrap();
                PayloadRef::Archived(archived)
            }
        }
    }
}

/// Reference to payload data - either native or archived
pub enum PayloadRef<'a, T: Archive> {
    Native(&'a T),
    Archived(&'a Archived<T>),
}
```

### When Serialization Occurs

| Scenario | Serialization | Cost |
|----------|--------------|------|
| Same-process, same type | None | Zero |
| Same-process, type-erased | None (Arc sharing) | Vtable lookup |
| Cross-process (shared memory) | At send, validate at receive | One write, one validate |
| Network | At send, validate at receive | One write, one validate |
| Tee to multiple processes | Once (shared bytes) | Amortized |

---

## Design Decisions

All prior decisions remain in effect:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Async runtime | **Tokio** | Ecosystem support |
| Channels | **Kanal** | Fast, actively maintained, MPMC, sync+async |
| Parser | **winnow** | Faster than nom, better errors |
| Properties | **Both** (string k/v + typed) | Flexibility |
| Error handling | **anyhow + thiserror** | Library + app errors |
| Graph structure | **daggy** | Enforces acyclicity |
| Element dispatch | **Pure trait objects** | Simple, not bottleneck |
| Async sources | **Separate `AsyncSource` trait** | Explicit async boundaries |
| Graph cycles | **No** | DAG only |
| Multi-output pads | **Yes** | Demuxers, flexible routing |
| Metrics | **Yes (metrics-rs)** | Essential for debugging |
| Hot reloading | **Design for it** | Future element swapping |
| Serialization | **rkyv** | Zero-copy IPC/network |
| Memory model | **Shared memory first** | Zero-copy multi-process |

### New Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Default memory backend | **Heap** (opt-in shared memory) | Simple start, explicit IPC |
| Slot sizing | **Fixed size per pool** | Simple, predictable |
| Buffer pool scope | **Per-pipeline** | Isolation, easier cleanup |
| Default channel capacity | **16** | Balance latency/throughput |
| Pad naming | **src/sink, src_0/src_1** | GStreamer familiarity |
| IPC transport | **memfd + Unix socket** | No filesystem, fd passing |
| GPU integration | **Explicit (Phase 3+)** | Thin wrappers, user controls |

---

## Implementation Phases

### Phase 1: Core Types & Memory Foundation

**Goal:** Establish the memory abstraction and buffer types.

**Deliverables:**
1. `MemorySegment` trait
2. `HeapSegment` implementation (default backend)
3. `MemoryPool` with loan semantics and atomic bitmap
4. `LoanedSlot` RAII guard
5. `Buffer<T>` with `MemoryHandle`
6. `Metadata` struct
7. Basic error types

**Key Files:**
```
src/
├── lib.rs
├── memory/
│   ├── mod.rs
│   ├── segment.rs      # MemorySegment trait
│   ├── heap.rs         # HeapSegment implementation
│   ├── pool.rs         # MemoryPool, LoanedSlot
│   └── bitmap.rs       # AtomicBitmap for slot tracking
├── buffer.rs           # Buffer<T>, MemoryHandle
├── metadata.rs         # Metadata, BufferFlags
└── error.rs            # Error types
```

**Tests:**
- Pool acquire/release cycle
- Buffer creation and cloning
- Slot reuse after drop

### Phase 2: Elements & Pipeline Core

**Goal:** Define element traits and basic pipeline execution.

**Deliverables:**
1. `Element`, `Source`, `Sink` traits (sync)
2. `AsyncSource` trait
3. `ElementDyn` trait for dynamic pipelines
4. `Pad` abstraction
5. `Pipeline` struct with DAG (daggy)
6. Basic executor (Tokio tasks + Kanal channels)
7. Built-in elements: `PassThrough`, `NullSink`, `Tee`

**Key Files:**
```
src/
├── element/
│   ├── mod.rs
│   ├── traits.rs       # Element, Source, Sink, AsyncSource
│   ├── dynamic.rs      # ElementDyn, type-erased wrapper
│   ├── pad.rs          # Pad, PadDirection
│   └── registry.rs     # Element factory registry
├── pipeline/
│   ├── mod.rs
│   ├── graph.rs        # Pipeline, Node, Edge (daggy-based)
│   └── executor.rs     # Task spawning, channel wiring
└── elements/
    ├── mod.rs
    ├── passthrough.rs
    ├── tee.rs
    └── null_sink.rs
```

### Phase 3: Sources, Parser & Shared Memory

**Goal:** Add real sources, pipeline parsing, and shared memory backend.

**Deliverables:**
1. `FileSrc` (file reader)
2. `TcpSrc` (async TCP source)
3. Pipeline string parser (winnow)
4. `PosixSharedMemory` backend
5. `IpcHandle` for cross-process sharing
6. Multi-process link (Unix socket + fd passing)

**Key Files:**
```
src/
├── memory/
│   ├── shared.rs       # PosixSharedMemory, memfd
│   └── ipc.rs          # IpcHandle, cross-process sharing
├── pipeline/
│   └── parser.rs       # winnow-based parser
├── link/
│   ├── mod.rs
│   ├── local.rs        # In-process Kanal links
│   └── ipc.rs          # Cross-process links
└── elements/
    ├── filesrc.rs
    └── tcpsrc.rs
```

**Parser Syntax:**
```
filesrc location=test.bin ! passthrough ! tee name=t ! sink
t. ! othersink
```

### Phase 4: Typed Pipeline Builder & Serialization

**Goal:** Compile-time safe pipeline construction and rkyv integration.

**Deliverables:**
1. `TypedPipeline` builder with `.then()`, `.tee()`, `.sink()`
2. Typed-to-dynamic conversion
3. rkyv derives for Buffer, Metadata, common types
4. Automatic serialization at IPC boundaries
5. Network protocol framing

**Key Files:**
```
src/
├── typed/
│   ├── mod.rs
│   └── builder.rs      # TypedPipeline, Then, Tee combinators
├── buffer.rs           # Add rkyv derives
└── link/
    └── network.rs      # TCP transport with rkyv framing
```

**Network Protocol:**
```
┌──────────────────────────────────────┐
│ Magic: "STRM" (4 bytes)              │
│ Version: u16                         │
│ Flags: u16                           │
│ Payload length: u32 (LE)             │
│ CRC32: u32                           │
├──────────────────────────────────────┤
│ rkyv-serialized Buffer               │
│ (validated with bytecheck)           │
└──────────────────────────────────────┘
```

### Phase 5: Events, Observability & Validation

**Goal:** Pipeline monitoring and debugging support.

**Deliverables:**
1. `Event` enum (EOS, Error, StateChanged, etc.)
2. Event channel (Kanal)
3. DOT graph export
4. JSON graph export
5. metrics-rs integration (counters, histograms)
6. tracing spans for debugging
7. Graph validation (cycles, caps compatibility)

**Key Files:**
```
src/
├── pipeline/
│   ├── events.rs       # Event enum, event stream
│   ├── export.rs       # DOT, JSON export
│   └── validate.rs     # Graph validation
└── observability/
    ├── mod.rs
    ├── metrics.rs      # Pipeline metrics
    └── tracing.rs      # Span integration
```

### Phase 6: Advanced Memory Backends

**Goal:** Performance optimizations for specialized use cases.

**Deliverables:**
1. `HugePageSegment` (2MB/1GB pages)
2. `MappedFileSegment` (persistent buffers)
3. GPU pinned memory backend (feature-gated)
4. RDMA memory registration (feature-gated)
5. Memory backend selection in pipeline config

**Key Files:**
```
src/
├── memory/
│   ├── huge_pages.rs   # HugePageSegment
│   ├── mapped_file.rs  # MappedFileSegment
│   ├── gpu.rs          # PinnedHostMemory (optional)
│   └── rdma.rs         # RdmaPool (optional)
└── config.rs           # PipelineConfig with memory settings
```

### Phase 7: Plugin System

**Goal:** Dynamic element loading.

**Deliverables:**
1. Plugin trait definition
2. C ABI for cross-language plugins
3. Dynamic library loading (libloading)
4. Plugin discovery and registration
5. Hot-swap infrastructure

### Phase 8: Optimizations & Polish

**Goal:** Production readiness.

**Deliverables:**
1. Lock-free bitmap optimization
2. NUMA-aware allocation (optional)
3. Comprehensive benchmarks
4. Documentation
5. Example applications

---

## Directory Structure

```
streamer/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── error.rs
│   │
│   ├── memory/
│   │   ├── mod.rs
│   │   ├── segment.rs      # MemorySegment trait
│   │   ├── heap.rs         # HeapSegment
│   │   ├── shared.rs       # PosixSharedMemory
│   │   ├── huge_pages.rs   # HugePageSegment
│   │   ├── mapped_file.rs  # MappedFileSegment
│   │   ├── pool.rs         # MemoryPool, LoanedSlot
│   │   ├── bitmap.rs       # AtomicBitmap
│   │   └── ipc.rs          # IpcHandle
│   │
│   ├── buffer.rs           # Buffer<T>, MemoryHandle, Payload
│   ├── metadata.rs         # Metadata, BufferFlags
│   ├── caps.rs             # Caps trait, DynCaps
│   │
│   ├── element/
│   │   ├── mod.rs
│   │   ├── traits.rs       # Element, Source, Sink, AsyncSource
│   │   ├── dynamic.rs      # ElementDyn
│   │   ├── pad.rs          # Pad, PadDirection
│   │   ├── properties.rs   # Properties, PropertyValue
│   │   └── registry.rs     # ElementRegistry
│   │
│   ├── pipeline/
│   │   ├── mod.rs
│   │   ├── graph.rs        # Pipeline DAG
│   │   ├── executor.rs     # Task spawning
│   │   ├── parser.rs       # String parser
│   │   ├── events.rs       # Event enum
│   │   ├── validate.rs     # Validation
│   │   └── export.rs       # DOT, JSON export
│   │
│   ├── typed/
│   │   ├── mod.rs
│   │   └── builder.rs      # TypedPipeline
│   │
│   ├── link/
│   │   ├── mod.rs
│   │   ├── local.rs        # In-process links
│   │   ├── ipc.rs          # Cross-process links
│   │   └── network.rs      # TCP links
│   │
│   ├── elements/
│   │   ├── mod.rs
│   │   ├── filesrc.rs
│   │   ├── tcpsrc.rs
│   │   ├── passthrough.rs
│   │   ├── tee.rs
│   │   ├── console_sink.rs
│   │   └── null_sink.rs
│   │
│   └── observability/
│       ├── mod.rs
│       ├── metrics.rs
│       └── tracing.rs
│
├── examples/
│   ├── simple_pipeline.rs
│   ├── typed_pipeline.rs
│   ├── multi_process.rs
│   └── tcp_source.rs
│
├── benches/
│   ├── throughput.rs
│   ├── memory_pool.rs
│   └── zero_copy.rs
│
└── tests/
    ├── buffer_tests.rs
    ├── pool_tests.rs
    ├── pipeline_tests.rs
    └── ipc_tests.rs
```

---

## Dependencies

```toml
[package]
name = "parallax"
version = "0.1.0"
edition = "2024"
rust-version = "1.75"

[dependencies]
# Async runtime
tokio = { version = "1", features = ["rt-multi-thread", "net", "fs", "io-util", "macros", "sync"] }

# Channels
kanal = "0.1"

# Graph structure
daggy = { version = "0.9", features = ["stable_dag"] }

# Parsing
winnow = "0.6"

# Serialization
rkyv = { version = "0.8", features = ["validation", "bytecheck"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Buffer utilities
bytes = "1"
smallvec = "1"

# Error handling
thiserror = "2"
anyhow = "1"

# Metrics & logging
metrics = "0.24"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

# Linux system APIs (memfd, mmap, SCM_RIGHTS, etc.)
rustix = { version = "0.38", features = ["mm", "fs", "net", "process"] }

[dev-dependencies]
tokio-test = "0.4"
criterion = "0.5"
proptest = "1"

[features]
default = []
huge-pages = []       # MAP_HUGETLB support
io-uring = []         # Future: io_uring async I/O
gpu = []              # Future: CUDA/Vulkan pinned memory
rdma = []             # Future: RDMA support

[[bench]]
name = "throughput"
harness = false

[[bench]]
name = "memory_pool"
harness = false
```

**Note:** We use `rustix` instead of `nix` for Linux APIs - it's lower-level, faster, and has better ergonomics for our use case.

---

## Key APIs (Preview)

### Creating Buffers

```rust
// From a pool (recommended for hot paths)
let pool = MemoryPool::new(HeapSegment::new(1024 * 1024), 64 * 1024, 16);
let slot = pool.loan().expect("pool exhausted");
let buffer = slot.write_bytes(&data);

// Direct (for simple cases)
let buffer = Buffer::from_bytes(data, Metadata::default());
```

### Dynamic Pipeline

```rust
let mut pipeline = Pipeline::new();

// Parse from string
pipeline.parse("filesrc location=input.bin ! passthrough ! consolesink")?;

// Or build programmatically
let src = pipeline.add("src", FileSrc::new("input.bin"))?;
let filter = pipeline.add("filter", PassThrough::new())?;
let sink = pipeline.add("sink", ConsoleSink::new())?;
pipeline.link(src, filter)?;
pipeline.link(filter, sink)?;

// Run
pipeline.play().await?;
pipeline.wait_eos().await?;
```

### Typed Pipeline

```rust
let pipeline = TypedPipeline::from_source(FileSrc::new("input.bin"))
    .then(PassThrough::new())
    .then(MyTransform::new())
    .sink(ConsoleSink::new());

// Compile-time type checking ensures compatibility
pipeline.run().await?;
```

### Multi-Process Pipeline

```rust
// Process A: Source
let pool = MemoryPool::new(PosixSharedMemory::new("pipeline-pool", size)?, slot_size, num_slots);
let link = IpcLink::publisher("pipeline-stream")?;

let mut src = FileSrc::with_pool("input.bin", pool);
while let Some(buffer) = src.produce()? {
    link.send(buffer).await?;
}

// Process B: Sink
let link = IpcLink::subscriber("pipeline-stream")?;
let mut sink = ConsoleSink::new();

while let Some(buffer) = link.recv().await? {
    sink.consume(buffer)?;
}
```

---

## Performance Guarantees

| Operation | Cost | Notes |
|-----------|------|-------|
| Buffer clone (same process) | O(1) | Arc increment only |
| Buffer access (typed, same process) | O(1) | Direct pointer |
| Buffer access (archived) | O(1) | Pointer + validation once |
| Pool slot acquire | O(1) amortized | Atomic bitmap scan |
| Pool slot release | O(1) | Atomic bit flip |
| Tee (N outputs) | O(N) Arc clones | No data copying |
| Cross-process send | O(data size) | One copy to shared memory |
| Cross-process receive | O(1) | Validation only, no copy |

---

## Resolved Design Questions

All implementation details have been finalized:

| Question | Decision | Rationale |
|----------|----------|-----------|
| **Target platform** | **Linux only** | Leverage Linux-specific APIs (memfd, io_uring future, etc.) |
| **Validation caching** | **Validate once, cache result** | Use atomic flag in buffer to track validation state - zero overhead after first access |
| **Pool exhaustion** | **Return error immediately** | Caller decides: retry, backpressure, or fallback |
| **Shared memory naming** | **User-provided + UUID fallback** | Explicit control when needed, convenience otherwise |
| **Shared memory API** | **memfd_create + fd passing** | Anonymous, no filesystem pollution, secure |

### Validation Caching Implementation

```rust
pub struct Buffer<T = ()> {
    memory: MemoryHandle,
    metadata: Metadata,
    /// Atomic flag: 0 = not validated, 1 = validated OK
    validated: AtomicU8,
    _marker: PhantomData<T>,
}

impl<T: Archive> Buffer<T> {
    /// Access archived data - validates once, then zero-cost
    pub fn access(&self) -> Result<&Archived<T>, ValidationError> {
        // Fast path: already validated
        if self.validated.load(Ordering::Acquire) == 1 {
            return Ok(unsafe { self.access_unchecked() });
        }
        
        // Slow path: validate and cache result
        let bytes = self.memory.as_slice();
        rkyv::access::<Archived<T>, rkyv::rancor::Error>(bytes)?;
        self.validated.store(1, Ordering::Release);
        Ok(unsafe { self.access_unchecked() })
    }
    
    /// Skip validation (for trusted sources like same-process)
    pub unsafe fn access_unchecked(&self) -> &Archived<T> {
        rkyv::access_unchecked::<Archived<T>>(self.memory.as_slice())
    }
}
```

### Linux-Specific Optimizations

Since we target Linux only, we can use:

| Feature | API | Benefit |
|---------|-----|---------|
| Anonymous shared memory | `memfd_create()` | No filesystem, auto-cleanup |
| File descriptor passing | `SCM_RIGHTS` | Secure IPC handle transfer |
| Huge pages | `mmap()` + `MAP_HUGETLB` | Reduced TLB misses |
| Memory sealing | `fcntl(F_ADD_SEALS)` | Immutable shared buffers |
| io_uring (future) | `io_uring_*` | Async I/O without syscalls |
| NUMA awareness | `mbind()`, `set_mempolicy()` | Optimal memory placement |

---

## Success Criteria

The prototype succeeds when:

1. [ ] `Pipeline::parse("filesrc location=x ! passthrough ! sink")` works
2. [ ] Typed pipeline with compile-time checking works
3. [ ] Both share the same execution engine
4. [ ] Backpressure works (bounded channels)
5. [ ] Tee element splits to multiple outputs (zero-copy)
6. [ ] EOS propagates correctly
7. [ ] Multi-process pipeline works (shared memory + Unix socket)
8. [ ] TcpSrc demonstrates async source pattern
9. [ ] DOT export generates valid Graphviz
10. [ ] Benchmarks show expected zero-copy behavior

---

## Next Steps

1. **Review this plan** and let me know if you have questions or changes
2. **Start Phase 1**: Memory abstractions and buffer types
3. **Iterate** with tests at each phase

Ready to begin implementation when you approve.

---

## Ecosystem Impact Assessment

### Can Parallax Improve the Rust & Open Source Ecosystem?

**Yes, significantly.** Here's why:

### Gap in the Ecosystem

Currently, Rust lacks a general-purpose streaming pipeline framework that is:
- **Media-agnostic** (GStreamer is media-focused)
- **Zero-copy by design** (most frameworks copy at boundaries)
- **Multi-process native** (shared memory first, not an afterthought)
- **Both dynamic and typed** (pick flexibility or safety, not both)

| Existing Solution | Limitation |
|-------------------|------------|
| **GStreamer (gstreamer-rs)** | C library with Rust bindings; media-focused; complex |
| **Timely Dataflow** | Academic focus; steep learning curve; no IPC |
| **Arroyo** | Distributed streaming (Kafka-like); heavy runtime |
| **Hydroflow** | Research project; compiler-based; less flexible |
| **tokio channels** | Low-level building blocks, not a framework |

### What Parallax Offers

1. **Zero-Copy Multi-Process Pipelines**
   - No existing Rust crate provides GStreamer-like pipelines with shared memory IPC
   - iceoryx2 does IPC but not pipelines; we combine both paradigms

2. **Progressive Typing**
   - Start dynamic (rapid prototyping), graduate to typed (production safety)
   - Unique in the dataflow space

3. **Linux-Optimized Performance**
   - memfd_create, huge pages, potential io_uring
   - Shows how to build Linux-native Rust infrastructure

4. **rkyv Integration Pattern**
   - Demonstrates zero-copy serialization in a real system
   - Reference implementation for others

### Potential Adopters

| Domain | Use Case |
|--------|----------|
| **Robotics** | Sensor fusion pipelines (ROS2-like but Rust-native) |
| **Video/Audio** | Processing pipelines without GStreamer dependency |
| **IoT/Edge** | Data processing on constrained Linux devices |
| **Finance** | Low-latency data pipelines |
| **Observability** | Log/metrics processing pipelines |
| **ML Inference** | Pre/post-processing around model inference |
| **Game Engines** | Asset processing pipelines |

### Contributions Back to Ecosystem

Even if Parallax itself doesn't become widely adopted, it will produce:

1. **Reference Implementations**
   - Lock-free memory pool with atomic bitmap
   - memfd + SCM_RIGHTS IPC pattern
   - rkyv validation caching
   - Typed-to-dynamic pipeline bridging

2. **Documentation**
   - How to build zero-copy systems in Rust
   - Linux memory APIs from Rust (rustix patterns)
   - Pipeline execution patterns

3. **Benchmarks**
   - Comparative data on channel implementations
   - Memory pool strategies
   - Zero-copy verification

4. **Potential Spin-off Crates**
   - `parallax-memory` - standalone memory pool
   - `parallax-ipc` - shared memory IPC primitives
   - Could be useful independently

### Realistic Assessment

**Strengths:**
- Fills a real gap (no Rust-native GStreamer alternative)
- Modern design (rkyv, shared memory, typed+dynamic)
- Linux focus allows optimization depth

**Challenges:**
- GStreamer has 20+ years of plugins/ecosystem
- Adoption requires demonstrating clear benefits
- Documentation and examples are crucial

**Success Indicators:**
- Used in at least one production system
- Cited as reference for Rust pipeline patterns
- Spin-off crates used independently
- Community contributions (elements, backends)

### Bottom Line

Parallax can meaningfully contribute to the Rust ecosystem by:
1. Providing a missing abstraction (streaming pipelines)
2. Demonstrating advanced Rust patterns (zero-copy, progressive typing)
3. Producing reusable components (memory pools, IPC primitives)

The Linux-only focus is actually an advantage - it allows demonstrating what's possible when you don't compromise for portability, setting a performance bar that others can aspire to.
