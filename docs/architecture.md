# Parallax Architecture

This document describes the internal architecture of Parallax.

## Overview

Parallax is built around three core principles:

1. **Zero-copy by default**: Buffers are passed by reference, not copied
2. **Progressive typing**: Start dynamic, graduate to typed
3. **Shared memory first**: Multi-process pipelines without serialization

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              User API Layer                                  │
├──────────────────────────────┬──────────────────────────────────────────────┤
│       Dynamic Pipeline       │           Typed Pipeline                     │
│  - String parsing            │  - Compile-time type checking                │
│  - Runtime element creation  │  - Zero-cost abstractions                    │
│  - Plugin loading            │  - Type-safe operators                       │
└──────────────────────────────┴──────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Execution Engine                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Pipeline Graph (DAG)                         │   │
│  │  - Nodes: Elements (Source, Transform, Sink)                        │   │
│  │  - Edges: Links with optional properties                            │   │
│  │  - Validation: Cycle detection, connectivity                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                      │                                      │
│                                      ▼                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       Pipeline Executor                              │   │
│  │  - Spawns Tokio tasks for each element                              │   │
│  │  - Connects elements via Kanal channels                             │   │
│  │  - Handles backpressure via bounded channels                        │   │
│  │  - Propagates EOS and errors                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            Memory Subsystem                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │    Buffer    │  │ MemoryHandle │  │MemorySegment │  │  MemoryPool  │    │
│  │    <T>       │──│              │──│   (trait)    │──│              │    │
│  └──────────────┘  └──────────────┘  └──────────────┘  └──────────────┘    │
│                                             │                               │
│                    ┌────────────────────────┼────────────────────────┐     │
│                    ▼                        ▼                        ▼     │
│           ┌──────────────┐         ┌──────────────┐         ┌──────────┐  │
│           │ HeapSegment  │         │SharedMemory  │         │HugePages │  │
│           └──────────────┘         └──────────────┘         └──────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Core Components

### Buffer System

The buffer system provides zero-copy data passing:

```rust
pub struct Buffer<T = ()> {
    memory: MemoryHandle,      // Reference to memory
    metadata: Metadata,        // Timestamps, flags, etc.
    validated: AtomicU8,       // rkyv validation cache
    _marker: PhantomData<T>,   // Type safety
}

pub struct MemoryHandle {
    segment: Arc<dyn MemorySegment>,  // Shared ownership
    offset: usize,                     // Offset in segment
    len: usize,                        // Data length
}
```

**Key properties:**
- Cloning a buffer is O(1) - only increments Arc reference count
- Multiple buffers can share the same segment
- Type parameter `T` enables compile-time type checking

### Memory Segments

Memory segments abstract different memory backends:

```rust
pub trait MemorySegment: Send + Sync {
    fn as_ptr(&self) -> *const u8;
    fn len(&self) -> usize;
    fn memory_type(&self) -> MemoryType;
    fn ipc_handle(&self) -> Option<IpcHandle>;
}
```

**Implementations:**

| Segment | Description | IPC |
|---------|-------------|-----|
| `HeapSegment` | Regular heap allocation | No |
| `SharedMemorySegment` | POSIX shared memory (memfd) | Yes |
| `HugePageSegment` | 2MB/1GB huge pages | Yes |
| `MappedFileSegment` | Memory-mapped file | Yes |

### Memory Pool

The memory pool provides efficient buffer allocation:

```rust
pub struct MemoryPool {
    segment: Arc<dyn MemorySegment>,
    slot_size: usize,
    num_slots: usize,
    bitmap: AtomicBitmap,  // Lock-free slot tracking
}
```

**Loan semantics:**
1. Call `pool.loan()` to acquire a slot
2. Write data to the slot
3. Slot automatically returns to pool when dropped

### Element Traits

Elements are defined by traits:

```rust
// Produces buffers
pub trait Source: Send {
    fn produce(&mut self) -> Result<Option<Buffer<()>>>;
}

// Consumes buffers
pub trait Sink: Send {
    fn consume(&mut self, buffer: Buffer<()>) -> Result<()>;
}

// Transforms buffers
pub trait Element: Send {
    fn process(&mut self, buffer: Buffer<()>) -> Result<Option<Buffer<()>>>;
}

// Async source for I/O-bound operations
pub trait AsyncSource: Send {
    async fn produce_async(&mut self) -> Result<Option<Buffer<()>>>;
}
```

### Pipeline Graph

Pipelines are directed acyclic graphs (DAG):

```rust
pub struct Pipeline {
    graph: StableDag<PipelineNode, Link>,  // daggy-based
    nodes: HashMap<String, NodeId>,         // Name lookup
    state: PipelineState,                   // Ready, Running, etc.
}
```

**Properties:**
- Guaranteed acyclic (enforced by daggy)
- Supports multiple inputs/outputs per node
- Named nodes for easy reference

### Executor

The executor runs the pipeline:

```rust
pub struct PipelineExecutor {
    config: ExecutorConfig,
}

impl PipelineExecutor {
    pub async fn run(&self, pipeline: &mut Pipeline) -> Result<()>;
    pub fn start(&self, pipeline: &mut Pipeline) -> Result<PipelineHandle>;
}
```

**Execution model:**
1. Validate the pipeline graph
2. Create Kanal channels between nodes
3. Spawn a Tokio task for each node
4. Wait for all tasks to complete

## Data Flow

```
┌─────────┐    Buffer    ┌───────────┐    Buffer    ┌─────────┐
│ Source  │─────────────▶│ Transform │─────────────▶│  Sink   │
└─────────┘              └───────────┘              └─────────┘
     │                        │                          │
     │ produce()              │ process()                │ consume()
     │                        │                          │
     ▼                        ▼                          ▼
  Ok(Some(buf))           Ok(Some(buf))              Ok(())
  Ok(None) = EOS          Ok(None) = filter          
```

**Flow rules:**
- Sources produce until they return `Ok(None)` (EOS)
- Transforms can filter by returning `Ok(None)`
- Sinks consume until channel closes
- Errors propagate and abort the pipeline

## IPC Architecture

For multi-process pipelines:

```
┌─────────────────────────────┐        ┌─────────────────────────────┐
│         Process A           │        │         Process B           │
│  ┌───────┐    ┌──────────┐ │        │ ┌──────────┐    ┌───────┐  │
│  │Source │───▶│Publisher │─┼────────┼▶│Subscriber│───▶│ Sink  │  │
│  └───────┘    └──────────┘ │        │ └──────────┘    └───────┘  │
│                     │       │        │       │                    │
│                     ▼       │        │       ▼                    │
│         ┌───────────────┐  │        │  ┌───────────────┐         │
│         │ Shared Memory │◀─┼────────┼─▶│ Shared Memory │         │
│         │   (memfd)     │  │        │  │   (mapped)    │         │
│         └───────────────┘  │        │  └───────────────┘         │
└─────────────────────────────┘        └─────────────────────────────┘
                 │                                │
                 └────────────────────────────────┘
                      Unix Socket (fd passing)
```

**Protocol:**
1. Publisher creates shared memory segment (memfd)
2. Publisher sends file descriptor via SCM_RIGHTS
3. Subscriber maps the same memory
4. Buffers reference the shared memory - no copying!

## Plugin System

Plugins extend Parallax with new elements:

```rust
pub struct PluginDescriptor {
    pub abi_version: u32,
    pub name: *const c_char,
    pub version: *const c_char,
    pub elements: *const ElementDescriptor,
    pub num_elements: usize,
}
```

**Loading process:**
1. `PluginLoader` scans directories for `.so` files
2. Loads library and calls `parallax_plugin_descriptor()`
3. Verifies ABI version
4. Registers elements in `PluginRegistry`

## Event System

Events provide pipeline observability:

```rust
pub enum PipelineEvent {
    Started,
    Stopped,
    StateChanged { old: PipelineState, new: PipelineState },
    Eos,
    Error { message: String, node: Option<String> },
    BufferProcessed { node: String, count: u64 },
    NodeStarted { name: String },
    NodeFinished { name: String, buffers: u64 },
    Custom(String),
}
```

**Delivery:**
- Broadcast channel (multiple subscribers)
- Non-blocking sends
- Subscribers can filter events

## Performance Characteristics

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Buffer clone | O(1) | Arc increment |
| Pool loan | O(n) worst | Bitmap scan, typically O(1) |
| Pool return | O(1) | Atomic bit flip |
| Channel send | O(1) | Kanal bounded channel |
| IPC send | O(n) | n = data size, one copy |
| IPC receive | O(1) | Validation only |
