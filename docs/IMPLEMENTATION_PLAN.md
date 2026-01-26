# Implementation Plan: Phases 1-3

This plan covers the work needed to make Parallax production-ready:
- **Phase 1**: Memory Foundation (unified CpuSegment, arena allocation)
- **Phase 2**: Caps Negotiation (global constraint solving)
- **Phase 3**: Process Isolation (seccomp, namespaces, supervisor)

All three phases are required for production readiness.

---

## Phase 1: Memory Foundation

**Goal**: Unify memory model so all CPU buffers are memfd-backed and IPC-ready by default.

### 1.1 Unified CpuSegment

**Current state**: Separate `HeapSegment` and `SharedMemorySegment`.

**Target**: Single `CpuSegment` that's always memfd-backed.

```rust
// src/memory/cpu.rs (NEW)

/// CPU memory segment - always memfd-backed, always IPC-ready.
/// Replaces both HeapSegment and SharedMemorySegment.
pub struct CpuSegment {
    fd: OwnedFd,
    ptr: NonNull<u8>,
    len: usize,
    name: Option<String>,
}

impl CpuSegment {
    /// Allocate new CPU memory (works like malloc, but shareable).
    pub fn new(size: usize) -> Result<Self>;
    
    /// Allocate with a debug name.
    pub fn with_name(name: &str, size: usize) -> Result<Self>;
    
    /// Reconstruct from received fd (cross-process).
    pub unsafe fn from_fd(fd: OwnedFd, size: usize) -> Result<Self>;
    
    /// Get fd for IPC (always available).
    pub fn fd(&self) -> BorrowedFd<'_>;
}

impl MemorySegment for CpuSegment {
    fn memory_type(&self) -> MemoryType { MemoryType::Cpu }
    fn ipc_handle(&self) -> Option<IpcHandle> { /* always Some */ }
}
```

**Tasks**:
- [ ] Create `src/memory/cpu.rs` with `CpuSegment`
- [ ] Update `MemoryType` enum: remove `Heap`, `SharedMemory`, add `Cpu`
- [ ] Deprecate `HeapSegment` (keep for backward compat, internally use CpuSegment)
- [ ] Deprecate `SharedMemorySegment` (alias to CpuSegment)
- [ ] Update all tests

### 1.2 Arena Allocation (CpuArena)

**Problem**: fd limits (default 1024) could be exhausted with many buffers.

**Solution**: One fd per arena, not per buffer.

```rust
// src/memory/arena.rs (NEW)

/// Arena of CPU memory slots (single fd for entire pool).
pub struct CpuArena {
    fd: OwnedFd,
    base: NonNull<u8>,
    total_size: usize,
    slot_size: usize,
    slot_count: usize,
    free_slots: AtomicBitmap,
    arena_id: u64,  // Globally unique ID for IPC
}

impl CpuArena {
    /// Create a new arena with fixed-size slots.
    pub fn new(slot_size: usize, slot_count: usize) -> Result<Self>;
    
    /// Acquire a slot (returns None if full).
    pub fn acquire(&self) -> Option<ArenaSlot>;
    
    /// Get arena ID for cross-process reference.
    pub fn id(&self) -> u64;
    
    /// Get fd for sharing with other processes.
    pub fn fd(&self) -> BorrowedFd<'_>;
}

/// A slot within an arena (RAII, returns on drop).
pub struct ArenaSlot {
    arena: Arc<CpuArena>,
    index: usize,
    offset: usize,
    len: usize,
}

impl ArenaSlot {
    pub fn as_slice(&self) -> &[u8];
    pub fn as_mut_slice(&mut self) -> &mut [u8];
    
    /// Get IPC reference (arena_id + offset).
    pub fn ipc_ref(&self) -> IpcSlotRef;
}

/// Cross-process slot reference (serializable).
#[derive(Clone, Copy, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct IpcSlotRef {
    pub arena_id: u64,
    pub offset: usize,
    pub len: usize,
}
```

**Tasks**:
- [ ] Create `src/memory/arena.rs` with `CpuArena`, `ArenaSlot`, `IpcSlotRef`
- [ ] Add global arena ID generator (AtomicU64)
- [ ] Integrate with existing `MemoryPool` or replace it
- [ ] Add arena fd caching for receivers (HashMap<arena_id, mmap>)
- [ ] Tests for arena allocation/release

### 1.3 Update Buffer to Support Arena

**Current**: `Buffer` holds `MemoryHandle` which wraps `Arc<dyn MemorySegment>`.

**Target**: `MemoryHandle` can also hold `ArenaSlot`.

```rust
// src/buffer.rs (UPDATE)

pub enum MemoryHandle {
    /// Standalone segment (owns its memory).
    Segment(Arc<dyn MemorySegment>),
    /// Slot in an arena (more efficient for pools).
    Arena(ArenaSlot),
}

impl MemoryHandle {
    pub fn as_slice(&self) -> &[u8];
    pub fn as_mut_slice(&mut self) -> Option<&mut [u8]>;
    pub fn ipc_ref(&self) -> Option<IpcSlotRef>;
    pub fn memory_type(&self) -> MemoryType;
}
```

**Tasks**:
- [ ] Add `Arena(ArenaSlot)` variant to `MemoryHandle`
- [ ] Implement all `MemoryHandle` methods for arena variant
- [ ] Update `Buffer::clone()` to handle arena slots
- [ ] Tests for arena-backed buffers

### 1.4 ProcessContext API

**Goal**: Elements don't allocate; pipeline provides pre-allocated output.

```rust
// src/element/context.rs (UPDATE)

/// Context provided to elements for processing.
pub struct ProcessContext<'a> {
    /// Input buffer (read-only view).
    input: MemoryView<'a>,
    /// Output buffer (write-only view, pre-allocated).
    output: MemoryViewMut<'a>,
    /// How much of output was actually used.
    committed_len: usize,
}

impl<'a> ProcessContext<'a> {
    pub fn input(&self) -> &[u8];
    pub fn output(&mut self) -> &mut [u8];
    
    /// Commit the output (set actual size written).
    pub fn commit(&mut self, len: usize);
    
    /// For in-place processing (input == output).
    pub fn in_place(&mut self) -> Option<&mut [u8]>;
}

/// Memory view with explicit access rights.
pub struct MemoryView<'a> {
    data: &'a [u8],
}

pub struct MemoryViewMut<'a> {
    data: &'a mut [u8],
}
```

**Tasks**:
- [ ] Update `ProcessContext` in `src/element/context.rs`
- [ ] Add `MemoryView` and `MemoryViewMut` types
- [ ] Add new `Element` trait method: `fn process_ctx(&mut self, ctx: &mut ProcessContext) -> Result<()>`
- [ ] Keep existing `process(Buffer) -> Result<Option<Buffer>>` for backward compat
- [ ] Update executor to use `ProcessContext` when element supports it

---

## Phase 2: Caps Negotiation

**Goal**: Global constraint-based format + memory negotiation.

### 2.1 CapsValue with Ranges/Lists

**Current**: `Caps` holds `SmallVec<[MediaFormat; 2]>`.

**Target**: Each field can be fixed, range, or list.

```rust
// src/format.rs (UPDATE)

/// A value that can be fixed, range, or list.
#[derive(Clone, Debug, PartialEq)]
pub enum CapsValue<T> {
    /// Exact value.
    Fixed(T),
    /// Range (inclusive).
    Range { min: T, max: T },
    /// List of acceptable values (ordered by preference).
    List(Vec<T>),
    /// Any value accepted.
    Any,
}

impl<T: Ord + Clone> CapsValue<T> {
    pub fn intersect(&self, other: &Self) -> Option<Self>;
    pub fn fixate(&self) -> Option<T>;
    pub fn accepts(&self, value: &T) -> bool;
}

/// Video format with constraints (for negotiation).
#[derive(Clone, Debug)]
pub struct VideoFormatCaps {
    pub width: CapsValue<u32>,
    pub height: CapsValue<u32>,
    pub pixel_format: CapsValue<PixelFormat>,
    pub framerate: CapsValue<Framerate>,
}

/// Audio format with constraints.
#[derive(Clone, Debug)]
pub struct AudioFormatCaps {
    pub sample_rate: CapsValue<u32>,
    pub channels: CapsValue<u16>,
    pub sample_format: CapsValue<SampleFormat>,
}
```

**Tasks**:
- [ ] Add `CapsValue<T>` enum with intersection/fixation logic
- [ ] Add `VideoFormatCaps`, `AudioFormatCaps` structs
- [ ] Add `FormatCaps` enum wrapping all format types
- [ ] Update `Caps` to use new constraint types
- [ ] Tests for intersection and fixation

### 2.2 Memory Caps

**Goal**: Memory type is part of format negotiation, not separate.

```rust
// src/format.rs (UPDATE)

/// Memory capabilities.
#[derive(Clone, Debug)]
pub struct MemoryCaps {
    /// Supported memory types (ordered by preference).
    pub types: CapsValue<MemoryType>,
    /// Can import from these types.
    pub can_import: Vec<MemoryType>,
    /// Can export to these types.
    pub can_export: Vec<MemoryType>,
    /// DRM format modifiers (for DmaBuf).
    pub drm_modifiers: Option<Vec<u64>>,
}

impl MemoryCaps {
    pub fn cpu_only() -> Self;
    pub fn gpu_preferred() -> Self;
    pub fn any() -> Self;
}

/// Combined format + memory caps.
#[derive(Clone, Debug)]
pub struct MediaCaps {
    pub format: FormatCaps,
    pub memory: MemoryCaps,
}
```

**Tasks**:
- [ ] Add `MemoryCaps` struct
- [ ] Add `MediaCaps` combining format + memory
- [ ] Update element traits to return `MediaCaps` instead of `Caps`
- [ ] Backward compat: `Caps` converts to `MediaCaps` with `MemoryCaps::any()`

### 2.3 Negotiation Solver

**Goal**: Solve all constraints globally (not link-by-link).

```rust
// src/negotiation/mod.rs (NEW)
// src/negotiation/solver.rs (NEW)
// src/negotiation/error.rs (NEW)

pub struct NegotiationSolver {
    graph: &PipelineGraph,
    constraints: Vec<Constraint>,
}

impl NegotiationSolver {
    pub fn new(graph: &PipelineGraph) -> Self;
    
    /// Solve all constraints, return negotiated formats per link.
    pub fn solve(&self) -> Result<NegotiationResult, NegotiationError>;
}

pub struct NegotiationResult {
    /// Negotiated format for each link.
    pub link_formats: HashMap<LinkId, MediaFormat>,
    /// Negotiated memory type for each link.
    pub link_memory: HashMap<LinkId, MemoryType>,
    /// Converters to insert.
    pub converters: Vec<ConverterInsertion>,
}

pub struct ConverterInsertion {
    pub link_id: LinkId,
    pub converter: Box<dyn ElementDyn>,
    pub reason: String,
}

/// Rich error with path and suggestions.
#[derive(Debug, thiserror::Error)]
pub enum NegotiationError {
    #[error("No common format:\n{}", .explanation)]
    NoCommonFormat { explanation: String },
    
    #[error("No common memory type:\n{}", .explanation)]
    NoCommonMemory { explanation: String },
    
    #[error("Cycle detected in pipeline")]
    CycleDetected,
}
```

**Tasks**:
- [ ] Create `src/negotiation/mod.rs` module
- [ ] Implement constraint collection from elements
- [ ] Implement constraint propagation (like type inference)
- [ ] Implement intersection/fixation algorithm
- [ ] Implement memory placement optimization (minimize copies)
- [ ] Implement converter insertion
- [ ] Rich error messages with suggestions
- [ ] Integration with pipeline validation

### 2.4 Converter Registry

**Goal**: Know how to convert between formats/memory types.

```rust
// src/negotiation/converters.rs (NEW)

pub struct ConverterRegistry {
    format_converters: HashMap<(FormatId, FormatId), ConverterFactory>,
    memory_converters: HashMap<(MemoryType, MemoryType), ConverterFactory>,
}

impl ConverterRegistry {
    pub fn new() -> Self;
    
    /// Register a format converter.
    pub fn register_format(&mut self, from: FormatId, to: FormatId, factory: ConverterFactory);
    
    /// Register a memory converter.
    pub fn register_memory(&mut self, from: MemoryType, to: MemoryType, factory: ConverterFactory);
    
    /// Find conversion path with cost.
    pub fn find_path(&self, from: &MediaCaps, to: &MediaCaps) -> Option<(Vec<ConverterFactory>, u32)>;
}

pub type ConverterFactory = Box<dyn Fn() -> Box<dyn ElementDyn> + Send + Sync>;
```

**Tasks**:
- [ ] Create `src/negotiation/converters.rs`
- [ ] Implement converter registry with cost model
- [ ] Register built-in converters (colorspace, gpu upload/download)
- [ ] Path finding with minimum cost (Dijkstra)

---

## Phase 3: Process Isolation

**Goal**: Per-element sandboxing with seccomp, namespaces, crash recovery.

### 3.1 Execution Modes

```rust
// src/execution/mod.rs (NEW)
// src/execution/mode.rs (NEW)

#[derive(Clone, Debug)]
pub enum ExecutionMode {
    /// All elements as Tokio tasks in ONE runtime.
    InProcess,
    
    /// Each element in separate sandboxed process.
    Isolated { sandbox: ElementSandbox },
    
    /// Group elements to minimize processes.
    Grouped {
        isolated_patterns: Vec<String>,
        sandbox: ElementSandbox,
        groups: Option<HashMap<String, GroupId>>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GroupId(pub u32);

impl GroupId {
    pub const SUPERVISOR: GroupId = GroupId(0);
}
```

**Tasks**:
- [ ] Create `src/execution/mod.rs` module
- [ ] Define `ExecutionMode` enum
- [ ] Define `GroupId` type
- [ ] Add pattern matching for element names

### 3.2 Element Sandbox

```rust
// src/execution/sandbox.rs (NEW)

pub struct ElementSandbox {
    /// Seccomp filter (syscall allowlist).
    pub seccomp: SeccompPolicy,
    /// Drop to unprivileged user.
    pub uid_gid: Option<(u32, u32)>,
    /// Filesystem isolation.
    pub mount_namespace: bool,
    /// Network isolation.
    pub network_namespace: bool,
    /// Memory/CPU limits.
    pub cgroup_limits: Option<CgroupLimits>,
}

impl Default for ElementSandbox {
    fn default() -> Self {
        Self {
            seccomp: SeccompPolicy::minimal_compute(),
            uid_gid: None,  // Don't change user by default
            mount_namespace: true,
            network_namespace: true,
            cgroup_limits: None,
        }
    }
}

pub enum SeccompPolicy {
    /// Minimal: read, write, mmap, exit, etc. No fs, no net.
    MinimalCompute,
    /// Allow network syscalls.
    WithNetwork,
    /// Allow filesystem syscalls.
    WithFilesystem,
    /// Custom filter.
    Custom(Vec<SeccompRule>),
}

pub struct CgroupLimits {
    pub memory_max: Option<u64>,
    pub cpu_quota: Option<f32>,
}
```

**Tasks**:
- [ ] Create `src/execution/sandbox.rs`
- [ ] Implement `SeccompPolicy` with libseccomp or seccompiler
- [ ] Implement namespace creation (clone flags)
- [ ] Implement cgroup configuration
- [ ] Tests for sandbox creation

### 3.3 Supervisor

```rust
// src/execution/supervisor.rs (NEW)

pub struct Supervisor {
    /// Running element processes.
    processes: HashMap<ElementId, ElementProcess>,
    /// Shared memory arenas (owned by supervisor).
    arenas: HashMap<u64, Arc<CpuArena>>,
    /// Control channel to each element.
    control_channels: HashMap<ElementId, UnixStream>,
    /// Restart policy.
    restart_policy: RestartPolicy,
}

impl Supervisor {
    pub fn new() -> Self;
    
    /// Spawn an element in a sandboxed process.
    pub async fn spawn_element(
        &mut self,
        id: ElementId,
        element: Box<dyn ElementDyn>,
        sandbox: &ElementSandbox,
    ) -> Result<()>;
    
    /// Run the pipeline with given execution mode.
    pub async fn run(&mut self, mode: ExecutionMode) -> Result<()>;
    
    /// Handle element crash.
    async fn handle_crash(&mut self, id: ElementId, status: ExitStatus) -> Result<()>;
}

pub struct ElementProcess {
    pid: Pid,
    control: UnixStream,
    data_channels: Vec<DataChannel>,
}

pub struct RestartPolicy {
    pub max_restarts: u32,
    pub restart_delay: Duration,
    pub backoff: BackoffStrategy,
}
```

**Tasks**:
- [ ] Create `src/execution/supervisor.rs`
- [ ] Implement element spawning with fork/exec
- [ ] Implement control channel (Unix socket + rkyv)
- [ ] Implement crash detection (waitpid)
- [ ] Implement restart with backoff
- [ ] Implement graceful shutdown

### 3.4 Control Protocol

```rust
// src/execution/protocol.rs (NEW)

/// Control message (sent over Unix socket).
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ControlMessage {
    /// Initialize element with caps.
    Init { caps: MediaCaps },
    /// Buffer ready in shared memory.
    BufferReady { slot: IpcSlotRef, metadata: Metadata },
    /// Buffer processed, slot can be reused.
    BufferDone { slot: IpcSlotRef },
    /// State change request.
    StateChange { new_state: ElementState },
    /// Error report.
    Error { code: u32, message: String },
    /// Shutdown request.
    Shutdown,
    /// Heartbeat (for liveness detection).
    Ping,
    Pong,
}

/// Element state.
#[derive(Clone, Copy, Debug, rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ElementState {
    Null,
    Ready,
    Playing,
    Paused,
}
```

**Tasks**:
- [ ] Create `src/execution/protocol.rs`
- [ ] Implement message serialization with rkyv
- [ ] Implement message framing (length-prefixed)
- [ ] Add heartbeat/timeout handling

### 3.5 IPC Elements

**Goal**: Bridge elements for cross-process/cross-binary pipelines.

```rust
// src/elements/ipc_sink.rs (NEW)
// src/elements/ipc_src.rs (NEW)

/// Sink that writes to IPC channel.
pub struct IpcSink {
    path: PathBuf,
    socket: Option<UnixStream>,
    arena: Option<Arc<CpuArena>>,
}

impl IpcSink {
    pub fn new(path: impl AsRef<Path>) -> Self;
}

impl Sink for IpcSink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        // 1. Ensure buffer is in arena (copy if needed)
        // 2. Send IpcSlotRef over socket
        // 3. Wait for BufferDone
    }
}

/// Source that reads from IPC channel.
pub struct IpcSrc {
    path: PathBuf,
    socket: Option<UnixStream>,
    arena_cache: HashMap<u64, MappedArena>,
}

impl IpcSrc {
    pub fn new(path: impl AsRef<Path>) -> Self;
}

impl Source for IpcSrc {
    fn produce(&mut self) -> Result<Option<Buffer>> {
        // 1. Receive IpcSlotRef from socket
        // 2. Map arena if not cached
        // 3. Return buffer view
    }
}
```

**Tasks**:
- [ ] Create `src/elements/ipc_sink.rs`
- [ ] Create `src/elements/ipc_src.rs`
- [ ] Implement arena fd passing (SCM_RIGHTS)
- [ ] Implement arena caching on receiver side
- [ ] Register in ElementFactory
- [ ] Integration tests

### 3.6 Per-Buffer Access Rights

**Goal**: OS-enforced read-only/write-only permissions.

```rust
// src/memory/access.rs (NEW)

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Access {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

impl CpuArena {
    /// Map a slot with specific access rights.
    pub fn map_slot_with_access(&self, index: usize, access: Access) -> Result<MappedSlot>;
}

impl MappedSlot {
    /// Get read-only view (always available).
    pub fn as_slice(&self) -> &[u8];
    
    /// Get mutable view (fails if ReadOnly).
    pub fn as_mut_slice(&mut self) -> Result<&mut [u8], AccessError>;
}
```

**Tasks**:
- [ ] Add `Access` enum
- [ ] Implement mmap with PROT_READ / PROT_WRITE
- [ ] Update `IpcSlotRef` to include access rights
- [ ] Supervisor enforces access based on data flow direction

---

## Integration Tasks

### Update Pipeline API

```rust
impl Pipeline {
    /// Run with specific execution mode.
    pub async fn run_with_mode(&mut self, mode: ExecutionMode) -> Result<()>;
    
    /// Run with default mode (InProcess).
    pub async fn run(&mut self) -> Result<()> {
        self.run_with_mode(ExecutionMode::InProcess).await
    }
}
```

### Update ElementFactory

- Register `ipc_sink`, `ipc_src` elements
- Support loading elements for isolated execution

### CLI Tool

```rust
// src/bin/parallax-launch.rs (NEW)

fn main() -> Result<()> {
    let args = Args::parse();
    
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let mut pipeline = Pipeline::parse(&args.pipeline)?;
        
        let mode = if args.isolated {
            ExecutionMode::Isolated { sandbox: ElementSandbox::default() }
        } else {
            ExecutionMode::InProcess
        };
        
        pipeline.run_with_mode(mode).await
    })
}
```

**Tasks**:
- [ ] Create `src/bin/parallax-launch.rs`
- [ ] Add clap for argument parsing
- [ ] Support `--isolated`, `--list-elements`, `--inspect`

---

## Testing Strategy

### Unit Tests
- Each new module has its own tests
- Memory allocation/deallocation
- Caps intersection and fixation
- Seccomp filter generation

### Integration Tests
- Pipeline with arena-backed buffers
- Cross-process pipeline (IpcSink → IpcSrc)
- Crash recovery (kill element, verify restart)
- Negotiation with format conversion

### Stress Tests
- Many buffers through arena (fd limit)
- Many element processes (runtime overhead)
- Long-running pipeline (memory leaks)

---

## Dependencies to Add

```toml
[dependencies]
# Existing
rustix = { version = "0.38", features = ["fs", "mm", "net", "process"] }
rkyv = { version = "0.8", features = ["validation"] }

# New for Phase 3
seccompiler = "0.4"  # Or libseccomp
nix = { version = "0.29", features = ["process", "sched", "signal"] }
clap = { version = "4", features = ["derive"] }
```

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| 1.1 CpuSegment | 1 day | None |
| 1.2 CpuArena | 2 days | 1.1 |
| 1.3 Buffer update | 1 day | 1.2 |
| 1.4 ProcessContext | 1 day | 1.3 |
| 2.1 CapsValue | 1 day | None |
| 2.2 MemoryCaps | 1 day | 2.1 |
| 2.3 Solver | 3 days | 2.2 |
| 2.4 Converters | 2 days | 2.3 |
| 3.1 ExecutionMode | 0.5 day | None |
| 3.2 Sandbox | 2 days | 3.1 |
| 3.3 Supervisor | 3 days | 3.2 |
| 3.4 Protocol | 1 day | 3.3 |
| 3.5 IPC Elements | 2 days | 3.4, 1.2 |
| 3.6 Access Rights | 1 day | 3.5 |
| CLI + Integration | 2 days | All |

**Total**: ~23 days of focused work

---

## Order of Implementation

Recommended order (some parallelism possible):

1. **Phase 1.1-1.2**: CpuSegment + CpuArena (memory foundation)
2. **Phase 1.3-1.4**: Buffer + ProcessContext (element interface)
3. **Phase 2.1-2.2**: CapsValue + MemoryCaps (format types)
4. **Phase 3.1-3.2**: ExecutionMode + Sandbox (isolation primitives)
5. **Phase 2.3-2.4**: Solver + Converters (negotiation logic)
6. **Phase 3.3-3.4**: Supervisor + Protocol (process management)
7. **Phase 3.5-3.6**: IPC Elements + Access Rights (cross-process data)
8. **Integration**: CLI + tests
