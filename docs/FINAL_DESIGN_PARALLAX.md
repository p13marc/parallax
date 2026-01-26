# Parallax: Final Design Document

## Executive Summary

Parallax is a Rust-native streaming pipeline engine designed to compete with GStreamer while offering significant improvements in developer experience, memory efficiency, and type safety. This document consolidates our design decisions and evaluates competitiveness.

---

## Competitive Analysis

### Why Build Another Pipeline Framework?

| Challenge | GStreamer | Parallax Solution |
|-----------|-----------|-------------------|
| **Security** | All elements in-process, no isolation | Per-element sandbox (seccomp, namespaces) |
| **Crash Handling** | One crash = pipeline dies | Element restarts, pipeline continues |
| **Language** | C with GObject (complex, error-prone) | Pure Rust (memory safety, no GObject) |
| **Caps Negotiation** | Link-by-link, cryptic errors | Global constraint solving, rich errors |
| **Memory Management** | Separate ALLOCATION query, complex | Unified format + memory negotiation |
| **Zero-Copy IPC** | Requires careful BufferPool setup | All buffers IPC-ready by default (memfd) |
| **Buffer Access** | Trust-based (any element can write) | OS-enforced per-buffer permissions |
| **Element Development** | GObject boilerplate, C macros | Simple Rust traits, no boilerplate |
| **GPU Integration** | Bolted on (vaapi, nvcodec plugins) | First-class Vulkan Video design |
| **Type Safety** | Runtime caps checking | Optional typed pipelines (compile-time) |

### Current State of Rust Multimedia

From my research:
- **gstreamer-rs**: Rust bindings to GStreamer (still uses C underneath)
- **rust-av**: Pure Rust codecs, but no pipeline framework
- **rust-media**: Abandoned, was for Servo
- **FFmpeg wrappers**: ez-ffmpeg, ffmpeg-next (wraps C)
- **kornia-rs**: Computer vision focused, wraps GStreamer

**Gap**: No pure Rust pipeline framework with zero-copy memory management and modern GPU integration. Parallax fills this gap.

---

## Core Design Principles

### 0. Security-First: Process Isolation by Default

**Key insight**: Inter-process pipelines should be the default, not an optimization.

```
┌─────────────────────────────────────────────────────────────────┐
│                      SUPERVISOR PROCESS                         │
│  • Spawns element processes                                     │
│  • Owns shared memory allocation & lifetime                     │
│  • Handles IPC, restart, teardown                               │
│  • Never runs untrusted code                                    │
└─────────────────────────────────────────────────────────────────┘
        │              │              │              │
        ▼              ▼              ▼              ▼
   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
   │ Element │    │ Element │    │ Element │    │ Element │
   │  (src)  │───▶│ (codec) │───▶│(filter) │───▶│ (sink)  │
   └─────────┘    └─────────┘    └─────────┘    └─────────┘
   seccomp        seccomp        seccomp        seccomp
   no-net         no-net         no-net         net-only
   no-fs          no-fs          no-fs          no-fs
```

**Why processes, even with shared memory?**
- Separate address spaces → memory bugs are contained
- Shared memory is for **performance**, not for **trust**
- Elements only access explicitly mapped buffers

> **Buffers are shared; authority is not.**

**GStreamer's problem**: All elements run in-process. A bug in one codec can corrupt the entire pipeline or leak data from other elements.

**Parallax's solution**: Each element runs in a sandboxed process:

| Threat | GStreamer | Parallax |
|--------|-----------|----------|
| Buffer overflow in codec | Corrupts entire process | Contained to element |
| Use-after-free | Undefined behavior | SIGSEGV, element restarts |
| Malicious plugin | Full process access | Sandboxed, minimal syscalls |
| Crash in element | Pipeline dies | Element restarts, pipeline continues |

**Linux security primitives used**:

```rust
pub struct ElementSandbox {
    /// Minimal syscall allowlist
    pub seccomp: SeccompFilter,
    /// Drop to unprivileged user
    pub uid_gid: Option<(Uid, Gid)>,
    /// Filesystem isolation
    pub mount_namespace: bool,
    /// Network isolation (most elements need none)
    pub network_namespace: bool,
    /// Memory limits
    pub cgroup_limits: Option<CgroupLimits>,
}

impl Default for ElementSandbox {
    fn default() -> Self {
        Self {
            seccomp: SeccompFilter::minimal_compute(),  // No fs, no net
            uid_gid: Some((Uid::nobody(), Gid::nogroup())),
            mount_namespace: true,
            network_namespace: true,  // No network by default
            cgroup_limits: Some(CgroupLimits::default()),
        }
    }
}
```

**Security outcome**:
- A compromised element **cannot** read other elements' memory
- A compromised element **cannot** escape its syscall sandbox
- A compromised element **cannot** access filesystem or network (unless explicitly granted)
- Failures are contained and recoverable

**Data plane vs Control plane separation**:

| Plane | Transport | Content |
|-------|-----------|---------|
| **Control** | Unix socket (IPC) | Caps negotiation, state, errors |
| **Data** | Shared memory (memfd) | Zero-copy buffers, explicit lifetime |

```rust
/// Control message (small, via IPC socket)
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub enum ControlMessage {
    /// Negotiate format
    Caps { offered: MediaCaps },
    /// Buffer ready (data is in shared memory)
    BufferReady { slot: IpcSlotRef, metadata: Metadata },
    /// State change
    StateChange { new_state: ElementState },
    /// Error report
    Error { code: ErrorCode, message: String },
    /// Shutdown request
    Shutdown,
}

/// Data is NEVER in ControlMessage - always via shared memory
pub struct IpcSlotRef {
    pub arena_id: u64,   // Which shared memory region
    pub offset: usize,   // Offset within region
    pub len: usize,      // Size of data
    pub access: Access,  // Read-only, write-only, read-write
}

pub enum Access {
    ReadOnly,   // Element can only read this buffer
    WriteOnly,  // Element can only write (for output buffers)
    ReadWrite,  // Element can modify in-place
}
```

**Per-buffer access rights** (OS-enforced via mmap):

```rust
impl ElementProcess {
    /// Map a buffer with specific access rights
    fn map_buffer(&self, slot: &IpcSlotRef) -> Result<MappedBuffer> {
        let prot = match slot.access {
            Access::ReadOnly => PROT_READ,
            Access::WriteOnly => PROT_WRITE,
            Access::ReadWrite => PROT_READ | PROT_WRITE,
        };
        
        // OS enforces these permissions - element CANNOT exceed them
        let ptr = mmap(None, slot.len, prot, MAP_SHARED, self.arena_fd, slot.offset)?;
        Ok(MappedBuffer { ptr, len: slot.len, access: slot.access })
    }
}
```

**Crash containment & restart**:

```rust
impl Supervisor {
    async fn run_pipeline(&mut self) -> Result<()> {
        loop {
            tokio::select! {
                // Element crashed
                Some((element_id, exit_status)) = self.wait_any_child() => {
                    log::warn!("Element {} crashed: {:?}", element_id, exit_status);
                    
                    if self.restart_policy.should_restart(element_id) {
                        // Respawn element, renegotiate caps
                        self.restart_element(element_id).await?;
                    } else {
                        // Too many restarts, fail pipeline
                        return Err(Error::ElementFailed(element_id));
                    }
                }
                
                // Normal buffer flow continues...
            }
        }
    }
}
```

**Execution modes** (same code, different isolation):

```rust
pub enum ExecutionMode {
    /// All elements in supervisor process (fast, no isolation)
    /// Use for: development, debugging, trusted pipelines
    /// Runtime cost: 1 Tokio runtime, N tasks (most efficient)
    InProcess,
    
    /// Each element in separate process (max isolation, crash-safe)
    /// Use for: production with fully untrusted pipelines
    /// Runtime cost: N+1 Tokio runtimes (supervisor + 1 per element)
    Isolated {
        sandbox: ElementSandbox,
    },
    
    /// Group elements to minimize processes while isolating untrusted code
    /// Use for: complex pipelines with mixed trust levels
    /// Runtime cost: M+1 Tokio runtimes where M = number of groups
    Grouped {
        /// Elements matching these patterns run in isolated processes
        isolated_patterns: Vec<String>,
        /// Sandbox config for isolated elements
        sandbox: ElementSandbox,
        /// Optional: explicit grouping for fine-grained control
        /// Elements in same group share a process (and Tokio runtime)
        /// If None, all isolated elements share ONE process
        groups: Option<HashMap<String, GroupId>>,
    },
}

/// Group identifier for element grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GroupId(pub u32);

impl GroupId {
    /// Reserved: trusted elements run in supervisor process
    pub const SUPERVISOR: GroupId = GroupId(0);
}
```

**Runtime efficiency comparison:**

| Mode | Pipeline with 20 elements | Tokio Runtimes | OS Processes |
|------|---------------------------|----------------|--------------|
| `InProcess` | All trusted | 1 | 1 |
| `Isolated` | All untrusted | 21 | 21 |
| `Grouped` (auto) | 2 codecs untrusted | 2 | 2 |
| `Grouped` (explicit) | Encoders + Decoders separate | 3 | 3 |

**Usage examples:**

```rust
let pipeline = Pipeline::parse(
    "videotestsrc ! h264enc ! mux ! h264dec ! filter ! display"
)?;

// Development: fast, no isolation (1 runtime, 6 tasks)
pipeline.run(ExecutionMode::InProcess).await?;

// Full isolation: every element sandboxed (7 runtimes)
pipeline.run(ExecutionMode::Isolated {
    sandbox: ElementSandbox::default(),
}).await?;

// Grouped (auto): codecs isolated together (2 runtimes)
// - Supervisor: videotestsrc, mux, filter, display (4 tasks)
// - Isolated: h264enc, h264dec (2 tasks, 1 process)
pipeline.run(ExecutionMode::Grouped {
    isolated_patterns: vec!["*enc".into(), "*dec".into()],
    sandbox: ElementSandbox::default(),
    groups: None,  // All isolated share one process
}).await?;

// Grouped (explicit): separate encoder/decoder groups (3 runtimes)
// - Supervisor: videotestsrc, mux, filter, display
// - Group 1: h264enc (encoders)
// - Group 2: h264dec (decoders)
pipeline.run(ExecutionMode::Grouped {
    isolated_patterns: vec!["*enc".into(), "*dec".into()],
    sandbox: ElementSandbox::default(),
    groups: Some(HashMap::from([
        ("h264enc".into(), GroupId(1)),
        ("h265enc".into(), GroupId(1)),
        ("av1enc".into(), GroupId(1)),
        ("h264dec".into(), GroupId(2)),
        ("h265dec".into(), GroupId(2)),
        ("av1dec".into(), GroupId(2)),
    ])),
}).await?;
```

**Why grouping matters:**
- Tokio runtime has overhead (~1-2MB memory, thread pool)
- 20 isolated elements = 20 runtimes = 20+ threads just for scheduling
- Grouped mode: same security for untrusted code, fewer resources
- Elements in same group can still crash independently (task-level), but share runtime

**Non-goals** (explicit):
- In-process plugin model as the default
- Implicit trust between elements
- Sacrificing isolation for maximum performance

---

### 1. Unified Memory Model (memfd-backed CPU memory)

**Key Insight**: There's no reason to distinguish Heap vs SharedMemory for pipeline buffers.

```rust
pub enum MemoryType {
    /// CPU memory (memfd-backed) - always IPC-ready, zero overhead
    Cpu,
    /// DMA-BUF (Linux buffer sharing, GPU-importable)
    DmaBuf,
    /// GPU device memory (Vulkan/wgpu)
    GpuDevice,
    /// GPU-accessible pinned memory
    GpuAccessible,
    /// RDMA-registered (network zero-copy)
    RdmaRegistered,
    /// Memory-mapped file (persistent)
    MappedFile,
}
```

**Why memfd everywhere?**
- `memfd_create` + `MAP_SHARED` has **zero overhead** vs `malloc`
- Every buffer is automatically shareable via fd passing
- No conversion needed before IPC
- Cross-process = same physical pages (true zero-copy)

### 2. Arena Allocation (Solve fd Limits)

**Problem**: Default 1024 fd limit could be exhausted.

**Solution**: One fd per arena, not per buffer.

```rust
pub struct CpuArena {
    fd: OwnedFd,              // ONE fd for entire arena
    base: *mut u8,            // mmap'd base
    total_size: usize,        // e.g., 256MB
    slot_size: usize,         // Per-buffer size
    free_slots: AtomicBitmap, // Lock-free allocation
}

// Cross-process: send arena fd once, then just offsets
pub struct IpcSlotRef {
    arena_id: u64,   // Receiver caches mmap by arena_id
    offset: usize,
    len: usize,
}
```

**Result**: fd usage = O(arenas) ≈ O(pipelines), not O(buffers).

### 3. Pipeline-Managed Allocation (Elements Don't Allocate)

**Problem**: If elements allocate their own buffers, negotiation is bypassed.

**Solution**: Pipeline provides pre-allocated output in the correct memory location.

```rust
pub trait Element: Send {
    /// Process input into pre-allocated output
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<()>;
    
    /// How much output space needed?
    fn output_size_hint(&self, input_len: usize) -> OutputSizeHint;
    
    /// Can process in-place? (zero allocation path)
    fn can_process_in_place(&self) -> bool { false }
}

pub struct ProcessContext<'a> {
    input: MemoryView<'a>,      // Read from here
    output: MemoryViewMut<'a>,  // Write here (pre-allocated)
    committed_len: usize,        // Actual output size
}
```

**Element writer experience**:
```rust
impl Element for BrightnessFilter {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<()> {
        // Don't care if this is GPU, shared mem, RDMA, etc.
        let input = ctx.input();
        let output = ctx.output();
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = (*i as i32 + self.adjustment).clamp(0, 255) as u8;
        }
        Ok(())
    }
    
    fn can_process_in_place(&self) -> bool { true }
}
```

### 4. Global Constraint-Based Negotiation

**GStreamer problem**: Negotiates link-by-link, misses global optima.

**Parallax solution**: Solve all constraints simultaneously (like type inference).

```rust
impl NegotiationSolver {
    pub fn solve(&self) -> Result<NegotiationResult> {
        // 1. Collect format constraints from all elements
        // 2. Collect memory constraints
        // 3. Build constraint graph
        // 4. Solve globally (minimize conversions + copies)
        // 5. Return optimal format + memory placement for each link
    }
}
```

**Benefits**:
- Finds globally optimal format (not just locally compatible)
- Minimizes memory copies across entire pipeline
- Inserts converters only where truly needed
- Rich error messages showing full path and suggestions

### 5. Unified Format + Memory Negotiation

**GStreamer problem**: Caps negotiation and ALLOCATION query are separate phases.

**Parallax solution**: Memory type is part of the format constraint.

```rust
pub struct MediaCaps {
    pub format: FormatCaps,   // Resolution, pixel format, etc.
    pub memory: MemoryCaps,   // Memory type preferences
}

pub struct MemoryCaps {
    pub types: CapsValue<MemoryType>,  // Ordered by preference
    pub can_import: Vec<MemoryType>,
    pub can_export: Vec<MemoryType>,
    pub drm_modifiers: Option<Vec<u64>>,
}
```

This allows expressing constraints like "I support RGB only as DMA-BUF" in one place.

---

## Memory Negotiation: Same Machine vs Cross-Machine

### Same Process
```
Cost: 0
Any memory type works. Pipeline chooses based on downstream needs.
```

### Cross-Process (Same Machine)
```
Cpu → Cpu:       cost 1 (fd passing + mmap = zero-copy!)
DmaBuf → DmaBuf: cost 1 (fd passing)
DmaBuf → GPU:    cost 2 (fd + import)
GPU → GPU:       INVALID (must go through DmaBuf)
```

### Cross-Machine (Network)
```
Cpu → Network → Cpu:         cost 50 (serialize + latency)
GPU → Cpu → Network → GPU:   cost 70 (download + network + upload)
RdmaReg → RDMA → RdmaReg:    cost 5 (near zero-copy)
GPU → GPUDirect RDMA → GPU:  cost 10 (true zero-copy network)

DmaBuf → DmaBuf:             INVALID (fd is local)
Cpu(fd) → Cpu(fd):           INVALID (fd passing doesn't work)
```

---

## Comparison: Element Development

### GStreamer (C)
```c
// 200+ lines of boilerplate for a simple filter
G_DEFINE_TYPE (GstMyFilter, gst_my_filter, GST_TYPE_ELEMENT);

static void gst_my_filter_class_init (GstMyFilterClass *klass) {
    GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
    GstElementClass *element_class = GST_ELEMENT_CLASS (klass);
    
    gobject_class->set_property = gst_my_filter_set_property;
    gobject_class->get_property = gst_my_filter_get_property;
    
    g_object_class_install_property (gobject_class, PROP_BRIGHTNESS,
        g_param_spec_int ("brightness", "Brightness", "Brightness adjustment",
            -255, 255, 0, G_PARAM_READWRITE));
    
    gst_element_class_set_static_metadata (element_class, ...);
    gst_element_class_add_pad_template (element_class, ...);
    // ... 150 more lines ...
}

static GstFlowReturn gst_my_filter_chain (GstPad *pad, GstObject *parent, GstBuffer *buf) {
    GstMyFilter *filter = GST_MY_FILTER (parent);
    GstMapInfo map;
    
    gst_buffer_map (buf, &map, GST_MAP_READWRITE);
    // Finally, the actual processing
    for (gsize i = 0; i < map.size; i++) {
        map.data[i] = CLAMP (map.data[i] + filter->brightness, 0, 255);
    }
    gst_buffer_unmap (buf, &map);
    
    return gst_pad_push (filter->srcpad, buf);
}
```

### Parallax (Rust)
```rust
// Complete filter in ~20 lines
pub struct BrightnessFilter {
    pub adjustment: i32,
}

impl Element for BrightnessFilter {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<()> {
        let input = ctx.input();
        let output = ctx.output();
        
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = (*i as i32 + self.adjustment).clamp(0, 255) as u8;
        }
        Ok(())
    }
    
    fn can_process_in_place(&self) -> bool { true }
}
```

**Lines of code**: ~200 (GStreamer) vs ~20 (Parallax) = **10x reduction**

---

## GPU Integration: Vulkan Video

### Why Vulkan Video?

| Approach | Codecs | Portability | Maintenance |
|----------|--------|-------------|-------------|
| VA-API | H.264, H.265, VP9, AV1 | Linux Intel/AMD | Per-vendor |
| NVENC/NVDEC | H.264, H.265, AV1 | NVIDIA only | NVIDIA only |
| Vulkan Video | H.264, H.265, AV1, VP9 | All GPUs | Single API |

As of 2025, Vulkan Video supports:
- **Decode**: H.264, H.265, AV1, VP9 (all finalized)
- **Encode**: H.264, H.265, AV1 (all finalized)
- **Drivers**: AMD (RADV), Intel (ANV), NVIDIA (proprietary + NVK)

### rust-gpu for CPU/GPU Code Sharing

Write algorithm once, run on both CPU and GPU:

```rust
// Shared algorithm - compiles to both CPU native and SPIR-V
pub fn rgb_to_yuv(rgb: u32) -> u32 {
    let r = ((rgb >> 16) & 0xFF) as f32;
    let g = ((rgb >> 8) & 0xFF) as f32;
    let b = (rgb & 0xFF) as f32;
    
    let y = (0.299 * r + 0.587 * g + 0.114 * b) as u32;
    // ...
    (y << 16) | (u << 8) | v
}

// GPU path uses same function
#[spirv(compute(threads(256)))]
pub fn color_convert_kernel(...) {
    output[idx] = rgb_to_yuv(input[idx]);  // Same function!
}

// CPU path uses same function
pub fn color_convert_cpu(input: &[u32], output: &mut [u32]) {
    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = rgb_to_yuv(*i);  // Same function!
    }
}
```

---

## Is This Competitive Against GStreamer?

### Where Parallax Wins

| Aspect | Advantage |
|--------|-----------|
| **Developer Experience** | 10x less code, no GObject, Rust safety |
| **Memory Efficiency** | Unified memfd (always zero-copy IPC) |
| **Error Messages** | Type-inference-style errors with suggestions |
| **GPU Integration** | First-class Vulkan Video, rust-gpu sharing |
| **Type Safety** | Optional compile-time typed pipelines |
| **Cross-Process** | Zero-copy by default, arena allocation |

### Where GStreamer Wins (Currently)

| Aspect | GStreamer Advantage |
|--------|---------------------|
| **Ecosystem** | 1000+ elements, decades of plugins |
| **Platform Support** | Windows, macOS, Android, iOS, embedded |
| **Hardware Support** | Every capture card, encoder, decoder |
| **Maturity** | Battle-tested in production for 20+ years |
| **Community** | Large community, extensive documentation |

### Realistic Assessment

**Short-term (1-2 years)**: Parallax won't replace GStreamer for complex production workloads. The ecosystem gap is too large.

**Medium-term (3-5 years)**: Parallax can be competitive for:
- New Rust-native applications (no C dependencies desired)
- GPU-centric pipelines (Vulkan Video primary path)
- Embedded Linux with specific codec needs
- Cross-process/distributed pipelines (Zenoh integration)

**Long-term**: If the Rust multimedia ecosystem grows, Parallax could become the "native" choice for Rust applications, similar to how Tokio became the default async runtime.

---

## Pipeline Deployment Modes

A key design goal: **the same pipeline syntax works across all deployment modes**.

### The Three Deployment Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| **Single Binary** | All elements in one process | CLI tools, simple apps, development |
| **Multi-Process** | Supervisor + sandboxed element processes | Production, untrusted codecs |
| **Multi-Binary** | Separate binaries connected via IPC/network | Microservices, distributed systems |

### Single Binary (gst-launch equivalent)

```bash
# CLI tool: parallax-launch
parallax-launch "videotestsrc ! h264enc ! filesink location=out.h264"

# With properties
parallax-launch "filesrc location=input.raw ! videoconvert ! autovideosink"
```

**Rust API:**
```rust
// Simplest form - all in-process
let mut pipeline = Pipeline::parse("videotestsrc ! display")?;
pipeline.run().await?;
```

**Implementation:** Already works today via `Pipeline::parse()` + `ElementFactory`.

### Multi-Process (Isolated Elements)

Same pipeline string, different execution mode:

```rust
// Same pipeline definition
let pipeline = Pipeline::parse("videotestsrc ! h264enc ! filesink location=out.h264")?;

// Run with isolation - each element in sandboxed subprocess
pipeline.run_with_mode(ExecutionMode::Isolated {
    sandbox: ElementSandbox::default(),
}).await?;
```

**Key point:** The pipeline author doesn't change their code. Isolation is a deployment decision.

### Multi-Binary (Federated Pipelines)

For truly separate binaries, we use **bridge elements** that connect pipelines across process/network boundaries:

```rust
// Binary A: Camera service (e.g., systemd service)
let pipeline = Pipeline::parse("v4l2src ! ipc_sink path=/run/parallax/camera")?;
pipeline.run().await?;

// Binary B: Encoder service (separate binary, maybe different user)
let pipeline = Pipeline::parse(
    "ipc_src path=/run/parallax/camera ! h264enc ! ipc_sink path=/run/parallax/h264"
)?;
pipeline.run().await?;

// Binary C: Streamer (yet another binary)
let pipeline = Pipeline::parse(
    "ipc_src path=/run/parallax/h264 ! rtpsink host=10.0.0.1 port=5000"
)?;
pipeline.run().await?;
```

**Cross-machine with Zenoh:**
```rust
// Machine A: Camera
let pipeline = Pipeline::parse("v4l2src ! zenoh_pub key=factory/camera/1")?;

// Machine B: Processing (different machine entirely)
let pipeline = Pipeline::parse(
    "zenoh_sub key=factory/camera/1 ! ai_detect ! zenoh_pub key=factory/detections"
)?;

// Machine C: Dashboard
let pipeline = Pipeline::parse("zenoh_sub key=factory/detections ! display")?;
```

### Bridge Elements

| Element | Transport | Use Case |
|---------|-----------|----------|
| `ipc_sink` / `ipc_src` | Unix socket + memfd | Same machine, zero-copy |
| `zenoh_pub` / `zenoh_sub` | Zenoh (TCP/UDP/shared-mem) | Any topology |
| `rdma_sink` / `rdma_src` | RDMA verbs | HPC, datacenter |

**IPC Bridge (same machine, zero-copy):**
```rust
pub struct IpcSink {
    path: PathBuf,           // Unix socket path
    arena: Option<CpuArena>, // Shared memory arena
}

pub struct IpcSrc {
    path: PathBuf,
    arena_cache: HashMap<u64, MappedArena>,  // Cache mmaps by arena_id
}
```

Data flow:
1. `ipc_sink` allocates buffer in shared arena
2. Sends `IpcSlotRef { arena_id, offset, len }` over Unix socket
3. First message per arena includes fd (SCM_RIGHTS)
4. `ipc_src` mmaps arena once, then just reads offsets
5. **Zero copies** - same physical memory pages

**Zenoh Bridge (any network topology):**
```rust
pub struct ZenohPub {
    session: Arc<zenoh::Session>,
    key_expr: KeyExpr<'static>,
}

pub struct ZenohSub {
    session: Arc<zenoh::Session>,
    subscriber: Subscriber<'static>,
}
```

Zenoh handles:
- Discovery (no hardcoded addresses)
- Routing (peer-to-peer, routed, or mesh)
- Reliability (QoS settings)
- Shared memory optimization (same-machine Zenoh uses shm)

### Pipeline Description Language Extensions

For complex multi-binary setups, a declarative format:

```yaml
# pipeline.yaml - describes a distributed pipeline
pipelines:
  camera-service:
    binary: parallax-camera
    elements: "v4l2src device=/dev/video0 ! ipc_sink path=/run/parallax/camera"
    restart: always
    
  encoder-service:
    binary: parallax-encoder
    elements: "ipc_src path=/run/parallax/camera ! h264enc ! zenoh_pub key=video/h264"
    depends_on: [camera-service]
    sandbox:
      seccomp: minimal
      network: none
      
  streamer:
    binary: parallax-streamer
    elements: "zenoh_sub key=video/h264 ! rtpsink host=${STREAM_HOST}"
    sandbox:
      network: egress-only
```

**Benefits:**
- Single file describes entire distributed system
- Dependency ordering
- Per-pipeline sandbox configuration
- Environment variable substitution
- Integrates with systemd/container orchestration

### Comparison with GStreamer

| Aspect | GStreamer | Parallax |
|--------|-----------|----------|
| **Single binary** | `gst-launch-1.0 "..."` | `parallax-launch "..."` |
| **Multi-process** | Manual (separate pipelines + IPC) | Built-in `ExecutionMode::Isolated` |
| **Multi-binary** | `shmsink`/`shmsrc` (limited) | `ipc_sink`/`ipc_src` (zero-copy) |
| **Cross-machine** | `tcpserversink`/`tcpclientsink` | `zenoh_pub`/`zenoh_sub` (discovery) |
| **Same syntax everywhere** | No (different elements per mode) | **Yes** (just change bridge elements) |

### CLI Tool: `parallax-launch`

```bash
# Basic usage (like gst-launch)
parallax-launch "videotestsrc ! autovideosink"

# With isolation
parallax-launch --isolated "filesrc location=untrusted.mp4 ! decoder ! display"

# List available elements
parallax-launch --list-elements

# Inspect element properties
parallax-launch --inspect h264enc

# Generate pipeline graph (DOT format)
parallax-launch --dot "src ! filter ! sink" > pipeline.dot
```

**Implementation:**
```rust
// src/bin/parallax-launch.rs
fn main() -> Result<()> {
    let args = Args::parse();
    
    let runtime = tokio::runtime::Runtime::new()?;
    runtime.block_on(async {
        let factory = ElementFactory::new();
        
        // Load plugins if specified
        if let Some(plugin_dir) = &args.plugin_dir {
            let registry = PluginRegistry::load_from_dir(plugin_dir)?;
            factory.set_plugin_registry(Arc::new(registry));
        }
        
        let mut pipeline = Pipeline::parse_with_factory(&args.pipeline, &factory)?;
        
        let mode = if args.isolated {
            ExecutionMode::Isolated { sandbox: ElementSandbox::default() }
        } else {
            ExecutionMode::InProcess
        };
        
        pipeline.run_with_mode(mode).await
    })
}
```

---

## Design Decisions (Finalized)

### 1. Dynamic + Static Pipelines (Both)

Both are first-class citizens:

```rust
// Dynamic: Runtime construction, configuration-driven
let pipeline = Pipeline::parse("videotestsrc ! h264enc ! rtpsink")?;

// Typed: Compile-time checked, IDE autocomplete, refactor-safe
let pipeline = source
    .pipe(H264Encoder::new())
    .pipe(RtpSink::new(addr));
```

**Rationale**: Dynamic for flexibility (config files, user input), typed for safety (library APIs, complex pipelines).

### 2. Plugin System: Static + Dynamic (stabby)

Two plugin mechanisms:

**Static Plugins** (feature flags):
```toml
[dependencies]
parallax = { version = "0.1", features = ["video-codecs", "rtp", "zenoh"] }
```

**Dynamic Plugins** (stabby for ABI stability):
```rust
// Plugin crate uses stabby for stable ABI
#[stabby::stabby]
pub trait ElementPlugin {
    fn name(&self) -> &str;
    fn create(&self) -> Box<dyn ElementDyn>;
    fn caps(&self) -> Caps;
}

// Main binary loads at runtime
let plugin = parallax::load_plugin("./libmy_filter.so")?;
registry.register(plugin);
```

**Why stabby?** Rust has no stable ABI. [stabby](https://github.com/ZettaScaleLabs/stabby) provides:
- Stable struct layouts across compiler versions
- Safe trait objects across dylib boundaries
- No C FFI boilerplate

### 3. Platform: Linux-Only

**Decision**: Linux-only, no cross-platform abstraction.

**Rationale**:
- memfd_create, DMA-BUF, io_uring are Linux-specific
- Zero-copy IPC depends on these primitives
- Abstracting would add overhead and complexity
- Target use cases (embedded, servers, Zenoh) are Linux-centric

**Explicit non-goals**: Windows, macOS, iOS, Android support.

### 4. Codec Strategy: Vulkan Video + Pure Rust Fallback

```
┌─────────────────────────────────────────────────┐
│                  Parallax Codec                 │
├─────────────────────────────────────────────────┤
│  Try: Vulkan Video (H.264, H.265, AV1, VP9)    │
│       └─ GPU decode/encode, zero-copy           │
│                                                 │
│  Fallback: Pure Rust                            │
│       ├─ rav1d (AV1 decode) - production ready  │
│       ├─ rav1e (AV1 encode) - production ready  │
│       └─ Future: H.264/H.265 if available       │
│                                                 │
│  NO: FFmpeg/libav (C dependency)                │
└─────────────────────────────────────────────────┘
```

**Rationale**:
- Vulkan Video is now mature (H.264, H.265, AV1, VP9 all finalized)
- Pure Rust fallback for headless/CPU-only systems
- No C dependencies keeps builds simple and safe
- AV1 is the future (royalty-free, 30% of Netflix traffic)

**Current pure Rust codec status**:
| Codec | Decode | Encode | Status |
|-------|--------|--------|--------|
| AV1 | rav1d | rav1e | Production ready |
| VP9 | - | - | No pure Rust impl |
| H.265 | - | - | No pure Rust impl |
| H.264 | openh264-rs? | - | Uncertain |

For H.264/H.265 without GPU: accept that pure Rust options are limited. Users needing these can:
1. Use a system with Vulkan Video support
2. Wait for pure Rust implementations
3. Fork and add FFmpeg (their choice, not ours)

---

## Final Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SUPERVISOR PROCESS (trusted)                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │   Dynamic   │     │    Typed    │     │   Plugin    │           │
│  │  Pipeline   │     │  Pipeline   │     │  Registry   │           │
│  │  (parser)   │     │ (generics)  │     │  (stabby)   │           │
│  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘           │
│         │                   │                   │                   │
│         └───────────────────┼───────────────────┘                   │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    NEGOTIATION SOLVER                         │  │
│  │  • Global constraint solving (format + memory)               │  │
│  │  • Automatic converter insertion                             │  │
│  │  • Rich error messages with suggestions                      │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      SUPERVISOR                               │  │
│  │  • Spawns element processes (sandboxed)                      │  │
│  │  • Owns all memory allocation (arenas)                       │  │
│  │  • Routes control messages (IPC)                             │  │
│  │  • Handles crash recovery & restart                          │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                             │                                       │
│         ┌───────────────────┼───────────────────┐                   │
│         ▼                   ▼                   ▼                   │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐           │
│  │  CpuArena   │     │  GpuPool    │     │  DmaBufPool │           │
│  │  (memfd)    │     │  (Vulkan)   │     │  (kernel)   │           │
│  └─────────────┘     └─────────────┘     └─────────────┘           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  ELEMENT PROCESSES (untrusted, sandboxed)                           │
│                                                                     │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐            │
│  │   Element    │   │   Element    │   │   Element    │            │
│  │   Process    │──▶│   Process    │──▶│   Process    │            │
│  ├──────────────┤   ├──────────────┤   ├──────────────┤            │
│  │ seccomp: min │   │ seccomp: min │   │ seccomp: net │            │
│  │ netns: none  │   │ netns: none  │   │ netns: allow │            │
│  │ fs: none     │   │ fs: none     │   │ fs: none     │            │
│  │ uid: nobody  │   │ uid: nobody  │   │ uid: nobody  │            │
│  └──────────────┘   └──────────────┘   └──────────────┘            │
│        ▲                   ▲                   ▲                    │
│        │ mmap (RO)         │ mmap (RW)         │ mmap (WO)          │
│        └───────────────────┴───────────────────┘                    │
│                    Shared Memory (per-buffer permissions)           │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  TRANSPORT                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐            │
│  │  Local   │  │   IPC    │  │  Zenoh   │  │   RDMA   │            │
│  │(in-proc) │  │(fd pass) │  │(network) │  │(zero-cp) │            │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘            │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│  CODECS                                                             │
│  ┌────────────────────────┐  ┌────────────────────────┐            │
│  │     Vulkan Video       │  │     Pure Rust          │            │
│  │  H.264, H.265, AV1,VP9 │  │  rav1d, rav1e          │            │
│  │  (GPU, zero-copy)      │  │  (CPU fallback)        │            │
│  └────────────────────────┘  └────────────────────────┘            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Roadmap

### Phase 1: Memory Foundation (Current → Q2)
- [x] Basic memory segments (Heap, SharedMemory)
- [ ] Unified CpuSegment (memfd-backed)
- [ ] CpuArena (single fd per pool)
- [ ] ProcessContext API (elements don't allocate)
- [ ] In-place processing optimization

### Phase 2: Negotiation (Q2-Q3)
- [ ] CapsValue with ranges/lists
- [ ] MediaCaps (format + memory unified)
- [ ] Global constraint solver
- [ ] Converter registry with cost model
- [ ] Rich error messages with suggestions

### Phase 3: Cross-Process IPC (Q3)
- [ ] IpcSlotRef (arena_id + offset)
- [ ] Arena fd caching on receiver
- [ ] IpcSrc/IpcSink elements
- [ ] Copy-on-Write for shared buffers

### Phase 4: GPU Integration (Q3-Q4)
- [ ] Vulkan Video decode (H.264, H.265, AV1)
- [ ] Vulkan Video encode
- [ ] DMA-BUF import/export
- [ ] GpuNative trait for elements
- [ ] rust-gpu compute shaders (color convert, scale)

### Phase 5: Pure Rust Codecs (Q4)
- [ ] rav1d integration (AV1 decode)
- [ ] rav1e integration (AV1 encode)
- [ ] Automatic fallback when no GPU

### Phase 6: Process Isolation & Sandboxing (Q4)
- [ ] Supervisor/element process architecture
- [ ] Element spawning with fork/exec
- [ ] Control plane IPC (Unix socket + rkyv)
- [ ] seccomp filters (minimal syscall allowlist)
- [ ] Namespace isolation (mount, network, user)
- [ ] Per-buffer mmap permissions (read-only, write-only)
- [ ] Crash detection and element restart
- [ ] ExecutionMode (InProcess, Isolated, Hybrid)

### Phase 7: Plugin System (Q4-Q1 next)
- [ ] stabby-based plugin trait
- [ ] Dynamic loading with validation
- [ ] Feature-flagged static plugins
- [ ] Plugin sandboxing (plugins run isolated by default)
- [ ] Plugin documentation/examples

### Phase 8: Distribution (Future)
- [ ] Zenoh transport for cross-machine
- [ ] RDMA support (RdmaSegment)
- [ ] GPUDirect RDMA (stretch goal)

---

## What Makes Parallax Unique

### 1. Security-First: Process Isolation

**No other pipeline framework does this by default.**

| Framework | Isolation | Untrusted Code |
|-----------|-----------|----------------|
| GStreamer | None (in-process) | Runs with full access |
| FFmpeg | None (in-process) | Runs with full access |
| PipeWire | Process-based | Designed for trusted system services |
| **Parallax** | **Per-element sandbox** | **seccomp + namespaces + cgroups** |

A malicious or buggy codec in Parallax:
- Cannot read other elements' memory
- Cannot access filesystem (unless granted)
- Cannot access network (unless granted)
- Cannot make arbitrary syscalls
- Can crash without killing the pipeline

> **Buffers are shared; authority is not.**

### 2. Memory Model Innovation

```
All CPU buffers → memfd (zero overhead, always IPC-ready)
     ↓
Arena allocation (1 fd per pool, not per buffer)
     ↓
Supervisor owns allocation (elements just process)
     ↓
Per-buffer access rights (OS-enforced via mmap)
```

**Result**: Zero-copy IPC with OS-enforced read/write permissions per buffer.

### 3. Unified Negotiation

GStreamer: caps negotiation → ALLOCATION query → buffer pool setup (3 phases)
Parallax: single global constraint solve for format + memory + placement

### 4. Element Simplicity

| Framework | Lines for simple filter |
|-----------|------------------------|
| GStreamer (C) | ~200 |
| gstreamer-rs | ~100 |
| Parallax | ~20 |

### 5. No C Dependencies

Pure Rust stack (except system libs):
- Vulkan Video via ash/wgpu (Rust bindings, no C code)
- rav1d/rav1e for software codecs
- stabby for plugins (no C FFI)

### 6. Crash Recovery

Elements can crash without killing the pipeline:
- Supervisor detects child exit
- Respawns element with fresh state
- Renegotiates caps
- Resumes data flow

GStreamer: one SIGSEGV = entire pipeline dies.

---

## Conclusion

**Is this competitive against GStreamer?**

**Architecturally**: Yes, significantly better.
- Unified memory model (memfd everywhere)
- Global constraint solving (not link-by-link)
- 10x simpler element development
- First-class GPU design (Vulkan Video, rust-gpu)
- Type-safe pipelines (optional)

**Practically**: Not yet.
- GStreamer has 1000+ plugins, 20+ years of ecosystem
- Parallax has ~43 elements, mostly networking/utility
- Video codecs need Vulkan Video or pure Rust implementations

**Strategy to compete**:

1. **Focus on niches where GStreamer is weak**:
   - Pure Rust applications (no C dependencies)
   - GPU-centric pipelines (Vulkan Video primary)
   - Distributed systems (Zenoh integration)
   - Embedded Linux (minimal footprint)

2. **Don't try to replace GStreamer for everything**:
   - Accept that legacy codec support may never match
   - Target new applications, not migrations

3. **Leverage Rust ecosystem growth**:
   - As Rust multimedia libraries mature (rav1d, etc.)
   - Parallax becomes the natural integration point

**Bottom line**: Parallax can become the "native" pipeline framework for Rust, similar to how Tokio became the default async runtime. It won't replace GStreamer for legacy workloads, but it can be the better choice for new Rust applications.

---

## References

### GStreamer & Pipeline Frameworks
- [GStreamer Performance on Large Pipelines](https://gstconf.ubicast.tv/videos/gstreamer-performance-on-large-pipelines/)
- [Embedded GStreamer Performance Tuning](https://developer.ridgerun.com/wiki/index.php/Embedded_GStreamer_Performance_Tuning)
- [FFmpeg vs GStreamer Comparison](https://medium.com/@contact_45426/ffmpeg-vs-gstreamer-a-comprehensive-comparison-23217be772d3)
- [PipeWire Guide](https://github.com/mikeroyal/PipeWire-Guide)
- [GStreamer Caps Negotiation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/negotiation.html)
- [GStreamer Latency Optimization 2025](https://www.byteplus.com/en/topic/179639)

### Linux Memory & Zero-Copy
- [Linux DMA-BUF Documentation](https://docs.kernel.org/driver-api/dma-buf.html)
- [Real-Time Video Processing with V4L2/DRM/KMS](https://openlib.io/real-time-video-processing-pipelines-with-v4l2-drm-kms-and-hardware-encoders-in-linux/)
- [memfd_create Manual](https://man7.org/linux/man-pages/man2/memfd_create.2.html)
- [Intel Media Pipeline Memory Sharing](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-0/media-pipeline-inter-operation-and-memory-sharing.html)
- [GStreamer DMA-BUF Design](https://gstreamer.freedesktop.org/documentation/additional/design/dmabuf.html)

### Vulkan Video
- [Khronos Vulkan Video AV1 Decode](https://www.khronos.org/blog/khronos-releases-vulkan-video-av1-decode-extension-vulkan-sdk-now-supports-h.264-h.265-encode)
- [Vulkan Video Encode AV1 Extension](https://www.khronos.org/blog/khronos-announces-vulkan-video-encode-av1-encode-quantization-map-extensions)
- [Vulkan Video 2025 Status](https://dabrain34.github.io/jekyll/update/2025/03/11/Vulkankised2025.html)
- [Vulkanised 2025 - Igalia](https://blogs.igalia.com/scerveau/vulkanised-2025/)
- [Intel Vulkan AV1 Decode](https://www.phoronix.com/news/Intel-Vulkan-Video-AV1-Decode)

### Rust Ecosystem
- [stabby: Stable ABI for Rust](https://github.com/ZettaScaleLabs/stabby)
- [stabby Tutorial](https://docs.rs/stabby/latest/stabby/_tutorial_/index.html)
- [Plugins in Rust: Reducing the Pain](https://nullderef.com/blog/plugin-abi-stable/)
- [Rust Multimedia Libraries](https://lib.rs/multimedia)
- [rust-av Project](https://github.com/rust-av)
- [kornia-rs](https://arxiv.org/html/2505.12425v1)

### Pure Rust Codecs
- [rav1e: AV1 Encoder](https://github.com/xiph/rav1e)
- [rav1d: AV1 Decoder](https://github.com/memorysafety/rav1d)
