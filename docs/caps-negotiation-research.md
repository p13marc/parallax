# Plan: Caps Negotiation System for Parallax

## Overview

This plan adds format negotiation to Parallax, enabling automatic format compatibility checking, supporting both push and pull modes, and learning from the strengths and weaknesses of existing systems (GStreamer, FFmpeg, DirectShow, PipeWire).

---

## Analysis of Existing Systems

### GStreamer

**Strengths:**
- Mature, battle-tested negotiation system
- Supports both push (downstream) and pull (upstream) negotiation
- Rich caps structure with ranges, lists, and fractions
- Dynamic renegotiation via RECONFIGURE events

**Weaknesses ([documented issues](https://discourse.gstreamer.org/t/issues-with-caps-negotiation-delayed-linking-deinterleave-src-0-to-queue-sink/649)):**
- Complex mental model (push vs pull, upstream vs downstream)
- Pull mode requires knowing format before pulling (chicken-and-egg)
- Dynamic pipeline linking causes negotiation failures
- Caps meaning changes depending on pipeline state (confusing)
- Cryptic error messages when negotiation fails
- "Fixed caps" elements are inflexible

### FFmpeg libavfilter

**Strengths ([format negotiation docs](http://www.normalesup.org/~george/articles/format_negotiation_in_libavfilter/)):**
- Constraint-based system with iterative reduction
- Automatic insertion of conversion filters (scale, aresample)
- Global optimization across entire graph
- 187+ pixel formats handled without universal support

**Weaknesses:**
- `query_formats()` can be called multiple times (complex state)
- "Fragile" handling of filters with complex constraints (like `amerge`)
- Less flexible for runtime changes

### DirectShow (Windows)

**Strengths ([Microsoft docs](https://learn.microsoft.com/en-us/windows/win32/directshow/negotiating-media-types)):**
- Simple enumeration-based negotiation
- Clear AM_MEDIA_TYPE structure (major type, subtype, format block)
- Transform filters must connect input before output (clear ordering)

**Weaknesses:**
- Legacy system (superseded by Media Foundation)
- Less flexible than GStreamer's approach

### PipeWire

**Strengths ([PipeWire docs](https://docs.pipewire.org/)):**
- Session manager (WirePlumber) handles negotiation externally
- Parameters expose enumerations, manager writes chosen format
- DMA-BUF negotiation delegates modifier selection to allocator
- Clean separation of concerns

**Weaknesses:**
- Requires external session manager
- DMA-BUF negotiation is "complicated"

### Reactive Streams / Dataflow Programming

**Relevant concepts ([backpressure article](https://medium.com/@jayphelps/backpressure-explained-the-flow-of-data-through-software-2350b3e77ce7)):**
- Backpressure propagation through the graph
- Type inference via constraint solving ([research](https://arxiv.org/abs/2011.04876))
- Pull-based gives consumer control, push-based minimizes latency

---

## Design Improvements Over GStreamer

Based on research, Parallax can improve in several areas:

### 1. Unified Push/Pull Model with Clear Semantics

**Problem:** GStreamer's push vs pull confusion.

**Solution:** Single negotiation algorithm that works for both modes:

```rust
pub enum DataFlowMode {
    /// Source drives timing (live sources, generators)
    Push,
    /// Sink drives timing (file reading, seeking)
    Pull,
    /// Adaptive (can work either way)
    Either,
}
```

Negotiation is always the same; only data flow timing differs.

### 2. Constraint-Based Global Optimization (FFmpeg-inspired)

**Problem:** GStreamer negotiates link-by-link, missing global optima.

**Solution:** Treat negotiation as constraint satisfaction:

```rust
/// Negotiation as constraint solving
pub struct NegotiationSolver {
    constraints: Vec<FormatConstraint>,
}

impl NegotiationSolver {
    /// Solve all constraints simultaneously
    pub fn solve(&self) -> Result<NegotiatedFormats> {
        // 1. Collect all constraints from all elements
        // 2. Propagate constraints through graph (like type inference)
        // 3. Find minimal set of conversions needed
        // 4. Return globally optimal solution
    }
}
```

### 3. Better Error Messages (Type-Error-Style)

**Problem:** GStreamer errors are cryptic.

**Solution:** Inspired by [constraint-based type error research](https://dl.acm.org/doi/10.1145/3622812):

```rust
#[error("Format negotiation failed:\n{}", .explanation)]
NegotiationFailed {
    explanation: NegotiationErrorExplanation,
}

pub struct NegotiationErrorExplanation {
    /// The path through the graph where negotiation failed
    pub path: Vec<String>,
    /// What upstream produces
    pub upstream_format: Caps,
    /// What downstream accepts
    pub downstream_format: Caps,
    /// Suggested fixes (e.g., "insert a colorspace converter")
    pub suggestions: Vec<String>,
}
```

Example output:
```
Format negotiation failed:
  videotestsrc [produces: video/raw, format=RGB24, 1920x1080]
       ↓
  h264_encoder [accepts: video/raw, format={I420, NV12}, *x*]

  No common format found.
  
  Suggestions:
  - Insert a colorspace converter: videotestsrc ! colorconvert ! h264_encoder
  - Change videotestsrc output format to I420
```

### 4. Lazy Negotiation with Caching

**Problem:** GStreamer re-queries formats repeatedly.

**Solution:** Cache format information, invalidate on changes:

```rust
pub struct CachedCaps {
    caps: Caps,
    generation: u64,  // Incremented when element config changes
}

impl Element {
    fn output_caps_cached(&self) -> &CachedCaps {
        // Return cached caps if generation matches
        // Otherwise recompute and cache
    }
}
```

### 5. Explicit Conversion Registry (FFmpeg-inspired)

**Problem:** Ad-hoc converter insertion.

**Solution:** Formal registry of format converters:

```rust
pub struct ConverterRegistry {
    converters: HashMap<(MediaType, MediaType), ConverterFactory>,
}

impl ConverterRegistry {
    /// Find converter from src format to dst format
    pub fn find(&self, src: &MediaFormat, dst: &MediaFormat) -> Option<&ConverterFactory>;
    
    /// Find shortest conversion path (may need multiple converters)
    pub fn find_path(&self, src: &MediaFormat, dst: &MediaFormat) -> Option<Vec<ConverterFactory>>;
}
```

---

## Current State

### What Exists
- `Caps` struct with `intersects()` and `negotiate()` methods
- `MediaFormat` enum (VideoRaw, Video, AudioRaw, Audio, Rtp, MpegTs, Bytes)
- `VideoFormat`, `AudioFormat`, `RtpFormat` with detailed fields
- All element traits have `input_caps()` / `output_caps()` methods (default to `Caps::any()`)
- `Pad` and `PadTemplate` structures (minimal, no caps)
- `Link` struct for edges (no format info)

### What's Missing
- Pads don't carry format information
- No negotiation phase during pipeline startup
- No format validation between connected elements
- No push/pull mode distinction
- No mechanism for runtime format changes
- Elements don't know what format was negotiated
- No converter registry

---

## Design Principles

1. **Constraint-based negotiation** - Solve globally, not link-by-link
2. **Unified push/pull** - Same negotiation for both data flow modes
3. **Fail-fast with helpful errors** - Detect issues early, explain clearly
4. **Automatic conversion** - Insert converters when needed (opt-in)
5. **Preference ordering** - First format in Caps is preferred
6. **Backward compatible** - `Caps::any()` continues to work
7. **Cacheable** - Avoid redundant format queries
8. **Memory-aware** - Optimize memory placement to minimize copies

---

## Memory Placement Optimization

### The Problem: Unnecessary Memory Copies

A typical video pipeline might have:
```
Camera (DMA-BUF) → Decoder (GPU) → Filter (GPU) → Encoder (GPU) → Network (Heap)
```

Without memory negotiation, each element might:
1. Camera outputs DMA-BUF
2. Decoder copies DMA-BUF → GPU memory (necessary)
3. Filter copies GPU → Heap → GPU (unnecessary!)
4. Encoder copies GPU → Heap → GPU (unnecessary!)
5. Network copies GPU → Heap (necessary)

With memory negotiation, we can eliminate the unnecessary copies in steps 3 and 4.

### How GStreamer Handles This

GStreamer uses a separate [ALLOCATION query](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html) after caps negotiation:

1. After format negotiation, srcpad sends `GST_QUERY_ALLOCATION`
2. Downstream elements respond with supported allocators and buffer pools
3. Srcpad chooses allocation strategy based on responses

**Problems with GStreamer's approach:**
- Separate from caps negotiation (two-phase)
- Complex bufferpool renegotiation when config changes
- [DMA-BUF modifier negotiation is "complicated"](https://blogs.igalia.com/vjaquez/dmabuf-modifier-negotiation-in-gstreamer/)

### How PipeWire Handles This

PipeWire [handles format and buffer types independently](https://docs.pipewire.org/page_dma_buf.html):

- Format negotiated via `SPA_PARAM_Format`
- Buffer type via `SPA_PARAM_Buffers` with `dataType` bitmask
- Producer decides final allocation based on consumer capabilities

**Problems:**
- Format and buffer type are independent, can't express "I support RGB only as DMA-BUF"
- Producer must handle fallback logic

### Parallax Improvement: Unified Format + Memory Negotiation

**Key insight:** Memory type is part of the format constraint, not separate.

```rust
/// Extended caps that include memory requirements
#[derive(Debug, Clone)]
pub struct MediaCaps {
    /// Format constraints (resolution, pixel format, etc.)
    pub format: FormatCaps,
    
    /// Memory type constraints
    pub memory: MemoryCaps,
}

#[derive(Debug, Clone)]
pub struct MemoryCaps {
    /// Supported memory types (ordered by preference)
    pub types: CapsValue<MemoryType>,
    
    /// Can import from these memory types (for sinks/transforms)
    pub can_import: Vec<MemoryType>,
    
    /// Can export to these memory types (for sources/transforms)
    pub can_export: Vec<MemoryType>,
    
    /// DMA-BUF specific: supported DRM format modifiers
    pub drm_modifiers: Option<Vec<u64>>,
    
    /// Vulkan specific: external memory handle types
    pub vk_external_memory: Option<VkExternalMemoryHandleTypes>,
}

/// Memory types (simplified: no separate Heap vs SharedMemory)
/// 
/// Design decision: All CPU memory uses memfd_create + mmap.
/// This has zero overhead vs malloc but is always shareable via fd.
/// There's no reason to have non-shareable heap memory for buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MemoryType {
    /// CPU-accessible memory (memfd-backed, always IPC-ready)
    /// - Works like heap for single-process (zero overhead)
    /// - Can be shared via fd passing when needed
    /// - This replaces both "Heap" and "SharedMemory"
    Cpu,
    /// DMA-BUF (Linux kernel buffer sharing, GPU-importable)
    DmaBuf,
    /// GPU device memory (Vulkan/CUDA)
    GpuDevice,
    /// GPU-accessible host memory (pinned for fast transfers)
    GpuAccessible,
    /// RDMA-registered memory (for network zero-copy)
    RdmaRegistered,
    /// Huge pages (for large buffers, TLB efficiency)
    HugePages,
    /// Memory-mapped file (persistent storage)
    MappedFile,
}

impl MemoryType {
    /// Can this memory type be shared across processes (same machine)?
    pub fn supports_ipc(&self) -> bool {
        match self {
            MemoryType::Cpu => true,           // memfd, always has fd
            MemoryType::DmaBuf => true,        // fd-based
            MemoryType::MappedFile => true,    // fd-based
            MemoryType::HugePages => true,     // memfd with MFD_HUGETLB
            MemoryType::GpuAccessible => true, // can be memfd-backed
            MemoryType::GpuDevice => false,    // must export to DmaBuf first
            MemoryType::RdmaRegistered => true,// special network handling
        }
    }
    
    /// Can this memory type be sent over network?
    pub fn supports_network(&self) -> bool {
        match self {
            MemoryType::Cpu => true,           // serialize + send
            MemoryType::RdmaRegistered => true,// RDMA write
            MemoryType::HugePages => true,     // serialize + send
            MemoryType::MappedFile => true,    // serialize + send
            MemoryType::DmaBuf => false,       // fd is local
            MemoryType::GpuDevice => false,    // must download first
            MemoryType::GpuAccessible => true, // can serialize
        }
    }
}
```

### Global Memory Optimization Algorithm

The negotiation solver can now optimize memory placement globally:

```rust
impl NegotiationSolver {
    pub fn solve(&self) -> Result<NegotiationResult> {
        // 1. Collect format constraints
        let format_constraints = self.collect_format_constraints()?;
        
        // 2. Collect memory constraints
        let memory_constraints = self.collect_memory_constraints()?;
        
        // 3. Build memory domain graph
        let memory_graph = self.build_memory_graph(&memory_constraints)?;
        
        // 4. Find optimal memory placement (minimize copies)
        let memory_placement = self.optimize_memory_placement(&memory_graph)?;
        
        // 5. Fixate formats considering memory constraints
        let formats = self.fixate_with_memory(&format_constraints, &memory_placement)?;
        
        Ok(NegotiationResult {
            formats,
            memory_placement,
            converters: vec![],
        })
    }
    
    /// Build graph of memory domains and transition costs
    fn build_memory_graph(&self, constraints: &[MemoryConstraint]) -> MemoryGraph {
        // Nodes: memory types
        // Edges: transitions with costs
        //
        // Within same process:
        //   Cpu ↔ Cpu: 0 (same domain, memfd is transparent)
        //   GPU ↔ GPU: 0 (same domain)
        //   Cpu → GPU: 10 (upload to device)
        //   GPU → Cpu: 10 (download from device)
        //   DMA-BUF → GPU: 1 (import, near zero-copy)
        //   GPU → DMA-BUF: 1 (export, near zero-copy)
        //   Cpu ↔ DMA-BUF: 2 (mmap the dmabuf)
        //
        // Cross-process (same machine):
        //   Cpu → Cpu: 1 (fd passing + mmap, zero-copy!)
        //   DMA-BUF → DMA-BUF: 1 (fd passing)
        //   DMA-BUF → GPU: 2 (fd pass + import)
        //
        // Cross-machine: see network section
    }
    
    /// Find minimum-cost memory placement using shortest paths
    fn optimize_memory_placement(&self, graph: &MemoryGraph) -> Result<MemoryPlacement> {
        // For each subgraph of connected elements:
        // 1. Identify "anchor" elements with fixed memory requirements
        //    (e.g., GPU decoder must output GPU memory)
        // 2. Propagate memory domains from anchors
        // 3. Insert memory transfer elements where domains change
        // 4. Prefer longer runs in same memory domain
    }
}
```

### Memory Placement Result

```rust
pub struct MemoryPlacement {
    /// Memory type for each link
    pub link_memory: HashMap<LinkId, MemoryType>,
    
    /// Memory transfers to insert
    pub transfers: Vec<MemoryTransfer>,
    
    /// Suggested buffer pool configuration
    pub pool_config: HashMap<NodeId, BufferPoolConfig>,
}

pub struct MemoryTransfer {
    pub link_id: LinkId,
    pub from_type: MemoryType,
    pub to_type: MemoryType,
    pub element: Box<dyn ElementDyn>,  // e.g., GpuUpload, GpuDownload
}

pub struct BufferPoolConfig {
    pub memory_type: MemoryType,
    pub buffer_count: usize,
    pub buffer_size: usize,
    /// For DMA-BUF: allocator to use
    pub allocator: Option<Box<dyn BufferAllocator>>,
}
```

### Element Memory Capabilities

Elements declare their memory capabilities:

```rust
pub trait Element: Send {
    // ... existing methods ...
    
    /// Memory capabilities for input
    fn input_memory_caps(&self) -> MemoryCaps {
        MemoryCaps::cpu_only()  // Default: accepts CPU memory (memfd-backed)
    }
    
    /// Memory capabilities for output
    fn output_memory_caps(&self) -> MemoryCaps {
        MemoryCaps::cpu_only()  // Default: produces CPU memory (memfd-backed)
    }
}

// Example: Vulkan decoder
impl Element for VulkanH264Decoder {
    fn input_memory_caps(&self) -> MemoryCaps {
        MemoryCaps {
            types: CapsValue::List(vec![MemoryType::Cpu]),  // Accepts CPU memory
            can_import: vec![MemoryType::Cpu],
            can_export: vec![],
            drm_modifiers: None,
            vk_external_memory: None,
        }
    }
    
    fn output_memory_caps(&self) -> MemoryCaps {
        MemoryCaps {
            types: CapsValue::List(vec![
                MemoryType::GpuDevice,  // Preferred: keep on GPU
                MemoryType::DmaBuf,     // Can export as DMA-BUF
                MemoryType::Cpu,        // Can download to CPU
            ]),
            can_import: vec![],
            can_export: vec![MemoryType::DmaBuf, MemoryType::Cpu],
            drm_modifiers: Some(self.supported_modifiers()),
            vk_external_memory: Some(VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT),
        }
    }
}

// Example: Iced video sink (uses wgpu)
impl Sink for IcedVideoSink {
    fn input_memory_caps(&self) -> MemoryCaps {
        MemoryCaps {
            types: CapsValue::List(vec![
                MemoryType::GpuDevice,  // Preferred: already on GPU
                MemoryType::DmaBuf,     // Can import DMA-BUF
                MemoryType::Cpu,        // Can upload from CPU
            ]),
            can_import: vec![MemoryType::DmaBuf, MemoryType::GpuDevice, MemoryType::Cpu],
            can_export: vec![],
            drm_modifiers: Some(self.supported_modifiers()),
            vk_external_memory: Some(VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT),
        }
    }
}
```

### Zero-Copy Pipeline Example

```rust
// This pipeline can be fully zero-copy on GPU:
let pipeline = Pipeline::parse("vulkan_h264_dec ! vulkan_filter ! iced_video_sink")?;

// Negotiation result:
// - vulkan_h264_dec outputs: GpuDevice
// - vulkan_filter input/output: GpuDevice (no copy!)
// - iced_video_sink input: GpuDevice (no copy!)
// 
// Total copies: 0 (all on GPU)
```

```rust
// This pipeline requires one transfer:
let pipeline = Pipeline::parse("vulkan_h264_dec ! tcp_sink")?;

// Negotiation result:
// - vulkan_h264_dec outputs: GpuDevice
// - AUTO-INSERTED: gpu_download (GpuDevice → Heap)
// - tcp_sink input: Heap
//
// Total copies: 1 (necessary)
```

### Memory Transfer Elements

Built-in elements for memory domain transitions:

```rust
/// Upload data from CPU to GPU
pub struct GpuUpload {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

/// Download data from GPU to CPU
pub struct GpuDownload {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
}

/// Import DMA-BUF into GPU memory
pub struct DmaBufImport {
    device: Arc<wgpu::Device>,
}

/// Export GPU memory as DMA-BUF
pub struct DmaBufExport {
    device: Arc<wgpu::Device>,
}
```

### DMA-BUF Modifier Negotiation

For DMA-BUF, modifiers must be negotiated (PipeWire-style):

```rust
impl NegotiationSolver {
    fn negotiate_drm_modifiers(&self, 
        upstream: &MemoryCaps, 
        downstream: &MemoryCaps
    ) -> Option<u64> {
        let upstream_mods = upstream.drm_modifiers.as_ref()?;
        let downstream_mods = downstream.drm_modifiers.as_ref()?;
        
        // Find common modifiers
        let common: Vec<u64> = upstream_mods.iter()
            .filter(|m| downstream_mods.contains(m))
            .copied()
            .collect();
        
        if common.is_empty() {
            return None;
        }
        
        // Prefer optimal modifiers (tiled > linear)
        // LINEAR (0) is always last resort
        common.into_iter()
            .filter(|&m| m != DRM_FORMAT_MOD_LINEAR)
            .next()
            .or(Some(DRM_FORMAT_MOD_LINEAR))
    }
}
```

### Cross-Process Memory Negotiation

When elements run in different processes, memory negotiation has additional constraints:

```
Process A: [camera_src] ──IPC──> Process B: [encoder] ──IPC──> Process C: [streamer]
```

**The fundamental constraint:** Heap memory cannot cross process boundaries.

#### Process Boundary Detection

The pipeline must know where process boundaries exist:

```rust
/// Describes where an element runs
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProcessLocation {
    /// Same process as pipeline controller
    Local,
    /// Remote process, connected via IPC
    Remote {
        process_id: ProcessId,
        /// IPC channel for this process
        channel: IpcChannelId,
    },
}

/// Extended link information
pub struct Link {
    pub from_pad: String,
    pub to_pad: String,
    pub negotiated_format: Option<MediaFormat>,
    pub flow_mode: DataFlowMode,
    
    /// Does this link cross a process boundary?
    pub crosses_process: bool,
    
    /// IPC transport for cross-process links
    pub ipc_transport: Option<IpcTransport>,
}

/// How buffers are transported across process boundaries
pub enum IpcTransport {
    /// File descriptor passing via Unix socket (SCM_RIGHTS)
    /// Works for: SharedMemory (memfd), DMA-BUF, MappedFile
    FdPassing {
        socket: UnixSocket,
    },
    /// Zenoh-based transport (serializes data)
    /// Works for: Any memory type (with copy)
    Zenoh {
        session: Arc<zenoh::Session>,
        key_expr: String,
    },
    /// Custom transport
    Custom(Box<dyn IpcTransportImpl>),
}
```

#### Memory Type Validity by Context

```rust
impl MemoryType {
    /// Can this memory type be used for cross-process communication?
    pub fn supports_ipc(&self) -> bool {
        match self {
            MemoryType::Heap => false,           // Cannot cross process
            MemoryType::SharedMemory => true,    // memfd + fd passing
            MemoryType::DmaBuf => true,          // fd passing
            MemoryType::GpuDevice => false,      // Must export first
            MemoryType::GpuAccessible => false,  // Process-local pinned memory
            MemoryType::HugePages => true,       // Can be shared via memfd
            MemoryType::MappedFile => true,      // fd passing
        }
    }
    
    /// Can export to an IPC-capable type?
    pub fn can_export_for_ipc(&self) -> Option<MemoryType> {
        match self {
            MemoryType::GpuDevice => Some(MemoryType::DmaBuf),  // Export as DMA-BUF
            MemoryType::Heap => Some(MemoryType::SharedMemory), // Copy to shmem
            _ if self.supports_ipc() => Some(*self),
            _ => None,
        }
    }
}
```

#### Cross-Process Negotiation Algorithm

```rust
impl NegotiationSolver {
    fn solve_with_process_boundaries(&self) -> Result<NegotiationResult> {
        // 1. Identify process boundaries
        let boundaries = self.find_process_boundaries()?;
        
        // 2. For each boundary, constrain memory types
        for boundary in &boundaries {
            self.add_ipc_constraint(boundary)?;
        }
        
        // 3. Run normal negotiation with IPC constraints
        let result = self.solve_constrained()?;
        
        // 4. Configure IPC transports
        self.configure_ipc_transports(&result, &boundaries)?;
        
        Ok(result)
    }
    
    fn add_ipc_constraint(&mut self, boundary: &ProcessBoundary) -> Result<()> {
        let link = &boundary.link;
        
        // Get upstream's output memory caps
        let upstream_caps = self.get_output_memory_caps(link.from_element)?;
        
        // Filter to IPC-capable types only
        let ipc_capable: Vec<MemoryType> = upstream_caps.types
            .iter()
            .filter(|t| t.supports_ipc() || t.can_export_for_ipc().is_some())
            .copied()
            .collect();
        
        if ipc_capable.is_empty() {
            return Err(NegotiationError::NoIpcCapableMemory {
                element: link.from_element.clone(),
                available: upstream_caps.types.clone(),
            });
        }
        
        // Add constraint: this link MUST use IPC-capable memory
        self.constraints.push(MemoryConstraint::IpcRequired {
            link_id: link.id,
            allowed_types: ipc_capable,
        });
        
        Ok(())
    }
}
```

#### Memory Type Selection for IPC

```rust
/// Choose optimal memory type for cross-process link
fn choose_ipc_memory_type(
    upstream: &MemoryCaps,
    downstream: &MemoryCaps,
    transport: &IpcTransport,
) -> Result<(MemoryType, Option<MemoryTransfer>)> {
    // Priority order for IPC:
    // 1. DMA-BUF (zero-copy, GPU-compatible)
    // 2. SharedMemory (zero-copy, CPU-only)
    // 3. MappedFile (zero-copy, persistent)
    // 4. Heap → SharedMemory (requires copy)
    
    let upstream_ipc: Vec<MemoryType> = upstream.types.iter()
        .filter(|t| t.supports_ipc())
        .copied()
        .collect();
    
    let downstream_ipc: Vec<MemoryType> = downstream.can_import.iter()
        .filter(|t| t.supports_ipc())
        .copied()
        .collect();
    
    // Find common IPC-capable types
    for preferred in [MemoryType::DmaBuf, MemoryType::SharedMemory, MemoryType::MappedFile] {
        if upstream_ipc.contains(&preferred) && downstream_ipc.contains(&preferred) {
            return Ok((preferred, None));
        }
    }
    
    // No direct match - need transfer element
    
    // Case: Upstream is GPU, downstream wants SharedMemory
    if upstream.types.contains(&MemoryType::GpuDevice) 
        && downstream_ipc.contains(&MemoryType::DmaBuf) 
    {
        // GPU can export as DMA-BUF
        return Ok((MemoryType::DmaBuf, Some(MemoryTransfer::GpuToDmaBuf)));
    }
    
    // Case: Upstream is GPU, downstream only supports SharedMemory  
    if upstream.types.contains(&MemoryType::GpuDevice)
        && downstream_ipc.contains(&MemoryType::SharedMemory)
    {
        // Must download from GPU to shared memory
        return Ok((MemoryType::SharedMemory, Some(MemoryTransfer::GpuToSharedMemory)));
    }
    
    // Case: Upstream is Heap (shouldn't happen after constraints, but fallback)
    if upstream.types.contains(&MemoryType::Heap) {
        // Copy heap to shared memory
        return Ok((MemoryType::SharedMemory, Some(MemoryTransfer::HeapToSharedMemory)));
    }
    
    Err(NegotiationError::NoCommonIpcMemory {
        upstream: upstream.clone(),
        downstream: downstream.clone(),
    })
}
```

#### IPC Buffer Protocol

For cross-process links, the buffer itself contains an `IpcHandle`:

```rust
/// Handle that can be sent across process boundaries
#[derive(Debug, Clone)]
pub enum IpcHandle {
    /// File descriptor (for SharedMemory, DMA-BUF, MappedFile)
    Fd {
        /// The fd to pass (will be different number in receiving process)
        fd: RawFd,
        /// Size of the memory region
        size: usize,
        /// Type hint for receiver
        memory_type: MemoryType,
        /// For DMA-BUF: DRM format modifier
        drm_modifier: Option<u64>,
    },
    /// Serialized data (fallback, involves copy)
    Serialized {
        data: Vec<u8>,
    },
}

/// Message sent over IPC for each buffer
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct IpcBufferMessage {
    /// Buffer metadata (timestamp, flags, etc.)
    pub metadata: Metadata,
    
    /// Memory handle type
    pub handle_type: IpcHandleType,
    
    /// For Fd handles: size and type info (fd sent via SCM_RIGHTS)
    pub fd_info: Option<FdInfo>,
    
    /// For serialized: inline data
    pub inline_data: Option<Vec<u8>>,
}

#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct FdInfo {
    pub size: usize,
    pub memory_type: MemoryType,
    pub drm_modifier: Option<u64>,
}
```

#### Receiving Process Reconstruction

```rust
impl Buffer {
    /// Reconstruct buffer from IPC message (receiving side)
    pub fn from_ipc(msg: IpcBufferMessage, received_fd: Option<RawFd>) -> Result<Self> {
        let memory = match msg.handle_type {
            IpcHandleType::Fd => {
                let fd = received_fd.ok_or(Error::MissingFd)?;
                let info = msg.fd_info.ok_or(Error::MissingFdInfo)?;
                
                match info.memory_type {
                    MemoryType::SharedMemory => {
                        // mmap the received fd
                        MemoryHandle::SharedMemory(SharedMemorySegment::from_fd(fd, info.size)?)
                    }
                    MemoryType::DmaBuf => {
                        // Import DMA-BUF
                        MemoryHandle::DmaBuf(DmaBufSegment::from_fd(fd, info.size, info.drm_modifier)?)
                    }
                    MemoryType::MappedFile => {
                        MemoryHandle::MappedFile(MappedFileSegment::from_fd(fd, info.size)?)
                    }
                    _ => return Err(Error::UnsupportedIpcMemoryType(info.memory_type)),
                }
            }
            IpcHandleType::Serialized => {
                let data = msg.inline_data.ok_or(Error::MissingInlineData)?;
                MemoryHandle::Heap(HeapSegment::from_vec(data))
            }
        };
        
        Ok(Buffer {
            memory,
            metadata: msg.metadata,
            ..Default::default()
        })
    }
}
```

#### Example: Multi-Process Pipeline

```rust
// Process A: Camera capture
let camera_pipeline = Pipeline::new()
    .add_source("cam", V4l2Src::new("/dev/video0"))  // Outputs DMA-BUF
    .add_sink("ipc_out", IpcSink::new("unix:///tmp/camera.sock"))
    .link("cam", "ipc_out")?;

// Process B: Encoding (separate process for isolation)
let encoder_pipeline = Pipeline::new()
    .add_source("ipc_in", IpcSrc::new("unix:///tmp/camera.sock"))
    .add_element("enc", VulkanH264Encoder::new())  // Imports DMA-BUF directly to GPU
    .add_sink("ipc_out", IpcSink::new("unix:///tmp/encoded.sock"))
    .link_all(&["ipc_in", "enc", "ipc_out"])?;

// Negotiation result:
// - cam → ipc_out: DMA-BUF (v4l2 native)
// - ipc_in → enc: DMA-BUF (zero-copy import to GPU)
// - enc → ipc_out: SharedMemory (encoded bitstream, no need for DMA-BUF)
```

#### Cost Model Update for IPC

```rust
fn build_memory_graph(&self, constraints: &[MemoryConstraint]) -> MemoryGraph {
    // Within same process:
    //   Cpu ↔ Cpu: 0 (memfd is transparent, no overhead)
    //   GPU ↔ GPU: 0
    //   Cpu → GPU: 10 (upload)
    //   GPU → Cpu: 10 (download)
    //   DMA-BUF → GPU: 1 (import)
    
    // Cross-process LOCAL (same machine):
    //   Cpu → Cpu: 1 (fd passing + mmap, TRUE ZERO-COPY)
    //   DMA-BUF → DMA-BUF: 1 (fd passing, zero-copy)
    //   DMA-BUF → GPU (other process): 2 (fd passing + import)
    //   GPU → DMA-BUF → GPU (other process): 3 (export + fd + import)
    
    // Note: No "Heap" anymore - all CPU memory is memfd-backed
    // so cross-process is always zero-copy for CPU memory!
    
    // INVALID cross-process:
    //   GPU → GPU: INFINITY (must go through DMA-BUF)
}
```

#### File Descriptor Limits and Arena Allocation

Fd limits can be a concern (default 1024 per process). Solution: **Arena allocation** - one fd per pool, not per buffer.

```rust
/// Single memfd arena for many buffers (1 fd for entire pool)
pub struct CpuArena {
    fd: OwnedFd,           // ONE fd for entire arena
    base: *mut u8,         // mmap'd base pointer
    total_size: usize,     // e.g., 256MB
    slot_size: usize,      // Per-buffer size
    free_slots: AtomicBitmap,
}

/// A slot within the arena (no additional fd!)
pub struct ArenaSlot {
    arena: Arc<CpuArena>,
    offset: usize,
    len: usize,
}

impl ArenaSlot {
    pub fn as_slice(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.arena.base.add(self.offset),
                self.len
            )
        }
    }
}

/// For cross-process: send arena fd once, then just offsets
#[derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)]
pub struct IpcSlotRef {
    pub arena_id: u64,     // Identifies which arena (receiver caches mmap)
    pub offset: usize,
    pub len: usize,
}
```

Cross-process flow:
1. First buffer: send arena fd + slot offset → receiver mmaps arena once
2. Subsequent buffers: just send offset (no fd!) → receiver uses cached mmap

This reduces fd usage from `O(buffers)` to `O(arenas)` ≈ `O(pipelines)`.

---

#### Unified CPU Memory (memfd-backed)

All CPU memory is backed by `memfd_create`, making it **always shareable**:

```rust
/// CPU memory segment - always IPC-ready via fd
/// This replaces both "Heap" and "SharedMemory" - there's no distinction.
pub struct CpuSegment {
    /// The memfd file descriptor (always present)
    fd: OwnedFd,
    /// mmap'd pointer for local access
    ptr: *mut u8,
    /// Size of the region
    size: usize,
}

impl CpuSegment {
    /// Allocate new CPU memory (works like malloc, but shareable)
    pub fn new(size: usize) -> Result<Self> {
        let fd = memfd_create(
            CStr::from_bytes_with_nul(b"parallax\0").unwrap(),
            MemfdFlags::CLOEXEC,
        )?;
        ftruncate(&fd, size as i64)?;
        
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,  // Always shared - no overhead, enables IPC
                fd.as_raw_fd(),
                0,
            )?
        };
        
        Ok(Self { fd, ptr: ptr.cast(), size })
    }
    
    /// Reconstruct from received fd (cross-process)
    pub fn from_fd(fd: OwnedFd, size: usize) -> Result<Self> {
        // mmap the same region - TRUE ZERO COPY!
        let ptr = unsafe {
            mmap(
                std::ptr::null_mut(),
                size,
                PROT_READ | PROT_WRITE,
                MAP_SHARED,  // Same physical pages as sender
                fd.as_raw_fd(),
                0,
            )?
        };
        
        Ok(Self { fd, ptr: ptr.cast(), size })
    }
    
    /// Get fd for IPC (always available - no conversion!)
    pub fn fd(&self) -> BorrowedFd<'_> {
        self.fd.as_fd()
    }
    
    /// Use like regular memory
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.size) }
    }
    
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.size) }
    }
}
```

**Key insight:** `memfd_create` + `MAP_SHARED` has **zero overhead** vs `malloc`, but gives us IPC for free.

#### Copy-on-Write for Mutations

If an element needs to modify a buffer that's shared cross-process:

```rust
impl Buffer {
    /// Get mutable access, CoW if shared across processes
    pub fn make_mut(&mut self) -> Result<&mut [u8]> {
        match &mut self.memory {
            MemoryHandle::Cpu(segment) => {
                if self.is_shared_cross_process() {
                    // CoW: copy to new segment, then mutate
                    let new_segment = CpuSegment::new(segment.len())?;
                    new_segment.as_mut_slice().copy_from_slice(segment.as_slice());
                    self.memory = MemoryHandle::Cpu(new_segment);
                }
                Ok(self.memory.as_cpu_mut().as_mut_slice())
            }
            // GPU memory: download first, then mutate
            MemoryHandle::GpuDevice(_) => {
                Err(Error::CannotMutateGpuDirectly)
            }
            // ...
        }
    }
}
```

#### Buffer Pool (Zero Allocation in Steady State)

For high-throughput pipelines, pre-allocate a pool:

```rust
/// Pool of CPU buffers for zero-copy pipelines
pub struct CpuBufferPool {
    segments: Vec<CpuSegment>,
    free_list: crossbeam::queue::ArrayQueue<usize>,
}

impl CpuBufferPool {
    pub fn new(buffer_count: usize, buffer_size: usize) -> Result<Self> {
        let segments: Vec<_> = (0..buffer_count)
            .map(|_| CpuSegment::new(buffer_size))
            .collect::<Result<_>>()?;
        
        let free_list = ArrayQueue::new(buffer_count);
        for i in 0..buffer_count {
            free_list.push(i).unwrap();
        }
        
        Ok(Self { segments, free_list })
    }
    
    /// Acquire a buffer (backpressure if none available)
    pub fn acquire(&self) -> LoanedBuffer { /* ... */ }
}
```

Flow:
1. Producer acquires from pool → writes data
2. Sends fd to consumer (zero-copy, same physical pages)
3. Consumer processes → releases back to pool
4. No allocations in steady state

### Cross-Machine Memory Negotiation (Network)

When elements are on **different machines**, fd passing and shared memory don't work. The negotiation must handle this differently.

```
Machine A: [camera] ──Network──> Machine B: [encoder] ──Network──> Machine C: [display]
```

#### Machine Boundary Detection

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ProcessLocation {
    /// Same process as pipeline controller
    Local,
    /// Different process, same machine (IPC possible)
    RemoteProcess {
        process_id: ProcessId,
        channel: IpcChannelId,
    },
    /// Different machine entirely (network only)
    RemoteMachine {
        machine_id: MachineId,
        /// Network transport endpoint
        endpoint: NetworkEndpoint,
    },
}

#[derive(Debug, Clone)]
pub enum NetworkEndpoint {
    /// Zenoh session (pub/sub, query)
    Zenoh {
        session: Arc<zenoh::Session>,
        key_expr: String,
    },
    /// Direct TCP/UDP
    Socket {
        addr: SocketAddr,
        protocol: TransportProtocol,
    },
    /// RDMA (InfiniBand, RoCE)
    Rdma {
        device: RdmaDevice,
        qp: QueuePair,
    },
}
```

#### Memory Types for Network Transport

```rust
// Note: supports_network() is already defined in MemoryType above.
// Here's the network transport method:

impl MemoryType {
    /// Network transport method for this memory type
    pub fn network_transport_method(&self) -> NetworkMethod {
        match self {
            MemoryType::RdmaRegistered => NetworkMethod::RdmaWrite,  // True zero-copy
            _ => NetworkMethod::Serialize,  // Must copy to network buffer
        }
    }
}

pub enum NetworkMethod {
    /// Serialize data and send (involves copy)
    Serialize,
    /// RDMA write directly to remote memory (zero-copy)
    RdmaWrite,
}
```

#### Cross-Machine Negotiation Algorithm

```rust
impl NegotiationSolver {
    fn add_network_constraint(&mut self, boundary: &MachineBoundary) -> Result<()> {
        let link = &boundary.link;
        let endpoint = &boundary.endpoint;
        
        // Check if RDMA is available
        let rdma_available = matches!(endpoint, NetworkEndpoint::Rdma { .. });
        
        if rdma_available {
            // Prefer RDMA for zero-copy
            self.constraints.push(MemoryConstraint::NetworkLink {
                link_id: link.id,
                preferred: vec![MemoryType::RdmaRegistered],
                fallback: vec![MemoryType::Cpu],  // Cpu memory works too
                method: NetworkMethod::RdmaWrite,
            });
        } else {
            // Standard network - must serialize
            // Cpu memory is simplest (already shareable, easy to serialize)
            self.constraints.push(MemoryConstraint::NetworkLink {
                link_id: link.id,
                preferred: vec![MemoryType::Cpu],
                fallback: vec![MemoryType::GpuAccessible],
                method: NetworkMethod::Serialize,
            });
        }
        
        Ok(())
    }
}
```

#### Cost Model for Network

```rust
fn build_memory_graph(&self, constraints: &[MemoryConstraint]) -> MemoryGraph {
    // ... existing costs (see above) ...
    
    // Cross-machine (network):
    //   Cpu → Network → Cpu: 50 (serialize + network latency + deserialize)
    //   GPU → Cpu → Network → Cpu: 60 (download + serialize + network)
    //   GPU → Cpu → Network → Cpu → GPU: 70 (download + network + upload)
    //   
    // Cross-machine with RDMA:
    //   RdmaRegistered → RDMA → RdmaRegistered: 5 (near zero-copy, just latency)
    //   Cpu → RdmaRegistered → RDMA → RdmaRegistered: 15 (register + rdma)
    //   GPU → RdmaRegistered (GPUDirect): 8 (if GPUDirect RDMA available)
    //
    // INVALID cross-machine:
    //   DmaBuf → DmaBuf: INFINITY (fd is local to machine)
    //   Cpu (fd) → Cpu (fd): INFINITY (fd passing doesn't work cross-machine)
    //   
    // Note: Cross-machine ALWAYS requires serialization or RDMA.
    // The unified Cpu type (memfd) doesn't help here - fd is local.
}
```

#### RDMA Support

For high-performance cross-machine pipelines, RDMA provides near zero-copy:

```rust
/// RDMA-registered memory segment
pub struct RdmaSegment {
    /// Local memory buffer
    buffer: Vec<u8>,
    /// RDMA memory region handle
    mr: ibverbs::MemoryRegion,
    /// Remote key (for remote access)
    rkey: u32,
    /// Local key
    lkey: u32,
}

impl RdmaSegment {
    pub fn new(size: usize, pd: &ibverbs::ProtectionDomain) -> Result<Self> {
        let buffer = vec![0u8; size];
        let mr = pd.reg_mr(&buffer, ibverbs::AccessFlags::all())?;
        Ok(Self {
            buffer,
            rkey: mr.rkey(),
            lkey: mr.lkey(),
            mr,
        })
    }
}

/// RDMA buffer transfer (sender side)
impl RdmaTransport {
    pub fn send_buffer(&self, buffer: &Buffer, remote: &RdmaRemoteInfo) -> Result<()> {
        match buffer.memory_type() {
            MemoryType::RdmaRegistered => {
                // Direct RDMA write - no CPU involvement
                self.qp.post_send_write(
                    buffer.as_rdma_segment()?,
                    remote.addr,
                    remote.rkey,
                )?;
            }
            _ => {
                // Copy to registered buffer, then RDMA write
                let staging = self.staging_pool.acquire()?;
                staging.copy_from(buffer.as_slice());
                self.qp.post_send_write(&staging, remote.addr, remote.rkey)?;
            }
        }
        Ok(())
    }
}
```

#### GPUDirect RDMA (Future)

For GPU→GPU across machines without CPU involvement:

```rust
/// GPUDirect RDMA - GPU memory directly accessible via RDMA
pub struct GpuDirectSegment {
    /// CUDA/Vulkan device memory
    gpu_buffer: GpuBuffer,
    /// RDMA memory region registered on GPU memory
    mr: ibverbs::MemoryRegion,  // Registered via nvidia_peermem or similar
}

// Cost model addition:
//   GPU (Machine A) → GPUDirect RDMA → GPU (Machine B): 10
//   (vs GPU → CPU → Network → CPU → GPU: 70)
```

#### Zenoh Integration for Network Transport

Zenoh handles the network complexity:

```rust
/// Network transport via Zenoh
pub struct ZenohTransport {
    session: Arc<zenoh::Session>,
    key_expr: KeyExpr<'static>,
}

impl ZenohTransport {
    pub async fn send_buffer(&self, buffer: &Buffer) -> Result<()> {
        // Zenoh handles serialization and routing
        let payload = buffer.serialize_for_network()?;
        self.session.put(&self.key_expr, payload).await?;
        Ok(())
    }
    
    pub async fn recv_buffer(&self) -> Result<Buffer> {
        let sample = self.subscriber.recv_async().await?;
        Buffer::deserialize_from_network(sample.payload())
    }
}

impl Buffer {
    fn serialize_for_network(&self) -> Result<ZBytes> {
        // Use rkyv for zero-copy serialization where possible
        let bytes = rkyv::to_bytes::<_, 256>(self)?;
        Ok(ZBytes::from(bytes.as_slice()))
    }
    
    fn deserialize_from_network(payload: &ZBytes) -> Result<Self> {
        // Zero-copy deserialization with rkyv
        let archived = rkyv::check_archived_root::<BufferData>(payload.as_slice())?;
        Ok(Self::from_archived(archived))
    }
}
```

#### Negotiation Decision Tree

```
Is link cross-process?
├─ No → Use any memory type (Heap, GPU, etc.)
└─ Yes → Is link cross-machine?
    ├─ No (same machine) → Must use IPC-capable memory
    │   ├─ DMA-BUF (preferred for GPU pipelines)
    │   ├─ SharedMemory (CPU pipelines)
    │   └─ Insert transfer element if needed
    └─ Yes (different machines) → Must serialize OR use RDMA
        ├─ RDMA available?
        │   ├─ Yes → Use RdmaRegistered (near zero-copy)
        │   │   └─ GPUDirect available? → GPU→RDMA→GPU
        │   └─ No → Serialize (Heap preferred, avoid indirection)
        └─ Insert download element if source is GPU
```

#### Example: Distributed Pipeline

```rust
// Machine A: Camera capture
let camera_pipeline = Pipeline::new()
    .add_source("cam", V4l2Src::new("/dev/video0"))
    .add_sink("net", ZenohSink::new("pipeline/video/raw"))
    .link("cam", "net")?;

// Machine B: GPU encoding (different machine)
let encoder_pipeline = Pipeline::new()
    .add_source("net", ZenohSrc::new("pipeline/video/raw"))
    .add_element("enc", VulkanH264Encoder::new())
    .add_sink("out", ZenohSink::new("pipeline/video/h264"))
    .link_all(&["net", "enc", "out"])?;

// Negotiation on Machine A:
// - cam outputs DMA-BUF (v4l2 native)
// - net requires serializable → AUTO-INSERT: DmaBufToHeap
// - Serialized via Zenoh

// Negotiation on Machine B:
// - net outputs Heap (deserialized)
// - enc prefers GPU → AUTO-INSERT: GpuUpload
// - enc outputs GPU → out requires serializable → AUTO-INSERT: GpuDownload
```

#### High-Performance Variant with RDMA

```rust
// Machine A: With RDMA NIC
let camera_pipeline = Pipeline::new()
    .add_source("cam", V4l2Src::new("/dev/video0"))
    .add_sink("rdma", RdmaSink::new("ib0", remote_addr))
    .link("cam", "rdma")?;

// Machine B: With RDMA NIC + GPUDirect
let encoder_pipeline = Pipeline::new()
    .add_source("rdma", RdmaSrc::new("ib0").with_gpu_direct(true))
    .add_element("enc", VulkanH264Encoder::new())
    .link("rdma", "enc")?;

// Negotiation:
// - cam outputs DMA-BUF
// - rdma_sink can accept DMA-BUF (RDMA registered via dmabuf)
// - RDMA write to Machine B's GPU memory directly (GPUDirect)
// - enc receives in GPU memory - no CPU copies anywhere!
```

---

#### Error Messages for IPC Issues

```rust
pub enum NegotiationError {
    // ... existing variants ...
    
    /// Element cannot produce IPC-capable memory
    NoIpcCapableMemory {
        element: String,
        available: Vec<MemoryType>,
        suggestion: String,  // e.g., "Add GpuDownload before IPC boundary"
    },
    
    /// No common memory type for cross-process link
    NoCommonIpcMemory {
        upstream_process: ProcessId,
        downstream_process: ProcessId,
        upstream_caps: MemoryCaps,
        downstream_caps: MemoryCaps,
    },
    
    /// DMA-BUF modifier mismatch across processes
    DrmModifierMismatch {
        link: LinkId,
        upstream_modifiers: Vec<u64>,
        downstream_modifiers: Vec<u64>,
    },
}

// Example error output:
// 
// Cross-process negotiation failed:
//   Process A: gpu_filter [outputs: GpuDevice]
//        ──IPC──>
//   Process B: cpu_encoder [accepts: Heap, SharedMemory]
//
//   GPU memory cannot directly cross process boundaries.
//
//   Suggestions:
//   - Insert gpu_download before IPC: gpu_filter ! gpu_download ! ipc_sink
//   - Use DMA-BUF export: gpu_filter output_memory=dmabuf ! ipc_sink
```

---

### Integration with Existing Memory System

Parallax already has the memory infrastructure:

| Existing | Change |
|----------|--------|
| `MemoryType` enum | Simplify: merge Heap+SharedMemory → `Cpu` |
| `HeapSegment` | Replace with `CpuSegment` (memfd-backed) |
| `SharedMemorySegment` | Merge into `CpuSegment` |
| `MemoryPool` | Keep, use with `CpuSegment` |
| `IpcHandle` | Keep for fd passing |

New additions needed:
- `GpuSegment` - Vulkan/wgpu device memory
- `DmaBufSegment` - Linux DMA-BUF wrapper
- `RdmaSegment` - RDMA-registered memory
- Memory transfer elements (GpuUpload, GpuDownload, etc.)

---

## Phase 1: Core Negotiation Infrastructure

### 1.1 Enhance Caps with Ranges and Constraints

**File:** `src/format.rs`

```rust
/// A value that can be fixed or a range/set
#[derive(Debug, Clone, PartialEq)]
pub enum CapsValue<T> {
    /// Exact value required
    Fixed(T),
    /// Range of acceptable values (inclusive)
    Range { min: T, max: T },
    /// List of acceptable values (ordered by preference)
    List(Vec<T>),
    /// Any value accepted
    Any,
}

impl<T: Ord + Clone> CapsValue<T> {
    /// Intersect two caps values, return None if no overlap
    pub fn intersect(&self, other: &Self) -> Option<Self>;
    
    /// Fix to a concrete value (pick first/best)
    pub fn fixate(&self) -> Option<T>;
    
    /// Check if a fixed value satisfies this constraint
    pub fn accepts(&self, value: &T) -> bool;
}

/// Video format with optional constraints (for negotiation)
#[derive(Debug, Clone)]
pub struct VideoFormatCaps {
    pub width: CapsValue<u32>,
    pub height: CapsValue<u32>,
    pub pixel_format: CapsValue<PixelFormat>,
    pub framerate: CapsValue<Framerate>,
}

impl VideoFormatCaps {
    /// Any video format
    pub fn any() -> Self;
    
    /// Fixed video format
    pub fn fixed(format: VideoFormat) -> Self;
    
    /// Intersect with another, return None if no overlap
    pub fn intersect(&self, other: &Self) -> Option<Self>;
    
    /// Fix to a concrete format
    pub fn fixate(&self) -> Option<VideoFormat>;
}
```

Similarly for `AudioFormatCaps`, `RtpFormatCaps`.

### 1.2 Add Data Flow Mode

**File:** `src/element/traits.rs`

```rust
/// How data flows through an element
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataFlowMode {
    /// Element pushes data downstream (sources, live inputs)
    Push,
    /// Element pulls data from upstream (sinks that control timing)
    Pull,
    /// Element can work in either mode
    #[default]
    Either,
}

pub trait Source: Send {
    /// Data flow mode for this source
    fn flow_mode(&self) -> DataFlowMode { DataFlowMode::Push }
    
    // ... existing methods
}

pub trait Sink: Send {
    /// Data flow mode for this sink
    fn flow_mode(&self) -> DataFlowMode { DataFlowMode::Either }
    
    // ... existing methods
}
```

### 1.3 Enhanced Pad Structure

**File:** `src/element/pad.rs`

```rust
pub struct PadTemplate {
    pub name: String,
    pub direction: PadDirection,
    pub presence: PadPresence,
    pub caps: Caps,
    pub flow_mode: DataFlowMode,
}

pub struct Pad {
    name: String,
    direction: PadDirection,
    template: Option<Arc<PadTemplate>>,
    
    /// Supported caps (from element query)
    caps: Caps,
    
    /// Negotiated fixed format (after negotiation)
    negotiated: Option<MediaFormat>,
    
    /// Caps cache generation (for invalidation)
    caps_generation: u64,
}

impl Pad {
    pub fn caps(&self) -> &Caps;
    pub fn negotiated_format(&self) -> Option<&MediaFormat>;
    pub fn set_negotiated(&mut self, format: MediaFormat);
    
    /// Invalidate caps cache (call when element config changes)
    pub fn invalidate_caps(&mut self);
}
```

### 1.4 Store Negotiated Format on Links

**File:** `src/pipeline/graph.rs`

```rust
pub struct Link {
    pub from_pad: String,
    pub to_pad: String,
    pub negotiated_format: Option<MediaFormat>,
    pub flow_mode: DataFlowMode,
}
```

---

## Phase 2: Constraint-Based Negotiation Algorithm

### 2.1 Format Constraints

**File:** `src/pipeline/negotiation.rs`

```rust
/// A constraint on formats at a specific point in the graph
#[derive(Debug, Clone)]
pub struct FormatConstraint {
    /// Which link this constraint applies to
    pub link_id: LinkId,
    /// Upstream element's output caps
    pub upstream_caps: Caps,
    /// Downstream element's input caps  
    pub downstream_caps: Caps,
}

/// Result of constraint solving
pub struct NegotiationResult {
    /// Negotiated format for each link
    pub formats: HashMap<LinkId, MediaFormat>,
    /// Converters to insert (if auto-conversion enabled)
    pub converters: Vec<ConverterInsertion>,
}

pub struct ConverterInsertion {
    pub link_id: LinkId,
    pub converter: Box<dyn ElementDyn>,
}
```

### 2.2 Negotiation Solver

```rust
pub struct NegotiationSolver<'a> {
    pipeline: &'a Pipeline,
    allow_auto_conversion: bool,
    converter_registry: Option<&'a ConverterRegistry>,
}

impl<'a> NegotiationSolver<'a> {
    pub fn new(pipeline: &'a Pipeline) -> Self;
    
    /// Enable automatic converter insertion
    pub fn with_auto_conversion(mut self, registry: &'a ConverterRegistry) -> Self;
    
    /// Solve negotiation for entire pipeline
    pub fn solve(&self) -> Result<NegotiationResult> {
        // 1. Determine data flow mode for pipeline
        let flow_mode = self.determine_flow_mode()?;
        
        // 2. Collect constraints from all links
        let constraints = self.collect_constraints()?;
        
        // 3. Propagate constraints (iterative fixpoint)
        let propagated = self.propagate_constraints(constraints)?;
        
        // 4. Try to fixate all formats
        match self.fixate_all(&propagated) {
            Ok(formats) => Ok(NegotiationResult { formats, converters: vec![] }),
            Err(failures) if self.allow_auto_conversion => {
                // 5. Try inserting converters
                self.solve_with_converters(failures)
            }
            Err(failures) => Err(self.build_error(failures)),
        }
    }
    
    /// Determine overall pipeline flow mode
    fn determine_flow_mode(&self) -> Result<DataFlowMode> {
        // Find sources and sinks
        // If any source is Push-only, pipeline is Push
        // If any sink is Pull-only, pipeline is Pull
        // Otherwise, prefer Push (lower latency)
    }
    
    /// Propagate constraints through graph (like type inference)
    fn propagate_constraints(&self, initial: Vec<FormatConstraint>) -> Result<Vec<FormatConstraint>> {
        let mut constraints = initial;
        let mut changed = true;
        
        while changed {
            changed = false;
            
            // For each transform, propagate input constraints to output
            for node in self.pipeline.transforms() {
                if let Some(new_constraint) = self.propagate_through_transform(node, &constraints)? {
                    constraints.push(new_constraint);
                    changed = true;
                }
            }
        }
        
        Ok(constraints)
    }
}
```

### 2.3 Push vs Pull Negotiation

```rust
impl<'a> NegotiationSolver<'a> {
    /// Push mode: sources drive, negotiate downstream
    fn negotiate_push(&self) -> Result<NegotiationResult> {
        let order = self.pipeline.topological_order()?;
        
        for node_id in order {
            self.negotiate_node_push(node_id)?;
        }
    }
    
    /// Pull mode: sinks drive, negotiate upstream
    fn negotiate_pull(&self) -> Result<NegotiationResult> {
        let order = self.pipeline.reverse_topological_order()?;
        
        for node_id in order {
            self.negotiate_node_pull(node_id)?;
        }
    }
}
```

### 2.4 Rich Error Messages

```rust
#[derive(Debug)]
pub struct NegotiationError {
    pub failures: Vec<LinkFailure>,
}

#[derive(Debug)]
pub struct LinkFailure {
    pub upstream_element: String,
    pub upstream_caps: Caps,
    pub downstream_element: String,
    pub downstream_caps: Caps,
    pub suggestions: Vec<Suggestion>,
}

#[derive(Debug)]
pub enum Suggestion {
    InsertConverter { converter_name: String },
    ChangeElementProperty { element: String, property: String, suggested_value: String },
    UseAlternativeElement { current: String, alternative: String },
}

impl std::fmt::Display for NegotiationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Format negotiation failed:\n")?;
        
        for failure in &self.failures {
            writeln!(f, "  {} [produces: {}]", failure.upstream_element, failure.upstream_caps)?;
            writeln!(f, "       ↓")?;
            writeln!(f, "  {} [accepts: {}]", failure.downstream_element, failure.downstream_caps)?;
            writeln!(f)?;
            
            if !failure.suggestions.is_empty() {
                writeln!(f, "  Suggestions:")?;
                for suggestion in &failure.suggestions {
                    writeln!(f, "  - {}", suggestion)?;
                }
            }
            writeln!(f)?;
        }
        
        Ok(())
    }
}
```

---

## Phase 3: Pipeline Integration

### 3.1 Add Negotiation to Pipeline Executor

**File:** `src/pipeline/executor.rs`

```rust
impl PipelineExecutor {
    pub async fn start(&mut self) -> Result<()> {
        // 1. Validate structure
        self.pipeline.validate()?;
        
        // 2. Negotiate formats
        let negotiation = NegotiationSolver::new(&self.pipeline)
            .with_auto_conversion(&self.converter_registry)  // Optional
            .solve()?;
        
        // 3. Apply negotiation results
        self.apply_negotiation(negotiation)?;
        
        // 4. Spawn tasks
        self.spawn_tasks()?;
        
        Ok(())
    }
    
    fn apply_negotiation(&mut self, result: NegotiationResult) -> Result<()> {
        // Store negotiated formats on links
        for (link_id, format) in result.formats {
            self.pipeline.link_mut(link_id)?.negotiated_format = Some(format);
        }
        
        // Insert converters if needed
        for insertion in result.converters {
            self.pipeline.insert_element(insertion.link_id, insertion.converter)?;
        }
        
        // Notify elements of their negotiated formats
        self.notify_elements_of_formats()?;
        
        Ok(())
    }
}
```

### 3.2 Element Context with Format Info

**File:** `src/element/context.rs`

```rust
pub struct ElementContext {
    pub name: String,
    
    /// Format negotiated for each input pad
    pub input_formats: HashMap<String, MediaFormat>,
    
    /// Format negotiated for each output pad
    pub output_formats: HashMap<String, MediaFormat>,
    
    /// Pipeline's data flow mode
    pub flow_mode: DataFlowMode,
}

impl ElementContext {
    /// Get input format (for single-input elements)
    pub fn input_format(&self) -> Option<&MediaFormat> {
        self.input_formats.get("sink")
    }
    
    /// Get output format (for single-output elements)
    pub fn output_format(&self) -> Option<&MediaFormat> {
        self.output_formats.get("src")
    }
}
```

---

## Phase 4: Converter Registry

### 4.1 Converter Registration

**File:** `src/pipeline/converters.rs`

```rust
pub type ConverterFactory = Box<dyn Fn() -> Box<dyn ElementDyn> + Send + Sync>;

pub struct ConverterRegistry {
    /// Converters indexed by (from_type, to_type)
    converters: HashMap<(MediaType, MediaType), Vec<RegisteredConverter>>,
}

pub struct RegisteredConverter {
    pub name: String,
    pub factory: ConverterFactory,
    /// Cost of this conversion (lower = preferred)
    pub cost: u32,
    /// Does this converter preserve quality?
    pub lossless: bool,
}

impl ConverterRegistry {
    pub fn new() -> Self;
    
    /// Register a converter
    pub fn register(&mut self, 
        from: MediaType, 
        to: MediaType, 
        name: &str,
        factory: ConverterFactory,
        cost: u32,
        lossless: bool,
    );
    
    /// Find best converter for a specific conversion
    pub fn find(&self, from: &MediaFormat, to: &MediaFormat) -> Option<&RegisteredConverter>;
    
    /// Find shortest path of conversions (Dijkstra)
    pub fn find_path(&self, from: &MediaFormat, to: &MediaFormat) -> Option<Vec<&RegisteredConverter>>;
}

impl Default for ConverterRegistry {
    fn default() -> Self {
        let mut registry = Self::new();
        
        // Register built-in converters
        registry.register(
            MediaType::VideoRaw, MediaType::VideoRaw,
            "colorconvert",
            Box::new(|| Box::new(ColorConvert::new())),
            cost: 10,
            lossless: false,
        );
        
        registry.register(
            MediaType::AudioRaw, MediaType::AudioRaw,
            "audioresample",
            Box::new(|| Box::new(AudioResample::new())),
            cost: 5,
            lossless: false,
        );
        
        registry
    }
}
```

---

## Phase 5: Dynamic Renegotiation

### 5.1 Format Change Events

**File:** `src/pipeline/mod.rs`

```rust
pub enum PipelineEvent {
    // Existing...
    
    /// Element's caps have changed, renegotiation needed
    CapsInvalidated {
        element: String,
        pad: String,
    },
    
    /// Format was renegotiated
    CapsChanged {
        element: String,
        pad: String,
        old_format: Option<MediaFormat>,
        new_format: MediaFormat,
    },
}
```

### 5.2 Renegotiation Mechanism

```rust
impl Pipeline {
    /// Request renegotiation (called when element config changes)
    pub fn request_renegotiation(&mut self, element: &str) -> Result<()> {
        // 1. Mark element's caps as invalid
        self.node_mut(element)?.invalidate_caps();
        
        // 2. Emit event
        self.emit_event(PipelineEvent::CapsInvalidated {
            element: element.to_string(),
            pad: "src".to_string(),
        });
        
        // 3. If pipeline is running, trigger renegotiation
        if self.state() == PipelineState::Playing {
            self.renegotiate()?;
        }
        
        Ok(())
    }
    
    /// Perform renegotiation on running pipeline
    fn renegotiate(&mut self) -> Result<()> {
        // Pause data flow
        // Re-run negotiation solver
        // Resume data flow
    }
}
```

---

## Phase 6: Pull Mode Support

### 6.1 Pull-Mode Sources

For sources that support pull mode (random access):

```rust
pub trait PullSource: Send {
    /// Pull data starting at offset
    fn pull(&mut self, offset: u64, size: usize) -> Result<Option<Buffer>>;
    
    /// Get total size (if known)
    fn size(&self) -> Option<u64>;
    
    /// Can seek to arbitrary positions?
    fn seekable(&self) -> bool;
    
    fn output_caps(&self) -> Caps { Caps::any() }
}
```

### 6.2 Pull-Mode Sinks

For sinks that drive timing:

```rust
pub trait PullSink: Send {
    /// Request next buffer (sink controls timing)
    fn request(&mut self) -> Result<BufferRequest>;
    
    /// Consume the pulled buffer
    fn consume(&mut self, buffer: Buffer) -> Result<()>;
    
    fn input_caps(&self) -> Caps { Caps::any() }
}

pub struct BufferRequest {
    pub offset: Option<u64>,  // None = next sequential
    pub size: Option<usize>,  // None = any size
    pub deadline: Option<Instant>,  // When buffer is needed
}
```

### 6.3 Hybrid Pipelines

Support mixed push/pull:

```rust
pub struct PipelineConfig {
    /// Default flow mode
    pub default_flow_mode: DataFlowMode,
    
    /// Allow mixing push and pull in same pipeline?
    /// (requires queue elements at boundaries)
    pub allow_mixed_flow: bool,
}
```

---

## Transparent Memory Abstraction for Element Writers

### The Goal

Element writers should be able to write processing code **once** and have it work regardless of where the data actually lives (heap, shared memory, GPU, DMA-BUF). The negotiation system handles memory placement; elements just process bytes.

### What Doesn't Directly Solve This

**Raw mmap:** Works for heap/shared memory, but:
- GPU VRAM cannot be directly mmap'd
- DMA-BUF mmap requires explicit sync (not transparent)

**CUDA Unified Memory:** Automatic page migration, but:
- NVIDIA-only
- Page fault overhead on first access
- Not available in Vulkan ecosystem

### What Can Work: Abstraction Layer

We provide abstractions that hide memory location from element implementations:

#### 1. MemoryView: Read-Only Access

```rust
/// A view into buffer data, regardless of where it lives
pub struct MemoryView<'a> {
    inner: MemoryViewInner<'a>,
}

enum MemoryViewInner<'a> {
    /// Direct slice (heap, shared memory, mapped file)
    Direct(&'a [u8]),
    
    /// DMA-BUF with active mapping
    DmaBuf {
        ptr: *const u8,
        len: usize,
        _guard: DmaBufMapGuard,
    },
    
    /// Staged from GPU (copied to temp buffer)
    Staged {
        staging: Box<[u8]>,
    },
}

impl<'a> MemoryView<'a> {
    /// Element writers use this - they don't know/care where data is
    pub fn as_slice(&self) -> &[u8] {
        match &self.inner {
            MemoryViewInner::Direct(slice) => slice,
            MemoryViewInner::DmaBuf { ptr, len, .. } => 
                unsafe { std::slice::from_raw_parts(*ptr, *len) },
            MemoryViewInner::Staged { staging } => staging,
        }
    }
    
    /// Hint: is this a zero-copy view or did we have to copy?
    pub fn is_zero_copy(&self) -> bool {
        !matches!(self.inner, MemoryViewInner::Staged { .. })
    }
}
```

#### 2. MemoryViewMut: Read-Write Access

```rust
pub struct MemoryViewMut<'a> {
    inner: MemoryViewMutInner<'a>,
}

enum MemoryViewMutInner<'a> {
    Direct(&'a mut [u8]),
    DmaBuf {
        ptr: *mut u8,
        len: usize,
        _guard: DmaBufMapGuard,
    },
    /// Staged: write to temp, sync back to GPU on drop
    Staged {
        staging: Box<[u8]>,
        gpu_handle: GpuBufferHandle,
        needs_writeback: bool,
    },
}

impl Drop for MemoryViewMutInner<'_> {
    fn drop(&mut self) {
        if let MemoryViewMutInner::Staged { staging, gpu_handle, needs_writeback: true } = self {
            // Schedule async copy back to GPU
            gpu_handle.schedule_upload(staging);
        }
    }
}
```

#### 3. Buffer API for Element Writers

```rust
impl Buffer {
    /// Get read-only view (may stage from GPU if necessary)
    pub fn view(&self) -> MemoryView<'_> {
        match &self.memory {
            MemoryHandle::Heap(segment) => 
                MemoryView::direct(segment.as_slice()),
            MemoryHandle::SharedMemory(segment) => 
                MemoryView::direct(segment.as_slice()),
            MemoryHandle::DmaBuf(dmabuf) => 
                MemoryView::dmabuf(dmabuf.map_read()?),
            MemoryHandle::Gpu(gpu) => 
                MemoryView::staged(gpu.download_sync()?),
        }
    }
    
    /// Get mutable view (may stage and writeback)
    pub fn view_mut(&mut self) -> MemoryViewMut<'_> {
        // Similar logic with writeback on drop
    }
}
```

#### 4. Element Trait Remains Simple

```rust
pub trait Element: Send {
    /// Elements work with views - memory location is hidden
    fn process(&mut self, input: &Buffer, output: &mut Buffer) -> Result<()> {
        let input_data = input.view();
        let mut output_data = output.view_mut();
        
        // Just work with slices, don't care about GPU/heap/etc
        process_bytes(input_data.as_slice(), output_data.as_mut_slice());
        
        Ok(())
    }
    
    /// Declare preferences (affects negotiation, not implementation)
    fn memory_preferences(&self) -> MemoryPreferences {
        MemoryPreferences::default() // "I work with anything"
    }
}
```

### For GPU-Native Elements

Elements that *can* process directly on GPU get an additional trait:

```rust
/// Marker: this element can process data on GPU without staging
pub trait GpuNative: Element {
    /// Process using GPU compute/render pipeline
    fn process_gpu(
        &mut self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()>;
}

/// The pipeline chooses which path to use
impl Pipeline {
    fn run_element(&self, element: &dyn Element, input: Buffer, output: &mut Buffer) {
        if let Some(gpu_element) = element.as_gpu_native() {
            if input.is_gpu() && output.is_gpu() {
                // Zero-copy GPU path
                return gpu_element.process_gpu(input.as_gpu(), output.as_gpu_mut(), encoder);
            }
        }
        
        // Fallback: staging path (works everywhere)
        element.process(&input, output)
    }
}
```

### rust-gpu: Write Once, Run on CPU or GPU

rust-gpu compiles Rust to SPIR-V, enabling **single-source** code that runs on both CPU and GPU. This is powerful for element implementations:

```rust
// Shared algorithm - compiles to both CPU and SPIR-V
// Put in a separate crate with `#![cfg_attr(target_arch = "spirv", no_std)]`
pub fn rgb_to_yuv(rgb: u32) -> u32 {
    let r = ((rgb >> 16) & 0xFF) as f32;
    let g = ((rgb >> 8) & 0xFF) as f32;
    let b = (rgb & 0xFF) as f32;
    
    let y = (0.299 * r + 0.587 * g + 0.114 * b) as u32;
    let u = ((-0.169 * r - 0.331 * g + 0.5 * b) + 128.0) as u32;
    let v = ((0.5 * r - 0.419 * g - 0.081 * b) + 128.0) as u32;
    
    (y << 16) | (u << 8) | v
}

// GPU entry point - uses shared algorithm
#[cfg(target_arch = "spirv")]
#[spirv(compute(threads(256)))]
pub fn color_convert_kernel(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] input: &[u32],
    #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] output: &mut [u32],
) {
    let idx = id.x as usize;
    if idx < input.len() {
        output[idx] = rgb_to_yuv(input[idx]);  // Same function!
    }
}

// CPU entry point - uses shared algorithm  
#[cfg(not(target_arch = "spirv"))]
pub fn color_convert_cpu(input: &[u32], output: &mut [u32]) {
    for (i, o) in input.iter().zip(output.iter_mut()) {
        *o = rgb_to_yuv(*i);  // Same function!
    }
}
```

**Key benefits:**
- Algorithm written once (`rgb_to_yuv`)
- rust-gpu compiles it to SPIR-V for GPU execution
- Standard rustc compiles it for CPU execution
- No divergence between CPU and GPU implementations

**Limitations (rust-gpu subset):**
- No `std` library on GPU (use `core`/`libm`)
- No dynamic allocation
- No recursion (GPU shader limitation)
- Limited control flow in some cases

**Integration with Parallax:**

```rust
impl ColorConvert {
    // Element trait - CPU path via MemoryView
    fn process(&mut self, input: &Buffer, output: &mut Buffer) -> Result<()> {
        let src = input.view().as_slice();
        let dst = output.view_mut().as_mut_slice();
        color_convert_cpu(bytemuck::cast_slice(src), bytemuck::cast_slice_mut(dst));
        Ok(())
    }
}

impl GpuNative for ColorConvert {
    // GPU path - dispatch SPIR-V shader
    fn process_gpu(&mut self, input: &GpuBuffer, output: &mut GpuBuffer, 
                   encoder: &mut wgpu::CommandEncoder) -> Result<()> {
        // Bind buffers, dispatch compute shader compiled from color_convert_kernel
        self.dispatch_shader(encoder, input, output);
        Ok(())
    }
}
```

The pipeline chooses:
- **GPU path** when data is already on GPU (zero-copy)
- **CPU path** when data is on heap/shared memory (avoids upload/download)

Both paths use the same `rgb_to_yuv` algorithm - no code duplication.

### Linux HMM (Future)

Linux Heterogeneous Memory Management allows:
- GPU to page-fault into CPU memory
- Transparent access without explicit copies
- Supported by AMD ROCm, newer NVIDIA drivers

When available, we can add:

```rust
enum MemoryViewInner<'a> {
    // ...existing variants...
    
    /// HMM-backed: GPU can access this directly via page faults
    Hmm {
        ptr: *const u8,
        len: usize,
    },
}
```

For now, this is GPU-driver-specific and not portable. Our staging approach works everywhere.

### Cost Model

| Access Pattern | Cost |
|---------------|------|
| Heap via `view()` | 0 (direct pointer) |
| SharedMemory via `view()` | 0 (direct pointer) |
| DMA-BUF via `view()` | ~0 (mmap, cached) |
| GPU via `view()` | High (staging copy) |
| GPU via `GpuNative::process_gpu()` | 0 (native) |

The negotiation solver uses this cost model to minimize total copies across the pipeline.

### Pipeline Provides Output Buffers (Element Writers Don't Allocate)

Element writers should **never allocate memory**. The pipeline provides pre-allocated output buffers in the correct memory location based on negotiation.

#### The Element Trait (Final Design)

```rust
pub trait Element: Send {
    /// Process input into pre-allocated output
    /// Pipeline provides both - element just processes
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<()>;
    
    /// Tell pipeline what output size you need
    fn output_size_hint(&self, input_len: usize) -> OutputSizeHint {
        OutputSizeHint::SameAsInput
    }
    
    /// Can process in-place? (reuse input buffer as output)
    fn can_process_in_place(&self) -> bool {
        false
    }
}

pub enum OutputSizeHint {
    SameAsInput,
    Fixed(usize),
    Ratio { numerator: usize, denominator: usize },  // e.g., 1/10 for compression
    Unknown,  // Pipeline allocates conservatively
}
```

#### ProcessContext: Zero-Allocation Processing

```rust
/// Everything element needs - no allocation required
pub struct ProcessContext<'a> {
    /// Input data (read-only view)
    input: MemoryView<'a>,
    
    /// Output location (pre-allocated by pipeline)
    output: MemoryViewMut<'a>,
    
    /// For variable-size output
    committed_len: usize,
}

impl<'a> ProcessContext<'a> {
    /// Read input - works regardless of memory location
    pub fn input(&self) -> &[u8] {
        self.input.as_slice()
    }
    
    /// Write to output - already in correct memory location
    pub fn output(&mut self) -> &mut [u8] {
        self.output.as_mut_slice()
    }
    
    /// For variable-size output: commit actual size used
    pub fn commit(&mut self, actual_len: usize) {
        self.committed_len = actual_len;
    }
}
```

#### Pipeline Manages All Allocation

```rust
impl PipelineExecutor {
    async fn run_element(
        &mut self,
        element: &mut dyn Element,
        input: Buffer,
    ) -> Result<Buffer> {
        // 1. Get negotiated output memory type for this element
        let output_mem_type = self.negotiated_output_memory(element.id());
        
        // 2. Calculate output size
        let output_size = match element.output_size_hint(input.len()) {
            OutputSizeHint::SameAsInput => input.len(),
            OutputSizeHint::Fixed(n) => n,
            OutputSizeHint::Ratio { numerator, denominator } => 
                input.len() * numerator / denominator,
            OutputSizeHint::Unknown => input.len(),  // Conservative
        };
        
        // 3. Try in-place first (zero allocation!)
        if element.can_process_in_place() && self.can_reuse_buffer(&input, output_mem_type) {
            let mut buffer = input;
            let mut ctx = ProcessContext::in_place(&mut buffer);
            element.process(&mut ctx)?;
            buffer.truncate(ctx.committed_len);
            return Ok(buffer);
        }
        
        // 4. Acquire output from correct pool (based on negotiation)
        let mut output = self.pools.acquire(output_mem_type, output_size)?;
        
        // 5. Create context with input view and output view
        let mut ctx = ProcessContext::new(input.view(), output.view_mut());
        
        // 6. Element processes - writes directly to final destination
        element.process(&mut ctx)?;
        
        output.truncate(ctx.committed_len);
        Ok(output)
    }
}
```

#### Example: Writing an Element (Final Experience)

```rust
/// Brightness filter - simple, no memory concerns
pub struct BrightnessFilter {
    adjustment: i32,
}

impl Element for BrightnessFilter {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<()> {
        // Just get slices - don't care about memory type
        let input = ctx.input();
        let output = ctx.output();
        
        // Process
        for (i, o) in input.iter().zip(output.iter_mut()) {
            *o = (*i as i32 + self.adjustment).clamp(0, 255) as u8;
        }
        
        Ok(())
    }
    
    fn can_process_in_place(&self) -> bool {
        true  // Can modify buffer in place
    }
}

/// H.264 Encoder - variable output size
pub struct H264Encoder {
    encoder: x264::Encoder,
}

impl Element for H264Encoder {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<()> {
        let input = ctx.input();
        let output = ctx.output();
        
        // Encode into output buffer
        let encoded_len = self.encoder.encode(input, output)?;
        
        // Tell pipeline actual size used
        ctx.commit(encoded_len);
        
        Ok(())
    }
    
    fn output_size_hint(&self, input_len: usize) -> OutputSizeHint {
        // Compressed typically smaller, but be safe
        OutputSizeHint::SameAsInput
    }
}

/// Color converter with GPU support
pub struct ColorConverter {
    gpu_shader: Option<ComputeShader>,
}

impl Element for ColorConverter {
    fn process(&mut self, ctx: &mut ProcessContext) -> Result<()> {
        // Works on CPU memory (staged if needed)
        let input = ctx.input();
        let output = ctx.output();
        color_convert_cpu(input, output);
        Ok(())
    }
}

impl GpuNative for ColorConverter {
    fn process_gpu(
        &mut self,
        input: &GpuBuffer,
        output: &mut GpuBuffer,
        encoder: &mut wgpu::CommandEncoder,
    ) -> Result<()> {
        // Zero-copy GPU path when both buffers on GPU
        self.gpu_shader.as_ref().unwrap().dispatch(encoder, input, output);
        Ok(())
    }
}
```

### Summary for Element Writers

1. **Never allocate:** Pipeline provides input and output buffers
2. **Use `ctx.input()` and `ctx.output()`:** Works regardless of memory location
3. **Implement `output_size_hint()`:** Tell pipeline how much space you need
4. **Implement `can_process_in_place()`:** Enable zero-copy when possible
5. **Optionally implement `GpuNative`:** For zero-copy GPU processing
6. **Don't think about IPC/network:** Pipeline handles cross-process/machine transparently

---

## Implementation Order

### Milestone 1: Core Infrastructure
1. [ ] Add `CapsValue<T>` with intersect/fixate
2. [ ] Add `VideoFormatCaps`, `AudioFormatCaps`
3. [ ] Add `DataFlowMode` to elements
4. [ ] Enhance `Pad` with caps and negotiated format
5. [ ] Add negotiated_format to `Link`

### Milestone 2: Basic Negotiation
1. [ ] Implement `NegotiationSolver`
2. [ ] Implement push-mode negotiation
3. [ ] Add `NegotiationError` with rich messages
4. [ ] Integrate into `PipelineExecutor::start()`
5. [ ] Add integration tests

### Milestone 3: Element Updates
1. [ ] Add real caps to `VideoTestSrc`
2. [ ] Add real caps to `IcedVideoSink`
3. [ ] Create `ElementContext` and pass to elements
4. [ ] Update adapters to provide context

### Milestone 4: Converter Registry
1. [ ] Implement `ConverterRegistry`
2. [ ] Add path-finding (Dijkstra)
3. [ ] Create `ColorConvert` element
4. [ ] Create `AudioResample` element
5. [ ] Enable auto-conversion in solver

### Milestone 5: Memory Negotiation
1. [ ] Add `MemoryCaps` structure
2. [ ] Add `input_memory_caps()` / `output_memory_caps()` to element traits
3. [ ] Implement memory domain graph
4. [ ] Implement `optimize_memory_placement()` algorithm
5. [ ] Add `MemoryPlacement` to `NegotiationResult`
6. [ ] Create `GpuUpload` / `GpuDownload` elements
7. [ ] Add `GpuSegment` and `DmaBufSegment` to memory module
8. [ ] Add DRM modifier negotiation

### Milestone 6: Memory Abstraction for Elements
1. [ ] Add `MemoryView` and `MemoryViewMut` types
2. [ ] Implement `Buffer::view()` and `Buffer::view_mut()`
3. [ ] Add `DmaBufMapGuard` with proper sync
4. [ ] Add `GpuNative` trait for GPU-capable elements
5. [ ] Implement staging fallback for GPU→CPU access
6. [ ] Update `Pipeline` to choose GPU vs CPU path
7. [ ] Document element writer guide

### Milestone 7: Arena Allocation & ProcessContext API
1. [ ] Implement `CpuArena` (single memfd for many buffers)
2. [ ] Implement `ArenaSlot` with offset-based addressing
3. [ ] Add `IpcSlotRef` for cross-process (arena_id + offset, no fd per buffer)
4. [ ] Implement `ProcessContext` with input/output views
5. [ ] Add `OutputSizeHint` enum
6. [ ] Update `Element` trait to use `ProcessContext`
7. [ ] Implement in-place processing optimization
8. [ ] Update `PipelineExecutor` to manage all buffer allocation
9. [ ] Add pool-per-memory-type in executor

### Milestone 8: Cross-Process Memory Negotiation (Same Machine)
1. [ ] Add `ProcessLocation` enum (Local, RemoteProcess, RemoteMachine)
2. [ ] Add `crosses_process` and `crosses_machine` to links
3. [ ] Implement `IpcTransport` enum (FdPassing, Zenoh, Custom)
4. [ ] Add IPC constraints to negotiation solver
5. [ ] Add `IpcBufferMessage` with rkyv serialization
6. [ ] Implement arena fd caching on receiver side
7. [ ] Add `IpcSrc` and `IpcSink` elements
8. [ ] Implement Copy-on-Write for cross-process shared buffers

### Milestone 9: Cross-Machine Memory Negotiation (Network)
1. [ ] Add `NetworkEndpoint` enum (Zenoh, Socket, Rdma)
2. [ ] Add `MemoryType::supports_network()` method
3. [ ] Add network constraints to negotiation solver
4. [ ] Implement `ZenohTransport` for buffer serialization
5. [ ] Add `ZenohSrc` and `ZenohSink` elements
6. [ ] Update cost model with network latency costs
7. [ ] Add `RdmaSegment` memory type
8. [ ] Implement `RdmaTransport` for zero-copy network
9. [ ] Add `RdmaSrc` and `RdmaSink` elements
10. [ ] (Future) GPUDirect RDMA support

### Milestone 10: Pull Mode
1. [ ] Add `PullSource` and `PullSink` traits
2. [ ] Implement pull-mode negotiation
3. [ ] Add `FileSrc` pull mode support
4. [ ] Add hybrid pipeline support

### Milestone 11: Dynamic Renegotiation
1. [ ] Add `CapsInvalidated` event
2. [ ] Implement `request_renegotiation()`
3. [ ] Handle runtime renegotiation
4. [ ] Add tests for dynamic pipelines

### Milestone 12: Update All Elements
1. [ ] Audit all 43 elements
2. [ ] Add format caps to sources
3. [ ] Add format caps to sinks
4. [ ] Add format caps to transforms
5. [ ] Add memory caps to GPU elements
6. [ ] Add memory caps to network elements

---

## Testing Strategy

### Unit Tests
```rust
#[test]
fn test_caps_value_intersect_ranges() {
    let a = CapsValue::Range { min: 720, max: 2160 };
    let b = CapsValue::Range { min: 1080, max: 4320 };
    let result = a.intersect(&b);
    assert_eq!(result, Some(CapsValue::Range { min: 1080, max: 2160 }));
}

#[test]
fn test_caps_value_intersect_empty() {
    let a = CapsValue::Range { min: 720, max: 1080 };
    let b = CapsValue::Range { min: 2160, max: 4320 };
    assert_eq!(a.intersect(&b), None);
}

#[test]
fn test_video_format_caps_fixate() {
    let caps = VideoFormatCaps {
        width: CapsValue::List(vec![1920, 1280, 640]),
        height: CapsValue::Any,
        pixel_format: CapsValue::Fixed(PixelFormat::I420),
        framerate: CapsValue::Range { min: Framerate::new(24, 1), max: Framerate::new(60, 1) },
    };
    let fixed = caps.fixate().unwrap();
    assert_eq!(fixed.width, 1920);  // First in list
    assert_eq!(fixed.pixel_format, PixelFormat::I420);
}

#[test]
fn test_memory_caps_intersection() {
    let gpu_element = MemoryCaps {
        types: CapsValue::List(vec![MemoryType::GpuDevice, MemoryType::DmaBuf]),
        can_import: vec![],
        can_export: vec![MemoryType::DmaBuf, MemoryType::Heap],
        ..Default::default()
    };
    
    let display_element = MemoryCaps {
        types: CapsValue::List(vec![MemoryType::GpuDevice, MemoryType::DmaBuf, MemoryType::Heap]),
        can_import: vec![MemoryType::DmaBuf, MemoryType::Heap],
        can_export: vec![],
        ..Default::default()
    };
    
    let result = gpu_element.intersect(&display_element);
    // Should prefer GpuDevice (zero-copy), then DmaBuf
    assert_eq!(result.unwrap().types.fixate(), Some(MemoryType::GpuDevice));
}
```

### Integration Tests
```rust
#[test]
fn test_negotiation_push_mode() {
    let pipeline = Pipeline::parse("videotestsrc ! iced_video_sink")?;
    let result = NegotiationSolver::new(&pipeline).solve()?;
    
    assert!(result.converters.is_empty());
    assert!(result.formats.values().all(|f| f.is_some()));
}

#[test]
fn test_negotiation_needs_converter() {
    let registry = ConverterRegistry::default();
    let pipeline = Pipeline::parse("videotestsrc format=RGB24 ! h264enc")?;
    
    let result = NegotiationSolver::new(&pipeline)
        .with_auto_conversion(&registry)
        .solve()?;
    
    assert_eq!(result.converters.len(), 1);
    assert_eq!(result.converters[0].converter.name(), "colorconvert");
}

#[test]
fn test_negotiation_error_message() {
    let pipeline = Pipeline::parse("videotestsrc format=RGB24 ! h264enc")?;
    let err = NegotiationSolver::new(&pipeline).solve().unwrap_err();
    
    let msg = err.to_string();
    assert!(msg.contains("videotestsrc"));
    assert!(msg.contains("h264enc"));
    assert!(msg.contains("Suggestions"));
}

#[test]
fn test_memory_negotiation_zero_copy_gpu() {
    // All GPU elements -> should stay on GPU
    let pipeline = Pipeline::parse("vulkan_h264_dec ! vulkan_filter ! iced_video_sink")?;
    let result = NegotiationSolver::new(&pipeline).solve()?;
    
    // No memory transfers needed
    assert!(result.memory_placement.transfers.is_empty());
    
    // All links use GPU memory
    for (_, mem_type) in &result.memory_placement.link_memory {
        assert_eq!(*mem_type, MemoryType::GpuDevice);
    }
}

#[test]
fn test_memory_negotiation_auto_transfer() {
    // GPU decoder -> CPU sink requires download
    let pipeline = Pipeline::parse("vulkan_h264_dec ! tcp_sink")?;
    let result = NegotiationSolver::new(&pipeline).solve()?;
    
    // Should insert one GPU download
    assert_eq!(result.memory_placement.transfers.len(), 1);
    assert_eq!(result.memory_placement.transfers[0].from_type, MemoryType::GpuDevice);
    assert_eq!(result.memory_placement.transfers[0].to_type, MemoryType::Heap);
}

#[test]
fn test_memory_negotiation_dmabuf_preferred() {
    // When both support DMA-BUF, prefer it over heap copy
    let pipeline = Pipeline::parse("v4l2src ! vulkan_h264_enc")?;
    let result = NegotiationSolver::new(&pipeline).solve()?;
    
    // Should use DMA-BUF for zero-copy camera -> encoder
    let link_mem = result.memory_placement.link_memory.values().next().unwrap();
    assert_eq!(*link_mem, MemoryType::DmaBuf);
}
```

---

## API Examples

### Basic Usage
```rust
let pipeline = Pipeline::parse("videotestsrc ! iced_video_sink")?;
pipeline.start().await?;  // Negotiation happens automatically
```

### With Auto-Conversion
```rust
let mut executor = PipelineExecutor::new(pipeline);
executor.enable_auto_conversion(ConverterRegistry::default());
executor.start().await?;  // Converters inserted as needed
```

### Querying Negotiated Format
```rust
pipeline.start().await?;

let link = pipeline.link_between("videotestsrc", "sink")?;
println!("Negotiated: {:?}", link.negotiated_format);
// Output: Some(VideoRaw { width: 1920, height: 1080, format: I420, framerate: 30/1 })
```

### Pull Mode Pipeline
```rust
let pipeline = Pipeline::new()
    .add_source("file", FileSrc::new("video.mp4").pull_mode(true))
    .add_sink("sink", FileSink::new("output.raw"))
    .link("file", "sink")?;

// Sink drives the timing, pulling from source
pipeline.start().await?;
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Breaking existing pipelines | `Caps::any()` remains default, backward compatible |
| Constraint solving too slow | Cache results, limit iterations |
| Pull mode complexity | Start with push-only, add pull later |
| Converter quality loss | Mark converters as lossy/lossless, prefer lossless |
| Dynamic renegotiation race conditions | Pause data flow during renegotiation |

---

## References

### Other Systems
- [GStreamer Caps Negotiation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/negotiation.html)
- [GStreamer Buffer Pool / Allocation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/allocation.html)
- [GStreamer DMA-BUF Design](https://gstreamer.freedesktop.org/documentation/additional/design/dmabuf.html)
- [FFmpeg libavfilter Format Negotiation](http://www.normalesup.org/~george/articles/format_negotiation_in_libavfilter/)
- [DirectShow Media Type Negotiation](https://learn.microsoft.com/en-us/windows/win32/directshow/negotiating-media-types)
- [PipeWire Documentation](https://docs.pipewire.org/)
- [PipeWire DMA-BUF Sharing](https://docs.pipewire.org/page_dma_buf.html)

### Memory and Zero-Copy
- [DMA-BUF Modifier Negotiation in GStreamer](https://blogs.igalia.com/vjaquez/dmabuf-modifier-negotiation-in-gstreamer/)
- [Linux DMA-BUF Kernel Documentation](https://docs.kernel.org/driver-api/dma-buf.html)
- [Vulkan VK_EXT_external_memory_dma_buf](https://docs.vulkan.org/refpages/latest/refpages/source/VK_EXT_external_memory_dma_buf.html)
- [CUDA Unified Memory](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/unified-memory.html)
- [Grus: GPU Graph Processing with Unified Memory](https://dl.acm.org/doi/10.1145/3444844)

### Research
- [Data Flow Refinement Type Inference](https://arxiv.org/abs/2011.04876)
- [Better Type Error Messages for Constraint-Based Inference](https://dl.acm.org/doi/10.1145/3622812)
- [Backpressure in Reactive Systems](https://medium.com/@jayphelps/backpressure-explained-the-flow-of-data-through-software-2350b3e77ce7)
- [UMH: Hardware-Based Unified Memory Hierarchy](https://dl.acm.org/doi/10.1145/2996190)

### GStreamer Issues
- [Caps Negotiation Confusion](https://gstreamer-devel.narkive.com/UWBFvbbt/caps-negotiation-confusion)
- [Dynamic Pipeline Negotiation Issues](https://discourse.gstreamer.org/t/issues-with-caps-negotiation-delayed-linking-deinterleave-src-0-to-queue-sink/649)
