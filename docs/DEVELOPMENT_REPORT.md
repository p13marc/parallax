# Parallax Development Report

**Date:** January 2026  
**Scope:** MPEG-TS muxer, KLV/STANAG metadata, video scaling, AV1 codecs, Pipeline API integration

---

## Executive Summary

During recent development of media pipeline features (MPEG-TS muxing, STANAG metadata injection, video transcoding), several pain points emerged that highlight areas for improvement in the Parallax architecture. While the core memory model and element traits are solid, the Pipeline API ergonomics and metadata handling need attention.

---

## 1. Pain Points Encountered

### 1.1 Pipeline API vs Manual Orchestration Gap

**Problem:** Examples 28-30 were written with manual element orchestration (direct function calls) rather than using the Pipeline API. This happened because:

- The Pipeline API requires multiple adapter layers (`SourceAdapter`, `SinkAdapter`, `ElementAdapter`, `DynAsyncElement::new_box()`)
- No clear documentation on when to use which adapter
- The `Element` trait returns `Option<Buffer>` while `Transform` returns `Output` - confusing overlap

**Evidence:**
```rust
// Manual approach (what was naturally written):
let scaled = scaler.scale_yuv420(&input)?;
let encoded = encoder.encode_frame(&scaled)?;
let ts_data = mux.write_pes(pid, &encoded, pts, None)?;

// Pipeline API (requires boilerplate):
let src_node = pipeline.add_node(
    "video_src",
    DynAsyncElement::new_box(SourceAdapter::new(appsrc)),
);
```

**Impact:** Developers bypass the Pipeline API for "quick" implementations, losing benefits of automatic scheduling, isolation, and caps negotiation.

### 1.2 Metadata Attachment to Buffers

**Problem:** No first-class way to attach arbitrary metadata (like KLV packets) to buffers flowing through the pipeline.

**Current state:**
- `Metadata` struct has `sequence`, `timestamp`, `duration`, `flags`
- No extensible metadata mechanism for domain-specific data (KLV, SEI NALUs, closed captions)
- Had to log KLV data separately rather than attach it to video frames

**Desired:**
```rust
// Attach KLV to buffer metadata
buffer.metadata_mut().set_custom("klv", klv_bytes);

// Later, in muxer:
if let Some(klv) = buffer.metadata().get_custom::<Vec<u8>>("klv") {
    self.write_klv_pes(klv, pts)?;
}
```

### 1.3 Muxer Element Model Mismatch

**Problem:** The current element model assumes 1-input-1-output for transforms. Muxers need N-inputs-1-output with synchronized timing.

**Current workarounds:**
- `TsMux` is a standalone struct, not a pipeline element
- Manual timestamp synchronization in examples
- No way to express "wait for both video AND metadata before outputting TS packet"

**What's needed:**
- Proper `Muxer` trait with multiple input pads that the executor understands
- Timestamp-based synchronization ("output when all inputs have data for PTS X")

### 1.4 Encoder/Decoder as Elements

**Problem:** Codecs don't fit cleanly into the `Element` trait because:

1. **Latency:** Encoders buffer multiple frames before outputting (B-frames, lookahead)
2. **1-to-N output:** One input frame may produce 0, 1, or multiple output packets
3. **State:** Need explicit flush/drain at EOS
4. **Configuration:** Complex config that doesn't fit property model

**Current state:**
- `Rav1eEncoder` is a standalone struct with manual `encode_frame()` / `flush()` calls
- Not integrated as a pipeline element
- `OpenH264Encoder` has same issues

### 1.5 No Rust Crate for MPEG-TS Muxing

**Problem:** Had to write MPEG-TS muxer from scratch (~600 lines) because:
- `mpeg2ts-reader` is demux-only
- `va-ts` is incomplete
- No pure-Rust TS muxer crate exists

**Impact:** Maintenance burden, potential bugs in PSI table generation, CRC calculation.

### 1.6 Caps Negotiation Not Used in Practice

**Problem:** While caps negotiation exists, the examples don't use it because:
- Elements return `Caps::any()` by default
- No automatic format conversion insertion
- Video elements don't declare their pixel formats

**Evidence:** `VideoScale` doesn't declare it needs YUV420 input/output:
```rust
fn input_caps(&self) -> Caps {
    Caps::any()  // Should be: Caps::video_raw(PixelFormat::I420, width, height)
}
```

---

## 2. Missing Features

### 2.1 High Priority

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Custom metadata API** | Extensible key-value metadata on buffers | KLV, SEI, closed captions |
| **Codec element wrappers** | `EncoderElement<E>` that handles buffering/flush | Pipeline-integrated encoding |
| **Muxer synchronization** | Multi-input element with PTS-based sync | A/V mux, metadata injection |
| **Buffer pool integration** | Elements request buffers from pipeline pool | Zero-copy through transforms |
| **Format conversion elements** | YUV↔RGB, pixel format conversion | Automatic caps bridging |

### 2.2 Medium Priority

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Pipeline builder DSL** | Fluent API alternative to string parsing | Type-safe pipeline construction |
| **Element properties** | Runtime-adjustable parameters | Bitrate changes, filter coefficients |
| **Latency query** | Ask pipeline for end-to-end latency | Live streaming QoS |
| **Seeking support** | Seek to timestamp, segment handling | File playback |
| **Tagging/Events** | Out-of-band messages (tags, EOS, flush) | Stream control |

### 2.3 Lower Priority

| Feature | Description | Use Case |
|---------|-------------|----------|
| **Dynamic pipeline** | Add/remove elements while running | Adaptive streaming |
| **QoS feedback** | Downstream signals upstream about drops | Frame dropping under load |
| **Clock synchronization** | Master clock for live pipelines | Multi-device sync |
| **Timed metadata** | Schedule metadata for specific PTS | Subtitle insertion |

---

## 3. Architecture Issues

### 3.1 Element Trait Proliferation

**Problem:** Too many overlapping traits:
- `Source`, `AsyncSource`
- `Sink`, `AsyncSink`  
- `Element`, `Transform`, `AsyncTransform`
- `Demuxer`, `Muxer`

Plus adapters for each: `SourceAdapter`, `AsyncSourceAdapter`, etc.

**Recommendation:** Consider consolidating:
```rust
// Single unified trait with associated types
trait PipelineElement {
    type Input;   // () for sources, Buffer for others
    type Output;  // () for sinks, Buffer/Output for others
    
    async fn process(&mut self, input: Self::Input) -> Result<Self::Output>;
}
```

### 3.2 Sync vs Async Confusion

**Problem:** The design says "sync processing, async orchestration" but:
- `AsyncSource`/`AsyncSink` exist for I/O-bound elements
- Elements must choose sync OR async at compile time
- No way to do "mostly sync with occasional async" (e.g., sync decode, async disk write)

**Recommendation:** All elements should be async-capable, with sync elements auto-wrapped:
```rust
// Sync element (common case)
impl Element for MyFilter {
    fn process(&mut self, buf: Buffer) -> Result<Option<Buffer>> { ... }
}

// Executor wraps in spawn_blocking automatically based on is_rt_safe()
```

### 3.3 Buffer Ownership Model

**Problem:** Unclear when buffers are borrowed vs owned:
- `ProduceContext` provides a mutable slice to write into (borrowed)
- `ConsumeContext` provides a read-only reference (borrowed)
- But `Element::process()` takes ownership of `Buffer`
- `Output::Multiple` requires owned `Vec<Buffer>`

**Impact:** Can't do zero-copy forwarding through transforms easily.

**Recommendation:** Consider reference-counted buffer segments that can be cheaply cloned and sub-sliced:
```rust
// Buffer is always a view into refcounted memory
let sub_buffer = buffer.slice(offset, len);  // O(1), shares underlying memory
```

### 3.4 No Pipeline-Level Buffer Pool

**Problem:** Each source creates its own buffers. No way for pipeline to pre-allocate a pool and have elements draw from it.

**Evidence:** In example 31, each frame creates a new `HeapSegment`:
```rust
let segment = Arc::new(HeapSegment::new(yuv_size).expect("alloc"));
```

**Recommendation:** Pipeline should provide buffer allocator:
```rust
impl ProduceContext {
    fn alloc(&mut self, size: usize) -> &mut [u8];  // From pipeline's pool
}
```

### 3.5 Error Handling Inconsistency

**Problem:** Mix of error handling approaches:
- `Result<T>` with `parallax::Error`
- `Option<Buffer>` for "no output"
- `ProduceResult::Eos` for end-of-stream
- `ProduceResult::WouldBlock` for async readiness

**Recommendation:** Unify into a single enum:
```rust
enum ProcessResult<T> {
    Ready(T),
    Eos,
    WouldBlock,
    Error(Error),
}
```

---

## 4. Documentation Gaps

### 4.1 Missing Documentation

- **When to use which adapter** - No guide on `SourceAdapter` vs `AsyncSourceAdapter`
- **Caps negotiation tutorial** - How to declare and use caps properly
- **Muxer/Demuxer patterns** - How to handle multi-pad elements
- **Real-time scheduling** - When to use `Affinity::RealTime`, what `is_rt_safe()` means
- **Process isolation** - How `Isolated` mode actually works, IPC overhead

### 4.2 Example Gaps

- No example showing caps negotiation in action
- No example with encoder as pipeline element (not standalone)
- No example with audio + video muxing
- No example with seeking/segment handling

---

## 5. Recommended Action Items

### Immediate (Before Next Feature Work)

1. **Add custom metadata API** to `Metadata` struct
2. **Create `EncoderElement<E>`** wrapper that handles codec buffering
3. **Document adapter selection** - which adapter for which use case
4. **Add YUV/RGB format caps** to video elements

### Short-term (Next 2-4 Weeks)

5. **Implement muxer synchronization** - PTS-based multi-input handling
6. **Add pipeline buffer pool** - centralized allocation
7. **Create format conversion elements** - colorspace, pixel format
8. **Write caps negotiation tutorial** with working example

### Medium-term (1-2 Months)

9. **Consolidate element traits** - reduce adapter complexity
10. **Add element properties system** - runtime-adjustable parameters
11. **Implement latency query** - end-to-end latency measurement
12. **Add seeking support** - segments, flush events

---

## 6. Positive Observations

Despite the pain points, several aspects of Parallax work well:

1. **Memory model is solid** - `SharedArena`, memfd, cross-process refcounting work correctly
2. **Pipeline DAG** - daggy-based graph prevents cycles, introspection works
3. **Async executor** - `pipeline.run().await` correctly orchestrates elements
4. **AppSrc/AppSink** - External injection/extraction pattern works well
5. **Element isolation** - Process isolation architecture is sound
6. **Pure Rust stack** - rav1e, KLV encoder, MPEG-TS muxer all work without C dependencies

---

## 7. Comparison with GStreamer

| Aspect | GStreamer | Parallax | Assessment |
|--------|-----------|----------|------------|
| Element model | Pad-based, any topology | DAG, typed traits | Parallax simpler but less flexible |
| Caps negotiation | Mature, automatic | Basic, manual | GStreamer ahead |
| Buffer pools | Integrated, negotiated | Per-element | GStreamer ahead |
| Metadata | GstMeta, extensible | Fixed struct | GStreamer ahead |
| Memory safety | C with conventions | Rust ownership | Parallax ahead |
| Process isolation | Via pipewire | Built-in | Parallax ahead |
| Zero-copy IPC | Manual setup | Automatic memfd | Parallax ahead |
| Codec integration | Mature plugins | Basic wrappers | GStreamer ahead |
| Documentation | Extensive | Sparse | GStreamer ahead |

---

## Appendix: Code Examples of Pain Points

### A. Metadata Attachment (Current vs Desired)

```rust
// CURRENT: Metadata logged separately, not attached to buffer
if let Some(ref sensor_meta) = latest_metadata {
    let klv_data = sensor_meta.to_klv();
    println!("klv={} bytes", klv_data.len());  // Lost after this point
}
let buffer = Buffer::new(handle, metadata);  // No KLV attached

// DESIRED: KLV travels with buffer through pipeline
let mut metadata = Metadata::from_sequence(frame_num);
metadata.attach("stanag/klv", sensor_meta.to_klv());
let buffer = Buffer::new(handle, metadata);
// Later in muxer:
if let Some(klv) = buffer.metadata().get::<Vec<u8>>("stanag/klv") {
    self.write_klv_pes(&klv, pts)?;
}
```

### B. Encoder as Element (Current vs Desired)

```rust
// CURRENT: Manual encoder management
let mut encoder = Rav1eEncoder::new(config)?;
for frame in frames {
    let packets = encoder.encode_frame(&frame)?;  // May return 0-N packets
    for pkt in packets {
        mux.write_pes(video_pid, &pkt, pts, dts)?;
    }
}
// Must remember to flush at end
for pkt in encoder.flush()? {
    mux.write_pes(video_pid, &pkt, pts, dts)?;
}

// DESIRED: Encoder as pipeline element
let encoder_node = pipeline.add_node(
    "encoder",
    DynAsyncElement::new_box(EncoderElement::new(Rav1eEncoder::new(config)?)),
);
// Executor handles buffering, flush on EOS automatically
```

### C. Muxer Multi-Input (Current vs Desired)

```rust
// CURRENT: Manual synchronization
let video_ts = mux.write_pes(video_pid, &video_data, pts, None)?;
let klv_ts = mux.write_pes(klv_pid, &klv_data, pts, None)?;
output.extend(video_ts);
output.extend(klv_ts);

// DESIRED: Muxer element with multiple input pads
let mux_node = pipeline.add_node("tsmux", TsMuxElement::new(config));
pipeline.link_pads(encoder_node, "src", mux_node, "video")?;
pipeline.link_pads(klv_node, "src", mux_node, "klv")?;
// Executor synchronizes inputs by PTS automatically
```

---

*Report generated from development session implementing MPEG-TS muxing, KLV/STANAG metadata encoding, video scaling, and AV1 transcoding pipelines.*
