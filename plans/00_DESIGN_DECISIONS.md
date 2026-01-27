# Design Decisions

This document captures key architectural decisions for Parallax, informed by research into [GStreamer](https://gstreamer.freedesktop.org/documentation/), [PipeWire](https://docs.pipewire.org/), and academic literature on media streaming.

---

## Decision 1: Custom Metadata Serialization for IPC

**Question:** Should custom metadata be serializable for IPC?

**Decision:** Yes, custom metadata MUST be serializable for cross-process pipelines.

### Rationale

Both GStreamer and PipeWire support metadata serialization:

- **GStreamer** ([GstMeta docs](https://gstreamer.freedesktop.org/documentation/gstreamer/gstmeta.html)): Since 1.24, GstMeta supports `serialize` and `deserialize` functions. Each metadata type registers these functions via `GstMetaInfo`. For cross-pipeline/cross-process scenarios, explicit serialization is required.

- **PipeWire** ([SPA POD](https://docs.pipewire.org/page_spa_pod.html)): Uses Plain Object Data (POD) encoding - a simple format with 32-bit size, 32-bit type, and content. PODs are designed for zero-allocation parsing in real-time threads. Metadata, data chunks, and buffer structures can all be transported via shared memory.

### Implementation

```rust
/// Metadata values must implement this for IPC support
pub trait MetaSerialize: Send + Sync + 'static {
    /// Serialize to bytes (rkyv recommended)
    fn serialize(&self) -> Vec<u8>;
    
    /// Deserialize from bytes
    fn deserialize(bytes: &[u8]) -> Result<Self> where Self: Sized;
    
    /// Type identifier for registry lookup
    fn type_id() -> &'static str where Self: Sized;
}

// For simple types, provide blanket impl via rkyv
impl<T> MetaSerialize for T 
where 
    T: rkyv::Archive + rkyv::Serialize<...> + Send + Sync + 'static,
{
    fn serialize(&self) -> Vec<u8> {
        rkyv::to_bytes::<_, 256>(self).unwrap().to_vec()
    }
    
    fn deserialize(bytes: &[u8]) -> Result<Self> {
        Ok(rkyv::from_bytes(bytes)?)
    }
}
```

**Key insight from PipeWire:** Pointer types are NOT serialized - only data that can live in shared memory. We should follow this pattern.

---

## Decision 2: Muxer Synchronization Strategy

**Question:** What sync strategy should be default (strict vs loose)?

**Decision:** Default to **live-aware adaptive synchronization**, similar to GStreamer's `GstAggregator`.

### Rationale

From [GStreamer Aggregator documentation](https://gstreamer.freedesktop.org/documentation/base/gstaggregator.html) and [N-to-1 Element design](https://gstreamer.freedesktop.org/documentation/plugin-development/element-types/n-to-one.html):

> "The main noteworthy thing about N-to-1 elements is that each pad is push-based in its own thread, and the N-to-1 element synchronizes those streams by expected-timestamp-based logic. This means it lets all streams wait except for the one that provides the earliest next-expected timestamp."

Key considerations:

1. **Live vs Non-Live:** Live sources have real-time constraints; non-live can wait indefinitely
2. **Sparse Streams:** Metadata/subtitles may have irregular timing (see [GitLab issue #259](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/issues/259))
3. **Latency Budget:** Live pipelines have maximum acceptable latency

### Implementation

```rust
#[derive(Debug, Clone)]
pub struct MuxerSyncConfig {
    /// Sync mode
    pub mode: SyncMode,
    /// Maximum latency to wait for slow streams (live mode)
    pub latency: Duration,
    /// Timeout for sparse streams before outputting without them
    pub sparse_timeout: Duration,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SyncMode {
    /// Adaptive: strict for non-live, latency-bounded for live
    #[default]
    Auto,
    /// Wait for all required streams (may block indefinitely)
    Strict,
    /// Output when primary stream ready, include others if available
    Loose,
    /// Output at fixed intervals regardless of input
    Timed { interval: Duration },
}

impl Default for MuxerSyncConfig {
    fn default() -> Self {
        Self {
            mode: SyncMode::Auto,
            latency: Duration::from_millis(200),  // 200ms default
            sparse_timeout: Duration::from_millis(500),
        }
    }
}
```

**Behavior by mode:**

| Mode | Non-Live | Live |
|------|----------|------|
| Auto | Wait for all required | Wait up to latency budget |
| Strict | Wait for all | Wait for all (may cause underruns) |
| Loose | Output when video ready | Output when video ready |
| Timed | Output every N ms | Output every N ms |

**Sparse stream handling:**
- Mark pads as `sparse` in `PadInfo`
- Sparse pads use `sparse_timeout` instead of blocking
- Send GAP events when sparse stream has no data

---

## Decision 3: Events Channel Architecture

**Question:** Should events use a separate channel from buffers?

**Decision:** **No** - events flow through the same channel as buffers, but with different serialization semantics.

### Rationale

From [GStreamer Events design](https://gstreamer.freedesktop.org/documentation/additional/design/events.html):

> "Events are passed between elements in parallel to the data stream. Some events are serialized with buffers, others are not."

Key distinction:
- **Serialized events** (EOS, SEGMENT, TAGS): Travel in-order with buffers
- **Non-serialized events** (FLUSH_START, FLUSH_STOP): Bypass queues, travel instantly

Using a single channel with `PipelineItem` enum:

```rust
pub enum PipelineItem {
    Buffer(Buffer),
    Event(Event),
}
```

**Advantages:**
1. **Ordering preserved:** Serialized events maintain order relative to buffers
2. **Simpler implementation:** One channel per link, not two
3. **Natural backpressure:** Events participate in flow control

**For non-serialized events (flush):**
- Use a separate **control channel** (unbounded, non-blocking)
- Or use atomic flags that elements check before processing

```rust
pub struct LinkChannels {
    /// Main data channel (buffers + serialized events)
    pub data: kanal::Sender<PipelineItem>,
    /// Control channel for flush/interrupt (non-blocking)
    pub control: kanal::Sender<ControlSignal>,
}

pub enum ControlSignal {
    FlushStart,
    FlushStop { reset_time: bool },
    Interrupt,
}
```

---

## Decision 4: Backward Compatibility

**Question:** Should we keep backward-compatible aliases during refactoring?

**Decision:** **No** - break compatibility freely since we're pre-production.

### Rationale

The codebase hasn't reached production. Maintaining compatibility:
- Adds complexity
- Slows development
- Creates confusion (old vs new API)

### Approach

1. **Make breaking changes directly** - no deprecation period
2. **Update all examples** when changing APIs
3. **Document changes** in CHANGELOG.md
4. **Single major refactoring** in Plan 05 to minimize churn

---

## Decision 5: Element Trait Design

**Question:** Should we keep separate sync/async traits or unify?

**Decision:** **Unified async trait** with blanket implementations for sync elements.

### Rationale

From PipeWire's design: everything is fundamentally async (event-loop driven), but most processing is sync within that model.

```rust
// Single core trait (async)
#[async_trait]
pub trait PipelineElement: Send + Sync {
    async fn process(&mut self, input: Option<PipelineItem>) -> Result<ProcessOutput>;
}

// Sync convenience trait with blanket impl
pub trait SyncElement: Send + Sync {
    fn process_sync(&mut self, input: Option<PipelineItem>) -> Result<ProcessOutput>;
}

#[async_trait]
impl<T: SyncElement + 'static> PipelineElement for T {
    async fn process(&mut self, input: Option<PipelineItem>) -> Result<ProcessOutput> {
        // For CPU-bound work, use spawn_blocking
        if self.execution_hints().processing == ProcessingHint::CpuBound {
            tokio::task::block_in_place(|| self.process_sync(input))
        } else {
            self.process_sync(input)
        }
    }
}
```

---

## Decision 6: Buffer Pool Negotiation

**Question:** How should buffer pools be negotiated between elements?

**Decision:** Pipeline-level pool with per-link size negotiation.

### Rationale

From [PipeWire Buffers](https://docs.pipewire.org/page_spa_buffer.html):

> "The metadata memory, the data and chunks can be directly transported in shared memory while the buffer structure can be negotiated separately."

And from GStreamer: pools are negotiated via ALLOCATION query.

### Implementation

```rust
impl Pipeline {
    /// After caps negotiation, create pools for each link
    pub fn allocate_pools(&mut self) -> Result<()> {
        for link in self.links() {
            let buffer_size = link.negotiated_format
                .map(|f| f.buffer_size())
                .unwrap_or(DEFAULT_BUFFER_SIZE);
            
            let pool = FixedSizePool::new(buffer_size, DEFAULT_POOL_SIZE)?;
            self.set_link_pool(link.id, pool)?;
        }
        Ok(())
    }
}
```

---

## Decision 7: Metadata Storage Format

**Question:** HashMap vs SmallVec for custom metadata?

**Decision:** Start with **HashMap**, optimize later if profiling shows need.

### Rationale

- HashMap: O(1) lookup, simple API, good for arbitrary keys
- SmallVec: Better cache locality, but requires known set of keys

PipeWire uses a fixed set of metadata types (spa_meta_*), but we need extensibility for domain-specific data (KLV, SEI, etc.).

**Optimization path if needed:**
1. Profile real workloads
2. If hot, use `SmallVec<[MetaEntry; 4]>` with linear scan for small N
3. Or use perfect hashing for known keys + overflow HashMap

---

## Decision 8: Timestamp Representation

**Question:** What timestamp format for synchronization?

**Decision:** Use **nanoseconds as i64** for all timestamps, following GStreamer's model.

### Rationale

From [GStreamer Synchronisation](https://gstreamer.freedesktop.org/documentation/additional/design/synchronisation.html):

> "Running time is calculated from timestamps and segment information"

Using nanoseconds:
- Sufficient precision (sub-microsecond)
- i64 range: ±292 years - plenty for media
- Compatible with `std::time::Duration`
- Negative values useful for pre-roll

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClockTime(i64);  // Nanoseconds

impl ClockTime {
    pub const ZERO: Self = Self(0);
    pub const NONE: Self = Self(i64::MIN);  // Invalid/unset
    
    pub fn from_nanos(ns: i64) -> Self { Self(ns) }
    pub fn from_micros(us: i64) -> Self { Self(us * 1_000) }
    pub fn from_millis(ms: i64) -> Self { Self(ms * 1_000_000) }
    pub fn from_secs(s: i64) -> Self { Self(s * 1_000_000_000) }
    
    pub fn nanos(&self) -> i64 { self.0 }
    pub fn is_valid(&self) -> bool { self.0 != i64::MIN }
}
```

---

## Summary Table

| # | Question | Decision |
|---|----------|----------|
| 1 | Metadata serialization for IPC? | Yes, via rkyv with MetaSerialize trait |
| 2 | Default muxer sync strategy? | Auto (adaptive based on live/non-live) |
| 3 | Separate events channel? | No, unified channel + control signals for flush |
| 4 | Backward compatibility? | No, break freely (pre-production) |
| 5 | Sync/async element traits? | Unified async with sync blanket impl |
| 6 | Buffer pool negotiation? | Pipeline-level, per-link size negotiation |
| 7 | Metadata storage format? | HashMap (optimize later if needed) |
| 8 | Timestamp format? | i64 nanoseconds (ClockTime) |

---

## References

### GStreamer
- [GstMeta](https://gstreamer.freedesktop.org/documentation/gstreamer/gstmeta.html)
- [GstAggregator](https://gstreamer.freedesktop.org/documentation/base/gstaggregator.html)
- [Events Design](https://gstreamer.freedesktop.org/documentation/additional/design/events.html)
- [Synchronisation](https://gstreamer.freedesktop.org/documentation/additional/design/synchronisation.html)
- [N-to-1 Elements](https://gstreamer.freedesktop.org/documentation/plugin-development/element-types/n-to-one.html)

### PipeWire
- [SPA POD](https://docs.pipewire.org/page_spa_pod.html)
- [SPA Buffers](https://docs.pipewire.org/page_spa_buffer.html)
- [Native Protocol](https://docs.pipewire.org/page_native_protocol.html)

### Research
- [Audio-Video Synchronization using Timestamps](https://www.researchgate.net/publication/4360047_Using_timestamp_to_realize_audio-video_synchronization_in_Real-Time_streaming_media_transmission)
- [End-to-End Video Streaming Survey](https://arxiv.org/abs/2403.05192)
- [Adaptive Content-Based Synchronization](https://onlinelibrary.wiley.com/doi/10.1155/2011/914062)
