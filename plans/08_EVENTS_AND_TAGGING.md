# Plan 08: Events and Tagging System

**Priority:** Medium  
**Effort:** Medium (1 week)  
**Dependencies:** Plan 01 (Custom Metadata API)  
**Addresses:** Missing Features 2.5 (Tagging/Events), 2.4 (Seeking support)

---

## Problem Statement

Currently, pipelines can only pass data buffers. There's no mechanism for:

1. **Out-of-band events:** EOS, flush, segment, stream-start
2. **Tags/metadata:** Stream tags (title, artist, codec info)
3. **Seeking:** Jump to a specific timestamp
4. **Stream changes:** Dynamic format changes, discontinuities

These are essential for real-world media pipelines.

---

## Proposed Solution

Implement an event system similar to GStreamer's, with:
1. **Downstream events:** Flow with data (EOS, segment, tags)
2. **Upstream events:** Flow against data (seek, QoS)
3. **Serialized events:** Respect buffer ordering
4. **Non-serialized events:** Bypass queues (flush)

---

## Design

### Event Types

```rust
/// Events that flow through the pipeline
#[derive(Debug, Clone)]
pub enum Event {
    // ========== Downstream Events ==========
    
    /// Start of a new stream
    StreamStart(StreamStartEvent),
    
    /// Defines a playback segment (timeline)
    Segment(SegmentEvent),
    
    /// Stream tags (metadata)
    Tags(TagsEvent),
    
    /// End of stream
    Eos,
    
    /// Caps changed mid-stream
    CapsChanged(CapsChangedEvent),
    
    /// Gap in data (silence, black frames)
    Gap(GapEvent),
    
    // ========== Upstream Events ==========
    
    /// Seek request
    Seek(SeekEvent),
    
    /// Quality of Service feedback
    Qos(QosEvent),
    
    /// Request for latency info
    LatencyQuery,
    
    // ========== Bidirectional Events ==========
    
    /// Flush start (discard buffered data)
    FlushStart,
    
    /// Flush stop (resume normal operation)
    FlushStop(FlushStopEvent),
    
    /// Custom application event
    Custom(CustomEvent),
}

impl Event {
    /// Is this a downstream event?
    pub fn is_downstream(&self) -> bool {
        matches!(self, 
            Event::StreamStart(_) | Event::Segment(_) | Event::Tags(_) |
            Event::Eos | Event::CapsChanged(_) | Event::Gap(_))
    }
    
    /// Is this an upstream event?
    pub fn is_upstream(&self) -> bool {
        matches!(self, Event::Seek(_) | Event::Qos(_) | Event::LatencyQuery)
    }
    
    /// Should this event be serialized with buffers?
    pub fn is_serialized(&self) -> bool {
        !matches!(self, Event::FlushStart | Event::FlushStop(_))
    }
}
```

### Specific Event Types

```rust
/// Stream start event - begins a new logical stream
#[derive(Debug, Clone)]
pub struct StreamStartEvent {
    /// Unique stream identifier
    pub stream_id: String,
    /// Stream flags
    pub flags: StreamFlags,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct StreamFlags: u32 {
        const NONE = 0;
        const SPARSE = 1 << 0;      // Sparse stream (subtitles)
        const SELECT = 1 << 1;       // Should be selected by default
        const UNSELECT = 1 << 2;     // Should not be selected
    }
}

/// Segment event - defines the playback timeline
#[derive(Debug, Clone)]
pub struct SegmentEvent {
    /// Segment format (time, bytes, etc.)
    pub format: SegmentFormat,
    /// Segment start position
    pub start: i64,
    /// Segment stop position (-1 for none)
    pub stop: i64,
    /// Current position in segment
    pub position: i64,
    /// Playback rate (1.0 = normal)
    pub rate: f64,
    /// Applied rate (for trick modes)
    pub applied_rate: f64,
    /// Base time for running time calculation
    pub base: i64,
    /// Segment flags
    pub flags: SegmentFlags,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentFormat {
    Time,       // Nanoseconds
    Bytes,      // Byte offset
    Default,    // Element-specific
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct SegmentFlags: u32 {
        const NONE = 0;
        const RESET = 1 << 0;        // Reset running time
        const SKIP = 1 << 1;         // Skip to position
        const SEGMENT = 1 << 2;      // Segment seek
    }
}

/// Tags event - stream metadata
#[derive(Debug, Clone)]
pub struct TagsEvent {
    /// Tag list
    pub tags: TagList,
    /// Merge mode for existing tags
    pub mode: TagMergeMode,
}

#[derive(Debug, Clone, Default)]
pub struct TagList {
    tags: HashMap<String, TagValue>,
}

#[derive(Debug, Clone)]
pub enum TagValue {
    String(String),
    UInt(u64),
    Double(f64),
    DateTime(String),  // ISO 8601
    Binary(Vec<u8>),
    List(Vec<TagValue>),
}

impl TagList {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<TagValue>) {
        self.tags.insert(key.into(), value.into());
    }
    
    pub fn get(&self, key: &str) -> Option<&TagValue> {
        self.tags.get(key)
    }
    
    // Common tag accessors
    pub fn title(&self) -> Option<&str> {
        self.get("title").and_then(|v| v.as_string())
    }
    
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.set("title", TagValue::String(title.into()));
    }
    
    pub fn duration(&self) -> Option<ClockTime> {
        self.get("duration").and_then(|v| v.as_uint()).map(ClockTime::from_nanos)
    }
    
    pub fn set_duration(&mut self, duration: ClockTime) {
        self.set("duration", TagValue::UInt(duration.nanos()));
    }
    
    // More common tags: artist, album, codec, bitrate, etc.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TagMergeMode {
    /// Replace all existing tags
    Replace,
    /// Keep existing, add new
    #[default]
    Append,
    /// Add new, keep existing if conflict
    Keep,
    /// Replace only matching keys
    ReplaceMatching,
}

/// Seek event - request to jump to position
#[derive(Debug, Clone)]
pub struct SeekEvent {
    /// Seek rate (1.0 = normal, 2.0 = 2x speed)
    pub rate: f64,
    /// Format of start/stop positions
    pub format: SegmentFormat,
    /// Seek flags
    pub flags: SeekFlags,
    /// Start type and position
    pub start: SeekPosition,
    /// Stop type and position
    pub stop: SeekPosition,
}

#[derive(Debug, Clone)]
pub struct SeekPosition {
    pub seek_type: SeekType,
    pub position: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekType {
    /// Don't change this position
    None,
    /// Absolute position
    Set,
    /// Relative to current
    Current,
    /// Relative to end
    End,
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct SeekFlags: u32 {
        const NONE = 0;
        const FLUSH = 1 << 0;        // Flush pipeline before seek
        const ACCURATE = 1 << 1;     // Seek to exact position
        const KEY_UNIT = 1 << 2;     // Seek to nearest keyframe
        const SNAP_BEFORE = 1 << 4;  // Snap to position before
        const SNAP_AFTER = 1 << 5;   // Snap to position after
    }
}

/// QoS event - quality feedback
#[derive(Debug, Clone)]
pub struct QosEvent {
    /// QoS type
    pub qos_type: QosType,
    /// Proportion of buffers dropped (0.0 - 1.0)
    pub proportion: f64,
    /// Difference between expected and actual time
    pub diff: ClockTime,
    /// Timestamp of the buffer that caused this
    pub timestamp: ClockTime,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QosType {
    /// Underflow (not enough data)
    Underflow,
    /// Overflow (too much data)
    Overflow,
    /// Throttle (rate too high)
    Throttle,
}

/// Gap event - no data for a period
#[derive(Debug, Clone)]
pub struct GapEvent {
    /// Start of gap
    pub timestamp: ClockTime,
    /// Duration of gap
    pub duration: ClockTime,
}

/// Flush stop event
#[derive(Debug, Clone)]
pub struct FlushStopEvent {
    /// Reset running time?
    pub reset_time: bool,
}

/// Caps changed mid-stream
#[derive(Debug, Clone)]
pub struct CapsChangedEvent {
    pub new_caps: MediaCaps,
}

/// Custom application event
#[derive(Debug, Clone)]
pub struct CustomEvent {
    pub name: String,
    pub data: HashMap<String, TagValue>,
}
```

### Pipeline Item (Buffer or Event)

```rust
/// Item that flows through the pipeline
#[derive(Debug)]
pub enum PipelineItem {
    Buffer(Buffer),
    Event(Event),
}

impl From<Buffer> for PipelineItem {
    fn from(buffer: Buffer) -> Self {
        PipelineItem::Buffer(buffer)
    }
}

impl From<Event> for PipelineItem {
    fn from(event: Event) -> Self {
        PipelineItem::Event(event)
    }
}
```

### Element Event Handling

Update element traits to handle events:

```rust
pub trait PipelineElement: Send + Sync {
    /// Process a buffer or event.
    async fn process(&mut self, input: Option<PipelineItem>) -> Result<ProcessOutput>;
    
    /// Handle an upstream event (seek, QoS).
    /// Return true if handled, false to propagate upstream.
    fn handle_upstream_event(&mut self, event: &Event) -> bool {
        false  // Default: propagate
    }
    
    /// Handle a downstream event before processing.
    /// Return Some(event) to forward, None to consume.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)  // Default: forward
    }
}
```

### ProcessOutput with Events

```rust
pub enum ProcessOutput {
    None,
    One(Buffer),
    Many(Vec<Buffer>),
    Eos,
    Pending,
    
    /// Output an event instead of a buffer
    Event(Event),
    
    /// Output buffer(s) followed by event
    BuffersThenEvent(Vec<Buffer>, Event),
}
```

---

## Event Flow

### Downstream Events

```
Source                Transform              Sink
   │                     │                     │
   │─── StreamStart ────▶│─── StreamStart ────▶│
   │─── Segment ────────▶│─── Segment ────────▶│
   │─── Tags ───────────▶│─── Tags ───────────▶│
   │─── Buffer ─────────▶│─── Buffer ─────────▶│
   │─── Buffer ─────────▶│─── Buffer ─────────▶│
   │─── EOS ────────────▶│─── EOS ────────────▶│
```

### Upstream Events

```
Source                Transform              Sink
   │                     │                     │
   │◀─── Seek ──────────│◀─── Seek ──────────│
   │                     │                     │
   │─── FlushStart ─────▶│─── FlushStart ─────▶│
   │─── FlushStop ──────▶│─── FlushStop ──────▶│
   │─── Segment ────────▶│─── Segment ────────▶│
   │─── Buffer ─────────▶│─── Buffer ─────────▶│
```

### Flush Handling

```rust
impl MyDecoder {
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        match &event {
            Event::FlushStart => {
                // Discard all buffered data
                self.pending_frames.clear();
                Some(event)
            }
            Event::FlushStop(e) => {
                // Reset decoder state
                self.reset_state();
                if e.reset_time {
                    self.last_pts = ClockTime::ZERO;
                }
                Some(event)
            }
            _ => Some(event),
        }
    }
}
```

---

## Seeking Implementation

### Seek Flow

1. **Application sends seek event upstream**
2. **Source receives seek, flushes pipeline**
3. **Source seeks to new position**
4. **Source sends new segment event**
5. **Pipeline resumes with new data**

```rust
impl FileSource {
    fn handle_upstream_event(&mut self, event: &Event) -> bool {
        match event {
            Event::Seek(seek) => {
                // Send flush start
                self.send_event(Event::FlushStart);
                
                // Perform actual seek
                let position = match seek.format {
                    SegmentFormat::Time => self.time_to_byte(seek.start.position),
                    SegmentFormat::Bytes => seek.start.position,
                    _ => return false,
                };
                
                self.file.seek(SeekFrom::Start(position as u64)).ok();
                
                // Send flush stop
                self.send_event(Event::FlushStop(FlushStopEvent {
                    reset_time: seek.flags.contains(SeekFlags::FLUSH),
                }));
                
                // Send new segment
                self.send_event(Event::Segment(SegmentEvent {
                    format: seek.format,
                    start: seek.start.position,
                    stop: seek.stop.position,
                    position: seek.start.position,
                    rate: seek.rate,
                    applied_rate: 1.0,
                    base: 0,
                    flags: SegmentFlags::RESET,
                }));
                
                true  // Handled
            }
            _ => false,
        }
    }
}
```

---

## Implementation Steps

### Step 1: Define Event Types

**File:** `src/event/mod.rs`

- `Event` enum
- All specific event types
- `PipelineItem` enum

### Step 2: Define Tag System

**File:** `src/event/tags.rs`

- `TagList`
- `TagValue`
- `TagMergeMode`
- Common tag constants

### Step 3: Update Element Traits

**File:** `src/element/traits.rs`

- Add `handle_upstream_event()`
- Add `handle_downstream_event()`
- Update `ProcessOutput` with events

### Step 4: Update Executor

**File:** `src/pipeline/unified_executor.rs`

- Handle `PipelineItem::Event`
- Route upstream vs downstream events
- Handle flush specially (non-serialized)

### Step 5: Update Built-in Elements

- `FileSrc`: Handle seek, send segments
- `Queue`: Handle flush (clear buffer)
- `Demuxer`: Forward events per stream
- `Muxer`: Merge events from inputs

### Step 6: Create Examples

**File:** `examples/37_events_tags.rs`

```rust
// Demonstrate tags
let mut tags = TagList::new();
tags.set_title("My Video");
tags.set("artist", "Claude");
tags.set("duration", TagValue::UInt(60_000_000_000));  // 60 seconds

pipeline.send_event(Event::Tags(TagsEvent {
    tags,
    mode: TagMergeMode::Append,
}));
```

**File:** `examples/38_seeking.rs`

```rust
// Seek to 30 seconds
pipeline.send_upstream_event(Event::Seek(SeekEvent {
    rate: 1.0,
    format: SegmentFormat::Time,
    flags: SeekFlags::FLUSH | SeekFlags::KEY_UNIT,
    start: SeekPosition {
        seek_type: SeekType::Set,
        position: 30_000_000_000,  // 30 seconds in nanos
    },
    stop: SeekPosition {
        seek_type: SeekType::None,
        position: -1,
    },
}));
```

---

## Validation Criteria

- [ ] `Event` enum with all event types
- [ ] `TagList` with common tag accessors
- [ ] `PipelineItem` enum (Buffer | Event)
- [ ] Elements can handle events
- [ ] Executor routes events correctly
- [ ] Flush interrupts buffer flow
- [ ] Seek works with FileSource
- [ ] Tags propagate through pipeline
- [ ] Example demonstrates events
- [ ] All existing tests pass

---

## Future Enhancements

1. **Sticky events:** Events that stick to pads (segment, tags)
2. **Event queries:** Ask elements about capabilities
3. **Navigation events:** DVD menus, interactive content
4. **Toc events:** Table of contents (chapters)
5. **Protection events:** DRM information

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `src/event/mod.rs` | New: Event enum, PipelineItem |
| `src/event/tags.rs` | New: TagList, TagValue |
| `src/event/seek.rs` | New: SeekEvent, SegmentEvent |
| `src/element/traits.rs` | Event handling methods |
| `src/pipeline/unified_executor.rs` | Event routing |
| `src/elements/io/filesrc.rs` | Seek handling |
| `src/elements/flow/queue.rs` | Flush handling |
| `examples/37_events_tags.rs` | New example |
| `examples/38_seeking.rs` | New example |
