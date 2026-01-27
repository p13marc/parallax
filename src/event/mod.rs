//! Events and tagging system for pipelines.
//!
//! This module provides out-of-band events that flow through pipelines alongside
//! buffers, enabling features like seeking, flushing, stream metadata (tags),
//! and end-of-stream signaling.
//!
//! # Event Types
//!
//! Events are categorized by their flow direction:
//!
//! - **Downstream events**: Flow with data (EOS, segment, tags, stream-start)
//! - **Upstream events**: Flow against data (seek, QoS)
//! - **Bidirectional events**: Can flow either way (flush)
//!
//! # Serialization
//!
//! Events can be either:
//!
//! - **Serialized**: Respect buffer ordering (most events)
//! - **Non-serialized**: Bypass queues for immediate effect (flush)
//!
//! # Example
//!
//! ```rust
//! use parallax::event::{Event, TagList, TagsEvent, TagMergeMode};
//!
//! // Create stream tags
//! let mut tags = TagList::new();
//! tags.set_title("My Video");
//! tags.set("artist", "Parallax");
//!
//! let event = Event::Tags(TagsEvent {
//!     tags,
//!     mode: TagMergeMode::Append,
//! });
//!
//! assert!(event.is_downstream());
//! assert!(event.is_serialized());
//! ```

mod tags;

pub use tags::{TagList, TagMergeMode, TagValue};

use crate::buffer::Buffer;
use crate::clock::ClockTime;
use crate::format::MediaCaps;

// ============================================================================
// Event Enum
// ============================================================================

/// Events that flow through the pipeline.
///
/// Events provide out-of-band signaling for control flow, metadata,
/// and synchronization. They can flow downstream (with data) or
/// upstream (against data flow).
#[derive(Debug, Clone)]
pub enum Event {
    // ========== Downstream Events ==========
    /// Start of a new stream.
    StreamStart(StreamStartEvent),

    /// Defines a playback segment (timeline).
    Segment(SegmentEvent),

    /// Stream tags (metadata like title, artist, duration).
    Tags(TagsEvent),

    /// End of stream - no more data will be produced.
    Eos,

    /// Caps changed mid-stream.
    CapsChanged(CapsChangedEvent),

    /// Gap in data (silence, black frames).
    Gap(GapEvent),

    // ========== Upstream Events ==========
    /// Seek request.
    Seek(SeekEvent),

    /// Quality of Service feedback.
    Qos(QosEvent),

    /// Request for latency info.
    LatencyQuery,

    // ========== Bidirectional Events ==========
    /// Flush start - immediately discard buffered data.
    FlushStart,

    /// Flush stop - resume normal operation.
    FlushStop(FlushStopEvent),

    /// Custom application event.
    Custom(CustomEvent),
}

impl Event {
    /// Check if this is a downstream event (flows with data).
    pub fn is_downstream(&self) -> bool {
        matches!(
            self,
            Event::StreamStart(_)
                | Event::Segment(_)
                | Event::Tags(_)
                | Event::Eos
                | Event::CapsChanged(_)
                | Event::Gap(_)
        )
    }

    /// Check if this is an upstream event (flows against data).
    pub fn is_upstream(&self) -> bool {
        matches!(self, Event::Seek(_) | Event::Qos(_) | Event::LatencyQuery)
    }

    /// Check if this is a bidirectional event.
    pub fn is_bidirectional(&self) -> bool {
        matches!(
            self,
            Event::FlushStart | Event::FlushStop(_) | Event::Custom(_)
        )
    }

    /// Check if this event should be serialized with buffers.
    ///
    /// Serialized events respect buffer ordering and go through queues.
    /// Non-serialized events (flush) bypass queues for immediate effect.
    pub fn is_serialized(&self) -> bool {
        !matches!(self, Event::FlushStart | Event::FlushStop(_))
    }

    /// Get a human-readable name for this event type.
    pub fn name(&self) -> &str {
        match self {
            Event::StreamStart(_) => "stream-start",
            Event::Segment(_) => "segment",
            Event::Tags(_) => "tags",
            Event::Eos => "eos",
            Event::CapsChanged(_) => "caps-changed",
            Event::Gap(_) => "gap",
            Event::Seek(_) => "seek",
            Event::Qos(_) => "qos",
            Event::LatencyQuery => "latency-query",
            Event::FlushStart => "flush-start",
            Event::FlushStop(_) => "flush-stop",
            Event::Custom(c) => &c.name,
        }
    }
}

// ============================================================================
// Stream Start Event
// ============================================================================

/// Stream start event - begins a new logical stream.
///
/// This event is sent at the beginning of a stream to establish
/// stream identity and properties.
#[derive(Debug, Clone)]
pub struct StreamStartEvent {
    /// Unique stream identifier.
    pub stream_id: String,
    /// Stream flags.
    pub flags: StreamFlags,
}

impl StreamStartEvent {
    /// Create a new stream start event.
    pub fn new(stream_id: impl Into<String>) -> Self {
        Self {
            stream_id: stream_id.into(),
            flags: StreamFlags::empty(),
        }
    }

    /// Create with flags.
    pub fn with_flags(stream_id: impl Into<String>, flags: StreamFlags) -> Self {
        Self {
            stream_id: stream_id.into(),
            flags,
        }
    }
}

/// Flags for stream start events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct StreamFlags(u32);

impl StreamFlags {
    /// No special flags.
    pub const NONE: Self = Self(0);
    /// Sparse stream (e.g., subtitles).
    pub const SPARSE: Self = Self(1 << 0);
    /// Should be selected by default.
    pub const SELECT: Self = Self(1 << 1);
    /// Should not be selected by default.
    pub const UNSELECT: Self = Self(1 << 2);

    /// Create empty flags.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Check if empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if contains a flag.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Union of flags.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// ============================================================================
// Segment Event
// ============================================================================

/// Segment event - defines the playback timeline.
///
/// A segment describes a contiguous range of time or bytes in the stream.
/// It's used to synchronize playback and handle seeking.
#[derive(Debug, Clone)]
pub struct SegmentEvent {
    /// Segment format (time, bytes, etc.).
    pub format: SegmentFormat,
    /// Segment start position.
    pub start: i64,
    /// Segment stop position (-1 for none/unlimited).
    pub stop: i64,
    /// Current position in segment.
    pub position: i64,
    /// Playback rate (1.0 = normal speed).
    pub rate: f64,
    /// Applied rate for trick modes.
    pub applied_rate: f64,
    /// Base time for running time calculation.
    pub base: i64,
    /// Segment flags.
    pub flags: SegmentFlags,
}

impl SegmentEvent {
    /// Create a new time-based segment.
    pub fn new_time(start: ClockTime, stop: Option<ClockTime>) -> Self {
        Self {
            format: SegmentFormat::Time,
            start: start.nanos() as i64,
            stop: stop.map(|t| t.nanos() as i64).unwrap_or(-1),
            position: start.nanos() as i64,
            rate: 1.0,
            applied_rate: 1.0,
            base: 0,
            flags: SegmentFlags::empty(),
        }
    }

    /// Create a new byte-based segment.
    pub fn new_bytes(start: u64, stop: Option<u64>) -> Self {
        Self {
            format: SegmentFormat::Bytes,
            start: start as i64,
            stop: stop.map(|s| s as i64).unwrap_or(-1),
            position: start as i64,
            rate: 1.0,
            applied_rate: 1.0,
            base: 0,
            flags: SegmentFlags::empty(),
        }
    }

    /// Set the playback rate.
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = rate;
        self
    }

    /// Set segment flags.
    pub fn with_flags(mut self, flags: SegmentFlags) -> Self {
        self.flags = flags;
        self
    }
}

impl Default for SegmentEvent {
    fn default() -> Self {
        Self::new_time(ClockTime::ZERO, None)
    }
}

/// Format of segment positions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SegmentFormat {
    /// Positions in nanoseconds.
    #[default]
    Time,
    /// Positions in bytes.
    Bytes,
    /// Element-specific default format.
    Default,
}

/// Flags for segment events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SegmentFlags(u32);

impl SegmentFlags {
    /// No special flags.
    pub const NONE: Self = Self(0);
    /// Reset running time to 0.
    pub const RESET: Self = Self(1 << 0);
    /// Skip to position (don't play intermediate data).
    pub const SKIP: Self = Self(1 << 1);
    /// This is a segment seek.
    pub const SEGMENT: Self = Self(1 << 2);

    /// Create empty flags.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Check if empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if contains a flag.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Union of flags.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// ============================================================================
// Tags Event
// ============================================================================

/// Tags event - stream metadata.
///
/// Contains metadata about the stream like title, artist, codec info, etc.
#[derive(Debug, Clone)]
pub struct TagsEvent {
    /// The tag list.
    pub tags: TagList,
    /// How to merge with existing tags.
    pub mode: TagMergeMode,
}

impl TagsEvent {
    /// Create a new tags event.
    pub fn new(tags: TagList) -> Self {
        Self {
            tags,
            mode: TagMergeMode::default(),
        }
    }

    /// Create with a specific merge mode.
    pub fn with_mode(tags: TagList, mode: TagMergeMode) -> Self {
        Self { tags, mode }
    }
}

// ============================================================================
// Seek Event
// ============================================================================

/// Seek event - request to jump to a position.
///
/// Sent upstream to request that sources seek to a new position.
#[derive(Debug, Clone)]
pub struct SeekEvent {
    /// Seek rate (1.0 = normal, 2.0 = 2x speed, -1.0 = reverse).
    pub rate: f64,
    /// Format of start/stop positions.
    pub format: SegmentFormat,
    /// Seek flags.
    pub flags: SeekFlags,
    /// Start position.
    pub start: SeekPosition,
    /// Stop position.
    pub stop: SeekPosition,
}

impl SeekEvent {
    /// Create a simple time-based seek to a position.
    pub fn new_time(position: ClockTime) -> Self {
        Self {
            rate: 1.0,
            format: SegmentFormat::Time,
            flags: SeekFlags::FLUSH.union(SeekFlags::KEY_UNIT),
            start: SeekPosition {
                seek_type: SeekType::Set,
                position: position.nanos() as i64,
            },
            stop: SeekPosition {
                seek_type: SeekType::None,
                position: -1,
            },
        }
    }

    /// Create a byte-based seek.
    pub fn new_bytes(position: u64) -> Self {
        Self {
            rate: 1.0,
            format: SegmentFormat::Bytes,
            flags: SeekFlags::FLUSH,
            start: SeekPosition {
                seek_type: SeekType::Set,
                position: position as i64,
            },
            stop: SeekPosition {
                seek_type: SeekType::None,
                position: -1,
            },
        }
    }

    /// Set the seek rate.
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = rate;
        self
    }

    /// Set seek flags.
    pub fn with_flags(mut self, flags: SeekFlags) -> Self {
        self.flags = flags;
        self
    }

    /// Set the stop position.
    pub fn with_stop(mut self, stop: SeekPosition) -> Self {
        self.stop = stop;
        self
    }
}

/// Position in a seek event.
#[derive(Debug, Clone)]
pub struct SeekPosition {
    /// Type of seek.
    pub seek_type: SeekType,
    /// Position value (interpretation depends on seek_type).
    pub position: i64,
}

impl SeekPosition {
    /// Create an absolute position.
    pub fn set(position: i64) -> Self {
        Self {
            seek_type: SeekType::Set,
            position,
        }
    }

    /// Create a relative position (from current).
    pub fn current(offset: i64) -> Self {
        Self {
            seek_type: SeekType::Current,
            position: offset,
        }
    }

    /// Create a position relative to end.
    pub fn end(offset: i64) -> Self {
        Self {
            seek_type: SeekType::End,
            position: offset,
        }
    }

    /// No position change.
    pub fn none() -> Self {
        Self {
            seek_type: SeekType::None,
            position: -1,
        }
    }
}

/// Type of seek position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SeekType {
    /// Don't change this position.
    #[default]
    None,
    /// Absolute position.
    Set,
    /// Relative to current position.
    Current,
    /// Relative to end.
    End,
}

/// Flags for seek events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SeekFlags(u32);

impl SeekFlags {
    /// No special flags.
    pub const NONE: Self = Self(0);
    /// Flush pipeline before seek.
    pub const FLUSH: Self = Self(1 << 0);
    /// Seek to exact position (may be slower).
    pub const ACCURATE: Self = Self(1 << 1);
    /// Seek to nearest keyframe.
    pub const KEY_UNIT: Self = Self(1 << 2);
    /// Snap to position before target.
    pub const SNAP_BEFORE: Self = Self(1 << 4);
    /// Snap to position after target.
    pub const SNAP_AFTER: Self = Self(1 << 5);

    /// Create empty flags.
    pub const fn empty() -> Self {
        Self(0)
    }

    /// Check if empty.
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Check if contains a flag.
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0
    }

    /// Union of flags.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }
}

// ============================================================================
// QoS Event
// ============================================================================

/// Quality of Service event - feedback about processing performance.
///
/// Sent upstream to indicate that elements are having trouble
/// keeping up with the data rate.
#[derive(Debug, Clone)]
pub struct QosEvent {
    /// Type of QoS issue.
    pub qos_type: QosType,
    /// Proportion of frames being dropped (0.0 - 1.0).
    pub proportion: f64,
    /// Difference between expected and actual processing time.
    pub diff: ClockTime,
    /// Timestamp of the problematic buffer.
    pub timestamp: ClockTime,
}

impl QosEvent {
    /// Create a new QoS event.
    pub fn new(qos_type: QosType, proportion: f64, diff: ClockTime, timestamp: ClockTime) -> Self {
        Self {
            qos_type,
            proportion,
            diff,
            timestamp,
        }
    }
}

/// Type of QoS issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QosType {
    /// Not enough data (underflow).
    Underflow,
    /// Too much data (overflow).
    Overflow,
    /// Data rate too high (throttling needed).
    Throttle,
}

// ============================================================================
// Gap Event
// ============================================================================

/// Gap event - indicates no data for a period.
///
/// Used to signal silence in audio or black frames in video.
#[derive(Debug, Clone)]
pub struct GapEvent {
    /// Start of the gap.
    pub timestamp: ClockTime,
    /// Duration of the gap.
    pub duration: ClockTime,
}

impl GapEvent {
    /// Create a new gap event.
    pub fn new(timestamp: ClockTime, duration: ClockTime) -> Self {
        Self {
            timestamp,
            duration,
        }
    }
}

// ============================================================================
// Caps Changed Event
// ============================================================================

/// Caps changed event - format changed mid-stream.
///
/// Indicates that the stream format has changed and downstream
/// elements need to reconfigure.
#[derive(Debug, Clone)]
pub struct CapsChangedEvent {
    /// The new caps.
    pub new_caps: MediaCaps,
}

impl CapsChangedEvent {
    /// Create a new caps changed event.
    pub fn new(new_caps: MediaCaps) -> Self {
        Self { new_caps }
    }
}

// ============================================================================
// Flush Stop Event
// ============================================================================

/// Flush stop event - resume normal operation after flush.
#[derive(Debug, Clone)]
pub struct FlushStopEvent {
    /// Whether to reset running time to 0.
    pub reset_time: bool,
}

impl FlushStopEvent {
    /// Create a new flush stop event.
    pub fn new(reset_time: bool) -> Self {
        Self { reset_time }
    }
}

impl Default for FlushStopEvent {
    fn default() -> Self {
        Self { reset_time: true }
    }
}

// ============================================================================
// Custom Event
// ============================================================================

/// Custom application event.
///
/// Allows applications to send custom events through the pipeline.
#[derive(Debug, Clone)]
pub struct CustomEvent {
    /// Event name.
    pub name: String,
    /// Event data as key-value pairs.
    pub data: std::collections::HashMap<String, TagValue>,
}

impl CustomEvent {
    /// Create a new custom event.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            data: std::collections::HashMap::new(),
        }
    }

    /// Add data to the event.
    pub fn with_data(mut self, key: impl Into<String>, value: impl Into<TagValue>) -> Self {
        self.data.insert(key.into(), value.into());
        self
    }

    /// Get a value from the event data.
    pub fn get(&self, key: &str) -> Option<&TagValue> {
        self.data.get(key)
    }
}

// ============================================================================
// Pipeline Item
// ============================================================================

/// Item that flows through the pipeline - either a buffer or an event.
///
/// This unified type allows buffers and events to flow through the same
/// channels, ensuring proper ordering of serialized events with data.
#[derive(Debug)]
pub enum PipelineItem {
    /// A data buffer.
    Buffer(Buffer),
    /// An event.
    Event(Event),
}

impl PipelineItem {
    /// Check if this is a buffer.
    pub fn is_buffer(&self) -> bool {
        matches!(self, PipelineItem::Buffer(_))
    }

    /// Check if this is an event.
    pub fn is_event(&self) -> bool {
        matches!(self, PipelineItem::Event(_))
    }

    /// Get the buffer if this is a buffer.
    pub fn as_buffer(&self) -> Option<&Buffer> {
        match self {
            PipelineItem::Buffer(b) => Some(b),
            _ => None,
        }
    }

    /// Get the event if this is an event.
    pub fn as_event(&self) -> Option<&Event> {
        match self {
            PipelineItem::Event(e) => Some(e),
            _ => None,
        }
    }

    /// Take the buffer if this is a buffer.
    pub fn into_buffer(self) -> Option<Buffer> {
        match self {
            PipelineItem::Buffer(b) => Some(b),
            _ => None,
        }
    }

    /// Take the event if this is an event.
    pub fn into_event(self) -> Option<Event> {
        match self {
            PipelineItem::Event(e) => Some(e),
            _ => None,
        }
    }
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

// ============================================================================
// Control Signal (for non-serialized events)
// ============================================================================

/// Control signals that bypass the data channel.
///
/// These are used for non-serialized events like flush that need
/// to take effect immediately without waiting in queues.
#[derive(Debug, Clone)]
pub enum ControlSignal {
    /// Flush start - immediately discard buffered data.
    FlushStart,
    /// Flush stop - resume normal operation.
    FlushStop {
        /// Whether to reset running time.
        reset_time: bool,
    },
    /// Pause processing.
    Pause,
    /// Resume processing.
    Resume,
    /// Shutdown the element.
    Shutdown,
}

// ============================================================================
// Event Result
// ============================================================================

/// Result of handling an event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventResult {
    /// Event was handled, don't propagate.
    Handled,
    /// Event was not handled, propagate to next element.
    NotHandled,
    /// Event handling failed.
    Error,
}

impl EventResult {
    /// Check if the event was handled.
    pub fn is_handled(&self) -> bool {
        matches!(self, EventResult::Handled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    fn make_test_buffer() -> Buffer {
        let segment = Arc::new(HeapSegment::new(16).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::from_sequence(0))
    }

    #[test]
    fn test_event_direction() {
        assert!(Event::Eos.is_downstream());
        assert!(Event::Seek(SeekEvent::new_time(ClockTime::ZERO)).is_upstream());
        assert!(Event::FlushStart.is_bidirectional());
    }

    #[test]
    fn test_event_serialization() {
        assert!(Event::Eos.is_serialized());
        assert!(Event::Tags(TagsEvent::new(TagList::new())).is_serialized());
        assert!(!Event::FlushStart.is_serialized());
        assert!(!Event::FlushStop(FlushStopEvent::default()).is_serialized());
    }

    #[test]
    fn test_stream_start() {
        let event = StreamStartEvent::new("stream-001");
        assert_eq!(event.stream_id, "stream-001");
        assert!(event.flags.is_empty());

        let event = StreamStartEvent::with_flags("stream-002", StreamFlags::SPARSE);
        assert!(event.flags.contains(StreamFlags::SPARSE));
    }

    #[test]
    fn test_segment() {
        let seg = SegmentEvent::new_time(ClockTime::from_secs(10), Some(ClockTime::from_secs(60)));
        assert_eq!(seg.format, SegmentFormat::Time);
        assert_eq!(seg.start, 10_000_000_000);
        assert_eq!(seg.stop, 60_000_000_000);
        assert_eq!(seg.rate, 1.0);
    }

    #[test]
    fn test_seek() {
        let seek = SeekEvent::new_time(ClockTime::from_secs(30));
        assert_eq!(seek.format, SegmentFormat::Time);
        assert!(seek.flags.contains(SeekFlags::FLUSH));
        assert!(seek.flags.contains(SeekFlags::KEY_UNIT));
        assert_eq!(seek.start.position, 30_000_000_000);
    }

    #[test]
    fn test_pipeline_item() {
        let buffer = make_test_buffer();
        let item: PipelineItem = buffer.into();
        assert!(item.is_buffer());

        let event = Event::Eos;
        let item: PipelineItem = event.into();
        assert!(item.is_event());
    }

    #[test]
    fn test_custom_event() {
        let event = CustomEvent::new("my-event")
            .with_data("key1", "value1")
            .with_data("key2", 42u64);

        assert_eq!(event.name, "my-event");
        assert!(event.get("key1").is_some());
        assert!(event.get("key2").is_some());
    }
}
