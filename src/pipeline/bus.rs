//! Pipeline message bus for element-to-application communication.
//!
//! The bus provides a thread-safe channel for elements to post structured
//! messages (errors, warnings, tags, QoS, buffering progress) that the
//! application can consume synchronously or asynchronously.
//!
//! Modeled after GStreamer's `GstBus`.
//!
//! # Architecture
//!
//! ```text
//! Elements ──post()──> BusHandle ──mpsc──> Bus ──> Application
//!                                           │
//!                                           └──broadcast──> Subscribers
//! ```
//!
//! # Example
//!
//! ```rust
//! use parallax::pipeline::bus::{Bus, MessageKind};
//!
//! let (mut bus, handle) = Bus::new();
//!
//! // Element posts a message
//! let elem_handle = handle.for_element("decoder");
//! elem_handle.post(MessageKind::Eos);
//!
//! // Application polls
//! let msg = bus.poll().unwrap();
//! assert!(matches!(msg.kind, MessageKind::Eos));
//! ```

use std::collections::HashMap;
use std::fmt;
use std::time::Instant;

use tokio::sync::{broadcast, mpsc};

use crate::clock::ClockTime;
use crate::element::muxer::StreamType;
use crate::pipeline::PipelineState;
use crate::pipeline::tags::TagList;

// ============================================================================
// Message
// ============================================================================

/// A message posted to the pipeline bus.
#[derive(Debug, Clone)]
pub struct Message {
    /// Source element name (or "pipeline" for pipeline-level messages).
    pub source: String,
    /// Timestamp when the message was posted.
    pub timestamp: Instant,
    /// Message payload.
    pub kind: MessageKind,
}

impl fmt::Display for Message {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.source, self.kind)
    }
}

// ============================================================================
// MessageKind
// ============================================================================

/// Standard message types posted to the pipeline bus.
#[derive(Debug, Clone)]
pub enum MessageKind {
    // --- Lifecycle ---
    /// Element or pipeline changed state.
    StateChanged {
        /// Previous state.
        old: PipelineState,
        /// New state.
        new: PipelineState,
    },
    /// End-of-stream reached.
    Eos,
    /// Fatal error — pipeline cannot continue.
    Error {
        /// Error description.
        error: String,
        /// Optional debug details.
        debug: Option<String>,
    },
    /// Non-fatal warning.
    Warning {
        /// Warning description.
        warning: String,
        /// Optional debug details.
        debug: Option<String>,
    },
    /// Informational message.
    Info {
        /// Info text.
        info: String,
    },

    // --- Stream info ---
    /// Stream tags/metadata discovered (e.g., ID3 tags, codec info).
    Tag {
        /// Discovered tags.
        tags: TagList,
    },
    /// Stream duration changed or became known.
    DurationChanged {
        /// New duration, or `None` if duration is no longer known.
        duration: Option<ClockTime>,
    },
    /// New stream collection available (for multi-stream sources).
    StreamCollection {
        /// Available streams.
        streams: Vec<StreamInfo>,
    },

    // --- Quality ---
    /// Quality-of-service: element is dropping data or running late.
    Qos {
        /// The kind of pressure reported.
        qos_type: crate::event::QosType,
        /// Required-vs-achieved rate over the window (1.0 = keeping up,
        /// 2.0 = only half the frames were presentable in time).
        proportion: f64,
        /// Worst lateness observed in the window (ns; negative = early).
        jitter_ns: i64,
        /// Running time of the triggering frame.
        running_time: ClockTime,
        /// Frames presented in the window.
        processed: u64,
        /// Frames dropped in the window.
        dropped: u64,
    },
    /// The pipeline's aggregate declared latency (#184). Posted once at
    /// start when any element declares a latency (jitter buffers, pacing
    /// sinks): the worst source→sink path's sum. Also queryable via
    /// `PipelineHandle::latency()`.
    LatencyChanged {
        /// Minimum latency along the worst path.
        min: ClockTime,
        /// Maximum latency along the worst path.
        max: ClockTime,
    },

    // --- Buffering ---
    /// Network buffering progress (0-100%).
    Buffering {
        /// Buffering percentage (0 = empty, 100 = ready).
        percent: u32,
        /// Buffering mode.
        mode: BufferingMode,
        /// Average input rate (bytes/sec) if known.
        avg_in_rate: Option<u64>,
        /// Average output rate (bytes/sec) if known.
        avg_out_rate: Option<u64>,
        /// Estimated time to complete buffering.
        estimated_total: Option<ClockTime>,
    },

    /// Progressive-download progress from a download-mode `Queue2` (#164).
    ///
    /// The range-structured companion to [`Buffering`](Self::Buffering)'s
    /// percentage: a seek-bar UI shades `ranges` to show which byte seeks
    /// are instant. Posted throttled (the queue's rate-estimate cadence),
    /// not per buffer. Poll-anytime access to the same data is
    /// `Queue2RangesHandle`.
    DownloadProgress {
        /// Merged downloaded byte spans `(start, end)`, ascending, in true
        /// stream offsets.
        ranges: Vec<(u64, u64)>,
        /// Total expected stream size, when known.
        total: Option<u64>,
        /// Stream offset of the next byte the queue expects to receive.
        write_pos: u64,
    },

    // --- Async state changes ---
    /// Async state change started.
    AsyncStart,
    /// Async state change completed.
    AsyncDone {
        /// Running time when async operation completed.
        running_time: ClockTime,
    },

    // --- Clock ---
    /// Clock was lost, pipeline needs to select a new clock.
    ClockLost,
    /// New clock was selected.
    NewClock {
        /// Name of the selected clock.
        clock_name: String,
    },

    // --- Seeking ---
    /// Seek completed (all elements flushed and repositioned).
    SeekDone {
        /// Seqnum of the [`SeekEvent`](crate::event::SeekEvent) this
        /// completes — correlate rapid seeks with their completions
        /// (GStreamer's seqnum discipline). A multi-source pipeline posts
        /// one SeekDone per handling source, all sharing the seek's seqnum.
        seqnum: u64,
        /// The element that handled the seek.
        source: String,
        /// Interpretation of `position` (Time = nanoseconds, Bytes = offset).
        format: crate::event::SegmentFormat,
        /// Where the stream actually landed — the element's reported landing
        /// position when it knows one (keyframe-snapping demuxers land off
        /// target), else the requested position; `None` for a relative seek
        /// whose handler reported nothing.
        position: Option<u64>,
    },

    /// A [`SeekFlags::SEGMENT`](crate::event::SeekFlags::SEGMENT) seek's
    /// playback range finished (#165). Posted INSTEAD of the sinks seeing
    /// EOS — the producer stays alive awaiting the next seek. Respond with
    /// a non-flushing SEGMENT seek (back to the start for a gapless loop);
    /// stop the pipeline with [`PipelineHandle::stop`] when done.
    ///
    /// [`PipelineHandle::stop`]: crate::pipeline::PipelineHandle::stop
    SegmentDone {
        /// Seqnum of the SEGMENT seek this completes (see
        /// [`SeekDone`](Self::SeekDone) for the correlation discipline).
        seqnum: u64,
        /// The producing element whose segment ran out.
        source: String,
        /// Interpretation of `position`.
        format: crate::event::SegmentFormat,
        /// The seek's stop when it had one; `None` for a natural end
        /// (no stop set — the stream simply ran out).
        position: Option<u64>,
    },

    // --- Application/Element-specific ---
    /// Element-specific structured message.
    Element {
        /// Structure name (e.g., "barcode", "level", "spectrum").
        structure_name: String,
        /// Key-value payload.
        fields: HashMap<String, MessageValue>,
    },
    /// Application-defined message (posted via [`BusHandle::post_application`]).
    Application {
        /// Structure name.
        structure_name: String,
        /// Key-value payload.
        fields: HashMap<String, MessageValue>,
    },
}

impl fmt::Display for MessageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageKind::StateChanged { old, new } => {
                write!(f, "StateChanged: {old:?} -> {new:?}")
            }
            MessageKind::Eos => write!(f, "EOS"),
            MessageKind::Error { error, debug } => {
                write!(f, "Error: {error}")?;
                if let Some(d) = debug {
                    write!(f, " ({d})")?;
                }
                Ok(())
            }
            MessageKind::Warning { warning, debug } => {
                write!(f, "Warning: {warning}")?;
                if let Some(d) = debug {
                    write!(f, " ({d})")?;
                }
                Ok(())
            }
            MessageKind::Info { info } => write!(f, "Info: {info}"),
            MessageKind::Tag { tags } => write!(f, "Tag: {tags}"),
            MessageKind::DurationChanged { duration } => {
                write!(f, "DurationChanged: {duration:?}")
            }
            MessageKind::StreamCollection { streams } => {
                write!(f, "StreamCollection: {} streams", streams.len())
            }
            MessageKind::Qos {
                qos_type,
                proportion,
                processed,
                dropped,
                ..
            } => {
                write!(
                    f,
                    "QoS: {qos_type:?} proportion={proportion:.2} \
                     ({processed} presented, {dropped} dropped)"
                )
            }
            MessageKind::LatencyChanged { min, max } => {
                write!(f, "LatencyChanged: {min} .. {max}")
            }
            MessageKind::Buffering { percent, mode, .. } => {
                write!(f, "Buffering: {percent}% ({mode:?})")
            }
            MessageKind::DownloadProgress {
                ranges,
                total,
                write_pos,
            } => {
                write!(
                    f,
                    "DownloadProgress: {} range(s), write_pos {write_pos}, total {total:?}",
                    ranges.len()
                )
            }
            MessageKind::AsyncStart => write!(f, "AsyncStart"),
            MessageKind::AsyncDone { running_time } => {
                write!(f, "AsyncDone: {running_time}")
            }
            MessageKind::ClockLost => write!(f, "ClockLost"),
            MessageKind::NewClock { clock_name } => write!(f, "NewClock: {clock_name}"),
            MessageKind::SeekDone {
                seqnum,
                source,
                format,
                position,
            } => match position {
                Some(p) => write!(f, "SeekDone #{seqnum} ({source}): {p} ({format:?})"),
                None => write!(f, "SeekDone #{seqnum} ({source})"),
            },
            MessageKind::SegmentDone {
                seqnum,
                source,
                format,
                position,
            } => match position {
                Some(p) => write!(f, "SegmentDone #{seqnum} ({source}): {p} ({format:?})"),
                None => write!(f, "SegmentDone #{seqnum} ({source})"),
            },
            MessageKind::Element {
                structure_name,
                fields,
            } => {
                write!(f, "Element({structure_name}): {} fields", fields.len())
            }
            MessageKind::Application {
                structure_name,
                fields,
            } => {
                write!(f, "Application({structure_name}): {} fields", fields.len())
            }
        }
    }
}

// ============================================================================
// MessageValue
// ============================================================================

/// Typed value for structured message fields.
#[derive(Debug, Clone, PartialEq)]
pub enum MessageValue {
    /// Boolean value.
    Bool(bool),
    /// Signed integer.
    Int(i64),
    /// Unsigned integer.
    Uint(u64),
    /// Floating-point value.
    Float(f64),
    /// String value.
    String(String),
    /// Raw bytes.
    Bytes(Vec<u8>),
    /// Time value.
    ClockTime(ClockTime),
    /// Nested list of values.
    List(Vec<MessageValue>),
}

impl fmt::Display for MessageValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MessageValue::Bool(v) => write!(f, "{v}"),
            MessageValue::Int(v) => write!(f, "{v}"),
            MessageValue::Uint(v) => write!(f, "{v}"),
            MessageValue::Float(v) => write!(f, "{v}"),
            MessageValue::String(v) => write!(f, "{v}"),
            MessageValue::Bytes(v) => write!(f, "[{} bytes]", v.len()),
            MessageValue::ClockTime(v) => write!(f, "{v}"),
            MessageValue::List(v) => write!(f, "[{} items]", v.len()),
        }
    }
}

impl From<bool> for MessageValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<i64> for MessageValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}
impl From<u64> for MessageValue {
    fn from(v: u64) -> Self {
        Self::Uint(v)
    }
}
impl From<f64> for MessageValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}
impl From<String> for MessageValue {
    fn from(v: String) -> Self {
        Self::String(v)
    }
}
impl From<&str> for MessageValue {
    fn from(v: &str) -> Self {
        Self::String(v.to_string())
    }
}

// ============================================================================
// BufferingMode
// ============================================================================

/// Buffering mode for network sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferingMode {
    /// Stream buffering with watermarks.
    Stream,
    /// Download to temporary file.
    Download,
    /// Fixed-size ring buffer for timeshift/DVR.
    Timeshift,
    /// Live source jitter buffering.
    Live,
}

// ============================================================================
// StreamInfo
// ============================================================================

/// Information about a stream within a multi-stream source.
#[derive(Debug, Clone, PartialEq)]
pub struct StreamInfo {
    /// Unique stream identifier.
    pub stream_id: String,
    /// Stream type (video, audio, subtitle, data).
    pub stream_type: StreamType,
    /// Tags associated with this stream.
    pub tags: TagList,
    /// Human-readable caps description.
    pub caps: Option<String>,
}

// ============================================================================
// Bus
// ============================================================================

/// Thread-safe pipeline message bus.
///
/// Elements post messages via [`BusHandle`] (cloneable sender).
/// Applications consume messages via polling ([`poll`](Bus::poll)),
/// async waiting ([`next`](Bus::next)), or broadcast subscription
/// ([`subscribe`](Bus::subscribe)).
///
/// The bus uses an MPSC channel internally. A background drain task
/// (spawned by the executor) fans messages out to broadcast subscribers.
pub struct Bus {
    /// Incoming messages from elements.
    receiver: mpsc::UnboundedReceiver<Message>,
    /// Broadcast sender for async subscribers.
    broadcast_tx: broadcast::Sender<Message>,
    /// Pending message for peek().
    peeked: Option<Message>,
}

/// Cloneable handle for elements to post messages to the bus.
///
/// Each element gets its own `BusHandle` with a pre-set `source_name`.
/// Posting a message automatically fills in the source and timestamp.
#[derive(Clone)]
pub struct BusHandle {
    sender: mpsc::UnboundedSender<Message>,
    source_name: String,
}

impl Bus {
    /// Create a new bus, returning the bus and a root handle.
    ///
    /// The root handle has source name "pipeline". Use
    /// [`BusHandle::for_element`] to create element-specific handles.
    pub fn new() -> (Bus, BusHandle) {
        let (tx, rx) = mpsc::unbounded_channel();
        let (broadcast_tx, _) = broadcast::channel(256);
        let bus = Bus {
            receiver: rx,
            broadcast_tx,
            peeked: None,
        };
        let handle = BusHandle {
            sender: tx,
            source_name: "pipeline".into(),
        };
        (bus, handle)
    }

    /// Poll for the next message (non-blocking).
    ///
    /// Returns `None` if no message is available.
    pub fn poll(&mut self) -> Option<Message> {
        if let Some(msg) = self.peeked.take() {
            return Some(msg);
        }
        self.receiver.try_recv().ok()
    }

    /// Wait for the next message (async).
    ///
    /// Returns `None` if all senders have been dropped.
    pub async fn next(&mut self) -> Option<Message> {
        if let Some(msg) = self.peeked.take() {
            return Some(msg);
        }
        self.receiver.recv().await
    }

    /// Peek at the next message without consuming it.
    pub fn peek(&mut self) -> Option<&Message> {
        if self.peeked.is_none() {
            self.peeked = self.receiver.try_recv().ok();
        }
        self.peeked.as_ref()
    }

    /// Get a broadcast receiver for async multi-subscriber consumption.
    ///
    /// Multiple subscribers can receive the same messages. Messages are
    /// forwarded from the MPSC channel to broadcast by calling
    /// [`flush_to_broadcast`](Bus::flush_to_broadcast) or by the
    /// executor's background drain task.
    pub fn subscribe(&self) -> broadcast::Receiver<Message> {
        self.broadcast_tx.subscribe()
    }

    /// Drain all pending messages, forwarding to broadcast subscribers.
    ///
    /// Call this periodically if you need broadcast subscriber support
    /// without the executor's background drain task.
    pub fn flush_to_broadcast(&mut self) {
        while let Ok(msg) = self.receiver.try_recv() {
            let _ = self.broadcast_tx.send(msg);
        }
    }

    /// Wait for a message matching a predicate.
    ///
    /// Non-matching messages are consumed and discarded.
    pub async fn wait_for<F>(&mut self, predicate: F) -> Option<Message>
    where
        F: Fn(&MessageKind) -> bool,
    {
        loop {
            let msg = self.next().await?;
            if predicate(&msg.kind) {
                return Some(msg);
            }
        }
    }

    /// Wait for EOS or an error.
    ///
    /// Returns `Ok(())` on EOS, `Err(error_message)` on error.
    /// All other messages are consumed and discarded.
    pub async fn wait_for_eos_or_error(&mut self) -> std::result::Result<(), String> {
        loop {
            match self.next().await {
                Some(msg) => match &msg.kind {
                    MessageKind::Eos => return Ok(()),
                    MessageKind::Error { error, .. } => return Err(error.clone()),
                    _ => continue,
                },
                None => return Err("Bus closed".into()),
            }
        }
    }

    /// Get a reference to the broadcast sender (for background drain task).
    #[allow(dead_code)]
    pub(crate) fn broadcast_sender(&self) -> &broadcast::Sender<Message> {
        &self.broadcast_tx
    }

    /// Convert the bus into an async `Stream` for use with `select!`
    /// and stream combinators.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// use futures::StreamExt;
    ///
    /// let mut stream = bus.into_stream();
    /// while let Some(msg) = stream.next().await {
    ///     println!("{}", msg);
    /// }
    /// ```
    pub fn into_stream(self) -> BusStream {
        BusStream { bus: self }
    }
}

/// Async `Stream` adapter for [`Bus`].
///
/// Created by [`Bus::into_stream()`]. Enables use with `select!`,
/// `StreamExt` combinators, and other async stream patterns.
pub struct BusStream {
    bus: Bus,
}

impl BusStream {
    /// Get a mutable reference to the underlying bus.
    pub fn bus_mut(&mut self) -> &mut Bus {
        &mut self.bus
    }

    /// Consume the stream and get the bus back.
    pub fn into_inner(self) -> Bus {
        self.bus
    }
}

impl futures::Stream for BusStream {
    type Item = Message;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        // Return peeked message first
        if let Some(msg) = self.bus.peeked.take() {
            return std::task::Poll::Ready(Some(msg));
        }
        // Poll the underlying mpsc receiver
        self.bus.receiver.poll_recv(cx)
    }
}

impl BusHandle {
    /// Create a handle for a specific element.
    ///
    /// The new handle shares the same underlying channel but posts
    /// messages with the given element name as source.
    pub fn for_element(&self, name: &str) -> BusHandle {
        BusHandle {
            sender: self.sender.clone(),
            source_name: name.to_string(),
        }
    }

    /// Get the source name of this handle.
    pub fn source_name(&self) -> &str {
        &self.source_name
    }

    /// Post a message to the bus.
    pub fn post(&self, kind: MessageKind) {
        let msg = Message {
            source: self.source_name.clone(),
            timestamp: Instant::now(),
            kind,
        };
        let _ = self.sender.send(msg);
    }

    /// Post an error message.
    pub fn post_error(&self, error: impl Into<String>, debug: Option<String>) {
        self.post(MessageKind::Error {
            error: error.into(),
            debug,
        });
    }

    /// Post a warning message.
    pub fn post_warning(&self, warning: impl Into<String>, debug: Option<String>) {
        self.post(MessageKind::Warning {
            warning: warning.into(),
            debug,
        });
    }

    /// Post an info message.
    pub fn post_info(&self, info: impl Into<String>) {
        self.post(MessageKind::Info { info: info.into() });
    }

    /// Post a tag discovery message.
    pub fn post_tags(&self, tags: TagList) {
        self.post(MessageKind::Tag { tags });
    }

    /// Post a buffering progress message.
    pub fn post_buffering(&self, percent: u32, mode: BufferingMode) {
        self.post(MessageKind::Buffering {
            percent,
            mode,
            avg_in_rate: None,
            avg_out_rate: None,
            estimated_total: None,
        });
    }

    /// Post an element-specific structured message.
    pub fn post_element(
        &self,
        structure_name: impl Into<String>,
        fields: HashMap<String, MessageValue>,
    ) {
        self.post(MessageKind::Element {
            structure_name: structure_name.into(),
            fields,
        });
    }

    /// Post an application-defined message.
    pub fn post_application(
        &self,
        structure_name: impl Into<String>,
        fields: HashMap<String, MessageValue>,
    ) {
        self.post(MessageKind::Application {
            structure_name: structure_name.into(),
            fields,
        });
    }

    /// Post a QoS message.
    pub fn post_qos(&self, qos: &crate::event::QosEvent) {
        self.post(MessageKind::Qos {
            qos_type: qos.qos_type,
            proportion: qos.proportion,
            jitter_ns: qos.jitter_ns,
            running_time: qos.timestamp,
            processed: qos.processed,
            dropped: qos.dropped,
        });
    }

    /// Post a state change message.
    pub fn post_state_changed(&self, old: PipelineState, new: PipelineState) {
        self.post(MessageKind::StateChanged { old, new });
    }

    /// Post an EOS message.
    pub fn post_eos(&self) {
        self.post(MessageKind::Eos);
    }
}

impl fmt::Debug for BusHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BusHandle")
            .field("source_name", &self.source_name)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for Bus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Bus")
            .field("peeked", &self.peeked.is_some())
            .finish_non_exhaustive()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bus_poll() {
        let (mut bus, handle) = Bus::new();
        assert!(bus.poll().is_none());

        handle.post(MessageKind::Eos);
        let msg = bus.poll().unwrap();
        assert!(matches!(msg.kind, MessageKind::Eos));
        assert_eq!(msg.source, "pipeline");
    }

    #[test]
    fn test_bus_handle_for_element() {
        let (mut bus, handle) = Bus::new();
        let decoder_handle = handle.for_element("decoder");
        decoder_handle.post_error("codec error", Some("frame 42".into()));

        let msg = bus.poll().unwrap();
        assert_eq!(msg.source, "decoder");
        match msg.kind {
            MessageKind::Error { error, debug } => {
                assert_eq!(error, "codec error");
                assert_eq!(debug.as_deref(), Some("frame 42"));
            }
            _ => panic!("Expected Error message"),
        }
    }

    #[test]
    fn test_bus_peek() {
        let (mut bus, handle) = Bus::new();
        handle.post(MessageKind::Eos);

        // Peek should return the message without consuming it
        assert!(bus.peek().is_some());
        assert!(bus.peek().is_some()); // Still there

        // Poll should consume it
        let msg = bus.poll().unwrap();
        assert!(matches!(msg.kind, MessageKind::Eos));
        assert!(bus.poll().is_none());
    }

    #[tokio::test]
    async fn test_bus_next() {
        let (mut bus, handle) = Bus::new();

        tokio::spawn(async move {
            handle.post(MessageKind::Info {
                info: "hello".into(),
            });
        });

        let msg = bus.next().await.unwrap();
        assert!(matches!(msg.kind, MessageKind::Info { .. }));
    }

    #[tokio::test]
    async fn test_bus_wait_for_eos_or_error_eos() {
        let (mut bus, handle) = Bus::new();

        tokio::spawn(async move {
            handle.post(MessageKind::Info {
                info: "starting".into(),
            });
            handle.post(MessageKind::Eos);
        });

        let result = bus.wait_for_eos_or_error().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_bus_wait_for_eos_or_error_error() {
        let (mut bus, handle) = Bus::new();

        tokio::spawn(async move {
            handle.post_error("fatal", None);
        });

        let result = bus.wait_for_eos_or_error().await;
        assert_eq!(result.unwrap_err(), "fatal");
    }

    #[test]
    fn test_bus_flush_to_broadcast() {
        let (mut bus, handle) = Bus::new();
        let mut sub = bus.subscribe();

        handle.post(MessageKind::Eos);
        handle.post(MessageKind::Info {
            info: "test".into(),
        });

        bus.flush_to_broadcast();

        let msg1 = sub.try_recv().unwrap();
        assert!(matches!(msg1.kind, MessageKind::Eos));
        let msg2 = sub.try_recv().unwrap();
        assert!(matches!(msg2.kind, MessageKind::Info { .. }));
    }

    #[test]
    fn test_message_display() {
        let msg = Message {
            source: "decoder".into(),
            timestamp: Instant::now(),
            kind: MessageKind::Error {
                error: "failed".into(),
                debug: Some("details".into()),
            },
        };
        let s = format!("{msg}");
        assert!(s.contains("decoder"));
        assert!(s.contains("failed"));
    }

    #[test]
    fn test_bus_handle_post_tags() {
        let (mut bus, handle) = Bus::new();

        let mut tags = TagList::new();
        tags.set(
            "title",
            crate::pipeline::tags::TagValue::String("Test".into()),
        );
        handle.post_tags(tags);

        let msg = bus.poll().unwrap();
        match msg.kind {
            MessageKind::Tag { tags } => {
                assert_eq!(tags.get_string("title"), Some("Test"));
            }
            _ => panic!("Expected Tag message"),
        }
    }
}
