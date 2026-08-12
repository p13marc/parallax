//! Core element traits.

// Allow missing docs on dynosaur-generated code
#![allow(missing_docs)]

use crate::buffer::Buffer;
use crate::clock::{Clock, ClockProvider, ClockTime};
use crate::element::context::{ConsumeContext, ProduceContext, ProduceResult};
use crate::error::Result;
use crate::event::{Event, EventResult};
use crate::format::{Caps, ElementMediaCaps};
use dynosaur::dynosaur;
use smallvec::SmallVec;
use std::any::Any;
use std::sync::Arc;

// ============================================================================
// Execution Hints
// ============================================================================

/// Hints about an element's execution characteristics.
///
/// Elements declare *facts* about their capabilities; the executor *decides*
/// the scheduling strategy. This PipeWire-inspired design makes contradictions
/// structurally impossible — there is no way for an element to request
/// conflicting scheduling preferences.
///
/// The executor derives the strategy automatically:
/// - `rt_safe + low latency` → RT thread
/// - `io_bound` → Tokio async task
/// - `untrusted / native` → isolated process
/// - everything else → Tokio async task (default)
///
/// Use the profile constructors for common patterns:
/// - `ExecutionHints::rt_safe()` — RT-safe transforms (Gain, filters, PassThrough)
/// - `ExecutionHints::io_bound()` — I/O devices (ALSA, PipeWire, network)
/// - `ExecutionHints::native()` — FFI/native code (codecs wrapping C libraries)
/// - `ExecutionHints::untrusted()` — Untrusted input handlers (must be isolated)
///
/// # Example
///
/// ```rust,ignore
/// impl Element for MyGainFilter {
///     fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> { /* ... */ }
///
///     fn execution_hints(&self) -> ExecutionHints {
///         ExecutionHints::rt_safe()
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionHints {
    // === Scheduling ===
    /// Whether this element can safely run in a real-time context.
    ///
    /// RT-safe means: no heap allocation, no blocking I/O, no locks
    /// shared with non-RT threads, bounded deterministic execution time.
    ///
    /// Default: `false` (conservative).
    pub rt_safe: bool,

    // === Isolation ===
    /// Trust level of the data being processed.
    pub trust_level: TrustLevel,
    /// Whether the element might crash on bad input.
    pub crash_safe: bool,
    /// Whether the element uses native code (FFI).
    pub uses_native_code: bool,

    // === Performance characteristics ===
    /// Processing characteristics (CPU vs I/O bound).
    pub processing: ProcessingHint,
    /// Latency requirements.
    pub latency: LatencyHint,
    /// Memory usage hint (helps with scheduling decisions).
    pub memory: MemoryHint,
}

impl Default for ExecutionHints {
    fn default() -> Self {
        Self {
            rt_safe: false,
            trust_level: TrustLevel::Trusted,
            processing: ProcessingHint::Unknown,
            latency: LatencyHint::Normal,
            crash_safe: true,
            uses_native_code: false,
            memory: MemoryHint::Normal,
        }
    }
}

impl ExecutionHints {
    /// Create hints for a trusted, lightweight element (same as default).
    pub fn trusted() -> Self {
        Self::default()
    }

    /// RT-safe transform: CPU-bound, low latency, trusted.
    ///
    /// For elements like Gain, PassThrough, filters — no allocations,
    /// no blocking I/O, bounded deterministic execution.
    pub fn rt_safe() -> Self {
        Self {
            rt_safe: true,
            processing: ProcessingHint::CpuBound,
            latency: LatencyHint::Low,
            ..Self::default()
        }
    }

    /// I/O-bound element: not RT-safe, will be scheduled in Tokio.
    ///
    /// For device sources/sinks (ALSA, PipeWire, V4L2, network).
    /// The executor automatically assigns I/O-bound elements to async tasks.
    pub fn io_bound() -> Self {
        Self {
            processing: ProcessingHint::IoBound,
            ..Self::default()
        }
    }

    /// Native/FFI element: may crash, should be isolated.
    ///
    /// For codecs wrapping C libraries.
    pub fn native() -> Self {
        Self {
            uses_native_code: true,
            crash_safe: false,
            ..Self::default()
        }
    }

    /// Untrusted input handler: must be isolated.
    pub fn untrusted() -> Self {
        Self {
            trust_level: TrustLevel::Untrusted,
            crash_safe: false,
            ..Self::default()
        }
    }

    /// CPU-intensive but not RT-safe.
    pub fn cpu_intensive() -> Self {
        Self {
            processing: ProcessingHint::CpuBound,
            ..Self::default()
        }
    }

    /// Low-latency element (auto RT detection).
    pub fn low_latency() -> Self {
        Self {
            latency: LatencyHint::Low,
            ..Self::default()
        }
    }

    /// Whether this element can run in an RT context.
    pub fn is_rt_safe(&self) -> bool {
        self.rt_safe
    }

    /// Whether this element should be isolated.
    pub fn should_isolate(&self) -> bool {
        self.trust_level == TrustLevel::Untrusted || (self.uses_native_code && !self.crash_safe)
    }

    /// Builder: set the RT-safe flag.
    pub fn with_rt_safe(mut self, rt_safe: bool) -> Self {
        self.rt_safe = rt_safe;
        self
    }
}

/// Trust level of the data being processed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TrustLevel {
    /// Data comes from a trusted source (e.g., internal pipeline).
    #[default]
    Trusted,
    /// Data comes from a semi-trusted source (e.g., local file).
    SemiTrusted,
    /// Data comes from an untrusted source (e.g., network, user input).
    /// Elements handling untrusted data should be isolated.
    Untrusted,
}

/// Hint about the processing characteristics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ProcessingHint {
    /// Unknown or mixed processing.
    #[default]
    Unknown,
    /// Primarily CPU-bound (computationally intensive).
    /// Good candidates for RT threads or dedicated cores.
    CpuBound,
    /// Primarily I/O-bound (waiting on I/O operations).
    /// Best suited for async runtime.
    IoBound,
    /// Memory-bound (large data transfers).
    MemoryBound,
}

/// Latency requirements hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum LatencyHint {
    /// Ultra-low latency required (< 1ms).
    /// Must run in RT thread.
    UltraLow,
    /// Low latency required (< 10ms).
    /// Prefer RT thread if available.
    Low,
    /// Normal latency acceptable (< 100ms).
    #[default]
    Normal,
    /// High latency acceptable (> 100ms).
    /// Can be scheduled opportunistically.
    Relaxed,
}

/// Memory usage hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum MemoryHint {
    /// Normal memory usage.
    #[default]
    Normal,
    /// Low memory usage (good for isolation).
    Low,
    /// High memory usage (may need special handling).
    High,
    /// Streaming (processes data in chunks, doesn't accumulate).
    Streaming,
}

// ============================================================================
// Output Type
// ============================================================================

/// Output of element processing.
///
/// Represents the result of processing a buffer:
/// - `None`: No output (buffer was filtered/dropped)
/// - `Single`: One output buffer
/// - `Multiple`: Multiple output buffers
///
/// # Examples
///
/// ```rust
/// use parallax::element::Output;
/// use parallax::buffer::Buffer;
///
/// // No output (filter)
/// let out = Output::none();
///
/// // Single output (most common)
/// // let out = Output::single(buffer);
///
/// // Multiple outputs (e.g., from Chunk or FlatMap)
/// // let out = Output::from(vec![buf1, buf2, buf3]);
/// ```
#[derive(Debug, Default)]
#[allow(clippy::large_enum_variant)] // Intentional: avoid heap allocation on hot path
pub enum Output {
    /// No output (buffer was filtered/consumed).
    #[default]
    None,
    /// Single output buffer.
    Single(Buffer),
    /// Multiple output buffers (same destination).
    Multiple(Vec<Buffer>),
}

impl Output {
    /// Create a single buffer output.
    #[inline]
    pub fn single(buf: Buffer) -> Self {
        Self::Single(buf)
    }

    /// Create an empty output.
    #[inline]
    pub fn none() -> Self {
        Self::None
    }

    /// Check if there is no output.
    #[inline]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Check if there is exactly one output.
    #[inline]
    pub fn is_single(&self) -> bool {
        matches!(self, Self::Single(_))
    }

    /// Check if there are multiple outputs.
    #[inline]
    pub fn is_multiple(&self) -> bool {
        matches!(self, Self::Multiple(_))
    }

    /// Get the number of output buffers.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Single(_) => 1,
            Self::Multiple(v) => v.len(),
        }
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to a Vec of buffers.
    pub fn into_vec(self) -> Vec<Buffer> {
        match self {
            Self::None => vec![],
            Self::Single(b) => vec![b],
            Self::Multiple(v) => v,
        }
    }

    /// Get a reference to the single buffer, if any.
    pub fn as_single(&self) -> Option<&Buffer> {
        match self {
            Self::Single(b) => Some(b),
            _ => None,
        }
    }

    /// Take the single buffer, returning None for other variants.
    pub fn into_single(self) -> Option<Buffer> {
        match self {
            Self::Single(b) => Some(b),
            _ => None,
        }
    }
}

// Ergonomic conversions
impl From<Buffer> for Output {
    #[inline]
    fn from(b: Buffer) -> Self {
        Self::Single(b)
    }
}

impl From<Option<Buffer>> for Output {
    fn from(opt: Option<Buffer>) -> Self {
        match opt {
            Some(b) => Self::Single(b),
            None => Self::None,
        }
    }
}

impl From<Vec<Buffer>> for Output {
    fn from(v: Vec<Buffer>) -> Self {
        match v.len() {
            0 => Self::None,
            1 => Self::Single(v.into_iter().next().unwrap()),
            _ => Self::Multiple(v),
        }
    }
}

impl FromIterator<Buffer> for Output {
    fn from_iter<I: IntoIterator<Item = Buffer>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}

impl IntoIterator for Output {
    type Item = Buffer;
    type IntoIter = OutputIter;

    fn into_iter(self) -> Self::IntoIter {
        OutputIter(match self {
            Output::None => OutputIterInner::None,
            Output::Single(b) => OutputIterInner::Single(Some(b)),
            Output::Multiple(v) => OutputIterInner::Multiple(v.into_iter()),
        })
    }
}

/// Iterator over Output buffers.
pub struct OutputIter(OutputIterInner);

#[allow(clippy::large_enum_variant)] // Internal iterator, avoid heap allocation
enum OutputIterInner {
    None,
    Single(Option<Buffer>),
    Multiple(std::vec::IntoIter<Buffer>),
}

impl Iterator for OutputIter {
    type Item = Buffer;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            OutputIterInner::None => None,
            OutputIterInner::Single(opt) => opt.take(),
            OutputIterInner::Multiple(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.0 {
            OutputIterInner::None => (0, Some(0)),
            OutputIterInner::Single(opt) => {
                let n = if opt.is_some() { 1 } else { 0 };
                (n, Some(n))
            }
            OutputIterInner::Multiple(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for OutputIter {}

// ============================================================================
// Source Result (for proper WouldBlock handling)
// ============================================================================

/// Result of a source producing data.
///
/// This enum allows sources to properly signal when no data is available
/// yet (WouldBlock) vs end of stream (Eos), which the executor handles
/// differently.
#[derive(Debug)]
pub enum SourceResult {
    /// A buffer was produced.
    Buffer(Box<Buffer>),
    /// No data available yet, try again later.
    WouldBlock,
    /// End of stream reached.
    Eos,
}

/// Result of one demuxer step, with the pad routing intact.
///
/// This is what [`AsyncElementDyn::process_demux`] returns to the executor:
/// unlike `process`/`process_all`, which erase *which pad* each buffer
/// belongs on, the routed form is what makes per-pad delivery possible at
/// all (#76).
#[derive(Debug)]
pub enum DemuxResult {
    /// Buffers routed by output pad name. May be empty — this input (or this
    /// produce step) yielded nothing to emit.
    Routed(Vec<(String, Buffer)>),
    /// No data available yet, try again later (source-style demuxers only).
    WouldBlock,
    /// End of stream reached (source-style demuxers only).
    Eos,
}

/// Result of a source-style demuxer producing data on its own.
///
/// Returned by [`Demuxer::produce`] — the input-less twin of
/// [`Demuxer::demux`] for demuxers that own their reader (a file demuxer)
/// and act as the pipeline's source.
#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // Intentional: RoutedOutput keeps small outputs inline, off the heap
pub enum DemuxerProduce {
    /// Routed output was produced (may be empty).
    Routed(RoutedOutput),
    /// No data available yet, try again later.
    WouldBlock,
    /// End of stream reached.
    Eos,
}

// ============================================================================
// Source Trait
// ============================================================================

/// A source element that produces buffers.
///
/// Sources are the entry points of a pipeline. They generate data from
/// external sources like files, network connections, or hardware devices.
///
/// # Pool-Aware Buffer Production (PipeWire-style)
///
/// The framework provides a [`ProduceContext`] with a pre-allocated buffer
/// from the pool. Sources write data into this buffer and return how many
/// bytes were produced. This enables true zero-allocation operation.
///
/// # Lifecycle
///
/// - `produce()` is called repeatedly by the executor
/// - The executor provides a `ProduceContext` with a pre-allocated buffer
/// - Write to `ctx.output()`, set metadata via `ctx.metadata_mut()`
/// - Return `ProduceResult::Produced(n)` to emit n bytes
/// - Return `ProduceResult::Eos` to signal end-of-stream
/// - Return `ProduceResult::OwnBuffer(buffer)` for sources with their own memory
/// - Return `Err(...)` to signal an error
///
/// # Example
///
/// ```rust,ignore
/// struct CounterSource {
///     count: u64,
///     max: u64,
/// }
///
/// impl Source for CounterSource {
///     fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
///         if self.count >= self.max {
///             return Ok(ProduceResult::Eos);
///         }
///
///         // Write to the provided buffer
///         let output = ctx.output();
///         let bytes = self.count.to_le_bytes();
///         output[..8].copy_from_slice(&bytes);
///
///         // Set metadata
///         ctx.set_sequence(self.count);
///         self.count += 1;
///
///         Ok(ProduceResult::Produced(8))
///     }
/// }
/// ```
///
/// # Own Buffer Fallback
///
/// Sources that manage their own memory (e.g., mmap, external APIs) can
/// return `ProduceResult::OwnBuffer`:
///
/// ```rust,ignore
/// impl Source for MmapSource {
///     fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
///         // Source has its own buffer management
///         let buffer = self.mmap_next_region()?;
///         Ok(ProduceResult::OwnBuffer(buffer))
///     }
/// }
/// ```
pub trait Source: Send {
    /// Produce the next buffer.
    ///
    /// The framework provides a `ProduceContext` with a pre-allocated output buffer.
    /// Write data to `ctx.output()`, set metadata via `ctx.metadata_mut()`, and
    /// return `ProduceResult::Produced(n)` to indicate how many bytes were written.
    ///
    /// Return `ProduceResult::Eos` when the source is exhausted.
    /// Return `ProduceResult::OwnBuffer(buffer)` if the source manages its own memory.
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult>;

    /// Get the name of this source (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the output caps (what formats this source produces).
    fn output_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the output media caps with format+memory constraints.
    ///
    /// This is the enhanced version of `output_caps()` that allows specifying
    /// multiple format+memory combinations. For example, a camera might support
    /// YUYV on CPU and NV12 on DMA-BUF.
    ///
    /// Default implementation converts from `output_caps()` assuming CPU memory.
    fn output_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.output_caps())
    }

    /// Get the preferred buffer size for this source.
    ///
    /// The executor uses this hint when allocating pool buffers.
    /// Return `None` to use the default pool buffer size.
    fn preferred_buffer_size(&self) -> Option<usize> {
        None
    }

    /// Get execution hints for automatic scheduling decisions.
    ///
    /// Override this to provide hints about processing characteristics,
    /// trust level, latency requirements, etc. The executor uses these
    /// hints to automatically determine the best execution strategy.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// Handle an upstream event (seek, QoS, etc.).
    ///
    /// Sources are typically the handlers of upstream events like seek.
    /// Return `EventResult::Handled` if the event was processed,
    /// `EventResult::NotHandled` otherwise.
    ///
    /// Default implementation does not handle any events.
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }

    /// Whether this source supports seeking.
    ///
    /// Override this to return `true` if the source can handle seek events.
    fn is_seekable(&self) -> bool {
        false
    }

    /// Query the current position.
    ///
    /// Returns the current read position in the source's native format
    /// (typically bytes for file sources, time for demuxers).
    fn query_position(&self) -> Option<crate::pipeline::seek::PositionQuery> {
        None
    }

    /// Query the total duration.
    ///
    /// Returns the total duration/size of the source if known.
    fn query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        None
    }

    /// Handle a flow control signal from downstream.
    ///
    /// This is called when downstream elements signal backpressure
    /// (Busy, Drop) or other flow control events. Sources should
    /// respond according to their `flow_policy()`.
    ///
    /// Default implementation ignores the signal (backward compatible).
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// impl Source for LiveCameraSource {
    ///     fn handle_flow_signal(&mut self, signal: FlowSignal) {
    ///         self.current_signal = signal;
    ///     }
    ///
    ///     fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
    ///         if self.current_signal == FlowSignal::Drop {
    ///             // Skip frame production
    ///             return Ok(ProduceResult::WouldBlock);
    ///         }
    ///         // ... normal production
    ///     }
    /// }
    /// ```
    fn handle_flow_signal(&mut self, _signal: crate::pipeline::flow::FlowSignal) {
        // Default: ignore (backward compatibility)
    }

    /// Get the flow control policy for this source.
    ///
    /// This determines how the source responds to backpressure:
    /// - `Block`: Wait until downstream is ready (default, safe for files)
    /// - `Drop`: Drop frames when busy (best for live sources)
    /// - `RingBuffer`: Buffer frames, drop oldest when full
    /// - `Adaptive`: Start blocking, switch to dropping after timeout
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// impl Source for ScreenCaptureSource {
    ///     fn flow_policy(&self) -> FlowPolicy {
    ///         FlowPolicy::Drop {
    ///             log_drops: true,
    ///             max_consecutive: Some(30), // Error after 1s at 30fps
    ///         }
    ///     }
    /// }
    /// ```
    fn flow_policy(&self) -> crate::pipeline::flow::FlowPolicy {
        crate::pipeline::flow::FlowPolicy::Block // Safe default
    }

    /// How many buffers the downstream graph can hold in flight.
    ///
    /// See [`Element::set_output_budget`] — same contract, same reason to
    /// override it: this source owns the arena its output buffers come from.
    fn set_output_budget(&mut self, _budget: crate::memory::OutputBudget) {
        // Default: ignore
    }
}

/// An async source element that produces buffers asynchronously.
///
/// Use this for sources that need to await I/O operations, such as
/// network receivers or async file readers.
///
/// # Example
///
/// ```rust,ignore
/// struct TcpSource {
///     reader: TcpStream,
/// }
///
/// impl AsyncSource for TcpSource {
///     async fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
///         let output = ctx.output();
///         let n = self.reader.read(output).await?;
///         if n == 0 {
///             return Ok(ProduceResult::Eos); // Connection closed
///         }
///         Ok(ProduceResult::Produced(n))
///     }
/// }
/// ```
pub trait AsyncSource: Send {
    /// Produce the next buffer asynchronously.
    ///
    /// The framework provides a `ProduceContext` with a pre-allocated output buffer.
    fn produce(
        &mut self,
        ctx: &mut ProduceContext<'_>,
    ) -> impl std::future::Future<Output = Result<ProduceResult>> + Send;

    /// Get the name of this source (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the output caps (what formats this source produces).
    fn output_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the output media caps with format+memory constraints.
    ///
    /// Default implementation converts from `output_caps()` assuming CPU memory.
    fn output_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.output_caps())
    }

    /// Get the preferred buffer size for this source.
    fn preferred_buffer_size(&self) -> Option<usize> {
        None
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        // Async sources are typically I/O-bound
        ExecutionHints::io_bound()
    }

    /// Handle an upstream event (seek, QoS, etc.).
    ///
    /// Sources are typically the handlers of upstream events like seek.
    /// Return `EventResult::Handled` if the event was processed,
    /// `EventResult::NotHandled` otherwise.
    ///
    /// Default implementation does not handle any events.
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }

    /// Whether this async source supports seeking.
    fn is_seekable(&self) -> bool {
        false
    }

    /// Query the current position.
    fn query_position(&self) -> Option<crate::pipeline::seek::PositionQuery> {
        None
    }

    /// Query the total duration.
    fn query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        None
    }

    /// Handle a flow control signal from downstream.
    ///
    /// This is called when downstream elements signal backpressure.
    /// Default implementation ignores the signal (backward compatible).
    fn handle_flow_signal(&mut self, _signal: crate::pipeline::flow::FlowSignal) {
        // Default: ignore (backward compatibility)
    }

    /// Get the flow control policy for this source.
    ///
    /// Default is `Block` which waits when downstream is busy.
    fn flow_policy(&self) -> crate::pipeline::flow::FlowPolicy {
        crate::pipeline::flow::FlowPolicy::Block
    }

    /// How many buffers the downstream graph can hold in flight.
    ///
    /// See [`Element::set_output_budget`] — same contract, same reason to
    /// override it: this source owns the arena its output buffers come from.
    fn set_output_budget(&mut self, _budget: crate::memory::OutputBudget) {
        // Default: ignore
    }
}

// ============================================================================
// Sink Trait
// ============================================================================

/// A sink element that consumes buffers.
///
/// Sinks are the exit points of a pipeline. They write data to external
/// destinations like files, network connections, or displays.
///
/// # ConsumeContext
///
/// The framework provides a [`ConsumeContext`] that gives read-only access
/// to the buffer data and metadata. This ensures the sink cannot accidentally
/// modify buffers that may be shared with other elements.
///
/// # Example
///
/// ```rust,ignore
/// struct FileSink {
///     file: std::fs::File,
/// }
///
/// impl Sink for FileSink {
///     fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
///         self.file.write_all(ctx.input())?;
///         Ok(())
///     }
/// }
/// ```
pub trait Sink: Send {
    /// Consume a buffer.
    ///
    /// The `ConsumeContext` provides read-only access to the buffer data
    /// and metadata.
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()>;

    /// Get the name of this sink (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the input caps (what formats this sink accepts).
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the input media caps with format+memory constraints.
    ///
    /// Default implementation converts from `input_caps()` assuming CPU memory.
    fn input_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.input_caps())
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// Handle a downstream event (EOS, segment, tags, etc.).
    ///
    /// Sinks receive downstream events and can process them.
    /// Return `Some(event)` to indicate the event was seen (for logging),
    /// or `None` to consume it silently.
    ///
    /// Default implementation forwards all events.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)
    }

    /// Get a clock provider if this sink can provide a clock.
    ///
    /// Audio sinks typically provide a hardware clock that can be used
    /// as the pipeline's master clock for A/V synchronization.
    fn as_clock_provider(&self) -> Option<&dyn ClockProvider> {
        None
    }
}

/// An async sink element that consumes buffers asynchronously.
///
/// Use this for sinks that need to await I/O operations, such as
/// network writers or async file writers.
///
/// # Example
///
/// ```rust,ignore
/// struct TcpSink {
///     writer: TcpStream,
/// }
///
/// impl AsyncSink for TcpSink {
///     async fn consume(&mut self, ctx: &ConsumeContext<'_>) -> Result<()> {
///         self.writer.write_all(ctx.input()).await?;
///         Ok(())
///     }
/// }
/// ```
pub trait AsyncSink: Send {
    /// Consume a buffer asynchronously.
    ///
    /// The `ConsumeContext` provides read-only access to the buffer data.
    fn consume(
        &mut self,
        ctx: &ConsumeContext<'_>,
    ) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Get the name of this sink (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the input caps (what formats this sink accepts).
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the input media caps with format+memory constraints.
    ///
    /// Default implementation converts from `input_caps()` assuming CPU memory.
    fn input_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.input_caps())
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        // Async sinks are typically I/O-bound
        ExecutionHints::io_bound()
    }

    /// Handle a downstream event (EOS, segment, tags, etc.).
    ///
    /// Sinks receive downstream events and can process them.
    /// Return `Some(event)` to indicate the event was seen (for logging),
    /// or `None` to consume it silently.
    ///
    /// Default implementation forwards all events.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)
    }

    /// Get a clock provider if this sink can provide a clock.
    ///
    /// Audio sinks typically provide a hardware clock that can be used
    /// as the pipeline's master clock for A/V synchronization.
    fn as_clock_provider(&self) -> Option<&dyn ClockProvider> {
        None
    }
}

// ============================================================================
// Element Trait (Legacy)
// ============================================================================

/// A transform element that processes buffers.
///
/// Elements sit in the middle of a pipeline, receiving buffers from
/// upstream and sending transformed buffers downstream.
///
/// # Return Values
///
/// - `Ok(Some(buffer))`: Emit a buffer downstream
/// - `Ok(None)`: Drop this buffer (filter it out)
/// - `Err(...)`: Signal an error
///
/// # Note
///
/// This is the legacy trait that returns `Option<Buffer>`. For new elements
/// that produce multiple outputs, use [`Transform`] which returns [`Output`].
///
/// # Example
///
/// ```rust,ignore
/// struct UppercaseFilter;
///
/// impl Element for UppercaseFilter {
///     fn process(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
///         // This is a simplified example - real implementation would
///         // need to handle the buffer's memory properly
///         Ok(Some(buffer))
///     }
/// }
/// ```
pub trait Element: Send {
    /// Process an input buffer and optionally produce an output buffer.
    ///
    /// Return `Ok(None)` to filter out (drop) the buffer.
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>>;

    /// Flush any buffered data at end-of-stream.
    ///
    /// Called by the executor when EOS is received. Elements that buffer
    /// data (like video encoders) should drain their internal buffers here.
    ///
    /// The executor will call this method repeatedly until it returns
    /// `Ok(None)`, allowing elements to produce multiple outputs during flush.
    ///
    /// Default implementation returns `Ok(None)` (no buffered data).
    fn flush(&mut self) -> Result<Option<Buffer>> {
        Ok(None)
    }

    /// Set the bus handle for posting messages.
    ///
    /// Called by the adapter before pipeline execution. Override this
    /// to receive a bus handle for posting QoS, error, or other messages.
    fn set_bus(&mut self, _bus: crate::pipeline::bus::BusHandle) {
        // Default: ignore
    }

    /// How many buffers the downstream graph can hold in flight.
    ///
    /// Called by the executor before the pipeline starts. Override it if this
    /// element owns an output [`SharedArena`](crate::memory::SharedArena):
    /// store the budget and size the arena from it, because an arena smaller
    /// than what the links can hold will run out of slots as soon as a consumer
    /// falls behind.
    ///
    /// Arenas are usually built lazily from the first frame's geometry, so
    /// store the budget rather than acting on it here — and consult it again on
    /// the rebuild that follows a resolution change:
    ///
    /// ```rust,ignore
    /// fn set_output_budget(&mut self, budget: OutputBudget) {
    ///     self.budget = Some(budget);
    /// }
    ///
    /// // ...later, on the first frame:
    /// let slots = self.budget.unwrap_or_default().resolve(
    ///     defaults::VIDEO_ENCODER_SLOT_COUNT, slot_size);
    /// self.arena = Some(SharedArena::new(slot_size, slots)?);
    /// ```
    ///
    /// Default: ignore, which is right for any element that does not allocate
    /// its own output buffers.
    fn set_output_budget(&mut self, _budget: crate::memory::OutputBudget) {
        // Default: ignore
    }

    /// Get the name of this element (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the input caps (what formats this element accepts).
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the output caps (what formats this element produces).
    fn output_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the input media caps with format+memory constraints.
    ///
    /// Default implementation converts from `input_caps()` assuming CPU memory.
    fn input_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.input_caps())
    }

    /// Get the output media caps with format+memory constraints.
    ///
    /// Default implementation converts from `output_caps()` assuming CPU memory.
    fn output_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.output_caps())
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// Handle a downstream event (EOS, segment, tags, etc.).
    ///
    /// Transform elements can intercept and modify events.
    /// Return `Some(event)` to forward, `None` to consume.
    ///
    /// Default implementation forwards all events.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)
    }

    /// Handle an upstream event (seek, QoS, etc.).
    ///
    /// Return `EventResult::Handled` if processed, `EventResult::NotHandled` to pass upstream.
    ///
    /// Default implementation passes all events upstream.
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }
}

// ============================================================================
// Transform Trait (New)
// ============================================================================

/// A transform element that processes buffers and may produce multiple outputs.
///
/// This is the modern version of [`Element`] that returns [`Output`] instead
/// of `Option<Buffer>`, allowing elements to produce zero, one, or multiple
/// output buffers per input.
///
/// # When to Use
///
/// - Use `Transform` for new elements that may produce multiple outputs
/// - Use `Element` for simple 1-to-1 or filter elements
///
/// # Example
///
/// ```rust,ignore
/// struct LineSplitter;
///
/// impl Transform for LineSplitter {
///     fn transform(&mut self, buffer: Buffer) -> Result<Output> {
///         let lines: Vec<Buffer> = buffer.as_bytes()
///             .split(|&b| b == b'\n')
///             .filter(|line| !line.is_empty())
///             .map(|line| /* create buffer from line */)
///             .collect();
///         Ok(Output::from(lines))
///     }
/// }
/// ```
pub trait Transform: Send {
    /// Transform an input buffer into output(s).
    fn transform(&mut self, buffer: Buffer) -> Result<Output>;

    /// Flush any buffered data at end-of-stream.
    ///
    /// Called by the executor when EOS is received. Transforms that buffer
    /// data (like video encoders) should drain their internal buffers here.
    ///
    /// The executor will call this method repeatedly until it returns
    /// `Ok(Output::None)`, allowing transforms to produce multiple outputs during flush.
    ///
    /// Default implementation returns `Ok(Output::None)` (no buffered data).
    fn flush(&mut self) -> Result<Output> {
        Ok(Output::None)
    }

    /// Get the name of this transform (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the input caps (what formats this transform accepts).
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the output caps (what formats this transform produces).
    fn output_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the input media caps with format+memory constraints.
    ///
    /// Default implementation converts from `input_caps()` assuming CPU memory.
    fn input_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.input_caps())
    }

    /// Get the output media caps with format+memory constraints.
    ///
    /// Default implementation converts from `output_caps()` assuming CPU memory.
    fn output_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.output_caps())
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// Handle a downstream event (EOS, segment, tags, etc.).
    ///
    /// Transform elements can intercept and modify events.
    /// Return `Some(event)` to forward, `None` to consume.
    ///
    /// Default implementation forwards all events.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)
    }

    /// Handle an upstream event (seek, QoS, etc.).
    ///
    /// Return `EventResult::Handled` if processed, `EventResult::NotHandled` to pass upstream.
    ///
    /// Default implementation passes all events upstream.
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }

    /// How many buffers the downstream graph can hold in flight.
    ///
    /// See [`Element::set_output_budget`] — same contract, same reason to
    /// override it: this transform owns the arena its output buffers come from.
    fn set_output_budget(&mut self, _budget: crate::memory::OutputBudget) {
        // Default: ignore
    }
}

/// An async transform element.
///
/// Like [`Transform`] but for elements that need async I/O.
pub trait AsyncTransform: Send {
    /// Transform an input buffer asynchronously.
    fn transform(
        &mut self,
        buffer: Buffer,
    ) -> impl std::future::Future<Output = Result<Output>> + Send;

    /// Get the name of this transform (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the input caps (what formats this transform accepts).
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the output caps (what formats this transform produces).
    fn output_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the input media caps with format+memory constraints.
    ///
    /// Default implementation converts from `input_caps()` assuming CPU memory.
    fn input_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.input_caps())
    }

    /// Get the output media caps with format+memory constraints.
    ///
    /// Default implementation converts from `output_caps()` assuming CPU memory.
    fn output_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.output_caps())
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        // Async transforms are typically I/O-bound
        ExecutionHints::io_bound()
    }

    /// Flush any buffered data at end-of-stream.
    ///
    /// Called when EOS is received. Transforms that buffer data
    /// (like encoders) should emit remaining buffers here.
    fn flush(&mut self) -> impl std::future::Future<Output = Result<Output>> + Send {
        async { Ok(Output::None) }
    }

    /// Handle a downstream event (EOS, segment, tags, etc.).
    ///
    /// Transform elements can intercept and modify events.
    /// Return `Some(event)` to forward, `None` to consume.
    ///
    /// Default implementation forwards all events.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)
    }

    /// Handle an upstream event (seek, QoS, etc.).
    ///
    /// Return `EventResult::Handled` if processed, `EventResult::NotHandled` to pass upstream.
    ///
    /// Default implementation passes all events upstream.
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }

    /// How many buffers the downstream graph can hold in flight.
    ///
    /// See [`Element::set_output_budget`] — same contract, same reason to
    /// override it: this transform owns the arena its output buffers come from.
    fn set_output_budget(&mut self, _budget: crate::memory::OutputBudget) {
        // Default: ignore
    }
}

// Blanket implementation: Element implements Transform
impl<T: Element> Transform for T {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        self.process(buffer).map(Output::from)
    }

    fn name(&self) -> &str {
        Element::name(self)
    }

    fn input_caps(&self) -> Caps {
        Element::input_caps(self)
    }

    fn output_caps(&self) -> Caps {
        Element::output_caps(self)
    }

    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        Element::set_output_budget(self, budget);
    }
}

// ============================================================================
// Demuxer Trait
// ============================================================================

/// Output pad identifier for demuxers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PadId(pub u32);

impl PadId {
    /// Create a new pad ID.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
}

impl From<u32> for PadId {
    fn from(id: u32) -> Self {
        Self(id)
    }
}

impl From<PadId> for u32 {
    fn from(id: PadId) -> Self {
        id.0
    }
}

/// Routed output for demuxers (buffer with destination pad).
///
/// Uses SmallVec to avoid allocation for common cases (1-2 outputs).
#[derive(Debug)]
pub struct RoutedOutput(pub SmallVec<[(PadId, Buffer); 2]>);

impl RoutedOutput {
    /// Create an empty routed output.
    pub fn new() -> Self {
        Self(SmallVec::new())
    }

    /// Create a routed output with a single buffer.
    pub fn single(pad: PadId, buffer: Buffer) -> Self {
        let mut r = Self::new();
        r.push(pad, buffer);
        r
    }

    /// Add a buffer to a specific pad.
    pub fn push(&mut self, pad: PadId, buffer: Buffer) {
        self.0.push((pad, buffer));
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Get the number of routed buffers.
    pub fn len(&self) -> usize {
        self.0.len()
    }
}

impl Default for RoutedOutput {
    fn default() -> Self {
        Self::new()
    }
}

impl IntoIterator for RoutedOutput {
    type Item = (PadId, Buffer);
    type IntoIter = smallvec::IntoIter<[(PadId, Buffer); 2]>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a RoutedOutput {
    type Item = &'a (PadId, Buffer);
    type IntoIter = std::slice::Iter<'a, (PadId, Buffer)>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

/// Callback type for pad addition events.
pub type PadAddedCallback = Box<dyn FnMut(PadId, Caps) + Send>;

/// A demuxer element that routes input to multiple output pads.
///
/// Demuxers are used for splitting a single stream into multiple streams
/// based on content (e.g., MPEG-TS demuxer splitting into video and audio).
///
/// # Dynamic Pads
///
/// Demuxers can have dynamic pads that are created at runtime as streams
/// are discovered. Use `on_pad_added` to receive notifications.
///
/// # Example
///
/// ```rust,ignore
/// struct MyDemuxer {
///     outputs: Vec<(PadId, Caps)>,
///     callback: Option<PadAddedCallback>,
/// }
///
/// impl Demuxer for MyDemuxer {
///     fn demux(&mut self, buffer: Buffer) -> Result<RoutedOutput> {
///         // Parse buffer and route to appropriate output
///         let mut output = RoutedOutput::new();
///         output.push(PadId(0), buffer);
///         Ok(output)
///     }
///
///     fn outputs(&self) -> &[(PadId, Caps)] {
///         &self.outputs
///     }
///
///     fn on_pad_added(&mut self, callback: PadAddedCallback) {
///         self.callback = Some(callback);
///     }
/// }
/// ```
pub trait Demuxer: Send {
    /// Process input and route to output pads.
    fn demux(&mut self, buffer: Buffer) -> Result<RoutedOutput>;

    /// Produce routed output without input.
    ///
    /// For demuxers that own their reader (a file demuxer) and act as the
    /// pipeline's *source*: a node added with no input links is driven
    /// through this instead of [`demux`](Self::demux). The default reports
    /// EOS immediately — input-driven demuxers have nothing to produce.
    fn produce(&mut self) -> Result<DemuxerProduce> {
        Ok(DemuxerProduce::Eos)
    }

    /// The link-level name of an output pad.
    ///
    /// Buffers routed to `pad` are delivered only to links attached with
    /// this name (`pipeline.link_pads(demux, name, ..)`). The default is
    /// `src_{n}`; demuxers with semantic pads override it (`"video"`,
    /// `"audio"`).
    fn pad_name(&self, pad: PadId) -> String {
        format!("src_{}", pad.0)
    }

    /// Handle an upstream event (seek, QoS).
    ///
    /// A *source-style* demuxer receives runtime events exactly like a
    /// source does — `PipelineHandle::seek` lands here. Return `Handled`
    /// after repositioning and the executor runs the flush sequence on
    /// every output pad.
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }

    /// Whether this demuxer supports seeking.
    ///
    /// A source-style demuxer that implements `handle_upstream_event` for
    /// [`Event::Seek`] must return `true` — the executor snapshots this at
    /// start and *gates* `PipelineHandle::seek*` on it: seeks are only ever
    /// dispatched to elements that declared support, exactly like
    /// GStreamer's `GST_QUERY_SEEKING`.
    fn is_seekable(&self) -> bool {
        false
    }

    /// Current position, if this demuxer tracks one.
    fn query_position(&self) -> Option<crate::pipeline::seek::PositionQuery> {
        None
    }

    /// Stream duration, if known.
    ///
    /// Snapshotted at start and served by `PipelineHandle::duration()`, so
    /// applications no longer need to reach around the framework to the
    /// demuxer object.
    fn query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        None
    }

    /// Get the name of this demuxer (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the current output pads and their formats.
    fn outputs(&self) -> &[(PadId, Caps)];

    /// Get the input caps (what formats this demuxer accepts).
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Register a callback for when new pads are added.
    ///
    /// This is called when the demuxer discovers new streams in the input.
    fn on_pad_added(&mut self, callback: PadAddedCallback);

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// How many buffers the downstream graph can hold in flight.
    ///
    /// See [`Element::set_output_budget`] — same contract, same reason to
    /// override it: this demuxer owns the arena its output buffers come from.
    fn set_output_budget(&mut self, _budget: crate::memory::OutputBudget) {
        // Default: ignore
    }
}

/// Input from a specific pad for muxers.
#[derive(Debug)]
pub struct MuxerInput {
    /// The pad ID this buffer came from.
    pub pad: PadId,
    /// The buffer data.
    pub buffer: Buffer,
}

impl MuxerInput {
    /// Create a new muxer input.
    pub fn new(pad: PadId, buffer: Buffer) -> Self {
        Self { pad, buffer }
    }
}

/// A muxer element that combines multiple input pads into one output.
///
/// Muxers are used for combining multiple streams into a single stream
/// (e.g., combining video and audio into a container format).
///
/// # Dynamic Pads
///
/// Muxers can have dynamic input pads that are created at runtime.
/// Use `on_pad_added` to receive notifications.
///
/// # Example
///
/// ```rust,ignore
/// struct MyMuxer {
///     inputs: Vec<(PadId, Caps)>,
///     callback: Option<PadAddedCallback>,
/// }
///
/// impl Muxer for MyMuxer {
///     fn mux(&mut self, input: MuxerInput) -> Result<Option<Buffer>> {
///         // Combine inputs and produce output
///         // Return Some(buffer) when ready to output
///         Ok(Some(input.buffer))
///     }
///
///     fn inputs(&self) -> &[(PadId, Caps)] {
///         &self.inputs
///     }
///
///     fn on_pad_added(&mut self, callback: PadAddedCallback) {
///         self.callback = Some(callback);
///     }
/// }
/// ```
pub trait Muxer: Send {
    /// Accept a buffer from an input pad and optionally produce output.
    ///
    /// The muxer receives buffers from multiple input pads and combines them.
    /// It returns `Some(buffer)` when it has produced an output buffer,
    /// or `None` if it's still waiting for more data.
    fn mux(&mut self, input: MuxerInput) -> Result<Option<Buffer>>;

    /// Get the name of this muxer (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the current input pads and their formats.
    fn inputs(&self) -> &[(PadId, Caps)];

    /// Get the output caps (what format this muxer produces).
    fn output_caps(&self) -> Caps {
        Caps::any()
    }

    /// Register a callback for when new input pads are added.
    ///
    /// This is called when the muxer accepts a new input stream.
    fn on_pad_added(&mut self, callback: PadAddedCallback);

    /// Flush the muxer and produce any remaining output.
    ///
    /// Called when EOS is received on all input pads.
    fn flush(&mut self) -> Result<Option<Buffer>> {
        Ok(None)
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// How many buffers the downstream graph can hold in flight.
    ///
    /// See [`Element::set_output_budget`] — same contract, same reason to
    /// override it: this muxer owns the arena its output buffers come from.
    fn set_output_budget(&mut self, _budget: crate::memory::OutputBudget) {
        // Default: ignore
    }
}

// ============================================================================
// Dynamic Async Element (Type-Erased)
// ============================================================================

/// The type of an element in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// A source element (produces buffers).
    Source,
    /// A sink element (consumes buffers).
    Sink,
    /// A transform element (transforms buffers).
    Transform,
    /// A demuxer element (one input, multiple outputs).
    Demuxer,
    /// A muxer element (multiple inputs, one output).
    Muxer,
}

/// Async dynamic (type-erased) element trait for pipeline execution.
///
/// This trait is used internally by the pipeline executor to handle
/// elements uniformly with native async support.
///
/// The `#[dynosaur]` macro generates `DynAsyncElement` which provides
/// object-safe async dispatch. The `SendAsyncElementDyn` variant ensures
/// the futures are `Send` for use with multi-threaded executors.
///
/// # Usage
///
/// Most users should implement [`Source`], [`Sink`], [`AsyncSource`],
/// [`AsyncSink`], or [`Transform`] and use the corresponding adapter.
#[trait_variant::make(SendAsyncElementDyn: Send)]
#[dynosaur(pub DynAsyncElement = dyn(box) SendAsyncElementDyn, bridge(dyn))]
pub trait AsyncElementDyn {
    /// Get the element's name.
    fn name(&self) -> &str;

    /// Get the element's type (source, sink, or transform).
    fn element_type(&self) -> ElementType;

    /// Process or produce a buffer asynchronously.
    ///
    /// - For sources: `input` is `None`, returns produced buffer
    /// - For sinks: `input` is `Some`, returns `None`
    /// - For transforms: `input` is `Some`, returns transformed buffer
    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>>;

    /// Process and return all outputs (for transforms that produce multiple).
    ///
    /// Default implementation wraps `process` in a single-element output.
    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output>;

    /// Get the input caps (for validation).
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the output caps (for validation).
    fn output_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the input media caps with format+memory constraints.
    ///
    /// Default implementation converts from `input_caps()` assuming CPU memory.
    fn input_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.input_caps())
    }

    /// Get the output media caps with format+memory constraints.
    ///
    /// Default implementation converts from `output_caps()` assuming CPU memory.
    fn output_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::from(self.output_caps())
    }

    /// Get execution hints for automatic scheduling decisions.
    ///
    /// These hints help the executor determine the optimal execution
    /// strategy for each element.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// Flush any buffered data at end-of-stream.
    ///
    /// Called by the executor when EOS is received. Elements that
    /// buffer data (like encoders) should emit remaining buffers here.
    ///
    /// Default implementation returns no output.
    fn flush(&mut self) -> impl std::future::Future<Output = Result<Output>> + Send {
        async { Ok(Output::None) }
    }

    /// Handle a downstream event (flows with data: EOS, segment, tags, etc.).
    ///
    /// Called when an event arrives from upstream. The element can:
    /// - Process the event and return `Some(event)` to forward it
    /// - Consume the event and return `None` to stop propagation
    /// - Modify the event before forwarding
    ///
    /// Default implementation forwards all events unchanged.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)
    }

    /// Handle an upstream event (flows against data: seek, QoS, etc.).
    ///
    /// Called when an event arrives from downstream. The element can:
    /// - Handle the event and return `EventResult::Handled`
    /// - Pass it upstream by returning `EventResult::NotHandled`
    ///
    /// Default implementation passes all events upstream.
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }

    /// Whether this element's underlying source supports seeking.
    fn is_seekable(&self) -> bool {
        false
    }

    /// Query the current position from the underlying source.
    fn source_query_position(&self) -> Option<crate::pipeline::seek::PositionQuery> {
        None
    }

    /// Query the total duration from the underlying source.
    fn source_query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        None
    }

    /// Process a source element, properly distinguishing WouldBlock from Eos.
    ///
    /// This method is used by the executor for source elements to handle
    /// the case where no data is available yet (WouldBlock) differently
    /// from end of stream (Eos).
    ///
    /// Default implementation returns `Eos`. Source adapters must override
    /// this to properly handle buffer production and `WouldBlock`.
    fn process_source(&mut self) -> impl std::future::Future<Output = Result<SourceResult>> + Send {
        async { Ok(SourceResult::Eos) }
    }

    /// Process a demuxer element, keeping the pad routing.
    ///
    /// `Some(input)` demuxes one buffer into per-pad outputs; `None` asks a
    /// *source-style* demuxer (one that owns its reader) to produce. Only
    /// `ElementType::Demuxer` nodes are driven through this — `process`
    /// erases which pad a buffer belongs on, which is exactly the executor
    /// bug this method exists to fix (#76).
    ///
    /// Default implementation reports EOS, like `process_source`;
    /// [`DemuxerAdapter`] overrides it.
    fn process_demux(
        &mut self,
        _input: Option<Buffer>,
    ) -> impl std::future::Future<Output = Result<DemuxResult>> + Send {
        async { Ok(DemuxResult::Eos) }
    }

    /// Set the pipeline clock for this element.
    ///
    /// Called by the executor when starting the pipeline to provide
    /// source elements with access to the pipeline clock. Sources can
    /// use this to calculate running time from their hardware timestamps.
    ///
    /// Default implementation does nothing. Source adapters override this
    /// to pass the clock to `ProduceContext`.
    fn set_clock(&mut self, _clock: Arc<dyn Clock>, _base_time: ClockTime) {
        // Default: do nothing
    }

    /// Set the bus handle for this element.
    ///
    /// Called by the executor when starting the pipeline so elements
    /// can post messages (errors, warnings, tags, QoS) to the bus.
    ///
    /// Default implementation does nothing. Adapters override this to
    /// store the handle and pass it to element contexts.
    fn set_bus(&mut self, _bus: crate::pipeline::bus::BusHandle) {
        // Default: do nothing
    }

    /// Tell this element how many buffers the downstream graph can hold.
    ///
    /// Called by the executor when starting the pipeline, before the element
    /// moves into its task. Elements that own an output arena size it from
    /// this; see [`Element::set_output_budget`].
    ///
    /// Default implementation does nothing. Adapters override this to forward
    /// to the element they wrap.
    fn set_output_budget(&mut self, _budget: crate::memory::OutputBudget) {
        // Default: do nothing
    }

    /// Get the inner element as `&dyn Any` for downcasting.
    ///
    /// This enables GStreamer-like element retrieval after pipeline creation:
    ///
    /// ```rust,ignore
    /// let pipeline = Pipeline::parse("filesrc location=test.bin ! sink")?;
    /// if let Some(filesrc) = pipeline.get_element::<FileSrc>("filesrc0") {
    ///     filesrc.set_location("other.bin");
    /// }
    /// ```
    ///
    /// Adapters must implement this to return the inner element type.
    fn as_any(&self) -> &dyn Any;

    /// Get the inner element as `&mut dyn Any` for mutable downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Get a clock provider interface if this element can provide a clock.
    ///
    /// Elements that implement [`ClockProvider`] (e.g., audio sinks with
    /// hardware clocks) return `Some`. All others return `None`.
    ///
    /// Used by [`Pipeline::select_clock()`](crate::pipeline::Pipeline::select_clock) to auto-select the best clock.
    fn as_clock_provider(&self) -> Option<&dyn ClockProvider> {
        None
    }

    /// Get a sync processing interface if this element supports it.
    ///
    /// Elements that implement [`SyncElement`] and are wrapped via
    /// [`SyncElementAdapter`] return `Some`. All others return `None`.
    ///
    /// This is used by the RT data thread to dispatch processing without
    /// going through the async `process()` method.
    fn as_sync_element(&mut self) -> Option<&mut dyn SyncElement> {
        None
    }
}

// ============================================================================
// Sync Element Trait (RT-safe)
// ============================================================================

/// Trait for RT-safe synchronous element processing.
///
/// Elements implementing this trait can run in the RT data thread
/// without blocking on async operations. They must not:
/// - Allocate memory (no `Vec::push`, no `Box::new`)
/// - Block on I/O (no file/network operations)
/// - Take locks that might be held by non-RT threads
///
/// The buffer is processed in-place or replaced.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::element::SyncElement;
/// use parallax::buffer::Buffer;
/// use parallax::error::Result;
///
/// struct Gain { factor: f32 }
///
/// impl SyncElement for Gain {
///     fn process_sync(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
///         let data = buffer.data_mut();
///         for sample in data.chunks_exact_mut(4) {
///             let val = f32::from_le_bytes(sample.try_into().unwrap());
///             let out = val * self.factor;
///             sample.copy_from_slice(&out.to_le_bytes());
///         }
///         Ok(Some(buffer))
///     }
/// }
/// ```
pub trait SyncElement: Send {
    /// Process a buffer synchronously. Returns `None` for filtered/dropped buffers.
    fn process_sync(&mut self, buffer: Buffer) -> Result<Option<Buffer>>;

    /// Flush at EOS. Returns any buffered data.
    fn flush_sync(&mut self) -> Result<Option<Buffer>> {
        Ok(None)
    }

    /// How many buffers the downstream graph can hold in flight.
    ///
    /// See [`Element::set_output_budget`] — same contract, same reason to
    /// override it: this element owns the arena its output buffers come from.
    fn set_output_budget(&mut self, _budget: crate::memory::OutputBudget) {
        // Default: ignore
    }
}

// Blanket impl: every sync Element is also a SyncElement.
// This allows ElementAdapter to return as_sync_element() for any Element,
// but the RT thread only uses it when is_rt_safe() is true.
impl<T: Element> SyncElement for T {
    fn process_sync(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.process(buffer)
    }

    fn flush_sync(&mut self) -> Result<Option<Buffer>> {
        self.flush()
    }

    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        Element::set_output_budget(self, budget);
    }
}

/// Adapter: [`SyncElement`] → [`AsyncElementDyn`].
///
/// Wraps a `SyncElement` so it can be used in the pipeline graph
/// (which requires `AsyncElementDyn`). In the RT data thread, the
/// executor calls `as_sync_element()` to bypass the async path entirely.
pub struct SyncElementAdapter<T: SyncElement> {
    inner: T,
    name: String,
}

impl<T: SyncElement> SyncElementAdapter<T> {
    /// Create a new sync element adapter.
    pub fn new(element: T, name: impl Into<String>) -> Self {
        Self {
            inner: element,
            name: name.into(),
        }
    }

    /// Get a reference to the inner element.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get a mutable reference to the inner element.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }
}

impl<T: SyncElement + 'static> SendAsyncElementDyn for SyncElementAdapter<T> {
    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.inner.set_output_budget(budget);
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn element_type(&self) -> ElementType {
        ElementType::Transform
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        match input {
            Some(buffer) => self.inner.process_sync(buffer),
            None => Ok(None),
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        match input {
            Some(buffer) => self.inner.process_sync(buffer).map(Output::from),
            None => Ok(Output::None),
        }
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::rt_safe()
    }

    async fn flush(&mut self) -> Result<Output> {
        self.inner.flush_sync().map(Output::from)
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }

    fn as_sync_element(&mut self) -> Option<&mut dyn SyncElement> {
        Some(&mut self.inner)
    }
}

// ============================================================================
// Adapters for Sync Elements -> AsyncElementDyn
// ============================================================================

/// Wrapper to adapt a sync [`Source`] to [`AsyncElementDyn`].
///
/// Sync sources run directly in the async context without blocking,
/// as they are typically fast CPU operations.
///
/// # Pool-Aware Execution
///
/// The adapter holds an optional arena for providing pre-allocated buffers
/// to the source via `ProduceContext`. Set the arena with [`set_arena()`](Self::set_arena).
pub struct SourceAdapter<S: Source> {
    inner: S,
    /// Arena for pre-allocated buffers (uses SharedArena for cross-process support).
    arena: Option<crate::memory::SharedArena>,
    /// Buffer pool for high-level pool API with backpressure.
    pool: Option<std::sync::Arc<dyn crate::memory::BufferPool>>,
    /// Pipeline clock for timestamp calculation.
    clock: Option<Arc<dyn Clock>>,
    /// Base time for running time calculation.
    base_time: ClockTime,
    /// Bus handle for posting messages.
    bus: Option<crate::pipeline::bus::BusHandle>,
}

impl<S: Source> SourceAdapter<S> {
    /// Create a new source adapter.
    pub fn new(source: S) -> Self {
        Self {
            inner: source,
            arena: None,
            pool: None,
            clock: None,
            base_time: ClockTime::ZERO,
            bus: None,
        }
    }

    /// Create a source adapter with an arena for buffer allocation.
    pub fn with_arena(source: S, arena: crate::memory::SharedArena) -> Self {
        Self {
            inner: source,
            arena: Some(arena),
            pool: None,
            clock: None,
            base_time: ClockTime::ZERO,
            bus: None,
        }
    }

    /// Create a source adapter with a buffer pool.
    pub fn with_pool(source: S, pool: std::sync::Arc<dyn crate::memory::BufferPool>) -> Self {
        Self {
            inner: source,
            arena: None,
            pool: Some(pool),
            clock: None,
            base_time: ClockTime::ZERO,
            bus: None,
        }
    }

    /// Set the arena for buffer allocation.
    pub fn set_arena(&mut self, arena: crate::memory::SharedArena) {
        self.arena = Some(arena);
    }

    /// Set the buffer pool for high-level pool API.
    pub fn set_pool(&mut self, pool: std::sync::Arc<dyn crate::memory::BufferPool>) {
        self.pool = Some(pool);
    }

    /// Get a reference to the inner source.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get a mutable reference to the inner source.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S: Source + Send + 'static> SendAsyncElementDyn for SourceAdapter<S> {
    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.inner.set_output_budget(budget);
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Source
    }

    async fn process(&mut self, _input: Option<Buffer>) -> Result<Option<Buffer>> {
        // Priority: pool > arena > no buffer
        //
        // If a pool is configured, use it (provides backpressure and stats).
        // If only arena is configured, use it (simpler, no backpressure).
        // Otherwise, source must provide its own buffer.

        if let Some(pool) = &self.pool {
            // Pool-aware path: source can use ctx.acquire_buffer() or we provide a slot
            if let Some(arena) = &self.arena {
                // Have both pool and arena - provide slot and pool access
                if let Some(slot) = arena.acquire() {
                    let mut ctx = ProduceContext::with_pool(slot, pool.as_ref());
                    match self.inner.produce(&mut ctx)? {
                        ProduceResult::Produced(n) => Ok(Some(ctx.finalize(n))),
                        ProduceResult::Eos => Ok(None),
                        ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
                        ProduceResult::OwnDmaBuf(dmabuf) => Ok(Some(dmabuf.to_buffer(arena)?)),
                        ProduceResult::WouldBlock => Ok(None),
                    }
                } else {
                    // Arena exhausted - provide pool-only access
                    let mut ctx = ProduceContext::with_pool_only(pool.as_ref());
                    match self.inner.produce(&mut ctx)? {
                        ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
                        ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                            "arena exhausted, cannot convert DmaBuf to Buffer".into(),
                        )),
                        ProduceResult::Eos => Ok(None),
                        ProduceResult::WouldBlock => Ok(None),
                        ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                            "arena exhausted and source doesn't provide own buffer".into(),
                        )),
                    }
                }
            } else {
                // Pool only - source must use ctx.acquire_buffer() or provide own
                let mut ctx = ProduceContext::with_pool_only(pool.as_ref());
                match self.inner.produce(&mut ctx)? {
                    ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
                    ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                        "no arena configured, cannot convert DmaBuf to Buffer".into(),
                    )),
                    ProduceResult::Eos => Ok(None),
                    ProduceResult::WouldBlock => Ok(None),
                    ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                        "no arena configured and source doesn't provide own buffer".into(),
                    )),
                }
            }
        } else if let Some(arena) = &self.arena {
            // Arena-only path (legacy)
            if let Some(slot) = arena.acquire() {
                let mut ctx = ProduceContext::new(slot);
                match self.inner.produce(&mut ctx)? {
                    ProduceResult::Produced(n) => Ok(Some(ctx.finalize(n))),
                    ProduceResult::Eos => Ok(None),
                    ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
                    ProduceResult::OwnDmaBuf(dmabuf) => Ok(Some(dmabuf.to_buffer(arena)?)),
                    ProduceResult::WouldBlock => Ok(None),
                }
            } else {
                // Arena exhausted
                let mut ctx = ProduceContext::without_buffer();
                match self.inner.produce(&mut ctx)? {
                    ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
                    ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                        "arena exhausted, cannot convert DmaBuf to Buffer".into(),
                    )),
                    ProduceResult::Eos => Ok(None),
                    ProduceResult::WouldBlock => Ok(None),
                    ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                        "arena exhausted and source doesn't provide own buffer".into(),
                    )),
                }
            }
        } else {
            // No pool or arena - source must provide its own buffer
            let mut ctx = ProduceContext::without_buffer();
            match self.inner.produce(&mut ctx)? {
                ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
                ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                    "no arena configured, cannot convert DmaBuf to Buffer".into(),
                )),
                ProduceResult::Eos => Ok(None),
                ProduceResult::WouldBlock => Ok(None),
                ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                    "no arena configured and source doesn't provide own buffer".into(),
                )),
            }
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        AsyncElementDyn::process(self, input)
            .await
            .map(Output::from)
    }

    fn output_caps(&self) -> Caps {
        self.inner.output_caps()
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        self.inner.output_media_caps()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    async fn process_source(&mut self) -> Result<SourceResult> {
        // Helper to configure clock and bus on context
        let configure_clock = |ctx: &mut ProduceContext| {
            if let Some(ref clock) = self.clock {
                ctx.set_clock(clock.clone(), self.base_time);
            }
            if let Some(ref bus) = self.bus {
                ctx.set_bus(bus.clone());
            }
        };

        // This implementation properly distinguishes WouldBlock from Eos
        if let Some(pool) = &self.pool {
            if let Some(arena) = &self.arena {
                if let Some(slot) = arena.acquire() {
                    let mut ctx = ProduceContext::with_pool(slot, pool.as_ref());
                    configure_clock(&mut ctx);
                    match self.inner.produce(&mut ctx)? {
                        ProduceResult::Produced(n) => {
                            Ok(SourceResult::Buffer(Box::new(ctx.finalize(n))))
                        }
                        ProduceResult::Eos => Ok(SourceResult::Eos),
                        ProduceResult::OwnBuffer(buffer) => {
                            Ok(SourceResult::Buffer(Box::new(buffer)))
                        }
                        ProduceResult::OwnDmaBuf(dmabuf) => {
                            Ok(SourceResult::Buffer(Box::new(dmabuf.to_buffer(arena)?)))
                        }
                        ProduceResult::WouldBlock => Ok(SourceResult::WouldBlock),
                    }
                } else {
                    let mut ctx = ProduceContext::with_pool_only(pool.as_ref());
                    configure_clock(&mut ctx);
                    match self.inner.produce(&mut ctx)? {
                        ProduceResult::OwnBuffer(buffer) => {
                            Ok(SourceResult::Buffer(Box::new(buffer)))
                        }
                        ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                            "arena exhausted, cannot convert DmaBuf to Buffer".into(),
                        )),
                        ProduceResult::Eos => Ok(SourceResult::Eos),
                        ProduceResult::WouldBlock => Ok(SourceResult::WouldBlock),
                        ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                            "arena exhausted and source doesn't provide own buffer".into(),
                        )),
                    }
                }
            } else {
                let mut ctx = ProduceContext::with_pool_only(pool.as_ref());
                configure_clock(&mut ctx);
                match self.inner.produce(&mut ctx)? {
                    ProduceResult::OwnBuffer(buffer) => Ok(SourceResult::Buffer(Box::new(buffer))),
                    ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                        "no arena configured, cannot convert DmaBuf to Buffer".into(),
                    )),
                    ProduceResult::Eos => Ok(SourceResult::Eos),
                    ProduceResult::WouldBlock => Ok(SourceResult::WouldBlock),
                    ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                        "no arena configured and source doesn't provide own buffer".into(),
                    )),
                }
            }
        } else if let Some(arena) = &self.arena {
            if let Some(slot) = arena.acquire() {
                let mut ctx = ProduceContext::new(slot);
                configure_clock(&mut ctx);
                match self.inner.produce(&mut ctx)? {
                    ProduceResult::Produced(n) => {
                        Ok(SourceResult::Buffer(Box::new(ctx.finalize(n))))
                    }
                    ProduceResult::Eos => Ok(SourceResult::Eos),
                    ProduceResult::OwnBuffer(buffer) => Ok(SourceResult::Buffer(Box::new(buffer))),
                    ProduceResult::OwnDmaBuf(dmabuf) => {
                        Ok(SourceResult::Buffer(Box::new(dmabuf.to_buffer(arena)?)))
                    }
                    ProduceResult::WouldBlock => Ok(SourceResult::WouldBlock),
                }
            } else {
                let mut ctx = ProduceContext::without_buffer();
                configure_clock(&mut ctx);
                match self.inner.produce(&mut ctx)? {
                    ProduceResult::OwnBuffer(buffer) => Ok(SourceResult::Buffer(Box::new(buffer))),
                    ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                        "arena exhausted, cannot convert DmaBuf to Buffer".into(),
                    )),
                    ProduceResult::Eos => Ok(SourceResult::Eos),
                    ProduceResult::WouldBlock => Ok(SourceResult::WouldBlock),
                    ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                        "arena exhausted and source doesn't provide own buffer".into(),
                    )),
                }
            }
        } else {
            let mut ctx = ProduceContext::without_buffer();
            configure_clock(&mut ctx);
            match self.inner.produce(&mut ctx)? {
                ProduceResult::OwnBuffer(buffer) => Ok(SourceResult::Buffer(Box::new(buffer))),
                ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                    "no arena configured, cannot convert DmaBuf to Buffer".into(),
                )),
                ProduceResult::Eos => Ok(SourceResult::Eos),
                ProduceResult::WouldBlock => Ok(SourceResult::WouldBlock),
                ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                    "no arena configured and source doesn't provide own buffer".into(),
                )),
            }
        }
    }

    fn set_clock(&mut self, clock: Arc<dyn Clock>, base_time: ClockTime) {
        self.clock = Some(clock);
        self.base_time = base_time;
    }

    fn set_bus(&mut self, bus: crate::pipeline::bus::BusHandle) {
        self.bus = Some(bus);
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        self.inner.handle_upstream_event(event)
    }

    fn is_seekable(&self) -> bool {
        self.inner.is_seekable()
    }

    fn source_query_position(&self) -> Option<crate::pipeline::seek::PositionQuery> {
        self.inner.query_position()
    }

    fn source_query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        self.inner.query_duration()
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

/// Wrapper to adapt a sync [`Sink`] to [`AsyncElementDyn`].
pub struct SinkAdapter<S: Sink> {
    inner: S,
    /// The pipeline clock, once the executor hands it over at start.
    clock: Option<(Arc<dyn Clock>, ClockTime)>,
}

impl<S: Sink> SinkAdapter<S> {
    /// Create a new sink adapter.
    pub fn new(sink: S) -> Self {
        Self {
            inner: sink,
            clock: None,
        }
    }

    /// Get a reference to the inner sink.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get a mutable reference to the inner sink.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S: Sink + Send + 'static> SendAsyncElementDyn for SinkAdapter<S> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Sink
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        if let Some(ref buffer) = input {
            let mut ctx = ConsumeContext::new(buffer);
            if let Some((clock, base_time)) = &self.clock {
                ctx.set_clock(clock, *base_time);
            }
            self.inner.consume(&ctx)?;
        }
        Ok(None)
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        AsyncElementDyn::process(self, input)
            .await
            .map(Output::from)
    }

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        self.inner.input_media_caps()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        self.inner.handle_downstream_event(event)
    }

    fn set_clock(&mut self, clock: Arc<dyn Clock>, base_time: ClockTime) {
        self.clock = Some((clock, base_time));
    }

    fn as_clock_provider(&self) -> Option<&dyn ClockProvider> {
        self.inner.as_clock_provider()
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

/// Wrapper to adapt a sync [`Element`] to [`AsyncElementDyn`].
pub struct ElementAdapter<E: Element> {
    inner: E,
}

impl<E: Element> ElementAdapter<E> {
    /// Create a new element adapter.
    pub fn new(element: E) -> Self {
        Self { inner: element }
    }
}

impl<E: Element + Send + 'static> SendAsyncElementDyn for ElementAdapter<E> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Transform
    }

    fn set_bus(&mut self, bus: crate::pipeline::bus::BusHandle) {
        self.inner.set_bus(bus);
    }

    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.inner.set_output_budget(budget);
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        match input {
            Some(buffer) => self.inner.process(buffer),
            None => Ok(None),
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        AsyncElementDyn::process(self, input)
            .await
            .map(Output::from)
    }

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }

    fn output_caps(&self) -> Caps {
        self.inner.output_caps()
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        self.inner.input_media_caps()
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        self.inner.output_media_caps()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    async fn flush(&mut self) -> Result<Output> {
        self.inner.flush().map(Output::from)
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }

    fn as_sync_element(&mut self) -> Option<&mut dyn SyncElement> {
        if self.inner.execution_hints().is_rt_safe() {
            Some(&mut self.inner)
        } else {
            None
        }
    }
}

/// Wrapper to adapt a boxed [`Element`] trait object to [`AsyncElementDyn`].
///
/// This is used when the element type is not known at compile time,
/// such as when creating elements from a factory.
pub struct BoxedElementAdapter {
    inner: Box<dyn Element + Send>,
}

impl BoxedElementAdapter {
    /// Create a new boxed element adapter.
    pub fn new(element: Box<dyn Element + Send>) -> Self {
        Self { inner: element }
    }
}

impl SendAsyncElementDyn for BoxedElementAdapter {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Transform
    }

    fn set_bus(&mut self, bus: crate::pipeline::bus::BusHandle) {
        self.inner.set_bus(bus);
    }

    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.inner.set_output_budget(budget);
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        match input {
            Some(buffer) => self.inner.process(buffer),
            None => Ok(None),
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        AsyncElementDyn::process(self, input)
            .await
            .map(Output::from)
    }

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }

    fn output_caps(&self) -> Caps {
        self.inner.output_caps()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    async fn flush(&mut self) -> Result<Output> {
        self.inner.flush().map(Output::from)
    }

    fn as_any(&self) -> &dyn Any {
        // BoxedElementAdapter contains a trait object, not a concrete type.
        // We return self since the original type is not recoverable.
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Wrapper to adapt a sync [`Transform`] to [`AsyncElementDyn`].
///
/// This adapter supports transforms that produce multiple outputs.
pub struct TransformAdapter<T: Transform> {
    inner: T,
    pending: Vec<Buffer>,
}

impl<T: Transform> TransformAdapter<T> {
    /// Create a new transform adapter.
    pub fn new(transform: T) -> Self {
        Self {
            inner: transform,
            pending: Vec::new(),
        }
    }
}

impl<T: Transform + Send + 'static> SendAsyncElementDyn for TransformAdapter<T> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Transform
    }

    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.inner.set_output_budget(budget);
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        // First, return any pending buffers from previous multi-output
        if !self.pending.is_empty() {
            return Ok(Some(self.pending.remove(0)));
        }

        match input {
            Some(buffer) => {
                let output = self.inner.transform(buffer)?;
                match output {
                    Output::None => Ok(None),
                    Output::Single(b) => Ok(Some(b)),
                    Output::Multiple(mut v) => {
                        if v.is_empty() {
                            Ok(None)
                        } else {
                            let first = v.remove(0);
                            self.pending = v;
                            Ok(Some(first))
                        }
                    }
                }
            }
            None => Ok(None),
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        // Return pending first
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            return Ok(Output::from(pending));
        }

        match input {
            Some(buffer) => self.inner.transform(buffer),
            None => Ok(Output::None),
        }
    }

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }

    fn output_caps(&self) -> Caps {
        self.inner.output_caps()
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        self.inner.input_media_caps()
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        self.inner.output_media_caps()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    async fn flush(&mut self) -> Result<Output> {
        // Include any pending buffers plus flush output
        let mut all = std::mem::take(&mut self.pending);
        let flush_output = self.inner.flush()?;
        match flush_output {
            Output::None => {}
            Output::Single(b) => all.push(b),
            Output::Multiple(v) => all.extend(v),
        }
        Ok(Output::from(all))
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

// ============================================================================
// Adapters for Async Elements -> AsyncElementDyn
// ============================================================================

/// Wrapper to adapt an [`AsyncSource`] to [`AsyncElementDyn`].
///
/// This is the native async adapter - no blocking required.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::element::AsyncSourceAdapter;
/// use parallax::elements::AsyncVideoTestSrc;
///
/// let src = AsyncVideoTestSrc::new()
///     .with_pattern(VideoPattern::MovingBall)
///     .with_framerate(30, 1)
///     .live(true);
///
/// let adapter = AsyncSourceAdapter::new(src);
/// ```
pub struct AsyncSourceAdapter<S: AsyncSource> {
    inner: S,
    /// Arena for pre-allocated buffers (uses SharedArena for cross-process support).
    arena: Option<crate::memory::SharedArena>,
    /// Pipeline clock for timestamp calculation.
    clock: Option<Arc<dyn Clock>>,
    /// Base time for running time calculation.
    base_time: ClockTime,
    /// Bus handle for posting messages.
    bus: Option<crate::pipeline::bus::BusHandle>,
}

impl<S: AsyncSource> AsyncSourceAdapter<S> {
    /// Create a new async source adapter.
    pub fn new(source: S) -> Self {
        Self {
            inner: source,
            arena: None,
            clock: None,
            base_time: ClockTime::ZERO,
            bus: None,
        }
    }

    /// Create an async source adapter with an arena for buffer allocation.
    pub fn with_arena(source: S, arena: crate::memory::SharedArena) -> Self {
        Self {
            inner: source,
            arena: Some(arena),
            clock: None,
            base_time: ClockTime::ZERO,
            bus: None,
        }
    }

    /// Set the arena for buffer allocation.
    pub fn set_arena(&mut self, arena: crate::memory::SharedArena) {
        self.arena = Some(arena);
    }

    /// Get a reference to the inner source.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get a mutable reference to the inner source.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S: AsyncSource + Send + 'static> SendAsyncElementDyn for AsyncSourceAdapter<S> {
    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.inner.set_output_budget(budget);
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Source
    }

    async fn process(&mut self, _input: Option<Buffer>) -> Result<Option<Buffer>> {
        // Try to acquire a slot from the arena for the ProduceContext
        if let Some(arena) = &self.arena {
            if let Some(slot) = arena.acquire() {
                let mut ctx = ProduceContext::new(slot);
                match self.inner.produce(&mut ctx).await? {
                    ProduceResult::Produced(n) => Ok(Some(ctx.finalize(n))),
                    ProduceResult::Eos => Ok(None),
                    ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
                    ProduceResult::OwnDmaBuf(dmabuf) => Ok(Some(dmabuf.to_buffer(arena)?)),
                    ProduceResult::WouldBlock => Ok(None),
                }
            } else {
                // Arena exhausted, try without buffer
                let mut ctx = ProduceContext::without_buffer();
                match self.inner.produce(&mut ctx).await? {
                    ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
                    ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                        "arena exhausted, cannot convert DmaBuf to Buffer".into(),
                    )),
                    ProduceResult::Eos => Ok(None),
                    ProduceResult::WouldBlock => Ok(None),
                    ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                        "arena exhausted and source doesn't provide own buffer".into(),
                    )),
                }
            }
        } else {
            // No arena, source must provide its own buffer
            let mut ctx = ProduceContext::without_buffer();
            match self.inner.produce(&mut ctx).await? {
                ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
                ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                    "no arena configured, cannot convert DmaBuf to Buffer".into(),
                )),
                ProduceResult::Eos => Ok(None),
                ProduceResult::WouldBlock => Ok(None),
                ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                    "no arena configured and source doesn't provide own buffer".into(),
                )),
            }
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        AsyncElementDyn::process(self, input)
            .await
            .map(Output::from)
    }

    fn output_caps(&self) -> Caps {
        self.inner.output_caps()
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        self.inner.output_media_caps()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    async fn process_source(&mut self) -> Result<SourceResult> {
        // Helper to configure clock and bus on context
        let configure_clock = |ctx: &mut ProduceContext| {
            if let Some(ref clock) = self.clock {
                ctx.set_clock(clock.clone(), self.base_time);
            }
            if let Some(ref bus) = self.bus {
                ctx.set_bus(bus.clone());
            }
        };

        // This implementation properly distinguishes WouldBlock from Eos
        if let Some(arena) = &self.arena {
            if let Some(slot) = arena.acquire() {
                let mut ctx = ProduceContext::new(slot);
                configure_clock(&mut ctx);
                match self.inner.produce(&mut ctx).await? {
                    ProduceResult::Produced(n) => {
                        Ok(SourceResult::Buffer(Box::new(ctx.finalize(n))))
                    }
                    ProduceResult::Eos => Ok(SourceResult::Eos),
                    ProduceResult::OwnBuffer(buffer) => Ok(SourceResult::Buffer(Box::new(buffer))),
                    ProduceResult::OwnDmaBuf(dmabuf) => {
                        Ok(SourceResult::Buffer(Box::new(dmabuf.to_buffer(arena)?)))
                    }
                    ProduceResult::WouldBlock => Ok(SourceResult::WouldBlock),
                }
            } else {
                let mut ctx = ProduceContext::without_buffer();
                configure_clock(&mut ctx);
                match self.inner.produce(&mut ctx).await? {
                    ProduceResult::OwnBuffer(buffer) => Ok(SourceResult::Buffer(Box::new(buffer))),
                    ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                        "arena exhausted, cannot convert DmaBuf to Buffer".into(),
                    )),
                    ProduceResult::Eos => Ok(SourceResult::Eos),
                    ProduceResult::WouldBlock => Ok(SourceResult::WouldBlock),
                    ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                        "arena exhausted and source doesn't provide own buffer".into(),
                    )),
                }
            }
        } else {
            let mut ctx = ProduceContext::without_buffer();
            configure_clock(&mut ctx);
            match self.inner.produce(&mut ctx).await? {
                ProduceResult::OwnBuffer(buffer) => Ok(SourceResult::Buffer(Box::new(buffer))),
                ProduceResult::OwnDmaBuf(_) => Err(crate::error::Error::BufferPool(
                    "no arena configured, cannot convert DmaBuf to Buffer".into(),
                )),
                ProduceResult::Eos => Ok(SourceResult::Eos),
                ProduceResult::WouldBlock => Ok(SourceResult::WouldBlock),
                ProduceResult::Produced(_) => Err(crate::error::Error::BufferPool(
                    "no arena configured and source doesn't provide own buffer".into(),
                )),
            }
        }
    }

    fn set_clock(&mut self, clock: Arc<dyn Clock>, base_time: ClockTime) {
        self.clock = Some(clock);
        self.base_time = base_time;
    }

    fn set_bus(&mut self, bus: crate::pipeline::bus::BusHandle) {
        self.bus = Some(bus);
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        self.inner.handle_upstream_event(event)
    }

    fn is_seekable(&self) -> bool {
        self.inner.is_seekable()
    }

    fn source_query_position(&self) -> Option<crate::pipeline::seek::PositionQuery> {
        self.inner.query_position()
    }

    fn source_query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        self.inner.query_duration()
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

/// Wrapper to adapt an [`AsyncSink`] to [`AsyncElementDyn`].
pub struct AsyncSinkAdapter<S: AsyncSink> {
    inner: S,
    /// The pipeline clock, once the executor hands it over at start.
    clock: Option<(Arc<dyn Clock>, ClockTime)>,
}

impl<S: AsyncSink> AsyncSinkAdapter<S> {
    /// Create a new async sink adapter.
    pub fn new(sink: S) -> Self {
        Self {
            inner: sink,
            clock: None,
        }
    }

    /// Get a reference to the inner sink.
    pub fn inner(&self) -> &S {
        &self.inner
    }

    /// Get a mutable reference to the inner sink.
    pub fn inner_mut(&mut self) -> &mut S {
        &mut self.inner
    }
}

impl<S: AsyncSink + Send + 'static> SendAsyncElementDyn for AsyncSinkAdapter<S> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Sink
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        if let Some(ref buffer) = input {
            let mut ctx = ConsumeContext::new(buffer);
            if let Some((clock, base_time)) = &self.clock {
                ctx.set_clock(clock, *base_time);
            }
            self.inner.consume(&ctx).await?;
        }
        Ok(None)
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        AsyncElementDyn::process(self, input)
            .await
            .map(Output::from)
    }

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        self.inner.input_media_caps()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    fn set_clock(&mut self, clock: Arc<dyn Clock>, base_time: ClockTime) {
        self.clock = Some((clock, base_time));
    }

    // Forwarded here for the same reason `SinkAdapter` forwards it: without
    // this an async sink never learns the stream ended, so it cannot drain or
    // close its device on EOS.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        self.inner.handle_downstream_event(event)
    }

    fn as_clock_provider(&self) -> Option<&dyn ClockProvider> {
        self.inner.as_clock_provider()
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

/// Wrapper to adapt an [`AsyncTransform`] to [`AsyncElementDyn`].
pub struct AsyncTransformAdapter<T: AsyncTransform> {
    inner: T,
    pending: Vec<Buffer>,
}

impl<T: AsyncTransform> AsyncTransformAdapter<T> {
    /// Create a new async transform adapter.
    pub fn new(transform: T) -> Self {
        Self {
            inner: transform,
            pending: Vec::new(),
        }
    }
}

impl<T: AsyncTransform + Send + 'static> SendAsyncElementDyn for AsyncTransformAdapter<T> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Transform
    }

    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.inner.set_output_budget(budget);
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        // First, return any pending buffers from previous multi-output
        if !self.pending.is_empty() {
            return Ok(Some(self.pending.remove(0)));
        }

        match input {
            Some(buffer) => {
                let output = self.inner.transform(buffer).await?;
                match output {
                    Output::None => Ok(None),
                    Output::Single(b) => Ok(Some(b)),
                    Output::Multiple(mut v) => {
                        if v.is_empty() {
                            Ok(None)
                        } else {
                            let first = v.remove(0);
                            self.pending = v;
                            Ok(Some(first))
                        }
                    }
                }
            }
            None => Ok(None),
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        // Return pending first
        if !self.pending.is_empty() {
            let pending = std::mem::take(&mut self.pending);
            return Ok(Output::from(pending));
        }

        match input {
            Some(buffer) => self.inner.transform(buffer).await,
            None => Ok(Output::None),
        }
    }

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }

    fn output_caps(&self) -> Caps {
        self.inner.output_caps()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    async fn flush(&mut self) -> Result<Output> {
        // Include any pending buffers plus flush output
        let mut all = std::mem::take(&mut self.pending);
        let flush_output = self.inner.flush().await?;
        match flush_output {
            Output::None => {}
            Output::Single(b) => all.push(b),
            Output::Multiple(v) => all.extend(v),
        }
        Ok(Output::from(all))
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

// ============================================================================
// Adapters for Demuxer/Muxer -> AsyncElementDyn
// ============================================================================

/// Wrapper to adapt a [`Demuxer`] to [`AsyncElementDyn`].
///
/// Demuxers process a single input stream and route buffers to multiple output pads.
/// The adapter stores routed output and returns buffers one at a time via `process()`.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::element::DemuxerAdapter;
///
/// struct MyDemuxer { /* ... */ }
/// impl Demuxer for MyDemuxer { /* ... */ }
///
/// let demuxer = MyDemuxer::new();
/// let adapter = DemuxerAdapter::new(demuxer);
/// ```
pub struct DemuxerAdapter<D: Demuxer> {
    inner: D,
    /// Pending routed outputs to return.
    pending: Vec<(PadId, Buffer)>,
}

impl<D: Demuxer> DemuxerAdapter<D> {
    /// Create a new demuxer adapter.
    pub fn new(demuxer: D) -> Self {
        Self {
            inner: demuxer,
            pending: Vec::new(),
        }
    }

    /// Get a reference to the inner demuxer.
    pub fn inner(&self) -> &D {
        &self.inner
    }

    /// Get a mutable reference to the inner demuxer.
    pub fn inner_mut(&mut self) -> &mut D {
        &mut self.inner
    }
}

impl<D: Demuxer + Send + 'static> SendAsyncElementDyn for DemuxerAdapter<D> {
    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.inner.set_output_budget(budget);
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Demuxer
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        // First return any pending buffers
        if !self.pending.is_empty() {
            let (_, buffer) = self.pending.remove(0);
            return Ok(Some(buffer));
        }

        match input {
            Some(buffer) => {
                let routed = self.inner.demux(buffer)?;
                let mut iter = routed.into_iter();

                // Return the first buffer, store the rest
                if let Some((_, first_buffer)) = iter.next() {
                    self.pending.extend(iter);
                    Ok(Some(first_buffer))
                } else {
                    Ok(None)
                }
            }
            None => Ok(None),
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        // Return pending first
        if !self.pending.is_empty() {
            let pending: Vec<Buffer> = std::mem::take(&mut self.pending)
                .into_iter()
                .map(|(_, b)| b)
                .collect();
            return Ok(Output::from(pending));
        }

        match input {
            Some(buffer) => {
                let routed = self.inner.demux(buffer)?;
                let buffers: Vec<Buffer> = routed.into_iter().map(|(_, b)| b).collect();
                Ok(Output::from(buffers))
            }
            None => Ok(Output::None),
        }
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        self.inner.handle_upstream_event(event)
    }

    // Forwarded like SourceAdapter does: a source-style demuxer IS the
    // pipeline's source, and without these a demuxer-rooted pipeline
    // reported `seekable() == false` while seeking perfectly well.
    fn is_seekable(&self) -> bool {
        self.inner.is_seekable()
    }

    fn source_query_position(&self) -> Option<crate::pipeline::seek::PositionQuery> {
        self.inner.query_position()
    }

    fn source_query_duration(&self) -> Option<crate::pipeline::seek::DurationQuery> {
        self.inner.query_duration()
    }

    async fn process_demux(&mut self, input: Option<Buffer>) -> Result<DemuxResult> {
        let routed = match input {
            Some(buffer) => self.inner.demux(buffer)?,
            None => match self.inner.produce()? {
                DemuxerProduce::Routed(routed) => routed,
                DemuxerProduce::WouldBlock => return Ok(DemuxResult::WouldBlock),
                DemuxerProduce::Eos => return Ok(DemuxResult::Eos),
            },
        };
        Ok(DemuxResult::Routed(
            routed
                .into_iter()
                .map(|(pad, buffer)| (self.inner.pad_name(pad), buffer))
                .collect(),
        ))
    }

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }

    fn output_caps(&self) -> Caps {
        // For demuxers, output caps is the union of all output pad caps
        // For now, return Any - proper per-pad caps are accessed via outputs()
        Caps::any()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

/// Wrapper to adapt a [`Muxer`] to [`AsyncElementDyn`].
///
/// Muxers accept buffers from multiple input pads and combine them into a single output.
/// The adapter tracks which pad buffers come from based on buffer metadata.
///
/// # Note
///
/// Since the standard `process()` interface doesn't carry pad information,
/// muxers used through this adapter will receive all inputs on PadId(0).
/// For proper multi-pad muxing, use the executor's native muxer support.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::element::MuxerAdapter;
///
/// struct MyMuxer { /* ... */ }
/// impl Muxer for MyMuxer { /* ... */ }
///
/// let muxer = MyMuxer::new();
/// let adapter = MuxerAdapter::new(muxer);
/// ```
pub struct MuxerAdapter<M: Muxer> {
    inner: M,
    /// Counter to generate pad IDs from input order.
    input_counter: u32,
}

impl<M: Muxer> MuxerAdapter<M> {
    /// Create a new muxer adapter.
    pub fn new(muxer: M) -> Self {
        Self {
            inner: muxer,
            input_counter: 0,
        }
    }

    /// Get a reference to the inner muxer.
    pub fn inner(&self) -> &M {
        &self.inner
    }

    /// Get a mutable reference to the inner muxer.
    pub fn inner_mut(&mut self) -> &mut M {
        &mut self.inner
    }
}

impl<M: Muxer + Send + 'static> SendAsyncElementDyn for MuxerAdapter<M> {
    fn set_output_budget(&mut self, budget: crate::memory::OutputBudget) {
        self.inner.set_output_budget(budget);
    }

    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Muxer
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        match input {
            Some(buffer) => {
                // When used through the generic adapter, we don't have pad info
                // Use a simple counter as pad ID
                let pad = PadId(self.input_counter);
                self.input_counter = self.input_counter.wrapping_add(1);

                self.inner.mux(MuxerInput::new(pad, buffer))
            }
            None => {
                // EOS - flush the muxer
                self.inner.flush()
            }
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        AsyncElementDyn::process(self, input)
            .await
            .map(Output::from)
    }

    fn input_caps(&self) -> Caps {
        // For muxers, input caps is the union of all input pad caps
        // For now, return Any - proper per-pad caps are accessed via inputs()
        Caps::any()
    }

    fn output_caps(&self) -> Caps {
        self.inner.output_caps()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    fn as_any(&self) -> &dyn Any {
        &self.inner
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;

    struct TestSource {
        count: u64,
        max: u64,
    }

    impl Source for TestSource {
        fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
            if self.count >= self.max {
                return Ok(ProduceResult::Eos);
            }

            // Write sequence number to the provided buffer
            let output = ctx.output();
            let bytes = self.count.to_le_bytes();
            output[..8].copy_from_slice(&bytes);

            // Set metadata
            ctx.set_sequence(self.count);
            self.count += 1;

            Ok(ProduceResult::Produced(8))
        }
    }

    struct TestSink {
        received: Vec<u64>,
    }

    impl Sink for TestSink {
        fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
            self.received.push(ctx.sequence());
            Ok(())
        }
    }

    struct PassThrough;

    impl Element for PassThrough {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            Ok(Some(buffer))
        }
    }

    #[test]
    fn test_output_none() {
        let out = Output::none();
        assert!(out.is_none());
        assert!(out.is_empty());
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn test_output_single() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));

        let out = Output::single(buffer);
        assert!(out.is_single());
        assert!(!out.is_empty());
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_output_from_vec() {
        let out: Output = vec![].into();
        assert!(out.is_none());

        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));
        let out: Output = vec![buffer].into();
        assert!(out.is_single());

        let slot1 = arena.acquire().unwrap();
        let slot2 = arena.acquire().unwrap();
        let buf1 = Buffer::new(MemoryHandle::new(slot1), Metadata::from_sequence(1));
        let buf2 = Buffer::new(MemoryHandle::new(slot2), Metadata::from_sequence(2));
        let out: Output = vec![buf1, buf2].into();
        assert!(out.is_multiple());
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_output_iterator() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot1 = arena.acquire().unwrap();
        let slot2 = arena.acquire().unwrap();
        let buf1 = Buffer::new(MemoryHandle::new(slot1), Metadata::from_sequence(1));
        let buf2 = Buffer::new(MemoryHandle::new(slot2), Metadata::from_sequence(2));
        let out: Output = vec![buf1, buf2].into();

        let seqs: Vec<u64> = out.into_iter().map(|b| b.metadata().sequence).collect();
        assert_eq!(seqs, vec![1, 2]);
    }

    #[tokio::test]
    async fn test_source_adapter() {
        let source = TestSource { count: 0, max: 3 };
        let arena = SharedArena::new(1024, 8).unwrap();
        let mut adapter = SourceAdapter::with_arena(source, arena);

        assert_eq!(AsyncElementDyn::element_type(&adapter), ElementType::Source);

        // Should produce 3 buffers then None
        assert!(
            AsyncElementDyn::process(&mut adapter, None)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            AsyncElementDyn::process(&mut adapter, None)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            AsyncElementDyn::process(&mut adapter, None)
                .await
                .unwrap()
                .is_some()
        );
        assert!(
            AsyncElementDyn::process(&mut adapter, None)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn test_sink_adapter() {
        let sink = TestSink { received: vec![] };
        let mut adapter = SinkAdapter::new(sink);

        assert_eq!(AsyncElementDyn::element_type(&adapter), ElementType::Sink);

        // Create and consume some buffers
        let arena = SharedArena::new(1024, 8).unwrap();
        for i in 0..3 {
            let slot = arena.acquire().unwrap();
            let handle = MemoryHandle::new(slot);
            let buffer = Buffer::new(handle, Metadata::from_sequence(i));
            AsyncElementDyn::process(&mut adapter, Some(buffer))
                .await
                .unwrap();
        }
    }

    #[tokio::test]
    async fn test_element_adapter() {
        let element = PassThrough;
        let mut adapter = ElementAdapter::new(element);

        assert_eq!(
            AsyncElementDyn::element_type(&adapter),
            ElementType::Transform
        );

        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));

        let result = AsyncElementDyn::process(&mut adapter, Some(buffer))
            .await
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    #[test]
    fn test_element_implements_transform() {
        let mut element = PassThrough;

        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));

        // Use Transform trait on Element
        let output = Transform::transform(&mut element, buffer).unwrap();
        assert!(output.is_single());
    }

    #[test]
    fn test_pad_id() {
        let pad = PadId::new(42);
        assert_eq!(pad.0, 42);

        let pad: PadId = 123u32.into();
        let id: u32 = pad.into();
        assert_eq!(id, 123);
    }

    #[test]
    fn test_routed_output() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));

        let mut output = RoutedOutput::new();
        assert!(output.is_empty());

        output.push(PadId(0), buffer);
        assert_eq!(output.len(), 1);

        for (pad, buf) in output {
            assert_eq!(pad.0, 0);
            assert_eq!(buf.metadata().sequence, 42);
        }
    }

    #[test]
    fn test_muxer_input() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));

        let input = MuxerInput::new(PadId(3), buffer);
        assert_eq!(input.pad.0, 3);
        assert_eq!(input.buffer.metadata().sequence, 42);
    }

    #[test]
    fn test_element_type_variants() {
        // Test all ElementType variants
        assert_eq!(format!("{:?}", ElementType::Source), "Source");
        assert_eq!(format!("{:?}", ElementType::Sink), "Sink");
        assert_eq!(format!("{:?}", ElementType::Transform), "Transform");
        assert_eq!(format!("{:?}", ElementType::Demuxer), "Demuxer");
        assert_eq!(format!("{:?}", ElementType::Muxer), "Muxer");

        // Test equality
        assert_eq!(ElementType::Source, ElementType::Source);
        assert_ne!(ElementType::Source, ElementType::Sink);
        assert_ne!(ElementType::Demuxer, ElementType::Muxer);
    }

    // Test demuxer adapter
    struct TestDemuxer {
        outputs: Vec<(PadId, Caps)>,
    }

    impl Demuxer for TestDemuxer {
        fn demux(&mut self, buffer: Buffer) -> Result<RoutedOutput> {
            // Route all buffers to pad 0
            Ok(RoutedOutput::single(PadId(0), buffer))
        }

        fn outputs(&self) -> &[(PadId, Caps)] {
            &self.outputs
        }

        fn on_pad_added(&mut self, _callback: PadAddedCallback) {}
    }

    #[tokio::test]
    async fn test_demuxer_adapter() {
        let demuxer = TestDemuxer {
            outputs: vec![(PadId(0), Caps::any())],
        };
        let mut adapter = DemuxerAdapter::new(demuxer);

        assert_eq!(
            AsyncElementDyn::element_type(&adapter),
            ElementType::Demuxer
        );

        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));

        let result = AsyncElementDyn::process(&mut adapter, Some(buffer))
            .await
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }

    // Test muxer adapter
    struct TestMuxer {
        inputs: Vec<(PadId, Caps)>,
        buffer_count: u32,
    }

    impl Muxer for TestMuxer {
        fn mux(&mut self, input: MuxerInput) -> Result<Option<Buffer>> {
            self.buffer_count += 1;
            // Pass through the buffer
            Ok(Some(input.buffer))
        }

        fn inputs(&self) -> &[(PadId, Caps)] {
            &self.inputs
        }

        fn on_pad_added(&mut self, _callback: PadAddedCallback) {}
    }

    #[tokio::test]
    async fn test_muxer_adapter() {
        let muxer = TestMuxer {
            inputs: vec![(PadId(0), Caps::any()), (PadId(1), Caps::any())],
            buffer_count: 0,
        };
        let mut adapter = MuxerAdapter::new(muxer);

        assert_eq!(AsyncElementDyn::element_type(&adapter), ElementType::Muxer);

        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::new(slot);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));

        let result = AsyncElementDyn::process(&mut adapter, Some(buffer))
            .await
            .unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);

        // Verify the muxer received a buffer
        assert_eq!(adapter.inner().buffer_count, 1);
    }

    // ========================================================================
    // ExecutionHints and RT-safety tests
    // ========================================================================

    #[test]
    fn test_default_hints() {
        let hints = ExecutionHints::default();
        assert!(!hints.is_rt_safe());
        assert_eq!(hints.processing, ProcessingHint::Unknown);
        assert_eq!(hints.latency, LatencyHint::Normal);
    }

    #[test]
    fn test_rt_safe_profile() {
        let hints = ExecutionHints::rt_safe();
        assert!(hints.is_rt_safe());
        assert_eq!(hints.processing, ProcessingHint::CpuBound);
        assert_eq!(hints.latency, LatencyHint::Low);
    }

    #[test]
    fn test_io_bound_profile() {
        let hints = ExecutionHints::io_bound();
        assert!(!hints.is_rt_safe());
        assert_eq!(hints.processing, ProcessingHint::IoBound);
    }

    #[test]
    fn test_source_default_hints() {
        let source = TestSource { count: 0, max: 1 };
        let hints = source.execution_hints();
        assert!(!hints.is_rt_safe());
    }

    #[test]
    fn test_sink_default_hints() {
        let sink = TestSink { received: vec![] };
        let hints = sink.execution_hints();
        assert!(!hints.is_rt_safe());
    }

    #[test]
    fn test_element_default_hints() {
        let element = PassThrough;
        let hints = Element::execution_hints(&element);
        assert!(!hints.is_rt_safe());
    }

    // Test custom RT-safe element
    struct RtSafeProcessor;

    impl Element for RtSafeProcessor {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            // No allocations, just pass through
            Ok(Some(buffer))
        }

        fn execution_hints(&self) -> ExecutionHints {
            ExecutionHints::rt_safe()
        }
    }

    #[test]
    fn test_custom_rt_safe_element() {
        let element = RtSafeProcessor;
        let hints = Element::execution_hints(&element);
        assert!(hints.is_rt_safe());
        assert_eq!(hints.latency, LatencyHint::Low);
    }

    #[tokio::test]
    async fn test_adapter_forwards_hints() {
        // Test that adapters correctly forward execution_hints
        let element = RtSafeProcessor;
        let adapter = ElementAdapter::new(element);

        let hints = SendAsyncElementDyn::execution_hints(&adapter);
        assert!(hints.is_rt_safe());
    }

    // Test async source with I/O-bound hints
    struct TestAsyncSource;

    impl AsyncSource for TestAsyncSource {
        async fn produce(&mut self, _ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
            Ok(ProduceResult::Eos)
        }
    }

    #[tokio::test]
    async fn test_async_source_default_hints() {
        let source = TestAsyncSource;
        // Async sources default to io_bound hints
        let hints = source.execution_hints();
        assert_eq!(hints.processing, ProcessingHint::IoBound);
        assert!(!hints.is_rt_safe());
    }

    #[tokio::test]
    async fn test_async_source_adapter_hints() {
        let source = TestAsyncSource;
        let adapter = AsyncSourceAdapter::new(source);

        let hints = SendAsyncElementDyn::execution_hints(&adapter);
        assert_eq!(hints.processing, ProcessingHint::IoBound);
        assert!(!hints.is_rt_safe());
    }

    // Test async sink with I/O-bound hints
    struct TestAsyncSink;

    impl AsyncSink for TestAsyncSink {
        async fn consume(&mut self, _ctx: &ConsumeContext<'_>) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_async_sink_default_hints() {
        let sink = TestAsyncSink;
        let hints = sink.execution_hints();
        assert_eq!(hints.processing, ProcessingHint::IoBound);
        assert!(!hints.is_rt_safe());
    }

    #[test]
    fn test_demuxer_default_hints() {
        let demuxer = TestDemuxer {
            outputs: vec![(PadId(0), Caps::any())],
        };
        let hints = demuxer.execution_hints();
        assert!(!hints.is_rt_safe());
    }

    #[test]
    fn test_muxer_default_hints() {
        let muxer = TestMuxer {
            inputs: vec![(PadId(0), Caps::any())],
            buffer_count: 0,
        };
        let hints = muxer.execution_hints();
        assert!(!hints.is_rt_safe());
    }
}
