//! Core element traits.

use crate::buffer::Buffer;
use crate::element::context::{ConsumeContext, ProduceContext, ProduceResult};
use crate::error::Result;
use crate::format::Caps;
use dynosaur::dynosaur;
use smallvec::SmallVec;

// ============================================================================
// Scheduling Affinity
// ============================================================================

/// Scheduling affinity for elements in hybrid execution mode.
///
/// This determines whether an element runs in the Tokio async runtime
/// (suitable for I/O-bound operations) or in a dedicated real-time thread
/// (suitable for low-latency processing).
///
/// # Example
///
/// ```rust,ignore
/// impl Source for MyAudioSource {
///     fn produce(&mut self) -> Result<Option<Buffer>> {
///         // ... produce audio samples
///     }
///
///     fn affinity(&self) -> Affinity {
///         Affinity::RealTime  // Low-latency audio processing
///     }
///
///     fn is_rt_safe(&self) -> bool {
///         true  // No allocations, no blocking I/O
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Affinity {
    /// Runs in Tokio async runtime.
    ///
    /// Use this for elements that perform async I/O (network, file),
    /// or that may block. This is the default for async elements.
    Async,

    /// Runs in dedicated real-time thread(s).
    ///
    /// Use this for elements that need deterministic, low-latency execution.
    /// Elements with this affinity should be RT-safe (no allocations,
    /// no blocking syscalls in the hot path).
    RealTime,

    /// Let the executor decide based on element characteristics.
    ///
    /// The executor will check `is_rt_safe()` and other hints to
    /// determine the best scheduling strategy.
    #[default]
    Auto,
}

// ============================================================================
// Execution Hints
// ============================================================================

/// Hints about an element's execution characteristics.
///
/// These hints are used by the executor to automatically determine
/// the best execution strategy (async, RT, isolated) for each element.
///
/// # Example
///
/// ```rust,ignore
/// impl Source for H264Decoder {
///     fn produce(&mut self) -> Result<Option<Buffer>> { /* ... */ }
///
///     fn execution_hints(&self) -> ExecutionHints {
///         ExecutionHints {
///             // Decoders process untrusted input
///             trust_level: TrustLevel::Untrusted,
///             // CPU-intensive processing
///             processing: ProcessingHint::CpuBound,
///             // Low latency needed for video
///             latency: LatencyHint::Low,
///             // Might crash on malformed input
///             crash_safe: false,
///             ..Default::default()
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionHints {
    /// Trust level of the data being processed.
    pub trust_level: TrustLevel,
    /// Processing characteristics (CPU vs I/O bound).
    pub processing: ProcessingHint,
    /// Latency requirements.
    pub latency: LatencyHint,
    /// Whether the element might crash on bad input.
    /// If true, isolation is recommended.
    pub crash_safe: bool,
    /// Whether the element uses native code (FFI).
    /// Native code is harder to sandbox.
    pub uses_native_code: bool,
    /// Memory usage hint (helps with scheduling decisions).
    pub memory: MemoryHint,
}

impl Default for ExecutionHints {
    fn default() -> Self {
        Self {
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
    /// Create hints for a trusted, lightweight element.
    pub fn trusted() -> Self {
        Self::default()
    }

    /// Create hints for an untrusted input handler (e.g., decoder).
    pub fn untrusted() -> Self {
        Self {
            trust_level: TrustLevel::Untrusted,
            crash_safe: false,
            ..Default::default()
        }
    }

    /// Create hints for a CPU-intensive element.
    pub fn cpu_intensive() -> Self {
        Self {
            processing: ProcessingHint::CpuBound,
            ..Default::default()
        }
    }

    /// Create hints for an I/O-bound element.
    pub fn io_bound() -> Self {
        Self {
            processing: ProcessingHint::IoBound,
            ..Default::default()
        }
    }

    /// Create hints for a low-latency element.
    pub fn low_latency() -> Self {
        Self {
            latency: LatencyHint::Low,
            ..Default::default()
        }
    }

    /// Create hints for an element using native/FFI code.
    pub fn native() -> Self {
        Self {
            uses_native_code: true,
            crash_safe: false,
            ..Default::default()
        }
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

    /// Get the preferred buffer size for this source.
    ///
    /// The executor uses this hint when allocating pool buffers.
    /// Return `None` to use the default pool buffer size.
    fn preferred_buffer_size(&self) -> Option<usize> {
        None
    }

    /// Get the scheduling affinity for this source.
    ///
    /// Override this to specify whether this source should run in the
    /// async runtime or real-time thread.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if this source is safe to run in a real-time context.
    ///
    /// An RT-safe source must:
    /// - Not allocate memory in `produce()` (use pre-allocated buffers)
    /// - Not perform blocking I/O
    /// - Not take locks that could be held by non-RT threads
    /// - Complete in bounded, deterministic time
    ///
    /// Returns `false` by default (conservative).
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints for automatic scheduling decisions.
    ///
    /// Override this to provide hints about processing characteristics,
    /// trust level, latency requirements, etc. The executor uses these
    /// hints to automatically determine the best execution strategy.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
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

    /// Get the preferred buffer size for this source.
    fn preferred_buffer_size(&self) -> Option<usize> {
        None
    }

    /// Get the scheduling affinity for this source.
    ///
    /// Async sources default to `Async` affinity since they typically
    /// perform I/O operations that require the async runtime.
    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    /// Check if this source is safe to run in a real-time context.
    ///
    /// Async sources are generally not RT-safe since they use async I/O.
    /// Returns `false` by default.
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        // Async sources are typically I/O-bound
        ExecutionHints::io_bound()
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

    /// Get the scheduling affinity for this sink.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if this sink is safe to run in a real-time context.
    ///
    /// An RT-safe sink must:
    /// - Not allocate memory in `consume()` (use pre-allocated buffers)
    /// - Not perform blocking I/O (or use mmap/RT-safe I/O)
    /// - Not take locks that could be held by non-RT threads
    /// - Complete in bounded, deterministic time
    ///
    /// Returns `false` by default (conservative).
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
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

    /// Get the scheduling affinity for this sink.
    ///
    /// Async sinks default to `Async` affinity since they typically
    /// perform I/O operations that require the async runtime.
    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    /// Check if this sink is safe to run in a real-time context.
    ///
    /// Async sinks are generally not RT-safe since they use async I/O.
    /// Returns `false` by default.
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        // Async sinks are typically I/O-bound
        ExecutionHints::io_bound()
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

    /// Get the scheduling affinity for this element.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if this element is safe to run in a real-time context.
    ///
    /// An RT-safe element must:
    /// - Not allocate memory in `process()` (use pre-allocated buffers)
    /// - Not perform blocking I/O
    /// - Not take locks that could be held by non-RT threads
    /// - Complete in bounded, deterministic time
    ///
    /// Returns `false` by default (conservative).
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
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

    /// Get the scheduling affinity for this transform.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if this transform is safe to run in a real-time context.
    ///
    /// An RT-safe transform must:
    /// - Not allocate memory in `transform()` (use pre-allocated buffers)
    /// - Not perform blocking I/O
    /// - Not take locks that could be held by non-RT threads
    /// - Complete in bounded, deterministic time
    ///
    /// Returns `false` by default (conservative).
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
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

    /// Get the scheduling affinity for this transform.
    ///
    /// Async transforms default to `Async` affinity since they typically
    /// perform I/O operations that require the async runtime.
    fn affinity(&self) -> Affinity {
        Affinity::Async
    }

    /// Check if this transform is safe to run in a real-time context.
    ///
    /// Async transforms are generally not RT-safe since they use async I/O.
    /// Returns `false` by default.
    fn is_rt_safe(&self) -> bool {
        false
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

    /// Get the scheduling affinity for this demuxer.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if this demuxer is safe to run in a real-time context.
    ///
    /// An RT-safe demuxer must:
    /// - Not allocate memory in `demux()` (use pre-allocated buffers)
    /// - Not perform blocking I/O
    /// - Not take locks that could be held by non-RT threads
    /// - Complete in bounded, deterministic time
    ///
    /// Returns `false` by default (conservative).
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
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

    /// Get the scheduling affinity for this muxer.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if this muxer is safe to run in a real-time context.
    ///
    /// An RT-safe muxer must:
    /// - Not allocate memory in `mux()` (use pre-allocated buffers)
    /// - Not perform blocking I/O
    /// - Not take locks that could be held by non-RT threads
    /// - Complete in bounded, deterministic time
    ///
    /// Returns `false` by default (conservative).
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
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

    /// Get the scheduling affinity for this element.
    ///
    /// This determines whether the element runs in the async runtime
    /// or in a dedicated real-time thread.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if this element is safe to run in a real-time context.
    ///
    /// An RT-safe element must not allocate, block, or take locks
    /// in the hot path.
    fn is_rt_safe(&self) -> bool {
        false
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
    /// Arena for pre-allocated buffers.
    arena: Option<std::sync::Arc<crate::memory::CpuArena>>,
    /// Buffer pool for high-level pool API with backpressure.
    pool: Option<std::sync::Arc<dyn crate::memory::BufferPool>>,
}

impl<S: Source> SourceAdapter<S> {
    /// Create a new source adapter.
    pub fn new(source: S) -> Self {
        Self {
            inner: source,
            arena: None,
            pool: None,
        }
    }

    /// Create a source adapter with an arena for buffer allocation.
    pub fn with_arena(source: S, arena: std::sync::Arc<crate::memory::CpuArena>) -> Self {
        Self {
            inner: source,
            arena: Some(arena),
            pool: None,
        }
    }

    /// Create a source adapter with a buffer pool.
    pub fn with_pool(source: S, pool: std::sync::Arc<dyn crate::memory::BufferPool>) -> Self {
        Self {
            inner: source,
            arena: None,
            pool: Some(pool),
        }
    }

    /// Set the arena for buffer allocation.
    pub fn set_arena(&mut self, arena: std::sync::Arc<crate::memory::CpuArena>) {
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
                        ProduceResult::WouldBlock => Ok(None),
                    }
                } else {
                    // Arena exhausted - provide pool-only access
                    let mut ctx = ProduceContext::with_pool_only(pool.as_ref());
                    match self.inner.produce(&mut ctx)? {
                        ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
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
                    ProduceResult::WouldBlock => Ok(None),
                }
            } else {
                // Arena exhausted
                let mut ctx = ProduceContext::without_buffer();
                match self.inner.produce(&mut ctx)? {
                    ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
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

    fn affinity(&self) -> Affinity {
        self.inner.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.inner.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }
}

/// Wrapper to adapt a sync [`Sink`] to [`AsyncElementDyn`].
pub struct SinkAdapter<S: Sink> {
    inner: S,
}

impl<S: Sink> SinkAdapter<S> {
    /// Create a new sink adapter.
    pub fn new(sink: S) -> Self {
        Self { inner: sink }
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
            let ctx = ConsumeContext::new(buffer);
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

    fn affinity(&self) -> Affinity {
        self.inner.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.inner.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
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

    fn affinity(&self) -> Affinity {
        self.inner.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.inner.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }

    async fn flush(&mut self) -> Result<Output> {
        self.inner.flush().map(Output::from)
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

    fn affinity(&self) -> Affinity {
        self.inner.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.inner.is_rt_safe()
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
    /// Arena for pre-allocated buffers.
    arena: Option<std::sync::Arc<crate::memory::CpuArena>>,
}

impl<S: AsyncSource> AsyncSourceAdapter<S> {
    /// Create a new async source adapter.
    pub fn new(source: S) -> Self {
        Self {
            inner: source,
            arena: None,
        }
    }

    /// Create an async source adapter with an arena for buffer allocation.
    pub fn with_arena(source: S, arena: std::sync::Arc<crate::memory::CpuArena>) -> Self {
        Self {
            inner: source,
            arena: Some(arena),
        }
    }

    /// Set the arena for buffer allocation.
    pub fn set_arena(&mut self, arena: std::sync::Arc<crate::memory::CpuArena>) {
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
                    ProduceResult::WouldBlock => Ok(None),
                }
            } else {
                // Arena exhausted, try without buffer
                let mut ctx = ProduceContext::without_buffer();
                match self.inner.produce(&mut ctx).await? {
                    ProduceResult::OwnBuffer(buffer) => Ok(Some(buffer)),
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

    fn affinity(&self) -> Affinity {
        self.inner.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.inner.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }
}

/// Wrapper to adapt an [`AsyncSink`] to [`AsyncElementDyn`].
pub struct AsyncSinkAdapter<S: AsyncSink> {
    inner: S,
}

impl<S: AsyncSink> AsyncSinkAdapter<S> {
    /// Create a new async sink adapter.
    pub fn new(sink: S) -> Self {
        Self { inner: sink }
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
            let ctx = ConsumeContext::new(buffer);
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

    fn affinity(&self) -> Affinity {
        self.inner.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.inner.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
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

    fn affinity(&self) -> Affinity {
        self.inner.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.inner.is_rt_safe()
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

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }

    fn output_caps(&self) -> Caps {
        // For demuxers, output caps is the union of all output pad caps
        // For now, return Any - proper per-pad caps are accessed via outputs()
        Caps::any()
    }

    fn affinity(&self) -> Affinity {
        self.inner.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.inner.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
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

    fn affinity(&self) -> Affinity {
        self.inner.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.inner.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.inner.execution_hints()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::{CpuArena, HeapSegment};
    use crate::metadata::Metadata;
    use std::sync::Arc;

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
        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
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

        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));
        let out: Output = vec![buffer].into();
        assert!(out.is_single());

        let segment1 = Arc::new(HeapSegment::new(8).unwrap());
        let segment2 = Arc::new(HeapSegment::new(8).unwrap());
        let buf1 = Buffer::new(
            MemoryHandle::from_segment(segment1),
            Metadata::from_sequence(1),
        );
        let buf2 = Buffer::new(
            MemoryHandle::from_segment(segment2),
            Metadata::from_sequence(2),
        );
        let out: Output = vec![buf1, buf2].into();
        assert!(out.is_multiple());
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_output_iterator() {
        let segment1 = Arc::new(HeapSegment::new(8).unwrap());
        let segment2 = Arc::new(HeapSegment::new(8).unwrap());
        let buf1 = Buffer::new(
            MemoryHandle::from_segment(segment1),
            Metadata::from_sequence(1),
        );
        let buf2 = Buffer::new(
            MemoryHandle::from_segment(segment2),
            Metadata::from_sequence(2),
        );
        let out: Output = vec![buf1, buf2].into();

        let seqs: Vec<u64> = out.into_iter().map(|b| b.metadata().sequence).collect();
        assert_eq!(seqs, vec![1, 2]);
    }

    #[tokio::test]
    async fn test_source_adapter() {
        let source = TestSource { count: 0, max: 3 };
        let arena = CpuArena::new(1024, 8).unwrap();
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
        for i in 0..3 {
            let segment = Arc::new(HeapSegment::new(8).unwrap());
            let handle = MemoryHandle::from_segment(segment);
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

        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
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

        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
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
        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
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
        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
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

        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
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

        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
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
    // Affinity and RT-safety tests
    // ========================================================================

    #[test]
    fn test_affinity_default() {
        // Default affinity should be Auto
        assert_eq!(Affinity::default(), Affinity::Auto);
    }

    #[test]
    fn test_affinity_variants() {
        // Test all variants exist and are distinct
        assert_ne!(Affinity::Async, Affinity::RealTime);
        assert_ne!(Affinity::Async, Affinity::Auto);
        assert_ne!(Affinity::RealTime, Affinity::Auto);

        // Test debug formatting
        assert_eq!(format!("{:?}", Affinity::Async), "Async");
        assert_eq!(format!("{:?}", Affinity::RealTime), "RealTime");
        assert_eq!(format!("{:?}", Affinity::Auto), "Auto");
    }

    #[test]
    fn test_source_default_affinity() {
        let source = TestSource { count: 0, max: 1 };
        // Default affinity should be Auto
        assert_eq!(source.affinity(), Affinity::Auto);
        // Default is_rt_safe should be false (conservative)
        assert!(!source.is_rt_safe());
    }

    #[test]
    fn test_sink_default_affinity() {
        let sink = TestSink { received: vec![] };
        assert_eq!(sink.affinity(), Affinity::Auto);
        assert!(!sink.is_rt_safe());
    }

    #[test]
    fn test_element_default_affinity() {
        let element = PassThrough;
        // Use Element trait explicitly to avoid ambiguity with Transform blanket impl
        assert_eq!(Element::affinity(&element), Affinity::Auto);
        assert!(!Element::is_rt_safe(&element));
    }

    // Test custom RT-safe element
    struct RtSafeProcessor;

    impl Element for RtSafeProcessor {
        fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
            // No allocations, just pass through
            Ok(Some(buffer))
        }

        fn affinity(&self) -> Affinity {
            Affinity::RealTime
        }

        fn is_rt_safe(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_custom_rt_safe_element() {
        let element = RtSafeProcessor;
        assert_eq!(Element::affinity(&element), Affinity::RealTime);
        assert!(Element::is_rt_safe(&element));
    }

    #[tokio::test]
    async fn test_adapter_forwards_affinity() {
        // Test that adapters correctly forward affinity and is_rt_safe
        let element = RtSafeProcessor;
        let adapter = ElementAdapter::new(element);

        assert_eq!(SendAsyncElementDyn::affinity(&adapter), Affinity::RealTime);
        assert!(SendAsyncElementDyn::is_rt_safe(&adapter));
    }

    // Test async source with Async affinity
    struct TestAsyncSource;

    impl AsyncSource for TestAsyncSource {
        async fn produce(&mut self, _ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
            Ok(ProduceResult::Eos)
        }
    }

    #[tokio::test]
    async fn test_async_source_default_affinity() {
        let source = TestAsyncSource;
        // Async sources default to Async affinity
        assert_eq!(AsyncSource::affinity(&source), Affinity::Async);
        assert!(!AsyncSource::is_rt_safe(&source));
    }

    #[tokio::test]
    async fn test_async_source_adapter_affinity() {
        let source = TestAsyncSource;
        let adapter = AsyncSourceAdapter::new(source);

        assert_eq!(SendAsyncElementDyn::affinity(&adapter), Affinity::Async);
        assert!(!SendAsyncElementDyn::is_rt_safe(&adapter));
    }

    // Test async sink with Async affinity
    struct TestAsyncSink;

    impl AsyncSink for TestAsyncSink {
        async fn consume(&mut self, _ctx: &ConsumeContext<'_>) -> Result<()> {
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_async_sink_default_affinity() {
        let sink = TestAsyncSink;
        assert_eq!(AsyncSink::affinity(&sink), Affinity::Async);
        assert!(!AsyncSink::is_rt_safe(&sink));
    }

    #[test]
    fn test_demuxer_default_affinity() {
        let demuxer = TestDemuxer {
            outputs: vec![(PadId(0), Caps::any())],
        };
        assert_eq!(demuxer.affinity(), Affinity::Auto);
        assert!(!demuxer.is_rt_safe());
    }

    #[test]
    fn test_muxer_default_affinity() {
        let muxer = TestMuxer {
            inputs: vec![(PadId(0), Caps::any())],
            buffer_count: 0,
        };
        assert_eq!(muxer.affinity(), Affinity::Auto);
        assert!(!muxer.is_rt_safe());
    }
}
