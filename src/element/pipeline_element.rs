//! Unified PipelineElement trait for all pipeline elements.
//!
//! This module provides a single trait that all pipeline elements implement,
//! with simple wrapper types that eliminate most boilerplate.
//!
//! # Architecture
//!
//! ```text
//!                     ┌─────────────────────────────────────────┐
//!                     │           PipelineElement               │
//!                     │      (single async trait object)        │
//!                     └─────────────────────────────────────────┘
//!                                         ▲
//!                     ┌───────────────────┼───────────────────┐
//!                     │                   │                   │
//!             ┌───────┴───────┐   ┌───────┴───────┐   ┌───────┴───────┐
//!             │ Src<T>        │   │ Snk<T>        │   │ Xfm<T>        │
//!             │ (wraps Source)│   │ (wraps Sink)  │   │(wraps Xform)  │
//!             └───────────────┘   └───────────────┘   └───────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::element::{SimpleSource, SimpleSink, ProcessOutput};
//! use parallax::element::{Src, Snk, Xfm};
//!
//! // Define a simple source
//! struct MySource { count: u32 }
//!
//! impl SimpleSource for MySource {
//!     fn produce(&mut self) -> Result<ProcessOutput> {
//!         if self.count < 10 {
//!             self.count += 1;
//!             Ok(ProcessOutput::buffer(create_buffer()))
//!         } else {
//!             Ok(ProcessOutput::Eos)
//!         }
//!     }
//! }
//!
//! // Wrap and add to pipeline
//! let mut pipeline = Pipeline::new();
//! pipeline.add_element("src", Box::new(Src(MySource { count: 0 })));
//! ```

use crate::buffer::Buffer;
use crate::error::Result;
use crate::event::{Event, EventResult};
use crate::format::Caps;

use super::traits::{Affinity, ElementType, ExecutionHints};

// ============================================================================
// ProcessOutput - Unified output type
// ============================================================================

/// Result of element processing.
///
/// This unified output type represents all possible results from processing:
/// - No output (filtered, buffering, etc.)
/// - One output buffer
/// - Multiple output buffers
/// - End of stream
/// - Would block (for async polling)
///
/// # Example
///
/// ```rust
/// use parallax::element::ProcessOutput;
///
/// // No output (filter dropped the buffer)
/// let out = ProcessOutput::None;
///
/// // Single output (most common)
/// // let out = ProcessOutput::buffer(my_buffer);
///
/// // Multiple outputs (e.g., demuxer)
/// // let out = ProcessOutput::multiple(vec![buf1, buf2]);
///
/// // End of stream
/// let out = ProcessOutput::Eos;
/// ```
#[derive(Debug, Default)]
pub enum ProcessOutput {
    /// No output (buffer was filtered/consumed/buffering).
    #[default]
    None,
    /// Single output buffer.
    Buffer(Buffer),
    /// Multiple output buffers.
    Buffers(Vec<Buffer>),
    /// End of stream reached.
    Eos,
    /// Would block (for async polling, no data available yet).
    Pending,
}

impl ProcessOutput {
    /// Create a single buffer output.
    #[inline]
    pub fn buffer(buf: Buffer) -> Self {
        Self::Buffer(buf)
    }

    /// Create a multiple buffer output.
    #[inline]
    pub fn multiple(bufs: Vec<Buffer>) -> Self {
        match bufs.len() {
            0 => Self::None,
            1 => Self::Buffer(bufs.into_iter().next().unwrap()),
            _ => Self::Buffers(bufs),
        }
    }

    /// Create an empty output.
    #[inline]
    pub fn none() -> Self {
        Self::None
    }

    /// Create an end-of-stream output.
    #[inline]
    pub fn eos() -> Self {
        Self::Eos
    }

    /// Create a pending output.
    #[inline]
    pub fn pending() -> Self {
        Self::Pending
    }

    /// Check if this is end-of-stream.
    #[inline]
    pub fn is_eos(&self) -> bool {
        matches!(self, Self::Eos)
    }

    /// Check if this is pending (would block).
    #[inline]
    pub fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }

    /// Check if there is no output.
    #[inline]
    pub fn is_none(&self) -> bool {
        matches!(self, Self::None)
    }

    /// Check if there is exactly one output buffer.
    #[inline]
    pub fn is_single(&self) -> bool {
        matches!(self, Self::Buffer(_))
    }

    /// Check if there are multiple output buffers.
    #[inline]
    pub fn is_multiple(&self) -> bool {
        matches!(self, Self::Buffers(_))
    }

    /// Check if there is any output (buffer or buffers).
    #[inline]
    pub fn has_output(&self) -> bool {
        matches!(self, Self::Buffer(_) | Self::Buffers(_))
    }

    /// Get the number of output buffers.
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::None | Self::Eos | Self::Pending => 0,
            Self::Buffer(_) => 1,
            Self::Buffers(v) => v.len(),
        }
    }

    /// Check if empty (no buffers).
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to a Vec of buffers.
    pub fn into_vec(self) -> Vec<Buffer> {
        match self {
            Self::None | Self::Eos | Self::Pending => vec![],
            Self::Buffer(b) => vec![b],
            Self::Buffers(v) => v,
        }
    }

    /// Get a reference to the single buffer, if any.
    pub fn as_buffer(&self) -> Option<&Buffer> {
        match self {
            Self::Buffer(b) => Some(b),
            _ => None,
        }
    }

    /// Take the single buffer, returning None for other variants.
    pub fn into_buffer(self) -> Option<Buffer> {
        match self {
            Self::Buffer(b) => Some(b),
            _ => None,
        }
    }

    /// Take the first buffer if available.
    pub fn take_first(&mut self) -> Option<Buffer> {
        match self {
            Self::Buffer(_) => {
                let old = std::mem::take(self);
                match old {
                    Self::Buffer(b) => Some(b),
                    _ => None,
                }
            }
            Self::Buffers(v) if !v.is_empty() => Some(v.remove(0)),
            _ => None,
        }
    }
}

// Ergonomic conversions
impl From<Buffer> for ProcessOutput {
    #[inline]
    fn from(b: Buffer) -> Self {
        Self::Buffer(b)
    }
}

impl From<Option<Buffer>> for ProcessOutput {
    fn from(opt: Option<Buffer>) -> Self {
        match opt {
            Some(b) => Self::Buffer(b),
            None => Self::None,
        }
    }
}

impl From<Vec<Buffer>> for ProcessOutput {
    fn from(v: Vec<Buffer>) -> Self {
        Self::multiple(v)
    }
}

impl FromIterator<Buffer> for ProcessOutput {
    fn from_iter<I: IntoIterator<Item = Buffer>>(iter: I) -> Self {
        Self::multiple(iter.into_iter().collect())
    }
}

// Convert from the existing Output type
impl From<super::traits::Output> for ProcessOutput {
    fn from(output: super::traits::Output) -> Self {
        match output {
            super::traits::Output::None => Self::None,
            super::traits::Output::Single(b) => Self::Buffer(b),
            super::traits::Output::Multiple(v) => Self::multiple(v),
        }
    }
}

// Convert to the existing Output type (for compatibility)
impl From<ProcessOutput> for super::traits::Output {
    fn from(output: ProcessOutput) -> Self {
        match output {
            ProcessOutput::None | ProcessOutput::Eos | ProcessOutput::Pending => {
                super::traits::Output::None
            }
            ProcessOutput::Buffer(b) => super::traits::Output::Single(b),
            ProcessOutput::Buffers(v) => super::traits::Output::Multiple(v),
        }
    }
}

impl IntoIterator for ProcessOutput {
    type Item = Buffer;
    type IntoIter = ProcessOutputIter;

    fn into_iter(self) -> Self::IntoIter {
        ProcessOutputIter(match self {
            ProcessOutput::None | ProcessOutput::Eos | ProcessOutput::Pending => {
                ProcessOutputIterInner::None
            }
            ProcessOutput::Buffer(b) => ProcessOutputIterInner::Single(Some(b)),
            ProcessOutput::Buffers(v) => ProcessOutputIterInner::Multiple(v.into_iter()),
        })
    }
}

/// Iterator over ProcessOutput buffers.
pub struct ProcessOutputIter(ProcessOutputIterInner);

enum ProcessOutputIterInner {
    None,
    Single(Option<Buffer>),
    Multiple(std::vec::IntoIter<Buffer>),
}

impl Iterator for ProcessOutputIter {
    type Item = Buffer;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.0 {
            ProcessOutputIterInner::None => None,
            ProcessOutputIterInner::Single(opt) => opt.take(),
            ProcessOutputIterInner::Multiple(iter) => iter.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.0 {
            ProcessOutputIterInner::None => (0, Some(0)),
            ProcessOutputIterInner::Single(opt) => {
                let n = if opt.is_some() { 1 } else { 0 };
                (n, Some(n))
            }
            ProcessOutputIterInner::Multiple(iter) => iter.size_hint(),
        }
    }
}

impl ExactSizeIterator for ProcessOutputIter {}

// ============================================================================
// PipelineElement - Unified element trait
// ============================================================================

/// Core trait for all pipeline elements.
///
/// This is the unified async trait that all pipeline elements implement.
/// Use the wrapper types `Src`, `Snk`, and `Xfm` to wrap simple element
/// implementations.
///
/// # Element Types
///
/// - **Source**: Produces buffers (input is `None`)
/// - **Sink**: Consumes buffers (returns `ProcessOutput::None`)
/// - **Transform**: Transforms buffers (1 input → 0/1/N outputs)
/// - **Demuxer**: Routes buffers (1 input → N outputs with routing info)
/// - **Muxer**: Combines buffers (N inputs → 1 output)
///
/// # Example
///
/// Implementing directly (for complex elements):
///
/// ```rust,ignore
/// struct MyElement { /* ... */ }
///
/// impl PipelineElement for MyElement {
///     fn element_type(&self) -> ElementType {
///         ElementType::Transform
///     }
///
///     async fn process(&mut self, input: Option<Buffer>) -> Result<ProcessOutput> {
///         match input {
///             Some(buf) => Ok(ProcessOutput::buffer(transform(buf))),
///             None => Ok(ProcessOutput::Eos),
///         }
///     }
/// }
/// ```
///
/// Or use the simpler traits with wrappers:
///
/// ```rust,ignore
/// struct MyTransform;
///
/// impl SimpleTransform for MyTransform {
///     fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput> {
///         Ok(ProcessOutput::buffer(buffer))
///     }
/// }
///
/// // Wrap it:
/// let element = Xfm(MyTransform);
/// ```
#[trait_variant::make(SendPipelineElement: Send)]
pub trait PipelineElement {
    /// Get the element's type (source, sink, transform, demuxer, muxer).
    fn element_type(&self) -> ElementType;

    /// Get the element's name (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Process input and produce output.
    ///
    /// - **Sources**: `input` is `None`, produce data
    /// - **Sinks**: consume `input`, return `ProcessOutput::None`
    /// - **Transforms**: transform `input` to output
    async fn process(&mut self, input: Option<Buffer>) -> Result<ProcessOutput>;

    /// Flush any buffered data (called at EOS).
    ///
    /// Elements that buffer data (like encoders) should drain their
    /// internal buffers here. Called repeatedly until it returns
    /// `ProcessOutput::None` or `ProcessOutput::Eos`.
    async fn flush(&mut self) -> Result<ProcessOutput>;

    /// Get the input capabilities (what formats this element accepts).
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the output capabilities (what formats this element produces).
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
    /// - Not allocate memory in the hot path
    /// - Not perform blocking I/O
    /// - Not take locks that could be held by non-RT threads
    /// - Complete in bounded, deterministic time
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints for automatic scheduling decisions.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// Handle a downstream event (flows with data: EOS, segment, tags).
    ///
    /// Return `Some(event)` to forward, `None` to consume.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)
    }

    /// Handle an upstream event (flows against data: seek, QoS).
    ///
    /// Return `EventResult::Handled` if processed, `NotHandled` to pass upstream.
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }
}

// ============================================================================
// Simple element traits
// ============================================================================

/// A simple source that produces buffers.
///
/// Implement this trait and wrap with `Src` to create a pipeline element.
///
/// # Example
///
/// ```rust,ignore
/// struct Counter { count: u32, max: u32 }
///
/// impl SimpleSource for Counter {
///     fn produce(&mut self) -> Result<ProcessOutput> {
///         if self.count >= self.max {
///             return Ok(ProcessOutput::Eos);
///         }
///         self.count += 1;
///         Ok(ProcessOutput::buffer(create_buffer(self.count)))
///     }
/// }
///
/// let element = Src(Counter { count: 0, max: 10 });
/// ```
pub trait SimpleSource: Send {
    /// Produce the next buffer.
    ///
    /// Return `ProcessOutput::Eos` when the source is exhausted.
    fn produce(&mut self) -> Result<ProcessOutput>;

    /// Get the element's name.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get output capabilities.
    fn output_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get scheduling affinity.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if RT-safe.
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// Handle upstream events (e.g., seek).
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }
}

/// A simple sink that consumes buffers.
///
/// Implement this trait and wrap with `Snk` to create a pipeline element.
///
/// # Example
///
/// ```rust,ignore
/// struct Logger;
///
/// impl SimpleSink for Logger {
///     fn consume(&mut self, buffer: &Buffer) -> Result<()> {
///         println!("Received buffer: {} bytes", buffer.len());
///         Ok(())
///     }
/// }
///
/// let element = Snk(Logger);
/// ```
pub trait SimpleSink: Send {
    /// Consume a buffer.
    fn consume(&mut self, buffer: &Buffer) -> Result<()>;

    /// Get the element's name.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get input capabilities.
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get scheduling affinity.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if RT-safe.
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// Handle downstream events (e.g., EOS).
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)
    }
}

/// A simple transform (1 input → 0/1/N outputs).
///
/// Implement this trait and wrap with `Xfm` to create a pipeline element.
///
/// # Example
///
/// ```rust,ignore
/// struct PassThrough;
///
/// impl SimpleTransform for PassThrough {
///     fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput> {
///         Ok(ProcessOutput::buffer(buffer))
///     }
/// }
///
/// let element = Xfm(PassThrough);
/// ```
pub trait SimpleTransform: Send {
    /// Transform an input buffer into output(s).
    fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput>;

    /// Flush any buffered data at EOS.
    fn flush(&mut self) -> Result<ProcessOutput> {
        Ok(ProcessOutput::None)
    }

    /// Get the element's name.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get input capabilities.
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get output capabilities.
    fn output_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get scheduling affinity.
    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    /// Check if RT-safe.
    fn is_rt_safe(&self) -> bool {
        false
    }

    /// Get execution hints.
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints::default()
    }

    /// Handle downstream events.
    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        Some(event)
    }

    /// Handle upstream events.
    fn handle_upstream_event(&mut self, _event: &Event) -> EventResult {
        EventResult::NotHandled
    }
}

// ============================================================================
// Wrapper types (newtype pattern for zero-cost abstraction)
// ============================================================================

/// Wrapper to convert a `SimpleSource` into a `PipelineElement`.
///
/// # Example
///
/// ```rust,ignore
/// let element = Src(MySource::new());
/// pipeline.add_element("src", Box::new(element));
/// ```
pub struct Src<T: SimpleSource>(pub T);

impl<T: SimpleSource + 'static> SendPipelineElement for Src<T> {
    fn element_type(&self) -> ElementType {
        ElementType::Source
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    async fn process(&mut self, _input: Option<Buffer>) -> Result<ProcessOutput> {
        self.0.produce()
    }

    async fn flush(&mut self) -> Result<ProcessOutput> {
        Ok(ProcessOutput::None)
    }

    fn output_caps(&self) -> Caps {
        self.0.output_caps()
    }

    fn affinity(&self) -> Affinity {
        self.0.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.0.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.0.execution_hints()
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        self.0.handle_upstream_event(event)
    }
}

/// Wrapper to convert a `SimpleSink` into a `PipelineElement`.
///
/// # Example
///
/// ```rust,ignore
/// let element = Snk(MySink::new());
/// pipeline.add_element("sink", Box::new(element));
/// ```
pub struct Snk<T: SimpleSink>(pub T);

impl<T: SimpleSink + 'static> SendPipelineElement for Snk<T> {
    fn element_type(&self) -> ElementType {
        ElementType::Sink
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<ProcessOutput> {
        match input {
            Some(ref buffer) => {
                self.0.consume(buffer)?;
                Ok(ProcessOutput::None)
            }
            None => Ok(ProcessOutput::Eos),
        }
    }

    async fn flush(&mut self) -> Result<ProcessOutput> {
        Ok(ProcessOutput::None)
    }

    fn input_caps(&self) -> Caps {
        self.0.input_caps()
    }

    fn affinity(&self) -> Affinity {
        self.0.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.0.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.0.execution_hints()
    }

    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        self.0.handle_downstream_event(event)
    }
}

/// Wrapper to convert a `SimpleTransform` into a `PipelineElement`.
///
/// # Example
///
/// ```rust,ignore
/// let element = Xfm(MyTransform::new());
/// pipeline.add_element("transform", Box::new(element));
/// ```
pub struct Xfm<T: SimpleTransform>(pub T);

impl<T: SimpleTransform + 'static> SendPipelineElement for Xfm<T> {
    fn element_type(&self) -> ElementType {
        ElementType::Transform
    }

    fn name(&self) -> &str {
        self.0.name()
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<ProcessOutput> {
        match input {
            Some(buffer) => self.0.transform(buffer),
            None => Ok(ProcessOutput::Eos),
        }
    }

    async fn flush(&mut self) -> Result<ProcessOutput> {
        self.0.flush()
    }

    fn input_caps(&self) -> Caps {
        self.0.input_caps()
    }

    fn output_caps(&self) -> Caps {
        self.0.output_caps()
    }

    fn affinity(&self) -> Affinity {
        self.0.affinity()
    }

    fn is_rt_safe(&self) -> bool {
        self.0.is_rt_safe()
    }

    fn execution_hints(&self) -> ExecutionHints {
        self.0.execution_hints()
    }

    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        self.0.handle_downstream_event(event)
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        self.0.handle_upstream_event(event)
    }
}

// ============================================================================
// Bridge to legacy AsyncElementDyn
// ============================================================================

/// Adapter that wraps a `PipelineElement` to implement `AsyncElementDyn`.
///
/// This enables gradual migration: elements can implement the new
/// `PipelineElement` trait and still work with the existing executor.
///
/// # Example
///
/// ```rust,ignore
/// let source = Src(MySource::new());
/// let adapted = PipelineElementAdapter::new(source);
/// pipeline.add_node("src", Box::new(adapted));
/// ```
pub struct PipelineElementAdapter<T: SendPipelineElement> {
    inner: T,
}

impl<T: SendPipelineElement> PipelineElementAdapter<T> {
    /// Create a new adapter wrapping a `PipelineElement`.
    pub fn new(element: T) -> Self {
        Self { inner: element }
    }

    /// Get a reference to the inner element.
    pub fn inner(&self) -> &T {
        &self.inner
    }

    /// Get a mutable reference to the inner element.
    pub fn inner_mut(&mut self) -> &mut T {
        &mut self.inner
    }

    /// Consume the adapter and return the inner element.
    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T: SendPipelineElement + 'static> super::traits::SendAsyncElementDyn
    for PipelineElementAdapter<T>
{
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        self.inner.element_type()
    }

    async fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        let output = SendPipelineElement::process(&mut self.inner, input).await?;
        match output {
            ProcessOutput::None => Ok(None),
            ProcessOutput::Buffer(b) => Ok(Some(b)),
            ProcessOutput::Buffers(mut v) => {
                // Return first buffer, discard rest (limitation of legacy interface)
                // For proper multi-output, use process_all
                Ok(if v.is_empty() {
                    None
                } else {
                    Some(v.remove(0))
                })
            }
            ProcessOutput::Eos => Ok(None),
            ProcessOutput::Pending => Ok(None),
        }
    }

    async fn process_all(&mut self, input: Option<Buffer>) -> Result<super::traits::Output> {
        let output = SendPipelineElement::process(&mut self.inner, input).await?;
        Ok(output.into())
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

    async fn flush(&mut self) -> Result<super::traits::Output> {
        let output = SendPipelineElement::flush(&mut self.inner).await?;
        Ok(output.into())
    }

    fn handle_downstream_event(&mut self, event: Event) -> Option<Event> {
        self.inner.handle_downstream_event(event)
    }

    fn handle_upstream_event(&mut self, event: &Event) -> EventResult {
        self.inner.handle_upstream_event(event)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::CpuSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    fn make_buffer(seq: u64) -> Buffer {
        let segment = Arc::new(CpuSegment::new(64).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    // ========================================================================
    // ProcessOutput tests
    // ========================================================================

    #[test]
    fn test_process_output_none() {
        let out = ProcessOutput::none();
        assert!(out.is_none());
        assert!(out.is_empty());
        assert_eq!(out.len(), 0);
        assert!(!out.is_eos());
        assert!(!out.is_pending());
        assert!(!out.has_output());
    }

    #[test]
    fn test_process_output_eos() {
        let out = ProcessOutput::eos();
        assert!(out.is_eos());
        assert!(!out.is_none());
        assert!(out.is_empty());
        assert!(!out.has_output());
    }

    #[test]
    fn test_process_output_pending() {
        let out = ProcessOutput::pending();
        assert!(out.is_pending());
        assert!(!out.is_eos());
        assert!(out.is_empty());
        assert!(!out.has_output());
    }

    #[test]
    fn test_process_output_buffer() {
        let buf = make_buffer(42);
        let out = ProcessOutput::buffer(buf);
        assert!(out.is_single());
        assert!(out.has_output());
        assert_eq!(out.len(), 1);
        assert!(!out.is_empty());
    }

    #[test]
    fn test_process_output_multiple() {
        let bufs = vec![make_buffer(1), make_buffer(2), make_buffer(3)];
        let out = ProcessOutput::multiple(bufs);
        assert!(out.is_multiple());
        assert!(out.has_output());
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_process_output_multiple_empty() {
        let out = ProcessOutput::multiple(vec![]);
        assert!(out.is_none());
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn test_process_output_multiple_single() {
        let out = ProcessOutput::multiple(vec![make_buffer(1)]);
        assert!(out.is_single()); // Collapsed to single
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_process_output_from_buffer() {
        let buf = make_buffer(42);
        let out: ProcessOutput = buf.into();
        assert!(out.is_single());
    }

    #[test]
    fn test_process_output_from_option() {
        let out: ProcessOutput = Some(make_buffer(42)).into();
        assert!(out.is_single());

        let out: ProcessOutput = None.into();
        assert!(out.is_none());
    }

    #[test]
    fn test_process_output_from_vec() {
        let out: ProcessOutput = vec![make_buffer(1), make_buffer(2)].into();
        assert!(out.is_multiple());
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn test_process_output_into_vec() {
        let out = ProcessOutput::multiple(vec![make_buffer(1), make_buffer(2)]);
        let v = out.into_vec();
        assert_eq!(v.len(), 2);
    }

    #[test]
    fn test_process_output_as_buffer() {
        let out = ProcessOutput::buffer(make_buffer(42));
        assert!(out.as_buffer().is_some());
        assert_eq!(out.as_buffer().unwrap().metadata().sequence, 42);

        let out = ProcessOutput::None;
        assert!(out.as_buffer().is_none());
    }

    #[test]
    fn test_process_output_into_buffer() {
        let out = ProcessOutput::buffer(make_buffer(42));
        let buf = out.into_buffer();
        assert!(buf.is_some());
        assert_eq!(buf.unwrap().metadata().sequence, 42);
    }

    #[test]
    fn test_process_output_take_first() {
        let mut out = ProcessOutput::multiple(vec![make_buffer(1), make_buffer(2)]);
        let first = out.take_first();
        assert!(first.is_some());
        assert_eq!(first.unwrap().metadata().sequence, 1);
        // After taking first, should have 1 buffer left
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn test_process_output_iterator() {
        let out = ProcessOutput::multiple(vec![make_buffer(1), make_buffer(2), make_buffer(3)]);
        let seqs: Vec<u64> = out.into_iter().map(|b| b.metadata().sequence).collect();
        assert_eq!(seqs, vec![1, 2, 3]);
    }

    #[test]
    fn test_process_output_from_output() {
        use super::super::traits::Output;

        let out: ProcessOutput = Output::None.into();
        assert!(out.is_none());

        let out: ProcessOutput = Output::Single(make_buffer(42)).into();
        assert!(out.is_single());

        let out: ProcessOutput = Output::Multiple(vec![make_buffer(1), make_buffer(2)]).into();
        assert!(out.is_multiple());
    }

    // ========================================================================
    // Src wrapper tests
    // ========================================================================

    struct TestSimpleSource {
        count: u32,
        max: u32,
    }

    impl SimpleSource for TestSimpleSource {
        fn produce(&mut self) -> Result<ProcessOutput> {
            if self.count >= self.max {
                return Ok(ProcessOutput::Eos);
            }
            self.count += 1;
            Ok(ProcessOutput::buffer(make_buffer(self.count as u64)))
        }
    }

    #[tokio::test]
    async fn test_src_wrapper() {
        let mut src = Src(TestSimpleSource { count: 0, max: 3 });

        assert_eq!(SendPipelineElement::element_type(&src), ElementType::Source);

        // Should produce 3 buffers then EOS
        let out = SendPipelineElement::process(&mut src, None).await.unwrap();
        assert!(out.is_single());

        let out = SendPipelineElement::process(&mut src, None).await.unwrap();
        assert!(out.is_single());

        let out = SendPipelineElement::process(&mut src, None).await.unwrap();
        assert!(out.is_single());

        let out = SendPipelineElement::process(&mut src, None).await.unwrap();
        assert!(out.is_eos());
    }

    // ========================================================================
    // Snk wrapper tests
    // ========================================================================

    struct TestSimpleSink {
        received: Vec<u64>,
    }

    impl SimpleSink for TestSimpleSink {
        fn consume(&mut self, buffer: &Buffer) -> Result<()> {
            self.received.push(buffer.metadata().sequence);
            Ok(())
        }
    }

    #[tokio::test]
    async fn test_snk_wrapper() {
        let mut snk = Snk(TestSimpleSink { received: vec![] });

        assert_eq!(SendPipelineElement::element_type(&snk), ElementType::Sink);

        // Consume some buffers
        for i in 1..=3 {
            let out = SendPipelineElement::process(&mut snk, Some(make_buffer(i)))
                .await
                .unwrap();
            assert!(out.is_none());
        }

        assert_eq!(snk.0.received, vec![1, 2, 3]);

        // EOS
        let out = SendPipelineElement::process(&mut snk, None).await.unwrap();
        assert!(out.is_eos());
    }

    // ========================================================================
    // Xfm wrapper tests
    // ========================================================================

    struct TestSimpleTransform;

    impl SimpleTransform for TestSimpleTransform {
        fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput> {
            Ok(ProcessOutput::buffer(buffer))
        }
    }

    #[tokio::test]
    async fn test_xfm_wrapper() {
        let mut xfm = Xfm(TestSimpleTransform);

        assert_eq!(
            SendPipelineElement::element_type(&xfm),
            ElementType::Transform
        );

        // Transform a buffer
        let buf = make_buffer(42);
        let out = SendPipelineElement::process(&mut xfm, Some(buf))
            .await
            .unwrap();
        assert!(out.is_single());
        assert_eq!(out.as_buffer().unwrap().metadata().sequence, 42);

        // EOS
        let out = SendPipelineElement::process(&mut xfm, None).await.unwrap();
        assert!(out.is_eos());
    }

    // ========================================================================
    // Multi-output transform test
    // ========================================================================

    struct SplittingTransform;

    impl SimpleTransform for SplittingTransform {
        fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput> {
            // Split one buffer into two
            let seq = buffer.metadata().sequence;
            Ok(ProcessOutput::multiple(vec![
                make_buffer(seq * 10),
                make_buffer(seq * 10 + 1),
            ]))
        }
    }

    #[tokio::test]
    async fn test_xfm_multi_output() {
        let mut xfm = Xfm(SplittingTransform);

        let buf = make_buffer(1);
        let out = SendPipelineElement::process(&mut xfm, Some(buf))
            .await
            .unwrap();

        assert!(out.is_multiple());
        assert_eq!(out.len(), 2);

        let seqs: Vec<u64> = out.into_iter().map(|b| b.metadata().sequence).collect();
        assert_eq!(seqs, vec![10, 11]);
    }

    // ========================================================================
    // Flush test
    // ========================================================================

    struct BufferingTransform {
        buffered: Option<Buffer>,
    }

    impl SimpleTransform for BufferingTransform {
        fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput> {
            // Buffer one, output previous
            let prev = self.buffered.take();
            self.buffered = Some(buffer);
            Ok(prev.into())
        }

        fn flush(&mut self) -> Result<ProcessOutput> {
            Ok(self.buffered.take().into())
        }
    }

    #[tokio::test]
    async fn test_xfm_flush() {
        let mut xfm = Xfm(BufferingTransform { buffered: None });

        // First buffer is buffered, no output
        let out = SendPipelineElement::process(&mut xfm, Some(make_buffer(1)))
            .await
            .unwrap();
        assert!(out.is_none());

        // Second buffer outputs first
        let out = SendPipelineElement::process(&mut xfm, Some(make_buffer(2)))
            .await
            .unwrap();
        assert!(out.is_single());
        assert_eq!(out.as_buffer().unwrap().metadata().sequence, 1);

        // Flush outputs the buffered one
        let out = SendPipelineElement::flush(&mut xfm).await.unwrap();
        assert!(out.is_single());
        assert_eq!(out.as_buffer().unwrap().metadata().sequence, 2);

        // Flush again returns none
        let out = SendPipelineElement::flush(&mut xfm).await.unwrap();
        assert!(out.is_none());
    }
}
