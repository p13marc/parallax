//! Core element traits.

use crate::buffer::Buffer;
use crate::error::Result;
use crate::format::Caps;
use smallvec::SmallVec;

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
#[derive(Debug)]
pub enum Output {
    /// No output (buffer was filtered/consumed).
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

impl Default for Output {
    fn default() -> Self {
        Self::None
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
/// # Lifecycle
///
/// - `produce()` is called repeatedly by the executor
/// - Return `Ok(Some(buffer))` to emit a buffer
/// - Return `Ok(None)` to signal end-of-stream (EOS)
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
///     fn produce(&mut self) -> Result<Option<Buffer>> {
///         if self.count >= self.max {
///             return Ok(None); // EOS
///         }
///         let buffer = Buffer::from_bytes(
///             self.count.to_le_bytes().to_vec(),
///             Metadata::from_sequence(self.count),
///         );
///         self.count += 1;
///         Ok(Some(buffer))
///     }
/// }
/// ```
pub trait Source: Send {
    /// Produce the next buffer.
    ///
    /// Returns `Ok(None)` when the source is exhausted (end of stream).
    fn produce(&mut self) -> Result<Option<Buffer>>;

    /// Get the name of this source (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the output caps (what formats this source produces).
    fn output_caps(&self) -> Caps {
        Caps::any()
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
///     async fn produce(&mut self) -> Result<Option<Buffer>> {
///         let mut buf = vec![0u8; 4096];
///         let n = self.reader.read(&mut buf).await?;
///         if n == 0 {
///             return Ok(None); // Connection closed
///         }
///         buf.truncate(n);
///         Ok(Some(Buffer::from_bytes(buf, Metadata::default())))
///     }
/// }
/// ```
pub trait AsyncSource: Send {
    /// Produce the next buffer asynchronously.
    fn produce(&mut self) -> impl std::future::Future<Output = Result<Option<Buffer>>> + Send;

    /// Get the name of this source (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the output caps (what formats this source produces).
    fn output_caps(&self) -> Caps {
        Caps::any()
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
/// # Example
///
/// ```rust,ignore
/// struct FileSink {
///     file: std::fs::File,
/// }
///
/// impl Sink for FileSink {
///     fn consume(&mut self, buffer: Buffer) -> Result<()> {
///         self.file.write_all(buffer.as_bytes())?;
///         Ok(())
///     }
/// }
/// ```
pub trait Sink: Send {
    /// Consume a buffer.
    fn consume(&mut self, buffer: Buffer) -> Result<()>;

    /// Get the name of this sink (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the input caps (what formats this sink accepts).
    fn input_caps(&self) -> Caps {
        Caps::any()
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
///     async fn consume(&mut self, buffer: Buffer) -> Result<()> {
///         self.writer.write_all(buffer.as_bytes()).await?;
///         Ok(())
///     }
/// }
/// ```
pub trait AsyncSink: Send {
    /// Consume a buffer asynchronously.
    fn consume(&mut self, buffer: Buffer) -> impl std::future::Future<Output = Result<()>> + Send;

    /// Get the name of this sink (for debugging/logging).
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }

    /// Get the input caps (what formats this sink accepts).
    fn input_caps(&self) -> Caps {
        Caps::any()
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
}

// ============================================================================
// Dynamic Element (Type-Erased)
// ============================================================================

/// Dynamic (type-erased) element trait.
///
/// This trait is used internally by the pipeline executor to handle
/// elements uniformly, regardless of their concrete type.
///
/// Most users should implement [`Source`], [`Sink`], or [`Element`] instead.
pub trait ElementDyn: Send {
    /// Get the element's name.
    fn name(&self) -> &str;

    /// Get the element's type (source, sink, or transform).
    fn element_type(&self) -> ElementType;

    /// Process or produce a buffer.
    ///
    /// - For sources: `input` is `None`, returns produced buffer
    /// - For sinks: `input` is `Some`, returns `None`
    /// - For transforms: `input` is `Some`, returns transformed buffer
    fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>>;

    /// Process and return all outputs (for transforms that produce multiple).
    ///
    /// Default implementation wraps `process` in a single-element output.
    fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
        self.process(input).map(Output::from)
    }

    /// Get the input caps (for validation).
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    /// Get the output caps (for validation).
    fn output_caps(&self) -> Caps {
        Caps::any()
    }
}

/// The type of an element in the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ElementType {
    /// A source element (produces buffers).
    Source,
    /// A sink element (consumes buffers).
    Sink,
    /// A transform element (transforms buffers).
    Transform,
}

// ============================================================================
// Adapters
// ============================================================================

/// Wrapper to adapt a [`Source`] to [`ElementDyn`].
pub struct SourceAdapter<S: Source> {
    inner: S,
}

impl<S: Source> SourceAdapter<S> {
    /// Create a new source adapter.
    pub fn new(source: S) -> Self {
        Self { inner: source }
    }
}

impl<S: Source + 'static> ElementDyn for SourceAdapter<S> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Source
    }

    fn process(&mut self, _input: Option<Buffer>) -> Result<Option<Buffer>> {
        self.inner.produce()
    }

    fn output_caps(&self) -> Caps {
        self.inner.output_caps()
    }
}

/// Wrapper to adapt a [`Sink`] to [`ElementDyn`].
pub struct SinkAdapter<S: Sink> {
    inner: S,
}

impl<S: Sink> SinkAdapter<S> {
    /// Create a new sink adapter.
    pub fn new(sink: S) -> Self {
        Self { inner: sink }
    }
}

impl<S: Sink + 'static> ElementDyn for SinkAdapter<S> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Sink
    }

    fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        if let Some(buffer) = input {
            self.inner.consume(buffer)?;
        }
        Ok(None)
    }

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }
}

/// Wrapper to adapt an [`Element`] to [`ElementDyn`].
pub struct ElementAdapter<E: Element> {
    inner: E,
}

impl<E: Element> ElementAdapter<E> {
    /// Create a new element adapter.
    pub fn new(element: E) -> Self {
        Self { inner: element }
    }
}

impl<E: Element + 'static> ElementDyn for ElementAdapter<E> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Transform
    }

    fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
        match input {
            Some(buffer) => self.inner.process(buffer),
            None => Ok(None),
        }
    }

    fn input_caps(&self) -> Caps {
        self.inner.input_caps()
    }

    fn output_caps(&self) -> Caps {
        self.inner.output_caps()
    }
}

/// Wrapper to adapt a [`Transform`] to [`ElementDyn`].
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

impl<T: Transform + 'static> ElementDyn for TransformAdapter<T> {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn element_type(&self) -> ElementType {
        ElementType::Transform
    }

    fn process(&mut self, input: Option<Buffer>) -> Result<Option<Buffer>> {
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

    fn process_all(&mut self, input: Option<Buffer>) -> Result<Output> {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::Arc;

    struct TestSource {
        count: u64,
        max: u64,
    }

    impl Source for TestSource {
        fn produce(&mut self) -> Result<Option<Buffer>> {
            if self.count >= self.max {
                return Ok(None);
            }
            let segment = Arc::new(HeapSegment::new(8).unwrap());
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::new(handle, Metadata::from_sequence(self.count));
            self.count += 1;
            Ok(Some(buffer))
        }
    }

    struct TestSink {
        received: Vec<u64>,
    }

    impl Sink for TestSink {
        fn consume(&mut self, buffer: Buffer) -> Result<()> {
            self.received.push(buffer.metadata().sequence);
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

    #[test]
    fn test_source_adapter() {
        let source = TestSource { count: 0, max: 3 };
        let mut adapter = SourceAdapter::new(source);

        assert_eq!(adapter.element_type(), ElementType::Source);

        // Should produce 3 buffers then None
        assert!(adapter.process(None).unwrap().is_some());
        assert!(adapter.process(None).unwrap().is_some());
        assert!(adapter.process(None).unwrap().is_some());
        assert!(adapter.process(None).unwrap().is_none());
    }

    #[test]
    fn test_sink_adapter() {
        let sink = TestSink { received: vec![] };
        let mut adapter = SinkAdapter::new(sink);

        assert_eq!(adapter.element_type(), ElementType::Sink);

        // Create and consume some buffers
        for i in 0..3 {
            let segment = Arc::new(HeapSegment::new(8).unwrap());
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::new(handle, Metadata::from_sequence(i));
            adapter.process(Some(buffer)).unwrap();
        }
    }

    #[test]
    fn test_element_adapter() {
        let element = PassThrough;
        let mut adapter = ElementAdapter::new(element);

        assert_eq!(adapter.element_type(), ElementType::Transform);

        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::from_sequence(42));

        let result = adapter.process(Some(buffer)).unwrap();
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
}
