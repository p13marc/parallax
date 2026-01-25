//! Core element traits.

use crate::buffer::Buffer;
use crate::error::Result;

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
///             Metadata::with_sequence(self.count),
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
}

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
}

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
}

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
            let buffer = Buffer::new(handle, Metadata::with_sequence(self.count));
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
            let buffer = Buffer::new(handle, Metadata::with_sequence(i));
            adapter.process(Some(buffer)).unwrap();
        }

        // Can't easily check received because adapter owns the sink
        // In real code, we'd use channels or other mechanisms
    }

    #[test]
    fn test_element_adapter() {
        let element = PassThrough;
        let mut adapter = ElementAdapter::new(element);

        assert_eq!(adapter.element_type(), ElementType::Transform);

        let segment = Arc::new(HeapSegment::new(8).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::with_sequence(42));

        let result = adapter.process(Some(buffer)).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().metadata().sequence, 42);
    }
}
