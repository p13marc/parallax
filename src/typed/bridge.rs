//! Bridge between typed and dynamic pipelines.
//!
//! This module provides adapters to convert typed elements to dynamic elements,
//! allowing typed pipelines to be used with the dynamic pipeline executor.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{ElementAdapter, ElementDyn, SinkAdapter, SourceAdapter};
use crate::error::Result;
use crate::memory::{HeapSegment, MemorySegment};
use crate::metadata::Metadata;
use crate::pipeline::Pipeline;
use std::sync::Arc;

use super::element::{TypedSink, TypedSource, TypedTransform};

// ============================================================================
// Typed Source to Dynamic Source Adapter
// ============================================================================

/// Adapter that wraps a TypedSource as a dynamic Source.
///
/// The typed output is serialized to bytes for the dynamic pipeline.
pub struct TypedSourceBridge<S: TypedSource> {
    inner: S,
    sequence: u64,
}

impl<S: TypedSource> TypedSourceBridge<S> {
    /// Create a new bridge for a typed source.
    pub fn new(source: S) -> Self {
        Self {
            inner: source,
            sequence: 0,
        }
    }
}

impl<S> crate::element::Source for TypedSourceBridge<S>
where
    S: TypedSource,
    S::Output: Into<Vec<u8>>,
{
    fn produce(&mut self) -> Result<Option<Buffer>> {
        match self.inner.produce()? {
            Some(output) => {
                let bytes: Vec<u8> = output.into();
                let segment = Arc::new(HeapSegment::new(bytes.len())?);

                // Copy data to segment (using unsafe as required by the trait)
                // SAFETY: HeapSegment is mutable and we have exclusive access via Arc
                unsafe {
                    if let Some(slice) = segment.as_mut_slice() {
                        slice.copy_from_slice(&bytes);
                    }
                }

                let handle = MemoryHandle::from_segment(segment);
                let metadata = Metadata::with_sequence(self.sequence);
                self.sequence += 1;

                Ok(Some(Buffer::new(handle, metadata)))
            }
            None => Ok(None),
        }
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// ============================================================================
// Typed Sink to Dynamic Sink Adapter
// ============================================================================

/// Adapter that wraps a TypedSink as a dynamic Sink.
///
/// The dynamic buffer bytes are deserialized to the typed input.
pub struct TypedSinkBridge<K: TypedSink> {
    inner: K,
}

impl<K: TypedSink> TypedSinkBridge<K> {
    /// Create a new bridge for a typed sink.
    pub fn new(sink: K) -> Self {
        Self { inner: sink }
    }

    /// Get the inner sink (consumes the bridge).
    pub fn into_inner(self) -> K {
        self.inner
    }
}

impl<K> crate::element::Sink for TypedSinkBridge<K>
where
    K: TypedSink,
    K::Input: TryFrom<Vec<u8>>,
    <K::Input as TryFrom<Vec<u8>>>::Error: std::fmt::Debug,
{
    fn consume(&mut self, buffer: Buffer) -> Result<()> {
        let bytes = buffer.as_bytes().to_vec();
        let item = K::Input::try_from(bytes)
            .map_err(|e| crate::error::Error::InvalidCaps(format!("{:?}", e)))?;
        self.inner.consume(item)
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// ============================================================================
// Typed Transform to Dynamic Element Adapter
// ============================================================================

/// Adapter that wraps a TypedTransform as a dynamic Element.
pub struct TypedTransformBridge<T: TypedTransform> {
    inner: T,
}

impl<T: TypedTransform> TypedTransformBridge<T> {
    /// Create a new bridge for a typed transform.
    pub fn new(transform: T) -> Self {
        Self { inner: transform }
    }
}

impl<T> crate::element::Element for TypedTransformBridge<T>
where
    T: TypedTransform,
    T::Input: TryFrom<Vec<u8>>,
    T::Output: Into<Vec<u8>>,
    <T::Input as TryFrom<Vec<u8>>>::Error: std::fmt::Debug,
{
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let bytes = buffer.as_bytes().to_vec();
        let metadata = buffer.metadata().clone();

        let item = T::Input::try_from(bytes)
            .map_err(|e| crate::error::Error::InvalidCaps(format!("{:?}", e)))?;

        match self.inner.transform(item)? {
            Some(output) => {
                let out_bytes: Vec<u8> = output.into();
                let segment = Arc::new(HeapSegment::new(out_bytes.len())?);

                // SAFETY: HeapSegment is mutable and we have exclusive access via Arc
                unsafe {
                    if let Some(slice) = segment.as_mut_slice() {
                        slice.copy_from_slice(&out_bytes);
                    }
                }

                let handle = MemoryHandle::from_segment(segment);
                Ok(Some(Buffer::new(handle, metadata)))
            }
            None => Ok(None),
        }
    }

    fn name(&self) -> &str {
        self.inner.name()
    }
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Convert a typed source to a boxed dynamic ElementDyn.
pub fn source_to_dyn<S>(source: S) -> Box<dyn ElementDyn>
where
    S: TypedSource + 'static,
    S::Output: Into<Vec<u8>>,
{
    Box::new(SourceAdapter::new(TypedSourceBridge::new(source)))
}

/// Convert a typed sink to a boxed dynamic ElementDyn.
pub fn sink_to_dyn<K>(sink: K) -> Box<dyn ElementDyn>
where
    K: TypedSink + 'static,
    K::Input: TryFrom<Vec<u8>>,
    <K::Input as TryFrom<Vec<u8>>>::Error: std::fmt::Debug,
{
    Box::new(SinkAdapter::new(TypedSinkBridge::new(sink)))
}

/// Convert a typed transform to a boxed dynamic ElementDyn.
pub fn transform_to_dyn<T>(transform: T) -> Box<dyn ElementDyn>
where
    T: TypedTransform + 'static,
    T::Input: TryFrom<Vec<u8>>,
    T::Output: Into<Vec<u8>>,
    <T::Input as TryFrom<Vec<u8>>>::Error: std::fmt::Debug,
{
    Box::new(ElementAdapter::new(TypedTransformBridge::new(transform)))
}

/// Build a dynamic pipeline from typed elements.
///
/// This is a convenience builder for creating dynamic pipelines
/// from typed components.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::typed::{from_iter, map, collect};
/// use parallax::typed::bridge::DynamicPipelineBuilder;
///
/// let mut builder = DynamicPipelineBuilder::new();
/// builder
///     .source("src", from_iter(vec![1u32, 2, 3]))
///     .transform("double", map(|x: u32| x * 2))
///     .sink("sink", collect::<u32>());
///
/// let pipeline = builder.build();
/// ```
pub struct DynamicPipelineBuilder {
    pipeline: Pipeline,
    last_node: Option<crate::pipeline::NodeId>,
}

impl DynamicPipelineBuilder {
    /// Create a new builder.
    pub fn new() -> Self {
        Self {
            pipeline: Pipeline::new(),
            last_node: None,
        }
    }

    /// Add a typed source to the pipeline.
    pub fn source<S>(&mut self, name: &str, source: S) -> &mut Self
    where
        S: TypedSource + 'static,
        S::Output: Into<Vec<u8>>,
    {
        let node = self.pipeline.add_node(name, source_to_dyn(source));
        self.last_node = Some(node);
        self
    }

    /// Add a typed transform to the pipeline.
    pub fn transform<T>(&mut self, name: &str, transform: T) -> &mut Self
    where
        T: TypedTransform + 'static,
        T::Input: TryFrom<Vec<u8>>,
        T::Output: Into<Vec<u8>>,
        <T::Input as TryFrom<Vec<u8>>>::Error: std::fmt::Debug,
    {
        let node = self.pipeline.add_node(name, transform_to_dyn(transform));
        if let Some(prev) = self.last_node {
            let _ = self.pipeline.link(prev, node);
        }
        self.last_node = Some(node);
        self
    }

    /// Add a typed sink to the pipeline.
    pub fn sink<K>(&mut self, name: &str, sink: K) -> &mut Self
    where
        K: TypedSink + 'static,
        K::Input: TryFrom<Vec<u8>>,
        <K::Input as TryFrom<Vec<u8>>>::Error: std::fmt::Debug,
    {
        let node = self.pipeline.add_node(name, sink_to_dyn(sink));
        if let Some(prev) = self.last_node {
            let _ = self.pipeline.link(prev, node);
        }
        self.last_node = Some(node);
        self
    }

    /// Build the dynamic pipeline.
    pub fn build(self) -> Pipeline {
        self.pipeline
    }
}

impl Default for DynamicPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dynamic_pipeline_builder() {
        // Just test that the builder works
        let builder = DynamicPipelineBuilder::new();

        // We can't easily test the full pipeline because we need
        // From/TryFrom implementations for the types.
        // The builder itself works correctly.
        assert_eq!(builder.pipeline.node_count(), 0);
    }
}
