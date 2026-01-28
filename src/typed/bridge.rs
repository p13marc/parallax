//! Bridge between typed and dynamic pipelines.
//!
//! This module provides adapters to convert typed elements to dynamic elements,
//! allowing typed pipelines to be used with the dynamic pipeline executor.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::{
    ConsumeContext, DynAsyncElement, ElementAdapter, ProduceContext, ProduceResult, SinkAdapter,
    SourceAdapter,
};
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use crate::metadata::Metadata;
use crate::pipeline::Pipeline;

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
    /// Arena for allocating buffers when context doesn't provide one.
    arena: Option<SharedArena>,
}

impl<S: TypedSource> TypedSourceBridge<S> {
    /// Create a new bridge for a typed source.
    pub fn new(source: S) -> Self {
        Self {
            inner: source,
            sequence: 0,
            arena: None,
        }
    }
}

impl<S> crate::element::Source for TypedSourceBridge<S>
where
    S: TypedSource,
    S::Output: Into<Vec<u8>>,
{
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        match self.inner.produce()? {
            Some(output) => {
                let bytes: Vec<u8> = output.into();

                // Check if we have a pre-allocated buffer from the context
                if ctx.has_buffer() && ctx.capacity() >= bytes.len() {
                    // Use the pre-allocated buffer
                    let output_buf = ctx.output();
                    output_buf[..bytes.len()].copy_from_slice(&bytes);
                    ctx.set_sequence(self.sequence);
                    self.sequence += 1;
                    Ok(ProduceResult::Produced(bytes.len()))
                } else {
                    // Fall back to creating our own buffer using SharedArena
                    let slot_size = bytes.len().max(4096).next_power_of_two();
                    if self.arena.is_none()
                        || self.arena.as_ref().map(|a| a.slot_size()).unwrap_or(0) < bytes.len()
                    {
                        self.arena = Some(SharedArena::new(slot_size, 32)?);
                    }

                    let arena = self.arena.as_mut().unwrap();
                    // Reclaim any released slots first
                    arena.reclaim();
                    let mut slot = arena
                        .acquire()
                        .ok_or_else(|| Error::Element("arena exhausted".into()))?;

                    // Copy data to slot
                    slot.data_mut()[..bytes.len()].copy_from_slice(&bytes);

                    let handle = MemoryHandle::with_len(slot, bytes.len());
                    let metadata = Metadata::from_sequence(self.sequence);
                    self.sequence += 1;

                    Ok(ProduceResult::OwnBuffer(Buffer::new(handle, metadata)))
                }
            }
            None => Ok(ProduceResult::Eos),
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
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let bytes = ctx.input().to_vec();
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
    /// Arena for allocating output buffers.
    arena: Option<SharedArena>,
}

impl<T: TypedTransform> TypedTransformBridge<T> {
    /// Create a new bridge for a typed transform.
    pub fn new(transform: T) -> Self {
        Self {
            inner: transform,
            arena: None,
        }
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

                // Ensure we have an arena with sufficient slot size
                let slot_size = out_bytes.len().max(4096).next_power_of_two();
                if self.arena.is_none()
                    || self.arena.as_ref().map(|a| a.slot_size()).unwrap_or(0) < out_bytes.len()
                {
                    self.arena = Some(SharedArena::new(slot_size, 32)?);
                }

                let arena = self.arena.as_mut().unwrap();
                // Reclaim any released slots first
                arena.reclaim();
                let mut slot = arena
                    .acquire()
                    .ok_or_else(|| Error::Element("arena exhausted".into()))?;

                // Copy data to slot
                slot.data_mut()[..out_bytes.len()].copy_from_slice(&out_bytes);

                let handle = MemoryHandle::with_len(slot, out_bytes.len());
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

/// Convert a typed source to a boxed dynamic DynAsyncElement.
pub fn source_to_dyn<S>(source: S) -> Box<DynAsyncElement<'static>>
where
    S: TypedSource + 'static,
    S::Output: Into<Vec<u8>>,
{
    DynAsyncElement::new_box(SourceAdapter::new(TypedSourceBridge::new(source)))
}

/// Convert a typed sink to a boxed dynamic DynAsyncElement.
pub fn sink_to_dyn<K>(sink: K) -> Box<DynAsyncElement<'static>>
where
    K: TypedSink + 'static,
    K::Input: TryFrom<Vec<u8>>,
    <K::Input as TryFrom<Vec<u8>>>::Error: std::fmt::Debug,
{
    DynAsyncElement::new_box(SinkAdapter::new(TypedSinkBridge::new(sink)))
}

/// Convert a typed transform to a boxed dynamic DynAsyncElement.
pub fn transform_to_dyn<T>(transform: T) -> Box<DynAsyncElement<'static>>
where
    T: TypedTransform + 'static,
    T::Input: TryFrom<Vec<u8>>,
    T::Output: Into<Vec<u8>>,
    <T::Input as TryFrom<Vec<u8>>>::Error: std::fmt::Debug,
{
    DynAsyncElement::new_box(ElementAdapter::new(TypedTransformBridge::new(transform)))
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
