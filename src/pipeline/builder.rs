//! Fluent pipeline builder DSL.
//!
//! This module provides an ergonomic builder pattern for constructing pipelines.
//! Instead of manually creating adapters and linking nodes, you can use method
//! chaining or the `>>` operator.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::PipelineBuilder;
//!
//! // Method chaining
//! let pipeline = PipelineBuilder::new()
//!     .source(VideoTestSrc::new())
//!     .then(VideoScale::new(1920, 1080, 1280, 720))
//!     .sink(FileSink::new("output.yuv"))
//!     .build()?;
//!
//! pipeline.run().await?;
//! ```

use crate::element::{
    AsyncSink, AsyncSource, DynAsyncElement, Element, ElementAdapter, Sink, SinkAdapter, Source,
    SourceAdapter, Transform, TransformAdapter,
};
use crate::error::Result;
use crate::memory::CpuArena;
use crate::pipeline::{NodeId, Pipeline};
use std::marker::PhantomData;
use std::sync::Arc;

// ============================================================================
// State Markers
// ============================================================================

/// Marker: Pipeline has no elements yet.
pub struct Empty;

/// Marker: Pipeline has at least one source.
pub struct HasSource;

/// Marker: Pipeline is complete (has source and sink).
pub struct Complete;

// ============================================================================
// PipelineBuilder
// ============================================================================

/// A fluent builder for constructing pipelines.
///
/// The builder uses state markers to enforce correct construction order at
/// compile time:
/// - `Empty`: No elements added yet
/// - `HasSource`: At least one source added, can add transforms or sink
/// - `Complete`: Has both source and sink, ready to build
///
/// # Example
///
/// ```rust,ignore
/// use parallax::pipeline::PipelineBuilder;
///
/// let pipeline = PipelineBuilder::new()
///     .source(MySource::new())
///     .then(MyTransform::new())
///     .sink(MySink::new())
///     .build()?;
/// ```
pub struct PipelineBuilder<State = Empty> {
    pipeline: Pipeline,
    current_node: Option<NodeId>,
    name_counter: u64,
    arena: Option<Arc<CpuArena>>,
    _state: PhantomData<State>,
}

impl Default for PipelineBuilder<Empty> {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineBuilder<Empty> {
    /// Create a new empty pipeline builder.
    pub fn new() -> Self {
        Self {
            pipeline: Pipeline::new(),
            current_node: None,
            name_counter: 0,
            arena: None,
            _state: PhantomData,
        }
    }

    /// Set an arena for buffer allocation.
    ///
    /// Sources will use this arena to allocate buffers.
    pub fn with_arena(mut self, arena: Arc<CpuArena>) -> Self {
        self.arena = Some(arena);
        self
    }

    /// Create an arena with the given slot size and count.
    pub fn with_new_arena(mut self, slot_size: usize, slot_count: usize) -> Result<Self> {
        let arena = CpuArena::new(slot_size, slot_count)?;
        self.arena = Some(arena);
        Ok(self)
    }

    /// Add a source element.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = PipelineBuilder::new()
    ///     .source(VideoTestSrc::new());
    /// ```
    pub fn source<S: Source + 'static>(self, source: S) -> PipelineBuilder<HasSource> {
        let name = Self::auto_name::<S>(&mut { self.name_counter });
        self.source_named(name, source)
    }

    /// Add a source element with a specific name.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = PipelineBuilder::new()
    ///     .source_named("video_src", VideoTestSrc::new());
    /// ```
    pub fn source_named<S: Source + 'static>(
        mut self,
        name: impl Into<String>,
        source: S,
    ) -> PipelineBuilder<HasSource> {
        let name = name.into();
        let adapter = if let Some(arena) = &self.arena {
            SourceAdapter::with_arena(source, arena.clone())
        } else {
            SourceAdapter::new(source)
        };
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        PipelineBuilder {
            pipeline: self.pipeline,
            current_node: Some(node_id),
            name_counter: self.name_counter + 1,
            arena: self.arena,
            _state: PhantomData,
        }
    }

    /// Add an async source element.
    pub fn async_source<S: AsyncSource + 'static>(self, source: S) -> PipelineBuilder<HasSource> {
        let name = Self::auto_name::<S>(&mut { self.name_counter });
        self.async_source_named(name, source)
    }

    /// Add an async source element with a specific name.
    pub fn async_source_named<S: AsyncSource + 'static>(
        mut self,
        name: impl Into<String>,
        source: S,
    ) -> PipelineBuilder<HasSource> {
        use crate::element::AsyncSourceAdapter;

        let name = name.into();
        let adapter = if let Some(arena) = &self.arena {
            AsyncSourceAdapter::with_arena(source, arena.clone())
        } else {
            AsyncSourceAdapter::new(source)
        };
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        PipelineBuilder {
            pipeline: self.pipeline,
            current_node: Some(node_id),
            name_counter: self.name_counter + 1,
            arena: self.arena,
            _state: PhantomData,
        }
    }

    fn auto_name<T>(counter: &mut u64) -> String {
        let type_name = std::any::type_name::<T>();
        let short_name = type_name.rsplit("::").next().unwrap_or(type_name);
        let name = format!("{}_{}", short_name, *counter);
        *counter += 1;
        name
    }
}

impl PipelineBuilder<HasSource> {
    /// Add a transform element.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let builder = PipelineBuilder::new()
    ///     .source(src)
    ///     .then(VideoScale::new(1920, 1080, 1280, 720))
    ///     .then(Rav1eEncoder::new(config)?);
    /// ```
    pub fn then<T: Transform + 'static>(self, transform: T) -> Self {
        let name = self.auto_name_inner::<T>();
        self.then_named(name, transform)
    }

    /// Add a transform element with a specific name.
    pub fn then_named<T: Transform + 'static>(
        mut self,
        name: impl Into<String>,
        transform: T,
    ) -> Self {
        let name = name.into();
        let adapter = TransformAdapter::new(transform);
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        // Link to previous element
        if let Some(prev) = self.current_node {
            self.pipeline
                .link(prev, node_id)
                .expect("failed to link elements");
        }

        self.current_node = Some(node_id);
        self.name_counter += 1;
        self
    }

    /// Add an Element (legacy trait that returns `Option<Buffer>`).
    pub fn then_element<E: Element + 'static>(self, element: E) -> Self {
        let name = self.auto_name_inner::<E>();
        self.then_element_named(name, element)
    }

    /// Add an Element with a specific name.
    pub fn then_element_named<E: Element + 'static>(
        mut self,
        name: impl Into<String>,
        element: E,
    ) -> Self {
        let name = name.into();
        let adapter = ElementAdapter::new(element);
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        if let Some(prev) = self.current_node {
            self.pipeline
                .link(prev, node_id)
                .expect("failed to link elements");
        }

        self.current_node = Some(node_id);
        self.name_counter += 1;
        self
    }

    /// Add a sink element and complete the pipeline.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = PipelineBuilder::new()
    ///     .source(src)
    ///     .then(transform)
    ///     .sink(FileSink::new("output.bin"))
    ///     .build()?;
    /// ```
    pub fn sink<S: Sink + 'static>(self, sink: S) -> PipelineBuilder<Complete> {
        let name = self.auto_name_inner::<S>();
        self.sink_named(name, sink)
    }

    /// Add a sink element with a specific name.
    pub fn sink_named<S: Sink + 'static>(
        mut self,
        name: impl Into<String>,
        sink: S,
    ) -> PipelineBuilder<Complete> {
        let name = name.into();
        let adapter = SinkAdapter::new(sink);
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        if let Some(prev) = self.current_node {
            self.pipeline
                .link(prev, node_id)
                .expect("failed to link elements");
        }

        PipelineBuilder {
            pipeline: self.pipeline,
            current_node: Some(node_id),
            name_counter: self.name_counter + 1,
            arena: self.arena,
            _state: PhantomData,
        }
    }

    /// Add an async sink element.
    pub fn async_sink<S: AsyncSink + 'static>(self, sink: S) -> PipelineBuilder<Complete> {
        let name = self.auto_name_inner::<S>();
        self.async_sink_named(name, sink)
    }

    /// Add an async sink element with a specific name.
    pub fn async_sink_named<S: AsyncSink + 'static>(
        mut self,
        name: impl Into<String>,
        sink: S,
    ) -> PipelineBuilder<Complete> {
        use crate::element::AsyncSinkAdapter;

        let name = name.into();
        let adapter = AsyncSinkAdapter::new(sink);
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        if let Some(prev) = self.current_node {
            self.pipeline
                .link(prev, node_id)
                .expect("failed to link elements");
        }

        PipelineBuilder {
            pipeline: self.pipeline,
            current_node: Some(node_id),
            name_counter: self.name_counter + 1,
            arena: self.arena,
            _state: PhantomData,
        }
    }

    /// Create a tee (branch point) in the pipeline.
    ///
    /// The tee duplicates buffers to all branches. Each branch must end with
    /// a sink.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// let pipeline = PipelineBuilder::new()
    ///     .source(src)
    ///     .tee(|t| {
    ///         t.branch(|b| b
    ///             .then(scale_720p)
    ///             .sink(file_720p));
    ///         t.branch(|b| b
    ///             .then(scale_480p)
    ///             .sink(file_480p));
    ///     })
    ///     .build()?;
    /// ```
    pub fn tee<F>(mut self, f: F) -> PipelineBuilder<Complete>
    where
        F: FnOnce(&mut TeeBuilder),
    {
        use crate::elements::Tee;

        // Add the tee element
        let tee_name = format!("tee_{}", self.name_counter);
        self.name_counter += 1;
        let tee_element = Tee::new();
        let adapter = ElementAdapter::new(tee_element);
        let tee_id = self
            .pipeline
            .add_node(&tee_name, DynAsyncElement::new_box(adapter));

        // Link previous element to tee
        if let Some(prev) = self.current_node {
            self.pipeline
                .link(prev, tee_id)
                .expect("failed to link to tee");
        }

        // Build branches
        let mut tee_builder = TeeBuilder {
            pipeline: &mut self.pipeline,
            tee_node: tee_id,
            name_counter: &mut self.name_counter,
            arena: self.arena.clone(),
        };
        f(&mut tee_builder);

        PipelineBuilder {
            pipeline: self.pipeline,
            current_node: Some(tee_id),
            name_counter: self.name_counter,
            arena: self.arena,
            _state: PhantomData,
        }
    }

    fn auto_name_inner<T>(&self) -> String {
        let type_name = std::any::type_name::<T>();
        let short_name = type_name.rsplit("::").next().unwrap_or(type_name);
        format!("{}_{}", short_name, self.name_counter)
    }
}

impl PipelineBuilder<Complete> {
    /// Build the pipeline.
    ///
    /// Returns the constructed pipeline ready for execution.
    pub fn build(self) -> Result<Pipeline> {
        Ok(self.pipeline)
    }

    /// Build and validate the pipeline.
    ///
    /// Runs validation checks to ensure the pipeline is correctly constructed.
    pub fn build_validated(self) -> Result<Pipeline> {
        let pipeline = self.pipeline;
        pipeline.validate()?;
        Ok(pipeline)
    }
}

// ============================================================================
// TeeBuilder
// ============================================================================

/// Builder for tee branches.
///
/// Used inside the `tee()` callback to define branches.
pub struct TeeBuilder<'a> {
    pipeline: &'a mut Pipeline,
    tee_node: NodeId,
    name_counter: &'a mut u64,
    arena: Option<Arc<CpuArena>>,
}

impl<'a> TeeBuilder<'a> {
    /// Add a branch from the tee point.
    ///
    /// Each branch receives a copy of every buffer that passes through the tee.
    pub fn branch<F>(&mut self, f: F)
    where
        F: FnOnce(BranchBuilder<'_>) -> BranchBuilder<'_, Complete>,
    {
        let branch = BranchBuilder {
            pipeline: self.pipeline,
            start_node: self.tee_node,
            current_node: self.tee_node,
            name_counter: self.name_counter,
            arena: self.arena.clone(),
            _state: PhantomData,
        };
        f(branch);
    }
}

// ============================================================================
// BranchBuilder
// ============================================================================

/// Builder for a single branch in a tee.
pub struct BranchBuilder<'a, State = Empty> {
    pipeline: &'a mut Pipeline,
    start_node: NodeId,
    current_node: NodeId,
    name_counter: &'a mut u64,
    arena: Option<Arc<CpuArena>>,
    _state: PhantomData<State>,
}

impl<'a> BranchBuilder<'a, Empty> {
    /// Add a transform to this branch.
    pub fn then<T: Transform + 'static>(self, transform: T) -> BranchBuilder<'a, HasSource> {
        let name = self.auto_name::<T>();
        let adapter = TransformAdapter::new(transform);
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        self.pipeline
            .link(self.current_node, node_id)
            .expect("failed to link branch element");

        *self.name_counter += 1;

        BranchBuilder {
            pipeline: self.pipeline,
            start_node: self.start_node,
            current_node: node_id,
            name_counter: self.name_counter,
            arena: self.arena,
            _state: PhantomData,
        }
    }

    /// Add a sink to this branch (no transforms).
    pub fn sink<S: Sink + 'static>(self, sink: S) -> BranchBuilder<'a, Complete> {
        let name = self.auto_name::<S>();
        let adapter = SinkAdapter::new(sink);
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        self.pipeline
            .link(self.current_node, node_id)
            .expect("failed to link branch sink");

        *self.name_counter += 1;

        BranchBuilder {
            pipeline: self.pipeline,
            start_node: self.start_node,
            current_node: node_id,
            name_counter: self.name_counter,
            arena: self.arena,
            _state: PhantomData,
        }
    }

    fn auto_name<T>(&self) -> String {
        let type_name = std::any::type_name::<T>();
        let short_name = type_name.rsplit("::").next().unwrap_or(type_name);
        format!("branch_{}_{}", short_name, self.name_counter)
    }
}

impl<'a> BranchBuilder<'a, HasSource> {
    /// Add another transform to this branch.
    pub fn then<T: Transform + 'static>(mut self, transform: T) -> Self {
        let name = self.auto_name::<T>();
        let adapter = TransformAdapter::new(transform);
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        self.pipeline
            .link(self.current_node, node_id)
            .expect("failed to link branch element");

        self.current_node = node_id;
        *self.name_counter += 1;
        self
    }

    /// Add a sink to complete this branch.
    pub fn sink<S: Sink + 'static>(self, sink: S) -> BranchBuilder<'a, Complete> {
        let name = self.auto_name::<S>();
        let adapter = SinkAdapter::new(sink);
        let node_id = self
            .pipeline
            .add_node(&name, DynAsyncElement::new_box(adapter));

        self.pipeline
            .link(self.current_node, node_id)
            .expect("failed to link branch sink");

        *self.name_counter += 1;

        BranchBuilder {
            pipeline: self.pipeline,
            start_node: self.start_node,
            current_node: node_id,
            name_counter: self.name_counter,
            arena: self.arena,
            _state: PhantomData,
        }
    }

    fn auto_name<T>(&self) -> String {
        let type_name = std::any::type_name::<T>();
        let short_name = type_name.rsplit("::").next().unwrap_or(type_name);
        format!("branch_{}_{}", short_name, self.name_counter)
    }
}

// ============================================================================
// Operator Syntax (>>)
// ============================================================================

/// A pipeline fragment that can be extended with `>>`.
pub struct PipelineFragment<State> {
    builder: PipelineBuilder<State>,
}

impl<S: Source + 'static> std::ops::Shr<S> for PipelineBuilder<Empty> {
    type Output = PipelineFragment<HasSource>;

    fn shr(self, source: S) -> Self::Output {
        PipelineFragment {
            builder: self.source(source),
        }
    }
}

impl<T: Transform + 'static> std::ops::Shr<T> for PipelineFragment<HasSource> {
    type Output = PipelineFragment<HasSource>;

    fn shr(self, transform: T) -> Self::Output {
        PipelineFragment {
            builder: self.builder.then(transform),
        }
    }
}

/// Marker type for creating a source start point with `>>`.
pub struct FromSource<S>(pub S);

impl<S: Source + 'static, T: Transform + 'static> std::ops::Shr<T> for FromSource<S> {
    type Output = ChainedTransform<S, T>;

    fn shr(self, transform: T) -> Self::Output {
        ChainedTransform {
            source: self.0,
            transform,
        }
    }
}

/// A source chained with a transform.
pub struct ChainedTransform<S, T> {
    source: S,
    transform: T,
}

impl<S: Source + 'static, T: Transform + 'static, T2: Transform + 'static> std::ops::Shr<T2>
    for ChainedTransform<S, T>
{
    type Output = ChainedTransform2<S, T, T2>;

    fn shr(self, transform2: T2) -> Self::Output {
        ChainedTransform2 {
            source: self.source,
            transform1: self.transform,
            transform2,
        }
    }
}

impl<S: Source + 'static, T: Transform + 'static, Sk: Sink + 'static> std::ops::Shr<ToSink<Sk>>
    for ChainedTransform<S, T>
{
    type Output = BuiltPipeline;

    fn shr(self, sink: ToSink<Sk>) -> Self::Output {
        let pipeline = PipelineBuilder::new()
            .source(self.source)
            .then(self.transform)
            .sink(sink.0)
            .build()
            .expect("failed to build pipeline");

        BuiltPipeline { pipeline }
    }
}

/// A source chained with two transforms.
pub struct ChainedTransform2<S, T1, T2> {
    source: S,
    transform1: T1,
    transform2: T2,
}

impl<S: Source + 'static, T1: Transform + 'static, T2: Transform + 'static, Sk: Sink + 'static>
    std::ops::Shr<ToSink<Sk>> for ChainedTransform2<S, T1, T2>
{
    type Output = BuiltPipeline;

    fn shr(self, sink: ToSink<Sk>) -> Self::Output {
        let pipeline = PipelineBuilder::new()
            .source(self.source)
            .then(self.transform1)
            .then(self.transform2)
            .sink(sink.0)
            .build()
            .expect("failed to build pipeline");

        BuiltPipeline { pipeline }
    }
}

/// Marker type for creating a sink end point with `>>`.
pub struct ToSink<S>(pub S);

/// A fully built pipeline from operator syntax.
pub struct BuiltPipeline {
    pipeline: Pipeline,
}

impl BuiltPipeline {
    /// Run the pipeline to completion.
    pub async fn run(mut self) -> Result<()> {
        self.pipeline.run().await
    }

    /// Get the underlying pipeline.
    pub fn into_pipeline(self) -> Pipeline {
        self.pipeline
    }

    /// Get a reference to the underlying pipeline.
    pub fn pipeline(&self) -> &Pipeline {
        &self.pipeline
    }

    /// Get a mutable reference to the underlying pipeline.
    pub fn pipeline_mut(&mut self) -> &mut Pipeline {
        &mut self.pipeline
    }
}

// ============================================================================
// Convenience functions
// ============================================================================

/// Create a source wrapper for operator syntax.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::pipeline::{from, to};
///
/// let pipeline = (
///     from(VideoTestSrc::new())
///     >> VideoScale::new(...)
///     >> to(FileSink::new("out.yuv"))
/// ).into_pipeline();
/// ```
pub fn from<S: Source>(source: S) -> FromSource<S> {
    FromSource(source)
}

/// Create a sink wrapper for operator syntax.
pub fn to<S: Sink>(sink: S) -> ToSink<S> {
    ToSink(sink)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::element::{ConsumeContext, ProduceContext, ProduceResult};

    struct TestSource {
        count: u32,
        max: u32,
    }

    impl Source for TestSource {
        fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
            if self.count >= self.max {
                return Ok(ProduceResult::Eos);
            }
            let data = self.count.to_le_bytes();
            let output = ctx.output();
            output[..4].copy_from_slice(&data);
            self.count += 1;
            Ok(ProduceResult::Produced(4))
        }
    }

    struct TestTransform;

    impl Transform for TestTransform {
        fn transform(&mut self, buffer: crate::buffer::Buffer) -> Result<crate::element::Output> {
            Ok(crate::element::Output::Single(buffer))
        }
    }

    struct TestSink {
        received: std::sync::Arc<std::sync::atomic::AtomicU32>,
    }

    impl Sink for TestSink {
        fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
            self.received
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn test_builder_basic() {
        let received = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

        let result = PipelineBuilder::new()
            .source(TestSource { count: 0, max: 5 })
            .sink(TestSink {
                received: received.clone(),
            })
            .build();

        assert!(result.is_ok());
        let pipeline = result.unwrap();
        assert_eq!(pipeline.node_count(), 2);
    }

    #[test]
    fn test_builder_with_transform() {
        let received = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

        let result = PipelineBuilder::new()
            .source(TestSource { count: 0, max: 5 })
            .then(TestTransform)
            .sink(TestSink {
                received: received.clone(),
            })
            .build();

        assert!(result.is_ok());
        let pipeline = result.unwrap();
        assert_eq!(pipeline.node_count(), 3);
    }

    #[test]
    fn test_builder_named() {
        let received = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

        let pipeline = PipelineBuilder::new()
            .source_named("my_source", TestSource { count: 0, max: 5 })
            .then_named("my_transform", TestTransform)
            .sink_named("my_sink", TestSink { received })
            .build()
            .unwrap();

        assert!(pipeline.get_node_id("my_source").is_some());
        assert!(pipeline.get_node_id("my_transform").is_some());
        assert!(pipeline.get_node_id("my_sink").is_some());
    }

    #[tokio::test]
    async fn test_builder_run() {
        let received = std::sync::Arc::new(std::sync::atomic::AtomicU32::new(0));

        let result = PipelineBuilder::new()
            .with_new_arena(64, 8)
            .unwrap()
            .source(TestSource { count: 0, max: 5 })
            .sink(TestSink {
                received: received.clone(),
            })
            .build();

        assert!(result.is_ok());
        let mut pipeline = result.unwrap();
        let run_result = pipeline.run().await;
        assert!(run_result.is_ok());
        assert_eq!(
            received.load(std::sync::atomic::Ordering::Relaxed),
            5,
            "should have received 5 buffers"
        );
    }
}
