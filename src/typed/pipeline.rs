//! Typed pipeline builder with compile-time type validation.
//!
//! The typed pipeline API uses Rust's type system to validate
//! that element connections are compatible at compile time.

use crate::error::Result;

use super::element::{TypedSink, TypedSource, TypedTransform};

// ============================================================================
// Transform Chain Types
// ============================================================================

/// Identity transform chain (no-op).
///
/// Used as the base case when there are no transforms in the pipeline.
pub struct Identity;

/// A chain of transforms that applies transforms in order (first added = first applied).
///
/// The chain is built as `Chain<First, Chain<Second, Identity>>` and applies
/// `First` then `Second`.
pub struct Chain<T, Tail>
where
    T: TypedTransform,
{
    transform: T,
    tail: Tail,
}

impl<T, Tail> Chain<T, Tail>
where
    T: TypedTransform,
{
    /// Create a new chain with this transform at the head.
    pub fn new(transform: T, tail: Tail) -> Self {
        Self { transform, tail }
    }

    /// Append a transform to the end of this chain.
    pub fn append<T2>(self, next: T2) -> Chain<T, Tail::Appended>
    where
        T2: TypedTransform,
        Tail: AppendTransform<T2>,
    {
        Chain {
            transform: self.transform,
            tail: self.tail.append(next),
        }
    }
}

/// Trait for appending a transform to a chain.
pub trait AppendTransform<T: TypedTransform>: Sized {
    /// The resulting chain type after appending.
    type Appended;

    /// Append the transform to this chain.
    fn append(self, transform: T) -> Self::Appended;
}

// Appending to Identity creates a single-element chain
impl<T: TypedTransform> AppendTransform<T> for Identity {
    type Appended = Chain<T, Identity>;

    fn append(self, transform: T) -> Self::Appended {
        Chain::new(transform, Identity)
    }
}

// Appending to a Chain recursively appends to the tail
impl<Head, Tail, T> AppendTransform<T> for Chain<Head, Tail>
where
    Head: TypedTransform,
    Tail: AppendTransform<T>,
    T: TypedTransform,
{
    type Appended = Chain<Head, Tail::Appended>;

    fn append(self, transform: T) -> Self::Appended {
        Chain {
            transform: self.transform,
            tail: self.tail.append(transform),
        }
    }
}

/// Trait for applying a transform chain to an input.
pub trait TransformChain<In>: Send {
    /// The output type after applying all transforms.
    type Output: Send + 'static;

    /// Apply all transforms in the chain.
    fn apply(&mut self, input: In) -> Result<Option<Self::Output>>;
}

// Identity just passes through
impl<In: Send + 'static> TransformChain<In> for Identity {
    type Output = In;

    fn apply(&mut self, input: In) -> Result<Option<In>> {
        Ok(Some(input))
    }
}

// Chain applies head first, then tail
impl<T, Tail> TransformChain<T::Input> for Chain<T, Tail>
where
    T: TypedTransform,
    Tail: TransformChain<T::Output>,
{
    type Output = Tail::Output;

    fn apply(&mut self, input: T::Input) -> Result<Option<Self::Output>> {
        match self.transform.transform(input)? {
            Some(output) => self.tail.apply(output),
            None => Ok(None),
        }
    }
}

// ============================================================================
// Pipeline Builder
// ============================================================================

/// A pipeline with a source.
pub struct PipelineWithSource<S: TypedSource> {
    source: S,
}

impl<S: TypedSource + 'static> PipelineWithSource<S> {
    /// Create a new pipeline with the given source.
    pub fn new(source: S) -> Self {
        Self { source }
    }

    /// Add a transform to the pipeline.
    pub fn then<T>(self, transform: T) -> PipelineWithTransforms<S, Chain<T, Identity>>
    where
        T: TypedTransform<Input = S::Output>,
    {
        PipelineWithTransforms {
            source: self.source,
            chain: Chain::new(transform, Identity),
        }
    }

    /// Terminate with a sink (no transforms).
    pub fn sink<K>(self, sink: K) -> RunnablePipeline<S, Identity, K>
    where
        K: TypedSink<Input = S::Output>,
    {
        RunnablePipeline {
            source: self.source,
            chain: Identity,
            sink,
        }
    }
}

/// A pipeline with source and transforms.
///
/// Type parameters:
/// - `S`: Source type
/// - `C`: Chain of transforms
pub struct PipelineWithTransforms<S, C>
where
    S: TypedSource,
{
    source: S,
    chain: C,
}

impl<S, C> PipelineWithTransforms<S, C>
where
    S: TypedSource + 'static,
    C: TransformChain<S::Output>,
{
    /// Add another transform to the pipeline.
    ///
    /// The new transform's input must match the current output type.
    pub fn then<T>(self, transform: T) -> PipelineWithTransforms<S, C::Appended>
    where
        T: TypedTransform<Input = C::Output>,
        C: AppendTransform<T>,
    {
        PipelineWithTransforms {
            source: self.source,
            chain: self.chain.append(transform),
        }
    }

    /// Terminate with a sink.
    pub fn sink<K>(self, sink: K) -> RunnablePipeline<S, C, K>
    where
        K: TypedSink<Input = C::Output>,
    {
        RunnablePipeline {
            source: self.source,
            chain: self.chain,
            sink,
        }
    }
}

/// A complete, runnable pipeline.
pub struct RunnablePipeline<S, C, K>
where
    S: TypedSource,
    K: TypedSink,
{
    source: S,
    chain: C,
    sink: K,
}

impl<S, C, K> RunnablePipeline<S, C, K>
where
    S: TypedSource,
    C: TransformChain<S::Output, Output = K::Input>,
    K: TypedSink,
{
    /// Run the pipeline to completion.
    pub fn run(mut self) -> Result<K> {
        while let Some(item) = self.source.produce()? {
            if let Some(output) = self.chain.apply(item)? {
                self.sink.consume(output)?;
            }
        }
        Ok(self.sink)
    }

    /// Run the pipeline asynchronously.
    ///
    /// This wraps the synchronous execution in a blocking task.
    pub async fn run_async(self) -> Result<K>
    where
        S: Send + 'static,
        C: Send + 'static,
        K: Send + 'static,
    {
        tokio::task::spawn_blocking(|| self.run())
            .await
            .map_err(|e| crate::error::Error::InvalidSegment(format!("join error: {}", e)))?
    }
}

// ============================================================================
// Shr (>>) Operator for Pipeline Building
// ============================================================================

/// Implement >> for source >> transform
impl<S, T> std::ops::Shr<T> for PipelineWithSource<S>
where
    S: TypedSource + 'static,
    T: TypedTransform<Input = S::Output>,
{
    type Output = PipelineWithTransforms<S, Chain<T, Identity>>;

    fn shr(self, transform: T) -> Self::Output {
        self.then(transform)
    }
}

/// Implement >> for transforms >> transform
impl<S, C, T> std::ops::Shr<T> for PipelineWithTransforms<S, C>
where
    S: TypedSource + 'static,
    C: TransformChain<S::Output> + AppendTransform<T>,
    T: TypedTransform<Input = C::Output>,
{
    type Output = PipelineWithTransforms<S, C::Appended>;

    fn shr(self, transform: T) -> Self::Output {
        self.then(transform)
    }
}

// ============================================================================
// Convenience function
// ============================================================================

/// Create a pipeline from a source.
pub fn pipeline<S: TypedSource + 'static>(source: S) -> PipelineWithSource<S> {
    PipelineWithSource::new(source)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::operators::*;

    #[test]
    fn test_simple_pipeline() {
        let source = from_iter(vec![1u32, 2, 3, 4, 5]);
        let sink = pipeline(source).sink(collect::<u32>());

        let result = sink.run().unwrap();
        assert_eq!(result.into_inner(), vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_pipeline_with_map() {
        let source = from_iter(vec![1u32, 2, 3]);
        let sink = pipeline(source)
            .then(map(|x: u32| x * 2))
            .sink(collect::<u32>());

        let result = sink.run().unwrap();
        assert_eq!(result.into_inner(), vec![2, 4, 6]);
    }

    #[test]
    fn test_pipeline_with_filter() {
        let source = from_iter(vec![1u32, 2, 3, 4, 5]);
        let sink = pipeline(source)
            .then(filter(|x: &u32| *x % 2 == 0))
            .sink(collect::<u32>());

        let result = sink.run().unwrap();
        assert_eq!(result.into_inner(), vec![2, 4]);
    }

    #[test]
    fn test_pipeline_with_multiple_transforms() {
        let source = from_iter(vec![1u32, 2, 3, 4, 5, 6]);
        let sink = pipeline(source)
            .then(filter(|x: &u32| *x % 2 == 0))
            .then(map(|x: u32| x * 10))
            .sink(collect::<u32>());

        let result = sink.run().unwrap();
        // Filter keeps 2, 4, 6 -> map produces 20, 40, 60
        assert_eq!(result.into_inner(), vec![20, 40, 60]);
    }

    #[test]
    fn test_shr_operator() {
        let source = from_iter(vec![1u32, 2, 3]);
        let pipe = pipeline(source) >> map(|x: u32| x + 1);
        let result = pipe.sink(collect()).run().unwrap();
        assert_eq!(result.into_inner(), vec![2, 3, 4]);
    }

    #[test]
    fn test_shr_operator_chained() {
        let source = from_iter(vec![1u32, 2, 3, 4, 5, 6]);
        let pipe = pipeline(source) >> filter(|x: &u32| *x % 2 == 0) >> map(|x: u32| x * 10);
        let result = pipe.sink(collect()).run().unwrap();
        assert_eq!(result.into_inner(), vec![20, 40, 60]);
    }
}
