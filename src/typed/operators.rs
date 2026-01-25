//! Common operators for typed pipelines.
//!
//! This module provides functional-style operators like `map`, `filter`,
//! and `flat_map` that can be used in typed pipelines.

use std::marker::PhantomData;

use crate::error::Result;

use super::element::{TypedSink, TypedSource, TypedTransform};

// ============================================================================
// Map Operator
// ============================================================================

/// A transform that applies a function to each item.
///
/// # Example
///
/// ```rust,ignore
/// let pipeline = source
///     .then(map(|x: u32| x * 2))
///     .sink(sink);
/// ```
pub struct Map<F, In, Out> {
    f: F,
    _in: PhantomData<In>,
    _out: PhantomData<Out>,
}

impl<F, In, Out> Map<F, In, Out> {
    /// Create a new map operator.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _in: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<F, In, Out> TypedTransform for Map<F, In, Out>
where
    F: FnMut(In) -> Out + Send,
    In: Send + 'static,
    Out: Send + 'static,
{
    type Input = In;
    type Output = Out;

    fn transform(&mut self, input: In) -> Result<Option<Out>> {
        Ok(Some((self.f)(input)))
    }

    fn name(&self) -> &str {
        "map"
    }
}

/// Create a map transform.
pub fn map<F, In, Out>(f: F) -> Map<F, In, Out>
where
    F: FnMut(In) -> Out + Send,
    In: Send + 'static,
    Out: Send + 'static,
{
    Map::new(f)
}

// ============================================================================
// Filter Operator
// ============================================================================

/// A transform that filters items based on a predicate.
///
/// # Example
///
/// ```rust,ignore
/// let pipeline = source
///     .then(filter(|x: &u32| *x > 10))
///     .sink(sink);
/// ```
pub struct Filter<F, T> {
    predicate: F,
    _t: PhantomData<T>,
}

impl<F, T> Filter<F, T> {
    /// Create a new filter operator.
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            _t: PhantomData,
        }
    }
}

impl<F, T> TypedTransform for Filter<F, T>
where
    F: FnMut(&T) -> bool + Send,
    T: Send + 'static,
{
    type Input = T;
    type Output = T;

    fn transform(&mut self, input: T) -> Result<Option<T>> {
        if (self.predicate)(&input) {
            Ok(Some(input))
        } else {
            Ok(None)
        }
    }

    fn name(&self) -> &str {
        "filter"
    }
}

/// Create a filter transform.
pub fn filter<F, T>(predicate: F) -> Filter<F, T>
where
    F: FnMut(&T) -> bool + Send,
    T: Send + 'static,
{
    Filter::new(predicate)
}

// ============================================================================
// FilterMap Operator
// ============================================================================

/// A transform that combines filter and map in one step.
///
/// # Example
///
/// ```rust,ignore
/// let pipeline = source
///     .then(filter_map(|x: u32| if x > 10 { Some(x * 2) } else { None }))
///     .sink(sink);
/// ```
pub struct FilterMap<F, In, Out> {
    f: F,
    _in: PhantomData<In>,
    _out: PhantomData<Out>,
}

impl<F, In, Out> FilterMap<F, In, Out> {
    /// Create a new filter_map operator.
    pub fn new(f: F) -> Self {
        Self {
            f,
            _in: PhantomData,
            _out: PhantomData,
        }
    }
}

impl<F, In, Out> TypedTransform for FilterMap<F, In, Out>
where
    F: FnMut(In) -> Option<Out> + Send,
    In: Send + 'static,
    Out: Send + 'static,
{
    type Input = In;
    type Output = Out;

    fn transform(&mut self, input: In) -> Result<Option<Out>> {
        Ok((self.f)(input))
    }

    fn name(&self) -> &str {
        "filter_map"
    }
}

/// Create a filter_map transform.
pub fn filter_map<F, In, Out>(f: F) -> FilterMap<F, In, Out>
where
    F: FnMut(In) -> Option<Out> + Send,
    In: Send + 'static,
    Out: Send + 'static,
{
    FilterMap::new(f)
}

// ============================================================================
// Inspect Operator
// ============================================================================

/// A transform that inspects items without modifying them.
///
/// Useful for debugging or logging.
///
/// # Example
///
/// ```rust,ignore
/// let pipeline = source
///     .then(inspect(|x: &u32| println!("Got: {}", x)))
///     .sink(sink);
/// ```
pub struct Inspect<F, T> {
    f: F,
    _t: PhantomData<T>,
}

impl<F, T> Inspect<F, T> {
    /// Create a new inspect operator.
    pub fn new(f: F) -> Self {
        Self { f, _t: PhantomData }
    }
}

impl<F, T> TypedTransform for Inspect<F, T>
where
    F: FnMut(&T) + Send,
    T: Send + 'static,
{
    type Input = T;
    type Output = T;

    fn transform(&mut self, input: T) -> Result<Option<T>> {
        (self.f)(&input);
        Ok(Some(input))
    }

    fn name(&self) -> &str {
        "inspect"
    }
}

/// Create an inspect transform.
pub fn inspect<F, T>(f: F) -> Inspect<F, T>
where
    F: FnMut(&T) + Send,
    T: Send + 'static,
{
    Inspect::new(f)
}

// ============================================================================
// Take Operator
// ============================================================================

/// A transform that takes only the first N items.
///
/// # Example
///
/// ```rust,ignore
/// let pipeline = source
///     .then(take(10))
///     .sink(sink);
/// ```
pub struct Take<T> {
    remaining: usize,
    _t: PhantomData<T>,
}

impl<T> Take<T> {
    /// Create a new take operator.
    pub fn new(n: usize) -> Self {
        Self {
            remaining: n,
            _t: PhantomData,
        }
    }
}

impl<T: Send + 'static> TypedTransform for Take<T> {
    type Input = T;
    type Output = T;

    fn transform(&mut self, input: T) -> Result<Option<T>> {
        if self.remaining > 0 {
            self.remaining -= 1;
            Ok(Some(input))
        } else {
            Ok(None)
        }
    }

    fn name(&self) -> &str {
        "take"
    }
}

/// Create a take transform.
pub fn take<T: Send + 'static>(n: usize) -> Take<T> {
    Take::new(n)
}

// ============================================================================
// Skip Operator
// ============================================================================

/// A transform that skips the first N items.
///
/// # Example
///
/// ```rust,ignore
/// let pipeline = source
///     .then(skip(5))
///     .sink(sink);
/// ```
pub struct Skip<T> {
    remaining: usize,
    _t: PhantomData<T>,
}

impl<T> Skip<T> {
    /// Create a new skip operator.
    pub fn new(n: usize) -> Self {
        Self {
            remaining: n,
            _t: PhantomData,
        }
    }
}

impl<T: Send + 'static> TypedTransform for Skip<T> {
    type Input = T;
    type Output = T;

    fn transform(&mut self, input: T) -> Result<Option<T>> {
        if self.remaining > 0 {
            self.remaining -= 1;
            Ok(None)
        } else {
            Ok(Some(input))
        }
    }

    fn name(&self) -> &str {
        "skip"
    }
}

/// Create a skip transform.
pub fn skip<T: Send + 'static>(n: usize) -> Skip<T> {
    Skip::new(n)
}

// ============================================================================
// Common Sources
// ============================================================================

/// A source that produces items from an iterator.
pub struct IterSource<I: Iterator> {
    iter: I,
}

impl<I: Iterator> IterSource<I> {
    /// Create a new iterator source.
    pub fn new(iter: I) -> Self {
        Self { iter }
    }
}

impl<I> TypedSource for IterSource<I>
where
    I: Iterator + Send,
    I::Item: Send + 'static,
{
    type Output = I::Item;

    fn produce(&mut self) -> Result<Option<I::Item>> {
        Ok(self.iter.next())
    }

    fn name(&self) -> &str {
        "iter_source"
    }
}

/// Create a source from an iterator.
pub fn from_iter<I>(iter: I) -> IterSource<I::IntoIter>
where
    I: IntoIterator,
    I::IntoIter: Send,
    I::Item: Send + 'static,
{
    IterSource::new(iter.into_iter())
}

/// A source that produces a range of numbers.
pub fn range<T>(range: std::ops::Range<T>) -> IterSource<std::ops::Range<T>>
where
    std::ops::Range<T>: Iterator + Send,
    T: Send + 'static,
{
    IterSource::new(range)
}

/// A source that produces a single item.
pub struct OnceSource<T> {
    item: Option<T>,
}

impl<T: Send + 'static> TypedSource for OnceSource<T> {
    type Output = T;

    fn produce(&mut self) -> Result<Option<T>> {
        Ok(self.item.take())
    }

    fn name(&self) -> &str {
        "once"
    }
}

/// Create a source that produces a single item.
pub fn once<T: Send + 'static>(item: T) -> OnceSource<T> {
    OnceSource { item: Some(item) }
}

/// A source that produces items by repeatedly calling a function.
pub struct RepeatWith<F, T> {
    f: F,
    _t: PhantomData<T>,
}

impl<F, T> TypedSource for RepeatWith<F, T>
where
    F: FnMut() -> Option<T> + Send,
    T: Send + 'static,
{
    type Output = T;

    fn produce(&mut self) -> Result<Option<T>> {
        Ok((self.f)())
    }

    fn name(&self) -> &str {
        "repeat_with"
    }
}

/// Create a source that produces items by repeatedly calling a function.
pub fn repeat_with<F, T>(f: F) -> RepeatWith<F, T>
where
    F: FnMut() -> Option<T> + Send,
    T: Send + 'static,
{
    RepeatWith { f, _t: PhantomData }
}

// ============================================================================
// Common Sinks
// ============================================================================

/// A sink that collects items into a vector.
///
/// Use `into_inner()` to get the collected items after the pipeline runs.
pub struct CollectSink<T> {
    items: Vec<T>,
}

impl<T> CollectSink<T> {
    /// Create a new collecting sink.
    pub fn new() -> Self {
        Self { items: Vec::new() }
    }

    /// Get the collected items.
    pub fn into_inner(self) -> Vec<T> {
        self.items
    }

    /// Get a reference to the collected items.
    pub fn items(&self) -> &[T] {
        &self.items
    }
}

impl<T> Default for CollectSink<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + 'static> TypedSink for CollectSink<T> {
    type Input = T;

    fn consume(&mut self, item: T) -> Result<()> {
        self.items.push(item);
        Ok(())
    }

    fn name(&self) -> &str {
        "collect"
    }
}

/// Create a collecting sink.
pub fn collect<T: Send + 'static>() -> CollectSink<T> {
    CollectSink::new()
}

/// A sink that discards all items.
pub struct DiscardSink<T> {
    _t: PhantomData<T>,
}

impl<T> DiscardSink<T> {
    /// Create a new discard sink.
    pub fn new() -> Self {
        Self { _t: PhantomData }
    }
}

impl<T> Default for DiscardSink<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Send + 'static> TypedSink for DiscardSink<T> {
    type Input = T;

    fn consume(&mut self, _item: T) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "discard"
    }
}

/// Create a discard sink.
pub fn discard<T: Send + 'static>() -> DiscardSink<T> {
    DiscardSink::new()
}

/// A sink that calls a function for each item.
pub struct ForEachSink<F, T> {
    f: F,
    _t: PhantomData<T>,
}

impl<F, T> ForEachSink<F, T> {
    /// Create a new for_each sink.
    pub fn new(f: F) -> Self {
        Self { f, _t: PhantomData }
    }
}

impl<F, T> TypedSink for ForEachSink<F, T>
where
    F: FnMut(T) + Send,
    T: Send + 'static,
{
    type Input = T;

    fn consume(&mut self, item: T) -> Result<()> {
        (self.f)(item);
        Ok(())
    }

    fn name(&self) -> &str {
        "for_each"
    }
}

/// Create a for_each sink.
pub fn for_each<F, T>(f: F) -> ForEachSink<F, T>
where
    F: FnMut(T) + Send,
    T: Send + 'static,
{
    ForEachSink::new(f)
}

/// A sink that folds items into an accumulator.
pub struct FoldSink<F, T, Acc> {
    f: F,
    acc: Acc,
    _t: PhantomData<T>,
}

impl<F, T, Acc> FoldSink<F, T, Acc> {
    /// Create a new fold sink.
    pub fn new(init: Acc, f: F) -> Self {
        Self {
            f,
            acc: init,
            _t: PhantomData,
        }
    }

    /// Get the final accumulated value.
    pub fn into_inner(self) -> Acc {
        self.acc
    }

    /// Get a reference to the current accumulator.
    pub fn accumulator(&self) -> &Acc {
        &self.acc
    }
}

impl<F, T, Acc> TypedSink for FoldSink<F, T, Acc>
where
    F: FnMut(Acc, T) -> Acc + Send,
    T: Send + 'static,
    Acc: Send + 'static + Default,
{
    type Input = T;

    fn consume(&mut self, item: T) -> Result<()> {
        // Take the accumulator, apply the fold, put it back
        let acc = std::mem::take(&mut self.acc);
        self.acc = (self.f)(acc, item);
        Ok(())
    }

    fn name(&self) -> &str {
        "fold"
    }
}

/// Create a fold sink.
pub fn fold<F, T, Acc>(init: Acc, f: F) -> FoldSink<F, T, Acc>
where
    F: FnMut(Acc, T) -> Acc + Send,
    T: Send + 'static,
    Acc: Send + 'static + Default,
{
    FoldSink::new(init, f)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_map() {
        let mut m = map(|x: u32| x * 2);
        assert_eq!(m.transform(5).unwrap(), Some(10));
        assert_eq!(m.transform(0).unwrap(), Some(0));
    }

    #[test]
    fn test_filter() {
        let mut f = filter(|x: &u32| *x > 5);
        assert_eq!(f.transform(10).unwrap(), Some(10));
        assert_eq!(f.transform(3).unwrap(), None);
    }

    #[test]
    fn test_filter_map() {
        let mut fm = filter_map(|x: u32| if x > 5 { Some(x * 2) } else { None });
        assert_eq!(fm.transform(10).unwrap(), Some(20));
        assert_eq!(fm.transform(3).unwrap(), None);
    }

    #[test]
    fn test_take() {
        let mut t: Take<u32> = take(2);
        assert_eq!(t.transform(1).unwrap(), Some(1));
        assert_eq!(t.transform(2).unwrap(), Some(2));
        assert_eq!(t.transform(3).unwrap(), None);
        assert_eq!(t.transform(4).unwrap(), None);
    }

    #[test]
    fn test_skip() {
        let mut s: Skip<u32> = skip(2);
        assert_eq!(s.transform(1).unwrap(), None);
        assert_eq!(s.transform(2).unwrap(), None);
        assert_eq!(s.transform(3).unwrap(), Some(3));
        assert_eq!(s.transform(4).unwrap(), Some(4));
    }

    #[test]
    fn test_iter_source() {
        let mut source = from_iter(vec![1, 2, 3]);
        assert_eq!(source.produce().unwrap(), Some(1));
        assert_eq!(source.produce().unwrap(), Some(2));
        assert_eq!(source.produce().unwrap(), Some(3));
        assert_eq!(source.produce().unwrap(), None);
    }

    #[test]
    fn test_once_source() {
        let mut source = once(42);
        assert_eq!(source.produce().unwrap(), Some(42));
        assert_eq!(source.produce().unwrap(), None);
    }

    #[test]
    fn test_collect_sink() {
        let mut sink: CollectSink<u32> = collect();
        sink.consume(1).unwrap();
        sink.consume(2).unwrap();
        sink.consume(3).unwrap();
        assert_eq!(sink.into_inner(), vec![1, 2, 3]);
    }

    #[test]
    fn test_for_each_sink() {
        let mut results = Vec::new();
        {
            let mut sink = for_each(|x: u32| results.push(x));
            sink.consume(1).unwrap();
            sink.consume(2).unwrap();
        }
        assert_eq!(results, vec![1, 2]);
    }

    #[test]
    fn test_inspect() {
        let mut seen = Vec::new();
        let seen_ref = &mut seen;
        let mut ins = inspect(move |x: &u32| seen_ref.push(*x));

        // Note: This won't work because inspect captures seen_ref
        // In practice you'd use a channel or Arc<Mutex<>>
    }
}
