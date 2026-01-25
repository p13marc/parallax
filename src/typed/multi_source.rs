//! Multi-source operators for combining streams.
//!
//! This module provides operators that combine multiple input streams:
//!
//! - [`merge`]: Interleaves items from multiple sources
//! - [`zip`]: Pairs items from two sources one-to-one
//! - [`join`]: Joins items based on a key function
//! - [`temporal_join`]: Joins items based on timestamps
//!
//! # Examples
//!
//! ```rust,ignore
//! use parallax::typed::{from_iter, merge, zip, pipeline};
//!
//! // Merge two sources
//! let source1 = from_iter(vec![1, 2, 3]);
//! let source2 = from_iter(vec![4, 5, 6]);
//! let merged = merge(source1, source2);
//!
//! // Zip two sources
//! let left = from_iter(vec!["a", "b", "c"]);
//! let right = from_iter(vec![1, 2, 3]);
//! let zipped = zip(left, right); // yields ("a", 1), ("b", 2), ("c", 3)
//! ```

use super::element::TypedSource;
use crate::error::Result;
use crate::temporal::{AlignmentStrategy, JoinWindow, TemporalJoin, Timestamp};
use std::time::Duration;

/// A source that merges two sources by alternating their outputs.
pub struct Merge<A, B>
where
    A: TypedSource,
    B: TypedSource<Output = A::Output>,
{
    left: A,
    right: B,
    use_left: bool,
    left_done: bool,
    right_done: bool,
}

impl<A, B> Merge<A, B>
where
    A: TypedSource,
    B: TypedSource<Output = A::Output>,
{
    /// Create a new merge source.
    pub fn new(left: A, right: B) -> Self {
        Self {
            left,
            right,
            use_left: true,
            left_done: false,
            right_done: false,
        }
    }
}

impl<A, B> TypedSource for Merge<A, B>
where
    A: TypedSource,
    B: TypedSource<Output = A::Output>,
{
    type Output = A::Output;

    fn produce(&mut self) -> Result<Option<Self::Output>> {
        loop {
            if self.left_done && self.right_done {
                return Ok(None);
            }

            if self.use_left && !self.left_done {
                match self.left.produce()? {
                    Some(item) => {
                        self.use_left = false;
                        return Ok(Some(item));
                    }
                    None => {
                        self.left_done = true;
                    }
                }
            }

            if !self.use_left && !self.right_done {
                match self.right.produce()? {
                    Some(item) => {
                        self.use_left = true;
                        return Ok(Some(item));
                    }
                    None => {
                        self.right_done = true;
                    }
                }
            }

            // If one side is done, keep pulling from the other
            if self.left_done && !self.right_done {
                self.use_left = false;
            } else if self.right_done && !self.left_done {
                self.use_left = true;
            }
        }
    }
}

/// Create a merge source from two sources.
pub fn merge<A, B>(left: A, right: B) -> Merge<A, B>
where
    A: TypedSource,
    B: TypedSource<Output = A::Output>,
{
    Merge::new(left, right)
}

/// A source that zips two sources together, producing pairs.
pub struct Zip<A, B>
where
    A: TypedSource,
    B: TypedSource,
{
    left: A,
    right: B,
}

impl<A, B> Zip<A, B>
where
    A: TypedSource,
    B: TypedSource,
{
    /// Create a new zip source.
    pub fn new(left: A, right: B) -> Self {
        Self { left, right }
    }
}

impl<A, B> TypedSource for Zip<A, B>
where
    A: TypedSource,
    B: TypedSource,
{
    type Output = (A::Output, B::Output);

    fn produce(&mut self) -> Result<Option<Self::Output>> {
        match (self.left.produce()?, self.right.produce()?) {
            (Some(a), Some(b)) => Ok(Some((a, b))),
            _ => Ok(None),
        }
    }
}

/// Create a zip source from two sources.
pub fn zip<A, B>(left: A, right: B) -> Zip<A, B>
where
    A: TypedSource,
    B: TypedSource,
{
    Zip::new(left, right)
}

/// A source that joins two sources based on a key function.
pub struct Join<A, B, K, FA, FB>
where
    A: TypedSource,
    B: TypedSource,
    K: Eq + std::hash::Hash,
    FA: Fn(&A::Output) -> K,
    FB: Fn(&B::Output) -> K,
{
    left: A,
    right: B,
    left_key: FA,
    right_key: FB,
    left_buffer: std::collections::HashMap<K, A::Output>,
    right_buffer: std::collections::HashMap<K, B::Output>,
    left_done: bool,
    right_done: bool,
}

impl<A, B, K, FA, FB> Join<A, B, K, FA, FB>
where
    A: TypedSource,
    B: TypedSource,
    K: Eq + std::hash::Hash + Clone,
    A::Output: Clone,
    B::Output: Clone,
    FA: Fn(&A::Output) -> K,
    FB: Fn(&B::Output) -> K,
{
    /// Create a new join source.
    pub fn new(left: A, right: B, left_key: FA, right_key: FB) -> Self {
        Self {
            left,
            right,
            left_key,
            right_key,
            left_buffer: std::collections::HashMap::new(),
            right_buffer: std::collections::HashMap::new(),
            left_done: false,
            right_done: false,
        }
    }
}

impl<A, B, K, FA, FB> TypedSource for Join<A, B, K, FA, FB>
where
    A: TypedSource,
    B: TypedSource,
    K: Eq + std::hash::Hash + Clone + Send + 'static,
    A::Output: Clone + Send + 'static,
    B::Output: Clone + Send + 'static,
    FA: Fn(&A::Output) -> K + Send,
    FB: Fn(&B::Output) -> K + Send,
{
    type Output = (A::Output, B::Output);

    fn produce(&mut self) -> Result<Option<Self::Output>> {
        loop {
            // Try to find a match in buffers
            let matching_key = self
                .left_buffer
                .keys()
                .find(|k| self.right_buffer.contains_key(*k))
                .cloned();

            if let Some(key) = matching_key {
                let left_val = self.left_buffer.remove(&key).unwrap();
                let right_val = self.right_buffer.remove(&key).unwrap();
                return Ok(Some((left_val, right_val)));
            }

            // Both done and no matches
            if self.left_done && self.right_done {
                return Ok(None);
            }

            // Pull from left
            if !self.left_done {
                match self.left.produce()? {
                    Some(item) => {
                        let key = (self.left_key)(&item);
                        if let Some(right_val) = self.right_buffer.remove(&key) {
                            return Ok(Some((item, right_val)));
                        }
                        self.left_buffer.insert(key, item);
                    }
                    None => self.left_done = true,
                }
            }

            // Pull from right
            if !self.right_done {
                match self.right.produce()? {
                    Some(item) => {
                        let key = (self.right_key)(&item);
                        if let Some(left_val) = self.left_buffer.remove(&key) {
                            return Ok(Some((left_val, item)));
                        }
                        self.right_buffer.insert(key, item);
                    }
                    None => self.right_done = true,
                }
            }
        }
    }
}

/// Create a join source that joins two sources based on key functions.
pub fn join<A, B, K, FA, FB>(
    left: A,
    right: B,
    left_key: FA,
    right_key: FB,
) -> Join<A, B, K, FA, FB>
where
    A: TypedSource,
    B: TypedSource,
    K: Eq + std::hash::Hash + Clone,
    A::Output: Clone,
    B::Output: Clone,
    FA: Fn(&A::Output) -> K,
    FB: Fn(&B::Output) -> K,
{
    Join::new(left, right, left_key, right_key)
}

/// A timestamped value for temporal operations.
#[derive(Debug, Clone)]
pub struct Timestamped<T> {
    /// The timestamp.
    pub timestamp: Timestamp,
    /// The value.
    pub value: T,
}

impl<T> Timestamped<T> {
    /// Create a new timestamped value.
    pub fn new(timestamp: Timestamp, value: T) -> Self {
        Self { timestamp, value }
    }

    /// Create a timestamped value from milliseconds.
    pub fn from_millis(millis: u64, value: T) -> Self {
        Self::new(Timestamp::from_millis(millis), value)
    }
}

/// A source that performs temporal joining of two timestamped sources.
pub struct TemporalJoinSource<A, B>
where
    A: TypedSource,
    B: TypedSource,
    A::Output: Clone,
    B::Output: Clone,
{
    left: A,
    right: B,
    joiner: TemporalJoin<A::Output, B::Output>,
    left_done: bool,
    right_done: bool,
    get_left_ts: Box<dyn Fn(&A::Output) -> Timestamp + Send>,
    get_right_ts: Box<dyn Fn(&B::Output) -> Timestamp + Send>,
}

impl<A, B> TemporalJoinSource<A, B>
where
    A: TypedSource,
    B: TypedSource,
    A::Output: Clone,
    B::Output: Clone,
{
    /// Create a new temporal join source.
    pub fn new<FL, FR>(
        left: A,
        right: B,
        window: JoinWindow,
        get_left_ts: FL,
        get_right_ts: FR,
    ) -> Self
    where
        FL: Fn(&A::Output) -> Timestamp + Send + 'static,
        FR: Fn(&B::Output) -> Timestamp + Send + 'static,
    {
        Self {
            left,
            right,
            joiner: TemporalJoin::with_config(window),
            left_done: false,
            right_done: false,
            get_left_ts: Box::new(get_left_ts),
            get_right_ts: Box::new(get_right_ts),
        }
    }
}

impl<A, B> TypedSource for TemporalJoinSource<A, B>
where
    A: TypedSource,
    B: TypedSource,
    A::Output: Clone + Send + 'static,
    B::Output: Clone + Send + 'static,
{
    type Output = (A::Output, B::Output);

    fn produce(&mut self) -> Result<Option<Self::Output>> {
        use crate::temporal::JoinResult;

        loop {
            // Try to emit a match
            if let Some(result) = self.joiner.try_emit() {
                match result {
                    JoinResult::Matched(a, b) => return Ok(Some((a, b))),
                    JoinResult::LeftOnly(_) | JoinResult::RightOnly(_) => {
                        // Skip unmatched items, continue trying
                        continue;
                    }
                    JoinResult::Pending | JoinResult::Dropped => {}
                }
            }

            // Both sources exhausted and no more matches
            if self.left_done && self.right_done && self.joiner.is_empty() {
                return Ok(None);
            }

            // Pull from left
            if !self.left_done {
                match self.left.produce()? {
                    Some(item) => {
                        let ts = (self.get_left_ts)(&item);
                        self.joiner.push_left(ts, item);
                    }
                    None => self.left_done = true,
                }
            }

            // Pull from right
            if !self.right_done {
                match self.right.produce()? {
                    Some(item) => {
                        let ts = (self.get_right_ts)(&item);
                        self.joiner.push_right(ts, item);
                    }
                    None => self.right_done = true,
                }
            }

            // If both are done but joiner still has items, try to emit
            if self.left_done && self.right_done {
                if self.joiner.is_empty() {
                    return Ok(None);
                }
                // One more try to emit remaining matches
                if let Some(result) = self.joiner.try_emit() {
                    if let JoinResult::Matched(a, b) = result {
                        return Ok(Some((a, b)));
                    }
                }
            }
        }
    }
}

/// Create a temporal join source with tolerance-based alignment.
pub fn temporal_join<A, B, FL, FR>(
    left: A,
    right: B,
    tolerance: Duration,
    get_left_ts: FL,
    get_right_ts: FR,
) -> TemporalJoinSource<A, B>
where
    A: TypedSource,
    B: TypedSource,
    A::Output: Clone,
    B::Output: Clone,
    FL: Fn(&A::Output) -> Timestamp + Send + 'static,
    FR: Fn(&B::Output) -> Timestamp + Send + 'static,
{
    let window = JoinWindow::default().with_strategy(AlignmentStrategy::Tolerance(tolerance));
    TemporalJoinSource::new(left, right, window, get_left_ts, get_right_ts)
}

/// Create a temporal join source with custom window configuration.
pub fn temporal_join_with_window<A, B, FL, FR>(
    left: A,
    right: B,
    window: JoinWindow,
    get_left_ts: FL,
    get_right_ts: FR,
) -> TemporalJoinSource<A, B>
where
    A: TypedSource,
    B: TypedSource,
    A::Output: Clone,
    B::Output: Clone,
    FL: Fn(&A::Output) -> Timestamp + Send + 'static,
    FR: Fn(&B::Output) -> Timestamp + Send + 'static,
{
    TemporalJoinSource::new(left, right, window, get_left_ts, get_right_ts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::typed::from_iter;

    #[test]
    fn test_merge_two_sources() {
        let left = from_iter(vec![1, 3, 5]);
        let right = from_iter(vec![2, 4, 6]);
        let mut merged = merge(left, right);

        let mut results = Vec::new();
        while let Ok(Some(item)) = merged.produce() {
            results.push(item);
        }

        // Should alternate
        assert_eq!(results, vec![1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_merge_unequal_lengths() {
        let left = from_iter(vec![1, 2]);
        let right = from_iter(vec![3, 4, 5, 6]);
        let mut merged = merge(left, right);

        let mut results = Vec::new();
        while let Ok(Some(item)) = merged.produce() {
            results.push(item);
        }

        assert_eq!(results.len(), 6);
        assert!(results.contains(&1));
        assert!(results.contains(&6));
    }

    #[test]
    fn test_zip_two_sources() {
        let left = from_iter(vec!["a", "b", "c"]);
        let right = from_iter(vec![1, 2, 3]);
        let mut zipped = zip(left, right);

        let mut results = Vec::new();
        while let Ok(Some(item)) = zipped.produce() {
            results.push(item);
        }

        assert_eq!(results, vec![("a", 1), ("b", 2), ("c", 3)]);
    }

    #[test]
    fn test_zip_unequal_lengths() {
        let left = from_iter(vec![1, 2, 3, 4, 5]);
        let right = from_iter(vec![10, 20]);
        let mut zipped = zip(left, right);

        let mut results = Vec::new();
        while let Ok(Some(item)) = zipped.produce() {
            results.push(item);
        }

        // Stops at shorter length
        assert_eq!(results, vec![(1, 10), (2, 20)]);
    }

    #[test]
    fn test_join_by_key() {
        #[derive(Clone, Debug, PartialEq)]
        struct Left {
            id: i32,
            name: String,
        }
        #[derive(Clone, Debug, PartialEq)]
        struct Right {
            id: i32,
            value: i32,
        }

        let left = from_iter(vec![
            Left {
                id: 1,
                name: "one".to_string(),
            },
            Left {
                id: 2,
                name: "two".to_string(),
            },
            Left {
                id: 3,
                name: "three".to_string(),
            },
        ]);
        let right = from_iter(vec![
            Right { id: 2, value: 200 },
            Right { id: 1, value: 100 },
            Right { id: 4, value: 400 }, // No match
        ]);

        let mut joined = join(left, right, |l| l.id, |r| r.id);

        let mut results = Vec::new();
        while let Ok(Some(item)) = joined.produce() {
            results.push(item);
        }

        assert_eq!(results.len(), 2);
        // Should have matched id=1 and id=2
        assert!(results.iter().any(|(l, r)| l.id == 1 && r.value == 100));
        assert!(results.iter().any(|(l, r)| l.id == 2 && r.value == 200));
    }

    #[test]
    fn test_timestamped() {
        let ts = Timestamped::from_millis(100, "data");
        assert_eq!(ts.timestamp.as_millis(), 100);
        assert_eq!(ts.value, "data");
    }

    #[test]
    fn test_temporal_join_exact() {
        let left = from_iter(vec![
            Timestamped::from_millis(100, "a"),
            Timestamped::from_millis(200, "b"),
            Timestamped::from_millis(300, "c"),
        ]);
        let right = from_iter(vec![
            Timestamped::from_millis(100, 1),
            Timestamped::from_millis(200, 2),
            Timestamped::from_millis(300, 3),
        ]);

        let mut joined = temporal_join(
            left,
            right,
            Duration::from_millis(10),
            |l| l.timestamp,
            |r| r.timestamp,
        );

        let mut results = Vec::new();
        while let Ok(Some((l, r))) = joined.produce() {
            results.push((l.value, r.value));
        }

        assert_eq!(results.len(), 3);
        assert!(results.contains(&("a", 1)));
        assert!(results.contains(&("b", 2)));
        assert!(results.contains(&("c", 3)));
    }

    #[test]
    fn test_temporal_join_with_tolerance() {
        let left = from_iter(vec![
            Timestamped::from_millis(100, "a"),
            Timestamped::from_millis(200, "b"),
        ]);
        let right = from_iter(vec![
            Timestamped::from_millis(95, 1),  // Within 10ms of 100
            Timestamped::from_millis(205, 2), // Within 10ms of 200
        ]);

        let mut joined = temporal_join(
            left,
            right,
            Duration::from_millis(10),
            |l| l.timestamp,
            |r| r.timestamp,
        );

        let mut results = Vec::new();
        while let Ok(Some((l, r))) = joined.produce() {
            results.push((l.value, r.value));
        }

        assert_eq!(results.len(), 2);
    }
}
