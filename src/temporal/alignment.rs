//! Temporal alignment for multi-source joins.
//!
//! This module provides utilities for joining multiple streams based on
//! timestamp alignment. This is essential for sensor fusion, multi-camera
//! systems, and other applications that need to correlate data in time.

use super::timestamp::Timestamp;
use std::collections::VecDeque;
use std::time::Duration;

/// Strategy for aligning buffers from multiple sources.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlignmentStrategy {
    /// Match buffers with exactly equal timestamps.
    Exact,
    /// Match buffers within a time tolerance.
    Tolerance(Duration),
    /// Match the nearest buffer within a window.
    Nearest(Duration),
    /// Interpolate between buffers (for numeric data).
    Interpolate(Duration),
}

impl Default for AlignmentStrategy {
    fn default() -> Self {
        AlignmentStrategy::Tolerance(Duration::from_millis(10))
    }
}

/// Configuration for a temporal join window.
#[derive(Debug, Clone)]
pub struct JoinWindow {
    /// Maximum time to wait for matching buffers.
    pub max_delay: Duration,
    /// Strategy for matching buffers.
    pub strategy: AlignmentStrategy,
    /// Maximum number of buffered items per source.
    pub max_buffer_size: usize,
}

impl Default for JoinWindow {
    fn default() -> Self {
        Self {
            max_delay: Duration::from_millis(100),
            strategy: AlignmentStrategy::default(),
            max_buffer_size: 64,
        }
    }
}

impl JoinWindow {
    /// Create a new join window with the given max delay.
    pub fn with_max_delay(max_delay: Duration) -> Self {
        Self {
            max_delay,
            ..Default::default()
        }
    }

    /// Set the alignment strategy.
    pub fn with_strategy(mut self, strategy: AlignmentStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set the maximum buffer size.
    pub fn with_max_buffer_size(mut self, size: usize) -> Self {
        self.max_buffer_size = size;
        self
    }
}

/// A timestamped item in a buffer.
#[derive(Debug, Clone)]
pub struct TimestampedItem<T> {
    /// The timestamp of the item.
    pub timestamp: Timestamp,
    /// The item data.
    pub data: T,
}

impl<T> TimestampedItem<T> {
    /// Create a new timestamped item.
    pub fn new(timestamp: Timestamp, data: T) -> Self {
        Self { timestamp, data }
    }
}

/// Result of a temporal join operation.
#[derive(Debug, Clone)]
pub enum JoinResult<A, B> {
    /// Both streams have matching data.
    Matched(A, B),
    /// Only left stream has data (right is missing or late).
    LeftOnly(A),
    /// Only right stream has data (left is missing or late).
    RightOnly(B),
    /// No data available yet (need more input).
    Pending,
    /// Data was dropped due to being too old.
    Dropped,
}

/// A temporal join operator that aligns two streams by timestamp.
///
/// Buffers incoming data from both streams and emits matched pairs
/// when timestamps align according to the configured strategy.
pub struct TemporalJoin<A, B> {
    /// Buffer for left stream.
    left_buffer: VecDeque<TimestampedItem<A>>,
    /// Buffer for right stream.
    right_buffer: VecDeque<TimestampedItem<B>>,
    /// Join configuration.
    config: JoinWindow,
    /// Current watermark (oldest timestamp we're still waiting for).
    watermark: Option<Timestamp>,
}

impl<A, B> TemporalJoin<A, B>
where
    A: Clone,
    B: Clone,
{
    /// Create a new temporal join with default configuration.
    pub fn new() -> Self {
        Self::with_config(JoinWindow::default())
    }

    /// Create a temporal join with custom configuration.
    pub fn with_config(config: JoinWindow) -> Self {
        Self {
            left_buffer: VecDeque::new(),
            right_buffer: VecDeque::new(),
            config,
            watermark: None,
        }
    }

    /// Push a left item into the join.
    pub fn push_left(&mut self, timestamp: Timestamp, data: A) {
        if self.left_buffer.len() >= self.config.max_buffer_size {
            self.left_buffer.pop_front();
        }
        self.left_buffer
            .push_back(TimestampedItem::new(timestamp, data));
        self.update_watermark(timestamp);
    }

    /// Push a right item into the join.
    pub fn push_right(&mut self, timestamp: Timestamp, data: B) {
        if self.right_buffer.len() >= self.config.max_buffer_size {
            self.right_buffer.pop_front();
        }
        self.right_buffer
            .push_back(TimestampedItem::new(timestamp, data));
        self.update_watermark(timestamp);
    }

    /// Update the watermark based on new timestamp.
    fn update_watermark(&mut self, ts: Timestamp) {
        let new_watermark = ts.saturating_sub(self.config.max_delay);
        match self.watermark {
            Some(current) if new_watermark > current => {
                self.watermark = Some(new_watermark);
            }
            None => {
                self.watermark = Some(new_watermark);
            }
            _ => {}
        }
    }

    /// Try to emit a matched pair.
    pub fn try_emit(&mut self) -> Option<JoinResult<A, B>> {
        // Clean up old items past the watermark
        self.cleanup_old_items();

        if self.left_buffer.is_empty() && self.right_buffer.is_empty() {
            return None;
        }

        match self.config.strategy {
            AlignmentStrategy::Exact => self.try_emit_exact(),
            AlignmentStrategy::Tolerance(tol) => self.try_emit_tolerance(tol),
            AlignmentStrategy::Nearest(window) => self.try_emit_nearest(window),
            AlignmentStrategy::Interpolate(_) => {
                // Interpolation is more complex and typically needs numeric types
                // For now, fall back to nearest
                self.try_emit_nearest(Duration::from_millis(10))
            }
        }
    }

    /// Remove items older than the watermark.
    fn cleanup_old_items(&mut self) {
        if let Some(watermark) = self.watermark {
            while let Some(front) = self.left_buffer.front() {
                if front.timestamp < watermark {
                    self.left_buffer.pop_front();
                } else {
                    break;
                }
            }
            while let Some(front) = self.right_buffer.front() {
                if front.timestamp < watermark {
                    self.right_buffer.pop_front();
                } else {
                    break;
                }
            }
        }
    }

    /// Try to emit with exact matching.
    fn try_emit_exact(&mut self) -> Option<JoinResult<A, B>> {
        let left_ts = self.left_buffer.front()?.timestamp;
        let right_ts = self.right_buffer.front()?.timestamp;

        if left_ts == right_ts {
            let left = self.left_buffer.pop_front()?.data;
            let right = self.right_buffer.pop_front()?.data;
            Some(JoinResult::Matched(left, right))
        } else if left_ts < right_ts {
            let left = self.left_buffer.pop_front()?.data;
            Some(JoinResult::LeftOnly(left))
        } else {
            let right = self.right_buffer.pop_front()?.data;
            Some(JoinResult::RightOnly(right))
        }
    }

    /// Try to emit with tolerance matching.
    fn try_emit_tolerance(&mut self, tolerance: Duration) -> Option<JoinResult<A, B>> {
        let left_item = self.left_buffer.front()?;
        let left_ts = left_item.timestamp;

        // Find a matching right item within tolerance
        for (i, right_item) in self.right_buffer.iter().enumerate() {
            if left_ts.within_tolerance(&right_item.timestamp, tolerance) {
                let left = self.left_buffer.pop_front()?.data;
                let right = self.right_buffer.remove(i)?.data;
                return Some(JoinResult::Matched(left, right));
            }
        }

        // Check if left is too old to ever match
        if let Some(right_front) = self.right_buffer.front() {
            if left_ts + tolerance < right_front.timestamp {
                let left = self.left_buffer.pop_front()?.data;
                return Some(JoinResult::LeftOnly(left));
            }
        }

        None
    }

    /// Try to emit with nearest matching within a window.
    fn try_emit_nearest(&mut self, window: Duration) -> Option<JoinResult<A, B>> {
        let left_item = self.left_buffer.front()?;
        let left_ts = left_item.timestamp;

        // Find the nearest right item within the window
        let mut best_match: Option<(usize, Duration)> = None;
        for (i, right_item) in self.right_buffer.iter().enumerate() {
            let diff = left_ts.abs_diff(&right_item.timestamp);
            if diff <= window {
                match best_match {
                    Some((_, best_diff)) if diff < best_diff => {
                        best_match = Some((i, diff));
                    }
                    None => {
                        best_match = Some((i, diff));
                    }
                    _ => {}
                }
            }
        }

        if let Some((idx, _)) = best_match {
            let left = self.left_buffer.pop_front()?.data;
            let right = self.right_buffer.remove(idx)?.data;
            return Some(JoinResult::Matched(left, right));
        }

        // Check if left is too old to ever match
        if let Some(right_front) = self.right_buffer.front() {
            if left_ts + window < right_front.timestamp {
                let left = self.left_buffer.pop_front()?.data;
                return Some(JoinResult::LeftOnly(left));
            }
        }

        None
    }

    /// Get the number of buffered left items.
    pub fn left_len(&self) -> usize {
        self.left_buffer.len()
    }

    /// Get the number of buffered right items.
    pub fn right_len(&self) -> usize {
        self.right_buffer.len()
    }

    /// Check if both buffers are empty.
    pub fn is_empty(&self) -> bool {
        self.left_buffer.is_empty() && self.right_buffer.is_empty()
    }

    /// Clear all buffered items.
    pub fn clear(&mut self) {
        self.left_buffer.clear();
        self.right_buffer.clear();
        self.watermark = None;
    }
}

impl<A: Clone, B: Clone> Default for TemporalJoin<A, B> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_join_window_default() {
        let window = JoinWindow::default();
        assert_eq!(window.max_delay, Duration::from_millis(100));
        assert_eq!(window.max_buffer_size, 64);
    }

    #[test]
    fn test_join_window_builder() {
        let window = JoinWindow::with_max_delay(Duration::from_millis(50))
            .with_strategy(AlignmentStrategy::Exact)
            .with_max_buffer_size(32);

        assert_eq!(window.max_delay, Duration::from_millis(50));
        assert_eq!(window.strategy, AlignmentStrategy::Exact);
        assert_eq!(window.max_buffer_size, 32);
    }

    #[test]
    fn test_temporal_join_exact_match() {
        let mut join: TemporalJoin<i32, i32> = TemporalJoin::with_config(
            JoinWindow::default().with_strategy(AlignmentStrategy::Exact),
        );

        join.push_left(Timestamp::from_millis(100), 1);
        join.push_right(Timestamp::from_millis(100), 2);

        match join.try_emit() {
            Some(JoinResult::Matched(a, b)) => {
                assert_eq!(a, 1);
                assert_eq!(b, 2);
            }
            other => panic!("Expected Matched, got {:?}", other),
        }
    }

    #[test]
    fn test_temporal_join_tolerance_match() {
        let mut join: TemporalJoin<i32, i32> = TemporalJoin::with_config(
            JoinWindow::default()
                .with_strategy(AlignmentStrategy::Tolerance(Duration::from_millis(10))),
        );

        join.push_left(Timestamp::from_millis(100), 1);
        join.push_right(Timestamp::from_millis(105), 2);

        match join.try_emit() {
            Some(JoinResult::Matched(a, b)) => {
                assert_eq!(a, 1);
                assert_eq!(b, 2);
            }
            other => panic!("Expected Matched, got {:?}", other),
        }
    }

    #[test]
    fn test_temporal_join_no_match() {
        let mut join: TemporalJoin<i32, i32> = TemporalJoin::with_config(
            JoinWindow::default()
                .with_strategy(AlignmentStrategy::Tolerance(Duration::from_millis(5))),
        );

        join.push_left(Timestamp::from_millis(100), 1);
        join.push_right(Timestamp::from_millis(200), 2);

        // Should emit LeftOnly since left is too old
        match join.try_emit() {
            Some(JoinResult::LeftOnly(a)) => {
                assert_eq!(a, 1);
            }
            other => panic!("Expected LeftOnly, got {:?}", other),
        }
    }

    #[test]
    fn test_temporal_join_nearest() {
        let mut join: TemporalJoin<i32, i32> = TemporalJoin::with_config(
            JoinWindow::default()
                .with_strategy(AlignmentStrategy::Nearest(Duration::from_millis(50))),
        );

        join.push_left(Timestamp::from_millis(100), 1);
        join.push_right(Timestamp::from_millis(80), 2);
        join.push_right(Timestamp::from_millis(95), 3); // Nearest to 100
        join.push_right(Timestamp::from_millis(120), 4);

        match join.try_emit() {
            Some(JoinResult::Matched(a, b)) => {
                assert_eq!(a, 1);
                assert_eq!(b, 3); // Nearest match
            }
            other => panic!("Expected Matched, got {:?}", other),
        }
    }

    #[test]
    fn test_temporal_join_buffer_len() {
        let mut join: TemporalJoin<i32, i32> = TemporalJoin::new();

        assert_eq!(join.left_len(), 0);
        assert_eq!(join.right_len(), 0);
        assert!(join.is_empty());

        join.push_left(Timestamp::from_millis(100), 1);
        join.push_left(Timestamp::from_millis(200), 2);
        join.push_right(Timestamp::from_millis(150), 3);

        assert_eq!(join.left_len(), 2);
        assert_eq!(join.right_len(), 1);
        assert!(!join.is_empty());
    }

    #[test]
    fn test_temporal_join_clear() {
        let mut join: TemporalJoin<i32, i32> = TemporalJoin::new();

        join.push_left(Timestamp::from_millis(100), 1);
        join.push_right(Timestamp::from_millis(100), 2);

        join.clear();

        assert!(join.is_empty());
    }

    #[test]
    fn test_timestamped_item() {
        let item = TimestampedItem::new(Timestamp::from_millis(100), "data");
        assert_eq!(item.timestamp.as_millis(), 100);
        assert_eq!(item.data, "data");
    }
}
