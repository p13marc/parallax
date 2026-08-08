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
    /// Resample the right stream onto the left stream's timestamps.
    ///
    /// For each left item at time `t`, the two right items that *bracket* `t`
    /// are found and combined with [`Lerp::lerp`] at the fractional position
    /// of `t` between them. The `Duration` is the widest bracket accepted: a
    /// gap wider than this is treated as missing data rather than something to
    /// interpolate across.
    ///
    /// # Requires `B: Lerp`
    ///
    /// Interpolation has to *combine* two values, which `Clone` cannot
    /// express. This variant is therefore only honoured by
    /// [`TemporalJoin::try_emit_interpolated`], which is available when the
    /// right-hand type implements [`Lerp`]. The unbounded
    /// [`TemporalJoin::try_emit`] declines it rather than silently substituting
    /// another strategy.
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

/// Linear interpolation between two values of the same type.
///
/// Implement this for a stream's payload to make
/// [`AlignmentStrategy::Interpolate`] usable with it — see
/// [`TemporalJoin::try_emit_interpolated`].
///
/// `t` is the fractional position between `self` (`t == 0.0`) and `other`
/// (`t == 1.0`). Callers in this module always pass `t` within `0.0..=1.0`,
/// but an implementation should behave sensibly if handed a value outside that
/// range rather than panicking.
///
/// # Example
///
/// ```
/// use parallax::temporal::Lerp;
///
/// #[derive(Clone, Debug, PartialEq)]
/// struct Position { x: f64, y: f64 }
///
/// impl Lerp for Position {
///     fn lerp(&self, other: &Self, t: f64) -> Self {
///         Position { x: self.x.lerp(&other.x, t), y: self.y.lerp(&other.y, t) }
///     }
/// }
///
/// let a = Position { x: 0.0, y: 10.0 };
/// let b = Position { x: 4.0, y: 20.0 };
/// assert_eq!(a.lerp(&b, 0.25), Position { x: 1.0, y: 12.5 });
/// ```
pub trait Lerp {
    /// Interpolate between `self` and `other` at fraction `t`.
    fn lerp(&self, other: &Self, t: f64) -> Self;
}

impl Lerp for f64 {
    fn lerp(&self, other: &Self, t: f64) -> Self {
        self + (other - self) * t
    }
}

impl Lerp for f32 {
    fn lerp(&self, other: &Self, t: f64) -> Self {
        (*self as f64 + (*other as f64 - *self as f64) * t) as f32
    }
}

/// Integer interpolation rounds to nearest and saturates at the type's bounds,
/// so a wide interpolation can never wrap or panic.
macro_rules! impl_lerp_int {
    ($($ty:ty),* $(,)?) => {
        $(impl Lerp for $ty {
            fn lerp(&self, other: &Self, t: f64) -> Self {
                let a = *self as f64;
                let v = (a + (*other as f64 - a) * t).round();
                if v <= <$ty>::MIN as f64 {
                    <$ty>::MIN
                } else if v >= <$ty>::MAX as f64 {
                    <$ty>::MAX
                } else {
                    v as $ty
                }
            }
        })*
    };
}

impl_lerp_int!(i8, i16, i32, i64, isize, u8, u16, u32, u64, usize);

impl Lerp for Duration {
    fn lerp(&self, other: &Self, t: f64) -> Self {
        Duration::from_secs_f64(self.as_secs_f64().lerp(&other.as_secs_f64(), t).max(0.0))
    }
}

impl<A: Lerp, B: Lerp> Lerp for (A, B) {
    fn lerp(&self, other: &Self, t: f64) -> Self {
        (self.0.lerp(&other.0, t), self.1.lerp(&other.1, t))
    }
}

impl<A: Lerp, B: Lerp, C: Lerp> Lerp for (A, B, C) {
    fn lerp(&self, other: &Self, t: f64) -> Self {
        (
            self.0.lerp(&other.0, t),
            self.1.lerp(&other.1, t),
            self.2.lerp(&other.2, t),
        )
    }
}

impl<T: Lerp, const N: usize> Lerp for [T; N] {
    fn lerp(&self, other: &Self, t: f64) -> Self {
        std::array::from_fn(|i| self[i].lerp(&other[i], t))
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
                // Interpolation needs to *combine* two right-hand values, which
                // `B: Clone` cannot express — see `try_emit_interpolated`,
                // which is available when `B: Lerp`.
                //
                // This used to silently run `Nearest(10ms)` instead, discarding
                // both the strategy and the caller's own `Duration`. Declining
                // is the honest answer.
                static WARNED: std::sync::Once = std::sync::Once::new();
                WARNED.call_once(|| {
                    tracing::warn!(
                        "AlignmentStrategy::Interpolate ignored by TemporalJoin::try_emit: \
                         the right-hand type must implement parallax::temporal::Lerp, and \
                         you must call try_emit_interpolated(). No items will be emitted."
                    );
                });
                None
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
        if let Some(right_front) = self.right_buffer.front()
            && left_ts + tolerance < right_front.timestamp
        {
            let left = self.left_buffer.pop_front()?.data;
            return Some(JoinResult::LeftOnly(left));
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
        if let Some(right_front) = self.right_buffer.front()
            && left_ts + window < right_front.timestamp
        {
            let left = self.left_buffer.pop_front()?.data;
            return Some(JoinResult::LeftOnly(left));
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

impl<A, B> TemporalJoin<A, B>
where
    A: Clone,
    B: Clone + Lerp,
{
    /// Emit a pair, honouring [`AlignmentStrategy::Interpolate`].
    ///
    /// This is a superset of [`try_emit`](Self::try_emit): every other
    /// strategy behaves identically, so it is a drop-in replacement whenever
    /// the right-hand type implements [`Lerp`].
    ///
    /// Under `Interpolate`, the **right** stream is resampled onto the left
    /// stream's timestamps — "give me B as it would have been at each A".
    /// Only `B` needs to be interpolable, because only `B` is ever
    /// synthesised; left items are always emitted as they arrived.
    ///
    /// Right items are *not* consumed by a match: the item before the left
    /// timestamp is usually the left bracket for the next left item too. They
    /// age out on the watermark like everything else.
    ///
    /// # Example
    ///
    /// ```
    /// use parallax::temporal::{AlignmentStrategy, JoinResult, JoinWindow, TemporalJoin, Timestamp};
    /// use std::time::Duration;
    ///
    /// let mut join: TemporalJoin<&str, f64> = TemporalJoin::with_config(
    ///     JoinWindow::default()
    ///         .with_strategy(AlignmentStrategy::Interpolate(Duration::from_millis(100))),
    /// );
    ///
    /// // Right samples at 100ms and 200ms; a left item lands between them.
    /// join.push_right(Timestamp::from_millis(100), 10.0);
    /// join.push_right(Timestamp::from_millis(200), 20.0);
    /// join.push_left(Timestamp::from_millis(125), "reading");
    ///
    /// match join.try_emit_interpolated() {
    ///     Some(JoinResult::Matched(_, b)) => assert!((b - 12.5).abs() < 1e-9),
    ///     other => panic!("expected an interpolated match, got {other:?}"),
    /// }
    /// ```
    pub fn try_emit_interpolated(&mut self) -> Option<JoinResult<A, B>> {
        if let AlignmentStrategy::Interpolate(max_span) = self.config.strategy {
            self.cleanup_old_items();
            if self.left_buffer.is_empty() && self.right_buffer.is_empty() {
                return None;
            }
            self.try_emit_interpolate(max_span)
        } else {
            self.try_emit()
        }
    }

    /// Resample the right stream at the oldest left item's timestamp.
    fn try_emit_interpolate(&mut self, max_span: Duration) -> Option<JoinResult<A, B>> {
        let left_ts = self.left_buffer.front()?.timestamp;

        // Find the tightest bracket [lo, hi] around `left_ts`. The buffers are
        // pushed in arrival order, which is not necessarily timestamp order for
        // a jittery source, so scan rather than assuming sortedness.
        let mut lo: Option<usize> = None;
        let mut hi: Option<usize> = None;
        for (i, item) in self.right_buffer.iter().enumerate() {
            if item.timestamp <= left_ts {
                match lo {
                    Some(j) if self.right_buffer[j].timestamp >= item.timestamp => {}
                    _ => lo = Some(i),
                }
            }
            if item.timestamp >= left_ts {
                match hi {
                    Some(j) if self.right_buffer[j].timestamp <= item.timestamp => {}
                    _ => hi = Some(i),
                }
            }
        }

        if let (Some(lo), Some(hi)) = (lo, hi) {
            let lo_item = &self.right_buffer[lo];
            let hi_item = &self.right_buffer[hi];
            let span = hi_item.timestamp.abs_diff(&lo_item.timestamp);

            // A gap wider than the caller's window is missing data, not
            // something to invent a value across.
            if span > max_span {
                let left = self.left_buffer.pop_front()?.data;
                return Some(JoinResult::LeftOnly(left));
            }

            // Zero span means an exact hit (or two samples sharing a
            // timestamp); either way `lo` is the answer and t would be 0/0.
            let value = if span.is_zero() {
                lo_item.data.clone()
            } else {
                let t = left_ts.abs_diff(&lo_item.timestamp).as_secs_f64() / span.as_secs_f64();
                lo_item.data.lerp(&hi_item.data, t.clamp(0.0, 1.0))
            };

            let left = self.left_buffer.pop_front()?.data;
            return Some(JoinResult::Matched(left, value));
        }

        // Only a later bracket exists: this left item predates the right
        // stream entirely and can never be bracketed. Extrapolating would be
        // inventing data, so emit it unmatched — same shape as the other
        // strategies' too-old check.
        if lo.is_none()
            && let Some(earliest) = self.right_buffer.iter().map(|i| i.timestamp).min()
            && left_ts + max_span < earliest
        {
            let left = self.left_buffer.pop_front()?.data;
            return Some(JoinResult::LeftOnly(left));
        }

        // Otherwise the upper bracket simply hasn't arrived yet.
        None
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

    // ========================================================================
    // Interpolation (#5)
    // ========================================================================

    #[test]
    fn lerp_floats_and_integers() {
        assert_eq!(0.0f64.lerp(&10.0, 0.25), 2.5);
        assert_eq!(0.0f32.lerp(&10.0, 0.5), 5.0);
        // Integers round to nearest rather than truncating.
        assert_eq!(0i32.lerp(&10, 0.25), 3);
        assert_eq!(0i32.lerp(&10, 0.24), 2);
        // Endpoints are exact.
        assert_eq!((-5i8).lerp(&5, 0.0), -5);
        assert_eq!((-5i8).lerp(&5, 1.0), 5);
    }

    #[test]
    fn lerp_integers_saturate_instead_of_wrapping() {
        // A huge extrapolation factor must not wrap or panic.
        assert_eq!(0u8.lerp(&255, 100.0), u8::MAX);
        assert_eq!(0i8.lerp(&127, -100.0), i8::MIN);
    }

    #[test]
    fn lerp_composites() {
        assert_eq!((0.0f64, 0i32).lerp(&(10.0, 100), 0.5), (5.0, 50));
        assert_eq!([0i32, 10].lerp(&[10, 20], 0.5), [5, 15]);
        assert_eq!(
            Duration::from_millis(0).lerp(&Duration::from_millis(100), 0.5),
            Duration::from_millis(50)
        );
    }

    #[test]
    fn interpolate_resamples_the_right_stream_onto_the_left_timestamps() {
        let mut join: TemporalJoin<&str, f64> = TemporalJoin::with_config(
            JoinWindow::default()
                .with_strategy(AlignmentStrategy::Interpolate(Duration::from_millis(200))),
        );

        join.push_right(Timestamp::from_millis(100), 10.0);
        join.push_right(Timestamp::from_millis(200), 20.0);
        join.push_left(Timestamp::from_millis(125), "a");

        match join.try_emit_interpolated() {
            Some(JoinResult::Matched(a, b)) => {
                assert_eq!(a, "a");
                assert!((b - 12.5).abs() < 1e-9, "expected 12.5, got {b}");
            }
            other => panic!("expected an interpolated match, got {other:?}"),
        }
    }

    #[test]
    fn interpolate_does_not_consume_the_right_bracket() {
        // The sample before a left item is usually the left bracket for the
        // next left item too, so a match must not remove it.
        let mut join: TemporalJoin<u32, f64> = TemporalJoin::with_config(
            JoinWindow::default()
                .with_strategy(AlignmentStrategy::Interpolate(Duration::from_millis(200))),
        );

        join.push_right(Timestamp::from_millis(100), 0.0);
        join.push_right(Timestamp::from_millis(200), 100.0);
        join.push_left(Timestamp::from_millis(125), 1);
        join.push_left(Timestamp::from_millis(175), 2);

        let first = join.try_emit_interpolated();
        let second = join.try_emit_interpolated();

        match (first, second) {
            (Some(JoinResult::Matched(1, b1)), Some(JoinResult::Matched(2, b2))) => {
                assert!((b1 - 25.0).abs() < 1e-9, "first: {b1}");
                assert!((b2 - 75.0).abs() < 1e-9, "second: {b2}");
            }
            other => panic!("expected two interpolated matches, got {other:?}"),
        }
        assert_eq!(
            join.right_len(),
            2,
            "both brackets should still be buffered"
        );
    }

    #[test]
    fn interpolate_honours_the_callers_duration_as_the_bracket_limit() {
        // The bug this replaced ignored the Duration entirely and substituted a
        // hardcoded 10ms Nearest window.
        let mut join: TemporalJoin<u32, f64> = TemporalJoin::with_config(
            JoinWindow::with_max_delay(Duration::from_secs(10))
                .with_strategy(AlignmentStrategy::Interpolate(Duration::from_millis(50))),
        );

        // A 100ms gap is wider than the 50ms limit: that is missing data, not
        // something to invent a value across.
        join.push_right(Timestamp::from_millis(100), 0.0);
        join.push_right(Timestamp::from_millis(200), 100.0);
        join.push_left(Timestamp::from_millis(150), 1);

        match join.try_emit_interpolated() {
            Some(JoinResult::LeftOnly(1)) => {}
            other => panic!("expected LeftOnly for an over-wide bracket, got {other:?}"),
        }
    }

    #[test]
    fn interpolate_waits_for_the_upper_bracket() {
        let mut join: TemporalJoin<u32, f64> = TemporalJoin::with_config(
            JoinWindow::with_max_delay(Duration::from_secs(10))
                .with_strategy(AlignmentStrategy::Interpolate(Duration::from_millis(200))),
        );

        join.push_right(Timestamp::from_millis(100), 0.0);
        join.push_left(Timestamp::from_millis(150), 1);

        // Only a lower bracket so far — emitting now would mean extrapolating.
        assert!(join.try_emit_interpolated().is_none());

        join.push_right(Timestamp::from_millis(200), 100.0);
        match join.try_emit_interpolated() {
            Some(JoinResult::Matched(1, b)) => assert!((b - 50.0).abs() < 1e-9),
            other => panic!("expected a match once bracketed, got {other:?}"),
        }
    }

    #[test]
    fn interpolate_emits_an_exact_hit_unchanged() {
        let mut join: TemporalJoin<u32, f64> = TemporalJoin::with_config(
            JoinWindow::default()
                .with_strategy(AlignmentStrategy::Interpolate(Duration::from_millis(200))),
        );

        join.push_right(Timestamp::from_millis(100), 42.0);
        join.push_left(Timestamp::from_millis(100), 1);

        match join.try_emit_interpolated() {
            Some(JoinResult::Matched(1, b)) => assert_eq!(b, 42.0),
            other => panic!("expected the exact sample, got {other:?}"),
        }
    }

    #[test]
    fn try_emit_declines_interpolate_instead_of_silently_using_nearest() {
        // Before the fix this ran Nearest(10ms) and returned a Matched pair,
        // so a caller could not tell that interpolation never happened.
        let mut join: TemporalJoin<u32, f64> = TemporalJoin::with_config(
            JoinWindow::default()
                .with_strategy(AlignmentStrategy::Interpolate(Duration::from_millis(200))),
        );

        join.push_right(Timestamp::from_millis(100), 10.0);
        join.push_right(Timestamp::from_millis(200), 20.0);
        join.push_left(Timestamp::from_millis(125), 1);

        assert!(
            join.try_emit().is_none(),
            "try_emit must not fabricate a match for Interpolate"
        );
    }

    #[test]
    fn try_emit_interpolated_matches_try_emit_for_other_strategies() {
        let make = || {
            let mut join: TemporalJoin<u32, f64> = TemporalJoin::with_config(
                JoinWindow::default().with_strategy(AlignmentStrategy::Exact),
            );
            join.push_left(Timestamp::from_millis(100), 1);
            join.push_right(Timestamp::from_millis(100), 2.0);
            join
        };

        let a = matches!(make().try_emit(), Some(JoinResult::Matched(1, _)));
        let b = matches!(
            make().try_emit_interpolated(),
            Some(JoinResult::Matched(1, _))
        );
        assert!(
            a && b,
            "the two entry points must agree off the Interpolate path"
        );
    }
}
