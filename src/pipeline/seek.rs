//! Seeking, position queries, and playback rate control.
//!
//! This module provides the pipeline-level API for seeking in media streams,
//! querying position/duration, and controlling playback rate.
//!
//! # Seek Flow
//!
//! ```text
//! Application → pipeline.seek()
//!   1. FlushStart sent downstream (clears queues)
//!   2. SeekEvent sent upstream to sources
//!   3. Source repositions, emits new Segment downstream
//!   4. FlushStop sent downstream (resume processing)
//!   5. SeekDone posted to bus
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::seek::SeekableQuery;
//! use parallax::event::{SeekEvent, SegmentFormat};
//! use parallax::clock::ClockTime;
//!
//! // Check if seeking is supported
//! let seekable = pipeline.query_seekable();
//! if seekable.seekable {
//!     // Seek to 30 seconds
//!     pipeline.seek_simple(ClockTime::from_secs(30)).await?;
//!
//!     // Query current position
//!     if let Some(pos) = pipeline.query_position() {
//!         println!("Position: {pos}");
//!     }
//! }
//! ```

use crate::clock::ClockTime;
use crate::event::SegmentFormat;

// ============================================================================
// Query Types
// ============================================================================

/// Result of a position query.
#[derive(Debug, Clone)]
pub struct PositionQuery {
    /// Format of the position value.
    pub format: SegmentFormat,
    /// Current position, or `None` if unknown.
    pub position: Option<u64>,
}

/// Result of a duration query.
#[derive(Debug, Clone)]
pub struct DurationQuery {
    /// Format of the duration value.
    pub format: SegmentFormat,
    /// Total duration, or `None` if unknown (e.g., live stream).
    pub duration: Option<u64>,
}

/// Result of a seekable range query.
#[derive(Debug, Clone)]
pub struct SeekableQuery {
    /// Whether seeking is supported.
    pub seekable: bool,
    /// Start of seekable range (nanoseconds or bytes).
    pub start: u64,
    /// End of seekable range (nanoseconds or bytes).
    pub stop: u64,
}

impl SeekableQuery {
    /// Create a non-seekable query result.
    pub fn not_seekable() -> Self {
        Self {
            seekable: false,
            start: 0,
            stop: 0,
        }
    }
}

// ============================================================================
// Seekable Source Trait
// ============================================================================

/// Trait for sources that support seeking and position/duration queries.
///
/// Implement this trait on your [`Source`](crate::element::Source) to enable
/// seeking in pipelines. The pipeline will call these methods when the
/// application requests seek operations or queries.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::pipeline::seek::{SeekableSource, PositionQuery, DurationQuery, SeekableQuery};
/// use parallax::event::{SeekEvent, SegmentFormat};
///
/// impl SeekableSource for MyFileSource {
///     fn is_seekable(&self) -> bool { true }
///
///     fn query_position(&self) -> Option<PositionQuery> {
///         Some(PositionQuery {
///             format: SegmentFormat::Bytes,
///             position: Some(self.offset),
///         })
///     }
///
///     fn query_duration(&self) -> Option<DurationQuery> {
///         Some(DurationQuery {
///             format: SegmentFormat::Bytes,
///             duration: Some(self.file_size),
///         })
///     }
/// }
/// ```
pub trait SeekableSource {
    /// Whether this source supports seeking.
    fn is_seekable(&self) -> bool {
        false
    }

    /// Query the current position.
    fn query_position(&self) -> Option<PositionQuery> {
        None
    }

    /// Query the total duration.
    fn query_duration(&self) -> Option<DurationQuery> {
        None
    }

    /// Query the seekable range.
    fn query_seekable(&self) -> SeekableQuery {
        if self.is_seekable() {
            if let Some(dur) = self.query_duration() {
                return SeekableQuery {
                    seekable: true,
                    start: 0,
                    stop: dur.duration.unwrap_or(0),
                };
            }
        }
        SeekableQuery::not_seekable()
    }
}

// ============================================================================
// Pipeline Seek Request
// ============================================================================

/// A simplified seek request for pipeline-level API.
///
/// This is a convenience wrapper around [`SeekEvent`](crate::event::SeekEvent)
/// for common seek operations.
#[derive(Debug, Clone)]
pub struct SeekRequest {
    /// Target position in nanoseconds.
    pub position: ClockTime,
    /// Playback rate (1.0 = normal).
    pub rate: f64,
    /// Whether to flush (responsive seek).
    pub flush: bool,
    /// Whether to seek to nearest keyframe.
    pub key_unit: bool,
}

impl SeekRequest {
    /// Create a simple seek to a time position with flush.
    pub fn to_time(position: ClockTime) -> Self {
        Self {
            position,
            rate: 1.0,
            flush: true,
            key_unit: true,
        }
    }

    /// Create a rate change without seeking.
    pub fn rate_change(rate: f64) -> Self {
        Self {
            position: ClockTime::NONE,
            rate,
            flush: true,
            key_unit: false,
        }
    }

    /// Set the playback rate.
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate = rate;
        self
    }

    /// Set whether to flush.
    pub fn with_flush(mut self, flush: bool) -> Self {
        self.flush = flush;
        self
    }

    /// Set whether to seek to keyframe.
    pub fn with_key_unit(mut self, key_unit: bool) -> Self {
        self.key_unit = key_unit;
        self
    }

    /// Convert to a [`SeekEvent`](crate::event::SeekEvent).
    pub fn to_seek_event(&self) -> crate::event::SeekEvent {
        use crate::event::{SeekEvent, SeekFlags, SeekPosition};

        let mut flags = SeekFlags::empty();
        if self.flush {
            flags = flags.union(SeekFlags::FLUSH);
        }
        if self.key_unit {
            flags = flags.union(SeekFlags::KEY_UNIT);
        }

        let start = if self.position == ClockTime::NONE {
            SeekPosition::none()
        } else {
            SeekPosition::set(self.position.nanos() as i64)
        };

        SeekEvent {
            rate: self.rate,
            format: SegmentFormat::Time,
            flags,
            start,
            stop: SeekPosition::none(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::SegmentEvent;

    #[test]
    fn test_segment_running_time_normal() {
        // Normal playback: rate=1.0, start=0, base=0
        let seg = SegmentEvent::new_time(ClockTime::ZERO, None);
        let pts = ClockTime::from_secs(5);
        let rt = seg.to_running_time(pts);
        assert_eq!(rt, ClockTime::from_secs(5));
    }

    #[test]
    fn test_segment_running_time_after_seek() {
        // Seek to 10s, base=5s (accumulated running time)
        let seg = SegmentEvent {
            format: SegmentFormat::Time,
            start: ClockTime::from_secs(10).nanos() as i64,
            stop: -1,
            position: ClockTime::from_secs(10).nanos() as i64,
            rate: 1.0,
            applied_rate: 1.0,
            base: ClockTime::from_secs(5).nanos() as i64,
            flags: Default::default(),
        };

        // PTS=10s → running_time = (10-10)/1 + 5 = 5s
        let rt = seg.to_running_time(ClockTime::from_secs(10));
        assert_eq!(rt, ClockTime::from_secs(5));

        // PTS=12s → running_time = (12-10)/1 + 5 = 7s
        let rt = seg.to_running_time(ClockTime::from_secs(12));
        assert_eq!(rt, ClockTime::from_secs(7));
    }

    #[test]
    fn test_segment_running_time_2x_rate() {
        // 2x playback: rate=2.0, start=0, base=0
        let seg = SegmentEvent::new_time(ClockTime::ZERO, None).with_rate(2.0);
        // PTS=10s → running_time = 10/2 = 5s
        let rt = seg.to_running_time(ClockTime::from_secs(10));
        assert_eq!(rt, ClockTime::from_secs(5));
    }

    #[test]
    fn test_segment_running_time_before_start() {
        let seg = SegmentEvent::new_time(ClockTime::from_secs(10), None);
        // PTS=5s is before segment start → NONE
        let rt = seg.to_running_time(ClockTime::from_secs(5));
        assert_eq!(rt, ClockTime::NONE);
    }

    #[test]
    fn test_segment_stream_time() {
        // Seek to 10s, position=10s
        let seg = SegmentEvent::new_time(ClockTime::from_secs(10), None);
        // PTS=15s → stream_time = (15-10)*1 + 10 = 15s
        let st = seg.to_stream_time(ClockTime::from_secs(15));
        assert_eq!(st, ClockTime::from_secs(15));
    }

    #[test]
    fn test_segment_running_time_roundtrip() {
        let seg = SegmentEvent {
            format: SegmentFormat::Time,
            start: ClockTime::from_secs(10).nanos() as i64,
            stop: -1,
            position: ClockTime::from_secs(10).nanos() as i64,
            rate: 1.0,
            applied_rate: 1.0,
            base: ClockTime::from_secs(5).nanos() as i64,
            flags: Default::default(),
        };

        let pts = ClockTime::from_secs(15);
        let rt = seg.to_running_time(pts);
        let roundtrip = seg.running_time_to_pts(rt);
        assert_eq!(roundtrip, pts);
    }

    #[test]
    fn test_segment_none_handling() {
        let seg = SegmentEvent::default();
        assert_eq!(seg.to_running_time(ClockTime::NONE), ClockTime::NONE);
        assert_eq!(seg.to_stream_time(ClockTime::NONE), ClockTime::NONE);
        assert_eq!(seg.running_time_to_pts(ClockTime::NONE), ClockTime::NONE);
    }

    #[test]
    fn test_seek_request_to_event() {
        let req = SeekRequest::to_time(ClockTime::from_secs(30));
        let event = req.to_seek_event();
        assert_eq!(event.rate, 1.0);
        assert!(event.flags.contains(crate::event::SeekFlags::FLUSH));
        assert!(event.flags.contains(crate::event::SeekFlags::KEY_UNIT));
        assert_eq!(event.start.position, 30_000_000_000);
    }

    #[test]
    fn test_seekable_query_not_seekable() {
        let q = SeekableQuery::not_seekable();
        assert!(!q.seekable);
    }
}
