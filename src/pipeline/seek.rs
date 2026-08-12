//! Seeking and position/duration query types.
//!
//! # The two surfaces
//!
//! **Element authors** declare seek support on the trait they implement —
//! [`Source`], [`Demuxer`] and [`SimpleSource`] each carry optional
//! `is_seekable()` / `query_position()` / `query_duration()` methods plus
//! `handle_upstream_event` for the actual reposition. Declaring
//! `is_seekable() == true` is what makes seeks reach the element at all.
//!
//! **Applications** seek and query through [`PipelineHandle`] on a running
//! pipeline (`seekable()`, `query_seekable()`, `duration()`, `seek_time()`,
//! …) — the graph-level `Pipeline::seek_*`/`query_*` methods only work
//! *before* `Executor::start` moves the elements into their tasks.
//!
//! # Seek flow (running pipeline)
//!
//! ```text
//! Application → handle.seek_time(t)
//!   0. Gate: no source declared is_seekable() → returns false, nothing
//!      happens (GStreamer parity)
//!   1. The SeekEvent enters the graph at the SINKS and travels upstream
//!      hop-by-hop (#163): each hop's element may handle it or pass it on
//!      (EventResult::NotHandled forwards to every parent; seqnum dedup
//!      collapses multi-path delivery in fan-out diamonds). Hybrid
//!      pipelines fall back to direct fan-out to seekable sources — RT
//!      segments carry no events.
//!   2. The handling element repositions (handle_upstream_event); its own
//!      task then runs the flush sequence — mid-graph handlers included
//!   3. FlushStart sent downstream in-band; the flush epoch sheds the
//!      queued pre-seek backlog at receive speed
//!   4. FlushStop sent downstream (resume processing)
//!   5. New Segment sent downstream (re-anchors the timeline, carrying
//!      the seek's rate and stop and the element-reported landing)
//!   6. SeekDone posted to the bus, naming the handling element
//! ```
//!
//! # Segment discipline (#165)
//!
//! Every buffer on the wire is preceded by a Segment: each producing task
//! emits StreamStart at startup and a lazy initial Segment anchored at its
//! first buffer's PTS (Bytes-from-0 for byte-oriented sources like FileSrc/
//! HttpSrc, whose duration queries answer in Bytes). Demuxers and muxers
//! OWN their pads' stream identity — per-pad StreamStart/Segment on the way
//! out, upstream StreamStart/Segment swallowed (offered to the element
//! first), and pads re-anchor after an upstream FlushStop. `AutoVideoSink`
//! paces in segment running time via [`SegmentEvent::to_running_time`].
//!
//! [`SegmentEvent::to_running_time`]: crate::event::SegmentEvent::to_running_time
//!
//! [`Source`]: crate::element::Source
//! [`Demuxer`]: crate::element::Demuxer
//! [`SimpleSource`]: crate::element::SimpleSource
//! [`PipelineHandle`]: crate::pipeline::PipelineHandle

use crate::event::SegmentFormat;

// ============================================================================
// Query Types
// ============================================================================

/// Result of a position query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PositionQuery {
    /// Format of the position value.
    pub format: SegmentFormat,
    /// Current position, or `None` if unknown.
    pub position: Option<u64>,
}

/// Result of a duration query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DurationQuery {
    /// Format of the duration value.
    pub format: SegmentFormat,
    /// Total duration, or `None` if unknown (e.g., live stream).
    pub duration: Option<u64>,
}

/// Result of a seekable range query.
#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Fold per-source `(is_seekable, duration)` answers into one
/// [`SeekableQuery`]: the first seekable source wins, its duration (when
/// known) bounds the range.
///
/// Shared by the pre-start `Pipeline::query_seekable` and the runtime
/// `PipelineHandle::query_seekable`, so the two surfaces cannot drift.
pub(crate) fn aggregate_seekable<'a>(
    sources: impl IntoIterator<Item = (bool, Option<&'a DurationQuery>)>,
) -> SeekableQuery {
    for (seekable, duration) in sources {
        if seekable {
            return SeekableQuery {
                seekable: true,
                start: 0,
                stop: duration.and_then(|d| d.duration).unwrap_or(0),
            };
        }
    }
    SeekableQuery::not_seekable()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::ClockTime;
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
    fn test_seekable_query_not_seekable() {
        let q = SeekableQuery::not_seekable();
        assert!(!q.seekable);
    }

    #[test]
    fn aggregate_prefers_the_first_seekable_source() {
        let time = DurationQuery {
            format: SegmentFormat::Time,
            duration: Some(2_000_000_000),
        };
        let q = aggregate_seekable([(false, None), (true, Some(&time)), (true, None)]);
        assert_eq!(
            q,
            SeekableQuery {
                seekable: true,
                start: 0,
                stop: 2_000_000_000
            }
        );

        assert!(!aggregate_seekable([(false, Some(&time))]).seekable);
        // Seekable with unknown duration: open-ended range, stop = 0.
        let q = aggregate_seekable([(true, None)]);
        assert!(q.seekable);
        assert_eq!(q.stop, 0);
    }
}
