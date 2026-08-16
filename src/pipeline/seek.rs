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

/// An element's declared processing latency (#184).
///
/// Declared via the `latency()` trait method (jitter buffers, pacing
/// sinks), summed along each source→sink path at start, with the pipeline
/// figure being the worst path. Static and honest-but-coarse: it bounds
/// buffering an element *deliberately* introduces, not measured wall time
/// (that is [`LatencyTracer`](crate::pipeline::LatencyTracer)'s job).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LatencyRange {
    /// Minimum latency the element always introduces.
    pub min: crate::clock::ClockTime,
    /// Upper bound the element may introduce.
    pub max: crate::clock::ClockTime,
}

impl LatencyRange {
    /// A fixed latency (min == max).
    pub fn fixed(latency: crate::clock::ClockTime) -> Self {
        Self {
            min: latency,
            max: latency,
        }
    }

    /// Zero to `max` — an element that may hold data up to a bound.
    pub fn up_to(max: crate::clock::ClockTime) -> Self {
        Self {
            min: crate::clock::ClockTime::ZERO,
            max,
        }
    }

    /// Sum of two ranges (saturating), for path accumulation.
    pub fn plus(self, other: Self) -> Self {
        Self {
            min: crate::clock::ClockTime::from_nanos(
                self.min.nanos().saturating_add(other.min.nanos()),
            ),
            max: crate::clock::ClockTime::from_nanos(
                self.max.nanos().saturating_add(other.max.nanos()),
            ),
        }
    }
}

/// Result of a seekable range query.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeekableQuery {
    /// Whether seeking is supported.
    pub seekable: bool,
    /// The format `start`/`stop` are expressed in, and the format a seek on
    /// this pipeline should use.
    ///
    /// Usually the seekable source's own format, but a mid-graph element
    /// that translates seeks *replaces* it: `filesrc ! tsdemux` seeks in
    /// TIME even though only the source can seek, and it seeks in BYTES.
    pub format: SegmentFormat,
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
            format: SegmentFormat::Time,
            start: 0,
            stop: 0,
        }
    }
}

/// A format conversion a mid-graph element performs on upstream seeks.
///
/// Declared by [`AsyncElementDyn::seek_translations`] and snapshotted at
/// start: an element returning `{from: Time, to: Bytes}` accepts a TIME seek
/// and forwards a BYTES one, which is what makes a byte-seekable source
/// underneath present as TIME-seekable to the application.
///
/// [`AsyncElementDyn::seek_translations`]: crate::element::AsyncElementDyn::seek_translations
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SeekTranslation {
    /// The format this element accepts from downstream.
    pub from: SegmentFormat,
    /// The format it forwards upstream.
    pub to: SegmentFormat,
    /// What this element knows about the stream's total in `from`'s format.
    ///
    /// Almost always `None`: an element that could answer a TIME duration
    /// query usually would not need to translate in the first place. When
    /// it is known it bounds the translated range, which is otherwise open.
    pub duration: Option<DurationQuery>,
}

/// Fold per-source `(is_seekable, duration)` answers and the graph's seek
/// translations into one [`SeekableQuery`].
///
/// The first seekable source wins and its duration, when known, bounds the
/// range. A translation whose `to` matches that source's format then
/// **replaces** the reported format with its `from` — GStreamer's discipline,
/// where a demuxer refuses BYTE seekability downstream and offers TIME
/// instead. The range does not survive the swap (bytes are not nanoseconds),
/// so it reopens unless the translating element itself knows a duration.
///
/// Shared by the pre-start `Pipeline::query_seekable` and the runtime
/// `PipelineHandle::query_seekable`, so the two surfaces cannot drift.
pub(crate) fn aggregate_seekable<'a>(
    sources: impl IntoIterator<Item = (bool, Option<&'a DurationQuery>)>,
    translations: &[SeekTranslation],
) -> SeekableQuery {
    for (seekable, duration) in sources {
        if !seekable {
            continue;
        }
        let format = duration.map(|d| d.format).unwrap_or(SegmentFormat::Time);
        let mut query = SeekableQuery {
            seekable: true,
            format,
            start: 0,
            stop: duration.and_then(|d| d.duration).unwrap_or(0),
        };
        if let Some(t) = translations.iter().find(|t| t.to == format) {
            query.format = t.from;
            query.stop = t
                .duration
                .as_ref()
                .filter(|d| d.format == t.from)
                .and_then(|d| d.duration)
                .unwrap_or(0);
        }
        return query;
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
        let q = aggregate_seekable([(false, None), (true, Some(&time)), (true, None)], &[]);
        assert_eq!(
            q,
            SeekableQuery {
                seekable: true,
                format: SegmentFormat::Time,
                start: 0,
                stop: 2_000_000_000
            }
        );

        assert!(!aggregate_seekable([(false, Some(&time))], &[]).seekable);
        // Seekable with unknown duration: open-ended range, stop = 0.
        let q = aggregate_seekable([(true, None)], &[]);
        assert!(q.seekable);
        assert_eq!(q.stop, 0);
    }

    #[test]
    fn a_translation_replaces_the_reported_seek_format() {
        let bytes = DurationQuery {
            format: SegmentFormat::Bytes,
            duration: Some(4096),
        };
        let to_time = SeekTranslation {
            from: SegmentFormat::Time,
            to: SegmentFormat::Bytes,
            duration: None,
        };

        // filesrc alone: BYTES, bounded by the file size.
        let q = aggregate_seekable([(true, Some(&bytes))], &[]);
        assert_eq!(q.format, SegmentFormat::Bytes);
        assert_eq!(q.stop, 4096);

        // filesrc ! tsdemux: TIME, and the byte range does NOT carry over —
        // reporting 4096 nanoseconds of seekable media would be a lie.
        let q = aggregate_seekable([(true, Some(&bytes))], std::slice::from_ref(&to_time));
        assert_eq!(q.format, SegmentFormat::Time);
        assert_eq!(q.stop, 0, "a byte count is not a duration");

        // A translator that knows the duration bounds the new range.
        let knows = SeekTranslation {
            duration: Some(DurationQuery {
                format: SegmentFormat::Time,
                duration: Some(30_000_000_000),
            }),
            ..to_time.clone()
        };
        let q = aggregate_seekable([(true, Some(&bytes))], &[knows]);
        assert_eq!(q.stop, 30_000_000_000);

        // A translation that does not apply to the source's format is inert.
        let unrelated = SeekTranslation {
            to: SegmentFormat::Default,
            ..to_time
        };
        let q = aggregate_seekable([(true, Some(&bytes))], &[unrelated]);
        assert_eq!(q.format, SegmentFormat::Bytes);
        assert_eq!(q.stop, 4096);
    }
}
