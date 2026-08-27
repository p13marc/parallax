//! Where cues wait for the host to draw them.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::clock::ClockTime;

use super::Cue;

/// How many cues are kept at once.
///
/// Subtitles are sparse — a busy film runs about one cue a second — so this
/// is a backstop against a malformed stream, not a working limit.
const CAPACITY: usize = 64;

/// The cues a host should be drawing, and the ones about to be due.
///
/// Modelled on [`FrameMailbox`](crate::video::FrameMailbox) — same flush
/// generations, same `Arc` sharing, same "the host asks, nothing is pushed at
/// it" shape — with one deliberate difference: a cue is an **interval**, so
/// this *retains* until the out-time rather than handing each item over once.
/// `FrameMailbox::take_due` pops and forgets, which for a subtitle would show
/// each line for exactly one repaint.
///
/// The other consequence of intervals: [`retire`](Self::retire) clears the
/// cue *on screen* as well as the queue. A seek that left the old subtitle
/// burnt over the new picture would be the exact bug the video path's
/// generation discipline exists to prevent.
#[derive(Debug, Default)]
pub struct CueTrack {
    cues: Mutex<Vec<Cue>>,
    generation: AtomicU64,
}

impl CueTrack {
    /// An empty track.
    pub fn new() -> Self {
        Self::default()
    }

    /// The flush generation cues must belong to to be shown.
    pub fn generation(&self) -> u64 {
        self.generation.load(Ordering::Acquire)
    }

    /// Add a cue, dropping it if it belongs to a superseded position.
    pub fn push(&self, cue: Cue) {
        if cue.generation < self.generation() || cue.is_blank() {
            return;
        }
        let mut cues = self.lock();
        // Keep them in start order so `active` reads cheaply and a late
        // arrival (a demuxer emitting out of order) still lands correctly.
        let at = cues
            .iter()
            .position(|existing| existing.start > cue.start)
            .unwrap_or(cues.len());
        cues.insert(at, cue);
        if cues.len() > CAPACITY {
            cues.remove(0);
        }
    }

    /// The cues on screen at `now`, in start order.
    ///
    /// Also drops everything that has expired, which is what keeps this from
    /// growing: a host that calls it at the frame rate never accumulates.
    pub fn active(&self, now: ClockTime) -> Vec<Cue> {
        let generation = self.generation();
        let mut cues = self.lock();
        cues.retain(|cue| {
            if cue.generation < generation {
                return false;
            }
            // Expired only if the clock has actually passed the end. Without
            // a clock nothing expires, because nothing is being paced.
            match (now.to_option(), cue.end().to_option()) {
                (Some(now), Some(end)) => now.nanos() < end.nanos(),
                _ => true,
            }
        });
        cues.iter()
            .filter(|cue| cue.active_at(now))
            .cloned()
            .collect()
    }

    /// Everything queued belongs to the position being left.
    ///
    /// Clears the cue on screen too — see the type's documentation.
    pub fn retire(&self) {
        self.generation.fetch_add(1, Ordering::AcqRel);
        self.lock().clear();
    }

    /// A `FlushStop` names the generation the next cues belong to.
    pub fn open(&self, generation: u64) {
        self.generation.store(generation, Ordering::Release);
    }

    /// Drop everything, without moving the generation.
    ///
    /// For a host switching subtitle tracks, where the old track's cues are
    /// wrong but the position has not changed.
    pub fn clear(&self) {
        self.lock().clear();
    }

    /// How many cues are held, for diagnostics.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Whether anything is held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Cue>> {
        // A poisoned lock here means a panic while holding cues, which loses
        // subtitles and nothing else — the picture keeps playing.
        self.cues.lock().unwrap_or_else(|e| e.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cue(text: &str, start_ms: u64, dur_ms: u64) -> Cue {
        Cue::new(
            text,
            ClockTime::from_millis(start_ms),
            ClockTime::from_millis(dur_ms),
        )
    }

    #[test]
    fn a_cue_is_retained_for_its_whole_interval() {
        let track = CueTrack::new();
        track.push(cue("hello", 1_000, 2_000));

        assert!(track.active(ClockTime::from_millis(500)).is_empty());
        // The same cue, over and over — which is the difference from a frame
        // mailbox, where taking it once is taking it for good.
        for at in [1_000, 1_500, 2_999] {
            let active = track.active(ClockTime::from_millis(at));
            assert_eq!(active.len(), 1, "at {at} ms");
            assert_eq!(active[0].text, "hello");
        }
        assert!(track.active(ClockTime::from_millis(3_000)).is_empty());
        assert!(track.is_empty(), "an expired cue is dropped, not kept");
    }

    #[test]
    fn overlapping_cues_are_both_on_screen_in_order() {
        let track = CueTrack::new();
        // Pushed out of order, as a demuxer interleaving tracks might.
        track.push(cue("second", 1_500, 2_000));
        track.push(cue("first", 1_000, 2_000));

        let active = track.active(ClockTime::from_millis(2_000));
        assert_eq!(active.len(), 2);
        assert_eq!(active[0].text, "first");
        assert_eq!(active[1].text, "second");
    }

    #[test]
    fn a_seek_clears_the_cue_on_screen_too() {
        let track = CueTrack::new();
        track.push(cue("before the seek", 0, 10_000));
        assert_eq!(track.active(ClockTime::from_millis(1_000)).len(), 1);

        track.retire();
        assert!(
            track.active(ClockTime::from_millis(1_000)).is_empty(),
            "the old subtitle must not stay burnt over the new picture"
        );

        // Cues from the position being left are refused even if they arrive
        // after the flush — which is exactly when they do arrive.
        track.push(cue("in flight", 500, 10_000));
        assert!(track.active(ClockTime::from_millis(1_000)).is_empty());

        // And the new generation flows again.
        track.open(track.generation());
        let mut fresh = cue("after the seek", 0, 10_000);
        fresh.generation = track.generation();
        track.push(fresh);
        assert_eq!(track.active(ClockTime::from_millis(100)).len(), 1);
    }

    #[test]
    fn a_timing_only_event_draws_nothing() {
        let track = CueTrack::new();
        track.push(cue("   ", 0, 1_000));
        assert!(
            track.is_empty(),
            "an ASS effect event has no text; an empty box over the picture is worse than nothing"
        );
    }

    #[test]
    fn a_runaway_stream_does_not_grow_without_bound() {
        let track = CueTrack::new();
        for i in 0..(CAPACITY as u64 * 3) {
            track.push(cue("x", i * 10, 1_000_000));
        }
        assert_eq!(track.len(), CAPACITY);
    }
}
