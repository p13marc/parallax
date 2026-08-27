//! One subtitle, and when it is on screen.

use crate::clock::ClockTime;

/// A line of timed text and the interval it belongs to.
///
/// The interval is the whole difference between a cue and a frame, and it is
/// why subtitles need their own store rather than a [`FrameMailbox`]: a frame
/// is shown and forgotten, a cue is *active* from `start` until
/// `start + duration` and has to be redrawn on every repaint in between.
///
/// [`FrameMailbox`]: crate::video::FrameMailbox
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Cue {
    /// The text, already stripped of container and styling syntax. Newlines
    /// are real newlines; the host wraps the rest.
    pub text: String,
    /// Running time the cue appears at.
    pub start: ClockTime,
    /// How long it stays up.
    ///
    /// Containers state this (Matroska's `BlockDuration`, SRT's second
    /// timestamp), so it is not a guess. A cue with no stated duration is
    /// given [`DEFAULT_DURATION`] rather than staying up forever.
    pub duration: ClockTime,
    /// The flush generation this cue belongs to (#265's discipline).
    ///
    /// A seek retires everything from the position being left, including the
    /// cue currently on screen — otherwise the old subtitle stays burnt over
    /// the new picture.
    pub generation: u64,
}

/// How long a cue with no stated duration stays up.
///
/// Long enough to read a line, short enough that a malformed file does not
/// leave one subtitle on screen for the rest of the film.
pub const DEFAULT_DURATION: ClockTime = ClockTime::from_secs(5);

impl Cue {
    /// A cue with a stated interval.
    pub fn new(text: impl Into<String>, start: ClockTime, duration: ClockTime) -> Self {
        Self {
            text: into_lines(text.into()),
            start,
            duration: match duration.to_option() {
                Some(d) if d.nanos() > 0 => duration,
                _ => DEFAULT_DURATION,
            },
            generation: 0,
        }
    }

    /// When the cue comes off screen.
    pub fn end(&self) -> ClockTime {
        match (self.start.to_option(), self.duration.to_option()) {
            (Some(start), Some(duration)) => {
                ClockTime::from_nanos(start.nanos().saturating_add(duration.nanos()))
            }
            // An unpaced cue has no end: it is shown until something replaces
            // it, which is what a stream with no timestamps deserves.
            _ => ClockTime::MAX,
        }
    }

    /// Whether the cue is on screen at `now`.
    pub fn active_at(&self, now: ClockTime) -> bool {
        let Some(now) = now.to_option() else {
            // No clock: a host draws whatever arrived, the same rule the
            // video path uses for an unpaced frame.
            return true;
        };
        let after_start = self
            .start
            .to_option()
            .is_none_or(|start| now.nanos() >= start.nanos());
        let before_end = self
            .end()
            .to_option()
            .is_none_or(|end| now.nanos() < end.nanos());
        after_start && before_end
    }

    /// Whether the cue is empty once stripped — a timing-only event, which
    /// ASS files use for effects and which must not draw an empty box.
    pub fn is_blank(&self) -> bool {
        self.text.trim().is_empty()
    }
}

/// Normalise the line breaks a subtitle format can spell three ways.
fn into_lines(text: String) -> String {
    text.replace("\r\n", "\n").replace('\r', "\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cue_is_on_screen_for_its_interval() {
        let cue = Cue::new("hello", ClockTime::from_secs(1), ClockTime::from_secs(2));
        assert!(!cue.active_at(ClockTime::from_millis(999)));
        assert!(cue.active_at(ClockTime::from_secs(1)));
        assert!(cue.active_at(ClockTime::from_millis(2_999)));
        // The end is exclusive: at exactly `start + duration` it is gone, so
        // two back-to-back cues never overlap by a frame.
        assert!(!cue.active_at(ClockTime::from_secs(3)));
    }

    #[test]
    fn a_cue_with_no_duration_does_not_stay_up_forever() {
        let cue = Cue::new("x", ClockTime::ZERO, ClockTime::ZERO);
        assert_eq!(cue.duration, DEFAULT_DURATION);
        let cue = Cue::new("x", ClockTime::ZERO, ClockTime::NONE);
        assert_eq!(cue.duration, DEFAULT_DURATION);
    }

    #[test]
    fn without_a_clock_a_cue_is_simply_shown() {
        let cue = Cue::new("x", ClockTime::from_secs(9), ClockTime::from_secs(1));
        assert!(
            cue.active_at(ClockTime::NONE),
            "an unpaced host draws what arrived, as the video path does"
        );
    }

    #[test]
    fn line_breaks_are_normalised_and_blanks_are_recognised() {
        let cue = Cue::new(
            "one\r\ntwo\rthree",
            ClockTime::ZERO,
            ClockTime::from_secs(1),
        );
        assert_eq!(cue.text, "one\ntwo\nthree");
        assert!(!cue.is_blank());
        // ASS uses timing-only events for effects; drawing an empty box for
        // one would be a flickering black rectangle over the picture.
        assert!(Cue::new("   \n ", ClockTime::ZERO, ClockTime::from_secs(1)).is_blank());
    }
}
