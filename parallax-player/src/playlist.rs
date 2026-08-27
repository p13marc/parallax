//! What to play, and what to play next.
//!
//! Deliberately a plain data structure with no knowledge of pipelines,
//! engines or Iced: everything here is a decision about indices, which makes
//! it the part of open-next that can be tested without a file, a window or a
//! sound card.

use std::path::PathBuf;

/// What happens when a file ends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Repeat {
    /// Play through the list and stop.
    #[default]
    Off,
    /// Play the same file again.
    One,
    /// Play through the list and start over.
    All,
}

impl Repeat {
    /// The next mode in the cycle, for the button and the key.
    pub fn next(self) -> Self {
        match self {
            Self::Off => Self::All,
            Self::All => Self::One,
            Self::One => Self::Off,
        }
    }

    /// What to call it on screen.
    pub fn label(self) -> &'static str {
        match self {
            Self::Off => "Repeat off",
            Self::All => "Repeat all",
            Self::One => "Repeat one",
        }
    }
}

/// One file in the list.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Where it is.
    pub path: PathBuf,
    /// Whether opening it has already failed once.
    ///
    /// Kept rather than removed: a file that failed to probe is still
    /// something the user put in the list, and silently dropping it looks
    /// like the drop never happened.
    pub failed: bool,
}

impl Entry {
    /// What to show for it — the file name, not the whole path.
    pub fn label(&self) -> String {
        self.path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.display().to_string())
    }
}

/// The files to play, and where in them we are.
#[derive(Debug, Default)]
pub struct Playlist {
    entries: Vec<Entry>,
    current: Option<usize>,
    /// What to do at the end of a file.
    pub repeat: Repeat,
}

impl Playlist {
    /// Everything in the list.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Which entry is playing, if any.
    pub fn current(&self) -> Option<usize> {
        self.current
    }

    /// Whether there is anything to play.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many files are queued.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Add files to the end, and return the index of the first new one.
    ///
    /// Adding to an empty list is how a drop or a dialog starts playback:
    /// the caller plays the returned index.
    pub fn extend<I: IntoIterator<Item = PathBuf>>(&mut self, paths: I) -> Option<usize> {
        let first = self.entries.len();
        for path in paths {
            self.entries.push(Entry {
                path,
                failed: false,
            });
        }
        (self.entries.len() > first).then_some(first)
    }

    /// Mark the entry playing, whatever it is.
    pub fn select(&mut self, index: usize) -> Option<PathBuf> {
        let path = self.entries.get(index)?.path.clone();
        self.current = Some(index);
        Some(path)
    }

    /// Record that an entry could not be opened.
    pub fn mark_failed(&mut self, index: usize) {
        if let Some(entry) = self.entries.get_mut(index) {
            entry.failed = true;
        }
    }

    /// Drop an entry, keeping `current` pointing at the same file.
    ///
    /// Removing what is playing returns `true`, because the caller has to
    /// decide what to do about that — this type will not start or stop
    /// anything on its own.
    pub fn remove(&mut self, index: usize) -> bool {
        if index >= self.entries.len() {
            return false;
        }
        self.entries.remove(index);
        match self.current {
            Some(current) if current == index => {
                self.current = None;
                return true;
            }
            // Everything after the hole shifted down by one.
            Some(current) if current > index => self.current = Some(current - 1),
            _ => {}
        }
        false
    }

    /// Where to go when the current file ends by itself.
    ///
    /// `Repeat::One` answers the same index, which is how repeat-one works
    /// at this level: the pipeline is rebuilt. The demuxer's own `--repeat`
    /// is a different mechanism — it suppresses end-of-stream entirely, so
    /// this is never asked.
    pub fn advance(&mut self) -> Option<usize> {
        let current = self.current?;
        let next = match self.repeat {
            Repeat::One => current,
            Repeat::All => (current + 1) % self.entries.len().max(1),
            Repeat::Off => current + 1,
        };
        (next < self.entries.len()).then_some(next)
    }

    /// Where to go when the user asks for the next file.
    ///
    /// An explicit press outranks `Repeat::One` — someone reaching for Next
    /// wants a different file, not the same one again.
    pub fn next(&mut self) -> Option<usize> {
        let current = self.current?;
        let next = current + 1;
        match (next < self.entries.len(), self.repeat) {
            (true, _) => Some(next),
            (false, Repeat::All) => Some(0),
            (false, _) => None,
        }
    }

    /// Where to go when the user asks for the previous file.
    pub fn previous(&mut self) -> Option<usize> {
        let current = self.current?;
        match (current, self.repeat) {
            (0, Repeat::All) => self.entries.len().checked_sub(1),
            (0, _) => None,
            (current, _) => Some(current - 1),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn list(count: usize) -> Playlist {
        let mut playlist = Playlist::default();
        playlist.extend((0..count).map(|i| PathBuf::from(format!("{i}.mp4"))));
        playlist
    }

    #[test]
    fn a_list_plays_through_and_stops() {
        let mut playlist = list(3);
        playlist.select(0);
        assert_eq!(playlist.advance(), Some(1));
        playlist.select(1);
        assert_eq!(playlist.advance(), Some(2));
        playlist.select(2);
        // The end of the list under Repeat::Off is the end.
        assert_eq!(playlist.advance(), None);
    }

    #[test]
    fn repeat_all_wraps_and_repeat_one_stays() {
        let mut playlist = list(3);
        playlist.repeat = Repeat::All;
        playlist.select(2);
        assert_eq!(playlist.advance(), Some(0));

        playlist.repeat = Repeat::One;
        playlist.select(1);
        assert_eq!(playlist.advance(), Some(1));
        // But an explicit Next still moves: a press outranks the mode.
        assert_eq!(playlist.next(), Some(2));
    }

    #[test]
    fn next_and_previous_stop_at_the_ends_unless_repeating_all() {
        let mut playlist = list(3);
        playlist.select(0);
        assert_eq!(playlist.previous(), None);
        playlist.select(2);
        assert_eq!(playlist.next(), None);

        playlist.repeat = Repeat::All;
        playlist.select(0);
        assert_eq!(playlist.previous(), Some(2));
        playlist.select(2);
        assert_eq!(playlist.next(), Some(0));
    }

    #[test]
    fn removing_an_entry_keeps_current_on_the_same_file() {
        let mut playlist = list(4);
        playlist.select(2);

        // Removing something earlier shifts the index down.
        assert!(!playlist.remove(0));
        assert_eq!(playlist.current(), Some(1));
        assert_eq!(playlist.entries()[1].label(), "2.mp4");

        // Removing something later leaves it alone.
        assert!(!playlist.remove(2));
        assert_eq!(playlist.current(), Some(1));

        // Removing what is playing says so, and leaves nothing current.
        assert!(playlist.remove(1));
        assert_eq!(playlist.current(), None);
        // With nothing current there is nowhere to advance to.
        assert_eq!(playlist.advance(), None);
    }

    #[test]
    fn an_out_of_range_remove_is_a_no_op() {
        let mut playlist = list(2);
        playlist.select(1);
        assert!(!playlist.remove(9));
        assert_eq!(playlist.len(), 2);
        assert_eq!(playlist.current(), Some(1));
    }

    #[test]
    fn the_repeat_cycle_reaches_every_mode_and_returns() {
        let mut mode = Repeat::Off;
        let mut seen = vec![mode];
        for _ in 0..3 {
            mode = mode.next();
            seen.push(mode);
        }
        assert_eq!(
            seen,
            vec![Repeat::Off, Repeat::All, Repeat::One, Repeat::Off]
        );
    }
}
