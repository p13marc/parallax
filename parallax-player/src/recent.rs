//! The files opened lately.
//!
//! A newline-separated list under `$XDG_STATE_HOME`, newest first. Parallax
//! is Linux-only by design principle, so resolving that by hand costs five
//! lines and no dependency — and the result is a file that `cat` explains.

use std::path::{Path, PathBuf};

/// How many are remembered. A list long enough to scroll is a file manager.
const LIMIT: usize = 10;

/// Where the list lives.
///
/// `PARALLAX_PLAYER_STATE_DIR` overrides it, which is how the round-trip test
/// avoids touching a real home directory.
fn store() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("PARALLAX_PLAYER_STATE_DIR") {
        return Some(PathBuf::from(dir).join("recent"));
    }
    let base = match std::env::var("XDG_STATE_HOME") {
        Ok(dir) if !dir.is_empty() => PathBuf::from(dir),
        _ => PathBuf::from(std::env::var("HOME").ok()?).join(".local/state"),
    };
    Some(base.join("parallax-player").join("recent"))
}

/// The files opened lately, newest first.
///
/// A missing or unreadable list is an empty one: this is a convenience, and
/// failing to open a file nobody asked for is not worth a message.
pub fn load() -> Vec<PathBuf> {
    let Some(path) = store() else {
        return Vec::new();
    };
    let Ok(text) = std::fs::read_to_string(&path) else {
        return Vec::new();
    };
    text.lines()
        .filter(|line| !line.trim().is_empty())
        .map(PathBuf::from)
        .take(LIMIT)
        .collect()
}

/// Record that a file was opened, moving it to the front.
pub fn remember(path: &Path) {
    let Some(store) = store() else {
        return;
    };
    let updated = promote(&load(), path);
    let text: String = updated
        .iter()
        .map(|path| format!("{}\n", path.display()))
        .collect();

    if let Some(dir) = store.parent()
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        tracing::debug!("could not create {}: {e}", dir.display());
        return;
    }
    if let Err(e) = std::fs::write(&store, text) {
        tracing::debug!("could not write {}: {e}", store.display());
    }
}

/// `path` at the front, whatever it was before, capped at [`LIMIT`].
fn promote(existing: &[PathBuf], path: &Path) -> Vec<PathBuf> {
    let mut out = Vec::with_capacity(existing.len() + 1);
    out.push(path.to_path_buf());
    out.extend(
        existing
            .iter()
            .filter(|other| other.as_path() != path)
            .cloned(),
    );
    out.truncate(LIMIT);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_newest_goes_first_and_is_not_listed_twice() {
        let existing = vec![PathBuf::from("a.mp4"), PathBuf::from("b.mp4")];
        let after = promote(&existing, Path::new("c.mp4"));
        assert_eq!(
            after,
            vec![
                PathBuf::from("c.mp4"),
                PathBuf::from("a.mp4"),
                PathBuf::from("b.mp4")
            ]
        );

        // Re-opening something already in the list moves it rather than
        // adding a second copy.
        let again = promote(&after, Path::new("b.mp4"));
        assert_eq!(
            again,
            vec![
                PathBuf::from("b.mp4"),
                PathBuf::from("c.mp4"),
                PathBuf::from("a.mp4")
            ]
        );
    }

    #[test]
    fn the_list_stops_growing() {
        let existing: Vec<PathBuf> = (0..LIMIT)
            .map(|i| PathBuf::from(format!("{i}.mp4")))
            .collect();
        let after = promote(&existing, Path::new("new.mp4"));
        assert_eq!(after.len(), LIMIT);
        assert_eq!(after[0], PathBuf::from("new.mp4"));
        // The oldest fell off the end.
        assert!(!after.contains(&PathBuf::from(format!("{}.mp4", LIMIT - 1))));
    }
}
