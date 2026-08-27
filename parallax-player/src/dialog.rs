//! Asking the desktop for a file.
//!
//! Through the XDG portal rather than a toolkit dialog, because parallax
//! already depends on `ashpd` for the screencast portal and a second file
//! chooser crate would bring a second copy of it. The portal is also the only
//! thing that works from a sandbox.
//!
//! Drag-and-drop is the other way in, and it is the *garnish*:
//! `window::Event::FileDropped` says "**Wayland:** Not implemented" in its own
//! documentation, and this player enables both backends.

use std::path::PathBuf;

use ashpd::desktop::file_chooser::{FileFilter, OpenFileRequest};

/// Ask the desktop for video files to play.
///
/// An empty answer covers every ordinary way this ends — the user cancelled,
/// or there is no portal running at all, which is the case on a bare X session
/// and on this project's headless dev box. The caller says so; there is
/// nothing here worth failing over.
pub async fn open_files() -> Result<Vec<PathBuf>, String> {
    let filter = FileFilter::new("Video")
        .glob("*.mp4")
        .glob("*.mov")
        .glob("*.m4v")
        .glob("*.mkv")
        .glob("*.webm");

    let request = OpenFileRequest::default()
        .title("Open video")
        .multiple(true)
        .modal(true)
        .filter(filter)
        .filter(FileFilter::new("All files").glob("*"));

    let response = request
        .send()
        .await
        .map_err(|e| format!("no file dialog available ({e})"))?
        .response()
        .map_err(|e| format!("the file dialog was closed ({e})"))?;

    Ok(response
        .uris()
        .iter()
        .filter_map(|uri| path_of(uri.as_str()))
        .collect())
}

/// The local path a `file://` URI names.
///
/// Anything else — a `http://` URI from a portal backend that offers remote
/// locations — is skipped rather than mangled into a path that does not
/// exist.
fn path_of(uri: &str) -> Option<PathBuf> {
    let rest = uri.strip_prefix("file://")?;
    // `file:///path` after the authority, which is empty for local files.
    let (_authority, path) = rest.split_once('/')?;
    let decoded = percent_encoding::percent_decode_str(path)
        .decode_utf8()
        .ok()?;
    Some(PathBuf::from(format!("/{decoded}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_file_uri_becomes_the_path_it_names() {
        assert_eq!(
            path_of("file:///home/me/a movie.mp4".replace(' ', "%20").as_str()),
            Some(PathBuf::from("/home/me/a movie.mp4"))
        );
        assert_eq!(
            path_of("file:///srv/video/clip.mkv"),
            Some(PathBuf::from("/srv/video/clip.mkv"))
        );
        // A hash and a plus are both legal in a file name and both encoded.
        assert_eq!(
            path_of("file:///tmp/a%23b%2Bc.mp4"),
            Some(PathBuf::from("/tmp/a#b+c.mp4"))
        );
        // Anything that is not a local file is skipped, not guessed at.
        assert_eq!(path_of("https://example.com/clip.mp4"), None);
        assert_eq!(path_of("file://"), None);
    }
}
