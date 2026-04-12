//! Stream tag/metadata system.
//!
//! Tags carry stream-level metadata discovered during processing,
//! such as ID3 tags, codec information, or media properties.
//!
//! Modeled after GStreamer's `GstTagList`.

use std::collections::HashMap;
use std::fmt;

use crate::clock::ClockTime;

// ============================================================================
// Standard Tag Keys
// ============================================================================

/// Standard tag keys for stream metadata.
///
/// Use these constants as keys in [`TagList`] for interoperability.
pub mod tag_keys {
    /// Track/stream title.
    pub const TITLE: &str = "title";
    /// Artist or performer name.
    pub const ARTIST: &str = "artist";
    /// Album name.
    pub const ALBUM: &str = "album";
    /// Genre.
    pub const GENRE: &str = "genre";
    /// Free-form comment.
    pub const COMMENT: &str = "comment";
    /// Date string (e.g., "2024-01-15").
    pub const DATE: &str = "date";
    /// Track number within album.
    pub const TRACK_NUMBER: &str = "track-number";
    /// Stream duration.
    pub const DURATION: &str = "duration";
    /// Overall bitrate (bits/sec).
    pub const BITRATE: &str = "bitrate";
    /// Codec name (human-readable).
    pub const CODEC: &str = "codec";
    /// Container format name.
    pub const CONTAINER_FORMAT: &str = "container-format";
    /// Video codec name.
    pub const VIDEO_CODEC: &str = "video-codec";
    /// Audio codec name.
    pub const AUDIO_CODEC: &str = "audio-codec";
    /// Cover art / thumbnail (raw bytes).
    pub const IMAGE: &str = "image";
    /// Geographic latitude.
    pub const GEO_LOCATION_LATITUDE: &str = "geo-location-latitude";
    /// Geographic longitude.
    pub const GEO_LOCATION_LONGITUDE: &str = "geo-location-longitude";
    /// Sample rate (Hz).
    pub const SAMPLE_RATE: &str = "sample-rate";
    /// Number of audio channels.
    pub const CHANNELS: &str = "channels";
    /// Video width in pixels.
    pub const VIDEO_WIDTH: &str = "video-width";
    /// Video height in pixels.
    pub const VIDEO_HEIGHT: &str = "video-height";
    /// Video framerate (fps).
    pub const FRAMERATE: &str = "framerate";
    /// Encoder/tool that created the stream.
    pub const ENCODER: &str = "encoder";
    /// Language code (ISO 639).
    pub const LANGUAGE_CODE: &str = "language-code";
}

// ============================================================================
// TagValue
// ============================================================================

/// A typed value in a tag list.
#[derive(Debug, Clone, PartialEq)]
pub enum TagValue {
    /// String value (titles, names, codec IDs).
    String(String),
    /// Unsigned integer (bitrate, track number, dimensions).
    Uint(u64),
    /// Floating-point value (framerate, geo coordinates).
    Float(f64),
    /// Raw bytes (cover art, binary metadata).
    Bytes(Vec<u8>),
    /// Time value (duration).
    ClockTime(ClockTime),
    /// Multiple string values (multiple artists, genres).
    StringList(Vec<String>),
}

impl fmt::Display for TagValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TagValue::String(s) => write!(f, "{s}"),
            TagValue::Uint(n) => write!(f, "{n}"),
            TagValue::Float(v) => write!(f, "{v:.2}"),
            TagValue::Bytes(b) => write!(f, "[{} bytes]", b.len()),
            TagValue::ClockTime(t) => write!(f, "{t}"),
            TagValue::StringList(list) => {
                let joined = list.join(", ");
                write!(f, "{joined}")
            }
        }
    }
}

// ============================================================================
// TagMergeMode
// ============================================================================

/// How to merge tags when combining tag lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TagMergeMode {
    /// Replace existing value with the new one.
    #[default]
    Replace,
    /// Keep existing value, ignore new one.
    Keep,
    /// Append: for string types, convert to `StringList`; otherwise replace.
    Append,
}

// ============================================================================
// TagList
// ============================================================================

/// A collection of stream metadata tags.
///
/// Tags are key-value pairs where keys are strings (see [`tag_keys`]) and
/// values are typed ([`TagValue`]).
///
/// # Example
///
/// ```rust
/// use parallax::pipeline::tags::{TagList, TagValue, tag_keys};
///
/// let mut tags = TagList::new();
/// tags.set(tag_keys::TITLE, TagValue::String("My Song".into()));
/// tags.set(tag_keys::BITRATE, TagValue::Uint(320_000));
///
/// assert_eq!(tags.get_string(tag_keys::TITLE), Some("My Song"));
/// assert_eq!(tags.get_uint(tag_keys::BITRATE), Some(320_000));
/// ```
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TagList {
    tags: HashMap<String, TagValue>,
}

impl TagList {
    /// Create an empty tag list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a tag value, replacing any existing value for this key.
    pub fn set(&mut self, key: impl Into<String>, value: TagValue) {
        self.tags.insert(key.into(), value);
    }

    /// Get a tag value by key.
    pub fn get(&self, key: &str) -> Option<&TagValue> {
        self.tags.get(key)
    }

    /// Get a string tag value.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        match self.tags.get(key) {
            Some(TagValue::String(s)) => Some(s),
            _ => None,
        }
    }

    /// Get an unsigned integer tag value.
    pub fn get_uint(&self, key: &str) -> Option<u64> {
        match self.tags.get(key) {
            Some(TagValue::Uint(n)) => Some(*n),
            _ => None,
        }
    }

    /// Get a float tag value.
    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.tags.get(key) {
            Some(TagValue::Float(v)) => Some(*v),
            _ => None,
        }
    }

    /// Get a bytes tag value.
    pub fn get_bytes(&self, key: &str) -> Option<&[u8]> {
        match self.tags.get(key) {
            Some(TagValue::Bytes(b)) => Some(b),
            _ => None,
        }
    }

    /// Get a ClockTime tag value.
    pub fn get_clock_time(&self, key: &str) -> Option<ClockTime> {
        match self.tags.get(key) {
            Some(TagValue::ClockTime(t)) => Some(*t),
            _ => None,
        }
    }

    /// Remove a tag by key, returning the old value if present.
    pub fn remove(&mut self, key: &str) -> Option<TagValue> {
        self.tags.remove(key)
    }

    /// Check if the tag list contains a key.
    pub fn contains(&self, key: &str) -> bool {
        self.tags.contains_key(key)
    }

    /// Returns true if the tag list is empty.
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Returns the number of tags.
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Iterate over all tags.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TagValue)> {
        self.tags.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Merge another tag list into this one using the given mode.
    pub fn merge(&mut self, other: &TagList, mode: TagMergeMode) {
        for (key, value) in &other.tags {
            match mode {
                TagMergeMode::Replace => {
                    self.tags.insert(key.clone(), value.clone());
                }
                TagMergeMode::Keep => {
                    self.tags.entry(key.clone()).or_insert_with(|| value.clone());
                }
                TagMergeMode::Append => {
                    if let Some(existing) = self.tags.get_mut(key) {
                        // Append strings into a StringList
                        match (existing, value) {
                            (TagValue::String(a), TagValue::String(b)) => {
                                let list = vec![a.clone(), b.clone()];
                                self.tags.insert(key.clone(), TagValue::StringList(list));
                            }
                            (TagValue::StringList(list), TagValue::String(b)) => {
                                list.push(b.clone());
                            }
                            (TagValue::StringList(list), TagValue::StringList(b)) => {
                                list.extend(b.iter().cloned());
                            }
                            // For non-string types, replace
                            (existing, new) => {
                                *existing = new.clone();
                            }
                        }
                    } else {
                        self.tags.insert(key.clone(), value.clone());
                    }
                }
            }
        }
    }
}

impl fmt::Display for TagList {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut first = true;
        // Sort keys for deterministic output
        let mut keys: Vec<&String> = self.tags.keys().collect();
        keys.sort();
        for key in keys {
            if let Some(value) = self.tags.get(key) {
                if !first {
                    write!(f, ", ")?;
                }
                write!(f, "{key}: {value}")?;
                first = false;
            }
        }
        Ok(())
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_list_basic() {
        let mut tags = TagList::new();
        assert!(tags.is_empty());

        tags.set(tag_keys::TITLE, TagValue::String("Test".into()));
        tags.set(tag_keys::BITRATE, TagValue::Uint(128_000));

        assert_eq!(tags.len(), 2);
        assert_eq!(tags.get_string(tag_keys::TITLE), Some("Test"));
        assert_eq!(tags.get_uint(tag_keys::BITRATE), Some(128_000));
        assert!(tags.contains(tag_keys::TITLE));
    }

    #[test]
    fn test_tag_list_remove() {
        let mut tags = TagList::new();
        tags.set("key", TagValue::String("val".into()));
        assert!(tags.contains("key"));
        tags.remove("key");
        assert!(!tags.contains("key"));
    }

    #[test]
    fn test_tag_merge_replace() {
        let mut a = TagList::new();
        a.set("key", TagValue::String("old".into()));

        let mut b = TagList::new();
        b.set("key", TagValue::String("new".into()));
        b.set("extra", TagValue::Uint(42));

        a.merge(&b, TagMergeMode::Replace);
        assert_eq!(a.get_string("key"), Some("new"));
        assert_eq!(a.get_uint("extra"), Some(42));
    }

    #[test]
    fn test_tag_merge_keep() {
        let mut a = TagList::new();
        a.set("key", TagValue::String("original".into()));

        let mut b = TagList::new();
        b.set("key", TagValue::String("ignored".into()));
        b.set("extra", TagValue::Uint(42));

        a.merge(&b, TagMergeMode::Keep);
        assert_eq!(a.get_string("key"), Some("original"));
        assert_eq!(a.get_uint("extra"), Some(42));
    }

    #[test]
    fn test_tag_merge_append() {
        let mut a = TagList::new();
        a.set("artist", TagValue::String("Artist A".into()));

        let mut b = TagList::new();
        b.set("artist", TagValue::String("Artist B".into()));

        a.merge(&b, TagMergeMode::Append);
        match a.get("artist") {
            Some(TagValue::StringList(list)) => {
                assert_eq!(list, &["Artist A", "Artist B"]);
            }
            other => panic!("Expected StringList, got {other:?}"),
        }
    }

    #[test]
    fn test_tag_display() {
        let mut tags = TagList::new();
        tags.set("title", TagValue::String("Song".into()));
        let s = format!("{tags}");
        assert!(s.contains("title: Song"));
    }
}
