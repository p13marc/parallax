//! Tag system for stream metadata.
//!
//! Tags provide metadata about streams such as title, artist, codec info,
//! duration, and other descriptive information.
//!
//! # Common Tags
//!
//! | Tag Name | Type | Description |
//! |----------|------|-------------|
//! | `title` | String | Stream/track title |
//! | `artist` | String | Artist name |
//! | `album` | String | Album name |
//! | `duration` | UInt | Duration in nanoseconds |
//! | `bitrate` | UInt | Bitrate in bits/second |
//! | `codec` | String | Codec name |
//! | `container` | String | Container format |
//! | `video-codec` | String | Video codec name |
//! | `audio-codec` | String | Audio codec name |
//!
//! # Example
//!
//! ```rust
//! use parallax::event::{TagList, TagValue};
//!
//! let mut tags = TagList::new();
//! tags.set_title("My Video");
//! tags.set("artist", "Parallax");
//! tags.set("bitrate", 1_500_000u64);
//!
//! assert_eq!(tags.title(), Some("My Video"));
//! assert_eq!(tags.get_string("artist"), Some("Parallax"));
//! ```

use crate::clock::ClockTime;
use std::collections::HashMap;

// ============================================================================
// Tag Value
// ============================================================================

/// Value that can be stored in a tag list.
///
/// Tags support various types for flexibility in storing metadata.
#[derive(Debug, Clone, PartialEq)]
pub enum TagValue {
    /// String value.
    String(String),
    /// Unsigned integer (used for durations, bitrates, etc.).
    UInt(u64),
    /// Signed integer.
    Int(i64),
    /// Floating point value.
    Double(f64),
    /// Boolean value.
    Bool(bool),
    /// Date/time as ISO 8601 string.
    DateTime(String),
    /// Binary data.
    Binary(Vec<u8>),
    /// List of values (for multi-value tags).
    List(Vec<TagValue>),
}

impl TagValue {
    /// Get as string if this is a String variant.
    pub fn as_string(&self) -> Option<&str> {
        match self {
            TagValue::String(s) => Some(s),
            _ => None,
        }
    }

    /// Get as u64 if this is a UInt variant.
    pub fn as_uint(&self) -> Option<u64> {
        match self {
            TagValue::UInt(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as i64 if this is an Int variant.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            TagValue::Int(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as f64 if this is a Double variant.
    pub fn as_double(&self) -> Option<f64> {
        match self {
            TagValue::Double(n) => Some(*n),
            _ => None,
        }
    }

    /// Get as bool if this is a Bool variant.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            TagValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Get as bytes if this is a Binary variant.
    pub fn as_binary(&self) -> Option<&[u8]> {
        match self {
            TagValue::Binary(b) => Some(b),
            _ => None,
        }
    }

    /// Get as list if this is a List variant.
    pub fn as_list(&self) -> Option<&[TagValue]> {
        match self {
            TagValue::List(l) => Some(l),
            _ => None,
        }
    }
}

// Conversions from common types
impl From<String> for TagValue {
    fn from(s: String) -> Self {
        TagValue::String(s)
    }
}

impl From<&str> for TagValue {
    fn from(s: &str) -> Self {
        TagValue::String(s.to_string())
    }
}

impl From<u64> for TagValue {
    fn from(n: u64) -> Self {
        TagValue::UInt(n)
    }
}

impl From<u32> for TagValue {
    fn from(n: u32) -> Self {
        TagValue::UInt(n as u64)
    }
}

impl From<i64> for TagValue {
    fn from(n: i64) -> Self {
        TagValue::Int(n)
    }
}

impl From<i32> for TagValue {
    fn from(n: i32) -> Self {
        TagValue::Int(n as i64)
    }
}

impl From<f64> for TagValue {
    fn from(n: f64) -> Self {
        TagValue::Double(n)
    }
}

impl From<f32> for TagValue {
    fn from(n: f32) -> Self {
        TagValue::Double(n as f64)
    }
}

impl From<bool> for TagValue {
    fn from(b: bool) -> Self {
        TagValue::Bool(b)
    }
}

impl From<Vec<u8>> for TagValue {
    fn from(b: Vec<u8>) -> Self {
        TagValue::Binary(b)
    }
}

impl From<&[u8]> for TagValue {
    fn from(b: &[u8]) -> Self {
        TagValue::Binary(b.to_vec())
    }
}

impl From<Vec<TagValue>> for TagValue {
    fn from(l: Vec<TagValue>) -> Self {
        TagValue::List(l)
    }
}

// ============================================================================
// Tag List
// ============================================================================

/// A collection of stream metadata tags.
///
/// Tags are key-value pairs where keys are strings and values can be
/// various types (string, number, binary, etc.).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TagList {
    tags: HashMap<String, TagValue>,
}

impl TagList {
    /// Create a new empty tag list.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a tag value.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<TagValue>) {
        self.tags.insert(key.into(), value.into());
    }

    /// Get a tag value.
    pub fn get(&self, key: &str) -> Option<&TagValue> {
        self.tags.get(key)
    }

    /// Get a tag as a string.
    pub fn get_string(&self, key: &str) -> Option<&str> {
        self.get(key).and_then(TagValue::as_string)
    }

    /// Get a tag as a u64.
    pub fn get_uint(&self, key: &str) -> Option<u64> {
        self.get(key).and_then(TagValue::as_uint)
    }

    /// Get a tag as an i64.
    pub fn get_int(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(TagValue::as_int)
    }

    /// Get a tag as a f64.
    pub fn get_double(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(TagValue::as_double)
    }

    /// Get a tag as a bool.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(TagValue::as_bool)
    }

    /// Remove a tag.
    pub fn remove(&mut self, key: &str) -> Option<TagValue> {
        self.tags.remove(key)
    }

    /// Check if a tag exists.
    pub fn contains(&self, key: &str) -> bool {
        self.tags.contains_key(key)
    }

    /// Get the number of tags.
    pub fn len(&self) -> usize {
        self.tags.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.tags.is_empty()
    }

    /// Iterate over all tags.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &TagValue)> {
        self.tags.iter().map(|(k, v)| (k.as_str(), v))
    }

    /// Get all tag names.
    pub fn keys(&self) -> impl Iterator<Item = &str> {
        self.tags.keys().map(String::as_str)
    }

    /// Merge another tag list into this one.
    pub fn merge(&mut self, other: &TagList, mode: TagMergeMode) {
        match mode {
            TagMergeMode::Replace => {
                self.tags = other.tags.clone();
            }
            TagMergeMode::Append => {
                for (k, v) in &other.tags {
                    self.tags.insert(k.clone(), v.clone());
                }
            }
            TagMergeMode::Keep => {
                for (k, v) in &other.tags {
                    self.tags.entry(k.clone()).or_insert_with(|| v.clone());
                }
            }
            TagMergeMode::ReplaceMatching => {
                for (k, v) in &other.tags {
                    if self.tags.contains_key(k) {
                        self.tags.insert(k.clone(), v.clone());
                    }
                }
            }
        }
    }

    // ========================================================================
    // Common Tag Accessors
    // ========================================================================

    /// Get the title tag.
    pub fn title(&self) -> Option<&str> {
        self.get_string("title")
    }

    /// Set the title tag.
    pub fn set_title(&mut self, title: impl Into<String>) {
        self.set("title", title.into());
    }

    /// Get the artist tag.
    pub fn artist(&self) -> Option<&str> {
        self.get_string("artist")
    }

    /// Set the artist tag.
    pub fn set_artist(&mut self, artist: impl Into<String>) {
        self.set("artist", artist.into());
    }

    /// Get the album tag.
    pub fn album(&self) -> Option<&str> {
        self.get_string("album")
    }

    /// Set the album tag.
    pub fn set_album(&mut self, album: impl Into<String>) {
        self.set("album", album.into());
    }

    /// Get the duration tag as ClockTime.
    pub fn duration(&self) -> Option<ClockTime> {
        self.get_uint("duration").map(ClockTime::from_nanos)
    }

    /// Set the duration tag.
    pub fn set_duration(&mut self, duration: ClockTime) {
        self.set("duration", duration.nanos() as u64);
    }

    /// Get the bitrate tag (bits per second).
    pub fn bitrate(&self) -> Option<u64> {
        self.get_uint("bitrate")
    }

    /// Set the bitrate tag.
    pub fn set_bitrate(&mut self, bitrate: u64) {
        self.set("bitrate", bitrate);
    }

    /// Get the codec tag.
    pub fn codec(&self) -> Option<&str> {
        self.get_string("codec")
    }

    /// Set the codec tag.
    pub fn set_codec(&mut self, codec: impl Into<String>) {
        self.set("codec", codec.into());
    }

    /// Get the video codec tag.
    pub fn video_codec(&self) -> Option<&str> {
        self.get_string("video-codec")
    }

    /// Set the video codec tag.
    pub fn set_video_codec(&mut self, codec: impl Into<String>) {
        self.set("video-codec", codec.into());
    }

    /// Get the audio codec tag.
    pub fn audio_codec(&self) -> Option<&str> {
        self.get_string("audio-codec")
    }

    /// Set the audio codec tag.
    pub fn set_audio_codec(&mut self, codec: impl Into<String>) {
        self.set("audio-codec", codec.into());
    }

    /// Get the container format tag.
    pub fn container(&self) -> Option<&str> {
        self.get_string("container")
    }

    /// Set the container format tag.
    pub fn set_container(&mut self, container: impl Into<String>) {
        self.set("container", container.into());
    }

    /// Get the language tag (ISO 639 code).
    pub fn language(&self) -> Option<&str> {
        self.get_string("language")
    }

    /// Set the language tag.
    pub fn set_language(&mut self, language: impl Into<String>) {
        self.set("language", language.into());
    }

    /// Get the description/comment tag.
    pub fn description(&self) -> Option<&str> {
        self.get_string("description")
    }

    /// Set the description/comment tag.
    pub fn set_description(&mut self, description: impl Into<String>) {
        self.set("description", description.into());
    }

    /// Get the date tag (ISO 8601).
    pub fn date(&self) -> Option<&str> {
        self.get_string("date")
    }

    /// Set the date tag.
    pub fn set_date(&mut self, date: impl Into<String>) {
        self.set("date", date.into());
    }

    /// Get the track number.
    pub fn track_number(&self) -> Option<u64> {
        self.get_uint("track-number")
    }

    /// Set the track number.
    pub fn set_track_number(&mut self, number: u64) {
        self.set("track-number", number);
    }

    /// Get the total track count.
    pub fn track_count(&self) -> Option<u64> {
        self.get_uint("track-count")
    }

    /// Set the total track count.
    pub fn set_track_count(&mut self, count: u64) {
        self.set("track-count", count);
    }
}

// ============================================================================
// Tag Merge Mode
// ============================================================================

/// How to merge tag lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum TagMergeMode {
    /// Replace all existing tags with new ones.
    Replace,
    /// Add new tags, replace existing ones with same key.
    #[default]
    Append,
    /// Add new tags, keep existing ones if key exists.
    Keep,
    /// Only replace tags where key already exists.
    ReplaceMatching,
}

// ============================================================================
// Common Tag Constants
// ============================================================================

/// Common tag names as constants for convenience.
pub mod tag_names {
    /// Stream/track title.
    pub const TITLE: &str = "title";
    /// Artist name.
    pub const ARTIST: &str = "artist";
    /// Album name.
    pub const ALBUM: &str = "album";
    /// Duration in nanoseconds.
    pub const DURATION: &str = "duration";
    /// Bitrate in bits/second.
    pub const BITRATE: &str = "bitrate";
    /// Codec name.
    pub const CODEC: &str = "codec";
    /// Video codec name.
    pub const VIDEO_CODEC: &str = "video-codec";
    /// Audio codec name.
    pub const AUDIO_CODEC: &str = "audio-codec";
    /// Container format.
    pub const CONTAINER: &str = "container";
    /// Language (ISO 639 code).
    pub const LANGUAGE: &str = "language";
    /// Description/comment.
    pub const DESCRIPTION: &str = "description";
    /// Date (ISO 8601).
    pub const DATE: &str = "date";
    /// Track number.
    pub const TRACK_NUMBER: &str = "track-number";
    /// Total track count.
    pub const TRACK_COUNT: &str = "track-count";
    /// Sample rate (audio).
    pub const SAMPLE_RATE: &str = "sample-rate";
    /// Number of channels (audio).
    pub const CHANNELS: &str = "channels";
    /// Width (video).
    pub const WIDTH: &str = "width";
    /// Height (video).
    pub const HEIGHT: &str = "height";
    /// Frame rate (video).
    pub const FRAMERATE: &str = "framerate";
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tag_value_conversions() {
        let v: TagValue = "hello".into();
        assert_eq!(v.as_string(), Some("hello"));

        let v: TagValue = 42u64.into();
        assert_eq!(v.as_uint(), Some(42));

        let v: TagValue = 3.14f64.into();
        assert!((v.as_double().unwrap() - 3.14).abs() < 0.001);

        let v: TagValue = true.into();
        assert_eq!(v.as_bool(), Some(true));
    }

    #[test]
    fn test_tag_list_basic() {
        let mut tags = TagList::new();
        tags.set("title", "My Song");
        tags.set("bitrate", 320_000u64);

        assert_eq!(tags.get_string("title"), Some("My Song"));
        assert_eq!(tags.get_uint("bitrate"), Some(320_000));
        assert_eq!(tags.len(), 2);
    }

    #[test]
    fn test_tag_list_common_accessors() {
        let mut tags = TagList::new();
        tags.set_title("Test Title");
        tags.set_artist("Test Artist");
        tags.set_duration(ClockTime::from_secs(180));

        assert_eq!(tags.title(), Some("Test Title"));
        assert_eq!(tags.artist(), Some("Test Artist"));
        assert_eq!(tags.duration(), Some(ClockTime::from_secs(180)));
    }

    #[test]
    fn test_tag_merge_append() {
        let mut tags1 = TagList::new();
        tags1.set("a", "1");
        tags1.set("b", "2");

        let mut tags2 = TagList::new();
        tags2.set("b", "new");
        tags2.set("c", "3");

        tags1.merge(&tags2, TagMergeMode::Append);

        assert_eq!(tags1.get_string("a"), Some("1"));
        assert_eq!(tags1.get_string("b"), Some("new")); // replaced
        assert_eq!(tags1.get_string("c"), Some("3"));
    }

    #[test]
    fn test_tag_merge_keep() {
        let mut tags1 = TagList::new();
        tags1.set("a", "1");
        tags1.set("b", "2");

        let mut tags2 = TagList::new();
        tags2.set("b", "new");
        tags2.set("c", "3");

        tags1.merge(&tags2, TagMergeMode::Keep);

        assert_eq!(tags1.get_string("a"), Some("1"));
        assert_eq!(tags1.get_string("b"), Some("2")); // kept
        assert_eq!(tags1.get_string("c"), Some("3"));
    }

    #[test]
    fn test_tag_merge_replace() {
        let mut tags1 = TagList::new();
        tags1.set("a", "1");
        tags1.set("b", "2");

        let mut tags2 = TagList::new();
        tags2.set("c", "3");

        tags1.merge(&tags2, TagMergeMode::Replace);

        assert!(!tags1.contains("a"));
        assert!(!tags1.contains("b"));
        assert_eq!(tags1.get_string("c"), Some("3"));
    }

    #[test]
    fn test_tag_iteration() {
        let mut tags = TagList::new();
        tags.set("a", "1");
        tags.set("b", "2");

        let keys: Vec<_> = tags.keys().collect();
        assert_eq!(keys.len(), 2);
        assert!(keys.contains(&"a"));
        assert!(keys.contains(&"b"));
    }
}
