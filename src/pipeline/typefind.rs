//! Media type detection from byte signatures.
//!
//! The typefind system detects media formats by examining the first bytes
//! of a stream, similar to GStreamer's typefinding.
//!
//! # Example
//!
//! ```rust
//! use parallax::pipeline::typefind::{TypeFindRegistry, MediaType, TypeFindProbability};
//!
//! let registry = TypeFindRegistry::with_builtins();
//!
//! // Detect from bytes (MP4 ftyp box)
//! let mp4_header = b"\x00\x00\x00\x20ftypisom";
//! let result = registry.detect(mp4_header).unwrap();
//! assert_eq!(result.media_type, MediaType::Mp4);
//!
//! // Detect from file extension
//! let media_type = registry.detect_from_extension("mkv").unwrap();
//! assert_eq!(media_type, MediaType::Matroska);
//! ```

// ============================================================================
// Media Types
// ============================================================================

/// Known media types that can be detected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MediaType {
    // Containers
    /// MP4/MOV/M4V container.
    Mp4,
    /// Matroska container (MKV/MKA).
    Matroska,
    /// WebM container (subset of Matroska).
    WebM,
    /// MPEG Transport Stream.
    MpegTs,
    /// Flash Video.
    Flv,
    /// AVI container.
    Avi,
    /// Ogg container.
    Ogg,
    /// WAV audio container.
    Wav,

    // Video codecs
    /// H.264/AVC (Annex B bytestream).
    H264,
    /// H.265/HEVC (Annex B bytestream).
    H265,
    /// AV1 video.
    Av1,
    /// VP8 video.
    Vp8,
    /// VP9 video.
    Vp9,

    // Audio codecs/formats
    /// MP3 audio.
    Mp3,
    /// FLAC audio.
    Flac,
    /// AAC audio.
    Aac,
    /// Opus audio.
    Opus,

    // Image formats
    /// JPEG image.
    Jpeg,
    /// PNG image.
    Png,

    // Other
    /// Custom/unknown type with MIME string.
    Custom(String),
}

impl std::fmt::Display for MediaType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MediaType::Mp4 => write!(f, "video/mp4"),
            MediaType::Matroska => write!(f, "video/x-matroska"),
            MediaType::WebM => write!(f, "video/webm"),
            MediaType::MpegTs => write!(f, "video/mpegts"),
            MediaType::Flv => write!(f, "video/x-flv"),
            MediaType::Avi => write!(f, "video/x-msvideo"),
            MediaType::Ogg => write!(f, "application/ogg"),
            MediaType::Wav => write!(f, "audio/wav"),
            MediaType::H264 => write!(f, "video/h264"),
            MediaType::H265 => write!(f, "video/h265"),
            MediaType::Av1 => write!(f, "video/av1"),
            MediaType::Vp8 => write!(f, "video/vp8"),
            MediaType::Vp9 => write!(f, "video/vp9"),
            MediaType::Mp3 => write!(f, "audio/mpeg"),
            MediaType::Flac => write!(f, "audio/flac"),
            MediaType::Aac => write!(f, "audio/aac"),
            MediaType::Opus => write!(f, "audio/opus"),
            MediaType::Jpeg => write!(f, "image/jpeg"),
            MediaType::Png => write!(f, "image/png"),
            MediaType::Custom(mime) => write!(f, "{mime}"),
        }
    }
}

// ============================================================================
// Detection Confidence
// ============================================================================

/// Confidence level of type detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TypeFindProbability {
    /// Possible match (weak signature).
    Possible = 1,
    /// Likely match (partial signature).
    Likely = 2,
    /// Near certain (full magic bytes match).
    NearCertain = 3,
    /// Maximum confidence (complete header validation).
    Maximum = 4,
}

/// Result of type detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeFindResult {
    /// Detected media type.
    pub media_type: MediaType,
    /// Confidence level.
    pub probability: TypeFindProbability,
}

// ============================================================================
// TypeFinder
// ============================================================================

/// A registered type detection function for a specific media type.
pub struct TypeFinder {
    /// The media type this finder detects.
    pub media_type: MediaType,
    /// File extensions associated with this type.
    pub extensions: Vec<&'static str>,
    /// Detection function: given first N bytes, return confidence or None.
    pub detect: Box<dyn Fn(&[u8]) -> Option<TypeFindProbability> + Send + Sync>,
}

impl std::fmt::Debug for TypeFinder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeFinder")
            .field("media_type", &self.media_type)
            .field("extensions", &self.extensions)
            .finish_non_exhaustive()
    }
}

// ============================================================================
// TypeFindRegistry
// ============================================================================

/// Registry of type detection functions.
///
/// Use [`with_builtins()`](TypeFindRegistry::with_builtins) to create a
/// registry with all built-in detectors pre-registered.
pub struct TypeFindRegistry {
    finders: Vec<TypeFinder>,
}

impl TypeFindRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            finders: Vec::new(),
        }
    }

    /// Create a registry with all built-in detectors.
    pub fn with_builtins() -> Self {
        let mut registry = Self::new();
        register_builtin_types(&mut registry);
        registry
    }

    /// Register a custom type finder.
    pub fn register(&mut self, finder: TypeFinder) {
        self.finders.push(finder);
    }

    /// Detect media type from the first bytes of a stream.
    ///
    /// Returns the best match (highest probability). Returns `None` if
    /// no detector matched.
    pub fn detect(&self, data: &[u8]) -> Option<TypeFindResult> {
        let mut best: Option<TypeFindResult> = None;

        for finder in &self.finders {
            if let Some(prob) = (finder.detect)(data)
                && best.as_ref().is_none_or(|b| prob > b.probability)
            {
                best = Some(TypeFindResult {
                    media_type: finder.media_type.clone(),
                    probability: prob,
                });
            }
        }

        best
    }

    /// Detect media type from a file extension.
    ///
    /// Returns the first matching type. Case-insensitive.
    pub fn detect_from_extension(&self, ext: &str) -> Option<MediaType> {
        let ext_lower = ext.to_lowercase();
        for finder in &self.finders {
            if finder.extensions.iter().any(|e| *e == ext_lower) {
                return Some(finder.media_type.clone());
            }
        }
        None
    }

    /// Detect from either bytes or file extension, preferring byte detection.
    pub fn detect_with_fallback(
        &self,
        data: &[u8],
        extension: Option<&str>,
    ) -> Option<TypeFindResult> {
        // Try byte detection first
        if let Some(result) = self.detect(data) {
            return Some(result);
        }
        // Fall back to extension
        if let Some(ext) = extension
            && let Some(media_type) = self.detect_from_extension(ext)
        {
            return Some(TypeFindResult {
                media_type,
                probability: TypeFindProbability::Likely,
            });
        }
        None
    }

    /// Get all registered media types.
    pub fn media_types(&self) -> Vec<&MediaType> {
        self.finders.iter().map(|f| &f.media_type).collect()
    }

    /// Number of registered type finders.
    pub fn len(&self) -> usize {
        self.finders.len()
    }

    /// Whether the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.finders.is_empty()
    }
}

impl Default for TypeFindRegistry {
    fn default() -> Self {
        Self::with_builtins()
    }
}

impl std::fmt::Debug for TypeFindRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypeFindRegistry")
            .field("finders", &self.finders.len())
            .finish()
    }
}

// ============================================================================
// Built-in Type Detectors
// ============================================================================

fn register_builtin_types(registry: &mut TypeFindRegistry) {
    // MP4/MOV: ftyp box at offset 4
    registry.register(TypeFinder {
        media_type: MediaType::Mp4,
        extensions: vec!["mp4", "m4v", "m4a", "mov", "3gp"],
        detect: Box::new(|data| {
            if data.len() < 8 {
                return None;
            }
            if &data[4..8] == b"ftyp" {
                Some(TypeFindProbability::Maximum)
            } else {
                None
            }
        }),
    });

    // Matroska/WebM: EBML header 0x1A45DFA3
    registry.register(TypeFinder {
        media_type: MediaType::Matroska,
        extensions: vec!["mkv", "mka", "webm"],
        detect: Box::new(|data| {
            if data.len() < 4 {
                return None;
            }
            if data[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
                Some(TypeFindProbability::Maximum)
            } else {
                None
            }
        }),
    });

    // MPEG-TS: sync byte 0x47 at 188-byte intervals
    registry.register(TypeFinder {
        media_type: MediaType::MpegTs,
        extensions: vec!["ts", "mts", "m2ts"],
        detect: Box::new(|data| {
            if data.len() < 188 * 3 {
                return None;
            }
            let sync_count = (0..3).filter(|&i| data[i * 188] == 0x47).count();
            match sync_count {
                3 => Some(TypeFindProbability::Maximum),
                2 => Some(TypeFindProbability::Likely),
                _ => None,
            }
        }),
    });

    // FLV: "FLV" + version byte
    registry.register(TypeFinder {
        media_type: MediaType::Flv,
        extensions: vec!["flv"],
        detect: Box::new(|data| {
            if data.len() < 4 {
                return None;
            }
            if &data[0..3] == b"FLV" {
                Some(TypeFindProbability::Maximum)
            } else {
                None
            }
        }),
    });

    // AVI: RIFF....AVI
    registry.register(TypeFinder {
        media_type: MediaType::Avi,
        extensions: vec!["avi"],
        detect: Box::new(|data| {
            if data.len() < 12 {
                return None;
            }
            if &data[0..4] == b"RIFF" && &data[8..12] == b"AVI " {
                Some(TypeFindProbability::Maximum)
            } else {
                None
            }
        }),
    });

    // WAV: RIFF....WAVE
    registry.register(TypeFinder {
        media_type: MediaType::Wav,
        extensions: vec!["wav"],
        detect: Box::new(|data| {
            if data.len() < 12 {
                return None;
            }
            if &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE" {
                Some(TypeFindProbability::Maximum)
            } else {
                None
            }
        }),
    });

    // Ogg: OggS magic
    registry.register(TypeFinder {
        media_type: MediaType::Ogg,
        extensions: vec!["ogg", "oga", "ogv", "opus"],
        detect: Box::new(|data| {
            if data.len() < 4 {
                return None;
            }
            if &data[0..4] == b"OggS" {
                Some(TypeFindProbability::Maximum)
            } else {
                None
            }
        }),
    });

    // FLAC: fLaC magic
    registry.register(TypeFinder {
        media_type: MediaType::Flac,
        extensions: vec!["flac"],
        detect: Box::new(|data| {
            if data.len() < 4 {
                return None;
            }
            if &data[0..4] == b"fLaC" {
                Some(TypeFindProbability::Maximum)
            } else {
                None
            }
        }),
    });

    // PNG: 89 50 4E 47 0D 0A 1A 0A
    registry.register(TypeFinder {
        media_type: MediaType::Png,
        extensions: vec!["png"],
        detect: Box::new(|data| {
            if data.len() < 8 {
                return None;
            }
            if data[0..8] == [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A] {
                Some(TypeFindProbability::Maximum)
            } else {
                None
            }
        }),
    });

    // JPEG: FF D8 FF
    registry.register(TypeFinder {
        media_type: MediaType::Jpeg,
        extensions: vec!["jpg", "jpeg"],
        detect: Box::new(|data| {
            if data.len() < 3 {
                return None;
            }
            if data[0..3] == [0xFF, 0xD8, 0xFF] {
                Some(TypeFindProbability::Maximum)
            } else {
                None
            }
        }),
    });

    // H.264 Annex B: 00 00 00 01 or 00 00 01
    registry.register(TypeFinder {
        media_type: MediaType::H264,
        extensions: vec!["h264", "264"],
        detect: Box::new(|data| {
            if data.len() < 5 {
                return None;
            }
            if data[0..4] == [0, 0, 0, 1] || data[0..3] == [0, 0, 1] {
                let nal_offset = if data[2] == 1 { 3 } else { 4 };
                if nal_offset < data.len() {
                    let nal_type = data[nal_offset] & 0x1F;
                    if nal_type == 7 || nal_type == 8 {
                        return Some(TypeFindProbability::Likely);
                    }
                }
                Some(TypeFindProbability::Possible)
            } else {
                None
            }
        }),
    });

    // MP3: ID3 tag or frame sync
    registry.register(TypeFinder {
        media_type: MediaType::Mp3,
        extensions: vec!["mp3"],
        detect: Box::new(|data| {
            if data.len() < 3 {
                return None;
            }
            // ID3v2 tag
            if &data[0..3] == b"ID3" {
                return Some(TypeFindProbability::Maximum);
            }
            // Frame sync (0xFF followed by 0xE0-0xFF)
            if data.len() >= 4 && data[0] == 0xFF && (data[1] & 0xE0) == 0xE0 {
                Some(TypeFindProbability::Likely)
            } else {
                None
            }
        }),
    });
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn registry() -> TypeFindRegistry {
        TypeFindRegistry::with_builtins()
    }

    #[test]
    fn test_detect_mp4() {
        let r = registry();
        let data = b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00";
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Mp4);
        assert_eq!(result.probability, TypeFindProbability::Maximum);
    }

    #[test]
    fn test_detect_matroska() {
        let r = registry();
        let data = &[0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00];
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Matroska);
    }

    #[test]
    fn test_detect_mpeg_ts() {
        let r = registry();
        let mut data = vec![0u8; 188 * 3];
        data[0] = 0x47;
        data[188] = 0x47;
        data[376] = 0x47;
        let result = r.detect(&data).unwrap();
        assert_eq!(result.media_type, MediaType::MpegTs);
        assert_eq!(result.probability, TypeFindProbability::Maximum);
    }

    #[test]
    fn test_detect_wav() {
        let r = registry();
        let data = b"RIFF\x00\x00\x00\x00WAVEfmt ";
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Wav);
    }

    #[test]
    fn test_detect_flac() {
        let r = registry();
        let data = b"fLaC\x00\x00\x00\x22";
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Flac);
    }

    #[test]
    fn test_detect_png() {
        let r = registry();
        let data = &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Png);
    }

    #[test]
    fn test_detect_jpeg() {
        let r = registry();
        let data = &[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Jpeg);
    }

    #[test]
    fn test_detect_mp3_id3() {
        let r = registry();
        let data = b"ID3\x03\x00\x00\x00\x00\x00\x00";
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Mp3);
        assert_eq!(result.probability, TypeFindProbability::Maximum);
    }

    #[test]
    fn test_detect_mp3_frame_sync() {
        let r = registry();
        let data = &[0xFF, 0xFB, 0x90, 0x64];
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Mp3);
        assert_eq!(result.probability, TypeFindProbability::Likely);
    }

    #[test]
    fn test_detect_flv() {
        let r = registry();
        let data = b"FLV\x01\x05\x00\x00\x00\x09";
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Flv);
    }

    #[test]
    fn test_detect_avi() {
        let r = registry();
        let data = b"RIFF\x00\x00\x00\x00AVI LIST";
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Avi);
    }

    #[test]
    fn test_detect_ogg() {
        let r = registry();
        let data = b"OggS\x00\x02\x00\x00";
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::Ogg);
    }

    #[test]
    fn test_detect_h264() {
        let r = registry();
        // NAL start code + SPS (type 7)
        let data = &[0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00];
        let result = r.detect(data).unwrap();
        assert_eq!(result.media_type, MediaType::H264);
        assert_eq!(result.probability, TypeFindProbability::Likely);
    }

    #[test]
    fn test_detect_unknown() {
        let r = registry();
        let data = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
        assert!(r.detect(data).is_none());
    }

    #[test]
    fn test_detect_too_short() {
        let r = registry();
        assert!(r.detect(b"").is_none());
        assert!(r.detect(b"ab").is_none());
    }

    #[test]
    fn test_detect_from_extension() {
        let r = registry();
        assert_eq!(r.detect_from_extension("mp4"), Some(MediaType::Mp4));
        assert_eq!(r.detect_from_extension("mkv"), Some(MediaType::Matroska));
        assert_eq!(r.detect_from_extension("ts"), Some(MediaType::MpegTs));
        assert_eq!(r.detect_from_extension("wav"), Some(MediaType::Wav));
        assert_eq!(r.detect_from_extension("flac"), Some(MediaType::Flac));
        assert_eq!(r.detect_from_extension("png"), Some(MediaType::Png));
        assert_eq!(r.detect_from_extension("jpg"), Some(MediaType::Jpeg));
        assert_eq!(r.detect_from_extension("mp3"), Some(MediaType::Mp3));
        assert!(r.detect_from_extension("xyz").is_none());
    }

    #[test]
    fn test_detect_with_fallback() {
        let r = registry();

        // Byte detection succeeds
        let data = b"fLaC\x00\x00\x00\x22";
        let result = r.detect_with_fallback(data, Some("mp3")).unwrap();
        assert_eq!(result.media_type, MediaType::Flac); // Bytes win over extension

        // Byte detection fails, extension fallback
        let data = &[0xDE, 0xAD, 0xBE, 0xEF];
        let result = r.detect_with_fallback(data, Some("mp4")).unwrap();
        assert_eq!(result.media_type, MediaType::Mp4);
        assert_eq!(result.probability, TypeFindProbability::Likely);

        // Both fail
        assert!(r.detect_with_fallback(data, Some("xyz")).is_none());
    }

    #[test]
    fn test_media_type_display() {
        assert_eq!(format!("{}", MediaType::Mp4), "video/mp4");
        assert_eq!(format!("{}", MediaType::Flac), "audio/flac");
        assert_eq!(format!("{}", MediaType::Png), "image/png");
    }

    #[test]
    fn test_registry_len() {
        let r = registry();
        assert!(r.len() >= 12); // At least 12 built-in detectors
        assert!(!r.is_empty());
    }
}
