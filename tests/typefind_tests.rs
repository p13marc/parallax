//! Integration tests for media type detection.

use parallax::pipeline::typefind::{MediaType, TypeFindProbability, TypeFindRegistry};

/// Test that the default registry has a reasonable number of detectors.
#[test]
fn test_registry_has_builtins() {
    let registry = TypeFindRegistry::with_builtins();
    assert!(registry.len() >= 12, "Expected at least 12 built-in detectors, got {}", registry.len());
}

/// Test detection of all major container formats.
#[test]
fn test_detect_containers() {
    let r = TypeFindRegistry::with_builtins();

    // MP4
    let result = r.detect(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00").unwrap();
    assert_eq!(result.media_type, MediaType::Mp4);

    // Matroska
    let result = r.detect(&[0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00]).unwrap();
    assert_eq!(result.media_type, MediaType::Matroska);

    // WAV
    let result = r.detect(b"RIFF\x00\x00\x00\x00WAVEfmt ").unwrap();
    assert_eq!(result.media_type, MediaType::Wav);

    // FLV
    let result = r.detect(b"FLV\x01\x05\x00\x00\x00\x09").unwrap();
    assert_eq!(result.media_type, MediaType::Flv);

    // AVI
    let result = r.detect(b"RIFF\x00\x00\x00\x00AVI LIST").unwrap();
    assert_eq!(result.media_type, MediaType::Avi);

    // Ogg
    let result = r.detect(b"OggS\x00\x02\x00\x00").unwrap();
    assert_eq!(result.media_type, MediaType::Ogg);
}

/// Test detection of audio formats.
#[test]
fn test_detect_audio() {
    let r = TypeFindRegistry::with_builtins();

    // FLAC
    let result = r.detect(b"fLaC\x00\x00\x00\x22").unwrap();
    assert_eq!(result.media_type, MediaType::Flac);

    // MP3 (ID3)
    let result = r.detect(b"ID3\x03\x00\x00\x00\x00\x00\x00").unwrap();
    assert_eq!(result.media_type, MediaType::Mp3);
    assert_eq!(result.probability, TypeFindProbability::Maximum);

    // MP3 (frame sync)
    let result = r.detect(&[0xFF, 0xFB, 0x90, 0x64]).unwrap();
    assert_eq!(result.media_type, MediaType::Mp3);
    assert_eq!(result.probability, TypeFindProbability::Likely);
}

/// Test detection of image formats.
#[test]
fn test_detect_images() {
    let r = TypeFindRegistry::with_builtins();

    // PNG
    let result = r.detect(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]).unwrap();
    assert_eq!(result.media_type, MediaType::Png);

    // JPEG
    let result = r.detect(&[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]).unwrap();
    assert_eq!(result.media_type, MediaType::Jpeg);
}

/// Test extension fallback.
#[test]
fn test_extension_fallback() {
    let r = TypeFindRegistry::with_builtins();

    // Unknown bytes but known extension
    let unknown_data = &[0xDE, 0xAD, 0xBE, 0xEF, 0x00, 0x00, 0x00, 0x00];
    let result = r.detect_with_fallback(unknown_data, Some("mp4")).unwrap();
    assert_eq!(result.media_type, MediaType::Mp4);
    assert_eq!(result.probability, TypeFindProbability::Likely);

    // Unknown bytes AND unknown extension
    assert!(r.detect_with_fallback(unknown_data, Some("xyz")).is_none());
}

/// Test MediaType display produces MIME types.
#[test]
fn test_media_type_mime() {
    assert_eq!(format!("{}", MediaType::Mp4), "video/mp4");
    assert_eq!(format!("{}", MediaType::Flac), "audio/flac");
    assert_eq!(format!("{}", MediaType::Png), "image/png");
    assert_eq!(format!("{}", MediaType::MpegTs), "video/mpegts");
}
