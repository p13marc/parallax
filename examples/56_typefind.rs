//! # Media Type Detection (TypeFind)
//!
//! Demonstrates automatic detection of media formats from byte signatures.
//! The TypeFindRegistry can identify 14 formats by their magic bytes,
//! with file extension fallback.
//!
//! Run: `cargo run --example 56_typefind`

use parallax::pipeline::typefind::{TypeFindProbability, TypeFindRegistry};

fn main() {
    let registry = TypeFindRegistry::with_builtins();

    println!("=== Media Type Detection ===\n");
    println!("Registered detectors: {}\n", registry.len());

    // Test various byte signatures
    let samples: &[(&str, &[u8])] = &[
        ("MP4 (ftyp)", b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00"),
        ("Matroska (EBML)", &[0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00]),
        ("FLAC", b"fLaC\x00\x00\x00\x22"),
        ("PNG", &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]),
        ("JPEG", &[0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10]),
        ("WAV", b"RIFF\x00\x00\x00\x00WAVEfmt "),
        ("Ogg", b"OggS\x00\x02\x00\x00"),
        ("FLV", b"FLV\x01\x05\x00\x00\x00\x09"),
        ("MP3 (ID3)", b"ID3\x03\x00\x00\x00\x00\x00\x00"),
        ("Unknown", &[0xDE, 0xAD, 0xBE, 0xEF]),
    ];

    println!("{:<20} {:<20} {:<15}", "Input", "Detected", "Confidence");
    println!("{}", "-".repeat(55));

    for (name, data) in samples {
        match registry.detect(data) {
            Some(result) => {
                let confidence = match result.probability {
                    TypeFindProbability::Possible => "Possible",
                    TypeFindProbability::Likely => "Likely",
                    TypeFindProbability::NearCertain => "NearCertain",
                    TypeFindProbability::Maximum => "Maximum",
                };
                println!("{:<20} {:<20} {:<15}", name, result.media_type, confidence);
            }
            None => {
                println!("{:<20} {:<20} {:<15}", name, "(none)", "-");
            }
        }
    }

    // Extension-based detection
    println!("\n=== Extension-Based Detection ===\n");
    for ext in &[
        "mp4", "mkv", "ts", "wav", "flac", "png", "jpg", "mp3", "xyz",
    ] {
        match registry.detect_from_extension(ext) {
            Some(media_type) => println!("  .{ext:<5} → {media_type}"),
            None => println!("  .{ext:<5} → (unknown)"),
        }
    }
}
