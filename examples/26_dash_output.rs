//! Example 26: DASH Output
//!
//! Demonstrates DASH (Dynamic Adaptive Streaming over HTTP) output with MPD manifests.
//!
//! # What This Example Shows
//!
//! 1. Creating a DASH sink with configuration
//! 2. Generating media segments (.m4s files)
//! 3. Generating MPD (Media Presentation Description) manifests
//! 4. Configuring adaptation sets for ABR streaming
//! 5. Segment timing and rotation
//!
//! # Output Structure
//!
//! ```text
//! /tmp/dash_example/
//! ├── manifest.mpd       # MPD manifest
//! ├── init.mp4           # Initialization segment
//! ├── chunk_000001.m4s   # Media segment
//! ├── chunk_000002.m4s
//! └── ...
//! ```
//!
//! # Run
//!
//! ```bash
//! cargo run --example 26_dash_output
//! ```
//!
//! Then play with:
//! ```bash
//! ffplay /tmp/dash_example/manifest.mpd
//! # or
//! vlc /tmp/dash_example/manifest.mpd
//! ```

use std::path::PathBuf;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::SimpleSink;
use parallax::elements::streaming::{DashAdaptationSet, DashConfig, DashRepresentation, DashSink};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== DASH Output Example ===\n");

    // Create output directory
    let output_dir = PathBuf::from("/tmp/dash_example");
    std::fs::create_dir_all(&output_dir)?;
    println!("Output directory: {:?}\n", output_dir);

    // Configure DASH sink with adaptation sets for ABR
    let config = DashConfig {
        output_dir: output_dir.clone(),
        segment_duration: 4.0, // 4-second segments (DASH standard)
        segment_window: 5,     // Keep 5 segments for live
        manifest_name: "manifest.mpd".to_string(),
        segment_prefix: "chunk_".to_string(),
        init_segment_name: "init.mp4".to_string(),
        is_live: false, // VOD mode for this example
        min_buffer_time: 6.0,
        suggested_presentation_delay: 10.0,
        profile: "urn:mpeg:dash:profile:isoff-live:2011".to_string(),
        adaptation_sets: vec![
            // Video adaptation set with multiple representations
            DashAdaptationSet::video(0, "avc1.64001f")
                .with_representation(
                    DashRepresentation::video("1080p", 5_000_000, 1920, 1080)
                        .with_frame_rate("30/1"),
                )
                .with_representation(
                    DashRepresentation::video("720p", 2_500_000, 1280, 720).with_frame_rate("30/1"),
                )
                .with_representation(
                    DashRepresentation::video("480p", 1_000_000, 854, 480).with_frame_rate("30/1"),
                ),
            // Audio adaptation set
            DashAdaptationSet::audio(1, "mp4a.40.2")
                .with_representation(DashRepresentation::audio("audio_high", 128_000, 48000, 2))
                .with_representation(DashRepresentation::audio("audio_low", 64_000, 44100, 2)),
        ],
    };

    println!("DASH Configuration:");
    println!("  Segment duration: {}s", config.segment_duration);
    println!("  Segment window: {} segments", config.segment_window);
    println!("  Profile: {}", config.profile);
    println!(
        "  Adaptation sets: {} configured",
        config.adaptation_sets.len()
    );
    println!(
        "  Output: {:?}",
        config.output_dir.join(&config.manifest_name)
    );
    println!();

    // Create DASH sink
    let mut dash_sink = DashSink::new(config)?;

    // Write initialization segment
    // In a real pipeline, this would come from an fMP4 muxer
    let init_data = create_fake_init_segment();
    dash_sink.write_init_segment(&init_data)?;
    println!("Wrote initialization segment: {} bytes\n", init_data.len());

    // Create a shared arena for test data
    let arena = SharedArena::new(
        1024 * 64, // 64KB per slot (fMP4 fragments)
        64,        // 64 slots
    )?;

    // Simulate fragmented MP4 data with timing
    println!("Generating test segments...\n");

    let mut pts: i64 = 0;
    let pts_increment: i64 = 33_333_333; // ~30fps in nanoseconds
    let segment_count = 10;
    let frames_per_segment = 120; // 4 seconds at 30fps

    for segment in 0..segment_count {
        println!("  Segment {}/{}", segment + 1, segment_count);

        for frame in 0..frames_per_segment {
            // Reclaim released slots
            arena.reclaim();

            // Create test fMP4 data (moof+mdat)
            let mut slot = arena.acquire().expect("Failed to acquire slot");
            let data = slot.data_mut();

            // Fill with fake fMP4 data
            let frame_data = create_fake_moof_mdat(segment as u32, frame as u32);
            let len = frame_data.len().min(data.len());
            data[..len].copy_from_slice(&frame_data[..len]);

            // Create buffer with PTS
            let mut metadata = Metadata::new();
            metadata.pts = ClockTime::from_nanos(pts as u64);

            // Mark first frame of each segment as keyframe
            if frame == 0 {
                metadata.set("video/keyframe", true);
            }

            let buffer = Buffer::new(MemoryHandle::with_len(slot, len), metadata);

            // Feed to DASH sink
            dash_sink.consume(&buffer)?;

            pts += pts_increment;
        }
    }

    // Finalize (writes final manifest)
    dash_sink.finalize()?;

    // Print statistics
    let stats = dash_sink.stats();
    println!("\nDASH Statistics:");
    println!("  Total segments: {}", stats.total_segments);
    println!("  Total bytes: {} KB", stats.total_bytes / 1024);
    println!("  Total duration: {:.2}s", stats.total_duration);
    println!("  Current segments in window: {}", stats.current_segments);
    println!("  Segment number: {}", stats.segment_number);

    // Read and display the MPD
    let manifest_path = output_dir.join("manifest.mpd");
    if manifest_path.exists() {
        println!("\n=== Generated MPD Manifest ===\n");
        let content = std::fs::read_to_string(&manifest_path)?;
        // Print first 60 lines
        for (i, line) in content.lines().enumerate() {
            if i >= 60 {
                println!(
                    "  ... (truncated, {} more lines)",
                    content.lines().count() - 60
                );
                break;
            }
            println!("{}", line);
        }
    }

    // List generated files
    println!("\n=== Generated Files ===\n");
    for entry in std::fs::read_dir(&output_dir)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        println!(
            "  {:30} {:>10} bytes",
            entry.file_name().to_string_lossy(),
            metadata.len()
        );
    }

    println!("\n=== Adaptation Set Configuration ===\n");
    println!("DASH supports multiple quality levels via AdaptationSets:\n");
    println!("```rust");
    println!("let config = DashConfig {{");
    println!("    adaptation_sets: vec![");
    println!("        DashAdaptationSet::video(0, \"avc1.64001f\")");
    println!("            .with_representation(");
    println!("                DashRepresentation::video(\"1080p\", 5_000_000, 1920, 1080)");
    println!("            )");
    println!("            .with_representation(");
    println!("                DashRepresentation::video(\"720p\", 2_500_000, 1280, 720)");
    println!("            ),");
    println!("        DashAdaptationSet::audio(1, \"mp4a.40.2\")");
    println!("            .with_representation(");
    println!("                DashRepresentation::audio(\"audio\", 128_000, 48000, 2)");
    println!("            ),");
    println!("    ],");
    println!("    ..Default::default()");
    println!("}};");
    println!("```\n");

    println!("=== Pipeline Integration ===\n");
    println!("In a real pipeline:");
    println!();
    println!("```rust");
    println!("let pipeline = Pipeline::parse(\"");
    println!("    videotestsrc !");
    println!("    h264enc bitrate=2000000 !");
    println!("    mp4mux fragment_duration=4000 !");
    println!("    dashsink output_dir=/var/www/stream segment_duration=4");
    println!("\")?;");
    println!("pipeline.run().await?;");
    println!("```");

    println!("\nDone! Play with: ffplay {:?}", manifest_path);

    Ok(())
}

/// Create a fake ftyp+moov box for initialization segment.
fn create_fake_init_segment() -> Vec<u8> {
    let mut data = Vec::new();

    // ftyp box
    data.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x18, // size: 24
        b'f', b't', b'y', b'p', // type: ftyp
        b'i', b's', b'o', b'm', // major brand: isom
        0x00, 0x00, 0x02, 0x00, // minor version
        b'i', b's', b'o', b'm', // compatible brand
        b'm', b'p', b'4', b'1', // compatible brand
    ]);

    // Minimal moov box (just headers, no real codec info)
    data.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x08, // size: 8 (empty moov)
        b'm', b'o', b'o', b'v', // type: moov
    ]);

    data
}

/// Create a fake moof+mdat box for a media segment.
fn create_fake_moof_mdat(segment_num: u32, frame_num: u32) -> Vec<u8> {
    let mut data = Vec::new();

    // Minimal moof box
    let moof_size = 24u32;
    data.extend_from_slice(&moof_size.to_be_bytes());
    data.extend_from_slice(b"moof");

    // mfhd (movie fragment header)
    data.extend_from_slice(&[
        0x00, 0x00, 0x00, 0x10, // size: 16
        b'm', b'f', b'h', b'd', // type: mfhd
        0x00, 0x00, 0x00, 0x00, // version + flags
    ]);
    let sequence = segment_num * 120 + frame_num + 1;
    data.extend_from_slice(&sequence.to_be_bytes());

    // Fake mdat with some payload
    let payload = format!("Frame {} of segment {}", frame_num, segment_num);
    let mdat_size = (8 + payload.len()) as u32;
    data.extend_from_slice(&mdat_size.to_be_bytes());
    data.extend_from_slice(b"mdat");
    data.extend_from_slice(payload.as_bytes());

    data
}
