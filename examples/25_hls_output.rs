//! Example 25: HLS Output
//!
//! Demonstrates HLS (HTTP Live Streaming) output with M3U8 playlists.
//!
//! # What This Example Shows
//!
//! 1. Creating an HLS sink with configuration
//! 2. Generating media segments (.ts files)
//! 3. Generating M3U8 playlists
//! 4. Segment timing and rotation
//! 5. Master playlist for ABR (multi-bitrate)
//!
//! # Output Structure
//!
//! ```text
//! /tmp/hls_example/
//! ├── playlist.m3u8      # Media playlist
//! ├── segment_000001.ts  # Media segment
//! ├── segment_000002.ts
//! └── ...
//! ```
//!
//! # Run
//!
//! ```bash
//! cargo run --example 25_hls_output
//! ```
//!
//! Then play with:
//! ```bash
//! ffplay /tmp/hls_example/playlist.m3u8
//! # or
//! vlc /tmp/hls_example/playlist.m3u8
//! ```

use std::path::PathBuf;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::SimpleSink;
use parallax::elements::streaming::{HlsConfig, HlsSink, HlsVariant};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== HLS Output Example ===\n");

    // Create output directory
    let output_dir = PathBuf::from("/tmp/hls_example");
    std::fs::create_dir_all(&output_dir)?;
    println!("Output directory: {:?}\n", output_dir);

    // Configure HLS sink
    let config = HlsConfig {
        output_dir: output_dir.clone(),
        segment_duration: 2.0, // 2-second segments for demo
        playlist_length: 5,    // Keep 5 segments in playlist
        playlist_name: "playlist.m3u8".to_string(),
        segment_prefix: "segment".to_string(),
        is_vod: true, // Set to true so playlist gets ENDLIST
        ..Default::default()
    };

    println!("HLS Configuration:");
    println!("  Segment duration: {}s", config.segment_duration);
    println!("  Playlist length: {} segments", config.playlist_length);
    println!(
        "  Output: {:?}",
        config.output_dir.join(&config.playlist_name)
    );
    println!();

    // Create HLS sink
    let mut hls_sink = HlsSink::new(config)?;

    // Create a shared arena for test data
    let arena = SharedArena::new(188 * 100, 64)?; // TS packet size * count

    // Simulate TS packets with timing
    // In a real pipeline, this would come from TsMux
    println!("Generating test segments...\n");

    let mut pts: i64 = 0;
    let pts_increment: i64 = 33_333_333; // ~30fps in nanoseconds
    let segment_count = 10;
    let frames_per_segment = 60; // 2 seconds at 30fps

    for segment in 0..segment_count {
        println!("  Segment {}/{}", segment + 1, segment_count);

        for frame in 0..frames_per_segment {
            // Reclaim released slots
            arena.reclaim();

            // Create test TS data (188-byte packets)
            let mut slot = arena.acquire().expect("Failed to acquire slot");
            let data = slot.data_mut();

            // Fill with fake TS sync bytes and padding
            for i in 0..(188 * 10) {
                data[i] = if i % 188 == 0 { 0x47 } else { 0xFF }; // TS sync byte
            }

            // Create buffer with PTS
            let mut metadata = Metadata::new();
            metadata.pts = ClockTime::from_nanos(pts as u64);

            // Mark first frame of each segment as keyframe
            if frame == 0 {
                metadata.set("video/keyframe", true);
            }

            let buffer = Buffer::new(MemoryHandle::with_len(slot, 188 * 10), metadata);

            // Feed to HLS sink
            hls_sink.consume(&buffer)?;

            pts += pts_increment;
        }
    }

    // Finalize (writes final playlist)
    hls_sink.finalize()?;

    // Print statistics
    let stats = hls_sink.stats();
    println!("\nHLS Statistics:");
    println!("  Total segments: {}", stats.total_segments);
    println!("  Total bytes: {} KB", stats.total_bytes / 1024);
    println!("  Current segments in playlist: {}", stats.current_segments);
    println!("  Media sequence: {}", stats.media_sequence);

    // Read and display the playlist
    let playlist_path = output_dir.join("playlist.m3u8");
    if playlist_path.exists() {
        println!("\n=== Generated Playlist ===\n");
        let content = std::fs::read_to_string(&playlist_path)?;
        println!("{}", content);
    }

    // List generated files
    println!("=== Generated Files ===\n");
    for entry in std::fs::read_dir(&output_dir)? {
        let entry = entry?;
        let metadata = entry.metadata()?;
        println!(
            "  {:30} {:>10} bytes",
            entry.file_name().to_string_lossy(),
            metadata.len()
        );
    }

    println!("\n=== Master Playlist Example ===\n");
    println!("For adaptive bitrate streaming, configure variants:\n");
    println!("```rust");
    println!("let config = HlsConfig {{");
    println!("    variants: vec![");
    println!("        HlsVariant::new(\"1080p\", 5_000_000)");
    println!("            .with_resolution(1920, 1080)");
    println!("            .with_codecs(\"avc1.640028,mp4a.40.2\"),");
    println!("        HlsVariant::new(\"720p\", 2_500_000)");
    println!("            .with_resolution(1280, 720)");
    println!("            .with_codecs(\"avc1.64001f,mp4a.40.2\"),");
    println!("        HlsVariant::new(\"480p\", 1_000_000)");
    println!("            .with_resolution(854, 480)");
    println!("            .with_codecs(\"avc1.640015,mp4a.40.2\"),");
    println!("    ],");
    println!("    ..Default::default()");
    println!("}};");
    println!("```\n");

    // Demonstrate variant configuration
    let variant = HlsVariant::new("720p", 2_500_000)
        .with_resolution(1280, 720)
        .with_codecs("avc1.64001f,mp4a.40.2");

    println!("Example variant: {:?}", variant);

    println!("\n=== Pipeline Integration ===\n");
    println!("In a real pipeline:");
    println!();
    println!("```rust");
    println!("let pipeline = Pipeline::parse(\"");
    println!("    videotestsrc !");
    println!("    h264enc bitrate=2000000 !");
    println!("    tsmux !");
    println!("    hlssink output_dir=/var/www/stream segment_duration=6");
    println!("\")?;");
    println!("pipeline.run().await?;");
    println!("```");

    println!("\nDone! Play with: ffplay {:?}", playlist_path);

    Ok(())
}
