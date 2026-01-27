//! Example 25: MP4 Container Format
//!
//! This example demonstrates how to use the MP4 demuxer and muxer
//! for reading and creating MP4 container files.
//!
//! - **Mp4Demux**: Reads MP4 files and extracts elementary streams (video, audio)
//! - **Mp4Mux**: Creates MP4 files from elementary streams
//!
//! Run with: cargo run --example 25_mp4_container --features mp4-demux

use std::io::Cursor;

#[cfg(feature = "mp4-demux")]
fn main() {
    use parallax::elements::demux::{Mp4Codec, Mp4Demux, Mp4TrackType};
    use parallax::elements::mux::{Mp4AudioTrackConfig, Mp4Mux, Mp4MuxConfig, Mp4VideoTrackConfig};

    println!("=== MP4 Container Format Example ===\n");

    // Part 1: Create an MP4 file using the muxer
    println!("1. Creating MP4 file with Mp4Mux:");
    println!("   - Adding H.264 video track (640x480)");
    println!("   - Adding AAC audio track (48kHz stereo)");
    println!();

    let output_buffer = Cursor::new(Vec::new());
    let mut mux =
        Mp4Mux::new(output_buffer, Mp4MuxConfig::default()).expect("Failed to create MP4 muxer");

    // H.264 codec parameters (simplified - real SPS/PPS would come from encoder)
    let sps = vec![
        0x67, 0x42, 0x00, 0x1f, 0xe9, 0x01, 0x40, 0x7a, 0xc1, 0x00, 0x00, 0x03, 0x00, 0x01, 0x00,
        0x00, 0x03, 0x00, 0x3c, 0x8f, 0x16, 0x2d, 0x96,
    ];
    let pps = vec![0x68, 0xce, 0x3c, 0x80];

    // Add video track
    let video_config = Mp4VideoTrackConfig::h264(640, 480, &sps, &pps);
    let video_track = mux
        .add_video_track(video_config)
        .expect("Failed to add video track");
    println!("   Created video track (ID: {})", video_track);

    // Add audio track
    let audio_config = Mp4AudioTrackConfig::aac(48000, 2);
    let audio_track = mux
        .add_audio_track(audio_config)
        .expect("Failed to add audio track");
    println!("   Created audio track (ID: {})", audio_track);

    // Write some sample data (simulated encoded frames)
    println!();
    println!("2. Writing samples:");

    // Write video frames (IDR + P frames at ~30fps)
    for i in 0..10 {
        let is_keyframe = i == 0;
        let pts_ms = i * 33; // ~30fps

        // Simulate NAL unit data (in real app, this would be actual encoded data)
        let nal_type = if is_keyframe { 0x65 } else { 0x41 };
        let frame_data = vec![0x00, 0x00, 0x00, 0x01, nal_type, 0x88, 0x84, 0x00];

        mux.write_video_sample(video_track, &frame_data, pts_ms, is_keyframe)
            .expect("Failed to write video sample");

        println!(
            "   Video frame {}: pts={}ms, keyframe={}",
            i, pts_ms, is_keyframe
        );
    }

    // Write audio frames (AAC frames, ~21ms each for 48kHz)
    for i in 0..15 {
        let pts_ms = i * 21; // AAC frame duration

        // Simulate AAC frame (in real app, this would be actual AAC data)
        let audio_data = vec![0xFF, 0xF1, 0x50, 0x80, 0x00, 0x1F, 0xFC];

        mux.write_audio_sample(audio_track, &audio_data, pts_ms)
            .expect("Failed to write audio sample");

        println!("   Audio frame {}: pts={}ms", i, pts_ms);
    }

    // Get statistics before finishing
    let stats = mux.stats();
    println!();
    println!("3. Muxer statistics:");
    println!("   Total samples written: {}", stats.samples_written);
    println!("   Video samples: {}", stats.video_samples);
    println!("   Audio samples: {}", stats.audio_samples);
    println!("   Keyframes: {}", stats.keyframes);
    println!("   Bytes written: {}", stats.bytes_written);

    // Finalize the MP4 file
    let output = mux.finish().expect("Failed to finalize MP4");
    let mp4_data = output.into_inner();
    println!();
    println!("   Final MP4 size: {} bytes", mp4_data.len());

    // Part 2: Read the MP4 file using the demuxer
    println!();
    println!("4. Reading MP4 file with Mp4Demux:");

    let reader = Cursor::new(mp4_data.clone());
    let mut demux =
        Mp4Demux::new(reader, mp4_data.len() as u64).expect("Failed to create MP4 demuxer");

    // Get track information
    let tracks = demux.tracks();
    println!("   Found {} tracks:", tracks.len());

    for track in tracks {
        println!();
        println!("   Track {}:", track.id);
        println!("     Type: {:?}", track.track_type);
        println!("     Codec: {}", track.codec);
        println!("     Duration: {}ms", track.duration_ns / 1_000_000);

        if let Some(video_info) = &track.video_info {
            let fps_str = match video_info.frame_rate {
                Some(fps) => format!("{:.2}fps", fps),
                None => "unknown fps".to_string(),
            };
            println!(
                "     Video: {}x{} @ {}",
                video_info.width, video_info.height, fps_str
            );
        }

        if let Some(audio_info) = &track.audio_info {
            println!(
                "     Audio: {}Hz, {} channels",
                audio_info.sample_rate, audio_info.channels
            );
        }
    }

    // Read samples from video track
    println!();
    println!("5. Reading video samples:");

    if let Some(video_id) = demux.video_track_id() {
        let samples = demux
            .read_all_samples(video_id)
            .expect("Failed to read video samples");
        println!("   Read {} video samples:", samples.len());

        for (i, sample) in samples.iter().enumerate() {
            println!(
                "     Frame {}: pts={}ns, dts={}ns, size={}, keyframe={}",
                i,
                sample.pts_ns,
                sample.dts_ns,
                sample.buffer.len(),
                sample.is_keyframe
            );
        }
    }

    // Reset and read audio samples
    println!();
    println!("6. Reading audio samples:");

    // Create a new demuxer instance to read audio
    let reader = Cursor::new(mp4_data.clone());
    let mut demux =
        Mp4Demux::new(reader, mp4_data.len() as u64).expect("Failed to create MP4 demuxer");

    if let Some(audio_id) = demux.audio_track_id() {
        let samples = demux
            .read_all_samples(audio_id)
            .expect("Failed to read audio samples");
        println!("   Read {} audio samples:", samples.len());

        for (i, sample) in samples.iter().take(5).enumerate() {
            println!(
                "     Frame {}: pts={}ns, size={}",
                i,
                sample.pts_ns,
                sample.buffer.len()
            );
        }
        if samples.len() > 5 {
            println!("     ... and {} more", samples.len() - 5);
        }
    }

    // Demonstrate codec detection
    println!();
    println!("7. Codec information:");

    let mp4_len = mp4_data.len() as u64;
    let reader = Cursor::new(mp4_data);
    let demux = Mp4Demux::new(reader, mp4_len).expect("Failed to create MP4 demuxer");

    for track in demux.tracks() {
        match track.codec {
            Mp4Codec::H264 => println!("   Track {}: H.264/AVC video codec", track.id),
            Mp4Codec::H265 => println!("   Track {}: H.265/HEVC video codec", track.id),
            Mp4Codec::Vp9 => println!("   Track {}: VP9 video codec", track.id),
            Mp4Codec::Aac => println!("   Track {}: AAC audio codec", track.id),
            Mp4Codec::Ttxt => println!("   Track {}: Timed text (subtitles)", track.id),
            Mp4Codec::Unknown => println!("   Track {}: Unknown codec", track.id),
        }

        match track.track_type {
            Mp4TrackType::Video => println!("     -> Video track"),
            Mp4TrackType::Audio => println!("     -> Audio track"),
            Mp4TrackType::Subtitle => println!("     -> Subtitle track"),
            Mp4TrackType::Unknown => println!("     -> Unknown track type"),
        }
    }

    println!();
    println!("=== Example Complete ===");
    println!();
    println!("Key concepts demonstrated:");
    println!("  - Mp4Mux: Create MP4 containers with video/audio tracks");
    println!("  - Mp4MuxConfig: Configure container format (brands, timescale)");
    println!("  - Mp4VideoTrackConfig: H.264/H.265/VP9 video track setup");
    println!("  - Mp4AudioTrackConfig: AAC audio track setup");
    println!("  - Mp4Demux: Parse MP4 files and extract track info");
    println!("  - Mp4Track: Track metadata (codec, duration, video/audio info)");
    println!("  - Mp4Sample: Individual media samples with timestamps");
}

#[cfg(not(feature = "mp4-demux"))]
fn main() {
    println!("This example requires the 'mp4-demux' feature.");
    println!("Run with: cargo run --example 25_mp4_container --features mp4-demux");
}
