//! Example 20: Opus Audio Encoding/Decoding
//!
//! This example demonstrates Opus audio codec usage in Parallax:
//! - Generating a test audio signal (sine wave)
//! - Encoding with OpusEncoder
//! - Decoding with OpusDecoder
//! - Verifying roundtrip works correctly
//!
//! # Requirements
//!
//! - libopus system library installed
//! - Feature flag: `--features opus`
//!
//! # Build Dependencies
//!
//! - **Fedora/RHEL**: `sudo dnf install opus-devel`
//! - **Debian/Ubuntu**: `sudo apt install libopus-dev`
//! - **Arch**: `sudo pacman -S opus`
//! - **macOS**: `brew install opus`
//!
//! # Run
//!
//! ```bash
//! cargo run --example 20_opus_audio --features opus
//! ```

use parallax::elements::codec::{
    AudioDecoder, AudioDecoderElement, AudioEncoder, AudioEncoderElement, AudioSamples,
    OpusApplication, OpusDecoder, OpusEncoder,
};
use parallax::error::Result;
use std::f32::consts::PI;

fn main() -> Result<()> {
    println!("=== Opus Audio Codec Example ===\n");

    // Parameters
    let sample_rate = 48000u32;
    let channels = 2u32;
    let bitrate = 128000u32; // 128 kbps
    let duration_ms = 100; // 100ms of audio

    println!("Configuration:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Channels: {} (stereo)", channels);
    println!("  Bitrate: {} kbps", bitrate / 1000);
    println!("  Duration: {} ms\n", duration_ms);

    // Create encoder and decoder
    let mut encoder = OpusEncoder::new(sample_rate, channels, bitrate, OpusApplication::Audio)?;
    let mut decoder = OpusDecoder::new(sample_rate, channels)?;

    println!("Created Opus encoder (application: Audio)");
    println!("Created Opus decoder\n");

    // Generate test audio: stereo sine wave at 440 Hz (A4 note)
    let samples_per_channel = (sample_rate as usize * duration_ms) / 1000;
    let frequency = 440.0f32;
    let amplitude = 0.7f32;

    println!(
        "Generating {} samples of {} Hz sine wave...",
        samples_per_channel, frequency
    );

    let mut pcm_data: Vec<i16> = Vec::with_capacity(samples_per_channel * channels as usize);
    for i in 0..samples_per_channel {
        let t = i as f32 / sample_rate as f32;
        let sample = (amplitude * (2.0 * PI * frequency * t).sin() * 32767.0) as i16;
        // Stereo: same signal on both channels
        pcm_data.push(sample); // Left
        pcm_data.push(sample); // Right
    }

    let input_samples = AudioSamples::from_s16(&pcm_data, channels, sample_rate);
    let input_bytes = input_samples.data.len();
    println!(
        "  Input size: {} bytes ({} samples)\n",
        input_bytes,
        pcm_data.len()
    );

    // Encode
    println!("Encoding...");
    let mut all_packets = Vec::new();
    let packets = encoder.encode(&input_samples)?;
    all_packets.extend(packets);

    // Flush any remaining buffered samples
    let flush_packets = encoder.flush()?;
    all_packets.extend(flush_packets);

    let total_encoded_bytes: usize = all_packets.iter().map(|p| p.len()).sum();
    println!("  Encoded to {} packets", all_packets.len());
    println!("  Total encoded size: {} bytes", total_encoded_bytes);
    println!(
        "  Compression ratio: {:.1}:1\n",
        input_bytes as f64 / total_encoded_bytes as f64
    );

    // Decode
    println!("Decoding...");
    let mut decoded_samples_count = 0usize;
    let mut decoded_bytes = 0usize;

    for (i, packet) in all_packets.iter().enumerate() {
        let decoded = decoder.decode(packet)?;
        decoded_samples_count += decoded.samples_per_channel;
        decoded_bytes += decoded.data.len();
        println!(
            "  Packet {}: {} bytes -> {} samples",
            i + 1,
            packet.len(),
            decoded.samples_per_channel
        );
    }

    println!("\nSummary:");
    println!(
        "  Input: {} samples, {} bytes",
        samples_per_channel, input_bytes
    );
    println!(
        "  Encoded: {} packets, {} bytes",
        all_packets.len(),
        total_encoded_bytes
    );
    println!(
        "  Decoded: {} samples, {} bytes",
        decoded_samples_count, decoded_bytes
    );

    // Verify sample counts match (approximately - Opus may add/remove samples at boundaries)
    let sample_diff = (decoded_samples_count as i64 - samples_per_channel as i64).abs();
    if sample_diff <= 480 {
        // Allow up to 10ms difference due to padding
        println!("\n[OK] Roundtrip successful! Sample counts match (within tolerance)");
    } else {
        println!(
            "\n[WARN] Sample count mismatch: input={}, decoded={} (diff={})",
            samples_per_channel, decoded_samples_count, sample_diff
        );
    }

    // Demonstrate different frame sizes
    println!("\n=== Frame Size Examples ===\n");
    demonstrate_frame_sizes()?;

    // Demonstrate pipeline integration
    println!("\n=== Pipeline Element Usage ===\n");
    demonstrate_pipeline_elements()?;

    println!("\nDone!");
    Ok(())
}

/// Demonstrate different Opus frame sizes.
fn demonstrate_frame_sizes() -> Result<()> {
    let sample_rate = 48000u32;
    let channels = 2u32;

    // Frame sizes at 48kHz
    let frame_sizes = [
        (120, "2.5ms"),
        (240, "5ms"),
        (480, "10ms"),
        (960, "20ms"), // default
        (1920, "40ms"),
        (2880, "60ms"),
    ];

    for (samples, duration) in frame_sizes {
        let mut encoder = OpusEncoder::new(sample_rate, channels, 64000, OpusApplication::Voip)?;
        encoder.set_frame_size(samples)?;

        // Generate exactly one frame of silence
        let pcm: Vec<i16> = vec![0i16; samples * channels as usize];
        let input = AudioSamples::from_s16(&pcm, channels, sample_rate);

        let packets = encoder.encode(&input)?;
        if let Some(packet) = packets.first() {
            println!(
                "  {} ({} samples): {} bytes encoded",
                duration,
                samples,
                packet.len()
            );
        }
    }

    Ok(())
}

/// Demonstrate using Opus with AudioEncoderElement/AudioDecoderElement wrappers.
fn demonstrate_pipeline_elements() -> Result<()> {
    let sample_rate = 48000u32;
    let channels = 2u32;

    // Create wrapped elements (would be used in a pipeline)
    let encoder = OpusEncoder::new(sample_rate, channels, 96000, OpusApplication::Audio)?;
    let _enc_element = AudioEncoderElement::new_s16(encoder, sample_rate, channels)?;
    println!("  Created AudioEncoderElement<OpusEncoder>");

    let decoder = OpusDecoder::new(sample_rate, channels)?;
    let _dec_element = AudioDecoderElement::new(decoder);
    println!("  Created AudioDecoderElement<OpusDecoder>");

    println!("\n  In a pipeline, these would be used as:");
    println!("    pipeline.add_element(\"opus_enc\", Xfm(enc_element));");
    println!("    pipeline.add_element(\"opus_dec\", Xfm(dec_element));");

    Ok(())
}
