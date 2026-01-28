//! # ALSA Audio Capture and Playback
//!
//! Demonstrates ALSA device enumeration and audio capture/playback.
//! ALSA is the fallback when PipeWire is not available.
//!
//! ## Requirements
//!
//! - Install: `libasound2-dev` (Ubuntu) or `alsa-lib-devel` (Fedora)
//!
//! ## Run
//!
//! ```bash
//! cargo run --example 43_alsa_audio --features alsa
//! ```

#[cfg(feature = "alsa")]
use parallax::elements::device::alsa::{
    AlsaFormat, AlsaSampleFormat, AlsaSink, AlsaSrc, enumerate_devices, is_available,
};

#[cfg(feature = "alsa")]
fn main() {
    println!("=== ALSA Audio Example ===\n");

    // Check if ALSA is available
    if !is_available() {
        eprintln!("ALSA is not available on this system.");
        return;
    }
    println!("ALSA is available.\n");

    // Enumerate devices
    println!("Enumerating ALSA devices...");
    match enumerate_devices() {
        Ok(devices) => {
            if devices.is_empty() {
                println!("No ALSA devices found.");
            } else {
                println!("Found {} device(s):\n", devices.len());
                for dev in &devices {
                    println!("  {} - {}", dev.name, dev.description);
                    println!(
                        "       Capture: {}, Playback: {}",
                        dev.is_capture, dev.is_playback
                    );
                    println!();
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to enumerate devices: {}", e);
        }
    }

    // Show format configuration
    let format = AlsaFormat {
        sample_rate: 48000,
        channels: 2,
        format: AlsaSampleFormat::S16LE,
        buffer_frames: 4096,
        period_frames: 1024,
    };
    println!("Audio format configuration:");
    println!("  Sample rate: {} Hz", format.sample_rate);
    println!("  Channels: {}", format.channels);
    println!("  Format: {:?}", format.format);
    println!("  Buffer: {} frames", format.buffer_frames);
    println!("  Period: {} frames", format.period_frames);
    println!();

    // Try to create capture source
    println!("Creating ALSA capture source (default device)...");
    match AlsaSrc::new("default", format.clone()) {
        Ok(_src) => {
            println!("Successfully created ALSA capture source.");
        }
        Err(e) => {
            eprintln!("Failed to create capture source: {}", e);
            eprintln!("This may be normal if no microphone is available.");
        }
    }

    // Try to create playback sink
    println!("Creating ALSA playback sink (default device)...");
    match AlsaSink::new("default", format) {
        Ok(_sink) => {
            println!("Successfully created ALSA playback sink.");
        }
        Err(e) => {
            eprintln!("Failed to create playback sink: {}", e);
        }
    }

    println!();
    println!("In a real application, you would use these in a pipeline:");
    println!();
    println!("  let mut pipeline = Pipeline::new();");
    println!("  let src = pipeline.add_async_source(\"mic\", src);");
    println!("  let sink = pipeline.add_async_sink(\"speaker\", sink);");
    println!("  pipeline.link(src, sink)?;");
    println!("  pipeline.run().await?;");
}

#[cfg(not(feature = "alsa"))]
fn main() {
    eprintln!("This example requires the 'alsa' feature.");
    eprintln!("Run with: cargo run --example 43_alsa_audio --features alsa");
}
