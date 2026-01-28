//! # ALSA Audio Capture and Playback
//!
//! Demonstrates ALSA audio capture from microphone and playback to speakers.
//!
//! ALSA (Advanced Linux Sound Architecture) provides direct access to audio
//! hardware. Use this as a fallback when PipeWire is not available.
//!
//! ## Requirements
//!
//! - Install: `libasound2-dev` (Ubuntu) or `alsa-lib-devel` (Fedora)
//!
//! ## Status
//!
//! The ALSA feature requires updates to match the current alsa-rs crate API.
//! This example shows the intended usage pattern.
//!
//! ## Intended Usage
//!
//! ```rust,ignore
//! use parallax::elements::device::alsa::{AlsaSrc, AlsaSink, AlsaFormat, enumerate_devices};
//!
//! // List available ALSA devices
//! let devices = enumerate_devices()?;
//! for dev in &devices {
//!     println!("{}: {} (capture: {}, playback: {})",
//!         dev.name, dev.description, dev.is_capture, dev.is_playback);
//! }
//!
//! // Audio format: CD quality (44.1kHz, 16-bit stereo)
//! let format = AlsaFormat {
//!     sample_rate: 44100,
//!     channels: 2,
//!     format: SampleFormat::S16Le,
//! };
//!
//! // Create capture source (microphone)
//! let mic = AlsaSrc::new("default", format)?;
//!
//! // Create playback sink (speakers)
//! let speaker = AlsaSink::new("default", format)?;
//!
//! // Use in pipeline
//! let mut pipeline = Pipeline::new();
//! let src = pipeline.add_async_source("mic", mic);
//! let sink = pipeline.add_async_sink("speaker", speaker);
//! pipeline.link(src, sink)?;
//! pipeline.run().await?;
//! ```

fn main() {
    println!("=== ALSA Audio Example ===\n");
    println!("ALSA provides direct access to audio hardware on Linux.");
    println!("Use as a fallback when PipeWire is not available.");
    println!();
    println!("The 'alsa' feature provides:");
    println!("  - AlsaSrc: Audio capture from microphones");
    println!("  - AlsaSink: Audio playback to speakers");
    println!("  - enumerate_devices(): List available ALSA devices");
    println!();
    println!("Requirements:");
    println!("  Fedora: sudo dnf install alsa-lib-devel");
    println!("  Ubuntu: sudo apt install libasound2-dev");
    println!();
    println!("NOTE: The alsa feature currently needs updates to match");
    println!("the latest alsa-rs crate API. See src/elements/device/alsa.rs");
}
