//! # PipeWire Audio Capture and Playback
//!
//! Demonstrates PipeWire audio capture from microphone and playback to speakers.
//!
//! PipeWire is the modern audio/video server for Linux, replacing PulseAudio
//! and JACK. It provides low-latency audio with proper session management.
//!
//! ## Requirements
//!
//! - PipeWire must be running (default on Fedora, Ubuntu 22.04+)
//! - Install: `libpipewire-0.3-dev` (Ubuntu) or `pipewire-devel` (Fedora)
//!
//! ## Status
//!
//! The PipeWire feature requires updates to match the current pipewire-rs API.
//! This example shows the intended usage pattern.
//!
//! ## Intended Usage
//!
//! ```rust,ignore
//! use parallax::elements::device::pipewire::{enumerate_audio_nodes, PipeWireSrc, PipeWireSink};
//!
//! // List available audio nodes
//! let nodes = enumerate_audio_nodes()?;
//! for node in &nodes {
//!     println!("{}: {} (capture: {}, playback: {})",
//!         node.id, node.description, node.is_capture, node.is_playback);
//! }
//!
//! // Create audio capture source (microphone)
//! let mut mic = PipeWireSrc::audio(None)?;  // None = default device
//!
//! // Create audio playback sink (speakers)
//! let speaker = PipeWireSink::audio(None)?;
//!
//! // Use in pipeline
//! let mut pipeline = Pipeline::new();
//! let src = pipeline.add_async_source("mic", mic);
//! let sink = pipeline.add_async_sink("speaker", speaker);
//! pipeline.link(src, sink)?;
//! pipeline.run().await?;
//! ```

fn main() {
    println!("=== PipeWire Audio Example ===\n");
    println!("PipeWire is the modern audio/video server for Linux.");
    println!();
    println!("The 'pipewire' feature provides:");
    println!("  - PipeWireSrc: Audio/video capture from microphones, cameras, screens");
    println!("  - PipeWireSink: Audio playback to speakers");
    println!("  - enumerate_audio_nodes(): List available audio devices");
    println!();
    println!("Requirements:");
    println!("  Fedora: sudo dnf install pipewire-devel");
    println!("  Ubuntu: sudo apt install libpipewire-0.3-dev");
    println!();
    println!("NOTE: The pipewire feature currently needs updates to match");
    println!("the latest pipewire-rs crate API. See src/elements/device/pipewire.rs");
}
