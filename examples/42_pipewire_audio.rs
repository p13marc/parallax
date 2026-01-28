//! # PipeWire Audio Capture
//!
//! Demonstrates PipeWire audio node enumeration and capture.
//!
//! ## Requirements
//!
//! - PipeWire daemon must be running
//! - Install: `libpipewire-0.3-dev` (Ubuntu) or `pipewire-devel` (Fedora)
//!
//! ## Run
//!
//! ```bash
//! cargo run --example 42_pipewire_audio --features pipewire
//! ```

#[cfg(feature = "pipewire")]
use parallax::elements::device::pipewire::{PipeWireSrc, enumerate_audio_nodes, is_available};

#[cfg(feature = "pipewire")]
fn main() {
    println!("=== PipeWire Audio Example ===\n");

    // Check if PipeWire is available
    if !is_available() {
        eprintln!("PipeWire is not available on this system.");
        eprintln!("Make sure the PipeWire daemon is running.");
        return;
    }
    println!("PipeWire is available.\n");

    // Enumerate audio nodes
    println!("Enumerating audio nodes...");
    match enumerate_audio_nodes() {
        Ok(nodes) => {
            if nodes.is_empty() {
                println!("No audio nodes found.");
            } else {
                println!("Found {} audio node(s):\n", nodes.len());
                for node in &nodes {
                    println!("  [{}] {} - {}", node.id, node.name, node.media_class);
                    println!("       Description: {}", node.description);
                    println!(
                        "       Capture: {}, Playback: {}",
                        node.is_capture, node.is_playback
                    );
                    println!();
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to enumerate audio nodes: {}", e);
        }
    }

    // Try to create a capture source
    println!("Creating audio capture source (default microphone)...");
    match PipeWireSrc::audio(None) {
        Ok(_src) => {
            println!("Successfully created PipeWire audio source.");
            println!("In a real application, you would use this in a pipeline:");
            println!();
            println!("  let mut pipeline = Pipeline::new();");
            println!("  let src = pipeline.add_async_source(\"mic\", src);");
            println!("  // ... add processing and sink");
            println!("  pipeline.run().await?;");
        }
        Err(e) => {
            eprintln!("Failed to create audio source: {}", e);
            eprintln!("This may be normal if no microphone is available.");
        }
    }
}

#[cfg(not(feature = "pipewire"))]
fn main() {
    eprintln!("This example requires the 'pipewire' feature.");
    eprintln!("Run with: cargo run --example 42_pipewire_audio --features pipewire");
}
