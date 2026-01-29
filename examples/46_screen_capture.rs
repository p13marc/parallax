//! Screen capture via XDG Desktop Portal.
//!
//! This example demonstrates capturing screen content on Wayland/X11 using
//! the XDG Desktop Portal. The user will be prompted to select a screen or
//! window to capture.
//!
//! Run with: cargo run --example 46_screen_capture --features screen-capture
//!
//! Requirements:
//! - XDG Desktop Portal service running
//! - PipeWire session manager
//! - Portal backend (xdg-desktop-portal-gnome, xdg-desktop-portal-kde, etc.)

use parallax::elements::device::{CaptureSourceType, ScreenCaptureConfig, ScreenCaptureSrc};
use parallax::error::Result;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing for debug output
    tracing_subscriber::fmt()
        .with_env_filter("parallax=debug")
        .init();

    println!("Screen Capture Example");
    println!("======================");
    println!();
    println!("This will prompt you to select a screen or window to capture.");
    println!();

    // Create screen capture configuration
    let config = ScreenCaptureConfig {
        source_type: CaptureSourceType::Any, // Let user choose monitor or window
        show_cursor: true,
        persist_session: false,
    };

    // Create the screen capture source
    let mut capture = ScreenCaptureSrc::new(config);

    // Initialize the portal session (this shows the permission dialog)
    println!("Requesting screen capture permission...");
    let info = capture.initialize().await?;

    println!();
    println!("Screen capture initialized:");
    println!("  Resolution: {}x{}", info.width, info.height);
    println!("  PipeWire node ID: {}", info.node_id);
    println!();

    // Note: To actually receive frames, you would need to:
    // 1. Connect to PipeWire using the node_id
    // 2. Set up a PipeWire stream to receive video frames
    // 3. Process frames in the produce() method
    //
    // This example just demonstrates the portal session setup.
    // For a complete implementation, see the PipeWireSrc element
    // with screen_capture() constructor.

    println!("Portal session created successfully!");
    println!();
    println!("To capture frames, use this source in a pipeline:");
    println!("  let pipeline = Pipeline::new();");
    println!("  pipeline.add_element(\"screen\", Src(capture));");
    println!("  // ... add sink and run");

    Ok(())
}
