//! # libcamera Video Capture
//!
//! Demonstrates libcamera video capture from cameras.
//!
//! libcamera is the modern camera stack for Linux, providing a unified API
//! for complex camera hardware including:
//! - USB webcams
//! - MIPI CSI cameras (Raspberry Pi, embedded systems)
//! - Cameras requiring ISP processing
//!
//! ## Requirements
//!
//! - Install: `libcamera-dev` (Ubuntu) or `libcamera-devel` (Fedora)
//! - A camera supported by libcamera
//!
//! ## Status
//!
//! The libcamera feature requires updates to match the current libcamera-rs API.
//! This example shows the intended usage pattern.
//!
//! ## Intended Usage
//!
//! ```rust,ignore
//! use parallax::elements::device::libcamera::{LibCameraSrc, LibCameraConfig, enumerate_cameras};
//!
//! // List available cameras
//! let cameras = enumerate_cameras()?;
//! for cam in &cameras {
//!     println!("{}: {} ({:?})", cam.id, cam.model, cam.location);
//! }
//!
//! // Create camera source with default configuration
//! let camera = LibCameraSrc::new()?;
//!
//! // Or with specific configuration
//! let config = LibCameraConfig {
//!     width: 1920,
//!     height: 1080,
//!     format: PixelFormat::NV12,
//!     buffer_count: 4,
//! };
//! let camera = LibCameraSrc::with_config(config)?;
//!
//! // Use in pipeline
//! let mut pipeline = Pipeline::new();
//! let src = pipeline.add_async_source("camera", camera);
//! let sink = pipeline.add_sink("display", display_sink);
//! pipeline.link(src, sink)?;
//! pipeline.run().await?;
//! ```
//!
//! ## libcamera vs V4L2
//!
//! | Aspect | V4L2 | libcamera |
//! |--------|------|-----------|
//! | API Level | Low (ioctl) | High (C++ lib) |
//! | Complex cameras | Manual ISP | Automatic |
//! | 3A algorithms | None | AWB, AE, AF |
//! | Raspberry Pi | Requires custom | First-class |
//! | Simple webcams | Easy | Works (overkill) |

fn main() {
    println!("=== libcamera Video Capture Example ===\n");
    println!("libcamera is the modern camera stack for Linux.");
    println!();
    println!("The 'libcamera' feature provides:");
    println!("  - LibCameraSrc: Video capture from cameras");
    println!("  - enumerate_cameras(): List available cameras");
    println!("  - Automatic ISP and 3A (AWB, AE, AF) configuration");
    println!("  - DMA-BUF support for zero-copy");
    println!();
    println!("Requirements:");
    println!("  Fedora: sudo dnf install libcamera-devel");
    println!("  Ubuntu: sudo apt install libcamera-dev");
    println!();
    println!("Check available cameras with: cam -l");
    println!();
    println!("NOTE: The libcamera feature currently needs updates to match");
    println!("the latest libcamera-rs crate API. See src/elements/device/libcamera.rs");
    println!();
    println!("For simple webcams, consider using V4L2 instead:");
    println!("  cargo run --example 23_v4l2_display --features v4l2,display");
}
