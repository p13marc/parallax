//! # libcamera Video Capture
//!
//! Demonstrates libcamera camera enumeration and video capture.
//! libcamera is the modern camera stack for Linux, handling complex
//! camera pipelines including ISP and 3A algorithms.
//!
//! ## Requirements
//!
//! - Install: `libcamera-dev` (Ubuntu) or `libcamera-devel` (Fedora)
//! - A camera supported by libcamera (USB webcams, Raspberry Pi cameras, etc.)
//!
//! ## Run
//!
//! ```bash
//! cargo run --example 44_libcamera_capture --features libcamera
//! ```

#[cfg(feature = "libcamera")]
use parallax::elements::device::libcamera::{
    LibCameraConfig, LibCameraSrc, enumerate_cameras, is_available,
};

#[cfg(feature = "libcamera")]
fn main() {
    println!("=== libcamera Video Capture Example ===\n");

    // Check if libcamera is available
    if !is_available() {
        eprintln!("libcamera is not available on this system.");
        eprintln!("Make sure libcamera is installed and cameras are connected.");
        return;
    }
    println!("libcamera is available.\n");

    // Enumerate cameras
    println!("Enumerating cameras...");
    match enumerate_cameras() {
        Ok(cameras) => {
            if cameras.is_empty() {
                println!("No cameras found.");
                println!("Connect a USB webcam or use a device with a built-in camera.");
                return;
            }

            println!("Found {} camera(s):\n", cameras.len());
            for cam in &cameras {
                println!("  ID: {}", cam.id);
                println!("  Model: {}", cam.model);
                println!("  Location: {:?}", cam.location);
                println!();
            }

            // Try to open the first camera
            let first_camera = &cameras[0];
            println!("Opening camera: {}", first_camera.id);

            let config = LibCameraConfig {
                width: 640,
                height: 480,
                format: None, // Use default format
                buffer_count: 4,
            };

            match LibCameraSrc::with_config(config) {
                Ok(_src) => {
                    println!("Successfully opened camera.");
                    println!();
                    println!("In a real application, you would use this in a pipeline:");
                    println!();
                    println!("  let mut pipeline = Pipeline::new();");
                    println!("  let src = pipeline.add_async_source(\"camera\", src);");
                    println!("  // ... add encoder and sink");
                    println!("  pipeline.run().await?;");
                }
                Err(e) => {
                    eprintln!("Failed to open camera: {}", e);
                    eprintln!("The camera may be in use by another application.");
                }
            }
        }
        Err(e) => {
            eprintln!("Failed to enumerate cameras: {}", e);
        }
    }
}

#[cfg(not(feature = "libcamera"))]
fn main() {
    eprintln!("This example requires the 'libcamera' feature.");
    eprintln!("Run with: cargo run --example 44_libcamera_capture --features libcamera");
}
