//! V4L2 video capture with Iced GUI display.
//!
//! This example demonstrates capturing video frames from a V4L2 device
//! (webcam) and displaying them in a native window using the Iced GUI framework.
//!
//! Run with: `cargo run --example 23_v4l2_iced_display --features "v4l2,iced-sink"`
//!
//! Requirements:
//! - Linux system with V4L2 support
//! - A video capture device (webcam, etc.)
//! - User must be in the 'video' group or have permissions to /dev/video*
//!
//! # Current Limitation
//!
//! Parallax cannot yet express this as a simple pipeline like:
//!
//! ```rust,ignore
//! Pipeline::parse("v4l2src ! videoconvert ! icedsink")?.run()?;
//! ```
//!
//! This is because IcedVideoSink requires running the Iced event loop on the
//! main thread (a GUI framework constraint). Future work will add a
//! `Pipeline::run_with_display()` method that handles this automatically.
//!
//! For now, this example shows the manual approach with explicit threading.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::converters::{PixelFormat, VideoConvert};
use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::elements::device::{V4l2Src, enumerate_video_devices};
use parallax::elements::{IcedVideoSink, IcedVideoSinkConfig, InputPixelFormat};
use parallax::error::Result;
use parallax::memory::{CpuArena, HeapSegment, MemorySegment};
use parallax::metadata::Metadata;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("parallax=info,warn")
        .init();

    println!("V4L2 Video Capture with Iced Display");
    println!("=====================================\n");

    // List available video devices
    println!("Scanning for video capture devices...\n");

    let devices = enumerate_video_devices()?;

    if devices.is_empty() {
        println!("No video capture devices found!");
        println!("\nPossible reasons:");
        println!("  - No camera connected");
        println!("  - Missing permissions (try: sudo usermod -aG video $USER)");
        println!("  - V4L2 drivers not loaded");
        return Ok(());
    }

    println!("Found {} device(s):\n", devices.len());
    for (i, dev) in devices.iter().enumerate() {
        println!(
            "  [{}] {} - {} (backend: {:?})",
            i, dev.id, dev.name, dev.backend
        );
    }

    // Use first device
    let device_path = &devices[0].id;
    println!("\nUsing device: {}\n", device_path);

    // Create V4L2 source
    let mut src = V4l2Src::new(device_path)?;

    let width = src.width();
    let height = src.height();
    let fourcc = *src.fourcc();

    println!("Capture configuration:");
    println!("  Resolution: {}x{}", width, height);
    println!(
        "  Format: {}",
        std::str::from_utf8(&fourcc).unwrap_or("????")
    );

    // Determine input format
    let input_format = PixelFormat::from_fourcc(&fourcc);
    println!("  Detected format: {:?}", input_format);

    // Create converter if needed (YUYV -> RGBA for Iced)
    let converter = match input_format {
        Some(PixelFormat::Yuyv) | Some(PixelFormat::Uyvy) => {
            println!("  Creating YUYV -> RGBA converter");
            Some(VideoConvert::new(
                input_format.unwrap(),
                PixelFormat::Rgba,
                width,
                height,
            )?)
        }
        Some(PixelFormat::Rgb24) | Some(PixelFormat::Rgba) => {
            println!("  No conversion needed (already RGB/RGBA)");
            None
        }
        _ => {
            println!("  Warning: Unknown format, attempting direct display");
            None
        }
    };

    // Calculate buffer sizes
    let input_size = src
        .preferred_buffer_size()
        .unwrap_or((width * height * 2) as usize);
    let output_size = (width * height * 4) as usize; // RGBA

    println!("  Input buffer: {} bytes", input_size);
    println!("  Output buffer: {} bytes", output_size);

    // Create Iced video sink
    let config = IcedVideoSinkConfig {
        title: format!("V4L2 Capture - {} ({}x{})", device_path, width, height),
        width,
        height,
        show_stats: true,
        pixel_format: InputPixelFormat::Rgba32,
    };

    let (mut sink, handle) = IcedVideoSink::with_config(config);

    println!("\nStarting capture... Press Ctrl+C or close window to stop.\n");

    // Shared flag for shutdown
    let running = Arc::new(AtomicBool::new(true));
    let running_clone = running.clone();

    // Create an arena for capture buffers
    let arena = CpuArena::new(input_size, 4)?;

    // Spawn capture thread
    let capture_thread = thread::spawn(move || {
        let mut frame_count = 0u64;
        let mut rgba_buffer = vec![0u8; output_size];

        while running_clone.load(Ordering::Relaxed) && sink.is_window_open() {
            // Acquire a slot from the arena for capture
            let slot = match arena.acquire() {
                Some(s) => s,
                None => {
                    // Arena exhausted, wait a bit
                    thread::sleep(std::time::Duration::from_millis(1));
                    continue;
                }
            };

            // Create produce context with the arena slot
            let mut ctx = ProduceContext::new(slot);

            // Capture a frame
            match src.produce(&mut ctx) {
                Ok(ProduceResult::Produced(len)) => {
                    frame_count += 1;

                    // Get the captured data from the context
                    let captured_data = &ctx.output()[..len];

                    // Convert to RGBA and display
                    if display_frame(
                        captured_data,
                        &converter,
                        &mut rgba_buffer,
                        output_size,
                        frame_count,
                        &mut sink,
                    ) {
                        // Successfully displayed
                    } else {
                        break;
                    }

                    if frame_count % 30 == 0 {
                        println!("Captured {} frames", frame_count);
                    }
                }
                Ok(ProduceResult::OwnBuffer(buf)) => {
                    // Handle OwnBuffer case - the source provided its own buffer
                    frame_count += 1;
                    let captured_data = buf.as_bytes();

                    if display_frame(
                        captured_data,
                        &converter,
                        &mut rgba_buffer,
                        output_size,
                        frame_count,
                        &mut sink,
                    ) {
                        // Successfully displayed
                    } else {
                        break;
                    }

                    if frame_count % 30 == 0 {
                        println!("Captured {} frames", frame_count);
                    }
                }
                Ok(ProduceResult::Eos) => {
                    println!("End of stream");
                    break;
                }
                Ok(ProduceResult::WouldBlock) => {
                    thread::sleep(std::time::Duration::from_millis(1));
                }
                Err(e) => {
                    eprintln!("Capture error: {}", e);
                    break;
                }
            }
        }

        // Explicitly stop the source to release the device
        src.stop();
        println!("Capture thread finished. Total frames: {}", frame_count);
    });

    // Run Iced window (blocks until closed)
    if let Err(e) = handle.run() {
        eprintln!("Window error: {}", e);
    }

    // Signal capture thread to stop
    running.store(false, Ordering::Relaxed);

    // Wait for capture thread
    if capture_thread.join().is_err() {
        eprintln!("Capture thread panicked");
    }

    println!("\nDone!");
    Ok(())
}

/// Convert captured data to RGBA and send to the Iced sink.
/// Returns true if successful, false if the sink is closed.
fn display_frame(
    captured_data: &[u8],
    converter: &Option<VideoConvert>,
    rgba_buffer: &mut [u8],
    output_size: usize,
    frame_count: u64,
    sink: &mut IcedVideoSink,
) -> bool {
    // Convert to RGBA if needed
    let rgba_data = if let Some(conv) = converter {
        if conv.convert(captured_data, rgba_buffer).is_ok() {
            &rgba_buffer[..]
        } else {
            return true; // Skip this frame but continue
        }
    } else {
        // No conversion - copy directly (assuming already RGBA or compatible)
        let copy_len = captured_data.len().min(output_size);
        rgba_buffer[..copy_len].copy_from_slice(&captured_data[..copy_len]);
        &rgba_buffer[..output_size]
    };

    // Create buffer for sink
    let segment = match HeapSegment::new(rgba_data.len()) {
        Ok(s) => Arc::new(s),
        Err(_) => return true, // Skip frame
    };

    // Copy RGBA data to segment
    unsafe {
        std::ptr::copy_nonoverlapping(
            rgba_data.as_ptr(),
            segment.as_mut_ptr().unwrap(),
            rgba_data.len(),
        );
    }

    let handle = MemoryHandle::from_segment(segment);
    let metadata = Metadata::default().with_sequence(frame_count);
    let buffer = Buffer::new(handle, metadata);

    // Send to sink via ConsumeContext
    let ctx = ConsumeContext::new(&buffer);
    sink.consume(&ctx).is_ok()
}
