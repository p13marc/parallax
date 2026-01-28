//! V4L2 video capture with window display.
//!
//! This example demonstrates capturing video frames from a V4L2 device
//! (webcam) and displaying them in a native window using AutoVideoSink.
//!
//! Run with: `cargo run --example 23_v4l2_display --features "v4l2,display"`
//!
//! Requirements:
//! - Linux system with V4L2 support
//! - A video capture device (webcam, etc.)
//! - User must be in the 'video' group or have permissions to /dev/video*

use std::sync::Arc;
use std::thread;

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::converters::{PixelFormat, VideoConvert};
use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::elements::app::AutoVideoSink;
use parallax::elements::device::{V4l2Src, enumerate_video_devices};
use parallax::error::Result;
use parallax::memory::{CpuArena, HeapSegment, MemorySegment};
use parallax::metadata::Metadata;

fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("parallax=info,warn")
        .init();

    println!("V4L2 Video Capture with Display");
    println!("================================\n");

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

    // Create converter if needed (YUYV -> RGBA for display)
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
        Some(PixelFormat::Rgb24) => {
            println!("  Creating RGB24 -> RGBA converter");
            Some(VideoConvert::new(
                PixelFormat::Rgb24,
                PixelFormat::Rgba,
                width,
                height,
            )?)
        }
        Some(PixelFormat::Rgba) => {
            println!("  No conversion needed (already RGBA)");
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

    // Create AutoVideoSink
    let mut sink = AutoVideoSink::new()
        .with_title(format!(
            "V4L2 Capture - {} ({}x{})",
            device_path, width, height
        ))
        .with_size(width, height);

    println!("\nStarting capture... Close window to stop.\n");

    // Create an arena for capture buffers
    let arena = CpuArena::new(input_size, 4)?;

    let mut frame_count = 0u64;
    let mut rgba_buffer = vec![0u8; output_size];

    // Main capture loop - runs until display window is closed
    loop {
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
                let captured_data = &ctx.output()[..len];

                if !display_frame(
                    captured_data,
                    &converter,
                    &mut rgba_buffer,
                    output_size,
                    frame_count,
                    &mut sink,
                ) {
                    println!("Display closed");
                    break;
                }

                if frame_count % 60 == 0 {
                    println!("Captured {} frames", frame_count);
                }
            }
            Ok(ProduceResult::OwnBuffer(buf)) => {
                frame_count += 1;
                let captured_data = buf.as_bytes();

                if !display_frame(
                    captured_data,
                    &converter,
                    &mut rgba_buffer,
                    output_size,
                    frame_count,
                    &mut sink,
                ) {
                    println!("Display closed");
                    break;
                }

                if frame_count % 60 == 0 {
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
    println!("\nDone! Total frames: {}", frame_count);
    Ok(())
}

/// Convert captured data to RGBA and send to the sink.
/// Returns true if successful, false if the sink is closed.
fn display_frame(
    captured_data: &[u8],
    converter: &Option<VideoConvert>,
    rgba_buffer: &mut [u8],
    output_size: usize,
    frame_count: u64,
    sink: &mut AutoVideoSink,
) -> bool {
    // Convert to RGBA if needed
    let rgba_data = if let Some(conv) = converter {
        if conv.convert(captured_data, rgba_buffer).is_ok() {
            &rgba_buffer[..]
        } else {
            return true; // Skip this frame but continue
        }
    } else {
        // No conversion - assume already RGBA or copy what we have
        let copy_len = captured_data.len().min(output_size);
        rgba_buffer[..copy_len].copy_from_slice(&captured_data[..copy_len]);
        // Pad with alpha if needed
        if copy_len < output_size {
            for i in (copy_len..output_size).step_by(4) {
                if i + 3 < output_size {
                    rgba_buffer[i + 3] = 255; // Alpha
                }
            }
        }
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
