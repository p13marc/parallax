//! DMA-BUF Format + Memory Negotiation Example.
//!
//! This example demonstrates Parallax's automatic format + memory negotiation:
//! - V4L2 source declares both DMA-BUF and CPU memory capabilities
//! - A simulated GPU sink prefers DMA-BUF memory
//! - Pipeline automatically negotiates DMA-BUF path (zero-copy)
//!
//! Run with: `cargo run --example 45_dmabuf_negotiation --features v4l2`
//!
//! Requirements:
//! - Linux system with V4L2 support
//! - A video capture device that supports VIDIOC_EXPBUF (most modern cameras)
//! - User must be in the 'video' group or have permissions to /dev/video*
//!
//! Note: Not all V4L2 devices support DMA-BUF export. If your device doesn't
//! support it, you'll see an error during pipeline setup.

use parallax::buffer::Buffer;
use parallax::element::{SimpleSink, Snk, Source};
use parallax::elements::device::{V4l2Config, V4l2Src, enumerate_video_devices};
use parallax::error::Result;
use parallax::format::Caps;
use parallax::memory::MemoryType;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

/// A sink that tracks memory types of received buffers.
///
/// In a real application, this would be a GPU video sink that imports
/// the DMA-BUF directly into Vulkan/OpenGL for rendering.
struct GpuPreferredSink {
    frames_received: Arc<AtomicU32>,
    dmabuf_frames: Arc<AtomicU32>,
    cpu_frames: Arc<AtomicU32>,
}

impl GpuPreferredSink {
    fn new() -> Self {
        Self {
            frames_received: Arc::new(AtomicU32::new(0)),
            dmabuf_frames: Arc::new(AtomicU32::new(0)),
            cpu_frames: Arc::new(AtomicU32::new(0)),
        }
    }
}

impl SimpleSink for GpuPreferredSink {
    fn consume(&mut self, buffer: &Buffer) -> Result<()> {
        let frame_num = self.frames_received.fetch_add(1, Ordering::Relaxed) + 1;

        // Check memory type
        let mem_type = buffer.memory_type();
        match mem_type {
            MemoryType::DmaBuf => {
                self.dmabuf_frames.fetch_add(1, Ordering::Relaxed);
            }
            MemoryType::Cpu => {
                self.cpu_frames.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }

        if frame_num <= 5 || frame_num % 30 == 0 {
            println!(
                "Frame {}: {} bytes, memory type: {:?}",
                frame_num,
                buffer.len(),
                mem_type
            );
        }

        Ok(())
    }

    fn input_caps(&self) -> Caps {
        // Accept any video format - the memory type negotiation happens
        // at a different level (ElementMediaCaps)
        Caps::any()
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("parallax=info")
        .init();

    println!("DMA-BUF Format + Memory Negotiation Example");
    println!("============================================\n");

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

    println!("Found {} device(s):", devices.len());
    for dev in &devices {
        println!("  {} - {}", dev.id, dev.name);
    }

    // Use first device
    let device_path = &devices[0].id;
    println!("\nUsing device: {}\n", device_path);

    // Create V4L2 source with DMA-BUF export enabled
    let config = V4l2Config {
        width: 640,
        height: 480,
        dmabuf_export: true, // Enable DMA-BUF export
        ..Default::default()
    };

    let camera = match V4l2Src::with_config(device_path, config) {
        Ok(src) => src,
        Err(e) => {
            println!("Failed to create V4L2 source with DMA-BUF: {}", e);
            println!("\nThis may happen if your V4L2 device doesn't support VIDIOC_EXPBUF.");
            println!("Trying without DMA-BUF export...\n");

            // Fallback: try without DMA-BUF
            let fallback_config = V4l2Config {
                width: 640,
                height: 480,
                dmabuf_export: false,
                ..Default::default()
            };
            V4l2Src::with_config(device_path, fallback_config)?
        }
    };

    println!("Camera configuration:");
    println!("  Resolution: {}x{}", camera.width(), camera.height());
    println!(
        "  Format: {}",
        std::str::from_utf8(camera.fourcc()).unwrap_or("????")
    );
    println!("  DMA-BUF export: {}", camera.is_dmabuf_export());

    // Show camera capabilities (using Source trait)
    println!("\nCamera output capabilities (ElementMediaCaps):");
    for (i, cap) in camera.output_media_caps().iter().enumerate() {
        println!(
            "  [{}] format: {:?}, memory: {:?}",
            i, cap.format, cap.memory
        );
    }

    // Create GPU sink and keep handles for stats
    let sink = GpuPreferredSink::new();
    let frames_received = sink.frames_received.clone();
    let dmabuf_frames = sink.dmabuf_frames.clone();
    let cpu_frames = sink.cpu_frames.clone();

    // Build pipeline: v4l2src (DmaBuf) -> gpu_sink
    let mut pipeline = Pipeline::new();

    let src_node = pipeline.add_source("camera", camera);
    let sink_node = pipeline.add_element("gpu_sink", Snk(sink));
    pipeline.link(src_node, sink_node)?;

    println!("\n--- Pipeline Negotiation ---");
    println!("The pipeline automatically negotiates the best format + memory combination.");
    println!("Since source has dmabuf_export=true, it declares DMA-BUF as preferred.");
    println!("Buffers will be backed by DMA-BUF file descriptors for zero-copy GPU access.\n");

    println!("Capturing for 3 seconds...\n");

    // Run for a short duration
    tokio::select! {
        result = pipeline.run() => {
            if let Err(e) = result {
                eprintln!("Pipeline error: {}", e);
            }
        }
        _ = tokio::time::sleep(Duration::from_secs(3)) => {
            println!("\nCapture complete!");
        }
    }

    // Print statistics
    let total = frames_received.load(Ordering::Relaxed);
    let dmabuf = dmabuf_frames.load(Ordering::Relaxed);
    let cpu = cpu_frames.load(Ordering::Relaxed);

    println!("\n--- Statistics ---");
    println!("Total frames received: {}", total);
    println!(
        "DMA-BUF frames: {} ({:.1}%)",
        dmabuf,
        100.0 * dmabuf as f32 / total.max(1) as f32
    );
    println!(
        "CPU frames: {} ({:.1}%)",
        cpu,
        100.0 * cpu as f32 / total.max(1) as f32
    );

    if dmabuf > 0 {
        println!("\nSuccess! DMA-BUF negotiation worked - zero-copy path achieved!");
    } else if total > 0 {
        println!("\nNote: Using CPU path (device may not support DMA-BUF export).");
    }

    println!("\nDone!");
    Ok(())
}
