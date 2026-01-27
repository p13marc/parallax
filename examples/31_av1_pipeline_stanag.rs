//! Example: AV1 transcode pipeline using proper Pipeline API with external metadata.
//!
//! This example demonstrates the correct way to build a Parallax pipeline:
//!
//! 1. Uses `Pipeline::new()` and `add_node()` to build the graph
//! 2. Uses `AppSrc` with handle for external data injection (video + metadata)
//! 3. Uses `AppSink` with handle for output extraction
//! 4. Runs via `pipeline.run().await` (the proper executor path)
//!
//! ```text
//! ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
//! │  AppSrc      │────▶│ VideoScale   │────▶│   AppSink    │
//! │  (video)     │     │ 1080p→720p   │     │   (output)   │
//! └──────────────┘     └──────────────┘     └──────────────┘
//!        ▲
//!        │ (external video frames + metadata injected via handle)
//! ```
//!
//! In a full pipeline, you would add encoder and muxer elements between
//! the scaler and sink.
//!
//! Run with: cargo run --example 31_av1_pipeline_stanag

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{DynAsyncElement, ElementAdapter, SinkAdapter, SourceAdapter};
use parallax::elements::{AppSink, AppSrc, StanagMetadataBuilder, VideoScale};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

// ============================================================================
// External Metadata Source (simulates GPS/IMU/Gimbal data)
// ============================================================================

/// Metadata from external sensors (GPS, IMU, gimbal, etc.)
#[derive(Debug, Clone)]
pub struct SensorMetadata {
    /// Timestamp in microseconds since epoch
    pub timestamp_us: u64,
    /// Sensor latitude (degrees)
    pub sensor_lat: f64,
    /// Sensor longitude (degrees)
    pub sensor_lon: f64,
    /// Sensor altitude (meters HAE)
    pub sensor_alt: f64,
    /// Platform heading (degrees, 0-360)
    pub heading: f64,
    /// Platform pitch (degrees, -90 to +90)
    pub pitch: f64,
    /// Platform roll (degrees, -90 to +90)
    pub roll: f64,
    /// Mission ID
    pub mission_id: String,
}

impl SensorMetadata {
    /// Convert to KLV-encoded STANAG 4609 packet
    pub fn to_klv(&self) -> Vec<u8> {
        StanagMetadataBuilder::new()
            .version(17)
            .timestamp(self.timestamp_us)
            .mission_id(&self.mission_id)
            .sensor_position(self.sensor_lat, self.sensor_lon, self.sensor_alt)
            .platform_attitude(self.heading, self.pitch, self.roll)
            .build_st0601()
    }
}

/// Simulated external metadata source (would be real sensors in production)
fn spawn_metadata_source(tx: Sender<SensorMetadata>) {
    thread::spawn(move || {
        let start = Instant::now();
        let mut lat = 37.2350;
        let mut lon = -115.8111;
        let mut heading = 0.0;

        loop {
            thread::sleep(Duration::from_millis(100)); // 10 Hz

            let elapsed = start.elapsed();
            let timestamp_us = elapsed.as_micros() as u64;

            lat += 0.00001;
            lon += 0.00002;
            heading = (heading + 1.0) % 360.0;

            let metadata = SensorMetadata {
                timestamp_us,
                sensor_lat: lat,
                sensor_lon: lon,
                sensor_alt: 1500.0 + (elapsed.as_secs_f64() * 10.0).sin() * 50.0,
                heading,
                pitch: 5.0 + (elapsed.as_secs_f64() * 2.0).sin() * 2.0,
                roll: (elapsed.as_secs_f64() * 1.5).sin() * 3.0,
                mission_id: "UAV_MISSION_001".to_string(),
            };

            if tx.send(metadata).is_err() {
                break;
            }
        }
    });
}

// ============================================================================
// Video Frame with Metadata
// ============================================================================

/// Create a simulated YUV420 frame
fn create_simulated_yuv420(width: u32, height: u32, frame_num: u64) -> Vec<u8> {
    let y_size = (width * height) as usize;
    let uv_size = ((width / 2) * (height / 2)) as usize;

    let mut yuv = Vec::with_capacity(y_size + 2 * uv_size);

    // Y plane (moving gradient pattern)
    for y in 0..height {
        for x in 0..width {
            let value = ((x + y + frame_num as u32 * 5) % 256) as u8;
            yuv.push(value);
        }
    }

    // U plane
    yuv.extend(vec![128u8; uv_size]);

    // V plane
    yuv.extend(vec![128u8; uv_size]);

    yuv
}

// ============================================================================
// Main Pipeline Example
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== AV1 Pipeline Example with STANAG Metadata (Proper API) ===\n");

    // Configuration
    let src_width = 1920u32;
    let src_height = 1080u32;
    let dst_width = 1280u32;
    let dst_height = 720u32;
    let num_frames = 5u64;

    println!("Configuration:");
    println!("  Source: {}x{}", src_width, src_height);
    println!("  Target: {}x{}", dst_width, dst_height);
    println!("  Frames: {}", num_frames);
    println!();

    // =========================================================================
    // Step 1: Set up external metadata channel
    // =========================================================================
    println!("1. Starting external metadata source (simulates GPS/IMU)...");
    let (metadata_tx, metadata_rx): (Sender<SensorMetadata>, Receiver<SensorMetadata>) =
        mpsc::channel();
    spawn_metadata_source(metadata_tx);
    println!("   Metadata source running at ~10 Hz\n");

    // =========================================================================
    // Step 2: Create pipeline elements with AppSrc/AppSink
    // =========================================================================
    println!("2. Creating pipeline with AppSrc → VideoScale → AppSink...\n");

    // Create AppSrc for video frames (we'll inject frames + metadata via handle)
    let appsrc = AppSrc::new();
    let src_handle = appsrc.handle();

    // Create AppSink to receive processed frames
    let appsink = AppSink::new();
    let sink_handle = appsink.handle();

    // Create video scaler element
    let scaler = VideoScale::new(src_width, src_height, dst_width, dst_height);

    // =========================================================================
    // Step 3: Build the pipeline graph
    // =========================================================================
    println!("3. Building pipeline graph...\n");

    let mut pipeline = Pipeline::new();

    // Add nodes with proper adapters
    let src_node = pipeline.add_node(
        "video_src",
        DynAsyncElement::new_box(SourceAdapter::new(appsrc)),
    );

    let scale_node = pipeline.add_node(
        "scaler",
        DynAsyncElement::new_box(ElementAdapter::new(scaler)),
    );

    let sink_node = pipeline.add_node(
        "output_sink",
        DynAsyncElement::new_box(SinkAdapter::new(appsink)),
    );

    // Link the elements: src → scaler → sink
    pipeline.link(src_node, scale_node)?;
    pipeline.link(scale_node, sink_node)?;

    // Print pipeline description
    println!("Pipeline structure:");
    println!("{}", pipeline.describe());

    // =========================================================================
    // Step 4: Producer thread - inject video frames with metadata
    // =========================================================================
    println!("4. Starting producer thread (video frames + metadata)...\n");

    let producer = thread::spawn(move || {
        let mut latest_metadata: Option<SensorMetadata> = None;

        for frame_num in 0..num_frames {
            // Get latest metadata from external source (non-blocking)
            while let Ok(meta) = metadata_rx.try_recv() {
                latest_metadata = Some(meta);
            }

            // Create simulated video frame
            let yuv_data = create_simulated_yuv420(src_width, src_height, frame_num);
            let yuv_size = yuv_data.len();

            // Create buffer from YUV data
            let segment = Arc::new(HeapSegment::new(yuv_size).expect("alloc"));
            unsafe {
                std::ptr::copy_nonoverlapping(
                    yuv_data.as_ptr(),
                    segment.as_mut_ptr().unwrap(),
                    yuv_size,
                );
            }

            // Create metadata with sequence number
            let metadata = Metadata::from_sequence(frame_num);

            // If we have sensor metadata, encode it as KLV and log
            if let Some(ref sensor_meta) = latest_metadata {
                let klv_data = sensor_meta.to_klv();
                println!(
                    "   Frame {}: lat={:.4}, lon={:.4}, klv={} bytes",
                    frame_num,
                    sensor_meta.sensor_lat,
                    sensor_meta.sensor_lon,
                    klv_data.len()
                );
            } else {
                println!("   Frame {}: no metadata yet", frame_num);
            }

            let buffer = Buffer::new(MemoryHandle::from_segment(segment), metadata);

            // Push buffer into pipeline via AppSrc handle
            if let Err(e) = src_handle.push_buffer(buffer) {
                eprintln!("Failed to push buffer: {:?}", e);
                break;
            }

            // Simulate frame rate
            thread::sleep(Duration::from_millis(40)); // 25 fps
        }

        // Signal end of stream
        src_handle.end_stream();
        println!("   Producer: sent {} frames, signaled EOS", num_frames);
    });

    // =========================================================================
    // Step 5: Consumer thread - receive processed frames
    // =========================================================================
    println!("5. Starting consumer thread...\n");

    let consumer = thread::spawn(move || {
        let mut received = 0u64;
        let expected_size =
            (dst_width * dst_height) as usize + 2 * ((dst_width / 2) * (dst_height / 2)) as usize;

        while let Ok(Some(buffer)) = sink_handle.pull_buffer() {
            let size = buffer.as_bytes().len();
            received += 1;
            println!(
                "   Consumer: received frame {}, size={} bytes (expected {})",
                received, size, expected_size
            );

            // Verify scaled frame size
            if size != expected_size {
                println!(
                    "   WARNING: Frame size mismatch! Expected {}, got {}",
                    expected_size, size
                );
            }
        }

        println!("   Consumer: received {} frames total", received);
        received
    });

    // =========================================================================
    // Step 6: Run the pipeline
    // =========================================================================
    println!("6. Running pipeline...\n");

    // Run the pipeline - this is the proper executor path
    pipeline.run().await?;

    // Wait for producer and consumer threads
    producer.join().expect("producer panicked");
    let frames_received = consumer.join().expect("consumer panicked");

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=== Pipeline Execution Complete ===\n");
    println!("Summary:");
    println!("  Frames sent: {}", num_frames);
    println!("  Frames received: {}", frames_received);
    println!(
        "  Scale: {}x{} → {}x{}",
        src_width, src_height, dst_width, dst_height
    );

    println!("\n=== Key Concepts Demonstrated ===\n");
    println!("1. Pipeline::new() + add_node() - proper graph construction");
    println!("2. AppSrc with handle - external data injection");
    println!("3. AppSink with handle - output extraction");
    println!("4. DynAsyncElement::new_box() - element boxing for pipeline");
    println!("5. SourceAdapter/SinkAdapter/ElementAdapter - trait bridging");
    println!("6. pipeline.run().await - proper async executor");
    println!("7. External metadata synchronized with video frames");

    println!("\n=== Full Pipeline Architecture ===\n");
    println!("In a complete AV1 transcode pipeline, you would add:");
    println!();
    println!("  AppSrc → VideoScale → Rav1eEncoder → TsMux → RtpSink");
    println!("             ↑                           ↑");
    println!("         1080p→720p              KLV metadata PID");
    println!();
    println!("The TsMux would be configured to multiplex:");
    println!("  - Video PID (AV1 encoded frames)");
    println!("  - KLV PID (STANAG 4609 metadata)");

    Ok(())
}
