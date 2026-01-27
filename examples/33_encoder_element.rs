//! Example 33: Encoder Element Wrapper
//!
//! Demonstrates using the `EncoderElement` wrapper to integrate video encoders
//! into pipelines. The wrapper handles:
//!
//! - Converting raw video frames to the encoder's input format
//! - Buffering and multiple output packets per frame
//! - Automatic flush at end-of-stream
//!
//! # Running
//!
//! ```bash
//! cargo run --example 33_encoder_element --features av1-encode
//! ```

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{
    ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
    SourceAdapter, TransformAdapter,
};
use parallax::elements::codec::{
    EncoderElement, PixelFormat, Rav1eConfig, Rav1eEncoder, VideoFrame,
};
use parallax::error::Result;
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::pipeline::Pipeline;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Source that generates raw video frames for encoding.
struct VideoFrameSource {
    width: u32,
    height: u32,
    frame_count: u32,
    max_frames: u32,
}

impl VideoFrameSource {
    fn new(width: u32, height: u32, max_frames: u32) -> Self {
        Self {
            width,
            height,
            frame_count: 0,
            max_frames,
        }
    }

    fn create_test_frame(&self, frame_num: u32) -> VideoFrame {
        let mut frame = VideoFrame::new(self.width, self.height, PixelFormat::I420);
        frame.pts = frame_num as i64;

        // Fill with a simple gradient pattern that changes each frame
        let y_plane_size = (self.width * self.height) as usize;
        let uv_width = self.width as usize / 2;
        let uv_height = self.height as usize / 2;

        // Y plane: gradient with frame-dependent offset
        for y in 0..self.height as usize {
            for x in 0..self.width as usize {
                let luma = ((x + y + frame_num as usize * 10) % 256) as u8;
                frame.data[y * self.width as usize + x] = luma;
            }
        }

        // U plane: neutral (128)
        for i in 0..uv_width * uv_height {
            frame.data[y_plane_size + i] = 128;
        }

        // V plane: neutral (128)
        for i in 0..uv_width * uv_height {
            frame.data[y_plane_size + uv_width * uv_height + i] = 128;
        }

        frame
    }
}

impl Source for VideoFrameSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.frame_count >= self.max_frames {
            return Ok(ProduceResult::Eos);
        }

        let frame = self.create_test_frame(self.frame_count);
        self.frame_count += 1;

        // Create buffer with frame data
        let segment = Arc::new(HeapSegment::new(frame.data.len())?);
        unsafe {
            std::ptr::copy_nonoverlapping(
                frame.data.as_ptr(),
                segment.as_mut_ptr().unwrap(),
                frame.data.len(),
            );
        }

        let mut metadata = parallax::metadata::Metadata::new();
        metadata.pts = parallax::clock::ClockTime::from_nanos(frame.pts as u64 * 33_333_333); // ~30fps

        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::from_segment(segment),
            metadata,
        )))
    }

    fn name(&self) -> &str {
        "video_frame_source"
    }
}

/// Sink that collects encoded AV1 packets.
struct PacketCollector {
    packets: Arc<AtomicUsize>,
    total_bytes: Arc<AtomicUsize>,
}

impl PacketCollector {
    fn new() -> Self {
        Self {
            packets: Arc::new(AtomicUsize::new(0)),
            total_bytes: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn packets(&self) -> Arc<AtomicUsize> {
        self.packets.clone()
    }

    fn total_bytes(&self) -> Arc<AtomicUsize> {
        self.total_bytes.clone()
    }
}

impl Sink for PacketCollector {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let buffer = ctx.buffer();
        let size = buffer.len();
        let packet_num = self.packets.fetch_add(1, Ordering::SeqCst) + 1;
        self.total_bytes.fetch_add(size, Ordering::SeqCst);

        println!(
            "Received AV1 packet #{}: {} bytes (pts: {:?})",
            packet_num,
            size,
            buffer.metadata().pts
        );

        Ok(())
    }

    fn name(&self) -> &str {
        "packet_collector"
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Encoder Element Example ===\n");

    // Configuration
    let width = 64;
    let height = 64;
    let num_frames = 10;

    println!(
        "Creating video source: {}x{}, {} frames",
        width, height, num_frames
    );

    // Create rav1e encoder configuration
    let encoder_config = Rav1eConfig::default()
        .dimensions(width as usize, height as usize)
        .speed(10) // Fastest preset for this example
        .quantizer(100);

    // Create the encoder
    let encoder = Rav1eEncoder::new(encoder_config)?;

    // Wrap in EncoderElement for pipeline integration
    // EncoderElement implements Transform and handles buffering + flush
    let encoder_element = EncoderElement::new(encoder, width, height);

    // Create pipeline components
    let source = VideoFrameSource::new(width, height, num_frames);
    let collector = PacketCollector::new();
    let packets = collector.packets();
    let total_bytes = collector.total_bytes();

    // Build the pipeline
    let mut pipeline = Pipeline::new();

    // Add source
    let src_id = pipeline.add_node(
        "source",
        DynAsyncElement::new_box(SourceAdapter::new(source)),
    );

    // Add encoder (using TransformAdapter since EncoderElement implements Transform)
    let enc_id = pipeline.add_node(
        "encoder",
        DynAsyncElement::new_box(TransformAdapter::new(encoder_element)),
    );

    // Add sink
    let sink_id = pipeline.add_node(
        "sink",
        DynAsyncElement::new_box(SinkAdapter::new(collector)),
    );

    // Connect: source -> encoder -> sink
    pipeline.link(src_id, enc_id)?;
    pipeline.link(enc_id, sink_id)?;

    println!("\nStarting pipeline...\n");

    // Run the pipeline
    pipeline.run().await?;

    // Print results
    let packet_count = packets.load(Ordering::SeqCst);
    let byte_count = total_bytes.load(Ordering::SeqCst);

    println!("\n=== Results ===");
    println!("Input frames: {}", num_frames);
    println!("Output packets: {}", packet_count);
    println!("Total encoded bytes: {}", byte_count);

    if packet_count > 0 {
        println!("Average packet size: {} bytes", byte_count / packet_count);
        let raw_size = num_frames as usize * width as usize * height as usize * 3 / 2;
        let compression_ratio = raw_size as f64 / byte_count as f64;
        println!("Compression ratio: {:.1}x", compression_ratio);
    }

    println!("\nNote: Encoder may buffer frames, so output packets may differ from input frames.");
    println!("The EncoderElement wrapper handles flush() at EOS to emit remaining packets.");

    Ok(())
}
