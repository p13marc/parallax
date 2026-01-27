//! Example 26: H.264 Video Codec
//!
//! This example demonstrates H.264 video encoding and decoding using OpenH264.
//!
//! - **H264Encoder**: Encodes YUV420 frames to H.264 NAL units
//! - **H264Decoder**: Decodes H.264 NAL units to YUV420 frames
//!
//! Run with: cargo run --example 26_h264_codec --features h264
//!
//! Note: Requires a C++ compiler (g++) to build OpenH264.

#[cfg(feature = "h264")]
fn main() {
    use parallax::elements::codec::{H264Decoder, H264Encoder, H264EncoderConfig};

    println!("=== H.264 Video Codec Example ===\n");

    // Configuration
    let width = 128;
    let height = 128;
    let num_frames = 30;

    println!("Configuration:");
    println!("  Resolution: {}x{}", width, height);
    println!("  Frames: {}", num_frames);
    println!();

    // Create encoder
    println!("1. Creating H.264 encoder...");
    let config = H264EncoderConfig::new(width, height)
        .frame_rate(30.0)
        .qp(26);

    let mut encoder = H264Encoder::new(config).expect("Failed to create encoder");
    println!("   Encoder created successfully");
    println!("   Config: {:?}", encoder.config());
    println!();

    // Create decoder
    println!("2. Creating H.264 decoder...");
    let mut decoder = H264Decoder::new().expect("Failed to create decoder");
    println!("   Decoder created successfully");
    println!();

    // Generate and encode frames
    println!("3. Encoding {} YUV420 frames...", num_frames);

    let y_size = width * height;
    let uv_size = (width / 2) * (height / 2);
    let frame_size = y_size + uv_size * 2;

    let mut encoded_frames = Vec::new();
    let mut total_encoded_bytes = 0;

    for frame_idx in 0..num_frames {
        // Generate a YUV420 test frame (moving gradient pattern)
        let mut yuv_data = vec![0u8; frame_size];

        // Y plane - moving diagonal gradient
        for y in 0..height {
            for x in 0..width {
                let offset = (frame_idx * 4) % 256;
                yuv_data[y * width + x] = ((x + y + offset) % 256) as u8;
            }
        }

        // U plane - constant (neutral blue)
        for i in 0..uv_size {
            yuv_data[y_size + i] = 128;
        }

        // V plane - constant (neutral red)
        for i in 0..uv_size {
            yuv_data[y_size + uv_size + i] = 128;
        }

        // Encode the frame
        let encoded = encoder
            .encode_yuv420(&yuv_data)
            .expect("Failed to encode frame");

        total_encoded_bytes += encoded.len();
        encoded_frames.push(encoded);

        if frame_idx == 0 || (frame_idx + 1) % 10 == 0 {
            println!(
                "   Frame {}: {} bytes encoded",
                frame_idx,
                encoded_frames.last().unwrap().len()
            );
        }
    }

    println!();
    println!("   Encoding complete!");
    println!("   Total frames: {}", encoder.frame_count());
    println!("   Total encoded: {} bytes", total_encoded_bytes);
    println!(
        "   Average frame size: {} bytes",
        total_encoded_bytes / num_frames
    );
    println!(
        "   Compression ratio: {:.2}x",
        (frame_size * num_frames) as f64 / total_encoded_bytes as f64
    );
    println!();

    // Decode the frames
    println!("4. Decoding {} H.264 frames...", encoded_frames.len());

    let mut decoded_count = 0;
    let mut total_decoded_bytes = 0;

    for (frame_idx, encoded) in encoded_frames.iter().enumerate() {
        match decoder.decode(encoded) {
            Ok(Some(frame)) => {
                decoded_count += 1;
                let yuv_data = frame.to_yuv420_planar();
                total_decoded_bytes += yuv_data.len();

                if decoded_count == 1 || decoded_count % 10 == 0 {
                    println!(
                        "   Frame {}: decoded {}x{} ({} bytes)",
                        frame_idx,
                        frame.width(),
                        frame.height(),
                        yuv_data.len()
                    );
                }
            }
            Ok(None) => {
                // Frame buffered, need more data
                if frame_idx < 5 {
                    println!("   Frame {}: buffered (need more data)", frame_idx);
                }
            }
            Err(e) => {
                println!("   Frame {}: decode error: {}", frame_idx, e);
            }
        }
    }

    // Flush remaining frames
    println!();
    println!("5. Flushing decoder...");
    match decoder.flush() {
        Ok(remaining) => {
            println!("   Flushed {} remaining frames", remaining.len());
            for frame in remaining {
                decoded_count += 1;
                total_decoded_bytes += frame.to_yuv420_planar().len();
            }
        }
        Err(e) => println!("   Flush error: {}", e),
    }

    println!();
    println!("   Decoding complete!");
    println!("   Total frames decoded: {}", decoded_count);
    println!("   Total decoded: {} bytes", total_decoded_bytes);
    println!(
        "   Decoder stats: {} frames, {} bytes",
        decoder.frame_count(),
        decoder.bytes_decoded()
    );
    println!();

    // Demonstrate different encoder configurations
    println!("6. Encoder configuration presets:");
    println!();

    let low_latency = H264EncoderConfig::low_latency(1920, 1080);
    println!("   Low latency (1920x1080):");
    println!("     Frame rate: {} fps", low_latency.max_frame_rate);
    println!(
        "     Keyframe interval: {} frames",
        low_latency.keyframe_interval
    );
    println!("     QP: {}", low_latency.qp);
    println!();

    let high_quality = H264EncoderConfig::high_quality(1920, 1080);
    println!("   High quality (1920x1080):");
    println!("     Frame rate: {} fps", high_quality.max_frame_rate);
    println!(
        "     Keyframe interval: {} frames",
        high_quality.keyframe_interval
    );
    println!("     QP: {}", high_quality.qp);
    println!();

    println!("=== Example Complete ===");
    println!();
    println!("Key concepts demonstrated:");
    println!("  - H264Encoder: YUV420 frames -> H.264 NAL units");
    println!("  - H264Decoder: H.264 NAL units -> YUV420 frames");
    println!("  - H264EncoderConfig: Bitrate, frame rate, QP, keyframe interval");
    println!("  - DecodedFrame: Access Y/U/V planes and convert to planar format");
    println!("  - Codec stats: Frame count, bytes encoded/decoded");
}

#[cfg(not(feature = "h264"))]
fn main() {
    println!("This example requires the 'h264' feature.");
    println!("Run with: cargo run --example 26_h264_codec --features h264");
    println!();
    println!("Note: Building with the h264 feature requires a C++ compiler:");
    println!("  - Fedora/RHEL: sudo dnf install gcc-c++");
    println!("  - Debian/Ubuntu: sudo apt install g++");
    println!("  - Arch: sudo pacman -S gcc");
    println!("  - macOS: xcode-select --install");
}
