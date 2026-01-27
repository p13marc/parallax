//! Example 27: RTSP to RTP Transcoding Pipeline
//!
//! This example demonstrates a real-world video transcoding pipeline:
//!
//! ```text
//! RTSP Camera → H.264 Decode → H.264 Encode (lower bitrate) → RTP Output
//! ```
//!
//! This is useful for:
//! - Reducing bandwidth from high-bitrate cameras
//! - Converting to a different quality/resolution
//! - Re-streaming camera feeds to multiple destinations
//!
//! # Usage
//!
//! ```bash
//! # With a real RTSP camera:
//! cargo run --example 27_rtsp_transcode_rtp --features "rtsp,h264" -- \
//!     --input rtsp://admin:password@192.168.1.100/stream1 \
//!     --output 192.168.1.50:5004 \
//!     --bitrate 1000000
//!
//! # To receive the output stream (using ffplay):
//! ffplay -protocol_whitelist file,udp,rtp -i receive.sdp
//! ```
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
//! │  RtspSrc    │────▶│ H264Decoder │────▶│ H264Encoder │────▶│  RtpSink    │
//! │ (camera)    │     │   (YUV420)  │     │ (low bitrate)│     │   (UDP)     │
//! └─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
//!       │                    │                   │                   │
//!   H.264 NALs          YUV frames          H.264 NALs          RTP packets
//! ```

#[cfg(all(feature = "rtsp", feature = "h264"))]
use std::env;
#[cfg(all(feature = "rtsp", feature = "h264"))]
use std::net::UdpSocket;
#[cfg(all(feature = "rtsp", feature = "h264"))]
use std::time::Instant;

#[cfg(all(feature = "rtsp", feature = "h264"))]
use parallax::elements::codec::{H264Decoder, H264Encoder, H264EncoderConfig};
#[cfg(all(feature = "rtsp", feature = "h264"))]
use parallax::elements::rtp::{RtspFrame, RtspSrc, RtspTransport, StreamSelection};

#[cfg(all(feature = "rtsp", feature = "h264"))]
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RTSP to RTP Transcoding Pipeline ===\n");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();

    let rtsp_url = args
        .iter()
        .position(|a| a == "--input")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("rtsp://localhost:8554/test");

    let output_addr = args
        .iter()
        .position(|a| a == "--output")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("127.0.0.1:5004");

    let target_bitrate: u32 = args
        .iter()
        .position(|a| a == "--bitrate")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(500_000); // 500 kbps default

    let max_frames: usize = args
        .iter()
        .position(|a| a == "--frames")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(300); // ~10 seconds at 30fps

    println!("Configuration:");
    println!("  RTSP Input:    {}", rtsp_url);
    println!("  RTP Output:    {}", output_addr);
    println!("  Target Bitrate: {} bps", target_bitrate);
    println!("  Max Frames:    {}", max_frames);
    println!();

    // Step 1: Connect to RTSP source
    println!("1. Connecting to RTSP source...");
    let rtsp_src = RtspSrc::new(rtsp_url)
        .with_transport(RtspTransport::TcpInterleaved)
        .with_stream_selection(StreamSelection::VideoOnly);

    let mut session = match rtsp_src.connect().await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("   Failed to connect: {}", e);
            eprintln!();
            eprintln!("To test without a real camera, you can use:");
            eprintln!("  1. VLC: vlc -vvv file.mp4 --sout '#rtp{{sdp=rtsp://:8554/test}}'");
            eprintln!(
                "  2. FFmpeg: ffmpeg -re -i file.mp4 -c copy -f rtsp rtsp://localhost:8554/test"
            );
            eprintln!(
                "  3. GStreamer: gst-launch-1.0 videotestsrc ! x264enc ! rtph264pay ! udpsink"
            );
            return Ok(());
        }
    };

    println!("   Connected!");
    println!("   Streams:");
    for stream in session.streams() {
        println!(
            "     [{}] {} - {} @ {} Hz",
            stream.index, stream.media_type, stream.codec, stream.clock_rate
        );
    }
    println!();

    // Step 2: Create H.264 decoder
    println!("2. Creating H.264 decoder...");
    let mut decoder = H264Decoder::new()?;
    println!("   Decoder ready");
    println!();

    // Step 3: Create H.264 encoder with lower bitrate
    // Note: We'll create this after we know the video dimensions
    println!("3. Will create H.264 encoder after detecting video dimensions...");
    println!();

    // Step 4: Create UDP socket for RTP output
    println!("4. Creating RTP output socket...");
    let output_socket = UdpSocket::bind("0.0.0.0:0")?;
    output_socket.connect(output_addr)?;
    println!(
        "   RTP output: {} -> {}",
        output_socket.local_addr()?,
        output_addr
    );
    println!();

    // Step 5: Run the transcoding pipeline
    println!("5. Starting transcoding pipeline...");
    println!("   Press Ctrl+C to stop");
    println!();

    let start_time = Instant::now();
    let mut frames_received = 0u64;
    let mut frames_decoded = 0u64;
    let mut frames_encoded = 0u64;
    let mut bytes_in = 0u64;
    let mut bytes_out = 0u64;
    let mut encoder: Option<H264Encoder> = None;
    let mut video_width = 0u32;
    let mut video_height = 0u32;
    let mut rtp_sequence: u16 = 0;
    let mut rtp_timestamp: u32 = 0;

    while frames_received < max_frames as u64 {
        // Receive frame from RTSP
        match session.next_frame().await {
            Ok(Some(frame)) => {
                if let RtspFrame::Video(buffer) = frame {
                    frames_received += 1;
                    let input_data = buffer.as_bytes();
                    bytes_in += input_data.len() as u64;

                    // Decode the H.264 frame
                    match decoder.decode(input_data) {
                        Ok(Some(decoded_frame)) => {
                            frames_decoded += 1;

                            // Initialize encoder on first decoded frame
                            if encoder.is_none() {
                                video_width = decoded_frame.width() as u32;
                                video_height = decoded_frame.height() as u32;

                                println!(
                                    "   Detected video: {}x{}, creating encoder...",
                                    video_width, video_height
                                );

                                let config = H264EncoderConfig::new(video_width, video_height)
                                    .bitrate(target_bitrate)
                                    .frame_rate(30.0)
                                    .keyframe_interval(30);

                                encoder = Some(H264Encoder::new(config)?);
                                println!(
                                    "   Encoder created with target bitrate: {} bps",
                                    target_bitrate
                                );
                                println!();
                            }

                            // Convert decoded frame to YUV420 planar
                            let yuv_data = decoded_frame.to_yuv420_planar();

                            // Encode with new settings
                            if let Some(ref mut enc) = encoder {
                                match enc.encode_yuv420(&yuv_data) {
                                    Ok(encoded_data) => {
                                        if !encoded_data.is_empty() {
                                            frames_encoded += 1;
                                            bytes_out += encoded_data.len() as u64;

                                            // Send as RTP packets
                                            // For simplicity, we send the whole NAL as one packet
                                            // In production, you'd use RtpH264Pay for proper fragmentation
                                            let rtp_packet = create_rtp_packet(
                                                &encoded_data,
                                                rtp_sequence,
                                                rtp_timestamp,
                                                96, // H.264 dynamic payload type
                                            );

                                            if let Err(e) = output_socket.send(&rtp_packet) {
                                                eprintln!("   RTP send error: {}", e);
                                            }

                                            rtp_sequence = rtp_sequence.wrapping_add(1);
                                            rtp_timestamp = rtp_timestamp.wrapping_add(3000); // 90kHz / 30fps
                                        }
                                    }
                                    Err(e) => {
                                        eprintln!("   Encode error: {}", e);
                                    }
                                }
                            }
                        }
                        Ok(None) => {
                            // Decoder needs more data
                        }
                        Err(e) => {
                            eprintln!("   Decode error: {}", e);
                        }
                    }

                    // Print progress
                    if frames_received % 30 == 0 {
                        let elapsed = start_time.elapsed().as_secs_f64();
                        let fps = frames_received as f64 / elapsed;
                        let bitrate_in = (bytes_in * 8) as f64 / elapsed / 1000.0;
                        let bitrate_out = (bytes_out * 8) as f64 / elapsed / 1000.0;
                        let ratio = if bitrate_in > 0.0 {
                            bitrate_out / bitrate_in * 100.0
                        } else {
                            0.0
                        };

                        println!(
                            "   Frame {:4} | {:.1} fps | In: {:6.0} kbps | Out: {:6.0} kbps | Ratio: {:.1}%",
                            frames_received, fps, bitrate_in, bitrate_out, ratio
                        );
                    }
                }
            }
            Ok(None) => {
                println!("   Stream ended");
                break;
            }
            Err(e) => {
                eprintln!("   RTSP error: {}", e);
                break;
            }
        }
    }

    // Print final statistics
    let elapsed = start_time.elapsed();
    println!();
    println!("=== Pipeline Statistics ===");
    println!("  Duration:        {:.2} seconds", elapsed.as_secs_f64());
    println!("  Frames received: {}", frames_received);
    println!("  Frames decoded:  {}", frames_decoded);
    println!("  Frames encoded:  {}", frames_encoded);
    println!(
        "  Bytes in:        {} ({:.2} MB)",
        bytes_in,
        bytes_in as f64 / 1_000_000.0
    );
    println!(
        "  Bytes out:       {} ({:.2} MB)",
        bytes_out,
        bytes_out as f64 / 1_000_000.0
    );

    if elapsed.as_secs_f64() > 0.0 {
        let fps = frames_received as f64 / elapsed.as_secs_f64();
        let bitrate_in = (bytes_in * 8) as f64 / elapsed.as_secs_f64() / 1000.0;
        let bitrate_out = (bytes_out * 8) as f64 / elapsed.as_secs_f64() / 1000.0;
        let compression = if bytes_in > 0 {
            (1.0 - bytes_out as f64 / bytes_in as f64) * 100.0
        } else {
            0.0
        };

        println!("  Average FPS:     {:.2}", fps);
        println!("  Input bitrate:   {:.2} kbps", bitrate_in);
        println!("  Output bitrate:  {:.2} kbps", bitrate_out);
        println!("  Compression:     {:.1}% reduction", compression);
    }

    println!();
    println!("=== Example Complete ===");
    println!();
    println!("Key concepts demonstrated:");
    println!("  - RtspSrc: Connect to RTSP cameras/servers");
    println!("  - H264Decoder: Decode H.264 NAL units to YUV420");
    println!("  - H264Encoder: Re-encode with different bitrate/quality");
    println!("  - UDP socket: Send RTP packets for streaming");
    println!();
    println!("For production use, consider adding:");
    println!("  - RtpH264Pay for proper NAL fragmentation");
    println!("  - RTCP for quality feedback");
    println!("  - Jitter buffer for smoother output");
    println!("  - Error recovery and reconnection");

    Ok(())
}

/// Create a simple RTP packet (for demonstration)
#[cfg(all(feature = "rtsp", feature = "h264"))]
fn create_rtp_packet(payload: &[u8], sequence: u16, timestamp: u32, payload_type: u8) -> Vec<u8> {
    let mut packet = Vec::with_capacity(12 + payload.len());

    // RTP header (12 bytes)
    packet.push(0x80); // V=2, P=0, X=0, CC=0
    packet.push(0x80 | payload_type); // M=1, PT
    packet.extend_from_slice(&sequence.to_be_bytes()); // Sequence number
    packet.extend_from_slice(&timestamp.to_be_bytes()); // Timestamp
    packet.extend_from_slice(&0x12345678u32.to_be_bytes()); // SSRC

    // Payload
    packet.extend_from_slice(payload);

    packet
}

#[cfg(not(all(feature = "rtsp", feature = "h264")))]
fn main() {
    println!("RTSP to RTP Transcoding Pipeline Example");
    println!();
    println!("This example requires the 'rtsp' and 'h264' features.");
    println!();
    println!("Run with:");
    println!("  cargo run --example 27_rtsp_transcode_rtp --features \"rtsp,h264\"");
    println!();
    println!("Build requirements:");
    println!("  - h264 feature requires a C++ compiler (g++)");
    println!("  - rtsp feature requires libretina");
    println!();
    println!("Install dependencies:");
    println!("  Fedora: sudo dnf install gcc-c++");
    println!("  Ubuntu: sudo apt install g++");
}
