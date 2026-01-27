//! Example: Complete RTSP to RTP/UDP transcode pipeline with STANAG metadata injection.
//!
//! This example demonstrates a real-world UAV video pipeline:
//!
//! ```text
//! RTSP (H.264/MPEG-TS) → Demux → Decode → Scale → Encode → Mux (+ KLV) → RTP/UDP
//! ```
//!
//! Features:
//! - Receives H.264 video over RTSP
//! - Demultiplexes MPEG-TS to extract video frames
//! - Decodes H.264 to YUV420
//! - Scales video to target resolution
//! - Re-encodes with new bitrate
//! - Injects STANAG 4609 / MISB ST 0601 metadata from external source
//! - Multiplexes video + metadata into MPEG-TS
//! - Sends over RTP/UDP
//!
//! The STANAG metadata comes from an external channel (simulated here with a channel),
//! representing real-world scenarios where metadata comes from:
//! - GPS receiver
//! - IMU/INS
//! - Gimbal controller
//! - Mission computer
//!
//! Run with: cargo run --example 29_rtsp_transcode_stanag --features "rtsp,h264,mpeg-ts"

use std::net::UdpSocket;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use parallax::clock::ClockTime;
use parallax::elements::{
    // STANAG/KLV
    StanagMetadataBuilder,
    // MPEG-TS
    TsDemux,
    TsMux,
    TsMuxConfig,
    TsMuxStreamType,
    TsMuxTrack,
    TsStreamType,
    // Video processing
    VideoScale,
};

#[cfg(feature = "h264")]
use parallax::elements::{H264Decoder, H264Encoder, H264EncoderConfig};

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
    /// Frame center latitude (degrees)
    pub frame_center_lat: f64,
    /// Frame center longitude (degrees)
    pub frame_center_lon: f64,
    /// Platform heading (degrees, 0-360)
    pub heading: f64,
    /// Platform pitch (degrees, -90 to +90)
    pub pitch: f64,
    /// Platform roll (degrees, -90 to +90)
    pub roll: f64,
    /// Slant range to target (meters)
    pub slant_range: f64,
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
            .frame_center(self.frame_center_lat, self.frame_center_lon)
            .platform_attitude(self.heading, self.pitch, self.roll)
            .slant_range(self.slant_range)
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
            // Simulate sensor data at ~10 Hz
            thread::sleep(Duration::from_millis(100));

            let elapsed = start.elapsed();
            let timestamp_us = elapsed.as_micros() as u64;

            // Simulate movement
            lat += 0.00001;
            lon += 0.00002;
            heading = (heading + 1.0) % 360.0;

            let metadata = SensorMetadata {
                timestamp_us,
                sensor_lat: lat,
                sensor_lon: lon,
                sensor_alt: 1500.0 + (elapsed.as_secs_f64() * 10.0).sin() * 50.0,
                frame_center_lat: lat - 0.001,
                frame_center_lon: lon - 0.001,
                heading,
                pitch: 5.0 + (elapsed.as_secs_f64() * 2.0).sin() * 2.0,
                roll: (elapsed.as_secs_f64() * 1.5).sin() * 3.0,
                slant_range: 2500.0,
                mission_id: "UAV_MISSION_001".to_string(),
            };

            if tx.send(metadata).is_err() {
                break; // Receiver dropped, exit thread
            }
        }
    });
}

// ============================================================================
// Pipeline Configuration
// ============================================================================

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct TranscodeConfig {
    /// RTSP source URL
    pub rtsp_url: String,
    /// Output UDP destination (host:port)
    pub udp_dest: String,
    /// Target video width
    pub target_width: u32,
    /// Target video height
    pub target_height: u32,
    /// Target bitrate (bps)
    pub target_bitrate: u32,
    /// Target frame rate
    pub target_fps: f32,
    /// Video PID in output TS
    pub video_pid: u16,
    /// KLV metadata PID in output TS
    pub klv_pid: u16,
}

impl Default for TranscodeConfig {
    fn default() -> Self {
        Self {
            rtsp_url: "rtsp://camera.example.com/stream".to_string(),
            udp_dest: "239.1.1.1:5004".to_string(),
            target_width: 1280,
            target_height: 720,
            target_bitrate: 2_000_000,
            target_fps: 25.0,
            video_pid: 256,
            klv_pid: 257,
        }
    }
}

// ============================================================================
// Main Pipeline (Simulation Mode)
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== RTSP to RTP/UDP Transcode with STANAG Metadata ===\n");

    // Since we can't connect to a real RTSP server in this example,
    // we'll demonstrate the complete pipeline architecture with simulated data.
    run_simulated_pipeline()?;

    Ok(())
}

/// Run a simulated version of the pipeline to demonstrate the architecture.
///
/// In production, you would replace the simulated video source with actual RTSP.
fn run_simulated_pipeline() -> Result<(), Box<dyn std::error::Error>> {
    let config = TranscodeConfig {
        rtsp_url: "rtsp://192.168.1.100/stream1".to_string(),
        udp_dest: "127.0.0.1:5004".to_string(),
        target_width: 640,
        target_height: 480,
        target_bitrate: 1_000_000,
        target_fps: 25.0,
        video_pid: 256,
        klv_pid: 257,
    };

    println!("Configuration:");
    println!("  RTSP URL: {} (simulated)", config.rtsp_url);
    println!("  UDP Output: {}", config.udp_dest);
    println!(
        "  Target: {}x{} @ {} bps",
        config.target_width, config.target_height, config.target_bitrate
    );
    println!(
        "  Video PID: {}, KLV PID: {}",
        config.video_pid, config.klv_pid
    );
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
    // Step 2: Initialize MPEG-TS demuxer (for incoming stream)
    // =========================================================================
    println!("2. Initializing MPEG-TS demuxer...");
    let mut demux = TsDemux::video_only();
    println!("   Demuxer ready (video streams only)\n");

    // =========================================================================
    // Step 3: Initialize H.264 codec (feature-gated)
    // =========================================================================
    #[cfg(feature = "h264")]
    let (mut decoder, mut encoder) = {
        println!("3. Initializing H.264 codec...");
        let decoder = H264Decoder::new()?;
        let encoder_config = H264EncoderConfig::new(config.target_width, config.target_height)
            .bitrate(config.target_bitrate)
            .frame_rate(config.target_fps);
        let encoder = H264Encoder::new(encoder_config)?;
        println!(
            "   Encoder: {}x{} @ {} bps\n",
            config.target_width, config.target_height, config.target_bitrate
        );
        (decoder, encoder)
    };

    #[cfg(not(feature = "h264"))]
    {
        println!("3. H.264 codec not available (enable 'h264' feature)");
        println!("   Skipping encode/decode steps\n");
    }

    // =========================================================================
    // Step 4: Initialize video scaler
    // =========================================================================
    println!("4. Initializing video scaler...");
    // Assume source is 1920x1080, scale to target
    let mut scaler = VideoScale::new(1920, 1080, config.target_width, config.target_height);
    println!(
        "   Scale: 1920x1080 → {}x{}\n",
        config.target_width, config.target_height
    );

    // =========================================================================
    // Step 5: Initialize MPEG-TS muxer (for output)
    // =========================================================================
    println!("5. Initializing MPEG-TS muxer...");
    let mux_config = TsMuxConfig::new()
        .program_number(1)
        .pmt_pid(0x1000)
        .pcr_interval_ms(40)
        .add_track(TsMuxTrack::new(config.video_pid, TsMuxStreamType::H264).video())
        .add_track(TsMuxTrack::new(config.klv_pid, TsMuxStreamType::Klv).private_data());
    let mut mux = TsMux::new(mux_config);
    println!(
        "   Tracks: Video (PID {}), KLV (PID {})\n",
        config.video_pid, config.klv_pid
    );

    // =========================================================================
    // Step 6: Initialize UDP socket for RTP output
    // =========================================================================
    println!("6. Initializing UDP socket for RTP output...");
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    socket.set_nonblocking(true)?;
    println!("   Bound to: {}", socket.local_addr()?);
    println!("   Destination: {}\n", config.udp_dest);

    // =========================================================================
    // Step 7: Process simulated frames
    // =========================================================================
    println!("7. Processing pipeline (5 simulated frames)...\n");

    let mut latest_metadata: Option<SensorMetadata> = None;
    let mut total_ts_bytes = 0;
    let mut rtp_sequence: u16 = 0;
    let ssrc: u32 = rand::random();

    // Write PSI tables first
    let psi = mux.write_psi();
    total_ts_bytes += psi.len();
    println!("   PSI tables: {} bytes", psi.len());

    for frame_num in 0..5 {
        let pts = ClockTime::from_millis(frame_num * 40);

        // Drain metadata channel (get latest)
        while let Ok(meta) = metadata_rx.try_recv() {
            latest_metadata = Some(meta);
        }

        // Simulate incoming MPEG-TS with H.264 video
        let simulated_ts = create_simulated_ts_packet(frame_num);

        // Demux the TS packet
        let frames = demux.push(&simulated_ts)?;
        let demux_info = if frames.is_empty() {
            "parsing...".to_string()
        } else {
            format!("{} frame(s)", frames.len())
        };

        // In a real pipeline, we would:
        // 1. Decode H.264 to YUV420
        // 2. Scale YUV420 to target resolution
        // 3. Re-encode to H.264 with new bitrate

        #[cfg(feature = "h264")]
        let encoded_frame = {
            // For simulation, create a fake YUV frame
            let src_yuv = create_simulated_yuv420(1920, 1080);

            // Scale to target resolution
            let scaled_yuv = scaler.scale_yuv420(&src_yuv)?;

            // Encode (in real pipeline, this would produce H.264 NAL units)
            // For now, just use the encoded frame directly
            create_fake_h264_nal(frame_num)
        };

        #[cfg(not(feature = "h264"))]
        let encoded_frame = create_fake_h264_nal(frame_num);

        // Write video to muxer
        let video_ts = mux.write_pes(config.video_pid, &encoded_frame, Some(pts), None)?;
        total_ts_bytes += video_ts.len();

        // Inject STANAG metadata (synchronized with video frame)
        let klv_ts = if let Some(ref meta) = latest_metadata {
            let klv_data = meta.to_klv();
            let ts = mux.write_pes(config.klv_pid, &klv_data, Some(pts), None)?;
            total_ts_bytes += ts.len();
            ts
        } else {
            Vec::new()
        };

        // Combine video + metadata TS packets
        let mut output_ts = video_ts.clone();
        output_ts.extend(&klv_ts);

        // Send over RTP/UDP (TS packets in RTP payload)
        let rtp_packets = packetize_ts_to_rtp(&output_ts, &mut rtp_sequence, ssrc, pts);

        // Send RTP packets (would actually send in production)
        for rtp in &rtp_packets {
            // socket.send_to(rtp, &config.udp_dest)?;
            let _ = rtp; // Suppress unused warning
        }

        let meta_info = latest_metadata
            .as_ref()
            .map(|m| format!("lat={:.4}, lon={:.4}", m.sensor_lat, m.sensor_lon))
            .unwrap_or_else(|| "none".to_string());

        println!(
            "   Frame {}: demux={}, video={} bytes, klv={} bytes, rtp={} pkts, meta=[{}]",
            frame_num,
            demux_info,
            video_ts.len(),
            klv_ts.len(),
            rtp_packets.len(),
            meta_info
        );
    }

    // =========================================================================
    // Step 8: Statistics
    // =========================================================================
    println!("\n8. Pipeline statistics:\n");
    let mux_stats = mux.stats();
    println!(
        "   Demux: {} packets processed",
        demux.stats().packets_processed
    );
    println!(
        "   Mux: {} TS packets, {} bytes",
        mux_stats.packets_written, mux_stats.bytes_written
    );
    println!(
        "   Mux: {} PES packets, {} PCR insertions",
        mux_stats.pes_packets, mux_stats.pcr_count
    );
    println!("   Scaler: {} frames processed", scaler.frames_processed());
    println!("   Total TS output: {} bytes", total_ts_bytes);

    // =========================================================================
    // Summary
    // =========================================================================
    println!("\n=== Pipeline Architecture ===\n");
    println!("┌─────────────┐     ┌─────────────┐     ┌─────────────┐");
    println!("│  RTSP Src   │────▶│  TS Demux   │────▶│  H.264 Dec  │");
    println!("│  (H.264/TS) │     │             │     │             │");
    println!("└─────────────┘     └─────────────┘     └──────┬──────┘");
    println!("                                               │");
    println!("                                               ▼");
    println!("┌─────────────┐     ┌─────────────┐     ┌─────────────┐");
    println!("│ Sensor Data │     │  TS Mux     │◀────│ Video Scale │");
    println!("│ (GPS/IMU)   │────▶│  (+KLV PID) │     │ 1080p→720p  │");
    println!("└─────────────┘     └──────┬──────┘     └──────┬──────┘");
    println!("                           │                   │");
    println!("                           ▼                   │");
    println!("                    ┌─────────────┐     ┌──────┴──────┐");
    println!("                    │  RTP/UDP    │     │  H.264 Enc  │");
    println!("                    │  Output     │◀────│  (new bps)  │");
    println!("                    └─────────────┘     └─────────────┘");

    println!("\n=== Key Points ===\n");
    println!("1. External metadata (SensorMetadata) comes via channel from sensors");
    println!("2. Metadata is converted to KLV using StanagMetadataBuilder");
    println!("3. Video and KLV are muxed with separate PIDs into single TS stream");
    println!("4. TS packets are packetized into RTP for UDP transmission");
    println!("5. All processing is frame-synchronized via PTS timestamps");

    println!("\n=== Example Complete ===");
    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simulated TS packet (for demonstration)
fn create_simulated_ts_packet(frame_num: u64) -> Vec<u8> {
    // Create a minimal valid-ish TS packet
    let mut packet = vec![0x47]; // Sync byte
    packet.extend(&[0x40, 0x00]); // PUSI=1, PID=0 (PAT)
    packet.extend(&[0x10]); // Payload only, CC=0
    packet.extend(vec![0xFF; 184]); // Stuffing

    // Add more packets for a "frame"
    for _ in 0..10 {
        packet.push(0x47);
        packet.extend(&[0x01, 0x00]); // PID=256 (video)
        packet.extend(&[0x10]);
        packet.extend(vec![0xAB; 184]);
    }

    let _ = frame_num; // Used conceptually
    packet
}

/// Create a simulated YUV420 frame
fn create_simulated_yuv420(width: u32, height: u32) -> Vec<u8> {
    let y_size = (width * height) as usize;
    let uv_size = ((width / 2) * (height / 2)) as usize;

    let mut yuv = Vec::with_capacity(y_size + 2 * uv_size);

    // Y plane (gray gradient)
    for y in 0..height {
        for x in 0..width {
            let value = ((x + y) % 256) as u8;
            yuv.push(value);
        }
    }

    // U plane (neutral)
    yuv.extend(vec![128u8; uv_size]);

    // V plane (neutral)
    yuv.extend(vec![128u8; uv_size]);

    yuv
}

/// Create a fake H.264 NAL unit for demonstration
fn create_fake_h264_nal(frame_num: u64) -> Vec<u8> {
    let mut nal = vec![0x00, 0x00, 0x00, 0x01]; // Start code

    if frame_num % 30 == 0 {
        nal.push(0x65); // IDR
    } else {
        nal.push(0x41); // P-frame
    }

    nal.extend(vec![0xAB; 200 + (frame_num as usize % 100)]);
    nal
}

/// Packetize TS data into RTP packets
fn packetize_ts_to_rtp(ts_data: &[u8], seq: &mut u16, ssrc: u32, pts: ClockTime) -> Vec<Vec<u8>> {
    const RTP_PAYLOAD_SIZE: usize = 7 * 188; // 7 TS packets per RTP

    let mut packets = Vec::new();

    for chunk in ts_data.chunks(RTP_PAYLOAD_SIZE) {
        let mut rtp = Vec::with_capacity(12 + chunk.len());

        // RTP header (12 bytes)
        rtp.push(0x80); // V=2, P=0, X=0, CC=0
        rtp.push(33); // M=0, PT=33 (MP2T)
        rtp.extend(&seq.to_be_bytes()); // Sequence number
        let rtp_ts = (pts.nanos() * 90 / 1_000_000) as u32; // 90kHz
        rtp.extend(&rtp_ts.to_be_bytes()); // Timestamp
        rtp.extend(&ssrc.to_be_bytes()); // SSRC

        // Payload
        rtp.extend_from_slice(chunk);

        packets.push(rtp);
        *seq = seq.wrapping_add(1);
    }

    packets
}

/// Random number (simple implementation for example)
mod rand {
    use std::cell::Cell;
    use std::time::SystemTime;

    thread_local! {
        static SEED: Cell<u64> = Cell::new(
            SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .unwrap()
                .as_nanos() as u64
        );
    }

    pub fn random<T: FromRandom>() -> T {
        SEED.with(|seed| {
            let mut s = seed.get();
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            seed.set(s);
            T::from_random(s)
        })
    }

    pub trait FromRandom {
        fn from_random(value: u64) -> Self;
    }

    impl FromRandom for u32 {
        fn from_random(value: u64) -> Self {
            value as u32
        }
    }
}
