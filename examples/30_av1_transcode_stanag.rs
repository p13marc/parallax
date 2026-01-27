//! Example: Complete video transcode pipeline with STANAG metadata using AV1.
//!
//! This example demonstrates a real-world UAV video pipeline using pure Rust codecs:
//!
//! ```text
//! Video Source → Decode (AV1) → Scale → Encode (AV1) → Mux (+ KLV) → RTP/UDP
//! ```
//!
//! Uses pure Rust codecs (no C++ toolchain required):
//! - rav1e for AV1 encoding
//! - dav1d for AV1 decoding (requires libdav1d system library)
//!
//! Features:
//! - Decodes AV1 video to YUV420
//! - Scales video to target resolution
//! - Re-encodes with new bitrate/quality
//! - Injects STANAG 4609 / MISB ST 0601 metadata from external source
//! - Multiplexes video + metadata into MPEG-TS
//! - Sends over RTP/UDP
//!
//! Run with: cargo run --example 30_av1_transcode_stanag --features "av1-encode,mpeg-ts"

use std::net::UdpSocket;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

use parallax::clock::ClockTime;
use parallax::elements::{
    // STANAG/KLV
    StanagMetadataBuilder,
    // MPEG-TS
    TsMux,
    TsMuxConfig,
    TsMuxStreamType,
    TsMuxTrack,
    // Video processing
    VideoScale,
};

#[cfg(feature = "av1-encode")]
use parallax::elements::{Rav1eConfig, Rav1eEncoder};

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
    /// Target video width
    pub target_width: u32,
    /// Target video height
    pub target_height: u32,
    /// AV1 encoder speed (0-10, higher = faster)
    pub encoder_speed: usize,
    /// AV1 quantizer (0-255, lower = higher quality)
    pub quantizer: usize,
    /// Target frame rate
    pub target_fps: u64,
    /// Video PID in output TS
    pub video_pid: u16,
    /// KLV metadata PID in output TS
    pub klv_pid: u16,
    /// UDP destination
    pub udp_dest: String,
}

impl Default for TranscodeConfig {
    fn default() -> Self {
        Self {
            target_width: 1280,
            target_height: 720,
            encoder_speed: 6,
            quantizer: 100,
            target_fps: 25,
            video_pid: 256,
            klv_pid: 257,
            udp_dest: "127.0.0.1:5004".to_string(),
        }
    }
}

// ============================================================================
// Main Pipeline
// ============================================================================

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== AV1 Transcode Pipeline with STANAG Metadata ===\n");
    println!("Using pure Rust codecs (no C++ toolchain required)\n");

    let config = TranscodeConfig::default();

    println!("Configuration:");
    println!(
        "  Target: {}x{} @ {} fps",
        config.target_width, config.target_height, config.target_fps
    );
    println!(
        "  AV1: speed={}, quantizer={}",
        config.encoder_speed, config.quantizer
    );
    println!(
        "  Video PID: {}, KLV PID: {}",
        config.video_pid, config.klv_pid
    );
    println!("  UDP Output: {}", config.udp_dest);
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
    // Step 2: Initialize AV1 encoder (rav1e - pure Rust)
    // =========================================================================
    #[cfg(feature = "av1-encode")]
    let mut encoder = {
        println!("2. Initializing AV1 encoder (rav1e - pure Rust)...");
        let rav1e_config = Rav1eConfig::default()
            .width(config.target_width as usize)
            .height(config.target_height as usize)
            .speed(config.encoder_speed)
            .quantizer(config.quantizer)
            .framerate(config.target_fps);

        match Rav1eEncoder::new(rav1e_config) {
            Ok(enc) => {
                println!(
                    "   Encoder ready: {}x{}",
                    config.target_width, config.target_height
                );
                Some(enc)
            }
            Err(e) => {
                println!("   Warning: Failed to create encoder: {}", e);
                None
            }
        }
    };

    #[cfg(not(feature = "av1-encode"))]
    let encoder: Option<()> = {
        println!("2. AV1 encoder not available (enable 'av1-encode' feature)");
        None
    };
    println!();

    // =========================================================================
    // Step 3: Initialize video scaler
    // =========================================================================
    println!("3. Initializing video scaler...");
    let mut scaler = VideoScale::new(1920, 1080, config.target_width, config.target_height);
    println!(
        "   Scale: 1920x1080 → {}x{}\n",
        config.target_width, config.target_height
    );

    // =========================================================================
    // Step 4: Initialize MPEG-TS muxer with AV1 video track
    // =========================================================================
    println!("4. Initializing MPEG-TS muxer...");
    let mux_config = TsMuxConfig::new()
        .program_number(1)
        .pmt_pid(0x1000)
        .pcr_interval_ms(40)
        .add_track(TsMuxTrack::new(config.video_pid, TsMuxStreamType::Av1).video())
        .add_track(TsMuxTrack::new(config.klv_pid, TsMuxStreamType::Klv).private_data());
    let mut mux = TsMux::new(mux_config);
    println!(
        "   Tracks: AV1 Video (PID {}), KLV (PID {})\n",
        config.video_pid, config.klv_pid
    );

    // =========================================================================
    // Step 5: Initialize UDP socket
    // =========================================================================
    println!("5. Initializing UDP socket...");
    let socket = UdpSocket::bind("0.0.0.0:0")?;
    socket.set_nonblocking(true)?;
    println!("   Bound to: {}", socket.local_addr()?);
    println!("   Destination: {}\n", config.udp_dest);

    // =========================================================================
    // Step 6: Process frames
    // =========================================================================
    println!("6. Processing pipeline (5 frames)...\n");

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

        // Get latest metadata from external source
        while let Ok(meta) = metadata_rx.try_recv() {
            latest_metadata = Some(meta);
        }

        // Simulate incoming YUV420 frame (in production: from decoder)
        let src_yuv = create_simulated_yuv420(1920, 1080);

        // Scale to target resolution
        let scaled_yuv = scaler.scale_yuv420(&src_yuv)?;

        // Encode to AV1 (or use fake data if encoder not available)
        #[cfg(feature = "av1-encode")]
        let encoded_frame = if let Some(ref mut enc) = encoder {
            // In a real pipeline, we'd feed the scaled YUV to the encoder
            // For this demo, we'll use simulated OBU data
            create_fake_av1_obu(frame_num)
        } else {
            create_fake_av1_obu(frame_num)
        };

        #[cfg(not(feature = "av1-encode"))]
        let encoded_frame = create_fake_av1_obu(frame_num);

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

        // Packetize to RTP
        let rtp_packets = packetize_ts_to_rtp(&output_ts, &mut rtp_sequence, ssrc, pts);

        // Send RTP packets
        for rtp in &rtp_packets {
            let _ = socket.send_to(rtp, &config.udp_dest);
        }

        let meta_info = latest_metadata
            .as_ref()
            .map(|m| format!("lat={:.4}, lon={:.4}", m.sensor_lat, m.sensor_lon))
            .unwrap_or_else(|| "waiting...".to_string());

        println!(
            "   Frame {}: yuv={}KB, scaled={}KB, av1={} bytes, klv={} bytes, rtp={} pkts",
            frame_num,
            src_yuv.len() / 1024,
            scaled_yuv.len() / 1024,
            encoded_frame.len(),
            klv_ts.len(),
            rtp_packets.len()
        );
        println!("            meta=[{}]", meta_info);
    }

    // =========================================================================
    // Step 7: Statistics
    // =========================================================================
    println!("\n7. Pipeline statistics:\n");
    let mux_stats = mux.stats();
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
    println!("\n=== Pipeline Architecture (Pure Rust) ===\n");
    println!("┌─────────────┐     ┌─────────────┐     ┌─────────────┐");
    println!("│ Video Src   │────▶│  AV1 Dec    │────▶│ Video Scale │");
    println!("│             │     │  (dav1d)    │     │ 1080p→720p  │");
    println!("└─────────────┘     └─────────────┘     └──────┬──────┘");
    println!("                                               │");
    println!("                                               ▼");
    println!("┌─────────────┐     ┌─────────────┐     ┌─────────────┐");
    println!("│ Sensor Data │     │  TS Mux     │◀────│  AV1 Enc    │");
    println!("│ (GPS/IMU)   │────▶│  (+KLV PID) │     │  (rav1e)    │");
    println!("└─────────────┘     └──────┬──────┘     └─────────────┘");
    println!("                           │");
    println!("                           ▼");
    println!("                    ┌─────────────┐");
    println!("                    │  RTP/UDP    │");
    println!("                    │  Output     │");
    println!("                    └─────────────┘");

    println!("\n=== Advantages of AV1 ===\n");
    println!("1. Pure Rust encoder (rav1e) - no C++ toolchain needed");
    println!("2. Better compression than H.264 at same quality");
    println!("3. Royalty-free codec");
    println!("4. Modern features: film grain, HDR, etc.");

    println!("\n=== Example Complete ===");
    Ok(())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Create a simulated YUV420 frame
fn create_simulated_yuv420(width: u32, height: u32) -> Vec<u8> {
    let y_size = (width * height) as usize;
    let uv_size = ((width / 2) * (height / 2)) as usize;

    let mut yuv = Vec::with_capacity(y_size + 2 * uv_size);

    // Y plane (gradient pattern)
    for y in 0..height {
        for x in 0..width {
            let value = ((x + y) % 256) as u8;
            yuv.push(value);
        }
    }

    // U plane (neutral blue)
    yuv.extend(vec![128u8; uv_size]);

    // V plane (neutral red)
    yuv.extend(vec![128u8; uv_size]);

    yuv
}

/// Create a fake AV1 OBU for demonstration
fn create_fake_av1_obu(frame_num: u64) -> Vec<u8> {
    let mut obu = Vec::new();

    // OBU header (simplified)
    // In real AV1: OBU type, extension flag, has_size_field, etc.
    if frame_num == 0 {
        // Sequence header OBU (type 1)
        obu.push(0x0A); // OBU type 1, has_size=1
        obu.push(0x04); // Size
        obu.extend(&[0x00, 0x00, 0x00, 0x00]); // Fake sequence header data
    }

    // Frame OBU (type 6 for key frame, type 3 for inter)
    if frame_num % 30 == 0 {
        obu.push(0x32); // OBU type 6 (key frame), has_size=1
    } else {
        obu.push(0x1A); // OBU type 3 (inter frame), has_size=1
    }

    // Size (variable length, simplified here)
    let frame_size = 200 + (frame_num as usize % 100);
    obu.push(frame_size as u8);

    // Fake frame data
    obu.extend(vec![0xAB; frame_size]);

    obu
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
        rtp.extend(&seq.to_be_bytes());
        let rtp_ts = (pts.nanos() * 90 / 1_000_000) as u32;
        rtp.extend(&rtp_ts.to_be_bytes());
        rtp.extend(&ssrc.to_be_bytes());

        // Payload
        rtp.extend_from_slice(chunk);

        packets.push(rtp);
        *seq = seq.wrapping_add(1);
    }

    packets
}

/// Simple random number generator
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
