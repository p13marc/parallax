//! Example: MPEG-TS muxer with STANAG 4609 metadata injection.
//!
//! This example demonstrates:
//! - Creating MPEG-TS streams with video and metadata tracks
//! - Encoding KLV metadata using MISB ST 0601 format
//! - Multiplexing video and metadata into a single transport stream
//!
//! This is a common pattern for UAV/drone video systems where sensor
//! metadata (GPS position, attitude, timestamps) needs to be synchronized
//! with video frames.
//!
//! Run with: cargo run --example 28_mpegts_stanag --features "mpeg-ts"

use parallax::clock::ClockTime;
use parallax::elements::{
    KlvEncoder, KlvTag, StanagMetadataBuilder, TsMux, TsMuxConfig, TsMuxStreamType, TsMuxTrack, Uls,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== MPEG-TS Muxer with STANAG 4609 Metadata ===\n");

    // =========================================================================
    // Part 1: Configure the TS Muxer
    // =========================================================================
    println!("1. Configuring MPEG-TS muxer with video + KLV tracks...\n");

    // Create muxer configuration with:
    // - PID 256: H.264 video (with PCR)
    // - PID 257: KLV metadata (MISB ST 0601)
    let config = TsMuxConfig::new()
        .program_number(1)
        .pmt_pid(0x1000)
        .pcr_interval_ms(40)
        .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
        .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());

    let mut mux = TsMux::new(config);

    println!("   Tracks configured:");
    println!("   - PID 256: H.264 video (PCR)");
    println!("   - PID 257: KLV/STANAG metadata\n");

    // =========================================================================
    // Part 2: Create KLV Metadata using the Builder
    // =========================================================================
    println!("2. Creating STANAG 4609 metadata (MISB ST 0601)...\n");

    // Use the convenient builder for typical UAV metadata
    let klv_data = StanagMetadataBuilder::new()
        .version(17) // ST 0601.17
        .timestamp_now()
        .mission_id("DEMO_MISSION_001")
        .platform_designation("Test Platform")
        .sensor_position(
            37.2350,   // Latitude (degrees)
            -115.8111, // Longitude (degrees)
            1500.0,    // Altitude (meters HAE)
        )
        .frame_center(37.2300, -115.8100)
        .platform_attitude(
            180.0, // Heading (degrees)
            5.0,   // Pitch (degrees)
            0.0,   // Roll (degrees)
        )
        .slant_range(2500.0)
        .build_st0601();

    println!("   Metadata created:");
    println!("   - Sensor: 37.2350, -115.8111 @ 1500m");
    println!("   - Frame center: 37.2300, -115.8100");
    println!("   - Platform: heading=180, pitch=5, roll=0");
    println!("   - KLV packet size: {} bytes\n", klv_data.len());

    // =========================================================================
    // Part 3: Create KLV using the Low-Level Encoder
    // =========================================================================
    println!("3. Creating KLV with low-level encoder...\n");

    let mut encoder = KlvEncoder::new();
    encoder
        .add_current_timestamp()
        .add_string(KlvTag::MissionId, "MANUAL_ENCODE")
        .add_sensor_latitude(40.7128)
        .add_sensor_longitude(-74.0060)
        .add_sensor_altitude(300.0)
        .add_platform_heading(90.0);

    let manual_klv = encoder.encode_with_uls(Uls::MisbSt0601);
    println!("   Manual KLV packet size: {} bytes\n", manual_klv.len());

    // =========================================================================
    // Part 4: Mux Video and Metadata
    // =========================================================================
    println!("4. Multiplexing video and metadata into MPEG-TS...\n");

    let mut total_ts_bytes = 0;

    // Write PSI tables (PAT + PMT)
    let psi = mux.write_psi();
    total_ts_bytes += psi.len();
    println!(
        "   PSI tables: {} bytes ({} TS packets)",
        psi.len(),
        psi.len() / 188
    );

    // Simulate writing video frames with synchronized metadata
    for frame_num in 0..5 {
        let pts = ClockTime::from_millis(frame_num * 40); // 25 fps

        // Simulate H.264 NAL unit (in reality this would be encoded video)
        let fake_video = create_fake_h264_frame(frame_num);

        // Write video frame
        let video_ts = mux.write_pes(256, &fake_video, Some(pts), None)?;
        total_ts_bytes += video_ts.len();

        // Create metadata for this frame
        let frame_klv = StanagMetadataBuilder::new()
            .timestamp_now()
            .sensor_position(
                37.2350 + (frame_num as f64 * 0.0001), // Slight movement
                -115.8111,
                1500.0,
            )
            .frame_center(37.2300, -115.8100)
            .build_st0601();

        // Write KLV metadata
        let klv_ts = mux.write_pes(257, &frame_klv, Some(pts), None)?;
        total_ts_bytes += klv_ts.len();

        println!(
            "   Frame {}: video={} bytes, klv={} bytes, pts={}ms",
            frame_num,
            video_ts.len(),
            klv_ts.len(),
            pts.millis()
        );
    }

    println!("\n   Total TS output: {} bytes", total_ts_bytes);

    // =========================================================================
    // Part 5: Statistics
    // =========================================================================
    println!("\n5. Muxer statistics:\n");
    let stats = mux.stats();
    println!("   TS packets written: {}", stats.packets_written);
    println!("   Total bytes: {}", stats.bytes_written);
    println!("   PES packets: {}", stats.pes_packets);
    println!("   PAT packets: {}", stats.pat_packets);
    println!("   PMT packets: {}", stats.pmt_packets);
    println!("   PCR count: {}", stats.pcr_count);

    // =========================================================================
    // Part 6: Demonstrate ULS Types
    // =========================================================================
    println!("\n6. Available Universal Label Sets (ULS):\n");
    println!(
        "   MISB ST 0601 (UAS Datalink): {:02X?}...",
        &Uls::MisbSt0601.as_bytes()[..8]
    );
    println!(
        "   MISB ST 0102 (Security):     {:02X?}...",
        &Uls::MisbSt0102.as_bytes()[..8]
    );
    println!(
        "   MISB ST 0903 (VMTI):         {:02X?}...",
        &Uls::MisbSt0903.as_bytes()[..8]
    );

    // =========================================================================
    // Part 7: KLV Tag Reference
    // =========================================================================
    println!("\n7. Common MISB ST 0601 tags:\n");
    let tags = [
        (KlvTag::UnixTimeStamp, "Unix Timestamp"),
        (KlvTag::MissionId, "Mission ID"),
        (KlvTag::SensorLatitude, "Sensor Latitude"),
        (KlvTag::SensorLongitude, "Sensor Longitude"),
        (KlvTag::SensorTrueAltitude, "Sensor Altitude"),
        (KlvTag::FrameCenterLatitude, "Frame Center Lat"),
        (KlvTag::FrameCenterLongitude, "Frame Center Lon"),
        (KlvTag::PlatformHeadingAngle, "Platform Heading"),
        (KlvTag::SlantRange, "Slant Range"),
    ];

    for (tag, name) in tags {
        println!("   Tag {:2}: {}", u8::from(tag), name);
    }

    println!("\n=== Example complete ===");
    Ok(())
}

/// Create a fake H.264 frame for demonstration.
///
/// In a real application, this would be actual encoded video data.
fn create_fake_h264_frame(frame_num: u64) -> Vec<u8> {
    let mut data = Vec::new();

    // NAL start code
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);

    if frame_num == 0 {
        // SPS NAL unit (type 7)
        data.push(0x67);
        data.extend_from_slice(&[0x42, 0x00, 0x1E, 0x8D, 0x68, 0x10]); // Fake SPS

        // PPS NAL unit (type 8)
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        data.push(0x68);
        data.extend_from_slice(&[0xCE, 0x3C, 0x80]); // Fake PPS
    }

    // IDR or P-frame NAL unit
    data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
    if frame_num % 30 == 0 {
        // IDR frame (type 5)
        data.push(0x65);
    } else {
        // P-frame (type 1)
        data.push(0x41);
    }

    // Fake frame data (would be actual encoded data in reality)
    data.extend_from_slice(&vec![0xAB; 500 + (frame_num as usize % 200)]);

    data
}
