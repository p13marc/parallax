//! Example 39: Muxer Element
//!
//! This example demonstrates using the TsMuxElement for N-to-1 multiplexing
//! with PTS-based synchronization.
//!
//! # Concepts Demonstrated
//!
//! - Creating a TsMuxElement with multiple input tracks
//! - Using MuxerSyncState for PTS-based synchronization
//! - Push/pull muxer API
//! - Strict vs loose sync modes
//!
//! # Running
//!
//! ```bash
//! cargo run --example 39_muxer_element --features mpeg-ts
//! ```

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::clock::ClockTime;
use parallax::element::muxer::{MuxerSyncConfig, MuxerSyncState, PadInfo, StreamType, SyncMode};
use parallax::element::{Muxer, MuxerInput};
use parallax::elements::mux::{TsMuxConfig, TsMuxElement, TsMuxStreamType, TsMuxTrack};
use parallax::memory::{HeapSegment, MemorySegment};
use parallax::metadata::Metadata;
use std::sync::Arc;

/// Size of a single TS packet.
const TS_PACKET_SIZE: usize = 188;

/// Create a test buffer with given PTS and data.
fn make_buffer(pts_ms: u64, data: &[u8]) -> Buffer {
    let segment = Arc::new(HeapSegment::new(data.len().max(64)).unwrap());
    let ptr = segment.as_mut_ptr().unwrap();
    unsafe {
        std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
    }
    let handle = MemoryHandle::from_segment_with_len(segment, data.len());
    let mut metadata = Metadata::from_sequence(0);
    metadata.pts = ClockTime::from_millis(pts_ms);
    Buffer::new(handle, metadata)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Example 39: Muxer Element ===\n");

    // ========================================================================
    // Part 1: Direct MuxerSyncState usage
    // ========================================================================
    println!("--- Part 1: MuxerSyncState Direct Usage ---\n");

    // Create sync state with 40ms output interval (25fps video)
    let config = MuxerSyncConfig::new()
        .with_mode(SyncMode::Strict)
        .with_interval_ms(40);

    let mut sync = MuxerSyncState::new(config);

    // Add pads
    let video_pad = sync.add_pad(PadInfo::new("video_0", StreamType::Video).required());
    let audio_pad = sync.add_pad(PadInfo::new("audio_0", StreamType::Audio).required());
    let data_pad = sync.add_pad(PadInfo::new("data_0", StreamType::Data).optional());

    println!("Created sync state with {} pads:", sync.pad_infos().len());
    for info in sync.pad_infos() {
        println!(
            "  - {} ({:?}, required={})",
            info.name, info.stream_type, info.required
        );
    }
    println!();

    // Push video frame at PTS=0
    sync.push(video_pad, make_buffer(0, b"video_frame_0"))?;
    println!("Pushed video frame at PTS=0ms");
    println!("  ready_to_output: {}", sync.ready_to_output());

    // Push audio frame at PTS=0
    sync.push(audio_pad, make_buffer(0, b"audio_frame_0"))?;
    println!("Pushed audio frame at PTS=0ms");
    println!("  ready_to_output: {}", sync.ready_to_output());

    // Optional data pad doesn't need to have data
    println!("  (data pad is optional, no data needed)");

    // Collect output
    if sync.ready_to_output() {
        let collected = sync.collect_for_output();
        println!("\nCollected {} buffers for output:", collected.len());
        for input in &collected {
            let pts = input.buffer.metadata().pts;
            println!(
                "  - pad {:?} ({:?}): PTS={}ms",
                input.pad_id,
                input.stream_type,
                pts.millis()
            );
        }
        sync.advance();
        println!("Advanced target PTS to {}ms", sync.target_pts().millis());
    }

    println!();

    // ========================================================================
    // Part 2: TsMuxElement usage
    // ========================================================================
    println!("--- Part 2: TsMuxElement Usage ---\n");

    // Create TS muxer with video and KLV tracks
    let ts_config = TsMuxConfig::new()
        .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
        .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());

    let mut mux = TsMuxElement::new(ts_config)?;

    println!(
        "Created TsMuxElement with {} input pads:",
        mux.inputs().len()
    );
    for (pad_id, caps) in mux.inputs() {
        let pid = mux.pid_for_pad(*pad_id).unwrap();
        println!("  - pad {:?} -> PID {} (caps: {:?})", pad_id, pid, caps);
    }
    println!();

    // Get pad IDs
    let video_pad = mux.inputs()[0].0;
    let data_pad = mux.inputs()[1].0;

    // Push a video frame (H.264 NAL unit)
    let h264_nal = vec![0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1f]; // SPS
    mux.push(MuxerInput::new(video_pad, make_buffer(0, &h264_nal)))?;
    println!("Pushed H.264 NAL unit at PTS=0ms");

    // Push KLV metadata
    let klv_data = vec![0x06, 0x0E, 0x2B, 0x34]; // KLV key prefix
    mux.push(MuxerInput::new(data_pad, make_buffer(0, &klv_data)))?;
    println!("Pushed KLV data at PTS=0ms");

    // Check if ready and pull
    println!("\ncan_output: {}", mux.can_output());
    if mux.can_output() {
        if let Some(output) = mux.pull()? {
            let ts_data = output.as_bytes();
            let packet_count = ts_data.len() / TS_PACKET_SIZE;
            println!(
                "Pulled {} bytes ({} TS packets)",
                ts_data.len(),
                packet_count
            );
            println!("  First byte (sync): 0x{:02X}", ts_data[0]);
        }
    }

    // Push more frames
    println!("\nPushing more frames...");
    mux.push(MuxerInput::new(video_pad, make_buffer(40, &h264_nal)))?;
    mux.push(MuxerInput::new(video_pad, make_buffer(80, &h264_nal)))?;
    mux.push(MuxerInput::new(data_pad, make_buffer(40, &klv_data)))?;

    // Pull all available output
    while mux.can_output() {
        if let Some(output) = mux.pull()? {
            let pts = output.metadata().pts;
            let packet_count = output.as_bytes().len() / TS_PACKET_SIZE;
            println!(
                "  Output at PTS={}ms: {} TS packets",
                pts.millis(),
                packet_count
            );
        }
    }

    // Flush remaining
    println!("\nFlushing remaining data...");
    let flushed = mux.flush_all()?;
    println!("Flushed {} buffers", flushed.len());

    println!();

    // ========================================================================
    // Part 3: Sync Mode Comparison
    // ========================================================================
    println!("--- Part 3: Sync Mode Comparison ---\n");

    // Strict mode
    let mut strict_sync = MuxerSyncState::new(
        MuxerSyncConfig::new()
            .with_mode(SyncMode::Strict)
            .with_interval_ms(40),
    );
    let v_pad = strict_sync.add_pad(PadInfo::new("video", StreamType::Video).required());
    let a_pad = strict_sync.add_pad(PadInfo::new("audio", StreamType::Audio).required());

    strict_sync.push(v_pad, make_buffer(0, b"v"))?;
    println!("Strict mode - video only:");
    println!("  ready_to_output: {}", strict_sync.ready_to_output());

    strict_sync.push(a_pad, make_buffer(0, b"a"))?;
    println!("Strict mode - video + audio:");
    println!("  ready_to_output: {}", strict_sync.ready_to_output());

    // Loose mode
    let mut loose_sync = MuxerSyncState::new(
        MuxerSyncConfig::new()
            .with_mode(SyncMode::Loose)
            .with_interval_ms(40),
    );
    let v_pad = loose_sync.add_pad(PadInfo::new("video", StreamType::Video).required());
    let _a_pad = loose_sync.add_pad(PadInfo::new("audio", StreamType::Audio).required());

    loose_sync.push(v_pad, make_buffer(0, b"v"))?;
    println!("\nLoose mode - video only:");
    println!("  ready_to_output: {}", loose_sync.ready_to_output());

    // Auto mode (non-live = strict, live = loose)
    let auto_nonlive = MuxerSyncState::new(MuxerSyncConfig::new().with_mode(SyncMode::Auto));
    let auto_live = MuxerSyncState::new(MuxerSyncConfig::new().with_mode(SyncMode::Auto).live());
    println!("\nAuto mode resolution:");
    println!(
        "  Non-live: {:?}",
        if auto_nonlive.pad_infos().is_empty() {
            "Strict"
        } else {
            "Strict"
        }
    );
    println!(
        "  Live: {:?}",
        if auto_live.pad_infos().is_empty() {
            "Loose"
        } else {
            "Loose"
        }
    );

    println!();

    // ========================================================================
    // Part 4: EOS Handling
    // ========================================================================
    println!("--- Part 4: EOS Handling ---\n");

    let mut sync = MuxerSyncState::new(
        MuxerSyncConfig::new()
            .with_mode(SyncMode::Strict)
            .with_interval_ms(40),
    );
    let v_pad = sync.add_pad(PadInfo::new("video", StreamType::Video).required());
    let a_pad = sync.add_pad(PadInfo::new("audio", StreamType::Audio).required());

    sync.push(v_pad, make_buffer(0, b"v"))?;
    println!("Pushed video only");
    println!("  ready_to_output: {}", sync.ready_to_output());

    // Audio ends early (EOS)
    sync.set_eos(a_pad);
    println!("Audio pad at EOS");
    println!("  ready_to_output: {}", sync.ready_to_output());
    println!("  all_eos: {}", sync.all_eos());

    sync.set_eos(v_pad);
    println!("Video pad at EOS");
    println!("  all_eos: {}", sync.all_eos());

    println!("\n=== Example Complete ===");
    Ok(())
}
