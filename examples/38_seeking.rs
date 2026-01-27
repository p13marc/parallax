//! Example: Seeking API
//!
//! This example demonstrates how seeking works in Parallax:
//! - Creating seek events for different scenarios
//! - Seek flags and their meanings
//! - Segment events that follow seeks
//! - The seek/flush/segment flow
//!
//! Note: This example shows the seeking API. Full seeking support
//! in FileSrc and other elements is implementation-dependent.
//!
//! Run with: cargo run --example 38_seeking

use parallax::clock::ClockTime;
use parallax::event::{
    Event, FlushStopEvent, SeekEvent, SeekFlags, SeekPosition, SeekType, SegmentEvent,
    SegmentFlags, SegmentFormat,
};

fn main() {
    println!("=== Parallax Seeking API Demo ===\n");

    // ========================================================================
    // Part 1: Simple Time-Based Seek
    // ========================================================================

    println!("--- Part 1: Simple Time-Based Seek ---\n");

    // The most common case: seek to a specific timestamp
    let seek = SeekEvent::new_time(ClockTime::from_secs(30));

    println!("Simple seek to 30 seconds:");
    println!("  Rate: {}", seek.rate);
    println!("  Format: {:?}", seek.format);
    println!(
        "  Start position: {} ns ({} seconds)",
        seek.start.position,
        seek.start.position / 1_000_000_000
    );
    println!("  Start type: {:?}", seek.start.seek_type);
    println!("  Flags:");
    println!(
        "    FLUSH: {} (discard buffered data)",
        seek.flags.contains(SeekFlags::FLUSH)
    );
    println!(
        "    KEY_UNIT: {} (seek to keyframe)",
        seek.flags.contains(SeekFlags::KEY_UNIT)
    );

    // ========================================================================
    // Part 2: Byte-Based Seek
    // ========================================================================

    println!("\n--- Part 2: Byte-Based Seek ---\n");

    // Useful for raw files or when you know the byte offset
    let byte_seek = SeekEvent::new_bytes(10 * 1024 * 1024); // 10 MB

    println!("Seek to 10 MB offset:");
    println!("  Format: {:?}", byte_seek.format);
    println!("  Position: {} bytes", byte_seek.start.position);

    // ========================================================================
    // Part 3: Accurate vs Keyframe Seek
    // ========================================================================

    println!("\n--- Part 3: Accurate vs Keyframe Seek ---\n");

    // Keyframe seek (fast, may not land exactly on requested position)
    let keyframe_seek = SeekEvent::new_time(ClockTime::from_secs(45))
        .with_flags(SeekFlags::FLUSH.union(SeekFlags::KEY_UNIT));

    println!("Keyframe seek (fast):");
    println!(
        "  KEY_UNIT: {}",
        keyframe_seek.flags.contains(SeekFlags::KEY_UNIT)
    );
    println!(
        "  ACCURATE: {}",
        keyframe_seek.flags.contains(SeekFlags::ACCURATE)
    );
    println!("  -> Seeks to nearest keyframe, may decode less frames");

    // Accurate seek (slower, lands exactly on requested position)
    let accurate_seek = SeekEvent::new_time(ClockTime::from_secs(45))
        .with_flags(SeekFlags::FLUSH.union(SeekFlags::ACCURATE));

    println!("\nAccurate seek (precise):");
    println!(
        "  KEY_UNIT: {}",
        accurate_seek.flags.contains(SeekFlags::KEY_UNIT)
    );
    println!(
        "  ACCURATE: {}",
        accurate_seek.flags.contains(SeekFlags::ACCURATE)
    );
    println!("  -> Seeks to keyframe, then decodes forward to exact position");

    // ========================================================================
    // Part 4: Segment Seek (Range)
    // ========================================================================

    println!("\n--- Part 4: Segment Seek ---\n");

    // Seek to a specific segment (play from 10s to 30s)
    let segment_seek = SeekEvent {
        rate: 1.0,
        format: SegmentFormat::Time,
        flags: SeekFlags::FLUSH.union(SeekFlags::KEY_UNIT),
        start: SeekPosition::set(ClockTime::from_secs(10).nanos() as i64),
        stop: SeekPosition::set(ClockTime::from_secs(30).nanos() as i64),
    };

    println!("Segment seek (play 10s to 30s):");
    println!(
        "  Start: {} seconds",
        segment_seek.start.position / 1_000_000_000
    );
    println!(
        "  Stop: {} seconds",
        segment_seek.stop.position / 1_000_000_000
    );

    // ========================================================================
    // Part 5: Relative Seeks
    // ========================================================================

    println!("\n--- Part 5: Relative Seeks ---\n");

    // Seek relative to current position (skip forward 10 seconds)
    let relative_seek = SeekEvent {
        rate: 1.0,
        format: SegmentFormat::Time,
        flags: SeekFlags::FLUSH.union(SeekFlags::KEY_UNIT),
        start: SeekPosition::current(ClockTime::from_secs(10).nanos() as i64),
        stop: SeekPosition::none(),
    };

    println!("Relative seek (+10 seconds from current):");
    println!("  Start type: {:?}", relative_seek.start.seek_type);
    println!("  Offset: {} ns", relative_seek.start.position);

    // Seek relative to end (last 30 seconds)
    let from_end_seek = SeekEvent {
        rate: 1.0,
        format: SegmentFormat::Time,
        flags: SeekFlags::FLUSH.union(SeekFlags::KEY_UNIT),
        start: SeekPosition::end(-(ClockTime::from_secs(30).nanos() as i64)),
        stop: SeekPosition::none(),
    };

    println!("\nSeek to last 30 seconds:");
    println!("  Start type: {:?}", from_end_seek.start.seek_type);
    println!(
        "  Offset: {} ns (negative = from end)",
        from_end_seek.start.position
    );

    // ========================================================================
    // Part 6: Fast Forward / Rewind
    // ========================================================================

    println!("\n--- Part 6: Playback Rate ---\n");

    // 2x fast forward
    let fast_forward = SeekEvent::new_time(ClockTime::ZERO).with_rate(2.0);

    println!("2x fast forward:");
    println!("  Rate: {}", fast_forward.rate);

    // Reverse playback
    let reverse = SeekEvent::new_time(ClockTime::from_secs(60)).with_rate(-1.0);

    println!("\nReverse playback:");
    println!("  Rate: {} (negative = reverse)", reverse.rate);

    // ========================================================================
    // Part 7: The Seek Flow
    // ========================================================================

    println!("\n--- Part 7: The Seek Flow ---\n");

    println!("When an element handles a seek, the typical flow is:");
    println!();
    println!("  1. Application sends SeekEvent upstream");
    println!("     -> Event flows: Sink -> Transform -> Source");
    println!();
    println!("  2. Source handles the seek:");
    println!("     a. Sends FlushStart downstream");
    let flush_start = Event::FlushStart;
    println!("        {:?}", flush_start);
    println!();
    println!("     b. Performs the actual seek (file.seek(), etc.)");
    println!();
    println!("     c. Sends FlushStop downstream");
    let flush_stop = Event::FlushStop(FlushStopEvent::new(true));
    println!("        {:?}", flush_stop);
    println!();
    println!("     d. Sends new Segment event downstream");
    let new_segment = Event::Segment(
        SegmentEvent::new_time(ClockTime::from_secs(30), None).with_flags(SegmentFlags::RESET),
    );
    if let Event::Segment(ref seg) = new_segment {
        println!(
            "        Segment {{ start: {}s, flags: RESET }}",
            seg.start / 1_000_000_000
        );
    }
    println!();
    println!("  3. Elements resume normal processing at new position");

    // ========================================================================
    // Part 8: Non-Flushing Seek
    // ========================================================================

    println!("\n--- Part 8: Non-Flushing Seek ---\n");

    // Sometimes you want to seek without flushing (e.g., for seamless looping)
    let non_flush_seek = SeekEvent {
        rate: 1.0,
        format: SegmentFormat::Time,
        flags: SeekFlags::NONE, // No flush!
        start: SeekPosition::set(0),
        stop: SeekPosition::none(),
    };

    println!("Non-flushing seek (for seamless looping):");
    println!(
        "  FLUSH: {}",
        non_flush_seek.flags.contains(SeekFlags::FLUSH)
    );
    println!("  -> Queued data continues to play while seeking");
    println!("  -> Useful for gapless playback / looping");

    // ========================================================================
    // Part 9: Snap Flags
    // ========================================================================

    println!("\n--- Part 9: Snap Flags ---\n");

    // Snap to position before target
    let snap_before = SeekFlags::FLUSH.union(SeekFlags::SNAP_BEFORE);
    println!("SNAP_BEFORE: Seek to position <= target");
    println!(
        "  Flags: contains SNAP_BEFORE = {}",
        snap_before.contains(SeekFlags::SNAP_BEFORE)
    );

    // Snap to position after target
    let snap_after = SeekFlags::FLUSH.union(SeekFlags::SNAP_AFTER);
    println!("\nSNAP_AFTER: Seek to position >= target");
    println!(
        "  Flags: contains SNAP_AFTER = {}",
        snap_after.contains(SeekFlags::SNAP_AFTER)
    );

    // ========================================================================
    // Part 10: Putting It Together
    // ========================================================================

    println!("\n--- Part 10: Complete Example ---\n");

    // A realistic seek: accurate seek to 1:30, play until 2:00
    let realistic_seek = SeekEvent {
        rate: 1.0,
        format: SegmentFormat::Time,
        flags: SeekFlags::FLUSH.union(SeekFlags::ACCURATE),
        start: SeekPosition {
            seek_type: SeekType::Set,
            position: ClockTime::from_secs(90).nanos() as i64, // 1:30
        },
        stop: SeekPosition {
            seek_type: SeekType::Set,
            position: ClockTime::from_secs(120).nanos() as i64, // 2:00
        },
    };

    println!("Realistic seek: Play from 1:30 to 2:00 (accurate)");
    println!("  Format: {:?}", realistic_seek.format);
    println!("  Rate: {}", realistic_seek.rate);
    println!(
        "  Start: {:?} @ {} ({} seconds)",
        realistic_seek.start.seek_type,
        realistic_seek.start.position,
        realistic_seek.start.position / 1_000_000_000
    );
    println!(
        "  Stop: {:?} @ {} ({} seconds)",
        realistic_seek.stop.seek_type,
        realistic_seek.stop.position,
        realistic_seek.stop.position / 1_000_000_000
    );
    println!("  Flags:");
    println!(
        "    FLUSH: {}",
        realistic_seek.flags.contains(SeekFlags::FLUSH)
    );
    println!(
        "    ACCURATE: {}",
        realistic_seek.flags.contains(SeekFlags::ACCURATE)
    );

    println!("\n=== Demo Complete ===");
}
