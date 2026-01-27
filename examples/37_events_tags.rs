//! Example: Events and Tags System
//!
//! This example demonstrates the Parallax event and tagging system:
//! - Creating and inspecting events
//! - Building tag lists with metadata
//! - Event direction (downstream vs upstream)
//! - Event serialization properties
//! - Custom events for application-specific data
//!
//! Run with: cargo run --example 37_events_tags

use parallax::clock::ClockTime;
use parallax::event::{
    CapsChangedEvent, ControlSignal, CustomEvent, Event, EventResult, FlushStopEvent, GapEvent,
    PipelineItem, QosEvent, QosType, SeekEvent, SeekFlags, SeekPosition, SeekType, SegmentEvent,
    SegmentFlags, SegmentFormat, StreamFlags, StreamStartEvent, TagList, TagMergeMode, TagValue,
    TagsEvent,
};
use parallax::format::{FormatCaps, MediaCaps, VideoFormatCaps};

fn main() {
    println!("=== Parallax Events and Tags Demo ===\n");

    // ========================================================================
    // Part 1: Tag Lists
    // ========================================================================

    println!("--- Part 1: Tag Lists ---\n");

    // Create a tag list with common media metadata
    let mut tags = TagList::new();
    tags.set_title("Sample Video");
    tags.set_artist("Parallax Framework");
    tags.set_duration(ClockTime::from_secs(120));
    tags.set_bitrate(5_000_000); // 5 Mbps
    tags.set_video_codec("AV1");
    tags.set_audio_codec("Opus");
    tags.set_container("WebM");

    println!("Created tag list:");
    println!("  Title: {:?}", tags.title());
    println!("  Artist: {:?}", tags.artist());
    println!("  Duration: {:?}", tags.duration());
    println!("  Bitrate: {:?} bps", tags.bitrate());
    println!("  Video codec: {:?}", tags.video_codec());
    println!("  Audio codec: {:?}", tags.audio_codec());
    println!("  Container: {:?}", tags.container());

    // Add custom tags
    tags.set("custom/recording-date", "2024-01-15");
    tags.set("custom/camera-model", "Sony A7IV");
    tags.set("custom/frame-count", 3600u64);

    println!("\nCustom tags:");
    println!(
        "  Recording date: {:?}",
        tags.get_string("custom/recording-date")
    );
    println!(
        "  Camera model: {:?}",
        tags.get_string("custom/camera-model")
    );
    println!("  Frame count: {:?}", tags.get_uint("custom/frame-count"));

    // ========================================================================
    // Part 2: Tag Merging
    // ========================================================================

    println!("\n--- Part 2: Tag Merging ---\n");

    let mut tags1 = TagList::new();
    tags1.set_title("Original Title");
    tags1.set_artist("Original Artist");

    let mut tags2 = TagList::new();
    tags2.set_title("Updated Title");
    tags2.set("album", "New Album");

    // Append mode: replace existing, add new
    let mut merged_append = tags1.clone();
    merged_append.merge(&tags2, TagMergeMode::Append);
    println!("Merge with Append mode:");
    println!("  Title: {:?} (replaced)", merged_append.title());
    println!("  Artist: {:?} (kept)", merged_append.artist());
    println!("  Album: {:?} (added)", merged_append.album());

    // Keep mode: keep existing, add new
    let mut merged_keep = tags1.clone();
    merged_keep.merge(&tags2, TagMergeMode::Keep);
    println!("\nMerge with Keep mode:");
    println!("  Title: {:?} (kept original)", merged_keep.title());
    println!("  Album: {:?} (added)", merged_keep.album());

    // ========================================================================
    // Part 3: Downstream Events
    // ========================================================================

    println!("\n--- Part 3: Downstream Events ---\n");

    // Stream Start - marks the beginning of a new stream
    let stream_start = Event::StreamStart(StreamStartEvent::new("video-stream-001"));
    println!("StreamStart event:");
    println!("  Name: {}", stream_start.name());
    println!("  Is downstream: {}", stream_start.is_downstream());
    println!("  Is serialized: {}", stream_start.is_serialized());

    // Stream start with flags
    let sparse_stream = Event::StreamStart(StreamStartEvent::with_flags(
        "subtitle-stream",
        StreamFlags::SPARSE,
    ));
    if let Event::StreamStart(ref ss) = sparse_stream {
        println!(
            "\nSparse stream '{}': is_sparse={}",
            ss.stream_id,
            ss.flags.contains(StreamFlags::SPARSE)
        );
    }

    // Segment - defines the playback timeline
    let segment = Event::Segment(SegmentEvent::new_time(
        ClockTime::from_secs(10),
        Some(ClockTime::from_secs(120)),
    ));
    println!("\nSegment event:");
    if let Event::Segment(ref seg) = segment {
        println!("  Format: {:?}", seg.format);
        println!("  Start: {} ns", seg.start);
        println!("  Stop: {} ns", seg.stop);
        println!("  Rate: {}", seg.rate);
    }

    // Segment with rate (for fast forward)
    let fast_segment = SegmentEvent::new_time(ClockTime::ZERO, None).with_rate(2.0);
    println!("\nFast-forward segment: rate = {}", fast_segment.rate);

    // Tags event
    let tags_event = Event::Tags(TagsEvent::new(tags.clone()));
    println!("\nTags event: {} tags", tags.len());

    // Gap event - indicates silence/black frames
    let gap = Event::Gap(GapEvent::new(
        ClockTime::from_secs(30),
        ClockTime::from_millis(500),
    ));
    println!("\nGap event:");
    if let Event::Gap(ref g) = gap {
        println!("  At: {:?}", g.timestamp);
        println!("  Duration: {:?}", g.duration);
    }

    // Caps changed - format changed mid-stream
    let caps_changed = Event::CapsChanged(CapsChangedEvent::new(MediaCaps::from_format(
        FormatCaps::VideoRaw(VideoFormatCaps::yuv420_size(1920, 1080)),
    )));
    println!("\nCapsChanged event: format changed");

    // EOS - end of stream
    let eos = Event::Eos;
    println!("\nEOS event:");
    println!("  Name: {}", eos.name());
    println!("  Is downstream: {}", eos.is_downstream());

    // ========================================================================
    // Part 4: Upstream Events
    // ========================================================================

    println!("\n--- Part 4: Upstream Events ---\n");

    // Simple seek to 30 seconds
    let seek = Event::Seek(SeekEvent::new_time(ClockTime::from_secs(30)));
    println!("Seek event (simple):");
    if let Event::Seek(ref s) = seek {
        println!("  Format: {:?}", s.format);
        println!("  Position: {} ns", s.start.position);
        println!(
            "  Flags: flush={}, key_unit={}",
            s.flags.contains(SeekFlags::FLUSH),
            s.flags.contains(SeekFlags::KEY_UNIT)
        );
    }
    println!("  Is upstream: {}", seek.is_upstream());

    // Complex seek with segment
    let segment_seek = SeekEvent {
        rate: 1.0,
        format: SegmentFormat::Time,
        flags: SeekFlags::FLUSH.union(SeekFlags::ACCURATE),
        start: SeekPosition::set(ClockTime::from_secs(10).nanos() as i64),
        stop: SeekPosition::set(ClockTime::from_secs(60).nanos() as i64),
    };
    println!("\nSegment seek (10s to 60s):");
    println!(
        "  Start: {} ns, Stop: {} ns",
        segment_seek.start.position, segment_seek.stop.position
    );

    // Byte seek
    let byte_seek = Event::Seek(SeekEvent::new_bytes(1024 * 1024)); // 1MB
    println!("\nByte seek:");
    if let Event::Seek(ref s) = byte_seek {
        println!("  Format: {:?}", s.format);
        println!("  Position: {} bytes", s.start.position);
    }

    // QoS event - quality feedback
    let qos = Event::Qos(QosEvent::new(
        QosType::Overflow,
        0.1, // 10% frames dropped
        ClockTime::from_millis(50),
        ClockTime::from_secs(45),
    ));
    println!("\nQoS event:");
    if let Event::Qos(ref q) = qos {
        println!("  Type: {:?}", q.qos_type);
        println!("  Proportion dropped: {:.1}%", q.proportion * 100.0);
        println!("  Diff: {:?}", q.diff);
    }

    // ========================================================================
    // Part 5: Bidirectional Events (Flush)
    // ========================================================================

    println!("\n--- Part 5: Flush Events ---\n");

    let flush_start = Event::FlushStart;
    let flush_stop = Event::FlushStop(FlushStopEvent::new(true));

    println!("FlushStart:");
    println!(
        "  Is serialized: {} (bypasses queues!)",
        flush_start.is_serialized()
    );
    println!("  Is bidirectional: {}", flush_start.is_bidirectional());

    println!("\nFlushStop:");
    if let Event::FlushStop(ref fs) = flush_stop {
        println!("  Reset time: {}", fs.reset_time);
    }

    // ========================================================================
    // Part 6: Custom Events
    // ========================================================================

    println!("\n--- Part 6: Custom Events ---\n");

    let custom = Event::Custom(
        CustomEvent::new("app/frame-annotation")
            .with_data("object-count", 5u64)
            .with_data("confidence", 0.95f64)
            .with_data("label", "person"),
    );

    println!("Custom event:");
    if let Event::Custom(ref c) = custom {
        println!("  Name: {}", c.name);
        println!("  Object count: {:?}", c.get("object-count"));
        println!("  Confidence: {:?}", c.get("confidence"));
        println!("  Label: {:?}", c.get("label"));
    }

    // ========================================================================
    // Part 7: Pipeline Items
    // ========================================================================

    println!("\n--- Part 7: Pipeline Items ---\n");

    // Events and buffers can be combined in PipelineItem
    let event_item: PipelineItem = Event::Eos.into();
    println!("PipelineItem from Event:");
    println!("  Is buffer: {}", event_item.is_buffer());
    println!("  Is event: {}", event_item.is_event());

    if let Some(e) = event_item.as_event() {
        println!("  Event name: {}", e.name());
    }

    // ========================================================================
    // Part 8: Control Signals
    // ========================================================================

    println!("\n--- Part 8: Control Signals ---\n");

    // Control signals bypass the data channel entirely
    let signals = [
        ControlSignal::FlushStart,
        ControlSignal::FlushStop { reset_time: true },
        ControlSignal::Pause,
        ControlSignal::Resume,
        ControlSignal::Shutdown,
    ];

    println!("Control signals (bypass data channel):");
    for signal in &signals {
        println!("  {:?}", signal);
    }

    // ========================================================================
    // Part 9: Event Results
    // ========================================================================

    println!("\n--- Part 9: Event Results ---\n");

    let handled = EventResult::Handled;
    let not_handled = EventResult::NotHandled;
    let error = EventResult::Error;

    println!("EventResult variants:");
    println!("  Handled: is_handled={}", handled.is_handled());
    println!("  NotHandled: is_handled={}", not_handled.is_handled());
    println!("  Error: is_handled={}", error.is_handled());

    // ========================================================================
    // Part 10: Tag Value Types
    // ========================================================================

    println!("\n--- Part 10: Tag Value Types ---\n");

    let values: Vec<(&str, TagValue)> = vec![
        ("string", "Hello".into()),
        ("uint", 42u64.into()),
        ("int", (-10i64).into()),
        ("double", 3.14159f64.into()),
        ("bool", true.into()),
        ("binary", vec![0xDE, 0xAD, 0xBE, 0xEF].into()),
    ];

    println!("TagValue type support:");
    for (name, value) in values {
        println!("  {}: {:?}", name, value);
    }

    println!("\n=== Demo Complete ===");
}
