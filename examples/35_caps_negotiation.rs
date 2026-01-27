//! Example: Caps Negotiation System
//!
//! This example demonstrates Parallax's constraint-based caps negotiation:
//! - CapsValue<T>: Fixed, Range, List, or Any constraints
//! - VideoFormatCaps/AudioFormatCaps: Format constraints for negotiation
//! - Intersection: Finding common ground between constraints
//! - Fixation: Choosing concrete values from constraints
//! - Convenience constructors for common formats
//!
//! Run with: cargo run --example 35_caps_negotiation

use parallax::format::{
    AudioFormatCaps, CapsValue, Framerate, MediaCaps, PixelFormat, VideoFormat, VideoFormatCaps,
};

fn main() {
    println!("=== Parallax Caps Negotiation Demo ===\n");

    // ========================================================================
    // Part 1: CapsValue<T> - the building block of constraints
    // ========================================================================

    println!("--- Part 1: CapsValue<T> Constraints ---\n");

    // Fixed: exactly one value
    let fixed: CapsValue<u32> = CapsValue::Fixed(1920);
    println!("Fixed(1920):");
    println!("  accepts 1920: {}", fixed.accepts(&1920));
    println!("  accepts 1080: {}", fixed.accepts(&1080));
    println!("  fixate: {:?}", fixed.fixate());

    // Range: any value within bounds (inclusive)
    let range: CapsValue<u32> = CapsValue::Range {
        min: 720,
        max: 1920,
    };
    println!("\nRange {{ 720..=1920 }}:");
    println!("  accepts 480: {}", range.accepts(&480));
    println!("  accepts 1080: {}", range.accepts(&1080));
    println!("  accepts 4096: {}", range.accepts(&4096));
    println!("  fixate: {:?}", range.fixate()); // prefers min

    // List: any value from the list (ordered by preference)
    let list: CapsValue<u32> = CapsValue::List(vec![1920, 1280, 720]);
    println!("\nList [1920, 1280, 720]:");
    println!("  accepts 1920: {}", list.accepts(&1920));
    println!("  accepts 1080: {}", list.accepts(&1080));
    println!("  accepts 720: {}", list.accepts(&720));
    println!("  fixate: {:?}", list.fixate()); // prefers first

    // Any: accepts anything (unconstrained)
    let any: CapsValue<u32> = CapsValue::Any;
    println!("\nAny:");
    println!("  accepts 0: {}", any.accepts(&0));
    println!("  accepts MAX: {}", any.accepts(&u32::MAX));
    println!("  fixate: {:?}", any.fixate()); // None - needs default
    println!(
        "  fixate_with_default(1080): {}",
        any.fixate_with_default(1080)
    );

    // ========================================================================
    // Part 2: Intersection - finding common ground
    // ========================================================================

    println!("\n--- Part 2: Constraint Intersection ---\n");

    // Fixed vs Range
    let fixed_1080: CapsValue<u32> = CapsValue::Fixed(1080);
    let range_720_1920: CapsValue<u32> = CapsValue::Range {
        min: 720,
        max: 1920,
    };
    println!("Fixed(1080) intersect Range(720..=1920):");
    println!("  result: {:?}", fixed_1080.intersect(&range_720_1920));

    // Fixed outside range
    let fixed_480: CapsValue<u32> = CapsValue::Fixed(480);
    println!("\nFixed(480) intersect Range(720..=1920):");
    println!("  result: {:?}", fixed_480.intersect(&range_720_1920));

    // Range vs Range (overlapping)
    let range_1080_4k: CapsValue<u32> = CapsValue::Range {
        min: 1080,
        max: 4096,
    };
    println!("\nRange(720..=1920) intersect Range(1080..=4096):");
    println!("  result: {:?}", range_720_1920.intersect(&range_1080_4k));

    // List vs Range
    let list_sizes: CapsValue<u32> = CapsValue::List(vec![4096, 1920, 1080, 720, 480]);
    println!("\nList [4096, 1920, 1080, 720, 480] intersect Range(720..=1920):");
    println!("  result: {:?}", list_sizes.intersect(&range_720_1920));

    // List vs List
    let list_a: CapsValue<u32> = CapsValue::List(vec![1920, 1280, 720]);
    let list_b: CapsValue<u32> = CapsValue::List(vec![1080, 720, 480]);
    println!("\nList [1920, 1280, 720] intersect List [1080, 720, 480]:");
    println!("  result: {:?}", list_a.intersect(&list_b));

    // Any intersects with anything
    let any_val: CapsValue<u32> = CapsValue::Any;
    println!("\nAny intersect Fixed(1920):");
    println!("  result: {:?}", any_val.intersect(&fixed));

    // ========================================================================
    // Part 3: VideoFormatCaps - video format constraints
    // ========================================================================

    println!("\n--- Part 3: VideoFormatCaps ---\n");

    // Element A: Produces 1080p or 720p I420 at 30fps
    let producer_caps = VideoFormatCaps {
        width: CapsValue::List(vec![1920, 1280]),
        height: CapsValue::List(vec![1080, 720]),
        pixel_format: CapsValue::Fixed(PixelFormat::I420),
        framerate: CapsValue::Fixed(Framerate::FPS_30),
    };
    println!("Producer caps:");
    println!("  width: {:?}", producer_caps.width);
    println!("  height: {:?}", producer_caps.height);
    println!("  pixel_format: {:?}", producer_caps.pixel_format);
    println!("  framerate: {:?}", producer_caps.framerate);

    // Element B: Accepts any size, YUV formats, 24-60 fps
    let consumer_caps = VideoFormatCaps {
        width: CapsValue::Any,
        height: CapsValue::Any,
        pixel_format: CapsValue::List(vec![
            PixelFormat::I420,
            PixelFormat::Nv12,
            PixelFormat::I422,
        ]),
        framerate: CapsValue::Range {
            min: Framerate::FPS_24,
            max: Framerate::FPS_60,
        },
    };
    println!("\nConsumer caps:");
    println!("  width: {:?}", consumer_caps.width);
    println!("  height: {:?}", consumer_caps.height);
    println!("  pixel_format: {:?}", consumer_caps.pixel_format);
    println!("  framerate: {:?}", consumer_caps.framerate);

    // Negotiate
    let negotiated = producer_caps.intersect(&consumer_caps);
    println!("\nNegotiated caps:");
    if let Some(caps) = &negotiated {
        println!("  width: {:?}", caps.width);
        println!("  height: {:?}", caps.height);
        println!("  pixel_format: {:?}", caps.pixel_format);
        println!("  framerate: {:?}", caps.framerate);

        // Fixate to concrete format
        if let Some(format) = caps.fixate() {
            println!("\nFixated format:");
            println!(
                "  {}x{} {:?} @ {:.2} fps",
                format.width,
                format.height,
                format.pixel_format,
                format.framerate.fps()
            );
            println!("  frame size: {} bytes", format.frame_size());
        }
    }

    // ========================================================================
    // Part 4: Convenience constructors
    // ========================================================================

    println!("\n--- Part 4: Convenience Constructors ---\n");

    // Video format convenience constructors
    let yuv420 = VideoFormatCaps::yuv420();
    println!("VideoFormatCaps::yuv420():");
    println!("  pixel_format: {:?}", yuv420.pixel_format);

    let yuv420_1080p = VideoFormatCaps::yuv420_size(1920, 1080);
    println!("\nVideoFormatCaps::yuv420_size(1920, 1080):");
    println!(
        "  {}x{} {:?}",
        yuv420_1080p.width.as_fixed().unwrap(),
        yuv420_1080p.height.as_fixed().unwrap(),
        yuv420_1080p.pixel_format.as_fixed().unwrap()
    );

    let rgb_with_rate = VideoFormatCaps::rgb()
        .with_size_range(640, 1920, 480, 1080)
        .with_framerate(Framerate::FPS_60);
    println!("\nVideoFormatCaps::rgb().with_size_range(...).with_framerate(60):");
    println!("  pixel_format: {:?}", rgb_with_rate.pixel_format);
    println!("  width: {:?}", rgb_with_rate.width);
    println!("  height: {:?}", rgb_with_rate.height);
    println!("  framerate: {:?}", rgb_with_rate.framerate);

    // Audio format convenience constructors
    let s16_stereo = AudioFormatCaps::s16().with_channels(2).with_rate(48000);
    println!("\nAudioFormatCaps::s16().with_channels(2).with_rate(48000):");
    if let Some(format) = s16_stereo.fixate() {
        println!(
            "  {} Hz, {} ch, {:?}",
            format.sample_rate, format.channels, format.sample_format
        );
    }

    let standard_audio = AudioFormatCaps::standard_rates().with_channels(2);
    let standard_intersected = standard_audio.intersect(&AudioFormatCaps::s16());
    println!("\nAudioFormatCaps::standard_rates() intersect s16():");
    if let Some(caps) = standard_intersected {
        println!("  sample_rate: {:?}", caps.sample_rate);
        println!("  channels: {:?}", caps.channels);
        println!("  sample_format: {:?}", caps.sample_format);
    }

    // ========================================================================
    // Part 5: Fixation with defaults
    // ========================================================================

    println!("\n--- Part 5: Fixation with Defaults ---\n");

    // Unconstrained caps
    let any_video = VideoFormatCaps::any();
    println!("VideoFormatCaps::any():");
    println!("  fixate(): {:?}", any_video.fixate()); // None

    let default_format = any_video.fixate_with_defaults();
    println!(
        "  fixate_with_defaults(): {}x{} {:?} @ {:.2} fps",
        default_format.width,
        default_format.height,
        default_format.pixel_format,
        default_format.framerate.fps()
    );

    let any_audio = AudioFormatCaps::any();
    println!("\nAudioFormatCaps::any():");
    let default_audio = any_audio.fixate_with_defaults();
    println!(
        "  fixate_with_defaults(): {} Hz, {} ch, {:?}",
        default_audio.sample_rate, default_audio.channels, default_audio.sample_format
    );

    // ========================================================================
    // Part 6: MediaCaps - combined format + memory
    // ========================================================================

    println!("\n--- Part 6: MediaCaps (Format + Memory) ---\n");

    use parallax::format::FormatCaps;

    let video_caps = MediaCaps::from_format(FormatCaps::VideoRaw(VideoFormatCaps::yuv420_size(
        1920, 1080,
    )));
    println!("MediaCaps for 1080p I420:");
    if let Some(format) = video_caps.fixate_format() {
        println!("  fixated format: {:?}", format);
    }

    // ========================================================================
    // Part 7: Pixel format variety
    // ========================================================================

    println!("\n--- Part 7: Pixel Format Variety ---\n");

    let formats = [
        (PixelFormat::I420, "YUV 4:2:0 planar (most common)"),
        (
            PixelFormat::Nv12,
            "YUV 4:2:0 semi-planar (hardware decoders)",
        ),
        (PixelFormat::I420_10Le, "YUV 4:2:0 10-bit (HDR)"),
        (PixelFormat::P010, "YUV 4:2:0 10-bit semi-planar"),
        (PixelFormat::I422, "YUV 4:2:2 planar (broadcast)"),
        (PixelFormat::Yuyv, "YUV 4:2:2 packed (webcams)"),
        (PixelFormat::Uyvy, "YUV 4:2:2 packed (pro video)"),
        (PixelFormat::I444, "YUV 4:4:4 planar (full chroma)"),
        (PixelFormat::Rgb24, "RGB 24-bit"),
        (PixelFormat::Rgba, "RGBA 32-bit"),
        (PixelFormat::Bgra, "BGRA 32-bit (common on Windows)"),
        (PixelFormat::Argb, "ARGB 32-bit"),
        (PixelFormat::Gray8, "8-bit grayscale"),
        (PixelFormat::Gray16Le, "16-bit grayscale"),
    ];

    println!("Available pixel formats and frame sizes for 1920x1080:");
    for (format, desc) in formats {
        let vf = VideoFormat::new(1920, 1080, format, Framerate::FPS_30);
        println!("  {:?}: {} bytes - {}", format, vf.frame_size(), desc);
    }

    // ========================================================================
    // Part 8: Real-world negotiation scenario
    // ========================================================================

    println!("\n--- Part 8: Real-World Scenario ---\n");

    // Scenario: Camera -> Encoder -> Network
    // Camera outputs NV12 or I420 at various resolutions
    // Encoder accepts I420 only, 720p-1080p
    // Network prefers smaller frames

    let camera_caps = VideoFormatCaps {
        width: CapsValue::List(vec![1920, 1280, 640]),
        height: CapsValue::List(vec![1080, 720, 480]),
        pixel_format: CapsValue::List(vec![PixelFormat::Nv12, PixelFormat::I420]),
        framerate: CapsValue::Range {
            min: Framerate::FPS_24,
            max: Framerate::FPS_60,
        },
    };

    let encoder_caps = VideoFormatCaps {
        width: CapsValue::Range {
            min: 640,
            max: 1920,
        },
        height: CapsValue::Range {
            min: 480,
            max: 1080,
        },
        pixel_format: CapsValue::Fixed(PixelFormat::I420),
        framerate: CapsValue::Any,
    };

    println!("Camera caps: NV12/I420, multiple resolutions, 24-60fps");
    println!("Encoder caps: I420 only, 640-1920 x 480-1080, any fps");

    if let Some(negotiated) = camera_caps.intersect(&encoder_caps) {
        println!("\nNegotiated:");
        println!("  width: {:?}", negotiated.width);
        println!("  height: {:?}", negotiated.height);
        println!("  pixel_format: {:?}", negotiated.pixel_format);
        println!("  framerate: {:?}", negotiated.framerate);

        if let Some(format) = negotiated.fixate() {
            println!("\nFinal format chosen:");
            println!(
                "  {}x{} {:?} @ {:.2} fps",
                format.width,
                format.height,
                format.pixel_format,
                format.framerate.fps()
            );
            println!("  Frame size: {} bytes", format.frame_size());
        }
    }

    println!("\n=== Demo Complete ===");
}
