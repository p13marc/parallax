//! Example 18: GPU Video Decode
//!
//! Demonstrates hardware-accelerated video decoding using Vulkan Video.
//!
//! # Prerequisites
//!
//! - Vulkan 1.3+ capable GPU
//! - Mesa 23.1+ (AMD/Intel) or NVIDIA 525+ drivers
//! - Vulkan Video extensions support
//!
//! # What This Example Shows
//!
//! 1. Checking for Vulkan Video availability
//! 2. Creating a VulkanContext
//! 3. Querying decode capabilities
//! 4. Creating an H.264 decoder
//! 5. Wrapping the decoder in HwDecoderElement for pipeline use
//!
//! # Run
//!
//! ```bash
//! cargo run --example 18_gpu_decode --features vulkan-video
//! ```

use parallax::gpu::{
    Codec, VideoSession, VideoSessionConfig, VulkanContext, VulkanH264Decoder,
    vulkan_video_available,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Check if Vulkan Video is available
    println!("Checking Vulkan Video availability...");

    if !vulkan_video_available() {
        println!("Vulkan Video is NOT available on this system.");
        println!();
        println!("Possible reasons:");
        println!("  - No Vulkan driver installed");
        println!("  - Vulkan version < 1.3");
        println!("  - GPU doesn't support Vulkan Video extensions");
        println!("  - Driver doesn't support Vulkan Video");
        println!();
        println!("Requirements:");
        println!("  - AMD: RADV driver (Mesa 23.1+)");
        println!("  - Intel: ANV driver (Mesa 23.1+)");
        println!("  - NVIDIA: Proprietary driver 525+ or NVK");
        return Ok(());
    }

    println!("Vulkan Video is available!");
    println!();

    // Create Vulkan context
    let ctx = VulkanContext::new()?;

    // Print device info
    println!("GPU: {}", ctx.device_name());
    println!();

    // Query decode capabilities
    println!("Decode capabilities:");
    for caps in ctx.all_decode_capabilities() {
        println!(
            "  {}: profiles {:?}, max {}x{}, {} bit",
            caps.codec,
            caps.profiles,
            caps.max_width,
            caps.max_height,
            caps.bit_depths
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join("/"),
        );
    }

    if ctx.all_decode_capabilities().is_empty() {
        println!("  (none - GPU doesn't expose video decode extensions)");
    }

    println!();

    // Query encode capabilities
    println!("Encode capabilities:");
    for caps in ctx.all_encode_capabilities() {
        println!(
            "  {}: profiles {:?}, max {}x{}, {} bit",
            caps.codec,
            caps.profiles,
            caps.max_width,
            caps.max_height,
            caps.bit_depths
                .iter()
                .map(|b| b.to_string())
                .collect::<Vec<_>>()
                .join("/"),
        );
    }

    if ctx.all_encode_capabilities().is_empty() {
        println!("  (none - GPU doesn't expose video encode extensions)");
    }

    println!();

    // Check specific codec support
    let codecs = [Codec::H264, Codec::H265, Codec::Av1, Codec::Vp9];
    println!("Codec support:");
    for codec in codecs {
        let decode = ctx.supports_decode(codec);
        let encode = ctx.supports_encode(codec);
        println!("  {}: decode={}, encode={}", codec, decode, encode);
    }

    println!();

    // Create H.264 decoder if supported
    if ctx.supports_decode(Codec::H264) {
        println!("Creating H.264 decoder (1920x1080)...");
        let _decoder = VulkanH264Decoder::new(&ctx, 1920, 1080)?;
        println!("H.264 decoder created successfully!");

        // Try to create a VideoSession (the core Vulkan Video object)
        println!();
        println!("Creating VideoSession for H.264 decode...");
        let config = VideoSessionConfig {
            profile: parallax::gpu::VideoProfile {
                codec: Codec::H264,
                profile: 100, // High profile
                level: 51,    // Level 5.1
                chroma_format: parallax::gpu::ChromaFormat::Yuv420,
                bit_depth: 8,
            },
            max_width: 1920,
            max_height: 1080,
            ..Default::default()
        };

        match VideoSession::new_decode(&ctx, config) {
            Ok(session) => {
                println!("VideoSession created successfully!");
                println!(
                    "  Coded extent: {}x{}",
                    session.coded_extent().width,
                    session.coded_extent().height
                );
                println!("  Max DPB slots: {}", session.max_dpb_slots());
                println!("  Max active refs: {}", session.max_active_refs());
            }
            Err(e) => {
                println!("VideoSession creation failed: {}", e);
                println!("(This is expected if the driver doesn't fully support Vulkan Video)");
            }
        }

        // In a real application, you would:
        // 1. Wrap the decoder in HwDecoderElement
        // 2. Add it to a pipeline
        // 3. Feed H.264 NAL units to decode

        println!();
        println!("Example usage in pipeline:");
        println!("  use parallax::elements::codec::HwDecoderElement;");
        println!("  let element = HwDecoderElement::new(decoder);");
        println!(
            "  pipeline.add_node(\"hw_decoder\", DynAsyncElement::new_box(TransformAdapter::new(element)));"
        );
    } else {
        println!("H.264 decode not supported on this GPU.");
    }

    println!();
    println!("Note: This example demonstrates GPU decode capabilities.");
    println!("The implementation includes:");
    println!("  + VulkanContext for device/instance management");
    println!("  + VideoSession for Vulkan Video session creation");
    println!("  + DPB (Decoded Picture Buffer) management");
    println!("  + HwDecoderElement for pipeline integration");
    println!();
    println!("Remaining work for full decode:");
    println!("  - H.264 SPS/PPS parsing and session parameters");
    println!("  - Decode command buffer recording");
    println!("  - Output frame handling");

    Ok(())
}
