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
//! 4. Creating an H.264 decoder with full Vulkan Video setup:
//!    - VideoSession creation with memory bindings
//!    - DPB (Decoded Picture Buffer) management
//!    - H.264 SPS/PPS parsing
//!    - Decode command buffer recording
//!
//! # Run
//!
//! ```bash
//! cargo run --example 18_gpu_decode --features vulkan-video
//! ```

use parallax::gpu::{
    Codec, Dpb, H264ParameterSets, VideoSession, VideoSessionConfig, VideoSessionParameters,
    VulkanContext, VulkanH264Decoder, parse_annexb, vulkan_video_available,
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

        // Demonstrate VideoSession creation
        println!();
        println!("=== VideoSession Creation ===");
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
                println!("  Picture format: {:?}", session.picture_format());

                // Show capabilities
                let caps = session.capabilities();
                println!("  Decode caps: {:?}", caps.decode_caps);

                // Create DPB (Decoded Picture Buffer)
                println!();
                println!("=== DPB Management ===");
                match Dpb::new(&session, None) {
                    Ok(mut dpb) => {
                        println!("DPB created with {} slots", dpb.max_slots());
                        println!("  Format: {:?}", dpb.format());
                        println!("  Extent: {}x{}", dpb.extent().width, dpb.extent().height);

                        // Demonstrate DPB slot management
                        if let Some(slot_idx) = dpb.acquire_slot() {
                            println!("  Acquired slot {} for output", slot_idx);
                            dpb.mark_as_reference(slot_idx, 0, 0, false);
                            println!("  Marked as short-term reference (POC=0)");
                            println!("  Active refs: {}", dpb.active_count());
                            dpb.release_slot(slot_idx);
                            println!("  Released slot {}", slot_idx);
                        }
                    }
                    Err(e) => {
                        println!("DPB creation failed: {}", e);
                    }
                }

                // Create session parameters
                println!();
                println!("=== Session Parameters ===");
                match VideoSessionParameters::new_empty(&session) {
                    Ok(_params) => {
                        println!("Session parameters created successfully!");
                        // In a real decode, we'd update with SPS/PPS via
                        // vkUpdateVideoSessionParametersKHR
                    }
                    Err(e) => {
                        println!("Session parameters creation failed: {}", e);
                    }
                }
            }
            Err(e) => {
                println!("VideoSession creation failed: {}", e);
                println!("(This is expected if the driver doesn't fully support Vulkan Video)");
            }
        }

        // Demonstrate H.264 SPS/PPS parsing
        println!();
        println!("=== H.264 Parameter Parsing ===");

        // Example H.264 bitstream with SPS and PPS
        // This is a minimal baseline profile 320x240 SPS/PPS
        let example_bitstream = [
            // Start code + SPS NAL
            0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0xc0, 0x1e, 0xd9, 0x00, 0x50, 0x05, 0xb8, 0x10,
            0x00, 0x00, 0x3e, 0x90, 0x00, 0x0b, 0xb8, 0x08, 0xf1, 0x83, 0x19, 0x60,
            // Start code + PPS NAL
            0x00, 0x00, 0x00, 0x01, 0x68, 0xce, 0x38, 0x80,
        ];

        // Parse NAL units
        let nals = parse_annexb(&example_bitstream);
        println!("Found {} NAL units in bitstream", nals.len());

        for (i, nal) in nals.iter().enumerate() {
            let nal_type = nal[0] & 0x1F;
            let nal_type_name = match nal_type {
                1 => "Non-IDR Slice",
                5 => "IDR Slice",
                6 => "SEI",
                7 => "SPS",
                8 => "PPS",
                9 => "AUD",
                _ => "Other",
            };
            println!(
                "  NAL {}: type={} ({}), {} bytes",
                i,
                nal_type,
                nal_type_name,
                nal.len()
            );
        }

        // Parse into H264ParameterSets
        let mut param_sets = H264ParameterSets::new();
        for nal in &nals {
            match param_sets.parse_nal(nal) {
                Ok(unit_type) => {
                    println!("  Parsed NAL unit: {:?}", unit_type);
                }
                Err(e) => {
                    println!("  Parse error: {}", e);
                }
            }
        }

        if param_sets.has_parameters() {
            println!("Successfully parsed SPS and PPS!");
            if let Some(sps) = param_sets.get_sps(0) {
                println!(
                    "  SPS[0]: profile_idc={}, level_idc={}",
                    sps.profile_idc, sps.level_idc
                );
                if let Some((w, h)) = param_sets.picture_dimensions(0) {
                    println!("  Dimensions: {}x{}", w, h);
                }
            }
        }

        // In a real application, you would:
        // 1. Wrap the decoder in HwDecoderElement
        // 2. Add it to a pipeline
        // 3. Feed H.264 NAL units to decode

        println!();
        println!("=== Pipeline Integration ===");
        println!("Example usage:");
        println!("  use parallax::elements::codec::HwDecoderElement;");
        println!("  let element = HwDecoderElement::new(decoder);");
        println!(
            "  pipeline.add_node(\"hw_decoder\", DynAsyncElement::new_box(TransformAdapter::new(element)));"
        );
    } else {
        println!("H.264 decode not supported on this GPU.");
    }

    println!();
    println!("=== Implementation Status ===");
    println!("Complete:");
    println!("  ✓ VulkanContext for device/instance management");
    println!("  ✓ VideoSession creation with memory bindings");
    println!("  ✓ DPB (Decoded Picture Buffer) management");
    println!("  ✓ H.264 SPS/PPS parsing");
    println!("  ✓ Decode command buffer recording");
    println!("  ✓ HwDecoderElement pipeline wrapper");
    println!("  ✓ HwEncoderElement pipeline wrapper");
    println!();
    println!("In Progress:");
    println!("  - Slice header parsing");
    println!("  - Reference list construction");
    println!("  - Output frame readback");
    println!("  - VulkanH264Encoder implementation");

    Ok(())
}
