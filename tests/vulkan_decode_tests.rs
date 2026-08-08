//! Vulkan Video H.264 decode integration test.
//!
//! Skips (passes with a message) on machines without a Vulkan Video capable
//! device — e.g. Intel Gen9.5, where ANV does not expose the video queue.
//! On capable hardware (RADV, ANV Gen12+, NVIDIA 525+) it decodes a real
//! 3-frame 64x64 stream and checks the pixels came from the GPU, not from
//! uninitialised memory.

#![cfg(feature = "vulkan-video")]

use std::sync::Arc;

use parallax::gpu::{Codec, HwVideoDecoder, VulkanContext, VulkanH264Decoder};

const FIXTURE: &[u8] = include_bytes!("fixtures/tiny_64x64.h264");

/// A context on a device that can actually decode H.264, or `None` → skip.
fn decode_capable_context() -> Option<Arc<VulkanContext>> {
    let ctx = match VulkanContext::new() {
        Ok(ctx) => ctx,
        Err(e) => {
            eprintln!("skipping: no Vulkan context ({e})");
            return None;
        }
    };
    if ctx.decode_queue().is_none() {
        eprintln!("skipping: device has no video decode queue");
        return None;
    }
    if !ctx.supports_decode(Codec::H264) {
        eprintln!("skipping: device does not expose H.264 decode");
        return None;
    }
    Some(Arc::new(ctx))
}

#[test]
fn decodes_a_real_stream_when_hardware_is_present() {
    let Some(ctx) = decode_capable_context() else {
        return;
    };

    let mut decoder = VulkanH264Decoder::new(ctx).expect("decoder on a capable device");

    let frames = decoder.decode(FIXTURE, 0).expect("decode the fixture");
    assert_eq!(frames.len(), 3, "the fixture holds three IDR frames");

    for (i, frame) in frames.iter().enumerate() {
        assert_eq!((frame.width, frame.height), (64, 64), "frame {i} geometry");
        assert!(frame.is_keyframe, "every fixture frame is an IDR");
        assert_eq!(
            frame.buffer.size,
            64 * 64 * 3 / 2,
            "frame {i} is NV12-sized"
        );
    }

    // The luma plane of a testsrc frame is not a constant field — if it is,
    // the bytes never came from the decoder.
    let frame = &frames[0];
    let mut pixels = vec![0u8; frame.format.frame_size(frame.width, frame.height)];
    decoder
        .read_frame(frame, &mut pixels)
        .expect("read decoded pixels");
    let luma = &pixels[..(frame.width * frame.height) as usize];
    assert!(
        luma.iter().any(|&b| b != luma[0]),
        "decoded luma plane is a constant field — the pixels never came from the GPU"
    );

    assert!(!decoder.has_pending(), "synchronous decode holds nothing");
    let drained = decoder.flush().expect("flush");
    assert!(drained.is_empty());
}

#[test]
fn reset_allows_a_fresh_start() {
    let Some(ctx) = decode_capable_context() else {
        return;
    };

    let mut decoder = VulkanH264Decoder::new(ctx).expect("decoder on a capable device");
    assert_eq!(decoder.decode(FIXTURE, 0).expect("first pass").len(), 3);
    decoder.reset().expect("reset");
    assert_eq!(decoder.decode(FIXTURE, 0).expect("second pass").len(), 3);
}
