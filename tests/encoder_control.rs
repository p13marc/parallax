//! Integration tests for live bandwidth control (#33).
//!
//! Elements are moved into their executor tasks at `start()`, so every knob
//! that governs bandwidth — bitrate, resolution, framerate, keyframe spacing —
//! is reachable only through a control handle cloned *before* start. These
//! tests drive a **running** `AppSrc → VideoScale → H264Encoder → AppSink`
//! pipeline, change those knobs mid-stream, and decode the result to check that
//! the stream really did change.
#![cfg(feature = "h264")]

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::control::Controllable;
use parallax::element::Element;
use parallax::elements::app::{AppSink, AppSrc, Pulled};
use parallax::elements::codec::{H264Decoder, H264Encoder, H264EncoderConfig, RateControlMode};
use parallax::elements::transform::VideoScale;
use parallax::format::PixelFormat;
use parallax::memory::SharedArena;
use parallax::metadata::{BufferFlags, Metadata};
use parallax::pipeline::{Executor, Pipeline};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 480;
const FRAME_SIZE: usize = (WIDTH as usize) * (HEIGHT as usize) * 3 / 2;

fn test_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(FRAME_SIZE, 64).unwrap())
}

/// A 640x480 I420 frame with spatial detail (so encoding is non-degenerate),
/// declaring its geometry the way any real source element would.
fn yuv_frame(seq: u64) -> Buffer {
    let arena = test_arena();
    arena.reclaim();
    let mut slot = arena.acquire().expect("arena slot");
    let data = slot.data_mut();
    data.fill(128);
    for y in 0..HEIGHT as usize {
        for x in 0..WIDTH as usize {
            data[y * WIDTH as usize + x] = ((x * 7 + y * 13) as u8).wrapping_add(seq as u8);
        }
    }

    let mut metadata = Metadata::from_sequence(seq);
    metadata.set_video_dims(WIDTH, HEIGHT, PixelFormat::I420);
    Buffer::new(MemoryHandle::with_len(slot, FRAME_SIZE), metadata)
}

/// Annex-B scan for an IDR NAL (type 5).
fn contains_idr(data: &[u8]) -> bool {
    let mut i = 0;
    while i + 3 < data.len() {
        let offset = if data[i..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if data[i..].starts_with(&[0, 0, 1]) {
            3
        } else {
            i += 1;
            continue;
        };
        if i + offset < data.len() && data[i + offset] & 0x1F == 5 {
            return true;
        }
        i += offset;
    }
    false
}

/// Wait until the sink has at least `n` buffers, so the pipeline is provably
/// caught up before we change anything.
async fn wait_for(sink: &parallax::elements::app::AppSinkHandle, n: usize) {
    let deadline = Instant::now() + Duration::from_secs(20);
    while sink.queue_len() < n {
        assert!(
            Instant::now() < deadline,
            "sink never received {n} buffers (got {})",
            sink.queue_len()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// The whole epic in one test: on a **running** pipeline, drop the bitrate and
/// the resolution, and confirm the encoded stream actually shrinks and resizes.
///
/// This is what ZenSight's `max_height` and bitrate knobs ride on.
#[tokio::test(flavor = "multi_thread")]
async fn bitrate_and_resolution_change_on_a_running_pipeline() {
    let src = AppSrc::new();
    let src_handle = src.handle();
    let sink = AppSink::new();
    let sink_handle = sink.handle();

    let scaler = VideoScale::new();
    let scale = scaler.control();

    let mut config = H264EncoderConfig::new()
        .bitrate(4_000_000)
        .rate_control(RateControlMode::Bitrate)
        // Hold the rate by spending quality, not by dropping frames — otherwise
        // the encoder emits nothing for some inputs and the stream silently
        // loses frames rather than bytes.
        .skip_frames(false);
    config.scene_change_detect = false; // deterministic IDR placement
    let encoder = H264Encoder::new(config).unwrap();
    let control = encoder.control();

    let mut pipeline = Pipeline::new();
    let s = pipeline.add_source("src", src);
    let sc = pipeline.add_filter("scale", scaler);
    let e = pipeline.add_filter("enc", encoder);
    let k = pipeline.add_async_sink("sink", sink);
    pipeline.link(s, sc).unwrap();
    pipeline.link(sc, e).unwrap();
    pipeline.link(e, k).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Phase 1: 640x480 at 4 Mbps.
    for seq in 0..10 {
        src_handle.push_buffer(yuv_frame(seq)).await.unwrap();
    }
    wait_for(&sink_handle, 10).await;

    // The change, mid-flight, with no teardown: quarter the pixels and drop the
    // bitrate 10x.
    scale.set_max_height(240);
    control.set_bitrate(400_000);

    // Phase 2: 320x240 at 400 kbps.
    for seq in 10..20 {
        src_handle.push_buffer(yuv_frame(seq)).await.unwrap();
    }
    src_handle.end_stream();
    handle.wait().await.unwrap();

    let mut outputs = Vec::new();
    while let Pulled::Buffer(buffer) = sink_handle.try_pull_buffer() {
        outputs.push(buffer);
    }
    assert_eq!(outputs.len(), 20, "one encoded frame per input frame");

    // The stream must remain decodable across the change, and the decoded
    // frames must actually change size.
    let mut decoder = H264Decoder::new().unwrap();
    let mut sizes = Vec::new();
    for buffer in &outputs {
        // The decoder is an Element (its owned-frame API went private in
        // #160); geometry travels in the decoded buffer's metadata.
        if let Some(frame) = decoder.process(buffer.clone()).unwrap() {
            let (w, h) = frame.metadata().video_dims().expect("decoded dims");
            sizes.push((w as usize, h as usize));
        }
    }
    assert!(
        sizes.contains(&(640, 480)),
        "expected 640x480 frames before the change, got {sizes:?}"
    );
    assert!(
        sizes.contains(&(320, 240)),
        "expected 320x240 frames after the change, got {sizes:?}"
    );

    // The frame carrying the change must be an IDR, or a viewer that joins
    // after it can never decode.
    assert!(
        contains_idr(outputs[10].as_bytes()),
        "the first frame after the change must be a keyframe"
    );
    assert!(
        outputs[10]
            .metadata()
            .flags
            .contains(BufferFlags::SYNC_POINT),
        "and must be flagged SYNC_POINT for downstream consumers"
    );

    // And the point of the whole exercise: fewer bytes on the wire.
    let mean = |frames: &[Buffer]| -> usize {
        frames.iter().map(|b| b.as_bytes().len()).sum::<usize>() / frames.len()
    };
    // Skip the IDR at each phase boundary — it is expected to be large.
    let before = mean(&outputs[2..10]);
    let after = mean(&outputs[12..20]);

    assert!(
        after * 2 < before,
        "quartering the pixels and cutting the bitrate 10x must shrink the \
         stream materially (got {after} vs {before} bytes/frame)"
    );
}

/// The scaler alone: resolution can be changed without touching the encoder,
/// and changed back.
#[tokio::test(flavor = "multi_thread")]
async fn resolution_can_be_lowered_and_restored_while_running() {
    let src = AppSrc::new();
    let src_handle = src.handle();
    let sink = AppSink::new();
    let sink_handle = sink.handle();

    let scaler = VideoScale::new();
    let scale = scaler.control();

    let mut config = H264EncoderConfig::new();
    config.scene_change_detect = false;
    let encoder = H264Encoder::new(config).unwrap();

    let mut pipeline = Pipeline::new();
    let s = pipeline.add_source("src", src);
    let sc = pipeline.add_filter("scale", scaler);
    let e = pipeline.add_filter("enc", encoder);
    let k = pipeline.add_async_sink("sink", sink);
    pipeline.link(s, sc).unwrap();
    pipeline.link(sc, e).unwrap();
    pipeline.link(e, k).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    src_handle.push_buffer(yuv_frame(0)).await.unwrap();
    wait_for(&sink_handle, 1).await;

    scale.set_max_height(120);
    src_handle.push_buffer(yuv_frame(1)).await.unwrap();
    wait_for(&sink_handle, 2).await;

    scale.passthrough();
    src_handle.push_buffer(yuv_frame(2)).await.unwrap();
    src_handle.end_stream();
    handle.wait().await.unwrap();

    let mut outputs = Vec::new();
    while let Pulled::Buffer(buffer) = sink_handle.try_pull_buffer() {
        outputs.push(buffer);
    }
    assert_eq!(outputs.len(), 3);

    let mut decoder = H264Decoder::new().unwrap();
    let mut sizes = Vec::new();
    for buffer in &outputs {
        if let Some(frame) = decoder.process(buffer.clone()).unwrap() {
            let (w, h) = frame.metadata().video_dims().expect("decoded dims");
            sizes.push((w as usize, h as usize));
        }
    }

    assert!(
        sizes.contains(&(160, 120)),
        "expected a 160x120 frame after set_max_height(120), got {sizes:?}"
    );
    assert!(
        sizes.iter().filter(|&&s| s == (640, 480)).count() >= 2,
        "expected full resolution before the change and after passthrough(), got {sizes:?}"
    );
}

/// Encode `frames` frames, applying `change` to the control handle after
/// `change_at` of them have reached the sink. Returns the encoded buffers.
async fn run_with_change(
    encoder: H264Encoder,
    frames: u64,
    change_at: u64,
    change: impl FnOnce(&parallax::control::EncoderControl),
) -> Vec<Buffer> {
    let src = AppSrc::new();
    let src_handle = src.handle();
    let sink = AppSink::new();
    let sink_handle = sink.handle();
    let control = encoder.control();

    let mut pipeline = Pipeline::new();
    let s = pipeline.add_source("src", src);
    let e = pipeline.add_filter("enc", encoder);
    let k = pipeline.add_async_sink("sink", sink);
    pipeline.link(s, e).unwrap();
    pipeline.link(e, k).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for seq in 0..change_at {
        src_handle.push_buffer(yuv_frame(seq)).await.unwrap();
    }
    wait_for(&sink_handle, change_at as usize).await;

    change(&control);

    for seq in change_at..frames {
        src_handle.push_buffer(yuv_frame(seq)).await.unwrap();
    }
    src_handle.end_stream();
    handle.wait().await.unwrap();

    let mut outputs = Vec::new();
    while let Pulled::Buffer(buffer) = sink_handle.try_pull_buffer() {
        outputs.push(buffer);
    }
    outputs
}

fn quiet_encoder(config: H264EncoderConfig) -> H264Encoder {
    let mut config = config;
    config.scene_change_detect = false;
    config.keyframe_interval = 0; // no periodic IDRs: only frame 0 and forced ones
    H264Encoder::new(config).unwrap()
}

/// #42: a bitrate change goes in through OpenH264's `SetOption`, so the GOP is
/// not broken — the stream keeps flowing, no IDR, and the bytes still fall.
///
/// Contrast with the resolution change above, which *must* IDR because the
/// encoder re-initialises.
#[tokio::test(flavor = "multi_thread")]
async fn a_bitrate_change_is_seamless_and_forces_no_idr() {
    let encoder = quiet_encoder(
        H264EncoderConfig::new()
            .bitrate(4_000_000)
            .rate_control(RateControlMode::Bitrate),
    );

    let outputs = run_with_change(encoder, 24, 12, |control| control.set_bitrate(200_000)).await;
    assert_eq!(outputs.len(), 24);

    for (i, buffer) in outputs.iter().enumerate().skip(1) {
        assert!(
            !contains_idr(buffer.as_bytes()),
            "frame {i}: a bitrate change must not break the GOP (only frame 0 is an IDR)"
        );
    }

    let mean = |frames: &[Buffer]| -> usize {
        frames.iter().map(|b| b.as_bytes().len()).sum::<usize>() / frames.len()
    };
    let before = mean(&outputs[2..12]);
    let after = mean(&outputs[16..24]);
    assert!(
        after < before,
        "the new bitrate must actually take effect ({after} vs {before} bytes/frame)"
    );
}

/// #41: setting a parameter to the value it already has is a no-op. It used to
/// rebuild the encoder unconditionally, forcing a pointless IDR — brutal on a
/// controller that reasserts its desired bitrate periodically.
#[tokio::test(flavor = "multi_thread")]
async fn a_no_op_change_forces_no_idr() {
    let encoder = quiet_encoder(
        H264EncoderConfig::new()
            .bitrate(2_000_000)
            .rate_control(RateControlMode::Bitrate),
    );

    // Set the bitrate to exactly what it already is, and change the GOP to the
    // value it already has. Neither is a change.
    let outputs = run_with_change(encoder, 10, 5, |control| {
        control.set_bitrate(2_000_000);
        control.set_keyframe_interval(0);
    })
    .await;

    assert_eq!(outputs.len(), 10);
    for (i, buffer) in outputs.iter().enumerate().skip(1) {
        assert!(
            !contains_idr(buffer.as_bytes()),
            "frame {i}: a no-op reconfigure must not force a keyframe"
        );
    }
}

/// #40: rate control is a live knob, not a construction-time one. Switching to
/// Bitrate mode mid-stream makes the encoder start honouring the target.
#[tokio::test(flavor = "multi_thread")]
async fn rate_control_can_be_switched_on_a_running_encoder() {
    // Start quality-first with a tiny nominal target it is free to ignore.
    let encoder = quiet_encoder(
        H264EncoderConfig::new()
            .bitrate(150_000)
            .qp(20)
            .rate_control(RateControlMode::Quality),
    );

    let outputs = run_with_change(encoder, 24, 12, |control| {
        control.set_rate_control(RateControlMode::Bitrate);
    })
    .await;
    assert_eq!(outputs.len(), 24);

    let mean = |frames: &[Buffer]| -> usize {
        frames.iter().map(|b| b.as_bytes().len()).sum::<usize>() / frames.len()
    };
    // Skip the IDR the mode switch produces (it rebuilds the encoder).
    let before = mean(&outputs[2..12]);
    let after = mean(&outputs[14..24]);
    assert!(
        after < before,
        "honouring a 150 kbps target must cost fewer bytes than ignoring it \
         ({after} vs {before} bytes/frame)"
    );
}

/// #44: the counters are readable while the encoder is inside its executor task.
/// `frame_count()` and `bytes_encoded()` are `&self` on the element, and the
/// element is moved at start() — so they never could be.
#[tokio::test(flavor = "multi_thread")]
async fn encoder_stats_are_readable_on_a_running_pipeline() {
    let src = AppSrc::new();
    let src_handle = src.handle();
    let sink = AppSink::new();
    let sink_handle = sink.handle();

    let encoder = quiet_encoder(H264EncoderConfig::new());
    let stats = encoder.stats(); // BEFORE start

    let mut pipeline = Pipeline::new();
    let s = pipeline.add_source("src", src);
    let e = pipeline.add_filter("enc", encoder);
    let k = pipeline.add_async_sink("sink", sink);
    pipeline.link(s, e).unwrap();
    pipeline.link(e, k).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    assert_eq!(stats.frames_encoded(), 0);

    for seq in 0..8 {
        src_handle.push_buffer(yuv_frame(seq)).await.unwrap();
    }
    wait_for(&sink_handle, 8).await;

    // Read them mid-flight — the pipeline is still running.
    let mid = stats.snapshot();
    assert_eq!(mid.frames_encoded, 8);
    assert!(mid.bytes_encoded > 0);
    assert!(mid.last_encode_ns > 0, "encode time must be recorded");
    assert_eq!(
        mid.frames_dropped_by_rc, 0,
        "skip_frames is off by default, so nothing may be dropped"
    );

    src_handle.end_stream();
    handle.wait().await.unwrap();

    let end = stats.snapshot();
    assert_eq!(end.frames_encoded, 8);
    assert_eq!(
        end.bytes_encoded,
        sink_handle_total(&sink_handle),
        "bytes_encoded must match what actually reached the sink"
    );
}

fn sink_handle_total(sink: &parallax::elements::app::AppSinkHandle) -> u64 {
    let mut total = 0;
    while let Pulled::Buffer(buffer) = sink.try_pull_buffer() {
        total += buffer.as_bytes().len() as u64;
    }
    total
}
