//! Integration tests for runtime keyframe requests (#12).
//!
//! Elements are moved into their executor tasks at `start()`, so the only way
//! to force an IDR on a live encoder is the [`KeyframeHandle`] cloned before
//! start, or the in-band [`KEYFRAME_REQUEST`] metadata flag. Both paths are
//! exercised here against a running `AppSrc → H264Encoder → AppSink` pipeline.
#![cfg(feature = "h264")]

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::elements::app::{AppSink, AppSrc};
use parallax::elements::codec::{
    EncoderElement, H264Encoder, H264EncoderConfig, KEYFRAME_REQUEST, KeyframeHandle,
};
use parallax::memory::SharedArena;
use parallax::metadata::{BufferFlags, Metadata};
use parallax::pipeline::{Executor, Pipeline};

const WIDTH: u32 = 320;
const HEIGHT: u32 = 240;
const FRAME_SIZE: usize = (WIDTH as usize) * (HEIGHT as usize) * 3 / 2;

fn test_arena() -> &'static SharedArena {
    static ARENA: OnceLock<SharedArena> = OnceLock::new();
    ARENA.get_or_init(|| SharedArena::new(FRAME_SIZE, 64).unwrap())
}

/// An I420 frame with spatial detail (so encoding is non-degenerate).
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
    Buffer::new(
        MemoryHandle::with_len(slot, FRAME_SIZE),
        Metadata::from_sequence(seq),
    )
}

/// Encoder config with no periodic or scene-change IDRs, so the only
/// keyframes are frame 0 and explicitly requested ones.
fn quiet_encoder() -> H264Encoder {
    let mut config = H264EncoderConfig::new(WIDTH, HEIGHT);
    config.scene_change_detect = false;
    H264Encoder::new(config).unwrap()
}

fn is_sync_point(buffer: &Buffer) -> bool {
    buffer.metadata().flags.contains(BufferFlags::SYNC_POINT)
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

/// Requesting a keyframe on a RUNNING pipeline forces an IDR on the very
/// next frame the encoder processes, flagged SYNC_POINT.
#[tokio::test(flavor = "multi_thread")]
async fn keyframe_handle_reaches_running_encoder() {
    let src = AppSrc::new();
    let src_handle = src.handle();
    let sink = AppSink::new();
    let sink_handle = sink.handle();
    let encoder = quiet_encoder();
    let keyframes: KeyframeHandle = encoder.keyframe_handle();

    let mut pipeline = Pipeline::new();
    let s = pipeline.add_source("src", src);
    let e = pipeline.add_filter("enc", encoder);
    let k = pipeline.add_sink("sink", sink);
    pipeline.link(s, e).unwrap();
    pipeline.link(e, k).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    // Phase 1: five frames, then wait until the sink has all of them —
    // at that point the encoder is provably idle.
    for seq in 0..5 {
        src_handle.push_buffer(yuv_frame(seq)).unwrap();
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while sink_handle.queue_len() < 5 {
        assert!(Instant::now() < deadline, "sink never received 5 frames");
        tokio::time::sleep(Duration::from_millis(10)).await;
    }

    // The request must land on the NEXT frame (index 5).
    keyframes.request();

    for seq in 5..10 {
        src_handle.push_buffer(yuv_frame(seq)).unwrap();
    }
    src_handle.end_stream();
    handle.wait().await.unwrap();

    let mut outputs = Vec::new();
    while let Some(buffer) = sink_handle.try_pull_buffer() {
        outputs.push(buffer);
    }
    assert_eq!(outputs.len(), 10, "one encoded output per input frame");

    for (i, buffer) in outputs.iter().enumerate() {
        let idr = contains_idr(buffer.as_bytes());
        let sync = is_sync_point(buffer);
        assert_eq!(
            idr, sync,
            "SYNC_POINT flag must track IDR presence (frame {i})"
        );
        let expect_idr = i == 0 || i == 5;
        assert_eq!(
            idr, expect_idr,
            "frame {i}: expected IDR only at 0 (stream start) and 5 (requested)"
        );
    }
}

/// The in-band KEYFRAME_REQUEST metadata flag forces an IDR on the frame
/// that carries it (AppSrc-fed pipelines stamp it on injected buffers).
#[tokio::test(flavor = "multi_thread")]
async fn in_band_metadata_flag_forces_idr() {
    let src = AppSrc::new();
    let src_handle = src.handle();
    let sink = AppSink::new();
    let sink_handle = sink.handle();

    let mut pipeline = Pipeline::new();
    let s = pipeline.add_source("src", src);
    let e = pipeline.add_filter("enc", quiet_encoder());
    let k = pipeline.add_sink("sink", sink);
    pipeline.link(s, e).unwrap();
    pipeline.link(e, k).unwrap();

    let executor = Executor::new();
    let handle = executor.start(&mut pipeline).unwrap();

    for seq in 0..6 {
        let mut buffer = yuv_frame(seq);
        if seq == 3 {
            buffer.metadata_mut().set(KEYFRAME_REQUEST, true);
        }
        src_handle.push_buffer(buffer).unwrap();
    }
    src_handle.end_stream();
    handle.wait().await.unwrap();

    let mut outputs = Vec::new();
    while let Some(buffer) = sink_handle.try_pull_buffer() {
        outputs.push(buffer);
    }
    assert_eq!(outputs.len(), 6);
    for (i, buffer) in outputs.iter().enumerate() {
        let expect_idr = i == 0 || i == 3;
        assert_eq!(
            contains_idr(buffer.as_bytes()),
            expect_idr,
            "frame {i}: expected IDR only at 0 and 3 (metadata-flagged)"
        );
    }
}

/// EncoderElement routes handle requests through VideoEncoder::force_keyframe.
#[test]
fn encoder_element_honors_handle_via_trait() {
    let encoder = quiet_encoder();
    let mut element = EncoderElement::new(encoder, WIDTH, HEIGHT).unwrap();
    let keyframes = element.keyframe_handle();

    use parallax::element::Transform;
    let mut idr_frames = Vec::new();
    for seq in 0..6 {
        if seq == 4 {
            keyframes.request();
        }
        let out = element.transform(yuv_frame(seq)).unwrap();
        for buffer in out.into_iter() {
            if contains_idr(buffer.as_bytes()) {
                idr_frames.push(seq);
            }
        }
    }
    assert_eq!(
        idr_frames,
        vec![0, 4],
        "IDR at stream start and at the requested frame"
    );
}
