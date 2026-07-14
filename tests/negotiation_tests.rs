//! Integration tests for automatic converter insertion (#37).
//!
//! A pipeline whose source pins one geometry and pixel format and whose sink
//! pins another cannot run as linked. Negotiation must notice *which axes*
//! disagree and splice a chain that covers all of them — or refuse. Before this
//! the registry held one converter per `(VideoRaw, VideoRaw, Cpu, Cpu)` key, so
//! a scaler could not even be registered without evicting `videoconvert`, and
//! the single converter it did insert was hardcoded to RGBA regardless of what
//! the sink asked for.

use std::sync::{Arc, Mutex};

use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{ConsumeContext, ProduceContext, ProduceResult, Sink, Source};
use parallax::error::Result;
use parallax::format::{
    CapsValue, ElementMediaCaps, FormatMemoryCap, Framerate, MemoryCaps, PixelFormat, VideoFormat,
    VideoFormatCaps,
};
use parallax::memory::SharedArena;
use parallax::metadata::Metadata;
use parallax::pipeline::{ConverterPolicy, Pipeline};

/// Bytes in one frame of this geometry and pixel format.
fn frame_size(width: u32, height: u32, pixel_format: PixelFormat) -> usize {
    VideoFormat::new(width, height, pixel_format, Framerate::FPS_30).frame_size()
}

/// Caps pinning exactly one geometry and pixel format, like a camera or an
/// `EncoderElement` does.
fn pinned(width: u32, height: u32, pixel_format: PixelFormat) -> ElementMediaCaps {
    ElementMediaCaps::single(FormatMemoryCap::new(
        VideoFormatCaps {
            width: CapsValue::Fixed(width),
            height: CapsValue::Fixed(height),
            pixel_format: CapsValue::Fixed(pixel_format),
            ..VideoFormatCaps::any()
        }
        .into(),
        MemoryCaps::cpu_only(),
    ))
}

/// A source that emits a fixed number of frames at a fixed size and pixel
/// format, and declares both — the way a camera or a screen grabber does.
struct PinnedSource {
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    remaining: usize,
    arena: SharedArena,
}

impl PinnedSource {
    fn new(width: u32, height: u32, pixel_format: PixelFormat, frames: usize) -> Self {
        let size = frame_size(width, height, pixel_format);
        Self {
            width,
            height,
            pixel_format,
            remaining: frames,
            arena: SharedArena::new(size, 16).unwrap(),
        }
    }
}

impl Source for PinnedSource {
    fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
        if self.remaining == 0 {
            return Ok(ProduceResult::Eos);
        }
        self.remaining -= 1;

        let size = frame_size(self.width, self.height, self.pixel_format);
        self.arena.reclaim();
        let mut slot = self.arena.acquire().expect("arena slot");
        slot.data_mut().fill(0x40);

        let mut metadata = Metadata::from_sequence(self.remaining as u64);
        metadata.set_video_dims(self.width, self.height, self.pixel_format);
        Ok(ProduceResult::OwnBuffer(Buffer::new(
            MemoryHandle::with_len(slot, size),
            metadata,
        )))
    }

    fn output_media_caps(&self) -> ElementMediaCaps {
        pinned(self.width, self.height, self.pixel_format)
    }
}

/// A sink that only accepts one geometry and pixel format, and records what it
/// actually received.
struct PinnedSink {
    width: u32,
    height: u32,
    pixel_format: PixelFormat,
    seen: Arc<Mutex<Vec<(u32, u32, usize)>>>,
}

impl Sink for PinnedSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let buffer = ctx.buffer();
        let (width, height) = buffer
            .metadata()
            .video_dims()
            .expect("geometry in metadata");
        self.seen
            .lock()
            .unwrap()
            .push((width, height, buffer.as_bytes().len()));
        Ok(())
    }

    fn input_media_caps(&self) -> ElementMediaCaps {
        pinned(self.width, self.height, self.pixel_format)
    }
}

/// Node names of every auto-inserted converter, in graph order.
fn auto_nodes(pipeline: &Pipeline) -> Vec<String> {
    pipeline
        .nodes()
        .map(|(_, node)| node.name().to_string())
        .filter(|name| name.starts_with("auto_"))
        .collect()
}

/// The zensight case: a 1080p RGB camera into a 720p I420 encoder. Both axes
/// disagree, so both converters must be inserted — and the scale must come
/// first, because converting 1280x720 pixels beats converting 1920x1080 of them.
#[tokio::test]
async fn a_format_and_geometry_mismatch_inserts_a_scaler_and_a_converter() {
    let seen = Arc::new(Mutex::new(Vec::new()));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "camera",
        PinnedSource::new(1920, 1080, PixelFormat::Rgb24, 3),
    );
    let sink = pipeline.add_sink(
        "encoder",
        PinnedSink {
            width: 1280,
            height: 720,
            pixel_format: PixelFormat::I420,
            seen: seen.clone(),
        },
    );
    pipeline.link(src, sink).unwrap();

    pipeline.set_converter_policy(ConverterPolicy::Allow);
    pipeline.prepare().unwrap();

    let inserted = auto_nodes(&pipeline);
    assert_eq!(
        inserted.len(),
        2,
        "both the format and the geometry axis need covering, got {inserted:?}"
    );
    assert!(inserted.iter().any(|n| n.starts_with("auto_videoscale")));
    assert!(inserted.iter().any(|n| n.starts_with("auto_videoconvert")));

    pipeline.run().await.unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 3, "every frame reaches the sink");
    for (width, height, len) in seen.iter() {
        assert_eq!((*width, *height), (1280, 720), "scaled to the sink's size");
        assert_eq!(
            *len,
            frame_size(1280, 720, PixelFormat::I420),
            "and converted to the sink's pixel format"
        );
    }
}

/// Only the geometry disagrees: one scaler, no converter.
#[tokio::test]
async fn a_geometry_only_mismatch_inserts_only_a_scaler() {
    let seen = Arc::new(Mutex::new(Vec::new()));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("camera", PinnedSource::new(640, 480, PixelFormat::I420, 1));
    let sink = pipeline.add_sink(
        "encoder",
        PinnedSink {
            width: 320,
            height: 240,
            pixel_format: PixelFormat::I420,
            seen: seen.clone(),
        },
    );
    pipeline.link(src, sink).unwrap();

    pipeline.prepare_with_auto_converters().unwrap();

    let inserted = auto_nodes(&pipeline);
    assert_eq!(inserted.len(), 1, "geometry is the only conflict");
    assert!(inserted[0].starts_with("auto_videoscale"));

    pipeline.run().await.unwrap();

    let seen = seen.lock().unwrap();
    assert_eq!(seen[0].0, 320);
    assert_eq!(seen[0].1, 240);
    assert_eq!(seen[0].2, frame_size(320, 240, PixelFormat::I420));
}

/// Two chains in one graph would both have been named `auto_videoscale`, and
/// `nodes_by_name` is last-write-wins — so the earlier node became
/// unaddressable. Names must be unique.
#[tokio::test]
async fn converters_inserted_on_several_links_get_unique_names() {
    let seen_a = Arc::new(Mutex::new(Vec::new()));
    let seen_b = Arc::new(Mutex::new(Vec::new()));

    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source("camera", PinnedSource::new(640, 480, PixelFormat::I420, 1));
    let low = pipeline.add_sink(
        "low",
        PinnedSink {
            width: 320,
            height: 240,
            pixel_format: PixelFormat::I420,
            seen: seen_a.clone(),
        },
    );
    let lower = pipeline.add_sink(
        "lower",
        PinnedSink {
            width: 160,
            height: 120,
            pixel_format: PixelFormat::I420,
            seen: seen_b.clone(),
        },
    );
    pipeline.link(src, low).unwrap();
    pipeline.link(src, lower).unwrap();

    pipeline.prepare_with_auto_converters().unwrap();

    let mut inserted = auto_nodes(&pipeline);
    inserted.sort();
    inserted.dedup();
    assert_eq!(
        inserted.len(),
        2,
        "one scaler per branch, distinctly named: {inserted:?}"
    );
    for name in &inserted {
        assert!(
            pipeline.get_node_id(name).is_some(),
            "{name} must be addressable by name"
        );
    }

    pipeline.run().await.unwrap();
    assert_eq!(seen_a.lock().unwrap()[0].0, 320);
    assert_eq!(seen_b.lock().unwrap()[0].0, 160);
}

/// Under the default Deny policy the mismatch is an error naming the converters
/// that *would* have fixed it, not a silently mis-wired pipeline.
#[test]
fn the_default_policy_refuses_and_explains() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_source(
        "camera",
        PinnedSource::new(1920, 1080, PixelFormat::Rgb24, 1),
    );
    let sink = pipeline.add_sink(
        "encoder",
        PinnedSink {
            width: 1280,
            height: 720,
            pixel_format: PixelFormat::I420,
            seen: Arc::new(Mutex::new(Vec::new())),
        },
    );
    pipeline.link(src, sink).unwrap();

    let error = pipeline.prepare().unwrap_err().to_string();
    assert!(error.contains("videoscale"), "got: {error}");
    assert!(error.contains("videoconvert"), "got: {error}");
}
