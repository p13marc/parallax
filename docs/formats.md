# Formats, Caps & Negotiation

How elements describe what they can produce/consume, how the pipeline picks formats and memory types per link, and how conversion works.

## The caps model

Caps ("capabilities") are *constraints*, resolved to concrete formats during `prepare()`.

### Constraint values

```rust
use parallax::format::CapsValue;

CapsValue::Fixed(1920u32)                 // exactly this
CapsValue::Range { min: 320, max: 3840 }  // anything in range
CapsValue::List(vec![30, 60])             // one of these (preference-ordered)
CapsValue::Any                            // unconstrained (default)
```

`intersect()` combines two constraints (Range∩Range → overlap, Fixed∩List → membership, …); `fixate()` collapses to a concrete value (Fixed → itself, Range → min, List → first).

### Format constraints

```rust
use parallax::format::{CapsValue, PixelFormat, VideoFormatCaps};

let caps = VideoFormatCaps {
    width: CapsValue::Fixed(640),
    height: CapsValue::Fixed(480),
    pixel_format: CapsValue::List(vec![PixelFormat::Yuyv, PixelFormat::Rgb24]),
    framerate: CapsValue::Any,
    ..VideoFormatCaps::any()
};
```

- `VideoFormatCaps` — width/height/pixel_format/framerate constraints + `MemoryLayout` (SIMD alignment request: `NONE`/`SSE`/`AVX`/`AVX512`).
- `AudioFormatCaps` — sample_rate/channels/sample_format constraints.
- `FormatCaps` — the union: `VideoRaw(VideoFormatCaps)`, `Video(VideoCodec)`, `AudioRaw(AudioFormatCaps)`, `Audio(AudioCodec)`, `Rtp(..)`, `MpegTs`, `Bytes`, `Any`.

Concrete counterparts (post-fixation): `VideoFormat`, `AudioFormat`, `MediaFormat`, with `PixelFormat` (I420, NV12, P010, YUYV, RGB24, RGBA, BGRA, Gray8, …), `VideoCodec` (H264/H265/VP8/VP9/AV1), `AudioCodec` (Opus/Aac/Mp3/Pcmu/Pcma), `Framerate` (rational, with `FPS_30`-style consts).

### Memory constraints

Formats are coupled with acceptable memory types — this is how zero-copy device paths are negotiated:

```rust
use parallax::format::MemoryCaps;

MemoryCaps::cpu_only()          // plain arena memory
MemoryCaps::dmabuf_only()       // DMA-BUF fds only
MemoryCaps::dmabuf_preferred()  // DMA-BUF if possible, CPU fallback
MemoryCaps::any()
```

### Element capabilities

An element advertises a **preference-ordered list** of format+memory pairs:

```rust
use parallax::format::{ElementMediaCaps, FormatMemoryCap, MemoryCaps};

impl Source for MyCamera {
    fn output_media_caps(&self) -> ElementMediaCaps {
        ElementMediaCaps::new(vec![
            FormatMemoryCap::new(yuyv_caps.into(), MemoryCaps::dmabuf_only()), // preferred
            FormatMemoryCap::new(yuyv_caps.into(), MemoryCaps::cpu_only()),    // fallback
            FormatMemoryCap::new(rgb_caps.into(),  MemoryCaps::cpu_only()),
        ])
    }
}
// Sinks implement input_media_caps() the same way.
```

This mirrors GStreamer's caps-with-features: multiple structures, ordered by preference.

## Negotiation

During `pipeline.prepare()`, the solver (`negotiation::NegotiationSolver`) walks each link:

1. Fetch the source pad's and sink pad's `ElementMediaCaps`.
2. Try `intersect()` — pairwise, in preference order; **first match wins**.
3. On success, `fixate` the intersection into a concrete `MediaFormat` + `MemoryType` and record it on the link (visible via `pipeline.link_format(..)` / `link_memory_type(..)`).
4. On failure, consult the `ConverterPolicy`.

Notes on scope: negotiation is **per-link** (no cross-link constraint propagation), and converter search is **single-hop** (one converter inserted per link; multi-hop pathfinding is a TODO).

### Converter policy

```rust
use parallax::pipeline::ConverterPolicy;

// Default: Deny — prepare() fails with an error listing every attempted
// format combination and suggesting explicit converters.
pipeline.prepare()?;

// One-shot auto-insertion (logs what it inserts):
pipeline.prepare_with_auto_converters()?;

// Or change the policy:
pipeline.set_converter_policy(ConverterPolicy::Allow);
pipeline.prepare()?;
```

`Deny` is the default on purpose (the GStreamer lesson: silent auto-plugging makes pipelines hard to reason about). Explicit is better:

```rust
let p = Pipeline::parse("videotestsrc ! videoconvert ! autovideosink")?;
```

The `ConverterRegistry` (see `negotiation::builtin_registry()`) maps (format, memory) pairs to converter factories with costs: `videoconvert` (5), `audioconvert` (3), `audioresample` (8), `memorycopy` (20), `identity` (0).

### DMA-BUF example

```rust
use parallax::elements::device::{V4l2Config, V4l2Src};

let config = V4l2Config { dmabuf_export: true, ..Default::default() };
let camera = V4l2Src::with_config("/dev/video0", config)?;
// camera's output_media_caps now prefer MemoryType::DmaBuf.
// If the sink accepts DmaBuf → zero-copy fd path is negotiated;
// if the sink is CPU-only → mmap+copy fallback.
```

See `examples/45_dmabuf_negotiation.rs` and `examples/17_multi_format_caps.rs`.

## Converters

The real conversion engines live in `parallax::converters` (the pipeline elements `VideoConvertElement`/`AudioConvertElement`/`AudioResampleElement` wrap them):

### VideoConvert (colorspace)

```rust
use parallax::converters::{PixelFormat, VideoConvert};

let conv = VideoConvert::new(PixelFormat::I420, PixelFormat::Rgba, 1920, 1080)?;
conv.convert(&yuv_in, &mut rgba_out)?;
```

Supported directions: I420/NV12/YUYV/UYVY → RGB24/RGBA/BGR24/BGRA (UYVY: RGB24/RGBA only); RGB24/RGBA/BGR24/BGRA → I420/NV12; RGB↔BGR swizzles; alpha add/remove; Gray8 → RGB. Color matrices: BT.601 (default), BT.709.

**SIMD**: with feature `simd-colorspace`, conversions route through the `yuv` crate (runtime-detected AVX-512/AVX2/SSE4.1/NEON — ~0.9 ms for 1080p I420→RGBA). Without it, a scalar fallback is used. Request aligned buffers via `MemoryLayout::AVX` in caps (arena constructors `SharedArena::new_avx`/`new_avx512`).

```bash
cargo bench --features simd-colorspace --bench colorspace   # SIMD
cargo bench --bench colorspace                              # scalar comparison
```

### Audio

- `AudioConvert` — sample-format conversion (U8/S16/S32/F32/F64, both endiannesses) with `AudioChannelMix`/`ChannelLayout` (mono, stereo, 2.1, 5.1, 7.1).
- `AudioResample` — rate conversion (`ResampleQuality::Fast` = linear, `Medium` = cubic). Pure Rust.

### VideoScale

Nearest-neighbor or bilinear resize of raw frames.

## Pitfalls

- **Two `PixelFormat` enums exist**: `format::PixelFormat` (15 variants, used in caps) and `converters::PixelFormat` (9 variants, used by the conversion engine). Likewise two `SampleFormat` enums (`format::` vs `converters::audio::`). They are separate types — convert explicitly at the boundary.
- **`negotiation::builtin` re-exports stub types** named `VideoConvert`/`AudioConvert`/etc. whose `process()` is passthrough — they exist only to describe converter metadata to the registry. The real engines are in `parallax::converters`, and `builtin_registry()` correctly wires the real element wrappers.
- After changing element caps at runtime, call `pipeline.invalidate_negotiation()` (or check `needs_renegotiation()`).
