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

Notes on scope: negotiation is **per-link** — there is no cross-link constraint propagation.

### Which axes disagree

When two elements do not intersect, the solver works out *which axes* they disagree on rather than reaching for one converter and hoping:

| Axis | Meaning | Fixed by |
|------|---------|----------|
| `FORMAT` | pixel format; sample format + channels | `videoconvert`, `audioconvert` |
| `GEOMETRY` | width, height | `videoscale` |
| `RATE` | framerate; sample rate | `audioresample` |
| `MEMORY` | CPU / GPU / DMA-BUF | `memorycopy` |

`ConvertAxes` is a bitflag set; `diff_caps(source, sink)` reports the conflict. `Any` never conflicts, and `Fixed(1920)` against `Range { 1280..=1920 }` does not either — it fixates to 1920.

`ConverterRegistry::plan()` then covers those axes with the **cheapest chain** of converters. A 1080p RGB camera into a 720p I420 encoder disagrees on two axes and gets two elements — `videoscale ! videoconvert`, in that order, because scaling down first means converting fewer pixels (an upscale converts first instead).

If the registry cannot cover **every** conflicting axis, negotiation **fails**. It never emits a partial chain: that would leave a pipeline running and quietly wrong.

Each converter is handed a `ConversionRequest` describing both ends of the link, so it configures itself for the actual target (a sink that wants I420 gets an I420 converter).

**Honest limitation**: auto-insertion only fires when the *source* side pins the property in question. `AppSrc` and most transforms declare `Any`, and `Any ∩ Fixed = Fixed` never conflicts — so no converter is needed and none is inserted. It fires for `V4l2Src`/`ScreenCaptureSrc`-rooted graphs and for `EncoderElement`, which pin geometry.

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

The `ConverterRegistry` (see `negotiation::builtin_registry()`) holds converter factories keyed by (format type, memory type) — **several per key**, distinguished by the axes they fix and ordered by cost: `audioconvert` (3), `videoconvert` (5), `audioresample` (8), `videoscale` (10), `memorycopy` (20). `identity` fixes no axis, so it can never be auto-inserted to "resolve" a conflict.

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

Supported directions: I420/NV12/YUYV/UYVY → RGB24/RGBA/BGR24/BGRA (UYVY: RGB24/RGBA only); RGB24/RGBA/BGR24/BGRA → I420/NV12; **YUYV/UYVY → I420/NV12** (the path a webcam takes to an encoder); **I420 ↔ NV12**; RGB↔BGR swizzles; alpha add/remove; Gray8 → RGB. Color matrices: BT.601 (default), BT.709.

The YUV→YUV directions carry no colour-space maths — they de-interleave and subsample chroma (4:2:2 → 4:2:0 averages the two source rows) — so they are both cheaper and more accurate than routing through RGB.

**SIMD**: with feature `simd-colorspace`, conversions route through the `yuv` crate (runtime-detected AVX-512/AVX2/SSE4.1/NEON — ~0.9 ms for 1080p I420→RGBA). Without it, a scalar fallback is used. Request aligned buffers via `MemoryLayout::AVX` in caps (arena constructors `SharedArena::new_avx`/`new_avx512`).

```bash
cargo bench --features simd-colorspace --bench colorspace   # SIMD
cargo bench --bench colorspace                              # scalar comparison
```

### Audio

- `AudioConvert` — sample-format conversion (U8/S16/S32/F32/F64, both endiannesses) with `AudioChannelMix`/`ChannelLayout` (mono, stereo, 2.1, 5.1, 7.1).
- `AudioResample` — rate conversion (`ResampleQuality::Fast` = linear, `Medium` = cubic). Pure Rust.

### ScaleEngine

Nearest-neighbor or bilinear resize of raw frames, in any of I420, NV12, RGB24, BGR24, RGBA, BGRA, Gray8, YUYV, UYVY. The `VideoScale` element wraps this engine: it reads the pixel format and geometry from each buffer's metadata and rebuilds its engine when either changes; a target equal to the source is a zero-copy passthrough.

(The YUYV/UYVY paths are low-quality by the engine's own admission — assert structure and size on them, not pixel fidelity.)

## Pitfalls

- **Geometry travels in-band, in `Metadata`.** No element takes dimensions at construction; one that cannot determine its geometry from the buffer **errors** rather than falling back to a stale constructor value. Producers stamp it with `Metadata::set_video_dims(w, h, pixel_format)`, which writes *both* the `MediaFormat::VideoRaw` field and the legacy `"width"`/`"height"` keys — write only one and the other goes stale, silently mis-sizing frames downstream.
- **Two `PixelFormat` enums exist**: `format::PixelFormat` (15 variants, used in caps) and `converters::PixelFormat` (9 variants, used by the conversion engine). They are separate types, but conversion is now explicit and total in one direction: `From<converters::PixelFormat> for format::PixelFormat`, and `TryFrom` back (the caps enum names formats the engine cannot convert). Likewise two `SampleFormat` enums (`format::` vs `converters::audio::`).
- After changing element caps at runtime, call `pipeline.invalidate_negotiation()` (or check `needs_renegotiation()`).
