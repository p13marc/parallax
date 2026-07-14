# Bandwidth control for the ZenSight ↔ Parallax sensor

Status: filed, not started. Written 2026-07-14.

Tracked by [#33](https://github.com/p13marc/parallax/issues/33) (epic). Phase A is broken out into
[#24](https://github.com/p13marc/parallax/issues/24) (`EncoderControl` handle),
[#25](https://github.com/p13marc/parallax/issues/25) (H.264 live bitrate/GOP/QP),
[#26](https://github.com/p13marc/parallax/issues/26) (H.264 pinned resolution),
[#27](https://github.com/p13marc/parallax/issues/27) (missing openh264 knobs),
[#28](https://github.com/p13marc/parallax/issues/28) (`VideoScale`),
[#29](https://github.com/p13marc/parallax/issues/29) (`VideoConvertElement` latching),
[#30](https://github.com/p13marc/parallax/issues/30) (`Throttle`),
[#31](https://github.com/p13marc/parallax/issues/31) (`JpegEncoder` quality),
[#32](https://github.com/p13marc/parallax/issues/32) (V4L2 M2M live controls).
Phase B (the zensight sensor, wire protocol and GUI) is not filed yet — it lands in the zensight
repo once this API exists.

## Context

ZenSight's `zensight-sensor-parallax` crate streams live video (V4L2 cameras, RTSP, test
patterns) as H.264 + JPEG previews onto the Zenoh `@media` plane. It works, but **every
bandwidth-affecting parameter is frozen at process start**:

* `bitrate_kbps` and `gop_frames` come from `configs/parallax.json5`, are passed to
  `H264EncoderConfig` at construction, and cannot change without restarting the sensor
  (there is no config hot-reload).
* Resolution and framerate for real cameras are **not configurable at all** — the pipeline
  uses whatever the camera negotiated (`src.width()`, `src.height()`, `src.framerate()`).
* The one runtime knob on the wire, `StreamControl::OpenStream { max_height }`, is **inert**:
  the GUI never sends it (both call sites pass `None`), V4L2 ignores it with a
  `warn!("max_height below camera resolution ignored (no rescale element yet)")`
  (`zensight-sensor-parallax/src/pipeline.rs:288`), RTSP ignores it (passthrough), and
  re-opening an already-open stream with a different value silently discards it
  (`session.rs:417`).

The root cause is in Parallax, not ZenSight: **elements are moved out of the graph at
`Executor::start()`** (`Node::take_element()`, `pipeline/graph.rs:255`, called from
`unified_executor.rs:680,923`), so `get_element_mut()` returns `None` on a running pipeline.
The only way to touch a live element is an `Arc<Atomic*>` control handle cloned *before*
start. For codecs, exactly one such handle exists today: `KeyframeHandle`
(`elements/codec/control.rs:44`).

Goal: make bitrate, resolution, framerate, keyframe spacing and JPEG quality **live,
per-stream knobs** driven by Zenoh RPC and the GUI, with a preset ladder (`low`/`medium`/
`high`) on top — no pipeline teardown, no process restart.

---

## Part 1 — What actually controls bandwidth, and what Parallax supports today

### 1.1 The knobs, ranked by leverage

| Knob | Effect on bandwidth | Where it lives today | Live-changeable today? |
|---|---|---|---|
| **Target bitrate** | Direct, ~linear. The dominant knob for H.264. | `H264EncoderConfig::bitrate_bps` (`h264.rs:96`), constructor-only | ❌ |
| **Resolution / scale** | ~Linear in pixel count at constant quality; halving both dimensions ≈ ¼ the pixels and permits a ~2–3× bitrate cut. Also cuts CPU. | Camera-negotiated only; no scaler in the ZenSight graph | ❌ |
| **Framerate** | Near-linear at fixed quality; sub-linear at fixed bitrate (fewer frames simply get more bits each). Best used *together with* a bitrate cut. | `Throttle::rate()` — drop-based, `elements/timing/timeout.rs:260` — used for previews only | ❌ (`min_interval` is a plain `Duration`) |
| **Keyframe spacing (GOP)** | Large on low-motion scenes: an IDR costs 5–20× a P-frame. Longer GOP = less bandwidth, slower late-joiner start. | `H264EncoderConfig::keyframe_interval` (`h264.rs:109`) | ❌ — but **force-keyframe** ✔ via `KeyframeHandle` |
| **QP / quality band** | Quality-for-bytes trade at fixed resolution. Parallax applies `qp ± 4` as a band (`h264.rs:229`). | `H264EncoderConfig::qp` | ❌ |
| **Rate-control mode** | CBR vs VBR vs quality-mode decides how tightly the bitrate target is honoured. | **Not exposed at all** — openh264 falls back to its default quality-mode RC | ❌ |
| **Max NAL / slice size** | Not bandwidth per se, but MTU-friendliness. openh264 supports `max_slice_len`. | **Not exposed** | ❌ |
| **JPEG preview quality + fps** | Previews are a real cost: 2 fps @ q75 at 640×480 is ~50–100 kB/s per viewer. | `preview.quality`, `preview.fps` (config file) | ❌ |
| **Not sending at all** | 100%. ZenSight already does this well ("no viewer, no pixels" + `idle_timeout_secs`). | `session.rs` | ✔ |

### 1.2 What the encoder backends allow at runtime

**OpenH264 (`openh264` crate 0.9.7, feature `h264`).** Two findings make this cheap:

* `Encoder::encode()` **already re-initialises the encoder whenever the input frame
  dimensions change** — it calls `reinit()`, pushes new params through
  `SetOption(ENCODER_OPTION_SVC_ENCODE_PARAM_EXT)` and forces an IDR
  (`openh264-0.9.7/src/encoder.rs:912-914`, `:1028-1041`). Its own doc says so:
  *"The resolution of the encoded frame is allowed to change."* **Parallax is what prevents
  live resolution changes**, by pinning `width`/`height` in `H264EncoderConfig` and rejecting
  mismatched frames (`h264.rs:261-268`, `:444-449`).
* `Encoder::force_intra_frame()` is public and already used. But the crate exposes **no**
  bitrate setter — `Encoder.config` is private. Two ways out:
  * **(a) Re-create the inner `openh264::Encoder`** with an updated `EncoderConfig` when a
    control change lands. Safe, no new dependency, no `unsafe`. Cost: fresh SPS/PPS + an IDR
    (which you want on a bitrate step anyway) and a few ms. Rate-limit changes to ~1/s and it
    is free in practice. **Recommended.**
  * (b) Add `openh264-sys2` as a direct dependency and call
    `raw_api().set_option(ENCODER_OPTION_BITRATE, &mut SBitrateInfo)` (`encoder.rs:75` exposes
    the raw fn pointer). Seamless — no forced IDR — but `unsafe` plus a second dependency.
    Keep as a later optimisation.

**V4L2 M2M (hardware, feature `v4l2-m2m`).** The kernel's
[stateful encoder spec](https://www.kernel.org/doc/html/latest/userspace-api/media/v4l/dev-encoder.html)
says: *"The client is allowed to use `VIDIOC_S_CTRL()` to change encoder parameters at any
time"* — availability is driver-specific, and a driver that refuses returns `-EBUSY`. So
`V4L2_CID_MPEG_VIDEO_BITRATE` and `..._GOP_SIZE` / `..._H264_I_PERIOD` *can* be set live.
Parallax already holds the device as `Arc<Device>` (`v4l2_m2m.rs:196`) and already issues a
live `s_ctrl(FORCE_KEY_FRAME)` (`v4l2_m2m.rs:611`), but `apply_controls()` is private and
called only from `new()` (`v4l2_m2m.rs:356`). Adding live setters is a few lines. Caveat to
document: several drivers (TI, some Pi firmware paths) accept the ioctl but ignore GOP changes
mid-stream — treat GOP as best-effort on hardware.

**rav1e (AV1).** No keyframe-interval knob is exposed at all, and it does not override
`force_keyframe`, so keyframe requests are silently ignored for AV1. Runtime bitrate changes
need a new `rav1e::Context`. Out of scope for ZenSight (H.264 only), but the trait should
degrade gracefully.

**Opus.** libopus permits `set_bitrate` mid-stream; Parallax sets it once in
`OpusEncoder::new()` (`opus.rs:161`). Easy win, not needed by ZenSight today.

### 1.3 Structural gaps in Parallax that block this

1. **No runtime control handle for anything but keyframes.** The idiom exists and is proven —
   `KeyframeHandle` (`Arc<AtomicBool>`), `ValveControl` (`elements/flow/valve.rs:81`),
   `FlowStateHandle` (`pipeline/flow.rs:327`) — it just needs generalising to encoder params.
2. **No usable scaler in the pipeline path.** There are *three* `VideoScale` types: the real
   one in `converters/scale.rs` (a plain struct, not an Element); a second real one in
   `elements/transform/scale.rs` (implements `Element`, YUV420-only, fixed src *and* dst
   dims); and a **passthrough stub** in `negotiation/builtin.rs:62` that isn't even registered
   in `builtin_registry()`. Auto-negotiation can therefore never insert a scaler, and no
   scaler can be resized at runtime.
3. **`VideoConvertElement` latches its dimensions on the first buffer**
   (`videoconvert.rs:126-127`) and can never re-negotiate — it will break the moment a live
   resolution change flows through it.
4. **Framerate limiting exists but is not tunable.** `Throttle` is drop-based and correct for
   live sources (ZenSight already uses it for previews); its `min_interval: Duration` is
   simply not atomic.
5. **QoS is dead weight.** `Event::Qos` is defined but never emitted in-band and never handled
   by any element (`grep Event::Qos` → zero hits outside the enum). Not a usable feedback path.
6. **Probes only fire on source src-pads** (`unified_executor.rs:1164` is the only
   `invoke_buffer` call site); `invoke_event`/`invoke_idle`/`BufferMut` are never invoked. So
   probes are not a general control channel either.

### 1.4 Consequences for ZenSight's three pipeline shapes

* **Test pattern** and **V4L2**
  (`VideoTestSrc|V4l2Src → [JpegDecoder] → VideoConvert → H264Encoder → AppSink`): every knob
  becomes live once Parallax grows the handles and a resizable scaler. This is the main case.
* **RTSP video** is `AppSrc → AppSink` **passthrough** — no encoder, so bitrate/GOP/resolution
  simply do not apply. The honest options are (a) leave as-is and document it, or (b) offer an
  opt-in transcode path (`H264Decoder → VideoScale → H264Encoder`) at real CPU cost.
  Recommend (a) now, with a `transcode: bool` opt-in later.
* **Previews** (JPEG): `quality` and `fps` should become live knobs too — same handle pattern
  (a `Throttle` handle plus an `Arc<AtomicU8>` on `JpegEncoder`).

---

## Part 2 — Implementation plan

### Phase A — Parallax: runtime control primitives (ships as 0.2.0)

**A1. Generalise the control handle** — `src/elements/codec/control.rs`

Extend the existing module; keep `KeyframeHandle` as a re-export for back-compat.

```rust
/// Cloneable, lock-free handle for changing encoder parameters on a running pipeline.
/// Clone it *before* `executor.start()`.
pub struct EncoderControl(Arc<EncoderControlInner>);

struct EncoderControlInner {
    keyframe: AtomicBool,            // existing semantics
    bitrate_bps: AtomicU32,          // 0 = unchanged
    keyframe_interval: AtomicU32,    // u32::MAX = unchanged
    qp: AtomicU8,                    // u8::MAX = unchanged
    generation: AtomicU64,           // bumped on any set; encoder applies only when it moves
}
```

`generation` matters: it lets `process()` do a single relaxed load per frame in the common
case and rebuild the encoder only when something actually changed.

**A2. Widen the `VideoEncoder` trait** — `src/elements/codec/traits.rs:74-113`

Add defaulted methods (no-op or `Err(Unsupported)`) next to the existing `force_keyframe()`:
`set_bitrate(&mut self, bps: u32)`, `set_keyframe_interval(&mut self, frames: u32)`,
`set_qp(&mut self, qp: u8)`. Defaults keep every current impl compiling.

**A3. `H264Encoder`** — `src/elements/codec/h264.rs`

* `control_handle() -> EncoderControl` alongside the existing `keyframe_handle()`.
* At the top of `Element::process()` (`h264.rs:352`), if `generation` moved, rebuild the inner
  `openh264::Encoder` from an updated `EncoderConfig` (option (a) above) and force an IDR.
* **Unpin the resolution**: read width/height from `buffer.metadata().format`
  (`MediaFormat::VideoRaw`) per frame instead of `self.config.{width,height}`, and drop the
  hard mismatch check in `VideoEncoder::encode` (`h264.rs:444`). openh264 handles the reinit +
  IDR itself. Keep the config dims as the fallback when metadata carries no format.
* Expose the openh264 knobs that exist but are unused: `rate_control_mode` (CBR/VBR/quality),
  `max_slice_len` (MTU-sized NALs), `profile`, `level`, `usage_type`, `complexity`.

**A4. Live V4L2 M2M controls** — `src/elements/codec/v4l2_m2m.rs`

Split `apply_controls()` into `set_bitrate()` / `set_gop()` against the `Arc<Device>`,
implement the new `VideoEncoder` methods with them, and tolerate `-EBUSY` / silently-ignoring
drivers (log once, do not fail the pipeline).

**A5. Resizable scaler** — `src/elements/transform/scale.rs`

* Give `VideoScale` a `ScaleControl` handle (packed dst w/h in an `Arc<AtomicU64>` plus a
  generation counter) and a `control()` accessor.
* Derive **source** dims from `Metadata.format` per buffer (today they are constructor-pinned),
  rebuild the internal `converters::VideoScale` when either side changes, and **stamp the
  output `Metadata.format`** with the new dims — that is what lets the downstream encoder see
  the change.
* Fix `VideoConvertElement::ensure_converter` (`videoconvert.rs:126`) to rebuild when the input
  dims change instead of latching forever, and to propagate `Metadata.format`.

**A6. Tunable `Throttle`** — `src/elements/timing/timeout.rs:260`

`min_interval: Duration` → `Arc<AtomicU64>` (ns), plus
`Throttle::control() -> ThrottleControl { set_rate(fps) }`. Same shape as `ValveControl`.

**A7. `JpegEncoder` quality handle** — `src/elements/codec/image.rs:248` — `Arc<AtomicU8>` plus
`quality_control()`.

**A8. Tests** — grow `tests/keyframe_request.rs` into `tests/encoder_control.rs`. On a
*running* pipeline: (1) drop the bitrate and assert bytes/frame falls; (2) change the scaler's
dst size and assert decoded frames change dimensions and that the first frame after the change
is an IDR; (3) halve the throttle rate and assert the frame count halves. Feature-gate under
`h264` — note `elements::codec` only compiles when a codec feature is on, and CI's sensor combo
covers it.

### Phase B — ZenSight sensor: plumb the knobs

**B1. Pipeline shape** — `zensight-sensor-parallax/src/pipeline.rs`

```
V4l2Src → [JpegDecoder] → VideoConvert(→I420) → VideoScale → Throttle → H264Encoder → AppSink
```

Return a `PipelineControls` struct whose handles are all cloned *before* `executor.start()`,
exactly as `keyframe_handle()` is today (`pipeline.rs:240`):

```rust
pub struct PipelineControls {
    encoder: Option<EncoderControl>,   // None for RTSP passthrough
    scale:   Option<ScaleControl>,
    rate:    Option<ThrottleControl>,
    preview_quality: Option<Arc<AtomicU8>>,
    preview_rate:    Option<ThrottleControl>,
}
```

This also **closes the `max_height` hole**: it becomes a real `ScaleControl` target instead of
a `warn!` (`pipeline.rs:288-293`).

**B2. Config** — `src/config.rs` + `configs/parallax.json5`

Add `video.{width, height, fps, bitrate_kbps, gop_frames, rate_control}` and a **preset ladder**:

```json5
parallax: {
  video: { bitrate_kbps: 2000, gop_frames: 60, fps: 30, max_height: null },
  presets: {
    low:    { max_height: 240, fps: 10, bitrate_kbps: 400  },
    medium: { max_height: 480, fps: 20, bitrate_kbps: 1200 },
    high:   { max_height: 720, fps: 30, bitrate_kbps: 4000 },
  },
  default_preset: "medium",
}
```

**B3. Wire protocol** — `zensight-common/src/stream.rs`

Additive variants. Bump the schema version: an old decoder rejects an unknown CBOR variant.

```rust
enum StreamControl {
    OpenStream { stream, codec, max_height, preset: Option<String> },  // preset added
    CloseStream { stream },
    RequestKeyframe { stream },
    SetVideoParams {                       // new — applied live, no rebuild
        stream,
        preset: Option<String>,            // shorthand for the knobs below
        bitrate_kbps: Option<u32>,
        fps: Option<f32>,
        max_height: Option<u32>,
        gop_frames: Option<u32>,
        preview_quality: Option<u8>,
        preview_fps: Option<f32>,
    },
}
```

**B4. Session** — `src/session.rs`

Store `PipelineControls` plus the currently-applied params on `ProfileSession`; handle
`SetVideoParams` by writing to the handles, with no teardown. Fix the reuse check
(`session.rs:417`) so a re-`open_stream` with different params **applies** them rather than
discarding them. Report the applied values back in `StreamStatus` so the GUI renders actual
state, not requested state. Rate-limit applications (≥1 s apart) so a slider drag does not
rebuild the encoder 60×/s.

**B5. GUI** — `zensight/src/view/settings.rs` + `src/app.rs:5293,5413`

A per-stream video panel: preset selector (low/medium/high/custom) plus sliders for
bitrate/fps/max-height, sending `SetVideoParams`. Today both `OpenStream` call sites hard-code
`max_height: None`; they should send the selected preset.

**B6. Document the RTSP limitation** (`docs/streams.md` already flags it) and leave transcoding
as an explicit future opt-in.

### Deliberately out of scope

Automatic ABR. Zenoh runs over TCP/QUIC, which hides packet loss behind retransmission, so
there is no RTCP-style loss/jitter signal to close a loop on. The signals that *do* exist —
`kbps`, `drops`, `encode_ms` in `src/stats.rs`, and the `encoder_overrun` alert — describe the
*sender*, not the path. The preset ladder gives viewers a deliberate, predictable lever
instead. A real ABR loop should wait for a transport that reports congestion (or for the RTP
path, where Parallax's `RtcpHandler` could supply receiver reports).

---

## Dependency / release note

ZenSight consumes `parallax-pipeline = "0.1.3"` **from crates.io**, not a path dep. Phase A
therefore ships as a Parallax **0.2.0** release: the `VideoEncoder` trait gains defaulted
methods (source-compatible), but `H264Encoder`'s resolution behaviour changes, which is
semver-visible. During development, point ZenSight at a `path = "../parallax"` override; cut
the release before merging Phase B.

## Verification

1. **Parallax**: `just check` (fmt + clippy + tests), then
   `cargo nextest run --features "h264,v4l2,image-jpeg"` for the new `tests/encoder_control.rs`,
   and `just check-sensor` to mirror CI's sensor combo.
2. **Encoder, end-to-end**: run a 300-frame `videotestsrc → scale → throttle → h264enc`
   pipeline, flip bitrate 4 Mbps → 400 kbps at frame 100 and max-height 720 → 240 at frame 200,
   dump the Annex-B stream, and assert with `ffprobe -show_frames` that (a) bytes/frame drops
   ~10×, (b) the resolution changes at the expected frame, (c) each change is preceded by an IDR.
3. **ZenSight, live**: run the sensor against a real UVC camera, subscribe with the GUI, drive
   `SetVideoParams` over Zenoh RPC (`zensight/v1/<origin>/@rpc/parallax/stream/set`), and confirm
   from the stats plane that `kbps` tracks the requested bitrate within ~20% and `fps` matches,
   with no stream interruption (the H.264 tile in `view/specialized/parallax_h264.rs` must not
   drop to black).
4. **Hardware path** (optional): repeat with `v4l2-m2m` to see which controls the driver honours
   live; log and document the ones it ignores.
