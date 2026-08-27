# A real video player: Parallax on Iced

Status: **spine done**, on `feat/iced-player`. Written 2026-08-27, revised the
same day after building it.

A1 ([#263](https://git.marcpardo.eu/marcpardo/parallax/issues/263)),
A2 ([#264](https://git.marcpardo.eu/marcpardo/parallax/issues/264)),
B1 ([#267](https://git.marcpardo.eu/marcpardo/parallax/issues/267)),
B2 ([#268](https://git.marcpardo.eu/marcpardo/parallax/issues/268)),
B4 ([#270](https://git.marcpardo.eu/marcpardo/parallax/issues/270)),
B5 ([#271](https://git.marcpardo.eu/marcpardo/parallax/issues/271)),
C1 ([#272](https://git.marcpardo.eu/marcpardo/parallax/issues/272)) and
C2 ([#273](https://git.marcpardo.eu/marcpardo/parallax/issues/273)) are done:
`parallax-player` opens a file, plays it in sync, seeks, and gets its controls
out of the way. What the build taught that this plan did not know is in
§ "What building it changed" at the end.

Tracked by [#286](https://git.marcpardo.eu/marcpardo/parallax/issues/286) (**epic**),
milestone *A real video player: Parallax on Iced* —
23 issues across four phases.

> **Numbering note.** `CLAUDE.md` and `CHANGELOG.md` cite `#265` about a dozen times
> meaning the A/V-sync-on-seek work (merge commit `024858d`, branch
> `fix/av-sync-on-seek`). That number was never a real issue or PR on this forge — the
> highest PR is #261 and the counter stood at 262 — so the citation was always dangling,
> and the counter has now handed 265 to this milestone's A3. Read those citations as
> pointing at the commit.

`parallax-player` is a 1169-line demo: one `main.rs`, a crossterm status line redrawn
in place, and an `AutoVideoSink` that opens its own window. It exists to prove the
engine, not to be used. Now that presentation goes through wgpu (#190) and the A/V
clock has one shared, re-anchorable origin (#265), the expensive parts of a player —
decode, sync, seek, pacing — are already engine features. What is missing is a front
end. This plan replaces the demo with a real player on **Iced 0.14**, and closes the
engine gaps that a real player is the first thing to expose.

## Two facts that shape everything

**1. `AutoVideoSink` and Iced cannot coexist.** The sink builds the process's one winit
`EventLoop` on a spawned thread (`src/elements/app/autovideosink.rs:1329-1332`, via the
X11-only `EventLoopBuilderExtX11`). An Iced application owns that event loop, and there
is no external-window-handle API anywhere in the tree. So the Iced player does not use
`AutoVideoSink` at all — it needs a different sink, and the presentation maths has to
be reachable from outside the crate.

**2. Zero-copy dma-buf import is unavailable in any released Iced, and no architecture
choice changes that.** `wgpu-hal`'s `vulkan::Device::texture_from_dmabuf_fd` and the
`VULKAN_EXTERNAL_MEMORY_DMA_BUF` feature landed in **wgpu 30.0.0** (2026-07-01,
[wgpu#9412]). Iced 0.14 is on wgpu 27.0.1, Iced master (0.15-dev) on wgpu 29 with a
winit fork; crates.io has nothing newer than 0.14. Separately, `iced_wgpu`'s compositor
requests `required_features: wgpu::Features::empty()` and neither `iced_core::Settings`
nor `iced_graphics::Settings` can change that. So the binding constraint is **the wgpu
major version, not the feature bitfield** — and "integration mode" does not escape it
either, because `iced_wgpu::Engine::new` takes *Iced's* `wgpu::Device`.

The port cost was measured rather than guessed: forcing Iced 0.14 onto wgpu 30 leaves
**54 errors in `iced_wgpu`**, every one mechanical (`multiview_mask`,
`apply_limit_buckets`, `color_space` added; `multiview`, `push_constant_ranges`
removed; `Option<>`-wrapping; argument counts; `CurrentSurfaceTexture` replacing
`Result<_, SurfaceError>`), plus `cryoglyph` — Iced's 1222-line glyphon fork — needing
the same bump.

So v1 ships the **upload path** and import arrives in v2, either by forking those two
crates or by Iced reaching wgpu 30 on its own. That is a bounded regression, not a
broken feature: `VaapiDecoder` offers `MemoryCaps::dmabuf_preferred()`
(`src/elements/codec/vaapi.rs:717`) and falls back to the `VaFrame::read_plane`
de-tiling copy (`vaapi.rs:464-489`) for a CPU-only consumer, so gotcha 11c costs us the
copy, not the capability. The measured 1080p ladder is software 0.54 → hardware+upload
0.24 → hardware+import 0.12 cores; v1 lands on the middle rung and v2 recovers the
bottom one.

[wgpu#9412]: https://github.com/gfx-rs/wgpu/pull/9412

## Decisions

| Question | Answer | Why |
|---|---|---|
| Iced integration shape | Stock `iced::application` + the `shader` widget | Keeps the whole runtime: `Task`, `Subscription`, `daemon()`/multi-window, and — decisively — `RedrawRequest::At`, which is what makes precise presentation timing reachable at all. |
| Zero-copy | v2, behind a feature, written but off | Blocked on wgpu 30 reaching `iced_wgpu`. The whole video path is identical either way; only one `if` in `VideoPipeline::prepare` changes, so staging costs nothing. |
| Integration mode (we own wgpu+winit) | **Rejected** | Pays the entire cost — no `Task`, no `Subscription`, no multi-window, and you re-implement Iced's `ControlFlow::WaitUntil` redraw scheduler by hand — and buys nothing, because the device is still wgpu 27. |
| Newtype `Renderer` + custom `Compositor` | **Rejected as dominated** | Cheaper than it looks (`iced::application` is already generic over the renderer), but on wgpu 27 the only thing it can buy is a feature bit that does not exist. On the day we need it we are already on a patched Iced, where the same thing is a five-line diff. |
| Where the glue lives | New `parallax-iced` workspace crate | The widget must compile against *Iced's* wgpu; it can share no types with `display-gpu` (wgpu 30). Publishable later as the way anyone embeds parallax video in Iced. |
| Where the sink lives | `parallax` core, **no feature gate** | It needs only `tokio::sync` and `std`. Being unfeatured means it and `parallax::video` compile and test under a bare `cargo test` — which dodges gotcha 14, where colocated tests in gated modules never run in the default job. |
| `AutoVideoSink` / `display-gpu` | Keep both | Not just deferred: `parallax-launch` and examples 23/24/58/59 need an in-crate display path, and `display-gpu` is the only in-tree consumer that exercises `plane_imports` against a real Vulkan device. The endgame is deduplication, not deletion. |
| Pacing split | Sink owns the anchor gate and backpressure; the widget picks the frame due at the next redraw | `SyncHandle` is `Clone` and `Clock` is `Send + Sync`, so `pipeline::schedule` — a public pure function — can run on the GUI thread, where the vblank is. |

## The shape

```
       parallax (core, unfeatured)                parallax-iced             parallax-player
  ┌──────────────────────────────────┐      ┌────────────────────┐      ┌──────────────────┐
  │ demux → decode → AppVideoSink ───┼─────▶│ VideoView (shader::│◀─────┤ iced::application│
  │                    │             │ ring │   Program)         │ view │  stack![ video,  │
  │         → audiodec → AlsaSink    │      │ VideoPrimitive     │      │    controls ]    │
  │                     (clock 150)  │      │ VideoPipeline      │      │ Engine           │
  │ parallax::video (pure maths) ────┼─────▶│  ├ upload   (v1)   │      │  open/close/next │
  │  VideoFrame · PacedFrame         │ WGSL │  └ dmabuf   (v2)   │      └──────────────────┘
  │  FrameMailbox · PresentationClock│      └────────────────────┘
  │  layout · fit · color · WGSL     │
  └──────────────────────────────────┘
```

The `Buffer` never leaves shared memory. The sink hands the widget a `Buffer` clone
(two atomic increments, no syscall), `Primitive::prepare` does one `write_texture` per
plane straight from the arena, and `Primitive::draw` renders into the render pass Iced
already opened. In steady state there is **zero Iced message traffic per frame**: the
widget holds the `Arc<FrameMailbox>` and reads it in `update`/`draw`.

## Remaining gaps

From a sweep of the engine against what a desktop player needs. Ranked by how early the
player trips over them.

| Gap | State | Evidence |
|---|---|---|
| **sRGB double-encoding** | Iced picks an sRGB surface format (`iced_graphics::color::GAMMA_CORRECTION == true`). `present.wgsl` writes gamma-encoded R'G'B' to a non-sRGB `Bgra8Unorm` target today. Ported unchanged it will look washed out. | `iced_wgpu/src/window/compositor.rs:108-112` |
| **Colorimetry is not in metadata** | The GPU path picks BT.709 by a `height >= 720` heuristic; the CPU path is **always BT.601** and `VideoConvert::with_color_matrix` has no caller. Same file, two pictures. | `present/color.rs:22`, `converters/colorspace.rs:222-229`, `transform/videoconvert.rs:189` |
| **Limited range hardcoded** | `YuvRange::Limited` is a literal at every SIMD call site; full-range video renders crushed and cannot be described. | `converters/colorspace.rs:753…1270` |
| **No pixel aspect ratio** | Not in `VideoFormat`, `CodedVideoFormat`, `Metadata` or caps. Anamorphic content renders squished. | `src/format.rs:330-339, 725-734` |
| **No rotation** | MP4 `tkhd` is read only for the track id; the display matrix is skipped. Portrait phone video renders sideways. | `src/elements/demux/mp4_extra.rs:139-144` |
| **Rate ≠ 1.0 desyncs** | `INSTANT_RATE_CHANGE` is handled only by `AutoVideoSink`. `AlsaSink` — the clock master — ignores it, so the picture speeds up and the audio does not. Trick-play seeks drop audio entirely. | `event/mod.rs:799-809`, `mp4.rs:1382-1394` |
| **No track selection** | First video / first audio, hardcoded; `route()` is private; no runtime switch. `Mp4Track`/`MkvTrack` carry no language, title or default flag, so a picker has nothing to label rows with. | `mp4.rs:785-793`, `mkv.rs:448-470` |
| **`StreamCollection` / tags never posted** | The bus variants and `TagList` exist; nothing produces them. A media-info panel has no source. | `pipeline/bus.rs:114-117, 445-455, 699` |
| **Subtitles: nothing** | Recognised as a track *type* and then dropped; no SRT/ASS/WebVTT/CEA-608 parser; no overlay element anywhere in `src/`. | `mkv.rs:973-975`, `mp4_extra.rs:5` |
| **No frame stepping** | No `Step` event; `SeekFlags` bits 6-8 are reserved for a trickmode family that does not exist. | `event/mod.rs:60-135, 774-812` |
| **No out-of-band frame decode** | No thumbnailer; seek preview needs a whole second pipeline. | — |
| **HEVC not built** | `cros-codecs`' `h265` does not compile against its release; `vaapih265dec` is permanently unavailable. Files open, then fail. | `Cargo.toml:175-186`, `factory/mod.rs:79-85` (#200, #202) |
| **MP3/FLAC unusable** | `SymphoniaDecoder` re-probes a whole container per call — not a pipeline element. MKV identifies MP3 tracks it cannot decode. | `codec/audio.rs:110-118` |
| **Network playback is TS-only** | `Mp4Demux`/`MkvDemux` need an owning `Read + Seek`; no HTTP adapter exists. No HLS or DASH *source*. | `mp4.rs:535`, `mkv.rs:281` |
| **PipeWire output is a stub** | The device parameter is `_device` and unused; no format negotiation, no `ClockProvider`, no `latency()`. | `device/pipewire.rs:510-605` |
| **10-bit / HDR** | Enum variants only. VA-API hard-errors on non-NV12; converters take 9 of 15 formats; no shader branch. | `codec/vaapi.rs:360-369`, `converters/colorspace.rs:200-218` |

Reusable as-is, and load-bearing here: the whole `pipeline::sync` machinery
(`AnchorGate`, `Anchor`, `Pace`, `schedule`, `StallWatch`, `SyncHandle`,
`clip_to_segment` — all public, all headlessly tested in `tests/av_sync_tests.rs`), the
`PipelineHandle` transport surface, `AppSink`, `letterbox_rect`, `present.wgsl`,
`color.rs`, `AlsaSink::enumerate_devices`, and the player's seek-coalescing state
machine.

## Work items

Sizes: S ≈ a day, M ≈ 2-3 days, L ≈ a week, XL ≈ more.

### Phase A — core: a reusable presentation surface and a GUI seam

**A1 ([#263](https://git.marcpardo.eu/marcpardo/parallax/issues/263)). `parallax::video` — presentation maths, without wgpu (M).**
New `src/video/`, always compiled. Moves in, tests and all: `present::color`
(`ColorMatrix`, `coefficients`, `yuv_to_rgb`, the golden tests) → `video/color.rs`;
`letterbox_rect`, `yuv420_len` → `video/fit.rs`; `plane_uploads`/`PlaneUpload` and
`plane_imports`/`PlaneImport` → `video/layout.rs`. The load-bearing change there is
**`TexelFormat { R8, Rg8, Rgba8, Bgra8 }`** replacing `wgpu::TextureFormat` — that is
what lets one plane-geometry implementation serve wgpu 27 and wgpu 30, each backend
supplying a four-arm mapping. Adds `FrameView::from_buffer(&Buffer)`, the single place
that reads `video_pixel_format`/`video_dims`/`plane_layout` and validates the length,
today duplicated between `AutoVideoSink::consume` and `CpuBackend`. Exports
`pub const PRESENT_WGSL: &str` — plain WGSL, no extensions, one bind group, so it
compiles under naga 27 and naga 30 alike; it is the *only* thing shared across the two
wgpu worlds, and it works because it is a `&'static str`.
`present/mod.rs`, `present/wgpu_backend.rs` and `present/dmabuf_import.rs` are
re-pointed at it (net −170 lines). Side benefit worth having on its own: the golden
tests enter the **default** `cargo test` job for the first time — today they run only
under `--features display`.

**A2 ([#264](https://git.marcpardo.eu/marcpardo/parallax/issues/264)). `AppVideoSink` — a paced, toolkit-agnostic sink (L).**
`src/elements/app/appvideosink.rs`, beside `AppSrc`/`AppSink`, unfeatured. An
`AsyncSink` whose `consume` mirrors `AutoVideoSink::consume`'s front half — `FrameView`,
`clips_out_of_segment`, `AnchorGate::settle` once per flush generation — and then
computes the frame's `target` running time and enqueues a `PacedFrame`. Per-frame
*timing* is the host's.

```rust
pub struct PacedFrame { frame: VideoFrame, target: ClockTime,
                        generation: u64, must_present: bool }
pub struct FrameMailbox { /* bounded ring + Notify + stats atomics */ }
impl FrameMailbox {
    fn id(&self) -> u64;                       // stable; keys the Subscription recipe
    fn clock(&self) -> Option<PresentationClock>;   // Arc<dyn Clock> + SyncHandle
    fn take_due(&self, now: ClockTime, max_lateness: Duration) -> Option<PacedFrame>;
    fn next_due(&self) -> Option<ClockTime>;   // drives RedrawRequest::At
    fn doorbell(&self) -> &Notify;             // rung only on empty -> non-empty
    fn stats(&self) -> FrameStats;
}
```

The ring is **bounded, drop-oldest — not block**. That is the same policy
`AutoVideoSink` already has (`try_send` → `Err(Full)` → shed, `autovideosink.rs:930`)
and it matters more here: the drain is the GUI thread, and an occluded or minimised
window stops receiving `RedrawRequested`. Blocking would fill the video link,
back-pressure the demuxer, starve the audio branch, underrun the device and freeze the
master clock — the exact starvation `StallWatch` exists to break. Dropping degrades
video and leaves audio playing, which is what a player should do.
Depth is a parameter (`with_queue_depth`), so depth 1 *is* the publish-when-due model
and depth 3 is the select-at-redraw model — a one-line revert, not a rewrite.
Also: `retained_buffers()` = ring capacity + host-declared retention (injected, because
only the host knows); `latency()` must now include the queue's held time;
`FrameAcceptance { formats, memory }` is injected too, because only the host knows what
its GPU can take; `QosAggregator` is promoted from `autovideosink.rs` into
`pipeline::sync` and shared. Ships with `examples/64_appvideosink.rs`, a headless host
that pulls and checksums frames — so the seam has a non-Iced consumer in-tree, in the
default CI job.
Deliberately *not* here: rebuilding `AutoVideoSink` on top of this. The two differ in
the one place that matters (wait-then-send vs compute-then-enqueue) and every other
line is already a call into `sync.rs`; a shared skeleton would buy a trait object and
cost the ability to read either one.

**A3 ([#265](https://git.marcpardo.eu/marcpardo/parallax/issues/265)). Colorimetry in `Metadata` (M).** `ColorInfo { matrix, range, primaries, transfer }`
on `VideoFormat`, written by the demuxers (MKV `Colour`, MP4 `colr`) and the decoders
(H.264 VUI, VP9 header, AV1 sequence header); the height heuristic survives only as the
fallback when nothing signalled. `VideoConvert` finally calls `with_color_matrix`; full
range reaches the SIMD **and** scalar arms (both, per the twin rule in CLAUDE.md);
`present.wgsl` and the widget select from `ColorInfo`. Closes the "same file, two
pictures" bug.

**A4 ([#266](https://git.marcpardo.eu/marcpardo/parallax/issues/266)). Pixel aspect ratio and rotation (M).** PAR on `VideoFormat`/`CodedVideoFormat`,
from MP4 `pasp` + the `tkhd` matrix and MKV `DisplayWidth`/`DisplayHeight` + rotation
tag. `letterbox_rect` takes PAR; rotation is applied in the shader, and in `CpuBackend`'s
blit or refused there with a warning.

### Phase B — `parallax-iced`

**B1 ([#267](https://git.marcpardo.eu/marcpardo/parallax/issues/267)). Crate skeleton and the `VideoView` widget, upload path (L).**
New workspace member, `publish = false`. Follow the convention the two sibling repos
already set: `iced = { version = "0.14", features = ["tokio", …] }` and
`iced_test = "0.14"` (`zenkey/Cargo.toml:83-84`, `zensight/zensight/Cargo.toml:41-42`).
The `tokio` feature is not optional — it is what lets a parallax stream live in an
`iced::Subscription`, and what makes `update()` run inside `runtime.enter()`.
`VideoView: shader::Program` → `VideoPrimitive` → `VideoPipeline`. Textures unchanged
from `WgpuBackend::ensure_textures`: I420 → 3× `R8Unorm`, NV12 → `R8Unorm` + `Rg8Unorm`,
RGBA/BGRA → one, always three bindings with 1×1 dummies padding the unused ones so the
bind-group layout is fixed. Uploads are `write_texture` with `bytes_per_row` from
`PlaneDesc`, which is what makes dav1d's strided `External` frames free.
`VideoPipeline` keys its texture sets by `video_id` in an `FxHashMap`, because Iced
caches **one** pipeline per `TypeId` for the whole process; `trim()` evicts sets
untouched for ~120 frames. Iced's device clamps `max_bind_groups: 2` and
`max_non_sampler_bindings: 2048` — we need 1 and 5, so there is room, but the
downlevel fallback tier is worth testing at 4K.
**Letterbox is a uniform, not `set_viewport`.** Iced pre-sets the viewport to the
widget bounds and the scissor to the clip rect, but its load op is `LoadOp::Load`, not
`Clear` — so narrowing the viewport would leave whatever was underneath in the bars,
and `WgpuBackend`'s "the black clear *is* the bars" has no analogue. Draw the fullscreen
triangle over the whole widget and put the fit rect in `Params`, with the fragment
shader emitting black outside it. `WgpuBackend` passes `fit = (0,0,1,1)` and keeps its
`set_viewport`, so its golden behaviour is byte-identical.
Retention on the upload path is trivial — `write_texture` copies at call time — so the
host declares 1 (the frame the widget holds as displayed).

**B2 ([#268](https://git.marcpardo.eu/marcpardo/parallax/issues/268)). Frame delivery and redraw scheduling (M).**
`Program::update` on `Event::Window(RedrawRequested(now))` computes running time itself
via `PresentationClock`, calls `mailbox.take_due(now, max_lateness)`, stores the frame
in `State`, runs `StallWatch`, and returns
`Action::request_redraw_at(now + (next_due - current_running_time))` — a duration delta,
so no `Instant`↔`ClockTime` epoch mapping is needed. That becomes
`shell.request_redraw_at(..)` → `ControlFlow::WaitUntil(..)`, so the widget genuinely
tells Iced when to wake. Verified: `iced_winit/src/lib.rs:815-830` dispatches
`RedrawRequested(Instant::now())` through `UserInterface::update` to every widget before
drawing.
Two constraints found in the source and easy to get wrong. The frame-selection path must
**not** publish a `Message` from that `update` — the same loop rebuilds the interface
when a redraw produces messages, up to three times, then warns. And the doorbell
`Subscription` must be built with `Subscription::run_with`, whose data must be `Hash`
and whose builder must be a non-capturing `fn`: hash **only the mailbox id**, so
rebuilding the engine swaps the recipe and cross-talk between generations is not
representable. Everything else — position, duration, drops, bus — is polled on a 20 ms
`iced::time::every` tick, a direct port of today's `tokio::select!` arm.
When `mailbox.is_paused()`, return `RedrawRequest::Wait` and park: the last frame stays
on screen and resume always arrives through a user action, which redraws anyway.

**B3 ([#269](https://git.marcpardo.eu/marcpardo/parallax/issues/269)). Zero-copy dma-buf import (L, blocked on wgpu 30).**
Behind a `dmabuf-import` feature, written against `iced::widget::shader::wgpu`.
Requires forks of `cryoglyph` and `iced_wgpu` bumped to wgpu 30 on git.marcpardo.eu
(the pattern already used for `cros-codecs-extended`), pulled via a workspace
`[patch.crates-io]` block — workspace-local, which is the right scope because
`parallax-pipeline` gains **no** Iced dependency, so nothing reaches a crates.io
consumer. Plus a one-line device-features change in the compositor,
`required_features |= adapter.features() & (VULKAN_EXTERNAL_MEMORY_DMA_BUF |
VULKAN_EXTERNAL_MEMORY_FD)`, which should go upstream as a PR alongside the wgpu bump
Iced has to do anyway.
**Retention is the interesting part**, because Iced owns `queue.submit` and the exact
`SubmissionIndex` that `WgpuBackend` waits on is unavailable. The answer is
`Queue::on_submitted_work_done`, deferred by one frame: at the top of `prepare`, drive
callbacks with a non-blocking `device.poll(PollType::Poll)`, then hand *last* frame's
retained buffers to `on_submitted_work_done`. Iced's loop is strictly sequential on the
GUI thread — `prepare`(N) → `draw`(N) → `present` → `submit` → `prepare`(N+1) — so that
callback is fenced behind the submission that sampled those textures. One poll and one
registration per frame, no CPU/GPU sync, ≤2 frames retained in steady state with a hard
cap of 4. Plan B if a driver misbehaves: submit our own `copy_texture_to_texture` from
`prepare` and fence on its own index — ~20 lines and one on-GPU 3.1 MB copy. What we do
*not* do is hold for K frames without a fence: its failure mode is the decoder writing
into memory the GPU is sampling, which is silent corruption reported as "sometimes
tears on my machine".
Also: the probe that decides `FrameAcceptance` must **create the real device** and cache
it for the compositor to reuse, so probe and reality are the same object. Otherwise the
sink can advertise `MemoryType::DmaBuf` on one adapter while Iced renders on another —
and a Y-tiled VA-API surface has no CPU fallback.

**B4 ([#270](https://git.marcpardo.eu/marcpardo/parallax/issues/270)). Measure it (M).** Do the harness *before* B1 is finished, because one number can
still invalidate the shell: a throwaway Iced app with a `Shader` widget uploading a
static 1080p I420 texture at 60 fps, `perf stat`, against
`PARALLAX_FORCE_GPU=1 parallax-player`. That isolates the cost of Iced's per-redraw
`view()` + layout + text shaping, which `AutoVideoSink` never paid. Then the real
matrix — CPU cores, arena bytes, dropped frames — for software, VA-API + upload and
(later) VA-API + import, against 0.54 / 0.24 / 0.12 cores and 88 → 34 MB. **Extend
`scripts/decode_matrix.py`** rather than write a new harness: it already drives
`--exit-after`, parses `stats: position_ns=… dropped=…` and reports arena MB and drops.

**B5 ([#271](https://git.marcpardo.eu/marcpardo/parallax/issues/271)). Colour correctness, end to end (S).** `present.wgsl` gains an `srgb_target` flag;
`video::color` gains `srgb_to_linear`/`linear_to_srgb` as the scalar reference with
golden tests, the same pattern `coefficients`/`yuv_to_rgb` already follow, so shader
drift stays reviewable as a diff against `color.rs`. Proven by an integration test:
render a limited-range Y=16 / Y=235 pattern through the widget, screenshot via
`Compositor::screenshot`, assert `0x000000` and `0xFFFFFF`. That test catches this whole
class of bug forever.

### Phase C — the player

**C1 ([#272](https://git.marcpardo.eu/marcpardo/parallax/issues/272)). App shell and the `Engine` lifecycle (L).** `probe` (blocking, on
`spawn_blocking` — an MP4 index parse is 100 ms+) → `build` + `Executor::start`
(**synchronously in `update()`**: `start` is sync and fast, and running it on the GUI
thread means `PipelineHandle` never needs to be `Send`) → `teardown` (on a `Task`:
`stop()`, await `ended` with a 5 s timeout, `abort()`). Handles are re-taken per build,
before `start`: the mailbox, `GainControl`, the `DropCounter` tracer; then `take_bus`,
`duration`, `stopper`. Rebuilding bumps a generation, which is the mailbox id, which is
the subscription's recipe hash — so a stale frame from the previous file is not
representable.
This is what Iced unlocks: the "one winit event loop per process, forever" constraint
that forced `--loop` into the demuxer (`main.rs:114-116`, `demux/mp4.rs:1284-1288`)
disappears when there is no window in the pipeline. **Both of those comments must be
rewritten as part of this issue.** Keep `Mp4DemuxSource::with_loop` — it is cheaper than
a rebuild and emits a fresh `Segment` per lap so paced sinks stay locked — but it is now
*a* way to loop, not *the only* way. Opening a second file, a second window
(`daemon()`), picture-in-picture and recovering from a failed open all fall out.

**C2 ([#273](https://git.marcpardo.eu/marcpardo/parallax/issues/273)). Transport UI (L).** Video filling the window with an auto-hiding overlay
(`stack!`): play/pause, seek bar, elapsed/duration, volume + mute, speed, fullscreen.
The seek-coalescing state machine from `main.rs:905-1121` ports across essentially
unchanged and must — accumulate into a local target, debounce 120 ms, one seek in
flight, correlate by `SeekDone` seqnum, supersede after 1 s — because `position()` does
not move until the seek's `Segment` lands. Keep the chrome cheap and *unmounted* when
hidden, so a playing-fullscreen tree is one `Shader` widget.

**C3 ([#274](https://git.marcpardo.eu/marcpardo/parallax/issues/274)). Input and window UX (M). DONE.** All of it, plus `Esc` that leaves
fullscreen rather than toggling into it. Two notes for the next person: the
pointer hides under the *same* rule as the chrome (`MouseArea::interaction` only
overrides an interaction the content reported as `None`, which `shader` does), and
drag-and-drop is the garnish rather than the mechanism — Iced's `FileDropped` says
"Wayland: Not implemented" in its own docs. The dialog is `ashpd`'s portal, because
parallax already depends on ashpd for screencast and `rfd` would bring a second major
of it; `just check-iced` asserts exactly one, the way it does for wgpu.

**C4 ([#275](https://git.marcpardo.eu/marcpardo/parallax/issues/275)). Diagnostics (M). DONE.** Three of these were facts the player already had
and threw away — the decoder choice was logged and forgotten, `to_dot()` was called on a
local that went out of scope, and `DurationChanged` never reached the seek bar. Which
adapter is drawing has exactly one honest source: `wgpu::Device` has no `adapter_info()`
in wgpu 27 and the custom-primitive path is handed nothing else, so
`iced::system::information()` it is.

**C5 ([#276](https://git.marcpardo.eu/marcpardo/parallax/issues/276)). Playlist and open-next (M). DONE.** Advance on the bus's `MessageKind::Eos`,
which is the only end-of-stream signal that is an edge, arrives after every task
finished, and dies with the engine it belongs to — `mailbox.end_reason()` is a level
raised while frames are still queued, and `PipelineHandle::ended()` outlives its engine
while `stop()` reports `Eos`, so both would advance twice on a manual Next. The open
chain became a state machine because it had a real race: `open()` took the engine before
spawning the teardown, so a second press built a second pipeline over the first.

**C6 ([#277](https://git.marcpardo.eu/marcpardo/parallax/issues/277)). Retire the crossterm player (S). DONE.** The old crate is deleted and the
Iced one took its name, its binary path and its measurement flags
(`--no-sync`, `--channel-capacity`, `--frame-delay`, `--loop` as an alias), so
`scripts/decode_matrix.py` is unchanged. `--headless` is now real rather than
promised: the process becomes the host and drains the mailbox at the widget's
own due times, needing no compositor, GPU or sound card — which is how CI runs
the player end to end. Two things had to be re-homed: the 14-feature union the
old Cargo.toml was CI's only build of (now a `feature-union` job against the
root crate, which additionally compiles `dmabuf_import.rs`'s `hardware_tests`
for the first time), and the name `--headless`, which
`scripts/decode_matrix.py` used for the *opposite* meaning and which is
`--weston` there now.

### Phase D — engine gaps the player exposes

**D1 ([#278](https://git.marcpardo.eu/marcpardo/parallax/issues/278)). Track enumeration and selection (L).** Language, title, default/forced on
`Mp4Track`/`MkvTrack`; post `MessageKind::StreamCollection` and `TagList` for the first
time; a runtime select API on the demuxers.

**D2 ([#279](https://git.marcpardo.eu/marcpardo/parallax/issues/279)). Subtitles (XL).** Route `S_TEXT/UTF8`, `S_TEXT/ASS` and `tx3g` out of the demuxers
and parse them. **Render in the Iced layer, not as a pipeline element** — text over
video is a UI problem, Iced already has the text stack, and a `TextOverlay` element
would mean rasterising into video frames for no benefit.

**D3 ([#280](https://git.marcpardo.eu/marcpardo/parallax/issues/280)). Rate ≠ 1.0, honestly (M).** `AlsaSink` handles `INSTANT_RATE_CHANGE` — at minimum
by muting and saying so on the bus, rather than today's guaranteed desync, since it is
the clock master. Time-stretch is out of scope; the UI states what it does.

**D4 ([#281](https://git.marcpardo.eu/marcpardo/parallax/issues/281)). Frame stepping (M). DONE (the cheap half).** Pause + `ACCURATE` seek to
`position ± frame_duration`; the real `Event::Step` design is written up below
("A real step event") and deliberately not built.

**D5 ([#282](https://git.marcpardo.eu/marcpardo/parallax/issues/282)). Thumbnails and seek preview (L).** A second short-lived pipeline over a second file
handle (`Mp4Demux` → decoder → `AppSink` → `JpegEncoder`), driven from the GUI, with a
cache. Impossible for cue-less MKV and for HEVC; say so in the UI.

**D6 ([#283](https://git.marcpardo.eu/marcpardo/parallax/issues/283)). Snapshot (S).** `Valve` → `VideoConvert(→Rgb24)` → `PngEncoder` → `FileSink`
branch, triggered from the UI. Note the encoders take RGB/RGBA/Gray only, so the convert
is mandatory.

**D7 ([#284](https://git.marcpardo.eu/marcpardo/parallax/issues/284)). Audio output device picker (M).** Wire `AlsaSink::enumerate_devices` to the UI, and
make `PipeWireSink`'s device parameter actually do something.

**D8 ([#285](https://git.marcpardo.eu/marcpardo/parallax/issues/285)). Deduplicate the two presenters (S, decision). DECIDED — see "The two
presenters" below.**

Overlap with `plans/README.md`'s proposed list, to be reconciled rather than duplicated:
proposed plan **24** (video filters) already names Flip/Rotate and TextOverlay — A4's
rotation and D2's overlay decision are the player-driven slice of it; proposed plan
**26** covers H.265, which stays there.

Deferred, with reasons: HEVC (#200/#202, upstream-blocked), HLS/DASH sources and
`Read + Seek` over HTTP (a network workstream of its own), 10-bit/HDR (no end-to-end
path exists at any layer), MP3/FLAC streaming decoders.

## Recommended sequencing

1. **A1 ([#263](https://git.marcpardo.eu/marcpardo/parallax/issues/263))** — pure moves, `just check` proves it, unblocks everything, and ships
   value on its own: the colour golden tests enter the default test job for the first
   time.
2. **B4 ([#270](https://git.marcpardo.eu/marcpardo/parallax/issues/270))'s first experiment** — one day, and it can invalidate the shell choice
   before any real code exists.
3. **A2 ([#264](https://git.marcpardo.eu/marcpardo/parallax/issues/264))** plus `examples/64_appvideosink.rs` — testable with no GUI at all.
4. **B1 ([#267](https://git.marcpardo.eu/marcpardo/parallax/issues/267))**, **B5 ([#271](https://git.marcpardo.eu/marcpardo/parallax/issues/271))**, **B2 ([#268](https://git.marcpardo.eu/marcpardo/parallax/issues/268))**.
5. **C1 ([#272](https://git.marcpardo.eu/marcpardo/parallax/issues/272))**, then the rest of Phase C.
6. **B3 ([#269](https://git.marcpardo.eu/marcpardo/parallax/issues/269))** (fork Iced, upstream both PRs) and Phase D, in whatever order the
   player's use makes urgent.

## Breaking changes

- `parallax-player` is rewritten; its CLI keeps `--exit-after`, the `stats:`
  line, and the measurement flags (`--no-sync`, `--channel-capacity`,
  `--frame-delay`, `--dav1d-threads`, `--decode-ahead`, `--hwdec`,
  `--no-audio`, `--loop` as an alias for `--repeat`). `--no-gpu` is gone with
  the `display-gpu` path it selected, and `--headless` is new.
- `parallax::video` moves code out of `elements::app::present`; `present` is re-pointed
  at it. No public API is lost — `present` was `pub(crate)` — but every promoted item
  needs a doc comment, since `#![warn(missing_docs)]` is on.
- A3 and A4 change `VideoFormat`, which is in caps and on the wire: `wire::codes` is
  append-only, so new fields need new codes and a `WIRE_VERSION` bump, and
  `IpcDescriptor` may need to grow again (it went 128 → 192 bytes in #219).
- The plugin ABI should **not** need a bump: `AsyncSink` already carries everything
  `AppVideoSink` uses.

## Verification

- `just check` stays green; `just check-display` still covers `display` alone and
  `display,display-gpu`, which is what catches a module losing its `cfg`.
- New `just check-iced` — `cargo clippy -p parallax-iced --all-targets -- -D warnings`
  plus its tests — and a CI job for it, since `default-members = ["."]` means a bare
  `cargo` build never sees the new crate (the same reason `workspace-members` exists).
- A `cargo tree -d -p parallax-player` check that fails on a duplicated `wgpu`. Note the
  hazard: a workspace-wide feature union (`cargo build --workspace`) *would* unify
  `display-gpu` into the player and pull wgpu 30 alongside 27. It compiles — two majors
  coexist — it just doubles the build. Extend the `default-members` comment to name it.
- The `parallax::video` golden tests must produce byte-identical values before and after
  A1; the WGSL constants stay pinned to `color.rs`.
- `tests/av_sync_tests.rs` gains an `AppVideoSink` case beside its `PacedVideoSink`
  stand-in, so the new sink's anchor and flush behaviour is proven headlessly.
- B5's screenshot test is the colour gate; B4's harness is the performance gate.
- The UI is testable headlessly with `iced_test`, already a dependency in both sibling
  repos: transport state, seek coalescing and the overlay behave the same with a stub
  mailbox and no GPU.
- Docs move with the code: `docs/elements.md` gains `AppVideoSink`, `docs/README.md` an
  index entry for `parallax::video`, `CLAUDE.md` a gotcha for the wgpu-major split and
  the `default-members` hazard. `README.md`'s feature table is asserted against
  `Cargo.toml` by `tests/readme_features.rs`.
- Manual: play an MP4 and an MKV; seek forward and backward through a keyframe-sparse
  region; pause and resume; change volume; go fullscreen; minimise for a minute with
  audio playing and confirm audio never underruns while `dropped_full` climbs; open a
  second file without restarting; close the window.

## Risks

| Risk | Mitigation |
|---|---|
| Iced never reaches wgpu 30, so v2 never lands and we stay at 0.24 cores | v1 is correct and shippable alone; the import path is one `if`. If upstream stalls, our fork carries the bump — measured at 54 mechanical errors plus `cryoglyph`. |
| Iced's per-redraw `view()` + layout + text shaping costs more than `WgpuBackend` ever paid, wiping out the shell's benefit | Unmount the chrome when hidden; redraw at the *video* rate, not the display rate. **B4's first experiment measures this before anything is written** — it is the single most decision-relevant number in the plan. |
| sRGB double-encoding ships and the picture is washed out | B5's `srgb_target` uniform plus the screenshot test. |
| `on_submitted_work_done` retention is wrong or driver-flaky, and the decoder overwrites a frame being sampled — silent corruption | Plan B: submit our own copy and fence on its exact index. Validate the mechanism *today* by A/B-ing it inside `WgpuBackend` against the existing exact-index wait on the VA-API import path. |
| The GUI stops draining (occluded, minimised) and the pipeline starves | Drop-oldest ring plus `StallWatch` in the widget; `consume` never blocks indefinitely on mailbox space. |
| `FrameAcceptance` is decided by a probe that disagrees with the device Iced actually creates, so the sink advertises `DmaBuf` to a producer whose frames we cannot read | v2 makes the probe create the real device and cache it. Moot in v1, which never advertises `DmaBuf`. Worth logging both adapters today — `PowerPreference::HighPerformance` (Iced) and `default()` (parallax) may already differ on a multi-GPU box. |
| Cadence never gets built and 23.976/25 content judders more than `AutoVideoSink` did | `take_due` alone is no worse; the vsync-locked selector is additive. Ship a present-policy flag so a regression is measurable and revertible. |
| A3/A4 touch the wire format and caps — a wide blast radius | Extend `tests/wire_projection.rs` round-trip cases *before* adding the fields. |
| Scope: 23 issues is a large milestone | A+B+C are the spine; Phase D items are individually droppable. |


## What building it changed

Written after the fact, because a plan that only records what it predicted is
not much use to the next person.

**The clock reaches an element only through its consume context.** `set_clock`
is on the adapter, not on `AsyncSink`, so `FrameMailbox::clock()` cannot exist
until the first buffer arrives. Every host has to handle `None` by drawing on
arrival. That is an API consequence, not a detail, and it was invisible from
outside the trait.

**The wake-up policy has to *be* the take policy.** The widget first computed
its next redraw from the raw frame target and let `take_due` decide separately
whether a frame was due. Those disagree exactly where `MAX_WAIT`'s PTS-jump
escape hatch fires: a target more than a second ahead is due *now*, while
sleeping until it stalls the picture for as long as the jump was. The widget
calls `pipeline::schedule` for both now.

**Counters a UI reads must be monotonic.** The sink originally reset
`presented`/`dropped_late` when it rolled a QoS window, so a diagnostics panel
watched its own numbers go down and a measurement run reported nonsense. The
mailbox keeps cumulative counters; the sink keeps its own window baselines.

**A packed frame is not four bytes per pixel.** The geometry read inherited
that assumption from `AutoVideoSink`, whose caps admit nothing else. Shared, it
turned a 320×240 `Rgb24` frame into 278×207 — no error, just a wrong picture.

**Seeking from a GUI needed a synchronous dispatch.** `PipelineHandle::seek` is
`async` and never awaits; an Iced `update` is synchronous and cannot move the
handle into a task. Rather than rely on that, `dispatch_seek` says it.

**And a harness trap worth more than the code that found it.** weston stops
repainting when it goes idle, and a Wayland client that never receives a frame
callback never draws. That is indistinguishable from a stalled pipeline, and it
cost an hour of hunting a bug that was not there. `just play-headless` and the
matrix runner both start a fresh compositor for exactly this reason.

**What the second run changed.** Three things worth carrying forward.

*An `Engine` dropped instead of torn down is a leaked pipeline.* `PipelineHandle`
has no `Drop`, so dropping one detaches its tasks — decoder threads, arena, an open
sound device — and nothing ever ends them. The convention that held it together
("`self.engine` is only assigned when it is `None`") survived exactly until a
playlist, and the fix was to make the invariant a type rather than a habit.

*The three end-of-stream signals mean three different things.*
`FrameMailbox::end_reason()` is raised when the EOS *event* reaches the sink, with
frames still queued and the audio device buffer still to play, and stays raised.
`PipelineHandle::ended()` outlives the engine and reports `Eos` for a `stop()`. Only
the bus's `MessageKind::Eos` is an edge, posted after every task finished, on a bus
that dies with its engine. A player that picks the wrong one advances a playlist
twice on every manual Next, or truncates the tail of every file.

*A CI job that builds a crate as a dependency compiles none of its tests.*
`cargo clippy -p parallax-player` was the tree's only build of the display-gpu ×
vaapi × simd union — and because the root crate was a dependency there, `cfg(test)`
was off, so `dmabuf_import.rs`'s `hardware_tests` had never been compiled at all. It
did not compile. `--all-targets` on the root crate is the version of that job that
means something.

### What the dev box could and could not prove

`just play-headless` runs the real player with no display attached: 20 frames
presented from the 2 s H.264/AAC fixture with nothing dropped, and 83 across
four laps under `--repeat`, running time tracking the frame targets to within
100 ms. The offscreen render tests in `parallax-iced` check the pixels
themselves — limited-range black and white on the rails, mid-grey linearised by
exactly what `video::color` says, letterbox bars painted rather than left
behind.

What it **cannot** show is a cores number. There is no GPU here (`/dev/dri` is
empty, no `vainfo`), so the renderer is Mesa's software rasteriser and it
dominates any CPU measurement. `scripts/decode_matrix.py --weston` records
frame accounting, arena bytes and position honestly and says so in its own
help; the 0.54 / 0.24 / 0.12-core ladder from #190 and #62 is still owed by a
machine with hardware.

## The two presenters (#285, decided)

After this milestone parallax has two wgpu presenters: `WgpuBackend` +
`dmabuf_import` in the core crate on **wgpu 30**, and `parallax-iced`'s
pipeline on **Iced's wgpu 27**. The decision is to **keep both**, and this is
what it rests on rather than taste.

**`display-gpu` is not redundant with the widget, because they do not have the
same consumers.** `parallax-launch` — the gst-launch equivalent, and a
headline capability of a crate whose front door is pipeline strings — reaches
the screen only through `autovideosink`, as do examples 23, 24, 58 and 59. A
one-liner that ends in a *player crate* is not the same product.

**It is the only place `plane_imports` meets a real Vulkan device.** The
`parallax::video` maths (#263) is shared by both presenters and pinned by
golden tests, but golden tests do not catch a wrong DRM format modifier or a
mis-declared plane offset. `display-gpu` + `dmabuf_import` is the one in-tree
consumer that hands those numbers to a driver, and CI's `display-check` job is
what keeps them honest. Deleting it would leave the arithmetic tested and the
*use* of it untested.

**Deleting it makes playback measurably worse, not just differently
structured.** `AutoVideoSink`'s caps flip to `[Rgba, Bgra]` + `cpu_only()`
without the GPU backend (`autovideosink.rs:1020-1026`), so every playback
graph regains a mandatory full-frame `videoconvert`: at 1080p that is 8.3 MB
written per frame plus the letterbox blit, against the 3.1 MB upload the GPU
path collapses it to. Ten tests and a CI job go with it.

**And the duplication is already small and honest.** After A1 the shared
surface is the plane geometry, the letterbox rect, the colour matrices and the
WGSL — all in `parallax::video`, all unfeatured, all tested once. What is
duplicated is roughly *one `unsafe fn` per wgpu major*: the dmabuf import call
into wgpu-hal. The hal API genuinely differs between majors, and an
abstraction over it would have to lie about which one it is talking to.

**The option worth revisiting, and its precondition.** Re-expressing
`AutoVideoSink`'s GPU path *on top of* `parallax-iced` — a minimal Iced
application with a single video widget — would collapse the duplication
entirely and hand `autovideosink` window management, HiDPI and input handling
for free. It also makes the core crate depend on Iced for a gated feature,
which is a real cost to weigh, not a free win. Its precondition is B3 (#269):
one wgpu major across the tree. **Checked 2026-08-27: Iced master is
`0.15.0-dev` and pins `wgpu = "29"`**, so the precondition is still not met,
and nothing here can make it so. Revisit when a released Iced is on wgpu 30.

The related A2 follow-up — whether `AutoVideoSink` should be rebuilt on top of
`AppVideoSink` — resolves the same way and for a smaller reason: `AppVideoSink`
deliberately owns no window, and `AutoVideoSink` is *entirely* about owning
one. What they share is the pacing decision, and that is already shared:
`pipeline::schedule` is one function both call.

## A real step event (#281's follow-up, designed not built)

The shipped step is a flushing `ACCURATE` seek per frame. It is correct and it
is wasteful: a flush trio, a decoder reset and a keyframe re-decode to advance
by 40 ms. GStreamer's answer is `GstEventStep`, and the shape that fits this
tree is:

```rust
Event::Step { format: SegmentFormat, amount: u64, rate: f64, flush: bool }
```

**It is a third upstream family**, alongside QoS (an advisory) and
ForceKeyUnit (an imperative the first encoder consumes). A step is an
imperative the first *paced sink* consumes, and it travels no further —
exactly `INSTANT_RATE_CHANGE`'s sink-terminated shape (#165), for the same
reason: nothing upstream should reposition.

**The sink is where it lands, because the sink is what is holding the frames.**
A paused `AppVideoSink` has up to `capacity` frames queued and a host parked
on the doorbell. A step of N means: release exactly N, re-park, and post
`MessageKind::StepDone { amount, eos }`. No flush, no seek, no decoder reset —
and the decoder keeps running ahead, so a held `.` steps at decode speed
rather than at flush speed.

**Three things it must get right**, none of which the seek version has to:

1. **The clock does not move while paused**, so a released frame cannot be
   paced against running time — it is presented on arrival, and the sink's
   `position()` must advance by the frame it released or the readout lies.
2. **Running out of queued frames is not EOS.** A step past the end of the
   queue has to either unblock the producer for one frame (raising the pause
   gate for a single buffer, which the `watch` cannot express today) or answer
   `StepDone { eos: false, amount: 0 }` and let the caller retry. The second
   is honest and simple; the first is what makes holding `.` smooth.
3. **`rate` is not the playback rate.** GStreamer's step rate scales the
   *step*, which is how frame-accurate scrubbing at 0.1× works. Ignoring it
   is fine for v1 but the field should exist, or adding it later is another
   ABI bump.

**Cost**: a new `Event` variant (no ABI implication — `Event` is not returned
by value across the plugin boundary, unlike `MediaFormat`), a
`handle_upstream_event` arm on the paced sinks, a `PipelineHandle::step`, and
the pause-gate change in (2) if smoothness is wanted. Backward stepping stays
out: it needs decode-order buffering the engine does not do, and a backward
*seek* already covers the use case at a hitch's cost.
