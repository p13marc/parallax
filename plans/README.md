# Parallax Implementation Plans

This directory contains implementation plans for remaining Parallax features.

Completed plans have been removed (see git history). Only plans with an existing file are linked; unlinked entries are proposed work whose plan documents have not been written yet.

---

## Active Plans

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 11 | [GPU Codec Framework](11_GPU_CODEC_FRAMEWORK.md) | Medium-High | Large | 🟡 ~80% — H.264 decode fully wired (submission, POC/DPB refs, RAII memory) and fixture-tested; hardware validation pending (#3); no encode, no H.265/AV1 |
| 16 | [Process Isolation](16_PROCESS_ISOLATION.md) | Low | Large | ⬜ Not started — **note:** the earlier scaffolding (`src/execution/`) was removed in `da6df59`; any implementation starts from scratch with a spawn-based design |
| 29 | [Multi-machine video: RTSP → @media → browser](29_multi_machine_browser.md) | High | Large | 🟢 **Done** — all 31 issues in the milestone closed; epic [#204](https://git.marcpardo.eu/marcpardo/parallax/issues/204), PR [#242](https://git.marcpardo.eu/marcpardo/parallax/pulls/242). Make `ZenohSink`/`ZenohSrc` speak the ratified zenkey `@media` plane (CBOR `FrameMeta`, exact tier keys, `frame` QoS, byte-level keyframe promise), fix the encoders that never stamp `MediaFormat`, and harden RTSP ingest. WHEP egress stays deferred to its own milestone; the browser frontend is zenoh-ts + WebCodecs, tracked in zensight. See `docs/multi-machine.md` for the deployment guide. |
| 30 | [A real video player: Parallax on Iced](30_iced_player.md) | High | Large | 🟡 **Phases A–C done** — [#263](https://git.marcpardo.eu/marcpardo/parallax/issues/263)–[#268](https://git.marcpardo.eu/marcpardo/parallax/issues/268), [#270](https://git.marcpardo.eu/marcpardo/parallax/issues/270)–[#277](https://git.marcpardo.eu/marcpardo/parallax/issues/277) and [#280](https://git.marcpardo.eu/marcpardo/parallax/issues/280) merged: `parallax-player` is the Iced one now — it opens files, plays them in sync, seeks, keeps a playlist and says what the pipeline is doing. 8 issues left — epic [#286](https://git.marcpardo.eu/marcpardo/parallax/issues/286), milestone *A real video player: Parallax on Iced*. Replace the crossterm `parallax-player` demo with a real GUI player on Iced 0.14: a wgpu-free `parallax::video`, a paced toolkit-agnostic `AppVideoSink`, a new `parallax-iced` crate holding the `shader` widget, and the app itself. Zero-copy dma-buf display is deferred — `texture_from_dmabuf_fd` landed in wgpu 30 and Iced 0.14 is on wgpu 27, so v1 is the upload path (~0.24 cores at 1080p instead of ~0.12). Phase D closes the gaps a real player exposes: colorimetry, PAR/rotation, track selection, subtitles, rate≠1.0 audio. |
| — | [Bandwidth control (ZenSight ↔ Parallax sensor)](bandwidth-control.md) | Medium | Medium | 🟢 **Phase A done** — [#24](https://git.marcpardo.eu/marcpardo/parallax/issues/24)–[#32](https://git.marcpardo.eu/marcpardo/parallax/issues/32), epic [#33](https://git.marcpardo.eu/marcpardo/parallax/issues/33), all closed. Runtime control primitives shipped in `src/control/` (not the planned `elements/codec/control.rs`) — see the plan's reconciliation section. Phase B is zensight-side. |

## Proposed Plans (no plan file yet)

| # | Idea | Priority | Notes |
|---|------|----------|-------|
| 24 | Video processing filters | Medium | VideoCrop, Flip/Rotate, VideoRate, ColorBalance, Compositor, TextOverlay |
| 25 | Audio processing filters | Medium | AudioMixer, ChannelMix, Equalizer, Compressor/Limiter, AudioLevel |
| 26 | Video codec expansion | Medium-High | H.265 + VP8/VP9; consider rav1d for pure-Rust AV1 decode |
| 27 | Container format expansion | Medium | MKV/WebM, FLV, WAV, Ogg (pure Rust crates exist); FLV is prerequisite for RTMP |
| 28 | Streaming protocol expansion | Medium | MoQ sink (Rust stacks exist), SRT, RTMP. **WHIP/WHEP moved to plan 29** (str0m, ICE-lite, frame-level API) |
| — | Wayland `ext-image-copy-capture-v1` source | Low | Lower-latency screen capture without the portal/PipeWire path |

**Legend:** ⬜ Not Started | 🟡 In Progress | 🟢 **Done**

---

## Historical reports (kept for the rationale)

- **[ZenSight "parallax" video sensor — gap analysis & roadmap](zensight-parallax-sensor.md)**
  (2026-07-04). The study that started the sensor work. Superseded by plan 29 and by what
  actually shipped: several of its claims — notably "the `zenoh` feature does not compile" — were
  true when written and are not now. Kept because the keyspace and `FrameMeta` decisions in §2b
  and §4 are where the `@media` design came from.

## Completed Plans (files removed)

- **`parallax-launch` CLI + full ElementFactory registration** (#187): bin target behind the `cli` feature; ~90 factory names across all domains; strict unknown-property errors; gated-name diagnostics. See `docs/elements.md` § "Pipeline-string factory names".

- **Phase 1** (Plans 00–08): Metadata API, codec wrappers, muxer sync, buffer pool, element trait consolidation, caps negotiation, builder DSL, events/tagging
- **Phase 2** (Plans 09–10, 12–14): Format converters, code cleanup, additional codecs (Opus, AAC, Symphonia), device elements (V4L2, PipeWire, ALSA, libcamera, screen capture), streaming outputs (HLS, DASH)
- **Plan 15**: RT scheduling (SyncElement trait, RT thread spawning, driver integration, hybrid async/RT pipelines)
- **Plan 17**: Consolidated `affinity()`/`is_rt_safe()` into a single `execution_hints()` method; removed the `Affinity` enum
- **Plan 18**: Pipeline bus & messaging (Bus, BusHandle, MessageKind, TagList, BusStream)
- **Plan 19**: Seeking & position queries (SegmentEvent mapping, SeekableSource, FileSrc byte-seeking)
- **Plan 20**: Pad probes (ProbeType/ProbeReturn/ProbeData, ProbeRegistry, executor integration)
- **Plan 21**: Debugging & inspection (LatencyTracer/FramerateTracer/DropTracer, `PARALLAX_TRACERS`, `PARALLAX_DOT_DIR`, stats snapshot). The originally-planned `parallax-inspect`/`parallax-top` CLIs were **not** built.
- **Plan 22**: Typefinding (TypeFindRegistry, byte + extension detection). The originally-planned DecodeBin/PlayBin auto-plugging was **not** built.
- **Plan 23**: Network buffering (Queue2: stream/download/timeshift modes, rate estimation, bus buffering messages)
- **Clock work**: Clock/ClockProvider/PipelineClock, hardware timestamp extraction, `TimestampDebug`, automatic clock selection (`Pipeline::select_clock()`, AlsaSink provider)
- **Pipeline robustness**: arena reclaim hygiene, backpressure (FlowSignal/FlowPolicy/water marks), video scaler
- **Removed feature**: process isolation prototype (`src/execution/`) — deleted in `da6df59` (fork-bomb risk); tracked for redesign in Plan 16
