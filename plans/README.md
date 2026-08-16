# Parallax Implementation Plans

This directory contains implementation plans for remaining Parallax features.

Completed plans have been removed (see git history). Only plans with an existing file are linked; unlinked entries are proposed work whose plan documents have not been written yet.

---

## Active Plans

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 11 | [GPU Codec Framework](11_GPU_CODEC_FRAMEWORK.md) | Medium-High | Large | 🟡 ~80% — H.264 decode fully wired (submission, POC/DPB refs, RAII memory) and fixture-tested; hardware validation pending (#3); no encode, no H.265/AV1 |
| 16 | [Process Isolation](16_PROCESS_ISOLATION.md) | Low | Large | ⬜ Not started — **note:** the earlier scaffolding (`src/execution/`) was removed in `da6df59`; any implementation starts from scratch with a spawn-based design |

## Proposed Plans (no plan file yet)

| # | Idea | Priority | Notes |
|---|------|----------|-------|
| 24 | Video processing filters | Medium | VideoCrop, Flip/Rotate, VideoRate, ColorBalance, Compositor, TextOverlay |
| 25 | Audio processing filters | Medium | AudioMixer, ChannelMix, Equalizer, Compressor/Limiter, AudioLevel |
| 26 | Video codec expansion | Medium-High | H.265 + VP8/VP9; consider rav1d for pure-Rust AV1 decode |
| 27 | Container format expansion | Medium | MKV/WebM, FLV, WAV, Ogg (pure Rust crates exist); FLV is prerequisite for RTMP |
| 28 | Streaming protocol expansion | Medium-High | WHIP (RFC 9725) sink, MoQ sink (Rust stacks exist), SRT, RTMP |
| — | Wayland `ext-image-copy-capture-v1` source | Low | Lower-latency screen capture without the portal/PipeWire path |

**Legend:** ⬜ Not Started | 🟡 In Progress | ✅ Complete

---

## Completed Plans (files removed)

- **`parallax-launch` CLI + full ElementFactory registration** (#187): bin target behind the `cli` feature; ~65 factory names across all domains; strict unknown-property errors; gated-name diagnostics. See `docs/elements.md` § "Pipeline-string factory names".

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
