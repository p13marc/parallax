# Parallax Implementation Plans

This directory contains implementation plans for remaining Parallax features.

Completed plans have been removed. See git history for details.

---

## Remaining Plans

### Tier 1: Generic Pipeline Infrastructure (Highest Priority)

These are media-independent features that make the pipeline engine complete.

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 18 | [Pipeline Bus & Messaging](18_PIPELINE_BUS_MESSAGING.md) | Critical | Medium | ⬜ Not Started |
| 19 | [Seeking, Position Queries & Trick Modes](19_SEEKING_POSITION_QUERIES.md) | Critical | Large | ⬜ Not Started |
| 20 | [Pad Probes & Dynamic Reconfiguration](20_PAD_PROBES_DYNAMIC_RECONFIGURATION.md) | High | Medium | ⬜ Not Started |
| 21 | [Debugging & Inspection Tools](21_DEBUGGING_INSPECTION_TOOLS.md) | High | Medium | ⬜ Not Started |
| 22 | [Auto-Plugging & Typefinding](22_AUTO_PLUGGING_TYPEFINDING.md) | High | Large | ⬜ Not Started |
| 23 | [Network Buffering Strategies](23_NETWORK_BUFFERING_STRATEGIES.md) | High | Medium | ⬜ Not Started |

### Tier 2: Media Processing Elements

Media-specific elements for production pipelines.

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 24 | [Video Processing Filters](24_VIDEO_PROCESSING_FILTERS.md) | Medium | Medium-Large | ⬜ Not Started |
| 25 | [Audio Processing Filters](25_AUDIO_PROCESSING_FILTERS.md) | Medium | Medium | ⬜ Not Started |

### Tier 3: Codec & Format Coverage

Expanding media format support for competitive parity.

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 11 | [GPU Codec Framework](11_GPU_CODEC_FRAMEWORK.md) | Medium-High | Large | 🟡 ~60% |
| 26 | [Video Codec Expansion (H.265, VP8/VP9)](26_VIDEO_CODEC_EXPANSION.md) | Medium-High | Medium | ⬜ Not Started |
| 27 | [Container Format Expansion (MKV, FLV, WAV, Ogg)](27_CONTAINER_FORMAT_EXPANSION.md) | Medium | Medium | ⬜ Not Started |
| 28 | [Streaming Protocol Expansion (WebRTC, SRT, RTMP)](28_STREAMING_PROTOCOL_EXPANSION.md) | Medium-High | Large | ⬜ Not Started |

### Tier 4: Advanced Infrastructure

Complex features for specialized use cases.

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 16 | [Process Isolation](16_PROCESS_ISOLATION.md) | Low | Large | ⬜ Not Started |

**Legend:** ⬜ Not Started | 🟡 In Progress | ✅ Complete

---

## Recommended Implementation Order

The dependency graph and priority suggests this order:

```
Phase A (Foundation):
  18 → 19 → 20
  ↓         ↓
  21        23
  ↓
  22

Phase B (Media — can start in parallel with Phase A):
  26 → 27 → 28
  24
  25
  11

Phase C (Advanced):
  16
```

**Phase A** focuses on generic pipeline infrastructure. Plan 18 (Bus) is the foundation for Plans 19, 21, 22, and 23. Plan 19 (Seeking) is required for Plan 22 (Auto-plugging). Plan 20 (Pad Probes) enables dynamic reconfiguration.

**Phase B** focuses on media format coverage. Plans 24/25 (filters) and Plan 26 (codecs) can start independently. Plan 27 (containers) is needed before Plan 28 (RTMP needs FLV, WebRTC needs VP8/VP9 from Plan 26).

**Phase C** is for specialized advanced features after the core is solid.

---

## Plan Summaries

### Plan 11: GPU Codec Framework (Vulkan Video)
Hardware-accelerated video encoding/decoding via Vulkan Video. Skeleton and NAL parsing done. Remaining: complete H.264/H.265/AV1 decode, H.264/H.265 encode, video session management.

### Plan 16: Process Isolation
Production-ready process isolation with seccomp/namespace sandboxing. Scaffolding exists (IPC, supervisor protocol) but needs full sandbox implementation.

### Plan 18: Pipeline Bus & Messaging System
Thread-safe message bus for element-to-application communication. Typed messages (error, warning, tag, QoS, buffering, state change). Sync polling and async stream consumption. Foundation for Plans 19, 22, 23.

### Plan 19: Seeking, Position Queries & Trick Modes
Seek event propagation (upstream), flush events, position/duration queries, playback rate control (fast forward, slow motion), segment events for timestamp mapping. Required for media player use cases.

### Plan 20: Pad Probes & Dynamic Pipeline Reconfiguration
Buffer/event interception at pads with DROP/PASS/BLOCK semantics. Safe dynamic reconfiguration (block pad, relink, unblock). Add/remove elements while running. Enables recording triggers, stream switching.

### Plan 21: Debugging & Inspection Tools
`parallax-inspect` CLI (browse elements), DOT graph dumps, tracer framework (latency/framerate/queue level), `parallax-top` TUI monitor. Essential developer experience.

### Plan 22: Auto-Plugging & Typefinding
TypeFind (detect format from bytes), DecodeBin (auto demux+decode chain), PlayBin (auto-play any URI). Enables "just play this file" use case.

### Plan 23: Network Buffering Strategies
Queue2 with stream buffering (watermarks), download buffering (disk-backed), timeshift buffering (DVR rewind). Buffering progress via bus messages. Required for reliable network streaming.

### Plan 24: Video Processing Filters
VideoCrop, VideoFlip/Rotate, VideoRate (framerate adjust), ColorBalance, Compositor (PiP/grid), TextOverlay, ImageOverlay. Production video pipeline essentials.

### Plan 25: Audio Processing Filters
AudioMixer (N-to-1), ChannelMix (upmix/downmix), Equalizer (parametric biquad), Compressor/Limiter, AudioLevel (peak/RMS metering), AudioPanorama. Production audio pipeline essentials.

### Plan 26: Video Codec Expansion
H.265/HEVC encode (x265) and decode (libde265). VP8/VP9 encode/decode (libvpx). Needed for modern video content and WebRTC.

### Plan 27: Container Format Expansion
MKV/WebM (matroska-demuxer), FLV (flavors), WAV (hound), Ogg (ogg crate). All pure Rust or permissively licensed. FLV is prerequisite for RTMP.

### Plan 28: Streaming Protocol Expansion
WebRTC (str0m, WHIP/WHEP), SRT (srt-rs), RTMP (rml-rtmp). Highest-demand streaming protocols for live interactive and broadcast use cases.

---

## Completed Plans (removed)

Plans 00-10, 12-15, 17, Clock Provider, and Pipeline Robustness have been completed and their files removed. Key completed work:

- **Phase 1** (Plans 00-08): Metadata API, codec wrappers, muxer sync, buffer pool, element trait consolidation, caps negotiation, builder DSL, events/tagging
- **Phase 2** (Plans 09-10, 12-14): Format converters, code cleanup, additional codecs (Opus, AAC, Symphonia), device elements (V4L2, PipeWire, ALSA, libcamera, screen capture), streaming protocols (HLS, DASH)
- **Plan 15**: RT scheduling (SyncElement trait, RT thread spawning, driver integration, hybrid async/RT pipelines)
- **Plan 17**: Consolidated `affinity()`, `is_rt_safe()`, and `execution_hints()` into single `execution_hints()` method; removed `Affinity` enum (PipeWire-inspired capability-based scheduling)
- **Clock Provider**: Hardware timestamp extraction (PipeWire, V4L2, ALSA), Clock/ClockProvider traits, PipelineClock, TimestampDebug element
- **Auto Clock Selection**: `as_clock_provider()` on element traits, `Pipeline::select_clock()`, AlsaSink auto-provides clock
- **Pipeline Robustness**: Arena reclaim hygiene, backpressure system (FlowSignal, FlowPolicy, Queue water marks), video scaler
