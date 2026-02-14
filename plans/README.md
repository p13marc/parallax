# Parallax Implementation Plans

This directory contains implementation plans for remaining Parallax features.

Completed plans have been removed. See git history for details.

---

## Remaining Plans

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 11 | [GPU Codec Framework](11_GPU_CODEC_FRAMEWORK.md) | High | Large | 🟡 ~60% (skeleton done, decode/encode incomplete) |
| 16 | [Process Isolation](16_PROCESS_ISOLATION.md) | High | Large | ⬜ Not Started |
| - | [Auto Clock Selection](AUTO_CLOCK_SELECTION.md) | Medium | Small | ⬜ Not Started |

**Legend:** ⬜ Not Started | 🟡 In Progress

---

## Plan Summaries

### Plan 11: GPU Codec Framework (Vulkan Video)
Hardware-accelerated video encoding/decoding via Vulkan Video. Skeleton and NAL parsing done. Remaining: complete H.264/H.265/AV1 decode, H.264/H.265 encode, video session management.

### Plan 16: Process Isolation
Production-ready process isolation with seccomp/namespace sandboxing. Scaffolding exists (IPC, supervisor protocol) but needs full sandbox implementation.

### Auto Clock Selection
Add automatic clock provider selection to pipelines. The clock infrastructure exists (Clock, ClockProvider, PipelineClock, AlsaClock) but selection is manual. This plan adds `as_clock_provider()` to `AsyncElementDyn` and `Pipeline::select_clock()` to auto-select the highest-priority clock.

---

## Completed Plans (removed)

Plans 00-10, 12-15, 17, Clock Provider, and Pipeline Robustness have been completed and their files removed. Key completed work:

- **Phase 1** (Plans 00-08): Metadata API, codec wrappers, muxer sync, buffer pool, element trait consolidation, caps negotiation, builder DSL, events/tagging
- **Phase 2** (Plans 09-10, 12-14): Format converters, code cleanup, additional codecs (Opus, AAC, Symphonia), device elements (V4L2, PipeWire, ALSA, libcamera, screen capture), streaming protocols (HLS, DASH)
- **Plan 15**: RT scheduling (SyncElement trait, RT thread spawning, driver integration, hybrid async/RT pipelines)
- **Plan 17**: Consolidated `affinity()`, `is_rt_safe()`, and `execution_hints()` into single `execution_hints()` method; removed `Affinity` enum (PipeWire-inspired capability-based scheduling)
- **Clock Provider**: Hardware timestamp extraction (PipeWire, V4L2, ALSA), Clock/ClockProvider traits, PipelineClock, TimestampDebug element
- **Pipeline Robustness**: Arena reclaim hygiene, backpressure system (FlowSignal, FlowPolicy, Queue water marks), video scaler
