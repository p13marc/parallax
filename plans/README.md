# Parallax Implementation Plans

This directory contains implementation plans for remaining Parallax features.

Completed plans have been removed. See git history for details.

---

## Remaining Plans

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 11 | [GPU Codec Framework](11_GPU_CODEC_FRAMEWORK.md) | High | Large | 🟡 ~60% (skeleton done, decode/encode incomplete) |
| 15 | [RT Scheduling](15_RT_SCHEDULING.md) | High | Large | ⬜ Not Started |
| 16 | [Process Isolation](16_PROCESS_ISOLATION.md) | High | Large | ⬜ Not Started |
| - | [Clock Provider](clock-provider-implementation.md) | Medium | Medium | 🟡 ~40% (hardware timestamps done, clock negotiation pending) |
| - | [Pipeline Robustness](plan-11-pipeline-robustness.md) | High | Large | 🟡 ~30% (backpressure done, error recovery pending) |

**Legend:** ⬜ Not Started | 🟡 In Progress

---

## Plan Summaries

### Plan 11: GPU Codec Framework (Vulkan Video)
Hardware-accelerated video encoding/decoding via Vulkan Video. Skeleton and NAL parsing done. Remaining: complete H.264/H.265/AV1 decode, H.264/H.265 encode, video session management.

### Plan 15: RT Scheduling
Production-ready real-time scheduling. Infrastructure exists (RT scheduler, bridges, drivers) but needs hardening for production use.

### Plan 16: Process Isolation
Production-ready process isolation with seccomp/namespace sandboxing. Scaffolding exists (IPC, supervisor protocol) but needs full sandbox implementation.

### Clock Provider
Proper clock provider system for A/V synchronization. Hardware timestamp extraction from PipeWire/V4L2/ALSA is done. Clock negotiation and pipeline clock selection still needed.

### Pipeline Robustness
Make pipelines robust for real-time media processing. Backpressure/flow control is done. Remaining: encoder performance tuning, state machine hardening, error recovery.

---

## Completed Plans (removed)

Plans 00-10, 12-14, and Plan 17 (ExecutionHints consolidation) have been completed and their files removed. Key completed work:

- **Phase 1** (Plans 00-08): Metadata API, codec wrappers, muxer sync, buffer pool, element trait consolidation, caps negotiation, builder DSL, events/tagging
- **Phase 2** (Plans 09-10, 12-14): Format converters, code cleanup, additional codecs (Opus, AAC, Symphonia), device elements (V4L2, PipeWire, ALSA, libcamera, screen capture), streaming protocols (HLS, DASH)
- **Plan 17**: Consolidated `affinity()`, `is_rt_safe()`, and `execution_hints()` into single `execution_hints()` method; removed `Affinity` enum entirely (PipeWire-inspired capability-based scheduling)
