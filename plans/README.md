# Parallax Improvement Plans

This directory contains detailed implementation plans for improving Parallax, derived from the [Development Report](../docs/DEVELOPMENT_REPORT.md) and [Project Analysis Report](../docs/PROJECT_ANALYSIS_REPORT.md).

---

## Code Quality Reviews

- **[RUST_IDIOMS_REVIEW.md](RUST_IDIOMS_REVIEW.md)** - Review of Rust idioms, anti-patterns, and recommended crates

---

## Design Decisions

Key architectural decisions have been researched and documented in **[00_DESIGN_DECISIONS.md](00_DESIGN_DECISIONS.md)**, based on analysis of GStreamer, PipeWire, and academic literature.

| Decision | Choice |
|----------|--------|
| Metadata serialization for IPC | Yes, via rkyv with `MetaSerialize` trait |
| Default muxer sync strategy | Auto (adaptive based on live/non-live) |
| Separate events channel | No, unified channel + control signals for flush |
| Backward compatibility | No, break freely (pre-production) |
| Sync/async element traits | Unified async with sync blanket impl |
| Buffer pool negotiation | Pipeline-level, per-link size negotiation |
| Metadata storage format | HashMap (optimize later if needed) |
| Timestamp format | i64 nanoseconds (ClockTime) |

---

## Progress Overview

### Phase 1: Foundation (Complete)

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 00 | [Design Decisions](00_DESIGN_DECISIONS.md) | - | - | ✅ Complete |
| 01 | [Custom Metadata API](01_CUSTOM_METADATA_API.md) | High | Small | ✅ Complete |
| 02 | [Codec Element Wrappers](02_CODEC_ELEMENT_WRAPPERS.md) | High | Medium | ✅ Complete |
| 03 | [Muxer Synchronization](03_MUXER_SYNCHRONIZATION.md) | High | Large | ✅ Complete |
| 04 | [Pipeline Buffer Pool](04_PIPELINE_BUFFER_POOL.md) | High | Medium | ✅ Complete |
| 05 | [Element Trait Consolidation](05_ELEMENT_TRAIT_CONSOLIDATION.md) | Medium | Large | ✅ Complete |
| 06 | [Caps Negotiation](06_CAPS_NEGOTIATION.md) | Medium | Medium | ✅ Complete |
| 07 | [Pipeline Builder DSL](07_PIPELINE_BUILDER_DSL.md) | Medium | Small | ✅ Complete |
| 08 | [Events and Tagging](08_EVENTS_AND_TAGGING.md) | Medium | Medium | ✅ Complete |

### Phase 2: Enhancement (New)

| # | Plan | Priority | Effort | Progress |
|---|------|----------|--------|----------|
| 09 | [Format Converters](09_FORMAT_CONVERTERS.md) | High | Medium | ✅ Complete |
| 10 | [Code Cleanup](10_CODE_CLEANUP.md) | Low | Small | ✅ Complete |
| 11 | [GPU Codec Framework](11_GPU_CODEC_FRAMEWORK.md) | High | Large | ⬜ Not Started |
| 12 | [Additional Codecs](12_ADDITIONAL_CODECS.md) | Medium | Medium | ⬜ Not Started |
| 13 | [Device Elements](13_DEVICE_ELEMENTS.md) | Medium | Medium | ✅ Complete |
| 14 | [Streaming Protocols](14_STREAMING_PROTOCOLS.md) | Medium | Medium | ⬜ Not Started |

**Legend:** ⬜ Not Started | 🟡 In Progress | ✅ Complete

---

## Phase 2 Plan Summaries

### Plan 09: Format Converters
Implement actual video/audio format converters (currently stubs that return errors):
- Video: I420 ↔ RGB24, bilinear/nearest scaling
- Audio: S16 ↔ F32, sample rate conversion
- Enables automatic converter insertion in caps negotiation

### Plan 10: Code Cleanup
Resolve technical debt accumulated during development:
- Fix 35 compiler warnings
- Resolve or document 19 TODO comments
- Pass `cargo clippy -- -D warnings`

### Plan 11: GPU Codec Framework (Vulkan Video)
Hardware-accelerated video encoding/decoding:
- H.264, H.265, AV1 decode via Vulkan Video
- H.264, H.265 encode via Vulkan Video
- Zero-copy DMA-BUF integration
- Fallback to CPU codecs when unavailable

### Plan 12: Additional Codecs
Expand codec support beyond current offerings:
- Opus audio (encode/decode) - WebRTC standard
- AAC encode - streaming compatibility
- VP9 decode - YouTube legacy
- Document pure-Rust H.264 decoder limitations

### Plan 13: Device Elements
Hardware capture and playback:
- V4L2 video capture (`v4l2src`)
- ALSA audio capture/playback (`alsasrc`, `alsasink`)
- DMA-BUF export for GPU pipelines
- Screen capture (DRM/KMS)

### Plan 14: Streaming Protocols
Adaptive bitrate streaming support:
- HLS output (`hlssink`) with M3U8 playlists
- DASH output (`dashsink`) with MPD manifests
- Multi-rendition ABR pipelines
- RTMP output for live ingest

---

## Recommended Implementation Order (Phase 2)

### Quick Win (1-2 days)
1. **[Plan 10: Code Cleanup](10_CODE_CLEANUP.md)**
   - Zero warnings, clean codebase
   - Good starting point

### High Priority (2-3 weeks)
2. **[Plan 09: Format Converters](09_FORMAT_CONVERTERS.md)**
   - Completes caps negotiation (Plan 06)
   - Removes manual conversion requirement
   - Pure Rust, no new dependencies

### Medium Priority - Choose Based on Use Case

**For GPU-accelerated workflows:**

3. **[Plan 11: GPU Codec Framework](11_GPU_CODEC_FRAMEWORK.md)**
   - Major effort (4-6 weeks)
   - Unlocks hardware acceleration
   - Vulkan Video cross-vendor

**For audio/video capture:**

3. **[Plan 13: Device Elements](13_DEVICE_ELEMENTS.md)**
   - V4L2, ALSA support
   - Real-world input sources

**For streaming services:**

3. **[Plan 14: Streaming Protocols](14_STREAMING_PROTOCOLS.md)**
   - HLS/DASH output
   - Adaptive bitrate

**For expanded codec support:**

3. **[Plan 12: Additional Codecs](12_ADDITIONAL_CODECS.md)**
   - Opus for WebRTC
   - AAC for streaming

---

## Dependency Graph (Phase 2)

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1 COMPLETE                              │
│  Plans 01-08: Metadata, Codecs, Muxer, Pool, Caps, Events       │
└─────────────────────────────────────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               ▼               ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ Plan 09:     │ │ Plan 10:     │ │ Plan 11:     │
      │ Format Conv. │ │ Code Cleanup │ │ GPU Codecs   │
      │ (completes   │ │ (quick win)  │ │ (major feat.)│
      │  Plan 06)    │ │              │ │              │
      └──────────────┘ └──────────────┘ └──────┬───────┘
                                               │
                              ┌────────────────┤
                              │                │
                              ▼                ▼
                      ┌──────────────┐ ┌──────────────┐
                      │ Plan 12:     │ │ Plan 13:     │
                      │ Add. Codecs  │ │ Devices      │
                      │ (Opus, AAC)  │ │ (V4L2, ALSA) │
                      └──────────────┘ └──────────────┘
                              │
                              ▼
                      ┌──────────────┐
                      │ Plan 14:     │
                      │ Streaming    │
                      │ (HLS, DASH)  │
                      └──────────────┘
```

**Notes:**
- Plan 10 (Code Cleanup) can be done anytime
- Plan 09 (Converters) should be done before Plan 11 (GPU) for complete negotiation
- Plan 13 (Devices) benefits from Plan 11 (GPU) for DMA-BUF integration
- Plan 14 (Streaming) depends on muxers from Plan 03

---

## Effort Summary

### Phase 1 (Complete)
| Effort | Days | Plans |
|--------|------|-------|
| Small | 1-3 | 01, 07 |
| Medium | 3-7 | 02, 04, 06, 08 |
| Large | 7-14 | 03, 05 |
| **Total** | **6-10 weeks** | |

### Phase 2 (New)
| Effort | Days | Plans |
|--------|------|-------|
| Small | 1-2 | 10 |
| Medium | 7-14 | 09, 12, 13, 14 |
| Large | 20-30 | 11 |
| **Total** | **8-12 weeks** | |

---

## Master Checklist (Phase 1 - Complete)

<details>
<summary>Click to expand Phase 1 checklists</summary>

### Plan 01: Custom Metadata API ✅
- [x] Define `custom` HashMap field in `Metadata` struct
- [x] Implement `set<T>()` and `get<T>()` methods
- [x] Implement `set_bytes()` and `get_bytes()` convenience methods
- [x] Add `set_klv()` / `klv()` convenience methods
- [x] Add `metadata_mut()` to `Buffer` (already existed)
- [x] Write unit tests (15 tests in `src/metadata.rs`)
- [x] Update example 31 to use new API
- [x] Update documentation (CLAUDE.md)

### Plan 02: Codec Element Wrappers ✅
- [x] Define `VideoEncoder` trait
- [x] Define `VideoDecoder` trait
- [x] Define `VideoFrame` struct (already existed in common.rs)
- [x] Implement `EncoderElement<E>` wrapper
- [x] Implement `DecoderElement<D>` wrapper
- [x] Add `flush()` method to `Element` and `Transform` traits
- [x] Add `flush()` to `AsyncElementDyn` trait
- [x] Adapt `Rav1eEncoder` to `VideoEncoder`
- [x] Update executor to call `flush()` at EOS
- [x] Write unit tests (codec tests updated)
- [x] Create example 33 (encoder element)
- [x] Update documentation (CLAUDE.md)

### Plan 03: Muxer Synchronization ✅
- [x] Define `Muxer` trait with `push()` / `pull()` model
- [x] Define `MuxerInput`, `MuxerOutput`, `PadInfo` types
- [x] Implement `MuxerSyncState` for PTS synchronization
- [x] Implement `TsMuxElement` wrapping existing `TsMux`
- [x] Add `MuxerAdapter` for `AsyncElementDyn`
- [x] Update executor with `run_muxer_node()` for multi-input
- [x] Implement strict/loose/timed/auto sync modes
- [x] Write unit tests (37 tests in muxer.rs and ts_element.rs)
- [x] Create example 39 (muxer element)
- [x] Update documentation

### Plan 04: Pipeline Buffer Pool ✅
- [x] Define `BufferPool` trait
- [x] Implement `PooledBuffer` with Drop return-to-pool
- [x] Implement `FixedSizePool` using `CpuArena`
- [x] Add pool to `ProduceContext`
- [x] Add `set_pool()` and `create_pool()` to `Pipeline`
- [x] Update executor to pass pool to sources
- [x] Write unit tests
- [x] Create example 32 (buffer pool)
- [x] Update documentation

### Plan 05: Element Trait Consolidation ✅
- [x] Define `ProcessOutput` unified enum
- [x] Define `PipelineElement` async trait
- [x] Define `SimpleSource`, `SimpleSink`, `SimpleTransform` traits
- [x] Implement `Src<T>`, `Snk<T>`, `Xfm<T>` wrapper types
- [x] Add `PipelineElementAdapter` for legacy compatibility
- [x] Add `add_element()` method to `Pipeline`
- [x] Executor works via `PipelineElementAdapter` bridge
- [x] Create example 40 (unified elements)
- [x] Update documentation

### Plan 06: Caps Negotiation ✅
- [x] Define `PixelFormat` enum
- [x] Define `VideoCaps` with constraints
- [x] Define `SampleFormat` and `AudioCaps`
- [x] Define `MediaCaps` unified type
- [x] Implement `can_intersect()` and `intersect()`
- [x] Implement `ConverterRegistry`
- [x] Implement `YuvToRgbConverter` and `RgbToYuvConverter`
- [x] Add `negotiate_with_converters()` to `Pipeline`
- [x] Write unit tests
- [x] Create example 35
- [x] Update documentation

### Plan 07: Pipeline Builder DSL ✅
- [x] Implement `PipelineBuilder` with state markers
- [x] Implement `source()`, `then()`, `sink()` methods
- [x] Implement `TeeBuilder` for branching
- [x] Implement `>>` operator via `Shr` trait
- [x] Write unit tests
- [x] Create example 36
- [x] Update documentation

### Plan 08: Events and Tagging ✅
- [x] Define `Event` enum with all event types
- [x] Define `StreamStartEvent`, `SegmentEvent`, `SeekEvent`, etc.
- [x] Define `TagList` and `TagValue`
- [x] Define `PipelineItem` (Buffer | Event)
- [x] Add `handle_upstream_event()` to element traits
- [x] Add `handle_downstream_event()` to element traits
- [x] Write unit tests (20 tests for events/tags)
- [x] Create examples 37 and 38
- [x] Update documentation

</details>

---

## Master Checklist (Phase 2)

### Plan 09: Format Converters ✅
- [x] Create `src/converters/mod.rs` module
- [x] Implement `src/converters/colorspace.rs` (I420 ↔ RGB)
- [x] Implement `src/converters/scale.rs` (nearest, bilinear)
- [x] Implement `src/converters/audio.rs` (S16 ↔ F32)
- [x] Implement `src/converters/resample.rs` (sample rate)
- [x] Update `src/negotiation/builtin.rs` to use real converters
- [x] Create element wrappers: `AudioConvertElement`, `AudioResampleElement`
- [x] Create example: `examples/41_format_converters.rs`
- [x] Update plan documentation

### Plan 10: Code Cleanup ✅
- [x] Fix compiler warnings with `cargo fix`
- [x] Run and fix `cargo clippy -- -D warnings`
- [x] Process all TODO comments (converted to NOTEs)
- [x] Run `cargo doc --no-deps` and fix warnings
- [x] Remove dead code

### Plan 11: GPU Codec Framework ⬜
- [ ] Add `ash`, `gpu-allocator` dependencies
- [ ] Create `src/gpu/mod.rs` module structure
- [ ] Implement Vulkan instance/device creation
- [ ] Implement DMA-BUF import/export
- [ ] Implement H.264 decode
- [ ] Implement H.265 decode
- [ ] Implement AV1 decode
- [ ] Implement H.264 encode
- [ ] Implement H.265 encode
- [ ] Create `HwDecoderElement`/`HwEncoderElement`
- [ ] Create examples: `18_gpu_decode.rs`, `19_gpu_transcode.rs`
- [ ] Update documentation

### Plan 12: Additional Codecs ⬜
- [ ] Add `audiopus` dependency (feature-gated)
- [ ] Implement `OpusEncoder` and `OpusDecoder`
- [ ] Implement `OpusEncElement` and `OpusDecElement`
- [ ] Add `fdk-aac` dependency (feature-gated)
- [ ] Implement `AacEncoder`
- [ ] Verify dav1d VP9 support
- [ ] Create examples: `20_opus_audio.rs`, `21_vp9_decode.rs`
- [ ] Update documentation

### Plan 13: Device Elements ✅
- [x] Add `pipewire` dependency (feature-gated)
- [x] Implement `PipeWireSrc` and `PipeWireSink`
- [x] Add `libcamera` dependency (feature-gated)
- [x] Implement `LibCameraSrc` with camera enumeration
- [x] Add `v4l` dependency (feature-gated)
- [x] Implement `V4l2Src` with device enumeration
- [x] Add `alsa` dependency (feature-gated)
- [x] Implement `AlsaSrc` and `AlsaSink`
- [x] Create unified device enumeration API
- [x] Create examples: `22_v4l2_capture.rs`, `23_v4l2_display.rs`
- [x] Create documentation examples: `42_pipewire_audio.rs`, `43_alsa_audio.rs`, `44_libcamera_capture.rs`
- [x] Update documentation
- [x] Add DMA-BUF export mode for V4L2 (`V4l2Config::dmabuf_export`)
- [x] Screen capture via XDG portal (`ScreenCaptureSrc`, example `46_screen_capture.rs`)

### Plan 14: Streaming Protocols ⬜
- [ ] Implement `HlsSink` with M3U8 generation
- [ ] Implement segment rotation
- [ ] Add `quick-xml` dependency
- [ ] Implement `DashSink` with MPD generation
- [ ] Implement multi-rendition ABR pipeline
- [ ] Create examples: `25_hls_output.rs`, `26_dash_output.rs`
- [ ] Update documentation

---

## Breaking Changes Policy

**We can break backward compatibility freely** - the codebase is pre-production.

This means:
- No deprecation periods required
- Direct API changes without aliases
- All examples updated together with changes
- Changes documented in CHANGELOG.md

---

## Testing Strategy

Each plan should include:

1. **Unit tests** for new types and functions
2. **Integration tests** for element interactions
3. **Example updates** demonstrating new features
4. **Benchmark tests** for performance-sensitive changes

Run tests with:
```bash
just test              # All tests
just test-one NAME     # Specific test
cargo test --doc       # Doctests only
```

---

## Documentation Updates

After completing plans, update:

- [ ] `CLAUDE.md` - Architecture section, key types, roadmap status
- [ ] `docs/getting-started.md` - New features, updated examples
- [ ] `README.md` (root) - Feature list if user-facing changes
- [ ] Rustdoc comments - `///` on all public APIs
- [ ] `plans/README.md` - Progress checkboxes

---

## Contributing

When implementing a plan:

1. Create a feature branch: `git checkout -b plan-09-format-converters`
2. Follow the implementation steps in the plan
3. Run all tests: `just test`
4. Update the checkboxes in this README
5. Update the plan status in the Progress Overview table
6. Create a PR with the plan number in the title

---

*Phase 1 plans created January 2026 based on [Development Report](../docs/DEVELOPMENT_REPORT.md)*  
*Phase 2 plans created January 2026 based on [Project Analysis Report](../docs/PROJECT_ANALYSIS_REPORT.md)*
