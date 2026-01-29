# Development Report: Screen Capture to MP4 Pipeline

**Date**: 2026-01-29  
**Author**: Claude  
**Scope**: Screen capture via XDG Desktop Portal/PipeWire, H.264 encoding, MP4 muxing

---

## Executive Summary

Successfully implemented an end-to-end screen capture to MP4 pipeline:

```
ScreenCapture → VideoConvert → H264Encoder → Mp4Mux → FileSink
```

The pipeline captures screen content via XDG Desktop Portal, converts BGRA to I420, encodes to H.264, muxes to MP4 container, and writes to disk. A 5-second capture produces a ~2MB playable MP4 file at 1920x1080 resolution.

---

## What Was Built

### 1. Screen Capture Source (`src/elements/device/screen_capture.rs`)

- XDG Desktop Portal integration for Wayland/X11 screen capture
- PipeWire stream handling with proper ThreadLoop pattern
- SPA pod format negotiation for video formats
- Lazy arena initialization based on actual frame size

### 2. MP4 Muxer (`src/elements/mux/mp4.rs`)

- `Mp4Mux<W>`: Low-level muxer wrapping the `mp4` crate
- `Mp4FileSink`: Sink element that writes directly to file
- `Mp4MuxTransform`: Transform element that buffers frames and outputs complete MP4 on flush
- Annex-B to AVCC format conversion for H.264
- SPS/PPS extraction from H.264 stream

### 3. Video Format Converter (`src/elements/transform/videoconvert.rs`)

- BGRA → I420 (YUV420) conversion for encoder input
- Explicit dimension configuration via `with_size()`

---

## Pain Points Encountered

### 1. PipeWire Integration Complexity

**Problem**: The PipeWire process callback was never invoked despite the stream reaching "Streaming" state.

**Root Cause**: Incorrect initialization order. PipeWire's portal screen capture requires a specific pattern:
1. Start ThreadLoop
2. Lock ThreadLoop
3. Connect to core via fd
4. Create stream and register listeners
5. Connect stream to node
6. Unlock ThreadLoop (not start again)

**Solution**: Studied OBS Studio's implementation and replicated their pattern. The key insight is that `start()` must happen BEFORE `lock()`, not after.

**Time Spent**: ~2 hours debugging

**Recommendation**: Document the PipeWire ThreadLoop pattern prominently. Consider creating a `PipeWireHelper` abstraction that enforces correct usage.

### 2. MP4 Duration Was Zero

**Problem**: VLC showed 0-second duration, video displayed only first frame then jumped to last.

**Root Cause**: The `mp4` crate's `Mp4Sample` requires a non-zero `duration` field, but we were setting `duration: 0` expecting the crate to calculate it.

**Solution**: Added `duration_ms` parameter to all write methods:
```rust
// Before
mux.write_video_sample(track, &data, pts_ms, is_keyframe)?;

// After  
mux.write_video_sample(track, &data, pts_ms, duration_ms, is_keyframe)?;
```

**Time Spent**: ~30 minutes

**Recommendation**: Add integration tests that verify MP4 duration is correct using `gst-discoverer` or similar.

### 3. Arena Exhaustion

**Problem**: Pipeline failed with "No slots available in arena" after capturing only 8 frames.

**Root Cause**: Multiple issues:
1. Screen capture source created internal arena with only 8 slots
2. H264Encoder never called `arena.reclaim()`, so slots were never reused
3. Encoder is much slower than capture rate (capture: 30fps, encode: ~2fps)

**Solution**: 
- Increased screen capture arena to 200 slots
- Added `arena.reclaim()` before acquiring in H264Encoder
- Increased encoder arena to 64 slots

**Time Spent**: ~1 hour

**Recommendation**: This is a fundamental architecture issue. See "Architecture Issues" below.

### 4. kanal API Mismatch

**Problem**: Compilation errors with channel receive operations.

**Root Cause**: `kanal::try_recv()` returns `Result<Option<T>, ReceiveError>`, not `Result<T, ReceiveError>`.

**Solution**: Fixed pattern matching to handle the Option wrapper.

**Time Spent**: ~15 minutes

---

## Architecture Issues

### Issue 1: No Backpressure Mechanism

**Severity**: High

**Description**: Currently, sources produce data as fast as possible. If downstream elements are slower, the only options are:
1. Buffer indefinitely (memory exhaustion)
2. Fail when arena is exhausted
3. Drop frames silently

**Current Workaround**: Large arenas (200+ slots for screen capture = 1.6GB for 1080p)

**Recommended Solution**: Implement proper backpressure:

```rust
// Option A: Blocking acquire with timeout
let slot = arena.acquire_blocking(Duration::from_millis(100))?;

// Option B: Source-side frame dropping
if arena.available_slots() < threshold {
    // Skip this frame
    return Ok(ProduceResult::WouldBlock);
}

// Option C: Flow control signals
enum FlowSignal {
    Ok,
    Busy,  // Downstream is slow, reduce rate
    Drop,  // Drop frames until signal clears
}
```

**Impact**: Without backpressure, real-time pipelines will either consume excessive memory or fail unpredictably.

### Issue 2: Elements Create Internal Arenas

**Severity**: Medium

**Description**: Many elements create their own internal arenas:
- `ScreenCaptureSrc`: 200 slots × 8MB = 1.6GB
- `H264Encoder`: 64 slots × 1MB = 64MB
- `VideoConvert`: Uses input buffer's arena
- `Mp4MuxTransform`: Creates arena on flush

This leads to:
1. Memory fragmentation
2. No global memory management
3. Difficult to reason about total memory usage
4. `add_source_with_arena()` is ignored by some sources

**Recommended Solution**: Centralized arena management:

```rust
// Pipeline-level arena pool
let pipeline = Pipeline::new()
    .with_memory_budget(2 * 1024 * 1024 * 1024)  // 2GB total
    .with_arena_policy(ArenaPolicy::SharedPerStage);

// Elements request memory from pipeline
impl Element for MyElement {
    fn prepare(&mut self, ctx: &mut PrepareContext) -> Result<()> {
        self.arena = ctx.request_arena(slot_size, slot_count)?;
        Ok(())
    }
}
```

### Issue 3: Encoder Performance

**Severity**: Medium

**Description**: OpenH264 (pure Rust) encodes 1080p at ~2 fps. This is 15x slower than real-time 30fps capture.

**Impact**: 
- 5 seconds of capture takes ~75 seconds to encode
- Requires massive buffering
- Not suitable for real-time streaming

**Recommended Solutions**:

1. **Short-term**: Add resolution scaling before encoding
   ```rust
   let scaler = VideoScaler::new(1920, 1080, 640, 360);  // 6x fewer pixels
   ```

2. **Medium-term**: Hardware encoder support
   ```rust
   // VA-API (Intel/AMD)
   let encoder = VaapiH264Encoder::new(config)?;
   
   // NVENC (NVIDIA)  
   let encoder = NvencH264Encoder::new(config)?;
   
   // V4L2 M2M (Raspberry Pi, etc.)
   let encoder = V4l2M2MEncoder::new("/dev/video11", config)?;
   ```

3. **Long-term**: Vulkan Video encode (as per existing design docs)

### Issue 4: No Pipeline State Machine Enforcement

**Severity**: Low

**Description**: Elements can be in inconsistent states. There's no enforcement of the `Suspended → Idle → Running` state machine described in CLAUDE.md.

**Impact**: 
- Elements may not release resources properly
- No clean pause/resume semantics
- Difficult to implement seek

**Recommended Solution**: Implement explicit state transitions:

```rust
impl Element for MyElement {
    fn transition(&mut self, from: State, to: State) -> Result<()> {
        match (from, to) {
            (Suspended, Idle) => self.prepare()?,
            (Idle, Running) => self.activate()?,
            (Running, Idle) => self.pause()?,
            (Idle, Suspended) => self.release()?,
            _ => return Err(Error::InvalidTransition),
        }
        Ok(())
    }
}
```

---

## Missing Features

### 1. Frame Dropping / Rate Limiting

**Priority**: High

**Description**: Sources should be able to drop frames when downstream can't keep up.

```rust
pub struct ScreenCaptureConfig {
    // ...
    /// Drop frames if arena utilization exceeds this threshold (0.0-1.0)
    pub drop_threshold: Option<f32>,
    /// Target frame rate (drop frames to match)
    pub target_fps: Option<f32>,
}
```

### 2. Progress Reporting

**Priority**: Medium

**Description**: No way to know pipeline progress during long operations.

```rust
// Current: blocks until done
pipeline.run().await?;

// Proposed: progress callbacks
pipeline.run_with_progress(|progress| {
    println!("Frame {}/{}", progress.frames_processed, progress.total_frames);
}).await?;
```

### 3. Hardware Encoder Detection

**Priority**: Medium

**Description**: Automatically select best available encoder.

```rust
let encoder = VideoEncoder::auto_detect(config)?;
// Returns: VaapiEncoder, NvencEncoder, or SoftwareEncoder
```

### 4. Fragmented MP4 (fMP4) for Streaming

**Priority**: Medium

**Description**: Current MP4 muxer buffers everything in memory. For streaming, need fragmented MP4.

```rust
let mux = FragmentedMp4Mux::new(config)
    .fragment_duration(Duration::from_secs(2));

// Outputs fragments as they're ready, not at end
```

### 5. Pipeline Introspection

**Priority**: Low

**Description**: Debug/monitor pipeline state at runtime.

```rust
let stats = pipeline.stats();
println!("Source: {} fps, Encoder: {} fps, Queue depth: {}", 
    stats["screen_capture"].fps,
    stats["h264enc"].fps,
    stats["queue"].depth);
```

---

## Recommendations Summary

### Immediate (This Sprint)

1. **Add `arena.reclaim()` calls** to all elements that use internal arenas
2. **Add integration tests** for MP4 output validation
3. **Document PipeWire ThreadLoop pattern** in code comments

### Short-term (Next Sprint)

1. **Implement frame dropping** in ScreenCaptureSrc when arena is filling
2. **Add video scaler element** to reduce resolution before encoding
3. **Centralize arena creation** - elements should request from pipeline

### Medium-term (Next Month)

1. **Design backpressure system** - flow control between elements
2. **Add VA-API encoder** for Intel/AMD hardware acceleration
3. **Implement fragmented MP4** for streaming use cases

### Long-term (Next Quarter)

1. **Implement Vulkan Video** as per existing design docs
2. **Add pipeline state machine** with proper transitions
3. **Create arena pool manager** for global memory management

---

## Test Results

### Final Pipeline Output

```
Duration: 4.092 seconds
Resolution: 1920x1080
Frame rate: ~30 fps
Bitrate: 4.1 Mbps
File size: 2.1 MB
Frames decoded: 124
```

### Verification Commands

```bash
# Check file properties
gst-discoverer-1.0 screen_capture.mp4

# Verify playback
gst-launch-1.0 filesrc location=screen_capture.mp4 ! \
    qtdemux ! h264parse ! avdec_h264 ! videoconvert ! autovideosink

# Count frames
gst-launch-1.0 filesrc location=screen_capture.mp4 ! \
    qtdemux ! h264parse ! avdec_h264 ! fakesink -v 2>&1 | grep -c chain
```

---

## Commits

1. `26516a6` - Fix screen capture pipeline and add MP4 muxer
2. `82a5315` - Fix MP4 sample duration - video now plays correctly
3. `190df0a` - Fix arena exhaustion in screen capture pipeline

---

## Conclusion

The screen capture to MP4 pipeline works but exposed several architectural weaknesses:

1. **Memory management is ad-hoc** - each element manages its own arena
2. **No backpressure** - slow consumers cause memory exhaustion or failures
3. **Encoder performance** - pure-Rust H.264 is too slow for real-time 1080p

The most critical fix needed is a proper backpressure mechanism. Without it, real-time pipelines will always be fragile. The current "large arena" workaround works but wastes memory and doesn't scale.

Hardware encoder support would transform the usability of video pipelines, enabling true real-time screen capture and streaming.
