# Plan 11: Pipeline Robustness and Performance

**Status**: Draft  
**Priority**: High  
**Based on**: `docs/reports/2026-01-29-screen-capture-mp4-pipeline.md`  
**Estimated Effort**: 3-4 weeks

---

## Goal

Make Parallax pipelines robust for real-time media processing by addressing:
1. Memory management (backpressure, arena pooling)
2. Encoder performance (hardware acceleration)
3. Pipeline reliability (state machine, error recovery)

---

## Overview

| Phase | Description | Effort | Priority |
|-------|-------------|--------|----------|
| 1 | Arena Hygiene | 2 days | Critical |
| 2 | Backpressure System | 3-4 days | Critical |
| 3 | Video Scaler Element | 2 days | High |
| 4 | Hardware Encoders | 5-7 days | High |
| 5 | Pipeline Observability | 2-3 days | Medium |
| 6 | Fragmented MP4 | 2-3 days | Medium |

**Total: 16-21 days**

---

## Phase 1: Arena Hygiene (Critical)

**Goal**: Ensure all elements properly manage arena memory.

### 1.1 Audit and Fix All Elements

Search for `arena.acquire()` without preceding `arena.reclaim()`:

| File | Status |
|------|--------|
| `src/elements/codec/h264.rs` | Fixed |
| `src/elements/codec/encoder.rs` | Needs fix |
| `src/elements/codec/decoder.rs` | Needs fix |
| `src/elements/codec/encoder_element.rs` | Needs fix |
| `src/elements/codec/decoder_element.rs` | Needs fix |
| `src/elements/codec/audio.rs` | Needs fix |
| `src/elements/codec/image.rs` | Needs fix |
| `src/elements/rtp/rtp.rs` | Needs fix |
| `src/elements/network/zenoh.rs` | Needs fix |

**Action**: Add `arena.reclaim()` before every `arena.acquire()` call.

### 1.2 Standardize Arena Sizes

Create constants for default arena sizes:

```rust
// src/memory/defaults.rs
pub mod arena_defaults {
    /// Small buffers (RTP packets, audio frames)
    pub const SMALL_SLOT_COUNT: usize = 64;
    pub const SMALL_SLOT_SIZE: usize = 64 * 1024;  // 64KB
    
    /// Medium buffers (encoded video frames)
    pub const MEDIUM_SLOT_COUNT: usize = 64;
    pub const MEDIUM_SLOT_SIZE: usize = 1024 * 1024;  // 1MB
    
    /// Large buffers (raw video frames)
    pub const LARGE_SLOT_COUNT: usize = 32;
    pub const LARGE_SLOT_SIZE: usize = 8 * 1024 * 1024;  // 8MB
    
    /// Source buffers (need more slots for buffering)
    pub const SOURCE_SLOT_COUNT: usize = 200;
}
```

### 1.3 Add Arena Usage Metrics

```rust
impl SharedArena {
    /// Get current arena utilization (0.0 - 1.0)
    pub fn utilization(&self) -> f32 {
        let used = self.slot_count() - self.available_slots();
        used as f32 / self.slot_count() as f32
    }
    
    /// Get number of available slots
    pub fn available_slots(&self) -> usize;
    
    /// Get high-water mark (max slots ever used)
    pub fn high_water_mark(&self) -> usize;
}
```

### 1.4 Integration Tests

```rust
#[test]
fn test_arena_not_exhausted_under_load() {
    // Run pipeline for 1000 frames
    // Verify arena never exhausts
    // Verify utilization stays below 80%
}

#[test]
fn test_mp4_duration_correct() {
    // Create MP4 with known frame count
    // Verify duration matches expected
}
```

**Deliverables**:
- [ ] All elements call `reclaim()` before `acquire()`
- [ ] Arena size constants in `src/memory/defaults.rs`
- [ ] `utilization()` and `available_slots()` methods
- [ ] Integration tests for arena behavior

---

## Phase 2: Backpressure System (Critical)

**Goal**: Prevent memory exhaustion when consumers are slower than producers.

### 2.1 Design Decision

**Chosen approach**: Hybrid pull/push with flow signals

```
Producer ──data──> Queue ──data──> Consumer
    ^                                  │
    └────── FlowSignal::Busy ──────────┘
```

When downstream is busy:
1. Queue fills to high-water mark
2. Queue sends `FlowSignal::Busy` upstream
3. Source either blocks or drops frames
4. When queue drains to low-water mark, sends `FlowSignal::Ok`

### 2.2 Flow Control Types

```rust
// src/pipeline/flow.rs

/// Flow control signal from downstream to upstream
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlowSignal {
    /// Normal operation, continue producing
    Ok,
    /// Downstream is busy, reduce rate or buffer
    Busy,
    /// Drop frames until further notice
    Drop,
    /// End of stream acknowledged
    Eos,
}

/// Flow control policy for sources
#[derive(Debug, Clone)]
pub enum FlowPolicy {
    /// Block when downstream is busy (default for file sources)
    Block,
    /// Drop frames when downstream is busy (default for live sources)
    Drop { 
        /// Log when dropping
        log_drops: bool,
        /// Max consecutive drops before error
        max_consecutive: Option<u32>,
    },
    /// Buffer up to limit, then drop oldest
    RingBuffer { 
        capacity: usize,
    },
}
```

### 2.3 Source Trait Extension

```rust
pub trait Source: Send {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult>;
    
    /// Handle flow control signal from downstream
    fn handle_flow_signal(&mut self, signal: FlowSignal) {
        // Default: ignore (for backward compatibility)
    }
    
    /// Get flow policy for this source
    fn flow_policy(&self) -> FlowPolicy {
        FlowPolicy::Block  // Safe default
    }
}
```

### 2.4 Queue Element with Flow Control

```rust
// src/elements/flow/queue.rs

pub struct Queue {
    buffer: VecDeque<Buffer>,
    capacity: usize,
    high_water: usize,  // 80% of capacity
    low_water: usize,   // 20% of capacity
    upstream_signal: FlowSignal,
}

impl Queue {
    fn should_signal_busy(&self) -> bool {
        self.buffer.len() >= self.high_water
    }
    
    fn should_signal_ok(&self) -> bool {
        self.buffer.len() <= self.low_water && 
        self.upstream_signal == FlowSignal::Busy
    }
}
```

### 2.5 Executor Integration

```rust
// In unified_executor.rs

async fn run_source(&mut self, source: &mut dyn Source) {
    loop {
        // Check flow signal
        if self.flow_signal == FlowSignal::Busy {
            match source.flow_policy() {
                FlowPolicy::Block => {
                    // Wait for signal to clear
                    self.wait_for_flow_ok().await;
                }
                FlowPolicy::Drop { log_drops, .. } => {
                    if log_drops {
                        tracing::debug!("Dropping frame due to backpressure");
                    }
                    continue;  // Skip this iteration
                }
                FlowPolicy::RingBuffer { .. } => {
                    // Handled by queue
                }
            }
        }
        
        // Normal production
        let result = source.produce(&mut ctx)?;
        // ...
    }
}
```

### 2.6 Update ScreenCaptureSrc

```rust
impl Source for ScreenCaptureSrc {
    fn flow_policy(&self) -> FlowPolicy {
        FlowPolicy::Drop {
            log_drops: true,
            max_consecutive: Some(30),  // Error after 1 second of drops
        }
    }
    
    fn handle_flow_signal(&mut self, signal: FlowSignal) {
        self.current_flow_signal = signal;
    }
}
```

**Deliverables**:
- [ ] `FlowSignal` and `FlowPolicy` types
- [ ] `handle_flow_signal()` in Source trait
- [ ] Queue element with high/low water marks
- [ ] Executor flow signal propagation
- [ ] ScreenCaptureSrc with drop policy
- [ ] Tests for backpressure behavior

---

## Phase 3: Video Scaler Element (High)

**Goal**: Reduce resolution to improve encoder performance.

### 3.1 Scaler Implementation

```rust
// src/elements/transform/videoscale.rs

pub struct VideoScaler {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    algorithm: ScaleAlgorithm,
    arena: Option<SharedArena>,
}

#[derive(Debug, Clone, Copy)]
pub enum ScaleAlgorithm {
    NearestNeighbor,  // Fastest, lowest quality
    Bilinear,         // Good balance
    Lanczos,          // Best quality, slowest
}

impl VideoScaler {
    pub fn new(in_w: u32, in_h: u32, out_w: u32, out_h: u32) -> Self;
    
    /// Scale to fit within max dimensions, preserving aspect ratio
    pub fn fit(in_w: u32, in_h: u32, max_w: u32, max_h: u32) -> Self;
    
    /// Scale by factor (0.5 = half size)
    pub fn by_factor(in_w: u32, in_h: u32, factor: f32) -> Self;
}
```

### 3.2 Pure Rust Implementation

Use `fast_image_resize` crate for SIMD-optimized scaling:

```toml
[dependencies]
fast_image_resize = "2.7"
```

```rust
impl Element for VideoScaler {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let src = Image::from_slice_u8(
            self.input_width, 
            self.input_height,
            buffer.as_bytes(),
            PixelType::U8x4,  // RGBA
        )?;
        
        let mut dst = Image::new(
            self.output_width,
            self.output_height,
            PixelType::U8x4,
        );
        
        let mut resizer = Resizer::new(self.algorithm.into());
        resizer.resize(&src.view(), &mut dst.view_mut())?;
        
        // Copy to arena slot
        // ...
    }
}
```

### 3.3 Update Screen Capture Example

```rust
// examples/46_screen_capture.rs

// Scale down to 720p for faster encoding
let scaler = VideoScaler::fit(1920, 1080, 1280, 720);

pipeline.link(src, scaler)?;
pipeline.link(scaler, convert)?;  // Then to converter
```

**Deliverables**:
- [ ] `VideoScaler` element with multiple algorithms
- [ ] I420 and RGBA format support
- [ ] Aspect ratio preservation
- [ ] Updated screen capture example
- [ ] Benchmarks comparing scale factors

---

## Phase 4: Hardware Encoders (High)

**Goal**: Enable real-time encoding via hardware acceleration.

### 4.1 Encoder Abstraction

```rust
// src/elements/codec/encoder_backend.rs

pub trait VideoEncoderBackend: Send {
    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<u8>>;
    fn flush(&mut self) -> Result<Vec<Vec<u8>>>;
    fn codec(&self) -> VideoCodec;
}

pub struct VideoEncoderConfig {
    pub width: u32,
    pub height: u32,
    pub framerate: f32,
    pub bitrate: u32,
    pub codec: VideoCodec,
    pub backend: EncoderBackend,
}

pub enum EncoderBackend {
    Auto,           // Detect best available
    Software,       // OpenH264
    Vaapi,          // Intel/AMD
    Nvenc,          // NVIDIA
    V4l2M2M,        // Embedded (RPi, etc.)
    VulkanVideo,    // Future
}
```

### 4.2 VA-API Encoder (Intel/AMD)

```rust
// src/elements/codec/vaapi.rs (feature = "vaapi")

pub struct VaapiH264Encoder {
    display: VaDisplay,
    context: VaContext,
    surfaces: Vec<VaSurface>,
    config: VideoEncoderConfig,
}

impl VaapiH264Encoder {
    pub fn new(config: VideoEncoderConfig) -> Result<Self> {
        // Open DRM device
        let drm_fd = open("/dev/dri/renderD128")?;
        let display = VaDisplay::from_drm(drm_fd)?;
        
        // Query H.264 encode support
        let profiles = display.query_config_profiles()?;
        if !profiles.contains(&VAProfileH264Main) {
            return Err(Error::NotSupported("VA-API H.264 not available"));
        }
        
        // Create context and surfaces
        // ...
    }
}

impl VideoEncoderBackend for VaapiH264Encoder {
    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<u8>> {
        // Upload frame to VA surface
        // Execute encode pipeline
        // Read back bitstream
    }
}
```

### 4.3 Auto-Detection

```rust
// src/elements/codec/mod.rs

pub fn create_encoder(config: VideoEncoderConfig) -> Result<Box<dyn VideoEncoderBackend>> {
    match config.backend {
        EncoderBackend::Auto => {
            // Try in order of preference
            #[cfg(feature = "vaapi")]
            if let Ok(enc) = VaapiH264Encoder::new(config.clone()) {
                tracing::info!("Using VA-API encoder");
                return Ok(Box::new(enc));
            }
            
            #[cfg(feature = "nvenc")]
            if let Ok(enc) = NvencH264Encoder::new(config.clone()) {
                tracing::info!("Using NVENC encoder");
                return Ok(Box::new(enc));
            }
            
            // Fallback to software
            tracing::info!("Using software encoder (OpenH264)");
            Ok(Box::new(OpenH264Encoder::new(config)?))
        }
        EncoderBackend::Vaapi => {
            #[cfg(feature = "vaapi")]
            return Ok(Box::new(VaapiH264Encoder::new(config)?));
            #[cfg(not(feature = "vaapi"))]
            return Err(Error::NotSupported("VA-API not compiled"));
        }
        // ... other backends
    }
}
```

### 4.4 Feature Flags

```toml
# Cargo.toml
[features]
vaapi = ["libva", "drm"]
nvenc = ["nvidia-video-codec-sdk"]
v4l2-m2m = ["v4l"]
```

**Deliverables**:
- [ ] `VideoEncoderBackend` trait
- [ ] VA-API encoder implementation
- [ ] Auto-detection logic
- [ ] Feature flags for optional backends
- [ ] Documentation for hardware requirements

---

## Phase 5: Pipeline Observability (Medium)

**Goal**: Monitor pipeline health and performance.

### 5.1 Statistics Collection

```rust
// src/pipeline/stats.rs

#[derive(Debug, Clone, Default)]
pub struct ElementStats {
    /// Buffers processed
    pub buffers_in: u64,
    pub buffers_out: u64,
    
    /// Bytes processed
    pub bytes_in: u64,
    pub bytes_out: u64,
    
    /// Timing
    pub total_processing_time: Duration,
    pub last_buffer_time: Option<Instant>,
    
    /// Calculated rates
    pub input_fps: f32,
    pub output_fps: f32,
    pub throughput_mbps: f32,
    
    /// Errors
    pub errors: u64,
    pub drops: u64,
}

#[derive(Debug, Clone)]
pub struct PipelineStats {
    pub elements: HashMap<String, ElementStats>,
    pub total_latency: Duration,
    pub memory_used: usize,
    pub running_time: Duration,
}
```

### 5.2 Stats API

```rust
impl Pipeline {
    /// Get current pipeline statistics
    pub fn stats(&self) -> PipelineStats;
    
    /// Subscribe to stats updates
    pub fn stats_stream(&self, interval: Duration) -> impl Stream<Item = PipelineStats>;
    
    /// Get stats for specific element
    pub fn element_stats(&self, name: &str) -> Option<ElementStats>;
}
```

### 5.3 Progress Reporting

```rust
#[derive(Debug, Clone)]
pub struct Progress {
    pub frames_processed: u64,
    pub frames_total: Option<u64>,
    pub bytes_written: u64,
    pub elapsed: Duration,
    pub estimated_remaining: Option<Duration>,
}

impl Pipeline {
    pub async fn run_with_progress<F>(&mut self, callback: F) -> Result<()>
    where
        F: FnMut(Progress) + Send + 'static;
}
```

**Deliverables**:
- [ ] `ElementStats` and `PipelineStats` types
- [ ] Stats collection in executor
- [ ] `stats()` and `stats_stream()` methods
- [ ] Progress callback support
- [ ] Example with live stats display

---

## Phase 6: Fragmented MP4 (Medium)

**Goal**: Enable streaming via fragmented MP4 output.

### 6.1 fMP4 Muxer

```rust
// src/elements/mux/fmp4.rs

pub struct FragmentedMp4Mux {
    fragment_duration: Duration,
    current_fragment: Vec<Sample>,
    fragment_sequence: u32,
    init_segment: Option<Vec<u8>>,
}

pub struct FragmentedMp4Config {
    /// Duration of each fragment
    pub fragment_duration: Duration,
    /// Output initialization segment separately
    pub separate_init: bool,
}

impl Element for FragmentedMp4Mux {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.current_fragment.push(sample);
        
        if self.fragment_duration_reached() {
            let fragment = self.flush_fragment()?;
            return Ok(Some(fragment));
        }
        
        Ok(None)
    }
}
```

### 6.2 DASH/HLS Output

```rust
// Future: integrate with DASH/HLS segment writers
pub struct DashWriter {
    manifest: DashManifest,
    segment_dir: PathBuf,
    mux: FragmentedMp4Mux,
}
```

**Deliverables**:
- [ ] `FragmentedMp4Mux` element
- [ ] Init segment + media segment separation
- [ ] Configurable fragment duration
- [ ] Example streaming to network sink

---

## Verification

### Phase 1 Tests
```bash
cargo test arena_ --features "h264 mp4-demux"
cargo test --test pipeline_integration memory
```

### Phase 2 Tests
```bash
cargo test backpressure --features "h264"
cargo test flow_control
```

### Phase 3 Tests
```bash
cargo test videoscale
cargo run --example 46_screen_capture --features "screen-capture h264 mp4-demux" --release
# Should encode at >15 fps with 720p output
```

### Phase 4 Tests
```bash
# Requires hardware
cargo test vaapi --features "vaapi" -- --ignored
cargo run --example 46_screen_capture --features "screen-capture vaapi" --release
# Should encode at 30+ fps real-time
```

### Phase 5 Tests
```bash
cargo test pipeline_stats
cargo run --example pipeline_monitor
```

### Phase 6 Tests
```bash
cargo test fmp4
cargo run --example hls_streaming --features "h264 fmp4"
```

---

## Success Criteria

1. **Phase 1**: No arena exhaustion in 10-minute pipeline run
2. **Phase 2**: ScreenCapture drops <5% frames with slow encoder
3. **Phase 3**: 720p encoding at >15 fps (software)
4. **Phase 4**: 1080p encoding at 30 fps (VA-API)
5. **Phase 5**: Live FPS display in example
6. **Phase 6**: Playable fMP4 fragments

---

## Dependencies

| Phase | External Crates | System Requirements |
|-------|-----------------|---------------------|
| 1 | - | - |
| 2 | - | - |
| 3 | `fast_image_resize` | - |
| 4 | `libva`, `drm` | Intel/AMD GPU with VA-API |
| 5 | - | - |
| 6 | - | - |

---

## Risks

1. **VA-API complexity**: Driver quirks across Intel/AMD generations
2. **Backpressure design**: May need iteration to get right
3. **Memory overhead**: Stats collection adds small overhead

---

## Timeline

| Week | Phase | Milestone |
|------|-------|-----------|
| 1 | 1, 2 | Arena fixes, backpressure design |
| 2 | 2, 3 | Backpressure impl, video scaler |
| 3 | 4 | VA-API encoder |
| 4 | 4, 5, 6 | Encoder polish, stats, fMP4 |

---

## References

- PipeWire flow control: https://docs.pipewire.org/
- VA-API documentation: https://github.com/intel/libva
- GStreamer backpressure: https://gstreamer.freedesktop.org/documentation/
- fMP4 spec: ISO/IEC 14496-12 (ISOBMFF)
