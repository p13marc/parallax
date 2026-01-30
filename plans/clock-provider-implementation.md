# Clock Provider System Implementation Plan

## Overview

This plan implements a proper clock provider system for Parallax, addressing the fundamental issue that timestamps should represent when media was *captured*, not when our code processed it.

**Goals:**
1. Extract and use hardware timestamps from PipeWire, V4L2, and ALSA
2. Create a `ClockProvider` trait for elements that can provide timing references
3. Integrate the `PipelineClock` with the executor and elements
4. Enable proper A/V synchronization across different sources

---

## Phase 1: Hardware Timestamp Extraction (Short Term)

**Objective:** Use timestamps provided by hardware/drivers instead of `Instant::now()`

### Task 1.1: Extract PipeWire Timestamps in Screen Capture

**File:** `src/elements/device/screen_capture.rs`

**Current State:**
```rust
// In process callback - we use Instant::now() which is WRONG
let now = Instant::now();
let pts = {
    let elapsed = now.duration_since(capture_start);
    ClockTime::from_nanos(elapsed.as_nanos() as u64)
};
```

**Required Change:**
PipeWire buffers contain `spa_meta_header` with proper timestamps. We need to extract it:

```rust
// In the process callback
.process(move |stream, _user_data| {
    let Some(mut buffer) = stream.dequeue_buffer() else { return; };
    
    // Extract PipeWire's timestamp from spa_meta_header
    let pts = extract_pipewire_pts(&buffer);
    
    // ... rest of processing
})
```

**Implementation Details:**
1. Access `buffer.buffer()` to get the underlying `*mut spa_buffer`
2. Use `libspa_sys::spa_buffer_find_meta()` with `SPA_META_Header` type
3. Cast result to `*const spa_meta_header` and read `pts` field
4. Convert nanoseconds to `ClockTime`
5. Fall back to `Instant::now()` if meta not present (shouldn't happen)

**Helper Function:**
```rust
/// Extract PTS from PipeWire buffer's spa_meta_header
fn extract_pipewire_pts(buffer: &pw::stream::Buffer) -> Option<ClockTime> {
    use libspa_sys::{spa_buffer_find_meta, spa_meta_header, SPA_META_Header};
    
    unsafe {
        let spa_buf = buffer.buffer().as_ptr();
        let meta = spa_buffer_find_meta(spa_buf, SPA_META_Header);
        if meta.is_null() {
            return None;
        }
        let header = (*meta).data as *const spa_meta_header;
        if header.is_null() {
            return None;
        }
        let pts_ns = (*header).pts;
        if pts_ns < 0 {
            return None; // Invalid timestamp
        }
        Some(ClockTime::from_nanos(pts_ns as u64))
    }
}
```

**Testing:**
- Log extracted PTS vs Instant-based PTS to compare
- Verify timestamps are monotonically increasing
- Check that timestamps match expected frame intervals (~33ms at 30fps)

---

### Task 1.2: Extract V4L2 Timestamps

**File:** `src/elements/device/v4l2.rs`

**Current State:**
```rust
return Ok(ProduceResult::OwnBuffer(Buffer::new(
    handle,
    Metadata::new(),  // No timestamp!
)));
```

**Required Change:**
V4L2 buffers have timestamps in `v4l2_buffer.timestamp`:

```rust
// After dequeuing buffer
let v4l2_buf: v4l2::buffer::Buffer = /* dequeued */;

// Extract timestamp (microseconds since boot, CLOCK_MONOTONIC)
let timestamp = v4l2_buf.timestamp();
let pts = ClockTime::from_micros(
    timestamp.tv_sec as u64 * 1_000_000 + timestamp.tv_usec as u64
);

// Create metadata with timestamp
let metadata = Metadata::new().with_pts(pts);
return Ok(ProduceResult::OwnBuffer(Buffer::new(handle, metadata)));
```

**Implementation Details:**
1. The `v4l` crate's `Buffer` type has a `timestamp()` method returning `libc::timeval`
2. Convert to microseconds, then to `ClockTime`
3. V4L2 timestamps are typically `CLOCK_MONOTONIC` based (check with `V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC`)

**For DMA-BUF path:**
Same change needed for the DMA-BUF code path.

---

### Task 1.3: Add ALSA Timestamp Support

**File:** `src/elements/device/alsa.rs`

**Current State:**
ALSA source doesn't set any timestamps.

**Required Change:**
Calculate PTS from sample position and sample rate:

```rust
impl AsyncSource for AlsaSrc {
    async fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        // ... read samples ...
        
        // Calculate PTS from total samples produced
        let samples_per_sec = self.format.sample_rate as u64;
        let pts_nanos = (self.total_samples_produced * 1_000_000_000) / samples_per_sec;
        let pts = ClockTime::from_nanos(pts_nanos);
        
        // Or use ALSA status for hardware timestamp
        let status = self.pcm.status()?;
        let hw_timestamp = status.get_htstamp(); // Hardware timestamp if available
        
        ctx.set_pts(pts);
        // ...
    }
}
```

**Implementation Details:**
1. Track `total_samples_produced` counter
2. Calculate PTS as `(samples * 1e9) / sample_rate`
3. Optionally use `pcm.status().get_htstamp()` for hardware timestamps
4. Set via `ctx.set_pts()` or `Metadata::new().with_pts()`

---

### Task 1.4: Update PipeWire Audio Source

**File:** `src/elements/device/pipewire.rs` (if exists)

Same approach as screen capture - extract `spa_meta_header.pts`.

---

## Phase 2: Clock Provider Trait (Medium Term)

**Objective:** Allow elements to provide clocks for pipeline-wide synchronization

### Task 2.1: Define Clock Provider Traits

**File:** `src/clock.rs` (extend existing)

```rust
use std::sync::Arc;

/// Capabilities of a clock
bitflags::bitflags! {
    pub struct ClockFlags: u32 {
        /// Clock can be used as pipeline master
        const CAN_BE_MASTER = 0x01;
        /// Clock can slave to another clock
        const CAN_SET_MASTER = 0x02;
        /// Clock provides hardware timestamps
        const HARDWARE = 0x04;
        /// Clock is from network source (PTP, NTP)
        const NETWORK = 0x08;
        /// Clock is real-time (audio device)
        const REALTIME = 0x10;
    }
}

/// A clock source for timing
pub trait Clock: Send + Sync {
    /// Get current time in nanoseconds
    fn now(&self) -> ClockTime;
    
    /// Clock capabilities
    fn flags(&self) -> ClockFlags;
    
    /// Resolution in nanoseconds (0 = unknown)
    fn resolution(&self) -> u64 { 0 }
    
    /// Human-readable name
    fn name(&self) -> &str { "unknown" }
}

/// Elements that can provide a clock
pub trait ClockProvider {
    /// Return a clock if this element can provide one
    fn provide_clock(&self) -> Option<Arc<dyn Clock>>;
    
    /// Priority for clock selection (higher = preferred)
    /// - 0-99: Software clocks (system monotonic)
    /// - 100-199: Hardware clocks (audio devices)
    /// - 200-299: Network clocks (NTP)
    /// - 300+: Precision clocks (PTP)
    fn clock_priority(&self) -> u32 { 0 }
}
```

### Task 2.2: Implement System Clock

**File:** `src/clock.rs`

```rust
/// System monotonic clock (fallback)
pub struct SystemClock {
    epoch: std::time::Instant,
    name: String,
}

impl SystemClock {
    pub fn new() -> Self {
        Self {
            epoch: std::time::Instant::now(),
            name: "system-monotonic".to_string(),
        }
    }
}

impl Clock for SystemClock {
    fn now(&self) -> ClockTime {
        ClockTime::from_nanos(self.epoch.elapsed().as_nanos() as u64)
    }
    
    fn flags(&self) -> ClockFlags {
        ClockFlags::CAN_BE_MASTER
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}
```

### Task 2.3: Implement ALSA Device Clock

**File:** `src/elements/device/alsa.rs`

```rust
/// Clock based on ALSA audio device timing
pub struct AlsaClock {
    pcm: Arc<PCM>,
    sample_rate: u32,
    name: String,
}

impl Clock for AlsaClock {
    fn now(&self) -> ClockTime {
        // Get current hardware position
        if let Ok(status) = self.pcm.status() {
            let delay = status.get_delay();
            let avail = status.get_avail();
            // Calculate time based on buffer position
            // ...
        }
        ClockTime::NONE
    }
    
    fn flags(&self) -> ClockFlags {
        ClockFlags::CAN_BE_MASTER | ClockFlags::HARDWARE | ClockFlags::REALTIME
    }
    
    fn resolution(&self) -> u64 {
        // Resolution is 1 sample period
        1_000_000_000 / self.sample_rate as u64
    }
    
    fn name(&self) -> &str {
        &self.name
    }
}

impl ClockProvider for AlsaSink {
    fn provide_clock(&self) -> Option<Arc<dyn Clock>> {
        Some(Arc::new(AlsaClock {
            pcm: self.pcm.clone(),
            sample_rate: self.format.sample_rate,
            name: format!("alsa:{}", self.device),
        }))
    }
    
    fn clock_priority(&self) -> u32 {
        100 // Hardware audio clock
    }
}
```

---

## Phase 3: Pipeline Clock Integration (Medium Term)

**Objective:** Connect the clock system to the pipeline executor

### Task 3.1: Add Clock to Pipeline

**File:** `src/pipeline/graph.rs`

```rust
pub struct Pipeline {
    // ... existing fields ...
    
    /// The pipeline clock (selected from providers or system default)
    clock: Option<Arc<dyn Clock>>,
    
    /// Base time when pipeline started running
    base_time: ClockTime,
}

impl Pipeline {
    /// Select the best clock from all elements
    pub fn select_clock(&mut self) {
        let mut best_clock: Option<(Arc<dyn Clock>, u32)> = None;
        
        for element in self.elements() {
            if let Some(provider) = element.as_clock_provider() {
                if let Some(clock) = provider.provide_clock() {
                    let priority = provider.clock_priority();
                    if best_clock.is_none() || priority > best_clock.as_ref().unwrap().1 {
                        best_clock = Some((clock, priority));
                    }
                }
            }
        }
        
        self.clock = best_clock.map(|(c, _)| c)
            .or_else(|| Some(Arc::new(SystemClock::new())));
    }
    
    /// Get the pipeline clock
    pub fn clock(&self) -> Option<&Arc<dyn Clock>> {
        self.clock.as_ref()
    }
    
    /// Get current running time
    pub fn running_time(&self) -> ClockTime {
        if let Some(clock) = &self.clock {
            clock.now().saturating_sub(self.base_time)
        } else {
            ClockTime::NONE
        }
    }
}
```

### Task 3.2: Pass Clock to Elements via Context

**File:** `src/element/context.rs`

```rust
pub struct ProduceContext<'a> {
    // ... existing fields ...
    
    /// Reference to pipeline clock
    clock: Option<&'a Arc<dyn Clock>>,
    
    /// Pipeline base time
    base_time: ClockTime,
}

impl<'a> ProduceContext<'a> {
    /// Get the pipeline clock
    pub fn clock(&self) -> Option<&Arc<dyn Clock>> {
        self.clock
    }
    
    /// Get current running time
    pub fn running_time(&self) -> ClockTime {
        if let Some(clock) = &self.clock {
            clock.now().saturating_sub(self.base_time)
        } else {
            ClockTime::NONE
        }
    }
}
```

### Task 3.3: Update Executor to Distribute Clock

**File:** `src/pipeline/unified_executor.rs`

```rust
impl Executor {
    pub async fn start(&mut self, pipeline: &mut Pipeline) -> Result<()> {
        // Select and distribute clock before starting
        pipeline.select_clock();
        
        if let Some(clock) = pipeline.clock() {
            tracing::info!("Pipeline clock: {}", clock.name());
        }
        
        // Set base time
        pipeline.base_time = pipeline.clock()
            .map(|c| c.now())
            .unwrap_or(ClockTime::ZERO);
        
        // ... rest of startup ...
    }
}
```

---

## Phase 4: Timestamp Preservation Through Pipeline

**Objective:** Ensure timestamps flow correctly through all element types

### Task 4.1: Audit All Transform Elements

Review and fix timestamp preservation in:
- `src/elements/transform/videoconvert.rs` - Currently preserves metadata (OK)
- `src/elements/codec/h264.rs` - Fixed to preserve input metadata
- `src/elements/transform/*.rs` - Audit each one

**Pattern for transforms:**
```rust
impl Element for MyTransform {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // ... transform data ...
        
        // ALWAYS preserve input metadata (includes PTS)
        let output = Buffer::new(output_handle, buffer.metadata().clone());
        Ok(Some(output))
    }
}
```

### Task 4.2: Handle Timestamp Interpolation for Multi-Output

For elements that produce multiple outputs from one input (e.g., video decoder with B-frames):

```rust
fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
    let input_pts = buffer.metadata().pts;
    let input_duration = buffer.metadata().duration;
    
    // If producing N frames from 1 input
    for i in 0..n_outputs {
        let frame_pts = input_pts + input_duration.saturating_mul(i as u64);
        let mut meta = buffer.metadata().clone();
        meta.pts = frame_pts;
        // ...
    }
}
```

### Task 4.3: Handle Duration Calculation

Elements should set duration when known:

```rust
// For video at known framerate
let duration = ClockTime::from_nanos(1_000_000_000 / fps as u64);
metadata.duration = duration;

// For audio with known sample count and rate  
let duration = ClockTime::from_nanos(
    (sample_count as u64 * 1_000_000_000) / sample_rate as u64
);
metadata.duration = duration;
```

---

## Phase 5: Muxer Timestamp Handling (Already Partially Done)

### Task 5.1: Update MP4 Muxer (Done)

The MP4 muxer was updated to use buffer PTS when available.

### Task 5.2: Update Other Muxers

**Files:** 
- `src/elements/mux/mpegts.rs`
- `src/elements/mux/ts_element.rs`

Same pattern: use `buffer.metadata().pts` instead of calculating from frame count.

---

## Phase 6: Testing and Validation

### Task 6.1: Create Timestamp Validation Tests

**File:** `tests/timestamp_flow.rs`

```rust
#[test]
fn test_pts_flows_through_pipeline() {
    // Create source that sets known PTS
    // Run through transforms
    // Verify sink receives correct PTS
}

#[test]
fn test_hardware_timestamp_extraction() {
    // Mock PipeWire buffer with spa_meta_header
    // Verify extraction works
}

#[test]
fn test_clock_provider_selection() {
    // Pipeline with multiple clock providers
    // Verify highest priority wins
}
```

### Task 6.2: Add Timestamp Debugging

Add a debug element that logs timestamps:

```rust
pub struct TimestampDebug {
    name: String,
}

impl Element for TimestampDebug {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let meta = buffer.metadata();
        tracing::debug!(
            "[{}] PTS={} DTS={} Duration={} Seq={}",
            self.name,
            meta.pts,
            meta.effective_dts(),
            meta.duration,
            meta.sequence
        );
        Ok(Some(buffer))
    }
}
```

---

## Implementation Order

### Week 1: Hardware Timestamps (Phase 1)
1. [ ] Task 1.1: PipeWire screen capture - extract `spa_meta_header.pts`
2. [ ] Task 1.2: V4L2 capture - use `v4l2_buffer.timestamp`
3. [ ] Task 1.3: ALSA capture - calculate from sample position
4. [ ] Test: Verify timestamps in logs are correct

### Week 2: Clock Provider Infrastructure (Phase 2)
1. [ ] Task 2.1: Define `Clock` and `ClockProvider` traits
2. [ ] Task 2.2: Implement `SystemClock`
3. [ ] Task 2.3: Implement `AlsaClock` for audio sinks
4. [ ] Test: Unit tests for clock implementations

### Week 3: Pipeline Integration (Phase 3)
1. [ ] Task 3.1: Add clock to Pipeline struct
2. [ ] Task 3.2: Add clock to ProduceContext
3. [ ] Task 3.3: Update executor to select and distribute clock
4. [ ] Test: Integration test with clock selection

### Week 4: Validation and Polish (Phase 4-6)
1. [ ] Task 4.1: Audit all transform elements
2. [ ] Task 4.2-4.3: Duration handling
3. [ ] Task 5.2: Update other muxers
4. [ ] Task 6.1-6.2: Testing and debugging tools

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/clock.rs` | Add `Clock`, `ClockProvider`, `ClockFlags`, `SystemClock` |
| `src/elements/device/screen_capture.rs` | Extract `spa_meta_header.pts` |
| `src/elements/device/v4l2.rs` | Use `v4l2_buffer.timestamp` |
| `src/elements/device/alsa.rs` | Calculate PTS from samples, implement `AlsaClock` |
| `src/elements/device/pipewire.rs` | Extract `spa_meta_header.pts` (if exists) |
| `src/pipeline/graph.rs` | Add clock selection and distribution |
| `src/element/context.rs` | Add clock reference to `ProduceContext` |
| `src/pipeline/unified_executor.rs` | Select clock on startup |
| `src/elements/mux/*.rs` | Use buffer PTS |
| Various transforms | Ensure metadata preservation |

---

## Dependencies

No new external dependencies required. We use:
- `libspa_sys` (already included via `pipewire` crate) for `spa_meta_header`
- `v4l` crate (already used) for V4L2 timestamps  
- `alsa` crate (already used) for ALSA timing
- `bitflags` (already in dependencies) for `ClockFlags`

---

## Success Criteria

1. **Screen capture video plays at correct speed** - Not accelerated or slowed
2. **Timestamps match wall clock** - 5 second capture has ~5s of timestamps
3. **A/V sync works** - Audio and video from same source stay in sync
4. **Clock selection works** - Audio sinks provide clock when present
5. **Transforms preserve timestamps** - PTS unchanged through videoconvert, encoder

---

## Future Work (Not in This Plan)

- PTP (IEEE1588) clock support for professional sync
- NTP clock support for distributed systems
- Clock drift compensation with adaptive resampling
- Network clock distribution for multi-machine pipelines
