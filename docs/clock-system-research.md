# Clock System Research: GStreamer, PipeWire, and Hardware Timestamps

## Executive Summary

This document analyzes the clock and timestamp systems used by GStreamer and PipeWire, and evaluates whether Parallax needs a formal clock provider architecture. The current implementation uses `std::time::Instant` (monotonic system clock) for timestamps, which has limitations for hardware-synchronized capture and multi-device scenarios.

**Key Finding**: A proper clock provider system is essential for:
1. **Hardware timestamp support** - Using timestamps from the actual capture hardware (V4L2, ALSA, PipeWire)
2. **A/V synchronization** - Ensuring audio and video from different sources stay in sync
3. **Multi-device synchronization** - Synchronizing streams across multiple machines (PTP, NTP)
4. **Live playback** - Proper real-time synchronization with sink devices

---

## Current State in Parallax

Currently, the screen capture source uses:

```rust
let now = Instant::now();
let pts = {
    let elapsed = now.duration_since(capture_start);
    ClockTime::from_nanos(elapsed.as_nanos() as u64)
};
```

### Problems with this approach:

1. **Clock domain mismatch**: We're timestamping when we *receive* the frame in our code, not when PipeWire actually captured it
2. **Jitter**: Software timing adds variable latency between actual capture and our timestamp
3. **No hardware timestamp passthrough**: PipeWire provides `spa_meta_header.pts` with proper timestamps, but we ignore them
4. **A/V drift**: If audio and video come from different sources, each using its own `Instant`, they can drift apart

---

## GStreamer Clock Architecture

GStreamer has a sophisticated clock system designed for professional media applications.

### Clock Selection

The [`GstPipeline`](https://gstreamer.freedesktop.org/documentation/additional/design/clocks.html) is responsible for selecting and distributing a global `GstClock`:

- Clock selection happens when pipeline goes to PLAYING state
- Elements can *provide* clocks (e.g., audio sinks, network sources)
- Pipeline selects the best clock based on priority
- All elements receive the same clock and base time

### Clock Provider Election

Elements signal clock availability via bus messages:
- `GST_MESSAGE_CLOCK_PROVIDE` - Element can provide a clock
- `GST_MESSAGE_NEW_CLOCK` - Clock was selected
- `GST_MESSAGE_CLOCK_LOST` - Clock provider was removed

### Time Concepts

| Concept | Description |
|---------|-------------|
| **Clock Time** | Absolute time from the clock (nanoseconds) |
| **Base Time** | Clock time when pipeline started |
| **Running Time** | `clock_time - base_time` (time since pipeline start) |
| **Stream Time** | Position in the media stream |

The fundamental synchronization formula:
```
running_time = clock_time - base_time
```

### Clock Types in GStreamer

| Clock | Source | Use Case |
|-------|--------|----------|
| [`GstSystemClock`](https://gstreamer.freedesktop.org/documentation/gstreamer/gstclock.html) | System monotonic time | Default fallback |
| Audio Device Clock | Sound card word clock | Audio playback/capture |
| [`GstPtpClock`](https://gstreamer.freedesktop.org/documentation/net/gstptpclock.html) | IEEE1588 PTP network | Multi-device sync |
| [`GstNtpClock`](https://coaxion.net/blog/2015/05/ptp-network-clock-support-in-gstreamer/) | NTP server | Loose network sync |
| `GstNetClientClock` | Custom protocol | GStreamer-specific |

### Master/Slave Clocks

Clocks with `CAN_SET_MASTER` can be slaved to another clock:
- Automatic calibration via sampling
- Configurable window size and threshold
- Useful for syncing internal clocks to external references

---

## PipeWire Clock Architecture

PipeWire uses a driver-based timing model optimized for low-latency audio.

### spa_io_clock Structure

The [`spa_io_clock`](https://docs.pipewire.org/structspa__io__clock.html) provides timing for the processing graph:

```c
struct spa_io_clock {
    uint64_t nsec;           // Time in nanoseconds (CLOCK_MONOTONIC)
    struct spa_fraction rate; // Sample rate as fraction
    uint64_t position;        // Current position in samples
    uint64_t duration;        // Samples per cycle
    int64_t delay;           // Hardware latency in samples
    double rate_diff;        // Clock drift ratio
    uint64_t next_nsec;      // Expected next wakeup time
    // ... plus target values for next cycle
};
```

**Key insight**: Driver nodes update `spa_io_clock` *before* signaling the start of each graph cycle. This ensures all nodes see consistent timing.

### spa_meta_header for Buffer Timestamps

Each buffer carries [`spa_meta_header`](https://docs.pipewire.org/structspa__meta__header.html):

```c
struct spa_meta_header {
    uint32_t flags;
    uint32_t offset;        // Offset in current cycle
    int64_t pts;            // Presentation timestamp (nanoseconds)
    int64_t dts_offset;     // DTS as difference from PTS
    uint64_t seq;           // Sequence number
};
```

**This is what we should be using** - PipeWire already provides proper timestamps on captured frames.

### pw_time for Stream Timing

The [`pw_time`](https://docs.pipewire.org/structpw__time.html) structure provides stream timing information:

```c
struct pw_time {
    int64_t now;          // When this snapshot was taken
    struct spa_fraction rate;
    uint64_t ticks;       // Current position (monotonically increasing)
    int64_t delay;        // Time until sample reaches device
    uint64_t queued;      // Queued sample count
    // ...
};
```

### Clock Synchronization

PipeWire handles clock sync through:

1. **Same clock name** - Devices with matching `clock.name` skip resampling
2. **Pro Audio mode** - Assumes same device = same clock
3. **Adaptive resampling** - Compensates for clock drift between devices
4. **ALSA htimestamps** - Can use hardware timestamps (when driver supports it)

---

## Hardware Timestamp Sources

### ALSA Audio

[ALSA PCM timestamping](https://www.kernel.org/doc/html/latest/sound/designs/timestamping.html) provides:

| Timestamp | Source | Accuracy |
|-----------|--------|----------|
| `trigger_tstamp` | When playback/capture started | ~1ms |
| `tstamp` | Current system time | ~1ms |
| `audio_tstamp` | Hardware counter | sub-ms |

Clock options via `snd_pcm_sw_params_set_tstamp_type()`:
- `CLOCK_REALTIME` - Can jump (NTP adjustments)
- `CLOCK_MONOTONIC` - Never goes backward
- `CLOCK_MONOTONIC_RAW` - No NTP adjustments

**Best practice**: Use `CLOCK_MONOTONIC` for timestamps, with `audio_tstamp` for precise hardware timing when available.

### V4L2 Video

[V4L2 buffer timestamps](https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/buffer.html) are based on `CLOCK_MONOTONIC`:

```c
struct v4l2_buffer {
    struct timeval timestamp;  // Capture time
    // flags include:
    // V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC
    // V4L2_BUF_FLAG_TSTAMP_SRC_SOE (start of exposure)
    // V4L2_BUF_FLAG_TSTAMP_SRC_EOF (end of frame)
};
```

**Important**: The timestamp indicates when the *first byte* was captured, not when we dequeue the buffer.

### PipeWire Screen Capture

PipeWire provides timestamps in `spa_meta_header.pts`:
- Based on `CLOCK_MONOTONIC`
- Set by the compositor/portal
- Represents actual capture time

**This is what the screen capture source should use** instead of `Instant::now()`.

---

## Proposed Clock Provider Architecture for Parallax

### Design Goals

1. **Hardware timestamp passthrough** - Use timestamps from capture hardware
2. **Clock provider election** - Let sink devices provide the master clock
3. **Running time calculation** - Consistent time across all elements
4. **Clock drift compensation** - Handle devices with different clock rates
5. **Network sync support** - PTP/NTP for multi-machine scenarios

### Proposed Architecture

```rust
/// A clock provides monotonically increasing time
pub trait Clock: Send + Sync {
    /// Get current time in nanoseconds
    fn now(&self) -> ClockTime;
    
    /// Clock capabilities
    fn flags(&self) -> ClockFlags;
    
    /// Resolution in nanoseconds (0 = unknown)
    fn resolution(&self) -> u64 { 0 }
}

bitflags! {
    pub struct ClockFlags: u32 {
        /// Clock can be used as pipeline master
        const CAN_BE_MASTER = 0x01;
        /// Clock can slave to another clock
        const CAN_SET_MASTER = 0x02;
        /// Clock provides hardware timestamps
        const HARDWARE = 0x04;
        /// Clock is from network source (PTP, NTP)
        const NETWORK = 0x08;
    }
}

/// Clock provider trait for elements
pub trait ClockProvider {
    /// Return a clock if this element can provide one
    fn provide_clock(&self) -> Option<Arc<dyn Clock>>;
    
    /// Priority (higher = more preferred)
    fn clock_priority(&self) -> u32 { 0 }
}
```

### Clock Selection Priority

1. **PTP/NTP network clocks** (if configured) - Highest priority for distributed systems
2. **Audio sink clocks** - Sound cards provide accurate word clocks
3. **Hardware capture clocks** - V4L2/ALSA with hardware timestamps
4. **System monotonic clock** - Fallback

### Integration Points

#### Source Elements

```rust
impl Source for ScreenCaptureSrc {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        let frame = self.recv_frame()?;
        
        // Use PipeWire's timestamp, not Instant::now()
        let pts = frame.spa_header.pts;  // From spa_meta_header
        
        let mut metadata = Metadata::new().with_pts(ClockTime::from_nanos(pts as u64));
        // ...
    }
}
```

#### Sink Elements

```rust
impl ClockProvider for AlsaSink {
    fn provide_clock(&self) -> Option<Arc<dyn Clock>> {
        // Return ALSA device clock
        Some(Arc::new(AlsaClock::new(&self.pcm)))
    }
    
    fn clock_priority(&self) -> u32 { 100 }
}
```

#### Pipeline Integration

```rust
impl Pipeline {
    /// Select the best clock from all elements
    pub fn select_clock(&mut self) -> Option<Arc<dyn Clock>> {
        self.elements
            .iter()
            .filter_map(|e| e.provide_clock())
            .max_by_key(|c| c.priority())
    }
    
    /// Start the pipeline with synchronized timing
    pub async fn run(&mut self) -> Result<()> {
        let clock = self.select_clock().unwrap_or_else(|| Arc::new(SystemClock::new()));
        self.base_time = clock.now();
        // Distribute clock to all elements...
    }
}
```

---

## Immediate Improvements (Before Full Clock System)

### 1. Use PipeWire Timestamps for Screen Capture

Update `screen_capture.rs` to use `spa_meta_header.pts`:

```rust
// In the process callback
if let Some(meta) = buffer.find_meta::<spa_meta_header>() {
    frame.pts = ClockTime::from_nanos(meta.pts as u64);
}
```

### 2. Use V4L2 Buffer Timestamps

For V4L2 capture, use the buffer's timestamp:

```rust
let buffer: v4l2_buffer = /* dequeue */;
let pts = ClockTime::from_micros(
    buffer.timestamp.tv_sec as u64 * 1_000_000 + 
    buffer.timestamp.tv_usec as u64
);
```

### 3. Use ALSA Timestamps for Audio

```rust
let mut status = snd_pcm_status::new()?;
pcm.status(&mut status)?;
let tstamp = status.get_trigger_tstamp();
let audio_tstamp = status.get_audio_tstamp(); // Hardware time if available
```

---

## Recommendations

### Short Term

1. **Fix screen capture** - Extract `spa_meta_header.pts` from PipeWire buffers
2. **Fix V4L2 capture** - Use `v4l2_buffer.timestamp` 
3. **Fix ALSA capture** - Use `snd_pcm_status` timestamps
4. **Document clock requirements** - Each source should document its timestamp source

### Medium Term

1. **Implement `Clock` trait** - Basic abstraction for clock sources
2. **Add `ClockProvider`** - Let elements advertise clocks
3. **Pipeline clock selection** - Automatic best-clock selection
4. **Running time tracking** - Consistent time across pipeline

### Long Term

1. **PTP support** - IEEE1588 for professional/broadcast use
2. **Clock drift compensation** - Adaptive resampling when clocks differ
3. **Hardware timestamp passthrough** - Full support for ALSA/V4L2 hardware timestamps

---

## References

### GStreamer
- [Clocks Design](https://gstreamer.freedesktop.org/documentation/additional/design/clocks.html)
- [Clock Implementation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/clock.html)
- [GstPtpClock](https://gstreamer.freedesktop.org/documentation/net/gstptpclock.html)
- [PTP Support Blog Post](https://coaxion.net/blog/2015/05/ptp-network-clock-support-in-gstreamer/)

### PipeWire
- [spa_io_clock](https://docs.pipewire.org/structspa__io__clock.html)
- [spa_meta_header](https://docs.pipewire.org/structspa__meta__header.html)
- [pw_time](https://docs.pipewire.org/structpw__time.html)
- [SPA Buffers](https://docs.pipewire.org/page_spa_buffer.html)

### ALSA
- [PCM Timestamping](https://www.kernel.org/doc/html/latest/sound/designs/timestamping.html)

### V4L2
- [Buffer Documentation](https://www.kernel.org/doc/html/v4.9/media/uapi/v4l/buffer.html)

---

## Conclusion

Using `Instant::now()` for timestamps is fundamentally incorrect for a media framework. The timestamp should represent when the media was *captured* or *generated*, not when our code happened to process it.

The immediate fix is to extract and use hardware-provided timestamps from PipeWire (`spa_meta_header.pts`), V4L2 (`v4l2_buffer.timestamp`), and ALSA (`audio_tstamp`).

A full clock provider system like GStreamer's is warranted for:
- Professional/broadcast applications requiring PTP sync
- A/V synchronization across different capture devices
- Network streaming with precise timing
- Real-time playback synchronized to audio hardware

For now, correctly using the timestamps already provided by Linux subsystems will solve most timing issues.
