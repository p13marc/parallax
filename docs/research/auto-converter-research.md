# Auto-Converter Insertion Research Report

## Executive Summary

**Question:** Should Parallax automatically insert format converters when pipeline elements have incompatible formats?

**Recommendation:** Follow GStreamer's approach - **explicit converters by default**, with optional auto-insertion as an opt-in feature. Both frameworks have merits, but GStreamer's explicit approach gives users better control over performance-critical pipelines.

---

## The Problem

When connecting pipeline elements with incompatible formats (e.g., YUYV camera → RGB display), three options exist:

1. **Fail** - Require user to explicitly add converters
2. **Auto-insert** - Silently add converters to make it work
3. **Hybrid** - Fail by default, allow opt-in auto-insertion

Hidden costs of auto-insertion:
- **CPU overhead**: Format conversion is computationally expensive
- **Memory overhead**: May require additional buffer allocations
- **Quality loss**: Some conversions are lossy (e.g., color space, resampling)
- **Latency**: Adds processing time to the pipeline
- **Debugging difficulty**: Pipeline behavior differs from what user specified

---

## How GStreamer Handles It

### Philosophy: Explicit Control

GStreamer **never automatically inserts converters**. Users must explicitly add elements like `videoconvert`, `audioconvert`, `videoscale`, etc.

```bash
# This fails if camera outputs YUYV and display needs RGB:
gst-launch-1.0 v4l2src ! autovideosink

# User must explicitly add converter:
gst-launch-1.0 v4l2src ! videoconvert ! autovideosink
```

### Passthrough Optimization

GStreamer converters implement **passthrough mode** via `GstBaseTransform`:

> "When not needed, because its upstream and downstream elements can already understand each other, it acts in pass-through mode having minimal impact on the performance."

In passthrough mode:
- No buffer allocation occurs
- Input buffer is pushed through unchanged
- Effectively zero CPU cost
- No memory copies

This means users can defensively add converters without penalty when formats already match.

```c
// From GstBaseTransform documentation:
// "In passthrough mode, buffers are inspected but no metadata or buffer 
// data is changed. The input buffers don't need to be writable, and the 
// input buffer is simply pushed out again without modifications."
```

### Auto-Select Elements

GStreamer provides `autovideoconvert` and `autoconvert` elements that the user **explicitly adds** to the pipeline. These then internally select the best converter based on caps:

> "The autovideoconvert element selects the right color space converter based on the caps."

This is different from auto-insertion - the user consciously chooses to use an auto-selecting element.

### Negotiation Design

GStreamer identifies three negotiation modes:
1. **Fixed** - Element outputs one format only
2. **Transform** - Fixed transform based on properties
3. **Dynamic** - Converts fixed caps to unfixed caps

The negotiation system finds compatible formats but does not insert elements automatically.

### Strengths

| Aspect | Benefit |
|--------|---------|
| Predictability | Pipeline does exactly what user specifies |
| Cost visibility | User knows where conversion happens |
| Debugging | Easy to identify bottlenecks |
| Flexibility | User can choose converter quality/speed tradeoffs |
| Passthrough | Zero cost when no conversion needed |

### Weaknesses

| Aspect | Drawback |
|--------|----------|
| Verbosity | Pipelines require more elements |
| Learning curve | Users must understand format compatibility |
| Boilerplate | Common patterns require repetitive code |

---

## How PipeWire Handles It

### History and Design Goals

PipeWire was created by Wim Taymans (GStreamer co-creator) at Red Hat. **It was originally designed for video**, not audio:

> "The original reason it was created was that as desktop applications would be moving towards primarily being shipped as containerized Flatpaks, they would need something for video similar to what PulseAudio was doing for audio."

The project went through several names:
- **PulseVideo** (original name) → **Pinos** → **PipeWire** (2017)

> "Initially, Pinos only handled video streams. By early 2017, Taymans had started working on integrating audio streams."

A major early use case was **Wayland screen sharing**, which needed secure access to screen data that X11's insecure model couldn't provide.

### Philosophy: Transparent Conversion via Adapters

PipeWire uses an **adapter** abstraction that transparently handles format conversion:

> "A stream is a wrapper around a proxy for a pw_client_node with an adapter. This means the stream will automatically do conversion to the type required by the server."

The adapter handles:
- Format conversion (e.g., S16 ↔ F32)
- Sample rate conversion
- Channel remixing/remapping

### Vision: Unified Audio/Video Hub

PipeWire aims to be the central hub for all media:

> "Having PipeWire be the central hub means getting the same advantages for video that exist for audio—as the application developer, you interact with PipeWire regardless of whether you want screen capture, a camera feed, or video playback."

> "Just like you don't write audio applications directly to the ALSA API anymore, you shouldn't write video applications directly to the v4l2 kernel API anymore."

### Video Support Status

PipeWire has integrated libcamera for modern camera handling:

> "The integration work [libcamera into PipeWire] has been merged into PipeWire master."

However, video format conversion is less mature than audio:

> "Format negotiations are all broken :-( WebRTC is calculating NV12 buffer sizes incorrectly."

The SPA (Simple Plugin API) provides video format utilities, but automatic video conversion plugins are less developed compared to the audio side (`audioconvert`, `audioadapter`).

### Passthrough Mode

PipeWire's adapter can be configured for passthrough:

> "The audio adapter can also be configured in passthrough mode when it will not do any conversions but simply pass through the port information of the internal node."

Benefits of passthrough:
> "Audio passthrough has enabled faster audio paths for resource-intensive applications like video games as well as applications on embedded devices."

> "The use case for PCM passthrough is to reduce the load for resource-intensive applications on mobile devices, where every CPU cycle counts, both for performance and battery life."

### Configuration

Passthrough is configured via:
- `PortConfig` parameter
- `audio.no-dsp` option (disables channel splitting/merging)

### Strengths

| Aspect | Benefit |
|--------|---------|
| Unified design | Single API for audio, video, screen capture |
| Simplicity | Applications don't need to handle formats |
| Security | Portal-based access for sandboxed apps |
| Compatibility | Everything "just works" |
| Central optimization | Adapter can be optimized once |

### Weaknesses

| Aspect | Drawback |
|--------|----------|
| Hidden costs | Applications unaware of conversion overhead |
| Less control | Can't easily optimize specific paths |
| Debugging | Harder to trace format issues |
| Video maturity | Video conversion less polished than audio |

---

## Comparison

| Aspect | GStreamer | PipeWire | Winner |
|--------|-----------|----------|--------|
| **User control** | Full control | Limited control | GStreamer |
| **Ease of use** | Requires knowledge | Just works | PipeWire |
| **Cost visibility** | Explicit | Hidden | GStreamer |
| **Passthrough support** | Yes (per-element) | Yes (adapter-level) | Tie |
| **Video pipelines** | Mature, extensive | Growing, unified API | GStreamer (maturity) |
| **Audio pipelines** | Good | Excellent | PipeWire |
| **Debugging** | Easy | Harder | GStreamer |
| **Flexibility** | High | Medium | GStreamer |
| **Performance tuning** | User-controlled | System-controlled | GStreamer |
| **Sandboxed apps** | Manual setup | Portal integration | PipeWire |
| **System integration** | Application-level | System-level daemon | PipeWire |

### Which Handles It Better?

**It depends on the use case:**

#### GStreamer is better for:
- **Application-controlled pipelines** where developers need full control
- **Performance-critical video** where conversion costs matter (10-100ms/frame)
- **Quality-sensitive workflows** where colorspace choices affect output
- **Debugging and profiling** - explicit elements make bottlenecks obvious
- **Complex pipelines** with many elements and format variations

#### PipeWire is better for:
- **System-level media routing** (replacement for PulseAudio/JACK)
- **Sandboxed applications** (Flatpak, portals)
- **Simple applications** that just need "a camera" or "a microphone"
- **Desktop integration** (screen sharing, audio routing)
- **When the OS should manage device access**

### Different Design Goals

GStreamer and PipeWire solve different problems:

| | GStreamer | PipeWire |
|--|-----------|----------|
| **Level** | Application framework | System service |
| **Goal** | Build media pipelines | Route media between apps |
| **User** | Application developer | End user / system |
| **Conversion** | App's responsibility | System's responsibility |

**For Parallax** (an application-level pipeline framework like GStreamer), the GStreamer approach makes more sense - users building pipelines should control conversion costs.

---

## Zero-Cost Converters

### When Input = Output Format

Both GStreamer and PipeWire support **true zero-cost passthrough**:

```
Input Format: RGB24
Output Format: RGB24
Result: Buffer passed through unchanged, no copies, no allocation
```

This is achieved via:
- GStreamer: `passthrough_on_same_caps` in `GstBaseTransform`
- PipeWire: Adapter passthrough mode

### When Conversion Is Needed

**No conversion is truly zero-cost.** Even optimized implementations have overhead:

| Conversion | Typical Cost (1080p) | Notes |
|------------|---------------------|-------|
| YUV → RGB | 2-10ms | SIMD-optimized |
| RGB → YUV | 2-10ms | SIMD-optimized |
| Bilinear scale | 5-20ms | Depends on ratio |
| Audio S16 → F32 | <1ms | Very fast |
| Audio resample | 1-5ms | Quality dependent |

### Recommendation for Parallax

Implement passthrough optimization:

```rust
impl Element for VideoConvert {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Zero-cost passthrough when formats match
        if self.input_format == self.output_format {
            return Ok(Some(buffer));  // No copy, no conversion
        }
        
        // Actual conversion
        self.convert(buffer)
    }
}
```

---

## Recommendation for Parallax

### Default Behavior: Explicit (Like GStreamer)

```rust
// This should FAIL with a clear error:
let mut pipeline = Pipeline::new();
pipeline.add_source("camera", V4l2Src::new("/dev/video0")?);  // Outputs YUYV
pipeline.add_sink("display", DisplaySink::new());              // Needs RGB
pipeline.link(camera, display)?;
pipeline.prepare()?;  
// Error: "Cannot negotiate: camera outputs YUYV, display needs RGB. 
//         Consider adding a VideoConvert element."
```

### Opt-In Auto-Insertion

```rust
// Option 1: Pipeline-level policy
pipeline.set_converter_policy(ConverterPolicy::AutoInsert);
pipeline.prepare()?;  // Works, logs warning about inserted converter

// Option 2: Explicit method
pipeline.prepare_with_auto_converters()?;  // Opt-in for this prepare call

// Option 3: Per-link hint
pipeline.link_with_options(camera, display, LinkOptions::allow_converters())?;
```

### Warning When Auto-Inserting

```
[WARN] Auto-inserting VideoConvert between 'camera' and 'display'
       Conversion: YUYV → RGB24 (estimated cost: ~5ms/frame @ 1080p)
       To avoid this warning, explicitly add the converter or set policy to Allow.
```

### Passthrough Support

All converters should implement passthrough when input = output:

```rust
pub trait Converter {
    /// Returns true if this converter can passthrough (no-op) for the given formats.
    fn is_passthrough(&self) -> bool;
    
    /// Estimated cost per frame in microseconds (0 for passthrough).
    fn estimated_cost_us(&self) -> u64;
}
```

### API Design

```rust
/// Policy for automatic converter insertion during negotiation.
#[derive(Debug, Clone, Copy, Default)]
pub enum ConverterPolicy {
    /// Fail negotiation if formats don't match (default).
    /// User must explicitly add converters.
    #[default]
    Deny,
    
    /// Auto-insert converters but log warnings.
    /// Good for development/debugging.
    Warn,
    
    /// Auto-insert converters silently.
    /// Use with caution in production.
    Allow,
}

impl Pipeline {
    /// Set the converter insertion policy.
    pub fn set_converter_policy(&mut self, policy: ConverterPolicy);
    
    /// Prepare with explicit auto-converter opt-in (ignores policy).
    pub fn prepare_with_auto_converters(&mut self) -> Result<()>;
}
```

---

## Implementation Checklist

- [ ] Change default behavior to fail on format mismatch
- [ ] Add `ConverterPolicy` enum
- [ ] Add `set_converter_policy()` to Pipeline
- [ ] Add `prepare_with_auto_converters()` method
- [ ] Implement passthrough detection in converters
- [ ] Add cost estimation to converters
- [ ] Log warnings when auto-inserting with `Warn` policy
- [ ] Update documentation and examples
- [ ] Add clear error messages suggesting converters

---

## References

### GStreamer
- [Caps Negotiation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/negotiation.html)
- [autovideoconvert](https://gstreamer.freedesktop.org/documentation/autoconvert/autovideoconvert.html)
- [videoconvert](https://gstreamer.freedesktop.org/documentation/videoconvert/index.html)
- [GstBaseTransform](https://gstreamer.freedesktop.org/documentation/base/gstbasetransform.html)
- [Basic Tutorial 14: Handy Elements](https://gstreamer.freedesktop.org/documentation/tutorials/basic/handy-elements.html)
- [Transform Elements Design](https://gstreamer.freedesktop.org/documentation/additional/design/element-transform.html)

### PipeWire
- [Streams Documentation](https://docs.pipewire.org/page_streams.html)
- [Audio Documentation](https://docs.pipewire.org/page_audio.html)
- [DMA-BUF Sharing](https://docs.pipewire.org/page_dma_buf.html)
- [SPA (Simple Plugin API)](https://docs.pipewire.org/page_spa.html)
- [PipeWire Overview](https://docs.pipewire.org/page_overview.html)
- [PipeWire: A Year in Review](https://www.collabora.com/news-and-blog/blog/2022/03/08/pipewire-a-year-in-review-look-ahead/)

### PipeWire History and Design
- [Wim Taymans - PipeWire (GStreamer Conference 2017)](https://gstreamer.freedesktop.org/data/events/gstreamer-conference/2017/Wim%20Taymans%20-%20PipeWire.pdf)
- [Launching PipeWire - Christian Schaller](https://blogs.gnome.org/uraeus/2017/09/19/launching-pipewire/)
- [PipeWire and Fixing the Linux Video Capture Stack](https://blogs.gnome.org/uraeus/2021/10/01/pipewire-and-fixing-the-linux-video-capture-stack/)
- [Integrating libcamera into PipeWire - Collabora](https://www.collabora.com/news-and-blog/blog/2020/09/11/integrating-libcamera-into-pipewire/)
- [An Introduction to PipeWire - Bootlin](https://bootlin.com/blog/an-introduction-to-pipewire/)
- [PipeWire - Wikipedia](https://en.wikipedia.org/wiki/PipeWire)

---

*Report generated: January 2026*
*Author: Claude Code analysis of GStreamer and PipeWire documentation*
