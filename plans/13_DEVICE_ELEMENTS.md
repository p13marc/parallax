# Plan 13: Device Elements (V4L2, ALSA)

**Priority:** Medium  
**Effort:** Medium (2-3 weeks)  
**Dependencies:** None  

---

## Problem Statement

Parallax cannot capture from hardware devices. To build real-world applications (video conferencing, streaming, recording), we need:
- Camera capture (V4L2)
- Audio capture/playback (ALSA)
- Screen capture (DRM/KMS)

GStreamer has `v4l2src`, `alsasrc`, `alsasink`, `pipewiresrc`, etc. Parallax has none.

---

## Goals

1. Implement V4L2 video capture (`v4l2src`)
2. Implement ALSA audio capture/playback (`alsasrc`, `alsasink`)
3. Design for zero-copy where possible (DMA-BUF export)
4. Proper device enumeration and capability query
5. Linux-only (consistent with project goals)

---

## V4L2 Video Capture

### Design

```rust
/// V4L2 video capture source
pub struct V4l2Src {
    device: v4l::Device,
    stream: v4l::io::mmap::Stream<'static>,
    format: VideoFormat,
    framerate: Rational,
}

impl V4l2Src {
    /// Create capture source for device
    pub fn new(device_path: &str) -> Result<Self>;
    
    /// List available devices
    pub fn enumerate_devices() -> Result<Vec<V4l2DeviceInfo>>;
    
    /// Query device capabilities
    pub fn query_caps(device_path: &str) -> Result<Vec<VideoFormatCaps>>;
    
    /// Set capture format
    pub fn set_format(&mut self, format: VideoFormat, framerate: Rational) -> Result<()>;
}

pub struct V4l2DeviceInfo {
    pub path: PathBuf,
    pub name: String,
    pub driver: String,
    pub bus_info: String,
    pub capabilities: V4l2Capabilities,
}

impl Source for V4l2Src {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        // Dequeue buffer from V4L2
        let (buffer, metadata) = self.stream.next()?;
        
        // Copy to output buffer (or zero-copy via DMA-BUF)
        ctx.output()[..buffer.len()].copy_from_slice(buffer);
        
        Ok(ProduceResult::Produced(buffer.len()))
    }
}
```

### Memory Modes

| Mode | Description | Performance |
|------|-------------|-------------|
| **read()** | Simple read, copy to userspace | Slowest |
| **mmap** | Kernel buffers mapped to userspace | Good |
| **userptr** | Userspace provides buffers | Good |
| **DMA-BUF** | Export as file descriptor | Zero-copy |

**Goal:** Support mmap (default) and DMA-BUF (for GPU pipelines).

### DMA-BUF Export

```rust
impl V4l2Src {
    /// Enable DMA-BUF export mode
    pub fn enable_dmabuf_export(&mut self) -> Result<()> {
        // Request V4L2_MEMORY_DMABUF
        let req = v4l::buffer::RequestBuffers {
            count: 4,
            memory: v4l::memory::Memory::DmaBuf,
            ..Default::default()
        };
        self.device.request_buffers(req)?;
        Ok(())
    }
    
    /// Produce returns DMA-BUF fd instead of copying
    fn produce_dmabuf(&mut self) -> Result<(OwnedFd, VideoFrame)> {
        let buffer = self.stream.next()?;
        let fd = buffer.export_dmabuf()?;
        Ok((fd, VideoFrame { ... }))
    }
}
```

---

## ALSA Audio Capture/Playback

### Design

```rust
/// ALSA audio capture source
pub struct AlsaSrc {
    pcm: alsa::PCM,
    format: AudioFormat,
    buffer_size: usize,
}

impl AlsaSrc {
    /// Create capture source for device
    pub fn new(device: &str, format: AudioFormat) -> Result<Self>;
    
    /// List available capture devices
    pub fn enumerate_devices() -> Result<Vec<AlsaDeviceInfo>>;
}

/// ALSA audio playback sink
pub struct AlsaSink {
    pcm: alsa::PCM,
    format: AudioFormat,
}

impl AlsaSink {
    /// Create playback sink for device
    pub fn new(device: &str, format: AudioFormat) -> Result<Self>;
    
    /// List available playback devices
    pub fn enumerate_devices() -> Result<Vec<AlsaDeviceInfo>>;
}

pub struct AlsaDeviceInfo {
    pub name: String,
    pub description: String,
    pub is_capture: bool,
    pub is_playback: bool,
}

impl Source for AlsaSrc {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        let samples = self.pcm.io_i16()?.readi(ctx.output_as_i16_mut())?;
        let bytes = samples * self.format.channels as usize * 2;
        Ok(ProduceResult::Produced(bytes))
    }
}

impl Sink for AlsaSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        let samples = ctx.input().len() / (self.format.channels as usize * 2);
        self.pcm.io_i16()?.writei(ctx.input_as_i16())?;
        Ok(())
    }
}
```

### Async Support

ALSA is inherently blocking. Two approaches:

1. **spawn_blocking:** Wrap in `tokio::task::spawn_blocking`
2. **poll-based:** Use `alsa::PollDescriptors` with `tokio::io::unix::AsyncFd`

```rust
impl AsyncSource for AlsaSrc {
    async fn produce(&mut self, ctx: &mut ProduceContext<'_>) -> Result<ProduceResult> {
        // Wait for data to be available
        self.wait_readable().await?;
        
        // Non-blocking read
        let samples = self.pcm.io_i16()?.readi(ctx.output_as_i16_mut())?;
        Ok(ProduceResult::Produced(samples * self.frame_size()))
    }
    
    async fn wait_readable(&self) -> Result<()> {
        let fds = self.pcm.get_poll_descriptors()?;
        let async_fd = AsyncFd::new(fds[0].fd)?;
        async_fd.readable().await?;
        Ok(())
    }
}
```

---

## Screen Capture (DRM/KMS)

### Design

```rust
/// DRM/KMS screen capture source
pub struct DrmCaptureSrc {
    device: drm::Device,
    crtc: drm::control::crtc::Handle,
    framebuffer: drm::control::framebuffer::Handle,
}

impl DrmCaptureSrc {
    /// Create capture for primary display
    pub fn new() -> Result<Self>;
    
    /// Create capture for specific output
    pub fn new_for_output(output: &str) -> Result<Self>;
    
    /// List available outputs
    pub fn enumerate_outputs() -> Result<Vec<DrmOutputInfo>>;
}

impl Source for DrmCaptureSrc {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
        // Get framebuffer as DMA-BUF
        let dmabuf = self.device.buffer_to_dmabuf(self.framebuffer)?;
        
        // Map and copy (or export DMA-BUF directly)
        // ...
    }
}
```

**Note:** Screen capture is complex due to:
- Compositor integration (Wayland protocols)
- Permission requirements
- Performance (compositors may not expose efficient capture)

**Recommendation:** Implement basic DRM capture first, consider PipeWire integration later.

---

## Implementation Steps

### Phase 1: V4L2 Capture (1 week)

- [ ] Add `v4l` crate dependency
- [ ] Create `src/elements/device/mod.rs`
- [ ] Implement `src/elements/device/v4l2.rs`:
  - [ ] Device enumeration
  - [ ] Format negotiation
  - [ ] mmap streaming
  - [ ] Timestamp handling
- [ ] Implement `V4l2Src` as `Source` trait
- [ ] Add unit tests (may need test device)
- [ ] Create example: `22_v4l2_capture.rs`

### Phase 2: ALSA Audio (1 week)

- [ ] Add `alsa` crate dependency
- [ ] Implement `src/elements/device/alsa.rs`:
  - [ ] Device enumeration
  - [ ] Format configuration
  - [ ] Capture and playback
  - [ ] Async support via poll
- [ ] Implement `AlsaSrc` as `AsyncSource`
- [ ] Implement `AlsaSink` as `AsyncSink`
- [ ] Add unit tests
- [ ] Create example: `23_alsa_audio.rs`

### Phase 3: DMA-BUF Integration (3-5 days)

- [ ] Add V4L2 DMA-BUF export mode
- [ ] Integrate with `MemoryType::DmaBuf`
- [ ] Test zero-copy path to GPU (Plan 11)
- [ ] Document DMA-BUF usage

### Phase 4: Screen Capture (Optional, 3-5 days)

- [ ] Add `drm` crate dependency
- [ ] Implement basic DRM capture
- [ ] Handle permissions (DRM master)
- [ ] Create example: `24_screen_capture.rs`

### Phase 5: Integration (2-3 days)

- [ ] Update caps negotiation for device caps
- [ ] Add device info to admin space
- [ ] Create "loopback" example (camera → display)
- [ ] Update documentation

---

## Feature Flags

```toml
[features]
# Device capture features
v4l2 = ["dep:v4l"]
alsa = ["dep:alsa"]
drm-capture = ["dep:drm"]

# Combined
device-capture = ["v4l2", "alsa"]
```

---

## Dependencies

| Crate | Version | Purpose | License |
|-------|---------|---------|---------|
| `v4l` | 0.14 | V4L2 bindings | MIT/Apache-2.0 |
| `alsa` | 0.8 | ALSA bindings | MIT/Apache-2.0 |
| `drm` | 0.10 | DRM bindings | MIT |

---

## Error Handling

```rust
#[derive(Debug, thiserror::Error)]
pub enum DeviceError {
    #[error("Device not found: {0}")]
    NotFound(String),
    
    #[error("Device busy: {0}")]
    Busy(String),
    
    #[error("Permission denied: {0}")]
    PermissionDenied(String),
    
    #[error("Format not supported: {0:?}")]
    FormatNotSupported(VideoFormat),
    
    #[error("Device disconnected")]
    Disconnected,
    
    #[error("V4L2 error: {0}")]
    V4l2(#[from] v4l::Error),
    
    #[error("ALSA error: {0}")]
    Alsa(#[from] alsa::Error),
}
```

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_v4l2_enumerate() {
    let devices = V4l2Src::enumerate_devices().unwrap();
    // At least dummy video device should exist in CI
    // or skip test if no devices
    if devices.is_empty() {
        return;
    }
    assert!(!devices[0].path.as_os_str().is_empty());
}

#[test]
fn test_alsa_enumerate() {
    let devices = AlsaSrc::enumerate_devices().unwrap();
    // Similar - may not have audio devices in CI
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_v4l2_capture_pipeline() {
    // Skip if no camera available
    let devices = V4l2Src::enumerate_devices().unwrap();
    if devices.is_empty() {
        return;
    }
    
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node("src", V4l2Src::new(&devices[0].path)?);
    let sink = pipeline.add_node("sink", NullSink::new());
    pipeline.link(src, sink)?;
    
    // Capture 10 frames
    pipeline.run_for(Duration::from_secs(1)).await?;
}
```

### Manual Testing

```bash
# Test with v4l2-ctl
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats

# Test ALSA
arecord -l
aplay -l
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/elements/device/mod.rs` | Module root |
| `src/elements/device/v4l2.rs` | V4L2 capture |
| `src/elements/device/alsa.rs` | ALSA capture/playback |
| `src/elements/device/drm.rs` | DRM screen capture |
| `examples/22_v4l2_capture.rs` | Camera example |
| `examples/23_alsa_audio.rs` | Audio example |
| `examples/24_screen_capture.rs` | Screen example |

---

## Considerations

### Permissions

| Device | Required | Notes |
|--------|----------|-------|
| V4L2 | `video` group | Usually `/dev/video*` |
| ALSA | `audio` group | Usually accessible |
| DRM | DRM master or root | Complex, compositor may hold lock |

### Hotplug

Devices can be connected/disconnected at runtime:
- Use `udev` for device monitoring
- Handle `Disconnected` errors gracefully
- Consider reconnection logic

### PipeWire Integration (Future)

For desktop integration, PipeWire is preferred:
- Handles permissions (portal)
- Shares devices with other apps
- Better Wayland support

Could add `pipewiresrc`/`pipewiresink` later using `pipewire` crate.

---

## Success Criteria

- [ ] V4L2 capture works with USB webcams
- [ ] ALSA capture/playback works
- [ ] Device enumeration lists available devices
- [ ] Async sources don't block the runtime
- [ ] Examples demonstrate camera → file, mic → file
- [ ] DMA-BUF export works (verified with GPU pipeline)

---

## Future Work

| Feature | Notes |
|---------|-------|
| PipeWire | Better desktop integration |
| JACK | Pro audio support |
| PulseAudio | Legacy desktop audio |
| Hotplug | udev monitoring |
| V4L2 output | Video overlay |

---

*Created January 2026 based on [Project Analysis Report](../docs/PROJECT_ANALYSIS_REPORT.md)*
