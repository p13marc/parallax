# Plan 11: GPU Codec Framework (Vulkan Video)

**Priority:** High  
**Effort:** Large (4-6 weeks)  
**Dependencies:** Plan 09 (Format Converters) recommended but not required  

---

## Problem Statement

Parallax currently has **no hardware-accelerated codec support**. All video encoding/decoding uses CPU-only implementations (rav1e, dav1d, OpenH264). This limits performance for:
- Real-time 4K video processing
- Low-power embedded devices
- High-throughput transcoding

GStreamer has VA-API, NVENC/NVDEC plugins. Parallax needs equivalent functionality.

---

## Goals

1. Design a GPU codec abstraction that works across vendors
2. Implement Vulkan Video decode (H.264, H.265, AV1)
3. Implement Vulkan Video encode (H.264, H.265)
4. Integrate with existing caps negotiation
5. Zero-copy path: DMA-BUF → Vulkan → DMA-BUF

---

## Why Vulkan Video?

| Approach | Codecs | Portability | Status |
|----------|--------|-------------|--------|
| VA-API | H.264, H.265, VP9, AV1 | AMD, Intel (Linux) | Mature |
| NVENC/NVDEC | H.264, H.265, AV1 | NVIDIA only | Mature |
| **Vulkan Video** | H.264, H.265, AV1, VP9 | All GPUs | Finalized 2025 |

**Decision:** Vulkan Video is the only cross-vendor API. As of 2025:
- **Decode:** H.264, H.265, AV1, VP9 (all finalized)
- **Encode:** H.264, H.265, AV1 (all finalized)
- **Drivers:** AMD (RADV), Intel (ANV), NVIDIA (proprietary + NVK)

---

## Architecture

### Memory Model Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                    Parallax Memory Types                        │
├─────────────────────────────────────────────────────────────────┤
│  Cpu (memfd)  │  DmaBuf  │  GpuDevice  │  GpuAccessible        │
└───────┬───────┴────┬─────┴──────┬──────┴─────────┬─────────────┘
        │            │            │                │
        │            │            │                │
        ▼            ▼            ▼                ▼
┌───────────────────────────────────────────────────────────────┐
│                     Vulkan Video Pipeline                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐   │
│  │ Import      │───▶│ Decode/     │───▶│ Export          │   │
│  │ DMA-BUF     │    │ Encode      │    │ DMA-BUF         │   │
│  └─────────────┘    └─────────────┘    └─────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

### Trait Design

```rust
/// GPU memory that can be imported/exported
pub trait GpuMemory: Send + Sync {
    /// Import from DMA-BUF file descriptor
    fn import_dmabuf(&mut self, fd: OwnedFd, size: usize) -> Result<GpuBuffer>;
    
    /// Export to DMA-BUF for sharing
    fn export_dmabuf(&self, buffer: &GpuBuffer) -> Result<OwnedFd>;
    
    /// Allocate GPU-local memory
    fn allocate(&mut self, size: usize, usage: GpuUsage) -> Result<GpuBuffer>;
}

/// Hardware video decoder trait
pub trait HwVideoDecoder: Send {
    /// Decode compressed packet to raw frame
    fn decode(&mut self, packet: &[u8]) -> Result<Vec<GpuFrame>>;
    
    /// Flush any buffered frames
    fn flush(&mut self) -> Result<Vec<GpuFrame>>;
    
    /// Get output format
    fn output_format(&self) -> VideoFormat;
}

/// Hardware video encoder trait
pub trait HwVideoEncoder: Send {
    type Packet: AsRef<[u8]> + Send;
    
    /// Encode raw frame to compressed packet
    fn encode(&mut self, frame: &GpuFrame) -> Result<Vec<Self::Packet>>;
    
    /// Flush any buffered packets
    fn flush(&mut self) -> Result<Vec<Self::Packet>>;
}

/// GPU frame reference (zero-copy)
pub struct GpuFrame {
    pub buffer: GpuBuffer,
    pub format: PixelFormat,
    pub width: u32,
    pub height: u32,
    pub stride: u32,
    pub pts: i64,
}
```

---

## Implementation Steps

### Phase 1: Vulkan Foundation (1 week)

- [ ] Add dependencies: `ash` (Vulkan bindings), `gpu-allocator`
- [ ] Create `src/gpu/mod.rs` module structure
- [ ] Implement Vulkan instance/device creation with video extensions
- [ ] Query video decode/encode capabilities
- [ ] Implement `GpuMemory` trait for Vulkan
- [ ] Add DMA-BUF import/export via `VK_EXT_external_memory_dma_buf`

### Phase 2: H.264 Decode (1-2 weeks)

- [ ] Implement `VulkanH264Decoder` using `VK_KHR_video_decode_h264`
- [ ] Handle NAL unit parsing (use `h264-reader` crate)
- [ ] Implement reference frame management
- [ ] Handle DPB (Decoded Picture Buffer)
- [ ] Test with various H.264 profiles (Baseline, Main, High)
- [ ] Benchmark against CPU decoder

### Phase 3: H.265 Decode (1 week)

- [ ] Implement `VulkanH265Decoder` using `VK_KHR_video_decode_h265`
- [ ] Handle HEVC NAL unit parsing
- [ ] Implement reference frame management
- [ ] Test with H.265 Main/Main10 profiles

### Phase 4: AV1 Decode (1 week)

- [ ] Implement `VulkanAv1Decoder` using `VK_KHR_video_decode_av1`
- [ ] Handle OBU parsing
- [ ] Implement film grain synthesis (optional)
- [ ] Compare with dav1d performance

### Phase 5: Encode Support (1-2 weeks)

- [ ] Implement `VulkanH264Encoder` using `VK_KHR_video_encode_h264`
- [ ] Implement rate control (CBR, VBR)
- [ ] Implement `VulkanH265Encoder`
- [ ] Compare with rav1e/OpenH264 performance

### Phase 6: Pipeline Integration (1 week)

- [ ] Create `HwDecoderElement<D>` wrapper
- [ ] Create `HwEncoderElement<E>` wrapper
- [ ] Update caps negotiation for GPU memory types
- [ ] Implement automatic CPU↔GPU fallback
- [ ] Create examples demonstrating GPU decode/encode
- [ ] Update documentation

---

## Technical Details

### Vulkan Video Extensions

```rust
// Required extensions for video
const VIDEO_EXTENSIONS: &[&str] = &[
    "VK_KHR_video_queue",
    "VK_KHR_video_decode_queue",
    "VK_KHR_video_decode_h264",
    "VK_KHR_video_decode_h265",
    "VK_KHR_video_decode_av1",
    "VK_KHR_video_encode_queue",
    "VK_KHR_video_encode_h264",
    "VK_KHR_video_encode_h265",
];

// DMA-BUF import/export
const DMABUF_EXTENSIONS: &[&str] = &[
    "VK_KHR_external_memory",
    "VK_KHR_external_memory_fd",
    "VK_EXT_external_memory_dma_buf",
];
```

### Video Session Creation

```rust
pub struct VulkanVideoDecoder {
    device: Arc<ash::Device>,
    video_session: vk::VideoSessionKHR,
    video_session_params: vk::VideoSessionParametersKHR,
    dpb_images: Vec<vk::Image>,
    decode_queue: vk::Queue,
}

impl VulkanVideoDecoder {
    pub fn new(device: Arc<ash::Device>, profile: &VideoProfile) -> Result<Self> {
        // 1. Query video capabilities
        let caps = query_video_capabilities(&device, profile)?;
        
        // 2. Create video session
        let session_info = vk::VideoSessionCreateInfoKHR::builder()
            .queue_family_index(video_queue_family)
            .video_profile(profile.as_vulkan())
            .max_dpb_slots(caps.max_dpb_slots)
            .max_active_reference_pictures(caps.max_active_refs)
            .std_header_version(&caps.std_header_version)
            .build();
        
        let video_session = unsafe {
            device.create_video_session_khr(&session_info, None)?
        };
        
        // 3. Allocate DPB images
        // 4. Create session parameters
        
        Ok(Self { device, video_session, ... })
    }
}
```

### DMA-BUF Import

```rust
impl GpuMemory for VulkanGpuMemory {
    fn import_dmabuf(&mut self, fd: OwnedFd, size: usize) -> Result<GpuBuffer> {
        // Create external memory
        let import_info = vk::ImportMemoryFdInfoKHR::builder()
            .handle_type(vk::ExternalMemoryHandleTypeFlags::DMA_BUF_EXT)
            .fd(fd.as_raw_fd())
            .build();
        
        let alloc_info = vk::MemoryAllocateInfo::builder()
            .allocation_size(size as u64)
            .memory_type_index(external_memory_type)
            .push_next(&mut import_info)
            .build();
        
        let memory = unsafe {
            self.device.allocate_memory(&alloc_info, None)?
        };
        
        // Don't close fd - Vulkan owns it now
        std::mem::forget(fd);
        
        Ok(GpuBuffer { memory, size })
    }
}
```

---

## Feature Flags

```toml
[features]
# GPU codec support (requires Vulkan runtime)
vulkan-video = [
    "dep:ash",
    "dep:gpu-allocator",
    "dep:h264-reader",
    "dep:hevc-parser",
]

# Individual codec features
vulkan-h264-decode = ["vulkan-video"]
vulkan-h265-decode = ["vulkan-video"]
vulkan-av1-decode = ["vulkan-video"]
vulkan-h264-encode = ["vulkan-video"]
vulkan-h265-encode = ["vulkan-video"]
```

---

## Dependencies

| Crate | Purpose | License |
|-------|---------|---------|
| `ash` | Vulkan bindings | MIT/Apache-2.0 |
| `gpu-allocator` | GPU memory allocation | MIT/Apache-2.0 |
| `h264-reader` | H.264 NAL parsing | MIT/Apache-2.0 |
| `hevc-parser` | H.265 NAL parsing | MIT/Apache-2.0 |

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_vulkan_video_available() {
    if !vulkan_video_supported() {
        return; // Skip on systems without Vulkan Video
    }
    
    let ctx = VulkanContext::new().unwrap();
    assert!(ctx.supports_decode(Codec::H264));
}

#[test]
fn test_h264_decode_simple() {
    let decoder = VulkanH264Decoder::new(...)?;
    let packet = include_bytes!("../test_data/h264_idr.bin");
    let frames = decoder.decode(packet)?;
    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].width, 1920);
    assert_eq!(frames[0].height, 1080);
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_gpu_decode_pipeline() {
    if !vulkan_video_supported() {
        return;
    }
    
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node("src", FileSrc::new("test.h264"));
    let dec = pipeline.add_node("dec", VulkanH264Decoder::new()?);
    let sink = pipeline.add_node("sink", NullSink::new());
    
    pipeline.link_all(&[src, dec, sink])?;
    pipeline.run().await?;
}
```

---

## Performance Targets

| Operation | CPU (rav1e/dav1d) | GPU Target | Speedup |
|-----------|-------------------|------------|---------|
| H.264 1080p decode | 60 fps | 240+ fps | 4x |
| H.265 4K decode | 15 fps | 60+ fps | 4x |
| AV1 1080p decode | 30 fps | 120+ fps | 4x |
| H.264 1080p encode | 30 fps | 120+ fps | 4x |

---

## Fallback Behavior

```rust
pub fn create_decoder(codec: Codec, prefer_gpu: bool) -> Box<dyn VideoDecoder> {
    if prefer_gpu && vulkan_video_supported() {
        match VulkanDecoder::new(codec) {
            Ok(dec) => return Box::new(dec),
            Err(e) => log::warn!("GPU decode unavailable: {}, falling back to CPU", e),
        }
    }
    
    // CPU fallback
    match codec {
        Codec::H264 => Box::new(OpenH264Decoder::new()),
        Codec::H265 => panic!("No CPU H.265 decoder available"),
        Codec::Av1 => Box::new(Dav1dDecoder::new()),
    }
}
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/gpu/mod.rs` | GPU module root |
| `src/gpu/vulkan/mod.rs` | Vulkan backend |
| `src/gpu/vulkan/context.rs` | Device/instance management |
| `src/gpu/vulkan/memory.rs` | GPU memory, DMA-BUF |
| `src/gpu/vulkan/decode.rs` | Video decode |
| `src/gpu/vulkan/encode.rs` | Video encode |
| `src/gpu/traits.rs` | Hardware codec traits |
| `src/elements/codec/hw_decoder.rs` | HwDecoderElement |
| `src/elements/codec/hw_encoder.rs` | HwEncoderElement |
| `examples/18_gpu_decode.rs` | GPU decode example |
| `examples/19_gpu_transcode.rs` | Full transcode example |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Driver bugs | High | Test on multiple GPUs, graceful fallback |
| Limited testing hardware | Medium | CI with software Vulkan (lavapipe) |
| Complex Vulkan API | High | Use existing `ash` examples, incremental development |
| DMA-BUF compatibility | Medium | Test import/export paths thoroughly |

---

## Success Criteria

- [ ] Vulkan Video decode works on AMD (RADV) and Intel (ANV)
- [ ] H.264, H.265, AV1 decode all functional
- [ ] At least H.264 encode functional
- [ ] Zero-copy DMA-BUF path verified
- [ ] Performance meets targets
- [ ] Graceful fallback to CPU when GPU unavailable
- [ ] Examples demonstrate full pipeline

---

## References

- [Vulkan Video Extension Specification](https://registry.khronos.org/vulkan/specs/1.3-extensions/html/vkspec.html#video-coding)
- [Khronos Vulkan Video Blog Posts](https://www.khronos.org/blog/tag/vulkan-video)
- [ash-rs examples](https://github.com/ash-rs/ash/tree/master/examples)
- [FFmpeg Vulkan Video Implementation](https://git.ffmpeg.org/gitweb/ffmpeg.git/tree/HEAD:/libavcodec/vulkan)

---

*Created January 2026 based on [Project Analysis Report](../docs/PROJECT_ANALYSIS_REPORT.md)*
