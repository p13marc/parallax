# Plan: Vulkan Video Codec Elements for Parallax

## Overview

This plan adds hardware-accelerated video encoding and decoding to Parallax using Vulkan Video API. One implementation provides H.264, H.265, AV1, and VP9 support through a unified abstraction.

## Why Vulkan Video?

| Approach | Pros | Cons |
|----------|------|------|
| **Vulkan Video** | All codecs, hardware accel, portable, pure Rust possible | Complex API, driver-dependent |
| FFmpeg bindings | Battle-tested, all codecs | C dependency, licensing complexity |
| rav1e/rav1d | Pure Rust, memory-safe | AV1 only, CPU-based |

**Decision:** Vulkan Video as primary, with rav1e/rav1d as software fallback.

## Supported Operations (Vulkan 1.3.302+)

| Codec | Decode | Encode | Extension |
|-------|--------|--------|-----------|
| H.264 | ✅ | ✅ | `VK_KHR_video_decode_h264`, `VK_KHR_video_encode_h264` |
| H.265 | ✅ | ✅ | `VK_KHR_video_decode_h265`, `VK_KHR_video_encode_h265` |
| AV1 | ✅ | ✅ | `VK_KHR_video_decode_av1`, `VK_KHR_video_encode_av1` |
| VP9 | ✅ | ❌ | `VK_KHR_video_decode_vp9` |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Parallax Pipeline                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ VulkanH264Dec│    │ VulkanH265Dec│    │ VulkanAV1Dec │       │
│  │ VulkanH264Enc│    │ VulkanH265Enc│    │ VulkanAV1Enc │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                   │                │
│         └───────────────────┼───────────────────┘                │
│                             │                                    │
│                    ┌────────▼────────┐                          │
│                    │  VulkanCodec    │  (shared abstraction)    │
│                    │  - Session mgmt │                          │
│                    │  - DPB handling │                          │
│                    │  - Memory pools │                          │
│                    └────────┬────────┘                          │
│                             │                                    │
│         ┌───────────────────┼───────────────────┐                │
│         │                   │                   │                │
│  ┌──────▼───────┐   ┌───────▼──────┐   ┌───────▼──────┐        │
│  │ BitstreamParser│ │ VulkanDevice │   │  DPBManager  │        │
│  │ (h264-reader)  │ │   (ash)      │   │              │        │
│  └───────────────┘  └──────────────┘   └──────────────┘        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Foundation

### 1.1 Dependencies

**File:** `Cargo.toml`

```toml
[dependencies]
# Vulkan bindings
ash = "0.38"
ash-window = "0.13"

# Memory allocation
gpu-allocator = "0.27"

# Bitstream parsing
h264-reader = "0.7"
# h265-reader when available, or custom
# av1-parser for AV1

[features]
default = []
vulkan-video = ["ash", "gpu-allocator", "h264-reader"]
software-codecs = ["rav1e", "rav1d"]  # Fallback
```

### 1.2 Vulkan Instance and Device Setup

**File:** `src/codec/vulkan/instance.rs`

```rust
use ash::vk;
use ash::Entry;

pub struct VulkanVideoInstance {
    entry: Entry,
    instance: ash::Instance,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    video_queue: vk::Queue,
    video_queue_family: u32,
}

impl VulkanVideoInstance {
    pub fn new() -> Result<Self> {
        let entry = unsafe { Entry::load()? };
        
        // Required extensions
        let instance_extensions = [
            // KHR surface extensions if needed
        ];
        
        let device_extensions = [
            vk::KHR_VIDEO_QUEUE_NAME,
            vk::KHR_VIDEO_DECODE_QUEUE_NAME,
            vk::KHR_VIDEO_DECODE_H264_NAME,
            vk::KHR_VIDEO_DECODE_H265_NAME,
            vk::KHR_VIDEO_DECODE_AV1_NAME,
            vk::KHR_VIDEO_ENCODE_QUEUE_NAME,
            vk::KHR_VIDEO_ENCODE_H264_NAME,
            vk::KHR_VIDEO_ENCODE_H265_NAME,
            vk::KHR_VIDEO_ENCODE_AV1_NAME,
            vk::KHR_SYNCHRONIZATION_2_NAME,
        ];
        
        // Find physical device with video support
        // Create logical device with video queue
        // ...
    }
    
    /// Query supported codecs on this device
    pub fn supported_codecs(&self) -> SupportedCodecs {
        // Query VkVideoCapabilitiesKHR for each codec
    }
}

pub struct SupportedCodecs {
    pub h264_decode: bool,
    pub h264_encode: bool,
    pub h265_decode: bool,
    pub h265_encode: bool,
    pub av1_decode: bool,
    pub av1_encode: bool,
    pub vp9_decode: bool,
}
```

### 1.3 Video Session Abstraction

**File:** `src/codec/vulkan/session.rs`

```rust
pub struct VideoSession {
    session: vk::VideoSessionKHR,
    parameters: vk::VideoSessionParametersKHR,
    profile: VideoProfile,
    memory_bindings: Vec<vk::DeviceMemory>,
}

pub enum VideoProfile {
    DecodeH264 { profile: H264Profile },
    DecodeH265 { profile: H265Profile },
    DecodeAV1 { profile: AV1Profile },
    EncodeH264 { profile: H264Profile, level: H264Level },
    EncodeH265 { profile: H265Profile, level: H265Level },
    EncodeAV1 { profile: AV1Profile, level: AV1Level },
}

impl VideoSession {
    pub fn new_decode(
        instance: &VulkanVideoInstance,
        codec: VideoCodec,
        max_width: u32,
        max_height: u32,
    ) -> Result<Self>;
    
    pub fn new_encode(
        instance: &VulkanVideoInstance,
        codec: VideoCodec,
        width: u32,
        height: u32,
        config: EncoderConfig,
    ) -> Result<Self>;
    
    pub fn reset(&mut self) -> Result<()>;
}
```

### 1.4 Decoded Picture Buffer (DPB) Manager

**File:** `src/codec/vulkan/dpb.rs`

```rust
pub struct DPBManager {
    /// Image array for reference frames
    dpb_image: vk::Image,
    dpb_memory: vk::DeviceMemory,
    dpb_views: Vec<vk::ImageView>,
    
    /// Slot tracking
    max_slots: u32,
    active_slots: Vec<Option<DPBSlot>>,
    
    /// Reference frame queue (FIFO for eviction)
    reference_queue: VecDeque<ReferenceFrame>,
}

pub struct DPBSlot {
    pub slot_index: u32,
    pub frame_num: i32,      // H.264 frame_num
    pub poc: i32,            // Picture Order Count
    pub is_reference: bool,
}

pub struct ReferenceFrame {
    pub slot_index: u32,
    pub poc: i32,
    pub frame_type: FrameType,
}

impl DPBManager {
    pub fn new(
        instance: &VulkanVideoInstance,
        max_slots: u32,
        width: u32,
        height: u32,
        format: vk::Format,
    ) -> Result<Self>;
    
    /// Allocate slot for new decoded frame
    pub fn allocate_slot(&mut self, frame_info: &FrameInfo) -> Result<u32>;
    
    /// Mark frame as reference (keep in DPB)
    pub fn mark_as_reference(&mut self, slot: u32, poc: i32);
    
    /// Get reference frames for P/B frame decoding
    pub fn get_references(&self, ref_list: &[i32]) -> Vec<&DPBSlot>;
    
    /// Clear DPB (on IDR frame)
    pub fn clear(&mut self);
    
    /// Evict oldest non-reference frame
    pub fn evict_lru(&mut self) -> Option<u32>;
}
```

---

## Phase 2: Decoder Implementation

### 2.1 Bitstream Parser Integration

**File:** `src/codec/vulkan/parse/h264.rs`

```rust
use h264_reader::nal::{Nal, RefNal, UnitType};
use h264_reader::nal::pps::PicParameterSet;
use h264_reader::nal::sps::SeqParameterSet;
use h264_reader::nal::slice::SliceHeader;

pub struct H264Parser {
    sps_map: HashMap<u8, SeqParameterSet>,
    pps_map: HashMap<u8, PicParameterSet>,
}

pub struct ParsedFrame {
    pub nal_units: Vec<NalUnit>,
    pub frame_type: FrameType,
    pub frame_num: u32,
    pub poc: i32,
    pub reference_list_l0: Vec<i32>,
    pub reference_list_l1: Vec<i32>,  // For B-frames
    pub sps: Option<SeqParameterSet>,
    pub pps: Option<PicParameterSet>,
}

pub enum FrameType {
    I,  // Intra (keyframe)
    P,  // Predicted (forward reference)
    B,  // Bi-directional (forward + backward reference)
}

impl H264Parser {
    pub fn parse_access_unit(&mut self, data: &[u8]) -> Result<ParsedFrame>;
}
```

Similar parsers for H.265 and AV1.

### 2.2 Generic Decoder Trait

**File:** `src/codec/vulkan/decoder.rs`

```rust
pub trait VideoDecoder: Send {
    /// Decode a single access unit (frame)
    fn decode(&mut self, bitstream: &[u8]) -> Result<DecodedFrame>;
    
    /// Flush decoder (process remaining frames)
    fn flush(&mut self) -> Result<Vec<DecodedFrame>>;
    
    /// Get output format
    fn output_format(&self) -> VideoFormat;
    
    /// Reset decoder state
    fn reset(&mut self) -> Result<()>;
}

pub struct DecodedFrame {
    pub image: vk::Image,
    pub image_view: vk::ImageView,
    pub format: VideoFormat,
    pub pts: Option<u64>,
    pub dts: Option<u64>,
    pub frame_type: FrameType,
}
```

### 2.3 H.264 Decoder

**File:** `src/codec/vulkan/h264_decoder.rs`

```rust
pub struct VulkanH264Decoder {
    instance: Arc<VulkanVideoInstance>,
    session: VideoSession,
    dpb: DPBManager,
    parser: H264Parser,
    
    /// Bitstream buffer (host-mapped)
    bitstream_buffer: vk::Buffer,
    bitstream_memory: vk::DeviceMemory,
    bitstream_ptr: *mut u8,
    
    /// Output image (decoded frame)
    output_image: vk::Image,
    output_view: vk::ImageView,
    
    /// Command buffer for decode operations
    command_pool: vk::CommandPool,
    command_buffer: vk::CommandBuffer,
}

impl VulkanH264Decoder {
    pub fn new(instance: Arc<VulkanVideoInstance>, config: DecoderConfig) -> Result<Self> {
        // 1. Create video session with H.264 decode profile
        let profile = VideoProfile::DecodeH264 {
            profile: H264Profile::High,
        };
        let session = VideoSession::new_decode(&instance, profile, config.max_width, config.max_height)?;
        
        // 2. Create DPB
        let dpb = DPBManager::new(&instance, 16, config.max_width, config.max_height, vk::Format::G8_B8R8_2PLANE_420_UNORM)?;
        
        // 3. Create bitstream buffer (aligned)
        let bitstream_buffer = create_bitstream_buffer(&instance, BITSTREAM_BUFFER_SIZE)?;
        
        // 4. Create output image
        let output_image = create_decode_output_image(&instance, config.max_width, config.max_height)?;
        
        Ok(Self { /* ... */ })
    }
}

impl VideoDecoder for VulkanH264Decoder {
    fn decode(&mut self, bitstream: &[u8]) -> Result<DecodedFrame> {
        // 1. Parse NAL units
        let parsed = self.parser.parse_access_unit(bitstream)?;
        
        // 2. Update session parameters if SPS/PPS changed
        if let Some(sps) = &parsed.sps {
            self.update_sps(sps)?;
        }
        if let Some(pps) = &parsed.pps {
            self.update_pps(pps)?;
        }
        
        // 3. Copy bitstream to GPU buffer
        self.upload_bitstream(bitstream)?;
        
        // 4. Setup reference pictures
        let references = self.dpb.get_references(&parsed.reference_list_l0);
        let reference_slots: Vec<vk::VideoReferenceSlotInfoKHR> = /* build from references */;
        
        // 5. Allocate DPB slot for output
        let output_slot = self.dpb.allocate_slot(&parsed.into())?;
        
        // 6. Record decode command
        unsafe {
            let begin_info = vk::VideoBeginCodingInfoKHR::default()
                .video_session(self.session.session)
                .video_session_parameters(self.session.parameters)
                .reference_slots(&reference_slots);
            
            self.instance.device.cmd_begin_video_coding_khr(
                self.command_buffer,
                &begin_info,
            );
            
            let decode_info = vk::VideoDecodeInfoKHR::default()
                .src_buffer(self.bitstream_buffer)
                .src_buffer_offset(0)
                .src_buffer_range(bitstream.len() as u64)
                .dst_picture_resource(&output_resource)
                .setup_reference_slot(&setup_slot)
                .reference_slots(&reference_slots);
            
            self.instance.device.cmd_decode_video_khr(
                self.command_buffer,
                &decode_info,
            );
            
            self.instance.device.cmd_end_video_coding_khr(self.command_buffer);
        }
        
        // 7. Submit and wait
        self.submit_and_wait()?;
        
        // 8. Mark as reference if needed
        if parsed.frame_type != FrameType::B {
            self.dpb.mark_as_reference(output_slot, parsed.poc);
        }
        
        Ok(DecodedFrame {
            image: self.output_image,
            image_view: self.output_view,
            format: self.output_format(),
            pts: None,  // Caller provides
            dts: None,
            frame_type: parsed.frame_type,
        })
    }
}
```

### 2.4 Parallax Decoder Element

**File:** `src/elements/vulkan_decoder.rs`

```rust
use crate::codec::vulkan::{VulkanVideoInstance, VulkanH264Decoder, VulkanH265Decoder, VulkanAV1Decoder};

pub struct VulkanVideoDecoderElement {
    name: String,
    instance: Arc<VulkanVideoInstance>,
    decoder: Box<dyn VideoDecoder>,
    codec: VideoCodec,
}

impl VulkanVideoDecoderElement {
    pub fn h264(instance: Arc<VulkanVideoInstance>) -> Result<Self> {
        let decoder = VulkanH264Decoder::new(instance.clone(), DecoderConfig::default())?;
        Ok(Self {
            name: "vulkan_h264_dec".to_string(),
            instance,
            decoder: Box::new(decoder),
            codec: VideoCodec::H264,
        })
    }
    
    pub fn h265(instance: Arc<VulkanVideoInstance>) -> Result<Self>;
    pub fn av1(instance: Arc<VulkanVideoInstance>) -> Result<Self>;
    
    /// Auto-detect codec from input
    pub fn auto(instance: Arc<VulkanVideoInstance>) -> Self;
}

impl Element for VulkanVideoDecoderElement {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let bitstream = buffer.as_bytes();
        let decoded = self.decoder.decode(bitstream)?;
        
        // Convert Vulkan image to Parallax buffer
        // Option 1: Copy to CPU memory (simple but slow)
        // Option 2: Keep on GPU, pass handle (zero-copy)
        let output = self.vulkan_image_to_buffer(decoded)?;
        
        Ok(Some(output))
    }
    
    fn input_caps(&self) -> Caps {
        match self.codec {
            VideoCodec::H264 => Caps::new(MediaFormat::Video(VideoCodec::H264)),
            VideoCodec::H265 => Caps::new(MediaFormat::Video(VideoCodec::H265)),
            VideoCodec::AV1 => Caps::new(MediaFormat::Video(VideoCodec::AV1)),
        }
    }
    
    fn output_caps(&self) -> Caps {
        // Decoded raw video
        Caps::new(MediaFormat::VideoRaw(self.decoder.output_format()))
    }
}
```

---

## Phase 3: Encoder Implementation

### 3.1 Encoder Configuration

**File:** `src/codec/vulkan/encoder.rs`

```rust
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub framerate: Framerate,
    pub bitrate: Bitrate,
    pub rate_control: RateControlMode,
    pub gop_size: u32,        // Frames between keyframes
    pub b_frames: u32,        // Number of B-frames
    pub quality_preset: QualityPreset,
}

pub enum Bitrate {
    Constant(u32),           // CBR in kbps
    Variable { target: u32, max: u32 },  // VBR
    Quality(u32),            // CQ mode (0-51 for H.264)
}

pub enum RateControlMode {
    CBR,
    VBR,
    CQ,  // Constant Quality
}

pub enum QualityPreset {
    Speed,      // Fastest encoding
    Balanced,   // Default
    Quality,    // Best quality
}
```

### 3.2 Generic Encoder Trait

```rust
pub trait VideoEncoder: Send {
    /// Encode a single frame
    fn encode(&mut self, frame: &RawFrame) -> Result<EncodedPacket>;
    
    /// Flush encoder (get remaining packets)
    fn flush(&mut self) -> Result<Vec<EncodedPacket>>;
    
    /// Get encoder parameters (SPS/PPS for H.264)
    fn codec_data(&self) -> &[u8];
    
    /// Force keyframe on next encode
    fn force_keyframe(&mut self);
}

pub struct RawFrame {
    pub data: Vec<u8>,      // Or Vulkan image handle
    pub format: VideoFormat,
    pub pts: u64,
}

pub struct EncodedPacket {
    pub data: Vec<u8>,
    pub pts: u64,
    pub dts: u64,
    pub is_keyframe: bool,
    pub frame_type: FrameType,
}
```

### 3.3 H.264 Encoder

**File:** `src/codec/vulkan/h264_encoder.rs`

```rust
pub struct VulkanH264Encoder {
    instance: Arc<VulkanVideoInstance>,
    session: VideoSession,
    dpb: DPBManager,
    config: EncoderConfig,
    
    /// Rate control state
    rate_control: RateControlState,
    
    /// Frame counter
    frame_num: u64,
    
    /// GOP tracking
    frames_since_keyframe: u32,
    force_keyframe: bool,
    
    /// Output bitstream buffer
    output_buffer: vk::Buffer,
    output_memory: vk::DeviceMemory,
}

impl VideoEncoder for VulkanH264Encoder {
    fn encode(&mut self, frame: &RawFrame) -> Result<EncodedPacket> {
        // 1. Determine frame type
        let frame_type = self.determine_frame_type();
        
        // 2. Upload input frame to GPU (or use existing GPU image)
        self.upload_frame(frame)?;
        
        // 3. Setup reference pictures (for P/B frames)
        let references = if frame_type != FrameType::I {
            self.dpb.get_references_for_encode(frame_type)
        } else {
            self.dpb.clear();
            vec![]
        };
        
        // 4. Configure rate control for this frame
        let rc_info = self.rate_control.for_frame(frame_type)?;
        
        // 5. Record encode command
        unsafe {
            self.instance.device.cmd_begin_video_coding_khr(/* ... */);
            
            // Set rate control
            self.instance.device.cmd_control_video_coding_khr(
                self.command_buffer,
                &control_info,
            );
            
            let encode_info = vk::VideoEncodeInfoKHR::default()
                .dst_buffer(self.output_buffer)
                .dst_buffer_offset(0)
                .src_picture_resource(&input_resource)
                .setup_reference_slot(&setup_slot)
                .reference_slots(&reference_slots);
            
            self.instance.device.cmd_encode_video_khr(
                self.command_buffer,
                &encode_info,
            );
            
            self.instance.device.cmd_end_video_coding_khr(self.command_buffer);
        }
        
        // 6. Submit and wait
        self.submit_and_wait()?;
        
        // 7. Read encoded data
        let encoded_data = self.read_output_buffer()?;
        
        // 8. Update DPB if this frame is a reference
        if frame_type != FrameType::B {
            self.dpb.add_reference(/* ... */);
        }
        
        self.frame_num += 1;
        self.frames_since_keyframe += 1;
        
        Ok(EncodedPacket {
            data: encoded_data,
            pts: frame.pts,
            dts: self.calculate_dts(),
            is_keyframe: frame_type == FrameType::I,
            frame_type,
        })
    }
}
```

### 3.4 Parallax Encoder Element

**File:** `src/elements/vulkan_encoder.rs`

```rust
pub struct VulkanVideoEncoderElement {
    name: String,
    instance: Arc<VulkanVideoInstance>,
    encoder: Box<dyn VideoEncoder>,
    codec: VideoCodec,
    config: EncoderConfig,
}

impl VulkanVideoEncoderElement {
    pub fn h264(instance: Arc<VulkanVideoInstance>, config: EncoderConfig) -> Result<Self>;
    pub fn h265(instance: Arc<VulkanVideoInstance>, config: EncoderConfig) -> Result<Self>;
    pub fn av1(instance: Arc<VulkanVideoInstance>, config: EncoderConfig) -> Result<Self>;
}

impl Element for VulkanVideoEncoderElement {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let frame = RawFrame::from_buffer(&buffer)?;
        let packet = self.encoder.encode(&frame)?;
        
        let output = Buffer::from_bytes(packet.data);
        // Add metadata: pts, dts, keyframe flag
        
        Ok(Some(output))
    }
    
    fn input_caps(&self) -> Caps {
        // Accept raw video in configured format
        Caps::new(MediaFormat::VideoRaw(VideoFormat {
            width: self.config.width,
            height: self.config.height,
            pixel_format: PixelFormat::NV12,  // or I420
            framerate: self.config.framerate,
        }))
    }
    
    fn output_caps(&self) -> Caps {
        match self.codec {
            VideoCodec::H264 => Caps::new(MediaFormat::Video(VideoCodec::H264)),
            VideoCodec::H265 => Caps::new(MediaFormat::Video(VideoCodec::H265)),
            VideoCodec::AV1 => Caps::new(MediaFormat::Video(VideoCodec::AV1)),
        }
    }
}
```

---

## Phase 4: YUV/RGB Conversion

### 4.1 Vulkan YCbCr Sampler

Decoded video is typically YUV (NV12/I420). For display or further processing, convert to RGB:

**File:** `src/codec/vulkan/color_convert.rs`

```rust
pub struct YuvToRgbConverter {
    instance: Arc<VulkanVideoInstance>,
    sampler: vk::Sampler,
    ycbcr_conversion: vk::SamplerYcbcrConversion,
    pipeline: vk::Pipeline,
    descriptor_set: vk::DescriptorSet,
}

impl YuvToRgbConverter {
    pub fn new(instance: Arc<VulkanVideoInstance>, input_format: vk::Format) -> Result<Self> {
        // Create YCbCr conversion sampler
        let ycbcr_info = vk::SamplerYcbcrConversionCreateInfo::default()
            .format(input_format)
            .ycbcr_model(vk::SamplerYcbcrModelConversion::YCBCR_709)
            .ycbcr_range(vk::SamplerYcbcrRange::ITU_NARROW)
            .components(vk::ComponentMapping::default())
            .x_chroma_offset(vk::ChromaLocation::MIDPOINT)
            .y_chroma_offset(vk::ChromaLocation::MIDPOINT)
            .chroma_filter(vk::Filter::LINEAR);
        
        // Create compute pipeline for conversion
        // ...
    }
    
    pub fn convert(&self, yuv_image: vk::Image, rgb_image: vk::Image) -> Result<()>;
}
```

### 4.2 Color Convert Element

```rust
pub struct VulkanColorConvertElement {
    name: String,
    converter: YuvToRgbConverter,
    input_format: PixelFormat,
    output_format: PixelFormat,
}

impl Element for VulkanColorConvertElement {
    fn input_caps(&self) -> Caps {
        Caps::new(MediaFormat::VideoRaw(VideoFormat {
            pixel_format: self.input_format,
            ..VideoFormat::any()
        }))
    }
    
    fn output_caps(&self) -> Caps {
        Caps::new(MediaFormat::VideoRaw(VideoFormat {
            pixel_format: self.output_format,
            ..VideoFormat::any()
        }))
    }
}
```

---

## Phase 5: Software Fallback

### 5.1 rav1e/rav1d Integration

When Vulkan Video is unavailable:

**File:** `src/codec/software/av1.rs`

```rust
#[cfg(feature = "software-codecs")]
pub struct Rav1eEncoder {
    encoder: rav1e::Context<u8>,
    config: EncoderConfig,
}

#[cfg(feature = "software-codecs")]
impl VideoEncoder for Rav1eEncoder {
    fn encode(&mut self, frame: &RawFrame) -> Result<EncodedPacket> {
        // Convert to rav1e Frame
        // Encode
        // Return packet
    }
}

#[cfg(feature = "software-codecs")]
pub struct Rav1dDecoder {
    decoder: rav1d::Decoder,
}

#[cfg(feature = "software-codecs")]
impl VideoDecoder for Rav1dDecoder {
    fn decode(&mut self, bitstream: &[u8]) -> Result<DecodedFrame> {
        // Decode with rav1d
    }
}
```

### 5.2 Automatic Backend Selection

```rust
pub fn create_decoder(codec: VideoCodec) -> Result<Box<dyn VideoDecoder>> {
    // Try Vulkan first
    if let Ok(instance) = VulkanVideoInstance::new() {
        if instance.supported_codecs().supports(codec, Operation::Decode) {
            return match codec {
                VideoCodec::H264 => Ok(Box::new(VulkanH264Decoder::new(instance)?)),
                VideoCodec::H265 => Ok(Box::new(VulkanH265Decoder::new(instance)?)),
                VideoCodec::AV1 => Ok(Box::new(VulkanAV1Decoder::new(instance)?)),
                _ => {}
            };
        }
    }
    
    // Fall back to software
    #[cfg(feature = "software-codecs")]
    match codec {
        VideoCodec::AV1 => Ok(Box::new(Rav1dDecoder::new()?)),
        _ => Err(Error::CodecNotSupported(codec)),
    }
    
    #[cfg(not(feature = "software-codecs"))]
    Err(Error::CodecNotSupported(codec))
}
```

---

## Phase 6: wgpu Integration

### 6.1 Zero-Copy GPU Textures

For rendering decoded frames without CPU round-trip:

**File:** `src/codec/vulkan/wgpu_interop.rs`

```rust
pub struct VulkanWgpuBridge {
    vulkan_instance: Arc<VulkanVideoInstance>,
    wgpu_device: wgpu::Device,
    wgpu_queue: wgpu::Queue,
}

impl VulkanWgpuBridge {
    /// Import Vulkan image as wgpu texture (zero-copy)
    pub fn import_vulkan_image(&self, image: vk::Image) -> Result<wgpu::Texture> {
        // Use VK_KHR_external_memory to share
        // Create wgpu texture from external memory handle
    }
}
```

### 6.2 Integration with IcedVideoSink

```rust
impl IcedVideoSink {
    pub fn with_vulkan_input(bridge: VulkanWgpuBridge) -> Self {
        // Accept decoded frames directly on GPU
    }
}
```

---

## Implementation Order

### Milestone 1: Vulkan Foundation
1. [ ] Add ash dependency
2. [ ] Implement VulkanVideoInstance
3. [ ] Query device capabilities
4. [ ] Create video session abstraction
5. [ ] Implement DPB manager

### Milestone 2: H.264 Decoder
1. [ ] Integrate h264-reader for parsing
2. [ ] Implement VulkanH264Decoder
3. [ ] Create decoder element
4. [ ] Test with sample H.264 streams
5. [ ] Add YUV→RGB conversion

### Milestone 3: H.265 and AV1 Decoders
1. [ ] Add H.265 parser
2. [ ] Implement VulkanH265Decoder
3. [ ] Add AV1 parser
4. [ ] Implement VulkanAV1Decoder
5. [ ] Test all decoders

### Milestone 4: H.264 Encoder
1. [ ] Implement VulkanH264Encoder
2. [ ] Rate control (CBR, VBR)
3. [ ] GOP structure (I/P/B frames)
4. [ ] Create encoder element
5. [ ] Test encoding pipeline

### Milestone 5: H.265 and AV1 Encoders
1. [ ] Implement VulkanH265Encoder
2. [ ] Implement VulkanAV1Encoder
3. [ ] Test all encoders

### Milestone 6: Software Fallback
1. [ ] Integrate rav1e
2. [ ] Integrate rav1d
3. [ ] Automatic backend selection
4. [ ] Test on systems without Vulkan Video

### Milestone 7: wgpu Integration
1. [ ] Implement Vulkan↔wgpu interop
2. [ ] Zero-copy decode to wgpu texture
3. [ ] Update IcedVideoSink

---

## Testing Strategy

### Unit Tests
- Vulkan instance creation and capability query
- DPB slot allocation and eviction
- Bitstream parsing (H.264, H.265, AV1)
- Rate control calculations

### Integration Tests
```rust
#[test]
fn test_h264_decode() {
    let instance = VulkanVideoInstance::new().unwrap();
    let decoder = VulkanH264Decoder::new(instance, config).unwrap();
    
    let bitstream = include_bytes!("test_data/sample.h264");
    let frame = decoder.decode(bitstream).unwrap();
    
    assert_eq!(frame.format.width, 1920);
    assert_eq!(frame.format.height, 1080);
}

#[test]
fn test_encode_decode_roundtrip() {
    let raw_frame = create_test_frame(1920, 1080);
    
    let encoder = VulkanH264Encoder::new(instance.clone(), config).unwrap();
    let encoded = encoder.encode(&raw_frame).unwrap();
    
    let decoder = VulkanH264Decoder::new(instance, config).unwrap();
    let decoded = decoder.decode(&encoded.data).unwrap();
    
    // Compare decoded with original (allow for compression loss)
    assert_frames_similar(&raw_frame, &decoded);
}
```

### Performance Benchmarks
- Decode throughput (frames/sec)
- Encode throughput at various quality levels
- GPU memory usage
- Latency (time from input to output)

---

## File Structure

```
src/
├── codec/
│   ├── mod.rs              # Codec module exports
│   ├── traits.rs           # VideoDecoder, VideoEncoder traits
│   ├── vulkan/
│   │   ├── mod.rs
│   │   ├── instance.rs     # VulkanVideoInstance
│   │   ├── session.rs      # VideoSession
│   │   ├── dpb.rs          # DPBManager
│   │   ├── memory.rs       # Buffer allocation helpers
│   │   ├── h264_decoder.rs
│   │   ├── h265_decoder.rs
│   │   ├── av1_decoder.rs
│   │   ├── h264_encoder.rs
│   │   ├── h265_encoder.rs
│   │   ├── av1_encoder.rs
│   │   ├── color_convert.rs
│   │   ├── wgpu_interop.rs
│   │   └── parse/
│   │       ├── mod.rs
│   │       ├── h264.rs
│   │       ├── h265.rs
│   │       └── av1.rs
│   └── software/
│       ├── mod.rs
│       ├── rav1e.rs        # AV1 software encoder
│       └── rav1d.rs        # AV1 software decoder
├── elements/
│   ├── vulkan_decoder.rs   # VulkanVideoDecoderElement
│   ├── vulkan_encoder.rs   # VulkanVideoEncoderElement
│   └── vulkan_convert.rs   # VulkanColorConvertElement
```

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Driver support varies | Query capabilities, fall back to software |
| Complex Vulkan API | Use ash's safe wrappers, extensive testing |
| Memory alignment requirements | Query device limits, validate at runtime |
| Bitstream parsing complexity | Use existing crates (h264-reader) |
| Performance issues | Profile early, use GPU-optimal paths |
| Breaking ash changes | Pin ash version, wrap in abstraction |

---

## References

### Vulkan Video Documentation
- [Vulkan Video Spec](https://docs.vulkan.org/spec/latest/chapters/videocoding.html)
- [Khronos Blog: Vulkan Video Intro](https://www.khronos.org/blog/an-introduction-to-vulkan-video)
- [NVIDIA Vulkan Video Samples](https://github.com/nvpro-samples/vk_video_samples)
- [Wicked Engine: Vulkan Video Decoding](https://wickedengine.net/2023/05/vulkan-video-decoding/)

### Rust Crates
- [ash](https://crates.io/crates/ash) - Vulkan bindings
- [h264-reader](https://crates.io/crates/h264-reader) - H.264 parsing
- [rav1e](https://crates.io/crates/rav1e) - AV1 encoder
- [rav1d](https://crates.io/crates/rav1d) - AV1 decoder
- [gpu-allocator](https://crates.io/crates/gpu-allocator) - GPU memory allocation

### Implementation Examples
- [FFmpeg Vulkan Video](https://github.com/FFmpeg/FFmpeg/tree/master/libavcodec)
- [GStreamer Vulkan Video](https://gitlab.freedesktop.org/gstreamer/gstreamer/-/tree/main/subprojects/gst-plugins-bad/sys/vulkan)
- [vulkan_video crate](https://github.com/ralfbiedert/vulkan_video)
