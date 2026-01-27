# Plan 02: Codec Element Wrappers

**Priority:** High (Immediate)  
**Effort:** Medium (3-5 days)  
**Dependencies:** Plan 01 (Custom Metadata API) - for SEI/OBU attachment  
**Addresses:** Pain Point 1.4 (Encoder/Decoder as Elements)

---

## Problem Statement

Codecs (encoders/decoders) don't fit the `Element` trait model because:

1. **Latency/Buffering:** Encoders may buffer N frames before producing output (B-frames, lookahead)
2. **Variable Output:** One input frame may produce 0, 1, or multiple output packets
3. **Flush Requirement:** Must drain buffered frames at end-of-stream
4. **Complex Configuration:** Codec settings don't fit simple property model

Currently, `Rav1eEncoder` and `OpenH264Encoder` are standalone structs requiring manual `encode_frame()` and `flush()` calls, bypassing the Pipeline API.

---

## Proposed Solution

Create generic wrapper elements that adapt codec interfaces to the Pipeline element model:

```rust
// Wraps any encoder implementing a codec trait
let encoder_node = pipeline.add_node(
    "av1_encoder",
    DynAsyncElement::new_box(EncoderElement::new(Rav1eEncoder::new(config)?)),
);
// Executor handles buffering, multiple outputs, and EOS flush automatically
```

---

## Design

### Codec Traits

Define standard traits that codecs must implement:

```rust
/// Trait for video encoders
pub trait VideoEncoder: Send {
    /// Encoder configuration type
    type Config;
    
    /// Encoded packet type (usually Vec<u8> or custom struct)
    type Packet: AsRef<[u8]> + Send;
    
    /// Create encoder with configuration
    fn new(config: Self::Config) -> Result<Self> where Self: Sized;
    
    /// Encode a single frame, may return 0 or more packets
    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Self::Packet>>;
    
    /// Flush buffered frames, called at EOS
    fn flush(&mut self) -> Result<Vec<Self::Packet>>;
    
    /// Get codec-specific metadata (SPS/PPS for H.264, sequence header for AV1)
    fn codec_data(&self) -> Option<Vec<u8>> { None }
    
    /// Check if encoder has pending buffered frames
    fn has_pending(&self) -> bool { false }
}

/// Trait for video decoders
pub trait VideoDecoder: Send {
    type Config;
    
    /// Decode a packet, may return 0 or more frames
    fn decode(&mut self, packet: &[u8]) -> Result<Vec<VideoFrame>>;
    
    /// Flush buffered frames
    fn flush(&mut self) -> Result<Vec<VideoFrame>>;
}
```

### Video Frame Type

```rust
/// Represents a raw video frame
pub struct VideoFrame {
    /// Pixel data (Y, U, V planes for YUV formats)
    pub data: Vec<u8>,
    /// Frame width
    pub width: u32,
    /// Frame height  
    pub height: u32,
    /// Pixel format
    pub format: PixelFormat,
    /// Presentation timestamp
    pub pts: ClockTime,
    /// Decode timestamp (for B-frames)
    pub dts: Option<ClockTime>,
    /// Frame type hint
    pub frame_type: FrameType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    I420,      // YUV 4:2:0 planar
    NV12,      // YUV 4:2:0 semi-planar
    I422,      // YUV 4:2:2 planar
    I444,      // YUV 4:4:4 planar
    Rgb24,     // RGB 8-bit
    Rgba32,    // RGBA 8-bit
    Bgra32,    // BGRA 8-bit
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrameType {
    #[default]
    Unknown,
    Key,       // I-frame
    Inter,     // P-frame
    BiPred,    // B-frame
}
```

### EncoderElement Wrapper

```rust
/// Wraps a VideoEncoder to work as a pipeline Transform element.
/// Handles buffering, multiple outputs, and EOS flushing.
pub struct EncoderElement<E: VideoEncoder> {
    encoder: E,
    /// Queue of packets ready to be emitted
    pending_packets: VecDeque<E::Packet>,
    /// Whether we've seen EOS and need to flush
    flushing: bool,
    /// Whether flush is complete
    flushed: bool,
    /// Statistics
    frames_in: u64,
    packets_out: u64,
}

impl<E: VideoEncoder> EncoderElement<E> {
    pub fn new(encoder: E) -> Self {
        Self {
            encoder,
            pending_packets: VecDeque::new(),
            flushing: false,
            flushed: false,
            frames_in: 0,
            packets_out: 0,
        }
    }
}
```

### Transform Implementation for EncoderElement

```rust
impl<E: VideoEncoder + 'static> Transform for EncoderElement<E> {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        // If we have pending packets from previous call, emit one
        if let Some(packet) = self.pending_packets.pop_front() {
            // Re-queue current buffer for next call
            // ... this is tricky, see alternative below
            return Ok(Output::single(self.packet_to_buffer(packet)?));
        }
        
        // Convert buffer to VideoFrame
        let frame = self.buffer_to_frame(&buffer)?;
        
        // Encode
        self.frames_in += 1;
        let packets = self.encoder.encode(&frame)?;
        
        // Queue all packets
        for pkt in packets {
            self.pending_packets.push_back(pkt);
        }
        
        // Return first packet (or None if encoder is still buffering)
        match self.pending_packets.pop_front() {
            Some(packet) => {
                self.packets_out += 1;
                Ok(Output::single(self.packet_to_buffer(packet)?))
            }
            None => Ok(Output::None),  // Encoder buffering, no output yet
        }
    }
    
    fn name(&self) -> &str {
        "EncoderElement"
    }
}
```

### Handling EOS and Flush

The executor needs to signal EOS to elements. We need a new method:

```rust
pub trait Transform: Send {
    fn transform(&mut self, buffer: Buffer) -> Result<Output>;
    
    /// Called when end-of-stream is reached. Flush any buffered data.
    /// Default implementation returns empty output.
    fn flush(&mut self) -> Result<Output> {
        Ok(Output::None)
    }
}

impl<E: VideoEncoder + 'static> Transform for EncoderElement<E> {
    fn flush(&mut self) -> Result<Output> {
        if self.flushed {
            return Ok(Output::None);
        }
        
        // Get any remaining pending packets
        if let Some(packet) = self.pending_packets.pop_front() {
            return Ok(Output::single(self.packet_to_buffer(packet)?));
        }
        
        // Flush encoder
        let packets = self.encoder.flush()?;
        for pkt in packets {
            self.pending_packets.push_back(pkt);
        }
        
        // Return first flushed packet
        match self.pending_packets.pop_front() {
            Some(packet) => {
                self.packets_out += 1;
                Ok(Output::single(self.packet_to_buffer(packet)?))
            }
            None => {
                self.flushed = true;
                Ok(Output::None)
            }
        }
    }
}
```

### Alternative: Use Output::Multiple

Instead of managing pending queue, return all packets at once:

```rust
impl<E: VideoEncoder + 'static> Transform for EncoderElement<E> {
    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let frame = self.buffer_to_frame(&buffer)?;
        self.frames_in += 1;
        
        let packets = self.encoder.encode(&frame)?;
        
        if packets.is_empty() {
            return Ok(Output::None);  // Still buffering
        }
        
        let buffers: Vec<Buffer> = packets
            .into_iter()
            .map(|p| self.packet_to_buffer(p))
            .collect::<Result<Vec<_>>>()?;
        
        self.packets_out += buffers.len() as u64;
        Ok(Output::from(buffers))
    }
}
```

**Recommendation:** Use `Output::Multiple` for simplicity. The executor already handles multiple outputs.

---

## Implementation Steps

### Step 1: Define Codec Traits

**File:** `src/elements/codec/traits.rs`

```rust
pub trait VideoEncoder: Send { ... }
pub trait VideoDecoder: Send { ... }
pub struct VideoFrame { ... }
pub enum PixelFormat { ... }
pub enum FrameType { ... }
```

### Step 2: Implement EncoderElement

**File:** `src/elements/codec/encoder_element.rs`

```rust
pub struct EncoderElement<E: VideoEncoder> { ... }

impl<E: VideoEncoder + 'static> Transform for EncoderElement<E> { ... }
```

### Step 3: Implement DecoderElement

**File:** `src/elements/codec/decoder_element.rs`

```rust
pub struct DecoderElement<D: VideoDecoder> { ... }

impl<D: VideoDecoder + 'static> Transform for DecoderElement<D> { ... }
```

### Step 4: Adapt Existing Codecs

**File:** `src/elements/codec/rav1e.rs`

```rust
impl VideoEncoder for Rav1eEncoder {
    type Config = Rav1eConfig;
    type Packet = Vec<u8>;
    
    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Self::Packet>> {
        // Adapt existing encode_frame() method
    }
    
    fn flush(&mut self) -> Result<Vec<Self::Packet>> {
        // Adapt existing flush() method
    }
}
```

**File:** `src/elements/codec/openh264.rs`

```rust
impl VideoEncoder for OpenH264Encoder {
    type Config = OpenH264Config;
    type Packet = Vec<u8>;
    
    fn encode(&mut self, frame: &VideoFrame) -> Result<Vec<Self::Packet>> { ... }
    fn flush(&mut self) -> Result<Vec<Self::Packet>> { ... }
}
```

### Step 5: Add flush() to Transform Trait

**File:** `src/element/traits.rs`

Add `flush()` method with default empty implementation.

### Step 6: Update Executor to Call flush()

**File:** `src/pipeline/unified_executor.rs`

When EOS is received, call `element.flush()` repeatedly until it returns `Output::None`.

### Step 7: Create Example

**File:** `examples/32_encoder_element.rs`

```rust
// Create pipeline with encoder as proper element
let mut pipeline = Pipeline::new();

let src = pipeline.add_node("src", ...);
let encoder = pipeline.add_node(
    "encoder",
    DynAsyncElement::new_box(TransformAdapter::new(
        EncoderElement::new(Rav1eEncoder::new(config)?)
    )),
);
let sink = pipeline.add_node("sink", ...);

pipeline.link(src, encoder)?;
pipeline.link(encoder, sink)?;

pipeline.run().await?;
// Encoder flush happens automatically at EOS!
```

---

## Timestamp Handling

Encoders reorder frames (B-frames), so PTS/DTS handling is critical:

```rust
impl<E: VideoEncoder> EncoderElement<E> {
    fn packet_to_buffer(&self, packet: E::Packet, pts: ClockTime, dts: Option<ClockTime>) -> Result<Buffer> {
        let data = packet.as_ref();
        let segment = Arc::new(HeapSegment::new(data.len())?);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), segment.as_mut_ptr().unwrap(), data.len());
        }
        
        let mut metadata = Metadata::default();
        metadata.timestamp = pts;
        // DTS may differ from PTS for B-frames
        if let Some(dts) = dts {
            metadata.set("codec/dts", dts);
        }
        
        Ok(Buffer::new(MemoryHandle::from_segment(segment), metadata))
    }
}
```

---

## Validation Criteria

- [ ] `VideoEncoder` and `VideoDecoder` traits defined
- [ ] `EncoderElement<E>` wrapper implements `Transform`
- [ ] `DecoderElement<D>` wrapper implements `Transform`
- [ ] `Rav1eEncoder` implements `VideoEncoder`
- [ ] `OpenH264Encoder` implements `VideoEncoder`
- [ ] `flush()` method added to `Transform` trait
- [ ] Executor calls `flush()` at EOS
- [ ] Example 32 demonstrates encoder in pipeline
- [ ] All existing tests pass
- [ ] B-frame reordering handled correctly (PTS != DTS)

---

## Future Enhancements

1. **Async encoding:** For hardware encoders, support async encode calls
2. **Rate control feedback:** Allow downstream to signal bitrate changes
3. **Frame dropping:** QoS support to skip frames under load
4. **Codec negotiation:** Automatic codec selection based on caps
5. **Hardware acceleration:** Vulkan Video, VA-API integration

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `src/elements/codec/traits.rs` | New: codec traits, VideoFrame |
| `src/elements/codec/encoder_element.rs` | New: EncoderElement wrapper |
| `src/elements/codec/decoder_element.rs` | New: DecoderElement wrapper |
| `src/elements/codec/rav1e.rs` | Implement VideoEncoder trait |
| `src/elements/codec/openh264.rs` | Implement VideoEncoder trait |
| `src/elements/codec/mod.rs` | Export new types |
| `src/element/traits.rs` | Add flush() to Transform |
| `src/pipeline/unified_executor.rs` | Call flush() at EOS |
| `examples/32_encoder_element.rs` | New example |
