# Plan 12: Additional Codec Support

**Priority:** Medium  
**Effort:** Medium (2-3 weeks)  
**Dependencies:** Plan 02 (Codec Element Wrappers) - Complete  

---

## Problem Statement

Parallax has limited codec support compared to GStreamer:

| Codec | Parallax | GStreamer |
|-------|----------|-----------|
| H.264 decode | OpenH264 (C++) | Multiple (native, VA-API, NVDEC) |
| H.264 encode | OpenH264 (C++) | Multiple |
| H.265 decode | None | Multiple |
| H.265 encode | None | Multiple |
| AV1 decode | dav1d (C) | dav1d, Vulkan |
| AV1 encode | rav1e (Rust) | rav1e, SVT-AV1 |
| VP9 | None | Multiple |
| Opus | None | libopus |
| AAC encode | None | Multiple |

The analysis report identified **Opus** and **pure-Rust H.264 decoder** as high priorities.

---

## Goals

1. Add Opus audio codec support (encode + decode)
2. Explore pure-Rust H.264 decoder options
3. Add VP9 decode support (via dav1d or pure Rust)
4. Add AAC encoder support
5. Maintain pure-Rust preference where quality permits

---

## Codec Priority List

| Codec | Priority | Reason | Approach |
|-------|----------|--------|----------|
| **Opus encode/decode** | High | Voice/music streaming | `opus` or `audiopus` crate |
| **H.264 pure Rust decode** | High | Ubiquitous format | Evaluate `openh264-rs2` or similar |
| **AAC encode** | Medium | Streaming compatibility | `fdk-aac-sys` or pure Rust |
| **VP9 decode** | Low | YouTube legacy | dav1d supports VP9 |
| **FLAC encode** | Low | Archival audio | `flac-sys` or pure Rust |

---

## Implementation

### Opus Audio Codec

**Why Opus?**
- Best quality/bitrate for voice and music
- WebRTC standard
- Low latency (2.5ms to 60ms)
- Royalty-free

**Options:**

| Crate | Type | Notes |
|-------|------|-------|
| `opus` | Bindings to libopus | Well-tested, requires C library |
| `audiopus` | Bindings to libopus | Higher-level API |
| `opus-rs` | Pure Rust (WIP) | Not production-ready |

**Decision:** Use `audiopus` for now (mature), watch for pure Rust alternatives.

```rust
pub struct OpusEncoder {
    encoder: audiopus::coder::Encoder,
    sample_rate: u32,
    channels: u32,
    bitrate: u32,
}

impl OpusEncoder {
    pub fn new(sample_rate: u32, channels: u32, bitrate: u32) -> Result<Self> {
        let encoder = audiopus::coder::Encoder::new(
            audiopus::SampleRate::try_from(sample_rate as i32)?,
            if channels == 1 { Channels::Mono } else { Channels::Stereo },
            audiopus::Application::Audio,
        )?;
        encoder.set_bitrate(audiopus::Bitrate::BitsPerSecond(bitrate as i32))?;
        Ok(Self { encoder, sample_rate, channels, bitrate })
    }
    
    pub fn encode(&mut self, pcm: &[i16], output: &mut [u8]) -> Result<usize> {
        Ok(self.encoder.encode(pcm, output)?)
    }
}

pub struct OpusDecoder {
    decoder: audiopus::coder::Decoder,
    sample_rate: u32,
    channels: u32,
}

impl OpusDecoder {
    pub fn decode(&mut self, packet: &[u8], output: &mut [i16]) -> Result<usize> {
        Ok(self.decoder.decode(Some(packet), output, false)?)
    }
}
```

### H.264 Pure Rust Decoder

**Current State:**
- No mature pure-Rust H.264 decoder exists
- `openh264` (Cisco's implementation) requires C++ compiler
- Security concern: C/C++ code in codec hot path

**Options:**

| Option | Status | Notes |
|--------|--------|-------|
| Write from scratch | 6+ months | Complex, patents (until 2027) |
| Port OpenH264 to Rust | 3+ months | Massive effort |
| Use `openh264-rs` | Now | C++ dependency |
| Wait for community | Unknown | May never happen |
| Vulkan Video only | Now | GPU required |

**Decision:** Continue using `openh264` for CPU decode, invest in Vulkan Video (Plan 11) for the future. Document limitation clearly.

### AAC Encoder

**Options:**

| Crate | Type | License |
|-------|------|---------|
| `fdk-aac-sys` | Bindings to FDK-AAC | Non-free for some uses |
| `fdkaac` | Bindings | Similar |
| Pure Rust | None exists | - |

**Decision:** Add `fdk-aac` as optional feature with clear license documentation.

```rust
#[cfg(feature = "aac-encode")]
pub struct AacEncoder {
    encoder: fdk_aac::enc::Encoder,
    sample_rate: u32,
    channels: u32,
    bitrate: u32,
}

#[cfg(feature = "aac-encode")]
impl AacEncoder {
    pub fn new(sample_rate: u32, channels: u32, bitrate: u32) -> Result<Self> {
        let encoder = fdk_aac::enc::Encoder::new(
            fdk_aac::enc::EncoderParams {
                sample_rate,
                channels,
                bit_rate: fdk_aac::enc::BitRate::Cbr(bitrate),
                ..Default::default()
            }
        )?;
        Ok(Self { encoder, sample_rate, channels, bitrate })
    }
}
```

### VP9 Decoder

**Good news:** `dav1d` already supports VP9 in addition to AV1.

```rust
// dav1d can decode both AV1 and VP9
pub struct Dav1dDecoder {
    decoder: dav1d::Decoder,
    codec: Codec, // Av1 or Vp9
}

impl Dav1dDecoder {
    pub fn new(codec: Codec) -> Result<Self> {
        let settings = dav1d::Settings::new();
        let decoder = dav1d::Decoder::with_settings(&settings)?;
        Ok(Self { decoder, codec })
    }
}
```

---

## Implementation Steps

### Phase 1: Opus Support (3-5 days) - COMPLETE

- [x] Add `opus` dependency (feature-gated) - used `opus` crate v0.3.1 (not audiopus)
- [x] Create `src/elements/codec/opus.rs`
- [x] Implement `OpusEncoder` and `OpusDecoder`
- [x] Implement `AudioEncoder` and `AudioDecoder` traits (in `audio_traits.rs`)
- [x] Create `AudioEncoderElement` and `AudioDecoderElement` generic wrappers
- [x] Add unit tests with sine wave audio samples
- [x] Create example: `20_opus_audio.rs`

### Phase 2: AAC Encoder (2-3 days) - COMPLETE

- [x] Add `fdk-aac` dependency (feature-gated)
- [x] Document license implications in CLAUDE.md and mod.rs
- [x] Create `src/elements/codec/aac.rs`
- [x] Implement `AacEncoder`
- [x] Use generic `AudioEncoderElement` wrapper (no AAC-specific element needed)
- [x] Add unit tests
- [x] Update documentation

### Phase 3: VP9 Support (1-2 days) - NOT IMPLEMENTED

**Research finding:** dav1d does NOT support VP9 - it is AV1-only.

- [x] Verify dav1d VP9 support - **NEGATIVE**: dav1d is AV1-only
- [ ] VP9 would require `vpx` crate (libvpx bindings) - deferred
- [ ] No pure-Rust VP9 decoder exists

**Recommendation:** Use `vpx` crate for VP9 if needed in future. Low priority given VP9 is legacy format.

### Phase 4: H.264 Investigation (Ongoing) - DOCUMENTED

- [x] Document current OpenH264 limitations (in CLAUDE.md)
- [ ] Track pure-Rust H.264 decoder projects - none production-ready
- [ ] Link to Plan 11 (Vulkan Video) as recommended path for GPU decode

### Phase 5: Integration (1-2 days) - COMPLETE

- [x] Update CLAUDE.md with new codecs
- [x] Update elements/mod.rs with new exports
- [x] Update Cargo.toml with new feature flags
- [x] Audio codec traits work with caps negotiation via element wrappers

---

## Feature Flags (Implemented)

```toml
[features]
# Audio codecs - encoders (require system libraries)
opus = ["dep:opus"]           # Opus encoder/decoder (requires libopus)
aac-encode = ["dep:fdk-aac"]  # AAC encoder (FDK-AAC, license restrictions!)

# Existing audio decoders (pure Rust via Symphonia)
audio-codecs = ["audio-flac", "audio-mp3", "audio-aac", "audio-vorbis"]
```

---

## Dependencies

| Crate | Version | License | Feature |
|-------|---------|---------|---------|
| `audiopus` | 0.3 | MIT | `opus` |
| `fdk-aac` | 0.6 | Non-free* | `aac-encode` |

*FDK-AAC has patent license restrictions for commercial use. Document clearly.

---

## Audio Codec Traits

```rust
/// Audio encoder trait (analogous to VideoEncoder)
pub trait AudioEncoder: Send {
    type Packet: AsRef<[u8]> + Send;
    
    /// Encode PCM samples to compressed packet
    fn encode(&mut self, samples: &AudioSamples) -> Result<Vec<Self::Packet>>;
    
    /// Flush any buffered packets
    fn flush(&mut self) -> Result<Vec<Self::Packet>>;
    
    /// Get output format info
    fn output_format(&self) -> AudioFormat;
}

/// Audio decoder trait
pub trait AudioDecoder: Send {
    /// Decode compressed packet to PCM samples
    fn decode(&mut self, packet: &[u8]) -> Result<AudioSamples>;
    
    /// Flush any buffered samples
    fn flush(&mut self) -> Result<AudioSamples>;
    
    /// Get output format info
    fn output_format(&self) -> AudioFormat;
}

/// Audio samples container
pub struct AudioSamples {
    pub data: Vec<u8>,
    pub format: SampleFormat,
    pub channels: u32,
    pub sample_rate: u32,
    pub num_samples: usize,
}
```

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_opus_encode_decode_roundtrip() {
    let encoder = OpusEncoder::new(48000, 2, 128000).unwrap();
    let decoder = OpusDecoder::new(48000, 2).unwrap();
    
    // Generate sine wave
    let samples: Vec<i16> = (0..960)
        .map(|i| (((i as f32 * 440.0 * 2.0 * PI) / 48000.0).sin() * 32000.0) as i16)
        .collect();
    
    let mut encoded = [0u8; 1000];
    let len = encoder.encode(&samples, &mut encoded).unwrap();
    
    let mut decoded = [0i16; 960];
    let decoded_len = decoder.decode(&encoded[..len], &mut decoded).unwrap();
    
    assert_eq!(decoded_len, 960);
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_opus_pipeline() {
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node("src", AudioTestSrc::new(48000, 2));
    let enc = pipeline.add_node("enc", OpusEncElement::new(128000)?);
    let dec = pipeline.add_node("dec", OpusDecElement::new()?);
    let sink = pipeline.add_node("sink", AudioNullSink::new());
    
    pipeline.link_all(&[src, enc, dec, sink])?;
    pipeline.run().await?;
}
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `src/elements/codec/opus.rs` | Create |
| `src/elements/codec/aac_enc.rs` | Create |
| `src/elements/codec/audio.rs` | Create (traits) |
| `src/elements/codec/mod.rs` | Modify (add modules) |
| `Cargo.toml` | Add dependencies |
| `examples/20_opus_audio.rs` | Create |
| `examples/21_vp9_decode.rs` | Create |
| `CLAUDE.md` | Update codec table |
| `README.md` | Update codec table |

---

## Success Criteria

- [ ] Opus encode/decode works in pipeline
- [ ] AAC encode works (with feature flag)
- [ ] VP9 decode works via dav1d
- [ ] All codecs have pipeline element wrappers
- [ ] Caps negotiation handles new formats
- [ ] Examples demonstrate each codec
- [ ] License implications documented

---

## Future Considerations

| Codec | Status | Notes |
|-------|--------|-------|
| H.265 CPU | No pure Rust | Wait for community or use Vulkan Video |
| VP8 | Low priority | Legacy format |
| Vorbis encode | Low priority | Opus preferred |
| FLAC encode | Low priority | `flac` crate available |

---

*Created January 2026 based on [Project Analysis Report](../docs/PROJECT_ANALYSIS_REPORT.md)*
