# Media Streaming Elements Plan

This document outlines the implementation plan for pure Rust media streaming elements equivalent to GStreamer's `udpsrc`, `udpsink`, `rtspsrc`, `rtpsrc`, `rtpsink`, `videoscale`, and MPEG-TS related elements.

> **Prerequisites:** This plan depends on foundation improvements described in [foundation-design.md](foundation-design.md). Phase 0 (Caps, Multi-Output, Clock, Timestamps) must be completed first.

## Primary Dependency: webrtc-rs

The [webrtc-rs/webrtc](https://github.com/webrtc-rs/webrtc) monorepo provides a comprehensive, actively maintained pure Rust implementation covering most of our needs.

**Repository Stats:**
- 3,496 commits, 24 releases (latest v0.14.0, Sept 2025)
- 2,700+ dependent projects
- Dual licensed MIT/Apache-2.0

### webrtc-rs Crates We'll Use

| Crate | Purpose | Version |
|-------|---------|---------|
| `rtp` | RTP packet parsing/building | 0.14 |
| `rtcp` | RTCP sender/receiver reports | 0.12 |
| `webrtc-media` | Media sample handling | 0.11 |
| `sdp` | Session description parsing | - |
| `webrtc-util` | Utilities (buffers, etc.) | - |
| `interceptor` | RTP/RTCP packet processing | - |

### Supported Codecs (rtp crate)

The `rtp` crate includes packetizers and depacketizers for:

| Codec | Type | RFC |
|-------|------|-----|
| **H.264** | Video | RFC 6184 |
| **H.265** | Video | RFC 7798 |
| **VP8** | Video | RFC 7741 |
| **VP9** | Video | - |
| **AV1** | Video | - |
| **Opus** | Audio | RFC 7587 |
| **G.7xx** | Audio | RFC 3551 |

---

## Additional Pure Rust Crates

| Functionality | Crate | Status | Notes |
|---------------|-------|--------|-------|
| **RTSP Client** | [retina](https://github.com/scottlamb/retina) | Production | H.264/H.265, Moonfire NVR |
| **MPEG-TS Demux** | [mpeg2ts-reader](https://github.com/dholroyd/mpeg2ts-reader) | Stable | Zero-copy, trait-based |
| **MPEG-TS Mux** | Custom | TBD | Based on va-ts or from scratch |
| **Video Scale** | [fast_image_resize](https://github.com/Cykooz/fast_image_resize) | Active | SIMD (AVX2/SSE4.1/NEON) |

---

## Implementation Tiers

### Tier A: RTP/RTCP Foundation

Using `webrtc-rs` crates directly.

| Element | GStreamer Equiv | Complexity | Source |
|---------|-----------------|------------|--------|
| `RtpSrc` | `rtpsrc` | M | `rtp` crate + UDP |
| `RtpSink` | `rtpsink` | M | `rtp` crate + UDP |
| `RtcpSrc` | rtpbin component | M | `rtcp` crate |
| `RtcpSink` | rtpbin component | M | `rtcp` crate |
| `RtpJitterBuffer` | `rtpjitterbuffer` | H | `interceptor` or custom |

### Tier B: RTP Payloaders/Depayloaders

All available in `rtp::codecs` module.

| Element | GStreamer Equiv | Complexity | Module |
|---------|-----------------|------------|--------|
| `RtpH264Depay` | `rtph264depay` | L | `rtp::codecs::h264` |
| `RtpH264Pay` | `rtph264pay` | L | `rtp::codecs::h264` |
| `RtpH265Depay` | `rtph265depay` | L | `rtp::codecs::h265` |
| `RtpH265Pay` | `rtph265pay` | L | `rtp::codecs::h265` |
| `RtpVp8Depay` | `rtpvp8depay` | L | `rtp::codecs::vp8` |
| `RtpVp8Pay` | `rtpvp8pay` | L | `rtp::codecs::vp8` |
| `RtpVp9Depay` | `rtpvp9depay` | L | `rtp::codecs::vp9` |
| `RtpVp9Pay` | `rtpvp9pay` | L | `rtp::codecs::vp9` |
| `RtpOpusDepay` | `rtpopusdepay` | L | `rtp::codecs::opus` |
| `RtpOpusPay` | `rtpopuspay` | L | `rtp::codecs::opus` |
| `RtpAv1Depay` | `rtpav1depay` | L | `rtp::codecs::av1` |
| `RtpAv1Pay` | `rtpav1pay` | L | `rtp::codecs::av1` |

### Tier C: RTSP Elements

| Element | GStreamer Equiv | Complexity | Source |
|---------|-----------------|------------|--------|
| `RtspSrc` | `rtspsrc` | M | `retina` crate |

### Tier D: MPEG-TS Elements

| Element | GStreamer Equiv | Complexity | Source |
|---------|-----------------|------------|--------|
| `TsDemux` | `tsdemux` | M | `mpeg2ts-reader` |
| `TsMux` | `mpegtsmux` | H | Custom implementation |
| `TsParse` | `tsparse` | L | `mpeg2ts-reader` |

### Tier E: Video Processing

| Element | GStreamer Equiv | Complexity | Source |
|---------|-----------------|------------|--------|
| `VideoScale` | `videoscale` | M | `fast_image_resize` |
| `VideoConvert` | `videoconvert` | M | Custom (YUV/RGB) |

---

## Cargo Dependencies

```toml
[dependencies]
# webrtc-rs monorepo crates
rtp = { version = "0.14", optional = true }
rtcp = { version = "0.12", optional = true }
webrtc-media = { version = "0.11", optional = true }
sdp = { version = "0.6", optional = true }
webrtc-util = { version = "0.10", optional = true }

# RTSP client
retina = { version = "0.4", optional = true }

# MPEG-TS
mpeg2ts-reader = { version = "0.18", optional = true }

# Video processing
fast_image_resize = { version = "5", optional = true }

[features]
default = []
rtp = ["dep:rtp", "dep:rtcp", "dep:webrtc-util"]
media = ["rtp", "dep:webrtc-media"]
rtsp = ["media", "dep:retina"]
mpeg-ts = ["dep:mpeg2ts-reader"]
video-scale = ["dep:fast_image_resize"]
streaming = ["rtp", "media", "rtsp", "mpeg-ts", "video-scale"]
```

---

## Implementation Phases

### Phase 0: Foundation (5-6 weeks)

**See [foundation-design.md](foundation-design.md) for details.**

| Sub-Phase | Component | Duration |
|-----------|-----------|----------|
| 0.1 | Caps System | 1 week |
| 0.2 | Multi-Output Support | 1 week |
| 0.3 | Pipeline Clock | 1 week |
| 0.4 | Timestamp & Synchronization | 1 week |
| 0.5 | Caps Negotiation Integration | 1 week |
| 0.6 | Allocation Query | 3-4 days |
| 0.7 | AsyncElement | 3-4 days |

---

### Phase 1: RTP Core (1-2 weeks)

**Goal:** Basic RTP send/receive over UDP

```
1.1 RTP metadata types (integrate with Parallax Buffer metadata)
1.2 RtpSrc - UDP receiver, parse with rtp crate
1.3 RtpSink - UDP sender, build packets with rtp crate
1.4 Basic RTCP sender/receiver reports
```

**Key Integration Points:**
- Map RTP header fields to `Metadata` struct
- Use existing `UdpSrc`/`UdpSink` as transport layer
- Add `rtp_timestamp`, `rtp_sequence`, `rtp_ssrc`, `rtp_marker` to metadata

### Phase 2: Codec Payloaders/Depayloaders (1 week)

**Goal:** H.264/H.265/VP8/VP9 RTP handling

```
2.1 RtpH264Depay - wrap rtp::codecs::h264::H264Packet
2.2 RtpH264Pay - wrap rtp::codecs::h264 packetizer
2.3 RtpH265Depay/Pay - same pattern
2.4 RtpVp8Depay/Pay, RtpVp9Depay/Pay
```

**Design:** Thin wrappers around webrtc-rs codec modules:

```rust
pub struct RtpH264Depay {
    depacketizer: h264::H264Packet,
    // ...
}

impl Element for RtpH264Depay {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Extract RTP payload from buffer
        // Call depacketizer.depacketize()
        // Return NAL units as new buffer
    }
}
```

### Phase 3: Jitter Buffer (1-2 weeks)

**Goal:** Handle packet reordering, loss, timing

```
3.1 Basic reorder buffer (sequence-based)
3.2 Configurable buffer depth (ms or packets)
3.3 Packet loss detection and signaling
3.4 Integration with RTCP for statistics
```

**Options:**
- Use `interceptor` crate's NACK/jitter buffer
- Build custom optimized for Parallax buffer model

### Phase 4: RTSP Client (2 weeks)

**Goal:** Connect to RTSP cameras/servers

```
4.1 RtspSrc wrapper around retina
4.2 TCP interleaved transport (primary)
4.3 UDP transport (secondary)
4.4 Authentication (Basic/Digest)
4.5 Multi-stream handling (video + audio)
```

**Design:**

```rust
pub struct RtspSrc {
    session: retina::client::Session,
    // ...
}

impl AsyncSource for RtspSrc {
    async fn produce(&mut self) -> Result<Option<Buffer>> {
        // Receive frame from retina
        // Convert to Parallax Buffer with metadata
    }
}
```

### Phase 5: MPEG-TS (2 weeks)

**Goal:** Demux/mux MPEG-TS streams

```
5.1 TsDemux using mpeg2ts-reader traits
5.2 Elementary stream extraction (PES)
5.3 PAT/PMT parsing
5.4 TsMux (basic implementation)
```

**Design:**

```rust
pub struct TsDemux {
    demuxer: mpeg2ts_reader::demultiplex::Demultiplex<...>,
    // Output queues per PID
}

impl Element for TsDemux {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Feed TS packets to demuxer
        // Emit PES payloads as buffers with stream_id metadata
    }
}
```

### Phase 6: Video Scaling (1 week)

**Goal:** Resize video frames with SIMD

```
6.1 VideoScale element wrapping fast_image_resize
6.2 Algorithm selection (Nearest, Bilinear, Lanczos3)
6.3 Aspect ratio preservation
6.4 Pixel format handling (RGB, RGBA)
```

---

## Element API Examples

### RtpSrc

```rust
use parallax::elements::RtpSrc;

let rtp_src = RtpSrc::bind("0.0.0.0:5004")?
    .with_payload_type(96)
    .with_clock_rate(90000);

// Buffer metadata includes:
// - rtp.sequence: u16
// - rtp.timestamp: u32
// - rtp.ssrc: u32
// - rtp.marker: bool
// - rtp.payload_type: u8
```

### RtpH264Depay

```rust
use parallax::elements::{RtpSrc, RtpH264Depay};

// Pipeline: RtpSrc -> RtpH264Depay -> FileSink
let src = RtpSrc::bind("0.0.0.0:5004")?;
let depay = RtpH264Depay::new();

// Outputs complete H.264 NAL units (or access units)
// Handles FU-A fragmentation, STAP-A aggregation
```

### RtspSrc

```rust
use parallax::elements::RtspSrc;

let rtsp_src = RtspSrc::new("rtsp://192.168.1.100/stream1")
    .with_transport(Transport::TcpInterleaved)
    .with_auth("admin", "password")?;

// Automatically:
// - Connects and negotiates
// - Handles RTP depacketization
// - Outputs H.264/H.265 access units
```

### TsDemux

```rust
use parallax::elements::TsDemux;

let demux = TsDemux::new()
    .select_program(1);

// Input: MPEG-TS packets (188 bytes each)
// Output: PES payloads with stream metadata
//   - metadata.stream_id (PID)
//   - metadata.pts / metadata.dts
//   - metadata.stream_type (H264, AAC, etc.)
```

### VideoScale

```rust
use parallax::elements::VideoScale;

let scaler = VideoScale::new()
    .with_output_size(1280, 720)
    .with_algorithm(Algorithm::Lanczos3)
    .preserve_aspect_ratio(true);

// Input: RGB/RGBA frame buffer
// Output: Scaled frame buffer
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Parallax Media Pipeline                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   SOURCES                    PROCESSING                    SINKS         │
│   ───────                    ──────────                    ─────         │
│                                                                          │
│   ┌─────────┐              ┌──────────────┐              ┌─────────┐    │
│   │ RtspSrc │─────────────▶│ RtpH264Depay │─────────────▶│ FileSink│    │
│   └─────────┘              └──────────────┘              └─────────┘    │
│       │                           │                                      │
│       │ (retina)                  │ (rtp::codecs::h264)                 │
│                                                                          │
│   ┌─────────┐   ┌────────────┐   ┌──────────────┐       ┌─────────┐    │
│   │  RtpSrc │──▶│JitterBuffer│──▶│ RtpH265Depay │──────▶│ RtpSink │    │
│   └─────────┘   └────────────┘   └──────────────┘       └─────────┘    │
│       │              │                  │                    │          │
│       │ (UDP)        │ (interceptor)    │ (rtp::codecs)     │ (UDP)    │
│                                                                          │
│   ┌─────────┐              ┌──────────────┐              ┌─────────┐    │
│   │ TsDemux │─────────────▶│  VideoScale  │─────────────▶│  TsMux  │    │
│   └─────────┘              └──────────────┘              └─────────┘    │
│       │                           │                           │          │
│       │ (mpeg2ts-reader)          │ (fast_image_resize)      │ (custom) │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │      webrtc-rs ecosystem      │
                    │  ┌─────┐ ┌──────┐ ┌───────┐  │
                    │  │ rtp │ │ rtcp │ │ media │  │
                    │  └─────┘ └──────┘ └───────┘  │
                    └──────────────────────────────┘
```

---

## RTP Metadata Extension

Extend `Metadata` struct for RTP-specific fields:

```rust
// In src/metadata.rs
pub struct Metadata {
    // Existing fields...
    pub sequence: u64,
    pub timestamp: Option<Timestamp>,
    
    // New RTP fields
    pub rtp: Option<RtpMetadata>,
}

#[derive(Clone, Debug, Default)]
pub struct RtpMetadata {
    pub sequence: u16,
    pub timestamp: u32,
    pub ssrc: u32,
    pub payload_type: u8,
    pub marker: bool,
    pub csrc: Vec<u32>,
}
```

---

## Success Criteria

### MVP (Minimum Viable Product)
- [ ] RtpSrc/RtpSink over UDP
- [ ] RtpH264Depay working
- [ ] RtspSrc connecting to cameras
- [ ] TsDemux extracting streams
- [ ] VideoScale with Lanczos3

### Production Ready
- [ ] All Tier A-E elements implemented
- [ ] Jitter buffer with NACK support
- [ ] RTCP statistics reporting
- [ ] Full H.264/H.265/VP8/VP9 support
- [ ] Comprehensive tests
- [ ] Performance benchmarks

---

## References

- [webrtc-rs GitHub](https://github.com/webrtc-rs/webrtc)
- [rtp crate docs](https://docs.rs/rtp)
- [retina RTSP library](https://github.com/scottlamb/retina)
- [mpeg2ts-reader](https://github.com/dholroyd/mpeg2ts-reader)
- [fast_image_resize](https://github.com/Cykooz/fast_image_resize)
- [RFC 3550 - RTP](https://datatracker.ietf.org/doc/html/rfc3550)
- [RFC 6184 - RTP H.264](https://datatracker.ietf.org/doc/html/rfc6184)
- [RFC 7798 - RTP H.265](https://datatracker.ietf.org/doc/html/rfc7798)
