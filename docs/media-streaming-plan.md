# Media Streaming Elements Plan

This document outlines the implementation plan for pure Rust media streaming elements equivalent to GStreamer's `udpsrc`, `udpsink`, `rtspsrc`, `rtpsrc`, `rtpsink`, `videoscale`, and MPEG-TS related elements.

> **Status:** Phases 0-3 are **COMPLETE**. Ready for Phase 4 (RTSP Client).
>
> **Foundation completed items:**
> - ✅ `MediaFormat` enum with `VideoFormat`, `AudioFormat`, `RtpFormat`, `MpegTs`, `Bytes`
> - ✅ `Caps` system with format negotiation
> - ✅ `ClockTime` type (8-byte Copy, nanosecond precision, NONE sentinel)
> - ✅ `BufferFlags` as bitflags
> - ✅ `RtpMeta` struct (seq, ts, ssrc, pt, marker)
> - ✅ `Transform` trait with `Output` enum (None/Single/Multiple)
> - ✅ `Demuxer` trait for 1-to-N routing
> - ✅ `PipelineClock` for pipeline timing

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

### Phase 0: Foundation ✅ COMPLETE

**See [foundation-design.md](foundation-design.md) for details.**

All foundation components have been implemented:
- `src/format.rs` - MediaFormat, Caps, VideoFormat, AudioFormat, RtpFormat
- `src/clock.rs` - ClockTime, Clock trait, SystemClock, PipelineClock
- `src/metadata.rs` - BufferFlags (bitflags), RtpMeta, updated Metadata struct
- `src/element/traits.rs` - Output enum, Transform trait, Demuxer trait

---

### Phase 1: RTP Core ✅ COMPLETE

**Goal:** Basic RTP send/receive over UDP

```
1.1 ✅ RTP metadata types (RtpMeta already in Metadata)
1.2 ✅ RtpSrc - UDP receiver, parse with rtp crate
1.3 ✅ RtpSink - UDP sender, build packets with rtp crate
1.4 ✅ Basic RTCP sender/receiver reports
```

**Implemented in `src/elements/rtp.rs`:**
- `RtpSrc` - Receives UDP datagrams, parses RTP headers, outputs payload with RtpMeta
- `RtpSink` - Wraps buffers in RTP packets, sends over UDP
- `AsyncRtpSrc` / `AsyncRtpSink` - Async variants for tokio runtime
- Payload type and SSRC filtering
- Statistics tracking (packets received/sent, bytes, etc.)
- Clock rate conversion (RTP timestamp ↔ ClockTime)

**Implemented in `src/elements/rtcp.rs`:**
- `RtcpHandler` - Combined sender/receiver report handling
- `ReceptionStats` - Per-source reception statistics (jitter, loss, sequence tracking)
- `SenderStats` - Transmission statistics for senders
- Sender Reports (SR) for RTP senders with NTP/RTP timestamp correlation
- Receiver Reports (RR) for RTP receivers with quality metrics
- Configurable report intervals (default 5 seconds per RFC 3550)

**Key Integration Points:**
- ✅ RtpMeta struct with seq, ts, ssrc, pt, marker fields
- ✅ Uses webrtc-rs `rtp` crate for packet parsing/building
- ✅ Uses webrtc-rs `rtcp` crate for RTCP packet handling
- ✅ RtpFormat in MediaFormat for caps negotiation

### Phase 2: Codec Payloaders/Depayloaders ✅ COMPLETE

**Goal:** H.264/H.265/VP8/VP9 RTP handling

```
2.1 ✅ RtpH264Depay - wrap rtp::codecs::h264::H264Packet
2.2 ✅ RtpH264Pay - wrap rtp::codecs::h264 packetizer
2.3 ✅ RtpH265Depay/Pay - same pattern
2.4 ✅ RtpVp8Depay/Pay, RtpVp9Depay/Pay
2.5 ✅ RtpOpusDepay - Opus audio depacketizer
```

**Implemented in `src/elements/rtp_codecs.rs`:**

| Element | Description |
|---------|-------------|
| `RtpH264Depay` | H.264 depacketizer (FU-A, STAP-A), Annex B or AVC output |
| `RtpH264Pay` | H.264 packetizer with configurable MTU |
| `RtpH265Depay` | H.265/HEVC depacketizer |
| `RtpH265Pay` | H.265/HEVC packetizer |
| `RtpVp8Depay` | VP8 depacketizer with keyframe detection |
| `RtpVp8Pay` | VP8 packetizer |
| `RtpVp9Depay` | VP9 depacketizer |
| `RtpVp9Pay` | VP9 packetizer |
| `RtpOpusDepay` | Opus audio depacketizer |

**Features:**
- Thin wrappers around webrtc-rs `rtp::codecs` module
- Statistics tracking (`DepayStats`, `PayStats`)
- Keyframe detection for H.264 (IDR NAL) and VP8
- Configurable MTU for payloaders
- AVC format support for H.264

### Phase 3: Jitter Buffer ✅ COMPLETE

**Goal:** Handle packet reordering, loss, timing

```
3.1 ✅ Basic reorder buffer (sequence-based)
3.2 ✅ Configurable buffer depth (ms or packets)
3.3 ✅ Packet loss detection and signaling
3.4 ✅ Integration with RTCP for statistics
```

**Implemented in `src/elements/jitter_buffer.rs`:**

| Element | Description |
|---------|-------------|
| `RtpJitterBuffer` | Sync jitter buffer with sequence-based reordering |
| `AsyncJitterBuffer` | Async variant with timeout-based packet retrieval |
| `JitterBufferConfig` | Configuration (latency_ms, max_packets, clock_rate, drop_late) |
| `JitterBufferStats` | Statistics (received, output, dropped, lost, reordered, duplicate) |
| `LossInfo` | Loss statistics for RTCP reporting integration |

**Features:**
- Extended sequence number handling (16-bit wraparound)
- Configurable buffering latency (default 200ms)
- Maximum buffer size enforcement with overflow handling
- Late packet detection and optional dropping
- Duplicate packet detection
- Packet loss detection with gap tracking
- RTCP integration via `LossInfo` struct (expected, received, lost counts)
- Flush operation for draining remaining packets
- Reset operation for stream discontinuities

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

## RTP Metadata ✅ IMPLEMENTED

RTP metadata is already integrated in `src/metadata.rs`:

```rust
/// RTP header metadata (12 bytes, Copy)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtpMeta {
    pub seq: u16,       // RTP sequence number
    pub ts: u32,        // RTP timestamp
    pub ssrc: u32,      // Synchronization source
    pub pt: u8,         // Payload type
    pub marker: bool,   // Marker bit
}

pub struct Metadata {
    pub pts: ClockTime,
    pub dts: ClockTime,
    pub duration: ClockTime,
    pub sequence: u64,
    pub stream_id: u32,
    pub flags: BufferFlags,
    pub rtp: Option<RtpMeta>,  // RTP-specific metadata
    pub format: Option<MediaFormat>,
    pub offset: Option<u64>,
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
