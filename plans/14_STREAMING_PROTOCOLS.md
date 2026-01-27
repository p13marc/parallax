# Plan 14: Streaming Protocol Elements (HLS/DASH)

**Priority:** Medium  
**Effort:** Medium (2-3 weeks)  
**Dependencies:** Plan 03 (Muxer Synchronization) - Complete  

---

## Problem Statement

Parallax can stream via TCP, UDP, RTP, and WebSocket, but lacks support for **adaptive bitrate streaming** protocols used by major streaming services:
- **HLS (HTTP Live Streaming)** - Apple's protocol, dominant in OTT
- **DASH (Dynamic Adaptive Streaming over HTTP)** - MPEG standard
- **RTMP** - Still used for live ingest (Twitch, YouTube Live)

These are essential for building production streaming applications.

---

## Goals

1. Implement HLS output (`hlssink`)
2. Implement DASH output (`dashsink`)
3. Implement RTMP ingest output (`rtmpsink`)
4. Support adaptive bitrate (multiple renditions)
5. Integrate with existing muxers (TS for HLS, fMP4 for DASH)

---

## HLS Output

### HLS Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        HLS Structure                             │
├─────────────────────────────────────────────────────────────────┤
│  master.m3u8 (Master Playlist)                                  │
│    ├── 1080p/playlist.m3u8 → segment_001.ts, segment_002.ts...  │
│    ├── 720p/playlist.m3u8  → segment_001.ts, segment_002.ts...  │
│    └── 480p/playlist.m3u8  → segment_001.ts, segment_002.ts...  │
└─────────────────────────────────────────────────────────────────┘
```

### Design

```rust
/// HLS output sink
pub struct HlsSink {
    config: HlsConfig,
    segment_writer: SegmentWriter,
    playlist_writer: PlaylistWriter,
    current_segment: Option<SegmentBuffer>,
    segment_index: u64,
}

pub struct HlsConfig {
    /// Output directory for segments and playlists
    pub output_dir: PathBuf,
    /// Segment duration in seconds
    pub segment_duration: f64,
    /// Number of segments to keep in playlist
    pub playlist_length: u32,
    /// Enable low-latency HLS (LL-HLS)
    pub low_latency: bool,
    /// Multiple renditions for ABR
    pub variants: Vec<HlsVariant>,
}

pub struct HlsVariant {
    pub name: String,
    pub bandwidth: u32,
    pub width: u32,
    pub height: u32,
}

impl HlsSink {
    pub fn new(config: HlsConfig) -> Result<Self>;
    
    /// Generate master playlist
    fn write_master_playlist(&self) -> Result<()>;
    
    /// Generate media playlist for variant
    fn write_media_playlist(&self, variant: &HlsVariant) -> Result<()>;
    
    /// Finalize current segment and start new one
    fn rotate_segment(&mut self) -> Result<()>;
}

impl Sink for HlsSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        // Buffer must be TS-muxed data
        self.current_segment.as_mut().unwrap().write(ctx.input())?;
        
        // Check if segment duration exceeded
        if self.current_segment_duration() >= self.config.segment_duration {
            self.rotate_segment()?;
        }
        
        Ok(())
    }
}
```

### M3U8 Playlist Generation

```rust
/// Generate HLS media playlist
fn generate_media_playlist(segments: &[SegmentInfo], config: &HlsConfig) -> String {
    let mut playlist = String::new();
    
    playlist.push_str("#EXTM3U\n");
    playlist.push_str("#EXT-X-VERSION:3\n");
    playlist.push_str(&format!("#EXT-X-TARGETDURATION:{}\n", config.segment_duration.ceil() as u32));
    playlist.push_str(&format!("#EXT-X-MEDIA-SEQUENCE:{}\n", segments[0].sequence));
    
    for segment in segments {
        playlist.push_str(&format!("#EXTINF:{:.3},\n", segment.duration));
        playlist.push_str(&format!("{}\n", segment.filename));
    }
    
    // For live streams, don't include EXT-X-ENDLIST
    if config.is_vod {
        playlist.push_str("#EXT-X-ENDLIST\n");
    }
    
    playlist
}

/// Generate master playlist for ABR
fn generate_master_playlist(variants: &[HlsVariant]) -> String {
    let mut playlist = String::new();
    
    playlist.push_str("#EXTM3U\n");
    
    for variant in variants {
        playlist.push_str(&format!(
            "#EXT-X-STREAM-INF:BANDWIDTH={},RESOLUTION={}x{}\n",
            variant.bandwidth, variant.width, variant.height
        ));
        playlist.push_str(&format!("{}/playlist.m3u8\n", variant.name));
    }
    
    playlist
}
```

---

## DASH Output

### DASH Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                       DASH Structure                             │
├─────────────────────────────────────────────────────────────────┤
│  manifest.mpd (Media Presentation Description)                   │
│    ├── AdaptationSet (Video)                                    │
│    │   ├── Representation 1080p → init.mp4, seg_1.m4s, seg_2... │
│    │   ├── Representation 720p  → init.mp4, seg_1.m4s, seg_2... │
│    │   └── Representation 480p  → init.mp4, seg_1.m4s, seg_2... │
│    └── AdaptationSet (Audio)                                    │
│        └── Representation → init.mp4, seg_1.m4s, seg_2.m4s...   │
└─────────────────────────────────────────────────────────────────┘
```

### Design

```rust
/// DASH output sink
pub struct DashSink {
    config: DashConfig,
    mpd_writer: MpdWriter,
    segment_writers: HashMap<String, SegmentWriter>,
}

pub struct DashConfig {
    /// Output directory
    pub output_dir: PathBuf,
    /// Segment duration in seconds
    pub segment_duration: f64,
    /// Minimum buffer time
    pub min_buffer_time: f64,
    /// Presentation type
    pub presentation_type: DashPresentationType,
    /// Adaptation sets
    pub adaptation_sets: Vec<DashAdaptationSet>,
}

pub enum DashPresentationType {
    Static,  // VOD
    Dynamic, // Live
}

pub struct DashAdaptationSet {
    pub content_type: ContentType,
    pub representations: Vec<DashRepresentation>,
}

pub struct DashRepresentation {
    pub id: String,
    pub bandwidth: u32,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub codecs: String,
}

impl DashSink {
    /// Generate MPD manifest
    fn write_mpd(&self) -> Result<()>;
    
    /// Generate initialization segment (fMP4 header)
    fn write_init_segment(&self, repr: &DashRepresentation) -> Result<()>;
    
    /// Generate media segment
    fn write_media_segment(&mut self, repr: &str, data: &[u8]) -> Result<()>;
}
```

### MPD Generation

DASH uses XML manifests. Use `quick-xml` for generation:

```rust
fn generate_mpd(config: &DashConfig, segments: &DashSegments) -> String {
    let mut xml = String::new();
    
    xml.push_str(r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    xml.push_str(r#"<MPD xmlns="urn:mpeg:dash:schema:mpd:2011" "#);
    xml.push_str(&format!(r#"type="{}" "#, 
        if config.presentation_type == DashPresentationType::Dynamic { "dynamic" } else { "static" }
    ));
    xml.push_str(&format!(r#"minBufferTime="PT{}S">"#, config.min_buffer_time));
    
    xml.push_str("<Period>");
    
    for adaptation_set in &config.adaptation_sets {
        xml.push_str(&format!(r#"<AdaptationSet contentType="{}">"#, adaptation_set.content_type));
        
        for repr in &adaptation_set.representations {
            xml.push_str(&format!(
                r#"<Representation id="{}" bandwidth="{}" codecs="{}">"#,
                repr.id, repr.bandwidth, repr.codecs
            ));
            xml.push_str(&format!(
                r#"<SegmentTemplate media="{}/$Number$.m4s" initialization="{}/init.mp4" />"#,
                repr.id, repr.id
            ));
            xml.push_str("</Representation>");
        }
        
        xml.push_str("</AdaptationSet>");
    }
    
    xml.push_str("</Period>");
    xml.push_str("</MPD>");
    
    xml
}
```

---

## RTMP Output

### Design

```rust
/// RTMP output sink (for streaming to Twitch, YouTube, etc.)
pub struct RtmpSink {
    connection: RtmpConnection,
    stream_key: String,
    audio_track: Option<AudioTrack>,
    video_track: Option<VideoTrack>,
}

pub struct RtmpConfig {
    /// RTMP URL (e.g., rtmp://live.twitch.tv/app)
    pub url: String,
    /// Stream key
    pub stream_key: String,
}

impl RtmpSink {
    pub fn new(config: RtmpConfig) -> Result<Self>;
    
    /// Send FLV packet
    fn send_flv_packet(&mut self, packet: &FlvPacket) -> Result<()>;
}

impl Sink for RtmpSink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
        // Input must be FLV-muxed or raw H.264/AAC
        // Convert to RTMP messages and send
        let message = self.encode_rtmp_message(ctx.input())?;
        self.connection.send(message).await?;
        Ok(())
    }
}
```

**Note:** RTMP implementation is complex. Consider using existing crate or FFmpeg for initial implementation.

---

## Implementation Steps

### Phase 1: HLS Output (1 week)

- [ ] Create `src/elements/streaming/mod.rs`
- [ ] Implement `src/elements/streaming/hls.rs`:
  - [ ] `HlsSink` struct and configuration
  - [ ] M3U8 playlist generation
  - [ ] Segment file writing
  - [ ] Segment rotation based on duration
- [ ] Integrate with `TsMux` for segment creation
- [ ] Add unit tests
- [ ] Create example: `25_hls_output.rs`

### Phase 2: DASH Output (1 week)

- [ ] Add `quick-xml` dependency for MPD generation
- [ ] Implement `src/elements/streaming/dash.rs`:
  - [ ] `DashSink` struct and configuration
  - [ ] MPD manifest generation
  - [ ] fMP4 segment writing
  - [ ] Init segment handling
- [ ] Integrate with `Mp4Mux` for fMP4 segments
- [ ] Add unit tests
- [ ] Create example: `26_dash_output.rs`

### Phase 3: Adaptive Bitrate Pipeline (3-5 days)

- [ ] Create `AbrPipeline` helper for multi-rendition output
- [ ] Implement parallel encoding to multiple bitrates
- [ ] Synchronize segments across renditions
- [ ] Create example: `27_abr_pipeline.rs`

### Phase 4: RTMP Output (Optional, 1 week)

- [ ] Research RTMP crates (`rtmp-rs`, `flashrust`)
- [ ] Implement basic `RtmpSink`
- [ ] Support H.264 + AAC streams
- [ ] Create example: `28_rtmp_output.rs`

### Phase 5: HTTP Server Integration (3-5 days)

- [ ] Add optional embedded HTTP server for testing
- [ ] Implement segment serving
- [ ] Add CORS headers for player compatibility
- [ ] Create full demo: camera → HLS → browser

---

## Feature Flags

```toml
[features]
# Streaming protocol features
hls = ["mpeg-ts"]
dash = ["mp4-demux", "dep:quick-xml"]
rtmp = ["dep:rtmp"]  # TBD crate

# Combined
streaming-protocols = ["hls", "dash"]
```

---

## Dependencies

| Crate | Version | Purpose | License |
|-------|---------|---------|---------|
| `quick-xml` | 0.31 | MPD generation | MIT |
| `rtmp` | TBD | RTMP protocol | TBD |
| (existing) `mp4` | - | fMP4 muxing | MIT |

---

## HLS Segment Timing

```rust
/// Calculate segment boundaries based on keyframes
pub struct SegmentBoundaryDetector {
    target_duration: f64,
    current_duration: f64,
    last_keyframe_pts: Option<i64>,
}

impl SegmentBoundaryDetector {
    /// Check if we should start a new segment
    pub fn should_rotate(&mut self, pts: i64, is_keyframe: bool) -> bool {
        let duration = self.current_duration;
        
        // Always cut at keyframes when near target duration
        if is_keyframe && duration >= self.target_duration * 0.8 {
            return true;
        }
        
        // Force cut if way over target (shouldn't happen with proper encoding)
        if duration >= self.target_duration * 1.5 {
            log::warn!("Forcing segment cut without keyframe");
            return true;
        }
        
        false
    }
}
```

---

## Testing Strategy

### Unit Tests

```rust
#[test]
fn test_m3u8_generation() {
    let segments = vec![
        SegmentInfo { sequence: 1, duration: 6.0, filename: "seg_001.ts".into() },
        SegmentInfo { sequence: 2, duration: 6.0, filename: "seg_002.ts".into() },
    ];
    
    let playlist = generate_media_playlist(&segments, &HlsConfig::default());
    
    assert!(playlist.contains("#EXTM3U"));
    assert!(playlist.contains("#EXT-X-TARGETDURATION:6"));
    assert!(playlist.contains("seg_001.ts"));
}

#[test]
fn test_mpd_generation() {
    let config = DashConfig::default();
    let mpd = generate_mpd(&config, &DashSegments::empty());
    
    assert!(mpd.contains("<MPD"));
    assert!(mpd.contains("urn:mpeg:dash:schema:mpd:2011"));
}
```

### Integration Tests

```rust
#[tokio::test]
async fn test_hls_pipeline() {
    let temp_dir = tempfile::tempdir().unwrap();
    
    let mut pipeline = Pipeline::new();
    let src = pipeline.add_node("src", VideoTestSrc::new(640, 480, 30));
    let enc = pipeline.add_node("enc", H264Encoder::new(1_000_000)?);
    let mux = pipeline.add_node("mux", TsMux::new()?);
    let sink = pipeline.add_node("sink", HlsSink::new(HlsConfig {
        output_dir: temp_dir.path().into(),
        segment_duration: 2.0,
        ..Default::default()
    })?);
    
    pipeline.link_all(&[src, enc, mux, sink])?;
    pipeline.run_for(Duration::from_secs(10)).await?;
    
    // Verify output
    assert!(temp_dir.path().join("playlist.m3u8").exists());
    assert!(temp_dir.path().join("segment_001.ts").exists());
}
```

### Player Compatibility

Test with:
- `ffplay` / `mpv` (command line)
- `hls.js` (browser)
- Safari (native HLS)
- `dash.js` (browser DASH)

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/elements/streaming/mod.rs` | Module root |
| `src/elements/streaming/hls.rs` | HLS sink |
| `src/elements/streaming/dash.rs` | DASH sink |
| `src/elements/streaming/rtmp.rs` | RTMP sink |
| `src/elements/streaming/abr.rs` | ABR helpers |
| `examples/25_hls_output.rs` | HLS example |
| `examples/26_dash_output.rs` | DASH example |
| `examples/27_abr_pipeline.rs` | Multi-bitrate example |
| `examples/28_rtmp_output.rs` | RTMP example |

---

## Success Criteria

- [ ] HLS output plays in Safari and hls.js
- [ ] DASH output plays in dash.js
- [ ] Segments are correctly timed (±0.5s)
- [ ] Playlists update correctly for live streams
- [ ] Multi-rendition ABR works
- [ ] Examples demonstrate full pipelines

---

## Future Work

| Feature | Notes |
|---------|-------|
| Low-Latency HLS (LL-HLS) | Partial segments, playlist updates |
| Low-Latency DASH | Chunked transfer encoding |
| DRM (Widevine/FairPlay) | Content protection |
| SRT | Secure Reliable Transport |
| WebRTC output | Real-time to browsers |

---

*Created January 2026 based on [Project Analysis Report](../docs/PROJECT_ANALYSIS_REPORT.md)*
