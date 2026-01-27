# Plan 06: Caps Negotiation Improvements

**Priority:** Medium (Short-term)  
**Effort:** Medium (1 week)  
**Dependencies:** None  
**Addresses:** Pain Point 1.6 (Caps Negotiation Not Used in Practice)

---

## Problem Statement

While caps negotiation infrastructure exists, it's not used effectively:

1. **Elements return `Caps::any()`** by default, providing no constraints
2. **No automatic format conversion** when elements are incompatible
3. **Video elements don't declare pixel formats**
4. **No documentation** on how to use caps properly

Example of the problem:
```rust
// VideoScale accepts YUV420 but declares:
fn input_caps(&self) -> Caps {
    Caps::any()  // Should declare YUV420!
}
```

---

## Proposed Solution

1. Add rich format types for video/audio
2. Make elements declare proper caps
3. Implement automatic converter insertion
4. Add caps validation at link time

---

## Design

### Video Format Caps

```rust
/// Video-specific capabilities
#[derive(Debug, Clone, PartialEq)]
pub struct VideoCaps {
    /// Pixel format(s) supported
    pub formats: Vec<PixelFormat>,
    /// Width constraint
    pub width: DimensionConstraint,
    /// Height constraint
    pub height: DimensionConstraint,
    /// Framerate constraint
    pub framerate: FramerateConstraint,
    /// Color space
    pub colorspace: Option<ColorSpace>,
    /// Interlacing mode
    pub interlace_mode: InterlaceMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    // YUV formats
    I420,       // YUV 4:2:0 planar (Y, U, V separate)
    NV12,       // YUV 4:2:0 semi-planar (Y, UV interleaved)
    I422,       // YUV 4:2:2 planar
    I444,       // YUV 4:4:4 planar
    UYVY,       // YUV 4:2:2 packed
    YUY2,       // YUV 4:2:2 packed (YUYV)
    
    // RGB formats
    Rgb24,      // RGB 8-bit per channel
    Bgr24,      // BGR 8-bit per channel
    Rgba32,     // RGBA 8-bit per channel
    Bgra32,     // BGRA 8-bit per channel
    Argb32,     // ARGB 8-bit per channel
    
    // High bit depth
    I420_10LE,  // YUV 4:2:0 10-bit little endian
    P010,       // YUV 4:2:0 10-bit semi-planar
    
    // Gray
    Gray8,      // 8-bit grayscale
    Gray16,     // 16-bit grayscale
}

#[derive(Debug, Clone, PartialEq)]
pub enum DimensionConstraint {
    /// Any dimension
    Any,
    /// Exact value
    Exact(u32),
    /// Range [min, max]
    Range { min: u32, max: u32 },
    /// List of allowed values
    List(Vec<u32>),
    /// Multiple of N (for codec block sizes)
    MultipleOf(u32),
}

#[derive(Debug, Clone, PartialEq)]
pub enum FramerateConstraint {
    Any,
    Exact { num: u32, den: u32 },
    Range { min: f64, max: f64 },
    List(Vec<(u32, u32)>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Bt601,
    Bt709,
    Bt2020,
    Srgb,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InterlaceMode {
    #[default]
    Progressive,
    Interlaced,
    Mixed,
}
```

### Audio Format Caps

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct AudioCaps {
    /// Sample format(s) supported
    pub formats: Vec<SampleFormat>,
    /// Sample rate constraint
    pub rate: SampleRateConstraint,
    /// Channel layout
    pub channels: ChannelConstraint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    // Integer formats
    S16LE,      // Signed 16-bit little endian
    S16BE,      // Signed 16-bit big endian
    S24LE,      // Signed 24-bit little endian (in 32-bit container)
    S32LE,      // Signed 32-bit little endian
    U8,         // Unsigned 8-bit
    
    // Float formats
    F32LE,      // 32-bit float little endian
    F64LE,      // 64-bit float little endian
    
    // Planar formats
    S16P,       // Signed 16-bit planar
    F32P,       // 32-bit float planar
}

#[derive(Debug, Clone, PartialEq)]
pub enum SampleRateConstraint {
    Any,
    Exact(u32),
    Range { min: u32, max: u32 },
    List(Vec<u32>),  // e.g., [44100, 48000, 96000]
}

#[derive(Debug, Clone, PartialEq)]
pub enum ChannelConstraint {
    Any,
    Exact(u32),
    Range { min: u32, max: u32 },
    Layout(ChannelLayout),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelLayout {
    Mono,
    Stereo,
    Surround51,
    Surround71,
}
```

### Unified MediaCaps

```rust
/// Capabilities for any media type
#[derive(Debug, Clone, PartialEq)]
pub enum MediaCaps {
    /// Accept any format
    Any,
    /// Video format
    Video(VideoCaps),
    /// Audio format
    Audio(AudioCaps),
    /// Raw data (no format constraints)
    Data { mime_type: Option<String> },
    /// Encoded stream
    Encoded { codec: Codec },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Codec {
    // Video codecs
    H264,
    H265,
    Av1,
    Vp8,
    Vp9,
    Mpeg2,
    Jpeg,
    
    // Audio codecs
    Aac,
    Mp3,
    Opus,
    Vorbis,
    Flac,
    Pcm,
    
    // Data
    Klv,
    Scte35,
}
```

### Convenience Constructors

```rust
impl VideoCaps {
    /// YUV420 video of any size
    pub fn yuv420() -> Self {
        Self {
            formats: vec![PixelFormat::I420],
            width: DimensionConstraint::Any,
            height: DimensionConstraint::Any,
            framerate: FramerateConstraint::Any,
            colorspace: None,
            interlace_mode: InterlaceMode::Progressive,
        }
    }
    
    /// YUV420 video with specific dimensions
    pub fn yuv420_size(width: u32, height: u32) -> Self {
        Self {
            formats: vec![PixelFormat::I420],
            width: DimensionConstraint::Exact(width),
            height: DimensionConstraint::Exact(height),
            ..Self::yuv420()
        }
    }
    
    /// Multiple YUV formats
    pub fn yuv() -> Self {
        Self {
            formats: vec![PixelFormat::I420, PixelFormat::NV12, PixelFormat::I422],
            ..Self::yuv420()
        }
    }
    
    /// RGB formats
    pub fn rgb() -> Self {
        Self {
            formats: vec![PixelFormat::Rgb24, PixelFormat::Rgba32],
            width: DimensionConstraint::Any,
            height: DimensionConstraint::Any,
            framerate: FramerateConstraint::Any,
            colorspace: Some(ColorSpace::Srgb),
            interlace_mode: InterlaceMode::Progressive,
        }
    }
}

impl MediaCaps {
    pub fn video_yuv420() -> Self {
        MediaCaps::Video(VideoCaps::yuv420())
    }
    
    pub fn video_yuv420_size(width: u32, height: u32) -> Self {
        MediaCaps::Video(VideoCaps::yuv420_size(width, height))
    }
    
    pub fn video_any() -> Self {
        MediaCaps::Video(VideoCaps {
            formats: vec![],  // Empty = any format
            width: DimensionConstraint::Any,
            height: DimensionConstraint::Any,
            framerate: FramerateConstraint::Any,
            colorspace: None,
            interlace_mode: InterlaceMode::Progressive,
        })
    }
    
    pub fn encoded(codec: Codec) -> Self {
        MediaCaps::Encoded { codec }
    }
}
```

### Update VideoScale to Declare Caps

```rust
impl Transform for VideoScale {
    fn input_caps(&self) -> Caps {
        Caps::from(MediaCaps::Video(VideoCaps {
            formats: vec![PixelFormat::I420],
            width: DimensionConstraint::Exact(self.src_width),
            height: DimensionConstraint::Exact(self.src_height),
            framerate: FramerateConstraint::Any,
            colorspace: None,
            interlace_mode: InterlaceMode::Progressive,
        }))
    }
    
    fn output_caps(&self) -> Caps {
        Caps::from(MediaCaps::Video(VideoCaps {
            formats: vec![PixelFormat::I420],
            width: DimensionConstraint::Exact(self.dst_width),
            height: DimensionConstraint::Exact(self.dst_height),
            framerate: FramerateConstraint::Any,
            colorspace: None,
            interlace_mode: InterlaceMode::Progressive,
        }))
    }
}
```

### Caps Intersection

```rust
impl MediaCaps {
    /// Check if two caps can be connected.
    pub fn can_intersect(&self, other: &MediaCaps) -> bool {
        match (self, other) {
            (MediaCaps::Any, _) | (_, MediaCaps::Any) => true,
            (MediaCaps::Video(a), MediaCaps::Video(b)) => a.can_intersect(b),
            (MediaCaps::Audio(a), MediaCaps::Audio(b)) => a.can_intersect(b),
            (MediaCaps::Encoded { codec: a }, MediaCaps::Encoded { codec: b }) => a == b,
            _ => false,
        }
    }
    
    /// Find the common format between two caps.
    pub fn intersect(&self, other: &MediaCaps) -> Option<MediaCaps> {
        match (self, other) {
            (MediaCaps::Any, other) => Some(other.clone()),
            (this, MediaCaps::Any) => Some(this.clone()),
            (MediaCaps::Video(a), MediaCaps::Video(b)) => {
                a.intersect(b).map(MediaCaps::Video)
            }
            // ...
            _ => None,
        }
    }
}

impl VideoCaps {
    pub fn can_intersect(&self, other: &VideoCaps) -> bool {
        // Check format compatibility
        let format_ok = self.formats.is_empty() 
            || other.formats.is_empty()
            || self.formats.iter().any(|f| other.formats.contains(f));
        
        // Check dimension compatibility
        let width_ok = self.width.can_intersect(&other.width);
        let height_ok = self.height.can_intersect(&other.height);
        
        format_ok && width_ok && height_ok
    }
    
    pub fn intersect(&self, other: &VideoCaps) -> Option<VideoCaps> {
        // Find common formats
        let formats = if self.formats.is_empty() {
            other.formats.clone()
        } else if other.formats.is_empty() {
            self.formats.clone()
        } else {
            self.formats.iter()
                .filter(|f| other.formats.contains(f))
                .cloned()
                .collect()
        };
        
        if formats.is_empty() && !self.formats.is_empty() && !other.formats.is_empty() {
            return None;  // No common format
        }
        
        Some(VideoCaps {
            formats,
            width: self.width.intersect(&other.width)?,
            height: self.height.intersect(&other.height)?,
            framerate: self.framerate.intersect(&other.framerate)?,
            colorspace: self.colorspace.or(other.colorspace),
            interlace_mode: self.interlace_mode,
        })
    }
}
```

### Automatic Converter Insertion

When caps don't directly match, insert converters:

```rust
pub struct ConverterRegistry {
    converters: Vec<ConverterInfo>,
}

pub struct ConverterInfo {
    /// What this converter accepts
    pub input: MediaCaps,
    /// What this converter produces
    pub output: MediaCaps,
    /// Factory to create the converter element
    pub factory: Box<dyn Fn() -> Box<dyn PipelineElement> + Send + Sync>,
    /// Conversion cost (lower = preferred)
    pub cost: u32,
}

impl ConverterRegistry {
    pub fn new() -> Self {
        let mut registry = Self { converters: vec![] };
        
        // Register built-in converters
        registry.register(ConverterInfo {
            input: MediaCaps::video_yuv420(),
            output: MediaCaps::Video(VideoCaps::rgb()),
            factory: Box::new(|| Box::new(YuvToRgbConverter::new())),
            cost: 10,
        });
        
        registry.register(ConverterInfo {
            input: MediaCaps::Video(VideoCaps::rgb()),
            output: MediaCaps::video_yuv420(),
            factory: Box::new(|| Box::new(RgbToYuvConverter::new())),
            cost: 10,
        });
        
        registry
    }
    
    /// Find converters to bridge two caps.
    pub fn find_path(&self, from: &MediaCaps, to: &MediaCaps) -> Option<Vec<&ConverterInfo>> {
        // Direct connection possible?
        if from.can_intersect(to) {
            return Some(vec![]);
        }
        
        // Single converter?
        for conv in &self.converters {
            if from.can_intersect(&conv.input) && conv.output.can_intersect(to) {
                return Some(vec![conv]);
            }
        }
        
        // Multi-hop? (Dijkstra's algorithm for lowest cost path)
        // ...
        
        None
    }
}
```

### Pipeline Negotiation

```rust
impl Pipeline {
    /// Run caps negotiation and insert converters as needed.
    pub fn negotiate_with_converters(&mut self, registry: &ConverterRegistry) -> Result<()> {
        // First pass: collect caps from all elements
        let mut element_caps: HashMap<NodeId, (MediaCaps, MediaCaps)> = HashMap::new();
        
        for (id, node) in self.nodes() {
            let input = node.input_caps().to_media_caps();
            let output = node.output_caps().to_media_caps();
            element_caps.insert(id, (input, output));
        }
        
        // Second pass: check each link
        let mut converters_to_insert = vec![];
        
        for link in self.links() {
            let (_, src_output) = &element_caps[&link.source_id];
            let (sink_input, _) = &element_caps[&link.sink_id];
            
            if !src_output.can_intersect(sink_input) {
                // Need converter(s)
                match registry.find_path(src_output, sink_input) {
                    Some(path) if !path.is_empty() => {
                        converters_to_insert.push((link.id, path));
                    }
                    Some(_) => {
                        // Direct connection works
                    }
                    None => {
                        return Err(Error::NegotiationFailed(format!(
                            "No conversion path from {:?} to {:?} between {} and {}",
                            src_output, sink_input, link.source_name, link.sink_name
                        )));
                    }
                }
            }
        }
        
        // Third pass: insert converters
        for (link_id, converters) in converters_to_insert {
            self.insert_converters_at_link(link_id, converters)?;
        }
        
        Ok(())
    }
}
```

---

## Implementation Steps

### Step 1: Define Format Types

**File:** `src/format/video.rs`

- `PixelFormat`, `VideoCaps`, constraints

**File:** `src/format/audio.rs`

- `SampleFormat`, `AudioCaps`, constraints

**File:** `src/format/caps.rs`

- `MediaCaps`, `Codec`, intersection logic

### Step 2: Update Caps Type

**File:** `src/format/mod.rs`

- Update `Caps` to use `MediaCaps` internally
- Add conversion methods

### Step 3: Update Elements to Declare Caps

**Files:** Various elements
- `VideoScale`: `VideoCaps::yuv420_size()`
- `Rav1eEncoder`: input=raw, output=encoded AV1
- `TcpSrc`: `MediaCaps::Data`

### Step 4: Implement Converter Registry

**File:** `src/negotiation/converter.rs`

- `ConverterRegistry`
- `ConverterInfo`
- Path finding

### Step 5: Add Basic Converters

**File:** `src/elements/convert/`

- `YuvToRgbConverter`
- `RgbToYuvConverter`
- `PixelFormatConverter`

### Step 6: Update Pipeline Negotiation

**File:** `src/pipeline/graph.rs`

- `negotiate_with_converters()`
- Converter insertion

### Step 7: Create Example

**File:** `examples/35_caps_negotiation.rs`

```rust
// Create pipeline with incompatible elements
let mut pipeline = Pipeline::new();

let rgb_source = pipeline.add_element("src", Box::new(RgbTestSource::new()));
let yuv_sink = pipeline.add_element("sink", Box::new(YuvFileSink::new("out.yuv")));

pipeline.link(rgb_source, yuv_sink)?;

// Negotiation will auto-insert RGB→YUV converter
let registry = ConverterRegistry::new();
pipeline.negotiate_with_converters(&registry)?;

println!("{}", pipeline.describe());
// Shows: src → rgb_to_yuv_0 → sink

pipeline.run().await?;
```

---

## Validation Criteria

- [ ] `VideoCaps` and `AudioCaps` defined with constraints
- [ ] `MediaCaps` unifies all format types
- [ ] Caps intersection works correctly
- [ ] `VideoScale` declares proper caps
- [ ] Other video elements declare proper caps
- [ ] `ConverterRegistry` finds conversion paths
- [ ] Basic YUV↔RGB converters implemented
- [ ] Pipeline auto-inserts converters
- [ ] Example demonstrates negotiation
- [ ] All existing tests pass

---

## Future Enhancements

1. **Caps query:** Elements query downstream caps before producing
2. **Dynamic renegotiation:** Renegotiate when stream properties change
3. **Hardware converters:** GPU-accelerated format conversion
4. **Audio converters:** Sample rate, channel layout conversion
5. **Caps filtering:** User-specified caps constraints

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `src/format/video.rs` | New: PixelFormat, VideoCaps |
| `src/format/audio.rs` | New: SampleFormat, AudioCaps |
| `src/format/caps.rs` | Update: MediaCaps with intersection |
| `src/format/mod.rs` | Export new types |
| `src/negotiation/converter.rs` | New: ConverterRegistry |
| `src/elements/convert/mod.rs` | New: converter elements |
| `src/elements/convert/yuv_rgb.rs` | New: YUV↔RGB converters |
| `src/elements/transform/scale.rs` | Declare proper caps |
| `src/pipeline/graph.rs` | negotiate_with_converters() |
| `examples/35_caps_negotiation.rs` | New example |
