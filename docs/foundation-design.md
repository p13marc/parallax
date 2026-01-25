# Parallax Foundation Improvements Design

Architectural improvements for media streaming support. Design prioritizes clean API, type safety, and zero-cost abstractions.

## Design Principles

1. **Zero-cost abstractions** - No runtime overhead for unused features
2. **Type safety** - Leverage Rust's type system, avoid stringly-typed APIs
3. **Small, Copy types** - Avoid allocations in hot paths
4. **Explicit over implicit** - Clear ownership and data flow

---

## 1. Media Format & Caps

### MediaFormat - Type-Safe Format Description

```rust
/// Media format - describes buffer contents
#[derive(Clone, Debug, PartialEq)]
pub enum MediaFormat {
    /// Raw video frames
    VideoRaw(VideoFormat),
    /// Encoded video
    Video(VideoCodec),
    /// Raw audio samples
    AudioRaw(AudioFormat),
    /// Encoded audio
    Audio(AudioCodec),
    /// RTP packet
    Rtp(RtpFormat),
    /// MPEG-TS packet
    MpegTs,
    /// Raw bytes (no format constraints)
    Bytes,
}

/// Raw video format (24 bytes, Copy)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct VideoFormat {
    pub width: u32,
    pub height: u32,
    pub pixel_format: PixelFormat,
    pub framerate: Framerate,
}

/// Raw audio format (8 bytes, Copy)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AudioFormat {
    pub sample_rate: u32,
    pub channels: u16,
    pub sample_format: SampleFormat,
}

/// RTP stream format (8 bytes, Copy)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RtpFormat {
    pub payload_type: u8,
    pub clock_rate: u32,
    pub encoding: RtpEncoding,
}

/// What's inside the RTP payload
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RtpEncoding {
    H264,
    H265,
    Vp8,
    Vp9,
    Opus,
    Pcmu,
    Pcma,
    Dynamic(u8),
}

/// Pixel formats
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum PixelFormat {
    #[default]
    I420 = 0,    // YUV 4:2:0 planar (most common)
    Nv12,        // YUV 4:2:0 semi-planar
    Yuyv,        // YUV 4:2:2 packed
    Rgb24,
    Rgba,
    Bgr24,
    Bgra,
}

/// Audio sample formats
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
#[repr(u8)]
pub enum SampleFormat {
    #[default]
    S16 = 0,     // Signed 16-bit (most common)
    S32,
    F32,
    U8,
}

/// Video codecs
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum VideoCodec { H264, H265, Vp8, Vp9, Av1 }

/// Audio codecs  
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum AudioCodec { Opus, Aac, Mp3, Pcmu, Pcma }

/// Framerate as num/den (8 bytes, Copy)
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Framerate { pub num: u32, pub den: u32 }

impl Framerate {
    pub const fn new(num: u32, den: u32) -> Self { Self { num, den } }
    pub const FPS_24: Self = Self::new(24, 1);
    pub const FPS_25: Self = Self::new(25, 1);
    pub const FPS_30: Self = Self::new(30, 1);
    pub const FPS_60: Self = Self::new(60, 1);
    
    #[inline]
    pub fn fps(&self) -> f64 {
        self.num as f64 / self.den.max(1) as f64
    }
}

impl Default for Framerate {
    fn default() -> Self { Self::FPS_30 }
}
```

### Caps - Element Capabilities

```rust
/// Capabilities: what formats an element accepts/produces
#[derive(Clone, Debug, PartialEq)]
pub struct Caps(SmallVec<[MediaFormat; 2]>);

impl Caps {
    /// Accepts any format
    pub fn any() -> Self { Self(SmallVec::new()) }
    
    /// Single format
    pub fn new(format: MediaFormat) -> Self {
        let mut v = SmallVec::new();
        v.push(format);
        Self(v)
    }
    
    /// Multiple acceptable formats (first = preferred)
    pub fn many(formats: impl IntoIterator<Item = MediaFormat>) -> Self {
        Self(formats.into_iter().collect())
    }
    
    /// Is this "any format"?
    #[inline]
    pub fn is_any(&self) -> bool { self.0.is_empty() }
    
    /// Is this a single fixed format?
    #[inline]
    pub fn is_fixed(&self) -> bool { self.0.len() == 1 }
    
    /// Get formats
    #[inline]
    pub fn formats(&self) -> &[MediaFormat] { &self.0 }
    
    /// Preferred format (first one)
    #[inline]
    pub fn preferred(&self) -> Option<&MediaFormat> { self.0.first() }
    
    /// Check if compatible with another caps
    pub fn intersects(&self, other: &Caps) -> bool {
        if self.is_any() || other.is_any() { return true; }
        self.0.iter().any(|a| other.0.iter().any(|b| a.compatible(b)))
    }
    
    /// Find first compatible format
    pub fn negotiate(&self, other: &Caps) -> Option<MediaFormat> {
        if self.is_any() { return other.preferred().cloned(); }
        if other.is_any() { return self.preferred().cloned(); }
        self.0.iter()
            .find(|a| other.0.iter().any(|b| a.compatible(b)))
            .cloned()
    }
}

impl MediaFormat {
    /// Check compatibility (can data flow between these formats?)
    pub fn compatible(&self, other: &MediaFormat) -> bool {
        match (self, other) {
            (Self::Bytes, _) | (_, Self::Bytes) => true,
            (Self::VideoRaw(a), Self::VideoRaw(b)) => a == b,
            (Self::Video(a), Self::Video(b)) => a == b,
            (Self::AudioRaw(a), Self::AudioRaw(b)) => a == b,
            (Self::Audio(a), Self::Audio(b)) => a == b,
            (Self::Rtp(a), Self::Rtp(b)) => a.payload_type == b.payload_type,
            (Self::MpegTs, Self::MpegTs) => true,
            _ => false,
        }
    }
}

// Convenient construction
impl From<VideoFormat> for MediaFormat {
    fn from(v: VideoFormat) -> Self { Self::VideoRaw(v) }
}
impl From<AudioFormat> for MediaFormat {
    fn from(v: AudioFormat) -> Self { Self::AudioRaw(v) }
}
impl From<VideoCodec> for MediaFormat {
    fn from(v: VideoCodec) -> Self { Self::Video(v) }
}
impl From<AudioCodec> for MediaFormat {
    fn from(v: AudioCodec) -> Self { Self::Audio(v) }
}
impl From<MediaFormat> for Caps {
    fn from(f: MediaFormat) -> Self { Self::new(f) }
}
```

### Codec Data (Only When Needed)

```rust
/// Codec initialization data (SPS/PPS for H.264, etc.)
/// Only allocated when codec requires out-of-band config
#[derive(Clone, Debug, PartialEq)]
pub struct CodecData(Box<[u8]>);

impl CodecData {
    pub fn new(data: impl Into<Box<[u8]>>) -> Self { Self(data.into()) }
    #[inline] pub fn as_slice(&self) -> &[u8] { &self.0 }
}

impl AsRef<[u8]> for CodecData {
    fn as_ref(&self) -> &[u8] { &self.0 }
}
```

---

## 2. Element Traits

### Clean Multi-Output Design

```rust
/// Output of element processing
pub enum Output {
    /// No output (filtered/buffering)
    None,
    /// Single buffer
    Single(Buffer),
    /// Multiple buffers (same destination)
    Multiple(Vec<Buffer>),
}

impl Output {
    #[inline]
    pub fn single(buf: Buffer) -> Self { Self::Single(buf) }
    
    #[inline]
    pub fn none() -> Self { Self::None }
    
    pub fn into_vec(self) -> Vec<Buffer> {
        match self {
            Self::None => vec![],
            Self::Single(b) => vec![b],
            Self::Multiple(v) => v,
        }
    }
}

// Ergonomic conversions
impl From<Buffer> for Output {
    #[inline] fn from(b: Buffer) -> Self { Self::Single(b) }
}
impl From<Vec<Buffer>> for Output {
    fn from(v: Vec<Buffer>) -> Self {
        match v.len() {
            0 => Self::None,
            1 => Self::Single(v.into_iter().next().unwrap()),
            _ => Self::Multiple(v),
        }
    }
}
impl FromIterator<Buffer> for Output {
    fn from_iter<I: IntoIterator<Item = Buffer>>(iter: I) -> Self {
        iter.into_iter().collect::<Vec<_>>().into()
    }
}
```

### Element Traits

```rust
/// A source produces buffers
pub trait Source: Send {
    /// Produce next buffer, None = EOS
    fn produce(&mut self) -> Result<Option<Buffer>>;
    
    /// Element name
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    
    /// Output format (for pipeline validation)
    fn output_caps(&self) -> Caps { Caps::any() }
}

/// A sink consumes buffers
pub trait Sink: Send {
    /// Consume a buffer
    fn consume(&mut self, buffer: Buffer) -> Result<()>;
    
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    
    /// Accepted input format
    fn input_caps(&self) -> Caps { Caps::any() }
}

/// A transform processes buffers (1 input -> 0..N outputs)
pub trait Transform: Send {
    /// Process input, produce output(s)
    fn transform(&mut self, buffer: Buffer) -> Result<Output>;
    
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    
    fn input_caps(&self) -> Caps { Caps::any() }
    fn output_caps(&self) -> Caps { Caps::any() }
}

/// Async source
pub trait AsyncSource: Send {
    fn produce(&mut self) -> impl Future<Output = Result<Option<Buffer>>> + Send;
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    fn output_caps(&self) -> Caps { Caps::any() }
}

/// Async sink
pub trait AsyncSink: Send {
    fn consume(&mut self, buffer: Buffer) -> impl Future<Output = Result<()>> + Send;
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    fn input_caps(&self) -> Caps { Caps::any() }
}

/// Async transform
pub trait AsyncTransform: Send {
    fn transform(&mut self, buffer: Buffer) -> impl Future<Output = Result<Output>> + Send;
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    fn input_caps(&self) -> Caps { Caps::any() }
    fn output_caps(&self) -> Caps { Caps::any() }
}
```

### Demuxer (1 Input -> N Outputs by Stream ID)

```rust
/// Output pad identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PadId(pub u32);

/// Routed output (for demuxers)
pub struct RoutedOutput(pub SmallVec<[(PadId, Buffer); 2]>);

impl RoutedOutput {
    pub fn new() -> Self { Self(SmallVec::new()) }
    
    pub fn push(&mut self, pad: PadId, buffer: Buffer) {
        self.0.push((pad, buffer));
    }
    
    pub fn single(pad: PadId, buffer: Buffer) -> Self {
        let mut r = Self::new();
        r.push(pad, buffer);
        r
    }
}

impl IntoIterator for RoutedOutput {
    type Item = (PadId, Buffer);
    type IntoIter = smallvec::IntoIter<[(PadId, Buffer); 2]>;
    fn into_iter(self) -> Self::IntoIter { self.0.into_iter() }
}

/// Demuxer: routes input to multiple output pads
pub trait Demuxer: Send {
    /// Process input, route to output pads
    fn demux(&mut self, buffer: Buffer) -> Result<RoutedOutput>;
    
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    
    /// Current output pads and their formats
    fn outputs(&self) -> &[(PadId, Caps)];
    
    /// Subscribe to pad changes (for dynamic streams)
    fn on_pad_added(&mut self, callback: Box<dyn FnMut(PadId, Caps) + Send>);
}
```

---

## 3. Clock & Time

### ClockTime - Nanosecond Timestamp

```rust
/// Time in nanoseconds (8 bytes, Copy)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct ClockTime(u64);

impl ClockTime {
    pub const ZERO: Self = Self(0);
    pub const MAX: Self = Self(u64::MAX - 1);
    pub const NONE: Self = Self(u64::MAX);  // Invalid/unset
    
    #[inline] pub const fn from_nanos(ns: u64) -> Self { Self(ns) }
    #[inline] pub const fn from_micros(us: u64) -> Self { Self(us.saturating_mul(1_000)) }
    #[inline] pub const fn from_millis(ms: u64) -> Self { Self(ms.saturating_mul(1_000_000)) }
    #[inline] pub const fn from_secs(s: u64) -> Self { Self(s.saturating_mul(1_000_000_000)) }
    
    #[inline] pub const fn nanos(self) -> u64 { self.0 }
    #[inline] pub const fn micros(self) -> u64 { self.0 / 1_000 }
    #[inline] pub const fn millis(self) -> u64 { self.0 / 1_000_000 }
    #[inline] pub const fn secs(self) -> u64 { self.0 / 1_000_000_000 }
    
    #[inline] pub const fn is_none(self) -> bool { self.0 == u64::MAX }
    #[inline] pub const fn is_some(self) -> bool { self.0 != u64::MAX }
    
    #[inline]
    pub const fn saturating_add(self, rhs: Self) -> Self {
        if self.is_none() || rhs.is_none() { return Self::NONE; }
        Self(self.0.saturating_add(rhs.0))
    }
    
    #[inline]
    pub const fn saturating_sub(self, rhs: Self) -> Self {
        if self.is_none() || rhs.is_none() { return Self::NONE; }
        Self(self.0.saturating_sub(rhs.0))
    }
    
    #[inline]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        if self.is_none() || rhs.is_none() { return None; }
        match self.0.checked_sub(rhs.0) {
            Some(v) => Some(Self(v)),
            None => None,
        }
    }
}

impl std::ops::Add for ClockTime {
    type Output = Self;
    #[inline] fn add(self, rhs: Self) -> Self { self.saturating_add(rhs) }
}

impl std::ops::Sub for ClockTime {
    type Output = Self;
    #[inline] fn sub(self, rhs: Self) -> Self { self.saturating_sub(rhs) }
}

impl From<Duration> for ClockTime {
    #[inline] fn from(d: Duration) -> Self { Self(d.as_nanos() as u64) }
}

impl From<ClockTime> for Duration {
    #[inline] fn from(t: ClockTime) -> Self { 
        if t.is_none() { Duration::ZERO } else { Duration::from_nanos(t.0) }
    }
}

impl std::fmt::Display for ClockTime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_none() {
            write!(f, "NONE")
        } else {
            let secs = self.secs();
            let ms = (self.0 / 1_000_000) % 1000;
            write!(f, "{}.{:03}s", secs, ms)
        }
    }
}
```

### Clock Trait

```rust
/// Pipeline clock
pub trait Clock: Send + Sync {
    /// Current time
    fn now(&self) -> ClockTime;
}

/// System monotonic clock
pub struct SystemClock(Instant);

impl SystemClock {
    pub fn new() -> Self { Self(Instant::now()) }
}

impl Default for SystemClock {
    fn default() -> Self { Self::new() }
}

impl Clock for SystemClock {
    #[inline]
    fn now(&self) -> ClockTime {
        ClockTime::from_nanos(self.0.elapsed().as_nanos() as u64)
    }
}

/// Pipeline timing context
pub struct PipelineClock {
    clock: Arc<dyn Clock>,
    base_time: AtomicU64,
}

impl PipelineClock {
    pub fn new(clock: Arc<dyn Clock>) -> Self {
        Self { clock, base_time: AtomicU64::new(u64::MAX) }
    }
    
    pub fn system() -> Self {
        Self::new(Arc::new(SystemClock::new()))
    }
    
    /// Start the pipeline clock
    pub fn start(&self) {
        self.base_time.store(self.clock.now().0, Ordering::Release);
    }
    
    /// Running time since start
    #[inline]
    pub fn running_time(&self) -> ClockTime {
        let base = self.base_time.load(Ordering::Acquire);
        if base == u64::MAX { return ClockTime::NONE; }
        self.clock.now().saturating_sub(ClockTime(base))
    }
    
    /// Wait until running time reaches target
    pub async fn wait_until(&self, target: ClockTime) {
        if target.is_none() { return; }
        loop {
            let now = self.running_time();
            if now.is_none() || now >= target { break; }
            let wait = Duration::from(target.saturating_sub(now));
            tokio::time::sleep(wait.min(Duration::from_millis(10))).await;
        }
    }
}
```

---

## 4. Updated Metadata

### Clean Metadata Structure

```rust
/// Buffer metadata
#[derive(Clone, Debug, Default)]
pub struct Metadata {
    /// Presentation timestamp
    pub pts: ClockTime,
    /// Decode timestamp (if different from PTS)
    pub dts: ClockTime,
    /// Duration
    pub duration: ClockTime,
    /// Sequence number
    pub sequence: u64,
    /// Stream identifier (for demuxed streams)
    pub stream_id: u32,
    /// Flags
    pub flags: BufferFlags,
    /// RTP-specific fields (only for RTP buffers)
    pub rtp: Option<RtpMeta>,
    /// Media format (set on format changes or first buffer)
    pub format: Option<MediaFormat>,
}

/// Buffer flags (1 byte)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferFlags(u8);

impl BufferFlags {
    pub const NONE: Self = Self(0);
    pub const SYNC_POINT: Self = Self(1 << 0);  // Keyframe
    pub const EOS: Self = Self(1 << 1);         // End of stream
    pub const CORRUPTED: Self = Self(1 << 2);   // Data may be invalid
    pub const DISCONT: Self = Self(1 << 3);     // Discontinuity
    pub const DELTA: Self = Self(1 << 4);       // Depends on previous
    pub const HEADER: Self = Self(1 << 5);      // Contains header data
    
    #[inline] pub const fn contains(self, flag: Self) -> bool { (self.0 & flag.0) != 0 }
    #[inline] pub const fn insert(self, flag: Self) -> Self { Self(self.0 | flag.0) }
    #[inline] pub const fn remove(self, flag: Self) -> Self { Self(self.0 & !flag.0) }
    
    #[inline] pub const fn is_keyframe(self) -> bool { self.contains(Self::SYNC_POINT) }
    #[inline] pub const fn is_eos(self) -> bool { self.contains(Self::EOS) }
    #[inline] pub const fn is_discont(self) -> bool { self.contains(Self::DISCONT) }
}

impl std::ops::BitOr for BufferFlags {
    type Output = Self;
    #[inline] fn bitor(self, rhs: Self) -> Self { Self(self.0 | rhs.0) }
}

/// RTP header fields (12 bytes, Copy)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtpMeta {
    pub seq: u16,           // Sequence number
    pub ts: u32,            // RTP timestamp
    pub ssrc: u32,          // Synchronization source
    pub pt: u8,             // Payload type
    pub marker: bool,       // Marker bit (frame boundary)
}

impl RtpMeta {
    /// Convert RTP timestamp to ClockTime
    #[inline]
    pub fn timestamp_to_clock(self, clock_rate: u32) -> ClockTime {
        if clock_rate == 0 { return ClockTime::NONE; }
        ClockTime::from_nanos((self.ts as u64 * 1_000_000_000) / clock_rate as u64)
    }
    
    /// Convert ClockTime to RTP timestamp
    #[inline]
    pub fn clock_to_timestamp(time: ClockTime, clock_rate: u32) -> u32 {
        if time.is_none() { return 0; }
        ((time.nanos() * clock_rate as u64) / 1_000_000_000) as u32
    }
}

impl Metadata {
    pub fn new() -> Self { Self::default() }
    
    pub fn with_pts(mut self, pts: ClockTime) -> Self { self.pts = pts; self }
    pub fn with_dts(mut self, dts: ClockTime) -> Self { self.dts = dts; self }
    pub fn with_duration(mut self, d: ClockTime) -> Self { self.duration = d; self }
    pub fn with_sequence(mut self, seq: u64) -> Self { self.sequence = seq; self }
    pub fn with_stream_id(mut self, id: u32) -> Self { self.stream_id = id; self }
    pub fn with_flags(mut self, f: BufferFlags) -> Self { self.flags = f; self }
    pub fn with_rtp(mut self, rtp: RtpMeta) -> Self { self.rtp = Some(rtp); self }
    pub fn with_format(mut self, f: MediaFormat) -> Self { self.format = Some(f); self }
    
    pub fn keyframe(mut self) -> Self { 
        self.flags = self.flags.insert(BufferFlags::SYNC_POINT); 
        self 
    }
    
    pub fn eos(mut self) -> Self { 
        self.flags = self.flags.insert(BufferFlags::EOS); 
        self 
    }
}
```

---

## 5. Implementation Plan

### Phase 1: Core Types (2-3 days)

**New file: `src/format.rs`**
- `MediaFormat`, `VideoFormat`, `AudioFormat`, `RtpFormat`
- `Caps`
- `PixelFormat`, `SampleFormat`, `VideoCodec`, `AudioCodec`
- `Framerate`, `CodecData`

### Phase 2: Clock (1-2 days)

**New file: `src/clock.rs`**
- `ClockTime`
- `Clock` trait
- `SystemClock`
- `PipelineClock`

### Phase 3: Update Metadata (1 day)

**Modify: `src/metadata.rs`**
- Replace `pts: Option<Duration>` with `pts: ClockTime`
- Replace `dts: Option<Duration>` with `dts: ClockTime`
- Add `duration: ClockTime`
- Change `stream_id: Option<u64>` to `stream_id: u32`
- Replace `BufferFlags` with bitflags version
- Add `rtp: Option<RtpMeta>`
- Add `format: Option<MediaFormat>`
- Remove `extra: Vec<ExtraField>` (use typed fields)

### Phase 4: Element Traits (2-3 days)

**Modify: `src/element/traits.rs`**
- Rename `Element` to `Transform`
- Change return type to `Output`
- Add `input_caps()`, `output_caps()` methods
- Add `AsyncTransform` trait
- Add `Demuxer` trait with `PadId`, `RoutedOutput`

### Phase 5: Update Existing Elements (2-3 days)

- Update all elements to use new `Transform` trait
- Implement `input_caps()`/`output_caps()` where meaningful
- Migrate FlatMap, Chunk, BufferSplit to use `Output::Multiple`

### Phase 6: Executor Updates (1-2 days)

**Modify: `src/pipeline/executor.rs`**
- Handle `Output::Multiple`
- Add caps validation in pipeline setup
- Add `PipelineClock` to executor
- Support `Demuxer` with routed outputs

---

## Summary

| Type | Size | Copy | Heap |
|------|------|------|------|
| `ClockTime` | 8 bytes | ✓ | ✗ |
| `VideoFormat` | 24 bytes | ✓ | ✗ |
| `AudioFormat` | 8 bytes | ✓ | ✗ |
| `RtpFormat` | 8 bytes | ✓ | ✗ |
| `RtpMeta` | 12 bytes | ✓ | ✗ |
| `BufferFlags` | 1 byte | ✓ | ✗ |
| `Caps` | ~48 bytes | ✗ | SmallVec inline |
| `Output` | ~32 bytes | ✗ | Only for Multiple |
| `Metadata` | ~80 bytes | ✗ | Only if format/rtp set |

**Total estimated time: 10-12 days**

---

## References

- [GStreamer Caps](https://gstreamer.freedesktop.org/documentation/additional/design/caps.html)
- [GStreamer Clocks](https://gstreamer.freedesktop.org/documentation/additional/design/clocks.html)
