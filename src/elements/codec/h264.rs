//! H.264/AVC codec elements using OpenH264.
//!
//! This module provides H.264 encoding and decoding using Cisco's OpenH264 library.
//! The library is BSD-2 licensed and the source code is bundled with the crate.
//!
//! # Example - Encoding
//!
//! ```rust,ignore
//! use parallax::elements::codec::{H264Encoder, H264EncoderConfig};
//!
//! // Create encoder for 1920x1080 video
//! let config = H264EncoderConfig::new(1920, 1080);
//! let mut encoder = H264Encoder::new(config)?;
//!
//! // Encode YUV frames
//! let encoded = encoder.encode_yuv420(&yuv_data)?;
//! ```
//!
//! # Example - Decoding
//!
//! ```rust,ignore
//! use parallax::elements::codec::H264Decoder;
//!
//! let mut decoder = H264Decoder::new()?;
//!
//! // Decode NAL units
//! if let Some(frame) = decoder.decode(&nal_data)? {
//!     // Process decoded YUV frame
//!     let yuv_data = frame.yuv_data();
//! }
//! ```

use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::memory::SharedArena;
use crate::metadata::Metadata;

use openh264::OpenH264API;
use openh264::decoder::{DecodedYUV, Decoder};
use openh264::encoder::{BitRate, Encoder, EncoderConfig, FrameRate, IntraFramePeriod, QpRange};
use openh264::formats::YUVSource;

// ============================================================================
// Encoder Configuration
// ============================================================================

/// How the encoder manages SPS/PPS parameter-set IDs (mirrors OpenH264's
/// `eSpsPpsIdStrategy`).
///
/// Note: with OpenH264 0.9.x, the parameter sets themselves are written at
/// the start of **every** IDR access unit under every strategy (verified by
/// the mid-stream-join regression test below), so any keyframe is a
/// self-contained decoder entry point regardless of this setting. The
/// strategy only governs how the SPS/PPS *IDs* evolve across IDRs —
/// `IncreasingId` lets decoders detect missed IDRs, the listing variants
/// maintain parameter-set lists for multi-stream scenarios.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SpsPpsStrategy {
    /// Constant SPS/PPS ID across the whole encode session (OpenH264's and
    /// this crate's default).
    #[default]
    ConstantId,
    /// Increment the SPS/PPS ID with each IDR, letting decoders detect
    /// missing IDR frames.
    IncreasingId,
    /// Use SPS in the existing list if possible.
    SpsListing,
    /// SPS listing with increasing PPS IDs.
    SpsListingAndPpsIncreasing,
    /// Full SPS/PPS listing.
    SpsPpsListing,
}

impl SpsPpsStrategy {
    fn to_openh264(self) -> openh264::encoder::SpsPpsStrategy {
        use openh264::encoder::SpsPpsStrategy as O;
        match self {
            Self::ConstantId => O::ConstantId,
            Self::IncreasingId => O::IncreasingId,
            Self::SpsListing => O::SpsListing,
            Self::SpsListingAndPpsIncreasing => O::SpsListingAndPpsIncreasing,
            Self::SpsPpsListing => O::SpsPpsListing,
        }
    }
}

/// How the encoder trades bits against quality (mirrors OpenH264's `iRCMode`).
///
/// This is the knob that decides how seriously
/// [`bitrate_bps`](H264EncoderConfig::bitrate_bps) is taken. The default is
/// [`Quality`](Self::Quality), matching OpenH264 — a streaming sensor on a
/// constrained link usually wants [`Bitrate`](Self::Bitrate) instead.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum RateControlMode {
    /// Quality first: the bitrate target is a hint, not a budget (OpenH264's
    /// default, and this crate's).
    #[default]
    Quality,
    /// Bitrate first: hold the target, spending quality to do it. What you want
    /// when the link, not the picture, is the constraint.
    Bitrate,
    /// Ignore the bitrate target; adjust quality from buffer status alone.
    BufferBased,
    /// Rate control driven by frame timestamps.
    Timestamp,
    /// No rate control at all: quality is governed purely by the QP band.
    Off,
}

impl RateControlMode {
    fn to_openh264(self) -> openh264::encoder::RateControlMode {
        use openh264::encoder::RateControlMode as O;
        match self {
            Self::Quality => O::Quality,
            Self::Bitrate => O::Bitrate,
            Self::BufferBased => O::Bufferbased,
            Self::Timestamp => O::Timestamp,
            Self::Off => O::Off,
        }
    }
}

/// H.264 profile (decoder compatibility).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Profile {
    /// Baseline: no B-frames or CABAC. The most widely decodable.
    Baseline,
    /// Main: CABAC and B-frames. The usual choice for streaming.
    Main,
    /// High: Main plus 8x8 transforms. Best compression, still ubiquitous.
    High,
}

impl Profile {
    fn to_openh264(self) -> openh264::encoder::Profile {
        use openh264::encoder::Profile as O;
        match self {
            Self::Baseline => O::Baseline,
            Self::Main => O::Main,
            Self::High => O::High,
        }
    }
}

/// Encoder complexity: CPU spent per frame.
///
/// The knob to reach for when encoding cannot keep up — cheaper than dropping
/// resolution, and invisible to the receiver.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum Complexity {
    /// Fastest, lowest quality per bit.
    Low,
    /// OpenH264's default.
    #[default]
    Medium,
    /// Slowest, best quality per bit.
    High,
}

impl Complexity {
    fn to_openh264(self) -> openh264::encoder::Complexity {
        use openh264::encoder::Complexity as O;
        match self {
            Self::Low => O::Low,
            Self::Medium => O::Medium,
            Self::High => O::High,
        }
    }
}

/// What kind of content is being encoded, which tunes the encoder's heuristics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum UsageType {
    /// Camera video, real-time (the default).
    #[default]
    CameraRealtime,
    /// Screen content, real-time — for `ScreenCaptureSrc` and friends, where
    /// sharp edges and static regions dominate.
    ScreenRealtime,
    /// Camera video, non-real-time (offline transcode).
    CameraNonRealtime,
    /// Screen content, non-real-time.
    ScreenNonRealtime,
}

impl UsageType {
    fn to_openh264(self) -> openh264::encoder::UsageType {
        use openh264::encoder::UsageType as O;
        match self {
            Self::CameraRealtime => O::CameraVideoRealTime,
            Self::ScreenRealtime => O::ScreenContentRealTime,
            Self::CameraNonRealtime => O::CameraVideoNonRealTime,
            Self::ScreenNonRealtime => O::ScreenContentNonRealTime,
        }
    }
}

/// H.264 encoder configuration.
#[derive(Debug, Clone)]
pub struct H264EncoderConfig {
    /// Video width in pixels.
    pub width: u32,
    /// Video height in pixels.
    pub height: u32,
    /// Target bitrate in bits per second (0 = auto).
    pub bitrate_bps: u32,
    /// Maximum frame rate in Hz.
    pub max_frame_rate: f32,
    /// Target quantization parameter (0-51, lower = better quality, larger
    /// files). Default is 26. The encoder's rate control is constrained to
    /// a QP band of ±4 around this value, so it keeps some freedom to meet
    /// `bitrate_bps` while staying near the requested quality.
    pub qp: u8,
    /// Enable scene change detection.
    pub scene_change_detect: bool,
    /// Keyframe interval in frames: an IDR is emitted every N frames
    /// (0 = encoder decides). Late joiners on a stream wait at most one
    /// interval for a decodable frame.
    pub keyframe_interval: u32,
    /// Number of threads (0 = auto).
    pub num_threads: u32,
    /// SPS/PPS parameter-set ID strategy (see [`SpsPpsStrategy`]).
    pub sps_pps_strategy: SpsPpsStrategy,
    /// How strictly [`bitrate_bps`](Self::bitrate_bps) is honoured (see
    /// [`RateControlMode`]). Defaults to `Quality`, matching OpenH264.
    pub rate_control: RateControlMode,
    /// Cap on the size of each emitted NAL unit, in bytes.
    ///
    /// `None` (the default) emits one slice per frame. Setting this to just
    /// under the path MTU (~1200 bytes for RTP over Ethernet) makes the encoder
    /// produce packet-sized NALs, so the payloader does not have to fragment
    /// every slice.
    pub max_slice_len: Option<u32>,
    /// Whether rate control may **drop frames** to hold the bitrate target.
    ///
    /// `None` (the default) means "whatever the rate control needs": skipping
    /// is on when a bitrate target is set, off in quality mode. That is
    /// OpenH264's own behaviour, and it has a sharp edge — under a tight target
    /// the encoder simply emits nothing for some input frames, so a pipeline
    /// that expects one packet per frame quietly gets fewer.
    ///
    /// Set `Some(false)` when something upstream already limits the framerate
    /// (a [`Throttle`](crate::elements::Throttle), say) and you would rather
    /// every frame you send be encoded, spending quality instead of frames.
    pub skip_frames: Option<bool>,
    /// H.264 profile. `None` lets OpenH264 choose.
    pub profile: Option<Profile>,
    /// CPU spent per frame (see [`Complexity`]).
    pub complexity: Complexity,
    /// Content type, which tunes the encoder's heuristics (see [`UsageType`]).
    pub usage_type: UsageType,
}

impl H264EncoderConfig {
    /// Create a new encoder configuration with the given dimensions.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            bitrate_bps: 0,
            max_frame_rate: 30.0,
            qp: 26,
            scene_change_detect: true,
            keyframe_interval: 0,
            num_threads: 0,
            sps_pps_strategy: SpsPpsStrategy::default(),
            rate_control: RateControlMode::default(),
            max_slice_len: None,
            skip_frames: None,
            profile: None,
            complexity: Complexity::default(),
            usage_type: UsageType::default(),
        }
    }

    /// Set the target bitrate in bits per second.
    pub fn bitrate(mut self, bps: u32) -> Self {
        self.bitrate_bps = bps;
        self
    }

    /// Set the maximum frame rate.
    pub fn frame_rate(mut self, fps: f32) -> Self {
        self.max_frame_rate = fps;
        self
    }

    /// Set the quantization parameter (0-51).
    pub fn qp(mut self, qp: u8) -> Self {
        self.qp = qp.min(51);
        self
    }

    /// Set the keyframe interval.
    pub fn keyframe_interval(mut self, interval: u32) -> Self {
        self.keyframe_interval = interval;
        self
    }

    /// Set the number of encoding threads.
    pub fn threads(mut self, threads: u32) -> Self {
        self.num_threads = threads;
        self
    }

    /// Set the SPS/PPS emission strategy (see [`SpsPpsStrategy`]).
    pub fn sps_pps_strategy(mut self, strategy: SpsPpsStrategy) -> Self {
        self.sps_pps_strategy = strategy;
        self
    }

    /// Set how strictly the bitrate target is honoured (see [`RateControlMode`]).
    ///
    /// Pair [`RateControlMode::Bitrate`] with [`bitrate`](Self::bitrate) when
    /// the link is the constraint.
    pub fn rate_control(mut self, mode: RateControlMode) -> Self {
        self.rate_control = mode;
        self
    }

    /// Cap the size of each emitted NAL unit, in bytes (see
    /// [`max_slice_len`](Self::max_slice_len)).
    pub fn max_slice_len(mut self, bytes: u32) -> Self {
        self.max_slice_len = Some(bytes);
        self
    }

    /// Allow or forbid rate control dropping frames to hold the bitrate target
    /// (see [`skip_frames`](Self::skip_frames)).
    pub fn skip_frames(mut self, skip: bool) -> Self {
        self.skip_frames = Some(skip);
        self
    }

    /// Set the H.264 profile.
    pub fn profile(mut self, profile: Profile) -> Self {
        self.profile = Some(profile);
        self
    }

    /// Set the encoder complexity (CPU per frame).
    pub fn complexity(mut self, complexity: Complexity) -> Self {
        self.complexity = complexity;
        self
    }

    /// Set the content type (camera vs screen, real-time vs not).
    pub fn usage_type(mut self, usage: UsageType) -> Self {
        self.usage_type = usage;
        self
    }

    /// Create a configuration for low-latency streaming.
    pub fn low_latency(width: u32, height: u32) -> Self {
        Self::new(width, height)
            .frame_rate(30.0)
            .keyframe_interval(30) // Keyframe every second at 30fps
            .qp(28) // Slightly lower quality for speed
    }

    /// Create a configuration for high-quality encoding.
    pub fn high_quality(width: u32, height: u32) -> Self {
        Self::new(width, height)
            .frame_rate(30.0)
            .keyframe_interval(120) // Keyframe every 4 seconds
            .qp(20) // Higher quality
    }
}

impl Default for H264EncoderConfig {
    fn default() -> Self {
        Self::new(1920, 1080)
    }
}

// ============================================================================
// Encoder
// ============================================================================

/// H.264 encoder using OpenH264.
///
/// Encodes YUV420 frames to H.264 NAL units.
pub struct H264Encoder {
    encoder: Encoder,
    config: H264EncoderConfig,
    frame_count: u64,
    bytes_encoded: u64,
    /// Runtime control: keyframe requests plus bitrate/GOP/QP changes (shared
    /// with [`Self::control_handle`] and [`Self::keyframe_handle`]).
    control: super::EncoderControl,
    /// The control generation last applied to the encoder.
    applied_generation: u64,
    /// Arena for output buffer allocation.
    arena: SharedArena,
}

/// Build an OpenH264 encoder from our config.
///
/// Shared by [`H264Encoder::new`] and the runtime reconfigure path, so
/// start-time and live settings can never drift apart.
fn build_encoder(config: &H264EncoderConfig) -> Result<Encoder> {
    let mut encoder_config = EncoderConfig::new();

    if config.bitrate_bps > 0 {
        encoder_config = encoder_config.bitrate(BitRate::from_bps(config.bitrate_bps));
    }

    // Frame skipping. Default: on with a bitrate target (OpenH264's own
    // behaviour), off in quality mode — where OpenH264 would otherwise apply a
    // 120 kbps default target and silently drop most frames of real content.
    // Either way this is a knob worth knowing about: when skipping is on, a
    // tight target makes the encoder emit *nothing* for some input frames.
    let skip_frames = config.skip_frames.unwrap_or(config.bitrate_bps > 0);
    encoder_config = encoder_config.skip_frames(skip_frames);

    let qp = config.qp.min(51);
    encoder_config = encoder_config
        .max_frame_rate(FrameRate::from_hz(config.max_frame_rate))
        .scene_change_detect(config.scene_change_detect)
        .num_threads(config.num_threads as u16)
        .qp(QpRange::new(qp.saturating_sub(4), (qp + 4).min(51)))
        .intra_frame_period(IntraFramePeriod::from_num_frames(config.keyframe_interval))
        .sps_pps_strategy(config.sps_pps_strategy.to_openh264())
        .rate_control_mode(config.rate_control.to_openh264())
        .complexity(config.complexity.to_openh264())
        .usage_type(config.usage_type.to_openh264());

    if let Some(max_slice_len) = config.max_slice_len {
        encoder_config = encoder_config.max_slice_len(max_slice_len);
    }
    if let Some(profile) = config.profile {
        encoder_config = encoder_config.profile(profile.to_openh264());
    }

    let api = OpenH264API::from_source();
    Encoder::with_api_config(api, encoder_config)
        .map_err(|e| Error::Config(format!("Failed to create H.264 encoder: {:?}", e)))
}

impl H264Encoder {
    /// Create a new H.264 encoder with the given configuration.
    pub fn new(config: H264EncoderConfig) -> Result<Self> {
        let encoder = build_encoder(&config)?;

        // Create arena for encoded output buffers (typically < 1MB per frame)
        // Use 64 slots to handle buffering when downstream is slower than encoding
        let arena = SharedArena::new(1024 * 1024, 64)
            .map_err(|e| Error::Config(format!("Failed to create arena: {}", e)))?;

        Ok(Self {
            encoder,
            config,
            frame_count: 0,
            bytes_encoded: 0,
            control: super::EncoderControl::new(),
            applied_generation: 0,
            arena,
        })
    }

    /// Rebuild the OpenH264 encoder for the current config, keeping the arena.
    ///
    /// OpenH264 exposes no bitrate setter, so a live change means a new
    /// encoder. That is cheap (a few ms) and forces a fresh SPS/PPS + IDR —
    /// which is what you want on a rate step anyway, since decoders need the
    /// new parameter sets and a mid-GOP rate change looks bad. The **arena is
    /// deliberately reused**: allocating a fresh 64 MiB of slots on every
    /// bitrate step would be a silent memory leak in a long-running sensor.
    fn rebuild_encoder(&mut self) -> Result<()> {
        self.encoder = build_encoder(&self.config)?;
        // A new encoder starts a new sequence; make its first frame an IDR so
        // decoders pick up the new parameter sets immediately.
        self.encoder.force_intra_frame();
        Ok(())
    }

    /// Apply any parameter changes made through [`Self::control_handle`].
    fn apply_pending_control(&mut self) -> Result<()> {
        let Some(params) = self.control.poll(&mut self.applied_generation) else {
            return Ok(());
        };

        if let Some(bps) = params.bitrate_bps {
            self.config.bitrate_bps = bps;
        }
        if let Some(frames) = params.keyframe_interval {
            self.config.keyframe_interval = frames;
        }
        if let Some(qp) = params.qp {
            self.config.qp = qp.min(51);
        }

        tracing::info!(
            "H264Encoder: reconfigured — bitrate {} bps, keyframe interval {}, qp {}",
            self.config.bitrate_bps,
            self.config.keyframe_interval,
            self.config.qp
        );
        self.rebuild_encoder()
    }

    /// Get a cloneable handle for changing bitrate, keyframe interval and QP on
    /// a running pipeline.
    ///
    /// Clone this *before* the pipeline starts (elements are moved into their
    /// executor tasks at start). Each change rebuilds the underlying OpenH264
    /// encoder and emits an IDR, so rate-limit changes to roughly one per
    /// second rather than, say, once per slider pixel.
    pub fn control_handle(&self) -> super::EncoderControl {
        self.control.clone()
    }

    /// Encode a YUV420 frame at the configured resolution.
    ///
    /// The input data must be in YUV420 planar format:
    /// - Y plane: width * height bytes
    /// - U plane: (width/2) * (height/2) bytes
    /// - V plane: (width/2) * (height/2) bytes
    ///
    /// Returns the encoded H.264 bitstream (NAL units).
    pub fn encode_yuv420(&mut self, yuv_data: &[u8]) -> Result<Vec<u8>> {
        let (width, height) = (self.config.width, self.config.height);
        self.encode_yuv420_at(yuv_data, width, height)
    }

    /// Encode a YUV420 frame of the given size, changing resolution if needed.
    ///
    /// A size different from the last frame's re-initialises the encoder and
    /// starts a fresh IDR, so the switch is a clean decoder entry point. The
    /// configured resolution follows the frames — it is a seed, not a contract.
    pub fn encode_yuv420_at(
        &mut self,
        yuv_data: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>> {
        check_resolution(width, height)?;

        let expected_size = width as usize * height as usize * 3 / 2;
        if yuv_data.len() < expected_size {
            return Err(Error::Config(format!(
                "YUV data too small: expected {} bytes for {}x{}, got {}",
                expected_size,
                width,
                height,
                yuv_data.len()
            )));
        }

        if (width, height) != (self.config.width, self.config.height) {
            tracing::info!(
                "H264Encoder: resolution {}x{} -> {}x{} (re-init, IDR)",
                self.config.width,
                self.config.height,
                width,
                height
            );
            self.config.width = width;
            self.config.height = height;
        }

        let yuv = YuvFrame {
            data: yuv_data,
            width: width as usize,
            height: height as usize,
        };

        // OpenH264 notices the dimension change itself: it re-initialises via
        // SetOption(SVC_ENCODE_PARAM_EXT) and forces an IDR. We only have to
        // stop pinning the size and let the frame through.
        let bitstream = self
            .encoder
            .encode(&yuv)
            .map_err(|e| Error::Config(format!("H.264 encode failed: {:?}", e)))?;

        let encoded = bitstream.to_vec();
        self.frame_count += 1;
        self.bytes_encoded += encoded.len() as u64;

        Ok(encoded)
    }

    /// Force the next frame to be a keyframe (IDR frame).
    pub fn force_keyframe(&mut self) {
        self.encoder.force_intra_frame();
    }

    /// Get a cloneable handle for requesting keyframes at runtime.
    ///
    /// Clone this *before* the pipeline starts (elements are moved into
    /// their executor tasks at start); calling
    /// [`request()`](super::KeyframeHandle::request) on it makes the next
    /// frame processed by this encoder an IDR, flagged
    /// [`BufferFlags::SYNC_POINT`](crate::metadata::BufferFlags::SYNC_POINT).
    pub fn keyframe_handle(&self) -> super::KeyframeHandle {
        self.control.keyframe_handle()
    }

    /// Get the number of frames encoded.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the total bytes encoded.
    pub fn bytes_encoded(&self) -> u64 {
        self.bytes_encoded
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &H264EncoderConfig {
        &self.config
    }
}

impl std::fmt::Debug for H264Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H264Encoder")
            .field("config", &self.config)
            .field("frame_count", &self.frame_count)
            .field("bytes_encoded", &self.bytes_encoded)
            .finish()
    }
}

/// OpenH264's hard ceiling: it supports up to level 5.2, i.e. 3840x2160 in
/// either orientation, and errors out beyond that.
fn check_resolution(width: u32, height: u32) -> Result<()> {
    if width == 0 || height == 0 {
        return Err(Error::Config(format!(
            "H.264 resolution must be non-zero, got {width}x{height}"
        )));
    }
    let (long, short) = (width.max(height), width.min(height));
    if long > 3840 || short > 2160 {
        return Err(Error::Config(format!(
            "H.264 resolution {width}x{height} exceeds OpenH264's 3840x2160 limit"
        )));
    }
    Ok(())
}

/// Returns true if the Annex-B bitstream contains an IDR NAL unit (type 5).
fn contains_idr(data: &[u8]) -> bool {
    let mut i = 0;
    while i + 3 < data.len() {
        let offset = if data[i..].starts_with(&[0, 0, 0, 1]) {
            4
        } else if data[i..].starts_with(&[0, 0, 1]) {
            3
        } else {
            i += 1;
            continue;
        };
        if i + offset < data.len() && data[i + offset] & 0x1F == 5 {
            return true;
        }
        i += offset;
    }
    false
}

/// Element trait implementation for H264Encoder.
impl Element for H264Encoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Runtime reconfiguration (bitrate / keyframe interval / QP). Costs one
        // relaxed load when nothing changed; rebuilds the encoder when it did.
        self.apply_pending_control()?;

        // Runtime keyframe requests: from the shared handle or stamped
        // in-band on the buffer's metadata.
        let requested = self.control.take_keyframe()
            || buffer
                .metadata()
                .get::<bool>(super::KEYFRAME_REQUEST)
                .copied()
                .unwrap_or(false);
        if requested {
            self.encoder.force_intra_frame();
        }

        // The frame's own metadata decides the resolution: an upstream scaler
        // can retarget mid-stream, and the encoder follows it.
        let (width, height) = buffer
            .metadata()
            .video_dims()
            .unwrap_or((self.config.width, self.config.height));

        let input_data = buffer.as_bytes();
        let encoded = self.encode_yuv420_at(input_data, width, height)?;

        if encoded.is_empty() {
            return Ok(None);
        }
        let is_keyframe = contains_idr(&encoded);

        // Reclaim any released slots before acquiring
        self.arena.reclaim();

        // Acquire slot from arena and copy encoded data
        let mut slot = self
            .arena
            .acquire()
            .ok_or_else(|| Error::Config("Failed to acquire buffer slot".to_string()))?;

        // Copy encoded data to slot
        slot.data_mut()[..encoded.len()].copy_from_slice(&encoded);

        let handle = crate::buffer::MemoryHandle::with_len(slot, encoded.len());
        // Preserve input buffer's PTS for proper timing, update sequence number
        let mut metadata = buffer.metadata().clone();
        metadata.sequence = self.frame_count - 1;
        // SYNC_POINT must reflect the ENCODED stream, not the input: raw
        // sources flag every uncompressed frame as a sync point, and a
        // delta AU wrongly advertised as a keyframe sends fresh decoders
        // into an unrecoverable dsNoParamSets loop (#22).
        if is_keyframe {
            metadata.flags |= crate::metadata::BufferFlags::SYNC_POINT;
        } else {
            metadata.flags = metadata
                .flags
                .remove(crate::metadata::BufferFlags::SYNC_POINT);
        }
        Ok(Some(Buffer::new(handle, metadata)))
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Accept I420 (YUV420) video of any size
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, PixelFormat,
            VideoFormatCaps,
        };

        let format = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Fixed(PixelFormat::I420),
            ..VideoFormatCaps::any()
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(format),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Output is H.264 encoded video
        use crate::format::{
            ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, VideoCodec,
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::Video(VideoCodec::H264),
            MemoryCaps::cpu_only(),
        )])
    }
}

/// VideoEncoder trait implementation for H264Encoder.
///
/// This allows H264Encoder to be used with `EncoderElement` for pipeline integration.
impl super::traits::VideoEncoder for H264Encoder {
    type Packet = Vec<u8>;

    fn encode(&mut self, frame: &super::common::VideoFrame) -> Result<Vec<Self::Packet>> {
        // A frame of a different size is not an error: OpenH264 re-initialises
        // and emits a fresh IDR, which is exactly how a live resolution change
        // is supposed to look.
        let encoded = self.encode_yuv420_at(&frame.data, frame.width, frame.height)?;

        if encoded.is_empty() {
            Ok(Vec::new())
        } else {
            Ok(vec![encoded])
        }
    }

    fn flush(&mut self) -> Result<Vec<Self::Packet>> {
        // OpenH264 doesn't buffer frames, so flush is a no-op
        Ok(Vec::new())
    }

    fn codec_data(&self) -> Option<Vec<u8>> {
        // H.264 codec data (SPS/PPS) would be extracted from the first encoded frame
        // For now, return None - the first keyframe contains the headers inline
        None
    }

    fn force_keyframe(&mut self) {
        self.encoder.force_intra_frame();
    }

    fn set_bitrate(&mut self, bps: u32) -> Result<()> {
        if self.config.bitrate_bps == bps {
            return Ok(());
        }
        // 0 is not "unset" here: it selects quality mode, where rate control is
        // driven by the QP band instead of a target (see build_encoder).
        self.config.bitrate_bps = bps;
        self.rebuild_encoder()
    }

    fn set_keyframe_interval(&mut self, frames: u32) -> Result<()> {
        if self.config.keyframe_interval == frames {
            return Ok(());
        }
        self.config.keyframe_interval = frames;
        self.rebuild_encoder()
    }

    fn set_qp(&mut self, qp: u8) -> Result<()> {
        let qp = qp.min(51);
        if self.config.qp == qp {
            return Ok(());
        }
        self.config.qp = qp;
        self.rebuild_encoder()
    }
}

// ============================================================================
// Decoder
// ============================================================================

/// H.264 decoder using OpenH264.
///
/// Decodes H.264 NAL units to YUV420 frames.
pub struct H264Decoder {
    decoder: Decoder,
    frame_count: u64,
    bytes_decoded: u64,
    /// Arena for output buffer allocation.
    arena: SharedArena,
}

impl H264Decoder {
    /// Create a new H.264 decoder.
    pub fn new() -> Result<Self> {
        let decoder = Decoder::new()
            .map_err(|e| Error::Config(format!("Failed to create H.264 decoder: {:?}", e)))?;

        // Create arena for decoded YUV frames (1080p YUV420 = ~3MB per frame)
        let arena = SharedArena::new(4 * 1024 * 1024, 8)
            .map_err(|e| Error::Config(format!("Failed to create arena: {}", e)))?;

        Ok(Self {
            decoder,
            frame_count: 0,
            bytes_decoded: 0,
            arena,
        })
    }

    /// Decode H.264 NAL units.
    ///
    /// Returns the decoded YUV frame if a complete frame is available,
    /// or `None` if more data is needed.
    pub fn decode(&mut self, nal_data: &[u8]) -> Result<Option<DecodedFrame>> {
        self.bytes_decoded += nal_data.len() as u64;

        let result = self
            .decoder
            .decode(nal_data)
            .map_err(|e| Error::Config(format!("H.264 decode failed: {:?}", e)))?;

        match result {
            Some(yuv) => {
                self.frame_count += 1;
                Ok(Some(DecodedFrame::from_decoded_yuv(yuv)))
            }
            None => Ok(None),
        }
    }

    /// Flush the decoder and retrieve any remaining frames.
    pub fn flush(&mut self) -> Result<Vec<DecodedFrame>> {
        let frames = self
            .decoder
            .flush_remaining()
            .map_err(|e| Error::Config(format!("H.264 flush failed: {:?}", e)))?;

        self.frame_count += frames.len() as u64;
        Ok(frames
            .into_iter()
            .map(DecodedFrame::from_decoded_yuv)
            .collect())
    }

    /// Get the number of frames decoded.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the total bytes decoded.
    pub fn bytes_decoded(&self) -> u64 {
        self.bytes_decoded
    }
}

impl std::fmt::Debug for H264Decoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H264Decoder")
            .field("frame_count", &self.frame_count)
            .field("bytes_decoded", &self.bytes_decoded)
            .finish()
    }
}

/// Element trait implementation for H264Decoder.
impl Element for H264Decoder {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input_data = buffer.as_bytes();

        match self.decode(input_data)? {
            Some(frame) => {
                let yuv_data = frame.to_yuv420_planar();

                // Reclaim released slots and acquire new one
                self.arena.reclaim();
                let mut slot = self
                    .arena
                    .acquire()
                    .ok_or_else(|| Error::Config("Failed to acquire buffer slot".to_string()))?;

                // Copy YUV data to slot
                slot.data_mut()[..yuv_data.len()].copy_from_slice(&yuv_data);

                let handle = crate::buffer::MemoryHandle::with_len(slot, yuv_data.len());
                let mut metadata = Metadata::from_sequence(self.frame_count - 1);
                metadata.set("width", frame.width() as u64);
                metadata.set("height", frame.height() as u64);
                Ok(Some(Buffer::new(handle, metadata)))
            }
            None => Ok(None),
        }
    }
}

// ============================================================================
// Decoded Frame
// ============================================================================

/// A decoded YUV frame from the H.264 decoder.
#[derive(Debug)]
pub struct DecodedFrame {
    /// Y plane data.
    y_data: Vec<u8>,
    /// U plane data.
    u_data: Vec<u8>,
    /// V plane data.
    v_data: Vec<u8>,
    /// Frame width.
    width: usize,
    /// Frame height.
    height: usize,
    /// Y plane stride.
    y_stride: usize,
    /// U plane stride.
    u_stride: usize,
    /// V plane stride.
    v_stride: usize,
}

impl DecodedFrame {
    fn from_decoded_yuv(yuv: DecodedYUV) -> Self {
        let (width, height) = yuv.dimensions();
        let y_data = yuv.y().to_vec();
        let u_data = yuv.u().to_vec();
        let v_data = yuv.v().to_vec();
        let (y_stride, u_stride, v_stride) = yuv.strides();

        Self {
            y_data,
            u_data,
            v_data,
            width,
            height,
            y_stride,
            u_stride,
            v_stride,
        }
    }

    /// Get the frame width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Get the frame height.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get the Y plane data.
    pub fn y_plane(&self) -> &[u8] {
        &self.y_data
    }

    /// Get the U plane data.
    pub fn u_plane(&self) -> &[u8] {
        &self.u_data
    }

    /// Get the V plane data.
    pub fn v_plane(&self) -> &[u8] {
        &self.v_data
    }

    /// Get the strides for each plane (Y, U, V).
    pub fn strides(&self) -> (usize, usize, usize) {
        (self.y_stride, self.u_stride, self.v_stride)
    }

    /// Convert to contiguous YUV420 planar format.
    ///
    /// Returns a Vec with Y, U, V planes concatenated without padding.
    pub fn to_yuv420_planar(&self) -> Vec<u8> {
        let y_size = self.width * self.height;
        let uv_size = (self.width / 2) * (self.height / 2);
        let total_size = y_size + uv_size * 2;

        let mut output = Vec::with_capacity(total_size);

        // Copy Y plane (removing stride padding if any)
        for y in 0..self.height {
            let start = y * self.y_stride;
            let end = start + self.width;
            if end <= self.y_data.len() {
                output.extend_from_slice(&self.y_data[start..end]);
            }
        }

        // Copy U plane
        let uv_height = self.height / 2;
        let uv_width = self.width / 2;
        for y in 0..uv_height {
            let start = y * self.u_stride;
            let end = start + uv_width;
            if end <= self.u_data.len() {
                output.extend_from_slice(&self.u_data[start..end]);
            }
        }

        // Copy V plane
        for y in 0..uv_height {
            let start = y * self.v_stride;
            let end = start + uv_width;
            if end <= self.v_data.len() {
                output.extend_from_slice(&self.v_data[start..end]);
            }
        }

        output
    }
}

// ============================================================================
// Helper Types
// ============================================================================

/// Internal YUV frame wrapper for OpenH264.
struct YuvFrame<'a> {
    data: &'a [u8],
    width: usize,
    height: usize,
}

impl<'a> YUVSource for YuvFrame<'a> {
    fn dimensions(&self) -> (usize, usize) {
        (self.width, self.height)
    }

    fn strides(&self) -> (usize, usize, usize) {
        (self.width, self.width / 2, self.width / 2)
    }

    fn y(&self) -> &[u8] {
        &self.data[..self.width * self.height]
    }

    fn u(&self) -> &[u8] {
        let y_size = self.width * self.height;
        let u_size = (self.width / 2) * (self.height / 2);
        &self.data[y_size..y_size + u_size]
    }

    fn v(&self) -> &[u8] {
        let y_size = self.width * self.height;
        let u_size = (self.width / 2) * (self.height / 2);
        &self.data[y_size + u_size..]
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_config_default() {
        let config = H264EncoderConfig::default();
        assert_eq!(config.width, 1920);
        assert_eq!(config.height, 1080);
        assert_eq!(config.qp, 26);
    }

    #[test]
    fn test_encoder_config_builder() {
        let config = H264EncoderConfig::new(640, 480)
            .bitrate(1_000_000)
            .frame_rate(25.0)
            .qp(24)
            .keyframe_interval(60);

        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.bitrate_bps, 1_000_000);
        assert_eq!(config.max_frame_rate, 25.0);
        assert_eq!(config.qp, 24);
        assert_eq!(config.keyframe_interval, 60);
    }

    #[test]
    fn test_encoder_config_low_latency() {
        let config = H264EncoderConfig::low_latency(1280, 720);
        assert_eq!(config.width, 1280);
        assert_eq!(config.height, 720);
        assert_eq!(config.keyframe_interval, 30);
    }

    #[test]
    fn test_encoder_config_high_quality() {
        let config = H264EncoderConfig::high_quality(1920, 1080);
        assert_eq!(config.qp, 20);
        assert_eq!(config.keyframe_interval, 120);
    }

    /// Count IDR NAL units (type 5) in an Annex-B bitstream.
    fn count_idr_nals(data: &[u8]) -> usize {
        let mut count = 0;
        let mut i = 0;
        while i + 3 < data.len() {
            // 3- or 4-byte start code
            let (start, offset) = if data[i..].starts_with(&[0, 0, 0, 1]) {
                (true, 4)
            } else if data[i..].starts_with(&[0, 0, 1]) {
                (true, 3)
            } else {
                (false, 1)
            };
            if start && i + offset < data.len() {
                if data[i + offset] & 0x1F == 5 {
                    count += 1;
                }
                i += offset;
            } else {
                i += 1;
            }
        }
        count
    }

    /// Wrap raw I420 bytes in a Buffer that declares its geometry, as a real
    /// upstream element would.
    fn yuv_buffer(data: &[u8], width: u32, height: u32) -> Buffer {
        use crate::buffer::MemoryHandle;
        use crate::format::PixelFormat;

        let arena = SharedArena::new(data.len(), 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..data.len()].copy_from_slice(data);

        let mut metadata = Metadata::new();
        metadata.set_video_dims(width, height, PixelFormat::I420);
        Buffer::new(MemoryHandle::with_len(slot, data.len()), metadata)
    }

    /// A deterministic frame with spatial detail so QP visibly affects size.
    fn detailed_frame(width: usize, height: usize, seed: u8) -> Vec<u8> {
        let mut data = vec![128u8; width * height * 3 / 2];
        for y in 0..height {
            for x in 0..width {
                data[y * width + x] = ((x * 7 + y * 13) as u8).wrapping_add(seed);
            }
        }
        data
    }

    /// Sizes of every Annex-B NAL payload (excluding start codes).
    fn nal_sizes(data: &[u8]) -> Vec<usize> {
        let mut starts = Vec::new();
        let mut i = 0;
        while i + 3 < data.len() {
            if data[i..].starts_with(&[0, 0, 0, 1]) {
                starts.push(i + 4);
                i += 4;
            } else if data[i..].starts_with(&[0, 0, 1]) {
                starts.push(i + 3);
                i += 3;
            } else {
                i += 1;
            }
        }
        starts
            .iter()
            .enumerate()
            .map(|(n, &start)| {
                let end = starts.get(n + 1).map(|&s| s - 4).unwrap_or(data.len());
                end.saturating_sub(start)
            })
            .collect()
    }

    /// Rate control is the knob that decides whether `bitrate_bps` is a budget
    /// or a suggestion. Until now it was never set at all, so every stream ran
    /// on OpenH264's default (Quality) no matter what bitrate was asked for.
    #[test]
    fn bitrate_mode_tracks_the_target_more_tightly_than_no_rate_control() {
        let encode_all = |mode: RateControlMode| -> u64 {
            let mut config = H264EncoderConfig::new(320, 240)
                .bitrate(150_000)
                .rate_control(mode);
            config.scene_change_detect = false;
            let mut encoder = H264Encoder::new(config).unwrap();
            for i in 0..20 {
                encoder
                    .encode_yuv420(&detailed_frame(320, 240, i as u8))
                    .unwrap();
            }
            encoder.bytes_encoded()
        };

        let budgeted = encode_all(RateControlMode::Bitrate);
        let unconstrained = encode_all(RateControlMode::Off);

        assert!(
            budgeted < unconstrained,
            "Bitrate mode must honour a 150 kbps target more than no rate control at all \
             (got {budgeted} vs {unconstrained} bytes)"
        );
    }

    /// MTU-sized NALs: without this the RTP payloader fragments every slice.
    #[test]
    fn max_slice_len_caps_nal_size() {
        const CAP: u32 = 1200;
        let mut config = H264EncoderConfig::new(320, 240).max_slice_len(CAP);
        config.scene_change_detect = false;
        let mut encoder = H264Encoder::new(config).unwrap();

        let mut seen_any = false;
        for i in 0..5 {
            let encoded = encoder
                .encode_yuv420(&detailed_frame(320, 240, i as u8))
                .unwrap();
            for size in nal_sizes(&encoded) {
                seen_any = true;
                assert!(
                    size <= CAP as usize,
                    "NAL of {size} bytes exceeds the {CAP}-byte cap"
                );
            }
        }
        assert!(seen_any, "expected some NAL units");
    }

    #[test]
    fn profile_is_reflected_in_the_sps() {
        // SPS (NAL type 7) carries profile_idc in its first payload byte:
        // 66 = Baseline, 77 = Main, 100 = High.
        let profile_idc = |profile: Profile| -> u8 {
            let mut encoder =
                H264Encoder::new(H264EncoderConfig::new(320, 240).profile(profile)).unwrap();
            let encoded = encoder.encode_yuv420(&detailed_frame(320, 240, 0)).unwrap();

            let mut i = 0;
            while i + 4 < encoded.len() {
                if encoded[i..].starts_with(&[0, 0, 0, 1]) && encoded[i + 4] & 0x1F == 7 {
                    return encoded[i + 5];
                }
                i += 1;
            }
            panic!("no SPS in the stream");
        };

        assert_eq!(profile_idc(Profile::Baseline), 66);
        assert_eq!(profile_idc(Profile::High), 100);
    }

    /// Rate control may drop frames to hold its target — an encoder emitting
    /// *nothing* for an input frame is legal and surprising. Pin both sides of
    /// the knob.
    #[test]
    fn skip_frames_governs_whether_every_input_frame_is_encoded() {
        let count_encoded = |skip: bool| -> usize {
            let mut config = H264EncoderConfig::new(320, 240)
                .bitrate(20_000) // a target far too tight for this content
                .rate_control(RateControlMode::Bitrate)
                .skip_frames(skip);
            config.scene_change_detect = false;
            let mut encoder = H264Encoder::new(config).unwrap();

            (0..20)
                .filter(|i| {
                    !encoder
                        .encode_yuv420(&detailed_frame(320, 240, *i as u8))
                        .unwrap()
                        .is_empty()
                })
                .count()
        };

        assert_eq!(
            count_encoded(false),
            20,
            "with skipping off, every input frame must produce output"
        );
        assert!(
            count_encoded(true) < 20,
            "with skipping on, a tight bitrate target drops frames — the \
             behaviour a caller needs to be able to turn off"
        );
    }

    #[test]
    fn new_knobs_default_to_previous_behaviour() {
        // Guards against a silent quality/bitrate regression for existing users.
        let config = H264EncoderConfig::new(320, 240);
        assert_eq!(config.rate_control, RateControlMode::Quality);
        assert_eq!(config.max_slice_len, None);
        assert_eq!(config.skip_frames, None);
        assert_eq!(config.profile, None);
        assert_eq!(config.complexity, Complexity::Medium);
        assert_eq!(config.usage_type, UsageType::CameraRealtime);
    }

    #[test]
    fn config_knobs_survive_a_live_reconfigure() {
        use crate::element::Element;

        // The rebuild path must re-apply every knob, not just the three the
        // control handle carries.
        let config = H264EncoderConfig::new(320, 240)
            .bitrate(2_000_000)
            .rate_control(RateControlMode::Bitrate)
            .max_slice_len(1200)
            .profile(Profile::Baseline);
        let mut encoder = H264Encoder::new(config).unwrap();
        let control = encoder.control_handle();

        control.set_bitrate(400_000);
        let out = encoder
            .process(yuv_buffer(&detailed_frame(320, 240, 0), 320, 240))
            .unwrap()
            .unwrap();

        assert_eq!(encoder.config().rate_control, RateControlMode::Bitrate);
        assert_eq!(encoder.config().max_slice_len, Some(1200));
        for size in nal_sizes(out.as_bytes()) {
            assert!(size <= 1200, "slice cap lost across the rebuild: {size}");
        }
    }

    /// The point of the whole exercise: dropping the bitrate on a live encoder
    /// must actually shrink the stream.
    #[test]
    fn live_bitrate_change_shrinks_the_stream() {
        use crate::element::Element;

        let mut config = H264EncoderConfig::new(320, 240).bitrate(4_000_000);
        config.scene_change_detect = false;
        let mut encoder = H264Encoder::new(config).unwrap();
        let control = encoder.control_handle();

        // Encode at 4 Mbps, then at 200 kbps, and compare steady-state sizes.
        // The frame right after the change is an IDR (new parameter sets), so
        // it is excluded from the comparison — it is expected to be large.
        let encode_run = |encoder: &mut H264Encoder, seed_base: u8| -> Vec<usize> {
            (0..15)
                .map(|i| {
                    let frame = detailed_frame(320, 240, seed_base.wrapping_add(i));
                    encoder.process(yuv_buffer(&frame, 320, 240)).unwrap();
                    encoder.bytes_encoded()
                })
                .collect::<Vec<_>>()
                .windows(2)
                .map(|w| (w[1] - w[0]) as usize)
                .collect()
        };

        let fast = encode_run(&mut encoder, 0);
        control.set_bitrate(200_000);
        let slow = encode_run(&mut encoder, 100);

        // Skip the first few frames of each run (IDR + rate-control settling).
        let mean = |sizes: &[usize]| sizes[4..].iter().sum::<usize>() / sizes[4..].len();
        let (fast_mean, slow_mean) = (mean(&fast), mean(&slow));

        assert!(
            slow_mean * 2 < fast_mean,
            "200 kbps must produce materially smaller frames than 4 Mbps \
             (got {slow_mean} vs {fast_mean} bytes/frame)"
        );
    }

    #[test]
    fn live_bitrate_change_emits_an_idr() {
        use crate::element::Element;

        let mut config = H264EncoderConfig::new(320, 240).bitrate(4_000_000);
        config.scene_change_detect = false;
        let mut encoder = H264Encoder::new(config).unwrap();
        let control = encoder.control_handle();

        for i in 0..4 {
            encoder
                .process(yuv_buffer(&detailed_frame(320, 240, i), 320, 240))
                .unwrap();
        }

        control.set_bitrate(400_000);
        let out = encoder
            .process(yuv_buffer(&detailed_frame(320, 240, 9), 320, 240))
            .unwrap()
            .expect("a frame");

        assert_eq!(
            count_idr_nals(out.as_bytes()),
            1,
            "a rebuilt encoder must lead with an IDR so decoders get the new SPS/PPS"
        );
        assert!(
            out.metadata()
                .flags
                .contains(crate::metadata::BufferFlags::SYNC_POINT),
            "and the buffer must be flagged as a sync point"
        );
    }

    #[test]
    fn live_keyframe_interval_change_changes_idr_cadence() {
        use crate::element::Element;

        let mut config = H264EncoderConfig::new(320, 240).keyframe_interval(100);
        config.scene_change_detect = false;
        let mut encoder = H264Encoder::new(config).unwrap();
        let control = encoder.control_handle();

        control.set_keyframe_interval(5);
        let mut idrs = 0;
        for i in 0..15 {
            let out = encoder
                .process(yuv_buffer(&detailed_frame(320, 240, i), 320, 240))
                .unwrap()
                .unwrap();
            idrs += count_idr_nals(out.as_bytes());
        }

        // Frame 0 (rebuild IDR) plus one every 5 frames.
        assert!(
            idrs >= 3,
            "an interval of 5 over 15 frames should yield several IDRs, got {idrs}"
        );
    }

    #[test]
    fn unchanged_parameters_do_not_rebuild_the_encoder() {
        use super::super::traits::VideoEncoder;

        let mut encoder =
            H264Encoder::new(H264EncoderConfig::new(320, 240).bitrate(1_000_000)).unwrap();

        // Setting the value it already has must not force a spurious IDR.
        encoder.set_bitrate(1_000_000).unwrap();
        let encoded = encoder.encode_yuv420(&detailed_frame(320, 240, 1)).unwrap();
        let first_idrs = count_idr_nals(&encoded);

        encoder.set_bitrate(1_000_000).unwrap();
        let encoded = encoder.encode_yuv420(&detailed_frame(320, 240, 2)).unwrap();
        assert_eq!(
            count_idr_nals(&encoded),
            0,
            "a no-op set_bitrate must not rebuild the encoder (first frame had {first_idrs} IDRs)"
        );
    }

    /// The switch itself: a differently-sized frame must encode, must produce a
    /// decodable stream at the new size, and must lead with an IDR so a decoder
    /// (and any viewer joining after it) can follow the change.
    #[test]
    fn resolution_change_mid_stream_reinits_and_emits_an_idr() {
        let mut config = H264EncoderConfig::new(320, 240);
        config.scene_change_detect = false; // no incidental IDRs
        let mut encoder = H264Encoder::new(config).unwrap();
        let mut decoder = H264Decoder::new().unwrap();

        let mut sizes = Vec::new();
        for i in 0..6 {
            // Halve the resolution half-way through.
            let (w, h) = if i < 3 { (320, 240) } else { (160, 120) };
            let encoded = encoder
                .encode_yuv420_at(&detailed_frame(w, h, i as u8), w as u32, h as u32)
                .unwrap();

            if i == 3 {
                assert_eq!(
                    count_idr_nals(&encoded),
                    1,
                    "the first frame at the new size must be an IDR"
                );
            }

            if let Some(frame) = decoder.decode(&encoded).unwrap() {
                sizes.push((frame.width(), frame.height()));
            }
        }

        assert!(
            sizes.contains(&(320, 240)) && sizes.contains(&(160, 120)),
            "decoded frames must change size mid-stream, got {sizes:?}"
        );
    }

    /// The encoder follows the frames rather than pinning its constructor size.
    #[test]
    fn config_dimensions_track_the_encoded_frames() {
        let mut encoder = H264Encoder::new(H264EncoderConfig::new(320, 240)).unwrap();
        encoder
            .encode_yuv420_at(&detailed_frame(160, 120, 0), 160, 120)
            .unwrap();

        assert_eq!(
            (encoder.config().width, encoder.config().height),
            (160, 120)
        );
    }

    /// VideoEncoder::encode used to hard-error on any size mismatch.
    #[test]
    fn video_encoder_trait_accepts_a_changed_frame_size() {
        use super::super::common::{PixelFormat as CodecPixelFormat, VideoFrame};
        use super::super::traits::VideoEncoder;

        let mut encoder = H264Encoder::new(H264EncoderConfig::new(320, 240)).unwrap();
        let frame = VideoFrame {
            width: 160,
            height: 120,
            format: CodecPixelFormat::I420,
            pts: 0,
            data: detailed_frame(160, 120, 0),
            stride_y: 160,
            stride_u: 80,
            stride_v: 80,
        };

        let packets = encoder.encode(&frame).expect("a resize is not an error");
        assert!(!packets.is_empty());
    }

    #[test]
    fn oversized_resolution_is_rejected_clearly() {
        let mut encoder = H264Encoder::new(H264EncoderConfig::new(320, 240)).unwrap();
        let err = encoder
            .encode_yuv420_at(&[0u8; 16], 4096, 2160)
            .expect_err("beyond OpenH264's 3840x2160 ceiling");

        assert!(
            err.to_string().contains("3840x2160"),
            "the limit should be named, not surfaced as an opaque FFI failure: {err}"
        );
    }

    #[test]
    fn undersized_buffer_is_rejected_for_the_declared_size() {
        let mut encoder = H264Encoder::new(H264EncoderConfig::new(320, 240)).unwrap();
        // Claims 320x240 but carries a 160x120 frame's worth of bytes.
        let err = encoder
            .encode_yuv420_at(&detailed_frame(160, 120, 0), 320, 240)
            .expect_err("must not read past the end of the buffer");

        assert!(err.to_string().contains("too small"), "{err}");
    }

    #[test]
    fn test_keyframe_interval_controls_idr_cadence() {
        let interval = 10u32;
        let frames = 30usize;
        let config = H264EncoderConfig::new(320, 240).keyframe_interval(interval);
        let mut config = config;
        config.scene_change_detect = false; // deterministic IDR placement
        let mut encoder = H264Encoder::new(config).unwrap();

        let mut idr_count = 0;
        for i in 0..frames {
            let frame = detailed_frame(320, 240, i as u8);
            let encoded = encoder.encode_yuv420(&frame).unwrap();
            idr_count += count_idr_nals(&encoded);
        }
        // 30 frames at interval 10 => IDRs at frames 0, 10, 20.
        assert_eq!(
            idr_count, 3,
            "expected an IDR every {interval} frames over {frames} frames"
        );
    }

    #[test]
    fn test_qp_affects_output_size() {
        let encode_total = |qp: u8| -> u64 {
            let config = H264EncoderConfig::new(320, 240).qp(qp).keyframe_interval(1);
            let mut encoder = H264Encoder::new(config).unwrap();
            let mut total = 0u64;
            for i in 0..10 {
                let frame = detailed_frame(320, 240, i as u8);
                total += encoder.encode_yuv420(&frame).unwrap().len() as u64;
            }
            total
        };

        let high_quality = encode_total(10);
        let low_quality = encode_total(45);
        assert!(
            high_quality > low_quality,
            "low QP (high quality) must produce more bytes: qp10={high_quality} qp45={low_quality}"
        );
    }

    /// All NAL unit types (header & 0x1F) in an Annex-B bitstream, in order.
    fn nal_unit_types(data: &[u8]) -> Vec<u8> {
        let mut types = Vec::new();
        let mut i = 0;
        while i + 3 < data.len() {
            let offset = if data[i..].starts_with(&[0, 0, 0, 1]) {
                4
            } else if data[i..].starts_with(&[0, 0, 1]) {
                3
            } else {
                i += 1;
                continue;
            };
            if i + offset < data.len() {
                types.push(data[i + offset] & 0x1F);
            }
            i += offset;
        }
        types
    }

    /// Encode two GOPs, force a keyframe mid-GOP, and return the per-frame
    /// access units from the forced IDR onward.
    fn encode_with_midstream_forced_idr(strategy: SpsPpsStrategy) -> Vec<Vec<u8>> {
        let mut config = H264EncoderConfig::new(320, 240)
            .keyframe_interval(10)
            .sps_pps_strategy(strategy);
        config.scene_change_detect = false; // deterministic IDR placement
        let mut encoder = H264Encoder::new(config).unwrap();

        let mut tail = Vec::new();
        for i in 0..18usize {
            if i == 13 {
                // Mid-GOP forced IDR — the keyframe-on-subscribe path.
                encoder.force_keyframe();
            }
            let frame = detailed_frame(320, 240, i as u8);
            let encoded = encoder.encode_yuv420(&frame).unwrap();
            if i == 13 {
                assert!(
                    nal_unit_types(&encoded).contains(&5),
                    "forced keyframe must produce an IDR NAL"
                );
            }
            if i >= 13 {
                tail.push(encoded);
            }
        }
        tail
    }

    /// The keyframe-on-subscribe contract: a decoder that joins at a
    /// mid-stream *forced* IDR must be able to initialize and decode —
    /// under EVERY parameter-set ID strategy (#22).
    #[test]
    fn test_fresh_decoder_syncs_on_midstream_forced_idr() {
        for strategy in [
            SpsPpsStrategy::ConstantId,
            SpsPpsStrategy::IncreasingId,
            SpsPpsStrategy::SpsListing,
            SpsPpsStrategy::SpsListingAndPpsIncreasing,
            SpsPpsStrategy::SpsPpsListing,
        ] {
            let tail = encode_with_midstream_forced_idr(strategy);

            // The forced IDR access unit must carry its own SPS(7)/PPS(8).
            let idr_types = nal_unit_types(&tail[0]);
            assert!(
                idr_types.contains(&7) && idr_types.contains(&8),
                "{strategy:?}: forced IDR must carry SPS+PPS, got NAL types {idr_types:?}"
            );

            let mut decoder = H264Decoder::new().unwrap();
            let mut decoded_any = false;
            for au in &tail {
                if decoder
                    .decode(au)
                    .expect("mid-stream join must decode")
                    .is_some()
                {
                    decoded_any = true;
                }
            }
            assert!(
                decoded_any,
                "{strategy:?}: fresh decoder produced no frame from the forced IDR onward"
            );
        }
    }

    /// Regression (#22): the encoder's output SYNC_POINT flag must reflect
    /// the ENCODED stream, even when the input raw frames are all flagged as
    /// sync points (raw sources flag every frame). A delta AU advertised as
    /// a keyframe sends fresh downstream decoders into an unrecoverable
    /// dsNoParamSets loop.
    #[test]
    fn test_output_sync_point_ignores_input_flag() {
        use crate::buffer::MemoryHandle;
        use crate::memory::SharedArena;
        use crate::metadata::{BufferFlags, Metadata};

        let mut config = H264EncoderConfig::new(320, 240).keyframe_interval(5);
        config.scene_change_detect = false;
        let mut encoder = H264Encoder::new(config).unwrap();

        let arena = SharedArena::new(320 * 240 * 3 / 2, 8).unwrap();
        for i in 0..12usize {
            let frame = detailed_frame(320, 240, i as u8);
            arena.reclaim();
            let mut slot = arena.acquire().unwrap();
            slot.data_mut()[..frame.len()].copy_from_slice(&frame);
            let mut metadata = Metadata::from_sequence(i as u64);
            metadata.flags |= BufferFlags::SYNC_POINT; // raw frames all "keyframes"
            let buffer = Buffer::new(MemoryHandle::with_len(slot, frame.len()), metadata);

            let out = encoder.process(buffer).unwrap().expect("encoded AU");
            let is_idr = nal_unit_types(out.as_bytes()).contains(&5);
            assert_eq!(
                out.metadata().flags.contains(BufferFlags::SYNC_POINT),
                is_idr,
                "frame {i}: SYNC_POINT flag must match IDR presence"
            );
            // Interval 5 → IDRs exactly at frames 0, 5, 10.
            assert_eq!(is_idr, i % 5 == 0, "frame {i}: unexpected IDR cadence");
        }
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = H264Decoder::new();
        assert!(decoder.is_ok());
    }

    #[test]
    fn test_encoder_creation() {
        let config = H264EncoderConfig::new(320, 240);
        let encoder = H264Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Create a simple YUV420 frame (gray + neutral UV)
        let width = 64;
        let height = 64;
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);

        let mut yuv_data = vec![128u8; y_size + uv_size * 2];
        // Make Y plane a gradient
        for y in 0..height {
            for x in 0..width {
                yuv_data[y * width + x] = ((x + y) * 2) as u8;
            }
        }

        // Encode
        let config = H264EncoderConfig::new(width as u32, height as u32);
        let mut encoder = H264Encoder::new(config).expect("Failed to create encoder");

        let encoded = encoder.encode_yuv420(&yuv_data).expect("Failed to encode");
        assert!(!encoded.is_empty(), "Encoded data should not be empty");

        // Decode
        let mut decoder = H264Decoder::new().expect("Failed to create decoder");
        let decoded = decoder.decode(&encoded);

        // The first frame might need SPS/PPS, so decoding might return None
        // This is expected behavior
        assert!(decoded.is_ok());
    }

    #[test]
    fn test_encoder_stats() {
        let width = 64;
        let height = 64;
        let y_size = width * height;
        let uv_size = (width / 2) * (height / 2);
        let yuv_data = vec![128u8; y_size + uv_size * 2];

        let config = H264EncoderConfig::new(width as u32, height as u32);
        let mut encoder = H264Encoder::new(config).expect("Failed to create encoder");

        assert_eq!(encoder.frame_count(), 0);
        assert_eq!(encoder.bytes_encoded(), 0);

        encoder.encode_yuv420(&yuv_data).expect("Failed to encode");

        assert_eq!(encoder.frame_count(), 1);
        assert!(encoder.bytes_encoded() > 0);
    }

    #[test]
    fn test_decoder_stats() {
        let decoder = H264Decoder::new().expect("Failed to create decoder");
        assert_eq!(decoder.frame_count(), 0);
        assert_eq!(decoder.bytes_decoded(), 0);
    }
}
