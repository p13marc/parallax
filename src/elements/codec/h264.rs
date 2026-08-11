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
//! let config = H264EncoderConfig::new();
//! let mut encoder = H264Encoder::new(config)?;
//!
//! // Encode YUV frames
//! let encoded = encoder.encode_yuv420_at(&yuv_data, 1920, 1080)?;
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
use crate::control::{EncoderStatsHandle, RateControlMode};
use crate::element::Element;
use crate::error::{Error, Result};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;

use std::collections::VecDeque;

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

/// Map the crate-wide [`RateControlMode`] onto OpenH264's `iRCMode`.
fn rate_control_to_openh264(mode: RateControlMode) -> openh264::encoder::RateControlMode {
    use openh264::encoder::RateControlMode as O;
    match mode {
        RateControlMode::Quality => O::Quality,
        RateControlMode::Bitrate => O::Bitrate,
        RateControlMode::BufferBased => O::Bufferbased,
        RateControlMode::Timestamp => O::Timestamp,
        RateControlMode::Off => O::Off,
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
///
/// Carries **no dimensions**. Geometry travels in-band, in [`Metadata`], and the
/// encoder takes it from each frame — OpenH264 re-initialises itself on a size
/// change and emits a fresh IDR, so a mid-stream resize is free. Config
/// dimensions would be a value that *looks* authoritative and is silently
/// ignored the moment a scaler is in the graph.
#[derive(Debug, Clone)]
pub struct H264EncoderConfig {
    /// Target bitrate in bits per second.
    ///
    /// Defaults to 2 Mbps — a real budget, not `0`. Under the default
    /// [`RateControlMode::Bitrate`] a target of `0` is rejected at construction
    /// rather than silently handed to OpenH264, which would fall back to its own
    /// ~120 kbps default and drop most frames of real content.
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
    /// [`RateControlMode`]). Defaults to `Bitrate`: this crate's reason to exist
    /// is holding a bandwidth budget, and OpenH264's quality-first default makes
    /// the budget advisory.
    ///
    /// Changeable on a running encoder via
    /// [`EncoderControl::set_rate_control`](crate::control::EncoderControl::set_rate_control).
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
    /// Defaults to `false` — spend quality, not frames. OpenH264 turns skipping
    /// on whenever a bitrate target is set, and it has a sharp edge: under a
    /// tight target the encoder simply emits *nothing* for some input frames, so
    /// a pipeline expecting one packet per frame quietly gets fewer and every
    /// downstream fps/kbps figure is wrong. An upstream
    /// [`Throttle`](crate::elements::Throttle) is the way to shed frames
    /// deliberately.
    ///
    /// Changeable on a running encoder via
    /// [`EncoderControl::set_skip_frames`](crate::control::EncoderControl::set_skip_frames).
    pub skip_frames: bool,
    /// H.264 profile. `None` lets OpenH264 choose.
    pub profile: Option<Profile>,
    /// CPU spent per frame (see [`Complexity`]).
    pub complexity: Complexity,
    /// Content type, which tunes the encoder's heuristics (see [`UsageType`]).
    pub usage_type: UsageType,
}

impl H264EncoderConfig {
    /// Create a new encoder configuration.
    ///
    /// No dimensions: the encoder encodes whatever each buffer declares.
    pub fn new() -> Self {
        Self {
            bitrate_bps: 2_000_000,
            max_frame_rate: 30.0,
            qp: 26,
            scene_change_detect: true,
            keyframe_interval: 0,
            num_threads: 0,
            sps_pps_strategy: SpsPpsStrategy::default(),
            rate_control: RateControlMode::default(),
            max_slice_len: None,
            skip_frames: false,
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
        self.skip_frames = skip;
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
    pub fn low_latency() -> Self {
        Self::new()
            .frame_rate(30.0)
            .keyframe_interval(30) // Keyframe every second at 30fps
            .qp(28) // Slightly lower quality for speed
    }

    /// Create a configuration for high-quality encoding.
    pub fn high_quality() -> Self {
        Self::new()
            .frame_rate(30.0)
            .keyframe_interval(120) // Keyframe every 4 seconds
            .qp(20) // Higher quality
    }
}

impl Default for H264EncoderConfig {
    fn default() -> Self {
        Self::new()
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
    /// Frames, bytes, rate-control drops and encode time — readable while the
    /// element is inside its executor task (see [`Self::stats`]).
    stats: EncoderStatsHandle,
    /// Geometry of the last frame encoded. Observability and change detection
    /// only — it is *never* a fallback, because geometry travels in-band.
    dims: Option<(u32, u32)>,
    /// Runtime control: keyframe requests plus bitrate/GOP/QP changes (shared
    /// with [`Self::control`] and [`Self::keyframe_handle`]).
    control: super::EncoderControl,
    /// The control generation last applied to the encoder.
    applied_generation: u64,
    /// Arena for output buffer allocation, sized by the executor at start.
    output: OutputArena,
}

/// Build an OpenH264 encoder from our config.
///
/// Shared by [`H264Encoder::new`] and the runtime reconfigure path, so
/// start-time and live settings can never drift apart.
fn build_encoder(config: &H264EncoderConfig) -> Result<Encoder> {
    // Bitrate mode with no target is a trap: OpenH264 falls back to its own
    // ~120 kbps default and drops most frames of real content. Refuse instead.
    if config.rate_control == RateControlMode::Bitrate && config.bitrate_bps == 0 {
        return Err(Error::Config(
            "H264Encoder: RateControlMode::Bitrate needs a bitrate target — set \
             H264EncoderConfig::bitrate(), or pick RateControlMode::Quality"
                .into(),
        ));
    }

    let mut encoder_config = EncoderConfig::new();

    if config.bitrate_bps > 0 {
        encoder_config = encoder_config.bitrate(BitRate::from_bps(config.bitrate_bps));
    }

    encoder_config = encoder_config.skip_frames(config.skip_frames);

    // The QP band is what rate control has to work with.
    //
    // In Bitrate mode with frame skipping off — this crate's default — OpenH264
    // can only hold the target by raising QP, and a ±4 band around the target
    // does not give it enough room: the target is then simply missed (OpenH264
    // even warns that the bitrate "can't be controlled" without frame skipping).
    // So the band opens all the way down in quality: `qp` becomes a quality
    // *ceiling* the encoder may fall below to make budget, which is exactly
    // "spend quality, not frames".
    //
    // In every other mode `qp` is a target and the tight band around it is the
    // point.
    let qp = config.qp.min(51);
    let qp_range = if config.rate_control == RateControlMode::Bitrate && !config.skip_frames {
        QpRange::new(qp.saturating_sub(4), 51)
    } else {
        QpRange::new(qp.saturating_sub(4), (qp + 4).min(51))
    };

    encoder_config = encoder_config
        .max_frame_rate(FrameRate::from_hz(config.max_frame_rate))
        .scene_change_detect(config.scene_change_detect)
        .num_threads(config.num_threads as u16)
        .qp(qp_range)
        .intra_frame_period(IntraFramePeriod::from_num_frames(config.keyframe_interval))
        .sps_pps_strategy(config.sps_pps_strategy.to_openh264())
        .rate_control_mode(rate_control_to_openh264(config.rate_control))
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

        // Sized by the executor from link capacity, with a 1 MiB slot floor:
        // an encoded frame is usually well under that, but a keyframe at high
        // bitrate is not, and slot size is fixed once the arena is built.
        let output =
            OutputArena::new(defaults::VIDEO_ENCODER_SLOT_COUNT).with_min_slot_size(1024 * 1024);

        Ok(Self {
            encoder,
            config,
            stats: EncoderStatsHandle::default(),
            dims: None,
            control: super::EncoderControl::new(),
            applied_generation: 0,
            output,
        })
    }

    /// The geometry of the last frame encoded, if any.
    ///
    /// `None` before the first frame: this encoder takes its size from the data,
    /// so until data arrives it does not have one.
    pub fn dimensions(&self) -> Option<(u32, u32)> {
        self.dims
    }

    /// Rebuild the OpenH264 encoder for the current config, keeping the arena.
    ///
    /// This is the path for GOP, QP, rate-control mode and frame-skip changes,
    /// where a fresh parameter set is wanted anyway: decoders need the new SPS
    /// / PPS, so the forced IDR is a feature rather than a cost. **Bitrate does
    /// not come through here** — [`set_bitrate_live`](Self::set_bitrate_live)
    /// changes it on the running encoder with no IDR.
    ///
    /// The **arena is deliberately reused**: allocating a fresh 64 MiB of slots
    /// on every parameter step would be a silent memory leak in a long-running
    /// sensor.
    fn rebuild_encoder(&mut self) -> Result<()> {
        self.encoder = build_encoder(&self.config)?;
        // A new encoder starts a new sequence; make its first frame an IDR so
        // decoders pick up the new parameter sets immediately.
        self.encoder.force_intra_frame();
        Ok(())
    }

    /// Change the bitrate on the live encoder, with no IDR and no rebuild.
    ///
    /// OpenH264's Rust wrapper exposes no bitrate setter, but the C API does —
    /// `SetOption(ENCODER_OPTION_BITRATE)`, which OpenH264 itself uses
    /// internally. Reached through `Encoder::raw_api()`.
    ///
    /// Returns `false` if the encoder rejected the change, in which case the
    /// caller falls back to a rebuild. A bitrate step must never fail the
    /// pipeline.
    fn set_bitrate_live(&mut self, bps: u32) -> bool {
        use openh264_sys2::{ENCODER_OPTION_BITRATE, SBitrateInfo, SPATIAL_LAYER_ALL};

        let mut info = SBitrateInfo {
            iLayer: SPATIAL_LAYER_ALL,
            iBitrate: bps as std::os::raw::c_int,
        };

        // SAFETY: `info` outlives the call, and ENCODER_OPTION_BITRATE is
        // documented to take a *mut SBitrateInfo. The encoder is exclusively
        // borrowed, so nothing else touches it concurrently.
        let rc = unsafe {
            self.encoder
                .raw_api()
                .set_option(ENCODER_OPTION_BITRATE, (&raw mut info).cast())
        };
        rc == 0
    }

    /// Apply any parameter changes made through the [`EncoderControl`] handle.
    ///
    /// Two things matter here:
    ///
    /// * **A no-op costs nothing.** Every parameter is compared against the
    ///   current config, so `set_bitrate(same_value)` does not force an IDR —
    ///   which is what a naive `if changed { rebuild }` on the generation
    ///   counter alone used to do.
    /// * **A bitrate-only change is seamless.** It goes through `SetOption`, so
    ///   the GOP is not broken. Every other parameter (GOP length, QP, rate
    ///   control, frame skipping) needs a fresh encoder, and that rebuild picks
    ///   up any bitrate change with it.
    fn apply_pending_control(&mut self) -> Result<()> {
        let Some(params) = self.control.poll(&mut self.applied_generation) else {
            return Ok(());
        };

        let mut bitrate_change = None;
        let mut needs_rebuild = false;

        if let Some(bps) = params.bitrate_bps
            && bps != self.config.bitrate_bps
        {
            self.config.bitrate_bps = bps;
            bitrate_change = Some(bps);
        }
        if let Some(frames) = params.keyframe_interval
            && frames != self.config.keyframe_interval
        {
            self.config.keyframe_interval = frames;
            needs_rebuild = true;
        }
        if let Some(qp) = params.qp
            && qp.min(51) != self.config.qp
        {
            self.config.qp = qp.min(51);
            needs_rebuild = true;
        }
        if let Some(mode) = params.rate_control
            && mode != self.config.rate_control
        {
            self.config.rate_control = mode;
            needs_rebuild = true;
        }
        if let Some(skip) = params.skip_frames
            && skip != self.config.skip_frames
        {
            self.config.skip_frames = skip;
            needs_rebuild = true;
        }

        if needs_rebuild {
            tracing::info!(
                "H264Encoder: rebuilding — bitrate {} bps, keyframe interval {}, qp {}, \
                 rate control {:?}, skip frames {}",
                self.config.bitrate_bps,
                self.config.keyframe_interval,
                self.config.qp,
                self.config.rate_control,
                self.config.skip_frames,
            );
            return self.rebuild_encoder();
        }

        if let Some(bps) = bitrate_change {
            if self.set_bitrate_live(bps) {
                tracing::info!("H264Encoder: bitrate -> {bps} bps (seamless, no IDR)");
            } else {
                tracing::warn!("H264Encoder: live bitrate change rejected, rebuilding");
                return self.rebuild_encoder();
            }
        }

        Ok(())
    }

    /// Encode a YUV420 frame at the configured resolution.
    ///
    /// The input data must be in YUV420 planar format:
    /// - Y plane: width * height bytes
    /// - U plane: (width/2) * (height/2) bytes
    /// - V plane: (width/2) * (height/2) bytes
    ///
    /// Encode a YUV420 frame of the given size, changing resolution if needed.
    ///
    /// A size different from the last frame's re-initialises the encoder and
    /// starts a fresh IDR, so the switch is a clean decoder entry point. There is
    /// no configured resolution to contradict: geometry comes from the data.
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

        if let Some(previous) = self.dims
            && previous != (width, height)
        {
            tracing::info!(
                "H264Encoder: resolution {}x{} -> {}x{} (rebuild, IDR)",
                previous.0,
                previous.1,
                width,
                height
            );
            // OpenH264 would re-initialise itself on the size change — but from
            // the parameters it was *constructed* with, silently reverting any
            // bitrate applied since through SetOption. Rebuilding from our own
            // config keeps the config authoritative. A resolution change forces
            // a fresh IDR either way, so this costs nothing extra.
            self.rebuild_encoder()?;
        }
        self.dims = Some((width, height));

        let yuv = YuvFrame {
            data: yuv_data,
            width: width as usize,
            height: height as usize,
        };

        // OpenH264 notices the dimension change itself: it re-initialises via
        // SetOption(SVC_ENCODE_PARAM_EXT) and forces an IDR. We only have to
        // stop pinning the size and let the frame through.
        let started = std::time::Instant::now();
        let bitstream = self
            .encoder
            .encode(&yuv)
            .map_err(|e| Error::Config(format!("H.264 encode failed: {:?}", e)))?;

        let encoded = bitstream.to_vec();
        let elapsed_ns = started.elapsed().as_nanos() as u64;

        if encoded.is_empty() {
            // Rate control swallowed the frame. Routine once skip_frames is on,
            // and counted nowhere before this.
            self.stats.record_rc_drop(elapsed_ns);
        } else {
            self.stats.record_frame(encoded.len(), elapsed_ns);
        }

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
        self.stats.frames_encoded()
    }

    /// Get the total bytes encoded.
    pub fn bytes_encoded(&self) -> u64 {
        self.stats.bytes_encoded()
    }

    /// A cloneable handle to this encoder's counters.
    ///
    /// Clone it *before* `executor.start()`: the element is moved into its
    /// executor task there, so `frame_count()` and `bytes_encoded()` — plain
    /// `&self` methods on the element — can never be called while it is actually
    /// encoding. This handle can.
    ///
    /// It also exposes what those two never could: `frames_dropped_by_rc`
    /// (frames rate control swallowed) and `last_encode_ns`.
    pub fn stats(&self) -> EncoderStatsHandle {
        self.stats.clone()
    }

    /// Get the encoder configuration.
    pub fn config(&self) -> &H264EncoderConfig {
        &self.config
    }
}

impl super::Controllable for H264Encoder {
    type Control = super::EncoderControl;

    /// A handle for changing bitrate, keyframe interval, QP, rate-control mode
    /// and frame skipping on a running pipeline.
    ///
    /// Clone it *before* `executor.start()` — see [`crate::control`].
    fn control(&self) -> super::EncoderControl {
        self.control.clone()
    }
}

impl std::fmt::Debug for H264Encoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("H264Encoder")
            .field("config", &self.config)
            .field("stats", &self.stats.snapshot())
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
///
/// Thin alias for [`annexb::has_idr`](crate::codec::annexb::has_idr), which is
/// the one copy of this scan in the crate.
fn contains_idr(data: &[u8]) -> bool {
    crate::codec::annexb::has_idr(data)
}

/// Element trait implementation for H264Encoder.
impl Element for H264Encoder {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Admission control before the frame enters the GOP. Encoding a frame
        // whose packet is then shed would leave the decoder with a reference it
        // never received, corrupt until the next IDR; skipping the input just
        // lowers the frame rate, which is recoverable and what `skip_frames`
        // already does on purpose.
        self.output.admit()?;

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
        // can retarget mid-stream, and the encoder follows it. There is no
        // constructor value to fall back to — a frame that does not say how big
        // it is cannot be encoded, and inventing a size would silently produce
        // sheared or truncated video.
        let (width, height) = buffer.metadata().video_dims().ok_or_else(|| {
            Error::Config(
                "H264Encoder: buffer carries no video dimensions — the upstream element must \
                 call Metadata::set_video_dims()"
                    .into(),
            )
        })?;

        // Caps say I420 (`input_media_caps`), but nothing enforced it, so RGB
        // bytes were happily encoded as if they were YUV planes. Only reject a
        // format that is *declared* and wrong, so a hand-built I420 buffer with
        // only legacy keys still works.
        if let Some(pf) = buffer.metadata().video_pixel_format()
            && pf != crate::format::PixelFormat::I420
        {
            return Err(Error::Config(format!(
                "H264Encoder needs I420, got {pf:?} — insert a VideoConvertElement upstream"
            )));
        }

        let input_data = buffer.as_bytes();
        let encoded = self.encode_yuv420_at(input_data, width, height)?;

        if encoded.is_empty() {
            return Ok(None);
        }
        let is_keyframe = contains_idr(&encoded);

        let mut slot = self.output.acquire(encoded.len(), "h264encoder")?;
        slot.data_mut()[..encoded.len()].copy_from_slice(&encoded);

        let handle = crate::buffer::MemoryHandle::with_len(slot, encoded.len());
        // Preserve input buffer's PTS for proper timing, update sequence number
        let mut metadata = buffer.metadata().clone();
        metadata.sequence = self.frame_count().saturating_sub(1);
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
        // Same seamless path as the control-handle route: a bitrate step must
        // not cost an IDR just because the caller reached this encoder through
        // the VideoEncoder trait (i.e. wrapped in an EncoderElement) rather
        // than through EncoderControl.
        if self.set_bitrate_live(bps) {
            tracing::info!("H264Encoder: bitrate -> {bps} bps (seamless, no IDR)");
            Ok(())
        } else {
            tracing::warn!("H264Encoder: live bitrate change rejected, rebuilding");
            self.rebuild_encoder()
        }
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
    /// Arena for output buffer allocation, sized by the executor at start.
    output: OutputArena,
    /// Geometry of the last decoded frame, to spot a mid-stream resize.
    last_dims: Option<(u32, u32)>,
    /// Access-unit metadata waiting for its decoded frame.
    ///
    /// The decoder is driven through openh264's `decode_frame_no_delay`, so a
    /// `decode()` call returns at most one frame — but not necessarily the one
    /// belonging to the AU just fed: a reordered stream buffers pictures and
    /// hands back an earlier one. Output is in *display* order, so the entry to
    /// stamp a frame with is the one with the smallest PTS still in flight,
    /// which degenerates to plain FIFO for a stream without B-frames.
    pending: VecDeque<Metadata>,
    /// Frames drained by [`Element::flush`], handed out one per call.
    ///
    /// `None` until the first flush; `Some(empty)` afterwards, so the decoder
    /// is drained exactly once.
    flushed: Option<VecDeque<DecodedFrame>>,
    /// Buffers emitted so far, the source of the output sequence number.
    frames_out: u64,
}

/// Cap on in-flight access-unit metadata.
///
/// A stream that feeds parameter sets and never produces a picture would
/// otherwise grow this queue without bound. Real reorder depth is a handful of
/// frames; past this the oldest entry is the one that will never be claimed.
const MAX_PENDING_METADATA: usize = 64;

impl H264Decoder {
    /// Create a new H.264 decoder.
    pub fn new() -> Result<Self> {
        // NoFlush is load-bearing: the wrapper's default (`Flush::Flush`)
        // force-drains openh264's DPB on every decode call that produces no
        // immediate picture. B-frame streams legitimately delay output for
        // reordering, and a mid-stream force-flush corrupts the decoder's
        // NAL bookkeeping — openh264 then fails with dsOutOfMemory a few
        // AUs later (reproduced on H.264 Main WEB-DL streams). Delayed
        // frames drain through `Element::flush` at EOS instead.
        let config = openh264::decoder::DecoderConfig::new()
            .flush_after_decode(openh264::decoder::Flush::NoFlush);
        let decoder = Decoder::with_api_config(OpenH264API::from_source(), config)
            .map_err(|e| Error::Config(format!("Failed to create H.264 decoder: {:?}", e)))?;

        // Slot size comes from the first decoded frame, so 4K works without
        // the hard-coded 4 MiB ceiling this used to carry; the floor keeps a
        // 1080p arena from being rebuilt for a slightly larger frame.
        let output = OutputArena::new(defaults::VIDEO_DECODER_SLOT_COUNT)
            .with_min_slot_size(4 * 1024 * 1024);

        Ok(Self {
            decoder,
            frame_count: 0,
            bytes_decoded: 0,
            output,
            last_dims: None,
            pending: VecDeque::new(),
            flushed: None,
            frames_out: 0,
        })
    }

    /// Claim the metadata belonging to the frame just emitted.
    ///
    /// See [`pending`](Self::pending): frames come out in display order, so the
    /// smallest PTS still in flight is the right one. `ClockTime::NONE` is
    /// `u64::MAX`, so un-timestamped entries sort last and are consumed FIFO
    /// among themselves.
    fn take_pending_metadata(&mut self) -> Option<Metadata> {
        let oldest = self
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(_, meta)| meta.pts)
            .map(|(index, _)| index)?;
        self.pending.remove(oldest)
    }

    /// Build an output buffer for a decoded frame, carrying `source` forward.
    ///
    /// `source` is the originating access unit's metadata when there is one.
    /// Timestamps, duration and flags ride along; sequence and geometry are the
    /// decoder's to set.
    fn frame_to_buffer(
        &mut self,
        frame: &DecodedFrame,
        source: Option<Metadata>,
    ) -> Result<Buffer> {
        let yuv_data = frame.to_yuv420_planar();

        // A resize needs a differently-sized slot, and the arena's is
        // fixed once built.
        let dims = (frame.width() as u32, frame.height() as u32);
        if self.last_dims.is_some_and(|last| last != dims) {
            tracing::info!(
                "h264decoder: resolution changed to {}x{}, rebuilding the output arena",
                dims.0,
                dims.1
            );
            self.output.reset();
        }
        self.last_dims = Some(dims);

        let mut slot = self.output.acquire(yuv_data.len(), "h264decoder")?;
        slot.data_mut()[..yuv_data.len()].copy_from_slice(&yuv_data);

        let handle = crate::buffer::MemoryHandle::with_len(slot, yuv_data.len());

        // The input AU's timing is the frame's timing — regenerating it here
        // is what used to make a decoded stream unschedulable downstream (#64).
        let mut metadata = source.unwrap_or_default();
        // Counted separately from `frame_count`: the inherent `flush` bumps
        // that one in a single step, which would stamp every drained frame
        // with the same sequence number.
        metadata.sequence = self.frames_out;
        self.frames_out += 1;
        // Geometry travels in-band. `to_yuv420_planar` is in the name:
        // whatever the bitstream was, what we emit is I420. This also
        // rewrites `format` from `Video(H264)` to `VideoRaw`.
        metadata.set_video_dims(dims.0, dims.1, crate::format::PixelFormat::I420);

        Ok(Buffer::new(handle, metadata))
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
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    // No admission control here, deliberately: a skipped input is a
    // reference frame the decoder never sees, so everything after it
    // decodes wrong. Decode, then shed the output copy if there is
    // nowhere to put it — the decoder's own state stays intact.
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // Remember this AU's timing before decoding: the frame that comes out
        // may belong to an earlier one.
        if self.pending.len() >= MAX_PENDING_METADATA {
            self.pending.pop_front();
        }
        self.pending.push_back(buffer.metadata().clone());

        let input_data = buffer.as_bytes();

        match self.decode(input_data)? {
            Some(frame) => {
                let source = self.take_pending_metadata();
                Ok(Some(self.frame_to_buffer(&frame, source)?))
            }
            None => Ok(None),
        }
    }

    /// Drain pictures openh264 still holds when the stream ends.
    ///
    /// Without this the tail of every stream — everything the decoder had
    /// buffered for reordering — was silently discarded at EOS.
    fn flush(&mut self) -> Result<Option<Buffer>> {
        if self.flushed.is_none() {
            let frames = H264Decoder::flush(self)?;
            self.flushed = Some(frames.into());
        }

        match self.flushed.as_mut().and_then(VecDeque::pop_front) {
            Some(frame) => {
                let source = self.take_pending_metadata();
                Ok(Some(self.frame_to_buffer(&frame, source)?))
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
    use crate::clock::ClockTime;
    use crate::control::Controllable;
    use crate::memory::SharedArena;

    #[test]
    fn test_encoder_config_default() {
        let config = H264EncoderConfig::default();
        assert_eq!(config.qp, 26);
    }

    #[test]
    fn test_encoder_config_builder() {
        let config = H264EncoderConfig::new()
            .bitrate(1_000_000)
            .frame_rate(25.0)
            .qp(24)
            .keyframe_interval(60);

        assert_eq!(config.bitrate_bps, 1_000_000);
        assert_eq!(config.max_frame_rate, 25.0);
        assert_eq!(config.qp, 24);
        assert_eq!(config.keyframe_interval, 60);
    }

    #[test]
    fn test_encoder_config_low_latency() {
        let config = H264EncoderConfig::low_latency();
        assert_eq!(config.keyframe_interval, 30);
    }

    #[test]
    fn test_encoder_config_high_quality() {
        let config = H264EncoderConfig::high_quality();
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
            let mut config = H264EncoderConfig::new().bitrate(150_000).rate_control(mode);
            config.scene_change_detect = false;
            let mut encoder = H264Encoder::new(config).unwrap();
            for i in 0..20 {
                encoder
                    .encode_yuv420_at(&detailed_frame(320, 240, i as u8), 320, 240)
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
        let mut config = H264EncoderConfig::new().max_slice_len(CAP);
        config.scene_change_detect = false;
        let mut encoder = H264Encoder::new(config).unwrap();

        let mut seen_any = false;
        for i in 0..5 {
            let encoded = encoder
                .encode_yuv420_at(&detailed_frame(320, 240, i as u8), 320, 240)
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
            let mut encoder = H264Encoder::new(H264EncoderConfig::new().profile(profile)).unwrap();
            let encoded = encoder
                .encode_yuv420_at(&detailed_frame(320, 240, 0), 320, 240)
                .unwrap();

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
            let mut config = H264EncoderConfig::new()
                .bitrate(20_000) // a target far too tight for this content
                .rate_control(RateControlMode::Bitrate)
                .skip_frames(skip);
            config.scene_change_detect = false;
            let mut encoder = H264Encoder::new(config).unwrap();

            (0..20)
                .filter(|i| {
                    !encoder
                        .encode_yuv420_at(&detailed_frame(320, 240, *i as u8), 320, 240)
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
    fn the_defaults_hold_a_bitrate_budget_and_never_drop_frames() {
        // These two defaults deliberately diverge from OpenH264's. A crate whose
        // headline feature is live bandwidth control must treat the bitrate as a
        // budget, not a hint — and it must shed quality rather than frames, so
        // downstream fps accounting stays honest.
        let config = H264EncoderConfig::new();
        assert_eq!(config.rate_control, RateControlMode::Bitrate);
        assert_eq!(config.bitrate_bps, 2_000_000);
        assert!(!config.skip_frames);

        assert_eq!(config.max_slice_len, None);
        assert_eq!(config.profile, None);
        assert_eq!(config.complexity, Complexity::Medium);
        assert_eq!(config.usage_type, UsageType::CameraRealtime);
    }

    #[test]
    fn bitrate_mode_without_a_target_is_refused_not_silently_mishandled() {
        // OpenH264 would fall back to ~120 kbps and drop most frames of real
        // content. Refusing is the only honest option.
        let config = H264EncoderConfig::new()
            .rate_control(RateControlMode::Bitrate)
            .bitrate(0);
        let error = H264Encoder::new(config).unwrap_err().to_string();
        assert!(error.contains("needs a bitrate target"), "got: {error}");

        // Quality mode with no target is fine.
        let config = H264EncoderConfig::new()
            .rate_control(RateControlMode::Quality)
            .bitrate(0);
        assert!(H264Encoder::new(config).is_ok());
    }

    #[test]
    fn config_knobs_survive_a_live_reconfigure() {
        use crate::element::Element;

        // The rebuild path must re-apply every knob, not just the three the
        // control handle carries.
        let config = H264EncoderConfig::new()
            .bitrate(2_000_000)
            .rate_control(RateControlMode::Bitrate)
            .max_slice_len(1200)
            .profile(Profile::Baseline);
        let mut encoder = H264Encoder::new(config).unwrap();
        let control = encoder.control();

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

        let mut config = H264EncoderConfig::new().bitrate(4_000_000);
        config.scene_change_detect = false;
        let mut encoder = H264Encoder::new(config).unwrap();
        let control = encoder.control();

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
    fn a_live_bitrate_change_breaks_no_gop() {
        use crate::element::Element;

        // A bitrate step goes through OpenH264's SetOption, so the GOP survives
        // it. This used to rebuild the encoder and force an IDR — expensive, and
        // brutal for a controller that steps the rate often.
        let mut config = H264EncoderConfig::new().bitrate(4_000_000);
        config.scene_change_detect = false;
        let mut encoder = H264Encoder::new(config).unwrap();
        let control = encoder.control();

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
            0,
            "a bitrate change must not force a keyframe"
        );
        assert_eq!(encoder.config().bitrate_bps, 400_000);
    }

    #[test]
    fn a_no_op_bitrate_change_does_nothing_at_all() {
        use crate::element::Element;

        // Setting a parameter to the value it already holds must not rebuild the
        // encoder, and so must not force an IDR.
        let mut config = H264EncoderConfig::new().bitrate(2_000_000);
        config.scene_change_detect = false;
        config.keyframe_interval = 0;
        let mut encoder = H264Encoder::new(config).unwrap();
        let control = encoder.control();

        for i in 0..3 {
            encoder
                .process(yuv_buffer(&detailed_frame(320, 240, i), 320, 240))
                .unwrap();
        }

        control.set_bitrate(2_000_000); // same value
        control.set_qp(26); // same value
        let out = encoder
            .process(yuv_buffer(&detailed_frame(320, 240, 4), 320, 240))
            .unwrap()
            .expect("a frame");

        assert_eq!(
            count_idr_nals(out.as_bytes()),
            0,
            "a no-op reconfigure must not force a keyframe"
        );
    }

    #[test]
    fn stats_count_frames_bytes_and_encode_time() {
        use crate::element::Element;

        let mut encoder = H264Encoder::new(H264EncoderConfig::new()).unwrap();
        let stats = encoder.stats();
        assert_eq!(stats.frames_encoded(), 0);

        let mut total = 0u64;
        for i in 0..5 {
            let out = encoder
                .process(yuv_buffer(&detailed_frame(320, 240, i), 320, 240))
                .unwrap()
                .expect("a frame");
            total += out.as_bytes().len() as u64;
        }

        let snapshot = stats.snapshot();
        assert_eq!(snapshot.frames_encoded, 5);
        assert_eq!(snapshot.bytes_encoded, total);
        assert_eq!(snapshot.frames_dropped_by_rc, 0);
        assert!(snapshot.last_encode_ns > 0);
    }

    #[test]
    fn live_keyframe_interval_change_changes_idr_cadence() {
        use crate::element::Element;

        let mut config = H264EncoderConfig::new().keyframe_interval(100);
        config.scene_change_detect = false;
        let mut encoder = H264Encoder::new(config).unwrap();
        let control = encoder.control();

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

        let mut encoder = H264Encoder::new(H264EncoderConfig::new().bitrate(1_000_000)).unwrap();

        // Setting the value it already has must not force a spurious IDR.
        encoder.set_bitrate(1_000_000).unwrap();
        let encoded = encoder
            .encode_yuv420_at(&detailed_frame(320, 240, 1), 320, 240)
            .unwrap();
        let first_idrs = count_idr_nals(&encoded);

        encoder.set_bitrate(1_000_000).unwrap();
        let encoded = encoder
            .encode_yuv420_at(&detailed_frame(320, 240, 2), 320, 240)
            .unwrap();
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
        let mut config = H264EncoderConfig::new();
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
    fn dimensions_come_from_the_frames_not_the_config() {
        let mut encoder = H264Encoder::new(H264EncoderConfig::new()).unwrap();
        assert_eq!(
            encoder.dimensions(),
            None,
            "before the first frame the encoder has no size — it takes one from the data"
        );

        encoder
            .encode_yuv420_at(&detailed_frame(160, 120, 0), 160, 120)
            .unwrap();
        assert_eq!(encoder.dimensions(), Some((160, 120)));

        // ...and follows a mid-stream resize.
        encoder
            .encode_yuv420_at(&detailed_frame(320, 240, 1), 320, 240)
            .unwrap();
        assert_eq!(encoder.dimensions(), Some((320, 240)));
    }

    /// VideoEncoder::encode used to hard-error on any size mismatch.
    #[test]
    fn video_encoder_trait_accepts_a_changed_frame_size() {
        use super::super::common::{PixelFormat as CodecPixelFormat, VideoFrame};
        use super::super::traits::VideoEncoder;

        let mut encoder = H264Encoder::new(H264EncoderConfig::new()).unwrap();
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
        let mut encoder = H264Encoder::new(H264EncoderConfig::new()).unwrap();
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
        let mut encoder = H264Encoder::new(H264EncoderConfig::new()).unwrap();
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
        let config = H264EncoderConfig::new().keyframe_interval(interval);
        let mut config = config;
        config.scene_change_detect = false; // deterministic IDR placement
        let mut encoder = H264Encoder::new(config).unwrap();

        let mut idr_count = 0;
        for i in 0..frames {
            let frame = detailed_frame(320, 240, i as u8);
            let encoded = encoder.encode_yuv420_at(&frame, 320, 240).unwrap();
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
            let config = H264EncoderConfig::new().qp(qp).keyframe_interval(1);
            let mut encoder = H264Encoder::new(config).unwrap();
            let mut total = 0u64;
            for i in 0..10 {
                let frame = detailed_frame(320, 240, i as u8);
                total += encoder.encode_yuv420_at(&frame, 320, 240).unwrap().len() as u64;
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
        let mut config = H264EncoderConfig::new()
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
            let encoded = encoder.encode_yuv420_at(&frame, 320, 240).unwrap();
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

        let mut config = H264EncoderConfig::new().keyframe_interval(5);
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
            metadata.set_video_dims(320, 240, crate::format::PixelFormat::I420);
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
        let config = H264EncoderConfig::new();
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
        let config = H264EncoderConfig::new();
        let mut encoder = H264Encoder::new(config).expect("Failed to create encoder");

        let encoded = encoder
            .encode_yuv420_at(&yuv_data, width as u32, height as u32)
            .expect("Failed to encode");
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

        let config = H264EncoderConfig::new();
        let mut encoder = H264Encoder::new(config).expect("Failed to create encoder");

        assert_eq!(encoder.frame_count(), 0);
        assert_eq!(encoder.bytes_encoded(), 0);

        encoder
            .encode_yuv420_at(&yuv_data, width as u32, height as u32)
            .expect("Failed to encode");

        assert_eq!(encoder.frame_count(), 1);
        assert!(encoder.bytes_encoded() > 0);
    }

    #[test]
    fn test_decoder_stats() {
        let decoder = H264Decoder::new().expect("Failed to create decoder");
        assert_eq!(decoder.frame_count(), 0);
        assert_eq!(decoder.bytes_decoded(), 0);
    }

    // ------------------------------------------------------------------
    // Decoder timestamp passthrough (#64)
    // ------------------------------------------------------------------

    /// Wrap an Annex-B access unit in a Buffer stamped like a real demuxer
    /// would stamp it.
    fn au_buffer(data: &[u8], pts_ns: u64, duration_ns: u64) -> Buffer {
        use crate::buffer::MemoryHandle;

        let arena = SharedArena::new(data.len().max(1), 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..data.len()].copy_from_slice(data);

        let mut metadata = Metadata::new();
        metadata.pts = ClockTime::from_nanos(pts_ns);
        metadata.dts = ClockTime::from_nanos(pts_ns);
        metadata.duration = ClockTime::from_nanos(duration_ns);
        Buffer::new(MemoryHandle::with_len(slot, data.len()), metadata)
    }

    /// Encode `count` frames and push them through the decoder as an element,
    /// returning (input timing, output timing) as `(pts, dts, duration)`.
    #[allow(clippy::type_complexity)]
    fn decode_timestamped(count: u64) -> (Vec<(u64, u64, u64)>, Vec<(u64, u64, u64)>) {
        const FRAME_NS: u64 = 40_000_000; // 25 fps
        let (width, height) = (64usize, 64usize);

        let mut encoder = H264Encoder::new(H264EncoderConfig::new()).unwrap();
        let mut decoder = H264Decoder::new().unwrap();

        let mut fed = Vec::new();
        let mut out = Vec::new();

        for i in 0..count {
            let frame = detailed_frame(width, height, i as u8);
            let au = encoder
                .encode_yuv420_at(&frame, width as u32, height as u32)
                .unwrap();
            if au.is_empty() {
                continue;
            }

            let pts = i * FRAME_NS;
            fed.push((pts, pts, FRAME_NS));

            if let Some(decoded) = decoder.process(au_buffer(&au, pts, FRAME_NS)).unwrap() {
                let meta = decoded.metadata();
                out.push((meta.pts.nanos(), meta.dts.nanos(), meta.duration.nanos()));
            }
        }

        (fed, out)
    }

    #[test]
    fn decoded_frames_carry_the_input_timestamps() {
        let (fed, out) = decode_timestamped(10);

        assert!(
            !out.is_empty(),
            "decoder produced no frames — the test stream is broken, not the timing"
        );
        // openh264 is driven with `decode_frame_no_delay` and this stream has
        // no B-frames, so output order is input order: the decoded timings must
        // be a prefix of what was fed, not a regenerated sequence.
        assert_eq!(out, fed[..out.len()]);
    }

    #[test]
    fn decoded_frames_still_declare_their_geometry() {
        let (width, height) = (64usize, 64usize);
        let mut encoder = H264Encoder::new(H264EncoderConfig::new()).unwrap();
        let mut decoder = H264Decoder::new().unwrap();

        let mut saw_frame = false;
        for i in 0..5u64 {
            let frame = detailed_frame(width, height, i as u8);
            let au = encoder
                .encode_yuv420_at(&frame, width as u32, height as u32)
                .unwrap();
            if au.is_empty() {
                continue;
            }
            if let Some(decoded) = decoder.process(au_buffer(&au, i, 1)).unwrap() {
                // Cloning the input metadata must not lose the geometry the
                // decoder stamps on — nor leave `format` as `Video(H264)`.
                assert_eq!(
                    decoded.metadata().video_dims(),
                    Some((width as u32, height as u32))
                );
                assert_eq!(
                    decoded.metadata().video_pixel_format(),
                    Some(crate::format::PixelFormat::I420)
                );
                saw_frame = true;
            }
        }
        assert!(saw_frame);
    }

    #[test]
    fn output_sequence_numbers_still_count_from_zero() {
        let (width, height) = (64usize, 64usize);
        let mut encoder = H264Encoder::new(H264EncoderConfig::new()).unwrap();
        let mut decoder = H264Decoder::new().unwrap();

        let mut sequences = Vec::new();
        for i in 0..6u64 {
            let frame = detailed_frame(width, height, i as u8);
            let au = encoder
                .encode_yuv420_at(&frame, width as u32, height as u32)
                .unwrap();
            if au.is_empty() {
                continue;
            }
            // Input sequence numbers are deliberately nonsense: the decoder
            // owns this field, the demuxer does not.
            let mut buffer = au_buffer(&au, i, 1);
            buffer.metadata_mut().sequence = 900 + i;
            if let Some(decoded) = decoder.process(buffer).unwrap() {
                sequences.push(decoded.metadata().sequence);
            }
        }

        assert!(!sequences.is_empty());
        assert_eq!(sequences, (0..sequences.len() as u64).collect::<Vec<_>>());
    }

    #[test]
    fn flush_drains_the_decoder_exactly_once() {
        let (width, height) = (64usize, 64usize);
        let mut encoder = H264Encoder::new(H264EncoderConfig::new()).unwrap();
        let mut decoder = H264Decoder::new().unwrap();

        for i in 0..5u64 {
            let frame = detailed_frame(width, height, i as u8);
            let au = encoder
                .encode_yuv420_at(&frame, width as u32, height as u32)
                .unwrap();
            if !au.is_empty() {
                let _ = decoder.process(au_buffer(&au, i * 40_000_000, 40_000_000));
            }
        }

        // The executor calls flush until it returns None. Whatever openh264 was
        // still holding comes out here — and every drained frame is a real one,
        // geometry and all.
        let mut drained = 0;
        while let Some(buffer) = Element::flush(&mut decoder).unwrap() {
            assert_eq!(
                buffer.metadata().video_dims(),
                Some((width as u32, height as u32))
            );
            drained += 1;
            assert!(drained < 64, "flush never terminated");
        }

        // Draining is one-shot: a second round must not re-enter the decoder.
        assert!(Element::flush(&mut decoder).unwrap().is_none());
    }

    #[test]
    fn pending_metadata_cannot_grow_without_bound() {
        let mut decoder = H264Decoder::new().unwrap();

        // Junk that never yields a picture: without the cap this queue would
        // grow one entry per access unit, forever.
        for i in 0..(MAX_PENDING_METADATA * 2) {
            let _ = decoder.process(au_buffer(&[0, 0, 0, 1, 0x09, 0x10], i as u64, 1));
        }

        assert_eq!(decoder.pending.len(), MAX_PENDING_METADATA);
    }
}
