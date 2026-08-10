//! Runtime control: change a running pipeline without tearing it down.
//!
//! # The one invariant
//!
//! [`Executor::start`](crate::pipeline::Executor::start) **moves** every element
//! into its executor task, so
//! [`Pipeline::get_element_mut`](crate::pipeline::Pipeline::get_element_mut)
//! returns `None` for anything that is running. The *only* way to reach a live
//! element is through a control handle:
//!
//! > **Clone the handle from the element _before_ `executor.start()`.**
//!
//! Every handle here is an `Arc<Atomic…>` — lock-free, allocation-free, safe to
//! call from any thread or async task, and free on the hot path when nothing
//! has changed.
//!
//! # The handles
//!
//! Every controllable element implements [`Controllable`], so the accessor is
//! always called `control()`:
//!
//! | Handle | From | Changes |
//! |--------|------|---------|
//! | [`EncoderControl`] | `H264Encoder`, `EncoderElement` | bitrate, GOP, QP, rate-control mode, frame skipping, keyframe requests |
//! | [`EncoderStatsHandle`] | `H264Encoder`, `EncoderElement` | *read-only*: frames, bytes, rate-control drops, encode time |
//! | [`KeyframeHandle`] | `…::keyframe_handle()` | force the next frame to be an IDR |
//! | [`ScaleControl`] | `VideoScale` | target resolution (`set_max_height`, `passthrough`) |
//! | [`ThrottleControl`] | `Throttle` | framerate (drop-based) |
//! | `JpegQualityControl` | `JpegEncoder` | JPEG quality 1..=100 |
//! | [`ValveControl`] | `Valve` | open / close |
//! | [`GainControl`] | `Gain` | audio volume (`set_factor`, `set_db`) |
//! | [`FlowStateHandle`] | `Queue` | backpressure signalling to live sources |
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::control::{Controllable, EncoderControl};
//!
//! let encoder = H264Encoder::new(config)?;
//! let control = encoder.control();      // BEFORE start()
//! let stats = encoder.stats();          //  ""
//! pipeline.add_filter("enc", encoder);
//!
//! let handle = executor.start(&mut pipeline)?;
//! // ... later, when a new viewer subscribes:
//! control.request_keyframe();           // next encoded frame is an IDR
//! // ... or when the link gets congested:
//! control.set_bitrate(400_000);         // 400 kbps, applied seamlessly
//! println!("{} kbps", stats.bytes_encoded());
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

/// An element that exposes a runtime control handle.
///
/// The handle must be cloned *before* `executor.start()` — see the
/// [module docs](self).
pub trait Controllable {
    /// The handle type this element hands out.
    type Control;

    /// Clone this element's control handle.
    fn control(&self) -> Self::Control;
}

/// Buffer-metadata key requesting a keyframe from a downstream encoder.
///
/// Elements that inject buffers (e.g. via
/// [`AppSrcHandle`](crate::elements::app::AppSrcHandle)) can set this key to
/// `true` on a buffer's metadata; encoder elements honor it by forcing the
/// frame carrying it (or the next one) to be encoded as a keyframe.
pub const KEYFRAME_REQUEST: &str = "video/keyframe_request";

/// Cloneable handle to request a keyframe (IDR) from a running encoder.
///
/// A request is *sticky*: it stays pending until the encoder processes its
/// next frame, then it is consumed. Multiple requests arriving before that
/// frame coalesce into a single keyframe — the correct semantics for
/// "N subscribers joined, give them a decodable picture".
///
/// The handle is lock-free and allocation-free, so it is safe to check from
/// real-time encoder paths and to call [`request`](Self::request) from any
/// thread or async task.
#[derive(Clone, Debug, Default)]
pub struct KeyframeHandle(Arc<AtomicBool>);

impl KeyframeHandle {
    /// Create a new handle with no pending request.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request that the next encoded frame be a keyframe.
    pub fn request(&self) {
        self.0.store(true, Ordering::Release);
    }

    /// Whether a request is pending (not yet consumed by the encoder).
    pub fn is_pending(&self) -> bool {
        self.0.load(Ordering::Acquire)
    }

    /// Consume a pending request, returning whether one was pending.
    ///
    /// Called by encoder elements at the top of their processing path.
    /// Custom encoder elements outside this crate may use it the same way.
    pub fn take(&self) -> bool {
        self.0.swap(false, Ordering::AcqRel)
    }
}

/// Sentinel for "this parameter has never been set" in [`EncoderControl`].
///
/// A bitrate of 0 is *meaningful* (it selects quality mode in
/// `H264Encoder`), so 0 cannot double as "unset".
const UNSET_U32: u32 = u32::MAX;
/// Sentinel for "this parameter has never been set" (QP is 0-51, so 255 is free).
const UNSET_U8: u8 = u8::MAX;

/// How an encoder trades bits against quality.
///
/// This is the knob that decides how seriously a bitrate target is taken. It
/// lives here, not in the codec module, because it is a *live* parameter: it can
/// be changed on a running encoder through [`EncoderControl::set_rate_control`].
///
/// The default is [`Bitrate`](Self::Bitrate) — for a crate whose headline
/// feature is live bandwidth control, "the bitrate is a hint" is the wrong
/// default. (OpenH264's own default is quality-first; this crate deliberately
/// diverges.)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum RateControlMode {
    /// Bitrate first: hold the target, spending quality to do it. What you want
    /// when the link, not the picture, is the constraint.
    #[default]
    Bitrate = 0,
    /// Quality first: the bitrate target is a hint, not a budget.
    Quality = 1,
    /// Ignore the bitrate target; adjust quality from buffer status alone.
    BufferBased = 2,
    /// Rate control driven by frame timestamps.
    Timestamp = 3,
    /// No rate control at all: quality is governed purely by the QP band.
    Off = 4,
}

impl RateControlMode {
    /// The `u8` this mode is stored as inside [`EncoderControl`].
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Recover a mode from its `u8`, or `None` if it is not one.
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::Bitrate),
            1 => Some(Self::Quality),
            2 => Some(Self::BufferBased),
            3 => Some(Self::Timestamp),
            4 => Some(Self::Off),
            _ => None,
        }
    }
}

/// Encoder parameters pending application, as reported by
/// [`EncoderControl::poll`].
///
/// `None` means "not set through this handle — keep whatever the encoder was
/// constructed with".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncoderParams {
    /// Target bitrate in bits per second (0 selects quality mode, where
    /// supported).
    pub bitrate_bps: Option<u32>,
    /// Keyframe (IDR) interval in frames; 0 lets the encoder decide.
    pub keyframe_interval: Option<u32>,
    /// Target quantization parameter (0-51, lower = better quality).
    pub qp: Option<u8>,
    /// How strictly the bitrate target is honoured.
    pub rate_control: Option<RateControlMode>,
    /// Whether rate control may drop frames to hold the bitrate target.
    pub skip_frames: Option<bool>,
}

impl EncoderParams {
    /// Whether any parameter is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }
}

/// Cloneable handle for reconfiguring a **running** encoder.
///
/// Bandwidth is governed by the parameters behind this handle: bitrate is the
/// dominant knob, keyframe interval matters on low-motion scenes (an IDR costs
/// several times a delta frame), and QP trades quality for bytes at a fixed
/// resolution. Resolution — the other big lever — is not here: it travels
/// in-band as buffer format metadata, so it is the *scaler* you retarget (see
/// [`VideoScale`](crate::elements::transform::VideoScale)), not the encoder.
///
/// Like [`KeyframeHandle`], this must be cloned from the element **before**
/// `executor.start()` — elements are moved into their executor tasks at start
/// and cannot be reached through the graph afterwards.
///
/// Changes are *latched*, not queued: setting the bitrate three times before
/// the encoder next runs applies only the last value. Encoders observe changes
/// through [`poll`](Self::poll), which costs a single relaxed load per frame
/// when nothing has changed.
///
/// # Example
///
/// ```rust,ignore
/// let encoder = H264Encoder::new(H264EncoderConfig::new())?;
/// let control = encoder.control();          // BEFORE start
/// pipeline.add_filter("enc", encoder);
/// let handle = executor.start(&mut pipeline)?;
///
/// control.set_bitrate(800_000);        // 800 kbps
/// control.set_keyframe_interval(60);   // IDR every 2s at 30fps
/// ```
#[derive(Clone, Debug)]
pub struct EncoderControl(Arc<EncoderControlInner>);

impl Default for EncoderControl {
    /// Equivalent to [`EncoderControl::new`].
    ///
    /// Deliberately not derived: a derived `Default` would zero the atomics,
    /// and 0 is a *meaningful* bitrate (quality mode), not "unset".
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug)]
struct EncoderControlInner {
    /// Keyframe requests share the existing handle type, so
    /// [`EncoderControl::keyframe_handle`] and a directly-cloned
    /// [`KeyframeHandle`] address the same flag.
    keyframe: KeyframeHandle,
    bitrate_bps: AtomicU32,
    keyframe_interval: AtomicU32,
    qp: AtomicU8,
    /// A [`RateControlMode`] as `u8`, or [`UNSET_U8`].
    rate_control: AtomicU8,
    /// Tri-state: 0 = no, 1 = yes, [`UNSET_U8`] = never set.
    skip_frames: AtomicU8,
    /// Bumped by every setter. Encoders compare it against the generation they
    /// last applied, so an unchanged handle costs one relaxed load per frame.
    generation: AtomicU64,
}

impl EncoderControl {
    /// Create a new handle with no pending changes.
    pub fn new() -> Self {
        Self(Arc::new(EncoderControlInner {
            keyframe: KeyframeHandle::new(),
            bitrate_bps: AtomicU32::new(UNSET_U32),
            keyframe_interval: AtomicU32::new(UNSET_U32),
            qp: AtomicU8::new(UNSET_U8),
            rate_control: AtomicU8::new(UNSET_U8),
            skip_frames: AtomicU8::new(UNSET_U8),
            generation: AtomicU64::new(0),
        }))
    }

    /// Request that the next encoded frame be a keyframe (IDR).
    ///
    /// Identical to [`KeyframeHandle::request`] — the two handles share state.
    pub fn request_keyframe(&self) {
        self.0.keyframe.request();
    }

    /// The keyframe handle backing this control.
    pub fn keyframe_handle(&self) -> KeyframeHandle {
        self.0.keyframe.clone()
    }

    /// Set the target bitrate in bits per second.
    ///
    /// `0` means "no bitrate target" — encoders that support it fall back to
    /// quality-driven rate control.
    pub fn set_bitrate(&self, bps: u32) {
        self.0.bitrate_bps.store(bps, Ordering::Release);
        self.bump();
    }

    /// Set the keyframe (IDR) interval in frames. `0` lets the encoder decide.
    ///
    /// Longer intervals cost less bandwidth but make late joiners wait longer
    /// for a decodable frame.
    pub fn set_keyframe_interval(&self, frames: u32) {
        self.0.keyframe_interval.store(frames, Ordering::Release);
        self.bump();
    }

    /// Set the target quantization parameter (clamped to 0-51).
    pub fn set_qp(&self, qp: u8) {
        self.0.qp.store(qp.min(51), Ordering::Release);
        self.bump();
    }

    /// Set how strictly the bitrate target is honoured (see [`RateControlMode`]).
    pub fn set_rate_control(&self, mode: RateControlMode) {
        self.0.rate_control.store(mode.as_u8(), Ordering::Release);
        self.bump();
    }

    /// Allow or forbid rate control dropping frames to hold the bitrate target.
    ///
    /// With skipping on, a tight target makes the encoder emit *nothing* for
    /// some input frames, so a pipeline expecting one packet per frame quietly
    /// gets fewer. Off (the default) it spends quality instead.
    pub fn set_skip_frames(&self, skip: bool) {
        self.0.skip_frames.store(u8::from(skip), Ordering::Release);
        self.bump();
    }

    /// The generation counter, incremented by every setter.
    pub fn generation(&self) -> u64 {
        self.0.generation.load(Ordering::Acquire)
    }

    /// Return the pending parameters iff something changed since `last`.
    ///
    /// `last` is updated in place, so a second call with the same handle
    /// returns `None` until the next setter runs. Encoder elements call this at
    /// the top of their processing path:
    ///
    /// ```rust,ignore
    /// if let Some(params) = self.control.poll(&mut self.applied_generation) {
    ///     self.reconfigure(params)?;
    /// }
    /// ```
    pub fn poll(&self, last: &mut u64) -> Option<EncoderParams> {
        let generation = self.0.generation.load(Ordering::Acquire);
        if generation == *last {
            return None;
        }
        *last = generation;

        let params = EncoderParams {
            bitrate_bps: unset_u32(self.0.bitrate_bps.load(Ordering::Acquire)),
            keyframe_interval: unset_u32(self.0.keyframe_interval.load(Ordering::Acquire)),
            qp: unset_u8(self.0.qp.load(Ordering::Acquire)),
            rate_control: unset_u8(self.0.rate_control.load(Ordering::Acquire))
                .and_then(RateControlMode::from_u8),
            skip_frames: unset_u8(self.0.skip_frames.load(Ordering::Acquire)).map(|v| v != 0),
        };

        // A generation bump with nothing set cannot happen through the public
        // setters, but a caller-built handle could see it; report nothing
        // rather than an empty reconfigure.
        (!params.is_empty()).then_some(params)
    }

    /// Consume a pending keyframe request, returning whether one was pending.
    pub fn take_keyframe(&self) -> bool {
        self.0.keyframe.take()
    }

    fn bump(&self) {
        self.0.generation.fetch_add(1, Ordering::AcqRel);
    }
}

fn unset_u32(value: u32) -> Option<u32> {
    (value != UNSET_U32).then_some(value)
}

fn unset_u8(value: u8) -> Option<u8> {
    (value != UNSET_U8).then_some(value)
}

// ============================================================================
// Encoder statistics
// ============================================================================

/// Read-only handle to a running encoder's counters.
///
/// The numbers a streaming sender needs — how many frames went in, how many
/// bytes came out, how many frames the rate controller swallowed, and how long
/// the last encode took — are computed inside the element, which
/// `executor.start()` then moves out of reach. Clone this handle *before*
/// start, exactly like [`EncoderControl`].
///
/// Zenoh and other TCP/QUIC transports hide packet loss behind retransmission,
/// so there is no congestion signal to close a rate-control loop on. These
/// sender-side counters are the only bandwidth feedback that exists.
#[derive(Clone, Debug, Default)]
pub struct EncoderStatsHandle(Arc<EncoderStatsInner>);

#[derive(Debug, Default)]
struct EncoderStatsInner {
    frames_encoded: AtomicU64,
    bytes_encoded: AtomicU64,
    frames_dropped_by_rc: AtomicU64,
    last_encode_ns: AtomicU64,
}

impl EncoderStatsHandle {
    /// Create a new handle with all counters at zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Frames the encoder emitted a bitstream for.
    pub fn frames_encoded(&self) -> u64 {
        self.0.frames_encoded.load(Ordering::Relaxed)
    }

    /// Total bytes of encoded bitstream produced.
    pub fn bytes_encoded(&self) -> u64 {
        self.0.bytes_encoded.load(Ordering::Relaxed)
    }

    /// Input frames the encoder swallowed without emitting anything.
    ///
    /// Rate control is allowed to spend *frames* rather than *quality* to hold
    /// a bitrate target (see `H264EncoderConfig::skip_frames`). When it does,
    /// the input frame simply produces no output — which any downstream
    /// fps/kbps accounting has to know about.
    pub fn frames_dropped_by_rc(&self) -> u64 {
        self.0.frames_dropped_by_rc.load(Ordering::Relaxed)
    }

    /// Wall-clock duration of the most recent encode call, in nanoseconds.
    pub fn last_encode_ns(&self) -> u64 {
        self.0.last_encode_ns.load(Ordering::Relaxed)
    }

    /// A consistent-enough snapshot of every counter.
    pub fn snapshot(&self) -> EncoderStats {
        EncoderStats {
            frames_encoded: self.frames_encoded(),
            bytes_encoded: self.bytes_encoded(),
            frames_dropped_by_rc: self.frames_dropped_by_rc(),
            last_encode_ns: self.last_encode_ns(),
        }
    }

    /// Record an encoded frame. Called by encoder elements.
    pub fn record_frame(&self, bytes: usize, encode_ns: u64) {
        self.0.frames_encoded.fetch_add(1, Ordering::Relaxed);
        self.0
            .bytes_encoded
            .fetch_add(bytes as u64, Ordering::Relaxed);
        self.0.last_encode_ns.store(encode_ns, Ordering::Relaxed);
    }

    /// Record an input frame that rate control produced no output for.
    pub fn record_rc_drop(&self, encode_ns: u64) {
        self.0.frames_dropped_by_rc.fetch_add(1, Ordering::Relaxed);
        self.0.last_encode_ns.store(encode_ns, Ordering::Relaxed);
    }
}

/// A point-in-time copy of [`EncoderStatsHandle`]'s counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EncoderStats {
    /// Frames the encoder emitted a bitstream for.
    pub frames_encoded: u64,
    /// Total bytes of encoded bitstream produced.
    pub bytes_encoded: u64,
    /// Input frames swallowed by rate control.
    pub frames_dropped_by_rc: u64,
    /// Duration of the most recent encode, in nanoseconds.
    pub last_encode_ns: u64,
}

// ============================================================================
// Re-exports: one place to find every runtime handle
// ============================================================================

pub use crate::elements::flow::ValveControl;
pub use crate::elements::timing::ThrottleControl;
pub use crate::elements::transform::{GainControl, ScaleControl};
pub use crate::pipeline::flow::FlowStateHandle;

#[cfg(feature = "image-jpeg")]
pub use crate::elements::codec::JpegQualityControl;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_is_sticky_and_consumed_by_take() {
        let handle = KeyframeHandle::new();
        assert!(!handle.is_pending());
        assert!(!handle.take());

        handle.request();
        assert!(handle.is_pending());
        assert!(handle.take(), "take must observe the request");
        assert!(!handle.is_pending(), "take must consume the request");
        assert!(!handle.take());
    }

    #[test]
    fn multiple_requests_coalesce() {
        let handle = KeyframeHandle::new();
        handle.request();
        handle.request();
        handle.request();
        assert!(handle.take());
        assert!(!handle.take(), "requests coalesce into one keyframe");
    }

    #[test]
    fn clones_share_state() {
        let a = KeyframeHandle::new();
        let b = a.clone();
        b.request();
        assert!(a.take());
        assert!(!b.is_pending());
    }

    #[test]
    fn poll_reports_a_change_exactly_once() {
        let control = EncoderControl::new();
        let mut applied = 0;

        assert_eq!(control.poll(&mut applied), None, "nothing set yet");

        control.set_bitrate(800_000);
        assert_eq!(
            control.poll(&mut applied),
            Some(EncoderParams {
                bitrate_bps: Some(800_000),
                ..Default::default()
            })
        );
        assert_eq!(
            control.poll(&mut applied),
            None,
            "an applied change is not reported again"
        );
    }

    #[test]
    fn unset_parameters_stay_none() {
        let control = EncoderControl::new();
        let mut applied = 0;
        control.set_qp(30);

        let params = control.poll(&mut applied).expect("qp change");
        assert_eq!(params.qp, Some(30));
        assert_eq!(
            params.bitrate_bps, None,
            "an untouched knob must not be reported as a change"
        );
        assert_eq!(params.keyframe_interval, None);
    }

    #[test]
    fn zero_bitrate_is_a_value_not_absence() {
        // 0 selects quality mode; it must survive as Some(0), not read as unset.
        let control = EncoderControl::new();
        let mut applied = 0;
        control.set_bitrate(0);

        let params = control.poll(&mut applied).expect("bitrate change");
        assert_eq!(params.bitrate_bps, Some(0));
    }

    #[test]
    fn default_handle_has_no_pending_parameters() {
        // Guards the hand-written Default: a derived one would zero the atomics
        // and report bitrate 0 / interval 0 / qp 0 as pending changes.
        let control = EncoderControl::default();
        let mut applied = 0;
        assert_eq!(control.poll(&mut applied), None);
    }

    #[test]
    fn repeated_sets_latch_the_last_value() {
        let control = EncoderControl::new();
        let mut applied = 0;
        control.set_bitrate(4_000_000);
        control.set_bitrate(2_000_000);
        control.set_bitrate(400_000);

        let params = control.poll(&mut applied).expect("bitrate change");
        assert_eq!(
            params.bitrate_bps,
            Some(400_000),
            "changes latch; only the last value is applied"
        );
        assert_eq!(control.poll(&mut applied), None, "coalesced into one apply");
    }

    #[test]
    fn accumulated_parameters_are_reported_together() {
        let control = EncoderControl::new();
        let mut applied = 0;
        control.set_bitrate(1_000_000);
        control.set_keyframe_interval(60);
        control.set_qp(28);

        assert_eq!(
            control.poll(&mut applied),
            Some(EncoderParams {
                bitrate_bps: Some(1_000_000),
                keyframe_interval: Some(60),
                qp: Some(28),
                rate_control: None,
                skip_frames: None,
            })
        );
    }

    #[test]
    fn qp_is_clamped_to_the_h264_range() {
        let control = EncoderControl::new();
        let mut applied = 0;
        control.set_qp(200);
        assert_eq!(control.poll(&mut applied).unwrap().qp, Some(51));
    }

    #[test]
    fn keyframe_state_is_shared_with_the_keyframe_handle() {
        let control = EncoderControl::new();
        let keyframes = control.keyframe_handle();

        control.request_keyframe();
        assert!(keyframes.is_pending(), "handles address the same flag");
        assert!(control.take_keyframe());
        assert!(!keyframes.is_pending());

        keyframes.request();
        assert!(control.take_keyframe(), "and the reverse direction");
    }

    #[test]
    fn keyframe_requests_do_not_trigger_a_reconfigure() {
        let control = EncoderControl::new();
        let mut applied = 0;
        control.request_keyframe();
        assert_eq!(
            control.poll(&mut applied),
            None,
            "an IDR request is not an encoder reconfiguration"
        );
    }

    #[test]
    fn control_clones_share_state() {
        let a = EncoderControl::new();
        let b = a.clone();
        let mut applied = 0;

        b.set_bitrate(500_000);
        assert_eq!(a.poll(&mut applied).unwrap().bitrate_bps, Some(500_000));
    }
}
