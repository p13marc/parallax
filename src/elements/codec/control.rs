//! Runtime control handles for codec elements.
//!
//! Once a pipeline is started, its elements are moved into their executor
//! tasks and can no longer be reached through
//! [`Pipeline::get_element_mut`](crate::pipeline::Pipeline::get_element_mut).
//! Control handles bridge that gap: they are cloned from an element *before*
//! `executor.start()` and remain valid while the pipeline runs, exactly like
//! [`AppSinkHandle`](crate::elements::app::AppSinkHandle).
//!
//! # Example
//!
//! ```rust,ignore
//! let encoder = H264Encoder::new(config)?;
//! let keyframes = encoder.keyframe_handle();
//! let control = encoder.control_handle();
//! pipeline.add_filter("enc", encoder);
//!
//! let handle = executor.start(&mut pipeline)?;
//! // ... later, when a new viewer subscribes:
//! keyframes.request(); // next encoded frame is an IDR
//! // ... or when the link gets congested:
//! control.set_bitrate(400_000); // 400 kbps from the next frame on
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, Ordering};

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
/// [`H264Encoder`](super::H264Encoder)), so 0 cannot double as "unset".
const UNSET_U32: u32 = u32::MAX;
/// Sentinel for "this parameter has never been set" (QP is 0-51, so 255 is free).
const UNSET_U8: u8 = u8::MAX;

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
/// let encoder = H264Encoder::new(H264EncoderConfig::new(1280, 720))?;
/// let control = encoder.control_handle();   // BEFORE start
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
