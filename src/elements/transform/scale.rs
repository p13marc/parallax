//! Video scaling/resizing element.
//!
//! Scales raw video frames, preserving their pixel format. Every format the
//! conversion engines handle works here — I420, NV12, YUYV, UYVY, RGB24, BGR24,
//! RGBA, BGRA and Gray8 — because this element is a thin pipeline wrapper around
//! [`converters::ScaleEngine`](crate::converters::ScaleEngine), which does the
//! actual resampling.
//!
//! # Geometry travels in-band
//!
//! The scaler takes **no dimensions at construction**. The source size and pixel
//! format come from each buffer's [`Metadata`], and the *target* is a runtime
//! knob ([`ScaleControl`]) because it is one — resolution is the second-biggest
//! bandwidth lever after bitrate. A buffer that does not describe its own
//! geometry is an error, not a guess: the byte length alone cannot distinguish
//! I420 from NV12 (both are `w*h*3/2`), and guessing wrong scrambles chroma
//! silently.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::transform::{VideoScale, ScaleMode};
//!
//! let scaler = VideoScale::new().with_mode(ScaleMode::NearestNeighbor);
//! let scale = scaler.control();   // BEFORE executor.start()
//! pipeline.add_filter("scale", scaler);
//!
//! scale.set_max_height(360);      // downscale to 360p, aspect preserved
//! scale.passthrough();            // back to the source resolution
//! ```

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::buffer::{Buffer, MemoryHandle};
use crate::converters;
use crate::element::Element;
use crate::error::{Error, Result};
use crate::format::{
    CapsValue, ElementMediaCaps, FormatMemoryCap, MemoryCaps, PixelFormat, VideoFormatCaps,
};
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;

// ============================================================================
// Scale Mode
// ============================================================================

/// The one interpolation-mode enum, shared with the engine.
///
/// Lives in [`crate::converters`] (the element depends on the engine, never
/// the other way around) and is re-exported here so element users need not
/// know the split exists.
pub use crate::converters::ScaleMode;

// ============================================================================
// Runtime Control
// ============================================================================

/// Cloneable handle to retarget a **running** [`VideoScale`].
///
/// Resolution is the second-biggest bandwidth lever after bitrate — cutting
/// both dimensions in half quarters the pixel count — and the only one that
/// also cuts encoder CPU. Like the codec control handles, this must be cloned
/// from the element **before** `executor.start()`: elements are moved into
/// their executor tasks at start and cannot be reached through the graph
/// afterwards.
///
/// The target is resolved against the *current* source dimensions on every
/// frame, so [`set_max_height`](Self::set_max_height) keeps working when the
/// source itself changes resolution mid-stream.
///
/// # Example
///
/// ```rust,ignore
/// let scaler = VideoScale::new();
/// let scale = scaler.control();          // BEFORE start
/// pipeline.add_filter("scale", scaler);
/// let handle = executor.start(&mut pipeline)?;
///
/// scale.set_max_height(360);  // downscale to 360p, aspect preserved
/// scale.passthrough();        // back to source resolution
/// ```
#[derive(Clone, Debug)]
pub struct ScaleControl(Arc<ScaleTarget>);

/// The requested output size.
///
/// A `width` of 0 means "derive from the source aspect ratio"; a `height` of 0
/// with a 0 width means "no scaling".
#[derive(Debug, Default)]
struct ScaleTarget {
    width: AtomicU32,
    height: AtomicU32,
}

impl ScaleControl {
    /// Create a handle with no scaling requested (passthrough).
    pub fn new() -> Self {
        Self(Arc::new(ScaleTarget::default()))
    }

    /// Scale to exactly `width` x `height`.
    ///
    /// Odd dimensions are rounded down to even — YUV420 chroma is subsampled by
    /// two and cannot represent an odd width or height. A consequence at the
    /// bottom of the range: an axis of 1 rounds to 0, which is the
    /// "unconstrained" sentinel, so `set_target(1, 1)` is
    /// [`passthrough`](Self::passthrough).
    pub fn set_target(&self, width: u32, height: u32) {
        self.0.width.store(even(width), Ordering::Release);
        self.0.height.store(even(height), Ordering::Release);
    }

    /// Fit the frame within `height`, preserving the source aspect ratio.
    ///
    /// The width is derived per frame from the *current* source, so this
    /// survives a source resolution change. A source already at or below
    /// `height` is passed through untouched — this never upscales.
    pub fn set_max_height(&self, height: u32) {
        self.0.width.store(0, Ordering::Release);
        self.0.height.store(even(height), Ordering::Release);
    }

    /// Fit the frame within `width`, preserving the source aspect ratio.
    ///
    /// The mirror of [`set_max_height`](Self::set_max_height): the height is
    /// derived per frame from the *current* source, and a source already at or
    /// below `width` is passed through untouched.
    pub fn set_max_width(&self, width: u32) {
        self.0.width.store(even(width), Ordering::Release);
        self.0.height.store(0, Ordering::Release);
    }

    /// Stop scaling: emit frames at the source resolution.
    pub fn passthrough(&self) {
        self.0.width.store(0, Ordering::Release);
        self.0.height.store(0, Ordering::Release);
    }

    /// The requested target, as `(width, height)` where 0 means "unconstrained".
    pub fn target(&self) -> (u32, u32) {
        (
            self.0.width.load(Ordering::Acquire),
            self.0.height.load(Ordering::Acquire),
        )
    }

    /// The output geometry this scaler will produce for a source of
    /// `src_width` x `src_height` under the current target.
    ///
    /// Pure, and the *same* arithmetic `process()` uses — call it to advertise
    /// a stream's geometry ahead of the frames (a catalogue, per-tier status,
    /// wire metadata) instead of predicting it. Because
    /// [`new`](Self::new)/[`Default`] give a usable handle with no element
    /// attached, this works without a pipeline.
    ///
    /// # Contract
    ///
    /// This rounding is a committed part of the API; the steps are not
    /// interchangeable and the order matters:
    ///
    /// 1. A bound ([`set_max_height`](Self::set_max_height) /
    ///    [`set_max_width`](Self::set_max_width)) derives the other axis from
    ///    the source aspect ratio with **`div_ceil`** on a `u64` intermediate.
    /// 2. Both axes are then rounded **down** to even (YUV420 chroma is
    ///    subsampled by two and cannot represent an odd size), and clamped to a
    ///    minimum of 2.
    /// 3. A bound **never upscales**: a source already at or below the bound is
    ///    returned unchanged. An explicit [`set_target`](Self::set_target) is
    ///    honoured as given, up *or* down.
    ///
    /// Step 1 rounds up and step 2 rounds down, which is easy to get wrong in
    /// the other direction: 1280x720 bounded to height 480 resolves to
    /// **854**x480, not 852x480 (`1280 * 480 / 720` = 853.33, ceil 854, already
    /// even).
    pub fn resolve(&self, src_width: u32, src_height: u32) -> (u32, u32) {
        let (want_width, want_height) = self.target();

        match (want_width, want_height) {
            (0, 0) => (src_width, src_height),

            // Bounding height: derive the width from the source aspect ratio,
            // and pass through a source that already fits.
            (0, h) if src_height > 0 => {
                if even(h) >= src_height || even(h) == 0 {
                    return (src_width, src_height);
                }
                let w = (src_width as u64 * h as u64).div_ceil(src_height as u64) as u32;
                (even(w).max(2), even(h))
            }
            // Bounding width: the mirror case.
            (w, 0) if src_width > 0 => {
                if even(w) >= src_width || even(w) == 0 {
                    return (src_width, src_height);
                }
                let h = (src_height as u64 * w as u64).div_ceil(src_width as u64) as u32;
                (even(w), even(h).max(2))
            }

            // Explicit target: exactly what was asked for.
            (w, h) => (even(w).max(2), even(h).max(2)),
        }
    }
}

impl Default for ScaleControl {
    fn default() -> Self {
        Self::new()
    }
}

/// Round down to an even number (YUV420 cannot represent odd dimensions).
fn even(value: u32) -> u32 {
    value & !1
}

// ============================================================================
// Video Scaler
// ============================================================================

/// Video scaling element.
///
/// Resamples raw video frames to a target size, **preserving the pixel format**.
/// The work is delegated to [`converters::ScaleEngine`], so every format that
/// engine supports works here: I420, NV12, YUYV, UYVY, RGB24, BGR24, RGBA, BGRA
/// and Gray8.
///
/// Geometry travels in-band. The source size and pixel format are read from each
/// buffer's [`Metadata`]; the *target* is a runtime knob ([`ScaleControl`]).
/// Output buffers are stamped with the size actually produced — and with the
/// **input's** pixel format, since scaling does not change it — so a downstream
/// encoder follows a resize instead of encoding at a stale geometry.
///
/// A buffer that carries no geometry, or a pixel format no engine handles, is an
/// error. There is no constructor value to fall back to, and inferring the
/// format from the byte count is ambiguous (I420 and NV12 are both `w*h*3/2`).
pub struct VideoScale {
    /// The requested target, shared with [`Self::control`] handles.
    control: ScaleControl,
    /// Interpolation mode, fixed at construction.
    mode: ScaleMode,
    /// Cached engine, rebuilt whenever `engine_key` changes.
    engine: Option<converters::ScaleEngine>,
    /// The `(format, src_w, src_h, dst_w, dst_h)` the cached engine was built
    /// for. A change in *any* of them rebuilds it.
    engine_key: Option<(converters::PixelFormat, u32, u32, u32, u32)>,
    /// Geometry of the last frame — observability only, never a fallback.
    last_source: Option<(u32, u32, PixelFormat)>,
    /// Target of the last frame — observability only.
    last_target: Option<(u32, u32)>,
    /// Statistics.
    frames_processed: u64,
    /// Arena for output buffers, sized by the executor at start.
    output: OutputArena,
}

impl std::fmt::Debug for VideoScale {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VideoScale")
            .field("target", &self.control.target())
            .field("mode", &self.mode)
            .field("last_source", &self.last_source)
            .field("last_target", &self.last_target)
            .field("frames_processed", &self.frames_processed)
            .field("output", &self.output)
            .finish()
    }
}

impl VideoScale {
    /// Create a scaler that passes frames through until it is given a target.
    ///
    /// Takes no dimensions: the source is described by each buffer, and the
    /// target belongs on [`control`](Self::control) because it is a runtime knob.
    pub fn new() -> Self {
        Self {
            control: ScaleControl::new(),
            mode: ScaleMode::default(),
            engine: None,
            engine_key: None,
            last_source: None,
            last_target: None,
            frames_processed: 0,
            // A retarget changes the output size, and grow_to_fit is what
            // rebuilds for it — the same thing the hand-rolled check did.
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT).grow_to_fit(),
        }
    }

    /// Get a cloneable handle for retargeting this scaler at runtime.
    ///
    /// Clone it *before* the pipeline starts — see [`crate::control`].
    ///
    /// Kept as an inherent method as well as the
    /// [`Controllable`](crate::control::Controllable) impl so callers need not
    /// import the trait.
    pub fn control(&self) -> ScaleControl {
        self.control.clone()
    }

    /// Set the interpolation mode.
    pub fn with_mode(mut self, mode: ScaleMode) -> Self {
        self.mode = mode;
        self.engine_key = None; // force a rebuild with the new algorithm
        self
    }

    /// The source geometry of the most recent frame, if any.
    ///
    /// `None` before the first buffer: until one arrives, this element does not
    /// know its geometry, and saying so is the point.
    pub fn src_dimensions(&self) -> Option<(u32, u32)> {
        self.last_source.map(|(w, h, _)| (w, h))
    }

    /// The target geometry of the most recent frame, if any.
    pub fn dst_dimensions(&self) -> Option<(u32, u32)> {
        self.last_target
    }

    /// Get frames processed count.
    pub fn frames_processed(&self) -> u64 {
        self.frames_processed
    }

    /// Resolve what this buffer *is*, from what it says about itself.
    ///
    /// Errors rather than guessing. See the type-level docs for why.
    fn resolve_input(metadata: &Metadata) -> Result<(PixelFormat, u32, u32)> {
        let (width, height) = metadata.video_dims().ok_or_else(|| {
            Error::Element(
                "VideoScale: buffer carries no video dimensions — the upstream element must \
                 call Metadata::set_video_dims()"
                    .into(),
            )
        })?;

        // Defensive: video_dims() implies a VideoRaw format, which always
        // names a pixel format — this arm is unreachable since #160 retired
        // the format-free legacy keys, and stays as a guard.
        let format = metadata.video_pixel_format().ok_or_else(|| {
            Error::Element(format!(
                "VideoScale: buffer is {width}x{height} but carries no pixel format — the \
                 upstream element must call Metadata::set_video_dims(). The byte count \
                 cannot disambiguate it (I420 and NV12 are both w*h*3/2)."
            ))
        })?;

        Ok((format, width, height))
    }

    /// Build (or reuse) the engine for this exact conversion.
    ///
    /// Keyed on the format *and* both sizes, so a source resolution change, a
    /// retarget, or a mid-stream format change all rebuild it — and an unchanged
    /// stream costs one comparison per frame.
    fn ensure_engine(
        &mut self,
        format: converters::PixelFormat,
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
    ) -> Result<()> {
        let key = (format, src_w, src_h, dst_w, dst_h);
        if self.engine_key == Some(key) {
            return Ok(());
        }

        let engine =
            converters::ScaleEngine::new(src_w, src_h, dst_w, dst_h, format)?.with_mode(self.mode);

        tracing::info!(
            "VideoScale: {:?} {}x{} -> {}x{}{}",
            format,
            src_w,
            src_h,
            dst_w,
            dst_h,
            if self.engine_key.is_some() {
                " (renegotiated)"
            } else {
                ""
            }
        );

        self.engine = Some(engine);
        self.engine_key = Some(key);
        Ok(())
    }
}

impl Default for VideoScale {
    fn default() -> Self {
        Self::new()
    }
}

/// Every format the scaling engine can resize, at any resolution.
///
/// Deliberately a `List`, not `Any`: the caps vocabulary is wider than what the
/// engine handles (10-bit, I422, I444, …), and saying so is what lets
/// negotiation insert a `VideoConvertElement` in front of us for those.
///
/// Width and height are `Any` — this element *is* the answer to a geometry
/// mismatch, so it must not itself constrain geometry.
fn scalable_caps() -> ElementMediaCaps {
    let format = VideoFormatCaps {
        width: CapsValue::Any,
        height: CapsValue::Any,
        pixel_format: CapsValue::List(
            converters::PixelFormat::ALL
                .iter()
                .copied()
                .map(Into::into)
                .collect(),
        ),
        ..VideoFormatCaps::any()
    };

    ElementMediaCaps::new(vec![FormatMemoryCap::new(
        format.into(),
        MemoryCaps::cpu_only(),
    )])
}

impl crate::control::Controllable for VideoScale {
    type Control = ScaleControl;

    fn control(&self) -> ScaleControl {
        self.control.clone()
    }
}

impl Element for VideoScale {
    // #189: the same-size arm forwards the input buffer, and whether it
    // fires depends on the *stream's* geometry — unknowable at start, so
    // answer conservatively.
    fn passthrough(&self) -> bool {
        true
    }

    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // What is this buffer? Ask it — do not infer.
        let (caps_format, src_w, src_h) = Self::resolve_input(buffer.metadata())?;
        // Caps-only formats (I444, 10-bit, …) fail here with a message naming them.
        let format = converters::PixelFormat::try_from(caps_format)?;

        // Resolve the target against the *current* source, so an aspect-preserving
        // target survives a source resolution change.
        let (dst_w, dst_h) = self.control.resolve(src_w, src_h);

        self.frames_processed += 1;
        self.last_source = Some((src_w, src_h, caps_format));
        self.last_target = Some((dst_w, dst_h));

        // Passthrough. The buffer already describes itself correctly, so forward
        // it untouched — no engine, no arena, no copy.
        if (dst_w, dst_h) == (src_w, src_h) {
            return Ok(Some(buffer));
        }

        self.ensure_engine(format, src_w, src_h, dst_w, dst_h)?;
        let engine = self.engine.as_ref().expect("ensure_engine just built one");

        // Scale straight into the arena slot — the engine writes into any
        // caller slice, so a scratch staging buffer only added a redundant
        // full-frame memcpy (#140). Acquire-first is correct for a stateless
        // transform: on PoolExhausted the executor sheds this input either
        // way, so failing before the work is strictly better.
        let output_size = format.buffer_size(dst_w, dst_h);
        let mut slot = self.output.acquire(output_size, "videoscale")?;
        engine.scale(buffer.as_bytes(), &mut slot.data_mut()[..output_size])?;

        let mut metadata = buffer.metadata().clone();
        // The INPUT format, not a hardcoded I420: scaling preserves the format.
        // And the size we actually produced — without this the downstream encoder
        // keeps encoding at the old geometry and the resize never takes effect.
        metadata.set_video_dims(dst_w, dst_h, caps_format);

        Ok(Some(Buffer::new(
            MemoryHandle::with_len(slot, output_size),
            metadata,
        )))
    }

    fn input_media_caps(&self) -> ElementMediaCaps {
        scalable_caps()
    }

    fn output_media_caps(&self) -> ElementMediaCaps {
        scalable_caps()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::MediaFormat;
    use crate::memory::SharedArena;

    /// A raw frame of `format` at `w`x`h`, describing itself the way any real
    /// source or decoder now does.
    fn frame(format: PixelFormat, w: u32, h: u32) -> Buffer {
        let size = converters::PixelFormat::try_from(format)
            .unwrap()
            .buffer_size(w, h);
        frame_with_bytes(format, w, h, &vec![0x40; size])
    }

    /// A raw frame with an exact payload.
    fn frame_with_bytes(format: PixelFormat, w: u32, h: u32, bytes: &[u8]) -> Buffer {
        let arena = SharedArena::new(bytes.len().max(64), 8).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..bytes.len()].copy_from_slice(bytes);

        let mut metadata = Metadata::new();
        metadata.set_video_dims(w, h, format);
        Buffer::new(MemoryHandle::with_len(slot, bytes.len()), metadata)
    }

    fn scale_once(scaler: &mut VideoScale, buffer: Buffer) -> Buffer {
        scaler.process(buffer).unwrap().expect("an output frame")
    }

    // ------------------------------------------------------------------
    // #36 — the element used to treat every payload as planar I420
    // ------------------------------------------------------------------

    /// The headline regression. An RGB24 640x480 frame is 921 600 bytes, which
    /// sails past the old code's I420 size floor (460 800); its first 460 800
    /// bytes were then bilinear-scaled as if they were Y/U/V planes and emitted
    /// stamped `I420`. Garbage, no error, no way to find out but to look at the
    /// picture.
    ///
    /// Scale a frame of pure primaries with nearest-neighbour so no interpolation
    /// can muddy the check: every output pixel must still *be* one of the input
    /// primaries, with its channel triplet intact.
    #[test]
    fn rgb24_is_scaled_as_rgb_not_as_yuv_planes() {
        const RED: [u8; 3] = [255, 0, 0];
        const GREEN: [u8; 3] = [0, 255, 0];
        const BLUE: [u8; 3] = [0, 0, 255];
        const WHITE: [u8; 3] = [255, 255, 255];

        // 4x2 RGB24: two rows of [R G B W]
        let mut bytes = Vec::new();
        for _ in 0..2 {
            for px in [RED, GREEN, BLUE, WHITE] {
                bytes.extend_from_slice(&px);
            }
        }

        let mut scaler = VideoScale::new().with_mode(ScaleMode::NearestNeighbor);
        scaler.control().set_target(2, 2);

        let out = scale_once(
            &mut scaler,
            frame_with_bytes(PixelFormat::Rgb24, 4, 2, &bytes),
        );

        assert_eq!(out.len(), 3 * 2 * 2, "output must be RGB24-sized, not I420");
        for px in out.as_bytes().chunks_exact(3) {
            assert!(
                [RED, GREEN, BLUE, WHITE].contains(&[px[0], px[1], px[2]]),
                "output pixel {px:?} is not one of the input primaries — the bytes \
                 were resampled as something other than RGB"
            );
        }
    }

    /// Scaling preserves the pixel format. The old code hardcoded `I420` into the
    /// output metadata regardless of what went in, so a downstream element was
    /// told a lie about bytes it could not otherwise identify.
    #[test]
    fn output_metadata_carries_the_input_pixel_format() {
        for format in [PixelFormat::Rgb24, PixelFormat::Rgba, PixelFormat::Nv12] {
            let mut scaler = VideoScale::new();
            scaler.control().set_target(8, 8);

            let out = scale_once(&mut scaler, frame(format, 16, 16));
            assert_eq!(
                out.metadata().video_pixel_format(),
                Some(format),
                "{format:?} came out labelled as something else"
            );
            assert_eq!(out.metadata().video_dims(), Some((8, 8)));
        }
    }

    /// Every format the engine handles must survive a resize with the right
    /// output length. (Structure, not fidelity: the YUYV/UYVY path is a
    /// deliberately simplified resampler.)
    #[test]
    fn every_engine_format_can_be_scaled() {
        for engine_format in converters::PixelFormat::ALL {
            let format: PixelFormat = engine_format.into();

            let mut scaler = VideoScale::new();
            scaler.control().set_target(4, 4);

            let out = scale_once(&mut scaler, frame(format, 8, 8));
            assert_eq!(
                out.len(),
                engine_format.buffer_size(4, 4),
                "{format:?} produced the wrong number of bytes"
            );
            assert_eq!(out.metadata().video_pixel_format(), Some(format));
        }
    }

    // ------------------------------------------------------------------
    // Geometry travels in-band: refuse to guess
    // ------------------------------------------------------------------

    /// The retired legacy `"width"`/`"height"` custom keys carry no geometry
    /// any more (#160): a buffer wearing only them is a buffer without
    /// dimensions, and the error says how to fix it. (Dims-without-format is
    /// unrepresentable now — `MediaFormat::VideoRaw` always names one.)
    #[test]
    fn legacy_custom_keys_no_longer_count_as_dimensions() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let mut metadata = Metadata::new();
        metadata.set("width", 16u64);
        metadata.set("height", 16u64);
        let buffer = Buffer::new(MemoryHandle::with_len(slot, 384), metadata);

        let err = VideoScale::new().process(buffer).unwrap_err().to_string();
        assert!(
            err.contains("no video dimensions"),
            "unhelpful error: {err}"
        );
        assert!(err.contains("set_video_dims"), "unhelpful error: {err}");
    }

    #[test]
    fn a_buffer_without_dimensions_errors() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let buffer = Buffer::new(MemoryHandle::with_len(slot, 384), Metadata::new());

        let err = VideoScale::new().process(buffer).unwrap_err().to_string();
        assert!(
            err.contains("no video dimensions"),
            "unhelpful error: {err}"
        );
    }

    /// The caps vocabulary is wider than the engine: I444 can be *named* but not
    /// resampled. Say so, naming the format, rather than corrupting it.
    #[test]
    fn a_caps_only_format_errors() {
        let arena = SharedArena::new(4096, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let mut metadata = Metadata::new();
        metadata.set_video_dims(16, 16, PixelFormat::I444);
        let buffer = Buffer::new(MemoryHandle::with_len(slot, 768), metadata);

        let mut scaler = VideoScale::new();
        scaler.control().set_target(8, 8);

        let err = scaler.process(buffer).unwrap_err().to_string();
        assert!(
            err.contains("I444"),
            "the error must name the format: {err}"
        );
    }

    // ------------------------------------------------------------------
    // Passthrough and engine caching
    // ------------------------------------------------------------------

    /// With no target set, the buffer already describes itself correctly, so it
    /// is forwarded untouched — no engine, no arena, no copy.
    #[test]
    fn passthrough_forwards_the_buffer_untouched() {
        let mut scaler = VideoScale::new();
        let input = frame(PixelFormat::I420, 16, 16);
        let expected = input.as_bytes().to_vec();

        let out = scale_once(&mut scaler, input);

        assert_eq!(out.as_bytes(), &expected[..]);
        assert_eq!(out.metadata().video_dims(), Some((16, 16)));
        assert!(
            !scaler.output.is_built(),
            "passthrough must not allocate an output arena"
        );
        assert!(
            scaler.engine.is_none(),
            "passthrough must not build an engine"
        );
    }

    #[test]
    fn unchanged_geometry_reuses_the_cached_engine() {
        let mut scaler = VideoScale::new();
        scaler.control().set_target(8, 8);

        scale_once(&mut scaler, frame(PixelFormat::I420, 16, 16));
        let key = scaler.engine_key;
        assert!(key.is_some());

        scale_once(&mut scaler, frame(PixelFormat::I420, 16, 16));
        assert_eq!(scaler.engine_key, key, "the engine was needlessly rebuilt");
    }

    /// A mid-stream *format* change must rebuild the engine, not just a size
    /// change — the engine is built per (format, src, dst).
    #[test]
    fn a_pixel_format_change_mid_stream_rebuilds_the_engine() {
        let mut scaler = VideoScale::new();
        scaler.control().set_target(8, 8);

        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 16, 16));
        assert_eq!(out.metadata().video_pixel_format(), Some(PixelFormat::I420));

        let out = scale_once(&mut scaler, frame(PixelFormat::Rgb24, 16, 16));
        assert_eq!(
            out.metadata().video_pixel_format(),
            Some(PixelFormat::Rgb24)
        );
        assert_eq!(out.len(), 3 * 8 * 8, "now producing RGB, not I420");
    }

    // ------------------------------------------------------------------
    // ScaleControl (unchanged behaviour, ported to the new constructor)
    // ------------------------------------------------------------------

    #[test]
    fn source_dimensions_come_from_buffer_metadata() {
        let mut scaler = VideoScale::new();
        assert_eq!(
            scaler.src_dimensions(),
            None,
            "before the first buffer, the scaler does not know its geometry"
        );

        scale_once(&mut scaler, frame(PixelFormat::I420, 320, 240));
        assert_eq!(scaler.src_dimensions(), Some((320, 240)));
    }

    #[test]
    fn retarget_takes_effect_on_the_next_frame() {
        let mut scaler = VideoScale::new();
        let control = scaler.control();

        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 320, 240));
        assert_eq!(out.metadata().video_dims(), Some((320, 240)));

        control.set_target(160, 120);
        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 320, 240));
        assert_eq!(out.metadata().video_dims(), Some((160, 120)));
    }

    #[test]
    fn max_height_preserves_aspect_ratio() {
        let mut scaler = VideoScale::new();
        scaler.control().set_max_height(120);

        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 640, 480));
        assert_eq!(out.metadata().video_dims(), Some((160, 120)));
    }

    #[test]
    fn max_height_follows_a_source_resolution_change() {
        let mut scaler = VideoScale::new();
        scaler.control().set_max_height(120);

        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 640, 480));
        assert_eq!(out.metadata().video_dims(), Some((160, 120)));

        // 16:9 source now — the width must follow, not stay at 160.
        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 640, 360));
        assert_eq!(out.metadata().video_dims(), Some((214, 120)));
    }

    #[test]
    fn never_upscales() {
        let mut scaler = VideoScale::new();
        scaler.control().set_max_height(1080);

        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 320, 240));
        assert_eq!(
            out.metadata().video_dims(),
            Some((320, 240)),
            "a bound above the source means 'fits already', not 'invent detail'"
        );
    }

    #[test]
    fn passthrough_restores_the_source_resolution() {
        let mut scaler = VideoScale::new();
        let control = scaler.control();
        control.set_max_height(120);

        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 640, 480));
        assert_eq!(out.metadata().video_dims(), Some((160, 120)));

        control.passthrough();
        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 640, 480));
        assert_eq!(out.metadata().video_dims(), Some((640, 480)));
    }

    #[test]
    fn odd_targets_round_down_to_even() {
        let control = ScaleControl::new();
        control.set_target(101, 77);
        assert_eq!(control.target(), (100, 76));
    }

    // ---- resolve(): the published geometry contract -----------------------
    //
    // The tests above drive `process()`, which is what proves the element
    // really uses `resolve`. These pin the arithmetic itself, so a consumer
    // advertising geometry ahead of the frames can rely on it.

    #[test]
    fn resolve_rounds_the_derived_axis_up_before_evening_it_down() {
        // The witness from #86: a hand-mirrored `floor` gives 852 here, and
        // the frames then arrive at 854.
        let control = ScaleControl::new();
        control.set_max_height(480);
        assert_eq!(control.resolve(1280, 720), (854, 480));
    }

    #[test]
    fn resolve_matches_what_the_element_produces() {
        let control = ScaleControl::new();
        control.set_max_height(120);

        let mut scaler = VideoScale::new();
        scaler.control().set_max_height(120);

        for (w, h) in [(640, 480), (640, 360), (1280, 720), (100, 100)] {
            let out = scale_once(&mut scaler, frame(PixelFormat::I420, w, h));
            assert_eq!(
                Some(control.resolve(w, h)),
                out.metadata().video_dims(),
                "resolve disagrees with process() for {w}x{h}"
            );
        }
    }

    #[test]
    fn resolve_bounds_width_as_the_mirror_of_height() {
        let control = ScaleControl::new();
        control.set_max_width(854);
        assert_eq!(control.resolve(1280, 720), (854, 480));

        // ...and never upscales, exactly like the height bound.
        assert_eq!(control.resolve(320, 180), (320, 180));
    }

    #[test]
    fn resolve_passes_through_an_unset_target() {
        assert_eq!(ScaleControl::new().resolve(1280, 720), (1280, 720));
    }

    #[test]
    fn resolve_honours_an_explicit_target_in_both_directions() {
        let control = ScaleControl::new();
        control.set_target(1920, 1080);
        assert_eq!(
            control.resolve(640, 360),
            (1920, 1080),
            "an explicit target upscales; only a bound refuses to"
        );
        assert_eq!(control.resolve(3840, 2160), (1920, 1080));
    }

    #[test]
    fn a_target_below_two_is_indistinguishable_from_passthrough() {
        // Both axes round down to even *on store*, so `set_target(1, 1)` writes
        // (0, 0) — which is the passthrough sentinel. Surprising, but it is the
        // shipped behaviour and a consumer predicting geometry must know it.
        let control = ScaleControl::new();
        control.set_target(1, 1);
        assert_eq!(control.target(), (0, 0));
        assert_eq!(control.resolve(1920, 1080), (1920, 1080));
    }

    #[test]
    fn resolve_never_collapses_the_derived_axis() {
        // An extreme bound must still yield a codable frame, not a zero axis.
        let control = ScaleControl::new();
        control.set_max_height(2);
        assert_eq!(control.resolve(1920, 1080), (4, 2));

        control.set_max_width(2);
        assert_eq!(control.resolve(1920, 1080), (2, 2));
    }

    #[test]
    fn scaled_output_is_stamped_with_the_new_geometry() {
        let mut scaler = VideoScale::new();
        scaler.control().set_target(160, 120);

        let out = scale_once(&mut scaler, frame(PixelFormat::I420, 320, 240));
        let meta = out.metadata();

        // MediaFormat::VideoRaw is the single geometry representation (#160).
        assert!(matches!(
            meta.format,
            Some(MediaFormat::VideoRaw(vf)) if vf.width == 160 && vf.height == 120
        ));
        assert_eq!(meta.video_dims(), Some((160, 120)));
    }

    #[test]
    fn scale_mode_defaults_to_bilinear() {
        assert_eq!(ScaleMode::default(), ScaleMode::Bilinear);
    }

    #[test]
    fn frames_processed_counts_every_frame_including_passthrough() {
        let mut scaler = VideoScale::new();
        scale_once(&mut scaler, frame(PixelFormat::I420, 16, 16));
        scaler.control().set_target(8, 8);
        scale_once(&mut scaler, frame(PixelFormat::I420, 16, 16));

        assert_eq!(scaler.frames_processed(), 2);
    }
}
