//! Video format conversion element.
//!
//! Converts between pixel formats (e.g., YUYV -> RGBA for display).

use crate::buffer::{Buffer, MemoryHandle};
use crate::converters::{PixelFormat, VideoConvert};
use crate::element::Element;
use crate::error::{Error, Result};
use crate::format::{Caps, PlaneLayout};
use crate::memory::MemoryType;
use crate::memory::{OutputArena, OutputBudget, defaults};
use crate::metadata::Metadata;

/// Video format conversion element.
///
/// This element converts video frames between pixel formats. It's commonly
/// used to convert camera output (YUYV) to display format (RGBA).
///
/// # Auto-detection
///
/// If input format is not specified, the element will try to auto-detect
/// based on buffer size and common V4L2 formats.
///
/// # Example
///
/// ```rust,ignore
/// // Convert YUYV (640x480) to RGBA
/// let element = VideoConvertElement::new()
///     .with_input_format(PixelFormat::Yuyv)
///     .with_output_format(PixelFormat::Rgba)
///     .with_size(640, 480);
/// ```
pub struct VideoConvertElement {
    /// Input pixel format (None = auto-detect)
    input_format: Option<PixelFormat>,
    /// Output pixel format
    output_format: PixelFormat,
    /// Frame width (0 = auto-detect)
    width: u32,
    /// Frame height (0 = auto-detect)
    height: u32,
    /// Cached converter (rebuilt whenever [`Self::converter_key`] changes)
    converter: Option<VideoConvert>,
    /// The (input format, output format, width, height) the cached converter
    /// was built for. A converter is only valid for one of these — caching it
    /// unconditionally is what made mid-stream format changes impossible.
    converter_key: Option<(PixelFormat, PixelFormat, u32, u32)>,
    /// Element name
    name: String,
    /// Staging for a strided frame the engine cannot address by whole
    /// rows (#196) — see [`PlaneLayout::full_span_len`]. Stays empty for
    /// every producer that allocates whole rows, which is all of them.
    repack_scratch: Vec<u8>,
    /// Arena for output buffers
    output: OutputArena,
}

impl VideoConvertElement {
    /// Create a new video convert element with default settings.
    ///
    /// Defaults to RGBA output (most common for display).
    pub fn new() -> Self {
        Self {
            input_format: None,
            output_format: PixelFormat::Rgba,
            width: 0,
            height: 0,
            converter: None,
            converter_key: None,
            name: "videoconvert".to_string(),
            repack_scratch: Vec::new(),
            // `SharedArena::new` aligns every slot to a cache line, which is
            // stronger than the 32 bytes the AVX paths need — this used to ask
            // for `new_avx` and got *less* alignment than it does now.
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT).grow_to_fit(),
        }
    }

    /// Set the input pixel format.
    pub fn with_input_format(mut self, format: PixelFormat) -> Self {
        self.input_format = Some(format);
        self
    }

    /// Set the output pixel format.
    pub fn with_output_format(mut self, format: PixelFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set the frame dimensions.
    pub fn with_size(mut self, width: u32, height: u32) -> Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Try to detect input format from buffer size.
    fn detect_format(&self, buffer_size: usize) -> Option<(PixelFormat, u32, u32)> {
        // Common resolutions to try
        let resolutions = [
            (640, 480),
            (1280, 720),
            (1920, 1080),
            (320, 240),
            (800, 600),
            (1024, 768),
            (1280, 960),
            (352, 288),
            (176, 144),
        ];

        // Try YUYV first (most common V4L2 format)
        for (w, h) in resolutions {
            if PixelFormat::Yuyv.buffer_size(w, h) == buffer_size {
                return Some((PixelFormat::Yuyv, w, h));
            }
        }

        // Try RGB24
        for (w, h) in resolutions {
            if PixelFormat::Rgb24.buffer_size(w, h) == buffer_size {
                return Some((PixelFormat::Rgb24, w, h));
            }
        }

        // Try RGBA
        for (w, h) in resolutions {
            if PixelFormat::Rgba.buffer_size(w, h) == buffer_size {
                return Some((PixelFormat::Rgba, w, h));
            }
        }

        None
    }

    /// Resolve the input format and dimensions for a buffer.
    ///
    /// Buffer metadata wins (a decoder or scaler upstream knows what it just
    /// produced), then the constructor-configured values, then size-based
    /// auto-detection.
    fn resolve_input(
        &self,
        metadata: &Metadata,
        input_size: usize,
    ) -> Result<(PixelFormat, u32, u32)> {
        let format = metadata
            .video_pixel_format()
            .and_then(|pf| PixelFormat::try_from(pf).ok())
            .or(self.input_format);
        let dims = metadata
            .video_dims()
            .or((self.width > 0 && self.height > 0).then_some((self.width, self.height)));

        match (format, dims) {
            (Some(format), Some((width, height))) => Ok((format, width, height)),
            // Format known, dimensions not: only a buffer-size match can
            // recover them, and only for the sizes detect_format knows.
            (Some(format), None) => self
                .detect_format(input_size)
                .filter(|(detected, _, _)| *detected == format)
                .map(|(_, w, h)| (format, w, h))
                .ok_or_else(|| {
                    Error::Element(format!(
                        "Cannot determine dimensions for format {format:?} with buffer size {input_size}"
                    ))
                }),
            (None, _) => self.detect_format(input_size).ok_or_else(|| {
                Error::Element(format!(
                    "Cannot auto-detect video format for buffer size {input_size}"
                ))
            }),
        }
    }

    /// Build the converter for this input, reusing the cached one when the
    /// format and dimensions are unchanged (the common case, once per frame).
    fn ensure_converter(
        &mut self,
        input_format: PixelFormat,
        width: u32,
        height: u32,
    ) -> Result<()> {
        let key = (input_format, self.output_format, width, height);
        if self.converter_key == Some(key) {
            return Ok(());
        }

        let converter = VideoConvert::new(input_format, self.output_format, width, height)?;

        tracing::info!(
            "VideoConvert: {:?} {}x{} -> {:?} {}x{}{}",
            input_format,
            width,
            height,
            self.output_format,
            width,
            height,
            if self.converter_key.is_some() {
                " (renegotiated)"
            } else {
                ""
            }
        );

        self.width = width;
        self.height = height;
        self.input_format = Some(input_format);
        self.converter = Some(converter);
        self.converter_key = Some(key);

        Ok(())
    }
}

impl Default for VideoConvertElement {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for VideoConvertElement {
    // #189: the no-op arm forwards the input buffer, so unless both formats
    // are pinned unequal the upstream budget must accumulate through here.
    fn passthrough(&self) -> bool {
        match self.input_format {
            Some(input) => input == self.output_format,
            None => true,
        }
    }

    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input_data = buffer.as_bytes();

        tracing::debug!(
            "VideoConvert: received buffer with {} bytes",
            input_data.len()
        );

        // Resolve per buffer, not once: an upstream scaler or a renegotiating
        // camera can change the format mid-stream, and the converter must
        // follow it rather than keep converting at the first frame's geometry.
        let (input_format, width, height) =
            self.resolve_input(buffer.metadata(), input_data.len())?;
        let caps_format: crate::format::PixelFormat = input_format.into();

        // Plane geometry comes from the buffer and only from the buffer
        // (#194): `plane_layout()` answers the packed layout for an ordinary
        // frame and the producer's real strides for a codec-owned one. The
        // `unwrap_or_else` covers the auto-detected case, where `resolve_input`
        // recovered geometry the metadata never declared.
        let layout = buffer
            .metadata()
            .plane_layout()
            .unwrap_or_else(|| PlaneLayout::packed(caps_format, width, height));

        // No-op conversion: forward the input untouched (the VideoScale /
        // AudioDownmix passthrough precedent) — zero copies, zero slots.
        //
        // Only when the input is already what our *output* caps promise,
        // though. Since #196 the input side accepts External and strided
        // frames, and the output side is still packed CPU: forwarding one
        // across a link negotiated `Cpu` would hand the consumer a layout it
        // never agreed to read.
        if input_format == self.output_format {
            if buffer.memory_type() == MemoryType::Cpu && !buffer.metadata().has_strided_planes() {
                return Ok(Some(buffer));
            }
            let packed_len = PlaneLayout::packed(caps_format, width, height).required_len(
                caps_format,
                width,
                height,
            );
            let mut slot = self.output.acquire(packed_len, "videoconvert")?;
            layout
                .repack_into(
                    input_data,
                    caps_format,
                    width,
                    height,
                    &mut slot.data_mut()[..packed_len],
                )
                .map_err(Error::Element)?;
            let mut metadata = buffer.metadata().clone();
            metadata.set_video_dims(width, height, caps_format);
            return Ok(Some(Buffer::new(
                MemoryHandle::with_len(slot, packed_len),
                metadata,
            )));
        }

        self.ensure_converter(input_format, width, height)?;

        // Convert straight into the arena slot — every converter arm writes
        // into the caller's `&mut [u8]`, so a scratch staging buffer would
        // only add a redundant full-frame memcpy (#140). Acquiring before
        // converting is correct for a stateless transform: on PoolExhausted
        // the executor sheds this input either way, so failing first just
        // skips the wasted work.
        let output_size = self.output_format.buffer_size(width, height);
        let mut slot = self.output.acquire(output_size, "videoconvert")?;

        // Every conversion arm addresses planes by stride, so a strided
        // frame converts in place — provided each plane is addressable for
        // its whole `stride * rows` (#196). One that ends tight against its
        // last row is repacked first; real strided producers allocate past
        // it, so this stays cold.
        let (data, layout) = if layout.is_packed(caps_format, width, height)
            || input_data.len() >= layout.full_span_len(caps_format, width, height)
        {
            (input_data, layout)
        } else {
            let packed = PlaneLayout::packed(caps_format, width, height);
            let packed_len = packed.required_len(caps_format, width, height);
            self.repack_scratch.resize(packed_len, 0);
            layout
                .repack_into(
                    input_data,
                    caps_format,
                    width,
                    height,
                    &mut self.repack_scratch,
                )
                .map_err(Error::Element)?;
            (&self.repack_scratch[..], packed)
        };

        let converter = self.converter.as_ref().expect("installed above");
        converter.convert(data, layout, &mut slot.data_mut()[..output_size])?;

        // Size the handle by the converted data, not by the arena slot: the
        // slot is reused across geometry changes and is only ever >= the frame.
        let handle = MemoryHandle::with_len(slot, output_size);
        let mut metadata = buffer.metadata().clone();
        // Describe what we actually produced. Conversion preserves geometry, so
        // only the pixel format changes — but the input's dimension metadata
        // must be carried forward in both representations, or a downstream
        // encoder and AutoVideoSink can disagree about the frame size.
        metadata.set_video_dims(width, height, self.output_format.into());
        let output = Buffer::new(handle, metadata);

        Ok(Some(output))
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        Ok(None)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn input_caps(&self) -> Caps {
        // Accept any raw video format (will convert from input to output format)
        Caps::video_raw_any()
    }

    fn output_caps(&self) -> Caps {
        // Output is always the configured output format
        // Use 0x0 dimensions since we don't know them until we process the first frame
        Caps::video_raw_any_resolution(self.output_format.into())
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Accept any raw video format - truly any dimensions and pixel format.
        //
        // #196: also accept External (producer-owned, strided) memory. The
        // contract `external_or_cpu` carries is "reads geometry via
        // Metadata::plane_layout", which `process` now does — so a decoder
        // hands its own pictures straight over instead of de-striding them
        // into an arena first.
        //
        // No `MemoryLayout::AVX` request on the input any more: alignment is
        // something an arena can promise and a codec's own frame cannot. It
        // was never enforced (caps intersection *merges* layouts rather than
        // intersecting them), so this only stops advertising a guarantee we
        // do not get; the output side still asks for it.
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, MemoryLayout,
            VideoFormatCaps,
        };

        let format = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Any,
            framerate: CapsValue::Any,
            layout: MemoryLayout::NONE,
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(format),
            MemoryCaps::external_or_cpu(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Output is the configured output format with any dimensions
        // Produce AVX-aligned output for downstream SIMD elements
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, MemoryLayout,
            VideoFormatCaps,
        };

        let format = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Fixed(self.output_format.into()),
            framerate: CapsValue::Any,
            layout: MemoryLayout::AVX, // Produce aligned output
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(format),
            MemoryCaps::cpu_only(),
        )])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SharedArena;

    #[test]
    fn test_detect_yuyv_640x480() {
        let element = VideoConvertElement::new();
        let size = PixelFormat::Yuyv.buffer_size(640, 480);
        let detected = element.detect_format(size);
        assert_eq!(detected, Some((PixelFormat::Yuyv, 640, 480)));
    }

    #[test]
    fn test_detect_yuyv_1280x720() {
        let element = VideoConvertElement::new();
        let size = PixelFormat::Yuyv.buffer_size(1280, 720);
        let detected = element.detect_format(size);
        assert_eq!(detected, Some((PixelFormat::Yuyv, 1280, 720)));
    }

    /// A raw frame of `format` at `w`x`h`, carrying its geometry in metadata.
    fn frame(format: crate::format::PixelFormat, w: u32, h: u32) -> Buffer {
        let size = PixelFormat::try_from(format).unwrap().buffer_size(w, h);
        let arena = SharedArena::new(size, 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..size].fill(0x40);

        let mut metadata = Metadata::new();
        metadata.set_video_dims(w, h, format);
        Buffer::new(MemoryHandle::with_len(slot, size), metadata)
    }

    /// A strided frame whose buffer ends tight against its last row still
    /// converts correctly: the engine refuses it (its arms address whole
    /// rows), so the element repacks first.
    ///
    /// The engine's fast path is the common case — real strided producers
    /// allocate whole rows — but "uncommon" must not mean "wrong".
    #[test]
    fn a_tight_trailing_row_falls_back_to_the_repack_and_stays_correct() {
        use crate::converters::testutil::strided_twin;
        use crate::format::PixelFormat as Caps;
        const W: u32 = 32;
        const H: u32 = 24;

        let packed: Vec<u8> = (0..PixelFormat::I420.buffer_size(W, H))
            .map(|i| (i % 251) as u8)
            .collect();
        let (strided, layout) = strided_twin(&packed, Caps::I420, W, H, 11);
        let tight = layout.required_len(Caps::I420, W, H);
        assert!(
            tight < strided.len(),
            "test premise: the twin has a padded tail"
        );

        // The same frame, minus that tail.
        let arena = SharedArena::new(tight, 2).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..tight].copy_from_slice(&strided[..tight]);
        let mut metadata = Metadata::new();
        metadata.set_video_planes(W, H, Caps::I420, layout);
        let buffer = Buffer::new(MemoryHandle::with_len(slot, tight), metadata);

        let mut element = VideoConvertElement::new().with_output_format(PixelFormat::Bgra);
        let out = element.process(buffer).unwrap().expect("a converted frame");

        let reference = VideoConvert::new(PixelFormat::I420, PixelFormat::Bgra, W, H).unwrap();
        let mut want = vec![0u8; reference.output_size()];
        reference
            .convert(&packed, reference.packed_input_layout(), &mut want)
            .unwrap();
        assert_eq!(out.as_bytes(), want.as_slice());
    }

    fn rgb_to_i420() -> VideoConvertElement {
        VideoConvertElement::new().with_output_format(PixelFormat::I420)
    }

    /// The regression: the converter used to be cached on the first frame and
    /// never rebuilt, so a resolution change mid-stream was either mis-converted
    /// or rejected on a length check.
    #[test]
    fn resolution_change_mid_stream_rebuilds_the_converter() {
        let mut element = rgb_to_i420();
        use crate::format::PixelFormat as Caps;

        let big = element
            .process(frame(Caps::Rgb24, 640, 480))
            .unwrap()
            .expect("640x480 converts");
        assert_eq!(
            big.as_bytes().len(),
            PixelFormat::I420.buffer_size(640, 480)
        );

        let small = element
            .process(frame(Caps::Rgb24, 320, 240))
            .unwrap()
            .expect("320x240 must convert too, not reuse the 640x480 converter");
        assert_eq!(
            small.as_bytes().len(),
            PixelFormat::I420.buffer_size(320, 240)
        );
        assert_eq!(small.metadata().video_dims(), Some((320, 240)));
    }

    #[test]
    fn pixel_format_change_mid_stream_rebuilds_the_converter() {
        let mut element = rgb_to_i420();
        use crate::format::PixelFormat as Caps;

        element.process(frame(Caps::Rgb24, 640, 480)).unwrap();
        let out = element
            .process(frame(Caps::Rgba, 640, 480))
            .unwrap()
            .expect("a format change must rebuild the converter");
        assert_eq!(
            out.as_bytes().len(),
            PixelFormat::I420.buffer_size(640, 480)
        );
        assert_eq!(
            element.converter_key,
            Some((PixelFormat::Rgba, PixelFormat::I420, 640, 480))
        );
    }

    #[test]
    fn unchanged_geometry_reuses_the_cached_converter() {
        // The hot path: same format and size every frame must not rebuild.
        let mut element = rgb_to_i420();
        use crate::format::PixelFormat as Caps;

        element.process(frame(Caps::Rgb24, 640, 480)).unwrap();
        let key = element.converter_key;
        element.process(frame(Caps::Rgb24, 640, 480)).unwrap();

        assert_eq!(element.converter_key, key, "converter must be reused");
        assert_eq!(key, Some((PixelFormat::Rgb24, PixelFormat::I420, 640, 480)));
    }

    #[test]
    fn output_describes_what_was_produced() {
        let mut element = rgb_to_i420();
        let out = element
            .process(frame(crate::format::PixelFormat::Rgb24, 640, 480))
            .unwrap()
            .unwrap();

        assert_eq!(
            out.metadata().video_pixel_format(),
            Some(crate::format::PixelFormat::I420),
            "output metadata must carry the OUTPUT format, not the input's"
        );
        assert_eq!(out.metadata().video_dims(), Some((640, 480)));
    }

    /// The exact element the zensight sensor builds for a YUYV camera
    /// (`pipeline.rs:333`). It used to fail on the first buffer with
    /// "Unsupported conversion: Yuyv -> I420", so no non-MJPG webcam worked.
    #[test]
    fn yuyv_camera_can_feed_an_i420_encoder() {
        let mut element = VideoConvertElement::new()
            .with_input_format(PixelFormat::Yuyv)
            .with_output_format(PixelFormat::I420)
            .with_size(640, 480);

        let out = element
            .process(frame(crate::format::PixelFormat::Yuyv, 640, 480))
            .unwrap()
            .expect("a YUYV webcam must be able to reach the encoder");

        assert_eq!(
            out.as_bytes().len(),
            PixelFormat::I420.buffer_size(640, 480)
        );
        assert_eq!(
            out.metadata().video_pixel_format(),
            Some(crate::format::PixelFormat::I420)
        );
    }

    #[test]
    fn buffers_without_metadata_still_auto_detect() {
        // Sources that stamp no format at all keep working (unchanged behaviour).
        // Auto-detection guesses YUYV for this size, which only converts to RGB.
        let mut element = VideoConvertElement::new().with_output_format(PixelFormat::Rgb24);
        let size = PixelFormat::Yuyv.buffer_size(640, 480);
        let arena = SharedArena::new(size, 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..size].fill(0x40);
        let buffer = Buffer::new(MemoryHandle::with_len(slot, size), Metadata::new());

        let out = element.process(buffer).unwrap().expect("auto-detected");
        assert_eq!(
            out.as_bytes().len(),
            PixelFormat::Rgb24.buffer_size(640, 480)
        );
    }
}
