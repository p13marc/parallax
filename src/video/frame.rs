//! Reading a raw-video buffer's geometry, once, in one place.
//!
//! Every presenter needs the same four facts about a buffer — pixel format,
//! width, height, and whether the producer declared real strides — and every
//! presenter needs them validated the same way, because the failure mode of
//! getting it wrong is a sheared picture rather than an error. [`FrameView`]
//! is that read; [`VideoFrame`] is that read plus the buffer it describes.

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use crate::format::{PixelFormat, PlaneLayout};
use crate::metadata::Metadata;

use super::fit::yuv420_len;

/// A raw-video buffer's geometry, validated against its byte length.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameView {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Payload layout. Only `I420`, `Nv12`, `Rgba` and `Bgra` are presentable.
    pub format: PixelFormat,
    /// Non-packed plane layout when the producer declared one (#194):
    /// External frames carry the decoder's real strides. `None` = packed.
    pub layout: Option<PlaneLayout>,
    /// What the stream said about its colour (#265). `None` = nothing said,
    /// and a presenter falls back to its own guess.
    pub color: Option<crate::format::ColorInfo>,
    /// Pixel aspect ratio and rotation (#266). `None` = square and upright.
    pub display: Option<crate::format::DisplayGeometry>,
}

impl FrameView {
    /// Read a frame's geometry from its metadata, checking it against the
    /// payload length.
    ///
    /// An undeclared pixel format is treated as `Rgba` — the legacy contract
    /// from before geometry rode in metadata. YUV frames **require** declared
    /// dimensions, because plane layout is unrecoverable without them; packed
    /// RGB falls back to guessing from the byte count, which is what keeps
    /// old capture pipelines working.
    pub fn from_metadata(meta: &Metadata, data_len: usize) -> Result<Self> {
        let format = meta.video_pixel_format().unwrap_or(PixelFormat::Rgba);

        // A strided frame (#194: External zero-copy from a decoder) declares
        // its real layout; the size gate must judge against THAT, not the
        // packed size — a strided frame is larger.
        let layout = meta
            .has_strided_planes()
            .then(|| meta.plane_layout())
            .flatten();

        let (width, height) = match format {
            PixelFormat::I420 | PixelFormat::Nv12 => match meta.video_dims() {
                Some((w, h)) if w > 0 && h > 0 => {
                    let need = layout
                        .map(|l| l.required_len(format, w, h))
                        .unwrap_or_else(|| yuv420_len(w, h));
                    if data_len < need {
                        return Err(Error::Element(format!(
                            "video: {format:?} buffer is {data_len} bytes but {w}x{h} needs {need}"
                        )));
                    }
                    (w, h)
                }
                Some((w, h)) => {
                    return Err(Error::Element(format!(
                        "video: {format:?} buffer declares degenerate geometry {w}x{h}"
                    )));
                }
                None => {
                    return Err(Error::Element(format!(
                        "video: {format:?} buffer carries no video dimensions \
                         (decoders set them via Metadata::set_video_dims)"
                    )));
                }
            },
            // Packed formats: trust declared dimensions when the payload is
            // big enough to hold them. The length is checked against the
            // format's own packing rather than a hardcoded four bytes per
            // pixel — `AutoVideoSink` could assume that because its caps
            // admit nothing else, but this read is shared now, and a 24-bit
            // frame silently failing the check falls through to the guess and
            // reports a plausible wrong size.
            _ => match meta.video_dims() {
                Some((w, h))
                    if w > 0
                        && h > 0
                        && data_len
                            >= PlaneLayout::packed(format, w, h).required_len(format, w, h) =>
                {
                    (w, h)
                }
                _ => detect_dimensions(data_len),
            },
        };

        Ok(Self {
            width,
            height,
            format,
            layout,
            color: meta.color_info(),
            display: meta.display_geometry(),
        })
    }
}

/// A decoded frame: the pixels, plus everything needed to interpret them.
///
/// Cloning one is two atomic increments and no copy — the payload stays in
/// the arena slot, or in the decoder's own picture, that the producer wrote.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Pixel data, arena- or externally-backed (#194).
    pub data: Buffer,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Payload layout.
    pub format: PixelFormat,
    /// Real strides when the producer declared them; `None` = packed.
    pub layout: Option<PlaneLayout>,
    /// What the stream said about its colour (#265).
    pub color: Option<crate::format::ColorInfo>,
    /// Pixel aspect ratio and rotation (#266). `None` = square and upright.
    pub display: Option<crate::format::DisplayGeometry>,
}

impl VideoFrame {
    /// Read a buffer's geometry and take a reference to its pixels.
    pub fn from_buffer(buffer: &Buffer) -> Result<Self> {
        let view = FrameView::from_metadata(buffer.metadata(), buffer.len())?;
        Ok(Self::new(buffer.clone(), view))
    }

    /// Pair an already-validated geometry with its buffer.
    pub fn new(data: Buffer, view: FrameView) -> Self {
        Self {
            data,
            width: view.width,
            height: view.height,
            format: view.format,
            layout: view.layout,
            color: view.color,
            display: view.display,
        }
    }

    /// The geometry alone.
    pub fn view(&self) -> FrameView {
        FrameView {
            width: self.width,
            height: self.height,
            format: self.format,
            layout: self.layout,
            color: self.color,
            display: self.display,
        }
    }

    /// How to present this frame: what the stream said, or square and upright.
    pub fn resolved_display(&self) -> crate::format::DisplayGeometry {
        self.display.unwrap_or_default()
    }

    /// The colour to render with: what the stream said, or the height
    /// heuristic when it said nothing.
    pub fn resolved_color(&self) -> crate::format::ColorInfo {
        let mut color = self.color.unwrap_or_default();
        color.matrix = color.matrix_or(super::color::matrix_for_height(self.height));
        color
    }
}

/// Guess a packed 4-byte-per-pixel frame's dimensions from its byte count.
///
/// Only reachable for RGBA/BGRA buffers that declared no geometry — a
/// pre-#160 producer. Common resolutions first, then a 4:3 assumption.
fn detect_dimensions(size: usize) -> (u32, u32) {
    let pixels = size / 4;

    const COMMON: &[(u32, u32)] = &[
        (640, 480),
        (800, 600),
        (1024, 768),
        (1280, 720),
        (1280, 960),
        (1920, 1080),
        (1920, 1200),
        (2560, 1440),
        (3840, 2160),
        (320, 240),
        (176, 144),
        (352, 288),
    ];

    for &(w, h) in COMMON {
        if (w * h) as usize == pixels {
            return (w, h);
        }
    }

    // Fallback: assume 4:3 aspect ratio
    let height = ((pixels as f64).sqrt() * 0.866) as u32; // sqrt(3/4)
    let width = pixels as u32 / height.max(1);
    (width.max(1), height.max(1))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn i420_meta(w: u32, h: u32) -> Metadata {
        let mut m = Metadata::default();
        m.set_video_dims(w, h, PixelFormat::I420);
        m
    }

    #[test]
    fn yuv_needs_declared_geometry() {
        let bare = Metadata::default();
        // No format at all reads as RGBA and guesses, so ask for I420
        // explicitly with the dimensions left out.
        let mut m = Metadata::default();
        m.set_video_dims(0, 0, PixelFormat::I420);
        assert!(FrameView::from_metadata(&m, 1024).is_err(), "degenerate");
        // RGBA with no dims falls back to the size guess rather than erroring.
        let view = FrameView::from_metadata(&bare, 1280 * 720 * 4).expect("rgba guesses");
        assert_eq!((view.width, view.height), (1280, 720));
        assert_eq!(view.format, PixelFormat::Rgba);
    }

    #[test]
    fn a_short_yuv_buffer_is_refused() {
        let m = i420_meta(1920, 1080);
        assert!(FrameView::from_metadata(&m, 1024).is_err());
        let view = FrameView::from_metadata(&m, yuv420_len(1920, 1080)).expect("exact fit");
        assert_eq!((view.width, view.height), (1920, 1080));
        assert_eq!(view.layout, None, "packed frames declare no layout");
    }

    #[test]
    fn declared_rgba_geometry_wins_when_the_length_agrees() {
        let mut m = Metadata::default();
        m.set_video_dims(100, 50, PixelFormat::Rgba);
        let view = FrameView::from_metadata(&m, 100 * 50 * 4).unwrap();
        assert_eq!((view.width, view.height), (100, 50));
        // A payload too short for the declared size falls back to the guess,
        // which is how a mis-declared buffer used to keep working.
        let view = FrameView::from_metadata(&m, 640 * 480 * 4).unwrap();
        assert_eq!((view.width, view.height), (100, 50), "still fits");
        let view = FrameView::from_metadata(&m, 8).unwrap();
        assert_ne!((view.width, view.height), (100, 50), "too short to be that");
    }

    #[test]
    fn a_packed_format_is_measured_by_its_own_packing() {
        // Rgb24 is three bytes per pixel. Checking against a hardcoded four
        // would fail here and report a plausible wrong size from the guess —
        // 320x240 came out as 278x207 before this was fixed.
        let mut m = Metadata::default();
        m.set_video_dims(320, 240, PixelFormat::Rgb24);
        let view = FrameView::from_metadata(&m, 320 * 240 * 3).unwrap();
        assert_eq!((view.width, view.height), (320, 240));
        assert_eq!(view.format, PixelFormat::Rgb24);
    }

    #[test]
    fn the_size_guess_knows_the_common_resolutions() {
        assert_eq!(detect_dimensions(1920 * 1080 * 4), (1920, 1080));
        assert_eq!(detect_dimensions(640 * 480 * 4), (640, 480));
        // Anything else lands on a plausible 4:3-ish shape rather than zero.
        let (w, h) = detect_dimensions(1000 * 4);
        assert!(w > 0 && h > 0);
    }
}
