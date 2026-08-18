//! Pixel format conversion (colorspace conversion).
//!
//! Provides YUV ↔ RGB conversions using standard color matrices (BT.601, BT.709).
//!
//! When the `simd-colorspace` feature is enabled, this module uses the `yuv` crate
//! for SIMD-accelerated conversions (AVX2, AVX-512, SSE4.1, NEON). Without the
//! feature, a pure Rust implementation is used as fallback.

use crate::error::{Error, Result};
use crate::format::PlaneLayout;

#[cfg(feature = "simd-colorspace")]
use yuv::{
    BufferStoreMut,
    YuvBiPlanarImage,
    YuvBiPlanarImageMut,
    YuvConversionMode,
    YuvPackedImage,
    YuvPlanarImage,
    YuvPlanarImageMut,
    YuvRange,
    YuvStandardMatrix,
    // RGB -> NV12 (bi-planar YUV 4:2:0)
    bgr_to_yuv_nv12,
    // RGB -> I420 (planar YUV 4:2:0)
    bgr_to_yuv420,
    bgra_to_yuv_nv12,
    bgra_to_yuv420,
    rgb_to_yuv_nv12,
    rgb_to_yuv420,
    rgba_to_yuv_nv12,
    rgba_to_yuv420,
    // UYVY (packed YUV 4:2:2) -> RGB
    uyvy422_to_rgb,
    uyvy422_to_rgba,
    // NV12 (bi-planar YUV 4:2:0) -> RGB
    yuv_nv12_to_bgr,
    yuv_nv12_to_bgra,
    yuv_nv12_to_rgb,
    yuv_nv12_to_rgba,
    // I420 (planar YUV 4:2:0) -> RGB
    yuv420_to_bgr,
    yuv420_to_bgra,
    yuv420_to_rgb,
    yuv420_to_rgba,
    // YUYV (packed YUV 4:2:2) -> RGB
    yuyv422_to_bgr,
    yuyv422_to_bgra,
    yuyv422_to_rgb,
    yuyv422_to_rgba,
};

/// Pixel format enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// Planar YUV 4:2:0 (Y plane, then U plane, then V plane)
    I420,
    /// Semi-planar YUV 4:2:0 (Y plane, then interleaved UV plane)
    Nv12,
    /// Packed YUV 4:2:2 (YUYV/YUY2) - common webcam format
    /// Layout: Y0 U0 Y1 V0 (4 bytes per 2 pixels)
    Yuyv,
    /// Packed YUV 4:2:2 (UYVY) - alternative webcam format
    /// Layout: U0 Y0 V0 Y1 (4 bytes per 2 pixels)
    Uyvy,
    /// Packed RGB, 3 bytes per pixel (R, G, B)
    Rgb24,
    /// Packed RGBA, 4 bytes per pixel (R, G, B, A)
    Rgba,
    /// Packed BGR, 3 bytes per pixel (B, G, R)
    Bgr24,
    /// Packed BGRA, 4 bytes per pixel (B, G, R, A)
    Bgra,
    /// Grayscale, 1 byte per pixel
    Gray8,
}

impl PixelFormat {
    /// Returns the number of bytes per pixel for packed formats.
    /// For planar formats and packed YUV 4:2:2, returns None (use buffer_size instead).
    pub fn bytes_per_pixel(&self) -> Option<usize> {
        match self {
            PixelFormat::I420 | PixelFormat::Nv12 => None,
            PixelFormat::Yuyv | PixelFormat::Uyvy => None, // 2 bytes per pixel average
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => Some(3),
            PixelFormat::Rgba | PixelFormat::Bgra => Some(4),
            PixelFormat::Gray8 => Some(1),
        }
    }

    /// Calculate total buffer size for given dimensions.
    pub fn buffer_size(&self, width: u32, height: u32) -> usize {
        let w = width as usize;
        let h = height as usize;
        match self {
            PixelFormat::I420 => w * h + 2 * (w / 2) * (h / 2), // Y + U + V
            PixelFormat::Nv12 => w * h + (w / 2) * (h / 2) * 2, // Y + UV interleaved
            PixelFormat::Yuyv | PixelFormat::Uyvy => w * h * 2, // 4 bytes per 2 pixels
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => w * h * 3,
            PixelFormat::Rgba | PixelFormat::Bgra => w * h * 4,
            PixelFormat::Gray8 => w * h,
        }
    }

    /// Returns true if this is a YUV format.
    pub fn is_yuv(&self) -> bool {
        matches!(
            self,
            PixelFormat::I420 | PixelFormat::Nv12 | PixelFormat::Yuyv | PixelFormat::Uyvy
        )
    }

    /// Returns true if this is an RGB format.
    pub fn is_rgb(&self) -> bool {
        matches!(
            self,
            PixelFormat::Rgb24 | PixelFormat::Rgba | PixelFormat::Bgr24 | PixelFormat::Bgra
        )
    }

    /// Every format the conversion and scaling engines can actually touch.
    ///
    /// The caps vocabulary ([`format::PixelFormat`](crate::format::PixelFormat))
    /// is deliberately wider than this — it can *name* 10-bit, 4:2:2 and 4:4:4
    /// planar formats that no engine here handles. This is the subset an
    /// element should advertise if it delegates to `converters`.
    pub const ALL: [PixelFormat; 9] = [
        PixelFormat::I420,
        PixelFormat::Nv12,
        PixelFormat::Yuyv,
        PixelFormat::Uyvy,
        PixelFormat::Rgb24,
        PixelFormat::Rgba,
        PixelFormat::Bgr24,
        PixelFormat::Bgra,
        PixelFormat::Gray8,
    ];

    /// Try to parse a V4L2 fourcc code into a PixelFormat.
    pub fn from_fourcc(fourcc: &[u8; 4]) -> Option<Self> {
        match fourcc {
            b"YUYV" | b"YUY2" => Some(PixelFormat::Yuyv),
            b"UYVY" => Some(PixelFormat::Uyvy),
            b"I420" | b"YU12" => Some(PixelFormat::I420),
            b"NV12" => Some(PixelFormat::Nv12),
            b"RGB3" => Some(PixelFormat::Rgb24),
            b"BGR3" => Some(PixelFormat::Bgr24),
            b"RGBP" | b"RGB4" => Some(PixelFormat::Rgba),
            b"GREY" | b"Y800" => Some(PixelFormat::Gray8),
            _ => None,
        }
    }
}

/// A caps pixel format that no conversion engine can handle.
///
/// [`format::PixelFormat`](crate::format::PixelFormat) has 15 variants because
/// caps must be able to *describe* what a device produces; `converters` has 9
/// because those are the ones there is code for. Converting between them is
/// therefore total in one direction and fallible in the other.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsupportedPixelFormat(pub crate::format::PixelFormat);

impl std::fmt::Display for UnsupportedPixelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "pixel format {:?} has no conversion engine (supported: {:?})",
            self.0,
            PixelFormat::ALL
        )
    }
}

impl std::error::Error for UnsupportedPixelFormat {}

impl From<UnsupportedPixelFormat> for crate::error::Error {
    fn from(e: UnsupportedPixelFormat) -> Self {
        crate::error::Error::InvalidCaps(e.to_string())
    }
}

/// Total: every engine format has a name in the caps vocabulary.
impl From<PixelFormat> for crate::format::PixelFormat {
    fn from(pf: PixelFormat) -> Self {
        use crate::format::PixelFormat as Caps;
        match pf {
            PixelFormat::I420 => Caps::I420,
            PixelFormat::Nv12 => Caps::Nv12,
            PixelFormat::Yuyv => Caps::Yuyv,
            PixelFormat::Uyvy => Caps::Uyvy,
            PixelFormat::Rgb24 => Caps::Rgb24,
            PixelFormat::Rgba => Caps::Rgba,
            PixelFormat::Bgr24 => Caps::Bgr24,
            PixelFormat::Bgra => Caps::Bgra,
            PixelFormat::Gray8 => Caps::Gray8,
        }
    }
}

/// Partial: only 9 of the 15 caps formats have an engine behind them.
impl TryFrom<crate::format::PixelFormat> for PixelFormat {
    type Error = UnsupportedPixelFormat;

    fn try_from(pf: crate::format::PixelFormat) -> std::result::Result<Self, Self::Error> {
        use crate::format::PixelFormat as Caps;
        Ok(match pf {
            Caps::I420 => PixelFormat::I420,
            Caps::Nv12 => PixelFormat::Nv12,
            Caps::Yuyv => PixelFormat::Yuyv,
            Caps::Uyvy => PixelFormat::Uyvy,
            Caps::Rgb24 => PixelFormat::Rgb24,
            Caps::Rgba => PixelFormat::Rgba,
            Caps::Bgr24 => PixelFormat::Bgr24,
            Caps::Bgra => PixelFormat::Bgra,
            Caps::Gray8 => PixelFormat::Gray8,
            other => return Err(UnsupportedPixelFormat(other)),
        })
    }
}

/// Color matrix for YUV ↔ RGB conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorMatrix {
    /// BT.601 (SD video, most common)
    #[default]
    Bt601,
    /// BT.709 (HD video)
    Bt709,
}

#[cfg(feature = "simd-colorspace")]
impl ColorMatrix {
    /// Convert to yuv crate's YuvStandardMatrix.
    fn to_yuv_matrix(self) -> YuvStandardMatrix {
        match self {
            ColorMatrix::Bt601 => YuvStandardMatrix::Bt601,
            ColorMatrix::Bt709 => YuvStandardMatrix::Bt709,
        }
    }
}

/// Byte offsets within a packed 4:2:2 group (4 bytes = 2 pixels).
///
/// YUYV and UYVY carry identical data in a different order, so every packed-422
/// conversion is the same code with a different one of these.
#[derive(Clone, Copy, Debug)]
struct Packed422 {
    /// Offset of the first pixel's luma.
    y0: usize,
    /// Offset of the shared U (Cb) sample.
    u: usize,
    /// Offset of the second pixel's luma.
    y1: usize,
    /// Offset of the shared V (Cr) sample.
    v: usize,
}

impl Packed422 {
    /// `Y0 U Y1 V` — the common webcam format.
    const YUYV: Self = Self {
        y0: 0,
        u: 1,
        y1: 2,
        v: 3,
    };
    /// `U Y0 V Y1`.
    const UYVY: Self = Self {
        u: 0,
        y0: 1,
        v: 2,
        y1: 3,
    };
}

/// Video format converter.
///
/// Converts between pixel formats while maintaining the same resolution.
/// For resolution changes, use [`ScaleEngine`](super::ScaleEngine).
pub struct VideoConvert {
    input_format: PixelFormat,
    output_format: PixelFormat,
    width: u32,
    height: u32,
    color_matrix: ColorMatrix,
}

/// One input plane resolved against a buffer: its bytes from the first row
/// to the end of the buffer, plus the geometry needed to walk it.
#[derive(Clone, Copy, Debug)]
struct PlaneIn<'a> {
    /// The plane's `stride * rows` bytes.
    data: &'a [u8],
    /// Byte distance between consecutive rows.
    stride: usize,
}

impl VideoConvert {
    /// Create a new video converter.
    pub fn new(
        input_format: PixelFormat,
        output_format: PixelFormat,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        if width == 0 || height == 0 {
            return Err(Error::Config("Width and height must be non-zero".into()));
        }

        // Validate YUV formats require even dimensions
        if (input_format.is_yuv() || output_format.is_yuv())
            && (!width.is_multiple_of(2) || !height.is_multiple_of(2))
        {
            return Err(Error::Config(
                "YUV formats require even width and height".into(),
            ));
        }

        Ok(Self {
            input_format,
            output_format,
            width,
            height,
            color_matrix: ColorMatrix::default(),
        })
    }

    /// Set the color matrix for YUV conversions.
    pub fn with_color_matrix(mut self, matrix: ColorMatrix) -> Self {
        self.color_matrix = matrix;
        self
    }

    /// Get input format.
    pub fn input_format(&self) -> PixelFormat {
        self.input_format
    }

    /// Get output format.
    pub fn output_format(&self) -> PixelFormat {
        self.output_format
    }

    /// Get the required output buffer size.
    pub fn output_size(&self) -> usize {
        self.output_format.buffer_size(self.width, self.height)
    }

    /// The packed input layout for this converter's format and geometry —
    /// what a caller holding an ordinary arena buffer passes to
    /// [`convert`](Self::convert).
    pub fn packed_input_layout(&self) -> PlaneLayout {
        PlaneLayout::packed(self.input_format.into(), self.width, self.height)
    }

    /// One input plane, resolved against the buffer: exactly its
    /// `stride * rows` bytes.
    ///
    /// The full span, not `stride * (rows - 1) + row_bytes`, because the
    /// SIMD backend walks planes with `chunks_exact(stride)` and drops a
    /// final partial chunk — the last row would silently vanish.
    /// [`convert`](Self::convert) gates on
    /// [`PlaneLayout::full_span_len`] before any of this runs.
    fn input_plane<'a>(&self, input: &'a [u8], layout: PlaneLayout, index: usize) -> PlaneIn<'a> {
        let p = layout
            .resolved(self.input_format.into(), self.width, self.height)
            .nth(index)
            .expect("plane index within the format's plane count");
        PlaneIn {
            data: &input[p.offset..p.offset + p.stride * p.rows],
            stride: p.stride,
        }
    }

    /// The three planes of an I420 input, in Y/U/V order.
    fn i420_planes<'a>(&self, input: &'a [u8], layout: PlaneLayout) -> [PlaneIn<'a>; 3] {
        [0, 1, 2].map(|i| self.input_plane(input, layout, i))
    }

    /// The two planes of an NV12 input: luma, then interleaved chroma.
    fn nv12_planes<'a>(&self, input: &'a [u8], layout: PlaneLayout) -> [PlaneIn<'a>; 2] {
        [0, 1].map(|i| self.input_plane(input, layout, i))
    }

    /// Convert a frame from input format to output format.
    ///
    /// `input_layout` describes where the input's planes are and how far
    /// apart their rows sit — [`packed_input_layout`](Self::packed_input_layout)
    /// for an ordinary buffer, the producer's own layout for a strided one
    /// (#194). The **output is always packed**: every caller writes into a
    /// freshly sized arena slot, so there is no output layout to thread.
    pub fn convert(
        &self,
        input: &[u8],
        input_layout: PlaneLayout,
        output: &mut [u8],
    ) -> Result<()> {
        // `full_span_len`, not `required_len`: the arms address planes by
        // whole rows (see `input_plane`). A caller holding a frame that ends
        // tight against its last row must repack it first — the elements'
        // scaffold does exactly that.
        let expected_input =
            input_layout.full_span_len(self.input_format.into(), self.width, self.height);
        let expected_output = self.output_format.buffer_size(self.width, self.height);

        if input.len() < expected_input {
            return Err(Error::Config(format!(
                "Input buffer too small: {} < {}",
                input.len(),
                expected_input
            )));
        }

        if output.len() < expected_output {
            return Err(Error::Config(format!(
                "Output buffer too small: {} < {}",
                output.len(),
                expected_output
            )));
        }

        // Every remaining input format is single-plane, so one resolved
        // view serves all of their arms; the planar/semi-planar arms take
        // the whole layout instead.
        let p0 = self.input_plane(input, input_layout, 0);

        // Dispatch to specific conversion
        match (self.input_format, self.output_format) {
            // Same format: a row-copy through the layout, which degenerates
            // to one `copy_from_slice` per plane when the input is packed.
            // Not a flat slice copy — that would carry a strided frame's row
            // padding into a packed output.
            (a, b) if a == b => {
                input_layout
                    .repack_into(
                        input,
                        self.input_format.into(),
                        self.width,
                        self.height,
                        output,
                    )
                    .map_err(Error::Config)?;
            }

            // YUV to RGB conversions
            (PixelFormat::I420, PixelFormat::Rgb24) => {
                self.i420_to_rgb24(input, input_layout, output);
            }
            (PixelFormat::I420, PixelFormat::Rgba) => {
                self.i420_to_rgba(input, input_layout, output);
            }
            (PixelFormat::I420, PixelFormat::Bgr24) => {
                self.i420_to_bgr24(input, input_layout, output);
            }
            (PixelFormat::I420, PixelFormat::Bgra) => {
                self.i420_to_bgra(input, input_layout, output);
            }
            (PixelFormat::Nv12, PixelFormat::Rgb24) => {
                self.nv12_to_rgb24(input, input_layout, output);
            }
            (PixelFormat::Nv12, PixelFormat::Rgba) => {
                self.nv12_to_rgba(input, input_layout, output);
            }
            (PixelFormat::Nv12, PixelFormat::Bgr24) => {
                self.nv12_to_bgr24(input, input_layout, output);
            }
            (PixelFormat::Nv12, PixelFormat::Bgra) => {
                self.nv12_to_bgra(input, input_layout, output);
            }

            // YUYV (packed YUV 4:2:2) to RGB conversions
            (PixelFormat::Yuyv, PixelFormat::Rgb24) => {
                self.yuyv_to_rgb24(p0.data, p0.stride, output);
            }
            (PixelFormat::Yuyv, PixelFormat::Rgba) => {
                self.yuyv_to_rgba(p0.data, p0.stride, output);
            }
            (PixelFormat::Yuyv, PixelFormat::Bgr24) => {
                self.yuyv_to_bgr24(p0.data, p0.stride, output);
            }
            (PixelFormat::Yuyv, PixelFormat::Bgra) => {
                self.yuyv_to_bgra(p0.data, p0.stride, output);
            }

            // UYVY (packed YUV 4:2:2) to RGB conversions
            (PixelFormat::Uyvy, PixelFormat::Rgb24) => {
                self.uyvy_to_rgb24(p0.data, p0.stride, output);
            }
            (PixelFormat::Uyvy, PixelFormat::Rgba) => {
                self.uyvy_to_rgba(p0.data, p0.stride, output);
            }

            // RGB to YUV conversions
            (PixelFormat::Rgb24, PixelFormat::I420) => {
                self.rgb24_to_i420(p0.data, p0.stride, output);
            }
            (PixelFormat::Rgba, PixelFormat::I420) => {
                self.rgba_to_i420(p0.data, p0.stride, output);
            }
            (PixelFormat::Bgra, PixelFormat::I420) => {
                self.bgra_to_i420(p0.data, p0.stride, output);
            }
            (PixelFormat::Bgr24, PixelFormat::I420) => {
                self.bgr24_to_i420(p0.data, p0.stride, output);
            }

            // RGB to NV12 conversions
            (PixelFormat::Rgb24, PixelFormat::Nv12) => {
                self.rgb24_to_nv12(p0.data, p0.stride, output);
            }
            (PixelFormat::Rgba, PixelFormat::Nv12) => {
                self.rgba_to_nv12(p0.data, p0.stride, output);
            }
            (PixelFormat::Bgr24, PixelFormat::Nv12) => {
                self.bgr24_to_nv12(p0.data, p0.stride, output);
            }
            (PixelFormat::Bgra, PixelFormat::Nv12) => {
                self.bgra_to_nv12(p0.data, p0.stride, output);
            }

            // RGB swizzle conversions
            (PixelFormat::Rgb24, PixelFormat::Bgr24) => {
                self.rgb_bgr_swap(p0.data, p0.stride, output, 3);
            }
            (PixelFormat::Bgr24, PixelFormat::Rgb24) => {
                self.rgb_bgr_swap(p0.data, p0.stride, output, 3);
            }
            (PixelFormat::Rgba, PixelFormat::Bgra) => {
                self.rgb_bgr_swap(p0.data, p0.stride, output, 4);
            }
            (PixelFormat::Bgra, PixelFormat::Rgba) => {
                self.rgb_bgr_swap(p0.data, p0.stride, output, 4);
            }

            // Add/remove alpha channel
            (PixelFormat::Rgb24, PixelFormat::Rgba) => {
                self.add_alpha(p0.data, p0.stride, output, false);
            }
            (PixelFormat::Bgr24, PixelFormat::Bgra) => {
                self.add_alpha(p0.data, p0.stride, output, false);
            }
            (PixelFormat::Rgba, PixelFormat::Rgb24) => {
                self.remove_alpha(p0.data, p0.stride, output, false);
            }
            (PixelFormat::Bgra, PixelFormat::Bgr24) => {
                self.remove_alpha(p0.data, p0.stride, output, false);
            }

            // Packed YUV 4:2:2 -> planar/semi-planar YUV 4:2:0.
            //
            // The path a plain USB webcam needs to reach an encoder. These are
            // pure data reshuffles — de-interleave and subsample chroma — with
            // no colour-space maths, so they are both cheaper and more accurate
            // than the YUV -> RGB -> YUV detour they replace.
            (PixelFormat::Yuyv, PixelFormat::I420) => {
                self.packed422_to_i420(p0.data, p0.stride, output, Packed422::YUYV);
            }
            (PixelFormat::Uyvy, PixelFormat::I420) => {
                self.packed422_to_i420(p0.data, p0.stride, output, Packed422::UYVY);
            }
            (PixelFormat::Yuyv, PixelFormat::Nv12) => {
                self.packed422_to_nv12(p0.data, p0.stride, output, Packed422::YUYV);
            }
            (PixelFormat::Uyvy, PixelFormat::Nv12) => {
                self.packed422_to_nv12(p0.data, p0.stride, output, Packed422::UYVY);
            }

            // Planar <-> semi-planar YUV 4:2:0 (a chroma plane interleave).
            (PixelFormat::I420, PixelFormat::Nv12) => {
                self.i420_to_nv12(input, input_layout, output);
            }
            (PixelFormat::Nv12, PixelFormat::I420) => {
                self.nv12_to_i420(input, input_layout, output);
            }

            // Gray conversions
            (PixelFormat::Gray8, PixelFormat::Rgb24) => {
                self.gray_to_rgb24(p0.data, p0.stride, output);
            }
            (PixelFormat::Gray8, PixelFormat::Rgba) => {
                self.gray_to_rgba(p0.data, p0.stride, output);
            }

            _ => {
                return Err(Error::Config(format!(
                    "Unsupported conversion: {:?} -> {:?}",
                    self.input_format, self.output_format
                )));
            }
        }

        Ok(())
    }

    // -------------------------------------------------------------------------
    // YUV <-> YUV conversions
    //
    // No colour-space maths here: these only move bytes around, so there is no
    // SIMD variant and no colour matrix involved.
    // -------------------------------------------------------------------------

    /// Packed 4:2:2 -> planar I420.
    ///
    /// Luma is copied straight through. Chroma is subsampled vertically by
    /// averaging the two source rows that fall in each 2x2 block — 4:2:2 already
    /// carries one chroma sample per horizontal pair, so only the vertical
    /// direction loses resolution.
    fn packed422_to_i420(&self, input: &[u8], in_stride: usize, output: &mut [u8], p: Packed422) {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w / 2, h / 2);
        let y_size = w * h;
        let c_size = cw * ch;

        let (y_out, chroma) = output.split_at_mut(y_size);
        let (u_out, v_out) = chroma.split_at_mut(c_size);

        Self::copy_luma(input, in_stride, y_out, w, h, p);

        for cy in 0..ch {
            for cx in 0..cw {
                let (u, v) = Self::average_chroma(input, in_stride, cy, cx, p);
                u_out[cy * cw + cx] = u;
                v_out[cy * cw + cx] = v;
            }
        }
    }

    /// Packed 4:2:2 -> semi-planar NV12 (interleaved UV plane).
    fn packed422_to_nv12(&self, input: &[u8], in_stride: usize, output: &mut [u8], p: Packed422) {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w / 2, h / 2);
        let y_size = w * h;

        let (y_out, uv_out) = output.split_at_mut(y_size);

        Self::copy_luma(input, in_stride, y_out, w, h, p);

        for cy in 0..ch {
            for cx in 0..cw {
                let (u, v) = Self::average_chroma(input, in_stride, cy, cx, p);
                let idx = (cy * cw + cx) * 2;
                uv_out[idx] = u;
                uv_out[idx + 1] = v;
            }
        }
    }

    /// I420 -> NV12: interleave the two chroma planes.
    fn i420_to_nv12(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w / 2, h / 2);
        let y_size = w * h;

        let [yp, up, vp] = self.i420_planes(input, layout);
        for row in 0..h {
            let src = row * yp.stride;
            output[row * w..row * w + w].copy_from_slice(&yp.data[src..src + w]);
        }
        for row in 0..ch {
            let (u_row, v_row) = (row * up.stride, row * vp.stride);
            let dst = y_size + row * cw * 2;
            for col in 0..cw {
                output[dst + col * 2] = up.data[u_row + col];
                output[dst + col * 2 + 1] = vp.data[v_row + col];
            }
        }
    }

    /// NV12 -> I420: de-interleave the chroma plane.
    fn nv12_to_i420(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (w / 2, h / 2);
        let y_size = w * h;
        let c_size = cw * ch;

        let [yp, uvp] = self.nv12_planes(input, layout);
        for row in 0..h {
            let src = row * yp.stride;
            output[row * w..row * w + w].copy_from_slice(&yp.data[src..src + w]);
        }
        for row in 0..ch {
            let src = row * uvp.stride;
            for col in 0..cw {
                output[y_size + row * cw + col] = uvp.data[src + col * 2];
                output[y_size + c_size + row * cw + col] = uvp.data[src + col * 2 + 1];
            }
        }
    }

    /// Copy the luma plane out of a packed 4:2:2 frame.
    fn copy_luma(
        input: &[u8],
        in_stride: usize,
        y_out: &mut [u8],
        w: usize,
        h: usize,
        p: Packed422,
    ) {
        for row in 0..h {
            let src_row = row * in_stride;
            for pair in 0..w / 2 {
                let src = src_row + pair * 4;
                y_out[row * w + pair * 2] = input[src + p.y0];
                y_out[row * w + pair * 2 + 1] = input[src + p.y1];
            }
        }
    }

    /// The (U, V) for one 4:2:0 chroma cell, averaged over the two source rows.
    fn average_chroma(
        input: &[u8],
        in_stride: usize,
        cy: usize,
        cx: usize,
        p: Packed422,
    ) -> (u8, u8) {
        let top = (cy * 2) * in_stride + cx * 4;
        let bottom = top + in_stride;

        // Rounded mean of the two source rows.
        let mean = |a: u8, b: u8| (a as u16 + b as u16).div_ceil(2) as u8;
        (
            mean(input[top + p.u], input[bottom + p.u]),
            mean(input[top + p.v], input[bottom + p.v]),
        )
    }

    // -------------------------------------------------------------------------
    // YUV to RGB conversions (SIMD-accelerated when simd-colorspace feature enabled)
    // -------------------------------------------------------------------------

    #[cfg(feature = "simd-colorspace")]
    fn i420_to_rgb24(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;

        let [yp, up, vp] = self.i420_planes(input, layout);

        let planar = YuvPlanarImage {
            y_plane: yp.data,
            y_stride: yp.stride as u32,
            u_plane: up.data,
            u_stride: up.stride as u32,
            v_plane: vp.data,
            v_stride: vp.stride as u32,
            width: self.width,
            height: self.height,
        };

        yuv420_to_rgb(
            &planar,
            output,
            (w * 3) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD i420_to_rgb24 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn i420_to_rgb24(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let [yp, up, vp] = self.i420_planes(input, layout);

        for row in 0..h {
            for col in 0..w {
                let y = yp.data[row * yp.stride + col];
                let u = up.data[(row / 2) * up.stride + (col / 2)];
                let v = vp.data[(row / 2) * vp.stride + (col / 2)];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 3;
                output[dst_idx] = r;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = b;
            }
        }
    }

    #[cfg(feature = "simd-colorspace")]
    fn i420_to_rgba(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;

        let [yp, up, vp] = self.i420_planes(input, layout);

        let planar = YuvPlanarImage {
            y_plane: yp.data,
            y_stride: yp.stride as u32,
            u_plane: up.data,
            u_stride: up.stride as u32,
            v_plane: vp.data,
            v_stride: vp.stride as u32,
            width: self.width,
            height: self.height,
        };

        yuv420_to_rgba(
            &planar,
            output,
            (w * 4) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD i420_to_rgba conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn i420_to_rgba(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let [yp, up, vp] = self.i420_planes(input, layout);

        for row in 0..h {
            for col in 0..w {
                let y = yp.data[row * yp.stride + col];
                let u = up.data[(row / 2) * up.stride + (col / 2)];
                let v = vp.data[(row / 2) * vp.stride + (col / 2)];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 4;
                output[dst_idx] = r;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = b;
                output[dst_idx + 3] = 255;
            }
        }
    }

    #[cfg(feature = "simd-colorspace")]
    fn i420_to_bgr24(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;

        let [yp, up, vp] = self.i420_planes(input, layout);

        let planar = YuvPlanarImage {
            y_plane: yp.data,
            y_stride: yp.stride as u32,
            u_plane: up.data,
            u_stride: up.stride as u32,
            v_plane: vp.data,
            v_stride: vp.stride as u32,
            width: self.width,
            height: self.height,
        };

        yuv420_to_bgr(
            &planar,
            output,
            (w * 3) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD i420_to_bgr24 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn i420_to_bgr24(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let [yp, up, vp] = self.i420_planes(input, layout);

        for row in 0..h {
            for col in 0..w {
                let y = yp.data[row * yp.stride + col];
                let u = up.data[(row / 2) * up.stride + (col / 2)];
                let v = vp.data[(row / 2) * vp.stride + (col / 2)];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 3;
                output[dst_idx] = b;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = r;
            }
        }
    }

    #[cfg(feature = "simd-colorspace")]
    fn i420_to_bgra(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;

        let [yp, up, vp] = self.i420_planes(input, layout);

        let planar = YuvPlanarImage {
            y_plane: yp.data,
            y_stride: yp.stride as u32,
            u_plane: up.data,
            u_stride: up.stride as u32,
            v_plane: vp.data,
            v_stride: vp.stride as u32,
            width: self.width,
            height: self.height,
        };

        yuv420_to_bgra(
            &planar,
            output,
            (w * 4) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD i420_to_bgra conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn i420_to_bgra(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let [yp, up, vp] = self.i420_planes(input, layout);

        for row in 0..h {
            for col in 0..w {
                let y = yp.data[row * yp.stride + col];
                let u = up.data[(row / 2) * up.stride + (col / 2)];
                let v = vp.data[(row / 2) * vp.stride + (col / 2)];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 4;
                output[dst_idx] = b;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = r;
                output[dst_idx + 3] = 255;
            }
        }
    }

    // -------------------------------------------------------------------------
    // NV12 conversions (SIMD-accelerated when simd-colorspace feature enabled)
    // -------------------------------------------------------------------------

    #[cfg(feature = "simd-colorspace")]
    fn nv12_to_rgb24(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;

        let [yp, uvp] = self.nv12_planes(input, layout);

        let bi_planar = YuvBiPlanarImage {
            y_plane: yp.data,
            y_stride: yp.stride as u32,
            uv_plane: uvp.data,
            uv_stride: uvp.stride as u32,
            width: self.width,
            height: self.height,
        };

        yuv_nv12_to_rgb(
            &bi_planar,
            output,
            (w * 3) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD nv12_to_rgb24 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn nv12_to_rgb24(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let [yp, uvp] = self.nv12_planes(input, layout);

        for row in 0..h {
            for col in 0..w {
                let y = yp.data[row * yp.stride + col];
                let uv_idx = (row / 2) * uvp.stride + (col / 2) * 2;
                let u = uvp.data[uv_idx];
                let v = uvp.data[uv_idx + 1];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 3;
                output[dst_idx] = r;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = b;
            }
        }
    }

    #[cfg(feature = "simd-colorspace")]
    fn nv12_to_rgba(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;

        let [yp, uvp] = self.nv12_planes(input, layout);

        let bi_planar = YuvBiPlanarImage {
            y_plane: yp.data,
            y_stride: yp.stride as u32,
            uv_plane: uvp.data,
            uv_stride: uvp.stride as u32,
            width: self.width,
            height: self.height,
        };

        yuv_nv12_to_rgba(
            &bi_planar,
            output,
            (w * 4) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD nv12_to_rgba conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn nv12_to_rgba(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let [yp, uvp] = self.nv12_planes(input, layout);

        for row in 0..h {
            for col in 0..w {
                let y = yp.data[row * yp.stride + col];
                let uv_idx = (row / 2) * uvp.stride + (col / 2) * 2;
                let u = uvp.data[uv_idx];
                let v = uvp.data[uv_idx + 1];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 4;
                output[dst_idx] = r;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = b;
                output[dst_idx + 3] = 255;
            }
        }
    }

    /// Convert NV12 to BGR24 (SIMD-accelerated when available).
    #[cfg(feature = "simd-colorspace")]
    fn nv12_to_bgr24(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;

        let [yp, uvp] = self.nv12_planes(input, layout);

        let bi_planar = YuvBiPlanarImage {
            y_plane: yp.data,
            y_stride: yp.stride as u32,
            uv_plane: uvp.data,
            uv_stride: uvp.stride as u32,
            width: self.width,
            height: self.height,
        };

        yuv_nv12_to_bgr(
            &bi_planar,
            output,
            (w * 3) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD nv12_to_bgr24 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn nv12_to_bgr24(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let [yp, uvp] = self.nv12_planes(input, layout);

        for row in 0..h {
            for col in 0..w {
                let y = yp.data[row * yp.stride + col];
                let uv_idx = (row / 2) * uvp.stride + (col / 2) * 2;
                let u = uvp.data[uv_idx];
                let v = uvp.data[uv_idx + 1];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 3;
                output[dst_idx] = b;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = r;
            }
        }
    }

    /// Convert NV12 to BGRA (SIMD-accelerated when available).
    #[cfg(feature = "simd-colorspace")]
    fn nv12_to_bgra(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;

        let [yp, uvp] = self.nv12_planes(input, layout);

        let bi_planar = YuvBiPlanarImage {
            y_plane: yp.data,
            y_stride: yp.stride as u32,
            uv_plane: uvp.data,
            uv_stride: uvp.stride as u32,
            width: self.width,
            height: self.height,
        };

        yuv_nv12_to_bgra(
            &bi_planar,
            output,
            (w * 4) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD nv12_to_bgra conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn nv12_to_bgra(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let [yp, uvp] = self.nv12_planes(input, layout);

        for row in 0..h {
            for col in 0..w {
                let y = yp.data[row * yp.stride + col];
                let uv_idx = (row / 2) * uvp.stride + (col / 2) * 2;
                let u = uvp.data[uv_idx];
                let v = uvp.data[uv_idx + 1];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 4;
                output[dst_idx] = b;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = r;
                output[dst_idx + 3] = 255;
            }
        }
    }

    // -------------------------------------------------------------------------
    // YUYV/UYVY (packed YUV 4:2:2) to RGB conversions (SIMD when available)
    // -------------------------------------------------------------------------

    /// Convert YUYV (YUY2) to RGB24.
    /// YUYV layout: Y0 U0 Y1 V0 (4 bytes encode 2 pixels)
    #[cfg(feature = "simd-colorspace")]
    fn yuyv_to_rgb24(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;

        let packed = YuvPackedImage {
            yuy: input,
            yuy_stride: in_stride as u32,
            width: self.width,
            height: self.height,
        };

        yuyv422_to_rgb(
            &packed,
            output,
            (w * 3) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD yuyv_to_rgb24 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn yuyv_to_rgb24(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        for row in 0..h {
            for col in (0..w).step_by(2) {
                // Read 4 bytes: Y0 U Y1 V
                let src_idx = row * in_stride + col * 2;
                let y0 = input[src_idx];
                let u = input[src_idx + 1];
                let y1 = input[src_idx + 2];
                let v = input[src_idx + 3];

                // First pixel
                let (r0, g0, b0) = self.yuv_to_rgb(y0, u, v);
                let dst_idx0 = (row * w + col) * 3;
                output[dst_idx0] = r0;
                output[dst_idx0 + 1] = g0;
                output[dst_idx0 + 2] = b0;

                // Second pixel
                let (r1, g1, b1) = self.yuv_to_rgb(y1, u, v);
                let dst_idx1 = (row * w + col + 1) * 3;
                output[dst_idx1] = r1;
                output[dst_idx1 + 1] = g1;
                output[dst_idx1 + 2] = b1;
            }
        }
    }

    /// Convert YUYV (YUY2) to RGBA.
    #[cfg(feature = "simd-colorspace")]
    fn yuyv_to_rgba(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;

        let packed = YuvPackedImage {
            yuy: input,
            yuy_stride: in_stride as u32,
            width: self.width,
            height: self.height,
        };

        yuyv422_to_rgba(
            &packed,
            output,
            (w * 4) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD yuyv_to_rgba conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn yuyv_to_rgba(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        for row in 0..h {
            for col in (0..w).step_by(2) {
                let src_idx = row * in_stride + col * 2;
                let y0 = input[src_idx];
                let u = input[src_idx + 1];
                let y1 = input[src_idx + 2];
                let v = input[src_idx + 3];

                // First pixel
                let (r0, g0, b0) = self.yuv_to_rgb(y0, u, v);
                let dst_idx0 = (row * w + col) * 4;
                output[dst_idx0] = r0;
                output[dst_idx0 + 1] = g0;
                output[dst_idx0 + 2] = b0;
                output[dst_idx0 + 3] = 255;

                // Second pixel
                let (r1, g1, b1) = self.yuv_to_rgb(y1, u, v);
                let dst_idx1 = (row * w + col + 1) * 4;
                output[dst_idx1] = r1;
                output[dst_idx1 + 1] = g1;
                output[dst_idx1 + 2] = b1;
                output[dst_idx1 + 3] = 255;
            }
        }
    }

    /// Convert YUYV (YUY2) to BGR24.
    #[cfg(feature = "simd-colorspace")]
    fn yuyv_to_bgr24(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;

        let packed = YuvPackedImage {
            yuy: input,
            yuy_stride: in_stride as u32,
            width: self.width,
            height: self.height,
        };

        yuyv422_to_bgr(
            &packed,
            output,
            (w * 3) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD yuyv_to_bgr24 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn yuyv_to_bgr24(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        for row in 0..h {
            for col in (0..w).step_by(2) {
                let src_idx = row * in_stride + col * 2;
                let y0 = input[src_idx];
                let u = input[src_idx + 1];
                let y1 = input[src_idx + 2];
                let v = input[src_idx + 3];

                // First pixel (BGR order)
                let (r0, g0, b0) = self.yuv_to_rgb(y0, u, v);
                let dst_idx0 = (row * w + col) * 3;
                output[dst_idx0] = b0;
                output[dst_idx0 + 1] = g0;
                output[dst_idx0 + 2] = r0;

                // Second pixel
                let (r1, g1, b1) = self.yuv_to_rgb(y1, u, v);
                let dst_idx1 = (row * w + col + 1) * 3;
                output[dst_idx1] = b1;
                output[dst_idx1 + 1] = g1;
                output[dst_idx1 + 2] = r1;
            }
        }
    }

    /// Convert YUYV (YUY2) to BGRA.
    #[cfg(feature = "simd-colorspace")]
    fn yuyv_to_bgra(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;

        let packed = YuvPackedImage {
            yuy: input,
            yuy_stride: in_stride as u32,
            width: self.width,
            height: self.height,
        };

        yuyv422_to_bgra(
            &packed,
            output,
            (w * 4) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD yuyv_to_bgra conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn yuyv_to_bgra(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        for row in 0..h {
            for col in (0..w).step_by(2) {
                let src_idx = row * in_stride + col * 2;
                let y0 = input[src_idx];
                let u = input[src_idx + 1];
                let y1 = input[src_idx + 2];
                let v = input[src_idx + 3];

                // First pixel (BGRA order)
                let (r0, g0, b0) = self.yuv_to_rgb(y0, u, v);
                let dst_idx0 = (row * w + col) * 4;
                output[dst_idx0] = b0;
                output[dst_idx0 + 1] = g0;
                output[dst_idx0 + 2] = r0;
                output[dst_idx0 + 3] = 255;

                // Second pixel
                let (r1, g1, b1) = self.yuv_to_rgb(y1, u, v);
                let dst_idx1 = (row * w + col + 1) * 4;
                output[dst_idx1] = b1;
                output[dst_idx1 + 1] = g1;
                output[dst_idx1 + 2] = r1;
                output[dst_idx1 + 3] = 255;
            }
        }
    }

    /// Convert UYVY to RGB24.
    /// UYVY layout: U0 Y0 V0 Y1 (4 bytes encode 2 pixels)
    #[cfg(feature = "simd-colorspace")]
    fn uyvy_to_rgb24(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;

        let packed = YuvPackedImage {
            yuy: input,
            yuy_stride: in_stride as u32,
            width: self.width,
            height: self.height,
        };

        uyvy422_to_rgb(
            &packed,
            output,
            (w * 3) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD uyvy_to_rgb24 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn uyvy_to_rgb24(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        for row in 0..h {
            for col in (0..w).step_by(2) {
                // Read 4 bytes: U Y0 V Y1
                let src_idx = row * in_stride + col * 2;
                let u = input[src_idx];
                let y0 = input[src_idx + 1];
                let v = input[src_idx + 2];
                let y1 = input[src_idx + 3];

                // First pixel
                let (r0, g0, b0) = self.yuv_to_rgb(y0, u, v);
                let dst_idx0 = (row * w + col) * 3;
                output[dst_idx0] = r0;
                output[dst_idx0 + 1] = g0;
                output[dst_idx0 + 2] = b0;

                // Second pixel
                let (r1, g1, b1) = self.yuv_to_rgb(y1, u, v);
                let dst_idx1 = (row * w + col + 1) * 3;
                output[dst_idx1] = r1;
                output[dst_idx1 + 1] = g1;
                output[dst_idx1 + 2] = b1;
            }
        }
    }

    /// Convert UYVY to RGBA.
    #[cfg(feature = "simd-colorspace")]
    fn uyvy_to_rgba(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;

        let packed = YuvPackedImage {
            yuy: input,
            yuy_stride: in_stride as u32,
            width: self.width,
            height: self.height,
        };

        uyvy422_to_rgba(
            &packed,
            output,
            (w * 4) as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
        )
        .expect("SIMD uyvy_to_rgba conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn uyvy_to_rgba(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        for row in 0..h {
            for col in (0..w).step_by(2) {
                let src_idx = row * in_stride + col * 2;
                let u = input[src_idx];
                let y0 = input[src_idx + 1];
                let v = input[src_idx + 2];
                let y1 = input[src_idx + 3];

                // First pixel
                let (r0, g0, b0) = self.yuv_to_rgb(y0, u, v);
                let dst_idx0 = (row * w + col) * 4;
                output[dst_idx0] = r0;
                output[dst_idx0 + 1] = g0;
                output[dst_idx0 + 2] = b0;
                output[dst_idx0 + 3] = 255;

                // Second pixel
                let (r1, g1, b1) = self.yuv_to_rgb(y1, u, v);
                let dst_idx1 = (row * w + col + 1) * 4;
                output[dst_idx1] = r1;
                output[dst_idx1 + 1] = g1;
                output[dst_idx1 + 2] = b1;
                output[dst_idx1 + 3] = 255;
            }
        }
    }

    // -------------------------------------------------------------------------
    // RGB to YUV conversions (SIMD-accelerated when simd-colorspace feature enabled)
    // -------------------------------------------------------------------------

    #[cfg(feature = "simd-colorspace")]
    fn rgb24_to_i420(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_stride = w / 2;

        // Split output into Y, U, V planes
        let (y_plane, uv_planes) = output.split_at_mut(y_size);
        let (u_plane, v_plane) = uv_planes.split_at_mut(uv_stride * (h / 2));

        let mut planar = YuvPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y_plane),
            y_stride: w as u32,
            u_plane: BufferStoreMut::Borrowed(u_plane),
            u_stride: uv_stride as u32,
            v_plane: BufferStoreMut::Borrowed(v_plane),
            v_stride: uv_stride as u32,
            width: self.width,
            height: self.height,
        };

        rgb_to_yuv420(
            &mut planar,
            input,
            in_stride as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD rgb24_to_i420 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn rgb24_to_i420(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);

        // First pass: compute Y for all pixels
        for row in 0..h {
            for col in 0..w {
                let src_idx = row * in_stride + col * 3;
                let r = input[src_idx];
                let g = input[src_idx + 1];
                let b = input[src_idx + 2];

                let (y, _, _) = self.rgb_to_yuv(r, g, b);
                output[row * w + col] = y;
            }
        }

        // Second pass: average U/V values in 2x2 blocks
        for row in (0..h).step_by(2) {
            for col in (0..w).step_by(2) {
                let mut u_sum = 0u32;
                let mut v_sum = 0u32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_idx = (row + dy) * in_stride + (col + dx) * 3;
                        let r = input[src_idx];
                        let g = input[src_idx + 1];
                        let b = input[src_idx + 2];
                        let (_, u, v) = self.rgb_to_yuv(r, g, b);
                        u_sum += u as u32;
                        v_sum += v as u32;
                    }
                }

                let uv_idx = (row / 2) * (w / 2) + (col / 2);
                output[y_size + uv_idx] = (u_sum / 4) as u8;
                output[y_size + uv_size + uv_idx] = (v_sum / 4) as u8;
            }
        }
    }

    #[cfg(feature = "simd-colorspace")]
    fn rgba_to_i420(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_stride = w / 2;

        // Split output into Y, U, V planes
        let (y_plane, uv_planes) = output.split_at_mut(y_size);
        let (u_plane, v_plane) = uv_planes.split_at_mut(uv_stride * (h / 2));

        let mut planar = YuvPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y_plane),
            y_stride: w as u32,
            u_plane: BufferStoreMut::Borrowed(u_plane),
            u_stride: uv_stride as u32,
            v_plane: BufferStoreMut::Borrowed(v_plane),
            v_stride: uv_stride as u32,
            width: self.width,
            height: self.height,
        };

        rgba_to_yuv420(
            &mut planar,
            input,
            in_stride as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD rgba_to_i420 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn rgba_to_i420(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);

        // Compute Y for all pixels
        for row in 0..h {
            for col in 0..w {
                let src_idx = row * in_stride + col * 4;
                let r = input[src_idx];
                let g = input[src_idx + 1];
                let b = input[src_idx + 2];
                // Alpha ignored

                let (y, _, _) = self.rgb_to_yuv(r, g, b);
                output[row * w + col] = y;
            }
        }

        // Average U/V in 2x2 blocks
        for row in (0..h).step_by(2) {
            for col in (0..w).step_by(2) {
                let mut u_sum = 0u32;
                let mut v_sum = 0u32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_idx = (row + dy) * in_stride + (col + dx) * 4;
                        let r = input[src_idx];
                        let g = input[src_idx + 1];
                        let b = input[src_idx + 2];
                        let (_, u, v) = self.rgb_to_yuv(r, g, b);
                        u_sum += u as u32;
                        v_sum += v as u32;
                    }
                }

                let uv_idx = (row / 2) * (w / 2) + (col / 2);
                output[y_size + uv_idx] = (u_sum / 4) as u8;
                output[y_size + uv_size + uv_idx] = (v_sum / 4) as u8;
            }
        }
    }

    #[cfg(feature = "simd-colorspace")]
    fn bgra_to_i420(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_stride = w / 2;

        // Split output into Y, U, V planes
        let (y_plane, uv_planes) = output.split_at_mut(y_size);
        let (u_plane, v_plane) = uv_planes.split_at_mut(uv_stride * (h / 2));

        let mut planar = YuvPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y_plane),
            y_stride: w as u32,
            u_plane: BufferStoreMut::Borrowed(u_plane),
            u_stride: uv_stride as u32,
            v_plane: BufferStoreMut::Borrowed(v_plane),
            v_stride: uv_stride as u32,
            width: self.width,
            height: self.height,
        };

        bgra_to_yuv420(
            &mut planar,
            input,
            in_stride as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD bgra_to_i420 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn bgra_to_i420(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);

        // Compute Y for all pixels
        // BGRA layout: B=0, G=1, R=2, A=3
        for row in 0..h {
            for col in 0..w {
                let src_idx = row * in_stride + col * 4;
                let b = input[src_idx];
                let g = input[src_idx + 1];
                let r = input[src_idx + 2];
                // Alpha ignored

                let (y, _, _) = self.rgb_to_yuv(r, g, b);
                output[row * w + col] = y;
            }
        }

        // Average U/V in 2x2 blocks
        for row in (0..h).step_by(2) {
            for col in (0..w).step_by(2) {
                let mut u_sum = 0u32;
                let mut v_sum = 0u32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_idx = (row + dy) * in_stride + (col + dx) * 4;
                        let b = input[src_idx];
                        let g = input[src_idx + 1];
                        let r = input[src_idx + 2];
                        let (_, u, v) = self.rgb_to_yuv(r, g, b);
                        u_sum += u as u32;
                        v_sum += v as u32;
                    }
                }

                let uv_idx = (row / 2) * (w / 2) + (col / 2);
                output[y_size + uv_idx] = (u_sum / 4) as u8;
                output[y_size + uv_size + uv_idx] = (v_sum / 4) as u8;
            }
        }
    }

    /// Convert BGR24 to I420 (SIMD-accelerated when available).
    #[cfg(feature = "simd-colorspace")]
    fn bgr24_to_i420(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_stride = w / 2;

        // Split output into Y, U, V planes
        let (y_plane, uv_planes) = output.split_at_mut(y_size);
        let (u_plane, v_plane) = uv_planes.split_at_mut(uv_stride * (h / 2));

        let mut planar = YuvPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y_plane),
            y_stride: w as u32,
            u_plane: BufferStoreMut::Borrowed(u_plane),
            u_stride: uv_stride as u32,
            v_plane: BufferStoreMut::Borrowed(v_plane),
            v_stride: uv_stride as u32,
            width: self.width,
            height: self.height,
        };

        bgr_to_yuv420(
            &mut planar,
            input,
            in_stride as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD bgr24_to_i420 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn bgr24_to_i420(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);

        // BGR layout: B=0, G=1, R=2
        for row in 0..h {
            for col in 0..w {
                let src_idx = row * in_stride + col * 3;
                let b = input[src_idx];
                let g = input[src_idx + 1];
                let r = input[src_idx + 2];

                let (y, _, _) = self.rgb_to_yuv(r, g, b);
                output[row * w + col] = y;
            }
        }

        for row in (0..h).step_by(2) {
            for col in (0..w).step_by(2) {
                let mut u_sum = 0u32;
                let mut v_sum = 0u32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_idx = (row + dy) * in_stride + (col + dx) * 3;
                        let b = input[src_idx];
                        let g = input[src_idx + 1];
                        let r = input[src_idx + 2];
                        let (_, u, v) = self.rgb_to_yuv(r, g, b);
                        u_sum += u as u32;
                        v_sum += v as u32;
                    }
                }

                let uv_idx = (row / 2) * (w / 2) + (col / 2);
                output[y_size + uv_idx] = (u_sum / 4) as u8;
                output[y_size + uv_size + uv_idx] = (v_sum / 4) as u8;
            }
        }
    }

    // -------------------------------------------------------------------------
    // RGB to NV12 conversions (SIMD-accelerated when simd-colorspace feature enabled)
    // -------------------------------------------------------------------------

    /// Convert RGB24 to NV12 (SIMD-accelerated when available).
    #[cfg(feature = "simd-colorspace")]
    fn rgb24_to_nv12(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;

        let (y_plane, uv_plane) = output.split_at_mut(y_size);

        let mut bi_planar = YuvBiPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y_plane),
            y_stride: w as u32,
            uv_plane: BufferStoreMut::Borrowed(uv_plane),
            uv_stride: w as u32,
            width: self.width,
            height: self.height,
        };

        rgb_to_yuv_nv12(
            &mut bi_planar,
            input,
            in_stride as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD rgb24_to_nv12 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn rgb24_to_nv12(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;

        // Compute Y for all pixels
        for row in 0..h {
            for col in 0..w {
                let src_idx = row * in_stride + col * 3;
                let r = input[src_idx];
                let g = input[src_idx + 1];
                let b = input[src_idx + 2];

                let (y, _, _) = self.rgb_to_yuv(r, g, b);
                output[row * w + col] = y;
            }
        }

        // Compute interleaved UV (average 2x2 blocks)
        for row in (0..h).step_by(2) {
            for col in (0..w).step_by(2) {
                let mut u_sum = 0u32;
                let mut v_sum = 0u32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_idx = (row + dy) * in_stride + (col + dx) * 3;
                        let r = input[src_idx];
                        let g = input[src_idx + 1];
                        let b = input[src_idx + 2];
                        let (_, u, v) = self.rgb_to_yuv(r, g, b);
                        u_sum += u as u32;
                        v_sum += v as u32;
                    }
                }

                let uv_idx = y_size + (row / 2) * w + col;
                output[uv_idx] = (u_sum / 4) as u8;
                output[uv_idx + 1] = (v_sum / 4) as u8;
            }
        }
    }

    /// Convert RGBA to NV12 (SIMD-accelerated when available).
    #[cfg(feature = "simd-colorspace")]
    fn rgba_to_nv12(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;

        let (y_plane, uv_plane) = output.split_at_mut(y_size);

        let mut bi_planar = YuvBiPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y_plane),
            y_stride: w as u32,
            uv_plane: BufferStoreMut::Borrowed(uv_plane),
            uv_stride: w as u32,
            width: self.width,
            height: self.height,
        };

        rgba_to_yuv_nv12(
            &mut bi_planar,
            input,
            in_stride as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD rgba_to_nv12 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn rgba_to_nv12(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;

        for row in 0..h {
            for col in 0..w {
                let src_idx = row * in_stride + col * 4;
                let r = input[src_idx];
                let g = input[src_idx + 1];
                let b = input[src_idx + 2];

                let (y, _, _) = self.rgb_to_yuv(r, g, b);
                output[row * w + col] = y;
            }
        }

        for row in (0..h).step_by(2) {
            for col in (0..w).step_by(2) {
                let mut u_sum = 0u32;
                let mut v_sum = 0u32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_idx = (row + dy) * in_stride + (col + dx) * 4;
                        let r = input[src_idx];
                        let g = input[src_idx + 1];
                        let b = input[src_idx + 2];
                        let (_, u, v) = self.rgb_to_yuv(r, g, b);
                        u_sum += u as u32;
                        v_sum += v as u32;
                    }
                }

                let uv_idx = y_size + (row / 2) * w + col;
                output[uv_idx] = (u_sum / 4) as u8;
                output[uv_idx + 1] = (v_sum / 4) as u8;
            }
        }
    }

    /// Convert BGR24 to NV12 (SIMD-accelerated when available).
    #[cfg(feature = "simd-colorspace")]
    fn bgr24_to_nv12(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;

        let (y_plane, uv_plane) = output.split_at_mut(y_size);

        let mut bi_planar = YuvBiPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y_plane),
            y_stride: w as u32,
            uv_plane: BufferStoreMut::Borrowed(uv_plane),
            uv_stride: w as u32,
            width: self.width,
            height: self.height,
        };

        bgr_to_yuv_nv12(
            &mut bi_planar,
            input,
            in_stride as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD bgr24_to_nv12 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn bgr24_to_nv12(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;

        for row in 0..h {
            for col in 0..w {
                let src_idx = row * in_stride + col * 3;
                let b = input[src_idx];
                let g = input[src_idx + 1];
                let r = input[src_idx + 2];

                let (y, _, _) = self.rgb_to_yuv(r, g, b);
                output[row * w + col] = y;
            }
        }

        for row in (0..h).step_by(2) {
            for col in (0..w).step_by(2) {
                let mut u_sum = 0u32;
                let mut v_sum = 0u32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_idx = (row + dy) * in_stride + (col + dx) * 3;
                        let b = input[src_idx];
                        let g = input[src_idx + 1];
                        let r = input[src_idx + 2];
                        let (_, u, v) = self.rgb_to_yuv(r, g, b);
                        u_sum += u as u32;
                        v_sum += v as u32;
                    }
                }

                let uv_idx = y_size + (row / 2) * w + col;
                output[uv_idx] = (u_sum / 4) as u8;
                output[uv_idx + 1] = (v_sum / 4) as u8;
            }
        }
    }

    /// Convert BGRA to NV12 (SIMD-accelerated when available).
    #[cfg(feature = "simd-colorspace")]
    fn bgra_to_nv12(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;

        let (y_plane, uv_plane) = output.split_at_mut(y_size);

        let mut bi_planar = YuvBiPlanarImageMut {
            y_plane: BufferStoreMut::Borrowed(y_plane),
            y_stride: w as u32,
            uv_plane: BufferStoreMut::Borrowed(uv_plane),
            uv_stride: w as u32,
            width: self.width,
            height: self.height,
        };

        bgra_to_yuv_nv12(
            &mut bi_planar,
            input,
            in_stride as u32,
            YuvRange::Limited,
            self.color_matrix.to_yuv_matrix(),
            YuvConversionMode::Balanced,
        )
        .expect("SIMD bgra_to_nv12 conversion failed");
    }

    #[cfg(not(feature = "simd-colorspace"))]
    fn bgra_to_nv12(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;

        for row in 0..h {
            for col in 0..w {
                let src_idx = row * in_stride + col * 4;
                let b = input[src_idx];
                let g = input[src_idx + 1];
                let r = input[src_idx + 2];

                let (y, _, _) = self.rgb_to_yuv(r, g, b);
                output[row * w + col] = y;
            }
        }

        for row in (0..h).step_by(2) {
            for col in (0..w).step_by(2) {
                let mut u_sum = 0u32;
                let mut v_sum = 0u32;

                for dy in 0..2 {
                    for dx in 0..2 {
                        let src_idx = (row + dy) * in_stride + (col + dx) * 4;
                        let b = input[src_idx];
                        let g = input[src_idx + 1];
                        let r = input[src_idx + 2];
                        let (_, u, v) = self.rgb_to_yuv(r, g, b);
                        u_sum += u as u32;
                        v_sum += v as u32;
                    }
                }

                let uv_idx = y_size + (row / 2) * w + col;
                output[uv_idx] = (u_sum / 4) as u8;
                output[uv_idx + 1] = (v_sum / 4) as u8;
            }
        }
    }

    // -------------------------------------------------------------------------
    // RGB format conversions
    // -------------------------------------------------------------------------

    fn rgb_bgr_swap(
        &self,
        input: &[u8],
        in_stride: usize,
        output: &mut [u8],
        bytes_per_pixel: usize,
    ) {
        let (w, h) = (self.width as usize, self.height as usize);

        for row in 0..h {
            for col in 0..w {
                let src = row * in_stride + col * bytes_per_pixel;
                let dst = (row * w + col) * bytes_per_pixel;

                output[dst] = input[src + 2]; // R/B swap
                output[dst + 1] = input[src + 1]; // G stays
                output[dst + 2] = input[src]; // B/R swap

                if bytes_per_pixel == 4 {
                    output[dst + 3] = input[src + 3]; // Alpha stays
                }
            }
        }
    }

    fn add_alpha(&self, input: &[u8], in_stride: usize, output: &mut [u8], _is_bgr: bool) {
        let (w, h) = (self.width as usize, self.height as usize);

        for row in 0..h {
            for col in 0..w {
                let src = row * in_stride + col * 3;
                let dst = (row * w + col) * 4;

                output[dst] = input[src];
                output[dst + 1] = input[src + 1];
                output[dst + 2] = input[src + 2];
                output[dst + 3] = 255; // Opaque alpha
            }
        }
    }

    fn remove_alpha(&self, input: &[u8], in_stride: usize, output: &mut [u8], _is_bgr: bool) {
        let (w, h) = (self.width as usize, self.height as usize);

        for row in 0..h {
            for col in 0..w {
                let src = row * in_stride + col * 4;
                let dst = (row * w + col) * 3;

                output[dst] = input[src];
                output[dst + 1] = input[src + 1];
                output[dst + 2] = input[src + 2];
                // Alpha discarded
            }
        }
    }

    fn gray_to_rgb24(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let (w, h) = (self.width as usize, self.height as usize);

        for row in 0..h {
            let src = &input[row * in_stride..row * in_stride + w];
            let dst = &mut output[row * w * 3..(row + 1) * w * 3];
            for (gray, dst_chunk) in src.iter().zip(dst.chunks_exact_mut(3)) {
                dst_chunk[0] = *gray;
                dst_chunk[1] = *gray;
                dst_chunk[2] = *gray;
            }
        }
    }

    fn gray_to_rgba(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let (w, h) = (self.width as usize, self.height as usize);

        for row in 0..h {
            let src = &input[row * in_stride..row * in_stride + w];
            let dst = &mut output[row * w * 4..(row + 1) * w * 4];
            for (gray, dst_chunk) in src.iter().zip(dst.chunks_exact_mut(4)) {
                dst_chunk[0] = *gray;
                dst_chunk[1] = *gray;
                dst_chunk[2] = *gray;
                dst_chunk[3] = 255;
            }
        }
    }

    // -------------------------------------------------------------------------
    // Color space math
    // -------------------------------------------------------------------------

    /// Convert YUV to RGB using the configured color matrix.
    ///
    /// Only the scalar conversion paths call this; with `simd-colorspace`
    /// those paths are compiled out and the yuv crate does the math.
    #[cfg_attr(feature = "simd-colorspace", allow(dead_code))]
    #[inline]
    fn yuv_to_rgb(&self, y: u8, u: u8, v: u8) -> (u8, u8, u8) {
        let y = y as i32;
        let u = u as i32 - 128;
        let v = v as i32 - 128;

        // Use fixed-point arithmetic for speed
        let (r, g, b) = match self.color_matrix {
            ColorMatrix::Bt601 => {
                // BT.601 coefficients (scaled by 1024)
                // R = Y + 1.402 * V
                // G = Y - 0.344136 * U - 0.714136 * V
                // B = Y + 1.772 * U
                let r = y + ((1436 * v) >> 10);
                let g = y - ((352 * u + 731 * v) >> 10);
                let b = y + ((1815 * u) >> 10);
                (r, g, b)
            }
            ColorMatrix::Bt709 => {
                // BT.709 coefficients (scaled by 1024)
                // R = Y + 1.5748 * V
                // G = Y - 0.1873 * U - 0.4681 * V
                // B = Y + 1.8556 * U
                let r = y + ((1613 * v) >> 10);
                let g = y - ((192 * u + 479 * v) >> 10);
                let b = y + ((1900 * u) >> 10);
                (r, g, b)
            }
        };

        (
            r.clamp(0, 255) as u8,
            g.clamp(0, 255) as u8,
            b.clamp(0, 255) as u8,
        )
    }

    /// Convert RGB to YUV using the configured color matrix.
    #[cfg_attr(feature = "simd-colorspace", allow(dead_code))]
    #[inline]
    fn rgb_to_yuv(&self, r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let r = r as i32;
        let g = g as i32;
        let b = b as i32;

        // Use fixed-point arithmetic
        let (y, u, v) = match self.color_matrix {
            ColorMatrix::Bt601 => {
                // BT.601 coefficients (scaled by 1024)
                // Y = 0.299 * R + 0.587 * G + 0.114 * B
                // U = -0.169 * R - 0.331 * G + 0.5 * B + 128
                // V = 0.5 * R - 0.419 * G - 0.081 * B + 128
                let y = ((306 * r + 601 * g + 117 * b) >> 10).clamp(0, 255);
                let u = (((-173 * r - 339 * g + 512 * b) >> 10) + 128).clamp(0, 255);
                let v = (((512 * r - 429 * g - 83 * b) >> 10) + 128).clamp(0, 255);
                (y, u, v)
            }
            ColorMatrix::Bt709 => {
                // BT.709 coefficients (scaled by 1024)
                // Y = 0.2126 * R + 0.7152 * G + 0.0722 * B
                // U = -0.1146 * R - 0.3854 * G + 0.5 * B + 128
                // V = 0.5 * R - 0.4542 * G - 0.0458 * B + 128
                let y = ((218 * r + 732 * g + 74 * b) >> 10).clamp(0, 255);
                let u = (((-117 * r - 395 * g + 512 * b) >> 10) + 128).clamp(0, 255);
                let v = (((512 * r - 465 * g - 47 * b) >> 10) + 128).clamp(0, 255);
                (y, u, v)
            }
        };

        (y as u8, u as u8, v as u8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_engine_format_round_trips_through_the_caps_vocabulary() {
        for engine in PixelFormat::ALL {
            let caps: crate::format::PixelFormat = engine.into();
            assert_eq!(
                PixelFormat::try_from(caps),
                Ok(engine),
                "{engine:?} did not survive the round trip"
            );
        }
    }

    #[test]
    fn caps_only_formats_have_no_engine() {
        use crate::format::PixelFormat as Caps;
        // Caps can *name* these; there is no code that converts them.
        for caps in [
            Caps::I420_10Le,
            Caps::P010,
            Caps::I422,
            Caps::I444,
            Caps::Argb,
            Caps::Gray16Le,
        ] {
            let err = PixelFormat::try_from(caps).unwrap_err();
            assert_eq!(err, UnsupportedPixelFormat(caps));
            assert!(err.to_string().contains("no conversion engine"));
        }
    }

    // ========================================================================
    // Packed 4:2:2 -> 4:2:0 (#34)
    // ========================================================================

    /// A 4x4 YUYV frame with distinct luma per pixel and distinct chroma per
    /// horizontal pair, so a transposed or mis-sampled plane cannot pass.
    fn yuyv_4x4() -> Vec<u8> {
        let (w, h) = (4usize, 4usize);
        let mut data = vec![0u8; w * h * 2];
        for row in 0..h {
            for pair in 0..w / 2 {
                let i = row * w * 2 + pair * 4;
                data[i] = (row * 16 + pair * 2) as u8; // Y0
                data[i + 1] = (100 + row * 4 + pair) as u8; // U
                data[i + 2] = (row * 16 + pair * 2 + 1) as u8; // Y1
                data[i + 3] = (200 - row * 4 - pair) as u8; // V
            }
        }
        data
    }

    #[test]
    fn yuyv_to_i420_places_luma_and_chroma_correctly() {
        let src = yuyv_4x4();
        let conv = VideoConvert::new(PixelFormat::Yuyv, PixelFormat::I420, 4, 4).unwrap();
        let mut out = vec![0u8; conv.output_size()];
        conv.convert(&src, conv.packed_input_layout(), &mut out)
            .unwrap();

        // Luma passes through untouched, pixel for pixel.
        for row in 0..4usize {
            for col in 0..4usize {
                let pair = col / 2;
                let i = row * 8 + pair * 4;
                let expected = if col % 2 == 0 { src[i] } else { src[i + 2] };
                assert_eq!(out[row * 4 + col], expected, "luma at ({row}, {col})");
            }
        }

        // Chroma is averaged over the two rows of each 2x2 block. Row 0/1,
        // pair 0: U = mean(100, 104) = 102; V = mean(200, 196) = 198.
        let u_plane = &out[16..16 + 4];
        let v_plane = &out[20..20 + 4];
        assert_eq!(u_plane[0], 102);
        assert_eq!(v_plane[0], 198);
        // Row 2/3, pair 1: U = mean(100+8+1, 100+12+1) = mean(109, 113) = 111.
        assert_eq!(u_plane[3], 111);
        // V = mean(200-8-1, 200-12-1) = mean(191, 187) = 189.
        assert_eq!(v_plane[3], 189);
    }

    /// The direct path must agree with the round-trip it replaces: converting
    /// YUYV -> I420 -> RGB should land close to YUYV -> RGB.
    #[test]
    fn yuyv_to_i420_agrees_with_the_rgb_detour() {
        let src = yuyv_4x4();

        let direct = VideoConvert::new(PixelFormat::Yuyv, PixelFormat::Rgb24, 4, 4).unwrap();
        let mut rgb_direct = vec![0u8; direct.output_size()];
        direct
            .convert(&src, direct.packed_input_layout(), &mut rgb_direct)
            .unwrap();

        let to_i420 = VideoConvert::new(PixelFormat::Yuyv, PixelFormat::I420, 4, 4).unwrap();
        let mut i420 = vec![0u8; to_i420.output_size()];
        to_i420
            .convert(&src, to_i420.packed_input_layout(), &mut i420)
            .unwrap();

        let to_rgb = VideoConvert::new(PixelFormat::I420, PixelFormat::Rgb24, 4, 4).unwrap();
        let mut rgb_via_i420 = vec![0u8; to_rgb.output_size()];
        to_rgb
            .convert(&i420, to_rgb.packed_input_layout(), &mut rgb_via_i420)
            .unwrap();

        // Not identical: 4:2:0 loses vertical chroma resolution. But it must be
        // close — a wrong U/V assignment would swing colours wildly.
        for (i, (a, b)) in rgb_direct.iter().zip(&rgb_via_i420).enumerate() {
            let delta = (*a as i32 - *b as i32).abs();
            assert!(
                delta <= 24,
                "channel {i}: direct={a} via_i420={b} (delta {delta})"
            );
        }
    }

    #[test]
    fn uyvy_and_yuyv_carry_the_same_picture() {
        // Same data, different byte order: converting both to I420 must agree.
        let yuyv = yuyv_4x4();
        let mut uyvy = vec![0u8; yuyv.len()];
        for g in 0..yuyv.len() / 4 {
            let (i, j) = (g * 4, g * 4);
            uyvy[j] = yuyv[i + 1]; // U
            uyvy[j + 1] = yuyv[i]; // Y0
            uyvy[j + 2] = yuyv[i + 3]; // V
            uyvy[j + 3] = yuyv[i + 2]; // Y1
        }

        let from_yuyv = VideoConvert::new(PixelFormat::Yuyv, PixelFormat::I420, 4, 4).unwrap();
        let from_uyvy = VideoConvert::new(PixelFormat::Uyvy, PixelFormat::I420, 4, 4).unwrap();

        let mut a = vec![0u8; from_yuyv.output_size()];
        let mut b = vec![0u8; from_uyvy.output_size()];
        from_yuyv
            .convert(&yuyv, from_yuyv.packed_input_layout(), &mut a)
            .unwrap();
        from_uyvy
            .convert(&uyvy, from_uyvy.packed_input_layout(), &mut b)
            .unwrap();

        assert_eq!(
            a, b,
            "YUYV and UYVY of the same frame must yield the same I420"
        );
    }

    #[test]
    fn yuyv_to_nv12_interleaves_chroma() {
        let src = yuyv_4x4();

        let to_i420 = VideoConvert::new(PixelFormat::Yuyv, PixelFormat::I420, 4, 4).unwrap();
        let mut i420 = vec![0u8; to_i420.output_size()];
        to_i420
            .convert(&src, to_i420.packed_input_layout(), &mut i420)
            .unwrap();

        let to_nv12 = VideoConvert::new(PixelFormat::Yuyv, PixelFormat::Nv12, 4, 4).unwrap();
        let mut nv12 = vec![0u8; to_nv12.output_size()];
        to_nv12
            .convert(&src, to_nv12.packed_input_layout(), &mut nv12)
            .unwrap();

        assert_eq!(&nv12[..16], &i420[..16], "same luma plane");
        for i in 0..4 {
            assert_eq!(nv12[16 + i * 2], i420[16 + i], "U at chroma cell {i}");
            assert_eq!(nv12[16 + i * 2 + 1], i420[20 + i], "V at chroma cell {i}");
        }
    }

    #[test]
    fn i420_nv12_roundtrip_is_lossless() {
        // A plane interleave, so it must survive exactly.
        let src = yuyv_4x4();
        let to_i420 = VideoConvert::new(PixelFormat::Yuyv, PixelFormat::I420, 4, 4).unwrap();
        let mut i420 = vec![0u8; to_i420.output_size()];
        to_i420
            .convert(&src, to_i420.packed_input_layout(), &mut i420)
            .unwrap();

        let to_nv12 = VideoConvert::new(PixelFormat::I420, PixelFormat::Nv12, 4, 4).unwrap();
        let mut nv12 = vec![0u8; to_nv12.output_size()];
        to_nv12
            .convert(&i420, to_nv12.packed_input_layout(), &mut nv12)
            .unwrap();

        let back = VideoConvert::new(PixelFormat::Nv12, PixelFormat::I420, 4, 4).unwrap();
        let mut i420_again = vec![0u8; back.output_size()];
        back.convert(&nv12, back.packed_input_layout(), &mut i420_again)
            .unwrap();

        assert_eq!(i420, i420_again);
    }

    #[test]
    fn odd_dimensions_are_still_rejected_for_the_new_paths() {
        assert!(VideoConvert::new(PixelFormat::Yuyv, PixelFormat::I420, 5, 4).is_err());
        assert!(VideoConvert::new(PixelFormat::Yuyv, PixelFormat::Nv12, 4, 3).is_err());
    }

    #[test]
    fn test_pixel_format_buffer_size() {
        assert_eq!(PixelFormat::I420.buffer_size(4, 4), 4 * 4 + 2 * 2 + 2 * 2);
        assert_eq!(PixelFormat::Rgb24.buffer_size(4, 4), 4 * 4 * 3);
        assert_eq!(PixelFormat::Rgba.buffer_size(4, 4), 4 * 4 * 4);
        assert_eq!(PixelFormat::Gray8.buffer_size(4, 4), 4 * 4);
    }

    #[test]
    fn test_yuv_to_rgb_white() {
        let conv = VideoConvert::new(PixelFormat::I420, PixelFormat::Rgb24, 2, 2).unwrap();

        // White in YUV: Y=235, U=128, V=128 (full range would be Y=255)
        let (r, g, b) = conv.yuv_to_rgb(235, 128, 128);

        // Should be close to white
        assert!((r as i32 - 235).abs() < 5, "r={}", r);
        assert!((g as i32 - 235).abs() < 5, "g={}", g);
        assert!((b as i32 - 235).abs() < 5, "b={}", b);
    }

    #[test]
    fn test_yuv_to_rgb_black() {
        let conv = VideoConvert::new(PixelFormat::I420, PixelFormat::Rgb24, 2, 2).unwrap();

        // Black in YUV: Y=16, U=128, V=128 (limited range)
        let (r, g, b) = conv.yuv_to_rgb(16, 128, 128);

        // Should be close to black
        assert!((r as i32 - 16).abs() < 5, "r={}", r);
        assert!((g as i32 - 16).abs() < 5, "g={}", g);
        assert!((b as i32 - 16).abs() < 5, "b={}", b);
    }

    #[test]
    fn test_rgb_yuv_roundtrip() {
        let conv_to_yuv = VideoConvert::new(PixelFormat::Rgb24, PixelFormat::I420, 4, 4).unwrap();
        let conv_to_rgb = VideoConvert::new(PixelFormat::I420, PixelFormat::Rgb24, 4, 4).unwrap();

        // Create a uniform color pattern (minimizes chroma subsampling artifacts)
        // Each 2x2 block has the same color to avoid subsampling loss
        let mut rgb_in = vec![0u8; 4 * 4 * 3];
        let colors = [
            (180, 120, 80),  // Block 0,0
            (60, 180, 120),  // Block 0,1
            (120, 80, 180),  // Block 1,0
            (128, 128, 128), // Block 1,1 (gray)
        ];
        for row in 0..4 {
            for col in 0..4 {
                let block = (row / 2) * 2 + (col / 2);
                let (r, g, b) = colors[block];
                let idx = (row * 4 + col) * 3;
                rgb_in[idx] = r;
                rgb_in[idx + 1] = g;
                rgb_in[idx + 2] = b;
            }
        }

        let mut yuv = vec![0u8; PixelFormat::I420.buffer_size(4, 4)];
        let mut rgb_out = vec![0u8; 4 * 4 * 3];

        conv_to_yuv
            .convert(&rgb_in, conv_to_yuv.packed_input_layout(), &mut yuv)
            .unwrap();
        conv_to_rgb
            .convert(&yuv, conv_to_rgb.packed_input_layout(), &mut rgb_out)
            .unwrap();

        // Check that values are similar. With uniform 2x2 blocks, we should get
        // much closer values since no chroma information is lost to subsampling.
        // Allow up to 10 difference for rounding in the color matrix math.
        for i in 0..16 {
            let diff_r = (rgb_in[i * 3] as i32 - rgb_out[i * 3] as i32).abs();
            let diff_g = (rgb_in[i * 3 + 1] as i32 - rgb_out[i * 3 + 1] as i32).abs();
            let diff_b = (rgb_in[i * 3 + 2] as i32 - rgb_out[i * 3 + 2] as i32).abs();

            assert!(diff_r < 15, "pixel {} R diff {} too large", i, diff_r);
            assert!(diff_g < 15, "pixel {} G diff {} too large", i, diff_g);
            assert!(diff_b < 15, "pixel {} B diff {} too large", i, diff_b);
        }
    }

    #[test]
    fn test_rgb_bgr_swap() {
        let conv = VideoConvert::new(PixelFormat::Rgb24, PixelFormat::Bgr24, 2, 2).unwrap();

        let rgb = [255, 0, 0, 0, 255, 0, 0, 0, 255, 128, 128, 128]; // Red, Green, Blue, Gray
        let mut bgr = vec![0u8; 12];

        conv.convert(&rgb, conv.packed_input_layout(), &mut bgr)
            .unwrap();

        assert_eq!(bgr[0..3], [0, 0, 255]); // Red -> BGR
        assert_eq!(bgr[3..6], [0, 255, 0]); // Green stays
        assert_eq!(bgr[6..9], [255, 0, 0]); // Blue -> BGR
        assert_eq!(bgr[9..12], [128, 128, 128]); // Gray stays
    }

    #[test]
    fn test_add_remove_alpha() {
        let conv_add = VideoConvert::new(PixelFormat::Rgb24, PixelFormat::Rgba, 2, 2).unwrap();
        let conv_rem = VideoConvert::new(PixelFormat::Rgba, PixelFormat::Rgb24, 2, 2).unwrap();

        let rgb = [255, 128, 64, 32, 64, 128, 100, 150, 200, 50, 100, 150];
        let mut rgba = vec![0u8; 16];
        let mut rgb_out = vec![0u8; 12];

        conv_add
            .convert(&rgb, conv_add.packed_input_layout(), &mut rgba)
            .unwrap();

        // Check alpha was added
        assert_eq!(rgba[3], 255);
        assert_eq!(rgba[7], 255);
        assert_eq!(rgba[11], 255);
        assert_eq!(rgba[15], 255);

        conv_rem
            .convert(&rgba, conv_rem.packed_input_layout(), &mut rgb_out)
            .unwrap();

        // Check roundtrip
        assert_eq!(rgb, rgb_out.as_slice());
    }

    #[test]
    fn test_gray_to_rgb() {
        let conv = VideoConvert::new(PixelFormat::Gray8, PixelFormat::Rgb24, 2, 2).unwrap();

        let gray = [0, 85, 170, 255];
        let mut rgb = vec![0u8; 12];

        conv.convert(&gray, conv.packed_input_layout(), &mut rgb)
            .unwrap();

        assert_eq!(rgb[0..3], [0, 0, 0]);
        assert_eq!(rgb[3..6], [85, 85, 85]);
        assert_eq!(rgb[6..9], [170, 170, 170]);
        assert_eq!(rgb[9..12], [255, 255, 255]);
    }

    #[test]
    fn test_error_on_odd_dimensions_for_yuv() {
        let result = VideoConvert::new(PixelFormat::I420, PixelFormat::Rgb24, 3, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_same_format_copy() {
        let conv = VideoConvert::new(PixelFormat::Rgb24, PixelFormat::Rgb24, 2, 2).unwrap();

        let input = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let mut output = vec![0u8; 12];

        conv.convert(&input, conv.packed_input_layout(), &mut output)
            .unwrap();
        assert_eq!(input.as_slice(), output.as_slice());
    }

    /// Every engine format at every geometry the engine accepts: the packed
    /// [`PlaneLayout`] and the engine's own `buffer_size` must agree.
    ///
    /// They are computed independently — `plane_geometry` rounds chroma up
    /// with `div_ceil(2)`, `buffer_size` truncates with `/ 2` — and the
    /// stride work makes the layout the input-side authority, so a
    /// divergence would show up as a silent short-buffer read.
    #[test]
    fn packed_layout_len_agrees_with_engine_buffer_size() {
        const FORMATS: [PixelFormat; 9] = [
            PixelFormat::I420,
            PixelFormat::Nv12,
            PixelFormat::Yuyv,
            PixelFormat::Uyvy,
            PixelFormat::Rgb24,
            PixelFormat::Rgba,
            PixelFormat::Bgr24,
            PixelFormat::Bgra,
            PixelFormat::Gray8,
        ];
        for format in FORMATS {
            for (w, h) in [(2u32, 2u32), (16, 16), (64, 48), (640, 480), (1920, 1080)] {
                let caps: crate::format::PixelFormat = format.into();
                assert_eq!(
                    PlaneLayout::packed(caps, w, h).required_len(caps, w, h),
                    format.buffer_size(w, h),
                    "{format:?} at {w}x{h}"
                );
            }
            // Odd geometry only reaches the engine for non-YUV formats:
            // `VideoConvert::new` rejects odd dimensions whenever a YUV
            // format is involved.
            if !format.is_yuv() {
                for (w, h) in [(1u32, 1u32), (65, 49), (639, 481)] {
                    let caps: crate::format::PixelFormat = format.into();
                    assert_eq!(
                        PlaneLayout::packed(caps, w, h).required_len(caps, w, h),
                        format.buffer_size(w, h),
                        "{format:?} at {w}x{h}"
                    );
                }
            }
        }
    }

    /// Converting a strided frame must produce byte-identical output to
    /// converting its packed twin, for every `(input, output)` pair the
    /// engine supports.
    ///
    /// The two runs execute the same arithmetic over the same samples and
    /// differ only in addressing, so equality is exact. The padding carries
    /// a sentinel, so an arm that still derives a row start from width reads
    /// it and the comparison fails.
    ///
    /// Every input format the engine takes is covered: there is no arm
    /// left that derives a row start from width.
    #[test]
    fn strided_input_converts_identically_to_its_packed_twin() {
        use crate::converters::testutil::strided_twin;
        const W: u32 = 32;
        const H: u32 = 24;

        let mut pairs = 0usize;
        for input in PixelFormat::ALL {
            let caps: crate::format::PixelFormat = input.into();
            let packed: Vec<u8> = (0..input.buffer_size(W, H))
                .map(|i| (i % 251) as u8)
                .collect();
            let (strided, layout) = strided_twin(&packed, caps, W, H, 11);

            for output in PixelFormat::ALL {
                let Ok(conv) = VideoConvert::new(input, output, W, H) else {
                    continue;
                };
                let mut from_packed = vec![0u8; conv.output_size()];
                if conv
                    .convert(&packed, conv.packed_input_layout(), &mut from_packed)
                    .is_err()
                {
                    continue; // unsupported pair
                }
                let mut from_strided = vec![0u8; conv.output_size()];
                conv.convert(&strided, layout, &mut from_strided).unwrap();
                if from_packed != from_strided {
                    // Report where, not just that: a whole-image mismatch
                    // means a bad plane base, a trailing run means a
                    // dropped last row.
                    let first = from_packed
                        .iter()
                        .zip(&from_strided)
                        .position(|(a, b)| a != b)
                        .expect("differ");
                    let count = from_packed
                        .iter()
                        .zip(&from_strided)
                        .filter(|(a, b)| a != b)
                        .count();
                    panic!(
                        "{input:?} -> {output:?}: {count} of {} bytes differ, first at {first}",
                        from_packed.len()
                    );
                }
                pairs += 1;
            }
        }
        // Guards against the loop quietly covering nothing if `convert`
        // starts erroring for an unrelated reason.
        assert!(pairs >= 30, "only {pairs} pairs exercised");
    }

    /// A frame whose buffer ends tight against its last row is refused,
    /// not silently short-converted.
    ///
    /// The SIMD backend walks planes in `stride`-sized chunks and drops a
    /// partial trailing one, so the last row would vanish. Erroring is what
    /// lets the element repack instead — see
    /// [`PlaneLayout::full_span_len`](crate::format::PlaneLayout::full_span_len).
    #[test]
    fn a_tight_trailing_row_is_refused_by_the_engine() {
        use crate::converters::testutil::strided_twin;
        const W: u32 = 32;
        const H: u32 = 24;
        let caps: crate::format::PixelFormat = PixelFormat::I420.into();
        let packed: Vec<u8> = (0..PixelFormat::I420.buffer_size(W, H))
            .map(|i| (i % 251) as u8)
            .collect();
        let (strided, layout) = strided_twin(&packed, caps, W, H, 11);

        let tight = layout.required_len(caps, W, H);
        assert!(
            tight < layout.full_span_len(caps, W, H),
            "test premise: padding a stride leaves the last row short of a full chunk"
        );

        let conv = VideoConvert::new(PixelFormat::I420, PixelFormat::Rgba, W, H).unwrap();
        let mut out = vec![0u8; conv.output_size()];
        let err = conv
            .convert(&strided[..tight], layout, &mut out)
            .expect_err("a tight trailing row must not convert");
        assert!(format!("{err}").contains("Input buffer too small"), "{err}");
    }
}
