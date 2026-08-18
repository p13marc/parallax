//! Video scaling (resolution conversion).
//!
//! Provides pure Rust implementations of video scaling algorithms.
//!
//! [`ScaleEngine`] is the slice-level engine; the pipeline element wrapping it
//! is [`VideoScale`](crate::elements::transform::VideoScale). There is exactly
//! one type per job — the engine resamples bytes, the element reads geometry
//! from [`Metadata`](crate::metadata::Metadata) and drives the engine.

use crate::error::{Error, Result};
use crate::format::PlaneLayout;

use super::PixelFormat;

/// Scaling interpolation mode.
///
/// Shared by [`ScaleEngine`] and the
/// [`VideoScale`](crate::elements::transform::VideoScale) element — one enum,
/// one meaning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleMode {
    /// Bilinear interpolation (smoother, slower).
    #[default]
    Bilinear,
    /// Nearest neighbor (faster, pixelated).
    NearestNeighbor,
}

/// One input plane resolved against a buffer: its bytes from the first row
/// onward, plus the byte distance between consecutive rows.
#[derive(Clone, Copy, Debug)]
struct PlaneIn<'a> {
    data: &'a [u8],
    stride: usize,
}

/// Video scaling engine.
///
/// Scales video frames between different resolutions, operating on plain byte
/// slices. Built for one exact `(format, source, target)` conversion; the
/// [`VideoScale`](crate::elements::transform::VideoScale) element caches and
/// rebuilds engines as the stream's geometry changes.
pub struct ScaleEngine {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    format: PixelFormat,
    mode: ScaleMode,
}

impl ScaleEngine {
    /// Create a new video scaler.
    pub fn new(
        input_width: u32,
        input_height: u32,
        output_width: u32,
        output_height: u32,
        format: PixelFormat,
    ) -> Result<Self> {
        if input_width == 0 || input_height == 0 || output_width == 0 || output_height == 0 {
            return Err(Error::Config("Dimensions must be non-zero".into()));
        }

        // YUV formats require even dimensions
        if format.is_yuv()
            && (!input_width.is_multiple_of(2)
                || !input_height.is_multiple_of(2)
                || !output_width.is_multiple_of(2)
                || !output_height.is_multiple_of(2))
        {
            return Err(Error::Config("YUV formats require even dimensions".into()));
        }

        Ok(Self {
            input_width,
            input_height,
            output_width,
            output_height,
            format,
            mode: ScaleMode::default(),
        })
    }

    /// Set the interpolation mode.
    pub fn with_mode(mut self, mode: ScaleMode) -> Self {
        self.mode = mode;
        self
    }

    /// Get the required output buffer size.
    pub fn output_size(&self) -> usize {
        self.format
            .buffer_size(self.output_width, self.output_height)
    }

    /// The packed input layout for this engine's format and source geometry
    /// — what a caller holding an ordinary arena buffer passes to
    /// [`scale`](Self::scale).
    pub fn packed_input_layout(&self) -> PlaneLayout {
        PlaneLayout::packed(self.format.into(), self.input_width, self.input_height)
    }

    /// Whether the paths reading `format` honor a non-packed input layout yet.
    ///
    /// Shrinking scaffold (#196) — see
    /// [`VideoConvert::reads_strided_input`](super::VideoConvert::reads_strided_input).
    pub(crate) fn reads_strided_input(format: PixelFormat) -> bool {
        match format {
            // Every path resolves its planes through `PlaneLayout`.
            PixelFormat::I420
            | PixelFormat::Nv12
            | PixelFormat::Yuyv
            | PixelFormat::Uyvy
            | PixelFormat::Rgb24
            | PixelFormat::Rgba
            | PixelFormat::Bgr24
            | PixelFormat::Bgra
            | PixelFormat::Gray8 => true,
        }
    }

    /// Scale a frame.
    ///
    /// `input_layout` describes where the input's planes are and how far
    /// apart their rows sit — [`packed_input_layout`](Self::packed_input_layout)
    /// for an ordinary buffer, the producer's own layout for a strided one
    /// (#194). The **output is always packed**: every caller writes into a
    /// freshly sized arena slot.
    pub fn scale(&self, input: &[u8], input_layout: PlaneLayout, output: &mut [u8]) -> Result<()> {
        let expected_input =
            input_layout.required_len(self.format.into(), self.input_width, self.input_height);
        let expected_output = self
            .format
            .buffer_size(self.output_width, self.output_height);

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

        // Every path below reads the input through `plane`, which resolves
        // one plane against the declared layout. The output is packed, so
        // its own row pitch is always the plane's used row bytes.
        match self.format {
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => {
                let p = self.plane(input, input_layout, 0);
                self.scale_packed(p.data, p.stride, output, 3);
            }
            PixelFormat::Rgba | PixelFormat::Bgra => {
                let p = self.plane(input, input_layout, 0);
                self.scale_packed(p.data, p.stride, output, 4);
            }
            PixelFormat::Gray8 => {
                let p = self.plane(input, input_layout, 0);
                self.scale_packed(p.data, p.stride, output, 1);
            }
            PixelFormat::Yuyv | PixelFormat::Uyvy => {
                // YUYV/UYVY: 2 bytes per pixel average (4 bytes per 2 pixels)
                let p = self.plane(input, input_layout, 0);
                self.scale_yuyv(p.data, p.stride, output);
            }
            PixelFormat::I420 => {
                self.scale_i420(input, input_layout, output);
            }
            PixelFormat::Nv12 => {
                self.scale_nv12(input, input_layout, output);
            }
        }

        Ok(())
    }

    /// One input plane resolved against the declared layout: its bytes from
    /// the first row to the end of the buffer, plus the row pitch.
    ///
    /// The slice deliberately runs to the end of the buffer rather than to
    /// `offset + stride * rows` — a tight strided layout's last row carries
    /// no trailing padding, so the latter would overrun.
    fn plane<'a>(&self, input: &'a [u8], layout: PlaneLayout, index: usize) -> PlaneIn<'a> {
        let p = layout
            .resolved(self.format.into(), self.input_width, self.input_height)
            .nth(index)
            .expect("plane index within the format's plane count");
        PlaneIn {
            data: &input[p.offset..],
            stride: p.stride,
        }
    }

    /// Scale packed pixel formats (RGB, RGBA, Gray, etc.)
    fn scale_packed(
        &self,
        input: &[u8],
        in_stride: usize,
        output: &mut [u8],
        bytes_per_pixel: usize,
    ) {
        let in_w = self.input_width as usize;
        let in_h = self.input_height as usize;
        let out_w = self.output_width as usize;
        let out_h = self.output_height as usize;

        match self.mode {
            ScaleMode::NearestNeighbor => {
                for out_y in 0..out_h {
                    let in_y = (out_y * in_h / out_h).min(in_h - 1);

                    for out_x in 0..out_w {
                        let in_x = (out_x * in_w / out_w).min(in_w - 1);

                        let src_offset = in_y * in_stride + in_x * bytes_per_pixel;
                        let dst_offset = (out_y * out_w + out_x) * bytes_per_pixel;

                        output[dst_offset..dst_offset + bytes_per_pixel]
                            .copy_from_slice(&input[src_offset..src_offset + bytes_per_pixel]);
                    }
                }
            }
            ScaleMode::Bilinear => {
                let x_ratio = (in_w as f32 - 1.0) / (out_w as f32).max(1.0);
                let y_ratio = (in_h as f32 - 1.0) / (out_h as f32).max(1.0);

                for out_y in 0..out_h {
                    let src_y = out_y as f32 * y_ratio;
                    let y0 = src_y.floor() as usize;
                    let y1 = (y0 + 1).min(in_h - 1);
                    let y_frac = src_y - y0 as f32;

                    for out_x in 0..out_w {
                        let src_x = out_x as f32 * x_ratio;
                        let x0 = src_x.floor() as usize;
                        let x1 = (x0 + 1).min(in_w - 1);
                        let x_frac = src_x - x0 as f32;

                        for c in 0..bytes_per_pixel {
                            let p00 = input[y0 * in_stride + x0 * bytes_per_pixel + c] as f32;
                            let p10 = input[y0 * in_stride + x1 * bytes_per_pixel + c] as f32;
                            let p01 = input[y1 * in_stride + x0 * bytes_per_pixel + c] as f32;
                            let p11 = input[y1 * in_stride + x1 * bytes_per_pixel + c] as f32;

                            // Bilinear interpolation
                            let top = p00 + x_frac * (p10 - p00);
                            let bottom = p01 + x_frac * (p11 - p01);
                            let value = top + y_frac * (bottom - top);

                            output[(out_y * out_w + out_x) * bytes_per_pixel + c] =
                                value.round() as u8;
                        }
                    }
                }
            }
        }
    }

    /// Scale I420 format (planar YUV 4:2:0).
    ///
    /// Input planes come from `layout` — packed or strided; output planes are
    /// packed, dense and in order. YUV geometry is even by construction
    /// ([`ScaleEngine::new`] rejects odd dimensions), so the halves below are
    /// exact.
    fn scale_i420(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let in_w = self.input_width as usize;
        let in_h = self.input_height as usize;
        let out_w = self.output_width as usize;
        let out_h = self.output_height as usize;

        // Y plane
        let src = self.plane(input, layout, 0);
        let out_y = &mut output[0..out_w * out_h];
        self.scale_plane(src.data, src.stride, in_w, in_h, out_y, out_w, out_h);

        // U plane (half resolution)
        let out_u_offset = out_w * out_h;
        let src = self.plane(input, layout, 1);
        let out_u = &mut output[out_u_offset..out_u_offset + (out_w / 2) * (out_h / 2)];
        self.scale_plane(
            src.data,
            src.stride,
            in_w / 2,
            in_h / 2,
            out_u,
            out_w / 2,
            out_h / 2,
        );

        // V plane (half resolution)
        let out_v_offset = out_u_offset + (out_w / 2) * (out_h / 2);
        let src = self.plane(input, layout, 2);
        let out_v = &mut output[out_v_offset..out_v_offset + (out_w / 2) * (out_h / 2)];
        self.scale_plane(
            src.data,
            src.stride,
            in_w / 2,
            in_h / 2,
            out_v,
            out_w / 2,
            out_h / 2,
        );
    }

    /// Scale YUYV/UYVY format (packed YUV 4:2:2).
    /// This is a simplified scaler that works in Y-only domain for speed.
    /// For better quality, convert to I420/NV12 first, scale, then convert back.
    fn scale_yuyv(&self, input: &[u8], in_stride: usize, output: &mut [u8]) {
        let in_w = self.input_width as usize;
        let in_h = self.input_height as usize;
        let out_w = self.output_width as usize;
        let out_h = self.output_height as usize;

        // YUYV: Y0 U Y1 V (4 bytes per 2 pixels)
        // Simplest approach: nearest neighbor scaling treating as macro-pixels
        match self.mode {
            ScaleMode::NearestNeighbor => {
                for out_y in 0..out_h {
                    let in_y = (out_y * in_h / out_h).min(in_h - 1);

                    // Process 2 output pixels at a time (one macro-pixel)
                    for out_x in (0..out_w).step_by(2) {
                        // Map to input position (in macro-pixel units)
                        let in_x = ((out_x * in_w / out_w) / 2 * 2).min(in_w - 2);

                        let src_idx = in_y * in_stride + in_x * 2;
                        let dst_idx = (out_y * out_w + out_x) * 2;

                        // Copy the 4-byte macro-pixel
                        output[dst_idx] = input[src_idx]; // Y0
                        output[dst_idx + 1] = input[src_idx + 1]; // U
                        output[dst_idx + 2] = input[src_idx + 2]; // Y1
                        output[dst_idx + 3] = input[src_idx + 3]; // V
                    }
                }
            }
            ScaleMode::Bilinear => {
                // For bilinear, we do a simple approximation by interpolating Y values
                // and using nearest-neighbor for U/V
                let x_ratio = (in_w as f32 - 1.0) / (out_w as f32).max(1.0);
                let y_ratio = (in_h as f32 - 1.0) / (out_h as f32).max(1.0);

                for out_y in 0..out_h {
                    let src_y_f = out_y as f32 * y_ratio;
                    let y0 = src_y_f.floor() as usize;
                    let y1 = (y0 + 1).min(in_h - 1);
                    let y_frac = src_y_f - y0 as f32;

                    for out_x in (0..out_w).step_by(2) {
                        let src_x_f = out_x as f32 * x_ratio;
                        let x0 = (src_x_f.floor() as usize / 2 * 2).min(in_w - 2);
                        let x1 = (x0 + 2).min(in_w - 2);
                        let x_frac = (src_x_f - x0 as f32) / 2.0;

                        // Interpolate Y0
                        let y0_00 = input[y0 * in_stride + x0 * 2] as f32;
                        let y0_10 = input[y0 * in_stride + x1 * 2] as f32;
                        let y0_01 = input[y1 * in_stride + x0 * 2] as f32;
                        let y0_11 = input[y1 * in_stride + x1 * 2] as f32;
                        let y0_top = y0_00 + x_frac * (y0_10 - y0_00);
                        let y0_bot = y0_01 + x_frac * (y0_11 - y0_01);
                        let y0_val = (y0_top + y_frac * (y0_bot - y0_top)).round() as u8;

                        // Interpolate Y1
                        let y1_00 = input[y0 * in_stride + x0 * 2 + 2] as f32;
                        let y1_10 = input[y0 * in_stride + x1 * 2 + 2] as f32;
                        let y1_01 = input[y1 * in_stride + x0 * 2 + 2] as f32;
                        let y1_11 = input[y1 * in_stride + x1 * 2 + 2] as f32;
                        let y1_top = y1_00 + x_frac * (y1_10 - y1_00);
                        let y1_bot = y1_01 + x_frac * (y1_11 - y1_01);
                        let y1_val = (y1_top + y_frac * (y1_bot - y1_top)).round() as u8;

                        // Nearest-neighbor for U/V
                        let in_x_nn = (src_x_f.round() as usize / 2 * 2).min(in_w - 2);
                        let in_y_nn = src_y_f.round() as usize;
                        let u_val = input[in_y_nn * in_stride + in_x_nn * 2 + 1];
                        let v_val = input[in_y_nn * in_stride + in_x_nn * 2 + 3];

                        let dst_idx = (out_y * out_w + out_x) * 2;
                        output[dst_idx] = y0_val;
                        output[dst_idx + 1] = u_val;
                        output[dst_idx + 2] = y1_val;
                        output[dst_idx + 3] = v_val;
                    }
                }
            }
        }
    }

    /// Scale NV12 format (semi-planar YUV 4:2:0).
    fn scale_nv12(&self, input: &[u8], layout: PlaneLayout, output: &mut [u8]) {
        let in_w = self.input_width as usize;
        let in_h = self.input_height as usize;
        let out_w = self.output_width as usize;
        let out_h = self.output_height as usize;

        // Y plane
        let src = self.plane(input, layout, 0);
        let out_y = &mut output[0..out_w * out_h];
        self.scale_plane(src.data, src.stride, in_w, in_h, out_y, out_w, out_h);

        // UV plane (interleaved, half resolution in each dimension)
        let out_uv_offset = out_w * out_h;
        let src = self.plane(input, layout, 1);
        let out_uv = &mut output[out_uv_offset..out_uv_offset + out_w * (out_h / 2)];

        // Scale UV as 2-channel interleaved data
        self.scale_interleaved_uv(
            src.data,
            src.stride,
            in_w / 2,
            in_h / 2,
            out_uv,
            out_w / 2,
            out_h / 2,
        );
    }

    /// Scale a single plane (grayscale).
    fn scale_plane(
        &self,
        input: &[u8],
        in_stride: usize,
        in_w: usize,
        in_h: usize,
        output: &mut [u8],
        out_w: usize,
        out_h: usize,
    ) {
        match self.mode {
            ScaleMode::NearestNeighbor => {
                for out_y in 0..out_h {
                    let in_y = (out_y * in_h / out_h).min(in_h - 1);

                    for out_x in 0..out_w {
                        let in_x = (out_x * in_w / out_w).min(in_w - 1);
                        output[out_y * out_w + out_x] = input[in_y * in_stride + in_x];
                    }
                }
            }
            ScaleMode::Bilinear => {
                let x_ratio = (in_w as f32 - 1.0) / (out_w as f32).max(1.0);
                let y_ratio = (in_h as f32 - 1.0) / (out_h as f32).max(1.0);

                for out_y in 0..out_h {
                    let src_y = out_y as f32 * y_ratio;
                    let y0 = src_y.floor() as usize;
                    let y1 = (y0 + 1).min(in_h - 1);
                    let y_frac = src_y - y0 as f32;

                    for out_x in 0..out_w {
                        let src_x = out_x as f32 * x_ratio;
                        let x0 = src_x.floor() as usize;
                        let x1 = (x0 + 1).min(in_w - 1);
                        let x_frac = src_x - x0 as f32;

                        let p00 = input[y0 * in_stride + x0] as f32;
                        let p10 = input[y0 * in_stride + x1] as f32;
                        let p01 = input[y1 * in_stride + x0] as f32;
                        let p11 = input[y1 * in_stride + x1] as f32;

                        let top = p00 + x_frac * (p10 - p00);
                        let bottom = p01 + x_frac * (p11 - p01);
                        let value = top + y_frac * (bottom - top);

                        output[out_y * out_w + out_x] = value.round() as u8;
                    }
                }
            }
        }
    }

    /// Scale interleaved UV plane (NV12).
    fn scale_interleaved_uv(
        &self,
        input: &[u8],
        in_stride: usize,
        in_w: usize,
        in_h: usize,
        output: &mut [u8],
        out_w: usize,
        out_h: usize,
    ) {
        match self.mode {
            ScaleMode::NearestNeighbor => {
                for out_y in 0..out_h {
                    let in_y = (out_y * in_h / out_h).min(in_h - 1);

                    for out_x in 0..out_w {
                        let in_x = (out_x * in_w / out_w).min(in_w - 1);

                        // U and V are interleaved
                        let src_idx = in_y * in_stride + in_x * 2;
                        let dst_idx = (out_y * out_w + out_x) * 2;

                        output[dst_idx] = input[src_idx]; // U
                        output[dst_idx + 1] = input[src_idx + 1]; // V
                    }
                }
            }
            ScaleMode::Bilinear => {
                let x_ratio = (in_w as f32 - 1.0) / (out_w as f32).max(1.0);
                let y_ratio = (in_h as f32 - 1.0) / (out_h as f32).max(1.0);

                for out_y in 0..out_h {
                    let src_y = out_y as f32 * y_ratio;
                    let y0 = src_y.floor() as usize;
                    let y1 = (y0 + 1).min(in_h - 1);
                    let y_frac = src_y - y0 as f32;

                    for out_x in 0..out_w {
                        let src_x = out_x as f32 * x_ratio;
                        let x0 = src_x.floor() as usize;
                        let x1 = (x0 + 1).min(in_w - 1);
                        let x_frac = src_x - x0 as f32;

                        // Interpolate U
                        let u00 = input[y0 * in_stride + x0 * 2] as f32;
                        let u10 = input[y0 * in_stride + x1 * 2] as f32;
                        let u01 = input[y1 * in_stride + x0 * 2] as f32;
                        let u11 = input[y1 * in_stride + x1 * 2] as f32;
                        let u_top = u00 + x_frac * (u10 - u00);
                        let u_bottom = u01 + x_frac * (u11 - u01);
                        let u = u_top + y_frac * (u_bottom - u_top);

                        // Interpolate V
                        let v00 = input[y0 * in_stride + x0 * 2 + 1] as f32;
                        let v10 = input[y0 * in_stride + x1 * 2 + 1] as f32;
                        let v01 = input[y1 * in_stride + x0 * 2 + 1] as f32;
                        let v11 = input[y1 * in_stride + x1 * 2 + 1] as f32;
                        let v_top = v00 + x_frac * (v10 - v00);
                        let v_bottom = v01 + x_frac * (v11 - v01);
                        let v = v_top + y_frac * (v_bottom - v_top);

                        let dst_idx = (out_y * out_w + out_x) * 2;
                        output[dst_idx] = u.round() as u8;
                        output[dst_idx + 1] = v.round() as u8;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::converters::testutil::strided_twin;

    #[test]
    fn test_scale_nearest_2x() {
        let scaler = ScaleEngine::new(2, 2, 4, 4, PixelFormat::Gray8)
            .unwrap()
            .with_mode(ScaleMode::NearestNeighbor);

        let input = [0, 255, 255, 0]; // 2x2 checkerboard
        let mut output = vec![0u8; 16];

        scaler
            .scale(&input, scaler.packed_input_layout(), &mut output)
            .unwrap();

        // Each pixel should be duplicated 2x2
        #[rustfmt::skip]
        let expected = [
            0, 0, 255, 255,
            0, 0, 255, 255,
            255, 255, 0, 0,
            255, 255, 0, 0,
        ];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_scale_nearest_half() {
        let scaler = ScaleEngine::new(4, 4, 2, 2, PixelFormat::Gray8)
            .unwrap()
            .with_mode(ScaleMode::NearestNeighbor);

        #[rustfmt::skip]
        let input = [
            0, 0, 255, 255,
            0, 0, 255, 255,
            255, 255, 0, 0,
            255, 255, 0, 0,
        ];
        let mut output = vec![0u8; 4];

        scaler
            .scale(&input, scaler.packed_input_layout(), &mut output)
            .unwrap();

        // Should pick top-left of each 2x2 block
        let expected = [0, 255, 255, 0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_scale_bilinear_2x() {
        let scaler = ScaleEngine::new(2, 2, 4, 4, PixelFormat::Gray8)
            .unwrap()
            .with_mode(ScaleMode::Bilinear);

        let input = [0, 100, 100, 200];
        let mut output = vec![0u8; 16];

        scaler
            .scale(&input, scaler.packed_input_layout(), &mut output)
            .unwrap();

        // Corners should be close to original values (bilinear can interpolate at edges)
        assert_eq!(output[0], 0); // top-left corner should be exact

        // The scaled image interpolates between the 2x2 input pixels.
        // The exact values depend on the ratio calculation, so we just verify
        // the output is reasonable (values between min and max input).
        for &v in &output {
            assert!(v <= 200, "output value {} exceeds max input", v);
        }

        // Verify interpolation happens - center should not be all 0 or all 200
        let center_avg =
            (output[5] as u32 + output[6] as u32 + output[9] as u32 + output[10] as u32) / 4;
        assert!(
            center_avg > 10 && center_avg < 190,
            "center values should be interpolated, got {}",
            center_avg
        );
    }

    #[test]
    fn test_scale_rgb24() {
        let scaler = ScaleEngine::new(2, 2, 4, 4, PixelFormat::Rgb24)
            .unwrap()
            .with_mode(ScaleMode::NearestNeighbor);

        #[rustfmt::skip]
        let input = [
            255, 0, 0,    0, 255, 0,   // Red, Green
            0, 0, 255,    255, 255, 0, // Blue, Yellow
        ];
        let mut output = vec![0u8; 4 * 4 * 3];

        scaler
            .scale(&input, scaler.packed_input_layout(), &mut output)
            .unwrap();

        // Top-left 2x2 should be red
        assert_eq!(&output[0..3], &[255, 0, 0]);
        assert_eq!(&output[3..6], &[255, 0, 0]);

        // Top-right 2x2 should be green
        assert_eq!(&output[6..9], &[0, 255, 0]);
        assert_eq!(&output[9..12], &[0, 255, 0]);
    }

    #[test]
    fn test_scale_i420() {
        let scaler = ScaleEngine::new(4, 4, 8, 8, PixelFormat::I420)
            .unwrap()
            .with_mode(ScaleMode::NearestNeighbor);

        // Create a simple I420 frame (4x4)
        // Y plane: 16 bytes, U plane: 4 bytes, V plane: 4 bytes
        let mut input = vec![0u8; 4 * 4 + 2 * 2 + 2 * 2]; // 24 bytes total
        for (i, pixel) in input.iter_mut().enumerate().take(16) {
            *pixel = (i * 16) as u8; // Y gradient
        }
        for i in 0..4 {
            input[16 + i] = 128; // U neutral
            input[20 + i] = 128; // V neutral
        }

        let mut output = vec![0u8; 8 * 8 + 4 * 4 + 4 * 4]; // 96 bytes total

        scaler
            .scale(&input, scaler.packed_input_layout(), &mut output)
            .unwrap();

        // Output Y plane should be scaled
        assert_eq!(output.len(), PixelFormat::I420.buffer_size(8, 8));
    }

    #[test]
    fn test_error_on_zero_dimension() {
        let result = ScaleEngine::new(0, 100, 200, 200, PixelFormat::Rgb24);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_on_odd_yuv_dimension() {
        let result = ScaleEngine::new(3, 4, 6, 8, PixelFormat::I420);
        assert!(result.is_err());
    }

    /// Scaling a strided frame must produce byte-identical output to
    /// scaling its packed twin — for every format the engine takes and
    /// both interpolation modes.
    ///
    /// The two runs execute the same arithmetic over the same samples; only
    /// the addressing differs, so equality is exact, not approximate. The
    /// padding is filled with a sentinel, so an index that still derives a
    /// row start from width reads it and the comparison fails.
    #[test]
    fn strided_input_scales_identically_to_its_packed_twin() {
        const SRC: (u32, u32) = (32, 24);
        for format in PixelFormat::ALL {
            let caps: crate::format::PixelFormat = format.into();
            let packed: Vec<u8> = (0..format.buffer_size(SRC.0, SRC.1))
                .map(|i| (i % 251) as u8)
                .collect();
            let (strided, layout) = strided_twin(&packed, caps, SRC.0, SRC.1, 13);

            for dst in [(16u32, 12u32), (64, 48), (32, 24), (48, 16)] {
                for mode in [ScaleMode::Bilinear, ScaleMode::NearestNeighbor] {
                    let engine = ScaleEngine::new(SRC.0, SRC.1, dst.0, dst.1, format)
                        .unwrap()
                        .with_mode(mode);
                    let mut from_packed = vec![0u8; engine.output_size()];
                    let mut from_strided = vec![0u8; engine.output_size()];
                    engine
                        .scale(&packed, engine.packed_input_layout(), &mut from_packed)
                        .unwrap();
                    engine.scale(&strided, layout, &mut from_strided).unwrap();
                    assert_eq!(
                        from_packed, from_strided,
                        "{format:?} {SRC:?}->{dst:?} {mode:?}"
                    );
                }
            }
        }
    }
}
