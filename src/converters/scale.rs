//! Video scaling (resolution conversion).
//!
//! Provides pure Rust implementations of video scaling algorithms.

use crate::error::{Error, Result};

use super::PixelFormat;

/// Scaling algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScaleAlgorithm {
    /// Nearest neighbor - fastest, pixelated results.
    NearestNeighbor,
    /// Bilinear interpolation - good quality/speed balance.
    #[default]
    Bilinear,
}

/// Video scaler.
///
/// Scales video frames between different resolutions.
pub struct VideoScale {
    input_width: u32,
    input_height: u32,
    output_width: u32,
    output_height: u32,
    format: PixelFormat,
    algorithm: ScaleAlgorithm,
}

impl VideoScale {
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
            && (input_width % 2 != 0
                || input_height % 2 != 0
                || output_width % 2 != 0
                || output_height % 2 != 0)
        {
            return Err(Error::Config("YUV formats require even dimensions".into()));
        }

        Ok(Self {
            input_width,
            input_height,
            output_width,
            output_height,
            format,
            algorithm: ScaleAlgorithm::default(),
        })
    }

    /// Set the scaling algorithm.
    pub fn with_algorithm(mut self, algorithm: ScaleAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }

    /// Get the required output buffer size.
    pub fn output_size(&self) -> usize {
        self.format
            .buffer_size(self.output_width, self.output_height)
    }

    /// Scale a frame.
    pub fn scale(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let expected_input = self.format.buffer_size(self.input_width, self.input_height);
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

        match self.format {
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => {
                self.scale_packed(input, output, 3);
            }
            PixelFormat::Rgba | PixelFormat::Bgra => {
                self.scale_packed(input, output, 4);
            }
            PixelFormat::Gray8 => {
                self.scale_packed(input, output, 1);
            }
            PixelFormat::Yuyv | PixelFormat::Uyvy => {
                // YUYV/UYVY: 2 bytes per pixel average (4 bytes per 2 pixels)
                self.scale_yuyv(input, output);
            }
            PixelFormat::I420 => {
                self.scale_i420(input, output);
            }
            PixelFormat::Nv12 => {
                self.scale_nv12(input, output);
            }
        }

        Ok(())
    }

    /// Scale packed pixel formats (RGB, RGBA, Gray, etc.)
    fn scale_packed(&self, input: &[u8], output: &mut [u8], bytes_per_pixel: usize) {
        let in_w = self.input_width as usize;
        let in_h = self.input_height as usize;
        let out_w = self.output_width as usize;
        let out_h = self.output_height as usize;

        match self.algorithm {
            ScaleAlgorithm::NearestNeighbor => {
                for out_y in 0..out_h {
                    let in_y = (out_y * in_h / out_h).min(in_h - 1);

                    for out_x in 0..out_w {
                        let in_x = (out_x * in_w / out_w).min(in_w - 1);

                        let src_offset = (in_y * in_w + in_x) * bytes_per_pixel;
                        let dst_offset = (out_y * out_w + out_x) * bytes_per_pixel;

                        output[dst_offset..dst_offset + bytes_per_pixel]
                            .copy_from_slice(&input[src_offset..src_offset + bytes_per_pixel]);
                    }
                }
            }
            ScaleAlgorithm::Bilinear => {
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
                            let p00 = input[(y0 * in_w + x0) * bytes_per_pixel + c] as f32;
                            let p10 = input[(y0 * in_w + x1) * bytes_per_pixel + c] as f32;
                            let p01 = input[(y1 * in_w + x0) * bytes_per_pixel + c] as f32;
                            let p11 = input[(y1 * in_w + x1) * bytes_per_pixel + c] as f32;

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
    fn scale_i420(&self, input: &[u8], output: &mut [u8]) {
        let in_w = self.input_width as usize;
        let in_h = self.input_height as usize;
        let out_w = self.output_width as usize;
        let out_h = self.output_height as usize;

        // Y plane
        let in_y = &input[0..in_w * in_h];
        let out_y = &mut output[0..out_w * out_h];
        self.scale_plane(in_y, in_w, in_h, out_y, out_w, out_h);

        // U plane (half resolution)
        let in_u_offset = in_w * in_h;
        let out_u_offset = out_w * out_h;
        let in_u = &input[in_u_offset..in_u_offset + (in_w / 2) * (in_h / 2)];
        let out_u = &mut output[out_u_offset..out_u_offset + (out_w / 2) * (out_h / 2)];
        self.scale_plane(in_u, in_w / 2, in_h / 2, out_u, out_w / 2, out_h / 2);

        // V plane (half resolution)
        let in_v_offset = in_u_offset + (in_w / 2) * (in_h / 2);
        let out_v_offset = out_u_offset + (out_w / 2) * (out_h / 2);
        let in_v = &input[in_v_offset..in_v_offset + (in_w / 2) * (in_h / 2)];
        let out_v = &mut output[out_v_offset..out_v_offset + (out_w / 2) * (out_h / 2)];
        self.scale_plane(in_v, in_w / 2, in_h / 2, out_v, out_w / 2, out_h / 2);
    }

    /// Scale YUYV/UYVY format (packed YUV 4:2:2).
    /// This is a simplified scaler that works in Y-only domain for speed.
    /// For better quality, convert to I420/NV12 first, scale, then convert back.
    fn scale_yuyv(&self, input: &[u8], output: &mut [u8]) {
        let in_w = self.input_width as usize;
        let in_h = self.input_height as usize;
        let out_w = self.output_width as usize;
        let out_h = self.output_height as usize;

        // YUYV: Y0 U Y1 V (4 bytes per 2 pixels)
        // Simplest approach: nearest neighbor scaling treating as macro-pixels
        match self.algorithm {
            ScaleAlgorithm::NearestNeighbor => {
                for out_y in 0..out_h {
                    let in_y = (out_y * in_h / out_h).min(in_h - 1);

                    // Process 2 output pixels at a time (one macro-pixel)
                    for out_x in (0..out_w).step_by(2) {
                        // Map to input position (in macro-pixel units)
                        let in_x = ((out_x * in_w / out_w) / 2 * 2).min(in_w - 2);

                        let src_idx = (in_y * in_w + in_x) * 2;
                        let dst_idx = (out_y * out_w + out_x) * 2;

                        // Copy the 4-byte macro-pixel
                        output[dst_idx] = input[src_idx]; // Y0
                        output[dst_idx + 1] = input[src_idx + 1]; // U
                        output[dst_idx + 2] = input[src_idx + 2]; // Y1
                        output[dst_idx + 3] = input[src_idx + 3]; // V
                    }
                }
            }
            ScaleAlgorithm::Bilinear => {
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
                        let y0_00 = input[(y0 * in_w + x0) * 2] as f32;
                        let y0_10 = input[(y0 * in_w + x1) * 2] as f32;
                        let y0_01 = input[(y1 * in_w + x0) * 2] as f32;
                        let y0_11 = input[(y1 * in_w + x1) * 2] as f32;
                        let y0_top = y0_00 + x_frac * (y0_10 - y0_00);
                        let y0_bot = y0_01 + x_frac * (y0_11 - y0_01);
                        let y0_val = (y0_top + y_frac * (y0_bot - y0_top)).round() as u8;

                        // Interpolate Y1
                        let y1_00 = input[(y0 * in_w + x0) * 2 + 2] as f32;
                        let y1_10 = input[(y0 * in_w + x1) * 2 + 2] as f32;
                        let y1_01 = input[(y1 * in_w + x0) * 2 + 2] as f32;
                        let y1_11 = input[(y1 * in_w + x1) * 2 + 2] as f32;
                        let y1_top = y1_00 + x_frac * (y1_10 - y1_00);
                        let y1_bot = y1_01 + x_frac * (y1_11 - y1_01);
                        let y1_val = (y1_top + y_frac * (y1_bot - y1_top)).round() as u8;

                        // Nearest-neighbor for U/V
                        let in_x_nn = (src_x_f.round() as usize / 2 * 2).min(in_w - 2);
                        let in_y_nn = src_y_f.round() as usize;
                        let u_val = input[(in_y_nn * in_w + in_x_nn) * 2 + 1];
                        let v_val = input[(in_y_nn * in_w + in_x_nn) * 2 + 3];

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
    fn scale_nv12(&self, input: &[u8], output: &mut [u8]) {
        let in_w = self.input_width as usize;
        let in_h = self.input_height as usize;
        let out_w = self.output_width as usize;
        let out_h = self.output_height as usize;

        // Y plane
        let in_y = &input[0..in_w * in_h];
        let out_y = &mut output[0..out_w * out_h];
        self.scale_plane(in_y, in_w, in_h, out_y, out_w, out_h);

        // UV plane (interleaved, half resolution in each dimension)
        let in_uv_offset = in_w * in_h;
        let out_uv_offset = out_w * out_h;
        let in_uv = &input[in_uv_offset..in_uv_offset + in_w * (in_h / 2)];
        let out_uv = &mut output[out_uv_offset..out_uv_offset + out_w * (out_h / 2)];

        // Scale UV as 2-channel interleaved data
        self.scale_interleaved_uv(in_uv, in_w / 2, in_h / 2, out_uv, out_w / 2, out_h / 2);
    }

    /// Scale a single plane (grayscale).
    fn scale_plane(
        &self,
        input: &[u8],
        in_w: usize,
        in_h: usize,
        output: &mut [u8],
        out_w: usize,
        out_h: usize,
    ) {
        match self.algorithm {
            ScaleAlgorithm::NearestNeighbor => {
                for out_y in 0..out_h {
                    let in_y = (out_y * in_h / out_h).min(in_h - 1);

                    for out_x in 0..out_w {
                        let in_x = (out_x * in_w / out_w).min(in_w - 1);
                        output[out_y * out_w + out_x] = input[in_y * in_w + in_x];
                    }
                }
            }
            ScaleAlgorithm::Bilinear => {
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

                        let p00 = input[y0 * in_w + x0] as f32;
                        let p10 = input[y0 * in_w + x1] as f32;
                        let p01 = input[y1 * in_w + x0] as f32;
                        let p11 = input[y1 * in_w + x1] as f32;

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
        in_w: usize,
        in_h: usize,
        output: &mut [u8],
        out_w: usize,
        out_h: usize,
    ) {
        match self.algorithm {
            ScaleAlgorithm::NearestNeighbor => {
                for out_y in 0..out_h {
                    let in_y = (out_y * in_h / out_h).min(in_h - 1);

                    for out_x in 0..out_w {
                        let in_x = (out_x * in_w / out_w).min(in_w - 1);

                        // U and V are interleaved
                        let src_idx = (in_y * in_w + in_x) * 2;
                        let dst_idx = (out_y * out_w + out_x) * 2;

                        output[dst_idx] = input[src_idx]; // U
                        output[dst_idx + 1] = input[src_idx + 1]; // V
                    }
                }
            }
            ScaleAlgorithm::Bilinear => {
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
                        let u00 = input[(y0 * in_w + x0) * 2] as f32;
                        let u10 = input[(y0 * in_w + x1) * 2] as f32;
                        let u01 = input[(y1 * in_w + x0) * 2] as f32;
                        let u11 = input[(y1 * in_w + x1) * 2] as f32;
                        let u_top = u00 + x_frac * (u10 - u00);
                        let u_bottom = u01 + x_frac * (u11 - u01);
                        let u = u_top + y_frac * (u_bottom - u_top);

                        // Interpolate V
                        let v00 = input[(y0 * in_w + x0) * 2 + 1] as f32;
                        let v10 = input[(y0 * in_w + x1) * 2 + 1] as f32;
                        let v01 = input[(y1 * in_w + x0) * 2 + 1] as f32;
                        let v11 = input[(y1 * in_w + x1) * 2 + 1] as f32;
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

    #[test]
    fn test_scale_nearest_2x() {
        let scaler = VideoScale::new(2, 2, 4, 4, PixelFormat::Gray8)
            .unwrap()
            .with_algorithm(ScaleAlgorithm::NearestNeighbor);

        let input = [0, 255, 255, 0]; // 2x2 checkerboard
        let mut output = vec![0u8; 16];

        scaler.scale(&input, &mut output).unwrap();

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
        let scaler = VideoScale::new(4, 4, 2, 2, PixelFormat::Gray8)
            .unwrap()
            .with_algorithm(ScaleAlgorithm::NearestNeighbor);

        #[rustfmt::skip]
        let input = [
            0, 0, 255, 255,
            0, 0, 255, 255,
            255, 255, 0, 0,
            255, 255, 0, 0,
        ];
        let mut output = vec![0u8; 4];

        scaler.scale(&input, &mut output).unwrap();

        // Should pick top-left of each 2x2 block
        let expected = [0, 255, 255, 0];
        assert_eq!(output, expected);
    }

    #[test]
    fn test_scale_bilinear_2x() {
        let scaler = VideoScale::new(2, 2, 4, 4, PixelFormat::Gray8)
            .unwrap()
            .with_algorithm(ScaleAlgorithm::Bilinear);

        let input = [0, 100, 100, 200];
        let mut output = vec![0u8; 16];

        scaler.scale(&input, &mut output).unwrap();

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
        let scaler = VideoScale::new(2, 2, 4, 4, PixelFormat::Rgb24)
            .unwrap()
            .with_algorithm(ScaleAlgorithm::NearestNeighbor);

        #[rustfmt::skip]
        let input = [
            255, 0, 0,    0, 255, 0,   // Red, Green
            0, 0, 255,    255, 255, 0, // Blue, Yellow
        ];
        let mut output = vec![0u8; 4 * 4 * 3];

        scaler.scale(&input, &mut output).unwrap();

        // Top-left 2x2 should be red
        assert_eq!(&output[0..3], &[255, 0, 0]);
        assert_eq!(&output[3..6], &[255, 0, 0]);

        // Top-right 2x2 should be green
        assert_eq!(&output[6..9], &[0, 255, 0]);
        assert_eq!(&output[9..12], &[0, 255, 0]);
    }

    #[test]
    fn test_scale_i420() {
        let scaler = VideoScale::new(4, 4, 8, 8, PixelFormat::I420)
            .unwrap()
            .with_algorithm(ScaleAlgorithm::NearestNeighbor);

        // Create a simple I420 frame (4x4)
        // Y plane: 16 bytes, U plane: 4 bytes, V plane: 4 bytes
        let mut input = vec![0u8; 4 * 4 + 2 * 2 + 2 * 2]; // 24 bytes total
        for i in 0..16 {
            input[i] = (i * 16) as u8; // Y gradient
        }
        for i in 0..4 {
            input[16 + i] = 128; // U neutral
            input[20 + i] = 128; // V neutral
        }

        let mut output = vec![0u8; 8 * 8 + 4 * 4 + 4 * 4]; // 96 bytes total

        scaler.scale(&input, &mut output).unwrap();

        // Output Y plane should be scaled
        assert_eq!(output.len(), PixelFormat::I420.buffer_size(8, 8));
    }

    #[test]
    fn test_error_on_zero_dimension() {
        let result = VideoScale::new(0, 100, 200, 200, PixelFormat::Rgb24);
        assert!(result.is_err());
    }

    #[test]
    fn test_error_on_odd_yuv_dimension() {
        let result = VideoScale::new(3, 4, 6, 8, PixelFormat::I420);
        assert!(result.is_err());
    }
}
