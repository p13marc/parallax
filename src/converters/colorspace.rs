//! Pixel format conversion (colorspace conversion).
//!
//! Provides pure Rust implementations of YUV ↔ RGB conversions using
//! standard color matrices (BT.601, BT.709).

use crate::error::{Error, Result};

/// Pixel format enumeration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// Planar YUV 4:2:0 (Y plane, then U plane, then V plane)
    I420,
    /// Semi-planar YUV 4:2:0 (Y plane, then interleaved UV plane)
    Nv12,
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
    /// For planar formats, returns None.
    pub fn bytes_per_pixel(&self) -> Option<usize> {
        match self {
            PixelFormat::I420 | PixelFormat::Nv12 => None,
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
            PixelFormat::Rgb24 | PixelFormat::Bgr24 => w * h * 3,
            PixelFormat::Rgba | PixelFormat::Bgra => w * h * 4,
            PixelFormat::Gray8 => w * h,
        }
    }

    /// Returns true if this is a YUV format.
    pub fn is_yuv(&self) -> bool {
        matches!(self, PixelFormat::I420 | PixelFormat::Nv12)
    }

    /// Returns true if this is an RGB format.
    pub fn is_rgb(&self) -> bool {
        matches!(
            self,
            PixelFormat::Rgb24 | PixelFormat::Rgba | PixelFormat::Bgr24 | PixelFormat::Bgra
        )
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

/// Video format converter.
///
/// Converts between pixel formats while maintaining the same resolution.
/// For resolution changes, use [`VideoScale`](super::VideoScale).
pub struct VideoConvert {
    input_format: PixelFormat,
    output_format: PixelFormat,
    width: u32,
    height: u32,
    color_matrix: ColorMatrix,
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
        if (input_format.is_yuv() || output_format.is_yuv()) && (width % 2 != 0 || height % 2 != 0)
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

    /// Convert a frame from input format to output format.
    pub fn convert(&self, input: &[u8], output: &mut [u8]) -> Result<()> {
        let expected_input = self.input_format.buffer_size(self.width, self.height);
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

        // Dispatch to specific conversion
        match (self.input_format, self.output_format) {
            // Same format - just copy
            (a, b) if a == b => {
                output[..expected_input].copy_from_slice(&input[..expected_input]);
            }

            // YUV to RGB conversions
            (PixelFormat::I420, PixelFormat::Rgb24) => {
                self.i420_to_rgb24(input, output);
            }
            (PixelFormat::I420, PixelFormat::Rgba) => {
                self.i420_to_rgba(input, output);
            }
            (PixelFormat::I420, PixelFormat::Bgr24) => {
                self.i420_to_bgr24(input, output);
            }
            (PixelFormat::I420, PixelFormat::Bgra) => {
                self.i420_to_bgra(input, output);
            }
            (PixelFormat::Nv12, PixelFormat::Rgb24) => {
                self.nv12_to_rgb24(input, output);
            }
            (PixelFormat::Nv12, PixelFormat::Rgba) => {
                self.nv12_to_rgba(input, output);
            }

            // RGB to YUV conversions
            (PixelFormat::Rgb24, PixelFormat::I420) => {
                self.rgb24_to_i420(input, output);
            }
            (PixelFormat::Rgba, PixelFormat::I420) => {
                self.rgba_to_i420(input, output);
            }

            // RGB swizzle conversions
            (PixelFormat::Rgb24, PixelFormat::Bgr24) => {
                self.rgb_bgr_swap(input, output, 3);
            }
            (PixelFormat::Bgr24, PixelFormat::Rgb24) => {
                self.rgb_bgr_swap(input, output, 3);
            }
            (PixelFormat::Rgba, PixelFormat::Bgra) => {
                self.rgb_bgr_swap(input, output, 4);
            }
            (PixelFormat::Bgra, PixelFormat::Rgba) => {
                self.rgb_bgr_swap(input, output, 4);
            }

            // Add/remove alpha channel
            (PixelFormat::Rgb24, PixelFormat::Rgba) => {
                self.add_alpha(input, output, false);
            }
            (PixelFormat::Bgr24, PixelFormat::Bgra) => {
                self.add_alpha(input, output, false);
            }
            (PixelFormat::Rgba, PixelFormat::Rgb24) => {
                self.remove_alpha(input, output, false);
            }
            (PixelFormat::Bgra, PixelFormat::Bgr24) => {
                self.remove_alpha(input, output, false);
            }

            // Gray conversions
            (PixelFormat::Gray8, PixelFormat::Rgb24) => {
                self.gray_to_rgb24(input, output);
            }
            (PixelFormat::Gray8, PixelFormat::Rgba) => {
                self.gray_to_rgba(input, output);
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
    // YUV to RGB conversions
    // -------------------------------------------------------------------------

    fn i420_to_rgb24(&self, input: &[u8], output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let y_plane = &input[0..w * h];
        let u_plane = &input[w * h..w * h + (w / 2) * (h / 2)];
        let v_plane = &input[w * h + (w / 2) * (h / 2)..];

        for row in 0..h {
            for col in 0..w {
                let y = y_plane[row * w + col];
                let u = u_plane[(row / 2) * (w / 2) + (col / 2)];
                let v = v_plane[(row / 2) * (w / 2) + (col / 2)];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 3;
                output[dst_idx] = r;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = b;
            }
        }
    }

    fn i420_to_rgba(&self, input: &[u8], output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let y_plane = &input[0..w * h];
        let u_plane = &input[w * h..w * h + (w / 2) * (h / 2)];
        let v_plane = &input[w * h + (w / 2) * (h / 2)..];

        for row in 0..h {
            for col in 0..w {
                let y = y_plane[row * w + col];
                let u = u_plane[(row / 2) * (w / 2) + (col / 2)];
                let v = v_plane[(row / 2) * (w / 2) + (col / 2)];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 4;
                output[dst_idx] = r;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = b;
                output[dst_idx + 3] = 255;
            }
        }
    }

    fn i420_to_bgr24(&self, input: &[u8], output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let y_plane = &input[0..w * h];
        let u_plane = &input[w * h..w * h + (w / 2) * (h / 2)];
        let v_plane = &input[w * h + (w / 2) * (h / 2)..];

        for row in 0..h {
            for col in 0..w {
                let y = y_plane[row * w + col];
                let u = u_plane[(row / 2) * (w / 2) + (col / 2)];
                let v = v_plane[(row / 2) * (w / 2) + (col / 2)];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 3;
                output[dst_idx] = b;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = r;
            }
        }
    }

    fn i420_to_bgra(&self, input: &[u8], output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let y_plane = &input[0..w * h];
        let u_plane = &input[w * h..w * h + (w / 2) * (h / 2)];
        let v_plane = &input[w * h + (w / 2) * (h / 2)..];

        for row in 0..h {
            for col in 0..w {
                let y = y_plane[row * w + col];
                let u = u_plane[(row / 2) * (w / 2) + (col / 2)];
                let v = v_plane[(row / 2) * (w / 2) + (col / 2)];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 4;
                output[dst_idx] = b;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = r;
                output[dst_idx + 3] = 255;
            }
        }
    }

    fn nv12_to_rgb24(&self, input: &[u8], output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let y_plane = &input[0..w * h];
        let uv_plane = &input[w * h..];

        for row in 0..h {
            for col in 0..w {
                let y = y_plane[row * w + col];
                let uv_idx = (row / 2) * w + (col / 2) * 2;
                let u = uv_plane[uv_idx];
                let v = uv_plane[uv_idx + 1];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 3;
                output[dst_idx] = r;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = b;
            }
        }
    }

    fn nv12_to_rgba(&self, input: &[u8], output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;

        let y_plane = &input[0..w * h];
        let uv_plane = &input[w * h..];

        for row in 0..h {
            for col in 0..w {
                let y = y_plane[row * w + col];
                let uv_idx = (row / 2) * w + (col / 2) * 2;
                let u = uv_plane[uv_idx];
                let v = uv_plane[uv_idx + 1];

                let (r, g, b) = self.yuv_to_rgb(y, u, v);

                let dst_idx = (row * w + col) * 4;
                output[dst_idx] = r;
                output[dst_idx + 1] = g;
                output[dst_idx + 2] = b;
                output[dst_idx + 3] = 255;
            }
        }
    }

    // -------------------------------------------------------------------------
    // RGB to YUV conversions
    // -------------------------------------------------------------------------

    fn rgb24_to_i420(&self, input: &[u8], output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);

        // First pass: compute Y for all pixels
        for row in 0..h {
            for col in 0..w {
                let src_idx = (row * w + col) * 3;
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
                        let src_idx = ((row + dy) * w + (col + dx)) * 3;
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

    fn rgba_to_i420(&self, input: &[u8], output: &mut [u8]) {
        let w = self.width as usize;
        let h = self.height as usize;
        let y_size = w * h;
        let uv_size = (w / 2) * (h / 2);

        // Compute Y for all pixels
        for row in 0..h {
            for col in 0..w {
                let src_idx = (row * w + col) * 4;
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
                        let src_idx = ((row + dy) * w + (col + dx)) * 4;
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

    // -------------------------------------------------------------------------
    // RGB format conversions
    // -------------------------------------------------------------------------

    fn rgb_bgr_swap(&self, input: &[u8], output: &mut [u8], bytes_per_pixel: usize) {
        let pixel_count = (self.width * self.height) as usize;

        for i in 0..pixel_count {
            let src = i * bytes_per_pixel;
            let dst = i * bytes_per_pixel;

            output[dst] = input[src + 2]; // R/B swap
            output[dst + 1] = input[src + 1]; // G stays
            output[dst + 2] = input[src]; // B/R swap

            if bytes_per_pixel == 4 {
                output[dst + 3] = input[src + 3]; // Alpha stays
            }
        }
    }

    fn add_alpha(&self, input: &[u8], output: &mut [u8], _is_bgr: bool) {
        let pixel_count = (self.width * self.height) as usize;

        for i in 0..pixel_count {
            let src = i * 3;
            let dst = i * 4;

            output[dst] = input[src];
            output[dst + 1] = input[src + 1];
            output[dst + 2] = input[src + 2];
            output[dst + 3] = 255; // Opaque alpha
        }
    }

    fn remove_alpha(&self, input: &[u8], output: &mut [u8], _is_bgr: bool) {
        let pixel_count = (self.width * self.height) as usize;

        for i in 0..pixel_count {
            let src = i * 4;
            let dst = i * 3;

            output[dst] = input[src];
            output[dst + 1] = input[src + 1];
            output[dst + 2] = input[src + 2];
            // Alpha discarded
        }
    }

    fn gray_to_rgb24(&self, input: &[u8], output: &mut [u8]) {
        let pixel_count = (self.width * self.height) as usize;

        for i in 0..pixel_count {
            let gray = input[i];
            let dst = i * 3;

            output[dst] = gray;
            output[dst + 1] = gray;
            output[dst + 2] = gray;
        }
    }

    fn gray_to_rgba(&self, input: &[u8], output: &mut [u8]) {
        let pixel_count = (self.width * self.height) as usize;

        for i in 0..pixel_count {
            let gray = input[i];
            let dst = i * 4;

            output[dst] = gray;
            output[dst + 1] = gray;
            output[dst + 2] = gray;
            output[dst + 3] = 255;
        }
    }

    // -------------------------------------------------------------------------
    // Color space math
    // -------------------------------------------------------------------------

    /// Convert YUV to RGB using the configured color matrix.
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

        conv_to_yuv.convert(&rgb_in, &mut yuv).unwrap();
        conv_to_rgb.convert(&yuv, &mut rgb_out).unwrap();

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

        conv.convert(&rgb, &mut bgr).unwrap();

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

        conv_add.convert(&rgb, &mut rgba).unwrap();

        // Check alpha was added
        assert_eq!(rgba[3], 255);
        assert_eq!(rgba[7], 255);
        assert_eq!(rgba[11], 255);
        assert_eq!(rgba[15], 255);

        conv_rem.convert(&rgba, &mut rgb_out).unwrap();

        // Check roundtrip
        assert_eq!(rgb, rgb_out.as_slice());
    }

    #[test]
    fn test_gray_to_rgb() {
        let conv = VideoConvert::new(PixelFormat::Gray8, PixelFormat::Rgb24, 2, 2).unwrap();

        let gray = [0, 85, 170, 255];
        let mut rgb = vec![0u8; 12];

        conv.convert(&gray, &mut rgb).unwrap();

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

        conv.convert(&input, &mut output).unwrap();
        assert_eq!(input.as_slice(), output.as_slice());
    }
}
