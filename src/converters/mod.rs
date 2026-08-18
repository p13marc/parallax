//! Format converters for video and audio data.
//!
//! This module provides pure Rust implementations of common media format
//! conversions. These converters are used by the caps negotiation system
//! to automatically convert between incompatible formats.
//!
//! # Video Converters
//!
//! - [`VideoConvert`]: Pixel format conversion (YUV ↔ RGB)
//! - [`ScaleEngine`]: Resolution scaling (bilinear, nearest neighbor)
//!
//! # Audio Converters
//!
//! - [`AudioConvert`]: Sample format conversion (S16 ↔ F32, etc.)
//! - [`AudioResample`]: Sample rate conversion
//! - [`AudioChannelMix`]: Channel layout conversion (mono ↔ stereo, etc.)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::converters::{VideoConvert, PixelFormat};
//!
//! let converter = VideoConvert::new(
//!     PixelFormat::I420,
//!     PixelFormat::Rgb24,
//!     1920, 1080
//! )?;
//!
//! let mut rgb_output = vec![0u8; 1920 * 1080 * 3];
//! // Ordinary (packed) input; a codec-owned strided frame passes its
//! // own `Metadata::plane_layout()` here instead (#194).
//! let layout = converter.packed_input_layout();
//! converter.convert(&yuv_input, layout, &mut rgb_output)?;
//! ```

mod audio;
mod colorspace;
mod resample;
mod scale;

pub use audio::{AudioChannelMix, AudioConvert, ChannelLayout, SampleFormat};
pub use colorspace::{ColorMatrix, PixelFormat, UnsupportedPixelFormat, VideoConvert};
pub use resample::{AudioResample, ResampleQuality};
pub use scale::{ScaleEngine, ScaleMode};

/// Test-only helpers shared by the engine test modules.
#[cfg(test)]
pub(crate) mod testutil {
    use crate::format::PlaneLayout;

    /// Build a strided twin of a packed frame: every plane's rows are moved
    /// `pad` bytes further apart, with the gaps filled by a sentinel that
    /// must never be read.
    ///
    /// Sized to [`PlaneLayout::full_span_len`], so the last plane's final
    /// row keeps its trailing padding — the shape real strided producers
    /// allocate (dav1d aligns plane heights to 128 rows, V4L2 hands out
    /// `bytesperline * height`).
    pub(crate) fn strided_twin(
        packed: &[u8],
        format: crate::format::PixelFormat,
        width: u32,
        height: u32,
        pad: usize,
    ) -> (Vec<u8>, PlaneLayout) {
        use crate::format::PlaneDesc;
        const SENTINEL: u8 = 0x5A;

        let src_layout = PlaneLayout::packed(format, width, height);
        let mut descs = Vec::new();
        let mut offset = 0usize;
        for plane in src_layout.resolved(format, width, height) {
            let stride = plane.stride + pad;
            descs.push(PlaneDesc { offset, stride });
            offset += stride * plane.rows;
        }
        let dst_layout = PlaneLayout::from_planes(&descs);

        let mut out = vec![SENTINEL; dst_layout.full_span_len(format, width, height)];
        for (src, dst) in src_layout
            .resolved(format, width, height)
            .zip(dst_layout.resolved(format, width, height))
        {
            for row in 0..src.rows {
                let s = src.offset + row * src.stride;
                let d = dst.offset + row * dst.stride;
                out[d..d + dst.row_bytes].copy_from_slice(&packed[s..s + src.row_bytes]);
            }
        }
        (out, dst_layout)
    }
}
