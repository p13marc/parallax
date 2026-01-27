//! Format converters for video and audio data.
//!
//! This module provides pure Rust implementations of common media format
//! conversions. These converters are used by the caps negotiation system
//! to automatically convert between incompatible formats.
//!
//! # Video Converters
//!
//! - [`VideoConvert`]: Pixel format conversion (YUV ↔ RGB)
//! - [`VideoScale`]: Resolution scaling (bilinear, nearest neighbor)
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
//! converter.convert(&yuv_input, &mut rgb_output)?;
//! ```

mod audio;
mod colorspace;
mod resample;
mod scale;

pub use audio::{AudioChannelMix, AudioConvert, ChannelLayout, SampleFormat};
pub use colorspace::{ColorMatrix, PixelFormat, VideoConvert};
pub use resample::{AudioResample, ResampleQuality};
pub use scale::{ScaleAlgorithm, VideoScale};
