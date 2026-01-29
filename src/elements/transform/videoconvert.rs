//! Video format conversion element.
//!
//! Converts between pixel formats (e.g., YUYV -> RGBA for display).

use crate::buffer::{Buffer, MemoryHandle};
use crate::converters::{PixelFormat, VideoConvert};
use crate::element::Element;
use crate::error::{Error, Result};
use crate::format::Caps;
use crate::memory::SharedArena;

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
    /// Cached converter (created on first frame)
    converter: Option<VideoConvert>,
    /// Output buffer for conversion
    output_buffer: Vec<u8>,
    /// Element name
    name: String,
    /// Arena for output buffers
    arena: Option<SharedArena>,
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
            output_buffer: Vec::new(),
            name: "videoconvert".to_string(),
            arena: None,
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

    /// Initialize the converter if needed.
    fn ensure_converter(&mut self, input_size: usize) -> Result<()> {
        if self.converter.is_some() {
            return Ok(());
        }

        // Determine input format and dimensions
        let (input_format, width, height) = if let Some(fmt) = self.input_format {
            if self.width > 0 && self.height > 0 {
                (fmt, self.width, self.height)
            } else {
                // Try to derive dimensions from buffer size
                let expected = fmt.buffer_size(self.width.max(640), self.height.max(480));
                if input_size == expected {
                    (fmt, self.width.max(640), self.height.max(480))
                } else {
                    return Err(Error::Element(format!(
                        "Cannot determine dimensions for format {:?} with buffer size {}",
                        fmt, input_size
                    )));
                }
            }
        } else {
            // Auto-detect
            self.detect_format(input_size).ok_or_else(|| {
                Error::Element(format!(
                    "Cannot auto-detect video format for buffer size {}",
                    input_size
                ))
            })?
        };

        // Create converter
        let converter = VideoConvert::new(input_format, self.output_format, width, height)?;

        // Allocate output buffer
        let output_size = self.output_format.buffer_size(width, height);
        self.output_buffer.resize(output_size, 0);

        self.width = width;
        self.height = height;
        self.input_format = Some(input_format);
        self.converter = Some(converter);

        tracing::info!(
            "VideoConvert: {:?} {}x{} -> {:?}",
            input_format,
            width,
            height,
            self.output_format
        );

        Ok(())
    }
}

impl Default for VideoConvertElement {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for VideoConvertElement {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input_data = buffer.as_bytes();

        tracing::debug!(
            "VideoConvert: received buffer with {} bytes",
            input_data.len()
        );

        // Initialize converter on first frame
        self.ensure_converter(input_data.len())?;

        let converter = self.converter.as_ref().unwrap();

        // Convert
        converter.convert(input_data, &mut self.output_buffer)?;

        // Create output buffer
        let output_size = self.output_buffer.len();

        if self.arena.is_none() || self.arena.as_ref().unwrap().slot_size() < output_size {
            self.arena = Some(SharedArena::new(output_size, 32)?);
        }

        let arena = self.arena.as_mut().unwrap();
        arena.reclaim();
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("arena exhausted".into()))?;

        // Copy converted data
        slot.data_mut()[..output_size].copy_from_slice(&self.output_buffer);

        let handle = MemoryHandle::new(slot);
        let output = Buffer::new(handle, buffer.metadata().clone());

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
        Caps::video_raw_any_resolution(convert_pixel_format(self.output_format))
    }

    fn input_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Accept any raw video format - truly any dimensions and pixel format
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, VideoFormatCaps,
        };

        let format = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Any,
            framerate: CapsValue::Any,
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(format),
            MemoryCaps::cpu_only(),
        )])
    }

    fn output_media_caps(&self) -> crate::format::ElementMediaCaps {
        // Output is the configured output format with any dimensions
        use crate::format::{
            CapsValue, ElementMediaCaps, FormatCaps, FormatMemoryCap, MemoryCaps, VideoFormatCaps,
        };

        let format = VideoFormatCaps {
            width: CapsValue::Any,
            height: CapsValue::Any,
            pixel_format: CapsValue::Fixed(convert_pixel_format(self.output_format)),
            framerate: CapsValue::Any,
        };

        ElementMediaCaps::new(vec![FormatMemoryCap::new(
            FormatCaps::VideoRaw(format),
            MemoryCaps::cpu_only(),
        )])
    }
}

/// Convert from converters::PixelFormat to format::PixelFormat
fn convert_pixel_format(pf: PixelFormat) -> crate::format::PixelFormat {
    match pf {
        PixelFormat::Yuyv => crate::format::PixelFormat::Yuyv,
        PixelFormat::Uyvy => crate::format::PixelFormat::Uyvy,
        PixelFormat::Nv12 => crate::format::PixelFormat::Nv12,
        PixelFormat::I420 => crate::format::PixelFormat::I420,
        PixelFormat::Rgb24 => crate::format::PixelFormat::Rgb24,
        PixelFormat::Bgr24 => crate::format::PixelFormat::Bgr24,
        PixelFormat::Rgba => crate::format::PixelFormat::Rgba,
        PixelFormat::Bgra => crate::format::PixelFormat::Bgra,
        PixelFormat::Gray8 => crate::format::PixelFormat::Gray8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
