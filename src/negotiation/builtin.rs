//! Built-in converters for common format and memory type conversions.
//!
//! These converters handle the most common transformations needed in pipelines:
//! - Video scaling (resize frames)
//! - Video format conversion (pixel format changes)
//! - Audio resampling (sample rate changes)
//! - Audio format conversion (sample format/channel changes)
//! - Memory copies (CPU <-> GPU transfers)

use super::converters::{ConverterElement, ConverterFactory, ConverterRegistry, FormatType};
use crate::buffer::Buffer;
use crate::element::Element;
use crate::error::Result;
use crate::format::{
    AudioFormat, AudioFormatCaps, FormatCaps, Framerate, PixelFormat, SampleFormat, VideoFormat,
    VideoFormatCaps,
};
use crate::memory::MemoryType;
use std::sync::Arc;

// ============================================================================
// VideoScale - resize video frames
// ============================================================================

/// Video scaler for resizing frames.
///
/// Supports bilinear and nearest-neighbor scaling.
pub struct VideoScale {
    /// Target width (None = passthrough).
    target_width: Option<u32>,
    /// Target height (None = passthrough).
    target_height: Option<u32>,
    /// Scaling algorithm.
    algorithm: ScaleAlgorithm,
}

/// Scaling algorithm.
#[derive(Clone, Copy, Debug, Default)]
pub enum ScaleAlgorithm {
    /// Nearest neighbor (fast, blocky).
    Nearest,
    /// Bilinear interpolation (default).
    #[default]
    Bilinear,
}

impl VideoScale {
    /// Create a new video scaler.
    pub fn new(target_width: Option<u32>, target_height: Option<u32>) -> Self {
        Self {
            target_width,
            target_height,
            algorithm: ScaleAlgorithm::default(),
        }
    }

    /// Set the scaling algorithm.
    pub fn with_algorithm(mut self, algorithm: ScaleAlgorithm) -> Self {
        self.algorithm = algorithm;
        self
    }
}

impl Element for VideoScale {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // TODO: Implement actual scaling
        // For now, this is a passthrough that would be filled in with actual scaling logic
        Ok(Some(buffer))
    }
}

impl ConverterElement for VideoScale {
    fn name(&self) -> &str {
        "videoscale"
    }

    fn input_format(&self) -> FormatCaps {
        FormatCaps::VideoRaw(VideoFormatCaps::any())
    }

    fn output_format(&self) -> FormatCaps {
        let mut caps = VideoFormatCaps::any();
        if let Some(w) = self.target_width {
            caps.width = w.into();
        }
        if let Some(h) = self.target_height {
            caps.height = h.into();
        }
        FormatCaps::VideoRaw(caps)
    }

    fn input_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn output_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn cost(&self) -> u32 {
        10 // Moderate cost - CPU bound
    }
}

// ============================================================================
// VideoConvert - convert between pixel formats
// ============================================================================

/// Video format converter for pixel format changes.
///
/// Converts between different pixel formats (e.g., I420 to RGBA).
pub struct VideoConvert {
    /// Target pixel format.
    target_format: PixelFormat,
}

impl VideoConvert {
    /// Create a new video converter.
    pub fn new(target_format: PixelFormat) -> Self {
        Self { target_format }
    }
}

impl Element for VideoConvert {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // TODO: Implement actual pixel format conversion
        // For now, this is a passthrough that would be filled in with actual conversion logic
        Ok(Some(buffer))
    }
}

impl ConverterElement for VideoConvert {
    fn name(&self) -> &str {
        "videoconvert"
    }

    fn input_format(&self) -> FormatCaps {
        FormatCaps::VideoRaw(VideoFormatCaps::any())
    }

    fn output_format(&self) -> FormatCaps {
        let mut caps = VideoFormatCaps::any();
        caps.pixel_format = self.target_format.into();
        FormatCaps::VideoRaw(caps)
    }

    fn input_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn output_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn cost(&self) -> u32 {
        5 // Low-moderate cost - simple color conversion
    }
}

// ============================================================================
// AudioResample - resample audio
// ============================================================================

/// Audio resampler for sample rate conversion.
pub struct AudioResample {
    /// Target sample rate.
    target_rate: u32,
}

impl AudioResample {
    /// Create a new audio resampler.
    pub fn new(target_rate: u32) -> Self {
        Self { target_rate }
    }
}

impl Element for AudioResample {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // TODO: Implement actual resampling
        Ok(Some(buffer))
    }
}

impl ConverterElement for AudioResample {
    fn name(&self) -> &str {
        "audioresample"
    }

    fn input_format(&self) -> FormatCaps {
        FormatCaps::AudioRaw(AudioFormatCaps::any())
    }

    fn output_format(&self) -> FormatCaps {
        let mut caps = AudioFormatCaps::any();
        caps.sample_rate = self.target_rate.into();
        FormatCaps::AudioRaw(caps)
    }

    fn input_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn output_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn cost(&self) -> u32 {
        8 // Moderate cost - DSP operations
    }
}

// ============================================================================
// AudioConvert - convert sample formats
// ============================================================================

/// Audio format converter for sample format and channel changes.
pub struct AudioConvert {
    /// Target sample format.
    target_format: Option<SampleFormat>,
    /// Target channel count.
    target_channels: Option<u16>,
}

impl AudioConvert {
    /// Create a new audio converter.
    pub fn new() -> Self {
        Self {
            target_format: None,
            target_channels: None,
        }
    }

    /// Set target sample format.
    pub fn with_format(mut self, format: SampleFormat) -> Self {
        self.target_format = Some(format);
        self
    }

    /// Set target channel count.
    pub fn with_channels(mut self, channels: u16) -> Self {
        self.target_channels = Some(channels);
        self
    }
}

impl Default for AudioConvert {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for AudioConvert {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // TODO: Implement actual audio conversion
        Ok(Some(buffer))
    }
}

impl ConverterElement for AudioConvert {
    fn name(&self) -> &str {
        "audioconvert"
    }

    fn input_format(&self) -> FormatCaps {
        FormatCaps::AudioRaw(AudioFormatCaps::any())
    }

    fn output_format(&self) -> FormatCaps {
        let mut caps = AudioFormatCaps::any();
        if let Some(fmt) = self.target_format {
            caps.sample_format = fmt.into();
        }
        if let Some(ch) = self.target_channels {
            caps.channels = ch.into();
        }
        FormatCaps::AudioRaw(caps)
    }

    fn input_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn output_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn cost(&self) -> u32 {
        3 // Low cost - simple format changes
    }
}

// ============================================================================
// MemoryCopy - copy between memory types
// ============================================================================

/// Memory copier for transferring buffers between memory types.
///
/// Handles CPU <-> GPU transfers and similar operations.
pub struct MemoryCopy {
    /// Source memory type.
    source_type: MemoryType,
    /// Target memory type.
    target_type: MemoryType,
}

impl MemoryCopy {
    /// Create a new memory copier.
    pub fn new(source_type: MemoryType, target_type: MemoryType) -> Self {
        Self {
            source_type,
            target_type,
        }
    }

    /// Create a CPU to GPU uploader.
    pub fn cpu_to_gpu() -> Self {
        Self::new(MemoryType::Cpu, MemoryType::GpuDevice)
    }

    /// Create a GPU to CPU downloader.
    pub fn gpu_to_cpu() -> Self {
        Self::new(MemoryType::GpuDevice, MemoryType::Cpu)
    }
}

impl Element for MemoryCopy {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        // TODO: Implement actual memory transfer
        // Would use GPU APIs or DMA for efficient transfers
        Ok(Some(buffer))
    }
}

impl ConverterElement for MemoryCopy {
    fn name(&self) -> &str {
        "memorycopy"
    }

    fn input_format(&self) -> FormatCaps {
        FormatCaps::Any
    }

    fn output_format(&self) -> FormatCaps {
        FormatCaps::Any
    }

    fn input_memory(&self) -> MemoryType {
        self.source_type
    }

    fn output_memory(&self) -> MemoryType {
        self.target_type
    }

    fn cost(&self) -> u32 {
        // Memory transfers are expensive
        match (self.source_type, self.target_type) {
            (MemoryType::Cpu, MemoryType::Cpu) => 1,
            (MemoryType::GpuDevice, MemoryType::GpuDevice) => 2,
            _ => 20, // Cross-device transfers are costly
        }
    }
}

// ============================================================================
// Identity converter (passthrough)
// ============================================================================

/// Identity converter that passes data through unchanged.
///
/// Used when formats are compatible but the pipeline needs an explicit node.
pub struct Identity;

impl Element for Identity {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        Ok(Some(buffer))
    }
}

impl ConverterElement for Identity {
    fn name(&self) -> &str {
        "identity"
    }

    fn input_format(&self) -> FormatCaps {
        FormatCaps::Any
    }

    fn output_format(&self) -> FormatCaps {
        FormatCaps::Any
    }

    fn input_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn output_memory(&self) -> MemoryType {
        MemoryType::Cpu
    }

    fn cost(&self) -> u32 {
        0 // Zero cost - passthrough
    }
}

// ============================================================================
// Registry builder
// ============================================================================

/// Create a converter registry with built-in converters.
///
/// This registers commonly needed converters:
/// - Video scaling and format conversion
/// - Audio resampling and format conversion
/// - Memory type transfers
pub fn builtin_registry() -> ConverterRegistry {
    let mut registry = ConverterRegistry::new();

    // Video format conversions (same memory type)
    registry.register(
        FormatType::VideoRaw,
        FormatType::VideoRaw,
        MemoryType::Cpu,
        MemoryType::Cpu,
        5,
        Arc::new(|| Box::new(VideoConvert::new(PixelFormat::I420))),
    );

    // Audio format conversions
    registry.register(
        FormatType::AudioRaw,
        FormatType::AudioRaw,
        MemoryType::Cpu,
        MemoryType::Cpu,
        3,
        Arc::new(|| Box::new(AudioConvert::new())),
    );

    // CPU to GPU upload
    registry.register(
        FormatType::Any,
        FormatType::Any,
        MemoryType::Cpu,
        MemoryType::GpuDevice,
        20,
        Arc::new(|| Box::new(MemoryCopy::cpu_to_gpu())),
    );

    // GPU to CPU download
    registry.register(
        FormatType::Any,
        FormatType::Any,
        MemoryType::GpuDevice,
        MemoryType::Cpu,
        20,
        Arc::new(|| Box::new(MemoryCopy::gpu_to_cpu())),
    );

    // Identity (same format, same memory)
    registry.register(
        FormatType::Any,
        FormatType::Any,
        MemoryType::Cpu,
        MemoryType::Cpu,
        0,
        Arc::new(|| Box::new(Identity)),
    );

    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_video_scale_creation() {
        let scaler = VideoScale::new(Some(1280), Some(720));
        assert_eq!(ConverterElement::name(&scaler), "videoscale");
        assert_eq!(scaler.cost(), 10);
    }

    #[test]
    fn test_video_convert_creation() {
        let converter = VideoConvert::new(PixelFormat::Rgba);
        assert_eq!(ConverterElement::name(&converter), "videoconvert");
        assert_eq!(converter.cost(), 5);
    }

    #[test]
    fn test_audio_resample_creation() {
        let resampler = AudioResample::new(48000);
        assert_eq!(ConverterElement::name(&resampler), "audioresample");
        assert_eq!(resampler.cost(), 8);
    }

    #[test]
    fn test_audio_convert_creation() {
        let converter = AudioConvert::new()
            .with_format(SampleFormat::F32)
            .with_channels(2);
        assert_eq!(ConverterElement::name(&converter), "audioconvert");
        assert_eq!(converter.cost(), 3);
    }

    #[test]
    fn test_memory_copy_creation() {
        let uploader = MemoryCopy::cpu_to_gpu();
        assert_eq!(ConverterElement::name(&uploader), "memorycopy");
        assert_eq!(uploader.input_memory(), MemoryType::Cpu);
        assert_eq!(uploader.output_memory(), MemoryType::GpuDevice);
        assert_eq!(uploader.cost(), 20);
    }

    #[test]
    fn test_identity_creation() {
        let identity = Identity;
        assert_eq!(ConverterElement::name(&identity), "identity");
        assert_eq!(identity.cost(), 0);
    }

    #[test]
    fn test_builtin_registry() {
        let registry = builtin_registry();
        assert!(!registry.is_empty());

        // Should find video-to-video converter
        let video_path = registry.find_path(
            FormatType::VideoRaw,
            FormatType::VideoRaw,
            MemoryType::Cpu,
            MemoryType::Cpu,
        );
        assert!(video_path.is_some());

        // Should find audio-to-audio converter
        let audio_path = registry.find_path(
            FormatType::AudioRaw,
            FormatType::AudioRaw,
            MemoryType::Cpu,
            MemoryType::Cpu,
        );
        assert!(audio_path.is_some());

        // Should find CPU-to-GPU transfer
        let upload_path = registry.find_path(
            FormatType::Any,
            FormatType::Any,
            MemoryType::Cpu,
            MemoryType::GpuDevice,
        );
        assert!(upload_path.is_some());
    }

    #[test]
    fn test_video_scale_output_caps() {
        let scaler = VideoScale::new(Some(1920), Some(1080));
        let output = scaler.output_format();

        if let FormatCaps::VideoRaw(caps) = output {
            assert_eq!(caps.width.fixate(), Some(1920));
            assert_eq!(caps.height.fixate(), Some(1080));
        } else {
            panic!("Expected VideoRaw caps");
        }
    }

    #[test]
    fn test_converter_element_process_passthrough() {
        use crate::buffer::MemoryHandle;
        use crate::memory::HeapSegment;
        use crate::metadata::Metadata;
        use std::sync::Arc;

        let mut identity = Identity;
        let segment = Arc::new(HeapSegment::new(4).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));
        let result = Element::process(&mut identity, buffer).unwrap();
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 4);
    }
}
