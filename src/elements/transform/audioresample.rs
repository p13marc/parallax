//! Audio resampling element.
//!
//! Converts between different audio sample rates.

use crate::buffer::{Buffer, MemoryHandle};
use crate::converters::{AudioResample, ResampleQuality, SampleFormat};
use crate::element::Element;
use crate::error::{Error, Result};
use crate::format::Caps;
use crate::memory::SharedArena;

/// Audio resampling element.
///
/// This element converts audio between different sample rates. It's commonly
/// used to resample 48kHz audio to 44.1kHz for CD quality output.
///
/// # Example
///
/// ```rust,ignore
/// // Resample 48kHz stereo to 44.1kHz
/// let element = AudioResampleElement::new()
///     .with_input_rate(48000)
///     .with_output_rate(44100)
///     .with_channels(2)
///     .with_format(SampleFormat::F32Le);
/// ```
pub struct AudioResampleElement {
    /// Input sample rate
    input_rate: u32,
    /// Output sample rate
    output_rate: u32,
    /// Number of channels
    channels: u32,
    /// Sample format
    format: SampleFormat,
    /// Resampling quality
    quality: ResampleQuality,
    /// Cached resampler (created on first buffer)
    resampler: Option<AudioResample>,
    /// Output buffer for resampling
    output_buffer: Vec<u8>,
    /// Element name
    name: String,
    /// Arena for output buffers
    arena: Option<SharedArena>,
}

impl AudioResampleElement {
    /// Create a new audio resample element.
    ///
    /// Defaults to 48kHz -> 44.1kHz, stereo, F32LE.
    pub fn new() -> Self {
        Self {
            input_rate: 48000,
            output_rate: 44100,
            channels: 2,
            format: SampleFormat::F32Le,
            quality: ResampleQuality::Medium,
            resampler: None,
            output_buffer: Vec::new(),
            name: "audioresample".to_string(),
            arena: None,
        }
    }

    /// Set the input sample rate.
    pub fn with_input_rate(mut self, rate: u32) -> Self {
        self.input_rate = rate;
        self
    }

    /// Set the output sample rate.
    pub fn with_output_rate(mut self, rate: u32) -> Self {
        self.output_rate = rate;
        self
    }

    /// Set the number of channels.
    pub fn with_channels(mut self, channels: u32) -> Self {
        self.channels = channels;
        self
    }

    /// Set the sample format.
    pub fn with_format(mut self, format: SampleFormat) -> Self {
        self.format = format;
        self
    }

    /// Set the resampling quality.
    pub fn with_quality(mut self, quality: ResampleQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Initialize the resampler if needed.
    fn ensure_resampler(&mut self) -> Result<()> {
        if self.resampler.is_some() {
            return Ok(());
        }

        let resampler = AudioResample::new(
            self.input_rate,
            self.output_rate,
            self.channels,
            self.format,
        )?
        .with_quality(self.quality);

        tracing::info!(
            "AudioResample: {}Hz -> {}Hz ({} channels, {:?})",
            self.input_rate,
            self.output_rate,
            self.channels,
            self.format
        );

        self.resampler = Some(resampler);
        Ok(())
    }
}

impl Default for AudioResampleElement {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for AudioResampleElement {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input_data = buffer.as_bytes();

        tracing::debug!(
            "AudioResample: received buffer with {} bytes",
            input_data.len()
        );

        // Initialize resampler on first buffer
        self.ensure_resampler()?;

        let resampler = self.resampler.as_mut().unwrap();

        // Calculate output size and resize buffer with some extra headroom
        // The output_size() is an approximation; we add padding for rounding
        let output_size = resampler.output_size(input_data.len());
        let padded_size = output_size + 64; // Extra headroom for rounding
        self.output_buffer.resize(padded_size, 0);

        // Resample
        let written = resampler.resample(input_data, &mut self.output_buffer)?;

        if written == 0 {
            return Ok(None);
        }

        // Create output buffer
        if self.arena.is_none() || self.arena.as_ref().unwrap().slot_size() < written {
            self.arena = Some(SharedArena::new(written.max(4096), 32)?);
        }

        let arena = self.arena.as_mut().unwrap();
        arena.reclaim();
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("arena exhausted".into()))?;

        // Copy resampled data
        slot.data_mut()[..written].copy_from_slice(&self.output_buffer[..written]);

        let handle = MemoryHandle::with_len(slot, written);
        let output = Buffer::new(handle, buffer.metadata().clone());

        Ok(Some(output))
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        // Reset resampler state for next stream
        if let Some(ref mut resampler) = self.resampler {
            resampler.reset();
        }
        Ok(None)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn input_caps(&self) -> Caps {
        Caps::audio_raw(
            self.input_rate,
            self.channels as u16,
            convert_sample_format(self.format),
        )
    }

    fn output_caps(&self) -> Caps {
        Caps::audio_raw(
            self.output_rate,
            self.channels as u16,
            convert_sample_format(self.format),
        )
    }
}

/// Convert from converters::SampleFormat to format::SampleFormat.
///
/// Note: The converters module has more detailed endianness variants,
/// while format::SampleFormat is simplified. We map to the closest match.
fn convert_sample_format(sf: SampleFormat) -> crate::format::SampleFormat {
    match sf {
        SampleFormat::U8 => crate::format::SampleFormat::U8,
        SampleFormat::S16Le | SampleFormat::S16Be => crate::format::SampleFormat::S16,
        SampleFormat::S32Le | SampleFormat::S32Be => crate::format::SampleFormat::S32,
        SampleFormat::F32Le | SampleFormat::F32Be => crate::format::SampleFormat::F32,
        // F64 maps to F32 (closest available)
        SampleFormat::F64Le | SampleFormat::F64Be => crate::format::SampleFormat::F32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Metadata;

    #[test]
    fn test_same_rate_passthrough() {
        let mut element = AudioResampleElement::new()
            .with_input_rate(48000)
            .with_output_rate(48000)
            .with_channels(1)
            .with_format(SampleFormat::F32Le);

        // Create test buffer
        let samples: Vec<f32> = vec![0.0, 0.5, 1.0, -0.5, -1.0];
        let input_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let arena = SharedArena::new(input_bytes.len(), 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..input_bytes.len()].copy_from_slice(&input_bytes);
        let handle = MemoryHandle::with_len(slot, input_bytes.len());
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));

        let result = element.process(buffer).unwrap().unwrap();

        // Same rate should pass through unchanged
        assert_eq!(result.len(), input_bytes.len());
    }

    #[test]
    fn test_downsample_48k_to_44k() {
        let mut element = AudioResampleElement::new()
            .with_input_rate(48000)
            .with_output_rate(44100)
            .with_channels(1)
            .with_format(SampleFormat::F32Le);

        // Create test buffer: 480 samples at 48kHz = 10ms
        let input_samples: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin()).collect();
        let input_bytes: Vec<u8> = input_samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        // Use a larger arena to accommodate output buffer size calculation variance
        let arena = SharedArena::new(input_bytes.len() * 2, 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..input_bytes.len()].copy_from_slice(&input_bytes);
        let handle = MemoryHandle::with_len(slot, input_bytes.len());
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));

        let result = element.process(buffer).unwrap().unwrap();

        // Expected output: ~441 samples at 44.1kHz for 10ms
        // Allow some variance due to resampling algorithm
        let output_samples = result.len() / 4;
        assert!(
            output_samples >= 438 && output_samples <= 448,
            "Expected ~441 samples, got {}",
            output_samples
        );
    }
}
