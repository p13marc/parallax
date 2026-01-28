//! Audio format conversion element.
//!
//! Converts between audio sample formats (e.g., S16 -> F32).

use crate::buffer::{Buffer, MemoryHandle};
use crate::converters::{AudioConvert, SampleFormat};
use crate::element::Element;
use crate::error::{Error, Result};
use crate::format::Caps;
use crate::memory::SharedArena;

/// Audio format conversion element.
///
/// This element converts audio samples between different formats. It's commonly
/// used to convert integer samples to float for processing, or vice versa.
///
/// # Example
///
/// ```rust,ignore
/// // Convert S16 stereo to F32
/// let element = AudioConvertElement::new()
///     .with_input_format(SampleFormat::S16Le)
///     .with_output_format(SampleFormat::F32Le)
///     .with_channels(2);
/// ```
pub struct AudioConvertElement {
    /// Input sample format (required)
    input_format: SampleFormat,
    /// Output sample format
    output_format: SampleFormat,
    /// Number of channels
    channels: u32,
    /// Cached converter (created on first buffer)
    converter: Option<AudioConvert>,
    /// Output buffer for conversion
    output_buffer: Vec<u8>,
    /// Element name
    name: String,
    /// Arena for output buffers
    arena: Option<SharedArena>,
}

impl AudioConvertElement {
    /// Create a new audio convert element.
    ///
    /// Defaults to S16LE input, F32LE output, stereo.
    pub fn new() -> Self {
        Self {
            input_format: SampleFormat::S16Le,
            output_format: SampleFormat::F32Le,
            channels: 2,
            converter: None,
            output_buffer: Vec::new(),
            name: "audioconvert".to_string(),
            arena: None,
        }
    }

    /// Set the input sample format.
    pub fn with_input_format(mut self, format: SampleFormat) -> Self {
        self.input_format = format;
        self
    }

    /// Set the output sample format.
    pub fn with_output_format(mut self, format: SampleFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Set the number of channels.
    pub fn with_channels(mut self, channels: u32) -> Self {
        self.channels = channels;
        self
    }

    /// Initialize the converter if needed.
    fn ensure_converter(&mut self) -> Result<()> {
        if self.converter.is_some() {
            return Ok(());
        }

        let converter = AudioConvert::new(self.input_format, self.output_format, self.channels)?;

        tracing::info!(
            "AudioConvert: {:?} -> {:?} ({} channels)",
            self.input_format,
            self.output_format,
            self.channels
        );

        self.converter = Some(converter);
        Ok(())
    }
}

impl Default for AudioConvertElement {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for AudioConvertElement {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let input_data = buffer.as_bytes();

        tracing::debug!(
            "AudioConvert: received buffer with {} bytes",
            input_data.len()
        );

        // Initialize converter on first buffer
        self.ensure_converter()?;

        let converter = self.converter.as_ref().unwrap();

        // Calculate output size and resize buffer
        let output_size = converter.output_size(input_data.len());
        self.output_buffer.resize(output_size, 0);

        // Convert
        let written = converter.convert(input_data, &mut self.output_buffer)?;

        // Create output buffer
        if self.arena.is_none() || self.arena.as_ref().unwrap().slot_size() < written {
            self.arena = Some(SharedArena::new(written.max(4096), 32)?);
        }

        let arena = self.arena.as_mut().unwrap();
        arena.reclaim();
        let mut slot = arena
            .acquire()
            .ok_or_else(|| Error::Element("arena exhausted".into()))?;

        // Copy converted data
        slot.data_mut()[..written].copy_from_slice(&self.output_buffer[..written]);

        let handle = MemoryHandle::with_len(slot, written);
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
        Caps::audio_raw_any()
    }

    fn output_caps(&self) -> Caps {
        Caps::audio_raw_any()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metadata::Metadata;

    #[test]
    fn test_s16_to_f32_conversion() {
        let mut element = AudioConvertElement::new()
            .with_input_format(SampleFormat::S16Le)
            .with_output_format(SampleFormat::F32Le)
            .with_channels(1);

        // Create test buffer with S16 samples: silence (0), max positive, max negative
        let samples: Vec<i16> = vec![0, 32767, -32768];
        let input_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        let arena = SharedArena::new(input_bytes.len(), 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..input_bytes.len()].copy_from_slice(&input_bytes);
        let handle = MemoryHandle::with_len(slot, input_bytes.len());
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));

        let result = element.process(buffer).unwrap().unwrap();

        // Output should be 3 f32 samples = 12 bytes
        assert_eq!(result.len(), 12);

        // Parse output as f32
        let output_bytes = result.as_bytes();
        let f0 = f32::from_le_bytes([
            output_bytes[0],
            output_bytes[1],
            output_bytes[2],
            output_bytes[3],
        ]);
        let f1 = f32::from_le_bytes([
            output_bytes[4],
            output_bytes[5],
            output_bytes[6],
            output_bytes[7],
        ]);
        let f2 = f32::from_le_bytes([
            output_bytes[8],
            output_bytes[9],
            output_bytes[10],
            output_bytes[11],
        ]);

        // Check conversions (with small tolerance for floating point)
        assert!(
            (f0 - 0.0).abs() < 0.001,
            "Silence should be ~0.0, got {}",
            f0
        );
        assert!(
            (f1 - 1.0).abs() < 0.001,
            "Max positive should be ~1.0, got {}",
            f1
        );
        assert!(
            (f2 - (-1.0)).abs() < 0.001,
            "Max negative should be ~-1.0, got {}",
            f2
        );
    }
}
