//! Audio sample format conversion and channel mixing.
//!
//! Provides pure Rust implementations of audio format conversions.

use crate::error::{Error, Result};

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    /// Unsigned 8-bit (0-255, center at 128)
    U8,
    /// Signed 16-bit little-endian
    S16Le,
    /// Signed 16-bit big-endian
    S16Be,
    /// Signed 32-bit little-endian
    S32Le,
    /// Signed 32-bit big-endian
    S32Be,
    /// 32-bit float (-1.0 to 1.0)
    F32Le,
    /// 32-bit float big-endian
    F32Be,
    /// 64-bit float (-1.0 to 1.0)
    F64Le,
    /// 64-bit float big-endian
    F64Be,
}

impl SampleFormat {
    /// Get the number of bytes per sample.
    pub fn bytes_per_sample(&self) -> usize {
        match self {
            SampleFormat::U8 => 1,
            SampleFormat::S16Le | SampleFormat::S16Be => 2,
            SampleFormat::S32Le | SampleFormat::S32Be => 4,
            SampleFormat::F32Le | SampleFormat::F32Be => 4,
            SampleFormat::F64Le | SampleFormat::F64Be => 8,
        }
    }

    /// Returns true if this format is little-endian.
    pub fn is_little_endian(&self) -> bool {
        matches!(
            self,
            SampleFormat::U8
                | SampleFormat::S16Le
                | SampleFormat::S32Le
                | SampleFormat::F32Le
                | SampleFormat::F64Le
        )
    }

    /// Returns true if this format uses floating point.
    pub fn is_float(&self) -> bool {
        matches!(
            self,
            SampleFormat::F32Le | SampleFormat::F32Be | SampleFormat::F64Le | SampleFormat::F64Be
        )
    }
}

/// Audio channel layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ChannelLayout {
    /// Single channel (mono)
    Mono,
    /// Two channels (left, right)
    #[default]
    Stereo,
    /// 2.1 (left, right, LFE)
    Surround21,
    /// 5.1 (FL, FR, FC, LFE, BL, BR)
    Surround51,
    /// 7.1 (FL, FR, FC, LFE, BL, BR, SL, SR)
    Surround71,
}

impl ChannelLayout {
    /// Get the number of channels.
    pub fn channels(&self) -> usize {
        match self {
            ChannelLayout::Mono => 1,
            ChannelLayout::Stereo => 2,
            ChannelLayout::Surround21 => 3,
            ChannelLayout::Surround51 => 6,
            ChannelLayout::Surround71 => 8,
        }
    }
}

/// Audio sample format converter.
///
/// Converts between different audio sample formats (S16, F32, etc.)
pub struct AudioConvert {
    input_format: SampleFormat,
    output_format: SampleFormat,
    #[allow(dead_code)] // May be used for validation in future
    channels: u32,
}

impl AudioConvert {
    /// Create a new audio format converter.
    pub fn new(
        input_format: SampleFormat,
        output_format: SampleFormat,
        channels: u32,
    ) -> Result<Self> {
        if channels == 0 {
            return Err(Error::Config("Channels must be non-zero".into()));
        }

        Ok(Self {
            input_format,
            output_format,
            channels,
        })
    }

    /// Get the input format.
    pub fn input_format(&self) -> SampleFormat {
        self.input_format
    }

    /// Get the output format.
    pub fn output_format(&self) -> SampleFormat {
        self.output_format
    }

    /// Calculate output buffer size for given input size.
    pub fn output_size(&self, input_size: usize) -> usize {
        let samples = input_size / self.input_format.bytes_per_sample();
        samples * self.output_format.bytes_per_sample()
    }

    /// Convert audio samples from input format to output format.
    pub fn convert(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let input_sample_size = self.input_format.bytes_per_sample();
        let output_sample_size = self.output_format.bytes_per_sample();

        if input.len() % input_sample_size != 0 {
            return Err(Error::Config(format!(
                "Input size {} not aligned to sample size {}",
                input.len(),
                input_sample_size
            )));
        }

        let sample_count = input.len() / input_sample_size;
        let required_output = sample_count * output_sample_size;

        if output.len() < required_output {
            return Err(Error::Config(format!(
                "Output buffer too small: {} < {}",
                output.len(),
                required_output
            )));
        }

        // Same format - just copy
        if self.input_format == self.output_format {
            output[..input.len()].copy_from_slice(input);
            return Ok(input.len());
        }

        // Convert via intermediate f64 representation
        for i in 0..sample_count {
            let normalized = self.read_normalized(input, i);
            self.write_normalized(output, i, normalized);
        }

        Ok(required_output)
    }

    /// Read a sample as normalized f64 (-1.0 to 1.0).
    fn read_normalized(&self, data: &[u8], sample_idx: usize) -> f64 {
        let offset = sample_idx * self.input_format.bytes_per_sample();

        match self.input_format {
            SampleFormat::U8 => {
                let v = data[offset];
                (v as f64 - 128.0) / 128.0
            }
            SampleFormat::S16Le => {
                let v = i16::from_le_bytes([data[offset], data[offset + 1]]);
                v as f64 / 32768.0
            }
            SampleFormat::S16Be => {
                let v = i16::from_be_bytes([data[offset], data[offset + 1]]);
                v as f64 / 32768.0
            }
            SampleFormat::S32Le => {
                let v = i32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                v as f64 / 2147483648.0
            }
            SampleFormat::S32Be => {
                let v = i32::from_be_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                v as f64 / 2147483648.0
            }
            SampleFormat::F32Le => {
                let v = f32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                v as f64
            }
            SampleFormat::F32Be => {
                let v = f32::from_be_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]);
                v as f64
            }
            SampleFormat::F64Le => f64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]),
            SampleFormat::F64Be => f64::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]),
        }
    }

    /// Write a normalized f64 (-1.0 to 1.0) as the output format.
    fn write_normalized(&self, data: &mut [u8], sample_idx: usize, value: f64) {
        let offset = sample_idx * self.output_format.bytes_per_sample();

        // Clamp to valid range
        let value = value.clamp(-1.0, 1.0);

        match self.output_format {
            SampleFormat::U8 => {
                let v = ((value * 128.0) + 128.0).round() as u8;
                data[offset] = v;
            }
            SampleFormat::S16Le => {
                let v = (value * 32767.0).round() as i16;
                let bytes = v.to_le_bytes();
                data[offset] = bytes[0];
                data[offset + 1] = bytes[1];
            }
            SampleFormat::S16Be => {
                let v = (value * 32767.0).round() as i16;
                let bytes = v.to_be_bytes();
                data[offset] = bytes[0];
                data[offset + 1] = bytes[1];
            }
            SampleFormat::S32Le => {
                let v = (value * 2147483647.0).round() as i32;
                let bytes = v.to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::S32Be => {
                let v = (value * 2147483647.0).round() as i32;
                let bytes = v.to_be_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::F32Le => {
                let v = value as f32;
                let bytes = v.to_le_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::F32Be => {
                let v = value as f32;
                let bytes = v.to_be_bytes();
                data[offset..offset + 4].copy_from_slice(&bytes);
            }
            SampleFormat::F64Le => {
                let bytes = value.to_le_bytes();
                data[offset..offset + 8].copy_from_slice(&bytes);
            }
            SampleFormat::F64Be => {
                let bytes = value.to_be_bytes();
                data[offset..offset + 8].copy_from_slice(&bytes);
            }
        }
    }
}

/// Audio channel mixer.
///
/// Converts between different channel layouts (mono ↔ stereo, etc.)
pub struct AudioChannelMix {
    input_layout: ChannelLayout,
    output_layout: ChannelLayout,
    sample_format: SampleFormat,
}

impl AudioChannelMix {
    /// Create a new channel mixer.
    pub fn new(
        input_layout: ChannelLayout,
        output_layout: ChannelLayout,
        sample_format: SampleFormat,
    ) -> Self {
        Self {
            input_layout,
            output_layout,
            sample_format,
        }
    }

    /// Calculate output buffer size for given input size.
    pub fn output_size(&self, input_size: usize) -> usize {
        let sample_size = self.sample_format.bytes_per_sample();
        let frame_size_in = sample_size * self.input_layout.channels();
        let frame_count = input_size / frame_size_in;
        frame_count * sample_size * self.output_layout.channels()
    }

    /// Mix audio channels from input layout to output layout.
    pub fn mix(&self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        let sample_size = self.sample_format.bytes_per_sample();
        let in_channels = self.input_layout.channels();
        let out_channels = self.output_layout.channels();
        let frame_size_in = sample_size * in_channels;
        let frame_size_out = sample_size * out_channels;

        if input.len() % frame_size_in != 0 {
            return Err(Error::Config(format!(
                "Input size {} not aligned to frame size {}",
                input.len(),
                frame_size_in
            )));
        }

        let frame_count = input.len() / frame_size_in;
        let required_output = frame_count * frame_size_out;

        if output.len() < required_output {
            return Err(Error::Config(format!(
                "Output buffer too small: {} < {}",
                output.len(),
                required_output
            )));
        }

        // Same layout - just copy
        if self.input_layout == self.output_layout {
            output[..input.len()].copy_from_slice(input);
            return Ok(input.len());
        }

        // Process each frame
        match (self.input_layout, self.output_layout) {
            (ChannelLayout::Mono, ChannelLayout::Stereo) => {
                // Duplicate mono to both channels
                for frame in 0..frame_count {
                    let in_offset = frame * frame_size_in;
                    let out_offset = frame * frame_size_out;

                    // Copy mono sample to left channel
                    output[out_offset..out_offset + sample_size]
                        .copy_from_slice(&input[in_offset..in_offset + sample_size]);

                    // Copy same sample to right channel
                    output[out_offset + sample_size..out_offset + 2 * sample_size]
                        .copy_from_slice(&input[in_offset..in_offset + sample_size]);
                }
            }
            (ChannelLayout::Stereo, ChannelLayout::Mono) => {
                // Average stereo to mono
                self.stereo_to_mono(input, output, frame_count)?;
            }
            _ => {
                return Err(Error::Config(format!(
                    "Unsupported channel conversion: {:?} -> {:?}",
                    self.input_layout, self.output_layout
                )));
            }
        }

        Ok(required_output)
    }

    /// Convert stereo to mono by averaging channels.
    fn stereo_to_mono(&self, input: &[u8], output: &mut [u8], frame_count: usize) -> Result<()> {
        let sample_size = self.sample_format.bytes_per_sample();
        let frame_size_in = sample_size * 2;

        for frame in 0..frame_count {
            let in_offset = frame * frame_size_in;
            let out_offset = frame * sample_size;

            // Read both channels and average
            let left = self.read_sample(input, in_offset);
            let right = self.read_sample(input, in_offset + sample_size);
            let mono = (left + right) / 2.0;

            self.write_sample(output, out_offset, mono);
        }

        Ok(())
    }

    /// Read a sample as f64.
    fn read_sample(&self, data: &[u8], offset: usize) -> f64 {
        match self.sample_format {
            SampleFormat::U8 => (data[offset] as f64 - 128.0) / 128.0,
            SampleFormat::S16Le => {
                i16::from_le_bytes([data[offset], data[offset + 1]]) as f64 / 32768.0
            }
            SampleFormat::S16Be => {
                i16::from_be_bytes([data[offset], data[offset + 1]]) as f64 / 32768.0
            }
            SampleFormat::S32Le => {
                i32::from_le_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) as f64
                    / 2147483648.0
            }
            SampleFormat::S32Be => {
                i32::from_be_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) as f64
                    / 2147483648.0
            }
            SampleFormat::F32Le => f32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as f64,
            SampleFormat::F32Be => f32::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as f64,
            SampleFormat::F64Le => f64::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]),
            SampleFormat::F64Be => f64::from_be_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
                data[offset + 4],
                data[offset + 5],
                data[offset + 6],
                data[offset + 7],
            ]),
        }
    }

    /// Write a f64 sample.
    fn write_sample(&self, data: &mut [u8], offset: usize, value: f64) {
        let value = value.clamp(-1.0, 1.0);

        match self.sample_format {
            SampleFormat::U8 => {
                data[offset] = ((value * 128.0) + 128.0).round() as u8;
            }
            SampleFormat::S16Le => {
                let v = (value * 32767.0).round() as i16;
                let bytes = v.to_le_bytes();
                data[offset] = bytes[0];
                data[offset + 1] = bytes[1];
            }
            SampleFormat::S16Be => {
                let v = (value * 32767.0).round() as i16;
                let bytes = v.to_be_bytes();
                data[offset] = bytes[0];
                data[offset + 1] = bytes[1];
            }
            SampleFormat::S32Le => {
                let v = (value * 2147483647.0).round() as i32;
                data[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
            }
            SampleFormat::S32Be => {
                let v = (value * 2147483647.0).round() as i32;
                data[offset..offset + 4].copy_from_slice(&v.to_be_bytes());
            }
            SampleFormat::F32Le => {
                data[offset..offset + 4].copy_from_slice(&(value as f32).to_le_bytes());
            }
            SampleFormat::F32Be => {
                data[offset..offset + 4].copy_from_slice(&(value as f32).to_be_bytes());
            }
            SampleFormat::F64Le => {
                data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
            }
            SampleFormat::F64Be => {
                data[offset..offset + 8].copy_from_slice(&value.to_be_bytes());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s16_to_f32_conversion() {
        let conv = AudioConvert::new(SampleFormat::S16Le, SampleFormat::F32Le, 1).unwrap();

        // -32768, 0, 32767
        let input = [0x00u8, 0x80, 0x00, 0x00, 0xFF, 0x7F];
        let mut output = vec![0u8; 12]; // 3 floats

        let written = conv.convert(&input, &mut output).unwrap();
        assert_eq!(written, 12);

        let f0 = f32::from_le_bytes([output[0], output[1], output[2], output[3]]);
        let f1 = f32::from_le_bytes([output[4], output[5], output[6], output[7]]);
        let f2 = f32::from_le_bytes([output[8], output[9], output[10], output[11]]);

        assert!(
            (f0 - (-1.0)).abs() < 0.001,
            "min should be -1.0, got {}",
            f0
        );
        assert!(f1.abs() < 0.001, "zero should be 0.0, got {}", f1);
        assert!(
            (f2 - 0.99997).abs() < 0.001,
            "max should be ~1.0, got {}",
            f2
        );
    }

    #[test]
    fn test_f32_to_s16_conversion() {
        let conv = AudioConvert::new(SampleFormat::F32Le, SampleFormat::S16Le, 1).unwrap();

        // -1.0, 0.0, 1.0
        let mut input = vec![0u8; 12];
        input[0..4].copy_from_slice(&(-1.0f32).to_le_bytes());
        input[4..8].copy_from_slice(&(0.0f32).to_le_bytes());
        input[8..12].copy_from_slice(&(1.0f32).to_le_bytes());

        let mut output = vec![0u8; 6];
        let written = conv.convert(&input, &mut output).unwrap();
        assert_eq!(written, 6);

        let s0 = i16::from_le_bytes([output[0], output[1]]);
        let s1 = i16::from_le_bytes([output[2], output[3]]);
        let s2 = i16::from_le_bytes([output[4], output[5]]);

        assert_eq!(s0, -32767);
        assert_eq!(s1, 0);
        assert_eq!(s2, 32767);
    }

    #[test]
    fn test_s16_roundtrip() {
        let to_f32 = AudioConvert::new(SampleFormat::S16Le, SampleFormat::F32Le, 1).unwrap();
        let to_s16 = AudioConvert::new(SampleFormat::F32Le, SampleFormat::S16Le, 1).unwrap();

        let original: Vec<i16> = vec![-32768, -16384, 0, 16384, 32767];
        let mut input = vec![0u8; original.len() * 2];
        for (i, &v) in original.iter().enumerate() {
            input[i * 2..i * 2 + 2].copy_from_slice(&v.to_le_bytes());
        }

        let mut float_buf = vec![0u8; original.len() * 4];
        let mut result = vec![0u8; original.len() * 2];

        to_f32.convert(&input, &mut float_buf).unwrap();
        to_s16.convert(&float_buf, &mut result).unwrap();

        for (i, &expected) in original.iter().enumerate() {
            let actual = i16::from_le_bytes([result[i * 2], result[i * 2 + 1]]);
            // Allow ±1 for rounding
            assert!(
                (actual as i32 - expected as i32).abs() <= 1,
                "sample {} expected {} got {}",
                i,
                expected,
                actual
            );
        }
    }

    #[test]
    fn test_mono_to_stereo() {
        let mixer = AudioChannelMix::new(
            ChannelLayout::Mono,
            ChannelLayout::Stereo,
            SampleFormat::S16Le,
        );

        // Two mono samples
        let input = [0x00u8, 0x40, 0x00, 0x80]; // 16384, -32768
        let mut output = vec![0u8; 8]; // Two stereo frames

        let written = mixer.mix(&input, &mut output).unwrap();
        assert_eq!(written, 8);

        // Left and right should be identical
        assert_eq!(&output[0..2], &input[0..2]); // L0
        assert_eq!(&output[2..4], &input[0..2]); // R0
        assert_eq!(&output[4..6], &input[2..4]); // L1
        assert_eq!(&output[6..8], &input[2..4]); // R1
    }

    #[test]
    fn test_stereo_to_mono() {
        let mixer = AudioChannelMix::new(
            ChannelLayout::Stereo,
            ChannelLayout::Mono,
            SampleFormat::S16Le,
        );

        // Stereo: L=16384, R=-16384 -> should average to 0
        let input = [0x00u8, 0x40, 0x00, 0xC0];
        let mut output = vec![0u8; 2];

        let written = mixer.mix(&input, &mut output).unwrap();
        assert_eq!(written, 2);

        let mono = i16::from_le_bytes([output[0], output[1]]);
        assert!(mono.abs() < 10, "average should be ~0, got {}", mono);
    }

    #[test]
    fn test_channel_layout_counts() {
        assert_eq!(ChannelLayout::Mono.channels(), 1);
        assert_eq!(ChannelLayout::Stereo.channels(), 2);
        assert_eq!(ChannelLayout::Surround51.channels(), 6);
        assert_eq!(ChannelLayout::Surround71.channels(), 8);
    }

    #[test]
    fn test_sample_format_sizes() {
        assert_eq!(SampleFormat::U8.bytes_per_sample(), 1);
        assert_eq!(SampleFormat::S16Le.bytes_per_sample(), 2);
        assert_eq!(SampleFormat::S32Le.bytes_per_sample(), 4);
        assert_eq!(SampleFormat::F32Le.bytes_per_sample(), 4);
        assert_eq!(SampleFormat::F64Le.bytes_per_sample(), 8);
    }
}
