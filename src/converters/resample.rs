//! Audio sample rate conversion (resampling).
//!
//! Provides pure Rust implementations of audio resampling algorithms.

use crate::error::{Error, Result};

use super::SampleFormat;

/// Resampling quality level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResampleQuality {
    /// Linear interpolation - fast but lower quality.
    Fast,
    /// Cubic interpolation - good balance of speed and quality.
    #[default]
    Medium,
}

/// Audio resampler.
///
/// Converts audio between different sample rates.
pub struct AudioResample {
    input_rate: u32,
    output_rate: u32,
    channels: u32,
    format: SampleFormat,
    quality: ResampleQuality,
    /// Fractional sample position for accurate resampling
    phase: f64,
    /// History buffer for interpolation
    history: Vec<f64>,
}

impl AudioResample {
    /// Create a new audio resampler.
    pub fn new(
        input_rate: u32,
        output_rate: u32,
        channels: u32,
        format: SampleFormat,
    ) -> Result<Self> {
        if input_rate == 0 || output_rate == 0 {
            return Err(Error::Config("Sample rates must be non-zero".into()));
        }
        if channels == 0 {
            return Err(Error::Config("Channels must be non-zero".into()));
        }

        // History size: 4 samples per channel for cubic interpolation
        let history_size = channels as usize * 4;

        Ok(Self {
            input_rate,
            output_rate,
            channels,
            format,
            quality: ResampleQuality::default(),
            phase: 0.0,
            history: vec![0.0; history_size],
        })
    }

    /// Set the resampling quality.
    pub fn with_quality(mut self, quality: ResampleQuality) -> Self {
        self.quality = quality;
        self
    }

    /// Get the input sample rate.
    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    /// Get the output sample rate.
    pub fn output_rate(&self) -> u32 {
        self.output_rate
    }

    /// Calculate approximate output size for given input size.
    ///
    /// The actual output size may vary slightly due to fractional samples.
    pub fn output_size(&self, input_size: usize) -> usize {
        let sample_size = self.format.bytes_per_sample();
        let frame_size = sample_size * self.channels as usize;
        let input_frames = input_size / frame_size;

        let output_frames = (input_frames as f64 * self.output_rate as f64 / self.input_rate as f64)
            .ceil() as usize;

        output_frames * frame_size
    }

    /// Reset the resampler state.
    ///
    /// Call this when starting a new audio stream to clear history.
    pub fn reset(&mut self) {
        self.phase = 0.0;
        self.history.fill(0.0);
    }

    /// Resample audio data.
    ///
    /// Returns the number of bytes written to the output buffer.
    pub fn resample(&mut self, input: &[u8], output: &mut [u8]) -> Result<usize> {
        // Same rate - just copy
        if self.input_rate == self.output_rate {
            let len = input.len().min(output.len());
            output[..len].copy_from_slice(&input[..len]);
            return Ok(len);
        }

        let sample_size = self.format.bytes_per_sample();
        let frame_size = sample_size * self.channels as usize;

        if input.len() % frame_size != 0 {
            return Err(Error::Config(format!(
                "Input size {} not aligned to frame size {}",
                input.len(),
                frame_size
            )));
        }

        let input_frames = input.len() / frame_size;
        if input_frames == 0 {
            return Ok(0);
        }

        // Read input into f64 buffer
        let input_samples = self.read_samples(input);

        // Resample
        let output_samples = match self.quality {
            ResampleQuality::Fast => self.resample_linear(&input_samples),
            ResampleQuality::Medium => self.resample_cubic(&input_samples),
        };

        // Write output
        let written = self.write_samples(&output_samples, output)?;

        Ok(written)
    }

    /// Read input samples as f64 (interleaved).
    fn read_samples(&self, data: &[u8]) -> Vec<f64> {
        let sample_size = self.format.bytes_per_sample();
        let sample_count = data.len() / sample_size;
        let mut samples = Vec::with_capacity(sample_count);

        for i in 0..sample_count {
            let offset = i * sample_size;
            let value = self.read_sample(data, offset);
            samples.push(value);
        }

        samples
    }

    /// Write f64 samples to output buffer.
    fn write_samples(&self, samples: &[f64], output: &mut [u8]) -> Result<usize> {
        let sample_size = self.format.bytes_per_sample();
        let required = samples.len() * sample_size;

        if output.len() < required {
            return Err(Error::Config(format!(
                "Output buffer too small: {} < {}",
                output.len(),
                required
            )));
        }

        for (i, &sample) in samples.iter().enumerate() {
            let offset = i * sample_size;
            self.write_sample(output, offset, sample);
        }

        Ok(required)
    }

    /// Linear interpolation resampling.
    fn resample_linear(&mut self, input: &[f64]) -> Vec<f64> {
        let ratio = self.input_rate as f64 / self.output_rate as f64;
        let channels = self.channels as usize;
        let input_frames = input.len() / channels;

        // Estimate output frames
        let output_frames_est = ((input_frames as f64 / ratio) + 1.0).ceil() as usize;

        let mut output = Vec::with_capacity(output_frames_est * channels);

        while self.phase < input_frames as f64 {
            let idx0 = self.phase.floor() as usize;
            let frac = self.phase - idx0 as f64;

            for ch in 0..channels {
                let s0 = if idx0 < input_frames {
                    input[idx0 * channels + ch]
                } else {
                    0.0
                };

                let s1 = if idx0 + 1 < input_frames {
                    input[(idx0 + 1) * channels + ch]
                } else {
                    s0
                };

                // Linear interpolation
                let interpolated = s0 + frac * (s1 - s0);
                output.push(interpolated);
            }

            self.phase += ratio;
        }

        // Adjust phase for next call
        self.phase -= input_frames as f64;

        output
    }

    /// Cubic interpolation resampling.
    fn resample_cubic(&mut self, input: &[f64]) -> Vec<f64> {
        let ratio = self.input_rate as f64 / self.output_rate as f64;
        let channels = self.channels as usize;
        let input_frames = input.len() / channels;

        // Prepend history
        let mut extended = self.history.clone();
        extended.extend_from_slice(input);

        // History offset (4 samples per channel)
        let history_frames = 4;

        let mut output = Vec::new();

        // Adjust phase to account for history
        let mut pos = self.phase + history_frames as f64;

        while pos < (input_frames + history_frames) as f64 {
            let idx = pos.floor() as usize;
            let frac = pos - idx as f64;

            for ch in 0..channels {
                // Get 4 samples for cubic interpolation
                let get_sample = |i: usize| -> f64 {
                    if i < extended.len() / channels {
                        extended[i * channels + ch]
                    } else {
                        0.0
                    }
                };

                let s0 = get_sample(idx.saturating_sub(1));
                let s1 = get_sample(idx);
                let s2 = get_sample(idx + 1);
                let s3 = get_sample(idx + 2);

                // Catmull-Rom cubic interpolation
                let interpolated = cubic_interpolate(s0, s1, s2, s3, frac);
                output.push(interpolated.clamp(-1.0, 1.0));
            }

            pos += ratio;
        }

        // Update history (last 4 frames of input)
        let start = input.len().saturating_sub(history_frames * channels);
        self.history.clear();
        if start < input.len() {
            self.history.extend_from_slice(&input[start..]);
        }
        // Pad history if needed
        while self.history.len() < history_frames * channels {
            self.history.insert(0, 0.0);
        }

        // Adjust phase for next call
        self.phase = pos - (input_frames + history_frames) as f64;

        output
    }

    /// Read a single sample as f64.
    fn read_sample(&self, data: &[u8], offset: usize) -> f64 {
        match self.format {
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

    /// Write a single sample.
    fn write_sample(&self, data: &mut [u8], offset: usize, value: f64) {
        let value = value.clamp(-1.0, 1.0);

        match self.format {
            SampleFormat::U8 => {
                data[offset] = ((value * 128.0) + 128.0).round() as u8;
            }
            SampleFormat::S16Le => {
                let v = (value * 32767.0).round() as i16;
                data[offset..offset + 2].copy_from_slice(&v.to_le_bytes());
            }
            SampleFormat::S16Be => {
                let v = (value * 32767.0).round() as i16;
                data[offset..offset + 2].copy_from_slice(&v.to_be_bytes());
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

/// Catmull-Rom cubic interpolation.
///
/// Interpolates between s1 and s2 using s0 and s3 as control points.
/// t is the fractional position between s1 and s2 (0.0 to 1.0).
#[inline]
fn cubic_interpolate(s0: f64, s1: f64, s2: f64, s3: f64, t: f64) -> f64 {
    let t2 = t * t;
    let t3 = t2 * t;

    // Catmull-Rom coefficients
    let a0 = -0.5 * s0 + 1.5 * s1 - 1.5 * s2 + 0.5 * s3;
    let a1 = s0 - 2.5 * s1 + 2.0 * s2 - 0.5 * s3;
    let a2 = -0.5 * s0 + 0.5 * s2;
    let a3 = s1;

    a0 * t3 + a1 * t2 + a2 * t + a3
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_rate_passthrough() {
        let mut resampler = AudioResample::new(48000, 48000, 2, SampleFormat::S16Le).unwrap();

        let input = [0x00u8, 0x40, 0x00, 0xC0]; // Two S16 samples
        let mut output = vec![0u8; 4];

        let written = resampler.resample(&input, &mut output).unwrap();
        assert_eq!(written, 4);
        assert_eq!(output, input);
    }

    #[test]
    fn test_downsample_2x() {
        let mut resampler = AudioResample::new(48000, 24000, 1, SampleFormat::S16Le)
            .unwrap()
            .with_quality(ResampleQuality::Fast);

        // 4 samples at 48kHz should produce ~2 samples at 24kHz
        let input = [
            0x00u8, 0x40, // 16384
            0x00, 0x00, // 0
            0x00, 0xC0, // -16384
            0x00, 0x00, // 0
        ];
        let mut output = vec![0u8; 4];

        let written = resampler.resample(&input, &mut output).unwrap();

        // Should have written 2 samples (4 bytes)
        assert!(written >= 2 && written <= 6, "wrote {} bytes", written);
    }

    #[test]
    fn test_upsample_2x() {
        let mut resampler = AudioResample::new(24000, 48000, 1, SampleFormat::S16Le)
            .unwrap()
            .with_quality(ResampleQuality::Fast);

        // 2 samples at 24kHz should produce ~4 samples at 48kHz
        let input = [
            0x00u8, 0x40, // 16384
            0x00, 0xC0, // -16384
        ];
        let mut output = vec![0u8; 16];

        let written = resampler.resample(&input, &mut output).unwrap();

        // Should produce approximately 4 samples
        assert!(written >= 6 && written <= 10, "wrote {} bytes", written);
    }

    #[test]
    fn test_stereo_resample() {
        let mut resampler = AudioResample::new(48000, 24000, 2, SampleFormat::S16Le)
            .unwrap()
            .with_quality(ResampleQuality::Fast);

        // 2 stereo frames at 48kHz
        let input = [
            0x00u8, 0x40, 0x00, 0xC0, // L=16384, R=-16384
            0x00, 0x00, 0x00, 0x00, // L=0, R=0
        ];
        let mut output = vec![0u8; 8];

        let written = resampler.resample(&input, &mut output).unwrap();

        // Should produce approximately 1 stereo frame
        assert!(written >= 4, "wrote {} bytes", written);
    }

    #[test]
    fn test_cubic_interpolation() {
        // Test at midpoint
        let result = cubic_interpolate(0.0, 0.0, 1.0, 1.0, 0.5);
        assert!(
            (result - 0.5).abs() < 0.1,
            "midpoint should be ~0.5, got {}",
            result
        );

        // Test at boundaries
        let at_zero = cubic_interpolate(0.0, 1.0, 2.0, 3.0, 0.0);
        assert!(
            (at_zero - 1.0).abs() < 0.01,
            "at t=0 should be s1=1.0, got {}",
            at_zero
        );

        let at_one = cubic_interpolate(0.0, 1.0, 2.0, 3.0, 1.0);
        assert!(
            (at_one - 2.0).abs() < 0.01,
            "at t=1 should be s2=2.0, got {}",
            at_one
        );
    }

    #[test]
    fn test_output_size_estimate() {
        let resampler = AudioResample::new(48000, 44100, 2, SampleFormat::S16Le).unwrap();

        // 1 second of audio at 48kHz stereo S16 = 48000 * 2 * 2 = 192000 bytes
        let input_size = 192000;
        let output_size = resampler.output_size(input_size);

        // Should be approximately 44100 * 2 * 2 = 176400 bytes
        let expected = 44100 * 2 * 2;
        assert!(
            (output_size as i64 - expected as i64).abs() < 1000,
            "expected ~{}, got {}",
            expected,
            output_size
        );
    }

    #[test]
    fn test_reset() {
        let mut resampler = AudioResample::new(48000, 24000, 1, SampleFormat::S16Le).unwrap();

        // Process some audio
        let input = [0x00u8, 0x40, 0x00, 0x00];
        let mut output = vec![0u8; 4];
        resampler.resample(&input, &mut output).unwrap();

        // Reset should clear state
        resampler.reset();
        assert_eq!(resampler.phase, 0.0);
        assert!(resampler.history.iter().all(|&x| x == 0.0));
    }
}
