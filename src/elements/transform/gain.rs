//! Gain element for RT-safe audio amplitude adjustment.
//!
//! Multiplies audio samples by a constant factor in-place.
//! No allocations, no blocking — safe for real-time threads.

use crate::buffer::Buffer;
use crate::element::{Affinity, Element, ExecutionHints, LatencyHint, ProcessingHint};
use crate::error::Result;
use std::sync::atomic::{AtomicU32, Ordering};

/// RT-safe gain element that multiplies audio samples by a constant factor.
///
/// Operates on f32 (little-endian) samples in-place. No allocation occurs
/// during processing, making it suitable for RT data threads.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::elements::Gain;
///
/// // Double the volume
/// let gain = Gain::new(2.0);
///
/// // Halve the volume
/// let gain = Gain::new(0.5);
///
/// // Mute
/// let gain = Gain::new(0.0);
/// ```
pub struct Gain {
    /// Gain factor stored as atomic u32 (bit-cast f32) for lock-free updates.
    factor: AtomicU32,
    name: String,
}

impl Gain {
    /// Create a new gain element with the given factor.
    ///
    /// - `1.0` = unity gain (no change)
    /// - `0.0` = mute
    /// - `2.0` = double amplitude (+6 dB)
    /// - `0.5` = halve amplitude (-6 dB)
    pub fn new(factor: f32) -> Self {
        Self {
            factor: AtomicU32::new(factor.to_bits()),
            name: "gain".to_string(),
        }
    }

    /// Set a custom name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Get the current gain factor.
    pub fn factor(&self) -> f32 {
        f32::from_bits(self.factor.load(Ordering::Relaxed))
    }

    /// Set the gain factor (thread-safe, lock-free).
    ///
    /// Can be called from any thread while the pipeline is running.
    pub fn set_factor(&self, factor: f32) {
        self.factor.store(factor.to_bits(), Ordering::Relaxed);
    }

    /// Convert a linear gain factor to decibels.
    pub fn to_db(factor: f32) -> f32 {
        20.0 * factor.log10()
    }

    /// Convert decibels to a linear gain factor.
    pub fn from_db(db: f32) -> f32 {
        10.0_f32.powf(db / 20.0)
    }
}

impl Default for Gain {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl Element for Gain {
    fn process(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
        let factor = self.factor();

        // Fast path: unity gain — no work needed
        if (factor - 1.0).abs() < f32::EPSILON {
            return Ok(Some(buffer));
        }

        // Fast path: mute — zero the buffer
        if factor == 0.0 {
            buffer.as_bytes_mut().fill(0);
            return Ok(Some(buffer));
        }

        // Apply gain to f32 samples in-place
        let data = buffer.as_bytes_mut();
        for sample in data.chunks_exact_mut(4) {
            let val = f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
            let out = val * factor;
            sample.copy_from_slice(&out.to_le_bytes());
        }

        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_rt_safe(&self) -> bool {
        true
    }

    fn affinity(&self) -> Affinity {
        Affinity::Auto
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints {
            processing: ProcessingHint::CpuBound,
            latency: LatencyHint::Low,
            ..ExecutionHints::trusted()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(256, 64).unwrap())
    }

    fn create_f32_buffer(samples: &[f32], seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let byte_len = samples.len() * 4;
        let handle = MemoryHandle::with_len(slot, byte_len);
        let mut buffer = Buffer::new(handle, Metadata::from_sequence(seq));
        let data = buffer.as_bytes_mut();
        for (i, &sample) in samples.iter().enumerate() {
            data[i * 4..(i + 1) * 4].copy_from_slice(&sample.to_le_bytes());
        }
        buffer
    }

    fn read_f32_samples(buffer: &Buffer) -> Vec<f32> {
        buffer
            .as_bytes()
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect()
    }

    #[test]
    fn test_gain_unity() {
        let mut gain = Gain::new(1.0);
        let buffer = create_f32_buffer(&[1.0, -0.5, 0.25], 0);
        let result = gain.process(buffer).unwrap().unwrap();
        let samples = read_f32_samples(&result);
        assert_eq!(samples, vec![1.0, -0.5, 0.25]);
    }

    #[test]
    fn test_gain_double() {
        let mut gain = Gain::new(2.0);
        let buffer = create_f32_buffer(&[0.5, -0.25, 0.1], 0);
        let result = gain.process(buffer).unwrap().unwrap();
        let samples = read_f32_samples(&result);
        assert_eq!(samples, vec![1.0, -0.5, 0.2]);
    }

    #[test]
    fn test_gain_mute() {
        let mut gain = Gain::new(0.0);
        let buffer = create_f32_buffer(&[1.0, -0.5, 0.25], 0);
        let result = gain.process(buffer).unwrap().unwrap();
        let samples = read_f32_samples(&result);
        assert_eq!(samples, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_gain_set_factor() {
        let gain = Gain::new(1.0);
        assert!((gain.factor() - 1.0).abs() < f32::EPSILON);
        gain.set_factor(0.5);
        assert!((gain.factor() - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_gain_is_rt_safe() {
        let gain = Gain::new(1.0);
        assert!(gain.is_rt_safe());
    }

    #[test]
    fn test_gain_db_conversion() {
        assert!((Gain::from_db(0.0) - 1.0).abs() < 0.001);
        assert!((Gain::from_db(6.0) - 2.0).abs() < 0.05);
        assert!((Gain::from_db(-6.0) - 0.5).abs() < 0.02);
    }

    #[test]
    fn test_gain_default() {
        let gain = Gain::default();
        assert!((gain.factor() - 1.0).abs() < f32::EPSILON);
    }
}
