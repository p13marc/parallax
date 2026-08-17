//! Gain element for RT-safe audio amplitude adjustment.
//!
//! Multiplies audio samples by a constant factor in-place, branching on the
//! buffer's [`MediaFormat::AudioRaw`] metadata (S16/S32/F32/U8). A buffer
//! without that metadata is an error — the old behavior silently assumed
//! f32LE, which corrupted S16 streams into noise (#159).
//! No allocations, no blocking — safe for real-time threads.

use crate::buffer::Buffer;
use crate::element::{Element, ExecutionHints, LatencyHint, ProcessingHint};
use crate::error::{Error, Result};
use crate::format::{MediaFormat, SampleFormat};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

/// Control handle for a [`Gain`] element.
///
/// Clone it *before* `executor.start()` — `start` moves the element into its
/// task, so this handle is the only way to reach a live gain. Cloneable,
/// lock-free, safe to call from any thread.
///
/// # Example
///
/// ```rust,ignore
/// let gain = Gain::new(1.0);
/// let volume = gain.control();        // BEFORE start
/// pipeline.add_filter("volume", gain);
/// let handle = executor.start(&mut pipeline)?;
///
/// volume.set_db(-6.0);                // half amplitude, while playing
/// volume.set_factor(0.0);             // mute
/// ```
#[derive(Clone, Debug)]
pub struct GainControl(Arc<AtomicU32>);

impl GainControl {
    /// Set the linear gain factor (`1.0` = unity, `0.0` = mute).
    pub fn set_factor(&self, factor: f32) {
        self.0.store(factor.to_bits(), Ordering::Relaxed);
    }

    /// The current linear gain factor.
    pub fn factor(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }

    /// Set the gain in decibels (`0.0` dB = unity, `-6.0` dB ≈ half amplitude).
    ///
    /// There is no dB value for silence; use `set_factor(0.0)` to mute.
    pub fn set_db(&self, db: f32) {
        self.set_factor(Gain::from_db(db));
    }

    /// The current gain in decibels.
    ///
    /// Returns `f32::NEG_INFINITY` when muted, which is what `log10(0.0)`
    /// yields — a muted gain has no finite dB value.
    pub fn db(&self) -> f32 {
        Gain::to_db(self.factor())
    }
}

/// RT-safe gain element that multiplies audio samples by a constant factor.
///
/// Format-aware and in-place: each buffer's [`MediaFormat::AudioRaw`]
/// metadata (which every audio decoder stamps per buffer) selects the
/// sample interpretation — S16, S32, F32 or U8, little-endian. A buffer
/// carrying no `AudioRaw` metadata is a hard error, at any factor: the
/// alternative was assuming f32 and turning S16 audio into noise the
/// moment the volume moved off unity. No allocation occurs during
/// processing, making it suitable for RT data threads.
///
/// The factor can be changed on a running pipeline through
/// [`control`](Self::control).
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
    ///
    /// Shared with [`GainControl`] handles, so it can change while the
    /// pipeline runs. The `Arc` is touched only when a handle is cloned —
    /// `process` does a single relaxed load, so this stays RT-safe.
    factor: Arc<AtomicU32>,
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
            factor: Arc::new(AtomicU32::new(factor.to_bits())),
            name: "gain".to_string(),
        }
    }

    /// Get a cloneable handle for changing the gain at runtime.
    ///
    /// Clone it *before* the pipeline starts — see [`GainControl`].
    pub fn control(&self) -> GainControl {
        GainControl(Arc::clone(&self.factor))
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

impl crate::control::Controllable for Gain {
    type Control = GainControl;

    fn control(&self) -> GainControl {
        GainControl(Arc::clone(&self.factor))
    }
}

impl Element for Gain {
    // #189: may forward the input buffer — the upstream producer's arena
    // budget accumulates through this element.
    fn passthrough(&self) -> bool {
        true
    }

    fn process(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
        // Metadata first, before any fast path: erroring only once the
        // factor moves off unity would let a mis-wired pipeline play fine
        // until the user touches the volume — fail on the first buffer
        // instead. (The AudioRaw clone is a Copy struct; no allocation.)
        let Some(MediaFormat::AudioRaw(fmt)) = buffer.metadata().format.clone() else {
            return Err(Error::Config(format!(
                "{}: buffer carries no AudioRaw format metadata \
                 (decoders set it per buffer; see MediaFormat::AudioRaw)",
                self.name
            )));
        };

        let factor = self.factor();

        // Fast path: unity gain — no work needed
        if (factor - 1.0).abs() < f32::EPSILON {
            return Ok(Some(buffer));
        }

        // Fast path: mute. Digital silence is 0 for the signed and float
        // formats but the midpoint for unsigned 8-bit.
        if factor == 0.0 {
            let silence = match fmt.sample_format {
                SampleFormat::U8 => 0x80,
                _ => 0,
            };
            buffer.as_bytes_mut().fill(silence);
            return Ok(Some(buffer));
        }

        // Scale in-place, branched on the stream's actual sample format.
        let data = buffer.as_bytes_mut();
        match fmt.sample_format {
            SampleFormat::F32 => {
                for sample in data.chunks_exact_mut(4) {
                    let val = f32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
                    sample.copy_from_slice(&(val * factor).to_le_bytes());
                }
            }
            SampleFormat::S16 => {
                for sample in data.chunks_exact_mut(2) {
                    let val = i16::from_le_bytes([sample[0], sample[1]]);
                    let out = (f32::from(val) * factor).clamp(-32768.0, 32767.0) as i16;
                    sample.copy_from_slice(&out.to_le_bytes());
                }
            }
            SampleFormat::S32 => {
                // f64: f32's 24-bit mantissa cannot represent i32 exactly.
                for sample in data.chunks_exact_mut(4) {
                    let val = i32::from_le_bytes([sample[0], sample[1], sample[2], sample[3]]);
                    let out = (f64::from(val) * f64::from(factor))
                        .clamp(f64::from(i32::MIN), f64::from(i32::MAX))
                        as i32;
                    sample.copy_from_slice(&out.to_le_bytes());
                }
            }
            SampleFormat::U8 => {
                // Scale around the unsigned midpoint, not around zero.
                for sample in data.iter_mut() {
                    let centered = f32::from(*sample) - 128.0;
                    *sample = (centered * factor + 128.0).clamp(0.0, 255.0) as u8;
                }
            }
        }

        Ok(Some(buffer))
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints {
            rt_safe: true,
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

    fn audio_metadata(seq: u64, format: SampleFormat) -> Metadata {
        use crate::format::AudioFormat;
        let mut metadata = Metadata::from_sequence(seq);
        metadata.format = Some(MediaFormat::AudioRaw(AudioFormat::new(48_000, 2, format)));
        metadata
    }

    fn create_f32_buffer(samples: &[f32], seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let byte_len = samples.len() * 4;
        let handle = MemoryHandle::with_len(slot, byte_len);
        let mut buffer = Buffer::new(handle, audio_metadata(seq, SampleFormat::F32));
        let data = buffer.as_bytes_mut();
        for (i, &sample) in samples.iter().enumerate() {
            data[i * 4..(i + 1) * 4].copy_from_slice(&sample.to_le_bytes());
        }
        buffer
    }

    fn create_s16_buffer(samples: &[i16], seq: u64) -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let byte_len = samples.len() * 2;
        let handle = MemoryHandle::with_len(slot, byte_len);
        let mut buffer = Buffer::new(handle, audio_metadata(seq, SampleFormat::S16));
        let data = buffer.as_bytes_mut();
        for (i, &sample) in samples.iter().enumerate() {
            data[i * 2..(i + 1) * 2].copy_from_slice(&sample.to_le_bytes());
        }
        buffer
    }

    fn read_s16_samples(buffer: &Buffer) -> Vec<i16> {
        buffer
            .as_bytes()
            .chunks_exact(2)
            .map(|chunk| i16::from_le_bytes([chunk[0], chunk[1]]))
            .collect()
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
    fn s16_streams_scale_as_integers_not_reinterpreted_f32() {
        // The old code read S16 pairs as f32 — noise. Half amplitude must
        // be exact integer halving.
        let mut gain = Gain::new(0.5);
        let buffer = create_s16_buffer(&[10_000, -10_000, 0, 32_000], 0);
        let result = gain.process(buffer).unwrap().unwrap();
        assert_eq!(read_s16_samples(&result), vec![5_000, -5_000, 0, 16_000]);
    }

    #[test]
    fn s16_gain_clamps_instead_of_wrapping() {
        let mut gain = Gain::new(4.0);
        let buffer = create_s16_buffer(&[20_000, -20_000], 0);
        let result = gain.process(buffer).unwrap().unwrap();
        assert_eq!(read_s16_samples(&result), vec![32_767, -32_768]);
    }

    #[test]
    fn u8_mute_is_the_midpoint_not_zero() {
        use crate::format::AudioFormat;
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 4);
        let mut metadata = Metadata::from_sequence(0);
        metadata.format = Some(MediaFormat::AudioRaw(AudioFormat::new(
            8_000,
            1,
            SampleFormat::U8,
        )));
        let mut buffer = Buffer::new(handle, metadata);
        buffer.as_bytes_mut().copy_from_slice(&[0, 64, 192, 255]);

        let mut gain = Gain::new(0.0);
        let result = gain.process(buffer).unwrap().unwrap();
        // 0x80 is U8 digital silence; a fill(0) would be a full-scale DC
        // offset, i.e. a loud pop.
        assert_eq!(result.as_bytes(), &[0x80; 4]);
    }

    #[test]
    fn a_buffer_without_audio_metadata_is_an_error_even_at_unity() {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 8);
        let buffer = Buffer::new(handle, Metadata::from_sequence(0));

        // Unity too: erroring only once the volume moves would let a
        // mis-wired pipeline play fine until the first keypress.
        let mut gain = Gain::new(1.0);
        assert!(gain.process(buffer).is_err());
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
        assert!(gain.execution_hints().is_rt_safe());
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

    // ------------------------------------------------------------------
    // Runtime gain control (#75)
    // ------------------------------------------------------------------

    #[test]
    fn factor_change_takes_effect_immediately() {
        let mut gain = Gain::new(1.0);
        let volume = gain.control();

        let first = gain
            .process(create_f32_buffer(&[0.5, -0.5], 0))
            .unwrap()
            .unwrap();
        assert_eq!(read_f32_samples(&first), vec![0.5, -0.5]);

        volume.set_factor(2.0);

        let second = gain
            .process(create_f32_buffer(&[0.5, -0.5], 1))
            .unwrap()
            .unwrap();
        assert_eq!(read_f32_samples(&second), vec![1.0, -1.0]);
    }

    #[test]
    fn control_clones_share_state() {
        let gain = Gain::new(1.0);
        let a = gain.control();
        let b = a.clone();

        b.set_factor(0.25);

        assert!((a.factor() - 0.25).abs() < f32::EPSILON);
        assert!((gain.factor() - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn control_reports_the_current_gain() {
        let gain = Gain::new(0.5);
        let volume = gain.control();
        assert!((volume.factor() - 0.5).abs() < f32::EPSILON);

        // The element's own setter is visible through the handle, and back.
        gain.set_factor(2.0);
        assert!((volume.factor() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn set_db_matches_the_linear_conversion() {
        let gain = Gain::new(1.0);
        let volume = gain.control();

        volume.set_db(-6.0);
        assert!((volume.factor() - 0.5).abs() < 0.02);
        assert!((volume.db() + 6.0).abs() < 0.05);

        volume.set_db(0.0);
        assert!((volume.factor() - 1.0).abs() < 0.001);
    }

    #[test]
    fn muting_through_the_handle_has_no_finite_db() {
        let gain = Gain::new(1.0);
        let volume = gain.control();
        volume.set_factor(0.0);
        assert_eq!(volume.db(), f32::NEG_INFINITY);
    }

    #[test]
    fn controllable_and_inherent_control_agree() {
        use crate::control::Controllable;

        let gain = Gain::new(1.0);
        let inherent = Gain::control(&gain);
        let via_trait = Controllable::control(&gain);

        via_trait.set_factor(0.75);
        assert!((inherent.factor() - 0.75).abs() < f32::EPSILON);
    }
}
