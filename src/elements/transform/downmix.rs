//! Multichannel → stereo audio downmix element.
//!
//! Folds 5.1 (6-channel) and 7.1 (8-channel) interleaved PCM down to
//! stereo with ITU-R BS.775 coefficients; mono and stereo pass through
//! untouched. Channel order is the WAV/SMPTE convention every decoder in
//! the tree emits: FL FR FC LFE BL BR (SL SR).
//!
//! Geometry travels in-band: the channel count and sample format come
//! from each buffer's `MediaFormat::AudioRaw` metadata (which
//! `AudioDecoderElement` writes per buffer), and the output metadata is
//! rewritten to 2 channels — a stale channel count is exactly the kind of
//! silent mis-sizing the in-band convention exists to prevent.

use crate::buffer::{Buffer, MemoryHandle};
use crate::element::Element;
use crate::error::{Error, Result};
use crate::format::{AudioFormat, Caps, MediaFormat, SampleFormat};
use crate::memory::{OutputArena, OutputBudget, defaults};

/// ITU-R BS.775 fold-down gain for center and surround channels.
const SURROUND_GAIN: f32 = std::f32::consts::FRAC_1_SQRT_2; // −3 dB

/// Multichannel → stereo downmix, ITU-R BS.775 fold-down.
///
/// The module this lives in is private, so its docs are not published —
/// everything a caller needs is here and on the methods below.
///
/// ```rust,ignore
/// let node = pipeline.add_filter("downmix", AudioDownmix::new());
/// pipeline.link(decoder, node)?;
/// pipeline.link(node, sink)?;
/// ```
pub struct AudioDownmix {
    name: String,
    output: OutputArena,
}

impl AudioDownmix {
    /// Create a downmix element.
    pub fn new() -> Self {
        Self {
            name: "audiodownmix".to_string(),
            output: OutputArena::new(defaults::TRANSFORM_SLOT_COUNT)
                .with_min_slot_size(defaults::AUDIO_SLOT_SIZE)
                .grow_to_fit(),
        }
    }

    /// Fold one interleaved frame of `ch` f32 samples into stereo.
    ///
    /// Normalized by the total contribution per output channel so a
    /// full-scale input cannot clip: 1 / (1 + 0.707·(center+surrounds)).
    fn fold(frame: &[f32], ch: usize) -> (f32, f32) {
        let g = SURROUND_GAIN;
        match ch {
            6 => {
                // FL FR FC LFE BL BR — LFE dropped (§BS.775 default).
                let norm = 1.0 / (1.0 + g + g);
                (
                    (frame[0] + g * frame[2] + g * frame[4]) * norm,
                    (frame[1] + g * frame[2] + g * frame[5]) * norm,
                )
            }
            8 => {
                // FL FR FC LFE BL BR SL SR.
                let norm = 1.0 / (1.0 + g + g + g);
                (
                    (frame[0] + g * frame[2] + g * frame[4] + g * frame[6]) * norm,
                    (frame[1] + g * frame[2] + g * frame[5] + g * frame[7]) * norm,
                )
            }
            _ => unreachable!("caller checks the channel count"),
        }
    }
}

impl Default for AudioDownmix {
    fn default() -> Self {
        Self::new()
    }
}

impl Element for AudioDownmix {
    // #189: may forward the input buffer — the upstream producer's arena
    // budget accumulates through this element.
    fn passthrough(&self) -> bool {
        true
    }

    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        let Some(MediaFormat::AudioRaw(fmt)) = buffer.metadata().format.clone() else {
            return Err(Error::Config(
                "audiodownmix: buffer carries no AudioRaw format metadata \
                 (decoders set it per buffer; see MediaFormat::AudioRaw)"
                    .into(),
            ));
        };
        let ch = fmt.channels as usize;
        if ch <= 2 {
            return Ok(Some(buffer)); // passthrough, zero copy
        }
        if ch != 6 && ch != 8 {
            return Err(Error::Config(format!(
                "audiodownmix: unsupported channel count {ch} (5.1 and 7.1 are wired up)"
            )));
        }

        let data = buffer.as_bytes();
        let (frames, out_len) = match fmt.sample_format {
            SampleFormat::F32 | SampleFormat::S16 => {
                let bps = match fmt.sample_format {
                    SampleFormat::F32 => 4,
                    _ => 2,
                };
                let frames = data.len() / (ch * bps);
                (frames, frames * 2 * bps)
            }
            other => {
                return Err(Error::Config(format!(
                    "audiodownmix: unsupported sample format {other:?}"
                )));
            }
        };

        let mut slot = self.output.acquire(out_len, &self.name)?;
        {
            let out = &mut slot.data_mut()[..out_len];
            match fmt.sample_format {
                SampleFormat::F32 => {
                    for i in 0..frames {
                        let mut frame = [0f32; 8];
                        for (c, f) in frame.iter_mut().enumerate().take(ch) {
                            let o = (i * ch + c) * 4;
                            *f = f32::from_le_bytes([
                                data[o],
                                data[o + 1],
                                data[o + 2],
                                data[o + 3],
                            ]);
                        }
                        let (l, r) = Self::fold(&frame, ch);
                        out[i * 8..i * 8 + 4].copy_from_slice(&l.to_le_bytes());
                        out[i * 8 + 4..i * 8 + 8].copy_from_slice(&r.to_le_bytes());
                    }
                }
                _ => {
                    for i in 0..frames {
                        let mut frame = [0f32; 8];
                        for (c, f) in frame.iter_mut().enumerate().take(ch) {
                            let o = (i * ch + c) * 2;
                            *f = i16::from_le_bytes([data[o], data[o + 1]]) as f32;
                        }
                        let (l, r) = Self::fold(&frame, ch);
                        let l = (l.clamp(-32768.0, 32767.0)) as i16;
                        let r = (r.clamp(-32768.0, 32767.0)) as i16;
                        out[i * 4..i * 4 + 2].copy_from_slice(&l.to_le_bytes());
                        out[i * 4 + 2..i * 4 + 4].copy_from_slice(&r.to_le_bytes());
                    }
                }
            }
        }

        let mut metadata = buffer.metadata().clone();
        metadata.format = Some(MediaFormat::AudioRaw(AudioFormat::new(
            fmt.sample_rate,
            2,
            fmt.sample_format,
        )));

        Ok(Some(Buffer::new(
            MemoryHandle::with_len(slot, out_len),
            metadata,
        )))
    }

    fn flush(&mut self) -> Result<Option<Buffer>> {
        Ok(None)
    }

    fn name(&self) -> &str {
        &self.name
    }

    // True wildcards: the element adapts per buffer from AudioRaw metadata,
    // and `Caps::audio_raw_any()` is a *concrete* default format that would
    // fight a sink's pinned rate/sample-format during negotiation.
    fn input_caps(&self) -> Caps {
        Caps::any()
    }

    fn output_caps(&self) -> Caps {
        Caps::any()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;

    fn buffer_with_format(data: &[u8], rate: u32, ch: u16, sf: SampleFormat) -> Buffer {
        let arena = SharedArena::new(data.len().max(64), 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        slot.data_mut()[..data.len()].copy_from_slice(data);
        let mut m = Metadata::new();
        m.format = Some(MediaFormat::AudioRaw(AudioFormat::new(rate, ch, sf)));
        Buffer::new(MemoryHandle::with_len(slot, data.len()), m)
    }

    #[test]
    fn folds_5_1_f32_to_stereo() {
        // One frame: FL=1, FR=0.5, FC=0.4, LFE=1 (dropped), BL=0.2, BR=0.1.
        let samples: [f32; 6] = [1.0, 0.5, 0.4, 1.0, 0.2, 0.1];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let buf = buffer_with_format(&bytes, 48_000, 6, SampleFormat::F32);

        let mut dm = AudioDownmix::new();
        let out = dm.process(buf).unwrap().unwrap();

        let g = SURROUND_GAIN;
        let norm = 1.0 / (1.0 + 2.0 * g);
        let expect_l = (1.0 + g * 0.4 + g * 0.2) * norm;
        let expect_r = (0.5 + g * 0.4 + g * 0.1) * norm;

        let out_bytes = out.as_bytes();
        assert_eq!(out_bytes.len(), 8, "one stereo f32 frame");
        let l = f32::from_le_bytes(out_bytes[0..4].try_into().unwrap());
        let r = f32::from_le_bytes(out_bytes[4..8].try_into().unwrap());
        assert!((l - expect_l).abs() < 1e-6, "L {l} vs {expect_l}");
        assert!((r - expect_r).abs() < 1e-6, "R {r} vs {expect_r}");

        // Metadata rewritten to stereo.
        let Some(MediaFormat::AudioRaw(f)) = out.metadata().format.clone() else {
            panic!("AudioRaw metadata survives");
        };
        assert_eq!(f.channels, 2);
        assert_eq!(f.sample_rate, 48_000);
    }

    #[test]
    fn folds_5_1_s16_and_never_clips() {
        // Full-scale on every channel: normalization keeps it in range.
        let samples: [i16; 6] = [i16::MAX; 6];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let buf = buffer_with_format(&bytes, 48_000, 6, SampleFormat::S16);

        let mut dm = AudioDownmix::new();
        let out = dm.process(buf).unwrap().unwrap();
        let out_bytes = out.as_bytes();
        assert_eq!(out_bytes.len(), 4);
        let l = i16::from_le_bytes(out_bytes[0..2].try_into().unwrap());
        assert!(l > 0, "positive full-scale fold-down stays in range: {l}");
    }

    #[test]
    fn stereo_passes_through_untouched() {
        let samples: [f32; 4] = [0.1, 0.2, 0.3, 0.4];
        let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let buf = buffer_with_format(&bytes, 44_100, 2, SampleFormat::F32);

        let mut dm = AudioDownmix::new();
        let out = dm.process(buf).unwrap().unwrap();
        assert_eq!(out.as_bytes().len(), 16, "bit-exact passthrough");
        let Some(MediaFormat::AudioRaw(f)) = out.metadata().format.clone() else {
            panic!()
        };
        assert_eq!(f.channels, 2);
    }

    #[test]
    fn missing_format_metadata_errors() {
        let arena = SharedArena::new(64, 2).unwrap();
        let slot = arena.acquire().unwrap();
        let buf = Buffer::new(MemoryHandle::with_len(slot, 24), Metadata::new());
        assert!(AudioDownmix::new().process(buf).is_err());
    }

    #[test]
    fn odd_channel_counts_error() {
        let bytes = vec![0u8; 5 * 4];
        let buf = buffer_with_format(&bytes, 48_000, 5, SampleFormat::F32);
        assert!(AudioDownmix::new().process(buf).is_err());
    }
}
