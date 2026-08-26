//! Generic audio decoder element wrapper.
//!
//! This module provides [`AudioDecoderElement`], a wrapper that adapts any
//! [`AudioDecoder`] to work as a pipeline element.
//!
//! # Features
//!
//! - Handles decoding packets to PCM samples
//! - Automatic flush at end-of-stream
//! - Timestamp preservation
//! - Statistics tracking
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::elements::codec::{AudioDecoderElement, OpusDecoder};
//!
//! let decoder = OpusDecoder::new(48000, 2)?;
//! let element = AudioDecoderElement::new(decoder);
//!
//! pipeline.add_node("decoder", DynAsyncElement::new_box(TransformAdapter::new(element)));
//! ```

use super::audio_traits::{AudioDecoder, AudioSamples};
use crate::buffer::{Buffer, MemoryHandle};
use crate::clock::ClockTime;
use crate::element::{ExecutionHints, Output, Transform};
use crate::error::Result;
use crate::memory::{OutputArena, OutputBudget, defaults};

/// Wraps an [`AudioDecoder`] to work as a pipeline [`Transform`] element.
///
/// This wrapper handles:
/// - Converting input buffers (compressed packets) to decoded PCM
/// - Managing decoder state
/// - Flushing at EOS
/// - Preserving timestamps
pub struct AudioDecoderElement<D: AudioDecoder> {
    /// The underlying decoder.
    decoder: D,
    /// Whether we've started flushing.
    flushing: bool,
    /// Whether flush is complete.
    flushed: bool,
    /// Statistics: packets received.
    packets_in: u64,
    /// Statistics: frames produced.
    frames_out: u64,
    /// Where the next frame starts when the input carries no PTS of its
    /// own. `NONE` until a stamped buffer has been seen — a flush resets it,
    /// because the position it continued from was abandoned.
    current_pts: ClockTime,
    /// Arena for output buffer allocation.
    output: OutputArena,
}

impl<D: AudioDecoder> AudioDecoderElement<D> {
    /// Create a new audio decoder element wrapper.
    ///
    /// # Arguments
    ///
    /// * `decoder` - The audio decoder to wrap
    pub fn new(decoder: D) -> Self {
        Self {
            decoder,
            flushing: false,
            flushed: false,
            packets_in: 0,
            frames_out: 0,
            current_pts: ClockTime::NONE,
            output: OutputArena::new(defaults::AUDIO_SLOT_COUNT).with_min_slot_size(64 * 1024),
        }
    }

    /// Get the number of packets received.
    pub fn packets_in(&self) -> u64 {
        self.packets_in
    }

    /// Get the number of frames produced.
    pub fn frames_out(&self) -> u64 {
        self.frames_out
    }

    /// Get a reference to the inner decoder.
    pub fn decoder(&self) -> &D {
        &self.decoder
    }

    /// Get a mutable reference to the inner decoder.
    pub fn decoder_mut(&mut self) -> &mut D {
        &mut self.decoder
    }

    /// Widen the arena when a decoded frame outgrows its slots.
    ///
    /// Slot size is fixed at build time, so a longer packet than anything seen
    /// so far needs a rebuild.
    fn ensure_arena(&mut self, min_size: usize) {
        if self
            .output
            .arena()
            .is_some_and(|a| a.slot_size() < min_size)
        {
            self.output.reset();
        }
        self.output.set_min_slot_size(min_size.max(64 * 1024));
    }
}

/// The in-band description of decoded PCM, from the decoder's own account of
/// what it produced.
fn pcm_media_format(samples: &AudioSamples) -> crate::format::MediaFormat {
    use super::audio_traits::AudioSampleFormat;
    crate::format::MediaFormat::AudioRaw(crate::format::AudioFormat::new(
        samples.sample_rate,
        samples.channels as u16,
        match samples.format {
            AudioSampleFormat::S16 => crate::format::SampleFormat::S16,
            AudioSampleFormat::S32 => crate::format::SampleFormat::S32,
            AudioSampleFormat::F32 => crate::format::SampleFormat::F32,
        },
    ))
}

impl<D: AudioDecoder + 'static> Transform for AudioDecoderElement<D> {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);
    }

    fn transform(&mut self, buffer: Buffer) -> Result<Output> {
        let packet = buffer.as_bytes();
        // `to_option`, not `!= 0`: a buffer legitimately stamped at t=0 was
        // being treated as unstamped and given the running counter's value
        // instead — which after a seek is a stale position from before it.
        let input_pts = buffer.metadata().pts.to_option();
        self.packets_in += 1;

        // Decode packet
        let mut samples = self.decoder.decode(packet)?;

        // A packet can legitimately decode to nothing (Vorbis primes its
        // MDCT window on the first packet after init/seek) — emitting an
        // empty buffer downstream helps nobody.
        if samples.samples_per_channel == 0 {
            return Ok(Output::None);
        }

        // Set timestamp: the input's own when it has one, else continue from
        // where the last frame ended.
        let pts = input_pts.or(self.current_pts.to_option());
        samples.pts = pts.map(|p| p.nanos() as i64).unwrap_or(0);

        // Update PTS for next frame. Unknown stays unknown: inventing a
        // zero here would hand the next unstamped packet a timeline this
        // element made up.
        self.current_pts = pts
            .map(|p| p.saturating_add(ClockTime::from_nanos(samples.duration_nanos())))
            .unwrap_or(ClockTime::NONE);
        self.frames_out += 1;

        // Ensure arena is large enough
        self.ensure_arena(samples.data.len());
        let mut slot = self
            .output
            .acquire(samples.data.len(), "audiodecoderelement")?;
        slot.data_mut()[..samples.data.len()].copy_from_slice(&samples.data);
        // Hand the spent Vec back so the decoder reuses it next packet (#143).
        let data_len = samples.data.len();
        self.decoder.recycle(std::mem::take(&mut samples.data));

        // Preserve input metadata and update PTS/duration
        let mut metadata = buffer.metadata().clone();
        metadata.pts = pts.unwrap_or(ClockTime::NONE);
        metadata.duration = ClockTime::from_nanos(samples.duration_nanos());

        // Describe the PCM in-band so downstream (audioconvert, sinks) can
        // configure from the buffer instead of out-of-band knowledge (#68).
        metadata.format = Some(pcm_media_format(&samples));

        Ok(Output::single(Buffer::new(
            MemoryHandle::with_len(slot, data_len),
            metadata,
        )))
    }

    fn flush(&mut self) -> Result<Output> {
        if self.flushed {
            return Ok(Output::None);
        }

        if !self.flushing {
            self.flushing = true;
        }

        // Try to flush decoder
        match self.decoder.flush()? {
            Some(samples) => {
                self.frames_out += 1;

                self.ensure_arena(samples.data.len());
                let mut slot = self
                    .output
                    .acquire(samples.data.len(), "audiodecoderelement")?;
                slot.data_mut()[..samples.data.len()].copy_from_slice(&samples.data);

                let mut metadata = crate::metadata::Metadata::new();
                metadata.pts = self.current_pts;
                metadata.format = Some(pcm_media_format(&samples));

                Ok(Output::single(Buffer::new(
                    MemoryHandle::with_len(slot, samples.data.len()),
                    metadata,
                )))
            }
            None => {
                self.flushed = true;
                Ok(Output::None)
            }
        }
    }

    /// A flushing seek abandons everything the decoder is carrying (#265).
    ///
    /// Three things had to happen here and none of them did, because the
    /// wrapper implemented no event handler at all:
    ///
    /// - the wrapped decoder keeps inter-packet state (Opus's overlap,
    ///   Vorbis's MDCT priming) that belongs to the position being left;
    /// - `current_pts` is a running counter, and continuing it across a seek
    ///   stamps the new position with the old one's timeline;
    /// - `flushed` is a one-shot latch, and the executor calls `flush()` on
    ///   every `FlushStart` with the output *discarded* — so the first seek
    ///   of a run consumed the drain and the real EOS tail was lost. Exactly
    ///   the per-cycle latch `H264Decoder` documents; the audio side never
    ///   got it.
    fn handle_downstream_event(
        &mut self,
        event: crate::event::Event,
    ) -> Option<crate::event::Event> {
        if matches!(event, crate::event::Event::FlushStart) {
            self.decoder.reset();
            self.current_pts = ClockTime::NONE;
            self.flushing = false;
            self.flushed = false;
        }
        Some(event)
    }

    fn name(&self) -> &str {
        "AudioDecoderElement"
    }

    fn execution_hints(&self) -> ExecutionHints {
        // Audio decoders typically use native code (libopus, etc.)
        ExecutionHints::native()
    }
}

impl<D: AudioDecoder> std::fmt::Debug for AudioDecoderElement<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioDecoderElement")
            .field("packets_in", &self.packets_in)
            .field("frames_out", &self.frames_out)
            .field("flushing", &self.flushing)
            .field("flushed", &self.flushed)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::super::audio_traits::AudioSampleFormat;
    use super::*;
    use crate::event::Event;
    use crate::metadata::Metadata;

    /// A decoder whose whole job is to report what the wrapper did to it:
    /// one 10 ms stereo S16 frame per packet, a countable `reset`, and a
    /// one-frame EOS tail so the drain is observable.
    #[derive(Default)]
    struct SpyDecoder {
        resets: usize,
        drained: bool,
    }

    const RATE: u32 = 48_000;
    const FRAMES: usize = 480; // 10 ms

    fn frame() -> AudioSamples {
        AudioSamples {
            data: vec![0u8; FRAMES * 2 * 2],
            format: AudioSampleFormat::S16,
            channels: 2,
            sample_rate: RATE,
            samples_per_channel: FRAMES,
            pts: 0,
        }
    }

    impl AudioDecoder for SpyDecoder {
        fn decode(&mut self, _packet: &[u8]) -> Result<AudioSamples> {
            Ok(frame())
        }
        fn flush(&mut self) -> Result<Option<AudioSamples>> {
            if self.drained {
                return Ok(None);
            }
            self.drained = true;
            Ok(Some(frame()))
        }
        fn reset(&mut self) {
            self.resets += 1;
            self.drained = false;
        }
        fn output_sample_rate(&self) -> u32 {
            RATE
        }
        fn output_channels(&self) -> u32 {
            2
        }
        fn output_format(&self) -> AudioSampleFormat {
            AudioSampleFormat::S16
        }
    }

    fn packet(pts: Option<ClockTime>) -> Buffer {
        let arena = crate::memory::SharedArena::new(64, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let mut meta = Metadata::new();
        meta.pts = pts.unwrap_or(ClockTime::NONE);
        Buffer::new(MemoryHandle::with_len(slot, 8), meta)
    }

    fn pts_of(out: &Output) -> ClockTime {
        match out {
            Output::Single(b) => b.metadata().pts,
            other => panic!("expected one buffer, got {other:?}"),
        }
    }

    #[test]
    fn a_packet_stamped_at_zero_keeps_its_own_timestamp() {
        // `pts != 0` used to mean "unstamped", so a buffer legitimately at
        // t=0 was given the running counter instead — which after a seek is
        // a position from before it.
        let mut el = AudioDecoderElement::new(SpyDecoder::default());
        let _ = el
            .transform(packet(Some(ClockTime::from_secs(30))))
            .unwrap();
        let out = el.transform(packet(Some(ClockTime::ZERO))).unwrap();
        assert_eq!(pts_of(&out), ClockTime::ZERO);
    }

    #[test]
    fn an_unstamped_packet_continues_from_the_last_frame() {
        let mut el = AudioDecoderElement::new(SpyDecoder::default());
        let _ = el.transform(packet(Some(ClockTime::from_secs(1)))).unwrap();
        let out = el.transform(packet(None)).unwrap();
        assert_eq!(
            pts_of(&out),
            ClockTime::from_secs(1).saturating_add(ClockTime::from_millis(10))
        );
    }

    #[test]
    fn a_flush_resets_the_decoder_every_time_not_just_once() {
        let mut el = AudioDecoderElement::new(SpyDecoder::default());
        for _ in 0..3 {
            let _ = el.flush();
            el.handle_downstream_event(Event::FlushStart);
        }
        assert_eq!(el.decoder.resets, 3);
    }

    #[test]
    fn a_seek_does_not_eat_the_eos_drain() {
        // The executor calls `flush()` on FlushStart and discards it. The
        // `flushed` latch used to survive that, so after the first seek of a
        // run the real EOS tail was silently dropped.
        let mut el = AudioDecoderElement::new(SpyDecoder::default());
        let _ = el.transform(packet(Some(ClockTime::ZERO))).unwrap();

        // The seek's discarded drain.
        assert!(matches!(el.flush().unwrap(), Output::Single(_)));
        el.handle_downstream_event(Event::FlushStart);

        // The real one, after fresh data.
        let _ = el
            .transform(packet(Some(ClockTime::from_secs(30))))
            .unwrap();
        assert!(
            matches!(el.flush().unwrap(), Output::Single(_)),
            "EOS drain was consumed by the seek's flush"
        );
        assert!(matches!(el.flush().unwrap(), Output::None));
    }

    #[test]
    fn a_flush_forgets_the_running_timestamp() {
        let mut el = AudioDecoderElement::new(SpyDecoder::default());
        let _ = el
            .transform(packet(Some(ClockTime::from_secs(30))))
            .unwrap();
        el.handle_downstream_event(Event::FlushStart);
        // Unstamped after the flush: the position it would have continued
        // from was abandoned by the seek, so the honest answer is "unknown",
        // not "30 s and a bit".
        let out = el.transform(packet(None)).unwrap();
        assert_eq!(pts_of(&out), ClockTime::NONE);
    }
}
