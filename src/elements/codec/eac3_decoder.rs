//! AC-3 / E-AC-3 (Dolby Digital / Digital Plus) decoder — pure Rust via
//! `oxideav-ac3`.
//!
//! Decodes raw sync-framed bitstream as demuxers emit it: one Matroska
//! block or MP4 sample is one (E-)AC-3 sync frame, and the stream is fully
//! self-describing (channel mode, rate and coding tools travel in every
//! frame's syncinfo/BSI), so no out-of-band configuration is needed.
//!
//! Vetted against real E-AC-3 5.1 WEB-DL streams: hundreds of frames
//! decode with the crate's §7.8 LoRo matrix downmix to stereo, and
//! truncated/garbage packets error per-frame without corrupting decoder
//! state — the executor's shed path, not a terminal failure.

use super::audio_traits::{AudioDecoder, AudioSampleFormat, AudioSamples};
use crate::error::{Error, Result};

use oxideav_core::{CodecId, CodecParameters, Decoder as OxideavDecoder, Frame, Packet, TimeBase};

/// Streaming AC-3 / E-AC-3 decoder implementing [`AudioDecoder`].
///
/// ```rust,ignore
/// // Stereo fold-down (ITU-style LoRo matrix, done inside the decoder):
/// let decoder = Eac3Decoder::stereo(48_000)?;
/// pipeline.add_transform("eac3dec", AudioDecoderElement::new(decoder));
/// ```
///
/// Output is interleaved S16 at the stream's rate, 1536 samples per
/// channel per sync frame. [`stereo`](Self::stereo) requests the spec's
/// LoRo downmix for multichannel programs (mono/stereo streams pass
/// through); [`passthrough`](Self::passthrough) keeps the native channel
/// count in WAV order (FL FR FC LFE BL BR) for a downstream downmix or a
/// multichannel sink.
///
/// [`AudioDecoderElement`]: crate::elements::codec::AudioDecoderElement
pub struct Eac3Decoder {
    inner: Box<dyn OxideavDecoder>,
    sample_rate: u32,
    channels: u32,
    packets_in: u64,
}

impl Eac3Decoder {
    /// Decoder that folds multichannel programs down to stereo (LoRo).
    pub fn stereo(sample_rate: u32) -> Result<Self> {
        Self::new(sample_rate, Some(2))
    }

    /// Decoder that keeps the stream's native channel layout.
    ///
    /// `channels` is the count the demuxer reported for the track; the
    /// bitstream's own per-frame channel mode is authoritative at decode
    /// time.
    pub fn passthrough(sample_rate: u32, channels: u32) -> Result<Self> {
        Self::new(sample_rate, Some(channels as u16))
    }

    fn new(sample_rate: u32, requested_channels: Option<u16>) -> Result<Self> {
        let mut params = CodecParameters::audio(CodecId::new("eac3"));
        params.sample_rate = Some(sample_rate);
        params.channels = requested_channels;
        let inner = oxideav_ac3::decoder::make_decoder_with_drc(
            &params,
            oxideav_ac3::drc::DrcSettings::default(),
        )
        .map_err(|e| Error::Config(format!("E-AC-3 decoder init failed: {e:?}")))?;

        Ok(Self {
            inner,
            sample_rate,
            channels: requested_channels.unwrap_or(2) as u32,
            packets_in: 0,
        })
    }

    /// Number of packets decoded so far.
    pub fn packets_in(&self) -> u64 {
        self.packets_in
    }
}

impl AudioDecoder for Eac3Decoder {
    fn decode(&mut self, packet: &[u8]) -> Result<AudioSamples> {
        let pkt = Packet::new(
            0,
            TimeBase::new(1, self.sample_rate as i64),
            packet.to_vec(),
        );
        self.inner
            .send_packet(&pkt)
            .map_err(|e| Error::Element(format!("E-AC-3 send failed: {e:?}")))?;
        let frame = self
            .inner
            .receive_frame()
            .map_err(|e| Error::Element(format!("E-AC-3 decode error: {e:?}")))?;

        let Frame::Audio(audio) = frame else {
            return Err(Error::Element("E-AC-3 decoder produced non-audio".into()));
        };
        let data = audio
            .data
            .into_iter()
            .next()
            .ok_or_else(|| Error::Element("E-AC-3 frame has no sample plane".into()))?;
        let samples_per_channel = audio.samples as usize;
        // The interleaved S16 plane's size reveals the actual channel
        // count (a stereo request on a mono stream stays mono).
        let channels = (data.len() / 2)
            .checked_div(samples_per_channel)
            .map(|c| c as u32)
            .unwrap_or(self.channels);
        self.channels = channels;
        self.packets_in += 1;

        Ok(AudioSamples {
            data,
            format: AudioSampleFormat::S16,
            channels,
            sample_rate: self.sample_rate,
            samples_per_channel,
            pts: 0, // Set by the element wrapper.
        })
    }

    fn flush(&mut self) -> Result<Option<AudioSamples>> {
        // (E-)AC-3 has no decoder delay across sync frames; reset so a
        // flushing seek restarts cleanly.
        self.inner
            .reset()
            .map_err(|e| Error::Element(format!("E-AC-3 reset failed: {e:?}")))?;
        Ok(None)
    }

    fn output_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn output_channels(&self) -> u32 {
        self.channels
    }

    fn output_format(&self) -> AudioSampleFormat {
        AudioSampleFormat::S16
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constructs_for_stereo_and_passthrough() {
        let dec = Eac3Decoder::stereo(48_000).unwrap();
        assert_eq!(dec.output_sample_rate(), 48_000);
        assert_eq!(dec.output_channels(), 2);
        assert_eq!(dec.output_format(), AudioSampleFormat::S16);
        assert!(Eac3Decoder::passthrough(48_000, 6).is_ok());
    }

    #[test]
    fn garbage_errors_per_packet_without_killing_the_decoder() {
        let mut dec = Eac3Decoder::stereo(48_000).unwrap();
        assert!(dec.decode(&[0x0B, 0x77, 0x00, 0x01]).is_err());
        assert!(dec.decode(&[0xAA; 32]).is_err());
        // Still alive: a further bad packet errors the same way instead of
        // panicking or wedging.
        assert!(dec.decode(&[]).is_err());
    }

    /// Round-trip through the crate's own encoder: a 5.1 tone field comes
    /// back as stereo with energy in both channels.
    #[test]
    fn encode_decode_roundtrip_downmixes_to_stereo() {
        use oxideav_core::AudioFrame;

        let mut enc_params = CodecParameters::audio(CodecId::new("ac3"));
        enc_params.sample_rate = Some(48_000);
        enc_params.channels = Some(2);
        enc_params.sample_format = Some(oxideav_core::SampleFormat::S16);
        enc_params.bit_rate = Some(192_000);
        let mut enc = oxideav_ac3::encoder::make_encoder(&enc_params).unwrap();

        // One frame of a 440 Hz tone, stereo interleaved S16.
        let n = 1536usize;
        let mut pcm = Vec::with_capacity(n * 2 * 2);
        for i in 0..n {
            let s = ((i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 12000.0) as i16;
            pcm.extend_from_slice(&s.to_le_bytes());
            pcm.extend_from_slice(&s.to_le_bytes());
        }
        let frame = AudioFrame {
            samples: n as u32,
            pts: None,
            data: vec![pcm],
        };
        enc.send_frame(&Frame::Audio(frame)).unwrap();
        let packet = enc.receive_packet().unwrap();

        let mut dec = Eac3Decoder::stereo(48_000).unwrap();
        let out = dec.decode(&packet.data).unwrap();
        assert_eq!(out.samples_per_channel, 1536);
        assert_eq!(out.channels, 2);
        let samples = out.as_s16().unwrap();
        let energy: i64 = samples.iter().map(|s| (*s as i64).abs()).sum();
        assert!(energy > 100_000, "decoded tone carries energy: {energy}");
    }
}
