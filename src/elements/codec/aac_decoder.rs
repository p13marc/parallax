//! Streaming AAC decoder (symphonia, pure Rust).
//!
//! Decodes raw AAC access units — the form [`Mp4Demux`] emits them in — one
//! packet at a time, unlike [`SymphoniaDecoder`], which re-probes its input as
//! a *container* on every call and therefore cannot decode a demuxed
//! elementary stream at all.
//!
//! [`Mp4Demux`]: crate::elements::demux::Mp4Demux
//! [`SymphoniaDecoder`]: crate::elements::codec::SymphoniaDecoder

use super::audio_traits::{AudioDecoder, AudioSampleFormat, AudioSamples};
use crate::error::{Error, Result};

use symphonia::core::codecs::audio::well_known::CODEC_ID_AAC;
use symphonia::core::codecs::audio::{
    AudioCodecParameters, AudioDecoder as SymphoniaAudioDecoder, AudioDecoderOptions,
};
use symphonia::core::packet::PacketRef;
use symphonia::core::units::{Duration, Timestamp};

/// Streaming AAC decoder implementing [`AudioDecoder`].
///
/// Initialize it from the `AudioSpecificConfig` a demuxer provides — for MP4,
/// [`Mp4AudioInfo::audio_specific_config`] — and feed it raw access units.
/// Wrap it in [`AudioDecoderElement`] to use it as a pipeline element:
///
/// ```rust,ignore
/// let asc = demux.track(audio_id).unwrap()
///     .audio_info.as_ref().unwrap()
///     .audio_specific_config.clone()
///     .ok_or("track has no decoder configuration")?;
/// let decoder = AacDecoder::from_asc(&asc)?;
/// pipeline.add_transform("aacdec", AudioDecoderElement::new(decoder));
/// ```
///
/// Scope matches symphonia's AAC support: AAC-LC, mono or stereo, 1024-sample
/// frames. HE-AAC (SBR) and multichannel configurations error at
/// construction.
///
/// [`Mp4AudioInfo::audio_specific_config`]: crate::elements::demux::Mp4AudioInfo::audio_specific_config
/// [`AudioDecoderElement`]: crate::elements::codec::AudioDecoderElement
pub struct AacDecoder {
    inner: Box<dyn SymphoniaAudioDecoder>,
    sample_rate: u32,
    channels: u32,
    /// Scratch for the interleaved f32 samples of one decoded frame.
    samples: Vec<f32>,
    packets_in: u64,
}

impl AacDecoder {
    /// Create a decoder from `AudioSpecificConfig` bytes.
    ///
    /// This is the out-of-band configuration MP4 carries in the esds box —
    /// see [`Mp4AudioInfo::audio_specific_config`].
    ///
    /// [`Mp4AudioInfo::audio_specific_config`]: crate::elements::demux::Mp4AudioInfo::audio_specific_config
    pub fn from_asc(asc: &[u8]) -> Result<Self> {
        let mut params = AudioCodecParameters::new();
        params.codec = CODEC_ID_AAC;
        params.extra_data = Some(asc.into());
        Self::from_params(params)
    }

    fn from_params(params: AudioCodecParameters) -> Result<Self> {
        let inner = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(|e| Error::Config(format!("AAC decoder init failed: {e}")))?;

        // The symphonia decoder amends its parameters with what the ASC
        // declared; that is the authoritative output format until the first
        // frame confirms it.
        let amended = inner.codec_params();
        let sample_rate = amended.sample_rate.unwrap_or(0);
        let channels = amended
            .channels
            .as_ref()
            .map(|c| c.count() as u32)
            .unwrap_or(0);
        if sample_rate == 0 || channels == 0 {
            return Err(Error::Config(
                "AAC decoder configuration declares no sample rate or channels".into(),
            ));
        }

        Ok(Self {
            inner,
            sample_rate,
            channels,
            samples: Vec::new(),
            packets_in: 0,
        })
    }

    /// Number of packets decoded so far.
    pub fn packets_in(&self) -> u64 {
        self.packets_in
    }
}

impl AudioDecoder for AacDecoder {
    fn decode(&mut self, packet: &[u8]) -> Result<AudioSamples> {
        let packet_ref = PacketRef::new(0, Timestamp::ZERO, Duration::ZERO, packet);
        let decoded = self
            .inner
            .decode_ref(&packet_ref)
            .map_err(|e| Error::Element(format!("AAC decode error: {e}")))?;

        let spec = decoded.spec();
        self.sample_rate = spec.rate();
        self.channels = spec.channels().count() as u32;
        let frames = decoded.frames();

        decoded.copy_to_vec_interleaved(&mut self.samples);
        let data: Vec<u8> = self.samples.iter().flat_map(|s| s.to_le_bytes()).collect();

        self.packets_in += 1;

        Ok(AudioSamples {
            data,
            format: AudioSampleFormat::F32,
            channels: self.channels,
            sample_rate: self.sample_rate,
            samples_per_channel: frames,
            pts: 0, // Set by the element wrapper.
        })
    }

    fn flush(&mut self) -> Result<Option<AudioSamples>> {
        // AAC has no decoder delay symphonia surfaces at end of stream;
        // reset the channel state so the decoder can restart cleanly (a
        // flushing seek lands here via AudioDecoderElement).
        self.inner.reset();
        Ok(None)
    }

    fn output_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn output_channels(&self) -> u32 {
        self.channels
    }

    fn output_format(&self) -> AudioSampleFormat {
        AudioSampleFormat::F32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // AudioSpecificConfig for AAC-LC, 48 kHz, stereo — what Mp4Demux
    // synthesizes for the common case.
    const ASC_LC_48K_STEREO: [u8; 2] = [0x11, 0x90];
    // Same, mono.
    const ASC_LC_48K_MONO: [u8; 2] = [0x11, 0x88];
    // A hand-built silent AAC-LC frame for the mono configuration: an SCE
    // whose fields are all zero (ONLY_LONG window, max_sfb 0, no tools) is
    // 29 zero bits, then the END element (0b111) pads the fourth byte.
    const SILENT_MONO_FRAME: [u8; 4] = [0x00, 0x00, 0x00, 0x07];

    #[test]
    fn initializes_from_an_asc() {
        let dec = AacDecoder::from_asc(&ASC_LC_48K_STEREO).unwrap();
        assert_eq!(dec.output_sample_rate(), 48000);
        assert_eq!(dec.output_channels(), 2);
        assert_eq!(dec.output_format(), AudioSampleFormat::F32);
    }

    #[test]
    fn rejects_configurations_symphonia_cannot_decode() {
        // HE-AAC v1 (SBR object type 5, 24 kHz core): too complex for
        // symphonia's LC-only decoder — must error at construction, not
        // produce garbage per packet.
        let he_aac = [0x2B, 0x11];
        assert!(AacDecoder::from_asc(&he_aac).is_err());

        // Truncated/empty ASC.
        assert!(AacDecoder::from_asc(&[0x11]).is_err());
        assert!(AacDecoder::from_asc(&[]).is_err());
    }

    #[test]
    fn decodes_a_frame_and_reports_the_stream_format() {
        let mut dec = AacDecoder::from_asc(&ASC_LC_48K_MONO).unwrap();
        let out = dec.decode(&SILENT_MONO_FRAME).unwrap();
        assert_eq!(out.sample_rate, 48000);
        assert_eq!(out.channels, 1);
        assert_eq!(out.format, AudioSampleFormat::F32);
        assert_eq!(out.samples_per_channel, 1024, "one AAC-LC frame");
        assert_eq!(
            out.data.len(),
            out.samples_per_channel * out.channels as usize * 4,
            "interleaved f32 bytes"
        );
        assert!(out.data.iter().all(|b| *b == 0), "silence decodes to zeros");
        assert_eq!(dec.packets_in(), 1);
    }

    #[test]
    fn a_failed_packet_does_not_kill_the_decoder() {
        let mut dec = AacDecoder::from_asc(&ASC_LC_48K_MONO).unwrap();
        // Junk may or may not trip the parser (0xFF... reads as an immediate
        // END element and "decodes"); either way the decoder must keep
        // working for the next valid packet — per-packet errors are the
        // executor's shed path, not a terminal state.
        let _ = dec.decode(&[0xFFu8; 16]);
        let _ = dec.decode(&[0x55u8; 7]);
        let out = dec.decode(&SILENT_MONO_FRAME).unwrap();
        assert_eq!(out.samples_per_channel, 1024);
    }
}
