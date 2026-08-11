//! Streaming Vorbis decoder (symphonia, pure Rust).
//!
//! Decodes raw Vorbis packets — the form [`MkvDemux`] emits them in — one at
//! a time, unlike [`SymphoniaDecoder`], which re-probes its input as a
//! *container* on every call and therefore cannot decode a demuxed
//! elementary stream at all. The audio sibling of what #68 did for AAC.
//!
//! [`MkvDemux`]: crate::elements::demux::MkvDemux
//! [`SymphoniaDecoder`]: crate::elements::codec::SymphoniaDecoder

use super::audio_traits::{AudioDecoder, AudioSampleFormat, AudioSamples};
use crate::error::{Error, Result};

use symphonia::core::codecs::audio::well_known::CODEC_ID_VORBIS;
use symphonia::core::codecs::audio::{
    AudioCodecParameters, AudioDecoder as SymphoniaAudioDecoder, AudioDecoderOptions,
};
use symphonia::core::packet::PacketRef;
use symphonia::core::units::{Duration, Timestamp};

/// Streaming Vorbis decoder implementing [`AudioDecoder`].
///
/// Initialize it from the track's CodecPrivate — Matroska stores the three
/// Xiph-laced Vorbis headers (identification, comment, setup) there, and
/// symphonia consumes that laced form directly (a bare
/// identification+setup concatenation also works). Feed it raw audio
/// packets; wrap in [`AudioDecoderElement`] for pipeline use:
///
/// ```rust,ignore
/// let private = demux.audio_track().unwrap()
///     .audio_info.as_ref().unwrap()
///     .codec_private.clone()
///     .ok_or("track has no decoder configuration")?;
/// let decoder = VorbisDecoder::from_codec_private(&private)?;
/// pipeline.add_transform("vorbisdec", AudioDecoderElement::new(decoder));
/// ```
///
/// Output is interleaved f32 at the stream's declared rate. Vorbis frames
/// are variable-sized (the first packet after a seek decodes to zero
/// samples while the MDCT window primes — such packets yield an empty
/// `AudioSamples`, which [`AudioDecoderElement`] drops).
///
/// [`AudioDecoderElement`]: crate::elements::codec::AudioDecoderElement
pub struct VorbisDecoder {
    inner: Box<dyn SymphoniaAudioDecoder>,
    sample_rate: u32,
    channels: u32,
    /// Scratch for the interleaved f32 samples of one decoded frame.
    samples: Vec<f32>,
    /// Spent output Vec handed back by the element wrapper (#143).
    recycled: Vec<u8>,
    packets_in: u64,
}

impl VorbisDecoder {
    /// Create a decoder from the track's CodecPrivate bytes (Xiph-laced
    /// header triple as stored by Matroska, or identification+setup
    /// concatenated).
    pub fn from_codec_private(private: &[u8]) -> Result<Self> {
        let mut params = AudioCodecParameters::new();
        params.codec = CODEC_ID_VORBIS;
        params.extra_data = Some(private.into());

        let inner = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(|e| Error::Config(format!("Vorbis decoder init failed: {e}")))?;

        // symphonia's Vorbis decoder does not amend its parameters with the
        // identification header (unlike its AAC decoder), so read the rate
        // and channel count out of the header ourselves: `\x01vorbis`,
        // version u32, channels u8, sample rate u32 LE (Vorbis I §4.2.2).
        let ident = private
            .windows(7)
            .position(|w| w == b"\x01vorbis")
            .map(|p| &private[p..])
            .filter(|h| h.len() >= 16)
            .ok_or_else(|| {
                Error::Config("CodecPrivate holds no Vorbis identification header".into())
            })?;
        let channels = ident[11] as u32;
        let sample_rate = u32::from_le_bytes([ident[12], ident[13], ident[14], ident[15]]);
        if sample_rate == 0 || channels == 0 {
            return Err(Error::Config(
                "Vorbis identification header declares no sample rate or channels".into(),
            ));
        }

        Ok(Self {
            inner,
            sample_rate,
            channels,
            samples: Vec::new(),
            recycled: Vec::new(),
            packets_in: 0,
        })
    }

    /// Number of packets decoded so far.
    pub fn packets_in(&self) -> u64 {
        self.packets_in
    }
}

impl AudioDecoder for VorbisDecoder {
    fn decode(&mut self, packet: &[u8]) -> Result<AudioSamples> {
        let packet_ref = PacketRef::new(0, Timestamp::ZERO, Duration::ZERO, packet);
        let decoded = self
            .inner
            .decode_ref(&packet_ref)
            .map_err(|e| Error::Element(format!("Vorbis decode error: {e}")))?;

        let spec = decoded.spec();
        self.sample_rate = spec.rate();
        self.channels = spec.channels().count() as u32;
        let frames = decoded.frames();

        decoded.copy_to_vec_interleaved(&mut self.samples);
        // Pack into the recycled Vec with an exact reserve — the flat_map
        // collect this replaces grew by repeated realloc (#143).
        let mut data = std::mem::take(&mut self.recycled);
        data.clear();
        data.reserve(self.samples.len() * 4);
        for s in &self.samples {
            data.extend_from_slice(&s.to_le_bytes());
        }

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
        // Reset the MDCT window state so the decoder restarts cleanly after
        // a flushing seek (AudioDecoderElement routes seeks here).
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

    fn recycle(&mut self, mut data: Vec<u8>) {
        data.clear();
        self.recycled = data;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_garbage_codec_private() {
        assert!(VorbisDecoder::from_codec_private(&[]).is_err());
        assert!(VorbisDecoder::from_codec_private(&[0x02, 0xFF, 0xFF]).is_err());
        assert!(VorbisDecoder::from_codec_private(&[0xAA; 64]).is_err());
    }
}
