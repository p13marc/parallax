//! Opening a file, building its pipeline, and taking it down again.
//!
//! The interesting part is that this can happen more than once. The old
//! player looped inside the demuxer because winit allows one event loop per
//! process, ever — `AutoVideoSink` owned it, so the pipeline could never be
//! rebuilt without losing the window. With Iced owning the loop there is no
//! window in the pipeline at all, and an [`Engine`] is ordinary disposable
//! state: open a file, tear it down, open another.

use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, bail};

use parallax::clock::ClockTime;
use parallax::elements::transform::AudioDownmix;
use parallax::elements::{
    AacDecoder, AlsaFormat, AlsaSampleFormat, AlsaSink, AppVideoSink, AudioDecoderElement,
    Dav1dDecoder, Eac3Decoder, Gain, GainControl, H264Decoder, MkvCodec, MkvDemux, MkvTrackType,
    Mp4Codec, Mp4Demux, Mp4DemuxSource, Mp4TrackType, OpusDecoder, VorbisDecoder, VpxDecoder,
};
use parallax::pipeline::typefind::{MediaType, TypeFindRegistry};
use parallax::pipeline::{Bus, Executor, ExecutorConfig, LinkPolicy, Pipeline, PipelineHandle};
use parallax::video::FrameMailbox;

/// Whether to attempt hardware decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum HwDec {
    /// Try hardware, fall back to software.
    Auto,
    /// Software only.
    Off,
}

/// How to build a pipeline, from the command line.
#[derive(Debug, Clone)]
pub struct Options {
    /// Try the GPU video engine for the video track.
    pub hwdec: HwDec,
    /// Play the audio track, when there is one.
    pub audio: bool,
    /// Loop inside the demuxer — gapless, and cheaper than a rebuild.
    ///
    /// Named for the mechanism rather than the intent, because it is not the
    /// playlist's repeat: it suppresses end-of-stream entirely, so a player
    /// built with it never learns that a file finished and never advances.
    pub gapless_loop: bool,
    /// Finished frames the decoder may run ahead of the screen.
    pub decode_ahead: usize,
    /// dav1d thread count; 0 picks by resolution.
    pub dav1d_threads: u32,
    /// dav1d frame-delay bound; 0 leaves the decoder's own choice.
    pub frame_delay: u32,
    /// Executor link capacity for edges that do not set their own.
    pub channel_capacity: usize,
    /// Present frames at their due time. Off is the batch-decode floor — the
    /// pipeline runs as fast as it can, which is how #192 was measured.
    pub sync: bool,
    /// ALSA device to play through. `"default"` is whatever ALSA is
    /// configured to mean, which is usually dmix or PipeWire's bridge.
    pub audio_device: String,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            hwdec: HwDec::Auto,
            audio: true,
            gapless_loop: false,
            decode_ahead: 4,
            dav1d_threads: 0,
            frame_delay: 0,
            channel_capacity: 4,
            sync: true,
            audio_device: "default".into(),
        }
    }
}

/// Video codec of the selected track, container-agnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VideoCodecKind {
    /// H.264 / AVC.
    H264,
    /// H.265 / HEVC — recognised, and not playable here.
    H265,
    /// VP8.
    Vp8,
    /// VP9.
    Vp9,
    /// AV1.
    Av1,
    /// Something with no decoder in this tree.
    Other(String),
}

impl std::fmt::Display for VideoCodecKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoCodecKind::H264 => write!(f, "H.264"),
            VideoCodecKind::H265 => write!(f, "H.265"),
            VideoCodecKind::Vp8 => write!(f, "VP8"),
            VideoCodecKind::Vp9 => write!(f, "VP9"),
            VideoCodecKind::Av1 => write!(f, "AV1"),
            VideoCodecKind::Other(s) => write!(f, "{s}"),
        }
    }
}

/// Audio codec of the selected track, container-agnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AudioCodecKind {
    /// AAC-LC.
    Aac,
    /// Opus.
    Opus,
    /// Vorbis.
    Vorbis,
    /// AC-3 / E-AC-3.
    Eac3,
    /// Something with no decoder in this tree.
    Other(String),
}

impl std::fmt::Display for AudioCodecKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AudioCodecKind::Aac => write!(f, "AAC"),
            AudioCodecKind::Opus => write!(f, "Opus"),
            AudioCodecKind::Vorbis => write!(f, "Vorbis"),
            AudioCodecKind::Eac3 => write!(f, "E-AC-3"),
            AudioCodecKind::Other(s) => write!(f, "{s}"),
        }
    }
}

impl From<Mp4Codec> for VideoCodecKind {
    fn from(c: Mp4Codec) -> Self {
        match c {
            Mp4Codec::H264 => VideoCodecKind::H264,
            Mp4Codec::H265 => VideoCodecKind::H265,
            Mp4Codec::Vp9 => VideoCodecKind::Vp9,
            Mp4Codec::Av1 => VideoCodecKind::Av1,
            other => VideoCodecKind::Other(other.to_string()),
        }
    }
}

impl From<&MkvCodec> for VideoCodecKind {
    fn from(c: &MkvCodec) -> Self {
        match c {
            MkvCodec::H264 => VideoCodecKind::H264,
            MkvCodec::H265 => VideoCodecKind::H265,
            MkvCodec::Vp8 => VideoCodecKind::Vp8,
            MkvCodec::Vp9 => VideoCodecKind::Vp9,
            MkvCodec::Av1 => VideoCodecKind::Av1,
            other => VideoCodecKind::Other(other.to_string()),
        }
    }
}

impl From<&MkvCodec> for AudioCodecKind {
    fn from(c: &MkvCodec) -> Self {
        match c {
            MkvCodec::Aac => AudioCodecKind::Aac,
            MkvCodec::Opus => AudioCodecKind::Opus,
            MkvCodec::Vorbis => AudioCodecKind::Vorbis,
            MkvCodec::Eac3 => AudioCodecKind::Eac3,
            other => AudioCodecKind::Other(other.to_string()),
        }
    }
}

/// The selected video track.
#[derive(Debug, Clone)]
pub struct VideoStream {
    /// Codec of the track.
    pub codec: VideoCodecKind,
    /// Coded width, or 0 when the container did not say.
    pub width: u32,
    /// Coded height, or 0 when the container did not say.
    pub height: u32,
    /// Frames per second, when the container said.
    ///
    /// `None` is *unknown*, and has to stay distinguishable from a real rate:
    /// nothing on the buffer path carries a framerate at all — the demuxers
    /// emit encoded buffers, whose `MediaFormat::Video` has no slot for one,
    /// and `Framerate::default()` is 30 fps, so a decoded frame cannot be
    /// asked either. This is read from the container header and is the only
    /// honest source the player has. Frame stepping refuses rather than
    /// guessing 30 (#281).
    pub frame_rate: Option<f64>,
}

impl VideoStream {
    /// How long one frame lasts, when the framerate is known and sane.
    pub fn frame_duration(&self) -> Option<Duration> {
        let fps = self.frame_rate?;
        // A container that states 0 fps, or something absurd, has said
        // nothing useful. 1000 fps is well past any real file and keeps the
        // arithmetic away from zero-length steps.
        (fps.is_finite() && fps > 0.0 && fps <= 1000.0).then(|| Duration::from_secs_f64(1.0 / fps))
    }
}

/// The selected audio track.
///
/// `extradata` is the codec's out-of-band configuration: the
/// AudioSpecificConfig for AAC (esds-synthesized for MP4, raw CodecPrivate
/// for MKV), Vorbis's header triple, and so on.
#[derive(Debug, Clone)]
pub struct AudioStream {
    /// Codec of the track.
    pub codec: AudioCodecKind,
    /// Declared sample rate.
    pub sample_rate: u32,
    /// Declared channel count.
    pub channels: u32,
    /// Out-of-band decoder configuration.
    pub extradata: Option<Vec<u8>>,
}

/// What a file holds, and what the player will do with it.
#[derive(Debug, Clone)]
pub struct SourceDescription {
    /// The file this describes.
    pub path: PathBuf,
    /// The selected video track.
    pub video: Option<VideoStream>,
    /// The selected audio track.
    pub audio: Option<AudioStream>,
    /// Duration, or `ClockTime::NONE` when the container did not say.
    pub duration: ClockTime,
    /// Subtitle tracks present but not rendered (#279).
    pub subtitles: usize,
}

impl SourceDescription {
    /// A one-line summary for the interface.
    pub fn summary(&self) -> String {
        let name = self
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| self.path.display().to_string());
        let video = self
            .video
            .as_ref()
            .map(|v| format!("{} {}x{}", v.codec, v.width, v.height))
            .unwrap_or_else(|| "no video".into());
        let audio = self
            .audio
            .as_ref()
            .map(|a| format!("{} {} Hz", a.codec, a.sample_rate))
            .unwrap_or_else(|| "no audio".into());
        format!("{name} · {video} · {audio}")
    }
}

/// The opened demuxer, by container.
// Both boxed: each carries its own index and parser state, so bare variants
// would make every `Probed` the size of the larger one.
enum ProbedSource {
    Mp4(Box<Mp4Demux<BufReader<File>>>),
    Mkv(Box<MkvDemux<BufReader<File>>>),
}

/// An opened, described file, ready to be built into a pipeline.
///
/// `Debug` is shallow on purpose: an Iced `Message` must be `Debug`, and the
/// demuxers inside carry whole sample indices.
pub struct Probed {
    source: ProbedSource,
    /// What the probe found.
    pub description: SourceDescription,
}

impl std::fmt::Debug for Probed {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Probed")
            .field("description", &self.description)
            .finish()
    }
}

/// Open a file and describe it. Blocking — an MP4 index parse is not quick.
pub fn probe(path: &Path) -> anyhow::Result<Probed> {
    let mut file = File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let size = file.metadata()?.len();

    // Name what the file actually is before the demuxer's parse error can:
    // "AVI detected" beats "not a readable MP4".
    let mut head = [0u8; 8192];
    let n = file.read(&mut head)?;
    let media_type = match TypeFindRegistry::with_builtins().detect(&head[..n]) {
        Some(found) => found.media_type,
        None => bail!(
            "{}: not a media file this player recognizes",
            path.display()
        ),
    };
    file.seek(SeekFrom::Start(0))?;

    let (source, mut description) = match media_type {
        MediaType::Mp4 => {
            let demux = Mp4Demux::new(BufReader::with_capacity(1 << 20, file), size)
                .with_context(|| format!("{} is not a readable MP4", path.display()))?;
            let description = describe_mp4(path, &demux);
            (ProbedSource::Mp4(Box::new(demux)), description)
        }
        MediaType::Matroska | MediaType::WebM => {
            // 1 MiB: the matroska parser issues many small reads through a
            // mutex'd SharedReader, inline on a tokio worker.
            let demux = MkvDemux::new(BufReader::with_capacity(1 << 20, file))
                .with_context(|| format!("{} is not a readable Matroska/WebM", path.display()))?;
            let description = describe_mkv(path, &demux);
            (ProbedSource::Mkv(Box::new(demux)), description)
        }
        other => bail!(
            "{}: {other:?} detected — only MP4 and MKV/WebM are supported for now",
            path.display()
        ),
    };
    description.path = path.to_path_buf();

    let Some(video) = &description.video else {
        bail!("no video track — nothing to play");
    };
    match &video.codec {
        VideoCodecKind::Other(codec) => bail!("video track is {codec} — no decoder for it yet"),
        // HEVC has neither a software decoder in this tree nor, for now, a
        // hardware one: `cros-codecs`' `h265` feature does not compile against
        // its published release (#200). The container side still reads the
        // track, which is why this is a clean message rather than a failure
        // three seconds into playback.
        VideoCodecKind::H265 => bail!(
            "H.265 is not supported: parallax has no software HEVC decoder, and \
             hardware HEVC is disabled pending an upstream cros-codecs fix (#200)"
        ),
        _ => {}
    }

    Ok(Probed {
        source,
        description,
    })
}

fn describe_mp4(path: &Path, demux: &Mp4Demux<BufReader<File>>) -> SourceDescription {
    let video = demux
        .video_track_id()
        .and_then(|id| demux.track(id))
        .map(|t| {
            let info = t.video_info.as_ref();
            VideoStream {
                codec: t.codec.into(),
                width: info.map(|i| i.width).unwrap_or(0),
                height: info.map(|i| i.height).unwrap_or(0),
                frame_rate: info.and_then(|i| i.frame_rate),
            }
        });
    let audio = demux
        .audio_track_id()
        .and_then(|id| demux.track(id))
        .and_then(|t| {
            let info = t.audio_info.as_ref()?;
            Some(AudioStream {
                codec: match t.codec {
                    Mp4Codec::Aac => AudioCodecKind::Aac,
                    Mp4Codec::Opus => AudioCodecKind::Opus,
                    other => AudioCodecKind::Other(other.to_string()),
                },
                sample_rate: info.sample_rate,
                channels: info.channels as u32,
                extradata: info.audio_specific_config.clone(),
            })
        });
    SourceDescription {
        path: path.to_path_buf(),
        video,
        audio,
        duration: ClockTime::from_nanos(demux.duration_ns()),
        subtitles: demux
            .tracks()
            .iter()
            .filter(|t| t.track_type == Mp4TrackType::Subtitle)
            .count(),
    }
}

fn describe_mkv(path: &Path, demux: &MkvDemux<BufReader<File>>) -> SourceDescription {
    let video = demux.video_track().map(|t| {
        let info = t.video_info.as_ref();
        VideoStream {
            codec: (&t.codec).into(),
            width: info.map(|i| i.width).unwrap_or(0),
            height: info.map(|i| i.height).unwrap_or(0),
            frame_rate: info.and_then(|i| i.frame_rate),
        }
    });
    let audio = demux.audio_track().and_then(|t| {
        let info = t.audio_info.as_ref()?;
        Some(AudioStream {
            codec: (&t.codec).into(),
            sample_rate: info.sample_rate,
            channels: info.channels as u32,
            extradata: info.codec_private.clone(),
        })
    });
    SourceDescription {
        path: path.to_path_buf(),
        video,
        audio,
        // MKV without a Segment Duration legitimately does not know: that is
        // NONE, not zero, and clamping a seek against zero sends every one to
        // the start.
        duration: demux
            .duration_ns()
            .map(ClockTime::from_nanos)
            .unwrap_or(ClockTime::NONE),
        subtitles: demux
            .tracks()
            .iter()
            .filter(|t| t.track_type == MkvTrackType::Subtitle)
            .count(),
    }
}

/// The audio decoder chosen for the stream — one variant per codec, so each
/// concrete `AudioDecoderElement<D>` can be added to the pipeline.
enum AudioDec {
    Aac(AudioDecoderElement<AacDecoder>),
    Opus(AudioDecoderElement<OpusDecoder>),
    Vorbis(AudioDecoderElement<VorbisDecoder>),
    Eac3(AudioDecoderElement<Eac3Decoder>),
}

/// The hardware decoder for `codec`, if this machine offers one.
///
/// Every failure is a reason to use software, not a reason to stop: no DRM
/// node, no driver, or — commonly — a driver built without the codec the
/// backend needs to initialise.
fn hw_decoder(
    mode: HwDec,
    codec: parallax::gpu::Codec,
    label: &str,
    chosen: &mut String,
) -> Option<parallax::elements::VaapiDecoder> {
    if mode == HwDec::Off {
        *chosen = format!("{label} · software (--hwdec off)");
        return None;
    }
    match parallax::elements::VaapiDecoder::open(codec) {
        Ok(dec) => {
            tracing::info!("{label}: decoding on the GPU video engine");
            *chosen = format!("{label} · VA-API");
            Some(dec)
        }
        Err(e) => {
            tracing::warn!("{label}: software decode ({e})");
            // The reason travels with the answer: "software" on a machine
            // with a GPU is a question, and this is the answer to it.
            *chosen = format!("{label} · software ({e})");
            None
        }
    }
}

/// Build the audio branch's fallible pieces, explaining a fallback to
/// video-only instead of failing playback.
///
/// The decoder's output format decides the ALSA configuration, so no
/// conversion sits in between. The sink also provides the pipeline clock at
/// priority 150 — once this branch exists, video paces off audio hardware.
fn audio_branch(opts: &Options, audio: Option<&AudioStream>) -> Option<(AudioDec, AlsaSink)> {
    if !opts.audio {
        return None;
    }
    let stream = audio?;

    let (decoder, alsa_sample_format) = match &stream.codec {
        AudioCodecKind::Aac => {
            let asc = stream.extradata.as_ref()?;
            match AacDecoder::from_asc(asc) {
                Ok(d) => (
                    AudioDec::Aac(AudioDecoderElement::new(d)),
                    AlsaSampleFormat::F32LE,
                ),
                Err(e) => {
                    tracing::warn!("audio decoder unavailable ({e}) — video only");
                    return None;
                }
            }
        }
        AudioCodecKind::Opus => {
            // Opus always decodes at 48 kHz; the track's declared rate is the
            // original input rate, not the decode rate.
            match OpusDecoder::new(48_000, stream.channels) {
                Ok(d) => (
                    AudioDec::Opus(AudioDecoderElement::new(d)),
                    AlsaSampleFormat::S16LE,
                ),
                Err(e) => {
                    tracing::warn!("audio decoder unavailable ({e}) — video only");
                    return None;
                }
            }
        }
        AudioCodecKind::Vorbis => {
            let private = stream.extradata.as_ref()?;
            match VorbisDecoder::from_codec_private(private) {
                Ok(d) => (
                    AudioDec::Vorbis(AudioDecoderElement::new(d)),
                    AlsaSampleFormat::F32LE,
                ),
                Err(e) => {
                    tracing::warn!("audio decoder unavailable ({e}) — video only");
                    return None;
                }
            }
        }
        AudioCodecKind::Eac3 => {
            // The decoder folds 5.1 down to stereo internally (spec LoRo
            // matrix), so ALSA always gets 2 channels here.
            match Eac3Decoder::stereo(stream.sample_rate) {
                Ok(d) => (
                    AudioDec::Eac3(AudioDecoderElement::new(d)),
                    AlsaSampleFormat::S16LE,
                ),
                Err(e) => {
                    tracing::warn!("audio decoder unavailable ({e}) — video only");
                    return None;
                }
            }
        }
        other => {
            tracing::warn!("audio track is {other} — decode not wired up yet, video only");
            return None;
        }
    };

    let (sample_rate, channels) = match decoder {
        AudioDec::Aac(_) | AudioDec::Vorbis(_) => (stream.sample_rate, stream.channels),
        AudioDec::Opus(_) => (48_000, stream.channels),
        AudioDec::Eac3(_) => (stream.sample_rate, 2),
    };
    // The AudioDownmix behind the decoder folds anything above stereo, so
    // ALSA never opens with more than 2 channels.
    let format = AlsaFormat {
        sample_rate,
        channels: channels.min(2),
        format: alsa_sample_format,
        ..AlsaFormat::default()
    };
    match AlsaSink::new(&opts.audio_device, format) {
        Ok(sink) => Some((decoder, sink)),
        Err(e) => {
            tracing::warn!(
                "audio device '{}' unavailable ({e}) — video only",
                opts.audio_device
            );
            None
        }
    }
}

/// Counts every buffer the pipeline sheds, for the diagnostics panel.
struct DropCounter(Arc<AtomicU64>);

impl parallax::pipeline::tracer::Tracer for DropCounter {
    // Drops only: without this, registering the tracer puts a mutex and a
    // clock read on every buffer at every hop for two hooks it ignores.
    fn interests(&self) -> parallax::pipeline::tracer::TracerInterests {
        parallax::pipeline::tracer::TracerInterests::DROP
    }

    fn on_drop(&self, _element: &str, _ts: std::time::Instant) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn name(&self) -> &str {
        "player-drops"
    }
}

/// One running pipeline and everything the interface needs to drive it.
pub struct Engine {
    handle: PipelineHandle,
    bus: Option<Bus>,
    /// The frames, for the widget.
    pub mailbox: Arc<FrameMailbox>,
    volume: Option<GainControl>,
    /// What is playing.
    pub source: SourceDescription,
    dropped: Arc<AtomicU64>,
    /// Which decoder actually won — hardware or software, and which one.
    ///
    /// The choice is made once and then consumed by `add_filter`, so it has
    /// to be recorded on the way past. It used to be logged and forgotten,
    /// which meant the only way to find out whether VA-API was live was to
    /// read the logs at startup.
    pub decoder: String,
    /// The graph as GraphViz, captured before `start` moved the elements.
    dot: String,
}

impl Engine {
    /// Build a pipeline for a probed file and start it.
    ///
    /// Synchronous on purpose. `Executor::start` is sync and fast, and calling
    /// it from Iced's `update` — which runs inside `runtime.enter()` when the
    /// `tokio` feature is on, so `tokio::spawn` finds a runtime — means the
    /// `Engine` and its `PipelineHandle` never have to be `Send`.
    pub fn build(probed: Probed, opts: &Options) -> anyhow::Result<Self> {
        let Probed {
            source,
            description,
        } = probed;
        let audio = audio_branch(opts, description.audio.as_ref());

        let mut pipeline = Pipeline::new();
        let dropped = Arc::new(AtomicU64::new(0));
        pipeline
            .tracer_registry()
            .add(Box::new(DropCounter(Arc::clone(&dropped))));

        // Without an audio branch the video-only source keeps the audio pad
        // from existing at all, so nothing warns about an unlinked pad.
        let src = match source {
            ProbedSource::Mp4(demux) => {
                let s = if audio.is_some() {
                    Mp4DemuxSource::new(*demux)
                } else {
                    Mp4DemuxSource::video_only(*demux)
                }
                .with_loop(opts.gapless_loop);
                pipeline.add_demuxer("demux", s)
            }
            ProbedSource::Mkv(demux) => {
                let s = if audio.is_some() {
                    *demux
                } else {
                    demux.video_only()
                }
                .with_loop(opts.gapless_loop);
                pipeline.add_demuxer("demux", s)
            }
        };

        let video = description
            .video
            .as_ref()
            .expect("probe guarantees a video track");
        let height = video.height;
        let mut decoder = String::new();
        let dec = match &video.codec {
            VideoCodecKind::H264 => {
                match hw_decoder(
                    opts.hwdec,
                    parallax::gpu::Codec::H264,
                    "H.264",
                    &mut decoder,
                ) {
                    Some(hw) => pipeline.add_filter("decode", hw),
                    None => pipeline.add_filter("decode", H264Decoder::new()?),
                }
            }
            VideoCodecKind::Vp8 => {
                match hw_decoder(opts.hwdec, parallax::gpu::Codec::Vp8, "VP8", &mut decoder) {
                    Some(hw) => pipeline.add_filter("decode", hw),
                    None => pipeline.add_filter("decode", VpxDecoder::vp8()?),
                }
            }
            VideoCodecKind::Vp9 => {
                match hw_decoder(opts.hwdec, parallax::gpu::Codec::Vp9, "VP9", &mut decoder) {
                    Some(hw) => pipeline.add_filter("decode", hw),
                    None => pipeline.add_filter("decode", VpxDecoder::vp9()?),
                }
            }
            VideoCodecKind::Av1 => {
                // At paced rates dav1d worker CPU scales with pool size
                // (idle-worker task-queue churn), so size the pool to the
                // stream rather than the machine (#192).
                let threads = match opts.dav1d_threads {
                    0 if height > 0 && height <= 1440 => 2,
                    explicit_or_auto => explicit_or_auto,
                };
                let mut dav1d = Dav1dDecoder::new()?.with_threads(threads)?;
                if opts.frame_delay > 0 {
                    dav1d = dav1d.with_max_frame_delay(opts.frame_delay)?;
                }
                decoder = format!("AV1 · dav1d, {threads} threads");
                pipeline.add_filter("decode", dav1d)
            }
            // `probe` rejects these, so reaching here is a bug rather than a
            // user error.
            other => bail!("video track is {other} — no decoder for it yet"),
        };

        let sink = AppVideoSink::new()
            .with_sync(opts.sync)
            .with_host_retention(parallax_iced::HOST_RETENTION);
        let mailbox = sink.mailbox();
        let snk = pipeline.add_async_sink("display", sink);

        // Deep branch links decouple the two consumers: the demuxer emits in
        // DTS order, so if the video branch back-pressures at its exact rate
        // the audio branch starves, the device underruns, the audio-master
        // clock freezes, and the paced video sink waits on that frozen clock.
        pipeline.link_pads_full(src, "video", dec, "sink", LinkPolicy::Block, Some(24))?;
        // The decode-ahead window: the decoder blocks in send_output once it
        // is this many finished frames ahead of the screen (#192).
        let ahead = Some(opts.decode_ahead.max(1));
        pipeline.link_pads_full(dec, "src", snk, "sink", LinkPolicy::Block, ahead)?;

        let mut volume = None;
        if let Some((audio_decoder, alsa_sink)) = audio {
            let adec = match audio_decoder {
                AudioDec::Aac(d) => pipeline.add_transform("audiodec", d),
                AudioDec::Opus(d) => pipeline.add_transform("audiodec", d),
                AudioDec::Vorbis(d) => pipeline.add_transform("audiodec", d),
                AudioDec::Eac3(d) => pipeline.add_transform("audiodec", d),
            };
            let adownmix = pipeline.add_filter("adownmix", AudioDownmix::new());
            // Format-aware software volume: it reads each buffer's AudioRaw
            // metadata, so it sits safely on the S16 and F32 branches alike.
            // The handle must be taken BEFORE start moves the element.
            let gain = Gain::new(1.0).with_name("volume");
            volume = Some(gain.control());
            let avolume = pipeline.add_filter("volume", gain);
            let aout = pipeline.add_async_sink("speaker", alsa_sink);
            // ~2 s of audio read-ahead at ~20 ms per packet.
            pipeline.link_pads_full(src, "audio", adec, "sink", LinkPolicy::Block, Some(96))?;
            pipeline.link(adec, adownmix)?;
            pipeline.link(adownmix, avolume)?;
            pipeline.link(avolume, aout)?;
        }

        // Before `start`: it moves the elements into their tasks, and the
        // pipeline is a local that goes out of scope here.
        let dot = pipeline.to_dot();

        let executor = Executor::with_config(
            ExecutorConfig::default().with_channel_capacity(opts.channel_capacity),
        );
        let mut handle = executor.start(&mut pipeline)?;
        let bus = handle.take_bus();

        let mut source = description;
        // The pipeline's own answer beats the container header's.
        if let Some(d) = handle.duration().to_option() {
            source.duration = d;
        }

        Ok(Self {
            handle,
            bus,
            mailbox,
            volume,
            source,
            dropped,
            decoder,
            dot,
        })
    }

    /// The transport, for seeks and pause.
    pub fn handle(&self) -> &PipelineHandle {
        &self.handle
    }

    /// Drain whatever the pipeline has posted since the last look.
    pub fn poll_bus(&mut self) -> Vec<parallax::pipeline::Message> {
        let mut out = Vec::new();
        if let Some(bus) = self.bus.as_mut() {
            while let Some(msg) = bus.poll() {
                out.push(msg);
            }
        }
        out
    }

    /// Buffers the pipeline has shed, all causes, all elements.
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Set the output level as a linear factor. No-op on a video-only run.
    pub fn set_volume(&self, factor: f32) {
        if let Some(volume) = &self.volume {
            volume.set_factor(factor);
        }
    }

    /// The running graph as GraphViz, for `to_dot` dumps.
    pub fn dot(&self) -> &str {
        &self.dot
    }

    /// Whether this run has audio at all.
    pub fn has_audio(&self) -> bool {
        self.volume.is_some()
    }

    /// Where playback has reached.
    pub fn position(&self) -> ClockTime {
        self.handle.position()
    }

    /// Stop the pipeline and wait for it, bounded.
    ///
    /// Consuming, because a torn-down engine has nothing left to offer: the
    /// mailbox is ended, the handle is spent.
    pub async fn teardown(self) {
        let Self { handle, .. } = self;
        handle.stop();
        // A stop that does not land is a bug somewhere, but it must not
        // become a frozen interface.
        match tokio::time::timeout(std::time::Duration::from_secs(5), handle.wait()).await {
            Ok(Ok(())) => {}
            Ok(Err(e)) => tracing::warn!("pipeline ended with an error: {e}"),
            Err(_) => tracing::warn!("pipeline did not stop within 5s"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The 2 s H.264/AAC fixture: 20 frames at 10 fps, keyframe every 5.
    fn fixture() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../tests/fixtures/tiny_h264_aac.mkv")
    }

    #[test]
    fn the_container_states_a_frame_rate_and_it_reaches_the_transport() {
        let probed = probe(&fixture()).expect("the fixture probes");
        let video = probed.description.video.expect("it has video");
        // 10 fps, so one frame is 100 ms — the step size (#281).
        assert_eq!(
            video.frame_duration(),
            Some(Duration::from_millis(100)),
            "frame rate {:?} did not survive the probe",
            video.frame_rate
        );
    }

    /// Why there is no end-to-end frame-step test here.
    ///
    /// The player links the demuxer to the decoder 24 buffers deep
    /// (`Engine::build`), and the only fixture in the tree is two seconds —
    /// twenty frames. The whole file therefore fits in the link buffer, so
    /// the demuxer reads it to EOS within milliseconds of `start()` and there
    /// is nothing left to reposition by the time a seek could be issued;
    /// frames then drain out of the buffers at pace and playback looks
    /// perfectly normal. `tests/av_sync_tests.rs` hits the same wall and
    /// works around it by building its own pipeline with 4/8-deep links,
    /// which is exactly what makes it *not* the player's pipeline.
    ///
    /// So the arithmetic is tested here and the seek behaviour it depends on
    /// — ACCURATE landing on the requested time rather than a keyframe — is
    /// tested against a real MKV in `av_sync_tests`. Proving the two together
    /// needs a fixture longer than the link depth.
    const _: () = ();
}
