//! parallax-player — video player demo built on the parallax pipeline engine.
//!
//! M0 (#79): video-only playback at the stream's own speed, built entirely
//! on in-pipeline elements — `Mp4DemuxSource` (#76) routes the video track
//! straight into the decoder, and `AutoVideoSink(sync)` paces presentation
//! against the pipeline clock (#66). No app-side feeder thread.

use anyhow::{Context, bail};
use clap::Parser;
use parallax::clock::ClockTime;
use parallax::converters::PixelFormat;
use parallax::elements::Eac3Decoder;
use parallax::elements::codec::AudioDecoderElement;
use parallax::elements::demux::{
    MkvCodec, MkvDemux, MkvTrackType, Mp4Codec, Mp4Demux, Mp4DemuxSource, Mp4TrackType,
};
use parallax::elements::device::{AlsaFormat, AlsaSampleFormat, AlsaSink};
use parallax::elements::transform::{AudioDownmix, VideoConvertElement};
use parallax::elements::{
    AacDecoder, AutoVideoSink, Dav1dDecoder, Gain, H264Decoder, OpusDecoder, VideoKey,
    VideoWindowEvent, VorbisDecoder, VpxDecoder,
};
use parallax::pipeline::typefind::{MediaType, TypeFindRegistry};
use parallax::pipeline::{EndReason, Executor, LinkPolicy, Pipeline};
use std::fs::File;
use std::io::{BufReader, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A video player built on the parallax pipeline engine.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// The media file to play (MP4/MOV or MKV/WebM with H.264 video).
    file: PathBuf,

    /// Disable the audio branch.
    #[arg(long)]
    no_audio: bool,

    /// Loop playback when the stream ends.
    #[arg(long = "loop")]
    loop_playback: bool,
}

/// Why one playback pass ended.
enum Outcome {
    /// Stream ran out (loopable).
    Eos,
    /// The user quit (window closed or Ctrl-C) or playback failed.
    Stop,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    // Looping happens inside the source (Mp4DemuxSource::with_loop): winit
    // allows one event loop per process ever, so the pipeline cannot be
    // rebuilt to reopen the window.
    let _ = play(&args).await?;
    Ok(())
}

/// Video codec of the selected track, container-agnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
enum VideoCodecKind {
    H264,
    Vp8,
    Vp9,
    Av1,
    Other(String),
}

impl std::fmt::Display for VideoCodecKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VideoCodecKind::H264 => write!(f, "H.264"),
            VideoCodecKind::Vp8 => write!(f, "VP8"),
            VideoCodecKind::Vp9 => write!(f, "VP9"),
            VideoCodecKind::Av1 => write!(f, "AV1"),
            VideoCodecKind::Other(s) => write!(f, "{s}"),
        }
    }
}

/// Audio codec of the selected track, container-agnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
enum AudioCodecKind {
    Aac,
    Opus,
    Vorbis,
    Eac3,
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

/// The selected video track, container-agnostic.
struct VideoStream {
    codec: VideoCodecKind,
    width: u32,
    height: u32,
}

/// The selected audio track, container-agnostic.
///
/// `extradata` is the codec's out-of-band configuration: the
/// AudioSpecificConfig for AAC (esds-synthesized for MP4, raw CodecPrivate
/// for MKV) — later codecs put their own headers here.
struct AudioStream {
    codec: AudioCodecKind,
    sample_rate: u32,
    channels: u32,
    extradata: Option<Vec<u8>>,
}

/// Container-agnostic description of what the file holds.
struct StreamSummary {
    video: Option<VideoStream>,
    audio: Option<AudioStream>,
}

/// The opened demuxer, by container.
enum ProbedSource {
    Mp4(Mp4Demux<BufReader<File>>),
    Mkv(MkvDemux<BufReader<File>>),
}

/// Everything `play` needs from the container probe.
struct Probed {
    source: ProbedSource,
    summary: StreamSummary,
}

/// Probe an opened MP4: print the track table and summarize the selection.
fn probe_mp4(args: &Args, demux: &Mp4Demux<BufReader<File>>) -> StreamSummary {
    println!(
        "{} — {:.1}s, {} track(s)",
        args.file.display(),
        demux.duration_ns() as f64 / 1e9,
        demux.tracks().len()
    );
    for track in demux.tracks() {
        match track.track_type {
            Mp4TrackType::Video => {
                let info = track.video_info.as_ref();
                println!(
                    "  #{} video: {} {}x{} @ {:.2} fps, {} samples",
                    track.id,
                    track.codec,
                    info.map(|i| i.width).unwrap_or(0),
                    info.map(|i| i.height).unwrap_or(0),
                    info.and_then(|i| i.frame_rate).unwrap_or(0.0),
                    track.sample_count,
                );
            }
            Mp4TrackType::Audio => {
                let info = track.audio_info.as_ref();
                println!(
                    "  #{} audio: {} {} Hz, {} ch, {} samples",
                    track.id,
                    track.codec,
                    info.map(|i| i.sample_rate).unwrap_or(0),
                    info.map(|i| i.channels).unwrap_or(0),
                    track.sample_count,
                );
            }
            other => println!("  #{} {:?}: {}", track.id, other, track.codec),
        }
    }

    let video = demux
        .video_track_id()
        .and_then(|id| demux.track(id))
        .map(|t| {
            let info = t.video_info.as_ref();
            VideoStream {
                codec: t.codec.into(),
                width: info.map(|i| i.width).unwrap_or(0),
                height: info.map(|i| i.height).unwrap_or(0),
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
    StreamSummary { video, audio }
}

/// Probe an opened MKV/WebM: print the track table and summarize.
fn probe_mkv(args: &Args, demux: &MkvDemux<BufReader<File>>) -> StreamSummary {
    let duration_ns = demux.duration_ns().unwrap_or(0);
    println!(
        "{} — {:.1}s, {} track(s)",
        args.file.display(),
        duration_ns as f64 / 1e9,
        demux.tracks().len()
    );
    let mut subtitles = 0usize;
    for track in demux.tracks() {
        match track.track_type {
            MkvTrackType::Video => {
                let info = track.video_info.as_ref();
                println!(
                    "  #{} video: {} {}x{} @ {:.2} fps",
                    track.id,
                    track.codec,
                    info.map(|i| i.width).unwrap_or(0),
                    info.map(|i| i.height).unwrap_or(0),
                    info.and_then(|i| i.frame_rate).unwrap_or(0.0),
                );
            }
            MkvTrackType::Audio => {
                let info = track.audio_info.as_ref();
                println!(
                    "  #{} audio: {} {} Hz, {} ch",
                    track.id,
                    track.codec,
                    info.map(|i| i.sample_rate).unwrap_or(0),
                    info.map(|i| i.channels).unwrap_or(0),
                );
            }
            MkvTrackType::Subtitle => subtitles += 1,
            other => println!("  #{} {:?}: {}", track.id, other, track.codec),
        }
    }
    if subtitles > 0 {
        println!("  ({subtitles} subtitle track(s) — not rendered)");
    }

    let video = demux.video_track().map(|t| {
        let info = t.video_info.as_ref();
        VideoStream {
            codec: (&t.codec).into(),
            width: info.map(|i| i.width).unwrap_or(0),
            height: info.map(|i| i.height).unwrap_or(0),
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
    StreamSummary { video, audio }
}

/// Open the file and describe it; error out early on things we cannot play.
fn open_and_probe(args: &Args) -> anyhow::Result<Probed> {
    let mut file =
        File::open(&args.file).with_context(|| format!("cannot open {}", args.file.display()))?;
    let size = file.metadata()?.len();

    // Name what the file actually is before the demuxer's parse error can:
    // "AVI detected" beats "not a readable MP4".
    let mut head = [0u8; 8192];
    let n = file.read(&mut head)?;
    let media_type = match TypeFindRegistry::with_builtins().detect(&head[..n]) {
        Some(found) => found.media_type,
        None => bail!(
            "{}: not a media file this player recognizes",
            args.file.display()
        ),
    };
    file.seek(SeekFrom::Start(0))?;

    let (source, summary) = match media_type {
        MediaType::Mp4 => {
            let demux = Mp4Demux::new(BufReader::new(file), size)
                .with_context(|| format!("{} is not a readable MP4", args.file.display()))?;
            let summary = probe_mp4(args, &demux);
            (ProbedSource::Mp4(demux), summary)
        }
        MediaType::Matroska | MediaType::WebM => {
            let demux = MkvDemux::new(BufReader::new(file)).with_context(|| {
                format!("{} is not a readable Matroska/WebM", args.file.display())
            })?;
            let summary = probe_mkv(args, &demux);
            (ProbedSource::Mkv(demux), summary)
        }
        other => bail!(
            "{}: {other:?} detected — only MP4 and MKV/WebM are supported for now",
            args.file.display()
        ),
    };

    let Some(video) = &summary.video else {
        bail!("no video track — nothing to play");
    };
    if let VideoCodecKind::Other(codec) = &video.codec {
        bail!("video track is {codec} — no decoder for it yet");
    }

    Ok(Probed { source, summary })
}

/// The player's control commands, from window keys or the terminal.
enum Command {
    PauseToggle,
    SeekBy(i64),
    /// Volume step in dB (±2 per keypress, clamped in the handler).
    VolumeBy(f32),
    MuteToggle,
    Fullscreen,
    Quit,
}

fn command_for_key(key: &VideoKey) -> Option<Command> {
    match key {
        VideoKey::Space => Some(Command::PauseToggle),
        VideoKey::ArrowLeft => Some(Command::SeekBy(-10)),
        VideoKey::ArrowRight => Some(Command::SeekBy(10)),
        VideoKey::ArrowUp => Some(Command::VolumeBy(2.0)),
        VideoKey::ArrowDown => Some(Command::VolumeBy(-2.0)),
        VideoKey::Escape => Some(Command::Quit),
        VideoKey::Enter => Some(Command::Fullscreen),
        VideoKey::Character(c) => match c.as_str() {
            "q" => Some(Command::Quit),
            "f" => Some(Command::Fullscreen),
            "m" => Some(Command::MuteToggle),
            " " => Some(Command::PauseToggle),
            _ => None,
        },
        _ => None,
    }
}

/// Raw-mode terminal keys, mapped onto the same commands as the window's.
///
/// Raw mode swallows the terminal's own Ctrl-C handling, so it is mapped to
/// Quit here explicitly. Only armed when stdin is a terminal.
fn command_for_terminal() -> Option<Command> {
    use crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers, poll, read};
    while poll(std::time::Duration::ZERO).unwrap_or(false) {
        if let Ok(Event::Key(key)) = read() {
            if key.kind != KeyEventKind::Press {
                continue;
            }
            if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
                return Some(Command::Quit);
            }
            return match key.code {
                KeyCode::Char(' ') => Some(Command::PauseToggle),
                KeyCode::Left => Some(Command::SeekBy(-10)),
                KeyCode::Right => Some(Command::SeekBy(10)),
                KeyCode::Up => Some(Command::VolumeBy(2.0)),
                KeyCode::Down => Some(Command::VolumeBy(-2.0)),
                KeyCode::Char('m') => Some(Command::MuteToggle),
                // No Esc here, deliberately: with `poll(ZERO)` crossterm can
                // catch the lone ESC byte of a not-yet-complete arrow
                // sequence and report Esc — quitting on an arrow key. `q`
                // and Ctrl-C quit the terminal path; the window path keeps
                // Escape (winit delivers whole key events, no CSI ambiguity).
                KeyCode::Char('q') => Some(Command::Quit),
                KeyCode::Char('f') => Some(Command::Fullscreen),
                _ => None,
            };
        }
    }
    None
}

/// Counts dropped buffers across the whole pipeline, for the status line.
struct DropCounter(Arc<AtomicU64>);

impl parallax::pipeline::tracer::Tracer for DropCounter {
    fn on_drop(&self, _element_name: &str, _ts: std::time::Instant) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }

    fn name(&self) -> &str {
        "player-drops"
    }
}

/// `mm:ss` for the status line.
fn mmss(ns: u64) -> String {
    let secs = ns / 1_000_000_000;
    format!("{:02}:{:02}", secs / 60, secs % 60)
}

/// Keep raw mode strictly scoped to the playback loop, whatever exit path.
struct RawMode(bool);

impl RawMode {
    fn enable() -> Self {
        use crossterm::tty::IsTty;
        let armed = std::io::stdin().is_tty() && crossterm::terminal::enable_raw_mode().is_ok();
        RawMode(armed)
    }
}

impl Drop for RawMode {
    fn drop(&mut self) {
        if self.0 {
            let _ = crossterm::terminal::disable_raw_mode();
        }
    }
}

/// The audio decoder chosen for the stream — variants per codec, so each
/// concrete `AudioDecoderElement<D>` can be added to the pipeline.
enum AudioDec {
    Aac(AudioDecoderElement<AacDecoder>),
    Opus(AudioDecoderElement<OpusDecoder>),
    Vorbis(AudioDecoderElement<VorbisDecoder>),
    Eac3(AudioDecoderElement<Eac3Decoder>),
}

/// Build the audio branch's fallible pieces, explaining a fallback to
/// video-only instead of failing playback.
///
/// The decoder's output format decides the ALSA configuration: AAC emits
/// interleaved f32, Opus interleaved s16 — hand ALSA exactly that so no
/// conversion sits in between. The sink also provides the pipeline clock
/// (priority 150), so once this branch exists the video paces off the
/// audio hardware.
fn audio_branch(args: &Args, audio: Option<&AudioStream>) -> Option<(AudioDec, AlsaSink)> {
    if args.no_audio {
        return None;
    }
    let Some(stream) = audio else {
        println!("no audio track — video only");
        return None;
    };

    let (decoder, alsa_sample_format) = match &stream.codec {
        AudioCodecKind::Aac => {
            let Some(asc) = &stream.extradata else {
                println!("AAC track has no usable decoder config — video only");
                return None;
            };
            match AacDecoder::from_asc(asc) {
                Ok(d) => (
                    AudioDec::Aac(AudioDecoderElement::new(d)),
                    AlsaSampleFormat::F32LE,
                ),
                Err(e) => {
                    println!("audio decoder unavailable ({e}) — video only");
                    return None;
                }
            }
        }
        AudioCodecKind::Opus => {
            // Opus always decodes at 48 kHz; the track's declared rate is
            // the original input rate, not the decode rate.
            match OpusDecoder::new(48_000, stream.channels) {
                Ok(d) => (
                    AudioDec::Opus(AudioDecoderElement::new(d)),
                    AlsaSampleFormat::S16LE,
                ),
                Err(e) => {
                    println!("audio decoder unavailable ({e}) — video only");
                    return None;
                }
            }
        }
        AudioCodecKind::Vorbis => {
            let Some(private) = &stream.extradata else {
                println!("Vorbis track has no CodecPrivate headers — video only");
                return None;
            };
            match VorbisDecoder::from_codec_private(private) {
                Ok(d) => (
                    AudioDec::Vorbis(AudioDecoderElement::new(d)),
                    AlsaSampleFormat::F32LE,
                ),
                Err(e) => {
                    println!("audio decoder unavailable ({e}) — video only");
                    return None;
                }
            }
        }
        AudioCodecKind::Eac3 => {
            // The decoder folds 5.1 programs down to stereo internally
            // (spec LoRo matrix), so ALSA always gets 2 channels here.
            match Eac3Decoder::stereo(stream.sample_rate) {
                Ok(d) => (
                    AudioDec::Eac3(AudioDecoderElement::new(d)),
                    AlsaSampleFormat::S16LE,
                ),
                Err(e) => {
                    println!("audio decoder unavailable ({e}) — video only");
                    return None;
                }
            }
        }
        other => {
            println!("audio track is {other} — decode not wired up yet, video only");
            return None;
        }
    };

    let (sample_rate, channels) = match decoder {
        AudioDec::Aac(_) | AudioDec::Vorbis(_) => (stream.sample_rate, stream.channels),
        AudioDec::Opus(_) => (48_000, stream.channels),
        // Stereo fold-down happens inside the decoder.
        AudioDec::Eac3(_) => (stream.sample_rate, 2),
    };
    // The AudioDownmix behind the decoder folds anything above stereo, so
    // ALSA never opens with more than 2 channels.
    let channels = channels.min(2);
    let format = AlsaFormat {
        sample_rate,
        channels,
        format: alsa_sample_format,
        ..AlsaFormat::default()
    };
    match AlsaSink::new("default", format) {
        Ok(sink) => Some((decoder, sink)),
        Err(e) => {
            println!("audio device unavailable ({e}) — video only");
            None
        }
    }
}

/// One playback pass: build the pipeline, run it to EOS / close / Ctrl-C.
async fn play(args: &Args) -> anyhow::Result<Outcome> {
    let Probed { source, summary } = open_and_probe(args)?;
    let (width, height) = summary
        .video
        .as_ref()
        .map(|v| (v.width, v.height))
        .filter(|(w, h)| *w > 0 && *h > 0)
        .unwrap_or((0, 0));
    let audio = audio_branch(args, summary.audio.as_ref());

    // BGRA, not RGBA: same SIMD conversion cost, but BGRA bytes read as
    // little-endian u32 are already softbuffer's presentation layout, so
    // AutoVideoSink's blit is a copy instead of a per-pixel channel shuffle
    // (the shuffle measured ~37% of the player's total CPU).
    let mut convert = VideoConvertElement::new()
        .with_input_format(PixelFormat::I420)
        .with_output_format(PixelFormat::Bgra);
    if width > 0 {
        convert = convert.with_size(width, height);
    }

    let title = args
        .file
        .file_name()
        .map(|n| format!("{} — parallax", n.to_string_lossy()))
        .unwrap_or_else(|| "parallax".into());
    // `sync` is what makes this a *player*: frames present at their PTS on
    // the pipeline clock instead of as fast as the decoder can go.
    let mut sink = AutoVideoSink::new().with_title(title).with_sync(true);
    let window = sink.handle();

    let mut pipeline = Pipeline::new();
    let dropped = Arc::new(AtomicU64::new(0));
    pipeline
        .tracer_registry()
        .add(Box::new(DropCounter(dropped.clone())));
    // Without an audio branch the video-only source keeps the audio pad from
    // existing at all (no unlinked-pad drop warnings).
    let src = match source {
        ProbedSource::Mp4(demux) => {
            let s = if audio.is_some() {
                Mp4DemuxSource::new(demux)
            } else {
                Mp4DemuxSource::video_only(demux)
            }
            .with_loop(args.loop_playback);
            pipeline.add_demuxer("demux", s)
        }
        ProbedSource::Mkv(demux) => {
            let s = if audio.is_some() {
                demux
            } else {
                demux.video_only()
            }
            .with_loop(args.loop_playback);
            pipeline.add_demuxer("demux", s)
        }
    };
    // The video decoder follows the probed codec; every decoder emits I420
    // with in-band geometry, so the convert → display tail is codec-agnostic.
    let video_codec = summary
        .video
        .as_ref()
        .map(|v| v.codec.clone())
        .expect("probe guarantees a video track");
    let dec = match video_codec {
        VideoCodecKind::H264 => pipeline.add_filter("decode", H264Decoder::new()?),
        VideoCodecKind::Vp8 => pipeline.add_filter("decode", VpxDecoder::vp8()?),
        VideoCodecKind::Vp9 => pipeline.add_filter("decode", VpxDecoder::vp9()?),
        VideoCodecKind::Av1 => pipeline.add_filter("decode", Dav1dDecoder::new()?),
        VideoCodecKind::Other(codec) => bail!("video track is {codec} — no decoder for it yet"),
    };
    let cvt = pipeline.add_filter("convert", convert);
    let snk = pipeline.add_async_sink("display", sink);
    // Deep branch links decouple the two consumers: the demuxer emits in DTS
    // order, so if the video branch backpressures at its exact rate the audio
    // branch starves, the device underruns, the audio-master clock freezes,
    // and the paced video sink waits on that frozen clock — a crawl. The
    // demuxer's output arena is sized from these capacities (#84/#91), so
    // depth here is accounted memory, not a leak.
    pipeline.link_pads_full(src, "video", dec, "sink", LinkPolicy::Block, Some(24))?;
    pipeline.link(dec, cvt)?;
    pipeline.link(cvt, snk)?;

    // Volume handle, taken before start(); None on a video-only run.
    let mut volume: Option<parallax::elements::GainControl> = None;
    if let Some((audio_decoder, alsa_sink)) = audio {
        let adec = match audio_decoder {
            AudioDec::Aac(d) => pipeline.add_transform("audiodec", d),
            AudioDec::Opus(d) => pipeline.add_transform("audiodec", d),
            AudioDec::Vorbis(d) => pipeline.add_transform("audiodec", d),
            AudioDec::Eac3(d) => pipeline.add_transform("audiodec", d),
        };
        // Fold any multichannel PCM to stereo; mono/stereo pass through
        // untouched, so this is free for the common case and keeps a
        // future multichannel decoder from mis-sizing the sink.
        let adownmix = pipeline.add_filter("adownmix", AudioDownmix::new());
        // Software volume — format-aware (reads each buffer's AudioRaw
        // metadata), so it sits safely on the S16 and F32 branches alike.
        // The control handle must be taken BEFORE start moves the element.
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

    let executor = Executor::new();
    let mut handle = executor
        .start(&mut pipeline)
        .context("failed to start the pipeline")?;
    let mut bus = handle.take_bus();
    let mut ended = handle.ended();

    // Duration comes from the framework now — the demuxer reported it at
    // start and the handle serves it (#162). NONE means unknown (streamed
    // WebM without a Segment Duration): keep 0 so the seek clamp below
    // stays open-ended.
    let duration_ns = handle
        .duration()
        .to_option()
        .map(|d| d.nanos())
        .unwrap_or(0);

    println!("controls: Space pause · ←/→ seek 10s · ↑/↓ volume · m mute · f fullscreen · q quit");
    let raw_mode = RawMode::enable();
    let mut status_tick = tokio::time::interval(std::time::Duration::from_millis(250));

    // Volume state lives player-side: the Gain handle only knows a linear
    // factor, while the UI thinks in dB steps and a mute that remembers the
    // level it will restore.
    let mut vol_db: f32 = 0.0;
    let mut muted = false;

    // The window opens lazily on the first displayed frame, so is_open()
    // starts out false — only treat false as "closed" after it was seen open.
    let mut window_seen = false;
    let outcome = loop {
        tokio::select! {
            reason = &mut ended => {
                break match reason {
                    EndReason::Eos => Outcome::Eos,
                    EndReason::Aborted => Outcome::Stop,
                    EndReason::Error(e) => {
                        // The window closing mid-stream surfaces as a sink
                        // error; that is the normal way to quit a player.
                        if e.message().contains("window closed") {
                            Outcome::Stop
                        } else {
                            bail!("playback failed: {e}");
                        }
                    }
                };
            }
            _ = tokio::signal::ctrl_c() => {
                println!("\ninterrupted");
                handle.stop();
                break Outcome::Stop;
            }
            _ = status_tick.tick() => {
                // One line, redrawn in place ~4x/s: position / duration,
                // pause marker, drop count, and any bus warnings/errors.
                let position = handle.position().to_option().map(|p| p.nanos()).unwrap_or(0);
                let state = if handle.is_paused() { " ⏸" } else { "" };
                let drops = dropped.load(Ordering::Relaxed);
                let drops = if drops > 0 {
                    format!(" · {drops} dropped")
                } else {
                    String::new()
                };
                let mut notes = String::new();
                if let Some(bus) = bus.as_mut() {
                    use parallax::pipeline::bus::MessageKind;
                    while let Some(msg) = bus.poll() {
                        match msg.kind {
                            MessageKind::Error { error, .. } => {
                                notes = format!(" · ERROR: {error}");
                            }
                            MessageKind::Warning { warning, .. } => {
                                notes = format!(" · warning: {warning}");
                            }
                            MessageKind::Buffering { percent, .. } => {
                                notes = format!(" · buffering {percent}%");
                            }
                            _ => {}
                        }
                    }
                }
                let vol = if volume.is_none() {
                    String::new()
                } else if muted {
                    " · muted".to_string()
                } else if vol_db != 0.0 {
                    format!(" · vol {vol_db:+.0} dB")
                } else {
                    String::new()
                };
                print!(
                    "\r\x1b[K{} / {}{state}{vol}{drops}{notes}",
                    mmss(position),
                    mmss(duration_ns)
                );
                let _ = std::io::stdout().flush();
            }
            _ = tokio::time::sleep(std::time::Duration::from_millis(20)) => {
                let mut commands = Vec::new();
                while let Some(event) = window.try_event() {
                    if let VideoWindowEvent::KeyPressed(key) = event
                        && let Some(cmd) = command_for_key(&key)
                    {
                        commands.push(cmd);
                    }
                }
                if let Some(cmd) = command_for_terminal() {
                    commands.push(cmd);
                }
                for cmd in commands {
                    match cmd {
                        Command::PauseToggle => {
                            if handle.is_paused() {
                                handle.resume();
                            } else {
                                handle.pause();
                            }
                        }
                        Command::SeekBy(secs) => {
                            let position = handle.position().to_option()
                                .map(|p| p.nanos() as i64)
                                .unwrap_or(0);
                            // duration 0 means *unknown* (MKV/WebM with no
                            // Segment Duration), not an empty file — clamping
                            // against it sent every seek to t=0. Leave the
                            // upper end open and let the demuxer land it.
                            let mut target = (position + secs * 1_000_000_000).max(0);
                            if duration_ns > 0 {
                                target = target.min(duration_ns.saturating_sub(1) as i64);
                            }
                            handle.seek_time(ClockTime::from_nanos(target as u64)).await;
                        }
                        Command::VolumeBy(delta) => {
                            if let Some(volume) = &volume {
                                vol_db = (vol_db + delta).clamp(-40.0, 6.0);
                                // Changing the volume always unmutes — the
                                // user reaching for ↑ wants to hear it.
                                muted = false;
                                volume.set_db(vol_db);
                            }
                        }
                        Command::MuteToggle => {
                            if let Some(volume) = &volume {
                                muted = !muted;
                                if muted {
                                    volume.set_factor(0.0);
                                } else {
                                    volume.set_db(vol_db);
                                }
                            }
                        }
                        Command::Fullscreen => window.set_fullscreen(!window.is_fullscreen()),
                        Command::Quit => handle.stop(),
                    }
                }

                if window.is_open() {
                    window_seen = true;
                } else if window_seen {
                    handle.stop();
                }
            }
        }
    };

    drop(raw_mode);
    println!();
    let _ = handle.wait().await;
    Ok(outcome)
}
