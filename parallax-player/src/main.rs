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
use parallax::elements::codec::AudioDecoderElement;
use parallax::elements::demux::{Mp4Codec, Mp4Demux, Mp4DemuxSource, Mp4TrackType};
use parallax::elements::device::{AlsaFormat, AlsaSampleFormat, AlsaSink};
use parallax::elements::transform::VideoConvertElement;
use parallax::elements::{AacDecoder, AutoVideoSink, H264Decoder, VideoKey, VideoWindowEvent};
use parallax::pipeline::{EndReason, Executor, LinkPolicy, Pipeline};
use std::fs::File;
use std::io::{BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

/// A video player built on the parallax pipeline engine.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// The media file to play (MP4/MOV with H.264 video).
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

/// What the audio branch needs from the container: the ASC plus the stream
/// parameters, when the file has a playable AAC track.
struct AudioParams {
    asc: Vec<u8>,
    sample_rate: u32,
    channels: u32,
}

/// Everything `play` needs from the container probe.
struct Probed {
    demux: Mp4Demux<BufReader<File>>,
    width: u32,
    height: u32,
    audio: Option<AudioParams>,
    duration_ns: u64,
}

/// Open the file and describe it; error out early on things we cannot play.
fn open_and_probe(args: &Args) -> anyhow::Result<Probed> {
    let file =
        File::open(&args.file).with_context(|| format!("cannot open {}", args.file.display()))?;
    let size = file.metadata()?.len();
    let demux = Mp4Demux::new(BufReader::new(file), size)
        .with_context(|| format!("{} is not a readable MP4", args.file.display()))?;

    println!(
        "{} — {:.1}s, {} track(s)",
        args.file.display(),
        demux.duration_ns() as f64 / 1e9,
        demux.tracks().len()
    );

    let mut geometry = None;
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
                if track.codec == Mp4Codec::H264 {
                    geometry = info.map(|i| (i.width, i.height));
                }
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

    let Some(video_id) = demux.video_track_id() else {
        bail!("no video track — nothing to play");
    };
    let codec = demux.track(video_id).map(|t| t.codec);
    if codec != Some(Mp4Codec::H264) {
        bail!(
            "video track is {} — only H.264 is supported for now",
            codec.map(|c| c.to_string()).unwrap_or_default()
        );
    }
    let (width, height) = geometry.filter(|(w, h)| *w > 0 && *h > 0).unwrap_or((0, 0));

    // The audio branch needs an AAC track with a usable decoder config.
    let audio = demux
        .audio_track_id()
        .and_then(|id| demux.track(id))
        .filter(|t| t.codec == Mp4Codec::Aac)
        .and_then(|t| t.audio_info.as_ref())
        .and_then(|info| {
            info.audio_specific_config.as_ref().map(|asc| AudioParams {
                asc: asc.clone(),
                sample_rate: info.sample_rate,
                channels: info.channels as u32,
            })
        });

    let duration_ns = demux.duration_ns();
    Ok(Probed {
        demux,
        width,
        height,
        audio,
        duration_ns,
    })
}

/// The player's control commands, from window keys or the terminal.
enum Command {
    PauseToggle,
    SeekBy(i64),
    Fullscreen,
    Quit,
}

fn command_for_key(key: &VideoKey) -> Option<Command> {
    match key {
        VideoKey::Space => Some(Command::PauseToggle),
        VideoKey::ArrowLeft => Some(Command::SeekBy(-10)),
        VideoKey::ArrowRight => Some(Command::SeekBy(10)),
        VideoKey::Escape => Some(Command::Quit),
        VideoKey::Enter => Some(Command::Fullscreen),
        VideoKey::Character(c) => match c.as_str() {
            "q" => Some(Command::Quit),
            "f" => Some(Command::Fullscreen),
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
                KeyCode::Char('q') | KeyCode::Esc => Some(Command::Quit),
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

/// Build the audio branch's fallible pieces, explaining a fallback to
/// video-only instead of failing playback.
fn audio_branch(
    args: &Args,
    audio: Option<AudioParams>,
) -> Option<(AudioDecoderElement<AacDecoder>, AlsaSink)> {
    if args.no_audio {
        return None;
    }
    let Some(params) = audio else {
        println!("no playable AAC track — video only");
        return None;
    };
    let decoder = match AacDecoder::from_asc(&params.asc) {
        Ok(d) => d,
        Err(e) => {
            println!("audio decoder unavailable ({e}) — video only");
            return None;
        }
    };
    // The decoder emits interleaved f32 at the stream's rate; hand ALSA
    // exactly that so no conversion sits in between. The sink also provides
    // the pipeline clock (priority 150), so once this branch exists the
    // video paces off the audio hardware — that is M2's A/V sync.
    let format = AlsaFormat {
        sample_rate: params.sample_rate,
        channels: params.channels,
        format: AlsaSampleFormat::F32LE,
        ..AlsaFormat::default()
    };
    match AlsaSink::new("default", format) {
        Ok(sink) => Some((AudioDecoderElement::new(decoder), sink)),
        Err(e) => {
            println!("audio device unavailable ({e}) — video only");
            None
        }
    }
}

/// One playback pass: build the pipeline, run it to EOS / close / Ctrl-C.
async fn play(args: &Args) -> anyhow::Result<Outcome> {
    let Probed {
        demux,
        width,
        height,
        audio,
        duration_ns,
    } = open_and_probe(args)?;
    let audio = audio_branch(args, audio);

    // Without an audio branch the video-only source keeps the audio pad from
    // existing at all (no unlinked-pad drop warnings).
    let source = if audio.is_some() {
        Mp4DemuxSource::new(demux)
    } else {
        Mp4DemuxSource::video_only(demux)
    }
    .with_loop(args.loop_playback);

    let mut convert = VideoConvertElement::new()
        .with_input_format(PixelFormat::I420)
        .with_output_format(PixelFormat::Rgba);
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
    let src = pipeline.add_demuxer("mp4demux", source);
    let dec = pipeline.add_filter("decode", H264Decoder::new()?);
    let cvt = pipeline.add_filter("convert", convert);
    let snk = pipeline.add_sink("display", sink);
    // Deep branch links decouple the two consumers: the demuxer emits in DTS
    // order, so if the video branch backpressures at its exact rate the audio
    // branch starves, the device underruns, the audio-master clock freezes,
    // and the paced video sink waits on that frozen clock — a crawl. The
    // demuxer's output arena is sized from these capacities (#84/#91), so
    // depth here is accounted memory, not a leak.
    pipeline.link_pads_full(src, "video", dec, "sink", LinkPolicy::Block, Some(24))?;
    pipeline.link(dec, cvt)?;
    pipeline.link(cvt, snk)?;

    if let Some((audio_decoder, alsa_sink)) = audio {
        let adec = pipeline.add_transform("aacdec", audio_decoder);
        let aout = pipeline.add_async_sink("speaker", alsa_sink);
        // ~2 s of AAC read-ahead at 21 ms per packet.
        pipeline.link_pads_full(src, "audio", adec, "sink", LinkPolicy::Block, Some(96))?;
        pipeline.link(adec, aout)?;
    }

    let executor = Executor::new();
    let mut handle = executor
        .start(&mut pipeline)
        .context("failed to start the pipeline")?;
    let mut bus = handle.take_bus();
    let mut ended = handle.ended();

    println!("controls: Space pause · ←/→ seek 10s · f fullscreen · q quit");
    let raw_mode = RawMode::enable();
    let mut status_tick = tokio::time::interval(std::time::Duration::from_millis(250));

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
                print!(
                    "\r\x1b[K{} / {}{state}{drops}{notes}",
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
                            let target = (position + secs * 1_000_000_000)
                                .clamp(0, duration_ns.saturating_sub(1) as i64);
                            handle.seek_time(ClockTime::from_nanos(target as u64)).await;
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
