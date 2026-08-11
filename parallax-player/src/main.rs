//! parallax-player — video player demo built on the parallax pipeline engine.
//!
//! M0 (#79): video-only playback at the stream's own speed, built entirely
//! on in-pipeline elements — `Mp4DemuxSource` (#76) routes the video track
//! straight into the decoder, and `AutoVideoSink(sync)` paces presentation
//! against the pipeline clock (#66). No app-side feeder thread.

use anyhow::{Context, bail};
use clap::Parser;
use parallax::converters::PixelFormat;
use parallax::elements::codec::AudioDecoderElement;
use parallax::elements::demux::{Mp4Codec, Mp4Demux, Mp4DemuxSource, Mp4TrackType};
use parallax::elements::device::{AlsaFormat, AlsaSampleFormat, AlsaSink};
use parallax::elements::transform::VideoConvertElement;
use parallax::elements::{AacDecoder, AutoVideoSink, H264Decoder};
use parallax::pipeline::{EndReason, Executor, LinkPolicy, Pipeline};
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;

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

/// Open the file and describe it; error out early on things we cannot play.
fn open_and_probe(
    args: &Args,
) -> anyhow::Result<(Mp4Demux<BufReader<File>>, u32, u32, Option<AudioParams>)> {
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

    Ok((demux, width, height, audio))
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
    let (demux, width, height, audio) = open_and_probe(args)?;
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
    let handle = executor
        .start(&mut pipeline)
        .context("failed to start the pipeline")?;
    let mut ended = handle.ended();

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
            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {
                if window.is_open() {
                    window_seen = true;
                } else if window_seen {
                    handle.stop();
                }
            }
        }
    };

    let _ = handle.wait().await;
    Ok(outcome)
}
