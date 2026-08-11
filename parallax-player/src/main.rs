//! parallax-player — video player demo built on the parallax pipeline engine.
//!
//! M0 (#79): video-only playback at the stream's own speed, built entirely
//! on in-pipeline elements — `Mp4DemuxSource` (#76) routes the video track
//! straight into the decoder, and `AutoVideoSink(sync)` paces presentation
//! against the pipeline clock (#66). No app-side feeder thread.

use anyhow::{Context, bail};
use clap::Parser;
use parallax::converters::PixelFormat;
use parallax::elements::demux::{Mp4Codec, Mp4Demux, Mp4DemuxSource, Mp4TrackType};
use parallax::elements::transform::VideoConvertElement;
use parallax::elements::{AutoVideoSink, H264Decoder};
use parallax::pipeline::{EndReason, Executor, Pipeline};
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

/// Open the file and describe it; error out early on things we cannot play.
fn open_and_probe(args: &Args) -> anyhow::Result<(Mp4Demux<BufReader<File>>, u32, u32)> {
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
    Ok((demux, width, height))
}

/// One playback pass: build the pipeline, run it to EOS / close / Ctrl-C.
async fn play(args: &Args) -> anyhow::Result<Outcome> {
    let (demux, width, height) = open_and_probe(args)?;

    // Audio is #80; until then the video-only source keeps the audio pad
    // from existing at all (no unlinked-pad drop warnings).
    let source = Mp4DemuxSource::video_only(demux).with_loop(args.loop_playback);

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
    pipeline.link_pads(src, "video", dec, "sink")?;
    pipeline.link(dec, cvt)?;
    pipeline.link(cvt, snk)?;

    let executor = Executor::new();
    let handle = executor
        .start(&mut pipeline)
        .context("failed to start the pipeline")?;
    let mut ended = handle.ended();

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
                if !window.is_open() {
                    handle.stop();
                }
            }
        }
    };

    let _ = handle.wait().await;
    Ok(outcome)
}
