//! parallax-player — video player demo built on the parallax pipeline engine.
//!
//! M0 scaffold (#77): open the file, show what's inside, exit cleanly.
//! Playback lands with #79.

use anyhow::{Context, bail};
use clap::Parser;
use parallax::elements::demux::{Mp4Demux, Mp4TrackType};
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

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let args = Args::parse();

    let file =
        File::open(&args.file).with_context(|| format!("cannot open {}", args.file.display()))?;
    let size = file.metadata()?.len();
    let demux = Mp4Demux::new(BufReader::new(file), size)
        .with_context(|| format!("{} is not a readable MP4", args.file.display()))?;

    let duration_ns = demux.duration_ns();
    println!(
        "{} — {:.1}s, {} track(s)",
        args.file.display(),
        duration_ns as f64 / 1e9,
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
                    "  #{} audio: {} {} Hz, {} ch, {} samples{}",
                    track.id,
                    track.codec,
                    info.map(|i| i.sample_rate).unwrap_or(0),
                    info.map(|i| i.channels).unwrap_or(0),
                    track.sample_count,
                    if info
                        .and_then(|i| i.audio_specific_config.as_ref())
                        .is_some()
                    {
                        ""
                    } else {
                        " (no decoder config)"
                    }
                );
            }
            other => println!("  #{} {:?}: {}", track.id, other, track.codec),
        }
    }

    if demux.video_track_id().is_none() {
        bail!("no video track — nothing to play");
    }

    // Playback arrives with #79 (video) and #80 (audio).
    println!(
        "\nplayback not implemented yet (#79); flags: no_audio={}, loop={}",
        args.no_audio, args.loop_playback
    );
    Ok(())
}
