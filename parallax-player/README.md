# parallax-player

A video player demo built on the [parallax](..) pipeline engine. Lives in the
workspace as a demo app (`publish = false`); it may graduate to a standalone
project later.

```bash
cargo run -p parallax-player -- movie.mp4
```

## Scope

MP4/MOV and MKV/WebM files; video H.264, VP8, VP9 or AV1; audio AAC-LC,
Opus, Vorbis or E-AC-3 (5.1 folded to stereo); subtitle tracks are listed but not rendered.
The pipeline is built programmatically on the in-pipeline demuxers, with
the decoder chosen from the probed codec:

```text
Mp4DemuxSource │        H264Decoder │
MkvDemux       ├ video ─▶ VpxDecoder ├─▶ VideoConvert ─▶ AutoVideoSink(sync)
               │       Dav1dDecoder │
               └ audio ─▶ AacDecoder | OpusDecoder | VorbisDecoder ─▶ AlsaSink
```

## Roadmap

Phase 1 (`player-app` label, all done): #77 scaffold, #79 video-only M0,
#80 audio branch + ALSA clock, #81 pause/seek, #82 status line, #83
friendly errors and probing.

Phase 2 — format expansion (`video-player` label):

| Issue | What | Status |
|---|---|---|
| #121/#122 | Matroska/WebM container support | done |
| #123/#124 | VP8/VP9 decode (libvpx) + codec dispatch | done |
| #125 | AV1/Opus in MP4 | done |
| #126 | Vorbis audio | done |
| #127 | E-AC-3 audio (in-decoder stereo fold-down) | done |
| #128 | Generic 5.1/7.1 downmix element | done |

The core engine work the player rides on (in-pipeline MP4 demuxing with
per-pad routing, runtime pause/seek/position, the streaming AAC decoder,
PTS-paced presentation, the ALSA hardware clock) is already merged — see
issues #64–#76 for the history.

## A/V sync

The ALSA sink provides the pipeline clock (priority 150, counting frames the
card has actually played) and is the **sync master**: on the first in-segment
sample after every flush it publishes the pipeline's running-time origin, and
the video sink presents against that rather than against its own arrival. That
is what makes a seek land both branches on the same instant instead of on
whatever skew the two happened to restart with — see the engine's
`docs/pipeline.md` § A/V synchronization.

Arrow-key seeks coalesce: presses accumulate into one target, settle for
120 ms, and go out as a single seek whose completion is matched back by
seqnum. The status line shows the target while the seek is outstanding.

## Flags

- `--no-audio` — video only, even if the file has audio
- `--loop` — restart playback at EOS
