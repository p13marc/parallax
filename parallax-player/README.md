# parallax-player

A video player demo built on the [parallax](..) pipeline engine. Lives in the
workspace as a demo app (`publish = false`); it may graduate to a standalone
project later.

```bash
cargo run -p parallax-player -- movie.mp4
```

## Scope

MP4/MOV and MKV/WebM files with H.264 video and AAC-LC audio; subtitle
tracks are listed but not rendered. The pipeline is built programmatically
on the in-pipeline demuxers:

```text
Mp4DemuxSource │
MkvDemux       ├ video ─▶ H264Decoder ─▶ VideoConvert ─▶ AutoVideoSink(sync)
               └ audio ─▶ AacDecoder ─▶ AlsaSink
```

## Roadmap

Phase 1 (`player-app` label, all done): #77 scaffold, #79 video-only M0,
#80 audio branch + ALSA clock, #81 pause/seek, #82 status line, #83
friendly errors and probing.

Phase 2 — format expansion (`video-player` label):

| Issue | What | Status |
|---|---|---|
| #121/#122 | Matroska/WebM container support | done |
| #123/#124 | VP8/VP9 decode (libvpx) + codec dispatch | pending |
| #125 | AV1/Opus in MP4 | pending |
| #126 | Vorbis audio | pending |
| #127/#128 | E-AC-3 audio + 5.1 downmix | pending |

The core engine work the player rides on (in-pipeline MP4 demuxing with
per-pad routing, runtime pause/seek/position, the streaming AAC decoder,
PTS-paced presentation, the ALSA hardware clock) is already merged — see
issues #64–#76 for the history.

## Flags

- `--no-audio` — video only, even if the file has audio
- `--loop` — restart playback at EOS
