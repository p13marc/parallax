# parallax-player

A video player demo built on the [parallax](..) pipeline engine. Lives in the
workspace as a demo app (`publish = false`); it may graduate to a standalone
project later.

```bash
cargo run -p parallax-player -- movie.mp4
```

## Scope

MP4/MOV files with H.264 video (and AAC-LC audio, once the audio branch
lands). The pipeline is built programmatically on the in-pipeline demuxer:

```text
Mp4DemuxSource ─ video ─▶ H264Decoder ─▶ VideoConvert ─▶ AutoVideoSink(sync)
              └─ audio ─▶ AacDecoder ─▶ AudioConvert ─▶ AlsaSink
```

## Roadmap

Tracked on the Forgejo issue tracker under the `player-app` label:

| Issue | What | Status |
|---|---|---|
| #77 | Scaffold: CLI, probe the file, print track info | done |
| #79 | Video-only playback (M0) | done |
| #80 | Audio branch, ALSA clock as sync master | pending |
| #81 | Pause (Space) and ±10 s seek (arrows) | pending |
| #82 | Position/duration status line | pending |
| #83 | Friendly errors and media probing | pending |

The core engine work the player rides on (in-pipeline MP4 demuxing with
per-pad routing, runtime pause/seek/position, the streaming AAC decoder,
PTS-paced presentation, the ALSA hardware clock) is already merged — see
issues #64–#76 for the history.

## Flags

- `--no-audio` — video only, even if the file has audio
- `--loop` — restart playback at EOS
