# parallax-player

A video player built on the [parallax](../) pipeline engine and
[Iced](https://iced.rs) 0.14.

```bash
cargo run -p parallax-player --release -- movie.mp4
```

The engine does the hard parts — demux, decode, A/V sync, seeking — and this
is the front end: a window, a video surface, and the transport.

```text
demux ─┬─ video ─▶ decode ─▶ AppVideoSink ⇄ mailbox ⇄ VideoView (parallax-iced)
       └─ audio ─▶ decode ─▶ downmix ─▶ gain ─▶ AlsaSink (provides the clock)
```

`AppVideoSink` keeps everything that needs a pipeline context — the anchor
handshake, segment mapping, flush generations, out-of-segment clipping, QoS —
and hands each frame to the host stamped with the running time it is due at.
The host decides when to show it, because only the host knows when the next
vertical blank is.

MP4/MOV and MKV/WebM, with H.264, VP8, VP9 and AV1 video and AAC, Opus, Vorbis
and E-AC-3 audio. VA-API is tried first for H.264/VP8/VP9 and falls back to
software without comment when the driver has not got it.

## Keys

| | |
|---|---|
| `Space`, click | play / pause |
| `←` `→` | seek 10 s |
| `↑` `↓`, wheel | volume |
| `m` | mute |
| `f`, double-click | fullscreen |
| `Esc` | leave fullscreen, else quit |
| `s` | playback speed |
| `i` | diagnostics |
| `q` | quit |

Audio is muted at any speed other than 1×: `AlsaSink` plays at the hardware
rate and there is no pitch-preserving time-stretch in this tree, so the
alternative is chipmunks. The picture stays locked either way — the sink
writes `frames / |rate|` frames of silence, which keeps its clock, and
therefore the video, at wall rate.

## Flags

| | |
|---|---|
| `--no-audio` | video only |
| `--repeat` (`--loop`) | loop gaplessly inside the demuxer |
| `--hwdec auto\|off` | whether to try the GPU video engine |
| `--decode-ahead N` | finished frames the decoder may run ahead |
| `--dav1d-threads N` | 0 picks by resolution |
| `--frame-delay N` | dav1d frame-delay bound; 0 leaves its own choice |
| `--channel-capacity N` | executor link capacity |
| `--no-sync` | decode as fast as possible; the batch-decode floor |
| `--headless` | play with no window at all |
| `--exit-after N` | quit after N seconds, printing the summary |

## The machine-readable summary

`--exit-after` (in either mode) prints exactly one line to stdout:

```text
stats: position_ns=1900000000 dropped=0 presented=20 queued=0 late=0 full=0 stale=0 clipped=0
```

`position_ns` and `dropped` stay adjacent and in that order:
`scripts/decode_matrix.py` parses that pair. Everything after is frame
accounting — `late` is past the lateness budget, `stale` is superseded by a
seek, `full` is this host failing to drain fast enough, `clipped` is
out-of-segment pre-roll after an accurate seek. Logs go to stderr, so stdout
carries the line and nothing else.

`--headless` runs the same pipeline with this process as the host: it drains
the frame mailbox at the same due times, with the same lateness budget, that
the widget uses. It needs no compositor, no GPU and no sound card, which is
why CI runs it — but it exercises none of the shader, the upload or the
swapchain. `just play-headless` runs the real widget under weston for that,
and `just play-smoke` runs this.
