# Element Catalog

Complete reference of built-in elements, organized by module under `src/elements/`. Feature gates are shown as `[feature]`; unmarked elements are always available (default features).

For the element *trait system* (how to write your own), see [getting-started.md](getting-started.md) and [api.md](api.md).

## I/O — `elements::io`

| Element | Description |
|---------|-------------|
| `FileSrc` | Reads a file in chunks; seekable (`SeekableSource`), reports position/duration |
| `FileSink` | Writes buffers to a file |
| `FdSrc` / `FdSink` | Read/write a raw file descriptor |
| `ConsoleSink` | Debug-prints buffers (`ConsoleFormat`: text/hex/…) |

## Testing — `elements::testing`

| Element | Description |
|---------|-------------|
| `TestSrc` | Test-pattern byte buffers (`TestPattern`) |
| `VideoTestSrc` / `AsyncVideoTestSrc` | Video test patterns (`VideoPattern`), RGBA output |
| `DataSrc` | Emits buffers from inline data |
| `NullSource` / `NullSink` | Produce empty buffers / discard everything |

## Application integration — `elements::app`

| Element | Description |
|---------|-------------|
| `AppSrc` (+ `AppSrcHandle`) | Push buffers from application code into a pipeline |
| `AppSink` (+ `AppSinkHandle`) | Pull buffers out of a pipeline into application code — see [Reading the end of a stream](#reading-the-end-of-a-stream) |
| `AutoVideoSink` `[display]` | Display video in a window (winit + softbuffer); frame dimensions from per-buffer metadata (`Metadata::video_dims`, both conventions) when present, else guessed from RGBA buffer size. `sync=true` paces presentation — see below |

### Reading the end of a stream

A pull returns a `Pulled`, not an `Option`, because there are four distinct ways
one can end and a consumer needs to act differently on each:

```rust,ignore
match handle.pull_buffer_timeout(Duration::from_millis(100)).await {
    Pulled::Buffer(buf) => process(buf),
    Pulled::Empty      => {}                    // timed out; the stream is still live
    Pulled::Flushing   => {}                    // app-initiated, recoverable
    Pulled::Ended(EndReason::Eos)        => break,
    Pulled::Ended(EndReason::Error(err)) => return Err(err),
}
```

`EndReason` is sticky and first-writer-wins, so a clean EOS arriving after a
failure cannot hide it. Buffers queued before an element died are still valid and
are handed over **before** the end is reported.

Two ways to observe termination, and the difference matters:

- `end_reason()` — `Some` as soon as the reason is known, even with buffers still
  queued. Use it to react early (tear down, report status).
- `ended()` — resolves only once the queue is *also* drained, i.e. exactly when a
  pull would return `Pulled::Ended`. That is what makes it safe to `select!`
  against `pull_buffer()` without losing frames:

```rust,ignore
loop {
    tokio::select! {
        pulled = handle.pull_buffer() => match pulled {
            Pulled::Buffer(b) => process(b),
            Pulled::Ended(reason) => break reason,
            _ => continue,
        },
        reason = handle.ended() => break reason,
        _ = shutdown.cancelled() => break EndReason::Aborted,
    }
}
```

No polling, and no watchdog: an element that panics is caught and reported as
`EndReason::Error` like any other failure.

`EndReason` lives in `parallax::pipeline` (re-exported unchanged from
`parallax::elements`) because it is the whole graph's answer as much as one
sink's: `PipelineHandle::ended()` is the pipeline-level twin, with the same three
outcomes. A sink only ever sees `Eos` or `Error` — an aborted task cannot deliver
anything, so `Aborted` comes from the handle alone.

## Network — `elements::network`

| Element | Description |
|---------|-------------|
| `TcpSrc` / `TcpSink` / `AsyncTcpSrc` / `AsyncTcpSink` | TCP client/server streaming (`TcpMode`) |
| `UdpSrc` / `UdpSink` / `AsyncUdpSrc` / `AsyncUdpSink` | UDP datagrams |
| `UnixSrc` / `UnixSink` / `AsyncUnixSrc` / `AsyncUnixSink` | Unix domain sockets (`UnixMode`) |
| `UdpMulticastSrc` / `UdpMulticastSink` | Multicast group receive/send |
| `HttpSrc` `[http]` | HTTP GET source |
| `HttpSink` `[http]` | HTTP POST/PUT sink |
| `WebSocketSrc` / `WebSocketSink` `[websocket]` | WebSocket message I/O |
| `ZenohSrc` / `ZenohSink` `[zenoh]` | Zenoh subscribe/publish on key expressions |
| `ZenohQueryable` / `ZenohQuerier` `[zenoh]` | Zenoh query handling / querying |

## RTP / RTCP / RTSP — `elements::rtp` `[rtp]`

| Element | Description |
|---------|-------------|
| `RtpSrc` / `RtpSink` / `AsyncRtpSrc` / `AsyncRtpSink` | RTP over UDP receive/send |
| `RtpJitterBuffer` / `AsyncJitterBuffer` | Reordering + loss detection (`JitterBufferConfig`) |
| `RtcpHandler` | RTCP sender/receiver reports |
| `RtpH264Pay` / `RtpH264Depay` | H.264 payloading (RFC 6184) |
| `RtpH265Pay` / `RtpH265Depay` | H.265/HEVC payloading |
| `RtpVp8Pay` / `RtpVp8Depay`, `RtpVp9Pay` / `RtpVp9Depay` | VP8/VP9 payloading |
| `RtpOpusDepay` | Opus depayloading (no payloader yet) |
| `RtspSrc` `[rtsp]` | RTSP client source (retina): DESCRIBE/SETUP/PLAY, TCP-interleaved or UDP, Basic/Digest auth (also lifted from `rtsp://user:pass@host/...` URLs), `connect_timeout` enforced on each RTSP operation |

`RtspSrc` output framing is controlled by `RtspFrameFormat`: the default
`AnnexB` is self-describing (start codes, SPS/PPS prepended to every keyframe,
ADTS-wrapped AAC), so frames feed `H264Decoder`, typefind, or a raw file dump
directly; `LengthPrefixed` emits 4-byte-length NALs for MP4 muxing. See
`examples/57_rtsp_capture.rs` (record a playable `.h264` file) and
`examples/58_rtsp_display.rs` (decode and display in a window via
`AppSrc → H264Decoder → VideoConvert → AutoVideoSink`). A local test stream is
one command away: `just rtsp-server` (= `scripts/rtsp_test_server.py`).

**Dimensions can arrive after `connect()`.** Many cameras ship an SDP with no
`a=framesize` and no usable `sprop-parameter-sets`, leaving
`StreamInfo::dimensions` as `None`. The stream still announces its geometry in the
first in-band SPS, and parallax adopts it as soon as it arrives — usually within
one keyframe — so `None` is a "not yet", not a permanent property of the stream.

Because `RtspSession` is an `AsyncSource`, adding it to a pipeline moves it and
`session.streams()` becomes unreachable. Take a handle first, exactly as you would
a control handle:

```rust,ignore
let session = RtspSrc::new(url).connect().await?;
let info = session.stream_info_handle();      // BEFORE the move
pipeline.add_async_source("rtsp", session);

// Resolves at once if the SDP had geometry, otherwise one keyframe later.
// `None` means the session ended first.
let (w, h) = info.wait_for_dimensions(0).await.ok_or("session ended")?;
```

`RtspStreamInfoHandle` also has `streams()` and `stream(index)` for a snapshot.

## Flow — `elements::flow`

| Element | Description |
|---------|-------------|
| `Queue` | Bounded queue with backpressure, watermarks (`with_flow_control`, `with_water_marks`), leaky modes (`LeakyMode`) |
| `Queue2` | Network buffering: `stream` (memory ring), `download` (progressive file), `timeshift` (circular file); posts `Buffering` messages |
| `Inspect` | 1-in/1-out passthrough counter (buffers/bytes). **Not** a fan-out — it was called `Tee` and never was one. Fan-out needs no element: link one src-pad to several sinks (see [pipeline.md](pipeline.md#fan-out)). `tee` survives as a deprecated parse alias |
| `Funnel` | N-to-1 merge (`FunnelInput` handles) |
| `InputSelector` / `OutputSelector` | Switch between N inputs / route to one of N outputs |
| `Concat` | Sequential stream concatenation |
| `Valve` | On/off flow gate (`ValveControl`) |

## Transforms — `elements::transform`

| Element | Description |
|---------|-------------|
| `Map` / `FilterMap` / `FlatMap` / `Chunk` | Functional buffer transforms; 1→N; fixed-size splitting |
| `Filter` / `SampleFilter` / `MetadataFilter` | Predicate, statistical (every-Nth/random/first-N), and metadata-based filtering |
| `DuplicateFilter` / `RangeFilter` / `RegexFilter` | Dedup by content hash; size/sequence ranges; regex match |
| `Batch` / `Unbatch` | Aggregate N buffers into one and back |
| `BufferTrim` / `BufferSlice` / `BufferPad` | Trim to max / extract range / pad to min |
| `BufferSplit` / `BufferJoin` / `BufferConcat` | Split/join at delimiters; concatenate |
| `Gain` | RT-safe audio gain (PCM multiply) |
| `VideoScale` | Resize frames in **any** format the scaler engine supports (I420, NV12, RGB24, BGR24, RGBA, BGRA, Gray8, YUYV, UYVY) — it reads the pixel format from the buffer and errors if the buffer does not declare one. `ScaleMode` picks the filter; source geometry comes from the buffer, target is retargetable at runtime via `ScaleControl` (see [Runtime control](#runtime-control-bandwidth-knobs)). Target == source is a zero-copy passthrough |
| `VideoConvertElement` | Pixel-format conversion (see [formats.md](formats.md)); format and dimensions from buffer metadata, then `with_input_format`/`with_size`, then buffer-size auto-detection — and re-negotiated when they change mid-stream |
| `AudioConvertElement` | Sample-format conversion (S16 ↔ F32, …) |
| `AudioResampleElement` | Sample-rate conversion |
| `SequenceNumber` / `Timestamper` | Stamp sequence numbers / timestamps (`TimestampMode`) |
| `MetadataInject` / `MetadataExtract` | Inject stream id/duration/offset; extract metadata to a sideband channel |
| `TimestampDebug` | Log/collect PTS statistics (missing, backwards, jitter); `TimestampDebugLevel` |

## Timing — `elements::timing`

| Element | Description |
|---------|-------------|
| `Delay` / `AsyncDelay` | Fixed delay per buffer |
| `Timeout` | Emit fallback data when upstream stalls |
| `Debounce` | Suppress rapid bursts |
| `Throttle` | Drop buffers arriving faster than a rate; rate tunable at runtime via `ThrottleControl` (rate 0 drops everything) |
| `RateLimiter` | Limit throughput (`RateLimitMode`) |

## Utility — `elements::util`

| Element | Description |
|---------|-------------|
| `PassThrough` | Identity (also a parse-string element: `passthrough`) |
| `Identity` | Pass-through with inspection callbacks (`IdentityStats`) |

## IPC & memory — `elements::ipc`

| Element | Description |
|---------|-------------|
| `IpcSrc` / `IpcSink` | Zero-copy cross-process transport (shared arena + Unix socket, fd passing) |
| `MemorySrc` / `MemorySink` | In-memory source / collecting sink |
| `SharedMemorySink` | Thread-safe collecting sink |

## Mux / Demux — `elements::mux`, `elements::demux`

| Element | Description |
|---------|-------------|
| `TsMux` / `TsMuxElement` `[mpeg-ts]` | MPEG-TS muxer (config: `TsMuxConfig`/`TsMuxTrack`/`TsMuxStreamType` incl. H.264, AAC, KLV private data); `TsMuxElement` is the pipeline-ready `Muxer`; helpers `create_av_klv_muxer`, `create_video_klv_muxer` |
| `TsDemux` `[mpeg-ts]` | MPEG-TS demuxer (programs, elementary streams) |
| `Mp4Mux` / `Mp4MuxTransform` / `Mp4FileSink` `[mp4-demux]` | MP4 muxing as `Muxer`, transform, or all-in-one file sink; video+audio track configs |
| `Mp4Demux` `[mp4-demux]` | MP4/MOV demuxer (tracks, samples, codec info) |
| `MkvDemux` `[mkv-demux]` | Matroska/WebM demuxer (pure Rust via `matroska-demuxer`); source-style `Demuxer` with `video`/`audio` pads, cue-based seek with linear fallback, H.264 → Annex-B conversion; subtitle tracks listed but not routed |
| `StreamIdDemux` | Route buffers by `stream_id` metadata |

N-to-1 synchronization (PTS alignment across pads) is provided by `element::muxer::MuxerSyncState` — see [pipeline.md](pipeline.md#muxing-n-to-1).

## Codecs — `elements::codec`

Codec traits: `VideoEncoder`/`VideoDecoder` and `AudioEncoder`/`AudioDecoder` (all with `flush()` for EOS draining), wrapped into pipeline elements by `EncoderElement`/`DecoderElement`/`AudioEncoderElement`/`AudioDecoderElement`. Some codecs implement `Element` directly instead (noted below).

| Codec | Types | Feature | Notes |
|-------|-------|---------|-------|
| H.264 | `H264Encoder`, `H264Decoder` (impl `Element` directly) | `h264` | OpenH264 (BSD-2); needs a C++ compiler. Live bitrate/GOP/QP via `EncoderControl`; resolution follows the buffer's metadata. Knobs: `rate_control`, `skip_frames`, `max_slice_len`, `profile`, `complexity`, `usage_type` |
| H.264 hardware encode | `V4l2M2mH264Encoder` (impl `VideoEncoder`; wrap: `EncoderElement::new(enc)`) | `v4l2-m2m` | V4L2 M2M stateful encoder (RPi, i.MX, Rockchip…); locate with `find_m2m_encoder(b"H264")`; building needs libclang + kernel headers; VAAPI backend planned |
| AV1 encode | `Rav1eEncoder` (impl `VideoEncoder`; wrap: `EncoderElement::new(enc)`) | `av1-encode` | rav1e, pure Rust; install nasm for SIMD |
| AV1 decode | `Dav1dDecoder` (impl `Element`) | `av1-decode` | libdav1d system library |
| VP8/VP9 decode | `VpxDecoder::vp8()/vp9()` (impl `Element`) | `vpx` | libvpx system library (`libvpx-devel`; bindings generated via bindgen, needs libclang at build time); I420 output, display-order 1:1 timestamps, mid-stream resolution changes |
| FLAC/MP3/AAC/Vorbis decode | `SymphoniaDecoder` (impl `Element`) | `audio-flac`/`-mp3`/`-aac`/`-vorbis` | Symphonia, pure Rust |
| Vorbis streaming decode | `VorbisDecoder::from_codec_private` (impl `AudioDecoder`; wrap: `AudioDecoderElement`) | `audio-vorbis` | Symphonia packet-level; takes the Xiph-laced Matroska CodecPrivate directly |
| Opus | `OpusEncoder::new(rate, ch, bitrate, OpusApplication)`, `OpusDecoder` (impl audio traits) | `opus` | libopus; 48 kHz frame sizes 120–2880 samples |
| AAC encode | `AacEncoder` (impl `AudioEncoder`) | `aac-encode` | FDK-AAC — **license restrictions for commercial use** |
| JPEG | `JpegEncoder` / `JpegDecoder` | `image-jpeg` | zune-jpeg + jpeg-encoder, pure Rust |
| PNG | `PngEncoder` / `PngDecoder` | `image-png` | png crate, pure Rust |
| GPU H.264 decode | `HwDecoderElement` | `vulkan-video` | **experimental scaffold** — does not perform real hardware decode yet |

### Playing at the stream's speed, not the decoder's

`AutoVideoSink` blits every frame the moment it arrives, so playback speed equals
decode speed — right for a camera preview, wrong for a media player. `sync=true`
(off by default) makes it hold each frame until its PTS comes round on the pipeline
clock:

```rust,ignore
Pipeline::parse("filesrc location=clip.h264 ! … ! autovideosink sync=true max-lateness-ms=40")?
// or programmatically
AutoVideoSink::new().with_sync(true).with_max_lateness(Duration::from_millis(40))
```

Three things to know:

- **The wait is blocking, and that is the mechanism.** Holding the sink's task is
  what back-pressures the decoder and the source down to real time.
- **Late frames are dropped, not shown.** Past `max-lateness-ms` (default 40 ms —
  one frame at 25 fps) the frame is shed and counted via `parallax_buffers_dropped`,
  because showing it would also cost the *next* frame its slot.
- **It degrades rather than stalling.** No clock, no PTS, or `sync=false` all keep
  the historical blit-on-arrival path, and a PTS that jumps more than a second into
  the future is presented instead of slept through.

The first frame anchors the stream — its PTS is pinned to the running time at which
it arrived — so a stream whose timestamps do not start at zero plays immediately
instead of stalling for its start offset. A PTS that goes *backwards* (a seek)
re-anchors rather than condemning every later frame to look eternally late.

## Reading the clock from an element

The executor selects the pipeline clock before start (highest-priority
`ClockProvider` wins) and hands it to **every** element, so both ends of a graph can
schedule against the same time base.

| Context | Accessors |
|---------|-----------|
| `ProduceContext` (sources) | `clock()`, `running_time()`, `clock_time()`, `base_time()` |
| `ConsumeContext` (sinks) | `clock()`, `running_time()`, `clock_time()`, `base_time()` |

`running_time()` is the one to compare a buffer's PTS against — it is `clock_time()`
minus the base time captured when the pipeline started. Both return `ClockTime::NONE`
when no clock is available (the pipeline was never started, or a unit test built the
context by hand), and a sink that paces presentation must treat that as "consume as
fast as buffers arrive" rather than stalling:

```rust,ignore
fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
    let (Some(_), Some(pts)) = (ctx.clock(), ctx.metadata().pts.to_option()) else {
        return self.present(ctx.input());     // no clock, or no timestamp: blit now
    };
    let now = ctx.running_time();
    // ...wait until `pts`, or drop the frame if it is already too late
}
```

## Runtime control (bandwidth knobs)

Everything here lives in one module: **`parallax::control`**.

`Executor::start()` **moves** each element into its executor task, so
`pipeline.get_element_mut()` returns `None` for anything that is running. To change an
element while it runs you must clone a **control handle from it before `start()`**. The handle
is an `Arc<Atomic…>` — lock-free, allocation-free, safe to call from any thread or task, and
free on the hot path when nothing has changed.

Every controllable element implements `Controllable`, so the accessor is always `control()`.

| Handle | From | Changes |
|--------|------|---------|
| `EncoderControl` | `H264Encoder::control()`, `EncoderElement::control()` | `set_bitrate`, `set_keyframe_interval`, `set_qp`, `set_rate_control`, `set_skip_frames`, `request_keyframe` |
| `EncoderStatsHandle` | `H264Encoder::stats()`, `EncoderElement::stats()` | *read-only*: `frames_encoded`, `bytes_encoded`, `frames_dropped_by_rc`, `last_encode_ns` |
| `KeyframeHandle` | `…::keyframe_handle()` | `request()` — force the next frame to be an IDR |
| `ScaleControl` | `VideoScale::control()` | `set_target(w, h)`, `set_max_height(h)` / `set_max_width(w)` (aspect-preserving, never upscale), `passthrough()`, `resolve(src_w, src_h)` |
| `ThrottleControl` | `Throttle::control()` | `set_rate(fps)`, `set_min_interval(d)` |
| `JpegQualityControl` | `JpegEncoder::control()` | `set_quality(1..=100)` |
| `ValveControl` | `Valve::control()` | `open()` / `close()` |
| `GainControl` | `Gain::control()` | `set_factor(f)` / `set_db(db)`, `factor()` / `db()` |
| `FlowStateHandle` | `Queue::control()` | backpressure signalling to live sources |
| `AppSinkHandle` / `AppSrcHandle` | `AppSink::handle()` / `AppSrc::handle()` | async-first `pull_buffer`/`push_buffer` (+`_timeout`; `*_blocking` twins for plain threads), `try_pull_buffer`/`try_push_buffer`, `stats()` |
| `AutoVideoSinkHandle` | `AutoVideoSink::handle()` `[display]` | window events (`try_event`/`event_timeout` → `VideoWindowEvent`: keys as `VideoKey`, mouse, close, resize; bounded and sender-side lossy), `set_fullscreen(bool)`, `is_open()`. No handle taken ⇒ no events |

```rust,ignore
use parallax::control::Controllable;

// No dimensions anywhere: geometry travels in-band, in Metadata.
let scaler  = VideoScale::new();
let encoder = H264Encoder::new(H264EncoderConfig::new().bitrate(4_000_000))?;

// Clone the handles BEFORE the pipeline starts.
let scale   = scaler.control();
let control = encoder.control();
let stats   = encoder.stats();

pipeline.add_filter("scale", scaler);
pipeline.add_filter("enc", encoder);
let handle = executor.start(&mut pipeline)?;   // sync — elements move into their tasks here

// ...later, on a viewer's request or a congested link:
scale.set_max_height(360);      // quarter the pixels, aspect preserved
control.set_bitrate(800_000);   // and a fifth of the bits — no IDR, the GOP survives
println!("{} frames, {} bytes", stats.frames_encoded(), stats.bytes_encoded());
```

Both changes take effect on the next frame, with no teardown.

**Predicting the scaled geometry.** If you advertise a stream's size ahead of the frames
— a catalogue, per-tier status, wire metadata — ask the scaler rather than reimplementing
its rounding:

```rust,ignore
let (w, h) = scale.resolve(src_width, src_height);   // exactly what process() will emit
```

`ScaleControl::resolve` is pure and `ScaleControl::new()` needs no element, so this works
without a pipeline. The contract is committed: the derived axis is computed with
`div_ceil`, **then** both axes round *down* to even, and a bound never upscales. Those
first two steps round in opposite directions, which is what makes hand-mirroring the
arithmetic go wrong — 1280×720 bounded to height 480 is **854**×480, not 852×480.

**What costs an IDR and what does not:**

- **A bitrate change is seamless.** It goes in through OpenH264's `SetOption(ENCODER_OPTION_BITRATE)`,
  so the GOP is not broken. Setting the bitrate to the value it already has does nothing at all.
- **Everything else rebuilds the encoder** (GOP length, QP, rate-control mode, frame skipping),
  and a rebuilt encoder leads with an IDR so decoders pick up the new parameter sets.
- **A resolution change rebuilds too**, and must: the encoder re-initialises, and the IDR is what
  makes the new size a clean decoder entry point.

Resolution travels **in-band**: `VideoScale` stamps the produced size into the buffer's metadata
(`Metadata::set_video_dims`), and `H264Encoder` encodes at the size the buffer declares. No
element takes dimensions at construction; one that cannot determine its geometry from the buffer
errors rather than falling back to a stale constructor value.

Things worth knowing:

- **Defaults.** `rate_control` defaults to `RateControlMode::Bitrate` and `bitrate_bps` to 2 Mbps
  — for a crate whose headline feature is live bandwidth control, "the bitrate is a hint" is the
  wrong default. `Bitrate` mode with a zero target is a hard error (OpenH264 would silently fall
  back to ~120 kbps and drop most frames).
- **Frame skipping is off by default.** OpenH264 holds a bitrate target by dropping frames —
  emitting *nothing* for some inputs, which quietly breaks every downstream fps/kbps figure.
  Parallax spends quality instead; use an upstream `Throttle` to shed frames deliberately.
  `EncoderStatsHandle::frames_dropped_by_rc()` counts any that rate control does swallow.
- **In `Bitrate` mode with skipping off**, `qp` is a quality *ceiling*, not a target: the encoder
  may fall below it to make budget. (With a tight ±4 band it simply misses the target instead.)
- **Hardware.** `V4l2M2mH264Encoder` accepts live `set_bitrate`/`set_keyframe_interval` (V4L2
  permits control changes at any time), but some drivers accept the ioctl and then ignore it
  mid-stream — GOP is the usual casualty. Best-effort; verify on your driver.

## Devices — `elements::device`

Gated by any of `pipewire`, `libcamera`, `v4l2`, `alsa`. Backend detection/enumeration helpers: `detect_video_backend`, `enumerate_video_devices`, `detect_audio_backend`, `enumerate_audio_devices`.

| Element | Feature | Capabilities |
|---------|---------|--------------|
| `V4l2Src` | `v4l2` | Camera capture; **DMA-BUF export** (`V4l2Config { dmabuf_export: true }` → `ProduceResult::OwnDmaBuf`); configurable frame rate (`framerate: Some((30, 1))`, clamped rate read back via `framerate()`); V4L2 monotonic timestamps |
| `LibCameraSrc` | `libcamera` | Modern camera API (Raspberry Pi, embedded, UVC); configurable frame rate via `FrameDurationLimits` (best-effort — UVC pipelines may ignore it); PTS from `SensorTimestamp` |
| `PipeWireSrc` / `PipeWireSink` | `pipewire` | Audio/video via PipeWire; PTS from `spa_meta_header` |
| `ScreenCaptureSrc` | `screen-capture` | XDG portal ScreenCast (Wayland-safe); cursor modes, session restore tokens |
| `AlsaSrc` / `AlsaSink` | `alsa` | Audio capture/playback in S16/S32/F32/U8, each pinned in the element's caps so a mismatched upstream gets an `audioconvert` (or a clear `prepare()` error). `AlsaSink` **provides a hardware clock** (priority 150) that the pipeline auto-selects: its time base is frames the device actually consumed (`snd_pcm_delay`), so it stops when the device stalls and never runs backwards across an underrun |

All capture sources default to a `Drop` flow policy and accept `set_flow_state(handle)` for downstream backpressure — see [pipeline.md](pipeline.md#flow-control--backpressure).

Both camera sources stamp `Metadata.duration` with the configured frame duration; PTS is relative to the first captured frame.

### Hotplug monitoring — `DeviceMonitor` (feature `hotplug`)

`DeviceMonitor::new()` watches udev's `video4linux` subsystem on a background thread and emits `DeviceEvent::Added(VideoCaptureDevice)` / `DeviceEvent::Removed { id }` over an unbounded channel (async `recv().await`, `blocking_recv()`, or `try_recv()`). Only capture-capable nodes produce `Added` — metadata-only `/dev/video*` nodes are filtered out. With the `libcamera` feature also enabled, libcamera's own hotplug events are folded into the same stream (backend `LibCamera`, opaque libcamera ids), so one physical USB camera yields one `Added` per backend — filter on `VideoCaptureDevice::backend`. Event latency is ≤ 500 ms.

## Streaming output — `elements::streaming`

Not feature-gated (pure Rust).

| Element | Description |
|---------|-------------|
| `HlsSink` | HLS: MPEG-TS segments + M3U8 playlists; live (sliding window) or VOD; multi-variant master playlist for ABR (`HlsVariant`) |
| `DashSink` | MPEG-DASH: fragmented MP4 + MPD manifest; live or static; adaptation sets with multiple representations |
| `SegmentWriter` / `SegmentBoundaryDetector` | Keyframe-aligned segmentation helpers shared by both |

## Metadata — `elements::metadata`

| Element | Description |
|---------|-------------|
| `KlvEncoder` | KLV (Key-Length-Value) encoding for STANAG 4609 / MISB metadata (`KlvTag`, `Uls`, BER lengths) |
| `StanagMetadataBuilder` | Fluent builder for common MISB 0601 fields |

Attach KLV to buffers via `metadata.set_klv(bytes)` and mux it as a private data track in MPEG-TS (`TsMuxStreamType::Klv`).

## Converters — `parallax::converters`

Not elements themselves, but the engines behind `VideoConvertElement`/`AudioConvertElement`/`AudioResampleElement`: `VideoConvert` (YUV↔RGB, SIMD with feature `simd-colorspace`), `AudioConvert`, `AudioResample`, `ScaleEngine` (behind the `VideoScale` element). See [formats.md](formats.md#converters).
