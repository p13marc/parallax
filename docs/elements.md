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
| `AppSink` (+ `AppSinkHandle`) | Pull buffers out of a pipeline into application code |
| `AutoVideoSink` `[display]` | Display video in a window (winit + softbuffer) |

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
| `RtspSrc` `[rtsp]` | RTSP client source (retina): DESCRIBE/SETUP/PLAY, TCP-interleaved or UDP, Basic/Digest auth |

## Flow — `elements::flow`

| Element | Description |
|---------|-------------|
| `Queue` | Bounded queue with backpressure, watermarks (`with_flow_control`, `with_water_marks`), leaky modes (`LeakyMode`) |
| `Queue2` | Network buffering: `stream` (memory ring), `download` (progressive file), `timeshift` (circular file); posts `Buffering` messages |
| `Tee` | 1-to-N fan-out (refcount clones, no copies) |
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
| `VideoScale` | Resize YUV420 frames (`ScaleMode`) |
| `VideoConvertElement` | Pixel-format conversion (see [formats.md](formats.md)) |
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
| `Throttle` | Drop buffers arriving faster than a rate |
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
| `StreamIdDemux` | Route buffers by `stream_id` metadata |

N-to-1 synchronization (PTS alignment across pads) is provided by `element::muxer::MuxerSyncState` — see [pipeline.md](pipeline.md#muxing-n-to-1).

## Codecs — `elements::codec`

Codec traits: `VideoEncoder`/`VideoDecoder` and `AudioEncoder`/`AudioDecoder` (all with `flush()` for EOS draining), wrapped into pipeline elements by `EncoderElement`/`DecoderElement`/`AudioEncoderElement`/`AudioDecoderElement`. Some codecs implement `Element` directly instead (noted below).

| Codec | Types | Feature | Notes |
|-------|-------|---------|-------|
| H.264 | `H264Encoder`, `H264Decoder` (impl `Element` directly) | `h264` | OpenH264 (BSD-2); needs a C++ compiler |
| H.264 hardware encode | `V4l2M2mH264Encoder` (impl `VideoEncoder`; wrap: `EncoderElement::new(enc, VideoFormat)`) | `v4l2-m2m` | V4L2 M2M stateful encoder (RPi, i.MX, Rockchip…); locate with `find_m2m_encoder(b"H264")`; building needs libclang + kernel headers; VAAPI backend planned |
| AV1 encode | `Rav1eEncoder` (impl `VideoEncoder`; wrap: `EncoderElement::new(enc, VideoFormat)`) | `av1-encode` | rav1e, pure Rust; install nasm for SIMD |
| AV1 decode | `Dav1dDecoder` (impl `Element`) | `av1-decode` | libdav1d system library |
| FLAC/MP3/AAC/Vorbis decode | `SymphoniaDecoder` (impl `Element`) | `audio-flac`/`-mp3`/`-aac`/`-vorbis` | Symphonia, pure Rust |
| Opus | `OpusEncoder::new(rate, ch, bitrate, OpusApplication)`, `OpusDecoder` (impl audio traits) | `opus` | libopus; 48 kHz frame sizes 120–2880 samples |
| AAC encode | `AacEncoder` (impl `AudioEncoder`) | `aac-encode` | FDK-AAC — **license restrictions for commercial use** |
| JPEG | `JpegEncoder` / `JpegDecoder` | `image-jpeg` | zune-jpeg + jpeg-encoder, pure Rust |
| PNG | `PngEncoder` / `PngDecoder` | `image-png` | png crate, pure Rust |
| GPU H.264 decode | `HwDecoderElement` | `vulkan-video` | **experimental scaffold** — does not perform real hardware decode yet |

## Devices — `elements::device`

Gated by any of `pipewire`, `libcamera`, `v4l2`, `alsa`. Backend detection/enumeration helpers: `detect_video_backend`, `enumerate_video_devices`, `detect_audio_backend`, `enumerate_audio_devices`.

| Element | Feature | Capabilities |
|---------|---------|--------------|
| `V4l2Src` | `v4l2` | Camera capture; **DMA-BUF export** (`V4l2Config { dmabuf_export: true }` → `ProduceResult::OwnDmaBuf`); configurable frame rate (`framerate: Some((30, 1))`, clamped rate read back via `framerate()`); V4L2 monotonic timestamps |
| `LibCameraSrc` | `libcamera` | Modern camera API (Raspberry Pi, embedded, UVC); configurable frame rate via `FrameDurationLimits` (best-effort — UVC pipelines may ignore it); PTS from `SensorTimestamp` |
| `PipeWireSrc` / `PipeWireSink` | `pipewire` | Audio/video via PipeWire; PTS from `spa_meta_header` |
| `ScreenCaptureSrc` | `screen-capture` | XDG portal ScreenCast (Wayland-safe); cursor modes, session restore tokens |
| `AlsaSrc` / `AlsaSink` | `alsa` | Audio capture/playback; hardware timestamps; `AlsaSink` **provides a hardware clock** (priority 100) that the pipeline auto-selects |

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

Not elements themselves, but the engines behind `VideoConvertElement`/`AudioConvertElement`/`AudioResampleElement`: `VideoConvert` (YUV↔RGB, SIMD with feature `simd-colorspace`), `AudioConvert`, `AudioResample`, `VideoScale`. See [formats.md](formats.md#converters).
