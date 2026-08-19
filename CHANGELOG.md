# Changelog

Notable changes to `parallax-pipeline`. Format loosely follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions are
[semver](https://semver.org/), with the pre-1.0 caveat that breaking changes
ride minor bumps.

Earlier releases predate this file — their notes live in the `Bump to X.Y.Z`
commit messages and the Forgejo releases.

## [0.7.0] — 2026-08-19

175 commits since 0.6.0. Every out-of-tree plugin must be rebuilt: the plugin
ABI went **1 → 11** and the shared-arena format **v4 → v5**.

### Breaking

**Plugin ABI 1 → 11.** Element instances cross the boundary as double-boxed
`DynAsyncElement`, so any method added to `AsyncElementDyn` changes the vtable.
Ten bumps this cycle: `set_output_budget` (#84), `process_demux` (#76),
`EventResult::Handled` gaining a landing position, `EventResult::Forward` +
`seek_translations`, the inline fast-path slots (#175), `take_upstream_event`
(#173), `latency` (#184), `set_negotiated_memory` (#145), `retained_buffers` +
`passthrough` (#189), and `MemoryType::External` (#194).

**Arena format v4 → v5** (#177). `validate()` rejects v4. The byte layout is
unchanged; the bump guards the *access pattern* — each `SlotHeader` is now one
`AtomicU64` acquired with a single CAS, and a v4 peer CASing the state
half-word alone would race the v5 orphan sweep back into double allocation.

**IPC wire format** (#179). `ControlMessage` is incompatible with pre-#179
peers. The per-buffer path moved to a shared-memory descriptor ring with
eventfd doorbells; the socket now carries only registration and shutdown.

**Memory**
- `MemoryHandle` is a `#[non_exhaustive]` enum — `Cpu` | `DmaBuf` |
  `External` (#145, #194). `new`/`with_len` still build `Cpu`. `as_mut_slice`,
  `slot`, `ipc_ref` and the `arena_*` accessors return `Option`;
  `as_bytes_mut` panics on a read-only mapping, `try_as_bytes_mut` is the
  checked form.
- `ProduceResult::OwnDmaBuf` and `DmaBufBuffer` are gone — a dmabuf frame is
  `OwnBuffer` carrying `MemoryHandle::DmaBuf`, and `Buffer::copy_to_cpu`
  replaces `DmaBufBuffer::to_buffer`.
- `EventFd` moved from `pipeline::rt_bridge` to `memory` (#180).
- `MemorySegment`, `IpcHandle`, `MappedFileSegment`, `HugePageSegment`,
  `AtomicBitmap`, `MemorySrc`, `MemorySink`, `SharedMemorySink` deleted, along
  with the `huge-pages` and `rdma` features — nothing consumed them (#186).

**Pipeline**
- The `Queue` element is gone: the link *is* the queue (#167). `FlowPolicy`,
  `FlowStats` and `Source::handle_flow_signal` went with it, and `FlowSignal`
  narrows to `Ready | Busy`.
- `LinkPolicy::Drop` is now `DropNewest`, with no deprecated alias, and
  `DropOldest` joins it (#169).
- Fan-in into anything that is not a muxer is a **link-time error** (#183).
  Graphs that linked two producers into one sink used to compile and silently
  drop one of them.
- `parallax::link` deleted — `IpcPublisher`, `IpcSubscriber`, `NetworkSender`,
  `NetworkReceiver`. Use the IPC and network *elements* (#179).

**Elements**
- `AppSink` is an `AsyncSink` only: `add_sink` becomes `add_async_sink`
  (#168). `IpcSrc` likewise becomes `add_async_source` (#179), and
  `IpcSink::with_max_pending` is `with_capacity`.
- The `VideoDecoder` trait and `DecoderElement` are gone; video decoders
  implement `Element` directly (#160). The legacy `"width"`/`"height"` custom
  metadata keys carry no geometry any more — `Metadata::set_video_dims` and
  `video_dims()` are the representation.
- `Gain::process` requires `MediaFormat::AudioRaw` on every buffer and errors
  without it, at any factor including unity; mute writes per-format digital
  silence.
- Pull results are a `Pulled { Buffer, Empty, Flushing, Ended(EndReason) }`
  rather than a `Result`; `is_eos()` is replaced by `end_reason()`/`ended()`
  (#85).
- `Error::Shutdown` and `Error::Panic` added. `Shutdown` is a cooperative
  quit ending the run as `EndReason::Eos` — `AutoVideoSink` returns it on
  window close, so string-matching `"Display window closed"` must stop
  (#191, #85).
- `SeekableSource`, `SeekRequest`, `Event::LatencyQuery` removed;
  `MessageKind::Qos` redesigned to match the new `QosEvent` (#184).
- MP4 `seek_to_time`/`seek_all_to_time` take a `SeekSnap` (#166).
- `kanal` dropped for `tokio::sync::mpsc`; `MetadataExtract::new` returns a
  tokio `UnboundedReceiver`.

**VA-API** (new this cycle, so this is scoping rather than a break): HEVC
hardware decode is **not built**. `cros-codecs`' `h265` feature does not
compile against its published release, and a driver without an H.264 decode
config gets no VA-API at all, because the dependency initialises every backend
with a hardcoded H.264 config and panics without one. Both are upstream
defects tracked in #200; `[patch.crates-io]` used to hide them from us alone
and was removed, since cargo strips it from the published manifest (#202).

### Added

- **VA-API hardware decode** (#193) — H.264, VP8 and VP9 on the GPU video
  engine, each verified bit-exactly against its software counterpart, with the
  decoded dma-buf handed to the display instead of copied (#62). 1080p H.264:
  0.54 → 0.12 CPU cores, arena 88 MB → 34 MB.
- **`parallax-launch`** (#187, feature `cli`) — the gst-launch equivalent,
  plus factory registration for ~65 element names.
- **GPU presentation** (#190, feature `display-gpu`) — wgpu backend taking
  I420/NV12 and doing colorspace conversion and letterbox scaling in a shader.
- **`parallax-player`** — a real player over the engine: A/V sync, pause,
  seek, position/duration, volume, container and codec dispatch.
- **Seeking, end to end** (#162–#166, #173, #183) — seekability queries and
  gating, hop-by-hop upstream dispatch, seek translation for push-mode
  demuxers, `SNAP_BEFORE`/`SNAP_AFTER`, `ACCURATE` with iterative refinement,
  SEGMENT seeks with `SegmentDone`, keyframe-only fast-forward and reverse,
  instant rate change, and segment discipline throughout.
- **Zero-copy memory** — `MemoryType::External` with per-buffer `PlaneLayout`
  so producers keep their own strided frames (#194), strided input across the
  converters, scaler and encoders (#196), dmabuf flow-through (#145),
  executor-sized output arenas with admission control (#84, #91, #189), a
  release-queue doorbell (#180) and orphaned-slot recovery (#177).
- **Cross-process IPC on a descriptor ring** (#179) — zero-alloc hot path,
  eventfd doorbells, start-order independence.
- **`HttpCacheSrc`** (#188) — ranged HTTP download fused with a sparse
  write-through cache, so a backward seek into a downloaded span is served
  from disk. Queue2 gained buffering-aware seeking and downloaded ranges
  (#164).
- **QoS and latency** (#184) — sink-originated `Event::Qos` travelling
  upstream and mirrored to the bus, plus static latency aggregation.
- **Runtime pause/resume/position** (#71) and terminal-outcome reporting via
  `PipelineHandle::ended()`.
- New codecs and containers: VP8/VP9 (`vpx`), Vorbis, AAC and AC-3/E-AC-3
  (`eac3`) decoders, Matroska/WebM demux (`mkv-demux`), AV1 and Opus tracks in
  MP4, audio downmix.

### Fixed

- A fatal element error now winds the whole run down instead of letting
  siblings drain to EOS (#191).
- A panicking probe or tracer callback is caught, logged and unregistered
  rather than killing the pipeline (#92).
- Adapters drain the author `flush()` until `None`, so EOS no longer truncates
  held frames.
- `PipeWireSrc`/`PipeWireSink` never ran at all — an inverted shutdown check.
- `aac-encode` did not compile, against any published `fdk-aac`, and no CI job
  or `just` recipe built it. Fixed, and `aac-encode`, `websocket`, `alsa` and
  `v4l2-m2m` now have a CI job of their own.
- `screen-capture` ported to ashpd 0.13.

## [0.6.0] — 2026-08-08

Breaking: `GpuMemory::free` removed (GPU buffers free on `Drop` — a manual
free would double-free); `HwVideoDecoder` gained a required `read_frame`;
`HwDecoderElement::with_dimensions` removed per the geometry-in-Metadata
invariant; `h264_std::std_sps`/`std_pps` return the scaling lists alongside
the struct.

Also: `HwDecoderElement` bridges decoded pixels into pipeline buffers (#57),
`Mp4Demux` emits decodable Annex-B with in-band SPS/PPS (#58), `frame_num`
gaps error-then-skip-until-IDR instead of decoding corrupt (#59),
`AppSrcHandle::try_push_buffer` (#60), H.264 scaling matrices are marshalled
rather than rejected (#61).

0.4.0 and 0.5.0 were never published to crates.io, so 0.6.0 carried
everything since 0.3.0.

[0.7.0]: https://github.com/p13marc/parallax/releases/tag/0.7.0
[0.6.0]: https://github.com/p13marc/parallax/releases/tag/0.6.0
