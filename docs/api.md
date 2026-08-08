# API Overview

A map of the public API surface. For complete signatures and doc comments, run `cargo doc --open` — every public item is documented (`#![warn(missing_docs)]` is enforced).

## Prelude

```rust
use parallax::prelude::*;
```

brings in: `Buffer`, `Metadata`/`BufferFlags`/`RtpMeta`, clock types (`Clock`, `ClockTime`, `ClockProvider`, `PipelineClock`, `SystemClock`), element traits (`Source`, `Sink`, `Element`, `Transform`, async variants, `DynAsyncElement`), `Event`/`PipelineItem`/`TagList`, format types (`Caps`, `MediaCaps`, `ElementMediaCaps`, `FormatMemoryCap`, …), `MemorySegment`/`MemoryType`, `Pipeline`/`Executor`, and `Error`/`Result`.

## Modules

### `parallax::pipeline`

| Item | Purpose |
|------|---------|
| `Pipeline` | The graph: `new()`, `parse(desc)`, `parse_with_factory`, `add_source[_with_arena/_with_pool]`, `add_sink`, `add_transform`, `add_filter`, `add_element`, `link`, `link_pads`, `get_element::<T>`, state methods (`prepare`/`activate`/`pause`/`suspend`), `run()`, `run_with_bus`, `start()`, seek/query surface, probes, `to_dot`/`to_json`, `select_clock`/`set_clock`, `set_converter_policy`, `create_pool_from_caps` |
| `Executor`, `ExecutorConfig`, `SchedulingMode`, `RtConfig` | Execution: strategy config, hybrid RT setup; `Executor::start` (sync) → `PipelineHandle`; `run` (async) |
| `PipelineHandle` | `wait()`, `abort()`, `subscribe()` (typed events), bus access |
| `PipelineState`, `ConverterPolicy`, `DotOptions` | State enum; Deny/Warn/Allow; DOT export options |
| `PipelineBuilder` | Typestate fluent builder (`source().then().tee().sink().build()`) |
| `bus::{Bus, BusHandle, BusStream, Message, MessageKind, BufferingMode}` | Element→app messaging; `poll`/`next().await`/`subscribe`/`into_stream` |
| `parser::{parse_pipeline, ParsedPipeline, PropertyValue}` | The string grammar (linear chains only) |
| `factory::{ElementFactory, PluginRegistry integration}` | Registered element names for `parse` |
| `probe::{ProbeType, ProbeReturn, ProbeData, PadRef, ProbeRegistry, ProbeId}` | Pad probes |
| `tracer::{Tracer, TracerRegistry, LatencyTracer, FramerateTracer, DropTracer, PipelineStats}` | Runtime tracing; `PARALLAX_TRACERS` |
| `seek::{SeekRequest, PositionQuery, DurationQuery, SeekableQuery, SeekableSource}` | Seeking |
| `typefind::{TypeFindRegistry, MediaType, TypeFindResult, TypeFindProbability}` | Content detection |
| `flow::{FlowSignal, FlowPolicy, FlowStateHandle, WaterMarks, FlowStats}` | Backpressure primitives |
| `driver::{TimerDriver, ManualDriver, DriverConfig, DriverStats}` | RT cycle pacing |
| `rt_bridge::{AsyncRtBridge, BridgeConfig, EventFd}` | Async↔RT boundary |
| `events::{PipelineEvent, EventSender, EventReceiver, EventStream}` | Typed event channel (distinct from the bus) |

### `parallax::element`

| Item | Purpose |
|------|---------|
| `Source`, `Sink`, `Element`, `Transform` | Sync author traits (ctx-based produce/consume; `process`/`transform`) |
| `AsyncSource`, `AsyncSink`, `AsyncTransform` | Async variants for real I/O |
| `Demuxer`, `Muxer`, `SyncElement` | 1-to-N (`RoutedOutput`, `PadId`), N-to-1 (`MuxerInput`), RT-path processing |
| `SimpleSource`, `SimpleSink`, `SimpleTransform` + `Src`/`Snk`/`Xfm` | Boilerplate-free element API (`ProcessOutput`) |
| `PipelineElement` / `SendPipelineElement` | Unified next-gen element trait |
| `ProduceContext`, `ConsumeContext`, `ProcessContext`, `ElementContext` | Contexts: pooled buffers, metadata setters, clock access, bus posting |
| `ProduceResult` | `Produced(n)` / `Eos` / `OwnBuffer` / `OwnDmaBuf` / `WouldBlock` |
| `Output`, `ProcessOutput`, `SourceResult` | Multi-output enums |
| `ExecutionHints`, `TrustLevel`, `ProcessingHint`, `LatencyHint`, `MemoryHint` | Scheduling hints (profiles: `rt_safe()`, `io_bound()`, …) |
| `*Adapter` (Source/Sink/Element/Transform/Muxer/…) | Wrap author traits into `DynAsyncElement` |
| `muxer::{MuxerSyncState, MuxerSyncConfig, SyncMode, PadInfo, StreamType, CollectedInput}` | PTS-based N-to-1 synchronization |
| `pad::{Pad, PadTemplate, PadDirection, PadPresence}` | Pad descriptions |

### `parallax::memory`

| Item | Purpose |
|------|---------|
| `SharedArena`, `SharedSlotRef`, `SharedIpcSlotRef`, `SharedArenaCache`, `ArenaMetrics` | memfd arena with cross-process refcounting |
| `BufferPool` (trait), `FixedBufferPool`, `PooledBuffer`, `PoolStats` | Pipeline buffer pools with backpressure |
| `DmaBufSegment`, `HugePageSegment`/`HugePageSize`, `MappedFileSegment` | Alternative segments |
| `MemorySegment` (trait), `MemoryType`, `IpcHandle` | Segment abstraction |
| `AtomicBitmap` | Lock-free slot bitmap utility |
| `ipc::{send_fds, recv_fds, send_segment_handle, recv_segment_handle}` | SCM_RIGHTS fd passing |
| `defaults::*` | Slot size/count constants for common media |

### `parallax::buffer` / `parallax::metadata`

`Buffer<T = ()>` (`new`, `as_bytes[_mut]`, `slice`, `metadata[_mut]`, `into_dynamic`), `MemoryHandle`, `DmaBufBuffer`; `Metadata` (pts/dts/duration/sequence/stream_id/flags + typed custom map: `set`/`get`/`get_mut`/`remove`/`set_bytes`/`set_klv`/`set_sei`), `BufferFlags`, `RtpMeta`.

### `parallax::format` / `parallax::negotiation`

`CapsValue<T>`, `VideoFormatCaps`/`AudioFormatCaps`/`FormatCaps`, `MemoryCaps`, `MemoryLayout`, `FormatMemoryCap`, `ElementMediaCaps`, `MediaCaps`, `Caps`; concrete `MediaFormat`/`VideoFormat`/`AudioFormat`/`PixelFormat`/`VideoCodec`/`AudioCodec`/`Framerate`/`RtpFormat`; `NegotiationSolver`, `ConverterRegistry`, `ConverterElement`, `builtin_registry()`. See [formats.md](formats.md).

### `parallax::event` / `parallax::clock`

`Event` (StreamStart/Segment/Tags/Eos/CapsChanged/Gap | Seek/Qos/LatencyQuery | FlushStart/FlushStop/Custom), `SegmentEvent` (`to_running_time`/`to_stream_time`), `SeekEvent`/`SeekFlags`, `PipelineItem`, `ControlSignal`, `TagList`/`TagValue`/`TagMergeMode`; `ClockTime`, `Clock`, `ClockFlags`, `ClockProvider`, `SystemClock`, `PipelineClock`.

### `parallax::elements`

The built-in element library — full catalog in [elements.md](elements.md).

### `parallax::converters`

`VideoConvert`, `AudioConvert`/`AudioChannelMix`/`ChannelLayout`, `AudioResample`/`ResampleQuality`, `VideoScale`/`ScaleAlgorithm`, plus this module's own `PixelFormat`/`SampleFormat`/`ColorMatrix` (distinct from `format::` — see [formats.md](formats.md#pitfalls)).

### `parallax::link`

`LocalLink` (kanal channels, `bounded`/`unbounded` → sender/receiver pairs), `IpcPublisher`/`IpcSubscriber` (shared-memory IPC), `NetworkSender`/`NetworkReceiver` (TCP + rkyv, framed with magic/version/CRC32).

### `parallax::plugin`

`PluginDescriptor`/`ElementDescriptor` (`#[repr(C)]`), `PARALLAX_ABI_VERSION`, `Plugin`, `PluginLoader`, `PluginRegistry`, `PluginError`, `PluginInfo`/`ElementInfo`, `define_plugin!`. See [plugins.md](plugins.md).

### `parallax::observability`

`metrics::{init_metrics, ElementMetrics, PipelineMetrics, record_*}` (metric names `parallax_buffers_*`, `parallax_bytes_*`, `parallax_processing_time_ns`, …); `tracing_support::{TracingConfig, span_pipeline, span_element, …}`.

### `parallax::gpu` (feature `vulkan-video`)

`Codec`/`VideoProfile`/`ChromaFormat`, `GpuFrame`/`GpuPixelFormat`, traits `GpuMemory`/`HwVideoDecoder`/`HwVideoEncoder`, `vulkan::{VulkanContext, VulkanH264Decoder, VideoSession, Dpb, VulkanGpuMemory}`. **Experimental scaffold** — the decode path does not yet submit real hardware decode commands.

### `parallax::error`

```rust
#[non_exhaustive]
pub enum Error {
    PoolExhausted,
    BufferPool(String),
    AllocationFailed(String),
    InvalidSegment(String),
    ValidationFailed(String),
    InvalidCaps(String),
    Config(String),
    Pipeline(String),
    Element(String),
    Io(std::io::Error),
    System(rustix::io::Errno),
    Device(DeviceError),        // only with device features enabled
}
pub type Result<T> = std::result::Result<T, Error>;
```

## Typed pipelines

`parallax::typed`:

| Kind | Items |
|------|-------|
| Traits | `TypedSource` (`produce() -> Result<Option<T>>`), `TypedSink`, `TypedTransform` |
| Builder | `pipeline(source)` → `.then(t)` / `>>` operator → `.sink(k)` → `RunnablePipeline::run()` (sync) / `run_async()` (spawn_blocking) |
| Operators | `map`, `filter`, `filter_map`, `inspect`, `take`, `skip` |
| Sources | `from_iter`, `range`, `once`, `repeat_with` |
| Sinks | `collect`, `discard`, `for_each`, `fold` |
| Multi-source | `zip` (pairwise), `merge` (two-source interleave), `join` (hash join), `temporal_join[_with_window]` (timestamp-aligned) |
| Bridge | `source_to_dyn`, `sink_to_dyn`, `transform_to_dyn`, `DynamicPipelineBuilder` — connect typed stages into dynamic pipelines |
| Caps markers | `Bytes`, `Typed<T>`, `Video<F, W, H, FPS>`, `Audio<F, RATE>`, `TypedBuffer<C>` |

`parallax::temporal`: `Timestamp` (ns + `ClockSource`; **a separate type from `clock::ClockTime`**), `TimeRange`, `TemporalJoin`, `JoinWindow`, `AlignmentStrategy` (`Exact`/`Tolerance`/`Nearest`/`Interpolate`), `Lerp`, `JoinResult`, `TimestampedItem`. `Interpolate` resamples the **right** stream onto the left stream's timestamps and is honoured by `TemporalJoin::try_emit_interpolated`, available when `B: Lerp`; the unbounded `try_emit` declines that variant rather than substituting another strategy.

## Environment variables

| Variable | Effect |
|----------|--------|
| `PARALLAX_TRACERS` | Activate tracers, e.g. `"latency;framerate;drops"` |
| `PARALLAX_DOT_DIR` | Dump DOT graphs on pipeline state transitions |
