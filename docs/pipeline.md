# Pipelines

Everything about constructing, running, and controlling pipelines: the parse syntax, states, bus, events, seeking, probes, tracers, flow control, and type detection.

## Construction

### From a string

```rust
use parallax::pipeline::Pipeline;

let mut pipeline = Pipeline::parse(
    "filesrc name=reader location=input.bin ! passthrough ! filesink location=out.bin",
)?;
```

### Parse syntax

The grammar (winnow-based) is a **linear chain**:

```
pipeline   := element ( "!" element )*
element    := name ( property )*
property   := key "=" value
value      := "double-quoted" | 'single-quoted' | integer | float
            | true | false | yes | no | bareword
```

- Whitespace around `!` is optional.
- Identifiers may contain `-` and `_` (`buffer-size=4096`).
- `name=foo` is consumed by the pipeline (not passed to the element) and sets the node name for `get_element` lookups. Without it, nodes are auto-named `{factory}_{index}` (`filesrc_0`, `passthrough_1`, …).

**Limitations** (by design, for now):

- No caps filters (`video/x-raw,width=...` is not valid syntax).
- No branching — `tee name=t ! ... t. ! ...` style pad references don't exist. The grammar produces one **strictly linear chain**; [fan-out](#fan-out) requires the programmatic API or `PipelineBuilder`.
- No bins/parentheses.

**Built-in factory names**: `nullsource`, `nullsink`, `passthrough`, `inspect` (`tee` is a deprecated alias — it never fanned out), `filesrc` (`location`, `chunk-size`), `filesink` (`location`), `videotestsrc` (`pattern`, `width`, `height`, `num-buffers`, `framerate`; output is RGBA), `videoconvert`, plus `autovideosink` (feature `display`) and `v4l2src` (feature `v4l2`). Plugin-provided elements become parseable via `Pipeline::parse_with_factory(desc, &factory)` with an `ElementFactory` that has a `PluginRegistry` attached.

### Programmatically

```rust
let mut p = Pipeline::new();
let src  = p.add_source("src", MySource);              // Source impl
let xfm  = p.add_transform("xfm", MyTransform);        // Transform impl
let flt  = p.add_filter("flt", MyElement);             // Element impl
let sink = p.add_sink("sink", MySink);                 // Sink impl
let el   = p.add_element("el", Xfm(MySimpleXfm));      // unified Simple* API

p.link(src, xfm)?;                        // = link_pads(src, "src", xfm, "sink")
p.link_pads(xfm, "src", flt, "sink")?;    // explicit pads for multi-pad elements
```

Variants: `add_source_with_arena(name, src, arena)` and `add_source_with_pool(name, src, pool)` attach buffer backing; `add_element_auto` auto-names. `add_async_source` takes an `AsyncSource` (e.g. a connected `RtspSession`).

Cycles are rejected (`daggy` enforces a DAG). Introspection: `sources()`, `sinks()`, `children(id)`, `parents(id)`, `nodes()`, `links()`, `validate()`, `describe()`, `to_dot()`, `to_json()`.

### Fan-out

**Fan-out needs no element.** Src-pads are genuinely 1:N — link the same node twice and the
executor hands each branch a clone of the buffer, which is a refcount bump on shared memory,
not a copy. (There is no `Tee`: the element of that name was a 1-in/1-out counter and is now
honestly called `Inspect`.)

```rust,ignore
let src  = p.add_source("camera", V4l2Src::new("/dev/video0")?);
let rec  = p.add_sink("recorder", recorder);
let live = p.add_sink("preview", preview);

p.link(src, rec)?;          // both links leave the same "src" pad
p.link_lossy(src, live)?;   // and this one may drop when it falls behind
```

The parse grammar is a strictly linear chain and cannot express this.

#### Link policy

Each **link** — not each pad, since fan-out means several links leave one pad — carries a
`LinkPolicy` and an optional channel capacity:

| Policy | Behaviour |
|--------|-----------|
| `Block` (default) | Wait for room. Back-pressures upstream — and, on a fan-out, **every sibling branch with it**. |
| `Drop` | Drop the buffer and carry on. Degrades *this* branch alone. |

This matters more than it sounds. With every branch blocking, a persistently slow branch fills
its channel and stalls the source and all its siblings — so a cheap 2 fps preview drags down a
full-rate H.264 branch, which is the opposite of what anyone predicts. Make the branch that is
allowed to fall behind lossy.

```rust,ignore
p.link_lossy(src, preview)?;                                   // Drop, default pads
p.link_with(src, preview, LinkPolicy::Drop)?;                  // same thing, explicit
p.link_pads_with(src, "src", preview, "sink", LinkPolicy::Drop)?;
p.link_pads_full(src, "src", rec, "sink", LinkPolicy::Block, Some(64))?;  // deeper queue
```

**EOS is never dropped**, whatever the policy — a sink that missed it would wait forever. Only
buffers are. Dropped buffers are reported to `TracerRegistry::notify_drop`, so `DropTracer`
counts them.

### Element retrieval

```rust
use parallax::elements::io::FileSrc;

if let Some(src) = pipeline.get_element::<FileSrc>("reader") {
    println!("path: {}", src.path().display());
}
if let Some(src) = pipeline.get_element_mut::<FileSrc>("reader") {
    *src = FileSrc::new("other.bin");
}
```

Downcasting is checked — a wrong type or unknown name returns `None`.

### Fluent builder

```rust
use parallax::pipeline::PipelineBuilder;

let pipeline = PipelineBuilder::new()
    .source(my_source)             // or source_named("src", ...)
    .then(my_transform)            // or then_named("xfm", ...)
    .sink(my_sink)                 // or sink_named("sink", ...)
    .build()?;
```

The builder is typestate-checked (`Empty → HasSource → Complete`) and supports `fanout(|f| f.branch(...))` for fan-out — which inserts a plain `Inspect` junction and starts every branch from it, so the duplication comes from the node's src-pad being 1:N, not from any element. See `examples/10_builder.rs`.

## States

```
Suspended <──> Idle <──> Running
                            │
          Error ◄───────────┘   (recover via Suspended)
```

| Method | Transition | What happens |
|--------|------------|--------------|
| `prepare()` | Suspended → Idle | validate graph, negotiate caps, apply converter policy, allocate pools |
| `activate()` | Idle → Running | start processing |
| `pause()` | Running → Idle | stop processing, keep resources |
| `suspend()` | Idle → Suspended | release resources |

`prepare_with_auto_converters()` is `prepare()` with converter auto-insertion enabled for that call. `run()`/`start()` auto-prepare a `Suspended` pipeline. The PipeWire insight applies: "paused" and "stopped" are the same state (Idle) — the difference is intent.

## Running

```rust
// Simplest: run to completion
pipeline.run().await?;

// With a live bus handler (return false to stop; Error ends the run)
pipeline.run_with_bus(|msg| { println!("{msg}"); true }).await?;

// With a custom executor config
pipeline.run_with_config(config).await?;

// Detached: start returns a handle (synchronous call!)
let handle = pipeline.start()?;
// ... do other things ...
handle.wait().await?;
```

`PipelineHandle` gives you `wait()`, `abort()`, `stop()`, `subscribe()` (typed `PipelineEvent` channel), and access to the bus (`bus_mut()`, `take_bus()`, `bus_handle()`).

### How the run ended

`wait()` takes the handle by value, so the caller who keeps it to *control* the pipeline cannot use it to learn the outcome. `ended()` answers instead, and unlike `subscribe()` it **retains** the answer — an observer that arrives after the pipeline finished still gets it rather than waiting forever.

```rust
let ended = handle.ended();          // owned, so it survives `wait()`

tokio::select! {
    _ = tokio::signal::ctrl_c() => handle.abort(),
    reason = ended => match reason {
        EndReason::Eos           => println!("stream ran out"),
        EndReason::Error(err)    => eprintln!("{err}"),   // names the element
        EndReason::Aborted       => println!("torn down"),
    },
}
```

`end_reason()` is the non-blocking peek. `stopper()` hands out a `Stopper` that outlives the handle, for stopping a pipeline you have already `wait()`ed on elsewhere.

`stop()` reports `Eos` (the sources end their loop and EOS flows downstream as usual); `abort()` reports `Aborted`. The same `EndReason` is what an `AppSink` reports for its own branch — see [elements.md](elements.md).

In hybrid mode the answer waits for `wait()`/`abort()` to join the RT threads, which never end on their own.

Executor configuration and the async/RT scheduling model are covered in [scheduling.md](scheduling.md).

## Bus & messages

The bus is the GStreamer-style channel from elements to the application.

```rust
use parallax::pipeline::bus::MessageKind;

let mut bus = pipeline.take_bus().unwrap();

// Poll (non-blocking)
if let Some(msg) = bus.poll() {
    match msg.kind {
        MessageKind::Eos => println!("done"),
        MessageKind::Error { error, .. } => eprintln!("error: {error}"),
        MessageKind::Buffering { percent, .. } => println!("buffering {percent}%"),
        _ => {}
    }
}

// Await one message
let msg = bus.next().await;

// Wait for terminal condition
bus.wait_for_eos_or_error().await?;

// As a futures::Stream — works with select! and combinators
use futures::StreamExt;
let mut stream = bus.into_stream();
tokio::select! {
    Some(msg) = stream.next() => { /* ... */ }
    _ = tokio::signal::ctrl_c() => { /* shutdown */ }
}

// Broadcast to multiple consumers
let mut rx = bus.subscribe();
```

`MessageKind` variants: `StateChanged`, `Eos`, `Error`, `Warning`, `Info`, `Tag` (with `TagList`), `DurationChanged`, `StreamCollection`, `Qos`, `LatencyChanged`, `Buffering` (percent + rates + `BufferingMode`), `AsyncStart`/`AsyncDone`, `ClockLost`/`NewClock`, `SeekDone`, and generic `Element`/`Application` structures.

Elements post via their `BusHandle` (`ctx.post_message(...)` in produce/consume contexts, or typed helpers `post_error`, `post_tags`, `post_buffering`, …). See `examples/51_bus_messages.rs`.

**Terminal messages.** A run posts exactly one — `Eos`, or `Error` with `msg.source` naming the element that failed — never both, and never twice however many elements a single failure takes down. An aborted run posts neither: there is no `Aborted` message, and an `Eos` would claim the stream ran out when the caller cut it off. That is what makes `bus.wait_for_eos_or_error()` terminate.

`pipeline.run_with_bus(handler)` delivers messages to the handler *as they arrive*, not after the run; returning `false` cooperatively stops the pipeline, and an `Error` message stops it and becomes the call's error.

## In-band events

Distinct from bus messages, `Event`s travel *through* the pipeline with the data (`src/event/`):

- **Downstream**: `StreamStart`, `Segment`, `Tags`, `Eos`, `CapsChanged`, `Gap`
- **Upstream**: `Seek` (QoS feedback lives on the bus as `MessageKind::Qos`)
- **Bidirectional**: `FlushStart`, `FlushStop`, `Custom`

Serialized events share channels with buffers via `PipelineItem::{Buffer, Event}` so ordering is preserved; flush events also travel in-band; their immediacy comes from the flush epoch (a seek stamps all pre-seek buffers stale, and consumers shed them at receive speed). Elements handle them in `handle_downstream_event` / `handle_upstream_event` returning `EventResult::{Handled, NotHandled, Error}`.

## Seeking & position

```rust
let seekable = pipeline.query_seekable();
if seekable.seekable {
    pipeline.seek_bytes(1024)?;                       // byte seek
    pipeline.seek_time(ClockTime::from_secs(10))?;    // time seek

    if let Some(pos) = pipeline.query_position() {
        println!("{:?} @ {:?}", pos.format, pos.position);
    }
    let dur = pipeline.query_duration();
}
```

The graph-level `pipeline.seek_*` only reaches elements *before* `Executor::start` moves them into their tasks. To seek a **running** pipeline, use the handle — the seek is delivered to each source task, which repositions the source, bumps the pipeline-wide flush epoch (buffers are stamped with the epoch they were produced under, so consumers shed the queued pre-seek backlog at receive speed) and runs the flush sequence in-band (FlushStart → FlushStop → Segment, then `MessageKind::SeekDone` on the bus):

```rust
let handle = executor.start(&mut pipeline)?;
handle.seek_time(ClockTime::from_secs(30)).await;     // or seek_bytes / seek(SeekEvent)
// watch the bus for MessageKind::SeekDone
```

A source that cannot seek posts a bus `Warning` and keeps producing. Sources scheduled on RT threads have no control channel — runtime seek covers the async path only.

The handle also pauses and resumes a running pipeline, and reports the stream position:

```rust
handle.pause();                    // freezes the pipeline clock + gates sources
assert!(handle.is_paused());
let pos = handle.position();       // last-presented PTS (monotonic across pause/resume)
handle.resume();                   // gap-free: running time continues where it froze
```

Pause freezes the shared clock, so clock-paced sinks (e.g. `AutoVideoSink` with `sync`) stall mid-wait and resume without a burst of late frames; sources stop producing until resumed. Both post `MessageKind::StateChanged` (`Running ↔ Idle`). `position()` re-anchors on a runtime seek and falls back to running time before the first frame is presented.

Under the hood, a `SeekEvent` travels upstream to the source's `handle_upstream_event` (e.g. `FileSrc`). After a seek, a `SegmentEvent` establishes the timestamp mapping:

```rust
use parallax::event::SegmentEvent;

// seek to 10 s after 5 s already played:
let running = seg.to_running_time(ClockTime::from_secs(12)); // → 7 s
let stream  = seg.to_stream_time(ClockTime::from_secs(12));  // → 12 s
```

Build a custom `SeekEvent` (`SeekEvent::new_time`/`new_bytes`) and pass it to `PipelineHandle::seek` for non-default flags. See `examples/52_seeking.rs`.

### Snap direction

Container demuxers snap a time seek to a keyframe (`KEY_UNIT`). By default MP4 snaps *backward* to the GOP's keyframe and MKV *forward* to the next cue; `SNAP_BEFORE`/`SNAP_AFTER` pick the direction explicitly, and both bits together mean "nearest":

```rust
use parallax::event::{SeekEvent, SeekFlags};

let seek = SeekEvent::new_time(ClockTime::from_secs(30))
    .with_flags(SeekFlags::FLUSH | SeekFlags::KEY_UNIT | SeekFlags::SNAP_AFTER);
handle.seek(seek).await;
```

The synthesized `Segment` and `SeekDone` report the keyframe actually landed on. MKV resolves the direction at cue granularity and degrades to forward snapping on cue-less files; `ACCURATE` triggers iterative refinement in push-mode demuxers (#173): `TsDemuxElement` compares the first post-seek PTS with the target and, past a 500 ms threshold, forwards a corrected byte seek (same seqnum, next refinement round) up to 3 times before reporting the landing; the decoder-clipping half is still open.

### Seeking a fed demuxer (`filesrc ! tsdemux`)

Nothing in `filesrc ! tsdemux ! …` can seek in time: `FileSrc` seeks in bytes, and the demuxer does not own the reader. It works anyway, because the demuxer **translates** the seek:

```rust
let q = pipeline.query_seekable();
assert_eq!(q.format, SegmentFormat::Time);   // not Bytes — see below
handle.seek_time(ClockTime::from_secs(30)).await;
```

An element answers an upstream event with `EventResult::Forward(event)` to send a *different* event on in its place, built with `SeekEvent::derive(format, position)` so the seqnum survives — that identity is what keeps the flush epoch idempotent and lets `SeekDone` correlate. The executor forwards the replacement without bumping the epoch; the source that ultimately handles the derived seek does that, exactly as for an untranslated one.

Because the demuxer declares the conversion through `seek_translations()`, `query_seekable()` reports **`Time`, replacing the source's `Bytes`** — GStreamer's discipline, where a demuxer refuses byte seekability downstream. The range does not survive the swap (a byte count is not a duration), so `stop` reopens to 0 unless the demuxer itself knows a duration. `seek_bytes()` still works: it reaches `FileSrc` honestly.

Two completions reach the bus for one user seek, sharing a seqnum: the source's, in `Bytes`, at the offset it reached; and the demuxer's, in `Time`, carrying the PTS of the first buffer that actually arrived. The second is the one an application wants — it is measured, not estimated.

`TsDemux` places the seek with a `TsByteIndex`: sparse `(time, offset)` anchors harvested from the PCR clock as data flows past, interpolated linearly. Exact for CBR, approximate for VBR or ad-spliced streams. Known limits:

- an index with fewer than two anchors (under ~200 ms of stream seen, or no usable PCR) **refuses** the seek rather than guessing;
- a TS with head-only PSI is unseekable — a seek resets the parser, and a stream that never repeats its PAT/PMT never recovers;
- `duration()` stays `NONE` for a fed TS: the last PCR seen is a floor, not a total;
- a source that has already read to EOS is gone, and a seek arriving afterwards has nothing to act on. With a file small enough to fit entirely in the pipeline's channel buffers, that happens almost immediately.

## Pad probes

Intercept buffers and events at any pad for inspection, filtering, or dropping:

```rust
use parallax::pipeline::probe::{PadRef, ProbeData, ProbeReturn, ProbeType};

let probe_id = pipeline.add_probe(PadRef::src(src_node), ProbeType::BUFFER, move |data| {
    if let ProbeData::Buffer(buf) = data {
        // inspect buf ...
    }
    ProbeReturn::Ok      // pass through
    // ProbeReturn::Drop    — drop this buffer
    // ProbeReturn::Remove  — one-shot probe
    // ProbeReturn::Handled — consume (don't forward)
});
pipeline.remove_probe(probe_id);
```

`ProbeType` is a bitflag set: `BUFFER`, `EVENT_DOWN`, `EVENT_UP`, `EVENT_FLUSH`, `BLOCK`, `IDLE`. The executor invokes buffer probes before forwarding downstream.

Probes fire on **every** element's pads — a transform's `PadRef::sink(node)` (before the element sees the buffer) and `PadRef::src(node)` (after it produces one), as well as sources and sinks. See `examples/53_pad_probes.rs`.

Callbacks run inline on the data path, so keep them fast and non-blocking. A callback that **panics** is caught, logged at `error!`, and removed; the run carries on and still ends with `EndReason::Eos`. An observer does not get to end the run it is watching, and the other probes on the same pad keep working. The same holds for tracers.

## Tracers & debugging

```rust
use parallax::pipeline::tracer::{LatencyTracer, TracerRegistry};

let registry = TracerRegistry::new();
registry.add(Box::new(LatencyTracer::new()));
pipeline.set_tracer_registry(registry.clone());

// after the run:
for (name, report) in registry.reports() {
    println!("{name}:\n{report}");
}
```

Built-in tracers: `LatencyTracer` (per-element min/avg/max processing time), `FramerateTracer` (buffers/sec), `DropTracer` (buffers dropped on `LinkPolicy::Drop` links). They cover **every** element, transforms included — so an encoder's cost shows up in the latency report, which is usually the number you actually wanted. Environment activation:

```bash
PARALLAX_TRACERS="latency;framerate;drops" ./my_pipeline   # tracers
PARALLAX_DOT_DIR=/tmp/dots ./my_pipeline                   # DOT dumps on state changes
```

Also: `pipeline.to_dot()` / `to_dot_with_options(DotOptions::verbose())`, `to_json()`, `stats_snapshot()` (element/link counts and stats). See `examples/54_tracers.rs`.

## Flow control & backpressure

Live sources (cameras, screen, audio) produce at a fixed rate regardless of downstream speed. The flow-control primitives (`src/pipeline/flow.rs`) let downstream congestion reach the source.

### Link monitoring

In a task-per-element engine the link channel *is* the queue, so that is where occupancy is observable. `Pipeline::monitor_link` attaches watermarks to a link and returns a `FlowStateHandle` the executor drives at runtime — `Busy` when the channel fills past the high mark, back to `Ready` when it drains to the low mark (sampled after every send *and* every receive, so the signal releases even while a gated source is idle):

```rust
let link = pipeline.link_with(cam, enc, LinkPolicy::Drop)?;
let flow = pipeline.monitor_link(link)?;   // 80/20 of the link capacity
// or: pipeline.monitor_link_with(link, WaterMarks::new(24, 4))?
pipeline.get_element_mut::<V4l2Src>("cam").unwrap().set_flow_state(flow);
```

The source checks `should_produce()` before doing capture work and skips the frame while the link is backed up — cheaper than `LinkPolicy::Drop` alone, which discards the frame only *after* it was captured and copied. RT/bridge boundary edges carry no channel and are not monitored.

- **`FlowSignal`**: `Ready` | `Busy`, polled by the source via `FlowStateHandle::should_produce()`
- **`WaterMarks`**: high/low occupancy thresholds (default 80%/20% of the link capacity)

All device sources (`ScreenCaptureSrc`, `V4l2Src`, `LibCameraSrc`, `PipeWireSrc`, `AlsaSrc`) accept `set_flow_state`. In custom sources, check `flow_state.should_produce()` in `produce()` and return `ProduceResult::WouldBlock` when skipping. See `examples/47_flow_control.rs`.

### Network buffering: Queue2

`Queue2` provides GStreamer-queue2-style buffering strategies:

| Mode | Backing | Use case |
|------|---------|----------|
| `Queue2::stream(max_bytes)` | in-memory ring | HTTP/network streaming |
| `Queue2::download(path, total_size)` | progressive file | download with seek |
| `Queue2::timeshift(path, max_bytes)` | circular file | DVR rewind of live streams |

```rust
let q = Queue2::stream(10 * 1024 * 1024).with_watermarks(10, 95); // low%, high%
```

Buffering progress is posted as `MessageKind::Buffering { percent, mode, .. }` bus messages with rate estimates. See `examples/55_queue2_buffering.rs`.

## Media type detection

```rust
use parallax::pipeline::typefind::{MediaType, TypeFindRegistry};

let registry = TypeFindRegistry::with_builtins();
let result = registry.detect(b"\x00\x00\x00\x20ftypisom").unwrap();
assert_eq!(result.media_type, MediaType::Mp4);

let mt = registry.detect_from_extension("mkv");            // fallback by extension
let r  = registry.detect_with_fallback(&data, Some("mp4")); // bytes first, then ext
```

Built-in detectors cover MP4, Matroska/WebM, MPEG-TS, FLV, AVI, WAV, Ogg, FLAC, MP3, H.264 Annex B, PNG, JPEG, and more; results carry a `TypeFindProbability`. See `examples/56_typefind.rs`.

## Muxing (N-to-1)

Muxer elements synchronize multiple input pads by PTS before emitting:

```rust
use parallax::element::muxer::{MuxerSyncConfig, PadInfo, StreamType, SyncMode};

let config = MuxerSyncConfig::new()
    .with_mode(SyncMode::Strict)   // Auto | Strict | Loose | Timed { interval_ms }
    .with_interval_ms(40);         // 25 fps output cadence
```

`SyncMode::Auto` resolves to Strict for non-live and Loose for live inputs. Pipeline-ready muxers (`TsMuxElement`, `Mp4Mux`) build on `MuxerSyncState`; e.g. an MPEG-TS mux with video + KLV data tracks:

```rust
use parallax::elements::mux::{TsMuxConfig, TsMuxElement, TsMuxStreamType, TsMuxTrack};

let config = TsMuxConfig::new()
    .add_track(TsMuxTrack::new(256, TsMuxStreamType::H264).video())
    .add_track(TsMuxTrack::new(257, TsMuxStreamType::Klv).private_data());
let mut mux = TsMuxElement::new(config)?;
```
