# Multi-machine video

How a camera on one host reaches a viewer on another, using the pieces parallax
actually has. The wire contracts are in [zenoh-wire.md](zenoh-wire.md); this is
the how-to.

## The topology

```
  machine 1                              machine 2                  browser
  ┌────────────────────────┐             ┌──────────────────────┐   ┌──────────┐
  │ rtspsrc ! zenohsink    │  @media     │ zenohsrc ! h264dec   │   │ zenoh-ts │
  │   (profile=media)      │ ─────────►  │   ! autovideosink    │   │ + Web-   │
  └────────────────────────┘  zenoh      └──────────────────────┘   │ Codecs   │
                              routing                    ▲          └──────────┘
                                                         └──────────────┘
                                                     same key, two consumers
```

Three properties fall out of this shape and are worth stating before any code:

- **Nothing in the middle re-encodes.** The camera's H.264 access units cross
  the network as bytes and reach a decoder on the far side unchanged. parallax
  never transcodes on this path.
- **A consumer is not special.** Machine 2 and the browser subscribe to the
  same key expression. Adding a viewer adds a subscriber; it does not add a
  stream.
- **parallax does not own the namespace.** It publishes on a key it is handed
  and subscribes to a key it is handed. Discovery, catalogues and status
  documents belong to whatever daemon owns the streams — see
  [Where parallax stops](#where-parallax-stops).

## Key expressions

The `@media` plane's shape (normative reference: `zenkey/rfcs/07-bulk-planes.md`
§1):

```
<base>/v1/<origin>/@media/<producer>/<stream>/video/<codec>/<tier>
<base>/v1/<origin>/@media/<producer>/<stream>/preview/<format>
```

parallax reads only the trailing chunks — `video/<codec>/<tier>` or
`preview/<format>` — and only to answer one question: what codec is this? It
does not parse the base, validate an origin, or link the keyspace crate.

**Both sides must use exact keys.** A wildcard is refused at construction, not
warned about:

- a publisher wildcarding its origin is lying about its identity;
- a subscriber wildcarding the tier receives *every* tier of the stream
  interleaved on one subscriber, unseparable except by re-parsing the key per
  sample — the opposite of demand-driven simulcast.

Choosing the tier **is** choosing the quality. There is no negotiation step.

## QoS for live video

The `@media` profile fixes the plane's `frame` profile, and
`ZenohSink::media()` applies it for you:

| Setting | Value | Why |
|---|---|---|
| Reliability | best-effort | A retransmitted frame arrives after the frame that replaced it. |
| Congestion control | **drop** | The encoder must never block on the network. |
| Priority | interactive-high | Ahead of telemetry, behind control. |
| Express | on | No batching delay for a frame that is already whole. |
| Publisher | plain declared | Never an `AdvancedPublisher` — recovering a superseded frame is anti-useful. |

Turn **express off** on a constrained link: batching costs a few milliseconds
and buys back header overhead, which matters more than latency when the link is
the bottleneck. That is the one knob in this table worth changing per
deployment.

On the **parallax plane** the defaults are the opposite — `Block` and `Data` —
because that plane is used for links where losing a buffer is a bug, and where
backpressure to the source is the correct response to a slow consumer.

## What crosses a machine boundary, and what does not

| Crosses | Does not |
|---|---|
| Payload bytes, unmodified | Arena slots — the far side copies into its own |
| pts / dts / duration / sequence / flags | dma-buf and codec-owned (`External`) memory |
| The media format, and encoded geometry | The type-erased half of the metadata map |
| Whitelisted custom **byte** entries (KLV, SEI) | Anything registered as a typed custom value |
| End of stream, and whether it was an error | Seeks, and every other upstream event |

The last row on each side deserves a sentence.

**Custom metadata must be bytes to cross.** `Metadata`'s custom map is typed
and `&'static str`-keyed; only entries stored with `set_bytes` and named in
`ZenohSink::with_forward_custom_keys` are serialized. Everything else is
type-erased and stays local. Store it as bytes at the producer if it has to
travel.

**Upstream events stop at the machine boundary.** A seek is not carried over
zenoh, and neither is `ForceKeyUnit` — a subscriber cannot ask a camera on
another host for an IDR. What a publisher *does* have is demand signalling: it
learns when its first viewer arrives (`zenoh-unstable`'s matching listener) and
asks its own encoder for a keyframe then, so a late joiner does not wait out a
GOP. See `ZenohSinkHandle`.

**End of stream does cross**, which is not free and was deliberate: a terminal
sample on the parallax plane, a liveliness token on both. A receiving pipeline
that ends because the publisher *failed* reports `EndReason::Error`, not a
clean EOS. See [zenoh-wire.md](zenoh-wire.md) § Terminal samples.

## Recipe: two machines

### The consumer, as a command line

```sh
parallax-launch "zenohsrc key=<tier-key> profile=media ! h264dec ! autovideosink"
```

That is a working command, not an aspiration — but note **`zenohsrc` is an
async constructor**: it opens a session and declares a subscriber before it can
exist. `parallax-launch` uses `Pipeline::parse_async`, and so must any program
that builds this string itself. The synchronous `Pipeline::parse` meets
`zenohsrc` and says so, rather than pretending the feature is missing.

```rust
// The same thing in a program.
let mut pipeline = Pipeline::parse_async(
    "zenohsrc key=<tier-key> profile=media ! h264dec ! autovideosink",
).await?;
let handle = pipeline.start()?;
let stopper = handle.stopper();     // outlives the handle `wait()` consumes
handle.wait().await?;
```

`examples/63_media_consume.rs` is this, with Ctrl-C handling.

### The publisher

Programmatically, because a real one has a camera or an RTSP client in front of
it:

```rust
let sink = ZenohSink::media(tier_key).await?;   // applies the `frame` QoS
let demand = sink.control();                    // clone BEFORE start()

let mut pipeline = Pipeline::new();
let src = pipeline.add_async_source("rtsp", rtsp_session);
let out = pipeline.add_async_sink("out", sink);
pipeline.link(src, out)?;

let handle = Executor::with_config(ExecutorConfig::live_video()).start(&mut pipeline)?;
```

`ExecutorConfig::live_video()` is the preset for this shape: fully async,
`channel_capacity = 4` so the link queue is a latency floor rather than a
buffer, and shedding rather than failing when an arena runs dry.

### Both on one box first

`examples/62_zenoh.rs` re-execs itself into two processes — publisher and
subscriber — over loopback with scouting off. Run it before touching two
machines; it is the same code path with a shorter cable:

```sh
cargo run --example 62_zenoh --features zenoh
```

### Then on two

Delete the endpoint configuration and let zenoh scout on the LAN, or point the
consumer at a router:

```sh
export ZENOH_CONFIG=/etc/parallax/zenoh.json5   # read by the shared session
parallax-launch "zenohsrc key=<tier-key> profile=media ! h264dec ! autovideosink"
```

The elements built by the factory share **one process-wide zenoh session**,
configured from `ZENOH_CONFIG` if it is set. Programmatic construction can pass
its own session to `*_with_session` instead.

## The browser leg

A browser consumes the same key expression through
[zenoh-ts](https://github.com/eclipse-zenoh/zenoh-ts) talking to
`zenoh-plugin-remote-api`, and decodes with **WebCodecs**
(`VideoDecoder`). Two details make that work without a server-side transcode:

- WebCodecs decodes Annex-B natively when `description` is omitted from the
  configuration — which is exactly what `@media` carries, since the plane's
  samples are whole access units with in-band parameter sets on keyframes.
- The codec string (`avc1.<profile-level-id>`) can be derived from the SPS;
  `codec::annexb::h264_profile_level_id` does that, and
  `h264_browser_compatible` reports the cases a browser will refuse.

The keyframe gate matters here for the same reason it matters in parallax: a
decoder fed a mid-GOP access unit produces nothing and complains for a whole
GOP. `ZenohSrc` gates on the publisher's byte-level keyframe flag; a browser
consumer must do the same.

WHEP/WebRTC is the alternative and is deliberately **not** the primary path: it
needs an SFU or a per-viewer peer connection, where this needs neither.

## Where parallax stops

parallax carries pixels on a key it is given. It does **not**:

- build an `@media` key — it has no origin and no producer chunk, and should
  acquire neither;
- publish or read a catalogue or a status document;
- touch `@rpc`.

That boundary is load-bearing rather than tidy-minded. Consuming the plane's
`state/<producer>/stream/<stream>` document would require the zenkey key
grammar, the origin, the producer chunk and the `StreamStatus` schema —
four things parallax would then have to track a specification for. Instead it
exposes the lever and the application applies the policy:

```rust
let control = src.control();          // ZenohSrcHandle, cloned before start()
// … the application decides the stream is over, however it knows …
control.end_stream();
```

`PipelineHandle::stopper()` is the same idea one level up.

## Same machine? Use `IpcSink`/`IpcSrc`

If both ends are on one host, zenoh is the wrong tool: `IpcSink`/`IpcSrc` move
buffers through shared memory with no payload copy at all, and carry dma-buf and
codec-owned memory that a network transport cannot. zenoh's own shared-memory
feature is deliberately not wired — see [design.md](design.md) § Deployment
Modes for that argument.
