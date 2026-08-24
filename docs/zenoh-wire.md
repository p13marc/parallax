# Zenoh Wire Formats

parallax speaks **two** contracts over zenoh, selected by `WireProfile`. They
are deliberately not unified, and the rest of this document is mostly about
why.

| | `WireProfile::Parallax` | `WireProfile::Media` |
|---|---|---|
| For | parallax↔parallax links | the zenkey `@media` plane |
| Payload | raw buffer bytes | one whole access unit |
| Encoding | derived from `Metadata.format` | same, and **authoritative** |
| Attachment | `[b'P', b'X', ver] ++ rkyv(WireMetadata)` | CBOR `FrameMeta` |
| QoS default | Block · Data | best-effort · drop · interactive-high · express |
| Sequence | the buffer's | **the publisher's** |
| Keyframe flag | `SYNC_POINT` passthrough | **derived from the bytes** |
| Wildcards | allowed | refused at construction |
| End of stream | a terminal sample **and** liveliness | liveliness only |
| Owner | parallax | the `@media` specification |

## Where the schema lives

`WireMetadata` is **not** a zenoh type. It lives in `src/wire/`, always
compiled, because it is also what the IPC descriptor projects (#219). Keeping
it behind the `zenoh` feature is what let the two wires drift: `rtp` was
carried by one and dropped by the other, `planes` was dropped by both, and
`MediaFormat::Rtp` had an IPC tag and no zenoh mapping at all — so an
RTP-formatted buffer survived a *process* boundary and lost its format across
a *machine* boundary.

One schema, two encodings, one proof:

| | Encoding | Why |
|---|---|---|
| zenoh attachment | rkyv over `WireMetadata` | variable-length; absent fields cost nothing |
| IPC descriptor | 192-byte `#[repr(C)]` POD | fixed-stride array in shared memory on the per-buffer hot path |

`tests/wire_projection.rs` round-trips a corpus through both and asserts they
restore an identical `Metadata`. It runs under **default features**, because a
test that needed `zenoh` would not be guarding the build that ships the IPC
ring. The one documented loss is the custom map, which the descriptor overflows
through the control socket.

Every code both encodings write comes from `wire::codes`, whose lock test pins
each assignment. Reordering `PixelFormat` is therefore not a wire event.

## Chapter P — the parallax plane (contract version 4)

Each buffer becomes exactly one zenoh sample:

| Sample part | Content |
|---|---|
| payload | the raw buffer bytes, unmodified |
| encoding | derived from `Metadata.format` (`video/h264`, `image/jpeg`-style MIME hints; `zenoh/bytes` when unknown), overridable via `ZenohSink::with_encoding` |
| attachment | `[0x50 'P', 0x58 'X', 0x04] ++ rkyv(WireMetadata)` |

`WireMetadata` (`src/wire/metadata.rs`) carries pts/dts/duration
(ns, `u64::MAX` = unset), sequence, stream_id, `BufferFlags` bits, source byte
offset, the media format (stable u8 codec / pixel-format codes), the coded
geometry of encoded video (#236), and whitelisted custom byte entries (e.g.
`stanag/klv`, selected with `ZenohSink::with_forward_custom_keys`). The
type-erased part of the in-memory metadata map cannot cross the wire.

### Receiver behaviour

- Attachment present and valid → full `Metadata` restored. A gap in the
  restored `sequence` (samples lost to congestion drop, or a late join) sets
  `BufferFlags::DISCONT`.
- Attachment missing, wrong magic, unknown version, or malformed → the sample
  is still delivered with fabricated sequence-only metadata (one warning is
  logged). Foreign publishers never crash a parallax consumer.
- With wildcard subscriptions, the concrete key expression a sample arrived on
  is stored under `"zenoh/key_expr"`.

### Versioning

`WireMetadata` is decoupled from the in-memory `Metadata` type on purpose:
internal refactors are not wire breaks. Any change to the serialized layout,
the codec/pixel-format code tables, or the semantics of a field requires
bumping `WIRE_VERSION` — receivers treat unknown versions as foreign
publishers (fallback, not error) — the **IPC** wire takes the opposite policy
and gates on exact equality, because both ends there are parallax and a
mismatch is a deployment error rather than a stranger on the network.

v2 added the coded-geometry field; v3 added the terminal marker below; v4
added `rtp`, `planes` and `WireFormat::Rtp` (#219).

### Terminal samples (#224)

A stream ends with a **zero-payload sample whose attachment carries
`terminal`**: `WireTerminal::Eos`, or `WireTerminal::Error { message }` when
the publishing pipeline failed. The receiver maps the first to
`ProduceResult::Eos` and the second to an `Err`, so a stream that failed at the
far end does not read as a clean end at this one — #85's rule, extended across
the machine boundary.

Two details that are load-bearing rather than incidental:

- **It rides the media key expression**, never a sibling `.../@event` key.
  zenoh orders within a flow (publisher × priority × reliability), and two key
  expressions route, batch and deliver independently — so a terminal published
  elsewhere could overtake or lag the last frame arbitrarily, which is exactly
  the bug an end-of-stream marker exists to fix.
- **It is a field on `WireMetadata`, not a separate attachment kind**, so it
  carries the same `sequence` accounting as the frames before it: gap detection
  keeps working right up to the goodbye, and a zero-length *media* buffer stays
  distinguishable from one.

The sink publishes it from `AsyncSink::finish` (#221) — after the last
`consume` returned, on the same publisher, on the same key.

## Chapter M — the `@media` plane

parallax is a **conformant implementer of a contract it does not own**. The
normative reference is `zenkey/rfcs/07-bulk-planes.md` §1.

```
<base>/v1/<origin>/@media/<producer>/<stream>/video/<codec>/<tier>
<base>/v1/<origin>/@media/<producer>/<stream>/preview/<format>
```

What parallax guarantees on it:

- **One whole access unit per sample.** MTU slicing buys nothing here: losing
  a best-effort sample loses the whole AU whether it was one slice or twenty.
- **The codec is on the `Encoding`**, never sniffed, never in a payload
  envelope.
- **CBOR `FrameMeta` on the attachment** (`src/wire/frame_meta.rs`): keyframe,
  pts/dts/duration, sequence, width, height. Absent timing fields are
  *omitted from the map*, not null — that is wire shape, pinned by a test.
- **The `frame` QoS profile**: best-effort, drop, interactive-high, express. A
  plain declared publisher, never an `AdvancedPublisher` — recovering a
  superseded frame is anti-useful.
- **The keyframe flag is a byte-level promise.** The publisher runs a
  `ParamSetCache` (`src/codec/annexb.rs`), repairs a keyframe that arrived
  without its parameter sets, and then decides the flag by reading the bytes it
  is about to send — not by trusting what upstream attached.
- **The publisher owns the sequence.** `Metadata.sequence` is per-element and
  gets rewritten mid-graph (an encoder sets it to its own frame count), so only
  the publisher knows what actually went out. A subscriber's gap detection
  means nothing otherwise.
- **Exact keys only**, both sides. A publisher wildcarding its origin is lying
  about identity; a subscriber wildcarding the tier receives every tier
  interleaved on one subscriber.

There is deliberately **no terminal sample here**. RFC 07 §1 says the plane
carries pixels and nothing else, and a zero-length "goodbye" would reach a
decoder as a malformed access unit. End of stream is observable from
liveliness, below.

What parallax does **not** do on this plane: it never builds an `@media` key
(it has no origin and no producer chunk), never publishes a catalogue or a
status document, and never touches `@rpc`. It takes the key it is given and
carries pixels on it. Discovery and control belong to whatever daemon owns the
streams.

### Receiver behaviour

- Caps come from the **key expression**, not from received data. The codec is
  a literal chunk of the key, so `output_media_caps()` is honest before a
  single sample arrives — which is what makes `prepare()` deterministic under
  `ConverterPolicy::Deny`. A key naming no known codec reports `Bytes`, which
  intersects with anything and is true, rather than `Any`, which is not.
- **A keyframe gate** is armed at start: samples are dropped until the first
  one flagged as a decoder entry point, because a decoder fed a mid-GOP access
  unit produces nothing useful and complains for a whole GOP about it.
  `with_resync_on_discont(true)` re-arms it after a wire gap; the default is
  off, since re-arming throws away a whole GOP for one lost packet.
- `DISCONT` is set on a sequence gap **and on the first sample of a
  subscription** — joining a live stream partway is itself a discontinuity.
- A sample with no `FrameMeta` is delivered with synthesised metadata and one
  warning: a non-conformant publisher must not kill a consumer.

## Liveliness — both profiles (#224)

The sink declares a **liveliness token on its own key expression** when it
declares the publisher, and the source subscribes to it. `session.liveliness()`
is stable API in zenoh 1.10 — no `zenoh-unstable` involved — and a token is a
separate primitive with no payload, so it is not a sample on the plane and does
not offend RFC 07 §1. That is why both profiles get it.

A token disappears (`SampleKind::Delete`) on a clean shutdown, on `kill -9`,
and on a network partition alike. It is a **liveness** signal, not an
end-of-stream one — only a terminal sample distinguishes "the stream ended"
from "the publisher went away" — so `PublisherLoss::Wait` is the default:

| `on_publisher_loss` | Behaviour |
|---|---|
| `Wait` (default) | Warn on the bus, count it, keep the subscription. A camera pipeline restarting is normal operation; ending the receiver on every sender restart would turn a two-second gap into a dead viewer. On `@media` the keyframe gate re-arms, since whatever comes back starts a new GOP. |
| `Eos` | End the stream. Right for a bounded transfer, where the publisher vanishing means the data is not coming. |

Two more ways a source can end, for the case neither a terminal nor a token
catches — a publisher that is alive, silent, and never coming back:

- `ZenohSrcHandle::end_stream()`, the lever an **application** uses to apply a
  policy parallax has no business knowing. Watching the plane's `state`
  catalogue doc, for instance, needs the zenkey key grammar, the origin, the
  producer chunk and the `StreamStatus` schema; parallax has none of those and
  should acquire none. The application watches the doc, this is how it acts.
- `with_stream_timeout(d)` — off by default, and deliberately distinct from
  `with_timeout(d)`, which is a *heartbeat* that keeps the stream open. On
  `@media` the heartbeat emits **no buffer at all**, only a rate-limited
  warning: the plane's samples are whole access units, so a zero-length one
  flagged `TIMEOUT` is a malformed frame a decoder cannot tell apart from a
  real one.

## Copies (#226)

| Direction | Copies |
|---|---|
| Publish | **none** — zenoh is handed the arena's own bytes |
| Receive | **one**, straight into an arena slot |

Publishing wraps a cloned `Buffer` in `bytes::Bytes::from_owner`, which pins
the arena slot for exactly as long as the transport holds the slice; a `Buffer`
clone is two atomic increments and no syscall. On `@media` the
`ParamSetCache`'s `Cow` picks the path — borrowed (every delta frame, every
well-formed keyframe) publishes from the arena, and only a *repaired* keyframe,
which genuinely is new bytes, allocates.

Receiving walks `ZBytes::slices()` directly into an arena slot rather than
asking for a contiguous view, because a payload that arrived in several
transport fragments would otherwise be flattened into a `Vec` first.

The consequence, which is not optional: in-flight publishes pin producer slots,
so `ZenohSink` declares them through `retained_buffers()` (ABI 10) or the
upstream source starves as soon as the transport batches. On the **parallax
plane** that converts what used to be unbounded heap growth under
`CongestionControl::Block` into visible backpressure — the source runs out of
slots and answers `WouldBlock`. `@media` is `Drop` by contract and does not
block; nobody should "fix" it into blocking.

zenoh's own `shared-memory` feature is deliberately **not** wired; see
`docs/design.md` § Deployment Modes for the argument.

## Why the two are not unified

`FrameMeta` deliberately carries strictly less than `WireMetadata`: no
`RtpMeta`, no `PlaneLayout`, no custom entries, no `stream_id`, no byte
offset. That is not a gap to close. `@media` is a consumer-facing plane whose
schema is owned by its specification and read by things that are not parallax
— a GUI, a browser. Widening it to carry parallax's internals would break that
contract; narrowing `WireMetadata` to match would lose metadata on parallax's
own links, where the IPC descriptor carries all of it.

If a parallax pipeline ever needs to consume a *third* party's media plane,
the answer is another profile, not a wider schema.
