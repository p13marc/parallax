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
| Owner | parallax | the `@media` specification |

## Chapter P — the parallax plane (contract version 2)

Each buffer becomes exactly one zenoh sample:

| Sample part | Content |
|---|---|
| payload | the raw buffer bytes, unmodified |
| encoding | derived from `Metadata.format` (`video/h264`, `image/jpeg`-style MIME hints; `zenoh/bytes` when unknown), overridable via `ZenohSink::with_encoding` |
| attachment | `[0x50 'P', 0x58 'X', 0x02] ++ rkyv(WireMetadata)` |

`WireMetadata` (`src/elements/network/zenoh_wire.rs`) carries pts/dts/duration
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
publishers (fallback, not error). v2 added the coded-geometry field.

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
