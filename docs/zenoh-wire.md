# Zenoh Wire Format

How parallax's `ZenohSink`/`ZenohSrc` (feature `zenoh`) put buffers on the
zenoh network, and what other endpoints can rely on.

## Contract (version 1)

Each buffer becomes exactly one zenoh sample:

| Sample part | Content |
|---|---|
| payload | the raw buffer bytes, unmodified |
| encoding | derived from the buffer's `Metadata.format` (`video/h264`, `image/jpeg`-style MIME hints; `zenoh/bytes` when unknown), overridable via `ZenohSink::with_encoding` |
| attachment | `[0x50 'P', 0x58 'X', 0x01] ++ rkyv(WireMetadata)` |

`WireMetadata` (see `src/elements/network/zenoh_wire.rs`) carries:
pts/dts/duration (ns, `u64::MAX` = unset), sequence, stream_id,
`BufferFlags` bits, source byte offset, the media format (stable u8 codec /
pixel-format codes), and whitelisted custom byte entries (e.g. `stanag/klv`,
selected with `ZenohSink::with_forward_custom_keys`). The type-erased part of
the in-memory metadata map cannot cross the wire.

## Receiver behavior (`ZenohSrc`)

- Attachment present and valid → full `Metadata` restored. A gap in the
  restored `sequence` (samples lost to congestion drop, or a late join)
  sets `BufferFlags::DISCONT` on the buffer.
- Attachment missing, wrong magic, unknown version, or malformed → the
  sample is still delivered with fabricated sequence-only metadata (one
  warning is logged). Foreign publishers never crash a parallax consumer.
- With wildcard subscriptions, the concrete key expression a sample arrived
  on is stored under the `"zenoh/key_expr"` metadata key (`String`).

## Interop

- **Non-parallax subscribers** (zensight GUI, `z_sub`, …) can consume the
  payload directly — it is plain media bytes with a standard encoding hint —
  and ignore the attachment.
- **Publishing to attachment-averse consumers**: `ZenohSink::without_metadata()`
  omits the attachment entirely (PTS and all metadata are then lost on the
  wire — parallax↔parallax links should never use this).
- This format is deliberately **plane-scoped**: it serves parallax↔parallax
  links. Consumer-facing media planes define their own attachment schema
  (e.g. zensight's `@media` plane uses serde/CBOR `FrameMeta`); do not unify
  them.

## Versioning

`WireMetadata` is decoupled from the in-memory `Metadata` type on purpose:
internal refactors are not wire breaks. Any change to the serialized layout,
the codec/pixel-format code tables, or the semantics of a field requires
bumping the version byte (`WIRE_VERSION`) — receivers treat unknown versions
as foreign publishers (fallback, not error).

## QoS

Publisher knobs on `ZenohSink`: congestion control (`Block`/`Drop`),
priority (7 levels), `express` (immediate send vs batching). With the
`zenoh-unstable` feature (zenoh's own unstable API surface):
reliability (`Reliable`/`BestEffort`) and `matching_listener()` (fires when
the publisher gains/loses subscribers — the trigger for
keyframe-on-subscribe together with `KeyframeHandle`, see `docs/elements.md`).

Live video guidance (matches zensight's `QosClass::LiveVideo`): best-effort +
drop + `InteractiveHigh`, express off on constrained links.
