# Multi-machine video: RTSP → `@media` → browser

Status: filed, not started. Written 2026-08-24, **revised the same day** after
reading the two sibling repos — see § "What the first draft got wrong".

Tracked by [#204](https://git.marcpardo.eu/marcpardo/parallax/issues/204) (**epic**),
milestone *Multi-machine video: RTSP → @media → browser*. A second milestone,
*WebRTC/WHEP egress (deferred)*, holds the WHEP work that this revision took off
the critical path. Sibling workstreams: the browser frontend and sensor cleanups
in **zensight** ([#704](https://git.marcpardo.eu/marcpardo/zensight/issues/704)),
four RFC reconciliations in **zenkey** (#303–#306).

## Goal

```
machine 1:  RTSP camera → H.264 → @media                      [ALREADY SHIPS]
                      ↓  zenoh
browser:    zenoh-ts + WebCodecs                               [zensight #704]
machine 2:  zenohsrc → h264dec → autovideosink                 [this plan]
```

## What already exists — read this before planning anything

**`zensight-sensor-parallax`** is 7160 lines of shipped, e2e-tested daemon that
implements the whole of machine 1: RTSP/V4L2/test sources, a `low`/`medium`/`high`
tier ladder with one encoder each, demand-driven open/close with refcounts and an
idle reaper, matching-listener keyframes, SPS/PPS prepending, the `@rpc` control
surface and the `state` catalogue. **Do not re-plan it.**

**`zensight/src/view/specialized/parallax_h264.rs`** is a working viewer: a plain
zenoh subscriber on the exact tier key, keyframe-gated, decoding with
`parallax::elements::H264Decoder` used as a *library* (no pipeline, no executor),
resyncing on a `sequence` gap.

**`zenkey/rfcs/07-bulk-planes.md` §1** is the normative `@media` contract, ratified
at v1.17. parallax is an *implementer* of it, not its author.

## What the first draft got wrong

1. **It planned machine 1.** Already done, in zensight.
2. **It invented a `<base>/@decl` queryable** for stream description — a competing
   mechanism on a ratified plane whose catalogue is `state/<producer>/stream/<stream>`
   and whose control is `@rpc`. `@media` carries pixels and nothing else.
3. **It reached for WHEP.** zenoh-ts speaks real zenoh from a browser
   (`Session.get`, `declare_subscriber`, `liveliness`, and `Sample.attachment()`
   for the CBOR `FrameMeta`), and WebCodecs takes Annex-B natively with a codec
   string that is the SPS `profile-level-id`. A browser is a first-class consumer
   of the plane the iced GUI already reads — and gets a *hardware* decoder, where
   the GUI must hide openh264 behind a non-default cargo feature.

## The `@media` contract parallax must implement

```
<base>/v1/<origin>/@media/<producer>/<stream>/video/<codec>/<tier>
<base>/v1/<origin>/@media/<producer>/<stream>/preview/<format>
```

- payload = raw encoded bytes, **one whole access unit per sample**;
- codec on the sample `Encoding` — declared, never sniffed;
- per-frame metadata on the **attachment** as CBOR `FrameMeta`
  (`keyframe`, `pts_ns`/`dts_ns`/`duration_ns`, `sequence`, `width`, `height`);
- QoS profile `frame` = best-effort · drop · interactive-high · express;
  a **plain declared publisher, never an `AdvancedPublisher`**;
- **keyframe-on-subscribe** via a matching listener; the keyframe flag is a
  *byte-level promise* (parameter sets inline or prepended). The listener fires
  only on the 0↔N edge, so an Nth viewer uses a keyframe RPC;
- **no wildcard on `@media`** — the subscription *is* the quality choice;
- control is `@rpc`, catalogue is `state`, stats are `telemetry`.

## The blocker found while designing this

**parallax's encoders never rewrite `Metadata.format`** ([#236]). `H264Encoder::process`
clones the input's metadata (`src/elements/codec/h264.rs:915`), so an H.264 access
unit still claims to be `MediaFormat::VideoRaw{I420}` — and `encoding_for_format`
(`zenoh_wire.rs:416`) would publish `video/raw;I420;1920x1080` for an H.264 stream.
`JpegEncoder` sets `format = None` and loses geometry outright, and `VideoCodec` has
no `Mjpeg` variant at all, so `image/jpeg` is unrepresentable.

`@media` conformance is impossible until this is fixed, and it is fixed in the
format layer, not in the zenoh element. It also explains why the sensor passes the
`Encoding` in by hand — the derivation is broken, and the workaround hid the bug.

The fix is `MediaFormat::Video(CodedVideoFormat { codec, width, height, framerate })`,
mirroring `VideoRaw(VideoFormat)`. Not `set_video_dims`, which writes `VideoRaw` and
would tell every downstream element that a bitstream is packed I420.

## Decisions

| Question | Answer | Why |
|---|---|---|
| Browser transport | **zenoh-ts + WebCodecs** | The browser becomes another `@media` consumer, shaped exactly like the iced GUI. WHEP would be a second egress stack (DTLS, SRTP, ICE, SDP, an HTTP server) for a plane that already reaches the browser. |
| WHEP | **Deferred, not deleted** | Still the answer for consumers that cannot speak zenoh: mobile Firefox has no WebCodecs, and VLC/OBS/CDNs never will. |
| Two planes | **One element pair, two wire profiles** | `Parallax` (rkyv `WireMetadata`) and `Media` (CBOR `FrameMeta`). Separate elements would duplicate the `Controllable` impls, the arena-publish path, the factory registration and the `finish()` hook — and the conformance tests would stop exercising the code the other plane uses. |
| Merging the two attachments | **Never** | `FrameMeta` carries strictly less, on purpose. `docs/zenoh-wire.md`'s plane-scoping rule stands and gets stronger. |
| Tier ladder | **Stays in the sensor** | parallax stays a single-stream engine and exposes what the ladder needs: demand signals, live encoder control, honest stats. |
| Same-machine zero-copy | `ipcsink`/`ipcsrc`, **not** zenoh SHM | parallax already *is* an SHM transport, and its IPC path carries dmabuf and codec-owned memory that zenoh SHM (CPU-only) cannot. |

## Two things to state honestly

**There is no viewer count.** zenoh's `MatchingStatus` is a single `bool`. The
demand surface is `has_viewers()` plus monotonic rise/fall edge counters; anything
reporting a count would be inventing one.

**`session.liveliness()` is stable** (not `zenoh-unstable`), which is what lets
end-of-stream work on `@media` without a terminal sample — forbidden there, since
a zero-length sample on a video key is neither a whole access unit nor the declared
`Encoding`, and a conformant browser client would hand it to a decoder.

## Breaking changes

#236 (`MediaFormat::Video`), #219 (both wire formats + descriptor layout), #220
(custom metadata), #221 (plugin ABI 11 → 12), #222 (`ElementConstructor`, `Props`),
#208 (rtspsrc metadata), #218 (payloader trait), #223/#224/#231 (zenoh wire v2).
The wire-version movers should land in one release cycle.

## Verification

```bash
just check-sensor                  # zenoh,h264,v4l2,rtp,rtsp,image-jpeg,hotplug
cargo nextest run --features zenoh # tests/zenoh_media.rs — two distinct sessions,
                                   # scouting off, explicit 127.0.0.1 endpoints
```

The decisive parallax test: a published access unit returns byte-identical with its
`FrameMeta` intact, the keyframe gate holds before the first IDR, a `sequence` gap
sets `DISCONT`, and a `zenohsrc` on an exact tier key reports `Video(H264)` caps
under `ConverterPolicy::Deny` **before the first sample arrives**.

The milestone's acceptance test is one command:

```bash
parallax-launch "zenohsrc key=zensight/v1/h-xxxx/@media/parallax/cam0/video/h264/high \
                 ! h264dec ! autovideosink"
```

It needs #236, #232, #223 and #222, and honest caps from the key expression — and
it is impossible today.
