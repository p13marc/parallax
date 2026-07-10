# ZenSight "parallax" Video Sensor — Gap Analysis & Roadmap

*Report date: 2026-07-04. Revised 2026-07-07: zensight shipped the media-plane
enabler (#359) — see §1.4. Covers parallax @ master (6a61849), zensight @ master
(b190784), zenoh 1.9.*

*2026-07-07 (later): parallax-side work is now tracked on GitHub — umbrella issue
[#21](https://github.com/p13marc/parallax/issues/21), individual issues linked in the
gap tables below. Triage corrected two claims in this document: (a) **the `zenoh`
feature does not compile** (`ZenohSink`/`ZenohSrc` were never migrated to the
context-based traits), so §1.1's "Zenoh pub/sub ✅" is wrong at the compile level and
P1 became a rewrite ([#11](https://github.com/p13marc/parallax/issues/11)); (b)
**`get_element_mut()` does not work on a running pipeline** (`Executor::start` moves
elements into their tasks), so P7's "sensor holds a handle and calls
`force_keyframe()`" needs the `KeyframeHandle` mechanism from
[#12](https://github.com/p13marc/parallax/issues/12) — corrections inline below.*

## Goal

A new `zensight-sensor-parallax` crate (living in the zensight repo, per decision) that:

1. **Autodetects** available video streams (local cameras via V4L2/libcamera, screen capture, RTSP — static config + optional ONVIF discovery)
2. **Advertises** them on the zensight zenoh keyspace
3. **Opens streams on request** (control plane), builds a parallax pipeline per stream
4. **Decodes/encodes** (H.264 sw, MJPEG preview, AV1; hardware encode later)
5. **Ships media over zenoh**, consumable by the zensight Iced frontend and by other parallax pipelines

## TL;DR

**Update 2026-07-07: the zensight bucket is ~80% closed.** ZenSight #359 shipped
the zenoh-side media enabler, adopting this report's recommendations nearly
verbatim: AU-native wire format (not RTP), the `@media` verbatim-chunk keyspace,
`QosClass::LiveVideo` (best-effort · drop · interactive-high), a
`RawMediaPublisher` with matching-listener keyframe support, and the
`StreamControl`/`StreamDescriptor`/`StreamStatus` control types — all with an
end-to-end test over a real zenoh session. See §1.4.

The remaining work splits into three buckets:

1. **Parallax gaps** (~5 items): no JPEG *encoder* (preview path); no udev
   hotplug; no hardware encode at all; no ONVIF; no `RtpOpusPay`. The ZenohSink
   metadata-loss fix (P1) is **demoted from blocker to interop item** — the
   sensor now integrates via `AppSink → RawMediaPublisher` instead of
   `ZenohSink` (§3.3).
2. **ZenSight gaps**: Z1/Z2/Z5 **done** (#359). Z3 scaffolded (JPEG preview
   view helper exists; live subscription wiring + H.264 path remain). Z4
   (systemd sandbox) still open. **One new decision forced: the frame-metadata
   attachment schema is unpinned — see §2b (`FrameMeta`).**
3. **The sensor itself**: the daemon/session-manager layer (map of stream-id →
   `PipelineHandle`, driven by zenoh commands) exists in neither project and is
   the core new code — unchanged, but its zenoh-facing half is now a thin shim
   over shipped, tested zensight-core APIs.

---

## 1. What already exists

### 1.1 Parallax (surprisingly complete)

| Need | Status | Where |
|---|---|---|
| Enumerate cameras + formats | ✅ `enumerate_video_devices()` merges libcamera + V4L2; `V4l2Src::query_supported_formats()` gives fourcc + resolutions | `src/elements/device/mod.rs`, `v4l2.rs:399` |
| Enumerate audio | ✅ PipeWire + ALSA merged | `src/elements/device/` |
| Screen capture | ⚠️ XDG portal (interactive consent; `restore_token` helps, headless awkward) | `screen_capture.rs` |
| RTSP ingest | ✅ `RtspSrc` via retina: TCP-interleaved/UDP, digest/basic auth, per-stream codec from SDP | `src/elements/rtp/rtsp.rs` |
| RTP packetization | ✅ Pay/depay for H264/H265/VP8/VP9 are plain `Element`s (Buffer→Buffer), fully decoupled from sockets; MTU-configurable | `src/elements/rtp/rtp_codecs.rs` |
| RTCP / jitter buffer | ✅ `RtcpHandler`, `RtpJitterBuffer` (transport-independent logic) | `rtcp.rs`, `jitter_buffer.rs` |
| Zenoh pub/sub | ✅ `ZenohSink`/`ZenohSrc` with congestion-control + priority knobs, `Arc<Session>` sharing | `src/elements/network/zenoh.rs` (feature `zenoh`, dep `zenoh = "1.0"`) |
| Zenoh RPC | ✅ `ZenohQueryable::recv_query()` / `ZenohQuery::reply()` / `ZenohQuerier::get()` — a ready-made control plane | same file |
| App bridges | ✅ `AppSrc`/`AppSink` handles (push/pull, timeouts, bounded queues) — **now the primary egress path for the sensor** (§3.3) | `src/elements/app/` |
| Encoders (sw) | ✅ H.264 (openh264, constrained-baseline, `force_keyframe()` at `h264.rs:222`), AV1 (rav1e), Opus, AAC | `src/elements/codec/` |
| Decoders (sw) | ✅ H.264, AV1 (dav1d), JPEG, symphonia audio; Vulkan H.264 decode scaffold | same + `src/gpu/` |
| Per-stream lifecycle | ✅ Independent `Pipeline` + `PipelineHandle::abort()/wait()/subscribe()`; N pipelines per process compose fine | `src/pipeline/unified_executor.rs` |
| Codec probing | ✅ typefind registry; RTSP codec comes free from SDP | `src/pipeline/typefind.rs` |

### 1.2 ZenSight (framework + media enabler)

- **Sensor pattern**: no `Sensor` trait — you implement `SensorConfig` (JSON5) and compose
  `SensorRunner` (zenoh connect, health ticker on `@/health`, liveliness `@/alive`, status,
  SIGINT/SIGTERM shutdown). Your collection loop is your own tokio task via `runner.spawn()`.
  Copy the shape of `zensight-sensor-sysinfo`.
- **Control plane conventions fit "open stream on request"** and #359 added the
  concrete stream types: `@/commands/stream` carries
  `Command<StreamControl>`, `@/query/streams` serves `Vec<StreamDescriptor>`,
  `@/status/streams` serves `StreamStatus` (see §1.4). Zenoh matches `@`-chunks
  verbatim, so `zensight/**` subscribers never see `@` keys — load-bearing below.
- **Telemetry**: `TelemetryPoint` (JSON/CBOR) through per-key cached zenoh-ext
  `AdvancedPublisher` (10-sample cache, heartbeat). Right for stream *stats*; the media
  plane deliberately bypasses it (plain publisher, #359).
- **Large binary**: `zenoh-blob` — pull-based resumable file download. Not a live stream.
- **Frontend**: Iced 0.14 with `image-without-codecs`; `view/specialized/parallax.rs`
  decodes JPEG previews to RGBA via the `image` crate (jpeg codec only) and renders
  through `iced::widget::image` — placeholder frame + tests included. **No live zenoh
  media subscription wired yet** (`subscription.rs` untouched); H.264 display is still
  greenfield.

### 1.3 Ecosystem findings (web research)

- **Nobody ships RTP packets over zenoh.** The canonical demos and production bridges all
  send whole encoded frames per zenoh sample: zcam (JPEG per put; the Rust version publishes
  raw frames in zenoh SHM with an rkyv `FrameMeta` attachment), **your own gst-plugin-zenoh**
  (one GStreamer buffer per message, caps + PTS/DTS metadata preserved, CC/priority/express
  exposed), zenoh-plugin-ros2dds / rmw_zenoh (CDR `sensor_msgs` or `ffmpeg_image_transport`
  H.264 packets), EdgeFirst (CDR over GStreamer elements).
- **Modern precedent agrees**: Media-over-QUIC dropped RTP for a thin frame container (LOC);
  RTP-over-QUIC (draft-ietf-avtcore-rtp-over-quic) spends most of its text disabling RTP
  machinery the transport already provides. RTP survives at *edges* (RTSP-in via retina,
  RTP/UDP-out gateways).
- **Zenoh 1.x gives you**: per-publisher `CongestionControl::{Block,Drop}`, 7 priorities,
  `express` mode, automatic fragmentation (publish a whole IDR as one sample), per-keyexpr
  QoS overrides in config (since 1.1), implicit SHM promotion of large payloads between local
  peers (since 1.6 — free zero-copy to a co-located frontend), AdvancedPublisher caching for
  late joiners, liveliness tokens for presence, `Querier`/queryable RPC (rmw_zenoh pattern:
  attachment carries seq + client id).
- **Discovery crates**: `v4l` 0.14 (enumeration; no hotplug) + `udev` 0.9 (`video4linux`
  monitor) is the standard pair; libcamera-rs 0.7 has `subscribe_hotplug_events()`;
  `lumeohq/onvif-rs` is the most complete ONVIF/WS-Discovery implementation but is
  **git-only, unreleased**; `mdns-sd` for `_rtsp._tcp` as a cheap secondary probe.

### 1.4 NEW — ZenSight #359: the media-plane enabler (shipped)

ZenSight landed the zenoh-side half of this plan ("the H.264/parallax encoder
daemon is out of scope for #359 — this is the zenoh-side enabler only", per
their KEYSPACE.md). What exists now:

| Piece | Where (zensight repo) | Notes |
|---|---|---|
| `Protocol::Parallax` | `zensight-common/src/telemetry.rs` (+ `keyexpr.rs` matches) | Z1 done |
| `media_video_key()` / `media_preview_key()` | `zensight-common/src/keyexpr.rs:379` | `…/@media/<stream>/video/<codec>/<profile>`, `…/@media/<stream>/preview/jpeg`; acceptance test pins that neither `zensight/**` nor `zensight/*/@/**` can match a media key |
| `QosClass` module | `zensight-common/src/qos.rs` | `LiveVideo` = BestEffort · Drop · `Priority::InteractiveHigh` · **express off** (deliberate: batching beats latency on constrained links; priority already orders traffic — accept this divergence from the report's "maybe express") |
| `StreamControl::{OpenStream{stream,codec,max_height}, CloseStream, RequestKeyframe}` | `zensight-common/src/stream.rs` | `type`-tagged snake_case, rides the standard `Command<T>` envelope on `@/commands/stream` |
| `StreamDescriptor` / `StreamStatus` | same | catalogue on `@/query/streams`, sessions on `@/status/streams` |
| `Publisher::raw_media_publisher()` → `RawMediaPublisher` | `zensight-sensor-core/src/publisher.rs:178` | plain (non-Advanced) publisher declared with `QosClass::LiveVideo`; `put(payload, encoding, attachment)`, `matching_listener()`, `has_viewers()` |
| End-to-end test | `zensight-sensor-core/tests/media_e2e.rs` | full flow over a real zenoh session: catalogue query → OpenStream → publish → viewer subscribes → matching listener fires → keyframe-flagged frame observed. **P7's keyframe-on-subscribe pattern is proven without any parallax code.** |
| Frontend preview helper | `zensight/src/view/specialized/parallax.rs` | `preview_handle_from_jpeg()` (image crate → RGBA → iced handle), placeholder frame, tests; iced uses `image-without-codecs` so no codec deps enter iced |
| Docs | zensight `docs/KEYSPACE.md` §3.3 | Z5 done |

Deliberate simplification vs. §4's original flow: `OpenStream` is fire-and-forget
pub/sub — there is no reply carrying the concrete media key. The viewer
constructs the key itself via the keyexpr helpers (fully deterministic). Fine;
don't reintroduce the reply.

**Consequences for this plan:**

- The sensor should **not** use parallax's `ZenohSink` for the zensight media
  plane. Integrate via `AppSink → RawMediaPublisher` (§3.3) — QoS and keyspace
  policy stay in zensight where they're implemented and tested.
- P1 (ZenohSink metadata loss) is demoted to an interop item for
  parallax-native consumers.
- P7 trigger side is solved by zensight; the parallax side turned out to need
  real work (see corrected P7 row / [#12](https://github.com/p13marc/parallax/issues/12)):
  elements are unreachable once running, so the sensor clones a `KeyframeHandle`
  before start and calls `request()` on the matching listener's rising edge and
  on `RequestKeyframe` commands.
- One gap #359 left open: **the attachment schema is unpinned** (docs say
  "keyframe flag, PTS, sequence, …"; the e2e test uses ad-hoc JSON; no type in
  `zensight-common`). This is now the load-bearing producer↔consumer interface
  — decision in §2b.

---

## 2. Design decision — RESOLVED: AU-native, RTP as opt-in interop

*(Original question: RTP-in-zenoh vs. access units. #359 settled it.)*

ZenSight's media plane is AU-native: one encoded access unit per zenoh sample,
`Encoding` set to the real type (`video/h264`, `image/jpeg`), frame metadata in
an attachment, no serialization envelope. This matches every ecosystem
precedent (zcam, gst-plugin-zenoh, rmw_zenoh) and the report's recommendation.

RTP packetization remains available as a per-stream opt-in for future interop
(e.g. a thin zenoh→UDP bridge so ffplay/VLC work): insert `RtpH264Pay` before
the egress. Both modes share everything upstream. Not in scope until a
RTP-native consumer exists.

## 2b. NEW decision: `FrameMeta` — pin the attachment schema

The attachment on each media sample is the interface between the sensor (which
holds a `parallax::Metadata`) and two very different consumers: the iced
frontend (serde world) and parallax pipelines (rkyv world). #359 shipped the
transport but not the schema.

**Decision: define a small serde `FrameMeta` in `zensight-common/src/stream.rs`
(next to `StreamControl`), encoded as CBOR** via the existing
`serialization::{encode, decode}` helpers. Proposed fields:

```rust
pub struct FrameMeta {
    pub keyframe: bool,          // decodable entry point (IDR / full JPEG)
    pub pts_ns: Option<u64>,     // presentation time, ns (None = unknown)
    pub dts_ns: Option<u64>,     // decode time, ns (H.264 B-frames later; None for now)
    pub duration_ns: Option<u64>,
    pub sequence: u64,           // per-stream monotonic frame counter
    pub width: u32,              // coded dimensions — lets a viewer size the
    pub height: u32,             //   widget before the first decode
}
```

Rationale:

- rkyv would be alien in zensight-common (everything else there is serde
  JSON/CBOR); frame metadata is tiny, so zero-copy buys nothing.
- The frontend needs `keyframe` to gate H.264 decoder start, and
  `width`/`height` to lay out tiles before first decode.
- The sensor maps `parallax::Metadata` → `FrameMeta` at the publish boundary
  (trivial: pts/dts/duration are `ClockTime` ns, sequence is already there,
  keyframe = `!flags.contains(DELTA_UNIT)`).

**Plane-scoped formats — do not unify.** The zensight `@media` plane uses CBOR
`FrameMeta`; parallax-native `ZenohSink`/`ZenohSrc` links (P1, when it lands)
use rkyv `parallax::Metadata`. If a parallax pipeline later needs to consume
the zensight media plane directly, give `ZenohSrc` a pluggable attachment
decoder (or a tiny adapter element) rather than forcing one format on both
worlds.

Codec parameter sets (SPS/PPS): don't put them in `FrameMeta`. openh264
constrained-baseline emits Annex-B with in-band SPS/PPS on IDRs — combined
with keyframe-on-subscribe, late joiners get parameters for free. Revisit only
if a codec/profile without in-band parameters enters scope.

---

## 3. Gap analysis

### 3.1 Parallax gaps

| # | Gap | Impact | Effort | Notes |
|---|---|---|---|---|
| **P1** → [#11](https://github.com/p13marc/parallax/issues/11) | **`ZenohSink` drops buffer metadata** — and, found in triage, **the whole `zenoh` feature doesn't compile** (pre-context trait signatures). `ZenohSrc` fabricates `Metadata::from_sequence(seq)` (`src/elements/network/zenoh.rs:236`). | ~~Blocker~~ **Demoted (2026-07-07): the sensor egresses via `AppSink → RawMediaPublisher`, not ZenohSink.** Still needed for parallax↔parallax native links ("consumable by other parallax pipelines"). | M | Full v2 rewrite per #11: AsyncSink/AsyncSource, rkyv `WireMetadata` attachment (versioned), encoding tag, express/reliability knobs, zenoh 1.9. Keep format distinct from zensight's `FrameMeta` (§2b — plane-scoped). |
| **P2** → [#14](https://github.com/p13marc/parallax/issues/14) | **No JPEG encoder** (only `JpegDecoder`; `PngEncoder` exists). Needed for the MJPEG preview path when the camera doesn't emit MJPG natively. | Blocks preview for non-MJPG sources | S | Many cameras emit MJPG fourcc natively — V4L2 passthrough covers those with zero code. Add `JpegEncoder` (e.g. `jpeg-encoder` or `turbojpeg`) for the rest. The frontend preview consumer already exists (§1.4), so this lights up real UI. |
| **P3** → [#16](https://github.com/p13marc/parallax/issues/16) | **No hotplug.** Enumeration is scan-once; nothing watches `/dev/video*` appear/disappear. | Autodetect goes stale | S–M | `udev` crate monitor on subsystem `video4linux` (rustix ethos fits); libcamera-rs hotplug events as alternative. Emit add/remove events the sensor forwards as per-stream liveliness. |
| **P4** → [#17](https://github.com/p13marc/parallax/issues/17) | **No hardware encode.** `HwVideoEncoder` trait + `HwEncoderElement` wrapper exist but zero implementations; Vulkan module is decode-only; no VAAPI, no V4L2 M2M. openh264 is constrained-baseline CPU only. | Multiple HD streams on small boxes not viable | L | Two credible paths: **(a) V4L2 M2M** `H264` encoder element (kernel API, fits rustix/v4l stack, covers RPi/i.MX/Rockchip); **(b) VAAPI** via `cros-libva` (Intel/AMD desktops). Recommend (a) first — same ioctl family as existing `V4l2Src`. Vulkan encode is a bigger lift on an experimental scaffold. |
| **P5** | **No ONVIF/WS-Discovery, no mDNS.** | RTSP autodetect | M | `onvif-rs` (git-only — pin a rev) behind a config flag, per your decision; `mdns-sd` `_rtsp._tcp` browse as a cheap complement. Lives in the sensor crate (discovery policy, not pipeline machinery). |
| **P6** → [#19](https://github.com/p13marc/parallax/issues/19) | Minor: **no `RtpOpusPay`** (depay only); no VP8/VP9/H.265 encoders (pay/depay exist but nothing to feed them); AV1 has no RTP payloader. | Only if audio/RTP-interop matter early | S (Opus pay) | Defer unless audio streams are in scope for v1. |
| **P7** → [#12](https://github.com/p13marc/parallax/issues/12) | **Keyframe-request plumbing — CORRECTION (triage): parallax change IS needed.** `get_element_mut()` returns `None` once running (`Executor::start` takes elements into their tasks, `unified_executor.rs:884`); no event path reaches a running element. zensight's triggers (matching listener, `StreamControl::RequestKeyframe`) are proven, but the parallax side needs #12: a `KeyframeHandle` (Arc\<AtomicBool\>, cloned before start, checked in `process()`) + `VideoEncoder::force_keyframe()` trait method + the `"video/keyframe_request"` in-band constant. | S | Sensor flow: clone handle → `executor.start()` → `handle.request()` on viewer-joined. |
| **P8** → [#11](https://github.com/p13marc/parallax/issues/11) + [#20](https://github.com/p13marc/parallax/issues/20) | Housekeeping: `zenoh = "1.0"` → `"1.9"` (folded into #11); feature-combo verification became a real CI issue (#20) once triage found the zenoh feature had bit-rotted to non-compiling — nothing builds gated code today (no `.github/workflows`; default features are empty). | Build hygiene | XS–S | #20 pins the sensor combo `zenoh,h264,v4l2,rtp,rtsp,image-jpeg` in CI. |

New gaps found in triage (no P-number, filed directly):
[#13](https://github.com/p13marc/parallax/issues/13) `H264EncoderConfig.qp`/`keyframe_interval`
silently ignored (GOP control is load-bearing for late-joiner latency);
[#15](https://github.com/p13marc/parallax/issues/15) no fps control on `V4l2Src`/`LibCameraSrc`
(needed for the low-fps preview branch);
[#18](https://github.com/p13marc/parallax/issues/18) `RtspSrc` leaves SDP
dimensions/channels/sample_rate unparsed (stream catalog needs them).

Not gaps for this project (explicitly fine): no RTSP *server* (egress is zenoh), no live
element hot-swap (use one pipeline per stream, rebuild on change), no SRT.

### 3.2 ZenSight gaps

| # | Gap | Status | Notes |
|---|---|---|---|
| **Z1** | `Protocol::Parallax` variant | ✅ **Done (#359)** | `telemetry.rs` + both `keyexpr.rs` matches. |
| **Z2** | Framework QoS for video | ✅ **Done (#359), exceeded** | `QosClass` module + `Publisher::raw_media_publisher()` shipped in sensor-core (the "upstream later" item landed on day one). |
| **Z3** | Frontend video display | ⚠️ **Scaffolded** | JPEG preview decode helper + placeholder + tests exist (`view/specialized/parallax.rs`, iced `image-without-codecs`). **Remaining**: live media subscription in `subscription.rs` (subscribe to `@media/.../preview/jpeg`, parse `FrameMeta`, feed `preview_handle_from_jpeg`), stream catalogue/open-close UI, then the H.264 path (frontend depends on parallax features `zenoh, h264`; receive pipeline `AppSrc → H264Decoder → AppSink` fed from a zenoh subscriber — note: not `ZenohSrc`, since the plane's attachment is `FrameMeta`, §2b). |
| **Z3b** | **NEW: `FrameMeta` type in zensight-common** (§2b) | ❌ Open | Small serde struct + CBOR round-trip tests, next to `StreamControl`. The producer↔consumer contract; land before the sensor publishes anything. |
| **Z4** | Systemd sandbox (`DynamicUser`, `ProtectSystem=strict`) blocks `/dev/video*`, portals, LAN discovery | ❌ Open (expected — daemon out of scope for #359; no `DeviceAllow` in packaging yet) | Unit needs `SupplementaryGroups=video`, `DeviceAllow=char-video4linux`, network access for RTSP/ONVIF; document divergence in `packaging/systemd/PRIVILEGES`. |
| **Z5** | KEYSPACE/SENSORS docs | ✅ **Done (#359)** | KEYSPACE.md §3.3 documents the media plane, QoS row, keyexpr helpers, wildcard-vs-`@` rule. |

### 3.3 New code: the sensor daemon itself

The orchestration layer exists in neither repo (~the core deliverable). Its
zenoh-facing half is now a thin shim over shipped, tested zensight-core APIs:

- **Catalog**: periodic + hotplug-driven device scan (P3) merged with static config streams
  (RTSP URLs) and ONVIF/mDNS discoveries → serve `Vec<StreamDescriptor>` on
  `@/query/streams` (type shipped, #359).
- **Session manager**: `HashMap<StreamId, StreamSession { PipelineHandle, RawMediaPublisher, refcount }>`.
  `OpenStream` → build pipeline, `executor.start()`, store handle. `CloseStream` /
  zero matching subscribers (idle timeout) → `handle.abort()`.
- **Egress — the integration decision (2026-07-07)**: pipelines end in
  **`AppSink`, not `ZenohSink`**: `V4l2Src|RtspSrc → [decode →] H264Encoder → AppSink`,
  then a per-stream tokio task pulls buffers from the `AppSink` handle, maps
  `parallax::Metadata` → `FrameMeta` (§2b), and calls
  `raw_media_publisher.put(bytes, Encoding::VIDEO_H264, cbor(frame_meta))`.
  Keyspace + QoS policy stay in zensight; parallax stays transport-agnostic here.
- **Keyframe honesty**: `RawMediaPublisher::matching_listener()` rising edge and
  `StreamControl::RequestKeyframe` both → `keyframe_handle.request()` (handle cloned
  from the encoder before `executor.start()` — [#12](https://github.com/p13marc/parallax/issues/12);
  `get_element_mut` does NOT work on a running pipeline). `has_viewers()` / falling
  edge drives idle teardown.
- **Multi-profile**: a `tee` after decode feeding both a preview (MJPEG, low fps) and a full
  branch is programmatic-API territory (parse grammar has no tee branching — fine, don't
  use `Pipeline::parse` here).
- **Telemetry**: per-stream `TelemetryPoint`s (fps, bitrate, dropped, subscriber count,
  encode ms) on the normal cached bus under `…/<stream>/stats/<metric>` — this is the part
  that makes it a *zensight* sensor rather than a standalone daemon, and it lights up
  existing GUI charts for free.
- **Health/alerts**: `SensorHealth` device tracking maps 1:1 to cameras (3 consecutive
  failures → Offline); `AlertReporter` for "camera disappeared", "RTSP auth failed",
  "encoder overrun".

---

## 4. Keyspace — ADOPTED (#359, as proposed)

Shipped in zensight; helpers in `zensight-common/src/keyexpr.rs`:

```
zensight/parallax/<host>/@/alive                       # liveliness (from SensorRunner)
zensight/parallax/<host>/@/health, @/errors, @/status  # standard, free from sensor-core
zensight/parallax/<host>/@/devices/<stream>/alive      # per-stream liveliness = "advertised & openable"

zensight/parallax/<host>/@/query/streams               # queryable: Vec<StreamDescriptor>
zensight/parallax/<host>/@/commands/stream             # Command<StreamControl>
zensight/parallax/<host>/@/status/streams              # queryable: StreamStatus

zensight/parallax/<host>/<stream>/stats/<metric>       # TelemetryPoint (fps, kbps, drops, viewers)

zensight/parallax/<host>/@media/<stream>/video/<codec>/<profile>   # media_video_key()
zensight/parallax/<host>/@media/<stream>/preview/jpeg              # media_preview_key()
```

The `@media` chunk works as designed: zenoh matches `@`-prefixed chunks
**verbatim only**, so telemetry (`zensight/**`) and control (`zensight/*/@/**`)
subscribers are structurally incapable of ingesting the video firehose — now
pinned by an acceptance test in zensight.

Open/close flow as shipped (simplified vs. the original proposal): frontend
queries `@/query/streams`, sends `Command<StreamControl::OpenStream>` on
`@/commands/stream` (fire-and-forget — **no reply with the media key**; the
viewer constructs it deterministically via the keyexpr helpers), subscribes to
the media key; the sensor's matching listener forces a keyframe and keeps the
encoder honest; idle timeout or explicit `CloseStream` tears down.

---

## 5. Roadmap (revised 2026-07-07)

**Phase 0 — prove the wire (days).** ~~zensight half~~ **done (#359)**.
Remaining: Z3b (`FrameMeta` in zensight-common), parallax P8, and the
`zensight-sensor-parallax` skeleton: config, `SensorRunner`, `@/query/streams`
from `enumerate_video_devices()`, static RTSP list. MJPEG-passthrough preview
(native-MJPG cameras) published via `RawMediaPublisher` on
`@media/.../preview/jpeg`. First consumer can be the shipped frontend preview
helper instead of a throwaway CLI viewer — wire the zenoh subscription
(first slice of Z3).

**Phase 1 — H.264 end-to-end + control plane (1–2 weeks).** Session manager,
`OpenStream`/`CloseStream` handling, `V4l2Src|RtspSrc → H264Encoder → AppSink`
egress task (§3.3), per-stream telemetry + liveliness, keyframe-on-subscribe
via matching listener (pattern already proven in `media_e2e.rs`), P2 (JPEG
encoder) for universal preview, systemd unit (Z4). Consumer: parallax receive
pipeline (zenoh subscriber → `AppSrc → H264Decoder → …`).

**Phase 2 — frontend (1–2 weeks, parallelizable with Phase 1).** Z3 remainder:
live preview tiles (subscription wiring — decode helper already shipped), stream
catalogue/open-close UI, then H.264 via parallax receive pipeline + `AppSink`
into `iced::widget::image`.

**Phase 3 — discovery & robustness.** P3 (udev hotplug), P5 (ONVIF behind
config flag + mDNS), reconnect/backoff for RTSP, idle-timeout teardown, AV1
profile option. P1 (ZenohSink/ZenohSrc rkyv metadata) lands here for
parallax-native consumers, unless wanted earlier.

**Phase 4 — hardware encode (largest single item).** P4: V4L2 M2M encoder
element first, `cros-libva` VAAPI second; both slot behind the existing
`HwVideoEncoder` trait / `HwEncoderElement` so the sensor's pipeline builder
just swaps the encoder node.

## 6. Open questions

1. ~~**AU-native vs RTP-native**~~ — **RESOLVED by #359: AU-native** (§2). RTP
   stays a future opt-in interop mode.
2. **Audio in v1?** Determines whether `RtpOpusPay` (P6) and ALSA/PipeWire sources enter
   scope now. Recommend: video-only v1.
3. **Screen capture in a systemd sensor**: XDG portal wants an interactive session.
   Recommend deferring screen-capture streams to a per-user (`systemd --user`) deployment
   variant, not the hardened system unit.
4. ~~Should the parallax-side fixes land first?~~ — **Mostly moot**: the sensor
   path no longer goes through `ZenohSink` (§3.3), so P1 is off the critical
   path (Phase 3). P8 remains a Phase-0 one-liner.
5. **NEW — `FrameMeta` field review** (§2b): confirm the field set (keyframe,
   pts/dts/duration ns, sequence, width/height; SPS/PPS stay in-band) before the
   sensor publishes its first frame — it's the wire contract.

## 7. Key sources

- **zensight #359 implementation** (this revision's basis): `zensight-common/src/{stream,qos,keyexpr}.rs`,
  `zensight-sensor-core/src/publisher.rs`, `zensight-sensor-core/tests/media_e2e.rs`,
  `zensight/src/view/specialized/parallax.rs`, zensight `docs/KEYSPACE.md` §3.3
- zenoh-demos zcam (JPEG puts / SHM raw + rkyv attachment): <https://github.com/eclipse-zenoh/zenoh-demos>
- gst-plugin-zenoh (your GStreamer bridge — buffer-per-message + meta, QoS knobs): <https://github.com/p13marc/gst-plugin-zenoh>
- rmw_zenoh design (service RPC pattern, liveliness graph, QoS mapping): <https://github.com/ros2/rmw_zenoh/blob/rolling/docs/design.md>
- Zenoh 1.1 (Querier, per-keyexpr QoS config): <https://zenoh.io/blog/2024-12-12-zenoh-firesong-1.1.0/> · 1.6 implicit SHM: <https://zenoh.io/blog/2025-10-20-zenoh-imoogi/>
- RTP-over-QUIC draft (what RTP-in-a-message-transport costs): <https://datatracker.ietf.org/doc/draft-ietf-avtcore-rtp-over-quic/> · MoQ/LOC rationale: <https://www.meetecho.com/blog/moq-webrtc/>
- onvif-rs (WS-Discovery, git-only): <https://github.com/lumeohq/onvif-rs> · libcamera-rs hotplug: <https://github.com/lit-robotics/libcamera-rs> · libv4l-rs: <https://github.com/raymanfx/libv4l-rs>
