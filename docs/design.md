# Parallax: Design Document

*Last substantive revision: July 2026. This document records why Parallax is built the way it is, what was tried and abandoned, and how it positions against the current landscape.*

## Executive Summary

Parallax is a Rust-native streaming pipeline engine designed to compete with GStreamer for **new Rust applications**, with three structural bets:

1. **Shared memory is the default, not an optimization** — every CPU buffer is memfd-backed and cross-process refcounted in shared memory itself.
2. **Scheduling is a first-class engine concern** — one executor that places elements on Tokio or on real-time threads from declared hints, rather than a thread-per-element model with ad-hoc queues.
3. **Rust end-to-end** — element authoring in ~20 lines without GObject, pure-Rust codecs and containers where the ecosystem allows, typed pipelines when you want the compiler in the loop.

## Why Build Another Pipeline Framework?

| Challenge | GStreamer | Parallax approach |
|-----------|-----------|-------------------|
| Language / safety | C + GObject; bindings for Rust | Rust throughout; `unsafe` confined to memory/FFI boundaries |
| Zero-copy IPC | Opt-in (`unixfd` elements since 1.24, careful BufferPool setup) | Default: all buffers memfd-backed; refcounts in shared memory |
| Cross-process refcounting | Message-coordinated (also PipeWire's model) | Atomics in the shared mapping; lock-free release queue; no messages |
| Caps + allocation | Caps negotiation and ALLOCATION query are separate phases | Format and memory type negotiated together (`FormatMemoryCap`) |
| Element development | ~200 lines of GObject boilerplate in C; ~100 in gstreamer-rs | ~20 lines: implement one trait |
| Scheduling | Thread-per-element + queues; threadshare (Rust/Tokio) exists as a plugin | Unified executor: Tokio + RT threads with lock-free bridges, hint-driven |
| Auto-plugging | Implicit; can silently insert converters | Explicit by default (`ConverterPolicy::Deny`) with opt-in auto-insertion |
| Type safety | Runtime caps checks | Optional compile-time typed pipelines |

None of these make Parallax "better than GStreamer" in general — GStreamer's ecosystem is twenty years deep. They make Parallax *simpler and safer for the cases it targets*: Rust-native applications, distributed/multi-process pipelines, embedded Linux.

## Core Design Principles

### 1. Unified memory model: memfd everywhere

There is no Heap-vs-SharedMemory distinction for pipeline buffers. `memfd_create` + `MAP_SHARED` costs the same as malloc'd pages and is always shareable by fd. One fd per **arena** (not per buffer) keeps fd usage at O(pipelines).

The novel part is the refcount placement: slot refcounts and the release queue are data structures *inside* the shared mapping, so `clone`/`drop` are plain atomics from any process, and the owner reclaims released slots in O(k) from a lock-free MPSC ring. PipeWire, by contrast, coordinates per-process refcounts with messages; GStreamer's unixfd plugin transfers buffers but doesn't share ownership semantics.

Consequence accepted: trust granularity is the arena. Any process holding the arena fd can touch every slot. Per-buffer OS-enforced permissions (PROT_READ-only slot mappings) were part of the original design and remain future work — see "History" below.

### 2. Pipeline-managed allocation

Elements don't allocate on the hot path. The pipeline (via negotiated caps) sizes pools; sources receive pre-allocated slots through `ProduceContext` and report `Produced(n)`. This keeps the fast path allocation-free (RT-safe), makes backpressure natural (pool exhaustion blocks), and means memory placement follows negotiation rather than element whim.

### 3. Unified format + memory negotiation

A capability is a *(format constraint, memory constraint)* pair, and elements advertise preference-ordered lists of them. "I produce YUYV as DMA-BUF, or YUYV/RGB as CPU memory" is one declaration, and choosing the zero-copy path is just normal negotiation.

Current implementation reality: the solver negotiates **per link** (first intersection in preference order) with single-hop converter insertion. A global constraint solve (minimize conversions across the whole graph) remains the design goal but is not implemented — the module docs used to overstate this; treat "global" as roadmap.

### 4. Sync processing, async orchestration, RT where it matters

Element hot paths are synchronous functions. Orchestration is Tokio. Elements that declare `rt_safe` + low latency are placed on dedicated RT threads driven in fixed quanta (PipeWire-style drivers and activation records), bridged to the async world by lock-free SPSC rings with eventfd wakeups. This is the same shape as PipeWire's graph but integrated with a general async runtime instead of a daemon.

### 5. Progressive typing

Dynamic pipelines (string parse, runtime graphs) and typed pipelines (compile-time-checked stage chains) are both first-class, with bridges between them. Dynamic for configuration-driven tools; typed for library code where refactors should break loudly.

### 6. Linux-only

memfd, SCM_RIGHTS, eventfd, DMA-BUF, MAP_HUGETLB are the foundation. Abstracting them away would cost the exact properties the engine is built on. Explicit non-goals: Windows, macOS, mobile.

### 7. Pure Rust where the ecosystem allows

Preferred: rav1e, Symphonia, zune-jpeg, png, mp4, mpeg2ts-reader (pure Rust). Accepted C where unavoidable: OpenH264 (patent license value), dav1d (decode speed), libopus, FDK-AAC. Watching: **rav1d** (the Rust dav1d port, ~5% from parity) as a future pure-Rust AV1 decode swap.

## History: Explored and Removed

Honest record of designs that shipped in earlier drafts of this document and were backed out:

### Process isolation / sandboxing (removed)

The original "Principle 0" was per-element sandboxed processes (seccomp, namespaces, per-buffer mmap permissions, supervisor with crash-restart, `ExecutionMode::{InProcess, Isolated, Grouped}`). A prototype existed (`src/execution/`) and was **removed in commit `da6df59`**: the fork/supervisor design carried a fork-bomb risk, and the isolation boundary interacted badly with the executor rewrite. What remains: `ExecutionHints::trust_level`/`uses_native_code` (currently informational) and `plans/16_PROCESS_ISOLATION.md`, which tracks a possible reintroduction (likely spawn-based, element-group processes connected by the existing IPC elements rather than fork-per-element). Until then, isolation is an OS-level deployment concern — see [security.md](security.md).

### stabby plugin ABI (never implemented)

Early drafts chose [stabby](https://github.com/ZettaScaleLabs/stabby) for ABI-stable plugins. The shipped implementation is a hand-rolled `#[repr(C)]` descriptor + `extern "C"` factory functions loaded via libloading, with an ABI version gate (`PARALLAX_ABI_VERSION`) — simpler, no macro-heavy dependency, and sufficient because element trait objects are double-boxed and only cross the boundary opaquely. Cost: plugins must be built with the same toolchain/parallax version (enforced socially, not technically). stabby may be revisited if third-party plugin distribution becomes real.

### `parallax-launch` CLI (not built)

A gst-launch equivalent was designed (including a YAML multi-binary orchestration format). No binary ships today; `Pipeline::parse` provides the underlying capability. Still a good first-contribution target.

## Deployment Modes

The same pipeline vocabulary works across process boundaries — the difference is which bridge elements you use:

```rust
// Single process
Pipeline::parse("videotestsrc ! videoconvert ! autovideosink")?;

// Multi-binary, same machine (zero-copy via shared arena + Unix socket)
// Binary A:
Pipeline::parse("v4l2src ! ipc_sink path=/run/parallax/camera")?;
// Binary B:
Pipeline::parse("ipc_src path=/run/parallax/camera ! ... ")?;

// Cross-machine (Zenoh: discovery, routing, optional shm transport)
// Machine A:
Pipeline::parse("camera-element ! zenoh_pub key=factory/camera/1")?;
// Machine B:
Pipeline::parse("zenoh_sub key=factory/camera/1 ! display-element")?;
```

(Note: `ipc_src`/`zenoh_pub`-style names above are illustrative — today these elements are constructed programmatically; only a small built-in set is registered for string parsing. Registering the full library with the factory is planned.)

| Boundary | Bridge | Copy cost |
|----------|--------|-----------|
| In-process | the executor's per-edge `tokio::sync::mpsc` channel | zero (move) |
| Cross-process | `IpcSrc`/`IpcSink` | zero (shared pages; ~bytes of metadata per buffer) |
| Cross-machine | `ZenohSrc`/`ZenohSink`, TCP links | serialize (rkyv) |

## Competitive Landscape (as of mid-2026)

### GStreamer

- **1.26** (March 2025): H.266/VVC, LCEVC, JPEG XS, QUIC/RTP-over-QUIC elements, analytics API expansion. **1.28** (January 2026): Vulkan Video AV1+VP9 decode and H.264 encode, major analytics overhaul (tensor negotiation, LiteRT inference), udmabuf allocator, MPEG-H, AMD HIP.
- **The Rust absorption is the headline**: per the official 1.28 release notes, **over 35% of commits in the 1.28 cycle were Rust** (gst-plugins-rs). `rtspsrc2` (Rust) replaces the C rtspsrc; the Tokio-based threadshare plugin (shared-thread element scheduling — conceptually adjacent to Parallax's executor) got major work and MPL relicensing.
- **unixfd plugin** (1.24, written in Rust): fd-passing buffer transport between processes — the closest analogue to Parallax's model, but opt-in per-element rather than a default memory model, with message-based buffer release.

**Implication**: "Rust vs C" is not a moat. Parallax's differentiation is the *coherent whole* — memfd-by-default, shared-memory refcounts, hint-driven unified scheduling, and no GObject — not the implementation language.

### PipeWire

- **1.4** (March 2025): MIDI 2.0, PTP clocking in RTP. **1.6** (February 2026): up to 128 channels, ONNX/FFmpeg filter-graph plugins, and **Capability Params** — pre-format capability negotiation on links, converging with the caps ideas Parallax borrowed.
- Remains the reference for low-latency Linux graph scheduling; now covers video (screen capture, libcamera cameras) and thus overlaps Parallax's device layer. Its per-process refcounting still requires message coordination — the in-shared-memory refcount remains a real Parallax differentiator.

### FFmpeg

**8.0** (August 2025) shipped Whisper speech-to-text and — notably for Parallax's GPU plans — **pure Vulkan-compute encoders/decoders** that run on any Vulkan 1.3 GPU without fixed-function video hardware, plus AV1 Vulkan encode. This validates the "compute-shader codec" direction (rust-gpu) as a credible complement to fixed-function Vulkan Video.

### Vulkan Video ecosystem

Extension status: H.264/H.265 encode final since early 2024; AV1 decode (1.3.280) and AV1 encode (1.3.302, late 2024); VP9 decode (1.4.317, 2025). Driver reality: Mesa RADV has decode for H.264/H.265/VP9/AV1 and **AV1 encode since Mesa 25.2**; Intel ANV maturing through Mesa 26.x; NVIDIA proprietary complete. Both FFmpeg 8 and GStreamer 1.28 ship consumers. **Assessment**: Parallax is late to Vulkan decode but roughly on-time for the encode wave; the current scaffold (context/session/DPB real, decode submission missing) should target H.264 decode first on RADV.

### Rust media ecosystem

| Project | Status | Relevance |
|---------|--------|-----------|
| gst-plugins-rs | First-class in GStreamer; crates.io + static linking | The incumbent answer to "Rust media pipeline" |
| rav1e | v0.8 (2025), active | Parallax's AV1 encoder |
| rav1d | ~dav1d parity minus ~5%; ISRG perf bounty running | Future pure-Rust AV1 decode swap for dav1d |
| OpenH264 | v2.6.0 (Feb 2025), low velocity | Kept for the Cisco patent license |
| retina | Active, production RTSP client (Moonfire NVR) | Parallax's RTSP backend |
| moq (moq-lite/hang), cloudflare/moq-rs | Production Rust MoQ stacks; Cloudflare MoQ CDN live; 11-vendor interop at NAB 2026 | Natural future sink (`moq_sink`) |
| LiveKit rust-sdks, str0m | Active WebRTC-centric stacks | Potential interop targets, not competitors |
| kornia-rs | Active Rust CV | Analytics-element ecosystem partner |
| Membrane (Elixir) | Active | Proof of demand for non-C pipeline frameworks |

### Streaming egress protocols

- **WHIP is RFC 9725** (2025); OBS and every major CDN ingest support it — the deployable sub-second option now.
- **WebTransport hit browser Baseline in March 2026** (Safari 26.4 closed the gap), unblocking **MoQ**-class delivery; moq-transport is in late IETF draft with deployment ahead of standardization (Cloudflare CDN, multi-vendor interop).
- Parallax ships HLS/DASH sinks today; WHIP and MoQ sinks are the highest-leverage additions (the reference MoQ stacks are Rust).

### Capture / kernel infrastructure

- Wayland `ext-image-copy-capture-v1` is merged in wayland-protocols: a lower-latency, PipeWire-free screen-capture path worth adding alongside the portal-based `ScreenCaptureSrc` (portal+PipeWire remains the only sandbox-safe path).
- libcamera is still pre-1.0 with ABI breaks per minor (0.5.x, 2025) — keep it feature-gated and version-pinned.
- V4L2 stateless decode is mature for H.264/HEVC/VP8/VP9 on supported SoCs; AV1 stateless and stateless *encode* uAPIs are still in flux. GStreamer 1.28's udmabuf allocator (DMA-BUFs from user memory) is a zero-copy bridge trick worth matching.

## Honest Assessment

**Where Parallax wins today**: element authoring ergonomics; default zero-copy IPC with true cross-process refcounts; one engine spanning async I/O and RT processing; explicit-by-default negotiation errors; pure-Rust codec/container path; typed pipelines.

**Where GStreamer wins**: ~1000 elements vs ~100; every platform vs Linux-only; twenty years of hardware quirks encoded; hardware codec breadth; community size. The 35%-Rust trajectory means GStreamer also increasingly offers "Rust element authoring" itself.

**Strategy**: don't chase migrations. Target (1) new Rust-native applications that want a crate, not a C runtime; (2) distributed/multi-process pipelines where the shared-memory model shines; (3) embedded Linux with a small pure-Rust footprint; (4) modern egress (MoQ/WHIP) where everyone is greenfield. Become for Rust media what Tokio is for Rust async: the default *native* choice.

## Roadmap

Framework maturity phases (details in `plans/`):

| Area | Status |
|------|--------|
| Memory foundation, IPC, negotiation, executor (async+RT), bus/probes/tracers/seek, plugins, typed pipelines | **Shipped** |
| Device capture (V4L2/libcamera/PipeWire/ALSA/screen), RTP/RTSP, MP4/TS mux/demux, HLS/DASH | **Shipped, maturing** |
| Vulkan Video decode (finish H.264 submission path, then H.265/AV1) | **In progress** (~scaffold; plan 11) |
| Vulkan Video encode; rust-gpu compute converters | Planned |
| Process isolation (redesigned, spawn-based) | Deferred (plan 16) |
| WHIP sink; MoQ sink; Wayland ext-image-copy-capture source | Proposed |
| `parallax-launch` CLI; full element registration in the parse factory | Proposed |
| RDMA / GPUDirect | Future |

## References

### GStreamer & PipeWire
- [GStreamer 1.26 release notes](https://gstreamer.freedesktop.org/releases/1.26/) · [1.28 release notes](https://gstreamer.freedesktop.org/releases/1.28/)
- [Collabora: unixfd plugin in GStreamer 1.24](https://www.collabora.com/news-and-blog/news-and-events/new-unixfd-plugin-in-gstreamer-124.html)
- [gst-plugins-rs](https://github.com/GStreamer/gst-plugins-rs)
- [PipeWire 1.4](https://www.phoronix.com/news/PipeWire-1.4-Released) · [PipeWire 1.6](https://www.phoronix.com/news/PipeWire-1.6)
- [GStreamer caps negotiation](https://gstreamer.freedesktop.org/documentation/plugin-development/advanced/negotiation.html)

### Vulkan Video & GPU
- [Khronos: Vulkan Video AV1 encode + quantization maps](https://www.khronos.org/blog/khronos-announces-vulkan-video-encode-av1-encode-quantization-map-extensions)
- [Igalia: Vulkan Video status](https://blogs.igalia.com/vjaquez/vulkan-video-status/)
- [RADV AV1 encode merged (Mesa 25.2)](https://www.phoronix.com/news/RADV-Merges-AV1-Encode)
- [FFmpeg 8.0 release](https://www.phoronix.com/news/FFmpeg-8.0-Released)

### Linux memory & zero-copy
- [memfd_create(2)](https://man7.org/linux/man-pages/man2/memfd_create.2.html)
- [DMA-BUF documentation](https://docs.kernel.org/driver-api/dma-buf.html)
- [Kernel: exchanging pixel buffers (formats + modifiers)](https://docs.kernel.org/userspace-api/dma-buf-alloc-exchange.html)
- [ext-image-copy-capture-v1](https://wayland.app/protocols/ext-image-copy-capture-v1)

### Codecs & protocols
- [rav1e](https://github.com/xiph/rav1e) · [rav1d perf bounty](https://www.memorysafety.org/blog/rav1d-perf-bounty/) · [OpenH264 2.6.0](https://github.com/cisco/openh264/releases/tag/v2.6.0)
- [retina](https://github.com/scottlamb/retina) · [libcamera](https://libcamera.org/)
- [RFC 9725 (WHIP)](https://datatracker.ietf.org/doc/rfc9725/) · [IETF MoQ WG](https://datatracker.ietf.org/group/moq/about/) · [moq.dev](https://moq.dev/) · [kixelated/moq](https://github.com/kixelated/moq)

### Rust ecosystem
- [stabby](https://github.com/ZettaScaleLabs/stabby) (considered, not used) · [Plugins in Rust](https://nullderef.com/blog/plugin-abi-stable/)
- [LiveKit rust-sdks](https://github.com/livekit/rust-sdks) · [kornia-rs](https://github.com/kornia/kornia-rs) · [Membrane](https://hexdocs.pm/membrane_core/readme.html)
