# Parallax Documentation

## User guides

| Doc | Contents |
|-----|----------|
| [getting-started.md](getting-started.md) | Install, first pipeline, writing custom elements |
| [architecture.md](architecture.md) | System overview: graph, executor, memory, negotiation, links |
| [pipeline.md](pipeline.md) | Pipeline construction, parse syntax, states, bus, events, seeking, probes, tracers, flow control, typefind |
| [scheduling.md](scheduling.md) | Executor internals: auto strategy, hybrid RT scheduling, drivers, bridges, clocks |
| [memory.md](memory.md) | SharedArena, cross-process refcounting, buffer pools, DMA-BUF, fd passing |
| [elements.md](elements.md) | Complete catalog of built-in elements with feature flags |
| [formats.md](formats.md) | Caps model, negotiation, converters, SIMD colorspace |
| [multi-machine.md](multi-machine.md) | Deploying across hosts: the RTSP → `@media` → browser topology, key layout, QoS, and what crosses a machine boundary |
| [zenoh-wire.md](zenoh-wire.md) | The two zenoh wire contracts: the parallax plane's rkyv `WireMetadata` and the `@media` plane's CBOR `FrameMeta`, QoS profile, terminal samples and keyframe promise |
| [plugins.md](plugins.md) | Writing and loading dynamic plugins (C ABI) |
| [api.md](api.md) | Map of the public API surface (use `cargo doc --open` for the full reference) |
| [security.md](security.md) | Current security model, its limits, and history |

## Design documents

| Doc | Contents |
|-----|----------|
| [design.md](design.md) | Design rationale, principles, and competitive landscape |

The `parallax::video` module is API documentation rather than a guide: presentation
maths with no graphics dependency (plane geometry, letterbox, colour matrices, the
WGSL), shared by parallax's own renderer and by a host GUI on another graphics stack.

## Historical material

- [research/](research/) — research notes and design explorations that fed into the implementation. These are point-in-time documents: they may describe superseded designs (e.g. pre-implementation clock research, the original caps-negotiation survey, the abandoned stabby plugin ABI direction). Consult the user guides above for current behavior.
- [reports/](reports/) — dated development reports.
- [../plans/](../plans/README.md) — active implementation plans and status.
