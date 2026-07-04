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
| [plugins.md](plugins.md) | Writing and loading dynamic plugins (C ABI) |
| [api.md](api.md) | Map of the public API surface (use `cargo doc --open` for the full reference) |
| [security.md](security.md) | Current security model, its limits, and history |

## Design documents

| Doc | Contents |
|-----|----------|
| [design.md](design.md) | Design rationale, principles, and competitive landscape |

## Historical material

- [research/](research/) — research notes and design explorations that fed into the implementation. These are point-in-time documents: they may describe superseded designs (e.g. pre-implementation clock research, the original caps-negotiation survey, the abandoned stabby plugin ABI direction). Consult the user guides above for current behavior.
- [reports/](reports/) — dated development reports.
- [../plans/](../plans/README.md) — active implementation plans and status.
