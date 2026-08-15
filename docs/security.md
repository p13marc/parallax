# Security Model

This document describes what Parallax currently does — and deliberately does **not** do — for security.

## Summary

**All pipeline execution is in-process.** Every element you add to a pipeline, including dynamically loaded plugins, runs with the full privileges of your process. Parallax currently provides *memory safety by construction* (Rust, with audited `unsafe` at the shared-memory and FFI boundaries), but **no privilege isolation between elements**.

## What happened to the sandbox?

Earlier versions of this documentation described a process-isolation feature (`ElementSandbox`, seccomp policies, namespaces, `run_isolated()`, an `Isolated` execution strategy for untrusted elements). That feature was **prototyped and removed** (commit `da6df59`) — the supervisor/fork design had a fork-bomb risk and was not production-ready. The related hints survive in the API:

- `ExecutionHints::trust_level` (`Trusted`/`SemiTrusted`/`Untrusted`) and `uses_native_code` still exist but are **informational only** — the executor ignores them when choosing a strategy (which is always in-process `Async` or `RealTime`).
- `plans/16_PROCESS_ISOLATION.md` tracks a possible reintroduction with a sounder design.

Do not rely on any Parallax API for confinement of untrusted code today.

## Trust boundaries that do exist

### Plugins

Dynamic plugins (`.so` via `PluginLoader`) are **arbitrary native code**. Every loading entry point is marked `unsafe` for this reason. The loader validates the descriptor before use — ABI version match (`PARALLAX_ABI_VERSION`), non-null pointers, element-name sanity — which protects against *malformed* plugins, not *malicious* ones. Load plugins only from paths you control; note the default search paths (`.`, `/usr/lib/parallax/plugins`, `/usr/local/lib/parallax/plugins`) include the current directory.

### Shared-memory IPC

Cross-process pipelines share arenas by fd. Anyone holding the arena fd can read and write **every slot in the arena** — the granularity of trust is the arena, not the buffer. Consequences:

- Only share arena fds with processes you trust with all of that arena's data.
- A malicious/buggy peer can corrupt slot contents and metadata for all other mappers, and can disturb refcounts (a peer that increments without decrementing leaks slots; validation prevents *resurrecting* freed slots via `SharedIpcSlotRef`, and refcount overflow is checked).
- Denial of service by a peer (holding refs forever, flooding the release queue) is possible; the owner's `reclaim()` double-checks state so queue abuse cannot free live slots.
- Buffers received from other processes are data, not capabilities: `rkyv` deserialization of IPC references is validated (`bytecheck`), and slot lookups are bounds- and state-checked.

### Network links

The network elements (TCP, HTTP, WebSocket, RTP, Zenoh) carry **no encryption or authentication** of their own — run them over trusted networks or tunnel them (WireGuard, TLS-terminating proxies), unless the underlying transport is secured (e.g. Zenoh's own security config). Where payloads are rkyv-framed they are validated on receipt, and the IPC control protocol bounds message sizes and errors (never panics) on malformed input.

### Parsers and codecs

Media parsing is the classic attack surface. Parallax prefers pure-Rust parsers/codecs (Symphonia, zune-jpeg, png, mp4, mpeg2ts-reader, rav1e) precisely to keep memory-unsafety out of that surface. The exceptions are C/C++ codecs behind feature flags — OpenH264, dav1d, libopus, FDK-AAC — which carry the usual native-codec risk; keep them updated, and prefer the pure-Rust alternatives when latency/quality budgets allow.

### `unsafe` inventory

The crate denies `unsafe_op_in_unsafe_fn` and concentrates `unsafe` in:

- `src/memory/` — mmap/memfd management, shared-memory atomics, fd passing. Invariants (refcount overflow checks, slot-state validation, `Send`/`Sync` justifications) are documented inline.
- `src/plugin/` — dynamic loading and C-ABI marshalling (double-boxed trait objects).
- FFI codec/device bindings behind feature flags.

Known sharp edge: `MemorySegment::as_mut_slice` can alias if callers violate its exclusive-access contract, and `SharedArena::from_fd` trusts the fd's header (it is `unsafe` accordingly).

## Deployment guidance

Until in-engine isolation exists, isolate at the OS level:

- **Separate processes by trust**: put untrusted stages in their own binary connected via `IpcSrc`/`IpcSink` (dedicated arena per boundary), and sandbox that binary with systemd hardening (`SystemCallFilter=`, `MemoryDenyWriteExecute=`, namespaces), bubblewrap, or a container.
- **Drop privileges** before `run()`; nothing in Parallax requires root. RT scheduling wants `CAP_SYS_NICE` only.
- **Limit resources** with cgroups (RT threads at `SCHED_FIFO` 50 can starve a core on malfunction).
- **Fuzz** anything that parses untrusted input; the typefind and demuxer layers are good targets (`cargo fuzz` harnesses welcome).

## Reporting

If you find a security issue, please open a private report rather than a public issue.
