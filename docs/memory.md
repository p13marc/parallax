# Memory Model

Parallax's memory model is designed for zero-copy data passing, both within a process and across process boundaries. The central idea: **all CPU buffers are memfd-backed from the start**, so sharing with another process never requires a copy or a conversion step — only passing a file descriptor.

## Why memfd-first

A conventional design allocates on the heap and copies into shared memory when IPC is needed. Parallax inverts this: `memfd_create` + `MAP_SHARED` memory costs the same as heap memory (it's the same anonymous pages), but it can always be mapped by another process. One fd is created per *arena* (pool), not per buffer, so fd limits are never a concern.

## SharedArena

`SharedArena` is the primary allocator: a fixed number of fixed-size slots in one memfd.

```rust
use parallax::memory::SharedArena;

let arena = SharedArena::new(4096, 64)?;        // 64 slots × 4 KiB
let mut slot = arena.acquire().expect("free");  // owner-only; returns Option<SharedSlotRef>
slot.data_mut()[..5].copy_from_slice(b"hello");
```

Constructors: `new(slot_size, slot_count)`, `with_name(...)` (debugging), `new_avx`/`new_avx512` (32/64-byte aligned slot data for SIMD), `with_alignment(...)`.

### Layout

Everything — including the bookkeeping — lives inside the shared mapping:

```
┌──────────────────────────────────────────────────────────────┐
│ ArenaHeader (64 B, cache-line aligned)                       │
│   magic "PLX_AREN", version, slot_count, slot_size,          │
│   data_offset, arena_id, arena refcount                      │
├──────────────────────────────────────────────────────────────┤
│ ReleaseQueue — lock-free MPSC ring (1024 entries)            │
│   head: AtomicU32   ← owner pops (single consumer)           │
│   tail: AtomicU32   ← any process pushes (multi producer)    │
│   slots: [AtomicU32; 1024]                                   │
├──────────────────────────────────────────────────────────────┤
│ SlotHeader[0..N] — 8 B each                                  │
│   refcount: AtomicU32   ← shared across processes            │
│   state:    AtomicU32   (Free | Allocated)                   │
├──────────────────────────────────────────────────────────────┤
│ SlotData[0..N] — user data, aligned to the arena alignment   │
└──────────────────────────────────────────────────────────────┘
```

### Cross-process reference counting

`Arc<T>` cannot work across processes: its refcount lives on one process's heap. Parallax's refcounts are atomics **in the shared mapping**, so the same increment/decrement works from every process that maps the arena:

- **Clone** (`SharedSlotRef::clone`, `slice()`, `slot_from_ipc`) → `fetch_add` on the slot refcount.
- **Drop** → `fetch_sub`; the process that drops the last reference pushes the slot index onto the `ReleaseQueue` (lock-free, O(1)).
- **Reclaim** — the owner calls `arena.reclaim()` (done automatically by pools before acquiring) and pops released indices, marking slots free: O(k) in released slots, never O(n) in pool size.

Refcount overflow is guarded (panics past `i32::MAX`); reconstruction from IPC validates arena id, slot bounds, and `Allocated` state, so a peer cannot resurrect a freed slot.

### Sending buffers to another process

```rust
use parallax::memory::{SharedArena, SharedArenaCache};
use parallax::memory::ipc::{send_segment_handle, recv_segment_handle};

// ── Process A (owner) ─────────────────────────────────────────
let arena = SharedArena::new(4096, 16)?;
send_segment_handle(&socket, arena.fd(), arena.total_size() as u64)?; // fd, once

let mut slot = arena.acquire().expect("free slot");
slot.data_mut()[..5].copy_from_slice(b"hello");
let ipc_ref = slot.ipc_ref();   // SharedIpcSlotRef: arena_id + slot index + offset + len
// serialize ipc_ref (rkyv) over the socket — a few bytes per buffer

// ── Process B (client) ────────────────────────────────────────
let (fd, _size) = recv_segment_handle(&socket)?;
let mut cache = SharedArenaCache::new();
unsafe { cache.map_arena(fd)? };                 // mmap once, cached by arena_id

let slot = cache.get_slot(&ipc_ref).expect("live slot");
assert_eq!(&slot.data()[..5], b"hello");
// slot shares the refcount with process A — drop from either side is correct
```

The `IpcSrc`/`IpcSink` elements (and `link::IpcPublisher`/`IpcSubscriber`) implement this protocol for you, including EOS and error signaling.

`memory::ipc` primitives: `send_fds`/`recv_fds` (up to 4 fds per message via `SCM_RIGHTS`), `send_segment_handle`/`recv_segment_handle` (fd + size pair).

Arena format **v4** records `slot_stride` and `alignment` in the header, so a client reads the true stride instead of assuming 64-byte rounding — arenas from `new_avx` (32-byte alignment) are safe to share cross-process. `from_fd` validates the recorded stride against `slot_size` and `alignment` and that every slot fits the mapping, refusing the arena rather than handing back one that mis-addresses slots.

## Buffers and metadata

### `Buffer<T = ()>`

A `Buffer` is a `MemoryHandle` (slot reference + offset + length) plus `Metadata`:

- `Clone` is O(1) — two atomic refcount increments (slot and arena) plus a `Metadata` clone. No data copy, and no syscall.
- `slice(offset, len)` produces a zero-copy sub-buffer (shares the slot).
- `as_bytes()` / `as_bytes_mut()` access the data directly in the mapped arena.
- `Buffer<()>` is the dynamic form used by pipelines; `Buffer<T>` adds a compile-time type tag (`into_dynamic()` erases it).

### `Metadata`

Carried by every buffer:

| Field | Type | Meaning |
|-------|------|---------|
| `pts`, `dts`, `duration` | `ClockTime` | presentation/decode timestamps (`ClockTime::NONE` = unset) |
| `sequence` | `u64` | monotonic sequence number |
| `stream_id` | `u32` | for demuxed/multi-stream flows |
| `flags` | `BufferFlags` | `SYNC_POINT` (keyframe), `EOS`, `DISCONT`, `DELTA`, `HEADER`, `CORRUPTED`, `DECODE_ONLY`, `TIMEOUT` |
| `rtp` | `Option<RtpMeta>` | RTP seq/ts/ssrc/pt/marker |
| `format` | `Option<MediaFormat>` | negotiated format snapshot |
| `offset` | `Option<u64>` | byte offset (e.g. file position) |

Plus a **typed custom map** for domain metadata — any `Clone + Send + Sync + Debug + 'static` value under a `&'static str` key (`"domain/type"` convention):

```rust
let mut meta = Metadata::new();
meta.set("app/frame_id", 12345u64);
meta.set("sensor/gps", GpsPosition { lat: 37.0, lon: -122.0 });
assert_eq!(meta.get::<u64>("app/frame_id"), Some(&12345));

meta.set_bytes("h264/sei", vec![0x06, 0x05, 0x10]);   // raw-bytes helpers
meta.set_klv(klv_bytes);                              // STANAG/MISB ("stanag/klv")
```

Namespaces in use: `stanag/*`, `h264/*`/`h265/*`/`av1/*`, `caption/*`, `audio/*`, `sensor/*`, `app/*`.

**Transforms must clone/propagate metadata** when they construct new buffers, or PTS and custom data are silently lost.

## Buffer pools

`FixedBufferPool` layers pipeline-friendly behavior over an arena:

```rust
use parallax::memory::{BufferPool, FixedBufferPool};

let pool = FixedBufferPool::new(1024 * 1024, 10)?;  // 10 × 1 MiB → Arc<FixedBufferPool>

let pooled = pool.acquire()?;        // blocks when exhausted = natural backpressure
let pooled = pool.try_acquire();     // non-blocking
let stats  = pool.stats();           // acquisitions, waits, availability
```

- `PooledBuffer` returns its slot on drop, or `into_buffer()` detaches it into a free-standing `Buffer`.
- Attach to sources with `pipeline.add_source_with_pool(...)`.
- Inside `Source::produce`, `ctx.acquire_buffer()` uses the attached pool.

**Blocking `acquire()` is for sources only.** Element tasks run on the Tokio
runtime, so a transform that blocked here would park a worker — possibly the one
that would have drained the channel it is waiting on, which deadlocks rather than
merely stalling. Elements that allocate their own output buffers use an arena
sized up front instead; see [Output arenas](#output-arenas) below.

Note also that the pool's condvar only fires for *undetached* returns. A buffer
sent downstream has been through `into_buffer()`, so its slot comes back via the
arena's release queue and nothing rings the bell; `acquire()` therefore polls as
well as waiting.

Sensible slot sizes/counts for common media (1080p YUV, encoded video, audio periods, TS/MP4 mux buffers, …) are provided as constants in `parallax::memory::defaults`.

## Output arenas

An element that produces buffers owns an arena, and every buffer it emits pins a
slot until the last downstream reference drops. Size that arena below what the
graph can hold in flight and the element runs out of slots the first time a
consumer hesitates.

Only the executor knows the number — link capacity is its configuration — so it
computes an `OutputBudget` per node and hands it over before the element builds
anything. **Use `OutputArena`**, which packages the whole pattern; every built-in
element does, and hand-rolling the field is what #84 and #91 existed to undo.

```rust,ignore
struct MyEncoder { output: OutputArena }

impl MyEncoder {
    fn new() -> Self {
        Self {
            output: OutputArena::new(defaults::VIDEO_ENCODER_SLOT_COUNT)
                .with_min_slot_size(1024 * 1024),   // headroom for a big keyframe
        }
    }
}

impl Element for MyEncoder {
    fn set_output_budget(&mut self, budget: OutputBudget) {
        self.output.set_budget(budget);      // stored; the arena is built lazily
    }

    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
        self.output.admit()?;                // before irreversible work
        let encoded = self.encode(&buffer)?;
        let mut slot = self.output.acquire(encoded.len(), "myencoder")?;
        // ...
    }
}
```

The surface, and when to reach for each:

| | |
|---|---|
| `acquire(len, name)` | The default. Builds on first use, vets the length, `Error::PoolExhausted` when full. |
| `try_acquire(len, name)` | **Sources only.** `Ok(None)` instead of `PoolExhausted`, so a source can answer `ProduceResult::WouldBlock` — see below for why it must. |
| `admit()` | Encoders: check for room *before* doing work that cannot be undone. |
| `reset()` | A geometry change: the new frames need differently-sized slots. |
| `grow_to_fit()` | For an element whose output size follows its **input** and has no ceiling — a `Map`, an RTP depayloader. Leave it off where the size is genuinely fixed, so an oversize is reported rather than absorbed. |
| `set_slots(n)` | Pin the count, overriding the budget. Rarely right; warns when below it. |

Every slot is 64-byte aligned — `SharedArena::new` asks for a cache line, which
also satisfies `MemoryLayout::{SSE, AVX, AVX512}`. Nothing needs `new_avx`.

`set_output_budget` is on `Element`, `Transform`, `AsyncTransform`, `Source`,
`AsyncSource`, `Muxer`, `Demuxer`, `SyncElement`, `PipelineElement` and the three
`Simple*` traits — every kind of element that can emit a buffer.

The invariant is `slots ≥ max(downstream link capacity) + IN_FLIGHT_MARGIN`.
Per src pad it is the **maximum** over that pad's links, not the sum: fan-out
clones a `Buffer`, and a clone is a refcount bump on the *same* slot, so three
branches holding one buffer pin one slot. Across separate src pads (a demuxer's)
it *is* summed, because those carry different buffers. `resolve()` also clamps
the total to `MAX_OUTPUT_ARENA_BYTES`, since slot count and slot size are chosen
independently and 200 × 4K RGBA would otherwise ask for 6.6 GB.

**The budget is a floor, not a guarantee.** It bounds what the *channels* hold,
not what downstream *elements* hold: an `AppSink` queues up to its `max_buffers`,
a `Queue` up to its depth, and an application can retain every `Buffer` it pulls.
None of that is visible to the executor, so exhaustion stays possible — and is
handled where it happens rather than prevented here.

## Running out of slots

`Error::PoolExhausted` is the **one error the executor does not treat as fatal**
— in *transform* and *sink* tasks. When such an element cannot acquire an
output slot, the executor drops that buffer,
counts it on the `DropTracer` and the `parallax_buffers_dropped` metric, logs a
rate-limited warning (1st, 10th, 100th… consecutive), and carries on. For live
media that is the correct trade: a dropped frame is recoverable, a dead capture
session is not. Set `ExecutorConfig::shed_fatal_after` to opt back into failing —
a batch transcode should stop rather than quietly write a file with gaps.

Every other `Err` from `process()` still terminates the element. So do *all*
errors from a source, demuxer or muxer task, which have no shed arm — which
is why a source must use `try_acquire` and stall rather than reporting
exhaustion at all.

What an element should do on exhaustion depends on what it would corrupt:

| Element | On a full arena |
|---|---|
| Encoders | Call `OutputArena::admit()` **before** encoding and return early. A frame pushed into the GOP whose packet is then shed leaves a reference the decoder never receives — corrupt until the next IDR. Skipping the input only lowers the frame rate, which is what `skip_frames` already does deliberately. |
| Decoders | **Never** skip an input: it is a reference frame, and everything after it decodes wrong. Decode, then shed the output copy; the decoder's own state stays intact. |
| Muxers | Neither — a lost batch breaks continuity counters and PCR. Rely on the budget, which since #91 actually arrives. |
| Sources | `try_acquire`, and return `ProduceResult::WouldBlock` on `None`. The executor sleeps briefly and asks again. Where the slot is taken *before* the read — file, socket — stalling costs nothing at all; where the data has already been consumed it sheds that one frame, and the sequence counter must advance only after the slot is secured or the gap is permanent. |
| Stateless transforms | Shed freely. |

## Other segment types

All implement the `MemorySegment` trait (`as_ptr`, `len`, `memory_type`, `ipc_handle`, …):

| Type | Backing | IPC | Notes |
|------|---------|-----|-------|
| `DmaBufSegment` | DMA-BUF fd (V4L2 `VIDIOC_EXPBUF`, DRM, GPU export) | fd | `DmaBufBuffer` wraps it with metadata; `into_fd()` recovers the fd; `to_buffer(arena)` copies into CPU memory when needed |
| `MappedFileSegment` | file `MAP_SHARED` | by path | persistent buffers; `sync()`/`resize()` |
| `HugePageSegment` | `memfd_create(MFD_HUGETLB)` + `MAP_SHARED` (2 MB / 1 GB) | fd | `new()` errors when the hugetlb pool is empty; `new_or_fallback()` degrades to normal pages and *says so* — `fell_back()`, `memory_type()` reports `Cpu`, and `page_count()`/`prefault()` use `effective_page_size()` |

`MemoryType` (`Cpu`, `HugePages`, `MappedFile`, `DmaBuf`, `GpuAccessible`, `GpuDevice`, `RdmaRegistered`) participates in caps negotiation, so pipelines can select DMA-BUF vs CPU paths per link — see [formats.md](formats.md).

## Performance characteristics

Measured by `cargo bench --bench memory_pool` and `--bench throughput`; the
figures below are from a CometLake-U laptop, so treat them as orders of
magnitude rather than targets.

| Operation | Cost | Measured |
|-----------|------|----------|
| `SharedSlotRef::clone` | O(1), two atomic increments, no syscall | ~60 ns, flat in slot size |
| `Buffer::clone` (any process) | the above plus a `Metadata` clone | ~90 ns empty metadata |
| Slot acquire + release + reclaim | O(n) scan (n = pool size; pools pre-reclaim) | ~280 ns, flat in slot size |
| Slot release | O(1) lock-free queue push | — |
| Owner reclaim | O(k), k = released slots | — |
| Cross-process send | O(1) after one-time arena fd pass | ~13 µs to map an arena |
| `Buffer::slice` | O(1), zero-copy | — |

> `Buffer::clone` used to cost an `fcntl` **and** a `close` per clone, because
> `SharedSlotRef` holds a `SharedArena` by value and `SharedArena::clone` dup'd
> the fd. Sharing the fd through an `Arc` removed both syscalls and made clones
> ~90% faster; `benches/memory_pool.rs` guards against the regression.
