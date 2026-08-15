# Channel architecture analysis: tokio mpsc links vs. a custom shared-memory queue

*2026-08-15. Scope: the executor's per-edge links, the upstream inboxes, `AsyncRtBridge`, the leaky
channel, the arena `ReleaseQueue`, and the `IpcSink`/`IpcSrc` path — evaluated against the question:
given that Parallax's memory model is already shared-memory and cross-process capable, is it coherent
to keep `tokio::sync::mpsc` for inter-element links, or should we build a custom queue on the memory
model that integrates with tokio?*

---

## 1. TL;DR / verdict

**Q1 — Is it expected to have tokio mpsc links alongside our shared-memory model?**

Yes — and it is not an accident or a leftover; it is the correct division of labor. The channels are
a **signaling plane**, not a data plane. What crosses a link is a ~400-byte `Buffer` *handle*
(slot reference + metadata), moved by value with **zero payload copy and zero heap allocations per
buffer per hop** (§2). The payload never leaves the arena. This is the same architecture PipeWire
uses (shm buffers, fd-based wakeups) — the system Parallax's state machine and RT scheduler are
already modeled on. Putting the *data* in shared memory and the *notifications* in the runtime's
native channel is the standard shape of this design, not a contradiction of it.

**Q2 — Do we need a custom queue based on our memory model, integrated with tokio, for async
elements?**

For **in-process links: no.** A shm-resident ring would eliminate nothing that costs anything today
(no copies, no allocations to remove), while re-opening five solved correctness problems —
cancel-safe recv, closed-sender detection, two-ended occupancy sampling, three link policies with
control immunity, and coop-budget participation (§4). We already ran this experiment in reverse:
kanal was faster on synthetic ping-pong benchmarks and was removed anyway because its recv loses
messages on cancellation; the real media path measured the switch as noise (§3).

For **two narrow shm-adjacent paths: yes, a custom queue/doorbell is genuinely the right tool** —
and we already own the precedent design (`AsyncRtBridge`: SPSC ring + eventfd + `AsyncFd`):

1. **The `IpcSink`/`IpcSrc` per-buffer path** (§6.1) — today it is rkyv-framed messages over a Unix
   socket with a 1 ms sleep-poll for acks and an O(n) pending scan. A descriptor ring in the shared
   mapping plus an eventfd doorbell is the one place where "a queue built on our memory model" is
   exactly what the doctor ordered.
2. **The arena `ReleaseQueue` has no wakeup path at all** (§6.2) — `FixedBufferPool::acquire` papers
   over it with a 2 ms condvar poll. An optional eventfd doorbell deletes the poll.

Two latent defects in the existing shm ring should be fixed regardless of any new queue work (§6.3):
a full release ring **leaks the slot permanently** (the "full scan" fallback the code comments
mention does not exist), and `arena_id` is a process-local counter, so two peer processes minting
arenas can collide in `SharedArenaCache`.

---

## 2. How data actually flows today

### 2.1 Anatomy of a link

Every graph edge gets one channel, created in `make_link_channel`
(`src/pipeline/unified_executor.rs:1587-1611`):

```rust
let capacity = link.capacity.unwrap_or(self.config.channel_capacity).max(1); // default 16
let (tx, rx) = if link.policy == LinkPolicy::DropOldest {
    crate::pipeline::leaky::channel::<Message>(capacity)   // hand-rolled drop-oldest
} else {
    tokio::sync::mpsc::channel::<Message>(capacity)        // Block and DropNewest
};
```

What travels is the `Message` enum (`unified_executor.rs:2191-2218`):

```rust
#[allow(clippy::large_enum_variant)] // Intentional: avoid heap allocation on hot path
enum Message {
    Buffer(Buffer, u64),   // u64 = flush epoch (#157)
    Eos,
    Error(StreamError),
    Event(Event),          // FlushStart/FlushStop/Segment/StreamStart/... — FIFO with buffers
}
```

`Buffer` = `MemoryHandle` (a `SharedSlotRef` + offset + len) + `Metadata`. The slot ref holds the
arena by value plus raw pointers into the mapping — **no byte of payload ever crosses the channel**.
Hand-derived size (no `size_of` assertion exists in-repo, so treat as an estimate): `Metadata` ≈
216 B (SmallVec-inline custom map, `src/metadata.rs:464-503`), `Buffer` ≈ 384 B, `Message` ≈
**400 B**, moved by value. The enum is deliberately unboxed — the sibling `Incoming` enum carries
the comment "boxing would put an allocation back on the per-buffer path, which is the thing this
migration removes" (`unified_executor.rs:2716-2725`).

### 2.2 The measured cost profile

`tests/media_alloc_tests.rs` ratchets this with a tracking global allocator:

| Measurement | Allocations | Where |
|---|---|---|
| `Metadata::clone` (with video dims) | **0** | `:50-73` |
| Executor per-buffer **per hop** | **0** (budget 0.5) | `:287-305` |
| Per-buffer including source+sink intercept | 1.0 | `:285-304` |
| Decode+convert steady state | 4.0 (budget 8) | `:155-206` |

The history comment (`:213-225`): 2.0 → 1.0 allocs/hop when kanal's `PendingRecv` `Box::pin` was
deleted, 1.0 → 0.0 with #175's inline dispatch. **There is currently nothing on the link hot path
for a custom queue to remove**: no copy, no allocation, no syscall (`Buffer::clone` for fan-out is
3 atomics — slot refcount in shm, arena header refcount, `Arc<OwnedFd>` — guarded by
`benches/memory_pool.rs`).

Remaining per-message costs, for honesty's sake:

- The ~400 B memcpy of the `Message` value into and out of the channel slot, twice per hop.
- Muxer inputs use `FuturesUnordered::push(recv_one(...))` per message
  (`unified_executor.rs:4522-4537`, `:4607`) — one `Arc<Task>` allocation per buffer per input pad.
  This is the only remaining per-message allocation on a data path, and it is muxer-only.
- Fan-out >1 goes through `futures::future::join_all` (a `Vec` per broadcast,
  `:2413-2430`); the single-consumer case moves without cloning and allocates nothing.
- tokio mpsc allocates its 32-slot `Block`s as a queue grows but recycles them (consistent with the
  measured 0.0 steady state). At ~400 B/message that is ~12.8 KB resident per deep link.

### 2.3 The three channel kinds, and the two custom queues we already have

1. **Data links** — tokio mpsc (Block, DropNewest) or the **leaky channel** for DropOldest
   (`src/pipeline/leaky.rs`, #169). The leaky channel exists precisely because tokio's `Sender` has
   no pop (`leaky.rs:1-7`). Its documented invariants are the house rules any replacement must obey:
   sends never await (no `select!` cancellation hazard), control entries are never evicted (pushed
   past capacity), and `recv` is cancel-safe — `Notify::notified()` is `enable()`d *before* the
   queue check, and the pop happens synchronously inside the returning poll (`leaky.rs:184-203`).
2. **Upstream event inboxes** (seek dispatch, #163) — `tokio::sync::mpsc::unbounded_channel<Event>`,
   one per node (`unified_executor.rs:1709-1737`, `UpstreamHop` at `:2685-2699`). Unboundedness is a
   **correctness requirement**, documented in place: a bounded upstream inbox can deadlock against a
   parent parked in `send().await` into a full downstream data channel.
3. **`AsyncRtBridge`** at hybrid RT boundaries (`src/pipeline/rt_bridge.rs`) — a cache-line-padded
   lock-free SPSC ring (`:270-378`) plus **two eventfds** (`data_available`, `space_available`),
   async side integrated via `tokio::io::unix::AsyncFd` (`:140-160`), EOS/error signaled
   out-of-band, capacity from `RtConfig::bridge_capacity` (default 16, low-latency-audio 4). This is
   a complete, working, benchmarked custom queue — proof the project builds one where it pays.

So the honest description of today's architecture is not "tokio mpsc everywhere"; it is **"the
cheapest primitive that satisfies each edge's invariants"** — tokio mpsc where cancel-safety and
backpressure matter, a Notify-based deque where eviction is needed, an eventfd SPSC ring where one
side cannot run tokio at all.

### 2.4 A coupling to keep in mind

Link capacity is not just queue depth: `output_slot_budget` (`unified_executor.rs:1558-1579`)
derives every element's output-arena slot count from it (max per pad across links, summed across
pads, + `IN_FLIGHT_MARGIN = 4`; bridged edges contribute `bridge_capacity` the same way). Any
change to link queueing is also a change to the arena-sizing contract (`src/memory/budget.rs`,
`docs/scheduling.md:72-74`).

---

## 3. Why tokio mpsc is the right in-process primitive (the kanal record)

This exact trade was already litigated, with data, when kanal was removed (commits `260c0e6`,
`7bbd12f`; `CLAUDE.md` "Seeking" section):

- **kanal's async recv is cancel-unsafe in a way that loses data**: with a sender already committed
  to the handoff, dropping the `ReceiveFuture` completes the handoff and *discards the message*
  while the sender's `send` returned `Ok`. A swallowed `Message::Eos` hangs a consumer forever.
  tokio's `Receiver::recv` is documented cancel-safe ("no messages were received"), which is what
  lets the three select-carrying loops (sink `:3359-3396`, transform `:3640-3676`, fed demuxer
  `:4060-4096`) `select!` directly over `&mut rx` with no stashed-future apparatus and no per-buffer
  boxing.
- **The measured cost of correctness was noise on the real path**: synthetic zero-work channel
  benchmarks regressed (`passthrough_stages/4` +17%, `fanout_policy/block` +95% — kanal spins before
  parking and those benchmarks are nothing but channel ping-pong), while decode went 220.6 → 221.2 ms
  (0.3%), decode+convert 3% *faster*, MKV demux 124 → 131 µs. The commit message concludes: "paying
  synthetic throughput to stop losing messages is the right side of that trade."
- The surviving mirror rule — **never put `Sender::send().await` in a `select!` branch** (the
  message is not sent but *is* dropped) — is upheld throughout the executor
  (`unified_executor.rs:2289-2292`).

Any replacement queue for in-process links therefore has to re-solve, at parity, all of:

1. **Strictly cancel-safe recv** — a dropped future must have consumed nothing (three `select!`
   loops + the muxer's `FuturesUnordered` depend on it).
2. **Closed-sender detection distinct from full** (#85, `tests/no_hang_on_error.rs`) — otherwise a
   source whose consumer died spins at 100 % CPU.
3. **Occupancy readable from both ends without locking** — `LinkFlowMonitor`
   (`src/pipeline/flow.rs:154-177`) samples after every send *and* every receive (a gated source
   stops sending, so sender-only sampling never sees the drain).
4. **Three policies on one primitive** — block-for-room, drop-newest, drop-oldest — with **control
   immunity** (Eos/Error/Event never dropped or evicted, FIFO-ordered with buffers).
5. **Tokio coop-budget participation** — `Block` links get it free from `send().await`; the lossy
   paths charge `consume_budget()` manually (`:2332-2334`) because a never-yielding producer starves
   its consumer in the worker's non-stealable LIFO slot (the AppSink lesson, gotcha 15).
6. **The `OutputBudget` contract** (§2.4).

None of these are hypothetical; each one is a shipped bug fix with a regression test. Meanwhile the
prize for winning is the removal of a ~400 B memcpy and tokio's (well-amortized) internal machinery
— on a path whose end-to-end cost is dominated by codecs and conversion by three orders of
magnitude. The alloc ratchet exists "so nobody re-adds [kanal] for the benchmarks"
(`7bbd12f`); the same reasoning applies to a bespoke ring.

There is also a subtler point: **a shm-resident queue buys nothing in-process by construction.**
Its two selling points are (a) payload sharing across address spaces — already delivered by the
arenas, orthogonally to the channel — and (b) avoiding serialization — already absent in-process.
What remains is a wakeup-primitive comparison (futex/eventfd vs. tokio's waker protocol), and
inside one tokio runtime the native waker path is strictly cheaper than a syscall-based doorbell.

---

## 4. What the shared-memory model provides — and deliberately doesn't

The mapping (`src/memory/shared_refcount.rs`) is `ArenaHeader (64 B) | ReleaseQueue (4160 B) |
SlotHeader[N] (8 B each: refcount+state atomics) | SlotData[N]`, memfd-backed, v4 layout with
stride/alignment recorded in the header and validated by `from_fd` (`:753-865`).

The design intent is stated in `docs/design.md:19,34`: cross-process refcounting is "atomics in the
shared mapping; lock-free release queue; **no messages**" — explicitly contrasted with PipeWire's
message-coordinated model. The one queue that lives in shared memory, the `ReleaseQueue`
(`:230-240`), is minimal on purpose:

- 1024 entries of bare `u32` slot indices; two-phase push (CAS tail, then store index); pop with a
  128-iteration spin budget (#171) so a producer preempted mid-push can't livelock the consumer;
  single-consumer discipline enforced by `reclaim_lock` in the header, not by construction.
- **No notification of any kind.** `SharedSlotRef::drop` pushes the index and walks away
  (`:1361-1375`). Discovery is by the owner polling `reclaim()`. A grep across `src/`, `docs/`,
  `plans/` finds **no futex, semaphore, pshared mutex, or shm-adjacent eventfd anywhere** — every
  notifying primitive in the tree (eventfd in `rt_bridge.rs`, `driver.rs`, `rt_scheduler.rs`;
  `Notify` in `leaky.rs`) is heap/fd-local.

That "no messages" choice is what keeps release O(1) and wait-free for any process holding a ref.
The cost is that anyone *waiting* for a slot must poll — which is exactly where the model's one
in-process wart lives (§6.2) — and that the queue as-built cannot serve as a general message
channel (u32 payloads, no doorbell, owner-only consumer).

Cross-process, the flow is: arena fd over `SCM_RIGHTS` (`src/memory/ipc.rs`, sync `UnixStream`,
max 4 fds/message), `SharedArena::from_fd` on the peer, then per-buffer `SharedIpcSlotRef`
(`{arena_id, slot_index, data_offset, len}`, rkyv) resurrected via `slot_from_ipc` (`:1096-1126`,
which refuses non-Allocated slots so a freed slot can't be revived). The payload is genuinely
zero-copy across the boundary; only the descriptor travels.

---

## 5. Prior art

**PipeWire** — the closest relative, and the strongest validation of the current split. All node
resources (buffers, io areas, metadata) live in memfd/dmabuf shared memory set up *before*
scheduling; the wakeup path is `eventfd` per node, driven by activation records whose `required`/
`pending` counters are atomically decremented by the driver — when a node's counter hits zero its
eventfd is signaled. The design doctrine is "the fd is the primary identifier": memfd for data,
eventfd/timerfd/signalfd for events, epoll to compose them. Parallax already mirrors this at its RT
boundary (`ActivationRecord` in `rt_scheduler.rs:249-262` is a direct homage, and
`AsyncRtBridge` is eventfd+ring). PipeWire does **not** put its wakeup queue in shared memory
either — shared data, fd signaling. ([Graph scheduling docs](https://docs.pipewire.org/page_scheduling.html),
[LWN overview](https://lwn.net/Articles/847412/))

**iceoryx2** — the state of the art for pure zero-copy IPC in Rust: pool-allocated shm segments,
loan/publish/read-in-place, lock-free, plus an event service for push notifications. Two lessons:
(1) its architecture agrees with ours — payload in shm, notification as a separate primitive;
(2) **it has no async API — tokio integration is an open roadmap item**. The hard part of "a custom
queue that integrates nicely with tokio" is precisely the part nobody has shipped as a library. If
Parallax ever wants a second opinion on the cross-process data plane, iceoryx2 is the crate to
evaluate — but it would still need the AsyncFd-style bridging we already wrote for `AsyncRtBridge`.
([repo](https://github.com/eclipse-iceoryx/iceoryx2), [FAQ](https://github.com/eclipse-iceoryx/iceoryx2/blob/main/FAQ.md),
[zero-copy walkthrough](https://ekxide.io/blog/how-to-implement-zero-copy-communication/))

**GStreamer** — the incumbent's `queue`/`multiqueue` elements are a `GMutex` + two `GCond`s
(`item_add`/`item_del`), i.e. not even lock-free, because blocking semantics, state changes, and
flushing make the condvar the simpler correct tool. Worth remembering when tempted to treat
lock-freedom as table stakes for a media pipeline's links: the dominant implementation has shipped
mutexes on this path for twenty years, because the queue is never the bottleneck — the elements are.
([gstqueue.h](https://github.com/Xilinx/gstreamer/blob/master/plugins/elements/gstqueue.h))

**tokio integration pattern** — the canonical way to make a foreign queue async is
`tokio::io::unix::AsyncFd` over an eventfd: waker registration rides epoll, `clear_ready()` on
spurious wakeups. `EventFd::wait_async` (`rt_bridge.rs:140-160`) already implements it. One
improvement worth making if the pattern spreads: it currently constructs the `AsyncFd` **per call**
(an epoll ctl add/remove per wait); a cached registration per bridge would amortize that.
([AsyncFd docs](https://docs.rs/tokio/latest/tokio/io/unix/struct.AsyncFd.html))

**Channel performance folklore** — comparisons like thingbuf's show tokio mpsc mid-pack on raw
throughput and SPSC rings an order of magnitude faster on synthetic benchmarks. Our own kanal data
(§3) shows why this doesn't transfer: at 16-deep queues carrying 400 B handles between elements
doing real work, channel choice is measurement noise, and the failure modes (cancel-safety,
lost wakeups) dominate the engineering cost. ([thingbuf comparison](https://github.com/hawkw/thingbuf/blob/main/mpsc_perf_comparison.md))

---

## 6. Gap analysis: where a custom shm queue would and wouldn't pay

| Edge type | Today | Payload copy? | Wakeup | Verdict |
|---|---|---|---|---|
| In-process async link | tokio mpsc / leaky | none | tokio waker | **Keep.** Nothing to win; five invariants to lose. |
| Async ↔ RT boundary | `AsyncRtBridge` ring + eventfd | none | eventfd/AsyncFd | **Already the custom queue.** Minor: cache the `AsyncFd`. |
| Cross-process link | rkyv msgs over Unix socket + sleep-polls | none (descriptors only) | socket readability + **1 ms poll** | **The real gap. Build the shm ring here.** §6.1 |
| Slot release → waiting producer | `ReleaseQueue`, no doorbell | n/a | **2 ms condvar poll** | **Add an optional doorbell.** §6.2 |

### 6.1 The IPC data path is where "a queue on our memory model" belongs

Current per-buffer mechanics (`src/elements/ipc/ipc_elements.rs`):

- `IpcSink::consume` rkyv-serializes `ControlMessage::BufferReady{slot, metadata}` into a fresh
  `Vec` with a 4-byte length prefix; `IpcSrc` copies into an `AlignedVec` and fully deserializes —
  **two allocations per message per direction** (`protocol.rs:291-323`).
- Backpressure is `pending_slots: VecDeque` capped at `max_pending = 16`, drained by `process_acks`
  with an **O(n) `retain` per ack**, and when full, `consume` waits in a **1 ms
  `tokio::time::sleep` poll loop** (`stall_tick`, `:290-306`, warning after 5 s). The ack signal is
  the socket itself; there is no doorbell.

This is the mirror image of the in-process situation: here serialization, polling, and O(n) scans
*do* exist, and shared memory is *already mapped on both sides*. A descriptor ring in the mapping is
the textbook fix, and every ingredient is in the house style:

**Sketch.** Extend the (currently fd-passing-only) sink-side arena — or a dedicated 2-ring control
segment — with two SPSC rings modeled on `ReleaseQueue`'s two-phase commit but with wider entries:

```
#[repr(C, align(64))] struct IpcRing {           // one per direction
    head: AtomicU32, tail: AtomicU32, _pad: [u8; 56],
    entries: [IpcDescriptor; N],                  // power of two
}
#[repr(C)] struct IpcDescriptor {                 // BufferReady / BufferDone
    seq: AtomicU32,                               // commit flag (replaces QUEUE_EMPTY sentinel)
    slot_index: u32, data_offset: u64, len: u64,
    pts: u64, dts: u64, duration: u64, flags: u32, // fixed metadata fast path
    meta_overflow: u32,                            // 0, or a slot index carrying rkyv'd Metadata
}
```

- **Doorbell**: one eventfd per direction, passed alongside the arena fd in the existing
  `send_fds` registration message (`MAX_FDS_PER_MESSAGE = 4` already accommodates arena + 2
  doorbells). Producer: push, then `notify()`. Consumer: drain ring, then `AsyncFd::readable().await`
  — exactly `AsyncRtBridge::push_async`/`wait_async`, so the cancel-safety story is the one we
  already trust (the ring pop is synchronous; the await holds no message).
- **Backpressure**: ring-full replaces `max_pending`; the sink awaits the `space` doorbell instead
  of sleep-polling. The 5 s stall warning keeps its semantics (a timeout around the await).
- **Metadata**: hot fields (pts/dts/duration/flags/dims) go in the fixed-size descriptor; the rkyv
  path remains for the rare buffer with custom metadata, via an overflow slot. Common case: zero
  serialization, zero allocation, zero syscalls beyond one eventfd write/read pair (and even those
  amortize under load — an already-signaled eventfd needs no new syscall from the producer, and a
  busy consumer drains many entries per wakeup).
- The Unix socket stays for what it is good at: registration, fd passing, shutdown, errors —
  low-rate control, exactly PipeWire's split.

**Prerequisites (do first, they're bugs today):**
- `arena_id` comes from a process-local `static AtomicU64` (`shared_refcount.rs:513-518`); two peer
  processes creating arenas mint colliding ids, and `SharedArenaCache` keys on it. Needs a
  process-unique component (pid + counter, or random 64-bit).
- `memory::ipc` is sync `UnixStream` only; the registration path needs a non-blocking/tokio variant
  or must stay confined to setup (acceptable).

**What it must not break**: the sink-side arena's "exists only to pass its fd" comment
(`ipc_elements.rs:81-88`) — the data slots still come from the *upstream* element's arena; the ring
only replaces the message channel. Effort estimate: comparable to `rt_bridge.rs` (~800 lines with
tests), since it is structurally the same component with wider entries and an fd-passing setup step.

### 6.2 A doorbell for the release queue

`FixedBufferPool::acquire` documents its own wart (`buffer_pool.rs:44-50`, `:391-399`): a slot
detached by `into_buffer()` returns through the release queue, "which notifies nobody", so the
condvar waits in 2 ms slices (`DETACH_POLL`) and `try_acquire` calls `reclaim()` every time. The
same polling shape appears wherever someone waits on slot return.

Fix: an optional eventfd doorbell stored beside the arena (fd-local, *not* in the mapping — it can
be passed via SCM_RIGHTS for cross-process waiters). `SharedSlotRef::drop` rings it after a
successful `try_push`; `FixedBufferPool` waits on it instead of the timeout; an async waiter gets
`AsyncFd`. Cost on the release path: one `write(2)` per release *only when someone is waiting*
(guard with a `waiters: AtomicU32` in the header — a zero-waiter release stays syscall-free, which
preserves today's wait-free release for the common case). This also gives `OutputArena::admit` a
principled way to wait briefly instead of failing into the shed path when the arena is transiently
full.

### 6.3 Defects to record regardless

1. **Full release ring ⇒ permanent slot leak.** `SharedSlotRef::drop` (`:1361-1375`) drops the
   index on the floor when `try_push` fails; the comment says the slot "will be leaked until the
   owner does a full scan" — but no full-scan reclaim exists anywhere (`reclaim()` only drains the
   ring; nothing scans for `refcount == 0 && state == Allocated`). At 1024 entries vs. today's slot
   counts this is hard to hit, but it is silent and unrecoverable. Fix: implement the scan as a
   slow-path in `reclaim()` when `try_pop` returns nothing but `free_count()` disagrees with
   expectations, or on an explicit `reclaim_full()`.
2. **Head-of-line stall** in `try_pop` (mitigated by the #171 spin budget, but worth documenting):
   one preempted producer between CAS-tail and store-index stalls the whole drain until it
   reschedules.
3. **`AtomicBitmap`** (`src/memory/bitmap.rs`) is exported and used by nothing, and `acquire()` is
   an O(n) header scan — if slot counts ever grow past a few hundred, wire the bitmap in (it needs
   an shm-placeable rewrite first; it currently heap-allocates).

---

## 7. Recommendations

In order:

1. **Keep in-process links exactly as they are.** Document the "shm = data plane, channels =
   signaling plane" doctrine in `docs/design.md` so the question has a canonical answer (this
   report can seed it). The alloc ratchet and `benches/throughput.rs` already defend the status quo
   empirically.
2. **Fix the pre-existing defects**: arena-id uniqueness (§6.1 prereq), release-ring full-leak
   (§6.3.1). Both are small, self-contained, and independent of any queue work.
3. **Build the IPC descriptor ring + doorbells** (§6.1) when multi-process pipelines become a real
   workload. It is the only place where a custom queue on the memory model is the right tool, the
   design is a transliteration of `AsyncRtBridge`, and it deletes serialization, the 1 ms ack poll,
   and the O(n) pending scan in one move.
4. **Add the release-queue doorbell** (§6.2) opportunistically — it deletes the 2 ms pool poll and
   gives `admit()` a wait primitive. Low risk; keep it optional so the zero-waiter release path
   stays syscall-free.
5. **Minor**: cache the `AsyncFd` in `EventFd::wait_async` instead of re-registering per call;
   consider replacing the muxer's per-message `FuturesUnordered` re-push if muxer throughput ever
   matters (it is the last per-buffer allocation on a data path).

Explicitly **not** recommended: adopting iceoryx2 for the data plane (its async story is unshipped
and our arena refcounting is the part it doesn't replace), or a lock-free rewrite of the in-process
links (the kanal episode is the controlled experiment for that, and correctness lost less than the
benchmarks suggested it would).

---

## Appendix A — key evidence index

| Claim | Where |
|---|---|
| `Message` enum, unboxed by intent | `src/pipeline/unified_executor.rs:2191-2218` |
| Link channel creation, capacity default 16, `.max(1)` kanal fossil | `unified_executor.rs:1587-1611`, `:81`, `:111` |
| 0 allocs/buffer/hop ratchet + history | `tests/media_alloc_tests.rs:213-225`, `:287-305` |
| Buffer clone = 3 atomics, no syscall | `src/memory/shared_refcount.rs:1342,1361`, `benches/memory_pool.rs` |
| Leaky channel invariants (cancel-safe recv, control immunity) | `src/pipeline/leaky.rs:1-26`, `:117-154`, `:184-203` |
| Unbounded upstream inbox = deadlock avoidance | `unified_executor.rs:2685-2699` |
| `AsyncRtBridge` ring + 2 eventfds + AsyncFd | `src/pipeline/rt_bridge.rs:96-168`, `:270-378`, `:399-584` |
| kanal removal rationale + benchmarks | commits `260c0e6`, `7bbd12f`; `CLAUDE.md` |
| Closed-detection load-bearing | `unified_executor.rs:2280-2292`, `tests/no_hang_on_error.rs` |
| Flow monitor samples both ends | `src/pipeline/flow.rs:154-177`, `unified_executor.rs:2272-2278`, `:2381-2402` |
| Link capacity → arena budget | `unified_executor.rs:1558-1579`, `src/memory/budget.rs`, `src/memory/defaults.rs:156` |
| `ReleaseQueue` mechanics, spin budget, reclaim lock | `shared_refcount.rs:230-351`, `:400-408`, `:907-937` |
| No shm notification primitive exists | grep futex/semaphore/pshared over `src/`, `docs/`, `plans/` — zero hits |
| Pool 2 ms poll + rationale | `src/memory/buffer_pool.rs:44-50`, `:376-403`; `docs/memory.md:146-155` |
| IPC per-buffer path: rkyv framing, 1 ms ack poll, O(n) retain | `src/elements/ipc/ipc_elements.rs:290-357`, `src/elements/ipc/protocol.rs:291-323` |
| Arena fd via SCM_RIGHTS | `src/memory/ipc.rs:38-72`, `ipc_elements.rs:201-225` |
| `arena_id` process-local counter | `shared_refcount.rs:513-518` |
| Release-ring full ⇒ permanent leak (stale comment) | `shared_refcount.rs:1361-1375` |
| "No messages" design intent | `docs/design.md:19,34`; deployment table `:103-107` |

## Appendix B — external sources

- PipeWire graph scheduling & activation records: <https://docs.pipewire.org/page_scheduling.html>
- PipeWire design overview (fd-first doctrine): <https://lwn.net/Articles/847412/>, <https://bootlin.com/blog/an-introduction-to-pipewire/>
- iceoryx2 (zero-copy IPC, async on roadmap): <https://github.com/eclipse-iceoryx/iceoryx2>, <https://github.com/eclipse-iceoryx/iceoryx2/blob/main/FAQ.md>, <https://ekxide.io/blog/how-to-implement-zero-copy-communication/>
- GStreamer queue internals (mutex + GCond): <https://github.com/Xilinx/gstreamer/blob/master/plugins/elements/gstqueue.h>
- tokio `AsyncFd` (foreign-fd async integration): <https://docs.rs/tokio/latest/tokio/io/unix/struct.AsyncFd.html>
- MPSC channel performance comparison (context for §5): <https://github.com/hawkw/thingbuf/blob/main/mpsc_perf_comparison.md>
