# Plan 15: RT Scheduling — From Dead Code to Working System

**Priority:** High  
**Effort:** Large (2-3 weeks)  
**Depends on:** None (all infrastructure exists)

---

## Problem Statement

Parallax has a well-designed RT scheduling system inspired by PipeWire, but it's entirely dead code. The infrastructure — lock-free bridges, activation records, graph partitioning, driver system — is implemented and unit-tested, but never actually invoked at runtime. Every pipeline runs purely in Tokio async tasks.

### What Exists (and Works)

| Component | File | Status |
|-----------|------|--------|
| Lock-free SPSC bridge | `rt_bridge.rs` | Production-ready, tested |
| Activation records | `rt_scheduler.rs` | Implemented, tested |
| Graph partitioning | `rt_scheduler.rs` | Implemented, tested |
| Driver system | `driver.rs` | Implemented, tested |
| Strategy detection | `unified_executor.rs` | Implemented, tested |
| `spawn_data_thread()` | `rt_scheduler.rs` | Implemented but broken |

### What's Broken

1. **The unified executor never spawns RT threads** — `run_hybrid()` returns `rt_handles = Vec::new()` (line 678 of `unified_executor.rs`).

2. **All elements are async** — The RT data thread calls `element.process()` (an async method) via `block_on()`, which defeats the purpose of RT scheduling. There's no sync processing path.

3. **No elements declare themselves as RT-safe** — Every built-in element returns `is_rt_safe() = false` and `Affinity::Auto`, so the strategy detector never selects `ElementStrategy::RealTime`.

4. **Incomplete dependency signaling** — The data thread resets activation records but doesn't propagate signals to downstream RT nodes after processing.

5. **No driver integration** — The data thread's processing loop doesn't integrate with the `TimerDriver` or `ManualDriver`.

---

## Design Approach

### Core Insight

The fundamental problem is an async/sync impedance mismatch. The solution is NOT to add sync versions of all element traits (that would double the API surface). Instead:

1. **Add a `process_sync()` method to `DynAsyncElement`** that wraps `block_on()` in a controlled way for elements that aren't truly RT-safe, and provides a direct sync path for elements that are.
2. **Create a `SyncElement` trait** — a minimal sync-only trait for elements that are genuinely RT-safe. These can be wrapped to implement `AsyncElementDyn` for pipeline compatibility, but run natively in the RT data thread.
3. **Mark existing simple elements as RT-safe** — PassThrough, Identity, simple transforms that don't allocate.

### Architecture After This Plan

```
┌─────────────────────────────────────────────────────────────────┐
│                        Tokio Runtime                            │
│  ┌──────────────┐  ┌──────────────┐                             │
│  │  FileSrc     │  │  TcpSrc      │  (I/O-bound, async)         │
│  │  affinity:   │  │  affinity:   │                             │
│  │  Async       │  │  Async       │                             │
│  └──────┬───────┘  └──────┬───────┘                             │
│         │                 │                                      │
│         ▼                 ▼                                      │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  AsyncRtBridge (lock-free SPSC + eventfd)            │       │
│  │  push_async() on Tokio side                          │       │
│  └──────────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│               RT Data Thread (SCHED_FIFO)                       │
│                                                                  │
│  Driver: TimerDriver (e.g., 5.3ms quantum for 48kHz/256)        │
│                                                                  │
│  Each cycle:                                                     │
│  1. Wait for driver trigger (eventfd)                            │
│  2. Reset activation records                                     │
│  3. For each node in topo order:                                 │
│     a. try_pop() from input bridge (if boundary)                 │
│     b. process_sync(buffer)  ← NEW sync path                    │
│     c. try_push() to output bridge (if boundary)                 │
│     d. Decrement downstream pending counts                       │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │ Decoder  │──│ Mixer    │──│ AudioOut │                        │
│  │ (sync)   │  │ (sync)   │  │ (sync)   │                       │
│  └──────────┘  └──────────┘  └──────────┘                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Tokio Runtime                            │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  AsyncRtBridge (try_pop on RT side, async on Tokio)  │       │
│  └──────────────────────────────────────────────────────┘       │
│         │                                                        │
│         ▼                                                        │
│  ┌──────────────┐                                               │
│  │  FileSink    │  (I/O-bound, async)                            │
│  └──────────────┘                                               │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Add `SyncElement` Trait (Small)

**File:** `src/element/traits.rs`

Add a minimal trait for elements that process synchronously without any async, allocation, or blocking:

```rust
/// Trait for RT-safe synchronous element processing.
///
/// Elements implementing this trait can run in the RT data thread
/// without blocking on async operations. They must not:
/// - Allocate memory (no Vec::push, no Box::new)
/// - Block on I/O (no file/network operations)
/// - Take locks that might be held by non-RT threads
///
/// The buffer is processed in-place or replaced.
pub trait SyncElement: Send {
    /// Process a buffer synchronously. Returns None for filtered/dropped buffers.
    fn process_sync(&mut self, buffer: Buffer) -> Result<Option<Buffer>>;

    /// Flush at EOS. Returns any buffered data.
    fn flush_sync(&mut self) -> Result<Option<Buffer>> { Ok(None) }
}
```

This is intentionally minimal. Elements that are RT-safe implement `SyncElement`; the existing `Element`/`Transform` traits remain unchanged.

**Also add a wrapper** to bridge `SyncElement` into the async pipeline world:

```rust
/// Adapter: SyncElement → AsyncElementDyn
pub struct SyncElementAdapter<T: SyncElement> {
    inner: T,
}
```

### Step 2: Mark Existing Elements as RT-Safe (Small)

**Files:** Various element files

Audit and mark simple elements that don't allocate in their hot path:

| Element | File | RT-Safe? | Notes |
|---------|------|----------|-------|
| `PassThrough` | `elements/util/` | Yes | No-op |
| `Identity` | `elements/util/` | Yes | No-op |
| `Map` (with sync closure) | `elements/transform/` | Conditional | Depends on closure |
| `Filter` (with sync closure) | `elements/transform/` | Conditional | Depends on closure |
| `Valve` | `elements/flow/` | Yes | Just drops/passes |

For each, implement `SyncElement` and set `is_rt_safe() = true`, `affinity() = Auto`.

### Step 3: Create RT-Safe Audio Elements (Medium)

**Files:** New files in `src/elements/audio/` (or `src/elements/transform/`)

Create simple audio processing elements that are genuinely RT-safe:

1. **`Gain`** — Multiply samples by a constant. No allocation.
2. **`Mixer`** — Mix N inputs to 1 output. Pre-allocated output buffer.
3. **`SilenceDetect`** — Detect silence and tag metadata. No allocation.

These serve as proof-of-concept RT elements and are useful in their own right.

```rust
pub struct Gain {
    factor: f32,
}

impl SyncElement for Gain {
    fn process_sync(&mut self, mut buffer: Buffer) -> Result<Option<Buffer>> {
        // Modify samples in-place — no allocation
        let data = buffer.data_mut();
        for sample in data.chunks_exact_mut(4) {
            let val = f32::from_le_bytes(sample.try_into().unwrap());
            let out = val * self.factor;
            sample.copy_from_slice(&out.to_le_bytes());
        }
        Ok(Some(buffer))
    }
}

impl Gain {
    pub fn is_rt_safe(&self) -> bool { true }
    pub fn affinity(&self) -> Affinity { Affinity::Auto }
}
```

### Step 4: Wire Up RT Thread Spawning in Unified Executor (Large — Core Work)

**File:** `src/pipeline/unified_executor.rs`

Replace the placeholder in `run_hybrid()`:

```rust
// BEFORE (placeholder):
let rt_handles = Vec::new();

// AFTER (actual implementation):
let rt_handles = self.spawn_rt_threads(
    pipeline,
    partition,
    scheduler,
)?;
```

Implement `spawn_rt_threads()`:

1. **Extract RT elements from the pipeline graph** — Move element ownership from pipeline nodes to the data thread. This requires `Pipeline::take_element(node_id) -> Option<Box<DynAsyncElement>>`.

2. **Build input/output bridge maps** — Map each RT node to its input and output bridges.

3. **Create the driver** — If `self.config.driver` is set, use it. Otherwise, create a `TimerDriver` with the configured quantum.

4. **Call `spawn_data_thread()`** — Pass the extracted elements, bridges, activations, and driver trigger.

5. **Wire async tasks to bridges** — For async nodes adjacent to RT boundaries, connect their channel send/recv to the bridge's `push_async()` / `pop_async()` instead of to another channel.

### Step 5: Fix the Data Thread Processing Loop (Medium)

**File:** `src/pipeline/rt_scheduler.rs`, function `spawn_data_thread()`

Fix the three issues in the current loop:

**5a. Replace `block_on()` with sync dispatch:**

```rust
// BEFORE (broken):
let result = handle.block_on(element.process(input));

// AFTER (fixed):
let result = if let Some(sync_elem) = element.as_sync_element() {
    // True RT-safe path — no async, no blocking
    sync_elem.process_sync(input.unwrap_or_else(|| Buffer::empty()))
} else {
    // Fallback for non-RT-safe elements in Hybrid mode
    // This is suboptimal but allows gradual migration
    tracing::warn_once!("Element in RT thread is not SyncElement, using block_on fallback");
    let rt = tokio::runtime::Handle::try_current();
    match rt {
        Ok(handle) => handle.block_on(element.process(input)),
        Err(_) => {
            // Should not happen in a properly set up pipeline
            Err(Error::Element("No Tokio runtime available for non-sync element in RT thread".into()))
        }
    }
};
```

**5b. Add downstream dependency signaling:**

After processing a node, decrement pending counts for all downstream RT nodes:

```rust
// After processing node_id successfully:
for downstream_id in &downstream_map[&node_id] {
    if let Some(activation) = activations.get(downstream_id) {
        if activation.decrement_pending() {
            // Node is now ready to process
            activation.signal()?;
        }
    }
}
```

**5c. Integrate driver timing:**

Replace the spin-loop wait with proper driver integration:

```rust
// Wait for driver signal (start of cycle)
driver_trigger.wait()?;  // Blocks until next cycle

// Reset all activations for this cycle
for activation in activations.values() {
    activation.reset_pending();
}

// Process nodes with zero pending count (sources/bridge-fed nodes)
for &node_id in &processing_order {
    let activation = &activations[&node_id];
    if !activation.is_ready() {
        // Wait for this node's dependencies
        activation.trigger.wait()?;
    }
    // ... process node ...
}
```

### Step 6: Add `as_sync_element()` to Element Trait Hierarchy (Small)

**File:** `src/element/traits.rs`

Add a method to `AsyncElementDyn` (or its wrapper) to dynamically check if an element supports sync processing:

```rust
// In DynAsyncElement or a new trait:
pub trait MaybeSyncElement {
    /// If this element supports sync processing, return a reference.
    fn as_sync_element(&mut self) -> Option<&mut dyn SyncElement> { None }
}
```

Elements that implement `SyncElement` and are wrapped via `SyncElementAdapter` return `Some(self)`. All others return `None`.

### Step 7: Build Downstream Map for Dependency Signaling (Small)

**File:** `src/pipeline/rt_scheduler.rs`

Add a method to `RtScheduler` that builds a `HashMap<NodeId, Vec<NodeId>>` mapping each RT node to its downstream RT dependents. Pass this into `spawn_data_thread()`.

```rust
pub fn build_downstream_map(
    &self,
    partition: &GraphPartition,
    pipeline: &Pipeline,
) -> HashMap<NodeId, Vec<NodeId>> {
    let rt_set: HashSet<_> = partition.rt_nodes.iter().copied().collect();
    let mut map = HashMap::new();
    
    for &node_id in &partition.rt_nodes {
        let downstreams: Vec<NodeId> = pipeline.children(node_id)
            .filter(|(child_id, _)| rt_set.contains(child_id))
            .map(|(child_id, _)| child_id)
            .collect();
        map.insert(node_id, downstreams);
    }
    
    map
}
```

### Step 8: Async Bridge Tasks (Medium)

**File:** `src/pipeline/unified_executor.rs`

For async nodes adjacent to RT boundaries, spawn "bridge relay" tasks that shuttle data between kanal channels (used by async tasks) and `AsyncRtBridge` (used by RT threads):

```rust
/// Spawn a task that reads from an async channel and pushes to an RT bridge.
fn spawn_async_to_rt_relay(
    rx: AsyncReceiver<Message>,
    bridge: Arc<AsyncRtBridge>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(Message::Buffer(buf)) => {
                    bridge.push_async(buf).await?;
                }
                Ok(Message::Eos) => break,
                Err(_) => break,
            }
        }
        Ok(())
    })
}

/// Spawn a task that pops from an RT bridge and sends to an async channel.
fn spawn_rt_to_async_relay(
    bridge: Arc<AsyncRtBridge>,
    tx: AsyncSender<Message>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        loop {
            match bridge.pop_async().await {
                Ok(Some(buf)) => {
                    tx.send(Message::Buffer(buf)).await.ok();
                }
                Ok(None) => {
                    tx.send(Message::Eos).await.ok();
                    break;
                }
                Err(_) => break,
            }
        }
        Ok(())
    })
}
```

### Step 9: Example — Hybrid Pipeline (Small)

**File:** `examples/50_hybrid_pipeline.rs`

Create an example demonstrating the RT scheduling working end-to-end:

```rust
//! Example 50: Hybrid Pipeline (Async I/O + RT Processing)
//!
//! Demonstrates a pipeline where:
//! - FileSrc runs as async Tokio task (I/O-bound)
//! - Gain element runs in RT thread (CPU-bound, RT-safe)
//! - FileSink runs as async Tokio task (I/O-bound)
//!
//! cargo run --example 50_hybrid_pipeline
```

### Step 10: Integration Tests (Medium)

**File:** `tests/rt_integration.rs`

Write tests that verify the complete RT path:

1. **Test: RT thread is actually spawned** — Create a pipeline with an RT-safe element, verify the thread exists via thread name.
2. **Test: Data flows through RT bridge** — Source (async) → Bridge → PassThrough (RT) → Bridge → Sink (async). Verify all data arrives.
3. **Test: Driver timing** — Create a pipeline with a `ManualDriver`, verify processing only happens when the driver fires.
4. **Test: Mixed pipeline** — 3 async elements + 2 RT elements, verify correct partitioning and data flow.
5. **Test: RT priority** — If running with `CAP_SYS_NICE`, verify the thread has SCHED_FIFO priority.

---

## File Changes Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `src/element/traits.rs` | Add | `SyncElement` trait, `MaybeSyncElement` trait |
| `src/element/mod.rs` | Modify | Export new traits |
| `src/elements/util/passthrough.rs` | Modify | Implement `SyncElement`, `is_rt_safe() = true` |
| `src/elements/util/identity.rs` | Modify | Implement `SyncElement`, `is_rt_safe() = true` |
| `src/elements/flow/valve.rs` | Modify | Implement `SyncElement`, `is_rt_safe() = true` |
| `src/elements/audio/gain.rs` | Add | New RT-safe audio gain element |
| `src/pipeline/unified_executor.rs` | Modify | Wire up `spawn_rt_threads()`, bridge relay tasks |
| `src/pipeline/rt_scheduler.rs` | Modify | Fix data thread loop, add downstream map |
| `examples/50_hybrid_pipeline.rs` | Add | Hybrid pipeline example |
| `tests/rt_integration.rs` | Add | Integration tests |

---

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Element extraction breaks pipeline graph | Use `Pipeline::take_element()` that leaves a placeholder node |
| RT thread panics crash the process | Wrap data thread body in `catch_unwind`, signal error via eventfd |
| No `CAP_SYS_NICE` for RT priority | Graceful fallback with warning (already implemented) |
| EOS handling in hybrid mode | Bridge relay tasks propagate EOS signals both directions |
| `block_on()` fallback for non-sync elements | Log warning, document that true RT requires `SyncElement` |

---

## Success Criteria

1. `cargo run --example 50_hybrid_pipeline` runs and produces correct output
2. RT thread is visible via `/proc/<pid>/task/` with the expected name
3. Integration tests pass: data flows through async → RT → async boundaries
4. `ManualDriver` test proves driver-gated processing works
5. No regressions: all existing tests pass, all existing examples work unchanged
6. Performance: RT processing loop runs without allocation (verified via `#[global_allocator]` counting in test)
