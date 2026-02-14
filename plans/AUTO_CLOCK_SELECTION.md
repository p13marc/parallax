# Plan: Automatic Clock Selection

**Priority:** Medium
**Effort:** Small (1-2 days)

---

## Problem

Parallax has a complete clock infrastructure (`Clock`, `ClockProvider`, `PipelineClock`, `SystemClock`, `AlsaClock`) but clock selection is **manual**:

```rust
let sink = AlsaSink::new("default", format)?;
let clock = Arc::new(sink.create_clock());
pipeline.set_clock(clock);  // Manual!
```

GStreamer and PipeWire both auto-select the best clock from pipeline elements. We should too.

## Current State

What exists:
- `Clock` trait with `now()`, `flags()`, `resolution()`, `name()`
- `ClockProvider` trait with `provide_clock()`, `clock_priority()`
- `PipelineClock` wrapping an `Arc<dyn Clock>` with base time
- `Pipeline::set_clock()` and `Pipeline::use_clock_from()`
- `SystemClock` (default, priority 0)
- `AlsaClock` (priority 150, but AlsaSink can't impl ClockProvider due to `!Sync` PCM)

What's missing:
- `AsyncElementDyn` has no way to expose `ClockProvider`
- Pipeline can't enumerate elements to find clock providers
- No automatic selection during `prepare()` or `run()`

## Design

### Approach: Add `as_clock_provider()` to `AsyncElementDyn`

Same pattern as `as_sync_element()` and `as_any()`:

```rust
// In AsyncElementDyn trait
fn as_clock_provider(&self) -> Option<&dyn ClockProvider> {
    None  // Default: not a clock provider
}
```

Elements that can provide clocks override this. The pipeline enumerates all elements, collects clock providers, and selects the highest-priority one.

### The AlsaSink Problem

AlsaSink can't implement `ClockProvider` directly because `alsa::PCM` contains raw pointers (`!Sync`). But `AlsaClock` doesn't need the PCM handle — it uses `Instant` + sample rate.

Solution: AlsaSink creates an `AlsaClock` at construction time and stores it as `Arc<AlsaClock>`. Then `as_clock_provider()` returns a wrapper that provides this pre-created clock.

## Implementation Steps

### Step 1: Add `as_clock_provider()` to `AsyncElementDyn`

**File:** `src/element/traits.rs`

Add to `AsyncElementDyn` trait:

```rust
fn as_clock_provider(&self) -> Option<&dyn ClockProvider> {
    None
}
```

### Step 2: Implement for AlsaSink

**File:** `src/elements/device/alsa.rs`

```rust
struct AlsaSinkClockProvider {
    clock: Arc<AlsaClock>,
}

impl ClockProvider for AlsaSinkClockProvider {
    fn provide_clock(&self) -> Option<Arc<dyn Clock>> {
        Some(self.clock.clone())
    }
    fn clock_priority(&self) -> u32 { 150 }
}
```

Store the provider in AlsaSink, return from `as_clock_provider()` in its adapter.

### Step 3: Add `Pipeline::select_clock()`

**File:** `src/pipeline/graph.rs`

```rust
pub fn select_clock(&mut self) {
    let mut best: Option<(Arc<dyn Clock>, u32)> = None;

    for (_id, node) in self.nodes() {
        if let Some(provider) = node.element().as_clock_provider() {
            if let Some(clock) = provider.provide_clock() {
                let priority = provider.clock_priority();
                if best.as_ref().map_or(true, |(_, p)| priority > *p) {
                    best = Some((clock, priority));
                }
            }
        }
    }

    if let Some((clock, _)) = best {
        self.clock = PipelineClock::new(clock);
    }
    // Otherwise keep SystemClock default
}
```

### Step 4: Call from executor before start

**File:** `src/pipeline/unified_executor.rs`

In the `start()` method, call `pipeline.select_clock()` before distributing the clock to elements.

### Step 5: Implement for other device sinks (optional)

If PipeWireSink or other device sinks can provide clocks, add `as_clock_provider()` to their adapters too.

### Step 6: Test

- Unit test: pipeline with AlsaSink auto-selects AlsaClock over SystemClock
- Unit test: pipeline without clock providers keeps SystemClock
- Unit test: highest priority wins when multiple providers exist

## File Changes

| File | Change |
|------|--------|
| `src/element/traits.rs` | Add `as_clock_provider()` to `AsyncElementDyn` |
| `src/elements/device/alsa.rs` | Store `AlsaSinkClockProvider`, implement in adapter |
| `src/pipeline/graph.rs` | Add `select_clock()` method |
| `src/pipeline/unified_executor.rs` | Call `select_clock()` before start |
| `tests/timestamp_tests.rs` | Add auto-selection tests |

## Success Criteria

1. Pipeline with AlsaSink auto-selects AlsaClock without manual `set_clock()`
2. Pipeline without device sinks defaults to SystemClock
3. Manual `set_clock()` still works and overrides auto-selection
4. No regressions in existing tests
