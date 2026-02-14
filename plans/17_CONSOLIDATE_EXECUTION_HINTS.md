# Plan 17: Consolidate ExecutionHints / Affinity / is_rt_safe

## Problem Statement

Every element trait (`Source`, `Sink`, `Element`, `Transform`, `AsyncSource`, `AsyncSink`, `AsyncTransform`, `Demuxer`, `Muxer`, plus Simple* and Pipeline* variants) defines three separate methods for scheduling decisions:

```rust
fn affinity(&self) -> Affinity { ... }
fn is_rt_safe(&self) -> bool { ... }
fn execution_hints(&self) -> ExecutionHints { ... }
```

These three methods encode overlapping information, can contradict each other, and create a massive maintenance burden:

- **92+ method definitions** across `traits.rs` and `pipeline_element.rs` (31+31+30)
- **53 override sites** across element implementations in `src/elements/`
- **11 adapter types** each delegating all 3 methods (33 more delegation sites)
- **1 decision function** (`determine_element_strategy`) that consumes all three as separate parameters and applies ad-hoc priority rules to resolve conflicts

### The Redundancy

| Information | `is_rt_safe()` | `affinity()` | `execution_hints()` |
|-------------|---------------|-------------|---------------------|
| "Can run in RT thread" | **Primary** | Implied by `RealTime` | Implied by `latency: Low/UltraLow` |
| "Should run in RT thread" | Combined with affinity | **Primary** (`RealTime`) | `latency: UltraLow/Low` |
| "Should run async" | Negation | **Primary** (`Async`) | `processing: IoBound` |
| "Should be isolated" | - | - | `trust_level`, `uses_native_code` |

An element declaring `is_rt_safe() = true` but `affinity() = Async` and `latency = UltraLow` sends three contradictory signals. The decision function silently picks one based on rule ordering.

### Conflict Scenarios in Current Code

| Scenario | What Happens | Warning? |
|----------|-------------|----------|
| `affinity=RealTime` + `is_rt_safe=false` | Falls back to Async | Yes |
| `affinity=RealTime` + `processing=IoBound` | IoBound rule wins, forced Async | **No** |
| `affinity=Async` + `latency=UltraLow` + `is_rt_safe=true` | Async wins, latency ignored | **No** |
| `trust_level=Untrusted` + `is_rt_safe=true` + `affinity=RealTime` | Isolated, RT ignored | **No** |

### Real-World Pattern

Looking at actual element implementations, the pattern is always one of a few profiles:

```rust
// Pattern A: "I'm an RT-safe transform" (Gain, PassThrough, Tee, filters)
fn is_rt_safe(&self) -> bool { true }
fn affinity(&self) -> Affinity { Affinity::Auto }
fn execution_hints(&self) -> ExecutionHints {
    ExecutionHints { processing: CpuBound, latency: Low, ..trusted() }
}

// Pattern B: "I'm an I/O device" (ALSA, PipeWire, V4L2, screen capture)
fn is_rt_safe(&self) -> bool { false }
fn affinity(&self) -> Affinity { Affinity::Async }
fn execution_hints(&self) -> ExecutionHints { ExecutionHints::io_bound() }

// Pattern C: "I'm a codec with native code" (decoders, encoders)
// (only execution_hints, leaving affinity and is_rt_safe at default)
fn execution_hints(&self) -> ExecutionHints { ExecutionHints::native() }

// Pattern D: "I'm a simple element" (most elements)
// (no overrides, all defaults)
```

Every element falls into one of these 4 profiles. The three methods are never used independently — they always move together in predictable combinations.

## Design Research

### How Other Frameworks Handle This

| Framework | Scheduling API | # of Methods | Who Decides | Key Insight |
|-----------|---------------|-------------|-------------|-------------|
| **PipeWire** | Single properties bag + 2-bit node flags (`RT`, `ASYNC`) | 2 | Server | Flat key-value pairs, infinitely extensible, no conflicts possible because server interprets holistically |
| **JACK** | None — server dictates everything | 0 | Server | Clients declare nothing about scheduling; the server owns all decisions |
| **GStreamer** | Scattered across 4 subsystems (~13 functions) | ~13 | Negotiated between pads | Evolved organically, widely considered over-complex |
| **Vulkan** | Single struct with bitflags per queue family | 1 query | Application | Hardware declares capabilities in one place; application selects |
| **Bevy ECS** | Builder-pattern constraints at registration time | ~10 | Scheduler | Scheduling constraints are external to the system, not declared inside it |

**Key takeaway**: The best designs (PipeWire, Vulkan) use a **single declaration point**. The worst (GStreamer) scatter scheduling info across multiple overlapping APIs. We are currently closer to GStreamer.

**PipeWire's approach is most relevant** to Parallax because it solves the same problem (mixed RT/async/isolated scheduling for media elements):
- A node sets `SPA_NODE_FLAG_RT` (a single bit) to declare RT capability
- All other preferences go into a flat properties bag
- The daemon reads everything holistically and makes the final decision
- No conflicts are possible because there's one source of truth

## Proposed Solution

### Consolidate into a single `ExecutionHints` struct

Replace the three-method pattern with a single `execution_hints()` method. Move `is_rt_safe` and `affinity` into `ExecutionHints` as fields.

### New `ExecutionHints` struct

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct ExecutionHints {
    // === Scheduling (replaces affinity() and is_rt_safe()) ===

    /// Whether this element can safely run in a real-time context.
    ///
    /// RT-safe means: no heap allocation, no blocking I/O, no locks
    /// shared with non-RT threads, bounded deterministic execution time.
    ///
    /// Default: `false` (conservative).
    pub rt_safe: bool,

    /// Preferred scheduling strategy.
    ///
    /// - `Auto`: Let the executor decide based on other hints (default)
    /// - `Async`: Must run in Tokio (I/O-bound elements)
    /// - `RealTime`: Prefer RT thread (only honored if `rt_safe = true`)
    ///
    /// Default: `Auto`.
    pub affinity: Affinity,

    // === Isolation ===

    /// Trust level of the data being processed.
    pub trust_level: TrustLevel,

    /// Whether the element might crash on bad input.
    pub crash_safe: bool,

    /// Whether the element uses native code (FFI).
    pub uses_native_code: bool,

    // === Performance characteristics ===

    /// Processing characteristics (CPU vs I/O bound).
    pub processing: ProcessingHint,

    /// Latency requirements.
    pub latency: LatencyHint,

    /// Memory usage hint.
    pub memory: MemoryHint,
}
```

### Validation at Decision Time

The `determine_element_strategy` function becomes simpler and adds conflict warnings:

```rust
fn determine_element_strategy(hints: &ExecutionHints) -> ElementStrategy {
    // --- Validate and warn on conflicts ---
    if hints.affinity == Affinity::RealTime && !hints.rt_safe {
        tracing::warn!("Element requests RealTime affinity but rt_safe=false, using Async");
    }
    if hints.affinity == Affinity::RealTime && hints.processing == ProcessingHint::IoBound {
        tracing::warn!("Element requests RealTime affinity but processing=IoBound, using Async");
    }
    if hints.affinity == Affinity::Async && hints.latency == LatencyHint::UltraLow {
        tracing::warn!("Element requests Async affinity but latency=UltraLow");
    }

    // --- Decision rules (priority order) ---

    // 1. Isolation trumps everything
    if hints.trust_level == TrustLevel::Untrusted {
        return ElementStrategy::Isolated;
    }
    if hints.uses_native_code && !hints.crash_safe {
        return ElementStrategy::Isolated;
    }

    // 2. Explicit affinity (validated)
    if hints.affinity == Affinity::RealTime && hints.rt_safe {
        return ElementStrategy::RealTime;
    }
    if hints.affinity == Affinity::Async {
        return ElementStrategy::Async;
    }

    // 3. Auto-detect from characteristics
    if hints.rt_safe && matches!(hints.latency, LatencyHint::UltraLow | LatencyHint::Low) {
        return ElementStrategy::RealTime;
    }
    if hints.processing == ProcessingHint::IoBound {
        return ElementStrategy::Async;
    }

    // 4. Default
    ElementStrategy::Async
}
```

### Convenience Constructors (Profiles)

Instead of having element authors fill in 8 fields, provide profile constructors matching the 4 real-world patterns:

```rust
impl ExecutionHints {
    /// Default: trusted, not RT-safe, auto affinity, normal latency.
    pub fn default() -> Self { /* all defaults */ }

    /// RT-safe transform: CPU-bound, low latency, trusted.
    /// For elements like Gain, PassThrough, filters.
    pub fn rt_safe() -> Self {
        Self {
            rt_safe: true,
            processing: ProcessingHint::CpuBound,
            latency: LatencyHint::Low,
            ..Self::default()
        }
    }

    /// I/O-bound element: async affinity, not RT-safe.
    /// For device sources/sinks (ALSA, PipeWire, V4L2, network).
    pub fn io_bound() -> Self {
        Self {
            affinity: Affinity::Async,
            processing: ProcessingHint::IoBound,
            ..Self::default()
        }
    }

    /// Native/FFI element: may crash, should be isolated.
    /// For codecs wrapping C libraries.
    pub fn native() -> Self {
        Self {
            uses_native_code: true,
            crash_safe: false,
            ..Self::default()
        }
    }

    /// Untrusted input handler: must be isolated.
    pub fn untrusted() -> Self {
        Self {
            trust_level: TrustLevel::Untrusted,
            crash_safe: false,
            ..Self::default()
        }
    }

    /// Trusted, lightweight element (same as default).
    pub fn trusted() -> Self { Self::default() }

    /// CPU-intensive but not RT-safe.
    pub fn cpu_intensive() -> Self {
        Self {
            processing: ProcessingHint::CpuBound,
            ..Self::default()
        }
    }

    /// Low-latency element (auto RT detection).
    pub fn low_latency() -> Self {
        Self {
            latency: LatencyHint::Low,
            ..Self::default()
        }
    }
}
```

### Trait Changes

Every element trait drops `affinity()` and `is_rt_safe()`, keeping only `execution_hints()`:

```rust
// BEFORE (3 methods per trait, 11 traits = 33 default methods):
pub trait Element: Send {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>>;
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn is_rt_safe(&self) -> bool { false }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
    // ...
}

// AFTER (1 method per trait, 11 traits = 11 default methods):
pub trait Element: Send {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>>;
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
    // ...
}
```

This eliminates **~62 method definitions** from trait defaults and **~22 adapter delegation sites**.

### Element Migration

Each element that overrides these methods consolidates to a single override:

```rust
// BEFORE (Gain element, 3 overrides):
impl Element for Gain {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> { ... }
    fn is_rt_safe(&self) -> bool { true }
    fn affinity(&self) -> Affinity { Affinity::Auto }
    fn execution_hints(&self) -> ExecutionHints {
        ExecutionHints { processing: CpuBound, latency: Low, ..trusted() }
    }
}

// AFTER (1 override):
impl Element for Gain {
    fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> { ... }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::rt_safe() }
}
```

```rust
// BEFORE (ALSA source, 3 overrides):
impl AsyncSource for AlsaSrc {
    fn affinity(&self) -> Affinity { Affinity::Async }
    fn is_rt_safe(&self) -> bool { false }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::io_bound() }
}

// AFTER (1 override — or just use default since AsyncSource defaults to io_bound):
impl AsyncSource for AlsaSrc {
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::io_bound() }
}
```

```rust
// BEFORE (Tee, 2 overrides — we just added these):
impl Element for Tee {
    fn is_rt_safe(&self) -> bool { true }
    fn affinity(&self) -> Affinity { Affinity::Auto }
}

// AFTER (1 override):
impl Element for Tee {
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::rt_safe() }
}
```

### SyncElementAdapter Fix

Currently hardcodes `affinity=RealTime`, `is_rt_safe=true`, ignoring the inner element. After consolidation, it should respect the inner element's hints:

```rust
// BEFORE (hardcoded, dangerous):
impl<T: SyncElement> SendAsyncElementDyn for SyncElementAdapter<T> {
    fn affinity(&self) -> Affinity { Affinity::RealTime }      // HARDCODED
    fn is_rt_safe(&self) -> bool { true }                       // HARDCODED
    fn execution_hints(&self) -> ExecutionHints {                // HARDCODED
        ExecutionHints { processing: CpuBound, latency: Low, ..trusted() }
    }
}

// AFTER (SyncElementAdapter declares RT-safe by design):
impl<T: SyncElement> SendAsyncElementDyn for SyncElementAdapter<T> {
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::rt_safe() }
}
```

This is correct because `SyncElementAdapter` wraps elements explicitly placed in the RT path by the developer — its purpose is to signal "this element was intentionally chosen for RT".

### Accessor Methods on ExecutionHints

For call sites that just need a quick boolean check, add accessors:

```rust
impl ExecutionHints {
    /// Whether this element can run in an RT context.
    pub fn is_rt_safe(&self) -> bool { self.rt_safe }

    /// Whether this element should be isolated.
    pub fn should_isolate(&self) -> bool {
        self.trust_level == TrustLevel::Untrusted
            || (self.uses_native_code && !self.crash_safe)
    }
}
```

This lets existing code like `node.is_rt_safe()` change to `node.execution_hints().is_rt_safe()` at call sites like `rt_scheduler.rs`.

## Impact Analysis

### Lines Changed (Estimated)

| File | Change | Estimate |
|------|--------|----------|
| `element/traits.rs` | Remove ~62 method defs, update adapters | -120 lines |
| `element/pipeline_element.rs` | Remove ~16 method defs, update adapters | -40 lines |
| `pipeline/unified_executor.rs` | Simplify `determine_element_strategy` | -10 lines |
| `pipeline/rt_scheduler.rs` | Update 3 call sites | ~5 lines |
| `pipeline/graph.rs` | Remove 2 delegation methods | -10 lines |
| `elements/**` (53 override sites) | Consolidate to single method | -35 lines |
| `tests/rt_integration.rs` | Update test assertions | ~10 lines |
| **Total** | | **~-200 net lines removed** |

### Migration of Element Overrides

All 53 current override sites across `src/elements/` mapped to new API:

| Element(s) | Current Overrides | New ExecutionHints |
|------------|------------------|-------------------|
| Gain, PassThrough, Identity, Valve, Tee | `is_rt_safe=true`, `affinity=Auto` | `ExecutionHints::rt_safe()` |
| Filter (opt-in), SampleFilter, MetadataFilter | `is_rt_safe=true/flag`, `affinity=Auto` | `ExecutionHints::rt_safe()` or builder |
| SequenceNumber, MetadataInject | `is_rt_safe=true`, `affinity=Auto` | `ExecutionHints::rt_safe()` |
| AlsaSrc, AlsaSink | `affinity=Async`, `is_rt_safe=false`, `io_bound()` | `ExecutionHints::io_bound()` |
| PipeWireSrc, PipeWireSink | `affinity=Async`, `is_rt_safe=false`, `io_bound()` | `ExecutionHints::io_bound()` |
| V4l2Src | `affinity=Async`, `is_rt_safe=false`, `io_bound()` | `ExecutionHints::io_bound()` |
| LibCameraSrc | `affinity=Async`, `is_rt_safe=false`, `io_bound()` | `ExecutionHints::io_bound()` |
| ScreenCaptureSrc | `affinity=Async`, `is_rt_safe=false`, `io_bound()` | `ExecutionHints::io_bound()` |
| H264Encoder/Decoder | `execution_hints=native()` only | `ExecutionHints::native()` (unchanged) |
| Rav1eEncoder, Dav1dDecoder | `execution_hints=native()` only | `ExecutionHints::native()` (unchanged) |
| AudioEncoderElement, AudioDecoderElement | `execution_hints=native()` only | `ExecutionHints::native()` (unchanged) |
| HwEncoder, HwDecoder | `execution_hints=native()` only | `ExecutionHints::native()` (unchanged) |
| ImageEncoder/Decoder | `execution_hints=cpu_intensive()` | `ExecutionHints::cpu_intensive()` (unchanged) |
| HlsSink, DashSink | `execution_hints=io_bound()` only | `ExecutionHints::io_bound()` (unchanged) |
| Filter\<F\> | `is_rt_safe=self.rt_safe`, `affinity=Auto` | Builder with `.with_rt_safe(flag)` |

### Breaking Changes

This is a **breaking API change** for any element that overrides `affinity()` or `is_rt_safe()`. Since those methods are removed from the traits, any override will become a compile error, guiding developers to the new API.

Elements that only override `execution_hints()` (all codec elements) require **no changes**.

## Implementation Steps

### Step 1: Update `ExecutionHints` struct
- Add `rt_safe: bool` and `affinity: Affinity` fields
- Add `rt_safe()` profile constructor
- Add `is_rt_safe()` and `should_isolate()` accessor methods
- Update existing constructors (`io_bound()` sets `affinity: Async`, etc.)
- Add builder methods: `with_rt_safe(bool)`, `with_affinity(Affinity)`

### Step 2: Update `determine_element_strategy`
- Change signature from `(hints, affinity, rt_safe)` to `(hints)` — read all from struct
- Add conflict warnings
- Simplify rule logic

### Step 3: Update element traits
- Remove `affinity()` and `is_rt_safe()` from all 11 trait definitions
- Update default `execution_hints()` for async traits to include `affinity: Async`

### Step 4: Update adapters
- Remove `affinity()` and `is_rt_safe()` delegation from all 11 adapter types
- Fix `SyncElementAdapter` to use `ExecutionHints::rt_safe()`

### Step 5: Update `SendAsyncElementDyn` trait
- Remove `affinity()` and `is_rt_safe()` from the trait definition
- Update all implementations

### Step 6: Update `pipeline/graph.rs`
- Remove `affinity()` and `is_rt_safe()` from `NodeWrapper`
- Update call sites to use `execution_hints().affinity` and `execution_hints().is_rt_safe()`

### Step 7: Update `rt_scheduler.rs`
- Change partition logic to read from `execution_hints()`
- Update `effective_affinity()` to use hints struct

### Step 8: Update element implementations
- Migrate all 53 override sites in `src/elements/`
- Remove separate `affinity()` and `is_rt_safe()` overrides
- Consolidate into single `execution_hints()` override using profile constructors

### Step 9: Update `Filter<F>` opt-in RT
- Replace `rt_safe: bool` field + `.rt_safe()` builder with `hints: ExecutionHints` field
- Provide `.with_execution_hints(hints)` builder

### Step 10: Update tests
- Update `rt_integration.rs` assertions
- Verify all 1028 tests pass
- Run example 50 (hybrid pipeline)

### Step 11: Clean up Affinity enum
- `Affinity` stays as-is (it's still useful as an enum), but moves from being a trait method to being a field of `ExecutionHints`
- Remove `Affinity` from trait re-exports if no longer needed standalone
