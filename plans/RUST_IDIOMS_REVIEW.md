# Rust Idioms Review & Crate Recommendations

This document reviews the code in the plans for Rust idioms and identifies crates that could help implementation.

---

## Summary of Issues

| Severity | Count | Description |
|----------|-------|-------------|
| **Critical** | 2 | Would cause compile errors |
| **Major** | 4 | Anti-patterns that should be fixed |
| **Minor** | 5 | Style improvements |

---

## Critical Issues

### 1. Blanket Implementation Conflicts (Plan 05)

**Location:** `plans/05_ELEMENT_TRAIT_CONSOLIDATION.md`

**Problem:** The proposed blanket implementations will conflict:

```rust
// These cannot coexist - Rust's orphan rules prevent it
impl<T: Source + Sync + 'static> PipelineElement for T { ... }
impl<T: Sink + Sync + 'static> PipelineElement for T { ... }
impl<T: Transform + Sync + 'static> PipelineElement for T { ... }
```

If a type implements multiple traits (e.g., both `Source` and `Transform`), the compiler can't choose which impl to use.

**Fix:** Use wrapper types instead of blanket impls:

```rust
// Wrapper approach (idiomatic)
pub struct SourceElement<S: Source>(pub S);
pub struct SinkElement<S: Sink>(pub S);
pub struct TransformElement<T: Transform>(pub T);

impl<S: Source + Sync + 'static> PipelineElement for SourceElement<S> {
    async fn process(&mut self, _input: Option<Buffer>) -> Result<ProcessOutput> {
        self.0.produce()
    }
}

// Usage becomes:
pipeline.add_element("src", Box::new(SourceElement(MySource::new())));

// Or with a convenience trait:
trait IntoPipelineElement {
    fn into_element(self) -> Box<dyn PipelineElement>;
}

impl<S: Source + Sync + 'static> IntoPipelineElement for S {
    fn into_element(self) -> Box<dyn PipelineElement> {
        Box::new(SourceElement(self))
    }
}

// Usage:
pipeline.add_element("src", MySource::new().into_element());
```

### 2. ClockTime Sentinel Value (Plan 00)

**Location:** `plans/00_DESIGN_DECISIONS.md` - Decision 8

**Problem:** Using `i64::MIN` as a sentinel for "invalid" is a C-ism, not idiomatic Rust:

```rust
// Anti-pattern
pub const NONE: Self = Self(i64::MIN);  // Magic sentinel
pub fn is_valid(&self) -> bool { self.0 != i64::MIN }
```

**Fix:** Use `Option<ClockTime>` where timestamps can be unset:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClockTime(i64);  // Always valid

impl ClockTime {
    pub const ZERO: Self = Self(0);
    
    pub fn from_nanos(ns: i64) -> Self { Self(ns) }
    pub fn nanos(&self) -> i64 { self.0 }
}

// In Metadata:
pub struct Metadata {
    pub timestamp: Option<ClockTime>,  // None = unset
    pub duration: Option<ClockTime>,
    // ...
}
```

This is more explicit and leverages Rust's type system.

---

## Major Issues

### 3. Busy-Wait in Buffer Pool (Plan 04)

**Location:** `plans/04_PIPELINE_BUFFER_POOL.md`

**Problem:** The acquire loop uses busy-waiting:

```rust
fn acquire(&self) -> Result<PooledBuffer> {
    loop {
        if let Some(buffer) = self.try_acquire() {
            return Ok(buffer);
        }
        std::thread::yield_now();  // Busy wait!
    }
}
```

**Fix:** Use proper synchronization primitives:

```rust
use parking_lot::{Mutex, Condvar};
// Or for async:
use tokio::sync::Semaphore;

pub struct FixedSizePool {
    inner: Arc<PoolInner>,
    // Semaphore tracks available slots
    available: Arc<Semaphore>,
}

impl FixedSizePool {
    // Sync blocking acquire
    pub fn acquire_blocking(&self) -> Result<PooledBuffer> {
        // Blocks until permit available
        let _permit = self.available.blocking_acquire()?;
        self.do_acquire()
    }
    
    // Async acquire
    pub async fn acquire(&self) -> Result<PooledBuffer> {
        let _permit = self.available.acquire().await?;
        self.do_acquire()
    }
}
```

### 4. Builder Pattern Error Handling (Plan 07)

**Location:** `plans/07_PIPELINE_BUILDER_DSL.md`

**Problem:** Using `.expect()` in builder methods loses errors:

```rust
pub fn then_named<T>(...) -> Self {
    // ...
    self.pipeline.link(prev, node_id).expect("link failed");  // Panics!
    // ...
}
```

**Fix:** Use fallible builder pattern:

```rust
// Option A: Return Result from build()
pub struct PipelineBuilder {
    pipeline: Pipeline,
    errors: Vec<Error>,  // Collect errors
}

impl PipelineBuilder {
    pub fn then<T: Transform>(mut self, transform: T) -> Self {
        let node_id = self.pipeline.add_element(...);
        if let Some(prev) = self.current_node {
            if let Err(e) = self.pipeline.link(prev, node_id) {
                self.errors.push(e);
            }
        }
        self
    }
    
    pub fn build(self) -> Result<Pipeline> {
        if !self.errors.is_empty() {
            return Err(Error::BuilderErrors(self.errors));
        }
        Ok(self.pipeline)
    }
}

// Option B: Make each method return Result (typed-builder style)
pub fn then<T: Transform>(self, transform: T) -> Result<Self> {
    // ...
    self.pipeline.link(prev, node_id)?;
    Ok(self)
}
```

### 5. Missing `#[must_use]` Attributes

**Location:** Multiple plans

**Problem:** Functions returning `Result` or important values should have `#[must_use]`:

```rust
// Current
pub fn try_acquire(&self) -> Option<PooledBuffer> { ... }

// Should be
#[must_use]
pub fn try_acquire(&self) -> Option<PooledBuffer> { ... }
```

**Fix:** Add `#[must_use]` to:
- `try_acquire()`, `acquire()` in Plan 04
- `intersect()`, `can_intersect()` in Plan 06
- All `Result`-returning functions

### 6. Clone for Metadata with `Box<dyn Any>` (Plan 01)

**Location:** `plans/01_CUSTOM_METADATA_API.md`

**Problem:** The plan acknowledges but doesn't solve the Clone issue:

```rust
impl Clone for Metadata {
    fn clone(&self) -> Self {
        // custom: HashMap::new(),  // Empty by default - DATA LOSS!
    }
}
```

**Fix:** Use a trait that supports cloning:

```rust
// Option A: Use `dyn_clone` crate
use dyn_clone::DynClone;

pub trait MetaValue: Any + Send + Sync + DynClone {
    fn as_any(&self) -> &dyn Any;
}
dyn_clone::clone_trait_object!(MetaValue);

pub struct Metadata {
    custom: HashMap<&'static str, Box<dyn MetaValue>>,
}

impl Clone for Metadata {
    fn clone(&self) -> Self {
        Self {
            custom: self.custom.iter()
                .map(|(k, v)| (*k, dyn_clone::clone_box(&**v)))
                .collect(),
            // ...
        }
    }
}

// Option B: Store Arc instead of Box (cheaper clone, shared data)
pub struct Metadata {
    custom: HashMap<&'static str, Arc<dyn Any + Send + Sync>>,
}
```

---

## Minor Issues

### 7. Use `NonZeroU32` for Dimensions (Plan 06)

```rust
// Current
pub width: DimensionConstraint,  // Could be 0

// Better
use std::num::NonZeroU32;

#[derive(Debug, Clone, PartialEq)]
pub enum DimensionConstraint {
    Any,
    Exact(NonZeroU32),  // Can't be 0
    Range { min: NonZeroU32, max: NonZeroU32 },
    // ...
}
```

### 8. Use `SmallVec` for Small Collections (Plan 06, 08)

```rust
// Current
pub formats: Vec<PixelFormat>,  // Usually 1-3 items

// Better
use smallvec::SmallVec;
pub formats: SmallVec<[PixelFormat; 4]>,  // Inline for ≤4 items
```

### 9. Derive More Traits

Many structs should derive additional traits:

```rust
// Current
#[derive(Debug, Clone)]
pub struct SegmentEvent { ... }

// Better - enables use in HashMaps, comparisons
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SegmentEvent { ... }
```

### 10. Use `Cow<'static, str>` for Names (Plan 05, 07)

```rust
// Current - requires allocation for string literals
pub fn name(&self) -> &str { "my_element" }

// Better - zero-cost for static strings
use std::borrow::Cow;
pub fn name(&self) -> Cow<'static, str> { 
    Cow::Borrowed("my_element") 
}
```

### 11. Iterator Implementation for ProcessOutput (Plan 05)

```rust
impl ProcessOutput {
    pub fn into_iter(self) -> impl Iterator<Item = Buffer> {
        match self {
            Self::None | Self::Eos | Self::Pending => 
                itertools::Either::Left(std::iter::empty()),
            Self::One(b) => 
                itertools::Either::Left(std::iter::once(b)),
            Self::Many(v) => 
                itertools::Either::Right(v.into_iter()),
        }
    }
}
```

---

## Recommended Crates

### High Priority (Should Use)

| Crate | Version | Use Case | Plan |
|-------|---------|----------|------|
| [`smallvec`](https://crates.io/crates/smallvec) | 1.x | Inline small vectors | 01, 06, 08 |
| [`bitflags`](https://crates.io/crates/bitflags) | 2.x | Type-safe bitflags (already planned) | 08 |
| [`dyn-clone`](https://crates.io/crates/dyn-clone) | 1.x | Clone for trait objects | 01 |
| [`thiserror`](https://crates.io/crates/thiserror) | 1.x | Error types (already using) | All |

> **Note:** `parking_lot` is NOT recommended. Modern `std::sync::Mutex` (since Rust 1.62) is highly optimized and performs comparably. Stick with standard library primitives.

### Medium Priority (Consider Using)

| Crate | Version | Use Case | Plan |
|-------|---------|----------|------|
| [`petgraph`](https://crates.io/crates/petgraph) | 0.6 | Graph algorithms for converter path finding | 06 |
| [`typed-builder`](https://crates.io/crates/typed-builder) | 0.18 | Compile-time checked builders | 07 |
| [`derive_more`](https://crates.io/crates/derive_more) | 1.x | Derive From, Into, Display | All |
| [`strum`](https://crates.io/crates/strum) | 0.26 | Enum utilities (iter, strings) | 06, 08 |
| [`enum-as-inner`](https://crates.io/crates/enum-as-inner) | 0.6 | Generate `as_*()` methods for enums | 05, 08 |

### Lower Priority (Nice to Have)

| Crate | Version | Use Case | Plan |
|-------|---------|----------|------|
| [`anymap`](https://crates.io/crates/anymap) | 1.0.0-beta | Type-safe `Any` map | 01 |
| [`crossbeam`](https://crates.io/crates/crossbeam) | 0.8 | Lock-free data structures | 04 |
| [`num-rational`](https://crates.io/crates/num-rational) | 0.4 | Exact framerates (30000/1001) | 06 |
| [`compact_str`](https://crates.io/crates/compact_str) | 0.8 | Inline small strings | 08 |

---

## Crate Details

### `std::sync` for Buffer Pool

Use standard library primitives (no external crate needed):

```rust
use std::sync::{Mutex, Condvar};

pub struct PoolInner {
    state: Mutex<PoolState>,
    available: Condvar,
}

impl PoolInner {
    pub fn acquire(&self) -> PooledBuffer {
        let mut state = self.state.lock().unwrap();
        while state.available_count == 0 {
            state = self.available.wait(state).unwrap();
        }
        state.take_slot()
    }
    
    pub fn release(&self, slot: usize) {
        let mut state = self.state.lock().unwrap();
        state.return_slot(slot);
        self.available.notify_one();
    }
}
```

### `petgraph` for Caps Negotiation

Use Dijkstra's algorithm for finding conversion paths:

```rust
use petgraph::algo::dijkstra;
use petgraph::graph::{DiGraph, NodeIndex};

pub struct ConverterRegistry {
    graph: DiGraph<MediaCaps, u32>,  // Weight = conversion cost
    caps_to_node: HashMap<MediaCaps, NodeIndex>,
}

impl ConverterRegistry {
    pub fn find_path(&self, from: &MediaCaps, to: &MediaCaps) -> Option<Vec<Converter>> {
        let start = *self.caps_to_node.get(from)?;
        let end = *self.caps_to_node.get(to)?;
        
        let costs = dijkstra(&self.graph, start, Some(end), |e| *e.weight());
        
        // Reconstruct path from costs...
    }
}
```

### `typed-builder` for Pipeline Builder

Compile-time enforcement of required fields:

```rust
use typed_builder::TypedBuilder;

#[derive(TypedBuilder)]
pub struct PipelineConfig {
    #[builder(setter(into))]
    name: String,
    
    #[builder(default = 10)]
    buffer_count: usize,
    
    #[builder(default, setter(strip_option))]
    pool: Option<Arc<dyn BufferPool>>,
}

// Usage - compile error if name not set
let config = PipelineConfig::builder()
    .name("my_pipeline")
    .buffer_count(20)
    .build();
```

### `anymap` for Type-Safe Metadata

Simpler than manual `HashMap<&str, Box<dyn Any>>`:

```rust
use anymap::AnyMap;

pub struct Metadata {
    pub sequence: u64,
    pub timestamp: Option<ClockTime>,
    custom: AnyMap,  // Type-safe storage
}

impl Metadata {
    pub fn set<T: Any + Send + Sync>(&mut self, value: T) {
        self.custom.insert(value);
    }
    
    pub fn get<T: Any + Send + Sync>(&self) -> Option<&T> {
        self.custom.get::<T>()
    }
}

// Usage - type is the key!
meta.set(KlvPacket { ... });
let klv = meta.get::<KlvPacket>();
```

**Note:** This changes the API to be type-keyed instead of string-keyed. If you need string keys for dynamic lookup, stick with `HashMap`.

### `num-rational` for Framerates

Exact representation without floating-point errors:

```rust
use num_rational::Ratio;

pub type Framerate = Ratio<u32>;

// 29.97 fps exactly
let ntsc = Framerate::new(30000, 1001);

// 30 fps
let fps30 = Framerate::from_integer(30);

// Comparisons work correctly
assert!(fps30 > ntsc);
```

---

## Action Items

1. **Immediate (Before Implementation):**
   - [ ] Fix blanket impl conflict in Plan 05 (use wrapper types)
   - [ ] Change `ClockTime::NONE` to use `Option<ClockTime>`
   - [ ] Fix buffer pool busy-wait with `std::sync::Condvar`

2. **During Implementation:**
   - [ ] Add `#[must_use]` to appropriate functions
   - [ ] Use `dyn-clone` for metadata cloning
   - [ ] Use `smallvec` for small collections

3. **Nice to Have:**
   - [ ] Consider `petgraph` for caps negotiation
   - [ ] Consider `typed-builder` for Pipeline builder
   - [ ] Add more derive macros (PartialEq, Eq, Hash)

---

## Updated Cargo.toml Dependencies

```toml
[dependencies]
# Existing
thiserror = "1"
tokio = { version = "1", features = ["full"] }
rkyv = { version = "0.7", features = ["validation"] }

# Add these
smallvec = { version = "1", features = ["union", "const_generics"] }
bitflags = "2"
dyn-clone = "1"

# Optional but recommended
petgraph = "0.6"           # For caps negotiation
derive_more = "1"          # Convenient derives
strum = { version = "0.26", features = ["derive"] }
```

> **Note:** Use `std::sync::{Mutex, Condvar}` - no need for `parking_lot`.
