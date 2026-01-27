# Plan 05: Element Trait Consolidation

**Priority:** Medium (Medium-term)  
**Effort:** Large (2-3 weeks)  
**Dependencies:** Plans 01-04 should be done first (breaking change)  
**Addresses:** Architecture Issues 3.1, 3.2, 3.5 (Trait Proliferation, Sync/Async Confusion, Error Handling)

---

## Problem Statement

The current element system has too many overlapping traits:

### Current Traits (10+)
- `Source`, `AsyncSource`
- `Sink`, `AsyncSink`
- `Element`, `Transform`, `AsyncTransform`
- `Demuxer`, `Muxer`

### Current Adapters (8+)
- `SourceAdapter`, `AsyncSourceAdapter`
- `SinkAdapter`, `AsyncSinkAdapter`
- `ElementAdapter`, `TransformAdapter`, `AsyncTransformAdapter`
- `DemuxerAdapter`, `MuxerAdapter`

### Problems
1. **Confusion:** When to use `Element` vs `Transform`?
2. **Boilerplate:** Need to wrap everything in adapters
3. **Sync/Async split:** Must choose at compile time
4. **Inconsistent returns:** `Option<Buffer>` vs `Output` vs `ProduceResult`

---

## Proposed Solution

> **Design Decision:** Per [Decision 5 in 00_DESIGN_DECISIONS.md](./00_DESIGN_DECISIONS.md), we adopt **unified async `PipelineElement` with blanket implementations** for sync elements. This eliminates the adapter boilerplate entirely.

Consolidate into a unified element model with:
1. **Single async trait** as the core abstraction
2. **Blanket implementations** for sync elements
3. **Unified output type**
4. **Automatic adapter generation**

---

## Design

### Unified Output Type

```rust
/// Result of element processing
pub enum ProcessOutput {
    /// No output (filtered, buffering, etc.)
    None,
    /// One output buffer
    One(Buffer),
    /// Multiple output buffers
    Many(Vec<Buffer>),
    /// End of stream reached
    Eos,
    /// Would block (for async polling)
    Pending,
}

impl ProcessOutput {
    pub fn one(buffer: Buffer) -> Self { Self::One(buffer) }
    pub fn none() -> Self { Self::None }
    pub fn eos() -> Self { Self::Eos }
    pub fn pending() -> Self { Self::Pending }
    
    pub fn is_eos(&self) -> bool { matches!(self, Self::Eos) }
    pub fn is_pending(&self) -> bool { matches!(self, Self::Pending) }
}

impl From<Buffer> for ProcessOutput {
    fn from(b: Buffer) -> Self { Self::One(b) }
}

impl From<Option<Buffer>> for ProcessOutput {
    fn from(opt: Option<Buffer>) -> Self {
        match opt {
            Some(b) => Self::One(b),
            None => Self::None,
        }
    }
}

impl From<Vec<Buffer>> for ProcessOutput {
    fn from(v: Vec<Buffer>) -> Self {
        match v.len() {
            0 => Self::None,
            1 => Self::One(v.into_iter().next().unwrap()),
            _ => Self::Many(v),
        }
    }
}
```

### Core Element Trait (Async)

```rust
/// Core trait for all pipeline elements.
/// 
/// This is an async trait - sync elements get blanket implementations.
#[async_trait]
pub trait PipelineElement: Send + Sync {
    /// Element type determines pad configuration and execution behavior.
    fn element_type(&self) -> ElementType;
    
    /// Human-readable name for debugging.
    fn name(&self) -> &str {
        std::any::type_name::<Self>()
    }
    
    /// Process input and produce output.
    /// 
    /// - Sources: `input` is `None`, produce data
    /// - Sinks: consume `input`, return `ProcessOutput::None`
    /// - Transforms: transform `input` to output
    async fn process(&mut self, input: Option<Buffer>) -> Result<ProcessOutput>;
    
    /// Flush any buffered data (called at EOS).
    async fn flush(&mut self) -> Result<ProcessOutput> {
        Ok(ProcessOutput::None)
    }
    
    /// Input capabilities (what formats accepted).
    fn input_caps(&self) -> Caps { Caps::any() }
    
    /// Output capabilities (what formats produced).
    fn output_caps(&self) -> Caps { Caps::any() }
    
    /// Scheduling hints.
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ElementType {
    Source,     // 0 inputs, 1 output
    Sink,       // 1 input, 0 outputs
    Transform,  // 1 input, 1 output
    Demuxer,    // 1 input, N outputs
    Muxer,      // N inputs, 1 output
}
```

### Simplified Sync Traits

For convenience, keep simple sync traits with blanket async implementations:

```rust
/// Simple source that produces buffers.
pub trait Source: Send {
    fn produce(&mut self) -> Result<ProcessOutput>;
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    fn output_caps(&self) -> Caps { Caps::any() }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
}

/// Blanket implementation: Source -> PipelineElement
#[async_trait]
impl<T: Source + Sync + 'static> PipelineElement for T {
    fn element_type(&self) -> ElementType { ElementType::Source }
    fn name(&self) -> &str { Source::name(self) }
    fn output_caps(&self) -> Caps { Source::output_caps(self) }
    fn execution_hints(&self) -> ExecutionHints { Source::execution_hints(self) }
    
    async fn process(&mut self, _input: Option<Buffer>) -> Result<ProcessOutput> {
        // For CPU-bound sources, use spawn_blocking
        if self.execution_hints().processing == ProcessingHint::CpuBound {
            tokio::task::block_in_place(|| self.produce())
        } else {
            self.produce()
        }
    }
}
```

```rust
/// Simple sink that consumes buffers.
pub trait Sink: Send {
    fn consume(&mut self, buffer: Buffer) -> Result<()>;
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    fn input_caps(&self) -> Caps { Caps::any() }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
}

#[async_trait]
impl<T: Sink + Sync + 'static> PipelineElement for T {
    fn element_type(&self) -> ElementType { ElementType::Sink }
    
    async fn process(&mut self, input: Option<Buffer>) -> Result<ProcessOutput> {
        match input {
            Some(buffer) => {
                self.consume(buffer)?;
                Ok(ProcessOutput::None)
            }
            None => Ok(ProcessOutput::Eos),
        }
    }
}
```

```rust
/// Simple transform (1 input -> 0/1/N outputs).
pub trait Transform: Send {
    fn transform(&mut self, buffer: Buffer) -> Result<ProcessOutput>;
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    fn input_caps(&self) -> Caps { Caps::any() }
    fn output_caps(&self) -> Caps { Caps::any() }
    fn execution_hints(&self) -> ExecutionHints { ExecutionHints::default() }
    fn flush(&mut self) -> Result<ProcessOutput> { Ok(ProcessOutput::None) }
}

#[async_trait]
impl<T: Transform + Sync + 'static> PipelineElement for T {
    fn element_type(&self) -> ElementType { ElementType::Transform }
    
    async fn process(&mut self, input: Option<Buffer>) -> Result<ProcessOutput> {
        match input {
            Some(buffer) => self.transform(buffer),
            None => Ok(ProcessOutput::Eos),
        }
    }
    
    async fn flush(&mut self) -> Result<ProcessOutput> {
        Transform::flush(self)
    }
}
```

### Async Variants (When Needed)

For elements that genuinely need async (network I/O):

```rust
/// Async source for I/O-bound operations.
#[async_trait]
pub trait AsyncSource: Send + Sync {
    async fn produce(&mut self) -> Result<ProcessOutput>;
    fn name(&self) -> &str { std::any::type_name::<Self>() }
    fn output_caps(&self) -> Caps { Caps::any() }
    fn execution_hints(&self) -> ExecutionHints { 
        ExecutionHints::io_bound() 
    }
}

#[async_trait]
impl<T: AsyncSource + 'static> PipelineElement for T {
    fn element_type(&self) -> ElementType { ElementType::Source }
    
    async fn process(&mut self, _input: Option<Buffer>) -> Result<ProcessOutput> {
        AsyncSource::produce(self).await
    }
}
```

### No More Explicit Adapters!

With blanket implementations, adapters are gone:

```rust
// OLD (verbose):
let src = pipeline.add_node(
    "src",
    DynAsyncElement::new_box(SourceAdapter::new(my_source)),
);

// NEW (automatic):
let src = pipeline.add_node("src", Box::new(my_source));

// The blanket impl makes my_source: PipelineElement automatically
```

### Pipeline Uses Trait Objects

```rust
impl Pipeline {
    pub fn add_node(
        &mut self,
        name: impl Into<String>,
        element: Box<dyn PipelineElement>,
    ) -> NodeId {
        // ...
    }
}
```

---

## Migration Path

### Phase 1: Add New Traits (Non-Breaking)

1. Add `PipelineElement` trait
2. Add `ProcessOutput` enum
3. Add blanket implementations
4. Keep old traits working

### Phase 2: Update Pipeline to Accept Both

```rust
impl Pipeline {
    // New method
    pub fn add_element(&mut self, name: &str, element: Box<dyn PipelineElement>) -> NodeId;
    
    // Old method (deprecated)
    #[deprecated(note = "Use add_element instead")]
    pub fn add_node(&mut self, name: &str, element: Box<DynAsyncElement>) -> NodeId;
}
```

### Phase 3: Migrate Elements

Update built-in elements to use new traits:
- `TestSource` -> implements `Source`
- `FileSink` -> implements `Sink`
- `VideoScale` -> implements `Transform`

### Phase 4: Remove Old System

After all elements migrated:
1. Remove old traits (`Element`, old `Source`, etc.)
2. Remove adapters
3. Remove `DynAsyncElement`

---

## Implementation Steps

### Step 1: Define ProcessOutput

**File:** `src/element/output.rs`

```rust
pub enum ProcessOutput { None, One(Buffer), Many(Vec<Buffer>), Eos, Pending }
```

### Step 2: Define PipelineElement Trait

**File:** `src/element/pipeline_element.rs`

```rust
#[async_trait]
pub trait PipelineElement: Send + Sync { ... }
```

### Step 3: Add Simplified Sync Traits

**File:** `src/element/simple_traits.rs`

```rust
pub trait Source: Send { ... }
pub trait Sink: Send { ... }
pub trait Transform: Send { ... }
```

### Step 4: Add Blanket Implementations

**File:** `src/element/blanket_impls.rs`

```rust
impl<T: Source + Sync + 'static> PipelineElement for T { ... }
impl<T: Sink + Sync + 'static> PipelineElement for T { ... }
impl<T: Transform + Sync + 'static> PipelineElement for T { ... }
```

### Step 5: Update Pipeline

**File:** `src/pipeline/graph.rs`

```rust
pub fn add_element(&mut self, name: &str, element: Box<dyn PipelineElement>) -> NodeId
```

### Step 6: Update Executor

**File:** `src/pipeline/unified_executor.rs`

Use `PipelineElement::process()` directly.

### Step 7: Migrate Built-in Elements

Update all elements in `src/elements/` to use new traits.

### Step 8: Update Examples

Remove adapter boilerplate from all examples.

### Step 9: Remove Old System

Delete deprecated traits and adapters.

---

## Trait Comparison

### Before (Current)

```
                    ┌─────────────────────────────────────────┐
                    │            AsyncElementDyn              │
                    │  (dynosaur-generated, trait object)     │
                    └─────────────────────────────────────────┘
                                        ▲
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
            ┌───────┴───────┐   ┌───────┴───────┐   ┌───────┴───────┐
            │ SourceAdapter │   │  SinkAdapter  │   │ElementAdapter │
            │ AsyncSource   │   │  AsyncSink    │   │TransformAdapt │
            │   Adapter     │   │   Adapter     │   │AsyncTransform │
            └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                    │                   │                   │
            ┌───────┴───────┐   ┌───────┴───────┐   ┌───────┴───────┐
            │    Source     │   │     Sink      │   │   Element     │
            │  AsyncSource  │   │   AsyncSink   │   │   Transform   │
            └───────────────┘   └───────────────┘   │ AsyncTransform│
                                                    └───────────────┘
```

### After (Proposed)

```
                    ┌─────────────────────────────────────────┐
                    │           PipelineElement               │
                    │      (single async trait object)        │
                    └─────────────────────────────────────────┘
                                        ▲
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
            ┌───────┴───────┐   ┌───────┴───────┐   ┌───────┴───────┐
            │    Source     │   │     Sink      │   │   Transform   │
            │ (blanket impl)│   │ (blanket impl)│   │ (blanket impl)│
            └───────────────┘   └───────────────┘   └───────────────┘
                    │                   │                   │
                    │                   │                   │
            ┌───────┴───────┐   ┌───────┴───────┐   ┌───────┴───────┐
            │  AsyncSource  │   │   AsyncSink   │   │AsyncTransform │
            │ (blanket impl)│   │ (blanket impl)│   │ (blanket impl)│
            └───────────────┘   └───────────────┘   └───────────────┘
```

---

## Code Example: Before vs After

### Before (Current)

```rust
use parallax::element::{DynAsyncElement, SourceAdapter, SinkAdapter, ElementAdapter};

struct MySource;
impl Source for MySource {
    fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> { ... }
}

struct MySink;
impl Sink for MySink {
    fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> { ... }
}

let mut pipeline = Pipeline::new();

let src = pipeline.add_node(
    "src",
    DynAsyncElement::new_box(SourceAdapter::new(MySource)),
);
let sink = pipeline.add_node(
    "sink", 
    DynAsyncElement::new_box(SinkAdapter::new(MySink)),
);

pipeline.link(src, sink)?;
```

### After (Proposed)

```rust
use parallax::element::{Source, Sink, ProcessOutput};

struct MySource;
impl Source for MySource {
    fn produce(&mut self) -> Result<ProcessOutput> { ... }
}

struct MySink;
impl Sink for MySink {
    fn consume(&mut self, buffer: Buffer) -> Result<()> { ... }
}

let mut pipeline = Pipeline::new();

let src = pipeline.add_element("src", Box::new(MySource));
let sink = pipeline.add_element("sink", Box::new(MySink));

pipeline.link(src, sink)?;
```

**Reduction:** 6 lines → 2 lines for element creation.

---

## Validation Criteria

- [ ] `ProcessOutput` enum defined
- [ ] `PipelineElement` trait defined
- [ ] Blanket implementations work
- [ ] No adapters needed for basic elements
- [ ] All built-in elements migrated
- [ ] All examples updated
- [ ] Old traits deprecated then removed
- [ ] All existing tests pass
- [ ] Documentation updated

---

## Risks and Mitigations

### Risk: Breaking Change

**Mitigation:** 
- Phase the migration
- Deprecate before removing
- Provide migration guide

### Risk: Performance Impact

**Mitigation:**
- Benchmark before/after
- `block_in_place` for CPU-bound sync elements
- Profile async overhead

### Risk: Complexity of Blanket Impls

**Mitigation:**
- Thorough testing
- Clear documentation
- Explicit bounds on trait objects

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `src/element/output.rs` | New: ProcessOutput |
| `src/element/pipeline_element.rs` | New: PipelineElement trait |
| `src/element/simple_traits.rs` | New: simplified Source/Sink/Transform |
| `src/element/blanket_impls.rs` | New: blanket implementations |
| `src/element/mod.rs` | Re-exports, deprecations |
| `src/pipeline/graph.rs` | add_element() method |
| `src/pipeline/unified_executor.rs` | Use PipelineElement |
| `src/elements/**/*.rs` | Migrate all elements |
| `examples/*.rs` | Remove adapter boilerplate |
