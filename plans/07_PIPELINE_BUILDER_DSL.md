# Plan 07: Pipeline Builder DSL

**Priority:** Medium  
**Effort:** Small (2-3 days)  
**Dependencies:** Plan 05 (Element Trait Consolidation) helps but not required  
**Addresses:** Missing Feature 2.1 (Pipeline builder DSL)

---

## Problem Statement

Currently, building pipelines requires verbose code:

```rust
let mut pipeline = Pipeline::new();

let src = pipeline.add_node(
    "src",
    DynAsyncElement::new_box(SourceAdapter::new(VideoTestSrc::new())),
);
let scale = pipeline.add_node(
    "scale",
    DynAsyncElement::new_box(ElementAdapter::new(VideoScale::new(1920, 1080, 1280, 720))),
);
let sink = pipeline.add_node(
    "sink",
    DynAsyncElement::new_box(SinkAdapter::new(FileSink::new("out.yuv"))),
);

pipeline.link(src, scale)?;
pipeline.link(scale, sink)?;
```

This is error-prone and doesn't leverage Rust's type system.

---

## Proposed Solution

A fluent builder API that:
1. Chains element additions with `>>` or `.then()`
2. Type-checks connections at compile time (optional)
3. Provides ergonomic element construction
4. Supports branching (tee) and merging (funnel)

---

## Design

### Basic Builder Pattern

```rust
// Simple linear pipeline
let pipeline = PipelineBuilder::new()
    .source(VideoTestSrc::new())
    .then(VideoScale::new(1920, 1080, 1280, 720))
    .then(Rav1eEncoder::new(config)?)
    .sink(FileSink::new("out.av1"))
    .build()?;

pipeline.run().await?;
```

### Using `>>` Operator

```rust
// Operator-based syntax
let pipeline = (
    VideoTestSrc::new() 
    >> VideoScale::new(1920, 1080, 1280, 720)
    >> Rav1eEncoder::new(config)?
    >> FileSink::new("out.av1")
).build()?;
```

### Named Elements

```rust
let pipeline = PipelineBuilder::new()
    .source_named("video", VideoTestSrc::new())
    .then_named("scale", VideoScale::new(1920, 1080, 1280, 720))
    .then_named("encode", Rav1eEncoder::new(config)?)
    .sink_named("output", FileSink::new("out.av1"))
    .build()?;

// Access elements by name
let encoder = pipeline.get::<Rav1eEncoder>("encode")?;
```

### Branching (Tee)

```rust
// One source to multiple sinks
let pipeline = PipelineBuilder::new()
    .source(VideoTestSrc::new())
    .tee(|t| {
        t.branch(|b| b
            .then(VideoScale::new(1920, 1080, 1280, 720))
            .sink(FileSink::new("720p.yuv"))
        );
        t.branch(|b| b
            .then(VideoScale::new(1920, 1080, 640, 480))
            .sink(FileSink::new("480p.yuv"))
        );
    })
    .build()?;
```

### Merging (Funnel/Mux)

```rust
// Multiple sources to one sink
let pipeline = PipelineBuilder::new()
    .source_named("video", VideoTestSrc::new())
    .source_named("audio", AudioTestSrc::new())
    .mux::<TsMux>(|m| {
        m.input("video", "video_0");
        m.input("audio", "audio_0");
    })
    .sink(UdpSink::new("127.0.0.1:5000"))
    .build()?;
```

---

## API Design

### PipelineBuilder Struct

```rust
pub struct PipelineBuilder<State = Empty> {
    pipeline: Pipeline,
    current_node: Option<NodeId>,
    named_nodes: HashMap<String, NodeId>,
    _state: PhantomData<State>,
}

// State markers for type-safe building
pub struct Empty;
pub struct HasSource;
pub struct Complete;
```

### Builder Methods

```rust
impl PipelineBuilder<Empty> {
    pub fn new() -> Self {
        Self {
            pipeline: Pipeline::new(),
            current_node: None,
            named_nodes: HashMap::new(),
            _state: PhantomData,
        }
    }
    
    /// Add a source element.
    pub fn source<S: Source + 'static>(self, source: S) -> PipelineBuilder<HasSource> {
        self.source_named(Self::auto_name::<S>(), source)
    }
    
    /// Add a source element with a name.
    pub fn source_named<S: Source + 'static>(
        mut self, 
        name: impl Into<String>, 
        source: S
    ) -> PipelineBuilder<HasSource> {
        let name = name.into();
        let node_id = self.pipeline.add_element(&name, Box::new(source));
        self.named_nodes.insert(name, node_id);
        self.current_node = Some(node_id);
        
        PipelineBuilder {
            pipeline: self.pipeline,
            current_node: self.current_node,
            named_nodes: self.named_nodes,
            _state: PhantomData,
        }
    }
}

impl PipelineBuilder<HasSource> {
    /// Add a transform element.
    pub fn then<T: Transform + 'static>(self, transform: T) -> Self {
        self.then_named(Self::auto_name::<T>(), transform)
    }
    
    /// Add a transform element with a name.
    pub fn then_named<T: Transform + 'static>(
        mut self,
        name: impl Into<String>,
        transform: T,
    ) -> Self {
        let name = name.into();
        let node_id = self.pipeline.add_element(&name, Box::new(transform));
        self.named_nodes.insert(name, node_id);
        
        // Link to previous element
        if let Some(prev) = self.current_node {
            self.pipeline.link(prev, node_id).expect("link failed");
        }
        
        self.current_node = Some(node_id);
        self
    }
    
    /// Add a sink element and complete the pipeline.
    pub fn sink<S: Sink + 'static>(self, sink: S) -> PipelineBuilder<Complete> {
        self.sink_named(Self::auto_name::<S>(), sink)
    }
    
    /// Add a sink element with a name.
    pub fn sink_named<S: Sink + 'static>(
        mut self,
        name: impl Into<String>,
        sink: S,
    ) -> PipelineBuilder<Complete> {
        let name = name.into();
        let node_id = self.pipeline.add_element(&name, Box::new(sink));
        self.named_nodes.insert(name, node_id);
        
        if let Some(prev) = self.current_node {
            self.pipeline.link(prev, node_id).expect("link failed");
        }
        
        PipelineBuilder {
            pipeline: self.pipeline,
            current_node: Some(node_id),
            named_nodes: self.named_nodes,
            _state: PhantomData,
        }
    }
    
    /// Create a branch point (tee).
    pub fn tee<F>(mut self, f: F) -> Self
    where
        F: FnOnce(&mut TeeBuilder),
    {
        let tee_node = self.current_node.expect("tee requires previous element");
        let mut tee_builder = TeeBuilder::new(&mut self.pipeline, tee_node);
        f(&mut tee_builder);
        self
    }
}

impl PipelineBuilder<Complete> {
    /// Build the pipeline.
    pub fn build(self) -> Result<Pipeline> {
        Ok(self.pipeline)
    }
}
```

### TeeBuilder for Branching

```rust
pub struct TeeBuilder<'a> {
    pipeline: &'a mut Pipeline,
    tee_node: NodeId,
}

impl<'a> TeeBuilder<'a> {
    fn new(pipeline: &'a mut Pipeline, tee_node: NodeId) -> Self {
        Self { pipeline, tee_node }
    }
    
    /// Add a branch from the tee point.
    pub fn branch<F>(&mut self, f: F)
    where
        F: FnOnce(BranchBuilder) -> BranchBuilder<Complete>,
    {
        let branch = BranchBuilder::new(self.pipeline, self.tee_node);
        f(branch);
    }
}

pub struct BranchBuilder<'a, State = Empty> {
    pipeline: &'a mut Pipeline,
    start_node: NodeId,
    current_node: NodeId,
    _state: PhantomData<State>,
}

impl<'a> BranchBuilder<'a, Empty> {
    fn new(pipeline: &'a mut Pipeline, start_node: NodeId) -> Self {
        Self {
            pipeline,
            start_node,
            current_node: start_node,
            _state: PhantomData,
        }
    }
    
    pub fn then<T: Transform + 'static>(mut self, transform: T) -> BranchBuilder<'a, HasSource> {
        let node_id = self.pipeline.add_element(
            &format!("branch_transform_{}", self.pipeline.node_count()),
            Box::new(transform),
        );
        self.pipeline.link(self.current_node, node_id).unwrap();
        
        BranchBuilder {
            pipeline: self.pipeline,
            start_node: self.start_node,
            current_node: node_id,
            _state: PhantomData,
        }
    }
    
    pub fn sink<S: Sink + 'static>(mut self, sink: S) -> BranchBuilder<'a, Complete> {
        let node_id = self.pipeline.add_element(
            &format!("branch_sink_{}", self.pipeline.node_count()),
            Box::new(sink),
        );
        self.pipeline.link(self.current_node, node_id).unwrap();
        
        BranchBuilder {
            pipeline: self.pipeline,
            start_node: self.start_node,
            current_node: node_id,
            _state: PhantomData,
        }
    }
}
```

### Shr Operator (`>>`)

```rust
/// Trait for elements that can be chained
pub trait Chainable {
    type Output;
    fn chain_with<T: Transform + 'static>(self, next: T) -> Self::Output;
    fn chain_sink<S: Sink + 'static>(self, sink: S) -> BuiltPipeline;
}

/// Wrapper for source in chain
pub struct ChainedSource<S> {
    source: S,
}

/// Wrapper for chained elements
pub struct ChainedElements {
    builder: PipelineBuilder<HasSource>,
}

impl<S: Source + 'static> Shr<T> for S 
where
    T: Transform + 'static,
{
    type Output = ChainedElements;
    
    fn shr(self, transform: T) -> Self::Output {
        let builder = PipelineBuilder::new()
            .source(self)
            .then(transform);
        ChainedElements { builder }
    }
}

impl Shr<T> for ChainedElements
where
    T: Transform + 'static,
{
    type Output = ChainedElements;
    
    fn shr(self, transform: T) -> Self::Output {
        ChainedElements {
            builder: self.builder.then(transform),
        }
    }
}

impl<S: Sink + 'static> Shr<S> for ChainedElements {
    type Output = BuiltPipeline;
    
    fn shr(self, sink: S) -> Self::Output {
        BuiltPipeline {
            pipeline: self.builder.sink(sink).build().unwrap(),
        }
    }
}

pub struct BuiltPipeline {
    pipeline: Pipeline,
}

impl BuiltPipeline {
    pub async fn run(mut self) -> Result<()> {
        self.pipeline.run().await
    }
    
    pub fn into_pipeline(self) -> Pipeline {
        self.pipeline
    }
}
```

---

## Usage Examples

### Linear Pipeline

```rust
// Method chaining
let pipeline = PipelineBuilder::new()
    .source(VideoTestSrc::new())
    .then(VideoScale::new(1920, 1080, 1280, 720))
    .sink(NullSink::new())
    .build()?;

// Operator syntax
let pipeline = (
    VideoTestSrc::new()
    >> VideoScale::new(1920, 1080, 1280, 720)
    >> NullSink::new()
).into_pipeline();
```

### With Configuration

```rust
let pipeline = PipelineBuilder::new()
    .source(VideoTestSrc::new()
        .pattern(TestPattern::ColorBars)
        .framerate(30, 1)
        .width(1920)
        .height(1080))
    .then(VideoScale::new(1920, 1080, 1280, 720)
        .mode(ScaleMode::Bilinear))
    .then(Rav1eEncoder::new(
        Rav1eConfig::default()
            .speed(6)
            .quantizer(100)))
    .sink(FileSink::new("output.av1"))
    .build()?;
```

### Tee (One-to-Many)

```rust
let pipeline = PipelineBuilder::new()
    .source(VideoTestSrc::new())
    .tee(|t| {
        // High quality branch
        t.branch(|b| b
            .then(VideoScale::new(1920, 1080, 1920, 1080))
            .then(Rav1eEncoder::new(Rav1eConfig::default().quantizer(50)))
            .sink(FileSink::new("hq.av1")));
        
        // Low quality branch
        t.branch(|b| b
            .then(VideoScale::new(1920, 1080, 640, 480))
            .then(Rav1eEncoder::new(Rav1eConfig::default().quantizer(150)))
            .sink(FileSink::new("lq.av1")));
    })
    .build()?;
```

### Mux (Many-to-One)

```rust
let pipeline = PipelineBuilder::new()
    .sources(|s| {
        s.add_named("video", VideoTestSrc::new());
        s.add_named("klv", KlvSource::new());
    })
    .mux_with::<TsMux>(TsMuxConfig::default(), |m| {
        m.connect("video", "video_0");
        m.connect("klv", "data_0");
    })
    .sink(UdpSink::new("127.0.0.1:5000"))
    .build()?;
```

---

## Implementation Steps

### Step 1: Basic PipelineBuilder

**File:** `src/pipeline/builder.rs`

- `PipelineBuilder` struct with state markers
- `source()`, `then()`, `sink()`, `build()` methods

### Step 2: Named Elements

- `source_named()`, `then_named()`, `sink_named()` methods
- `named_nodes` HashMap

### Step 3: TeeBuilder

**File:** `src/pipeline/builder.rs`

- `TeeBuilder` struct
- `BranchBuilder` struct
- `tee()` method

### Step 4: Operator Syntax

**File:** `src/pipeline/operator.rs`

- `Shr` trait implementations
- `ChainedSource`, `ChainedElements`, `BuiltPipeline`

### Step 5: Mux Support

- `MuxBuilder` struct
- `sources()` and `mux_with()` methods

### Step 6: Documentation and Examples

**File:** `examples/36_builder_dsl.rs`

---

## Validation Criteria

- [ ] `PipelineBuilder` creates linear pipelines
- [ ] Named elements accessible after build
- [ ] `tee()` creates branching pipelines
- [ ] `>>` operator works for simple pipelines
- [ ] Mux support works for N-to-1
- [ ] Type states prevent invalid pipelines
- [ ] Example demonstrates all features
- [ ] All existing tests pass

---

## Future Enhancements

1. **Type-safe caps:** Compile-time format checking
2. **Macro DSL:** `pipeline!` macro for declarative syntax
3. **Dynamic building:** Add/remove elements at runtime
4. **Subgraphs:** Reusable pipeline fragments
5. **Visualization:** Generate DOT from builder

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `src/pipeline/builder.rs` | New: PipelineBuilder, TeeBuilder |
| `src/pipeline/operator.rs` | New: >> operator support |
| `src/pipeline/mod.rs` | Export builder |
| `examples/36_builder_dsl.rs` | New example |
