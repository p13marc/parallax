# API Reference

This document provides a reference for the main Parallax APIs.

## Module Overview

| Module | Description |
|--------|-------------|
| `parallax::buffer` | Buffer and memory handle types |
| `parallax::element` | Element traits (Source, Sink, Element) |
| `parallax::elements` | Built-in element implementations |
| `parallax::memory` | Memory segment and pool types |
| `parallax::metadata` | Buffer metadata |
| `parallax::pipeline` | Pipeline construction and execution |
| `parallax::typed` | Type-safe pipeline builder |
| `parallax::link` | Inter-element communication |
| `parallax::plugin` | Plugin system |
| `parallax::observability` | Metrics and tracing |

## Buffer Module

### `Buffer<T>`

A buffer containing data and metadata.

```rust
pub struct Buffer<T = ()> {
    // ...
}

impl<T> Buffer<T> {
    /// Create a new buffer
    pub fn new(memory: MemoryHandle, metadata: Metadata) -> Self;
    
    /// Get a reference to the memory handle
    pub fn memory(&self) -> &MemoryHandle;
    
    /// Get a reference to the metadata
    pub fn metadata(&self) -> &Metadata;
    
    /// Get the length of the buffer data
    pub fn len(&self) -> usize;
    
    /// Check if the buffer is empty
    pub fn is_empty(&self) -> bool;
}
```

### `MemoryHandle`

A reference to memory within a segment.

```rust
pub struct MemoryHandle {
    // ...
}

impl MemoryHandle {
    /// Create a new handle with explicit offset and length
    pub fn new(segment: Arc<dyn MemorySegment>, offset: usize, len: usize) -> Self;
    
    /// Create a handle covering the entire segment
    pub fn from_segment(segment: Arc<dyn MemorySegment>) -> Self;
    
    /// Create a handle with specific length from start
    pub fn from_segment_with_len(segment: Arc<dyn MemorySegment>, len: usize) -> Self;
    
    /// Get the underlying segment
    pub fn segment(&self) -> &Arc<dyn MemorySegment>;
    
    /// Get the offset within the segment
    pub fn offset(&self) -> usize;
    
    /// Get the length
    pub fn len(&self) -> usize;
    
    /// Get the memory type
    pub fn memory_type(&self) -> MemoryType;
}
```

## Element Module

### `Source` Trait

Produces buffers.

```rust
pub trait Source: Send {
    /// Produce the next buffer
    /// Returns Ok(None) for end-of-stream
    fn produce(&mut self) -> Result<Option<Buffer<()>>>;
}
```

### `Sink` Trait

Consumes buffers.

```rust
pub trait Sink: Send {
    /// Consume a buffer
    fn consume(&mut self, buffer: Buffer<()>) -> Result<()>;
}
```

### `Element` Trait

Transforms buffers.

```rust
pub trait Element: Send {
    /// Process a buffer, optionally producing output
    /// Returns Ok(None) to filter out the buffer
    fn process(&mut self, buffer: Buffer<()>) -> Result<Option<Buffer<()>>>;
}
```

### `AsyncSource` Trait

Async source for I/O-bound operations.

```rust
pub trait AsyncSource: Send {
    /// Produce the next buffer asynchronously
    fn produce_async(&mut self) -> impl Future<Output = Result<Option<Buffer<()>>>> + Send;
}
```

## Elements Module

### Built-in Elements

```rust
// File I/O
pub struct FileSrc { /* ... */ }
pub struct FileSink { /* ... */ }

// TCP networking
pub struct TcpSrc { /* ... */ }
pub struct TcpSink { /* ... */ }
pub struct AsyncTcpSrc { /* ... */ }
pub struct AsyncTcpSink { /* ... */ }

// Utilities
pub struct PassThrough;
pub struct Tee { /* ... */ }
pub struct NullSink;
pub struct NullSource { /* ... */ }
```

#### FileSrc

```rust
impl FileSrc {
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn with_buffer_size<P: AsRef<Path>>(path: P, buffer_size: usize) -> Result<Self>;
}
```

#### TcpSrc

```rust
pub enum TcpMode {
    Client { address: String },
    Server { bind_address: String },
}

impl TcpSrc {
    pub fn new(mode: TcpMode) -> Result<Self>;
    pub fn with_buffer_size(mode: TcpMode, buffer_size: usize) -> Result<Self>;
}
```

## Memory Module

### `MemorySegment` Trait

```rust
pub trait MemorySegment: Send + Sync {
    unsafe fn as_ptr(&self) -> *const u8;
    unsafe fn as_mut_ptr(&self) -> Option<*mut u8>;
    fn len(&self) -> usize;
    fn memory_type(&self) -> MemoryType;
    fn ipc_handle(&self) -> Option<IpcHandle>;
}
```

### Segment Implementations

```rust
// Heap memory
pub struct HeapSegment { /* ... */ }
impl HeapSegment {
    pub fn new(size: usize) -> Result<Self>;
}

// Shared memory
pub struct SharedMemorySegment { /* ... */ }
impl SharedMemorySegment {
    pub fn new(name: &str, size: usize) -> Result<Self>;
    pub fn from_raw_fd(fd: RawFd, size: usize) -> Result<Self>;
}

// Huge pages
pub struct HugePageSegment { /* ... */ }
impl HugePageSegment {
    pub fn new(size: HugePageSize, total_size: usize) -> Result<Self>;
    pub fn new_or_fallback(size: HugePageSize, total_size: usize) -> Result<Self>;
}

// Memory-mapped file
pub struct MappedFileSegment { /* ... */ }
impl MappedFileSegment {
    pub fn create<P: AsRef<Path>>(path: P, size: usize) -> Result<Self>;
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn open_readonly<P: AsRef<Path>>(path: P) -> Result<Self>;
}
```

### `MemoryPool`

```rust
pub struct MemoryPool { /* ... */ }

impl MemoryPool {
    pub fn new<S: MemorySegment + 'static>(segment: S, slot_size: usize) -> Result<Self>;
    pub fn loan(&self) -> Option<LoanedSlot<'_>>;
    pub fn slot_size(&self) -> usize;
    pub fn capacity(&self) -> usize;
    pub fn available(&self) -> usize;
}
```

### `LoanedSlot`

```rust
pub struct LoanedSlot<'pool> { /* ... */ }

impl<'pool> LoanedSlot<'pool> {
    pub fn as_ptr(&self) -> *const u8;
    pub fn as_mut_ptr(&self) -> *mut u8;
    pub fn len(&self) -> usize;
    pub fn as_slice(&self) -> &[u8];
    pub fn as_mut_slice(&mut self) -> &mut [u8];
}
// Automatically returns to pool when dropped
```

## Pipeline Module

### `Pipeline`

```rust
pub struct Pipeline { /* ... */ }

impl Pipeline {
    pub fn new() -> Self;
    
    /// Parse a pipeline from string
    pub fn parse(&mut self, description: &str) -> Result<()>;
    
    /// Add a node
    pub fn add_node(&mut self, name: &str, element: Box<dyn ElementDyn>) -> NodeId;
    
    /// Link two nodes
    pub fn link(&mut self, from: NodeId, to: NodeId) -> Result<()>;
    
    /// Get current state
    pub fn state(&self) -> PipelineState;
    
    /// Validate the pipeline
    pub fn validate(&self) -> Result<()>;
    
    /// Export to DOT format
    pub fn to_dot(&self) -> String;
    
    /// Export to JSON
    pub fn to_json(&self) -> String;
}
```

### `PipelineExecutor`

```rust
pub struct PipelineExecutor { /* ... */ }

impl PipelineExecutor {
    pub fn new() -> Self;
    pub fn with_config(config: ExecutorConfig) -> Self;
    
    /// Run pipeline to completion
    pub async fn run(&self, pipeline: &mut Pipeline) -> Result<()>;
    
    /// Start pipeline and return handle
    pub fn start(&self, pipeline: &mut Pipeline) -> Result<PipelineHandle>;
}
```

### `PipelineHandle`

```rust
pub struct PipelineHandle { /* ... */ }

impl PipelineHandle {
    /// Wait for pipeline to complete
    pub async fn wait(self) -> Result<()>;
    
    /// Abort the pipeline
    pub fn abort(self);
    
    /// Subscribe to events
    pub fn subscribe(&self) -> EventReceiver;
}
```

## Typed Module

### Pipeline Builder

```rust
/// Create a pipeline from a source
pub fn pipeline<S: TypedSource + 'static>(source: S) -> PipelineWithSource<S>;

/// Create a source from an iterator
pub fn from_iter<I: IntoIterator>(iter: I) -> IterSource<I::IntoIter>;
```

### Operators

```rust
/// Transform each item
pub fn map<F, In, Out>(f: F) -> Map<F, In, Out>;

/// Filter items by predicate
pub fn filter<F, T>(predicate: F) -> Filter<F, T>;

/// Transform and filter in one step
pub fn filter_map<F, In, Out>(f: F) -> FilterMap<F, In, Out>;

/// Take first n items
pub fn take<T>(n: usize) -> Take<T>;

/// Skip first n items
pub fn skip<T>(n: usize) -> Skip<T>;

/// Side-effect without modification
pub fn inspect<F, T>(f: F) -> Inspect<F, T>;
```

### Sinks

```rust
/// Collect items into a vector
pub fn collect<T>() -> CollectSink<T>;

/// Discard all items
pub fn discard<T>() -> DiscardSink<T>;

/// Call a function for each item
pub fn for_each<F, T>(f: F) -> ForEachSink<F, T>;
```

## Link Module

### `LocalLink`

In-process communication via channels.

```rust
pub struct LocalLink;

impl LocalLink {
    pub fn bounded(capacity: usize) -> (LocalSender, LocalReceiver);
    pub fn unbounded() -> (LocalSender, LocalReceiver);
}
```

### `IpcPublisher` / `IpcSubscriber`

Cross-process communication.

```rust
pub struct IpcPublisher { /* ... */ }

impl IpcPublisher {
    pub fn bind<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn accept(&mut self) -> Result<()>;
    pub fn send(&mut self, buffer: Buffer) -> Result<()>;
    pub fn send_eos(&mut self) -> Result<()>;
}

pub struct IpcSubscriber { /* ... */ }

impl IpcSubscriber {
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn recv(&mut self) -> Result<Option<Buffer>>;
}
```

## Observability Module

### Metrics

```rust
/// Initialize the metrics recorder
pub fn init_metrics();

/// Record a buffer being produced
pub fn record_buffer_produced(pipeline: &str, element: &str, size: usize);

/// Record a buffer being consumed
pub fn record_buffer_consumed(pipeline: &str, element: &str, size: usize);

/// Record processing time
pub fn record_processing_time(pipeline: &str, element: &str, duration: Duration);
```

### Tracing

```rust
/// Create a span for pipeline execution
pub fn span_pipeline(name: &str) -> tracing::Span;

/// Create a span for element processing
pub fn span_element(pipeline: &str, element: &str) -> tracing::Span;
```

## Error Handling

### `Error` Enum

```rust
pub enum Error {
    Io(std::io::Error),
    Memory(String),
    Element(String),
    Pipeline(String),
    Config(String),
    InvalidSegment(String),
}
```

### `Result` Type

```rust
pub type Result<T> = std::result::Result<T, Error>;
```
