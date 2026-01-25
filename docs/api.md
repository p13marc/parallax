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

// UDP networking
pub struct UdpSrc { /* ... */ }
pub struct UdpSink { /* ... */ }
pub struct AsyncUdpSrc { /* ... */ }
pub struct AsyncUdpSink { /* ... */ }

// File descriptor I/O
pub struct FdSrc { /* ... */ }
pub struct FdSink { /* ... */ }

// Application integration
pub struct AppSrc { /* ... */ }
pub struct AppSink { /* ... */ }

// Test/utility sources
pub struct DataSrc { /* ... */ }
pub struct TestSrc { /* ... */ }
pub struct NullSource { /* ... */ }

// Debug/utility sinks
pub struct ConsoleSink { /* ... */ }
pub struct NullSink;

// Transforms
pub struct PassThrough;
pub struct RateLimiter { /* ... */ }
pub struct Valve { /* ... */ }
pub struct Queue { /* ... */ }

// Routing
pub struct Tee { /* ... */ }
pub struct Funnel { /* ... */ }
pub struct InputSelector { /* ... */ }
pub struct OutputSelector { /* ... */ }
pub struct Concat { /* ... */ }
pub struct StreamIdDemux { /* ... */ }

// Memory-based I/O
pub struct MemorySrc { /* ... */ }
pub struct MemorySink { /* ... */ }
pub struct SharedMemorySink { /* ... */ }

// Identity and delay
pub struct Identity { /* ... */ }
pub struct Delay { /* ... */ }
pub struct AsyncDelay { /* ... */ }

// Metadata operations
pub struct SequenceNumber { /* ... */ }
pub struct Timestamper { /* ... */ }
pub struct MetadataInject { /* ... */ }

// Buffer operations
pub struct BufferTrim { /* ... */ }
pub struct BufferSlice { /* ... */ }
pub struct BufferPad { /* ... */ }

// Filtering
pub struct Filter<F> { /* ... */ }
pub struct SampleFilter { /* ... */ }
pub struct MetadataFilter { /* ... */ }

// Transform
pub struct Map<F> { /* ... */ }
pub struct FilterMap<F> { /* ... */ }
pub struct Chunk { /* ... */ }

// Batching
pub struct Batch { /* ... */ }
pub struct Unbatch { /* ... */ }

// Timing control
pub struct Timeout { /* ... */ }
pub struct Debounce { /* ... */ }
pub struct Throttle { /* ... */ }
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

#### AppSrc

Inject data from application code into a pipeline.

```rust
impl AppSrc {
    pub fn new() -> Self;
    pub fn with_max_buffers(max_buffers: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn handle(&self) -> AppSrcHandle;
}

impl AppSrcHandle {
    pub fn push_buffer(&self, buffer: Buffer) -> Result<()>;
    pub fn push_buffer_timeout(&self, buffer: Buffer, timeout: Option<Duration>) -> Result<()>;
    pub fn end_stream(&self);
}
```

#### AppSink

Extract data from a pipeline to application code.

```rust
impl AppSink {
    pub fn new() -> Self;
    pub fn with_max_buffers(max_buffers: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn drop_on_full(self, drop: bool) -> Self;
    pub fn handle(&self) -> AppSinkHandle;
}

impl AppSinkHandle {
    pub fn pull_buffer(&self) -> Result<Option<Buffer>>;
    pub fn pull_buffer_timeout(&self, timeout: Option<Duration>) -> Result<Option<Buffer>>;
    pub fn try_pull_buffer(&self) -> Option<Buffer>;
}
```

#### Queue

Async buffer queue with backpressure and leaky modes.

```rust
pub enum LeakyMode {
    None,      // Block until space (default)
    Upstream,  // Drop new buffers when full
    Downstream, // Drop old buffers when full
}

impl Queue {
    pub fn new(max_buffers: usize) -> Self;
    pub fn with_limits(max_buffers: usize, max_bytes: usize) -> Self;
    pub fn leaky(self, mode: LeakyMode) -> Self;
    pub fn push(&self, buffer: Buffer) -> Result<()>;
    pub fn pop(&self) -> Result<Option<Buffer>>;
    pub fn stats(&self) -> QueueStats;
}
```

#### Valve

Flow control on/off switch.

```rust
impl Valve {
    pub fn new() -> Self;
    pub fn with_state(open: bool) -> Self;
    pub fn control(&self) -> ValveControl;
}

impl ValveControl {
    pub fn open(&self);
    pub fn close(&self);
    pub fn toggle(&self) -> bool;
    pub fn is_open(&self) -> bool;
}
```

#### TestSrc

Generate test pattern buffers for testing and benchmarking.

```rust
pub enum TestPattern {
    Zero,        // All zeros
    Ones,        // All 0xFF
    Counter,     // Incrementing bytes
    Random,      // Random data
    Alternating, // 0x55/0xAA pattern
    Sequence,    // Sequence number repeated
}

impl TestSrc {
    pub fn new() -> Self;
    pub fn with_pattern(self, pattern: TestPattern) -> Self;
    pub fn with_buffer_size(self, size: usize) -> Self;
    pub fn with_num_buffers(self, count: u64) -> Self;
    pub fn with_rate(self, bytes_per_second: u64) -> Self;
}
```

#### Funnel

Merge multiple inputs into a single output (N-to-1).

```rust
impl Funnel {
    pub fn new() -> Self;
    pub fn with_max_buffers(max_buffers: usize) -> Self;
    pub fn new_input(&self) -> FunnelInput;
}

impl FunnelInput {
    pub fn push(&self, buffer: Buffer) -> Result<()>;
    pub fn end_stream(&self);
}
```

#### InputSelector / OutputSelector

Stream routing elements.

```rust
// N-to-1 selection
impl InputSelector {
    pub fn new() -> Self;
    pub fn new_input(&self) -> SelectorInput;
    pub fn select(&self, input: usize);
}

// 1-to-N routing
impl OutputSelector {
    pub fn new() -> Self;
    pub fn new_output(&self) -> SelectorOutput;
    pub fn select(&self, output: usize);
}
```

#### Concat

Concatenate multiple streams sequentially.

```rust
impl Concat {
    pub fn new() -> Self;
    pub fn add_stream(&self) -> ConcatStream;
    pub fn skip_to(&self, stream_index: usize);
    pub fn skip_next(&self);
}

impl ConcatStream {
    pub fn push(&self, buffer: Buffer) -> Result<()>;
    pub fn end_stream(&self);
}
```

#### Identity

Pass-through element with optional callbacks for debugging.

```rust
impl Identity {
    pub fn new() -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn with_callback<F: Fn(&Buffer) + Send + Sync + 'static>(self, f: F) -> Self;
    pub fn stats(&self) -> (u64, u64);  // (count, bytes)
    pub fn reset_stats(&self);
}
```

#### MemorySrc / MemorySink

Memory-based source and sink for testing and data manipulation.

```rust
impl MemorySrc {
    pub fn new(data: Vec<u8>) -> Self;
    pub fn with_chunk_size(self, chunk_size: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn remaining(&self) -> usize;
    pub fn reset(&mut self);
}

impl MemorySink {
    pub fn new() -> Self;
    pub fn with_max_size(max_size: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn data(&self) -> &[u8];
    pub fn take_data(&mut self) -> Vec<u8>;
    pub fn clear(&mut self);
}

// Thread-safe wrapper
impl SharedMemorySink {
    pub fn new() -> Self;
    pub fn with_max_size(max_size: usize) -> Self;
    pub fn data(&self) -> Vec<u8>;
    pub fn clear(&self);
}
```

#### Delay / AsyncDelay

Add fixed delay between buffer processing.

```rust
impl Delay {
    pub fn new(delay: Duration) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn set_delay(&self, delay: Duration);
    pub fn stats(&self) -> (u64, Duration);  // (count, total_delay)
    pub fn reset_stats(&self);
}

impl AsyncDelay {
    pub fn new(delay: Duration) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> u64;  // count
}
```

#### SequenceNumber

Adds sequence numbers to buffer metadata.

```rust
impl SequenceNumber {
    pub fn new() -> Self;
    pub fn starting_at(start: u64) -> Self;
    pub fn with_increment(self, increment: u64) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn current(&self) -> u64;
    pub fn reset(&self);
}
```

#### Timestamper

Adds timestamps to buffer metadata.

```rust
pub enum TimestampMode {
    SystemTime,   // System clock time
    Monotonic,    // Monotonic counter from start
    Preserve,     // Don't overwrite existing
    PtsOnly,      // Only set PTS
    DtsOnly,      // Only set DTS
}

impl Timestamper {
    pub fn new(mode: TimestampMode) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn buffer_count(&self) -> u64;
}
```

#### MetadataInject

Inject metadata fields into buffers.

```rust
impl MetadataInject {
    pub fn new() -> Self;
    pub fn with_stream_id(self, id: u64) -> Self;
    pub fn with_duration(self, duration: Duration) -> Self;
    pub fn with_offset(self, offset: u64) -> Self;
    pub fn with_offset_end(self, offset_end: u64) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn count(&self) -> u64;
}
```

#### BufferTrim / BufferSlice / BufferPad

Buffer size manipulation operations.

```rust
impl BufferTrim {
    pub fn new(max_size: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> BufferTrimStats;
}

impl BufferSlice {
    pub fn new(offset: usize, length: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
}

impl BufferPad {
    pub fn new(min_size: usize, fill_byte: u8) -> Self;
    pub fn with_zeros(min_size: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> BufferPadStats;
}
```

#### Filter / SampleFilter / MetadataFilter

Buffer filtering elements.

```rust
impl<F: FnMut(&Buffer) -> bool + Send> Filter<F> {
    pub fn new(predicate: F) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> (u64, u64);  // (passed, dropped)
}

pub enum SampleMode {
    EveryNth(u64),       // Pass every Nth buffer
    RandomPercent(u8),   // Random sampling (0-100%)
    FirstN(u64),         // Pass first N, then drop
    SkipFirst(u64),      // Skip first N, then pass
}

impl SampleFilter {
    pub fn every_nth(n: u64) -> Self;
    pub fn random_percent(percent: u8) -> Self;
    pub fn first_n(n: u64) -> Self;
    pub fn skip_first(n: u64) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> (u64, u64);  // (passed, dropped)
}

impl MetadataFilter {
    pub fn new() -> Self;
    pub fn with_stream_id(self, id: u64) -> Self;
    pub fn with_min_sequence(self, min: u64) -> Self;
    pub fn with_max_sequence(self, max: u64) -> Self;
    pub fn with_sequence_range(self, min: u64, max: u64) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
}
```

#### Map / FilterMap / Chunk

Data transformation elements.

```rust
impl<F: FnMut(&[u8]) -> Vec<u8> + Send> Map<F> {
    pub fn new(f: F) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn buffer_count(&self) -> u64;
}

impl<F: FnMut(&[u8]) -> Option<Vec<u8>> + Send> FilterMap<F> {
    pub fn new(f: F) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
}

impl Chunk {
    pub fn new(chunk_size: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> ChunkStats;
    pub fn flush(&mut self) -> Result<Option<Buffer>>;
}
```

#### Batch / Unbatch

Buffer aggregation elements.

```rust
impl Batch {
    pub fn by_count(max_count: usize) -> Self;
    pub fn by_size(max_bytes: usize) -> Self;
    pub fn with_limits(max_count: usize, max_bytes: usize) -> Self;
    pub fn with_timeout(self, timeout: Duration) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn pending_count(&self) -> usize;
    pub fn flush(&mut self) -> Result<Option<Buffer>>;
}

impl Unbatch {
    pub fn new(chunk_size: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> UnbatchStats;
}
```

#### Timeout / Debounce / Throttle

Timing control elements.

```rust
impl Timeout {
    pub fn new(timeout: Duration) -> Self;
    pub fn with_fallback_data(self, data: Vec<u8>) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn check_timeout(&mut self) -> Result<Option<Buffer>>;
}

impl Debounce {
    pub fn new(quiet_period: Duration) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn flush(&mut self) -> Option<Buffer>;
}

impl Throttle {
    pub fn new(min_interval: Duration) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> ThrottleStats;
}
```

#### UnixSrc / UnixSink

Unix domain socket elements for local IPC.

```rust
impl UnixSrc {
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn listen<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn with_buffer_size(self, size: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn with_read_timeout(self, timeout: Duration) -> Self;
    pub fn bytes_read(&self) -> u64;
    pub fn path(&self) -> &Path;
}

impl UnixSink {
    pub fn connect<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn listen<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn with_write_timeout(self, timeout: Duration) -> Self;
    pub fn bytes_written(&self) -> u64;
}
```

#### UdpMulticastSrc / UdpMulticastSink

UDP multicast elements for one-to-many distribution.

```rust
impl UdpMulticastSrc {
    pub fn new(multicast_addr: &str, port: u16) -> Result<Self>;
    pub fn with_buffer_size(self, size: usize) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn with_timeout(self, timeout: Duration) -> Result<Self>;
    pub fn multicast_addr(&self) -> Ipv4Addr;
    pub fn stats(&self) -> UdpMulticastStats;
}

impl UdpMulticastSink {
    pub fn new(multicast_addr: &str, port: u16) -> Result<Self>;
    pub fn with_ttl(self, ttl: u32) -> Result<Self>;
    pub fn with_loopback(self, enabled: bool) -> Result<Self>;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> UdpMulticastStats;
}
```

#### HttpSrc / HttpSink (requires `http` feature)

HTTP elements for web-based data transfer.

```rust
impl HttpSrc {
    pub fn new(url: impl Into<String>) -> Result<Self>;
    pub fn with_chunk_size(self, size: usize) -> Self;
    pub fn with_timeout(self, timeout: Duration) -> Self;
    pub fn with_header(self, name: impl Into<String>, value: impl Into<String>) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn status_code(&self) -> Option<u16>;
    pub fn bytes_read(&self) -> u64;
}

impl HttpSink {
    pub fn new(url: impl Into<String>, method: HttpMethod) -> Result<Self>;
    pub fn with_timeout(self, timeout: Duration) -> Self;
    pub fn with_header(self, name: impl Into<String>, value: impl Into<String>) -> Self;
    pub fn with_content_type(self, content_type: impl Into<String>) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn stats(&self) -> HttpSinkStats;
}

pub enum HttpMethod { Post, Put, Patch }
```

#### WebSocketSrc / WebSocketSink (requires `websocket` feature)

WebSocket elements for bidirectional communication.

```rust
impl WebSocketSrc {
    pub fn new(url: impl Into<String>) -> Result<Self>;
    pub fn connect(url: impl Into<String>) -> Result<Self>;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn is_connected(&self) -> bool;
    pub fn stats(&self) -> WebSocketStats;
    pub fn close(&mut self) -> Result<()>;
}

impl WebSocketSink {
    pub fn new(url: impl Into<String>) -> Result<Self>;
    pub fn connect(url: impl Into<String>) -> Result<Self>;
    pub fn with_message_type(self, msg_type: WsMessageType) -> Self;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn ping(&mut self) -> Result<()>;
    pub fn close(&mut self) -> Result<()>;
    pub fn stats(&self) -> WebSocketStats;
}

pub enum WsMessageType { Binary, Text }
```

#### ZenohSrc / ZenohSink (requires `zenoh` feature)

Zenoh pub/sub elements for distributed pipelines.

```rust
impl ZenohSrc {
    pub async fn new(key_expr: impl Into<String>) -> Result<Self>;
    pub async fn with_session(session: Arc<Session>, key_expr: impl Into<String>) -> Result<Self>;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn with_timeout(self, timeout: Duration) -> Self;
    pub fn key_expr(&self) -> &str;
    pub fn stats(&self) -> ZenohStats;
}

impl ZenohSink {
    pub async fn new(key_expr: impl Into<String>) -> Result<Self>;
    pub async fn with_session(session: Arc<Session>, key_expr: impl Into<String>) -> Result<Self>;
    pub fn with_name(self, name: impl Into<String>) -> Self;
    pub fn with_congestion_control(self, cc: ZenohCongestionControl) -> Self;
    pub fn with_priority(self, priority: ZenohPriority) -> Self;
    pub fn stats(&self) -> ZenohStats;
}
```

#### ZenohQueryable / ZenohQuerier (requires `zenoh` feature)

Zenoh query elements.

```rust
impl ZenohQueryable {
    pub async fn new(key_expr: impl Into<String>) -> Result<Self>;
    pub fn recv_query(&mut self) -> Result<Option<ZenohQuery>>;
    pub fn try_recv_query(&mut self) -> Option<ZenohQuery>;
}

impl ZenohQuerier {
    pub async fn new() -> Result<Self>;
    pub fn with_session(session: Arc<Session>) -> Self;
    pub fn with_timeout(self, timeout: Duration) -> Self;
    pub async fn get(&mut self, key_expr: &str) -> Result<Vec<Buffer>>;
    pub async fn get_with_value(&mut self, key_expr: &str, value: &[u8]) -> Result<Vec<Buffer>>;
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
