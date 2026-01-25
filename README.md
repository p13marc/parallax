# Parallax

A Rust-native streaming pipeline engine with zero-copy multi-process support.

Parallax provides both dynamic (runtime-configured) and typed (compile-time safe) pipeline construction, with buffers backed by shared memory for efficient multi-process communication.

## Features

- **Zero-copy buffers**: Shared memory with loan-based memory pools
- **Progressive typing**: Start dynamic, graduate to typed
- **Multi-process pipelines**: memfd + Unix socket IPC
- **rkyv serialization**: Zero-copy deserialization at boundaries
- **Linux-optimized**: memfd_create, huge pages, memory-mapped files
- **Plugin system**: Dynamic element loading with C ABI compatibility
- **Observability**: Built-in metrics and tracing support

## Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
parallax = "0.1"
```

### Dynamic Pipeline

Build pipelines at runtime using a GStreamer-like syntax:

```rust
use parallax::pipeline::{Pipeline, PipelineExecutor};

#[tokio::main]
async fn main() -> parallax::Result<()> {
    let mut pipeline = Pipeline::new();
    
    // Parse pipeline from string
    pipeline.parse("filesrc location=input.bin ! passthrough ! nullsink")?;
    
    // Or build programmatically
    let src = pipeline.add_node("src", Box::new(source));
    let sink = pipeline.add_node("sink", Box::new(sink));
    pipeline.link(src, sink)?;
    
    // Run the pipeline
    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await?;
    
    Ok(())
}
```

### Typed Pipeline

Build pipelines with compile-time type checking:

```rust
use parallax::typed::{pipeline, from_iter, map, filter, collect};

fn main() -> parallax::Result<()> {
    let source = from_iter(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    let result = pipeline(source)
        .then(filter(|x: &i32| x % 2 == 0))  // Keep even numbers
        .then(map(|x: i32| x * 10))           // Multiply by 10
        .sink(collect::<i32>())
        .run()?
        .into_inner();
    
    println!("Result: {:?}", result);  // [20, 40, 60, 80, 100]
    Ok(())
}
```

### Multi-Process Pipeline

Use shared memory for zero-copy IPC:

```rust
use parallax::memory::{SharedMemorySegment, MemoryPool};
use parallax::link::{IpcPublisher, IpcSubscriber};

// Process A: Publisher
let segment = SharedMemorySegment::new("my-pipeline", 1024 * 1024)?;
let pool = MemoryPool::new(segment, 4096)?;
let mut publisher = IpcPublisher::bind("/tmp/pipeline.sock")?;

// Process B: Subscriber  
let mut subscriber = IpcSubscriber::connect("/tmp/pipeline.sock")?;
while let Some(buffer) = subscriber.recv()? {
    // Zero-copy access to data in shared memory
    process(buffer);
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pipeline                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐      │
│  │ Source  │───▶│Transform│───▶│   Tee   │───▶│  Sink   │      │
│  └─────────┘    └─────────┘    └────┬────┘    └─────────┘      │
│                                     │                           │
│                                     ▼                           │
│                               ┌─────────┐                       │
│                               │  Sink   │                       │
│                               └─────────┘                       │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     Memory Backend Abstraction                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  HeapSegment │  │ SharedMemory │  │  HugePages   │          │
│  │  (default)   │  │   (IPC)      │  │  (2MB/1GB)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ MappedFile   │  │  GPU Pinned  │  │    RDMA      │          │
│  │ (persistent) │  │  (future)    │  │   (future)   │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Built-in Elements

| Element | Type | Description |
|---------|------|-------------|
| `FileSrc` | Source | Reads from a file |
| `FileSink` | Sink | Writes to a file |
| `TcpSrc` / `AsyncTcpSrc` | Source | Reads from TCP connection |
| `TcpSink` / `AsyncTcpSink` | Sink | Writes to TCP connection |
| `PassThrough` | Transform | Passes buffers unchanged |
| `Tee` | Transform | Duplicates to multiple outputs |
| `NullSink` | Sink | Discards all buffers |
| `NullSource` | Source | Produces empty buffers |

## Memory Backends

| Backend | Use Case | IPC Support |
|---------|----------|-------------|
| `HeapSegment` | Default, single-process | No |
| `SharedMemorySegment` | Multi-process pipelines | Yes |
| `HugePageSegment` | High-throughput (reduced TLB misses) | Yes |
| `MappedFileSegment` | Persistent buffers | Yes |

## Typed Operators

| Operator | Description |
|----------|-------------|
| `map(fn)` | Transform each item |
| `filter(predicate)` | Keep items matching predicate |
| `filter_map(fn)` | Transform and filter in one step |
| `take(n)` | Take first n items |
| `skip(n)` | Skip first n items |
| `inspect(fn)` | Side-effect without modification |

## Performance

| Operation | Cost | Notes |
|-----------|------|-------|
| Buffer clone (same process) | O(1) | Arc increment only |
| Buffer access (typed) | O(1) | Direct pointer |
| Pool slot acquire | O(1) amortized | Atomic bitmap scan |
| Tee (N outputs) | O(N) Arc clones | No data copying |
| Cross-process send | O(data size) | One copy to shared memory |
| Cross-process receive | O(1) | Validation only, no copy |

## Documentation

- [Getting Started](docs/getting-started.md)
- [Architecture](docs/architecture.md)
- [API Reference](docs/api.md)
- [Memory Model](docs/memory.md)
- [Plugin Development](docs/plugins.md)

## Examples

Run the examples:

```bash
# Simple dynamic pipeline
cargo run --example simple_pipeline

# Typed pipeline with operators
cargo run --example typed_pipeline

# Multi-process shared memory
cargo run --example multi_process
```

## Benchmarks

Run benchmarks:

```bash
cargo bench
```

## Requirements

- Rust 1.85+
- Linux (uses Linux-specific APIs: memfd_create, SCM_RIGHTS)

## Feature Flags

| Feature | Description |
|---------|-------------|
| `macros` | Plugin authoring convenience macros |
| `huge-pages` | Enable huge page support |

## License

Licensed under either of:

- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)

at your option.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
