# Getting Started with Parallax

This guide will walk you through creating your first Parallax pipeline.

## Installation

Add Parallax to your `Cargo.toml`:

```toml
[dependencies]
parallax = "0.1"
tokio = { version = "1", features = ["rt-multi-thread", "macros"] }
```

## Your First Pipeline

### Dynamic Pipeline

The simplest way to create a pipeline is using the dynamic API:

```rust
use parallax::buffer::{Buffer, MemoryHandle};
use parallax::element::{Source, SourceAdapter, Sink, SinkAdapter};
use parallax::error::Result;
use parallax::memory::HeapSegment;
use parallax::metadata::Metadata;
use parallax::pipeline::{Pipeline, PipelineExecutor};
use std::sync::Arc;

// Define a simple source that produces 10 buffers
struct CountingSource {
    current: u64,
    max: u64,
}

impl Source for CountingSource {
    fn produce(&mut self) -> Result<Option<Buffer<()>>> {
        if self.current >= self.max {
            return Ok(None);  // End of stream
        }
        
        let segment = Arc::new(HeapSegment::new(64)?);
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::with_sequence(self.current));
        
        self.current += 1;
        Ok(Some(buffer))
    }
}

// Define a simple sink that prints buffers
struct PrintingSink;

impl Sink for PrintingSink {
    fn consume(&mut self, buffer: Buffer<()>) -> Result<()> {
        println!("Received buffer: seq={}", buffer.metadata().sequence);
        Ok(())
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Create the pipeline
    let mut pipeline = Pipeline::new();
    
    // Add elements
    let src = pipeline.add_node(
        "source",
        Box::new(SourceAdapter::new(CountingSource { current: 0, max: 10 }))
    );
    let sink = pipeline.add_node(
        "sink", 
        Box::new(SinkAdapter::new(PrintingSink))
    );
    
    // Link them together
    pipeline.link(src, sink)?;
    
    // Run the pipeline
    let executor = PipelineExecutor::new();
    executor.run(&mut pipeline).await?;
    
    println!("Pipeline completed!");
    Ok(())
}
```

### Typed Pipeline

For compile-time type safety, use the typed API:

```rust
use parallax::typed::{pipeline, from_iter, map, filter, collect};
use parallax::error::Result;

fn main() -> Result<()> {
    // Create a source from an iterator
    let source = from_iter(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    
    // Build the pipeline with type-safe operators
    let result = pipeline(source)
        .then(filter(|x: &i32| x % 2 == 0))  // Keep even numbers
        .then(map(|x: i32| x * x))            // Square them
        .sink(collect::<i32>())               // Collect results
        .run()?
        .into_inner();
    
    println!("Squares of even numbers: {:?}", result);
    // Output: [4, 16, 36, 64, 100]
    
    Ok(())
}
```

## Key Concepts

### Buffers

Buffers are the unit of data in a pipeline. Each buffer contains:
- **Memory**: A reference to the underlying memory segment
- **Metadata**: Timestamps, sequence numbers, flags

```rust
use parallax::buffer::{Buffer, MemoryHandle};
use parallax::memory::HeapSegment;
use parallax::metadata::Metadata;

// Create a buffer
let segment = Arc::new(HeapSegment::new(1024)?);
let handle = MemoryHandle::from_segment(segment);
let buffer = Buffer::<()>::new(handle, Metadata::with_sequence(0));

// Access buffer properties
println!("Length: {}", buffer.len());
println!("Sequence: {}", buffer.metadata().sequence);
```

### Elements

Elements are the building blocks of pipelines:

| Type | Description | Trait |
|------|-------------|-------|
| **Source** | Produces buffers | `Source` |
| **Sink** | Consumes buffers | `Sink` |
| **Transform** | Modifies buffers | `Element` |

### Memory Backends

Parallax supports multiple memory backends:

```rust
use parallax::memory::{HeapSegment, SharedMemorySegment, MemoryPool};

// Heap memory (default, single-process)
let heap = HeapSegment::new(4096)?;

// Shared memory (multi-process)
let shared = SharedMemorySegment::new("my-segment", 1024 * 1024)?;

// Memory pool for efficient allocation
let pool = MemoryPool::new(heap, 1024)?;  // 1KB slots
let slot = pool.loan().expect("pool not exhausted");
```

## Next Steps

- [Architecture](architecture.md) - Understand Parallax internals
- [Memory Model](memory.md) - Learn about zero-copy buffers
- [API Reference](api.md) - Complete API documentation
- [Plugin Development](plugins.md) - Create custom elements

## Examples

Check out the examples directory:

```bash
# Simple dynamic pipeline
cargo run --example simple_pipeline

# Typed pipeline with operators
cargo run --example typed_pipeline

# Multi-process communication
cargo run --example multi_process
```
