# Memory Model

Parallax's memory model is designed for zero-copy data passing, both within a process and across process boundaries.

## Core Concepts

### Memory Segments

A memory segment is a contiguous region of memory:

```rust
pub trait MemorySegment: Send + Sync {
    /// Get a raw pointer to the start of the segment
    unsafe fn as_ptr(&self) -> *const u8;
    
    /// Get a mutable pointer (if supported)
    unsafe fn as_mut_ptr(&self) -> Option<*mut u8>;
    
    /// Get the total size of the segment
    fn len(&self) -> usize;
    
    /// Get the memory type for capability checking
    fn memory_type(&self) -> MemoryType;
    
    /// Get an IPC handle for cross-process sharing
    fn ipc_handle(&self) -> Option<IpcHandle>;
}
```

### Memory Types

```rust
pub enum MemoryType {
    Heap,           // Regular heap allocation
    SharedMemory,   // POSIX shared memory
    HugePages,      // 2MB or 1GB pages
    MappedFile,     // Memory-mapped file
}
```

## Memory Backends

### HeapSegment (Default)

Simple heap-allocated memory for single-process use:

```rust
use parallax::memory::HeapSegment;

let segment = HeapSegment::new(4096)?;  // 4KB segment
```

**Properties:**
- Fast allocation
- No IPC support
- Suitable for most single-process pipelines

### SharedMemorySegment

POSIX shared memory using `memfd_create`:

```rust
use parallax::memory::SharedMemorySegment;

// Create a new segment
let segment = SharedMemorySegment::new("my-segment", 1024 * 1024)?;

// Open an existing segment from a file descriptor
let segment = SharedMemorySegment::from_raw_fd(fd, size)?;
```

**Properties:**
- Supports IPC via file descriptor passing
- Anonymous (no filesystem footprint)
- Automatically cleaned up when all references closed

### HugePageSegment

Large pages for reduced TLB pressure:

```rust
use parallax::memory::{HugePageSegment, HugePageSize};

// Request 2MB huge pages
let segment = HugePageSegment::new(HugePageSize::MB2, 64 * 1024 * 1024)?;

// Or 1GB huge pages
let segment = HugePageSegment::new(HugePageSize::GB1, 1024 * 1024 * 1024)?;

// Graceful fallback if huge pages unavailable
let segment = HugePageSegment::new_or_fallback(HugePageSize::MB2, size)?;
```

**Properties:**
- Requires huge pages configured on the system
- Reduces TLB misses for large working sets
- Best for high-throughput pipelines

### MappedFileSegment

Memory-mapped files for persistent storage:

```rust
use parallax::memory::MappedFileSegment;

// Create a new mapped file
let segment = MappedFileSegment::create("/path/to/file", 1024 * 1024)?;

// Open an existing file
let segment = MappedFileSegment::open("/path/to/file")?;

// Open read-only
let segment = MappedFileSegment::open_readonly("/path/to/file")?;
```

**Properties:**
- Data persists to disk
- Kernel handles I/O automatically
- Can be shared via file path

## Memory Pools

Pools provide efficient buffer allocation from a segment:

```rust
use parallax::memory::{MemoryPool, HeapSegment};

// Create a pool with 64KB slots
let segment = HeapSegment::new(1024 * 1024)?;  // 1MB total
let pool = MemoryPool::new(segment, 64 * 1024)?;

println!("Pool capacity: {} slots", pool.capacity());
println!("Available: {} slots", pool.available());

// Loan a slot
if let Some(mut slot) = pool.loan() {
    // Write data to the slot
    slot.as_mut_slice()[..data.len()].copy_from_slice(&data);
    
    // Slot automatically returns to pool when dropped
}
```

### Pool Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     Memory Segment (1MB)                         │
├─────────────────┬─────────────────┬─────────────────┬───────────┤
│  Slot 0 (64KB)  │  Slot 1 (64KB)  │  Slot 2 (64KB)  │    ...    │
│     [used]      │     [free]      │     [used]      │           │
└─────────────────┴─────────────────┴─────────────────┴───────────┘
        │                                     │
        ▼                                     ▼
   AtomicBitmap: [1, 0, 1, 0, 0, 0, ...]
```

### Atomic Bitmap

Slot allocation uses a lock-free bitmap:

```rust
// Internally:
pub struct AtomicBitmap {
    bits: Box<[AtomicU64]>,
    num_bits: usize,
}

impl AtomicBitmap {
    pub fn acquire_slot(&self) -> Option<usize>;  // Find and set a free bit
    pub fn release_slot(&self, index: usize);      // Clear a bit
}
```

**Performance:**
- O(1) amortized acquisition
- O(1) release
- Lock-free (uses atomic CAS)

## Buffers and Memory Handles

Buffers reference memory through handles:

```rust
pub struct Buffer<T = ()> {
    memory: MemoryHandle,
    metadata: Metadata,
    // ...
}

pub struct MemoryHandle {
    segment: Arc<dyn MemorySegment>,
    offset: usize,
    len: usize,
}
```

### Creating Buffers

```rust
use parallax::buffer::{Buffer, MemoryHandle};
use parallax::memory::HeapSegment;

// From a segment
let segment = Arc::new(HeapSegment::new(1024)?);
let handle = MemoryHandle::from_segment(segment);
let buffer = Buffer::<()>::new(handle, Metadata::default());

// With specific length
let handle = MemoryHandle::from_segment_with_len(segment, 512);

// With offset and length
let handle = MemoryHandle::new(segment, 256, 256);  // offset=256, len=256
```

### Zero-Copy Cloning

```rust
let buffer1 = Buffer::new(handle, metadata);
let buffer2 = buffer1.clone();  // O(1) - just Arc increment

// Both buffers reference the same memory
assert_eq!(buffer1.memory().as_ptr(), buffer2.memory().as_ptr());
```

## IPC Memory Sharing

### Sending Memory Across Processes

```rust
use parallax::memory::{SharedMemorySegment, IpcHandle};

// Process A: Create shared memory
let segment = SharedMemorySegment::new("pipeline", 1024 * 1024)?;

// Get IPC handle (contains file descriptor)
if let Some(IpcHandle::Fd { fd, size }) = segment.ipc_handle() {
    // Send fd to Process B via Unix socket (SCM_RIGHTS)
}

// Process B: Receive and map
let segment = SharedMemorySegment::from_raw_fd(received_fd, size)?;
// Now both processes share the same memory!
```

### IPC Protocol

```
Process A                                    Process B
    │                                            │
    │ 1. Create SharedMemorySegment              │
    │    (memfd_create)                          │
    │                                            │
    │ 2. Send fd via Unix socket ──────────────▶ │
    │    (SCM_RIGHTS)                            │
    │                                            │
    │                              3. mmap fd ◀──│
    │                                            │
    │ 4. Write buffer to segment                 │
    │    └─▶ Send buffer header ──────────────▶ │
    │        (offset, len, metadata)             │
    │                                            │
    │                              5. Read from  │
    │                                 shared mem │
    │                                 (no copy!) │
```

## Best Practices

### Choosing a Memory Backend

| Use Case | Recommended Backend |
|----------|---------------------|
| Single process, small buffers | `HeapSegment` |
| Single process, large buffers | `HugePageSegment` |
| Multi-process pipelines | `SharedMemorySegment` |
| Persistent data | `MappedFileSegment` |

### Pool Sizing

```rust
// For streaming: many small slots
let pool = MemoryPool::new(segment, 4096)?;  // 4KB slots

// For video frames: fewer large slots
let pool = MemoryPool::new(segment, 8 * 1024 * 1024)?;  // 8MB slots
```

**Guidelines:**
- Slot size should match typical buffer size
- More slots = more concurrency
- Total size = slot_size * expected_concurrent_buffers * 2

### Avoiding Copies

```rust
// Good: Pass buffer by reference
fn process_buffer(buffer: &Buffer) { ... }

// Good: Clone buffer (O(1), just Arc increment)
let buffer2 = buffer.clone();

// Avoid: Copying data out of buffer
let data = buffer.as_slice().to_vec();  // Allocates and copies!
```

## Troubleshooting

### Pool Exhaustion

```rust
match pool.loan() {
    Some(slot) => { /* use slot */ }
    None => {
        // Pool exhausted - all slots in use
        // Options:
        // 1. Wait and retry
        // 2. Create a larger pool
        // 3. Apply backpressure upstream
    }
}
```

### Huge Page Failures

```bash
# Check available huge pages
cat /proc/meminfo | grep Huge

# Allocate huge pages (requires root)
echo 512 > /proc/sys/vm/nr_hugepages
```

### Shared Memory Limits

```bash
# Check shared memory limits
cat /proc/sys/kernel/shmmax

# Increase if needed (requires root)
echo 1073741824 > /proc/sys/kernel/shmmax  # 1GB
```
