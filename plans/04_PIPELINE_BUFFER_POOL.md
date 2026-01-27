# Plan 04: Pipeline Buffer Pool

**Priority:** High (Short-term)  
**Effort:** Medium (3-5 days)  
**Dependencies:** None (can be done in parallel with others)  
**Addresses:** Architecture Issue 3.4 (No Pipeline-Level Buffer Pool)

---

## Problem Statement

Currently, each source element allocates its own buffers independently:

```rust
// Current: Every frame allocates new memory
let segment = Arc::new(HeapSegment::new(yuv_size).expect("alloc"));
```

This causes:
1. **Fragmentation:** Many small allocations throughout pipeline lifetime
2. **Allocation overhead:** malloc/free on every frame
3. **No memory bounds:** Pipeline can consume unlimited memory
4. **Poor cache locality:** Buffers scattered in memory

---

## Proposed Solution

> **Design Decision:** Per [Decision 6 in 00_DESIGN_DECISIONS.md](./00_DESIGN_DECISIONS.md), we adopt **pipeline-level buffer pools** similar to PipeWire's approach. This simplifies the API (elements don't negotiate pools) and enables automatic pool sizing based on caps negotiation.

Implement a pipeline-level buffer pool that elements can draw from, similar to GStreamer's buffer pools and PipeWire's buffer management.

### Design Goals

1. **Centralized allocation:** Pipeline owns the pool, elements borrow from it
2. **Pre-allocation:** Allocate buffers once at pipeline start
3. **Automatic return:** Buffers return to pool when dropped
4. **Size negotiation:** Pool size determined by caps negotiation
5. **Memory limits:** Configurable maximum memory usage
6. **Zero-copy compatible:** Works with existing memfd/shared memory

---

## Design

### BufferPool Trait

```rust
/// A pool of reusable buffers
pub trait BufferPool: Send + Sync {
    /// Acquire a buffer from the pool.
    /// Blocks if no buffers available (backpressure).
    fn acquire(&self) -> Result<PooledBuffer>;
    
    /// Try to acquire without blocking.
    /// Returns None if pool is exhausted.
    fn try_acquire(&self) -> Option<PooledBuffer>;
    
    /// Acquire with timeout.
    fn acquire_timeout(&self, timeout: Duration) -> Result<Option<PooledBuffer>>;
    
    /// Get pool statistics
    fn stats(&self) -> PoolStats;
    
    /// Get the buffer size this pool provides
    fn buffer_size(&self) -> usize;
    
    /// Get the number of buffers in the pool
    fn capacity(&self) -> usize;
}

#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total buffers in pool
    pub capacity: usize,
    /// Currently available buffers
    pub available: usize,
    /// Currently in-use buffers
    pub in_use: usize,
    /// Total acquisitions
    pub acquisitions: u64,
    /// Acquisitions that had to wait
    pub waits: u64,
}
```

### PooledBuffer

A buffer that automatically returns to pool on drop:

```rust
/// A buffer borrowed from a pool.
/// Returns to pool when dropped.
pub struct PooledBuffer {
    /// The underlying buffer
    inner: Buffer,
    /// Handle to return buffer to pool
    pool_handle: Arc<PoolInner>,
    /// Slot index in pool
    slot_index: usize,
}

impl PooledBuffer {
    /// Get mutable access to the buffer data
    pub fn data_mut(&mut self) -> &mut [u8] {
        self.inner.as_bytes_mut()
    }
    
    /// Get read access to the buffer data
    pub fn data(&self) -> &[u8] {
        self.inner.as_bytes()
    }
    
    /// Set the valid data length (for variable-size content)
    pub fn set_len(&mut self, len: usize) {
        self.inner.set_len(len);
    }
    
    /// Access metadata
    pub fn metadata(&self) -> &Metadata {
        self.inner.metadata()
    }
    
    pub fn metadata_mut(&mut self) -> &mut Metadata {
        self.inner.metadata_mut()
    }
    
    /// Convert to a regular Buffer (detaches from pool, won't return)
    pub fn into_buffer(mut self) -> Buffer {
        // Mark slot as permanently taken
        self.pool_handle.mark_detached(self.slot_index);
        std::mem::take(&mut self.inner)
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        // Return buffer to pool
        self.pool_handle.release(self.slot_index);
    }
}

// Allow using PooledBuffer where Buffer is expected
impl AsRef<Buffer> for PooledBuffer {
    fn as_ref(&self) -> &Buffer {
        &self.inner
    }
}
```

### FixedSizePool Implementation

```rust
/// A fixed-size buffer pool using CpuArena.
pub struct FixedSizePool {
    inner: Arc<PoolInner>,
}

struct PoolInner {
    /// The underlying arena
    arena: CpuArena,
    /// Availability bitmap
    available: AtomicBitmap,
    /// Condition variable for waiting
    notify: tokio::sync::Notify,
    /// Statistics
    stats: PoolStatsInner,
}

impl FixedSizePool {
    /// Create a new pool with given buffer size and count.
    pub fn new(buffer_size: usize, buffer_count: usize) -> Result<Self> {
        let arena = CpuArena::new(buffer_size, buffer_count)?;
        let available = AtomicBitmap::new(buffer_count);
        available.set_all();  // All buffers start available
        
        Ok(Self {
            inner: Arc::new(PoolInner {
                arena,
                available,
                notify: tokio::sync::Notify::new(),
                stats: PoolStatsInner::default(),
            }),
        })
    }
    
    /// Create pool from existing SharedArena (for IPC)
    pub fn from_shared_arena(arena: SharedArena) -> Self {
        // ...
    }
}

impl BufferPool for FixedSizePool {
    fn acquire(&self) -> Result<PooledBuffer> {
        loop {
            if let Some(buffer) = self.try_acquire() {
                return Ok(buffer);
            }
            
            // Wait for a buffer to be released
            self.inner.stats.waits.fetch_add(1, Ordering::Relaxed);
            
            // Use async notification in async context
            // For sync context, use parking_lot condvar
            std::thread::yield_now();
        }
    }
    
    fn try_acquire(&self) -> Option<PooledBuffer> {
        // Find first available slot
        let slot = self.inner.available.find_and_clear_first()?;
        
        self.inner.stats.acquisitions.fetch_add(1, Ordering::Relaxed);
        
        // Get buffer from arena
        let segment = self.inner.arena.get_segment(slot);
        let handle = MemoryHandle::from_segment(segment);
        let buffer = Buffer::new(handle, Metadata::default());
        
        Some(PooledBuffer {
            inner: buffer,
            pool_handle: self.inner.clone(),
            slot_index: slot,
        })
    }
    
    fn buffer_size(&self) -> usize {
        self.inner.arena.slot_size()
    }
    
    fn capacity(&self) -> usize {
        self.inner.arena.slot_count()
    }
}

impl PoolInner {
    fn release(&self, slot: usize) {
        // Mark slot as available
        self.available.set(slot);
        // Wake any waiters
        self.notify.notify_one();
    }
    
    fn mark_detached(&self, slot: usize) {
        // Slot is permanently taken, don't return to pool
        // Could track this for debugging
    }
}
```

### Integration with ProduceContext

Update `ProduceContext` to provide pooled buffers:

```rust
pub struct ProduceContext<'a> {
    /// Pool to allocate from (if available)
    pool: Option<&'a dyn BufferPool>,
    /// Pre-allocated buffer (for pool-aware sources)
    buffer: Option<PooledBuffer>,
    /// Output slot for non-pool sources
    output: &'a mut [u8],
    /// Metadata
    metadata: &'a mut Metadata,
}

impl<'a> ProduceContext<'a> {
    /// Get a buffer from the pool.
    /// Preferred method for pool-aware sources.
    pub fn acquire_buffer(&mut self) -> Result<PooledBuffer> {
        match &self.pool {
            Some(pool) => pool.acquire(),
            None => Err(Error::NoPool),
        }
    }
    
    /// Get the pre-allocated output buffer.
    /// Used when pool has already allocated.
    pub fn output(&mut self) -> &mut [u8] {
        self.output
    }
}
```

### Pipeline Integration

```rust
impl Pipeline {
    /// Configure the buffer pool for this pipeline.
    pub fn set_pool(&mut self, pool: Arc<dyn BufferPool>) {
        self.pool = Some(pool);
    }
    
    /// Create and configure a pool based on negotiated caps.
    pub fn create_pool(&mut self, buffer_count: usize) -> Result<()> {
        // Determine max buffer size from caps
        let max_size = self.negotiate_buffer_size()?;
        
        let pool = FixedSizePool::new(max_size, buffer_count)?;
        self.pool = Some(Arc::new(pool));
        
        Ok(())
    }
    
    fn negotiate_buffer_size(&self) -> Result<usize> {
        // Find maximum buffer size across all links
        let mut max_size = 0;
        
        for link in self.links() {
            if let Some(format) = &link.negotiated_format {
                let size = format.buffer_size();
                max_size = max_size.max(size);
            }
        }
        
        if max_size == 0 {
            // Default for unknown formats
            max_size = 4 * 1024 * 1024;  // 4MB
        }
        
        Ok(max_size)
    }
}
```

### Executor Integration

```rust
// In executor, create context with pool
async fn run_source_node(
    source: &mut dyn Source,
    pool: Option<&dyn BufferPool>,
    output: Sender<Buffer>,
) -> Result<()> {
    loop {
        let buffer = if let Some(pool) = pool {
            // Pool-aware path
            let mut pooled = pool.acquire()?;
            
            let ctx = ProduceContext::with_pooled(&mut pooled);
            match source.produce(&mut ctx)? {
                ProduceResult::Produced(len) => {
                    pooled.set_len(len);
                    pooled.into_buffer()  // Detach from pool for downstream
                }
                ProduceResult::Eos => break,
                ProduceResult::WouldBlock => continue,
                ProduceResult::OwnBuffer(buf) => buf,
            }
        } else {
            // Legacy non-pool path
            // ...
        };
        
        output.send(buffer).await?;
    }
    Ok(())
}
```

---

## Pool Sizing Strategy

### Fixed Size Pool

Simple, predictable memory usage:

```rust
let pool = FixedSizePool::new(
    1920 * 1080 * 3 / 2,  // YUV420 1080p frame
    10,                    // 10 buffers
)?;
```

### Dynamic Pool

Grows up to a limit:

```rust
pub struct DynamicPool {
    fixed_pools: Vec<FixedSizePool>,
    max_memory: usize,
    current_memory: AtomicUsize,
}

impl DynamicPool {
    fn acquire(&self) -> Result<PooledBuffer> {
        // Try existing pools first
        for pool in &self.fixed_pools {
            if let Some(buf) = pool.try_acquire() {
                return Ok(buf);
            }
        }
        
        // Allocate new if under limit
        if self.current_memory.load(Ordering::Relaxed) < self.max_memory {
            // Create new buffer
        }
        
        // Otherwise wait
        // ...
    }
}
```

### Per-Link Pools

Different sizes for different stages:

```rust
// Pipeline automatically creates pools per link
pipeline.prepare()?;  // Negotiates, creates pools

// Or manually:
pipeline.set_link_pool(encoder_link, small_pool)?;
pipeline.set_link_pool(decoder_link, large_pool)?;
```

---

## Implementation Steps

### Step 1: Define BufferPool Trait

**File:** `src/memory/pool.rs`

- `BufferPool` trait
- `PooledBuffer` struct with Drop impl
- `PoolStats` struct

### Step 2: Implement FixedSizePool

**File:** `src/memory/fixed_pool.rs`

- Use `CpuArena` as backing storage
- `AtomicBitmap` for availability tracking
- Async-compatible waiting

### Step 3: Update ProduceContext

**File:** `src/element/context.rs`

- Add pool reference
- Add `acquire_buffer()` method

### Step 4: Pipeline Integration

**File:** `src/pipeline/graph.rs`

- Add pool field to Pipeline
- `set_pool()` and `create_pool()` methods
- Buffer size negotiation

### Step 5: Executor Integration

**File:** `src/pipeline/unified_executor.rs`

- Pass pool to source nodes
- Handle pooled vs non-pooled paths

### Step 6: Update Examples

**File:** `examples/31_av1_pipeline_stanag.rs`

```rust
// Before:
let segment = Arc::new(HeapSegment::new(yuv_size).expect("alloc"));

// After:
let pool = FixedSizePool::new(yuv_size, 10)?;
let mut buffer = pool.acquire()?;
buffer.data_mut().copy_from_slice(&yuv_data);
```

### Step 7: Add Tests

```rust
#[test]
fn test_pool_acquire_release() {
    let pool = FixedSizePool::new(1024, 2).unwrap();
    
    let buf1 = pool.acquire().unwrap();
    let buf2 = pool.acquire().unwrap();
    
    assert!(pool.try_acquire().is_none());  // Pool exhausted
    
    drop(buf1);  // Return to pool
    
    let buf3 = pool.try_acquire();
    assert!(buf3.is_some());  // Buffer available again
}

#[test]
fn test_pool_backpressure() {
    let pool = FixedSizePool::new(1024, 1).unwrap();
    
    let buf = pool.acquire().unwrap();
    
    // Spawn task that will wait
    let pool2 = pool.clone();
    let handle = std::thread::spawn(move || {
        pool2.acquire()  // Will block
    });
    
    std::thread::sleep(Duration::from_millis(100));
    drop(buf);  // Release
    
    handle.join().unwrap().unwrap();  // Should succeed
}
```

---

## Memory Layout Options

### Option A: Arena-per-Pool (Current CpuArena)

```
Pool Memory:
┌────────────────────────────────────────────┐
│ Slot 0: [buffer data 1MB]                  │
├────────────────────────────────────────────┤
│ Slot 1: [buffer data 1MB]                  │
├────────────────────────────────────────────┤
│ Slot 2: [buffer data 1MB]                  │
└────────────────────────────────────────────┘
```

### Option B: Ring Buffer Pool

For streaming with predictable timing:

```
Ring Buffer:
┌──────────────────────────────────────────────────┐
│ Write ──►  [frame N][frame N+1][frame N+2] ──► Read │
└──────────────────────────────────────────────────┘
```

### Option C: Hierarchical Pools

For different buffer sizes:

```
Pool Hierarchy:
┌─────────────────────────────────────────────────┐
│ Small Pool (4KB): metadata, control messages     │
├─────────────────────────────────────────────────┤
│ Medium Pool (64KB): audio frames, small video    │
├─────────────────────────────────────────────────┤
│ Large Pool (4MB): video frames, large data       │
└─────────────────────────────────────────────────┘
```

**Recommendation:** Start with Option A (Arena-per-Pool), add Option C later if needed.

---

## Validation Criteria

- [ ] `BufferPool` trait defined
- [ ] `FixedSizePool` implements `BufferPool`
- [ ] `PooledBuffer` returns to pool on drop
- [ ] `ProduceContext` supports pool acquisition
- [ ] Pipeline can create/configure pool
- [ ] Executor passes pool to sources
- [ ] Backpressure works (blocks when exhausted)
- [ ] Statistics tracking works
- [ ] All existing tests pass

---

## Future Enhancements

1. **Shared memory pools:** For IPC, pool backed by `SharedArena`
2. **GPU buffer pools:** Vulkan/CUDA memory allocation
3. **Pool resizing:** Grow/shrink based on usage patterns
4. **Pool sharing:** Multiple pipelines share a pool
5. **Pool metrics:** Prometheus/metrics-rs integration

---

## Files to Create/Modify

| File | Changes |
|------|---------|
| `src/memory/pool.rs` | New: BufferPool trait, PooledBuffer |
| `src/memory/fixed_pool.rs` | New: FixedSizePool implementation |
| `src/memory/mod.rs` | Export new types |
| `src/element/context.rs` | Add pool to ProduceContext |
| `src/pipeline/graph.rs` | Add pool management |
| `src/pipeline/unified_executor.rs` | Pool-aware execution |
| `examples/34_buffer_pool.rs` | New example |
