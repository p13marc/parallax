//! Pipeline-level buffer pool for efficient buffer allocation.
//!
//! This module provides a high-level buffer pool abstraction that:
//! - Pre-allocates buffers at pipeline start
//! - Automatically returns buffers to the pool when dropped
//! - Provides backpressure (blocks when pool is exhausted)
//! - Tracks statistics for monitoring
//!
//! # Design
//!
//! The buffer pool sits at the pipeline level, providing buffers to sources.
//! This is more efficient than per-element allocation because:
//! - Buffers are reused instead of allocated/freed each frame
//! - Memory usage is bounded and predictable
//! - Cache locality is improved (same memory reused)
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::memory::{BufferPool, FixedBufferPool};
//!
//! // Create a pool with 10 buffers of 1MB each
//! let pool = FixedBufferPool::new(1024 * 1024, 10)?;
//!
//! // Acquire a buffer (blocks if none available)
//! let mut buffer = pool.acquire()?;
//!
//! // Write data
//! buffer.data_mut()[..5].copy_from_slice(b"hello");
//! buffer.set_len(5);
//!
//! // Buffer returns to pool when dropped
//! drop(buffer);
//! ```

use crate::buffer::{Buffer, MemoryHandle};
use crate::error::Result;
use crate::memory::{ArenaSlot, CpuArena};
use crate::metadata::Metadata;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

// ============================================================================
// BufferPool Trait
// ============================================================================

/// A pool of reusable buffers.
///
/// Buffer pools pre-allocate memory and provide buffers to elements.
/// When a `PooledBuffer` is dropped, it automatically returns to the pool.
///
/// # Backpressure
///
/// When all buffers are in use, `acquire()` blocks until one is returned.
/// This provides natural backpressure in the pipeline - fast producers
/// wait for slow consumers.
pub trait BufferPool: Send + Sync {
    /// Acquire a buffer from the pool.
    ///
    /// Blocks if no buffers are available (backpressure).
    fn acquire(&self) -> Result<PooledBuffer>;

    /// Try to acquire a buffer without blocking.
    ///
    /// Returns `None` if the pool is exhausted.
    fn try_acquire(&self) -> Option<PooledBuffer>;

    /// Acquire with a timeout.
    ///
    /// Returns `None` if the timeout expires before a buffer is available.
    fn acquire_timeout(&self, timeout: Duration) -> Result<Option<PooledBuffer>>;

    /// Get pool statistics.
    fn stats(&self) -> PoolStats;

    /// Get the buffer size this pool provides.
    fn buffer_size(&self) -> usize;

    /// Get the total number of buffers in the pool.
    fn capacity(&self) -> usize;

    /// Get the number of currently available buffers.
    fn available(&self) -> usize;
}

/// Statistics about pool usage.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total buffers in the pool.
    pub capacity: usize,
    /// Currently available buffers.
    pub available: usize,
    /// Currently in-use buffers.
    pub in_use: usize,
    /// Total number of acquisitions.
    pub acquisitions: u64,
    /// Acquisitions that had to wait for a buffer.
    pub waits: u64,
}

// ============================================================================
// PooledBuffer
// ============================================================================

/// A buffer borrowed from a pool.
///
/// This wraps a `Buffer` and automatically returns it to the pool when dropped.
/// Use `into_buffer()` to detach the buffer from the pool (it won't return).
///
/// # Example
///
/// ```rust,ignore
/// let mut pooled = pool.acquire()?;
///
/// // Write data
/// pooled.data_mut()[..n].copy_from_slice(&data);
/// pooled.set_len(n);
///
/// // Set metadata
/// pooled.metadata_mut().sequence = seq;
///
/// // Convert to Buffer for downstream (detaches from pool)
/// let buffer = pooled.into_buffer();
/// ```
pub struct PooledBuffer {
    /// The underlying arena slot (will be returned to pool on drop).
    slot: Option<ArenaSlot>,
    /// Length of valid data in the buffer.
    len: usize,
    /// Buffer metadata.
    metadata: Metadata,
    /// Pool inner state for tracking and notification.
    pool_inner: Arc<PoolInner>,
}

impl PooledBuffer {
    /// Create a new pooled buffer from an arena slot.
    fn new(slot: ArenaSlot, pool_inner: Arc<PoolInner>) -> Self {
        let len = slot.len();
        Self {
            slot: Some(slot),
            len,
            metadata: Metadata::new(),
            pool_inner,
        }
    }

    /// Get the buffer data as a byte slice.
    #[inline]
    pub fn data(&self) -> &[u8] {
        match &self.slot {
            Some(slot) => &slot.data()[..self.len],
            None => &[],
        }
    }

    /// Get the buffer data as a mutable byte slice.
    #[inline]
    pub fn data_mut(&mut self) -> &mut [u8] {
        match &mut self.slot {
            Some(slot) => {
                let cap = slot.len();
                &mut slot.data_mut()[..cap]
            }
            None => &mut [],
        }
    }

    /// Get the current length of valid data.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the buffer capacity (total slot size).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.slot.as_ref().map(|s| s.len()).unwrap_or(0)
    }

    /// Set the length of valid data.
    ///
    /// # Panics
    ///
    /// Panics if `len > capacity()`.
    #[inline]
    pub fn set_len(&mut self, len: usize) {
        assert!(len <= self.capacity(), "length exceeds buffer capacity");
        self.len = len;
    }

    /// Get a reference to the metadata.
    #[inline]
    pub fn metadata(&self) -> &Metadata {
        &self.metadata
    }

    /// Get a mutable reference to the metadata.
    #[inline]
    pub fn metadata_mut(&mut self) -> &mut Metadata {
        &mut self.metadata
    }

    /// Convert to a regular Buffer, detaching from the pool.
    ///
    /// The buffer will NOT return to the pool when dropped.
    /// Use this when passing the buffer downstream.
    pub fn into_buffer(mut self) -> Buffer {
        let slot = self.slot.take().expect("PooledBuffer already consumed");
        let handle = MemoryHandle::from_arena_slot_with_len(slot, self.len);
        // Don't decrement in_use - the buffer is detached
        self.pool_inner
            .stats
            .detached
            .fetch_add(1, Ordering::Relaxed);
        Buffer::new(handle, std::mem::take(&mut self.metadata))
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if self.slot.is_some() {
            // Slot will be returned to arena when dropped
            self.pool_inner.stats.in_use.fetch_sub(1, Ordering::Relaxed);
            // Notify waiters that a buffer is available
            self.pool_inner.notify.notify_one();
        }
        // If slot is None, it was detached via into_buffer()
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("len", &self.len)
            .field("capacity", &self.capacity())
            .field("has_slot", &self.slot.is_some())
            .finish()
    }
}

// ============================================================================
// FixedBufferPool
// ============================================================================

/// A fixed-size buffer pool using `CpuArena`.
///
/// This is the primary buffer pool implementation. It allocates a single
/// large memory region (via memfd) and subdivides it into fixed-size slots.
///
/// # Features
///
/// - Pre-allocated memory (no allocation during processing)
/// - Lock-free slot acquisition
/// - Automatic return on drop
/// - Backpressure when exhausted
///
/// # Example
///
/// ```rust,ignore
/// // Create pool for 10 buffers of 1MB each
/// let pool = FixedBufferPool::new(1024 * 1024, 10)?;
///
/// // Acquire buffers
/// let buf1 = pool.acquire()?;
/// let buf2 = pool.try_acquire(); // Non-blocking
/// ```
pub struct FixedBufferPool {
    /// Shared inner state (shared with PooledBuffers for notification).
    inner: Arc<PoolInner>,
}

/// Shared pool state (referenced by both pool and pooled buffers).
struct PoolInner {
    /// The underlying arena.
    arena: Arc<CpuArena>,
    /// Statistics tracking.
    stats: PoolStatsInner,
    /// Notification for waiters.
    notify: std::sync::Condvar,
    /// Mutex for condition variable.
    mutex: std::sync::Mutex<()>,
}

/// Internal statistics tracking.
struct PoolStatsInner {
    /// Currently in-use buffers.
    in_use: AtomicUsize,
    /// Total acquisitions.
    acquisitions: AtomicU64,
    /// Acquisitions that waited.
    waits: AtomicU64,
    /// Buffers detached via into_buffer().
    detached: AtomicU64,
}

impl FixedBufferPool {
    /// Create a new fixed-size buffer pool.
    ///
    /// # Arguments
    ///
    /// * `buffer_size` - Size of each buffer in bytes.
    /// * `buffer_count` - Number of buffers in the pool.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// // Pool for 1080p YUV420 frames
    /// let frame_size = 1920 * 1080 * 3 / 2;
    /// let pool = FixedBufferPool::new(frame_size, 10)?;
    /// ```
    pub fn new(buffer_size: usize, buffer_count: usize) -> Result<Arc<Self>> {
        Self::with_name("parallax-buffer-pool", buffer_size, buffer_count)
    }

    /// Create a pool with a debug name.
    pub fn with_name(name: &str, buffer_size: usize, buffer_count: usize) -> Result<Arc<Self>> {
        let arena = CpuArena::with_name(name, buffer_size, buffer_count)?;

        Ok(Arc::new(Self {
            inner: Arc::new(PoolInner {
                arena,
                stats: PoolStatsInner {
                    in_use: AtomicUsize::new(0),
                    acquisitions: AtomicU64::new(0),
                    waits: AtomicU64::new(0),
                    detached: AtomicU64::new(0),
                },
                notify: std::sync::Condvar::new(),
                mutex: std::sync::Mutex::new(()),
            }),
        }))
    }

    /// Pre-fault all memory to avoid page faults during processing.
    ///
    /// Call this after creating the pool to ensure all pages are
    /// resident in memory.
    pub fn prefault(&self) {
        self.inner.arena.prefault();
    }

    /// Get the underlying arena (for advanced use cases).
    pub fn arena(&self) -> &Arc<CpuArena> {
        &self.inner.arena
    }
}

impl BufferPool for FixedBufferPool {
    fn acquire(&self) -> Result<PooledBuffer> {
        // Fast path: try to acquire without waiting
        if let Some(buffer) = self.try_acquire() {
            return Ok(buffer);
        }

        // Slow path: wait for a buffer
        self.inner.stats.waits.fetch_add(1, Ordering::Relaxed);

        let mut guard = self.inner.mutex.lock().unwrap();
        loop {
            if let Some(buffer) = self.try_acquire() {
                return Ok(buffer);
            }

            // Wait for notification (with spurious wakeup handling)
            guard = self.inner.notify.wait(guard).unwrap();
        }
    }

    fn try_acquire(&self) -> Option<PooledBuffer> {
        let slot = self.inner.arena.acquire()?;

        self.inner
            .stats
            .acquisitions
            .fetch_add(1, Ordering::Relaxed);
        self.inner.stats.in_use.fetch_add(1, Ordering::Relaxed);

        Some(PooledBuffer::new(slot, self.inner.clone()))
    }

    fn acquire_timeout(&self, timeout: Duration) -> Result<Option<PooledBuffer>> {
        // Fast path
        if let Some(buffer) = self.try_acquire() {
            return Ok(Some(buffer));
        }

        // Slow path with timeout
        self.inner.stats.waits.fetch_add(1, Ordering::Relaxed);

        let mut guard = self.inner.mutex.lock().unwrap();
        let deadline = std::time::Instant::now() + timeout;

        loop {
            if let Some(buffer) = self.try_acquire() {
                return Ok(Some(buffer));
            }

            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }

            let (new_guard, timeout_result) =
                self.inner.notify.wait_timeout(guard, remaining).unwrap();
            guard = new_guard;

            if timeout_result.timed_out() {
                // Try one more time
                return Ok(self.try_acquire());
            }
        }
    }

    fn stats(&self) -> PoolStats {
        let in_use = self.inner.stats.in_use.load(Ordering::Relaxed);
        let capacity = self.inner.arena.slot_count();

        PoolStats {
            capacity,
            available: capacity.saturating_sub(in_use),
            in_use,
            acquisitions: self.inner.stats.acquisitions.load(Ordering::Relaxed),
            waits: self.inner.stats.waits.load(Ordering::Relaxed),
        }
    }

    fn buffer_size(&self) -> usize {
        self.inner.arena.slot_size()
    }

    fn capacity(&self) -> usize {
        self.inner.arena.slot_count()
    }

    fn available(&self) -> usize {
        self.inner.arena.free_count()
    }
}

// Implement Drop to wake any waiting threads when pool is destroyed
impl Drop for FixedBufferPool {
    fn drop(&mut self) {
        self.inner.notify.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;
    use std::time::Duration;

    #[test]
    fn test_pool_creation() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();

        assert_eq!(pool.buffer_size(), 1024);
        assert_eq!(pool.capacity(), 4);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_pool_acquire_release() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();

        assert_eq!(pool.available(), 4);

        {
            let _buf1 = pool.acquire().unwrap();
            assert_eq!(pool.available(), 3);

            let _buf2 = pool.acquire().unwrap();
            assert_eq!(pool.available(), 2);
        }

        // Buffers should be returned
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_pool_exhaustion_and_try_acquire() {
        let pool = FixedBufferPool::new(1024, 2).unwrap();

        let _buf1 = pool.try_acquire().unwrap();
        let _buf2 = pool.try_acquire().unwrap();

        // Pool is exhausted
        assert!(pool.try_acquire().is_none());
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_pooled_buffer_data() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();

        let mut buf = pool.acquire().unwrap();
        assert_eq!(buf.capacity(), 1024);
        assert_eq!(buf.len(), 1024); // Initially full capacity

        // Write data
        buf.data_mut()[..5].copy_from_slice(b"hello");
        buf.set_len(5);

        assert_eq!(buf.len(), 5);
        assert_eq!(buf.data(), b"hello");
    }

    #[test]
    fn test_pooled_buffer_metadata() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();

        let mut buf = pool.acquire().unwrap();
        buf.metadata_mut().sequence = 42;
        buf.metadata_mut().set("app/test", "value".to_string());

        assert_eq!(buf.metadata().sequence, 42);
        assert_eq!(
            buf.metadata().get::<String>("app/test"),
            Some(&"value".to_string())
        );
    }

    #[test]
    fn test_pooled_buffer_into_buffer() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();

        let mut pooled = pool.acquire().unwrap();
        pooled.data_mut()[..5].copy_from_slice(b"hello");
        pooled.set_len(5);
        pooled.metadata_mut().sequence = 42;

        // Convert to Buffer
        let buffer = pooled.into_buffer();

        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_bytes(), b"hello");
        assert_eq!(buffer.metadata().sequence, 42);

        // Pool still has one slot used (detached, not returned)
        // The slot will be freed when buffer is dropped
    }

    #[test]
    fn test_pool_stats() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();

        let stats = pool.stats();
        assert_eq!(stats.capacity, 4);
        assert_eq!(stats.available, 4);
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.acquisitions, 0);

        {
            let _buf = pool.acquire().unwrap();
            let stats = pool.stats();
            assert_eq!(stats.in_use, 1);
            assert_eq!(stats.acquisitions, 1);
        }

        let stats = pool.stats();
        assert_eq!(stats.in_use, 0);
        assert_eq!(stats.acquisitions, 1);
    }

    #[test]
    fn test_pool_timeout() {
        let pool = FixedBufferPool::new(1024, 1).unwrap();

        let _buf = pool.acquire().unwrap();

        // Should timeout
        let result = pool.acquire_timeout(Duration::from_millis(10)).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_pool_backpressure() {
        let pool = FixedBufferPool::new(1024, 1).unwrap();

        let buf = pool.acquire().unwrap();

        // Spawn thread that will wait for buffer
        let pool2 = pool.clone();
        let handle = thread::spawn(move || {
            let start = std::time::Instant::now();
            let _buf = pool2.acquire().unwrap();
            start.elapsed()
        });

        // Wait a bit, then release
        thread::sleep(Duration::from_millis(50));
        drop(buf);

        let elapsed = handle.join().unwrap();
        assert!(elapsed >= Duration::from_millis(40));
    }

    #[test]
    fn test_pool_concurrent() {
        let pool = FixedBufferPool::new(1024, 8).unwrap();
        let pool = Arc::new(pool);

        let mut handles = vec![];

        for _ in 0..4 {
            let pool = pool.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..100 {
                    let mut buf = pool.acquire().unwrap();
                    buf.data_mut()[0] = 42;
                    buf.set_len(1);
                    thread::sleep(Duration::from_micros(10));
                }
            }));
        }

        for handle in handles {
            handle.join().unwrap();
        }

        // All buffers should be returned
        assert_eq!(pool.available(), 8);
    }

    #[test]
    #[should_panic(expected = "length exceeds buffer capacity")]
    fn test_pooled_buffer_set_len_overflow() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();
        let mut buf = pool.acquire().unwrap();
        buf.set_len(2048); // Should panic
    }
}
