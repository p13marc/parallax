//! Lock-free bridge for async↔RT thread communication.
//!
//! This module provides [`AsyncRtBridge`], a lock-free SPSC ring buffer with
//! eventfd-based signaling for efficient communication between Tokio async
//! tasks and real-time data threads.
//!
//! # Design
//!
//! The bridge uses:
//! - A lock-free SPSC ring buffer for data transfer
//! - eventfd for efficient signaling (Linux-specific)
//! - Async-aware waiting on the Tokio side
//! - Non-blocking operations on the RT side
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::rt_bridge::{AsyncRtBridge, BridgeConfig};
//! use parallax::buffer::Buffer;
//!
//! // Create a bridge with 16 slots
//! let bridge = AsyncRtBridge::new(BridgeConfig {
//!     capacity: 16,
//!     ..Default::default()
//! })?;
//!
//! // Async producer (Tokio task)
//! bridge.push_async(buffer).await?;
//!
//! // RT consumer (data thread) - non-blocking!
//! if let Some(buffer) = bridge.try_pop() {
//!     // Process buffer...
//! }
//! ```

use crate::buffer::Buffer;
use crate::error::{Error, Result};
use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

#[cfg(target_os = "linux")]
use rustix::event::{EventfdFlags, eventfd};
#[cfg(target_os = "linux")]
use rustix::fd::OwnedFd;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for an [`AsyncRtBridge`].
#[derive(Debug, Clone)]
pub struct BridgeConfig {
    /// Number of buffer slots in the ring buffer.
    ///
    /// Must be a power of 2. Default is 16.
    pub capacity: usize,

    /// Name for debugging/logging purposes.
    pub name: String,
}

impl Default for BridgeConfig {
    fn default() -> Self {
        Self {
            capacity: 16,
            name: String::from("bridge"),
        }
    }
}

impl BridgeConfig {
    /// Create a new config with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity,
            ..Default::default()
        }
    }

    /// Set the bridge name.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

// ============================================================================
// EventFd Wrapper
// ============================================================================

/// Cross-platform eventfd abstraction.
///
/// On Linux, this uses the native eventfd syscall.
/// On other platforms, this falls back to atomics with polling.
#[cfg(target_os = "linux")]
pub struct EventFd {
    fd: OwnedFd,
}

#[cfg(target_os = "linux")]
impl EventFd {
    /// Create a new eventfd with initial value 0.
    pub fn new() -> Result<Self> {
        let fd = eventfd(0, EventfdFlags::NONBLOCK | EventfdFlags::CLOEXEC)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eventfd: {}", e))))?;
        Ok(Self { fd })
    }

    /// Signal the eventfd (increment counter).
    ///
    /// This is safe to call from any thread, including RT threads.
    pub fn notify(&self) -> Result<()> {
        let val: u64 = 1;
        let bytes = val.to_ne_bytes();
        rustix::io::write(&self.fd, &bytes)
            .map_err(|e| Error::Io(std::io::Error::other(format!("eventfd write: {}", e))))?;
        Ok(())
    }

    /// Try to read from the eventfd (non-blocking).
    ///
    /// Returns `true` if the eventfd was signaled, `false` if it would block.
    pub fn try_wait(&self) -> Result<bool> {
        let mut buf = [0u8; 8];
        match rustix::io::read(&self.fd, &mut buf) {
            Ok(8) => Ok(true),
            Ok(_) => Ok(false),
            Err(rustix::io::Errno::WOULDBLOCK) => Ok(false),
            Err(e) => Err(Error::Io(std::io::Error::other(format!(
                "eventfd read: {}",
                e
            )))),
        }
    }

    /// Wait asynchronously for the eventfd to be signaled.
    ///
    /// This uses tokio's async file descriptor support.
    pub async fn wait_async(&self) -> Result<()> {
        use std::os::unix::io::AsFd;
        use tokio::io::Interest;
        use tokio::io::unix::AsyncFd;

        let async_fd =
            AsyncFd::with_interest(self.fd.as_fd(), Interest::READABLE).map_err(Error::Io)?;

        loop {
            let mut guard = async_fd.readable().await.map_err(Error::Io)?;

            match self.try_wait() {
                Ok(true) => return Ok(()),
                Ok(false) => {
                    guard.clear_ready();
                    continue;
                }
                Err(e) => return Err(e),
            }
        }
    }

    /// Get the raw file descriptor for use with epoll/select.
    #[cfg(target_os = "linux")]
    pub fn as_raw_fd(&self) -> std::os::unix::io::RawFd {
        use std::os::unix::io::AsRawFd;
        self.fd.as_raw_fd()
    }
}

/// Fallback EventFd for non-Linux platforms using atomics.
#[cfg(not(target_os = "linux"))]
pub struct EventFd {
    counter: AtomicU64,
}

#[cfg(not(target_os = "linux"))]
impl EventFd {
    pub fn new() -> Result<Self> {
        Ok(Self {
            counter: AtomicU64::new(0),
        })
    }

    pub fn notify(&self) -> Result<()> {
        self.counter.fetch_add(1, Ordering::Release);
        Ok(())
    }

    pub fn try_wait(&self) -> Result<bool> {
        loop {
            let val = self.counter.load(Ordering::Acquire);
            if val == 0 {
                return Ok(false);
            }
            if self
                .counter
                .compare_exchange(val, val - 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(true);
            }
        }
    }

    pub async fn wait_async(&self) -> Result<()> {
        // Polling fallback for non-Linux
        loop {
            if self.try_wait()? {
                return Ok(());
            }
            tokio::time::sleep(std::time::Duration::from_micros(100)).await;
        }
    }
}

// ============================================================================
// Ring Buffer Slot
// ============================================================================

/// A slot in the ring buffer.
struct Slot {
    /// The buffer stored in this slot.
    buffer: UnsafeCell<Option<Buffer>>,
}

impl Slot {
    fn new() -> Self {
        Self {
            buffer: UnsafeCell::new(None),
        }
    }
}

// SAFETY: Slot access is guarded by the ring buffer's head/tail atomics
unsafe impl Send for Slot {}
unsafe impl Sync for Slot {}

// ============================================================================
// Lock-Free SPSC Ring Buffer
// ============================================================================

/// Cache line size for padding.
const CACHE_LINE: usize = 64;

/// Padded atomic for avoiding false sharing.
#[repr(C)]
struct PaddedAtomicUsize {
    value: AtomicUsize,
    _padding: [u8; CACHE_LINE - std::mem::size_of::<AtomicUsize>()],
}

impl PaddedAtomicUsize {
    fn new(val: usize) -> Self {
        Self {
            value: AtomicUsize::new(val),
            _padding: [0; CACHE_LINE - std::mem::size_of::<AtomicUsize>()],
        }
    }

    fn load(&self, order: Ordering) -> usize {
        self.value.load(order)
    }

    fn store(&self, val: usize, order: Ordering) {
        self.value.store(val, order)
    }
}

/// A lock-free single-producer single-consumer ring buffer.
struct RingBuffer {
    /// Buffer slots.
    slots: Box<[Slot]>,

    /// Mask for index wrapping (capacity - 1, since capacity is power of 2).
    mask: usize,

    /// Write position (only modified by producer).
    head: PaddedAtomicUsize,

    /// Read position (only modified by consumer).
    tail: PaddedAtomicUsize,
}

impl RingBuffer {
    /// Create a new ring buffer with the given capacity.
    ///
    /// Capacity must be a power of 2.
    fn new(capacity: usize) -> Result<Self> {
        if !capacity.is_power_of_two() {
            return Err(Error::InvalidSegment(
                "Ring buffer capacity must be a power of 2".into(),
            ));
        }

        let slots: Vec<Slot> = (0..capacity).map(|_| Slot::new()).collect();

        Ok(Self {
            slots: slots.into_boxed_slice(),
            mask: capacity - 1,
            head: PaddedAtomicUsize::new(0),
            tail: PaddedAtomicUsize::new(0),
        })
    }

    /// Get the capacity.
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Check if the buffer is empty.
    fn is_empty(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head == tail
    }

    /// Check if the buffer is full.
    fn is_full(&self) -> bool {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail) >= self.capacity()
    }

    /// Get the number of items in the buffer.
    fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    /// Try to push a buffer (producer side).
    ///
    /// Returns `Err(buffer)` if the ring is full.
    fn try_push(&self, buffer: Buffer) -> std::result::Result<(), Buffer> {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Acquire);

        // Check if full
        if head.wrapping_sub(tail) >= self.capacity() {
            return Err(buffer);
        }

        // Write to slot
        let slot_idx = head & self.mask;
        // SAFETY: We're the only producer, and we've verified the slot is empty
        unsafe {
            *self.slots[slot_idx].buffer.get() = Some(buffer);
        }

        // Publish the write
        self.head.store(head.wrapping_add(1), Ordering::Release);

        Ok(())
    }

    /// Try to pop a buffer (consumer side).
    ///
    /// Returns `None` if the ring is empty.
    fn try_pop(&self) -> Option<Buffer> {
        let tail = self.tail.load(Ordering::Relaxed);
        let head = self.head.load(Ordering::Acquire);

        // Check if empty
        if head == tail {
            return None;
        }

        // Read from slot
        let slot_idx = tail & self.mask;
        // SAFETY: We're the only consumer, and we've verified the slot has data
        let buffer = unsafe { (*self.slots[slot_idx].buffer.get()).take() };

        // Publish the read
        self.tail.store(tail.wrapping_add(1), Ordering::Release);

        buffer
    }
}

// ============================================================================
// Async-RT Bridge
// ============================================================================

/// A lock-free bridge for communication between async and RT threads.
///
/// This bridge provides:
/// - Lock-free SPSC ring buffer for data transfer
/// - eventfd-based signaling for efficient wakeups
/// - Async-aware push on the Tokio side
/// - Non-blocking pop on the RT side
///
/// # Thread Safety
///
/// The bridge is designed for single-producer single-consumer use:
/// - One async task pushes buffers
/// - One RT thread pops buffers
///
/// Multiple producers or consumers require external synchronization.
pub struct AsyncRtBridge {
    /// The underlying ring buffer.
    ring: RingBuffer,

    /// Eventfd to signal data available (async → RT).
    data_available: EventFd,

    /// Eventfd to signal space available (RT → async).
    space_available: EventFd,

    /// Configuration.
    config: BridgeConfig,
}

impl AsyncRtBridge {
    /// Create a new bridge with the given configuration.
    pub fn new(config: BridgeConfig) -> Result<Self> {
        // Ensure capacity is power of 2
        let capacity = config.capacity.next_power_of_two();

        Ok(Self {
            ring: RingBuffer::new(capacity)?,
            data_available: EventFd::new()?,
            space_available: EventFd::new()?,
            config,
        })
    }

    /// Get the bridge name.
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get the capacity.
    pub fn capacity(&self) -> usize {
        self.ring.capacity()
    }

    /// Check if the bridge is empty.
    pub fn is_empty(&self) -> bool {
        self.ring.is_empty()
    }

    /// Check if the bridge is full.
    pub fn is_full(&self) -> bool {
        self.ring.is_full()
    }

    /// Get the number of buffered items.
    pub fn len(&self) -> usize {
        self.ring.len()
    }

    // ========================================================================
    // Async (Producer) Side
    // ========================================================================

    /// Push a buffer asynchronously, waiting if the bridge is full.
    ///
    /// This is intended for use from Tokio async tasks.
    pub async fn push_async(&self, buffer: Buffer) -> Result<()> {
        let mut buf = buffer;
        loop {
            match self.ring.try_push(buf) {
                Ok(()) => {
                    // Signal data available to RT side
                    self.data_available.notify()?;
                    return Ok(());
                }
                Err(returned_buf) => {
                    buf = returned_buf;
                    // Wait for space
                    self.space_available.wait_async().await?;
                }
            }
        }
    }

    /// Try to push a buffer without blocking.
    ///
    /// Returns `Err(buffer)` if the bridge is full.
    pub fn try_push(&self, buffer: Buffer) -> std::result::Result<(), Buffer> {
        match self.ring.try_push(buffer) {
            Ok(()) => {
                // Signal data available (ignore errors in non-blocking path)
                let _ = self.data_available.notify();
                Ok(())
            }
            Err(buf) => Err(buf),
        }
    }

    // ========================================================================
    // RT (Consumer) Side
    // ========================================================================

    /// Try to pop a buffer without blocking.
    ///
    /// This is intended for use from RT threads. It never blocks.
    pub fn try_pop(&self) -> Option<Buffer> {
        let buffer = self.ring.try_pop();
        if buffer.is_some() {
            // Signal space available to async side (ignore errors)
            let _ = self.space_available.notify();
        }
        buffer
    }

    /// Check if data is available without consuming the eventfd.
    pub fn has_data(&self) -> bool {
        !self.ring.is_empty()
    }

    /// Get the data-available eventfd for use with epoll.
    ///
    /// This allows the RT thread to wait on multiple bridges efficiently.
    #[cfg(target_os = "linux")]
    pub fn data_eventfd(&self) -> &EventFd {
        &self.data_available
    }

    /// Drain the data-available eventfd.
    ///
    /// Call this after processing all available data to reset the eventfd.
    pub fn drain_data_signal(&self) -> Result<()> {
        while self.data_available.try_wait()? {}
        Ok(())
    }
}

// SAFETY: The bridge is designed for cross-thread use
unsafe impl Send for AsyncRtBridge {}
unsafe impl Sync for AsyncRtBridge {}

// ============================================================================
// Shared Bridge Handle
// ============================================================================

/// A shared handle to an [`AsyncRtBridge`].
pub type SharedBridge = Arc<AsyncRtBridge>;

/// Create a new shared bridge.
pub fn shared_bridge(config: BridgeConfig) -> Result<SharedBridge> {
    Ok(Arc::new(AsyncRtBridge::new(config)?))
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::memory::CpuSegment;
    use crate::metadata::Metadata;

    fn make_buffer(seq: u64) -> Buffer {
        let segment = Arc::new(CpuSegment::new(64).unwrap());
        let handle = MemoryHandle::from_segment(segment);
        Buffer::new(handle, Metadata::from_sequence(seq))
    }

    #[test]
    fn test_ring_buffer_basic() {
        let ring = RingBuffer::new(4).unwrap();

        assert!(ring.is_empty());
        assert!(!ring.is_full());
        assert_eq!(ring.len(), 0);

        // Push some items
        ring.try_push(make_buffer(1)).unwrap();
        assert!(!ring.is_empty());
        assert_eq!(ring.len(), 1);

        ring.try_push(make_buffer(2)).unwrap();
        ring.try_push(make_buffer(3)).unwrap();
        ring.try_push(make_buffer(4)).unwrap();

        assert!(ring.is_full());
        assert_eq!(ring.len(), 4);

        // Can't push more
        let buf = make_buffer(5);
        assert!(ring.try_push(buf).is_err());

        // Pop items
        let b = ring.try_pop().unwrap();
        assert_eq!(b.metadata().sequence, 1);

        let b = ring.try_pop().unwrap();
        assert_eq!(b.metadata().sequence, 2);

        assert_eq!(ring.len(), 2);

        // Can push again
        ring.try_push(make_buffer(5)).unwrap();
        ring.try_push(make_buffer(6)).unwrap();

        // Pop remaining
        assert_eq!(ring.try_pop().unwrap().metadata().sequence, 3);
        assert_eq!(ring.try_pop().unwrap().metadata().sequence, 4);
        assert_eq!(ring.try_pop().unwrap().metadata().sequence, 5);
        assert_eq!(ring.try_pop().unwrap().metadata().sequence, 6);

        assert!(ring.is_empty());
        assert!(ring.try_pop().is_none());
    }

    #[test]
    fn test_ring_buffer_power_of_two() {
        // Non-power-of-2 should fail
        assert!(RingBuffer::new(3).is_err());
        assert!(RingBuffer::new(5).is_err());

        // Power-of-2 should succeed
        assert!(RingBuffer::new(1).is_ok());
        assert!(RingBuffer::new(2).is_ok());
        assert!(RingBuffer::new(4).is_ok());
        assert!(RingBuffer::new(8).is_ok());
    }

    #[test]
    fn test_eventfd_basic() {
        let efd = EventFd::new().unwrap();

        // Initially not signaled
        assert!(!efd.try_wait().unwrap());

        // Signal it
        efd.notify().unwrap();

        // Now it should be signaled
        assert!(efd.try_wait().unwrap());

        // After consuming, not signaled again
        assert!(!efd.try_wait().unwrap());

        // Multiple signals accumulate
        efd.notify().unwrap();
        efd.notify().unwrap();
        efd.notify().unwrap();

        // On Linux, eventfd accumulates into a counter
        // Reading once consumes all signals
        assert!(efd.try_wait().unwrap());
    }

    #[test]
    fn test_bridge_basic() {
        let bridge = AsyncRtBridge::new(BridgeConfig::with_capacity(4)).unwrap();

        assert!(bridge.is_empty());
        assert_eq!(bridge.capacity(), 4);

        // Push via try_push
        bridge.try_push(make_buffer(1)).unwrap();
        bridge.try_push(make_buffer(2)).unwrap();

        assert!(!bridge.is_empty());
        assert_eq!(bridge.len(), 2);

        // Pop
        let b = bridge.try_pop().unwrap();
        assert_eq!(b.metadata().sequence, 1);

        let b = bridge.try_pop().unwrap();
        assert_eq!(b.metadata().sequence, 2);

        assert!(bridge.is_empty());
    }

    #[tokio::test]
    async fn test_bridge_async_push() {
        let bridge = Arc::new(AsyncRtBridge::new(BridgeConfig::with_capacity(4)).unwrap());

        // Push asynchronously
        bridge.push_async(make_buffer(1)).await.unwrap();
        bridge.push_async(make_buffer(2)).await.unwrap();

        assert_eq!(bridge.len(), 2);

        // Pop synchronously (simulating RT thread)
        assert_eq!(bridge.try_pop().unwrap().metadata().sequence, 1);
        assert_eq!(bridge.try_pop().unwrap().metadata().sequence, 2);
    }

    #[tokio::test]
    async fn test_bridge_backpressure() {
        let bridge = Arc::new(AsyncRtBridge::new(BridgeConfig::with_capacity(2)).unwrap());
        let bridge_clone = bridge.clone();

        // Fill the bridge
        bridge.push_async(make_buffer(1)).await.unwrap();
        bridge.push_async(make_buffer(2)).await.unwrap();

        assert!(bridge.is_full());

        // Spawn a task to push (will block until space available)
        let push_task = tokio::spawn(async move {
            bridge_clone.push_async(make_buffer(3)).await.unwrap();
        });

        // Give it a moment to start waiting
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        // Pop to make space
        let _ = bridge.try_pop();

        // Push task should complete
        tokio::time::timeout(std::time::Duration::from_millis(100), push_task)
            .await
            .expect("Push task should complete")
            .unwrap();

        // Verify the buffer was pushed
        assert_eq!(bridge.len(), 2);
    }

    #[test]
    fn test_shared_bridge() {
        let bridge = shared_bridge(BridgeConfig::with_capacity(8).with_name("test")).unwrap();

        assert_eq!(bridge.name(), "test");
        assert_eq!(bridge.capacity(), 8);
    }
}
