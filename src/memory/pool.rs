//! Memory pool with loan semantics.

use super::{AtomicBitmap, MemorySegment, MemoryType};
use crate::error::{Error, Result};
use std::sync::Arc;

/// A memory pool that manages fixed-size slots for buffer allocation.
///
/// Inspired by iceoryx2's loan-based model: callers "loan" memory from the pool,
/// use it, and the memory is automatically returned when the loan is dropped.
///
/// # Design
///
/// - Fixed slot size: All slots in a pool are the same size
/// - Lock-free allocation: Uses atomic bitmap for slot tracking
/// - Zero-copy: Slots are views into the underlying memory segment
/// - RAII: `LoanedSlot` returns memory to pool on drop
///
/// # Example
///
/// ```rust
/// use parallax::memory::{MemoryPool, HeapSegment};
///
/// // Create a pool with 64KB slots
/// let segment = HeapSegment::new(16 * 64 * 1024).unwrap();
/// let pool = MemoryPool::new(segment, 64 * 1024).unwrap();
///
/// // Loan a slot
/// let mut slot = pool.loan().expect("pool has slots");
///
/// // Write data
/// slot.as_mut_slice()[..5].copy_from_slice(b"hello");
///
/// // Slot is returned to pool when dropped
/// drop(slot);
/// ```
pub struct MemoryPool {
    /// The underlying memory segment.
    segment: Arc<dyn MemorySegment>,
    /// Size of each slot in bytes.
    slot_size: usize,
    /// Number of slots in the pool.
    num_slots: usize,
    /// Bitmap tracking which slots are allocated.
    bitmap: AtomicBitmap,
}

impl MemoryPool {
    /// Create a new memory pool from a segment.
    ///
    /// # Arguments
    ///
    /// * `segment` - The memory segment to use as backing storage.
    /// * `slot_size` - Size of each slot in bytes.
    ///
    /// # Returns
    ///
    /// A new `MemoryPool`, or an error if the segment is too small.
    ///
    /// # Notes
    ///
    /// The number of slots is determined by `segment.len() / slot_size`.
    /// Any remainder is unused.
    pub fn new(segment: impl MemorySegment + 'static, slot_size: usize) -> Result<Self> {
        Self::new_arc(Arc::new(segment), slot_size)
    }

    /// Create a new memory pool from an Arc'd segment.
    pub fn new_arc(segment: Arc<dyn MemorySegment>, slot_size: usize) -> Result<Self> {
        if slot_size == 0 {
            return Err(Error::AllocationFailed("slot size must be > 0".into()));
        }

        let num_slots = segment.len() / slot_size;
        if num_slots == 0 {
            return Err(Error::AllocationFailed(
                "segment too small for even one slot".into(),
            ));
        }

        let bitmap = AtomicBitmap::new(num_slots);

        Ok(Self {
            segment,
            slot_size,
            num_slots,
            bitmap,
        })
    }

    /// Loan a slot from the pool.
    ///
    /// Returns a `LoanedSlot` that provides access to the memory. The slot
    /// is automatically returned to the pool when the `LoanedSlot` is dropped.
    ///
    /// # Returns
    ///
    /// `Some(LoanedSlot)` if a slot is available, `None` if the pool is exhausted.
    pub fn loan(&self) -> Option<LoanedSlot<'_>> {
        let slot_idx = self.bitmap.acquire_slot()?;

        Some(LoanedSlot {
            pool: self,
            slot_idx,
        })
    }

    /// Get the size of each slot in bytes.
    pub fn slot_size(&self) -> usize {
        self.slot_size
    }

    /// Get the total number of slots in the pool.
    pub fn capacity(&self) -> usize {
        self.num_slots
    }

    /// Get the number of currently available (free) slots.
    pub fn available(&self) -> usize {
        self.bitmap.count_free()
    }

    /// Get the memory type of the underlying segment.
    pub fn memory_type(&self) -> MemoryType {
        self.segment.memory_type()
    }

    /// Get a reference to the underlying segment.
    pub fn segment(&self) -> &Arc<dyn MemorySegment> {
        &self.segment
    }

    /// Calculate the pointer to a specific slot.
    fn slot_ptr(&self, slot_idx: usize) -> *mut u8 {
        debug_assert!(slot_idx < self.num_slots);
        let offset = slot_idx * self.slot_size;
        unsafe { (self.segment.as_ptr() as *mut u8).add(offset) }
    }
}

/// A loaned memory slot from a pool.
///
/// This is an RAII guard: when dropped, the slot is automatically returned
/// to the pool for reuse.
///
/// # Safety
///
/// The `LoanedSlot` provides mutable access to the memory. It is the caller's
/// responsibility to ensure proper synchronization if sharing data through
/// the slot with other threads.
pub struct LoanedSlot<'pool> {
    pool: &'pool MemoryPool,
    slot_idx: usize,
}

impl<'pool> LoanedSlot<'pool> {
    /// Get a pointer to the slot's memory.
    pub fn as_ptr(&self) -> *const u8 {
        self.pool.slot_ptr(self.slot_idx)
    }

    /// Get a mutable pointer to the slot's memory.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.pool.slot_ptr(self.slot_idx)
    }

    /// Get the size of this slot in bytes.
    pub fn len(&self) -> usize {
        self.pool.slot_size
    }

    /// Returns true if the slot has zero size (which shouldn't happen).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Get the slot as a byte slice.
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Get the slot as a mutable byte slice.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        unsafe { std::slice::from_raw_parts_mut(self.as_mut_ptr(), self.len()) }
    }

    /// Get the slot index within the pool.
    pub fn slot_index(&self) -> usize {
        self.slot_idx
    }

    /// Get the memory type of the underlying segment.
    pub fn memory_type(&self) -> MemoryType {
        self.pool.memory_type()
    }

    /// Write bytes to the slot.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() > self.len()`.
    pub fn write(&mut self, data: &[u8]) {
        assert!(
            data.len() <= self.len(),
            "data ({} bytes) exceeds slot size ({} bytes)",
            data.len(),
            self.len()
        );
        self.as_mut_slice()[..data.len()].copy_from_slice(data);
    }
}

impl Drop for LoanedSlot<'_> {
    fn drop(&mut self) {
        self.pool.bitmap.release_slot(self.slot_idx);
    }
}

// LoanedSlot is Send if the underlying memory is Send (which it is via MemorySegment: Send)
unsafe impl Send for LoanedSlot<'_> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HeapSegment;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_pool_creation() {
        let segment = HeapSegment::new(1024).unwrap();
        let pool = MemoryPool::new(segment, 256).unwrap();

        assert_eq!(pool.slot_size(), 256);
        assert_eq!(pool.capacity(), 4); // 1024 / 256
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_pool_loan_and_return() {
        let segment = HeapSegment::new(512).unwrap();
        let pool = MemoryPool::new(segment, 128).unwrap();

        assert_eq!(pool.available(), 4);

        {
            let _slot1 = pool.loan().unwrap();
            assert_eq!(pool.available(), 3);

            let _slot2 = pool.loan().unwrap();
            assert_eq!(pool.available(), 2);
        }

        // Slots should be returned
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_pool_exhaustion() {
        let segment = HeapSegment::new(256).unwrap();
        let pool = MemoryPool::new(segment, 128).unwrap();

        let slot1 = pool.loan();
        let slot2 = pool.loan();
        let slot3 = pool.loan(); // Should fail

        assert!(slot1.is_some());
        assert!(slot2.is_some());
        assert!(slot3.is_none());
    }

    #[test]
    fn test_slot_read_write() {
        let segment = HeapSegment::new(256).unwrap();
        let pool = MemoryPool::new(segment, 128).unwrap();

        let mut slot = pool.loan().unwrap();
        slot.write(b"hello world");

        assert_eq!(&slot.as_slice()[..11], b"hello world");
    }

    #[test]
    fn test_pool_concurrent_access() {
        let segment = HeapSegment::new(1024 * 1024).unwrap();
        let pool = Arc::new(MemoryPool::new(segment, 1024).unwrap());

        let initial_capacity = pool.capacity();
        let mut handles = vec![];

        for i in 0..4 {
            let pool = Arc::clone(&pool);
            handles.push(thread::spawn(move || {
                let mut slots = vec![];
                // Try to acquire many slots
                for _ in 0..100 {
                    if let Some(mut slot) = pool.loan() {
                        slot.write(&[i as u8; 100]);
                        slots.push(slot);
                    }
                }
                // Keep them briefly, then drop
                thread::sleep(std::time::Duration::from_millis(1));
                slots.len()
            }));
        }

        let total: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

        // All slots should be returned
        assert_eq!(pool.available(), initial_capacity);
        // Total acquired should not exceed capacity
        assert!(total <= initial_capacity);
    }

    #[test]
    fn test_slot_memory_type() {
        let segment = HeapSegment::new(256).unwrap();
        let pool = MemoryPool::new(segment, 128).unwrap();

        let slot = pool.loan().unwrap();
        assert_eq!(slot.memory_type(), MemoryType::Heap);
    }
}
