//! Lock-free atomic bitmap for slot tracking.

use std::sync::atomic::{AtomicU64, Ordering};

/// A lock-free bitmap for tracking slot allocation.
///
/// Uses atomic operations to allow concurrent allocation and deallocation
/// without locks. Each bit represents one slot: 0 = free, 1 = allocated.
///
/// # Performance
///
/// - `acquire_slot`: O(n/64) worst case, where n is number of slots
/// - `release_slot`: O(1)
///
/// The implementation uses 64-bit words for efficient scanning.
pub struct AtomicBitmap {
    /// Array of atomic 64-bit words.
    words: Box<[AtomicU64]>,
    /// Total number of slots (may be less than words.len() * 64).
    num_slots: usize,
}

impl AtomicBitmap {
    /// Create a new bitmap with all slots free.
    ///
    /// # Arguments
    ///
    /// * `num_slots` - Number of slots to track.
    pub fn new(num_slots: usize) -> Self {
        let num_words = num_slots.div_ceil(64);
        let words: Vec<AtomicU64> = (0..num_words).map(|_| AtomicU64::new(0)).collect();

        Self {
            words: words.into_boxed_slice(),
            num_slots,
        }
    }

    /// Try to acquire a free slot.
    ///
    /// Returns the slot index if successful, or `None` if all slots are allocated.
    ///
    /// This operation is lock-free and thread-safe.
    pub fn acquire_slot(&self) -> Option<usize> {
        for (word_idx, word) in self.words.iter().enumerate() {
            // Try to find and set a zero bit in this word
            loop {
                let current = word.load(Ordering::Relaxed);

                // All bits set? Try next word.
                if current == u64::MAX {
                    break;
                }

                // Find the first zero bit
                let bit_idx = (!current).trailing_zeros() as usize;
                let slot_idx = word_idx * 64 + bit_idx;

                // Check if this slot is within bounds
                if slot_idx >= self.num_slots {
                    // No more valid slots in this word or beyond
                    return None;
                }

                // Try to set this bit
                let new_value = current | (1u64 << bit_idx);
                match word.compare_exchange_weak(
                    current,
                    new_value,
                    Ordering::AcqRel,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => return Some(slot_idx),
                    Err(_) => continue, // Another thread modified; retry
                }
            }
        }

        None
    }

    /// Release a previously acquired slot.
    ///
    /// # Arguments
    ///
    /// * `slot_idx` - The slot index to release.
    ///
    /// # Panics
    ///
    /// Panics if `slot_idx` is out of bounds.
    pub fn release_slot(&self, slot_idx: usize) {
        assert!(slot_idx < self.num_slots, "slot index out of bounds");

        let word_idx = slot_idx / 64;
        let bit_idx = slot_idx % 64;

        // Clear the bit
        self.words[word_idx].fetch_and(!(1u64 << bit_idx), Ordering::Release);
    }

    /// Check if a slot is currently allocated.
    ///
    /// Note: This is a snapshot and may change immediately after returning.
    pub fn is_allocated(&self, slot_idx: usize) -> bool {
        if slot_idx >= self.num_slots {
            return false;
        }

        let word_idx = slot_idx / 64;
        let bit_idx = slot_idx % 64;

        (self.words[word_idx].load(Ordering::Relaxed) & (1u64 << bit_idx)) != 0
    }

    /// Count the number of free slots.
    ///
    /// Note: This is a snapshot and may change immediately after returning.
    pub fn count_free(&self) -> usize {
        let allocated: usize = self
            .words
            .iter()
            .enumerate()
            .map(|(i, word)| {
                let bits = word.load(Ordering::Relaxed);
                if i == self.words.len() - 1 {
                    // Last word may have unused bits
                    let valid_bits = self.num_slots - i * 64;
                    if valid_bits >= 64 {
                        bits.count_ones() as usize
                    } else {
                        (bits & ((1u64 << valid_bits) - 1)).count_ones() as usize
                    }
                } else {
                    bits.count_ones() as usize
                }
            })
            .sum();

        self.num_slots - allocated
    }

    /// Get the total number of slots.
    pub fn capacity(&self) -> usize {
        self.num_slots
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_bitmap_basic() {
        let bitmap = AtomicBitmap::new(10);
        assert_eq!(bitmap.capacity(), 10);
        assert_eq!(bitmap.count_free(), 10);

        // Acquire some slots
        let s0 = bitmap.acquire_slot();
        let s1 = bitmap.acquire_slot();
        let s2 = bitmap.acquire_slot();

        assert_eq!(s0, Some(0));
        assert_eq!(s1, Some(1));
        assert_eq!(s2, Some(2));
        assert_eq!(bitmap.count_free(), 7);

        // Release and re-acquire
        bitmap.release_slot(1);
        assert_eq!(bitmap.count_free(), 8);
        assert!(!bitmap.is_allocated(1));

        let s3 = bitmap.acquire_slot();
        assert_eq!(s3, Some(1)); // Should reuse slot 1
    }

    #[test]
    fn test_bitmap_exhaustion() {
        let bitmap = AtomicBitmap::new(3);

        assert!(bitmap.acquire_slot().is_some());
        assert!(bitmap.acquire_slot().is_some());
        assert!(bitmap.acquire_slot().is_some());
        assert!(bitmap.acquire_slot().is_none()); // Exhausted

        bitmap.release_slot(1);
        assert!(bitmap.acquire_slot().is_some()); // Slot 1 available again
        assert!(bitmap.acquire_slot().is_none()); // Exhausted again
    }

    #[test]
    fn test_bitmap_concurrent() {
        let bitmap = Arc::new(AtomicBitmap::new(1000));
        let mut handles = vec![];

        // Spawn threads that acquire and release slots
        for _ in 0..4 {
            let bitmap = Arc::clone(&bitmap);
            handles.push(thread::spawn(move || {
                let mut acquired = vec![];
                for _ in 0..100 {
                    if let Some(slot) = bitmap.acquire_slot() {
                        acquired.push(slot);
                    }
                }
                // Release half
                for slot in acquired.iter().take(acquired.len() / 2) {
                    bitmap.release_slot(*slot);
                }
                acquired.len()
            }));
        }

        let total_acquired: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();
        assert!(total_acquired <= 1000);
    }

    #[test]
    fn test_bitmap_non_aligned_size() {
        // Test with size that doesn't align to 64
        let bitmap = AtomicBitmap::new(100);
        assert_eq!(bitmap.capacity(), 100);

        // Acquire all slots
        for i in 0..100 {
            let slot = bitmap.acquire_slot();
            assert_eq!(slot, Some(i), "failed at slot {}", i);
        }

        // Should be exhausted
        assert!(bitmap.acquire_slot().is_none());
        assert_eq!(bitmap.count_free(), 0);
    }

    #[test]
    #[should_panic(expected = "slot index out of bounds")]
    fn test_bitmap_release_out_of_bounds() {
        let bitmap = AtomicBitmap::new(10);
        bitmap.release_slot(10); // Should panic
    }
}
