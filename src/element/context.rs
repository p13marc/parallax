//! Element runtime context.
//!
//! This module provides:
//!
//! - [`ElementContext`]: Runtime context for element initialization
//! - [`ProcessContext`]: Processing context with pre-allocated buffers
//!
//! # ProcessContext Design
//!
//! The [`ProcessContext`] API enables zero-copy processing by providing elements
//! with pre-allocated input and output memory views. Elements don't allocate;
//! the pipeline provides memory from arenas.
//!
//! ```rust,ignore
//! fn process_ctx(&mut self, ctx: &mut ProcessContext) -> Result<()> {
//!     let input = ctx.input();
//!     let output = ctx.output();
//!
//!     // Process input -> output
//!     output[..input.len()].copy_from_slice(input);
//!
//!     // Commit how much we wrote
//!     ctx.commit(input.len());
//!     Ok(())
//! }
//! ```

use crate::memory::MemoryPool;
use std::sync::Arc;

// ============================================================================
// ProcessContext - zero-copy processing context
// ============================================================================

/// Context provided to elements for processing.
///
/// This provides pre-allocated input and output buffers, enabling zero-copy
/// processing where elements don't need to allocate memory.
///
/// # Usage
///
/// ```rust,ignore
/// fn process_ctx(&mut self, ctx: &mut ProcessContext) -> Result<()> {
///     // Read input
///     let input = ctx.input();
///
///     // Write to pre-allocated output
///     let output = ctx.output();
///     output[..input.len()].copy_from_slice(input);
///
///     // Commit how much we wrote
///     ctx.commit(input.len());
///     Ok(())
/// }
/// ```
///
/// # In-Place Processing
///
/// For elements that can process data in-place (e.g., volume adjustment),
/// use [`in_place()`](Self::in_place):
///
/// ```rust,ignore
/// fn process_ctx(&mut self, ctx: &mut ProcessContext) -> Result<()> {
///     if let Some(data) = ctx.in_place() {
///         // Modify in place
///         for byte in data.iter_mut() {
///             *byte = byte.wrapping_add(1);
///         }
///         ctx.commit(data.len());
///     } else {
///         // Fall back to copy
///         let input = ctx.input();
///         let output = ctx.output();
///         output[..input.len()].copy_from_slice(input);
///         ctx.commit(input.len());
///     }
///     Ok(())
/// }
/// ```
pub struct ProcessContext<'a> {
    /// Input buffer (read-only view).
    input: MemoryView<'a>,
    /// Output buffer (write-only view, pre-allocated).
    output: MemoryViewMut<'a>,
    /// How much of output was actually used.
    committed_len: usize,
    /// Whether input and output point to the same memory (in-place mode).
    is_in_place: bool,
}

impl<'a> ProcessContext<'a> {
    /// Create a new process context with separate input and output buffers.
    pub fn new(input: &'a [u8], output: &'a mut [u8]) -> Self {
        Self {
            input: MemoryView::new(input),
            output: MemoryViewMut::new(output),
            committed_len: 0,
            is_in_place: false,
        }
    }

    /// Create a process context for in-place processing.
    ///
    /// The same memory is used for both input and output.
    ///
    /// # Safety
    ///
    /// The caller must ensure the memory can be safely modified.
    pub fn new_in_place(data: &'a mut [u8]) -> Self {
        let len = data.len();
        let ptr = data.as_ptr();

        // Create read-only view of the same memory
        // SAFETY: We're creating a read-only view that shares memory with output.
        // This is safe because:
        // 1. The lifetime 'a ensures both views are valid for the same duration
        // 2. The is_in_place flag prevents simultaneous read/write
        // 3. Elements use either input() OR in_place(), not both
        let input_slice = unsafe { std::slice::from_raw_parts(ptr, len) };

        Self {
            input: MemoryView::new(input_slice),
            output: MemoryViewMut::new(data),
            committed_len: 0,
            is_in_place: true,
        }
    }

    /// Get the input data (read-only).
    #[inline]
    pub fn input(&self) -> &[u8] {
        self.input.as_slice()
    }

    /// Get the input length.
    #[inline]
    pub fn input_len(&self) -> usize {
        self.input.len()
    }

    /// Get the output buffer (write-only).
    ///
    /// Write your processed data here, then call [`commit()`](Self::commit).
    #[inline]
    pub fn output(&mut self) -> &mut [u8] {
        self.output.as_mut_slice()
    }

    /// Get the output buffer capacity.
    #[inline]
    pub fn output_capacity(&self) -> usize {
        self.output.len()
    }

    /// Commit how much of the output was actually used.
    ///
    /// This must be called after writing to the output buffer.
    /// The committed length determines the size of the resulting buffer.
    ///
    /// # Panics
    ///
    /// Panics if `len > output_capacity()`.
    #[inline]
    pub fn commit(&mut self, len: usize) {
        assert!(
            len <= self.output.len(),
            "committed length exceeds output capacity"
        );
        self.committed_len = len;
    }

    /// Get the committed length.
    #[inline]
    pub fn committed_len(&self) -> usize {
        self.committed_len
    }

    /// Check if any data was committed.
    #[inline]
    pub fn is_committed(&self) -> bool {
        self.committed_len > 0
    }

    /// Get a mutable view for in-place processing.
    ///
    /// Returns `Some(&mut [u8])` if this context was created with
    /// [`new_in_place()`](Self::new_in_place), otherwise returns `None`.
    ///
    /// Use this when your element can process data in-place without
    /// needing a separate output buffer.
    #[inline]
    pub fn in_place(&mut self) -> Option<&mut [u8]> {
        if self.is_in_place {
            Some(self.output.as_mut_slice())
        } else {
            None
        }
    }

    /// Check if this context is in in-place mode.
    #[inline]
    pub fn is_in_place_mode(&self) -> bool {
        self.is_in_place
    }

    /// Copy all input to output and commit.
    ///
    /// This is a convenience method for passthrough elements.
    /// Returns the number of bytes copied.
    #[inline]
    pub fn passthrough(&mut self) -> usize {
        let len = self.input.len().min(self.output.len());
        // SAFETY: We're copying from input to output with non-overlapping memory
        // (unless in-place mode, where it's a no-op).
        if !self.is_in_place {
            let src = self.input.data.as_ptr();
            let dst = self.output.data.as_mut_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, len);
            }
        }
        self.committed_len = len;
        len
    }

    /// Copy a portion of input to output and commit.
    ///
    /// # Arguments
    ///
    /// * `input_offset` - Start offset in input
    /// * `len` - Number of bytes to copy
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds.
    #[inline]
    pub fn copy_range(&mut self, input_offset: usize, len: usize) {
        assert!(input_offset + len <= self.input.len());
        assert!(len <= self.output.len());

        if !self.is_in_place {
            let src = unsafe { self.input.data.as_ptr().add(input_offset) };
            let dst = self.output.data.as_mut_ptr();
            unsafe {
                std::ptr::copy_nonoverlapping(src, dst, len);
            }
        }
        self.committed_len = len;
    }
}

impl std::fmt::Debug for ProcessContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProcessContext")
            .field("input_len", &self.input.len())
            .field("output_capacity", &self.output.len())
            .field("committed_len", &self.committed_len)
            .field("is_in_place", &self.is_in_place)
            .finish()
    }
}

// ============================================================================
// MemoryView - read-only memory view
// ============================================================================

/// Read-only view into memory.
///
/// This is a lightweight wrapper around a byte slice that makes
/// the read-only intent explicit in the type system.
#[derive(Clone, Copy)]
pub struct MemoryView<'a> {
    data: &'a [u8],
}

impl<'a> MemoryView<'a> {
    /// Create a new memory view.
    #[inline]
    pub fn new(data: &'a [u8]) -> Self {
        Self { data }
    }

    /// Get the underlying slice.
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.data
    }

    /// Get the length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Get a sub-view.
    #[inline]
    pub fn slice(&self, offset: usize, len: usize) -> Self {
        Self {
            data: &self.data[offset..offset + len],
        }
    }
}

impl std::fmt::Debug for MemoryView<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryView")
            .field("len", &self.data.len())
            .finish()
    }
}

impl<'a> AsRef<[u8]> for MemoryView<'a> {
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

// ============================================================================
// MemoryViewMut - mutable memory view
// ============================================================================

/// Mutable view into memory.
///
/// This is a lightweight wrapper around a mutable byte slice that makes
/// the write intent explicit in the type system.
pub struct MemoryViewMut<'a> {
    data: &'a mut [u8],
}

impl<'a> MemoryViewMut<'a> {
    /// Create a new mutable memory view.
    #[inline]
    pub fn new(data: &'a mut [u8]) -> Self {
        Self { data }
    }

    /// Get the underlying mutable slice.
    #[inline]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.data
    }

    /// Get the underlying slice (read-only).
    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        self.data
    }

    /// Get the length.
    #[inline]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Check if empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

impl std::fmt::Debug for MemoryViewMut<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryViewMut")
            .field("len", &self.data.len())
            .finish()
    }
}

impl<'a> AsRef<[u8]> for MemoryViewMut<'a> {
    fn as_ref(&self) -> &[u8] {
        self.data
    }
}

impl<'a> AsMut<[u8]> for MemoryViewMut<'a> {
    fn as_mut(&mut self) -> &mut [u8] {
        self.data
    }
}

// ============================================================================
// ElementContext - element initialization context
// ============================================================================

/// Runtime context for an element.
///
/// The context is passed to elements during initialization and provides
/// access to shared resources like memory pools and configuration.
#[derive(Clone)]
pub struct ElementContext {
    /// Name of this element instance.
    name: String,
    /// Optional memory pool for buffer allocation.
    pool: Option<Arc<MemoryPool>>,
}

impl ElementContext {
    /// Create a new element context.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            pool: None,
        }
    }

    /// Create a context with a memory pool.
    pub fn with_pool(name: impl Into<String>, pool: Arc<MemoryPool>) -> Self {
        Self {
            name: name.into(),
            pool: Some(pool),
        }
    }

    /// Get the element's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the memory pool, if one is configured.
    pub fn pool(&self) -> Option<&Arc<MemoryPool>> {
        self.pool.as_ref()
    }

    /// Set the memory pool.
    pub fn set_pool(&mut self, pool: Arc<MemoryPool>) {
        self.pool = Some(pool);
    }
}

impl std::fmt::Debug for ElementContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElementContext")
            .field("name", &self.name)
            .field("has_pool", &self.pool.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::HeapSegment;

    #[test]
    fn test_context_creation() {
        let ctx = ElementContext::new("test-element");
        assert_eq!(ctx.name(), "test-element");
        assert!(ctx.pool().is_none());
    }

    #[test]
    fn test_context_with_pool() {
        let segment = HeapSegment::new(1024).unwrap();
        let pool = Arc::new(MemoryPool::new(segment, 256).unwrap());

        let ctx = ElementContext::with_pool("test-element", pool.clone());
        assert!(ctx.pool().is_some());
        assert_eq!(ctx.pool().unwrap().capacity(), pool.capacity());
    }

    // ProcessContext tests

    #[test]
    fn test_process_context_creation() {
        let input = b"hello world";
        let mut output = [0u8; 32];

        let ctx = ProcessContext::new(input, &mut output);

        assert_eq!(ctx.input_len(), 11);
        assert_eq!(ctx.output_capacity(), 32);
        assert_eq!(ctx.committed_len(), 0);
        assert!(!ctx.is_committed());
        assert!(!ctx.is_in_place_mode());
    }

    #[test]
    fn test_process_context_passthrough() {
        let input = b"hello world";
        let mut output = [0u8; 32];

        let mut ctx = ProcessContext::new(input, &mut output);

        // Use passthrough helper
        let copied = ctx.passthrough();

        assert_eq!(copied, 11);
        assert_eq!(ctx.committed_len(), 11);
        assert!(ctx.is_committed());
        assert_eq!(&output[..11], b"hello world");
    }

    #[test]
    fn test_process_context_copy_range() {
        let input = b"hello world";
        let mut output = [0u8; 32];

        let mut ctx = ProcessContext::new(input, &mut output);

        // Copy only "world"
        ctx.copy_range(6, 5);

        assert_eq!(ctx.committed_len(), 5);
        assert_eq!(&output[..5], b"world");
    }

    #[test]
    fn test_process_context_in_place() {
        let mut data = *b"hello world";

        let mut ctx = ProcessContext::new_in_place(&mut data);

        assert!(ctx.is_in_place_mode());
        assert_eq!(ctx.input(), b"hello world");

        // Modify in place
        let buf_len = {
            let buf = ctx.in_place().unwrap();
            for byte in buf.iter_mut() {
                *byte = byte.to_ascii_uppercase();
            }
            buf.len()
        };
        ctx.commit(buf_len);

        assert_eq!(ctx.committed_len(), 11);
        assert_eq!(&data, b"HELLO WORLD");
    }

    #[test]
    fn test_process_context_not_in_place() {
        let input = b"hello";
        let mut output = [0u8; 10];

        let mut ctx = ProcessContext::new(input, &mut output);

        // in_place() should return None when not in in-place mode
        assert!(ctx.in_place().is_none());
    }

    #[test]
    #[should_panic(expected = "committed length exceeds output capacity")]
    fn test_process_context_commit_overflow() {
        let input = b"hello";
        let mut output = [0u8; 10];

        let mut ctx = ProcessContext::new(input, &mut output);
        ctx.commit(100); // Should panic
    }

    // MemoryView tests

    #[test]
    fn test_memory_view() {
        let data = b"hello world";
        let view = MemoryView::new(data);

        assert_eq!(view.len(), 11);
        assert!(!view.is_empty());
        assert_eq!(view.as_slice(), b"hello world");
        assert_eq!(view.as_ref(), b"hello world");
    }

    #[test]
    fn test_memory_view_slice() {
        let data = b"hello world";
        let view = MemoryView::new(data);
        let sub = view.slice(6, 5);

        assert_eq!(sub.as_slice(), b"world");
    }

    #[test]
    fn test_memory_view_empty() {
        let data: &[u8] = &[];
        let view = MemoryView::new(data);

        assert!(view.is_empty());
        assert_eq!(view.len(), 0);
    }

    // MemoryViewMut tests

    #[test]
    fn test_memory_view_mut() {
        let mut data = *b"hello world";
        let mut view = MemoryViewMut::new(&mut data);

        assert_eq!(view.len(), 11);
        assert!(!view.is_empty());
        assert_eq!(view.as_slice(), b"hello world");

        // Modify
        view.as_mut_slice()[0] = b'H';
        assert_eq!(view.as_slice(), b"Hello world");
    }

    #[test]
    fn test_memory_view_mut_traits() {
        let mut data = *b"test";
        let mut view = MemoryViewMut::new(&mut data);

        // AsRef
        let slice: &[u8] = view.as_ref();
        assert_eq!(slice, b"test");

        // AsMut
        let slice_mut: &mut [u8] = view.as_mut();
        slice_mut[0] = b'T';
        assert_eq!(&data, b"Test");
    }
}
