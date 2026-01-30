//! Element runtime context.
//!
//! This module provides context types for pool-aware buffer processing:
//!
//! - [`ProduceContext`]: Context for sources to write into pre-allocated buffers
//! - [`ConsumeContext`]: Context for sinks to read from buffers
//! - [`ProcessContext`]: Context for transforms with input and output buffers
//! - [`ElementContext`]: Runtime context for element initialization
//!
//! # PipeWire-Style Buffer Management
//!
//! Parallax uses a PipeWire-inspired pattern where the framework provides
//! pre-allocated buffers to elements, enabling true zero-allocation processing:
//!
//! ```rust,ignore
//! // Source: framework provides output buffer
//! fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
//!     let output = ctx.output();
//!     let n = self.reader.read(output)?;
//!     Ok(ProduceResult::Produced(n))
//! }
//!
//! // Sink: framework provides input buffer
//! fn consume(&mut self, ctx: &mut ConsumeContext) -> Result<()> {
//!     let input = ctx.input();
//!     self.writer.write_all(input)?;
//!     Ok(())
//! }
//!
//! // Transform: framework provides both input and output
//! fn process(&mut self, ctx: &mut ProcessContext) -> Result<()> {
//!     let input = ctx.input();
//!     let output = ctx.output();
//!     output[..input.len()].copy_from_slice(input);
//!     ctx.commit(input.len());
//!     Ok(())
//! }
//! ```

use std::sync::Arc;

use crate::buffer::{Buffer, DmaBufBuffer, MemoryHandle};
use crate::clock::{Clock, ClockTime};
use crate::error::{Error, Result};
use crate::memory::{BufferPool, PooledBuffer, SharedSlotRef};
use crate::metadata::Metadata;

// ============================================================================
// ProduceResult - return type for source produce()
// ============================================================================

/// Result of a source's produce operation.
///
/// Sources return this to indicate:
/// - How many bytes were written to the provided buffer
/// - End-of-stream
/// - Or that they need to provide their own buffer
#[derive(Debug)]
#[allow(clippy::large_enum_variant)] // Intentional: avoid heap allocation on hot path
pub enum ProduceResult {
    /// Produced `n` bytes into the provided buffer.
    ///
    /// The framework will create a buffer with the first `n` bytes
    /// of the output buffer and the metadata from the context.
    Produced(usize),

    /// End of stream - no more data will be produced.
    Eos,

    /// Source provides its own buffer.
    ///
    /// Use this when:
    /// - The source has its own memory management (e.g., mmap)
    /// - The source receives buffers from external systems
    /// - Integration with legacy APIs that provide their own buffers
    OwnBuffer(Buffer),

    /// Source provides a DMA-BUF buffer.
    ///
    /// Use this when the source exports buffers as DMA-BUF file descriptors:
    /// - V4L2 camera capture with `VIDIOC_EXPBUF`
    /// - libcamera DMA-BUF frame buffers
    /// - GPU-produced frames
    ///
    /// DMA-BUF buffers enable zero-copy paths to GPU encoders/decoders.
    OwnDmaBuf(DmaBufBuffer),

    /// No data available yet (for non-blocking sources).
    ///
    /// The framework should retry later.
    WouldBlock,
}

impl ProduceResult {
    /// Check if this is end-of-stream.
    #[inline]
    pub fn is_eos(&self) -> bool {
        matches!(self, Self::Eos)
    }

    /// Check if data was produced.
    #[inline]
    pub fn is_produced(&self) -> bool {
        matches!(
            self,
            Self::Produced(_) | Self::OwnBuffer(_) | Self::OwnDmaBuf(_)
        )
    }

    /// Check if the source would block.
    #[inline]
    pub fn is_would_block(&self) -> bool {
        matches!(self, Self::WouldBlock)
    }
}

// ============================================================================
// ProduceContext - context for sources
// ============================================================================

/// Context provided to sources for producing data.
///
/// The framework provides a pre-allocated output buffer where sources
/// write their data. This enables zero-allocation production.
///
/// # Usage
///
/// ```rust,ignore
/// fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
///     // Get the output buffer to write into
///     let output = ctx.output();
///
///     // Write data (e.g., read from file, receive from network)
///     let n = self.reader.read(output)?;
///     if n == 0 {
///         return Ok(ProduceResult::Eos);
///     }
///
///     // Set metadata
///     ctx.metadata_mut().sequence = self.seq;
///     self.seq += 1;
///
///     Ok(ProduceResult::Produced(n))
/// }
/// ```
///
/// # Own Buffer Fallback
///
/// Some sources need to provide their own buffers (e.g., memory-mapped files,
/// external APIs). Use `ProduceResult::OwnBuffer` for these cases:
///
/// ```rust,ignore
/// fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
///     // Source manages its own memory
///     let buffer = self.external_api.get_buffer()?;
///     Ok(ProduceResult::OwnBuffer(buffer))
/// }
/// ```
pub struct ProduceContext<'a> {
    /// Pre-allocated output buffer from the pool.
    slot: Option<SharedSlotRef>,
    /// Mutable slice view into the slot (if slot is Some).
    output: Option<&'a mut [u8]>,
    /// Metadata for the produced buffer.
    metadata: Metadata,
    /// Capacity of the output buffer.
    capacity: usize,
    /// Optional buffer pool for acquiring additional buffers.
    pool: Option<&'a dyn BufferPool>,
    /// Optional pipeline clock for timing.
    clock: Option<Arc<dyn Clock>>,
    /// Base time (clock time when pipeline started).
    base_time: ClockTime,
}

impl<'a> ProduceContext<'a> {
    /// Create a new produce context with a pre-allocated slot.
    ///
    /// # Safety
    ///
    /// The slot must remain valid for the lifetime `'a`.
    pub fn new(slot: SharedSlotRef) -> Self {
        let capacity = slot.len();
        // SAFETY: We own the slot and it's valid for the duration of this context.
        // The pointer is valid because SharedSlotRef guarantees valid memory.
        let output = unsafe {
            let ptr = slot.as_ptr() as *mut u8;
            Some(std::slice::from_raw_parts_mut(ptr, capacity))
        };
        Self {
            slot: Some(slot),
            output,
            metadata: Metadata::new(),
            capacity,
            pool: None,
            clock: None,
            base_time: ClockTime::NONE,
        }
    }

    /// Create a new produce context with a pre-allocated slot and pool access.
    ///
    /// # Safety
    ///
    /// The slot must remain valid for the lifetime `'a`.
    pub fn with_pool(slot: SharedSlotRef, pool: &'a dyn BufferPool) -> Self {
        let capacity = slot.len();
        // SAFETY: We own the slot and it's valid for the duration of this context.
        // The pointer is valid because SharedSlotRef guarantees valid memory.
        let output = unsafe {
            let ptr = slot.as_ptr() as *mut u8;
            Some(std::slice::from_raw_parts_mut(ptr, capacity))
        };
        Self {
            slot: Some(slot),
            output,
            metadata: Metadata::new(),
            capacity,
            pool: Some(pool),
            clock: None,
            base_time: ClockTime::NONE,
        }
    }

    /// Create a produce context without a pre-allocated buffer.
    ///
    /// Sources using this context must return `ProduceResult::OwnBuffer`.
    pub fn without_buffer() -> Self {
        Self {
            slot: None,
            output: None,
            metadata: Metadata::new(),
            capacity: 0,
            pool: None,
            clock: None,
            base_time: ClockTime::NONE,
        }
    }

    /// Create a produce context with only pool access (no pre-allocated buffer).
    ///
    /// Sources can acquire buffers from the pool using `acquire_buffer()`.
    pub fn with_pool_only(pool: &'a dyn BufferPool) -> Self {
        Self {
            slot: None,
            output: None,
            metadata: Metadata::new(),
            capacity: 0,
            pool: Some(pool),
            clock: None,
            base_time: ClockTime::NONE,
        }
    }

    /// Set the pipeline clock for this context.
    ///
    /// This allows sources to access the pipeline clock for timing decisions.
    pub fn set_clock(&mut self, clock: Arc<dyn Clock>, base_time: ClockTime) {
        self.clock = Some(clock);
        self.base_time = base_time;
    }

    /// Get the pipeline clock, if available.
    pub fn clock(&self) -> Option<&Arc<dyn Clock>> {
        self.clock.as_ref()
    }

    /// Get the current running time (time since pipeline started).
    ///
    /// Returns `ClockTime::NONE` if no clock is configured or pipeline hasn't started.
    pub fn running_time(&self) -> ClockTime {
        if let Some(clock) = &self.clock {
            if self.base_time.is_some() {
                return clock.now().saturating_sub(self.base_time);
            }
        }
        ClockTime::NONE
    }

    /// Get the current clock time.
    ///
    /// Returns `ClockTime::NONE` if no clock is configured.
    pub fn clock_time(&self) -> ClockTime {
        self.clock
            .as_ref()
            .map(|c| c.now())
            .unwrap_or(ClockTime::NONE)
    }

    /// Get the base time (when pipeline started).
    pub fn base_time(&self) -> ClockTime {
        self.base_time
    }

    /// Get the output buffer to write into.
    ///
    /// # Panics
    ///
    /// Panics if no buffer was provided (use `has_buffer()` to check).
    #[inline]
    pub fn output(&mut self) -> &mut [u8] {
        self.output
            .as_deref_mut()
            .expect("ProduceContext has no buffer - use ProduceResult::OwnBuffer")
    }

    /// Check if a buffer was provided.
    #[inline]
    pub fn has_buffer(&self) -> bool {
        self.slot.is_some()
    }

    /// Check if a buffer pool is available.
    #[inline]
    pub fn has_pool(&self) -> bool {
        self.pool.is_some()
    }

    /// Acquire a buffer from the pool.
    ///
    /// This is the preferred method for pool-aware sources. The returned
    /// `PooledBuffer` can be written to and then converted to a `Buffer`
    /// using `into_buffer()`.
    ///
    /// # Errors
    ///
    /// Returns an error if no pool is configured.
    ///
    /// # Example
    ///
    /// ```rust,ignore
    /// fn produce(&mut self, ctx: &mut ProduceContext) -> Result<ProduceResult> {
    ///     if ctx.has_pool() {
    ///         let mut pooled = ctx.acquire_buffer()?;
    ///         let n = self.reader.read(pooled.data_mut())?;
    ///         pooled.set_len(n);
    ///         pooled.metadata_mut().sequence = self.seq;
    ///         Ok(ProduceResult::OwnBuffer(pooled.into_buffer()))
    ///     } else {
    ///         // Fall back to provided buffer
    ///         let output = ctx.output();
    ///         let n = self.reader.read(output)?;
    ///         Ok(ProduceResult::Produced(n))
    ///     }
    /// }
    /// ```
    pub fn acquire_buffer(&self) -> Result<PooledBuffer> {
        match self.pool {
            Some(pool) => pool.acquire(),
            None => Err(Error::Config("no buffer pool configured".into())),
        }
    }

    /// Try to acquire a buffer from the pool without blocking.
    ///
    /// Returns `None` if no pool is configured or the pool is exhausted.
    pub fn try_acquire_buffer(&self) -> Option<PooledBuffer> {
        self.pool.and_then(|pool| pool.try_acquire())
    }

    /// Get the buffer pool, if available.
    pub fn pool(&self) -> Option<&dyn BufferPool> {
        self.pool
    }

    /// Get the capacity of the output buffer.
    #[inline]
    pub fn capacity(&self) -> usize {
        self.capacity
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

    /// Set the presentation timestamp.
    #[inline]
    pub fn set_pts(&mut self, pts: crate::clock::ClockTime) {
        self.metadata.pts = pts;
    }

    /// Set the sequence number.
    #[inline]
    pub fn set_sequence(&mut self, seq: u64) {
        self.metadata.sequence = seq;
    }

    /// Mark as keyframe.
    #[inline]
    pub fn set_keyframe(&mut self) {
        self.metadata.flags = self
            .metadata
            .flags
            .insert(crate::metadata::BufferFlags::SYNC_POINT);
    }

    /// Mark as end of stream.
    #[inline]
    pub fn set_eos(&mut self) {
        self.metadata.flags = self
            .metadata
            .flags
            .insert(crate::metadata::BufferFlags::EOS);
    }

    /// Finalize and create a buffer with the produced data.
    ///
    /// This consumes the context and creates a buffer with the first `len` bytes.
    ///
    /// # Panics
    ///
    /// Panics if `len > capacity()` or if no buffer was provided.
    pub fn finalize(self, len: usize) -> Buffer {
        assert!(len <= self.capacity, "produced length exceeds capacity");
        let slot = self.slot.expect("cannot finalize without buffer");
        let handle = MemoryHandle::with_len(slot, len);
        Buffer::new(handle, self.metadata)
    }

    /// Take the metadata (for use when source provides own buffer).
    pub fn take_metadata(self) -> Metadata {
        self.metadata
    }
}

impl std::fmt::Debug for ProduceContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProduceContext")
            .field("has_buffer", &self.has_buffer())
            .field("capacity", &self.capacity)
            .field("metadata", &self.metadata)
            .finish()
    }
}

// ============================================================================
// ConsumeContext - context for sinks
// ============================================================================

/// Context provided to sinks for consuming data.
///
/// Provides read-only access to the buffer data and metadata.
///
/// # Usage
///
/// ```rust,ignore
/// fn consume(&mut self, ctx: &ConsumeContext) -> Result<()> {
///     let data = ctx.input();
///     let meta = ctx.metadata();
///
///     // Write to destination
///     self.writer.write_all(data)?;
///
///     // Log timestamp
///     if meta.pts.is_some() {
///         tracing::debug!(pts = %meta.pts, "wrote buffer");
///     }
///
///     Ok(())
/// }
/// ```
pub struct ConsumeContext<'a> {
    /// The buffer being consumed.
    buffer: &'a Buffer,
}

impl<'a> ConsumeContext<'a> {
    /// Create a new consume context.
    pub fn new(buffer: &'a Buffer) -> Self {
        Self { buffer }
    }

    /// Get the input data.
    #[inline]
    pub fn input(&self) -> &[u8] {
        self.buffer.as_bytes()
    }

    /// Get the buffer length.
    #[inline]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Check if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Get a reference to the metadata.
    #[inline]
    pub fn metadata(&self) -> &Metadata {
        self.buffer.metadata()
    }

    /// Get the sequence number.
    #[inline]
    pub fn sequence(&self) -> u64 {
        self.buffer.metadata().sequence
    }

    /// Check if this is end of stream.
    #[inline]
    pub fn is_eos(&self) -> bool {
        self.buffer.metadata().is_eos()
    }

    /// Check if this is a keyframe.
    #[inline]
    pub fn is_keyframe(&self) -> bool {
        self.buffer.metadata().is_keyframe()
    }

    /// Get the underlying buffer reference.
    ///
    /// Use this when you need full buffer access (e.g., for cloning).
    #[inline]
    pub fn buffer(&self) -> &Buffer {
        self.buffer
    }
}

impl std::fmt::Debug for ConsumeContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConsumeContext")
            .field("len", &self.len())
            .field("sequence", &self.sequence())
            .field("is_eos", &self.is_eos())
            .finish()
    }
}

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
/// access to shared resources like memory arenas and configuration.
#[derive(Clone)]
pub struct ElementContext {
    /// Name of this element instance.
    name: String,
    /// Optional shared arena for buffer allocation.
    arena: Option<crate::memory::SharedArena>,
}

impl ElementContext {
    /// Create a new element context.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            arena: None,
        }
    }

    /// Create a context with a shared arena.
    pub fn with_arena(name: impl Into<String>, arena: crate::memory::SharedArena) -> Self {
        Self {
            name: name.into(),
            arena: Some(arena),
        }
    }

    /// Get the element's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the shared arena, if one is configured.
    pub fn arena(&self) -> Option<&crate::memory::SharedArena> {
        self.arena.as_ref()
    }

    /// Set the shared arena.
    pub fn set_arena(&mut self, arena: crate::memory::SharedArena) {
        self.arena = Some(arena);
    }
}

impl std::fmt::Debug for ElementContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ElementContext")
            .field("name", &self.name)
            .field("has_arena", &self.arena.is_some())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_context_creation() {
        let ctx = ElementContext::new("test-element");
        assert_eq!(ctx.name(), "test-element");
        assert!(ctx.arena().is_none());
    }

    #[test]
    fn test_context_with_arena() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let arena_id = arena.id();

        let ctx = ElementContext::with_arena("test-element", arena);
        assert!(ctx.arena().is_some());
        assert_eq!(ctx.arena().unwrap().id(), arena_id);
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

    // ProduceContext tests

    use crate::memory::SharedArena;

    #[test]
    fn test_produce_context_with_buffer() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();

        let mut ctx = ProduceContext::new(slot);

        assert!(ctx.has_buffer());
        assert_eq!(ctx.capacity(), 1024);

        // Write some data
        let output = ctx.output();
        output[..5].copy_from_slice(b"hello");

        // Set metadata
        ctx.set_sequence(42);
        ctx.set_keyframe();

        // Finalize
        let buffer = ctx.finalize(5);
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_bytes(), b"hello");
        assert_eq!(buffer.metadata().sequence, 42);
        assert!(buffer.metadata().is_keyframe());
    }

    #[test]
    fn test_produce_context_without_buffer() {
        let ctx = ProduceContext::without_buffer();

        assert!(!ctx.has_buffer());
        assert_eq!(ctx.capacity(), 0);
    }

    #[test]
    #[should_panic(expected = "ProduceContext has no buffer")]
    fn test_produce_context_no_buffer_panic() {
        let mut ctx = ProduceContext::without_buffer();
        let _ = ctx.output(); // Should panic
    }

    #[test]
    #[should_panic(expected = "produced length exceeds capacity")]
    fn test_produce_context_finalize_overflow() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();

        let ctx = ProduceContext::new(slot);
        let _ = ctx.finalize(2048); // Should panic
    }

    #[test]
    fn test_produce_context_take_metadata() {
        let ctx = ProduceContext::without_buffer();
        let meta = ctx.take_metadata();
        assert_eq!(meta.sequence, 0);
    }

    #[test]
    fn test_produce_result_variants() {
        let result = ProduceResult::Produced(100);
        assert!(result.is_produced());
        assert!(!result.is_eos());
        assert!(!result.is_would_block());

        let result = ProduceResult::Eos;
        assert!(!result.is_produced());
        assert!(result.is_eos());
        assert!(!result.is_would_block());

        let result = ProduceResult::WouldBlock;
        assert!(!result.is_produced());
        assert!(!result.is_eos());
        assert!(result.is_would_block());
    }

    // ProduceContext pool integration tests

    use crate::memory::FixedBufferPool;

    #[test]
    fn test_produce_context_with_pool() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();

        let ctx = ProduceContext::with_pool(slot, pool.as_ref());

        assert!(ctx.has_buffer());
        assert!(ctx.has_pool());
        assert!(ctx.pool().is_some());
    }

    #[test]
    fn test_produce_context_pool_only() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();
        let ctx = ProduceContext::with_pool_only(pool.as_ref());

        assert!(!ctx.has_buffer());
        assert!(ctx.has_pool());
    }

    #[test]
    fn test_produce_context_acquire_buffer() {
        let pool = FixedBufferPool::new(1024, 4).unwrap();
        let ctx = ProduceContext::with_pool_only(pool.as_ref());

        let mut pooled = ctx.acquire_buffer().unwrap();
        pooled.data_mut()[..5].copy_from_slice(b"hello");
        pooled.set_len(5);
        pooled.metadata_mut().sequence = 42;

        let buffer = pooled.into_buffer();
        assert_eq!(buffer.len(), 5);
        assert_eq!(buffer.as_bytes(), b"hello");
        assert_eq!(buffer.metadata().sequence, 42);
    }

    #[test]
    fn test_produce_context_try_acquire_buffer() {
        let pool = FixedBufferPool::new(1024, 2).unwrap();
        let ctx = ProduceContext::with_pool_only(pool.as_ref());

        // Acquire both buffers
        let _buf1 = ctx.try_acquire_buffer().unwrap();
        let _buf2 = ctx.try_acquire_buffer().unwrap();

        // Pool exhausted
        assert!(ctx.try_acquire_buffer().is_none());
    }

    #[test]
    fn test_produce_context_no_pool_error() {
        let ctx = ProduceContext::without_buffer();

        assert!(!ctx.has_pool());
        assert!(ctx.acquire_buffer().is_err());
        assert!(ctx.try_acquire_buffer().is_none());
    }

    // ConsumeContext tests

    #[test]
    fn test_consume_context() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let mut slot = arena.acquire().unwrap();
        // Write some data
        slot.data_mut()[..11].copy_from_slice(b"hello world");
        let handle = MemoryHandle::with_len(slot, 11);
        let meta = Metadata::from_sequence(42).keyframe();
        let buffer = Buffer::new(handle, meta);

        let ctx = ConsumeContext::new(&buffer);

        assert_eq!(ctx.len(), 11);
        assert!(!ctx.is_empty());
        assert_eq!(ctx.input(), b"hello world");
        assert_eq!(ctx.sequence(), 42);
        assert!(ctx.is_keyframe());
        assert!(!ctx.is_eos());

        // Access underlying buffer
        assert_eq!(ctx.buffer().len(), 11);
    }

    #[test]
    fn test_consume_context_eos() {
        let arena = SharedArena::new(1024, 4).unwrap();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 8);
        let meta = Metadata::new().eos();
        let buffer = Buffer::new(handle, meta);

        let ctx = ConsumeContext::new(&buffer);
        assert!(ctx.is_eos());
    }
}
