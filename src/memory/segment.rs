//! Memory segment trait and types.

/// Type of memory backing a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryType {
    /// Regular heap memory (single-process only).
    Heap,
    /// POSIX shared memory (memfd_create + mmap).
    SharedMemory,
    /// Huge pages (2MB or 1GB).
    HugePages,
    /// Memory-mapped file.
    MappedFile,
    /// GPU-accessible pinned host memory.
    GpuAccessible,
    /// GPU device memory.
    GpuDevice,
    /// RDMA-registered memory.
    RdmaRegistered,
}

/// Handle for sharing memory across processes.
///
/// This can be serialized and sent to another process, which can then
/// open the same memory region.
#[derive(Debug, Clone)]
pub enum IpcHandle {
    /// File descriptor (for memfd or shm_open).
    /// The fd should be sent via SCM_RIGHTS over a Unix socket.
    Fd {
        /// The raw file descriptor.
        fd: std::os::unix::io::RawFd,
        /// Size of the memory region.
        size: usize,
    },
    /// Named shared memory segment.
    Named {
        /// Name of the shared memory segment.
        name: String,
        /// Size of the memory region.
        size: usize,
    },
}

/// Trait for memory segment backends.
///
/// A memory segment represents a contiguous region of memory that can be
/// used for buffer storage. Different implementations provide different
/// capabilities (heap-only, shared between processes, GPU-accessible, etc.).
///
/// # Safety
///
/// Implementations must ensure that:
/// - Pointers remain valid for the lifetime of the segment
/// - Thread-safety requirements are met (Send + Sync)
/// - Memory is properly aligned for the intended use
pub trait MemorySegment: Send + Sync {
    /// Get a raw pointer to the start of this segment.
    fn as_ptr(&self) -> *const u8;

    /// Get a mutable pointer to the start of this segment.
    ///
    /// Returns `None` if the segment is read-only or shared with readers.
    fn as_mut_ptr(&self) -> Option<*mut u8>;

    /// Total size of the segment in bytes.
    fn len(&self) -> usize;

    /// Returns true if the segment has zero length.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The type of memory backing this segment.
    fn memory_type(&self) -> MemoryType;

    /// Get an IPC handle for sharing this segment with other processes.
    ///
    /// Returns `None` if this segment type doesn't support cross-process sharing.
    fn ipc_handle(&self) -> Option<IpcHandle>;

    /// Get the segment as a byte slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure no mutable references exist to this memory.
    unsafe fn as_slice(&self) -> &[u8] {
        // SAFETY: Caller guarantees no mutable references exist.
        unsafe { std::slice::from_raw_parts(self.as_ptr(), self.len()) }
    }

    /// Get the segment as a mutable byte slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure exclusive access to this memory.
    /// This returns a mutable reference from `&self` because the underlying
    /// memory may be mutable even when the segment handle is shared (e.g.,
    /// for memory-mapped regions). Callers must ensure proper synchronization.
    #[allow(clippy::mut_from_ref)]
    unsafe fn as_mut_slice(&self) -> Option<&mut [u8]> {
        // SAFETY: Caller guarantees exclusive access.
        self.as_mut_ptr()
            .map(|ptr| unsafe { std::slice::from_raw_parts_mut(ptr, self.len()) })
    }
}

/// Extension trait for Arc<dyn MemorySegment>.
impl dyn MemorySegment {
    /// Check if this segment can be shared with other processes.
    pub fn is_shareable(&self) -> bool {
        self.ipc_handle().is_some()
    }
}
