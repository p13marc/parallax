//! Memory segment trait and types.

/// Type of memory backing a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MemoryType {
    /// Unified CPU memory (memfd-backed, always IPC-ready).
    ///
    /// This is the primary memory type for Parallax. It has zero overhead
    /// vs malloc but is always shareable via fd passing.
    Cpu,
    /// Regular heap memory (single-process only).
    ///
    /// **Deprecated**: Use `Cpu` instead. Heap memory cannot be shared
    /// across processes. This variant is kept for backward compatibility.
    #[deprecated(since = "0.2.0", note = "Use MemoryType::Cpu instead")]
    Heap,
    /// POSIX shared memory (memfd_create + mmap).
    ///
    /// **Deprecated**: Use `Cpu` instead. All CPU memory is now memfd-backed.
    /// This variant is kept for backward compatibility.
    #[deprecated(since = "0.2.0", note = "Use MemoryType::Cpu instead")]
    SharedMemory,
    /// Huge pages (2MB or 1GB).
    HugePages,
    /// Memory-mapped file.
    MappedFile,
    /// GPU-accessible pinned host memory.
    GpuAccessible,
    /// GPU device memory.
    GpuDevice,
    /// DMA-BUF (Linux buffer sharing, GPU-importable).
    DmaBuf,
    /// RDMA-registered memory.
    RdmaRegistered,
}

impl MemoryType {
    /// Can this memory type be shared across processes on the same machine?
    #[inline]
    pub fn supports_ipc(&self) -> bool {
        #[allow(deprecated)]
        match self {
            MemoryType::Cpu => true,
            MemoryType::Heap => false,
            MemoryType::SharedMemory => true,
            MemoryType::HugePages => true,
            MemoryType::MappedFile => true,
            MemoryType::GpuAccessible => true,
            MemoryType::GpuDevice => false, // Must export to DmaBuf first
            MemoryType::DmaBuf => true,
            MemoryType::RdmaRegistered => true,
        }
    }

    /// Can this memory type be sent over network?
    #[inline]
    pub fn supports_network(&self) -> bool {
        #[allow(deprecated)]
        match self {
            MemoryType::Cpu => true,
            MemoryType::Heap => true,
            MemoryType::SharedMemory => true,
            MemoryType::HugePages => true,
            MemoryType::MappedFile => true,
            MemoryType::GpuAccessible => true,
            MemoryType::GpuDevice => false, // Must download first
            MemoryType::DmaBuf => false,    // fd is local
            MemoryType::RdmaRegistered => true,
        }
    }

    /// Is this a CPU-accessible memory type?
    #[inline]
    pub fn is_cpu_accessible(&self) -> bool {
        #[allow(deprecated)]
        match self {
            MemoryType::GpuDevice => false,
            _ => true,
        }
    }
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
