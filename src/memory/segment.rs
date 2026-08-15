//! Memory-type vocabulary for caps negotiation.
//!
//! The old `MemorySegment` trait, `IpcHandle`, and the mapped-file /
//! hugepage segment backends were deleted in the 2026-08 dead-surface
//! sweep — an abstraction layer with no polymorphic consumer. What
//! remains is the [`MemoryType`] enum, which `MemoryCaps` negotiation and
//! `Buffer::memory_type` speak.

/// Type of memory backing a segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MemoryType {
    /// Unified CPU memory (memfd-backed, always IPC-ready).
    ///
    /// This is the primary memory type for Parallax. It has zero overhead
    /// vs malloc but is always shareable via fd passing.
    Cpu,
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
        match self {
            MemoryType::Cpu => true,
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
        match self {
            MemoryType::Cpu => true,
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
        !matches!(self, MemoryType::GpuDevice)
    }
}
