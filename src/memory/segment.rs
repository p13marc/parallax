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
    /// Producer-owned memory pinned into the pipeline (#194): a codec's
    /// own frame allocation (e.g. a refcounted `dav1d::Picture`) wrapped
    /// in an `ExternalSlot` instead of being copied into an arena.
    /// CPU-readable but read-only, never IPC-shareable, and — unlike every
    /// other type — its byte layout is producer-defined (strided planes,
    /// described by `Metadata` plane layout), so consumers must opt in
    /// explicitly ([`MemoryType::requires_explicit_optin`]).
    External,
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
            MemoryType::External => false, // producer-local pointer, no fd
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
            MemoryType::External => false, // producer-local pointer
        }
    }

    /// Is this a CPU-accessible memory type?
    #[inline]
    pub fn is_cpu_accessible(&self) -> bool {
        !matches!(self, MemoryType::GpuDevice)
    }

    /// Must a consumer name this type in its caps before negotiation may
    /// fixate it?
    ///
    /// `Cpu` and `DmaBuf` are "transparent": their bytes are CPU-readable
    /// and their plane geometry travels in `Metadata` like everyone
    /// else's, so an element with `Caps::any()` genuinely handles them by
    /// reading `as_bytes()`. `External` memory is a raw producer pointer
    /// with no fd and no IPC identity — a byte-reading consumer would
    /// silently misinterpret it — so the solver downgrades to `Cpu`
    /// unless the sink's caps explicitly list `External` (#194).
    ///
    /// The exception is a **non-linear modifier**: a tiled allocation is
    /// readable but is not a picture, so `as_bytes()` on it is garbage to
    /// anything that has not been told the layout. That is not handled by
    /// making `DmaBuf` opt-in — it would regress every working
    /// CPU-readable producer — but by the producer: a tiled frame is
    /// emitted only onto a link that negotiated `DmaBuf` deliberately (the
    /// `set_negotiated_memory` discipline), and `memorycopy` refuses to
    /// copy one to CPU rather than copying nonsense.
    /// [`DmaBufSlot::is_linear`](crate::memory::DmaBufSlot::is_linear) is
    /// the question to ask.
    #[inline]
    pub fn requires_explicit_optin(&self) -> bool {
        matches!(self, MemoryType::External)
    }
}
