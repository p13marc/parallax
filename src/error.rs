//! Error types for Parallax.

use thiserror::Error;

/// Result type alias using Parallax's Error.
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Parallax operations.
#[derive(Error, Debug)]
pub enum Error {
    /// Memory pool is exhausted (no slots available).
    #[error("memory pool exhausted: no slots available")]
    PoolExhausted,

    /// Buffer pool error (no pool configured or pool unavailable).
    #[error("buffer pool error: {0}")]
    BufferPool(String),

    /// Memory allocation failed.
    #[error("memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Invalid memory segment operation.
    #[error("invalid memory segment: {0}")]
    InvalidSegment(String),

    /// Buffer validation failed (rkyv).
    #[error("buffer validation failed: {0}")]
    ValidationFailed(String),

    /// Invalid caps / type mismatch.
    #[error("invalid caps: {0}")]
    InvalidCaps(String),

    /// Configuration error.
    #[error("configuration error: {0}")]
    Config(String),

    /// Pipeline error.
    #[error("pipeline error: {0}")]
    Pipeline(String),

    /// Element error.
    #[error("element error: {0}")]
    Element(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// System call error (via rustix).
    #[error("system error: {0}")]
    System(#[from] rustix::io::Errno),

    /// Device capture/playback error.
    #[cfg(any(
        feature = "pipewire",
        feature = "libcamera",
        feature = "screen-capture",
        feature = "v4l2",
        feature = "alsa"
    ))]
    #[error("device error: {0}")]
    Device(#[from] crate::elements::device::DeviceError),
}
