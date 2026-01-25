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

    /// Memory allocation failed.
    #[error("memory allocation failed: {0}")]
    AllocationFailed(String),

    /// Invalid memory segment operation.
    #[error("invalid memory segment: {0}")]
    InvalidSegment(String),

    /// Buffer validation failed (rkyv).
    #[error("buffer validation failed: {0}")]
    ValidationFailed(String),

    /// I/O error.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// System call error (via rustix).
    #[error("system error: {0}")]
    System(#[from] rustix::io::Errno),
}
