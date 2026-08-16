//! Error types for Parallax.

use thiserror::Error;

/// Result type alias using Parallax's Error.
pub type Result<T> = std::result::Result<T, Error>;

/// Main error type for Parallax operations.
#[derive(Error, Debug)]
#[non_exhaustive]
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

    /// Pipeline-description parse or element-construction error.
    #[error("{0}")]
    Parse(String),

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

    /// An element or pipeline thread panicked.
    ///
    /// Distinct from [`Element`](Self::Element) on purpose: a returned error is
    /// a runtime condition an element anticipated, while a panic means a broken
    /// invariant — a bug, not a bad input. Operators alert on them differently,
    /// and a supervisor that retries the former should usually not retry this.
    ///
    /// `node` is the element the panic came from, which is the part a bare
    /// `JoinError` cannot tell you.
    #[error("element '{node}' panicked: {message}")]
    Panic {
        /// The element whose task panicked.
        node: String,
        /// The panic payload, when it was a string.
        message: String,
    },

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
        feature = "alsa",
        feature = "v4l2-m2m"
    ))]
    #[error("device error: {0}")]
    Device(#[from] crate::elements::device::DeviceError),
}

/// Recover the message from a caught panic payload.
///
/// `panic!("...")` and `panic!("{}", x)` produce `&'static str` and `String`
/// respectively; anything else (a `panic_any` with a custom type) has no text
/// to show, and saying so beats an empty message.
pub fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&'static str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "panicked with a non-string payload".to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn panic_messages_come_back_from_both_string_forms() {
        let literal = std::panic::catch_unwind(|| panic!("a literal")).unwrap_err();
        assert_eq!(panic_message(literal.as_ref()), "a literal");

        let n = 7;
        let formatted = std::panic::catch_unwind(|| panic!("formatted {n}")).unwrap_err();
        assert_eq!(panic_message(formatted.as_ref()), "formatted 7");
    }

    #[test]
    fn an_exotic_payload_says_so_rather_than_going_blank() {
        let odd = std::panic::catch_unwind(|| std::panic::panic_any(42u8)).unwrap_err();
        assert_eq!(
            panic_message(odd.as_ref()),
            "panicked with a non-string payload"
        );
    }

    #[test]
    fn a_panic_error_names_the_element() {
        let err = Error::Panic {
            node: "encoder".into(),
            message: "index out of bounds".into(),
        };
        assert_eq!(
            err.to_string(),
            "element 'encoder' panicked: index out of bounds"
        );
    }
}
