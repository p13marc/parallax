//! Application integration elements.
//!
//! - [`AppSrc`]: Inject buffers from application code
//! - [`AppSink`]: Extract buffers to application code
//! - [`AutoVideoSink`]: Display video in a window (requires `display` feature)

mod appsink;
mod appsrc;

#[cfg(feature = "display")]
mod autovideosink;

pub use appsink::{AppSink, AppSinkHandle, AppSinkStats};
pub use appsrc::{AppSrc, AppSrcHandle, AppSrcStats};

#[cfg(feature = "display")]
pub use autovideosink::AutoVideoSink;
