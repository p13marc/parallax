//! Application integration elements.
//!
//! - [`AppSrc`]: Inject buffers from application code
//! - [`AppSink`]: Extract buffers to application code
//! - `AutoVideoSink`: Display video in a window (requires `display` feature)

mod appsink;
mod appsrc;

#[cfg(feature = "display")]
mod autovideosink;
#[cfg(feature = "display")]
mod present;

/// Re-exported from [`crate::pipeline`], where the terminal outcome now lives —
/// it is the pipeline's answer as much as a sink's.
pub use crate::pipeline::EndReason;
pub use appsink::{AppSink, AppSinkHandle, AppSinkStats, Pulled};
pub use appsrc::{AppSrc, AppSrcHandle, AppSrcStats};

#[cfg(feature = "display")]
pub use autovideosink::{AutoVideoSink, AutoVideoSinkHandle, VideoKey, VideoWindowEvent};
#[cfg(feature = "display")]
pub use present::{gpu_dmabuf_import_available, gpu_present_available};
