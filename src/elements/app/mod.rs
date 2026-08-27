//! Application integration elements.
//!
//! - [`AppSrc`]: Inject buffers from application code
//! - [`AppSink`]: Extract buffers to application code
//! - [`AppVideoSink`]: Hand paced video frames to a GUI toolkit for rendering
//! - `AutoVideoSink`: Display video in a window (requires `display` feature)

mod appsink;
mod appsrc;
mod appvideosink;

#[cfg(feature = "display")]
mod autovideosink;
#[cfg(feature = "display")]
mod present;

/// Re-exported from [`crate::pipeline`], where the terminal outcome now lives —
/// it is the pipeline's answer as much as a sink's.
pub use crate::pipeline::EndReason;
pub use appsink::{AppSink, AppSinkHandle, AppSinkStats, Pulled};
pub use appsrc::{AppSrc, AppSrcHandle, AppSrcStats};
pub use appvideosink::AppVideoSink;

#[cfg(feature = "display")]
pub use autovideosink::{AutoVideoSink, AutoVideoSinkHandle, VideoKey, VideoWindowEvent};
#[cfg(feature = "display")]
pub use present::{gpu_dmabuf_import_available, gpu_present_available};
