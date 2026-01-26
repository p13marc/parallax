//! Application integration elements.
//!
//! - [`AppSrc`]: Inject buffers from application code
//! - [`AppSink`]: Extract buffers to application code
//! - [`IcedVideoSink`]: Display video in Iced GUI (requires `iced-sink` feature)

mod appsink;
mod appsrc;

#[cfg(feature = "iced-sink")]
mod iced_sink;

pub use appsink::{AppSink, AppSinkHandle, AppSinkStats};
pub use appsrc::{AppSrc, AppSrcHandle, AppSrcStats};

#[cfg(feature = "iced-sink")]
pub use iced_sink::{
    IcedVideoSink, IcedVideoSinkConfig, IcedVideoSinkHandle, IcedVideoSinkStats, InputPixelFormat,
};
