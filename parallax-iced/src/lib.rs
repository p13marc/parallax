//! Embed parallax video in an [Iced](https://iced.rs) application.
//!
//! `AutoVideoSink` owns the process's one winit event loop, so an Iced
//! application cannot use it. Instead, put an
//! [`AppVideoSink`](parallax::elements::AppVideoSink) in the pipeline, take
//! its [`FrameMailbox`](parallax::video::FrameMailbox) before starting, and
//! draw it with a [`VideoView`] inside Iced's `shader` widget.
//!
//! ```no_run
//! use std::sync::Arc;
//!
//! use parallax::elements::AppVideoSink;
//! use parallax_iced::VideoView;
//!
//! # fn build(pipeline: &mut parallax::pipeline::Pipeline) -> parallax::error::Result<()> {
//! let sink = AppVideoSink::new()
//!     .with_sync(true)
//!     .with_host_retention(parallax_iced::HOST_RETENTION);
//! let mailbox = sink.mailbox();
//! pipeline.add_async_sink("display", sink);
//!
//! // ...and in `view()`:
//! let _video: iced::Element<'_, ()> = iced::widget::shader(VideoView::new(Arc::clone(&mailbox)))
//!     .width(iced::Fill)
//!     .height(iced::Fill)
//!     .into();
//! # Ok(())
//! # }
//! ```
//!
//! # One wgpu, by construction
//!
//! Everything here compiles against **Iced's** wgpu, reached only through the
//! `iced::wgpu` re-export — there is no direct `wgpu` dependency. Parallax's
//! own `display-gpu` backend is on a different major, and a binary that
//! enabled both would carry two. What the two share is
//! [`parallax::video`]: arithmetic, a `&'static str` of WGSL, and the golden
//! tests that pin them together.

#![warn(missing_docs)]

mod render;
mod subscription;
mod view;

pub use render::{VideoPipeline, VideoPrimitive};
pub use subscription::{FrameSignal, frames};
pub use view::{DEFAULT_MAX_LATENESS, State, VideoView};

/// Frames this renderer holds beyond the mailbox, for
/// [`AppVideoSink::with_host_retention`](parallax::elements::AppVideoSink::with_host_retention).
///
/// One for the frame on screen, plus whatever the renderer keeps in flight —
/// nothing on the upload path, because `write_texture` copies at call time.
pub const HOST_RETENTION: usize = 1 + render::RENDERER_RETENTION;
