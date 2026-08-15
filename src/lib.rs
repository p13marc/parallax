//! # Parallax
//!
//! A Rust-native streaming pipeline engine with zero-copy multi-process support.
//!
//! Parallax provides both dynamic (runtime-configured) and typed (compile-time safe)
//! pipeline construction, with buffers backed by shared memory for efficient
//! multi-process communication.
//!
//! ## Features
//!
//! - **Zero-copy buffers**: memfd-backed arenas with cross-process reference counting
//! - **Progressive typing**: Start dynamic, graduate to typed
//! - **Multi-process pipelines**: memfd + Unix socket IPC (SCM_RIGHTS)
//! - **Hybrid scheduling**: Tokio tasks + dedicated real-time threads
//! - **rkyv serialization**: Zero-copy deserialization at network boundaries
//! - **Linux-only**: memfd_create, huge pages, eventfd, DMA-BUF
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use parallax::prelude::*;
//! use parallax::typed::{pipeline, from_iter, map, filter, collect};
//!
//! // Dynamic pipeline (string syntax)
//! let mut pipeline = Pipeline::parse("videotestsrc num-buffers=60 ! videoconvert ! nullsink")?;
//! pipeline.run().await?;
//!
//! // Dynamic pipeline (programmatic)
//! let mut pipeline = Pipeline::new();
//! let src = pipeline.add_source("src", my_source);   // impl Source
//! let sink = pipeline.add_sink("sink", my_sink);     // impl Sink
//! pipeline.link(src, sink)?;
//! pipeline.run().await?;
//!
//! // Typed pipeline (compile-time checked)
//! let source = from_iter(vec![1, 2, 3, 4, 5]);
//! let result = pipeline(source)
//!     .then(filter(|x: &i32| x % 2 == 0))
//!     .then(map(|x: i32| x * 10))
//!     .sink(collect::<i32>())
//!     .run()?
//!     .into_inner();
//! // result: [20, 40]
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_op_in_unsafe_fn)]
// Allow these clippy lints that are intentional design choices
#![allow(clippy::too_many_arguments)] // Complex functions in scheduler/executor
#![allow(clippy::type_complexity)] // Complex types in typed pipelines and executor returns
#![allow(clippy::result_large_err)] // Buffer returned in Err for ring buffer full case

pub mod buffer;
pub mod clock;
pub mod codec;
pub mod control;
pub mod converters;
pub mod element;
pub mod elements;
pub mod error;
pub mod event;
pub mod format;
pub mod gpu;
pub mod memory;
pub mod metadata;
pub mod negotiation;
pub mod observability;
pub mod pipeline;
pub mod plugin;
pub mod temporal;
pub mod typed;

/// Prelude for convenient imports
pub mod prelude {
    pub use crate::buffer::Buffer;
    pub use crate::clock::{
        Clock, ClockFlags, ClockProvider, ClockTime, PausableClock, PipelineClock, SystemClock,
    };
    // Runtime control. `Controllable` is what makes `element.control()` resolve,
    // so it belongs in the prelude of a crate whose headline feature is changing
    // parameters on a running pipeline. See [`crate::control`] for the
    // clone-before-`start()` invariant that governs every handle here.
    pub use crate::control::{
        Controllable, EncoderControl, EncoderParams, EncoderStats, EncoderStatsHandle,
        KeyframeHandle, RateControlMode, ScaleControl, ThrottleControl, ValveControl,
    };
    pub use crate::element::{
        AsyncElementDyn, AsyncSink, AsyncSource, AsyncTransform, DynAsyncElement, Element, Output,
        Sink, Source, Transform,
    };
    pub use crate::error::{Error, Result};
    pub use crate::event::{Event, PipelineItem, TagList};
    pub use crate::format::{
        AudioFormat, Caps, ElementMediaCaps, FormatMemoryCap, MediaCaps, MediaFormat, RtpFormat,
        VideoFormat,
    };
    pub use crate::memory::MemoryType;
    pub use crate::metadata::{BufferFlags, Metadata, RtpMeta};
    pub use crate::pipeline::{Executor, Pipeline};
}

pub use error::{Error, Result};
