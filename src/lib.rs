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
//! - **Zero-copy buffers**: Shared memory with loan-based memory pools
//! - **Progressive typing**: Start dynamic, graduate to typed
//! - **Multi-process pipelines**: memfd + Unix socket IPC
//! - **rkyv serialization**: Zero-copy deserialization at boundaries
//! - **Linux-optimized**: memfd_create, huge pages, future io_uring
//!
//! ## Quick Start
//!
//! ```rust,ignore
//! use parallax::prelude::*;
//!
//! // Dynamic pipeline (string-based)
//! let mut pipeline = Pipeline::new();
//! pipeline.parse("filesrc location=input.bin ! passthrough ! consolesink")?;
//! pipeline.play().await?;
//!
//! // Typed pipeline (compile-time checked)
//! let pipeline = TypedPipeline::from_source(FileSrc::new("input.bin"))
//!     .then(PassThrough::new())
//!     .sink(ConsoleSink::new());
//! pipeline.run().await?;
//! ```

#![warn(missing_docs)]
#![warn(clippy::all)]
#![deny(unsafe_op_in_unsafe_fn)]

pub mod buffer;
pub mod clock;
pub mod element;
pub mod elements;
pub mod error;
pub mod format;
pub mod link;
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
    pub use crate::clock::{Clock, ClockTime, PipelineClock};
    pub use crate::element::{
        AsyncElementDyn, AsyncSink, AsyncSource, AsyncTransform, DynAsyncElement, Element, Output,
        Sink, Source, Transform,
    };
    pub use crate::error::{Error, Result};
    pub use crate::format::{AudioFormat, Caps, MediaFormat, RtpFormat, VideoFormat};
    pub use crate::memory::{MemoryPool, MemorySegment, MemoryType};
    pub use crate::metadata::{BufferFlags, Metadata, RtpMeta};
    pub use crate::pipeline::{Pipeline, PipelineExecutor};
}

pub use error::{Error, Result};
