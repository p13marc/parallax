//! Element system for Parallax pipelines.
//!
//! This module defines the core traits and types for pipeline elements:
//!
//! - [`Source`]: Produces buffers (e.g., file reader, network receiver)
//! - [`Sink`]: Consumes buffers (e.g., file writer, display)
//! - [`Element`]: Transforms buffers (e.g., filter, encoder)
//!
//! # Design
//!
//! Elements follow the "sync processing, async orchestration" principle:
//! - The `process`/`produce`/`consume` methods are **synchronous**
//! - The pipeline executor handles async scheduling via channels
//!
//! This keeps element implementations simple and deterministic while
//! allowing the pipeline to handle backpressure and concurrency.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::element::{Element, Source, Sink};
//! use parallax::Buffer;
//!
//! struct MyFilter;
//!
//! impl Element for MyFilter {
//!     fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
//!         // Transform the buffer
//!         Ok(Some(buffer))
//!     }
//! }
//! ```

mod context;
mod pad;
mod traits;

pub use context::ElementContext;
pub use pad::{Pad, PadDirection, PadTemplate};
pub use traits::{
    AsyncSource, Element, ElementAdapter, ElementDyn, ElementType, Sink, SinkAdapter, Source,
    SourceAdapter,
};
