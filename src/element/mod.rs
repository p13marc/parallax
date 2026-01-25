//! Element system for Parallax pipelines.
//!
//! This module defines the core traits and types for pipeline elements:
//!
//! - [`Source`]: Produces buffers (e.g., file reader, network receiver)
//! - [`Sink`]: Consumes buffers (e.g., file writer, display)
//! - [`Element`]: Transforms buffers (e.g., filter, encoder)
//! - [`Transform`]: Modern transform trait with multi-output support
//! - [`Demuxer`]: Routes input to multiple output pads
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
//! # Output Types
//!
//! The [`Output`] enum represents the result of processing:
//! - `Output::None`: Buffer was filtered/dropped
//! - `Output::Single`: One output buffer
//! - `Output::Multiple`: Multiple output buffers
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::element::{Element, Source, Sink, Transform, Output};
//! use parallax::Buffer;
//!
//! // Simple 1-to-1 filter (use Element)
//! struct MyFilter;
//!
//! impl Element for MyFilter {
//!     fn process(&mut self, buffer: Buffer) -> Result<Option<Buffer>> {
//!         Ok(Some(buffer))
//!     }
//! }
//!
//! // Multi-output transform (use Transform directly)
//! struct MySplitter;
//!
//! impl Transform for MySplitter {
//!     fn transform(&mut self, buffer: Buffer) -> Result<Output> {
//!         // Split buffer into multiple outputs
//!         Ok(Output::from(vec![buffer]))
//!     }
//! }
//! ```

mod context;
mod pad;
mod traits;

pub use context::ElementContext;
pub use pad::{Pad, PadDirection, PadTemplate};
pub use traits::{
    AsyncSink, AsyncSource, AsyncTransform, Demuxer, Element, ElementAdapter, ElementDyn,
    ElementType, Output, OutputIter, PadAddedCallback, PadId, RoutedOutput, Sink, SinkAdapter,
    Source, SourceAdapter, Transform, TransformAdapter,
};
