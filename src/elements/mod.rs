//! Built-in pipeline elements.
//!
//! This module provides common elements that are useful in most pipelines:
//!
//! - [`PassThrough`]: Passes buffers unchanged (useful for debugging/testing)
//! - [`Tee`]: Duplicates buffers to multiple outputs (fanout)
//! - [`NullSink`]: Discards all buffers (useful for benchmarking)
//! - [`NullSource`]: Produces empty buffers (useful for testing)
//! - [`FileSrc`]: Reads buffers from a file
//! - [`FileSink`]: Writes buffers to a file

mod file;
mod null;
mod passthrough;
mod tee;

pub use file::{FileSink, FileSrc};
pub use null::{NullSink, NullSource};
pub use passthrough::PassThrough;
pub use tee::Tee;
