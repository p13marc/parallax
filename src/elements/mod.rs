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
//! - [`TcpSrc`]: Reads buffers from a TCP connection
//! - [`TcpSink`]: Writes buffers to a TCP connection

mod file;
mod null;
mod passthrough;
mod tcp;
mod tee;

pub use file::{FileSink, FileSrc};
pub use null::{NullSink, NullSource};
pub use passthrough::PassThrough;
pub use tcp::{AsyncTcpSink, AsyncTcpSrc, TcpMode, TcpSink, TcpSrc};
pub use tee::Tee;
