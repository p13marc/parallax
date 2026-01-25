//! Built-in pipeline elements.
//!
//! This module provides common elements that are useful in most pipelines:
//!
//! ## Sources
//! - [`FileSrc`]: Reads buffers from a file
//! - [`TcpSrc`]: Reads buffers from a TCP connection
//! - [`UdpSrc`]: Reads datagrams from a UDP socket
//! - [`FdSrc`]: Reads from a raw file descriptor
//! - [`AppSrc`]: Injects buffers from application code
//! - [`DataSrc`]: Generates buffers from inline data
//! - [`TestSrc`]: Generates test pattern buffers
//! - [`NullSource`]: Produces empty buffers (useful for testing)
//!
//! ## Sinks
//! - [`FileSink`]: Writes buffers to a file
//! - [`TcpSink`]: Writes buffers to a TCP connection
//! - [`UdpSink`]: Sends datagrams to a UDP socket
//! - [`FdSink`]: Writes to a raw file descriptor
//! - [`AppSink`]: Extracts buffers to application code
//! - [`ConsoleSink`]: Prints buffers to console for debugging
//! - [`NullSink`]: Discards all buffers (useful for benchmarking)
//!
//! ## Transforms
//! - [`PassThrough`]: Passes buffers unchanged (useful for debugging/testing)
//! - [`RateLimiter`]: Limits buffer throughput rate
//! - [`Valve`]: Drops or passes buffers (on/off switch)
//! - [`Queue`]: Asynchronous buffer queue with backpressure
//!
//! ## Routing
//! - [`Tee`]: Duplicates buffers to multiple outputs (1-to-N fanout)
//! - [`Funnel`]: Merges multiple inputs into one output (N-to-1)
//! - [`InputSelector`]: Selects one of N inputs (N-to-1 switching)
//! - [`OutputSelector`]: Routes to one of N outputs (1-to-N routing)
//! - [`Concat`]: Concatenates streams sequentially
//! - [`StreamIdDemux`]: Demultiplexes by stream ID

mod appsink;
mod appsrc;
mod concat;
mod console;
mod datasrc;
mod fd;
mod file;
mod funnel;
mod null;
mod passthrough;
mod queue;
mod rate_limiter;
mod selector;
mod streamid_demux;
mod tcp;
mod tee;
mod testsrc;
mod udp;
mod valve;

// Sources
pub use appsrc::{AppSrc, AppSrcHandle, AppSrcStats};
pub use datasrc::DataSrc;
pub use fd::FdSrc;
pub use file::FileSrc;
pub use null::NullSource;
pub use tcp::{AsyncTcpSrc, TcpMode, TcpSrc};
pub use testsrc::{TestPattern, TestSrc};
pub use udp::{AsyncUdpSrc, UdpSrc};

// Sinks
pub use appsink::{AppSink, AppSinkHandle, AppSinkStats};
pub use console::{ConsoleFormat, ConsoleSink};
pub use fd::FdSink;
pub use file::FileSink;
pub use null::NullSink;
pub use tcp::{AsyncTcpSink, TcpSink};
pub use udp::{AsyncUdpSink, UdpSink};

// Transforms
pub use passthrough::PassThrough;
pub use queue::{LeakyMode, Queue, QueueStats};
pub use rate_limiter::{RateLimitMode, RateLimiter};
pub use valve::{Valve, ValveControl, ValveStats};

// Routing
pub use concat::{Concat, ConcatStats, ConcatStream};
pub use funnel::{Funnel, FunnelInput, FunnelStats};
pub use selector::{
    InputSelector, InputSelectorStats, OutputSelector, OutputSelectorStats, SelectorInput,
    SelectorOutput,
};
pub use streamid_demux::{StreamIdDemux, StreamIdDemuxStats, StreamOutput};
pub use tee::Tee;
