//! File and descriptor I/O elements.
//!
//! - [`FileSrc`], [`FileSink`]: File read/write
//! - [`FdSrc`], [`FdSink`]: Raw file descriptor I/O
//! - [`ConsoleSink`]: Debug output to console

mod console;
mod fd;
mod file;

pub use console::{ConsoleFormat, ConsoleSink};
pub use fd::{FdSink, FdSrc};
pub use file::{FileSink, FileSrc};
