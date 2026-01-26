//! Test and utility source/sink elements.
//!
//! - [`TestSrc`]: Generates test pattern buffers
//! - [`VideoTestSrc`], [`AsyncVideoTestSrc`]: Generates video test patterns
//! - [`DataSrc`]: Generates buffers from inline data
//! - [`NullSource`], [`NullSink`]: Null elements for testing/benchmarking

mod datasrc;
mod null;
mod testsrc;
mod videotestsrc;

pub use datasrc::DataSrc;
pub use null::{NullSink, NullSource};
pub use testsrc::{TestPattern, TestSrc};
pub use videotestsrc::{AsyncVideoTestSrc, PixelFormat, VideoPattern, VideoTestSrc};
