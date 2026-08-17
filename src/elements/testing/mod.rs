//! Test and utility source/sink elements.
//!
//! - [`TestSrc`]: Generates test pattern buffers
//! - [`VideoTestSrc`], [`AsyncVideoTestSrc`]: Generates video test patterns
//! - [`DataSrc`]: Generates buffers from inline data
//! - [`NullSource`], [`NullSink`]: Null elements for testing/benchmarking

mod datasrc;
mod dmabufsrc;
mod externalsrc;
mod null;
mod testsrc;
mod videotestsrc;

pub use datasrc::DataSrc;
pub use dmabufsrc::DmaBufTestSrc;
pub use externalsrc::{
    ExternalTestSrc, PAD_BYTE, TEST_HEIGHT, TEST_PADDING, TEST_WIDTH, packed_reference_frame,
    strided_test_layout,
};
pub use null::{NullSink, NullSource};
pub use testsrc::{TestPattern, TestSrc};
pub use videotestsrc::{AsyncVideoTestSrc, PixelFormat, VideoPattern, VideoTestSrc};
