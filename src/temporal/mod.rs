//! Temporal types for time-sensitive stream processing.
//!
//! This module provides types and utilities for working with timestamps,
//! durations, and temporal alignment of streams. Key features:
//!
//! - [`Timestamp`]: High-precision nanosecond timestamps
//! - [`TimeRange`]: Represents a time interval
//! - [`TimestampedBuffer`]: Buffer wrapper with temporal metadata
//! - Temporal alignment utilities for multi-source joins
//!
//! # Temporal Alignment
//!
//! When joining multiple streams, temporal alignment ensures that buffers
//! are matched based on their timestamps. This is critical for applications
//! like sensor fusion, where data from multiple sources must be correlated
//! in time.

mod alignment;
mod timestamp;

pub use alignment::{AlignmentStrategy, JoinResult, JoinWindow, TemporalJoin, TimestampedItem};
pub use timestamp::{ClockSource, TimeRange, Timestamp};
