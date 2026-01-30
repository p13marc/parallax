//! Data transformation elements.
//!
//! ## Generic Transforms
//! - [`Map`]: Transform buffer contents
//! - [`FilterMap`]: Transform and optionally filter
//! - [`FlatMap`]: One-to-many transformation
//! - [`Chunk`]: Split buffers into fixed-size chunks
//!
//! ## Video Processing
//! - [`VideoScale`]: Scale/resize YUV420 video frames
//! - [`VideoConvertElement`]: Convert between pixel formats (YUYV -> RGBA, etc.)
//!
//! ## Audio Processing
//! - [`AudioConvertElement`]: Convert between sample formats (S16 -> F32, etc.)
//! - [`AudioResampleElement`]: Convert between sample rates (48kHz -> 44.1kHz)
//!
//! ## Batching
//! - [`Batch`]: Combine multiple buffers into one
//! - [`Unbatch`]: Split one buffer into many
//!
//! ## Filtering
//! - [`Filter`]: Generic predicate-based filter
//! - [`SampleFilter`]: Statistical sampling
//! - [`MetadataFilter`]: Filter by metadata values
//!
//! ## Buffer Operations
//! - [`BufferTrim`]: Trim buffers to max size
//! - [`BufferSlice`]: Extract slice from buffer
//! - [`BufferPad`]: Pad buffers to min size
//!
//! ## Metadata Operations
//! - [`SequenceNumber`]: Add sequence numbers
//! - [`Timestamper`]: Add timestamps
//! - [`MetadataInject`]: Inject custom metadata
//!
//! ## Data Processing
//! - [`DuplicateFilter`]: Remove duplicates by content hash
//! - [`RangeFilter`]: Filter by size/sequence range
//! - [`RegexFilter`]: Filter by regex pattern
//! - [`MetadataExtract`]: Extract metadata to sideband
//! - [`BufferSplit`]: Split at delimiter boundaries
//! - [`BufferJoin`]: Join with delimiter
//! - [`BufferConcat`]: Concatenate buffer contents

mod audioconvert;
mod audioresample;
mod batch;
mod buffer_ops;
mod data_processing;
mod filter;
mod generic;
mod metadata_ops;
mod scale;
mod videoconvert;

// Generic transforms
pub use generic::{Chunk, FilterMap, FlatMap, Map};

// Video processing
pub use scale::{ScaleMode, VideoScale};
pub use videoconvert::VideoConvertElement;

// Audio processing
pub use audioconvert::AudioConvertElement;
pub use audioresample::AudioResampleElement;

// Batching
pub use batch::{Batch, BatchStats, Unbatch, UnbatchStats};

// Filtering
pub use filter::{Filter, FilterStats, MetadataFilter, SampleFilter, SampleMode};

// Buffer operations
pub use buffer_ops::{BufferPad, BufferPadStats, BufferSlice, BufferTrim, BufferTrimStats};

// Metadata operations
pub use metadata_ops::{
    MetadataInject, SequenceNumber, TimestampDebug, TimestampDebugLevel, TimestampDebugStats,
    TimestampFormat, TimestampMode, Timestamper,
};

// Data processing
pub use data_processing::{
    BufferConcat, BufferConcatStats, BufferJoin, BufferJoinStats, BufferSplit, BufferSplitStats,
    DuplicateFilter, DuplicateFilterStats, ExtractedMetadata, MetadataExtract, RangeFilter,
    RangeFilterStats, RegexFilter, RegexFilterStats,
};
