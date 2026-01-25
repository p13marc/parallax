//! Typed pipeline API with compile-time type safety.
//!
//! This module provides a type-safe pipeline builder that validates
//! element connections at compile time. Types flow through the pipeline,
//! ensuring that output types match input types.
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::typed::*;
//!
//! // Define typed elements
//! let pipeline = source::<u32>()
//!     .then(map(|x| x * 2))
//!     .then(sink());
//!
//! // Or use the >> operator
//! let pipeline = source::<u32>() >> map(|x| x * 2) >> sink();
//!
//! pipeline.run().await?;
//! ```
//!
//! # Multi-source Operations
//!
//! ```rust,ignore
//! use parallax::typed::*;
//!
//! // Merge two sources
//! let merged = merge(source1, source2);
//!
//! // Zip two sources
//! let zipped = zip(left, right);
//!
//! // Temporal join with timestamp alignment
//! let joined = temporal_join(sensor1, sensor2, tolerance, get_ts1, get_ts2);
//! ```

pub mod bridge;
mod element;
mod multi_source;
mod operators;
mod pipeline;

pub use element::*;
pub use multi_source::*;
pub use operators::*;
pub use pipeline::*;
