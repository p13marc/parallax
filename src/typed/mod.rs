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

pub mod bridge;
mod element;
mod operators;
mod pipeline;

pub use element::*;
pub use operators::*;
pub use pipeline::*;
