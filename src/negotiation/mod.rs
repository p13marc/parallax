//! Caps negotiation for pipelines.
//!
//! This module provides global constraint-based format and memory negotiation.
//! Instead of negotiating link-by-link, the solver considers all constraints
//! simultaneously to find an optimal solution.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    NegotiationSolver                            │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  1. Collect constraints from all elements                       │
//! │  2. Propagate constraints through the graph                     │
//! │  3. Intersect constraints at each link                          │
//! │  4. Fixate to concrete formats                                  │
//! │  5. Insert converters where needed                              │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::negotiation::{NegotiationSolver, NegotiationResult};
//! use parallax::pipeline::Pipeline;
//!
//! let pipeline = Pipeline::new();
//! // ... add elements and links ...
//!
//! let solver = NegotiationSolver::new(&pipeline);
//! let result = solver.solve()?;
//!
//! // Result contains:
//! // - Negotiated format for each link
//! // - Negotiated memory type for each link
//! // - Converters to insert (if any)
//! ```

mod builtin;
mod converters;
mod error;
mod solver;

pub use builtin::{
    AudioConvert, AudioResample, Identity, MemoryCopy, ScaleAlgorithm, VideoConvert, VideoScale,
    builtin_registry,
};
pub use converters::{
    ConverterElement, ConverterFactory, ConverterInfo, ConverterRegistry, FormatType,
};
pub use error::NegotiationError;
pub use solver::{
    ConverterInsertion, ElementCaps, LinkInfo, LinkNegotiation, NegotiationResult,
    NegotiationSolver,
};
