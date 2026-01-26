//! Negotiation error types.

use thiserror::Error;

/// Error during caps negotiation.
#[derive(Debug, Error)]
pub enum NegotiationError {
    /// No common format between connected elements.
    #[error("No common format between {upstream} and {downstream}:\n  {explanation}")]
    NoCommonFormat {
        /// Name of upstream element.
        upstream: String,
        /// Name of downstream element.
        downstream: String,
        /// Detailed explanation.
        explanation: String,
    },

    /// No common memory type between connected elements.
    #[error("No common memory type between {upstream} and {downstream}:\n  {explanation}")]
    NoCommonMemory {
        /// Name of upstream element.
        upstream: String,
        /// Name of downstream element.
        downstream: String,
        /// Detailed explanation.
        explanation: String,
    },

    /// Cannot fixate constraints (e.g., both ends are "Any").
    #[error("Cannot fixate format for link {link_id}: {reason}")]
    CannotFixate {
        /// Link identifier.
        link_id: usize,
        /// Reason for failure.
        reason: String,
    },

    /// No converter available for required conversion.
    #[error("No converter from {from_format} to {to_format}")]
    NoConverter {
        /// Source format description.
        from_format: String,
        /// Target format description.
        to_format: String,
    },

    /// Cycle detected in pipeline graph.
    #[error("Cycle detected in pipeline graph")]
    CycleDetected,

    /// Element not found in pipeline.
    #[error("Element not found: {name}")]
    ElementNotFound {
        /// Element name.
        name: String,
    },

    /// Link not found in pipeline.
    #[error("Link not found: {link_id}")]
    LinkNotFound {
        /// Link identifier.
        link_id: usize,
    },

    /// Internal error.
    #[error("Internal negotiation error: {0}")]
    Internal(String),
}

impl NegotiationError {
    /// Create a "no common format" error with suggestions.
    pub fn no_common_format(
        upstream: impl Into<String>,
        downstream: impl Into<String>,
        upstream_caps: &str,
        downstream_caps: &str,
    ) -> Self {
        Self::NoCommonFormat {
            upstream: upstream.into(),
            downstream: downstream.into(),
            explanation: format!(
                "Upstream produces: {}\nDownstream accepts: {}\nSuggestion: Insert a format converter",
                upstream_caps, downstream_caps
            ),
        }
    }

    /// Create a "no common memory" error with suggestions.
    pub fn no_common_memory(
        upstream: impl Into<String>,
        downstream: impl Into<String>,
        upstream_mem: &str,
        downstream_mem: &str,
    ) -> Self {
        Self::NoCommonMemory {
            upstream: upstream.into(),
            downstream: downstream.into(),
            explanation: format!(
                "Upstream memory: {}\nDownstream memory: {}\nSuggestion: Insert a memory converter (e.g., GPU upload/download)",
                upstream_mem, downstream_mem
            ),
        }
    }
}
