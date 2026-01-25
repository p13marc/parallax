//! Tracing integration for structured logging and spans.

use tracing::{Level, Span, span};

/// Configuration for tracing behavior.
#[derive(Debug, Clone)]
pub struct TracingConfig {
    /// Whether to create spans for pipeline execution.
    pub pipeline_spans: bool,
    /// Whether to create spans for element processing.
    pub element_spans: bool,
    /// Whether to create spans for buffer processing.
    pub buffer_spans: bool,
    /// Default span level.
    pub level: Level,
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            pipeline_spans: true,
            element_spans: true,
            buffer_spans: false, // Can be expensive
            level: Level::INFO,
        }
    }
}

impl TracingConfig {
    /// Create a new tracing config with all spans enabled.
    pub fn all() -> Self {
        Self {
            pipeline_spans: true,
            element_spans: true,
            buffer_spans: true,
            level: Level::DEBUG,
        }
    }

    /// Create a minimal config (pipeline spans only).
    pub fn minimal() -> Self {
        Self {
            pipeline_spans: true,
            element_spans: false,
            buffer_spans: false,
            level: Level::INFO,
        }
    }

    /// Disable all spans.
    pub fn none() -> Self {
        Self {
            pipeline_spans: false,
            element_spans: false,
            buffer_spans: false,
            level: Level::INFO,
        }
    }
}

/// Create a span for pipeline execution.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::observability::span_pipeline;
///
/// let span = span_pipeline("my-pipeline");
/// let _guard = span.enter();
/// // Pipeline execution here...
/// ```
#[inline]
pub fn span_pipeline(name: &str) -> Span {
    span!(Level::INFO, "pipeline", name = %name)
}

/// Create a span for element processing.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::observability::span_element;
///
/// let span = span_element("my-pipeline", "transform1", "Transform");
/// let _guard = span.enter();
/// // Element processing here...
/// ```
#[inline]
pub fn span_element(pipeline: &str, element: &str, element_type: &str) -> Span {
    span!(
        Level::DEBUG,
        "element",
        pipeline = %pipeline,
        element = %element,
        element_type = %element_type
    )
}

/// Instrument a pipeline execution with tracing.
///
/// This is a convenience wrapper that enters a span and returns a guard.
///
/// # Example
///
/// ```rust,ignore
/// use parallax::observability::instrument_pipeline;
///
/// let _guard = instrument_pipeline("my-pipeline");
/// // Pipeline execution is automatically traced
/// ```
pub fn instrument_pipeline(name: &str) -> tracing::span::EnteredSpan {
    span_pipeline(name).entered()
}

/// Instrument an element with tracing.
///
/// This is a convenience wrapper that enters a span and returns a guard.
pub fn instrument_element(
    pipeline: &str,
    element: &str,
    element_type: &str,
) -> tracing::span::EnteredSpan {
    span_element(pipeline, element, element_type).entered()
}

/// Log a buffer being produced.
#[inline]
pub fn trace_buffer_produced(pipeline: &str, element: &str, size: usize, sequence: u64) {
    tracing::debug!(
        pipeline = %pipeline,
        element = %element,
        size = size,
        sequence = sequence,
        "buffer produced"
    );
}

/// Log a buffer being consumed.
#[inline]
pub fn trace_buffer_consumed(pipeline: &str, element: &str, size: usize, sequence: u64) {
    tracing::debug!(
        pipeline = %pipeline,
        element = %element,
        size = size,
        sequence = sequence,
        "buffer consumed"
    );
}

/// Log a buffer being processed.
#[inline]
pub fn trace_buffer_processed(pipeline: &str, element: &str, size: usize, sequence: u64) {
    tracing::trace!(
        pipeline = %pipeline,
        element = %element,
        size = size,
        sequence = sequence,
        "buffer processed"
    );
}

/// Log an error.
#[inline]
pub fn trace_error(pipeline: &str, element: &str, error: &dyn std::error::Error) {
    tracing::error!(
        pipeline = %pipeline,
        element = %element,
        error = %error,
        "processing error"
    );
}

/// Log end-of-stream.
#[inline]
pub fn trace_eos(pipeline: &str, element: &str) {
    tracing::info!(
        pipeline = %pipeline,
        element = %element,
        "end of stream"
    );
}

/// Log pipeline state change.
#[inline]
pub fn trace_state_change(pipeline: &str, from: &str, to: &str) {
    tracing::info!(
        pipeline = %pipeline,
        from = %from,
        to = %to,
        "pipeline state changed"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tracing_config_default() {
        let config = TracingConfig::default();
        assert!(config.pipeline_spans);
        assert!(config.element_spans);
        assert!(!config.buffer_spans);
    }

    #[test]
    fn test_tracing_config_all() {
        let config = TracingConfig::all();
        assert!(config.pipeline_spans);
        assert!(config.element_spans);
        assert!(config.buffer_spans);
    }

    #[test]
    fn test_tracing_config_minimal() {
        let config = TracingConfig::minimal();
        assert!(config.pipeline_spans);
        assert!(!config.element_spans);
        assert!(!config.buffer_spans);
    }

    #[test]
    fn test_tracing_config_none() {
        let config = TracingConfig::none();
        assert!(!config.pipeline_spans);
        assert!(!config.element_spans);
        assert!(!config.buffer_spans);
    }

    #[test]
    fn test_span_creation() {
        // These should not panic
        let _span = span_pipeline("test-pipeline");
        let _span = span_element("test-pipeline", "element1", "Transform");
    }

    #[test]
    fn test_instrumentation() {
        // These should not panic
        let _guard = instrument_pipeline("test-pipeline");
        let _guard = instrument_element("test-pipeline", "element1", "Source");
    }

    #[test]
    fn test_trace_functions() {
        // These should not panic even without a subscriber
        trace_buffer_produced("test", "src", 100, 0);
        trace_buffer_consumed("test", "sink", 100, 0);
        trace_buffer_processed("test", "transform", 100, 0);
        trace_eos("test", "src");
        trace_state_change("test", "Stopped", "Playing");
    }
}
