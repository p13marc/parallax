//! Observability features: metrics and tracing.
//!
//! This module provides instrumentation for monitoring and debugging pipelines:
//!
//! - **Metrics**: Counters, gauges, and histograms via `metrics-rs`
//! - **Tracing**: Structured logging and spans via `tracing`
//!
//! ## Metrics
//!
//! Parallax exposes the following metrics:
//!
//! | Metric | Type | Description |
//! |--------|------|-------------|
//! | `parallax_buffers_produced` | Counter | Buffers produced by sources |
//! | `parallax_buffers_consumed` | Counter | Buffers consumed by sinks |
//! | `parallax_buffers_processed` | Counter | Buffers processed by transforms |
//! | `parallax_buffers_dropped` | Counter | Buffers dropped (filtered) |
//! | `parallax_bytes_produced` | Counter | Bytes produced |
//! | `parallax_bytes_consumed` | Counter | Bytes consumed |
//! | `parallax_processing_time_ns` | Histogram | Processing time per buffer |
//! | `parallax_pool_slots_available` | Gauge | Available slots in memory pool |
//! | `parallax_channel_depth` | Gauge | Messages pending in channel |
//!
//! ## Tracing
//!
//! Parallax emits spans for:
//! - Pipeline execution
//! - Element processing
//! - IPC/Network operations
//!
//! ## Example
//!
//! ```rust,ignore
//! use parallax::observability::{PipelineMetrics, init_metrics};
//!
//! // Initialize metrics (call once at startup)
//! init_metrics();
//!
//! // Metrics are automatically recorded during pipeline execution
//! // Use a metrics exporter (prometheus, statsd, etc.) to collect them
//! ```

mod metrics;
mod tracing_support;

pub use metrics::{
    ElementMetrics, PipelineMetrics, init_metrics, record_buffer_consumed, record_buffer_dropped,
    record_buffer_processed, record_buffer_produced, record_bytes_consumed, record_bytes_produced,
    record_channel_depth, record_pool_available, record_processing_time,
};
pub use tracing_support::{
    TracingConfig, instrument_element, instrument_pipeline, span_element, span_pipeline,
};
