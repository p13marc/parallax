//! Metrics collection using metrics-rs.

use metrics::{Counter, Histogram, Unit, counter, gauge, histogram};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Whether metrics have been initialized.
static METRICS_INITIALIZED: AtomicBool = AtomicBool::new(false);

// Metric names as constants for consistency
const BUFFERS_PRODUCED: &str = "parallax_buffers_produced";
const BUFFERS_CONSUMED: &str = "parallax_buffers_consumed";
const BUFFERS_PROCESSED: &str = "parallax_buffers_processed";
const BUFFERS_DROPPED: &str = "parallax_buffers_dropped";
const BYTES_PRODUCED: &str = "parallax_bytes_produced";
const BYTES_CONSUMED: &str = "parallax_bytes_consumed";
const PROCESSING_TIME_NS: &str = "parallax_processing_time_ns";
const POOL_SLOTS_AVAILABLE: &str = "parallax_pool_slots_available";
const CHANNEL_DEPTH: &str = "parallax_channel_depth";

/// Initialize metrics descriptions.
///
/// Call this once at application startup before using any metrics.
/// Safe to call multiple times (subsequent calls are no-ops).
pub fn init_metrics() {
    if METRICS_INITIALIZED.swap(true, Ordering::SeqCst) {
        return; // Already initialized
    }

    // Describe all metrics
    metrics::describe_counter!(
        BUFFERS_PRODUCED,
        Unit::Count,
        "Total number of buffers produced by sources"
    );
    metrics::describe_counter!(
        BUFFERS_CONSUMED,
        Unit::Count,
        "Total number of buffers consumed by sinks"
    );
    metrics::describe_counter!(
        BUFFERS_PROCESSED,
        Unit::Count,
        "Total number of buffers processed by transforms"
    );
    metrics::describe_counter!(
        BUFFERS_DROPPED,
        Unit::Count,
        "Total number of buffers dropped (filtered)"
    );
    metrics::describe_counter!(
        BYTES_PRODUCED,
        Unit::Bytes,
        "Total bytes produced by sources"
    );
    metrics::describe_counter!(BYTES_CONSUMED, Unit::Bytes, "Total bytes consumed by sinks");
    metrics::describe_histogram!(
        PROCESSING_TIME_NS,
        Unit::Nanoseconds,
        "Time to process a single buffer"
    );
    metrics::describe_gauge!(
        POOL_SLOTS_AVAILABLE,
        Unit::Count,
        "Available slots in memory pool"
    );
    metrics::describe_gauge!(
        CHANNEL_DEPTH,
        Unit::Count,
        "Number of messages pending in channel"
    );
}

/// Record a buffer produced by a source.
#[inline]
pub fn record_buffer_produced(pipeline: &str, element: &str) {
    counter!(BUFFERS_PRODUCED, "pipeline" => pipeline.to_string(), "element" => element.to_string())
        .increment(1);
}

/// Record a buffer consumed by a sink.
#[inline]
pub fn record_buffer_consumed(pipeline: &str, element: &str) {
    counter!(BUFFERS_CONSUMED, "pipeline" => pipeline.to_string(), "element" => element.to_string())
        .increment(1);
}

/// Record a buffer processed by a transform.
#[inline]
pub fn record_buffer_processed(pipeline: &str, element: &str) {
    counter!(BUFFERS_PROCESSED, "pipeline" => pipeline.to_string(), "element" => element.to_string())
        .increment(1);
}

/// Record a buffer dropped (filtered).
#[inline]
pub fn record_buffer_dropped(pipeline: &str, element: &str) {
    counter!(BUFFERS_DROPPED, "pipeline" => pipeline.to_string(), "element" => element.to_string())
        .increment(1);
}

/// Record bytes produced.
#[inline]
pub fn record_bytes_produced(pipeline: &str, element: &str, bytes: u64) {
    counter!(BYTES_PRODUCED, "pipeline" => pipeline.to_string(), "element" => element.to_string())
        .increment(bytes);
}

/// Record bytes consumed.
#[inline]
pub fn record_bytes_consumed(pipeline: &str, element: &str, bytes: u64) {
    counter!(BYTES_CONSUMED, "pipeline" => pipeline.to_string(), "element" => element.to_string())
        .increment(bytes);
}

/// Record processing time for a buffer.
#[inline]
pub fn record_processing_time(pipeline: &str, element: &str, duration: Duration) {
    histogram!(PROCESSING_TIME_NS, "pipeline" => pipeline.to_string(), "element" => element.to_string())
        .record(duration.as_nanos() as f64);
}

/// Record available pool slots.
#[inline]
pub fn record_pool_available(pool_name: &str, available: usize) {
    gauge!(POOL_SLOTS_AVAILABLE, "pool" => pool_name.to_string()).set(available as f64);
}

/// Record channel depth (pending messages).
#[inline]
pub fn record_channel_depth(channel_name: &str, depth: usize) {
    gauge!(CHANNEL_DEPTH, "channel" => channel_name.to_string()).set(depth as f64);
}

/// Metrics collector for a specific element.
///
/// Provides a convenient way to record metrics with pre-configured labels.
#[derive(Clone)]
pub struct ElementMetrics {
    pipeline: String,
    element: String,
    buffers_in: Counter,
    buffers_out: Counter,
    bytes_in: Counter,
    bytes_out: Counter,
    processing_time: Histogram,
}

impl ElementMetrics {
    /// Create a new element metrics collector.
    pub fn new(pipeline: &str, element: &str) -> Self {
        Self {
            pipeline: pipeline.to_string(),
            element: element.to_string(),
            buffers_in: counter!(
                BUFFERS_PROCESSED,
                "pipeline" => pipeline.to_string(),
                "element" => element.to_string(),
                "direction" => "in"
            ),
            buffers_out: counter!(
                BUFFERS_PROCESSED,
                "pipeline" => pipeline.to_string(),
                "element" => element.to_string(),
                "direction" => "out"
            ),
            bytes_in: counter!(
                BYTES_CONSUMED,
                "pipeline" => pipeline.to_string(),
                "element" => element.to_string()
            ),
            bytes_out: counter!(
                BYTES_PRODUCED,
                "pipeline" => pipeline.to_string(),
                "element" => element.to_string()
            ),
            processing_time: histogram!(
                PROCESSING_TIME_NS,
                "pipeline" => pipeline.to_string(),
                "element" => element.to_string()
            ),
        }
    }

    /// Record an incoming buffer.
    #[inline]
    pub fn record_in(&self, bytes: usize) {
        self.buffers_in.increment(1);
        self.bytes_in.increment(bytes as u64);
    }

    /// Record an outgoing buffer.
    #[inline]
    pub fn record_out(&self, bytes: usize) {
        self.buffers_out.increment(1);
        self.bytes_out.increment(bytes as u64);
    }

    /// Record processing time.
    #[inline]
    pub fn record_time(&self, duration: Duration) {
        self.processing_time.record(duration.as_nanos() as f64);
    }

    /// Start a timer and return a guard that records on drop.
    pub fn start_timer(&self) -> TimerGuard<'_> {
        TimerGuard {
            start: Instant::now(),
            metrics: self,
        }
    }

    /// Get the pipeline name.
    pub fn pipeline(&self) -> &str {
        &self.pipeline
    }

    /// Get the element name.
    pub fn element(&self) -> &str {
        &self.element
    }
}

/// Guard that records processing time when dropped.
pub struct TimerGuard<'a> {
    start: Instant,
    metrics: &'a ElementMetrics,
}

impl Drop for TimerGuard<'_> {
    fn drop(&mut self) {
        self.metrics.record_time(self.start.elapsed());
    }
}

/// Metrics collector for an entire pipeline.
#[derive(Clone)]
pub struct PipelineMetrics {
    name: String,
    buffers_total: Counter,
    bytes_total: Counter,
    errors: Counter,
}

impl PipelineMetrics {
    /// Create a new pipeline metrics collector.
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            buffers_total: counter!("parallax_pipeline_buffers_total", "pipeline" => name.to_string()),
            bytes_total: counter!("parallax_pipeline_bytes_total", "pipeline" => name.to_string()),
            errors: counter!("parallax_pipeline_errors_total", "pipeline" => name.to_string()),
        }
    }

    /// Record buffers processed by this pipeline.
    #[inline]
    pub fn record_buffers(&self, count: u64) {
        self.buffers_total.increment(count);
    }

    /// Record bytes processed by this pipeline.
    #[inline]
    pub fn record_bytes(&self, bytes: u64) {
        self.bytes_total.increment(bytes);
    }

    /// Record an error in this pipeline.
    #[inline]
    pub fn record_error(&self) {
        self.errors.increment(1);
    }

    /// Get the pipeline name.
    pub fn name(&self) -> &str {
        &self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_metrics() {
        // Should not panic
        init_metrics();
        // Should be idempotent
        init_metrics();
    }

    #[test]
    fn test_element_metrics() {
        let metrics = ElementMetrics::new("test-pipeline", "test-element");

        metrics.record_in(100);
        metrics.record_out(100);
        metrics.record_time(Duration::from_micros(50));

        assert_eq!(metrics.pipeline(), "test-pipeline");
        assert_eq!(metrics.element(), "test-element");
    }

    #[test]
    fn test_timer_guard() {
        let metrics = ElementMetrics::new("test-pipeline", "timer-test");

        {
            let _timer = metrics.start_timer();
            std::thread::sleep(Duration::from_millis(1));
            // Timer records on drop
        }
        // No panic means success
    }

    #[test]
    fn test_pipeline_metrics() {
        let metrics = PipelineMetrics::new("test-pipeline");

        metrics.record_buffers(10);
        metrics.record_bytes(1000);
        metrics.record_error();

        assert_eq!(metrics.name(), "test-pipeline");
    }

    #[test]
    fn test_global_recording_functions() {
        // These should not panic even without a recorder installed
        record_buffer_produced("test", "src");
        record_buffer_consumed("test", "sink");
        record_buffer_processed("test", "transform");
        record_buffer_dropped("test", "filter");
        record_bytes_produced("test", "src", 100);
        record_bytes_consumed("test", "sink", 100);
        record_processing_time("test", "element", Duration::from_micros(10));
        record_pool_available("pool1", 16);
        record_channel_depth("chan1", 5);
    }
}
