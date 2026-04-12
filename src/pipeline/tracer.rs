//! Tracer framework for pipeline debugging and performance analysis.
//!
//! Tracers observe pipeline behavior (buffers, events, state changes) without
//! modifying data flow. They are activated via the `PARALLAX_TRACERS` environment
//! variable or programmatically.
//!
//! # Built-in Tracers
//!
//! - **latency**: Per-element processing time (min/avg/max)
//! - **framerate**: Actual FPS at each source pad
//! - **queuelevel**: Queue fill percentages
//!
//! # Activation
//!
//! ```bash
//! # Via environment variable (semicolon-separated)
//! PARALLAX_TRACERS="latency;framerate;queuelevel" ./my_pipeline
//! ```
//!
//! ```rust,ignore
//! use parallax::pipeline::tracer::{TracerRegistry, LatencyTracer, FramerateTracer};
//!
//! let mut registry = TracerRegistry::new();
//! registry.add(Box::new(LatencyTracer::new()));
//! registry.add(Box::new(FramerateTracer::new()));
//!
//! // Attach to pipeline before running
//! pipeline.set_tracer_registry(registry);
//! ```

use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::buffer::Buffer;

// ============================================================================
// Tracer Trait
// ============================================================================

/// Hook points where tracers observe pipeline behavior.
///
/// All methods have default no-op implementations. Override only the
/// hooks you need for your tracer.
pub trait Tracer: Send + Sync {
    /// Called when a buffer passes through a source's output pad.
    fn on_buffer(&self, _element_name: &str, _buffer: &Buffer, _ts: Instant) {}

    /// Called when an element finishes processing a buffer.
    fn on_buffer_processed(&self, _element_name: &str, _ts: Instant) {}

    /// Called when a buffer is dropped (e.g., by a leaky queue).
    fn on_drop(&self, _element_name: &str, _ts: Instant) {}

    /// Called when the pipeline starts.
    fn on_pipeline_start(&self) {}

    /// Called when the pipeline stops.
    fn on_pipeline_stop(&self) {}

    /// Produce a human-readable final report.
    fn report(&self) -> Option<String> {
        None
    }

    /// Name of this tracer (for logging).
    fn name(&self) -> &str;
}

// ============================================================================
// TracerRegistry
// ============================================================================

/// Registry of active tracers.
///
/// Shared between pipeline and executor. Thread-safe via Arc<Mutex>.
#[derive(Clone, Default)]
pub struct TracerRegistry {
    inner: Arc<Mutex<Vec<Box<dyn Tracer>>>>,
}

impl TracerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tracer.
    pub fn add(&self, tracer: Box<dyn Tracer>) {
        self.inner.lock().unwrap().push(tracer);
    }

    /// Check if any tracers are registered.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().unwrap().is_empty()
    }

    /// Number of registered tracers.
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Notify all tracers of a buffer.
    pub fn notify_buffer(&self, element_name: &str, buffer: &Buffer) {
        let ts = Instant::now();
        let tracers = self.inner.lock().unwrap();
        for tracer in tracers.iter() {
            tracer.on_buffer(element_name, buffer, ts);
        }
    }

    /// Notify all tracers of a processed buffer.
    pub fn notify_buffer_processed(&self, element_name: &str) {
        let ts = Instant::now();
        let tracers = self.inner.lock().unwrap();
        for tracer in tracers.iter() {
            tracer.on_buffer_processed(element_name, ts);
        }
    }

    /// Notify all tracers of a dropped buffer.
    pub fn notify_drop(&self, element_name: &str) {
        let ts = Instant::now();
        let tracers = self.inner.lock().unwrap();
        for tracer in tracers.iter() {
            tracer.on_drop(element_name, ts);
        }
    }

    /// Notify all tracers that pipeline started.
    pub fn notify_start(&self) {
        let tracers = self.inner.lock().unwrap();
        for tracer in tracers.iter() {
            tracer.on_pipeline_start();
        }
    }

    /// Notify all tracers that pipeline stopped.
    pub fn notify_stop(&self) {
        let tracers = self.inner.lock().unwrap();
        for tracer in tracers.iter() {
            tracer.on_pipeline_stop();
        }
    }

    /// Collect reports from all tracers.
    pub fn reports(&self) -> Vec<(String, String)> {
        let tracers = self.inner.lock().unwrap();
        tracers
            .iter()
            .filter_map(|t| t.report().map(|r| (t.name().to_string(), r)))
            .collect()
    }
}

impl std::fmt::Debug for TracerRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracerRegistry")
            .field("count", &self.len())
            .finish()
    }
}

// ============================================================================
// Built-in: LatencyTracer
// ============================================================================

/// Measures per-element buffer processing latency.
///
/// Tracks the time between `on_buffer` (entry) and `on_buffer_processed`
/// (exit) for each element.
pub struct LatencyTracer {
    stats: Arc<Mutex<std::collections::HashMap<String, LatencyStats>>>,
    pending: Arc<Mutex<std::collections::HashMap<String, Instant>>>,
}

struct LatencyStats {
    min_ns: u64,
    max_ns: u64,
    sum_ns: u64,
    count: u64,
}

impl LatencyTracer {
    /// Create a new latency tracer.
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(std::collections::HashMap::new())),
            pending: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for LatencyTracer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracer for LatencyTracer {
    fn on_buffer(&self, element_name: &str, _buffer: &Buffer, ts: Instant) {
        self.pending
            .lock()
            .unwrap()
            .insert(element_name.to_string(), ts);
    }

    fn on_buffer_processed(&self, element_name: &str, ts: Instant) {
        let start = self
            .pending
            .lock()
            .unwrap()
            .remove(element_name);

        if let Some(start) = start {
            let elapsed = ts.duration_since(start).as_nanos() as u64;
            let mut stats = self.stats.lock().unwrap();
            let entry = stats.entry(element_name.to_string()).or_insert(LatencyStats {
                min_ns: u64::MAX,
                max_ns: 0,
                sum_ns: 0,
                count: 0,
            });
            entry.min_ns = entry.min_ns.min(elapsed);
            entry.max_ns = entry.max_ns.max(elapsed);
            entry.sum_ns += elapsed;
            entry.count += 1;
        }
    }

    fn report(&self) -> Option<String> {
        let stats = self.stats.lock().unwrap();
        if stats.is_empty() {
            return None;
        }
        let mut report = String::from("Latency Report:\n");
        let mut entries: Vec<_> = stats.iter().collect();
        entries.sort_by_key(|(name, _)| (*name).clone());
        for (name, s) in entries {
            if s.count > 0 {
                let avg = s.sum_ns / s.count;
                report.push_str(&format!(
                    "  {name}: avg={:.2}ms min={:.2}ms max={:.2}ms (n={})\n",
                    avg as f64 / 1_000_000.0,
                    s.min_ns as f64 / 1_000_000.0,
                    s.max_ns as f64 / 1_000_000.0,
                    s.count,
                ));
            }
        }
        Some(report)
    }

    fn name(&self) -> &str {
        "latency"
    }
}

// ============================================================================
// Built-in: FramerateTracer
// ============================================================================

/// Measures actual framerate (buffers/sec) at each element's output.
pub struct FramerateTracer {
    stats: Arc<Mutex<std::collections::HashMap<String, FramerateStats>>>,
}

struct FramerateStats {
    first_ts: Option<Instant>,
    count: u64,
}

impl FramerateTracer {
    /// Create a new framerate tracer.
    pub fn new() -> Self {
        Self {
            stats: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for FramerateTracer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracer for FramerateTracer {
    fn on_buffer(&self, element_name: &str, _buffer: &Buffer, ts: Instant) {
        let mut stats = self.stats.lock().unwrap();
        let entry = stats
            .entry(element_name.to_string())
            .or_insert(FramerateStats {
                first_ts: None,
                count: 0,
            });
        if entry.first_ts.is_none() {
            entry.first_ts = Some(ts);
        }
        entry.count += 1;
    }

    fn report(&self) -> Option<String> {
        let stats = self.stats.lock().unwrap();
        if stats.is_empty() {
            return None;
        }
        let now = Instant::now();
        let mut report = String::from("Framerate Report:\n");
        let mut entries: Vec<_> = stats.iter().collect();
        entries.sort_by_key(|(name, _)| (*name).clone());
        for (name, s) in entries {
            if let Some(first) = s.first_ts {
                let elapsed = now.duration_since(first).as_secs_f64();
                if elapsed > 0.0 {
                    let fps = s.count as f64 / elapsed;
                    report.push_str(&format!(
                        "  {name}: {fps:.1} buf/s ({} buffers in {elapsed:.1}s)\n",
                        s.count,
                    ));
                }
            }
        }
        Some(report)
    }

    fn name(&self) -> &str {
        "framerate"
    }
}

// ============================================================================
// Built-in: DropTracer
// ============================================================================

/// Counts dropped buffers per element.
pub struct DropTracer {
    drops: Arc<Mutex<std::collections::HashMap<String, u64>>>,
}

impl DropTracer {
    /// Create a new drop tracer.
    pub fn new() -> Self {
        Self {
            drops: Arc::new(Mutex::new(std::collections::HashMap::new())),
        }
    }
}

impl Default for DropTracer {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracer for DropTracer {
    fn on_drop(&self, element_name: &str, _ts: Instant) {
        let mut drops = self.drops.lock().unwrap();
        *drops.entry(element_name.to_string()).or_insert(0) += 1;
    }

    fn report(&self) -> Option<String> {
        let drops = self.drops.lock().unwrap();
        if drops.is_empty() {
            return None;
        }
        let mut report = String::from("Drop Report:\n");
        let mut entries: Vec<_> = drops.iter().collect();
        entries.sort_by_key(|(name, _)| (*name).clone());
        for (name, count) in entries {
            report.push_str(&format!("  {name}: {count} dropped\n"));
        }
        Some(report)
    }

    fn name(&self) -> &str {
        "drops"
    }
}

// ============================================================================
// Environment Variable Initialization
// ============================================================================

/// Initialize tracers from the `PARALLAX_TRACERS` environment variable.
///
/// Format: semicolon-separated tracer names.
///
/// ```bash
/// PARALLAX_TRACERS="latency;framerate;drops"
/// ```
pub fn init_tracers_from_env() -> TracerRegistry {
    let registry = TracerRegistry::new();
    if let Ok(tracers) = std::env::var("PARALLAX_TRACERS") {
        for spec in tracers.split(';') {
            let spec = spec.trim();
            match spec {
                "latency" => registry.add(Box::new(LatencyTracer::new())),
                "framerate" => registry.add(Box::new(FramerateTracer::new())),
                "drops" => registry.add(Box::new(DropTracer::new())),
                "" => {}
                other => {
                    tracing::warn!("Unknown tracer: {other}");
                }
            }
        }
    }
    registry
}

// ============================================================================
// Pipeline Stats Snapshot
// ============================================================================

/// Snapshot of pipeline statistics at a point in time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStats {
    /// Current pipeline state.
    pub state: crate::pipeline::PipelineState,
    /// Number of elements in the pipeline.
    pub element_count: usize,
    /// Number of links between elements.
    pub link_count: usize,
    /// Per-element statistics.
    pub elements: Vec<ElementStats>,
}

impl std::fmt::Display for PipelineStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Pipeline({:?}): {} elements, {} links",
            self.state, self.element_count, self.link_count
        )
    }
}

/// Statistics for a single element.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ElementStats {
    /// Element name.
    pub name: String,
    /// Element type (Source, Sink, Transform, etc.).
    pub element_type: String,
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::{Buffer, MemoryHandle};
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(64, 16).unwrap())
    }

    fn make_test_buffer() -> Buffer {
        let arena = test_arena();
        let slot = arena.acquire().unwrap();
        let handle = MemoryHandle::with_len(slot, 16);
        Buffer::new(handle, Metadata::from_sequence(0))
    }

    #[test]
    fn test_tracer_registry_basic() {
        let registry = TracerRegistry::new();
        assert!(registry.is_empty());
        registry.add(Box::new(LatencyTracer::new()));
        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
    }

    #[test]
    fn test_latency_tracer() {
        let tracer = LatencyTracer::new();
        let buffer = make_test_buffer();
        let ts = Instant::now();

        tracer.on_buffer("decoder", &buffer, ts);
        // Simulate some processing time
        std::thread::sleep(std::time::Duration::from_micros(100));
        tracer.on_buffer_processed("decoder", Instant::now());

        let report = tracer.report().unwrap();
        assert!(report.contains("decoder"));
        assert!(report.contains("avg="));
    }

    #[test]
    fn test_framerate_tracer() {
        let tracer = FramerateTracer::new();
        let buffer = make_test_buffer();

        for _ in 0..10 {
            tracer.on_buffer("source", &buffer, Instant::now());
        }

        let report = tracer.report().unwrap();
        assert!(report.contains("source"));
        assert!(report.contains("10 buffers"));
    }

    #[test]
    fn test_drop_tracer() {
        let tracer = DropTracer::new();
        tracer.on_drop("queue", Instant::now());
        tracer.on_drop("queue", Instant::now());

        let report = tracer.report().unwrap();
        assert!(report.contains("queue: 2 dropped"));
    }

    #[test]
    fn test_registry_notify() {
        let registry = TracerRegistry::new();
        registry.add(Box::new(LatencyTracer::new()));
        registry.add(Box::new(FramerateTracer::new()));

        let buffer = make_test_buffer();
        registry.notify_buffer("source", &buffer);
        registry.notify_buffer_processed("source");

        let reports = registry.reports();
        assert!(!reports.is_empty());
    }

    #[test]
    fn test_empty_registry_reports() {
        let registry = TracerRegistry::new();
        let reports = registry.reports();
        assert!(reports.is_empty());
    }

    #[test]
    fn test_init_tracers_no_env() {
        // Without env var set, should return empty registry
        // SAFETY: Only modifying test-specific env var, no other threads using it
        unsafe { std::env::remove_var("PARALLAX_TRACERS"); }
        let registry = init_tracers_from_env();
        assert!(registry.is_empty());
    }
}
