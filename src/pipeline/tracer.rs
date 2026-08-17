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

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::buffer::Buffer;

// ============================================================================
// Tracer Trait
// ============================================================================

/// Which per-buffer hooks a tracer actually implements (#189).
///
/// The per-buffer hooks are called from inside element tasks — a registered
/// tracer used to cost every buffer at every hop a mutex round-trip, an
/// `Instant::now()` and a `catch_unwind` *per hook*, even for hooks it left
/// as no-ops. Declaring interests lets the registry skip hooks nobody
/// implements. The rare lifecycle hooks (`on_pipeline_start`/`stop`,
/// `report`) are always delivered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TracerInterests(u8);

impl TracerInterests {
    /// `on_buffer`.
    pub const BUFFER: Self = Self(1);
    /// `on_buffer_processed`.
    pub const BUFFER_PROCESSED: Self = Self(1 << 1);
    /// `on_drop`.
    pub const DROP: Self = Self(1 << 2);
    /// Every hook — the safe default.
    pub const ALL: Self = Self(0b111);

    /// Combine two interest sets.
    pub const fn and(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Does this set include every hook in `other`?
    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }
}

/// Hook points where tracers observe pipeline behavior.
///
/// All methods have default no-op implementations. Override only the
/// hooks you need for your tracer — and declare them in
/// [`interests`](Tracer::interests) so unimplemented hot-path hooks cost
/// nothing.
pub trait Tracer: Send + Sync {
    /// Which per-buffer hooks this tracer implements. Default: all of them,
    /// so an undeclared tracer keeps working; a tracer that only counts
    /// drops should return [`TracerInterests::DROP`] and take its two
    /// no-op hooks off every buffer's hot path.
    fn interests(&self) -> TracerInterests {
        TracerInterests::ALL
    }

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
/// Shared between pipeline and executor. Thread-safe via `Arc<Mutex>`.
#[derive(Clone, Default)]
pub struct TracerRegistry {
    inner: Arc<Mutex<Vec<Box<dyn Tracer>>>>,
    /// Registered-tracer count, for the lock-free empty fast path (#142):
    /// with no tracers, every buffer used to pay two mutex round-trips and
    /// two `Instant::now()` clock reads per element hop.
    count: Arc<std::sync::atomic::AtomicUsize>,
    /// Per-hook interested-tracer counts (#189) — the same fast path, per
    /// hook: a drops-only tracer must not put `on_buffer`/`on_buffer_processed`
    /// back on every buffer's hot path.
    hooks: Arc<HookCounts>,
}

#[derive(Default)]
struct HookCounts {
    buffer: std::sync::atomic::AtomicUsize,
    buffer_processed: std::sync::atomic::AtomicUsize,
    drop: std::sync::atomic::AtomicUsize,
}

impl HookCounts {
    /// Recompute from the surviving tracers — runs under the registry lock,
    /// on add and on panic-removal.
    fn refresh(&self, tracers: &[Box<dyn Tracer>]) {
        use std::sync::atomic::Ordering;
        let mut buffer = 0;
        let mut processed = 0;
        let mut drop = 0;
        for tracer in tracers {
            let interests = tracer.interests();
            buffer += usize::from(interests.contains(TracerInterests::BUFFER));
            processed += usize::from(interests.contains(TracerInterests::BUFFER_PROCESSED));
            drop += usize::from(interests.contains(TracerInterests::DROP));
        }
        self.buffer.store(buffer, Ordering::Release);
        self.buffer_processed.store(processed, Ordering::Release);
        self.drop.store(drop, Ordering::Release);
    }
}

/// A tracer's name, or a stand-in if asking for it also panics.
///
/// Only ever called on the failure path. A tracer broken enough that `name()`
/// panics must still not be able to take the pipeline down through the very
/// call that was reporting it.
fn name_of(tracer: &dyn Tracer) -> String {
    match catch_unwind(AssertUnwindSafe(|| tracer.name().to_string())) {
        Ok(name) => name,
        Err(_) => "<unnamed>".to_string(),
    }
}

impl TracerRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Lock the registry, ignoring poisoning.
    ///
    /// Hook panics are caught in [`notify_each`](Self::notify_each) before they
    /// can poison anything, so a poisoned lock here only ever means someone else
    /// died holding it. Propagating that would turn one broken tracer into a
    /// second failure that kills every element task that touches the registry —
    /// exactly the fault this module is trying not to have.
    fn lock(&self) -> std::sync::MutexGuard<'_, Vec<Box<dyn Tracer>>> {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Add a tracer.
    pub fn add(&self, tracer: Box<dyn Tracer>) {
        let mut tracers = self.lock();
        tracers.push(tracer);
        self.count
            .store(tracers.len(), std::sync::atomic::Ordering::Release);
        self.hooks.refresh(&tracers);
    }

    /// Check if any tracers are registered (lock-free).
    pub fn is_empty(&self) -> bool {
        self.count.load(std::sync::atomic::Ordering::Acquire) == 0
    }

    /// Number of registered tracers.
    pub fn len(&self) -> usize {
        self.lock().len()
    }

    /// Run one hook against every tracer, dropping any that panics.
    ///
    /// A tracer is an observer, and observing must not be able to end the run:
    /// these hooks are called from inside element tasks, so a panicking one used
    /// to kill the element that happened to invoke it — and every sink below it
    /// — while naming that element as the culprit.
    ///
    /// The offender is removed rather than retried. Retrying would re-panic on
    /// every buffer, and removal leaves the pipeline behaving as it did before
    /// the tracer was attached. It is logged at `error!`, so nothing disappears
    /// quietly.
    fn notify_each(&self, hook: &'static str, call: impl Fn(&dyn Tracer)) {
        let mut tracers = self.lock();
        let mut failed = Vec::new();

        for (i, tracer) in tracers.iter().enumerate() {
            if let Err(payload) = catch_unwind(AssertUnwindSafe(|| call(tracer.as_ref()))) {
                tracing::error!(
                    tracer = name_of(tracer.as_ref()),
                    hook,
                    "tracer panicked and has been removed: {}",
                    crate::error::panic_message(payload.as_ref())
                );
                failed.push(i);
            }
        }

        if !failed.is_empty() {
            for i in failed.into_iter().rev() {
                tracers.remove(i);
            }
            self.count
                .store(tracers.len(), std::sync::atomic::Ordering::Release);
            self.hooks.refresh(&tracers);
        }
    }

    /// Notify all tracers of a buffer.
    pub fn notify_buffer(&self, element_name: &str, buffer: &Buffer) {
        use std::sync::atomic::Ordering;
        if self.hooks.buffer.load(Ordering::Acquire) == 0 {
            return;
        }
        let ts = Instant::now();
        self.notify_each("on_buffer", |t| t.on_buffer(element_name, buffer, ts));
    }

    /// Notify all tracers of a processed buffer.
    pub fn notify_buffer_processed(&self, element_name: &str) {
        use std::sync::atomic::Ordering;
        if self.hooks.buffer_processed.load(Ordering::Acquire) == 0 {
            return;
        }
        let ts = Instant::now();
        self.notify_each("on_buffer_processed", |t| {
            t.on_buffer_processed(element_name, ts)
        });
    }

    /// Notify all tracers of a dropped buffer.
    pub fn notify_drop(&self, element_name: &str) {
        use std::sync::atomic::Ordering;
        if self.hooks.drop.load(Ordering::Acquire) == 0 {
            return;
        }
        let ts = Instant::now();
        self.notify_each("on_drop", |t| t.on_drop(element_name, ts));
    }

    /// Notify all tracers that pipeline started.
    pub fn notify_start(&self) {
        self.notify_each("on_pipeline_start", |t| t.on_pipeline_start());
    }

    /// Notify all tracers that pipeline stopped.
    pub fn notify_stop(&self) {
        self.notify_each("on_pipeline_stop", |t| t.on_pipeline_stop());
    }

    /// Collect reports from all tracers.
    ///
    /// A tracer whose `report()` panics is skipped rather than allowed to take
    /// down the caller — reports are usually collected at shutdown, where a
    /// panic would lose every *other* tracer's report too.
    pub fn reports(&self) -> Vec<(String, String)> {
        let tracers = self.lock();
        tracers
            .iter()
            .filter_map(|t| {
                match catch_unwind(AssertUnwindSafe(|| {
                    t.report().map(|r| (t.name().to_string(), r))
                })) {
                    Ok(report) => report,
                    Err(payload) => {
                        tracing::error!(
                            tracer = name_of(t.as_ref()),
                            "tracer panicked while reporting: {}",
                            crate::error::panic_message(payload.as_ref())
                        );
                        None
                    }
                }
            })
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
    fn interests(&self) -> TracerInterests {
        TracerInterests::BUFFER.and(TracerInterests::BUFFER_PROCESSED)
    }

    fn on_buffer(&self, element_name: &str, _buffer: &Buffer, ts: Instant) {
        self.pending
            .lock()
            .unwrap()
            .insert(element_name.to_string(), ts);
    }

    fn on_buffer_processed(&self, element_name: &str, ts: Instant) {
        let start = self.pending.lock().unwrap().remove(element_name);

        if let Some(start) = start {
            let elapsed = ts.duration_since(start).as_nanos() as u64;
            let mut stats = self.stats.lock().unwrap();
            let entry = stats
                .entry(element_name.to_string())
                .or_insert(LatencyStats {
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
            if let Some(avg) = s.sum_ns.checked_div(s.count) {
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
    fn interests(&self) -> TracerInterests {
        TracerInterests::BUFFER
    }

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
    fn interests(&self) -> TracerInterests {
        TracerInterests::DROP
    }

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
    fn a_drops_only_tracer_keeps_buffer_hooks_off_the_hot_path() {
        // #189: interests gate the per-hook fast path. The panicking
        // `on_buffer` proves the point — if the registry called it, the
        // tracer would be removed and the drop below would go uncounted.
        struct DropsOnly(Arc<std::sync::atomic::AtomicUsize>);
        impl Tracer for DropsOnly {
            fn interests(&self) -> TracerInterests {
                TracerInterests::DROP
            }
            fn on_buffer(&self, _e: &str, _b: &Buffer, _ts: Instant) {
                panic!("on_buffer must not be dispatched to a drops-only tracer");
            }
            fn on_drop(&self, _e: &str, _ts: Instant) {
                self.0.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            fn name(&self) -> &str {
                "drops-only"
            }
        }

        let drops = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let registry = TracerRegistry::new();
        registry.add(Box::new(DropsOnly(drops.clone())));

        let buffer = make_test_buffer();
        registry.notify_buffer("source", &buffer);
        registry.notify_buffer_processed("source");
        registry.notify_drop("source");

        assert_eq!(drops.load(std::sync::atomic::Ordering::Relaxed), 1);
        assert_eq!(registry.len(), 1, "the tracer must not have been removed");
    }

    #[test]
    fn interests_refresh_after_a_panicking_tracer_is_removed() {
        // A BUFFER-interested tracer that panics is removed; the buffer hook
        // count must drop back to zero with it.
        struct Panics;
        impl Tracer for Panics {
            fn interests(&self) -> TracerInterests {
                TracerInterests::BUFFER
            }
            fn on_buffer(&self, _e: &str, _b: &Buffer, _ts: Instant) {
                panic!("boom");
            }
            fn name(&self) -> &str {
                "panics"
            }
        }

        let registry = TracerRegistry::new();
        registry.add(Box::new(Panics));
        let buffer = make_test_buffer();
        registry.notify_buffer("source", &buffer);
        assert_eq!(registry.len(), 0);
        use std::sync::atomic::Ordering;
        assert_eq!(registry.hooks.buffer.load(Ordering::Acquire), 0);
    }

    #[test]
    fn test_init_tracers_no_env() {
        // Without env var set, should return empty registry
        // SAFETY: Only modifying test-specific env var, no other threads using it
        unsafe {
            std::env::remove_var("PARALLAX_TRACERS");
        }
        let registry = init_tracers_from_env();
        assert!(registry.is_empty());
    }
}
