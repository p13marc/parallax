//! Supervisor for managing element processes.
//!
//! The supervisor is responsible for:
//! - Spawning element processes with appropriate sandboxing
//! - Managing shared memory arenas
//! - Routing data between elements via IPC
//! - Detecting and handling element crashes
//! - Implementing restart policies

use super::mode::{ExecutionMode, GroupId};
use super::protocol::ElementState;
use crate::error::{Error, Result};
use crate::memory::CpuArena;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Global element ID counter.
static NEXT_ELEMENT_ID: AtomicU64 = AtomicU64::new(1);

/// Unique identifier for an element instance.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ElementId(pub u64);

impl ElementId {
    /// Generate a new unique element ID.
    pub fn new() -> Self {
        Self(NEXT_ELEMENT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

impl Default for ElementId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for ElementId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Element({})", self.0)
    }
}

/// Restart policy for crashed elements.
#[derive(Clone, Debug)]
pub struct RestartPolicy {
    /// Maximum number of restarts before giving up.
    pub max_restarts: u32,
    /// Initial delay before restarting.
    pub restart_delay: Duration,
    /// Backoff strategy for repeated failures.
    pub backoff: BackoffStrategy,
    /// Time window for counting restarts.
    pub reset_window: Duration,
}

impl Default for RestartPolicy {
    fn default() -> Self {
        Self {
            max_restarts: 3,
            restart_delay: Duration::from_millis(100),
            backoff: BackoffStrategy::Exponential {
                factor: 2.0,
                max: Duration::from_secs(30),
            },
            reset_window: Duration::from_secs(60),
        }
    }
}

impl RestartPolicy {
    /// Create a policy that never restarts.
    pub fn never() -> Self {
        Self {
            max_restarts: 0,
            ..Default::default()
        }
    }

    /// Create a policy that always restarts (up to a limit).
    pub fn always(max_restarts: u32) -> Self {
        Self {
            max_restarts,
            ..Default::default()
        }
    }

    /// Calculate the delay for the nth restart.
    pub fn delay_for_restart(&self, restart_count: u32) -> Duration {
        match &self.backoff {
            BackoffStrategy::Fixed => self.restart_delay,
            BackoffStrategy::Linear { increment } => {
                self.restart_delay + *increment * restart_count
            }
            BackoffStrategy::Exponential { factor, max } => {
                let delay = self.restart_delay.as_secs_f64() * factor.powi(restart_count as i32);
                Duration::from_secs_f64(delay.min(max.as_secs_f64()))
            }
        }
    }
}

/// Backoff strategy for restart delays.
#[derive(Clone, Debug)]
pub enum BackoffStrategy {
    /// Fixed delay between restarts.
    Fixed,
    /// Linear increase in delay.
    Linear {
        /// Delay increment per restart.
        increment: Duration,
    },
    /// Exponential increase in delay.
    Exponential {
        /// Multiplier per restart.
        factor: f64,
        /// Maximum delay.
        max: Duration,
    },
}

/// Information about a running element process.
#[derive(Debug)]
pub struct ElementProcess {
    /// Element identifier.
    pub id: ElementId,
    /// Element name.
    pub name: String,
    /// Process ID (if running in separate process).
    pub pid: Option<u32>,
    /// Current state.
    pub state: ElementState,
    /// Group this element belongs to.
    pub group: GroupId,
    /// Number of times this element has been restarted.
    pub restart_count: u32,
    /// Statistics.
    pub stats: ElementStats,
}

/// Statistics for an element.
#[derive(Clone, Debug, Default)]
pub struct ElementStats {
    /// Number of buffers processed.
    pub buffers_processed: u64,
    /// Total bytes processed.
    pub bytes_processed: u64,
    /// Average processing time in nanoseconds.
    pub avg_process_time_ns: u64,
    /// Number of errors encountered.
    pub errors: u64,
}

/// Supervisor for managing pipeline execution.
///
/// The supervisor owns shared memory arenas, spawns element processes,
/// and coordinates data flow between them.
pub struct Supervisor {
    /// Execution mode.
    mode: ExecutionMode,
    /// Running element processes.
    elements: HashMap<ElementId, ElementProcess>,
    /// Shared memory arenas.
    arenas: HashMap<u64, Arc<CpuArena>>,
    /// Restart policy.
    restart_policy: RestartPolicy,
    /// Whether the supervisor is running.
    running: bool,
    /// Next arena ID.
    next_arena_id: u64,
}

impl Supervisor {
    /// Create a new supervisor with the given execution mode.
    pub fn new(mode: ExecutionMode) -> Self {
        Self {
            mode,
            elements: HashMap::new(),
            arenas: HashMap::new(),
            restart_policy: RestartPolicy::default(),
            running: false,
            next_arena_id: 1,
        }
    }

    /// Create a supervisor for in-process execution.
    pub fn in_process() -> Self {
        Self::new(ExecutionMode::InProcess)
    }

    /// Create a supervisor with isolated execution.
    pub fn isolated() -> Self {
        Self::new(ExecutionMode::isolated())
    }

    /// Set the restart policy.
    pub fn with_restart_policy(mut self, policy: RestartPolicy) -> Self {
        self.restart_policy = policy;
        self
    }

    /// Get the execution mode.
    pub fn mode(&self) -> &ExecutionMode {
        &self.mode
    }

    /// Check if the supervisor is running.
    pub fn is_running(&self) -> bool {
        self.running
    }

    /// Register an element with the supervisor.
    ///
    /// Returns the element ID assigned to this element.
    pub fn register_element(&mut self, name: impl Into<String>) -> ElementId {
        let name = name.into();
        let id = ElementId::new();
        let group = self.mode.get_group(&name).unwrap_or(GroupId::SUPERVISOR);

        let process = ElementProcess {
            id,
            name,
            pid: None,
            state: ElementState::Null,
            group,
            restart_count: 0,
            stats: ElementStats::default(),
        };

        self.elements.insert(id, process);
        id
    }

    /// Get information about an element.
    pub fn get_element(&self, id: ElementId) -> Option<&ElementProcess> {
        self.elements.get(&id)
    }

    /// Get mutable information about an element.
    pub fn get_element_mut(&mut self, id: ElementId) -> Option<&mut ElementProcess> {
        self.elements.get_mut(&id)
    }

    /// Get all registered elements.
    pub fn elements(&self) -> impl Iterator<Item = &ElementProcess> {
        self.elements.values()
    }

    /// Create a new shared memory arena.
    ///
    /// Returns the arena ID for cross-process reference.
    pub fn create_arena(&mut self, slot_size: usize, slot_count: usize) -> Result<u64> {
        let arena = CpuArena::new(slot_size, slot_count)?;
        let arena_id = self.next_arena_id;
        self.next_arena_id += 1;

        self.arenas.insert(arena_id, arena);
        Ok(arena_id)
    }

    /// Get an arena by ID.
    pub fn get_arena(&self, arena_id: u64) -> Option<&Arc<CpuArena>> {
        self.arenas.get(&arena_id)
    }

    /// Start the supervisor.
    pub fn start(&mut self) -> Result<()> {
        if self.running {
            return Err(Error::Pipeline("Supervisor already running".into()));
        }

        // Transition all elements to Ready state
        for element in self.elements.values_mut() {
            element.state = ElementState::Ready;
        }

        self.running = true;
        Ok(())
    }

    /// Stop the supervisor.
    pub fn stop(&mut self) -> Result<()> {
        if !self.running {
            return Ok(());
        }

        // Transition all elements to Null state
        for element in self.elements.values_mut() {
            element.state = ElementState::Null;
        }

        self.running = false;
        Ok(())
    }

    /// Set all elements to Playing state.
    pub fn play(&mut self) -> Result<()> {
        if !self.running {
            return Err(Error::Pipeline("Supervisor not running".into()));
        }

        for element in self.elements.values_mut() {
            if element.state == ElementState::Ready || element.state == ElementState::Paused {
                element.state = ElementState::Playing;
            }
        }

        Ok(())
    }

    /// Set all elements to Paused state.
    pub fn pause(&mut self) -> Result<()> {
        if !self.running {
            return Err(Error::Pipeline("Supervisor not running".into()));
        }

        for element in self.elements.values_mut() {
            if element.state == ElementState::Playing {
                element.state = ElementState::Paused;
            }
        }

        Ok(())
    }

    /// Handle an element crash.
    ///
    /// Returns true if the element should be restarted.
    pub fn handle_crash(&mut self, id: ElementId) -> bool {
        let Some(element) = self.elements.get_mut(&id) else {
            return false;
        };

        element.restart_count += 1;
        element.stats.errors += 1;

        if element.restart_count > self.restart_policy.max_restarts {
            // Too many restarts, give up
            element.state = ElementState::Null;
            false
        } else {
            // Will restart after delay
            element.state = ElementState::Ready;
            true
        }
    }

    /// Get the restart delay for an element.
    pub fn restart_delay(&self, id: ElementId) -> Duration {
        let restart_count = self.elements.get(&id).map(|e| e.restart_count).unwrap_or(0);

        self.restart_policy.delay_for_restart(restart_count)
    }

    /// Update element statistics.
    pub fn update_stats(&mut self, id: ElementId, buffers: u64, bytes: u64, avg_time_ns: u64) {
        if let Some(element) = self.elements.get_mut(&id) {
            element.stats.buffers_processed += buffers;
            element.stats.bytes_processed += bytes;
            // Running average
            if element.stats.avg_process_time_ns == 0 {
                element.stats.avg_process_time_ns = avg_time_ns;
            } else {
                element.stats.avg_process_time_ns =
                    (element.stats.avg_process_time_ns + avg_time_ns) / 2;
            }
        }
    }

    /// Get aggregate statistics for all elements.
    pub fn total_stats(&self) -> ElementStats {
        let mut total = ElementStats::default();
        for element in self.elements.values() {
            total.buffers_processed += element.stats.buffers_processed;
            total.bytes_processed += element.stats.bytes_processed;
            total.errors += element.stats.errors;
        }
        total
    }
}

impl Default for Supervisor {
    fn default() -> Self {
        Self::in_process()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_id_unique() {
        let id1 = ElementId::new();
        let id2 = ElementId::new();
        let id3 = ElementId::new();

        assert_ne!(id1, id2);
        assert_ne!(id2, id3);
        assert_ne!(id1, id3);
    }

    #[test]
    fn test_supervisor_creation() {
        let sup = Supervisor::new(ExecutionMode::InProcess);
        assert!(!sup.is_running());
        assert!(matches!(sup.mode(), ExecutionMode::InProcess));
    }

    #[test]
    fn test_supervisor_register_element() {
        let mut sup = Supervisor::in_process();

        let id = sup.register_element("test_element");
        let element = sup.get_element(id).unwrap();

        assert_eq!(element.name, "test_element");
        assert_eq!(element.state, ElementState::Null);
        assert_eq!(element.group, GroupId::SUPERVISOR);
    }

    #[test]
    fn test_supervisor_start_stop() {
        let mut sup = Supervisor::in_process();
        sup.register_element("elem1");
        sup.register_element("elem2");

        assert!(sup.start().is_ok());
        assert!(sup.is_running());

        // All elements should be Ready
        for elem in sup.elements() {
            assert_eq!(elem.state, ElementState::Ready);
        }

        assert!(sup.stop().is_ok());
        assert!(!sup.is_running());

        // All elements should be Null
        for elem in sup.elements() {
            assert_eq!(elem.state, ElementState::Null);
        }
    }

    #[test]
    fn test_supervisor_play_pause() {
        let mut sup = Supervisor::in_process();
        sup.register_element("elem1");
        sup.start().unwrap();

        assert!(sup.play().is_ok());
        for elem in sup.elements() {
            assert_eq!(elem.state, ElementState::Playing);
        }

        assert!(sup.pause().is_ok());
        for elem in sup.elements() {
            assert_eq!(elem.state, ElementState::Paused);
        }
    }

    #[test]
    fn test_supervisor_create_arena() {
        let mut sup = Supervisor::in_process();

        let arena_id = sup.create_arena(4096, 16).unwrap();
        assert!(sup.get_arena(arena_id).is_some());

        let arena = sup.get_arena(arena_id).unwrap();
        assert_eq!(arena.slot_count(), 16);
    }

    #[test]
    fn test_restart_policy_default() {
        let policy = RestartPolicy::default();
        assert_eq!(policy.max_restarts, 3);
        assert_eq!(policy.restart_delay, Duration::from_millis(100));
    }

    #[test]
    fn test_restart_policy_never() {
        let policy = RestartPolicy::never();
        assert_eq!(policy.max_restarts, 0);
    }

    #[test]
    fn test_restart_policy_exponential_backoff() {
        let policy = RestartPolicy {
            restart_delay: Duration::from_millis(100),
            backoff: BackoffStrategy::Exponential {
                factor: 2.0,
                max: Duration::from_secs(10),
            },
            ..Default::default()
        };

        assert_eq!(policy.delay_for_restart(0), Duration::from_millis(100));
        assert_eq!(policy.delay_for_restart(1), Duration::from_millis(200));
        assert_eq!(policy.delay_for_restart(2), Duration::from_millis(400));
        assert_eq!(policy.delay_for_restart(3), Duration::from_millis(800));

        // Should cap at max
        assert_eq!(policy.delay_for_restart(10), Duration::from_secs(10));
    }

    #[test]
    fn test_supervisor_handle_crash() {
        let mut sup = Supervisor::in_process();
        sup.restart_policy = RestartPolicy {
            max_restarts: 2,
            ..Default::default()
        };

        let id = sup.register_element("crashy");
        sup.start().unwrap();

        // First crash - should restart
        assert!(sup.handle_crash(id));
        assert_eq!(sup.get_element(id).unwrap().restart_count, 1);

        // Second crash - should restart
        assert!(sup.handle_crash(id));
        assert_eq!(sup.get_element(id).unwrap().restart_count, 2);

        // Third crash - exceeded max, should not restart
        assert!(!sup.handle_crash(id));
        assert_eq!(sup.get_element(id).unwrap().state, ElementState::Null);
    }

    #[test]
    fn test_supervisor_update_stats() {
        let mut sup = Supervisor::in_process();
        let id = sup.register_element("elem");

        sup.update_stats(id, 100, 1000, 5000);
        let elem = sup.get_element(id).unwrap();
        assert_eq!(elem.stats.buffers_processed, 100);
        assert_eq!(elem.stats.bytes_processed, 1000);
        assert_eq!(elem.stats.avg_process_time_ns, 5000);

        sup.update_stats(id, 100, 1000, 3000);
        let elem = sup.get_element(id).unwrap();
        assert_eq!(elem.stats.buffers_processed, 200);
        assert_eq!(elem.stats.bytes_processed, 2000);
        assert_eq!(elem.stats.avg_process_time_ns, 4000); // Running average
    }

    #[test]
    fn test_supervisor_total_stats() {
        let mut sup = Supervisor::in_process();
        let id1 = sup.register_element("elem1");
        let id2 = sup.register_element("elem2");

        sup.update_stats(id1, 100, 1000, 5000);
        sup.update_stats(id2, 200, 2000, 3000);

        let total = sup.total_stats();
        assert_eq!(total.buffers_processed, 300);
        assert_eq!(total.bytes_processed, 3000);
    }

    #[test]
    fn test_supervisor_grouped_mode() {
        let mode = ExecutionMode::grouped(vec!["*decoder*".into()]);
        let mut sup = Supervisor::new(mode);

        let decoder_id = sup.register_element("h264_decoder");
        let filter_id = sup.register_element("passthrough");

        // Decoder should be isolated (no group)
        let _decoder = sup.get_element(decoder_id).unwrap();
        // In grouped mode with pattern match, isolated elements still get SUPERVISOR group
        // but should_isolate returns true
        assert!(sup.mode().should_isolate("h264_decoder"));
        assert!(!sup.mode().should_isolate("passthrough"));

        let filter = sup.get_element(filter_id).unwrap();
        assert_eq!(filter.group, GroupId::SUPERVISOR);
    }
}
