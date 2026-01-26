//! Real-time scheduler for hybrid pipeline execution.
//!
//! This module provides the RT scheduling infrastructure that works alongside
//! the Tokio-based async executor to enable hybrid pipelines where:
//! - I/O-bound elements run in Tokio async tasks
//! - CPU-bound RT-safe elements run in dedicated real-time threads
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                        Tokio Runtime                            │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │  TcpSrc      │  │  FileSrc     │  │  HttpSrc     │          │
//! │  │  (async I/O) │  │  (async I/O) │  │  (async I/O) │          │
//! │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
//! │         │                 │                 │                   │
//! │         ▼                 ▼                 ▼                   │
//! │  ┌─────────────────────────────────────────────────────────┐   │
//! │  │              AsyncRtBridge (boundary)                   │   │
//! │  └─────────────────────────────────────────────────────────┘   │
//! └─────────────────────────────────────────────────────────────────┘
//!                               │
//!                               ▼
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    RT Data Thread(s)                            │
//! │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
//! │  │  Decoder     │──│  Mixer       │──│  AudioSink   │          │
//! │  │  (sync)      │  │  (sync)      │  │  (sync)      │          │
//! │  └──────────────┘  └──────────────┘  └──────────────┘          │
//! │                                                                 │
//! │  Driver-based scheduling, deterministic latency                │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Scheduling Model
//!
//! The RT scheduler uses a driver-based pull model inspired by PipeWire:
//!
//! 1. A **driver node** initiates each processing cycle (via timer or hardware)
//! 2. Nodes are processed in **dependency order** (topological sort)
//! 3. Each node's **activation record** tracks pending dependencies
//! 4. When all dependencies are satisfied, the node processes its data
//! 5. **eventfd** is used for efficient inter-thread signaling
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::rt_scheduler::{RtScheduler, RtConfig, SchedulingMode};
//!
//! let config = RtConfig {
//!     mode: SchedulingMode::Hybrid,
//!     quantum: 256,  // samples per cycle
//!     rt_priority: Some(50),
//!     data_threads: 1,
//! };
//!
//! let scheduler = RtScheduler::new(config)?;
//! ```

use crate::element::{Affinity, AsyncElementDyn, DynAsyncElement};
use crate::error::{Error, Result};
use crate::pipeline::rt_bridge::{AsyncRtBridge, BridgeConfig, EventFd};
use crate::pipeline::{NodeId, Pipeline};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU32, Ordering};
use std::thread::JoinHandle;

// ============================================================================
// Configuration
// ============================================================================

/// Scheduling mode for the pipeline executor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SchedulingMode {
    /// All nodes run in Tokio async tasks (current default behavior).
    #[default]
    Async,

    /// RT-safe nodes run in dedicated RT thread(s), async nodes in Tokio.
    ///
    /// The executor automatically partitions the graph based on element
    /// affinity and inserts bridges at async↔RT boundaries.
    Hybrid,

    /// All RT-safe nodes run in RT threads (no async fallback).
    ///
    /// Nodes that are not RT-safe will cause an error.
    RealTime,
}

/// Configuration for real-time scheduling.
#[derive(Debug, Clone)]
pub struct RtConfig {
    /// Scheduling mode.
    pub mode: SchedulingMode,

    /// Quantum (samples per processing cycle).
    ///
    /// For audio at 48kHz:
    /// - 64 samples = 1.3ms (professional audio)
    /// - 256 samples = 5.3ms (low latency)
    /// - 1024 samples = 21.3ms (standard)
    ///
    /// Default is 256.
    pub quantum: u32,

    /// Real-time thread priority (SCHED_FIFO).
    ///
    /// Requires CAP_SYS_NICE or root. Range 1-99.
    /// `None` means use default thread priority.
    pub rt_priority: Option<i32>,

    /// Number of RT data threads.
    ///
    /// - 0: Disable RT threads (all in Tokio)
    /// - 1: Single RT thread (default)
    /// - N: Multiple RT threads for parallel processing
    /// - -1: One thread per CPU core
    pub data_threads: i32,

    /// Bridge buffer capacity for async↔RT boundaries.
    pub bridge_capacity: usize,
}

impl Default for RtConfig {
    fn default() -> Self {
        Self {
            mode: SchedulingMode::default(),
            quantum: 256,
            rt_priority: None,
            data_threads: 1,
            bridge_capacity: 16,
        }
    }
}

impl RtConfig {
    /// Create a config for pure async execution (no RT threads).
    pub fn async_only() -> Self {
        Self {
            mode: SchedulingMode::Async,
            data_threads: 0,
            ..Default::default()
        }
    }

    /// Create a config for hybrid execution.
    pub fn hybrid() -> Self {
        Self {
            mode: SchedulingMode::Hybrid,
            ..Default::default()
        }
    }

    /// Create a config for low-latency audio.
    pub fn low_latency_audio() -> Self {
        Self {
            mode: SchedulingMode::Hybrid,
            quantum: 64,
            rt_priority: Some(50),
            data_threads: 1,
            bridge_capacity: 4,
        }
    }

    /// Set the quantum.
    pub fn with_quantum(mut self, quantum: u32) -> Self {
        self.quantum = quantum;
        self
    }

    /// Set the RT priority.
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.rt_priority = Some(priority);
        self
    }
}

// ============================================================================
// Activation Record
// ============================================================================

/// Processing state for a node.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeStatus {
    /// Node is idle, waiting for next cycle.
    Idle = 0,
    /// Node needs data from upstream.
    NeedData = 1,
    /// Node has data ready for downstream.
    HaveData = 2,
    /// Node is currently processing.
    Processing = 3,
}

impl From<u8> for NodeStatus {
    fn from(val: u8) -> Self {
        match val {
            0 => Self::Idle,
            1 => Self::NeedData,
            2 => Self::HaveData,
            3 => Self::Processing,
            _ => Self::Idle,
        }
    }
}

/// Activation record for a node in the RT scheduler.
///
/// This tracks the node's dependencies and processing state.
/// Inspired by PipeWire's activation records.
#[repr(C)]
pub struct ActivationRecord {
    /// Number of dependencies (set during graph setup).
    pub required: AtomicU32,

    /// Remaining dependencies for current cycle.
    pub pending: AtomicU32,

    /// Current processing status.
    pub status: AtomicU8,

    /// Eventfd to signal when node is ready to process.
    pub trigger: EventFd,
}

impl ActivationRecord {
    /// Create a new activation record.
    pub fn new() -> Result<Self> {
        Ok(Self {
            required: AtomicU32::new(0),
            pending: AtomicU32::new(0),
            status: AtomicU8::new(NodeStatus::Idle as u8),
            trigger: EventFd::new()?,
        })
    }

    /// Set the number of required dependencies.
    pub fn set_required(&self, count: u32) {
        self.required.store(count, Ordering::Release);
    }

    /// Reset pending to required at the start of a cycle.
    pub fn reset_pending(&self) {
        let req = self.required.load(Ordering::Acquire);
        self.pending.store(req, Ordering::Release);
    }

    /// Decrement pending count. Returns true if node is now ready (pending == 0).
    pub fn decrement_pending(&self) -> bool {
        let prev = self.pending.fetch_sub(1, Ordering::AcqRel);
        prev == 1 // Was 1, now 0
    }

    /// Check if the node is ready to process (pending == 0).
    pub fn is_ready(&self) -> bool {
        self.pending.load(Ordering::Acquire) == 0
    }

    /// Get the current status.
    pub fn status(&self) -> NodeStatus {
        NodeStatus::from(self.status.load(Ordering::Acquire))
    }

    /// Set the status.
    pub fn set_status(&self, status: NodeStatus) {
        self.status.store(status as u8, Ordering::Release);
    }

    /// Signal that this node is ready to process.
    pub fn signal(&self) -> Result<()> {
        self.trigger.notify()
    }

    /// Wait for the trigger (non-blocking).
    pub fn try_wait(&self) -> Result<bool> {
        self.trigger.try_wait()
    }
}

// ============================================================================
// Graph Partition
// ============================================================================

/// Result of partitioning a pipeline graph for hybrid execution.
#[derive(Debug)]
pub struct GraphPartition {
    /// Nodes that should run in Tokio async tasks.
    pub async_nodes: Vec<NodeId>,

    /// Nodes that should run in RT thread(s).
    pub rt_nodes: Vec<NodeId>,

    /// Edges that cross the async↔RT boundary.
    pub boundary_edges: Vec<BoundaryEdge>,
}

/// An edge that crosses the async↔RT boundary.
#[derive(Debug, Clone)]
pub struct BoundaryEdge {
    /// Source node (in one domain).
    pub source: NodeId,
    /// Source pad name.
    pub source_pad: String,
    /// Sink node (in the other domain).
    pub sink: NodeId,
    /// Sink pad name.
    pub sink_pad: String,
    /// Direction of the crossing.
    pub direction: BoundaryDirection,
}

/// Direction of a boundary crossing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryDirection {
    /// Data flows from async to RT.
    AsyncToRt,
    /// Data flows from RT to async.
    RtToAsync,
}

impl GraphPartition {
    /// Create an empty partition.
    pub fn new() -> Self {
        Self {
            async_nodes: Vec::new(),
            rt_nodes: Vec::new(),
            boundary_edges: Vec::new(),
        }
    }

    /// Check if any nodes are assigned to RT.
    pub fn has_rt_nodes(&self) -> bool {
        !self.rt_nodes.is_empty()
    }

    /// Check if there are boundary crossings.
    pub fn has_boundaries(&self) -> bool {
        !self.boundary_edges.is_empty()
    }
}

impl Default for GraphPartition {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// RT Scheduler
// ============================================================================

/// Real-time scheduler for hybrid pipeline execution.
pub struct RtScheduler {
    /// Configuration.
    config: RtConfig,

    /// Activation records for RT nodes.
    activations: HashMap<NodeId, Arc<ActivationRecord>>,

    /// Processing order (topologically sorted RT nodes).
    processing_order: Vec<NodeId>,

    /// Driver node (initiates each cycle).
    driver: Option<NodeId>,

    /// Bridges at async↔RT boundaries.
    bridges: HashMap<(NodeId, NodeId), Arc<AsyncRtBridge>>,
}

impl RtScheduler {
    /// Create a new RT scheduler with the given configuration.
    pub fn new(config: RtConfig) -> Self {
        Self {
            config,
            activations: HashMap::new(),
            processing_order: Vec::new(),
            driver: None,
            bridges: HashMap::new(),
        }
    }

    /// Get the configuration.
    pub fn config(&self) -> &RtConfig {
        &self.config
    }

    /// Partition a pipeline graph for hybrid execution.
    pub fn partition_graph(&self, pipeline: &Pipeline) -> Result<GraphPartition> {
        let mut partition = GraphPartition::new();

        // Classify each node
        for node_id in pipeline.node_ids() {
            let node = pipeline
                .get_node(node_id)
                .ok_or_else(|| Error::InvalidSegment("node not found".into()))?;

            let affinity = self.classify_node_affinity(node, pipeline);

            match affinity {
                Affinity::Async => partition.async_nodes.push(node_id),
                Affinity::RealTime => partition.rt_nodes.push(node_id),
                Affinity::Auto => {
                    // Auto: prefer RT if element is RT-safe, else async
                    if node.is_rt_safe() {
                        partition.rt_nodes.push(node_id)
                    } else {
                        partition.async_nodes.push(node_id)
                    }
                }
            }
        }

        // Find boundary edges
        for node_id in pipeline.node_ids() {
            let is_async = partition.async_nodes.contains(&node_id);

            for (child_id, link) in pipeline.children(node_id) {
                let child_is_async = partition.async_nodes.contains(&child_id);

                if is_async != child_is_async {
                    partition.boundary_edges.push(BoundaryEdge {
                        source: node_id,
                        source_pad: link.src_pad.clone(),
                        sink: child_id,
                        sink_pad: link.sink_pad.clone(),
                        direction: if is_async {
                            BoundaryDirection::AsyncToRt
                        } else {
                            BoundaryDirection::RtToAsync
                        },
                    });
                }
            }
        }

        Ok(partition)
    }

    /// Classify a node's affinity based on element and config.
    fn classify_node_affinity(
        &self,
        node: &crate::pipeline::Node,
        _pipeline: &Pipeline,
    ) -> Affinity {
        // Check scheduling mode
        match self.config.mode {
            SchedulingMode::Async => Affinity::Async,
            SchedulingMode::RealTime => {
                if node.is_rt_safe() {
                    Affinity::RealTime
                } else {
                    // In RealTime mode, non-RT-safe nodes are an error
                    // For now, fall back to their declared affinity
                    node.affinity()
                }
            }
            SchedulingMode::Hybrid => node.affinity(),
        }
    }

    /// Set up activation records for RT nodes.
    pub fn setup_activations(&mut self, partition: &GraphPartition) -> Result<()> {
        self.activations.clear();

        for &node_id in &partition.rt_nodes {
            let activation = Arc::new(ActivationRecord::new()?);
            self.activations.insert(node_id, activation);
        }

        Ok(())
    }

    /// Compute the processing order for RT nodes.
    pub fn compute_processing_order(
        &mut self,
        partition: &GraphPartition,
        pipeline: &Pipeline,
    ) -> Result<()> {
        // Topological sort of RT nodes based on dependencies
        let mut order = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let rt_set: std::collections::HashSet<_> = partition.rt_nodes.iter().copied().collect();

        for &node_id in &partition.rt_nodes {
            self.topo_visit(node_id, pipeline, &rt_set, &mut visited, &mut order)?;
        }

        self.processing_order = order;
        Ok(())
    }

    /// Topological sort helper.
    fn topo_visit(
        &self,
        node_id: NodeId,
        pipeline: &Pipeline,
        rt_set: &std::collections::HashSet<NodeId>,
        visited: &mut std::collections::HashSet<NodeId>,
        order: &mut Vec<NodeId>,
    ) -> Result<()> {
        if visited.contains(&node_id) {
            return Ok(());
        }
        visited.insert(node_id);

        // Visit all RT dependencies first
        for (parent_id, _link) in pipeline.parents(node_id) {
            if rt_set.contains(&parent_id) {
                self.topo_visit(parent_id, pipeline, rt_set, visited, order)?;
            }
        }

        order.push(node_id);
        Ok(())
    }

    /// Set up dependency counts in activation records.
    pub fn setup_dependencies(
        &mut self,
        partition: &GraphPartition,
        pipeline: &Pipeline,
    ) -> Result<()> {
        let rt_set: std::collections::HashSet<_> = partition.rt_nodes.iter().copied().collect();

        for &node_id in &partition.rt_nodes {
            // Count RT dependencies
            let mut dep_count = 0u32;
            for (parent_id, _link) in pipeline.parents(node_id) {
                if rt_set.contains(&parent_id) {
                    dep_count += 1;
                }
            }

            // Also count incoming bridges (async → RT)
            for edge in &partition.boundary_edges {
                if edge.sink == node_id && edge.direction == BoundaryDirection::AsyncToRt {
                    dep_count += 1;
                }
            }

            if let Some(activation) = self.activations.get(&node_id) {
                activation.set_required(dep_count);
            }
        }

        Ok(())
    }

    /// Create bridges for boundary edges.
    pub fn create_bridges(&mut self, partition: &GraphPartition) -> Result<()> {
        self.bridges.clear();

        for edge in &partition.boundary_edges {
            let bridge = Arc::new(AsyncRtBridge::new(
                BridgeConfig::with_capacity(self.config.bridge_capacity).with_name(format!(
                    "{}:{} -> {}:{}",
                    edge.source.0.index(),
                    edge.source_pad,
                    edge.sink.0.index(),
                    edge.sink_pad
                )),
            )?);
            self.bridges.insert((edge.source, edge.sink), bridge);
        }

        Ok(())
    }

    /// Get a bridge for a boundary edge.
    pub fn get_bridge(&self, source: NodeId, sink: NodeId) -> Option<Arc<AsyncRtBridge>> {
        self.bridges.get(&(source, sink)).cloned()
    }

    /// Get the activation record for a node.
    pub fn get_activation(&self, node_id: NodeId) -> Option<Arc<ActivationRecord>> {
        self.activations.get(&node_id).cloned()
    }

    /// Get the processing order.
    pub fn processing_order(&self) -> &[NodeId] {
        &self.processing_order
    }

    /// Set the driver node.
    pub fn set_driver(&mut self, node_id: NodeId) {
        self.driver = Some(node_id);
    }

    /// Get the driver node.
    pub fn driver(&self) -> Option<NodeId> {
        self.driver
    }

    /// Select a driver node automatically.
    ///
    /// Chooses the first sink node in the RT partition as the driver.
    pub fn select_driver(&mut self, partition: &GraphPartition, pipeline: &Pipeline) {
        // Find sinks in RT partition
        for &node_id in &partition.rt_nodes {
            if let Some(node) = pipeline.get_node(node_id) {
                if node.element_type() == crate::element::ElementType::Sink {
                    self.driver = Some(node_id);
                    return;
                }
            }
        }

        // If no sink, use the last node in processing order
        if let Some(&last) = self.processing_order.last() {
            self.driver = Some(last);
        }
    }
}

// ============================================================================
// Data Thread
// ============================================================================

/// Handle to a running data thread.
pub struct DataThreadHandle {
    /// Thread join handle.
    handle: Option<JoinHandle<Result<()>>>,

    /// Signal to stop the thread.
    stop_signal: Arc<std::sync::atomic::AtomicBool>,
}

impl DataThreadHandle {
    /// Check if the thread is still running.
    pub fn is_running(&self) -> bool {
        self.handle.as_ref().map_or(false, |h| !h.is_finished())
    }

    /// Signal the thread to stop.
    pub fn signal_stop(&self) {
        self.stop_signal
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Wait for the thread to finish.
    pub fn join(mut self) -> Result<()> {
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| Error::InvalidSegment("data thread panicked".into()))?
        } else {
            Ok(())
        }
    }
}

impl Drop for DataThreadHandle {
    fn drop(&mut self) {
        self.signal_stop();
    }
}

/// Spawn a data thread for RT processing.
///
/// The data thread runs a tight loop processing nodes in dependency order.
pub fn spawn_data_thread(
    name: String,
    config: RtConfig,
    processing_order: Vec<NodeId>,
    activations: HashMap<NodeId, Arc<ActivationRecord>>,
    mut elements: HashMap<NodeId, Box<DynAsyncElement<'static>>>,
    input_bridges: HashMap<NodeId, Arc<AsyncRtBridge>>,
    output_bridges: HashMap<NodeId, Arc<AsyncRtBridge>>,
    driver_trigger: Arc<EventFd>,
) -> Result<DataThreadHandle> {
    let stop_signal = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stop_signal_clone = stop_signal.clone();

    let handle = std::thread::Builder::new()
        .name(name.clone())
        .spawn(move || {
            // Set RT priority if configured
            #[cfg(target_os = "linux")]
            if let Some(priority) = config.rt_priority {
                if let Err(e) = set_rt_priority(priority) {
                    tracing::warn!("failed to set RT priority: {}", e);
                }
            }

            tracing::info!("data thread '{}' started", name);

            // Main processing loop
            while !stop_signal_clone.load(std::sync::atomic::Ordering::Acquire) {
                // Wait for driver signal (start of cycle)
                match driver_trigger.try_wait() {
                    Ok(true) => {}
                    Ok(false) => {
                        // No signal yet, yield
                        std::hint::spin_loop();
                        continue;
                    }
                    Err(e) => {
                        tracing::error!("driver trigger error: {}", e);
                        return Err(e);
                    }
                }

                // Reset pending counts for all nodes
                for activation in activations.values() {
                    activation.reset_pending();
                }

                // Process nodes in dependency order
                for &node_id in &processing_order {
                    // Wait until this node's dependencies are satisfied
                    if let Some(activation) = activations.get(&node_id) {
                        while !activation.is_ready() {
                            std::hint::spin_loop();
                        }

                        activation.set_status(NodeStatus::Processing);
                    }

                    // Get input from bridge if this is a boundary node
                    let input = if let Some(bridge) = input_bridges.get(&node_id) {
                        bridge.try_pop()
                    } else {
                        None
                    };

                    // Process the element synchronously
                    // Note: We use a blocking approach here since RT threads
                    // shouldn't use async
                    if let Some(element) = elements.get_mut(&node_id) {
                        // For now, we need to block on the async process
                        // In a full implementation, RT elements would have a sync interface
                        let rt = tokio::runtime::Handle::try_current();
                        let result = if let Ok(handle) = rt {
                            handle.block_on(element.process(input))
                        } else {
                            // No Tokio runtime, create a minimal one
                            let rt = tokio::runtime::Builder::new_current_thread()
                                .build()
                                .map_err(|e| Error::Io(e))?;
                            rt.block_on(element.process(input))
                        };

                        match result {
                            Ok(Some(buffer)) => {
                                // Send to output bridge if this is a boundary node
                                if let Some(bridge) = output_bridges.get(&node_id) {
                                    if let Err(buf) = bridge.try_push(buffer) {
                                        tracing::warn!(
                                            "node {:?}: output bridge full, dropping buffer",
                                            node_id
                                        );
                                        drop(buf);
                                    }
                                }
                            }
                            Ok(None) => {
                                // EOS or filtered
                            }
                            Err(e) => {
                                tracing::error!("node {:?} error: {}", node_id, e);
                            }
                        }
                    }

                    // Signal downstream nodes
                    if let Some(activation) = activations.get(&node_id) {
                        activation.set_status(NodeStatus::HaveData);
                    }

                    // Decrement pending count of downstream RT nodes
                    // (In a full implementation, we'd track the dependency graph)
                }
            }

            tracing::info!("data thread '{}' stopped", name);
            Ok(())
        })
        .map_err(|e| Error::Io(e))?;

    Ok(DataThreadHandle {
        handle: Some(handle),
        stop_signal,
    })
}

/// Set real-time thread priority (Linux-specific).
#[cfg(target_os = "linux")]
fn set_rt_priority(priority: i32) -> Result<()> {
    // SCHED_FIFO = 1
    const SCHED_FIFO: libc::c_int = 1;

    let param = libc::sched_param {
        sched_priority: priority,
    };

    // Set scheduler for current thread (tid = 0)
    let result = unsafe { libc::sched_setscheduler(0, SCHED_FIFO, &param) };

    if result == -1 {
        let err = std::io::Error::last_os_error();
        return Err(Error::Io(std::io::Error::other(format!(
            "sched_setscheduler failed: {} (hint: requires CAP_SYS_NICE or root)",
            err
        ))));
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn set_rt_priority(_priority: i32) -> Result<()> {
    // Not supported on non-Linux platforms
    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduling_mode_default() {
        assert_eq!(SchedulingMode::default(), SchedulingMode::Async);
    }

    #[test]
    fn test_rt_config_defaults() {
        let config = RtConfig::default();
        assert_eq!(config.mode, SchedulingMode::Async);
        assert_eq!(config.quantum, 256);
        assert_eq!(config.rt_priority, None);
        assert_eq!(config.data_threads, 1);
    }

    #[test]
    fn test_rt_config_presets() {
        let async_config = RtConfig::async_only();
        assert_eq!(async_config.mode, SchedulingMode::Async);
        assert_eq!(async_config.data_threads, 0);

        let hybrid_config = RtConfig::hybrid();
        assert_eq!(hybrid_config.mode, SchedulingMode::Hybrid);

        let audio_config = RtConfig::low_latency_audio();
        assert_eq!(audio_config.mode, SchedulingMode::Hybrid);
        assert_eq!(audio_config.quantum, 64);
        assert_eq!(audio_config.rt_priority, Some(50));
    }

    #[test]
    fn test_node_status_conversion() {
        assert_eq!(NodeStatus::from(0), NodeStatus::Idle);
        assert_eq!(NodeStatus::from(1), NodeStatus::NeedData);
        assert_eq!(NodeStatus::from(2), NodeStatus::HaveData);
        assert_eq!(NodeStatus::from(3), NodeStatus::Processing);
        assert_eq!(NodeStatus::from(255), NodeStatus::Idle); // Unknown defaults to Idle
    }

    #[test]
    fn test_activation_record() {
        let activation = ActivationRecord::new().unwrap();

        // Initially no dependencies
        assert!(activation.is_ready());

        // Set 2 dependencies
        activation.set_required(2);
        activation.reset_pending();
        assert!(!activation.is_ready());

        // Decrement once
        assert!(!activation.decrement_pending());
        assert!(!activation.is_ready());

        // Decrement again - now ready
        assert!(activation.decrement_pending());
        assert!(activation.is_ready());
    }

    #[test]
    fn test_graph_partition() {
        let partition = GraphPartition::new();
        assert!(!partition.has_rt_nodes());
        assert!(!partition.has_boundaries());
    }

    #[test]
    fn test_boundary_direction() {
        use daggy::NodeIndex;
        let edge = BoundaryEdge {
            source: NodeId(NodeIndex::new(0)),
            source_pad: "src".into(),
            sink: NodeId(NodeIndex::new(1)),
            sink_pad: "sink".into(),
            direction: BoundaryDirection::AsyncToRt,
        };
        assert_eq!(edge.direction, BoundaryDirection::AsyncToRt);
    }

    #[test]
    fn test_rt_scheduler_creation() {
        let scheduler = RtScheduler::new(RtConfig::hybrid());
        assert_eq!(scheduler.config().mode, SchedulingMode::Hybrid);
        assert!(scheduler.processing_order().is_empty());
        assert!(scheduler.driver().is_none());
    }
}
