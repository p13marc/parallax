//! Unified pipeline executor with automatic execution strategy.
//!
//! A single executor that handles all execution modes:
//! - **Auto** (default): Automatically determines optimal strategy per element
//! - Async: All elements run as Tokio tasks
//! - Hybrid: Mix of async tasks and RT threads
//! - Isolated: Process isolation for untrusted elements
//!
//! # Automatic Mode
//!
//! In automatic mode (default), the executor analyzes each element's
//! [`ExecutionHints`] to determine the best execution strategy:
//!
//! - **Untrusted elements** (decoders, parsers) → Process isolation
//! - **Low-latency elements** (audio processing) → RT threads
//! - **I/O-bound elements** (network, file) → Async tasks
//! - **CPU-bound elements** → Dedicated threads or RT threads
//!
//! # Example
//!
//! ```rust,ignore
//! use parallax::pipeline::{Pipeline, Executor};
//!
//! let mut pipeline = Pipeline::parse("filesrc ! h264dec ! display")?;
//!
//! // Automatic mode (default) - the executor will:
//! // - Run filesrc as async (I/O-bound)
//! // - Isolate h264dec (untrusted, native code)
//! // - Run display as async or RT based on latency needs
//! pipeline.run().await?;
//! ```

use crate::buffer::Buffer;
use crate::element::{
    Affinity, AsyncElementDyn, DynAsyncElement, ElementType, ExecutionHints, LatencyHint,
    ProcessingHint, TrustLevel,
};
use crate::error::{Error, Result};
use crate::execution::ExecutionMode;
use crate::pipeline::rt_bridge::AsyncRtBridge;
use crate::pipeline::rt_scheduler::{
    BoundaryDirection, GraphPartition, RtConfig, RtScheduler, SchedulingMode,
};
use crate::pipeline::{
    DriverConfig, EventReceiver, EventSender, NodeId, Pipeline, PipelineEvent, PipelineState,
    TimerDriver,
};
use kanal::{AsyncReceiver, AsyncSender, bounded_async};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use tokio::task::JoinHandle;

// ============================================================================
// Configuration
// ============================================================================

/// Unified executor configuration.
#[derive(Debug, Clone)]
pub struct ExecutorConfig {
    /// Scheduling mode (Auto, Async, Hybrid, RealTime).
    /// Auto (default) analyzes element hints to determine the best strategy.
    pub scheduling: SchedulingMode,

    /// Channel buffer size between elements.
    pub channel_capacity: usize,

    /// RT scheduling configuration (for Hybrid/RealTime modes).
    pub rt: RtConfig,

    /// Process isolation mode.
    /// None = auto-detect based on element hints.
    /// Some(mode) = use specified mode.
    pub isolation: Option<ExecutionMode>,

    /// Driver configuration (for timed execution).
    pub driver: Option<DriverConfig>,

    /// Enable automatic strategy detection from element hints.
    /// When true (default), the executor analyzes ExecutionHints to determine
    /// optimal scheduling and isolation per element.
    pub auto_strategy: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            scheduling: SchedulingMode::Async,
            channel_capacity: 16,
            rt: RtConfig::default(),
            isolation: None,
            driver: None,
            auto_strategy: true, // Enable automatic by default
        }
    }
}

impl ExecutorConfig {
    /// Create config for automatic strategy detection (default).
    ///
    /// The executor will analyze each element's `ExecutionHints` to determine:
    /// - Which elements need process isolation (untrusted, native code)
    /// - Which elements should run in RT threads (low-latency)
    /// - Which elements are I/O-bound (async tasks)
    pub fn auto() -> Self {
        Self::default()
    }

    /// Create config for pure async execution (no automatic detection).
    pub fn async_only() -> Self {
        Self {
            auto_strategy: false,
            ..Default::default()
        }
    }

    /// Create config for hybrid async + RT execution.
    pub fn hybrid() -> Self {
        Self {
            scheduling: SchedulingMode::Hybrid,
            rt: RtConfig::hybrid(),
            auto_strategy: false,
            ..Default::default()
        }
    }

    /// Create config for low-latency audio.
    pub fn low_latency_audio() -> Self {
        Self {
            scheduling: SchedulingMode::Hybrid,
            rt: RtConfig::low_latency_audio(),
            driver: Some(DriverConfig::low_latency_audio()),
            auto_strategy: false,
            ..Default::default()
        }
    }

    /// Create config for video processing.
    pub fn video(fps: u32) -> Self {
        Self {
            scheduling: SchedulingMode::Hybrid,
            rt: RtConfig::hybrid(),
            driver: Some(DriverConfig::video(fps)),
            auto_strategy: false,
            ..Default::default()
        }
    }

    /// Create config with process isolation for all elements.
    pub fn isolated() -> Self {
        Self {
            isolation: Some(ExecutionMode::isolated()),
            auto_strategy: false,
            ..Default::default()
        }
    }

    /// Set scheduling mode.
    pub fn with_scheduling(mut self, mode: SchedulingMode) -> Self {
        self.scheduling = mode;
        self
    }

    /// Set channel capacity.
    pub fn with_channel_capacity(mut self, capacity: usize) -> Self {
        self.channel_capacity = capacity;
        self
    }

    /// Set RT priority (requires CAP_SYS_NICE).
    pub fn with_rt_priority(mut self, priority: i32) -> Self {
        self.rt.rt_priority = Some(priority);
        self
    }

    /// Set quantum (samples per cycle).
    pub fn with_quantum(mut self, quantum: u32) -> Self {
        self.rt.quantum = quantum;
        self
    }

    /// Set process isolation mode.
    pub fn with_isolation(mut self, mode: ExecutionMode) -> Self {
        self.isolation = Some(mode);
        self
    }

    /// Set driver configuration.
    pub fn with_driver(mut self, driver: DriverConfig) -> Self {
        self.driver = Some(driver);
        self
    }

    /// Disable automatic strategy detection.
    pub fn without_auto_strategy(mut self) -> Self {
        self.auto_strategy = false;
        self
    }
}

// ============================================================================
// Execution Strategy (per-element)
// ============================================================================

/// Execution strategy for a single element, determined by analyzing hints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ElementStrategy {
    /// Run in Tokio async runtime.
    Async,
    /// Run in dedicated RT thread.
    RealTime,
    /// Run in isolated process.
    Isolated,
}

/// Analyzed execution plan for the entire pipeline.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Strategy per node.
    pub strategies: HashMap<NodeId, ElementStrategy>,
    /// Nodes that need isolation.
    pub isolated_nodes: HashSet<NodeId>,
    /// Nodes that need RT scheduling.
    pub rt_nodes: HashSet<NodeId>,
    /// Nodes that can run async.
    pub async_nodes: HashSet<NodeId>,
    /// Whether any isolation is needed.
    pub needs_isolation: bool,
    /// Whether any RT scheduling is needed.
    pub needs_rt: bool,
}

impl ExecutionPlan {
    /// Create an empty plan (all async).
    pub fn all_async() -> Self {
        Self {
            strategies: HashMap::new(),
            isolated_nodes: HashSet::new(),
            rt_nodes: HashSet::new(),
            async_nodes: HashSet::new(),
            needs_isolation: false,
            needs_rt: false,
        }
    }
}

/// Analyze a pipeline and determine the optimal execution strategy for each element.
fn analyze_pipeline(pipeline: &Pipeline) -> ExecutionPlan {
    let mut plan = ExecutionPlan::all_async();

    for node_id in pipeline.node_ids() {
        if let Some(node) = pipeline.get_node(node_id) {
            // Get hints from node (which delegates to element if present)
            let hints = node.execution_hints();
            let affinity = node.affinity();
            let rt_safe = node.is_rt_safe();

            let strategy = determine_element_strategy(&hints, affinity, rt_safe);

            match strategy {
                ElementStrategy::Isolated => {
                    plan.isolated_nodes.insert(node_id);
                    plan.needs_isolation = true;
                }
                ElementStrategy::RealTime => {
                    plan.rt_nodes.insert(node_id);
                    plan.needs_rt = true;
                }
                ElementStrategy::Async => {
                    plan.async_nodes.insert(node_id);
                }
            }

            plan.strategies.insert(node_id, strategy);
        }
    }

    tracing::debug!(
        "Execution plan: {} async, {} RT, {} isolated",
        plan.async_nodes.len(),
        plan.rt_nodes.len(),
        plan.isolated_nodes.len()
    );

    plan
}

/// Determine the execution strategy for a single element based on its hints.
fn determine_element_strategy(
    hints: &ExecutionHints,
    affinity: Affinity,
    rt_safe: bool,
) -> ElementStrategy {
    // Rule 1: Untrusted elements should be isolated
    if hints.trust_level == TrustLevel::Untrusted {
        return ElementStrategy::Isolated;
    }

    // Rule 2: Elements using native code that might crash should be isolated
    if hints.uses_native_code && !hints.crash_safe {
        return ElementStrategy::Isolated;
    }

    // Rule 3: Elements that explicitly request RT affinity
    if affinity == Affinity::RealTime {
        if rt_safe {
            return ElementStrategy::RealTime;
        } else {
            // Requested RT but not RT-safe - log warning and use async
            tracing::warn!(
                "Element requested RealTime affinity but is_rt_safe() returned false, using Async"
            );
            return ElementStrategy::Async;
        }
    }

    // Rule 4: Ultra-low latency requirements need RT
    if hints.latency == LatencyHint::UltraLow && rt_safe {
        return ElementStrategy::RealTime;
    }

    // Rule 5: Low latency with RT-safe element can use RT
    if hints.latency == LatencyHint::Low && rt_safe {
        return ElementStrategy::RealTime;
    }

    // Rule 6: CPU-bound elements can benefit from RT if they're RT-safe
    if hints.processing == ProcessingHint::CpuBound && rt_safe {
        // Only use RT for CPU-bound if we also have low latency requirements
        if matches!(hints.latency, LatencyHint::UltraLow | LatencyHint::Low) {
            return ElementStrategy::RealTime;
        }
    }

    // Rule 7: I/O-bound elements should always use async
    if hints.processing == ProcessingHint::IoBound {
        return ElementStrategy::Async;
    }

    // Rule 8: Explicit async affinity
    if affinity == Affinity::Async {
        return ElementStrategy::Async;
    }

    // Default: async is safest
    ElementStrategy::Async
}

// ============================================================================
// Handle
// ============================================================================

/// Handle to a running pipeline.
pub struct PipelineHandle {
    /// Tokio task handles.
    tasks: Vec<JoinHandle<Result<()>>>,
    /// RT thread handles (if any).
    rt_handles: Vec<crate::pipeline::rt_scheduler::DataThreadHandle>,
    /// Event sender.
    events: EventSender,
    /// Bridges (kept alive).
    #[allow(dead_code)]
    bridges: Vec<Arc<AsyncRtBridge>>,
    /// Driver handle (if any).
    #[allow(dead_code)]
    driver: Option<crate::pipeline::TimerDriverHandle>,
}

impl PipelineHandle {
    /// Wait for the pipeline to complete.
    pub async fn wait(mut self) -> Result<()> {
        let mut first_error = None;

        // Wait for all Tokio tasks
        for task in self.tasks {
            match task.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    self.events.send_error(e.to_string(), None);
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
                Err(e) => {
                    let err = Error::InvalidSegment(format!("task panicked: {e}"));
                    self.events.send_error(err.to_string(), None);
                    if first_error.is_none() {
                        first_error = Some(err);
                    }
                }
            }
        }

        // Signal RT threads to stop and wait
        for handle in self.rt_handles.drain(..) {
            handle.signal_stop();
            if let Err(e) = handle.join() {
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }

        if first_error.is_none() {
            self.events.send_eos();
        }

        match first_error {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    /// Abort all pipeline tasks.
    pub fn abort(mut self) {
        for task in self.tasks {
            task.abort();
        }

        for handle in self.rt_handles.drain(..) {
            handle.signal_stop();
            let _ = handle.join();
        }

        self.events.send(PipelineEvent::Stopped);
    }

    /// Subscribe to pipeline events.
    pub fn subscribe(&self) -> EventReceiver {
        self.events.subscribe()
    }

    /// Get the event sender.
    pub fn event_sender(&self) -> &EventSender {
        &self.events
    }
}

// ============================================================================
// Executor
// ============================================================================

/// Unified pipeline executor.
///
/// Handles all execution modes through a single interface.
pub struct Executor {
    config: ExecutorConfig,
}

impl Executor {
    /// Create a new executor with default configuration.
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
        }
    }

    /// Create an executor with custom configuration.
    pub fn with_config(config: ExecutorConfig) -> Self {
        Self { config }
    }

    /// Get the configuration.
    pub fn config(&self) -> &ExecutorConfig {
        &self.config
    }

    /// Run the pipeline to completion.
    pub async fn run(&self, pipeline: &mut Pipeline) -> Result<()> {
        let handle = self.start(pipeline)?;
        handle.wait().await
    }

    /// Start the pipeline and return a handle.
    pub fn start(&self, pipeline: &mut Pipeline) -> Result<PipelineHandle> {
        // Create event sender
        let events = EventSender::new(256);

        // Analyze pipeline for automatic strategy if enabled
        let plan = if self.config.auto_strategy {
            let plan = analyze_pipeline(pipeline);
            tracing::info!(
                "Auto-detected execution plan: {} async, {} RT, {} isolated",
                plan.async_nodes.len(),
                plan.rt_nodes.len(),
                plan.isolated_nodes.len()
            );
            Some(plan)
        } else {
            None
        };

        // Handle process isolation
        let needs_isolation =
            plan.as_ref().is_some_and(|p| p.needs_isolation) || self.config.isolation.is_some();

        if needs_isolation {
            // TODO: For now, log and continue without isolation
            // Full isolation support requires IsolatedExecutor integration
            if let Some(ref plan) = plan {
                for node_id in &plan.isolated_nodes {
                    if let Some(node) = pipeline.get_node(*node_id) {
                        tracing::warn!(
                            "Element '{}' should be isolated (untrusted/native), but isolation not yet implemented",
                            node.name()
                        );
                    }
                }
            }
        }

        // State transitions
        let old_state = pipeline.state();
        if old_state == PipelineState::Suspended {
            pipeline.prepare()?;
            events.send_state_changed(old_state, PipelineState::Idle);
        }

        // Determine effective scheduling mode
        let effective_scheduling = if self.config.auto_strategy {
            if let Some(ref plan) = plan {
                if plan.needs_rt {
                    SchedulingMode::Hybrid
                } else {
                    SchedulingMode::Async
                }
            } else {
                self.config.scheduling
            }
        } else {
            self.config.scheduling
        };

        // Partition graph for hybrid scheduling
        let mut scheduler = RtScheduler::new(self.config.rt.clone());
        let partition = if effective_scheduling != SchedulingMode::Async {
            scheduler.partition_graph(pipeline)?
        } else {
            // All async - create empty partition
            GraphPartition {
                async_nodes: pipeline.node_ids(),
                rt_nodes: Vec::new(),
                boundary_edges: Vec::new(),
            }
        };

        tracing::debug!(
            "Graph partition: {} async, {} RT, {} boundaries (mode: {:?})",
            partition.async_nodes.len(),
            partition.rt_nodes.len(),
            partition.boundary_edges.len(),
            effective_scheduling
        );

        // Execute based on scheduling mode
        let (tasks, rt_handles, bridges) = match effective_scheduling {
            SchedulingMode::Async => {
                let tasks = self.run_async(pipeline, &events)?;
                (tasks, Vec::new(), Vec::new())
            }
            SchedulingMode::Hybrid | SchedulingMode::RealTime => {
                if partition.rt_nodes.is_empty() {
                    // No RT nodes, fall back to async
                    let tasks = self.run_async(pipeline, &events)?;
                    (tasks, Vec::new(), Vec::new())
                } else {
                    self.run_hybrid(pipeline, &partition, &mut scheduler, &events)?
                }
            }
        };

        // Start driver if configured
        let driver = self.config.driver.as_ref().map(|config| {
            let driver = TimerDriver::new(config.clone());
            driver.start_async()
        });

        // Activate (Idle → Running)
        let idle_state = pipeline.state();
        pipeline.activate()?;
        events.send_state_changed(idle_state, PipelineState::Running);
        events.send(PipelineEvent::Started);

        Ok(PipelineHandle {
            tasks,
            rt_handles,
            events,
            bridges,
            driver,
        })
    }

    /// Run all nodes as async Tokio tasks.
    fn run_async(
        &self,
        pipeline: &mut Pipeline,
        events: &EventSender,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut channels = ChannelNetwork::new();

        // Build channels
        for src_id in pipeline.sources() {
            self.build_channels(pipeline, src_id, &mut channels);
        }

        // Spawn tasks
        self.spawn_tasks(pipeline, channels, events)
    }

    /// Run with hybrid async + RT execution.
    fn run_hybrid(
        &self,
        pipeline: &mut Pipeline,
        partition: &GraphPartition,
        scheduler: &mut RtScheduler,
        events: &EventSender,
    ) -> Result<(
        Vec<JoinHandle<Result<()>>>,
        Vec<crate::pipeline::rt_scheduler::DataThreadHandle>,
        Vec<Arc<AsyncRtBridge>>,
    )> {
        // Create bridges at boundaries
        scheduler.create_bridges(partition)?;
        scheduler.setup_activations(partition)?;
        scheduler.compute_processing_order(partition, pipeline)?;
        scheduler.setup_dependencies(partition, pipeline)?;
        scheduler.select_driver(partition, pipeline);

        // Build channels for async nodes
        let mut channels = ChannelNetwork::new();
        let async_set: std::collections::HashSet<_> =
            partition.async_nodes.iter().copied().collect();

        for src_id in pipeline.sources() {
            if async_set.contains(&src_id) {
                self.build_channels_for_async(
                    pipeline,
                    src_id,
                    &async_set,
                    partition,
                    scheduler,
                    &mut channels,
                );
            }
        }

        // Spawn async tasks
        let tasks =
            self.spawn_tasks_for_partition(pipeline, partition, channels, scheduler, events)?;

        // Collect bridges
        let bridges: Vec<_> = partition
            .boundary_edges
            .iter()
            .filter_map(|e| scheduler.get_bridge(e.source, e.sink))
            .collect();

        // TODO: spawn RT threads
        let rt_handles = Vec::new();

        Ok((tasks, rt_handles, bridges))
    }

    /// Build channel network recursively.
    fn build_channels(&self, pipeline: &Pipeline, node_id: NodeId, network: &mut ChannelNetwork) {
        for (child_id, link) in pipeline.children(node_id) {
            if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                let (tx, rx) = bounded_async::<Message>(self.config.channel_capacity);
                network.add_channel(
                    node_id,
                    link.src_pad.clone(),
                    child_id,
                    link.sink_pad.clone(),
                    tx,
                    rx,
                );
            }
            self.build_channels(pipeline, child_id, network);
        }
    }

    /// Build channels for async portion only.
    fn build_channels_for_async(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        async_set: &std::collections::HashSet<NodeId>,
        partition: &GraphPartition,
        _scheduler: &RtScheduler,
        network: &mut ChannelNetwork,
    ) {
        for (child_id, link) in pipeline.children(node_id) {
            let is_boundary = partition
                .boundary_edges
                .iter()
                .any(|e| e.source == node_id && e.sink == child_id);

            if is_boundary {
                continue; // Bridge handles this
            }

            if async_set.contains(&child_id) {
                if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                    let (tx, rx) = bounded_async::<Message>(self.config.channel_capacity);
                    network.add_channel(
                        node_id,
                        link.src_pad.clone(),
                        child_id,
                        link.sink_pad.clone(),
                        tx,
                        rx,
                    );
                }
                self.build_channels_for_async(
                    pipeline, child_id, async_set, partition, _scheduler, network,
                );
            }
        }
    }

    /// Spawn tasks for all nodes.
    fn spawn_tasks(
        &self,
        pipeline: &mut Pipeline,
        mut channels: ChannelNetwork,
        events: &EventSender,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut tasks = Vec::new();

        let node_ids: Vec<NodeId> = pipeline
            .sources()
            .into_iter()
            .chain(self.collect_reachable(pipeline))
            .collect();

        let mut seen = std::collections::HashSet::new();
        let node_ids: Vec<NodeId> = node_ids.into_iter().filter(|id| seen.insert(*id)).collect();

        for node_id in node_ids {
            let task = self.spawn_node_task(pipeline, node_id, &mut channels, events)?;
            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Spawn tasks for async partition only.
    fn spawn_tasks_for_partition(
        &self,
        pipeline: &mut Pipeline,
        partition: &GraphPartition,
        mut channels: ChannelNetwork,
        scheduler: &RtScheduler,
        events: &EventSender,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut tasks = Vec::new();

        for &node_id in &partition.async_nodes {
            let output_bridges: Vec<_> = partition
                .boundary_edges
                .iter()
                .filter(|e| e.source == node_id && e.direction == BoundaryDirection::AsyncToRt)
                .filter_map(|e| scheduler.get_bridge(e.source, e.sink))
                .collect();

            let input_bridges: Vec<_> = partition
                .boundary_edges
                .iter()
                .filter(|e| e.sink == node_id && e.direction == BoundaryDirection::RtToAsync)
                .filter_map(|e| scheduler.get_bridge(e.source, e.sink))
                .collect();

            let task = self.spawn_node_task_with_bridges(
                pipeline,
                node_id,
                &mut channels,
                output_bridges,
                input_bridges,
                events,
            )?;
            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Spawn a task for a single node.
    fn spawn_node_task(
        &self,
        pipeline: &mut Pipeline,
        node_id: NodeId,
        channels: &mut ChannelNetwork,
        events: &EventSender,
    ) -> Result<JoinHandle<Result<()>>> {
        self.spawn_node_task_with_bridges(
            pipeline,
            node_id,
            channels,
            Vec::new(),
            Vec::new(),
            events,
        )
    }

    /// Spawn a task with optional bridges.
    fn spawn_node_task_with_bridges(
        &self,
        pipeline: &mut Pipeline,
        node_id: NodeId,
        channels: &mut ChannelNetwork,
        output_bridges: Vec<Arc<AsyncRtBridge>>,
        input_bridges: Vec<Arc<AsyncRtBridge>>,
        events: &EventSender,
    ) -> Result<JoinHandle<Result<()>>> {
        let node = pipeline
            .get_node_mut(node_id)
            .ok_or_else(|| Error::InvalidSegment("node not found".into()))?;

        let element_type = node.element_type();
        let node_name = node.name().to_string();

        let element = node.take_element().ok_or_else(|| {
            Error::InvalidSegment(format!("element '{}' already taken", node_name))
        })?;

        let inputs = channels.take_inputs(node_id);
        let outputs = channels.take_outputs(node_id);
        let events_clone = events.clone();

        let task = match element_type {
            ElementType::Source => {
                spawn_source_task(node_name, element, outputs, output_bridges, events_clone)
            }
            ElementType::Sink => {
                spawn_sink_task(node_name, element, inputs, input_bridges, events_clone)
            }
            ElementType::Transform => spawn_transform_task(
                node_name,
                element,
                inputs,
                outputs,
                input_bridges,
                output_bridges,
                events_clone,
            ),
            ElementType::Demuxer => {
                let outputs_by_pad = channels.take_outputs_by_pad(node_id);
                spawn_demuxer_task(node_name, element, inputs, outputs_by_pad, events_clone)
            }
            ElementType::Muxer => {
                let inputs_by_pad = channels.take_inputs_by_pad(node_id);
                spawn_muxer_task(node_name, element, inputs_by_pad, outputs, events_clone)
            }
        };

        Ok(task)
    }

    /// Collect reachable nodes from sources.
    fn collect_reachable(&self, pipeline: &Pipeline) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();

        for src in pipeline.sources() {
            self.collect_from(pipeline, src, &mut result, &mut visited);
        }

        result
    }

    fn collect_from(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        result: &mut Vec<NodeId>,
        visited: &mut std::collections::HashSet<NodeId>,
    ) {
        if !visited.insert(node_id) {
            return;
        }

        for (child_id, _) in pipeline.children(node_id) {
            result.push(child_id);
            self.collect_from(pipeline, child_id, result, visited);
        }
    }
}

impl Default for Executor {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Internal Types
// ============================================================================

#[derive(Debug)]
enum Message {
    Buffer(Buffer),
    Eos,
}

type ChannelKey = (NodeId, String, NodeId, String);

struct ChannelNetwork {
    channels: HashMap<ChannelKey, (AsyncSender<Message>, AsyncReceiver<Message>)>,
    outputs: HashMap<(NodeId, String), Vec<AsyncSender<Message>>>,
    inputs: HashMap<(NodeId, String), Vec<AsyncReceiver<Message>>>,
}

impl ChannelNetwork {
    fn new() -> Self {
        Self {
            channels: HashMap::new(),
            outputs: HashMap::new(),
            inputs: HashMap::new(),
        }
    }

    fn has_channel(&self, src: NodeId, src_pad: &str, sink: NodeId, sink_pad: &str) -> bool {
        self.channels
            .contains_key(&(src, src_pad.to_string(), sink, sink_pad.to_string()))
    }

    fn add_channel(
        &mut self,
        src: NodeId,
        src_pad: String,
        sink: NodeId,
        sink_pad: String,
        tx: AsyncSender<Message>,
        rx: AsyncReceiver<Message>,
    ) {
        self.channels.insert(
            (src, src_pad.clone(), sink, sink_pad.clone()),
            (tx.clone(), rx.clone()),
        );
        self.outputs.entry((src, src_pad)).or_default().push(tx);
        self.inputs.entry((sink, sink_pad)).or_default().push(rx);
    }

    fn take_outputs_by_pad(&mut self, node: NodeId) -> HashMap<String, Vec<AsyncSender<Message>>> {
        let mut result = HashMap::new();
        let keys: Vec<_> = self
            .outputs
            .keys()
            .filter(|(n, _)| *n == node)
            .cloned()
            .collect();
        for (n, pad) in keys {
            if let Some(senders) = self.outputs.remove(&(n, pad.clone())) {
                result.insert(pad, senders);
            }
        }
        result
    }

    fn take_inputs_by_pad(&mut self, node: NodeId) -> HashMap<String, Vec<AsyncReceiver<Message>>> {
        let mut result = HashMap::new();
        let keys: Vec<_> = self
            .inputs
            .keys()
            .filter(|(n, _)| *n == node)
            .cloned()
            .collect();
        for (n, pad) in keys {
            if let Some(receivers) = self.inputs.remove(&(n, pad.clone())) {
                result.insert(pad, receivers);
            }
        }
        result
    }

    fn take_outputs(&mut self, node: NodeId) -> Vec<AsyncSender<Message>> {
        self.take_outputs_by_pad(node)
            .into_values()
            .flatten()
            .collect()
    }

    fn take_inputs(&mut self, node: NodeId) -> Vec<AsyncReceiver<Message>> {
        self.take_inputs_by_pad(node)
            .into_values()
            .flatten()
            .collect()
    }
}

// ============================================================================
// Task Spawning
// ============================================================================

fn spawn_source_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    outputs: Vec<AsyncSender<Message>>,
    output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("source '{}' started", name);
        events.send_node_started(&name);

        let mut count: u64 = 0;

        loop {
            match element.process(None).await {
                Ok(Some(buffer)) => {
                    count += 1;
                    for tx in &outputs {
                        let _ = tx.send(Message::Buffer(buffer.clone())).await;
                    }
                    for bridge in &output_bridges {
                        let _ = bridge.push_async(buffer.clone()).await;
                    }
                }
                Ok(None) => {
                    for tx in &outputs {
                        let _ = tx.send(Message::Eos).await;
                    }
                    break;
                }
                Err(e) => {
                    events.send_error(e.to_string(), Some(name.clone()));
                    return Err(e);
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    })
}

fn spawn_sink_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    _input_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("sink '{}' started", name);
        events.send_node_started(&name);

        let mut count: u64 = 0;

        if let Some(rx) = inputs.into_iter().next() {
            while let Ok(Message::Buffer(buffer)) = rx.recv().await {
                count += 1;
                if let Err(e) = element.process(Some(buffer)).await {
                    events.send_error(e.to_string(), Some(name.clone()));
                    return Err(e);
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    })
}

fn spawn_transform_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    outputs: Vec<AsyncSender<Message>>,
    _input_bridges: Vec<Arc<AsyncRtBridge>>,
    _output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("transform '{}' started", name);
        events.send_node_started(&name);

        let mut count: u64 = 0;

        if let Some(rx) = inputs.into_iter().next() {
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        count += 1;
                        match element.process(Some(buffer)).await {
                            Ok(Some(out)) => {
                                for tx in &outputs {
                                    let _ = tx.send(Message::Buffer(out.clone())).await;
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                events.send_error(e.to_string(), Some(name.clone()));
                                return Err(e);
                            }
                        }
                    }
                    Ok(Message::Eos) => {
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                    Err(_) => {
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    })
}

fn spawn_demuxer_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    outputs_by_pad: HashMap<String, Vec<AsyncSender<Message>>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("demuxer '{}' started", name);
        events.send_node_started(&name);

        let mut count: u64 = 0;

        if let Some(rx) = inputs.into_iter().next() {
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        count += 1;
                        match element.process(Some(buffer)).await {
                            Ok(Some(out)) => {
                                for senders in outputs_by_pad.values() {
                                    for tx in senders {
                                        let _ = tx.send(Message::Buffer(out.clone())).await;
                                    }
                                }
                            }
                            Ok(None) => {}
                            Err(e) => {
                                events.send_error(e.to_string(), Some(name.clone()));
                                return Err(e);
                            }
                        }
                    }
                    Ok(Message::Eos) | Err(_) => {
                        for senders in outputs_by_pad.values() {
                            for tx in senders {
                                let _ = tx.send(Message::Eos).await;
                            }
                        }
                        break;
                    }
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    })
}

fn spawn_muxer_task(
    name: String,
    mut element: Box<DynAsyncElement<'static>>,
    inputs_by_pad: HashMap<String, Vec<AsyncReceiver<Message>>>,
    outputs: Vec<AsyncSender<Message>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    use futures::stream::{FuturesUnordered, StreamExt};

    tokio::spawn(async move {
        tracing::debug!("muxer '{}' started", name);
        events.send_node_started(&name);

        let mut count: u64 = 0;

        let mut receivers: FuturesUnordered<_> = inputs_by_pad
            .into_iter()
            .flat_map(|(pad, rxs)| {
                rxs.into_iter().map(move |rx| {
                    let p = pad.clone();
                    async move { (p, rx.recv().await) }
                })
            })
            .collect();

        let total = receivers.len();
        let mut eos_count = 0;

        while let Some((_, msg)) = receivers.next().await {
            match msg {
                Ok(Message::Buffer(buffer)) => {
                    count += 1;
                    match element.process(Some(buffer)).await {
                        Ok(Some(out)) => {
                            for tx in &outputs {
                                let _ = tx.send(Message::Buffer(out.clone())).await;
                            }
                        }
                        Ok(None) => {}
                        Err(e) => {
                            events.send_error(e.to_string(), Some(name.clone()));
                            return Err(e);
                        }
                    }
                }
                Ok(Message::Eos) | Err(_) => {
                    eos_count += 1;
                    if eos_count >= total {
                        if let Ok(Some(out)) = element.process(None).await {
                            for tx in &outputs {
                                let _ = tx.send(Message::Buffer(out.clone())).await;
                            }
                        }
                        for tx in &outputs {
                            let _ = tx.send(Message::Eos).await;
                        }
                        break;
                    }
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buffer::MemoryHandle;
    use crate::element::{
        ConsumeContext, DynAsyncElement, ProduceContext, ProduceResult, Sink, SinkAdapter, Source,
        SourceAdapter,
    };
    use crate::memory::HeapSegment;
    use crate::metadata::Metadata;
    use std::sync::atomic::{AtomicU64, Ordering};

    struct CountingSource {
        count: u64,
        max: u64,
    }

    impl Source for CountingSource {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
            if self.count >= self.max {
                return Ok(ProduceResult::Eos);
            }
            let segment = Arc::new(HeapSegment::new(8).unwrap());
            let handle = MemoryHandle::from_segment(segment);
            let buffer = Buffer::new(handle, Metadata::from_sequence(self.count));
            self.count += 1;
            Ok(ProduceResult::OwnBuffer(buffer))
        }
    }

    struct CountingSink {
        received: Arc<AtomicU64>,
    }

    impl Sink for CountingSink {
        fn consume(&mut self, _ctx: &ConsumeContext) -> Result<()> {
            self.received.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }

    #[test]
    fn test_executor_config_defaults() {
        let config = ExecutorConfig::default();
        assert_eq!(config.scheduling, SchedulingMode::Async);
        assert_eq!(config.channel_capacity, 16);
        assert!(config.isolation.is_none());
        assert!(config.driver.is_none());
    }

    #[test]
    fn test_executor_config_presets() {
        let config = ExecutorConfig::hybrid();
        assert_eq!(config.scheduling, SchedulingMode::Hybrid);

        let config = ExecutorConfig::low_latency_audio();
        assert_eq!(config.scheduling, SchedulingMode::Hybrid);
        assert!(config.driver.is_some());

        let config = ExecutorConfig::video(60);
        assert!(config.driver.is_some());
    }

    #[tokio::test]
    async fn test_unified_executor() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 5 })),
        );
        let received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
                received: received.clone(),
            })),
        );
        pipeline.link(src, sink).unwrap();

        let executor = Executor::new();
        executor.run(&mut pipeline).await.unwrap();

        assert_eq!(received.load(Ordering::Relaxed), 5);
        assert_eq!(pipeline.state(), PipelineState::Running);
    }

    #[tokio::test]
    async fn test_unified_executor_with_config() {
        let mut pipeline = Pipeline::new();

        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 3 })),
        );
        let received = Arc::new(AtomicU64::new(0));
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
                received: received.clone(),
            })),
        );
        pipeline.link(src, sink).unwrap();

        let config = ExecutorConfig::default().with_channel_capacity(32);
        let executor = Executor::with_config(config);
        executor.run(&mut pipeline).await.unwrap();

        assert_eq!(received.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_determine_element_strategy_defaults() {
        let hints = ExecutionHints::default();

        // Default hints with Auto affinity -> Async
        let strategy = determine_element_strategy(&hints, Affinity::Auto, false);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_determine_element_strategy_untrusted() {
        let hints = ExecutionHints::untrusted();

        // Untrusted -> Isolated
        let strategy = determine_element_strategy(&hints, Affinity::Auto, false);
        assert_eq!(strategy, ElementStrategy::Isolated);
    }

    #[test]
    fn test_determine_element_strategy_native_unsafe() {
        let hints = ExecutionHints::native();

        // Native code that's not crash-safe -> Isolated
        let strategy = determine_element_strategy(&hints, Affinity::Auto, false);
        assert_eq!(strategy, ElementStrategy::Isolated);
    }

    #[test]
    fn test_determine_element_strategy_rt_affinity() {
        let hints = ExecutionHints::default();

        // Explicit RT affinity with RT-safe -> RealTime
        let strategy = determine_element_strategy(&hints, Affinity::RealTime, true);
        assert_eq!(strategy, ElementStrategy::RealTime);

        // Explicit RT affinity but NOT RT-safe -> Async (with warning)
        let strategy = determine_element_strategy(&hints, Affinity::RealTime, false);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_determine_element_strategy_low_latency() {
        let hints = ExecutionHints::low_latency();

        // Low latency + RT-safe -> RealTime
        let strategy = determine_element_strategy(&hints, Affinity::Auto, true);
        assert_eq!(strategy, ElementStrategy::RealTime);

        // Low latency but NOT RT-safe -> Async
        let strategy = determine_element_strategy(&hints, Affinity::Auto, false);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_determine_element_strategy_io_bound() {
        let hints = ExecutionHints::io_bound();

        // I/O-bound -> always Async
        let strategy = determine_element_strategy(&hints, Affinity::Auto, true);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_execution_plan_analysis() {
        let mut pipeline = Pipeline::new();

        // Add a simple source and sink
        let src = pipeline.add_node(
            "src",
            DynAsyncElement::new_box(SourceAdapter::new(CountingSource { count: 0, max: 5 })),
        );
        let sink = pipeline.add_node(
            "sink",
            DynAsyncElement::new_box(SinkAdapter::new(CountingSink {
                received: Arc::new(AtomicU64::new(0)),
            })),
        );
        pipeline.link(src, sink).unwrap();

        // Analyze the pipeline
        let plan = analyze_pipeline(&pipeline);

        // Default elements should be async
        assert!(!plan.needs_isolation);
        assert!(!plan.needs_rt);
        assert_eq!(plan.async_nodes.len(), 2);
        assert_eq!(plan.rt_nodes.len(), 0);
        assert_eq!(plan.isolated_nodes.len(), 0);
    }

    #[test]
    fn test_executor_config_auto_strategy() {
        // Default config should have auto_strategy enabled
        let config = ExecutorConfig::default();
        assert!(config.auto_strategy);

        // Preset configs should disable auto_strategy
        let config = ExecutorConfig::async_only();
        assert!(!config.auto_strategy);

        let config = ExecutorConfig::hybrid();
        assert!(!config.auto_strategy);

        // without_auto_strategy should disable it
        let config = ExecutorConfig::default().without_auto_strategy();
        assert!(!config.auto_strategy);
    }
}
