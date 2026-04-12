//! Unified pipeline executor with automatic execution strategy.
//!
//! A single executor that handles all execution modes:
//! - **Auto** (default): Automatically determines optimal strategy per element
//! - Async: All elements run as Tokio tasks
//! - Hybrid: Mix of async tasks and RT threads
//!
//! # Automatic Mode
//!
//! In automatic mode (default), the executor analyzes each element's
//! [`ExecutionHints`] to determine the best execution strategy:
//!
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
//! // - Run h264dec based on its hints
//! // - Run display as async or RT based on latency needs
//! pipeline.run().await?;
//! ```

use crate::buffer::Buffer;
use crate::clock::{Clock, ClockTime};
use crate::element::{
    AsyncElementDyn, DynAsyncElement, ElementType, ExecutionHints, LatencyHint, Output,
    ProcessingHint, SourceResult,
};
use crate::error::{Error, Result};
use crate::pipeline::rt_bridge::AsyncRtBridge;
use crate::pipeline::rt_scheduler::{
    BoundaryDirection, GraphPartition, RtConfig, RtScheduler, SchedulingMode,
};
use crate::pipeline::bus::{Bus, BusHandle};
use crate::pipeline::probe::{ProbeRegistry, ProbeReturn};
use crate::pipeline::tracer::TracerRegistry;
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

    /// Driver configuration (for timed execution).
    pub driver: Option<DriverConfig>,

    /// Enable automatic strategy detection from element hints.
    /// When true (default), the executor analyzes ExecutionHints to determine
    /// optimal scheduling per element.
    pub auto_strategy: bool,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            scheduling: SchedulingMode::Async,
            channel_capacity: 16,
            rt: RtConfig::default(),
            driver: None,
            auto_strategy: true, // Enable automatic by default
        }
    }
}

impl ExecutorConfig {
    /// Create config for automatic strategy detection (default).
    ///
    /// The executor will analyze each element's `ExecutionHints` to determine:
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
}

/// Analyzed execution plan for the entire pipeline.
#[derive(Debug)]
pub struct ExecutionPlan {
    /// Strategy per node.
    pub strategies: HashMap<NodeId, ElementStrategy>,
    /// Nodes that need RT scheduling.
    pub rt_nodes: HashSet<NodeId>,
    /// Nodes that can run async.
    pub async_nodes: HashSet<NodeId>,
    /// Whether any RT scheduling is needed.
    pub needs_rt: bool,
}

impl ExecutionPlan {
    /// Create an empty plan (all async).
    pub fn all_async() -> Self {
        Self {
            strategies: HashMap::new(),
            rt_nodes: HashSet::new(),
            async_nodes: HashSet::new(),
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
            let strategy = determine_element_strategy(&hints);

            match strategy {
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
        "Execution plan: {} async, {} RT",
        plan.async_nodes.len(),
        plan.rt_nodes.len(),
    );

    plan
}

/// Determine the execution strategy for a single element based on its hints.
///
/// Elements declare facts about their capabilities (rt_safe, processing type,
/// latency, etc.) and the executor derives the optimal strategy.
///
/// Priority order: RT (rt_safe + low latency) > I/O-bound > default async.
fn determine_element_strategy(hints: &ExecutionHints) -> ElementStrategy {
    // Rule 1: RT-safe + low latency → RT thread
    if hints.rt_safe && matches!(hints.latency, LatencyHint::UltraLow | LatencyHint::Low) {
        return ElementStrategy::RealTime;
    }

    // Rule 2: I/O-bound → async
    if hints.processing == ProcessingHint::IoBound {
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
    /// RT driver task (periodic trigger for RT data thread).
    /// Stored separately because it runs indefinitely and must be
    /// aborted when the pipeline finishes.
    rt_driver_task: Option<JoinHandle<Result<()>>>,
    /// Pipeline message bus (moved from Pipeline on start).
    bus: Option<Bus>,
    /// Bus handle for posting pipeline-level messages.
    bus_handle: Option<BusHandle>,
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

        // Abort the RT driver task (it runs indefinitely)
        if let Some(task) = self.rt_driver_task.take() {
            task.abort();
            let _ = task.await; // Ignore JoinError from abort
        }

        // Signal RT threads to stop and join (via spawn_blocking to avoid
        // blocking the async executor)
        for handle in self.rt_handles.drain(..) {
            handle.signal_stop();
            let join_result = tokio::task::spawn_blocking(move || handle.join()).await;
            match join_result {
                Ok(Err(e)) => {
                    if first_error.is_none() {
                        first_error = Some(e);
                    }
                }
                Err(e) => {
                    if first_error.is_none() {
                        first_error =
                            Some(Error::InvalidSegment(format!("RT join task panicked: {e}")));
                    }
                }
                Ok(Ok(())) => {}
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
        if let Some(task) = self.rt_driver_task.take() {
            task.abort();
        }
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

    /// Get a mutable reference to the pipeline bus for polling messages.
    ///
    /// Returns `None` if the bus was already taken via [`take_bus`](Self::take_bus).
    pub fn bus_mut(&mut self) -> Option<&mut Bus> {
        self.bus.as_mut()
    }

    /// Take the bus out of the handle (for moving to another task).
    pub fn take_bus(&mut self) -> Option<Bus> {
        self.bus.take()
    }

    /// Get the bus handle for posting pipeline-level messages.
    pub fn bus_handle(&self) -> Option<&BusHandle> {
        self.bus_handle.as_ref()
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
                "Auto-detected execution plan: {} async, {} RT",
                plan.async_nodes.len(),
                plan.rt_nodes.len(),
            );
            Some(plan)
        } else {
            None
        };

        // State transitions
        let old_state = pipeline.state();
        let bus_handle = pipeline.bus_handle().clone();
        if old_state == PipelineState::Suspended {
            pipeline.prepare()?;
            events.send_state_changed(old_state, PipelineState::Idle);
            bus_handle.post_state_changed(old_state, PipelineState::Idle);
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

        // Auto-select the best clock from pipeline elements, then start it
        pipeline.select_clock();
        pipeline.start_clock();
        let pipeline_clock = pipeline.clock();
        let clock_info: Option<(Arc<dyn Clock>, ClockTime)> = if pipeline_clock.is_started() {
            Some((pipeline_clock.clock(), pipeline_clock.base_time()))
        } else {
            None
        };

        // Execute based on scheduling mode
        let (tasks, rt_handles, bridges, rt_driver_task) = match effective_scheduling {
            SchedulingMode::Async => {
                let tasks = self.run_async(pipeline, clock_info.as_ref(), &events)?;
                (tasks, Vec::new(), Vec::new(), None)
            }
            SchedulingMode::Hybrid | SchedulingMode::RealTime => {
                if partition.rt_nodes.is_empty() {
                    // No RT nodes, fall back to async
                    let tasks = self.run_async(pipeline, clock_info.as_ref(), &events)?;
                    (tasks, Vec::new(), Vec::new(), None)
                } else {
                    self.run_hybrid(
                        pipeline,
                        &partition,
                        &mut scheduler,
                        clock_info.as_ref(),
                        &events,
                    )?
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
        bus_handle.post_state_changed(idle_state, PipelineState::Running);
        events.send(PipelineEvent::Started);
        pipeline.tracer_registry().notify_start();

        // Take the bus from the pipeline and store it on the handle.
        let bus = pipeline.take_bus();
        let bus_handle = Some(pipeline.bus_handle().clone());

        Ok(PipelineHandle {
            tasks,
            rt_handles,
            events,
            bridges,
            driver,
            rt_driver_task,
            bus,
            bus_handle,
        })
    }

    /// Run all nodes as async Tokio tasks.
    fn run_async(
        &self,
        pipeline: &mut Pipeline,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut channels = ChannelNetwork::new();

        // Build channels
        for src_id in pipeline.sources() {
            self.build_channels(pipeline, src_id, &mut channels);
        }

        // Spawn tasks
        self.spawn_tasks(pipeline, channels, clock_info, events)
    }

    /// Run with hybrid async + RT execution.
    fn run_hybrid(
        &self,
        pipeline: &mut Pipeline,
        partition: &GraphPartition,
        scheduler: &mut RtScheduler,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
    ) -> Result<(
        Vec<JoinHandle<Result<()>>>,
        Vec<crate::pipeline::rt_scheduler::DataThreadHandle>,
        Vec<Arc<AsyncRtBridge>>,
        Option<JoinHandle<Result<()>>>,
    )> {
        use crate::pipeline::rt_bridge::EventFd;
        use crate::pipeline::rt_scheduler::{BoundaryDirection, spawn_data_thread};

        // Create bridges at boundaries
        scheduler.create_bridges(partition)?;
        scheduler.setup_activations(partition)?;
        scheduler.compute_processing_order(partition, pipeline)?;
        scheduler.setup_dependencies(partition, pipeline)?;
        scheduler.select_driver(partition, pipeline);

        // Build downstream map for dependency signaling
        let downstream_map = scheduler.build_downstream_map(partition, pipeline);

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

        // Spawn async tasks for the async portion of the graph
        let tasks = self.spawn_tasks_for_partition(
            pipeline, partition, channels, scheduler, clock_info, events,
        )?;

        // Collect bridges (keep alive)
        let bridges: Vec<_> = partition
            .boundary_edges
            .iter()
            .filter_map(|e| scheduler.get_bridge(e.source, e.sink))
            .collect();

        // --- Spawn RT data thread ---

        // Extract RT elements from the pipeline graph
        let mut rt_elements: HashMap<NodeId, Box<DynAsyncElement<'static>>> = HashMap::new();
        for &node_id in &partition.rt_nodes {
            if let Some(node) = pipeline.get_node_mut(node_id) {
                if let Some(element) = node.take_element() {
                    rt_elements.insert(node_id, element);
                }
            }
        }

        // Build input/output bridge maps for the RT data thread
        let mut input_bridges: HashMap<NodeId, Arc<AsyncRtBridge>> = HashMap::new();
        let mut output_bridges: HashMap<NodeId, Arc<AsyncRtBridge>> = HashMap::new();

        for edge in &partition.boundary_edges {
            if let Some(bridge) = scheduler.get_bridge(edge.source, edge.sink) {
                match edge.direction {
                    BoundaryDirection::AsyncToRt => {
                        input_bridges.insert(edge.sink, bridge);
                    }
                    BoundaryDirection::RtToAsync => {
                        output_bridges.insert(edge.source, bridge);
                    }
                }
            }
        }

        // Create driver trigger (the data thread waits on this each cycle)
        let driver_trigger = Arc::new(EventFd::new()?);

        // Spawn the data thread
        let rt_handle = spawn_data_thread(
            "parallax-rt-0".to_string(),
            self.config.rt.clone(),
            scheduler.processing_order().to_vec(),
            scheduler.activations().clone(),
            rt_elements,
            input_bridges,
            output_bridges,
            downstream_map,
            driver_trigger.clone(),
        )?;

        // Spawn a driver task that triggers the RT thread periodically
        let driver_period = self
            .config
            .driver
            .as_ref()
            .map(|d| d.period)
            .unwrap_or(std::time::Duration::from_millis(5));

        let driver_task = tokio::spawn(async move {
            let mut interval = tokio::time::interval(driver_period);
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            loop {
                interval.tick().await;
                if let Err(e) = driver_trigger.notify() {
                    tracing::error!("driver trigger notify error: {}", e);
                    return Err(Error::Io(std::io::Error::other(format!(
                        "driver trigger: {e}"
                    ))));
                }
            }
            #[allow(unreachable_code)]
            Ok(())
        });
        // NOTE: driver_task is NOT pushed to `tasks` because it runs indefinitely.
        // It is stored in PipelineHandle.rt_driver_task and aborted on shutdown.

        let rt_handles = vec![rt_handle];

        Ok((tasks, rt_handles, bridges, Some(driver_task)))
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
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
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
            let task =
                self.spawn_node_task(pipeline, node_id, &mut channels, clock_info, events)?;
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
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
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
                clock_info,
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
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
    ) -> Result<JoinHandle<Result<()>>> {
        self.spawn_node_task_with_bridges(
            pipeline,
            node_id,
            channels,
            Vec::new(),
            Vec::new(),
            clock_info,
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
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
    ) -> Result<JoinHandle<Result<()>>> {
        let node = pipeline
            .get_node_mut(node_id)
            .ok_or_else(|| Error::InvalidSegment("node not found".into()))?;

        let element_type = node.element_type();
        let node_name = node.name().to_string();

        let mut element = node.take_element().ok_or_else(|| {
            Error::InvalidSegment(format!("element '{}' already taken", node_name))
        })?;

        // Set clock on source elements so they can provide it to ProduceContext
        if element_type == ElementType::Source {
            if let Some((clock, base_time)) = clock_info {
                element.set_clock(clock.clone(), *base_time);
            }
        }

        // Set bus handle so elements can post messages
        element.set_bus(pipeline.bus_handle().for_element(&node_name));

        let inputs = channels.take_inputs(node_id);
        let outputs = channels.take_outputs(node_id);
        let events_clone = events.clone();
        let probes = pipeline.probe_registry().clone();
        let tracers = pipeline.tracer_registry().clone();

        let task = match element_type {
            ElementType::Source => {
                spawn_source_task(node_name, node_id, element, outputs, output_bridges, events_clone, probes, tracers)
            }
            ElementType::Sink => {
                spawn_sink_task(node_name, element, inputs, input_bridges, events_clone, tracers)
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
#[allow(clippy::large_enum_variant)] // Intentional: avoid heap allocation on hot path
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
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    outputs: Vec<AsyncSender<Message>>,
    output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("source '{}' started", name);
        events.send_node_started(&name);

        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;
        let mut would_block_count: u64 = 0;

        loop {
            tracing::trace!("source '{}': calling process_source", name);
            match element.process_source().await {
                Ok(SourceResult::Buffer(buffer)) => {
                    let buffer = *buffer;
                    count += 1;
                    would_block_count = 0; // Reset

                    // Invoke buffer probes
                    match probe_registry.invoke_buffer(&src_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }

                    // Notify tracers
                    tracers.notify_buffer(&name, &buffer);

                    tracing::debug!(
                        "source '{}': produced buffer {} ({} bytes)",
                        name,
                        count,
                        buffer.len()
                    );
                    for tx in &outputs {
                        let _ = tx.send(Message::Buffer(buffer.clone())).await;
                    }
                    for bridge in &output_bridges {
                        let _ = bridge.push_async(buffer.clone()).await;
                    }
                }
                Ok(SourceResult::WouldBlock) => {
                    would_block_count += 1;
                    if would_block_count == 1 || would_block_count % 1000 == 0 {
                        tracing::debug!(
                            "source '{}': WouldBlock (count: {})",
                            name,
                            would_block_count
                        );
                    }
                    // No data available yet, sleep briefly and retry
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
                Ok(SourceResult::Eos) => {
                    tracing::info!("source '{}': EOS after {} buffers", name, count);
                    for tx in &outputs {
                        let _ = tx.send(Message::Eos).await;
                    }
                    for bridge in &output_bridges {
                        bridge.signal_eos();
                    }
                    break;
                }
                Err(e) => {
                    tracing::error!("source '{}': error: {}", name, e);
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
    input_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    tracers: TracerRegistry,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("sink '{}' started", name);
        events.send_node_started(&name);

        let mut count: u64 = 0;

        if let Some(rx) = inputs.into_iter().next() {
            // Standard path: read from kanal channel
            while let Ok(Message::Buffer(buffer)) = rx.recv().await {
                count += 1;
                tracers.notify_buffer(&name, &buffer);
                if let Err(e) = element.process(Some(buffer)).await {
                    events.send_error(e.to_string(), Some(name.clone()));
                    return Err(e);
                }
                tracers.notify_buffer_processed(&name);
            }
        } else if let Some(bridge) = input_bridges.into_iter().next() {
            // Bridge path: read from RT→Async bridge
            loop {
                // Drain all available buffers
                while let Some(buffer) = bridge.try_pop() {
                    count += 1;
                    if let Err(e) = element.process(Some(buffer)).await {
                        events.send_error(e.to_string(), Some(name.clone()));
                        return Err(e);
                    }
                }
                // Check if we're done (EOS + empty)
                if bridge.is_done() {
                    tracing::info!("sink '{}': bridge EOS after {} buffers", name, count);
                    break;
                }
                // Wait for more data or EOS signal
                match bridge.data_eventfd().wait_async().await {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::error!("sink '{}': bridge eventfd error: {}", name, e);
                        break;
                    }
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
    input_bridges: Vec<Arc<AsyncRtBridge>>,
    output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("transform '{}' started", name);
        events.send_node_started(&name);

        let mut count: u64 = 0;

        /// Helper to send output buffer to all downstream channels and bridges.
        async fn send_output(
            buffer: Buffer,
            outputs: &[AsyncSender<Message>],
            output_bridges: &[Arc<AsyncRtBridge>],
        ) {
            for tx in outputs {
                let _ = tx.send(Message::Buffer(buffer.clone())).await;
            }
            for bridge in output_bridges {
                let _ = bridge.push_async(buffer.clone()).await;
            }
        }

        /// Helper to send EOS to all downstream channels and bridges.
        async fn send_eos(outputs: &[AsyncSender<Message>], output_bridges: &[Arc<AsyncRtBridge>]) {
            for tx in outputs {
                let _ = tx.send(Message::Eos).await;
            }
            for bridge in output_bridges {
                bridge.signal_eos();
            }
        }

        if let Some(rx) = inputs.into_iter().next() {
            // Standard path: read from kanal channel
            loop {
                tracing::trace!("transform '{}': waiting for input", name);
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        count += 1;
                        tracing::debug!(
                            "transform '{}': received buffer {} ({} bytes)",
                            name,
                            count,
                            buffer.len()
                        );
                        match element.process(Some(buffer)).await {
                            Ok(Some(out)) => {
                                tracing::debug!(
                                    "transform '{}': produced output ({} bytes)",
                                    name,
                                    out.len()
                                );
                                send_output(out, &outputs, &output_bridges).await;
                            }
                            Ok(None) => {
                                tracing::debug!(
                                    "transform '{}': no output for buffer {}",
                                    name,
                                    count
                                );
                            }
                            Err(e) => {
                                tracing::error!("transform '{}': error: {}", name, e);
                                events.send_error(e.to_string(), Some(name.clone()));
                                return Err(e);
                            }
                        }
                    }
                    Ok(Message::Eos) => {
                        tracing::info!(
                            "transform '{}': received EOS after {} buffers, flushing",
                            name,
                            count
                        );
                        // Flush any buffered data before propagating EOS
                        match element.flush().await {
                            Ok(output) => {
                                let buffers = match output {
                                    Output::None => vec![],
                                    Output::Single(b) => vec![b],
                                    Output::Multiple(v) => v,
                                };
                                tracing::info!(
                                    "transform '{}': flush produced {} buffers",
                                    name,
                                    buffers.len()
                                );
                                for buffer in buffers {
                                    send_output(buffer, &outputs, &output_bridges).await;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("flush error in '{}': {}", name, e);
                            }
                        }
                        send_eos(&outputs, &output_bridges).await;
                        break;
                    }
                    Err(_) => {
                        send_eos(&outputs, &output_bridges).await;
                        break;
                    }
                }
            }
        } else if let Some(bridge) = input_bridges.into_iter().next() {
            // Bridge path: read from RT→Async bridge
            loop {
                while let Some(buffer) = bridge.try_pop() {
                    count += 1;
                    match element.process(Some(buffer)).await {
                        Ok(Some(out)) => {
                            send_output(out, &outputs, &output_bridges).await;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            events.send_error(e.to_string(), Some(name.clone()));
                            return Err(e);
                        }
                    }
                }
                if bridge.is_done() {
                    // Flush
                    match element.flush().await {
                        Ok(output) => {
                            let buffers = match output {
                                Output::None => vec![],
                                Output::Single(b) => vec![b],
                                Output::Multiple(v) => v,
                            };
                            for buffer in buffers {
                                send_output(buffer, &outputs, &output_bridges).await;
                            }
                        }
                        Err(e) => {
                            tracing::warn!("flush error in '{}': {}", name, e);
                        }
                    }
                    send_eos(&outputs, &output_bridges).await;
                    break;
                }
                match bridge.data_eventfd().wait_async().await {
                    Ok(()) => {}
                    Err(e) => {
                        tracing::error!("transform '{}': bridge eventfd error: {}", name, e);
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
                        // Flush any buffered data before propagating EOS
                        match element.flush().await {
                            Ok(output) => {
                                let buffers = match output {
                                    Output::None => vec![],
                                    Output::Single(b) => vec![b],
                                    Output::Multiple(v) => v,
                                };
                                for buffer in buffers {
                                    for senders in outputs_by_pad.values() {
                                        for tx in senders {
                                            let _ = tx.send(Message::Buffer(buffer.clone())).await;
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!("flush error in '{}': {}", name, e);
                            }
                        }
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
                        // Flush any remaining data from final processing
                        if let Ok(Some(out)) = element.process(None).await {
                            for tx in &outputs {
                                let _ = tx.send(Message::Buffer(out.clone())).await;
                            }
                        }
                        // Flush any buffered data before propagating EOS
                        match element.flush().await {
                            Ok(output) => {
                                let buffers = match output {
                                    Output::None => vec![],
                                    Output::Single(b) => vec![b],
                                    Output::Multiple(v) => v,
                                };
                                for buffer in buffers {
                                    for tx in &outputs {
                                        let _ = tx.send(Message::Buffer(buffer.clone())).await;
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!("flush error in '{}': {}", name, e);
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
    use crate::memory::SharedArena;
    use crate::metadata::Metadata;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn test_arena() -> &'static SharedArena {
        static ARENA: OnceLock<SharedArena> = OnceLock::new();
        ARENA.get_or_init(|| SharedArena::new(64, 64).unwrap())
    }

    struct CountingSource {
        count: u64,
        max: u64,
    }

    impl Source for CountingSource {
        fn produce(&mut self, _ctx: &mut ProduceContext) -> Result<ProduceResult> {
            if self.count >= self.max {
                return Ok(ProduceResult::Eos);
            }
            let arena = test_arena();
            let slot = arena.acquire().unwrap();
            let handle = MemoryHandle::new(slot);
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

        // Default hints -> Async
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_determine_element_strategy_rt_safe() {
        // RT-safe profile (rt_safe + low latency) -> RealTime
        let hints = ExecutionHints::rt_safe();
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::RealTime);

        // RT-safe but normal latency -> Async (RT needs low latency)
        let hints = ExecutionHints::default().with_rt_safe(true);
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_determine_element_strategy_low_latency() {
        // Low latency + RT-safe -> RealTime
        let hints = ExecutionHints::low_latency().with_rt_safe(true);
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::RealTime);

        // Low latency but NOT RT-safe -> Async
        let hints = ExecutionHints::low_latency().with_rt_safe(false);
        let strategy = determine_element_strategy(&hints);
        assert_eq!(strategy, ElementStrategy::Async);
    }

    #[test]
    fn test_determine_element_strategy_io_bound() {
        let hints = ExecutionHints::io_bound();

        // I/O-bound -> always Async
        let strategy = determine_element_strategy(&hints);
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
        assert!(!plan.needs_rt);
        assert_eq!(plan.async_nodes.len(), 2);
        assert_eq!(plan.rt_nodes.len(), 0);
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
