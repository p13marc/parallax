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
use crate::memory::{OutputBudget, defaults};
use crate::pipeline::bus::{Bus, BusHandle};
use crate::pipeline::probe::{ProbeRegistry, ProbeReturn};
use crate::pipeline::rt_bridge::AsyncRtBridge;
use crate::pipeline::rt_scheduler::{
    BoundaryDirection, GraphPartition, RtConfig, RtScheduler, SchedulingMode,
};
use crate::pipeline::tracer::TracerRegistry;
use crate::pipeline::{
    DriverConfig, EventReceiver, EventSender, LinkPolicy, NodeId, Pipeline, PipelineEvent,
    PipelineState, TimerDriver,
};
use kanal::{AsyncReceiver, AsyncSender, bounded_async};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
    /// Cooperative stop flag, checked by source tasks between `produce()`
    /// calls (see [`PipelineHandle::stop`]).
    stop: Arc<AtomicBool>,
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

    /// Signal all sources to stop cooperatively.
    ///
    /// Sources end their produce loop at the next iteration and propagate
    /// EOS downstream exactly like a natural end-of-stream, so sinks drain
    /// and [`wait`](Self::wait) returns `Ok(())`. This is the graceful way
    /// to stop a pipeline with live (infinite) sources — [`abort`](Self::abort)
    /// alone cannot end a source loop that never reaches an await point.
    ///
    /// Limitation: a source blocked *inside* a synchronous `produce()` call
    /// (e.g. waiting on a hardware frame with no timeout) only observes the
    /// flag once that call returns.
    pub fn stop(&self) {
        self.stop.store(true, Ordering::Release);
    }

    /// Abort all pipeline tasks.
    pub fn abort(mut self) {
        // Best effort for live sources: tasks blocked in a synchronous
        // produce() are never re-polled by abort(), so also raise the
        // cooperative stop flag — they exit at the next loop iteration.
        self.stop.store(true, Ordering::Release);
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

        // Cooperative stop flag shared with every source task (see
        // PipelineHandle::stop).
        let stop = Arc::new(AtomicBool::new(false));

        // Report what the element hints suggest. This is advisory only — see
        // the note on `effective_scheduling` below for why the suggestion is
        // not acted on automatically.
        if self.config.auto_strategy {
            let plan = analyze_pipeline(pipeline);
            if plan.needs_rt && self.config.scheduling == SchedulingMode::Async {
                tracing::info!(
                    "{} of {} elements report RT-safe, low-latency hints. The pipeline is \
                     running async; pass ExecutorConfig::with_scheduling(SchedulingMode::Hybrid) \
                     to put them on RT threads.",
                    plan.rt_nodes.len(),
                    plan.rt_nodes.len() + plan.async_nodes.len(),
                );
            }
        }

        // State transitions
        let old_state = pipeline.state();
        let bus_handle = pipeline.bus_handle().clone();
        if old_state == PipelineState::Suspended {
            pipeline.prepare()?;
            events.send_state_changed(old_state, PipelineState::Idle);
            bus_handle.post_state_changed(old_state, PipelineState::Idle);
        }

        // The configured mode is the effective mode.
        //
        // `auto_strategy` used to recompute this from the plan, promoting a
        // pipeline to `Hybrid` whenever any element looked RT-worthy — and
        // then discarding the result, because the partitioner reads
        // `RtConfig::mode`, which `auto_strategy` never touched. So the
        // promotion has never taken effect, and honouring it now would move
        // every pipeline containing an rt_safe low-latency element onto RT
        // threads by default: a large behavioural change, onto the code path
        // that still has no probes or tracers (#43). That is its own decision,
        // not a side effect of fixing #6, so auto-promotion stays off and the
        // plan is used for reporting only.
        //
        // What #6 needed is below: an explicitly configured mode is now
        // actually applied, instead of being silently overridden here and then
        // ignored by the partitioner.
        let effective_scheduling = self.config.scheduling;

        // Partition graph for hybrid scheduling.
        //
        // `ExecutorConfig::scheduling` is authoritative: `RtConfig::mode` is
        // derived from it here rather than trusted. Without this,
        // `ExecutorConfig::default().with_scheduling(RealTime)` left `rt.mode`
        // at its `Async` default, `should_run_rt` rejected every node, and the
        // pipeline *always* fell back to async no matter how many RT-safe
        // elements it contained — the root cause behind #6, one level below
        // the empty-partition fallback the issue described.
        let mut rt_config = self.config.rt.clone();
        rt_config.mode = effective_scheduling;
        let mut scheduler = RtScheduler::new(rt_config);
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
                let tasks = self.run_async(pipeline, clock_info.as_ref(), &events, &stop)?;
                (tasks, Vec::new(), Vec::new(), None)
            }
            SchedulingMode::Hybrid | SchedulingMode::RealTime => {
                if partition.rt_nodes.is_empty() {
                    // No node qualified, so there is nothing to schedule on an
                    // RT thread and the whole graph runs async. That is a
                    // legitimate outcome — but it silently discards the latency
                    // guarantee the caller asked for, so say so (#6).
                    //
                    // `RealTime` gets the louder message: `Hybrid` is a
                    // best-effort request by construction, whereas `RealTime`
                    // is chosen precisely when RT execution is the point.
                    let names = partition
                        .async_nodes
                        .iter()
                        .filter_map(|&id| pipeline.get_node(id).map(|n| n.name().to_string()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    if effective_scheduling == SchedulingMode::RealTime {
                        tracing::warn!(
                            "SchedulingMode::RealTime requested, but no element reported \
                             ExecutionHints::rt_safe — running fully async. \
                             Nodes: [{names}]. Implement SyncElement and return \
                             ExecutionHints::rt_safe() from execution_hints() to opt in."
                        );
                    } else {
                        tracing::warn!(
                            "SchedulingMode::Hybrid requested, but no element is both rt_safe \
                             and low-latency — running fully async. Nodes: [{names}]."
                        );
                    }
                    let tasks = self.run_async(pipeline, clock_info.as_ref(), &events, &stop)?;
                    (tasks, Vec::new(), Vec::new(), None)
                } else {
                    self.run_hybrid(
                        pipeline,
                        &partition,
                        &mut scheduler,
                        clock_info.as_ref(),
                        &events,
                        &stop,
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
            stop,
        })
    }

    /// Run all nodes as async Tokio tasks.
    fn run_async(
        &self,
        pipeline: &mut Pipeline,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
    ) -> Result<Vec<JoinHandle<Result<()>>>> {
        let mut channels = ChannelNetwork::new();

        // Build channels
        for src_id in pipeline.sources() {
            self.build_channels(pipeline, src_id, &mut channels);
        }

        // Spawn tasks
        self.spawn_tasks(pipeline, channels, clock_info, events, stop)
    }

    /// Run with hybrid async + RT execution.
    fn run_hybrid(
        &self,
        pipeline: &mut Pipeline,
        partition: &GraphPartition,
        scheduler: &mut RtScheduler,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
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
            pipeline, partition, channels, scheduler, clock_info, events, stop,
        )?;

        // Collect bridges (keep alive)
        let bridges: Vec<_> = partition
            .boundary_edges
            .iter()
            .filter_map(|e| scheduler.get_bridge(e.source, e.sink))
            .collect();

        // --- Spawn RT data thread ---

        // Extract RT elements from the pipeline graph.
        //
        // These bypass `spawn_node_task_with_bridges`, so the setters it
        // applies have to be repeated here — until this was noticed, an element
        // scheduled onto an RT thread silently received no clock, no bus and no
        // arena budget, and behaved differently from the same element in an
        // async graph.
        //
        // Budgets are computed up front because `children()` borrows the
        // pipeline shared while `get_node_mut` needs it mutable.
        let rt_budgets: HashMap<NodeId, OutputBudget> = partition
            .rt_nodes
            .iter()
            .map(|&node_id| {
                let bridged = partition.boundary_edges.iter().any(|e| {
                    e.source == node_id && matches!(e.direction, BoundaryDirection::RtToAsync)
                });
                (
                    node_id,
                    self.output_slot_budget(pipeline, node_id, usize::from(bridged)),
                )
            })
            .collect();

        let mut rt_elements: HashMap<NodeId, Box<DynAsyncElement<'static>>> = HashMap::new();
        for &node_id in &partition.rt_nodes {
            let node_name = pipeline
                .get_node(node_id)
                .map(|n| n.name().to_string())
                .unwrap_or_default();
            let bus = pipeline.bus_handle().for_element(&node_name);

            if let Some(node) = pipeline.get_node_mut(node_id)
                && let Some(mut element) = node.take_element()
            {
                if node.element_type() == ElementType::Source
                    && let Some((clock, base_time)) = clock_info
                {
                    element.set_clock(clock.clone(), *base_time);
                }
                element.set_bus(bus);
                if let Some(budget) = rt_budgets.get(&node_id) {
                    element.set_output_budget(*budget);
                }
                rt_elements.insert(node_id, element);
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

    /// How many buffers the graph downstream of `node_id` can hold at once.
    ///
    /// Elements that allocate their own output buffers need an arena at least
    /// this big, or they run out of slots the moment a consumer falls behind —
    /// which used to kill the pipeline. Only the executor knows the number:
    /// link capacity is its configuration.
    ///
    /// **Per pad it is the maximum, not the sum.** Fan-out clones a `Buffer`,
    /// and a clone is a refcount bump on the same slot, so three branches each
    /// holding the same buffer pin one slot between them. The producer stalls on
    /// whichever `Block` branch fills first, so the deepest link bounds the pad.
    /// Summing would over-allocate by up to N×.
    ///
    /// **Across pads it is the sum**, because separate src pads (a demuxer's)
    /// carry genuinely different buffers.
    ///
    /// A bridged edge contributes `RtConfig::bridge_capacity` the same way — the
    /// bridge is a queue like any other.
    fn output_slot_budget(
        &self,
        pipeline: &Pipeline,
        node_id: NodeId,
        output_bridges: usize,
    ) -> OutputBudget {
        let mut per_pad: HashMap<String, usize> = HashMap::new();

        for (_child_id, link) in pipeline.children(node_id) {
            let capacity = link.capacity.unwrap_or(self.config.channel_capacity);
            let deepest = per_pad.entry(link.src_pad.clone()).or_insert(0);
            *deepest = (*deepest).max(capacity);
        }

        let mut downstream_capacity: usize = per_pad.values().sum();
        if output_bridges > 0 {
            downstream_capacity =
                downstream_capacity.saturating_add(self.config.rt.bridge_capacity);
        }

        OutputBudget::new(downstream_capacity, defaults::IN_FLIGHT_MARGIN)
    }

    /// Build channel network recursively.
    fn build_channels(&self, pipeline: &Pipeline, node_id: NodeId, network: &mut ChannelNetwork) {
        for (child_id, link) in pipeline.children(node_id) {
            if !network.has_channel(node_id, &link.src_pad, child_id, &link.sink_pad) {
                let capacity = link.capacity.unwrap_or(self.config.channel_capacity);
                let (tx, rx) = bounded_async::<Message>(capacity);
                network.add_channel(
                    node_id,
                    link.src_pad.clone(),
                    child_id,
                    link.sink_pad.clone(),
                    tx,
                    rx,
                    link.policy,
                    node_name(pipeline, child_id),
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
                    let capacity = link.capacity.unwrap_or(self.config.channel_capacity);
                    let (tx, rx) = bounded_async::<Message>(capacity);
                    network.add_channel(
                        node_id,
                        link.src_pad.clone(),
                        child_id,
                        link.sink_pad.clone(),
                        tx,
                        rx,
                        link.policy,
                        node_name(pipeline, child_id),
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
        stop: &Arc<AtomicBool>,
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
                self.spawn_node_task(pipeline, node_id, &mut channels, clock_info, events, stop)?;
            tasks.push(task);
        }

        Ok(tasks)
    }

    /// Spawn tasks for async partition only.
    #[allow(clippy::too_many_arguments)]
    fn spawn_tasks_for_partition(
        &self,
        pipeline: &mut Pipeline,
        partition: &GraphPartition,
        mut channels: ChannelNetwork,
        scheduler: &RtScheduler,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
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
                stop,
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
        stop: &Arc<AtomicBool>,
    ) -> Result<JoinHandle<Result<()>>> {
        self.spawn_node_task_with_bridges(
            pipeline,
            node_id,
            channels,
            Vec::new(),
            Vec::new(),
            clock_info,
            events,
            stop,
        )
    }

    /// Spawn a task with optional bridges.
    #[allow(clippy::too_many_arguments)]
    fn spawn_node_task_with_bridges(
        &self,
        pipeline: &mut Pipeline,
        node_id: NodeId,
        channels: &mut ChannelNetwork,
        output_bridges: Vec<Arc<AsyncRtBridge>>,
        input_bridges: Vec<Arc<AsyncRtBridge>>,
        clock_info: Option<&(Arc<dyn Clock>, ClockTime)>,
        events: &EventSender,
        stop: &Arc<AtomicBool>,
    ) -> Result<JoinHandle<Result<()>>> {
        // Before `get_node_mut` borrows the pipeline mutably: `children()` needs
        // it shared.
        let budget = self.output_slot_budget(pipeline, node_id, output_bridges.len());

        let node = pipeline
            .get_node_mut(node_id)
            .ok_or_else(|| Error::InvalidSegment("node not found".into()))?;

        let element_type = node.element_type();
        let node_name = node.name().to_string();

        let mut element = node.take_element().ok_or_else(|| {
            Error::InvalidSegment(format!("element '{}' already taken", node_name))
        })?;

        // Set clock on source elements so they can provide it to ProduceContext
        if element_type == ElementType::Source
            && let Some((clock, base_time)) = clock_info
        {
            element.set_clock(clock.clone(), *base_time);
        }

        // Set bus handle so elements can post messages
        element.set_bus(pipeline.bus_handle().for_element(&node_name));

        // Tell the element how much the graph below it can hold, so it can size
        // its output arena before the first frame builds it.
        element.set_output_budget(budget);

        // Take the channels this element type actually needs, inside the match.
        //
        // These used to be hoisted out of the match, which quietly broke both
        // N-ary element types: `take_inputs`/`take_outputs` *drain* the maps
        // (they are `take_*_by_pad().into_values().flatten()`), so by the time
        // the Muxer arm called `take_inputs_by_pad` it got an empty map and the
        // muxer received no inputs at all — likewise the Demuxer arm's
        // `take_outputs_by_pad`. Muxers and demuxers added through
        // `add_muxer`/`add_demuxer` therefore never moved a single buffer.
        let events_clone = events.clone();
        let probes = pipeline.probe_registry().clone();
        let tracers = pipeline.tracer_registry().clone();

        let task = match element_type {
            ElementType::Source => spawn_source_task(
                node_name,
                node_id,
                element,
                channels.take_outputs(node_id),
                output_bridges,
                events_clone,
                probes,
                tracers,
                stop.clone(),
            ),
            ElementType::Sink => spawn_sink_task(
                node_name,
                node_id,
                element,
                channels.take_inputs(node_id),
                input_bridges,
                events_clone,
                probes,
                tracers,
            ),
            ElementType::Transform => spawn_transform_task(
                node_name,
                node_id,
                element,
                channels.take_inputs(node_id),
                channels.take_outputs(node_id),
                input_bridges,
                output_bridges,
                events_clone,
                probes,
                tracers,
            ),
            ElementType::Demuxer => {
                // Inputs flattened (one sink pad), outputs kept per pad — that
                // is the whole point of a demuxer.
                let inputs = channels.take_inputs(node_id);
                let outputs_by_pad = channels.take_outputs_by_pad(node_id);
                spawn_demuxer_task(
                    node_name,
                    node_id,
                    element,
                    inputs,
                    outputs_by_pad,
                    events_clone,
                    probes,
                    tracers,
                )
            }
            ElementType::Muxer => {
                // Mirror image: inputs per pad, outputs flattened.
                let inputs_by_pad = channels.take_inputs_by_pad(node_id);
                let outputs = channels.take_outputs(node_id);
                spawn_muxer_task(
                    node_name,
                    node_id,
                    element,
                    inputs_by_pad,
                    outputs,
                    events_clone,
                    probes,
                    tracers,
                )
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

/// One downstream branch of a src-pad, with the policy of the link that made it.
///
/// The old code flattened a pad's outputs to a bare `Vec<AsyncSender>`, which
/// threw away *which branch is which* — and with it any chance of treating a
/// slow branch differently from a fast one.
#[derive(Clone)]
struct OutputBranch {
    tx: AsyncSender<Message>,
    policy: LinkPolicy,
    /// Name of the element on the far end, for drop reporting.
    sink_name: String,
}

impl OutputBranch {
    /// Send a buffer, honouring the link policy.
    ///
    /// kanal's `try_send` returns `Ok(false)` when the channel is **full** (not
    /// an error) and `Err` only when it is closed — easy to get backwards.
    async fn send_buffer(&self, buffer: Buffer, tracers: &TracerRegistry) {
        match self.policy {
            LinkPolicy::Block => {
                let _ = self.tx.send(Message::Buffer(buffer)).await;
            }
            LinkPolicy::Drop => match self.tx.try_send(Message::Buffer(buffer)) {
                Ok(true) => {}
                Ok(false) => tracers.notify_drop(&self.sink_name),
                Err(_) => {} // closed
            },
        }
    }
}

/// Broadcast one buffer to every branch of a pad.
///
/// Blocking branches are awaited **concurrently**, not one after another: a
/// sequential loop makes each branch wait for the ones before it, which shows up
/// as latency skew between equal-speed branches. The single-output case — the
/// overwhelmingly common one — takes a direct await and allocates nothing.
async fn broadcast(branches: &[OutputBranch], buffer: Buffer, tracers: &TracerRegistry) {
    match branches {
        [] => {}
        [only] => only.send_buffer(buffer, tracers).await,
        many => {
            futures::future::join_all(
                many.iter()
                    .map(|branch| branch.send_buffer(buffer.clone(), tracers)),
            )
            .await;
        }
    }
}

/// Send EOS to every branch, **always blocking**.
///
/// A dropped EOS would leave the branch's sink waiting forever, so `Drop` does
/// not apply to it. Only buffers are ever dropped.
async fn broadcast_eos(branches: &[OutputBranch]) {
    for branch in branches {
        let _ = branch.tx.send(Message::Eos).await;
    }
}

/// Name of a node, for drop reporting on the link that feeds it.
fn node_name(pipeline: &Pipeline, node_id: NodeId) -> String {
    pipeline
        .get_node(node_id)
        .map(|n| n.name().to_string())
        .unwrap_or_default()
}

struct ChannelNetwork {
    channels: HashMap<ChannelKey, (AsyncSender<Message>, AsyncReceiver<Message>)>,
    outputs: HashMap<(NodeId, String), Vec<OutputBranch>>,
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

    #[allow(clippy::too_many_arguments)]
    fn add_channel(
        &mut self,
        src: NodeId,
        src_pad: String,
        sink: NodeId,
        sink_pad: String,
        tx: AsyncSender<Message>,
        rx: AsyncReceiver<Message>,
        policy: LinkPolicy,
        sink_name: String,
    ) {
        self.channels.insert(
            (src, src_pad.clone(), sink, sink_pad.clone()),
            (tx.clone(), rx.clone()),
        );
        self.outputs
            .entry((src, src_pad))
            .or_default()
            .push(OutputBranch {
                tx,
                policy,
                sink_name,
            });
        self.inputs.entry((sink, sink_pad)).or_default().push(rx);
    }

    fn take_outputs_by_pad(&mut self, node: NodeId) -> HashMap<String, Vec<OutputBranch>> {
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

    fn take_outputs(&mut self, node: NodeId) -> Vec<OutputBranch> {
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

#[allow(clippy::too_many_arguments)]
fn spawn_source_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    outputs: Vec<OutputBranch>,
    output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
    stop: Arc<AtomicBool>,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("source '{}' started", name);
        events.send_node_started(&name);

        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;
        let mut would_block_count: u64 = 0;

        loop {
            // Cooperative stop (PipelineHandle::stop/abort): end the loop
            // like a natural EOS so live sources release their device and
            // downstream drains cleanly.
            if stop.load(Ordering::Acquire) {
                tracing::info!("source '{}': stopped after {} buffers", name, count);
                broadcast_eos(&outputs).await;
                for bridge in &output_bridges {
                    bridge.signal_eos();
                }
                break;
            }
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
                    broadcast(&outputs, buffer.clone(), &tracers).await;
                    for bridge in &output_bridges {
                        let _ = bridge.push_async(buffer.clone()).await;
                    }
                }
                Ok(SourceResult::WouldBlock) => {
                    would_block_count += 1;
                    if would_block_count == 1 || would_block_count.is_multiple_of(1000) {
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
                    broadcast_eos(&outputs).await;
                    for bridge in &output_bridges {
                        bridge.signal_eos();
                    }
                    break;
                }
                Err(e) => {
                    tracing::error!("source '{}': error: {}", name, e);
                    events.send_error(e.to_string(), Some(name.clone()));
                    // A failed source will never produce again — propagate EOS
                    // downstream (like the Eos arm) so sinks terminate instead
                    // of waiting forever on a wedged pipeline.
                    broadcast_eos(&outputs).await;
                    for bridge in &output_bridges {
                        bridge.signal_eos();
                    }
                    return Err(e);
                }
            }
        }

        events.send_node_finished(&name, count);
        Ok(())
    })
}

#[allow(clippy::too_many_arguments)]
fn spawn_sink_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    input_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("sink '{}' started", name);
        events.send_node_started(&name);

        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let mut count: u64 = 0;

        let n_inputs = inputs.len();
        tracing::debug!("sink '{}': {} inputs", name, n_inputs);
        if let Some(rx) = inputs.into_iter().next() {
            // Standard path: read from kanal channel
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        count += 1;
                        tracing::debug!("sink '{}': received buffer {}", name, count);
                        match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }
                        tracers.notify_buffer(&name, &buffer);
                        if let Err(e) = element.process(Some(buffer)).await {
                            events.send_error(e.to_string(), Some(name.clone()));
                            return Err(e);
                        }
                        tracers.notify_buffer_processed(&name);
                    }
                    Ok(Message::Eos) => {
                        tracing::debug!("sink '{}': EOS after {}", name, count);
                        // Deliver EOS to the element so sinks with external
                        // consumers (AppSink) can unblock them.
                        let _ = element.handle_downstream_event(crate::event::Event::Eos);
                        break;
                    }
                    Err(e) => {
                        tracing::debug!("sink '{}': channel closed after {}: {}", name, count, e);
                        let _ = element.handle_downstream_event(crate::event::Event::Eos);
                        break;
                    }
                }
            }
        } else if let Some(bridge) = input_bridges.into_iter().next() {
            // Bridge path: read from RT→Async bridge. Observability mirrors the
            // channel path above; it used to be missing here entirely.
            loop {
                // Drain all available buffers
                while let Some(buffer) = bridge.try_pop() {
                    count += 1;
                    match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }
                    tracers.notify_buffer(&name, &buffer);
                    let result = element.process(Some(buffer)).await;
                    tracers.notify_buffer_processed(&name);
                    if let Err(e) = result {
                        events.send_error(e.to_string(), Some(name.clone()));
                        return Err(e);
                    }
                }
                // Check if we're done (EOS + empty)
                if bridge.is_done() {
                    tracing::info!("sink '{}': bridge EOS after {} buffers", name, count);
                    let _ = element.handle_downstream_event(crate::event::Event::Eos);
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

#[allow(clippy::too_many_arguments)]
fn spawn_transform_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    outputs: Vec<OutputBranch>,
    input_bridges: Vec<Arc<AsyncRtBridge>>,
    output_bridges: Vec<Arc<AsyncRtBridge>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("transform '{}' started", name);
        events.send_node_started(&name);

        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;

        /// Helper to send output buffer to all downstream channels and bridges.
        async fn send_output(
            buffer: Buffer,
            outputs: &[OutputBranch],
            output_bridges: &[Arc<AsyncRtBridge>],
            tracers: &TracerRegistry,
        ) {
            broadcast(outputs, buffer.clone(), tracers).await;
            for bridge in output_bridges {
                let _ = bridge.push_async(buffer.clone()).await;
            }
        }

        /// Helper to send EOS to all downstream channels and bridges.
        async fn send_eos(outputs: &[OutputBranch], output_bridges: &[Arc<AsyncRtBridge>]) {
            broadcast_eos(outputs).await;
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

                        // Sink-pad probes see the input before the element does.
                        match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }

                        // LatencyTracer pairs notify_buffer with
                        // notify_buffer_processed, so the second call must land
                        // BEFORE send_output — otherwise a downstream
                        // send().await sits between them and back-pressure gets
                        // billed as this element's processing time.
                        tracers.notify_buffer(&name, &buffer);
                        let result = element.process(Some(buffer)).await;
                        tracers.notify_buffer_processed(&name);

                        match result {
                            Ok(Some(out)) => {
                                tracing::debug!(
                                    "transform '{}': produced output ({} bytes)",
                                    name,
                                    out.len()
                                );
                                match probe_registry.invoke_buffer(&src_pad, &out) {
                                    ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                    _ => {}
                                }
                                send_output(out, &outputs, &output_bridges, &tracers).await;
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
                                // A failed transform stops processing — propagate
                                // EOS downstream so sinks terminate cleanly.
                                send_eos(&outputs, &output_bridges).await;
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
                                    send_output(buffer, &outputs, &output_bridges, &tracers).await;
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
            // Bridge path: read from RT→Async bridge.
            //
            // Observability here mirrors the channel path above, including the
            // notify_buffer_processed-before-send_output ordering. It used to
            // be absent entirely, so putting an element on an RT thread — the
            // elements most worth measuring — made it invisible to probes and
            // to LatencyTracer.
            loop {
                while let Some(buffer) = bridge.try_pop() {
                    count += 1;

                    match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }

                    tracers.notify_buffer(&name, &buffer);
                    let result = element.process(Some(buffer)).await;
                    tracers.notify_buffer_processed(&name);

                    match result {
                        Ok(Some(out)) => {
                            match probe_registry.invoke_buffer(&src_pad, &out) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            send_output(out, &outputs, &output_bridges, &tracers).await;
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
                                send_output(buffer, &outputs, &output_bridges, &tracers).await;
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

#[allow(clippy::too_many_arguments)]
fn spawn_demuxer_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    inputs: Vec<AsyncReceiver<Message>>,
    outputs_by_pad: HashMap<String, Vec<OutputBranch>>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
) -> JoinHandle<Result<()>> {
    tokio::spawn(async move {
        tracing::debug!("demuxer '{}' started", name);
        events.send_node_started(&name);

        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;

        if let Some(rx) = inputs.into_iter().next() {
            loop {
                match rx.recv().await {
                    Ok(Message::Buffer(buffer)) => {
                        count += 1;

                        match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                            ProbeReturn::Drop | ProbeReturn::Handled => continue,
                            _ => {}
                        }

                        // Same ordering rule as spawn_transform_task: the
                        // processed notification must land before the
                        // broadcast, or downstream back-pressure is billed as
                        // demux time.
                        tracers.notify_buffer(&name, &buffer);
                        let result = element.process(Some(buffer)).await;
                        tracers.notify_buffer_processed(&name);

                        match result {
                            Ok(Some(out)) => {
                                match probe_registry.invoke_buffer(&src_pad, &out) {
                                    ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                    _ => {}
                                }
                                for branches in outputs_by_pad.values() {
                                    broadcast(branches, out.clone(), &tracers).await;
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
                                    for branches in outputs_by_pad.values() {
                                        broadcast(branches, buffer.clone(), &tracers).await;
                                    }
                                }
                            }
                            Err(e) => {
                                tracing::warn!("flush error in '{}': {}", name, e);
                            }
                        }
                        for branches in outputs_by_pad.values() {
                            broadcast_eos(branches).await;
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

#[allow(clippy::too_many_arguments)]
fn spawn_muxer_task(
    name: String,
    node_id: NodeId,
    mut element: Box<DynAsyncElement<'static>>,
    inputs_by_pad: HashMap<String, Vec<AsyncReceiver<Message>>>,
    outputs: Vec<OutputBranch>,
    events: EventSender,
    probe_registry: ProbeRegistry,
    tracers: TracerRegistry,
) -> JoinHandle<Result<()>> {
    use futures::stream::{FuturesUnordered, StreamExt};

    tokio::spawn(async move {
        tracing::debug!("muxer '{}' started", name);
        events.send_node_started(&name);

        // A muxer has several sink pads, but probes are registered per node
        // rather than per named pad here, so all inputs share one PadRef —
        // consistent with how a transform's single sink pad is handled.
        let sink_pad = crate::pipeline::probe::PadRef::sink(node_id);
        let src_pad = crate::pipeline::probe::PadRef::src(node_id);
        let mut count: u64 = 0;

        // One pending receive per input, re-armed after each message.
        //
        // `FuturesUnordered` drops a future once it resolves, so the previous
        // version — which collected one `rx.recv()` future per input and never
        // pushed another — delivered exactly *one* buffer per input pad and
        // then fell out of the loop. A two-input muxer muxed two buffers,
        // whatever the stream length.
        async fn recv_one(
            pad: String,
            rx: AsyncReceiver<Message>,
        ) -> (String, AsyncReceiver<Message>, Option<Message>) {
            let msg = rx.recv().await.ok();
            (pad, rx, msg)
        }

        let mut receivers: FuturesUnordered<_> = inputs_by_pad
            .into_iter()
            .flat_map(|(pad, rxs)| {
                rxs.into_iter()
                    .map(move |rx| recv_one(pad.clone(), rx))
                    .collect::<Vec<_>>()
            })
            .collect();

        let total = receivers.len();
        let mut eos_count = 0;

        while let Some((pad, rx, msg)) = receivers.next().await {
            // Keep listening on this input unless it just ended; the EOS and
            // error arms deliberately let it drop.
            if matches!(msg, Some(Message::Buffer(_))) {
                receivers.push(recv_one(pad, rx));
            }
            let msg = match msg {
                Some(m) => Ok(m),
                None => Err(()),
            };
            match msg {
                Ok(Message::Buffer(buffer)) => {
                    count += 1;

                    match probe_registry.invoke_buffer(&sink_pad, &buffer) {
                        ProbeReturn::Drop | ProbeReturn::Handled => continue,
                        _ => {}
                    }

                    // Muxers buffer and interleave, so most inputs produce no
                    // output — LatencyTracer sees the pair regardless, which is
                    // what makes "how long is this muxer taking" answerable at
                    // all. It previously had no instrumentation whatsoever.
                    tracers.notify_buffer(&name, &buffer);
                    let result = element.process(Some(buffer)).await;
                    tracers.notify_buffer_processed(&name);

                    match result {
                        Ok(Some(out)) => {
                            match probe_registry.invoke_buffer(&src_pad, &out) {
                                ProbeReturn::Drop | ProbeReturn::Handled => continue,
                                _ => {}
                            }
                            broadcast(&outputs, out, &tracers).await;
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
                            broadcast(&outputs, out, &tracers).await;
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
                                    broadcast(&outputs, buffer, &tracers).await;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("flush error in '{}': {}", name, e);
                            }
                        }
                        broadcast_eos(&outputs).await;
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

    // ---- output_slot_budget ----------------------------------------------

    /// A source, and `branches` sinks hanging off its single src pad with the
    /// given per-link capacities.
    fn fan_out(capacities: &[Option<usize>]) -> (Pipeline, NodeId) {
        let mut pipeline = Pipeline::new();
        let src = pipeline.add_source("src", CountingSource { count: 0, max: 0 });
        for (i, capacity) in capacities.iter().enumerate() {
            let sink = pipeline.add_sink(
                format!("sink{i}"),
                CountingSink {
                    received: Arc::new(AtomicU64::new(0)),
                },
            );
            pipeline
                .link_pads_full(src, "src", sink, "sink", LinkPolicy::Block, *capacity)
                .unwrap();
        }
        (pipeline, src)
    }

    #[test]
    fn a_sink_gets_no_downstream_capacity() {
        let (pipeline, src) = fan_out(&[None]);
        let executor = Executor::new();
        let sink = pipeline.node_ids().into_iter().find(|&n| n != src).unwrap();

        let budget = executor.output_slot_budget(&pipeline, sink, 0);
        assert_eq!(budget.downstream_capacity, 0);
        assert_eq!(budget.in_flight_margin, defaults::IN_FLIGHT_MARGIN);
    }

    #[test]
    fn one_link_takes_the_configured_channel_capacity() {
        let (pipeline, src) = fan_out(&[None]);
        let executor = Executor::with_config(ExecutorConfig::default().with_channel_capacity(64));

        assert_eq!(
            executor.output_slot_budget(&pipeline, src, 0).slots(),
            64 + defaults::IN_FLIGHT_MARGIN
        );
    }

    #[test]
    fn a_per_link_capacity_override_is_honoured() {
        let (pipeline, src) = fan_out(&[Some(128)]);
        let executor = Executor::new();

        assert_eq!(
            executor.output_slot_budget(&pipeline, src, 0).slots(),
            128 + defaults::IN_FLIGHT_MARGIN
        );
    }

    #[test]
    fn fan_out_on_one_pad_takes_the_max_not_the_sum() {
        // Three branches off the same src pad share each buffer — a clone is a
        // refcount bump on the same slot — so the deepest link bounds the pad.
        // Summing would ask for 8 + 64 + 16 = 88 slots for 64 slots of work.
        let (pipeline, src) = fan_out(&[Some(8), Some(64), Some(16)]);
        let executor = Executor::new();

        let budget = executor.output_slot_budget(&pipeline, src, 0);
        assert_eq!(budget.downstream_capacity, 64);
        assert_eq!(budget.slots(), 64 + defaults::IN_FLIGHT_MARGIN);
    }

    #[test]
    fn a_bridged_edge_adds_the_bridge_capacity() {
        let (pipeline, src) = fan_out(&[Some(8)]);
        let executor = Executor::new();
        let bridge_capacity = executor.config.rt.bridge_capacity;

        let budget = executor.output_slot_budget(&pipeline, src, 1);
        assert_eq!(budget.downstream_capacity, 8 + bridge_capacity);
    }

    #[test]
    fn a_deep_link_outgrows_the_codec_floor() {
        // The point of the whole exercise: the arena must track the channel, so
        // a consumer holding a full channel cannot starve the producer.
        let (pipeline, src) = fan_out(&[None]);
        let executor = Executor::with_config(ExecutorConfig::default().with_channel_capacity(256));

        let slots = executor
            .output_slot_budget(&pipeline, src, 0)
            .resolve(defaults::MIN_OUTPUT_SLOT_COUNT, 4096);
        assert!(
            slots > 256,
            "{slots} slots cannot outlive a 256-deep channel"
        );
    }
}
